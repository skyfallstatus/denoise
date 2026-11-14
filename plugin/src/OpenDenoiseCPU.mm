// SPDX-License-Identifier: MIT
// Open Denoise — After Effects API entry point (EffectMain).
//
// Premiere uses this for parameter set-up always, and for rendering when
// Mercury is in software mode (or when the GPU filter refuses an instance).
// After Effects uses it for everything. Rendering converts the host's pixels
// to float BGRA in an MTLBuffer on the system default Metal device, runs the
// same core as the GPU path, and converts back. No Metal device: pass-through.
//
// Re-entrancy: to hand us the input at another time the host may have to
// render whatever is upstream, which can be another Open Denoise instance
// (applied twice, on a nest and inside it, on an AE precomp and inside it),
// on this thread or another one while this one waits. So no plug-in lock is
// held across any host callback: Render does its checkouts and suite calls
// first, then takes the render lock for the conversion and the core, and
// checks the neighbours in after releasing it.
#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>
#include <dlfcn.h>
#import <Metal/Metal.h>

#include "AEConfig.h"
#include "PrSDKTypes.h"
#include "AE_Effect.h"
#include "AE_EffectSuites.h"
#include "A.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "PrSDKAESupport.h"
#include "SPBasic.h"

#include <string.h>
#include <algorithm>
#include <cmath>
#include <atomic>
#include <exception>
#include <list>
#include <memory>
#include <new>
#include <vector>

#include "OpenDenoisePlugin.h"
#include "OpenDenoiseCapture.h"
#include "OpenDenoiseReservedParam.h"

using namespace odplug;

static_assert((A_u_long)PF_VERSION(OD_MAJOR_VERSION, OD_MINOR_VERSION, OD_BUG_VERSION,
                                   PF_Stage_DEVELOP, 0) == OD_PIPL_VERSION_VALUE,
              "OD_PIPL_VERSION_VALUE must equal PF_VERSION(...)");
static_assert((PF_OutFlag_WIDE_TIME_INPUT | PF_OutFlag_DEEP_COLOR_AWARE | PF_OutFlag_SEND_UPDATE_PARAMS_UI) == OD_OUT_FLAGS_VALUE,
              "OD_OUT_FLAGS_VALUE out of sync");
static_assert((PF_OutFlag2_PRESERVES_FULLY_OPAQUE_PIXELS | PF_OutFlag2_SUPPORTS_THREADED_RENDERING |
               PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG) == OD_OUT_FLAGS2_VALUE,
              "OD_OUT_FLAGS2_VALUE out of sync");

namespace {

enum class Fmt { Unknown, BGRA8, BGRA16, BGRA32F, ARGB8, ARGB16 };

int BytesPerPixel(Fmt f) {
  switch (f) {
    case Fmt::BGRA8: case Fmt::ARGB8: return 4;
    case Fmt::BGRA16: case Fmt::ARGB16: return 8;
    case Fmt::BGRA32F: return 16;
    default: return 0;
  }
}

bool IsPremiere(const PF_InData* in_data) { return in_data->appl_id == kAppID_Premiere; }

// Acquire/release without the SDK's throwing helpers: a missing suite is
// normal (After Effects has no Premiere suites) and must not be an error.
template <typename T>
struct Suite {
  SPBasicSuite* basic;
  const char* name;
  int32_t version;
  const T* p = nullptr;
  Suite(const PF_InData* in_data, const char* n, int32_t v) : basic(in_data->pica_basicP), name(n), version(v) {
    const void* s = nullptr;
    if (basic && basic->AcquireSuite(n, v, &s) == kSPNoError) p = (const T*)s;
  }
  ~Suite() { if (p) basic->ReleaseSuite(name, version); }
  explicit operator bool() const { return p != nullptr; }
  const T* operator->() const { return p; }
};

Fmt WorldFormat(const PF_InData* in_data, PF_EffectWorld* world) {
  if (IsPremiere(in_data)) {
    Suite<PF_PixelFormatSuite1> pf(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1);
    PrPixelFormat fmt = PrPixelFormat_BGRA_4444_8u;  // Premiere's default when the suite is absent
    if (pf && pf->GetPixelFormat(world, &fmt) != PF_Err_NONE) return Fmt::Unknown;
    switch (fmt) {
      case PrPixelFormat_BGRA_4444_8u: return Fmt::BGRA8;
      case PrPixelFormat_BGRA_4444_16u: return Fmt::BGRA16;
      case PrPixelFormat_BGRA_4444_32f: return Fmt::BGRA32F;
      default: return Fmt::Unknown;
    }
  }
  return PF_WORLD_IS_DEEP(world) ? Fmt::ARGB16 : Fmt::ARGB8;
}

// ---- pixel conversion. Float BGRA, tightly packed (rowBytes = width*16). ----

const float kInv255 = 1.f / 255.f;
const float kInv32768 = 1.f / 32768.f;  // 16-bit "white" is 32768 in both hosts

// Row y of a world, `width` pixels, to float BGRA at d.
void ToFloatRow(const PF_EffectWorld* w, Fmt fmt, int y, int width, float* d) {
  const char* row = (const char*)w->data + (ptrdiff_t)y * w->rowbytes;
  switch (fmt) {
    case Fmt::BGRA32F:
      memcpy(d, row, (size_t)width * 16);
      break;
    case Fmt::BGRA8: {
      const uint8_t* s = (const uint8_t*)row;
      for (int i = 0; i < width * 4; ++i) d[i] = s[i] * kInv255;
      break;
    }
    case Fmt::BGRA16: {
      const uint16_t* s = (const uint16_t*)row;
      for (int i = 0; i < width * 4; ++i) d[i] = s[i] * kInv32768;
      break;
    }
    case Fmt::ARGB8: {
      const uint8_t* s = (const uint8_t*)row;
      for (int x = 0; x < width; ++x, s += 4, d += 4) {
        d[0] = s[3] * kInv255; d[1] = s[2] * kInv255; d[2] = s[1] * kInv255; d[3] = s[0] * kInv255;
      }
      break;
    }
    case Fmt::ARGB16: {
      const uint16_t* s = (const uint16_t*)row;
      for (int x = 0; x < width; ++x, s += 4, d += 4) {
        d[0] = s[3] * kInv32768; d[1] = s[2] * kInv32768; d[2] = s[1] * kInv32768; d[3] = s[0] * kInv32768;
      }
      break;
    }
    default: break;
  }
}

void ToFloat(const PF_EffectWorld* w, Fmt fmt, float* dst) {
  for (int y = 0; y < w->height; ++y) ToFloatRow(w, fmt, y, w->width, dst + (size_t)y * w->width * 4);
}

inline uint8_t To8(float v) {
  v = v * 255.f + 0.5f;
  return (uint8_t)(v < 0.f ? 0.f : (v > 255.f ? 255.f : v));  // NaN -> 0
}
inline uint16_t To16(float v) {
  v = v * 32768.f + 0.5f;
  return (uint16_t)(v < 0.f ? 0.f : (v > 32768.f ? 32768.f : v));
}

// Float BGRA at s to row y of a world, `width` pixels.
void FromFloatRow(const float* s, Fmt fmt, PF_EffectWorld* w, int y, int width) {
  char* row = (char*)w->data + (ptrdiff_t)y * w->rowbytes;
  switch (fmt) {
    case Fmt::BGRA32F:
      memcpy(row, s, (size_t)width * 16);
      break;
    case Fmt::BGRA8: {
      uint8_t* d = (uint8_t*)row;
      for (int i = 0; i < width * 4; ++i) d[i] = To8(s[i]);
      break;
    }
    case Fmt::BGRA16: {
      uint16_t* d = (uint16_t*)row;
      for (int i = 0; i < width * 4; ++i) d[i] = To16(s[i]);
      break;
    }
    case Fmt::ARGB8: {
      uint8_t* d = (uint8_t*)row;
      for (int x = 0; x < width; ++x, s += 4, d += 4) {
        d[3] = To8(s[0]); d[2] = To8(s[1]); d[1] = To8(s[2]); d[0] = To8(s[3]);
      }
      break;
    }
    case Fmt::ARGB16: {
      uint16_t* d = (uint16_t*)row;
      for (int x = 0; x < width; ++x, s += 4, d += 4) {
        d[3] = To16(s[0]); d[2] = To16(s[1]); d[1] = To16(s[2]); d[0] = To16(s[3]);
      }
      break;
    }
    default: break;
  }
}

void FromFloat(const float* src, Fmt fmt, PF_EffectWorld* w) {
  for (int y = 0; y < w->height; ++y) FromFloatRow(src + (size_t)y * w->width * 4, fmt, w, y, w->width);
}

// Premiere's pixel format of a world as it is, known to this file or not.
bool RawPixelFormat(const PF_InData* in_data, PF_EffectWorld* world, PrPixelFormat* out) {
  if (!IsPremiere(in_data)) return false;
  Suite<PF_PixelFormatSuite1> pf(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1);
  *out = PrPixelFormat_BGRA_4444_8u;  // Premiere's default when the suite is absent
  return !pf || pf->GetPixelFormat(world, out) == PF_Err_NONE;
}

// The unprocessed input in the output: the fallback for every failure. Only
// what is a valid frame in the output's format is written: the same format
// as the input (copied), or two formats this file converts (through float);
// where the output is larger than the input it is transparent black. An
// unknown format is copied only when both worlds have the same Premiere
// format and size. In place (one buffer) it already is the input. false:
// nothing valid could be written (the caller returns an error, never a frame
// it did not write or a copy across formats).
bool PassThrough(const PF_InData* in_data, PF_EffectWorld* in, PF_EffectWorld* out) {
  if (!in || !out || !in->data || !out->data) return false;
  if (in->data == out->data) return in->width == out->width && in->height == out->height;
  const Fmt fi = WorldFormat(in_data, in), fo = WorldFormat(in_data, out);
  if (fi == Fmt::Unknown || fo == Fmt::Unknown) {
    PrPixelFormat a = PrPixelFormat_Invalid, b = PrPixelFormat_Invalid;
    if (!RawPixelFormat(in_data, in, &a) || !RawPixelFormat(in_data, out, &b) || a != b) return false;
    if (in->width != out->width || in->height != out->height || in->height <= 0) return false;
    const size_t n = (size_t)std::min(std::abs(in->rowbytes), std::abs(out->rowbytes));
    for (int y = 0; y < out->height; ++y)
      memcpy((char*)out->data + (ptrdiff_t)y * out->rowbytes, (const char*)in->data + (ptrdiff_t)y * in->rowbytes, n);
    return true;
  }
  const int bi = BytesPerPixel(fi), bo = BytesPerPixel(fo);
  if (std::abs(in->rowbytes) < in->width * bi || std::abs(out->rowbytes) < out->width * bo) return false;
  const int w = std::min(in->width, out->width), h = std::min(in->height, out->height);
  std::vector<float> row(fi == fo ? 0 : (size_t)std::max(0, w) * 4);
  for (int y = 0; y < out->height; ++y) {
    char* d = (char*)out->data + (ptrdiff_t)y * out->rowbytes;
    int done = 0;
    if (y < h && w > 0) {
      if (fi == fo) {
        memcpy(d, (const char*)in->data + (ptrdiff_t)y * in->rowbytes, (size_t)w * bo);
      } else {
        ToFloatRow(in, fi, y, w, row.data());
        FromFloatRow(row.data(), fo, out, y, w);
      }
      done = w;
    }
    if (done < out->width) memset(d + (size_t)done * bo, 0, (size_t)(out->width - done) * bo);
  }
  return true;
}

// What a render returns when it could not give the output a valid frame.
const PF_Err kNoValidOutput = PF_Err_INTERNAL_STRUCT_DAMAGED;

// ---- CPU-path state: one Metal device, a few sessions, a buffer pool --------

struct CPUState {
  bool triedDevice = false;
  id<MTLDevice> device = nil;
  ODContext* ctx = nullptr;
  // Sessions keyed by ClipIdentity (Premiere's filter instance ID and the
  // clip it renders), most recent first. Bounded.
  std::list<std::pair<uint64_t, std::unique_ptr<Session>>> sessions;
  // Without a complete identity (After Effects; Premiere without an instance
  // ID or without the clip's timeline, start and duration) every render
  // starts from a reset session: the automatic profile of one layer must not
  // be applied to another, and the result then depends only on the frame and
  // its neighbours, not on render order.
  std::unique_ptr<Session> stateless;
  NSMutableArray<id<MTLBuffer>>* pool = nil;
  size_t poolBufferBytes = 0;
  double lastRenderMs = 0;
  bool idleArmed = false;  // the idle-release timer is set
};
enum { kMaxCPUSessions = 4, kMaxPooledBuffers = 6 };

// The buffer pool is up to six float frames (800 MB at 3840x2160). It goes
// once no software render has come for kIdleMs (OD_CPU_IDLE_MS overrides),
// on a timer of its own: one dispatch timer source, re-armed by the first
// render after it fired, cancelled (and waited for) at global setdown.
double IdleReleaseMs() {
  static const double ms = [] {
    const char* e = getenv("OD_CPU_IDLE_MS");
    const double v = e ? atof(e) : 5000.0;
    return v > 0 ? v : 5000.0;
  }();
  return ms;
}
dispatch_queue_t gIdleQueue = nil;
dispatch_source_t gIdleTimer = nil;  // created under gCPUMutex, cancelled at setdown

double NowMs() { return CFAbsoluteTimeGetCurrent() * 1000.0; }

// Serialises the conversion and the core for every CPU render (bounds
// memory, guards CPUState). Never held across a host callback.
std::mutex gCPUMutex;
CPUState& State() {
  static CPUState* s = new CPUState();
  return *s;
}

void IdleTick();

// Under gCPUMutex, at the end of every render.
void ArmIdleRelease(CPUState& st) {
  st.lastRenderMs = NowMs();
  if (st.idleArmed || st.pool.count == 0) return;
  if (!gIdleTimer) {
    gIdleQueue = dispatch_queue_create("dev.opendenoise.cpu-idle",
                                       dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0));
    gIdleTimer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, gIdleQueue);
    if (!gIdleTimer) return;
    dispatch_source_set_event_handler(gIdleTimer, ^{ IdleTick(); });
    dispatch_resume(gIdleTimer);  // no time set yet: does not fire
  }
  st.idleArmed = true;
  const double ms = IdleReleaseMs();
  dispatch_source_set_timer(gIdleTimer, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(ms * 1e6)), DISPATCH_TIME_FOREVER,
                            (uint64_t)(ms * 1e5));
}

void IdleTick() {
  std::lock_guard<std::mutex> lock(gCPUMutex);
  CPUState& st = State();
  if (!st.idleArmed || !gIdleTimer) return;
  const double idle = NowMs() - st.lastRenderMs, ms = IdleReleaseMs();
  if (idle >= ms) {
    st.pool = nil;
    st.poolBufferBytes = 0;
    st.idleArmed = false;
    return;
  }
  dispatch_source_set_timer(gIdleTimer, dispatch_time(DISPATCH_TIME_NOW, (int64_t)((ms - idle) * 1e6)),
                            DISPATCH_TIME_FOREVER, (uint64_t)(ms * 1e5));
}

Session* SessionFor(CPUState& st, uint64_t identity) {
  for (auto it = st.sessions.begin(); it != st.sessions.end(); ++it) {
    if (it->first == identity) {
      st.sessions.splice(st.sessions.begin(), st.sessions, it);
      return st.sessions.front().second.get();
    }
  }
  std::unique_ptr<Session> s(Session::Create(st.ctx));
  if (!s) return nullptr;
  st.sessions.emplace_front(identity, std::move(s));
  while (st.sessions.size() > kMaxCPUSessions) st.sessions.pop_back();
  return st.sessions.front().second.get();
}

void TearDownCPUState() {
  // The timer's handler takes gCPUMutex: cancel and wait for it first. After
  // the cancel no handler is started, and one already running finishes before
  // the empty block on the same serial queue.
  dispatch_source_t timer = nil;
  dispatch_queue_t queue = nil;
  {
    std::lock_guard<std::mutex> lock(gCPUMutex);
    timer = gIdleTimer;
    queue = gIdleQueue;
    gIdleTimer = nil;
    gIdleQueue = nil;
    State().idleArmed = false;
  }
  if (timer) {
    dispatch_source_cancel(timer);
    dispatch_sync(queue, ^{});
  }
  std::lock_guard<std::mutex> lock(gCPUMutex);
  CPUState& st = State();
  st.sessions.clear();
  st.stateless.reset();
  st.pool = nil;
  st.poolBufferBytes = 0;
  st.ctx = nullptr;  // owned by the context registry
  st.device = nil;
  st.triedDevice = false;
  ReleaseAllContexts();
}

// ---- commands ---------------------------------------------------------------

PF_Err GlobalSetup(PF_InData* in_data, PF_OutData* out_data) {
  PinImage();  // the host may unload us while the core still has work queued (OpenDenoisePlugin.h)
  out_data->my_version = PF_VERSION(OD_MAJOR_VERSION, OD_MINOR_VERSION, OD_BUG_VERSION, PF_Stage_DEVELOP, 0);
  out_data->out_flags = OD_OUT_FLAGS_VALUE;    // see OpenDenoisePlugin.h for the reasons
  out_data->out_flags2 = OD_OUT_FLAGS2_VALUE;

  if (IsPremiere(in_data)) {
    // Tell Premiere which CPU formats we take, best first. Without this it
    // only ever hands us 8-bit BGRA.
    Suite<PF_PixelFormatSuite1> pf(in_data, kPFPixelFormatSuite, kPFPixelFormatSuiteVersion1);
    if (pf) {
      pf->ClearSupportedPixelFormats(in_data->effect_ref);
      pf->AddSupportedPixelFormat(in_data->effect_ref, PrPixelFormat_BGRA_4444_32f);
      pf->AddSupportedPixelFormat(in_data->effect_ref, PrPixelFormat_BGRA_4444_16u);
      pf->AddSupportedPixelFormat(in_data->effect_ref, PrPixelFormat_BGRA_4444_8u);
    }
    // Frames checked out at other times should come in the render's own
    // format instead of always 8-bit. (Members up to this one are laid out
    // identically in every version of the suite; 4 is the first that has it.)
    Suite<PF_UtilitySuite4> util(in_data, kPFUtilitySuite, kPFUtilitySuiteVersion4);
    if (util) util->EffectWantsCheckedOutFramesToMatchRenderPixelFormat(in_data->effect_ref);
  }
  return PF_Err_NONE;
}

// The parameters, in the order of kParamOrder (OpenDenoisePlugin.h): Effect Controls shows
// them in that order, and saved values find their control by disk ID wherever it sits.
// Defaults of everything added since the nine original controls are the core's neutral
// values, or (Look) never reach the core, so a project saved before a control existed, which
// the host fills with its default or its old-project value, renders exactly as before.
PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data) {
  PF_ParamDef def;

  // A float slider in percent, tenths, slider range = valid range.
#define OD_PERCENT(NAME, MINV, MAXV, DFLT, FLAGS)                                              \
  AEFX_CLR_STRUCT(def);                                                                        \
  PF_ADD_FLOAT_SLIDERX(NAME, MINV, MAXV, MINV, MAXV, DFLT, PF_Precision_TENTHS,                \
                       PF_ValueDisplayFlag_PERCENT, FLAGS, id)
  // Every topic starts collapsed. Each follows its
  // own flag because out_flags2 has
  // PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG; without it every topic
  // would start collapsed (AE_Effect.h).
#define OD_TOPIC(NAME) PF_ADD_TOPICX(NAME, PF_ParamFlag_START_COLLAPSED, id)
#define OD_TOPIC_END() \
  AEFX_CLR_STRUCT(def); \
  PF_END_TOPIC(id)
  // Premiere uses the value label (NAME_B) as the checkbox's display name.
  // CHECKBOXX always leaves it empty. AE already shows the parameter name;
  // keep its value label empty to avoid displaying the same text twice.
#define OD_CHECKBOX(NAME, DFLT) \
  AEFX_CLR_STRUCT(def); \
  PF_ADD_CHECKBOX(NAME, IsPremiere(in_data) ? NAME : "", DFLT, PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_SUPERVISE, id)
#define OD_LEVEL(NAME) OD_PERCENT(NAME, OD_BAND_LEVEL_MIN, OD_BAND_LEVEL_MAX, OD_DFLT_BAND_LEVEL, flags)
#define OD_AMOUNT(NAME) OD_PERCENT(NAME, 0, 100, OD_DFLT_BAND_AMOUNT, flags)

  for (const int id : kParamOrder) {
    if (IsUnusedTopic(id)) {
      AEFX_CLR_STRUCT(def);
      def.param_type = (id == 10 || id == 22 || id == 32 || id == 36 || id == 42 || id == 48)
                          ? PF_Param_GROUP_START : PF_Param_GROUP_END;
      def.flags = def.param_type == PF_Param_GROUP_START ? PF_ParamFlag_START_COLLAPSED : 0;
      def.ui_flags = PF_PUI_NO_ECW_UI | PF_PUI_INVISIBLE;
      def.uu.id = id;
      if (const PF_Err e = PF_ADD_PARAM(in_data,-1,&def)) return e;
      continue;
    }
    // A manual change of a look-set control re-labels the Look
    // (UserChangedParam): those are supervised. Flags are not saved.
    const PF_ParamFlags flags = PF_ParamFlag_SUPERVISE;
    if (const DBSlider* spec = DBSliderFor(id)) {
      AEFX_CLR_STRUCT(def);
      // x -> 30*x^1.5: 0..20 dB uses 76.3% of travel (linear: 66.7%).
      // Only travel changes: the host still persists and animates literal dB.
      def.flags = PF_ParamFlag_SUPERVISE;
      def.u.fs_d.useExponent = id == OD_P_DB_LUMA || id == OD_P_DB_CHROMA;
      def.u.fs_d.exponent = def.u.fs_d.useExponent ? 1.5f : 1.f;
      PF_ADD_FLOAT_SLIDER(spec->name, spec->lo, spec->hi, spec->dragLo, spec->dragHi,
                          AEFX_DEFAULT_CURVE_TOLERANCE, spec->dflt,
                          id == OD_P_DB_GRAIN ? PF_Precision_HUNDREDTHS : PF_Precision_TENTHS,
                          0, false, id);
      continue;
    }
    switch (id) {
      case OD_P_CONTROL_SET: case OD_P_PROCESSING: {
        AEFX_CLR_STRUCT(def);
        def.flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_USE_VALUE_FOR_OLD_PROJECTS;
        def.param_type = PF_Param_POPUP;
        const bool controlSet = id == OD_P_CONTROL_SET;
        if (controlSet) def.ui_flags = PF_PUI_NO_ECW_UI;
        PF_STRNNCPY(def.PF_DEF_NAME, controlSet ? "Control set" : "Processing", sizeof def.PF_DEF_NAME);
        def.u.pd.num_choices = 2;
        def.u.pd.value = 1; def.u.pd.dephault = 2;
        def.u.pd.u.PF_DEF_NAMESPTR = controlSet ? "Previous controls|dB controls" :
                                                       "Version 1|Version 2";
        def.uu.id = id;
        if (const PF_Err e = PF_ADD_PARAM(in_data,-1,&def)) return e;
        break;
      }
      case OD_P_DB_LOOK:
        AEFX_CLR_STRUCT(def);
        def.flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY;
        PF_ADD_POPUP("Look", 6, 1, OD_DB_LOOK_ITEMS, id);
        break;
      case OD_P_DB_FRAMES:
        AEFX_CLR_STRUCT(def);
        // New ID: frame count is static; previous IDs retain their flags.
        def.flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY;
        PF_ADD_POPUP("Frames used", 6, 3, OD_FRAMES_ITEMS, id);
        break;
      case OD_P_STATUS_NOISE:
        PF_ADD_BUTTON("Noise: measuring...", "Update", PF_PUI_NO_ECW_UI | PF_PUI_INVISIBLE, PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_SUPERVISE, id);
        break;
      case OD_P_STATUS_HELP:
        PF_ADD_BUTTON("Help", "Open guide", 0, PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_SUPERVISE, id);
        break;
      case OD_P_SWITCH_DB:
        PF_ADD_BUTTON("Switch to dB controls", "Switch", 0, PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_SUPERVISE, id);
        break;
      case OD_P_ANALYSIS_KEY_LO: case OD_P_ANALYSIS_KEY_HI:
        AEFX_CLR_STRUCT(def);
        def.flags = PF_ParamFlag_CANNOT_TIME_VARY;
        def.ui_flags = PF_PUI_NO_ECW_UI;
        PF_ADD_SLIDER(id == OD_P_ANALYSIS_KEY_LO ? "Analysis key low" : "Analysis key high",
                      INT32_MIN,INT32_MAX,INT32_MIN,INT32_MAX,0,id);
        break;
      case OD_P_TOPIC_PREVIOUS: OD_TOPIC("Previous controls (old project)"); break;
      case OD_P_TOPIC_PREVIOUS_END: OD_TOPIC_END(); break;
      case OD_P_LOOK: {
        // Not animatable (a Look is a recipe applied once, not a curve).
        // A project saved before Look existed gets Custom: it carries its own
        // values and must not claim to be Natural. A new or reset effect gets
        // Natural, whose recipe is every default.
        AEFX_CLR_STRUCT(def);
        def.flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_USE_VALUE_FOR_OLD_PROJECTS;
        def.param_type = PF_Param_POPUP;
        PF_STRNNCPY(def.PF_DEF_NAME, "Look", sizeof(def.PF_DEF_NAME));
        def.u.pd.num_choices = OD_NUM_LOOKS;
        def.u.pd.dephault = OD_DFLT_LOOK;
        def.u.pd.value = OD_LOOK_FOR_OLD_PROJECTS;
        def.u.pd.u.PF_DEF_NAMESPTR = OD_LOOK_ITEMS;
        def.uu.id = id;
        if (const PF_Err e = PF_ADD_PARAM(in_data, -1, &def)) return e;
        break;
      }
      case OD_P_LUMA: OD_PERCENT("Luma strength", 0, 100, OD_DFLT_LUMA, flags); break;
      case OD_P_CHROMA: OD_PERCENT("Colour strength", 0, 100, OD_DFLT_CHROMA, flags); break;
      case OD_P_GRAIN_FLOOR: OD_PERCENT("Minimum grain", 0, OD_GRAIN_FLOOR_MAX, OD_DFLT_GRAIN_FLOOR, flags); break;
      case OD_P_GRAIN_MODE: {
        // Where Minimum grain applies (core ODParams.grainMode). A new or
        // reset effect gets Finest grain only (dephault); a project saved
        // before the control existed gets Whole picture (previous) (value,
        // PF_ParamFlag_USE_VALUE_FOR_OLD_PROJECTS): the picture it was saved
        // with, bit for bit. Not animatable; supervised, it is in the look
        // set.
        AEFX_CLR_STRUCT(def);
        def.flags = PF_ParamFlag_SUPERVISE | PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_USE_VALUE_FOR_OLD_PROJECTS;
        def.param_type = PF_Param_POPUP;
        PF_STRNNCPY(def.PF_DEF_NAME, "Keep grain applies to", sizeof(def.PF_DEF_NAME));
        def.u.pd.num_choices = OD_NUM_GRAIN_MODES;
        def.u.pd.dephault = OD_DFLT_GRAIN_MODE;
        def.u.pd.value = OD_GRAIN_MODE_FOR_OLD_PROJECTS;
        def.u.pd.u.PF_DEF_NAMESPTR = OD_GRAIN_MODE_ITEMS;
        def.uu.id = id;
        if (const PF_Err e = PF_ADD_PARAM(in_data, -1, &def)) return e;
        break;
      }
      case OD_P_TOPIC_PROFILE:
        OD_TOPIC("Noise profile");
        break;
      // Saved profile payload, capture controls and identity words.
      case OD_P_PROFILE_POLICY:
        AEFX_CLR_STRUCT(def);
        def.flags = PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_SUPERVISE | PF_ParamFlag_USE_VALUE_FOR_OLD_PROJECTS;
        PF_ADD_POPUP("Profile", 2, OD_PROFILE_AUTOMATIC, "Automatic|Use captured", id);
        break;
      case OD_P_PROFILE_CAPTURE:
        PF_ADD_BUTTON("Capture profile", "Capture / Save", 0, PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_SUPERVISE,
                      id);
        break;
      case OD_P_PROFILE_REVISION:
        AEFX_CLR_STRUCT(def);
        def.flags = PF_ParamFlag_CANNOT_TIME_VARY;
        def.ui_flags = PF_PUI_NO_ECW_UI;
        PF_ADD_SLIDER("Capture revision", 0, INT32_MAX, 0, INT32_MAX, 0, id);
        break;
      case OD_P_CAPTURE_REQUEST_LO: case OD_P_CAPTURE_REQUEST_HI:
        AEFX_CLR_STRUCT(def);
        def.flags = PF_ParamFlag_CANNOT_TIME_VARY;
        def.ui_flags = PF_PUI_NO_ECW_UI;
        PF_ADD_SLIDER(id == OD_P_CAPTURE_REQUEST_LO ? "Capture request low" : "Capture request high",
                      INT32_MIN, INT32_MAX, INT32_MIN, INT32_MAX, 0, id);
        break;
      case OD_P_PROFILE_PAYLOAD:
        AEFX_CLR_STRUCT(def);
        PF_ADD_ARBITRARY2("Captured profile", 0, 0, PF_ParamFlag_CANNOT_TIME_VARY, PF_PUI_NO_ECW_UI, nullptr, id, nullptr);
        break;
      case OD_P_PROFILE_DIGEST0: case OD_P_PROFILE_DIGEST0 + 1: case OD_P_PROFILE_DIGEST0 + 2:
      case OD_P_PROFILE_DIGEST0 + 3: case OD_P_PROFILE_DIGEST0 + 4: case OD_P_PROFILE_DIGEST0 + 5:
      case OD_P_PROFILE_DIGEST0 + 6: case OD_P_PROFILE_DIGEST_LAST: {
        AEFX_CLR_STRUCT(def);
        def.flags = PF_ParamFlag_CANNOT_TIME_VARY;
        def.ui_flags = PF_PUI_NO_ECW_UI;
        char label[32];
        snprintf(label, sizeof label, "Capture identity %d", id - OD_P_PROFILE_DIGEST0 + 1);
        PF_ADD_SLIDER(label, INT32_MIN, INT32_MAX, INT32_MIN, INT32_MAX, 0, id);
        break;
      }
      case OD_P_PROFILE_INFO:
        PF_ADD_BUTTON("Profile info", "Refresh", 0, PF_ParamFlag_CANNOT_TIME_VARY | PF_ParamFlag_SUPERVISE,
                      id);
        break;
      case OD_P_PROFILE_PANEL:
        AEFX_CLR_STRUCT(def);
        // Retain the type, position and one-byte payload for saved projects.
        PF_ADD_ARBITRARY2("Profile graph", 0, 0, PF_ParamFlag_CANNOT_TIME_VARY,
                          PF_PUI_NO_ECW_UI | PF_PUI_INVISIBLE, nullptr, id, nullptr);
        break;
      case OD_P_HOLD_PROFILE: OD_CHECKBOX("Freeze (this session)", OD_DFLT_HOLD_PROFILE); break;
      case OD_P_NOISE_LEVEL:
        OD_PERCENT("Noise level", OD_NOISE_LEVEL_MIN, OD_NOISE_LEVEL_MAX, OD_DFLT_NOISE_LEVEL, flags);
        break;
      case OD_P_QUALITY:
        AEFX_CLR_STRUCT(def);
        def.flags = PF_ParamFlag_SUPERVISE;
        PF_ADD_POPUP("Quality", 3, OD_DFLT_QUALITY, OD_QUALITY_ITEMS, id);
        break;
      case OD_P_VIEW:
        AEFX_CLR_STRUCT(def);
        def.flags = PF_ParamFlag_SUPERVISE;
        PF_ADD_POPUP("View", OD_NUM_VIEWS, OD_DFLT_VIEW, OD_VIEW_ITEMS, id);
        break;
      case OD_P_TOPIC_TEMPORAL: OD_TOPIC("Temporal"); break;
      case OD_P_TEMPORAL_RADIUS:
        AEFX_CLR_STRUCT(def);
        def.flags = PF_ParamFlag_SUPERVISE;
        PF_ADD_SLIDER("Frames each side", 0, OD_MAX_RADIUS, 0, OD_MAX_RADIUS, OD_DFLT_TEMPORAL_RADIUS, id);
        break;
      case OD_P_TEMPORAL_STRENGTH: OD_PERCENT("Temporal strength", 0, 100, OD_DFLT_TEMPORAL_STRENGTH, flags); break;
      case OD_P_MOTION_THRESHOLD:
        OD_PERCENT("Motion threshold", OD_MOTION_THRESHOLD_MIN, OD_MOTION_THRESHOLD_MAX, OD_DFLT_MOTION_THRESHOLD,
                   flags);
        break;
      case OD_P_EDGE_GATE: OD_PERCENT("Edge ghost protection", 0, 100, OD_DFLT_EDGE_GATE, flags); break;
      case OD_P_TOPIC_LUMA: OD_TOPIC("Luma by size"); break;
      case OD_P_LUMA_AMOUNT_FINE: OD_AMOUNT("Luma fine strength"); break;
      case OD_P_LUMA_AMOUNT_MEDIUM: OD_AMOUNT("Luma medium strength"); break;
      case OD_P_LUMA_AMOUNT_COARSE: OD_AMOUNT("Luma coarse strength"); break;
      case OD_P_LUMA_AMOUNT_VCOARSE: OD_AMOUNT("Luma very coarse strength"); break;
      case OD_P_REMOVE_SPECKS: OD_CHECKBOX("Remove specks", OD_DFLT_REMOVE_SPECKS); break;
      case OD_P_TOPIC_CHROMA: OD_TOPIC("Colour by size"); break;
      case OD_P_CHROMA_AMOUNT_FINE: OD_AMOUNT("Colour fine strength"); break;
      case OD_P_CHROMA_AMOUNT_MEDIUM: OD_AMOUNT("Colour medium strength"); break;
      case OD_P_CHROMA_AMOUNT_COARSE: OD_AMOUNT("Colour coarse strength"); break;
      case OD_P_CHROMA_AMOUNT_VCOARSE: OD_AMOUNT("Colour very coarse strength"); break;
      case OD_P_TOPIC_SHARPEN: OD_TOPIC("Sharpen"); break;
      case OD_P_SHARPEN: OD_PERCENT("Sharpen", 0, 100, OD_DFLT_SHARPEN, flags); break;
      case OD_P_SHARPEN_FINE: OD_PERCENT("Sharpen fine", 0, OD_SHARPEN_BAND_MAX, OD_DFLT_SHARPEN_BAND, flags); break;
      case OD_P_SHARPEN_MEDIUM:
        OD_PERCENT("Sharpen medium", 0, OD_SHARPEN_BAND_MAX, OD_DFLT_SHARPEN_BAND, flags);
        break;
      case OD_P_SHARPEN_NOISE_FLOOR:
        OD_PERCENT("Sharpen noise floor", OD_SHARPEN_NOISE_FLOOR_MIN, OD_SHARPEN_NOISE_FLOOR_MAX,
                   OD_DFLT_SHARPEN_NOISE_FLOOR, flags);
        break;
      case OD_P_SHARPEN_HALO:
        OD_PERCENT("Sharpen halo limit", 0, OD_SHARPEN_HALO_MAX, OD_DFLT_SHARPEN_HALO, flags);
        break;
      case OD_P_TOPIC_EXPERT: OD_TOPIC("Advanced"); break;
      case OD_P_SPATIAL: OD_PERCENT("Spatial strength", 0, 100, OD_DFLT_SPATIAL, flags); break;
      case OD_P_NOISE_Y: OD_LEVEL("Luma noise level"); break;
      case OD_P_NOISE_CB: OD_LEVEL("Cb noise level"); break;
      case OD_P_NOISE_CR: OD_LEVEL("Cr noise level"); break;
      case OD_P_LUMA_LEVEL_FINE: OD_LEVEL("Luma fine noise level"); break;
      case OD_P_LUMA_LEVEL_MEDIUM: OD_LEVEL("Luma medium noise level"); break;
      case OD_P_LUMA_LEVEL_COARSE: OD_LEVEL("Luma coarse noise level"); break;
      case OD_P_LUMA_LEVEL_VCOARSE: OD_LEVEL("Luma very coarse noise level"); break;
      case OD_P_CHROMA_LEVEL_FINE: OD_LEVEL("Colour fine noise level"); break;
      case OD_P_CHROMA_LEVEL_MEDIUM: OD_LEVEL("Colour medium noise level"); break;
      case OD_P_CHROMA_LEVEL_COARSE: OD_LEVEL("Colour coarse noise level"); break;
      case OD_P_CHROMA_LEVEL_VCOARSE: OD_LEVEL("Colour very coarse noise level"); break;
      // Optional core API v9 controls; zero preserves the previous picture.
      case OD_P_TOPIC_EXPERIMENTAL: OD_TOPIC("Experimental"); break;
      case OD_P_GRAIN_STABILITY: OD_PERCENT("Grain stability", 0, 100, OD_DFLT_GRAIN_STABILITY, flags); break;
      case OD_P_COARSE_CHROMA: OD_CHECKBOX("Coarse colour cleanup", OD_DFLT_COARSE_CHROMA); break;
      case OD_P_TOPIC_LUMA_END: case OD_P_TOPIC_CHROMA_END: case OD_P_TOPIC_TEMPORAL_END: case OD_P_TOPIC_SHARPEN_END:
      case OD_P_TOPIC_PROFILE_END: case OD_P_TOPIC_EXPERIMENTAL_END: case OD_P_TOPIC_EXPERT_END:
        OD_TOPIC_END();
        break;
      default:
        // kParamOrder names an ID this switch does not know: a build error in
        // spirit (the harness's table fails first). Never register a guess.
        Log("CPU: parameter setup: no definition for disk ID %d", id);
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
  }
#undef OD_TOPIC
#undef OD_TOPIC_END
#undef OD_CHECKBOX
#undef OD_LEVEL
#undef OD_AMOUNT
#undef OD_PERCENT

  out_data->num_params = kNumParams;
  return PF_Err_NONE;
}

// The render controls, read by disk ID wherever they sit (kParamOrder).
UIParams ReadParams(PF_ParamDef* params[]) {
  UIParams ui;
  ui.luma = params[PfIndex(OD_P_LUMA)]->u.fs_d.value;
  ui.chroma = params[PfIndex(OD_P_CHROMA)]->u.fs_d.value;
  ui.temporalRadius = params[PfIndex(OD_P_TEMPORAL_RADIUS)]->u.sd.value;
  ui.temporalStrength = params[PfIndex(OD_P_TEMPORAL_STRENGTH)]->u.fs_d.value;
  ui.spatial = params[PfIndex(OD_P_SPATIAL)]->u.fs_d.value;
  ui.sharpen = params[PfIndex(OD_P_SHARPEN)]->u.fs_d.value;
  ui.noiseLevel = params[PfIndex(OD_P_NOISE_LEVEL)]->u.fs_d.value;
  ui.quality = params[PfIndex(OD_P_QUALITY)]->u.pd.value;
  ui.view = params[PfIndex(OD_P_VIEW)]->u.pd.value;
  for (int pf = 1; pf < kNumParams; ++pf) {
    const int id = IdAtPf(pf);
    double* v = AdvancedField(ui, id);
    if (!v) continue;  // one of the nine above, a topic start or end, Look, the saved profile
    *v = IsAdvancedCheckbox(id) ? (params[pf]->u.bd.value ? 1.0 : 0.0)
         : IsAdvancedPopup(id)  ? (double)params[pf]->u.pd.value
                                : params[pf]->u.fs_d.value;
  }
  return ui;
}

// PF_Cmd_USER_CHANGED_PARAM. Only the UI is changed here, through
// PF_ChangeFlag_CHANGED_VALUE (undoable); the render never reads Look, so nothing here can
// change how a project renders except the values it writes, which are visible controls. -
// Look chosen: Natural, Cleaner, Phone and Preserve detail write their recipe into the look
// set, marking only the controls whose value differs (no keyframe for a control that already
// holds it). Custom writes nothing. - A look-set control changed by hand: Look becomes the
// first Look whose recipe every look-set value matches, else Custom; written only if it
// changes. After a Look is written its recipe matches, so a host that reported those writes
// as user changes would find nothing to do. Other parameters return at once.
PF_Err UserChangedParam(PF_ParamDef* params[], const PF_UserChangedParamExtra* extra) {
  if (!params || !extra) return PF_Err_NONE;
  const int changed = IdAtPf((int)extra->param_index);
  const bool db = EffectiveControlSet(ReadParams(params)) == OD_CONTROLS_DB;
  if (db) {
    const int lookPf = PfIndex(OD_P_DB_LOOK);
    auto value = [](const PF_ParamDef* d) { return d->param_type == PF_Param_POPUP ? double(d->u.pd.value) : d->u.fs_d.value; };
    if (changed == OD_P_DB_LOOK) {
      const int look = params[lookPf]->u.pd.value;
      if (look < 1 || look >= OD_DB_LOOK_CUSTOM) return PF_Err_NONE;
      for (int id : kDBLookSet) {
        auto* d = params[PfIndex(id)];
        const double v = DBLookRecipeValue(look,id);
        if (value(d) != v) {
          if (d->param_type == PF_Param_POPUP) d->u.pd.value = (A_long)v;
          else d->u.fs_d.value = v;
          d->uu.change_flags |= PF_ChangeFlag_CHANGED_VALUE;
        }
      }
    } else if (IsDBLookSetControl(changed)) {
      double values[kDBLookSetSize];
      for (int i=0;i<kDBLookSetSize;++i) values[i] = value(params[PfIndex(kDBLookSet[i])]);
      const int match = MatchingDBLook(values);
      if (params[lookPf]->u.pd.value != match) {
        params[lookPf]->u.pd.value = match;
        params[lookPf]->uu.change_flags |= PF_ChangeFlag_CHANGED_VALUE;
      }
    }
    return PF_Err_NONE;
  }
  const int lookPf = PfIndex(OD_P_LOOK);
  if (lookPf < 0 || !params[lookPf]) return PF_Err_NONE;
  // The look set holds float sliders and one popup (Minimum grain applies
  // to, its 1-based item).
  auto value = [](const PF_ParamDef* d) {
    return d->param_type == PF_Param_POPUP ? (double)d->u.pd.value : d->u.fs_d.value;
  };
  if (changed == OD_P_LOOK) {
    const int look = (int)params[lookPf]->u.pd.value;
    if (look < OD_LOOK_NATURAL || look > OD_LOOK_PRESERVE) return PF_Err_NONE;  // Custom (or unknown): nothing
    for (const int id : kLookSet) {
      PF_ParamDef* d = params[PfIndex(id)];
      if (!d) continue;
      const double v = LookRecipeValue(look, id);
      if (std::fabs(value(d) - v) > kLookTolerance) {
        if (d->param_type == PF_Param_POPUP) d->u.pd.value = (A_long)std::lround(v);
        else d->u.fs_d.value = v;
        d->uu.change_flags |= PF_ChangeFlag_CHANGED_VALUE;
      }
    }
    return PF_Err_NONE;
  }
  if (!IsLookSetControl(changed)) return PF_Err_NONE;
  double values[kLookSetSize];
  for (int i = 0; i < kLookSetSize; ++i) {
    const PF_ParamDef* d = params[PfIndex(kLookSet[i])];
    values[i] = d ? value(d) : std::nan("");
  }
  const int match = MatchingLook(values);
  if ((int)params[lookPf]->u.pd.value != match) {
    params[lookPf]->u.pd.value = match;
    params[lookPf]->uu.change_flags |= PF_ChangeFlag_CHANGED_VALUE;
  }
  return PF_Err_NONE;
}

// Neighbours checked out from the host. The destructor checks every one in
// again, on every way out of Render (EffectMain catches exceptions above it).
// Declared before the render lock is taken, so the check-ins happen after it
// is released.
class Checkouts {
 public:
  explicit Checkouts(PF_InData* in_data) : in_data_(in_data) {}
  ~Checkouts() {
    for (int i = 0; i < count_; ++i)
      if (in_data_->inter.checkin_param) PF_CHECKIN_PARAM(in_data_, &defs_[i]);
  }
  Checkouts(const Checkouts&) = delete;
  Checkouts& operator=(const Checkouts&) = delete;
  // Checks out the input layer at `time`. Always counted for check-in, even
  // when the checkout fails (the host expects that).
  PF_EffectWorld* Checkout(A_long time, PF_Err* err) {
    if (count_ >= OD_MAX_FRAMES - 1) return nullptr;
    PF_ParamDef* def = &defs_[count_];
    AEFX_CLR_STRUCT(*def);
    ++count_;
    *err = PF_CHECKOUT_PARAM(in_data_, OD_P_INPUT, time, in_data_->time_step, in_data_->time_scale, def);
    return &def->u.ld;
  }

 private:
  PF_InData* in_data_;
  PF_ParamDef defs_[OD_MAX_FRAMES - 1];
  int count_ = 0;
};

uint64_t CaptureToken(PF_ParamDef* params[]) {
  return uint64_t(uint32_t(params[PfIndex(OD_P_CAPTURE_REQUEST_LO)]->u.sd.value)) |
         uint64_t(uint32_t(params[PfIndex(OD_P_CAPTURE_REQUEST_HI)]->u.sd.value)) << 32;
}
void SetCaptureToken(PF_ParamDef* params[], uint64_t token) {
  for (int i = 0; i < 2; ++i) {
    uint32_t word = uint32_t(token >> (i*32)); int32_t signedWord;
    memcpy(&signedWord, &word, sizeof word);
    auto* p = params[PfIndex(OD_P_CAPTURE_REQUEST_LO+i)];
    p->u.sd.value = signedWord;
    p->uu.change_flags |= PF_ChangeFlag_CHANGED_VALUE;
  }
}
std::string FrameLabel(const Capture& c) {
  std::string label = "f";
  for (unsigned i = 0; i < c.count; ++i) {
    if (i) label += ",";
    label += std::to_string(c.times[i]/c.timeStep);
  }
  if (label.size() > 31 || c.times[0] % c.timeStep || c.times[c.count-1] % c.timeStep) {
    char b[64]; snprintf(b, sizeof b, "f%.1f..%.1f (%u)", double(c.times[0])/c.timeStep,
                        double(c.times[c.count-1])/c.timeStep, c.count);
    label = b;
  }
  return label;
}
PF_Err UpdateCaptureUI(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[]) {
  if (!params) return PF_Err_NONE;
  CaptureBytes bytes; Capture c;
  bool valid = !ReadCaptureHandle(in_data, params[PfIndex(OD_P_PROFILE_PAYLOAD)]->u.arb_d.value, bytes) && DecodeCapture(bytes,c);
  if (valid) {
    valid = c.revision == uint32_t(params[PfIndex(OD_P_PROFILE_REVISION)]->u.sd.value);
    const auto digest = c.count ? DigestCapture(bytes) : CaptureDigest{};
    for (int i = 0; i < 8; ++i) valid = valid && digest[i] == uint32_t(params[PfIndex(OD_P_PROFILE_DIGEST0+i)]->u.sd.value);
  }
  auto p = PendingCapture(CaptureToken(params));
  std::string action = "Capture profile", detail;
  if (p.active) { action = p.action; detail = p.detail; }
  else if (!valid) detail = "Invalid profile - use Automatic";
  else if (!c.count) detail = "No capture - Automatic";
  else {
    char b[64]; snprintf(b, sizeof b, "%s %u frames; fit %.0f%%",
       params[PfIndex(OD_P_PROFILE_POLICY)]->u.pd.value == OD_PROFILE_CAPTURED ? "Saved" : "Stored", c.count, 100*c.profile.confidence);
    action = b; detail = FrameLabel(c);
  }
  Suite<PF_ParamUtilsSuite3> ui(in_data, kPFParamUtilsSuite, kPFParamUtilsSuiteVersion3);
  if (ui && ui->PF_UpdateParamUI) {
    for (const auto& entry : {std::make_pair(OD_P_PROFILE_CAPTURE, action), std::make_pair(OD_P_PROFILE_INFO, detail)}) {
      PF_ParamDef copy = *params[PfIndex(entry.first)];
      snprintf(copy.PF_DEF_NAME, sizeof copy.PF_DEF_NAME, "%s", entry.second.c_str());
      const PF_Err err = ui->PF_UpdateParamUI(in_data->effect_ref, PfIndex(entry.first), &copy);
      Log("Capture UI: id=%d name=\"%s\" update_rc=%d", entry.first, copy.PF_DEF_NAME, int(err));
    }
  } else {
    Log("Capture UI: PF_ParamUtilsSuite3 unavailable; status=%s / %s", action.c_str(), detail.c_str());
  }
  const auto status = p.active ? (p.action + ". " + p.detail) : valid ? CaptureStatus(c) : detail;
  snprintf(out_data->return_msg, sizeof out_data->return_msg, "%s", status.c_str());
  return PF_Err_NONE;
}

#include "OpenDenoisePanelUI.h"

PF_Err UserChanged(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_UserChangedParamExtra* e) {
  if (!e || !params) return PF_Err_NONE;
  // Refresh the custom graph after a user action, not on every UPDATE_UI.
  out_data->out_flags |= PF_OutFlag_REFRESH_UI;
  const int changed = IdAtPf((int)e->param_index);
  if (changed != OD_P_PROFILE_CAPTURE) {
    if (changed == OD_P_PROFILE_POLICY && params[PfIndex(OD_P_PROFILE_POLICY)]->u.pd.value == OD_PROFILE_AUTOMATIC && CaptureToken(params)) {
      CancelCapture(CaptureToken(params)); SetCaptureToken(params,0);
    }
    return UpdateCaptureUI(in_data, out_data, params);
  }
  const uint64_t previousToken = CaptureToken(params);
  if (in_data->inter.abort && in_data->inter.abort(in_data->effect_ref)) {
    CancelCapture(previousToken);
    return PF_Interrupt_CANCEL;
  }
  auto pending = PendingCapture(previousToken);
  if (pending.failed && !AcknowledgeCaptureFailure(previousToken))
    return UpdateCaptureUI(in_data, out_data, params); // show reason before offering retry
  CaptureBytes oldBytes{};
  const PF_Err readError = ReadCaptureHandle(in_data, params[PfIndex(OD_P_PROFILE_PAYLOAD)]->u.arb_d.value, oldBytes);
  if (readError) return readError;
  if (pending.ready && CaptureBaseMatches(previousToken, oldBytes)) {
    CaptureBytes bytes;
    const Capture& c = pending.capture;
    if (!EncodeCapture(c, bytes)) return PF_Err_INTERNAL_STRUCT_DAMAGED;
    const auto digest = DigestCapture(bytes);
    const auto status = CaptureStatus(c); // allocate before the transaction
    PF_Handle& h = params[PfIndex(OD_P_PROFILE_PAYLOAD)]->u.arb_d.value;
    PF_Err err = h ? WriteCaptureHandle(in_data, h, bytes) : NewCaptureHandle(in_data, bytes, &h, "ui-save");
    if (err) return err;
    params[PfIndex(OD_P_PROFILE_POLICY)]->u.pd.value = OD_PROFILE_CAPTURED;
    params[PfIndex(OD_P_PROFILE_REVISION)]->u.sd.value = (A_long)c.revision;
    for (int i = 0; i < 8; ++i) {
      int32_t word; memcpy(&word, &digest[i], sizeof word);
      params[PfIndex(OD_P_PROFILE_DIGEST0+i)]->u.sd.value = word;
    }
    for (int id : std::initializer_list<int>{OD_P_PROFILE_POLICY, OD_P_PROFILE_REVISION, OD_P_PROFILE_PAYLOAD,
                   OD_P_PROFILE_DIGEST0, OD_P_PROFILE_DIGEST0+1, OD_P_PROFILE_DIGEST0+2,
                   OD_P_PROFILE_DIGEST0+3, OD_P_PROFILE_DIGEST0+4, OD_P_PROFILE_DIGEST0+5,
                   OD_P_PROFILE_DIGEST0+6, OD_P_PROFILE_DIGEST_LAST})
      params[PfIndex(id)]->uu.change_flags |= PF_ChangeFlag_CHANGED_VALUE;
    SetCaptureToken(params, 0);
    CancelCapture(previousToken);
    Log("Capture: saved: %s", status.c_str());
  } else if (pending.active && !pending.failed && !pending.ready) {
    // A second click before three usable frames is a status refresh, not a
    // silent restart. Profile info also refreshes without touching values.
    Log("Capture: waiting: %s", pending.detail.c_str());
  } else {
    const int32_t oldRevision = params[PfIndex(OD_P_PROFILE_REVISION)]->u.sd.value;
    const uint32_t revision = oldRevision > 0 && oldRevision < INT32_MAX ? oldRevision+1 : 1;
    CancelCapture(previousToken);
    const uint64_t token = ArmCapture(oldBytes, revision,
        in_data->width, in_data->height, in_data->pixel_aspect_ratio.num ?: 1, in_data->pixel_aspect_ratio.den ?: 1);
    if (!token) return PF_Err_OUT_OF_MEMORY;
    SetCaptureToken(params, token);
  }
  out_data->out_flags |= PF_OutFlag_FORCE_RERENDER;
  return UpdateCaptureUI(in_data, out_data, params);
}

PF_Err Render(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output) {
  (void)out_data;
  if (!output || !output->data) return PF_Err_NONE;  // nothing to write
  PF_EffectWorld* input = &params[OD_P_INPUT]->u.ld;
  if (!input->data) {
    static std::atomic<uint64_t> seen{0};
    OD_LOG_COUNTED(seen, "CPU: the host gave no input pixels; returning an error");
    return kNoValidOutput;
  }

  Capture captured;
  bool useCaptured = params[PfIndex(OD_P_PROFILE_POLICY)]->u.pd.value == OD_PROFILE_CAPTURED;
  if (useCaptured) {
    CaptureBytes bytes;
    if (ReadCaptureHandle(in_data, params[PfIndex(OD_P_PROFILE_PAYLOAD)]->u.arb_d.value, bytes) ||
        !DecodeCapture(bytes, captured) ||
        captured.revision != uint32_t(params[PfIndex(OD_P_PROFILE_REVISION)]->u.sd.value)) {
      snprintf(out_data->return_msg, sizeof out_data->return_msg, "Captured profile unavailable or corrupt. Capture again or select Automatic.");
      return kNoValidOutput;
    }
    if (!captured.count) {
      useCaptured = false;
      for (int i = 0; i < 8; ++i)
        if (params[PfIndex(OD_P_PROFILE_DIGEST0+i)]->u.sd.value) return kNoValidOutput;
    }
    auto digest = captured.count ? DigestCapture(bytes) : CaptureDigest{};
    for (int i = 0; i < 8; ++i)
      if (digest[i] != uint32_t(params[PfIndex(OD_P_PROFILE_DIGEST0 + i)]->u.sd.value)) return kNoValidOutput;
    if (useCaptured && (captured.width != uint32_t(output->width) || captured.height != uint32_t(output->height) ||
        in_data->field != PF_Field_FRAME)) {
      snprintf(out_data->return_msg, sizeof out_data->return_msg, "Captured profile requires %ux%u. Use the captured resolution or capture again.", captured.width, captured.height);
      return kNoValidOutput;
    }
  } else if (params[PfIndex(OD_P_PROFILE_POLICY)]->u.pd.value != OD_PROFILE_AUTOMATIC) return kNoValidOutput;

  const Fmt fmt = WorldFormat(in_data, output);
  const int bpp = BytesPerPixel(fmt);
  const int w = output->width, h = output->height;
  // Anything we cannot process is passed through, never left black; and
  // never "passed through" across formats or into a frame not written.
  if (fmt == Fmt::Unknown || WorldFormat(in_data, input) != fmt || input->width != w || input->height != h ||
      w <= 0 || h <= 0 || std::abs(input->rowbytes) < w * bpp || std::abs(output->rowbytes) < w * bpp) {
    if (!useCaptured && PassThrough(in_data, input, output)) return PF_Err_NONE;
    static std::atomic<uint64_t> seen{0};
    OD_LOG_COUNTED(seen, "CPU: input and output cannot be matched (formats or sizes); returning an error");
    return kNoValidOutput;
  }
  // From here on input and output have one format and size: the fallback is
  // a plain copy (no host call, so it may run under the render lock).
  auto passThrough = [&] {
    if (input->data != output->data)
      for (int y = 0; y < h; ++y)
        memcpy((char*)output->data + (ptrdiff_t)y * output->rowbytes,
               (const char*)input->data + (ptrdiff_t)y * input->rowbytes, (size_t)w * bpp);
    return useCaptured ? kNoValidOutput : PF_Err_NONE;
  };

  const UIParams ui = ReadParams(params);
  const ODParams mapped = ToCoreParams(ui,true,false);
  int radius = DependencyRadius(mapped);
  if (!in_data->inter.checkout_param || in_data->time_step <= 0) radius = 0;

  // ---- host calls, with no lock held ----
  // Which session: Premiere's filter instance ID and the clip it renders
  // (ClipIdentity). Pasted copies of an effect share an ID; without the clip
  // the render starts from a reset session rather than another clip's.
  long instance = 0;
  bool haveInstance = false;
  ClipOwner owner;
  if (IsPremiere(in_data)) {
    Suite<PF_UtilitySuite4> util(in_data, kPFUtilitySuite, kPFUtilitySuiteVersion4);
    A_long id = 0;
    if (util && util->GetFilterInstanceID(in_data->effect_ref, &id) == PF_Err_NONE) {
      instance = id;
      haveInstance = true;
    } else {
      static std::atomic<bool> logged{false};
      OD_LOG_ONCE(logged, "CPU: no filter instance ID from Premiere; every render starts from a reset session");
    }
    if (haveInstance) {
      PrTimelineID timeline = 0;
      A_long itemStart = 0, clipStart = 0, clipDuration = 0;
      double speed = 0;
      owner.known = util->GetContainingTimelineID && util->GetTrackItemStart && util->GetClipStart &&
                    util->GetClipDuration && util->GetClipSpeed &&
                    util->GetContainingTimelineID(in_data->effect_ref, &timeline) == PF_Err_NONE &&
                    util->GetTrackItemStart(in_data->effect_ref, &itemStart) == PF_Err_NONE &&
                    util->GetClipStart(in_data->effect_ref, &clipStart) == PF_Err_NONE &&
                    util->GetClipDuration(in_data->effect_ref, &clipDuration) == PF_Err_NONE &&
                    util->GetClipSpeed(in_data->effect_ref, &speed) == PF_Err_NONE;
      owner.timelineID = timeline;
      owner.trackItemStart = itemStart;
      owner.clipStart = clipStart;
      owner.clipDuration = clipDuration;
      owner.clipSpeed = speed;
      if (!owner.known) {
        static std::atomic<bool> logged{false};
        OD_LOG_ONCE(logged, "CPU: no clip information from Premiere (timeline, track item start, clip start, "
                    "duration or speed); every render starts from a reset session");
      }
    }
  }
  const uint64_t identity = ClipIdentity(haveInstance, instance, owner);

  // Temporal neighbours: the input layer at other times, where the host has
  // one. Whatever fails to check out, or comes back in another size or
  // format, is skipped; with none left the render is spatial only.
  const uint64_t captureToken = CaptureToken(params);
  const bool capturePending = CapturePending(captureToken);
  if (capturePending) {
    PrPixelFormat raw = PrPixelFormat_Invalid;
    RawPixelFormat(in_data, input, &raw);
    Log("Capture sample: CPU current time=%d checkout=n/a world=%dx%d rowbytes=%d format=0x%x",
        int(in_data->current_time), w, h, int(input->rowbytes), unsigned(raw));
  }
  Checkouts checkouts(in_data);
  PF_EffectWorld* neighbours[OD_MAX_FRAMES];
  int neighbourOffsets[OD_MAX_FRAMES];
  int neighbourCount = 0;
  for (int off : NeighbourOffsets(radius, in_data->current_time, in_data->time_step, TimeBounds())) {
    PF_Err cerr = PF_Err_NONE;
    PF_EffectWorld* nw = checkouts.Checkout(in_data->current_time + off * in_data->time_step, &cerr);
    if (capturePending) {
      PrPixelFormat raw = PrPixelFormat_Invalid;
      if (nw && nw->data) RawPixelFormat(in_data, nw, &raw);
      Log("Capture sample: CPU checkout time=%lld rc=%d world=%dx%d rowbytes=%d data=%d format=0x%x",
          (long long)in_data->current_time + int64_t(off)*in_data->time_step, int(cerr), nw ? int(nw->width) : 0,
          nw ? int(nw->height) : 0, nw ? int(nw->rowbytes) : 0, nw && nw->data, unsigned(raw));
    }
    if (nw && cerr == PF_Err_NONE && nw->data && nw->width == w && nw->height == h &&
        std::abs(nw->rowbytes) >= w * bpp && WorldFormat(in_data, nw) == fmt) {
      neighbours[neighbourCount] = nw;
      neighbourOffsets[neighbourCount++] = off;
    }
  }

  // ---- conversion and core, under the render lock; no host calls ----
  std::lock_guard<std::mutex> lock(gCPUMutex);
  @autoreleasepool {
    CPUState& st = State();
    if (!st.triedDevice) {
      st.triedDevice = true;
      st.device = MTLCreateSystemDefaultDevice();
      if (st.device) {
        char err[256];
        st.ctx = ContextFor((__bridge void*)st.device, nullptr, err, sizeof err);
        if (!st.ctx) Log("CPU: od_context_create failed: %s", err);
      } else {
        Log("CPU: no Metal device; frames pass through unchanged");
      }
    }
    if (!st.ctx) return passThrough();

    Session* session = nullptr;
    if (identity) {
      session = SessionFor(st, identity);
    } else {
      if (!st.stateless) st.stateless.reset(Session::Create(st.ctx));
      session = st.stateless.get();
      if (session) session->Reset();
    }
    if (!session) return passThrough();

    // Buffers: tightly packed float BGRA, shared storage so we can fill them.
    // Only used under the lock, so at most 2r+2 are live at once.
    const size_t bufBytes = (size_t)w * h * 16;
    if (st.poolBufferBytes != bufBytes) {
      st.pool = [NSMutableArray array];
      st.poolBufferBytes = bufBytes;
    }
    NSUInteger used = 0;
    auto takeBuffer = [&]() -> id<MTLBuffer> {
      if (used < st.pool.count) return st.pool[used++];
      id<MTLBuffer> b = [st.device newBufferWithLength:bufBytes options:MTLResourceStorageModeShared];
      if (b) { [st.pool addObject:b]; ++used; }
      return b;
    };
    auto frameOf = [&](id<MTLBuffer> b) {
      ODFrame f = {};
      f.mtlBuffer = (__bridge void*)b;
      f.width = (uint32_t)w; f.height = (uint32_t)h; f.rowBytes = (uint32_t)w * 16;
      f.format = OD_PIXEL_BGRA32F;
      // No caching by key here: on this path we cannot tell when the content
      // upstream of the effect changed, and a stale frame would be worse than
      // redoing the per-frame work.
      f.frameKey = OD_NO_KEY;
      return f;
    };

    ODFrame frames[OD_MAX_FRAMES];
    int offsets[OD_MAX_FRAMES];
    int count = 0;

    id<MTLBuffer> cur = takeBuffer();
    id<MTLBuffer> outBuf = takeBuffer();
    if (!cur || !outBuf) {
      static std::atomic<uint64_t> seen{0};
      OD_LOG_COUNTED(seen, "CPU: could not allocate %zu-byte frame buffers; passing the frame through", bufBytes);
      return passThrough();
    }
    ToFloat(input, fmt, (float*)cur.contents);
    frames[count] = frameOf(cur);
    offsets[count++] = 0;
    for (int i = 0; i < neighbourCount && count < OD_MAX_FRAMES; ++i) {
      id<MTLBuffer> b = takeBuffer();
      if (!b) break;
      ToFloat(neighbours[i], fmt, (float*)b.contents);
      frames[count] = frameOf(b);
      offsets[count++] = neighbourOffsets[i];
    }

    if (capturePending) {
      int64_t times[OD_MAX_FRAMES];
      for (int i = 0; i < count; ++i) times[i] = int64_t(in_data->current_time) + int64_t(offsets[i])*in_data->time_step;
      ObserveCapture(captureToken, identity ? identity : (uintptr_t)in_data->effect_ref, *session, nullptr,
                     frames, times, count, in_data->time_scale, in_data->time_step,
                     in_data->field == PF_Field_FRAME && int64_t(in_data->downsample_x.num) == in_data->downsample_x.den &&
                     int64_t(in_data->downsample_y.num) == in_data->downsample_y.den, &mapped);
    }

    // The software path is what exports and non-Metal hosts use; it is far
    // from real time whatever the core does, so Quality = Auto means full.
    ODParams hp = ToCoreParams(ui, /*hostWantsFull*/ true, /*wait*/ true);  // we read the result back
    if (useCaptured) { hp.manualProfile = &captured.profile; hp.profileHold = 0; }
    ODFrame outFrame = frameOf(outBuf);

    char err[256];
    int rc = session->Render(frames, offsets, count, &hp, &outFrame, nullptr, err, sizeof err);
    PF_Err result = PF_Err_NONE;
    if (rc == 0) {
      FromFloat((const float*)outBuf.contents, fmt, output);
    } else {
      static std::atomic<uint64_t> seen{0};
      OD_LOG_COUNTED(seen, "CPU: od_render failed (%d): %s; passing the frame through", rc, err);
      result = passThrough();
    }

    // Keep a few buffers for the next frame, not a whole radius-5 window,
    // and none once software renders stop.
    while (st.pool.count > kMaxPooledBuffers) [st.pool removeLastObject];
    ArmIdleRelease(st);
    return result;
  }
}

// After an exception in a command: a render gets the unprocessed input if a
// valid copy can be made, otherwise an error; any other command an error.
PF_Err Recover(PF_Cmd cmd, PF_InData* in_data, PF_ParamDef* params[], PF_LayerDef* output, PF_Err err) {
  if (cmd != PF_Cmd_RENDER || !in_data || !params || !params[OD_P_INPUT]) return err;
  if (params[PfIndex(OD_P_PROFILE_POLICY)] && params[PfIndex(OD_P_PROFILE_POLICY)]->u.pd.value == OD_PROFILE_CAPTURED) {
    CaptureBytes b; Capture c;
    if (ReadCaptureHandle(in_data, params[PfIndex(OD_P_PROFILE_PAYLOAD)]->u.arb_d.value, b) ||
        !DecodeCapture(b,c) || c.count || params[PfIndex(OD_P_PROFILE_REVISION)]->u.sd.value) return err;
    for (int i = 0; i < 8; ++i) if (params[PfIndex(OD_P_PROFILE_DIGEST0+i)]->u.sd.value) return err;
  }
  if (!output || !output->data) return PF_Err_NONE;
  @try {
    try {
      if (PassThrough(in_data, &params[OD_P_INPUT]->u.ld, output)) {
        static std::atomic<uint64_t> seen{0};
        OD_LOG_COUNTED(seen, "CPU: the render was interrupted; the frame was passed through");
        return PF_Err_NONE;
      }
    } catch (...) {
    }
  } @catch (NSException*) {
  }
  static std::atomic<uint64_t> seen{0};
  OD_LOG_COUNTED(seen, "CPU: the render was interrupted and the frame could not be passed through; returning an "
                 "error");
  return err;
}

PF_Err About(PF_InData* in_data, PF_OutData* out_data) {
  (void)in_data;
  snprintf(out_data->return_msg, sizeof out_data->return_msg, "%s %d.%d.%d\rOpen-source video noise reduction (MIT).\rCore %s",
           OD_EFFECT_NAME, OD_MAJOR_VERSION, OD_MINOR_VERSION, OD_BUG_VERSION, od_version());
  return PF_Err_NONE;
}

}  // namespace

extern "C" __attribute__((visibility("default"))) PF_Err EffectMain(
    PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output,
    void* extra) {
  PF_Err err = PF_Err_NONE;
  // Never let an exception reach the host, and never report success for a
  // render whose output was not written (Recover).
  @try {
    try {
      switch (cmd) {
        case PF_Cmd_ABOUT: err = About(in_data, out_data); break;
        case PF_Cmd_GLOBAL_SETUP: err = GlobalSetup(in_data, out_data); break;
        case PF_Cmd_GLOBAL_SETDOWN:
          TearDownCPUState();
          GlobalTeardown("CPU", /*gpuSessions*/ false);  // the GPU entry's shutdown owns those
          break;
        case PF_Cmd_PARAMS_SETUP: err = ParamsSetup(in_data, out_data); break;
        case PF_Cmd_RENDER: err = Render(in_data, out_data, params, output); break;
        // One dispatcher for every supervised parameter: the Look menu and the controls it
        // writes, and the saved profile's policy, Capture and info buttons. Each handler
        // returns at once for a parameter it does not own, and neither writes the other's
        // parameters (the look set holds none of 43, 52..64).
        case PF_Cmd_USER_CHANGED_PARAM:
          err = UserChangedParam(params, (const PF_UserChangedParamExtra*)extra);
          if (!err) err = UserChanged(in_data, out_data, params, (PF_UserChangedParamExtra*)extra);
          if (!err) err = PanelUserChanged(in_data, out_data, params, (PF_UserChangedParamExtra*)extra);
          break;
        case PF_Cmd_UPDATE_PARAMS_UI:
          err = UpdateCaptureUI(in_data, out_data, params);
          if (!err) err = UpdatePanelUI(in_data,out_data,params);
          break;
        case PF_Cmd_ARBITRARY_CALLBACK:
          err = extra && ((PF_ArbParamsExtra*)extra)->id == OD_P_PROFILE_PANEL
                  ? ProfilePanelArbitrary(in_data, (PF_ArbParamsExtra*)extra)
                  : CaptureArbitrary(in_data, (PF_ArbParamsExtra*)extra);
          break;
        default: break;
      }
    } catch (const std::bad_alloc&) {
      Log("CPU: out of memory in command %d", (int)cmd);
      err = Recover(cmd, in_data, params, output, PF_Err_OUT_OF_MEMORY);
    } catch (const std::exception& e) {
      Log("CPU: exception in command %d: %s", (int)cmd, e.what());
      err = Recover(cmd, in_data, params, output, kNoValidOutput);
    } catch (...) {
      Log("CPU: unknown exception in command %d", (int)cmd);
      err = Recover(cmd, in_data, params, output, kNoValidOutput);
    }
  } @catch (NSException* e) {
    Log("CPU: NSException in command %d: %s", (int)cmd, e.reason.UTF8String ?: "?");
    err = Recover(cmd, in_data, params, output, kNoValidOutput);
  }
  return err;
}
