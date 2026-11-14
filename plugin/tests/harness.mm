// SPDX-License-Identifier: MIT
// Open Denoise — plug-in harness. dlopens the built binary and drives both
// entry points with a fake host, without Premiere or After Effects:
//   EffectMain:      global setup, params setup, renders in Premiere 8u / 16u /
//                    32f and After Effects ARGB 8 / 16 with temporal checkouts,
//                    a checkout that renders another Open Denoise instance
//                    (same thread and another thread), concurrent renders, and
//                    two layers rendered alternately.
//   xGPUFilterEntry: start-up, CreateInstance, GetFrameDependencies against
//                    track item bounds (in-point, speed, transition handles,
//                    missing or unexpected bounds), Render (16f and 32f, in
//                    place and into a separate output, short frame list),
//                    sessions shared across instances, dispose, shutdown.
// Modes:
//   stub core:      the output equals the input exactly; OD_STUB_FILL and
//                   OD_STUB_CHECK check the CPU path's pixel conversions in
//                   both directions, exactly; OD_STUB_RECORD shows the
//                   parameters, offsets, keys and sessions the plug-in uses.
//   --expect-passthrough (with OD_STUB_FAIL=1): every core call fails and
//                   every output must still equal the input.
//   --real:         the real core: noise must go down, more with neighbours
//                   than without, colour levels must stay, alpha bit-exact.
//
// Usage: opendenoise-plugin-harness <path to Denoise.plugin/Contents/MacOS/Denoise> [--expect-passthrough|--real]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <spawn.h>
#include <sys/wait.h>

#include "AEConfig.h"
#include "PrSDKTypes.h"
#include "AE_Effect.h"
#include "AE_EffectSuites.h"
#include "AE_EffectUI.h"
#include "A.h"
#include "AE_Macros.h"
#include "PrSDKAESupport.h"
#include "SPBasic.h"
#include "PrSDKGPUDeviceSuite.h"
#include "PrSDKGPUFilter.h"
#include "PrSDKMemoryManagerSuite.h"
#include "PrSDKPPixSuite.h"
#include "PrSDKPPix2Suite.h"
#include "PrSDKVideoSegmentSuite.h"
#include "PrSDKVideoSegmentProperties.h"

#include "OpenDenoisePlugin.h"
#include "RenderStats.h"  // tested directly (TestRenderStats); it logs through odplug::Log, defined here

static std::atomic<int> gFailures{0}, gChecks{0};

// RenderStats.h logs through odplug::Log; here the lines are kept for the
// tests that read them.
static std::mutex gLogMutex;
static std::vector<std::string> gLogLines;
namespace odplug {
void Log(const char* fmt, ...) {
  char buf[2048];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  std::lock_guard<std::mutex> lock(gLogMutex);
  gLogLines.push_back(buf);
}
}  // namespace odplug
#define CHECK(cond, ...)                                                    \
  do {                                                                      \
    ++gChecks;                                                              \
    if (!(cond)) {                                                          \
      ++gFailures;                                                          \
      printf("FAIL %s:%d  %s  -- ", __FILE__, __LINE__, #cond);             \
      printf(__VA_ARGS__);                                                  \
      printf("\n");                                                         \
    }                                                                       \
  } while (0)

static bool gReal = false;         // real core: output need not equal input
static bool gPassthrough = false;  // stub core told to fail every render
static bool StubPixels() { return !gReal && !gPassthrough; }

// The stub core's test patterns (plugin/stub/od_stub.mm), repeated here.
static float StubFillValue(int c, uint32_t y) {
  static const float base[4] = {0.25f, 0.5f, 0.75f, 1.0f};
  return c < 3 ? base[c] - (float)(y % 8) / 256.f : base[3];
}
static float StubCheckValue(int c, uint32_t x, uint32_t y) {
  return (float)((x * 7 + y * 13 + (uint32_t)c * 61) % 239) / 256.f + 0.03f;
}

// ---- the stub core's record (OD_STUB_RECORD) ---------------------------------
static std::string gRecordPath;
static long gRecordOffset = 0;
struct Rec {
  std::string kind;
  std::map<std::string, std::string> kv;
  std::string operator[](const char* k) const {
    auto it = kv.find(k);
    return it == kv.end() ? std::string() : it->second;
  }
  double num(const char* k) const { return atof((*this)[k].c_str()); }
};
// Records written since the last call.
static std::vector<Rec> NewRecords() {
  std::vector<Rec> out;
  FILE* f = fopen(gRecordPath.c_str(), "r");
  if (!f) return out;
  fseek(f, gRecordOffset, SEEK_SET);
  char line[8192];
  while (fgets(line, sizeof line, f)) {
    Rec r;
    char* save = nullptr;
    for (char* tok = strtok_r(line, " \n", &save); tok; tok = strtok_r(nullptr, " \n", &save)) {
      char* eq = strchr(tok, '=');
      if (!eq) r.kind = tok;
      else r.kv[std::string(tok, eq)] = eq + 1;
    }
    out.push_back(r);
  }
  gRecordOffset = ftell(f);
  fclose(f);
  return out;
}
// ---- what the plug-in logs ----------------------------------------------------
// main() sets OD_LOG_STDERR and sends stderr to <record>.log, so the plug-in's
// own log lines (odplug::Log in the bundle, not the one above) can be read
// back: PluginLogSize() before, PluginLogFrom() after.
static std::string gPluginLogPath;
static long PluginLogSize() {
  fflush(stderr);
  struct stat st = {};
  return gPluginLogPath.empty() || stat(gPluginLogPath.c_str(), &st) != 0 ? 0 : (long)st.st_size;
}
static std::string PluginLogFrom(long offset) {
  fflush(stderr);
  std::string out;
  FILE* f = gPluginLogPath.empty() ? nullptr : fopen(gPluginLogPath.c_str(), "r");
  if (!f) return out;
  fseek(f, offset, SEEK_SET);
  char buf[4096];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
  fclose(f);
  return out;
}
static std::vector<std::string> LinesWith(const std::string& text, const char* what) {
  std::vector<std::string> v;
  size_t at = 0;
  while (at < text.size()) {
    size_t end = text.find('\n', at);
    if (end == std::string::npos) end = text.size();
    const std::string line = text.substr(at, end - at);
    if (line.find(what) != std::string::npos) v.push_back(line);
    at = end + 1;
  }
  return v;
}

// Records accumulate here, so that a helper that renders does not hide what
// happened before it from the test that called it.
static std::vector<Rec> gStash;
static void Pull() {
  std::vector<Rec> v = NewRecords();
  gStash.insert(gStash.end(), v.begin(), v.end());
}
static std::vector<Rec> Take() {
  Pull();
  std::vector<Rec> v;
  v.swap(gStash);
  return v;
}
static const Rec* LastOf(const std::vector<Rec>& v, const char* kind) {
  for (auto it = v.rbegin(); it != v.rend(); ++it)
    if (it->kind == kind) return &*it;
  return nullptr;
}
static int CountOf(const std::vector<Rec>& v, const char* kind, const std::string& s = std::string()) {
  int n = 0;
  for (const Rec& r : v)
    if (r.kind == kind && (s.empty() || r["s"] == s)) ++n;
  return n;
}

// Runs fn on another thread; false if it has not returned after `seconds`.
// A plug-in that deadlocks cannot be recovered from: the caller exits.
static bool FinishesWithin(double seconds, std::function<void()> fn) {
  dispatch_semaphore_t done = dispatch_semaphore_create(0);
  auto* job = new std::function<void()>(std::move(fn));
  std::thread([job, done] {
    (*job)();
    delete job;
    dispatch_semaphore_signal(done);
  }).detach();
  return dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(seconds * NSEC_PER_SEC))) == 0;
}
static void DieDeadlocked(const char* what) {
  ++gFailures;
  printf("FAIL %s: did not return within the time limit (deadlock)\n", what);
  printf("FAILED: %d checks, %d failed\n", gChecks.load() + 1, gFailures.load());
  fflush(stdout);
  _exit(3);
}

// ============================ fake AE / Premiere CPU host ====================

typedef PF_Err (*EffectMainFn)(PF_Cmd, PF_InData*, PF_OutData*, PF_ParamDef*[], PF_LayerDef*, void*);

// Where checkouts of one effect instance (by effect_ref) come from: copies of
// `world`, or what `gen` writes for that time.
struct CheckoutSource {
  PF_EffectWorld world = {};
  PrPixelFormat format = PrPixelFormat_Invalid;
  std::function<void(A_long time, PF_EffectWorld& w)> gen;
};
struct CPUHost {
  std::mutex mutex;  // the fake host is called from several threads
  std::vector<PF_ParamDef> added;
  std::vector<PrPixelFormat> formats;
  int clearCalls = 0, wantsMatchCalls = 0, checkouts = 0, checkins = 0;
  std::map<void*, PrPixelFormat> worldFormat;  // by data pointer
  std::deque<std::vector<char>> neighbourStore;
  CheckoutSource source;                         // default
  std::map<PF_ProgPtr, CheckoutSource> sources;  // per effect_ref
  bool failCheckouts = false;
  bool throwInCheckout = false;                  // a checkout throws (fault injection)
  bool throwInGetFormat = false;                 // PF_PixelFormatSuite::GetPixelFormat throws
  std::vector<A_long> checkoutTimes;
  std::function<void(PF_ProgPtr)> onCheckout;    // runs before a checkout returns, no lock held
  // Premiere's filter instance ID by effect_ref, where it is not the ref
  // itself (pasted copies of an effect share one).
  std::map<PF_ProgPtr, A_long> instanceIdOf;
  // The clip of an effect_ref (PF_UtilitySuite: timeline, track item start,
  // clip start, duration, speed). By default every ref has a clip of its own
  // (start = ref * 1000); `ownerOf` overrides that, `noOwnerInfo` makes the
  // calls fail.
  struct Owner { A_long timeline, itemStart, clipStart, duration; double speed; };
  std::map<PF_ProgPtr, Owner> ownerOf;
  bool noOwnerInfo = false;
};
static CPUHost gCPU;
// Parameter-only checkout for Switch's Premiere fallback; no image allocation.
static std::map<PF_ProgPtr,std::vector<PF_ParamDef>> gPanelSampleParams;
static std::function<double(int,A_long,double)> gPanelSampleValue;
static bool gPanelSampleFail = false;
static int gPanelSamples = 0, gPanelSampleCheckins = 0;
static std::vector<A_long> gPanelSampleTimes;

static void SetWorldFormat(void* data, PrPixelFormat f) {
  std::lock_guard<std::mutex> lock(gCPU.mutex);
  gCPU.worldFormat[data] = f;
}
static void ResetCheckoutStats() {
  std::lock_guard<std::mutex> lock(gCPU.mutex);
  gCPU.checkouts = gCPU.checkins = 0;
  gCPU.checkoutTimes.clear();
}

static PF_Err FakeAddParam(PF_ProgPtr, PF_ParamIndex, PF_ParamDefPtr def) {
  gCPU.added.push_back(*def);
  return PF_Err_NONE;
}
static PF_Err FakeCheckout(PF_ProgPtr ref, PF_ParamIndex index, A_long what_time, A_long, A_u_long, PF_ParamDef* param) {
  if (index != 0) {
    ++gPanelSamples; gPanelSampleTimes.push_back(what_time);
    param->param_type=PF_Param_FLOAT_SLIDER;
    const auto it=gPanelSampleParams.find(ref);
    if(gPanelSampleFail || it==gPanelSampleParams.end() || index<0 || index>=(int)it->second.size())
      return PF_Err_BAD_CALLBACK_PARAM;
    *param=it->second[index];
    if(gPanelSampleValue) {
      double v=param->param_type==PF_Param_FLOAT_SLIDER?param->u.fs_d.value:
               param->param_type==PF_Param_SLIDER?param->u.sd.value:param->u.pd.value;
      v=gPanelSampleValue(index,what_time,v);
      if(param->param_type==PF_Param_FLOAT_SLIDER)param->u.fs_d.value=v;
      else if(param->param_type==PF_Param_SLIDER)param->u.sd.value=(A_long)v;
      else param->u.pd.value=(A_long)v;
    }
    return PF_Err_NONE;
  }
  if (gCPU.throwInCheckout) {
    {
      std::lock_guard<std::mutex> lock(gCPU.mutex);
      ++gCPU.checkouts;  // a checkout that fails still gets its checkin
    }
    throw std::runtime_error("injected: checkout");
  }
  if (gCPU.onCheckout) gCPU.onCheckout(ref);  // e.g. render something upstream
  std::lock_guard<std::mutex> lock(gCPU.mutex);
  ++gCPU.checkouts;
  gCPU.checkoutTimes.push_back(what_time);
  if (index != 0 || gCPU.failCheckouts) return PF_Err_BAD_CALLBACK_PARAM;
  auto it = gCPU.sources.find(ref);
  const CheckoutSource& src = it != gCPU.sources.end() ? it->second : gCPU.source;
  const PF_EffectWorld& in = src.world;
  // Its own storage, laid out like the input (bottom-up too).
  size_t bytes = (size_t)std::abs(in.rowbytes) * in.height;
  gCPU.neighbourStore.emplace_back(bytes);
  char* base = gCPU.neighbourStore.back().data();
  param->u.ld = in;
  param->u.ld.data = (PF_PixelPtr)(in.rowbytes >= 0 ? base : base + (ptrdiff_t)(in.height - 1) * -in.rowbytes);
  if (src.gen) {
    src.gen(what_time, param->u.ld);
  } else {
    // A copy of the input world's bytes (so a pass-through core still passes).
    const char* srcBase = in.rowbytes >= 0 ? (const char*)in.data : (const char*)in.data + (ptrdiff_t)(in.height - 1) * in.rowbytes;
    memcpy(base, srcBase, bytes);
  }
  gCPU.worldFormat[param->u.ld.data] = src.format;
  return PF_Err_NONE;
}
static PF_Err FakeCheckin(PF_ProgPtr, PF_ParamDef* param) {
  if(param->param_type!=PF_Param_LAYER) { ++gPanelSampleCheckins; return PF_Err_NONE; }
  std::lock_guard<std::mutex> lock(gCPU.mutex);
  ++gCPU.checkins;
  return PF_Err_NONE;
}

static PF_Err PFAddFormat(PF_ProgPtr, PrPixelFormat f) { gCPU.formats.push_back(f); return PF_Err_NONE; }
static PF_Err PFClearFormats(PF_ProgPtr) { ++gCPU.clearCalls; gCPU.formats.clear(); return PF_Err_NONE; }
static PF_Err PFGetFormat(PF_EffectWorld* w, PrPixelFormat* f) {
  if (gCPU.throwInGetFormat) throw std::runtime_error("injected: GetPixelFormat");
  std::lock_guard<std::mutex> lock(gCPU.mutex);
  auto it = gCPU.worldFormat.find(w->data);
  if (it == gCPU.worldFormat.end()) return PF_Err_BAD_CALLBACK_PARAM;
  *f = it->second;
  return PF_Err_NONE;
}
// The filter instance ID is the effect_ref: 0 -> 42, kNoInstanceRef -> failure.
static const intptr_t kNoInstanceRef = -1;
static PF_Err UtilInstanceID(PF_ProgPtr ref, A_long* out) {
  if ((intptr_t)ref == kNoInstanceRef) return PF_Err_BAD_CALLBACK_PARAM;
  auto it = gCPU.instanceIdOf.find(ref);
  *out = it != gCPU.instanceIdOf.end() ? it->second : ref ? (A_long)(intptr_t)ref : 42;
  return PF_Err_NONE;
}
static bool OwnerOf(PF_ProgPtr ref, CPUHost::Owner* o) {
  if (gCPU.noOwnerInfo) return false;
  auto it = gCPU.ownerOf.find(ref);
  *o = it != gCPU.ownerOf.end() ? it->second : CPUHost::Owner{7, (A_long)(intptr_t)ref * 1000, 0, 300, 1.0};
  return true;
}
static PF_Err UtilTimeline(PF_ProgPtr ref, PrTimelineID* out) {
  CPUHost::Owner o;
  if (!OwnerOf(ref, &o)) return PF_Err_BAD_CALLBACK_PARAM;
  *out = o.timeline;
  return PF_Err_NONE;
}
static PF_Err UtilItemStart(PF_ProgPtr ref, A_long* out) {
  CPUHost::Owner o;
  if (!OwnerOf(ref, &o)) return PF_Err_BAD_CALLBACK_PARAM;
  *out = o.itemStart;
  return PF_Err_NONE;
}
static PF_Err UtilClipStart(PF_ProgPtr ref, A_long* out) {
  CPUHost::Owner o;
  if (!OwnerOf(ref, &o)) return PF_Err_BAD_CALLBACK_PARAM;
  *out = o.clipStart;
  return PF_Err_NONE;
}
static PF_Err UtilClipDuration(PF_ProgPtr ref, A_long* out) {
  CPUHost::Owner o;
  if (!OwnerOf(ref, &o)) return PF_Err_BAD_CALLBACK_PARAM;
  *out = o.duration;
  return PF_Err_NONE;
}
static PF_Err UtilClipSpeed(PF_ProgPtr ref, double* out) {
  CPUHost::Owner o;
  if (!OwnerOf(ref, &o)) return PF_Err_BAD_CALLBACK_PARAM;
  *out = o.speed;
  return PF_Err_NONE;
}
static PF_Err UtilWantsMatch(PF_ProgPtr) { ++gCPU.wantsMatchCalls; return PF_Err_NONE; }

// ============================ fake Premiere GPU host =========================

struct FakePPix {
  id<MTLBuffer> buffer;
  PrPixelFormat format;
  int width, height, rowBytes;
};
static const csSDK_int32 kOwnerNode = 70;
static const csSDK_int32 kMediaNode = 71;  // the owner clip node's input 0
struct GPUHost {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  std::map<int, PrParam> params;  // by segment param index (PF index - 1)
  // Keyframed parameters, by segment index: the value at a clip time is the
  // last key at or before it (the first key before every key). Overrides
  // `params` for that index.
  std::map<int, std::map<PrTime, PrParam>> keyed;
  std::map<std::pair<int, std::string>, std::string> props;  // (node, key) -> value
  bool ownerFails = false;
  int ownersHeld = 0;  // owner and media nodes acquired and not released
  PrTime mediaOffset = 0;
  int allocatedPtrs = 0, disposedPtrs = 0;
  std::atomic<int> getParamCalls{0};
  PrGPUDeviceFramework framework = PrGPUDeviceFramework_Metal;
  // How the fake host numbers the GPU parameters (not documented for topic
  // entries): with topics, every PF index p is at p - 1 and the topic slots
  // hold nothing; without, the advanced entries are compacted. GetParamCount
  // answers paramCount, or fails.
  bool withTopics = true;
  int paramCount = odplug::kUseDynamicVisibility ? 106 : 108;  // kCountTopics (declared with the parameter table below)
  bool countFails = false;
  // Fault injection: GetGPUPPixData fails for these frames; the pixel aspect
  // query or GetParam throws; GetParam fails for one index (-1 = none).
  std::set<PPixHand> dataFails;
  bool throwInPAR = false, throwInGetParam = false, throwInParamCount = false;
  int failParamIndex = -1;
  // Frames CreateGPUPPix made (kept until the end), and Dispose calls.
  std::deque<FakePPix> created;
  int createdCount = 0, disposed = 0;
  NSUInteger createShortBy = 0;  // bytes the next created frames' buffers lack (an unusable output)
};
static GPUHost gGPU;

static prSuiteError DevGetInfo(csSDK_uint32, csSDK_uint32, PrGPUDeviceInfo* out) {
  memset(out, 0, sizeof *out);
  out->outDeviceFramework = gGPU.framework;
  out->outMeetsMinimumRequirementsForAcceleration = kPrTrue;
  out->outDeviceHandle = (__bridge void*)gGPU.device;
  out->outCommandQueueHandle = (__bridge void*)gGPU.queue;
  return suiteError_NoError;
}
static prSuiteError DevGetData(PPixHand h, void** out) {
  if (h && gGPU.dataFails.count(h)) return suiteError_Fail;
  *out = h ? (__bridge void*)((FakePPix*)h)->buffer : nullptr;
  return h ? suiteError_NoError : suiteError_InvalidParms;
}
static prSuiteError DevCreatePPix(csSDK_uint32, PrPixelFormat fmt, int w, int h, int, int, prFieldType, PPixHand* out) {
  FakePPix p;
  const int bpp = fmt == PrPixelFormat_GPU_BGRA_4444_32f ? 16 : 8;
  p.format = fmt; p.width = w; p.height = h; p.rowBytes = w * bpp;
  const NSUInteger len = (NSUInteger)p.rowBytes * h - std::min((NSUInteger)p.rowBytes * h - 16, gGPU.createShortBy);
  p.buffer = [gGPU.device newBufferWithLength:len options:MTLResourceStorageModeShared];
  memset(p.buffer.contents, 0xA5, len);  // never a valid frame of either format
  gGPU.created.push_back(p);
  ++gGPU.createdCount;
  *out = (PPixHand)&gGPU.created.back();
  return suiteError_NoError;
}
static prSuiteError PPixDispose(PPixHand) { ++gGPU.disposed; return suiteError_NoError; }
static prSuiteError PPixGetBounds(PPixHand h, prRect* r) {
  FakePPix* p = (FakePPix*)h;
  r->left = 0; r->top = 0; r->right = p->width; r->bottom = p->height;
  return suiteError_NoError;
}
static prSuiteError PPixGetRowBytes(PPixHand h, csSDK_int32* rb) { *rb = ((FakePPix*)h)->rowBytes; return suiteError_NoError; }
static prSuiteError PPixGetFormat(PPixHand h, PrPixelFormat* f) { *f = ((FakePPix*)h)->format; return suiteError_NoError; }
static prSuiteError PPixGetPAR(PPixHand, csSDK_uint32* n, csSDK_uint32* d) {
  if (gGPU.throwInPAR) throw std::runtime_error("injected: GetPixelAspectRatio");
  *n = 1; *d = 1;
  return suiteError_NoError;
}
static int SegIndex(int id);  // the fake host's GPU index of a disk ID (below)
static prSuiteError SegGetParam(csSDK_int32, csSDK_int32 index, PrTime time, PrParam* out) {
  ++gGPU.getParamCalls;
  // Thrown on Luma strength, the first control a render reads after the saved
  // profile's policy (which decides whether a failure may pass through).
  if (gGPU.throwInGetParam && index == SegIndex(OD_P_LUMA)) throw std::runtime_error("injected: GetParam");
  if (index == gGPU.failParamIndex) return suiteError_Fail;
  auto kf = gGPU.keyed.find(index);
  if (kf != gGPU.keyed.end() && !kf->second.empty()) {
    auto at = kf->second.upper_bound(time);
    *out = at == kf->second.begin() ? at->second : std::prev(at)->second;
    return suiteError_NoError;
  }
  auto it = gGPU.params.find(index);
  if (it == gGPU.params.end()) return suiteError_InvalidParms;
  *out = it->second;
  return suiteError_NoError;
}
static prSuiteError SegGetParamCount(csSDK_int32, csSDK_int32* n) {
  if (gGPU.throwInParamCount) throw std::runtime_error("injected: GetParamCount");
  if (gGPU.countFails) return suiteError_Fail;
  *n = gGPU.paramCount;
  return suiteError_NoError;
}
static prSuiteError SegGetNodeProperty(csSDK_int32 node, const char* key, PrMemoryPtr* out) {
  auto it = gGPU.props.find(std::make_pair((int)node, std::string(key)));
  if (it == gGPU.props.end()) return suiteError_InvalidParms;
  char* s = (char*)malloc(it->second.size() + 1);
  memcpy(s, it->second.c_str(), it->second.size() + 1);
  *out = (PrMemoryPtr)s;
  ++gGPU.allocatedPtrs;
  return suiteError_NoError;
}
static prSuiteError SegAcquireOwner(csSDK_int32 node, csSDK_int32* owner) {
  if (gGPU.ownerFails) return suiteError_Fail;  // reported for adjustment layers
  *owner = kOwnerNode;
  ++gGPU.ownersHeld;
  return suiteError_NoError;
}
static prSuiteError SegAcquireInput(csSDK_int32 node, csSDK_int32 index, PrTime* offset, csSDK_int32* input) {
  if (node != kOwnerNode || index != 0) return suiteError_Fail;
  *offset = gGPU.mediaOffset;
  *input = kMediaNode;
  ++gGPU.ownersHeld;
  return suiteError_NoError;
}
static prSuiteError SegReleaseNode(csSDK_int32 node) {
  if (node == kOwnerNode || node == kMediaNode) --gGPU.ownersHeld;
  return suiteError_NoError;
}
static void MemDisposePtr(PrMemoryPtr p) { ++gGPU.disposedPtrs; free(p); }

static void SetProp(int node, const char* key, long long v) { gGPU.props[std::make_pair(node, std::string(key))] = std::to_string(v); }
static void SetPropText(int node, const char* key, const char* v) { gGPU.props[std::make_pair(node, std::string(key))] = v; }
static void ClearProp(int node, const char* key) { gGPU.props.erase(std::make_pair(node, std::string(key))); }

// ============================ shared fake SPBasicSuite =======================

static std::map<std::pair<PF_ProgPtr,int>,std::string> gCaptureUI;
static std::map<std::pair<PF_ProgPtr,int>,PF_ParamDef> gPanelUI;
static std::map<int,PF_KeyIndex> gPanelKeys;
static bool gPanelKeysFail = false;
static PF_Err FakeKeyframeCount(PF_ProgPtr, PF_ParamIndex index, PF_KeyIndex* count) {
  if (gPanelKeysFail) return PF_Err_BAD_CALLBACK_PARAM;
  auto i=gPanelKeys.find(index);
  *count=i == gPanelKeys.end() ? PF_KeyIndex_NONE : i->second;
  return PF_Err_NONE;
}
static PF_Err FakeUpdateParamUI(PF_ProgPtr ref, PF_ParamIndex index, const PF_ParamDef* p) {
  gCaptureUI[{ref,int(index)}] = p->PF_DEF_NAME;
  gPanelUI[{ref,int(index)}] = *p;
  return PF_Err_NONE;
}
static PF_ParamUtilsSuite3 gParamUtils = [] {
  PF_ParamUtilsSuite3 s{}; s.PF_UpdateParamUI=FakeUpdateParamUI; s.PF_GetKeyframeCount=FakeKeyframeCount; return s;
}();
static PF_PixelFormatSuite1 gPixelFormatSuite = {};
static PF_UtilitySuite4 gUtilitySuite = {};
static PrSDKGPUDeviceSuite gDeviceSuite = {};
static PrSDKPPixSuite gPPixSuite = {};
static PrSDKVideoSegmentSuite gSegmentSuite = {};
static PrSDKMemoryManagerSuite gMemorySuite = {};
static std::atomic<int> gAcquired{0}, gReleased{0};
static bool gIsPremiere = true;
static bool gNoMemorySuite = false;

static int gRegisteredUI = 0;
static std::atomic<int> gAcquireAttempts{0};
static PF_Err RegisterCustomUI(PF_ProgPtr, PF_CustomUIInfo*) { ++gRegisteredUI; return 0; }

static SPErr FakeAcquire(const char* name, int version, const void** suite) {
  ++gAcquireAttempts;
  const void* s = nullptr;
  if (gIsPremiere && !strcmp(name, kPFPixelFormatSuite) && version == kPFPixelFormatSuiteVersion1) s = &gPixelFormatSuite;
  else if (gIsPremiere && !strcmp(name, kPFUtilitySuite) && version == kPFUtilitySuiteVersion4) s = &gUtilitySuite;
  else if (!strcmp(name, kPrSDKGPUDeviceSuite)) s = &gDeviceSuite;
  else if (!strcmp(name, kPrSDKPPixSuite)) s = &gPPixSuite;
  else if (!strcmp(name, kPrSDKVideoSegmentSuite)) s = &gSegmentSuite;
  else if (!strcmp(name, kPrSDKMemoryManagerSuite) && !gNoMemorySuite) s = &gMemorySuite;
  if (!strcmp(name,kPFParamUtilsSuite) && version==kPFParamUtilsSuiteVersion3) s=&gParamUtils;
  if (!s) return kSPSuiteNotFoundError;  // e.g. PPix2: optional
  ++gAcquired;
  *suite = s;
  return kSPNoError;
}
static SPErr FakeRelease(const char*, int) { ++gReleased; return kSPNoError; }
static SPBasicSuite gBasic = {};
static SPBasicSuite* GetBasic() { return &gBasic; }

// ============================ CPU helpers ====================================

struct World {
  std::vector<char> store;
  PF_EffectWorld w = {};
};

// Deterministic content with out-of-range values for float.
static World MakeWorld(int width, int height, int bpp, int pad, bool flip, bool deep, unsigned seed) {
  World out;
  int rb = width * bpp + pad;
  out.store.assign((size_t)rb * height, (char)0xA5);  // padding sentinel
  unsigned s = seed * 2654435761u + 12345u;
  for (int y = 0; y < height; ++y) {
    char* row = out.store.data() + (size_t)y * rb;
    for (int i = 0; i < width * bpp / (bpp == 16 ? 4 : (bpp == 8 ? 2 : 1)); ++i) {
      s = s * 1664525u + 1013904223u;
      if (bpp == 4) ((uint8_t*)row)[i] = (uint8_t)(s >> 24);
      else if (bpp == 8) ((uint16_t*)row)[i] = (uint16_t)((s >> 16) % 32769u);
      else ((float*)row)[i] = ((int)(s >> 16) % 3000 - 500) / 1000.f;  // -0.5 .. 2.5
    }
  }
  out.w.width = width; out.w.height = height;
  out.w.rowbytes = flip ? -rb : rb;
  out.w.data = (PF_PixelPtr)(flip ? out.store.data() + (size_t)(height - 1) * rb : out.store.data());
  if (deep) out.w.world_flags = PF_WorldFlag_DEEP;
  return out;
}

// A host pixel format on the CPU path.
struct HostFmt {
  const char* label;
  bool premiere;
  PrPixelFormat fmt;  // Premiere; PrPixelFormat_Invalid in After Effects
  int bpp;            // 4, 8 or 16 bytes per pixel
};
static const HostFmt kBGRA8 = {"Premiere BGRA 8u", true, PrPixelFormat_BGRA_4444_8u, 4};
static const HostFmt kBGRA16 = {"Premiere BGRA 16u", true, PrPixelFormat_BGRA_4444_16u, 8};
static const HostFmt kBGRA32 = {"Premiere BGRA 32f", true, PrPixelFormat_BGRA_4444_32f, 16};
static const HostFmt kARGB8 = {"After Effects ARGB 8", false, PrPixelFormat_Invalid, 4};
static const HostFmt kARGB16 = {"After Effects ARGB 16", false, PrPixelFormat_Invalid, 8};

// Memory position of channel c (0 B, 1 G, 2 R, 3 A) in a pixel.
static int Pos(const HostFmt& f, int c) { return f.premiere ? c : 3 - c; }
static char* RowOf(const PF_EffectWorld& w, int y) { return (char*)w.data + (ptrdiff_t)y * w.rowbytes; }
// Channel c of pixel x, scaled so that white is 1.
static double GetPx(const HostFmt& f, const PF_EffectWorld& w, int x, int y, int c) {
  const char* row = RowOf(w, y);
  const int i = x * 4 + Pos(f, c);
  if (f.bpp == 4) return ((const uint8_t*)row)[i] / 255.0;
  if (f.bpp == 8) return ((const uint16_t*)row)[i] / 32768.0;
  return ((const float*)row)[i];
}
// Stores v quantised the way a host would (round to nearest, clamp).
static void SetPx(const HostFmt& f, PF_EffectWorld& w, int x, int y, int c, double v) {
  char* row = RowOf(w, y);
  const int i = x * 4 + Pos(f, c);
  if (f.bpp == 4) ((uint8_t*)row)[i] = (uint8_t)std::lround(std::min(255.0, std::max(0.0, v * 255.0)));
  else if (f.bpp == 8) ((uint16_t*)row)[i] = (uint16_t)std::lround(std::min(32768.0, std::max(0.0, v * 32768.0)));
  else ((float*)row)[i] = (float)v;
}
static bool SameRaw(const HostFmt& f, const PF_EffectWorld& a, const PF_EffectWorld& b, int x, int y, int c) {
  const int n = f.bpp / 4;
  return memcmp(RowOf(a, y) + (size_t)(x * 4 + Pos(f, c)) * n, RowOf(b, y) + (size_t)(x * 4 + Pos(f, c)) * n, n) == 0;
}

// ---- The parameter table, written out here --------------------------------------- Not
// taken from the plug-in header, so that an accidental edit there fails. Effect Controls
// order: the PF index of entry n is n + 1; each control keeps its disk ID wherever it sits.
// The fake hosts address every parameter by disk ID through this table, as Premiere matches
// saved values by disk ID: the CPU host's params[] and the GPU host's GetParam index (with
// or without the topic entries) both follow it.
enum { kFS = PF_Param_FLOAT_SLIDER, kSL = PF_Param_SLIDER, kPU = PF_Param_POPUP, kCB = PF_Param_CHECKBOX,
       kGS = PF_Param_GROUP_START, kGE = PF_Param_GROUP_END, kBT = PF_Param_BUTTON, kAR = PF_Param_ARBITRARY_DATA };
static const int kCol = PF_ParamFlag_START_COLLAPSED, kFix = PF_ParamFlag_CANNOT_TIME_VARY,
                 kSup = PF_ParamFlag_SUPERVISE, kOld = PF_ParamFlag_USE_VALUE_FOR_OLD_PROJECTS;
struct Expect { const char* name; int type; int id; double dflt, lo, hi; int flags; };
static const std::vector<Expect> kExpect = [] {
  std::vector<Expect> v = {
    {"Look", kPU, 65, 1, 1, 5, kSup | kFix | kOld},
    {"Look", kPU, 72, 1, 1, 6, kSup | kFix},
    {"Switch to dB controls", kBT, 102, 0, 0, 0, kSup | kFix},
    {"Noise: measuring...", kBT, 73, 0, 0, 0, kSup | kFix},
    {"Help", kBT, 74, 0, 0, 0, kSup | kFix},
    {"Luma strength", kFS, 1, 80, 0, 100, kSup},
    {"Luma reduction (dB)", kFS, 75, 14, 0, 30, kSup},
    {"Colour strength", kFS, 2, 100, 0, 100, kSup},
    {"Colour reduction (dB)", kFS, 76, 30, 0, 30, kSup},
    {"Minimum grain", kFS, 19, 100, 0, 1000, kSup},
    {"Keep grain (10-bit)", kFS, 77, 0.657, 0, 6.57, kSup},
    {"Spatial strength", kFS, 5, 50, 0, 100, kSup},
    {"Noise threshold (dB)", kFS, 78, 0, -18, 6, kSup},
    {"Frames each side", kSL, 3, 2, 0, 5, kSup},
    {"Frames used", kPU, 79, 3, 1, 6, kSup | kFix},
    {"View", kPU, 9, 1, 1, 5, kSup},
    {"Quality", kPU, 8, 1, 1, 3, kSup},
    {"Noise profile", kGS, 42, 0, 0, 0, kCol},
    {"Profile graph", kAR, 105, 0, 0, 0, kFix},
    {"Profile", kPU, 52, 1, 1, 2, kFix | kSup | kOld},
    {"Capture profile", kBT, 53, 0, 0, 0, kFix | kSup},
    {"Capture revision", kSL, 54, 0, 0, 2147483647.0, kFix},
    {"Captured profile", kAR, 55, 0, 0, 0, kFix},
    {"Capture identity 1", kSL, 56, 0, -2147483648.0, 2147483647.0, kFix},
    {"Capture identity 2", kSL, 57, 0, -2147483648.0, 2147483647.0, kFix},
    {"Capture identity 3", kSL, 58, 0, -2147483648.0, 2147483647.0, kFix},
    {"Capture identity 4", kSL, 59, 0, -2147483648.0, 2147483647.0, kFix},
    {"Capture identity 5", kSL, 60, 0, -2147483648.0, 2147483647.0, kFix},
    {"Capture identity 6", kSL, 61, 0, -2147483648.0, 2147483647.0, kFix},
    {"Capture identity 7", kSL, 62, 0, -2147483648.0, 2147483647.0, kFix},
    {"Capture identity 8", kSL, 63, 0, -2147483648.0, 2147483647.0, kFix},
    {"Profile info", kBT, 64, 0, 0, 0, kFix | kSup},
    {"Freeze (this session)", kCB, 43, 0, 0, 1, kFix | kSup},
    {"Noise level", kFS, 7, 100, 25, 400, kSup},
    {"Profile correction (dB)", kFS, 80, 0, -12.05, 12.05, kSup},
    {"", kGE, 47, 0, 0, 0, 0},
    {"Temporal", kGS, 32, 0, 0, 0, kCol},
    {"Temporal strength", kFS, 4, 100, 0, 100, kSup},
    {"Motion threshold", kFS, 33, 100, 50, 200, kSup},
    {"Motion threshold (dB)", kFS, 81, 0, -6.02, 6.02, kSup},
    {"Edge ghost protection", kFS, 34, 75, 0, 100, kSup},
    {"", kGE, 35, 0, 0, 0, 0},
    {"Luma by size", kGS, 10, 0, 0, 0, kCol},
    {"Luma fine strength", kFS, 15, 100, 0, 100, kSup},
    {"Luma fine 1-2 px limit (dB)", kFS, 82, 30, 0, 30, kSup},
    {"Luma medium strength", kFS, 16, 100, 0, 100, kSup},
    {"Luma medium 2-4 px limit (dB)", kFS, 83, 30, 0, 30, kSup},
    {"Luma coarse strength", kFS, 17, 100, 0, 100, kSup},
    {"Luma coarse 4-8 px limit (dB)", kFS, 84, 30, 0, 30, kSup},
    {"Luma very coarse strength", kFS, 18, 100, 0, 100, kSup},
    {"Luma large 8+ px limit (dB)", kFS, 85, 30, 0, 30, kSup},
    {"Remove specks", kCB, 20, 1, 0, 1, kFix | kSup},
    {"", kGE, 21, 0, 0, 0, 0},
    {"Colour by size", kGS, 22, 0, 0, 0, kCol},
    {"Colour fine strength", kFS, 27, 100, 0, 100, kSup},
    {"Colour fine 1-2 px limit (dB)", kFS, 86, 30, 0, 30, kSup},
    {"Colour medium strength", kFS, 28, 100, 0, 100, kSup},
    {"Colour medium 2-4 px limit (dB)", kFS, 87, 30, 0, 30, kSup},
    {"Colour coarse strength", kFS, 29, 100, 0, 100, kSup},
    {"Colour coarse 4-8 px limit (dB)", kFS, 88, 30, 0, 30, kSup},
    {"Colour very coarse strength", kFS, 30, 100, 0, 100, kSup},
    {"Colour large 8+ px limit (dB)", kFS, 89, 30, 0, 30, kSup},
    {"", kGE, 31, 0, 0, 0, 0},
    {"Sharpen", kGS, 36, 0, 0, 0, kCol},
    {"Sharpen", kFS, 6, 0, 0, 100, kSup},
    {"Sharpen fine", kFS, 37, 100, 0, 250, kSup},
    {"Sharpen medium", kFS, 38, 100, 0, 250, kSup},
    {"Sharpen noise floor", kFS, 39, 100, 50, 300, kSup},
    {"Sharpen threshold (dB)", kFS, 90, 0, -6.02, 9.54, kSup},
    {"Sharpen halo limit", kFS, 40, 100, 0, 500, kSup},
    {"", kGE, 41, 0, 0, 0, 0},
    {"Advanced", kGS, 66, 0, 0, 0, kCol},
    {"Keep grain applies to", kPU, 68, 1, 1, 2, kSup | kFix | kOld},
    {"Processing", kPU, 106, 2, 1, 2, kSup | kFix | kOld},
    {"Luma noise level", kFS, 44, 100, 25, 400, kSup},
    {"Luma correction (dB)", kFS, 91, 0, -12, 12, kSup},
    {"Cb noise level", kFS, 45, 100, 25, 400, kSup},
    {"Cb correction (dB)", kFS, 92, 0, -12, 12, kSup},
    {"Cr noise level", kFS, 46, 100, 25, 400, kSup},
    {"Cr correction (dB)", kFS, 93, 0, -12, 12, kSup},
    {"Luma fine noise level", kFS, 11, 100, 25, 400, kSup},
    {"Luma fine correction (dB)", kFS, 94, 0, -12, 12, kSup},
    {"Luma medium noise level", kFS, 12, 100, 25, 400, kSup},
    {"Luma medium correction (dB)", kFS, 95, 0, -12, 12, kSup},
    {"Luma coarse noise level", kFS, 13, 100, 25, 400, kSup},
    {"Luma coarse correction (dB)", kFS, 96, 0, -12, 12, kSup},
    {"Luma very coarse noise level", kFS, 14, 100, 25, 400, kSup},
    {"Luma large correction (dB)", kFS, 97, 0, -12, 12, kSup},
    {"Colour fine noise level", kFS, 23, 100, 25, 400, kSup},
    {"Colour fine correction (dB)", kFS, 98, 0, -12, 12, kSup},
    {"Colour medium noise level", kFS, 24, 100, 25, 400, kSup},
    {"Colour medium correction (dB)", kFS, 99, 0, -12, 12, kSup},
    {"Colour coarse noise level", kFS, 25, 100, 25, 400, kSup},
    {"Colour coarse correction (dB)", kFS, 100, 0, -12, 12, kSup},
    {"Colour very coarse noise level", kFS, 26, 100, 25, 400, kSup},
    {"Colour large correction (dB)", kFS, 101, 0, -12, 12, kSup},
    {"", kGE, 67, 0, 0, 0, 0},
    {"Experimental", kGS, 48, 0, 0, 0, kCol},
    {"Grain stability", kFS, 49, 0, 0, 100, kSup},
    {"Coarse colour cleanup", kCB, 50, 0, 0, 1, kFix | kSup},
    {"", kGE, 51, 0, 0, 0, 0},
    {"Capture request low", kSL, 69, 0, -2147483648.0, 2147483647.0, kFix},
    {"Capture request high", kSL, 70, 0, -2147483648.0, 2147483647.0, kFix},
    {"Control set", kPU, 71, 2, 1, 2, kSup | kFix | kOld},
    {"Analysis key low", kSL, 103, 0, -2147483648, 2147483647, kFix},
    {"Analysis key high", kSL, 104, 0, -2147483648, 2147483647, kFix},
  };
  if (odplug::kUseMinimalPanel) {
    // Independent specification of the flattened order; never read kParamOrder.
    const int order[] = {
      65,72,1,75,2,76,19,77,9,74,66,102,5,78,3,79,8,
      52,53,54,55,56,57,58,59,60,61,62,63,64,43,7,80,
      4,33,81,34,15,82,16,83,17,84,18,85,20,
      27,86,28,87,29,88,30,89,6,37,38,39,90,40,
      44,91,45,92,46,93,11,94,12,95,13,96,14,97,
      23,98,24,99,25,100,26,101,68,106,49,50,67,
      73,105,69,70,71,103,104,42,47,32,35,10,21,22,31,36,41,48,51,
    };
    std::vector<Expect> minimal;
    for (int id : order) for (auto e : v) if (e.id == id) {
      if ((e.type == kGS || e.type == kGE) && id != 66 && id != 67) e.name = "";
      minimal.push_back(e);
    }
    return minimal;
  }
  if (!odplug::kUseDynamicVisibility) {
    const std::set<int> previous = {65,1,2,19,5,3,4,7,33,15,16,17,18,27,28,29,30,39,44,45,46,11,12,13,14,23,24,25,26};
    std::vector<Expect> fallback;
    for (const auto& e : v) if (!previous.count(e.id)) fallback.push_back(e);
    fallback.push_back({"Previous controls (old project)",kGS,107,0,0,0,kCol});
    for (const auto& e : v) if (previous.count(e.id)) fallback.push_back(e);
    fallback.push_back({"",kGE,108,0,0,0,0});
    return fallback;
  }
  return v;
}();
static const int kNumExpect = (int)kExpect.size();
static const std::vector<int> kTopics = [] {
  std::vector<int> v;
  for (size_t i=0;i<kExpect.size();++i) if (kExpect[i].type == kGS || kExpect[i].type == kGE) v.push_back((int)i+1);
  return v;
}();
static const int kCountTopics = odplug::kUseDynamicVisibility ? 106 : 108, kCountNoTopics = 92;
// PF index of the control with disk ID `id` (-1: this build has none).
static int PfOf(int id) {
  for (int i = 0; i < kNumExpect; ++i)
    if (kExpect[i].id == id) return i + 1;
  return -1;
}
static bool IsTopicAt(int pf) {
  return pf >= 1 && pf <= kNumExpect && (kExpect[pf - 1].type == kGS || kExpect[pf - 1].type == kGE);
}

// One PF_Cmd_RENDER call: in_data and params as the host would pass them.
struct CPUCall {
  PF_InData in_data = {};
  PF_OutData out_data = {};
  std::vector<PF_ParamDef> defs;
  std::vector<PF_ParamDef*> ptrs;
};
// ui: disk ID -> value for the controls to change from their defaults, at
// the PF index the parameter table gives (the host matches by disk ID).
// oldProject: a project saved by the build of the legacy build, which had none of the
// controls added since (disk IDs above 51): those take the value the plug-in
// registered for old projects where it asked for one
// (PF_ParamFlag_USE_VALUE_FOR_OLD_PROJECTS), else their default, as the
// After Effects SDK describes the host's fill.
static void PrepareCPU(CPUCall& c, bool premiere, const PF_EffectWorld& in, A_long time, intptr_t instanceRef,
                       const std::map<int, double>& ui = {}, bool oldProject = false) {
  c.in_data.appl_id = premiere ? kAppID_Premiere : 'FXTC';
  c.in_data.pica_basicP = &gBasic;
  c.in_data.effect_ref = (PF_ProgPtr)instanceRef;
  c.in_data.inter.checkout_param = FakeCheckout;
  c.in_data.inter.checkin_param = FakeCheckin;
  c.in_data.current_time = time;
  c.in_data.time_step = 1001;
  c.in_data.time_scale = 30000;
  c.in_data.width = in.width; c.in_data.height = in.height;
  c.in_data.downsample_x.num = c.in_data.downsample_x.den = 1;
  c.in_data.downsample_y.num = c.in_data.downsample_y.den = 1;
  c.defs.assign(gCPU.added.size() + 1, PF_ParamDef());
  memset(c.defs.data(), 0, sizeof(PF_ParamDef) * c.defs.size());
  c.defs[0].u.ld = in;
  for (size_t i = 1; i < c.defs.size(); ++i) {
    c.defs[i] = gCPU.added[i - 1];
    PF_ParamDef& d = c.defs[i];
    // Registered value: what an old project gets; dephault: a new effect.
    const bool old = oldProject && d.uu.id > 51 && (d.flags & PF_ParamFlag_USE_VALUE_FOR_OLD_PROJECTS);
    if (old) continue;
    if (d.param_type == PF_Param_FLOAT_SLIDER) d.u.fs_d.value = d.u.fs_d.dephault;
    else if (d.param_type == PF_Param_SLIDER) d.u.sd.value = d.u.sd.dephault;
    else if (d.param_type == PF_Param_POPUP) d.u.pd.value = d.u.pd.dephault;
    else if (d.param_type == PF_Param_CHECKBOX) d.u.bd.value = d.u.bd.dephault;
  }
  for (auto& kv : ui) {
    const int pf = PfOf(kv.first);
    if (pf < 1 || pf >= (int)c.defs.size()) {
      CHECK(false, "PrepareCPU: disk ID %d is not a parameter of this build", kv.first);
      continue;
    }
    PF_ParamDef& d = c.defs[pf];
    if (d.param_type == PF_Param_FLOAT_SLIDER) d.u.fs_d.value = kv.second;
    else if (d.param_type == PF_Param_SLIDER) d.u.sd.value = (A_long)kv.second;
    else if (d.param_type == PF_Param_POPUP) d.u.pd.value = (A_long)kv.second;
    else if (d.param_type == PF_Param_CHECKBOX) d.u.bd.value = (A_long)kv.second;
  }
  c.ptrs.clear();
  for (auto& d : c.defs) c.ptrs.push_back(&d);
}
static PF_Err CallRender(EffectMainFn effectMain, CPUCall& c, PF_EffectWorld& out) {
  return effectMain(PF_Cmd_RENDER, &c.in_data, &c.out_data, c.ptrs.data(), &out, nullptr);
}

// ---- the original shell checks (small random frames) -------------------------

static void RunCPURender(EffectMainFn effectMain, const char* label, bool premiere, PrPixelFormat fmt, int bpp,
                         bool deep, bool flip, int expectCheckouts) {
  const int W = 67, H = 41, PAD = 12;
  gIsPremiere = premiere;
  ResetCheckoutStats();

  World in = MakeWorld(W, H, bpp, PAD, flip, deep, 7);
  World out = MakeWorld(W, H, bpp, PAD, flip, deep, 99);
  std::vector<char> outBefore = out.store;
  SetWorldFormat(in.w.data, fmt);
  SetWorldFormat(out.w.data, fmt);
  gCPU.source = CheckoutSource();
  gCPU.source.world = in.w;
  gCPU.source.format = fmt;

  CPUCall call;
  PrepareCPU(call, premiere, in.w, 10 * 1001, 0);
  PF_Err err = CallRender(effectMain, call, out.w);
  CHECK(err == PF_Err_NONE, "%s: render returned %d", label, (int)err);
  CHECK(gCPU.checkouts == expectCheckouts, "%s: %d checkouts, expected %d", label, gCPU.checkouts, expectCheckouts);
  CHECK(gCPU.checkins == gCPU.checkouts, "%s: %d checkins for %d checkouts", label, gCPU.checkins, gCPU.checkouts);
  if (expectCheckouts == 4 && gCPU.checkoutTimes.size() == 4) {
    std::vector<A_long> t = gCPU.checkoutTimes;
    std::sort(t.begin(), t.end());
    CHECK(t[0] == 8 * 1001 && t[1] == 9 * 1001 && t[2] == 11 * 1001 && t[3] == 12 * 1001, "%s: checkout times", label);
  }

  // Compare pixels; padding must be untouched.
  const int rb = W * bpp + PAD;
  size_t pixelDiffs = 0, padDiffs = 0, nonFinite = 0;
  for (int y = 0; y < H; ++y) {
    const char* a = in.store.data() + (size_t)y * rb;
    const char* b = out.store.data() + (size_t)y * rb;
    if (memcmp(a, b, (size_t)W * bpp) != 0) ++pixelDiffs;
    if (memcmp(b + W * bpp, outBefore.data() + (size_t)y * rb + W * bpp, PAD) != 0) ++padDiffs;
    if (bpp == 16)
      for (int i = 0; i < W * 4; ++i)
        if (!std::isfinite(((const float*)b)[i])) ++nonFinite;
  }
  CHECK(padDiffs == 0, "%s: %zu rows had their padding overwritten", label, padDiffs);
  CHECK(nonFinite == 0, "%s: %zu non-finite output values", label, nonFinite);
  if (!gReal) CHECK(pixelDiffs == 0, "%s: %zu of %d rows differ from the input", label, pixelDiffs, H);
  CHECK(memcmp(out.store.data(), outBefore.data(), out.store.size()) != 0, "%s: output was never written", label);
  printf("  cpu %-28s ok=%d checkouts=%d rowsDiffer=%zu\n", label, err == PF_Err_NONE, gCPU.checkouts, pixelDiffs);
}

// The CPU path's float buffers (six frames at radius 2) are freed once software
// renders stop (OD_CPU_IDLE_MS, 300 ms here), not kept for the process.
static void RunCPUIdleRelease(EffectMainFn effectMain) {
  const int W = 1280, H = 720;
  const size_t frameBytes = (size_t)W * H * 16;
  gIsPremiere = true;
  ResetCheckoutStats();
  World in = MakeWorld(W, H, 16, 0, false, false, 31);
  World out = MakeWorld(W, H, 16, 0, false, false, 32);
  SetWorldFormat(in.w.data, PrPixelFormat_BGRA_4444_32f);
  SetWorldFormat(out.w.data, PrPixelFormat_BGRA_4444_32f);
  gCPU.source = CheckoutSource();
  gCPU.source.world = in.w;
  gCPU.source.format = PrPixelFormat_BGRA_4444_32f;
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  CPUCall call;
  PrepareCPU(call, true, in.w, 10 * 1001, 905);
  CHECK(CallRender(effectMain, call, out.w) == PF_Err_NONE, "cpu idle release: render failed");
  const double held = (double)device.currentAllocatedSize;
  bool released = false;
  double after = held;
  for (int i = 0; i < 40 && !released; ++i) {  // up to 2 s
    usleep(50000);
    after = (double)device.currentAllocatedSize;
    released = held - after >= 4.0 * (double)frameBytes;
  }
  printf("  cpu idle release: %.1f MB held after a 1280x720 render, %.1f MB after the idle time\n", held / 1048576.0,
         after / 1048576.0);
  CHECK(released, "cpu idle release: the buffer pool was not freed (%.1f MB, then %.1f MB; frames are %.1f MB)",
        held / 1048576.0, after / 1048576.0, (double)frameBytes / 1048576.0);
}

// ---- stub core: the conversions, exactly, in both directions -----------------

// OD_STUB_FILL: the core's output is a known float pattern; every output
// value must be that pattern converted to the host's format and channel order.
// Row-dependent, so a row-order mistake on bottom-up worlds shows too.
static void RunCPUFill(EffectMainFn effectMain, const HostFmt& f, bool flip) {
  const int W = 37, H = 19, PAD = 8;
  gIsPremiere = f.premiere;
  World in = MakeWorld(W, H, f.bpp, PAD, flip, f.bpp == 8, 3);
  World out = MakeWorld(W, H, f.bpp, PAD, flip, f.bpp == 8, 4);
  std::vector<char> outBefore = out.store;
  SetWorldFormat(in.w.data, f.fmt);
  SetWorldFormat(out.w.data, f.fmt);
  gCPU.source = CheckoutSource();
  gCPU.source.world = in.w;
  gCPU.source.format = f.fmt;
  CPUCall call;
  PrepareCPU(call, f.premiere, in.w, 10 * 1001, 0);
  setenv("OD_STUB_FILL", "1", 1);
  PF_Err err = CallRender(effectMain, call, out.w);
  unsetenv("OD_STUB_FILL");
  size_t bad = 0, padDiffs = 0;
  double worst = 0;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int c = 0; c < 4; ++c) {
        const double v = StubFillValue(c, (uint32_t)y);
        double want, got = GetPx(f, out.w, x, y, c);
        if (f.bpp == 4) want = std::floor(v * 255.0 + 0.5) / 255.0;
        else if (f.bpp == 8) want = std::floor(v * 32768.0 + 0.5) / 32768.0;
        else want = (float)v;
        if (got != want) { ++bad; worst = std::max(worst, std::fabs(got - want)); }
      }
    }
    const size_t rb = (size_t)W * f.bpp + PAD;
    const size_t row = flip ? (size_t)(H - 1 - y) : (size_t)y;
    if (memcmp(out.store.data() + row * rb + (size_t)W * f.bpp, outBefore.data() + row * rb + (size_t)W * f.bpp, PAD)) ++padDiffs;
  }
  CHECK(err == PF_Err_NONE, "fill %s: err %d", f.label, (int)err);
  CHECK(bad == 0, "fill %s%s: %zu output values differ from the core's output converted (worst %.6f)", f.label,
        flip ? " bottom-up" : "", bad, worst);
  CHECK(padDiffs == 0, "fill %s: padding overwritten in %zu rows", f.label, padDiffs);
  printf("  cpu stub fill  %-22s%s: %zu wrong values\n", f.label, flip ? " bottom-up" : "", bad);
}

// OD_STUB_CHECK: the input (and so every neighbour, a copy of it) holds a known
// pattern in the host's format; the stub compares what it receives, as float
// BGRA, within half a code value, and answers with white (all matched) or
// black. Catches errors that cancel out on the way back (channel order,
// white level).
static void RunCPUCheck(EffectMainFn effectMain, const HostFmt& f, bool flip) {
  const int W = 41, H = 23, PAD = 4;
  gIsPremiere = f.premiere;
  ResetCheckoutStats();
  World in = MakeWorld(W, H, f.bpp, PAD, flip, f.bpp == 8, 5);
  World out = MakeWorld(W, H, f.bpp, PAD, flip, f.bpp == 8, 6);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      for (int c = 0; c < 4; ++c) SetPx(f, in.w, x, y, c, StubCheckValue(c, (uint32_t)x, (uint32_t)y));
  SetWorldFormat(in.w.data, f.fmt);
  SetWorldFormat(out.w.data, f.fmt);
  gCPU.source = CheckoutSource();
  gCPU.source.world = in.w;
  gCPU.source.format = f.fmt;
  CPUCall call;
  PrepareCPU(call, f.premiere, in.w, 10 * 1001, 0);
  const double tol = f.bpp == 4 ? 0.5 / 255 + 1e-6 : f.bpp == 8 ? 0.5 / 32768 + 1e-7 : 1e-7;
  char tolText[32];
  snprintf(tolText, sizeof tolText, "%.9g", tol);
  setenv("OD_STUB_CHECK", tolText, 1);
  PF_Err err = CallRender(effectMain, call, out.w);
  unsetenv("OD_STUB_CHECK");
  size_t notWhite = 0;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      for (int c = 0; c < 4; ++c)
        if (GetPx(f, out.w, x, y, c) != 1.0) ++notWhite;
  CHECK(err == PF_Err_NONE && gCPU.checkouts == 4, "check %s: err %d, %d checkouts", f.label, (int)err, gCPU.checkouts);
  CHECK(notWhite == 0, "check %s%s: the core did not receive the host's values (current frame and 4 neighbours) "
        "as float BGRA within half a code value", f.label, flip ? " bottom-up" : "");
  printf("  cpu stub check %-22s%s: %s\n", f.label, flip ? " bottom-up" : "", notWhite ? "MISMATCH" : "matched");
}

// ---- real core: levels, alpha and noise in every format ----------------------

static const int kRealW = 640, kRealH = 360;
static const float kRealSigma = 0.02f;

struct Rng {
  unsigned s;
  explicit Rng(unsigned seed) : s(seed * 747796405u + 2891336453u) {}
  float uni() { s = s * 1664525u + 1013904223u; return (float)(s >> 8) / 16777216.f; }
  // Sum of four uniforms: variance 1/3, so scale by sqrt(3).
  float gauss() { return (uni() + uni() + uni() + uni() - 2.f) * 1.7320508f; }
};

// Flat grey with Gaussian-like noise, alpha 1 (GPU frames, float BGRA).
static void FillNoisy(float* px, size_t pixels, unsigned seed) {
  Rng r(seed);
  for (size_t i = 0; i < pixels; ++i) {
    for (int c = 0; c < 3; ++c) px[i * 4 + c] = 0.4f + kRealSigma * r.gauss();
    px[i * 4 + 3] = 1.f;
  }
}

// Distinct level per colour channel, noise, and an alpha that varies.
static const double kLevel[3] = {0.30, 0.45, 0.60};  // B, G, R
static double AlphaAt(const HostFmt& f, int x, int y) {
  if (f.bpp == 4) return ((x * 3 + y * 5) % 200 + 40) / 255.0;
  if (f.bpp == 8) return ((x * 389 + y * 97) % 30000 + 1000) / 32768.0;
  return 0.1 + ((x * 7 + y * 3) % 80) / 100.0;
}
static void FillLevels(const HostFmt& f, PF_EffectWorld& w, unsigned seed, double sigma) {
  Rng r(seed);
  for (int y = 0; y < w.height; ++y)
    for (int x = 0; x < w.width; ++x) {
      for (int c = 0; c < 3; ++c) SetPx(f, w, x, y, c, kLevel[c] + sigma * r.gauss());
      SetPx(f, w, x, y, 3, AlphaAt(f, x, y));
    }
}
struct ChannelStats { double mean[3], sigma[3]; };
static ChannelStats StatsOf(const HostFmt& f, const PF_EffectWorld& w) {
  ChannelStats s = {};
  for (int c = 0; c < 3; ++c) {
    double sum = 0, sum2 = 0;
    size_t n = 0;
    for (int y = 16; y < w.height - 16; ++y)
      for (int x = 16; x < w.width - 16; ++x) {
        const double v = GetPx(f, w, x, y, c);
        sum += v; sum2 += v * v; ++n;
      }
    s.mean[c] = sum / (double)n;
    s.sigma[c] = std::sqrt(std::max(0.0, sum2 / (double)n - s.mean[c] * s.mean[c]));
  }
  return s;
}

// Each host format at 640x360: neighbours with their own noise, then none.
static void RunCPURealFormat(EffectMainFn effectMain, const HostFmt& f, unsigned seed) {
  const int W = kRealW, H = kRealH;
  gIsPremiere = f.premiere;
  double ratio[2] = {0, 0};
  for (int pass = 0; pass < 2; ++pass) {
    const bool withNeighbours = pass == 0;
    ResetCheckoutStats();
    World in = MakeWorld(W, H, f.bpp, 0, false, f.bpp == 8, 1);
    World out = MakeWorld(W, H, f.bpp, 0, false, f.bpp == 8, 2);
    FillLevels(f, in.w, seed + (unsigned)pass * 100, kRealSigma);
    SetWorldFormat(in.w.data, f.fmt);
    SetWorldFormat(out.w.data, f.fmt);
    gCPU.source = CheckoutSource();
    gCPU.source.world = in.w;
    gCPU.source.format = f.fmt;
    const unsigned nseed = seed + (unsigned)pass * 100;
    gCPU.source.gen = [f, nseed](A_long t, PF_EffectWorld& w) { FillLevels(f, w, nseed + 1 + (unsigned)t, kRealSigma); };
    gCPU.failCheckouts = !withNeighbours;
    CPUCall call;
    // Luma 100 %: at the default 80 % the amount, not the filter, sets much
    // of what is left, and neighbours would show less.
    PrepareCPU(call, f.premiere, in.w, (40 + 20 * pass) * 1001, 0, {{OD_P_LUMA, 100}});
    PF_Err err = CallRender(effectMain, call, out.w);
    gCPU.failCheckouts = false;
    gCPU.source.gen = nullptr;
    const ChannelStats a = StatsOf(f, in.w), b = StatsOf(f, out.w);
    size_t alphaDiffs = 0;
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        if (!SameRaw(f, in.w, out.w, x, y, 3)) ++alphaDiffs;
    CHECK(err == PF_Err_NONE, "real %s: err %d", f.label, (int)err);
    CHECK(alphaDiffs == 0, "real %s: alpha changed in %zu pixels", f.label, alphaDiffs);
    for (int c = 0; c < 3; ++c)
      CHECK(std::fabs(b.mean[c] - a.mean[c]) < 1.0 / 255, "real %s: channel %c mean %.5f -> %.5f (more than one 8-bit code)",
            f.label, "BGR"[c], a.mean[c], b.mean[c]);
    ratio[pass] = b.sigma[1] / a.sigma[1];
    CHECK(ratio[pass] < 0.8, "real %s %s: noise %.5f -> %.5f", f.label, withNeighbours ? "with neighbours" : "spatial only",
          a.sigma[1], b.sigma[1]);
    printf("  cpu real %-22s %-15s: means B %.4f->%.4f G %.4f->%.4f R %.4f->%.4f, alpha diffs %zu, noise x%.2f\n",
           f.label, withNeighbours ? "4 neighbours" : "no neighbours", a.mean[0], b.mean[0], a.mean[1], b.mean[1],
           a.mean[2], b.mean[2], alphaDiffs, ratio[pass]);
  }
  CHECK(ratio[0] < 0.7 * ratio[1], "real %s: neighbours must count: noise x%.2f with, x%.2f without", f.label, ratio[0],
        ratio[1]);
}

// ---- two layers rendered alternately (After Effects; Premiere without ID) ----
// The result for a layer must not depend on what another layer rendered
// before it. Layer A: flat grey, heavy noise. Layer B: almost clean, with fine
// texture.
static std::vector<std::vector<char>> RenderLayers(EffectMainFn effectMain, const HostFmt& f, const char* order,
                                                   intptr_t refA, intptr_t refB) {
  const int W = kRealW, H = kRealH;
  std::vector<std::vector<char>> outputs;
  int frame[2] = {0, 0};
  for (const char* p = order; *p; ++p) {
    const int layer = *p == 'A' ? 0 : 1;
    const intptr_t ref = layer == 0 ? refA : refB;
    const unsigned seed = 1000u * (unsigned)(layer + 1) + (unsigned)frame[layer];
    auto fill = [f, layer](PF_EffectWorld& w, unsigned s) {
      Rng r(s);
      for (int y = 0; y < w.height; ++y)
        for (int x = 0; x < w.width; ++x) {
          const double tex = layer == 1 && ((x / 5 + y / 5) & 1) ? 0.012 : 0.0;
          const double sigma = layer == 0 ? 0.02 : 0.002;
          for (int c = 0; c < 3; ++c) SetPx(f, w, x, y, c, 0.45 + tex + sigma * r.gauss());
          SetPx(f, w, x, y, 3, 1.0);
        }
    };
    World in = MakeWorld(W, H, f.bpp, 0, false, f.bpp == 8, 1);
    World out = MakeWorld(W, H, f.bpp, 0, false, f.bpp == 8, 2);
    fill(in.w, seed);
    SetWorldFormat(in.w.data, f.fmt);
    SetWorldFormat(out.w.data, f.fmt);
    CheckoutSource src;
    src.world = in.w;
    src.format = f.fmt;
    src.gen = [fill, seed](A_long t, PF_EffectWorld& w) { fill(w, seed * 31u + (unsigned)t); };
    {
      std::lock_guard<std::mutex> lock(gCPU.mutex);
      gCPU.sources[(PF_ProgPtr)ref] = src;
    }
    CPUCall call;
    PrepareCPU(call, f.premiere, in.w, (100 + frame[layer]) * 1001, ref);
    PF_Err err = CallRender(effectMain, call, out.w);
    CHECK(err == PF_Err_NONE, "layers %s: err %d", f.label, (int)err);
    outputs.push_back(out.store);
    ++frame[layer];
  }
  std::lock_guard<std::mutex> lock(gCPU.mutex);
  gCPU.sources.clear();
  return outputs;
}
static void RunCPULayers(EffectMainFn effectMain, const HostFmt& f, const char* label, intptr_t aloneA, intptr_t aloneB,
                         intptr_t altA, intptr_t altB, bool expectSame) {
  gIsPremiere = f.premiere;
  std::vector<std::vector<char>> a = RenderLayers(effectMain, f, "AAA", aloneA, aloneA);
  std::vector<std::vector<char>> b = RenderLayers(effectMain, f, "BBB", aloneB, aloneB);
  std::vector<std::vector<char>> alt = RenderLayers(effectMain, f, "ABABAB", altA, altB);
  // Equal within one code value (the GPU need not be bit-reproducible).
  const int n = f.bpp / 4;
  auto codeDiff = [&](const std::vector<char>& x, const std::vector<char>& y) {
    double worst = 0;
    for (size_t i = 0; i < x.size() / (size_t)n; ++i) {
      double d;
      if (n == 1) d = std::abs((int)(uint8_t)x[i] - (int)(uint8_t)y[i]);
      else if (n == 2) d = std::abs((int)((const uint16_t*)x.data())[i] - (int)((const uint16_t*)y.data())[i]);
      else d = std::fabs(((const float*)x.data())[i] - ((const float*)y.data())[i]) * 65535;
      worst = std::max(worst, d);
    }
    return worst;
  };
  int sameA = 0, sameB = 0;
  double worst = 0;
  for (int i = 0; i < 3; ++i) {
    const double da = codeDiff(alt[(size_t)(2 * i)], a[(size_t)i]), db = codeDiff(alt[(size_t)(2 * i + 1)], b[(size_t)i]);
    sameA += da <= 1;
    sameB += db <= 1;
    worst = std::max(worst, std::max(da, db));
  }
  if (expectSame) {
    CHECK(sameA == 3 && sameB == 3, "%s: rendered alternately, %d of 3 frames of A and %d of 3 of B equal their "
          "result alone", label, sameA, sameB);
  } else {
    CHECK(sameA + sameB < 6, "%s (control): one session for both layers must change a result, or this test "
          "cannot see sharing", label);
  }
  printf("  cpu layers %-34s: alternating = alone for %d/3 of A, %d/3 of B (largest difference %.0f codes)\n", label,
         sameA, sameB, worst);
}

// ---- re-entrancy: a checkout that renders another Open Denoise instance ------
// As a host must when what is upstream carries the effect too (a nest and
// inside it, an AE precomp and inside it). Same thread, then another thread
// while the host's checkout waits.
static std::atomic<bool> gInUpstream{false};
static void RunCPUReentrant(EffectMainFn effectMain, bool otherThread) {
  const int W = 320, H = 180;
  const HostFmt& f = kBGRA32;
  gIsPremiere = true;
  ResetCheckoutStats();
  World in = MakeWorld(W, H, 16, 0, false, false, 1), out = MakeWorld(W, H, 16, 0, false, false, 2);
  World upIn = MakeWorld(W / 2, H / 2, 16, 0, false, false, 3), upOut = MakeWorld(W / 2, H / 2, 16, 0, false, false, 4);
  FillNoisy((float*)in.store.data(), (size_t)W * H, 11);
  FillNoisy((float*)upIn.store.data(), (size_t)W * H / 4, 12);
  std::vector<char> outBefore = out.store, upBefore = upOut.store;
  for (World* w : {&in, &out, &upIn, &upOut}) SetWorldFormat(w->w.data, f.fmt);
  const intptr_t outerRef = 501, innerRef = 502;
  {
    std::lock_guard<std::mutex> lock(gCPU.mutex);
    gCPU.sources[(PF_ProgPtr)outerRef] = CheckoutSource{in.w, f.fmt, nullptr};
    gCPU.sources[(PF_ProgPtr)innerRef] = CheckoutSource{upIn.w, f.fmt, nullptr};
  }
  std::atomic<int> inner{0}, innerErrors{0};
  auto renderUpstream = [&] {
    CPUCall c;
    PrepareCPU(c, true, upIn.w, 20 * 1001, innerRef);
    if (CallRender(effectMain, c, upOut.w) != PF_Err_NONE) ++innerErrors;
    ++inner;
  };
  gCPU.onCheckout = [&](PF_ProgPtr ref) {
    if (ref != (PF_ProgPtr)outerRef || gInUpstream.exchange(true)) return;  // one level
    if (otherThread) {
      std::thread t(renderUpstream);
      t.join();
    } else {
      renderUpstream();
    }
    gInUpstream = false;
  };
  PF_Err err = PF_Err_BAD_CALLBACK_PARAM;  // until the render returns
  const char* label = otherThread ? "re-entrant render on another thread" : "re-entrant render on the same thread";
  const bool done = FinishesWithin(20, [&] {
    CPUCall c;
    PrepareCPU(c, true, in.w, 10 * 1001, outerRef);
    err = CallRender(effectMain, c, out.w);
  });
  if (!done) DieDeadlocked(label);
  gCPU.onCheckout = nullptr;
  {
    std::lock_guard<std::mutex> lock(gCPU.mutex);
    gCPU.sources.clear();
  }
  CHECK(err == PF_Err_NONE && innerErrors == 0, "%s: outer err %d, %d inner errors", label, (int)err, innerErrors.load());
  CHECK(inner == 4, "%s: %d upstream renders, expected one per checkout (4)", label, inner.load());
  CHECK(gCPU.checkins == gCPU.checkouts, "%s: %d checkins for %d checkouts", label, gCPU.checkins, gCPU.checkouts);
  CHECK(out.store != outBefore && upOut.store != upBefore, "%s: an output was never written", label);
  if (gReal) {
    // Output frames, not just returns: both were denoised.
    CHECK(StatsOf(f, out.w).sigma[1] < 0.8 * StatsOf(f, in.w).sigma[1], "%s: outer frame not denoised", label);
  } else {
    CHECK(out.store == in.store, "%s: outer output differs from the input under the stub core", label);
  }
  printf("  cpu %-40s: returned, %d upstream renders\n", label, inner.load());
}

// Two host threads rendering at once (Premiere and AE both do), with checkouts.
static void RunCPUConcurrent(EffectMainFn effectMain) {
  const int W = 160, H = 90;
  const HostFmt& f = kBGRA8;
  gIsPremiere = true;
  World in[2] = {MakeWorld(W, H, 4, 0, false, false, 21), MakeWorld(W, H, 4, 0, false, false, 22)};
  for (int i = 0; i < 2; ++i) {
    SetWorldFormat(in[i].w.data, f.fmt);
    std::lock_guard<std::mutex> lock(gCPU.mutex);
    gCPU.sources[(PF_ProgPtr)(intptr_t)(601 + i)] = CheckoutSource{in[i].w, f.fmt, nullptr};
  }
  std::atomic<int> errors{0}, written{0};
  const bool done = FinishesWithin(30, [&] {
    std::thread t[2];
    for (int i = 0; i < 2; ++i)
      t[i] = std::thread([&, i] {
        for (int k = 0; k < 6; ++k) {
          World out = MakeWorld(W, H, 4, 0, false, false, 50u + (unsigned)k);
          std::vector<char> before = out.store;
          SetWorldFormat(out.w.data, f.fmt);
          CPUCall c;
          PrepareCPU(c, true, in[i].w, (10 + k) * 1001, 601 + i);
          if (CallRender(effectMain, c, out.w) != PF_Err_NONE) ++errors;
          if (out.store != before) ++written;
        }
      });
    for (auto& th : t) th.join();
  });
  if (!done) DieDeadlocked("two threads rendering at once");
  {
    std::lock_guard<std::mutex> lock(gCPU.mutex);
    gCPU.sources.clear();
  }
  CHECK(errors == 0 && written == 12, "concurrent CPU renders: %d errors, %d of 12 outputs written", errors.load(),
        written.load());
  printf("  cpu 2 threads x 6 renders at once: errors=%d written=%d\n", errors.load(), written.load());
}

// ---- the advanced controls (core API v6) and the experimental ones (v9) --------
// By disk ID (the parameter table above gives their PF index).
// The core fields the stub records for the advanced controls, at their
// neutral values (od_params_default()).
static const std::map<std::string, double>& AdvancedDefaults() {
  static const std::map<std::string, double> d = {
      {"adv", 1},   {"nlY0", 1},  {"nlY1", 1},  {"nlY2", 1},   {"nlY3", 1},     {"nlC0", 1},  {"nlC1", 1},
      {"nlC2", 1},  {"nlC3", 1},  {"amY0", 1},  {"amY1", 1},   {"amY2", 1},     {"amY3", 1},  {"amC0", 1},
      {"amC1", 1},  {"amC2", 1},  {"amC3", 1},  {"cnY", 1},    {"cnCb", 1},     {"cnCr", 1},  {"floor", 1},
      {"speck", 1}, {"mthr", 1},  {"egate", 0.75}, {"shf", 1}, {"shm", 1},      {"shcore", 1}, {"shhalo", 1},
      {"hold", 0},  {"gstab", 0}, {"cchroma", 0}, {"grain", OD_GRAIN_FINEST}};
  return d;
}
// Every advanced field of a render record equals its default, except the
// ones in `want`, which must equal theirs. Returns a description of the first
// mismatch, empty if none.
static std::string AdvancedMismatch(const Rec& r, const std::map<std::string, double>& want = {}) {
  for (const auto& kv : AdvancedDefaults()) {
    auto w = want.find(kv.first);
    const double expect = w != want.end() ? w->second : kv.second;
    const std::string got = r[kv.first.c_str()];
    if (got.empty() || std::fabs(atof(got.c_str()) - expect) > 1e-5)
      return kv.first + " = " + (got.empty() ? "(missing)" : got) + ", expected " + std::to_string(expect);
  }
  return std::string();
}
// One advanced control set alone: PF index, UI value, core field, value.
struct AdvCase {
  int pf;
  double ui;
  const char* field;
  double want;
  bool checkbox;
  bool popup = false;  // Minimum grain applies to: the 1-based item
};
static std::vector<AdvCase> AdvancedCases() {
  std::vector<AdvCase> v;
  auto slider = [&](int pf, const char* f, double in, double lo, double hi, double inRange) {
    v.push_back({pf, inRange, f, inRange / 100.0, false});
    v.push_back({pf, lo - 10, f, lo / 100.0, false});   // below the range: clamped
    v.push_back({pf, hi + 100, f, hi / 100.0, false});  // above it
    (void)in;
  };
  const char* nlY[4] = {"nlY0", "nlY1", "nlY2", "nlY3"}, *nlC[4] = {"nlC0", "nlC1", "nlC2", "nlC3"};
  const char* amY[4] = {"amY0", "amY1", "amY2", "amY3"}, *amC[4] = {"amC0", "amC1", "amC2", "amC3"};
  for (int b = 0; b < 4; ++b) {
    slider(11 + b, nlY[b], 0, 25, 400, 150 + 10 * b);
    slider(15 + b, amY[b], 0, 0, 100, 40 + 5 * b);
    slider(23 + b, nlC[b], 0, 25, 400, 60 + 20 * b);
    slider(27 + b, amC[b], 0, 0, 100, 10 + 20 * b);
  }
  slider(19, "floor", 0, 0, 1000, 250);
  slider(33, "mthr", 0, 50, 200, 150);
  slider(34, "egate", 0, 0, 100, 30);
  slider(37, "shf", 0, 0, 250, 200);
  slider(38, "shm", 0, 0, 250, 50);
  slider(39, "shcore", 0, 50, 300, 200);
  slider(40, "shhalo", 0, 0, 500, 300);
  slider(44, "cnY", 0, 25, 400, 150);
  slider(45, "cnCb", 0, 25, 400, 180);
  slider(46, "cnCr", 0, 25, 400, 70);
  slider(49, "gstab", 0, 0, 100, 35);
  v.push_back({20, 0, "speck", 0, true});
  v.push_back({20, 1, "speck", 1, true});
  v.push_back({43, 1, "hold", 1, true});
  v.push_back({43, 0, "hold", 0, true});
  v.push_back({50, 1, "cchroma", 1, true});
  v.push_back({50, 0, "cchroma", 0, true});
  // Minimum grain applies to: Whole picture (previous) = OD_GRAIN_WHOLE (0),
  // Finest grain only = OD_GRAIN_FINEST (1), an item no build has = the
  // default (on the CPU path mapped to it; on the GPU path a read that
  // fails its check, so the render takes every advanced default).
  v.push_back({68, OD_GRAIN_MODE_WHOLE, "grain", OD_GRAIN_WHOLE, false, true});
  v.push_back({68, OD_GRAIN_MODE_FINEST, "grain", OD_GRAIN_FINEST, false, true});
  v.push_back({68, 7, "grain", OD_GRAIN_FINEST, false, true});
  return v;
}

// Every advanced and experimental control at once, each slider at a value of its own (inside
// every slider's range: 51 + 1.8 k %), every checkbox off its default: a render that swaps
// two sliders, or reads one at another's index, fails.
struct Sentinel {
  int pf;
  double ui;
  const char* field;
  bool checkbox;
  bool popup = false;  // Minimum grain applies to: the 1-based item
};
// The core value of a sentinel.
static double SentinelCore(const Sentinel& s) {
  if (s.popup) return s.ui == OD_GRAIN_MODE_WHOLE ? OD_GRAIN_WHOLE : OD_GRAIN_FINEST;
  return s.checkbox ? s.ui : s.ui / 100.0;
}
static std::vector<Sentinel> Sentinels() {
  static const struct { int pf; const char* field; } sliders[] = {
      {11, "nlY0"}, {12, "nlY1"}, {13, "nlY2"}, {14, "nlY3"}, {15, "amY0"}, {16, "amY1"}, {17, "amY2"},
      {18, "amY3"}, {19, "floor"}, {23, "nlC0"}, {24, "nlC1"}, {25, "nlC2"}, {26, "nlC3"}, {27, "amC0"},
      {28, "amC1"}, {29, "amC2"}, {30, "amC3"}, {33, "mthr"}, {34, "egate"}, {37, "shf"}, {38, "shm"},
      {39, "shcore"}, {40, "shhalo"}, {44, "cnY"}, {45, "cnCb"}, {46, "cnCr"}, {49, "gstab"}};
  std::vector<Sentinel> v;
  int k = 0;
  for (const auto& sl : sliders) v.push_back({sl.pf, 51.0 + 1.8 * k++, sl.field, false});
  v.push_back({20, 0, "speck", true});
  v.push_back({43, 1, "hold", true});
  v.push_back({50, 1, "cchroma", true});
  v.push_back({68, OD_GRAIN_MODE_WHOLE, "grain", false, true});  // off its default
  return v;
}
static std::map<std::string, double> SentinelWant() {
  std::map<std::string, double> want;
  for (const Sentinel& s : Sentinels()) want[s.field] = SentinelCore(s);
  return want;
}
// A project saved before the Experimental topic (API v9) existed: the host has values for
// every older control and fills Grain stability and Coarse chroma cleanup with their
// defaults, and Minimum grain applies to with the value registered for old projects, Whole
// picture (previous). The older controls keep their values, the two experimental ones arrive
// off and the grain mode is the picture they were saved with.
static bool IsExperimental(int pf) { return pf == OD_P_GRAIN_STABILITY || pf == OD_P_COARSE_CHROMA; }
static bool After584(int id) { return id > 51; }  // a control the legacy build did not have
static std::map<std::string, double> OldProjectWant() {
  std::map<std::string, double> want;
  for (const Sentinel& s : Sentinels())
    if (!IsExperimental(s.pf) && !After584(s.pf)) want[s.field] = SentinelCore(s);
  want["grain"] = OD_GRAIN_WHOLE;
  return want;
}
// A project saved by the legacy build: every control it had at its sentinel, the grain
// mode at its old-project value.
static std::map<std::string, double> Old584Want() {
  std::map<std::string, double> want;
  for (const Sentinel& s : Sentinels())
    if (!After584(s.pf)) want[s.field] = SentinelCore(s);
  want["grain"] = OD_GRAIN_WHOLE;
  return want;
}
// CPU/GPU parity: the advanced fields the stub recorded for the CPU path's
// render with every control at its sentinel, as text; every GPU numbering
// must hand the core the same text (RunCPUParams fills it, TestGPUAdvanced
// compares).
static std::map<std::string, std::string> gCpuSentinelFields;
static std::string ParityMismatch(const Rec& r) {
  if (gCpuSentinelFields.empty()) return "no CPU record to compare with";
  for (const auto& kv : gCpuSentinelFields)
    if (r[kv.first.c_str()] != kv.second)
      return kv.first + " = " + r[kv.first.c_str()] + " on the GPU path, " + kv.second + " on the CPU path";
  return std::string();
}

// ---- A project saved by the legacy build (before Effect Controls v2) ------------------ Every one
// of its 51 controls at a value of its own, the nine original ones included; Look, which it
// did not have, at the value the plug-in registered for old projects. most controls sit
// elsewhere in Effect Controls, so every value must still reach the core from its own
// control (matched by disk ID), on both paths and in every GPU numbering.
struct MainSentinel { int id; double ui; const char* field; double want; };
static const MainSentinel kMainSentinels[] = {
    {OD_P_LUMA, 37, "luma", 0.37},          {OD_P_CHROMA, 81, "chroma", 0.81},
    {OD_P_TEMPORAL_RADIUS, 1, "radius", 1}, {OD_P_TEMPORAL_STRENGTH, 64, "tstr", 0.64},
    {OD_P_SPATIAL, 23, "spatial", 0.23},    {OD_P_SHARPEN, 12, "sharpen", 0.12},
    {OD_P_NOISE_LEVEL, 150, "noise", 1.5},  {OD_P_QUALITY, OD_QUALITY_FULL, "quality", 1},
    {OD_P_VIEW, 3, "view", OD_VIEW_MOTION}};
static std::string MainMismatch(const Rec& r) {
  for (const MainSentinel& m : kMainSentinels)
    if ((r[m.field]).empty() || std::fabs(r.num(m.field) - m.want) > 1e-5)
      return std::string(m.field) + " = " + r[m.field] + ", expected " + std::to_string(m.want);
  return std::string();
}
// The CPU path's record of that project, every field, for the GPU's parity.
static std::map<std::string, std::string> gCpuOld584Fields;
static std::string Old584Parity(const Rec& r) {
  if (gCpuOld584Fields.empty()) return "no CPU record to compare with";
  for (const auto& kv : gCpuOld584Fields)
    if (r[kv.first.c_str()] != kv.second)
      return kv.first + " = " + r[kv.first.c_str()] + " on the GPU path, " + kv.second + " on the CPU path";
  return std::string();
}

// ---- The Look menu (PF_Cmd_USER_CHANGED_PARAM) --------------------------------- The look
// set and the recipes, written out here: values in percent; every look-set control a recipe
// does not name is at Natural's value, which is its default.
static const int kLookSetIds[20] = {1, 2, 19, 68, 15, 16, 17, 18, 27, 28, 29, 30, 11, 12, 13, 14, 23, 24, 25, 26};
static double RecipeOf(int look, int id) {
  // Natural: every default (Luma 80 %, Minimum grain applies to = Finest
  // grain only, the rest 100 %).
  double v = id == OD_P_LUMA ? 80 : id == OD_P_GRAIN_MODE ? OD_GRAIN_MODE_FINEST : 100;
  if (look == OD_LOOK_CLEANER && id == OD_P_GRAIN_FLOOR) v = 0;
  if (look == OD_LOOK_PHONE && id == OD_P_GRAIN_FLOOR) v = 0;
  if (look == OD_LOOK_PHONE && (id == OD_P_LUMA_LEVEL_MEDIUM || id == OD_P_LUMA_LEVEL_COARSE)) v = 150;
  if (look == OD_LOOK_PRESERVE && id == OD_P_LUMA) v = 70;
  return v;
}
// What each Look's recipe hands the core (stub record fields), beyond the
// defaults.
static std::map<std::string, double> RecipeWant(int look) {
  switch (look) {
    case OD_LOOK_CLEANER: return {{"floor", 0}};
    case OD_LOOK_PHONE: return {{"floor", 0}, {"nlY1", 1.5}, {"nlY2", 1.5}};
    default: return {};  // Natural, and Preserve detail (a main field: LookLuma)
  }
}
// Luma strength each Look hands the core.
static double LookLuma(int look) { return look == OD_LOOK_PRESERVE ? 0.7 : 0.8; }
static const char* LookName(int look) {
  static const char* n[] = {"?", "Natural", "Cleaner", "Phone", "Preserve detail", "Custom"};
  return look >= 1 && look <= 5 ? n[look] : "?";
}
// Every host passes the values of every parameter at the current time; the
// handler may write any of them and mark it with PF_ChangeFlag_CHANGED_VALUE.
struct LookHost {
  CPUCall c;
  EffectMainFn effectMain;
  World in;
  LookHost(EffectMainFn fn, bool premiere) : effectMain(fn), in(MakeWorld(8, 8, 4, 0, false, false, 1)) {
    PrepareCPU(c, premiere, in.w, 0, 0, {{71,1}});
  }
  PF_ParamDef& def(int id) { return c.defs[(size_t)PfOf(id)]; }
  double get(int id) {
    PF_ParamDef& d = def(id);
    return d.param_type == PF_Param_POPUP ? d.u.pd.value : d.param_type == PF_Param_FLOAT_SLIDER ? d.u.fs_d.value : 0;
  }
  void set(int id, double v) {
    PF_ParamDef& d = def(id);
    if (d.param_type == PF_Param_POPUP) d.u.pd.value = (A_long)v;
    else if (d.param_type == PF_Param_FLOAT_SLIDER) d.u.fs_d.value = v;
  }
  // The user changed `id`: the host calls the effect, then reads the flags.
  // (uu holds the change flags during this command, not the disk ID.)
  std::set<int> send(int id, PF_Err* err = nullptr) {
    for (size_t i = 1; i < c.defs.size(); ++i) c.defs[i].uu.change_flags = PF_ChangeFlag_NONE;
    PF_UserChangedParamExtra extra = {};
    extra.param_index = PfOf(id);
    const PF_Err e = effectMain(PF_Cmd_USER_CHANGED_PARAM, &c.in_data, &c.out_data, c.ptrs.data(), nullptr, &extra);
    if (err) *err = e;
    std::set<int> flagged;
    for (size_t i = 1; i < c.defs.size(); ++i)
      if (c.defs[i].uu.change_flags & PF_ChangeFlag_CHANGED_VALUE) flagged.insert(kExpect[i - 1].id);
    return flagged;
  }
  // The look set holds exactly `look`'s recipe.
  std::string recipeMismatch(int look) {
    for (int id : kLookSetIds)
      if (get(id) != RecipeOf(look, id))
        return "disk ID " + std::to_string(id) + " = " + std::to_string(get(id)) + ", recipe " +
               std::to_string(RecipeOf(look, id));
    return std::string();
  }
};
static std::string Show(const std::set<int>& s) {
  std::string out;
  for (int id : s) out += (out.empty() ? "" : ",") + std::to_string(id);
  return out.empty() ? "(none)" : out;
}
// Any core (the command renders nothing), in Premiere and in After Effects.
static void RunCPULook(EffectMainFn effectMain, bool premiere) {
  const char* host = premiere ? "Premiere" : "After Effects";
  gIsPremiere = premiere;
  LookHost h(effectMain, premiere);
  CHECK(h.get(OD_P_LOOK) == OD_LOOK_NATURAL && h.recipeMismatch(OD_LOOK_NATURAL).empty(),
        "look (%s): a new effect is Natural with Natural's values", host);
  // Each Look from the previous one: exactly the controls that differ are
  // written, to exactly the recipe; Look itself is not rewritten.
  struct Step { int look; std::set<int> written; };
  const Step steps[] = {
      {OD_LOOK_CLEANER, {19}},
      {OD_LOOK_PHONE, {12, 13}},
      {OD_LOOK_PRESERVE, {19, 12, 13, 1}},
      {OD_LOOK_NATURAL, {1}},
      {OD_LOOK_PHONE, {19, 12, 13}},
      {OD_LOOK_CLEANER, {12, 13}},
      {OD_LOOK_NATURAL, {19}},
  };
  int good = 0;
  for (const Step& st : steps) {
    h.set(OD_P_LOOK, st.look);
    PF_Err e = PF_Err_NONE;
    const std::set<int> w = h.send(OD_P_LOOK, &e);
    const std::string m = h.recipeMismatch(st.look);
    const bool ok = e == PF_Err_NONE && w == st.written && m.empty() && h.get(OD_P_LOOK) == st.look;
    CHECK(ok, "look (%s): choosing %s wrote %s, expected %s; %s", host, LookName(st.look), Show(w).c_str(),
          Show(st.written).c_str(), m.empty() ? "recipe held" : m.c_str());
    good += ok;
  }
  // Custom, and an index no Look has: nothing written, nothing changed.
  for (int look : {OD_LOOK_CUSTOM, 9, 0}) {
    h.set(OD_P_LOOK, look);
    const std::set<int> w = h.send(OD_P_LOOK);
    CHECK(w.empty() && h.recipeMismatch(OD_LOOK_NATURAL).empty(), "look (%s): Look item %d wrote %s", host, look,
          Show(w).c_str());
  }
  // A look-set control changed by hand: the matching Look, else Custom; Look
  // written only when it changes.
  h.set(OD_P_LOOK, OD_LOOK_NATURAL);
  h.set(OD_P_LUMA, 50);
  std::set<int> w = h.send(OD_P_LUMA);
  CHECK(w == std::set<int>{65} && h.get(OD_P_LOOK) == OD_LOOK_CUSTOM, "look (%s): Luma 50 gave %s, Look %g", host,
        Show(w).c_str(), h.get(OD_P_LOOK));
  h.set(OD_P_LUMA, 60);
  w = h.send(OD_P_LUMA);
  CHECK(w.empty() && h.get(OD_P_LOOK) == OD_LOOK_CUSTOM, "look (%s): Luma 60 (still Custom) wrote %s", host,
        Show(w).c_str());
  h.set(OD_P_LUMA, 70);
  w = h.send(OD_P_LUMA);
  CHECK(w == std::set<int>{65} && h.get(OD_P_LOOK) == OD_LOOK_PRESERVE, "look (%s): Luma 70 by hand gave %s, Look %g "
        "(expected Preserve detail)", host, Show(w).c_str(), h.get(OD_P_LOOK));
  h.set(OD_P_LUMA, 80);
  w = h.send(OD_P_LUMA);
  CHECK(w == std::set<int>{65} && h.get(OD_P_LOOK) == OD_LOOK_NATURAL, "look (%s): Luma back to 80 gave %s, Look %g",
        host, Show(w).c_str(), h.get(OD_P_LOOK));
  h.set(OD_P_GRAIN_FLOOR, 0);
  w = h.send(OD_P_GRAIN_FLOOR);
  CHECK(w == std::set<int>{65} && h.get(OD_P_LOOK) == OD_LOOK_CLEANER, "look (%s): Minimum grain 0 by hand gave %s, "
        "Look %g (expected Cleaner)", host, Show(w).c_str(), h.get(OD_P_LOOK));
  h.set(OD_P_LUMA_LEVEL_MEDIUM, 150);
  w = h.send(OD_P_LUMA_LEVEL_MEDIUM);
  CHECK(w == std::set<int>{65} && h.get(OD_P_LOOK) == OD_LOOK_CUSTOM, "look (%s): half of Phone gave %s, Look %g",
        host, Show(w).c_str(), h.get(OD_P_LOOK));
  h.set(OD_P_LUMA_LEVEL_COARSE, 150);
  w = h.send(OD_P_LUMA_LEVEL_COARSE);
  CHECK(w == std::set<int>{65} && h.get(OD_P_LOOK) == OD_LOOK_PHONE, "look (%s): all of Phone by hand gave %s, Look %g",
        host, Show(w).c_str(), h.get(OD_P_LOOK));
  // Idempotent: a host that reported the handler's own writes as user
  // changes finds nothing to do.
  w = h.send(OD_P_LUMA_LEVEL_COARSE);
  CHECK(w.empty() && h.get(OD_P_LOOK) == OD_LOOK_PHONE, "look (%s): a repeated change wrote %s", host, Show(w).c_str());
  // Minimum grain applies to (a popup in the look set): Whole picture
  // (previous) by hand is no Look's recipe; back to Finest grain only, the
  // Look again.
  h.set(OD_P_GRAIN_MODE, OD_GRAIN_MODE_WHOLE);
  w = h.send(OD_P_GRAIN_MODE);
  CHECK(w == std::set<int>{65} && h.get(OD_P_LOOK) == OD_LOOK_CUSTOM, "look (%s): Whole picture (previous) by hand "
        "gave %s, Look %g (expected Custom)", host, Show(w).c_str(), h.get(OD_P_LOOK));
  h.set(OD_P_GRAIN_MODE, OD_GRAIN_MODE_FINEST);
  w = h.send(OD_P_GRAIN_MODE);
  CHECK(w == std::set<int>{65} && h.get(OD_P_LOOK) == OD_LOOK_PHONE, "look (%s): Finest grain only again gave %s, "
        "Look %g (expected Phone)", host, Show(w).c_str(), h.get(OD_P_LOOK));
  // A Look chosen over Whole picture (previous) writes the popup back, and
  // only it among the unchanged controls.
  h.set(OD_P_GRAIN_MODE, OD_GRAIN_MODE_WHOLE);
  h.send(OD_P_GRAIN_MODE);
  h.set(OD_P_LOOK, OD_LOOK_PHONE);
  w = h.send(OD_P_LOOK);
  CHECK(w == std::set<int>{68} && h.recipeMismatch(OD_LOOK_PHONE).empty() &&
        h.get(OD_P_GRAIN_MODE) == OD_GRAIN_MODE_FINEST, "look (%s): Phone over Whole picture wrote %s", host,
        Show(w).c_str());
  // Controls outside the look set never touch Look, whatever their value.
  for (int id : {OD_P_SHARPEN, OD_P_QUALITY, OD_P_VIEW, OD_P_HOLD_PROFILE, OD_P_NOISE_LEVEL, OD_P_SPATIAL,
                 OD_P_NOISE_Y, OD_P_TEMPORAL_RADIUS, OD_P_GRAIN_STABILITY, OD_P_COARSE_CHROMA}) {
    if (h.def(id).param_type == PF_Param_FLOAT_SLIDER) h.set(id, 33);
    w = h.send(id);
    CHECK(w.empty() && h.get(OD_P_LOOK) == OD_LOOK_PHONE, "look (%s): a change of disk ID %d (outside the look set) "
          "wrote %s", host, id, Show(w).c_str());
  }
  // Tolerance: the UI shows tenths.
  h.set(OD_P_LOOK, OD_LOOK_NATURAL);
  h.send(OD_P_LOOK);
  h.set(OD_P_LUMA, 80.004);
  w = h.send(OD_P_LUMA);
  CHECK(w.empty() && h.get(OD_P_LOOK) == OD_LOOK_NATURAL, "look (%s): Luma 80.004 is still Natural (%s)", host,
        Show(w).c_str());
  h.set(OD_P_LUMA, 80.02);
  w = h.send(OD_P_LUMA);
  CHECK(w == std::set<int>{65} && h.get(OD_P_LOOK) == OD_LOOK_CUSTOM, "look (%s): Luma 80.02 is Custom (%s)", host,
        Show(w).c_str());
  // From any state, each Look writes its whole recipe, exactly, and marks
  // exactly the controls that differed.
  unsigned seed = 77;
  for (int look = OD_LOOK_NATURAL; look <= OD_LOOK_PRESERVE; ++look) {
    std::set<int> differ;
    for (int id : kLookSetIds) {
      // A value inside the control's range: strengths 0..100 %, Minimum
      // grain 0..1000 %, noise levels 25..400 % (tenths).
      seed = seed * 1664525u + 1013904223u;
      const double u = (double)(seed >> 8) / 16777216.0;  // [0, 1)
      const bool level = (id >= 11 && id <= 14) || (id >= 23 && id <= 26);
      const double v = id == OD_P_GRAIN_MODE    ? (u < 0.5 ? OD_GRAIN_MODE_FINEST : OD_GRAIN_MODE_WHOLE)
                       : id == OD_P_GRAIN_FLOOR ? std::round(u * 10000) / 10
                       : level                  ? 25 + std::round(u * 3750) / 10
                                                : std::round(u * 1000) / 10;
      h.set(id, v);
      if (h.get(id) != RecipeOf(look, id)) differ.insert(id);
    }
    h.set(OD_P_LOOK, look);
    w = h.send(OD_P_LOOK);
    const std::string m = h.recipeMismatch(look);
    CHECK(w == differ && m.empty(), "look (%s): %s from a random state wrote %s (expected %s); %s", host,
          LookName(look), Show(w).c_str(), Show(differ).c_str(), m.empty() ? "recipe held" : m.c_str());
  }
  // A host without the extra structure, or without params: nothing happens.
  PF_Err e = effectMain(PF_Cmd_USER_CHANGED_PARAM, &h.c.in_data, &h.c.out_data, h.c.ptrs.data(), nullptr, nullptr);
  CHECK(e == PF_Err_NONE, "look (%s): no extra structure: err %d", host, (int)e);
  PF_UserChangedParamExtra extra = {};
  extra.param_index = PfOf(OD_P_LOOK);
  e = effectMain(PF_Cmd_USER_CHANGED_PARAM, &h.c.in_data, &h.c.out_data, nullptr, nullptr, &extra);
  CHECK(e == PF_Err_NONE, "look (%s): no params: err %d", host, (int)e);
  printf("  look menu (%s): %d of %zu Look changes wrote exactly the recipe; manual edits relabel, Custom writes "
         "nothing\n", host, good, sizeof steps / sizeof steps[0]);
  gIsPremiere = true;
}

// ---- stub core: what the CPU path hands the core -----------------------------
static void RunCPUParams(EffectMainFn effectMain) {
  const HostFmt& f = kBGRA8;
  gIsPremiere = true;
  World in = MakeWorld(33, 17, 4, 0, false, false, 1), out = MakeWorld(33, 17, 4, 0, false, false, 2);
  SetWorldFormat(in.w.data, f.fmt);
  SetWorldFormat(out.w.data, f.fmt);
  gCPU.source = CheckoutSource{in.w, f.fmt, nullptr};
  struct Case { const char* what; int pf; double ui; const char* field; double want; };
  const Case cases[] = {
      {"defaults", 0, 0, "luma", 0.8},
      {"luma 37", OD_P_LUMA, 37, "luma", 0.37},
      {"chroma 81", OD_P_CHROMA, 81, "chroma", 0.81},
      {"temporal strength 64", OD_P_TEMPORAL_STRENGTH, 64, "tstr", 0.64},
      {"spatial 23", OD_P_SPATIAL, 23, "spatial", 0.23},
      {"sharpen 12", OD_P_SHARPEN, 12, "sharpen", 0.12},
      {"noise level 150", OD_P_NOISE_LEVEL, 150, "noise", 1.5},
      {"radius 3", OD_P_TEMPORAL_RADIUS, 3, "radius", 3},
      {"view Noise removed", OD_P_VIEW, 2, "view", OD_VIEW_NOISE},
      {"quality Auto", OD_P_QUALITY, OD_QUALITY_AUTO, "quality", 1},  // software path: Auto = full
      {"quality Fast", OD_P_QUALITY, OD_QUALITY_FAST, "quality", 0},
      {"quality Full", OD_P_QUALITY, OD_QUALITY_FULL, "quality", 1},
  };
  Take();
  int ok = 0;
  for (const Case& c : cases) {
    CPUCall call;
    std::map<int, double> ui;
    if (c.pf) ui[c.pf] = c.ui;
    PrepareCPU(call, true, in.w, 30 * 1001, 0, ui);
    CallRender(effectMain, call, out.w);
    std::vector<Rec> recs = Take();
    const Rec* r = LastOf(recs, "render");
    const bool good = r && std::fabs(r->num(c.field) - c.want) < 1e-5;
    CHECK(good, "cpu params %s: core %s = %s, expected %g", c.what, c.field, r ? (*r)[c.field].c_str() : "(no render)",
          c.want);
    if (r && !c.pf) {
      // Every parameter at its default (what the host fills in for a project
      // saved before the advanced controls existed): the core's neutral values.
      const std::string m = AdvancedMismatch(*r);
      CHECK(m.empty(), "cpu params defaults: advanced field %s", m.c_str());
      CHECK((*r)["off"] == "0,-1,1,-2,2", "cpu params: offsets %s, expected 0,-1,1,-2,2", (*r)["off"].c_str());
      CHECK((*r)["keys"] == "none,none,none,none,none", "cpu params: frame keys %s", (*r)["keys"].c_str());
      CHECK(r->num("wait") == 1 && r->num("manual") == 0 && r->num("luma") == 0.8 && r->num("chroma") == 1 &&
                r->num("radius") == 2 && r->num("tstr") == 1 && r->num("spatial") == 0.5 && r->num("sharpen") == 0 &&
                r->num("noise") == 1 && r->num("view") == 0,
            "cpu params: defaults reach the core unchanged");
    }
    if (r && c.pf == OD_P_TEMPORAL_RADIUS)
      CHECK((*r)["off"] == "0,-1,1,-2,2,-3,3", "cpu params radius 3: offsets %s", (*r)["off"].c_str());
    ok += good;
  }
  printf("  cpu stub params: %d of %zu UI values reach the core as expected\n", ok, sizeof cases / sizeof cases[0]);
  // Each advanced control alone, inside its range and beyond each end.
  {
    int good = 0, n = 0;
    for (const AdvCase& a : AdvancedCases()) {
      CPUCall call;
      PrepareCPU(call, true, in.w, 30 * 1001, 0, {{a.pf, a.ui}});
      CallRender(effectMain, call, out.w);
      std::vector<Rec> recs = Take();
      const Rec* r = LastOf(recs, "render");
      const std::string m = r ? AdvancedMismatch(*r, {{a.field, a.want}}) : std::string("no render");
      CHECK(m.empty(), "cpu advanced pf %d = %g: %s", a.pf, a.ui, m.c_str());
      good += m.empty();
      ++n;
    }
    printf("  cpu stub params: %d of %d advanced values reach the core as expected (clamped beyond the range)\n", good,
           n);
    // All at once, each at a value of its own.
    std::map<int, double> ui;
    for (const Sentinel& sn : Sentinels()) ui[sn.pf] = sn.ui;
    CPUCall call;
    PrepareCPU(call, true, in.w, 30 * 1001, 0, ui);
    CallRender(effectMain, call, out.w);
    std::vector<Rec> recs = Take();
    const Rec* r = LastOf(recs, "render");
    const std::string m = r ? AdvancedMismatch(*r, SentinelWant()) : std::string("no render");
    CHECK(m.empty(), "cpu advanced, every control at a value of its own: %s", m.c_str());
    if (r)
      for (const auto& kv : AdvancedDefaults()) gCpuSentinelFields[kv.first] = (*r)[kv.first.c_str()];
    // An old project: every older control at its sentinel, the experimental
    // ones as the host fills them (their defaults).
    std::map<int, double> old;
    for (const Sentinel& sn : Sentinels())
      if (!IsExperimental(sn.pf) && !After584(sn.pf)) old[sn.pf] = sn.ui;
    CPUCall oc;
    PrepareCPU(oc, true, in.w, 30 * 1001, 0, old, /*oldProject*/ true);
    CallRender(effectMain, oc, out.w);
    recs = Take();
    r = LastOf(recs, "render");
    const std::string mo = r ? AdvancedMismatch(*r, OldProjectWant()) : std::string("no render");
    CHECK(mo.empty(), "cpu advanced, a project saved before the Experimental topic: %s", mo.c_str());
    printf("  cpu stub params: every control at a value of its own, and an old project (Experimental off)\n");
    // A project saved by the legacy build: every one of its 51 controls at a value of
    // its own, the nine original ones included, wherever Effect Controls
    // shows them now; Look, which it did not have, at its old-project value.
    std::map<int, double> ui584;
    for (const Sentinel& sn : Sentinels())
      if (!After584(sn.pf)) ui584[sn.pf] = sn.ui;
    for (const MainSentinel& ms : kMainSentinels) ui584[ms.id] = ms.ui;
    CPUCall c584;
    PrepareCPU(c584, true, in.w, 30 * 1001, 0, ui584, /*oldProject*/ true);
    CHECK(c584.defs[(size_t)PfOf(OD_P_LOOK)].u.pd.value == OD_LOOK_CUSTOM,
          "cpu: a project saved by 5584635 gets Look %d, expected Custom (its registered old-project value)",
          (int)c584.defs[(size_t)PfOf(OD_P_LOOK)].u.pd.value);
    CHECK(c584.defs[(size_t)PfOf(OD_P_GRAIN_MODE)].u.pd.value == OD_GRAIN_MODE_WHOLE,
          "cpu: a project saved by 5584635 gets Minimum grain applies to %d, expected Whole picture (previous)",
          (int)c584.defs[(size_t)PfOf(OD_P_GRAIN_MODE)].u.pd.value);
    CallRender(effectMain, c584, out.w);
    recs = Take();
    r = LastOf(recs, "render");
    const std::string m584 = r ? AdvancedMismatch(*r, Old584Want()) + MainMismatch(*r) : std::string("no render");
    CHECK(m584.empty(), "cpu: a project saved by 5584635, every control at a value of its own: %s", m584.c_str());
    if (r) {
      for (const auto& kv : AdvancedDefaults()) gCpuOld584Fields[kv.first] = (*r)[kv.first.c_str()];
      for (const MainSentinel& ms : kMainSentinels) gCpuOld584Fields[ms.field] = (*r)[ms.field];
    }
    printf("  cpu stub params: a project saved by 5584635 (51 controls at values of their own): %s\n",
           m584.empty() ? "every value reached the core from its own control" : m584.c_str());
  }
  // Each Look chosen through the menu (PF_Cmd_USER_CHANGED_PARAM), then
  // rendered: the values the menu wrote reach the core, and only the
  // recipe's fields move off their defaults.
  {
    int good = 0;
    for (int look = OD_LOOK_NATURAL; look <= OD_LOOK_PRESERVE; ++look) {
      LookHost h(effectMain, true);
      h.set(OD_P_LOOK, look);
      h.send(OD_P_LOOK);
      std::map<int, double> ui;
      for (int id : kLookSetIds) ui[id] = h.get(id);
      CPUCall call;
      PrepareCPU(call, true, in.w, 30 * 1001, 0, ui);
      CallRender(effectMain, call, out.w);
      std::vector<Rec> recs = Take();
      const Rec* r = LastOf(recs, "render");
      const std::string m = r ? AdvancedMismatch(*r, RecipeWant(look)) : std::string("no render");
      const bool amounts =
          r && std::fabs(r->num("luma") - LookLuma(look)) < 1e-6 && std::fabs(r->num("chroma") - 1) < 1e-6;
      CHECK(m.empty() && amounts, "cpu: Look %s rendered: %s%s", LookName(look), m.c_str(),
            amounts ? "" : " (Luma or Colour strength moved)");
      good += m.empty() && amounts;
    }
    printf("  cpu stub params: %d of 4 Looks chosen through the menu reach the core as their recipes\n", good);
  }

  // Sessions: one per Premiere filter instance; a reset session for every
  // render where there is no instance ID (After Effects, or Premiere failing).
  auto renderWith = [&](bool premiere, intptr_t ref) {
    gIsPremiere = premiere;
    CPUCall call;
    PrepareCPU(call, premiere, in.w, 30 * 1001, ref);
    CallRender(effectMain, call, out.w);
    std::vector<Rec> recs = Take();
    const Rec* r = LastOf(recs, "render");
    std::string s = r ? (*r)["s"] : "";
    bool reset = false;
    for (size_t i = 0; i + 1 < recs.size(); ++i)
      if (recs[i].kind == "reset" && recs[i]["s"] == s && recs[i + 1].kind == "render") reset = true;
    return std::make_pair(s, reset);
  };
  auto p1 = renderWith(true, 701), p2 = renderWith(true, 702), p1b = renderWith(true, 701);
  CHECK(p1.first != p2.first && p1.first == p1b.first && !p1.second && !p1b.second,
        "cpu sessions: one per Premiere instance ID, never reset");
  auto ae1 = renderWith(false, 0), ae2 = renderWith(false, 0), none = renderWith(true, kNoInstanceRef);
  CHECK(ae1.second && ae2.second && none.second, "cpu sessions: After Effects and Premiere without an instance ID "
        "must start every render from a reset session");
  gIsPremiere = true;
}

// ---- failure paths on the CPU path ------------------- A render returns success only with
// a valid frame in the output: written by the core, or the unprocessed input (copied, or
// converted between two formats the plug-in knows). Otherwise an error. Before, an exception
// or an unmatched format returned success with an output not written, or a byte copy across
// formats.
static void RunCPUFailurePaths(EffectMainFn effectMain) {
  gIsPremiere = true;
  const int W = 37, H = 21;
  auto fresh = [&](World& out, PrPixelFormat f, int bpp) {
    out = MakeWorld(W, H, bpp, 0, false, false, 501);
    memset(out.store.data(), 0x5A, out.store.size());
    SetWorldFormat(out.w.data, f);
  };
  World in8 = MakeWorld(W, H, 4, 0, false, false, 500);
  SetWorldFormat(in8.w.data, PrPixelFormat_BGRA_4444_8u);
  gCPU.source = CheckoutSource{in8.w, PrPixelFormat_BGRA_4444_8u, nullptr};
  int ok = 0, n = 0;
  auto render = [&](const PF_EffectWorld& in, World& out) {
    CPUCall call;
    PrepareCPU(call, true, in, 30 * 1001, 0);
    return CallRender(effectMain, call, out.w);
  };

  // (a) A checkout throws (the render had not written anything): the
  // input, passed through.
  {
    World out;
    fresh(out, PrPixelFormat_BGRA_4444_8u, 4);
    gCPU.throwInCheckout = true;
    const PF_Err e = render(in8.w, out);
    gCPU.throwInCheckout = false;
    const bool good = e == PF_Err_NONE && out.store == in8.store;
    CHECK(good, "cpu failure: exception in a checkout: err %d, output %s the input", (int)e,
          out.store == in8.store ? "is" : "is not");
    ok += good; ++n;
    CHECK(gCPU.checkins == gCPU.checkouts, "cpu failure: %d checkins for %d checkouts after the exception",
          gCPU.checkins, gCPU.checkouts);
  }
  // (b) The pixel format cannot be read (it throws): nothing can be
  // verified, so no copy: an error, and the output untouched.
  {
    World out;
    fresh(out, PrPixelFormat_BGRA_4444_8u, 4);
    const std::vector<char> before = out.store;
    gCPU.throwInGetFormat = true;
    const PF_Err e = render(in8.w, out);
    gCPU.throwInGetFormat = false;
    const bool good = e != PF_Err_NONE && out.store == before;
    CHECK(good, "cpu failure: pixel format unreadable: err %d (expected an error), output %s", (int)e,
          out.store == before ? "untouched" : "written");
    ok += good; ++n;
  }
  // (c) Input 8u, output 32f (both formats the plug-in converts): the input,
  // converted, exactly v / 255.
  {
    World out;
    fresh(out, PrPixelFormat_BGRA_4444_32f, 16);
    const PF_Err e = render(in8.w, out);
    size_t bad = 0;
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W * 4; ++x)
        bad += ((const float*)RowOf(out.w, y))[x] != ((const uint8_t*)RowOf(in8.w, y))[x] * (1.f / 255.f);
    const bool good = e == PF_Err_NONE && bad == 0;
    CHECK(good, "cpu failure: 8u input into a 32f output: err %d, %zu values differ from the input converted", (int)e,
          bad);
    ok += good; ++n;
  }
  // (d) A format the plug-in does not know, different from the output's: an
  // error, no byte copy.
  {
    World inV = MakeWorld(W, H, 4, 0, false, false, 502), out;
    SetWorldFormat(inV.w.data, PrPixelFormat_VUYA_4444_8u);
    fresh(out, PrPixelFormat_BGRA_4444_8u, 4);
    const std::vector<char> before = out.store;
    const PF_Err e = render(inV.w, out);
    const bool good = e != PF_Err_NONE && out.store == before;
    CHECK(good, "cpu failure: VUYA input into a BGRA output: err %d (expected an error), output %s", (int)e,
          out.store == before ? "untouched" : "written");
    ok += good; ++n;
  }
  // (e) An input smaller than the output (same format): the input where it
  // is, transparent black around it; nothing left as it was.
  {
    World small = MakeWorld(W - 7, H - 5, 4, 0, false, false, 503), out;
    SetWorldFormat(small.w.data, PrPixelFormat_BGRA_4444_8u);
    fresh(out, PrPixelFormat_BGRA_4444_8u, 4);
    const PF_Err e = render(small.w, out);
    size_t wrong = 0;
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W * 4; ++x) {
        const uint8_t got = ((const uint8_t*)RowOf(out.w, y))[x];
        const uint8_t want = (y < H - 5 && x < (W - 7) * 4) ? ((const uint8_t*)RowOf(small.w, y))[x] : 0;
        wrong += got != want;
      }
    const bool good = e == PF_Err_NONE && wrong == 0;
    CHECK(good, "cpu failure: smaller input: err %d, %zu output values neither the input nor transparent black",
          (int)e, wrong);
    ok += good; ++n;
  }
  // (f) No input pixels: an error.
  {
    World out;
    fresh(out, PrPixelFormat_BGRA_4444_8u, 4);
    PF_EffectWorld none = in8.w;
    none.data = nullptr;
    const PF_Err e = render(none, out);
    CHECK(e != PF_Err_NONE, "cpu failure: no input pixels returned success");
    ok += e != PF_Err_NONE; ++n;
  }
  printf("  cpu failure paths: %d of %d as expected (pass-through only when valid, else an error)\n", ok, n);
}

// ---- session identity on the CPU path ---------------- Pasted copies of an effect share
// Premiere's filter instance ID. A session is shared only by renders of the same instance ID
// on the same clip (timeline, track item start, clip start, duration, speed); without the
// clip, every render starts from a reset session. Before, the ID alone chose the session.
static void RunCPUIdentity(EffectMainFn effectMain) {
  gIsPremiere = true;
  World in = MakeWorld(33, 17, 4, 0, false, false, 1), out = MakeWorld(33, 17, 4, 0, false, false, 2);
  SetWorldFormat(in.w.data, PrPixelFormat_BGRA_4444_8u);
  SetWorldFormat(out.w.data, PrPixelFormat_BGRA_4444_8u);
  gCPU.source = CheckoutSource{in.w, PrPixelFormat_BGRA_4444_8u, nullptr};
  const intptr_t a = 951, b = 952;
  gCPU.instanceIdOf[(PF_ProgPtr)a] = gCPU.instanceIdOf[(PF_ProgPtr)b] = 950;  // one ID for both copies
  auto renderWith = [&](intptr_t ref) {
    CPUCall call;
    PrepareCPU(call, true, in.w, 30 * 1001, ref);
    CallRender(effectMain, call, out.w);
    std::vector<Rec> recs = Take();
    const Rec* r = LastOf(recs, "render");
    std::string s = r ? (*r)["s"] : "";
    bool reset = false;
    for (size_t i = 0; i + 1 < recs.size(); ++i)
      if (recs[i].kind == "reset" && recs[i]["s"] == s && recs[i + 1].kind == "render") reset = true;
    return std::make_pair(s, reset);
  };
  Take();
  // Two clips (default: a track item start per ref).
  auto a1 = renderWith(a), b1 = renderWith(b), a2 = renderWith(a);
  CHECK(!a1.first.empty() && a1.first != b1.first && a1.first == a2.first && !a1.second && !b1.second && !a2.second,
        "cpu identity: one instance ID on two clips must give two sessions, each kept (%s %s %s)", a1.first.c_str(),
        b1.first.c_str(), a2.first.c_str());
  // The same clip (same timeline, start, in point, duration, speed): one session.
  gCPU.ownerOf[(PF_ProgPtr)a] = gCPU.ownerOf[(PF_ProgPtr)b] = CPUHost::Owner{7, 5000, 10, 300, 1.0};
  auto a3 = renderWith(a), b3 = renderWith(b);
  CHECK(!a3.first.empty() && a3.first == b3.first && !b3.second, "cpu identity: the same instance on the same clip "
        "keeps its session (%s %s)", a3.first.c_str(), b3.first.c_str());
  // A slip (another clip start) is another clip.
  gCPU.ownerOf[(PF_ProgPtr)b] = CPUHost::Owner{7, 5000, 60, 300, 1.0};
  auto b4 = renderWith(b);
  CHECK(b4.first != a3.first, "cpu identity: another clip start must give another session");
  gCPU.ownerOf.clear();
  // No clip information: every render from a reset session.
  gCPU.noOwnerInfo = true;
  auto a5 = renderWith(a), b5 = renderWith(b);
  gCPU.noOwnerInfo = false;
  CHECK(a5.second && b5.second, "cpu identity: without clip information every render must start from a reset "
        "session (%d %d)", (int)a5.second, (int)b5.second);
  gCPU.instanceIdOf.clear();
  printf("  cpu identity: one ID on two clips -> two sessions; same clip -> one; no clip information -> reset\n");
}

static void TestCPU(EffectMainFn effectMain) {
  printf("EffectMain\n");
  gIsPremiere = true;
  PF_InData in_data = {};
  PF_OutData out_data = {};
  in_data.appl_id = kAppID_Premiere;
  in_data.pica_basicP = &gBasic;
  in_data.inter.add_param = FakeAddParam;
  in_data.inter.register_ui = RegisterCustomUI;

  PF_Err err = effectMain(PF_Cmd_GLOBAL_SETUP, &in_data, &out_data, nullptr, nullptr, nullptr);
  CHECK(err == PF_Err_NONE, "global setup err %d", (int)err);
  CHECK(out_data.out_flags == OD_OUT_FLAGS_VALUE, "out_flags 0x%x", (unsigned)out_data.out_flags);
  CHECK(out_data.out_flags2 == OD_OUT_FLAGS2_VALUE, "out_flags2 0x%x", (unsigned)out_data.out_flags2);
  CHECK(!(out_data.out_flags & PF_OutFlag_CUSTOM_UI), "GlobalSetup does not advertise custom UI");
  // Any group meant to start open (no START_COLLAPSED in kExpect) only does
  // if the effect sets
  // PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG; without it every group
  // starts collapsed whatever its own flag says (AE_Effect.h).
  {
    const char* open = nullptr;
    for (const Expect& e : kExpect)
      if (e.type == kGS && !(e.flags & kCol)) open = e.name;
    CHECK(!open || (out_data.out_flags2 & PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG),
          "group \"%s\" is meant to start open but out_flags2 0x%x lacks PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG "
          "(every group would start collapsed)", open ? open : "", (unsigned)out_data.out_flags2);
  }
  CHECK((A_u_long)out_data.my_version == OD_PIPL_VERSION_VALUE, "my_version 0x%x", (unsigned)out_data.my_version);
  CHECK(gCPU.clearCalls == 1 && gCPU.formats.size() == 3, "pixel formats: clear=%d n=%zu", gCPU.clearCalls, gCPU.formats.size());
  if (gCPU.formats.size() == 3)
    CHECK(gCPU.formats[0] == PrPixelFormat_BGRA_4444_32f && gCPU.formats[1] == PrPixelFormat_BGRA_4444_16u &&
              gCPU.formats[2] == PrPixelFormat_BGRA_4444_8u, "pixel format order");
  CHECK(gCPU.wantsMatchCalls == 1, "EffectWantsCheckedOutFramesToMatchRenderPixelFormat calls %d", gCPU.wantsMatchCalls);
  printf("  global setup: out_flags=0x%08x out_flags2=0x%08x version=0x%x formats=%zu\n", (unsigned)out_data.out_flags,
         (unsigned)out_data.out_flags2, (unsigned)out_data.my_version, gCPU.formats.size());

  out_data = {};
  err = effectMain(PF_Cmd_PARAMS_SETUP, &in_data, &out_data, nullptr, nullptr, nullptr);
  CHECK(err == PF_Err_NONE, "params setup err %d", (int)err);
  CHECK(gRegisteredUI == 0, "no custom UI registered at parameter setup");
  CHECK(out_data.num_params == kNumExpect + 1, "num_params %d", (int)out_data.num_params);
  // The whole parameter table (kExpect, written out above, not derived from
  // the plug-in header): name, type, disk ID, default, valid range, flags, in
  // Effect Controls order.
  const Expect* expect = kExpect.data();
  const size_t kExpected = (size_t)kNumExpect;
  CHECK(kExpected == (size_t)kCountTopics && out_data.num_params == kCountTopics + 1, "the table has %zu entries, num_params %d: expected current layout and input",
        kExpected, (int)out_data.num_params);
  CHECK(gCPU.added.size() == kExpected, "%zu params added, expected %zu", gCPU.added.size(), kExpected);
  int depth = 0, topics = 0;
  bool nested = false;
  std::vector<std::string> names;
  for (size_t i = 0; i < gCPU.added.size() && i < kExpected; ++i) {
    const PF_ParamDef& d = gCPU.added[i];
    const Expect& e = expect[i];
    CHECK(!strcmp(d.PF_DEF_NAME, e.name), "param %zu name '%s', expected '%s'", i + 1, d.PF_DEF_NAME, e.name);
    CHECK(d.param_type == e.type, "param %zu type %d, expected %d", i + 1, (int)d.param_type, e.type);
    CHECK(d.uu.id == e.id, "param %zu disk id %d, expected %d", i + 1, (int)d.uu.id, e.id);
    CHECK((int)d.flags == e.flags, "param %zu (%s) flags 0x%x, expected 0x%x", i + 1, e.name, (unsigned)d.flags,
          (unsigned)e.flags);
    double dflt = 0, lo = 0, hi = 0;
    if (d.param_type == PF_Param_FLOAT_SLIDER) {
      dflt = d.u.fs_d.dephault; lo = d.u.fs_d.valid_min; hi = d.u.fs_d.valid_max;
      CHECK(d.u.fs_d.value == d.u.fs_d.dephault, "param %zu (%s): old-project value %g differs from the default %g",
            i + 1, e.name, d.u.fs_d.value, d.u.fs_d.dephault);
      const double dragLo = e.id == 78 ? -12 : e.id == 80 ? -12 : e.lo;
      const double dragHi = e.id == 77 ? 3 : e.id == 80 ? 12 : e.hi;
      CHECK(d.u.fs_d.slider_min == float(dragLo) && d.u.fs_d.slider_max == float(dragHi),
            "param %zu (%s): slider range differs from the valid range", i + 1, e.name);
      CHECK(d.u.fs_d.precision == (e.id == 77 ? PF_Precision_HUNDREDTHS : PF_Precision_TENTHS) &&
            bool(d.u.fs_d.display_flags & PF_ValueDisplayFlag_PERCENT) == (e.id < 71),
            "param %zu (%s): not a percent slider in tenths", i + 1, e.name);
    } else if (d.param_type == PF_Param_SLIDER) { dflt = d.u.sd.dephault; lo = d.u.sd.valid_min; hi = d.u.sd.valid_max; }
    else if (d.param_type == PF_Param_POPUP) {
      dflt = d.u.pd.dephault; lo = 1; hi = d.u.pd.num_choices;
      // Look: a project saved before it existed gets Custom (5), a new or
      // reset effect Natural (1). Minimum grain applies to: Whole picture
      // (previous) (2) for an old project, Finest grain only (1) for a new
      // effect. Every other popup: one value.
      const A_long oldValue = d.uu.id == 65 ? 5 : d.uu.id == 68 ? 2 : (d.uu.id == 71 || d.uu.id == 106) ? 1 : d.u.pd.dephault;
      CHECK(d.u.pd.value == oldValue, "param %zu (%s): old-project value %d, expected %d", i + 1, e.name,
            (int)d.u.pd.value, (int)oldValue);
    }
    else if (d.param_type == PF_Param_CHECKBOX) { dflt = d.u.bd.dephault; lo = 0; hi = 1; }
    CHECK(dflt == (e.type == kFS ? double(float(e.dflt)) : e.dflt) && lo == (e.type == kFS ? double(float(e.lo)) : e.lo) && hi == (e.type == kFS ? double(float(e.hi)) : e.hi), "param %zu (%s) default/range %g [%g,%g]", i + 1, e.name, dflt, lo,
          hi);
    if (d.param_type == PF_Param_GROUP_START) {
      nested = nested || depth != 0;
      ++depth;
      ++topics;
    } else if (d.param_type == PF_Param_GROUP_END) {
      nested = nested || depth != 1;
      --depth;
    }
    if (d.PF_DEF_NAME[0]) {
      CHECK(strlen(d.PF_DEF_NAME) <= PF_MAX_EFFECT_PARAM_NAME_LEN, "param %zu name too long", i + 1);
      CHECK(e.id == 65 || e.id == 72 || e.id == 6 || std::find(names.begin(), names.end(), std::string(d.PF_DEF_NAME)) == names.end(),
            "param %zu name '%s' is not unique", i + 1, d.PF_DEF_NAME);
      names.push_back(d.PF_DEF_NAME);
    }
    if (i < 10 || d.param_type == PF_Param_GROUP_START)
      printf("  param %2zu: %-24s id=%d default=%g range=[%g,%g]\n", i + 1, d.PF_DEF_NAME, (int)d.uu.id, dflt, lo, hi);
  }
  // The two checkbox strings are different host contracts: Premiere displays
  // NAME_B (including scripting); AE keeps NAME_A without a duplicate label.
  for (const PF_ParamDef& d : gCPU.added)
    if (d.param_type == PF_Param_CHECKBOX)
      CHECK(d.u.bd.u.PF_DEF_NAMEPTR && !strcmp(d.u.bd.u.PF_DEF_NAMEPTR, d.PF_DEF_NAME),
            "Premiere checkbox %d has an empty or wrong value label", (int)d.uu.id);
  const auto premiereParams = gCPU.added;
  gCPU.added.clear();
  in_data.appl_id = 'FXTC';
  err = effectMain(PF_Cmd_PARAMS_SETUP, &in_data, &out_data, nullptr, nullptr, nullptr);
  CHECK(err == PF_Err_NONE && gCPU.added.size() == premiereParams.size(), "AE params setup");
  for (size_t i = 0; i < gCPU.added.size() && i < premiereParams.size(); ++i) {
    const auto& d = gCPU.added[i];
    if (d.param_type != PF_Param_CHECKBOX) continue;
    CHECK(d.u.bd.u.PF_DEF_NAMEPTR && !d.u.bd.u.PF_DEF_NAMEPTR[0], "AE checkbox label duplicated");
    CHECK(!strcmp(d.PF_DEF_NAME, premiereParams[i].PF_DEF_NAME) && d.uu.id == premiereParams[i].uu.id &&
          d.u.bd.dephault == premiereParams[i].u.bd.dephault, "checkbox name/id/default changed across hosts");
  }
  gCPU.added = premiereParams;
  in_data.appl_id = kAppID_Premiere;
  CHECK(topics == (odplug::kUseDynamicVisibility ? 7 : 8) && depth == 0 && !nested, "topics: %d, depth at the end %d, nested %d", topics, depth, (int)nested);
  // Disk IDs: each once; 1..51 are those of the legacy build, every one still there; 52..64 are the
  // saved profile, all there, as one block in ID order; 65 Look, 66/67 the Expert topic, 68
  // Minimum grain applies to: the whole allocation, 1..68, each once (the design's table).
  {
    std::set<int> seen;
    for (const PF_ParamDef& d : gCPU.added) {
      const int id = (int)d.uu.id;
      CHECK(seen.insert(id).second, "disk ID %d registered twice", id);
      CHECK(id >= 1 && id <= kCountTopics, "disk ID %d is outside the allocation of this build", id);
    }
    for (int id = 1; id <= kCountTopics; ++id) CHECK(seen.count(id) == 1, "disk ID %d is missing", id);
    const int first = PfOf(OD_P_PROFILE_POLICY);
    for (int id = OD_P_PROFILE_POLICY; id <= OD_P_PROFILE_INFO; ++id)
      CHECK(first > 0 && PfOf(id) == first + (id - OD_P_PROFILE_POLICY) && first + (id - OD_P_PROFILE_POLICY) - 1 <
            (int)gCPU.added.size() && gCPU.added[(size_t)(first + (id - OD_P_PROFILE_POLICY) - 1)].uu.id == id,
            "saved-profile disk ID %d is not in its block", id);
  }
  for (int t : kTopics)
    CHECK(t - 1 < (int)gCPU.added.size() && (gCPU.added[(size_t)t - 1].param_type == PF_Param_GROUP_START ||
                                             gCPU.added[(size_t)t - 1].param_type == PF_Param_GROUP_END),
          "PF index %d is not a topic entry", t);
  if (gCPU.added.size() != kExpected) return;
  RunCPULook(effectMain, true);
  RunCPULook(effectMain, false);

  RunCPURender(effectMain, "Premiere BGRA 8u", true, PrPixelFormat_BGRA_4444_8u, 4, false, false, 4);
  RunCPURender(effectMain, "Premiere BGRA 16u", true, PrPixelFormat_BGRA_4444_16u, 8, false, false, 4);
  RunCPURender(effectMain, "Premiere BGRA 32f", true, PrPixelFormat_BGRA_4444_32f, 16, false, false, 4);
  RunCPURender(effectMain, "Premiere BGRA 8u flipped", true, PrPixelFormat_BGRA_4444_8u, 4, false, true, 4);
  // A format we did not ask for must pass through untouched, no checkouts.
  bool real = gReal; gReal = false;
  RunCPURender(effectMain, "Premiere VUYA 8u (foreign)", true, PrPixelFormat_VUYA_4444_8u, 4, false, false, 0);
  gReal = real;
  gCPU.failCheckouts = true;  // host without neighbours: spatial only, still renders
  RunCPURender(effectMain, "Premiere 32f, no neighbours", true, PrPixelFormat_BGRA_4444_32f, 16, false, false, 4);
  gCPU.failCheckouts = false;
  RunCPURender(effectMain, "After Effects ARGB 8", false, PrPixelFormat_Invalid, 4, false, false, 4);
  RunCPURender(effectMain, "After Effects ARGB 16", false, PrPixelFormat_Invalid, 8, true, false, 4);

  if (StubPixels()) {
    for (const HostFmt* f : {&kBGRA8, &kBGRA16, &kBGRA32, &kARGB8, &kARGB16}) {
      RunCPUFill(effectMain, *f, false);
      RunCPUCheck(effectMain, *f, false);
    }
    RunCPUFill(effectMain, kBGRA8, true);
    RunCPUFill(effectMain, kBGRA16, true);
    RunCPUCheck(effectMain, kBGRA16, true);
  }
  if (!gReal) RunCPUParams(effectMain);
  if (!gReal) RunCPUIdentity(effectMain);
  RunCPUFailurePaths(effectMain);
  if (gReal) {
    unsigned seed = 300;
    for (const HostFmt* f : {&kBGRA8, &kBGRA16, &kBGRA32, &kARGB8, &kARGB16}) RunCPURealFormat(effectMain, *f, seed += 10);
    // After Effects has no instance ID; Premiere may fail to give one.
    RunCPULayers(effectMain, kARGB16, "After Effects ARGB 16", 0, 0, 0, 0, true);
    RunCPULayers(effectMain, kBGRA16, "Premiere 16u, no instance ID", kNoInstanceRef, kNoInstanceRef, kNoInstanceRef,
                 kNoInstanceRef, true);
    RunCPULayers(effectMain, kBGRA16, "Premiere 16u, one ID per layer", 801, 802, 803, 804, true);
    // Pasted copies: one instance ID for both layers. With each layer's clip known, or with
    // no clip information (every render from a reset session), the layers stay apart.
    gCPU.instanceIdOf[(PF_ProgPtr)813] = gCPU.instanceIdOf[(PF_ProgPtr)814] = 810;
    RunCPULayers(effectMain, kBGRA16, "Premiere 16u, one ID, two clips", 811, 812, 813, 814, true);
    gCPU.noOwnerInfo = true;
    RunCPULayers(effectMain, kBGRA16, "Premiere 16u, one ID, no clip info", 815, 816, 813, 814, true);
    gCPU.noOwnerInfo = false;
    // Control: the host says both layers are the same clip, so they share a
    // session, and the test must see it.
    gCPU.ownerOf[(PF_ProgPtr)813] = gCPU.ownerOf[(PF_ProgPtr)814] = CPUHost::Owner{7, 9000, 0, 300, 1.0};
    RunCPULayers(effectMain, kBGRA16, "Premiere 16u, one ID, one clip", 817, 818, 813, 814, false);
    gCPU.ownerOf.clear();
    gCPU.instanceIdOf.clear();
  }
  RunCPUReentrant(effectMain, false);
  RunCPUReentrant(effectMain, true);
  RunCPUConcurrent(effectMain);
  RunCPUIdleRelease(effectMain);

  in_data = {};
  out_data = {};
  err = effectMain(PF_Cmd_ABOUT, &in_data, &out_data, nullptr, nullptr, nullptr);
  CHECK(err == PF_Err_NONE && !strncmp(out_data.return_msg, "Denoise ", 8), "about: '%s'", out_data.return_msg);
  CHECK(!strcmp(OD_EFFECT_MATCHNAME, "Open Denoise"), "permanent match name changed: '%s'", OD_EFFECT_MATCHNAME);
  err = effectMain(PF_Cmd_GLOBAL_SETDOWN, &in_data, &out_data, nullptr, nullptr, nullptr);
  CHECK(err == PF_Err_NONE, "global setdown err %d", (int)err);
  CHECK(gAcquired == gReleased, "suites acquired %d released %d", gAcquired.load(), gReleased.load());
}

// ============================ GPU tests ======================================

static FakePPix MakePPix(PrPixelFormat fmt, int w, int h, int pad, unsigned seed) {
  FakePPix p;
  int bpp = fmt == PrPixelFormat_GPU_BGRA_4444_32f ? 16 : 8;
  p.format = fmt; p.width = w; p.height = h; p.rowBytes = w * bpp + pad;
  p.buffer = [gGPU.device newBufferWithLength:(NSUInteger)p.rowBytes * h options:MTLResourceStorageModeShared];
  unsigned s = seed * 2246822519u + 1u;
  uint8_t* d = (uint8_t*)p.buffer.contents;
  if (bpp == 16) {
    for (size_t i = 0; i < (size_t)p.rowBytes * h / 4; ++i) { s = s * 1664525u + 1013904223u; ((float*)d)[i] = (s >> 8) / 16777216.f; }
  } else {
    // valid half floats in [0,1): exponent 0x3800 (0.5..1) / small mantissa
    for (size_t i = 0; i < (size_t)p.rowBytes * h / 2; ++i) { s = s * 1664525u + 1013904223u; ((uint16_t*)d)[i] = (uint16_t)(0x3800 | ((s >> 20) & 0x3FF)); }
  }
  return p;
}
static void WaitForGPU() {
  id<MTLCommandBuffer> cb = [gGPU.queue commandBuffer];
  [cb commit];
  [cb waitUntilCompleted];  // same queue: everything queued before has run
}
static bool SameBytes(const FakePPix& a, const FakePPix& b) {
  return a.buffer.length == b.buffer.length && memcmp(a.buffer.contents, b.buffer.contents, a.buffer.length) == 0;
}
static double GreenSigma(const float* px, int w, int h, int rowFloats) {
  double sum = 0, sum2 = 0;
  size_t n = 0;
  for (int y = 16; y < h - 16; ++y)
    for (int x = 16; x < w - 16; ++x) {
      const double v = px[(size_t)y * rowFloats + (size_t)x * 4 + 1];
      sum += v; sum2 += v * v; ++n;
    }
  const double mean = sum / (double)n;
  return std::sqrt(std::max(0.0, sum2 / (double)n - mean * mean));
}

static std::vector<PrTime> QueryDeps(PrGPUFilter& f, PrGPUFilterInstance* inst, const PrGPUFilterRenderParams& rp) {
  std::vector<PrTime> times;
  csSDK_int32 index = 0;
  for (int guard = 0; guard < 32; ++guard) {
    PrGPUFilterFrameDependency dep = {};
    csSDK_int32 before = index;
    prSuiteError e = f.GetFrameDependencies(inst, &rp, &index, &dep);
    if (e != suiteError_NoError) break;
    CHECK(dep.outDependencyType == PrGPUDependency_InputFrame && dep.outTrackID == 0, "dependency type/track");
    times.push_back(dep.outSequenceTime);
    if (index == before) break;  // no more
  }
  return times;
}

static const PrTime kTPF = 254016000000LL * 1001 / 30000;  // 29.97 fps
static const int kEffectNode = 7;
// Segment index of the control with disk ID `id` in the fake host's layout:
// its PF index (the parameter table) less one, and less the topic entries
// before it when the host leaves them out. (The helpers below take disk IDs.)
static int SegIndex(int id) {
  const int pf = PfOf(id);
  if (pf < 1) {
    CHECK(false, "SegIndex: disk ID %d is not a parameter of this build", id);
    return -1000;
  }
  if (gGPU.withTopics) return pf - 1;
  int before = 0;
  for (int p = 1; p < pf; ++p) before += IsTopicAt(p) ? 1 : 0;
  return pf - 1 - before;
}
static void SetF(int pf, double v) { PrParam p = {}; p.mType = kPrParamType_Float64; p.mFloat64 = v; gGPU.params[SegIndex(pf)] = p; }
// Checkboxes: as Bool (what Premiere is expected to send) or as Int32.
static void SetB(int pf, bool v, bool asInt = false) {
  PrParam p = {};
  if (asInt) { p.mType = kPrParamType_Int32; p.mInt32 = v ? 1 : 0; }
  else { p.mType = kPrParamType_Bool; p.mBool = v ? 1 : 0; }
  gGPU.params[SegIndex(pf)] = p;
}
// Popups are given here as the 1-based item the user picked (the CPU path's
// convention); Premiere hands the GPU side the 0-based index (observed in
// Premiere 26.5.2: View "Noise removed" arrived as 1), so the
// fake host stores item - 1, as Premiere does.
static void SetI(int pf, int v) {
  PrParam p = {}; p.mType = kPrParamType_Int32;
  const int at = PfOf(pf);
  p.mInt32 = at >= 1 && kExpect[at - 1].type == kPU ? v - 1 : v;
  gGPU.params[SegIndex(pf)] = p;
}
// Every parameter at its default, in the current layout (the topic slots
// hold nothing).
static void DefaultGPUParams() {
  gGPU.params.clear();
  gGPU.keyed.clear();
  SetI(OD_P_LOOK, OD_LOOK_NATURAL);  // held by the host; the GPU path never reads it
  SetF(OD_P_LUMA, 80); SetF(OD_P_CHROMA, 100); SetI(OD_P_TEMPORAL_RADIUS, 2); SetF(OD_P_TEMPORAL_STRENGTH, 100);
  SetF(OD_P_SPATIAL, 50); SetF(OD_P_SHARPEN, 0); SetF(OD_P_NOISE_LEVEL, 100); SetI(OD_P_QUALITY, 1); SetI(OD_P_VIEW, 1);
  for (int b = 0; b < 4; ++b) {
    SetF(11 + b, 100); SetF(15 + b, 100); SetF(23 + b, 100); SetF(27 + b, 100);
  }
  SetF(19, 100); SetB(20, true); SetF(33, 100); SetF(34, 75);
  SetF(37, 100); SetF(38, 100); SetF(39, 100); SetF(40, 100);
  SetB(43, false); SetF(44, 100); SetF(45, 100); SetF(46, 100);
  SetF(49, 0); SetB(50, false);
  SetI(OD_P_PROFILE_POLICY, OD_PROFILE_AUTOMATIC);
  SetI(OD_P_PROFILE_REVISION, 0);
  SetI(OD_P_CAPTURE_REQUEST_LO,0); SetI(OD_P_CAPTURE_REQUEST_HI,0);
  for (int i = 0; i < 8; ++i) SetI(OD_P_PROFILE_DIGEST0+i, 0);
  SetI(OD_P_GRAIN_MODE, OD_GRAIN_MODE_FINEST);  // a new effect
  for (const auto& e : kExpect) if (e.id >= 71) {
    if (e.type == kFS) SetF(e.id,e.dflt);
    else if (e.type == kPU || e.type == kSL) SetI(e.id,(int)e.dflt);
  }
}
// A sentinel or case value at its disk ID, typed as Premiere hands it.
static void SetGPU(int id, double ui, bool checkbox, bool popup) {
  if (checkbox) SetB(id, ui != 0);
  else if (popup) SetI(id, (int)ui);
  else SetF(id, ui);
}
// The controls added since the legacy build that registered a value for old projects
// (PF_ParamFlag_USE_VALUE_FOR_OLD_PROJECTS), as the host fills them for a
// project that build saved: Look Custom, Profile policy Automatic, Minimum
// grain applies to Whole picture (previous). The others keep their defaults.
static void OldProjectGPU() {
  SetI(71,1); SetI(106,1);
  SetI(OD_P_LOOK, OD_LOOK_CUSTOM);
  SetI(OD_P_PROFILE_POLICY, OD_PROFILE_AUTOMATIC);
  SetI(OD_P_GRAIN_MODE, OD_GRAIN_MODE_WHOLE);
}
// The fake host's GPU parameter numbering, then every parameter at its default.
static void UseLayout(bool withTopics, int count, bool countFails = false) {
  gGPU.withTopics = withTopics;
  gGPU.paramCount = count;
  gGPU.countFails = countFails;
  DefaultGPUParams();
}
// The effect node's properties as Premiere would report them: the track item
// sits at sequence frames [itemStart, itemEnd); effective bounds include
// transition handles; < 0 leaves a property out.
static void SetNode(long long runtimeID, long long itemStart, long long itemEnd, long long effStart, long long effEnd) {
  gGPU.props.clear();
  if (runtimeID >= 0) SetProp(kEffectNode, kVideoSegmentProperty_Effect_RuntimeInstanceID, runtimeID);
  SetProp(kEffectNode, kVideoSegmentProperty_Effect_EffectDuration, (itemEnd - itemStart) * kTPF);
  if (itemStart >= 0) {
    SetProp(kOwnerNode, kVideoSegmentProperty_Clip_TrackItemStartAsTicks, itemStart * kTPF);
    SetProp(kOwnerNode, kVideoSegmentProperty_Clip_TrackItemEndAsTicks, itemEnd * kTPF);
    SetProp(kOwnerNode, kVideoSegmentProperty_Clip_TrackID, 2);
  }
  if (effStart >= 0) {
    SetProp(kOwnerNode, kVideoSegmentProperty_Clip_EffectiveTrackItemStartAsTicks, effStart * kTPF);
    SetProp(kOwnerNode, kVideoSegmentProperty_Clip_EffectiveTrackItemEndAsTicks, effEnd * kTPF);
  }
}

struct GPUEnv {
  PrGPUFilter filter = {};
  piSuites suites = {};
  PlugUtilFuncs util = {};
};
static bool Create(GPUEnv& g, PrGPUFilterInstance& inst, csSDK_int32 timeline = 3) {
  inst = PrGPUFilterInstance();
  inst.piSuites = &g.suites;
  inst.inNodeID = kEffectNode;
  inst.inTimelineID = timeline;
  const int allocated = gGPU.allocatedPtrs, disposed = gGPU.disposedPtrs;
  prSuiteError e = g.filter.CreateInstance(&inst);
  CHECK(gGPU.ownersHeld == 0, "owner node acquired and not released (%d held)", gGPU.ownersHeld);
  CHECK(gGPU.allocatedPtrs - allocated == gGPU.disposedPtrs - disposed, "node property strings: %d allocated, %d freed",
        gGPU.allocatedPtrs - allocated, gGPU.disposedPtrs - disposed);
  return e == suiteError_NoError && inst.ioPrivatePluginData;
}

// Neighbours requested at sequence frame `seqFrame` (clip time `clipFrame`),
// relative to it.
static std::vector<int> DepsAt(GPUEnv& g, PrGPUFilterInstance& inst, long long seqFrame, long long clipFrame) {
  PrGPUFilterRenderParams rp = {};
  rp.inRenderTicksPerFrame = kTPF;
  rp.inDownsampleFactorX = rp.inDownsampleFactorY = 1.f;
  rp.inClipTime = clipFrame * kTPF;
  rp.inSequenceTime = seqFrame * kTPF;
  std::vector<int> rel;
  for (PrTime t : QueryDeps(g.filter, &inst, rp)) rel.push_back((int)((t - rp.inSequenceTime) / kTPF));
  return rel;
}
static std::string Show(const std::vector<int>& v) {
  std::string s;
  for (int x : v) s += (x > 0 ? " +" : " ") + std::to_string(x);
  return v.empty() ? " (none)" : s;
}

// GetFrameDependencies against track item bounds in sequence time. The item
// sits at sequence frames [1000, 1100); clip (media) time is whatever the
// host says and must not matter.
static void TestDependencies(GPUEnv& g) {
  const std::vector<int> all = {-1, 1, -2, 2};
  struct Case {
    const char* label;
    long long effStart, effEnd;  // < 0: no effective bounds
    bool ownerFails, plainBounds;
    long long inPoint;           // media frame at the item's first frame
    double speed;
    std::vector<std::pair<long long, std::vector<int>>> expect;  // item frame -> offsets
  };
  const Case cases[] = {
      {"in-point 3000", 1000, 1100, false, true, 3000, 1.0,
       {{0, {1, 2}}, {1, {-1, 1, 2}}, {50, all}, {98, {-1, 1, -2}}, {99, {-1, -2}}}},
      {"in-point 1 hour", 1000, 1100, false, true, 107892, 1.0, {{0, {1, 2}}, {50, all}, {99, {-1, -2}}}},
      {"200 % speed", 1000, 1100, false, true, 0, 2.0, {{0, {1, 2}}, {51, all}, {99, {-1, -2}}}},
      {"transition handles", 990, 1110, false, true, 500, 1.0,
       {{-10, {1, 2}}, {-5, all}, {0, all}, {99, all}, {104, all}, {109, {-1, -2}}}},
      {"only plain bounds", -1, -1, false, true, 500, 1.0, {{0, {1, 2}}, {99, {-1, -2}}}},
      {"owner lookup fails", 1000, 1100, true, true, 500, 1.0, {{0, all}, {50, all}, {99, all}}},
      {"no bounds properties", -1, -1, false, false, 500, 1.0, {{0, all}, {99, all}}},
  };
  for (const Case& c : cases) {
    SetNode(9100, c.plainBounds ? 1000 : -1, 1100, c.effStart, c.effEnd);
    gGPU.ownerFails = c.ownerFails;
    PrGPUFilterInstance inst;
    if (!Create(g, inst)) { CHECK(false, "deps %s: CreateInstance failed", c.label); continue; }
    for (auto& e : c.expect) {
      const long long itemFrame = e.first;
      const long long clip = c.inPoint + (long long)std::llround(itemFrame * c.speed);
      std::vector<int> d = DepsAt(g, inst, 1000 + itemFrame, clip);
      CHECK(d == e.second, "deps %s, item frame %lld:%s, expected%s", c.label, itemFrame, Show(d).c_str(),
            Show(e.second).c_str());
      printf("  deps %-22s item frame %4lld:%s\n", c.label, itemFrame, Show(d).c_str());
    }
    g.filter.DisposeInstance(&inst);
    gGPU.ownerFails = false;
  }

  // Bounds that do not contain the frame being rendered: the host's time base
  // is not what we expect. Every neighbour, never none.
  gGPU.props.clear();
  SetProp(kEffectNode, kVideoSegmentProperty_Effect_RuntimeInstanceID, 9101);
  SetProp(kOwnerNode, kVideoSegmentProperty_Clip_EffectiveTrackItemStartAsTicks, 1000);  // frames, not ticks
  SetProp(kOwnerNode, kVideoSegmentProperty_Clip_EffectiveTrackItemEndAsTicks, 1100);
  PrGPUFilterInstance inst;
  if (Create(g, inst)) {
    std::vector<int> d = DepsAt(g, inst, 1050, 3050);
    CHECK(d == all, "deps with bounds in another unit:%s, expected every neighbour", Show(d).c_str());
    printf("  deps %-22s item frame   50:%s\n", "bounds in another unit", Show(d).c_str());
    g.filter.DisposeInstance(&inst);
  }
  // Nothing before sequence time 0, bounds or not. First Premiere run:
  // renders at sequence time 0 of an instance whose bounds lay elsewhere
  // (15-20 s) asked for -1 and -2 as well, got 2 frames for 4 and went
  // spatial only.
  {
    struct Zero { const char* label; long long effStart, effEnd; bool ownerFails; long long seq;
                  std::vector<int> want; };
    const Zero zs[] = {
        {"time 0, bounds elsewhere", 450, 600, false, 0, {1, 2}},
        {"time 1, bounds elsewhere", 450, 600, false, 1, {-1, 1, 2}},
        {"time 0, no bounds", 0, 100, true, 0, {1, 2}},
        {"time 2, no bounds", 0, 100, true, 2, all},
        {"time 0, item at 0", 0, 100, false, 0, {1, 2}},
    };
    for (const Zero& z : zs) {
      SetNode(9103, z.effStart, z.effEnd, z.effStart, z.effEnd);
      gGPU.ownerFails = z.ownerFails;
      PrGPUFilterInstance zi;
      if (Create(g, zi)) {
        std::vector<int> d = DepsAt(g, zi, z.seq, 0);
        CHECK(d == z.want, "deps %s:%s, expected%s", z.label, Show(d).c_str(), Show(z.want).c_str());
        printf("  deps %-26s:%s\n", z.label, Show(d).c_str());
        g.filter.DisposeInstance(&zi);
      }
      gGPU.ownerFails = false;
    }
  }
  // Version 1 retains the historical request policy at radius/strength zero.
  SetNode(9102, 1000, 1100, 1000, 1100);
  SetI(106,1);
  if (Create(g, inst)) {
    SetI(OD_P_TEMPORAL_RADIUS, 0);
    CHECK(DepsAt(g, inst, 1050, 50).empty(), "radius 0 must request nothing");
    SetI(OD_P_TEMPORAL_RADIUS, 5);
    CHECK(DepsAt(g, inst, 1050, 50).size() == 10, "radius 5 must request 10 frames");
    SetF(OD_P_TEMPORAL_STRENGTH, 0);
    CHECK(DepsAt(g, inst, 1050, 50).empty(), "temporal strength 0 must request nothing");
    DefaultGPUParams();
    g.filter.DisposeInstance(&inst);
  }
}

// Renders five frames (current + 4) at sequence frame `seqFrame`; returns the
// stub's record of the render.
static Rec RecordedRender(GPUEnv& g, PrGPUFilterInstance& inst, long long seqFrame, PrRenderQuality q, int frames = 5,
                          long long clipFrame = LLONG_MIN, PrTime ticksPerFrame = kTPF) {
  static FakePPix fr[OD_MAX_FRAMES];
  static bool made = false;
  if (!made) {
    for (int i = 0; i < OD_MAX_FRAMES; ++i) fr[i] = MakePPix(PrPixelFormat_GPU_BGRA_4444_32f, 24, 16, 0, (unsigned)i + 40);
    made = true;
  }
  PrGPUFilterRenderParams rp = {};
  rp.inRenderTicksPerFrame = ticksPerFrame;
  rp.inDownsampleFactorX = rp.inDownsampleFactorY = 1.f;
  rp.inSequenceTime = seqFrame * ticksPerFrame;
  rp.inClipTime = (clipFrame == LLONG_MIN ? seqFrame + 4000 : clipFrame) * ticksPerFrame;
  rp.inQuality = q;
  PPixHand h[OD_MAX_FRAMES];
  for(int i=0;i<OD_MAX_FRAMES;++i) h[i]=(PPixHand)&fr[i];
  PPixHand out = (PPixHand)&fr[0];
  Pull();
  const size_t first = gStash.size();
  g.filter.Render(&inst, &rp, h, (csSDK_size_t)frames, &out);
  Pull();
  for (size_t i = gStash.size(); i-- > first;)
    if (gStash[i].kind == "render") return gStash[i];
  return Rec();
}

// Each high-quality trace can also be an export: decoding/encoding can pace
// requests, cached frames can omit calls, and the SDK permits out-of-order
// requests. Replaying both interpretations MUST produce the same Full result.
// This is deliberately a regression for the fail-closed rule, not a playback
// detector test that labels its own guessed intent as ground truth.
static void TestGPU27Patterns(GPUEnv& g) {
  struct Pattern { const char* name; int fps, gapMs, kind; PrRenderQuality q; };
  const Pattern cases[] = {
    {"25p paced playback / paced export",25,40,0,kPrRenderQuality_High},
    {"50p paced playback / paced export",50,20,0,kPrRenderQuality_High},
    {"back-to-back export",50,0,0,kPrRenderQuality_High},
    {"slow export below realtime, no skips",50,30,0,kPrRenderQuality_High},
    {"50p dropping playback / export with cached frames",50,20,1,kPrRenderQuality_High},
    {"scrub / out-of-order export",50,0,2,kPrRenderQuality_High},
    {"Premiere 26 low-quality previews",50,0,0,kPrRenderQuality_Low}
  };
  for (const auto& c : cases) {
    UseLayout(true, kCountTopics);
    SetNode(-1, -1, -1, -1, -1); // independent private request stream
    PrGPUFilterInstance inst;
    if (!Create(g, inst)) { CHECK(false, "pattern instance"); continue; }
    for (int i = 0; i < 32; ++i) {
      if (i && c.gapMs) usleep(c.gapMs * 1000);
      const int frame = c.kind == 1 ? i + i / 6 : c.kind == 2 ? (i % 3 == 0 ? 10 : i / 2) : i;
      const Rec r = RecordedRender(g, inst, frame, c.q, 5, LLONG_MIN, 254016000000LL / c.fps);
      const bool legacy = c.q == kPrRenderQuality_Low;
      CHECK(!r["quality"].empty() && r.num("quality") == (legacy ? 0 : 1) &&
            r.num("realtime") == (legacy ? 1 : 0), "%s: unsafe render %d", c.name, i);
    }
    g.filter.DisposeInstance(&inst);
    printf("  GPU 27 pattern: %s: safe mapping on 32 renders\n", c.name);
  }
  UseLayout(true, kCountTopics);
}

// Stub core: what the GPU path hands the core, one UI value at a time.
static void TestGPUParams(GPUEnv& g) {
  SetNode(9200, 1000, 1100, 1000, 1100);
  PrGPUFilterInstance inst;
  if (!Create(g, inst)) { CHECK(false, "gpu params: CreateInstance failed"); return; }
  Rec r = RecordedRender(g, inst, 1050, kPrRenderQuality_Draft);
  CHECK(r["off"] == "0,-1,1,-2,2", "gpu params: offsets %s, expected 0,-1,1,-2,2", r["off"].c_str());
  {
    // Keys identify frames for the core's profile (cuts, jumps, frames
    // already measured) and its frame cache, which checks every cached
    // neighbour against the host's buffer: the plug-in cannot tell when the
    // pixels behind a time change.
    auto split = [](const std::string& v) {
      std::vector<std::string> out;
      size_t a = 0;
      while (a <= v.size()) {
        size_t b = v.find(',', a);
        if (b == std::string::npos) b = v.size();
        out.push_back(v.substr(a, b - a));
        a = b + 1;
      }
      return out;
    };
    std::vector<std::string> k0 = split(r["keys"]);
    bool set = k0.size() == 5, distinct = true;
    for (size_t i = 0; i < k0.size(); ++i) {
      set = set && k0[i] != "none" && !k0[i].empty();
      for (size_t j = 0; j < i; ++j) distinct = distinct && k0[i] != k0[j];
    }
    CHECK(set && distinct, "gpu params: frame keys %s, expected five distinct keys", r["keys"].c_str());
    // The frame cache is on, with its GPU check of every cached neighbour.
    CHECK(r.num("nocache") == 0 && r["verify"] == "1", "gpu params: noFrameCache %g, verifyFrameCache %s, expected 0 "
          "and 1 (a neighbour reused by key only after a GPU check)", r.num("nocache"), r["verify"].c_str());
    // The same frame time gives the same key in the next render: offset +1
    // (index 2) of frame 1050 is offset 0 of frame 1051, and -1 of 1051 is 1050.
    Rec r1 = RecordedRender(g, inst, 1051, kPrRenderQuality_Draft);
    std::vector<std::string> k1 = split(r1["keys"]);
    CHECK(k1.size() == 5 && k0.size() == 5 && k1[0] == k0[2] && k1[1] == k0[0],
          "gpu params: keys not stable across renders (%s, then %s)", r["keys"].c_str(), r1["keys"].c_str());
    // At any clip speed (clip time = in-point + speed * item frame): frame
    // t+1's key as a neighbour of render t is its key as the current frame
    // of render t+1. Keys built from sequence time minus clip time changed
    // on every frame at any speed but 100 %, and the core then took every
    // render for a jump and blocked on it.
    struct Speed { const char* what; double speed; };
    const Speed speeds[] = {{"100 %", 1.0}, {"200 %", 2.0}, {"50 %", 0.5}, {"reverse", -1.0}, {"frame hold", 0.0}};
    for (const Speed& sp : speeds) {
      int agree = 0;
      for (int t = 0; t < 9; ++t) {
        const long long a = 1040 + t, b = a + 1;
        Rec ra = RecordedRender(g, inst, a, kPrRenderQuality_Draft, 5, 3000 + std::llround((a - 1000) * sp.speed));
        Rec rb = RecordedRender(g, inst, b, kPrRenderQuality_Draft, 5, 3000 + std::llround((b - 1000) * sp.speed));
        std::vector<std::string> ka = split(ra["keys"]), kb = split(rb["keys"]);
        agree += ka.size() == 5 && kb.size() == 5 && ka[2] == kb[0] && kb[1] == ka[0];
      }
      CHECK(agree == 9, "gpu keys at %s speed: the next frame's key agreed between renders in %d of 9", sp.what, agree);
      printf("  gpu keys at %-10s speed: neighbour and current-frame keys agree in %d of 9 renders\n", sp.what, agree);
    }
  }
  CHECK(r.num("wait") == 0 && r.num("manual") == 0 && r.num("luma") == 0.8 && r.num("chroma") == 1 &&
            r.num("radius") == 2 && r.num("tstr") == 1 && r.num("spatial") == 0.5 && r.num("sharpen") == 0 &&
            r.num("noise") == 1 && r.num("view") == 0 && r.num("quality") == 0,
        "gpu params: defaults (Auto under a draft render) reach the core unchanged");
  struct Case { const char* what; int pf; double ui; bool isInt; const char* field; double want; };
  const Case cases[] = {
      {"luma 37", OD_P_LUMA, 37, false, "luma", 0.37},
      {"chroma 81", OD_P_CHROMA, 81, false, "chroma", 0.81},
      {"temporal strength 64", OD_P_TEMPORAL_STRENGTH, 64, false, "tstr", 0.64},
      {"spatial 23", OD_P_SPATIAL, 23, false, "spatial", 0.23},
      {"sharpen 12", OD_P_SHARPEN, 12, false, "sharpen", 0.12},
      {"noise level 150", OD_P_NOISE_LEVEL, 150, false, "noise", 1.5},
      {"radius 1", OD_P_TEMPORAL_RADIUS, 1, true, "radius", 1},
      {"view Motion", OD_P_VIEW, 3, true, "view", OD_VIEW_MOTION},
  };
  int ok = 0;
  for (const Case& c : cases) {
    if (c.isInt) SetI(c.pf, (int)c.ui); else SetF(c.pf, c.ui);
    Rec rr = RecordedRender(g, inst, 1050, kPrRenderQuality_Draft, 5);
    const bool good = std::fabs(rr.num(c.field) - c.want) < 1e-5;
    CHECK(good, "gpu params %s: core %s = %s, expected %g", c.what, c.field, rr[c.field].c_str(), c.want);
    // The other fields stay at their defaults.
    CHECK((c.pf == OD_P_LUMA || rr.num("luma") == 0.8) && (c.pf == OD_P_CHROMA || rr.num("chroma") == 1) &&
              (c.pf == OD_P_SPATIAL || rr.num("spatial") == 0.5) && (c.pf == OD_P_NOISE_LEVEL || rr.num("noise") == 1),
          "gpu params %s: another field changed too", c.what);
    if (c.pf == OD_P_TEMPORAL_RADIUS) CHECK(rr["off"] == "0,-1,1,-2,2", "gpu params radius 1: includes fixed profiling window, offsets %s", rr["off"].c_str());
    ok += good;
    DefaultGPUParams();
  }
  struct Q { const char* what; int popup; PrRenderQuality host; int want; };
  const Q qs[] = {{"Auto, draft", OD_QUALITY_AUTO, kPrRenderQuality_Draft, 0},
                  {"Auto, low", OD_QUALITY_AUTO, kPrRenderQuality_Low, 0},
                  {"Auto, medium", OD_QUALITY_AUTO, kPrRenderQuality_Medium, 0},
                  {"Auto, invalid", OD_QUALITY_AUTO, kPrRenderQuality_Invalid, 0},
                  {"Auto, high", OD_QUALITY_AUTO, kPrRenderQuality_High, 1},
                  {"Auto, max", OD_QUALITY_AUTO, kPrRenderQuality_Max, 1},
                  {"Fast, high (an export too)", OD_QUALITY_FAST, kPrRenderQuality_High, 0},
                  {"Full, draft", OD_QUALITY_FULL, kPrRenderQuality_Draft, 1}};
  for (const Q& q : qs) {
    SetI(OD_P_QUALITY, q.popup);
    Rec rr = RecordedRender(g, inst, 1050, q.host);
    CHECK(rr.num("quality") == q.want && !rr["quality"].empty(), "gpu quality %s: core quality %s, expected %d", q.what,
          rr["quality"].c_str(), q.want);
    // Realtime (never wait for Premiere's queue) unless the host asked for a
    // high-quality render, whatever the Quality popup says.
    const int wantRT = (q.host == kPrRenderQuality_High || q.host == kPrRenderQuality_Max) ? 0 : 1;
    CHECK(!rr["realtime"].empty() && rr.num("realtime") == wantRT, "gpu quality %s: realtime %s, expected %d", q.what,
          rr["realtime"].c_str(), wantRT);
    ok += rr.num("quality") == q.want;
  }
  DefaultGPUParams();
  // Render uses the same neighbour list as GetFrameDependencies: at the
  // item's first frame the host hands over 2 neighbours (+1, +2).
  Rec edge = RecordedRender(g, inst, 1000, kPrRenderQuality_Draft, 3);
  CHECK(edge["off"] == "0,1,2", "gpu render at the item's first frame: offsets %s, expected 0,1,2", edge["off"].c_str());
  // The parameters are read once per frame: at its first dependency query, and its other
  // queries and its Render take them from there (a filter instance has a fixed set of
  // parameters). 69 host calls per frame instead of 69 per query and 69 more in Render.
  // Render reads the saved profile's policy once more (it decides whether a failure may pass
  // through).
  {
    const int before = gGPU.getParamCalls.load();
    std::vector<int> d = DepsAt(g, inst, 1060, 4060);
    const int afterDeps = gGPU.getParamCalls.load();
    Rec planned = RecordedRender(g, inst, 1060, kPrRenderQuality_Draft, 5, 4060);
    const int afterRender = gGPU.getParamCalls.load();
    CHECK(d.size() == 4 && afterDeps - before == 69, "gpu plan: %zu queries read %d parameters, expected 69", d.size(),
          afterDeps - before);
    CHECK(afterRender == afterDeps + 1, "gpu plan: Render read %d parameters after the queries, expected 1 policy read",
          afterRender - afterDeps);
    CHECK(planned["off"] == "0,-1,1,-2,2", "gpu plan: Render offsets %s", planned["off"].c_str());
    // Another frame's Render (no queries) reads them itself.
    RecordedRender(g, inst, 1061, kPrRenderQuality_Draft, 5, 4061);
    CHECK(gGPU.getParamCalls.load() - afterRender == 70, "gpu plan: a Render without queries read %d parameters",
          gGPU.getParamCalls.load() - afterRender);
    printf("  gpu parameters read per frame: %d at the queries, %d in Render\n", afterDeps - before,
           afterRender - afterDeps);
  }
  g.filter.DisposeInstance(&inst);
  // Owner lookup failing: Render still takes all four neighbours.
  gGPU.ownerFails = true;
  if (Create(g, inst)) {
    Rec fb = RecordedRender(g, inst, 1000, kPrRenderQuality_Draft, 5);
    CHECK(fb["off"] == "0,-1,1,-2,2", "gpu render without item bounds: offsets %s, expected all four", fb["off"].c_str());
    g.filter.DisposeInstance(&inst);
  }
  gGPU.ownerFails = false;
  printf("  gpu stub params: %d of %zu UI values and quality cases reach the core as expected\n", ok,
         sizeof cases / sizeof cases[0] + sizeof qs / sizeof qs[0]);
}

// Stub core: the advanced controls on the GPU path, in every parameter numbering the host
// might use. Topic entries may or may not take GPU indices (undocumented); GetParamCount
// tells them apart (54 or 40, 55 or 41 if the host counts the input), and anything else
// declines the instance: the software path renders it. The counts of earlier builds (51/52
// and 39 of the legacy build, 47/48 and 37/38 before the Experimental topic) cannot come from this
// build's definition and are declined too (OpenDenoiseGPU.mm, "Parameter layout"). most of
// the nine original controls sit inside topics, so they go through the layout as well.
static void TestGPUAdvanced(GPUEnv& g) {
  SetNode(9300, 1000, 1100, 1000, 1100);
  struct Layout { const char* name; bool withTopics; int count; bool countFails, noCountFn; bool known; };
  const Layout layouts[] = {
      {"with topics, count 68", true, kCountTopics, false, false, true},
      {"without topics, count 54", false, kCountNoTopics, false, false, true},
      {"with topics, input counted", true, kCountTopics + 1, false, false, true},
      {"without topics, count 55 (input counted)", false, kCountNoTopics + 1, false, false, true},
      {"count 51 (5584635, with topics)", true, 51, false, false, false},
      {"count 52 (5584635, input counted)", true, 52, false, false, false},
      {"count 39 (5584635, without topics)", false, 39, false, false, false},
      {"count 40 (5584635, without topics, input counted)", false, 40, false, false, false},
      {"count 64 (saved-profile layout, with topics)", true, 64, false, false, false},
      {"count 65 (saved-profile layout, input counted)", true, 65, false, false, false},
      {"count 67 (merge before ID 68, with topics)", true, 67, false, false, false},
      {"count 53 (merge before ID 68, without topics)", false, 53, false, false, false},
      {"count 73 (one more than this build)", true, 73, false, false, false},
      {"count 59 (one more, without topics)", false, 59, false, false, false},
      {"old count 47 (before Experimental)", true, 47, false, false, false},
      {"old count 48", true, 48, false, false, false},
      {"old count 37", false, 37, false, false, false},
      {"old count 38", false, 38, false, false, false},
      {"count 9", true, 9, false, false, false},
      {"GetParamCount failing", true, 47, true, false, false},
      {"no GetParamCount", true, 47, false, true, false},
  };
  for (const Layout& l : layouts) {
    UseLayout(l.withTopics, l.count, l.countFails);
    if (l.noCountFn) gSegmentSuite.GetParamCount = nullptr;
    if (!l.known) {
      // The advanced controls cannot be found: the instance is declined, and
      // Premiere renders the node in software (SDK guide 12.8.1), where every
      // control is read by its PF index (RunCPUParams). Before, the GPU
      // rendered with the advanced controls silently at their defaults.
      PrGPUFilterInstance inst = PrGPUFilterInstance();
      inst.piSuites = &g.suites;
      inst.inNodeID = kEffectNode;
      inst.inTimelineID = 3;
      const prSuiteError e = g.filter.CreateInstance(&inst);
      CHECK(e != suiteError_NoError && !inst.ioPrivatePluginData, "gpu advanced %s: the instance must be declined "
            "(err 0x%x)", l.name, (unsigned)e);
      if (inst.ioPrivatePluginData) g.filter.DisposeInstance(&inst);
      printf("  gpu advanced, %-42s declined: Premiere renders it in software\n", (std::string(l.name) + ":").c_str());
      gSegmentSuite.GetParamCount = SegGetParamCount;
      continue;
    }
    PrGPUFilterInstance inst;
    if (!Create(g, inst)) {
      CHECK(false, "gpu advanced %s: CreateInstance failed", l.name);
      gSegmentSuite.GetParamCount = SegGetParamCount;
      continue;
    }
    int good = 0, n = 0;
    for (const AdvCase& a : AdvancedCases()) {
      DefaultGPUParams();
      SetGPU(a.pf, a.ui, a.checkbox, a.popup);
      SetF(OD_P_LUMA, 61);  // one of the nine as well: it must arrive in every layout
      Rec r = RecordedRender(g, inst, 1050, kPrRenderQuality_Draft);
      std::map<std::string, double> want;
      if (l.known) want[a.field] = a.want;
      const std::string m = AdvancedMismatch(r, want);
      const bool lumaOk = std::fabs(r.num("luma") - 0.61) < 1e-5;
      CHECK(m.empty() && lumaOk, "gpu advanced %s, pf %d = %g: %s%s", l.name, a.pf, a.ui, m.c_str(),
            lumaOk ? "" : " (and Luma strength did not arrive)");
      good += m.empty() && lumaOk;
      ++n;
    }
    // Every control at once, each at a value of its own.
    DefaultGPUParams();
    for (const Sentinel& sn : Sentinels()) SetGPU(sn.pf, sn.ui, sn.checkbox, sn.popup);
    const Rec rs = RecordedRender(g, inst, 1050, kPrRenderQuality_Draft);
    const std::string ms = AdvancedMismatch(rs, SentinelWant());
    CHECK(ms.empty(), "gpu advanced %s, every control at a value of its own: %s", l.name, ms.c_str());
    const std::string mp = ParityMismatch(rs);
    CHECK(mp.empty(), "gpu advanced %s, CPU/GPU parity: %s", l.name, mp.c_str());
    // An old project: the experimental controls at the host's defaults, the ones at their
    // old-project values.
    DefaultGPUParams();
    for (const Sentinel& sn : Sentinels())
      if (!IsExperimental(sn.pf) && !After584(sn.pf)) SetGPU(sn.pf, sn.ui, sn.checkbox, sn.popup);
    OldProjectGPU();
    const std::string mo = AdvancedMismatch(RecordedRender(g, inst, 1050, kPrRenderQuality_Draft), OldProjectWant());
    CHECK(mo.empty(), "gpu advanced %s, a project saved before the Experimental topic: %s", l.name, mo.c_str());
    // A project saved by the legacy build: every one of its 51 controls at a value of
    // its own, Look at Custom; the core must get what the CPU path gave it.
    DefaultGPUParams();
    for (const Sentinel& sn : Sentinels())
      if (!After584(sn.pf)) SetGPU(sn.pf, sn.ui, sn.checkbox, sn.popup);
    for (const MainSentinel& ms : kMainSentinels) {
      if (ms.id == OD_P_TEMPORAL_RADIUS || ms.id == OD_P_QUALITY || ms.id == OD_P_VIEW) SetI(ms.id, (int)ms.ui);
      else SetF(ms.id, ms.ui);
    }
    OldProjectGPU();
    const Rec r584 = RecordedRender(g, inst, 1050, kPrRenderQuality_Draft, 3);  // radius 1: two neighbours
    const std::string m584 = AdvancedMismatch(r584, Old584Want()) + MainMismatch(r584);
    CHECK(m584.empty(), "gpu advanced %s, a project saved by 5584635: %s", l.name, m584.c_str());
    const std::string p584 = Old584Parity(r584);
    CHECK(p584.empty(), "gpu advanced %s, a project saved by 5584635, CPU/GPU parity: %s", l.name, p584.c_str());
    // Each Look's recipe: the same core values as on the CPU path.
    int looksOk = 0;
    for (int look = OD_LOOK_NATURAL; look <= OD_LOOK_PRESERVE; ++look) {
      DefaultGPUParams();
      for (int id : kLookSetIds) {
        if (id == OD_P_GRAIN_MODE) SetI(id, (int)RecipeOf(look, id));
        else SetF(id, RecipeOf(look, id));
      }
      SetI(OD_P_LOOK, look);
      const Rec rl = RecordedRender(g, inst, 1050, kPrRenderQuality_Draft);
      const std::string ml = AdvancedMismatch(rl, RecipeWant(look));
      const bool amounts = std::fabs(rl.num("luma") - LookLuma(look)) < 1e-6 && std::fabs(rl.num("chroma") - 1) < 1e-6;
      CHECK(ml.empty() && amounts, "gpu advanced %s, Look %s: %s", l.name, LookName(look), ml.c_str());
      looksOk += ml.empty() && amounts;
    }
    CHECK(looksOk == 4, "gpu advanced %s: %d of 4 Looks reached the core as their recipes", l.name, looksOk);
    // Grain stability keyframed (animatable): each frame reads it at its own
    // clip time, at its first dependency query or in Render. Keys at clip
    // frames 4000 (20 %) and 4010 (80 %); the fake host holds the last key.
    DefaultGPUParams();
    {
      PrParam a = {}, b = {};
      a.mType = b.mType = kPrParamType_Float64;
      a.mFloat64 = 20;
      b.mFloat64 = 80;
      gGPU.keyed[SegIndex(OD_P_GRAIN_STABILITY)] = {{4000 * kTPF, a}, {4010 * kTPF, b}};
    }
    const Rec k1 = RecordedRender(g, inst, 1003, kPrRenderQuality_Draft, 5, 4003);
    DepsAt(g, inst, 1012, 4012);  // read at the query, used by Render
    const Rec k2 = RecordedRender(g, inst, 1012, kPrRenderQuality_Draft, 5, 4012);
    const Rec k3 = RecordedRender(g, inst, 1004, kPrRenderQuality_Draft, 5, 4004);
    const std::string mk = AdvancedMismatch(k1, {{"gstab", 0.2}}) + AdvancedMismatch(k2, {{"gstab", 0.8}}) +
                           AdvancedMismatch(k3, {{"gstab", 0.2}});
    CHECK(mk.empty(), "gpu advanced %s, Grain stability keyframed 20 %% -> 80 %%: %s", l.name, mk.c_str());
    DefaultGPUParams();
    printf("  gpu advanced, %-42s %d of %d values as expected, all %zu at once (= CPU path), old projects (API v6, "
           "5584635), 4 Looks, keyframes\n", (std::string(l.name) + ":").c_str(), good, n, Sentinels().size());
    g.filter.DisposeInstance(&inst);
    gSegmentSuite.GetParamCount = SegGetParamCount;
  }
  // Checkboxes as Int32 work as well as Bool.
  {
    UseLayout(true, kCountTopics);
    PrGPUFilterInstance inst;
    if (Create(g, inst)) {
      SetB(OD_P_HOLD_PROFILE, true, /*asInt*/ true);
      SetB(OD_P_REMOVE_SPECKS, false, /*asInt*/ true);
      SetB(OD_P_COARSE_CHROMA, true, /*asInt*/ true);
      Rec r = RecordedRender(g, inst, 1050, kPrRenderQuality_Draft);
      const std::string m = AdvancedMismatch(r, {{"hold", 1}, {"speck", 0}, {"cchroma", 1}});
      CHECK(m.empty(), "gpu advanced: checkboxes delivered as Int32: %s", m.c_str());
      g.filter.DisposeInstance(&inst);
    }
  }
  // A numbering that is not the one the count implies, seen when the
  // instance is created (CreateInstance reads every advanced control once):
  // a Bool where a slider is expected, or a count of 39 from a host that
  // numbers the topics (the reads land on topic slots). Declined, for the
  // software path.
  auto declined = [&](const char* what) {
    PrGPUFilterInstance inst = PrGPUFilterInstance();
    inst.piSuites = &g.suites;
    inst.inNodeID = kEffectNode;
    inst.inTimelineID = 3;
    const prSuiteError e = g.filter.CreateInstance(&inst);
    CHECK(e != suiteError_NoError && !inst.ioPrivatePluginData, "gpu advanced: %s: the instance must be declined (err "
          "0x%x)", what, (unsigned)e);
    if (inst.ioPrivatePluginData) g.filter.DisposeInstance(&inst);
  };
  UseLayout(true, kCountTopics);
  SetB(OD_P_LUMA_LEVEL_FINE, true);
  declined("a Bool for a slider");
  UseLayout(true, kCountTopics);
  SetB(OD_P_GRAIN_STABILITY, true);
  declined("a Bool for Grain stability");
  UseLayout(true, kCountTopics);
  SetF(OD_P_COARSE_CHROMA, 1);
  declined("a float for Coarse chroma cleanup");
  UseLayout(true, kCountTopics);
  SetF(OD_P_GRAIN_MODE, 1);
  declined("a float for Minimum grain applies to");
  UseLayout(true, kCountTopics);
  SetI(OD_P_GRAIN_MODE, 5);  // 0-based 4: no such item
  declined("an item Minimum grain applies to does not have");
  UseLayout(true, kCountNoTopics);
  declined("a count of 54 with the topics numbered");
  UseLayout(false, kCountTopics);
  declined("a count of 68 with the topics not numbered");
  // A read that fails later, in a render (the layout passed the creation probe, which reads
  // every advanced control with its type): that render keeps the advanced controls at their
  // defaults (the nine still arrive), counted in the stats lines. Nothing else: a numbering
  // error puts the same type at the same index at every time, so the probe of each new
  // instance declines it; a failure that got past the probe is transient or depends on the
  // time. Without topics here: the rest of the harness uses the other layout.
  {
    UseLayout(false, kCountNoTopics);
    PrGPUFilterInstance inst;
    if (Create(g, inst)) {
      SetF(OD_P_LUMA_AMOUNT_FINE, 40);
      SetF(OD_P_LUMA, 61);
      gGPU.failParamIndex = SegIndex(OD_P_SHARPEN_HALO);
      Rec r = RecordedRender(g, inst, 1050, kPrRenderQuality_Draft);
      gGPU.failParamIndex = -1;
      const std::string m = AdvancedMismatch(r);
      CHECK(m.empty() && std::fabs(r.num("luma") - 0.61) < 1e-5,
            "gpu advanced: a failed read in a render must leave every advanced control at its default: %s", m.c_str());
      // The next render of the same instance reads them again.
      r = RecordedRender(g, inst, 1051, kPrRenderQuality_Draft);
      const std::string again = AdvancedMismatch(r, {{"amY0", 0.40}});
      CHECK(again.empty(), "gpu advanced: after one failed read in a render, the next render of that instance must "
            "read the advanced controls again: %s", again.c_str());
      g.filter.DisposeInstance(&inst);
      // New instances with that layout, on this clip and on others, in
      // other sequences: created, and they read the controls.
      int created = 0;
      for (int i = 0; i < 3; ++i) {
        SetNode(9310 + i, 2000 + 200 * i, 2100 + 200 * i, 2000 + 200 * i, 2100 + 200 * i);
        PrGPUFilterInstance next;
        if (Create(g, next, 10 + i)) {
          ++created;
          Rec rn = RecordedRender(g, next, 2050 + 200 * i, kPrRenderQuality_Draft);
          const std::string mn = AdvancedMismatch(rn, {{"amY0", 0.40}});
          CHECK(mn.empty(), "gpu advanced: a new instance after a failed read elsewhere: %s", mn.c_str());
          g.filter.DisposeInstance(&next);
        }
      }
      CHECK(created == 3, "gpu advanced: after one failed read in a render, %d of 3 new instances with that layout "
            "were created (the others declined: the layout was poisoned for the process)", created);
      printf("  gpu advanced, one failed read in a render: that render at the defaults, the next one and %d of 3 new "
             "instances read the controls\n", created);
      SetNode(9300, 1000, 1100, 1000, 1100);
    }
  }
  // The same failure on an instance whose grain mode is Whole picture: a project saved by
  // the legacy build (the host fills ID 68 with its old-project value) and a new effect switched to
  // it. That render keeps the grain mode the creation probe read (ID 68 cannot vary over
  // time, and every parameter change makes a new instance), not the new-effect default:
  // until the
  {
    int ok = 0;
    for (int k = 0; k < 2; ++k) {
      UseLayout(k == 0, k == 0 ? kCountTopics : kCountNoTopics);
      if (k == 0) OldProjectGPU();
      else SetI(OD_P_GRAIN_MODE, OD_GRAIN_MODE_WHOLE);
      SetNode(9320 + k, 3000 + 200 * k, 3100 + 200 * k, 3000 + 200 * k, 3100 + 200 * k);
      const char* what = k == 0 ? "a project saved by 5584635" : "a new effect set to Whole picture";
      PrGPUFilterInstance inst;
      if (!Create(g, inst, 20 + k)) {
        CHECK(false, "gpu advanced, %s: the instance was declined", what);
        continue;
      }
      const long long f = 3050 + 200 * k;
      const Rec r0 = RecordedRender(g, inst, f, kPrRenderQuality_Draft);
      gGPU.failParamIndex = SegIndex(OD_P_SHARPEN_HALO);
      const Rec r1 = RecordedRender(g, inst, f + 1, kPrRenderQuality_Draft);
      gGPU.failParamIndex = -1;
      const Rec r2 = RecordedRender(g, inst, f + 2, kPrRenderQuality_Draft);
      g.filter.DisposeInstance(&inst);
      const std::string m0 = AdvancedMismatch(r0, {{"grain", OD_GRAIN_WHOLE}});
      const std::string m1 = AdvancedMismatch(r1, {{"grain", OD_GRAIN_WHOLE}});
      const std::string m2 = AdvancedMismatch(r2, {{"grain", OD_GRAIN_WHOLE}});
      CHECK(m0.empty(), "gpu advanced, %s: a normal render: %s", what, m0.c_str());
      CHECK(m1.empty(), "gpu advanced, %s: a failed read in a render must keep Whole picture (the grain mode read at "
            "creation) and every other advanced control at its default: %s", what, m1.c_str());
      CHECK(m2.empty(), "gpu advanced, %s: the next render: %s", what, m2.c_str());
      ok += m0.empty() && m1.empty() && m2.empty();
    }
    printf("  gpu advanced, one failed read in a render, grain mode Whole picture: %d of 2 instances (old project, "
           "new effect switched) kept it in that render\n", ok);
    UseLayout(false, kCountNoTopics);
    SetNode(9300, 1000, 1100, 1000, 1100);
  }
  // Look is never read on the GPU path (the render does not depend on it):
  // an instance is created and renders the defaults whatever its slot holds,
  // a value of another type or nothing at all.
  {
    int made = 0;
    for (int k = 0; k < 2; ++k) {
      UseLayout(k == 0, k == 0 ? kCountTopics : kCountNoTopics);
      if (k == 0) SetB(OD_P_LOOK, true);
      else gGPU.params.erase(SegIndex(OD_P_LOOK));
      PrGPUFilterInstance inst;
      if (!Create(g, inst)) {
        CHECK(false, "gpu: the instance was declined over the Look slot (%s)", k == 0 ? "a Bool" : "empty");
        continue;
      }
      ++made;
      const Rec r = RecordedRender(g, inst, 1050, kPrRenderQuality_Draft);
      const std::string m = AdvancedMismatch(r);
      CHECK(m.empty() && std::fabs(r.num("luma") - 0.8) < 1e-6, "gpu: with an odd Look slot the render changed: %s",
            m.c_str());
      g.filter.DisposeInstance(&inst);
    }
    printf("  gpu: Look never read (%d of 2 instances made with a Bool or nothing in its slot)\n", made);
  }
  UseLayout(true, kCountTopics);
}

// Stub core: one session per effect on a track item, shared by the instances
// Premiere creates for each parameter change; parked, expired, bounded.
// Stub core: one session per effect on a track item, shared by the instances
// Premiere creates for each parameter change; parked, expired, bounded.
// ("Destroyed" means a destroy record after the session's last create or
// render.)
static bool DestroyedAfterLastUse(const std::vector<Rec>& log, const std::string& s) {
  if (s.empty()) return false;
  size_t last = 0;
  bool used = false;
  for (size_t i = 0; i < log.size(); ++i)
    if ((log[i].kind == "create" || log[i].kind == "render") && log[i]["s"] == s) { last = i; used = true; }
  for (size_t i = used ? last + 1 : 0; i < log.size(); ++i)
    if (log[i].kind == "destroy" && log[i]["s"] == s) return true;
  return false;
}
static void TestRegistry(GPUEnv& g) {
  std::vector<Rec> log;  // everything the stub recorded during this test
  auto pull = [&] {
    std::vector<Rec> v = Take();
    log.insert(log.end(), v.begin(), v.end());
  };
  auto sessionOf = [&](PrGPUFilterInstance& inst) {
    std::string s = RecordedRender(g, inst, 1050, kPrRenderQuality_Draft)["s"];
    pull();
    return s;
  };
  PrGPUFilterInstance a, b, c, d, e1, e2;
  SetNode(5001, 1000, 1100, 1000, 1100);
  Take();
  if (!Create(g, a)) { CHECK(false, "registry: CreateInstance failed"); return; }
  const std::string sA = sessionOf(a);
  g.filter.DisposeInstance(&a);  // a parameter change: dispose, then create
  if (!Create(g, b)) return;
  const std::string sB = sessionOf(b);
  CHECK(!sA.empty() && sA == sB, "registry: the next instance of the same effect must get its session (%s, %s)",
        sA.c_str(), sB.c_str());
  CHECK(CountOf(log, "destroy", sA) == 0 && CountOf(log, "create", sA) == 1,
        "registry: the session was destroyed or created again between the instances");

  SetNode(5002, 1000, 1100, 1000, 1100);
  if (!Create(g, c)) return;
  const std::string sC = sessionOf(c);
  CHECK(sC != sA, "registry: another effect (another runtime instance ID) must get its own session");
  SetNode(5001, 1000, 1100, 1000, 1100);
  PrGPUFilterInstance other;
  std::string sOther;
  if (Create(g, other, /*timeline*/ 4)) {
    sOther = sessionOf(other);
    CHECK(sOther != sA, "registry: the same ID in another sequence must get its own session");
    g.filter.DisposeInstance(&other);
  }
  // Copies of one effect pasted onto several clips of a sequence share the
  // runtime instance ID (observed in Premiere 26.5.2): another track item in
  // the same timeline is another session; the same item again is the same.
  std::string sItem2, sTrack3;
  {
    SetNode(5001, 1100, 1200, 1100, 1200);  // the next clip on the track
    PrGPUFilterInstance i2;
    if (Create(g, i2)) {
      sItem2 = RecordedRender(g, i2, 1150, kPrRenderQuality_Draft)["s"];
      pull();
      CHECK(!sItem2.empty() && sItem2 != sB, "registry: another clip with the same runtime instance ID in the same "
            "timeline must get its own session (%s, %s)", sItem2.c_str(), sB.c_str());
      g.filter.DisposeInstance(&i2);
    }
    SetNode(5001, 1000, 1100, 1000, 1100);
    SetProp(kOwnerNode, kVideoSegmentProperty_Clip_TrackID, 3);  // same times, another track
    PrGPUFilterInstance i3;
    if (Create(g, i3)) {
      sTrack3 = sessionOf(i3);
      CHECK(!sTrack3.empty() && sTrack3 != sB && sTrack3 != sItem2, "registry: the same times on another track must "
            "get their own session");
      g.filter.DisposeInstance(&i3);
    }
    SetNode(5001, 1000, 1100, 990, 1110);  // a transition added at the item's ends: still the same item
    PrGPUFilterInstance i4;
    if (Create(g, i4)) {
      CHECK(sessionOf(i4) == sB, "registry: the same track item (its effective bounds changed by a transition) must "
            "keep its session");
      g.filter.DisposeInstance(&i4);
    }
    SetNode(5001, 1000, 1100, 1000, 1100);
  }

  // No runtime instance ID (or no memory suite to read it): a private session,
  // destroyed with the instance.
  SetNode(-1, 1000, 1100, 1000, 1100);
  if (Create(g, d)) {
    const std::string sD = sessionOf(d);
    g.filter.DisposeInstance(&d);
    pull();
    CHECK(sD != sA && sD != sC && DestroyedAfterLastUse(log, sD), "registry: no ID must mean a private session");
  }
  gNoMemorySuite = true;
  SetNode(5001, 1000, 1100, 1000, 1100);
  if (Create(g, d)) {
    const std::string sD = sessionOf(d);
    CHECK(sD != sA, "registry: without the memory suite the ID cannot be read: private session");
    std::vector<int> deps = DepsAt(g, d, 1000, 0);
    CHECK((deps == std::vector<int>{-1, 1, -2, 2}), "without the memory suite: no bounds, every neighbour:%s",
          Show(deps).c_str());
    g.filter.DisposeInstance(&d);
    pull();
    CHECK(DestroyedAfterLastUse(log, sD), "registry: the private session goes with its instance");
  }
  gNoMemorySuite = false;

  // Two live instances of one effect (old and new may overlap).
  SetNode(5003, 1000, 1100, 1000, 1100);
  std::string sE;
  if (Create(g, e1) && Create(g, e2)) {
    sE = sessionOf(e1);
    CHECK(sessionOf(e2) == sE, "registry: overlapping instances of one effect share the session");
    g.filter.DisposeInstance(&e1);
    pull();
    CHECK(!DestroyedAfterLastUse(log, sE), "registry: destroyed while an instance still uses it");
    CHECK(sessionOf(e2) == sE, "registry: still usable by the remaining instance");
    g.filter.DisposeInstance(&e2);
  }
  g.filter.DisposeInstance(&b);
  g.filter.DisposeInstance(&c);

  // Parked sessions expire on their own (OD_PARK_MS=300 in this harness).
  const std::vector<std::string> parked = {sA, sC, sE, sOther, sItem2, sTrack3};
  auto gone = [&](const std::string& s) { return DestroyedAfterLastUse(log, s); };
  for (int i = 0; i < 100; ++i) {
    pull();
    if (std::all_of(parked.begin(), parked.end(), gone)) break;
    usleep(30000);
  }
  const int destroyed = (int)std::count_if(parked.begin(), parked.end(), gone);
  CHECK(destroyed == (int)parked.size(), "registry: %d of %zu parked sessions expired within 3 s", destroyed,
        parked.size());
  printf("  gpu registry: same effect -> same session, others separate, %d parked sessions expired\n", destroyed);

  // At most kMaxParked parked at once: parking six in a row (well within the
  // park time) destroys the oldest at once.
  Take();
  std::vector<std::string> six;
  for (int i = 0; i < 6; ++i) {
    SetNode(6001 + i, 1000, 1100, 1000, 1100);
    PrGPUFilterInstance t;
    if (Create(g, t)) {
      six.push_back(RecordedRender(g, t, 1050, kPrRenderQuality_Draft)["s"]);
      g.filter.DisposeInstance(&t);
    }
  }
  std::vector<Rec> recs = Take();
  const int early = (int)std::count_if(six.begin(), six.end(), [&](const std::string& s) { return DestroyedAfterLastUse(recs, s); });
  CHECK(six.size() == 6 && early >= 6 - odplug::kMaxParked && DestroyedAfterLastUse(recs, six[0]) &&
            DestroyedAfterLastUse(recs, six[1]),
        "registry: 6 sessions parked at once: %d destroyed at once (the oldest two at least), expected %d", early,
        6 - odplug::kMaxParked);
}

// Slow motion: with frame sampling, a sequence frame of a clip at 50 % speed
// (or of a 25p clip in a 50p sequence) repeats a source frame, so neighbours
// one sequence frame apart were copies of the current frame or of each
// other. They are requested round(1 / r) sequence frames apart (r = source
// frames per sequence frame), and their keys are those of the times asked.
static void TestSlowMotion(GPUEnv& g) {
  struct Case {
    const char* what;
    const char* speed;      // ClipNode::ClipSpeed, nullptr = not reported
    long long mediaTPF;     // MediaNode::StreamFrameRate (ticks per media frame), 0 = not reported
    std::vector<int> want;  // neighbours of sequence frame 1050, radius 2
  };
  const Case cases[] = {
      {"100 %", "1", kTPF, {-1, 1, -2, 2}},
      {"50 %", "0.5", kTPF, {-2, 2, -4, 4}},
      {"25 %", "0.25", kTPF, {-4, 4, -8, 8}},
      {"reverse 50 %", "-0.5", kTPF, {-2, 2, -4, 4}},
      {"200 %", "2", kTPF, {-1, 1, -2, 2}},
      {"media at half the sequence rate", "1", 2 * kTPF, {-2, 2, -4, 4}},
      {"50 % of media at twice the sequence rate", "0.5", kTPF / 2, {-1, 1, -2, 2}},
      {"no speed reported", nullptr, kTPF, {-1, 1, -2, 2}},
      {"no media frame rate", "0.5", 0, {-1, 1, -2, 2}},
  };
  for (const Case& c : cases) {
    SetNode(7300, 1000, 1100, 1000, 1100);
    if (c.speed) SetPropText(kOwnerNode, kVideoSegmentProperty_Clip_ClipSpeed, c.speed);
    if (c.mediaTPF) SetProp(kMediaNode, kVideoSegmentProperty_Media_StreamFrameRate, c.mediaTPF);
    PrGPUFilterInstance inst;
    if (!Create(g, inst)) { CHECK(false, "slow motion %s: CreateInstance failed", c.what); continue; }
    std::vector<int> deps = DepsAt(g, inst, 1050, 3000);
    CHECK(deps == c.want, "slow motion %s: neighbours%s, expected%s", c.what, Show(deps).c_str(), Show(c.want).c_str());
    if (!gReal && c.want[1] > 1) {
      // The core gets offsets in source frames (its radius filter), and the
      // key of the neighbour asked at +step is the key of that frame when it
      // is rendered itself.
      const int step = c.want[1];
      Rec r0 = RecordedRender(g, inst, 1050, kPrRenderQuality_Draft);
      Rec r1 = RecordedRender(g, inst, 1050 + step, kPrRenderQuality_Draft);
      auto split = [](const std::string& v) {
        std::vector<std::string> out;
        size_t a = 0;
        while (a <= v.size()) {
          size_t b = v.find(',', a);
          if (b == std::string::npos) b = v.size();
          out.push_back(v.substr(a, b - a));
          a = b + 1;
        }
        return out;
      };
      std::vector<std::string> k0 = split(r0["keys"]), k1 = split(r1["keys"]);
      CHECK(r0["off"] == "0,-1,1,-2,2", "slow motion %s: offsets handed to the core %s, expected 0,-1,1,-2,2", c.what,
            r0["off"].c_str());
      CHECK(k0.size() == 5 && k1.size() == 5 && k0[2] == k1[0] && k1[1] == k0[0],
            "slow motion %s: keys of the frames %d apart disagree (%s, then %s)", c.what, step, r0["keys"].c_str(),
            r1["keys"].c_str());
    }
    g.filter.DisposeInstance(&inst);
  }
  // Near the end of the item, at 50 %: only the requests that stay inside it.
  SetNode(7301, 1000, 1100, 1000, 1100);
  SetPropText(kOwnerNode, kVideoSegmentProperty_Clip_ClipSpeed, "0.5");
  SetProp(kMediaNode, kVideoSegmentProperty_Media_StreamFrameRate, kTPF);
  PrGPUFilterInstance inst;
  if (Create(g, inst)) {
    std::vector<int> deps = DepsAt(g, inst, 1097, 3000);
    CHECK((deps == std::vector<int>{-2, 2, -4}), "slow motion near the item end: neighbours%s, expected -2 +2 -4 "
          "(source frames -1 +1 -2)", Show(deps).c_str());
    g.filter.DisposeInstance(&inst);
  }
  printf("  gpu slow motion: neighbours one source frame apart at 50 %% and 25 %% speed and across frame rates\n");
}

// What maps the item's frames to source frames is part of every frame key: a slip (another
// media in-point at the same item start), a speed change or replaced media keep the session
// (the effect on its item) but give other keys, so the core restarts its profile there as at
// a jump instead of keeping the old footage's noise for frames it measured before. And an
// effect without an owner clip (an adjustment layer) is told apart from a copy of it by the
// effect's duration.
static void TestItemTiming(GPUEnv& g) {
  auto render = [&](long long inPoint, const char* speed, std::string* session) {
    SetNode(7400, 1000, 1100, 1000, 1100);
    SetProp(kMediaNode, kVideoSegmentProperty_Media_InPointMediaTimeAsTicks, inPoint);
    if (speed) SetPropText(kOwnerNode, kVideoSegmentProperty_Clip_ClipSpeed, speed);
    PrGPUFilterInstance inst;
    std::string keys;
    if (Create(g, inst)) {
      Rec r = RecordedRender(g, inst, 1050, kPrRenderQuality_Draft);
      keys = r["keys"];
      *session = r["s"];
      g.filter.DisposeInstance(&inst);  // a new instance after an edit; the session stays parked
    }
    return keys;
  };
  std::string s1, s2, s3, s4;
  const std::string k1 = render(5000 * kTPF, nullptr, &s1);
  const std::string k2 = render(5500 * kTPF, nullptr, &s2);  // slipped by 500 frames
  const std::string k3 = render(5000 * kTPF, nullptr, &s3);  // slipped back
  const std::string k4 = render(5000 * kTPF, "0.5", &s4);    // speed changed
  CHECK(!k1.empty() && k1 != k2, "item timing: a slip must change the frame keys (%s, %s)", k1.c_str(), k2.c_str());
  CHECK(k1 == k3, "item timing: the same timing must give the same keys (%s, %s)", k1.c_str(), k3.c_str());
  CHECK(k4 != k1, "item timing: a speed change must change the frame keys");
  CHECK(!s1.empty() && s1 == s2 && s2 == s3 && s3 == s4, "item timing: the effect on its item keeps its session through "
        "slips and speed changes (%s %s %s %s)", s1.c_str(), s2.c_str(), s3.c_str(), s4.c_str());

  // Adjustment layers: the owner lookup fails. The identity is incomplete, so every instance
  // has a private session.
  gGPU.ownerFails = true;
  auto adjust = [&](long long frames) {
    SetNode(7401, -1, -1, -1, -1);
    SetProp(kEffectNode, kVideoSegmentProperty_Effect_EffectDuration, frames * kTPF);
    PrGPUFilterInstance inst;
    std::string s;
    if (Create(g, inst)) {
      s = RecordedRender(g, inst, 1050, kPrRenderQuality_Draft)["s"];
      g.filter.DisposeInstance(&inst);
    }
    return s;
  };
  const std::string a1 = adjust(300), a2 = adjust(450), a3 = adjust(300);
  gGPU.ownerFails = false;
  CHECK(!a1.empty() && a1 != a2 && a1 != a3 && a2 != a3, "adjustment layers (no owner clip) must get private sessions "
        "(%s, %s, %s)", a1.c_str(), a2.c_str(), a3.c_str());
  Take();
  printf("  gpu item timing: slips and speed changes change the keys, not the session; no owner clip: private "
         "sessions\n");
}

// ---- failure paths on the GPU path ------------------- Render returns success only if the
// output holds a valid frame: rendered, the input in place, or a copy of the input of the
// same format and size. A failure before or after the output was allocated returns an error
// otherwise (and an output the plug-in allocated is disposed). Before, an exception or an
// unreadable frame returned success with the output not written, and a mismatched output got
// a byte copy across formats.
static void TestGPUFailurePaths(GPUEnv& g) {
  SetNode(9700, 1000, 1100, 1000, 1100);
  PrGPUFilterInstance inst;
  if (!Create(g, inst)) { CHECK(false, "gpu failure paths: CreateInstance failed"); return; }
  const int W = 48, H = 27;
  PrGPUFilterRenderParams rp = {};
  rp.inRenderTicksPerFrame = kTPF;
  rp.inDownsampleFactorX = rp.inDownsampleFactorY = 1.f;
  rp.inSequenceTime = 1050 * kTPF;
  rp.inClipTime = 4050 * kTPF;
  rp.inQuality = kPrRenderQuality_Draft;
  SetI(OD_P_TEMPORAL_RADIUS, 0);  // the current frame alone
  int ok = 0, n = 0;
  auto bytesOf = [](const FakePPix& p) {
    return std::vector<uint8_t>((const uint8_t*)p.buffer.contents, (const uint8_t*)p.buffer.contents + p.buffer.length);
  };
  auto sameRows = [&](const FakePPix& a, const FakePPix& b) {
    const int bpp = a.format == PrPixelFormat_GPU_BGRA_4444_32f ? 16 : 8;
    if (a.width != b.width || a.height != b.height || a.format != b.format) return false;
    for (int y = 0; y < a.height; ++y)
      if (memcmp((const char*)a.buffer.contents + (size_t)y * a.rowBytes, (const char*)b.buffer.contents + (size_t)y * b.rowBytes,
                 (size_t)a.width * bpp))
        return false;
    return true;
  };
  auto render = [&](FakePPix& in, PPixHand* out) {
    PPixHand frames[1] = {(PPixHand)&in};
    const prSuiteError e = g.filter.Render(&inst, &rp, frames, 1, out);
    WaitForGPU();
    return e;
  };
  auto expect = [&](bool good, const char* what, prSuiteError e) {
    CHECK(good, "gpu failure: %s (err 0x%x)", what, (unsigned)e);
    ok += good;
    ++n;
  };

  // (a) The input cannot be read (no GPU data).
  {
    FakePPix in = MakePPix(PrPixelFormat_GPU_BGRA_4444_32f, W, H, 0, 1), dst = MakePPix(PrPixelFormat_GPU_BGRA_4444_32f, W, H, 0, 2);
    const std::vector<uint8_t> inBefore = bytesOf(in), dstBefore = bytesOf(dst);
    gGPU.dataFails.insert((PPixHand)&in);
    PPixHand out = (PPixHand)&in;
    prSuiteError e = render(in, &out);
    expect(e == suiteError_NoError && bytesOf(in) == inBefore, "unreadable input, in place: the input as it is", e);
    out = (PPixHand)&dst;
    e = render(in, &out);
    expect(e != suiteError_NoError && bytesOf(dst) == dstBefore,
           "unreadable input into a separate output: must be an error, output untouched", e);
    out = nullptr;
    e = render(in, &out);
    expect(e != suiteError_NoError, "unreadable input, no output: must be an error", e);
    gGPU.dataFails.clear();
  }
  // (b) An input in a format the filter does not render, into an output of
  // the same format and size: a copy of the input.
  {
    auto foreign = [&](unsigned seed) {
      FakePPix p;
      p.format = PrPixelFormat_BGRA_4444_8u; p.width = W; p.height = H; p.rowBytes = W * 4;
      p.buffer = [gGPU.device newBufferWithLength:(NSUInteger)p.rowBytes * H options:MTLResourceStorageModeShared];
      for (size_t i = 0; i < p.buffer.length; ++i) ((uint8_t*)p.buffer.contents)[i] = (uint8_t)(i * 7 + seed);
      return p;
    };
    FakePPix in = foreign(3), dst = foreign(9);
    PPixHand out = (PPixHand)&dst;
    const prSuiteError e = render(in, &out);
    expect(e == suiteError_NoError && bytesOf(dst) == bytesOf(in), "foreign format, separate output of that format: "
           "the input copied", e);
  }
  // (c) An output of another format, (d) of another size: an error, no byte
  // copy.
  {
    FakePPix in = MakePPix(PrPixelFormat_GPU_BGRA_4444_32f, W, H, 0, 4);
    FakePPix other = MakePPix(PrPixelFormat_GPU_BGRA_4444_16f, W, H, 0, 5);
    FakePPix smaller = MakePPix(PrPixelFormat_GPU_BGRA_4444_32f, W - 8, H - 4, 0, 6);
    const std::vector<uint8_t> otherBefore = bytesOf(other), smallerBefore = bytesOf(smaller);
    PPixHand out = (PPixHand)&other;
    prSuiteError e = render(in, &out);
    expect(e != suiteError_NoError && bytesOf(other) == otherBefore, "output of another format: must be an error, "
           "output untouched", e);
    out = (PPixHand)&smaller;
    e = render(in, &out);
    expect(e != suiteError_NoError && bytesOf(smaller) == smallerBefore, "output of another size: must be an error, "
           "output untouched", e);
  }
  // (e) An exception before the output exists: an error, no output.
  {
    FakePPix in = MakePPix(PrPixelFormat_GPU_BGRA_4444_32f, W, H, 0, 7);
    PPixHand out = nullptr;
    gGPU.throwInPAR = true;
    const prSuiteError e = render(in, &out);
    gGPU.throwInPAR = false;
    expect(e != suiteError_NoError && out == nullptr, "exception before the output is allocated: must be an error "
           "with no output", e);
  }
  // (f) An exception after the plug-in allocated the output: the input in it.
  {
    FakePPix in = MakePPix(PrPixelFormat_GPU_BGRA_4444_32f, W, H, 0, 8);
    PPixHand out = nullptr;
    gGPU.throwInGetParam = true;  // the parameters are read after the output exists
    const prSuiteError e = render(in, &out);
    gGPU.throwInGetParam = false;
    const bool copied = out && sameRows(*(FakePPix*)out, in);
    expect(e == suiteError_NoError && copied, "exception after the output is allocated: the input passed through "
           "into it", e);
  }
  // (g) The core fails and the allocated output cannot take the copy (its
  // buffer is short): an error, and that output disposed.
  {
    FakePPix in = MakePPix(PrPixelFormat_GPU_BGRA_4444_32f, W, H, 0, 9);
    PPixHand out = nullptr;
    const int disposed = gGPU.disposed;
    gGPU.createShortBy = 64;
    const prSuiteError e = render(in, &out);
    gGPU.createShortBy = 0;
    expect(e != suiteError_NoError && out == nullptr && gGPU.disposed == disposed + 1,
           "unusable allocated output: must be an error, the output disposed and not returned", e);
  }
  // (h) No output from the host: one is allocated, and holds the input
  // (pass-through core) or a render (real core).
  {
    FakePPix in = MakePPix(PrPixelFormat_GPU_BGRA_4444_16f, W, H, 0, 10);
    PPixHand out = nullptr;
    const prSuiteError e = render(in, &out);
    const bool written = out && (gReal ? bytesOf(*(FakePPix*)out) != std::vector<uint8_t>(bytesOf(*(FakePPix*)out).size(), 0xA5)
                                       : sameRows(*(FakePPix*)out, in));
    expect(e == suiteError_NoError && out && out != (PPixHand)&in && written, "no output from the host: one allocated "
           "and written", e);
  }
  // (i) An exception in place: the input is the output, untouched.
  {
    FakePPix in = MakePPix(PrPixelFormat_GPU_BGRA_4444_32f, W, H, 0, 11);
    const std::vector<uint8_t> before = bytesOf(in);
    PPixHand out = (PPixHand)&in;
    gGPU.throwInGetParam = true;
    const prSuiteError e = render(in, &out);
    gGPU.throwInGetParam = false;
    expect(e == suiteError_NoError && bytesOf(in) == before, "exception in place: the input, untouched", e);
  }
  // (j) Every render that ends before the core call is in the stats lines (renders,
  // "failed", and how many returned an error), and a failure that repeats is logged by a
  // counter. Second 520 renders: at least one window of the stats (250 renders) is made of
  // them alone.
  {
    FakePPix in = MakePPix(PrPixelFormat_GPU_BGRA_4444_32f, W, H, 0, 12);
    FakePPix other = MakePPix(PrPixelFormat_GPU_BGRA_4444_16f, W, H, 0, 13);
    const long from = PluginLogSize();
    const int N = 520;
    int errors = 0;
    for (int i = 0; i < N; ++i) {
      PPixHand out = (PPixHand)&other;
      PPixHand frames[1] = {(PPixHand)&in};
      if (g.filter.Render(&inst, &rp, frames, 1, &out) != suiteError_NoError) ++errors;
    }
    const std::string log = PluginLogFrom(from);
    int windows = 0, whole = 0, best = 0;
    for (const std::string& l : LinesWith(log, "GPU: stats ")) {
      unsigned long long a = 0, b = 0;
      int failed = -1, errs = -1;
      const size_t r = l.find(", renders "), f = l.find("; failed ");
      if (r == std::string::npos || f == std::string::npos) continue;
      if (sscanf(l.c_str() + r, ", renders %llu-%llu", &a, &b) != 2) continue;
      if (sscanf(l.c_str() + f, "; failed %d (%d of them returned an error", &failed, &errs) != 2) continue;
      ++windows;
      const int count = (int)(b - a + 1);
      best = std::max(best, count);
      if (count >= 200 && failed == count && errs == count) ++whole;
    }
    const size_t lines = LinesWith(log, "an output frame of another format or size").size();
    CHECK(errors == N, "gpu failure: %d of %d renders into an output of another format returned an error", errors, N);
    CHECK(whole >= 1, "gpu failure: the stats lines during %d renders that returned an error: %d windows, none of "
          "at least 200 renders all counted failed and errors (largest window %d renders); log:\n%s", N, windows, best,
          log.substr(0, 4000).c_str());
    CHECK(lines >= 10 && lines <= 40, "gpu failure: %zu log lines for %d renders that returned an error (expected the "
          "first 16 occurrences, then the 32nd, 64th ...)", lines, N);
    printf("  gpu failure paths: %d renders into an output of another format: %d stats windows, %d made of them "
           "alone (all failed, all errors), %zu log lines\n", N, windows, whole, lines);
  }
  SetI(OD_P_TEMPORAL_RADIUS, 2);
  g.filter.DisposeInstance(&inst);
  printf("  gpu failure paths: %d of %d as expected (success only with a valid output)\n", ok, n);
}

// ---- session identity on the GPU path: incomplete ownership Copies of one effect share a
// runtime ID. Without the owner clip (adjustment layers, as reported) or its track, the
// session is private, never shared by what is left of the identity. Before, the same ID and
// effect duration shared one session, and a missing track ID counted as track 0.
static void TestGPUIdentity(GPUEnv& g) {
  auto sessionOf = [&](PrGPUFilterInstance& inst) { return RecordedRender(g, inst, 1050, kPrRenderQuality_Draft)["s"]; };
  Take();
  // No owner node, the same ID and duration for both (two adjustment layers
  // of one length, or a parameter change of one).
  gGPU.ownerFails = true;
  SetNode(9800, -1, -1, -1, -1);
  SetProp(kEffectNode, kVideoSegmentProperty_Effect_EffectDuration, 300 * kTPF);
  PrGPUFilterInstance a, b;
  std::string sa, sb;
  if (Create(g, a) && Create(g, b)) {
    sa = sessionOf(a);
    sb = sessionOf(b);
    CHECK(!sa.empty() && sa != sb, "gpu identity: no owner clip, one runtime ID and duration: the two instances must "
          "have sessions of their own (%s, %s)", sa.c_str(), sb.c_str());
    g.filter.DisposeInstance(&a);
    g.filter.DisposeInstance(&b);
  }
  gGPU.ownerFails = false;
  std::vector<Rec> recs = Take();
  CHECK(DestroyedAfterLastUse(recs, sa) && DestroyedAfterLastUse(recs, sb), "gpu identity: private sessions go with "
        "their instances");
  // The owner clip without a track ID: two clips at one start on two tracks.
  SetNode(9801, 1000, 1100, 1000, 1100);
  ClearProp(kOwnerNode, kVideoSegmentProperty_Clip_TrackID);
  PrGPUFilterInstance c, d;
  if (Create(g, c) && Create(g, d)) {
    const std::string sc = sessionOf(c), sd = sessionOf(d);
    CHECK(!sc.empty() && sc != sd, "gpu identity: no track ID: the instances must have sessions of their own (%s, %s)",
          sc.c_str(), sd.c_str());
    g.filter.DisposeInstance(&c);
    g.filter.DisposeInstance(&d);
  }
  Take();
  printf("  gpu identity: no owner clip or no track ID -> private sessions\n");

  // An exception inside CreateInstance: an error, and nothing left behind
  // (the instance object and the suites it acquired). Before, the object
  // leaked with its suites.
  {
    SetNode(9802, 1000, 1100, 1000, 1100);
    const int held = gAcquired.load() - gReleased.load();
    PrGPUFilterInstance inst = PrGPUFilterInstance();
    inst.piSuites = &g.suites;
    inst.inNodeID = kEffectNode;
    inst.inTimelineID = 3;
    gGPU.throwInParamCount = true;
    const prSuiteError e = g.filter.CreateInstance(&inst);
    gGPU.throwInParamCount = false;
    CHECK(e != suiteError_NoError && !inst.ioPrivatePluginData, "gpu: an exception in CreateInstance must return an "
          "error (0x%x)", (unsigned)e);
    CHECK(gAcquired.load() - gReleased.load() == held, "gpu: an exception in CreateInstance left %d suites acquired",
          gAcquired.load() - gReleased.load() - held);
  }
}

// ---- real core: two clips, one runtime ID, rendered alternately (GPU) --------
// Clip A: flat grey, heavy noise; clip B: almost clean, fine texture. Renders
// of A alternating with B must equal A's renders alone when the sessions are
// apart. Full quality under a high-quality render: the first renders block for
// their estimates, so the result does not depend on timing.
static void RunGPULayers(GPUEnv& g, const char* label, bool ownerA, bool ownerB, long long startB, bool expectSame) {
  const int W = kRealW / 2, H = kRealH / 2;
  auto fill = [&](FakePPix& p, int layer, unsigned seed) {
    Rng r(seed);
    float* px = (float*)p.buffer.contents;
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        const double tex = layer == 1 && ((x / 5 + y / 5) & 1) ? 0.012 : 0.0;
        const double sigma = layer == 0 ? 0.02 : 0.002;
        for (int c = 0; c < 3; ++c) px[(size_t)y * W * 4 + x * 4 + c] = (float)(0.45 + tex + sigma * r.gauss());
        px[(size_t)y * W * 4 + x * 4 + 3] = 1.f;
      }
  };
  // One render of `layer` frame k through `inst`; returns the output bytes.
  auto renderOne = [&](PrGPUFilterInstance& inst, int layer, int k, long long itemStart) {
    FakePPix fr[5];
    for (int i = 0; i < 5; ++i) {
      fr[i] = MakePPix(PrPixelFormat_GPU_BGRA_4444_32f, W, H, 0, 1);
      fill(fr[i], layer, 1000u * (unsigned)(layer + 1) + (unsigned)(k * 10 + i));
    }
    PrGPUFilterRenderParams rp = {};
    rp.inRenderTicksPerFrame = kTPF;
    rp.inDownsampleFactorX = rp.inDownsampleFactorY = 1.f;
    rp.inSequenceTime = (itemStart + 20 + k) * kTPF;
    rp.inClipTime = rp.inSequenceTime + 3000 * kTPF;
    rp.inQuality = kPrRenderQuality_High;
    PPixHand frames[5] = {(PPixHand)&fr[0], (PPixHand)&fr[1], (PPixHand)&fr[2], (PPixHand)&fr[3], (PPixHand)&fr[4]};
    PPixHand out = (PPixHand)&fr[0];
    const prSuiteError e = g.filter.Render(&inst, &rp, frames, 5, &out);
    WaitForGPU();
    CHECK(e == suiteError_NoError, "%s: render err 0x%x", label, (unsigned)e);
    return std::vector<float>((const float*)fr[0].buffer.contents, (const float*)fr[0].buffer.contents + (size_t)W * H * 4);
  };
  auto make = [&](PrGPUFilterInstance& inst, long long runtimeID, bool owner, long long start) {
    gGPU.ownerFails = !owner;
    SetNode(runtimeID, start, start + 100, start, start + 100);
    SetProp(kEffectNode, kVideoSegmentProperty_Effect_EffectDuration, 100 * kTPF);
    const bool made = Create(g, inst);
    gGPU.ownerFails = false;
    return made;
  };
  SetI(OD_P_QUALITY, OD_QUALITY_FULL);
  SetF(OD_P_LUMA, 100);
  std::vector<std::vector<float>> aloneA, aloneB, alt;
  // Fresh runtime IDs: no parked session from a case before. The renders
  // alone have the ownership of the alternating ones (the frame keys, which
  // choose the frames the profile measures, depend on it).
  static long long nextID = 9910;
  {
    PrGPUFilterInstance ia, ib;
    const long long idA = nextID++, idB = nextID++;
    if (!make(ia, idA, ownerA, 1000) || !make(ib, idB, ownerB, startB)) { CHECK(false, "%s: CreateInstance", label); return; }
    for (int k = 0; k < 3; ++k) aloneA.push_back(renderOne(ia, 0, k, 1000));
    for (int k = 0; k < 3; ++k) aloneB.push_back(renderOne(ib, 1, k, startB));
    g.filter.DisposeInstance(&ia);
    g.filter.DisposeInstance(&ib);
  }
  {
    PrGPUFilterInstance ia, ib;
    const long long id = nextID++;
    if (!make(ia, id, ownerA, 1000) || !make(ib, id, ownerB, startB)) { CHECK(false, "%s: CreateInstance", label); return; }
    for (int k = 0; k < 3; ++k) {
      alt.push_back(renderOne(ia, 0, k, 1000));
      alt.push_back(renderOne(ib, 1, k, startB));
    }
    g.filter.DisposeInstance(&ia);
    g.filter.DisposeInstance(&ib);
  }
  auto worst = [](const std::vector<float>& x, const std::vector<float>& y) {
    double w = 0;
    for (size_t i = 0; i < x.size(); ++i) w = std::max(w, (double)std::fabs(x[i] - y[i]));
    return w * 65535;  // in 16-bit codes
  };
  int sameA = 0, sameB = 0;
  double largest = 0;
  for (int k = 0; k < 3; ++k) {
    const double da = worst(alt[(size_t)(2 * k)], aloneA[(size_t)k]), db = worst(alt[(size_t)(2 * k + 1)], aloneB[(size_t)k]);
    sameA += da <= 1;
    sameB += db <= 1;
    largest = std::max(largest, std::max(da, db));
  }
  if (expectSame)
    CHECK(sameA == 3 && sameB == 3, "%s: rendered alternately, %d of 3 frames of A and %d of 3 of B equal their result "
          "alone", label, sameA, sameB);
  else
    CHECK(sameA + sameB < 6, "%s (control): one session for both clips must change a result, or this test cannot see "
          "sharing", label);
  printf("  gpu layers %-36s: alternating = alone for %d/3 of A, %d/3 of B (largest difference %.0f codes)\n", label,
         sameA, sameB, largest);
  DefaultGPUParams();
}

// RenderStats counters belong to each instance. Test CPU/GPU request accounting, failed
// renders, completion latency and teardown with independent fixtures.
static void TestRenderStats() {
  using odplug::RenderStats;
  using odplug::RenderSample;
  const int64_t tpf = kTPF;
  // Two streams (two clips, or two sequences) rendered interleaved at the
  // same sequence times, each one frame forward per request.
  {
    RenderStats rs(1000, /*flushFirst*/ false);
    double t = 1000;
    for (int k = 0; k < 20; ++k)
      for (uint64_t stream : {11ull, 22ull}) {
        rs.NoteEnter(stream, t, (1050 + k) * tpf, tpf, 33.4);
        RenderSample r;
        r.enterMs = t;
        rs.Note(r);
        t += 16.7;
      }
    const RenderStats::Snapshot s = rs.Current();
    CHECK(s.stepOne == 38 && s.repeats == 0 && s.jumps == 0 && s.otherSteps == 0 && s.streams == 2,
          "stats: two interleaved streams: %d steps of one frame (expected 38), %d asked again, %d jumps, %d other "
          "(expected 0), %d streams", s.stepOne, s.repeats, s.jumps, s.otherSteps, s.streams);
    CHECK(s.gaps == 0 && s.intervals == 38, "stats: gaps between requests of a stream: %d over 1.5 frames of %d",
          s.gaps, s.intervals);
  }
  // One stream that skips two frames, then asks for a frame again.
  {
    RenderStats rs(1000, false);
    const int64_t seq[] = {100, 101, 104, 105, 104};
    double t = 0;
    for (int64_t f : seq) {
      rs.NoteEnter(5, t, f * tpf, tpf, 33.4);
      t += 33;
    }
    const RenderStats::Snapshot s = rs.Current();
    CHECK(s.stepOne == 2 && s.jumps == 1 && s.notAsked == 2 && s.otherSteps == 1 && s.repeats == 1,
          "stats: one stream 100 101 104 105 104: steps %d, jumps %d (%d not asked), other %d, again %d; expected 2, 1 "
          "(2), 1, 1", s.stepOne, s.jumps, s.notAsked, s.otherSteps, s.repeats);
  }
  // GPU time: one sample per completed render, however the completions are
  // spread over the calls; failed ones counted.
  {
    RenderStats rs(1000, false);
    ODCompletion c[3] = {{1, 2.0, 0.1, 2.5, 0}, {2, 2.1, 0.2, 2.6, 0}, {3, 1.9, 0.1, 2.2, 1}};
    rs.NoteCompletions(c, 0, 0);      // call 1: nothing completed yet
    rs.NoteCompletions(c, 2, 0);      // call 2: renders 1 and 2
    rs.NoteCompletions(c + 2, 1, 1);  // call 3: render 3 (one lost before it)
    for (int i = 0; i < 3; ++i) {
      RenderSample r;
      r.reachedCore = true;  // the waits are those of a core call
      r.lockWaitMs = i == 1 ? 7.5 : 0;
      r.commandBufferMs = i == 2 ? 12.0 : 0;
      r.inFlight = i + 1;
      r.asyncFailures = i == 2 ? 1 : 0;
      rs.Note(r);
    }
    const RenderStats::Snapshot s = rs.Current();
    CHECK(s.gpuSamples == 3 && s.gpuFailed == 1 && s.completionsDropped == 1, "stats: %d GPU samples (expected 3, one "
          "per completed render), %d failed, %d lost", s.gpuSamples, s.gpuFailed, s.completionsDropped);
    CHECK(s.lockWaitMax == 7.5 && s.commandBufferMax == 12.0 && s.inFlightMax == 3 && s.asyncFailures == 1,
          "stats: waits for the session %.1f / a command buffer %.1f ms, in flight %d, failed later %d",
          s.lockWaitMax, s.commandBufferMax, s.inFlightMax, s.asyncFailures);
  }
  // Renders that never reached the core: counted as renders and as failed (errors apart),
  // kept out of the core call's percentiles and the neighbour averages; an unknown frame
  // size is not a size change.
  {
    RenderStats rs(1000, false);
    RenderSample ok;
    ok.reachedCore = true;
    ok.coreMs = 2.0;
    ok.width = 64;
    ok.height = 36;
    ok.wanted = ok.got = 4;
    rs.Note(ok);
    RenderSample early;  // an error before the core, size unknown
    early.failed = early.errorReturned = true;
    rs.Note(early);
    RenderSample passed;  // the input passed through, no error
    passed.failed = true;
    passed.width = 64;
    passed.height = 36;
    rs.Note(passed);
    const RenderStats::Snapshot s = rs.Current();
    CHECK(s.renders == 3 && s.failed == 2 && s.errors == 1 && s.reachedCore == 1 && s.coreSamples == 1,
          "stats: renders %d (expected 3), failed %d (2), errors %d (1), reached the core %d (1), core samples %d (1)",
          s.renders, s.failed, s.errors, s.reachedCore, s.coreSamples);
    CHECK(s.sizeChanges == 1 && s.lastW == 64 && s.lastH == 36, "stats: an unknown frame size counted: %d size "
          "changes (expected 1), last %ux%u", s.sizeChanges, s.lastW, s.lastH);
  }
  // The logged lines name the waits and say what the request counts are.
  {
    {
      std::lock_guard<std::mutex> lock(gLogMutex);
      gLogLines.clear();
    }
    RenderStats rs(2, false);
    RenderSample r;
    rs.Note(r);
    rs.Note(r);  // a window of 2: flushed
    std::lock_guard<std::mutex> lock(gLogMutex);
    std::string all;
    for (const std::string& l : gLogLines) all += l + "\n";
    CHECK(all.find("waiting for the session") != std::string::npos && all.find("for a command buffer") != std::string::npos &&
              all.find("returned an error to Premiere") != std::string::npos &&
              all.find("not frames shown or dropped") != std::string::npos &&
              all.find("GPU time per completed render") != std::string::npos,
          "stats: the logged lines do not name the waits and the request pattern:\n%s", all.c_str());
  }
  printf("  render stats: streams apart, completions once, waits reported\n");
}

static void TestGPU(PrGPUFilterEntry entry) {
  printf("xGPUFilterEntry\n");
  gIsPremiere = true;
  gGPU.device = MTLCreateSystemDefaultDevice();
  if (!gGPU.device) {
    printf("  no Metal device: GPU tests skipped\n");
    return;
  }
  gGPU.queue = [gGPU.device newCommandQueue];

  GPUEnv g;
  g.util.getSPBasicSuite = GetBasic;
  g.suites.utilFuncs = &g.util;
  PrGPUFilter& filter = g.filter;
  PrGPUFilterInfo info = {};
  csSDK_int32 index = 0;
  prSuiteError e = entry(PrSDKGPUFilterInterfaceVersion, &index, kPrTrue, &g.suites, &filter, &info);
  CHECK(e == suiteError_NoError && index == 0, "startup err 0x%x index %d", (unsigned)e, index);
  CHECK(info.outInterfaceVersion == PrSDKGPUFilterInterfaceVersion, "interface version %u", info.outInterfaceVersion);
  CHECK(filter.CreateInstance && filter.DisposeInstance && filter.GetFrameDependencies && filter.Precompute && filter.Render,
        "function table incomplete");
  csSDK_int32 second = 1;
  PrGPUFilter f2 = {}; PrGPUFilterInfo i2 = {};
  CHECK(entry(PrSDKGPUFilterInterfaceVersion, &second, kPrTrue, &g.suites, &f2, &i2) != suiteError_NoError, "index 1 must be refused");

  // UI values as Premiere would report them (float sliders as Float64, the
  // integer slider and popups as Int32). Radius 2.
  DefaultGPUParams();

  // A non-Metal device must be refused (Premiere then renders in software).
  PrGPUFilterInstance bad = {};
  bad.piSuites = &g.suites;
  gGPU.framework = PrGPUDeviceFramework_OpenCL;
  CHECK(filter.CreateInstance(&bad) != suiteError_NoError && !bad.ioPrivatePluginData, "OpenCL device must be refused");
  gGPU.framework = PrGPUDeviceFramework_Metal;

  TestDependencies(g);
  TestSlowMotion(g);
  if (!gReal) {
    TestGPUParams(g);
    TestGPU27Patterns(g);
    TestGPUAdvanced(g);
    TestRegistry(g);
    TestItemTiming(g);
    TestGPUIdentity(g);
  }
  TestGPUFailurePaths(g);
  if (gReal) {
    // Copies of one effect (one runtime ID) on two clips.
    RunGPULayers(g, "one ID, two clips (owner known)", true, true, 3000, true);
    RunGPULayers(g, "one ID, no owner clip", false, false, 1000, true);
    RunGPULayers(g, "one ID, one clip (control)", true, true, 1000, false);
  }
  TestRenderStats();

  SetNode(4242, 1000, 1100, 1000, 1100);
  PrGPUFilterInstance inst;
  if (!Create(g, inst)) { CHECK(false, "CreateInstance"); return; }
  CHECK(inst.outIsRealtime == kPrTrue, "outIsRealtime");

  PrGPUFilterRenderParams rp = {};
  rp.inRenderTicksPerFrame = kTPF;
  rp.inDownsampleFactorX = rp.inDownsampleFactorY = 1.f;
  rp.inRenderFieldType = prFieldsNone;
  const PrTime seqStart = 1000 * kTPF;  // the item sits 1000 frames into the sequence

  // ---- renders ----
  for (PrPixelFormat fmt : {PrPixelFormat_GPU_BGRA_4444_16f, PrPixelFormat_GPU_BGRA_4444_32f}) {
    const char* name = fmt == PrPixelFormat_GPU_BGRA_4444_32f ? "32f" : "16f";
    const int W = 131, H = 73, PAD = 24;
    rp.inClipTime = 50 * kTPF; rp.inSequenceTime = seqStart + 50 * kTPF;
    rp.inRenderWidth = W; rp.inRenderHeight = H;

    // In place, full neighbour list (what Premiere normally does).
    FakePPix cur = MakePPix(fmt, W, H, PAD, 1), ref = MakePPix(fmt, W, H, PAD, 1);
    FakePPix n[4] = {MakePPix(fmt, W, H, PAD, 2), MakePPix(fmt, W, H, PAD, 3), MakePPix(fmt, W, H, PAD, 4), MakePPix(fmt, W, H, PAD, 5)};
    PPixHand frames[5] = {(PPixHand)&cur, (PPixHand)&n[0], (PPixHand)&n[1], (PPixHand)&n[2], (PPixHand)&n[3]};
    PPixHand out = (PPixHand)&cur;
    e = filter.Render(&inst, &rp, frames, 5, &out);
    WaitForGPU();
    CHECK(e == suiteError_NoError && out == (PPixHand)&cur, "%s in-place render err 0x%x", name, (unsigned)e);
    if (!gReal) CHECK(SameBytes(cur, ref), "%s in-place: frame changed under a pass-through core", name);
    printf("  gpu %s in place, 4 neighbours: err=0x%x\n", name, (unsigned)e);

    // Separate output buffer.
    FakePPix dst = MakePPix(fmt, W, H, PAD, 77);
    out = (PPixHand)&dst;
    e = filter.Render(&inst, &rp, frames, 5, &out);
    WaitForGPU();
    CHECK(e == suiteError_NoError, "%s separate-output render err 0x%x", name, (unsigned)e);
    if (!gReal) {
      int bpp = fmt == PrPixelFormat_GPU_BGRA_4444_32f ? 16 : 8;
      size_t diff = 0;
      for (int y = 0; y < H; ++y)
        if (memcmp((char*)dst.buffer.contents + (size_t)y * dst.rowBytes, (char*)ref.buffer.contents + (size_t)y * ref.rowBytes, (size_t)W * bpp)) ++diff;
      CHECK(diff == 0, "%s separate output: %zu rows differ from the input", name, diff);
    }
    printf("  gpu %s separate output:        err=0x%x\n", name, (unsigned)e);

    // Short list (host gave fewer frames than asked), a null neighbour, a
    // neighbour of another size, and no output frame: none may crash or fail.
    out = (PPixHand)&cur;
    e = filter.Render(&inst, &rp, frames, 3, &out);
    CHECK(e == suiteError_NoError, "%s short frame list err 0x%x", name, (unsigned)e);
    FakePPix small = MakePPix(fmt, W / 2, H / 2, 0, 9);
    PPixHand odd[5] = {(PPixHand)&cur, nullptr, (PPixHand)&small, (PPixHand)&n[2], (PPixHand)&cur};
    e = filter.Render(&inst, &rp, odd, 5, &out);
    CHECK(e == suiteError_NoError, "%s odd neighbours err 0x%x", name, (unsigned)e);
    e = filter.Render(&inst, &rp, frames, 1, &out);
    CHECK(e == suiteError_NoError, "%s single frame err 0x%x", name, (unsigned)e);
    WaitForGPU();
    if (!gReal) CHECK(SameBytes(cur, ref), "%s: frame changed after edge-case renders", name);
    printf("  gpu %s short / odd / single:   ok\n", name);
  }

  // Half-resolution playback and an interlaced field: just must render.
  {
    FakePPix cur = MakePPix(PrPixelFormat_GPU_BGRA_4444_16f, 64, 36, 0, 1);
    PPixHand frames[1] = {(PPixHand)&cur};
    PPixHand out = (PPixHand)&cur;
    rp.inDownsampleFactorX = rp.inDownsampleFactorY = 0.5f;
    rp.inRenderFieldType = prFieldsUpperFirst;
    rp.inRenderField = pmFieldDisplay_ShowSecondField;
    SetI(OD_P_TEMPORAL_RADIUS, 0);
    e = filter.Render(&inst, &rp, frames, 1, &out);
    WaitForGPU();
    CHECK(e == suiteError_NoError, "half-res interlaced render err 0x%x", (unsigned)e);
    printf("  gpu half-res, second field:     err=0x%x\n", (unsigned)e);
    SetI(OD_P_TEMPORAL_RADIUS, 2);
  }

  // Real core: frames with independent noise, rendered in place and without
  // waiting, as Premiere does; each case with its four neighbours and again
  // with none (radius 0), on fresh noise and at its own time. The neighbours
  // must count, and Quality = Auto must follow the host.
  if (gReal) {
    const int W = kRealW, H = kRealH;
    rp.inDownsampleFactorX = rp.inDownsampleFactorY = 1.f;
    rp.inRenderFieldType = prFieldsNone;
    rp.inRenderWidth = W; rp.inRenderHeight = H;
    SetF(OD_P_LUMA, 100);
    struct Case { const char* label; int popup; PrRenderQuality host; };
    const Case cases[] = {{"Auto, host draft", OD_QUALITY_AUTO, kPrRenderQuality_Draft},
                          {"Auto, host high", OD_QUALITY_AUTO, kPrRenderQuality_High},
                          {"Fast", OD_QUALITY_FAST, kPrRenderQuality_High},
                          {"Full", OD_QUALITY_FULL, kPrRenderQuality_Draft}};
    double ratio[4][2] = {};
    int ci = 0;
    for (const Case& c : cases) {
      for (int pass = 0; pass < 2; ++pass) {
        const bool withNeighbours = pass == 0;
        FakePPix fr[5];
        for (int i = 0; i < 5; ++i) {
          fr[i] = MakePPix(PrPixelFormat_GPU_BGRA_4444_32f, W, H, 0, 1);
          FillNoisy((float*)fr[i].buffer.contents, (size_t)W * H, 100u + (unsigned)(ci * 20 + pass * 10 + i));
        }
        const double sIn = GreenSigma((const float*)fr[0].buffer.contents, W, H, W * 4);
        SetI(OD_P_QUALITY, c.popup);
        SetI(OD_P_TEMPORAL_RADIUS, withNeighbours ? 2 : 0);
        rp.inQuality = c.host;
        rp.inSequenceTime = seqStart + (10 + 9 * (ci * 2 + pass)) * kTPF;
        rp.inClipTime = rp.inSequenceTime + 3000 * kTPF;
        PPixHand frames[5] = {(PPixHand)&fr[0], (PPixHand)&fr[1], (PPixHand)&fr[2], (PPixHand)&fr[3], (PPixHand)&fr[4]};
        PPixHand out = (PPixHand)&fr[0];
        e = filter.Render(&inst, &rp, frames, withNeighbours ? 5 : 1, &out);
        WaitForGPU();
        const double sOut = GreenSigma((const float*)fr[0].buffer.contents, W, H, W * 4);
        CHECK(e == suiteError_NoError, "real core %s: err 0x%x", c.label, (unsigned)e);
        ratio[ci][pass] = sOut / sIn;
        printf("  gpu real core, %-16s %-13s: noise sigma %.5f -> %.5f (x%.3f)\n", c.label,
               withNeighbours ? "4 neighbours" : "no neighbours", sIn, sOut, sOut / sIn);
      }
      CHECK(ratio[ci][0] < 0.7 * ratio[ci][1], "real core %s: neighbours must count: x%.3f with, x%.3f without", c.label,
            ratio[ci][0], ratio[ci][1]);
      ++ci;
    }
    // Backstops, and Quality = Auto following the host.
    CHECK(ratio[0][0] < 0.15 && ratio[2][0] < 0.15, "real core: playback quality leaves x%.3f / x%.3f", ratio[0][0], ratio[2][0]);
    CHECK(ratio[1][0] < 0.09 && ratio[3][0] < 0.09, "real core: full quality leaves x%.3f / x%.3f", ratio[1][0], ratio[3][0]);
    CHECK(ratio[1][0] < 0.75 * ratio[0][0], "real core: Auto under a high-quality render (x%.3f) must be full quality, "
          "clearly below Auto under a draft (x%.3f)", ratio[1][0], ratio[0][0]);
    CHECK(std::fabs(ratio[1][0] - ratio[3][0]) < 0.1 * ratio[3][0], "real core: Auto, host high x%.3f vs Full x%.3f",
          ratio[1][0], ratio[3][0]);
    CHECK(std::fabs(ratio[0][0] - ratio[2][0]) < 0.1 * ratio[2][0], "real core: Auto, host draft x%.3f vs Playback x%.3f",
          ratio[0][0], ratio[2][0]);
    // An advanced control crosses the whole GPU path (parameter numbering,
    // ToCoreParams, the bands step): Luma fine amount 0 % leaves the fine
    // luma noise, which is most of white noise.
    {
      double left[2] = {0, 0};
      for (int k = 0; k < 2; ++k) {
        FakePPix fr[5];
        for (int i = 0; i < 5; ++i) {
          fr[i] = MakePPix(PrPixelFormat_GPU_BGRA_4444_32f, W, H, 0, 1);
          FillNoisy((float*)fr[i].buffer.contents, (size_t)W * H, 700u + (unsigned)(k * 10 + i));
        }
        const double sIn = GreenSigma((const float*)fr[0].buffer.contents, W, H, W * 4);
        SetI(OD_P_QUALITY, OD_QUALITY_FULL);
        SetI(OD_P_TEMPORAL_RADIUS, 2);
        SetF(OD_P_LUMA_AMOUNT_FINE, k ? 0 : 100);
        rp.inQuality = kPrRenderQuality_Draft;
        rp.inSequenceTime = seqStart + (90 + 9 * k) * kTPF;
        rp.inClipTime = rp.inSequenceTime + 3000 * kTPF;
        PPixHand frames[5] = {(PPixHand)&fr[0], (PPixHand)&fr[1], (PPixHand)&fr[2], (PPixHand)&fr[3], (PPixHand)&fr[4]};
        PPixHand out = (PPixHand)&fr[0];
        e = filter.Render(&inst, &rp, frames, 5, &out);
        WaitForGPU();
        CHECK(e == suiteError_NoError, "real core, luma fine amount: err 0x%x", (unsigned)e);
        left[k] = GreenSigma((const float*)fr[0].buffer.contents, W, H, W * 4) / sIn;
      }
      printf("  gpu real core, Luma fine amount 100 %% / 0 %%: noise left x%.3f / x%.3f\n", left[0], left[1]);
      CHECK(left[1] > 2 * left[0], "real core: Luma fine amount 0 %% left x%.3f of the noise, 100 %% x%.3f", left[1],
            left[0]);
    }
    // The experimental controls cross the GPU path as well. Grain stability
    // 100 % at the default luma amount (80 %): the grain put back is taken
    // from a reference whose coarse part is the temporal merge's, so less
    // noise is left than at 0 %. Coarse chroma cleanup on: the estimator and
    // the correction run (frames of 640x360, a +/-2 window) and the render
    // succeeds with finite output (on independent noise the estimate does
    // not qualify, so the picture is not compared here: tests/profile).
    {
      double left[3] = {0, 0, 0};
      for (int k = 0; k < 3; ++k) {
        FakePPix fr[5];
        for (int i = 0; i < 5; ++i) {
          fr[i] = MakePPix(PrPixelFormat_GPU_BGRA_4444_32f, W, H, 0, 1);
          FillNoisy((float*)fr[i].buffer.contents, (size_t)W * H, 900u + (unsigned)(k * 10 + i));
        }
        const double sIn = GreenSigma((const float*)fr[0].buffer.contents, W, H, W * 4);
        DefaultGPUParams();
        SetI(OD_P_QUALITY, OD_QUALITY_FULL);
        SetF(OD_P_GRAIN_STABILITY, k == 1 ? 100 : 0);
        SetB(OD_P_COARSE_CHROMA, k == 2);
        rp.inQuality = kPrRenderQuality_Draft;
        rp.inSequenceTime = seqStart + (120 + 9 * k) * kTPF;
        rp.inClipTime = rp.inSequenceTime + 3000 * kTPF;
        PPixHand frames[5] = {(PPixHand)&fr[0], (PPixHand)&fr[1], (PPixHand)&fr[2], (PPixHand)&fr[3], (PPixHand)&fr[4]};
        PPixHand out = (PPixHand)&fr[0];
        e = filter.Render(&inst, &rp, frames, 5, &out);
        WaitForGPU();
        CHECK(e == suiteError_NoError, "real core, experimental controls %d: err 0x%x", k, (unsigned)e);
        const float* px = (const float*)fr[0].buffer.contents;
        bool finite = true;
        for (size_t i = 0; i < (size_t)W * H * 4 && finite; ++i) finite = std::isfinite(px[i]);
        CHECK(finite, "real core, experimental controls %d: output not finite", k);
        left[k] = GreenSigma(px, W, H, W * 4) / sIn;
      }
      printf("  gpu real core, Grain stability 0 %% / 100 %%: noise left x%.3f / x%.3f; Coarse chroma cleanup on: x%.3f\n",
             left[0], left[1], left[2]);
      CHECK(left[1] < 0.95 * left[0], "real core: Grain stability 100 %% left x%.3f of the noise, 0 %% x%.3f", left[1],
            left[0]);
    }
    DefaultGPUParams();
    rp.inQuality = kPrRenderQuality_Draft;
  }

  // Concurrency: Premiere may run separate instances at the same time; we
  // also hammer one instance from two threads to prove the session lock.
  {
    SetI(OD_P_TEMPORAL_RADIUS, 1);
    SetNode(4243, 1000, 1100, 1000, 1100);
    PrGPUFilterInstance inst2;
    CHECK(Create(g, inst2), "second instance");
    PrGPUFilterInstance* insts[4] = {&inst, &inst, &inst2, &inst2};
    __block int errors = 0;
    PrGPUFilterRenderParams base = rp;
    base.inDownsampleFactorX = base.inDownsampleFactorY = 1.f;
    base.inRenderFieldType = prFieldsNone;
    PrGPUFilter* fp = &filter;
    PrGPUFilterInstance** ip = insts;
    dispatch_apply(4, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t t) {
      @autoreleasepool {
        FakePPix a = MakePPix(PrPixelFormat_GPU_BGRA_4444_16f, 96, 54, 8, (unsigned)t + 1);
        FakePPix b = MakePPix(PrPixelFormat_GPU_BGRA_4444_16f, 96, 54, 8, (unsigned)t + 11);
        FakePPix c = MakePPix(PrPixelFormat_GPU_BGRA_4444_16f, 96, 54, 8, (unsigned)t + 21);
        for (int i = 0; i < 25; ++i) {
          PrGPUFilterRenderParams r = base;
          r.inSequenceTime = seqStart + (20 + i) * kTPF;
          r.inClipTime = r.inSequenceTime;
          PPixHand fr[3] = {(PPixHand)&a, (PPixHand)&b, (PPixHand)&c};
          PPixHand o = (PPixHand)&a;
          if (fp->Render(ip[t], &r, fr, 3, &o) != suiteError_NoError) __sync_fetch_and_add(&errors, 1);
        }
      }
    });
    WaitForGPU();
    CHECK(errors == 0, "%d concurrent renders failed", errors);
    printf("  gpu 4 threads x 25 renders on 2 instances: errors=%d\n", errors);
    CHECK(filter.DisposeInstance(&inst2) == suiteError_NoError, "dispose second instance");
    DefaultGPUParams();
  }

  e = filter.DisposeInstance(&inst);
  CHECK(e == suiteError_NoError && !inst.ioPrivatePluginData, "DisposeInstance");

  // Shutdown with a session still parked: it is destroyed, then the context.
  if (!gReal) {
    SetNode(7001, 1000, 1100, 1000, 1100);
    PrGPUFilterInstance last;
    if (Create(g, last)) {
      RecordedRender(g, last, 1050, kPrRenderQuality_Draft);
      g.filter.DisposeInstance(&last);
    }
  }
  Take();
  index = 0;
  e = entry(PrSDKGPUFilterInterfaceVersion, &index, kPrFalse, &g.suites, &filter, &info);
  CHECK(e == suiteError_NoError, "shutdown err 0x%x", (unsigned)e);
  if (!gReal) {
    std::vector<Rec> recs = Take();
    CHECK(CountOf(recs, "destroy") >= 1 && CountOf(recs, "ctxdestroy") == 1,
          "shutdown: parked sessions destroyed (%d) and the context released (%d)", CountOf(recs, "destroy"),
          CountOf(recs, "ctxdestroy"));
  }
  CHECK(gAcquired == gReleased, "suites acquired %d released %d", gAcquired.load(), gReleased.load());
}

// ---- global teardown and unloading ------------------- The host may unload the bundle
// after its global teardown, while work the core queued (a Metal completion handler, a noise
// estimate) still has to run code of this image. Run in a child process (a crash must not
// take the harness down): a render is queued behind GPU work the harness holds back, the
// instance goes, the host shuts down and dlcloses the bundle, then the GPU runs. (A)
// Released 150 ms in: the teardown waits for it (bounded quiescence), then the host dlcloses
// the bundle (once, as it opened it once): the image stays mapped (pinned). (B) The plug-in
// started again and the GPU held past the teardown's bound (OD_TEARDOWN_MS=300): the
// teardown returns, and the late completion runs safely in the pinned image. Before, the
// teardown did not wait and dlclose unmapped the image: the completion crashed the process.
static double MonoMs() { return (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1e6; }
extern char** environ;

static int UnloadChild(const char* path) {
  alarm(30);
  void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    printf("  unload child: dlopen failed: %s\n", dlerror());
    return 2;
  }
  EffectMainFn effectMain = (EffectMainFn)dlsym(handle, "EffectMain");
  PrGPUFilterEntry entry = (PrGPUFilterEntry)dlsym(handle, PrGPUFilterEntryPointName);
  if (!effectMain || !entry) return 2;
  gGPU.device = MTLCreateSystemDefaultDevice();
  if (!gGPU.device) return 0;  // nothing to test without Metal
  gGPU.queue = [gGPU.device newCommandQueue];
  PF_InData in_data = {};
  PF_OutData out_data = {};
  in_data.appl_id = kAppID_Premiere;
  in_data.pica_basicP = &gBasic;
  in_data.inter.add_param = FakeAddParam;
  in_data.inter.register_ui = RegisterCustomUI;
  effectMain(PF_Cmd_GLOBAL_SETUP, &in_data, &out_data, nullptr, nullptr, nullptr);
  GPUEnv g;
  g.util.getSPBasicSuite = GetBasic;
  g.suites.utilFuncs = &g.util;
  const int W = kRealW, H = kRealH;
  FakePPix fr[5];
  for (int i = 0; i < 5; ++i) {
    fr[i] = MakePPix(PrPixelFormat_GPU_BGRA_4444_32f, W, H, 0, 1);
    FillNoisy((float*)fr[i].buffer.contents, (size_t)W * H, 300u + (unsigned)i);
  }
  bool closed = false;
  auto phase = [&](const char* name, double releaseMs, double* teardownMs) -> bool {
    PrGPUFilterInfo info = {};
    csSDK_int32 index = 0;
    if (entry(PrSDKGPUFilterInterfaceVersion, &index, kPrTrue, &g.suites, &g.filter, &info) != suiteError_NoError)
      return false;
    DefaultGPUParams();
    SetNode(9950, 1000, 1100, 1000, 1100);
    PrGPUFilterInstance inst;
    if (!Create(g, inst)) return false;
    PrGPUFilterRenderParams rp = {};
    rp.inRenderTicksPerFrame = kTPF;
    rp.inDownsampleFactorX = rp.inDownsampleFactorY = 1.f;
    rp.inSequenceTime = 1050 * kTPF;
    rp.inClipTime = 4050 * kTPF;
    rp.inQuality = kPrRenderQuality_Draft;
    PPixHand frames[5] = {(PPixHand)&fr[0], (PPixHand)&fr[1], (PPixHand)&fr[2], (PPixHand)&fr[3], (PPixHand)&fr[4]};
    PPixHand out = (PPixHand)&fr[0];
    // Version 2 scans its available window before returning the first result.
    // Finish that scan, then hold later asynchronous work across teardown.
    if (g.filter.Render(&inst, &rp, frames, 5, &out) != suiteError_NoError) {
      g.filter.DisposeInstance(&inst);
      return false;
    }
    WaitForGPU();
    rp.inSequenceTime += kTPF;
    rp.inClipTime += kTPF;
    id<MTLSharedEvent> ev = [gGPU.device newSharedEvent];
    id<MTLCommandBuffer> hold = [gGPU.queue commandBuffer];
    [hold encodeWaitForEvent:ev value:1];
    [hold commit];
    const prSuiteError e = g.filter.Render(&inst, &rp, frames, 5, &out);
    g.filter.DisposeInstance(&inst);
    std::thread releaser;
    if (releaseMs > 0)
      releaser = std::thread([ev, releaseMs] {
        usleep((useconds_t)(releaseMs * 1000));
        ev.signaledValue = 1;
      });
    const double t0 = MonoMs();
    index = 0;
    entry(PrSDKGPUFilterInterfaceVersion, &index, kPrFalse, &g.suites, &g.filter, &info);  // the GPU side's shutdown
    effectMain(PF_Cmd_GLOBAL_SETDOWN, &in_data, &out_data, nullptr, nullptr, nullptr);
    *teardownMs = MonoMs() - t0;
    if (!closed) dlclose(handle);  // what a host unloading the plug-in does (once: it opened it once)
    closed = true;
    if (releaser.joinable()) releaser.join();
    ev.signaledValue = 1;  // (B): only now
    WaitForGPU();
    usleep(200000);  // completion handlers and the noise estimates queued after them
    printf("  unload child, %s: render err 0x%x, teardown %.1f ms\n", name, (unsigned)e, *teardownMs);
    return e == suiteError_NoError;
  };
  double tA = 0, tB = 0;
  const bool a = phase("GPU released after 150 ms", 150, &tA);
  void* still = dlopen(path, RTLD_NOLOAD | RTLD_LAZY);
  const bool pinned = still != nullptr;
  if (still) dlclose(still);
  const bool b = a && pinned && phase("GPU held past the teardown's bound", 0, &tB);
  int fails = 0;
  if (!a || !b) { printf("FAIL unload child: a phase did not run\n"); ++fails; }
  if (!pinned) { printf("FAIL unload child: dlclose unmapped the image (not pinned)\n"); ++fails; }
  if (gReal && !(tA >= 120)) { printf("FAIL unload child: the teardown returned after %.1f ms, before the GPU work it had to wait for (released at 150 ms)\n", tA); ++fails; }
  if (!(tB < 1500)) { printf("FAIL unload child: the teardown took %.1f ms with the GPU held (bound 300 ms)\n", tB); ++fails; }
  printf("  unload child: %s, image %s after dlclose, teardown %.0f ms (GPU released at 150) / %.0f ms (held)\n",
         fails ? "FAILED" : "ok", pinned ? "mapped" : "unmapped", tA, tB);
  return fails ? 1 : 0;
}

// Runs UnloadChild in a child process; its exit status is the check.
static void TestUnload(const char* self, const char* plugin) {
  std::vector<std::string> env;
  for (char** e = environ; *e; ++e)
    if (strncmp(*e, "OD_TEARDOWN_MS=", 15)) env.push_back(*e);
  env.push_back("OD_TEARDOWN_MS=300");
  std::vector<char*> envp;
  for (std::string& e : env) envp.push_back((char*)e.c_str());
  envp.push_back(nullptr);
  std::vector<char*> args = {(char*)self, (char*)plugin, (char*)"--unload-child"};
  if (gReal) args.push_back((char*)"--real");
  args.push_back(nullptr);
  pid_t pid = 0;
  const int rc = posix_spawn(&pid, self, nullptr, nullptr, args.data(), envp.data());
  CHECK(rc == 0, "unload: posix_spawn failed (%d)", rc);
  if (rc != 0) return;
  int status = 0;
  waitpid(pid, &status, 0);
  CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0, "unload: the child %s %d (a crash after dlclose means work of "
        "the image ran after it was unmapped)", WIFSIGNALED(status) ? "died of signal" : "exited with",
        WIFSIGNALED(status) ? WTERMSIG(status) : WEXITSTATUS(status));
}

#include "capture_tests.h"
#include "panel_tests.h"
#include "picture_pin.h"
#include "analysis_panel_tests.h"

int main(int argc, char** argv) {
  alarm(60);  // hard deadline
  setvbuf(stdout, nullptr, _IOLBF, 0);
  if (argc < 2) {
    fprintf(stderr, "usage: %s <plugin binary> [--expect-passthrough|--real]\n", argv[0]);
    return 2;
  }
  bool unloadChild = false;
  const char* captureLoad = nullptr;
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--real")) gReal = true;
    if (!strcmp(argv[i], "--capture-load") && i+1 < argc) captureLoad = argv[++i];
    if (!strcmp(argv[i], "--expect-passthrough")) gPassthrough = true;
    if (!strcmp(argv[i], "--unload-child")) unloadChild = true;
  }
  if (getenv("OD_STUB_FAIL")) gPassthrough = true;
  // The stub core's record goes next to the harness binary.
  gRecordPath = std::string(argv[0]) + (captureLoad ? ".capture-child.record" : unloadChild ? ".unload.record" : gPassthrough ? ".fail.record" : ".record");
  unlink(gRecordPath.c_str());
  setenv("OD_STUB_RECORD", gRecordPath.c_str(), 1);
  setenv("OD_PARK_MS", "300", 1);  // read by the plug-in once, on first use
  setenv("OD_CPU_IDLE_MS", "300", 1);  // CPU buffer pool idle release, also read once

  gBasic.AcquireSuite = FakeAcquire;
  gBasic.ReleaseSuite = FakeRelease;
  gPixelFormatSuite.AddSupportedPixelFormat = PFAddFormat;
  gPixelFormatSuite.ClearSupportedPixelFormats = PFClearFormats;
  gPixelFormatSuite.GetPixelFormat = PFGetFormat;
  gUtilitySuite.GetFilterInstanceID = UtilInstanceID;
  gUtilitySuite.GetContainingTimelineID = UtilTimeline;
  gUtilitySuite.GetTrackItemStart = UtilItemStart;
  gUtilitySuite.GetClipStart = UtilClipStart;
  gUtilitySuite.GetClipDuration = UtilClipDuration;
  gUtilitySuite.GetClipSpeed = UtilClipSpeed;
  gUtilitySuite.EffectWantsCheckedOutFramesToMatchRenderPixelFormat = UtilWantsMatch;
  gDeviceSuite.GetDeviceInfo = DevGetInfo;
  gDeviceSuite.GetGPUPPixData = DevGetData;
  gDeviceSuite.CreateGPUPPix = DevCreatePPix;
  gPPixSuite.Dispose = PPixDispose;
  gPPixSuite.GetBounds = PPixGetBounds;
  gPPixSuite.GetRowBytes = PPixGetRowBytes;
  gPPixSuite.GetPixelFormat = PPixGetFormat;
  gPPixSuite.GetPixelAspectRatio = PPixGetPAR;
  gSegmentSuite.GetParam = SegGetParam;
  gSegmentSuite.GetParamCount = SegGetParamCount;
  gSegmentSuite.GetNodeProperty = SegGetNodeProperty;
  gSegmentSuite.AcquireOperatorOwnerNodeID = SegAcquireOwner;
  gSegmentSuite.AcquireInputNodeID = SegAcquireInput;
  gSegmentSuite.ReleaseVideoNodeID = SegReleaseNode;
  gMemorySuite.PrDisposePtr = MemDisposePtr;

  if (unloadChild) {
    @autoreleasepool {
      return UnloadChild(argv[1]);
    }
  }
  void* handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 2;
  }
  // The plug-in's log, for the tests that read it (PluginLogFrom), in a file
  // next to the record: OD_LOG_STDERR prints every line to stderr too.
  gPluginLogPath = gRecordPath + ".log";
  setenv("OD_LOG_STDERR", "1", 1);
  fflush(stderr);
  {
    const int fd = open(gPluginLogPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_APPEND, 0644);
    if (fd >= 0) {
      dup2(fd, 2);
      close(fd);
    } else {
      gPluginLogPath.clear();
    }
  }
  EffectMainFn effectMain = (EffectMainFn)dlsym(handle, "EffectMain");
  PrGPUFilterEntry gpuEntry = (PrGPUFilterEntry)dlsym(handle, PrGPUFilterEntryPointName);
  CHECK(effectMain != nullptr, "EffectMain not exported");
  CHECK(gpuEntry != nullptr, "%s not exported", PrGPUFilterEntryPointName);
  if (!effectMain || !gpuEntry) return 1;

  @autoreleasepool {
    if (captureLoad) return CaptureReopenChild(effectMain, gpuEntry, captureLoad);
    TestCPU(effectMain);
    TestAnalysisPanel(effectMain);
    TestGPUAnalysisPanel(effectMain,gpuEntry);
    TestReservedParam(effectMain);
    TestGPU(gpuEntry);
    TestCapture(effectMain, gpuEntry, argv[0], argv[1]);
    TestPicturePin(effectMain, gpuEntry);
    TestPanel(effectMain,gpuEntry);
    TestUnload(argv[0], argv[1]);
  }
  if (!gReal) {
    // Every session the plug-in created was destroyed by the end.
    gRecordOffset = 0;
    std::vector<Rec> all = NewRecords();
    CHECK(CountOf(all, "create") == CountOf(all, "destroy"), "sessions: %d created, %d destroyed",
          CountOf(all, "create"), CountOf(all, "destroy"));
  }
  printf("%s: %d checks, %d failed\n", gFailures ? "FAILED" : "PASSED", gChecks.load(), gFailures.load());
  return gFailures ? 1 : 0;
}
