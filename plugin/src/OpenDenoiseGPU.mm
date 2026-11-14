// SPDX-License-Identifier: MIT
// Open Denoise — Premiere Pro GPU filter (PrGPUFilter, Metal).
//
// Premiere calls xGPUFilterEntry at start-up when Mercury GPU acceleration is
// on, then CreateInstance for each effect on a track item (again each time a
// parameter changes), GetFrameDependencies and Render per frame.
//
// Time base. GetParam takes media time and the SDK samples pass inClipTime
// there, so inClipTime is not 0 at the track item's first frame and does not
// advance one frame per sequence frame under a speed change. Neighbours are
// therefore requested, and kept inside the track item, in sequence time: the
// item's bounds come from its clip node (ClipNode::EffectiveTrackItemStart /
// EndAsTicks, documented as sequence time). Without them every neighbour up to
// the radius is requested; the checks in Render and the core's cut detector
// deal with a frame from outside the item; nothing before sequence time 0 is
// ever requested. Frame keys are built from sequence
// time and the item's start, never from inClipTime: under a speed change,
// reverse, time remapping or a frame hold, sequence time minus clip time moves
// on every frame, and keys built from it never matched between one render and
// the next, so the core took every render for a jump. What the host passes is
// logged for the first renders of each instance (and its first high-quality
// render), so a run in Premiere settles the rest.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "AEConfig.h"
#include "PrSDKTypes.h"
#include "AE_Effect.h"
#include "PrSDKAESupport.h"
#include "PrSDKGPUDeviceSuite.h"
#include "PrSDKGPUFilter.h"
#include "PrSDKMemoryManagerSuite.h"
#include "PrSDKPPixSuite.h"
#include "PrSDKPPix2Suite.h"
#include "PrSDKVideoSegmentSuite.h"
#include "PrSDKVideoSegmentProperties.h"

#include <cmath>
#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "OpenDenoisePlugin.h"
#include "RenderStats.h"
#include "OpenDenoiseCapture.h"
#include <numeric>

using namespace odplug;

namespace {

template <typename T>
struct SuiteRef {
  SPBasicSuite* basic = nullptr;
  const char* name = nullptr;
  int32_t version = 0;
  const T* suite = nullptr;
  bool Acquire(SPBasicSuite* b, const char* n, int32_t v) {
    basic = b; name = n; version = v;
    const void* p = nullptr;
    if (!b || b->AcquireSuite(n, v, &p) != kSPNoError || !p) return false;
    suite = (const T*)p;
    return true;
  }
  ~SuiteRef() { if (suite && basic) basic->ReleaseSuite(name, version); }
  const T* operator->() const { return suite; }
  explicit operator bool() const { return suite != nullptr; }
};

double ParamAsDouble(const PrParam& p, double fallback) {
  switch (p.mType) {
    case kPrParamType_Int8: return p.mInt8;
    case kPrParamType_Int16: return p.mInt16;
    case kPrParamType_Int32: return p.mInt32;
    case kPrParamType_Int64: return (double)p.mInt64;
    case kPrParamType_Float32: return p.mFloat32;
    case kPrParamType_Float64: return p.mFloat64;
    case kPrParamType_Bool: return p.mBool ? 1 : 0;
    default: return fallback;
  }
}

bool IsIntegerType(PrParamType t) {
  return t == kPrParamType_Int8 || t == kPrParamType_Int16 || t == kPrParamType_Int32 || t == kPrParamType_Int64;
}

// ---- Parameter layout on the GPU side ------------------------------------------ Effect
// Controls order is kParamOrder (OpenDenoisePlugin.h): seven topics (eight in the compile-
// time fallback), with controls of the nine original ones inside some of them. A topic's
// start and end entries are parameters with PF indices of their own on the After Effects
// side; whether PrSDKVideoSegmentSuite::GetParam counts them too is not documented (SDK
// headers and guide). Only the entries before the first topic (Look, the two reductions,
// grain, threshold and frames) have the same GPU index either way, so the instance asks
// GetParamCount once, at creation, and every control goes through the layout: 106 (107 with
// the input): WithTopics, GPU index = pf - 1 92 (93 with the input): WithoutTopics, GPU
// index = pf - 1 - topic entries before pf fallback WithTopics: 108/109; WithoutTopics
// remains 92/93. anything else, or no GetParamCount: Unknown These counts include the saved
// profile's thirteen entries. Whether Premiere counts buttons, hidden and arbitrary-data
// parameters as the fake host does has not been observed; if it does not, the count is
// unknown and the instance declines (software). The counts of earlier builds are not meant:
// GetParamCount counts the parameters this binary registered in PF_Cmd_PARAMS_SETUP, which a
// project saved with fewer does not change (the host fills the missing ones, on both paths),
// so an old count cannot come from this build's definition. Unknown declines the instance
// and the software path, which reads every control by its disk ID, renders it. and then
// reads every render control once (a value of the wrong type, a slider not Float32/64 or a
// checkbox not Bool or an integer, or a GetParam error: the numbering is not the one the
// count implies). An Unknown or failed layout declines the instance: "Returning an error
// from CreateInstance will cause this node to be rendered in software for the current set of
// parameters" (PrSDKGPUFilter.h, SDK guide 12.8.1), and the software path reads every
// control by its PF index. The editor's settings are then rendered, slower, instead of being
// silently replaced by defaults. A read that fails later, in a render, cannot decline (no
// fallback in the middle of a render, guide 12.2.4): that render takes the advanced controls
// at their defaults, logged and counted in the stats lines ("advanced controls unread"), and
// the next render reads them again. Nothing outlives that render: a wrong numbering puts the
// same index on the same parameter type at every time, so each new instance's own creation
// probe declines it; a failure that got past that probe is transient or depends on the time.
enum class ParamLayout { Unknown = 0, WithTopics = 1, WithoutTopics = 2 };
constexpr int kCountWithTopics = kNumParams - 1;  // the input layer is not a GPU parameter
constexpr int kCountWithoutTopics = kCountWithTopics - kNumTopicEntries;
static_assert(kCountWithTopics == (kUseDynamicVisibility ? 106 : 108) && kCountWithoutTopics == 92, "parameter counts of the GPU layout");

const char* LayoutName(ParamLayout l) {
  return l == ParamLayout::WithTopics ? "WithTopics" : l == ParamLayout::WithoutTopics ? "WithoutTopics" : "Unknown";
}
ParamLayout LayoutForCount(int n) {
  if (n == kCountWithTopics || n == kCountWithTopics + 1) return ParamLayout::WithTopics;
  if (n == kCountWithoutTopics || n == kCountWithoutTopics + 1) return ParamLayout::WithoutTopics;
  return ParamLayout::Unknown;
}

// GetParam index of the parameter at PF index pf, or -1 when it cannot be read.
int GpuIndex(ParamLayout l, int pf) {
  if (pf < 1 || pf >= kNumParams) return -1;
  if (TopicEntriesBefore(pf) == 0 && !IsTopicEntry(IdAtPf(pf))) return pf - 1;  // before every topic
  if (l == ParamLayout::WithTopics) return pf - 1;
  if (l != ParamLayout::WithoutTopics) return -1;
  return pf - 1 - TopicEntriesBefore(pf);
}
// The same, by disk ID.
int GpuIndexOf(ParamLayout l, int id) { return GpuIndex(l, PfIndex(id)); }

struct FrameInfo {
  void* buffer = nullptr;  // id<MTLBuffer>, unretained
  PrPixelFormat format = PrPixelFormat_Invalid;
  uint32_t width = 0, height = 0;
  int32_t rowBytes = 0;
};

double NowMs() { return (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1e6; }

// ---- Per-render instrumentation: RenderStats.h ---------------------------------

std::atomic<int> gActiveRenders{0};
std::atomic<uint64_t> gRenderSerial{0};  // RenderSample::id

std::atomic<int> gLifecycleLogged{0};  // instances whose creation and disposal were logged
// Diagnostic budgets per process: instances whose lifecycle (and missing
// bounds) is logged, and time-base lines (LogTimeBase). os_log lines cost
// microseconds; the budgets only keep a long session from flooding the log.
constexpr int kLogInstances = 1024;
constexpr int kLogTimeBase = 2048;

class ODFilter {
 public:
  prSuiteError Initialize(PrGPUFilterInstance* inst) {
    if (!inst || !inst->piSuites || !inst->piSuites->utilFuncs) return suiteError_Fail;
    SPBasicSuite* basic = inst->piSuites->utilFuncs->getSPBasicSuite();
    mDeviceIndex = inst->inDeviceIndex;
    mNodeID = inst->inNodeID;

    if (!mDevice.Acquire(basic, kPrSDKGPUDeviceSuite, kPrSDKGPUDeviceSuiteVersion) ||
        !mPPix.Acquire(basic, kPrSDKPPixSuite, kPrSDKPPixSuiteVersion) ||
        !mSegment.Acquire(basic, kPrSDKVideoSegmentSuite, kPrSDKVideoSegmentSuiteVersion)) {
      Log("GPU: required suite missing");
      return suiteError_Fail;
    }
    mPPix2.Acquire(basic, kPrSDKPPix2Suite, kPrSDKPPix2SuiteVersion);      // optional
    mMemory.Acquire(basic, kPrSDKMemoryManagerSuite, kPrSDKMemoryManagerSuiteVersion);  // optional

    PrGPUDeviceInfo info = {};
    if (mDevice->GetDeviceInfo(kPrSDKGPUDeviceSuiteVersion, mDeviceIndex, &info) != suiteError_NoError)
      return suiteError_Fail;
    // Returning an error makes Premiere render this node in software, which
    // lands in OpenDenoiseCPU.mm.
    if (info.outDeviceFramework != PrGPUDeviceFramework_Metal || !info.outDeviceHandle) {
      Log("GPU: device %u is not Metal; using the software path", mDeviceIndex);
      return suiteError_Fail;
    }
    mMTLDevice = info.outDeviceHandle;
    mMTLQueue = info.outCommandQueueHandle;

    char err[256];
    ODContext* ctx = ContextFor(mMTLDevice, mMTLQueue, err, sizeof err);
    if (!ctx) {
      Log("GPU: od_context_create failed: %s", err);
      return suiteError_Fail;
    }

    // Where the track item is, in sequence time (see the top of this file),
    // and which item it is (track, start): read once, at creation.
    int64_t v = 0;
    if (ReadIntProperty(mNodeID, kVideoSegmentProperty_Effect_EffectDuration, &v)) mDuration = v;
    mItem.duration = mDuration;
    mOwnerResult = -1;
    if (mSegment->AcquireOperatorOwnerNodeID && mSegment->ReleaseVideoNodeID) {
      csSDK_int32 owner = 0;
      mOwnerResult = (int)mSegment->AcquireOperatorOwnerNodeID(mNodeID, &owner);
      if (mOwnerResult == (int)suiteError_NoError) {
        int64_t s = 0, e = 0, ps = 0, pe = 0, track = 0;
        const bool plain = ReadIntProperty(owner, kVideoSegmentProperty_Clip_TrackItemStartAsTicks, &ps) &&
                           ReadIntProperty(owner, kVideoSegmentProperty_Clip_TrackItemEndAsTicks, &pe);
        if (ReadIntProperty(owner, kVideoSegmentProperty_Clip_EffectiveTrackItemStartAsTicks, &s) &&
            ReadIntProperty(owner, kVideoSegmentProperty_Clip_EffectiveTrackItemEndAsTicks, &e)) {
          mBoundsKind = "effective track item";
        } else if (plain) {
          s = ps;
          e = pe;
          mBoundsKind = "track item";
        }
        if (mBoundsKind && e > s) {
          mBounds.known = true;
          mBounds.start = s;
          mBounds.end = e;
        }
        if (mBoundsKind) {
          // The item's own start (not moved by transitions at its head),
          // else the effective one. The session identity needs the track as
          // well (EffectIdentity); without it the session is private.
          mItem.known = true;
          mItem.start = plain ? ps : s;
          if (ReadIntProperty(owner, kVideoSegmentProperty_Clip_TrackID, &track)) {
            mItem.trackID = (int32_t)track;
            mItem.trackKnown = true;
          }
          ReadTiming(owner);
        }
        mSegment->ReleaseVideoNodeID(owner);
      }
    }
    if (!mBounds.known) {
      static std::atomic<int> logged{0};
      if (logged.fetch_add(1) < kLogInstances)
        Log("GPU: node %d, timeline %d: no track item bounds (owner node lookup 0x%x, %s); requesting every neighbour "
            "up to the radius", (int)mNodeID, (int)inst->inTimelineID, (unsigned)mOwnerResult,
            mBoundsKind ? "empty range" : "no bounds properties");
    }

    // The effect's identity on its track item, which survives the instance:
    // the SDK guide gives EffectNode::RuntimeInstanceID as the GPU-side
    // equivalent of GetFilterInstanceID(), "unique during a session". The
    // session (noise profile, texture pool) is shared through it by the
    // instances Premiere creates for each parameter change. Copies of one
    // effect pasted onto several clips share the runtime ID, so the track
    // item (track, start) is part of the identity (EffectIdentity). Without a
    // runtime ID the instance gets a session of its own.
    mRuntimeID = ReadIntProperty(mNodeID, kVideoSegmentProperty_Effect_RuntimeInstanceID, &v) && v > 0 &&
                         v <= (int64_t)UINT32_MAX
                     ? (uint32_t)v
                     : 0;
    mTimelineID = inst->inTimelineID;
    // Where the advanced controls are (see "Parameter layout" above).
    if (mSegment->GetParamCount) {
      csSDK_int32 n = 0;
      mParamCountResult = (int)mSegment->GetParamCount(mNodeID, &n);
      if (mParamCountResult == (int)suiteError_NoError) mParamCount = n;
    }
    const ParamLayout layout = LayoutForCount(mParamCount);
    mLayout = layout;
    if (layout == ParamLayout::Unknown) {
      static std::atomic<int> logged{0};
      if (logged.fetch_add(1) < kLogInstances)
        Log("GPU: node %d: parameter count %d (expected %d or %d; GetParamCount %s): the advanced controls cannot be "
            "found on the GPU side; declining the instance, Premiere renders it in software (logged for the first %d "
            "instances)", (int)mNodeID, mParamCount, kCountWithTopics, kCountWithoutTopics,
            !mSegment->GetParamCount ? "missing" : mParamCountResult ? "failed" : "returned it", kLogInstances);
      return suiteError_Fail;
    }
    {
      // Every advanced control once, as Render will read them: the media
      // in-point is a clip time of this item (GetParam takes clip time).
      bool unread = false;
      const UIParams probe = ReadParams(mMediaIn, &unread, /*atCreation*/ true);
      if (unread) return suiteError_Fail;  // logged by ReadParams
      // Minimum grain applies to cannot vary over time and a parameter
      // change makes a new instance: what the probe read holds for every
      // render of this one, including one whose advanced read fails.
      mGrainMode = probe.grainMode;
      mControlSet = EffectiveControlSet(probe);
      mProcessing = probe.processing;
    }
    // Do not pass mCapture as the output: ReadSavedProfile also consults the
    // instance's immutable pin when a host has released its arbitrary handle.
    std::shared_ptr<const CaptureSnapshot> initialCapture;
    if (!ReadSavedProfile(mMediaIn, initialCapture, true)) {
      Log("GPU: exact saved profile unavailable; declining instance for software rendering");
      return suiteError_Fail;
    }
    mCapture = std::move(initialCapture);
    mRequestToken = RequestToken(mMediaIn);
    const uint64_t identity = EffectIdentity(inst->inTimelineID, mRuntimeID, mItem);
    mSession = AcquireSharedSession(ctx, identity);
    if (!mSession) {
      Log("GPU: od_session_create failed");
      return suiteError_Fail;
    }
    // Together with mSession: the destructor gives the registry back what it
    // took, whatever happens after this point.
    mCtx = ctx;
    mIdentity = identity;
    // Log the first kLogInstances instance lifecycles, including runtime identity and
    // handoff between instances.
    mLogLifecycle = gLifecycleLogged.fetch_add(1) < kLogInstances;
    if (mLogLifecycle)
      Log("GPU: instance created: node %d, timeline %d, runtime instance ID %u, track %s%d, item start %lld, %s "
          "[%lld, %lld), timing %016llx (media node lookup 0x%x, clip speed %g, media ticks/frame %lld), session %s "
          "%016llx%s", (int)mNodeID, (int)inst->inTimelineID, mRuntimeID,
          mItem.known ? "" : "unknown ", (int)mItem.trackID, (long long)mItem.start,
          mBounds.known ? mBoundsKind : "no bounds", (long long)mBounds.start, (long long)mBounds.end,
          (unsigned long long)mItem.timing, (unsigned)mMediaResult, mItem.clipSpeed, (long long)mItem.mediaTicksPerFrame,
          identity ? "shared by identity" : "private", (unsigned long long)identity,
          identity ? "" : !mRuntimeID ? " (no runtime ID)" : !mItem.known ? " (no track item: owner node or start unknown)"
                                                                           : " (no track ID)");
    if (mLogLifecycle)
      Log("GPU: parameter layout %s (count %d, GetParamCount result 0x%x): node %d", LayoutName(mLayout), mParamCount,
          (unsigned)mParamCountResult, (int)mNodeID);
    inst->outIsRealtime = kPrTrue;
    return suiteError_NoError;
  }

  ~ODFilter() {
    if (mLogLifecycle)
      Log("GPU: instance disposed: node %d, timeline %d, runtime instance ID %u", (int)mNodeID, (int)mTimelineID,
          mRuntimeID);
    mSession.reset();
    if (mCtx) ReleaseSharedSession(mCtx, mIdentity);  // parks it for the next instance
  }

  // Reads an integer node property (the host returns them as decimal text,
  // allocated with PrNewPtr). Needs the optional memory suite to free it.
  bool ReadIntProperty(csSDK_int32 node, const char* key, int64_t* out) const {
    if (!mMemory) return false;
    PrMemoryPtr buffer = nullptr;
    if (mSegment->GetNodeProperty(node, key, &buffer) != suiteError_NoError || !buffer) return false;
    char* end = nullptr;
    errno = 0;
    const long long value = strtoll((const char*)buffer, &end, 10);
    const bool ok = end != (char*)buffer && errno == 0;
    mMemory->PrDisposePtr(buffer);
    if (ok) *out = value;
    return ok;
  }

  // A property as text (strings, keyframe data). false if absent.
  bool ReadText(csSDK_int32 node, const char* key, std::string* out) const {
    if (!mMemory) return false;
    PrMemoryPtr buffer = nullptr;
    if (mSegment->GetNodeProperty(node, key, &buffer) != suiteError_NoError || !buffer) return false;
    out->assign((const char*)buffer);
    mMemory->PrDisposePtr(buffer);
    return true;
  }

  // What maps the item's frames to source frames (TrackItem::timing), and
  // the clip speed and media frame duration for NeighbourStep. Each value
  // that can be read is mixed in with its own tag; none = 0. The media node
  // is the clip node's input 0 (the SDK's SDK_Segment_Utils sample walks
  // clip -> media the same way).
  void ReadTiming(csSDK_int32 owner) {
    uint64_t h = 0;
    auto mix = [&](uint64_t tag, uint64_t value) {
      uint64_t x = h ^ (tag * 0x9E3779B97F4A7C15ull) ^ value;
      x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
      x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
      h = (x ^ (x >> 31)) | 1;
    };
    auto hashText = [](const std::string& t) {
      uint64_t f = 1469598103934665603ull;  // FNV-1a
      for (unsigned char c : t) f = (f ^ c) * 1099511628211ull;
      return f;
    };
    std::string text;
    int64_t iv = 0;
    if (ReadText(owner, kVideoSegmentProperty_Clip_ClipSpeed, &text)) {
      mItem.clipSpeed = strtod(text.c_str(), nullptr);
      mix(1, hashText(text));
    }
    if (ReadIntProperty(owner, kVideoSegmentProperty_Clip_ClipBackwards, &iv)) mix(2, (uint64_t)iv);
    if (ReadText(owner, kVideoSegmentProperty_Clip_TimeRemapping, &text)) mix(3, hashText(text));
    if (ReadIntProperty(owner, kVideoSegmentProperty_Clip_FrameHoldAtTime, &iv)) mix(4, (uint64_t)iv);
    mMediaResult = -1;
    if (mSegment->AcquireInputNodeID) {
      csSDK_int32 media = 0;
      PrTime offset = 0;
      mMediaResult = (int)mSegment->AcquireInputNodeID(owner, 0, &offset, &media);
      if (mMediaResult == (int)suiteError_NoError) {
        mix(5, (uint64_t)offset);
        if (ReadIntProperty(media, kVideoSegmentProperty_Media_InPointMediaTimeAsTicks, &iv)) {
          mMediaIn = iv;
          mix(6, (uint64_t)iv);
        }
        if (ReadIntProperty(media, kVideoSegmentProperty_Media_ClipID, &iv)) mix(7, (uint64_t)iv);
        if (ReadIntProperty(media, kVideoSegmentProperty_Media_StreamFrameRate, &iv) && iv > 0)
          mItem.mediaTicksPerFrame = iv;
        if (mItem.clipSpeed == 0 && ReadText(media, kVideoSegmentProperty_Media_ClipSpeed, &text)) {
          mItem.clipSpeed = strtod(text.c_str(), nullptr);
          mix(8, hashText(text));
        }
        if (ReadIntProperty(media, kVideoSegmentProperty_Media_ClipBackwards, &iv)) mix(9, (uint64_t)iv);
        mSegment->ReleaseVideoNodeID(media);
      }
    }
    mItem.timing = h;
  }

  // atCreation: the probe of CreateInstance (a failure declines the instance there);
  // otherwise a render's read, whose failure gives that render the defaults (the next render
  // reads again), except the grain mode, which keeps what the probe read (mGrainMode): the
  // default, Finest grain only, would give a project saved before the control existed the
  // picture for that frame. *unread: the advanced controls could not be read.
  UIParams ReadParams(PrTime clipTime, bool* unread = nullptr, bool atCreation = false) const {
    if (unread) *unread = false;
    UIParams ui;
    if (!atCreation) {
      ui.grainMode = mGrainMode;
      ui.controlSet = mControlSet;
      ui.processing = mProcessing;
    }
    const ParamLayout layout = mLayout;
    // The nine original controls, each on its own: a failed read keeps that control's
    // default. GPU filters do not count the input layer, and most of the nine sit inside
    // topics, so each goes through the layout (GpuIndexOf; under Unknown only those before
    // every topic can be found, but an instance is never created with an Unknown layout).
    auto get = [&](int id, double fallback) {
      PrParam p = {};
      const int index = GpuIndexOf(layout, id);
      if (index < 0 || mSegment->GetParam(mNodeID, index, clipTime, &p) != suiteError_NoError) return fallback;
      return ParamAsDouble(p, fallback);
    };
    ui.luma = get(OD_P_LUMA, ui.luma);
    ui.chroma = get(OD_P_CHROMA, ui.chroma);
    ui.temporalRadius = get(OD_P_TEMPORAL_RADIUS, ui.temporalRadius);
    ui.temporalStrength = get(OD_P_TEMPORAL_STRENGTH, ui.temporalStrength);
    ui.spatial = get(OD_P_SPATIAL, ui.spatial);
    ui.sharpen = get(OD_P_SHARPEN, ui.sharpen);
    ui.noiseLevel = get(OD_P_NOISE_LEVEL, ui.noiseLevel);
    // Popups: GetParam hands the GPU side a 0-based item index (observed in
    // Premiere 26.5.2: View "Noise removed" arrived as 1), while
    // UIParams and the CPU path use the After Effects 1-based convention.
    ui.quality = (int)get(OD_P_QUALITY, ui.quality - 1) + 1;
    const double view = get(OD_P_VIEW, ui.view - 1);
    ui.view = (view >= 0 && view < OD_NUM_VIEWS && view == std::floor(view)) ? int(view) + 1 : OD_DFLT_VIEW;
    // Advanced and experimental controls, all or none: under an Unknown
    // layout (never for an instance that was created), or after any GetParam
    // error or unexpected type in this read, every one keeps its default.
    // Look is never read: the render does not depend on it.
    if (layout == ParamLayout::Unknown) {
      if (unread) *unread = true;
      return ui;
    }
    UIParams adv = ui;
    for (int pf = 1; pf < kNumParams; ++pf) {
      const int id = IdAtPf(pf);
      double* v = AdvancedField(adv, id);
      if (!v) continue;  // one of the nine, a topic start or end, Look: not read here
      PrParam p = {};
      const int index = GpuIndex(layout, pf);
      const prSuiteError r = mSegment->GetParam(mNodeID, index, clipTime, &p);
      const bool checkbox = IsAdvancedCheckbox(id), popup = IsAdvancedPopup(id);
      // A popup: an integer naming one of its items (0-based), which also
      // catches another integer control read at its index.
      const bool typeOk = checkbox ? (p.mType == kPrParamType_Bool || IsIntegerType(p.mType))
                          : popup  ? IsIntegerType(p.mType) && ParamAsDouble(p, -1) >= 0 &&
                                         ParamAsDouble(p, -1) < AdvancedPopupChoices(id)
                                   : (p.mType == kPrParamType_Float32 || p.mType == kPrParamType_Float64);
      if (r != suiteError_NoError || !typeOk) {
        if (unread) *unread = true;
        static std::atomic<int> logged{0};
        if (logged.fetch_add(1) < kLogInstances)
          Log("GPU: node %d: parameter %d (disk ID %d, GPU index %d, layout %s, count %d) came back %s %d at clip "
              "time %lld, expected a %s: %s",
              (int)mNodeID, pf, id, index, LayoutName(layout), mParamCount,
              r != suiteError_NoError ? "with error" : "as type",
              r != suiteError_NoError ? (int)r : (int)p.mType, (long long)clipTime,
              checkbox ? "checkbox (Bool or integer)" : popup ? "popup (an integer item)" : "float slider",
              atCreation ? "declining the instance, Premiere renders it in software"
                         : "this render takes the advanced controls at their defaults, the grain mode as read at "
                           "creation (counted in the stats lines); the next render reads them again");
        return ui;
      }
      // A popup arrives 0-based (as Quality and View); UIParams holds the
      // After Effects 1-based item.
      *v = popup ? ParamAsDouble(p, *v - 1) + 1 : ParamAsDouble(p, *v);
    }
    return adv;
  }

  struct ScalarRead {
    int id = -1, index = -1;
    PrParam raw{};
    prSuiteError result = suiteError_InvalidParms;
    bool exact = false;
    int64_t value = 0;
  };
  bool ReadInteger(int id, PrTime t, int64_t& v, ScalarRead* diagnostic = nullptr) const {
    ScalarRead r; r.id = id; r.index = GpuIndexOf(mLayout, id);
    if (r.index >= 0) r.result = mSegment->GetParam(mNodeID, r.index, t, &r.raw);
    if (r.result == suiteError_NoError) {
      if (IsIntegerType(r.raw.mType)) {
        r.value = r.raw.mType == kPrParamType_Int64 ? r.raw.mInt64 : (int64_t)ParamAsDouble(r.raw, 0);
        r.exact = r.value >= INT32_MIN && r.value <= INT32_MAX;
      } else if (r.raw.mType == kPrParamType_Float64) {
        // All signed 32-bit words are exactly representable in a double.
        // Check before casting; no rounding, NaN, overflow or Float32 guess.
        const double n = r.raw.mFloat64;
        r.exact = std::isfinite(n) && n >= INT32_MIN && n <= INT32_MAX && std::trunc(n) == n;
        if (r.exact) r.value = (int64_t)n;
      }
    }
    if (diagnostic) *diagnostic = r;
    if (r.exact) v = r.value;
    return r.exact;
  }
  void LogScalar(const ScalarRead& r) const {
    const char* type = "unsupported";
    char raw[64] = "unavailable";
    if (r.result == suiteError_NoError) {
      switch (r.raw.mType) {
        case kPrParamType_Int8: type="Int8"; snprintf(raw,sizeof raw,"%d",r.raw.mInt8); break;
        case kPrParamType_Int16: type="Int16"; snprintf(raw,sizeof raw,"%d",r.raw.mInt16); break;
        case kPrParamType_Int32: type="Int32"; snprintf(raw,sizeof raw,"%d",r.raw.mInt32); break;
        case kPrParamType_Int64: type="Int64"; snprintf(raw,sizeof raw,"%lld",(long long)r.raw.mInt64); break;
        case kPrParamType_Float32: type="Float32"; snprintf(raw,sizeof raw,"%.9g",r.raw.mFloat32); break;
        case kPrParamType_Float64: type="Float64"; snprintf(raw,sizeof raw,"%.17g",r.raw.mFloat64); break;
        case kPrParamType_Bool: type="Bool"; snprintf(raw,sizeof raw,"%d",r.raw.mBool); break;
        default: snprintf(raw,sizeof raw,"unsupported"); break; // never print/dereference a host pointer
      }
    }
    Log("GPU: capture scalar node=%d disk=%d pf=%d gpu=%d type=%s(%d) rc=0x%x raw=%s exact=%d word=%08x",
        (int)mNodeID,r.id,PfIndex(r.id),r.index,type,int(r.raw.mType),unsigned(r.result),raw,int(r.exact),uint32_t(r.value));
  }

  uint64_t RequestToken(PrTime t) const {
    int64_t lo = 0, hi = 0;
    ScalarRead l, h;
    const bool haveLo = ReadInteger(OD_P_CAPTURE_REQUEST_LO, t, lo, &l);
    const bool haveHi = ReadInteger(OD_P_CAPTURE_REQUEST_HI, t, hi, &h);
    if (!haveLo || !haveHi) {
      LogScalar(l); LogScalar(h);
      Log("Capture: GPU request identity unavailable (integer controls 69/70); capture cannot collect here");
      return 0;
    }
    return uint64_t(uint32_t(lo)) | uint64_t(uint32_t(hi)) << 32;
  }

  // Never read kPrParamType_PrMemoryPtr: the SDK does not define its framing
  // or ownership for AE arbitrary values. Scalars name an immutable snapshot
  // published by NEW/COPY/UNFLATTEN/UI callbacks in this process.
  bool ReadSavedProfile(PrTime t, std::shared_ptr<const CaptureSnapshot>& capture, bool atCreation = false) const {
    capture.reset();
    // By disk ID, through the layout (kParamOrder): the block sits inside the Noise profile
    // topic.
    int64_t policy = -1, revision = 0;
    std::array<ScalarRead, 10> reads;
    const bool havePolicy = ReadInteger(OD_P_PROFILE_POLICY, t, policy, &reads[0]);
    if (havePolicy && policy == OD_PROFILE_AUTOMATIC-1) return true;
    bool exact = ReadInteger(OD_P_PROFILE_REVISION, t, revision, &reads[1]);
    CaptureDigest digest{};
    for (int i = 0; i < 8; ++i) {
      int64_t word = 0;
      exact = ReadInteger(OD_P_PROFILE_DIGEST0 + i, t, word, &reads[2+i]) && exact;
      digest[i] = uint32_t(word);
    }
    const char* outcome = "invalid-scalar";
    bool ok = false;
    if (havePolicy && policy == OD_PROFILE_CAPTURED-1 && exact && revision >= 0) {
      // A genuinely empty profile is Automatic, including GPU eligibility.
      if (revision == 0) { ok = digest == CaptureDigest{}; outcome = ok ? "empty-automatic" : "invalid-empty-key"; }
      else {
        if (mCapture && mCapture->digest == digest && mCapture->capture.revision == revision) {
          capture = mCapture; outcome = "instance-hit";
        } else { capture = FindCapture(digest, (uint32_t)revision); outcome = capture ? "bridge-hit" : "bridge-miss"; }
        ok = capture != nullptr;
      }
    }
    // Creation records the entire transport once; render failures once per
    // instance. No logging or bridge lookup on a normal captured render hit.
    if (atCreation || (!ok && !mCaptureFailureLogged.exchange(true))) {
      Log("GPU: capture lookup node=%d bridge=%016llx layout=%s count=%d time=%lld policy=%lld revision=%lld sha256=%s result=%s",
          (int)mNodeID,(unsigned long long)CaptureBridgeID(),LayoutName(mLayout),mParamCount,(long long)t,
          (long long)policy,(long long)revision,CaptureDigestText(digest).c_str(),outcome);
      for (const auto& r : reads) LogScalar(r);
      LogCaptureBridge(outcome, !ok);
    }
    return ok;
  }

  // Sequence frames between requested neighbours (NeighbourStep). Field
  // renders keep 1: whether inRenderTicksPerFrame is then a frame or a field
  // has not been seen, and a wrong guess would skip real frames.
  int StepFor(const PrGPUFilterRenderParams* rp) const {
    if (rp->inRenderFieldType != prFieldsNone) return 1;
    return NeighbourStep(mItem, rp->inRenderTicksPerFrame);
  }

  // The same list must come out in GetFrameDependencies and in Render: both
  // derive it from the render parameters and the bounds read at creation.
  std::vector<int> Offsets(const PrGPUFilterRenderParams* rp, const UIParams& ui) const {
    const ODParams mapped = ToCoreParams(ui,true,false);
    const int radius = DependencyRadius(mapped);
    // Sequence time starts at 0: nothing before it is ever requested, bounds
    // or not (the first Premiere run: renders at sequence time 0 outside the
    // item's bounds asked for -1 and -2, got 2 frames for 4 and went spatial
    // only).
    return NeighbourOffsets(radius, rp->inSequenceTime, rp->inRenderTicksPerFrame, BoundsFor(rp), /*earliest*/ 0,
                            StepFor(rp));
  }

  // The track item's bounds, if they describe this render. A frame outside
  // them means the host's time base is not what we expect: then every
  // neighbour is requested, never none.
  TimeBounds BoundsFor(const PrGPUFilterRenderParams* rp) const {
    if (!mBounds.known) return TimeBounds();
    if (rp->inSequenceTime < mBounds.start || rp->inSequenceTime >= mBounds.end) {
      // Once per instance (and at most kLogInstances times per process).
      static std::atomic<int> logged{0};
      if (!mWarnedBounds.exchange(true) && logged.fetch_add(1) < kLogInstances)
        Log("GPU: node %d, timeline %d: sequence time %lld (clip time %lld, inQuality %d) is outside the %s "
            "[%lld, %lld); requesting every neighbour up to the radius", (int)mNodeID, (int)mTimelineID,
            (long long)rp->inSequenceTime, (long long)rp->inClipTime, (int)rp->inQuality, mBoundsKind,
            (long long)mBounds.start, (long long)mBounds.end);
      return TimeBounds();
    }
    return mBounds;
  }

  // What GetFrameDependencies worked out for a frame, for its next queries
  // and its Render: a filter instance has a fixed set of parameters
  // (PrSDKGPUFilter.h; animated ones are fixed functions of clip time), so
  // the values read at query 0 hold for the frame. Saves 9 host calls per
  // query (5 queries at radius 2) and per Render.
  struct FramePlan {
    bool live = false;
    PrTime sequenceTime = 0, clipTime = 0;
    int field = 0;
    double askedMs = 0;  // first query, for the instrumentation
    UIParams ui;
    bool advancedUnread = false;  // ReadParams could not read the advanced controls
    std::vector<int> offsets;
  };
  static int FieldOf(const PrGPUFilterRenderParams* rp) {
    return rp->inRenderFieldType == prFieldsNone ? 0 : 1 + (int)rp->inRenderField;
  }
  // Under mPlanMutex.
  FramePlan* FindPlan(const PrGPUFilterRenderParams* rp) {
    for (FramePlan& p : mPlans)
      if (p.live && p.sequenceTime == rp->inSequenceTime && p.clipTime == rp->inClipTime && p.field == FieldOf(rp))
        return &p;
    return nullptr;
  }

  prSuiteError GetFrameDependencies(const PrGPUFilterRenderParams* rp, csSDK_int32* ioQueryIndex,
                                    PrGPUFilterFrameDependency* out) {
    if (!rp || !ioQueryIndex || !out) return suiteError_InvalidParms;
    const double t0 = NowMs();
    std::vector<int> offsets;
    bool planned = false;
    if (*ioQueryIndex > 0) {
      std::lock_guard<std::mutex> lock(mPlanMutex);
      if (FramePlan* p = FindPlan(rp)) {
        offsets = p->offsets;
        planned = true;
      }
    }
    if (!planned) {
      bool unread = false;
      const UIParams ui = ReadParams(rp->inClipTime, &unread);
      offsets = Offsets(rp, ui);
      std::lock_guard<std::mutex> lock(mPlanMutex);
      FramePlan* p = FindPlan(rp);
      if (!p) {
        p = &mPlans[mNextPlan];
        mNextPlan = (mNextPlan + 1) % kPlans;
        p->askedMs = t0;
      }
      p->live = true;
      p->sequenceTime = rp->inSequenceTime;
      p->clipTime = rp->inClipTime;
      p->field = FieldOf(rp);
      p->ui = ui;
      p->advancedUnread = unread;
      p->offsets = offsets;
    }
    RenderStats::Get().NoteDeps(NowMs() - t0);
    csSDK_int32 index = *ioQueryIndex;
    // Nothing (more) to ask for. Field separation is left at the host default:
    // interlaced material arrives as full-height single-field frames.
    if (index < 0 || index >= (csSDK_int32)offsets.size()) return suiteError_NotImplemented;
    out->outDependencyType = PrGPUDependency_InputFrame;
    out->outTrackID = 0;
    // Offsets count source frames: NeighbourStep sequence frames apart.
    out->outSequenceTime =
        rp->inSequenceTime + (PrTime)offsets[index] * StepFor(rp) * rp->inRenderTicksPerFrame;
    if (index + 1 < (csSDK_int32)offsets.size()) *ioQueryIndex = index + 1;
    return suiteError_NoError;
  }

  bool Describe(PPixHand ppix, FrameInfo* f) const {
    if (!ppix) return false;
    void* data = nullptr;
    if (mDevice->GetGPUPPixData(ppix, &data) != suiteError_NoError || !data) return false;
    prRect b = {};
    csSDK_int32 rb = 0;
    PrPixelFormat fmt = PrPixelFormat_Invalid;
    if (mPPix->GetPixelFormat(ppix, &fmt) != suiteError_NoError) return false;
    if (mPPix->GetBounds(ppix, &b) != suiteError_NoError) return false;
    if (mPPix->GetRowBytes(ppix, &rb) != suiteError_NoError) return false;
    if (fmt != PrPixelFormat_GPU_BGRA_4444_32f && fmt != PrPixelFormat_GPU_BGRA_4444_16f) return false;
    int w = b.right - b.left, h = b.bottom - b.top;
    int bpp = fmt == PrPixelFormat_GPU_BGRA_4444_32f ? 16 : 8;
    if (w <= 0 || h <= 0 || rb < w * bpp) return false;
    // The buffer must hold the frame (the core checks it too; the
    // pass-through copy relies on it).
    if ((uint64_t)((__bridge id<MTLBuffer>)data).length < (uint64_t)rb * (uint64_t)(h - 1) + (uint64_t)w * bpp)
      return false;
    f->buffer = data; f->format = fmt; f->width = (uint32_t)w; f->height = (uint32_t)h; f->rowBytes = rb;
    return true;
  }

  // A frame as it is, in any pixel format (for a pass-through copy only).
  bool DescribeRaw(PPixHand ppix, FrameInfo* f) const {
    if (!ppix) return false;
    void* data = nullptr;
    prRect b = {};
    csSDK_int32 rb = 0;
    PrPixelFormat fmt = PrPixelFormat_Invalid;
    if (mDevice->GetGPUPPixData(ppix, &data) != suiteError_NoError || !data) return false;
    if (mPPix->GetPixelFormat(ppix, &fmt) != suiteError_NoError) return false;
    if (mPPix->GetBounds(ppix, &b) != suiteError_NoError) return false;
    if (mPPix->GetRowBytes(ppix, &rb) != suiteError_NoError) return false;
    const int w = b.right - b.left, h = b.bottom - b.top;
    if (w <= 0 || h <= 0 || rb <= 0) return false;
    f->buffer = data; f->format = fmt; f->width = (uint32_t)w; f->height = (uint32_t)h; f->rowBytes = rb;
    return true;
  }

  static ODFrame ToOD(const FrameInfo& f, int64_t key) {
    ODFrame h = {};
    h.mtlBuffer = f.buffer;
    h.width = f.width; h.height = f.height; h.rowBytes = (uint32_t)f.rowBytes;
    h.format = f.format == PrPixelFormat_GPU_BGRA_4444_32f ? OD_PIXEL_BGRA32F : OD_PIXEL_BGRA16F;
    h.frameKey = key;
    return h;
  }

  // Make the output show the unprocessed input. Only between frames of one
  // pixel format and size (a copy across formats or sizes is not a frame);
  // rows of different strides are copied row by row, as far as the shorter
  // stride (each holds a whole row of pixels). In place it already is the
  // input. False if it cannot be done or not even queued: the caller then
  // returns an error, never success for a frame it did not write.
  bool PassThrough(const FrameInfo& in, const FrameInfo& out) {
    if (!in.buffer || !out.buffer) return false;
    if (in.format != out.format || in.width != out.width || in.height != out.height) return false;
    if (in.buffer == out.buffer) return in.rowBytes == out.rowBytes;  // in place: it already is the input
    @autoreleasepool {
      id<MTLBuffer> src = (__bridge id<MTLBuffer>)in.buffer;
      id<MTLBuffer> dst = (__bridge id<MTLBuffer>)out.buffer;
      const NSUInteger rows = in.height;
      const NSUInteger rowLen = MIN((NSUInteger)in.rowBytes, (NSUInteger)out.rowBytes);
      if ((rows - 1) * (NSUInteger)in.rowBytes + rowLen > src.length ||
          (rows - 1) * (NSUInteger)out.rowBytes + rowLen > dst.length)
        return false;
      id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mMTLQueue;
      id<MTLCommandBuffer> cb = queue ? [queue commandBuffer] : nil;
      id<MTLBlitCommandEncoder> blit = cb ? [cb blitCommandEncoder] : nil;
      if (!blit) {
        static std::atomic<uint64_t> seen{0};
        OD_LOG_COUNTED(seen, "GPU: no command buffer for the pass-through copy");
        return false;
      }
      if (in.rowBytes == out.rowBytes) {
        [blit copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:(NSUInteger)in.rowBytes * rows];
      } else {
        for (NSUInteger y = 0; y < rows; ++y)
          [blit copyFromBuffer:src sourceOffset:y * (NSUInteger)in.rowBytes toBuffer:dst
             destinationOffset:y * (NSUInteger)out.rowBytes size:rowLen];
      }
      [blit endEncoding];
      [cb commit];
    }
    return true;
  }

  // What Render has done so far, for Recover after an exception, and its
  // sample for the stats lines (noted however the render ends).
  struct RenderState {
    PPixHand allocated = nullptr;  // the output we created (disposed if we fail)
    bool haveCur = false, haveOut = false, written = false;
    bool captureRequired = true; // unknown policy also fails closed on exceptions
    FrameInfo cur, out;
    RenderSample sample;
  };

  // A failure that leaves no valid output: the output we created goes.
  prSuiteError Fail(RenderState& st, PPixHand* outFrame) {
    if (st.allocated) {
      if (mPPix->Dispose) mPPix->Dispose(st.allocated);
      if (*outFrame == st.allocated) *outFrame = nullptr;
      st.allocated = nullptr;
    }
    return suiteError_Fail;
  }

  // After an exception in Render: success only if the output holds a valid
  // frame (written, the input in place, or a verified pass-through copy).
  prSuiteError Recover(RenderState& st, const PPixHand* inFrames, PPixHand* outFrame) {
    if (st.written) return suiteError_NoError;
    if (*outFrame && *outFrame == inFrames[0]) return suiteError_NoError;  // in place: it is the input
    if (st.haveCur && st.haveOut && PassThrough(st.cur, st.out)) {
      static std::atomic<uint64_t> seen{0};
      OD_LOG_COUNTED(seen, "GPU: the render was interrupted; the frame was passed through");
      return suiteError_NoError;
    }
    static std::atomic<uint64_t> seen{0};
    OD_LOG_COUNTED(seen, "GPU: the render was interrupted and the frame could not be passed through; returning an "
                   "error");
    return Fail(st, outFrame);
  }

  prSuiteError Render(const PrGPUFilterRenderParams* rp, const PPixHand* inFrames,
                      csSDK_size_t inFrameCount, PPixHand* outFrame) {
    if (!rp || !inFrames || inFrameCount < 1 || !outFrame) return suiteError_InvalidParms;
    RenderState st;
    RenderSample& sample = st.sample;
    sample.enterMs = NowMs();
    sample.id = gRenderSerial.fetch_add(1) + 1;
    sample.active = gActiveRenders.fetch_add(1) + 1;
    sample.inQuality = (int)rp->inQuality;
    sample.sequenceTime = rp->inSequenceTime;
    sample.failed = true;  // until the core call succeeds
    prSuiteError result = suiteError_Fail;
    bool returned = false;
    @try {
      try {
        result = RenderBody(rp, inFrames, inFrameCount, outFrame, st);
        returned = true;
      } catch (const std::exception& e) {
        static std::atomic<uint64_t> seen{0};
        OD_LOG_COUNTED(seen, "GPU: exception in Render: %s", e.what());
      } catch (...) {
        static std::atomic<uint64_t> seen{0};
        OD_LOG_COUNTED(seen, "GPU: unknown exception in Render");
      }
    } @catch (NSException* e) {
      static std::atomic<uint64_t> seen{0};
      OD_LOG_COUNTED(seen, "GPU: NSException in Render: %s", e.reason.UTF8String ?: "?");
    }
    if (!returned) result = st.captureRequired ? Fail(st, outFrame) : Recover(st, inFrames, outFrame);
    gActiveRenders.fetch_sub(1);
    // Every render, however it ended (the stats lines count renders, the
    // failed ones and those that returned an error).
    sample.errorReturned = result != suiteError_NoError;
    sample.totalMs = NowMs() - sample.enterMs;
    try {
      RenderStats::Get().Note(sample);
    } catch (...) {
    }
    return result;
  }

  prSuiteError RenderBody(const PrGPUFilterRenderParams* rp, const PPixHand* inFrames, csSDK_size_t inFrameCount,
                          PPixHand* outFrame, RenderState& rs) {
    RenderSample& sample = rs.sample;
    std::shared_ptr<const CaptureSnapshot> captured;
    if (!ReadSavedProfile(rp->inClipTime, captured)) return suiteError_Fail;
    rs.captureRequired = captured != nullptr;
    FrameInfo cur;
    if (!Describe(inFrames[0], &cur)) {
      if (captured) return suiteError_Fail;
      // Nothing we understand. In place (the host pre-set the output to the
      // input) leaving it alone is a pass-through; into another output only
      // a copy of the same format and size is; else there is nothing valid
      // to return.
      FrameInfo rawIn, rawOut;
      const bool haveIn = DescribeRaw(inFrames[0], &rawIn);
      if (haveIn) {
        sample.width = rawIn.width;
        sample.height = rawIn.height;
      }
      if (*outFrame && (*outFrame == inFrames[0] ||
                        (haveIn && DescribeRaw(*outFrame, &rawOut) && PassThrough(rawIn, rawOut)))) {
        static std::atomic<uint64_t> seen{0};
        OD_LOG_COUNTED(seen, "GPU: an input frame the filter cannot read (pixel format 0x%x, %ux%u); passed through",
                       haveIn ? (unsigned)rawIn.format : 0u, haveIn ? rawIn.width : 0u, haveIn ? rawIn.height : 0u);
        return suiteError_NoError;
      }
      static std::atomic<uint64_t> seen{0};
      OD_LOG_COUNTED(seen, "GPU: an input frame the filter cannot read (pixel format 0x%x, %ux%u) and no output it "
                     "can fill; returning an error", haveIn ? (unsigned)rawIn.format : 0u, haveIn ? rawIn.width : 0u,
                     haveIn ? rawIn.height : 0u);
      return suiteError_Fail;
    }
    if (captured && (captured->capture.width != cur.width || captured->capture.height != cur.height ||
                     rp->inRenderFieldType != prFieldsNone)) {
      Log("GPU: captured profile resolution/field mismatch; returning an error");
      return suiteError_Fail;
    }
    rs.cur = cur;
    rs.haveCur = true;
    sample.width = cur.width;
    sample.height = cur.height;

    // Output: the host normally hands us *outFrame == the current frame and we
    // work in place. If it did not, allocate one like the input.
    FrameInfo out;
    if (!*outFrame) {
      csSDK_uint32 parN = 1, parD = 1;
      mPPix->GetPixelAspectRatio(inFrames[0], &parN, &parD);
      prFieldType field = prFieldsNone;
      if (mPPix2) mPPix2->GetFieldOrder(inFrames[0], &field);
      PPixHand created = nullptr;
      if (mDevice->CreateGPUPPix(mDeviceIndex, cur.format, (int)cur.width, (int)cur.height, (int)parN,
                                 (int)parD, field, &created) != suiteError_NoError || !created) {
        static std::atomic<uint64_t> seen{0};
        OD_LOG_COUNTED(seen, "GPU: no output frame from the host and CreateGPUPPix failed (%ux%u); returning an error",
                       cur.width, cur.height);
        return suiteError_Fail;
      }
      *outFrame = created;
      rs.allocated = created;
    }
    if (!Describe(*outFrame, &out) || out.width != cur.width || out.height != cur.height ||
        out.format != cur.format) {
      // An output we cannot render into: a copy of the input if it has the
      // input's format and size, else an error (a byte copy across formats
      // or sizes is not a frame).
      if (captured) return Fail(rs, outFrame);
      FrameInfo rawOut;
      if (DescribeRaw(*outFrame, &rawOut) && PassThrough(cur, rawOut)) {
        static std::atomic<uint64_t> seen{0};
        OD_LOG_COUNTED(seen, "GPU: an output frame the filter cannot render into; the input copied into it");
        return suiteError_NoError;
      }
      static std::atomic<uint64_t> seen{0};
      OD_LOG_COUNTED(seen, "GPU: an output frame of another format or size than the input; returning an error");
      return Fail(rs, outFrame);
    }
    rs.out = out;
    rs.haveOut = true;

    UIParams ui;
    std::vector<int> wanted;
    bool planned = false;
    {
      std::lock_guard<std::mutex> lock(mPlanMutex);
      if (FramePlan* p = FindPlan(rp)) {
        ui = p->ui;
        wanted = std::move(p->offsets);
        sample.advancedUnread = p->advancedUnread;
        sample.depsToRenderMs = sample.enterMs - p->askedMs;
        p->live = false;
        planned = true;
      }
    }
    if (!planned) {
      const double t0 = NowMs();
      ui = ReadParams(rp->inClipTime, &sample.advancedUnread);
      wanted = Offsets(rp, ui);
      sample.paramsMs = NowMs() - t0;
    }
    if (captured && sample.advancedUnread) return Fail(rs, outFrame);
    LogTimeBase(rp, wanted.size(), cur.width, cur.height);

    // Frame keys identify frames for the core's automatic profile (which
    // frames it measured; cuts and jumps). Nothing tells us when the content
    // upstream of the effect changes under the same time (an effect below
    // this one, source settings, a proxy toggle), and Premiere replacing the
    // instance on such a change is not documented, so the core reuses no
    // frame between renders (ODParams.verifyFrameCache, set by ToCoreParams).
    const uint32_t field = rp->inRenderFieldType == prFieldsNone ? 0u : 1u + (uint32_t)rp->inRenderField;
    // The request stream of this render, for the statistics: the effect on
    // its item (or this instance, for a private session), timeline, field
    // and frame size.
    {
      uint64_t k = mIdentity ? mIdentity : (uint64_t)(uintptr_t)this;
      k = k * 0x9E3779B97F4A7C15ull ^ (uint64_t)(uint32_t)mTimelineID;
      k = k * 0x9E3779B97F4A7C15ull ^ (((uint64_t)cur.width << 32) | cur.height);
      k = k * 0x9E3779B97F4A7C15ull ^ field;
      sample.frameMs = rp->inRenderTicksPerFrame > 0 ? (double)rp->inRenderTicksPerFrame / 254016000000.0 * 1000.0 : 0;
      RenderStats::Get().NoteEnter(k, sample.enterMs, rp->inSequenceTime, rp->inRenderTicksPerFrame, sample.frameMs);
    }
    // The same for every render of this instance, whatever the clip's speed
    // (see the top of this file); a moved clip gets a new instance and start,
    // a slipped or re-timed one a new timing hash (TrackItem::timing). A
    // neighbour's key is that of the sequence time it was requested at.
    const int64_t base = KeyBase(mItem);
    const int step = StepFor(rp);
    auto keyFor = [&](int offset) {
      return MakeFrameKey(rp->inSequenceTime + (int64_t)offset * step * rp->inRenderTicksPerFrame, base, field,
                          cur.width, cur.height);
    };
    ODFrame frames[OD_MAX_FRAMES];
    int offsets[OD_MAX_FRAMES];
    int count = 0;
    frames[count] = ToOD(cur, keyFor(0));
    offsets[count++] = 0;

    const uint64_t captureToken = mRequestToken;
    const bool capturePending = CapturePending(captureToken);
    if (capturePending) Log("Capture sample: GPU delivered=%zu wanted=%zu current=%ux%u format=0x%x checkout=n/a",
                            size_t(inFrameCount), wanted.size()+1, cur.width, cur.height, unsigned(cur.format));
    // Neighbours arrive in the order we asked for them. If the host gave us a
    // different number than we asked for we cannot tell which is which, so we
    // use none of them (spatial only) rather than merge the wrong frames.
    if (inFrameCount - 1 == wanted.size()) {
      for (size_t i = 0; i < wanted.size() && count < OD_MAX_FRAMES; ++i) {
        FrameInfo n;
        if (!Describe(inFrames[i + 1], &n)) {
          if (capturePending) Log("Capture sample: GPU offset=%d describe_failed skip", wanted[i]);
          continue;
        }
        if (capturePending) Log("Capture sample: GPU offset=%d world=%ux%u format=0x%x alias=%d",
                                wanted[i], n.width, n.height, unsigned(n.format), n.buffer == cur.buffer || n.buffer == out.buffer);
        if (n.width != cur.width || n.height != cur.height || n.format != cur.format) continue;
        if (n.buffer == cur.buffer || n.buffer == out.buffer) continue;  // host clamped to the same frame
        frames[count] = ToOD(n, keyFor(wanted[i]));
        offsets[count++] = wanted[i];
      }
    } else {
      sample.spatialOnly = !wanted.empty();
      if (!mWarnedCount.exchange(true)) {
        std::string asked;
        for (int o : wanted) asked += (asked.empty() ? "" : " ") + std::string(o > 0 ? "+" : "") + std::to_string(o);
        Log("GPU: node %d: asked for %zu neighbours (%s), got %zu at sequence time %lld (clip time %lld, %s); "
            "rendering spatial only (logged once per instance)", (int)mNodeID, wanted.size(), asked.c_str(),
            (size_t)inFrameCount - 1, (long long)rp->inSequenceTime, (long long)rp->inClipTime,
            !mBounds.known ? "no item bounds"
            : (rp->inSequenceTime < mBounds.start || rp->inSequenceTime >= mBounds.end) ? "outside the item bounds"
                                                                                         : "inside the item bounds");
      }
    }

    const ODParams captureParams = ToCoreParams(ui, true, false);
    if (capturePending) {
      const int64_t divisor = std::gcd(int64_t(254016000000LL), int64_t(rp->inRenderTicksPerFrame));
      const int64_t scale = 254016000000LL/divisor, tickStep = rp->inRenderTicksPerFrame/divisor;
      if (scale > UINT32_MAX || tickStep <= 0 || tickStep > UINT32_MAX || rp->inClipTime % divisor) {
        FailCapture(captureToken, "Unsupported fractional render time");
      } else {
        int64_t times[OD_MAX_FRAMES];
        for (int i = 0; i < count; ++i) times[i] = rp->inClipTime/divisor + int64_t(offsets[i])*step*tickStep;
        ObserveCapture(captureToken, mIdentity ? mIdentity : (uint64_t)(uintptr_t)this, *mSession, mMTLQueue,
                       frames, times, count, uint32_t(scale), uint32_t(tickStep),
                       rp->inRenderFieldType == prFieldsNone && rp->inDownsampleFactorX == 1 && rp->inDownsampleFactorY == 1, &captureParams);
      }
    }

    // Premiere 27 uses High for playback AND export. Keep ambiguous High requests Full:
    // cadence, idle gaps and omitted requests cannot certify playback (the SDK permits
    // cached and out-of-order renders). The core's measured governor requires an independent
    // playback-intent guarantee; this GPU-filter ABI does not expose one.
    const bool hostWantsFull = rp->inQuality != kPrRenderQuality_Invalid && rp->inQuality >= kPrRenderQuality_High;
    // Not blocking: the work is queued on Premiere's own command queue, ahead
    // of whatever Premiere does next with the frame (same as the SDK samples).
    ODParams params = ToCoreParams(ui, hostWantsFull, /*wait*/ false);
    if (captured) { params.manualProfile = &captured->capture.profile; params.profileHold = 0; }
    ODFrame outH = ToOD(out, OD_NO_KEY);

    char err[256], asyncErr[256] = "";
    ODStats st = {};
    const double t0 = NowMs();
    double lockWait = 0;
    int rc = mSession->Render(frames, offsets, count, &params, &outH, &st, err, sizeof err, &lockWait, asyncErr);
    sample.reachedCore = true;
    sample.coreMs = NowMs() - t0;
    sample.coreId = st.renderId;
    sample.waitMs = st.waitMs;
    sample.lockWaitMs = lockWait;
    sample.commandBufferMs = st.commandBufferMs;
    sample.inFlight = st.inFlight;
    sample.asyncFailures = st.asyncFailures;
    sample.cached = st.cacheHits;  // 0 unless KeysNameContent() (ToCoreParams)
    sample.wanted = (int)wanted.size();
    sample.got = count - 1;
    sample.coreQuality = params.quality;
    sample.realtime = (int)params.realtime;
    sample.failed = rc != 0;
    RenderStats::Get().NoteCompletions(st.completed, st.completedCount, st.completedDropped);
    if (st.asyncFailures > 0) {
      // Earlier renders of this session returned success, then failed on the
      // GPU: their frames are not valid. Nothing can take them back; the
      // count goes in the stats lines and the first message here.
      static std::atomic<int> logged{0};
      if (logged.fetch_add(1) < kLogInstances)
        Log("GPU: node %d: %d earlier render%s failed on the GPU after returning (first error of the session: %s)",
            (int)mNodeID, st.asyncFailures, st.asyncFailures == 1 ? "" : "s", asyncErr[0] ? asyncErr : "unknown");
    }
    if (rc != 0) {
      static std::atomic<uint64_t> seen{0};
      OD_LOG_COUNTED(seen, "GPU: od_render failed (%d): %s; passing the frame through", rc, err);
      if (captured || !PassThrough(cur, out)) return Fail(rs, outFrame);
    }
    if (captured && !mCapturedRenderLogged.exchange(true))
      Log("GPU: captured render submitted node=%d bridge=%016llx revision=%u raster=%ux%u core_rc=0 (completion asynchronous)",
          (int)mNodeID,(unsigned long long)CaptureBridgeID(),captured->capture.revision,cur.width,cur.height);
    rs.written = true;
    return suiteError_NoError;
  }

  // What the host passes: the first two renders of each instance (two, so
  // that the clip time's step per sequence frame shows) and its first
  // high-quality render (export), up to kLogTimeBase lines per process. The
  // first Premiere run spent a budget of 8 lines per process on thumbnails
  // within 20 ms of opening the project and left nothing for playback or
  // export.
  void LogTimeBase(const PrGPUFilterRenderParams* rp, size_t wanted, uint32_t w, uint32_t h) {
    static std::atomic<int> logged{0};
    const bool high = rp->inQuality != kPrRenderQuality_Invalid && rp->inQuality >= kPrRenderQuality_High;
    const int n = mTimeBaseLines.fetch_add(1);
    const bool firstHigh = high && !mLoggedHigh.exchange(true);
    if (n >= 2 && !firstHigh) return;
    if (logged.fetch_add(1) >= kLogTimeBase) return;
    Log("GPU: time base: node %d, timeline %d, instance %u, %ux%u, inQuality %d%s, clip time %lld, sequence time "
        "%lld, ticks/frame %lld, effect duration %lld, owner node lookup 0x%x, track %d, item start %lld, %s "
        "[%lld, %lld), media node lookup 0x%x, media in-point %lld, clip speed %g, media ticks/frame %lld, timing "
        "%016llx, neighbour step %d, %zu neighbours requested",
        (int)mNodeID, (int)mTimelineID, mRuntimeID, w, h, (int)rp->inQuality, firstHigh ? " (first high-quality)" : "",
        (long long)rp->inClipTime, (long long)rp->inSequenceTime, (long long)rp->inRenderTicksPerFrame,
        (long long)mDuration, (unsigned)mOwnerResult, (int)mItem.trackID, (long long)mItem.start,
        mBounds.known ? mBoundsKind : "no bounds", (long long)mBounds.start, (long long)mBounds.end,
        (unsigned)mMediaResult, (long long)mMediaIn, mItem.clipSpeed, (long long)mItem.mediaTicksPerFrame,
        (unsigned long long)mItem.timing, StepFor(rp), wanted);
  }

 private:
  SuiteRef<PrSDKGPUDeviceSuite> mDevice;
  SuiteRef<PrSDKPPixSuite> mPPix;
  SuiteRef<PrSDKPPix2Suite> mPPix2;
  SuiteRef<PrSDKVideoSegmentSuite> mSegment;
  SuiteRef<PrSDKMemoryManagerSuite> mMemory;
  csSDK_uint32 mDeviceIndex = 0;
  csSDK_int32 mNodeID = 0;
  void* mMTLDevice = nullptr;
  void* mMTLQueue = nullptr;
  ODContext* mCtx = nullptr;
  uint64_t mIdentity = 0;  // registry key with mCtx; 0 = private session
  uint32_t mRuntimeID = 0;
  csSDK_int32 mTimelineID = 0;
  TrackItem mItem;                 // track and start of the item, for the identity and the keys
  bool mLogLifecycle = false;
  std::shared_ptr<Session> mSession;
  std::shared_ptr<const CaptureSnapshot> mCapture;
  mutable std::atomic<bool> mCaptureFailureLogged{false};
  std::atomic<bool> mCapturedRenderLogged{false};
  uint64_t mRequestToken = 0; // non-animatable; parameter edits create a new GPU instance
  TimeBounds mBounds;              // sequence time; unknown: no trimming
  const char* mBoundsKind = nullptr;
  int mOwnerResult = -1;           // AcquireOperatorOwnerNodeID's result, for the log
  int mParamCount = -1;            // GetParamCount's answer; -1 = none
  int mParamCountResult = -1;      // its result code, for the log
  // Decided at creation from the count (an instance is only created with a
  // known one); fixed from then on, read by renders on any thread.
  ParamLayout mLayout = ParamLayout::Unknown;
  // Minimum grain applies to (UIParams item) as the creation probe read it;
  // fixed from then on like mLayout. A render whose advanced read fails
  // keeps it (ReadParams).
  double mGrainMode = OD_DFLT_GRAIN_MODE;
  double mControlSet = OD_CONTROLS_DB, mProcessing = 2;
  int64_t mDuration = 0;           // EffectNode::EffectDuration (identity without an item, and the log)
  int mMediaResult = -1;           // AcquireInputNodeID's result for the media node, for the log
  int64_t mMediaIn = 0;            // MediaNode::InPointMediaTimeAsTicks, for the log
  std::atomic<int> mTimeBaseLines{0};
  std::atomic<bool> mLoggedHigh{false};
  mutable std::atomic<bool> mWarnedBounds{false};
  std::atomic<bool> mWarnedCount{false};
  static constexpr int kPlans = 8;  // frames whose dependencies were asked for and not yet rendered
  std::mutex mPlanMutex;
  FramePlan mPlans[kPlans];
  int mNextPlan = 0;
};

// Nothing may escape into Premiere: no C++ exception, no Objective-C exception.
template <typename F>
prSuiteError Guard(prSuiteError onThrow, F&& f) {
  @try {
    try {
      return f();
    } catch (const std::exception& e) {
      Log("GPU: exception: %s", e.what());
    } catch (...) {
      Log("GPU: unknown exception");
    }
  } @catch (NSException* e) {
    Log("GPU: NSException: %s", e.reason.UTF8String ?: "?");
  }
  return onThrow;
}

prSuiteError CreateInstance(PrGPUFilterInstance* inst) {
  return Guard(suiteError_Fail, [&]() -> prSuiteError {
    // Owned until the host has it: an exception in Initialize frees it, and
    // ~ODFilter gives back the shared session if it got one.
    std::unique_ptr<ODFilter> f(new ODFilter());
    prSuiteError r = f->Initialize(inst);
    if (r == suiteError_NoError) inst->ioPrivatePluginData = f.release();
    return r;
  });
}

prSuiteError DisposeInstance(PrGPUFilterInstance* inst) {
  return Guard(suiteError_NoError, [&]() -> prSuiteError {
    if (inst) {
      delete (ODFilter*)inst->ioPrivatePluginData;
      inst->ioPrivatePluginData = nullptr;
    }
    return suiteError_NoError;
  });
}

prSuiteError GetFrameDependencies(PrGPUFilterInstance* inst, const PrGPUFilterRenderParams* rp,
                                  csSDK_int32* ioQueryIndex, PrGPUFilterFrameDependency* out) {
  return Guard(suiteError_NotImplemented, [&]() -> prSuiteError {
    if (!inst || !inst->ioPrivatePluginData) return suiteError_NotImplemented;
    return ((ODFilter*)inst->ioPrivatePluginData)->GetFrameDependencies(rp, ioQueryIndex, out);
  });
}

prSuiteError Precompute(PrGPUFilterInstance*, const PrGPUFilterRenderParams*, csSDK_int32, PPixHand) {
  return suiteError_NotImplemented;
}

prSuiteError Render(PrGPUFilterInstance* inst, const PrGPUFilterRenderParams* rp, const PPixHand* inFrames,
                    csSDK_size_t inFrameCount, PPixHand* outFrame) {
  // ODFilter::Render recovers from exceptions itself (a pass-through, or an
  // error); this guard is the last resort: success only in place, where the
  // output is the input.
  const bool inPlace = inFrames && inFrameCount > 0 && outFrame && *outFrame && *outFrame == inFrames[0];
  return Guard(inPlace ? suiteError_NoError : suiteError_Fail, [&]() -> prSuiteError {
    if (!inst || !inst->ioPrivatePluginData) return suiteError_Fail;
    return ((ODFilter*)inst->ioPrivatePluginData)->Render(rp, inFrames, inFrameCount, outFrame);
  });
}

}  // namespace

extern "C" __attribute__((visibility("default"))) prSuiteError xGPUFilterEntry(
    csSDK_uint32 inHostInterfaceVersion, csSDK_int32* ioIndex, prBool inStartup, piSuitesPtr piSuites,
    PrGPUFilter* outFilter, PrGPUFilterInfo* outFilterInfo) {
  (void)inHostInterfaceVersion;
  (void)piSuites;
  if (inStartup) {
    if (!outFilter || !outFilterInfo) return suiteError_InvalidParms;
    if (ioIndex && *ioIndex > 0) return suiteError_InvalidParms;  // one filter in this module
    outFilter->CreateInstance = CreateInstance;
    outFilter->DisposeInstance = DisposeInstance;
    outFilter->GetFrameDependencies = GetFrameDependencies;
    outFilter->Precompute = Precompute;
    outFilter->Render = Render;
    outFilterInfo->outInterfaceVersion = PrSDKGPUFilterInterfaceVersion;
    PinImage();  // the host may unload us while the core still has work queued (OpenDenoisePlugin.h)
    // Null match name: Premiere pairs this filter with the effect in this
    // module's PiPL ("Open Denoise"), as documented in PrSDKGPUFilter.h.
    outFilterInfo->outMatchName = PrSDKString();
    return suiteError_NoError;
  }
  // Shutdown ("unloading any resources it loaded on startup", SDK guide
  // 12.2.6): parked sessions, a bounded wait for the core's asynchronous
  // work, the contexts (GlobalTeardown).
  return Guard(suiteError_NoError, [&]() -> prSuiteError {
    GlobalTeardown("GPU", /*gpuSessions*/ true);
    return suiteError_NoError;
  });
}
