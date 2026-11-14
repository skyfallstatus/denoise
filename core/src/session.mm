// SPDX-License-Identifier: MIT
// Session: per-frame cache, the fixed pipeline
// ingest -> profile -> temporal -> spatial -> sharpen -> output, and the C API.
// Scratch textures and the memory budget are the context's (memory.mm).
#include "session.h"
#include "kernel_trace.h"
#include "opendenoise/profiling.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <mach/mach_time.h>
#include <unistd.h>

#include "async.h"
#include "debug.h"

namespace od {

namespace {

// Every piece of asynchronous work of the process (od_wait_idle), and the
// render ids (ODStats.renderId).
std::atomic<int> gAsyncPending{0};
std::atomic<uint64_t> gRenderIds{0};

// Completed renders kept for the next call to report (ODStats.completed).
constexpr size_t kMaxPendingCompletions = 64;

double nowMs() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

uint64_t textureBytes(id<MTLTexture> t) { return t ? (uint64_t)t.allocatedSize : 0; }

bool hostTraceEnabled() { static const bool yes = getenv("OD_HOST_TRACE") != nullptr; return yes; }

double gpuMsOf(id<MTLCommandBuffer> cb) {
  double d = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
  return d > 0 ? d : 0;
}

// Host time in seconds, the clock of MTLCommandBuffer.GPUStartTime/EndTime.
double hostSeconds() {
  static const double scale = [] {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    return (double)tb.numer / (double)tb.denom * 1e-9;
  }();
  return (double)mach_absolute_time() * scale;
}

// From commit to the GPU starting the buffer, and to its completion, in ms.
void queueTimes(id<MTLCommandBuffer> cb, double committedAt, double* queueMs, double* latencyMs) {
  *queueMs = std::max(0.0, (cb.GPUStartTime - committedAt) * 1000.0);
  *latencyMs = std::max(0.0, (cb.GPUEndTime - committedAt) * 1000.0);
}

// The samples a verifying render keeps of a frame (ingest_check.metal): one
// pixel per 8x8 block, after a 64-byte header.
constexpr uint32_t kCheckBlock = 8;
constexpr NSUInteger kCheckHeaderBytes = 64;

struct CheckUniforms {
  uint32_t rowStride, width, height, stamp, blocksX, blocksY, pad0, pad1;
};

std::string cbError(id<MTLCommandBuffer> cb) {
  NSError* e = cb.error;
  return std::string("GPU command buffer failed: ") + (e ? e.localizedDescription.UTF8String : "unknown error");
}

float clamp01(float v) { return std::isfinite(v) ? std::min(1.0f, std::max(0.0f, v)) : 0.0f; }

inline uint32_t stageBit(ODStage s) { return 1u << (uint32_t)s; }

// A cache entry stays while one of the last kKeepRenders od_render calls used
// it. Sequential playback reuses the previous window; interlaced material
// (two fields with their own keys) and any two interleaved streams through one
// instance reuse the window before that; frames that arrive one or two places
// out of order the one before that. At radius 2 that is 7 entries in
// sequential playback (about 0.5 GB at 3840x2160), at most three windows in
// random access. The rest of the old 24-frame cache was never read again.
constexpr uint64_t kKeepRenders = 3;

// Frame sizes with an automatic profile of their own (playback resolution,
// paused full resolution, and one more).
constexpr size_t kMaxProfileSizes = 3;

// Cold start of a frame size's automatic profile: renders that may block for
// an estimate until the profile reaches kUsableConfidence (stage.h). Some
// frames give only an unusable estimate (on a real night clip about half of
// them: no sharp noise floor, most luminance levels dropped as texture), and
// a blocking estimate of confidence 0 installed as the profile left an
// export almost unfiltered until the asynchronous estimates caught up.
constexpr int kColdWaits = 4;
// Version 1 realtime renders (ODParams.realtime) wait at most this long for a size's
// first estimate, once. On an idle queue the measurement is back well inside
// it at 3840x2160; behind the host's own queued work it is not, and the frame
// is rendered with what there is (another size's profile, or none) instead
// of stalling the host's render thread for every frame queued ahead of it.
constexpr double kRealtimeWaitMs = 6.0;

// Sanitising limits for caller-supplied noise figures. sigma 1e6 times
// noiseLevel 1000, squared in the shaders, stays far below FLT_MAX.
constexpr float kMaxSigma = 1e6f;
constexpr float kMaxNoiseLevel = 1000.0f;

// The caller's manual profile as the stages may use it: false if any value is
// not finite (a corrupt preset must not black out the frame); otherwise every
// sigma clamped to [0, kMaxSigma] and the confidence to [0, 1].
bool sanitizeProfile(const ODNoiseProfile& in, ODNoiseProfile* out) {
  *out = in;
  auto fix = [](float* v) {
    if (!std::isfinite(*v)) return false;
    *v = std::min(kMaxSigma, std::max(0.0f, *v));
    return true;
  };
  bool ok = true;
  for (int c = 0; c < 3; ++c) {
    for (int b = 0; b < OD_BANDS; ++b)
      for (int l = 0; l < OD_LEVELS; ++l) ok = fix(&out->sigma[c][b][l]) && ok;
    for (int l = 0; l < OD_LEVELS; ++l) ok = fix(&out->sigmaFull[c][l]) && ok;
  }
  out->confidence = clamp01(out->confidence);
  out->valid = 1;
  return ok;
}

// Estimated working set of a render of a shape the session has not rendered
// yet, in working frames: kHeldBase + kHeldPerFrame per frame of the window
// (Session::estimateHeld).
constexpr double kHeldBase = 5.0;
constexpr double kHeldPerFrame = 3.0;

// Hold profile (ODParams.profileHold): a frame size's automatic profile is
// held once it has had this many estimates and reached kUsableConfidence.
constexpr uint64_t kHoldMinUpdates = 8;

}  // namespace

void processAsyncBegin() { gAsyncPending.fetch_add(1); }
void processAsyncEnd() { gAsyncPending.fetch_sub(1); }
int processAsyncPending() { return gAsyncPending.load(); }

// The advanced fields of API v6 (and the experimental ones of v9) at their
// neutral values: the picture of API v5, bit for bit (every stage multiplies
// by them only when they differ; the v9 switches add their passes only when
// on).
static void setAdvancedNeutral(ODParams* p) {
  p->advanced = 1;
  for (int g = 0; g < 2; ++g) {
    for (int b = 0; b < OD_BANDS; ++b) {
      p->bandNoiseLevel[g][b] = 1.0f;
      p->bandAmount[g][b] = 1.0f;
    }
  }
  for (int c = 0; c < 3; ++c) p->channelNoiseLevel[c] = 1.0f;
  p->grainFloor = 1.0f;
  p->speckRemoval = 1;
  p->temporalMotionThreshold = 1.0f;
  p->temporalEdgeGate = kEdgeGateNeutral;
  p->sharpenBand[0] = p->sharpenBand[1] = 1.0f;
  p->sharpenCoring = 1.0f;
  p->sharpenHalo = 1.0f;
  p->profileHold = 0;
  // API v9: the experimental controls, off.
  p->grainStability = 0.0f;
  p->coarseChroma = 0;
  // API v13: preserve the legacy colour estimator unless explicitly enabled.
  p->chromaRecovery = 0;
  // API v10: the grain floor over every size, the profile of API v9.
  p->grainMode = OD_GRAIN_WHOLE;
  p->temporalNoiseModel = 0;
}

namespace {

// The advanced fields as the stages read them: neutral when the caller left
// `advanced` at 0 (a zero-initialised ODParams renders as API v5 did);
// otherwise each value finite (neutral if not) and clamped to its range,
// switches 0 or 1.
void sanitizeAdvanced(ODParams* p) {
  if (p->advanced == 0) {
    setAdvancedNeutral(p);
    return;
  }
  auto fix = [](float* v, float neutral, float lo, float hi) {
    *v = std::isfinite(*v) ? std::min(hi, std::max(lo, *v)) : neutral;
  };
  for (int g = 0; g < 2; ++g) {
    for (int b = 0; b < OD_BANDS; ++b) {
      fix(&p->bandNoiseLevel[g][b], 1.0f, 0.25f, 4.0f);
      fix(&p->bandAmount[g][b], 1.0f, 0.0f, 1.0f);
    }
  }
  for (int c = 0; c < 3; ++c) fix(&p->channelNoiseLevel[c], 1.0f, 0.25f, 4.0f);
  fix(&p->grainFloor, 1.0f, 0.0f, 10.0f);
  fix(&p->temporalMotionThreshold, 1.0f, 0.5f, 2.0f);
  fix(&p->temporalEdgeGate, kEdgeGateNeutral, 0.0f, 1.0f);
  fix(&p->sharpenBand[0], 1.0f, 0.0f, 2.5f);
  fix(&p->sharpenBand[1], 1.0f, 0.0f, 2.5f);
  fix(&p->sharpenCoring, 1.0f, 0.5f, 3.0f);
  fix(&p->sharpenHalo, 1.0f, 0.0f, 5.0f);
  fix(&p->grainStability, 0.0f, 0.0f, 1.0f);
  p->speckRemoval = p->speckRemoval != 0 ? 1u : 0u;
  p->profileHold = p->profileHold != 0 ? 1u : 0u;
  p->coarseChroma = p->coarseChroma != 0 ? 1u : 0u;
  p->chromaRecovery = p->chromaRecovery == 1 ? 1u : 0u;
  p->grainMode = p->grainMode == OD_GRAIN_FINEST ? (uint32_t)OD_GRAIN_FINEST : (uint32_t)OD_GRAIN_WHOLE;
  p->temporalNoiseModel = p->temporalNoiseModel == 1 ? 1u : 0u;
  p->advanced = 1;
}

// A band amount below 1: the bands step has work.
bool anyBandAmount(const ODParams& p, bool fineGrain) {
  if (fineGrain && p.lumaStrength > 0.0f && p.grainFloor > 0.0f) return true;
  for (int g = 0; g < 2; ++g)
    for (int b = 0; b < OD_BANDS; ++b)
      if (p.bandAmount[g][b] < 1.0f) return true;
  return false;
}

// Hold profile: this slot (of one grain mode) is established and stays as it
// is. Called with Shared::mutex held.
bool profileHeld(const ProfileSlot::Mode& slot) {
  return slot.updates >= kHoldMinUpdates && slot.profile.valid && slot.profile.confidence >= kUsableConfidence;
}

}  // namespace

// ---------------------------------------------------------------------------
// RenderEnv
// ---------------------------------------------------------------------------
// No encoder (the command buffer refused one: device lost, out of memory)
// fails the render: every pass encoded on it would be a message to nil, and
// the render would return 0 without writing the output. With OD_KERNEL_TRACE
// (kernel_trace.h, diagnostics only) a labelled pass gets an encoder of its
// own with timestamps; without it the label is ignored.
id<MTLComputeCommandEncoder> RenderEnv::compute(const char* traceLabel) {
  if (traceLabel && cb_ && kernelTraceEnabled()) {
    endEncoding();
    ++computeCount_;
    compute_ = kernelTraceEncoder(device(), cb_, traceLabel, session_->renderIndex_, traceState_);
    if (!compute_) fail("the command buffer gave no compute encoder (GPU unavailable?)");
    return compute_;
  }
  if (blit_) {
    [blit_ endEncoding];
    blit_ = nil;
  }
  if (!compute_ && cb_) {
    ++computeCount_;
    compute_ = [cb_ computeCommandEncoder];
    if (!compute_) fail("the command buffer gave no compute encoder (GPU unavailable?)");
  }
  return compute_;
}

id<MTLBlitCommandEncoder> RenderEnv::blit() {
  if (compute_) {
    [compute_ endEncoding];
    compute_ = nil;
  }
  if (!blit_ && cb_) {
    ++blitCount_;
    blit_ = [cb_ blitCommandEncoder];
    if (!blit_) fail("the command buffer gave no blit encoder (GPU unavailable?)");
  }
  return blit_;
}

void RenderEnv::endEncoding() {
  if (compute_) {
    [compute_ endEncoding];
    compute_ = nil;
  }
  if (blit_) {
    [blit_ endEncoding];
    blit_ = nil;
  }
}

void RenderEnv::dispatch(id<MTLComputePipelineState> pso, uint32_t w, uint32_t h) {
  id<MTLComputeCommandEncoder> enc = compute();
  [enc setComputePipelineState:pso];
  [enc dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:Context::threadgroupFor(pso)];
}

id<MTLTexture> RenderEnv::temp(MTLPixelFormat format, uint32_t w, uint32_t h, uint32_t mipLevels) {
  id<MTLTexture> t = session_->acquireTexture(format, w, h, mipLevels);
  if (!t) {
    fail("could not allocate a %ux%u texture (format %lu)", w, h, (unsigned long)format);
    return nil;
  }
  session_->temps_.push_back(t);
  return t;
}

void RenderEnv::recycle(id<MTLTexture> tex) {
  auto& v = session_->temps_;
  auto it = std::find(v.begin(), v.end(), tex);
  if (it == v.end()) return;  // not a temp of this render: ignore
  v.erase(it);
  session_->recycled_.push_back(tex);  // this render may reuse it; others only after commit
}

id<MTLTexture> RenderEnv::frameTexture(const WorkFrame& frame, uint32_t slot, MTLPixelFormat format, uint32_t w,
                                       uint32_t h, uint32_t mipLevels, bool* needsFill) {
  if (mipLevels < 1) mipLevels = 1;
  if (needsFill) *needsFill = true;
  if (frame.cacheIndex < 0 || frame.cacheIndex >= (int)session_->entries_.size() ||
      !session_->entries_[(size_t)frame.cacheIndex].live) {
    return temp(format, w, h, mipLevels);
  }
  Session& s = *session_;
  CacheEntry& e = s.entries_[(size_t)frame.cacheIndex];
  for (CacheEntry::Slot& sl : e.slots) {
    if (sl.slotId != slot) continue;
    if (sl.tex.pixelFormat == format && sl.tex.width == w && sl.tex.height == h && sl.tex.mipmapLevelCount == mipLevels) {
      if (needsFill) *needsFill = false;
      ++s.cacheHits_;
      return sl.tex;
    }
    // Same slot, other shape: replace. The old texture may still be read by
    // passes already encoded, so it only returns to the pool at render end.
    s.cacheBytes_ -= std::min(s.cacheBytes_, textureBytes(sl.tex));
    s.temps_.push_back(sl.tex);
    sl.tex = s.acquireTexture(format, w, h, mipLevels);
    if (!sl.tex) {
      fail("could not allocate a %ux%u cache texture", w, h);
      return nil;
    }
    s.cacheBytes_ += textureBytes(sl.tex);
    return sl.tex;
  }
  id<MTLTexture> t = s.acquireTexture(format, w, h, mipLevels);
  if (!t) {
    fail("could not allocate a %ux%u cache texture", w, h);
    return nil;
  }
  // acquireTexture may have evicted other entries, never this one (it is in
  // use), and entries_ does not move: `e` is still valid.
  e.slots.push_back({slot, t});
  s.cacheBytes_ += textureBytes(t);
  return t;
}

void RenderEnv::fail(const char* fmt, ...) {
  if (failed_) return;
  failed_ = true;
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  session_->lastError_ = buf;
}

// ---------------------------------------------------------------------------
// ProfileStage default smoothing
// ---------------------------------------------------------------------------
void ProfileStage::smooth(const ODNoiseProfile& fresh, uint32_t knots, ODNoiseProfile* const slots[kGrainModes],
                          bool decisive) {
  (void)knots;
  (void)decisive;
  if (!fresh.valid) return;
  for (int m = 0; m < kGrainModes; ++m) {
    ODNoiseProfile* slot = slots[m];
    if (!slot) continue;
    if (!slot->valid) {
      *slot = fresh;
      continue;
    }
    float denom = std::max(slot->confidence, fresh.confidence);
    float w = denom > 0 ? 0.25f * fresh.confidence / denom : 0.25f;
    w = std::min(1.0f, std::max(0.0f, w));
    for (int c = 0; c < 3; ++c) {
      for (int b = 0; b < OD_BANDS; ++b)
        for (int l = 0; l < OD_LEVELS; ++l)
          slot->sigma[c][b][l] += w * (fresh.sigma[c][b][l] - slot->sigma[c][b][l]);
      for (int l = 0; l < OD_LEVELS; ++l) slot->sigmaFull[c][l] += w * (fresh.sigmaFull[c][l] - slot->sigmaFull[c][l]);
    }
    slot->confidence += w * (fresh.confidence - slot->confidence);
  }
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------
Session::Session(std::shared_ptr<Context> ctx)
    : ctx_(std::move(ctx)), memory_(ctx_->memory()), shared_(std::make_shared<Shared>()) {
  grainOverride_ = grainModeOverride();
  shared_->profileQueue = dispatch_queue_create(
      "dev.opendenoise.profile", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0));
  // Per-session cache byte limit, on top of the entry age rule (kKeepRenders)
  // and the context's budget over everything (memory.h): min(2 GiB, 1/16 of
  // the device's recommended working set). Entries used by the render in
  // progress are never evicted, so it is a limit on what is kept BETWEEN
  // renders. OD_CACHE_MB overrides it.
  uint64_t rec = (uint64_t)ctx_->device().recommendedMaxWorkingSetSize;
  cacheBudgetBytes_ = 2ull << 30;
  if (rec > 0 && rec / 16 < cacheBudgetBytes_) cacheBudgetBytes_ = rec / 16;
  if (const char* mb = getenv("OD_CACHE_MB")) {
    long v = atol(mb);
    if (v >= 0) cacheBudgetBytes_ = (uint64_t)v << 20;
  }

  std::string err;
  tracks_.reserve(2 * kMaxProfileSizes);
  ingest_ = IngestStage::make(*ctx_, &err);
  if (ingest_) output_ = OutputStage::make(*ctx_, &err);
  if (output_) analysis_ = AnalysisStage::make(*ctx_, &err);
  if (output_ && !analysis_) { initError_ = err; return; }
  std::shared_ptr<ProfileStage> profile;
  if (output_) profile = std::shared_ptr<ProfileStage>(makeProfileStage(*ctx_, &err));
  if (profile) {
    // The first frame size to come along claims it.
    tracks_.emplace_back();
    tracks_.back().stage = profile;
    tracks_.back().slot = std::make_shared<ProfileSlot>();
    temporal_ = makeTemporalStage(*ctx_, &err);
  }
  if (temporal_) spatial_ = makeSpatialStage(*ctx_, &err);
  if (spatial_) sharpen_ = makeSharpenStage(*ctx_, &err);
  if (sharpen_) bands_ = makeBandsStage(*ctx_, &err);
  if (!bands_) initError_ = err.empty() ? std::string("stage creation failed") : err;
  if (bands_) {
    // The sampling kernels of verifying renders (compiled once per context):
    // here, so that a cold Metal compiler cache (about 100 ms) never lands in
    // a render. If they do not compile, the profile hears of no upstream
    // change; the picture does not depend on them.
    std::string cerr;
    check32_ = ctx_->pipeline("ingest_check_bgra32f", &cerr);
    check16_ = ctx_->pipeline("ingest_check_bgra16f", &cerr);
  }
  lastUseMs_.store(nowMs());
  memory_.addSession(this);
}

Session::~Session() {
  // First, so the memory policy can no longer reach this session.
  memory_.removeSession(this);
  ResourceList out;
  dropCaches(&out);
  out.insert(out.end(), temps_.begin(), temps_.end());
  out.insert(out.end(), recycled_.begin(), recycled_.end());
  temps_.clear();
  recycled_.clear();
  // Freed on the memory queue: od_session_destroy runs on the host's thread.
  // In-flight command buffers keep their textures alive themselves.
  memory_.release(out);
}

void Session::trimIfIdle(double now, double idleMs, ResourceList* out) {
  if (!holdsMemory_.load() || now - lastUseMs_.load() < idleMs) return;
  std::unique_lock<std::mutex> lock(useMutex_, std::try_to_lock);
  if (!lock.owns_lock()) return;  // in a call right now: not idle
  if (now - lastUseMs_.load() < idleMs) return;
  // Only pointer moves: the textures are freed by the caller, off any render
  // thread and outside the session list's lock. Command buffers of this
  // session still in flight keep their textures alive themselves.
  dropCaches(out);
  for (ProfileTrack& t : tracks_) t.measuredKeys.clear();  // frames new to the cache: measure them again
  holdsMemory_.store(false);
}

uint64_t Session::shedCache(uint64_t want, ResourceList* out, bool ownCall) {
  std::unique_lock<std::mutex> lock(useMutex_, std::defer_lock);
  if (!ownCall && !lock.try_lock()) return 0;  // in a call on another thread
  uint64_t got = 0;
  while (got < want) {
    int lru = -1;
    for (size_t i = 0; i < entries_.size(); ++i) {
      const CacheEntry& e = entries_[i];
      if (!e.live || (ownCall && e.lastUse == renderIndex_)) continue;  // used by the call in progress
      if (lru < 0 || e.lastUse < entries_[(size_t)lru].lastUse) lru = (int)i;
    }
    if (lru < 0) break;
    got += entryBytes(entries_[(size_t)lru]);
    evictEntry(lru, out);
  }
  return got;
}

int Session::setError(const std::string& msg) {
  lastError_ = msg;
  return 1;
}

void Session::asyncBegin(Shared& shared) {
  shared.inFlight.fetch_add(1);
  processAsyncBegin();
}

void Session::asyncEnd(Shared& shared) {
  shared.inFlight.fetch_sub(1);
  processAsyncEnd();
}

void Session::noteCompletionLocked(Shared& shared, const ODCompletion& c) {
  shared.completions.push_back(c);
  while (shared.completions.size() > kMaxPendingCompletions) {
    shared.completions.pop_front();
    ++shared.completionsDropped;
  }
}

id<MTLCommandBuffer> Session::newCommandBuffer() {
  const double from = hostSeconds();
  id<MTLCommandBuffer> cb = [ctx_->queue() commandBuffer];
  commandBufferMs_ += (hostSeconds() - from) * 1000.0;
  ++commandBuffers_;
  return cb;
}

const char* Session::firstAsyncError() {
  std::lock_guard<std::mutex> lock(shared_->mutex);
  firstAsyncErrorCopy_ = shared_->firstAsyncError;
  return firstAsyncErrorCopy_.c_str();
}

bool Session::validateFrame(const ODFrame& f, const char* what, std::string* err) const {
  char buf[256];
  auto bad = [&](const char* why) {
    snprintf(buf, sizeof(buf), "%s: %s", what, why);
    *err = buf;
    return false;
  };
  if (!f.mtlBuffer) return bad("mtlBuffer is NULL");
  if (f.width == 0 || f.height == 0) return bad("zero width or height");
  if (f.width > 16384 || f.height > 16384) return bad("larger than 16384 pixels");
  if (f.format != OD_PIXEL_BGRA32F && f.format != OD_PIXEL_BGRA16F) return bad("unknown pixel format");
  const uint32_t bpp = bytesPerPixel(f.format);
  if (f.rowBytes < f.width * bpp) return bad("rowBytes smaller than width * bytes per pixel");
  if (f.rowBytes % 4 != 0) return bad("rowBytes must be a multiple of 4");
  id<MTLBuffer> b = (__bridge id<MTLBuffer>)f.mtlBuffer;
  uint64_t need = (uint64_t)f.rowBytes * (f.height - 1) + (uint64_t)f.width * bpp;
  if ((uint64_t)b.length < need) return bad("MTLBuffer is too small for the frame");
  if (b.device != ctx_->device()) return bad("MTLBuffer belongs to another device");
  return true;
}

bool Session::beginEnv(RenderEnv& env, const ODParams& p, uint32_t w, uint32_t h) {
  env.ctx_ = ctx_.get();
  env.session_ = this;
  env.params = p;
  env.params.manualProfile = nullptr;
  env.profile = ODNoiseProfile{};
  env.width = w;
  env.height = h;
  env.cb_ = newCommandBuffer();
  cacheHits_ = 0;
  // With no command buffer every encode and commit would be a message to nil
  // and the render would "succeed" without writing the output.
  if (!env.cb_) {
    env.fail("the command queue gave no command buffer (GPU unavailable?)");
    return false;
  }
  return true;
}

double Session::flushAndWait(RenderEnv& env, bool startNew) {
  env.endEncoding();
  id<MTLCommandBuffer> cb = env.cb_;
  if (!cb) {
    env.fail("the command queue gave no command buffer (GPU unavailable?)");
    return -1;
  }
  const double committedAt = hostSeconds();
  [cb commit];
  [cb waitUntilCompleted];
  waitMs_ += (hostSeconds() - committedAt) * 1000.0;
  env.cb_ = startNew ? newCommandBuffer() : nil;
  if (cb.status == MTLCommandBufferStatusError) {
    env.fail("%s", cbError(cb).c_str());
    return -1;
  }
  {
    std::lock_guard<std::mutex> lock(shared_->mutex);
    queueTimes(cb, committedAt, &shared_->lastQueueMs, &shared_->lastLatencyMs);
  }
  // The buffer waited for is fine; the next one may not exist.
  if (startNew && !env.cb_) env.fail("the command queue gave no command buffer (GPU unavailable?)");
  return gpuMsOf(cb);
}

int Session::findEntry(int64_t key, uint32_t w, uint32_t h, int format) const {
  for (size_t i = 0; i < entries_.size(); ++i) {
    const CacheEntry& e = entries_[i];
    if (e.live && e.key == key && e.width == w && e.height == h && e.format == format) return (int)i;
  }
  return -1;
}

int Session::newEntry() {
  for (size_t i = 0; i < entries_.size(); ++i) {
    if (!entries_[i].live) return (int)i;
  }
  entries_.emplace_back();
  return (int)entries_.size() - 1;
}

uint64_t Session::entryBytes(const CacheEntry& e) const {
  uint64_t n = textureBytes(e.ingest);
  for (const CacheEntry::Slot& s : e.slots) n += textureBytes(s.tex);
  if (e.check) n += (uint64_t)e.check.allocatedSize;
  return n;
}

void Session::evictEntry(int index, ResourceList* out) {
  CacheEntry& e = entries_[(size_t)index];
  cacheBytes_ -= std::min(cacheBytes_, entryBytes(e));
  // A frame that comes back after its entry went is measured again: nothing
  // can tell whether the host still hands over the pixels the profile learnt
  // from (with the entry and its samples the current frame's check can, see
  // refreshCheck_). Callers hold useMutex_, like every user of measuredKeys.
  for (ProfileTrack& t : tracks_)
    t.measuredKeys.erase(std::remove(t.measuredKeys.begin(), t.measuredKeys.end(), e.key), t.measuredKeys.end());
  if (e.ingest) out->push_back(e.ingest);
  for (CacheEntry::Slot& s : e.slots)
    if (s.tex) out->push_back(s.tex);
  // The check buffer goes with the textures (pooled under the context's
  // budget, or freed). Command buffers still in flight that use it keep it
  // alive; a later render's use of it runs after them on the same queue.
  if (e.check) out->push_back(e.check);
  e = CacheEntry{};
}

void Session::enforceCacheBudget() {
  // Called after the render's commit: evicted textures were only used by
  // committed command buffers and can serve any later render from the pool.
  ResourceList out;
  for (size_t i = 0; i < entries_.size(); ++i) {
    if (entries_[i].live && renderIndex_ - entries_[i].lastUse >= kKeepRenders) evictEntry((int)i, &out);
  }
  while (cacheBytes_ > cacheBudgetBytes_) {
    int lru = -1;
    for (size_t i = 0; i < entries_.size(); ++i) {
      const CacheEntry& e = entries_[i];
      if (!e.live || e.lastUse == renderIndex_) continue;  // used by the render just done
      if (lru < 0 || e.lastUse < entries_[(size_t)lru].lastUse) lru = (int)i;
    }
    if (lru < 0) break;
    evictEntry(lru, &out);
  }
  memory_.recycle(out);
}

void Session::dropCaches(ResourceList* out) {
  for (size_t i = 0; i < entries_.size(); ++i) {
    if (entries_[i].live) evictEntry((int)i, out);
  }
  entries_.clear();
  cacheBytes_ = 0;
}

id<MTLTexture> Session::acquireTexture(MTLPixelFormat format, uint32_t w, uint32_t h, uint32_t mips) {
  if (mips < 1) mips = 1;
  for (size_t i = recycled_.size(); i-- > 0;) {
    id<MTLTexture> t = recycled_[i];
    if (t.pixelFormat == format && t.width == w && t.height == h && t.mipmapLevelCount == mips) {
      recycled_.erase(recycled_.begin() + (long)i);
      return t;
    }
  }
  return memory_.acquire(format, w, h, mips, this);
}

void Session::reset() {
  std::lock_guard<std::mutex> use(useMutex_);
  ResourceList out;
  dropCaches(&out);
  out.insert(out.end(), temps_.begin(), temps_.end());
  out.insert(out.end(), recycled_.begin(), recycled_.end());
  temps_.clear();
  recycled_.clear();
  memory_.release(out);  // freed off this thread
  for (ProfileTrack& t : tracks_) {
    t.stage->reset();
    t.measuredKeys.clear();
    t.coldWaits = 0;
    t.realtimeWaited = false;
  }
  seenKeys_.clear();
  std::lock_guard<std::mutex> lock(shared_->mutex);
  for (ProfileTrack& t : tracks_) *t.slot = ProfileSlot{};
  ++shared_->generation;
  shared_->analysis = {};
  shared_->governor.reset();
  shared_->poisoned = false;  // the caches it was about are gone; the first error is kept
  shared_->cutKeys.clear();
  shared_->decisiveNext = false;
  shared_->decisiveNextPair = 0;
  shared_->decisiveRetries = 0;
}

void Session::finishRender(bool failed) {
  // Called after commit (or after giving up): the textures of this render can
  // now serve any later render, of any session, whose command buffers run
  // after this one.
  ResourceList back;
  back.reserve(temps_.size() + recycled_.size());
  back.insert(back.end(), temps_.begin(), temps_.end());
  back.insert(back.end(), recycled_.begin(), recycled_.end());
  temps_.clear();
  recycled_.clear();
  memory_.recycle(back);
  if (failed) {
    // Cache entries created by this render may never have been filled.
    ResourceList dropped;
    dropCaches(&dropped);
    memory_.recycle(dropped);
    for (ProfileTrack& t : tracks_) t.measuredKeys.clear();
  } else {
    enforceCacheBudget();
  }
  lastUseMs_.store(nowMs());  // idle from now on
}

bool Session::ingestFrame(RenderEnv& env, const ODFrame& f, WorkFrame* wf, int* cacheHits, bool refresh,
                          bool cache, bool verify) {
  wf->frameKey = f.frameKey;
  wf->cacheIndex = -1;
  if (cache && verify && f.frameKey != OD_NO_KEY) {
    // ODParams.verifyFrameCache: the key does not promise the pixels behind it, so nothing
    // derived from them is reused, whatever a check could say. The frame goes into a scratch
    // texture and the stages derive what they need from it again (cacheIndex -1). Its entry
    // keeps only samples of the host buffer, for the automatic profile (below).
    wf->tex = env.tempWorking();
    if (!wf->tex || !ingest_->encode(env, f, wf->tex)) return false;
    // The check reads the buffer the way the default ingest path does.
    if (env.params.debugFlags & (OD_DEBUG_IO_BLIT | OD_DEBUG_IO_VIEW)) return true;
    int idx = findEntry(f.frameKey, f.width, f.height, (int)f.format);
    if (idx < 0) {
      idx = newEntry();
      CacheEntry& e = entries_[(size_t)idx];
      e.live = true;
      e.key = f.frameKey;
      e.width = f.width;
      e.height = f.height;
      e.format = (int)f.format;
    }
    CacheEntry& e = entries_[(size_t)idx];
    e.lastUse = renderIndex_;
    dropPixels(e);  // filled by a render that trusted the key: not for this one
    // A neighbour's samples stay those of the key's last render as the
    // current frame (or of its first appearance): what the profile measured.
    if (!refresh && e.checked) return true;
    // The current frame: its samples are compared with those of the key's
    // last render; a difference means the host changed something upstream
    // under the same key (a slip, an effect below, replaced media), which
    // the automatic profile must hear of (noteCuts). A heuristic for the
    // profile only: the picture never depends on it.
    id<MTLBuffer> before = e.checked ? e.check : nil;
    if (!encodeCheck(env, f, e) && env.failed()) return false;
    if (refresh && before && e.check == before && e.checked) refreshCheck_ = before;
    return true;
  }
  if (cache && f.frameKey != OD_NO_KEY) {
    // The frame cache proper: the caller vouches that a key names one
    // content (verifyFrameCache 0; a host with a trustworthy content
    // generation mixes it into the key). A neighbour found here is used as
    // it is, with what the stages derived from it.
    int idx = findEntry(f.frameKey, f.width, f.height, (int)f.format);
    if (idx >= 0 && entries_[(size_t)idx].ingest) {
      CacheEntry& e = entries_[(size_t)idx];
      e.lastUse = renderIndex_;
      wf->tex = e.ingest;
      wf->cacheIndex = idx;
      if (!refresh) {
        if (cacheHits) ++*cacheHits;
        return true;
      }
      // The current frame is never taken from the cache: the output is
      // "source + (result - ingested source)", so a cached copy that no longer
      // matches the host's buffer (the host changed something upstream and
      // kept the key) would put the difference between the two into the
      // picture. Ingest it again into the same texture and drop what the
      // stages derived from it. Command buffers of earlier renders that still
      // read these textures run before this one.
      for (CacheEntry::Slot& sl : e.slots) {
        cacheBytes_ -= std::min(cacheBytes_, textureBytes(sl.tex));
        recycled_.push_back(sl.tex);
      }
      e.slots.clear();
      e.checked = false;  // samples, if a verifying render took any, no longer describe the texture
      return ingest_->encode(env, f, wf->tex);
    }
    // A miss, or an entry that holds only samples (a verifying render's).
    if (idx < 0) {
      idx = newEntry();
      CacheEntry& e = entries_[(size_t)idx];
      e.live = true;
      e.key = f.frameKey;
      e.width = f.width;
      e.height = f.height;
      e.format = (int)f.format;
    }
    entries_[(size_t)idx].lastUse = renderIndex_;  // in use: an eviction below never takes it
    id<MTLTexture> t = acquireTexture(kWorkingFormat, f.width, f.height, 1);
    if (!t) {
      env.fail("could not allocate a %ux%u working texture", f.width, f.height);
      return false;
    }
    CacheEntry& e = entries_[(size_t)idx];
    e.ingest = t;
    e.checked = false;
    cacheBytes_ += textureBytes(t);
    wf->tex = t;
    wf->cacheIndex = idx;
    return ingest_->encode(env, f, wf->tex);
  }
  wf->tex = env.tempWorking();
  if (!wf->tex) return false;
  return ingest_->encode(env, f, wf->tex);
}

void Session::dropPixels(CacheEntry& e) {
  // Textures go back with this render's (only it may reuse them before its
  // commit); earlier command buffers that read them run first.
  if (e.ingest) {
    cacheBytes_ -= std::min(cacheBytes_, textureBytes(e.ingest));
    recycled_.push_back(e.ingest);
    e.ingest = nil;
  }
  for (CacheEntry::Slot& sl : e.slots) {
    cacheBytes_ -= std::min(cacheBytes_, textureBytes(sl.tex));
    recycled_.push_back(sl.tex);
  }
  e.slots.clear();
}

bool Session::encodeCheck(RenderEnv& env, const ODFrame& f, CacheEntry& e) {
  if (!check32_ || !check16_) {
    std::string err;
    check32_ = ctx_->pipeline("ingest_check_bgra32f", &err);
    check16_ = ctx_->pipeline("ingest_check_bgra16f", &err);
    if (!check32_ || !check16_) {
      e.checked = false;
      return false;  // no samples: the profile hears of no upstream change
    }
  }
  const uint32_t bx = (f.width + kCheckBlock - 1) / kCheckBlock, by = (f.height + kCheckBlock - 1) / kCheckBlock;
  const NSUInteger need = kCheckHeaderBytes + (NSUInteger)bx * by * 16;
  if (!e.check || e.check.length != need) {
    if (e.check) {
      cacheBytes_ -= std::min(cacheBytes_, (uint64_t)e.check.allocatedSize);
      ResourceList old{e.check};
      memory_.release(old);
    }
    e.checked = false;
    // From the context's pool, counted in its budget. A buffer that served
    // another entry carries that entry's samples and stamp: the first check
    // below only takes samples, and stamps are unique in the context.
    e.check = memory_.acquireBuffer(need, this);
    if (!e.check) return false;
    cacheBytes_ += (uint64_t)e.check.allocatedSize;
  }
  CheckUniforms u = {};
  u.rowStride = f.rowBytes / 4;
  u.width = f.width;
  u.height = f.height;
  if (!checkStamp_) checkStamp_ = memory_.nextStamp();
  u.stamp = checkStamp_;
  u.blocksX = bx;
  u.blocksY = by;
  id<MTLComputeCommandEncoder> enc = env.compute(f.format == OD_PIXEL_BGRA16F ? "ingest_check_bgra16f" : "ingest_check_bgra32f");
  [enc setBuffer:(__bridge id<MTLBuffer>)f.mtlBuffer offset:0 atIndex:0];
  [enc setBuffer:e.check offset:0 atIndex:1];
  [enc setBytes:&u length:sizeof(u) atIndex:2];
  env.dispatch(f.format == OD_PIXEL_BGRA16F ? check16_ : check32_, bx, by);
  e.checked = true;
  return true;
}

ProfileTrack* Session::profileTrack(uint32_t w, uint32_t h, std::string* err, bool chromaRecovery) {
  ProfileTrack* unclaimed = nullptr;
  ProfileTrack* lru = nullptr;
  size_t modeSizes = 0;
  for (ProfileTrack& t : tracks_) {
    if (t.width == w && t.height == h && t.chromaRecovery == chromaRecovery) {
      t.lastUse = renderIndex_;
      return &t;
    }
    if (t.width == 0 && !unclaimed) unclaimed = &t;
    if (t.width != 0 && t.chromaRecovery == chromaRecovery) {
      ++modeSizes;
      if (!lru || t.lastUse < lru->lastUse) lru = &t;
    }
  }
  ProfileTrack* t = unclaimed;
  if (!t) {
    // A size of its own: a new stage, so nothing of another size (its ring,
    // throttles, estimates still in flight) can reach this one.
    std::shared_ptr<ProfileStage> stage(makeProfileStage(*ctx_, err));
    if (!stage) return nullptr;
    if (modeSizes < kMaxProfileSizes) {
      tracks_.emplace_back();
      t = &tracks_.back();
    } else {
      t = lru;  // the size used longest ago gives way
      *t = ProfileTrack{};
    }
    t->stage = stage;
    t->slot = std::make_shared<ProfileSlot>();
  }
  t->width = w;
  t->height = h;
  t->chromaRecovery = chromaRecovery;
  t->lastUse = renderIndex_;
  return t;
}

ODNoiseProfile Session::fallbackProfile(const ProfileTrack* except, int grainMode) {
  // Called with shared_->mutex held. Another size's profile of the same
  // grain mode.
  const double want = except ? (double)except->width * except->height : 0;
  const ProfileTrack* best = nullptr;
  double bestDist = 0;
  for (const ProfileTrack& t : tracks_) {
    const ProfileSlot::Mode& sl = t.slot->mode[grainMode];
    if (&t == except || (except && t.chromaRecovery != except->chromaRecovery) ||
        sl.updates == 0 || !sl.profile.valid) continue;
    const double d = std::fabs(std::log(std::max(1.0, (double)t.width * t.height) / std::max(1.0, want)));
    if (!best || d < bestDist) {
      best = &t;
      bestDist = d;
    }
  }
  return best ? best->slot->mode[grainMode].profile : ODNoiseProfile{};
}

void Session::resolveInto(Shared& shared, ProfileStage& stage, ProfileSlot& slot, const ProfileMeasurement& m,
                          uint64_t generation) {
  // The blocking path and the profile queue both come here. resolve() runs
  // outside the lock (a few ms of CPU); the estimate only ever goes into the
  // slot of the frame size it was measured at.
  ODNoiseProfile fresh = {};
  uint32_t knots = 0;
  const bool got = stage.resolve(m, &fresh, &knots) && fresh.valid;
  std::lock_guard<std::mutex> lock(shared.mutex);
  if (shared.generation != generation) return;  // od_session_reset, or a restart, since
  // Hold profile: the slot stays as it is. The slot of the grain mode in use
  // decides whether the estimate is dropped altogether (a session used in
  // one mode behaves as if the other did not exist); the other mode's slot
  // is left alone too if it is held.
  const int active = shared.grainMode;
  if (shared.hold && profileHeld(slot.mode[active])) return;
  if (m.decisive) {
    // An unusable estimate where the footage may have changed restarts
    // nothing (smooth()): measure the next frame decisively too, a couple
    // of times, instead of waiting for two usable far estimates in a row.
    if (!got || fresh.confidence < kUsableConfidence) {
      if (shared.decisiveRetries > 0 && !shared.decisiveNext) {
        --shared.decisiveRetries;
        shared.decisiveNext = true;
        shared.decisiveNextPair = 0;
      }
    } else {
      shared.decisiveRetries = 0;
    }
  }
  if (!got) return;
  ODNoiseProfile* slots[kGrainModes];
  for (int k = 0; k < kGrainModes; ++k)
    slots[k] = k != active && shared.hold && profileHeld(slot.mode[k]) ? nullptr : &slot.mode[k].profile;
  stage.smooth(fresh, knots, slots, m.decisive);
  for (int k = 0; k < kGrainModes; ++k)
    if (slots[k]) ++slot.mode[k].updates;
}

void Session::boundedResolve(RenderEnv& env, ProfileTrack& track, const ProfileMeasurement& m, uint64_t generation) {
  env.endEncoding();
  id<MTLCommandBuffer> cb = env.cb_;
  // Who resolves: this thread if the GPU is done within kRealtimeWaitMs,
  // otherwise the completion handler hands it to the profile queue.
  struct Handoff {
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    std::atomic<int> state{0};  // 0 waiting, 1 GPU done, 2 the render thread gave up
    std::atomic<bool> failed{false};
  };
  auto h = std::make_shared<Handoff>();
  std::shared_ptr<Shared> shared = shared_;
  std::shared_ptr<ProfileStage> stage = track.stage;
  std::shared_ptr<ProfileSlot> slot = track.slot;
  ProfileMeasurement mm = m;
  std::shared_ptr<std::atomic<bool>> renderFailed = renderGpuFailed_;
  asyncBegin(*shared);
  [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
    const bool error = c.status == MTLCommandBufferStatusError;
    if (error) h->failed.store(true);
    const bool gaveUp = h->state.exchange(1) == 2;
    if (error) {
      std::lock_guard<std::mutex> lock(shared->mutex);
      shared->poisoned = true;
      // The render stopped waiting and went on: it may well return 0, so
      // the failure is the next call's to report (once per render).
      if (gaveUp && renderFailed && !renderFailed->exchange(true)) {
        if (shared->firstAsyncError.empty()) shared->firstAsyncError = cbError(c);
        ++shared->asyncFailures;
      }
    }
    if (gaveUp) {
      if (error) {
        asyncEnd(*shared);
      } else {
        dispatch_async(shared->profileQueue, ^{
          resolveInto(*shared, *stage, *slot, mm, generation);
          asyncEnd(*shared);
        });
      }
    }
    dispatch_semaphore_signal(h->done);
  }];
  [cb commit];
  env.cb_ = newCommandBuffer();
  if (!env.cb_) env.fail("the command queue gave no command buffer (GPU unavailable?)");
  const double waitFrom = hostSeconds();
  const bool inTime =
      dispatch_semaphore_wait(h->done, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(kRealtimeWaitMs * 1e6))) == 0;
  waitMs_ += (hostSeconds() - waitFrom) * 1000.0;
  // The GPU may finish between the timeout and this exchange: then the
  // handler has already run and left the estimate to this thread.
  const bool here = inTime || h->state.exchange(2) == 1;
  if (!here) return;
  if (h->failed.load()) {
    env.fail("%s", cbError(cb).c_str());
  } else {
    resolveInto(*shared_, *track.stage, *track.slot, m, generation);
  }
  asyncEnd(*shared);
}

void Session::noteCuts(Shared& shared, const GateReadback& g, uint64_t generation) {
  if (g.stampBuffer) {
    const uint32_t word = *(const uint32_t*)g.stampBuffer.contents;
    std::lock_guard<std::mutex> lock(shared.mutex);
    if (shared.gateBuffers.size() < 8) shared.gateBuffers.push_back(g.stampBuffer);
    if (word == g.stamp && shared.generation == generation) {
      // New pixels under a key the session had sampled: whatever the
      // profile learnt from that frame is about other footage. The next
      // render measures decisively (a far estimate replaces the history).
      if (g.stampKey != OD_NO_KEY &&
          std::find(shared.changedKeys.begin(), shared.changedKeys.end(), g.stampKey) == shared.changedKeys.end()) {
        shared.changedKeys.push_back(g.stampKey);
        if (shared.changedKeys.size() > 16) shared.changedKeys.pop_front();
      }
      if (!shared.decisiveNext) {
        shared.decisiveNext = true;
        shared.decisiveNextPair = 0;
      }
    }
  }
  if (!g.buffer) return;
  // RG16Float rows: x = weight, y = cut value.
  const __fp16* h = (const __fp16*)g.buffer.contents;
  int aheadOffset = OD_MAX_RADIUS + 1;
  int64_t aheadKey = OD_NO_KEY;
  bool startsShot = false;
  for (int a = 0; a < g.count; ++a) {
    const float cut = (float)h[2 * a + 1];
    if (!(cut > 0.5f)) continue;
    const int off = g.offsets[a];
    if (off > 0 && off < aheadOffset) {
      aheadOffset = off;
      aheadKey = g.keys[a];
    }
    if (off == -1) startsShot = true;  // the cut lies between this frame and the one before
  }
  std::lock_guard<std::mutex> lock(shared.mutex);
  if (shared.gateBuffers.size() < 8) shared.gateBuffers.push_back(g.buffer);
  if (shared.generation != generation) return;
  if (aheadOffset <= OD_MAX_RADIUS) {
    if (aheadKey != OD_NO_KEY) {
      // The first frame after the cut: restart when it is rendered.
      if (std::find(shared.cutKeys.begin(), shared.cutKeys.end(), aheadKey) == shared.cutKeys.end()) {
        shared.cutKeys.push_back(aheadKey);
        if (shared.cutKeys.size() > 16) shared.cutKeys.pop_front();
      }
    } else if (aheadOffset == 1) {
      shared.decisiveNext = true;  // no keys: the next render, in forward playback
      shared.decisiveNextPair = 1;
    }
  }
  if (startsShot && !g.decisive && !shared.decisiveNext) {
    shared.decisiveNext = true;  // this frame started a shot: correct the profile from the next one on
    shared.decisiveNextPair = 0;
  }
}

bool Session::isJump(const ODFrame* frames, int frameCount) const {
  // The current frame alone (radius 0, or a host that delivered none of the
  // neighbours) says nothing about where the host is: its key is new on every
  // render of sequential playback.
  if (frameCount < 2) return false;
  bool keyed = false;
  for (int i = 0; i < frameCount; ++i) {
    if (frames[i].frameKey == OD_NO_KEY) continue;
    keyed = true;
    if (std::find(seenKeys_.begin(), seenKeys_.end(), frames[i].frameKey) != seenKeys_.end()) return false;
  }
  return keyed;
}

void Session::rememberKeys(const ODFrame* frames, int frameCount) {
  for (int i = 0; i < frameCount; ++i) {
    const int64_t k = frames[i].frameKey;
    if (k == OD_NO_KEY || std::find(seenKeys_.begin(), seenKeys_.end(), k) != seenKeys_.end()) continue;
    seenKeys_.push_back(k);
    if (seenKeys_.size() > 96) seenKeys_.pop_front();
  }
}

int Session::render(const ODFrame* frames, const int* offsets, int frameCount, const ODParams* params,
                    const ODFrame* out, ODStats* stats, bool deterministic,
                    const ODQualityRequest* quality, ODQualityDecision* decision) {
  const double enteredAt = quality ? hostSeconds() * 1000.0 : 0;
  if (decision) *decision = {};
  // ODStats.cpuMs counts the waits before the render starts too: admission
  // against the device's budget, then the session's lock (ODStats.lockWaitMs).
  const double t0 = nowMs();
  // Admission against the device's budget (memory.h): the working set of a
  // render in progress cannot be evicted, so the renders of every context on
  // the device take turns when theirs do not fit together. Before useMutex_,
  // so that the cache of a session waiting here can still be evicted. Ends
  // when this call returns, after its scratch went back to the pool.
  RenderShape shape;
  if (frames && params && frameCount >= 1) {
    shape = {frames[0].width, frames[0].height, (int)frames[0].format, frameCount,
             std::min(OD_MAX_RADIUS, std::max(0, params->temporalRadius)), params->quality};
  }
  const uint64_t estimate = estimateHeld(shape);
  const bool admitRealtime = params && params->realtime != 0 && params->waitUntilCompleted == 0 &&
                             !(params->debugFlags & OD_DEBUG_STAGE_TIMINGS);
  memory_.admitRender(estimate, admitRealtime);
  struct Admitted {
    Memory& memory;
    uint64_t bytes;
    ~Admitted() { memory.endRender(bytes); }
  } admitted{memory_, estimate};
  std::lock_guard<std::mutex> use(useMutex_);
  const double lockWaitMs = nowMs() - t0;
  lastUseMs_.store(nowMs());
  holdsMemory_.store(true);
  memory_.noteActivity();
  @autoreleasepool {
    if (stats) *stats = ODStats{};
    lastError_.clear();
    if (!initError_.empty()) return setError(initError_);
    if (!frames || !offsets || !params || !out) return setError("od_render: NULL argument");
    if (frameCount < 1 || frameCount > OD_MAX_FRAMES) return setError("od_render: frameCount out of range");
    if (offsets[0] != 0) return setError("od_render: offsets[0] must be 0");
    std::string verr;
    if (!validateFrame(frames[0], "frames[0]", &verr)) return setError(verr);
    if (!validateFrame(*out, "out", &verr)) return setError(verr);
    if (out->width != frames[0].width || out->height != frames[0].height || out->format != frames[0].format)
      return setError("od_render: out must have the size and format of frames[0]");
    for (int i = 1; i < frameCount; ++i) {
      char what[32];
      snprintf(what, sizeof(what), "frames[%d]", i);
      if (!validateFrame(frames[i], what, &verr)) return setError(verr);
      if (frames[i].width != frames[0].width || frames[i].height != frames[0].height ||
          frames[i].format != frames[0].format)
        return setError("od_render: all frames must share width, height and format");
      if (frames[i].mtlBuffer == out->mtlBuffer && frames[0].mtlBuffer != out->mtlBuffer)
        return setError("od_render: out may alias frames[0] only");
    }

    // Sanitised copy of the params.
    ODParams p = *params;
    p.lumaStrength = clamp01(p.lumaStrength);
    p.chromaStrength = clamp01(p.chromaStrength);
    p.temporalStrength = clamp01(p.temporalStrength);
    p.spatialStrength = clamp01(p.spatialStrength);
    p.sharpen = clamp01(p.sharpen);
    p.temporalRadius = std::min(OD_MAX_RADIUS, std::max(0, p.temporalRadius));
    if (!std::isfinite(p.noiseLevel) || p.noiseLevel < 0) p.noiseLevel = 1.0f;
    p.noiseLevel = std::min(p.noiseLevel, kMaxNoiseLevel);
    sanitizeAdvanced(&p);
    // The manual profile the stages see (and ODStats reports): checked and
    // clamped here, once, for every stage.
    ODNoiseProfile manual = {};
    const bool useManual = params->manualProfile && params->manualProfile->valid;
    if (useManual && !sanitizeProfile(*params->manualProfile, &manual))
      return setError("od_render: manualProfile contains non-finite values");
    const bool timing = (p.debugFlags & OD_DEBUG_STAGE_TIMINGS) != 0;
    // Analysis is a still-frame diagnostic too (seek/export). Its readback
    // must finish before the overlay is drawn, even on a realtime request.
    const bool wait = timing || p.waitUntilCompleted != 0 || p.view == OD_VIEW_ANALYSIS;
    // The host is playing: nothing in this call waits for the GPU to run
    // the work ahead of ours (the queue may be the host's, with several
    // frames of it), beyond the bounded first estimate. Obtaining a command
    // buffer can still block on a full queue (commandBufferMs_).
    const bool realtime = p.realtime != 0 && !wait;
    ODQualityDecision selected = {};
    double callStart = enteredAt;
    if (quality) {
      // Hash initialized individual fields, never struct padding or pointers.
      // New settings/raster start a new measurement domain. No old completion
      // can downgrade it. Profile changes themselves do not reset the governor.
      uint64_t config = 1469598103934665603ull;
      auto mix = [&](const auto& v) {
        const unsigned char* b = reinterpret_cast<const unsigned char*>(&v);
        for (size_t i = 0; i < sizeof(v); ++i) config = (config ^ b[i]) * 1099511628211ull;
      };
      mix(frames[0].width); mix(frames[0].height); mix(frames[0].format);
      mix(p.lumaStrength); mix(p.chromaStrength); mix(p.temporalRadius);
      mix(p.temporalStrength); mix(p.spatialStrength); mix(p.sharpen); mix(p.noiseLevel);
      mix(p.view); mix(p.debugFlags); mix(p.debugSkipStages); mix(p.noFrameCache); mix(p.verifyFrameCache);
      mix(p.advanced); mix(p.bandNoiseLevel); mix(p.bandAmount); mix(p.profileHold);
      mix(p.channelNoiseLevel); mix(p.grainFloor); mix(p.speckRemoval);
      mix(p.temporalMotionThreshold); mix(p.temporalEdgeGate);
      mix(p.grainStability); mix(p.coarseChroma); mix(p.chromaRecovery);
      mix(p.temporalNoiseModel);
      mix(useManual); if (useManual) mix(manual);
      mix(p.sharpenBand); mix(p.sharpenCoring); mix(p.sharpenHalo);
      if (std::isfinite(quality->callerWaitMs) && quality->callerWaitMs > 0)
        callStart -= quality->callerWaitMs;
      std::lock_guard<std::mutex> lock(shared_->mutex);
      selected = shared_->governor.begin(*quality, realtime, p.quality, config, enteredAt);
      p.quality = selected.quality;
      if (decision) *decision = selected;
    }

    const uint32_t skip = p.debugSkipStages;
    const bool cacheFrames = p.noFrameCache == 0 && !deterministic;
    const bool verifyCache = cacheFrames && p.verifyFrameCache != 0;
    waitMs_ = 0;
    commandBufferMs_ = 0;
    commandBuffers_ = 0;
    renderId_ = ++gRenderIds;
    renderGpuFailed_ = std::make_shared<std::atomic<bool>>(false);
    // From here on, on every way out, success or failure: what concerns the
    // host rather than this render's picture (API v7). Asynchronous failures
    // and completions are handed over exactly once.
    struct Report {
      Session* s;
      ODStats* stats;
      double lockWaitMs;
      ~Report() {
        Shared& sh = *s->shared_;
        std::lock_guard<std::mutex> lock(sh.mutex);
        if (!stats) {
          sh.completions.clear();  // nobody to report them to
          return;
        }
        stats->lockWaitMs = lockWaitMs;
        stats->commandBufferMs = s->commandBufferMs_;
        stats->renderId = s->renderId_;
        stats->asyncFailures = sh.asyncFailures;
        sh.asyncFailures = 0;
        stats->completedDropped = sh.completionsDropped;
        sh.completionsDropped = 0;
        int n = 0;
        while (n < OD_MAX_COMPLETIONS && !sh.completions.empty()) {
          stats->completed[n++] = sh.completions.front();
          sh.completions.pop_front();
        }
        stats->completedCount = n;
        stats->inFlight = sh.inFlight.load();
      }
    } report{this, stats, lockWaitMs};
    // Frames whose pixels changed under their key in an earlier render (the
    // frame cache check): neither measured nor seen any more, so they are
    // measured again, and the decisive measurement noteCuts asked for can
    // restart the profile on them.
    {
      std::deque<int64_t> changed;
      {
        std::lock_guard<std::mutex> lock(shared_->mutex);
        changed.swap(shared_->changedKeys);
      }
      for (int64_t k : changed) {
        seenKeys_.erase(std::remove(seenKeys_.begin(), seenKeys_.end(), k), seenKeys_.end());
        for (ProfileTrack& t : tracks_)
          t.measuredKeys.erase(std::remove(t.measuredKeys.begin(), t.measuredKeys.end(), k), t.measuredKeys.end());
      }
    }
    // None of this window's frames rendered recently: scrubbing, or parked on
    // another shot. The footage may differ from what the profile describes.
    ODFrame analysisFrames[OD_MAX_FRAMES];
    int analysisCount = 1;
    analysisFrames[0] = frames[0];
    if (p.temporalNoiseModel) {
      for (int i = 1; i < frameCount; ++i)
        if (offsets[i] != 0 && std::abs(offsets[i]) <= OD_PROFILE_RADIUS)
          analysisFrames[analysisCount++] = frames[i];
    }
    const bool jump = p.temporalNoiseModel ? isJump(analysisFrames, analysisCount) : isJump(frames, frameCount);

    {
      std::lock_guard<std::mutex> lock(shared_->mutex);
      if (shared_->poisoned) {
        // The failure itself is reported by this call (Report above) and its
        // message kept (od_first_async_error); the caches may hold frames
        // that were never written.
        shared_->poisoned = false;
        ResourceList dropped;
        dropCaches(&dropped);  // an earlier asynchronous render failed on the GPU
        memory_.recycle(dropped);
      }
    }
    ++renderIndex_;
    checkStamp_ = 0;  // a new one if this render checks a cached frame
    refreshCheck_ = nil;

    // Which neighbours take part: |offset| <= radius, no duplicates.
    int order[OD_MAX_FRAMES];
    int used = 1;
    order[0] = 0;
    for (int i = 1; i < frameCount; ++i) {
      const int radius = p.temporalNoiseModel ? std::max(p.temporalRadius, OD_PROFILE_RADIUS) : p.temporalRadius;
      if (offsets[i] == 0 || std::abs(offsets[i]) > radius) continue;
      bool dup = false;
      for (int k = 1; k < used; ++k) dup = dup || offsets[order[k]] == offsets[i];
      if (!dup) order[used++] = i;
    }
    std::sort(order + 1, order + used, [&](int a, int b) {
      int da = std::abs(offsets[a]), db = std::abs(offsets[b]);
      return da != db ? da < db : offsets[a] < offsets[b];
    });
    int mergeCount = used, profileCount = used;
    if (p.temporalNoiseModel) {
      mergeCount = profileCount = 1;
      for (int k = 1; k < used; ++k) {
        if (std::abs(offsets[order[k]]) <= p.temporalRadius) ++mergeCount;
        if (std::abs(offsets[order[k]]) <= OD_PROFILE_RADIUS) ++profileCount;
      }
    }

    // Stage decisions (stage.h, section 1).
    const bool anyStrength = p.lumaStrength > 0 || p.chromaStrength > 0;
    const bool wantTemporal = p.temporalStrength > 0 && anyStrength && mergeCount > 1;
    const bool wantSpatial = p.spatialStrength > 0 && anyStrength;
    const bool wantSharpen = p.sharpen > 0;
    const bool doTemporal = wantTemporal && !(skip & stageBit(OD_STAGE_TEMPORAL));
    const bool doSpatial = wantSpatial && !(skip & stageBit(OD_STAGE_SPATIAL));
    const bool doSharpen = wantSharpen && !(skip & stageBit(OD_STAGE_SHARPEN));
    const bool identity = !wantTemporal && !wantSpatial && !wantSharpen && p.view == OD_VIEW_OUTPUT &&
                          !(p.debugFlags & OD_DEBUG_FORCE_PROFILE);

    const bool viewProfile = p.view == OD_VIEW_PROFILE && !(skip & stageBit(OD_STAGE_PROFILE));
    const bool forceProfile = (p.debugFlags & OD_DEBUG_FORCE_PROFILE) != 0;

    RenderEnv env;
    if (!beginEnv(env, p, frames[0].width, frames[0].height)) {
      finishRender(true);
      return 1;
    }
    // What the stages are told (stage.h section 1). The profile view replaces
    // all three; the motion view ends the pipeline after temporal.
    env.temporalRuns = doTemporal && !viewProfile;
    env.spatialRuns = doSpatial && !viewProfile && !(p.view == OD_VIEW_MOTION && doTemporal);
    env.sharpenRuns = doSharpen && !viewProfile && !(p.view == OD_VIEW_MOTION && doTemporal);
    // The bands step (a band amount below 1): after a denoising stage, never
    // in the views that replace the picture (the profile view clears both
    // flags above; the motion view replaces it only when the temporal stage
    // runs, otherwise the denoised picture goes out, amounts and all). It is
    // called only if one of them returned a new image (below).
    // The grain mode of this render (ODParams.grainMode, or the session's
    // test override): where the grain floor applies, and which of the
    // session's automatic profiles it reads (gm).
    const int gm = grainOverride_ >= 0 ? grainOverride_ : (int)p.grainMode;
    env.grainProfileV2 = gm == OD_GRAIN_FINEST;
    env.bandsRun = (env.temporalRuns || env.spatialRuns) && anyBandAmount(p, env.grainProfileV2) &&
                   !(p.view == OD_VIEW_MOTION && doTemporal);
    double gpuMs = 0;
    double stageMs[OD_STAGE_COUNT] = {0};
    int framesUsed = 1;
    int hits = 0;
    ProfileMeasurement pending;
    bool decisive = false;
    GateReadback gate;
    AnalysisReadback analysis;
    analysis.snapshot.valid = 1; analysis.snapshot.renderId = renderId_;
    analysis.snapshot.frameKey = frames[0].frameKey;
    analysis.snapshot.width = env.width; analysis.snapshot.height = env.height;
    std::shared_ptr<ProfileStage> pendingStage;
    std::shared_ptr<ProfileSlot> pendingSlot;
    uint64_t generation = 0;
    {
      std::lock_guard<std::mutex> lock(shared_->mutex);
      generation = shared_->generation;
    }

    // In timing mode every stage gets its own command buffer, waited for.
    auto boundary = [&](ODStage s) -> bool {
      if (env.failed()) return false;
      if (!timing) return true;
      double ms = flushAndWait(env, true);
      if (ms < 0) return false;
      stageMs[s] += ms;
      gpuMs += ms;
      return !env.failed();
    };

    WorkFrame work[OD_MAX_FRAMES];
    bool ok = true;

    if (identity) {
      // Nothing to remove: copy (or leave the buffer alone when in place).
      ok = output_->encodeCopy(env, frames[0], *out) && boundary(OD_STAGE_OUTPUT);
    } else {
      // ---- ingest ----
      const int ingestCount = p.temporalNoiseModel ? used : doTemporal ? used : 1;
      // Every cached frame of this window is in use from now on: a texture
      // this render allocates must not evict one of them (memory budget).
      for (int k = 0; k < ingestCount; ++k) {
        const ODFrame& f = frames[order[k]];
        if (!cacheFrames || f.frameKey == OD_NO_KEY) continue;
        const int idx = findEntry(f.frameKey, f.width, f.height, (int)f.format);
        if (idx >= 0) entries_[(size_t)idx].lastUse = renderIndex_;
      }
      for (int k = 0; ok && k < ingestCount; ++k) {
        work[k].offset = offsets[order[k]];
        ok = ingestFrame(env, frames[order[k]], &work[k], &hits, /*refresh*/ k == 0, cacheFrames, verifyCache);
      }
      env.frames = work;
      env.frameCount = p.temporalNoiseModel ? profileCount : ingestCount;
      ok = ok && boundary(OD_STAGE_INGEST);

      // ---- profile ----
      const bool anyStage = doTemporal || doSpatial || doSharpen || p.view == OD_VIEW_PROFILE || p.view == OD_VIEW_ANALYSIS || forceProfile;
      ProfileTrack* track = nullptr;
      ProfileAnalysisInfo analysisInfo;
      if (ok && (viewProfile || (!useManual && anyStage && !(skip & stageBit(OD_STAGE_PROFILE))))) {
        std::string terr;
        track = profileTrack(env.width, env.height, &terr, p.chromaRecovery != 0);
        if (!track) {
          env.fail("%s", terr.empty() ? "profile stage creation failed" : terr.c_str());
          ok = false;
        }
      }
      // Hold profile: once this size's automatic profile is established, no
      // measurement, no blocking, no restart at a cut or a jump (cut keys and
      // pending restarts are left for when Hold goes off); the slot is used
      // as it is, and estimates still in flight are dropped (resolveInto).
      bool held = false;
      if (ok && track && !useManual) {
        std::lock_guard<std::mutex> lock(shared_->mutex);
        shared_->hold = p.profileHold != 0;
        shared_->grainMode = gm;
        held = shared_->hold && profileHeld(track->slot->mode[gm]);
      }
      if (ok && useManual) {
        env.profile = manual;
      } else if (ok && deterministic && anyStage && !(skip & stageBit(OD_STAGE_PROFILE))) {
        // od_render_deterministic (profiling.h): this window's own estimate,
        // measured and resolved before filtering; no slot, no history, no
        // Hold (a held slot is history).
        ProfileMeasurement m;
        m.width = env.width;
        m.height = env.height;
        m.frameKey = work[0].frameKey;
        m.force = true;
        m.deterministic = true;
        ok = track->stage->encodeMeasure(env, work[0], m) && !env.failed();
        if (ok && m.data) {
          const double ms = flushAndWait(env, true);
          ok = ms >= 0 && !env.failed();
          if (ok) {
            gpuMs += ms;
            stageMs[OD_STAGE_PROFILE] += ms;
            // Never borrow another render's estimate when evidence is missing.
            ODNoiseProfile fresh = {};
            if (track->stage->resolve(m, &fresh, nullptr)) {
              env.profile = fresh;
              std::lock_guard<std::mutex> lock(shared_->mutex); analysisInfo = track->stage->analysisInfo(gm);
            }
          }
        }
      } else if (ok && held && anyStage && !(skip & stageBit(OD_STAGE_PROFILE))) {
        std::lock_guard<std::mutex> lock(shared_->mutex);
        env.profile = track->slot->mode[gm].profile;
        analysisInfo = track->stage->analysisInfo(gm);
      } else if (ok && anyStage && !(skip & stageBit(OD_STAGE_PROFILE))) {
        // The automatic profile of this frame size. A size it has not yet
        // measured usably renders cold (blocks for its first estimates, at
        // most kColdWaits renders), like a new session, instead of using
        // another size's noise figures; a realtime render borrows them, or
        // waits a bounded time once.
        const int64_t key = work[0].frameKey;
        bool cold, scanWindow;
        bool haveProfile;  // something to render with if this render does not wait
        int pairOffset = 0;
        {
          std::lock_guard<std::mutex> lock(shared_->mutex);
          const ProfileSlot::Mode& sl = track->slot->mode[gm];
          const bool usable = sl.updates > 0 && sl.profile.valid && sl.profile.confidence >= kUsableConfidence;
          cold = !usable && track->coldWaits < kColdWaits;
          // Version 2 establishes the initial estimate from the available
          // profiling window before returning its very first frame.
          scanWindow = p.temporalNoiseModel && sl.updates == 0 && track->coldWaits == 0;
          haveProfile = (sl.updates > 0 && sl.profile.valid) || fallbackProfile(track, gm).valid;
          // New footage may start here (stage.h, section 5): the first frame
          // after a cut the temporal stage saw, the frame after one that
          // started a shot, or a jump. Only with a profile to restart.
          bool fresh = jump;  // a new event, not a retry
          if (key != OD_NO_KEY) {
            auto it = std::find(shared_->cutKeys.begin(), shared_->cutKeys.end(), key);
            if (it != shared_->cutKeys.end()) {
              shared_->cutKeys.erase(it);
              decisive = true;
              fresh = true;
              pairOffset = 1;  // the frames before this one are the other shot
            }
          }
          if (shared_->decisiveNext) {
            shared_->decisiveNext = false;
            if (!decisive) pairOffset = shared_->decisiveNextPair;
            decisive = true;
          }
          decisive = (decisive || jump) && sl.updates > 0;
          if (decisive && fresh) shared_->decisiveRetries = 2;
        }
        // Coarse chroma cleanup is estimated only inside a measurement, and a
        // key is measured once: when the switch changes, the keys measured
        // so far count as unmeasured (a parked frame, a replayed range).
        // Switched on, this frame's measurement is forced, and a render that
        // may wait waits for it, so the frame it returns is corrected.
        bool restate = false;
        if (coarseChromaOn(p) != track->measuredCoarse) {
          track->measuredCoarse = !track->measuredCoarse;
          track->measuredKeys.clear();
          restate = track->measuredCoarse;
        }
        const bool measured = key != OD_NO_KEY && std::find(track->measuredKeys.begin(), track->measuredKeys.end(),
                                                            key) != track->measuredKeys.end();
        ProfileMeasurement m;
        m.width = env.width;
        m.height = env.height;
        m.frameKey = key;
        m.force = decisive || cold || restate;  // never throttled while the size still needs a usable estimate
        m.decisive = decisive;
        m.pairOffset = decisive ? pairOffset : 0;
        if (!measured || cold || decisive) {
          ok = track->stage->encodeMeasure(env, work[0], m) && !env.failed();
          if (ok && m.data && key != OD_NO_KEY && !measured) {
            track->measuredKeys.push_back(key);
            if (track->measuredKeys.size() > 64) track->measuredKeys.pop_front();
          }
        }
        // How this render gets its estimate: blocking (a size's first
        // estimates, new footage, timing mode), a bounded wait (a realtime
        // render of a size with nothing to render with yet), or later.
        const bool block = ok && m.data && (scanWindow || timing || (!realtime && (cold || decisive || restate)));
        const bool bounded = ok && m.data && !block && realtime && cold && !haveProfile && !track->realtimeWaited;
        if (ok && m.data && realtime && decisive && !block) {
          // Estimates of earlier renders still in flight describe the footage
          // before this point: drop them. This render's own estimate, which
          // lands a render or two later, restarts the profile.
          std::lock_guard<std::mutex> lock(shared_->mutex);
          generation = ++shared_->generation;
        }
        if (bounded) {
          track->realtimeWaited = true;
          ++profileWaits_;
          boundedResolve(env, *track, m, generation);
          ok = !env.failed();
        } else if (ok && m.data) {
          if (block) {
            if (cold) ++track->coldWaits;
            if (cold || decisive || restate) ++profileWaits_;
            double ms = flushAndWait(env, true);
            ok = ms >= 0 && !env.failed();
            if (ms >= 0) {
              gpuMs += ms;
              stageMs[OD_STAGE_PROFILE] += ms;
              if (decisive) {
                // Estimates of earlier renders still in flight describe the
                // footage before this point: drop them.
                std::lock_guard<std::mutex> lock(shared_->mutex);
                generation = ++shared_->generation;
              }
              resolveInto(*shared_, *track->stage, *track->slot, m, generation);
            }
            // Version 2's first scan includes the whole supplied +/-2
            // profiling window, even if the current estimate is usable.
            // Version 1 keeps its two-neighbour low-confidence fallback.
            // Neighbours are already ingested; their spatial estimates and
            // keys enter the normal aggregate. At most five measurements,
            // once per new size/session; later renders retain adaptation.
            for (int k = 1; ok && cold && k < env.frameCount && k <= (scanWindow ? 2*OD_PROFILE_RADIUS : 2); ++k) {
              {
                std::lock_guard<std::mutex> lock(shared_->mutex);
                const ProfileSlot::Mode& sl = track->slot->mode[gm];
                if (!scanWindow && sl.updates > 0 && sl.profile.valid && sl.profile.confidence >= kUsableConfidence) break;
              }
              const int64_t nk = work[k].frameKey;
              if ((!scanWindow && track->coldWaits >= kColdWaits) ||
                  (nk != OD_NO_KEY &&
                   std::find(track->measuredKeys.begin(), track->measuredKeys.end(), nk) != track->measuredKeys.end()))
                continue;
              ProfileMeasurement mk;
              mk.width = env.width;
              mk.height = env.height;
              mk.frameKey = nk;
              mk.force = true;
              ok = track->stage->encodeMeasure(env, work[k], mk) && !env.failed();
              if (!ok || !mk.data) break;
              if (nk != OD_NO_KEY) {
                track->measuredKeys.push_back(nk);
                if (track->measuredKeys.size() > 64) track->measuredKeys.pop_front();
              }
              ++track->coldWaits;
              ++profileWaits_;
              const double ms2 = flushAndWait(env, true);
              ok = ms2 >= 0 && !env.failed();
              if (!ok) break;
              gpuMs += ms2;
              stageMs[OD_STAGE_PROFILE] += ms2;
              resolveInto(*shared_, *track->stage, *track->slot, mk, generation);
            }
          } else {
            pending = m;  // resolved once this render's command buffer is done
            pendingStage = track->stage;
            pendingSlot = track->slot;
          }
        }
        if (ok) {
          std::lock_guard<std::mutex> lock(shared_->mutex);
          // A size with no estimate yet (a frame too small to measure, or one
          // that gave no usable estimate) borrows the nearest size's.
          const ProfileSlot::Mode& sl = track->slot->mode[gm];
          env.profile = sl.updates > 0 && sl.profile.valid ? sl.profile : fallbackProfile(track, gm);
          analysisInfo = track->stage->analysisInfo(gm);
        }
      }

      // The current frame's check result goes back with this render (noteCuts).
      if (ok && refreshCheck_ && track && !useManual && checkStamp_) {
        id<MTLBuffer> sb = nil;
        {
          std::lock_guard<std::mutex> lock(shared_->mutex);
          if (!shared_->gateBuffers.empty()) {
            sb = shared_->gateBuffers.back();
            shared_->gateBuffers.pop_back();
          }
        }
        if (!sb) sb = [ctx_->device() newBufferWithLength:4 * (OD_MAX_FRAMES - 1) options:MTLResourceStorageModeShared];
        if (sb) {
          [env.blit() copyFromBuffer:refreshCheck_ sourceOffset:0 toBuffer:sb destinationOffset:0 size:4];
          gate.stampBuffer = sb;
          gate.stamp = checkStamp_;
          gate.stampKey = work[0].frameKey;
        }
      }

      // ---- temporal / spatial / sharpen ----
      if (p.temporalNoiseModel) env.frameCount = mergeCount;
      id<MTLTexture> image = work[0].tex;
      id<MTLTexture> count = nil;
      OutputMode mode = p.view == OD_VIEW_NOISE ? OutputMode::NoiseView : OutputMode::Delta;
      bool viewDone = false;
      if (ok && viewProfile) {
        id<MTLTexture> v = track->stage->encodeView(env, work[0]);
        ok = !env.failed();
        if (ok && v) {
          image = v;
          mode = OutputMode::Absolute;
          viewDone = true;
        }
        ok = ok && boundary(OD_STAGE_PROFILE);
      }
      if (ok && !viewDone && doTemporal) {
        TemporalResult r;
        ok = temporal_->encode(env, &r) && !env.failed();
        if (ok && !r.image) {
          env.fail("temporal stage returned no image");
          ok = false;
        }
        // New-mode profiling must not restart from a cut discovered only by
        // the user's selected merge radius. Its fixed-window measurements,
        // upstream-change checks and profile change detector remain active.
        if (ok && r.gate && track && !useManual && !p.temporalNoiseModel) {
          // The cut values go back to the CPU with this render (noteCuts).
          const int n = std::min((int)r.gate.height, std::min(env.frameCount - 1, OD_MAX_FRAMES - 1));
          id<MTLBuffer> buf = nil;
          {
            std::lock_guard<std::mutex> lock(shared_->mutex);
            if (!shared_->gateBuffers.empty()) {
              buf = shared_->gateBuffers.back();
              shared_->gateBuffers.pop_back();
            }
          }
          if (!buf) buf = [ctx_->device() newBufferWithLength:4 * (OD_MAX_FRAMES - 1) options:MTLResourceStorageModeShared];
          if (buf && n > 0) {
            [env.blit() copyFromTexture:r.gate
                            sourceSlice:0
                            sourceLevel:0
                           sourceOrigin:MTLOriginMake(0, 0, 0)
                             sourceSize:MTLSizeMake(1, (NSUInteger)n, 1)
                               toBuffer:buf
                      destinationOffset:0
                 destinationBytesPerRow:4
               destinationBytesPerImage:4 * (NSUInteger)n];
            gate.buffer = buf;
            gate.count = n;
            gate.decisive = decisive;
            for (int a = 0; a < n; ++a) {
              gate.offsets[a] = work[a + 1].offset;
              gate.keys[a] = work[a + 1].frameKey;
            }
          }
        }
        if (ok) {
          image = r.image;
          count = r.count;
          framesUsed = std::max(1, r.framesUsed);
          if (p.view == OD_VIEW_MOTION) {
            mode = OutputMode::Absolute;
            viewDone = true;
          }
        }
        ok = ok && boundary(OD_STAGE_TEMPORAL);
      }
      if (ok && !viewDone && doSpatial) {
        image = spatial_->encode(env, image, count);
        ok = image != nil && !env.failed();
        if (!ok) env.fail("spatial stage returned no image");
        ok = ok && boundary(OD_STAGE_SPATIAL);
      }
      if (ok && !viewDone && env.bandsRun) {
        if (image == work[0].tex) {
          env.bandsRun = false;  // nothing was denoised (no profile yet): nothing to weigh
        } else {
          image = bands_->encode(env, image);
          ok = image != nil && !env.failed();
          if (!ok) env.fail("bands step returned no image");
          ok = ok && boundary(OD_STAGE_SPATIAL);  // not an ODStage: timed with spatial
        }
      }
      if (ok && !viewDone && doSharpen) {
        image = sharpen_->encode(env, image, count);
        ok = image != nil && !env.failed();
        if (!ok) env.fail("sharpen stage returned no image");
        ok = ok && boundary(OD_STAGE_SHARPEN);
      }

      // Observe before any view overlay. No analysis value is read by a filter.
      if (ok && !viewDone && !(p.debugFlags & OD_DEBUG_NO_ANALYSIS)) {
        analysis = analysis_->encode(env, frames[0], image, analysisInfo, renderId_, held, useManual);
        ok = !env.failed();
      }
      // ---- output ----
      ok = ok && output_->encode(env, frames[0], *out, work[0].tex, image, mode) && !env.failed();
      if (ok && p.view == OD_VIEW_ANALYSIS) {
        // Finish this frame's measurements and output before CPU drawing.
        // Never borrow a previous frame's snapshot, including when this one
        // cannot be measured. All other views keep their asynchronous path.
        const double ms = flushAndWait(env, true);
        ok = ms >= 0 && !env.failed();
        if (ok) {
          gpuMs += ms;
          stageMs[OD_STAGE_OUTPUT] += ms;
          const ODAnalysis current = AnalysisStage::resolve(analysis);
          ok = analysis_->overlay(env, *out, &current);
        }
      }
      ok = ok && boundary(OD_STAGE_OUTPUT);
    }

    if (!ok || env.failed()) {
      env.endEncoding();
      finishRender(true);
      if (lastError_.empty()) lastError_ = "render failed";
      return 1;
    }

    env.endEncoding();
    id<MTLCommandBuffer> cb = env.cb_;
    if (timing) {
      // Everything was committed stage by stage; the last buffer is empty.
      env.cb_ = nil;
      noteCuts(*shared_, gate, generation);
      std::lock_guard<std::mutex> lock(shared_->mutex);
      if (generation == shared_->generation) shared_->analysis = AnalysisStage::resolve(analysis);
      noteCompletionLocked(*shared_, ODCompletion{renderId_, gpuMs, shared_->lastQueueMs, shared_->lastLatencyMs, 0});
    } else if (wait) {
      double ms = flushAndWait(env, false);
      if (ms < 0) {
        finishRender(true);
        return 1;
      }
      gpuMs += ms;
      noteCuts(*shared_, gate, generation);
      if (pending.data) resolveInto(*shared_, *pendingStage, *pendingSlot, pending, generation);
      std::lock_guard<std::mutex> lock(shared_->mutex);
      if (generation == shared_->generation) shared_->analysis = AnalysisStage::resolve(analysis);
      noteCompletionLocked(*shared_, ODCompletion{renderId_, gpuMs, shared_->lastQueueMs, shared_->lastLatencyMs, 0});
    } else {
      std::shared_ptr<Shared> shared = shared_;
      std::shared_ptr<ProfileStage> stage = pendingStage;
      std::shared_ptr<ProfileSlot> slot = pendingSlot;
      ProfileMeasurement m = pending;
      GateReadback g = gate;
      const uint64_t renderId = renderId_;
      const bool recordQuality = quality != nullptr;
      std::shared_ptr<std::atomic<bool>> renderFailed = renderGpuFailed_;
      asyncBegin(*shared);
      const double committedAt = hostSeconds();
      [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
        if (recordQuality) {
          const double end = done.GPUEndTime > 0 ? done.GPUEndTime * 1000.0 : hostSeconds() * 1000.0;
          completeQuality(*shared, {selected.renderId, selected.runId, selected.quality,
                          done.status == MTLCommandBufferStatusError || renderFailed->load(),
                          std::max(0.0, end - callStart), gpuMsOf(done), selected.budgetMs});
        }
        if (done.status == MTLCommandBufferStatusError) {
          std::lock_guard<std::mutex> lock(shared->mutex);
          shared->poisoned = true;
          // od_render returned 0 for this render: the next call reports it
          // (once per render, whichever of its buffers failed first).
          if (!renderFailed->exchange(true)) {
            if (shared->firstAsyncError.empty()) shared->firstAsyncError = cbError(done);
            ++shared->asyncFailures;
          }
          double q = 0, l = 0;
          queueTimes(done, committedAt, &q, &l);
          noteCompletionLocked(*shared, ODCompletion{renderId, gpuMsOf(done), q, l, 1});
          asyncEnd(*shared);
          return;
        }
        {
          std::lock_guard<std::mutex> lock(shared->mutex);
          if (!renderFailed->load() && generation == shared->generation &&
              analysis.snapshot.renderId >= shared->analysis.renderId)
            shared->analysis = AnalysisStage::resolve(analysis);
          shared->lastGpuMs = gpuMsOf(done);
          queueTimes(done, committedAt, &shared->lastQueueMs, &shared->lastLatencyMs);
          noteCompletionLocked(*shared, ODCompletion{renderId, shared->lastGpuMs, shared->lastQueueMs,
                                                     shared->lastLatencyMs, renderFailed->load() ? 1 : 0});
          if (hostTraceEnabled())
            fprintf(stderr, "COMPLETE,%llu,%.6f,%.6f,%.6f\n", (unsigned long long)renderId, shared->lastGpuMs,
                    shared->lastQueueMs, shared->lastLatencyMs);
        }
        noteCuts(*shared, g, generation);
        if (!m.data) {
          asyncEnd(*shared);
          return;
        }
        // resolve() costs a few ms of CPU: not on Metal's completion thread
        // (the queue may be the host's). Serial, so estimates arrive in
        // render order.
        dispatch_async(shared->profileQueue, ^{
          resolveInto(*shared, *stage, *slot, m, generation);
          asyncEnd(*shared);
        });
      }];
      [cb commit];
      env.cb_ = nil;
    }
    if (quality && wait) {
      const double end = !timing && cb.GPUEndTime > 0 ? cb.GPUEndTime * 1000.0 : hostSeconds() * 1000.0;
      completeQuality(*shared_, {selected.renderId, selected.runId, selected.quality, 0,
                      std::max(0.0, end - callStart), gpuMs, selected.budgetMs});
    }
    double queueMs = 0, latencyMs = 0;
    {
      std::lock_guard<std::mutex> lock(shared_->mutex);
      if (wait) {
        shared_->lastGpuMs = gpuMs;
      } else {
        gpuMs += shared_->lastGpuMs;  // most recent completed render (+ any blocking part of this one)
      }
      queueMs = shared_->lastQueueMs;
      latencyMs = shared_->lastLatencyMs;
    }

    // OD_HOST_TRACE (diagnostics): one line per render, the same figures as
    // ODStats (lockWaitMs, commandBufferMs, waitMs, cpuMs) plus the encoders.
    if (hostTraceEnabled())
      fprintf(stderr, "HOST,%llu,lock=%.6f,acquire=%.6f,wait=%.6f,cpu=%.6f,commandBuffers=%u,computeEncoders=%u,blitEncoders=%u\n",
              (unsigned long long)renderId_, lockWaitMs, commandBufferMs_, waitMs_, nowMs() - t0, commandBuffers_,
              env.computeCount_, env.blitCount_);
    hits += cacheHits_;
    {
      // What this render held: its scratch (all back to the pool below) and
      // the cache entries it used. The next render of this shape is
      // admitted on it.
      uint64_t held = 0;
      for (id<MTLTexture> t : temps_) held += textureBytes(t);
      for (id<MTLTexture> t : recycled_) held += textureBytes(t);
      for (const CacheEntry& e : entries_)
        if (e.live && e.lastUse == renderIndex_) held += entryBytes(e);
      std::lock_guard<std::mutex> lock(heldMutex_);
      held_ = held;
      heldShape_ = shape;
    }
    finishRender(false);
    if (p.temporalNoiseModel) rememberKeys(analysisFrames, analysisCount);
    else rememberKeys(frames, frameCount);

    if (stats) {
      stats->gpuMs = gpuMs;
      stats->framesUsed = framesUsed;
      stats->cacheHits = hits;
      stats->profile = env.profile;
      for (int i = 0; i < OD_STAGE_COUNT; ++i) stats->stageMs[i] = stageMs[i];
      stats->waitMs = waitMs_;
      stats->gpuQueueMs = queueMs;
      stats->gpuLatencyMs = latencyMs;
      stats->verifiedFrames = 0;  // a verifying render reuses no frame (ODParams.verifyFrameCache)
      stats->cpuMs = nowMs() - t0;
    }
    return 0;
  }
}

int Session::lastAnalysis(ODAnalysis* out) {
  if (!out) return -1;
  std::unique_lock<std::mutex> lock(shared_->mutex, std::try_to_lock);
  if (!lock.owns_lock()) { *out = {}; return 0; }
  *out = shared_->analysis;
  return out->valid ? 1 : 0;
}

int Session::profileFrame(const ODFrame* frame, ODNoiseProfile* outProfile, bool chromaRecovery) {
  std::lock_guard<std::mutex> use(useMutex_);
  lastUseMs_.store(nowMs());
  holdsMemory_.store(true);
  memory_.noteActivity();
  @autoreleasepool {
    lastError_.clear();
    if (!initError_.empty()) return setError(initError_);
    if (!frame || !outProfile) return setError("od_profile_frame: NULL argument");
    std::string verr;
    if (!validateFrame(*frame, "frame", &verr)) return setError(verr);
    *outProfile = ODNoiseProfile{};
    // Not a render: renderIndex_ (the cache's age) does not move.
    std::string terr;
    ProfileTrack* track = profileTrack(frame->width, frame->height, &terr, chromaRecovery);
    if (!track) return setError(terr.empty() ? "profile stage creation failed" : terr);
    RenderEnv env;
    ODParams p = od_params_default();
    p.chromaRecovery = chromaRecovery ? 1u : 0u;
    if (!beginEnv(env, p, frame->width, frame->height)) {
      finishRender(true);
      return 1;
    }
    WorkFrame wf;
    ProfileMeasurement m;
    m.width = frame->width;
    m.height = frame->height;
    m.frameKey = frame->frameKey;
    m.force = true;  // an explicit request: never throttled
    bool ok = ingestFrame(env, *frame, &wf, nullptr, true, true);
    env.frames = &wf;
    env.frameCount = 1;
    ok = ok && track->stage->encodeMeasure(env, wf, m) && !env.failed();
    ok = ok && flushAndWait(env, false) >= 0;
    env.endEncoding();
    finishRender(!ok);
    if (!ok) {
      if (lastError_.empty()) lastError_ = "profile failed";
      return 1;
    }
    if (!m.data || !track->stage->resolve(m, outProfile, nullptr) || !outProfile->valid) {
      *outProfile = ODNoiseProfile{};
      return setError("od_profile_frame: no usable noise estimate for this frame");
    }
    return 0;
  }
}

int Session::readWorking(const ODFrame* frame, float* outYCCA) {
  std::lock_guard<std::mutex> use(useMutex_);
  lastUseMs_.store(nowMs());
  holdsMemory_.store(true);
  memory_.noteActivity();
  @autoreleasepool {
    lastError_.clear();
    if (!initError_.empty()) return setError(initError_);
    std::string verr;
    if (!frame || !outYCCA) return setError("NULL argument");
    if (!validateFrame(*frame, "frame", &verr)) return setError(verr);
    RenderEnv env;
    if (!beginEnv(env, od_params_default(), frame->width, frame->height)) {
      finishRender(true);
      return 1;
    }
    ODFrame f = *frame;
    f.frameKey = OD_NO_KEY;
    WorkFrame wf;
    bool ok = ingestFrame(env, f, &wf, nullptr, true, false);
    const NSUInteger rowBytes = (NSUInteger)f.width * 8;
    id<MTLBuffer> rb = ok ? [ctx_->device() newBufferWithLength:rowBytes * f.height options:MTLResourceStorageModeShared] : nil;
    ok = ok && rb;
    if (ok) {
      [env.blit() copyFromTexture:wf.tex
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(f.width, f.height, 1)
                         toBuffer:rb
                destinationOffset:0
           destinationBytesPerRow:rowBytes
         destinationBytesPerImage:rowBytes * f.height];
    }
    // flushAndWait also catches a missing command buffer.
    ok = ok && flushAndWait(env, false) >= 0;
    env.endEncoding();
    finishRender(!ok);
    if (!ok) {
      if (lastError_.empty()) lastError_ = "read-back failed";
      return 1;
    }
    const __fp16* h = (const __fp16*)rb.contents;
    const size_t n = (size_t)f.width * f.height * 4;
    for (size_t i = 0; i < n; ++i) outYCCA[i] = (float)h[i];
    return 0;
  }
}

uint64_t Session::estimateHeld(const RenderShape& shape) const {
  if (shape.width == 0) return 0;  // a call that fails its argument checks
  {
    std::lock_guard<std::mutex> lock(heldMutex_);
    if (held_ > 0 && shape == heldShape_) return held_;
  }
  // A shape this session has not rendered yet: a rough estimate in working
  // frames (8 bytes per pixel) per frame of the window, from what renders
  // held at radius 0 to 5 (calibrated against heldBytes()).
  const double frame = (double)shape.width * shape.height * 8.0;
  return (uint64_t)(frame * (kHeldBase + kHeldPerFrame * (double)std::max(1, shape.frames)));
}

void Session::completeQuality(Shared& shared, const ODQualityCompletion& c) {
  std::lock_guard<std::mutex> lock(shared.mutex);
  shared.governor.complete(c, hostSeconds() * 1000.0);
  if (shared.qualityCompletions.size() == 256) {
    shared.qualityCompletions.pop_front(); ++shared.qualityDropped;
  }
  shared.qualityCompletions.push_back(c);
}

int Session::pollQuality(ODQualityCompletion* out, int capacity, uint64_t* dropped) {
  if (!out || capacity <= 0) return 0;
  std::lock_guard<std::mutex> lock(shared_->mutex);
  int n = 0;
  while (n < capacity && !shared_->qualityCompletions.empty()) {
    out[n++] = shared_->qualityCompletions.front(); shared_->qualityCompletions.pop_front();
  }
  if (dropped) { *dropped = shared_->qualityDropped; shared_->qualityDropped = 0; }
  return n;
}

bool Session::drain(double timeoutMs) {
  const double t0 = nowMs();
  while (shared_->inFlight.load() > 0) {
    if (nowMs() - t0 > timeoutMs) return false;
    usleep(500);
  }
  return true;
}

int Session::takeCompletions(ODCompletion* out, int max) {
  std::lock_guard<std::mutex> lock(shared_->mutex);
  int n = 0;
  while (out && n < max && !shared_->completions.empty()) {
    out[n++] = shared_->completions.front();
    shared_->completions.pop_front();
  }
  return n;
}

void Session::cacheInfo(int* entries, uint64_t* bytes, int* poolTextures, uint64_t* poolBytes,
                        uint64_t* texturesCreated) const {
  std::lock_guard<std::mutex> use(useMutex_);  // the memory queue may be trimming
  int live = 0;
  for (const CacheEntry& e : entries_) live += e.live ? 1 : 0;
  if (entries) *entries = live;
  if (bytes) *bytes = cacheBytes_;
  memory_.poolInfo(poolTextures, poolBytes, texturesCreated);
}

}  // namespace od


// ---------------------------------------------------------------------------
// C API
// ---------------------------------------------------------------------------
extern "C" {

ODContext* od_context_create(void* device, void* queue, char* err, size_t errLen) {
  @autoreleasepool {
    auto fail = [&](const std::string& msg) -> ODContext* {
      if (err && errLen > 0) snprintf(err, errLen, "%s", msg.c_str());
      return nullptr;
    };
    if (err && errLen > 0) err[0] = 0;
    if (!device) return fail("od_context_create: device is NULL");
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
    id<MTLCommandQueue> q = queue ? (__bridge id<MTLCommandQueue>)queue : [dev newCommandQueue];
    if (!q) return fail("od_context_create: could not create a command queue");
    auto ctx = std::make_shared<od::Context>(dev, q);
    // Compile the always-needed kernels now so a broken build fails here.
    std::string e;
    for (const char* k : {"ingest_bgra32f", "ingest_bgra16f", "output_bgra32f", "output_bgra16f"}) {
      if (!ctx->pipeline(k, &e)) return fail(e);
    }
    ODContext* c = new ODContext();
    c->impl = ctx;
    return c;
  }
}

void od_context_destroy(ODContext* c) { delete c; }

ODSession* od_session_create(ODContext* c) {
  if (!c) return nullptr;
  @autoreleasepool {
    return new ODSession(c->impl);
  }
}

void od_session_destroy(ODSession* s) {
  @autoreleasepool {
    delete s;
  }
}

void od_session_reset(ODSession* s) {
  if (s) s->impl.reset();
}

ODParams od_params_default(void) {
  ODParams p;
  memset(&p, 0, sizeof(p));
  p.lumaStrength = 0.8f;
  p.chromaStrength = 1.0f;
  p.temporalRadius = 2;
  p.temporalStrength = 1.0f;
  p.spatialStrength = 0.5f;
  p.sharpen = 0.0f;
  p.noiseLevel = 1.0f;
  p.quality = 1;
  p.view = OD_VIEW_OUTPUT;
  p.waitUntilCompleted = 0;
  p.manualProfile = NULL;
  p.debugFlags = 0;
  p.debugSkipStages = 0;
  od::setAdvancedNeutral(&p);
  // API v10: the default of a newly applied effect. Zero, a zero-initialised ODParams or
  // `advanced` 0, keeps the picture of API v9.
  p.grainMode = OD_GRAIN_FINEST;
  return p;
}

int od_render(ODSession* s, const ODFrame* frames, const int* offsets, int frameCount,
                const ODParams* params, const ODFrame* out, ODStats* statsOrNull) {
  if (!s) return 1;
  return s->impl.render(frames, offsets, frameCount, params, out, statsOrNull);
}

int od_session_last_analysis(ODSession* s, ODAnalysis* out) {
  if (!s || !out) { if (out) *out = {}; return -1; }
  return s->impl.lastAnalysis(out);
}

int od_profile_frame(ODSession* s, const ODFrame* frame, ODNoiseProfile* outProfile) {
  if (!s) return 1;
  return s->impl.profileFrame(frame, outProfile);
}

int od_profile_frame_with_params(ODSession* s, const ODFrame* frame, const ODParams* params,
                                 ODNoiseProfile* outProfile) {
  if (!s) return 1;
  return s->impl.profileFrame(frame, outProfile,
                             params && params->advanced != 0 && params->chromaRecovery == 1);
}

int od_render_quality(ODSession* s, const ODFrame* frames, const int* offsets, int count,
                      const ODParams* params, const ODFrame* out, ODStats* stats,
                      const ODQualityRequest* quality, ODQualityDecision* decision) {
  if (!s) return 1;
  return s->impl.render(frames, offsets, count, params, out, stats, false, quality, decision);
}

int od_quality_poll(ODSession* s, ODQualityCompletion* out, int capacity, uint64_t* dropped) {
  return s ? s->impl.pollQuality(out, capacity, dropped) : 0;
}

int od_render_deterministic(ODSession* s, const ODFrame* frames, const int* offsets, int frameCount,
                            const ODParams* params, const ODFrame* out, ODStats* stats) {
  if (!s) return 1;
  return s->impl.render(frames, offsets, frameCount, params, out, stats, true);
}

const char* od_last_error(ODSession* s) { return s ? s->impl.lastError() : "NULL session"; }

const char* od_first_async_error(ODSession* s) { return s ? s->impl.firstAsyncError() : ""; }

int od_wait_idle(double timeoutMs) {
  // Polled: a teardown call, not a render path. The count also covers work
  // of sessions and contexts already destroyed (their blocks hold what they
  // use).
  const auto t0 = std::chrono::steady_clock::now();
  for (;;) {
    const int n = od::processAsyncPending();
    if (n <= 0) return 0;
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!(ms < timeoutMs)) return n;
    usleep(200);
  }
}

const char* od_version(void) { return "0.1.0"; }

int od_debug_compile_all(ODContext* c, char* err, size_t errLen) {
  if (!c) return 1;
  @autoreleasepool {
    std::string e;
    if (c->impl->compileAll(&e)) return 0;
    if (err && errLen > 0) snprintf(err, errLen, "%s", e.c_str());
    return 1;
  }
}

int od_debug_read_working(ODSession* s, const ODFrame* frame, float* outYCCA) {
  if (!s) return 1;
  return s->impl.readWorking(frame, outYCCA);
}

void od_debug_cache_info(ODSession* s, int* cacheEntries, uint64_t* cacheBytes, int* poolTextures,
                           uint64_t* poolBytes, uint64_t* texturesCreated) {
  if (s) s->impl.cacheInfo(cacheEntries, cacheBytes, poolTextures, poolBytes, texturesCreated);
}

void od_debug_memory_info(ODContext* c, uint64_t* allocatedBytes, uint64_t* budgetBytes, int* poolTextures,
                            uint64_t* poolBytes) {
  if (!c) return;
  od::Memory& m = c->impl->memory();
  if (allocatedBytes) *allocatedBytes = m.allocatedBytes();
  if (budgetBytes) *budgetBytes = m.budgetBytes();
  m.poolInfo(poolTextures, poolBytes, nullptr);
}

void od_debug_memory_drain(ODContext* c) {
  if (c) c->impl->memory().drain();
}

uint64_t od_debug_profile_waits(ODSession* s) { return s ? s->impl.profileWaits() : 0; }

uint64_t od_debug_session_held(ODSession* s) { return s ? s->impl.heldBytes() : 0; }

void od_debug_device_memory(ODContext* c, ODDeviceMemory* out) {
  if (!c || !out) return;
  od::DeviceMemoryStats st;
  c->impl->memory().ledger().stats(&st);
  out->budgetBytes = st.budget;
  out->logicalBytes = st.logical;
  out->logicalPeakBytes = st.logicalPeak;
  out->physicalBytes = st.physical;
  out->physicalPeakBytes = st.physicalPeak;
  out->contexts = st.contexts;
  out->rendersActive = st.rendersActive;
  out->rendersActiveMax = st.rendersActiveMax;
  out->admissionWaits = st.admissionWaits;
  out->admissionsForced = st.admissionsForced;
  out->admissionWaitMs = st.admissionWaitMs;
  out->overruns = st.overruns;
  out->overrunPeakBytes = st.overrunPeak;
}

void od_debug_device_memory_reset_peaks(ODContext* c) {
  if (c) c->impl->memory().ledger().resetPeaks();
}

int od_debug_session_drain(ODSession* s) {
  if (!s) return 1;
  return s->impl.drain(10000) ? 0 : 1;
}

int od_debug_take_completions(ODSession* s, ODCompletion* out, int max) {
  return s ? s->impl.takeCompletions(out, max) : 0;
}

}  // extern "C"
