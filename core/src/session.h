// SPDX-License-Identifier: MIT
// od::Session — one per effect instance. Owns the per-frame cache, the
// stages, the automatic profile (one per frame size) and the last error.
// Scratch textures come from the context's pool (memory.h). Calls on one
// session are serialised by the caller; the state touched from other threads
// lives in Session::Shared behind its mutex (Metal completion handlers) or is
// reached by the memory policy under useMutex_.
#ifndef OD_SESSION_H
#define OD_SESSION_H

#import <Metal/Metal.h>

#include <dispatch/dispatch.h>

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "opendenoise/opendenoise.h"

#include "context.h"
#include "quality_governor.h"
#include "stages/ingest.h"
#include "stages/output.h"
#include "stages/stage.h"
#include "analysis_internal.h"

namespace od {

struct CacheEntry {
  bool live = false;
  int64_t key = OD_NO_KEY;
  uint32_t width = 0, height = 0;
  int format = 0;
  id<MTLTexture> ingest = nil;
  struct Slot {
    uint32_t slotId;
    id<MTLTexture> tex;
  };
  std::vector<Slot> slots;
  uint64_t lastUse = 0;  // render index of the last od_render that used it
  // ODParams.verifyFrameCache: such renders keep no pixels (ingest is nil,
  // no slots), only samples of the host buffer (ingest_check.metal) taken
  // when the key was last the current frame, or first seen, for the
  // automatic profile's upstream-change check; `checked` = they are there.
  id<MTLBuffer> check = nil;
  bool checked = false;
};

// Automatic profile of one frame size. A size has its own noise spectrum
// (playback at 1/2 resolution, the full-resolution frame when paused), so each
// size has its own profile stage, ring of estimates and slot, and an estimate
// only ever lands in the slot of the size it was measured at.
struct ProfileSlot {
  // One smoothed auto profile per grain mode (ODParams.grainMode), both fed
  // the same estimates, each aggregated its own way (ProfileStage::smooth);
  // a render reads that of its mode. Under Shared::mutex.
  struct Mode {
    ODNoiseProfile profile = {};
    uint64_t updates = 0;
  };
  Mode mode[kGrainModes];
};
struct ProfileTrack {
  uint32_t width = 0, height = 0;  // 0 = not yet claimed by a size
  std::shared_ptr<ProfileStage> stage;
  std::shared_ptr<ProfileSlot> slot;
  std::deque<int64_t> measuredKeys;  // frameKeys of this size profiled recently
  // coarseChromaOn() of the measurements behind measuredKeys. The coarse
  // chroma estimate is made only inside a measurement, so a change of the
  // switch measures again (session.mm).
  bool measuredCoarse = false;
  bool chromaRecovery = false;  // separate measurement history per estimator mode
  uint64_t lastUse = 0;              // render index
  // Renders of this size that blocked for a first estimate. A size blocks
  // until its profile reaches kUsableConfidence, at most kColdWaits times: a
  // size whose frames give no usable estimate (black, too small, too few flat
  // areas) would otherwise stall every render. Cleared by od_session_reset.
  int coldWaits = 0;
  // A realtime render of this size already waited (bounded) for a first
  // estimate; later realtime renders do not wait again.
  bool realtimeWaited = false;
};

// What decides how much memory a render holds (its working set): the
// device's admission of renders estimates a render from the last one of the
// same shape (Memory::admitRender).
struct RenderShape {
  uint32_t width = 0, height = 0;
  int format = -1, frames = 0, radius = 0, quality = 0;
  bool operator==(const RenderShape& o) const {
    return width == o.width && height == o.height && format == o.format && frames == o.frames &&
           radius == o.radius && quality == o.quality;
  }
};

// What a render needs to read the temporal stage's cut values back once its
// command buffer has completed (Session::noteCuts).
struct GateReadback {
  id<MTLBuffer> buffer = nil;  // RG16Float rows copied from TemporalResult.gate
  int count = 0;
  int offsets[OD_MAX_FRAMES - 1] = {};
  int64_t keys[OD_MAX_FRAMES - 1] = {};
  bool decisive = false;  // this render already restarted on its own estimate
  // ODParams.verifyFrameCache: word 0 of the current frame's check buffer
  // (ingest_check.metal), copied after its check. Equal to `stamp`: the host
  // handed other pixels under a key whose last samples the session holds.
  id<MTLBuffer> stampBuffer = nil;
  uint32_t stamp = 0;
  int64_t stampKey = OD_NO_KEY;
};

class Session {
 public:
  explicit Session(std::shared_ptr<Context> ctx);
  ~Session();

  int render(const ODFrame* frames, const int* offsets, int frameCount, const ODParams* params,
             const ODFrame* out, ODStats* stats, bool deterministic = false,
             const ODQualityRequest* quality = nullptr, ODQualityDecision* decision = nullptr);
  int pollQuality(ODQualityCompletion*, int capacity, uint64_t* dropped);
  int lastAnalysis(ODAnalysis*);
  int profileFrame(const ODFrame* frame, ODNoiseProfile* out, bool chromaRecovery = false);
  void reset();
  const char* lastError() const { return lastError_.c_str(); }
  // The first asynchronous GPU failure's message, "" if none (a copy owned
  // by the session; calls on a session are serialised by the caller).
  const char* firstAsyncError();

  // ---- called by Memory (memory.h), from other threads ----
  double lastUseMs() const { return lastUseMs_.load(); }
  bool holdsMemory() const { return holdsMemory_.load(); }
  // Idle trim, on the memory queue: if this session has not been used for
  // idleMs and is not in a call, its frame cache goes to *out. The automatic
  // profile is kept.
  void trimIfIdle(double nowMs, double idleMs, ResourceList* out);
  // Budget: gives least recently used cache entries to *out until `want`
  // bytes are there or nothing is left that may go. ownCall: called from
  // within this session's call in progress (entries it uses stay); otherwise
  // from another session's render, skipped if this session is in a call.
  uint64_t shedCache(uint64_t want, ResourceList* out, bool ownCall);

  // Debug / tests (core/src/debug.h).
  int readWorking(const ODFrame* frame, float* outYCCA);
  // Waits (at most timeoutMs) until every asynchronous render has completed
  // and its estimate, if any, has been folded in. false on timeout.
  bool drain(double timeoutMs);
  // Completions no render has handed over yet (ODStats.completed), oldest
  // first, at most max, taken as a render takes them. Returns the count.
  int takeCompletions(ODCompletion* out, int max);
  void cacheInfo(int* entries, uint64_t* cacheBytes, int* poolTextures, uint64_t* poolBytes,
                 uint64_t* texturesCreated) const;
  uint64_t profileWaits() const { return profileWaits_; }
  // Bytes the last render held (scratch and the cache entries it used), and
  // what a render of `shape` is estimated to hold.
  uint64_t heldBytes() const {
    std::lock_guard<std::mutex> lock(heldMutex_);
    return held_;
  }
  uint64_t estimateHeld(const RenderShape& shape) const;

 private:
  friend class RenderEnv;

  // State shared with completion handlers.
  struct Shared {
    std::mutex mutex;
    uint64_t generation = 0;     // bumped by reset(); stale results are dropped
    ODAnalysis analysis = {};
    double lastGpuMs = 0;
    QualityGovernor governor;
    std::deque<ODQualityCompletion> qualityCompletions;
    uint64_t qualityDropped = 0;
    bool poisoned = false;       // an async command buffer failed: caches are suspect
    // The first asynchronous command buffer error of the session
    // (od_first_async_error): kept until the session goes, whatever comes
    // after. asyncFailures counts those not yet reported (ODStats).
    std::string firstAsyncError;
    int asyncFailures = 0;
    // Completed renders not yet reported (ODStats.completed), oldest first,
    // at most kMaxPendingCompletions; dropped beyond that, and counted.
    std::deque<ODCompletion> completions;
    int completionsDropped = 0;
    // Asynchronous renders whose completion (and estimate, if any) has not
    // been handled yet. For od_debug_session_drain and ODStats.inFlight;
    // every change also moves the process-wide count od_wait_idle waits on
    // (asyncBegin / asyncEnd).
    std::atomic<int> inFlight{0};
    // Serial queue for resolve() + smooth() of asynchronous renders, so that
    // work never runs on Metal's completion thread and estimates are folded
    // in render order.
    dispatch_queue_t profileQueue = nil;
    // Cuts read back from the temporal stage: keys of the first frame after a
    // cut seen ahead of the current frame, and "the frame just rendered
    // starts a new shot" when only the past side showed it. The render that
    // meets either measures decisively (stage.h, section 5).
    std::deque<int64_t> cutKeys;
    bool decisiveNext = false;
    int decisiveNextPair = 0;  // ProfileMeasurement::pairOffset for that render
    // A decisive estimate below kUsableConfidence cannot restart the profile
    // (smooth()): the next render measures decisively again, this many more
    // times after a cut or a jump.
    int decisiveRetries = 0;
    std::vector<id<MTLBuffer>> gateBuffers;  // idle read-back buffers
    // Keys whose pixels changed upstream (GateReadback::stamp): the next
    // render forgets that it measured or saw them, and measures decisively.
    std::deque<int64_t> changedKeys;
    double lastQueueMs = 0;    // most recent completed render: commit -> GPU start
    double lastLatencyMs = 0;  // and commit -> completed
    // ODParams.profileHold of the most recent render that had an automatic
    // profile: estimates that land for a held slot (profileHeld) are
    // dropped, so those still in flight when Hold went on do not move it.
    bool hold = false;
    // The grain mode of that render (ODParams.grainMode): whose held slot
    // drops an estimate altogether (resolveInto).
    int grainMode = OD_GRAIN_WHOLE;
  };

  static void completeQuality(Shared&, const ODQualityCompletion&);
  int setError(const std::string& msg);
  // Asynchronous work of the session starts / ends (inFlight and the
  // process-wide count). Any thread.
  static void asyncBegin(Shared& shared);
  static void asyncEnd(Shared& shared);
  // A completed render, reported by the next call (Shared::mutex held).
  static void noteCompletionLocked(Shared& shared, const ODCompletion& c);
  // A command buffer from the context's queue; the time spent obtaining it
  // goes to commandBufferMs_ (Metal blocks while the queue has none free).
  id<MTLCommandBuffer> newCommandBuffer();
  bool validateFrame(const ODFrame& f, const char* what, std::string* err) const;
  // Starts the render's command buffer; fails env when the queue gives none.
  bool beginEnv(RenderEnv& env, const ODParams& p, uint32_t w, uint32_t h);
  // Ends encoders, commits and waits. Returns GPU ms, or < 0 on error (error set).
  double flushAndWait(RenderEnv& env, bool startNew);
  void finishRender(bool failed);
  // refresh: ingest again even when the frame is cached (the current frame).
  // cache: false = ODParams.noFrameCache (the key is kept for the profile).
  // verify: ODParams.verifyFrameCache (the key does not promise the pixels:
  // nothing is reused, the entry keeps samples for the profile).
  bool ingestFrame(RenderEnv& env, const ODFrame& f, WorkFrame* wf, int* cacheHits, bool refresh, bool cache,
                   bool verify = false);
  // Samples entry e's frame into its check buffer; if they differ from the
  // samples already there, the render's stamp goes into the buffer's word 0
  // (ingest_check.metal). false if no check could be encoded.
  bool encodeCheck(RenderEnv& env, const ODFrame& f, CacheEntry& e);
  // An entry's textures (a render that trusted the key filled them) go back
  // with this render's; its samples stay.
  void dropPixels(CacheEntry& e);
  int findEntry(int64_t key, uint32_t w, uint32_t h, int format) const;
  int newEntry();
  void evictEntry(int index, ResourceList* out);
  void enforceCacheBudget();
  void dropCaches(ResourceList* out);
  uint64_t entryBytes(const CacheEntry& e) const;
  // Texture for the render in progress: one handed back earlier in this
  // render (recycle, replaced slots) if one has the shape, else the
  // context's pool. nil if the device refuses.
  id<MTLTexture> acquireTexture(MTLPixelFormat format, uint32_t w, uint32_t h, uint32_t mips);
  ProfileTrack* profileTrack(uint32_t w, uint32_t h, std::string* err, bool chromaRecovery = false);
  // Valid slot of the nearest other frame size, for a size that has none yet.
  ODNoiseProfile fallbackProfile(const ProfileTrack* except, int grainMode);
  // Resolves an estimate and folds it into the slot of the size it was
  // measured at, unless od_session_reset came in between. Render thread
  // (blocking renders) or the profile queue (asynchronous ones).
  static void resolveInto(Shared& shared, ProfileStage& stage, ProfileSlot& slot, const ProfileMeasurement& m,
                          uint64_t generation);
  // Realtime renders: commits the work encoded so far (ingest + measure),
  // waits for it at most kRealtimeWaitMs and resolves the estimate here if
  // it came in time; otherwise it is resolved on the profile queue when the
  // GPU gets to it. Continues the render in a fresh command buffer.
  void boundedResolve(RenderEnv& env, ProfileTrack& track, const ProfileMeasurement& m, uint64_t generation);
  // Reads a completed render's cut values (any thread) into Shared.
  static void noteCuts(Shared& shared, const GateReadback& g, uint64_t generation);
  // A render with neighbours whose window holds no frame this session
  // rendered recently. A render of the current frame alone is never a jump.
  bool isJump(const ODFrame* frames, int frameCount) const;
  void rememberKeys(const ODFrame* frames, int frameCount);

  // Held for the whole of every public call. The caller already serialises
  // calls on one session; this lock only keeps the memory policy (another
  // thread) out while a call is in progress.
  mutable std::mutex useMutex_;
  std::atomic<double> lastUseMs_{0};
  std::atomic<bool> holdsMemory_{false};

  std::shared_ptr<Context> ctx_;
  Memory& memory_;
  std::vector<CacheEntry> entries_;
  uint64_t cacheBytes_ = 0;            // running total of entries_
  std::vector<id<MTLTexture>> temps_;  // temporaries of the render in progress
  // Textures handed back during the render in progress (RenderEnv::recycle,
  // replaced or dropped slots). Only this render may reuse them: another
  // session could otherwise commit work on them before this render commits.
  std::vector<id<MTLTexture>> recycled_;
  uint64_t renderIndex_ = 0;           // od_render calls; entry ages count these
  uint64_t cacheBudgetBytes_ = 0;
  int cacheHits_ = 0;  // slot hits of the render in progress
  std::deque<int64_t> seenKeys_;  // keys of the frames recently rendered (any role)
  uint64_t profileWaits_ = 0;     // renders that blocked for their estimate (od_debug_profile_waits)
  // OD_GRAIN_PROFILE_V2 when the session was made (grainModeOverride(): -1 =
  // none, else the grain mode every render of this session uses; tests).
  int grainOverride_ = -1;
  // Samples of verifying renders (ODParams.verifyFrameCache): kernels
  // (fetched when the session is made, so their first compile never lands in
  // a render), and the render's stamp, unique in the context
  // (Memory::nextStamp): check buffers come from the context's pool and move
  // between sessions.
  id<MTLComputePipelineState> check32_ = nil, check16_ = nil;
  uint32_t checkStamp_ = 0;
  // The current frame's check buffer when it held samples of an earlier
  // ingest of the same key (the render reads its stamp back).
  id<MTLBuffer> refreshCheck_ = nil;
  unsigned commandBuffers_ = 0;  // command buffers the render in progress obtained (OD_HOST_TRACE)
  double waitMs_ = 0;  // CPU ms the render in progress spent waiting for the GPU
  double commandBufferMs_ = 0;  // and obtaining command buffers (newCommandBuffer)
  // The render in progress: its id (ODStats.renderId) and whether one of its
  // command buffers failed on the GPU after it stopped waiting (set once by
  // the completion handler that saw it first, which counts the failure).
  uint64_t renderId_ = 0;
  std::shared_ptr<std::atomic<bool>> renderGpuFailed_;
  std::string firstAsyncErrorCopy_;  // od_first_async_error's storage
  // The last render that was encoded to the end and the bytes it held.
  // Read before useMutex_ (the admission), so under a lock of their own: a
  // caller that overlaps calls on one session (ODStats.lockWaitMs) must not
  // race the render that writes them.
  mutable std::mutex heldMutex_;
  RenderShape heldShape_;
  uint64_t held_ = 0;

  std::unique_ptr<IngestStage> ingest_;
  std::unique_ptr<OutputStage> output_;
  std::vector<ProfileTrack> tracks_;
  std::unique_ptr<TemporalStage> temporal_;
  std::unique_ptr<SpatialStage> spatial_;
  std::unique_ptr<SharpenStage> sharpen_;
  std::unique_ptr<BandsStage> bands_;
  std::unique_ptr<AnalysisStage> analysis_;

  std::shared_ptr<Shared> shared_;
  std::string initError_;
  std::string lastError_;
};

}  // namespace od

struct ODSession {
  od::Session impl;
  explicit ODSession(std::shared_ptr<od::Context> c) : impl(std::move(c)) {}
};

#endif
