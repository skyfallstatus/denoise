// SPDX-License-Identifier: MIT
// od::Memory — the GPU memory policy of one context, and
// od::DeviceLedger, what every context on one device shares.
//
// * One pool of idle textures per context (a context is one command queue),
//   so scratch memory is one working set per render in progress, not one per
//   effect instance. A texture enters the pool only once every command
//   buffer that used it has been committed: command buffers of one queue run
//   in commit order, so the next user's work runs after the previous user's.
//   Never reused by another context: its queue gives no such order.
// * One byte budget per DEVICE over everything the core's contexts on it
//   allocate (scratch, pools, every session's frame cache and samples): the
//   contexts made with the same setting share one DeviceLedger. Default: 1/8
//   of the device's recommended working set, at most 8 GiB, at least 1 GiB;
//   OD_MEMORY_MB overrides it (0 = no limit), read when a context is made (a
//   host process has one setting, so all its contexts share). Checked when
//   a texture or buffer has to be created: idle pool resources no render has
//   taken for 250 ms go first (surplus: this context's, then the other
//   contexts'), then cache entries of sessions not rendering, least recently
//   used first across every context of the device, then the rest of this
//   context's pool (the working set of the renders in progress, which would
//   only be created again), then the other contexts' pools, then the
//   caller's own entries that the render in progress does not use. Another
//   context's resources are freed, never reused here. What an eviction
//   frees beyond the bytes asked for goes to the idle pool when a texture of
//   the wanted shape came with it, so the renders that follow (other entries
//   of the same sizes) reuse it instead of creating textures again.
// * Renders in progress are admitted against the same budget (admit): the
//   working set of a render cannot be evicted, so a render waits at its
//   start until the estimated working sets of the renders in progress on the
//   device, its own included, fit the budget, or nothing else renders. The
//   estimate is what the session's last render of the same shape held. A
//   realtime render waits at most kAdmitRealtimeMs, any other kAdmitMs;
//   then it runs anyway (counted as forced). Overrun policy: the logical
//   total stays within the budget except when one render alone needs more
//   than it (it then runs alone) or a render was admitted at its deadline; a
//   render is never refused for the budget, the overrun is counted, its peak
//   kept (od_debug_device_memory) and logged once. Physical memory (what
//   Metal reports for the device) is reported apart: it includes the host's
//   own buffers and resources the core let go of that command buffers in
//   flight still reference.
// * An idle trim on a timer of its own (a utility-QoS serial queue), never on
//   a render thread: a session that has not been used for OD_IDLE_TRIM_MS
//   (default 3000; 0 turns the trim off) loses its frame cache; the session
//   used last keeps its cache five times longer, so a short pause does not
//   cost a cold restart. Idle pool textures go after OD_IDLE_TRIM_MS while
//   other renders carry on without them, and after five times that when
//   nothing renders. The timer only runs while something holds memory.
// * Textures are freed on that queue too: a trim, an eviction, a reset or a
//   destroy only moves pointers on the calling thread.
#ifndef OD_MEMORY_H
#define OD_MEMORY_H

#import <Metal/Metal.h>

#include <dispatch/dispatch.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace od {

class Session;

// Textures and the frame cache's check buffers (ODParams.verifyFrameCache):
// both are pooled, counted and trimmed the same way.
using ResourceList = std::vector<id<MTLResource>>;

class Memory;

// What od_debug_device_memory reports (debug.h, ODDeviceMemory).
struct DeviceMemoryStats {
  uint64_t budget = 0, logical = 0, logicalPeak = 0, physical = 0, physicalPeak = 0;
  int contexts = 0, rendersActive = 0, rendersActiveMax = 0;
  uint64_t admissionWaits = 0, admissionsForced = 0, overruns = 0, overrunPeak = 0;
  double admissionWaitMs = 0;
};

// The memory of one device shared by the contexts made on it with the same
// budget setting: the logical byte count, the budget, the admission of
// renders and the peaks. Thread-safe; members hold it, it holds them weakly.
class DeviceLedger {
 public:
  static std::shared_ptr<DeviceLedger> get(id<MTLDevice> device, uint64_t budget);
  uint64_t budget() const { return budget_; }
  uint64_t allocated() const { return allocated_.load(); }
  // Bytes about to be created are reserved first (the returned total, held
  // plus reserved, includes them), so that creations on several threads see
  // each other; commit drops the reservation and counts the created size (0
  // if nothing was created), updating the peaks. sub: bytes let go of.
  uint64_t reserve(uint64_t bytes);
  void commit(uint64_t reserved, uint64_t created);
  void sub(uint64_t bytes);
  void join(const std::shared_ptr<Memory>& m);
  void leave(const Memory* m);
  std::vector<std::shared_ptr<Memory>> members();
  // A render starts, holding about `estimate` bytes until it ends (end()).
  // Waits up to deadlineMs while the renders in progress leave no room;
  // false if it was admitted at the deadline.
  bool admit(uint64_t estimate, double deadlineMs);
  void end(uint64_t estimate);
  void samplePhysical();
  void stats(DeviceMemoryStats* out);
  void resetPeaks();

 private:
  DeviceLedger(id<MTLDevice> device, uint64_t budget) : device_(device), budget_(budget) {}
  static void raise(std::atomic<uint64_t>& peak, uint64_t v);
  id<MTLDevice> device_;
  const uint64_t budget_;
  std::atomic<uint64_t> allocated_{0}, pending_{0}, peak_{0}, physicalPeak_{0}, overruns_{0}, overrunPeak_{0};
  std::atomic<bool> warned_{false};
  // Leaf: never held while another lock is taken, nor while a member could
  // be destroyed (a shared_ptr made under it is never dropped under it).
  std::mutex membersMutex_;
  std::vector<std::pair<const Memory*, std::weak_ptr<Memory>>> members_;
  std::mutex admitMutex_;  // leaf
  std::condition_variable admitCv_;
  int active_ = 0, activeMax_ = 0;
  uint64_t reserved_ = 0, waits_ = 0, forced_ = 0;
  double waitMs_ = 0;
};

class Memory : public std::enable_shared_from_this<Memory> {
 public:
  static std::shared_ptr<Memory> create(id<MTLDevice> device);
  ~Memory();
  // Context teardown: stops the timer and waits for pending releases.
  void shutdown();

  // Sessions register for the trim and for eviction under the budget.
  // removeSession must be the first thing ~Session does: a session is only
  // ever trimmed or evicted under the same mutex.
  void addSession(Session* s);
  void removeSession(Session* s);

  // A private texture (shaderRead | shaderWrite) of exactly this shape: the
  // most recently released idle one, else a new one. `requester` is the
  // session whose call is in progress (it may lose cache entries the call does
  // not use). nil if the device refuses even after freeing what it can.
  id<MTLTexture> acquire(MTLPixelFormat format, uint32_t w, uint32_t h, uint32_t mips, Session* requester);
  // A private buffer of exactly this length, under the same pool and budget.
  id<MTLBuffer> acquireBuffer(NSUInteger length, Session* requester);
  // Back to the idle pool. Every command buffer that used them is committed.
  void recycle(ResourceList& resources);
  // Freed, on the memory queue, never on the calling thread. Empties the list.
  void release(ResourceList& resources);
  // A stamp no other render of this context has had (never 0): the frame
  // cache's check buffers move between sessions through the pool.
  uint32_t nextStamp();
  // A session holds memory now: make sure the idle trim will look at it.
  void noteActivity();

  // Renders in progress (DeviceLedger::admit / end). realtime: the host is
  // playing (a short wait). Each admit is matched by one endRender with the
  // same estimate.
  void admitRender(uint64_t estimate, bool realtime);
  void endRender(uint64_t estimate);

  uint64_t budgetBytes() const { return budget_; }
  uint64_t allocatedBytes() const { return allocated_.load(); }  // this context's
  DeviceLedger& ledger() const { return *ledger_; }

  // ---- called by another context of the same device (makeRoom) ----
  // Frees up to `need` bytes of idle pool resources (with surplusOnly, only
  // those no render has taken for a while). Returns the bytes let go of.
  uint64_t shedIdle(uint64_t need, bool surplusOnly);
  // Sessions of this context and when each was last used.
  void listSessions(const Session* except, std::vector<std::pair<Session*, double>>* out);
  // If `s` is still one of this context's sessions and not in a call, frees
  // up to `want` bytes of its cache (resources go to *out, still counted).
  uint64_t shedSession(Session* s, uint64_t want, ResourceList* out);
  double idleTrimMs() const { return idleTrimMs_; }
  void poolInfo(int* textures, uint64_t* bytes, uint64_t* created);
  // Tests: waits until every release requested so far has happened.
  void drain();

 private:
  explicit Memory(id<MTLDevice> device);
  struct Idle {
    id<MTLResource> res;
    uint64_t bytes;
    double lastUseMs;
    MTLPixelFormat format;  // MTLPixelFormatInvalid for a buffer
    uint64_t w;             // a buffer's length
    uint32_t h, mips;
  };
  void start();
  void tick();
  void arm();
  void disarmIfIdle();
  bool holdsAnything();
  // Frees up to `need` bytes for an allocation by `requester`. A resource of
  // the wanted shape found among evicted cache resources is returned instead
  // of being freed, and then the rest of what was evicted goes to the idle
  // pool (format MTLPixelFormatInvalid and w = length: a buffer).
  id<MTLResource> makeRoom(uint64_t need, MTLPixelFormat format, uint64_t w, uint32_t h, uint32_t mips,
                           Session* requester);
  // What acquire and acquireBuffer share: pool, budget, creation.
  id<MTLResource> obtain(MTLPixelFormat format, uint64_t w, uint32_t h, uint32_t mips, Session* requester);
  void releaseNow(ResourceList& resources);  // on the memory queue only
  // Lets go of `n` bytes in this context's and the ledger's counts.
  void uncount(uint64_t n);
  bool onQueue() const;

  id<MTLDevice> device_;
  uint64_t budget_ = 0;  // 0 = no limit; the device's (ledger_)
  std::shared_ptr<DeviceLedger> ledger_;
  double idleTrimMs_ = 3000;
  double lastIdleMs_ = 15000;  // the session used last, and the pool when nothing renders
  double periodMs_ = 750;

  std::mutex poolMutex_;  // leaf lock
  std::vector<Idle> idle_;
  uint64_t created_ = 0;  // textures (odbench reports them)
  std::atomic<uint32_t> stamp_{0};
  std::atomic<uint64_t> allocated_{0};  // this context's share of the ledger's count
  std::atomic<double> lastActivityMs_{0};

  // Lock order: a session's useMutex_ (its own call) -> one context's
  // sessionsMutex_ at a time (this one's or, making room, another's of the
  // same device) -> another session's useMutex_ with try_lock only ->
  // poolMutex_. The ledger's mutexes are leaves.
  std::mutex sessionsMutex_;
  std::vector<Session*> sessions_;

  dispatch_queue_t queue_ = nil;
  dispatch_source_t timer_ = nil;
  std::atomic<bool> armed_{false};
  bool down_ = false;  // memory queue only
};

}  // namespace od

#endif
