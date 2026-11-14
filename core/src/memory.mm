// SPDX-License-Identifier: MIT
#include "memory.h"

#include "async.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>

#include "session.h"

namespace od {

namespace {

double monotonicMs() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

uint64_t bytesOf(const ResourceList& list) {
  uint64_t n = 0;
  for (id<MTLResource> r : list) n += r ? (uint64_t)r.allocatedSize : 0;
  return n;
}

bool sameShape(id<MTLResource> r, MTLPixelFormat format, uint64_t w, uint32_t h, uint32_t mips) {
  if (format == MTLPixelFormatInvalid) {
    if (![r conformsToProtocol:@protocol(MTLBuffer)]) return false;
    return ((id<MTLBuffer>)r).length == w;
  }
  if (![r conformsToProtocol:@protocol(MTLTexture)]) return false;
  id<MTLTexture> t = (id<MTLTexture>)r;
  return t.pixelFormat == format && t.width == w && t.height == h && t.mipmapLevelCount == mips;
}

char kQueueKey;

// An idle resource no render has taken for this long is surplus (another
// size, a smaller radius) and goes first when room is needed; younger ones
// are the working set of the renders in progress.
constexpr double kSurplusMs = 250.0;

// How long a render waits at its start for room in the device's budget
// (Memory::admitRender): a realtime render (the host is playing; a stall
// drops frames) only a few milliseconds, others up to two seconds.
constexpr double kAdmitRealtimeMs = 4.0;
constexpr double kAdmitMs = 2000.0;

}  // namespace

// ---------------------------------------------------------------------------
// DeviceLedger
// ---------------------------------------------------------------------------
std::shared_ptr<DeviceLedger> DeviceLedger::get(id<MTLDevice> device, uint64_t budget) {
  static std::mutex* mutex = new std::mutex;  // never destroyed: hosts may call during static teardown
  static auto* ledgers = new std::map<std::pair<uint64_t, uint64_t>, std::weak_ptr<DeviceLedger>>();
  std::lock_guard<std::mutex> lock(*mutex);
  const auto key = std::make_pair((uint64_t)device.registryID, budget);
  if (std::shared_ptr<DeviceLedger> l = (*ledgers)[key].lock()) return l;
  for (auto it = ledgers->begin(); it != ledgers->end();) it = it->second.expired() ? ledgers->erase(it) : ++it;
  std::shared_ptr<DeviceLedger> l(new DeviceLedger(device, budget));
  (*ledgers)[key] = l;
  return l;
}

void DeviceLedger::raise(std::atomic<uint64_t>& peak, uint64_t v) {
  uint64_t cur = peak.load();
  while (v > cur && !peak.compare_exchange_weak(cur, v)) {
  }
}

uint64_t DeviceLedger::reserve(uint64_t bytes) { return pending_.fetch_add(bytes) + bytes + allocated_.load(); }

void DeviceLedger::commit(uint64_t reserved, uint64_t created) {
  pending_.fetch_sub(reserved);
  if (created == 0) return;
  const uint64_t now = allocated_.fetch_add(created) + created;
  raise(peak_, now);
  if (budget_ > 0 && now > budget_) {
    overruns_.fetch_add(1);
    raise(overrunPeak_, now - budget_);
    if (!warned_.exchange(true)) {
      fprintf(stderr,
              "Denoise: the renders in progress need more than the GPU memory budget of %llu MB; "
              "allocating beyond it (OD_MEMORY_MB sets the budget)\n",
              (unsigned long long)(budget_ >> 20));
    }
  }
  samplePhysical();
}

void DeviceLedger::sub(uint64_t bytes) {
  uint64_t cur = allocated_.load();
  while (!allocated_.compare_exchange_weak(cur, cur - std::min(cur, bytes))) {
  }
}

void DeviceLedger::samplePhysical() { raise(physicalPeak_, (uint64_t)device_.currentAllocatedSize); }

void DeviceLedger::join(const std::shared_ptr<Memory>& m) {
  std::lock_guard<std::mutex> lock(membersMutex_);
  members_.push_back({m.get(), m});
}

void DeviceLedger::leave(const Memory* m) {
  // By address, never through lock(): a shared_ptr made here could be the
  // last one of another member and destroy it under this mutex.
  std::lock_guard<std::mutex> lock(membersMutex_);
  members_.erase(std::remove_if(members_.begin(), members_.end(),
                                [&](const std::pair<const Memory*, std::weak_ptr<Memory>>& e) {
                                  return e.first == m || e.second.expired();
                                }),
                 members_.end());
}

std::vector<std::shared_ptr<Memory>> DeviceLedger::members() {
  // The caller drops them, outside the mutex.
  std::vector<std::shared_ptr<Memory>> out;
  std::lock_guard<std::mutex> lock(membersMutex_);
  for (const auto& e : members_)
    if (std::shared_ptr<Memory> p = e.second.lock()) out.push_back(p);
  return out;
}

bool DeviceLedger::admit(uint64_t estimate, double deadlineMs) {
  std::unique_lock<std::mutex> lock(admitMutex_);
  bool inTime = true;
  if (budget_ > 0 && active_ > 0 && reserved_ + estimate > budget_) {
    ++waits_;
    const double t0 = monotonicMs();
    const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds((int64_t)(deadlineMs * 1000.0));
    while (active_ > 0 && reserved_ + estimate > budget_) {
      if (admitCv_.wait_until(lock, until) == std::cv_status::timeout) {
        inTime = !(active_ > 0 && reserved_ + estimate > budget_);
        break;
      }
    }
    if (!inTime) ++forced_;
    waitMs_ += monotonicMs() - t0;
  }
  ++active_;
  reserved_ += estimate;
  activeMax_ = std::max(activeMax_, active_);
  return inTime;
}

void DeviceLedger::end(uint64_t estimate) {
  {
    std::lock_guard<std::mutex> lock(admitMutex_);
    active_ = std::max(0, active_ - 1);
    reserved_ -= std::min(reserved_, estimate);
  }
  admitCv_.notify_all();
  samplePhysical();
}

void DeviceLedger::stats(DeviceMemoryStats* out) {
  out->budget = budget_;
  out->logical = allocated_.load();
  out->logicalPeak = peak_.load();
  out->physical = (uint64_t)device_.currentAllocatedSize;
  out->physicalPeak = std::max(physicalPeak_.load(), out->physical);
  out->overruns = overruns_.load();
  out->overrunPeak = overrunPeak_.load();
  out->contexts = (int)members().size();
  std::lock_guard<std::mutex> lock(admitMutex_);
  out->rendersActive = active_;
  out->rendersActiveMax = activeMax_;
  out->admissionWaits = waits_;
  out->admissionsForced = forced_;
  out->admissionWaitMs = waitMs_;
}

void DeviceLedger::resetPeaks() {
  peak_.store(allocated_.load());
  physicalPeak_.store((uint64_t)device_.currentAllocatedSize);
  overruns_.store(0);
  overrunPeak_.store(0);
  std::lock_guard<std::mutex> lock(admitMutex_);
  activeMax_ = active_;
  waits_ = forced_ = 0;
  waitMs_ = 0;
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

std::shared_ptr<Memory> Memory::create(id<MTLDevice> device) {
  std::shared_ptr<Memory> m(new Memory(device));
  m->ledger_ = DeviceLedger::get(device, m->budget_);
  m->ledger_->join(m);
  m->start();
  return m;
}

Memory::Memory(id<MTLDevice> device) : device_(device) {
  const uint64_t rec = (uint64_t)device.recommendedMaxWorkingSetSize;
  const uint64_t GiB = 1ull << 30;
  budget_ = rec > 0 ? std::min(8 * GiB, std::max(1 * GiB, rec / 8)) : 4 * GiB;
  if (const char* e = getenv("OD_MEMORY_MB")) {
    const long v = atol(e);
    if (v >= 0) budget_ = (uint64_t)v << 20;
  }
  if (const char* e = getenv("OD_IDLE_TRIM_MS")) {
    const double v = atof(e);
    if (v >= 0) idleTrimMs_ = v;
  }
  lastIdleMs_ = 5 * idleTrimMs_;
  periodMs_ = std::min(1000.0, std::max(25.0, idleTrimMs_ / 4));
}

void Memory::start() {
  queue_ = dispatch_queue_create("dev.opendenoise.memory",
                                 dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0));
  dispatch_queue_set_specific(queue_, &kQueueKey, this, nullptr);
  if (idleTrimMs_ <= 0) return;
  timer_ = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, queue_);
  // The handler holds the policy weakly: a context that is gone (or leaked
  // by a host on purpose) never keeps it alive, and an idle one gets no
  // wake-ups because the timer disarms itself.
  std::weak_ptr<Memory> weak = shared_from_this();
  dispatch_source_set_event_handler(timer_, ^{
    if (std::shared_ptr<Memory> m = weak.lock()) m->tick();
  });
  dispatch_source_set_timer(timer_, DISPATCH_TIME_FOREVER, DISPATCH_TIME_FOREVER, 0);
  dispatch_activate(timer_);
}

Memory::~Memory() {
  if (timer_) dispatch_source_cancel(timer_);
  if (ledger_) ledger_->leave(this);
  // Whatever is still pooled goes with us (sessions are gone by now).
  uint64_t n = 0;
  for (const Idle& i : idle_) n += i.bytes;
  uncount(n);
  idle_.clear();
}

bool Memory::onQueue() const { return dispatch_get_specific(&kQueueKey) == (void*)this; }

void Memory::shutdown() {
  if (timer_) dispatch_source_cancel(timer_);
  if (!onQueue()) {
    // Waits for a tick in progress and for every release queued so far.
    dispatch_sync(queue_, ^{
      down_ = true;
    });
  } else {
    down_ = true;
  }
  if (ledger_) ledger_->leave(this);
  ResourceList all;
  {
    std::lock_guard<std::mutex> lock(poolMutex_);
    for (Idle& i : idle_) all.push_back(i.res);
    idle_.clear();
  }
  @autoreleasepool {
    uncount(bytesOf(all));
    all.clear();
  }
}

void Memory::uncount(uint64_t n) {
  uint64_t cur = allocated_.load();
  while (!allocated_.compare_exchange_weak(cur, cur - std::min(cur, n))) {
  }
  // The ledger loses what this context counted, never more.
  if (ledger_) ledger_->sub(std::min(cur, n));
}

void Memory::admitRender(uint64_t estimate, bool realtime) {
  ledger_->admit(estimate, realtime ? kAdmitRealtimeMs : kAdmitMs);
}

void Memory::endRender(uint64_t estimate) { ledger_->end(estimate); }

void Memory::addSession(Session* s) {
  std::lock_guard<std::mutex> lock(sessionsMutex_);
  sessions_.push_back(s);
}

void Memory::removeSession(Session* s) {
  std::lock_guard<std::mutex> lock(sessionsMutex_);
  auto it = std::find(sessions_.begin(), sessions_.end(), s);
  if (it != sessions_.end()) sessions_.erase(it);
}

id<MTLTexture> Memory::acquire(MTLPixelFormat format, uint32_t w, uint32_t h, uint32_t mips, Session* requester) {
  if (w == 0 || h == 0 || format == MTLPixelFormatInvalid) return nil;
  return (id<MTLTexture>)obtain(format, w, h, mips < 1 ? 1 : mips, requester);
}

id<MTLBuffer> Memory::acquireBuffer(NSUInteger length, Session* requester) {
  if (length == 0) return nil;
  return (id<MTLBuffer>)obtain(MTLPixelFormatInvalid, length, 0, 0, requester);
}

uint32_t Memory::nextStamp() {
  uint32_t s = stamp_.fetch_add(1) + 1;
  if (s == 0) s = stamp_.fetch_add(1) + 1;
  return s;
}

id<MTLResource> Memory::obtain(MTLPixelFormat format, uint64_t w, uint32_t h, uint32_t mips, Session* requester) {
  const bool buffer = format == MTLPixelFormatInvalid;
  {
    // Most recently released first: surplus resources of a shape still in
    // use then age out instead of being refreshed by every render.
    std::lock_guard<std::mutex> lock(poolMutex_);
    for (size_t i = idle_.size(); i-- > 0;) {
      const Idle& e = idle_[i];
      if (e.format == format && e.w == w && e.h == h && e.mips == mips) {
        id<MTLResource> r = e.res;
        idle_.erase(idle_.begin() + (long)i);
        return r;
      }
    }
  }
  MTLTextureDescriptor* d = nil;
  uint64_t need = 0;
  if (buffer) {
    need = [device_ heapBufferSizeAndAlignWithLength:(NSUInteger)w options:MTLResourceStorageModePrivate].size;
  } else {
    d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:(NSUInteger)w height:h mipmapped:NO];
    d.mipmapLevelCount = mips;
    d.storageMode = MTLStorageModePrivate;
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    need = (uint64_t)[device_ heapTextureSizeAndAlignWithDescriptor:d].size;
  }
  auto make = [&]() -> id<MTLResource> {
    if (buffer) return [device_ newBufferWithLength:(NSUInteger)w options:MTLResourceStorageModePrivate];
    return [device_ newTextureWithDescriptor:d];
  };
  // The device's count (every context made on it with this budget), with
  // these bytes reserved in it, so that creations on other threads see them.
  const uint64_t after = ledger_->reserve(need);
  if (budget_ > 0 && after > budget_) {
    id<MTLResource> r = makeRoom(after - budget_, format, w, h, mips, requester);
    if (r) {
      ledger_->commit(need, 0);
      return r;
    }
  }
  id<MTLResource> r = make();
  if (!r) {
    // The device refused: give back everything that is not in use, once.
    id<MTLResource> reuse = makeRoom(UINT64_MAX, format, w, h, mips, requester);
    if (reuse) {
      ledger_->commit(need, 0);
      return reuse;
    }
    r = make();
  }
  const uint64_t n = r ? (uint64_t)r.allocatedSize : 0;
  if (r) allocated_.fetch_add(n);
  ledger_->commit(need, n);  // counts an overrun if the device's total is now beyond the budget
  if (r && !buffer) {
    std::lock_guard<std::mutex> lock(poolMutex_);
    ++created_;
  }
  return r;
}

id<MTLResource> Memory::makeRoom(uint64_t need, MTLPixelFormat format, uint64_t w, uint32_t h, uint32_t mips,
                                 Session* requester) {
  ResourceList freed;
  uint64_t got = 0;
  // Idle resources, oldest first; with surplusOnly, only those no render has taken for
  // kSurplusMs. The rest of the pool is the working set of the renders in progress (every
  // render takes the same shapes again): freed to make room for a cache entry, it had to be
  // created again by the next render, which then evicted a cache entry, and so on.
  auto takeIdle = [&](bool surplusOnly) {
    const double now = monotonicMs();
    std::lock_guard<std::mutex> lock(poolMutex_);
    size_t k = 0;
    for (size_t i = 0; i < idle_.size(); ++i) {
      if (got < need && (!surplusOnly || now - idle_[i].lastUseMs > kSurplusMs)) {
        got += idle_[i].bytes;
        freed.push_back(idle_[i].res);
      } else {
        idle_[k++] = idle_[i];
      }
    }
    idle_.erase(idle_.begin() + (long)k, idle_.end());
  };
  // The other contexts of the device (their own queues): what they give up
  // is freed by them, never reused here.
  std::vector<std::shared_ptr<Memory>> others;
  for (std::shared_ptr<Memory>& m : ledger_->members())
    if (m.get() != this) others.push_back(m);
  auto othersIdle = [&](bool surplusOnly) {
    for (std::shared_ptr<Memory>& m : others) {
      if (got >= need) break;
      got += m->shedIdle(need - got, surplusOnly);
    }
  };
  takeIdle(true);
  if (got < need) othersIdle(true);
  id<MTLResource> reuse = nil;
  ResourceList evicted;
  if (got < need) {
    // Cache entries of sessions that are not rendering, least recently used
    // session first, across every context of the device.
    struct Candidate {
      Memory* owner;
      Session* session;
      double lastUseMs;
    };
    std::vector<Candidate> candidates;
    std::vector<std::pair<Session*, double>> list;
    listSessions(requester, &list);
    for (auto& e : list) candidates.push_back({this, e.first, e.second});
    for (std::shared_ptr<Memory>& m : others) {
      list.clear();
      m->listSessions(nullptr, &list);
      for (auto& e : list) candidates.push_back({m.get(), e.first, e.second});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.lastUseMs < b.lastUseMs; });
    for (const Candidate& c : candidates) {
      if (got >= need) break;
      if (c.owner == this) {
        got += shedSession(c.session, need - got, &evicted);
      } else {
        ResourceList out;
        got += c.owner->shedSession(c.session, need - got, &out);
        c.owner->release(out);
      }
    }
  }
  if (got < need) takeIdle(false);
  if (got < need) othersIdle(false);
  if (got < need && requester) got += requester->shedCache(need - got, &evicted, /*ownCall*/ true);
  if (!evicted.empty()) {
    // Their command buffers are committed, so a resource of the wanted shape
    // can serve this render directly.
    for (size_t i = 0; i < evicted.size(); ++i) {
      if (sameShape(evicted[i], format, w, h, mips)) {
        reuse = evicted[i];
        evicted.erase(evicted.begin() + (long)i);
        break;
      }
    }
    if (reuse && need != UINT64_MAX) {
      // Nothing new is allocated, so the rest of the entry (its pyramid, its
      // check buffer) stays counted, in the idle pool, where the next
      // acquires (the new entry's own pyramid) find the same shapes.
      recycle(evicted);
    } else {
      freed.insert(freed.end(), evicted.begin(), evicted.end());
    }
  }
  release(freed);
  return reuse;
}

uint64_t Memory::shedIdle(uint64_t need, bool surplusOnly) {
  ResourceList freed;
  uint64_t got = 0;
  {
    const double now = monotonicMs();
    std::lock_guard<std::mutex> lock(poolMutex_);
    size_t k = 0;
    for (size_t i = 0; i < idle_.size(); ++i) {
      if (got < need && (!surplusOnly || now - idle_[i].lastUseMs > kSurplusMs)) {
        got += idle_[i].bytes;
        freed.push_back(idle_[i].res);
      } else {
        idle_[k++] = idle_[i];
      }
    }
    idle_.erase(idle_.begin() + (long)k, idle_.end());
  }
  release(freed);
  return got;
}

void Memory::listSessions(const Session* except, std::vector<std::pair<Session*, double>>* out) {
  std::lock_guard<std::mutex> lock(sessionsMutex_);
  for (Session* s : sessions_)
    if (s != except) out->push_back({s, s->lastUseMs()});
}

uint64_t Memory::shedSession(Session* s, uint64_t want, ResourceList* out) {
  // Under the session list's lock: a session is removed from it first thing
  // in its destructor, so one found here is alive.
  std::lock_guard<std::mutex> lock(sessionsMutex_);
  if (std::find(sessions_.begin(), sessions_.end(), s) == sessions_.end()) return 0;
  return s->shedCache(want, out, /*ownCall*/ false);
}

void Memory::recycle(ResourceList& resources) {
  if (resources.empty()) return;
  const double now = monotonicMs();
  std::vector<Idle> add;
  add.reserve(resources.size());
  for (id<MTLResource> r : resources) {
    if (!r) continue;
    if ([r conformsToProtocol:@protocol(MTLTexture)]) {
      id<MTLTexture> t = (id<MTLTexture>)r;
      add.push_back({r, (uint64_t)r.allocatedSize, now, t.pixelFormat, (uint64_t)t.width, (uint32_t)t.height,
                     (uint32_t)t.mipmapLevelCount});
    } else {
      add.push_back({r, (uint64_t)r.allocatedSize, now, MTLPixelFormatInvalid, (uint64_t)((id<MTLBuffer>)r).length, 0, 0});
    }
  }
  resources.clear();
  std::lock_guard<std::mutex> lock(poolMutex_);
  idle_.insert(idle_.end(), add.begin(), add.end());
}

void Memory::release(ResourceList& textures) {
  if (textures.empty()) return;
  uncount(bytesOf(textures));
  // The last references die on the memory queue: dropping a hundred Metal
  // textures takes tens of milliseconds, which must not land on a render
  // thread, nor under a lock that od_session_create/destroy take.
  ResourceList* heap = new ResourceList();
  heap->swap(textures);
  processAsyncBegin();  // od_wait_idle waits for it (async.h)
  dispatch_async(queue_, ^{
    @autoreleasepool {
      delete heap;
    }
    processAsyncEnd();
  });
}

void Memory::releaseNow(ResourceList& textures) {
  if (textures.empty()) return;
  uncount(bytesOf(textures));
  @autoreleasepool {
    textures.clear();
  }
}

void Memory::noteActivity() {
  lastActivityMs_.store(monotonicMs());
  if (!timer_ || armed_.exchange(true)) return;
  std::weak_ptr<Memory> weak = weak_from_this();
  dispatch_async(queue_, ^{
    if (std::shared_ptr<Memory> m = weak.lock()) m->arm();
  });
}

void Memory::arm() {
  if (down_ || !timer_) return;
  const uint64_t period = (uint64_t)(periodMs_ * 1e6);
  dispatch_source_set_timer(timer_, dispatch_time(DISPATCH_TIME_NOW, (int64_t)period), period, period / 10);
}

bool Memory::holdsAnything() {
  {
    std::lock_guard<std::mutex> lock(sessionsMutex_);
    for (Session* s : sessions_)
      if (s->holdsMemory()) return true;
  }
  std::lock_guard<std::mutex> lock(poolMutex_);
  return !idle_.empty();
}

void Memory::disarmIfIdle() {
  if (holdsAnything()) return;
  armed_.store(false);
  dispatch_source_set_timer(timer_, DISPATCH_TIME_FOREVER, DISPATCH_TIME_FOREVER, 0);
  // A session may have started a call between the check and the store: it
  // either saw armed_ still true (then this re-check sees its memory) or it
  // armed the timer itself.
  if (holdsAnything() && !armed_.exchange(true)) arm();
}

void Memory::tick() {
  if (down_) return;
  @autoreleasepool {
    const double now = monotonicMs();
    ResourceList freed;
    {
      // Only pointer moves under the lock; sessions in a call are skipped
      // (try_lock in trimIfIdle).
      std::lock_guard<std::mutex> lock(sessionsMutex_);
      Session* latest = nullptr;
      double latestMs = -1;
      for (Session* s : sessions_) {
        const double t = s->lastUseMs();
        if (t > latestMs) {
          latestMs = t;
          latest = s;
        }
      }
      for (Session* s : sessions_) s->trimIfIdle(now, s == latest ? lastIdleMs_ : idleTrimMs_, &freed);
    }
    {
      // Pool: a texture the renders have not needed for idleTrimMs while they
      // carry on is surplus (another size, a smaller radius); when nothing
      // renders the pool is the next render's working set and stays as long
      // as the last session's cache.
      std::lock_guard<std::mutex> lock(poolMutex_);
      const bool active = now - lastActivityMs_.load() < idleTrimMs_;
      size_t k = 0;
      for (size_t i = 0; i < idle_.size(); ++i) {
        const double age = now - idle_[i].lastUseMs;
        if (age > lastIdleMs_ || (active && age > idleTrimMs_)) {
          freed.push_back(idle_[i].res);
        } else {
          idle_[k++] = idle_[i];
        }
      }
      idle_.erase(idle_.begin() + (long)k, idle_.end());
    }
    releaseNow(freed);
  }
  disarmIfIdle();
}

void Memory::poolInfo(int* textures, uint64_t* bytes, uint64_t* created) {
  std::lock_guard<std::mutex> lock(poolMutex_);
  if (textures) {
    int n = 0;
    for (const Idle& i : idle_) n += i.format != MTLPixelFormatInvalid ? 1 : 0;
    *textures = n;
  }
  if (bytes) {
    uint64_t n = 0;
    for (const Idle& i : idle_) n += i.bytes;
    *bytes = n;
  }
  if (created) *created = created_;
}

void Memory::drain() {
  if (onQueue()) return;
  dispatch_sync(queue_, ^{
  });
}

}  // namespace od
