// SPDX-License-Identifier: MIT
// Host error and lifetime contract: encoder failures, deferred GPU failures, completion
// accounting, bounded idle waits and teardown. Synthetic buffers only; no application or
// external media required.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include "opendenoise/opendenoise.h"

#include "debug.h"

static int gFailures = 0;
static int gChecks = 0;
static const char* gTest = "";

#define CHECK(cond, ...)                                               \
  do {                                                                 \
    ++gChecks;                                                         \
    if (!(cond)) {                                                     \
      ++gFailures;                                                     \
      fprintf(stderr, "FAIL [%s] %s:%d: ", gTest, __FILE__, __LINE__); \
      fprintf(stderr, __VA_ARGS__);                                    \
      fprintf(stderr, "\n");                                           \
    }                                                                  \
  } while (0)

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;

static double nowMs() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// ---- fault injection ---------------------------------------------------------

@class HostQueue;

// A command buffer of HostQueue: forwards everything to the real one, except
// that it can refuse an encoder (the queue counts the requests) and make its
// completion handlers see a failed buffer.
@interface HostCommandBuffer : NSProxy
- (instancetype)initWithBuffer:(id<MTLCommandBuffer>)cb queue:(HostQueue*)q fail:(int)failNumber;
@end

// A command queue that forwards to a real one. failComputeAt / failBlitAt:
// the n-th compute / blit encoder request (1-based, counted over every
// buffer) returns nil. failBufferAt: the n-th command buffer reports
// MTLCommandBufferStatusError to its completion handlers (its work runs).
@interface HostQueue : NSProxy
@property(nonatomic) int failComputeAt, failBlitAt, failBufferAt;
@property(nonatomic) int buffers, computeCalls, blitCalls, failuresInjected;
- (instancetype)initWithQueue:(id<MTLCommandQueue>)queue;
@end

@implementation HostCommandBuffer {
  id<MTLCommandBuffer> _cb;
  HostQueue* _q;
  int _fail;  // 0, or the number of the injected failure
}
- (instancetype)initWithBuffer:(id<MTLCommandBuffer>)cb queue:(HostQueue*)q fail:(int)failNumber {
  _cb = cb;
  _q = q;
  _fail = failNumber;
  return self;
}
- (id<MTLComputeCommandEncoder>)computeCommandEncoder {
  _q.computeCalls = _q.computeCalls + 1;
  if (_q.failComputeAt && _q.computeCalls == _q.failComputeAt) return nil;
  return [_cb computeCommandEncoder];
}
- (id<MTLBlitCommandEncoder>)blitCommandEncoder {
  _q.blitCalls = _q.blitCalls + 1;
  if (_q.failBlitAt && _q.blitCalls == _q.failBlitAt) return nil;
  return [_cb blitCommandEncoder];
}
- (void)addCompletedHandler:(MTLCommandBufferHandler)block {
  if (!_fail) {
    [_cb addCompletedHandler:block];
    return;
  }
  HostCommandBuffer* me = self;
  [_cb addCompletedHandler:^(id<MTLCommandBuffer>) {
    block((id<MTLCommandBuffer>)me);
  }];
}
- (MTLCommandBufferStatus)status {
  const MTLCommandBufferStatus s = _cb.status;
  return _fail && s == MTLCommandBufferStatusCompleted ? MTLCommandBufferStatusError : s;
}
- (NSError*)error {
  if (!_fail) return _cb.error;
  return [NSError errorWithDomain:@"test"
                             code:_fail
                         userInfo:@{NSLocalizedDescriptionKey : [NSString stringWithFormat:@"injected failure %d", _fail]}];
}
- (NSMethodSignature*)methodSignatureForSelector:(SEL)sel {
  return [(NSObject*)_cb methodSignatureForSelector:sel];
}
- (void)forwardInvocation:(NSInvocation*)inv {
  [inv invokeWithTarget:_cb];
}
@end

@implementation HostQueue {
  id<MTLCommandQueue> _queue;
}
- (instancetype)initWithQueue:(id<MTLCommandQueue>)queue {
  _queue = queue;
  return self;
}
- (id<MTLCommandBuffer>)commandBuffer {
  id<MTLCommandBuffer> cb = [_queue commandBuffer];
  if (!cb) return nil;
  ++_buffers;
  int fail = 0;
  if (_failBufferAt && _buffers == _failBufferAt) fail = ++_failuresInjected;
  return (id<MTLCommandBuffer>)[[HostCommandBuffer alloc] initWithBuffer:cb queue:self fail:fail];
}
- (NSMethodSignature*)methodSignatureForSelector:(SEL)sel {
  return [(NSObject*)_queue methodSignatureForSelector:sel];
}
- (void)forwardInvocation:(NSInvocation*)inv {
  [inv invokeWithTarget:_queue];
}
@end

// ---- frames ------------------------------------------------------------------

struct Buf {
  id<MTLBuffer> buffer;
  uint32_t width = 0, height = 0, rowBytes = 0;
  ODFrame frame(int64_t key) const {
    ODFrame f = {};
    f.mtlBuffer = (__bridge void*)buffer;
    f.width = width;
    f.height = height;
    f.rowBytes = rowBytes;
    f.format = OD_PIXEL_BGRA32F;
    f.frameKey = key;
    return f;
  }
  uint8_t* bytes() const { return (uint8_t*)buffer.contents; }
  size_t size() const { return (size_t)rowBytes * height; }
};

static Buf makeBuf(uint32_t w, uint32_t h, uint32_t seed) {
  Buf b;
  b.width = w;
  b.height = h;
  b.rowBytes = w * 16;
  b.buffer = [gDevice newBufferWithLength:b.size() options:MTLResourceStorageModeShared];
  uint32_t s = seed * 2654435761u + 7u;
  float* p = (float*)b.bytes();
  for (size_t i = 0; i < (size_t)w * h; ++i) {
    for (int c = 0; c < 3; ++c) {
      s = s * 1664525u + 1013904223u;
      p[i * 4 + c] = 0.4f + 0.05f * ((float)(s >> 8) / 16777216.f - 0.5f);
    }
    p[i * 4 + 3] = 1.f;
  }
  return b;
}

static ODNoiseProfile whiteProfile(float s) {
  ODNoiseProfile p;
  memset(&p, 0, sizeof(p));
  for (int c = 0; c < 3; ++c) {
    for (int b = 0; b < OD_BANDS; ++b)
      for (int l = 0; l < OD_LEVELS; ++l) p.sigma[c][b][l] = s * (b == 0 ? 0.89f : 0.2f / (float)b);
    for (int l = 0; l < OD_LEVELS; ++l) p.sigmaFull[c][l] = s;
  }
  p.confidence = 1;
  p.valid = 1;
  return p;
}

// Frame `idx` of `src` with its neighbours within R (keys 1000 + i).
static int renderAt(ODSession* s, const std::vector<Buf>& src, int idx, int R, const ODParams& p, const Buf& out,
                    ODStats* st = nullptr) {
  ODFrame frames[OD_MAX_FRAMES];
  int offsets[OD_MAX_FRAMES];
  int n = 0;
  frames[n] = src[(size_t)idx].frame(1000 + idx);
  offsets[n++] = 0;
  for (int d = 1; d <= R; ++d)
    for (int sg : {-1, 1}) {
      const int j = idx + sg * d;
      if (j < 0 || j >= (int)src.size()) continue;
      frames[n] = src[(size_t)j].frame(1000 + j);
      offsets[n++] = sg * d;
    }
  ODFrame o = out.frame(OD_NO_KEY);
  return od_render(s, frames, offsets, n, &p, &o, st);
}

static void waitForQueue(id<MTLCommandQueue> q) {
  id<MTLCommandBuffer> cb = [q commandBuffer];
  [cb commit];
  [cb waitUntilCompleted];
}

static ODContext* contextOn(id queue) {
  char err[512] = {0};
  ODContext* c = od_context_create((__bridge void*)gDevice, (__bridge void*)queue, err, sizeof err);
  CHECK(c != nullptr, "od_context_create: %s", err);
  return c;
}

// Commits `n` command buffers on `q` that wait for `ev` to reach 1: the GPU
// runs nothing queued after them until the event is signalled.
static void blockQueue(id<MTLCommandQueue> q, id<MTLSharedEvent> ev, int n) {
  for (int i = 0; i < n; ++i) {
    id<MTLCommandBuffer> cb = [q commandBuffer];
    [cb encodeWaitForEvent:ev value:1];
    [cb commit];
  }
}

// ---- tests -------------------------------------------------------------------

// A command buffer that gives no encoder: the render fails with a message,
// leaves the output alone, keeps nothing it could not fill, and the next
// render is a fresh session's.
static void testEncoderFailure() {
  gTest = "encoder_failure";
  const uint32_t w = 320, h = 180;
  const int R = 2;
  std::vector<Buf> src;
  for (int i = 0; i < 7; ++i) src.push_back(makeBuf(w, h, 10 + (uint32_t)i));
  Buf out = makeBuf(w, h, 1), ref = makeBuf(w, h, 2);
  const ODNoiseProfile prof = whiteProfile(0.03f);
  HostQueue* q = [[HostQueue alloc] initWithQueue:gQueue];
  ODContext* ctx = contextOn(q);
  if (!ctx) return;
  struct Case {
    const char* name;
    bool blit;      // refuse the blit encoder (an identity render copies with one)
    int at;         // which request of the render is refused
    int wait;
    bool manual;
    bool identity;  // every strength 0: the output is a copy
    bool realtime;
  } cases[] = {
      {"first compute encoder, blocking", false, 1, 1, true, false, false},
      {"first compute encoder, asynchronous", false, 1, 0, true, false, false},
      {"first compute encoder, automatic profile", false, 1, 1, false, false, false},
      // A size's first estimate is waited for: the render goes on in a
      // second command buffer, whose encoder is refused.
      {"second buffer's encoder, cold profile, blocking", false, 2, 0, false, false, false},
      {"second buffer's encoder, cold profile, realtime", false, 2, 0, false, false, true},
      {"blit encoder of a copy", true, 1, 1, true, true, false},
  };
  for (const Case& c : cases) {
    ODSession* s = od_session_create(ctx);
    ODParams p = od_params_default();
    p.waitUntilCompleted = c.wait;
    p.realtime = c.realtime ? 1 : 0;
    if (c.manual) p.manualProfile = &prof;
    if (c.identity) p.lumaStrength = p.chromaStrength = p.spatialStrength = p.temporalStrength = 0;
    // With a manual profile a healthy render first (neighbours cached), then
    // the failing one; an automatic profile fails on its cold start.
    if (c.manual) CHECK(renderAt(s, src, 2, R, p, out) == 0, "%s: healthy render failed: %s", c.name, od_last_error(s));
    waitForQueue(gQueue);
    q.computeCalls = q.blitCalls = 0;
    if (c.blit) q.failBlitAt = c.at;
    else q.failComputeAt = c.at;
    memset(out.bytes(), 0x5A, out.size());
    std::vector<uint8_t> before(out.bytes(), out.bytes() + out.size());
    const int rc = renderAt(s, src, 3, R, p, out);
    waitForQueue(gQueue);
    const int calls = c.blit ? q.blitCalls : q.computeCalls;
    q.failComputeAt = q.failBlitAt = 0;
    CHECK(calls >= c.at, "%s: the render asked for %d encoders, the fault was at %d (test does not reach it)", c.name,
          calls, c.at);
    CHECK(rc != 0, "%s: render without an encoder returned 0", c.name);
    CHECK(strstr(od_last_error(s), "encoder") != nullptr, "%s: error message '%s'", c.name, od_last_error(s));
    CHECK(memcmp(before.data(), out.bytes(), out.size()) == 0, "%s: the output was written by a failed render",
          c.name);
    int entries = -1;
    od_debug_cache_info(s, &entries, nullptr, nullptr, nullptr, nullptr);
    CHECK(entries == 0, "%s: %d cache entries kept after the failure", c.name, entries);
    const std::string msg = od_last_error(s);
    // The next render: a fresh session's picture.
    ODParams pw = p;
    pw.waitUntilCompleted = 1;
    pw.realtime = 0;
    CHECK(renderAt(s, src, 3, R, pw, out) == 0, "%s: render after the failure failed: %s", c.name, od_last_error(s));
    ODSession* fresh = od_session_create(ctx);
    CHECK(renderAt(fresh, src, 3, R, pw, ref) == 0, "%s: fresh render failed", c.name);
    CHECK(memcmp(out.bytes(), ref.bytes(), out.size()) == 0, "%s: the render after the failure differs from a fresh "
          "session's", c.name);
    printf("  %-50s rc=%d: %s\n", c.name, rc, msg.c_str());
    od_session_destroy(fresh);
    od_session_destroy(s);
  }
  waitForQueue(gQueue);
  od_context_destroy(ctx);
}

// A render that did not wait and failed on the GPU: od_render had returned
// 0, so the next call says so, once; the first message stays.
static void testAsyncFailure() {
  gTest = "async_failure";
  const uint32_t w = 320, h = 180;
  const int R = 2;
  std::vector<Buf> src;
  for (int i = 0; i < 9; ++i) src.push_back(makeBuf(w, h, 40 + (uint32_t)i));
  Buf out = makeBuf(w, h, 3), ref = makeBuf(w, h, 4);
  const ODNoiseProfile prof = whiteProfile(0.03f);
  HostQueue* q = [[HostQueue alloc] initWithQueue:gQueue];
  ODContext* ctx = contextOn(q);
  if (!ctx) return;
  ODSession* s = od_session_create(ctx);
  ODParams p = od_params_default();
  p.manualProfile = &prof;  // one command buffer per render
  p.waitUntilCompleted = 0;
  ODStats st;
  CHECK(renderAt(s, src, 2, R, p, out, &st) == 0, "healthy render failed");
  od_debug_session_drain(s);
  CHECK(strlen(od_first_async_error(s)) == 0, "an error before any failure: '%s'", od_first_async_error(s));

  // Render 3's buffer fails on the GPU after od_render returned.
  q.failBufferAt = q.buffers + 1;
  ODStats failed;
  const int rc = renderAt(s, src, 3, R, p, out, &failed);
  CHECK(rc == 0, "the asynchronous render returned %d before its GPU work ran", rc);
  CHECK(od_debug_session_drain(s) == 0, "drain timed out");
  q.failBufferAt = 0;
  const std::string first = od_first_async_error(s);
  CHECK(first.find("injected failure 1") != std::string::npos, "first asynchronous error '%s'", first.c_str());

  // The next call reports it, with the failed render's id.
  ODStats next;
  CHECK(renderAt(s, src, 4, R, p, out, &next) == 0, "render after the failure failed: %s", od_last_error(s));
  CHECK(next.asyncFailures == 1, "the next call reported %d asynchronous failures, expected 1", next.asyncFailures);
  bool listed = false;
  for (int i = 0; i < next.completedCount; ++i)
    if (next.completed[i].renderId == failed.renderId) listed = next.completed[i].failed == 1;
  CHECK(listed, "the failed render (id %llu) is not among the completions marked failed",
        (unsigned long long)failed.renderId);
  od_debug_session_drain(s);
  ODStats third;
  CHECK(renderAt(s, src, 5, R, p, out, &third) == 0, "third render failed");
  CHECK(third.asyncFailures == 0, "the failure was reported again (%d)", third.asyncFailures);
  CHECK(first == od_first_async_error(s), "a later successful render changed the first error: '%s'",
        od_first_async_error(s));

  // A second failure: counted, the first message stays; od_session_reset
  // keeps it too.
  q.failBufferAt = q.buffers + 1;
  CHECK(renderAt(s, src, 6, R, p, out) == 0, "second asynchronous render failed early");
  od_debug_session_drain(s);
  q.failBufferAt = 0;
  od_session_reset(s);
  ODStats fourth;
  CHECK(renderAt(s, src, 6, R, p, out, &fourth) == 0, "render after the second failure failed");
  CHECK(fourth.asyncFailures == 1, "second failure: %d reported", fourth.asyncFailures);
  CHECK(first == od_first_async_error(s), "the second failure replaced the first message: '%s'",
        od_first_async_error(s));
  // And the picture after a failure is a fresh session's (caches dropped).
  od_debug_session_drain(s);
  q.failBufferAt = q.buffers + 1;
  CHECK(renderAt(s, src, 6, R, p, out) == 0, "third asynchronous render failed early");
  od_debug_session_drain(s);
  q.failBufferAt = 0;
  ODParams pw = p;
  pw.waitUntilCompleted = 1;
  CHECK(renderAt(s, src, 7, R, pw, out) == 0, "blocking render failed");
  ODSession* fresh = od_session_create(ctx);
  CHECK(renderAt(fresh, src, 7, R, pw, ref) == 0, "fresh render failed");
  CHECK(memcmp(out.bytes(), ref.bytes(), out.size()) == 0, "after an asynchronous failure the picture differs from "
        "a fresh session's");
  printf("  asynchronous failure: reported once by the next call (%d), first message kept: \"%s\"\n",
         next.asyncFailures, first.c_str());
  od_session_destroy(fresh);
  od_session_destroy(s);
  waitForQueue(gQueue);
  od_context_destroy(ctx);
}

// Every render's GPU timing, exactly once, under a unique increasing id.
static void testCompletionsOnce() {
  gTest = "completions_once";
  const uint32_t w = 320, h = 180;
  const int R = 2, N = 40;
  std::vector<Buf> src;
  for (int i = 0; i < N + 2 * R + 1; ++i) src.push_back(makeBuf(w, h, 80 + (uint32_t)i));
  Buf out = makeBuf(w, h, 5);
  ODContext* ctx = contextOn(gQueue);
  if (!ctx) return;
  ODSession* s = od_session_create(ctx);
  ODParams p = od_params_default();  // automatic profile: asynchronous estimates too
  p.waitUntilCompleted = 0;
  p.realtime = 1;
  std::vector<uint64_t> ids, reported;
  int zeroGpu = 0, dropped = 0;
  uint64_t lastId = 0;
  bool increasing = true;
  for (int i = 0; i < N; ++i) {
    ODStats st;
    CHECK(renderAt(s, src, R + i, R, p, out, &st) == 0, "render %d failed: %s", i, od_last_error(s));
    increasing = increasing && st.renderId > lastId;
    lastId = st.renderId;
    ids.push_back(st.renderId);
    for (int k = 0; k < st.completedCount; ++k) {
      reported.push_back(st.completed[k].renderId);
      zeroGpu += st.completed[k].gpuMs <= 0;
    }
    dropped += st.completedDropped;
  }
  od_debug_session_drain(s);
  // A blocking render reports what is left, and itself.
  ODParams pw = p;
  pw.waitUntilCompleted = 1;
  pw.realtime = 0;
  ODStats last;
  CHECK(renderAt(s, src, R + N, R, pw, out, &last) == 0, "blocking render failed");
  for (int k = 0; k < last.completedCount; ++k) reported.push_back(last.completed[k].renderId);
  // More than OD_MAX_COMPLETIONS may be waiting: they come with the next calls.
  for (int extra = 0; extra < 8 && reported.size() < ids.size() + 1; ++extra) {
    ODStats more;
    CHECK(renderAt(s, src, R + N, R, pw, out, &more) == 0, "blocking render failed");
    for (int k = 0; k < more.completedCount; ++k) reported.push_back(more.completed[k].renderId);
    ids.push_back(more.renderId);
  }
  ids.push_back(last.renderId);
  // Nothing left: a blocking render reports itself, alone, in its own call.
  ODStats self;
  CHECK(renderAt(s, src, R + N, R, pw, out, &self) == 0, "blocking render failed");
  const bool selfReported = self.completedCount == 1 && self.completed[0].renderId == self.renderId &&
                            self.completed[0].failed == 0 && self.completed[0].gpuMs > 0;
  std::multiset<uint64_t> seen(reported.begin(), reported.end());
  int missing = 0, twice = 0;
  for (uint64_t id : ids) {
    const size_t n = seen.count(id);
    missing += n == 0;
    twice += n > 1;
  }
  CHECK(increasing && ids.front() > 0, "render ids not unique and increasing");
  CHECK(missing == 0 && twice == 0 && dropped == 0, "of %zu renders, %d never reported, %d reported twice, %d dropped",
        ids.size(), missing, twice, dropped);
  CHECK(selfReported, "a blocking render does not report its own completion");
  CHECK(zeroGpu == 0, "%d completions with no GPU time", zeroGpu);
  printf("  %zu renders, each completion reported exactly once (%zu reports)\n", ids.size(), reported.size());
  od_session_destroy(s);
  od_context_destroy(ctx);
}

// Time blocked obtaining a command buffer (a full queue) and waiting for the
// session's lock is reported; neither is waitMs.
static void testCommandBufferWait() {
  gTest = "command_buffer_wait";
  const uint32_t w = 320, h = 180;
  const int R = 1;
  std::vector<Buf> src;
  for (int i = 0; i < 5; ++i) src.push_back(makeBuf(w, h, 120 + (uint32_t)i));
  Buf out = makeBuf(w, h, 6), out2 = makeBuf(w, h, 7);
  const ODNoiseProfile prof = whiteProfile(0.03f);
  id<MTLCommandQueue> small = [gDevice newCommandQueueWithMaxCommandBufferCount:2];
  ODContext* ctx = contextOn(small);
  if (!ctx) return;
  ODSession* s = od_session_create(ctx);
  ODParams p = od_params_default();
  p.manualProfile = &prof;
  p.waitUntilCompleted = 0;
  p.realtime = 1;
  CHECK(renderAt(s, src, 2, R, p, out) == 0, "warm-up render failed");
  waitForQueue(small);

  // The queue's two command buffers wait for an event signalled 80 ms later.
  const double kBlockMs = 80;
  {
    id<MTLSharedEvent> ev = [gDevice newSharedEvent];
    blockQueue(small, ev, 2);
    std::thread signaller([ev, kBlockMs] {
      usleep((useconds_t)(kBlockMs * 1000));
      ev.signaledValue = 1;
    });
    ODStats st;
    const double t0 = nowMs();
    const int rc = renderAt(s, src, 2, R, p, out, &st);
    const double wall = nowMs() - t0;
    signaller.join();
    CHECK(rc == 0, "render failed: %s", od_last_error(s));
    CHECK(st.commandBufferMs > 0.6 * kBlockMs, "blocked %.1f ms for a command buffer, reported %.2f", wall,
          st.commandBufferMs);
    CHECK(st.waitMs < 5, "waitMs %.2f: obtaining a command buffer is not waiting for the GPU", st.waitMs);
    const double unaccounted = st.cpuMs - st.commandBufferMs - st.waitMs - st.lockWaitMs;
    CHECK(unaccounted < 25, "%.1f ms of the call (%.1f) not accounted for", unaccounted, st.cpuMs);
    printf("  full queue: call %.1f ms, command buffer %.1f ms, waiting for the GPU %.2f ms\n", st.cpuMs,
           st.commandBufferMs, st.waitMs);
  }
  waitForQueue(small);

  // Two calls on one session: the second waits for the first, which is
  // stuck on the full queue.
  {
    id<MTLSharedEvent> ev = [gDevice newSharedEvent];
    blockQueue(small, ev, 2);
    ODStats a, b;
    std::atomic<int> rcA{-1}, rcB{-1};
    std::thread first([&] { rcA = renderAt(s, src, 2, R, p, out, &a); });
    usleep(15000);
    std::thread second([&] { rcB = renderAt(s, src, 3, R, p, out2, &b); });
    usleep((useconds_t)(kBlockMs * 1000));
    ev.signaledValue = 1;
    first.join();
    second.join();
    CHECK(rcA == 0 && rcB == 0, "renders failed (%d, %d)", rcA.load(), rcB.load());
    CHECK(b.lockWaitMs > 0.4 * kBlockMs, "the second call waited %.2f ms for the session, expected about %.0f",
          b.lockWaitMs, kBlockMs - 15);
    CHECK(b.cpuMs >= b.lockWaitMs, "lockWaitMs %.1f beyond the call's own time %.1f", b.lockWaitMs, b.cpuMs);
    printf("  two calls on one session: the second waited %.1f ms for the lock (first: %.1f ms for a buffer)\n",
           b.lockWaitMs, a.commandBufferMs);
  }
  waitForQueue(small);
  od_session_destroy(s);
  od_context_destroy(ctx);
}

// A host that renders faster than the GPU: the session's outstanding work
// stays within the queue's capacity, and the wait it causes is reported.
static void testInFlightBounded() {
  gTest = "in_flight_bounded";
  const uint32_t w = 640, h = 360;
  const int R = 2, N = 60;
  std::vector<Buf> src;
  for (int i = 0; i < 9; ++i) src.push_back(makeBuf(w, h, 160 + (uint32_t)i));
  Buf out = makeBuf(w, h, 8);
  const int kCapacity = 4;
  id<MTLCommandQueue> q = [gDevice newCommandQueueWithMaxCommandBufferCount:kCapacity];
  ODContext* ctx = contextOn(q);
  if (!ctx) return;
  ODSession* s = od_session_create(ctx);
  const ODNoiseProfile prof = whiteProfile(0.03f);
  ODParams p = od_params_default();
  p.manualProfile = &prof;
  p.waitUntilCompleted = 0;
  p.realtime = 1;
  int maxInFlight = 0;
  double blocked = 0;
  for (int i = 0; i < N; ++i) {
    ODStats st;
    CHECK(renderAt(s, src, 2 + i % 5, R, p, out, &st) == 0, "render %d failed", i);
    maxInFlight = std::max(maxInFlight, st.inFlight);
    blocked += st.commandBufferMs;
  }
  od_debug_session_drain(s);
  // A buffer's slot frees when it completes; its completion handler, which
  // ends its count, may run a moment later: one or two over at most.
  CHECK(maxInFlight <= kCapacity + 2, "%d renders in flight on a queue of %d command buffers", maxInFlight, kCapacity);
  CHECK(maxInFlight >= 1, "nothing was ever in flight (the test does not exercise the path)");
  printf("  %d renders as fast as possible on a queue of %d: at most %d in flight, %.1f ms blocked for buffers\n", N,
         kCapacity, maxInFlight, blocked);
  od_session_destroy(s);
  od_context_destroy(ctx);
}

// od_wait_idle: what a plug-in calls at global teardown.
static void testWaitIdle() {
  gTest = "wait_idle";
  const uint32_t w = 320, h = 180;
  const int R = 2;
  std::vector<Buf> src;
  for (int i = 0; i < 7; ++i) src.push_back(makeBuf(w, h, 200 + (uint32_t)i));
  Buf out = makeBuf(w, h, 9);
  waitForQueue(gQueue);
  CHECK(od_wait_idle(2000) == 0, "work pending before the test");
  id<MTLCommandQueue> q = [gDevice newCommandQueue];
  ODContext* ctx = contextOn(q);
  if (!ctx) return;
  ODSession* s = od_session_create(ctx);
  ODParams p = od_params_default();  // automatic profile: estimates queued after the GPU work
  p.waitUntilCompleted = 0;
  p.realtime = 1;
  // Warm up (first estimates), then block the GPU and queue renders behind it.
  for (int i = 0; i < 3; ++i) renderAt(s, src, 2 + i, R, p, out);
  waitForQueue(q);
  od_debug_session_drain(s);
  id<MTLSharedEvent> ev = [gDevice newSharedEvent];
  blockQueue(q, ev, 1);
  for (int i = 0; i < 4; ++i) CHECK(renderAt(s, src, 2 + i % 3, R, p, out) == 0, "render %d failed", i);
  // The session and the context go while their work is still queued.
  od_session_destroy(s);
  od_context_destroy(ctx);
  const double t0 = nowMs();
  const int stuck = od_wait_idle(50);
  const double waited = nowMs() - t0;
  CHECK(stuck > 0, "od_wait_idle found nothing pending while the GPU was blocked");
  CHECK(waited >= 45 && waited < 1000, "od_wait_idle(50) returned after %.1f ms", waited);
  std::thread signaller([ev] {
    usleep(60000);
    ev.signaledValue = 1;
  });
  const double t1 = nowMs();
  const int left = od_wait_idle(10000);
  const double drained = nowMs() - t1;
  signaller.join();
  CHECK(left == 0, "%d items still pending after the GPU ran", left);
  CHECK(drained >= 40, "od_wait_idle returned after %.1f ms, before the GPU could run (signalled at 60 ms)", drained);
  printf("  blocked GPU: %d items pending, od_wait_idle(50) returned after %.1f ms; after the signal idle in %.1f ms\n",
         stuck, waited, drained);
}

int main() {
  alarm(120);  // a stuck GPU wait must not hang the suite
  @autoreleasepool {
    gDevice = MTLCreateSystemDefaultDevice();
    if (!gDevice) {
      fprintf(stderr, "no Metal device\n");
      return 2;
    }
    gQueue = [gDevice newCommandQueue];
    struct {
      const char* name;
      void (*fn)();
    } tests[] = {
        {"encoder_failure", testEncoderFailure},     {"async_failure", testAsyncFailure},
        {"completions_once", testCompletionsOnce},   {"command_buffer_wait", testCommandBufferWait},
        {"in_flight_bounded", testInFlightBounded}, {"wait_idle", testWaitIdle},
    };
    for (auto& t : tests) {
      const int before = gFailures;
      @autoreleasepool {
        t.fn();
      }
      printf("%-24s %s\n", t.name, gFailures == before ? "ok" : "FAILED");
    }
    printf("host errors: %d checks, %d failures (device: %s)\n", gChecks, gFailures, gDevice.name.UTF8String);
    return gFailures ? 1 : 0;
  }
}
