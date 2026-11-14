// SPDX-License-Identifier: MIT
// Open Denoise — per-render instrumentation of the GPU path (OpenDenoiseGPU.mm),
// header-only so that the plug-in harness can test the aggregation itself.
//
// What a run in Premiere needs to explain dropped frames (3840x2160 at 50
// fps, radius 2, dropped frames while the GPU time per frame was under half
// the frame interval): for every window of renders (250, or 5 s with at
// least 20), the distribution of the time spent in Render and in the core
// call, of the waits inside the core call (for the GPU, for the session's
// lock, for a command buffer), of the time Premiere took between asking for
// a frame's neighbours and handing them over, of our command buffers' GPU
// time and of how long they sat on the queue; the inQuality values and frame
// sizes Premiere asked for; how many renders ran at once and how much of our
// work was still in flight. Plus a line for every slow render and every
// change of inQuality (budgeted). One mutex and a few stores per render; the
// percentiles are computed when a window is logged.
//
// Attribution: every render has a process-unique id; the GPU time of a render is counted
// once, when the core hands over its completion (ODStats.completed), not sampled per call;
// how sequence time moves between requests is followed per request stream (one effect on one
// track item in one timeline, field and frame size), so two clips or two sequences rendered
// at the same times are not "frames rendered again". Those counts describe what Premiere
// asked for, not frames it displayed or dropped.
//
// Every render is counted, however it ends. The waits and the core call's percentiles cover
// the renders that reached the core.
#ifndef OPENDENOISE_RENDER_STATS_H
#define OPENDENOISE_RENDER_STATS_H

#include <pthread.h>
#include <stdint.h>
#include <algorithm>
#include <map>
#include <mutex>
#include <vector>

#include "OpenDenoisePlugin.h"

namespace odplug {

struct RenderSample {
  uint64_t id = 0;           // the plug-in's render id, unique in the process
  uint64_t coreId = 0;       // ODStats.renderId (0: the core call failed early)
  double enterMs = 0;        // Render entry, monotonic
  double totalMs = 0;        // Render entry to return
  double coreMs = 0;         // the od_render call
  double paramsMs = 0;       // reading the parameters (0 when GetFrameDependencies had them)
  double depsToRenderMs = -1;  // first GetFrameDependencies of this frame to Render; < 0 unknown
  double waitMs = 0;         // ODStats.waitMs: the core waited for the GPU
  double lockWaitMs = 0;     // waited before the render started (the plug-in's lock, the core's admission and lock)
  double commandBufferMs = 0;  // ODStats.commandBufferMs: blocked obtaining command buffers
  double frameMs = 0;        // one frame of the sequence, from inRenderTicksPerFrame
  int64_t sequenceTime = 0;
  int wanted = 0, got = 0, cached = 0, inQuality = 0, coreQuality = 0, realtime = 0, active = 0;
  int inFlight = 0;          // ODStats.inFlight
  int asyncFailures = 0;     // ODStats.asyncFailures: earlier renders that failed on the GPU
  uint32_t width = 0, height = 0;  // 0: not known (the input could not be described)
  bool spatialOnly = false, advancedUnread = false;
  bool reachedCore = false;   // the core was called (coreMs and the waits are its)
  bool failed = false;        // no denoised frame: the core call failed, the render ended before it, or an
                              // exception; the output then holds the input, or an error was returned
  bool errorReturned = false;  // Render returned an error to Premiere (no valid output)
};

class RenderStats {
 public:
  // window: renders per logged window; flushFirst: log the first render on
  // its own (the plug-in does, so that a run shows the stats from its start).
  explicit RenderStats(int window = 250, bool flushFirst = true) : window_(window), flushFirst_(flushFirst) {}
  static RenderStats& Get() {
    static auto* s = new RenderStats();  // never destroyed: renders may outlive static teardown
    return *s;
  }

  void NoteDeps(double ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    depsCalls_ += 1;
    depsMs_ += ms;
  }

  // At Render entry, in the order Premiere calls, for one request stream:
  // how sequence time moves from one request to the next (one frame forward
  // in steady playback; more than one when Premiere skips frames), requests
  // for a time the stream asked for recently, and the time between requests.
  void NoteEnter(uint64_t stream, double enterMs, int64_t sequenceTime, int64_t ticksPerFrame, double frameMs) {
    std::lock_guard<std::mutex> lock(mutex_);
    Stream& s = streams_[stream];
    s.lastUse = ++useClock_;
    if (s.have && ticksPerFrame > 0) {
      const int64_t d = sequenceTime - s.lastSeq;
      if (d == ticksPerFrame) {
        ++stepOne_;
      } else if (d > ticksPerFrame && d <= 50 * ticksPerFrame) {
        ++jumps_;
        notAsked_ += (int)(d / ticksPerFrame) - 1;
      } else {
        ++otherSteps_;
      }
      const double gap = enterMs - s.lastEnterMs;
      if (gap >= 0) {
        interval_.push_back(gap);
        if (frameMs > 0 && gap > 1.5 * frameMs) ++gaps_;
      }
    }
    for (int i = 0; i < s.recentCount; ++i)
      if (s.recent[i] == sequenceTime) {
        ++repeats_;
        break;
      }
    s.recent[s.recentNext] = sequenceTime;
    s.recentNext = (s.recentNext + 1) % kRecent;
    s.recentCount = std::min(kRecent, s.recentCount + 1);
    s.have = true;
    s.lastSeq = sequenceTime;
    s.lastEnterMs = enterMs;
    // Bounded: the least recently used stream goes.
    if (streams_.size() > kMaxStreams) {
      auto oldest = streams_.begin();
      for (auto it = streams_.begin(); it != streams_.end(); ++it)
        if (it->second.lastUse < oldest->second.lastUse) oldest = it;
      streams_.erase(oldest);
    }
  }

  // Renders whose GPU work completed (ODStats.completed): each counted once.
  void NoteCompletions(const ODCompletion* c, int n, int dropped) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < n; ++i) {
      gpu_.push_back(c[i].gpuMs);
      queue_.push_back(c[i].queueMs);
      latency_.push_back(c[i].latencyMs);
      gpuFailed_ += c[i].failed ? 1 : 0;
    }
    completionsDropped_ += dropped;
  }

  void Note(const RenderSample& r) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++total_;
    if (n_ == 0) windowStart_ = r.enterMs;
    ++n_;
    total_ms_.push_back(r.totalMs);
    if (r.reachedCore) {
      ++reached_;
      core_.push_back(r.coreMs);
      params_.push_back(r.paramsMs);
      lockWait_.push_back(r.lockWaitMs);
      cbWait_.push_back(r.commandBufferMs);
      if (r.waitMs > 0.05) {
        ++waited_;
        waitMax_ = std::max(waitMax_, r.waitMs);
      }
    }
    if (r.depsToRenderMs >= 0) deps_.push_back(r.depsToRenderMs);
    inQ_[(r.inQuality >= 0 && r.inQuality < 7) ? r.inQuality : 7]++;
    realtime_ += r.realtime;
    q1_ += r.coreQuality;
    wanted_ += r.wanted;
    got_ += r.got;
    cached_ += r.cached;
    spatialOnly_ += r.spatialOnly;
    failed_ += r.failed;
    errors_ += r.errorReturned;
    asyncFailures_ += r.asyncFailures;
    advancedUnread_ += r.advancedUnread;
    activeMax_ = std::max(activeMax_, r.active);
    inFlightMax_ = std::max(inFlightMax_, r.inFlight);
    const uint64_t tid = (uint64_t)pthread_mach_thread_np(pthread_self());
    if (std::find(threads_.begin(), threads_.end(), tid) == threads_.end() && threads_.size() < 64) threads_.push_back(tid);
    if (r.width && r.height) {
      if (r.width != lastW_ || r.height != lastH_) ++sizeChanges_;
      lastW_ = r.width;
      lastH_ = r.height;
    }

    // inQuality changes (what Premiere passes while playing, paused and on
    // export), and slow renders, each on a line of their own.
    if (r.inQuality != lastInQuality_ && qualityLines_ < 256) {
      ++qualityLines_;
      Log("GPU: inQuality %d -> %d at render %llu (%ux%u, sequence time %lld)", lastInQuality_, r.inQuality,
          (unsigned long long)r.id, r.width, r.height, (long long)r.sequenceTime);
    }
    lastInQuality_ = r.inQuality;
    if ((r.totalMs > 8.0 || r.coreMs > 4.0) && slowLines_ < 512) {
      ++slowLines_;
      Log("GPU: slow render %llu (core render %llu): %.2f ms in Render (core call %.2f: waited %.2f for the GPU, "
          "%.2f for the session, %.2f for a command buffer; parameters %.2f), %ux%u, sequence time %lld, inQuality "
          "%d, realtime %d, neighbours %d of %d, %d running at once, %d of our renders in flight, neighbours asked "
          "%.2f ms before",
          (unsigned long long)r.id, (unsigned long long)r.coreId, r.totalMs, r.coreMs, r.waitMs, r.lockWaitMs,
          r.commandBufferMs, r.paramsMs, r.width, r.height, (long long)r.sequenceTime, r.inQuality, r.realtime,
          r.got, r.wanted, r.active, r.inFlight, r.depsToRenderMs);
    }
    if ((flushFirst_ && total_ == 1) || n_ >= window_ || (n_ >= 20 && r.enterMs - windowStart_ >= 5000))
      Flush(r.enterMs);
  }

  // The current window's counts (tests).
  struct Snapshot {
    int renders, stepOne, jumps, notAsked, otherSteps, repeats, streams, gaps, intervals;
    int gpuSamples, gpuFailed, completionsDropped, failed, errors, asyncFailures, advancedUnread, inFlightMax;
    int reachedCore, coreSamples, sizeChanges;
    uint32_t lastW, lastH;
    double lockWaitMax, commandBufferMax;
    uint64_t windows;
  };
  Snapshot Current() {
    std::lock_guard<std::mutex> lock(mutex_);
    Snapshot s = {};
    s.renders = n_;
    s.stepOne = stepOne_;
    s.jumps = jumps_;
    s.notAsked = notAsked_;
    s.otherSteps = otherSteps_;
    s.repeats = repeats_;
    s.streams = (int)streams_.size();
    s.gaps = gaps_;
    s.intervals = (int)interval_.size();
    s.gpuSamples = (int)gpu_.size();
    s.gpuFailed = gpuFailed_;
    s.completionsDropped = completionsDropped_;
    s.failed = failed_;
    s.errors = errors_;
    s.reachedCore = reached_;
    s.coreSamples = (int)core_.size();
    s.sizeChanges = sizeChanges_;
    s.lastW = lastW_;
    s.lastH = lastH_;
    s.asyncFailures = asyncFailures_;
    s.advancedUnread = advancedUnread_;
    s.inFlightMax = inFlightMax_;
    s.lockWaitMax = lockWait_.empty() ? 0 : *std::max_element(lockWait_.begin(), lockWait_.end());
    s.commandBufferMax = cbWait_.empty() ? 0 : *std::max_element(cbWait_.begin(), cbWait_.end());
    s.windows = windows_;
    return s;
  }

 private:
  struct Pct {
    double p50 = 0, p95 = 0, max = 0;
  };
  static Pct Of(std::vector<double>& v) {
    Pct p;
    if (v.empty()) return p;
    auto at = [&](double q) {
      const size_t k = std::min(v.size() - 1, (size_t)(q * (double)(v.size() - 1) + 0.5));
      std::nth_element(v.begin(), v.begin() + (long)k, v.end());
      return v[k];
    };
    p.p50 = at(0.5);
    p.p95 = at(0.95);
    p.max = *std::max_element(v.begin(), v.end());
    return p;
  }
  void Flush(double now) {
    ++windows_;
    const double secs = std::max(1e-3, (now - windowStart_) / 1000.0);
    const Pct total = Of(total_ms_), core = Of(core_), params = Of(params_), deps = Of(deps_), gap = Of(interval_),
              gpu = Of(gpu_), queue = Of(queue_), lat = Of(latency_), lockw = Of(lockWait_), cbw = Of(cbWait_);
    const double n = (double)n_;
    const double reached = (double)std::max(1, reached_);
    Log("GPU: stats %llu, renders %llu-%llu in %.2f s (%.1f/s), last %ux%u (%d size changes), inQuality "
        "[%d %d %d %d %d %d %d | other %d], realtime %d, core quality 1 on %d; ms p50/p95/max: Render %.2f/%.2f/%.2f, "
        "core call %.2f/%.2f/%.2f (%d renders reached it), of which waiting for the session %.2f/%.2f/%.2f and for "
        "a command buffer %.2f/%.2f/%.2f, parameters %.3f/%.3f/%.3f; waited for the GPU in %d (max %.1f ms); failed "
        "%d (%d of them returned an error to Premiere, the others the input), found failed on the GPU after "
        "returning %d; advanced controls unread in %d",
        (unsigned long long)windows_, (unsigned long long)(total_ - n_ + 1), (unsigned long long)total_, secs, n / secs,
        lastW_, lastH_, sizeChanges_, inQ_[0], inQ_[1], inQ_[2], inQ_[3], inQ_[4], inQ_[5], inQ_[6], inQ_[7],
        realtime_, q1_, total.p50, total.p95, total.max, core.p50, core.p95, core.max, reached_, lockw.p50,
        lockw.p95, lockw.max, cbw.p50, cbw.p95, cbw.max, params.p50, params.p95, params.max, waited_, waitMax_,
        failed_, errors_, asyncFailures_, advancedUnread_);
    Log("GPU: stats %llu, host side: ms p50/p95/max: between requests of a stream %.2f/%.2f/%.2f (%d gaps over 1.5 "
        "frames), neighbours asked to Render %.2f/%.2f/%.2f (%zu), GPU time per completed render %.2f/%.2f/%.2f (%zu "
        "completions, %d failed, %d not reported), queued before the GPU started %.2f/%.2f/%.2f, commit to done "
        "%.2f/%.2f/%.2f; GetFrameDependencies %llu calls, %.3f ms each; at most %d renders at once on %zu threads, at "
        "most %d of ours in flight; neighbours per render that reached the core %.2f wanted, %.2f received, %.2f "
        "taken from the frame cache (0 while the keys do not name the content); spatial only %d; request pattern "
        "over %zu streams (what Premiere "
        "asked for, not frames shown or dropped): %d steps of one frame, %d jumps forward (%d frames not asked for), "
        "%d other steps, %d times asked again",
        (unsigned long long)windows_, gap.p50, gap.p95, gap.max, gaps_, deps.p50, deps.p95, deps.max, deps_.size(),
        gpu.p50, gpu.p95, gpu.max, gpu_.size(), gpuFailed_, completionsDropped_, queue.p50, queue.p95, queue.max,
        lat.p50, lat.p95, lat.max, (unsigned long long)depsCalls_, depsCalls_ ? depsMs_ / (double)depsCalls_ : 0.0,
        activeMax_, threads_.size(), inFlightMax_, wanted_ / reached, got_ / reached, cached_ / reached, spatialOnly_,
        streams_.size(), stepOne_, jumps_, notAsked_, otherSteps_, repeats_);
    n_ = 0;
    total_ms_.clear(); core_.clear(); params_.clear(); deps_.clear(); interval_.clear(); lockWait_.clear();
    cbWait_.clear(); gpu_.clear(); queue_.clear(); latency_.clear(); threads_.clear();
    gaps_ = waited_ = realtime_ = q1_ = spatialOnly_ = failed_ = errors_ = reached_ = activeMax_ = sizeChanges_ = 0;
    asyncFailures_ = advancedUnread_ = inFlightMax_ = gpuFailed_ = completionsDropped_ = 0;
    stepOne_ = jumps_ = notAsked_ = otherSteps_ = repeats_ = 0;
    wanted_ = got_ = cached_ = 0;
    waitMax_ = depsMs_ = 0;
    depsCalls_ = 0;
    for (int& q : inQ_) q = 0;
  }

  static constexpr int kRecent = 32;
  static constexpr size_t kMaxStreams = 64;
  struct Stream {
    bool have = false;
    int64_t lastSeq = 0;
    double lastEnterMs = 0;
    int64_t recent[kRecent] = {};
    int recentNext = 0, recentCount = 0;
    uint64_t lastUse = 0;
  };

  const int window_;
  const bool flushFirst_;
  std::mutex mutex_;
  uint64_t total_ = 0, windows_ = 0, depsCalls_ = 0, useClock_ = 0;
  int n_ = 0;
  double windowStart_ = 0, waitMax_ = 0, depsMs_ = 0;
  std::vector<double> total_ms_, core_, params_, deps_, interval_, lockWait_, cbWait_, gpu_, queue_, latency_;
  std::vector<uint64_t> threads_;
  int inQ_[8] = {};
  int gaps_ = 0, waited_ = 0, realtime_ = 0, q1_ = 0, spatialOnly_ = 0, failed_ = 0, errors_ = 0, reached_ = 0;
  int activeMax_ = 0, sizeChanges_ = 0;
  int asyncFailures_ = 0, advancedUnread_ = 0, inFlightMax_ = 0, gpuFailed_ = 0, completionsDropped_ = 0;
  double wanted_ = 0, got_ = 0, cached_ = 0;
  uint32_t lastW_ = 0, lastH_ = 0;
  int lastInQuality_ = -1;
  int qualityLines_ = 0, slowLines_ = 0;
  std::map<uint64_t, Stream> streams_;
  int stepOne_ = 0, jumps_ = 0, notAsked_ = 0, otherSteps_ = 0, repeats_ = 0;
};

}  // namespace odplug

#endif  // OPENDENOISE_RENDER_STATS_H
