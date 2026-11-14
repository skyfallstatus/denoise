// SPDX-License-Identifier: MIT
#include "kernel_trace.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace od {
namespace {
int skip() {
  static const int n = [] { const char* p = getenv("OD_KERNEL_TRACE"); return p ? std::max(0, atoi(p)) : -1; }();
  return n;
}
struct Sample { double ms = 0; unsigned passes = 0; };
struct Totals {
  std::mutex mutex;
  std::map<std::string, std::map<uint64_t, Sample>> rows;
  ~Totals() {
    // Single-session diagnostic runs only. Missing (throttled) frames are
    // zero, so the mean is per rendered frame, not per measured profile.
    uint64_t last = (uint64_t)std::max(skip(), 0);
    for (const auto& row : rows) if (!row.second.empty()) last = std::max(last, row.second.rbegin()->first);
    const uint64_t n = last - (uint64_t)std::max(skip(), 0);
    if (!n) return;
    fprintf(stderr, "kernel trace: %llu renders, overlapping pass durations; mean/median ms, passes/frame\n", (unsigned long long)n);
    for (const auto& row : rows) {
      std::vector<double> v((size_t)n, 0);
      double sum = 0; unsigned count = 0;
      for (const auto& r : row.second) {
        v[(size_t)(r.first - skip() - 1)] = r.second.ms;
        sum += r.second.ms; count += r.second.passes;
      }
      std::sort(v.begin(), v.end());
      fprintf(stderr, "KERNEL,%s,%.6f,%.6f,%.3f\n", row.first.c_str(), sum / n, v[v.size()/2], (double)count/n);
    }
  }
};
Totals& totals() { static Totals t; return t; }
}  // namespace
bool kernelTraceEnabled() { return skip() >= 0; }
struct KernelTraceState {
  uintptr_t command = 0;
  id<MTLCounterSampleBuffer> samples = nil;
  std::vector<std::string> labels;
};
id<MTLComputeCommandEncoder> kernelTraceEncoder(id<MTLDevice> device, id<MTLCommandBuffer> cb,
                                                const char* label, uint64_t render,
                                                std::shared_ptr<KernelTraceState>& state) {
  if (render <= (uint64_t)skip()) return [cb computeCommandEncoder];
  constexpr size_t maxPasses = 512;
  if (!state || state->command != (uintptr_t)(__bridge void*)cb) {
    state = std::make_shared<KernelTraceState>();
    state->command = (uintptr_t)(__bridge void*)cb;
    id<MTLCounterSet> set = nil;
    for (id<MTLCounterSet> s in device.counterSets)
      if ([s.name isEqualToString:MTLCommonCounterSetTimestamp]) set = s;
    if (!set || ![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])
      return [cb computeCommandEncoder];
    MTLCounterSampleBufferDescriptor* d = [MTLCounterSampleBufferDescriptor new];
    d.counterSet = set; d.storageMode = MTLStorageModeShared; d.sampleCount = maxPasses * 2;
    state->samples = [device newCounterSampleBufferWithDescriptor:d error:nil];
    const auto saved = state;
    (void)totals();
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
      if (done.status != MTLCommandBufferStatusCompleted || !saved->samples) return;
      const size_t n = saved->labels.size();
      NSData* data = [saved->samples resolveCounterRange:NSMakeRange(0, 2*n)];
      if (data.length < 2*n*sizeof(MTLCounterResultTimestamp)) return;
      const auto* t = (const MTLCounterResultTimestamp*)data.bytes;
      Totals& g = totals();
      std::lock_guard<std::mutex> lock(g.mutex);
      for (size_t i = 0; i < n; ++i) {
        if (t[2*i].timestamp == MTLCounterErrorValue || t[2*i+1].timestamp == MTLCounterErrorValue || t[2*i+1].timestamp < t[2*i].timestamp) {
          fprintf(stderr, "KERNEL_INVALID,%s\n", saved->labels[i].c_str());
          continue;
        }
        Sample& s = g.rows[saved->labels[i]][render];
        s.ms += (t[2*i+1].timestamp-t[2*i].timestamp)*1e-6;
        ++s.passes;
      }
    }];
  }
  if (!state->samples || state->labels.size() >= maxPasses) return [cb computeCommandEncoder];
  MTLComputePassDescriptor* pass = [MTLComputePassDescriptor computePassDescriptor];
  pass.sampleBufferAttachments[0].sampleBuffer = state->samples;
  pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = state->labels.size()*2;
  pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = state->labels.size()*2+1;
  state->labels.emplace_back(label);
  return [cb computeCommandEncoderWithDescriptor:pass];
}
}  // namespace od
