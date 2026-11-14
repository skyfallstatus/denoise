// SPDX-License-Identifier: MIT
#pragma once
#import <Metal/Metal.h>
#include <cstdint>
#include <memory>

namespace od {
// Opt-in diagnostics. Separate encoder timestamps change scheduling; do not
// add the overlapping pass durations to predict production frame latency.
struct KernelTraceState;
bool kernelTraceEnabled();
id<MTLComputeCommandEncoder> kernelTraceEncoder(id<MTLDevice> device, id<MTLCommandBuffer> cb,
                                                const char* label, uint64_t render, std::shared_ptr<KernelTraceState>& state);
}  // namespace od
