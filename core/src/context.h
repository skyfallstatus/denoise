// SPDX-License-Identifier: MIT
// od::Context — one per MTLDevice. Owns the device, the command queue and
// the cache of shader libraries and compute pipeline states. Thread-safe.
#ifndef OD_CONTEXT_H
#define OD_CONTEXT_H

#import <Metal/Metal.h>

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "memory.h"

namespace od {

class Context {
 public:
  Context(id<MTLDevice> device, id<MTLCommandQueue> queue);
  ~Context();

  id<MTLDevice> device() const { return device_; }
  id<MTLCommandQueue> queue() const { return queue_; }

  // Compute pipeline for the kernel function `kernelName` (the name after
  // `kernel void` in a core/shaders unit). Compiles the unit's library and the
  // pipeline on first use, then returns the cached object; safe to call from
  // any thread and cheap after the first call. Returns nil on failure, with a
  // message naming the shader file in *err (if non-null). Stages should fetch
  // their pipelines once, in their constructor.
  id<MTLComputePipelineState> pipeline(const char* kernelName, std::string* err = nullptr);

  // Compiles every embedded unit and every kernel. Used by tests so a broken
  // shader is caught without rendering. Returns false and the first error.
  bool compileAll(std::string* err);

  // Threadgroup size used by RenderEnv::dispatch for this pipeline.
  static MTLSize threadgroupFor(id<MTLComputePipelineState> pso);

  // GPU memory shared by the sessions of this context: texture pool, byte
  // budget over pool and caches, idle trim (memory.h).
  Memory& memory() const { return *memory_; }

  // One object per context and key, made by `make` on first use and kept as
  // long as the context: state a stage shares across the sessions of a
  // context instead of keeping one per session (for example the profile
  // stage's statistics buffers). Thread-safe.
  std::shared_ptr<void> shared(const std::string& key, const std::function<std::shared_ptr<void>()>& make);

 private:
  id<MTLLibrary> libraryLocked(int unit, std::string* err);

  id<MTLDevice> device_;
  id<MTLCommandQueue> queue_;
  std::mutex mutex_;
  std::vector<id<MTLLibrary>> libraries_;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines_;

  std::unordered_map<std::string, std::shared_ptr<void>> shared_;  // under mutex_
  std::shared_ptr<Memory> memory_;
};

}  // namespace od

// The public opaque handle.
struct ODContext {
  std::shared_ptr<od::Context> impl;
};

#endif
