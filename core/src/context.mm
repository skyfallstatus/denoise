// SPDX-License-Identifier: MIT
#include "context.h"

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

#include "shaders.h"

namespace od {

Context::Context(id<MTLDevice> device, id<MTLCommandQueue> queue)
    : device_(device), queue_(queue), memory_(Memory::create(device)) {
  libraries_.resize((size_t)kShaderUnitCount, nil);
}

// Sessions hold the context, so none is left by now.
Context::~Context() { memory_->shutdown(); }

id<MTLLibrary> Context::libraryLocked(int unit, std::string* err) {
  if (libraries_[(size_t)unit]) return libraries_[(size_t)unit];
  const ShaderUnit& u = kShaderUnits[unit];
  MTLCompileOptions* opts = [MTLCompileOptions new];
  // Math mode is a per-unit, deliberate choice (see scripts/embed_shaders.py).
  // Default safe: the strength-0 identity and NaN scrubbing depend on IEEE
  // behaviour. Heavy filter units opt in to relaxed/fast.
  if (@available(macOS 15.0, *)) {
    opts.mathMode = u.mathMode == 2 ? MTLMathModeFast : (u.mathMode == 1 ? MTLMathModeRelaxed : MTLMathModeSafe);
  } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    opts.fastMathEnabled = u.mathMode != 0;
#pragma clang diagnostic pop
  }
  NSError* e = nil;
  id<MTLLibrary> lib = [device_ newLibraryWithSource:[NSString stringWithUTF8String:u.source] options:opts error:&e];
  if (!lib) {
    if (err) {
      *err = std::string("shader ") + u.name + " failed to compile: " +
             (e ? e.localizedDescription.UTF8String : "unknown error");
    }
    return nil;
  }
  lib.label = [NSString stringWithUTF8String:u.name];
  libraries_[(size_t)unit] = lib;
  return lib;
}

id<MTLComputePipelineState> Context::pipeline(const char* kernelName, std::string* err) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = pipelines_.find(kernelName);
  if (it != pipelines_.end()) return it->second;

  int unit = -1;
  for (int i = 0; i < kShaderKernelCount; ++i) {
    if (std::strcmp(kShaderKernels[i].function, kernelName) == 0) {
      unit = kShaderKernels[i].unit;
      break;
    }
  }
  if (unit < 0) {
    if (err) *err = std::string("no embedded shader defines kernel '") + kernelName + "'";
    return nil;
  }
  id<MTLLibrary> lib = libraryLocked(unit, err);
  if (!lib) return nil;
  id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:kernelName]];
  if (!fn) {
    if (err) *err = std::string("shader ") + kShaderUnits[unit].name + ": kernel '" + kernelName + "' not found after compile";
    return nil;
  }
  NSError* e = nil;
  MTLComputePipelineDescriptor* pd = [MTLComputePipelineDescriptor new];
  pd.computeFunction = fn;
  pd.label = [NSString stringWithUTF8String:kernelName];
  id<MTLComputePipelineState> pso = [device_ newComputePipelineStateWithDescriptor:pd options:MTLPipelineOptionNone reflection:nil error:&e];
  if (!pso) {
    if (err) {
      *err = std::string("shader ") + kShaderUnits[unit].name + ": pipeline for kernel '" + kernelName +
             "' failed: " + (e ? e.localizedDescription.UTF8String : "unknown error");
    }
    return nil;
  }
  if (getenv("OD_KERNEL_TRACE")) fprintf(stderr, "PIPELINE,%s,simd=%lu,maxThreads=%lu,staticTGBytes=%lu\n", kernelName, (unsigned long)pso.threadExecutionWidth, (unsigned long)pso.maxTotalThreadsPerThreadgroup, (unsigned long)pso.staticThreadgroupMemoryLength);
  pipelines_.emplace(kernelName, pso);
  return pso;
}

bool Context::compileAll(std::string* err) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < kShaderUnitCount; ++i) {
      if (!libraryLocked(i, err)) return false;
    }
  }
  for (int i = 0; i < kShaderKernelCount; ++i) {
    if (!pipeline(kShaderKernels[i].function, err)) return false;
  }
  return true;
}

std::shared_ptr<void> Context::shared(const std::string& key, const std::function<std::shared_ptr<void>()>& make) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::shared_ptr<void>& slot = shared_[key];
  if (!slot) slot = make();
  return slot;
}

MTLSize Context::threadgroupFor(id<MTLComputePipelineState> pso) {
  NSUInteger w = pso.threadExecutionWidth;
  static const int cap = [] { const char* e = getenv("OD_TG_CAP"); return e ? std::max(32, atoi(e)) : 0; }();
  NSUInteger limit = cap ? std::min((NSUInteger)cap, pso.maxTotalThreadsPerThreadgroup) : pso.maxTotalThreadsPerThreadgroup;
  NSUInteger h = limit / w;
  if (h < 1) h = 1;
  return MTLSizeMake(w, h, 1);
}

}  // namespace od
