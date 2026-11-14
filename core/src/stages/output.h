// SPDX-License-Identifier: MIT
// output stage: working texture -> host BGRA MTLBuffer, as
//   out = source pixel + toRGB(final - ingested)      (see output_buffer.metal)
#ifndef OD_OUTPUT_H
#define OD_OUTPUT_H

#include "stage.h"

namespace od {

enum class OutputMode : uint32_t {
  Delta = 0,     // normal
  NoiseView = 1, // 0.5 - gain * removed
  Absolute = 2,  // final image as is (motion / profile views)
};

constexpr float kNoiseViewGain = 4.0f;

class OutputStage {
 public:
  static std::unique_ptr<OutputStage> make(Context& ctx, std::string* err);
  // src: frames[0] as given by the host. out may be the same buffer.
  bool encode(RenderEnv& env, const ODFrame& src, const ODFrame& out, id<MTLTexture> ingested,
              id<MTLTexture> final, OutputMode mode);
  // Strength-0 path: copy src to out without touching the working space.
  // Does nothing when they are the same buffer.
  bool encodeCopy(RenderEnv& env, const ODFrame& src, const ODFrame& out);

 private:
  id<MTLComputePipelineState> pso32_, pso16_, psoTex_, psoCopy_;
};

}  // namespace od

#endif
