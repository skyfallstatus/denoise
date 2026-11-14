// SPDX-License-Identifier: MIT
// ingest stage: host BGRA MTLBuffer -> working texture (RGBA16Float Y,Cb,Cr,A).
#ifndef OD_INGEST_H
#define OD_INGEST_H

#include "stage.h"

namespace od {

class IngestStage {
 public:
  static std::unique_ptr<IngestStage> make(Context& ctx, std::string* err);
  // Appends the conversion of `frame` into `dst` (working format, frame size). The default
  // path is one compute kernel that reads the buffer directly. OD_DEBUG_IO_BLIT and
  // OD_DEBUG_IO_VIEW in env.params.debugFlags select the alternatives.
  bool encode(RenderEnv& env, const ODFrame& frame, id<MTLTexture> dst);

 private:
  id<MTLComputePipelineState> pso32_, pso16_, psoTex_;
};

// Shared with output: bytes per pixel and the RGBA float format that matches
// the host layout (channels hold B,G,R,A).
inline uint32_t bytesPerPixel(ODPixelFormat f) { return f == OD_PIXEL_BGRA16F ? 8u : 16u; }
inline MTLPixelFormat hostTextureFormat(ODPixelFormat f) {
  return f == OD_PIXEL_BGRA16F ? MTLPixelFormatRGBA16Float : MTLPixelFormatRGBA32Float;
}
// Linear texture view over a host buffer; nil if Metal refuses.
id<MTLTexture> viewOfBuffer(id<MTLDevice> device, const ODFrame& frame, bool writable);

}  // namespace od

#endif
