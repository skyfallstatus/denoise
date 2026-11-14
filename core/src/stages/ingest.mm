// SPDX-License-Identifier: MIT
#include "ingest.h"

namespace od {

struct IngestUniforms {
  uint32_t rowStride, pad0, pad1, pad2;
};

std::unique_ptr<IngestStage> IngestStage::make(Context& ctx, std::string* err) {
  auto s = std::unique_ptr<IngestStage>(new IngestStage());
  s->pso32_ = ctx.pipeline("ingest_bgra32f", err);
  s->pso16_ = ctx.pipeline("ingest_bgra16f", err);
  s->psoTex_ = ctx.pipeline("ingest_from_texture", err);
  if (!s->pso32_ || !s->pso16_ || !s->psoTex_) return nullptr;
  return s;
}

id<MTLTexture> viewOfBuffer(id<MTLDevice> device, const ODFrame& frame, bool writable) {
  id<MTLBuffer> buf = (__bridge id<MTLBuffer>)frame.mtlBuffer;
  MTLPixelFormat fmt = hostTextureFormat(frame.format);
  NSUInteger align = [device minimumLinearTextureAlignmentForPixelFormat:fmt];
  if (frame.rowBytes % align != 0) return nil;
  if (buf.length < (NSUInteger)frame.rowBytes * frame.height) return nil;
  MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt
                                                                               width:frame.width
                                                                              height:frame.height
                                                                           mipmapped:NO];
  d.storageMode = buf.storageMode;
  d.usage = writable ? (MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite) : MTLTextureUsageShaderRead;
  return [buf newTextureWithDescriptor:d offset:0 bytesPerRow:frame.rowBytes];
}

bool IngestStage::encode(RenderEnv& env, const ODFrame& frame, id<MTLTexture> dst) {
  id<MTLBuffer> buf = (__bridge id<MTLBuffer>)frame.mtlBuffer;
  const uint32_t flags = env.params.debugFlags;

  if (flags & (OD_DEBUG_IO_BLIT | OD_DEBUG_IO_VIEW)) {
    id<MTLTexture> src = nil;
    if (flags & OD_DEBUG_IO_VIEW) {
      src = viewOfBuffer(env.device(), frame, false);
      if (!src) {
        env.fail("ingest: this buffer cannot be viewed as a linear texture (storage mode %d, rowBytes %u)",
                 (int)buf.storageMode, frame.rowBytes);
        return false;
      }
    } else {
      if (buf.length < (NSUInteger)frame.rowBytes * frame.height || frame.rowBytes % bytesPerPixel(frame.format)) {
        env.fail("ingest: blit path needs rowBytes*height bytes and rowBytes a multiple of the pixel size");
        return false;
      }
      src = env.temp(hostTextureFormat(frame.format), frame.width, frame.height);
      if (!src) return false;
      [env.blit() copyFromBuffer:buf
                    sourceOffset:0
               sourceBytesPerRow:frame.rowBytes
             sourceBytesPerImage:(NSUInteger)frame.rowBytes * frame.height
                      sourceSize:MTLSizeMake(frame.width, frame.height, 1)
                       toTexture:src
                destinationSlice:0
                destinationLevel:0
               destinationOrigin:MTLOriginMake(0, 0, 0)];
    }
    id<MTLComputeCommandEncoder> enc = env.compute();
    [enc setTexture:src atIndex:0];
    [enc setTexture:dst atIndex:1];
    env.dispatch(psoTex_, frame.width, frame.height);
    return true;
  }

  const bool half = frame.format == OD_PIXEL_BGRA16F;
  IngestUniforms u = {frame.rowBytes / (half ? 2u : 4u), 0, 0, 0};
  id<MTLComputeCommandEncoder> enc = env.compute();
  [enc setBuffer:buf offset:0 atIndex:0];
  [enc setBytes:&u length:sizeof(u) atIndex:1];
  [enc setTexture:dst atIndex:0];
  env.dispatch(half ? pso16_ : pso32_, frame.width, frame.height);
  return true;
}

}  // namespace od
