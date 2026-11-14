// SPDX-License-Identifier: MIT
#include "output.h"

#include "ingest.h"

namespace od {

struct OutputUniforms {
  uint32_t srcStride, dstStride, mode;
  float noiseGain;
};
struct CopyUniforms {
  uint32_t srcStride, dstStride, wordsPerPixel, width;
};

std::unique_ptr<OutputStage> OutputStage::make(Context& ctx, std::string* err) {
  auto s = std::unique_ptr<OutputStage>(new OutputStage());
  s->pso32_ = ctx.pipeline("output_bgra32f", err);
  s->pso16_ = ctx.pipeline("output_bgra16f", err);
  s->psoTex_ = ctx.pipeline("output_to_texture", err);
  s->psoCopy_ = ctx.pipeline("output_copy_rows", err);
  if (!s->pso32_ || !s->pso16_ || !s->psoTex_ || !s->psoCopy_) return nullptr;
  return s;
}

bool OutputStage::encode(RenderEnv& env, const ODFrame& src, const ODFrame& out, id<MTLTexture> ingested,
                         id<MTLTexture> final, OutputMode mode) {
  id<MTLBuffer> sbuf = (__bridge id<MTLBuffer>)src.mtlBuffer;
  id<MTLBuffer> dbuf = (__bridge id<MTLBuffer>)out.mtlBuffer;
  const bool half = out.format == OD_PIXEL_BGRA16F;
  const uint32_t el = half ? 2u : 4u;
  OutputUniforms u = {src.rowBytes / el, out.rowBytes / el, env.params.view == OD_VIEW_SPLIT ? 3u : (uint32_t)mode, kNoiseViewGain};
  const uint32_t flags = env.params.debugFlags;

  if (flags & (OD_DEBUG_IO_BLIT | OD_DEBUG_IO_VIEW)) {
    const bool view = flags & OD_DEBUG_IO_VIEW;
    id<MTLTexture> st = nil, dt = nil;
    if (view) {
      st = viewOfBuffer(env.device(), src, false);
      dt = viewOfBuffer(env.device(), out, true);
      if (!st || !dt) {
        env.fail("output: this buffer cannot be viewed as a linear texture");
        return false;
      }
    } else {
      if (sbuf.length < (NSUInteger)src.rowBytes * src.height || dbuf.length < (NSUInteger)out.rowBytes * out.height ||
          src.rowBytes % bytesPerPixel(src.format) || out.rowBytes % bytesPerPixel(out.format)) {
        env.fail("output: blit path needs rowBytes*height bytes and rowBytes a multiple of the pixel size");
        return false;
      }
      st = env.temp(hostTextureFormat(src.format), src.width, src.height);
      dt = env.temp(hostTextureFormat(out.format), out.width, out.height);
      if (!st || !dt) return false;
      [env.blit() copyFromBuffer:sbuf
                    sourceOffset:0
               sourceBytesPerRow:src.rowBytes
             sourceBytesPerImage:(NSUInteger)src.rowBytes * src.height
                      sourceSize:MTLSizeMake(src.width, src.height, 1)
                       toTexture:st
                destinationSlice:0
                destinationLevel:0
               destinationOrigin:MTLOriginMake(0, 0, 0)];
    }
    id<MTLComputeCommandEncoder> enc = env.compute((psoTex_).label.UTF8String);
    [enc setTexture:st atIndex:0];
    [enc setTexture:ingested atIndex:1];
    [enc setTexture:final atIndex:2];
    [enc setTexture:dt atIndex:3];
    [enc setBytes:&u length:sizeof(u) atIndex:0];
    env.dispatch(psoTex_, out.width, out.height);
    if (!view) {
      [env.blit() copyFromTexture:dt
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(out.width, out.height, 1)
                         toBuffer:dbuf
                destinationOffset:0
           destinationBytesPerRow:out.rowBytes
         destinationBytesPerImage:(NSUInteger)out.rowBytes * out.height];
    }
    return true;
  }

  id<MTLComputeCommandEncoder> enc = env.compute((half ? pso16_ : pso32_).label.UTF8String);
  [enc setBuffer:sbuf offset:0 atIndex:0];
  [enc setBuffer:dbuf offset:0 atIndex:1];
  [enc setBytes:&u length:sizeof(u) atIndex:2];
  [enc setTexture:ingested atIndex:0];
  [enc setTexture:final atIndex:1];
  env.dispatch(half ? pso16_ : pso32_, out.width, out.height);
  return true;
}

bool OutputStage::encodeCopy(RenderEnv& env, const ODFrame& src, const ODFrame& out) {
  if (src.mtlBuffer == out.mtlBuffer) return true;
  id<MTLBuffer> sbuf = (__bridge id<MTLBuffer>)src.mtlBuffer;
  id<MTLBuffer> dbuf = (__bridge id<MTLBuffer>)out.mtlBuffer;
  const uint32_t bpp = bytesPerPixel(out.format);
  if (src.rowBytes == out.rowBytes && src.rowBytes == src.width * bpp) {
    // Tightly packed on both sides: one blit. With row padding the kernel
    // below is used instead so the destination's padding bytes stay untouched.
    NSUInteger n = (NSUInteger)src.rowBytes * src.height;
    [env.blit() copyFromBuffer:sbuf sourceOffset:0 toBuffer:dbuf destinationOffset:0 size:n];
    return true;
  }
  CopyUniforms u = {src.rowBytes / 4u, out.rowBytes / 4u, bpp / 4u, out.width};
  id<MTLComputeCommandEncoder> enc = env.compute((psoCopy_).label.UTF8String);
  [enc setBuffer:sbuf offset:0 atIndex:0];
  [enc setBuffer:dbuf offset:0 atIndex:1];
  [enc setBytes:&u length:sizeof(u) atIndex:2];
  env.dispatch(psoCopy_, out.width, out.height);
  return true;
}

}  // namespace od
