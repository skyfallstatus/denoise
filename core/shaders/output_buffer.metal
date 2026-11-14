// SPDX-License-Identifier: MIT
// output: working texture -> host BGRA buffer.
//
// The output is written as   out = src + toRGB(final - ingested)
// where `src` is the original pixel of frames[0] read straight from the host
// buffer, `ingested` is the working texture ingest made from it and `final` is
// what the last stage produced. Consequences:
//  * a pixel no stage changed comes out bit-identical to the input (the
//    kernel copies it), at 32f as well as 16f, negatives, super-whites, NaN
//    and values beyond the half range included;
//  * so does a pixel whose source colour is not finite (NaN or Inf in any of
//    B, G, R), in the normal view: ingest scrubbed it for the stages, so its
//    delta means nothing (an Inf green would turn the other channels into
//    NaN or +-65504);
//  * the working space can stay rgba16Float for 32f hosts: half rounding of
//    the working copy cancels in the difference, only the correction itself
//    is computed at half precision;
//  * alpha is always the source alpha, bit for bit (NaN and Inf included), in
//    every view and both formats.
// Math must stay `safe`: with fast math the compiler may reassociate the
// delta and the identity would be lost.
// od: math=safe

struct OutputUniforms {
  uint srcStride;  // elements (floats or halfs) between rows of the source
  uint dstStride;  // same for the destination
  uint mode;       // 0 = delta (normal), 1 = noise view, 2 = absolute (final only)
  float noiseGain; // amplification of the removed noise in mode 1
};

static inline float4 output_pixel(float4 srcBGRA, float4 w0, float4 w1, uint mode, float gain) {
  if (mode == 2u) {
    return float4(od_ycc_to_rgb(w1.xyz).zyx, srcBGRA.w);
  }
  float3 d = w1.xyz - w0.xyz;
  if (mode == 1u) {
    return float4((0.5f - gain * od_ycc_to_rgb(d)).zyx, srcBGRA.w);
  }
  if (all(d == float3(0.0f)) || any(isnan(srcBGRA.xyz) || isinf(srcBGRA.xyz))) return srcBGRA;
  return float4(srcBGRA.xyz + od_ycc_to_rgb(d).zyx, srcBGRA.w);
}

kernel void output_bgra32f(const device float* src [[buffer(0)]],
                           device float* dst [[buffer(1)]],
                           constant OutputUniforms& u [[buffer(2)]],
                           texture2d<float, access::read> ingested [[texture(0)]],
                           texture2d<float, access::read> final [[texture(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= final.get_width() || gid.y >= final.get_height()) return;
  const device packed_float4* s = (const device packed_float4*)(src + gid.y * u.srcStride + gid.x * 4u);
  if (u.mode == 3u && gid.x <= final.get_width()/2) {
    device packed_float4* d = (device packed_float4*)(dst + gid.y*u.dstStride + gid.x*4u);
    float4 v=float4(*s);if(gid.x==final.get_width()/2)v.xyz=float3(.65f);*d=packed_float4(v);return;
  }
  float4 o = output_pixel(float4(*s), ingested.read(gid), final.read(gid), u.mode, u.noiseGain);
  device packed_float4* d = (device packed_float4*)(dst + gid.y * u.dstStride + gid.x * 4u);
  *d = packed_float4(o);
}

kernel void output_bgra16f(const device half* src [[buffer(0)]],
                           device half* dst [[buffer(1)]],
                           constant OutputUniforms& u [[buffer(2)]],
                           texture2d<float, access::read> ingested [[texture(0)]],
                           texture2d<float, access::read> final [[texture(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= final.get_width() || gid.y >= final.get_height()) return;
  const device packed_half4* s = (const device packed_half4*)(src + gid.y * u.srcStride + gid.x * 4u);
  half4 sh = half4(*s);
  float4 w0 = ingested.read(gid), w1 = final.read(gid);
  device packed_half4* d = (device packed_half4*)(dst + gid.y * u.dstStride + gid.x * 4u);
  if (u.mode == 3u && gid.x <= final.get_width()/2) {
    if(gid.x==final.get_width()/2)sh.xyz=half3(.65h);*d=packed_half4(sh);return;
  }
  if ((u.mode == 0u || u.mode == 3u) && (all(w1.xyz == w0.xyz) || any(isnan(sh.xyz) || isinf(sh.xyz)))) {
    *d = packed_half4(sh);  // untouched (or non-finite) pixel: copy the bits
    return;
  }
  float4 o = output_pixel(float4(sh), w0, w1, u.mode, u.noiseGain);
  // The colour is clamped (a finite source plus a finite delta can overflow a
  // half); alpha keeps the source's bits.
  half4 oh = half4(float4(clamp(o.xyz, -OD_HALF_MAX, OD_HALF_MAX), 0.0f));
  oh.w = sh.w;
  *d = packed_half4(oh);
}

// Measurement variants (OD_DEBUG_IO_BLIT / OD_DEBUG_IO_VIEW): source and
// destination are RGBA float textures holding B,G,R,A.
kernel void output_to_texture(texture2d<float, access::read> src [[texture(0)]],
                              texture2d<float, access::read> ingested [[texture(1)]],
                              texture2d<float, access::read> final [[texture(2)]],
                              texture2d<float, access::write> dst [[texture(3)]],
                              constant OutputUniforms& u [[buffer(0)]],
                              uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= final.get_width() || gid.y >= final.get_height()) return;
  if(u.mode==3u && gid.x<=final.get_width()/2){float4 v=src.read(gid);if(gid.x==final.get_width()/2)v.xyz=float3(.65f);dst.write(v,gid);return;}
  dst.write(output_pixel(src.read(gid), ingested.read(gid), final.read(gid), u.mode, u.noiseGain), gid);
}

// Strength-0 fast path when out is a different buffer with a different row
// stride: plain row-aware copy, 32-bit words (both formats are multiples).
struct CopyUniforms {
  uint srcStride;  // in 32-bit words
  uint dstStride;
  uint wordsPerPixel;  // 4 (32f) or 2 (16f)
  uint width;
};
kernel void output_copy_rows(const device uint* src [[buffer(0)]],
                             device uint* dst [[buffer(1)]],
                             constant CopyUniforms& u [[buffer(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= u.width) return;
  uint si = gid.y * u.srcStride + gid.x * u.wordsPerPixel;
  uint di = gid.y * u.dstStride + gid.x * u.wordsPerPixel;
  for (uint k = 0; k < u.wordsPerPixel; ++k) dst[di + k] = src[si + k];
}
