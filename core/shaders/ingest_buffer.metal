// SPDX-License-Identifier: MIT
// ingest: host BGRA buffer (32f or 16f, with a row stride) -> working texture
// (rgba16Float Y, Cb, Cr, A). Math stays `safe`: the NaN scrub below needs
// isnan() to mean something.
// od: math=safe

struct IngestUniforms {
  uint rowStride;  // distance between rows, in elements (floats or halfs)
  uint pad0, pad1, pad2;
};

static inline float4 ingest_convert(float4 bgra) {
  float3 ycc = od_rgb_to_ycc(bgra.zyx);
  float4 v = float4(ycc, bgra.w);
  // Stages must never see NaN or Inf. The output stage adds a delta to the
  // original pixel, so whatever is scrubbed here still round-trips.
  v = select(v, float4(0.0f), isnan(v));
  return od_round_half(clamp(v, -OD_HALF_MAX, OD_HALF_MAX));
}

kernel void ingest_bgra32f(const device float* src [[buffer(0)]],
                           constant IngestUniforms& u [[buffer(1)]],
                           texture2d<float, access::write> dst [[texture(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
  const device packed_float4* p = (const device packed_float4*)(src + gid.y * u.rowStride + gid.x * 4u);
  dst.write(ingest_convert(float4(*p)), gid);
}

kernel void ingest_bgra16f(const device half* src [[buffer(0)]],
                           constant IngestUniforms& u [[buffer(1)]],
                           texture2d<float, access::write> dst [[texture(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
  const device packed_half4* p = (const device packed_half4*)(src + gid.y * u.rowStride + gid.x * 4u);
  dst.write(ingest_convert(float4(half4(*p))), gid);
}

// Measurement variants (OD_DEBUG_IO_BLIT / OD_DEBUG_IO_VIEW): the buffer
// is first blitted into, or viewed as, an RGBA float texture whose channels
// are B,G,R,A in that order.
kernel void ingest_from_texture(texture2d<float, access::read> src [[texture(0)]],
                                texture2d<float, access::write> dst [[texture(1)]],
                                uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
  dst.write(ingest_convert(src.read(gid)), gid);
}
