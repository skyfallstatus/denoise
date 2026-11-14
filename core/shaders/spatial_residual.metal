// SPDX-License-Identifier: MIT
// Measure the merge on the SAME tent bands consumed by spatial
// NLM. Estimate residual noise from the correction D = raw - merged. In flat tiles, its
// energy earns a reduction in the spatial threshold; a nominal count alone does not. The
// independent count is a conservative lower bound, and the fallback where structure
// dominates (2..6 times the profile variance), to prevent excess texture smoothing. The
// profile is not used to cap the measured energy: that under-credits remaining noise when
// the profile is low. This is a guarded estimate, not an unbiased noise oracle: the identity
// sigma_T^2 = sigma_R^2 - E[D^2] requires an appropriate unbiased average. No output
// recursion, CPU readback, or dependence on render order. od: math=safe
constexpr sampler residualLinear(coord::pixel, address::clamp_to_edge, filter::linear);
struct ResidualU {
  float4 sigma[OD_LEVELS];
  uint4 flags;
};
kernel void spatial_residual_ratio(texture2d<float, access::read> raw [[texture(0)]],
                                   texture2d<float, access::sample> rawUp [[texture(1)]],
                                   texture2d<float, access::read> merged [[texture(2)]],
                                   texture2d<float, access::sample> mergedUp [[texture(3)]],
                                   texture2d<float, access::write> ratio [[texture(4)]],
                                   texture2d<float, access::sample> count [[texture(5)]],
                                   constant ResidualU& u [[buffer(0)]],
                                   uint2 gid [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
  const bool top = u.flags.x != 0;
  // Fetch each pixel in parallel, but reduce in the reference's row-major
  // order. Products stay in the ordered reduction to preserve multiply-add rounding.
  // 64 lanes, one 8x8 tile per group; partial edge tiles contain zero terms.
  threadgroup float3 rawSum[64], deltaSum[64], noise[64];
  threadgroup float inverseCount[64];
  uint2 p = gid * 8 + uint2(tid % 8, tid / 8);
  float3 r = 0, d = 0, snPixel = 0;
  float inv = 0;
  if (p.x < raw.get_width() && p.y < raw.get_height()) {
    r = raw.read(p).xyz;
    float3 t = merged.read(p).xyz;
    float knot = clamp(r.x, 0.0f, 1.0f) * float(OD_LEVELS - 1);
    int k = min(int(knot), OD_LEVELS - 2);
    float3 sigma = mix(u.sigma[k].xyz, u.sigma[k + 1].xyz, knot - float(k));
    snPixel = sigma;
    float2 cp = (float2(p) + 0.5f) / float2(raw.get_width(), raw.get_height()) *
                float2(count.get_width(), count.get_height());
    inv = 1.0f / max(count.sample(residualLinear, cp).x, 1.0f);
    if (!top) {
      float2 uv = (float2(p) + 0.5f) * 0.5f;
      r -= rawUp.sample(residualLinear, uv).xyz;
      t -= mergedUp.sample(residualLinear, uv).xyz;
    }
    d = r - t;
  }
  rawSum[tid] = r;
  deltaSum[tid] = d;
  noise[tid] = snPixel;
  inverseCount[tid] = inv;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid != 0) return;
  float3 er = 0, et = 0, sr = 0, st = 0, sn = 0;
  float rho2 = 0;
  const uint n = min(8u, raw.get_width() - gid.x * 8) * min(8u, raw.get_height() - gid.y * 8);
  for (uint i = 0; i < 64; ++i) {
    float3 r = rawSum[i], d = deltaSum[i], sigma = noise[i];
    er += r * r; et += d * d; sr += r; st += d;
    sn += sigma * sigma; rho2 += inverseCount[i];
  }
  if (top) {
    er = max(er - sr * sr / float(max(n, 1u)), 0.0f);
    et = max(et - st * st / float(max(n, 1u)), 0.0f);
  }
  // Never claim more reduction than the nominal independent-count bound.
  // Strong signal is not a reliable local noise sample: preserve its detail.
  float floorVariance = rho2 / float(max(n, 1u));
  float3 variance = clamp(1.0f - et / (er + 1e-12f), floorVariance, 1.0f);
  float3 structure = smoothstep(2.0f, 6.0f, er / (sn + 1e-12f));
  float3 factor = sqrt(mix(variance, float3(floorVariance), structure));
  ratio.write(float4(factor, 1), gid);
}

kernel void spatial_temporal_blend(texture2d<float, access::read> base [[texture(0)]],
                                   texture2d<float, access::read> temporal [[texture(1)]],
                                   texture2d<float, access::write> dst [[texture(2)]],
                                   constant float& strength [[buffer(0)]],
                                   uint2 gid [[thread_position_in_grid]]) {
  float4 a = base.read(gid), b = temporal.read(gid);
  dst.write(od_store_half(a + strength * (b - a)), gid);
}
