// SPDX-License-Identifier: MIT
// Scalar calibration reference from the initial API v11 implementation.
// Retained as a test oracle; never included in the production library.
#pragma once
static const char* kResidualReference = R"METAL(
#include <metal_stdlib>
using namespace metal;
#define OD_LEVELS 8
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
                                   uint2 gid [[thread_position_in_grid]]) {
  const bool top = u.flags.x != 0;
  float3 er = 0, et = 0, sr = 0, st = 0, sn = 0;
  float rho2 = 0;
  uint n = 0;
  for (uint y = 0; y < 8; ++y) for (uint x = 0; x < 8; ++x) {
    uint2 p = gid * 8 + uint2(x, y);
    if (p.x >= raw.get_width() || p.y >= raw.get_height()) continue;
    float3 r = raw.read(p).xyz, t = merged.read(p).xyz;
    float knot = clamp(r.x, 0.0f, 1.0f) * float(OD_LEVELS - 1);
    int k = min(int(knot), OD_LEVELS - 2);
    float3 sigma = mix(u.sigma[k].xyz, u.sigma[k + 1].xyz, knot - float(k));
    sn += sigma * sigma;
    float2 cp = (float2(p) + 0.5f) / float2(raw.get_width(), raw.get_height()) *
                float2(count.get_width(), count.get_height());
    rho2 += 1.0f / max(count.sample(residualLinear, cp).x, 1.0f);
    if (!top) {
      float2 uv = (float2(p) + 0.5f) * 0.5f;
      r -= rawUp.sample(residualLinear, uv).xyz;
      t -= mergedUp.sample(residualLinear, uv).xyz;
    }
    float3 d = r - t;
    er += r * r; et += d * d; sr += r; st += d; ++n;
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

)METAL";
