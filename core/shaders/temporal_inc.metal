// SPDX-License-Identifier: MIT
// Shared definitions of the temporal stage (included by
// temporal_motion.metal and temporal_merge.metal).
//
// Coordinates. Every sampler here uses PIXEL coordinates: texel (i, j) has its
// centre at (i + 0.5, j + 0.5). The tent pyramid is texel-centred: coarse
// texel c covers fine texels 2c and 2c + 1, so a fine position p maps to the
// coarse position p / 2.

constexpr sampler tLin(coord::pixel, address::clamp_to_edge, filter::linear);

// Motion vectors and gates of all neighbours are stacked in one texture:
// neighbour slot a (0 = nearest) owns the rows [a * rows, (a + 1) * rows).
#define T_MAX_NEIGHBOURS 10  // 2 * OD_MAX_RADIUS

// Bilinear read at position q (pixels) of neighbour a's part of such an
// atlas, with the edge clamp a texture of its own would have: a coordinate
// clamped to the outermost texel centres reads exactly those texels.
static inline float4 t_atlas(texture2d<float, access::sample> t, float2 q, uint a, uint rows) {
  const float2 hi = float2(float(t.get_width()), float(rows)) - 0.5f;
  return t.sample(tLin, clamp(q, float2(0.5f), hi) + float2(0.0f, float(a * rows)));
}

// Floor added in quadrature to every sigma used in a division (half-float
// quantisation, clipped or synthetic footage).
#define T_SIGMA_Q2 (2.0e-4f * 2.0e-4f)

// Tent pyramid, down: separable [1 3 3 1]/8 then decimate by 2, as 4 bilinear
// fetches at (+-0.75, +-0.75) fine texels around the coarse texel centre.
static inline float4 t_down(texture2d<float, access::sample> fine, uint2 gid) {
  float2 c = float2(gid) * 2.0f + 1.0f;
  return 0.25f * (fine.sample(tLin, c + float2(-0.75f, -0.75f)) + fine.sample(tLin, c + float2(0.75f, -0.75f)) +
                  fine.sample(tLin, c + float2(-0.75f, 0.75f)) + fine.sample(tLin, c + float2(0.75f, 0.75f)));
}

// Tent pyramid, up: bilinear interpolation of the coarse level at the centre
// of fine texel gid.
static inline float4 t_up(texture2d<float, access::sample> coarse, uint2 gid) {
  return coarse.sample(tLin, (float2(gid) + 0.5f) * 0.5f);
}

// Sigma of (Y, Cb, Cr) at luminance y from 8 knots (xyz of each float4).
static inline float3 t_sigma(constant float4* knots, float y) {
  float t = clamp(y, 0.0f, 1.0f) * 7.0f;
  int i = min(int(t), 6);
  return mix(knots[i].xyz, knots[i + 1].xyz, t - float(i));
}
