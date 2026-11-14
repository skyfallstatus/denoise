// SPDX-License-Identifier: MIT
// Open Denoise shared shader header. scripts/embed_shaders.py puts this file in front
// of every shader unit, so everything here is available everywhere. Keep it
// small: only things more than one stage needs.
#include <metal_stdlib>
using namespace metal;

// ---- Working space -------------------------------------------------------
// Working textures are rgba16Float holding (Y, Cb, Cr, A).
//   Y  = 0.2126 R + 0.7152 G + 0.0722 B          (BT.709 luma weights)
//   Cb = (B - Y) / 1.8556                        (zero-centred)
//   Cr = (R - Y) / 1.5748                        (zero-centred)
// Values are whatever the host supplies (not re-linearised) and are NOT
// clamped to [0,1]: negatives and super-whites are legal. Ingest clamps to the
// finite half range and replaces NaN by 0, so stages never see NaN/Inf.
#define OD_HALF_MAX 65504.0f

static inline float3 od_rgb_to_ycc(float3 rgb) {
  float y = dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));
  return float3(y, (rgb.b - y) * (1.0f / 1.8556f), (rgb.r - y) * (1.0f / 1.5748f));
}

static inline float3 od_ycc_to_rgb(float3 ycc) {
  float r = ycc.x + 1.5748f * ycc.z;
  float b = ycc.x + 1.8556f * ycc.y;
  float g = (ycc.x - 0.2126f * r - 0.0722f * b) * (1.0f / 0.7152f);
  return float3(r, g, b);
}

// Writing a float to an rgba16Float texture TRUNCATES toward zero on Apple
// GPUs (measured on M5 Max: errors up to 1 half ulp, always towards 0), which
// would darken the picture a little with every pass. Converting in the shader
// rounds to nearest. Wrap the value of every write to a half texture that
// carries image data:   dst.write(od_round_half(v), gid);
static inline float4 od_round_half(float4 v) { return float4(half4(v)); }

// The same for values that can leave the half range: a difference or a sum
// of two working values (up to twice OD_HALF_MAX) would round to infinity,
// and infinity minus infinity in the next pass is NaN, which every pyramid
// spreads over the whole frame. Clamped first, they stay finite.
static inline float4 od_store_half(float4 v) { return float4(half4(clamp(v, -OD_HALF_MAX, OD_HALF_MAX))); }

// Clamped texel fetch (edge replicate) for integer coordinates.
static inline float4 od_load(texture2d<float, access::read> t, int2 p) {
  int2 hi = int2(int(t.get_width()) - 1, int(t.get_height()) - 1);
  return t.read(uint2(clamp(p, int2(0), hi)));
}

// Luma amount after the grain floor (stage.h, kGrainFloor): never remove so
// much luma noise that less than `keep` of it is left. sigmaFull is the
// profile's full-band luma sigma at the pixel's luminance. Shared by every
// stage that applies or follows the amounts, so they cannot drift apart.
static inline float od_luma_amount(float amount, float keep, float sigmaFull) {
  return min(amount, 1.0f - min(keep / max(sigmaFull, 1e-7f), 1.0f));
}

// ODNoiseProfile luminance lookup: OD_LEVELS knots uniform over Y in
// [0,1]; linear interpolation, clamped outside. `knots` points at 8 floats.
#define OD_LEVELS 8
#define OD_BANDS 4
static inline float od_sigma_at(const float knots[OD_LEVELS], float y) {
  float t = clamp(y, 0.0f, 1.0f) * float(OD_LEVELS - 1);
  int i = min(int(t), OD_LEVELS - 2);
  return mix(knots[i], knots[i + 1], t - float(i));
}
