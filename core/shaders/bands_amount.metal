// SPDX-License-Identifier: MIT
// Bands step (core/src/stages/bands.mm): the editor's amount per frequency
// band (ODParams.bandAmount, Luma detail / Chroma detail "amount"), applied
// once to what the denoising stages removed, after the last of them and
// before sharpen. It runs only when an amount is below 100 %.
//
// Pyramid ("tent", texel-centred), the same as the spatial stage's
// (spatial_nlm.metal), so the octaves are the ones its levels filter:
//   down  D : [1 3 3 1]/8 separable, decimate by 2 (4 bilinear fetches)
//   up    U : bilinear (1 fetch)
// With d = image - orig (the removed signal, negated):
//   G_1 = D(d),  G_(k+1) = D(G_k)                         bands_down_first, bands_down
//   L_0 = d - U(G_1),  L_k = G_k - U(G_(k+1))            profile bands 0, 1, 2
//   G_K (K = 3, or the top of a frame too small for 3)    band 3 and everything coarser
//   R_K = wTop G_K,  R_k = w_k L_k + U(R_(k+1))           bands_collapse, k = K-1 .. 1
//   out = orig + w_0 L_0 + U(R_1),  alpha from image      bands_final
// w_k = (luma amount, chroma amount, chroma amount) of band k; wTop that of
// band 3. With every weight 1 this is image again (U is linear). A channel
// group whose weights are all 1 is copied from image instead, bit for bit.
//
// math: safe (the default), like every unit whose output must be exact.

struct BandsU {
  float4 w;     // xyz: weights (Y, Cb, Cr) of this level's octave
  float4 wTop;  // xyz: weights of the top level (band 3 and coarser)
  float4 grain; // global luma amount, full-band-equivalent retained fine grain
  float sigma[OD_LEVELS];
  uint4 flags;  // x = 1: the next coarser level is the top (its R is wTop G); y = 1: luma from image;
                // z = 1: chroma from image; w = 1: no coarser level at all (out = orig + wTop d)
};

constexpr sampler bandsLinear(coord::pixel, address::clamp_to_edge, filter::linear);

static inline float4 bands_down4(texture2d<float, access::sample> src, uint2 gid) {
  const float2 c = float2(gid) * 2.0f + 1.0f;  // centre of the 2x2 block, in src pixels
  return 0.25f * (src.sample(bandsLinear, c + float2(-0.75f, -0.75f)) +
                  src.sample(bandsLinear, c + float2(0.75f, -0.75f)) +
                  src.sample(bandsLinear, c + float2(-0.75f, 0.75f)) +
                  src.sample(bandsLinear, c + float2(0.75f, 0.75f)));
}

// G_1 = D(image) - D(orig) = D(image - orig).
kernel void bands_down_first(texture2d<float, access::sample> image [[texture(0)]],
                             texture2d<float, access::sample> orig [[texture(1)]],
                             texture2d<float, access::write> dst [[texture(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
  const float4 d = bands_down4(image, gid) - bands_down4(orig, gid);
  dst.write(od_store_half(float4(d.xyz, 0.0f)), gid);
}

kernel void bands_down(texture2d<float, access::sample> src [[texture(0)]],
                       texture2d<float, access::write> dst [[texture(1)]],
                       uint2 gid [[thread_position_in_grid]]) {
  dst.write(od_store_half(bands_down4(src, gid)), gid);
}

// R_k = w_k (G_k - U(G_(k+1))) + U(R_(k+1)).
kernel void bands_collapse(texture2d<float, access::read> g [[texture(0)]],
                           texture2d<float, access::sample> gUp [[texture(1)]],
                           texture2d<float, access::sample> rUp [[texture(2)]],
                           texture2d<float, access::write> dst [[texture(3)]],
                           constant BandsU& u [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
  const float2 pc = (float2(gid) + 0.5f) * 0.5f;
  const float3 gu = gUp.sample(bandsLinear, pc).xyz;
  const float3 ru = u.flags.x != 0u ? u.wTop.xyz * gu : rUp.sample(bandsLinear, pc).xyz;
  const float3 r = u.w.xyz * (g.read(gid).xyz - gu) + ru;
  dst.write(od_store_half(float4(r, 0.0f)), gid);
}

// out = orig + w_0 (d - U(G_1)) + U(R_1), alpha from image (the spatial
// stage leaves 1/sqrt(count) there for sharpen).
kernel void bands_final(texture2d<float, access::read> image [[texture(0)]],
                        texture2d<float, access::read> orig [[texture(1)]],
                        texture2d<float, access::sample> g1 [[texture(2)]],
                        texture2d<float, access::sample> r1 [[texture(3)]],
                        texture2d<float, access::write> dst [[texture(4)]],
                        constant BandsU& u [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
  const float4 im = image.read(gid);
  const float3 o = orig.read(gid).xyz;
  const float3 d = im.xyz - o;
  float3 w = u.w.xyz;
  float fineRatio = 1.0f;
  if (u.grain.x > 0.0f && u.grain.y > 0.0f) {
    // d already includes the global amount. Cap only L0, leaving U(R1)
    // controlled by Luma throughout its range. The floor keeps its existing
    // full-band-equivalent calibration; it no longer protects coarse mottle.
    const float t = clamp(o.x, 0.0f, 1.0f) * float(OD_LEVELS - 1);
    const int i = min(int(t), OD_LEVELS - 2);
    const float sigma = mix(u.sigma[i], u.sigma[i + 1], t - float(i));
    fineRatio = od_luma_amount(u.grain.x, u.grain.y, sigma) / u.grain.x;
    w.x *= fineRatio;
  }
  float3 v;
  if (u.flags.w != 0u) {
    // No pyramid fits: retain the existing top-band amount convention.
    v = o + float3(u.wTop.x * fineRatio, u.wTop.yz) * d;
  } else {
    const float2 pc = (float2(gid) + 0.5f) * 0.5f;
    const float3 gu = g1.sample(bandsLinear, pc).xyz;
    const float3 ru = u.flags.x != 0u ? u.wTop.xyz * gu : r1.sample(bandsLinear, pc).xyz;
    v = o + w * (d - gu) + ru;
  }
  const float y = u.flags.y != 0u ? im.x : v.x;
  const float2 c = u.flags.z != 0u ? im.yz : v.yz;
  dst.write(od_store_half(float4(y, c, im.w)), gid);
}
