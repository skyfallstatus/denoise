// SPDX-License-Identifier: MIT
// Spatial stage: coarse-to-fine pyramid NL-means with luma-
// guided chroma. Implementation: core/src/stages/spatial.mm (header
// comment).
//
// Pyramid ("tent", texel-centred): a coarse texel covers 2x2 fine texels.
//   down  D : [1 3 3 1]/8 separable, decimate by 2 (4 bilinear fetches)
//   up    U : bilinear (1 fetch)
// Per level k, coarse to fine:
//   X_k  = G_k + U(Dn_(k+1) - G_(k+1))        spatial_prepare
//   Dn_k = NLM(X_k)                           spatial_nlm5 / spatial_nlm3
// The alpha channel of the stage's own textures carries rho = 1/sqrt(count),
// the factor by which the temporal stage already reduced the noise there, and
// so does the alpha of the stage's result, for the sharpen stage.
//
// od: math=fast
// (fast: the stage measured 24 % faster than with safe at 3840x2160, with
// the same PSNR to 0.01 dB; every value here is finite and clamped before it
// is squared, and the "amount 0" pass-through is a select, not arithmetic.)

// Everything a pass needs. Mirrors SpatialUniforms in spatial.mm.
struct SpatialU {
  // Per luminance knot: xyz = noise sigma of THIS pyramid level for Y, Cb, Cr
  // (profile x band conversion x noiseLevel x strength); w = unscaled
  // full-band luma sigma, used by the grain floor on the final level.
  float4 sig[OD_LEVELS];
  float4 amount;  // x = luma amount, y = chroma amount, z = grain-floor sigma,
                  // w = upper end of the outlier range for a pixel darker than its neighbours (normalised)
  float4 shape;   // x = 1/ETA luma, y = 1/ETA chroma, z = luma weight in the chroma guide, w = knee q0
  float4 robust;  // x = share of rho the luma threshold follows on this level (1 = all of it),
                  // y, z = outlier range in normalised units, w = cap of one pixel's term in q (luma)
  uint4 flags;    // x = 1: final level (blend against the original), y = filter luma, z = filter chroma,
                  // w = 1: the count map is the temporal stage's own (final level: rho goes into the result's alpha)
  float4 research; // x: soft speck option, y: diagnostic selector (test aid only)
};

#define SPATIAL_SIGMA_Q 2e-4f

constexpr sampler spatialNearest(coord::pixel, address::clamp_to_edge, filter::nearest);
constexpr sampler spatialLinear(coord::pixel, address::clamp_to_edge, filter::linear);
constexpr sampler spatialLinearN(coord::normalized, address::clamp_to_edge, filter::linear);

static inline float4 spatial_sigma(constant SpatialU& u, float y) {
  float t = clamp(y, 0.0f, 1.0f) * float(OD_LEVELS - 1);
  int i = min(int(t), OD_LEVELS - 2);
  return mix(u.sig[i], u.sig[i + 1], t - float(i));
}

// ---- pyramid ---------------------------------------------------------------

static inline float4 spatial_down4(texture2d<float, access::sample> src, uint2 gid) {
  float2 c = float2(gid) * 2.0f + 1.0f;  // centre of the 2x2 block, in src pixels
  return 0.25f * (src.sample(spatialLinear, c + float2(-0.75f, -0.75f)) +
                  src.sample(spatialLinear, c + float2(0.75f, -0.75f)) +
                  src.sample(spatialLinear, c + float2(-0.75f, 0.75f)) +
                  src.sample(spatialLinear, c + float2(0.75f, 0.75f)));
}

// G_1 from the input image; alpha = rho_1 from the count map (a 1x1 texture
// holding 1 when the temporal stage gave none).
kernel void spatial_down_first(texture2d<float, access::sample> src [[texture(0)]],
                               texture2d<float, access::sample> count [[texture(1)]],
                               texture2d<float, access::write> dst [[texture(2)]],
                               uint2 gid [[thread_position_in_grid]]) {
  float4 s = spatial_down4(src, gid);
  float2 uv = (float2(gid) * 2.0f + 1.0f) / float2(src.get_width(), src.get_height());
  float n = max(count.sample(spatialLinearN, uv).x, 1.0f);
  s.w = rsqrt(n);
  dst.write(od_round_half(s), gid);
}

// G_(k+1) from G_k, rho averaged down with the picture.
kernel void spatial_down(texture2d<float, access::sample> src [[texture(0)]],
                         texture2d<float, access::write> dst [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
  dst.write(od_round_half(spatial_down4(src, gid)), gid);
}

// spatial_down_first, plus T_1 = D(original - merged), the chroma the
// temporal merge took off, on the same lattice (xy of an rg16f target; the
// coarser T_k come from spatial_down). One pass instead of two: the fetches
// of the merged frame are shared.
kernel void spatial_down_first_t(texture2d<float, access::sample> src [[texture(0)]],
                                 texture2d<float, access::sample> count [[texture(1)]],
                                 texture2d<float, access::write> dst [[texture(2)]],
                                 texture2d<float, access::sample> orig [[texture(3)]],
                                 texture2d<float, access::write> dstT [[texture(4)]],
                                 uint2 gid [[thread_position_in_grid]]) {
  float4 s = spatial_down4(src, gid);
  const float4 o = spatial_down4(orig, gid);
  dstT.write(od_store_half(float4(o.yz - s.yz, 0.0f, 0.0f)), gid);
  float2 uv = (float2(gid) * 2.0f + 1.0f) / float2(src.get_width(), src.get_height());
  float n = max(count.sample(spatialLinearN, uv).x, 1.0f);
  s.w = rsqrt(n);
  dst.write(od_round_half(s), gid);
}

// Chroma noise that persists from frame to frame (in-camera noise
// reduction, codecs), as a share of the noise variance, measured on one level
// m: rho = 1/sqrt(count) says how many frames the merge averaged, not whether
// their noise was independent. With E = local mean (3x3) of the squared
// level-m detail of T (T_m - U(T_(m+1)), or T_m itself on the top level), a
// merge of N frames of independent noise gives E = sigma^2 (1 - rho^2), and
// of persistent noise E = 0, so
//   persist = 1 - E / (sigma^2 (1 - rho^2)),  clamped to 0..1,
// and the noise left after the merge is sigma^2 (persist + (1 - persist) rho^2).
// Where the merge took (almost) nothing (rho near 1) the share is not
// measurable and is set to 0; the threshold is then the full sigma anyway.
struct SpatialPersistU {
  float4 sig[OD_LEVELS];  // yz = chroma noise sigma of level m at each luminance knot (not threshold-scaled)
  float4 gate;            // x, y: chroma structure from x to y times the noise variance turns the share off
  uint4 flags;            // x = 1: level m is the top level
};

kernel void spatial_chroma_persist(texture2d<float, access::sample> t [[texture(0)]],
                                   texture2d<float, access::sample> tCoarse [[texture(1)]],
                                   texture2d<float, access::sample> g [[texture(2)]],
                                   texture2d<float, access::sample> gCoarse [[texture(3)]],
                                   texture2d<float, access::write> dst [[texture(4)]],
                                   constant SpatialPersistU& u [[buffer(0)]],
                                   uint2 gid [[thread_position_in_grid]]) {
  const float2 c = float2(gid) + 0.5f;
  float2 e = 0.0f, eo = 0.0f;  // energy of the merge's correction, of the original frame (level-m detail)
  #pragma clang loop unroll(full)
  for (int y = -1; y <= 1; ++y) {
    #pragma clang loop unroll(full)
    for (int x = -1; x <= 1; ++x) {
      const float2 p = c + float2(x, y);
      float2 v = t.sample(spatialNearest, p).xy;
      float2 o = v + g.sample(spatialNearest, p).yz;
      if (u.flags.x == 0) {
        const float2 tc = tCoarse.sample(spatialLinear, p * 0.5f).xy;
        v -= tc;
        o -= tc + gCoarse.sample(spatialLinear, p * 0.5f).yz;
      }
      e += v * v;
      eo += o * o;
    }
  }
  e *= 1.0f / 9.0f;
  eo *= 1.0f / 9.0f;
  const float4 x = g.sample(spatialNearest, c);
  const float rho = x.w;
  float tk = clamp(x.x, 0.0f, 1.0f) * float(OD_LEVELS - 1);
  int i = min(int(tk), OD_LEVELS - 2);
  const float2 sn = max(mix(u.sig[i].yz, u.sig[i + 1].yz, tk - float(i)), 1e-7f);
  const float took = 1.0f - rho * rho;
  float2 persist = took > 0.05f ? clamp(1.0f - e / (sn * sn * took), 0.0f, 1.0f) : float2(0.0f);
  // Only where the frame's own chroma is about as flat as noise: where it has
  // structure, a merge that took little off is more likely to have rejected
  // the neighbours' chroma (motion, a colour edge) than to have met noise
  // that persists, and the count map is kept.
  if (u.flags.x == 0) persist *= 1.0f - smoothstep(u.gate.x, u.gate.y, eo / (sn * sn));
  dst.write(float4(persist, 0.0f, 0.0f), gid);
}

// X_k = G_k + U(Dn_(k+1) - G_(k+1)); alpha = rho_k. On the finest level rho
// comes from the count map, on the others from G_k's own alpha.
kernel void spatial_prepare(texture2d<float, access::read> g [[texture(0)]],
                            texture2d<float, access::sample> dnCoarse [[texture(1)]],
                            texture2d<float, access::sample> gCoarse [[texture(2)]],
                            texture2d<float, access::sample> count [[texture(3)]],
                            texture2d<float, access::write> dst [[texture(4)]],
                            constant SpatialU& u [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
  float4 x = g.read(gid);
  float2 pc = (float2(gid) + 0.5f) * 0.5f;
  float3 a = dnCoarse.sample(spatialLinear, pc).xyz;
  float3 b = gCoarse.sample(spatialLinear, pc).xyz;
  float rho = x.w;
  if (u.flags.x != 0) {
    float2 uv = (float2(gid) + 0.5f) / float2(g.get_width(), g.get_height());
    rho = rsqrt(max(count.sample(spatialLinearN, uv).x, 1.0f));
  }
  dst.write(od_store_half(float4(x.xyz + (a - b), rho)), gid);
}

// Test aid (OD_SPATIAL_TEST_COUNT): a synthetic count map. mode 0 = n
// everywhere, 1 = left half 1 and right half n.
kernel void spatial_test_count(texture2d<float, access::write> dst [[texture(0)]],
                               constant float4& p [[buffer(0)]],
                               uint2 gid [[thread_position_in_grid]]) {
  float n = p.x;
  if (p.y > 0.5f && gid.x * 2 < dst.get_width()) n = 1.0f;
  dst.write(float4(n), gid);
}

// ---- NL-means --------------------------------------------------------------
// Search window (2R+1)^2, patch 3x3: the kernel needs the (2R+3)^2 texels
// around the pixel, each exactly once. They are fetched as half floats with
// constant offsets (no coordinate arithmetic per fetch) and turned in place
// into their difference to the centre pixel in units of sigma*sqrt(18),
// clamped to +-SPATIAL_CLAMP. A patch SSD is then directly q (expected value
// 1 for noise alone), and half precision is enough for it: 25 or 49 half3
// values fit the GPU's registers.
//
// On this GPU the kernel is bound by its arithmetic, not by its fetches, so what counts is
// the number of operations per pixel.
#define SPATIAL_CLAMP 40.0f  // |normalised value| limit: 9 * (2*40)^2 < half max

// PERSIST: the chroma threshold follows the persistence map `left` (spatial_chroma_persist).
// A separate instance, so that the levels without it keep the plain kernel's register budget
// (with the branch inside it the 5x5 kernel was 4 % slower on every level). diagnostic: the
// opt-in OD_SPATIAL_TUNE softspeck/diagnostic branches, in instances of their own
// (spatial_nlm*_research) so that the default ones keep their arithmetic. TILED: the window
// comes from the fused kernel's threadgroup tile (spatial_fused*) instead of X.
template <int R, bool PERSIST, bool RESEARCH = false, bool TILED = false, bool RESIDUAL = false>
static inline half4 spatial_nlm(texture2d<half, access::sample> X, texture2d<half, access::read> orig,
                                texture2d<half, access::sample> left, constant SpatialU& u, uint2 gid,
                                threadgroup half4* tile = nullptr, uint2 local = uint2(0)) {
  const int P = R + 1;      // window radius
  const int N = 2 * P + 1;  // window edge
  const int C = P * N + P;  // index of the centre
  const float2 c = float2(gid) + 0.5f;

  half3 win[N * N];
  half rhoh = 1.0h;
  #pragma clang loop unroll(full)
  for (int y = 0; y < N; ++y) {
    #pragma clang loop unroll(full)
    for (int x = 0; x < N; ++x) {
      const half4 t = TILED ? tile[(local.y + y) * (32 + 2 * P) + local.x + x]
                            : X.sample(spatialNearest, c, int2(x - P, y - P));
      win[y * N + x] = t.xyz;
      if (x == P && y == P) rhoh = t.w;
    }
  }
  // Luminance for the sigma lookup: mean of the centre 3x3 (the threshold
  // moves by well under 1 % with the pixel's own noise).
  float ysum = 0.0f;
  #pragma clang loop unroll(full)
  for (int y = -1; y <= 1; ++y) {
    #pragma clang loop unroll(full)
    for (int x = -1; x <= 1; ++x) ysum += float(win[C + y * N + x].x);
  }
  const float3 measured = RESIDUAL ? float3(left.sample(spatialLinear, c / 8.0f).xyz) : float3(1.0f);
  const float rho = RESIDUAL ? measured.x : float(rhoh);
  const half3 ctrh = win[C];
  const float3 ctr = float3(ctrh);
  const float4 sg = spatial_sigma(u, ysum * (1.0f / 9.0f));
  // rho = 1/sqrt(count) is what the temporal merge took off the pixel noise;
  // at coarse scales it takes off less (noise that persists between frames,
  // the coarse bands of a merge that rejects per band), so on the coarse
  // levels the luma threshold follows only part of it.
  const float rhoY = u.robust.x >= 1.0f ? rho : mix(1.0f, rho, u.robust.x);
  // Chroma: the share of the noise that persists between frames was not
  // averaged away by the merge (spatial_chroma_persist), whatever the count.
  float3 s = sg.xyz * float3(rhoY, rho, rho);
  if (PERSIST) {
    const float2 pc = float2(left.sample(spatialLinearN, c / float2(X.get_width(), X.get_height())).xy);
    s.yz = sg.yz * sqrt(pc + (1.0f - pc) * (rho * rho));
  }
  if (RESIDUAL) s = sg.xyz * measured;
  // q = (patch SSD / 9) / (2 sigma^2 + SIGMA_Q^2) = sum of (delta * k)^2
  const float3 var18 = 18.0f * s * s + 9.0f * SPATIAL_SIGMA_Q * SPATIAL_SIGMA_Q;
  const float3 k = rsqrt(var18);

  // Mean of the 8 neighbours relative to the centre, or 0 (= the centre
  // itself, no alternative) when the centre is close enough to be ordinary
  // noise or texture, or too far out to be noise at all. This is the one
  // yes-or-no decision of the kernel, so its input is normalised in single
  // precision: the rounding of the half-precision values below would decide
  // it for more pixels than necessary.
  float m8 = 0.0f;
  #pragma clang loop unroll(full)
  for (int y = -1; y <= 1; ++y) {
    #pragma clang loop unroll(full)
    for (int x = -1; x <= 1; ++x) {
      const float v = (float(win[C + y * N + x].x) - ctr.x) * k.x;
      m8 += float(half(clamp(v, -SPATIAL_CLAMP, SPATIAL_CLAMP)));
    }
  }
  m8 *= 0.125f;
  // m8 > 0: the centre is darker than its neighbours. The dark tail of shot
  // noise in log footage, and dead pixels, reach far beyond 12 sigma; a bright
  // pixel that far out is more likely detail (a star, a specular) and keeps
  // the 12-sigma limit.
  half mref = (abs(m8) > u.robust.y && abs(m8) < (m8 > 0.0f ? u.amount.w : u.robust.z)) ? half(m8) : half(0);

  if (RESEARCH && u.research.x > 0.0f) {
    // A one-sigma ramp at each endpoint replaces the discontinuous window.
    // The high-end fade preserves isolated bright features outside the window.
    float hi = m8 > 0.0f ? u.amount.w : u.robust.z;
    float width = u.research.x / sqrt(18.0f);
    mref = half(m8 * smoothstep(u.robust.y, u.robust.y + width, abs(m8)) *
                (1.0f - smoothstep(hi - width, hi, abs(m8))));
  }

  // The product can overflow to infinity (a super-white next to a dark pixel
  // at a tiny sigma); the clamp brings that back to the limit. The limit is
  // also what makes everything beyond a very strong edge look alike.
  const half3 kh = half3(k);
  #pragma clang loop unroll(full)
  for (int i = 0; i < N * N; ++i) win[i] = clamp((win[i] - ctrh) * kh, half(-SPATIAL_CLAMP), half(SPATIAL_CLAMP));

  const half ieY = half(u.shape.x), ieC = half(u.shape.y), guide = half(u.shape.z), knee = half(u.shape.w);
  const half tcap = half(u.robust.w);
  half3 acc = half3(0.0h);  // weighted sums: at most 24 terms of |value| < 3
  half2 ws = half2(0.0h);   // weight sums, luma and chroma
  half2 wm = half2(0.0h);   // largest weights
  // Candidates come in pairs +o and -o. The squared difference of the pixels
  // u and u + o is a term of both patch distances (of +o at patch position
  // u, of -o at patch position u + o) wherever both positions are inside the
  // 3x3 patch, so it is computed once: 52 instead of 72 terms for the 3x3
  // search, 180 instead of 216 for 5x5.
  #pragma clang loop unroll(full)
  for (int oy = 0; oy <= R; ++oy) {
    #pragma clang loop unroll(full)
    for (int ox = -R; ox <= R; ++ox) {
      if (oy == 0 && ox <= 0) continue;  // one half of the search window
      const int o = oy * N + ox;
      half2 dP = half2(0.0h), dM = half2(0.0h);  // x = luma, y = chroma distance of +o and of -o
      #pragma clang loop unroll(full)
      for (int uy = -1 - R; uy <= 1; ++uy) {
        #pragma clang loop unroll(full)
        for (int ux = -1 - R; ux <= 1 + R; ++ux) {
          const bool inP = abs(ux) <= 1 && abs(uy) <= 1;            // a term of +o
          const bool inM = abs(ux + ox) <= 1 && abs(uy + oy) <= 1;  // a term of -o
          if (!inP && !inM) continue;
          const int i = C + uy * N + ux;
          const half3 e = win[i + o] - win[i];
          const half tC = e.y * e.y + e.z * e.z;
          // Luma: the term of one pixel AROUND the centre is capped, so a
          // single outlier of the noise (the dark specks of shot noise in log
          // footage) next to this pixel, or next to the candidate, does not
          // veto the match. The centre's own term follows below.
          const half tY = min(e.x * e.x, tcap);
          if (inP) {
            dP.y += tC;
            if (ux != 0 || uy != 0) dP.x += tY;
          }
          if (inM) {
            dM.y += tC;
            if (ux + ox != 0 || uy + oy != 0) dM.x += tY;
          }
        }
      }
      #pragma clang loop unroll(full)
      for (int side = 0; side < 2; ++side) {
        const half3 v = win[side == 0 ? C + o : C - o];
        const half2 d = side == 0 ? dP : dM;
        // The centre term is never capped (a candidate that is itself an
        // outlier must not be averaged in), but when THIS pixel stands out
        // from its own 8 neighbours like an outlier of the noise does
        // (mref != 0), a candidate may match their mean instead: that removes
        // the speck.
        const half dY = d.x + min(v.x * v.x, (v.x - mref) * (v.x - mref));
        // Weight: 1 up to the knee (plain noise, whose q scatters around 1,
        // is then averaged uniformly instead of the filter following the
        // grain), exponential beyond it.
        const half qC = max(guide * dY, 0.5h * d.y);
        const half wY = exp(-max(dY - knee, 0.0h) * ieY);
        const half wC = exp(-max(qC - knee, 0.0h) * ieC);
        acc.x += wY * v.x;
        acc.yz += wC * v.yz;
        ws += half2(wY, wC);
        wm = max(wm, half2(wY, wC));
      }
    }
  }
  // The centre gets the weight of its best match, so a pixel without matches
  // keeps itself instead of a noisy mean of bad ones.
  const float wsY = float(ws.x) + max(float(wm.x), 1e-3f);
  const float wsC = float(ws.y) + max(float(wm.y), 1e-3f);
  // Back to working units, faded out as sigma falls to the quantisation
  // floor so that the stage is continuous down to strength 0.
  const float3 unit = var18 * k * (18.0f * s * s / var18);
  float3 dn = ctr;
  if (u.flags.y != 0) dn.x += float(acc.x) / wsY * unit.x;
  if (u.flags.z != 0) dn.yz += float2(acc.yz) / wsC * unit.yz;

  if (u.flags.x == 0) return half4(od_store_half(float4(dn, rho)));

  // Final level: the amounts, against the ORIGINAL frame (notes 2.4), and the
  // grain floor: never remove more luma grain than leaves `keep` of it, so a
  // clean gradient still carries dither-level noise when quantised later.
  const float3 f0 = float3(orig.read(gid).xyz);
  const float aY = od_luma_amount(u.amount.x, u.amount.z, sg.w);
  if (RESEARCH && u.research.y == 5.0f) return half4(wsY / 25.0f, 0.0h, 0.0h, 1.0h);
  if (RESEARCH && u.research.y == 6.0f) return half4(mref != 0.0h ? 1.0h : 0.0h, 0.0h, 0.0h, 1.0h);
  if (RESEARCH && u.research.y == 2.0f) return half4(aY, 0.0h, 0.0h, 1.0h);
  if (RESEARCH && u.research.y == 4.0f) return half4(od_store_half(float4(dn, rho)));
  float3 o;
  o.x = u.amount.x > 0.0f ? f0.x + aY * (dn.x - f0.x) : f0.x;
  o.yz = u.amount.y > 0.0f ? f0.yz + u.amount.y * (dn.yz - f0.yz) : f0.yz;
  // Alpha: rho for the sharpen stage (sharpen_usm.metal), which would
  // otherwise have to sample the count map again. Not the rho of the test
  // aid's map, which the sharpen stage never sees.
  return half4(od_store_half(float4(o, u.flags.w != 0 ? rho : 1.0f)));
}

// The results are written as half4: converting in the shader rounds to
// nearest (see od_round_half in common.metal).
kernel void spatial_nlm5(texture2d<half, access::sample> X [[texture(0)]],
                         texture2d<half, access::read> orig [[texture(1)]],
                         texture2d<half, access::write> dst [[texture(2)]],
                         constant SpatialU& u [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
  dst.write(spatial_nlm<2, false>(X, orig, X, u, gid), gid);
}

kernel void spatial_nlm3(texture2d<half, access::sample> X [[texture(0)]],
                         texture2d<half, access::read> orig [[texture(1)]],
                         texture2d<half, access::write> dst [[texture(2)]],
                         constant SpatialU& u [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
  dst.write(spatial_nlm<1, false>(X, orig, X, u, gid), gid);
}

// Separate instances preserve the legacy shaders' arithmetic and registers.
kernel void spatial_nlm5_residual(texture2d<half, access::sample> X [[texture(0)]],
                                  texture2d<half, access::read> orig [[texture(1)]],
                                  texture2d<half, access::write> dst [[texture(2)]],
                                  texture2d<half, access::sample> left [[texture(3)]],
                                  constant SpatialU& u [[buffer(0)]],
                                  uint2 gid [[thread_position_in_grid]]) {
  dst.write(spatial_nlm<2, false, true, false, true>(X, orig, left, u, gid), gid);
}
kernel void spatial_nlm3_residual(texture2d<half, access::sample> X [[texture(0)]],
                                  texture2d<half, access::read> orig [[texture(1)]],
                                  texture2d<half, access::write> dst [[texture(2)]],
                                  texture2d<half, access::sample> left [[texture(3)]],
                                  constant SpatialU& u [[buffer(0)]],
                                  uint2 gid [[thread_position_in_grid]]) {
  dst.write(spatial_nlm<1, false, true, false, true>(X, orig, left, u, gid), gid);
}

// The coarse levels whose chroma threshold follows the persistence map.
kernel void spatial_nlm5_persist(texture2d<half, access::sample> X [[texture(0)]],
                                 texture2d<half, access::read> orig [[texture(1)]],
                                 texture2d<half, access::write> dst [[texture(2)]],
                                 texture2d<half, access::sample> left [[texture(3)]],
                                 constant SpatialU& u [[buffer(0)]],
                                 uint2 gid [[thread_position_in_grid]]) {
  dst.write(spatial_nlm<2, true>(X, orig, left, u, gid), gid);
}

// Fused prepare + final NLM. A 32x8 output tile carries its 2/3-pixel halo
// through the SAME half rounding as spatial_prepare. Only texture transport
// is replaced; patch arithmetic, accumulation order and amounts are shared.
// Uniform threadgroups are intentional: every lane reaches the barrier,
// including the padded lanes of odd-sized images.
template <int R>
static inline void spatial_fused(texture2d<half, access::sample> g,
                                  texture2d<half, access::read> orig,
                                  texture2d<half, access::write> dst,
                                  texture2d<float, access::sample> dnCoarse,
                                  texture2d<float, access::sample> gCoarse,
                                  texture2d<float, access::sample> count,
                                  constant SpatialU& u, uint2 gid, uint2 local,
                                  uint2 group, threadgroup half4* tile) {
  constexpr uint P = R + 1, TW = 32 + 2*P, TH = 8 + 2*P;
  const uint lane = local.y * 32 + local.x;
  for (uint i = lane; i < TW*TH; i += 256) {
    const int2 q = int2(group * uint2(32, 8)) + int2(i % TW, i / TW) - int(P);
    const uint2 p = uint2(clamp(q, int2(0), int2(g.get_width()-1, g.get_height()-1)));
    const float4 x = float4(g.read(p));
    const float2 pc = (float2(p) + 0.5f) * 0.5f;
    const float3 a = dnCoarse.sample(spatialLinear, pc).xyz;
    const float3 b = gCoarse.sample(spatialLinear, pc).xyz;
    const float2 uv = (float2(p) + 0.5f) / float2(g.get_width(), g.get_height());
    const float rho = rsqrt(max(count.sample(spatialLinearN, uv).x, 1.0f));
    tile[i] = half4(od_store_half(float4(x.xyz + (a-b), rho)));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
  dst.write(spatial_nlm<R, false, false, true>(g, orig, g, u, gid, tile, local), gid);
}

kernel void spatial_fused3(texture2d<half, access::sample> g [[texture(0)]],
                            texture2d<half, access::read> orig [[texture(1)]],
                            texture2d<half, access::write> dst [[texture(2)]],
                            texture2d<float, access::sample> dnCoarse [[texture(4)]],
                            texture2d<float, access::sample> gCoarse [[texture(5)]],
                            texture2d<float, access::sample> count [[texture(6)]],
                            constant SpatialU& u [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]],
                            uint2 local [[thread_position_in_threadgroup]],
                            uint2 group [[threadgroup_position_in_grid]]) {
  threadgroup half4 tile[36*12];
  spatial_fused<1>(g, orig, dst, dnCoarse, gCoarse, count, u, gid, local, group, tile);
}

kernel void spatial_fused5(texture2d<half, access::sample> g [[texture(0)]],
                            texture2d<half, access::read> orig [[texture(1)]],
                            texture2d<half, access::write> dst [[texture(2)]],
                            texture2d<float, access::sample> dnCoarse [[texture(4)]],
                            texture2d<float, access::sample> gCoarse [[texture(5)]],
                            texture2d<float, access::sample> count [[texture(6)]],
                            constant SpatialU& u [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]],
                            uint2 local [[thread_position_in_threadgroup]],
                            uint2 group [[threadgroup_position_in_grid]]) {
  threadgroup half4 tile[38*14];
  spatial_fused<2>(g, orig, dst, dnCoarse, gCoarse, count, u, gid, local, group, tile);
}

// Stateless frequency-selective reference for the final LUMA amount blend.
// B is a full-resolution separable [1 2 1]/4 filter (four bilinear samples).
// R' = R - stable * ((1-fine)*(R-T) + fine*B(R-T)).
// fine=1 keeps the finest residual; fine=0 is the full temporal reference.
// Chroma and alpha come unchanged from R. No synthetic noise or history.
kernel void spatial_stable_reference(texture2d<float, access::sample> orig [[texture(0)]],
                                      texture2d<float, access::sample> temporal [[texture(1)]],
                                      texture2d<float, access::write> dst [[texture(2)]],
                                      constant float4& u [[buffer(0)]],
                                      uint2 gid [[thread_position_in_grid]]) {
  float2 p = float2(gid) + 0.5f;
  float4 r = orig.sample(spatialNearest, p);
  float residual = r.x - temporal.sample(spatialNearest, p).x;
  float low = 0;
  for (int y = -1; y <= 1; y += 2)
    for (int x = -1; x <= 1; x += 2) {
      float2 q = p + 0.5f * float2(x, y);
      low += orig.sample(spatialLinear, q).x - temporal.sample(spatialLinear, q).x;
    }
  r.x -= u.x * ((1.0f - u.y) * residual + u.y * low * 0.25f);
  dst.write(od_round_half(r), gid);
}

// Encodes the effective count in neutral luma for offline diagnostics.
kernel void spatial_count_view(texture2d<float, access::read> count [[texture(0)]],
                                texture2d<float, access::write> dst [[texture(1)]],
                                uint2 gid [[thread_position_in_grid]]) {
  dst.write(float4(count.read(gid).x / 11.0f, 0, 0, 1), gid);
}

// Separate diagnostic instances keep opt-in diagnostic branches and soft-speck arithmetic
// completely out of the original shader specialization.
kernel void spatial_nlm5_research(texture2d<half, access::sample> X [[texture(0)]],
                                   texture2d<half, access::read> orig [[texture(1)]],
                                   texture2d<half, access::write> dst [[texture(2)]],
                                   constant SpatialU& u [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
  dst.write(spatial_nlm<2, false, true>(X, orig, X, u, gid), gid);
}
kernel void spatial_nlm3_research(texture2d<half, access::sample> X [[texture(0)]],
                                   texture2d<half, access::read> orig [[texture(1)]],
                                   texture2d<half, access::write> dst [[texture(2)]],
                                   constant SpatialU& u [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
  dst.write(spatial_nlm<1, false, true>(X, orig, X, u, gid), gid);
}
kernel void spatial_nlm5_persist_research(texture2d<half, access::sample> X [[texture(0)]],
                                           texture2d<half, access::read> orig [[texture(1)]],
                                           texture2d<half, access::write> dst [[texture(2)]],
                                           texture2d<half, access::sample> left [[texture(3)]],
                                           constant SpatialU& u [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
  dst.write(spatial_nlm<2, true, true>(X, orig, left, u, gid), gid);
}

// Additional correction after the ordinary spatial result. It cannot alter
// luma or an unqualified luminance interval. Only the measured bands run.
struct CoarseU { float4 sig[OD_LEVELS]; }; // x = luma guard, yz = sigma, w = qualified
static inline float4 coarse_sigma(constant CoarseU& u,float y) {
  float t=clamp(y,0.0f,1.0f)*float(OD_LEVELS-1);int i=min(int(t),OD_LEVELS-2);
  return mix(u.sig[i],u.sig[i+1],t-float(i));
}
kernel void spatial_coarse_filter(texture2d<float,access::sample> g [[texture(0)]],
                                   texture2d<float,access::sample> parentDelta [[texture(1)]],
                                   texture2d<float,access::write> delta [[texture(2)]],
                                   constant CoarseU& u [[buffer(0)]],
                                   constant uint& top [[buffer(1)]],
                                   uint2 gid [[thread_position_in_grid]]) {
  float2 p=float2(gid)+.5f;
  float3 center=g.sample(spatialLinear,p).xyz;
  float4 s=coarse_sigma(u,center.x);
  if(s.w<=0){delta.write(float4(0),gid);return;}
  float2 sigma=max(s.yz,1e-7f);
  float2 acc=center.yz;float weight=1;
  // Centred luma guide from the diagnostic variant. Uncentred colour patch distance protects
  // equal-luma boundaries; offsets remain symmetric.
  for(int y=-2;y<=2;++y)for(int x=-2;x<=2;++x){
    if(x==0&&y==0)continue;
    float dy=0,dy2=0,dc=0;
    for(int j=-1;j<=1;++j)for(int i=-1;i<=1;++i){
      float3 a=g.sample(spatialLinear,p+float2(i,j)).xyz;
      float3 b=g.sample(spatialLinear,p+float2(i+x,j+y)).xyz;
      float v=(a.x-b.x)/max(s.x,2e-4f);dy+=v;dy2+=v*v;
      float2 c=(a.yz-b.yz)/sigma;dc+=dot(c,c);
    }
    float q=max(.25f*max(dy2-dy*dy/9.0f,0.0f)/18.0f,dc/36.0f);
    float w=exp(-max(q-2.0f,0.0f)*2.0f);
    acc+=w*g.sample(spatialLinear,p+float2(x,y)).yz;weight+=w;
  }
  float2 d=acc/weight-center.yz;
  if(!top)d+=parentDelta.sample(spatialLinear,p*.5f).xy;
  // Confidence controls admission, not a false independent-frame reduction.
  delta.write(float4(d*clamp(s.w,0.0f,1.0f),0,0),gid);
}
kernel void spatial_coarse_apply(texture2d<float,access::read> base [[texture(0)]],
                                  texture2d<float,access::sample> delta [[texture(1)]],
                                  texture2d<float,access::sample> guide [[texture(2)]],
                                  texture2d<float,access::write> dst [[texture(3)]],
                                  constant CoarseU& u [[buffer(0)]],
                                  constant float& amount [[buffer(1)]],
                                  uint2 gid [[thread_position_in_grid]]) {
  float4 v=base.read(gid);float2 pos=(float2(gid)+.5f)/8.0f;
  float4 s=coarse_sigma(u,v.x);
  float2 d=delta.sample(spatialLinear,pos).xy;
  float3 coarse=guide.sample(spatialLinear,pos).xyz;
  float2 contrast=abs(v.yz-coarse.yz)/max(s.yz,2e-4f);
  float gate=1.0f-smoothstep(4.0f,8.0f,max(contrast.x,contrast.y));
  gate*=1.0f-smoothstep(4.0f,8.0f,abs(v.x-coarse.x)/max(s.x,2e-4f));
  if(s.w>0&&gate>0)v.yz+=d*gate*amount;
  dst.write(od_store_half(v),gid);
}
