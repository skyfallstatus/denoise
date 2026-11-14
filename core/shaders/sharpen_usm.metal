// SPDX-License-Identifier: MIT
// Sharpen stage: cored unsharp masking on the profile's own
// a-trous bands 0 and 1 of the luma, with a local min/max overshoot limit.
//
//   c1 = B3(Y)            d0 = Y - c1      (band 0)
//   c2 = B3 spaced 2 (c1) d1 = c1 - c2     (band 1)
//   Y' = Y + g0 core(d0, T0) + g1 core(d1, T1),  core(d, T) = d * smoothstep(T, 3T, |d|)
//   Y' = clamp(Y', mn - r, mx + r),  r = k min(mx - mn, 1)   mn, mx over the 3x3 neighbourhood
// The allowance is a share of the local range up to a range of 1 (all of
// 0..1 footage), and 0.1 at most beyond: next to a super-white, an HDR
// highlight or a hot pixel at the end of the half range a share of the range
// drove the neighbours far below zero (thousands below next to 65504). So the
// overshoot on highlights above 1 is limited to 0.1 above the local maximum.
// T_b is kCore (sharpen.mm) x the noise expected to be left in band b at this
// pixel, so detail below the noise floor is never amplified:
//   left = (1 - a) + a * s / sqrt(count)
// a = luma amount of the denoising stages as they applied it, i.e. capped by
// the grain floor (V2: band 0 only; od_luma_amount, stage.h kGrainFloor),
// s = share the spatial stage leaves,
// count = the temporal stage's effective frame count. 1/sqrt(count) comes
// from the alpha channel of the image when the spatial stage ran (it writes
// it there for this purpose, see spatial_nlm.metal), else from the count map
// (a 1x1 texture holding 1 when there is none). Chroma and alpha are copied;
// a pixel whose detail is below the floor comes out bit-identical.
//
// Every kernel here is bound by its texture fetches and by the bytes it moves, not by
// arithmetic, so the centre texel is fetched once and the limiter's neighbours come from
// gathers.
//
// od: math=safe

struct SharpenU {
  float4 thr[OD_LEVELS];  // per luminance knot: x, y = kCore x profiled sigma of bands 0, 1,
                            // z = full-band luma sigma (unscaled, for the grain floor)
  float4 gain;              // x = g0, y = g1, z = overshoot allowance k, w = threshold floor
  float4 left;              // x = luma amount a (0 = nothing was denoised), y = s,
                            // z = 1: the image's alpha holds 1/sqrt(count), 0: sample the count map,
                            // w = grain floor the denoising stages applied to a (0 = none)
  uint4 flags;               // x = fine-only grain retention (V2)
};

constexpr sampler sharpenNearest(coord::pixel, address::clamp_to_edge, filter::nearest);
constexpr sampler sharpenLinear(coord::pixel, address::clamp_to_edge, filter::linear);
constexpr sampler sharpenLinearN(coord::normalized, address::clamp_to_edge, filter::linear);

// Thresholds of bands 0 and 1 at luminance y for the pixel at gid. alpha is
// the image's alpha at that pixel.
static inline float2 sharpen_thr(constant SharpenU& u, float y, float alpha, texture2d<float, access::sample> count,
                                 uint2 gid, uint2 size) {
  float t = clamp(y, 0.0f, 1.0f) * float(OD_LEVELS - 1);
  int i = min(int(t), OD_LEVELS - 2);
  const float4 k = mix(u.thr[i], u.thr[i + 1], t - float(i));
  const float2 T = k.xy;
  float rho = alpha;
  if (u.left.z == 0.0f) {
    float2 uv = (float2(gid) + 0.5f) / float2(size);
    rho = rsqrt(max(count.sample(sharpenLinearN, uv).x, 1.0f));
  }
  const float a = u.left.w > 0.0f ? od_luma_amount(u.left.x, u.left.w, k.z) : u.left.x;
  if (u.flags.x == 0u) {
    float left = (1.0f - a) + a * u.left.y * rho;
    return max(T * left, float2(u.gain.w));
  }
  const float2 amount = float2(a, u.left.x);
  float2 left = (1.0f - amount) + amount * u.left.y * rho;
  return max(T * left, float2(u.gain.w));
}

static inline float sharpen_core(float d, float T) { return d * smoothstep(T, 3.0f * T, abs(d)); }

// 5x5 B3 of the luma around the texel centre c, whose own luma is `centre`:
// [1 4 6 4 1]/16 is [5 6 5]/16 at offsets -1.2, 0, +1.2, so the 24 texels
// around the centre take 8 bilinear fetches.
static inline float sharpen_b3_luma(texture2d<float, access::sample> src, float2 c, float centre) {
  const float edges = src.sample(sharpenLinear, c + float2(-1.2f, 0.0f)).x + src.sample(sharpenLinear, c + float2(1.2f, 0.0f)).x +
                      src.sample(sharpenLinear, c + float2(0.0f, -1.2f)).x + src.sample(sharpenLinear, c + float2(0.0f, 1.2f)).x;
  const float corners = src.sample(sharpenLinear, c + float2(-1.2f, -1.2f)).x + src.sample(sharpenLinear, c + float2(1.2f, -1.2f)).x +
                        src.sample(sharpenLinear, c + float2(-1.2f, 1.2f)).x + src.sample(sharpenLinear, c + float2(1.2f, 1.2f)).x;
  return (36.0f * centre + 30.0f * edges + 25.0f * corners) * (1.0f / 256.0f);
}

// Luma min and max over the 3x3 (full) or the 5 points of the plus around
// the texel gid, whose own luma is s. A gather returns the 2x2 texels around
// a texel corner as (x: left below, y: right below, z: right above, w: left
// above), "below" being the larger row; edges replicate.
static inline float2 sharpen_minmax(texture2d<float, access::sample> src, uint2 gid, float s, bool full) {
  const float2 p = float2(gid);
  const float4 a = src.gather(sharpenLinear, p, int2(0), component::x);         // columns x-1, x; rows y-1, y
  const float4 d = src.gather(sharpenLinear, p + 1.0f, int2(0), component::x);  // columns x, x+1; rows y, y+1
  float mn = min(min(s, a.x), min(a.z, min(d.z, d.x)));  // left, above, right, below
  float mx = max(max(s, a.x), max(a.z, max(d.z, d.x)));
  if (full) {
    const float4 b = src.gather(sharpenLinear, p + float2(1.0f, 0.0f), int2(0), component::x);  // columns x, x+1; rows y-1, y
    const float4 e = src.gather(sharpenLinear, p + float2(0.0f, 1.0f), int2(0), component::x);  // columns x-1, x; rows y, y+1
    mn = min(min(mn, a.w), min(d.y, min(b.z, e.x)));  // the four diagonal neighbours
    mx = max(max(mx, a.w), max(d.y, max(b.z, e.x)));
  }
  return float2(mn, mx);
}

// The overshoot limit (header: the allowance is bounded in absolute terms),
// and the end of the half range: a value near that end must not be written as
// infinity (stage.h section 2), whatever the allowance is tuned to.
static inline float sharpen_limit(float y, float2 mm, float k) {
  float r = k * min(mm.y - mm.x, 1.0f);
  return clamp(y, max(mm.x - r, -OD_HALF_MAX), min(mm.y + r, OD_HALF_MAX));
}

// c1 = B3(Y), r16f.
kernel void sharpen_b3(texture2d<float, access::sample> src [[texture(0)]],
                       texture2d<float, access::write> dst [[texture(1)]],
                       uint2 gid [[thread_position_in_grid]]) {
  const float2 c = float2(gid) + 0.5f;
  dst.write(float4(sharpen_b3_luma(src, c, src.sample(sharpenNearest, c).x)), gid);
}

// c2 = B3 with taps spaced 2 of c1, r16f.
kernel void sharpen_b3_dilated(texture2d<float, access::sample> c1 [[texture(0)]],
                               texture2d<float, access::write> dst [[texture(1)]],
                               uint2 gid [[thread_position_in_grid]]) {
  const float k[5] = {1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f};
  float2 c = float2(gid) + 0.5f;
  float a = 0.0f;
  #pragma clang loop unroll(full)
  for (int j = -2; j <= 2; ++j) {
    #pragma clang loop unroll(full)
    for (int i = -2; i <= 2; ++i) {
      a += k[i + 2] * k[j + 2] * c1.sample(sharpenNearest, c, 2 * int2(i, j)).x;
    }
  }
  dst.write(float4(a), gid);
}

// quality 1: bands 0 and 1, 3x3 limiter.
kernel void sharpen_apply(texture2d<float, access::sample> src [[texture(0)]],
                          texture2d<float, access::read> c1t [[texture(1)]],
                          texture2d<float, access::read> c2t [[texture(2)]],
                          texture2d<half, access::write> dst [[texture(3)]],
                          texture2d<float, access::sample> count [[texture(4)]],
                          constant SharpenU& u [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
  float4 s = src.sample(sharpenNearest, float2(gid) + 0.5f);
  float c1 = c1t.read(gid).x, c2 = c2t.read(gid).x;
  float2 T = sharpen_thr(u, c2, s.w, count, gid, uint2(dst.get_width(), dst.get_height()));
  float add = u.gain.x * sharpen_core(s.x - c1, T.x) + u.gain.y * sharpen_core(c1 - c2, T.y);
  float4 o = s;
  if (add != 0.0f) o.x = sharpen_limit(s.x + add, sharpen_minmax(src, gid, s.x, true), u.gain.z);
  dst.write(half4(o), gid);  // converting in the shader rounds to nearest (od_round_half)
}

// quality 0: band 0 only, one pass, 5-point limiter.
kernel void sharpen_apply_fast(texture2d<float, access::sample> src [[texture(0)]],
                               texture2d<half, access::write> dst [[texture(1)]],
                               texture2d<float, access::sample> count [[texture(2)]],
                               constant SharpenU& u [[buffer(0)]],
                               uint2 gid [[thread_position_in_grid]]) {
  float2 c = float2(gid) + 0.5f;
  float4 s = src.sample(sharpenNearest, c);
  float c1 = sharpen_b3_luma(src, c, s.x);
  float add = u.gain.x * sharpen_core(s.x - c1, sharpen_thr(u, c1, s.w, count, gid, uint2(dst.get_width(), dst.get_height())).x);
  float4 o = s;
  if (add != 0.0f) o.x = sharpen_limit(s.x + add, sharpen_minmax(src, gid, s.x, false), u.gain.z);
  dst.write(half4(o), gid);
}
