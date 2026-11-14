// SPDX-License-Identifier: MIT
// Noise profiler: GPU side. Reduces a frame (and, when a temporal neighbour is
// available, the difference to that neighbour) to 24 floats per 32x32 tile.
// core/src/stages/profile.mm turns those into an ODNoiseProfile.
//
// Band definition: undecimated B3 a-trous, d_b = c_b - c_(b+1). The coefficients are EXACT
// values of that transform, but only evaluated where they are needed: d_0 at every pixel
// (c_1 is computed at every pixel, kept only on the even lattice: x, y multiples of 2) ->
// C1, half size d_1 on the even lattice (c_2's taps are 2 px apart, so they are on that
// lattice); c_2 is kept on the 4-lattice -> C2, quarter d_2, d_3 on the 4-lattice (taps 4
// and 8 px apart) -> C3, quarter So no full-resolution target is written. The lattice
// textures are rgba32Float: c_b - c_(b+1) is a small difference of large numbers, and half
// precision would add its own quantisation noise to bands 1..3.
//
// Passes, for one image (the frame, or the frame difference):
//   diff   (temporal only) cur - prev, full size, rgba16Float
//   l0     thread = 2x2-pixel block: c_1, sum of d_0^2         (reads the frame)
//   l1     thread = 4x4-pixel cell: c_2, sum of d_1^2, gradient tensor (reads C1)
//   b3     c_3                                                 (reads C2)
//   l3     c_4 (not stored), d_2^2, d_3^2, (c_0 - c_4)^2       (reads C3)
//   sum2   twice: 2x2 cells -> 8-pixel blocks -> 16-pixel blocks
//   tiles  2x2 blocks -> tile record in the statistics buffer
// Every kernel is short and loop-free, and there are many light threads
// rather than a few heavy ones. Both matter a lot on this GPU. Measured at
// 4K, each pass on its own: a tile kernel looping over its own 1024 pixels
// 1-2.5 ms; l0 as one thread per 4x4 cell looping over its four blocks
// 1.9 ms; l0 per 2x2 block with loops and local arrays 0.8 ms; l0 as written
// here 0.2 ms (about 1 ms inside the whole chain).
// od: math=safe

static inline uint2 prof_mirror(int2 p, int2 n) {
  p = abs(p);
  p = select(p, 2 * n - 2 - p, p >= n);
  return uint2(clamp(p, int2(0), n - 1));
}

static inline float3 prof_b3(float3 a, float3 b, float3 c, float3 d, float3 e) {
  return (a + e) * 0.0625f + (b + d) * 0.25f + c * 0.375f;
}

// c_1 and c_0 at the four pixels of a 2x2 block, from the 6x6 neighbourhood
// whose top-left texel is `o`. Written out without arrays or loops on
// purpose (see the timings at the top of this file).
struct ProfBlock {
  float3 c1_00, c1_10, c1_01, c1_11;  // B3(c_0) at (dx, dy) = (0,0) (1,0) (0,1) (1,1)
  float3 c0_00, c0_10, c0_01, c0_11;
  float w_00, w_10, w_01, w_11;       // the fourth channel of the four centre texels
};

#define PROF_ROW(tex, j, expr)                                             \
  const float4 a##j = expr(tex, o + int2(0, j), n), b##j = expr(tex, o + int2(1, j), n), \
               c##j = expr(tex, o + int2(2, j), n), d##j = expr(tex, o + int2(3, j), n), \
               e##j = expr(tex, o + int2(4, j), n), f##j = expr(tex, o + int2(5, j), n); \
  const float3 h##j = prof_b3(a##j.xyz, b##j.xyz, c##j.xyz, d##j.xyz, e##j.xyz);         \
  const float3 g##j = prof_b3(b##j.xyz, c##j.xyz, d##j.xyz, e##j.xyz, f##j.xyz);

#define PROF_BLOCK(blk)                          \
  blk.c1_00 = prof_b3(h0, h1, h2, h3, h4);       \
  blk.c1_10 = prof_b3(g0, g1, g2, g3, g4);       \
  blk.c1_01 = prof_b3(h1, h2, h3, h4, h5);       \
  blk.c1_11 = prof_b3(g1, g2, g3, g4, g5);       \
  blk.c0_00 = c2.xyz;                            \
  blk.c0_10 = d2.xyz;                            \
  blk.c0_01 = c3.xyz;                            \
  blk.c0_11 = d3.xyz;                            \
  blk.w_00 = c2.w;                               \
  blk.w_10 = d2.w;                               \
  blk.w_01 = c3.w;                               \
  blk.w_11 = d3.w;

static inline float4 prof_read(texture2d<float, access::read> t, int2 p, int2 n) { return t.read(prof_mirror(p, n)); }

static inline ProfBlock prof_block(texture2d<float, access::read> t, int2 o, int2 n) {
  PROF_ROW(t, 0, prof_read)
  PROF_ROW(t, 1, prof_read)
  PROF_ROW(t, 2, prof_read)
  PROF_ROW(t, 3, prof_read)
  PROF_ROW(t, 4, prof_read)
  PROF_ROW(t, 5, prof_read)
  ProfBlock blk;
  PROF_BLOCK(blk)
  return blk;
}

// xyz = sum of d_0^2 over the block, w = sum of d_0.Y^4.
static inline float4 prof_block_energy(ProfBlock b) {
  const float3 p = b.c0_00 - b.c1_00, q = b.c0_10 - b.c1_10, r = b.c0_01 - b.c1_01, s = b.c0_11 - b.c1_11;
  const float3 e = p * p + q * q + r * r + s * s;
  return float4(e, p.x * p.x * p.x * p.x + q.x * q.x * q.x * q.x + r.x * r.x * r.x * r.x + s.x * s.x * s.x * s.x);
}

// Dead pixels of a block: no fine detail at all in Y (clipped highlights,
// synthetic flat), or pinned at black (exactly 0). Noise that is cut off at
// black is weaker than the noise just above it; measuring it would pull the
// darkest knot down and, through the interpolation, every level up to the
// next knot. Values BELOW 0 are not dead: a float pipeline carries sub-black
// (video-range sources, linear raw), and negative values prove the noise is
// not cut off there. Counting them dead left a uniformly dark frame with no
// profile at all, so nothing was denoised. Noise clipped at a level below 0
// is only caught where a whole neighbourhood is constant (the first test).
static inline float prof_dead(float c0, float c1) { return float(fabs(c0 - c1) < 1e-6f || fabs(c0) <= 1e-6f); }
// The same with "no fine detail in any channel": on footage whose luma has no
// detail anywhere (constant luma under chroma noise) the test above kills
// every tile, and the profile uses this count instead (resolve(), record 21).
static inline float prof_dead_all(float3 c0, float3 c1) {
  return float(all(abs(c0 - c1) < float3(1e-6f)) || fabs(c0.x) <= 1e-6f);
}
// Both counts travel in one float through the chain: luma test +
// PROF_DEAD_ALL * all-channel test. They are integers and a tile counts at
// most 1024 pixels, so every sum stays exact in float32 (below 2^24) and
// profile_tiles separates them; record 19 is bit for bit the luma count.
#define PROF_DEAD_ALL 4096.0f
static inline float prof_block_dead(ProfBlock b) {
  return prof_dead(b.c0_00.x, b.c1_00.x) + prof_dead(b.c0_10.x, b.c1_10.x) + prof_dead(b.c0_01.x, b.c1_01.x) +
         prof_dead(b.c0_11.x, b.c1_11.x) +
         PROF_DEAD_ALL * (prof_dead_all(b.c0_00, b.c1_00) + prof_dead_all(b.c0_10, b.c1_10) +
                          prof_dead_all(b.c0_01, b.c1_01) + prof_dead_all(b.c0_11, b.c1_11));
}

// l0, one frame. Thread = one 2x2-pixel block (half-size grid). Deliberately
// not one thread per 4x4 cell: fewer, longer threads measured 2-3 times
// slower on this GPU although they write less.
//   outC1 (half size)  xyz = c_1 on the even lattice, w = dead pixels of the block
//   outE0 (half size)  xyz = sum over the block of d_0^2, w = sum of d_0.Y^4
kernel void profile_l0(texture2d<float, access::read> cur [[texture(0)]],
                       texture2d<float, access::write> outC1 [[texture(1)]],
                       texture2d<float, access::write> outE0 [[texture(2)]],
                       uint2 gid [[thread_position_in_grid]]) {
  const int2 n = int2(cur.get_width(), cur.get_height());
  const ProfBlock b = prof_block(cur, int2(gid) * 2 - 2, n);
  outC1.write(float4(b.c1_00, prof_block_dead(b)), gid);
  outE0.write(prof_block_energy(b), gid);
}

// Frame difference D = cur - prev for the temporal estimator, full size,
// rgba16Float. Half precision is enough here: where nothing moved D is of
// the order of the noise, so its rounding error is 2^-11 of the noise. The
// a-trous transform is linear, so the transform of D is the difference of
// the transforms; running l0 on D costs far less than a kernel that fetches
// both frames (measured 2.4 ms against 0.2 + 0.2 ms at 4K).
//   w = 1 where the two frames are identical in all three channels
kernel void profile_diff(texture2d<float, access::read> cur [[texture(0)]],
                         texture2d<float, access::read> prev [[texture(1)]],
                         texture2d<float, access::write> dst [[texture(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
  const float3 d = cur.read(gid).xyz - prev.read(gid).xyz;
  dst.write(float4(float3(half3(d)), float(all(d == float3(0.0f)))), gid);
}

// l0 on the frame difference: as profile_l0, but a pixel is dead where the
// two frames are identical (a duplicated frame), not where d_0.Y is zero.
kernel void profile_l0_diff(texture2d<float, access::read> diff [[texture(0)]],
                            texture2d<float, access::write> outC1 [[texture(1)]],
                            texture2d<float, access::write> outE0 [[texture(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
  const int2 n = int2(diff.get_width(), diff.get_height());
  const ProfBlock b = prof_block(diff, int2(gid) * 2 - 2, n);
  outC1.write(float4(b.c1_00, (b.w_00 + b.w_10 + b.w_01 + b.w_11) * (1.0f + PROF_DEAD_ALL)), gid);
  outE0.write(prof_block_energy(b), gid);
}

// l1. Thread = one cell q = four even-lattice texels 2q + {0,1}^2 of C1.
//   outC2 (quarter)  xyz = c_2 on the 4-lattice
//   outE1 (quarter)  xyz = sum of d_1^2 over the four texels, w = dead pixels of the cell
//   outJ  (quarter)  x, y, z = gx^2, gy^2, gx*gy of the gradient of c_1.Y in the cell
//   outE0 (quarter)  E0 summed over the cell's four blocks
kernel void profile_l1(texture2d<float, access::read> C1 [[texture(0)]],
                       texture2d<float, access::write> outC2 [[texture(1)]],
                       texture2d<float, access::write> outE1 [[texture(2)]],
                       texture2d<float, access::write> outJ [[texture(3)]],
                       texture2d<float, access::read> E0 [[texture(4)]],
                       texture2d<float, access::write> outE0 [[texture(5)]],
                       uint2 gid [[thread_position_in_grid]]) {
  const int2 n = int2(C1.get_width(), C1.get_height());
  const ProfBlock b = prof_block(C1, int2(gid) * 2 - 2, n);
  const float4 e = prof_block_energy(b);
  const float gx = 0.5f * ((b.c0_10.x - b.c0_00.x) + (b.c0_11.x - b.c0_01.x));
  const float gy = 0.5f * ((b.c0_01.x - b.c0_00.x) + (b.c0_11.x - b.c0_10.x));
  const uint2 h0 = gid * 2;
  const uint2 h1 = min(h0 + 1, uint2(n) - 1);
  // (at an odd right/bottom edge the clamped texels count twice; tiles never
  // reach that far)
  const float4 e0 = E0.read(h0) + E0.read(uint2(h1.x, h0.y)) + E0.read(uint2(h0.x, h1.y)) + E0.read(h1);
  outC2.write(float4(b.c1_00, 0.0f), gid);
  outE1.write(float4(e.xyz, b.w_00 + b.w_10 + b.w_01 + b.w_11), gid);
  outJ.write(float4(gx * gx, gy * gy, gx * gy, 0.0f), gid);
  outE0.write(e0, gid);
}

// 5x5 B3 around texel c with taps `step` texels apart.
#define PROF_B3ROW(tex, j)                                                                       \
  prof_b3(prof_read(tex, c + int2(-2, j) * step, n).xyz, prof_read(tex, c + int2(-1, j) * step, n).xyz, \
          prof_read(tex, c + int2(0, j) * step, n).xyz, prof_read(tex, c + int2(1, j) * step, n).xyz,   \
          prof_read(tex, c + int2(2, j) * step, n).xyz)

static inline float3 prof_b3_5x5(texture2d<float, access::read> t, int2 c, int step, int2 n) {
  return prof_b3(PROF_B3ROW(t, -2), PROF_B3ROW(t, -1), PROF_B3ROW(t, 0), PROF_B3ROW(t, 1), PROF_B3ROW(t, 2));
}

// c_3 on the 4-lattice: B3 of C2 with adjacent taps (4 px apart).
kernel void profile_b3(texture2d<float, access::read> src [[texture(0)]],
                       texture2d<float, access::write> dst [[texture(1)]],
                       uint2 gid [[thread_position_in_grid]]) {
  const int2 n = int2(src.get_width(), src.get_height());
  dst.write(float4(prof_b3_5x5(src, int2(gid), 1, n), 0.0f), gid);
}

// l3. Thread = one cell: c_4 on the 4-lattice (taps 8 px = 2 texels of C3
// apart; not stored) and what depends on it.
//   outV2  xyz = d_2^2,  w = c_4.Y
//   outV3  xyz = d_3^2
//   outV4  xyz = (c_0 - c_4)^2 - (c_0 - c_1)^2, the part of the full-band
//          energy that is not band 0 (band 0 is measured on every pixel)
kernel void profile_l3(texture2d<float, access::read> C3 [[texture(0)]],
                       texture2d<float, access::read> C2 [[texture(1)]],
                       texture2d<float, access::read> C1 [[texture(2)]],
                       texture2d<float, access::read> cur [[texture(3)]],  // the frame, or the difference
                       texture2d<float, access::write> outV2 [[texture(4)]],
                       texture2d<float, access::write> outV3 [[texture(5)]],
                       texture2d<float, access::write> outV4 [[texture(6)]],
                       uint2 gid [[thread_position_in_grid]]) {
  const int2 n = int2(C3.get_width(), C3.get_height());
  const float3 c4 = prof_b3_5x5(C3, int2(gid), 2, n);
  const float3 c3 = C3.read(gid).xyz;
  const float3 c2 = C2.read(gid).xyz;
  const uint2 hp = min(gid * 2, uint2(C1.get_width() - 1, C1.get_height() - 1));
  const uint2 fp = min(gid * 4, uint2(cur.get_width() - 1, cur.get_height() - 1));
  const float3 c1 = C1.read(hp).xyz;
  const float3 c0 = cur.read(fp).xyz;
  const float3 d2 = c2 - c3, d3 = c3 - c4, dF = c0 - c4, d0 = c0 - c1;
  outV2.write(float4(d2 * d2, c4.x), gid);
  outV3.write(float4(d3 * d3, 0.0f), gid);
  outV4.write(float4(dF * dF - d0 * d0, 0.0f), gid);
}

// Sum of 2x2 texels of six textures. Run twice: cells (4 px) -> 8 px -> 16 px
// blocks. (One 4x4 pass, 96 fetches per thread on a small grid, measured
// 0.48 ms at 4K; two of these 0.2 ms.)
kernel void profile_sum2(texture2d<float, access::read> a0 [[texture(0)]],
                         texture2d<float, access::read> a1 [[texture(1)]],
                         texture2d<float, access::read> a2 [[texture(2)]],
                         texture2d<float, access::read> a3 [[texture(3)]],
                         texture2d<float, access::read> a4 [[texture(4)]],
                         texture2d<float, access::read> a5 [[texture(5)]],
                         texture2d<float, access::write> b0 [[texture(6)]],
                         texture2d<float, access::write> b1 [[texture(7)]],
                         texture2d<float, access::write> b2 [[texture(8)]],
                         texture2d<float, access::write> b3 [[texture(9)]],
                         texture2d<float, access::write> b4 [[texture(10)]],
                         texture2d<float, access::write> b5 [[texture(11)]],
                         uint2 gid [[thread_position_in_grid]]) {
  // Texels beyond an odd right/bottom edge count as zero; tiles never reach
  // that far.
  const uint2 n = uint2(a0.get_width(), a0.get_height());
  const uint2 p00 = gid * 2, p11 = p00 + 1;
  const float wx = p11.x < n.x ? 1.0f : 0.0f, wy = p11.y < n.y ? 1.0f : 0.0f;
  const uint2 q11 = min(p11, n - 1);
  const uint2 p10 = uint2(q11.x, p00.y), p01 = uint2(p00.x, q11.y);
#define PROF_SUM2(t) (t.read(p00) + wx * t.read(p10) + wy * t.read(p01) + wx * wy * t.read(q11))
  b0.write(PROF_SUM2(a0), gid);
  b1.write(PROF_SUM2(a1), gid);
  b2.write(PROF_SUM2(a2), gid);
  b3.write(PROF_SUM2(a3), gid);
  b4.write(PROF_SUM2(a4), gid);
  b5.write(PROF_SUM2(a5), gid);
}

// Tile record (PROFILE_RECORD_FLOATS floats; layout mirrored in profile.mm).
// A tile holds two records: spatial at 0, temporal (statistics of cur - prev)
// at PROFILE_RECORD_FLOATS.
//   0      mean of c_4.Y (the tile's luminance)
//   1..12  E[c][b] = mean of d_b,c^2, index 1 + c * 4 + b
//   13..15 EF[c]   = mean of (c_0 - c_4)^2: band 0 from every pixel, the
//          rest from the 4-lattice (so chroma-subsampling parity cannot bias it)
//   16..18 Jxx, Jyy, Jxy of the gradient of c_1.Y
//   19     fraction of dead pixels (prof_dead, the luma test)
//   20     mean of d_0.Y^4
//   21     fraction of dead pixels by the all-channel test (prof_dead_all)
//   22..23 unused (0)
#define PROFILE_RECORD_FLOATS 24
#define PROFILE_TILE_FLOATS 48

struct ProfileTileParams {
  uint originX, originY;  // pixel position of tile (0,0); multiples of 16
  uint stride;            // pixels between tile origins (16 or 32)
  uint tilesW, tilesH;
  uint base;              // 0 or PROFILE_RECORD_FLOATS
  uint pad0, pad1;
};

kernel void profile_tiles(texture2d<float, access::read> B0 [[texture(0)]],  // E0: sum d_0^2, d_0.Y^4
                          texture2d<float, access::read> B1 [[texture(1)]],  // E1: sum d_1^2, dead pixels
                          texture2d<float, access::read> B2 [[texture(2)]],  // J:  Jxx, Jyy, Jxy
                          texture2d<float, access::read> B3 [[texture(3)]],  // V2: sum d_2^2, sum c_4.Y
                          texture2d<float, access::read> B4 [[texture(4)]],  // V3: sum d_3^2
                          texture2d<float, access::read> B5 [[texture(5)]],  // V4: full-band minus band 0
                          device float* out [[buffer(0)]],
                          constant ProfileTileParams& p [[buffer(1)]],
                          uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= p.tilesW || gid.y >= p.tilesH) return;
  const uint2 b = (uint2(p.originX, p.originY) + gid * p.stride) / 16;
#define PROF_SUM4(t) (t.read(b) + t.read(b + uint2(1, 0)) + t.read(b + uint2(0, 1)) + t.read(b + uint2(1, 1)))
  const float4 s0 = PROF_SUM4(B0), s1 = PROF_SUM4(B1), s2 = PROF_SUM4(B2);
  const float4 s3 = PROF_SUM4(B3), s4 = PROF_SUM4(B4), s5 = PROF_SUM4(B5);
  device float* o = out + (gid.y * p.tilesW + gid.x) * PROFILE_TILE_FLOATS + p.base;
  const float3 e0 = s0.xyz * (1.0f / 1024.0f), e1 = s1.xyz * (1.0f / 256.0f);
  const float3 e2 = s3.xyz * (1.0f / 64.0f), e3 = s4.xyz * (1.0f / 64.0f);
  const float3 eF = max(s5.xyz * (1.0f / 64.0f) + e0, float3(0.0f));
  o[0] = s3.w * (1.0f / 64.0f);
  o[1] = e0.x;
  o[2] = e1.x;
  o[3] = e2.x;
  o[4] = e3.x;
  o[5] = e0.y;
  o[6] = e1.y;
  o[7] = e2.y;
  o[8] = e3.y;
  o[9] = e0.z;
  o[10] = e1.z;
  o[11] = e2.z;
  o[12] = e3.z;
  o[13] = eF.x;
  o[14] = eF.y;
  o[15] = eF.z;
  o[16] = s2.x;
  o[17] = s2.y;
  o[18] = s2.z;
  const float deadAll = floor(s1.w * (1.0f / PROF_DEAD_ALL));  // exact: integers below 2^24
  o[19] = (s1.w - deadAll * PROF_DEAD_ALL) * (1.0f / 1024.0f);
  o[20] = s0.w * (1.0f / 1024.0f);
  o[21] = deadAll * (1.0f / 1024.0f);
  o[22] = 0.0f;
  o[23] = 0.0f;
}

// OD_VIEW_PROFILE: tint every tile by what the profiler makes of it.
//   green  flat: its fine-band energy sits on the profiled noise floor (these
//          are the tiles a measurement of this frame uses)
//   red    structure: oriented detail, or more fine-band energy than noise
//   blue   clipped or synthetic flat (no noise to measure)
//   dimmed outside the tile grid
struct ProfileViewParams {
  uint originX, originY;
  uint stride;
  uint tilesW, tilesH;
  uint haveProfile;
  float thr0, thr1;
  float coherenceMax, deadMax;
  float pad0, pad1;
  float sig0[OD_LEVELS];  // profiled sigma of Y band 0 per knot
  float sig1[OD_LEVELS];  // Y band 1
};

kernel void profile_view(texture2d<float, access::read> src [[texture(0)]],
                         texture2d<float, access::write> dst [[texture(1)]],
                         const device float* stats [[buffer(0)]],
                         constant ProfileViewParams& p [[buffer(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
  float4 v = src.read(gid);
  // The tile whose central stride x stride part covers this pixel; the
  // outermost tiles also own their outer margin.
  const int stride = int(p.stride);
  const int half_overlap = (32 - stride) / 2;
  const int2 rel = int2(gid) - int2(int(p.originX), int(p.originY));
  const int2 tmax = int2(int(p.tilesW), int(p.tilesH)) - 1;
  const bool inside = p.tilesW > 0 && p.tilesH > 0 && all(rel >= 0) && all(rel < tmax * stride + 32);
  float2 tint = float2(0.0f);
  float keep = 0.35f;
  float gain = 1.0f;
  if (inside) {
    const int2 t = clamp((rel - half_overlap) / stride, int2(0), tmax);
    const device float* o = stats + (uint(t.y) * p.tilesW + uint(t.x)) * PROFILE_TILE_FLOATS;
    const float y = o[0], e0 = o[1], e1 = o[2];
    const float jxx = o[16], jyy = o[17], jxy = o[18], dead = o[19];
    const float tr = jxx + jyy;
    const float coh = tr > 0.0f ? ((jxx - jyy) * (jxx - jyy) + 4.0f * jxy * jxy) / (tr * tr) : 0.0f;
    const float kt = clamp(y, 0.0f, 1.0f) * float(OD_LEVELS - 1);
    const int ki = min(int(kt), OD_LEVELS - 2);
    const float s0 = mix(p.sig0[ki], p.sig0[ki + 1], kt - float(ki));
    const float s1 = mix(p.sig1[ki], p.sig1[ki + 1], kt - float(ki));
    const bool clipped = dead > p.deadMax;
    const bool flat = p.haveProfile != 0 && coh <= p.coherenceMax && e0 < p.thr0 * s0 * s0 && e1 < p.thr1 * s1 * s1;
    if (clipped) {
      tint = float2(0.12f, -0.03f);
    } else if (flat) {
      tint = float2(-0.07f, -0.07f);
    } else if (p.haveProfile != 0) {
      tint = float2(-0.03f, 0.12f);
    } else {
      keep = 1.0f;
    }
    const int2 m = (rel + stride - half_overlap) % stride;  // thin grid between tiles
    if (m.x == 0 || m.y == 0) gain = 0.85f;
  } else {
    gain = 0.4f;
  }
  dst.write(od_round_half(float4(v.x * gain, v.yz * keep * gain + tint, v.w)), gid);
}

// Direct tent-pyramid samples for the confidence-qualified coarse estimator.
// Three L3..L6 and G5..G7 sets, reduced to 20 floats per 16x16 region.
constexpr sampler coarseLinear(coord::pixel, address::clamp_to_edge, filter::linear);
kernel void profile_coarse_down(array<texture2d<float, access::sample>,3> src [[texture(0)]],
                                array<texture2d<float, access::write>,3> dst [[texture(3)]],
                                uint3 pos [[thread_position_in_grid]]) {
  uint2 p=pos.xy; uint t=pos.z;
  float2 c = float2(p) * 2.0f + 1.0f;
  float4 v = .25f * (src[t].sample(coarseLinear, c + float2(-.75f,-.75f)) +
                     src[t].sample(coarseLinear, c + float2(.75f,-.75f)) +
                     src[t].sample(coarseLinear, c + float2(-.75f,.75f)) +
                     src[t].sample(coarseLinear, c + float2(.75f,.75f)));
  dst[t].write(od_round_half(v), p);
}
// Exactly 256 threads per region (eight SIMD groups). The six basis vectors
// are orthonormal on the 16x16 lattice, so the normal matrix is the identity.
// Keep residuals in registers and return only second/fourth/cross moments.
inline float4 coarse_value(texture2d<float, access::read> g,
                           texture2d<float, access::sample> next, uint2 p, bool gaussian) {
  float4 v = g.read(p);
  return float4(v.xyz - (gaussian ? float3(0) : next.sample(coarseLinear, (float2(p)+.5f)*.5f).xyz), v.x);
}
kernel void profile_coarse_stats(
    texture2d<float, access::read> g0 [[texture(0)]],
    texture2d<float, access::sample> n0 [[texture(1)]],
    texture2d<float, access::read> g1 [[texture(2)]],
    texture2d<float, access::sample> n1 [[texture(3)]],
    texture2d<float, access::read> g2 [[texture(4)]],
    texture2d<float, access::sample> n2 [[texture(5)]],
    device float* stats [[buffer(0)]], constant uint4& cfg [[buffer(1)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  uint2 origin = 2 + group * 16;
  uint2 p = origin + uint2(tid % 16, tid / 16);
  float4 raw[3] = {coarse_value(g0,n0,p,cfg.y), coarse_value(g1,n1,p,cfg.y), coarse_value(g2,n2,p,cfg.y)};
  // Remove an exact local anchor first. This improves floating point accuracy
  // for Gaussian levels with large DC and tiny residuals without changing the fit.
  float3 r[3] = {raw[0].xyz - coarse_value(g0,n0,origin,cfg.y).xyz,
                 raw[1].xyz - coarse_value(g1,n1,origin,cfg.y).xyz,
                 raw[2].xyz - coarse_value(g2,n2,origin,cfg.y).xyz};
  float u = float(tid % 16)-7.5f, v = float(tid / 16)-7.5f;
  float basis[6] = {.0625f, u*.01355815361366601f, v*.01355815361366601f,
                   (u*u-21.25f)*.0033078507751560194f,
                   (v*v-21.25f)*.0033078507751560194f, u*v*.0029411764705882353f};
  threadgroup float3 partial[18*8];
  threadgroup float3 coef[18];
  for (uint t=0; t<3; ++t)
    for (uint f=0; f<6; ++f) {
      float3 val = simd_sum(r[t]*basis[f]);
      if (lane==0) partial[(t*6+f)*8+sg] = val;
    }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid<18) {
    float3 sum=0;
    for (uint i=0; i<8; ++i) sum += partial[tid*8+i];
    coef[tid]=sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint t=0; t<3; ++t)
    for (uint f=0; f<6; ++f) r[t] -= coef[t*6+f]*basis[f];
  float m[20];
  for (uint t=0; t<3; ++t)
    for (uint c=0; c<3; ++c) m[t*3+c] = r[t][c]*r[t][c]/250.0f;
  for (uint c=1; c<3; ++c) {
    uint j=9+(c-1)*3;
    for (uint t=0; t<3; ++t)
      for (uint z=t+1; z<3; ++z) m[j++] = r[t][c]*r[z][c]/250.0f;
  }
  m[15]=r[0].y*r[0].z/250.0f;
  m[16]=r[0].y*r[0].y*r[0].y*r[0].y/256.0f;
  m[17]=r[0].z*r[0].z*r[0].z*r[0].z/256.0f;
  m[18]=raw[0].w/256.0f;
  m[19]=0;
  for (uint t=0; t<3; ++t)
    if (!all(isfinite(raw[t].xyz)) || any(abs(raw[t].xyz)>=16)) m[19]=1;
  threadgroup float sums[20*8];
  for (uint j=0; j<20; ++j) {
    float val=simd_sum(m[j]);
    if (lane==0) sums[j*8+sg]=val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid<20) {
    float sum=0;
    for (uint i=0; i<8; ++i) sum+=sums[tid*8+i];
    stats[(cfg.x+group.y*cfg.z+group.x)*20+tid]=sum;
  }
}
