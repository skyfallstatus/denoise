// SPDX-License-Identifier: MIT
// Temporal stage, part 1: luma pyramid and motion estimation.
// Implementation: header comment of core/src/stages/temporal.mm.
//
// Luma pyramid P_k (r16f, cached per frame): P_0 = Y, P_k = D(P_(k-1)).
// Vectors live on a block grid: at level k one vector per 4x4 level-k pixels,
// in level-k pixel units, pointing from the current frame INTO the neighbour:
// current(x) matches neighbour(x + v). The matching window is the 8x8 level-k
// pixels centred on the block.
//
// All neighbours in one dispatch. These kernels work on small grids, where a
// dispatch costs a fixed 10 to 30 microseconds whatever it does, so every
// kernel takes the neighbour as the third grid dimension (thread z + U.first
// = neighbour slot a, 0 = nearest) instead of being dispatched once per
// neighbour. Per-neighbour results are stacked in one texture ("atlas"):
// neighbour a owns the rows [a * rows, (a + 1) * rows). The pyramid levels
// come as an array: P[0] is the current frame, P[a + 1] neighbour a.
//
// od: math=safe
#include "temporal_inc.metal"

struct TMotionU {
  float knots[8];   // luma noise sigma of the pyramid level of this dispatch at each luminance knot (sP_k(y))
  float penK;       // BETA: the cost per pixel of |v - v_seed|_1, and the bias against secondary seeds, is
                    // penK * s^2, s = knots at the block's own luminance (that of the current frame)
  float muK;        // lk: Lucas-Kanade damping = muK * s^2 (64)
  float voteK;      // cutstats: a block votes if var(cur) + var(ref) exceeds voteK * s^2 (16)
  float noiseK;     // cutstats: noise term of the mismatch test, noiseK * s^2 (4)
  float zeroK;      // block matching: any vector but zero costs zeroK * s^2 more (prefer "no motion" unless
                    // a match is better by more than the noise of the SSD itself)
  uint pad0, pad1, pad2;
  float inScale;    // lk: factor from the input vector to level-1 pixels
  float outScale;   // lk: factor from level-1 pixels to the stored vector
  float minVotes;   // gate: fewer voting blocks than this = no cut decision
  uint flags;       // bit 0: lk input is on the half-size grid (upsample)
  uint first;       // neighbour slot of thread z = 0
  uint rows;        // rows per neighbour in the atlas this kernel writes (and in one of the same level it reads)
  uint inRows;      // rows per neighbour in the atlas of the coarser level (bm_level), of the input vectors (lk)
  uint count;       // gate, reduce: number of neighbours
  float seedScale[T_MAX_NEIGHBOURS];  // bm_top: scale of the vector predicted from the inner neighbour (0 = none)
  int inner[T_MAX_NEIGHBOURS];        // slot of the next neighbour inwards on the same side, -1 = none
};

// Noise variance of the level at luminance y. The noise of log footage
// depends strongly on luminance (several times higher in the darks), so the
// penalties and thresholds of the motion search are taken at each block's own
// luminance, never from one figure for the whole frame.
static inline float t_var(constant TMotionU& U, float y) {
  float t = clamp(y, 0.0f, 1.0f) * 7.0f;
  int i = min(int(t), 6);
  float s = mix(U.knots[i], U.knots[i + 1], t - float(i));
  return s * s;
}

typedef array<texture2d<float, access::sample>, T_MAX_NEIGHBOURS + 1> TPyr;

constexpr sampler tGat(coord::pixel, address::clamp_to_edge, filter::linear);

// The 2x2 texels p, p + (1,0), p + (0,1), p + (1,1) in one fetch, edge-clamped
// (checked against clamped reads, inside and outside the texture).
static inline float4 t_quad(texture2d<float, access::sample> t, int2 p) {
  return t.gather(tGat, float2(p) + 1.0f).wzxy;
}

// Sum of squared differences between the 8x8 window of `cur` at oc and the
// 8x8 window of `ref` at orf. 32 fetches, everything stays in registers: a
// thread that walks a local array instead is several times slower, and a
// thread that does much more than this in sequence holds the whole dispatch
// up (measured: 0.35 ms for a 30x17 grid doing 81 candidates per thread).
static inline float t_ssd8(texture2d<float, access::sample> cur, texture2d<float, access::sample> ref, int2 oc,
                           int2 orf) {
  float4 acc = float4(0.0f);
  for (int y = 0; y < 8; y += 2)
    for (int x = 0; x < 8; x += 2) {
      float4 d = t_quad(ref, orf + int2(x, y)) - t_quad(cur, oc + int2(x, y));
      acc += d * d;
    }
  return acc.x + acc.y + acc.z + acc.w;
}

// Cost of vector v for the 8x8 window at oc: mean squared difference, plus
// `zero` unless v is the zero vector.
static inline float t_cost8(texture2d<float, access::sample> cur, texture2d<float, access::sample> ref, int2 oc,
                            int2 v, float zero) {
  float c = t_ssd8(cur, ref, oc, oc + v) * (1.0f / 64.0f);
  return any(v != int2(0)) ? c + zero : c;
}

// P_k = D(P_(k-1)). The source is either the working texture (Y in .x) or an
// r16f level.
kernel void temporal_pyr_down(texture2d<float, access::sample> src [[texture(0)]],
                              texture2d<float, access::write> dst [[texture(1)]],
                              uint2 gid [[thread_position_in_grid]]) {
  dst.write(od_round_half(float4(t_down(src, gid).x, 0.0f, 0.0f, 0.0f)), gid);
}

// Top level, full search +-4, one thread per block and candidate: the grid is
// 9 x 9 times the block grid. x = cost of the candidate around the zero
// vector, y = cost of the same candidate around the predicted vector, zw =
// the predicted vector (so that the pick need not read the vector atlas it
// writes). The prediction is the inner neighbour's top-level vector, so a
// dispatch holds only neighbours whose inner neighbour is already done; the
// cost atlas has the rows of that dispatch only (z, not the slot).
kernel void temporal_bm_top_cost(TPyr P [[texture(0)]],
                                 texture2d<float, access::read> flowTop [[texture(11)]],
                                 texture2d<float, access::write> cost [[texture(12)]],
                                 constant TMotionU& U [[buffer(0)]],
                                 uint3 gid [[thread_position_in_grid]]) {
  const uint a = U.first + gid.z;
  const uint2 blk = gid.xy / 9u;
  const int2 dv = int2(gid.xy % 9u) - 4;
  const int2 o = int2(blk) * 4 - 2;
  // Luminance of the block: the 2x2 texels at the centre of its window.
  const float4 lq = t_quad(P[0], o + 3);
  const float var = t_var(U, 0.25f * (lq.x + lq.y + lq.z + lq.w));
  const float penS = U.penK * var, zero = U.zeroK * var;
  const float pen = penS * float(abs(dv.x) + abs(dv.y));
  float c0 = t_cost8(P[0], P[a + 1u], o, dv, zero) + pen;
  float c1 = 3.0e38f;
  int2 sp = int2(0);
  if (U.seedScale[a] != 0.0f) {
    sp = int2(round(flowTop.read(uint2(blk.x, blk.y + uint(U.inner[a]) * U.rows)).xy * U.seedScale[a]));
    if (any(sp != int2(0))) c1 = t_cost8(P[0], P[a + 1u], o, sp + dv, zero) + pen + penS;
  }
  cost.write(float4(c0, c1, float2(sp)), uint2(gid.x, gid.y + gid.z * U.rows * 9u));
}

kernel void temporal_bm_top_pick(texture2d<float, access::read> cost [[texture(11)]],
                                 texture2d<float, access::write> flow [[texture(12)]],
                                 constant TMotionU& U [[buffer(0)]],
                                 uint3 gid [[thread_position_in_grid]]) {
  const uint2 base = uint2(gid.x, gid.y + gid.z * U.rows) * 9u;
  const int2 sp = int2(cost.read(base).zw);
  float best = 3.0e38f;
  int2 bv = int2(0);
  for (int dy = 0; dy < 9; ++dy)
    for (int dx = 0; dx < 9; ++dx) {
      float2 c = cost.read(base + uint2(dx, dy)).xy;
      if (c.x < best) {
        best = c.x;
        bv = int2(dx, dy) - 4;
      }
      if (c.y < best) {
        best = c.y;
        bv = sp + int2(dx, dy) - 4;
      }
    }
  flow.write(float4(float2(bv), 0.0f, 0.0f), uint2(gid.x, gid.y + (U.first + gid.z) * U.rows));
}

// Levels below the top: seeds are twice the vector of the 3 nearest parent
// blocks and the zero vector; the best seed is refined by +-1.
kernel void temporal_bm_level(TPyr P [[texture(0)]],
                              texture2d<float, access::read> parent [[texture(11)]],
                              texture2d<float, access::write> flow [[texture(12)]],
                              constant TMotionU& U [[buffer(0)]],
                              uint3 gid3 [[thread_position_in_grid]]) {
  const uint a = U.first + gid3.z;
  const uint2 gid = gid3.xy;
  const texture2d<float, access::sample> cur = P[0], ref = P[a + 1u];
  const int2 phi = int2(int(parent.get_width()) - 1, int(U.inRows) - 1);
  const int2 prow = int2(0, int(a * U.inRows));
  const int2 o = int2(gid) * 4 - 2;
  const float4 lq = t_quad(cur, o + 3);  // luminance of the block, as in temporal_bm_top_cost
  const float var = t_var(U, 0.25f * (lq.x + lq.y + lq.z + lq.w));
  const float pen = U.penK * var, zero = U.zeroK * var;

  const int2 gp = min(int2(gid) >> 1, phi);
  const int2 side = int2(int(gid.x & 1u) * 2 - 1, int(gid.y & 1u) * 2 - 1);
  const int2 nx = clamp(int2(gp.x + side.x, gp.y), int2(0), phi);
  const int2 ny = clamp(int2(gp.x, gp.y + side.y), int2(0), phi);
  const int2 s0 = int2(round(parent.read(uint2(gp + prow)).xy * 2.0f));
  const int2 s1 = int2(round(parent.read(uint2(nx + prow)).xy * 2.0f));
  const int2 s2 = int2(round(parent.read(uint2(ny + prow)).xy * 2.0f));

  int2 seed = s0;
  float cs = 3.0e38f;  // cost of the chosen seed if it was evaluated here
  if (any(s1 != s0)) {
    cs = t_cost8(cur, ref, o, s0, zero);
    float c = t_cost8(cur, ref, o, s1, zero);
    if (c + pen < cs) {
      cs = c;
      seed = s1;
    }
  }
  if (any(s2 != s0) && any(s2 != s1)) {
    if (cs > 1.0e38f) cs = t_cost8(cur, ref, o, s0, zero);
    float c = t_cost8(cur, ref, o, s2, zero);
    if (c + pen < cs) {
      cs = c;
      seed = s2;
    }
  }
  if (any(s0 != int2(0)) && any(s1 != int2(0)) && any(s2 != int2(0))) {
    if (cs > 1.0e38f) cs = t_cost8(cur, ref, o, s0, zero);
    float c = t_cost8(cur, ref, o, int2(0), zero);
    if (c + pen < cs) {
      cs = c;
      seed = int2(0);
    }
  }

  // The seed and its 8 neighbours in one sweep: per 2x2 quad of the current
  // window, the 4x4 texels of the neighbour frame around it (5 fetches) feed
  // nine accumulators.
  float acc[9];
  for (int i = 0; i < 9; ++i) acc[i] = 0.0f;
  const int2 b = o + seed - 1;
  for (int y = 0; y < 8; y += 2)
    for (int x = 0; x < 8; x += 2) {
      const float4 c = t_quad(cur, o + int2(x, y));
      const float4 q00 = t_quad(ref, b + int2(x, y)), q20 = t_quad(ref, b + int2(x + 2, y));
      const float4 q02 = t_quad(ref, b + int2(x, y + 2)), q22 = t_quad(ref, b + int2(x + 2, y + 2));
      const float4 row[4] = {float4(q00.x, q00.y, q20.x, q20.y), float4(q00.z, q00.w, q20.z, q20.w),
                             float4(q02.x, q02.y, q22.x, q22.y), float4(q02.z, q02.w, q22.z, q22.w)};
      for (int dy = 0; dy < 3; ++dy)
        for (int dx = 0; dx < 3; ++dx) {
          float4 d = float4(row[dy][dx], row[dy][dx + 1], row[dy + 1][dx], row[dy + 1][dx + 1]) - c;
          acc[dy * 3 + dx] += dot(d, d);
        }
    }
  float best = 3.0e38f;
  int2 bv = seed;
  for (int dy = 0; dy < 3; ++dy)
    for (int dx = 0; dx < 3; ++dx) {
      const int2 v = seed + int2(dx, dy) - 1;
      float c = acc[dy * 3 + dx] * (1.0f / 64.0f) + pen * float(abs(dx - 1) + abs(dy - 1));
      if (any(v != int2(0))) c += zero;
      if (c < best) {
        best = c;
        bv = seed + int2(dx, dy) - 1;
      }
    }
  flow.write(float4(float2(bv), 0.0f, 0.0f), uint2(gid.x, gid.y + a * U.rows));
}

// One damped Lucas-Kanade step on the level-1 block grid (inverse
// compositional: gradients from the current frame). Writes the refined vector
// times outScale. (A variant that fetched the window with gathers and kept it
// in registers measured slower, 0.076 against 0.061 ms on 480x270 blocks.)
kernel void temporal_lk(TPyr P [[texture(0)]],
                        texture2d<float, access::sample> flowIn [[texture(11)]],
                        texture2d<float, access::write> flow [[texture(12)]],
                        constant TMotionU& U [[buffer(0)]],
                        uint3 gid3 [[thread_position_in_grid]]) {
  const uint slot = U.first + gid3.z;
  const uint2 gid = gid3.xy;
  const texture2d<float, access::sample> cur = P[0], ref = P[slot + 1u];
  const int2 hi = int2(int(cur.get_width()) - 1, int(cur.get_height()) - 1);
  const int2 o = int2(gid) * 4 - 2;
  float2 v;
  if (U.flags & 1u) v = t_atlas(flowIn, (float2(gid) + 0.5f) * 0.5f, slot, U.inRows).xy * U.inScale;
  else v = flowIn.read(uint2(gid.x, gid.y + slot * U.inRows)).xy * U.inScale;

  float c[100];
  for (int y = 0; y < 10; ++y)
    for (int x = 0; x < 10; ++x) c[y * 10 + x] = cur.read(uint2(clamp(o + int2(x - 1, y - 1), int2(0), hi))).x;

  float gxx = 0.0f, gyy = 0.0f, gxy = 0.0f, bx = 0.0f, by = 0.0f, sum = 0.0f;
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x) {
      int i = (y + 1) * 10 + (x + 1);
      sum += c[i];
      float gx = 0.5f * (c[i + 1] - c[i - 1]);
      float gy = 0.5f * (c[i + 10] - c[i - 10]);
      float e = ref.sample(tLin, float2(o + int2(x, y)) + 0.5f + v).x - c[i];
      gxx += gx * gx;
      gyy += gy * gy;
      gxy += gx * gy;
      bx -= gx * e;
      by -= gy * e;
    }
  const float mu = max(U.muK * t_var(U, sum * (1.0f / 64.0f)), 1.0e-12f);
  const float a = gxx + mu, d = gyy + mu;
  const float det = a * d - gxy * gxy;  // >= mu^2 > 0
  float2 dv = float2(d * bx - gxy * by, a * by - gxy * bx) / det;
  dv = clamp(dv, float2(-1.0f), float2(1.0f));
  flow.write(float4((v + dv) * U.outScale, 0.0f, 0.0f), uint2(gid.x, gid.y + slot * U.rows));
}

// Per block, after matching at the statistics level:
//   x = 1 if the block is textured enough to say anything (it "votes")
//   y = 1 if it votes and the best match is still unrelated content
//   z = sum of |neighbour - inner neighbour| over the window, undisplaced
//       (zero over the whole frame = the neighbour may repeat the inner
//       frame: temporal_repeat decides)
// "Unrelated" uses the zero-mean SSD relative to the two window variances, so
// a brightness change between the frames is not taken for a cut.
kernel void temporal_cutstats(TPyr P [[texture(0)]],
                              texture2d<float, access::read> flowIn [[texture(11)]],
                              texture2d<float, access::write> stats [[texture(12)]],
                              constant TMotionU& U [[buffer(0)]],
                              uint3 gid3 [[thread_position_in_grid]]) {
  const uint a = U.first + gid3.z;
  const uint2 gid = uint2(gid3.x, gid3.y + a * U.rows);
  const texture2d<float, access::sample> cur = P[0], ref = P[a + 1u], inner = P[U.inner[a] + 1];
  const int2 o = int2(gid3.xy) * 4 - 2;
  const int2 v = int2(round(flowIn.read(gid).xy));
  // Sums are taken relative to one texel of the window: the variances are tiny
  // against the values themselves.
  const float base = t_quad(cur, o).x;
  float4 sc = float4(0.0f), scc = float4(0.0f), sr = float4(0.0f), srr = float4(0.0f), sdd = float4(0.0f);
  float4 sad = float4(0.0f);
  for (int y = 0; y < 8; y += 2)
    for (int x = 0; x < 8; x += 2) {
      const int2 p = o + int2(x, y);
      const float4 c = t_quad(cur, p) - base;
      const float4 r = t_quad(ref, p + v) - base;
      const float4 d = r - c;
      sc += c;
      scc += c * c;
      sr += r;
      srr += r * r;
      sdd += d * d;
      sad += abs(t_quad(ref, p) - t_quad(inner, p));
    }
  const float k = 1.0f / 64.0f;
  const float mc = (sc.x + sc.y + sc.z + sc.w) * k, mr = (sr.x + sr.y + sr.z + sr.w) * k;
  float varc = max((scc.x + scc.y + scc.z + scc.w) * k - mc * mc, 0.0f);
  float varr = max((srr.x + srr.y + srr.z + srr.w) * k - mr * mr, 0.0f);
  float md = mr - mc;
  float zssd = max((sdd.x + sdd.y + sdd.z + sdd.w) * k - md * md, 0.0f);
  const float nv = t_var(U, mc + base);  // noise variance at the block's luminance
  float vote = (varc + varr > U.voteK * nv) ? 1.0f : 0.0f;
  float mism = (zssd > 0.5f * (varc + varr + U.noiseK * nv)) ? vote : 0.0f;
  stats.write(float4(vote, mism, sad.x + sad.y + sad.z + sad.w, 0.0f), gid);
}

// Sum over 8x8 texels (rgba32f) within each neighbour's rows: applied
// repeatedly down to one texel per neighbour. U.inRows and U.rows are the
// rows per neighbour of the source and of the result.
kernel void temporal_reduce(texture2d<float, access::read> src [[texture(11)]],
                            texture2d<float, access::write> dst [[texture(12)]],
                            constant TMotionU& U [[buffer(0)]],
                            uint3 gid [[thread_position_in_grid]]) {
  const uint2 size = uint2(src.get_width(), U.inRows);
  float4 s = float4(0.0f);
  for (uint y = 0; y < 8u; ++y)
    for (uint x = 0; x < 8u; ++x) {
      uint2 p = gid.xy * 8u + uint2(x, y);
      if (p.x < size.x && p.y < size.y) s += src.read(uint2(p.x, p.y + gid.z * U.inRows));
    }
  dst.write(s, uint2(gid.x, gid.y + gid.z * U.rows));
}

// Exact repeats. The statistics' z (luma of the statistics level, a quarter or an eighth of
// the resolution) is only a candidate test: equal there, a neighbour may still differ from
// its inner neighbour in chroma, or in full- resolution luma. Candidates (z == 0) are
// compared texel by texel, Y, Cb and Cr bits, at full resolution: temporal_repeat_arm (one
// thread) lists the candidates and writes the indirect dispatch arguments of temporal_repeat
// (zero groups when there are none: ordinary footage launches nothing) and clears their
// flags; temporal_repeat sets a candidate's flag if any texel differs; temporal_gate drops a
// neighbour only if z == 0 and its flag is clear. Buffer `rep` (uint words): 0..2 indirect
// arguments (threadgroups x, y, z), 4..4+T_MAX_NEIGHBOURS-1 the candidate slots, 16.. one
// flag per slot.
constant uint kRepList = 4u, kRepFlag = 16u;
constant uint kRepTexels = 4u;  // texels per thread along x (8: no faster)

struct TRepeatU {
  uint groupsX, groupsY;  // threadgroups covering the frame (kRepTexels texels per thread along x)
  uint pad0, pad1;
};

kernel void temporal_repeat_arm(texture2d<float, access::read> stats [[texture(11)]],
                                device uint* rep [[buffer(1)]],
                                constant TMotionU& U [[buffer(0)]],
                                constant TRepeatU& R [[buffer(2)]],
                                uint tid [[thread_position_in_grid]]) {
  if (tid != 0) return;
  uint n = 0;
  for (uint a = 0; a < U.count; ++a) {
    rep[kRepFlag + a] = 0u;
    if (stats.read(uint2(0u, a)).z == 0.0f) rep[kRepList + n++] = a;
  }
  rep[0] = n ? R.groupsX : 0u;
  rep[1] = n ? R.groupsY : 0u;
  rep[2] = n;
}

typedef array<texture2d<half, access::read>, T_MAX_NEIGHBOURS + 1> TFrames;

kernel void temporal_repeat(TFrames F [[texture(0)]],
                            device uint* rep [[buffer(1)]],
                            constant TMotionU& U [[buffer(0)]],
                            uint3 gid [[thread_position_in_grid]]) {
  const uint a = simd_broadcast_first(rep[kRepList + gid.z]);  // uniform: the textures are indexed by it
  const texture2d<half, access::read> ref = F[a + 1u], inner = F[U.inner[a] + 1];
  const uint w = ref.get_width(), y = gid.y, x0 = gid.x * kRepTexels;
  bool differs = false;
  if (y < ref.get_height()) {
    for (uint x = x0; x < min(x0 + kRepTexels, w); ++x) {
      const ushort4 r = as_type<ushort4>(ref.read(uint2(x, y))), q = as_type<ushort4>(inner.read(uint2(x, y)));
      differs = differs || any(r.xyz != q.xyz);
    }
  }
  if (simd_any(differs) && simd_is_first())
    atomic_store_explicit((device atomic_uint*)(rep + kRepFlag + a), 1u, memory_order_relaxed);
}

// One thread: the global gate of every neighbour, nearest first, one texel
// each. x = weight of the neighbour (0 = dropped), y = scene-cut value, which
// is handed on to the neighbours beyond this one.
kernel void temporal_gate(texture2d<float, access::read> stats [[texture(11)]],
                          texture2d<float, access::write> gate [[texture(12)]],
                          constant TMotionU& U [[buffer(0)]],
                          const device uint* rep [[buffer(1)]]) {
  float cuts[T_MAX_NEIGHBOURS];
  for (uint a = 0; a < U.count; ++a) {
    float4 s = stats.read(uint2(0u, a));
    float cut = (s.x >= U.minVotes) ? smoothstep(0.35f, 0.6f, s.y / max(s.x, 1.0f)) : 0.0f;
    if (U.inner[a] >= 0) cut = max(cut, cuts[U.inner[a]]);
    cuts[a] = cut;
    float dup = (s.z == 0.0f && rep[kRepFlag + a] == 0u) ? 1.0f : 0.0f;
    gate.write(float4((1.0f - cut) * (1.0f - dup), cut, 0.0f, 0.0f), uint2(0u, a));
  }
}
