// SPDX-License-Identifier: MIT
// Temporal stage, part 2: motion compensation and the robust
// band-wise merge. Implementation: header comment of
// core/src/stages/temporal.mm.
//
// Per neighbour j:  delta_j = warp(F_j) - F_0 and its validity v_j (0 where
// the warp leaves the frame), and their tent pyramids G_b. Per level b:
//   beta  = G_b[delta] - U(G_(b+1)[delta])   (the top level uses G_b itself)
//   e     = beta^2 / (2 sigma_b^2)       temporal_energy
//   q     = [1 2 1]^2 average of e       expectation 1 for noise alone
//   r     = max(ss(t0, t1, qY), 0.75 U(r_(b+1)), strong)   (luma's t0 doubled at level 0)
//   rc    = max(r, ss(t0c, t1c, qC))                         (chroma: t0, t1 of the other levels)
//   strong = max(ss(s0, s1, qY), U(strong_(b+1)))      a mismatch far beyond noise
//   a     = gate * G_b[v] * (1 - r)      weight of this neighbour in this band
//   M_b   = sum a beta / (1 + sum a)     Delta_b = M_b + U(Delta_(b+1))
//
// Layout. Neighbours are handled four at a time, and a group's data is stored
// by channel, not by neighbour: dY, dCb and dCr hold that component of the
// delta of neighbour j in colour channel j (r16f, rg16f or rgba16f for 1, 2
// or more neighbours), dV its validity (8-bit: it is 1 except within a pixel
// of where the warp leaves the frame). The same goes for the energies and the
// rejection. Every kernel therefore works on float4 = "all neighbours" and
// reads each texture once. The stage is limited by memory traffic, not by
// arithmetic (see temporal.mm): what counts is the number of bytes a
// pass reads and writes.
//
// Order: warp; the delta pyramid; then per level, from the top down, energy
// and merge. More than four neighbours accumulate through temporal_merge_acc.
//
// od: math=fast
#include "temporal_inc.metal"

struct TWarpU {
  float2 size;  // frame size in pixels
  uint taps;    // 9 = Catmull-Rom, 5 = Catmull-Rom without the corner taps, 1 = bilinear
  uint n;       // neighbours in this group (1..4)
  uint first;   // slot of the group's first neighbour in the vector atlas
  uint rows;    // rows per neighbour in the vector atlas
  float bias;   // added to every delta before it is stored (TMergeU)
  uint pad0;
};

struct TMergeU {
  float4 knots[8];  // xyz = sigma of this level for Y, Cb, Cr at each luminance knot
  float4 mask;      // 1 in the channels that hold a neighbour of this group, else 0
  float t0, t1;     // rejection thresholds on the averaged normalised energy
  float gateK;      // weight of the coarser level's rejection (0 = no gating)
  uint hasPrev;     // an earlier group left sums in accIn
  uint isTop;       // top level: beta = G_b
  uint lumUp;       // luminance comes from the next coarser level (level 0)
  float aY, aC;     // amounts applied to the luma / chroma correction (level 0)
  uint first;       // slot of the group's first neighbour in the gate atlas
  float bias;       // the delta textures of this level hold delta + bias
  float biasUp;     // the same for the next coarser level
  float keep;       // level 0, amounts applied here: luma grain floor (stage.h kGrainFloor; knots[].w =
                    // full-band luma sigma). 0 = no floor (the spatial stage applies the amounts)
  float s0, s1;     // strong mismatch: a band energy above s0..s1 rejects the neighbour fully at this level
                    // and at every finer one (rOut then carries r + strong, 0..2)
  float t0c, t1c;   // the chroma test's thresholds: t0, t1 of the other levels (only luma's t0 is raised at level 0)
};

struct TViewU {
  float invCount;  // 1 / (frames - 1)
  uint rows;       // rows per neighbour in the vector atlas
  uint pad0, pad1;
};

typedef texture2d<float, access::sample> TTex;

typedef array<texture2d<float, access::sample>, 4> TTex4;

// Catmull-Rom through bilinear fetches. q is in pixel coordinates.
static inline float4 t_catrom(TTex t, float2 q, uint taps) {
  if (taps == 1u) return t.sample(tLin, q);
  float2 c = floor(q - 0.5f) + 0.5f;
  float2 f = q - c;
  float2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
  float2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
  float2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
  float2 w3 = f * f * (-0.5f + 0.5f * f);
  float2 w12 = w1 + w2;
  float2 p0 = c - 1.0f, p3 = c + 2.0f, p12 = c + w2 / w12;
  float4 s = t.sample(tLin, float2(p12.x, p12.y)) * (w12.x * w12.y) +
             t.sample(tLin, float2(p0.x, p12.y)) * (w0.x * w12.y) +
             t.sample(tLin, float2(p3.x, p12.y)) * (w3.x * w12.y) +
             t.sample(tLin, float2(p12.x, p0.y)) * (w12.x * w0.y) +
             t.sample(tLin, float2(p12.x, p3.y)) * (w12.x * w3.y);
  if (taps == 5u) {
    float wsum = w12.x * w12.y + (w0.x + w3.x) * w12.y + w12.x * (w0.y + w3.y);
    return s / wsum;
  }
  s += t.sample(tLin, float2(p0.x, p0.y)) * (w0.x * w0.y) + t.sample(tLin, float2(p3.x, p0.y)) * (w3.x * w0.y) +
       t.sample(tLin, float2(p0.x, p3.y)) * (w0.x * w3.y) + t.sample(tLin, float2(p3.x, p3.y)) * (w3.x * w3.y);
  return s;
}

// All neighbours of the group in one pass: delta_j = neighbour_j(x + V_j(x)) -
// current(x) by component, and the validity of the warp (without the
// neighbour's global gate, which the merge multiplies in).
kernel void temporal_warp(texture2d<float, access::read> cur [[texture(0)]],
                          TTex4 nb [[texture(1)]],
                          TTex flow [[texture(5)]],
                          texture2d<float, access::write> dY [[texture(6)]],
                          texture2d<float, access::write> dCb [[texture(7)]],
                          texture2d<float, access::write> dCr [[texture(8)]],
                          texture2d<float, access::write> dV [[texture(9)]],
                          constant TWarpU& U [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
  const float2 p = float2(gid) + 0.5f;
  const float3 c = cur.read(gid).xyz;
  float4 y = float4(0.0f), cb = float4(0.0f), cr = float4(0.0f), v = float4(0.0f);
  for (uint j = 0; j < U.n; ++j) {
    float2 q = p + t_atlas(flow, p * 0.125f, U.first + j, U.rows).xy;
    float edge = min(min(q.x, U.size.x - q.x), min(q.y, U.size.y - q.y));
    float3 s = t_catrom(nb[j], q, U.taps).xyz - c;
    y[j] = s.x;
    cb[j] = s.y;
    cr[j] = s.z;
    v[j] = saturate(edge + 0.5f);
  }
  dY.write(od_store_half(y + U.bias), gid);
  dCb.write(od_store_half(cb + U.bias), gid);
  dCr.write(od_store_half(cr + U.bias), gid);
  dV.write(v, gid);
}

// The next level of the delta pyramid of the whole group.
kernel void temporal_down(TTex dY [[texture(0)]],
                          TTex dCb [[texture(1)]],
                          TTex dCr [[texture(2)]],
                          TTex dV [[texture(3)]],
                          texture2d<float, access::write> nY [[texture(4)]],
                          texture2d<float, access::write> nCb [[texture(5)]],
                          texture2d<float, access::write> nCr [[texture(6)]],
                          texture2d<float, access::write> nV [[texture(7)]],
                          constant TMergeU& U [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
  // U is that of the level read: bias -> biasUp
  const float rebias = U.biasUp - U.bias;
  nY.write(od_store_half(t_down(dY, gid) + rebias), gid);
  nCb.write(od_store_half(t_down(dCb, gid) + rebias), gid);
  nCr.write(od_store_half(t_down(dCr, gid) + rebias), gid);
  nV.write(t_down(dV, gid), gid);
}

// Normalised band energy of every neighbour of the group. d* = this level of
// the delta pyramid, u* = the next coarser one (not read at the top level).
kernel void temporal_energy(TTex dY [[texture(0)]],
                            TTex dCb [[texture(1)]],
                            TTex dCr [[texture(2)]],
                            TTex uY [[texture(3)]],
                            TTex uCb [[texture(4)]],
                            TTex uCr [[texture(5)]],
                            TTex lum [[texture(6)]],
                            texture2d<float, access::write> eY [[texture(7)]],
                            texture2d<float, access::write> eC [[texture(8)]],
                            constant TMergeU& U [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
  float y = U.lumUp ? t_up(lum, gid).x : lum.read(gid).x;
  float3 s = t_sigma(U.knots, y);
  float3 inv = 1.0f / (2.0f * s * s + T_SIGMA_Q2);
  float4 bY = dY.read(gid), bCb = dCb.read(gid), bCr = dCr.read(gid);
  if (U.isTop) {
    bY -= U.bias;
    bCb -= U.bias;
    bCr -= U.bias;
  } else {
    const float rebias = U.bias - U.biasUp;
    bY -= t_up(uY, gid) + rebias;
    bCb -= t_up(uCb, gid) + rebias;
    bCr -= t_up(uCr, gid) + rebias;
  }
  eY.write(min(bY * bY * inv.x, 1000.0f), gid);
  eC.write(min(0.5f * (bCb * bCb * inv.y + bCr * bCr * inv.z), 1000.0f), gid);
}

struct TSums {
  float3 n;  // sum a * beta (Y, Cb, Cr)
  float sy;  // sum aY
  float sc;  // sum aC
  float qy;  // sum aY^2
  float4 r;  // luma rejection of each neighbour plus its strong-mismatch part (for the next finer level)
};

// Textures 0..10 of the three merge kernels.
#define T_MERGE_INPUTS                                                                  \
  TTex dY [[texture(0)]], TTex dCb [[texture(1)]], TTex dCr [[texture(2)]], TTex dV [[texture(3)]], \
      TTex uY [[texture(4)]], TTex uCb [[texture(5)]], TTex uCr [[texture(6)]],         \
      TTex eY [[texture(7)]], TTex eC [[texture(8)]], TTex rUp [[texture(9)]],                      \
      texture2d<float, access::read> gate [[texture(10)]]
#define T_MERGE_ARGS dY, dCb, dCr, dV, uY, uCb, uCr, eY, eC, rUp, gate

// d* = this level of the delta pyramid, u* = the next coarser one.
static inline TSums t_merge(TTex dY, TTex dCb, TTex dCr, TTex dV, TTex uY, TTex uCb, TTex uCr, TTex eY, TTex eC,
                            TTex rUp, texture2d<float, access::read> gate, constant TMergeU& U, uint2 gid) {
  float2 p = float2(gid) + 0.5f;
  float4 qy = 0.25f * (eY.sample(tLin, p + float2(-0.5f, -0.5f)) + eY.sample(tLin, p + float2(0.5f, -0.5f)) +
                       eY.sample(tLin, p + float2(-0.5f, 0.5f)) + eY.sample(tLin, p + float2(0.5f, 0.5f)));
  float4 qc = 0.25f * (eC.sample(tLin, p + float2(-0.5f, -0.5f)) + eC.sample(tLin, p + float2(0.5f, -0.5f)) +
                       eC.sample(tLin, p + float2(-0.5f, 0.5f)) + eC.sample(tLin, p + float2(0.5f, 0.5f)));
  float4 r = smoothstep(U.t0, U.t1, qy);
  // A coarser level hands its rejection down at gateK, and a strong mismatch
  // (far beyond any noise: a bright edge out of place) in full. s0 >= t1, so
  // strong > 0 implies r = 1 at the level that measured it.
  float4 strong = smoothstep(U.s0, U.s1, qy);
  if (U.gateK > 0.0f) {
    const float4 ru = t_up(rUp, gid);
    strong = max(strong, saturate(ru - 1.0f));
    r = max(r, max(U.gateK * min(ru, 1.0f), strong));
  }
  float4 rc = max(r, smoothstep(U.t0c, U.t1c, qc));
  float4 bY = dY.read(gid), bCb = dCb.read(gid), bCr = dCr.read(gid);
  if (U.isTop) {
    bY -= U.bias;
    bCb -= U.bias;
    bCr -= U.bias;
  } else {
    const float rebias = U.bias - U.biasUp;
    bY -= t_up(uY, gid) + rebias;
    bCb -= t_up(uCb, gid) + rebias;
    bCr -= t_up(uCr, gid) + rebias;
  }
  // Channels past the group's last neighbour are masked: any row will do.
  const uint gl = gate.get_height() - 1u;
  const float4 g = float4(gate.read(uint2(0u, U.first)).x, gate.read(uint2(0u, min(U.first + 1u, gl))).x,
                          gate.read(uint2(0u, min(U.first + 2u, gl))).x, gate.read(uint2(0u, min(U.first + 3u, gl))).x);
  const float4 v = dV.read(gid) * g * U.mask;
  const float4 ay = v * (1.0f - r), ac = v * (1.0f - rc);
  TSums o;
  o.n = float3(dot(ay, bY), dot(ac, bCb), dot(ac, bCr));
  o.sy = ay.x + ay.y + ay.z + ay.w;
  o.sc = ac.x + ac.y + ac.z + ac.w;
  o.qy = dot(ay, ay);
  o.r = r + strong;
  return o;
}

// A group that is not the last one: add to the running sums.
//   acc1 = (N_Y, N_Cb, N_Cr, S_Y)   acc2 = (S_C, Q_Y, 0, 0)
kernel void temporal_merge_acc(T_MERGE_INPUTS,
                               texture2d<float, access::read> acc1In [[texture(11)]],
                               texture2d<float, access::read> acc2In [[texture(12)]],
                               texture2d<float, access::write> acc1 [[texture(13)]],
                               texture2d<float, access::write> acc2 [[texture(14)]],
                               texture2d<float, access::write> rOut [[texture(15)]],
                               constant TMergeU& U [[buffer(0)]],
                               uint2 gid [[thread_position_in_grid]]) {
  TSums s = t_merge(T_MERGE_ARGS, U, gid);
  float4 a1 = float4(s.n, s.sy), a2 = float4(s.sc, s.qy, 0.0f, 0.0f);
  if (U.hasPrev) {
    a1 += acc1In.read(gid);
    a2 += acc2In.read(gid);
  }
  acc1.write(od_store_half(a1), gid);
  acc2.write(od_round_half(a2), gid);
  rOut.write(s.r, gid);
}

// Last group, levels above 0: normalise and collapse.
kernel void temporal_merge_fin(T_MERGE_INPUTS,
                               texture2d<float, access::read> acc1In [[texture(11)]],
                               texture2d<float, access::read> acc2In [[texture(12)]],
                               TTex dUp [[texture(13)]],
                               texture2d<float, access::write> dOut [[texture(14)]],
                               texture2d<float, access::write> rOut [[texture(15)]],
                               constant TMergeU& U [[buffer(0)]],
                               uint2 gid [[thread_position_in_grid]]) {
  TSums s = t_merge(T_MERGE_ARGS, U, gid);
  if (U.hasPrev) {
    float4 a1 = acc1In.read(gid), a2 = acc2In.read(gid);
    s.n += a1.xyz;
    s.sy += a1.w;
    s.sc += a2.x;
  }
  float3 m = float3(s.n.x / (1.0f + s.sy), s.n.yz / (1.0f + s.sc));
  if (!U.isTop) m += t_up(dUp, gid).xyz;
  dOut.write(od_store_half(float4(m, 0.0f)), gid);
  rOut.write(s.r, gid);
}

// Last group, level 0: the merged frame and the effective frame count.
kernel void temporal_merge_fin0(T_MERGE_INPUTS,
                                texture2d<float, access::read> acc1In [[texture(11)]],
                                texture2d<float, access::read> acc2In [[texture(12)]],
                                TTex dUp [[texture(13)]],
                                texture2d<float, access::read> cur [[texture(14)]],
                                texture2d<float, access::write> image [[texture(15)]],
                                texture2d<float, access::write> count [[texture(16)]],
                                constant TMergeU& U [[buffer(0)]],
                                uint2 gid [[thread_position_in_grid]]) {
  TSums s = t_merge(T_MERGE_ARGS, U, gid);
  if (U.hasPrev) {
    float4 a1 = acc1In.read(gid), a2 = acc2In.read(gid);
    s.n += a1.xyz;
    s.sy += a1.w;
    s.sc += a2.x;
    s.qy += a2.y;
  }
  float3 m = float3(s.n.x / (1.0f + s.sy), s.n.yz / (1.0f + s.sc)) + t_up(dUp, gid).xyz;
  float4 c = cur.read(gid);
  float aY = U.aY;
  if (U.keep > 0.0f) {
    const float t = clamp(c.x, 0.0f, 1.0f) * 7.0f;
    const int i = min(int(t), 6);
    aY = od_luma_amount(aY, U.keep, mix(U.knots[i].w, U.knots[i + 1].w, t - float(i)));
  }
  // A zero correction leaves the pixel bit-identical (c + 0 == c).
  image.write(od_store_half(float4(c.xyz + float3(aY, U.aC, U.aC) * m, c.w)), gid);
  count.write(float4((1.0f + s.sy) * (1.0f + s.sy) / (1.0f + s.qy), 0.0f, 0.0f, 0.0f), gid);
}

// OD_VIEW_MOTION: Y = share of the neighbours merged, Cb/Cr = the vector of
// the nearest neighbour (+-0.5 at 32 pixels).
kernel void temporal_view_motion(texture2d<float, access::read> count [[texture(0)]],
                                 texture2d<float, access::sample> flow [[texture(1)]],
                                 texture2d<float, access::write> image [[texture(2)]],
                                 constant TViewU& U [[buffer(0)]],
                                 uint2 gid [[thread_position_in_grid]]) {
  float y = saturate((count.read(gid).x - 1.0f) * U.invCount);
  float2 v = clamp(t_atlas(flow, (float2(gid) + 0.5f) * 0.125f, 0u, U.rows).xy * (1.0f / 64.0f), float2(-0.5f),
                   float2(0.5f));
  image.write(float4(y, v.x, v.y, 1.0f), gid);
}
