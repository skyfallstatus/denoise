// SPDX-License-Identifier: MIT
// Sampled host-buffer checks for automatic-profile invalidation. Each keyed frame keeps a
// 64-byte header and one raw sample per 8x8 block. A changed sample stores the render stamp
// in the header. These samples never authorize reuse of image data: verifyFrameCache ingests
// every frame afresh.
// od: math=safe

struct CheckUniforms {
  uint rowStride;  // distance between rows of the host buffer, in 32-bit words
  uint width, height;
  uint stamp;      // the render's stamp (never 0)
  uint blocksX, blocksY;
  uint pad0, pad1;
};

constant uint kCheckHeaderWords = 16;

static inline uint check_mix(uint v, uint k) {
  uint h = v * k;
  h ^= h >> 15;
  h *= 0x2C1B3C6Du;
  h ^= h >> 12;
  return h;
}

constant uint kCheckPerm[8] = {3u, 7u, 0u, 4u, 6u, 1u, 5u, 2u};

static inline uint2 check_position(uint2 block, constant CheckUniforms& u) {
  // One pixel per block, stratified: along a row of blocks the sampled rows
  // are a permutation of 0..7 within every 8 consecutive blocks, and along a
  // column of blocks the sampled columns likewise. So a change along one
  // pixel row or column longer than 70 pixels always meets a sample (a
  // free pseudo-random position per block let 1-pixel lines of 600 to 1000
  // pixels through). The phase and the permutation (kCheckPerm XOR a value)
  // change from row to row and from column to column, so a regular pattern
  // in the picture cannot line up with the samples.
  const uint hy = check_mix(block.y, 0x85EBCA77u), hx = check_mix(block.x, 0x9E3779B1u);
  const uint oy = kCheckPerm[(block.x + hy) & 7u] ^ ((hy >> 3) & 7u);
  const uint ox = kCheckPerm[(block.y + hx) & 7u] ^ ((hx >> 3) & 7u);
  const uint x = min(block.x * 8u + ox, u.width - 1u);
  const uint y = min(block.y * 8u + oy, u.height - 1u);
  return uint2(x, y);
}

static inline void check_sample(uint4 v, device uint* check, constant CheckUniforms& u, uint2 block) {
  device uint4* s = (device uint4*)(check + kCheckHeaderWords) + (block.y * u.blocksX + block.x);
  const uint4 old = *s;
  if (any(old != v)) atomic_store_explicit((device atomic_uint*)check, u.stamp, memory_order_relaxed);
  *s = v;
}

kernel void ingest_check_bgra32f(const device uint* src [[buffer(0)]],
                                 device uint* check [[buffer(1)]],
                                 constant CheckUniforms& u [[buffer(2)]],
                                 uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= u.blocksX || gid.y >= u.blocksY) return;
  const uint2 p = check_position(gid, u);
  const device uint* q = src + p.y * u.rowStride + p.x * 4u;
  check_sample(uint4(q[0], q[1], q[2], q[3]), check, u, gid);
}

kernel void ingest_check_bgra16f(const device uint* src [[buffer(0)]],
                                 device uint* check [[buffer(1)]],
                                 constant CheckUniforms& u [[buffer(2)]],
                                 uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= u.blocksX || gid.y >= u.blocksY) return;
  const uint2 p = check_position(gid, u);
  const device uint* q = src + p.y * u.rowStride + p.x * 2u;
  check_sample(uint4(q[0], q[1], 0u, 0u), check, u, gid);
}
