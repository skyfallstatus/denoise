// SPDX-License-Identifier: MIT
// Advanced controls (API v6, ODParams.advanced and the fields after it)
// through od_render, on synthetic frames:
//   * neutral values: the defaults, an API v5 caller (advanced = 0, every
//     field zeroed) and NaN in every field are the same render, over quality,
//     radius, spatial, sharpen, view and format; and the default picture
//     itself matches output hashes recorded from the core before API v6
//     (default_picture, on the GPU they were recorded on)
//   * each control moves the picture the way opendenoise.h says (ratios
//     against the default render, measured in the octaves of the core's own
//     tent pyramid, reimplemented here), in each stage that reads it: noise
//     levels in the spatial filter, the temporal merge alone, sharpen coring
//     and the coarse-chroma persistence measure; the grain floor in the
//     spatial stage, the temporal merge alone and sharpen's noise estimate
//   * the bands step: odd and tiny frames, in place, 16f and 32f, alpha, the
//     stages it follows, the views it must not touch (and the Motion view
//     without a temporal stage, which shows the denoised picture), its
//     timing, the alpha and noise estimate it hands to sharpen
//   * Hold profile: no drift and no restart at a cut once held, on and off
//     mid-shot, estimates in flight when it goes on
//   * out-of-range and non-finite values: clamped, finite
// Memory: a few dozen 320x256 frames (< 100 MB).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "opendenoise/opendenoise.h"

#include "debug.h"

static bool grainProfileV2() {
  const char* e = std::getenv("OD_GRAIN_PROFILE_V2");
  return e && std::strcmp(e, "1") == 0;
}

static int gFailures = 0;
static int gChecks = 0;
static const char* gTest = "";

#define CHECK(cond, ...)                                                \
  do {                                                                  \
    ++gChecks;                                                          \
    if (!(cond)) {                                                      \
      ++gFailures;                                                      \
      fprintf(stderr, "FAIL [%s] %s:%d: ", gTest, __FILE__, __LINE__);  \
      fprintf(stderr, __VA_ARGS__);                                     \
      fprintf(stderr, "\n");                                            \
    }                                                                   \
  } while (0)

static const uint32_t W = 320, H = 256;
static id<MTLDevice> gDevice;
static ODContext* gCtx;

// ---- frames ------------------------------------------------------------------
struct Img {
  uint32_t w = 0, h = 0, rowBytes = 0;
  ODPixelFormat fmt = OD_PIXEL_BGRA32F;
  id<MTLBuffer> buf = nil;
  uint32_t bpp() const { return fmt == OD_PIXEL_BGRA16F ? 8 : 16; }
  size_t bytes() const { return (size_t)rowBytes * h; }
  ODFrame od(int64_t key) const {
    ODFrame f;
    f.mtlBuffer = (__bridge void*)buf;
    f.width = w;
    f.height = h;
    f.rowBytes = rowBytes;
    f.format = fmt;
    f.frameKey = key;
    return f;
  }
  float get(uint32_t x, uint32_t y, int c) const {
    const char* row = (const char*)buf.contents + (size_t)y * rowBytes;
    return fmt == OD_PIXEL_BGRA32F ? ((const float*)row)[x * 4 + c] : (float)((const __fp16*)row)[x * 4 + c];
  }
  void set(uint32_t x, uint32_t y, int c, float v) const {
    char* row = (char*)buf.contents + (size_t)y * rowBytes;
    if (fmt == OD_PIXEL_BGRA32F) ((float*)row)[x * 4 + c] = v;
    else ((__fp16*)row)[x * 4 + c] = (__fp16)v;
  }
};

static Img makeImg(uint32_t w = W, uint32_t h = H, ODPixelFormat fmt = OD_PIXEL_BGRA32F) {
  Img m;
  m.w = w;
  m.h = h;
  m.fmt = fmt;
  m.rowBytes = w * m.bpp();
  m.buf = [gDevice newBufferWithLength:m.bytes() options:MTLResourceStorageModeShared];
  memset(m.buf.contents, 0, m.bytes());
  return m;
}
static Img copyImg(const Img& a) {
  Img b = makeImg(a.w, a.h, a.fmt);
  memcpy(b.buf.contents, a.buf.contents, a.bytes());
  return b;
}
static bool sameBytes(const Img& a, const Img& b) { return memcmp(a.buf.contents, b.buf.contents, a.bytes()) == 0; }
static bool allFinite(const Img& a) {
  for (uint32_t y = 0; y < a.h; ++y)
    for (uint32_t x = 0; x < a.w; ++x)
      for (int c = 0; c < 4; ++c)
        if (!std::isfinite(a.get(x, y, c))) return false;
  return true;
}

// Working space of the core (common.metal).
static void rgbToYcc(const float rgb[3], float ycc[3]) {
  const float y = 0.2126f * rgb[0] + 0.7152f * rgb[1] + 0.0722f * rgb[2];
  ycc[0] = y;
  ycc[1] = (rgb[2] - y) / 1.8556f;
  ycc[2] = (rgb[0] - y) / 1.5748f;
}
static void yccToRgb(const float ycc[3], float rgb[3]) {
  rgb[0] = ycc[0] + 1.5748f * ycc[2];
  rgb[2] = ycc[0] + 1.8556f * ycc[1];
  rgb[1] = (ycc[0] - 0.2126f * rgb[0] - 0.0722f * rgb[2]) / 0.7152f;
}

// One plane per channel (Y, Cb, Cr) of a frame.
struct Planes {
  uint32_t w = 0, h = 0;
  std::vector<float> c[3];
};
static Planes makePlanes(uint32_t w, uint32_t h) {
  Planes p;
  p.w = w;
  p.h = h;
  for (auto& v : p.c) v.assign((size_t)w * h, 0.0f);
  return p;
}
static Planes planesOf(const Img& im) {
  Planes p = makePlanes(im.w, im.h);
  for (uint32_t y = 0; y < im.h; ++y)
    for (uint32_t x = 0; x < im.w; ++x) {
      const float rgb[3] = {im.get(x, y, 2), im.get(x, y, 1), im.get(x, y, 0)};
      float ycc[3];
      rgbToYcc(rgb, ycc);
      for (int c = 0; c < 3; ++c) p.c[c][(size_t)y * im.w + x] = ycc[c];
    }
  return p;
}
static void writePlanes(const Img& im, const Planes& p, float alpha = 1.0f) {
  for (uint32_t y = 0; y < im.h; ++y)
    for (uint32_t x = 0; x < im.w; ++x) {
      const size_t i = (size_t)y * im.w + x;
      const float ycc[3] = {p.c[0][i], p.c[1][i], p.c[2][i]};
      float rgb[3];
      yccToRgb(ycc, rgb);
      im.set(x, y, 0, rgb[2]);
      im.set(x, y, 1, rgb[1]);
      im.set(x, y, 2, rgb[0]);
      im.set(x, y, 3, alpha);
    }
}
static std::vector<float> diff(const std::vector<float>& a, const std::vector<float>& b) {
  std::vector<float> d(a.size());
  for (size_t i = 0; i < a.size(); ++i) d[i] = a[i] - b[i];
  return d;
}

static uint32_t hash32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}
static float gauss(uint32_t seed) {
  const float u1 = ((float)(hash32(seed) & 0xffffff) + 1.0f) / 16777217.0f;
  const float u2 = (float)(hash32(seed ^ 0x9e3779b9u) & 0xffffff) / 16777216.0f;
  return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}
static std::vector<float> whiteField(uint32_t w, uint32_t h, uint32_t seed) {
  std::vector<float> v((size_t)w * h);
  for (uint32_t y = 0; y < h; ++y)
    for (uint32_t x = 0; x < w; ++x) v[(size_t)y * w + x] = gauss((x * 73856093u) ^ (y * 19349663u) ^ (seed * 2654435761u));
  return v;
}
// Separable box blur of radius r, edges clamped.
static std::vector<float> boxBlur(const std::vector<float>& in, uint32_t w, uint32_t h, int r) {
  std::vector<float> t(in.size()), out(in.size());
  const float k = 1.0f / (float)(2 * r + 1);
  for (uint32_t y = 0; y < h; ++y)
    for (uint32_t x = 0; x < w; ++x) {
      float s = 0;
      for (int d = -r; d <= r; ++d) s += in[(size_t)y * w + (uint32_t)std::min((int)w - 1, std::max(0, (int)x + d))];
      t[(size_t)y * w + x] = s * k;
    }
  for (uint32_t y = 0; y < h; ++y)
    for (uint32_t x = 0; x < w; ++x) {
      float s = 0;
      for (int d = -r; d <= r; ++d) s += t[(size_t)std::min((int)h - 1, std::max(0, (int)y + d)) * w + x];
      out[(size_t)y * w + x] = s * k;
    }
  return out;
}
// Unit-variance low-frequency noise (white noise box-blurred with radius r).
static std::vector<float> lowField(uint32_t w, uint32_t h, uint32_t seed, int r) {
  std::vector<float> v = boxBlur(whiteField(w, h, seed), w, h, r);
  double s = 0;
  for (float x : v) s += (double)x * x;
  const float k = (float)(1.0 / std::sqrt(s / (double)v.size()));
  for (float& x : v) x *= k;
  return v;
}

// ---- the core's tent pyramid, on the CPU (spatial_nlm.metal, bands_amount.metal) ----
static std::vector<float> tentDown(const std::vector<float>& x, uint32_t w, uint32_t h, uint32_t* ow, uint32_t* oh) {
  const uint32_t cw = (w + 1) / 2, ch = (h + 1) / 2;
  auto at = [&](int i, int j) {
    return x[(size_t)std::min((int)h - 1, std::max(0, j)) * w + (uint32_t)std::min((int)w - 1, std::max(0, i))];
  };
  const float k[4] = {0.125f, 0.375f, 0.375f, 0.125f};
  std::vector<float> o((size_t)cw * ch);
  for (uint32_t j = 0; j < ch; ++j)
    for (uint32_t i = 0; i < cw; ++i) {
      float s = 0;
      for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b) s += k[a] * k[b] * at(2 * (int)i - 1 + b, 2 * (int)j - 1 + a);
      o[(size_t)j * cw + i] = s;
    }
  *ow = cw;
  *oh = ch;
  return o;
}
static std::vector<float> tentUp(const std::vector<float>& c, uint32_t cw, uint32_t ch, uint32_t w, uint32_t h) {
  std::vector<float> o((size_t)w * h);
  auto at = [&](int i, int j) {
    return c[(size_t)std::min((int)ch - 1, std::max(0, j)) * cw + (uint32_t)std::min((int)cw - 1, std::max(0, i))];
  };
  for (uint32_t y = 0; y < h; ++y)
    for (uint32_t x = 0; x < w; ++x) {
      const float u = ((float)x + 0.5f) * 0.5f - 0.5f, v = ((float)y + 0.5f) * 0.5f - 0.5f;
      const int i = (int)floorf(u), j = (int)floorf(v);
      const float fu = u - (float)i, fv = v - (float)j;
      o[(size_t)y * w + x] = (1 - fv) * ((1 - fu) * at(i, j) + fu * at(i + 1, j)) + fv * ((1 - fu) * at(i, j + 1) + fu * at(i + 1, j + 1));
    }
  return o;
}
// rms of octave b of x (0..2: G_b - U(G_(b+1)), 3: G_3 itself, band 3 and
// coarser), away from the frame border.
static double bandRms(const std::vector<float>& x, uint32_t w, uint32_t h, int b) {
  std::vector<float> g = x;
  uint32_t gw = w, gh = h;
  for (int k = 0; k < b; ++k) g = tentDown(g, gw, gh, &gw, &gh);
  std::vector<float> l = g;
  if (b < 3) {
    uint32_t cw, ch;
    std::vector<float> c = tentDown(g, gw, gh, &cw, &ch);
    std::vector<float> u = tentUp(c, cw, ch, gw, gh);
    for (size_t i = 0; i < l.size(); ++i) l[i] -= u[i];
  }
  const uint32_t m = std::max(2u, 16u >> b);
  double s = 0;
  size_t n = 0;
  for (uint32_t y = m; y + m < gh; ++y)
    for (uint32_t x2 = m; x2 + m < gw; ++x2) {
      s += (double)l[(size_t)y * gw + x2] * l[(size_t)y * gw + x2];
      ++n;
    }
  return n ? std::sqrt(s / (double)n) : 0;
}
// rms over a region (default: away from the border).
static double rmsOf(const std::vector<float>& d, uint32_t w, uint32_t h,
                    const std::function<bool(uint32_t, uint32_t)>& in = nullptr) {
  double s = 0;
  size_t n = 0;
  for (uint32_t y = 8; y + 8 < h; ++y)
    for (uint32_t x = 8; x + 8 < w; ++x) {
      if (in && !in(x, y)) continue;
      s += (double)d[(size_t)y * w + x] * d[(size_t)y * w + x];
      ++n;
    }
  return n ? std::sqrt(s / (double)n) : 0;
}

// ---- sequences -----------------------------------------------------------------
// A clean scene (Y, Cb, Cr per pixel and frame) plus noise:
//   rgb    white noise of this sigma on R, G and B
//   yLow   low-frequency luma noise (box radius 6) of this sigma
//   cWhite white noise of this sigma on Cb and Cr
//   cLow   low-frequency chroma noise (half box radius 2, half radius 6)
struct Noise {
  float rgb = 0, yLow = 0, cWhite = 0, cLow = 0;
};
typedef std::function<void(uint32_t x, uint32_t y, int frame, float ycc[3])> Scene;

static Planes cleanPlanes(const Scene& scene, int frame, uint32_t w = W, uint32_t h = H) {
  Planes p = makePlanes(w, h);
  for (uint32_t y = 0; y < h; ++y)
    for (uint32_t x = 0; x < w; ++x) {
      float ycc[3];
      scene(x, y, frame, ycc);
      for (int c = 0; c < 3; ++c) p.c[c][(size_t)y * w + x] = ycc[c];
    }
  return p;
}
static Img noisyFrame(const Scene& scene, int frame, const Noise& n, uint32_t seed, ODPixelFormat fmt = OD_PIXEL_BGRA32F,
                      uint32_t w = W, uint32_t h = H, float alpha = 1.0f) {
  Planes p = cleanPlanes(scene, frame, w, h);
  const uint32_t s = seed * 7919u + (uint32_t)frame * 104729u;
  if (n.yLow > 0) {
    std::vector<float> l = lowField(w, h, s + 1, 6);
    for (size_t i = 0; i < l.size(); ++i) p.c[0][i] += n.yLow * l[i];
  }
  for (int c = 1; c < 3; ++c) {
    if (n.cWhite > 0) {
      std::vector<float> v = whiteField(w, h, s + 10 + (uint32_t)c);
      for (size_t i = 0; i < v.size(); ++i) p.c[c][i] += n.cWhite * v[i];
    }
    if (n.cLow > 0) {
      std::vector<float> a = lowField(w, h, s + 20 + (uint32_t)c, 2), b = lowField(w, h, s + 30 + (uint32_t)c, 6);
      for (size_t i = 0; i < a.size(); ++i) p.c[c][i] += n.cLow * 0.7071f * (a[i] + b[i]);
    }
  }
  Img im = makeImg(w, h, fmt);
  writePlanes(im, p, alpha);
  if (n.rgb > 0) {
    for (uint32_t y = 0; y < h; ++y)
      for (uint32_t x = 0; x < w; ++x)
        for (int c = 0; c < 3; ++c) {
          const uint32_t k = (x * 73856093u) ^ (y * 19349663u) ^ ((uint32_t)c * 83492791u) ^ (s * 40503u) ^ 0x5bd1e995u;
          im.set(x, y, c, im.get(x, y, c) + n.rgb * gauss(k));
        }
  }
  return im;
}
static std::vector<Img> sequence(const Scene& scene, int count, const Noise& n, uint32_t seed,
                                 ODPixelFormat fmt = OD_PIXEL_BGRA32F, uint32_t w = W, uint32_t h = H) {
  std::vector<Img> v;
  for (int f = 0; f < count; ++f) v.push_back(noisyFrame(scene, f, n, seed, fmt, w, h));
  return v;
}

// Scenes.
static void flatScene(uint32_t, uint32_t, int, float ycc[3]) {
  ycc[0] = 0.45f;
  ycc[1] = ycc[2] = 0.0f;
}
// The pipeline test's scene: flat patches (left), texture (middle), a
// gradient with a bright square moving 3 px per frame (right).
static void mixedScene(uint32_t x, uint32_t y, int frame, float ycc[3]) {
  float v;
  if (x < W / 3) {
    v = 0.08f + 0.1f * (float)(y * 8 / H);
  } else if (x < 2 * W / 3) {
    v = 0.45f + 0.024f * sinf(0.9f * (float)x + 0.4f * (float)y) * cosf(0.7f * (float)y);
  } else {
    v = 0.25f + 0.3f * (float)(x - 2 * W / 3) / (float)(W / 3);
    const int sx = (int)(2 * W / 3) + 20 + 3 * frame, sy = 100;
    if ((int)x >= sx && (int)x < sx + 40 && (int)y >= sy && (int)y < sy + 40) v = 0.85f;
  }
  ycc[0] = v;
  ycc[1] = 0.02f;
  ycc[2] = -0.015f;
}
// A textured 64 px square moving 4 px per frame over a flat background.
static const int kSqY = 96, kSq = 64;
static int squareX(int frame) { return 60 + 4 * frame; }
static void movingScene(uint32_t x, uint32_t y, int frame, float ycc[3]) {
  const int sx = squareX(frame);
  ycc[0] = 0.3f + 0.0002f * (float)x;
  ycc[1] = ycc[2] = 0.0f;
  const int lx = (int)x - sx, ly = (int)y - kSqY;
  if (lx >= 0 && lx < kSq && ly >= 0 && ly < kSq) {
    ycc[0] = 0.62f + 0.12f * sinf(0.55f * (float)lx) * cosf(0.45f * (float)ly);
    ycc[1] = 0.03f;
  }
}

// ---- rendering -------------------------------------------------------------------
// White Gaussian noise of pixel sigma s on R, G and B, as a profile.
static ODNoiseProfile whiteProfile(float s) {
  ODNoiseProfile p;
  memset(&p, 0, sizeof(p));
  const float chan[3] = {0.7497f, 0.6416f, 0.6770f};
  const float band[OD_BANDS] = {0.8908f, 0.2007f, 0.0855f, 0.0412f};
  for (int c = 0; c < 3; ++c) {
    for (int b = 0; b < OD_BANDS; ++b)
      for (int l = 0; l < OD_LEVELS; ++l) p.sigma[c][b][l] = s * chan[c] * band[b];
    for (int l = 0; l < OD_LEVELS; ++l) p.sigmaFull[c][l] = s * chan[c] * 0.9987f;
  }
  p.confidence = 1;
  p.valid = 1;
  return p;
}

static ODParams base() {
  ODParams p = od_params_default();
  p.waitUntilCompleted = 1;
  return p;
}

// Renders seq[c] with its neighbours up to the radius (keys keyBase + index),
// on `session` or a fresh one.
static bool renderAt(const std::vector<Img>& seq, int c, const ODParams& p, const Img& out, ODSession* session = nullptr,
                     ODStats* st = nullptr, int64_t keyBase = 1000) {
  ODSession* s = session ? session : od_session_create(gCtx);
  ODFrame frames[OD_MAX_FRAMES];
  int offsets[OD_MAX_FRAMES];
  int n = 0;
  frames[n] = seq[(size_t)c].od(keyBase + c);
  offsets[n++] = 0;
  for (int d = 1; d <= p.temporalRadius; ++d)
    for (int sg = -1; sg <= 1; sg += 2) {
      const int j = c + sg * d;
      if (j < 0 || j >= (int)seq.size()) continue;
      frames[n] = seq[(size_t)j].od(keyBase + j);
      offsets[n++] = sg * d;
    }
  ODFrame o = out.od(OD_NO_KEY);
  const int rc = od_render(s, frames, offsets, n, &p, &o, st);
  if (rc != 0) fprintf(stderr, "od_render: %s\n", od_last_error(s));
  if (!session) od_session_destroy(s);
  return rc == 0;
}
// The same, returning the result.
static Img rendered(const std::vector<Img>& seq, int c, const ODParams& p) {
  Img out = makeImg(seq[0].w, seq[0].h, seq[0].fmt);
  CHECK(renderAt(seq, c, p, out), "render failed");
  return out;
}

// The API v5 caller: advanced = 0 and every field after it zeroed.
static void zeroAdvanced(ODParams* p) {
  memset(&p->advanced, 0, sizeof(ODParams) - offsetof(ODParams, advanced));
}
// advanced = 1 with every float field NaN: each takes its neutral value.
static void nanAdvanced(ODParams* p) {
  const float q = NAN;
  p->advanced = 1;
  for (int g = 0; g < 2; ++g)
    for (int b = 0; b < OD_BANDS; ++b) p->bandNoiseLevel[g][b] = p->bandAmount[g][b] = q;
  for (int c = 0; c < 3; ++c) p->channelNoiseLevel[c] = q;
  p->grainFloor = p->temporalMotionThreshold = p->temporalEdgeGate = q;
  p->sharpenBand[0] = p->sharpenBand[1] = p->sharpenCoring = p->sharpenHalo = q;
  p->speckRemoval = 1;
  p->profileHold = 0;
  p->grainStability = q;  // API v9
  p->coarseChroma = 0;
}

// ---- tests -------------------------------------------------------------------------

// od_params_default(): advanced = 1 and the neutral values (opendenoise.h).
static void testDefaults() {
  gTest = "defaults";
  const ODParams p = od_params_default();
  bool ok = p.advanced == 1 && p.grainFloor == 1.0f && p.speckRemoval == 1 && p.temporalMotionThreshold == 1.0f &&
            p.temporalEdgeGate == 0.75f && p.sharpenBand[0] == 1.0f && p.sharpenBand[1] == 1.0f &&
            p.sharpenCoring == 1.0f && p.sharpenHalo == 1.0f && p.profileHold == 0 && p.grainStability == 0.0f &&
            p.coarseChroma == 0;
  for (int g = 0; g < 2; ++g)
    for (int b = 0; b < OD_BANDS; ++b) ok = ok && p.bandNoiseLevel[g][b] == 1.0f && p.bandAmount[g][b] == 1.0f;
  for (int c = 0; c < 3; ++c) ok = ok && p.channelNoiseLevel[c] == 1.0f;
  CHECK(ok, "od_params_default() does not give the neutral advanced values of opendenoise.h");
  // v6 brought these fields; v7 (ODStats for hosts) and v8 (coarse chroma
  // fields of ODNoiseProfile) left ODParams alone; v9 appended
  // grainStability and coarseChroma; v10 grainMode, whose zero is the old
  // picture and whose default (a new effect's) is the finest-grain mode.
  CHECK(OD_API_VERSION >= 10, "OD_API_VERSION %d", OD_API_VERSION);
  CHECK(p.grainMode == OD_GRAIN_FINEST && OD_GRAIN_WHOLE == 0,
        "od_params_default() grain mode %u, expected OD_GRAIN_FINEST; OD_GRAIN_WHOLE must be 0", p.grainMode);
}

// Grain stability (API v9, ODParams.grainStability): at 1 the bytes of the diagnostic
// setting OD_SPATIAL_TUNE="stable=1,fine=0.75"; 0, below 0 and NaN the default picture;
// above 1 clamped to 1; a tune that names `stable` overrides it; luma only (32f: chroma
// unchanged); and the reference is linear in the value, so at 0.5 the luma is the mean of
// the luma at 0 and 1 up to the rounding of the half-precision result. On a static noisy
// scene the frame-to-frame change of the output falls at every step of 0, 0.25,..., 1.
static void testGrainStability() {
  gTest = "grain_stability";
  Noise n;
  n.rgb = 0.02f;
  for (ODPixelFormat fmt : {OD_PIXEL_BGRA16F, OD_PIXEL_BGRA32F}) {
    const char* fn = fmt == OD_PIXEL_BGRA16F ? "16f" : "32f";
    std::vector<Img> seq = sequence(mixedScene, 7, n, 11, fmt);
    for (int q = 0; q <= 1; ++q) {
      auto at = [&](float g, const char* tune) {
        if (tune) setenv("OD_SPATIAL_TUNE", tune, 1);  // read when the session is created
        else unsetenv("OD_SPATIAL_TUNE");
        ODParams p = base();
        p.quality = q;
        p.grainStability = g;
        Img o = rendered(seq, 3, p);
        unsetenv("OD_SPATIAL_TUNE");
        return o;
      };
      const Img d = at(0, nullptr), one = at(1, nullptr), half = at(0.5f, nullptr);
      CHECK(sameBytes(one, at(0, "stable=1,fine=0.75")), "%s q%d: Grain stability 1 differs from the explicit stabilized setting",
            fn, q);
      CHECK(!sameBytes(one, d), "%s q%d: Grain stability 1 changed nothing", fn, q);
      CHECK(sameBytes(at(NAN, nullptr), d) && sameBytes(at(-1, nullptr), d), "%s q%d: NaN or -1 is not the default",
            fn, q);
      CHECK(sameBytes(at(3, nullptr), one), "%s q%d: 3 is not clamped to 1", fn, q);
      CHECK(sameBytes(at(1, "stable=0"), d), "%s q%d: OD_SPATIAL_TUNE=stable=0 does not override it", fn, q);
      const Planes pd = planesOf(d), p1 = planesOf(one), ph = planesOf(half);
      double maxDev = 0, sumDev = 0, sumMove = 0, maxC = 0;
      for (size_t i = 0; i < pd.c[0].size(); ++i) {
        const double dev = std::fabs(ph.c[0][i] - 0.5 * (pd.c[0][i] + p1.c[0][i]));
        maxDev = std::max(maxDev, dev);
        sumDev += dev;
        sumMove += std::fabs(p1.c[0][i] - pd.c[0][i]);
        for (int c = 1; c < 3; ++c) maxC = std::max(maxC, (double)std::fabs(p1.c[c][i] - pd.c[c][i]));
      }
      printf("  %s q%d: luma at 0.5 vs the mean of 0 and 1: max %.2e, mean %.2e (the change 0 -> 1: mean %.2e); "
             "chroma max change %.1e\n", fn, q, maxDev, sumDev / pd.c[0].size(), sumMove / pd.c[0].size(), maxC);
      // The final blend is linear in the reference and rounds its result to
      // half: three independent roundings (ulp 2.44e-4 at Y in [0.25, 0.5))
      // leave |o(.5) - (o(0) + o(1)) / 2| at 0.35 ulp x 0.8 = 6.9e-5 on
      // average, whatever the change 0 -> 1 is.
      CHECK(maxDev < 1.5e-3 && sumDev / pd.c[0].size() < 1.25e-4 && sumDev < 0.15 * sumMove,
            "%s q%d: Grain stability 0.5 is not halfway (max %.2e, mean %.2e)", fn, q, maxDev, sumDev / pd.c[0].size());
      if (fmt == OD_PIXEL_BGRA32F) CHECK(maxC < 2e-6, "%s q%d: Grain stability changed chroma (%.2e)", fn, q, maxC);
    }
  }
  // Monotone in temporal grain: a static flat scene, frames 2..5 on one
  // session (blocking: the same profile for every value).
  std::vector<Img> flat = sequence(flatScene, 8, n, 23);
  double prev = 1e30;
  for (float g : {0.0f, 0.25f, 0.5f, 0.75f, 1.0f}) {
    ODParams p = base();
    p.grainStability = g;
    ODSession* s = od_session_create(gCtx);
    std::vector<Planes> outs;
    for (int c = 2; c <= 5; ++c) {
      Img o = makeImg();
      CHECK(renderAt(flat, c, p, o, s), "render failed");
      outs.push_back(planesOf(o));
    }
    od_session_destroy(s);
    double ms = 0;
    for (size_t k = 1; k < outs.size(); ++k) ms += std::pow(rmsOf(diff(outs[k].c[0], outs[k - 1].c[0]), W, H), 2);
    const double flicker = std::sqrt(ms / (double)(outs.size() - 1));
    printf("  flat scene, Grain stability %.2f: frame-to-frame luma change rms %.6f\n", g, flicker);
    CHECK(flicker < prev, "Grain stability %.2f: frame-to-frame change %.6f is not below the previous step's %.6f", g,
          flicker, prev);
    prev = flicker;
  }
}

// An API v5 caller (advanced 0, fields zeroed), od_params_default() and NaN
// in every field give the same bytes, over quality x radius x spatial x
// sharpen x view x format, with the automatic profile. (sanitizeAdvanced maps
// all three to the same parameters before any stage runs, so this checks that
// mapping; that the render is the picture of API v5 is default_picture's job.)
static void testNeutral() {
  gTest = "neutral";
  int combos = 0, equalB = 0, equalC = 0;
  for (ODPixelFormat fmt : {OD_PIXEL_BGRA16F, OD_PIXEL_BGRA32F}) {
    Noise n;
    n.rgb = 0.02f;
    std::vector<Img> seq = sequence(mixedScene, 5, n, 3, fmt);
    for (int q = 0; q <= 1; ++q)
      for (int r : {0, 2})
        for (float sp : {0.0f, 0.5f})
          for (float sh : {0.0f, 0.6f})
            for (int v : {OD_VIEW_OUTPUT, OD_VIEW_NOISE}) {
              ODParams a = base();
              a.quality = q;
              a.temporalRadius = r;
              a.spatialStrength = sp;
              a.sharpen = sh;
              a.view = v;
              ODParams b = a, c = a;
              zeroAdvanced(&b);
              nanAdvanced(&c);
              Img oa = rendered(seq, 2, a), ob = rendered(seq, 2, b), oc = rendered(seq, 2, c);
              ++combos;
              const bool eb = sameBytes(oa, ob), ec = sameBytes(oa, oc);
              equalB += eb;
              equalC += ec;
              CHECK(eb, "%s q%d r%d spatial %.1f sharpen %.1f view %d: defaults differ from an API v5 caller",
                    fmt == OD_PIXEL_BGRA16F ? "16f" : "32f", q, r, sp, sh, v);
              CHECK(ec, "%s q%d r%d spatial %.1f sharpen %.1f view %d: NaN advanced values differ from the defaults",
                    fmt == OD_PIXEL_BGRA16F ? "16f" : "32f", q, r, sp, sh, v);
            }
  }
  printf("  %d combinations: API v5 caller identical in %d, NaN values identical in %d\n", combos, equalB, equalC);
}

// Sharpen fine and medium at 0: the sharpen stage adds nothing (both gains 0),
// so the bytes are those of Sharpen 0.
static void testSharpenBandsZero() {
  gTest = "sharpen_bands_zero";
  Noise n;
  n.rgb = 0.02f;
  std::vector<Img> seq = sequence(mixedScene, 5, n, 5);
  const ODNoiseProfile prof = whiteProfile(0.02f);
  for (int q = 0; q <= 1; ++q) {
    ODParams p = base();
    p.manualProfile = &prof;
    p.quality = q;
    Img off = rendered(seq, 2, p);
    p.sharpen = 0.6f;
    p.sharpenBand[0] = p.sharpenBand[1] = 0.0f;
    Img zero = rendered(seq, 2, p);
    p.sharpenBand[0] = p.sharpenBand[1] = 1.0f;
    Img on = rendered(seq, 2, p);
    CHECK(sameBytes(off, zero), "q%d: Sharpen 0.6 with both bands at 0 differs from Sharpen 0", q);
    CHECK(!sameBytes(off, on), "q%d: Sharpen 0.6 changed nothing: the test says nothing", q);
  }
}

// Noise level per band and channel: more removes more in that band.
static void testLevels() {
  gTest = "levels";
  // Luma fine: white noise, manual profile, luma 100 %.
  {
    Noise n;
    n.rgb = 0.02f;
    std::vector<Img> seq = sequence(flatScene, 5, n, 11);
    const Planes clean = cleanPlanes(flatScene, 2);
    const ODNoiseProfile prof = whiteProfile(0.02f);
    // A gentle baseline (spatial only, half the default threshold): at the
    // defaults the finest band is already averaged uniformly over the search
    // window, and a higher level cannot take more there.
    auto fine = [&](float level) {
      ODParams p = base();
      p.manualProfile = &prof;
      p.lumaStrength = 1;
      p.temporalRadius = 0;
      p.spatialStrength = 0.25f;
      p.bandNoiseLevel[0][0] = level;
      Planes o = planesOf(rendered(seq, 2, p));
      return bandRms(diff(o.c[0], clean.c[0]), W, H, 0);
    };
    const double d = fine(1), hi = fine(2), lo = fine(0.5f);
    printf("  luma fine level 50 / 100 / 200 %%: fine residual %.6f / %.6f / %.6f\n", lo, d, hi);
    CHECK(hi < 0.9 * d && lo > 1.1 * d, "luma fine level: %.6f / %.6f / %.6f", lo, d, hi);
  }
  // Luma very coarse: low-frequency luma noise, automatic profile.
  {
    Noise n;
    n.rgb = 0.004f;
    n.yLow = 0.015f;
    std::vector<Img> seq = sequence(flatScene, 5, n, 12);
    const Planes clean = cleanPlanes(flatScene, 2);
    auto coarse = [&](float level) {
      ODParams p = base();
      p.lumaStrength = 1;
      p.bandNoiseLevel[0][3] = level;
      Planes o = planesOf(rendered(seq, 2, p));
      return bandRms(diff(o.c[0], clean.c[0]), W, H, 3);
    };
    const double d = coarse(1), hi = coarse(2), lo = coarse(0.5f);
    printf("  luma very coarse level 50 / 100 / 200 %%: very coarse residual %.6f / %.6f / %.6f\n", lo, d, hi);
    CHECK(hi < d && lo > d, "luma very coarse level: %.6f / %.6f / %.6f", lo, d, hi);
  }
  // Chroma, each band, and the Cb channel level: multi-scale chroma noise.
  {
    Noise n;
    n.rgb = 0.004f;
    n.cWhite = 0.015f;
    n.cLow = 0.012f;
    std::vector<Img> seq = sequence(flatScene, 5, n, 13);
    const Planes clean = cleanPlanes(flatScene, 2);
    struct R {
      double y, cb[4], cr[4];
    };
    auto run = [&](const ODParams& p) {
      Planes o = planesOf(rendered(seq, 2, p));
      R r;
      r.y = rmsOf(diff(o.c[0], clean.c[0]), W, H);
      for (int b = 0; b < 4; ++b) {
        r.cb[b] = bandRms(diff(o.c[1], clean.c[1]), W, H, b);
        r.cr[b] = bandRms(diff(o.c[2], clean.c[2]), W, H, b);
      }
      return r;
    };
    // Gentle baseline as above (the defaults average every chroma band
    // uniformly here already).
    ODParams g = base();
    g.temporalRadius = 0;
    g.spatialStrength = 0.25f;
    const R d = run(g);
    for (int b = 0; b < OD_BANDS; ++b) {
      ODParams p = g;
      p.bandNoiseLevel[1][b] = 2.0f;
      const R hi = run(p);
      p.bandNoiseLevel[1][b] = 0.5f;
      const R lo = run(p);
      const double dc = std::hypot(d.cb[b], d.cr[b]), hc = std::hypot(hi.cb[b], hi.cr[b]),
                   lc = std::hypot(lo.cb[b], lo.cr[b]);
      printf("  chroma band %d level 50 / 100 / 200 %%: residual %.6f / %.6f / %.6f; luma %.6f / %.6f / %.6f\n", b, lc,
             dc, hc, lo.y, d.y, hi.y);
      CHECK(hc < dc && lc > dc, "chroma band %d level: %.6f / %.6f / %.6f", b, lc, dc, hc);
      CHECK(std::fabs(hi.y / d.y - 1) < 0.01 && std::fabs(lo.y / d.y - 1) < 0.01,
            "chroma band %d level moved the luma: %.6f / %.6f / %.6f", b, lo.y, d.y, hi.y);
    }
    ODParams p = g;
    p.channelNoiseLevel[1] = 2.0f;
    const R cb = run(p);
    double dcb = 0, dcr = 0, hcb = 0, hcr = 0;
    for (int b = 0; b < 4; ++b) {
      dcb += d.cb[b] * d.cb[b];
      dcr += d.cr[b] * d.cr[b];
      hcb += cb.cb[b] * cb.cb[b];
      hcr += cb.cr[b] * cb.cr[b];
    }
    dcb = std::sqrt(dcb);
    dcr = std::sqrt(dcr);
    hcb = std::sqrt(hcb);
    hcr = std::sqrt(hcr);
    printf("  Cb noise level 200 %%: Cb residual %.6f -> %.6f, Cr %.6f -> %.6f, luma %.6f -> %.6f\n", dcb, hcb, dcr, hcr,
           d.y, cb.y);
    CHECK(hcb < 0.95 * dcb, "Cb noise level 200 %%: Cb residual %.6f -> %.6f", dcb, hcb);
    // The chroma weights are joint (one distance over Cb and Cr), so Cr
    // follows a little; Cb must move much more.
    CHECK(hcb / dcb < hcr / dcr - 0.02, "Cb noise level 200 %%: Cr moved as much as Cb (%.3f, %.3f)", hcb / dcb,
          hcr / dcr);
    CHECK(std::fabs(cb.y / d.y - 1) < 0.01, "Cb noise level moved the luma: %.6f -> %.6f", d.y, cb.y);
  }
}

// Fine and very coarse parts of a plane, as the design measures them: x minus
// its 3x3 box, and its 17x17 box.
static double fineRms(const std::vector<float>& x) {
  const std::vector<float> b = boxBlur(x, W, H, 1);
  return rmsOf(diff(x, b), W, H);
}
static double coarseRms(const std::vector<float>& x) { return rmsOf(boxBlur(x, W, H, 8), W, H); }

// Amount per band: the bands step keeps that share of the removed signal of
// the band, and leaves the other bands alone. The pyramid's octaves overlap
// (short tent filters, bilinear upsampling), so a band set to 0 still shows
// a few per cent in a measurement with other filters.
static void testAmounts() {
  gTest = "amounts";
  const ODNoiseProfile prof = whiteProfile(0.02f);
  Noise n;
  n.rgb = 0.02f;
  std::vector<Img> seq = sequence(flatScene, 5, n, 21);
  const Planes in = planesOf(seq[2]);
  ODParams p0 = base();
  p0.manualProfile = &prof;
  p0.lumaStrength = 1;
  const Planes d = planesOf(rendered(seq, 2, p0));
  const std::vector<float> rd = diff(d.c[0], in.c[0]);
  // Luma fine amount 0.
  {
    ODParams p = p0;
    p.bandAmount[0][0] = 0;
    const Planes o = planesOf(rendered(seq, 2, p));
    const std::vector<float> r = diff(o.c[0], in.c[0]);
    const double f = fineRms(r), fd = fineRms(rd), c = coarseRms(r), cd = coarseRms(rd);
    const double t0 = bandRms(r, W, H, 0), t0d = bandRms(rd, W, H, 0);
    printf("  luma fine amount 0: removed fine %.6f (default %.6f; tent octave 0 %.6f of %.6f), very coarse %.6f "
           "(default %.6f)\n", f, fd, t0, t0d, c, cd);
    CHECK(f < 0.15 * fd, "luma fine amount 0 still removes fine noise: %.6f of %.6f", f, fd);
  }
  // Every amount 0.5 is luma and chroma strength halved (with the grain
  // floor off: it caps the luma amount below 1 here), every amount 0 the
  // input.
  {
    ODParams p = p0;
    p.grainFloor = 0;
    for (int g = 0; g < 2; ++g)
      for (int b = 0; b < OD_BANDS; ++b) p.bandAmount[g][b] = 0.5f;
    const Planes half = planesOf(rendered(seq, 2, p));
    ODParams ps = p0;
    ps.grainFloor = 0;
    ps.lumaStrength = 0.5f;
    ps.chromaStrength = 0.5f;
    const Planes strength = planesOf(rendered(seq, 2, ps));
    for (int g = 0; g < 2; ++g)
      for (int b = 0; b < OD_BANDS; ++b) p.bandAmount[g][b] = 0.0f;
    const Img noneImg = rendered(seq, 2, p);
    const Planes none = planesOf(noneImg);
    ODParams pf = p0;
    pf.grainFloor = 0;
    const Planes full = planesOf(rendered(seq, 2, pf));
    double e = 0, z = 0;
    for (int c = 0; c < 3; ++c) {
      const double a = rmsOf(diff(half.c[c], strength.c[c]), W, H), r = rmsOf(diff(full.c[c], in.c[c]), W, H);
      e = std::max(e, a / r);
      z = std::max(z, rmsOf(diff(none.c[c], in.c[c]), W, H) / r);
    }
    // Every weight 0 gives the ingested frame back exactly, and the output
    // writes source + (result - ingested): the host's own bytes.
    CHECK(sameBytes(noneImg, seq[2]), "every amount 0 does not give the input's bytes back");
    printf("  every amount 50 %%: off the half-strength render by %.4f of the removed signal; every amount 0: off the "
           "input by %.4f\n", e, z);
    CHECK(e < 0.02, "every amount 50 %% is not half the removed signal (%.4f off)", e);
    CHECK(z < 0.02, "every amount 0 is not the input (%.4f off)", z);
  }
  // Chroma amounts alone leave the luma (Y is copied), and the other way round.
  {
    ODParams p = p0;
    for (int b = 0; b < OD_BANDS; ++b) p.bandAmount[1][b] = 0.25f * (float)b;
    const Planes o = planesOf(rendered(seq, 2, p));
    double dy = 0, dc = 0;
    for (size_t i = 0; i < o.c[0].size(); ++i) {
      dy = std::max(dy, (double)std::fabs(o.c[0][i] - d.c[0][i]));
      dc = std::max(dc, (double)std::fabs(o.c[1][i] - d.c[1][i]));
    }
    printf("  chroma amounts: luma moved by at most %.2g, chroma by %.2g\n", dy, dc);
    CHECK(dy < 2e-6 && dc > 1e-3, "chroma amounts: luma moved by %.2g (chroma %.2g)", dy, dc);
    p = p0;
    for (int b = 0; b < OD_BANDS; ++b) p.bandAmount[0][b] = 0.25f * (float)b;
    const Planes l = planesOf(rendered(seq, 2, p));
    double cy = 0, cc = 0;
    for (size_t i = 0; i < l.c[0].size(); ++i) {
      cy = std::max(cy, (double)std::fabs(l.c[0][i] - d.c[0][i]));
      cc = std::max({cc, (double)std::fabs(l.c[1][i] - d.c[1][i]), (double)std::fabs(l.c[2][i] - d.c[2][i])});
    }
    CHECK(cc < 2e-6 && cy > 1e-3, "luma amounts: chroma moved by %.2g (luma %.2g)", cc, cy);
  }
  // Luma very coarse amount 0, on low-frequency noise.
  {
    Noise ln;
    ln.rgb = 0.006f;
    ln.yLow = 0.015f;
    std::vector<Img> ls = sequence(flatScene, 5, ln, 22);
    const Planes lin = planesOf(ls[2]);
    ODParams p = base();
    p.lumaStrength = 1;
    const Planes ld = planesOf(rendered(ls, 2, p));
    p.bandAmount[0][3] = 0;
    const Planes lo = planesOf(rendered(ls, 2, p));
    p.bandAmount[0][3] = 1;
    p.bandAmount[0][0] = 0;
    const Planes lf = planesOf(rendered(ls, 2, p));
    const std::vector<float> r = diff(lo.c[0], lin.c[0]), r0 = diff(ld.c[0], lin.c[0]), rf = diff(lf.c[0], lin.c[0]);
    const double c = coarseRms(r), cd = coarseRms(r0), f = fineRms(r), fd = fineRms(r0);
    const double c33 = rmsOf(boxBlur(r, W, H, 16), W, H), c33d = rmsOf(boxBlur(r0, W, H, 16), W, H);
    printf("  luma very coarse amount 0: removed very coarse %.6f (default %.6f; 33x33 box %.6f of %.6f; tent G3 %.6f "
           "of %.6f), fine %.6f (default %.6f)\n", c, cd, c33, c33d, bandRms(r, W, H, 3), bandRms(r0, W, H, 3), f, fd);
    // The 17x17 box also sees the 4-8 px octave, which stays removed; the
    // tent pyramid's band 3 is U^3(G_3), so the 8-16 px detail of G_3 that
    // the bilinear upsampling does not rebuild is in the finer octaves.
    CHECK(c < 0.4 * cd && c33 < 0.25 * c33d, "luma very coarse amount 0 still removes coarse noise: %.6f of %.6f "
          "(33x33: %.6f of %.6f)", c, cd, c33, c33d);
    CHECK(std::fabs(f / fd - 1) < 0.05, "luma very coarse amount 0 moved the fine band: %.6f, default %.6f", f, fd);
    const double fc = coarseRms(rf);
    printf("  luma fine amount 0 on the same: removed very coarse %.6f (default %.6f), fine %.6f\n", fc, cd, fineRms(rf));
    CHECK(std::fabs(fc / cd - 1) < 0.05, "luma fine amount 0 moved the very coarse part: %.6f, default %.6f", fc, cd);
  }
}

// Luma grain floor: 0 removes all the luma noise the amount allows, more
// keeps more grain.
static void testGrainFloor() {
  gTest = "grain_floor";
  Noise n;
  n.rgb = 0.002f;
  std::vector<Img> seq = sequence(flatScene, 5, n, 31);
  const Planes clean = cleanPlanes(flatScene, 2);
  const ODNoiseProfile prof = whiteProfile(0.002f);
  const double in = rmsOf(diff(planesOf(seq[2]).c[0], clean.c[0]), W, H);
  auto res = [&](float floor) {
    ODParams p = base();
    p.manualProfile = &prof;
    p.lumaStrength = 1;
    p.grainFloor = floor;
    return rmsOf(diff(planesOf(rendered(seq, 2, p)).c[0], clean.c[0]), W, H);
  };
  const double f0 = res(0), f1 = res(1), f4 = res(4);
  printf("  luma noise %.6f; residual with grain floor 0 / 100 / 400 %%: %.6f / %.6f / %.6f\n", in, f0, f1, f4);
  CHECK(f0 < f1 && f1 < f4, "grain floor not monotonic: %.6f / %.6f / %.6f", f0, f1, f4);
  CHECK(f4 >= 0.9 * std::min(in, 4 * 7.5e-4), "grain floor 400 %% kept %.6f of %.6f", f4, in);
}

// Remove specks off keeps single dark pixels the filter otherwise removes.
static void testSpecks() {
  gTest = "specks";
  Noise n;
  n.rgb = 0.02f;
  std::vector<Img> seq = sequence(flatScene, 1, n, 41);
  const Planes clean = cleanPlanes(flatScene, 0);
  Planes p0 = planesOf(seq[0]);
  std::vector<size_t> at;
  for (int k = 0; k < 50; ++k) {
    const uint32_t x = 24 + (uint32_t)(k % 10) * 28, y = 24 + (uint32_t)(k / 10) * 44;
    at.push_back((size_t)y * W + x);
    p0.c[0][at.back()] -= 15.0f * 0.015f;  // -15 times the luma noise
  }
  writePlanes(seq[0], p0);
  const ODNoiseProfile prof = whiteProfile(0.02f);
  auto err = [&](uint32_t speck) {
    ODParams p = base();
    p.manualProfile = &prof;
    p.temporalRadius = 0;
    p.lumaStrength = 1;
    p.speckRemoval = speck;
    Planes o = planesOf(rendered(seq, 0, p));
    double e = 0;
    for (size_t i : at) e += std::fabs(o.c[0][i] - clean.c[0][i]);
    return e / (double)at.size();
  };
  const double on = err(1), off = err(0);
  printf("  mean error at 50 dark specks: remove specks on %.5f, off %.5f\n", on, off);
  CHECK(off > 3 * on, "remove specks off kept no more of the specks: on %.5f, off %.5f", on, off);
}

// Motion threshold and edge ghost protection (temporal stage alone, luma
// 100 %, manual profile).
static const float gPhaseAmp = 0.03f;  // about 2 noise sigmas of luma
static const float gBumpAmp = 0.016f;
static void phaseScene(uint32_t x, uint32_t y, int frame, float ycc[3]) {
  // A static square whose texture shifts phase by a quarter turn per frame:
  // a change the motion search cannot explain, a few noise sigmas large.
  ycc[0] = 0.4f;
  ycc[1] = ycc[2] = 0.0f;
  const int lx = (int)x - 60, ly = (int)y - kSqY;
  if (lx >= 0 && lx < kSq && ly >= 0 && ly < kSq)
    ycc[0] = 0.5f + gPhaseAmp * sinf(0.9f * (float)lx + 1.5707963f * (float)frame) * cosf(0.35f * (float)ly);
}
static void bumpScene(uint32_t x, uint32_t y, int frame, float ycc[3]) {
  // A broad bump whose height changes from frame to frame: a mismatch at the
  // coarse levels only.
  const float dx = (float)x - 160.0f, dy = (float)y - 128.0f;
  ycc[0] = 0.4f + gBumpAmp * (float)frame * expf(-(dx * dx + dy * dy) / (2.0f * 40.0f * 40.0f));
  ycc[1] = ycc[2] = 0.0f;
}
static void testTemporalControls() {
  gTest = "temporal_controls";
  const ODNoiseProfile prof = whiteProfile(0.02f);
  Noise n;
  n.rgb = 0.02f;
  auto params = [&](float mthr, float gate) {
    ODParams p = base();
    p.manualProfile = &prof;
    p.lumaStrength = 1;
    p.spatialStrength = 0;
    p.temporalMotionThreshold = mthr;
    p.temporalEdgeGate = gate;
    return p;
  };
  auto square = [&](int sx) {
    return [sx](uint32_t x, uint32_t y) {
      return (int)x >= sx + 2 && (int)x < sx + kSq - 2 && (int)y >= kSqY + 2 && (int)y < kSqY + kSq - 2;
    };
  };
  // (1) A textured square the motion search follows: a higher threshold
  // averages more on it; static areas are left alone.
  {
    std::vector<Img> seq = sequence(movingScene, 5, n, 51);
    const Planes clean = cleanPlanes(movingScene, 2);
    auto stat = [](uint32_t x, uint32_t) { return x >= 200; };
    double m[3], st[3];
    const float t[3] = {0.5f, 1.0f, 2.0f};
    for (int i = 0; i < 3; ++i) {
      const std::vector<float> e = diff(planesOf(rendered(seq, 2, params(t[i], 0.75f))).c[0], clean.c[0]);
      m[i] = rmsOf(e, W, H, square(squareX(2)));
      st[i] = rmsOf(e, W, H, stat);
    }
    printf("  motion threshold 50 / 100 / 200 %%, tracked moving square: error %.5f / %.5f / %.5f; static flat %.5f / "
           "%.5f / %.5f\n", m[0], m[1], m[2], st[0], st[1], st[2]);
    CHECK(m[2] <= m[0], "motion threshold: the moving square is not cleaner at 200 %% (%.5f) than at 50 %% (%.5f)", m[2],
          m[0]);
    CHECK(st[2] <= st[0] * 1.001, "motion threshold: static flat residual at 200 %% (%.5f) above 50 %% (%.5f)", st[2],
          st[0]);
  }
  // (2) A change the motion search cannot explain: a stricter threshold
  // leaves the neighbours out (no smearing), a looser one blends them in.
  // (Measured at 0.03 / 0.05 / 0.08: the default is the lowest error at each;
  // stricter leaves more noise, looser smears.)
  {
    std::vector<Img> seq = sequence(phaseScene, 5, n, 52);
    const Planes clean = cleanPlanes(phaseScene, 2);
    double e[3];
    const float t[3] = {0.5f, 1.0f, 2.0f};
    for (int i = 0; i < 3; ++i)
      e[i] = rmsOf(diff(planesOf(rendered(seq, 2, params(t[i], 0.75f))).c[0], clean.c[0]), W, H, square(60));
    printf("  motion threshold 50 / 100 / 200 %%, texture changing by %.2f: error on it %.5f / %.5f / %.5f\n", gPhaseAmp,
           e[0], e[1], e[2]);
    CHECK(e[2] > e[1] && e[2] > e[0], "motion threshold 200 %% does not smear a change in (%.5f / %.5f / %.5f)", e[0],
          e[1], e[2]);
  }
  // (3) Edge ghost protection: a coarse-only mismatch over fine noise. The
  // more of the coarse rejection the fine levels take, the less fine noise is
  // averaged there.
  // (At a bump step of 0.002 or 0.004 no level rejects and the control has
  // nothing to hand down; measured equal.)
  {
    {
      std::vector<Img> seq = sequence(bumpScene, 5, n, 53);
      const Planes clean = cleanPlanes(bumpScene, 2);
      auto core = [](uint32_t x, uint32_t y) {
        const float dx = (float)x - 160.0f, dy = (float)y - 128.0f;
        return dx * dx + dy * dy < 40.0f * 40.0f;
      };
      double f[3];
      const float g[3] = {0.0f, 0.75f, 1.0f};
      for (int i = 0; i < 3; ++i) {
        const std::vector<float> e = diff(planesOf(rendered(seq, 2, params(1.0f, g[i]))).c[0], clean.c[0]);
        std::vector<float> fe = diff(e, boxBlur(e, W, H, 1));
        f[i] = rmsOf(fe, W, H, core);
      }
      printf("  edge ghost protection 0 / 75 / 100 %%, bump step %.3f: fine residual %.5f / %.5f / %.5f\n", gBumpAmp,
             f[0], f[1], f[2]);
      CHECK(f[0] < f[1] && f[1] < f[2], "edge ghost protection: fine residual next to a coarse change %.5f / %.5f / %.5f",
            f[0], f[1], f[2]);
    }
  }
}

// Sharpen detail: band gains, coring threshold, overshoot.
static void testSharpenControls() {
  gTest = "sharpen_controls";
  // Bands 0 and 1 rise with Sharpen fine / medium.
  {
    auto scene = [](uint32_t x, uint32_t y, int, float ycc[3]) {
      ycc[0] = 0.3f + (x > W / 2 ? 0.2f : 0.0f) + 0.04f * sinf(1.4f * (float)x) * cosf(1.1f * (float)y) +
               0.04f * sinf(0.45f * (float)x + 0.3f * (float)y);
      ycc[1] = ycc[2] = 0.0f;
    };
    Noise n;
    n.rgb = 0.003f;
    std::vector<Img> seq = sequence(scene, 1, n, 61);
    const ODNoiseProfile prof = whiteProfile(0.003f);
    auto energy = [&](int which, float g, int band) {
      ODParams p = base();
      p.manualProfile = &prof;
      p.temporalRadius = 0;
      p.spatialStrength = 0;
      p.sharpen = 0.6f;
      p.sharpenBand[which] = g;
      return bandRms(planesOf(rendered(seq, 0, p)).c[0], W, H, band);
    };
    for (int b = 0; b < 2; ++b) {
      const double e0 = energy(b, 0, b), e1 = energy(b, 1, b), e2 = energy(b, 2.5f, b);
      printf("  sharpen %s 0 / 100 / 250 %%: band %d energy %.5f / %.5f / %.5f\n", b ? "medium" : "fine", b, e0, e1, e2);
      CHECK(e0 < e1 && e1 < e2, "sharpen %s: band %d energy %.5f / %.5f / %.5f", b ? "medium" : "fine", b, e0, e1, e2);
    }
  }
  // Sharpen noise floor: a lower threshold sharpens more of the grain.
  {
    Noise n;
    n.rgb = 0.02f;
    std::vector<Img> seq = sequence(flatScene, 1, n, 62);
    const Planes clean = cleanPlanes(flatScene, 0);
    const ODNoiseProfile prof = whiteProfile(0.02f);
    auto err = [&](float coring) {
      ODParams p = base();
      p.manualProfile = &prof;
      p.temporalRadius = 0;
      p.spatialStrength = 0;
      p.sharpen = 1;
      p.sharpenCoring = coring;
      return rmsOf(diff(planesOf(rendered(seq, 0, p)).c[0], clean.c[0]), W, H);
    };
    const double a = err(0.5f), b = err(1), c = err(3);
    printf("  sharpen noise floor 50 / 100 / 300 %%: flat error %.5f / %.5f / %.5f\n", a, b, c);
    CHECK(a > b && b > c, "sharpen noise floor: flat error %.5f / %.5f / %.5f", a, b, c);
  }
  // Sharpen halo limit: overshoot beyond the 3x3 range of the input.
  {
    auto scene = [](uint32_t x, uint32_t, int, float ycc[3]) {
      ycc[0] = x < W / 2 ? 0.2f : 0.8f;
      ycc[1] = ycc[2] = 0.0f;
    };
    Noise n;
    n.rgb = 0.001f;
    std::vector<Img> seq = sequence(scene, 1, n, 63);
    const Planes in = planesOf(seq[0]);
    const ODNoiseProfile prof = whiteProfile(0.001f);
    auto over = [&](float halo) {
      ODParams p = base();
      p.manualProfile = &prof;
      p.temporalRadius = 0;
      p.spatialStrength = 0;
      p.sharpen = 1;
      p.sharpenHalo = halo;
      const Planes o = planesOf(rendered(seq, 0, p));
      double m = 0;
      for (uint32_t y = 1; y + 1 < H; ++y)
        for (uint32_t x = 1; x + 1 < W; ++x) {
          float mn = 1e9f, mx = -1e9f;
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
              const float v = in.c[0][(size_t)((int)y + dy) * W + (uint32_t)((int)x + dx)];
              mn = std::min(mn, v);
              mx = std::max(mx, v);
            }
          const float v = o.c[0][(size_t)y * W + x];
          m = std::max(m, (double)std::max(v - mx, mn - v));
        }
      return m;
    };
    const double h0 = over(0), h1 = over(1), h5 = over(5);
    printf("  sharpen halo limit 0 / 100 / 500 %%: overshoot %.5f / %.5f / %.5f\n", h0, h1, h5);
    CHECK(h0 <= 1e-3, "sharpen halo limit 0 overshoots by %.5f", h0);
    CHECK(h0 < h1 && h1 < h5, "sharpen halo limit: overshoot %.5f / %.5f / %.5f", h0, h1, h5);
  }
}

// Hold profile: shot A, then a cut to shot B (contiguous keys).
static void testHold() {
  gTest = "hold";
  const int kCut = 12, kTotal = 20;
  Noise na, nb;
  na.rgb = 0.01f;
  nb.rgb = 0.035f;
  auto mirrored = [](uint32_t x, uint32_t y, int f, float ycc[3]) { mixedScene(W - 1 - x, y, f, ycc); };
  std::vector<Img> seq;
  for (int f = 0; f < kTotal; ++f)
    seq.push_back(f < kCut ? noisyFrame(mixedScene, f, na, 71) : noisyFrame(mirrored, f, nb, 72));
  Img out = makeImg();
  auto y0 = [](const ODStats& st) { return st.profile.sigma[0][0][3]; };
  // hold: per render index, 0 / 1.
  auto play = [&](const std::function<int(int)>& hold, std::vector<float>* prof, bool async) {
    ODSession* s = od_session_create(gCtx);
    for (int f = 0; f < kTotal; ++f) {
      ODParams p = base();
      p.waitUntilCompleted = async ? 0 : 1;
      p.profileHold = (uint32_t)hold(f);
      ODStats st = {};
      CHECK(renderAt(seq, f, p, out, s, &st, 500), "render %d failed", f);
      if (async) od_debug_session_drain(s);
      prof->push_back(y0(st));
    }
    od_session_destroy(s);
  };
  std::vector<float> off, on, mid;
  play([](int) { return 0; }, &off, false);
  play([](int) { return 1; }, &on, false);
  play([&](int f) { return f >= 10 && f < 16 ? 1 : 0; }, &mid, false);
  printf("  Y band 0 at frames 7 / 11 / 12 (cut) / 13 / 19: hold off %.5f %.5f %.5f %.5f %.5f; on %.5f %.5f %.5f %.5f "
         "%.5f; on from 10 to 15 %.5f %.5f %.5f %.5f %.5f\n",
         off[7], off[11], off[12], off[13], off[19], on[7], on[11], on[12], on[13], on[19], mid[7], mid[11], mid[12],
         mid[13], mid[19]);
  CHECK(off[13] > 2 * off[11], "hold off: no restart at the cut (%.5f -> %.5f)", off[11], off[13]);
  bool moved = false, constant = true;
  for (int f = 1; f < 8; ++f) moved = moved || on[f] != on[0];
  for (int f = 9; f < kTotal; ++f) constant = constant && on[f] == on[9];
  CHECK(moved, "hold on: the profile never moved in its first renders");
  CHECK(constant, "hold on: the profile moved after it was established (frame 9 %.5f, 13 %.5f, 19 %.5f)", on[9], on[13],
        on[19]);
  CHECK(std::fabs(on[13] / on[11] - 1) < 0.01f, "hold on: restart at the cut (%.5f -> %.5f)", on[11], on[13]);
  bool held = true;
  for (int f = 10; f < 16; ++f) held = held && mid[f] == mid[10];
  CHECK(held, "hold switched on at frame 10: the profile moved before it went off (%.5f, %.5f at the cut)", mid[10],
        mid[12]);
  CHECK(std::fabs(mid[19] / mid[10] - 1) > 0.2f, "hold switched off at frame 16: the profile did not follow (%.5f -> "
        "%.5f)", mid[10], mid[19]);
  // Estimates still in flight when Hold goes on do not move the held profile.
  {
    Noise loud;
    loud.rgb = 0.025f;
    std::vector<Img> more;  // shot A's picture, louder noise: no cut, estimates far from the slot
    for (int f = 0; f < 8; ++f) more.push_back(noisyFrame(mixedScene, kCut + f, loud, 73));
    std::vector<Img> all(seq.begin(), seq.begin() + kCut);
    all.insert(all.end(), more.begin(), more.end());
    for (int pass = 0; pass < 2; ++pass) {
      ODSession* s = od_session_create(gCtx);
      ODParams p = base();
      p.waitUntilCompleted = 0;
      ODStats st = {};
      for (int f = 0; f < kCut; ++f) CHECK(renderAt(all, f, p, out, s, &st, 900), "render failed");
      od_debug_session_drain(s);
      for (int f = kCut; f < kCut + 4; ++f) CHECK(renderAt(all, f, p, out, s, &st, 900), "render failed");
      p.profileHold = pass == 1 ? 1 : 0;  // pass 0: the control, hold stays off
      CHECK(renderAt(all, kCut + 4, p, out, s, &st, 900), "render failed");
      const float at = y0(st);
      od_debug_session_drain(s);
      CHECK(renderAt(all, kCut + 5, p, out, s, &st, 900), "render failed");
      od_debug_session_drain(s);
      const float after = y0(st);
      printf("  %s: Y band 0 %.6f when it went on, %.6f once the estimates in flight landed\n",
             pass ? "hold on with estimates in flight" : "control (hold off)", at, after);
      if (pass == 1) CHECK(after == at, "hold on: estimates in flight moved the profile (%.6f -> %.6f)", at, after);
      od_session_destroy(s);
    }
  }
}

// Out-of-range and non-finite values: finite output; NaN is the neutral
// value, anything else the clamped one.
static void testRanges() {
  gTest = "ranges";
  Noise n;
  n.rgb = 0.02f;
  std::vector<Img> seq = sequence(mixedScene, 5, n, 81);
  const ODNoiseProfile prof = whiteProfile(0.02f);
  ODParams p0 = base();
  p0.manualProfile = &prof;
  p0.sharpen = 0.3f;
  const Img ref = rendered(seq, 2, p0);
  struct F {
    const char* name;
    float* (*at)(ODParams*);
    float lo, hi;
  };
  const F fields[] = {
      {"bandNoiseLevel[0][0]", [](ODParams* p) { return &p->bandNoiseLevel[0][0]; }, 0.25f, 4},
      {"bandNoiseLevel[1][3]", [](ODParams* p) { return &p->bandNoiseLevel[1][3]; }, 0.25f, 4},
      {"channelNoiseLevel[2]", [](ODParams* p) { return &p->channelNoiseLevel[2]; }, 0.25f, 4},
      {"bandAmount[0][1]", [](ODParams* p) { return &p->bandAmount[0][1]; }, 0, 1},
      {"bandAmount[1][2]", [](ODParams* p) { return &p->bandAmount[1][2]; }, 0, 1},
      {"grainFloor", [](ODParams* p) { return &p->grainFloor; }, 0, 10},
      {"temporalMotionThreshold", [](ODParams* p) { return &p->temporalMotionThreshold; }, 0.5f, 2},
      {"temporalEdgeGate", [](ODParams* p) { return &p->temporalEdgeGate; }, 0, 1},
      {"sharpenBand[0]", [](ODParams* p) { return &p->sharpenBand[0]; }, 0, 2.5f},
      {"sharpenBand[1]", [](ODParams* p) { return &p->sharpenBand[1]; }, 0, 2.5f},
      {"sharpenCoring", [](ODParams* p) { return &p->sharpenCoring; }, 0.5f, 3},
      {"sharpenHalo", [](ODParams* p) { return &p->sharpenHalo; }, 0, 5},
  };
  int good = 0, total = 0;
  for (const F& f : fields) {
    ODParams a = p0;
    *f.at(&a) = NAN;
    const Img nan = rendered(seq, 2, a);
    ++total;
    const bool okNan = sameBytes(nan, ref);
    CHECK(okNan, "%s = NaN does not render the neutral picture", f.name);
    good += okNan;
    for (int side = 0; side < 2; ++side) {
      ODParams c = p0;
      *f.at(&c) = side ? f.hi : f.lo;
      const Img clamped = rendered(seq, 2, c);
      // Beyond the range: the clamped value. Not finite: the neutral value
      // (opendenoise.h), as for noiseLevel.
      for (float v : side ? std::vector<float>{INFINITY, 1e9f} : std::vector<float>{-INFINITY, -1e9f}) {
        ODParams b = p0;
        *f.at(&b) = v;
        const Img o = rendered(seq, 2, b);
        ++total;
        const bool ok = allFinite(o) && sameBytes(o, std::isfinite(v) ? clamped : ref);
        CHECK(ok, "%s = %g does not render the %s picture", f.name, v, std::isfinite(v) ? "clamped" : "neutral");
        good += ok;
      }
    }
  }
  // Switches: anything but 0 is on.
  {
    ODParams a = p0, b = p0;
    a.speckRemoval = 7;
    b.speckRemoval = 1;
    ++total;
    const bool ok = sameBytes(rendered(seq, 2, a), rendered(seq, 2, b));
    CHECK(ok, "speckRemoval 7 is not on");
    good += ok;
  }
  printf("  %d of %d out-of-range or non-finite cases render the neutral or clamped picture\n", good, total);
}

// The bands step on frames of every shape, in place, both formats, after
// either denoising stage, never in the Motion and Profile views; its time
// counts under spatial.
static void testBandsStep() {
  gTest = "bands_step";
  struct Size {
    uint32_t w, h;
  };
  for (Size sz : {Size{333, 197}, Size{17, 17}, Size{12, 12}, Size{W, H}}) {
    for (ODPixelFormat fmt : {OD_PIXEL_BGRA16F, OD_PIXEL_BGRA32F}) {
      Noise n;
      n.rgb = 0.02f;
      std::vector<Img> seq;
      for (int f = 0; f < 5; ++f) seq.push_back(noisyFrame(mixedScene, f, n, 91, fmt, sz.w, sz.h, 0.0f));
      // Alpha that varies: it must come through bit-exact.
      for (Img& im : seq)
        for (uint32_t y = 0; y < sz.h; ++y)
          for (uint32_t x = 0; x < sz.w; ++x) im.set(x, y, 3, 0.25f + 0.5f * (float)((x * 7 + y * 3) % 5) / 4.0f);
      const ODNoiseProfile prof = whiteProfile(0.02f);
      ODParams p = base();
      p.manualProfile = &prof;
      p.lumaStrength = 1;
      const Img d = rendered(seq, 2, p);
      p.bandAmount[0][0] = 0.3f;
      p.bandAmount[0][3] = 0.6f;
      p.bandAmount[1][1] = 0.0f;
      const Img o = rendered(seq, 2, p);
      // In place: the current frame's buffer is the output.
      std::vector<Img> inPlace = seq;
      inPlace[2] = copyImg(seq[2]);
      CHECK(renderAt(inPlace, 2, p, inPlace[2]), "in-place render failed");
      bool alpha = true;
      for (uint32_t y = 0; y < sz.h; ++y)
        for (uint32_t x = 0; x < sz.w; ++x) alpha = alpha && o.get(x, y, 3) == seq[2].get(x, y, 3);
      const char* f = fmt == OD_PIXEL_BGRA16F ? "16f" : "32f";
      CHECK(allFinite(o), "%ux%u %s: non-finite output", sz.w, sz.h, f);
      CHECK(alpha, "%ux%u %s: alpha changed", sz.w, sz.h, f);
      CHECK(!sameBytes(o, d), "%ux%u %s: the band amounts changed nothing", sz.w, sz.h, f);
      CHECK(sameBytes(inPlace[2], o), "%ux%u %s: in place differs from a separate output", sz.w, sz.h, f);
      if (sz.w == 12) {
        // No coarser level fits. The established convention assigns the
        // whole residual to the top band; the fine-grain change must not
        // silently substitute band 0 for the editor's band-3 amount.
        p.grainFloor = 0;
        const Img top = rendered(seq, 2, p);
        p.bandAmount[0][0] = 0.9f;
        CHECK(sameBytes(top, rendered(seq, 2, p)), "%s: tiny frame used the fine amount instead of the top", f);
      }
    }
  }
  Noise n;
  n.rgb = 0.02f;
  std::vector<Img> seq = sequence(mixedScene, 5, n, 92);
  const Planes in = planesOf(seq[2]);
  const ODNoiseProfile prof = whiteProfile(0.02f);
  // After the temporal stage alone, and after the spatial stage alone: every
  // amount 0 gives the input back.
  for (int mode = 0; mode < 2; ++mode) {
    ODParams p = base();
    p.manualProfile = &prof;
    if (mode == 0) p.spatialStrength = 0;
    else p.temporalRadius = 0;
    const Planes d = planesOf(rendered(seq, 2, p));
    for (int g = 0; g < 2; ++g)
      for (int b = 0; b < OD_BANDS; ++b) p.bandAmount[g][b] = 0;
    const Planes z = planesOf(rendered(seq, 2, p));
    const double r = rmsOf(diff(z.c[0], in.c[0]), W, H) / rmsOf(diff(d.c[0], in.c[0]), W, H);
    printf("  %s alone, every amount 0: off the input by %.4f of what it removes\n", mode ? "spatial" : "temporal", r);
    CHECK(r < 0.02, "%s alone: every amount 0 is off the input by %.4f", mode ? "spatial" : "temporal", r);
  }
  // Views: Motion and Profile are not touched; Noise removed shows the amounts.
  for (int v : {OD_VIEW_MOTION, OD_VIEW_PROFILE, OD_VIEW_NOISE}) {
    ODParams p = base();
    p.view = v;
    const Img d = rendered(seq, 2, p);
    p.bandAmount[0][0] = 0.5f;
    p.bandAmount[1][2] = 0.5f;
    const Img o = rendered(seq, 2, p);
    if (v == OD_VIEW_NOISE) CHECK(!sameBytes(o, d), "the Noise removed view does not show the band amounts");
    else CHECK(sameBytes(o, d), "view %d: the band amounts changed it", v);
  }
  // The Motion view without the temporal stage (radius 0, temporal strength 0)
  // cannot replace the picture: the denoised picture goes out, and it must be
  // the one the amounts make, as in the Output view.
  for (int which = 0; which < 2; ++which) {
    ODParams p = base();
    p.manualProfile = &prof;
    if (which == 0) p.temporalRadius = 0;
    else p.temporalStrength = 0;
    p.bandAmount[0][0] = 0.0f;
    p.bandAmount[1][1] = 0.5f;
    const Img out = rendered(seq, 2, p);
    p.view = OD_VIEW_MOTION;
    const Img motion = rendered(seq, 2, p);
    for (int g = 0; g < 2; ++g)
      for (int b = 0; b < OD_BANDS; ++b) p.bandAmount[g][b] = 1.0f;
    const Img motionDefault = rendered(seq, 2, p);
    const char* why = which == 0 ? "radius 0" : "temporal strength 0";
    CHECK(sameBytes(motion, out), "Motion view, %s: the picture differs from the Output view with the same amounts", why);
    CHECK(!sameBytes(motion, motionDefault), "Motion view, %s: the band amounts changed nothing", why);
  }
  // Timing mode: the step's time lands under spatial (here the spatial stage
  // itself does not run).
  {
    ODParams p = base();
    p.manualProfile = &prof;
    p.spatialStrength = 0;
    p.debugFlags = OD_DEBUG_STAGE_TIMINGS;
    p.grainFloor = 0;  // isolate band amounts from the fine-grain pass
    ODStats st = {};
    Img out = makeImg();
    CHECK(renderAt(seq, 2, p, out, nullptr, &st), "timed render failed");
    const double without = st.stageMs[OD_STAGE_SPATIAL];
    p.bandAmount[0][1] = 0.5f;
    CHECK(renderAt(seq, 2, p, out, nullptr, &st), "timed render failed");
    printf("  timing mode, temporal only: spatial %.3f ms without the bands step, %.3f ms with it\n", without,
           st.stageMs[OD_STAGE_SPATIAL]);
    CHECK(without == 0 && st.stageMs[OD_STAGE_SPATIAL] > 0, "the bands step's time is not under spatial (%.3f, %.3f)",
          without, st.stageMs[OD_STAGE_SPATIAL]);
  }
}

// Noise levels where the spatial filter is not the stage that decides: the
// temporal merge alone, the sharpen coring threshold, and the coarse-chroma
// persistence measure (testLevels runs the spatial filter).
static void testLevelsInStages() {
  gTest = "levels_in_stages";
  const ODNoiseProfile prof = whiteProfile(0.02f);
  // (1) Temporal merge alone (spatial 0, radius 2): a lower level rejects
  // more of the neighbours and leaves more noise; a higher one never less.
  {
    Noise n;
    n.rgb = 0.02f;
    std::vector<Img> seq = sequence(flatScene, 5, n, 101);
    const Planes clean = cleanPlanes(flatScene, 2);
    struct R {
      double y0, cb, cr;
    };
    auto run = [&](const ODParams& p) {
      Planes o = planesOf(rendered(seq, 2, p));
      R r;
      r.y0 = bandRms(diff(o.c[0], clean.c[0]), W, H, 0);
      r.cb = rmsOf(diff(o.c[1], clean.c[1]), W, H);
      r.cr = rmsOf(diff(o.c[2], clean.c[2]), W, H);
      return r;
    };
    ODParams g = base();
    g.manualProfile = &prof;
    g.lumaStrength = 1;
    g.spatialStrength = 0;
    const R d = run(g);
    ODParams p = g;
    p.bandNoiseLevel[0][0] = 0.5f;
    const R lo = run(p);
    p.bandNoiseLevel[0][0] = 2.0f;
    const R hi = run(p);
    printf("  temporal merge alone, luma fine level 50 / 100 / 200 %%: fine residual %.6f / %.6f / %.6f\n", lo.y0, d.y0,
           hi.y0);
    CHECK(lo.y0 > 1.05 * d.y0 && hi.y0 <= 1.001 * d.y0, "temporal merge alone, luma fine level: %.6f / %.6f / %.6f",
          lo.y0, d.y0, hi.y0);
    p = g;
    p.channelNoiseLevel[1] = 0.5f;
    const R cb = run(p);
    printf("  temporal merge alone, Cb noise level 50 %%: Cb residual %.6f -> %.6f, Cr %.6f -> %.6f\n", d.cb, cb.cb, d.cr,
           cb.cr);
    CHECK(cb.cb > 1.05 * d.cb, "temporal merge alone, Cb noise level 50 %%: Cb residual %.6f -> %.6f", d.cb, cb.cb);
  }
  // (2) Sharpen alone (radius 0, spatial 0): the coring threshold follows
  // the luma level of the band it sharpens (fine at playback quality, fine
  // and medium at full quality), as Sharpen noise floor does.
  {
    Noise n;
    n.rgb = 0.02f;
    std::vector<Img> seq = sequence(flatScene, 1, n, 102);
    const Planes clean = cleanPlanes(flatScene, 0);
    for (int band = 0; band < 2; ++band) {
      auto err = [&](float level) {
        ODParams p = base();
        p.manualProfile = &prof;
        p.temporalRadius = 0;
        p.spatialStrength = 0;
        p.sharpen = 1;
        p.quality = band;  // band 1 is sharpened at full quality only
        p.sharpenBand[1 - band] = 0.0f;  // the other band adds nothing
        p.bandNoiseLevel[0][band] = level;
        return rmsOf(diff(planesOf(rendered(seq, 0, p)).c[0], clean.c[0]), W, H);
      };
      const double a = err(0.5f), b = err(1), c = err(3);
      printf("  sharpen alone, luma %s level 50 / 100 / 300 %%: flat error %.5f / %.5f / %.5f\n",
             band ? "medium" : "fine", a, b, c);
      // At 100 % the medium band's coring already keeps almost all of the
      // grain here: a higher level can only keep the rest.
      CHECK(a > 1.005 * b && c <= b, "sharpen alone, luma %s level: flat error %.5f / %.5f / %.5f",
            band ? "medium" : "fine", a, b, c);
    }
  }
  // (3) Chroma after the merge (radius 2, spatial and coarse-chroma
  // persistence measure running): Cb and Cr noise levels of x render exactly
  // what a profile with x times the chroma noise renders, since every
  // threshold that decides what chroma noise is (merge, spatial filter, the
  // share of the chroma noise the merge left) takes the corrected sigma and
  // nothing else reads the chroma sigma. Powers of two, so the products are
  // exact in either order: the same bytes.
  {
    Noise n;
    n.rgb = 0.02f;
    std::vector<Img> seq = sequence(mixedScene, 5, n, 103);
    for (float x : {2.0f, 0.5f}) {
      ODNoiseProfile scaled = prof;
      for (int c = 1; c < 3; ++c) {
        for (int b = 0; b < OD_BANDS; ++b)
          for (int l = 0; l < OD_LEVELS; ++l) scaled.sigma[c][b][l] *= x;
        for (int l = 0; l < OD_LEVELS; ++l) scaled.sigmaFull[c][l] *= x;
      }
      ODParams a = base();
      a.manualProfile = &prof;
      a.channelNoiseLevel[1] = a.channelNoiseLevel[2] = x;
      ODParams b = base();
      b.manualProfile = &scaled;
      const Img oa = rendered(seq, 2, a), ob = rendered(seq, 2, b);
      CHECK(sameBytes(oa, ob), "Cb and Cr noise levels %.1f differ from a profile with %.1f times the chroma noise", x, x);
      ODParams d = base();
      d.manualProfile = &prof;
      CHECK(!sameBytes(oa, rendered(seq, 2, d)), "Cb and Cr noise levels %.1f changed nothing: the test says nothing", x);
    }
  }
}

// Luma grain floor where the spatial stage does not apply it: the temporal
// merge when spatial is 0, and sharpen's estimate of the noise left.
static void testGrainFloorPaths() {
  gTest = "grain_floor_paths";
  Noise n;
  n.rgb = 0.002f;
  std::vector<Img> seq = sequence(flatScene, 5, n, 111);
  const Planes clean = cleanPlanes(flatScene, 2);
  const ODNoiseProfile prof = whiteProfile(0.002f);
  const double in = rmsOf(diff(planesOf(seq[2]).c[0], clean.c[0]), W, H);
  // (1) Temporal merge alone.
  {
    auto res = [&](float floor) {
      ODParams p = base();
      p.manualProfile = &prof;
      p.lumaStrength = 1;
      p.spatialStrength = 0;
      p.grainFloor = floor;
      return rmsOf(diff(planesOf(rendered(seq, 2, p)).c[0], clean.c[0]), W, H);
    };
    const double f0 = res(0), f1 = res(1), f4 = res(4);
    printf("  temporal merge alone, luma noise %.6f; residual with grain floor 0 / 100 / 400 %%: %.6f / %.6f / %.6f\n",
           in, f0, f1, f4);
    CHECK(f0 < f1 && f1 < f4, "temporal merge alone: grain floor not monotonic: %.6f / %.6f / %.6f", f0, f1, f4);
    CHECK(f4 >= 0.9 * std::min(in, 4 * 7.5e-4), "temporal merge alone: grain floor 400 %% kept %.6f of %.6f", f4, in);
  }
  if (!grainProfileV2()) {
    // Preserve the legacy all-band floor contract when the diagnostic mode is off.
    ODParams a = base();
    a.manualProfile = &prof;
    a.lumaStrength = 1;
    a.grainFloor = 4;
    a.sharpen = 1;
    ODParams b = a;
    b.lumaStrength = 0;
    b.grainFloor = 1;
    const Img oa = rendered(seq, 2, a), ob = rendered(seq, 2, b);
    a.sharpen = b.sharpen = 0;
    const Img na = rendered(seq, 2, a), nb = rendered(seq, 2, b);
    CHECK(sameBytes(na, nb), "grain floor 400 %% and luma amount 0 differ before sharpen");
    CHECK(sameBytes(oa, ob), "sharpen after grain floor 400 %% differs from luma amount 0");
    CHECK(!sameBytes(oa, na), "sharpen changed nothing: the test says nothing");
    return;
  }
  // (2) At a floor above the noise, fine grain survives but medium/coarse
  // correction still follows Luma. This was an exact identity on the old path.
  for (int q = 0; q <= 1; ++q) for (int mode = 0; mode < 3; ++mode) {
    ODParams p = base();
    p.manualProfile = &prof; p.quality = q; p.grainFloor = 4;
    if (mode == 0) p.spatialStrength = 0;
    if (mode == 1) p.temporalRadius = 0;
    p.lumaStrength = .8f;
    const Planes a = planesOf(rendered(seq, 2, p));
    p.lumaStrength = .9f;
    const Planes b = planesOf(rendered(seq, 2, p));
    p.lumaStrength = 1;
    const Planes c = planesOf(rendered(seq, 2, p));
    const auto input = planesOf(seq[2]);
    const double ab = rmsOf(diff(a.c[0], b.c[0]), W, H);
    const double bc = rmsOf(diff(b.c[0], c.c[0]), W, H);
    CHECK(ab > 1e-6 && bc > 1e-6, "q%d mode%d: floor killed slider (%.3g, %.3g)", q, mode, ab, bc);
    const double fine = fineRms(diff(c.c[0], clean.c[0]));
    const double before = fineRms(diff(input.c[0], clean.c[0]));
    CHECK(fine > .85 * before, "q%d mode%d: fine grain lost", q, mode);
    CHECK(coarseRms(diff(c.c[0], clean.c[0])) < coarseRms(diff(a.c[0], clean.c[0])),
          "q%d mode%d: coarse noise does not follow Luma", q, mode);
    p.sharpen = 1;
    const Planes sharp = planesOf(rendered(seq, 2, p));
    CHECK(fineRms(diff(sharp.c[0], clean.c[0])) < 1.1 * fine,
          "q%d mode%d: sharpen amplified retained fine grain", q, mode);
  }
}

// The bands step and sharpen: the step keeps its input's alpha, where the
// spatial stage left 1/sqrt(count) for sharpen, and sharpen counts the noise
// the amounts left in place.
static void testBandsAndSharpen() {
  gTest = "bands_and_sharpen";
  Noise n;
  n.rgb = 0.02f;
  std::vector<Img> seq = sequence(mixedScene, 5, n, 121);
  const ODNoiseProfile prof = whiteProfile(0.02f);
  // (1) A chroma amount just below 100 %: the step runs, copies the luma, and
  // sharpen (which reads the luma and the alpha) must sharpen exactly as
  // without the step.
  for (int q = 0; q <= 1; ++q) {
    ODParams p = base();
    p.manualProfile = &prof;
    p.quality = q;
    p.sharpen = 0.6f;
    const Planes d = planesOf(rendered(seq, 2, p));
    p.bandAmount[1][3] = 0.999f;
    const Planes o = planesOf(rendered(seq, 2, p));
    double dy = 0, dc = 0;
    for (size_t i = 0; i < o.c[0].size(); ++i) {
      dy = std::max(dy, (double)std::fabs(o.c[0][i] - d.c[0][i]));
      dc = std::max({dc, (double)std::fabs(o.c[1][i] - d.c[1][i]), (double)std::fabs(o.c[2][i] - d.c[2][i])});
    }
    printf("  q%d, sharpen 0.6, chroma very coarse amount 99.9 %%: luma moved by at most %.2g, chroma by %.2g\n", q, dy,
           dc);
    CHECK(dy < 2e-6, "q%d: sharpen after the bands step moved the luma by %.2g (alpha or luma not kept)", q, dy);
    CHECK(dc > 0, "q%d: the chroma amount changed nothing: the test says nothing", q);
  }
  // (2) Every luma amount at 30 % leaves the noise Luma 30 % leaves (testAmounts,
  // with the grain floor off). Sharpen must count it as left and sharpen as
  // after Luma 30 %; assuming the full strength removed it would lower the
  // coring threshold and sharpen the grain that was kept.
  for (int q = 0; q <= 1; ++q) {
    ODParams a = base();
    a.manualProfile = &prof;
    a.quality = q;
    a.lumaStrength = 1;
    a.grainFloor = 0;
    for (int b = 0; b < OD_BANDS; ++b) a.bandAmount[0][b] = 0.3f;
    ODParams s = base();
    s.manualProfile = &prof;
    s.quality = q;
    s.lumaStrength = 0.3f;
    s.grainFloor = 0;
    const Planes a0 = planesOf(rendered(seq, 2, a)), s0 = planesOf(rendered(seq, 2, s));
    a.sharpen = s.sharpen = 1;
    const Planes a1 = planesOf(rendered(seq, 2, a)), s1 = planesOf(rendered(seq, 2, s));
    const double before = rmsOf(diff(a0.c[0], s0.c[0]), W, H);
    const double after = rmsOf(diff(a1.c[0], s1.c[0]), W, H);
    const double sharpened = rmsOf(diff(s1.c[0], s0.c[0]), W, H);
    printf("  q%d, every luma amount 30 %% against Luma 30 %%: off by %.6f before sharpen, %.6f after (sharpen adds "
           "%.6f)\n", q, before, after, sharpened);
    CHECK(sharpened > 1e-3, "q%d: sharpen added little (%.6f): the test says nothing", q, sharpened);
    CHECK(after < before + 0.25 * sharpened, "q%d: sharpen treats the noise the amounts left as removed: off by %.6f "
          "after sharpen against %.6f before (sharpen adds %.6f)", q, after, before, sharpened);
  }
}

// ---- the default picture --------------------------------------------------------
// The renders at the neutral values, pinned as FNV-1a hashes of the output
// bytes. testNeutral shows that advanced = 0, NaN and the defaults are the
// same render; this shows that render is the picture recorded here, so a
// change of a stage's neutral branch that moves the default picture by one
// rounding step fails. The hashes are those of the core before the advanced
// controls (commit 79e190d) on the GPU named below; on another GPU (different
// rounding) the check is skipped. Re-record (`test_advanced
// --print-default-picture`) only after a deliberate change of the default
// picture, and say so in the commit, as for tests/spatial/pinned_reference.h.
struct PinCase {
  const char* name;
  ODPixelFormat fmt;
  float noise;
  bool manual;
  int quality, radius;
  float spatial, sharpen, luma;
  int view;
};
static const PinCase kPinCases[] = {
    {"auto, defaults", OD_PIXEL_BGRA32F, 0.02f, false, 1, 2, 0.5f, 0.0f, 0.8f, OD_VIEW_OUTPUT},
    {"auto, low noise", OD_PIXEL_BGRA32F, 0.003f, false, 1, 2, 0.5f, 0.0f, 0.8f, OD_VIEW_OUTPUT},
    {"temporal alone, low noise", OD_PIXEL_BGRA32F, 0.003f, true, 1, 2, 0.0f, 0.0f, 0.8f, OD_VIEW_OUTPUT},
    {"temporal alone, q0, luma 100 %", OD_PIXEL_BGRA32F, 0.02f, true, 0, 2, 0.0f, 0.0f, 1.0f, OD_VIEW_OUTPUT},
    {"spatial alone", OD_PIXEL_BGRA32F, 0.02f, true, 1, 0, 0.5f, 0.0f, 0.8f, OD_VIEW_OUTPUT},
    {"spatial alone, q0, low noise", OD_PIXEL_BGRA32F, 0.003f, true, 0, 0, 1.0f, 0.0f, 1.0f, OD_VIEW_OUTPUT},
    {"sharpen 0.6", OD_PIXEL_BGRA32F, 0.02f, true, 1, 2, 0.5f, 0.6f, 0.8f, OD_VIEW_OUTPUT},
    {"sharpen 1, q0, low noise", OD_PIXEL_BGRA32F, 0.003f, true, 0, 2, 0.5f, 1.0f, 0.8f, OD_VIEW_OUTPUT},
    {"noise view", OD_PIXEL_BGRA32F, 0.02f, true, 1, 2, 0.5f, 0.0f, 0.8f, OD_VIEW_NOISE},
    {"motion view", OD_PIXEL_BGRA32F, 0.02f, true, 1, 2, 0.5f, 0.0f, 0.8f, OD_VIEW_MOTION},
    {"16f, auto, defaults", OD_PIXEL_BGRA16F, 0.02f, false, 1, 2, 0.5f, 0.0f, 0.8f, OD_VIEW_OUTPUT},
    {"16f, q0, low noise", OD_PIXEL_BGRA16F, 0.003f, true, 0, 2, 0.5f, 0.0f, 0.8f, OD_VIEW_OUTPUT},
};
static const int kPinCount = (int)(sizeof(kPinCases) / sizeof(kPinCases[0]));
static const char* const kPinDevice = "Apple M5 Max";
static const uint64_t kPinHash[kPinCount] = {
    0x4b5fc7deab271dd1ull,  // auto, defaults
    0xf37bb10e23e3c3a4ull,  // auto, low noise
    0x3ae3caf0b131be22ull,  // temporal alone, low noise
    0x147d3faa53b3c807ull,  // temporal alone, q0, luma 100 %
    0x0879e4d502a7cd2aull,  // spatial alone
    0x13d05bbce8f8c114ull,  // spatial alone, q0, low noise
    0x67939a3067e5355aull,  // sharpen 0.6
    0xbb7757e86bed7d57ull,  // sharpen 1, q0, low noise
    0xb2b83de1ac5e314eull,  // noise view
    0xede92127d53e7b22ull,  // motion view
    0x2c530b1410c590f1ull,  // 16f, auto, defaults
    0x02e9841c1f45c73full,  // 16f, q0, low noise
};

// Deliberate diagnostic-mode reference, recorded independently. The legacy
// hashes above remain unchanged;
static const uint64_t kPinHashV2[kPinCount] = {
    0xf5f880e0584710d4ull, 0xc22cf563f5e5e1f6ull, 0xfe38130c60e7684bull,
    0x57a803de12fad4f0ull, 0x0879e4d502a7cd2aull, 0x7ac19edcc3ee76adull,
    0x67939a3067e5355aull, 0x948cf9e5a1b255bdull, 0xb2b83de1ac5e314eull,
    0xede92127d53e7b22ull, 0xa6afaec0f6788bdcull, 0x56ce579d0bd1e174ull,
};

static uint64_t fnv1a(const Img& im) {
  uint64_t h = 0xcbf29ce484222325ull;
  const unsigned char* b = (const unsigned char*)im.buf.contents;
  for (size_t i = 0; i < im.bytes(); ++i) {
    h ^= b[i];
    h *= 0x100000001b3ull;
  }
  return h;
}

// Output hash of each case (API v5 fields only).
static void defaultPictureHashes(uint64_t out[kPinCount]) {
  for (int i = 0; i < kPinCount; ++i) {
    const PinCase& c = kPinCases[i];
    Noise n;
    n.rgb = c.noise;
    std::vector<Img> seq = sequence(mixedScene, 5, n, 131, c.fmt);
    const ODNoiseProfile prof = whiteProfile(c.noise);
    ODParams p = base();
    if (c.manual) p.manualProfile = &prof;
    p.quality = c.quality;
    p.temporalRadius = c.radius;
    p.spatialStrength = c.spatial;
    p.sharpen = c.sharpen;
    p.lumaStrength = c.luma;
    p.view = c.view;
    out[i] = fnv1a(rendered(seq, 2, p));
  }
}

static bool gPrintDefaultPicture = false;

// API v10: the grain mode is chosen per render. A session whose renders
// switch mode (an editor flipping Minimum grain applies to on a playing clip)
// gives each render the picture a session used in that mode alone gives it:
// both automatic profiles are fed the same estimates and every decision reads
// the render's own mode. Blocking renders at full quality, automatic profile,
// keyed frames. The test override is lifted while its sessions are made.
static void testGrainModeSwitch() {
  gTest = "grain_mode_switch";
  const char* forced = std::getenv("OD_GRAIN_PROFILE_V2");
  const std::string saved = forced ? forced : "";
  unsetenv("OD_GRAIN_PROFILE_V2");
  Noise n;
  n.rgb = 0.004f;
  const int kFrames = 8;
  const std::vector<Img> seq = sequence(mixedScene, kFrames, n, 91);
  auto play = [&](const std::function<uint32_t(int)>& mode, std::vector<Img>* outs) {
    ODSession* s = od_session_create(gCtx);
    for (int f = 0; f < kFrames; ++f) {
      ODParams p = base();
      p.grainMode = mode(f);
      Img out = makeImg();
      CHECK(renderAt(seq, f, p, out, s, nullptr, 700), "grain mode switch: render %d failed", f);
      outs->push_back(out);
    }
    od_session_destroy(s);
  };
  std::vector<Img> whole, fine, alt, late;
  play([](int) { return (uint32_t)OD_GRAIN_WHOLE; }, &whole);
  play([](int) { return (uint32_t)OD_GRAIN_FINEST; }, &fine);
  play([](int f) { return (uint32_t)(f % 2 ? OD_GRAIN_FINEST : OD_GRAIN_WHOLE); }, &alt);
  play([](int f) { return (uint32_t)(f < 5 ? OD_GRAIN_WHOLE : OD_GRAIN_FINEST); }, &late);
  int same = 0, differ = 0;
  for (int f = 0; f < kFrames; ++f) {
    const Img& wantAlt = f % 2 ? fine[(size_t)f] : whole[(size_t)f];
    const Img& wantLate = f < 5 ? whole[(size_t)f] : fine[(size_t)f];
    CHECK(sameBytes(alt[(size_t)f], wantAlt), "alternating modes, frame %d differs from its mode alone", f);
    CHECK(sameBytes(late[(size_t)f], wantLate), "a switch at frame 5, frame %d differs from its mode alone", f);
    same += sameBytes(alt[(size_t)f], wantAlt) + sameBytes(late[(size_t)f], wantLate);
    differ += !sameBytes(whole[(size_t)f], fine[(size_t)f]);
  }
  CHECK(differ == kFrames, "the two modes gave the same picture on %d of %d frames: the test says nothing",
        kFrames - differ, kFrames);
  printf("  %d of %d renders of switching sessions identical to their mode alone; the modes differ on %d of %d "
         "frames\n", same, 2 * kFrames, differ, kFrames);
  if (forced) setenv("OD_GRAIN_PROFILE_V2", saved.c_str(), 1);
}

static void testDefaultPicture() {
  gTest = "default_picture";
  uint64_t h[kPinCount];
  defaultPictureHashes(h);
  if (gPrintDefaultPicture) {
    printf("  device: %s\n", gDevice.name.UTF8String);
    for (int i = 0; i < kPinCount; ++i) printf("  0x%016llxull,  // %s\n", (unsigned long long)h[i], kPinCases[i].name);
    return;
  }
  if (strcmp(gDevice.name.UTF8String, kPinDevice) != 0) {
    printf("  skipped: the hashes were recorded on %s, this is %s\n", kPinDevice, gDevice.name.UTF8String);
    return;
  }
  const uint64_t* reference = grainProfileV2() ? kPinHashV2 : kPinHash;
  int same = 0;
  for (int i = 0; i < kPinCount; ++i) {
    same += h[i] == reference[i];
    CHECK(h[i] == reference[i], "%s: the default picture changed (hash %016llx, recorded %016llx); if on purpose, "
          "re-record with --print-default-picture", kPinCases[i].name, (unsigned long long)h[i],
          (unsigned long long)reference[i]);
  }
  printf("  %d of %d default renders identical to the recorded picture\n", same, kPinCount);
}

int main(int argc, char** argv) {
  @autoreleasepool {
    // Both grain modes are exercised by `make test` (test.mk sets the
    // session override, OD_GRAIN_PROFILE_V2=0 and =1, so zeroed and default
    // ODParams render alike); run directly, the default mode (finest grain).
    setenv("OD_GRAIN_PROFILE_V2", "1", 0);
    for (int i = 1; i < argc; ++i) {
      if (strcmp(argv[i], "--print-default-picture") == 0) {
        gPrintDefaultPicture = true;
      } else {
        fprintf(stderr, "usage: %s [--print-default-picture]\n", argv[0]);
        return 2;
      }
    }
    gDevice = MTLCreateSystemDefaultDevice();
    if (!gDevice) {
      fprintf(stderr, "no Metal device\n");
      return 1;
    }
    char err[1024] = {0};
    gCtx = od_context_create((__bridge void*)gDevice, nullptr, err, sizeof(err));
    if (!gCtx) {
      fprintf(stderr, "od_context_create: %s\n", err);
      return 1;
    }
    struct {
      const char* name;
      void (*fn)();
    } tests[] = {
        {"defaults", testDefaults},
        {"neutral", testNeutral},
        {"sharpen_bands_zero", testSharpenBandsZero},
        {"levels", testLevels},
        {"amounts", testAmounts},
        {"grain_floor", testGrainFloor},
        {"specks", testSpecks},
        {"temporal_controls", testTemporalControls},
        {"sharpen_controls", testSharpenControls},
        {"hold", testHold},
        {"ranges", testRanges},
        {"bands_step", testBandsStep},
        {"levels_in_stages", testLevelsInStages},
        {"grain_floor_paths", testGrainFloorPaths},
        {"bands_and_sharpen", testBandsAndSharpen},
        {"grain_stability", testGrainStability},
        {"grain_mode_switch", testGrainModeSwitch},
        {"default_picture", testDefaultPicture},
    };
    for (auto& t : tests) {
      if (gPrintDefaultPicture && t.fn != testDefaultPicture) continue;
      const int before = gFailures;
      @autoreleasepool {
        t.fn();
      }
      printf("%-20s %s\n", t.name, gFailures == before ? "ok" : "FAILED");
    }
    od_context_destroy(gCtx);
    printf("advanced tests: %d checks, %d failures\n", gChecks, gFailures);
    return gFailures ? 1 : 0;
  }
}
