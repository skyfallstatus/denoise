// SPDX-License-Identifier: MIT
// Correctness tests for the spatial and sharpen stages, through the public
// API with a hand-filled noise profile and the temporal stage off (radius 0).
// Synthetic frames only: no media needed. Memory: a handful of frames of at
// most 641x359 (a few MB).
//
// What is asserted (each at quality 0 and 1 where it matters):
//   - strength 0 is a bit-exact pass-through; luma-only leaves chroma alone
//     and chroma-only leaves luma alone
//   - a constant frame comes out bit-identical at every size, odd ones and
//     tiny ones included; no NaN/Inf anywhere
//   - flat noisy frames: noise drops by the expected factor, the mean is
//     kept, and the frame border is neither darker, brighter nor noisier than
//     the inside
//   - a strong edge is not blurred and gets no halo; a chroma-only edge too
//   - values below 0 and above 1 survive; pixels at +-60000 next to ordinary
//     ones (differences that overflow half precision once normalised by a
//     small sigma) give no NaN and no Inf and do not disturb the rest
//   - the count map (OD_SPATIAL_TEST_COUNT) scales the thresholds like a
//     profile divided by sqrt(count)
//   - sharpen: exact on a constant frame, chroma untouched, noise at the
//     profiled level is not amplified, a soft edge gets steeper, overshoot is
//     bounded
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>

#include "opendenoise/opendenoise.h"

static int gFailures = 0;
static int gChecks = 0;
static const char* gTest = "";

#define CHECK(cond, ...)                                               \
  do {                                                                 \
    ++gChecks;                                                         \
    if (!(cond)) {                                                     \
      ++gFailures;                                                     \
      fprintf(stderr, "FAIL [%s] %s:%d: ", gTest, __FILE__, __LINE__); \
      fprintf(stderr, __VA_ARGS__);                                    \
      fprintf(stderr, "\n");                                           \
    }                                                                  \
  } while (0)

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;
static ODContext* gCtx;

static uint32_t hash32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

// Unit Gaussian, deterministic in (x, y, c, seed).
static float gauss(uint32_t x, uint32_t y, uint32_t c, uint32_t seed) {
  uint32_t h1 = hash32(x * 73856093u ^ y * 19349663u ^ c * 83492791u ^ seed * 2654435761u);
  uint32_t h2 = hash32(h1 ^ 0x9E3779B9u);
  float u1 = ((float)(h1 >> 8) + 1.0f) / 16777217.0f;
  float u2 = (float)(h2 >> 8) / 16777216.0f;
  return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

// A frame as planar Y, Cb, Cr (the working space), w x h.
struct Img {
  int w = 0, h = 0;
  std::vector<float> p[3];
  Img() {}
  Img(int w_, int h_) : w(w_), h(h_) {
    for (auto& v : p) v.assign((size_t)w * h, 0.0f);
  }
  float& at(int c, int x, int y) { return p[c][(size_t)y * w + x]; }
  float at(int c, int x, int y) const { return p[c][(size_t)y * w + x]; }
};

// Working space <-> RGB, the matrix of core/shaders/common.metal.
static void yccToBgra(const Img& im, std::vector<float>& bgra) {
  bgra.resize((size_t)im.w * im.h * 4);
  for (size_t i = 0; i < (size_t)im.w * im.h; ++i) {
    float y = im.p[0][i], cb = im.p[1][i], cr = im.p[2][i];
    float r = y + 1.5748f * cr, b = y + 1.8556f * cb;
    float g = (y - 0.2126f * r - 0.0722f * b) / 0.7152f;
    bgra[i * 4] = b;
    bgra[i * 4 + 1] = g;
    bgra[i * 4 + 2] = r;
    bgra[i * 4 + 3] = 0.75f;
  }
}
static Img bgraToYcc(const std::vector<float>& bgra, int w, int h) {
  Img im(w, h);
  for (size_t i = 0; i < (size_t)w * h; ++i) {
    float b = bgra[i * 4], g = bgra[i * 4 + 1], r = bgra[i * 4 + 2];
    float y = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    im.p[0][i] = y;
    im.p[1][i] = (b - y) / 1.8556f;
    im.p[2][i] = (r - y) / 1.5748f;
  }
  return im;
}

// White noise of sigma s[c] per working channel: the matching profile.
static ODNoiseProfile whiteProfile(const float s[3]) {
  ODNoiseProfile p;
  memset(&p, 0, sizeof(p));
  const float gain[OD_BANDS] = {0.8908f, 0.2007f, 0.0855f, 0.0412f};
  for (int c = 0; c < 3; ++c)
    for (int l = 0; l < OD_LEVELS; ++l) {
      for (int b = 0; b < OD_BANDS; ++b) p.sigma[c][b][l] = s[c] * gain[b];
      p.sigmaFull[c][l] = s[c];
    }
  p.confidence = 1;
  p.valid = 1;
  return p;
}

static void addNoise(Img& im, const float s[3], uint32_t seed) {
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < im.h; ++y)
      for (int x = 0; x < im.w; ++x) im.at(c, x, y) += s[c] * gauss((uint32_t)x, (uint32_t)y, (uint32_t)c, seed);
}

static ODParams baseParams(const ODNoiseProfile* prof) {
  ODParams p = od_params_default();
  p.temporalRadius = 0;
  p.lumaStrength = 1;
  p.chromaStrength = 1;
  p.spatialStrength = 0.5f;
  p.sharpen = 0;
  p.waitUntilCompleted = 1;
  p.manualProfile = prof;
  return p;
}

// Renders `in` (BGRA float) and returns the output BGRA. Fails the test (and
// returns the input) on error.
static std::vector<float> render(ODSession* s, const std::vector<float>& in, int w, int h, const ODParams& p,
                                 bool half = false) {
  const uint32_t bpp = half ? 8 : 16;
  const size_t bytes = (size_t)w * h * bpp;
  id<MTLBuffer> a = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> b = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  if (half) {
    __fp16* d = (__fp16*)a.contents;
    for (size_t i = 0; i < in.size(); ++i) d[i] = (__fp16)in[i];
  } else {
    memcpy(a.contents, in.data(), bytes);
  }
  memset(b.contents, 0, bytes);
  ODFrame f;
  f.mtlBuffer = (__bridge void*)a;
  f.width = (uint32_t)w;
  f.height = (uint32_t)h;
  f.rowBytes = (uint32_t)w * bpp;
  f.format = half ? OD_PIXEL_BGRA16F : OD_PIXEL_BGRA32F;
  f.frameKey = OD_NO_KEY;
  ODFrame o = f;
  o.mtlBuffer = (__bridge void*)b;
  int off = 0;
  int rc = od_render(s, &f, &off, 1, &p, &o, nullptr);
  CHECK(rc == 0, "od_render failed (%dx%d): %s", w, h, od_last_error(s));
  std::vector<float> out(in.size());
  if (rc != 0) return in;
  if (half) {
    const __fp16* sp = (const __fp16*)b.contents;
    for (size_t i = 0; i < out.size(); ++i) out[i] = (float)sp[i];
  } else {
    memcpy(out.data(), b.contents, bytes);
  }
  return out;
}

static Img run(ODSession* s, const Img& in, const ODParams& p, std::vector<float>* rawIn = nullptr,
               std::vector<float>* rawOut = nullptr, bool half = false) {
  std::vector<float> bgra;
  yccToBgra(in, bgra);
  std::vector<float> out = render(s, bgra, in.w, in.h, p, half);
  if (rawIn) *rawIn = bgra;
  if (rawOut) *rawOut = out;
  return bgraToYcc(out, in.w, in.h);
}

struct Stat {
  double mean = 0, rms = 0, maxAbs = 0;
  size_t n = 0;
};
// Statistics of (a - b) for channel c over the pixels where pick(x, y).
static Stat diffStat(const Img& a, const Img& b, int c, const std::function<bool(int, int)>& pick) {
  Stat s;
  double sum = 0, sum2 = 0;
  for (int y = 0; y < a.h; ++y)
    for (int x = 0; x < a.w; ++x) {
      if (!pick(x, y)) continue;
      double d = (double)a.at(c, x, y) - b.at(c, x, y);
      sum += d;
      sum2 += d * d;
      s.maxAbs = std::max(s.maxAbs, std::fabs(d));
      ++s.n;
    }
  if (s.n) {
    s.mean = sum / (double)s.n;
    s.rms = std::sqrt(sum2 / (double)s.n);
  }
  return s;
}
static bool allFinite(const std::vector<float>& v) {
  for (float f : v)
    if (!std::isfinite(f)) return false;
  return true;
}
static const std::function<bool(int, int)> kAll = [](int, int) { return true; };

static const float kSig[3] = {0.02f, 0.012f, 0.012f};
static ODSession* s_hotSession = nullptr;  // the session of main(), for tests written without a parameter

// ---------------------------------------------------------------------------
static void testPassThrough(ODSession* s) {
  gTest = "pass-through";
  ODNoiseProfile prof = whiteProfile(kSig);
  Img clean(96, 64);
  for (int y = 0; y < clean.h; ++y)
    for (int x = 0; x < clean.w; ++x) {
      clean.at(0, x, y) = 0.2f + 0.5f * (float)x / 96.0f;
      clean.at(1, x, y) = 0.05f * sinf(0.2f * (float)y);
      clean.at(2, x, y) = -0.04f;
    }
  Img noisy = clean;
  addNoise(noisy, kSig, 11);
  std::vector<float> in, out;

  ODParams p = baseParams(&prof);
  p.spatialStrength = 0;
  run(s, noisy, p, &in, &out);
  CHECK(memcmp(in.data(), out.data(), in.size() * 4) == 0, "spatialStrength 0 is not bit-exact");

  p = baseParams(&prof);
  p.lumaStrength = 0;
  p.chromaStrength = 0;
  p.sharpen = 0;
  run(s, noisy, p, &in, &out);
  CHECK(memcmp(in.data(), out.data(), in.size() * 4) == 0, "luma 0 + chroma 0 is not bit-exact");

  for (int q = 0; q <= 1; ++q) {
    // Luma only: chroma must stay (to float rounding of the RGB round trip).
    p = baseParams(&prof);
    p.quality = q;
    p.chromaStrength = 0;
    Img o = run(s, noisy, p);
    Stat dy = diffStat(o, noisy, 0, kAll), dcb = diffStat(o, noisy, 1, kAll), dcr = diffStat(o, noisy, 2, kAll);
    CHECK(dy.rms > 0.5 * kSig[0], "q%d luma-only: luma barely changed (rms %.5f)", q, dy.rms);
    CHECK(dcb.maxAbs < 2e-6 && dcr.maxAbs < 2e-6, "q%d luma-only: chroma changed (max %.2e / %.2e)", q, dcb.maxAbs,
          dcr.maxAbs);
    // Chroma only.
    p = baseParams(&prof);
    p.quality = q;
    p.lumaStrength = 0;
    o = run(s, noisy, p);
    dy = diffStat(o, noisy, 0, kAll);
    dcb = diffStat(o, noisy, 1, kAll);
    CHECK(dcb.rms > 0.5 * kSig[1], "q%d chroma-only: chroma barely changed (rms %.5f)", q, dcb.rms);
    CHECK(dy.maxAbs < 2e-6, "q%d chroma-only: luma changed (max %.2e)", q, dy.maxAbs);
    // Continuity towards strength 0.
    p = baseParams(&prof);
    p.quality = q;
    p.spatialStrength = 0.002f;
    o = run(s, noisy, p);
    dy = diffStat(o, noisy, 0, kAll);
    CHECK(dy.maxAbs < 0.05 * kSig[0], "q%d spatialStrength 0.002 moved luma by %.5f", q, dy.maxAbs);
    // Alpha comes from the source.
    std::vector<float> rin, rout;
    p = baseParams(&prof);
    p.quality = q;
    run(s, noisy, p, &rin, &rout);
    bool alphaSame = true;
    for (size_t i = 3; i < rin.size(); i += 4) alphaSame = alphaSame && rin[i] == rout[i];
    CHECK(alphaSame, "q%d alpha changed", q);
  }
}

static void testSizesAndBorders(ODSession* s) {
  gTest = "sizes-and-borders";
  ODNoiseProfile prof = whiteProfile(kSig);
  const int sizes[][2] = {{1, 1}, {2, 3}, {7, 5}, {17, 9}, {37, 23}, {101, 67}, {255, 129}, {641, 359}};
  for (auto& sz : sizes) {
    const int w = sz[0], h = sz[1];
    for (int q = 0; q <= 1; ++q) {
      ODParams p = baseParams(&prof);
      p.quality = q;
      p.sharpen = 0.5f;  // both stages
      // Constant frame: bit-identical out.
      Img flat(w, h);
      for (size_t i = 0; i < (size_t)w * h; ++i) {
        flat.p[0][i] = 0.37f;
        flat.p[1][i] = 0.02f;
        flat.p[2][i] = -0.03f;
      }
      std::vector<float> in, out;
      run(s, flat, p, &in, &out);
      CHECK(memcmp(in.data(), out.data(), in.size() * 4) == 0, "%dx%d q%d: constant frame changed", w, h, q);

      // Noisy flat frame: finite, mean kept, border like the inside.
      Img noisy = flat;
      addNoise(noisy, kSig, 100u + (uint32_t)w);
      p.sharpen = 0;
      Img o = run(s, noisy, p, &in, &out);
      CHECK(allFinite(out), "%dx%d q%d: NaN or Inf in the output", w, h, q);
      for (int c = 0; c < 3; ++c) {
        Stat all = diffStat(o, flat, c, kAll);
        Stat nin = diffStat(noisy, flat, c, kAll);
        CHECK(all.rms <= nin.rms * 1.001 + 1e-6, "%dx%d q%d ch%d: error grew (%.5f -> %.5f)", w, h, q, c, nin.rms, all.rms);
        // The output mean must sit where the input mean sits.
        // (Frames of a few pixels are mostly replicated border: skip those.)
        if (all.n >= 100)
          CHECK(std::fabs(all.mean - nin.mean) < 0.25 * kSig[c] / std::sqrt((double)all.n) + 2e-4,
                "%dx%d q%d ch%d: mean moved by %.5f", w, h, q, c, all.mean - nin.mean);
        if (w >= 37 && h >= 23) {
          auto border = [&](int x, int y) { return x < 2 || y < 2 || x >= w - 2 || y >= h - 2; };
          auto inside = [&](int x, int y) { return x >= 8 && y >= 8 && x < w - 8 && y < h - 8; };
          Stat bo = diffStat(o, flat, c, border), in_ = diffStat(o, flat, c, inside);
          Stat bi = diffStat(noisy, flat, c, border);
          // No dark or bright frame edge: the border's mean error stays within
          // what its own input noise explains.
          CHECK(std::fabs(bo.mean - bi.mean) < 0.2 * kSig[c], "%dx%d q%d ch%d: border mean shifted by %.5f (sigma %.3f)", w,
                h, q, c, bo.mean - bi.mean, kSig[c]);
          CHECK(std::fabs(bo.mean) < 5.0 * kSig[c] / std::sqrt((double)bo.n) + 0.05 * kSig[c],
                "%dx%d q%d ch%d: border mean error %.5f", w, h, q, c, bo.mean);
          // And it is denoised too (fewer real neighbours, so allow less).
          CHECK(bo.rms < 0.75 * kSig[c], "%dx%d q%d ch%d: border left noisy (rms %.5f, inside %.5f)", w, h, q, c, bo.rms,
                in_.rms);
        }
      }
    }
  }
  // 16-bit float host frames.
  {
    ODParams p = baseParams(&prof);
    Img flat(101, 67);
    for (size_t i = 0; i < flat.p[0].size(); ++i) flat.p[0][i] = 0.4f;
    Img noisy = flat;
    addNoise(noisy, kSig, 5);
    Img o = run(s, noisy, p, nullptr, nullptr, true);
    Stat r = diffStat(o, flat, 0, kAll);
    CHECK(r.rms < 0.45 * kSig[0], "16f host: luma residual %.5f", r.rms);
  }
}

static void testNoiseReduction(ODSession* s) {
  gTest = "noise-reduction";
  ODNoiseProfile prof = whiteProfile(kSig);
  const int w = 320, h = 200;
  Img flat(w, h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      flat.at(0, x, y) = 0.15f + 0.6f * (float)x / (float)w;  // smooth ramp: all luminance knots
      flat.at(1, x, y) = 0.03f;
      flat.at(2, x, y) = -0.02f;
    }
  Img noisy = flat;
  addNoise(noisy, kSig, 21);
  auto inside = [&](int x, int y) { return x >= 8 && y >= 8 && x < w - 8 && y < h - 8; };
  // Residual noise / input noise the stage must reach on white noise, per
  // quality. Measured: 0.28 / 0.21 (quality 0, luma / chroma) and
  // 0.16 / 0.10 (quality 1).
  const double limY[2] = {0.34, 0.20}, limC[2] = {0.26, 0.14};
  for (int q = 0; q <= 1; ++q) {
    ODParams p = baseParams(&prof);
    p.quality = q;
    Img o = run(s, noisy, p);
    for (int c = 0; c < 3; ++c) {
      Stat r = diffStat(o, flat, c, inside);
      double ratio = r.rms / kSig[c];
      printf("  noise left, quality %d, channel %d: %.3f of the input (mean error %+.5f)\n", q, c, ratio, r.mean);
      CHECK(ratio < (c ? limC[q] : limY[q]), "q%d ch%d: %.3f of the noise left", q, c, ratio);
      CHECK(std::fabs(r.mean) < 0.03 * kSig[c], "q%d ch%d: mean error %.5f", q, c, r.mean);
    }
    // The amount: luma 0.5 leaves about half of the luma noise in.
    p.lumaStrength = 0.5f;
    Img o2 = run(s, noisy, p);
    double ratio = diffStat(o2, flat, 0, inside).rms / kSig[0];
    CHECK(ratio > 0.48 && ratio < 0.62, "q%d: luma amount 0.5 left %.3f of the noise", q, ratio);
    // Stronger setting removes more, weaker less.
    p = baseParams(&prof);
    p.quality = q;
    p.spatialStrength = 0.15f;
    double weak = diffStat(run(s, noisy, p), flat, 0, inside).rms;
    p.spatialStrength = 0.5f;
    double mid = diffStat(run(s, noisy, p), flat, 0, inside).rms;
    CHECK(weak > 1.5 * mid, "q%d: spatialStrength 0.15 vs 0.5 residual %.5f vs %.5f", q, weak, mid);
    // noiseLevel scales the thresholds the same way.
    p.noiseLevel = 0.3f;
    double low = diffStat(run(s, noisy, p), flat, 0, inside).rms;
    CHECK(std::fabs(low - weak) < 0.02 * kSig[0], "q%d: noiseLevel 0.3 (%.5f) and spatialStrength 0.15 (%.5f) differ", q, low,
          weak);
  }
  // Cb and Cr noise differ (0.008 and 0.020): each chroma channel must be
  // filtered by its own sigma. With the two exchanged, Cr would be filtered
  // at 0.4 of its noise and keep most of it.
  {
    const float sig2[3] = {0.02f, 0.008f, 0.020f};
    ODNoiseProfile p2 = whiteProfile(sig2);
    Img n2 = flat;
    addNoise(n2, sig2, 23);
    for (int q = 0; q <= 1; ++q) {
      ODParams p = baseParams(&p2);
      p.quality = q;
      Img o = run(s, n2, p);
      for (int c = 1; c < 3; ++c) {
        const double ratio = diffStat(o, flat, c, inside).rms / sig2[c];
        printf("  unequal chroma noise, quality %d, channel %d (sigma %.3f): %.3f of the input left\n", q, c, sig2[c],
               ratio);
        CHECK(ratio < limC[q] * 1.1, "q%d ch%d with sigma %.3f: %.3f of the noise left", q, c, sig2[c], ratio);
      }
    }
  }
  // Grain floor: very fine noise survives even when broader luma is cleaned.
  {
    const float tiny[3] = {5e-4f, 5e-4f, 5e-4f};
    ODNoiseProfile pt = whiteProfile(tiny);
    Img n2 = flat;
    addNoise(n2, tiny, 3);
    ODParams p = baseParams(&pt);
    Img o = run(s, n2, p);
    const char* mode = std::getenv("OD_GRAIN_PROFILE_V2");
    if (!(mode && std::strcmp(mode, "1") == 0)) {
      const Stat d = diffStat(o, n2, 0, kAll);
      CHECK(d.maxAbs < 1e-6, "noise below the legacy grain floor: luma changed by %.2e", d.maxAbs);
    }
    double inputFine = 0, outputFine = 0;
    for (int y = 8; y < n2.h - 8; ++y) for (int x = 8; x < n2.w - 8; ++x) {
      const double a = n2.at(0, x, y) - n2.at(0, x - 1, y);
      const double b = o.at(0, x, y) - o.at(0, x - 1, y);
      inputFine += a * a; outputFine += b * b;
    }
    CHECK(outputFine > 0.85 * inputFine, "below-floor fine grain was erased (energy ratio %.3f)", outputFine/inputFine);
  }
}

static void testEdges(ODSession* s) {
  gTest = "edges";
  ODNoiseProfile prof = whiteProfile(kSig);
  const int w = 200, h = 120, ex = 100;
  for (int kind = 0; kind < 2; ++kind) {  // 0: luma edge, 1: chroma-only edge
    const int c = kind == 0 ? 0 : 1;
    const float lo = kind == 0 ? 0.25f : -0.08f, hi = kind == 0 ? 0.55f : 0.10f;
    Img clean(w, h);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        clean.at(0, x, y) = 0.4f;
        clean.at(1, x, y) = 0.0f;
        clean.at(2, x, y) = 0.01f;
        clean.at(c, x, y) = x < ex ? lo : hi;
      }
    Img noisy = clean;
    addNoise(noisy, kSig, 31u + (uint32_t)kind);
    for (int q = 0; q <= 1; ++q) {
      ODParams p = baseParams(&prof);
      p.quality = q;
      Img o = run(s, noisy, p);
      const double step = hi - lo;
      // Column means next to the edge: no bleeding across it, no halo.
      for (int dx = -6; dx <= 5; ++dx) {
        int x = ex + dx;
        double m = 0;
        for (int y = 8; y < h - 8; ++y) m += o.at(c, x, y);
        m /= (double)(h - 16);
        double want = x < ex ? lo : hi;
        CHECK(std::fabs(m - want) < 0.02 * step, "%s edge q%d: column %+d mean off by %.4f (step %.2f)",
              kind ? "chroma" : "luma", q, dx, m - want, step);
      }
      // Both sides are still denoised right up to the edge.
      auto nearEdge = [&](int x, int y) { return y >= 8 && y < h - 8 && std::abs(x - ex) <= 6 && x != ex - 1 && x != ex; };
      Stat r = diffStat(o, clean, c, nearEdge);
      CHECK(r.rms < 0.6 * kSig[c], "%s edge q%d: noise left next to the edge %.5f (sigma %.3f)", kind ? "chroma" : "luma", q,
            r.rms, kSig[c]);
    }
  }
}

// Values at the end of the half range next to ordinary ones. The NL-means
// kernel works on differences divided by sigma in half precision: with a
// small sigma those overflow, and the kernel has to bring them back before
// they meet (Inf - Inf). And the sharpen stage must not push a value near
// the end of the range over it.
static void testExtremes(ODSession* s) {
  gTest = "extremes";
  const float tiny[3] = {2e-4f, 2e-4f, 2e-4f};
  const float* sigmas[2] = {kSig, tiny};
  const int w = 96, h = 64;
  Img clean(w, h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float v = 0.4f;
      if (x >= 20 && x < 24 && y >= 20 && y < 40) v = 60000.0f;   // a block and
      if (x == 60 && y == 30) v = 60000.0f;                       // single pixels,
      if (x == 61 && y == 30) v = -60000.0f;                      // both signs, touching
      if (x >= 70 && y >= 50) v = -60000.0f;                      // a corner of the frame
      clean.at(0, x, y) = v;
      clean.at(1, x, y) = (x == 40 && y == 10) ? 30000.0f : 0.01f;  // chroma too
      clean.at(2, x, y) = (x == 41 && y == 10) ? -30000.0f : -0.02f;
    }
  for (int k = 0; k < 2; ++k) {
    ODNoiseProfile prof = whiteProfile(sigmas[k]);
    Img noisy = clean;
    addNoise(noisy, sigmas[k], 77);
    for (int q = 0; q <= 1; ++q) {
      ODParams p = baseParams(&prof);
      p.quality = q;
      std::vector<float> in, out;
      // With the sharpen stage: finite. (It may move the extreme pixels by
      // its overshoot allowance, a tenth of a step of 120000.)
      p.sharpen = 0.5f;
      run(s, noisy, p, &in, &out);
      CHECK(allFinite(out), "sigma %.0e q%d, sharpen: NaN or Inf", sigmas[k][0], q);
      // Spatial alone: finite, the extreme pixels keep their value (to the
      // rounding of 60000 in half precision and of the RGB round trip), and
      // the ordinary ones 8 pixels or more away from any of them are
      // denoised as usual.
      p.sharpen = 0;
      Img o = run(s, noisy, p, &in, &out);
      CHECK(allFinite(out), "sigma %.0e q%d: NaN or Inf", sigmas[k][0], q);
      auto far = [&](int x, int y) {
        for (int dy = -8; dy <= 8; ++dy)
          for (int dx = -8; dx <= 8; ++dx) {
            const int xx = x + dx, yy = y + dy;
            if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
            if (std::fabs(clean.at(0, xx, yy)) > 1 || std::fabs(clean.at(1, xx, yy)) > 1 || std::fabs(clean.at(2, xx, yy)) > 1)
              return false;
          }
        return true;
      };
      auto extreme = [&](int x, int y) { return std::fabs(clean.at(0, x, y)) > 1; };
      Stat e = diffStat(o, noisy, 0, extreme), f = diffStat(o, clean, 0, far), fin = diffStat(noisy, clean, 0, far);
      CHECK(e.n > 0 && e.maxAbs < 64.0, "sigma %.0e q%d: an extreme pixel moved by %.1f", sigmas[k][0], q, e.maxAbs);
      // (1e-4: the working space is half precision, whose step at 0.4 is
      // 2.4e-4, more than the smaller of the two sigmas.)
      CHECK(f.n > 1000 && f.rms <= fin.rms * 1.001 + 1e-4, "sigma %.0e q%d: error grew away from the extremes (%.6f -> %.6f)",
            sigmas[k][0], q, fin.rms, f.rms);
    }
  }
}

// A hot pixel (Inf, the end of the half range, or just below it) or a super-white block on a
// flat frame, with sharpening on. The overshoot limit allows a share of the local range:
// without an absolute bound that range is tens of thousands next to such a pixel, which
// drove its neighbours far below zero and pushed the pixel itself to Inf in the working
// texture (stage.h section 2), coming out as (inf, nan, inf). Here: the hot pixel comes out
// bit-identical, every value that was finite stays finite, and no neighbour falls more than
// 0.1 below its 3x3 minimum.
static void testHotPixels() {
  gTest = "hot-pixels";
  const int w = 64, h = 64, hx = 31, hy = 29;
  const float sig[3] = {0.02f, 0.02f, 0.02f};
  ODNoiseProfile prof = whiteProfile(sig);
  const float inf = std::numeric_limits<float>::infinity();
  struct Case {
    float value;
    int size;  // 1 = one pixel, n = an n x n block
  } cases[] = {{inf, 1}, {65504.0f, 1}, {60000.0f, 1}, {20.0f, 2}, {20.0f, 8}, {4.0f, 1}};
  for (const Case& cs : cases)
    for (int half = 0; half <= 1; ++half)
      for (int q = 0; q <= 1; ++q)
        for (float sharpen : {0.25f, 1.0f}) {
          std::vector<float> in((size_t)w * h * 4);
          auto hot = [&](int x, int y) { return x >= hx && x < hx + cs.size && y >= hy && y < hy + cs.size; };
          for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
              for (int c = 0; c < 4; ++c) in[((size_t)y * w + x) * 4 + c] = c == 3 ? 1.0f : hot(x, y) ? cs.value : 0.2f;
          ODParams p = baseParams(&prof);
          p.quality = q;
          p.sharpen = sharpen;
          std::vector<float> out = render(s_hotSession, in, w, h, p, half != 0);
          // What the host actually handed over (a 16f host rounds 60000 to 59968).
          std::vector<float> src = in;
          if (half)
            for (float& v : src) v = (float)(__fp16)v;
          bool finite = true, same = true;
          double lowest = 1e9;
          for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
              const size_t i = ((size_t)y * w + x) * 4;
              for (int c = 0; c < 4; ++c)
                if (std::isfinite(src[i + c]) && !std::isfinite(out[i + c])) finite = false;
              if (hot(x, y)) {
                if (cs.size == 1 && memcmp(&out[i], &src[i], 16) != 0) same = false;
                continue;
              }
              const double yv = 0.2126 * out[i + 2] + 0.7152 * out[i + 1] + 0.0722 * out[i];
              lowest = std::min(lowest, yv);
            }
          CHECK(finite, "%g (%dx%d) %s q%d sharpen %.2f: a finite value came out non-finite", cs.value, cs.size, cs.size,
                half ? "16f" : "32f", q, sharpen);
          // 0.1: the overshoot allowance; 0.012: what the spatial stage alone
          // moves a neighbour of such a pixel by (it feeds the limiter).
          CHECK(lowest >= 0.2 - 0.1 - 0.012, "%g (%dx%d) %s q%d sharpen %.2f: a neighbour went down to %.4f (flat 0.2)",
                cs.value, cs.size, cs.size, half ? "16f" : "32f", q, sharpen, lowest);
          if (cs.size == 1 && cs.value > 1000.0f)
            CHECK(same, "%g %s q%d sharpen %.2f: the hot pixel came out as %g %g %g", cs.value, half ? "16f" : "32f", q,
                  sharpen, out[((size_t)hy * w + hx) * 4], out[((size_t)hy * w + hx) * 4 + 1],
                  out[((size_t)hy * w + hx) * 4 + 2]);
        }
}

// The output stage on a 16f host: alpha is always the source alpha, bit for bit, NaN and Inf
// included, on every pixel the stages changed and in the Noise view; and a source pixel
// whose colour is not finite comes out as it went in, in both formats (it used to leave a
// 16f frame as -65504 or as (65504, -65504, 65504), and a 32f one as (inf, nan, inf)).
static void testOutputNonFinite() {
  gTest = "output-non-finite";
  const int w = 96, h = 64;
  ODNoiseProfile prof = whiteProfile(kSig);
  Img noisy(w, h);
  for (size_t i = 0; i < noisy.p[0].size(); ++i) noisy.p[0][i] = 0.4f;
  addNoise(noisy, kSig, 91);
  std::vector<float> in;
  yccToBgra(noisy, in);
  const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
  const float specials[6] = {nan, inf, -inf, 0.0f, -0.0f, 2.5f};
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) in[((size_t)y * w + x) * 4 + 3] = specials[(x + 3 * y) % 6];
  // Colour: a NaN pixel, a pixel with only green = +Inf, one with blue = -Inf.
  const int bad[3][2] = {{30, 7}, {40, 7}, {50, 20}};
  for (int c = 0; c < 3; ++c) in[((size_t)7 * w + 30) * 4 + c] = nan;
  in[((size_t)7 * w + 40) * 4 + 1] = inf;
  in[((size_t)20 * w + 50) * 4 + 0] = -inf;
  for (int half = 0; half <= 1; ++half)
    for (int view : {OD_VIEW_OUTPUT, OD_VIEW_NOISE})
      for (int q = 0; q <= 1; ++q) {
        ODParams p = baseParams(&prof);
        p.quality = q;
        p.sharpen = 0.5f;
        p.view = view;
        std::vector<float> out = render(s_hotSession, in, w, h, p, half != 0);
        std::vector<float> src = in;
        if (half)
          for (float& v : src) v = (float)(__fp16)v;
        size_t alphaChanged = 0, changed = 0;
        for (size_t i = 0; i < (size_t)w * h; ++i) {
          if (memcmp(&out[i * 4 + 3], &src[i * 4 + 3], 4) != 0) ++alphaChanged;
          if (memcmp(&out[i * 4], &src[i * 4], 12) != 0) ++changed;
        }
        CHECK(alphaChanged == 0, "%s view %d q%d: %zu alpha values changed", half ? "16f" : "32f", view, q, alphaChanged);
        CHECK(changed > (size_t)w * h / 2, "%s view %d q%d: only %zu pixels changed (the test needs the stages to work)",
              half ? "16f" : "32f", view, q, changed);
        if (view != OD_VIEW_OUTPUT) continue;
        for (const auto& b : bad) {
          const size_t i = ((size_t)b[1] * w + b[0]) * 4;
          CHECK(memcmp(&out[i], &src[i], 16) == 0, "%s q%d: non-finite source pixel (%d,%d) came out as %g %g %g",
                half ? "16f" : "32f", q, b[0], b[1], out[i], out[i + 1], out[i + 2]);
        }
      }
}

// Isolated pixels far outside the noise: a pixel 20 sigma DARKER than its neighbours (the
// dark tail of shot noise in log footage, a dead pixel) is a speck and must go, at full luma
// amount; one 20 sigma BRIGHTER (a star, a specular, a distant light) is kept. The spatial
// stage used to keep both beyond 12 sigma, and the dark ones stood out as black dots on an
// otherwise clean surface.
static void testSpecks(ODSession* s) {
  gTest = "specks";
  ODNoiseProfile prof = whiteProfile(kSig);
  const int w = 160, h = 96;
  Img flat(w, h);
  for (size_t i = 0; i < flat.p[0].size(); ++i) flat.p[0][i] = 0.4f;
  Img noisy = flat;
  addNoise(noisy, kSig, 101);
  const int dark[3][2] = {{40, 30}, {80, 50}, {120, 70}}, bright[3][2] = {{60, 20}, {100, 40}, {30, 75}};
  for (const auto& d : dark) noisy.at(0, d[0], d[1]) = 0.4f - 20.0f * kSig[0];
  for (const auto& b : bright) noisy.at(0, b[0], b[1]) = 0.4f + 20.0f * kSig[0];
  for (int q = 0; q <= 1; ++q) {
    ODParams p = baseParams(&prof);
    p.quality = q;
    Img o = run(s, noisy, p);
    double worstDark = 0, worstBright = 1e9;
    for (const auto& d : dark) worstDark = std::max(worstDark, std::fabs((double)o.at(0, d[0], d[1]) - 0.4));
    for (const auto& b : bright) worstBright = std::min(worstBright, (double)o.at(0, b[0], b[1]) - 0.4);
    printf("  specks, quality %d: dark ones end %.4f from the surface (were %.2f), bright ones %.4f above it (were %.2f)\n",
           q, worstDark, 20.0 * kSig[0], worstBright, 20.0 * kSig[0]);
    // What is left: the noise at luma 100 %, the grain floor's share of the
    // original deviation (about 4 % of it here), and the centre's own weight
    // in the mean (1/9 of it with the 3x3 search of quality 0, 1/25 with 5x5).
    CHECK(worstDark < (q ? 3.0 : 6.0) * kSig[0], "q%d: an isolated dark pixel was kept (%.4f off the surface)", q,
          worstDark);
    CHECK(worstBright > 15.0 * kSig[0], "q%d: an isolated bright pixel was flattened (%.4f above the surface)", q,
          worstBright);
  }
}

static void testRange(ODSession* s) {
  gTest = "range";
  ODNoiseProfile prof = whiteProfile(kSig);
  const int w = 160, h = 90;
  Img clean(w, h);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      clean.at(0, x, y) = -0.5f + 4.0f * (float)x / (float)w;  // -0.5 .. 3.5
      clean.at(1, x, y) = 0.3f * ((float)y / (float)h - 0.5f);
    }
  Img noisy = clean;
  addNoise(noisy, kSig, 41);
  for (int q = 0; q <= 1; ++q) {
    ODParams p = baseParams(&prof);
    p.quality = q;
    p.sharpen = 0.5f;
    std::vector<float> in, out;
    Img o = run(s, noisy, p, &in, &out);
    CHECK(allFinite(out), "q%d: NaN or Inf", q);
    Stat r = diffStat(o, clean, 0, kAll);
    CHECK(r.rms < kSig[0] && std::fabs(r.mean) < 0.1 * kSig[0], "q%d: out-of-range ramp, rms %.5f mean %.5f", q, r.rms,
          r.mean);
    float mn = 1e9f, mx = -1e9f;
    for (float v : o.p[0]) {
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    }
    CHECK(mn < -0.4f && mx > 3.4f, "q%d: range clipped to %.3f .. %.3f", q, mn, mx);
  }
}

// Needs its own sessions: the test count map is read when a session is made.
static void testCount() {
  gTest = "count-map";
  const int w = 256, h = 128;
  Img flat(w, h);
  for (size_t i = 0; i < flat.p[0].size(); ++i) flat.p[0][i] = 0.45f;
  const float halfSig[3] = {kSig[0] * 0.5f, kSig[1] * 0.5f, kSig[2] * 0.5f};
  Img noisyHalf = flat;
  addNoise(noisyHalf, halfSig, 51);
  Img noisy = flat;
  addNoise(noisy, kSig, 52);
  ODNoiseProfile prof = whiteProfile(kSig), profHalf = whiteProfile(halfSig);
  auto inside = [&](int x, int y) { return x >= 8 && y >= 8 && x < w - 8 && y < h - 8; };

  // Reference: noise sigma/2 with a profile that says sigma/2, no count map.
  ODSession* s0 = od_session_create(gCtx);
  ODParams p = baseParams(&profHalf);
  Img ref = run(s0, noisyHalf, p);
  od_session_destroy(s0);

  // Same picture, profile says sigma, count map says 4 frames were averaged:
  // the thresholds must come out the same (chroma: exactly; luma differs only
  // by the grain floor, which looks at the profile's own sigma).
  setenv("OD_SPATIAL_TEST_COUNT", "4", 1);
  ODSession* s1 = od_session_create(gCtx);
  p = baseParams(&prof);
  Img got = run(s1, noisyHalf, p);
  od_session_destroy(s1);
  for (int c = 1; c < 3; ++c) {
    Stat d = diffStat(got, ref, c, kAll);
    CHECK(d.maxAbs < 1e-4, "count 4 vs profile/2, channel %d: max difference %.2e", c, d.maxAbs);
  }
  Stat dy = diffStat(got, ref, 0, kAll);
  CHECK(dy.rms < 0.05 * halfSig[0], "count 4 vs profile/2, luma: rms difference %.5f", dy.rms);

  // Left half count 1, right half count 16, the same noise everywhere: the
  // right half must be filtered much less.
  setenv("OD_SPATIAL_TEST_COUNT", "split:16", 1);
  ODSession* s2 = od_session_create(gCtx);
  Img sp = run(s2, noisy, p);
  od_session_destroy(s2);
  unsetenv("OD_SPATIAL_TEST_COUNT");
  auto left = [&](int x, int y) { return inside(x, y) && x < w / 2 - 24; };
  auto right = [&](int x, int y) { return inside(x, y) && x >= w / 2 + 24; };
  for (int c = 0; c < 3; ++c) {
    double l = diffStat(sp, flat, c, left).rms / kSig[c], r = diffStat(sp, flat, c, right).rms / kSig[c];
    printf("  count map, channel %d: noise left %.3f where count = 1, %.3f where count = 16\n", c, l, r);
    CHECK(r > 2.0 * l && r > 0.5, "channel %d: count 16 side %.3f, count 1 side %.3f", c, r, l);
  }
}

static void testSharpen(ODSession* s) {
  gTest = "sharpen";
  ODNoiseProfile prof = whiteProfile(kSig);
  const int w = 200, h = 120;
  for (int q = 0; q <= 1; ++q) {
    ODParams p = baseParams(&prof);
    p.quality = q;
    p.spatialStrength = 0;  // sharpen alone
    p.sharpen = 1.0f;

    // Noise at the profiled level is not amplified.
    Img flat(w, h);
    for (size_t i = 0; i < flat.p[0].size(); ++i) {
      flat.p[0][i] = 0.4f;
      flat.p[1][i] = 0.02f;
    }
    Img noisy = flat;
    addNoise(noisy, kSig, 61);
    std::vector<float> in, out;
    Img o = run(s, noisy, p, &in, &out);
    double before = diffStat(noisy, flat, 0, kAll).rms, after = diffStat(o, flat, 0, kAll).rms;
    printf("  sharpen 1.0 on plain noise, quality %d: luma rms %.5f -> %.5f (x%.3f)\n", q, before, after, after / before);
    CHECK(after < 1.03 * before, "q%d: noise amplified x%.3f", q, after / before);
    CHECK(diffStat(o, noisy, 1, kAll).maxAbs < 2e-6 && diffStat(o, noisy, 2, kAll).maxAbs < 2e-6, "q%d: chroma changed", q);
    CHECK(allFinite(out), "q%d: NaN or Inf", q);

    // A soft edge (clean, so the whole edge is above the floor) gets steeper,
    // with bounded overshoot.
    Img edge(w, h);
    const float lo = 0.3f, hi = 0.6f;
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        float t = ((float)x - 100.0f) / 1.5f;
        edge.at(0, x, y) = lo + (hi - lo) * 0.5f * (1.0f + tanhf(t));
      }
    const float small[3] = {0.002f, 0.002f, 0.002f};
    ODNoiseProfile profSmall = whiteProfile(small);
    ODParams pe = p;
    pe.manualProfile = &profSmall;
    Img es = run(s, edge, pe);
    auto maxSlope = [&](const Img& im) {
      float m = 0;
      for (int x = 80; x < 120; ++x) m = std::max(m, im.at(0, x + 1, 60) - im.at(0, x, 60));
      return m;
    };
    float s0 = maxSlope(edge), s1 = maxSlope(es);
    float mn = 1e9f, mx = -1e9f;
    for (float v : es.p[0]) {
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    }
    printf("  sharpen 1.0 on a soft edge, quality %d: max slope %.4f -> %.4f, overshoot %.4f / %.4f of the step\n", q, s0, s1,
           (lo - mn) / (hi - lo), (mx - hi) / (hi - lo));
    CHECK(s1 > 1.10f * s0, "q%d: edge not sharpened (slope %.4f -> %.4f)", q, s0, s1);
    CHECK(mn > lo - 0.03f * (hi - lo) && mx < hi + 0.03f * (hi - lo), "q%d: overshoot, range %.4f .. %.4f", q, mn, mx);
    // The flat parts of that frame are untouched.
    Stat far = diffStat(es, edge, 0, [&](int x, int) { return x < 60 || x > 140; });
    CHECK(far.maxAbs < 1e-6, "q%d: flat area changed by %.2e", q, far.maxAbs);

    // A hard step: the limiter keeps the overshoot within a tenth of the
    // local range.
    Img stepI(w, h);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) stepI.at(0, x, y) = x < 100 ? lo : hi;
    Img ss = run(s, stepI, pe);
    mn = 1e9f;
    mx = -1e9f;
    for (float v : ss.p[0]) {
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    }
    CHECK(mn < lo - 0.02f * (hi - lo) && mx > hi + 0.02f * (hi - lo), "q%d: hard step not sharpened at all", q);
    CHECK(mn >= lo - 0.10f * (hi - lo) - 5e-4f && mx <= hi + 0.10f * (hi - lo) + 5e-4f,
          "q%d: hard step overshoot, range %.4f .. %.4f", q, mn, mx);

    // After the spatial stage, with sharpening on, a noisy flat frame must
    // still end up cleaner than without sharpening would be noisy: no
    // re-amplification of what was left.
    ODParams p2 = baseParams(&prof);
    p2.quality = q;
    double plain = diffStat(run(s, noisy, p2), flat, 0, kAll).rms;
    p2.sharpen = 1.0f;
    double sharp = diffStat(run(s, noisy, p2), flat, 0, kAll).rms;
    CHECK(sharp < 1.25 * plain + 1e-5, "q%d: sharpen after spatial raised the residual %.5f -> %.5f", q, plain, sharp);
  }
}

// OD_SPATIAL_FUSED=1 (opt-in): the fused final prepare + NL-means renders the
// separate passes' picture bit for bit: sizes that do not fill a 32x8 tile,
// both qualities, a temporal count map, with and without the stable
// reference. Textured so the NL-means weights vary.
static void testFusedIdentity() {
  gTest = "fused_identity";
  ODNoiseProfile prof = whiteProfile(kSig);
  ODParams p = baseParams(&prof);
  p.temporalRadius = 2;
  for (int sz : {0, 1, 2}) {
    const int w = sz == 0 ? 129 : sz == 1 ? 67 : 300, h = sz == 0 ? 73 : sz == 1 ? 41 : 171;
    std::vector<id<MTLBuffer>> bufs;
    for (int i = 0; i < 7; ++i) {
      Img im(w, h);
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          im.p[0][(size_t)y * w + x] = 0.2f + 0.5f * x / w + (((x + i) / 6 + y / 6) % 2 ? 0.08f : 0.0f);
          im.p[1][(size_t)y * w + x] = 0.03f * std::sin(0.07f * (x + 2 * y));
          im.p[2][(size_t)y * w + x] = -0.02f * std::cos(0.05f * (2 * x - y));
        }
      addNoise(im, kSig, 900 + i + 10 * sz);
      std::vector<float> bgra;
      yccToBgra(im, bgra);
      bufs.push_back([gDevice newBufferWithBytes:bgra.data() length:bgra.size() * sizeof(float)
                                         options:MTLResourceStorageModeShared]);
    }
    auto render = [&](bool fused, const char* tune) {
      if (fused) setenv("OD_SPATIAL_FUSED", "1", 1); else unsetenv("OD_SPATIAL_FUSED");
      if (tune) setenv("OD_SPATIAL_TUNE", tune, 1); else unsetenv("OD_SPATIAL_TUNE");
      ODSession* ses = od_session_create(gCtx);
      std::vector<float> all;
      for (int t = 2; t < 5; ++t) {
        ODFrame fr[5] = {};
        int offsets[5] = {}, n = 0;
        for (int k : {0, -1, 1, -2, 2}) {
          fr[n] = {(__bridge void*)bufs[t + k], (uint32_t)w, (uint32_t)h, (uint32_t)(w * 16), OD_PIXEL_BGRA32F,
                   OD_NO_KEY};
          offsets[n++] = k;
        }
        id<MTLBuffer> b = [gDevice newBufferWithLength:w * h * 16 options:MTLResourceStorageModeShared];
        ODFrame out = fr[0];
        out.mtlBuffer = (__bridge void*)b;
        CHECK(od_render(ses, fr, offsets, n, &p, &out, nullptr) == 0, "render failed: %s", od_last_error(ses));
        const float* px = (const float*)b.contents;
        all.insert(all.end(), px, px + (size_t)w * h * 4);
      }
      od_session_destroy(ses);
      unsetenv("OD_SPATIAL_FUSED");
      unsetenv("OD_SPATIAL_TUNE");
      return all;
    };
    for (int q : {0, 1}) {
      p.quality = q;
      for (const char* tune : {(const char*)nullptr, "stable=1,fine=0.75"}) {
        const std::vector<float> a = render(false, tune), b = render(true, tune);
        size_t differ = 0;
        for (size_t i = 0; i < a.size() && a.size() == b.size(); ++i) differ += memcmp(&a[i], &b[i], 4) != 0;
        CHECK(a.size() == b.size() && differ == 0, "%dx%d q%d %s: fused and separate passes differ in %zu values", w,
              h, q, tune ? tune : "default", differ);
      }
    }
  }
}

// The optional reference must reduce temporal grain without touching chroma,
// alpha or the spatial-only fallback. A fixed manual profile removes startup.
static void testStableReference() {
  gTest = "stable_reference";
  const int w = 129, h = 73;
  ODNoiseProfile prof = whiteProfile(kSig);
  ODParams p = baseParams(&prof);
  p.temporalRadius = 2;
  p.lumaStrength = 0.8f;
  Img flat(w, h);
  for (auto& v : flat.p[0]) v = 0.4f;
  std::vector<std::vector<float>> inputs;
  std::vector<id<MTLBuffer>> bufs;
  for (int i = 0; i < 8; ++i) {
    Img im = flat;
    addNoise(im, kSig, 800 + i);
    inputs.emplace_back(); yccToBgra(im, inputs.back());
    id<MTLBuffer> b = [gDevice newBufferWithBytes:inputs.back().data()
      length:inputs.back().size() * sizeof(float) options:MTLResourceStorageModeShared];
    bufs.push_back(b);
  }
  auto sequence = [&](const char* tune, int radius, float amount) {
    if (tune) setenv("OD_SPATIAL_TUNE", tune, 1); else unsetenv("OD_SPATIAL_TUNE");
    ODSession* ses = od_session_create(gCtx);
    p.temporalRadius = radius; p.lumaStrength = amount;
    std::vector<std::vector<float>> result;
    for (int t = 2; t < 6; ++t) {
      ODFrame fr[5] = {}; int offsets[5] = {}; int n = 0;
      for (int k : {0, -1, 1, -2, 2}) {
        if (std::abs(k) > radius) continue;
        fr[n] = {(__bridge void*)bufs[t+k], (uint32_t)w, (uint32_t)h,
                 (uint32_t)(w*16), OD_PIXEL_BGRA32F, OD_NO_KEY};
        offsets[n++] = k;
      }
      id<MTLBuffer> b = [gDevice newBufferWithLength:w*h*16 options:MTLResourceStorageModeShared];
      ODFrame out = fr[0]; out.mtlBuffer = (__bridge void*)b;
      CHECK(od_render(ses, fr, offsets, n, &p, &out, nullptr) == 0, "stable render failed");
      result.emplace_back(w*h*4); memcpy(result.back().data(), b.contents, w*h*16);
      CHECK(allFinite(result.back()), "stable reference is not finite");
      int alphaChanged = 0;
      for (int i = 0; i < w*h; ++i) alphaChanged += result.back()[4*i+3] != inputs[t][4*i+3];
      CHECK(alphaChanged == 0, "alpha changed in %d pixels", alphaChanged);
    }
    od_session_destroy(ses); unsetenv("OD_SPATIAL_TUNE");
    return result;
  };
  for (int q : {0, 1}) {
    p.quality = q;
    auto a = sequence(nullptr, 2, 0.8f);
    auto b = sequence("stable=1,fine=0.75", 2, 0.8f);
    double va = 0, vb = 0;
    for (size_t i = 1; i < a.size(); ++i) {
      Img a0=bgraToYcc(a[i-1],w,h), a1=bgraToYcc(a[i],w,h);
      Img b0=bgraToYcc(b[i-1],w,h), b1=bgraToYcc(b[i],w,h);
      auto inside = [&](int x,int y){return x>8 && y>8 && x<w-8 && y<h-8;};
      va += std::pow(diffStat(a1,a0,0,inside).rms,2);
      vb += std::pow(diffStat(b1,b0,0,inside).rms,2);
      for (int c : {1,2}) CHECK(diffStat(a1,b1,c,kAll).maxAbs<2e-6, "stable reference changed chroma");
    }
    CHECK(vb < va*0.9, "q%d temporal grain did not drop: %.6g -> %.6g",q,va,vb);
    CHECK(sequence(nullptr,0,0.8f)==sequence("stable=1,fine=0.75",0,0.8f), "spatial-only fallback changed");
    CHECK(sequence(nullptr,2,1e-7f)==sequence("stable=1,fine=0.75",2,1e-7f), "near-zero luma reference did not fade");
    p.spatialStrength = 1e-7f;
    CHECK(sequence(nullptr,2,0.8f)==sequence("stable=1,fine=0.75",2,0.8f), "near-zero spatial reference did not fade");
    p.spatialStrength = 0.5f;
    CHECK(sequence(nullptr,2,0)==sequence("stable=1,fine=0.75",2,0), "luma zero changed");
  }
}

// Correlated noise exposes the old count assumption even without camera media.
// At rho near one neighbours differ (so are not exact repeats), but do not
// carry independent observations of the noise. Full-resolution odd raster.
static void testTemporalResidual() {
  gTest = "temporal_residual";
  const int w=193,h=129;
  ODNoiseProfile prof=whiteProfile(kSig);
  Img clean(w,h); for (auto& v:clean.p[0]) v=.35f;
  for (int q : {0,1}) for (float correlation : {0.0f,.9f,.999f}) {
    std::vector<id<MTLBuffer>> bufs;
    for (int j=0;j<11;++j) {
      Img a=clean;
      for (int c=0;c<3;++c) for(int y=0;y<h;++y) for(int x=0;x<w;++x)
        a.at(c,x,y)+=kSig[c]*(sqrtf(correlation)*gauss(x,y,c,799)+sqrtf(1-correlation)*gauss(x,y,c,400+j));
      std::vector<float> rgb; yccToBgra(a,rgb);
      bufs.push_back([gDevice newBufferWithBytes:rgb.data() length:rgb.size()*sizeof(float) options:MTLResourceStorageModeShared]);
    }
    ODSession* ses=od_session_create(gCtx);
    ODParams p=baseParams(&prof); p.quality=q; p.lumaStrength=.8f; p.temporalNoiseModel=1;
    auto calc=[&](int radius,float strength) {
      p.temporalRadius=radius; p.temporalStrength=strength;
      ODFrame fr[11]={}; int offsets[11]={}; int n=0;
      for (int k : {0,-1,1,-2,2,-3,3,-4,4,-5,5}) if(abs(k)<=radius) {
        fr[n]={(__bridge void*)bufs[k+5],(uint32_t)w,(uint32_t)h,(uint32_t)(w*16),OD_PIXEL_BGRA32F,OD_NO_KEY}; offsets[n++]=k;
      }
      id<MTLBuffer> b=[gDevice newBufferWithLength:w*h*16 options:MTLResourceStorageModeShared];
      ODFrame out=fr[0];out.mtlBuffer=(__bridge void*)b;
      CHECK(od_render(ses,fr,offsets,n,&p,&out,nullptr)==0,"residual render: %s",od_last_error(ses));
      std::vector<float> rgb(w*h*4);memcpy(rgb.data(),b.contents,w*h*16);
      CHECK(allFinite(rgb),"nonfinite output");
      for(int i=0;i<w*h;++i) if(rgb[i*4+3]!=.75f) {CHECK(false,"alpha changed");break;}
      return rgb;
    };
    auto error=[&](const std::vector<float>& rgb) {return diffStat(bgraToYcc(rgb,w,h),clean,0,[](int x,int y){return x>16&&y>16&&x<w-16&&y<h-16;}).rms;};
    double prev=1;
    for(int r:{0,1,2,3,5}) {
      double e=error(calc(r,1));
      printf("residual q%d rho %.3f frames %d rms %.9f\n",q,correlation,2*r+1,e);
      CHECK(e<=prev*1.01,"more frames increased noise: %.8f -> %.8f",prev,e);prev=e;
    }
    prev=1;
    for(float strength:{0.f,.25f,.5f,.75f,1.f}) {
      double e=error(calc(2,strength));
      printf("residual q%d rho %.3f strength %.2f rms %.9f\n",q,correlation,strength,e);
      CHECK(e<=prev*1.01,"more strength increased noise: %.8f -> %.8f",prev,e);prev=e;
    }
    p.temporalNoiseModel=0;auto off=calc(0,1);auto legacyZero=calc(2,0);
    p.temporalNoiseModel=1;CHECK(off==calc(0,1),"radius zero changed");CHECK(legacyZero==calc(2,0),"strength zero changed");
    p.temporalNoiseModel=29;auto invalid=calc(2,1);p.temporalNoiseModel=0;CHECK(invalid==calc(2,1),"invalid model not legacy");
    od_session_destroy(ses);
  }
}

int main() {
  @autoreleasepool {
    // Both grain modes are exercised by `make test` (test.mk sets the
    // session override, OD_GRAIN_PROFILE_V2=0 and =1, so zeroed and default
    // ODParams render alike); run directly, the default mode (finest grain).
    setenv("OD_GRAIN_PROFILE_V2", "1", 0);
    gDevice = MTLCreateSystemDefaultDevice();
    if (!gDevice) {
      fprintf(stderr, "no Metal device\n");
      return 2;
    }
    gQueue = [gDevice newCommandQueue];
    char err[2048] = {0};
    gCtx = od_context_create((__bridge void*)gDevice, (__bridge void*)gQueue, err, sizeof(err));
    if (!gCtx) {
      fprintf(stderr, "od_context_create: %s\n", err);
      return 2;
    }
    unsetenv("OD_SPATIAL_TEST_COUNT");
    unsetenv("OD_SPATIAL_TUNE");
    ODSession* s = od_session_create(gCtx);
    if (!s) {
      fprintf(stderr, "od_session_create failed\n");
      return 2;
    }
    testPassThrough(s);
    testSizesAndBorders(s);
    testNoiseReduction(s);
    testEdges(s);
    testRange(s);
    testSpecks(s);
    testExtremes(s);
    testSharpen(s);
    s_hotSession = s;
    testHotPixels();
    testOutputNonFinite();
    od_session_destroy(s);
    testCount();
    testStableReference();
    testTemporalResidual();
    testFusedIdentity();
    od_context_destroy(gCtx);
    printf("test_spatial: %d checks, %d failures\n", gChecks, gFailures);
    return gFailures ? 1 : 0;
  }
}
