// SPDX-License-Identifier: MIT
// Integration tests: the real stages running together through od_render,
// with the hand-offs between them (profile -> both filters, the temporal
// stage's effective frame count -> spatial and sharpen, amounts applied once,
// the debug views, quality 0 / 1, od_session_reset, the idle trim, one profile
// per frame size, the asynchronous profile path, the cache and pool limits,
// the profile's cold start and restarts, realtime renders behind a held
// command queue, and chroma noise that persists from frame to frame).
//
// The stage suites (tests/profile, tests/temporal, tests/spatial) test each
// stage alone with a profile handed to it. Here the frames are synthetic, the
// clean signal is known, and every check is a comparison between two renders
// or against the clean signal. Memory: about forty 640x368 frames and a few
// sessions' caches at that size (< 300 MB).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <unistd.h>

#include "opendenoise/opendenoise.h"

#include "debug.h"

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

static const uint32_t W = 640, H = 368;
static const int kFrames = 7;  // the window of frame 3 at radius 3
static const float kSigma = 0.02f;

static id<MTLDevice> gDevice;
static ODContext* gCtx;

// ---- scene -----------------------------------------------------------------
// Left third: flat patches at eight grey levels (the profiler's food).
// Middle third: fine static texture, amplitude about 1.2 sigma.
// Right third: a smooth gradient with a bright square moving 3 px per frame.
static void cleanRGB(uint32_t x, uint32_t y, int frame, float rgb[3]) {
  float v;
  if (x < W / 3) {
    const int patch = (int)(y * 8 / H);
    v = 0.08f + 0.1f * (float)patch;
  } else if (x < 2 * W / 3) {
    v = 0.45f + 0.024f * sinf(0.9f * (float)x + 0.4f * (float)y) * cosf(0.7f * (float)y);
  } else {
    v = 0.25f + 0.3f * (float)(x - 2 * W / 3) / (float)(W / 3);
    const int sx = (int)(2 * W / 3) + 30 + 3 * frame, sy = 150;
    if ((int)x >= sx && (int)x < sx + 48 && (int)y >= sy && (int)y < sy + 48) v = 0.85f;
  }
  rgb[0] = v * 1.05f;
  rgb[1] = v;
  rgb[2] = v * 0.9f;
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

struct Frame {
  id<MTLBuffer> buffer;
  float* px() const { return (float*)buffer.contents; }  // B, G, R, A
  ODFrame od(int64_t key) const {
    ODFrame f;
    f.mtlBuffer = (__bridge void*)buffer;
    f.width = W;
    f.height = H;
    f.rowBytes = W * 16;
    f.format = OD_PIXEL_BGRA32F;
    f.frameKey = key;
    return f;
  }
};

static Frame newFrame() {
  Frame f;
  f.buffer = [gDevice newBufferWithLength:(NSUInteger)W * H * 16 options:MTLResourceStorageModeShared];
  return f;
}

static void fill(const Frame& f, int frame, float sigma, uint32_t seed) {
  float* p = f.px();
  for (uint32_t y = 0; y < H; ++y) {
    for (uint32_t x = 0; x < W; ++x) {
      float rgb[3];
      cleanRGB(x, y, frame, rgb);
      for (int c = 0; c < 3; ++c) {
        const uint32_t s = (x * 73856093u) ^ (y * 19349663u) ^ ((uint32_t)c * 83492791u) ^
                           ((uint32_t)frame * 2654435761u) ^ (seed * 40503u);
        rgb[c] += sigma * gauss(s);
      }
      float* o = p + ((size_t)y * W + x) * 4;
      o[0] = rgb[2];
      o[1] = rgb[1];
      o[2] = rgb[0];
      o[3] = 1.0f;
    }
  }
}

static inline float lumaOf(const float* bgra) { return 0.2126f * bgra[2] + 0.7152f * bgra[1] + 0.0722f * bgra[0]; }

enum Region { kAll, kFlat, kTexture, kMoving };

static bool inRegion(Region r, uint32_t x, uint32_t y) {
  if (x < 8 || y < 8 || x >= W - 8 || y >= H - 8) return false;  // frame borders
  switch (r) {
    case kFlat: return x < W / 3 - 8 && (y * 8 % H) > 8 * 8 && (y * 8 % H) < H - 8 * 8;  // away from patch edges
    case kTexture: return x >= W / 3 + 8 && x < 2 * W / 3 - 8;
    case kMoving: return x >= 2 * W / 3 + 8;
    default: return true;
  }
}

// rms luma difference between two frames over a region.
static double rmsLuma(const Frame& a, const Frame& b, Region r) {
  double s = 0;
  size_t n = 0;
  for (uint32_t y = 0; y < H; ++y) {
    for (uint32_t x = 0; x < W; ++x) {
      if (!inRegion(r, x, y)) continue;
      const size_t i = ((size_t)y * W + x) * 4;
      const double d = lumaOf(a.px() + i) - lumaOf(b.px() + i);
      s += d * d;
      ++n;
    }
  }
  return n ? sqrt(s / (double)n) : 0;
}

static double rmsChroma(const Frame& a, const Frame& b, Region r) {
  double s = 0;
  size_t n = 0;
  for (uint32_t y = 0; y < H; ++y) {
    for (uint32_t x = 0; x < W; ++x) {
      if (!inRegion(r, x, y)) continue;
      const size_t i = ((size_t)y * W + x) * 4;
      const double da = (a.px()[i] - lumaOf(a.px() + i)) - (b.px()[i] - lumaOf(b.px() + i));
      s += da * da;
      ++n;
    }
  }
  return n ? sqrt(s / (double)n) : 0;
}

static bool allFinite(const Frame& f) {
  const float* p = f.px();
  for (size_t i = 0; i < (size_t)W * H * 4; ++i)
    if (!std::isfinite(p[i])) return false;
  return true;
}

// The profile of white noise of sigma s on R, G and B (same numbers as
// odbench --manual-sigma).
static ODNoiseProfile whiteProfile(float s) {
  ODNoiseProfile p;
  memset(&p, 0, sizeof(p));
  const float chan[3] = {0.7497f, 0.6416f, 0.6770f};
  const float band[OD_BANDS] = {0.8908f, 0.2007f, 0.0855f, 0.0412f};
  for (int c = 0; c < 3; ++c) {
    for (int b = 0; b < OD_BANDS; ++b)
      for (int l = 0; l < OD_LEVELS; ++l) p.sigma[c][b][l] = s * chan[c] * band[b];
    // c_0 - c_4 holds 0.9987 of white noise; the bands are correlated, so it
    // is not the root sum of squares of the band gains (0.918).
    for (int l = 0; l < OD_LEVELS; ++l) p.sigmaFull[c][l] = s * chan[c] * 0.9987f;
  }
  p.confidence = 1;
  p.valid = 1;
  return p;
}

static std::vector<Frame> gNoisy;  // frames 0..6
static Frame gClean;               // frame 3, no noise

// Renders frame 3 of `src` with the window the radius allows. A fresh session
// per call unless one is given.
static bool render(const std::vector<Frame>& src, ODParams p, const Frame& out, ODStats* stats = nullptr,
                   ODSession* session = nullptr) {
  ODSession* s = session ? session : od_session_create(gCtx);
  ODFrame frames[OD_MAX_FRAMES];
  int offsets[OD_MAX_FRAMES];
  int n = 0;
  frames[n] = src[3].od(3);
  offsets[n++] = 0;
  for (int d = 1; d <= p.temporalRadius && d <= 3; ++d) {
    for (int sgn = -1; sgn <= 1; sgn += 2) {
      frames[n] = src[(size_t)(3 + sgn * d)].od(3 + sgn * d);
      offsets[n++] = sgn * d;
    }
  }
  ODFrame o = out.od(OD_NO_KEY);
  const int rc = od_render(s, frames, offsets, n, &p, &o, stats);
  if (rc != 0) fprintf(stderr, "od_render: %s\n", od_last_error(s));
  if (!session) od_session_destroy(s);
  return rc == 0;
}

static ODParams base() {
  ODParams p = od_params_default();
  p.waitUntilCompleted = 1;
  return p;
}

// ---- tests -----------------------------------------------------------------

// Every stage alone and all together, with the automatic profile.
static void testStagesTogether() {
  gTest = "stages_together";
  const double in = rmsLuma(gNoisy[3], gClean, kAll);
  const double inC = rmsChroma(gNoisy[3], gClean, kAll);
  for (int q = 0; q <= 1; ++q) {
    Frame oT = newFrame(), oS = newFrame(), oB = newFrame();
    ODParams p = base();
    p.quality = q;
    p.lumaStrength = 1;
    ODParams pt = p, ps = p;
    pt.spatialStrength = 0;
    ps.temporalRadius = 0;
    ODStats st = {};
    CHECK(render(gNoisy, pt, oT), "temporal only failed");
    CHECK(render(gNoisy, ps, oS), "spatial only failed");
    CHECK(render(gNoisy, p, oB, &st), "full pipeline failed");
    const double eT = rmsLuma(oT, gClean, kAll), eS = rmsLuma(oS, gClean, kAll), eB = rmsLuma(oB, gClean, kAll);
    printf("  quality %d, luma rms error against clean: input %.5f, temporal %.5f, spatial %.5f, both %.5f\n", q, in,
           eT, eS, eB);
    CHECK(st.profile.valid == 1, "no automatic profile");
    CHECK(st.framesUsed == 5, "framesUsed %d, expected 5", st.framesUsed);
    CHECK(eT < 0.62 * in, "temporal alone removed too little: %.5f of %.5f", eT, in);
    CHECK(eS < 0.62 * in, "spatial alone removed too little: %.5f of %.5f", eS, in);
    CHECK(eB < eT && eB < eS, "both stages (%.5f) not better than temporal (%.5f) and spatial (%.5f) alone", eB, eT, eS);
    CHECK(eB < 0.4 * in, "full pipeline removed too little: %.5f of %.5f", eB, in);
    const double cB = rmsChroma(oB, gClean, kAll);
    CHECK(cB < 0.35 * inC, "chroma: %.5f of %.5f left", cB, inC);
    CHECK(allFinite(oT) && allFinite(oS) && allFinite(oB), "non-finite output");
    // The automatic profile on this scene against the truth it was made with.
    const ODNoiseProfile truth = whiteProfile(kSigma);
    const float r0 = st.profile.sigma[0][0][3] / truth.sigma[0][0][3];
    printf("  quality %d, automatic profile: Y band 0 at knot 3 is x%.2f of the truth, confidence %.2f\n", q, r0,
           st.profile.confidence);
    CHECK(r0 > 0.7f && r0 < 1.3f, "automatic profile off: x%.2f of the truth", r0);
    // sigmaFull is the sigma of c_0 - c_4 (opendenoise.h): 1.121 times band 0
    // for white noise (0.9987 / 0.8908), not 1.031 (0.918 / 0.8908).
    const float rf = st.profile.sigmaFull[0][3] / st.profile.sigma[0][0][3];
    printf("  quality %d, automatic profile: sigmaFull / band 0 at knot 3 = %.3f (white noise: 1.121)\n", q, rf);
    CHECK(rf > 1.08f && rf < 1.16f, "sigmaFull / band 0 = %.3f, expected about 1.121 for white noise", rf);
  }
}

// The effective frame count must reach the spatial stage: after a merge of
// five frames the noise is 1/sqrt(5) of the profile, and a spatial filter
// that does not know it smooths the texture away.
static void testCountReachesSpatial() {
  gTest = "count_handoff";
  const ODNoiseProfile prof = whiteProfile(kSigma);
  ODParams p = base();
  p.manualProfile = &prof;
  p.lumaStrength = 1;
  // (a) the pipeline: temporal then spatial, count handed over.
  Frame both = newFrame();
  CHECK(render(gNoisy, p, both), "pipeline failed");
  // (b) the same two stages in two renders: the spatial stage then gets the
  // merged frame as if it were a plain noisy one (count = 1).
  ODParams pt = p;
  pt.spatialStrength = 0;
  std::vector<Frame> merged = gNoisy;
  merged[3] = newFrame();
  CHECK(render(gNoisy, pt, merged[3]), "temporal failed");
  ODParams ps = p;
  ps.temporalRadius = 0;
  Frame blind = newFrame();
  CHECK(render(merged, ps, blind), "spatial failed");
  const double tA = rmsLuma(both, gClean, kTexture), tB = rmsLuma(blind, gClean, kTexture);
  const double tT = rmsLuma(merged[3], gClean, kTexture);
  const double fA = rmsLuma(both, gClean, kFlat), fT = rmsLuma(merged[3], gClean, kFlat);
  printf("  texture rms error: temporal %.5f, + spatial with the count %.5f, + spatial without it %.5f\n", tT, tA, tB);
  printf("  flat rms error:    temporal %.5f, + spatial with the count %.5f\n", fT, fA);
  CHECK(tA < 0.9 * tB, "the count map does not protect the texture: %.5f with, %.5f without", tA, tB);
  CHECK(fA < 0.8 * fT, "spatial after temporal removes too little in flat areas: %.5f after %.5f", fA, fT);
}

// Amounts are applied once: at luma 0.5 the output is half way between the
// input and the luma-1 output, whichever stages run.
static void testAmountsOnce() {
  gTest = "amounts_once";
  const ODNoiseProfile prof = whiteProfile(kSigma);
  for (int mode = 0; mode < 3; ++mode) {
    ODParams p = base();
    p.manualProfile = &prof;
    if (mode == 1) p.spatialStrength = 0;
    if (mode == 2) p.temporalRadius = 0;
    Frame full = newFrame(), half = newFrame(), none = newFrame();
    p.lumaStrength = 1;
    p.chromaStrength = 1;
    CHECK(render(gNoisy, p, full), "render failed");
    p.lumaStrength = 0.5f;
    p.chromaStrength = 0.5f;
    CHECK(render(gNoisy, p, half), "render failed");
    p.lumaStrength = 0;
    p.chromaStrength = 0;
    CHECK(render(gNoisy, p, none), "render failed");
    double num = 0, den = 0;
    for (size_t i = 0; i < (size_t)W * H * 4; ++i) {
      const double in = gNoisy[3].px()[i];
      const double want = in + 0.5 * (full.px()[i] - in);
      num += (half.px()[i] - want) * (half.px()[i] - want);
      den += (full.px()[i] - in) * (full.px()[i] - in);
    }
    const double rel = sqrt(num / den);
    printf("  mode %d (0 both, 1 temporal, 2 spatial): amount 0.5 is off the half-way point by %.3f of the change\n", mode,
           rel);
    CHECK(rel < 0.06, "mode %d: amount 0.5 is not half of amount 1 (%.3f)", mode, rel);
    CHECK(memcmp(none.px(), gNoisy[3].px(), (size_t)W * H * 16) == 0, "mode %d: amount 0 changed the frame", mode);
  }
}

// Each strength does what opendenoise.h says: more of it removes more.
static void testStrengths() {
  gTest = "strengths";
  const ODNoiseProfile prof = whiteProfile(kSigma);
  auto flatErr = [&](ODParams p) {
    Frame o = newFrame();
    p.manualProfile = &prof;
    CHECK(render(gNoisy, p, o), "render failed");
    return rmsLuma(o, gClean, kFlat);
  };
  ODParams p = base();
  p.lumaStrength = 1;
  {
    ODParams a = p, b = p, c = p;
    a.spatialStrength = 0.2f;
    b.spatialStrength = 0.5f;
    c.spatialStrength = 1.0f;
    a.temporalRadius = b.temporalRadius = c.temporalRadius = 0;
    const double ea = flatErr(a), eb = flatErr(b), ec = flatErr(c);
    printf("  spatialStrength 0.2 / 0.5 / 1: flat rms %.5f %.5f %.5f\n", ea, eb, ec);
    CHECK(ea > eb && eb > ec, "spatialStrength is not monotonic: %.5f %.5f %.5f", ea, eb, ec);
  }
  {
    double prev = 1e9;
    for (int r = 0; r <= 3; ++r) {
      ODParams a = p;
      a.spatialStrength = 0;
      a.temporalRadius = r;
      const double e = flatErr(a);
      printf("  temporalRadius %d: flat rms %.5f\n", r, e);
      if (r > 0) CHECK(e < prev, "radius %d (%.5f) is not better than radius %d (%.5f)", r, e, r - 1, prev);
      prev = e;
    }
  }
  {
    ODParams a = p, b = p, c = p;
    a.noiseLevel = 0.5f;
    b.noiseLevel = 1.0f;
    c.noiseLevel = 2.0f;
    const double ea = flatErr(a), eb = flatErr(b), ec = flatErr(c);
    printf("  noiseLevel 0.5 / 1 / 2: flat rms %.5f %.5f %.5f\n", ea, eb, ec);
    CHECK(ea > eb && eb >= ec * 0.98, "noiseLevel is not monotonic: %.5f %.5f %.5f", ea, eb, ec);
  }
  {
    ODParams a = p, b = p;
    a.temporalStrength = 0.25f;
    b.temporalStrength = 1.0f;
    a.spatialStrength = b.spatialStrength = 0;
    const double ea = flatErr(a), eb = flatErr(b);
    printf("  temporalStrength 0.25 / 1: flat rms %.5f %.5f\n", ea, eb);
    CHECK(ea > eb, "temporalStrength is not monotonic: %.5f %.5f", ea, eb);
  }
  {
    ODParams a = p, b = p;
    a.lumaStrength = 1;
    a.chromaStrength = 0;
    b.lumaStrength = 0;
    b.chromaStrength = 1;
    Frame oa = newFrame(), ob = newFrame();
    a.manualProfile = b.manualProfile = &prof;
    CHECK(render(gNoisy, a, oa) && render(gNoisy, b, ob), "render failed");
    const double in = rmsLuma(gNoisy[3], gClean, kFlat), inC = rmsChroma(gNoisy[3], gClean, kFlat);
    printf("  luma only: luma %.2f, chroma %.2f of the input; chroma only: luma %.2f, chroma %.2f\n",
           rmsLuma(oa, gClean, kFlat) / in, rmsChroma(oa, gClean, kFlat) / inC, rmsLuma(ob, gClean, kFlat) / in,
           rmsChroma(ob, gClean, kFlat) / inC);
    CHECK(rmsLuma(ob, gClean, kFlat) > 0.98 * in, "chroma only changed the luma");
    CHECK(rmsChroma(ob, gClean, kFlat) < 0.5 * inC, "chroma only left the chroma noise");
    CHECK(rmsLuma(oa, gClean, kFlat) < 0.5 * in, "luma only left the luma noise");
  }
}

// Sharpen after denoising: no noise amplification in flat areas (the coring
// threshold follows what is left, count map included), detail gets stronger.
static void testSharpen() {
  gTest = "sharpen";
  const ODNoiseProfile prof = whiteProfile(kSigma);
  for (int q = 0; q <= 1; ++q) {
    for (int mode = 0; mode < 2; ++mode) {  // 0: full pipeline, 1: temporal + sharpen
      ODParams p = base();
      p.manualProfile = &prof;
      p.quality = q;
      p.lumaStrength = 1;
      if (mode == 1) p.spatialStrength = 0;
      Frame plain = newFrame(), sharp = newFrame();
      CHECK(render(gNoisy, p, plain), "render failed");
      p.sharpen = 1;
      CHECK(render(gNoisy, p, sharp), "render failed");
      const double f0 = rmsLuma(plain, gClean, kFlat), f1 = rmsLuma(sharp, gClean, kFlat);
      const double change = rmsLuma(sharp, plain, kMoving);
      printf("  quality %d, %s: flat rms %.5f -> %.5f with sharpen 1; change near the square %.5f\n", q,
             mode ? "temporal + sharpen" : "full pipeline", f0, f1, change);
      CHECK(f1 < 1.25 * f0, "sharpen amplifies the noise left in flat areas: %.5f -> %.5f", f0, f1);
      CHECK(change > 1e-4, "sharpen 1 did nothing to the edges");
      CHECK(allFinite(sharp), "non-finite output");
    }
  }
}

// The sharpen threshold follows what the denoising stages left (repo-8): a
// fine texture whose band-0 amplitude lies between the coring threshold after
// a temporal merge of five frames (kCore x sigma x 1/sqrt(5)) and the one with
// nothing denoised (kCore x sigma) is sharpened after the temporal stage and
// left bit-identical by sharpen alone. Catches a threshold that ignores the
// luma amount, the count map or which stages ran, at both qualities.
static void testSharpenThreshold() {
  gTest = "sharpen_threshold";
  const ODNoiseProfile prof = whiteProfile(kSigma);
  // Band 0 of a period-4 sinusoid is 0.75 of its amplitude, band 1 0.25.
  // Luma sigma of band 0 is 0.668 kSigma, of band 1 0.150 kSigma: thresholds
  // 0.033 and 0.0075 undenoised, 0.015 and 0.0034 after the merge. A = 0.026:
  // band 0 at 0.0195, band 1 at 0.0065.
  const float A = 0.026f;
  std::vector<Frame> frames;
  for (int f = 0; f < 5; ++f) {
    frames.push_back(newFrame());
    float* px = frames.back().px();
    for (uint32_t y = 0; y < H; ++y)
      for (uint32_t x = 0; x < W; ++x) {
        const float v = 0.45f + A * sinf(1.5707963f * (float)x);
        for (int c = 0; c < 3; ++c) {
          const uint32_t sd = (x * 73856093u) ^ (y * 19349663u) ^ ((uint32_t)c * 83492791u) ^ ((uint32_t)f * 2654435761u);
          px[((size_t)y * W + x) * 4 + c] = v + 0.002f * gauss(sd ^ 0x5151u);  // no exact repeats
        }
        px[((size_t)y * W + x) * 4 + 3] = 1.0f;
      }
  }
  auto renderMid = [&](ODParams p, const Frame& out) {
    ODSession* s = od_session_create(gCtx);
    ODFrame fr[5];
    const int off[5] = {0, -1, 1, -2, 2};
    for (int k = 0; k < 5; ++k) fr[k] = frames[(size_t)(2 + off[k])].od(OD_NO_KEY);
    ODFrame o = out.od(OD_NO_KEY);
    const int rc = od_render(s, fr, off, p.temporalRadius > 0 ? 5 : 1, &p, &o, nullptr);
    od_session_destroy(s);
    return rc == 0;
  };
  auto rmsDiff = [](const Frame& a, const Frame& b) {
    double sum = 0;
    size_t n = 0, same = 0;
    for (uint32_t y = 16; y < H - 16; ++y)
      for (uint32_t x = 16; x < W - 16; ++x) {
        const size_t i = ((size_t)y * W + x) * 4;
        const double d = lumaOf(a.px() + i) - lumaOf(b.px() + i);
        sum += d * d;
        ++n;
        same += memcmp(a.px() + i, b.px() + i, 16) == 0;
      }
    return std::make_pair(sqrt(sum / (double)n), (double)same / (double)n);
  };
  for (int q = 0; q <= 1; ++q) {
    ODParams p = base();
    p.manualProfile = &prof;
    p.quality = q;
    p.lumaStrength = 1;
    p.spatialStrength = 0;
    Frame plain = newFrame(), sharp = newFrame();
    // temporal + sharpen
    CHECK(renderMid(p, plain), "render failed");
    p.sharpen = 1;
    CHECK(renderMid(p, sharp), "render failed");
    const auto t = rmsDiff(sharp, plain);
    // sharpen alone: nothing was denoised, the threshold is the full sigma
    ODParams ps = p;
    ps.temporalRadius = 0;
    ps.sharpen = 1;
    Frame alone = newFrame();
    CHECK(renderMid(ps, alone), "render failed");
    size_t changed = 0;  // away from the frame border, where edge replication raises the bands
    for (uint32_t y = 16; y < H - 16; ++y)
      for (uint32_t x = 16; x < W - 16; ++x) {
        const size_t i = ((size_t)y * W + x) * 4;
        changed += memcmp(&alone.px()[i], &frames[2].px()[i], 16) != 0;
      }
    printf("  quality %d: texture at %.3f: sharpen after the temporal merge changes it by %.5f rms; sharpen alone "
           "changes %zu pixels\n", q, A, t.first, changed);
    CHECK(t.first > 0.03 * A, "q=%d: sharpen after a five-frame merge left the texture alone (%.5f rms): its threshold "
          "ignores what was denoised", q, t.first);
    CHECK(changed == 0, "q=%d: sharpen alone changed %zu pixels of a texture under its full-noise threshold", q,
          changed);
  }
}

static void testViews() {
  gTest = "views";
  ODParams p = base();
  Frame out = newFrame(), normal = newFrame();
  CHECK(render(gNoisy, p, normal), "render failed");
  for (int v = OD_VIEW_NOISE; v <= OD_VIEW_PROFILE; ++v) {
    ODParams pv = p;
    pv.view = v;
    memset(out.px(), 0, (size_t)W * H * 16);
    CHECK(render(gNoisy, pv, out), "view %d failed", v);
    CHECK(allFinite(out), "view %d: non-finite output", v);
    const double d = rmsLuma(out, normal, kAll);
    printf("  view %d differs from the normal output by %.4f rms\n", v, d);
    CHECK(d > 0.01, "view %d looks like the normal output", v);
    // Alpha is the source's in every view.
    bool alpha = true;
    for (size_t i = 3; i < (size_t)W * H * 4; i += 4) alpha = alpha && out.px()[i] == 1.0f;
    CHECK(alpha, "view %d changed alpha", v);
  }
  // Noise view: mid-grey where nothing was removed (amounts 0 is the copy
  // path, so use a tiny amount).
  ODParams pn = p;
  pn.view = OD_VIEW_NOISE;
  CHECK(render(gNoisy, pn, out), "noise view failed");
  double mean = 0;
  for (size_t i = 0; i < (size_t)W * H; ++i) mean += lumaOf(out.px() + i * 4);
  mean /= (double)W * H;
  CHECK(fabs(mean - 0.5) < 0.02, "noise view is not centred on mid-grey: %.4f", mean);
}

// After od_session_reset the very next render has a profile again, also in
// asynchronous playback mode, and the output is denoised.
static void testReset() {
  gTest = "reset";
  ODSession* s = od_session_create(gCtx);
  ODParams p = od_params_default();
  p.quality = 0;
  p.waitUntilCompleted = 0;
  p.lumaStrength = 1;
  Frame out = newFrame();
  ODStats st = {};
  for (int i = 0; i < 12; ++i) CHECK(render(gNoisy, p, out, &st, s), "render %d failed", i);
  CHECK(st.profile.valid == 1, "no profile after 12 renders");
  od_session_reset(s);
  st = ODStats{};
  CHECK(render(gNoisy, p, out, &st, s), "render after reset failed");
  CHECK(st.profile.valid == 1, "the first render after a reset had no profile");
  // Wait for the GPU, then look at the result.
  p.waitUntilCompleted = 1;
  CHECK(render(gNoisy, p, out, &st, s), "blocking render failed");
  const double e = rmsLuma(out, gClean, kAll), in = rmsLuma(gNoisy[3], gClean, kAll);
  CHECK(e < 0.5 * in, "after a reset the frame is not denoised: %.5f of %.5f", e, in);
  od_session_destroy(s);
}

// A host that changes what is upstream and keeps the frame keys (the cache
// then holds frames that no longer exist): the output must still be the new
// frame, denoised, never a mix with the old one.
static void testStaleKeys() {
  gTest = "stale_keys";
  const ODNoiseProfile prof = whiteProfile(kSigma);
  ODSession* s = od_session_create(gCtx);
  ODParams p = base();
  p.manualProfile = &prof;
  p.lumaStrength = 1;
  Frame out = newFrame();
  CHECK(render(gNoisy, p, out, nullptr, s), "first render failed");
  // The same keys, other pixels: the whole window 0.15 brighter, new noise.
  std::vector<Frame> other;
  for (int f = 0; f < kFrames; ++f) {
    other.push_back(newFrame());
    fill(other.back(), f, kSigma, 77);
    float* px = other.back().px();
    for (size_t i = 0; i < (size_t)W * H; ++i)
      for (int c = 0; c < 3; ++c) px[i * 4 + c] += 0.15f;
  }
  Frame cleanOther = newFrame();
  fill(cleanOther, 3, 0, 0);
  for (size_t i = 0; i < (size_t)W * H; ++i)
    for (int c = 0; c < 3; ++c) cleanOther.px()[i * 4 + c] += 0.15f;
  // Only the current frame is new to the session; the neighbours it has
  // cached are the old, darker ones.
  CHECK(render(other, p, out, nullptr, s), "second render failed");
  const double in = rmsLuma(other[3], cleanOther, kAll), e = rmsLuma(out, cleanOther, kAll);
  printf("  same keys, new pixels: rms error %.5f (noisy input %.5f)\n", e, in);
  CHECK(e < 0.7 * in, "stale cache leaked into the output: error %.5f, input %.5f", e, in);
  od_session_destroy(s);
}

// Many effect instances: a session that has not rendered for a while gives
// its frame cache back (to the context, on the memory queue, without any other
// render), the session used last too after a longer wait, the context's idle
// pool with it, and each still renders the same picture afterwards.
static void testIdleTrim() {
  gTest = "idle_trim";
  setenv("OD_IDLE_TRIM_MS", "200", 1);  // others after 200 ms, the last one after 1000 ms
  char err[512] = {0};
  ODContext* ctx = od_context_create((__bridge void*)gDevice, nullptr, err, sizeof(err));
  unsetenv("OD_IDLE_TRIM_MS");
  CHECK(ctx != nullptr, "context: %s", err);
  if (!ctx) return;
  const ODNoiseProfile prof = whiteProfile(kSigma);
  ODParams p = base();
  p.manualProfile = &prof;
  ODSession* a = od_session_create(ctx);
  ODSession* b = od_session_create(ctx);
  Frame first = newFrame(), again = newFrame(), other = newFrame();
  od_debug_memory_drain(gCtx);  // earlier tests' sessions are freed on gCtx's queue
  const uint64_t dev0 = (uint64_t)gDevice.currentAllocatedSize;
  CHECK(render(gNoisy, p, first, nullptr, a), "render on A failed");
  int entries = 0, pooled = 0;
  uint64_t bytes = 0, poolBytes = 0, held = 0;
  od_debug_cache_info(a, &entries, &bytes, &pooled, &poolBytes, nullptr);
  CHECK(entries == 5 && bytes > 0, "A should hold its window: %d frames, %llu bytes", entries, (unsigned long long)bytes);
  // B renders straight away: A is not idle yet and keeps its cache.
  CHECK(render(gNoisy, p, other, nullptr, b), "render on B failed");
  od_debug_cache_info(a, &entries, nullptr, nullptr, nullptr, nullptr);
  CHECK(entries == 5, "A was trimmed while still in use: %d frames", entries);
  usleep(350 * 1000);
  CHECK(render(gNoisy, p, other, nullptr, b), "second render on B failed");
  od_debug_memory_drain(ctx);
  od_debug_cache_info(a, &entries, &bytes, nullptr, nullptr, nullptr);
  uint64_t bytesB = 0;
  od_debug_cache_info(b, nullptr, &bytesB, &pooled, &poolBytes, nullptr);
  od_debug_memory_info(ctx, &held, nullptr, nullptr, nullptr);
  printf("  idle session while another one renders: %d cached frames; the context holds %.1f MB (B's cache %.1f MB, "
         "pool %d textures %.1f MB)\n",
         entries, held / 1048576.0, bytesB / 1048576.0, pooled, poolBytes / 1048576.0);
  CHECK(entries == 0 && bytes == 0, "idle session kept its cache: %d frames (%llu bytes)", entries,
        (unsigned long long)bytes);
  CHECK(held == bytesB + poolBytes, "the context holds %llu bytes beyond B's cache and the pool",
        (unsigned long long)(held - bytesB - poolBytes));
  CHECK(render(gNoisy, p, again, nullptr, a), "render on A after the trim failed");
  CHECK(memcmp(first.px(), again.px(), (size_t)W * H * 16) == 0, "A renders a different picture after being trimmed");
  // Nothing renders any more: the session used last (A) goes too, after its
  // longer deadline, and the pool with it.
  usleep(1500 * 1000);
  od_debug_memory_drain(ctx);
  int entriesB = 0;
  od_debug_cache_info(a, &entries, &bytes, &pooled, &poolBytes, nullptr);
  od_debug_cache_info(b, &entriesB, nullptr, nullptr, nullptr, nullptr);
  od_debug_memory_info(ctx, &held, nullptr, nullptr, nullptr);
  const double devLeft = ((double)gDevice.currentAllocatedSize - (double)dev0) / 1048576.0;
  printf("  nothing rendering: %d + %d cached frames, %d pooled textures, context %.1f MB, device %+.1f MB\n", entries,
         entriesB, pooled, held / 1048576.0, devLeft);
  CHECK(entries == 0 && entriesB == 0 && pooled == 0 && held == 0,
        "with nothing rendering, memory stayed: %d + %d frames, %d pooled textures, %llu bytes", entries, entriesB,
        pooled, (unsigned long long)held);
  CHECK(fabs(devLeft) < 1.0, "the device still holds %+.1f MB after the trim", devLeft);
  CHECK(render(gNoisy, p, again, nullptr, a), "render on A after the second trim failed");
  CHECK(memcmp(first.px(), again.px(), (size_t)W * H * 16) == 0, "A renders a different picture after the last trim");
  od_session_destroy(a);
  od_session_destroy(b);
  od_context_destroy(ctx);
}

// One effect instance sees two frame sizes (playback at 1/2 resolution, then
// the paused frame at full resolution): each size is denoised with its own
// noise profile from its first frame on, also when estimates of the other size
// are still in flight, and going back does not start the first size again.
static void testSizeChange() {
  gTest = "size_change";
  const uint32_t w2 = W / 2, h2 = H / 2;
  // Half-size frames: the 2x2 box average of the full-size ones, so white
  // noise drops to half.
  std::vector<id<MTLBuffer>> half;
  for (int f = 0; f < kFrames; ++f) {
    id<MTLBuffer> hb = [gDevice newBufferWithLength:(NSUInteger)w2 * h2 * 16 options:MTLResourceStorageModeShared];
    const float* s = gNoisy[(size_t)f].px();
    float* d = (float*)hb.contents;
    for (uint32_t y = 0; y < h2; ++y)
      for (uint32_t x = 0; x < w2; ++x)
        for (int c = 0; c < 4; ++c) {
          const size_t i0 = ((size_t)(2 * y) * W + 2 * x) * 4 + c, i1 = i0 + 4, i2 = i0 + (size_t)W * 4, i3 = i2 + 4;
          d[((size_t)y * w2 + x) * 4 + c] = 0.25f * (s[i0] + s[i1] + s[i2] + s[i3]);
        }
    half.push_back(hb);
  }
  id<MTLBuffer> outHalf = [gDevice newBufferWithLength:(NSUInteger)w2 * h2 * 16 options:MTLResourceStorageModeShared];
  Frame outFull = newFrame();
  // Frame 3 with its radius-2 window, keys keyBase + frame.
  auto renderAt = [&](ODSession* s, bool full, int64_t keyBase, const ODParams& p, ODStats* st) {
    ODFrame frames[5];
    const int offsets[5] = {0, -1, 1, -2, 2};
    for (int k = 0; k < 5; ++k) {
      const int f = 3 + offsets[k];
      if (full) {
        frames[k] = gNoisy[(size_t)f].od(keyBase + f);
      } else {
        frames[k].mtlBuffer = (__bridge void*)half[(size_t)f];
        frames[k].width = w2;
        frames[k].height = h2;
        frames[k].rowBytes = w2 * 16;
        frames[k].format = OD_PIXEL_BGRA32F;
        frames[k].frameKey = keyBase + f;
      }
    }
    ODFrame o = full ? outFull.od(OD_NO_KEY) : frames[0];
    if (!full) o.mtlBuffer = (__bridge void*)outHalf;
    o.frameKey = OD_NO_KEY;
    const int rc = od_render(s, frames, offsets, 5, &p, &o, st);
    if (rc != 0) fprintf(stderr, "od_render: %s\n", od_last_error(s));
    return rc == 0;
  };
  for (int wait = 1; wait >= 0; --wait) {
    ODParams p = od_params_default();
    p.quality = 0;
    p.lumaStrength = 1;
    p.waitUntilCompleted = wait;
    ODSession* s = od_session_create(gCtx);
    ODSession* fresh = od_session_create(gCtx);
    ODStats st = {}, sf = {};
    // Playback at half size; a new key each time, so each render measures.
    for (int r = 0; r < 12; ++r) CHECK(renderAt(s, false, 1000 * (r + 1), p, &st), "half-size render failed");
    const float halfSigma = st.profile.sigma[0][0][3];
    // Paused: one frame at full size, straight away (asynchronous: the last
    // half-size estimates are still in flight).
    CHECK(renderAt(s, true, 50000, p, &st), "full-size render failed");
    CHECK(renderAt(fresh, true, 50000, p, &sf), "fresh full-size render failed");
    const float first = st.profile.sigma[0][0][3], want = sf.profile.sigma[0][0][3];
    // More full-size frames on both, then everything folded in.
    for (int r = 0; r < 4; ++r) {
      CHECK(renderAt(s, true, 60000 + 1000 * r, p, &st), "full-size render failed");
      CHECK(renderAt(fresh, true, 60000 + 1000 * r, p, &sf), "fresh full-size render failed");
    }
    CHECK(od_debug_session_drain(s) == 0 && od_debug_session_drain(fresh) == 0, "asynchronous estimates never landed");
    ODParams pb = p;
    pb.waitUntilCompleted = 1;
    CHECK(renderAt(s, true, 63000, pb, &st), "full-size render failed");  // measured key: reports the slot
    CHECK(renderAt(fresh, true, 63000, pb, &sf), "fresh full-size render failed");
    const float later = st.profile.sigma[0][0][3], laterWant = sf.profile.sigma[0][0][3];
    // Back to playback: the half-size profile is still there.
    CHECK(renderAt(s, false, 12000, p, &st), "half-size render failed");
    const float back = st.profile.sigma[0][0][3];
    printf("  %s: Y band 0 sigma, half size %.5f; first full-size frame %.5f (fresh session %.5f); after 5 more "
           "%.5f (%.5f); back at half size %.5f\n",
           wait ? "blocking" : "asynchronous", halfSigma, first, want, later, laterWant, back);
    CHECK(halfSigma < 0.7f * want, "the two sizes do not differ enough for this test: %.5f and %.5f", halfSigma, want);
    CHECK(fabsf(first / want - 1) < 0.05f, "first full-size frame used sigma %.5f, a fresh session %.5f", first, want);
    CHECK(fabsf(later / laterWant - 1) < 0.03f, "full-size profile %.5f after the switch, fresh session %.5f", later,
          laterWant);
    CHECK(fabsf(back / halfSigma - 1) < 0.05f, "back at half size the profile is %.5f, it was %.5f", back, halfSigma);
    od_session_destroy(s);
    od_session_destroy(fresh);
  }
}

// Defaults: what od_params_default() promises in opendenoise.h, and a default
// render leaves 20 % of the luma noise amplitude at most plus what the filter
// could not remove.
static void testDefaults() {
  gTest = "defaults";
  const ODParams d = od_params_default();
  CHECK(d.lumaStrength == 0.8f && d.chromaStrength == 1.0f && d.temporalRadius == 2 && d.temporalStrength == 1.0f &&
            d.spatialStrength == 0.5f && d.sharpen == 0.0f && d.noiseLevel == 1.0f && d.quality == 1 &&
            d.view == OD_VIEW_OUTPUT && d.waitUntilCompleted == 0 && d.manualProfile == nullptr &&
            d.debugFlags == 0 && d.debugSkipStages == 0,
        "od_params_default() does not match opendenoise.h");
  Frame out = newFrame();
  CHECK(render(gNoisy, base(), out), "default render failed");
  const double in = rmsLuma(gNoisy[3], gClean, kFlat), e = rmsLuma(out, gClean, kFlat);
  const double inC = rmsChroma(gNoisy[3], gClean, kFlat), eC = rmsChroma(out, gClean, kFlat);
  printf("  defaults: flat luma %.2f of the input (amount 0.8 leaves at least 0.20), chroma %.2f\n", e / in, eC / inC);
  CHECK(e > 0.19 * in && e < 0.45 * in, "default luma result out of range: %.2f of the input", e / in);
  CHECK(eC < 0.3 * inC, "default chroma result: %.2f of the input", eC / inC);
}

// The current frame is ingested afresh and what the stages derived from its
// old cached copy is dropped: a render after the host changed the pixels under
// the same keys is exactly that of a fresh session holding the same frames
// (the old neighbours, the new current frame).
static void testStaleExact() {
  gTest = "stale_exact";
  const ODNoiseProfile prof = whiteProfile(kSigma);
  std::vector<Frame> other;
  for (int f = 0; f < kFrames; ++f) {
    other.push_back(newFrame());
    fill(other.back(), f, kSigma, 77);
    float* px = other.back().px();
    for (size_t i = 0; i < (size_t)W * H; ++i)
      for (int c = 0; c < 3; ++c) px[i * 4 + c] += 0.15f;
  }
  std::vector<Frame> mixed = gNoisy;
  mixed[3] = other[3];
  for (int q = 0; q <= 1; ++q) {
    ODParams p = base();
    p.manualProfile = &prof;
    p.lumaStrength = 1;
    p.quality = q;
    ODSession* s = od_session_create(gCtx);
    Frame a = newFrame(), b = newFrame();
    CHECK(render(gNoisy, p, a, nullptr, s), "first render failed");
    CHECK(render(other, p, a, nullptr, s), "second render failed");
    CHECK(render(mixed, p, b), "fresh render failed");
    size_t diff = 0;
    for (size_t i = 0; i < (size_t)W * H * 4; ++i) diff += memcmp(a.px() + i, b.px() + i, 4) != 0;
    CHECK(diff == 0, "quality %d: stale derived data, %zu values differ from a fresh session", q, diff);
    od_session_destroy(s);
  }
}

// ODParams.noFrameCache (what the plug-in passes): the same keys with new
// pixels behind them, a sub-sigma upstream change (+0.25 sigma), must give
// exactly the picture of a fresh session given the new frames: nothing is
// reused by key. With the cache, the session reuses the old neighbours by
// design (the key promises the content), so the picture differs (core-11).
static void testNoFrameCache() {
  gTest = "no_frame_cache";
  const ODNoiseProfile prof = whiteProfile(kSigma);
  std::vector<Frame> lifted;
  for (int f = 0; f < kFrames; ++f) {
    lifted.push_back(newFrame());
    memcpy(lifted.back().px(), gNoisy[(size_t)f].px(), (size_t)W * H * 16);
    float* px = lifted.back().px();
    for (size_t i = 0; i < (size_t)W * H; ++i)
      for (int c = 0; c < 3; ++c) px[i * 4 + c] += 0.25f * kSigma;
  }
  for (int cache = 0; cache <= 1; ++cache) {
    ODParams p = base();
    p.manualProfile = &prof;
    p.noFrameCache = cache ? 0 : 1;
    ODSession* s = od_session_create(gCtx);
    Frame a = newFrame(), b = newFrame();
    CHECK(render(gNoisy, p, a, nullptr, s), "first render failed");
    CHECK(render(lifted, p, a, nullptr, s), "render after the change failed");  // same keys
    CHECK(render(lifted, p, b), "fresh render failed");
    size_t diff = 0;
    for (size_t i = 0; i < (size_t)W * H * 4; ++i) diff += memcmp(a.px() + i, b.px() + i, 4) != 0;
    printf("  %s: %zu values differ from a fresh session after a +0.25 sigma change under the same keys\n",
           cache ? "frame cache" : "noFrameCache", diff);
    if (cache)
      CHECK(diff > 0, "with the frame cache the old neighbours were not reused (the test says nothing)");
    else
      CHECK(diff == 0, "noFrameCache: %zu values differ from a fresh session: a neighbour came from the cache", diff);
    od_session_destroy(s);
  }
}

// The noise view is 0.5 + 4 * (input - output) on every colour value.
static void testNoiseSign() {
  gTest = "noise_sign";
  ODParams p = base();
  const ODNoiseProfile prof = whiteProfile(kSigma);
  p.manualProfile = &prof;
  Frame normal = newFrame(), noise = newFrame();
  CHECK(render(gNoisy, p, normal), "render failed");
  p.view = OD_VIEW_NOISE;
  CHECK(render(gNoisy, p, noise), "render failed");
  size_t bad = 0;
  double corr = 0;
  for (size_t i = 0; i < (size_t)W * H; ++i)
    for (int c = 0; c < 3; ++c) {
      const double removed = (double)gNoisy[3].px()[i * 4 + c] - normal.px()[i * 4 + c];
      corr += (noise.px()[i * 4 + c] - 0.5) * removed;
      bad += fabs(noise.px()[i * 4 + c] - (0.5 + 4.0 * removed)) > 1e-4;
    }
  CHECK(corr > 0, "the noise view has the wrong sign");
  CHECK(bad == 0, "the noise view is not 0.5 + 4 * (in - out) on %zu values", bad);
}

// OD_CACHE_MB caps one session's cache below what the entry-age rule keeps
// (the last three windows), and the cap is what binds.
static void testCacheBudget() {
  gTest = "cache_budget";
  const ODNoiseProfile prof = whiteProfile(kSigma);
  ODParams p = base();
  p.manualProfile = &prof;
  Frame out = newFrame();
  const int N = 30, R = 2;
  auto run = [&](const char* mb, int* maxEntries, uint64_t* maxBytes) {
    if (mb) setenv("OD_CACHE_MB", mb, 1);
    ODSession* s = od_session_create(gCtx);
    unsetenv("OD_CACHE_MB");
    *maxEntries = 0;
    *maxBytes = 0;
    for (int i = 0; i < N; ++i) {
      ODFrame frames[OD_MAX_FRAMES];
      int offsets[OD_MAX_FRAMES];
      int n = 0;
      for (int d = 0; d <= 2 * R; ++d) {
        const int o = d == 0 ? 0 : (d % 2 ? (d + 1) / 2 : -d / 2), j = i + o;
        if (j < 0 || j >= N) continue;
        frames[n] = gNoisy[(size_t)(j % kFrames)].od(1000 + j);
        offsets[n++] = o;
      }
      ODFrame o = out.od(OD_NO_KEY);
      CHECK(od_render(s, frames, offsets, n, &p, &o, nullptr) == 0, "render %d failed", i);
      int e = 0;
      uint64_t b = 0;
      od_debug_cache_info(s, &e, &b, nullptr, nullptr, nullptr);
      *maxEntries = std::max(*maxEntries, e);
      *maxBytes = std::max(*maxBytes, b);
    }
    od_session_destroy(s);
  };
  int freeEntries = 0, capEntries = 0;
  uint64_t freeBytes = 0, capBytes = 0;
  run(nullptr, &freeEntries, &freeBytes);
  const uint64_t perEntry = freeBytes / (uint64_t)std::max(1, freeEntries);
  // Room for the window (5 entries) and a half, not for the 7 the age rule keeps.
  const uint64_t capMB = (perEntry * 11 / 2 + (1u << 20) - 1) >> 20;
  char mb[16];
  snprintf(mb, sizeof(mb), "%llu", (unsigned long long)capMB);
  run(mb, &capEntries, &capBytes);
  printf("  cache without a cap: %d entries %.1f MB; OD_CACHE_MB=%s: %d entries %.1f MB\n", freeEntries,
         freeBytes / 1048576.0, mb, capEntries, capBytes / 1048576.0);
  CHECK(freeEntries == 2 * R + 3, "the age rule kept %d entries, expected %d", freeEntries, 2 * R + 3);
  CHECK(capBytes <= (capMB << 20), "the cache grew to %.1f MB with OD_CACHE_MB=%s", capBytes / 1048576.0, mb);
  CHECK(capEntries == 2 * R + 1, "with the cap the cache held %d entries, expected the window (%d)", capEntries,
        2 * R + 1);
}

// The context's pool gives textures of a shape the renders no longer use
// back after the idle period, while renders of another shape carry on.
static void testPoolTrim() {
  gTest = "pool_trim";
  const ODNoiseProfile prof = whiteProfile(kSigma);
  ODParams p = base();
  p.manualProfile = &prof;
  // 304x200: no pyramid level coincides with one of 640x368.
  const uint32_t w = 304, h = 200;
  std::vector<id<MTLBuffer>> small;
  for (int f = 0; f < 6; ++f) {
    id<MTLBuffer> b = [gDevice newBufferWithLength:(NSUInteger)w * h * 16 options:MTLResourceStorageModeShared];
    float* px = (float*)b.contents;
    for (uint32_t i = 0; i < w * h; ++i) {
      for (int c = 0; c < 3; ++c) px[i * 4 + c] = 0.4f + kSigma * gauss(i * 31u + (uint32_t)c * 7u + (uint32_t)f * 1000003u);
      px[i * 4 + 3] = 1;
    }
    small.push_back(b);
  }
  const int offs[5] = {0, -1, 1, -2, 2};
  auto renderSmall = [&](ODSession* s) {
    ODFrame frames[5];
    for (int k = 0; k < 5; ++k) {
      frames[k].mtlBuffer = (__bridge void*)small[(size_t)k];
      frames[k].width = w;
      frames[k].height = h;
      frames[k].rowBytes = w * 16;
      frames[k].format = OD_PIXEL_BGRA32F;
      frames[k].frameKey = OD_NO_KEY;
    }
    ODFrame o = frames[0];
    o.mtlBuffer = (__bridge void*)small[5];
    return od_render(s, frames, offs, 5, &p, &o, nullptr) == 0;
  };
  Frame bigOut = newFrame();
  auto renderBig = [&](ODSession* s) {
    ODFrame frames[5];
    for (int k = 0; k < 5; ++k) frames[k] = gNoisy[(size_t)(3 + offs[k])].od(OD_NO_KEY);
    ODFrame o = bigOut.od(OD_NO_KEY);
    return od_render(s, frames, offs, 5, &p, &o, nullptr) == 0;
  };
  setenv("OD_IDLE_TRIM_MS", "200", 1);
  char err[512] = {0};
  ODContext* c1 = od_context_create((__bridge void*)gDevice, nullptr, err, sizeof(err));
  ODContext* c2 = od_context_create((__bridge void*)gDevice, nullptr, err, sizeof(err));
  unsetenv("OD_IDLE_TRIM_MS");
  if (!c1 || !c2) {
    CHECK(false, "context: %s", err);
    return;
  }
  ODSession* a = od_session_create(c1);
  ODSession* b = od_session_create(c2);
  uint64_t big = 0, after = 0, fresh = 0;
  CHECK(renderBig(a), "big render failed");
  od_debug_memory_info(c1, nullptr, nullptr, nullptr, &big);
  // Small renders for 0.6 s on both: c1's big textures go unused meanwhile.
  const double t0 = CFAbsoluteTimeGetCurrent();
  while (CFAbsoluteTimeGetCurrent() - t0 < 0.6) {
    CHECK(renderSmall(a) && renderSmall(b), "small render failed");
    usleep(20000);
  }
  od_debug_memory_drain(c1);
  od_debug_memory_drain(c2);
  od_debug_memory_info(c1, nullptr, nullptr, nullptr, &after);
  od_debug_memory_info(c2, nullptr, nullptr, nullptr, &fresh);
  printf("  idle pool: %.1f MB after the big render, %.1f MB after 0.6 s of small renders (a context that only did "
         "small renders: %.1f MB)\n",
         big / 1048576.0, after / 1048576.0, fresh / 1048576.0);
  CHECK(big > 2 * fresh, "sizes too close for the test to mean anything");
  CHECK(after == fresh, "the pool kept textures of the big shape: %llu bytes against %llu", (unsigned long long)after,
        (unsigned long long)fresh);
  od_session_destroy(a);
  od_session_destroy(b);
  od_context_destroy(c1);
  od_context_destroy(c2);
}

// The asynchronous profile path (what Premiere uses): estimates of later
// frames reach the session, the result matches blocking renders, and
// od_session_reset really empties the profile.
static void testAsyncProfile() {
  gTest = "async_profile";
  // Three noise realisations of the scene, rendered under distinct keys.
  std::vector<std::vector<Frame>> sets(3);
  sets[0] = gNoisy;
  for (int k = 1; k < 3; ++k)
    for (int f = 0; f < kFrames; ++f) {
      sets[(size_t)k].push_back(newFrame());
      fill(sets[(size_t)k].back(), f, kSigma, 10 + (uint32_t)k);
    }
  Frame out = newFrame();
  auto renderKeyed = [&](ODSession* s, const std::vector<Frame>& src, int64_t keyBase, const ODParams& p, ODStats* st) {
    ODFrame frames[5];
    const int offsets[5] = {0, -1, 1, -2, 2};
    for (int k = 0; k < 5; ++k) frames[k] = src[(size_t)(3 + offsets[k])].od(keyBase + 3 + offsets[k]);
    ODFrame o = out.od(OD_NO_KEY);
    return od_render(s, frames, offsets, 5, &p, &o, st) == 0;
  };
  ODParams pa = base();
  pa.waitUntilCompleted = 0;
  ODParams pb = base();
  ODSession* as = od_session_create(gCtx);
  ODSession* bs = od_session_create(gCtx);
  ODStats st = {};
  float cold = 0;
  for (int r = 0; r < 9; ++r) {
    CHECK(renderKeyed(as, sets[(size_t)(r % 3)], 100 * r, pa, &st), "asynchronous render failed");
    if (r == 0) cold = st.profile.sigma[0][0][3];
    CHECK(renderKeyed(bs, sets[(size_t)(r % 3)], 100 * r, pb, nullptr), "blocking render failed");
  }
  CHECK(od_debug_session_drain(as) == 0, "asynchronous estimates never landed");
  ODStats sa = {}, sb = {};
  CHECK(renderKeyed(as, sets[2], 800, pb, &sa) && renderKeyed(bs, sets[2], 800, pb, &sb), "render failed");  // measured keys
  const float fa = sa.profile.sigma[0][0][3], fb = sb.profile.sigma[0][0][3];
  printf("  asynchronous: cold estimate %.6f, after 9 frames %.6f; blocking %.6f\n", cold, fa, fb);
  CHECK(fa != cold, "the asynchronous estimates never reached the profile (still the cold one, %.6f)", cold);
  CHECK(fabsf(fa / fb - 1) < 0.01f, "asynchronous profile %.6f, blocking %.6f", fa, fb);
  // Reset, then a clip with 2.5 times the noise: its own profile at once.
  std::vector<Frame> loud;
  for (int f = 0; f < kFrames; ++f) {
    loud.push_back(newFrame());
    fill(loud.back(), f, 2.5f * kSigma, 20);
  }
  od_session_reset(as);
  CHECK(renderKeyed(as, loud, 5000, pa, &sa), "render after reset failed");
  ODSession* fresh = od_session_create(gCtx);
  CHECK(renderKeyed(fresh, loud, 5000, pb, &sb), "fresh render failed");
  printf("  after reset, louder clip: %.6f (fresh session %.6f, before the reset %.6f)\n", sa.profile.sigma[0][0][3],
         sb.profile.sigma[0][0][3], fa);
  CHECK(fabsf(sa.profile.sigma[0][0][3] / sb.profile.sigma[0][0][3] - 1) < 0.01f,
        "the first render after od_session_reset used %.6f, a fresh session %.6f", sa.profile.sigma[0][0][3],
        sb.profile.sigma[0][0][3]);
  od_session_destroy(as);
  od_session_destroy(bs);
  od_session_destroy(fresh);
}

// ---- new footage under one session ---------------------- Shot A (the scene, sigma 0.01)
// cut to shot B (the scene mirrored, so the temporal stage sees unrelated content, sigma
// 0.035) in one session, played in order with radius 2. The first frame of B must be
// rendered with B's noise figures, not A's: the temporal stage sees the cut two frames ahead
// and the session measures that frame decisively. A jump (scrubbing to frames the session
// has not rendered) restarts the same way; a jump within the same footage must leave the
// profile alone.
static void fillShot(const Frame& f, int frame, bool shotB, uint32_t seed) {
  fill(f, frame, shotB ? 0.035f : 0.01f, seed);
  if (!shotB) return;
  // Mirror left-right: the flat patches stay, everything else moves.
  float* p = f.px();
  for (uint32_t y = 0; y < H; ++y)
    for (uint32_t x = 0; x < W / 2; ++x)
      for (int c = 0; c < 4; ++c) std::swap(p[((size_t)y * W + x) * 4 + c], p[((size_t)y * W + (W - 1 - x)) * 4 + c]);
}

static void testCutRestart() {
  gTest = "cut_restart";
  const int kCut = 12, kTotal = 24;
  std::vector<Frame> seq;
  for (int f = 0; f < kTotal; ++f) {
    seq.push_back(newFrame());
    fillShot(seq.back(), f, f >= kCut, 300);
  }
  Frame out = newFrame();
  // Renders frame idx of seq with radius 2, neighbours limited to [lo, hi).
  auto renderAt = [&](ODSession* s, int idx, int lo, int hi, ODParams p, bool keyed, ODStats* st) {
    ODFrame frames[5];
    int offsets[5];
    int n = 0;
    for (int d : {0, -1, 1, -2, 2}) {
      const int j = idx + d;
      if (j < lo || j >= hi) continue;
      frames[n] = seq[(size_t)j].od(keyed ? 1000 + j : OD_NO_KEY);
      offsets[n++] = d;
    }
    ODFrame o = out.od(OD_NO_KEY);
    const int rc = od_render(s, frames, offsets, n, &p, &o, st);
    if (rc != 0) fprintf(stderr, "od_render: %s\n", od_last_error(s));
    return rc == 0;
  };
  auto y0 = [](const ODStats& st) { return st.profile.sigma[0][0][3]; };

  // Shot B's own profile: a session that only ever saw B.
  ODStats st = {};
  float refB = 0, refA = 0;
  {
    ODSession* s = od_session_create(gCtx);
    for (int f = kCut; f < kCut + 6; ++f) CHECK(renderAt(s, f, kCut, kTotal, base(), true, &st), "B alone failed");
    refB = y0(st);
    od_session_destroy(s);
    s = od_session_create(gCtx);
    for (int f = 0; f < 6; ++f) CHECK(renderAt(s, f, 0, kCut, base(), true, &st), "A alone failed");
    refA = y0(st);
    od_session_destroy(s);
  }
  printf("  shot A Y band 0 %.5f, shot B %.5f\n", refA, refB);
  CHECK(refB > 2.5f * refA, "the two shots are not far apart (%.5f, %.5f): the test says nothing", refA, refB);

  struct Mode {
    const char* name;
    bool keyed, async, noCache;
    int quality;
  } modes[] = {
      {"blocking, keys", true, false, false, 1},
      {"blocking, keys, playback quality", true, false, false, 0},
      {"asynchronous paced, keys", true, true, false, 1},
      {"blocking, keys without frame cache", true, false, true, 1},
      {"blocking, no keys", false, false, false, 1},
      {"asynchronous paced, no keys", false, true, false, 1},
  };
  for (const Mode& m : modes) {
    ODSession* s = od_session_create(gCtx);
    ODParams p = base();
    p.quality = m.quality;
    p.waitUntilCompleted = m.async ? 0 : 1;
    p.noFrameCache = m.noCache ? 1 : 0;
    float firstB = 0, secondB = 0;
    uint64_t waitsInA = 0;
    for (int f = 0; f < kCut + 2; ++f) {
      CHECK(renderAt(s, f, 0, kTotal, p, m.keyed, &st), "%s: render %d failed", m.name, f);
      if (m.async) od_debug_session_drain(s);  // paced: every render done before the next
      if (f == kCut - 1) waitsInA = od_debug_profile_waits(s);
      if (f == kCut) firstB = y0(st);
      if (f == kCut + 1) secondB = y0(st);
    }
    // One blocking estimate to start, one at the cut: playback must not stall.
    CHECK(waitsInA == 1, "%s: %llu renders of shot A blocked for an estimate (expected the first only)", m.name,
          (unsigned long long)waitsInA);
    CHECK(od_debug_profile_waits(s) == 2, "%s: %llu blocking estimates over both shots (expected 2)", m.name,
          (unsigned long long)od_debug_profile_waits(s));
    printf("  %-38s first frame of B %.5f, second %.5f (B alone %.5f)\n", m.name, firstB, secondB, refB);
    // Without keys the session learns from the frame before the cut that the
    // next frame is another shot (forward playback).
    CHECK(fabsf(firstB / refB - 1) < 0.1f, "%s: the first frame of shot B used Y sigma %.5f, shot B's own is %.5f (A %.5f)",
          m.name, firstB, refB, refA);
    CHECK(fabsf(secondB / refB - 1) < 0.1f, "%s: the second frame of shot B used Y sigma %.5f, shot B's own is %.5f",
          m.name, secondB, refB);
    od_session_destroy(s);
  }

  // A jump: play A, then render B frames the session has never seen.
  {
    ODSession* s = od_session_create(gCtx);
    for (int f = 0; f < 8; ++f) CHECK(renderAt(s, f, 0, kCut, base(), true, &st), "A failed");
    const float settled = y0(st);
    CHECK(renderAt(s, kCut + 6, kCut, kTotal, base(), true, &st), "jump to B failed");
    printf("  jump to shot B: %.5f (B alone %.5f, A before the jump %.5f)\n", y0(st), refB, settled);
    CHECK(fabsf(y0(st) / refB - 1) < 0.1f, "a jump to shot B used Y sigma %.5f, shot B's own is %.5f", y0(st), refB);
    od_session_destroy(s);
  }
  // Footage that never gives an estimate (black) blocks for its first
  // kColdWaits (4) renders, not on every render.
  {
    ODSession* s = od_session_create(gCtx);
    Frame black = newFrame();
    memset(black.px(), 0, (size_t)W * H * 16);
    ODParams p = base();
    p.waitUntilCompleted = 0;
    for (int f = 0; f < 6; ++f) {
      ODFrame fr = black.od(7000 + f), o = out.od(OD_NO_KEY);
      int off = 0;
      CHECK(od_render(s, &fr, &off, 1, &p, &o, nullptr) == 0, "black render failed");
    }
    od_debug_session_drain(s);
    printf("  black footage: %llu of 6 renders blocked for an estimate\n",
           (unsigned long long)od_debug_profile_waits(s));
    CHECK(od_debug_profile_waits(s) == 4, "black footage: %llu of 6 renders blocked for an estimate (expected 4)",
          (unsigned long long)od_debug_profile_waits(s));
    od_session_destroy(s);
  }
  // A jump within the same footage: the profile stays.
  {
    ODSession* s = od_session_create(gCtx);
    std::vector<Frame> more;
    for (int f = 0; f < kCut; ++f) CHECK(renderAt(s, f, 0, kCut, base(), true, &st), "A failed");
    const float settled = y0(st);
    for (int f = 0; f < 5; ++f) {
      more.push_back(newFrame());
      fillShot(more.back(), f, false, 900);  // same shot, other noise realisation
    }
    ODFrame frames[5];
    const int offsets[5] = {0, -1, 1, -2, 2};
    for (int k = 0; k < 5; ++k) frames[k] = more[(size_t)(2 + offsets[k])].od(5000 + offsets[k]);
    ODFrame o = out.od(OD_NO_KEY);
    ODParams p = base();
    CHECK(od_render(s, frames, offsets, 5, &p, &o, &st) == 0, "jump within A failed");
    printf("  jump within shot A: %.5f (before %.5f)\n", y0(st), settled);
    CHECK(fabsf(y0(st) / settled - 1) < 0.1f, "a jump within the same footage moved the profile from %.5f to %.5f",
          settled, y0(st));
    od_session_destroy(s);
  }
}


// ---- the profile's cold start, steadiness and realtime renders ----

// A frame the profiler cannot trust: isotropic texture everywhere, at least
// twice the noise, with an amplitude that varies across the frame, so the
// tile energies form a continuum with no noise floor (fitConfidence ~ 0).
static void fillTextured(const Frame& f, uint32_t seed) {
  float* p = f.px();
  for (uint32_t y = 0; y < H; ++y)
    for (uint32_t x = 0; x < W; ++x) {
      const float a = 0.04f + 0.16f * (float)x / (float)W + 0.06f * (float)y / (float)H;
      const float t = sinf(0.55f * (float)x) * sinf(0.55f * (float)y) + 0.6f * sinf(0.21f * (float)x + 0.17f * (float)y) +
                      0.5f * sinf(1.3f * (float)x - 0.9f * (float)y);
      const float v = 0.45f + a * t;
      float* o = p + ((size_t)y * W + x) * 4;
      for (int c = 0; c < 3; ++c) {
        const uint32_t s = (x * 73856093u) ^ (y * 19349663u) ^ ((uint32_t)c * 83492791u) ^ (seed * 2654435761u);
        o[2 - c] = v + kSigma * gauss(s);
      }
      o[3] = 1.0f;
    }
}

// The first frame of a clip gives only an unusable estimate (confidence 0),
// the second a good one. Before the fix the unusable one became the
// profile, scaled down, and stayed in use until asynchronous estimates
// arrived (17 frames of an export with almost no luma filtering on a real
// night clip). Now the size is not done with its cold start until it has a
// usable estimate: the second render blocks again and uses the good one,
// and when the first render has neighbours they are measured at once.
static void testColdLowConfidence() {
  gTest = "cold_low_confidence";
  Frame bad = newFrame();
  fillTextured(bad, 77);
  ODNoiseProfile pb = {};
  {
    ODSession* s = od_session_create(gCtx);
    ODFrame f = bad.od(OD_NO_KEY);
    const int rc = od_profile_frame(s, &f, &pb);
    od_session_destroy(s);
    printf("  textured frame alone: %s, confidence %.2f\n", rc == 0 ? "estimate" : "no estimate", pb.confidence);
    CHECK(rc != 0 || pb.confidence < 0.3f, "the textured frame gave a usable estimate (confidence %.2f): the test "
          "says nothing", pb.confidence);
  }
  // The good profile: the second render's window in a fresh session.
  Frame out = newFrame();
  ODStats st = {};
  float ref[OD_LEVELS] = {};
  ODFrame fr[3] = {gNoisy[3].od(701), bad.od(700), gNoisy[4].od(702)};
  const int offs[3] = {0, -1, 1};
  {
    ODSession* s = od_session_create(gCtx);
    ODParams p = base();
    ODFrame o = out.od(OD_NO_KEY);
    CHECK(od_render(s, fr, offs, 3, &p, &o, &st) == 0, "reference render failed");
    for (int l = 0; l < OD_LEVELS; ++l) ref[l] = st.profile.sigma[0][0][l];
    printf("  second window alone: confidence %.2f\n", st.profile.confidence);
    od_session_destroy(s);
  }
  struct Mode { const char* name; int wait; } modes[] = {{"blocking", 1}, {"asynchronous", 0}};
  for (const Mode& m : modes) {
    ODSession* s = od_session_create(gCtx);
    ODParams p = base();
    p.waitUntilCompleted = m.wait;
    // Render 0: the unusable frame alone. Render 1: the scene, with the
    // first frame as its previous neighbour (so it is no jump).
    ODFrame f0 = bad.od(700), o = out.od(OD_NO_KEY);
    int off0 = 0;
    CHECK(od_render(s, &f0, &off0, 1, &p, &o, &st) == 0, "%s: render 0 failed", m.name);
    const float first = st.profile.sigma[0][0][3];
    CHECK(od_render(s, fr, offs, 3, &p, &o, &st) == 0, "%s: render 1 failed", m.name);
    float worst = 0;
    int worstL = 0;
    for (int l = 0; l < OD_LEVELS; ++l)
      if (fabsf(st.profile.sigma[0][0][l] / ref[l] - 1) > worst) {
        worst = fabsf(st.profile.sigma[0][0][l] / ref[l] - 1);
        worstL = l;
      }
    if (getenv("OD_TEST_VERBOSE"))
      for (int l = 0; l < OD_LEVELS; ++l) printf("    knot %d: %.5f (scene %.5f)\n", l, st.profile.sigma[0][0][l], ref[l]);
    printf("  %-12s render 0 used Y b0 %.5f (confidence %s), render 1 %.5f (scene alone %.5f), worst knot %.1f %%, "
           "%llu blocking\n", m.name, first, "low", st.profile.sigma[0][0][3], ref[3], 100 * worst + 0 * worstL,
           (unsigned long long)od_debug_profile_waits(s));
    CHECK(worst < 0.03f, "%s: the second render's profile is %.1f %% off the scene's own at some knot (first render "
          "%.5f, second %.5f, scene %.5f)", m.name, 100 * worst, first, st.profile.sigma[0][0][3], ref[3]);
    CHECK(od_debug_profile_waits(s) == 2, "%s: %llu renders blocked (expected 2: the unusable estimate, then the good "
          "one)", m.name, (unsigned long long)od_debug_profile_waits(s));
    od_debug_session_drain(s);
    od_session_destroy(s);
  }
  // The unusable frame first in a window whose neighbours are good (the
  // first frame of an export): the neighbours are measured in the same
  // render, so even that frame uses a usable profile.
  ODNoiseProfile good = {};
  {
    ODSession* s = od_session_create(gCtx);
    ODFrame f = gNoisy[3].od(OD_NO_KEY);
    CHECK(od_profile_frame(s, &f, &good) == 0, "od_profile_frame on the scene failed");
    od_session_destroy(s);
  }
  for (const Mode& m : modes) {
    ODSession* s = od_session_create(gCtx);
    ODParams p = base();
    p.waitUntilCompleted = m.wait;
    ODFrame w[3] = {bad.od(800), gNoisy[3].od(801), gNoisy[4].od(802)}, o = out.od(OD_NO_KEY);
    const int wo[3] = {0, 1, 2};
    CHECK(od_render(s, w, wo, 3, &p, &o, &st) == 0, "%s: first render with neighbours failed", m.name);
    printf("  %-12s unusable first frame with good neighbours: confidence %.2f, Y b0 %.5f (neighbour alone %.5f), "
           "%llu blocking\n", m.name, st.profile.confidence, st.profile.sigma[0][0][3], good.sigma[0][0][3],
           (unsigned long long)od_debug_profile_waits(s));
    CHECK(st.profile.confidence >= 0.3f && fabsf(st.profile.sigma[0][0][3] / good.sigma[0][0][3] - 1) < 0.03f,
          "%s: the first frame used confidence %.2f, Y b0 %.5f (its neighbour alone %.5f)", m.name,
          st.profile.confidence, st.profile.sigma[0][0][3], good.sigma[0][0][3]);
    od_debug_session_drain(s);
    od_session_destroy(s);
  }
}

// Version 2 returns a useful first picture after scanning its supplied
// window even for async/realtime callers. A parked frame does not rescan.
// Uses the existing seven-frame fixture plus two output frames (< 8 MiB).
static void testFirstWindowV2() {
  gTest = "first_window_v2";
  Frame out = newFrame(), first = newFrame();
  for (int model : {0,1}) for (int wait : {0,1}) for (int realtime : {0,1})
    for (int count : {1,3,5}) {
      if (model == 0 && realtime) continue; // legacy's bounded wait is tested separately
      ODSession* s = od_session_create(gCtx);
      ODParams p = base(); p.temporalNoiseModel=model; p.chromaRecovery=model;
      p.waitUntilCompleted=wait; p.realtime=realtime; p.verifyFrameCache=1;
      const int offsets[] = {0,-1,1,-2,2};
      ODFrame f[5]; for(int k=0;k<count;++k) f[k]=gNoisy[3+offsets[k]].od(100+offsets[k]);
      ODFrame o=out.od(OD_NO_KEY); ODStats st{};
      CHECK(od_render(s,f,offsets,count,&p,&o,&st)==0,"first render model%d wait%d realtime%d",model,wait,realtime);
      CHECK(st.profile.valid && st.profile.confidence>=0.3f,"first render has a usable estimate");
      CHECK(od_debug_session_drain(s)==0,"first render completes");
      memcpy(first.px(),out.px(),W*H*16);
      const double residual=rmsLuma(first,gClean,kFlat), source=rmsLuma(gNoisy[3],gClean,kFlat);
      CHECK(residual<source*.6,"first frame is already cleaned: %.6f vs input %.6f",residual,source);
      const auto waits=od_debug_profile_waits(s);
      CHECK(waits==uint64_t(model?count:1),"first scan measures available window only in V2 (%llu)",(unsigned long long)waits);
      for(int n=1;n<=20;++n) CHECK(od_render(s,f,offsets,count,&p,&o,&st)==0,"parked render %d",n);
      CHECK(od_debug_session_drain(s)==0,"parked renders complete");
      CHECK(od_debug_profile_waits(s)==waits,"parked frame never repeats initial scan");
      CHECK(rmsLuma(first,out,kFlat)<1e-6,"first and settled parked picture agree");
      od_session_destroy(s);
    }
}

// Steady footage whose highest measured luminance knot comes and goes: a
// patch at knot 3 (Y 3/7, less noise than below it) has enough tiles for the
// knot in frames of type A and a few too few in type B (a 2 % change of the
// frame: no cut). Without it, knots 3..7 of a B estimate hold copies of knot
// 2. Before the fix two B estimates in a row (as on a real night clip)
// counted as new footage, erased the ring and stepped the profile; now only
// the knots both measured are compared, and a knot is aggregated over the
// estimates that measured it.
static void fillKnots(const Frame& f, bool bigPatch, uint32_t seed) {
  float* p = f.px();
  const uint32_t pw = bigPatch ? 176 : 128, ph = bigPatch ? 96 : 64;
  for (uint32_t y = 0; y < H; ++y)
    for (uint32_t x = 0; x < W; ++x) {
      float v = 2.0f / 7.0f, sig = 0.02f;
      if (y >= 2 * H / 3) v = 1.0f / 7.0f;
      if (x >= 432 && x < 432 + pw && y >= 32 && y < 32 + ph) {
        v = 3.0f / 7.0f;
        sig = 0.008f;
      }
      float* o = p + ((size_t)y * W + x) * 4;
      for (int c = 0; c < 3; ++c) {
        const uint32_t s = (x * 73856093u) ^ (y * 19349663u) ^ ((uint32_t)c * 83492791u) ^ (seed * 2654435761u);
        o[2 - c] = v + sig * gauss(s);
      }
      o[3] = 1.0f;
    }
}

static void testKnotAlternation() {
  gTest = "knot_alternation";
  const int kReal = 6;  // noise realisations per type
  std::vector<Frame> A, B;
  for (int i = 0; i < kReal; ++i) {
    A.push_back(newFrame());
    fillKnots(A.back(), true, 500 + (uint32_t)i);
    B.push_back(newFrame());
    fillKnots(B.back(), false, 600 + (uint32_t)i);
  }
  Frame out = newFrame();
  // Frame t of a sequence: type A, except B at 40, 41, 60, 61, 80, 81.
  auto typeA = [](int t) { return !(t >= 40 && t % 20 < 2); };
  auto frameOf = [&](int t, bool alwaysA) -> const Frame& {
    return (alwaysA || typeA(t)) ? A[(size_t)(t % kReal)] : B[(size_t)(t % kReal)];
  };
  auto renderAt = [&](ODSession* s, int t, const ODParams& p, bool alwaysA, ODStats* st) {
    ODFrame fr[5];
    int offs[5], n = 0;
    for (int d : {0, -1, 1, -2, 2}) {
      if (t + d < 0) continue;
      fr[n] = frameOf(t + d, alwaysA).od(9000 + t + d);
      offs[n++] = d;
    }
    ODFrame o = out.od(OD_NO_KEY);
    return od_render(s, fr, offs, n, &p, &o, st) == 0;
  };
  ODStats st = {};
  float ref[2][OD_LEVELS] = {};
  {
    ODSession* s = od_session_create(gCtx);
    for (int t = 0; t < 30; ++t) CHECK(renderAt(s, t, base(), true, &st), "reference render failed");
    for (int b = 0; b < 2; ++b)
      for (int l = 0; l < OD_LEVELS; ++l) ref[b][l] = st.profile.sigma[0][b][l];
    printf("  type A alone: Y b0 at knots 1..3: %.5f %.5f %.5f (confidence %.2f)\n", ref[0][1], ref[0][2], ref[0][3],
           st.profile.confidence);
    od_session_destroy(s);
  }
  CHECK(ref[0][2] > 1.5f * ref[0][3], "knot 3 is not quieter than knot 2 (%.5f, %.5f): the test says nothing",
        ref[0][3], ref[0][2]);
  struct Mode { const char* name; int wait; } modes[] = {{"blocking", 1}, {"asynchronous paced", 0}};
  for (const Mode& m : modes) {
    ODSession* s = od_session_create(gCtx);
    ODParams p = base();
    p.waitUntilCompleted = m.wait;
    float worst[2] = {0, 0};
    int worstT = -1;
    for (int t = 0; t < 100; ++t) {
      CHECK(renderAt(s, t, p, false, &st), "%s: render %d failed", m.name, t);
      if (!m.wait) od_debug_session_drain(s);
      if (getenv("OD_TEST_VERBOSE"))
        printf("    t %d %c: b0 %.5f %.5f %.5f %.5f conf %.2f waits %llu\n", t, typeA(t) ? 'A' : 'B', st.profile.sigma[0][0][0],
               st.profile.sigma[0][0][1], st.profile.sigma[0][0][2], st.profile.sigma[0][0][3], st.profile.confidence,
               (unsigned long long)od_debug_profile_waits(s));
      if (t < 10) continue;
      for (int b = 0; b < 2; ++b)
        for (int l = 1; l <= 3; ++l) {
          const float e = fabsf(st.profile.sigma[0][b][l] / ref[b][l] - 1);
          if (e > worst[b]) {
            worst[b] = e;
            if (b == 0) worstT = t;
          }
        }
    }
    printf("  %-20s worst deviation from type A alone over frames 10..99, knots 1..3: band 0 %.1f %% (frame %d), "
           "band 1 %.1f %%\n", m.name, 100 * worst[0], worstT, 100 * worst[1]);
    CHECK(worst[0] < 0.05f && worst[1] < 0.05f, "%s: the profile moved by %.1f %% (band 0) / %.1f %% (band 1) on "
          "steady footage", m.name, 100 * worst[0], 100 * worst[1]);
    od_debug_session_drain(s);
    od_session_destroy(s);
  }
}

// Spatial-only renders (radius 0, or a host that delivered none of the
// neighbours at a clip's edge) carry only the current frame, whose key is new
// on every render: they must not count as jumps (each one used to block
// for a decisive estimate and restart the profile).
static void testSingleFrameNoJump() {
  gTest = "single_frame_no_jump";
  Frame out = newFrame();
  ODSession* s = od_session_create(gCtx);
  ODParams p = base();
  p.temporalRadius = 0;
  ODStats st = {};
  float first = 0, last = 0;
  for (int t = 0; t < 30; ++t) {
    ODFrame f = gNoisy[(size_t)(t % kFrames)].od(4000 + t), o = out.od(OD_NO_KEY);
    int off = 0;
    CHECK(od_render(s, &f, &off, 1, &p, &o, &st) == 0, "radius 0 render failed");
    if (t == 1) first = st.profile.sigma[0][0][3];
    last = st.profile.sigma[0][0][3];
  }
  printf("  radius 0, 30 renders: %llu blocking, Y b0 %.5f then %.5f\n", (unsigned long long)od_debug_profile_waits(s),
         first, last);
  CHECK(od_debug_profile_waits(s) == 1, "radius 0: %llu of 30 renders blocked (expected the first only)",
        (unsigned long long)od_debug_profile_waits(s));
  od_session_destroy(s);
  // A clip's edge: the first two frames without neighbours, then full windows.
  s = od_session_create(gCtx);
  p = base();
  p.waitUntilCompleted = 0;
  for (int t = 0; t < 8; ++t) {
    ODFrame fr[5];
    int offs[5], n = 0;
    for (int d : {0, -1, 1, -2, 2}) {
      if (t + d < 0 || (t < 2 && d != 0)) continue;
      fr[n] = gNoisy[(size_t)(t + d) % kFrames].od(4100 + t + d);
      offs[n++] = d;
    }
    ODFrame o = out.od(OD_NO_KEY);
    CHECK(od_render(s, fr, offs, n, &p, &o, nullptr) == 0, "edge render failed");
  }
  od_debug_session_drain(s);
  printf("  clip edge without neighbours on two frames: %llu blocking\n", (unsigned long long)od_debug_profile_waits(s));
  CHECK(od_debug_profile_waits(s) == 1, "clip edge: %llu renders blocked (expected the first only)",
        (unsigned long long)od_debug_profile_waits(s));
  od_session_destroy(s);
}

// Coarse chroma blotches that persist from frame to frame (in-camera noise
// reduction, codecs) are not averaged away by the temporal merge, although
// its count map says the frames were merged. The spatial stage measures the
// share of the chroma noise the merge left (spatial_chroma_persist) and lets
// the coarse chroma thresholds follow it. Here the same luma-neutral blotch
// field (cells of 12 px) sits on every frame on top of independent noise.
// Compared with the stage that trusts the count map (OD_SPATIAL_TUNE
// cpersist=0): the blotches in the flat third must be much reduced, and on
// independent noise alone (gNoisy) nothing may get worse.
static float blotch(uint32_t x, uint32_t y, uint32_t seed) {
  const float kCell = 12.0f;
  const float fx = (float)x / kCell, fy = (float)y / kCell;
  const uint32_t ix = (uint32_t)fx, iy = (uint32_t)fy;
  const float tx = fx - (float)ix, ty = fy - (float)iy;
  auto g = [&](uint32_t a, uint32_t b) { return gauss(a * 92821u ^ b * 68917u ^ seed * 40503u); };
  const float sx = tx * tx * (3 - 2 * tx), sy = ty * ty * (3 - 2 * ty);
  return (g(ix, iy) * (1 - sx) + g(ix + 1, iy) * sx) * (1 - sy) + (g(ix, iy + 1) * (1 - sx) + g(ix + 1, iy + 1) * sx) * sy;
}

static void addBlotches(const Frame& f, float amp) {
  float* p = f.px();
  for (uint32_t y = 0; y < H; ++y) {
    for (uint32_t x = 0; x < W; ++x) {
      const float cb = amp * blotch(x, y, 11), cr = amp * blotch(x, y, 12);
      const float dr = 1.5748f * cr, db = 1.8556f * cb, dg = -(0.2126f * dr + 0.0722f * db) / 0.7152f;
      float* o = p + ((size_t)y * W + x) * 4;
      o[0] += db;
      o[1] += dg;
      o[2] += dr;
    }
  }
}

// rms of the chroma (R - Y) error after a 9x9 box low-pass: the blotch scale.
static double rmsCoarseChroma(const Frame& a, const Frame& b, Region r) {
  std::vector<double> d((size_t)W * H), t((size_t)W * H);
  for (size_t i = 0; i < (size_t)W * H; ++i)
    d[i] = (a.px()[i * 4 + 2] - lumaOf(a.px() + i * 4)) - (b.px()[i * 4 + 2] - lumaOf(b.px() + i * 4));
  for (int pass = 0; pass < 2; ++pass) {
    for (uint32_t y = 0; y < H; ++y)
      for (uint32_t x = 0; x < W; ++x) {
        double s = 0;
        int n = 0;
        for (int k = -4; k <= 4; ++k) {
          const int xx = pass == 0 ? (int)x + k : (int)x, yy = pass == 0 ? (int)y : (int)y + k;
          if (xx < 0 || yy < 0 || xx >= (int)W || yy >= (int)H) continue;
          s += d[(size_t)yy * W + (size_t)xx];
          ++n;
        }
        t[(size_t)y * W + x] = s / n;
      }
    d.swap(t);
  }
  double s = 0;
  size_t n = 0;
  for (uint32_t y = 0; y < H; ++y)
    for (uint32_t x = 0; x < W; ++x) {
      if (!inRegion(r, x, y)) continue;
      s += d[(size_t)y * W + x] * d[(size_t)y * W + x];
      ++n;
    }
  return n ? sqrt(s / (double)n) : 0;
}

static void testPersistentChroma() {
  gTest = "persistent_chroma";
  std::vector<Frame> blotchy;
  for (int f = 0; f < kFrames; ++f) {
    blotchy.push_back(newFrame());
    fill(blotchy.back(), f, kSigma, 1);
    addBlotches(blotchy.back(), 0.012f);
  }
  ODParams p = base();
  Frame on = newFrame(), off = newFrame(), onI = newFrame(), offI = newFrame();
  CHECK(render(blotchy, p, on), "render failed");
  CHECK(render(gNoisy, p, onI), "render failed");
  setenv("OD_SPATIAL_TUNE", "cpersist=0", 1);
  CHECK(render(blotchy, p, off), "render failed");
  CHECK(render(gNoisy, p, offI), "render failed");
  unsetenv("OD_SPATIAL_TUNE");
  const double in = rmsCoarseChroma(blotchy[3], gClean, kFlat);
  const double eOn = rmsCoarseChroma(on, gClean, kFlat), eOff = rmsCoarseChroma(off, gClean, kFlat);
  printf("  persistent blotches, flat third, coarse chroma rms: input %.5f, count map only %.5f, measured %.5f\n", in,
         eOff, eOn);
  CHECK(eOn < 0.75 * eOff, "persistent blotches: %.5f left, %.5f with the count map alone", eOn, eOff);
  CHECK(eOn < 0.5 * in, "persistent blotches: %.5f of %.5f left", eOn, in);
  for (Region r : {kFlat, kTexture, kMoving}) {
    const double a = rmsChroma(onI, gClean, r), b = rmsChroma(offI, gClean, r);
    printf("  independent noise, region %d, chroma rms error: count map only %.5f, measured %.5f\n", (int)r, b, a);
    CHECK(a < 1.03 * b + 1e-5, "independent noise, region %d: chroma error %.5f against %.5f", (int)r, a, b);
  }
  const double tOn = rmsChroma(on, gClean, kTexture), tOff = rmsChroma(off, gClean, kTexture);
  printf("  persistent blotches, texture third, chroma rms error: count map only %.5f, measured %.5f\n", tOff, tOn);
  CHECK(tOn < 1.05 * tOff, "persistent blotches, texture: chroma error %.5f against %.5f", tOn, tOff);
}

// Realtime renders (the host playing) do not wait for the GPU work queued ahead of theirs
// (the queue has free command buffers here): here the queue is held up by a command buffer
// waiting on an event (as Premiere's own queued work would hold it), and a cold start and a
// jump must return in well under a frame interval. The same cold start without realtime
// blocks until the queue moves (an export: fine). Once the queue moves the estimates land
// and the profile is the footage's own. The host changes the pixels behind frame keys the
// session has already measured (a slip edit, an effect below Open Denoise, replaced media, a
// proxy toggle): with the checked frame cache (what the plug-in passes) the current frame's
// check sees new pixels under its key, and the profile follows the new footage instead of
// keeping the old one for every frame it had measured. Same keys, 3x the noise: within a few
// renders the profile is the new footage's, blocking and at playback quality, asynchronous.
static void testUpstreamChange() {
  gTest = "upstream_change";
  const int kN = 12;
  std::vector<Frame> a, b;
  for (int f = 0; f < kN; ++f) {
    a.push_back(newFrame());
    fill(a.back(), f, kSigma, 610);
    b.push_back(newFrame());
    fill(b.back(), f, 3 * kSigma, 620);
  }
  Frame out = newFrame();
  auto renderAt = [&](ODSession* s, const std::vector<Frame>& seq, int idx, const ODParams& p, ODStats* st) {
    ODFrame frames[5];
    int offsets[5], n = 0;
    for (int d : {0, -1, 1, -2, 2}) {
      const int j = idx + d;
      if (j < 0 || j >= kN) continue;
      frames[n] = seq[(size_t)j].od(5000 + j);
      offsets[n++] = d;
    }
    ODFrame o = out.od(OD_NO_KEY);
    const bool ok = od_render(s, frames, offsets, n, &p, &o, st) == 0;
    if (!p.waitUntilCompleted) od_debug_session_drain(s);
    return ok;
  };
  auto y0 = [](const ODStats& st) { return st.profile.sigma[0][0][3]; };
  struct Mode {
    const char* name;
    bool async;
    int quality;
  } modes[] = {{"blocking, quality 1", false, 1}, {"asynchronous playback, quality 0", true, 0}};
  for (const Mode& m : modes) {
    ODParams p = base();
    p.verifyFrameCache = 1;
    p.quality = m.quality;
    if (m.async) {
      p.waitUntilCompleted = 0;
      p.realtime = 1;
    }
    ODStats st = {};
    // The new footage's own profile, from a session that only saw it.
    ODSession* ref = od_session_create(gCtx);
    for (int pass = 0; pass < 2; ++pass)
      for (int f = 2; f < kN - 2; ++f) CHECK(renderAt(ref, b, f, p, &st), "%s: reference render failed", m.name);
    const float refB = y0(st);
    od_session_destroy(ref);

    ODSession* s = od_session_create(gCtx);
    for (int pass = 0; pass < 2; ++pass)  // every key measured and recently seen
      for (int f = 2; f < kN - 2; ++f) CHECK(renderAt(s, a, f, p, &st), "%s: render of A failed", m.name);
    const float refA = y0(st);
    float after[kN] = {0};
    for (int f = 2; f < kN - 2; ++f) {
      CHECK(renderAt(s, b, f, p, &st), "%s: render after the change failed", m.name);
      after[f] = y0(st);
    }
    const float last = after[kN - 3];
    printf("  %s: Y band 0 before %.5f, new footage alone %.5f; same keys after the change:", m.name, refA, refB);
    for (int f = 2; f < kN - 2; ++f) printf(" %.5f", after[f]);
    printf("\n");
    CHECK(refB > 2.0f * refA, "%s: the two clips are not far apart (%.5f, %.5f): the test says nothing", m.name, refA,
          refB);
    // Playing through: the keys whose entries went are measured again, and
    // two far estimates in a row restart the profile (at playback quality
    // one frame in four is measured once the ring has eight).
    const int by = m.async ? 7 : 4;
    CHECK(std::fabs(after[by] / refB - 1.0f) < 0.2f && std::fabs(last / refB - 1.0f) < 0.2f,
          "%s: the profile did not follow new pixels under the same keys: %.5f and %.5f after %d and 8 renders, new "
          "footage %.5f, old %.5f", m.name, after[by], last, by - 1, refB, refA);
    od_session_destroy(s);

    // Parked on a frame: its entry and samples are in the cache, so the
    // render that gets the new pixels sees them in its check, and the next
    // render of that frame measures it decisively.
    s = od_session_create(gCtx);
    for (int pass = 0; pass < 2; ++pass)
      for (int f = 2; f < kN - 2; ++f) CHECK(renderAt(s, a, f, p, &st), "%s: render of A failed", m.name);
    CHECK(renderAt(s, a, 6, p, &st), "%s: parked render failed", m.name);
    float parked[4] = {0};
    for (int r = 0; r < 4; ++r) {
      CHECK(renderAt(s, b, 6, p, &st), "%s: parked render after the change failed", m.name);
      parked[r] = y0(st);
    }
    printf("  %s, parked on one frame: %.5f %.5f %.5f %.5f\n", m.name, parked[0], parked[1], parked[2], parked[3]);
    const int pr = m.async ? 2 : 1;
    CHECK(std::fabs(parked[pr] / refB - 1.0f) < 0.2f, "%s: parked frame re-rendered %d times with new pixels under its "
          "key: profile %.5f, new footage %.5f, old %.5f", m.name, pr + 1, parked[pr], refB, refA);
    od_session_destroy(s);
  }
}

static void testRealtimeNoStall() {
  gTest = "realtime_no_stall";
  id<MTLCommandQueue> q = [gDevice newCommandQueue];
  char err[256] = {0};
  ODContext* ctx = od_context_create((__bridge void*)gDevice, (__bridge void*)q, err, sizeof err);
  CHECK(ctx != nullptr, "context on our own queue: %s", err);
  if (!ctx) return;
  Frame cur = newFrame(), out = newFrame();
  memcpy(cur.px(), gNoisy[3].px(), (size_t)W * H * 16);
  // Holds the queue: waits for `ev` to reach `value`, then writes `cur` (so
  // every later read of `cur` on this queue is ordered after it).
  id<MTLSharedEvent> ev = [gDevice newSharedEvent];
  uint64_t value = 0;
  auto hold = [&]() {
    id<MTLCommandBuffer> cb = [q commandBuffer];
    [cb encodeWaitForEvent:ev value:++value];
    id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
    [bl copyFromBuffer:gNoisy[3].buffer sourceOffset:0 toBuffer:cur.buffer destinationOffset:0 size:(NSUInteger)W * H * 16];
    [bl endEncoding];
    [cb commit];
  };
  auto release = [&](double afterMs) {
    const uint64_t v = value;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(afterMs * 1e6)),
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{ ev.signaledValue = v; });
  };
  auto now = [] { return CFAbsoluteTimeGetCurrent() * 1000.0; };
  auto window = [&](int64_t keyBase, ODFrame* fr, int* offs) {
    const int d[5] = {0, -1, 1, -2, 2};
    for (int k = 0; k < 5; ++k) {
      fr[k] = (k == 0 ? cur : gNoisy[(size_t)(3 + d[k])]).od(keyBase + d[k]);
      offs[k] = d[k];
    }
  };
  ODStats st = {};
  float ref = 0;
  {
    ODSession* s = od_session_create(ctx);
    CHECK(render(gNoisy, base(), out, &st, s), "reference render failed");
    ref = st.profile.sigma[0][0][3];
    od_session_destroy(s);
  }
  ODFrame fr[5], o = out.od(OD_NO_KEY);
  int offs[5];
  // Control: an export-style cold start (no realtime) waits for the queue.
  {
    ODSession* s = od_session_create(ctx);
    ODParams p = base();
    p.waitUntilCompleted = 0;
    hold();
    release(300);
    window(100, fr, offs);
    const double t0 = now();
    CHECK(od_render(s, fr, offs, 5, &p, &o, &st) == 0, "blocking cold render failed");
    const double ms = now() - t0;
    printf("  cold start, not realtime, queue held 300 ms: od_render %.1f ms\n", ms);
    CHECK(ms > 250, "the non-realtime cold start returned after %.1f ms with the queue held for 300: this test "
          "cannot see a stall", ms);
    od_debug_session_drain(s);
    od_session_destroy(s);
  }
  // Realtime cold start behind the held queue.
  ODSession* s = od_session_create(ctx);
  ODParams p = base();
  p.waitUntilCompleted = 0;
  p.realtime = 1;
  p.quality = 0;
  hold();
  window(200, fr, offs);
  double t0 = now();
  CHECK(od_render(s, fr, offs, 5, &p, &o, &st) == 0, "realtime cold render failed");
  double ms = now() - t0;
  const bool coldValid = st.profile.valid != 0;
  release(0);
  printf("  cold start, realtime, queue held: od_render %.1f ms (profile %s)\n", ms, coldValid ? "valid" : "none yet");
  CHECK(ms < 25, "a realtime cold start took %.1f ms behind a held queue (expected well under a frame interval)", ms);
  CHECK(od_debug_session_drain(s) == 0, "the realtime cold start's estimate never landed");
  window(201, fr, offs);  // the next frame: keys 199..203, of which 199..202 were seen
  CHECK(od_render(s, fr, offs, 5, &p, &o, &st) == 0, "realtime render failed");
  printf("  next realtime render: Y b0 %.5f (blocking, idle queue %.5f)\n", st.profile.sigma[0][0][3], ref);
  CHECK(st.profile.valid && fabsf(st.profile.sigma[0][0][3] / ref - 1) < 0.1f,
        "after the queue moved the realtime session used Y b0 %.5f, the scene's own is %.5f",
        st.profile.sigma[0][0][3], ref);
  // A jump (frames never rendered, another shot) behind a held queue.
  std::vector<Frame> shotB;
  for (int f = 0; f < kFrames; ++f) {
    shotB.push_back(newFrame());
    fillShot(shotB.back(), f, true, 41);
  }
  ODStats stB = {};
  {
    ODSession* r = od_session_create(ctx);
    CHECK(render(shotB, base(), out, &stB, r), "shot B reference failed");
    od_session_destroy(r);
  }
  hold();
  ODFrame jb[5];
  for (int k = 0; k < 5; ++k) jb[k] = shotB[(size_t)(3 + offs[k])].od(7000 + offs[k]);
  t0 = now();
  CHECK(od_render(s, jb, offs, 5, &p, &o, &st) == 0, "realtime jump failed");
  ms = now() - t0;
  release(0);
  printf("  jump to another shot, realtime, queue held: od_render %.1f ms\n", ms);
  CHECK(ms < 25, "a realtime jump took %.1f ms behind a held queue", ms);
  CHECK(od_debug_session_drain(s) == 0, "the jump's estimate never landed");
  for (int k = 0; k < 5; ++k) jb[k] = shotB[(size_t)(3 + offs[k])].od(7001 + offs[k]);
  CHECK(od_render(s, jb, offs, 5, &p, &o, &st) == 0, "realtime render after the jump failed");
  printf("  after the jump: Y b0 %.5f (shot B alone %.5f, before %.5f)\n", st.profile.sigma[0][0][3],
         stB.profile.sigma[0][0][3], ref);
  CHECK(fabsf(st.profile.sigma[0][0][3] / stB.profile.sigma[0][0][3] - 1) < 0.1f,
        "after a realtime jump the session used Y b0 %.5f, shot B's own is %.5f", st.profile.sigma[0][0][3],
        stB.profile.sigma[0][0][3]);
  od_debug_session_drain(s);
  od_session_destroy(s);
  od_context_destroy(ctx);
}

int main() {
  @autoreleasepool {
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
    for (int f = 0; f < kFrames; ++f) {
      gNoisy.push_back(newFrame());
      fill(gNoisy.back(), f, kSigma, 1);
    }
    gClean = newFrame();
    fill(gClean, 3, 0, 0);

    testDefaults();
    testStagesTogether();
    testCountReachesSpatial();
    testAmountsOnce();
    testStrengths();
    testSharpen();
    testSharpenThreshold();
    testViews();
    testReset();
    testStaleKeys();
    testIdleTrim();
    testSizeChange();
    testStaleExact();
    testNoFrameCache();
    testNoiseSign();
    testCacheBudget();
    testPoolTrim();
    testAsyncProfile();
    testCutRestart();
    testColdLowConfidence();
    testFirstWindowV2();
    testKnotAlternation();
    testSingleFrameNoJump();
    testRealtimeNoStall();
    testPersistentChroma();
    testUpstreamChange();

    od_context_destroy(gCtx);
    printf("pipeline tests: %d checks, %d failures\n", gChecks, gFailures);
    return gFailures ? 1 : 0;
  }
}
