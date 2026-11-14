// SPDX-License-Identifier: MIT
// Noise profiler tests against ground truth. No media needed:
// frames are synthesised (profile_synth.h) with luminance-dependent, per-channel, coloured
// noise on scenes with and without flat areas, and the true profile is measured from (noisy
// - clean) with the band definition of (profile_truth.h).
//
// Every case prints estimate / truth per channel and band, so the output is
// also the accuracy report. Memory: a handful of 1280x720 float frames.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "opendenoise/opendenoise.h"

#include "profile_synth.h"
#include "profile_truth.h"

using namespace odtest;

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

static id<MTLDevice> gDevice;
static ODContext* gCtx;
static const uint32_t W = 1280, H = 720;

struct Buf {
  id<MTLBuffer> buffer;
  uint32_t w, h;
  Buf() : buffer(nil), w(0), h(0) {}
  Buf(uint32_t ww, uint32_t hh) : w(ww), h(hh) {
    buffer = [gDevice newBufferWithLength:(NSUInteger)ww * hh * 16 options:MTLResourceStorageModeShared];
  }
  void set(const std::vector<float>& bgra) { memcpy(buffer.contents, bgra.data(), (size_t)w * h * 16); }
  ODFrame frame(int64_t key = OD_NO_KEY) const {
    ODFrame f;
    f.mtlBuffer = (__bridge void*)buffer;
    f.width = w;
    f.height = h;
    f.rowBytes = w * 16;
    f.format = OD_PIXEL_BGRA32F;
    f.frameKey = key;
    return f;
  }
};

static bool finiteProfile(const ODNoiseProfile& p) {
  for (int c = 0; c < 3; ++c)
    for (int l = 0; l < OD_LEVELS; ++l) {
      for (int b = 0; b < OD_BANDS; ++b)
        if (!(std::isfinite(p.sigma[c][b][l]) && p.sigma[c][b][l] > 0)) return false;
      if (!(std::isfinite(p.sigmaFull[c][l]) && p.sigmaFull[c][l] > 0)) return false;
    }
  return std::isfinite(p.confidence) && p.confidence >= 0 && p.confidence <= 1;
}

// Checks one profile against the truth. Bounds are on the pixel-weighted
// median and 95th percentile of estimate / truth:
//   fine  = bands 0, 1 and the full band: median in [fineLo, fineHi], P95 <= fineP95
//   coarse = bands 2, 3:                  median in [coarseLo, coarseHi], P95 <= coarseP95
struct Bounds {
  double fineLo = 0.90, fineHi = 1.10, fineP95 = 1.40;
  double coarseLo = 0.80, coarseHi = 1.45, coarseP95 = 2.30;
};

static RatioStats checkAgainst(const char* what, const ODNoiseProfile& p, const Truth& t, const Bounds& bd = Bounds(),
                               double maxY = 2.0) {
  RatioStats rs;
  rs.add(p, t, 0.005, maxY);
  char title[160];
  snprintf(title, sizeof(title), "  %s (confidence %.2f)", what, p.confidence);
  rs.print(title);
  for (int c = 0; c < 3; ++c) {
    for (int b = 0; b <= OD_BANDS; ++b) {
      const bool coarse = b == 2 || b == 3;
      const double lo = coarse ? bd.coarseLo : bd.fineLo, hi = coarse ? bd.coarseHi : bd.fineHi;
      const double p95 = coarse ? bd.coarseP95 : bd.fineP95;
      const double m = rs.median(c, b);
      CHECK(m >= lo && m <= hi, "%s: channel %d band %d median ratio %.3f outside [%.2f, %.2f]", what, c, b, m, lo, hi);
      CHECK(rs.hi(c, b) <= p95, "%s: channel %d band %d 95th percentile ratio %.3f above %.2f", what, c, b, rs.hi(c, b),
            p95);
    }
  }
  return rs;
}

static NoiseModel coloured(float sy, float sc) {
  NoiseModel m;
  m.sigmaY = sy;
  m.sigmaC = sc;
  m.shadowGain = 1.5f;
  m.white = 1.0f;
  m.medium = 0.6f;
  m.coarse = 0.4f;
  return m;
}

// One noisy frame of a scene as a host buffer, plus its truth.
struct Shot {
  Buf buf;
  Truth truth;
};
static void makeShot(Scene scene, const NoiseModel& m, uint32_t noiseSeed, int shift, Shot* shot, bool wantTruth = true,
                     uint32_t w = W, uint32_t h = H) {
  Ycc clean;
  makeScene(scene, w, h, 11, shift, &clean);
  std::vector<float> noisy, cleanBgra;
  makeFrame(clean, w, h, m, noiseSeed, &noisy, &cleanBgra);
  shot->buf = Buf(w, h);
  shot->buf.set(noisy);
  if (wantTruth) measureTruth(noisy.data(), cleanBgra.data(), w, h, &shot->truth);
}

// One frame through od_profile_frame.
static bool profileOne(ODSession* s, Scene scene, const NoiseModel& m, uint32_t seed, ODNoiseProfile* p, Truth* t) {
  Shot shot;
  makeShot(scene, m, seed, 0, &shot, t != nullptr);
  ODFrame f = shot.buf.frame();
  const int rc = od_profile_frame(s, &f, p);
  if (t) *t = shot.truth;
  return rc == 0;
}

// Renders `cur` with `prev` as temporal neighbour (blocking) and returns the
// profile the session holds afterwards (the one the next render would use).
static bool gNoFrameCache = false;  // renderPair: keys identify frames, nothing is cached
static bool renderPair(ODSession* s, const Buf& cur, int64_t curKey, const Buf* prev, int64_t prevKey, int quality,
                       ODNoiseProfile* after) {
  static Buf out;
  if (out.w != cur.w || out.h != cur.h) out = Buf(cur.w, cur.h);
  ODParams p = od_params_default();
  p.noFrameCache = gNoFrameCache ? 1 : 0;
  p.temporalRadius = 1;
  p.quality = quality;
  p.waitUntilCompleted = 1;
  ODFrame fr[2] = {cur.frame(curKey), prev ? prev->frame(prevKey) : cur.frame(curKey)};
  const int offs[2] = {0, -1};
  ODFrame o = out.frame();
  ODStats st = {};
  if (od_render(s, fr, offs, prev ? 2 : 1, &p, &o, &st) != 0) return false;
  // The slot after this frame's measurement: a second render of the same
  // frame reports it and does not measure again (the key is remembered).
  if (od_render(s, fr, offs, prev ? 2 : 1, &p, &o, &st) != 0) return false;
  *after = st.profile;
  return true;
}

// ---------------------------------------------------------------------------
static void testWhiteFlat() {
  gTest = "white_flat";
  printf("%s: white noise, flat patches at 16 luminance levels\n", gTest);
  ODSession* s = od_session_create(gCtx);
  NoiseModel m;
  m.sigmaY = 0.01f;
  m.sigmaC = 0.006f;
  m.shadowGain = 0;
  ODNoiseProfile p;
  Truth t;
  CHECK(profileOne(s, Scene::Patches, m, 1, &p, &t), "profile failed: %s", od_last_error(s));
  CHECK(p.valid && finiteProfile(p), "profile not valid / finite");
  Bounds bd;
  bd.fineLo = 0.96;
  bd.fineHi = 1.04;
  bd.fineP95 = 1.06;
  bd.coarseLo = 0.93;
  bd.coarseHi = 1.18;
  bd.coarseP95 = 1.30;
  checkAgainst("vs truth", p, t, bd);
  // Against theory: a-trous gains of white noise.
  const double gain[5] = {0.8908, 0.2007, 0.0855, 0.0412, 0.9987};
  for (int b = 0; b <= OD_BANDS; ++b) {
    const double est = b < OD_BANDS ? p.sigma[0][b][3] : p.sigmaFull[0][3];
    const double want = 0.01 * gain[b];
    printf("  Y band %d at knot 3: %.6f, theory %.6f (ratio %.3f)\n", b, est, want, est / want);
    CHECK(std::fabs(est / want - 1) < (b < 3 ? 0.04 : 0.12), "Y band %d: %.6f vs theory %.6f", b, est, want);
  }
  CHECK(p.confidence > 0.8, "confidence %.2f on a frame that is all flat", p.confidence);
  od_session_destroy(s);
}

// Cb and Cr noise differ (2.5 to 1 here): each channel's profile follows its
// own noise. A profile with the two exchanged reads 0.4 instead of 2.5.
static void testCbCr() {
  gTest = "cb_cr";
  printf("%s: white noise, Cb sigma 0.004, Cr sigma 0.010, flat patches\n", gTest);
  ODSession* s = od_session_create(gCtx);
  NoiseModel m;
  m.sigmaY = 0.01f;
  m.sigmaC = 0.004f;
  m.sigmaCr = 0.010f;
  m.shadowGain = 0;
  ODNoiseProfile p;
  Truth t;
  CHECK(profileOne(s, Scene::Patches, m, 3, &p, &t), "profile failed: %s", od_last_error(s));
  CHECK(p.valid && finiteProfile(p), "profile not valid / finite");
  Bounds bd;
  bd.fineLo = 0.94;
  bd.fineHi = 1.06;
  bd.fineP95 = 1.10;
  bd.coarseLo = 0.90;
  bd.coarseHi = 1.20;
  bd.coarseP95 = 1.35;
  checkAgainst("vs truth", p, t, bd);
  for (int b = 0; b < 2; ++b) {
    const double r = p.sigma[2][b][3] / p.sigma[1][b][3];
    printf("  band %d at knot 3: Cb %.6f, Cr %.6f, Cr/Cb %.3f (noise 2.5)\n", b, p.sigma[1][b][3], p.sigma[2][b][3], r);
    CHECK(std::fabs(r / 2.5 - 1) < 0.1, "band %d: Cr/Cb %.3f, the noise is 2.5", b, r);
  }
  od_session_destroy(s);
}

static void testMixed() {
  gTest = "mixed_coloured";
  printf("%s: flat patches + ramps + stripes + fine, soft and uneven texture; coloured noise, stronger in the shadows.\n"
         "  Half of the luminance levels occur only on textured patches.\n",
         gTest);
  ODSession* s = od_session_create(gCtx);
  const float levels[4] = {0.0015f, 0.003f, 0.01f, 0.03f};
  for (float sy : levels) {
    ODNoiseProfile p;
    Truth t;
    CHECK(profileOne(s, Scene::Mixed, coloured(sy, sy * 0.6f), 3, &p, &t), "profile failed: %s", od_last_error(s));
    char what[64];
    snprintf(what, sizeof(what), "sigmaY %.4f", sy);
    Bounds bd;
    if (sy < 0.005f) {  // the textures are 10-30 times the noise here
      bd.fineP95 = 1.60;
      bd.coarseHi = 1.70;
    }
    checkAgainst(what, p, t, bd);
    CHECK(finiteProfile(p), "%s: not finite", what);
    CHECK(p.confidence > 0.3, "%s: confidence %.2f", what, p.confidence);
  }
  od_session_destroy(s);
}

// No flat tile anywhere: texture of varying strength over the whole frame.
// A single frame cannot know the noise; the requirement is that it does not
// grossly overestimate and says so through the confidence. With a static
// neighbour the temporal estimator must get it right.
static void testNoFlat() {
  gTest = "no_flat";
  printf("%s: texture everywhere (log-normal strength, the weakest still above the noise), no flat tile\n", gTest);
  const NoiseModel m = coloured(0.008f, 0.005f);
  Shot a, b, moved;
  makeShot(Scene::Textured, m, 21, 0, &a);
  makeShot(Scene::Textured, m, 22, 0, &b);
  makeShot(Scene::Textured, m, 23, 3, &moved);  // 3 px to the side

  ODSession* s = od_session_create(gCtx);
  ODNoiseProfile single = {};
  ODFrame f = b.buf.frame();
  const bool got = od_profile_frame(s, &f, &single) == 0;
  if (got) {
    RatioStats rs;
    rs.add(single, b.truth);
    char title[96];
    snprintf(title, sizeof(title), "  single frame (confidence %.2f)", single.confidence);
    rs.print(title);
    CHECK(finiteProfile(single), "single: not finite");
    CHECK(single.confidence < 0.3, "single frame without flat tiles reports confidence %.2f", single.confidence);
    printf("  -> single frame: x%.2f of the true Y band 0 (median), confidence %.2f\n", rs.median(0, 0), single.confidence);
    CHECK(rs.median(0, 0) < 1.3, "single frame: Y band 0 overestimated x%.2f", rs.median(0, 0));
    CHECK(rs.hi(0, 0) < 1.6 && rs.hi(0, 1) < 1.6, "single frame: 95th percentile x%.2f / x%.2f", rs.hi(0, 0), rs.hi(0, 1));
  } else {
    printf("  single frame: no estimate (%s)\n", od_last_error(s));
  }
  od_session_destroy(s);

  s = od_session_create(gCtx);
  ODNoiseProfile p = {};
  CHECK(renderPair(s, b.buf, 2, &a.buf, 1, 1, &p), "render failed: %s", od_last_error(s));
  CHECK(p.valid, "no profile from a static pair");
  Bounds bd;
  bd.coarseHi = 1.25;
  checkAgainst("static neighbour (temporal estimator)", p, b.truth, bd);
  CHECK(p.confidence > 0.6, "static pair: confidence %.2f", p.confidence);
  od_session_destroy(s);

  s = od_session_create(gCtx);
  ODNoiseProfile q = {};
  CHECK(renderPair(s, moved.buf, 4, &b.buf, 3, 1, &q), "render failed: %s", od_last_error(s));
  if (q.valid) {
    RatioStats rs;
    rs.add(q, moved.truth);
    char title[96];
    snprintf(title, sizeof(title), "  moving neighbour (confidence %.2f)", q.confidence);
    rs.print(title);
    CHECK(q.confidence < 0.3, "moving textured pair reports confidence %.2f", q.confidence);
    if (got) {
      RatioStats r1;
      r1.add(single, moved.truth);
      CHECK(rs.median(0, 0) <= r1.median(0, 0) * 1.1, "a moving neighbour made the estimate worse: x%.2f vs x%.2f",
            rs.median(0, 0), r1.median(0, 0));
    }
  }
  od_session_destroy(s);
}

// A static, uniformly textured frame (the odbench scene has such a texture
// under everything): the spatial floor is sharp and wrong, the temporal one
// is right, and the combination must follow the temporal one.
static void testUniformTexture() {
  gTest = "uniform_texture";
  printf("%s: static scene with a uniform fine texture over every patch\n", gTest);
  NoiseModel m;
  m.sigmaY = 0.01f;
  m.sigmaC = 0.006f;
  m.shadowGain = 0;
  auto shot = [&](uint32_t seed, Shot* out) {
    Ycc clean;
    makeScene(Scene::Patches, W, H, 11, 0, &clean);
    for (size_t i = 0; i < clean.y.size(); ++i) {
      const uint32_t x = (uint32_t)(i % W), y = (uint32_t)(i / W);
      clean.y[i] += 0.02f * sinf(0.9f * (float)x) * sinf(0.7f * (float)y);
    }
    std::vector<float> noisy, cleanBgra;
    makeFrame(clean, W, H, m, seed, &noisy, &cleanBgra);
    out->buf = Buf(W, H);
    out->buf.set(noisy);
    measureTruth(noisy.data(), cleanBgra.data(), W, H, &out->truth);
  };
  Shot a, b;
  shot(31, &a);
  shot(32, &b);
  ODSession* s = od_session_create(gCtx);
  ODNoiseProfile single = {}, pair = {};
  ODFrame f = b.buf.frame();
  if (od_profile_frame(s, &f, &single) == 0) {
    RatioStats rs;
    rs.add(single, b.truth);
    printf("  single frame: Y band 0 x%.2f, band 1 x%.2f of the truth (a known limit: the texture is taken for noise)\n",
           rs.median(0, 0), rs.median(0, 1));
  }
  od_session_destroy(s);
  s = od_session_create(gCtx);
  CHECK(renderPair(s, b.buf, 2, &a.buf, 1, 1, &pair), "render failed: %s", od_last_error(s));
  Bounds bd;
  checkAgainst("with a static neighbour", pair, b.truth, bd);
  od_session_destroy(s);
}

// Stability over a clip, independence of the order frames arrive in, and
// the reaction to new footage under the same session.
static void testSession() {
  gTest = "session";
  printf("%s: 40 frames of one clip in order, the same frames shuffled, then a clip with 2.5x the noise\n", gTest);
  const int N = 40;
  const NoiseModel m = coloured(0.008f, 0.005f);
  std::vector<Shot> clip((size_t)N);
  for (int i = 0; i < N; ++i) makeShot(Scene::Mixed, m, 100 + (uint32_t)i, 0, &clip[(size_t)i], i == 0);

  auto run = [&](const std::vector<int>& order, int quality, std::vector<ODNoiseProfile>* trace) {
    ODSession* s = od_session_create(gCtx);
    ODNoiseProfile p = {};
    for (int idx : order) {
      const Buf* prev = idx > 0 ? &clip[(size_t)idx - 1].buf : nullptr;
      CHECK(renderPair(s, clip[(size_t)idx].buf, 1000 + idx, prev, 1000 + idx - 1, quality, &p), "render failed: %s",
            od_last_error(s));
      if (trace) trace->push_back(p);
    }
    od_session_destroy(s);
    return p;
  };
  std::vector<int> inOrder, shuffled;
  for (int i = 0; i < N; ++i) inOrder.push_back(i);
  shuffled = inOrder;
  for (int i = N - 1; i > 0; --i) std::swap(shuffled[(size_t)i], shuffled[(size_t)(hash32(77u + (uint32_t)i) % (uint32_t)(i + 1))]);

  std::vector<ODNoiseProfile> trace;
  const ODNoiseProfile a = run(inOrder, 1, &trace);
  const ODNoiseProfile b = run(shuffled, 1, nullptr);
  const ODNoiseProfile c = run(inOrder, 0, nullptr);  // playback: throttled measurements

  // Pumping: the largest frame-to-frame step of any fine luma sigma after the
  // first few frames, and the total drift.
  double maxStep = 0, lo = 1e9, hi = 0;
  for (size_t i = 6; i < trace.size(); ++i) {
    for (int l = 0; l < OD_LEVELS; ++l) {
      maxStep = std::max(maxStep, std::fabs(std::log((double)trace[i].sigma[0][0][l] / trace[i - 1].sigma[0][0][l])));
    }
    lo = std::min(lo, (double)trace[i].sigma[0][0][3]);
    hi = std::max(hi, (double)trace[i].sigma[0][0][3]);
  }
  printf("  in order: largest frame-to-frame step of Y band 0 after frame 6: %.2f %%; range at knot 3: %.2f %%\n",
         100 * maxStep, 100 * (hi / lo - 1));
  CHECK(maxStep < 0.02, "profile steps %.2f %% between frames", 100 * maxStep);
  CHECK(hi / lo < 1.04, "profile drifts %.2f %% over the clip", 100 * (hi / lo - 1));

  auto maxDiff = [](const ODNoiseProfile& x, const ODNoiseProfile& y) {
    double d = 0;
    for (int ch = 0; ch < 3; ++ch)
      for (int l = 0; l < OD_LEVELS; ++l) {
        for (int band = 0; band < 2; ++band)
          d = std::max(d, std::fabs(std::log((double)x.sigma[ch][band][l] / y.sigma[ch][band][l])));
        d = std::max(d, std::fabs(std::log((double)x.sigmaFull[ch][l] / y.sigmaFull[ch][l])));
      }
    return d;
  };
  const double dOrder = maxDiff(a, b), dThrottle = maxDiff(a, c);
  printf("  shuffled vs in order: largest difference of any fine or full-band sigma %.2f %%\n", 100 * dOrder);
  printf("  playback (1 frame in 4 measured) vs every frame: %.2f %%\n", 100 * dThrottle);
  CHECK(dOrder < 0.03, "order of arrival changes the profile by %.2f %%", 100 * dOrder);
  CHECK(dThrottle < 0.05, "throttling changes the profile by %.2f %%", 100 * dThrottle);
  Bounds bd;
  checkAgainst("after 40 frames", a, clip[0].truth, bd);

  // New footage: same scene, 2.5 times the noise.
  NoiseModel m2 = m;
  m2.sigmaY *= 2.5f;
  m2.sigmaC *= 2.5f;
  ODSession* s = od_session_create(gCtx);
  ODNoiseProfile p = {};
  for (int i = 0; i < 12; ++i)
    renderPair(s, clip[(size_t)i].buf, 1000 + i, i ? &clip[(size_t)i - 1].buf : nullptr, 999 + i, 1, &p);
  const double before = p.sigma[0][0][3];
  int reached = -1;
  // The noise steps up inside a clip (a gain change): every frame of the new
  // part is rendered with its own previous frame as neighbour, the first one
  // included (frame 0 of the new part only serves as that neighbour). The
  // keys go on from the first part, so the session sees neither a cut nor a
  // jump; without a frame cache, so key 1011 is not the cached frame 11.
  gNoFrameCache = true;
  Shot prevShot;
  makeShot(Scene::Mixed, m2, 500, 0, &prevShot, false);
  for (int i = 0; i < 10; ++i) {
    Shot cur;
    makeShot(Scene::Mixed, m2, 501 + (uint32_t)i, 0, &cur, false);
    renderPair(s, cur.buf, 1012 + i, &prevShot.buf, 1011 + i, 1, &p);
    if (reached < 0 && p.sigma[0][0][3] > 0.85 * 2.5 * before) reached = i + 1;
    prevShot.buf = cur.buf;
  }
  printf("  noise x2.5: the profile follows after %d frames (Y band 0 at knot 3: %.5f -> %.5f)\n", reached, before,
         (double)p.sigma[0][0][3]);
  // Every frame of new footage filtered with the old profile is a visible
  // flash. Without a cut or a jump to go by (a gain step inside a clip), two
  // far estimates in a row are needed to tell new footage from one odd frame
  // (below). Cuts and jumps restart at once: tests/pipeline, cut_restart.
  CHECK(reached > 0 && reached <= 2, "the session needed %d frames to follow new footage", reached);
  // One odd frame in the sequence (rendered with the frame before it, so not
  // a jump) must not move the profile.
  Shot odd;
  NoiseModel m3 = m2;
  m3.sigmaY *= 3;
  makeShot(Scene::Mixed, m3, 900, 0, &odd, false);
  const double held = p.sigma[0][0][3];
  renderPair(s, odd.buf, 1022, &prevShot.buf, 1021, 1, &p);
  gNoFrameCache = false;
  printf("  one frame with 3x the noise moves the profile by %.2f %%\n", 100 * std::fabs(p.sigma[0][0][3] / held - 1));
  CHECK(std::fabs(p.sigma[0][0][3] / held - 1) < 0.10, "one odd frame moved the profile by %.1f %%",
        100 * std::fabs(p.sigma[0][0][3] / held - 1));
  od_session_destroy(s);
}

// Clipped blacks, blown-out (exactly constant) areas, very dark footage.
static void testClipping() {
  gTest = "clipping";
  printf("%s: black clipped at 0, constant white areas, flat patches at Y = 0.02, 0.06, 0.35\n", gTest);
  NoiseModel m;
  m.sigmaY = 0.006f;
  m.sigmaC = 0.004f;
  m.shadowGain = 0;
  m.clipBlack = true;
  ODSession* s = od_session_create(gCtx);
  ODNoiseProfile p;
  Truth t;
  CHECK(profileOne(s, Scene::DarkClip, m, 5, &p, &t), "profile failed: %s", od_last_error(s));
  CHECK(p.valid && finiteProfile(p), "profile not valid / finite");
  // The white areas hold no noise: they must not drag the estimate to zero,
  // and the bright knots must hold a usable value.
  const double want = 0.006 * 0.8908;
  printf("  Y band 0 per knot:");
  for (int l = 0; l < OD_LEVELS; ++l) printf(" %.5f", p.sigma[0][0][l]);
  printf("   (unclipped noise: %.5f)\n", want);
  for (int l = 1; l < OD_LEVELS; ++l)
    CHECK(p.sigma[0][0][l] > 0.8 * want && p.sigma[0][0][l] < 1.2 * want, "knot %d: %.5f vs %.5f", l, p.sigma[0][0][l], want);
  // Knot 0: the half-clipped blacks are not measured (they would pull the
  // darkest knot, and with it everything up to Y = 0.14, down).
  CHECK(p.sigma[0][0][0] > 0.8 * want && p.sigma[0][0][0] < 1.3 * want, "knot 0: %.5f", p.sigma[0][0][0]);
  // Against the truth below Y = 0.9 (the blown-out patches hold no noise at
  // all). The clipped blacks are deliberately given the noise of the levels
  // just above them, so the estimate may exceed the truth there.
  Bounds bd;
  bd.fineHi = 1.20;
  bd.coarseHi = 1.30;
  checkAgainst("vs truth below white (the clipped blacks have less noise than estimated, by design)", p, t, bd, 0.9);
  od_session_destroy(s);
}

// Very dark footage whose noise is NOT clipped at black: a float pipeline carries values
// below 0 (sub-black of video-range sources, linear raw), and negative values prove the
// noise is not cut off. A uniformly black frame (lens cap, night sky, a clip that opens on
// black) at 0 and at minus one sigma must still give a profile, and the render must denoise
// it. Only pixels pinned at exactly 0 (or without any fine detail) are dead.
static void testDarkUnclipped() {
  gTest = "dark-unclipped";
  const float sy = 0.015f, sc = 0.01f;
  const double want = sy * 0.8908;  // band 0 of white noise
  for (float black : {0.0f, -sy, 0.01f}) {
    Ycc clean;
    clean.y.assign((size_t)W * H, black);
    clean.cb.assign((size_t)W * H, 0.0f);
    clean.cr.assign((size_t)W * H, 0.0f);
    NoiseModel m;
    m.sigmaY = sy;
    m.sigmaC = sc;
    m.shadowGain = 0;
    m.clipBlack = false;
    std::vector<float> noisy, cleanBgra;
    makeFrame(clean, W, H, m, 71, &noisy, &cleanBgra);
    Buf in(W, H), out(W, H);
    in.set(noisy);
    ODSession* s = od_session_create(gCtx);
    ODNoiseProfile p;
    memset(&p, 0, sizeof(p));
    ODFrame f = in.frame();
    const int rc = od_profile_frame(s, &f, &p);
    printf("%s: black level %+.3f, noise not clipped: profile rc %d valid %d, Y band 0 at knot 0 %.5f (truth %.5f)\n",
           gTest, black, rc, p.valid, p.sigma[0][0][0], want);
    CHECK(rc == 0 && p.valid && finiteProfile(p), "black %+.3f: no profile (rc %d, valid %d)", black, rc, p.valid);
    CHECK(p.sigma[0][0][0] > 0.9 * want && p.sigma[0][0][0] < 1.1 * want, "black %+.3f: knot 0 %.5f vs %.5f", black,
          p.sigma[0][0][0], want);
    // The render (automatic profile) must denoise the frame, not pass it through.
    ODParams prm = od_params_default();
    prm.temporalRadius = 0;
    prm.waitUntilCompleted = 1;
    ODFrame o = out.frame();
    ODStats st = {};
    const int off0 = 0;
    CHECK(od_render(s, &f, &off0, 1, &prm, &o, &st) == 0, "render failed: %s", od_last_error(s));
    const float* a = (const float*)in.buffer.contents;
    const float* b = (const float*)out.buffer.contents;
    double ein = 0, eout = 0;
    for (size_t i = 0; i < (size_t)W * H; ++i) {
      const double gi = a[i * 4 + 1] - cleanBgra[i * 4 + 1], go = b[i * 4 + 1] - cleanBgra[i * 4 + 1];
      ein += gi * gi;
      eout += go * go;
    }
    ein = sqrt(ein / ((double)W * H));
    eout = sqrt(eout / ((double)W * H));
    printf("  render: green rms error %.5f -> %.5f (profile valid %d)\n", ein, eout, st.profile.valid);
    CHECK(st.profile.valid && eout < 0.7 * ein, "black %+.3f: not denoised (rms %.5f -> %.5f)", black, ein, eout);
    od_session_destroy(s);
  }
}

// Night footage in log: the noise rises steeply into the toe of the curve
// (here 4.5 times the mid-tone noise below Y 0.165, then a step down, as
// measured on a real S-Log3 clip) and 40 % of the frame sits in the toe. The profile used to drop every level of the toe as
// texture (a rise beyond 2.5 per knot interval) and gave the shadows the
// noise of the mid-tones: half of theirs. Two levels occur only on texture,
// one of them in the toe, below every flat patch: that one must not be taken
// for noise.
static void testLogToe() {
  gTest = "log_toe";
  printf("%s: noise 4.5x stronger in the toe (Y < 0.165), 40 %% of the frame there; textured-only levels at Y 0.075, 0.48\n",
         gTest);
  NoiseModel m;
  m.sigmaY = 0.001f;
  m.sigmaC = 0.0006f;
  m.shadowGain = 0;
  m.white = 1.0f;
  m.medium = 0.6f;
  m.coarse = 0.4f;
  // As measured on a real S-Log3 night clip:
  // 4.5 times the mid-tone noise up to Y 0.165, a fall by 2.6 within the next
  // 0.05 of Y (15 per knot interval), then by 2.1 per interval.
  m.toeGain = 4.5f;
  m.toeEnd = 0.165f;
  m.toeFall = 15.0f;
  m.toeSteep = 0.05f;
  m.toeFall2 = 2.1f;
  for (uint32_t seed : {41u, 42u}) {
    ODSession* s = od_session_create(gCtx);
    ODNoiseProfile p;
    Truth t;
    CHECK(profileOne(s, Scene::LogNight, m, seed, &p, &t), "profile failed: %s", od_last_error(s));
    CHECK(p.valid && finiteProfile(p), "profile not valid / finite");
    Bounds bd;
    bd.fineLo = 0.85;
    bd.fineHi = 1.15;
    bd.fineP95 = 1.45;
    bd.coarseHi = 1.60;
    char what[64];
    snprintf(what, sizeof(what), "seed %u, all levels", seed);
    checkAgainst(what, p, t, bd);
    // The toe alone (the flat toe patches, Y 0.09 - 0.13).
    RatioStats toe;
    for (int l = 0; l < kTruthBins; ++l) {
      if (t.share[l] < 0.005 || t.meanY[l] < 0.085 || t.meanY[l] > 0.135) continue;
      for (int b : {0, 1, OD_BANDS}) {
        const double est = profileAt(b < OD_BANDS ? p.sigma[0][b] : p.sigmaFull[0], t.meanY[l]);
        toe.r[0][b].emplace_back(est / t.sigma[0][b][l], t.share[l]);
      }
    }
    printf("  toe (Y 0.085-0.135): Y b0 %.2f  b1 %.2f  bF %.2f (median estimate / truth), confidence %.2f\n",
           toe.median(0, 0), toe.median(0, 1), toe.median(0, OD_BANDS), p.confidence);
    for (int b : {0, 1, OD_BANDS})
      CHECK(toe.median(0, b) > 0.85 && toe.median(0, b) < 1.15, "toe: Y band %d at %.2f of the truth", b, toe.median(0, b));
    // The textured-only level below the toe patches (texture 2 to 8 times the
    // noise there) is not taken for noise: the darkest knot stays within the
    // noise of the toe, not the texture's level.
    double toeNoise = 0, toeShare = 0;
    for (int l = 0; l < kTruthBins; ++l) {
      if (t.share[l] < 0.005 || t.meanY[l] < 0.085 || t.meanY[l] > 0.135) continue;
      toeNoise += t.share[l] * t.sigma[0][0][l];
      toeShare += t.share[l];
    }
    toeNoise /= std::max(toeShare, 1e-9);
    printf("  knot 0: Y b0 %.5f (toe noise %.5f)\n", p.sigma[0][0][0], toeNoise);
    CHECK(p.sigma[0][0][0] < 1.45 * toeNoise, "knot 0 at %.2f of the toe noise: texture taken for noise",
          p.sigma[0][0][0] / toeNoise);
    CHECK(p.confidence > 0.6, "confidence %.2f", p.confidence);
    od_session_destroy(s);
  }
}

// Light noise on bright content: the half-float working texture has a step of
// 4.9e-4 at Y > 0.5; the coarse bands must not pick up quantisation noise.
static void testLightNoise() {
  gTest = "light_noise";
  printf("%s: sigma 0.0012 on flat patches (half-float step at Y > 0.5 is 0.00049)\n", gTest);
  NoiseModel m;
  m.sigmaY = 0.0012f;
  m.sigmaC = 0.0008f;
  m.shadowGain = 0;
  ODSession* s = od_session_create(gCtx);
  ODNoiseProfile p;
  Truth t;
  CHECK(profileOne(s, Scene::Patches, m, 9, &p, &t), "profile failed: %s", od_last_error(s));
  Bounds bd;
  bd.fineHi = 1.12;
  bd.coarseHi = 1.35;
  checkAgainst("vs truth", p, t, bd);
  od_session_destroy(s);
}

// Frames too small for any tile, odd sizes, and the profile view.
static void testEdges() {
  gTest = "edges";
  printf("%s: tiny and odd frame sizes, profile view\n", gTest);
  const NoiseModel m = coloured(0.01f, 0.006f);
  ODSession* s = od_session_create(gCtx);
  {
    Shot tiny;
    makeShot(Scene::Patches, m, 1, 0, &tiny, false, 48, 40);
    ODNoiseProfile p = {};
    ODFrame f = tiny.buf.frame();
    CHECK(od_profile_frame(s, &f, &p) != 0 && !p.valid, "a 48x40 frame gave a profile");
    ODNoiseProfile after = {};
    CHECK(renderPair(s, tiny.buf, 1, nullptr, 0, 1, &after), "render of a 48x40 frame failed: %s", od_last_error(s));
  }
  const uint32_t sizes[3][2] = {{641, 363}, {482, 270}, {1921, 1081}};
  for (auto& sz : sizes) {
    Shot shot;
    makeShot(Scene::Patches, m, 2, 0, &shot, true, sz[0], sz[1]);
    ODNoiseProfile p = {};
    ODFrame f = shot.buf.frame();
    const int rc = od_profile_frame(s, &f, &p);
    CHECK(rc == 0 && p.valid && finiteProfile(p), "%ux%u: no profile (%s)", sz[0], sz[1], rc ? od_last_error(s) : "");
    if (rc == 0) {
      RatioStats rs;
      rs.add(p, shot.truth);
      printf("  %ux%u: Y band 0 x%.2f, full x%.2f of the truth, confidence %.2f\n", sz[0], sz[1], rs.median(0, 0),
             rs.median(0, 4), p.confidence);
      // 482x270 leaves one 32x32 tile per patch: a handful of candidates, low
      // confidence, and therefore a deliberately cautious (scaled-down) estimate.
      const bool starved = sz[0] < 600;
      CHECK(rs.median(0, 0) > (starved ? 0.25 : 0.85) && rs.median(0, 0) < 1.15, "%ux%u: Y band 0 ratio %.2f", sz[0], sz[1],
            rs.median(0, 0));
      CHECK(rs.median(0, 4) < 1.25, "%ux%u: full-band ratio %.2f", sz[0], sz[1], rs.median(0, 4));
    }
  }
  // OD_VIEW_PROFILE: produces an image that differs from the input (tint),
  // is finite, and keeps alpha.
  {
    Shot shot;
    makeShot(Scene::Mixed, m, 3, 0, &shot, false);
    Buf out(W, H);
    ODParams p = od_params_default();
    p.view = OD_VIEW_PROFILE;
    p.waitUntilCompleted = 1;
    ODFrame in = shot.buf.frame(77), o = out.frame();
    const int offs[1] = {0};
    int rc = od_render(s, &in, offs, 1, &p, &o, nullptr);
    rc = rc ? rc : od_render(s, &in, offs, 1, &p, &o, nullptr);  // second render: with the profile of the first
    CHECK(rc == 0, "profile view failed: %s", od_last_error(s));
    const float* a = (const float*)shot.buf.buffer.contents;
    const float* b = (const float*)out.buffer.contents;
    size_t changed = 0, bad = 0, alpha = 0, greenish = 0, reddish = 0;
    for (size_t i = 0; i < (size_t)W * H; ++i) {
      for (int k = 0; k < 3; ++k) {
        if (!std::isfinite(b[i * 4 + k])) ++bad;
      }
      if (std::fabs(b[i * 4] - a[i * 4]) > 0.02f || std::fabs(b[i * 4 + 2] - a[i * 4 + 2]) > 0.02f) ++changed;
      if (b[i * 4 + 3] != a[i * 4 + 3]) ++alpha;
      if (b[i * 4 + 1] > b[i * 4 + 2] + 0.05f && b[i * 4 + 1] > b[i * 4] + 0.05f) ++greenish;
      if (b[i * 4 + 2] > b[i * 4 + 1] + 0.05f) ++reddish;
    }
    printf("  view: %.0f %% of the pixels tinted, %.0f %% green (flat), %.0f %% red (structure)\n",
           100.0 * (double)changed / (W * H), 100.0 * (double)greenish / (W * H), 100.0 * (double)reddish / (W * H));
    CHECK(bad == 0, "view: %zu non-finite values", bad);
    CHECK(alpha == 0, "view: alpha changed on %zu pixels", alpha);
    CHECK(greenish > (size_t)W * H / 10, "view: only %zu pixels marked flat", greenish);
    CHECK(reddish > (size_t)W * H / 10, "view: only %zu pixels marked as structure", reddish);
  }
  od_session_destroy(s);
}

int main() {
  @autoreleasepool {
    gDevice = MTLCreateSystemDefaultDevice();
    if (!gDevice) {
      fprintf(stderr, "no Metal device\n");
      return 1;
    }
    char err[512] = "";
    gCtx = od_context_create((__bridge void*)gDevice, nullptr, err, sizeof(err));
    if (!gCtx) {
      fprintf(stderr, "context: %s\n", err);
      return 1;
    }
    if (getenv("OD_PROFILE_CALIBRATE")) {
      // Uniform grey + white noise at several sizes: prints the spread of the
      // tile energies (run with OD_PROFILE_DEBUG=3).
      ODSession* s = od_session_create(gCtx);
      NoiseModel m;
      m.shadowGain = 0;
      ODNoiseProfile p;
      for (int k = 0; k < 3; ++k) profileOne(s, Scene::Grey, m, 100 + (uint32_t)k, &p, nullptr);
      od_session_destroy(s);
      return 0;
    }
    testWhiteFlat();
    testCbCr();
    testMixed();
    testNoFlat();
    testUniformTexture();
    testSession();
    testClipping();
    testDarkUnclipped();
    testLogToe();
    testLightNoise();
    testEdges();
    od_context_destroy(gCtx);
    printf("%d checks, %d failures\n", gChecks, gFailures);
    return gFailures ? 1 : 0;
  }
}
