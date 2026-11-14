// SPDX-License-Identifier: MIT
// Tests of the temporal stage alone (profile, spatial and sharpen switched off,
// amounts at 1, a manual profile computed from the known synthetic noise).
// Self-contained: scenes are generated here, no media needed.
//
//   static      residual noise against 1/sqrt(2R+1), R = 1..5, luma and chroma
//   pan         sub-pixel global motion: noise reduction and no detail loss
//   bigmotion   24 px/frame pan at 1920x1080 (the whole motion hierarchy)
//   occlusion   a fast object over a textured background, strong, faint and
//               far above white: error near the moving edges must not exceed
//               the unfiltered one
//   cut         a scene cut inside the window: the other shot must not leak
//   repeat      repeated frames carry no weight; a window of repeats is identity
//   repeat_chroma  equal luma with other chroma is not a repeat
//   lighting    a brightness step in the neighbours must not shift the frame
//   cache       keyed (cached pyramids) and unkeyed renders are byte-identical
//   edges       one-sided windows, odd and small sizes, no profile, strength -> 0
//   view        OD_VIEW_MOTION is finite and reads 1 on a static scene
//   darkflat    noise that depends on luminance (log footage): a static dark
//               frame is merged fully, with near-zero vectors
//   mottle      temporal + spatial on a static frame whose low-frequency noise
//               partly persists between frames: noise at 16 and 32 px
//   floor       the luma grain floor applies whichever stage applies the
//               amounts: spatial strength 0 and 1e-6 give the same picture
//   pin         the picture against the stage before it was optimised
//
// Memory: at most 11 frames of 1920x1080 float32 (about 370 MB) at a time
// (darkflat: 7 frames and their clean copies, about 470 MB).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "opendenoise/opendenoise.h"

#include "debug.h"

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
static ODContext* gCtx;
static bool gResidualModel = false;

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------
struct Img {
  int w = 0, h = 0;
  std::vector<float> px;  // BGRA float32
  id<MTLBuffer> buf = nil;
  int64_t key = OD_NO_KEY;
  void alloc(int W, int H) {
    w = W;
    h = H;
    px.assign((size_t)W * H * 4, 0.0f);
  }
  void upload() {
    const size_t bytes = px.size() * sizeof(float);
    if (!buf || buf.length != bytes) buf = [gDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memcpy(buf.contents, px.data(), bytes);
  }
  ODFrame frame(bool keyed) const {
    ODFrame f;
    f.mtlBuffer = (__bridge void*)buf;
    f.width = (uint32_t)w;
    f.height = (uint32_t)h;
    f.rowBytes = (uint32_t)w * 16;
    f.format = OD_PIXEL_BGRA32F;
    f.frameKey = keyed ? key : OD_NO_KEY;
    return f;
  }
};

static uint32_t hash32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

// Scene: continuous position and frame index to RGB. Band-limited on purpose
// (nothing finer than about 4 pixels, soft edges): point-sampled hard edges
// would alias differently in every frame, which no camera does.
typedef std::function<void(float x, float y, float* rgb)> Scene;

static float softStep(float d, float width) {
  float t = std::min(1.0f, std::max(0.0f, 0.5f + d / width));
  return t * t * (3.0f - 2.0f * t);
}

static void sceneA(float x, float y, float* rgb) {
  float lum = 0.30f + 0.18f * sinf(0.011f * x + 0.004f * y) + 0.10f * sinf(0.045f * y - 0.02f * x);
  lum += 0.05f * sinf(0.9f * x) * sinf(0.7f * y);          // fine texture
  lum += 0.04f * sinf(0.33f * x + 1.0f) * sinf(0.41f * y);  // medium texture
  float bars = softStep(sinf(0.12f * x + 0.03f * y), 0.5f);  // soft-edged stripes
  if (y > 250.0f && y < 330.0f) lum = 0.18f + 0.45f * bars;
  rgb[0] = lum + 0.10f * sinf(0.02f * x);
  rgb[1] = lum;
  rgb[2] = lum + 0.10f * cosf(0.025f * y);
}

static void sceneB(float x, float y, float* rgb) {
  float lum = 0.42f + 0.20f * cosf(0.017f * x - 0.013f * y) + 0.06f * sinf(0.6f * x + 0.5f * y);
  float blob = softStep(60.0f - hypotf(x - 200.0f, y - 150.0f), 3.0f);
  lum = lum * (1.0f - blob) + 0.75f * blob;
  lum += 0.05f * sinf(0.21f * x) * sinf(0.8f * y);
  rgb[0] = lum - 0.08f * sinf(0.03f * y);
  rgb[1] = lum;
  rgb[2] = lum + 0.06f * sinf(0.015f * x);
}

static void renderScene(Img* img, int w, int h, const Scene& scene) {
  img->alloc(w, h);
  float* px = img->px.data();
  dispatch_apply((size_t)h, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t y) {
    for (int x = 0; x < w; ++x) {
      float rgb[3];
      scene((float)x, (float)y, rgb);
      float* o = px + (y * (size_t)w + (size_t)x) * 4;
      o[0] = rgb[2];
      o[1] = rgb[1];
      o[2] = rgb[0];
      o[3] = 1.0f;
    }
  });
}

// Independent Gaussian noise of the given sigma on R, G and B.
static void addNoise(Img* img, float sigma, uint32_t seed) {
  float* px = img->px.data();
  const int w = img->w;
  dispatch_apply((size_t)img->h, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t y) {
    for (int x = 0; x < w; ++x) {
      for (int c = 0; c < 3; ++c) {
        uint32_t a = hash32((uint32_t)x * 73856093u ^ (uint32_t)y * 19349663u ^ (uint32_t)c * 83492791u ^
                            seed * 2654435761u);
        uint32_t b = hash32(a ^ 0x9E3779B9u);
        float u1 = ((float)(a >> 8) + 1.0f) / 16777217.0f, u2 = (float)(b >> 8) / 16777216.0f;
        float g = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
        px[(y * (size_t)w + (size_t)x) * 4 + (size_t)c] += sigma * g;
      }
    }
  });
}

// The profile of white noise of sigma s on R, G and B, in the header's
// definition (a-trous band gains, working-space channel gains).
static ODNoiseProfile whiteProfile(float s) {
  const float chan[3] = {0.74958f, 0.64162f, 0.67702f};  // Y, Cb, Cr
  const float gain[OD_BANDS] = {0.8908f, 0.2007f, 0.0855f, 0.0412f};
  ODNoiseProfile p;
  memset(&p, 0, sizeof(p));
  for (int c = 0; c < 3; ++c)
    for (int l = 0; l < OD_LEVELS; ++l) {
      for (int b = 0; b < OD_BANDS; ++b) p.sigma[c][b][l] = s * chan[c] * gain[b];
      p.sigmaFull[c][l] = s * chan[c] * 0.9987f;
    }
  p.confidence = 1;
  p.valid = 1;
  return p;
}

static ODParams temporalOnly(const ODNoiseProfile* prof, int radius, int quality) {
  ODParams p = od_params_default();
  p.lumaStrength = 1;
  p.temporalNoiseModel = gResidualModel ? 1u : 0u;
  p.chromaStrength = 1;
  p.temporalRadius = radius;
  p.temporalStrength = 1;
  p.quality = quality;
  p.waitUntilCompleted = 1;
  p.manualProfile = prof;
  p.debugSkipStages = (1u << OD_STAGE_PROFILE) | (1u << OD_STAGE_SPATIAL) | (1u << OD_STAGE_SHARPEN);
  return p;
}

// Renders seq[cur] with the neighbours seq[cur + o] for o in offsets (those
// outside the sequence are dropped, as a host does at clip ends).
static bool renderFrame(ODSession* s, std::vector<Img>& seq, int cur, const std::vector<int>& offsets,
                        const ODParams& p, bool keyed, Img* out, ODStats* st = nullptr) {
  ODFrame frames[OD_MAX_FRAMES];
  int offs[OD_MAX_FRAMES];
  int n = 0;
  frames[n] = seq[(size_t)cur].frame(keyed);
  offs[n++] = 0;
  for (int o : offsets) {
    int j = cur + o;
    if (o == 0 || j < 0 || j >= (int)seq.size() || n >= OD_MAX_FRAMES) continue;
    frames[n] = seq[(size_t)j].frame(keyed);
    offs[n++] = o;
  }
  out->alloc(seq[(size_t)cur].w, seq[(size_t)cur].h);
  out->upload();
  ODFrame fo = out->frame(false);
  ODStats local;
  if (od_render(s, frames, offs, n, &p, &fo, st ? st : &local) != 0) {
    CHECK(false, "render failed: %s", od_last_error(s));
    return false;
  }
  memcpy(out->px.data(), out->buf.contents, out->px.size() * sizeof(float));
  if (gResidualModel && p.temporalStrength == 1 && (p.debugSkipStages & (1u << OD_STAGE_SPATIAL))) {
    ODParams legacy = p; legacy.temporalNoiseModel = 0;
    CHECK(od_render(s, frames, offs, n, &legacy, &fo, &local) == 0, "legacy parity render failed");
    CHECK(memcmp(out->px.data(), out->buf.contents, out->px.size()*sizeof(float)) == 0,
          "new mode altered full-strength temporal rejection/merge");
  }
  return true;
}

static std::vector<int> window(int radius) {
  std::vector<int> o;
  for (int d = 1; d <= radius; ++d) {
    o.push_back(-d);
    o.push_back(d);
  }
  return o;
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------
static inline void ycc(const float* bgra, float* o) {
  float y = 0.2126f * bgra[2] + 0.7152f * bgra[1] + 0.0722f * bgra[0];
  o[0] = y;
  o[1] = (bgra[0] - y) / 1.8556f;
  o[2] = (bgra[2] - y) / 1.5748f;
}

struct Err {
  double rms[3] = {0, 0, 0};  // Y, Cb, Cr
  double maxBlur = 0;         // largest |5x5 box mean of the luma error|: coherent error, i.e. ghosts
  double meanY = 0;           // mean luma error (bias)
  size_t n = 0;
  bool finite = true;
};

// Error of `a` against `ref` over the pixels where mask != 0 (mask may be
// null), leaving out a border.
static Err errorOf(const Img& a, const Img& ref, const std::vector<uint8_t>* mask, int border) {
  Err e;
  const int w = a.w, h = a.h;
  std::vector<float> ey((size_t)w * h);
  double s[3] = {0, 0, 0}, sm = 0;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      size_t i = (size_t)y * w + x;
      float pa[3], pr[3];
      ycc(&a.px[i * 4], pa);
      ycc(&ref.px[i * 4], pr);
      for (int c = 0; c < 4; ++c)
        if (!std::isfinite(a.px[i * 4 + (size_t)c])) e.finite = false;
      ey[i] = pa[0] - pr[0];
      if (x < border || y < border || x >= w - border || y >= h - border) continue;
      if (mask && !(*mask)[i]) continue;
      for (int c = 0; c < 3; ++c) s[c] += (double)(pa[c] - pr[c]) * (pa[c] - pr[c]);
      sm += ey[i];
      ++e.n;
    }
  if (e.n) {
    for (int c = 0; c < 3; ++c) e.rms[c] = sqrt(s[c] / (double)e.n);
    e.meanY = sm / (double)e.n;
  }
  const int b2 = std::max(border, 2);
  for (int y = b2; y < h - b2; ++y)
    for (int x = b2; x < w - b2; ++x) {
      size_t i = (size_t)y * w + x;
      if (mask && !(*mask)[i]) continue;
      double sum = 0;
      for (int dy = -2; dy <= 2; ++dy)
        for (int dx = -2; dx <= 2; ++dx) sum += ey[(size_t)(y + dy) * w + (x + dx)];
      e.maxBlur = std::max(e.maxBlur, fabs(sum / 25.0));
    }
  return e;
}

// Debugging aid: with OD_TEST_DUMP=<dir>, writes luma images as 8-bit PNG
// through ffmpeg (value = 0.5 + gain * (a - ref), or gain * a when ref is null).
static void dumpLuma(const char* name, const Img& a, const Img* ref, float gain) {
  const char* dir = getenv("OD_TEST_DUMP");
  if (!dir) return;
  char cmd[1024];
  snprintf(cmd, sizeof(cmd),
           "/opt/homebrew/bin/ffmpeg -v error -y -f rawvideo -pix_fmt gray -s %dx%d -i - '%s/%s.png'", a.w, a.h, dir,
           name);
  FILE* f = popen(cmd, "w");
  if (!f) return;
  std::vector<uint8_t> g((size_t)a.w * a.h);
  for (size_t i = 0; i < g.size(); ++i) {
    float pa[3], pr[3] = {0, 0, 0};
    ycc(&a.px[i * 4], pa);
    if (ref) ycc(&ref->px[i * 4], pr);
    float v = ref ? 0.5f + gain * (pa[0] - pr[0]) : gain * pa[0];
    g[i] = (uint8_t)std::min(255.0f, std::max(0.0f, v * 255.0f + 0.5f));
  }
  fwrite(g.data(), 1, g.size(), f);
  pclose(f);
}

static size_t diffBytes(const Img& a, const Img& b) {
  size_t n = 0;
  for (size_t i = 0; i < a.px.size(); ++i) n += memcmp(&a.px[i], &b.px[i], 4) != 0;
  return n;
}

static double maxAbsDiff(const Img& a, const Img& b) {
  double m = 0;
  for (size_t i = 0; i < a.px.size(); ++i) m = std::max(m, (double)fabsf(a.px[i] - b.px[i]));
  return m;
}

// A sequence: frame t = scene(t) + noise, uploaded, keyed from keyBase.
static void makeSequence(std::vector<Img>* noisy, std::vector<Img>* clean, int n, int w, int h, float sigma,
                         uint32_t seed, int64_t keyBase, const std::function<Scene(int)>& sceneAt) {
  noisy->assign((size_t)n, Img());
  clean->assign((size_t)n, Img());
  for (int t = 0; t < n; ++t) {
    renderScene(&(*clean)[(size_t)t], w, h, sceneAt(t));
    (*noisy)[(size_t)t] = (*clean)[(size_t)t];
    (*noisy)[(size_t)t].buf = nil;
    if (sigma > 0) addNoise(&(*noisy)[(size_t)t], sigma, seed + (uint32_t)t * 7919u);
    (*noisy)[(size_t)t].key = keyBase + t;
    (*noisy)[(size_t)t].upload();
  }
}

static const float kSigma = 0.03f;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
static void testStatic() {
  gTest = "static";
  const int W = 512, H = 384, N = 11, C = 5;
  std::vector<Img> noisy, clean;
  makeSequence(&noisy, &clean, N, W, H, kSigma, 11, 1000, [](int) { return Scene(sceneA); });
  ODNoiseProfile prof = whiteProfile(kSigma);
  Err in = errorOf(noisy[C], clean[C], nullptr, 16);
  printf("static %dx%d, sigma %.3f: input luma rms %.5f\n", W, H, kSigma, in.rms[0]);
  printf("  R  q  frames  ideal   lumaRatio  CbRatio  CrRatio   lumaRatio/ideal\n");
  for (int q = 0; q <= 1; ++q)
    for (int R = 1; R <= 5; ++R) {
      ODSession* s = od_session_create(gCtx);
      Img out;
      ODStats st;
      ODParams p = temporalOnly(&prof, R, q);
      if (renderFrame(s, noisy, C, window(R), p, true, &out, &st)) {
        Err e = errorOf(out, clean[C], nullptr, 16);
        double ideal = 1.0 / sqrt(2.0 * R + 1.0);
        double ry = e.rms[0] / in.rms[0], rb = e.rms[1] / in.rms[1], rr = e.rms[2] / in.rms[2];
        printf("  %d  %d  %d      %.4f  %.4f     %.4f   %.4f    %.3f\n", R, q, st.framesUsed, ideal, ry, rb, rr,
               ry / ideal);
        CHECK(e.finite, "R=%d q=%d: non-finite output", R, q);
        CHECK(st.framesUsed == 2 * R + 1, "R=%d: framesUsed %d", R, st.framesUsed);
        CHECK(ry < ideal * 1.10 && ry > ideal * 0.97, "R=%d q=%d: luma noise ratio %.4f, ideal %.4f", R, q, ry, ideal);
        CHECK(rb < ideal * 1.12 && rr < ideal * 1.12, "R=%d q=%d: chroma noise ratios %.4f %.4f, ideal %.4f", R, q, rb,
              rr, ideal);
        CHECK(fabs(e.meanY) < 2e-4, "R=%d q=%d: mean luma shifted by %.6f", R, q, e.meanY);
      }
      od_session_destroy(s);
    }
}

static void testPan(const char* name, int W, int H, float vx, float vy, double maxRatioQ1, double maxRatioQ0) {
  gTest = name;
  const int N = 5, C = 2, R = 2;
  std::vector<Img> noisy, clean;
  makeSequence(&noisy, &clean, N, W, H, kSigma, 23, 2000, [=](int t) {
    return Scene([=](float x, float y, float* rgb) { sceneA(x + vx * (float)t, y + vy * (float)t, rgb); });
  });
  ODNoiseProfile prof = whiteProfile(kSigma);
  // Leave out what the neighbours cannot see (the picture moved out of frame).
  const int border = 16 + (int)ceilf(2.0f * std::max(fabsf(vx), fabsf(vy)));
  Err in = errorOf(noisy[C], clean[C], nullptr, border);
  for (int q = 0; q <= 1; ++q) {
    ODSession* s = od_session_create(gCtx);
    Img out;
    ODParams p = temporalOnly(&prof, R, q);
    if (renderFrame(s, noisy, C, window(R), p, true, &out)) {
      Err e = errorOf(out, clean[C], nullptr, border);
      double ry = e.rms[0] / in.rms[0];
      printf("%s %dx%d v=(%.2f, %.2f) px/frame, R=2, q=%d: luma ratio %.4f (ideal %.4f), chroma %.4f %.4f, "
             "coherent error %.5f (input %.5f)\n",
             name, W, H, vx, vy, q, ry, 1 / sqrt(5.0), e.rms[1] / in.rms[1], e.rms[2] / in.rms[2], e.maxBlur,
             in.maxBlur);
      // The motion view carries the vector of the nearest neighbour (offset -1):
      // it must be the true motion. Frame t shows the scene at x + v t, so the
      // current pixel x is found in frame t - 1 at x + v.
      Img view;
      ODParams pv = p;
      pv.view = OD_VIEW_MOTION;
      if (renderFrame(s, noisy, C, window(R), pv, true, &view)) {
        double sx = 0, sy = 0, sxx = 0, syy = 0, share = 0;
        size_t n = 0, good = 0, fair = 0;
        for (int y = border; y < H - border; ++y)
          for (int x = border; x < W - border; ++x) {
            float v[3];
            ycc(&view.px[((size_t)y * W + x) * 4], v);
            double mx = v[1] * 64.0, my = v[2] * 64.0;
            sx += mx;
            sy += my;
            sxx += (mx - vx) * (mx - vx);
            syy += (my - vy) * (my - vy);
            share += v[0];
            double ev = hypot(mx - vx, my - vy);
            good += ev < 0.25;
            fair += ev < 1.0;
            ++n;
          }
        double errV = sqrt((sxx + syy) / (double)n);
        printf("    vectors of the nearest neighbour: mean (%.3f, %.3f), true (%.3f, %.3f), rms error %.3f px, within "
               "0.25 px: %.1f%%, within 1 px: %.1f%%; mean share of neighbours merged %.3f\n",
               sx / (double)n, sy / (double)n, vx, vy, errV, 100.0 * (double)good / (double)n,
               100.0 * (double)fair / (double)n, share / (double)n);
        // Where the scene has structure in one direction only, or repeats,
        // the motion is not determined: no bound on the rms, only on the bulk.
        CHECK((double)fair / (double)n > 0.70, "q=%d: only %.1f%% of the vectors are within 1 px", q,
              100.0 * (double)fair / (double)n);
        if (getenv("OD_TEST_DUMP")) {
          char nm[128];
          snprintf(nm, sizeof(nm), "%s_q%d_merged", name, q);
          dumpLuma(nm, view, nullptr, 1.0f);
          Img ev = view;
          for (size_t i = 0; i < ev.px.size() / 4; ++i) {
            float v[3];
            ycc(&view.px[i * 4], v);
            float g = (float)std::min(1.0, hypot(v[1] * 64.0 - vx, v[2] * 64.0 - vy) / 4.0);
            ev.px[i * 4] = ev.px[i * 4 + 1] = ev.px[i * 4 + 2] = g;
          }
          snprintf(nm, sizeof(nm), "%s_q%d_vecerr", name, q);
          dumpLuma(nm, ev, nullptr, 1.0f);
          snprintf(nm, sizeof(nm), "%s_q%d_clean", name, q);
          dumpLuma(nm, clean[C], nullptr, 1.0f);
        }
      }
      CHECK(e.finite, "q=%d: non-finite output", q);
      CHECK(ry < (q ? maxRatioQ1 : maxRatioQ0), "q=%d: luma noise ratio %.4f", q, ry);
      CHECK(e.maxBlur < in.maxBlur * 1.0, "q=%d: coherent error %.5f above the input's %.5f", q, e.maxBlur, in.maxBlur);
    }
    od_session_destroy(s);
  }
}

// A soft-edged textured disc crossing a static background.
// chromaOnly: the object has the luma of the background behind it and differs
// only in Cb and Cr (+0.08, -0.08), so a merge that compared luma alone would
// average the background's colour into the object's edge. Measured: chroma
// near the edges 0.51 of the input's; bound 0.7.
static void testOcclusion(const char* name, float contrast, float speed, bool chromaOnly = false, int R = 2) {
  gTest = name;
  const int W = 512, H = 384, N = 2*R+1, C = R;
  const float radius = 46.0f;
  auto centre = [=](int t, float* cx, float* cy) {
    *cx = 256.0f + speed * (float)(t - C);
    *cy = 190.0f + 0.35f * speed * (float)(t - C);
  };
  std::vector<Img> noisy, clean;
  makeSequence(&noisy, &clean, N, W, H, kSigma, 37, 3000, [=](int t) {
    float cx, cy;
    centre(t, &cx, &cy);
    return Scene([=](float x, float y, float* rgb) {
      sceneA(x, y, rgb);
      float a = softStep(radius - hypotf(x - cx, y - cy), 2.5f);
      if (a > 0) {
        // the object: opaque, its own texture, moving with it
        float u = x - cx, v = y - cy;
        float o = 0.30f + contrast + 0.04f * sinf(0.5f * u) * sinf(0.45f * v) + 0.03f * sinf(0.11f * u + 0.07f * v);
        const float tint[3] = {1.05f, 1.0f, 0.92f};
        if (chromaOnly) {
          // same luma as the background, chroma moved: BT.709 back to RGB
          const float yb = 0.2126f * rgb[0] + 0.7152f * rgb[1] + 0.0722f * rgb[2];
          const float cb = (rgb[2] - yb) / 1.8556f + 0.08f, cr = (rgb[0] - yb) / 1.5748f - 0.08f;
          const float r = yb + 1.5748f * cr, b = yb + 1.8556f * cb, g = (yb - 0.2126f * r - 0.0722f * b) / 0.7152f;
          const float obj[3] = {r, g, b};
          for (int c = 0; c < 3; ++c) rgb[c] = rgb[c] * (1.0f - a) + a * obj[c];
        } else {
          for (int c = 0; c < 3; ++c) rgb[c] = rgb[c] * (1.0f - a) + a * o * tint[c];
        }
      }
    });
  });
  // Region: within 6 px of the object's outline in any frame of the window,
  // plus everything the object covers in a neighbour but not now (where a
  // ghost or a trail would sit).
  std::vector<uint8_t> edge((size_t)W * H, 0), all((size_t)W * H, 0);
  float ccx, ccy;
  centre(C, &ccx, &ccy);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      bool e = false;
      for (int t = 0; t < N; ++t) {
        float cx, cy;
        centre(t, &cx, &cy);
        float d = hypotf((float)x - cx, (float)y - cy);
        if (fabsf(d - radius) < 6.0f) e = true;
        if (t != C && d < radius && hypotf((float)x - ccx, (float)y - ccy) > radius) e = true;
      }
      edge[(size_t)y * W + x] = e;
      all[(size_t)y * W + x] = 1;
    }
  ODNoiseProfile prof = whiteProfile(kSigma);
  Err inE = errorOf(noisy[C], clean[C], &edge, 16), inA = errorOf(noisy[C], clean[C], &all, 16);
  for (int q = 0; q <= 1; ++q) {
    ODSession* s = od_session_create(gCtx);
    Img out;
    ODParams p = temporalOnly(&prof, R, q);
    if (renderFrame(s, noisy, C, window(R), p, true, &out)) {
      Err e = errorOf(out, clean[C], &edge, 16), a = errorOf(out, clean[C], &all, 16);
      if (getenv("OD_TEST_DUMP")) {
        char nm[128];
        snprintf(nm, sizeof(nm), "%s_q%d_err_out", name, q);
        dumpLuma(nm, out, &clean[C], 8.0f);
        snprintf(nm, sizeof(nm), "%s_q%d_err_in", name, q);
        dumpLuma(nm, noisy[C], &clean[C], 8.0f);
        snprintf(nm, sizeof(nm), "%s_q%d_out", name, q);
        dumpLuma(nm, out, nullptr, 1.2f);
        Img view;
        ODParams pv = p;
        pv.view = OD_VIEW_MOTION;
        if (renderFrame(s, noisy, C, window(R), pv, true, &view)) {
          snprintf(nm, sizeof(nm), "%s_q%d_merged", name, q);
          dumpLuma(nm, view, nullptr, 1.0f);
        }
      }
      printf("%s contrast %.3f speed %.0f px/frame q=%d: moving-edge region (%zu px) rms %.5f (input %.5f), "
             "coherent error %.5f (input %.5f); whole frame rms ratio %.4f\n",
             name, contrast, speed, q, e.n, e.rms[0], inE.rms[0], e.maxBlur, inE.maxBlur, a.rms[0] / inA.rms[0]);
      CHECK(e.finite, "q=%d: non-finite output", q);
      // Absolute bounds, so that a re-pin cannot bake in weaker rejection:
      // measured 0.62 to 0.68 of the input near the moving edges (the
      // superwhite object highest), and a coherent error up to 0.96 of the
      // input's (the scenes are seeded: deterministic).
      CHECK(e.rms[0] <= inE.rms[0] * 0.72, "q=%d: rms near moving edges %.5f, more than 0.72 of unfiltered %.5f", q,
            e.rms[0], inE.rms[0]);
      CHECK(e.maxBlur <= inE.maxBlur, "q=%d: coherent error near moving edges %.5f worse than unfiltered %.5f", q,
            e.maxBlur, inE.maxBlur);
      printf("    chroma near the moving edges: Cb rms %.5f (input %.5f), Cr rms %.5f (input %.5f)\n", e.rms[1],
             inE.rms[1], e.rms[2], inE.rms[2]);
      if (chromaOnly)
        for (int c = 1; c < 3; ++c)
          CHECK(e.rms[c] <= inE.rms[c] * 0.7, "q=%d: chroma channel %d near the edges of a chroma-only object %.5f, "
                "input %.5f: the background's colour was merged in", q, c, e.rms[c], inE.rms[c]);
      CHECK(a.rms[0] / inA.rms[0] < 0.60, "q=%d: whole-frame noise ratio %.4f", q, a.rms[0] / inA.rms[0]);
    }
    od_session_destroy(s);
  }
}

// Thin lines that move against their background (cables, hair, rain): the 8x8 blocks lock
// onto the textured background, so in every neighbour the line sits elsewhere, and only the
// merge's test keeps the background from being blended into it. A 1-px line of 7 luma
// sigmas, and a 2-px line of red that leaves luma unchanged (12 luma sigmas of R), both 6 px
// per frame across a static textured background. 77 of its contrast and the red line 0.68 of
// its colour here; with luma's t0 alone doubled (kLevel0) 0.98 and 0.82, as with no change
// at level 0 (0.98, 0.82). The bounds sit between.
static void testThinLines(int R = 2) {
  gTest = "thin_lines";
  const int W = 640, H = 360, N = 2*R+5, C = R+2, speed = 6;
  const float sigma = 0.02f;
  const float ySigma = sigma * sqrtf(0.2126f * 0.2126f + 0.7152f * 0.7152f + 0.0722f * 0.0722f);
  const float lumaStep = 7.0f * ySigma, redStep = 12.0f * ySigma;
  // Line k of frame t starts at x0(k) + speed * t; rows 0..H/2 hold the luma
  // lines, rows H/2..H the red ones.
  auto x0 = [=](int k, int t) { return 60 + 120 * k + speed * t; };
  std::vector<Img> noisy, clean;
  for (int t = 0; t < N; ++t) {
    Img c;
    renderScene(&c, W, H, sceneA);
    for (int y = 0; y < H; ++y)
      for (int k = 0; k < 4; ++k) {
        const bool red = y >= H / 2;
        for (int dx = 0; dx < (red ? 2 : 1); ++dx) {
          float* o = &c.px[((size_t)y * W + (size_t)(x0(k, t) + dx)) * 4];
          if (red) {
            o[2] += redStep;                            // R
            o[1] -= redStep * 0.2126f / 0.7152f;        // G: luma unchanged
          } else {
            for (int ch = 0; ch < 3; ++ch) o[ch] += lumaStep;
          }
        }
      }
    Img n = c;
    addNoise(&n, sigma, 4100 + (uint32_t)t);
    c.key = n.key = 9000 + t;
    n.upload();
    clean.push_back(c);
    noisy.push_back(n);
  }
  ODNoiseProfile prof = whiteProfile(sigma);
  // Kept: (line - background 20 to 30 px to its left) in the output over the
  // same in the clean frame, on luma for the luma lines and on R - Y for the
  // red ones, averaged over the rows away from the edges.
  auto kept = [&](const Img& out, bool red) {
    double num = 0, den = 0;
    const int y0 = red ? H / 2 + 8 : 8, y1 = red ? H - 8 : H / 2 - 8;
    auto val = [&](const Img& im, int x, int y) {
      const float* p = &im.px[((size_t)y * W + (size_t)x) * 4];
      const float yy = 0.2126f * p[2] + 0.7152f * p[1] + 0.0722f * p[0];
      return red ? p[2] - yy : yy;
    };
    for (int k = 0; k < 4; ++k) {
      const int x = x0(k, C);
      for (int y = y0; y < y1; ++y) {
        double lo = 0, lc = 0, bo = 0, bc = 0;
        for (int dx = 0; dx < (red ? 2 : 1); ++dx) {
          lo += val(out, x + dx, y);
          lc += val(clean[C], x + dx, y);
        }
        for (int b = x - 30; b < x - 20; ++b) {
          bo += val(out, b, y);
          bc += val(clean[C], b, y);
        }
        const int lw = red ? 2 : 1;
        num += lo / lw - bo / 10;
        den += lc / lw - bc / 10;
      }
    }
    return den != 0 ? num / den : 0.0;
  };
  for (int q = 0; q <= 1; ++q) {
    ODSession* s = od_session_create(gCtx);
    Img out;
    if (renderFrame(s, noisy, C, window(R), temporalOnly(&prof, R, q), true, &out)) {
      const double kl = kept(out, false), kr = kept(out, true);
      printf("thin lines q=%d: 1-px luma line of 7 sigmas keeps %.3f of its contrast, 2-px red line of 12 sigmas "
             "%.3f of its colour\n", q, kl, kr);
      CHECK(kl >= 0.90, "q=%d: a thin moving luma line kept %.3f of its contrast (the background was merged in)", q,
            kl);
      CHECK(kr >= 0.75, "q=%d: a thin moving red line kept %.3f of its colour (the background's colour was merged "
            "in)", q, kr);
    }
    od_session_destroy(s);
  }
}

static void testCut() {
  gTest = "cut";
  const int W = 512, H = 384, N = 6, R = 2;
  std::vector<Img> noisy, clean;
  // frames 0..2: shot A, frames 3..5: shot B. Frame 3 is the first of B.
  makeSequence(&noisy, &clean, N, W, H, kSigma, 53, 4000,
               [](int t) { return t < 3 ? Scene(sceneA) : Scene(sceneB); });
  ODNoiseProfile prof = whiteProfile(kSigma);
  for (int cur : {3, 2}) {
    Err in = errorOf(noisy[(size_t)cur], clean[(size_t)cur], nullptr, 16);
    for (int q = 0; q <= 1; ++q) {
      ODSession* s = od_session_create(gCtx);
      ODSession* s2 = od_session_create(gCtx);
      Img out, same;
      ODParams p = temporalOnly(&prof, R, q);
      // same-shot neighbours only: what the result should be
      std::vector<int> sameShot = cur == 3 ? std::vector<int>{1, 2} : std::vector<int>{-1, -2};
      if (renderFrame(s, noisy, cur, window(R), p, true, &out) &&
          renderFrame(s2, noisy, cur, sameShot, p, true, &same)) {
        Err e = errorOf(out, clean[(size_t)cur], nullptr, 16);
        double leak = maxAbsDiff(out, same);
        printf("cut: frame %d (%s), q=%d: luma ratio %.4f (ideal with 2 same-shot neighbours %.4f), coherent "
               "error %.5f (input %.5f), max difference from the same-shot-only render %.2e\n",
               cur, cur == 3 ? "first after the cut" : "last before the cut", q, e.rms[0] / in.rms[0], 1 / sqrt(3.0),
               e.maxBlur, in.maxBlur, leak);
        CHECK(e.finite, "non-finite output");
        CHECK(leak < 1e-4, "frame %d q=%d: the other shot changed the result by up to %.2e", cur, q, leak);
        CHECK(e.rms[0] / in.rms[0] < 0.64, "frame %d q=%d: luma ratio %.4f (same-shot neighbours not used?)", cur, q,
              e.rms[0] / in.rms[0]);
        CHECK(e.maxBlur <= in.maxBlur, "frame %d q=%d: coherent error %.5f above the input's %.5f", cur, q, e.maxBlur,
              in.maxBlur);
      }
      od_session_destroy(s);
      od_session_destroy(s2);
    }
  }
  // A window made only of the other shot: nothing may be merged.
  ODSession* s = od_session_create(gCtx);
  Img out;
  ODParams p = temporalOnly(&prof, R, 1);
  if (renderFrame(s, noisy, 3, {-1, -2}, p, true, &out)) {
    double d = maxAbsDiff(out, noisy[3]);
    printf("cut: only other-shot neighbours: max change %.2e\n", d);
    CHECK(d < 1e-4, "a window of another shot changed the frame by up to %.2e", d);
  }
  od_session_destroy(s);
}

// Mean share of the neighbours merged (OD_VIEW_MOTION, Y) inside a border.
static double meanMotionShare(const Img& view, int W, int H, int border) {
  double sum = 0;
  size_t n = 0;
  for (int y = border; y < H - border; ++y)
    for (int x = border; x < W - border; ++x) {
      float v[3];
      ycc(&view.px[((size_t)y * W + x) * 4], v);
      sum += v[0];
      ++n;
    }
  return n ? sum / (double)n : 0;
}

static void testRepeat() {
  gTest = "repeat";
  const int W = 512, H = 384;
  ODNoiseProfile prof = whiteProfile(kSigma);
  // Distinct frames D0..D4 of a slow pan, each shown twice: D0 D0 D1 D1 D2 D2 ...
  std::vector<Img> distinct, clean;
  makeSequence(&distinct, &clean, 5, W, H, kSigma, 71, 5000, [](int t) {
    return Scene([=](float x, float y, float* rgb) { sceneA(x + 1.5f * (float)t, y, rgb); });
  });
  std::vector<Img> doubled;
  for (int t = 0; t < 5; ++t)
    for (int k = 0; k < 2; ++k) {
      doubled.push_back(distinct[(size_t)t]);
      doubled.back().buf = nil;
      doubled.back().key = 5100 + t * 2 + k;
      doubled.back().upload();
    }
  for (int q = 0; q <= 1; ++q) {
    ODSession* s = od_session_create(gCtx);
    ODSession* s2 = od_session_create(gCtx);
    Img out, ref;
    ODParams p = temporalOnly(&prof, 2, q);
    // current = first copy of D2 (index 4): window D1 D1 [D2] D2 D3. Distinct
    // content: D1 (once), D2, D3.
    const int cur = 4;
    std::vector<Img> three = {distinct[1], distinct[2], distinct[3]};
    for (size_t i = 0; i < three.size(); ++i) {
      three[i].buf = nil;
      three[i].key = 5200 + (int64_t)i;
      three[i].upload();
    }
    if (renderFrame(s, doubled, cur, window(2), p, true, &out) && renderFrame(s2, three, 1, window(1), p, true, &ref)) {
      Err e = errorOf(out, clean[2], nullptr, 16), in = errorOf(doubled[(size_t)cur], clean[2], nullptr, 16);
      Err r = errorOf(ref, clean[2], nullptr, 16);
      printf("repeat: doubled frames, R=2, q=%d: luma ratio %.4f; with the 3 distinct frames only %.4f; max "
             "difference %.2e\n",
             q, e.rms[0] / in.rms[0], r.rms[0] / in.rms[0], maxAbsDiff(out, ref));
      // Offsets differ between the two renders (the prediction seeds too), so
      // the vectors could differ slightly: not bit-exact, but within half-float
      // rounding (exactly 0 measured; averaging the repeats in instead gives
      // 3e-2 and moves the noise level by 2.65 % of the input's).
      CHECK(fabs(e.rms[0] - r.rms[0]) < 0.005 * in.rms[0], "q=%d: repeats changed the noise level: %.5f vs %.5f", q,
            e.rms[0], r.rms[0]);
      CHECK(maxAbsDiff(out, ref) < 1e-3, "q=%d: repeats changed the picture by up to %.2e", q, maxAbsDiff(out, ref));
      // The motion view: of the four neighbours, the repeat of the current
      // frame (and the repeat of D1 beyond D1) carry no weight, so at most
      // half are merged (0.48 measured; 0.97 without the repeat gate).
      Img view;
      ODParams pv = p;
      pv.view = OD_VIEW_MOTION;
      if (renderFrame(s, doubled, cur, window(2), pv, true, &view)) {
        const double share = meanMotionShare(view, W, H, 16);
        printf("repeat: doubled frames, q=%d: share of neighbours merged %.4f\n", q, share);
        CHECK(share < 0.55, "q=%d: the repeats were merged (share %.4f, at most 0.5 expected)", q, share);
      }
    }
    od_session_destroy(s);
    od_session_destroy(s2);
  }
  // Every neighbour is a bit-exact copy of the current frame: identity.
  std::vector<Img> frozen;
  for (int k = 0; k < 5; ++k) {
    frozen.push_back(distinct[2]);
    frozen.back().buf = nil;
    frozen.back().key = 5300 + k;
    frozen.back().upload();
  }
  ODSession* s = od_session_create(gCtx);
  Img out;
  ODParams p = temporalOnly(&prof, 2, 1);
  if (renderFrame(s, frozen, 2, window(2), p, true, &out)) {
    size_t d = diffBytes(out, frozen[2]);
    printf("repeat: frozen frame (5 identical frames): %zu values changed\n", d);
    CHECK(d == 0, "a window of identical frames changed %zu values", d);
    // Identity holds whether or not the repeats are detected (c + 0 == c):
    // the motion view tells. 0 merged with the gate, 1 without.
    Img view;
    ODParams pv = p;
    pv.view = OD_VIEW_MOTION;
    if (renderFrame(s, frozen, 2, window(2), pv, true, &view)) {
      const double share = meanMotionShare(view, W, H, 16);
      printf("repeat: frozen frame: share of neighbours merged %.4f\n", share);
      CHECK(share < 0.02, "a window of identical frames merged a share of %.4f (repeats must carry no weight)", share);
    }
  }
  od_session_destroy(s);
}

// A neighbour whose luma equals the next frame inwards but whose chroma differs is not a
// repeat. The frames' working-space Y is made bit-identical (half values converted to RGB in
// float, checked through the ingest); their chroma noise is independent. The temporal merge
// must take the chroma noise down as on any static scene.
static void testRepeatChroma() {
  gTest = "repeat_chroma";
  const int W = 256, H = 192, N = 5;
  const float sC = kSigma * 0.65f;  // chroma noise in working units
  ODNoiseProfile prof = whiteProfile(kSigma);
  std::vector<Img> seq((size_t)N);
  std::vector<float> cleanCC((size_t)W * H * 2);
  for (int t = 0; t < N; ++t) {
    Img& im = seq[(size_t)t];
    im.alloc(W, H);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        float rgb[3];
        sceneA((float)x, (float)y, rgb);
        const float y0 = 0.2126f * rgb[0] + 0.7152f * rgb[1] + 0.0722f * rgb[2];
        // A fixed luma noise pattern (the same in every frame), on half values.
        const uint32_t a = hash32((uint32_t)x * 73856093u ^ (uint32_t)y * 19349663u ^ 0x1234567u);
        const float Y = (float)(__fp16)(y0 + kSigma * 0.75f * (((float)(a & 0xffff) / 65535.0f) - 0.5f) * 3.4641f);
        const size_t i = (size_t)y * W + x;
        const float cb0 = (rgb[2] - y0) / 1.8556f, cr0 = (rgb[0] - y0) / 1.5748f;
        cleanCC[i * 2] = cb0;
        cleanCC[i * 2 + 1] = cr0;
        float cc[2] = {cb0, cr0};
        for (int c = 0; c < 2; ++c) {
          const uint32_t h1 =
              hash32((uint32_t)x * 2654435761u ^ (uint32_t)y * 40503u ^ (uint32_t)(t * 2 + c) * 97u ^ 0xC0FFEEu);
          const uint32_t h2 = hash32(h1 ^ 0x9E3779B9u);
          const float u1 = ((float)(h1 >> 8) + 1.0f) / 16777217.0f, u2 = (float)(h2 >> 8) / 16777216.0f;
          cc[c] += sC * sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
        }
        // The inverse of od_rgb_to_ycc (common.metal), in float.
        const float r = Y + 1.5748f * cc[1], b = Y + 1.8556f * cc[0];
        const float g = (Y - 0.2126f * r - 0.0722f * b) * (1.0f / 0.7152f);
        float* o = &im.px[i * 4];
        o[0] = b;
        o[1] = g;
        o[2] = r;
        o[3] = 1.0f;
      }
    im.key = 9100 + t;
    im.upload();
  }
  // Precondition: what the ingest makes of them, Y bit-identical in every
  // frame, chroma not.
  ODSession* s = od_session_create(gCtx);
  std::vector<std::vector<float>> work((size_t)N, std::vector<float>((size_t)W * H * 4));
  bool read = true;
  for (int t = 0; t < N; ++t) {
    ODFrame f = seq[(size_t)t].frame(false);
    read = read && od_debug_read_working(s, &f, work[(size_t)t].data()) == 0;
  }
  size_t yDiff = 0, cDiff = 0;
  for (int t = 1; t < N && read; ++t)
    for (size_t i = 0; i < (size_t)W * H; ++i) {
      yDiff += memcmp(&work[(size_t)t][i * 4], &work[0][i * 4], 4) != 0;
      cDiff += memcmp(&work[(size_t)t][i * 4 + 1], &work[0][i * 4 + 1], 8) != 0;
    }
  CHECK(read && yDiff == 0 && cDiff > (size_t)W * H, "working Y differs in %zu texels, chroma in %zu: the "
        "construction does not give equal luma with other chroma (the test says nothing)", yDiff, cDiff);
  for (int q = 0; q <= 1; ++q) {
    Img out;
    ODParams p = temporalOnly(&prof, 2, q);
    ODStats st = {};
    if (!renderFrame(s, seq, 2, window(2), p, true, &out, &st)) continue;
    const size_t changed = diffBytes(out, seq[2]);
    // Chroma error against the clean chroma, before and after.
    double ein = 0, eout = 0;
    for (int y = 16; y < H - 16; ++y)
      for (int x = 16; x < W - 16; ++x) {
        const size_t i = (size_t)y * W + x;
        float a[3], b[3];
        ycc(&seq[2].px[i * 4], a);
        ycc(&out.px[i * 4], b);
        for (int c = 1; c < 3; ++c) {
          const float ref = cleanCC[i * 2 + (size_t)c - 1];
          ein += (a[c] - ref) * (a[c] - ref);
          eout += (b[c] - ref) * (b[c] - ref);
        }
      }
    const double ratio = sqrt(eout / std::max(ein, 1e-30));
    printf("repeat_chroma: equal luma, independent chroma noise, q=%d: framesUsed %d, %zu of %zu values changed, "
           "chroma error ratio %.3f\n", q, st.framesUsed, changed, out.px.size(), ratio);
    CHECK(ratio < 0.7, "q=%d: equal-luma frames with other chroma were dropped as repeats: chroma error ratio %.3f "
          "(%zu values changed)", q, ratio, changed);
  }
  od_session_destroy(s);
}

// noiseLevel scales every sigma the stage uses (stage.h, env.sigmaScale): with
// the spatial stage off, a scene near the rejection threshold (neighbours 3 %
// brighter) must merge strictly more as noiseLevel rises, and noiseLevel 2
// with a profile of sigma s must give the picture of noiseLevel 1 with 2 s.
static void testNoiseLevel() {
  gTest = "noise_level";
  const int W = 512, H = 384, N = 5, C = 2, R = 2;
  const ODNoiseProfile prof = whiteProfile(kSigma), prof2 = whiteProfile(2 * kSigma);
  std::vector<Img> noisy, clean;
  makeSequence(&noisy, &clean, N, W, H, kSigma, 93, 6500, [=](int t) {
    return Scene([=](float x, float y, float* rgb) {
      sceneA(x, y, rgb);
      if (t != C)
        for (int c = 0; c < 3; ++c) rgb[c] *= 1.03f;
    });
  });
  ODSession* s = od_session_create(gCtx);
  double share[3] = {0, 0, 0};
  const float levels[3] = {0.5f, 1.0f, 2.0f};
  for (int k = 0; k < 3; ++k) {
    ODParams p = temporalOnly(&prof, R, 1);
    p.noiseLevel = levels[k];
    p.view = OD_VIEW_MOTION;
    Img view;
    if (renderFrame(s, noisy, C, window(R), p, true, &view)) share[k] = meanMotionShare(view, W, H, 16);
  }
  printf("noise_level: neighbours 3%% brighter, share merged at noise level 0.5 / 1 / 2: %.4f %.4f %.4f\n", share[0],
         share[1], share[2]);
  CHECK(share[0] < share[1] && share[1] < share[2], "the merged share does not rise with noiseLevel: %.4f %.4f %.4f",
        share[0], share[1], share[2]);
  CHECK(share[2] - share[0] > 0.05, "noiseLevel 0.5 to 2 changes the merged share by only %.4f", share[2] - share[0]);
  // The grain floor follows the measured noise, not the slider (stage.h,
  // kGrainFloor): a luma amount under both caps keeps it out of the way.
  Img a, b;
  ODParams pa = temporalOnly(&prof, R, 1), pb = temporalOnly(&prof2, R, 1);
  pa.noiseLevel = 2;
  pa.lumaStrength = pb.lumaStrength = 0.9f;
  if (renderFrame(s, noisy, C, window(R), pa, true, &a) && renderFrame(s, noisy, C, window(R), pb, true, &b)) {
    printf("noise_level: level 2 with sigma s against level 1 with 2 s: max difference %.2e\n", maxAbsDiff(a, b));
    CHECK(maxAbsDiff(a, b) < 1e-3, "noiseLevel 2 differs from a profile twice as high by up to %.2e", maxAbsDiff(a, b));
  }
  od_session_destroy(s);
}

static void testLighting() {
  gTest = "lighting";
  const int W = 512, H = 384, N = 5, C = 2, R = 2;
  ODNoiseProfile prof = whiteProfile(kSigma);
  for (float step : {0.02f, 0.10f}) {
    std::vector<Img> noisy, clean;
    // neighbours are brighter (flash, exposure step); the current frame is not
    makeSequence(&noisy, &clean, N, W, H, kSigma, 89, 6000 + (int64_t)(step * 1000), [=](int t) {
      return Scene([=](float x, float y, float* rgb) {
        sceneA(x, y, rgb);
        if (t != C)
          for (int c = 0; c < 3; ++c) rgb[c] *= 1.0f + step;
      });
    });
    Err in = errorOf(noisy[C], clean[C], nullptr, 16);
    for (int q = 0; q <= 1; ++q) {
      ODSession* s = od_session_create(gCtx);
      Img out;
      ODParams p = temporalOnly(&prof, R, q);
      if (renderFrame(s, noisy, C, window(R), p, true, &out)) {
        Err e = errorOf(out, clean[C], nullptr, 16);
        printf("lighting: neighbours %.0f%% brighter, q=%d: mean luma shift %.5f (the step is %.5f), rms %.5f "
               "(input %.5f), coherent error %.5f (input %.5f)\n",
               step * 100, q, e.meanY, step * 0.3, e.rms[0], in.rms[0], e.maxBlur, in.maxBlur);
        CHECK(e.finite, "non-finite output");
        CHECK(fabs(e.meanY) < 0.1 * step * 0.3 + 3e-4, "step %.2f q=%d: mean luma shifted by %.5f", step, q, e.meanY);
        CHECK(e.rms[0] <= in.rms[0] * 1.0, "step %.2f q=%d: rms %.5f worse than unfiltered %.5f", step, q, e.rms[0],
              in.rms[0]);
      }
      od_session_destroy(s);
    }
  }
}

static void testCache() {
  gTest = "cache";
  const int W = 384, H = 288, N = 9, R = 2;
  std::vector<Img> noisy, clean;
  makeSequence(&noisy, &clean, N, W, H, kSigma, 97, 7000, [](int t) {
    return Scene([=](float x, float y, float* rgb) { sceneA(x + 2.3f * (float)t, y - 0.7f * (float)t, rgb); });
  });
  ODNoiseProfile prof = whiteProfile(kSigma);
  for (int q = 0; q <= 1; ++q) {
    ODParams p = temporalOnly(&prof, R, q);
    ODSession* a = od_session_create(gCtx);  // keyed, in order
    ODSession* b = od_session_create(gCtx);  // unkeyed
    ODSession* c = od_session_create(gCtx);  // keyed, shuffled, each frame twice
    std::vector<Img> oa((size_t)N), ob((size_t)N), oc((size_t)N);
    int hits = 0, hitsB = 0;
    bool ok = true;
    for (int t = 0; ok && t < N; ++t) {
      ODStats st, stb;
      ok = renderFrame(a, noisy, t, window(R), p, true, &oa[(size_t)t], &st) &&
           renderFrame(b, noisy, t, window(R), p, false, &ob[(size_t)t], &stb);
      hits += st.cacheHits;
      hitsB += stb.cacheHits;
    }
    for (int t : {4, 0, 8, 2, 6, 1, 7, 3, 5, 4, 0, 8}) ok = ok && renderFrame(c, noisy, t, window(R), p, true, &oc[(size_t)t]);
    // unkeyed, shuffled, with repeats (the order tests/core uses on its own frames)
    ODSession* d = od_session_create(gCtx);
    std::vector<Img> od((size_t)N);
    for (int t : {3, 0, 6, 2, 5, 1, 4, 3, 3, 0, 7, 8}) ok = ok && renderFrame(d, noisy, t, window(R), p, false, &od[(size_t)t]);
    if (ok) {
      size_t dab = 0, dac = 0, dad = 0;
      for (int t = 0; t < N; ++t) {
        dab += diffBytes(oa[(size_t)t], ob[(size_t)t]);
        dac += diffBytes(oa[(size_t)t], oc[(size_t)t]);
        dad += diffBytes(oa[(size_t)t], od[(size_t)t]);
      }
      CHECK(dad == 0, "q=%d: unkeyed shuffled renders with repeats differ in %zu values", q, dad);
      printf("cache: q=%d, %d frames: cached vs uncached %zu values differ, in-order vs shuffled %zu; %d cache hits "
             "keyed, %d unkeyed\n",
             q, N, dab, dac, hits, hitsB);
      CHECK(dab == 0, "q=%d: cached and uncached renders differ in %zu values", q, dab);
      CHECK(dac == 0, "q=%d: in-order and shuffled renders differ in %zu values", q, dac);
      CHECK(hits > 0, "keyed renders had no cache hits");
      CHECK(hitsB == 0, "unkeyed renders reported %d cache hits", hitsB);
      Err in = errorOf(noisy[4], clean[4], nullptr, 24), e = errorOf(oa[4], clean[4], nullptr, 24);
      CHECK(e.rms[0] < 0.6 * in.rms[0], "q=%d: the sequence was not denoised (%.5f vs %.5f)", q, e.rms[0], in.rms[0]);
    }
    od_session_destroy(a);
    od_session_destroy(b);
    od_session_destroy(c);
    od_session_destroy(d);
  }
}

static void testEdges() {
  gTest = "edges";
  ODNoiseProfile prof = whiteProfile(kSigma);
  // one-sided windows (clip start and end) and a window with a hole
  {
    const int W = 512, H = 384, N = 5;
    std::vector<Img> noisy, clean;
    makeSequence(&noisy, &clean, N, W, H, kSigma, 101, 8000, [](int) { return Scene(sceneA); });
    struct Case {
      int cur;
      std::vector<int> offs;
      int frames;
    } cases[] = {{0, {-2, -1, 1, 2}, 3}, {4, {-2, -1, 1, 2}, 3}, {2, {-2, 2}, 3}, {2, {1}, 2}};
    for (const Case& c : cases) {
      ODSession* s = od_session_create(gCtx);
      Img out;
      ODStats st;
      ODParams p = temporalOnly(&prof, 2, 1);
      if (renderFrame(s, noisy, c.cur, c.offs, p, true, &out, &st)) {
        Err in = errorOf(noisy[(size_t)c.cur], clean[(size_t)c.cur], nullptr, 16);
        Err e = errorOf(out, clean[(size_t)c.cur], nullptr, 16);
        double ideal = 1 / sqrt((double)c.frames);
        printf("edges: frame %d with %d of the window: framesUsed %d, luma ratio %.4f (ideal %.4f)\n", c.cur,
               c.frames - 1, st.framesUsed, e.rms[0] / in.rms[0], ideal);
        CHECK(st.framesUsed == c.frames, "framesUsed %d, expected %d", st.framesUsed, c.frames);
        CHECK(e.rms[0] / in.rms[0] < ideal * 1.10, "luma ratio %.4f, ideal %.4f", e.rms[0] / in.rms[0], ideal);
      }
      od_session_destroy(s);
    }
  }
  // odd, small and tiny sizes; out-of-range values
  for (auto wh : {std::pair<int, int>{1921, 1081}, {301, 173}, {97, 33}, {40, 64}, {31, 200}, {8, 8}}) {
    const int W = wh.first, H = wh.second;
    std::vector<Img> noisy, clean;
    makeSequence(&noisy, &clean, 3, W, H, kSigma, 131, 9000 + W, [](int) {
      return Scene([](float x, float y, float* rgb) {
        sceneA(x, y, rgb);
        for (int c = 0; c < 3; ++c) rgb[c] = rgb[c] * 3.0f - 0.4f;  // negatives and super-whites
      });
    });
    for (int q = 0; q <= 1; ++q) {
      ODSession* s = od_session_create(gCtx);
      Img out;
      ODParams p = temporalOnly(&prof, 1, q);
      if (renderFrame(s, noisy, 1, window(1), p, true, &out)) {
        Err in = errorOf(noisy[1], clean[1], nullptr, 0), e = errorOf(out, clean[1], nullptr, 0);
        CHECK(e.finite, "%dx%d q=%d: non-finite output", W, H, q);
        CHECK(e.rms[0] <= in.rms[0] * 1.001, "%dx%d q=%d: rms %.5f worse than the input's %.5f", W, H, q, e.rms[0],
              in.rms[0]);
        if (q == 1)
          printf("edges: %dx%d: luma ratio %.4f (whole frame, borders included; ideal %.4f)\n", W, H,
                 e.rms[0] / in.rms[0], 1 / sqrt(3.0));
      }
      od_session_destroy(s);
    }
  }
  // no profile: pass-through; strength -> 0: continuous
  {
    const int W = 256, H = 192;
    std::vector<Img> noisy, clean;
    makeSequence(&noisy, &clean, 3, W, H, kSigma, 151, 9900, [](int) { return Scene(sceneA); });
    ODSession* s = od_session_create(gCtx);
    Img out;
    ODParams p = temporalOnly(nullptr, 1, 1);  // profile stage skipped and no manual profile
    if (renderFrame(s, noisy, 1, window(1), p, true, &out)) {
      size_t d = diffBytes(out, noisy[1]);
      CHECK(d == 0, "without a profile %zu values changed", d);
    }
    od_session_destroy(s);
    double prev = 0;
    for (float t : {0.001f, 0.05f, 0.25f, 1.0f}) {
      ODSession* s2 = od_session_create(gCtx);
      ODParams p2 = temporalOnly(&prof, 1, 1);
      p2.temporalStrength = t;
      if (renderFrame(s2, noisy, 1, window(1), p2, true, &out)) {
        double d = 0;
        for (size_t i = 0; i < out.px.size(); ++i) d += fabs(out.px[i] - noisy[1].px[i]);
        d /= (double)out.px.size();
        printf("edges: temporalStrength %.3f: mean absolute change %.6f\n", t, d);
        CHECK(d >= prev - 1e-6, "change is not monotonic in temporalStrength (%.6f after %.6f)", d, prev);
        if (t <= 0.001f) CHECK(d < 1e-5, "temporalStrength 0.001 still changes the frame by %.6f", d);
        prev = d;
      }
      od_session_destroy(s2);
    }
    // amounts: with spatial off, lumaStrength / chromaStrength scale the correction
    ODSession* s3 = od_session_create(gCtx);
    ODSession* s4 = od_session_create(gCtx);
    Img full, half;
    ODParams pf = temporalOnly(&prof, 1, 1), ph = pf;
    ph.lumaStrength = 0.5f;
    ph.chromaStrength = 0.0f;
    if (renderFrame(s3, noisy, 1, window(1), pf, true, &full) && renderFrame(s4, noisy, 1, window(1), ph, true, &half)) {
      double worstY = 0, worstC = 0;
      for (size_t i = 0; i < full.px.size() / 4; ++i) {
        float a[3], b[3], c[3];
        ycc(&noisy[1].px[i * 4], a);
        ycc(&full.px[i * 4], b);
        ycc(&half.px[i * 4], c);
        worstY = std::max(worstY, (double)fabsf((c[0] - a[0]) - 0.5f * (b[0] - a[0])));
        worstC = std::max(worstC, (double)std::max(fabsf(c[1] - a[1]), fabsf(c[2] - a[2])));
      }
      printf("edges: luma amount 0.5 / chroma amount 0: worst deviation from half the luma correction %.2e, worst "
             "chroma change %.2e\n",
             worstY, worstC);
      CHECK(worstY < 2e-3 && worstC < 2e-3, "amounts not applied: %.2e %.2e", worstY, worstC);
    }
    od_session_destroy(s3);
    od_session_destroy(s4);
  }
}

static void testView() {
  gTest = "view";
  const int W = 512, H = 384, N = 5, C = 2;
  ODNoiseProfile prof = whiteProfile(kSigma);
  std::vector<Img> noisy, clean;
  makeSequence(&noisy, &clean, N, W, H, kSigma, 171, 9950, [](int) { return Scene(sceneA); });
  ODSession* s = od_session_create(gCtx);
  Img out;
  ODParams p = temporalOnly(&prof, 2, 1);
  p.view = OD_VIEW_MOTION;
  if (renderFrame(s, noisy, C, window(2), p, true, &out)) {
    double sum = 0, mn = 1e9;
    bool finite = true;
    size_t n = 0;
    for (int y = 16; y < H - 16; ++y)
      for (int x = 16; x < W - 16; ++x) {
        float v[3];
        ycc(&out.px[((size_t)y * W + x) * 4], v);
        finite = finite && std::isfinite(v[0]);
        sum += v[0];
        mn = std::min(mn, (double)v[0]);
        ++n;
      }
    printf("view: motion view on a static scene: mean share of neighbours merged %.4f, minimum %.4f\n",
           sum / (double)n, mn);
    CHECK(finite, "non-finite motion view");
    CHECK(sum / (double)n > 0.93, "static scene: mean merged share %.4f", sum / (double)n);
  }
  od_session_destroy(s);
}

// --------------------------------------------------------------------------- darkflat:
// noise that depends on luminance, as in log footage, where the darks are several times
// noisier than the rest. A static, flat, dark frame whose profile is exact at the picture's
// luminance (knots 0 and 1) and `ratio` times lower at the brighter knots 2..7, with or
// without a bright textured band whose noise follows those knots. The motion search, the
// Lucas-Kanade damping and the scene-cut statistics must use the noise at each block's own
// luminance: with one noise figure for the whole frame (the median of the knots, which is
// the quiet end here) the vectors in the dark were noise and whole neighbours were dropped
// as another shot. Memory: 7 frames of 1920x1080 float32 plus their clean copies, about 470
// MB. ---------------------------------------------------------------------------
static void testDarkFlat() {
  gTest = "darkflat";
  const int W = 1920, H = 1080, N = 7, R = 2;
  const float s = kSigma, darkY = 0.07f;
  const int band = H / 5;  // the bright band: the top fifth of the frame
  struct Case {
    float ratio;
    bool bright;
  } cases[] = {{1.0f, false}, {2.0f, false}, {2.65f, false}, {3.4f, true}};
  for (const Case& cs : cases) {
    // The profile: white noise of sigma s at knots 0 and 1, s / ratio above.
    ODNoiseProfile prof = whiteProfile(s);
    for (int c = 0; c < 3; ++c)
      for (int l = 2; l < OD_LEVELS; ++l) {
        for (int b = 0; b < OD_BANDS; ++b) prof.sigma[c][b][l] /= cs.ratio;
        prof.sigmaFull[c][l] /= cs.ratio;
      }
    std::vector<Img> noisy, clean;
    noisy.assign((size_t)N, Img());
    clean.assign((size_t)N, Img());
    for (int t = 0; t < N; ++t) {
      Img& c = clean[(size_t)t];
      c.alloc(W, H);
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
          float rgb[3] = {darkY, darkY, darkY};
          if (cs.bright && y < band) sceneA((float)x, (float)y + 260.0f, rgb);  // the textured stripes of scene A
          float* o = &c.px[((size_t)y * W + x) * 4];
          o[0] = rgb[2];
          o[1] = rgb[1];
          o[2] = rgb[0];
          o[3] = 1.0f;
        }
      Img& n = noisy[(size_t)t];
      n = c;
      n.buf = nil;
      Img dark = n, lit = n;
      addNoise(&dark, s, 7001u + (uint32_t)t * 7919u);
      if (cs.bright) {
        addNoise(&lit, s / cs.ratio, 9001u + (uint32_t)t * 7919u);
        for (int y = 0; y < band; ++y)
          memcpy(&dark.px[(size_t)y * W * 4], &lit.px[(size_t)y * W * 4], (size_t)W * 4 * sizeof(float));
      }
      n.px = dark.px;
      n.key = 30000 + (int64_t)(cs.ratio * 100) * 10 + t;
      n.upload();
    }
    // Measured on the dark area, away from the band and the frame border.
    auto inDark = [&](int x, int y) { return x >= 32 && x < W - 32 && y >= (cs.bright ? band + 48 : 32) && y < H - 32; };
    std::vector<uint8_t> mask((size_t)W * H, 0);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) mask[(size_t)y * W + x] = inDark(x, y) ? 1 : 0;
    for (int q = 0; q <= 1; ++q) {
      double minShare = 1, worstNoise = 0, worstVec = 0;
      for (int cur = R; cur < N - R; ++cur) {
        ODSession* ses = od_session_create(gCtx);
        ODParams p = temporalOnly(&prof, R, q);
        Img out, view;
        bool ok = renderFrame(ses, noisy, cur, window(R), p, true, &out);
        p.view = OD_VIEW_MOTION;
        ok = ok && renderFrame(ses, noisy, cur, window(R), p, true, &view);
        od_session_destroy(ses);
        if (!ok) continue;
        const double noise = errorOf(out, clean[(size_t)cur], &mask, 0).rms[0] /
                             errorOf(noisy[(size_t)cur], clean[(size_t)cur], &mask, 0).rms[0];
        double share = 0;
        size_t n = 0;
        std::vector<float> len;
        for (int y = 0; y < H; y += 2)
          for (int x = 0; x < W; x += 2) {
            if (!inDark(x, y)) continue;
            float v[3];
            ycc(&view.px[((size_t)y * W + x) * 4], v);
            share += v[0];
            ++n;
            if (x % 8 == 4 && y % 8 == 4) len.push_back(64.0f * hypotf(v[1], v[2]));  // one per 8x8 block
          }
        share /= (double)std::max<size_t>(n, 1);
        std::nth_element(len.begin(), len.begin() + (long)len.size() / 2, len.end());
        const double vec = len.empty() ? 0.0 : len[len.size() / 2];
        minShare = std::min(minShare, share);
        worstNoise = std::max(worstNoise, noise);
        worstVec = std::max(worstVec, vec);
      }
      printf("darkflat: ratio %.2f%s, quality %d: share of neighbours merged >= %.3f, noise left <= %.3f (ideal %.3f), "
             "median vector <= %.2f px\n",
             cs.ratio, cs.bright ? " with a bright band" : "", q, minShare, worstNoise, 1 / sqrt(2.0 * R + 1), worstVec);
      CHECK(minShare > 0.95, "ratio %.2f q%d: only %.3f of the neighbours merged in a static dark frame", cs.ratio, q,
            minShare);
      CHECK(worstNoise < 0.50, "ratio %.2f q%d: noise left %.3f (ideal 0.447)", cs.ratio, q, worstNoise);
      CHECK(worstVec < 1.0, "ratio %.2f q%d: median vector %.2f px in a static frame", cs.ratio, q, worstVec);
    }
  }
}

// --------------------------------------------------------------------------- mottle: what
// the temporal and spatial stages leave together on a static flat frame, by scale. Pixel
// noise used to drop by a factor 10 but noise at 16 to 32 px by only 3 to 4, so the residual
// read as slow boiling once the shot was lifted in the grade. Part of the noise persists
// between frames here (as codec noise and fixed pattern do), which the temporal merge cannot
// average; the spatial stage's coarse luma levels must not count on it having done so. Luma
// amount 100 %, temporal radius 2, the exact profile of the per-frame part.
// --------------------------------------------------------------------------- Measured 12
static const double kMottle16 = 2.4, kMottle32 = 2.4;
static void testMottle() {
  gTest = "mottle";
  const int W = 512, H = 384, N = 9, R = 2;
  const float s = 0.02f;
  // Low-frequency noise (blurred white noise, about 10 px), rms half the
  // per-frame white noise, correlated 0.7 from frame to frame.
  auto lowField = [&](uint32_t seed) {
    Img f;
    f.alloc(W, H);
    addNoise(&f, 1.0f, seed);
    std::vector<float> a((size_t)W * H), b((size_t)W * H);
    for (size_t i = 0; i < a.size(); ++i) a[i] = f.px[i * 4 + 1];
    for (int pass = 0; pass < 4; ++pass) {  // box 9 four times
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
          float acc = 0;
          for (int d = -4; d <= 4; ++d) acc += a[(size_t)y * W + std::min(W - 1, std::max(0, x + d))];
          b[(size_t)y * W + x] = acc / 9.0f;
        }
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
          float acc = 0;
          for (int d = -4; d <= 4; ++d) acc += b[(size_t)std::min(H - 1, std::max(0, y + d)) * W + x];
          a[(size_t)y * W + x] = acc / 9.0f;
        }
    }
    double ss = 0;
    for (float v : a) ss += (double)v * v;
    const float k = (float)(1.0 / sqrt(ss / (double)a.size()));
    for (float& v : a) v *= k;
    return a;
  };
  const std::vector<float> common = lowField(4242);
  std::vector<Img> noisy, clean;
  makeSequence(&noisy, &clean, N, W, H, s, 777, 32000, [](int) {
    return Scene([](float, float, float* rgb) { rgb[0] = rgb[1] = rgb[2] = 0.3f; });
  });
  for (int t = 0; t < N; ++t) {
    const std::vector<float> own = lowField(5000u + (uint32_t)t);
    Img& f = noisy[(size_t)t];
    for (size_t i = 0; i < (size_t)W * H; ++i) {
      const float v = 0.5f * s * (0.7f * common[i] + 0.71414f * own[i]);
      for (int c = 0; c < 3; ++c) f.px[i * 4 + (size_t)c] += v;
    }
    f.upload();
  }
  const int scales[4] = {1, 8, 16, 32};
  // Temporal sd of s x s cell means of the luma residual over the frames
  // rendered, rms over the cells (1/1023 units).
  auto flicker = [&](const std::vector<Img>& frames, int sc) {
    const int cw = (W - 32) / sc, ch = (H - 32) / sc, n = (int)frames.size();
    std::vector<double> sum((size_t)cw * ch, 0.0), sum2((size_t)cw * ch, 0.0);
    for (const Img& f : frames)
      for (int cy = 0; cy < ch; ++cy)
        for (int cx = 0; cx < cw; ++cx) {
          double m = 0;
          for (int y = 0; y < sc; ++y)
            for (int x = 0; x < sc; ++x) {
              float v[3];
              ycc(&f.px[((size_t)(16 + cy * sc + y) * W + (16 + cx * sc + x)) * 4], v);
              m += v[0] - 0.3;
            }
          m /= (double)(sc * sc);
          sum[(size_t)cy * cw + cx] += m;
          sum2[(size_t)cy * cw + cx] += m * m;
        }
    double acc = 0;
    for (size_t i = 0; i < sum.size(); ++i) acc += sum2[i] / n - (sum[i] / n) * (sum[i] / n);
    return 1023.0 * sqrt(acc / (double)sum.size());
  };
  std::vector<Img> in(noisy.begin() + R, noisy.end() - R);
  for (int q = 0; q <= 1; ++q) {
    std::vector<Img> out;
    ODSession* ses = od_session_create(gCtx);
    for (int cur = R; cur < N - R; ++cur) {
      // Temporal and spatial with the automatic profile (blocking renders, so
      // it is deterministic): the coarse bands of this noise are not white.
      ODParams p = temporalOnly(nullptr, R, q);
      p.debugSkipStages = 1u << OD_STAGE_SHARPEN;
      p.spatialStrength = 0.5f;
      Img o;
      if (renderFrame(ses, noisy, cur, window(R), p, true, &o)) out.push_back(o);
    }
    od_session_destroy(ses);
    if (out.size() != in.size()) continue;
    double red[4];
    printf("mottle, quality %d: residual flicker by scale (input / output, 1/1023):", q);
    for (int k = 0; k < 4; ++k) {
      const double a = flicker(in, scales[k]), b = flicker(out, scales[k]);
      red[k] = a / std::max(b, 1e-9);
      printf("  %d px %.2f / %.2f (x%.2f)", scales[k], a, b, red[k]);
    }
    printf("\n");
    CHECK(red[0] > 4.0, "q%d: pixel noise reduced only x%.2f", q, red[0]);
    CHECK(red[2] > kMottle16, "q%d: noise at 16 px reduced only x%.2f", q, red[2]);
    CHECK(red[3] > kMottle32, "q%d: noise at 32 px reduced only x%.2f", q, red[3]);
  }
}

// --------------------------------------------------------------------------- floor: the
// luma grain floor (stage.h, kGrainFloor) belongs to the amounts, so it applies whichever
// stage applies them. It used to exist in the spatial stage only: on low-noise footage
// turning the spatial slider from 0 to anything above cut what the temporal merge removed by
// up to half, against stage.h section 1 (continuous as a strength goes to 0). Spatial
// strength 0 (temporal applies the amounts) and 1e-6 (spatial applies them, filtering
// nothing) must give the same picture, the profile, sharpen and chroma aside.
// ---------------------------------------------------------------------------
static void testFloor() {
  gTest = "floor";
  const int W = 512, H = 288, N = 5, C = 2;
  for (float sigma : {0.0012f, 0.002f, 0.02f}) {
    std::vector<Img> noisy, clean;
    makeSequence(&noisy, &clean, N, W, H, sigma, 61, 31000 + (int64_t)(sigma * 1e5), [](int) {
      return Scene([](float x, float y, float* rgb) {
        // A clean, dark gradient: the corrections compared here are a
        // fraction of a 10-bit code, and the half-float working texture must
        // be fine enough to carry them (its step is 6e-5 below 0.125, 4.9e-4
        // above 0.5, where the double rounding of the spatial path alone
        // takes 10 to 20 % off such a correction).
        const float v = 0.04f + 0.06f * x / 512.0f + 0.01f * y / 288.0f;
        rgb[0] = rgb[1] = rgb[2] = v;
      });
    });
    ODNoiseProfile prof = whiteProfile(sigma);
    for (float luma : {0.6f, 1.0f})
      for (int q = 0; q <= 1; ++q) {
        Img out[2];
        for (int k = 0; k < 2; ++k) {
          ODParams p = temporalOnly(&prof, 2, q);
          p.lumaStrength = luma;
          p.chromaStrength = 0;
          p.spatialStrength = k ? 1e-6f : 0.0f;
          p.debugSkipStages = (1u << OD_STAGE_PROFILE) | (1u << OD_STAGE_SHARPEN);
          ODSession* s = od_session_create(gCtx);
          renderFrame(s, noisy, C, window(2), p, true, &out[k]);
          od_session_destroy(s);
        }
        double rem[2] = {0, 0}, diff = 0;
        size_t n = 0;
        for (int y = 16; y < H - 16; ++y)
          for (int x = 16; x < W - 16; ++x) {
            const size_t i = ((size_t)y * W + x) * 4;
            float a[3], b[3], in[3];
            ycc(&out[0].px[i], a);
            ycc(&out[1].px[i], b);
            ycc(&noisy[C].px[i], in);
            rem[0] += (double)(a[0] - in[0]) * (a[0] - in[0]);
            rem[1] += (double)(b[0] - in[0]) * (b[0] - in[0]);
            diff += (double)(a[0] - b[0]) * (a[0] - b[0]);
            ++n;
          }
        for (double& r : rem) r = sqrt(r / (double)n);
        diff = sqrt(diff / (double)n);
        printf("floor: sigma %.4f luma %.1f q%d: luma removed %.6f at spatial 0, %.6f at spatial 1e-6; rms difference "
               "%.6f\n",
               sigma, luma, q, rem[0], rem[1], diff);
        CHECK(fabs(rem[1] / std::max(rem[0], 1e-12) - 1.0) < 0.03,
              "sigma %.4f luma %.1f q%d: removed %.6f at spatial 0 against %.6f just above", sigma, luma, q, rem[0], rem[1]);
        CHECK(diff < 0.1 * std::max(rem[0], rem[1]) + 1e-4, "sigma %.4f luma %.1f q%d: the pictures differ by %.6f rms",
              sigma, luma, q, diff);
      }
  }
}

// pin: fixed samples and whole-frame statistics for deterministic scenes, with separate
// legacy and finest-grain references. These expectations must remain unchanged for behavior-
// preserving edits. OD_PIN_WRITE writes a candidate reference for explicit output changes.
struct PinCase {
  const char* name;
  int w, h, radius, quality;
  int before, after;   // frames available before and after the current one
  float vx, vy;        // pan, pixels per frame
  float objSpeed;      // speed of the occluding object, 0 = none
  int cutAfter;        // frames after the current one that belong to the same shot (99 = no cut)
  int view;
  float temporalStrength, luma, chroma;
};

static const PinCase kPinCases[] = {
    {"static-r2-q0", 512, 384, 2, 0, 2, 2, 0.0f, 0.0f, 0.0f, 99, OD_VIEW_OUTPUT, 1.0f, 1.0f, 1.0f},
    {"pan-object-r2-q0", 640, 480, 2, 0, 2, 2, 1.37f, 0.61f, 9.0f, 99, OD_VIEW_OUTPUT, 1.0f, 1.0f, 1.0f},
    {"pan-object-r2-q1", 640, 480, 2, 1, 2, 2, 1.37f, 0.61f, 9.0f, 99, OD_VIEW_OUTPUT, 1.0f, 1.0f, 1.0f},
    {"pan-object-r1-q0", 640, 480, 1, 0, 1, 1, -0.8f, 2.3f, 5.0f, 99, OD_VIEW_OUTPUT, 1.0f, 1.0f, 1.0f},
    {"big-1080p-r1-q0", 1920, 1080, 1, 0, 1, 1, 24.3f, 6.6f, 0.0f, 99, OD_VIEW_OUTPUT, 1.0f, 1.0f, 1.0f},
    {"tall-2064-r1-q1", 2112, 2064, 1, 1, 1, 1, 37.0f, -11.5f, 0.0f, 99, OD_VIEW_OUTPUT, 1.0f, 1.0f, 1.0f},
    {"r3-q0", 512, 384, 3, 0, 3, 3, -2.2f, 0.8f, 6.0f, 99, OD_VIEW_OUTPUT, 1.0f, 1.0f, 1.0f},
    {"r5-q1", 384, 256, 5, 1, 5, 5, 0.7f, -0.4f, 4.0f, 99, OD_VIEW_OUTPUT, 1.0f, 1.0f, 1.0f},
    {"one-sided-odd-q1", 501, 333, 2, 1, 0, 2, 1.37f, 0.61f, 9.0f, 99, OD_VIEW_OUTPUT, 1.0f, 1.0f, 1.0f},
    {"three-neighbours-q0", 501, 333, 2, 0, 1, 2, -3.1f, 1.2f, 9.0f, 99, OD_VIEW_OUTPUT, 1.0f, 1.0f, 1.0f},
    {"cut-r2-q0", 512, 384, 2, 0, 2, 2, 1.0f, 0.0f, 0.0f, 1, OD_VIEW_OUTPUT, 1.0f, 1.0f, 1.0f},
    {"amounts-r2-q0", 640, 480, 2, 0, 2, 2, 1.37f, 0.61f, 9.0f, 99, OD_VIEW_OUTPUT, 0.6f, 0.5f, 0.8f},
    {"motion-view-r2-q0", 640, 480, 2, 0, 2, 2, 1.37f, 0.61f, 9.0f, 99, OD_VIEW_MOTION, 1.0f, 1.0f, 1.0f},
};
constexpr int kPinCaseCount = (int)(sizeof(kPinCases) / sizeof(kPinCases[0]));
constexpr int kPinSamples = 48;

struct PinStats {
  double mean[3];  // B, G, R of the output
  double rms[3];   // B, G, R of output - input
  float sample[kPinSamples][3];
};

struct PinRef {
  const char* name;
  PinStats st;
};
static const PinRef kPinRef[] = {
#include "pin_reference.inc"
};
static const PinRef kPinRefV2[] = {
#include "pin_reference_v2.inc"
};

// Tolerances. A step of the 16-bit float working texture is 2.4e-4 for values
// in [0.25, 0.5) and 4.9e-4 in [0.5, 1), and a step in Cb or Cr moves B or R
// by up to 1.86 times that; the scenes stay below 1. So a pixel may move by
// about 1e-3 through the rounding of the result alone. The whole-frame
// statistics average that rounding out: at full quality the optimised stage
// does the same arithmetic in another order and they must not move. Playback
// quality also rounds the differences between the frames more coarsely than
// before (temporal.mm, "Speed"), which moves a statistic by up to 1e-5. The
// motion view shows the frame count, whose own step is 1e-3 in that view.
constexpr double kPinPixelTol = 1.0e-3;
constexpr double kPinViewTol = 2.0e-3;
constexpr double kPinStatTolFull = 1.0e-6;
constexpr double kPinStatTolPlayback = 2.0e-5;

static void pinSamplePos(int ci, int k, int w, int h, int* x, int* y) {
  uint32_t a = hash32((uint32_t)ci * 0x9E3779B9u + (uint32_t)k * 0x85EBCA6Bu + 12345u);
  uint32_t b = hash32(a ^ 0xC2B2AE35u);
  *x = (int)(a % (uint32_t)w);
  *y = (int)(b % (uint32_t)h);
}

static bool pinRender(int ci, Img* in, Img* out) {
  const PinCase& c = kPinCases[ci];
  const int n = c.before + 1 + c.after, cur = c.before;
  std::vector<Img> noisy, clean;
  makeSequence(&noisy, &clean, n, c.w, c.h, kSigma, 900u + (uint32_t)ci * 31u, 20000 + ci * 100, [=](int t) {
    const float dx = c.vx * (float)(t - cur), dy = c.vy * (float)(t - cur);
    const float cx = 0.45f * (float)c.w + (c.objSpeed + c.vx) * (float)(t - cur);
    const float cy = 0.5f * (float)c.h + (0.35f * c.objSpeed + c.vy) * (float)(t - cur);
    const float radius = 0.09f * (float)c.w;
    const bool otherShot = t - cur > c.cutAfter;
    const bool object = c.objSpeed != 0.0f;
    return Scene([=](float x, float y, float* rgb) {
      if (otherShot) {
        sceneB(x, y, rgb);
        return;
      }
      sceneA(x - dx, y - dy, rgb);
      if (!object) return;
      float a = softStep(radius - hypotf(x - cx, y - cy), 2.5f);
      if (a > 0) {
        float u = x - cx, v = y - cy;
        float o = 0.48f + 0.04f * sinf(0.5f * u) * sinf(0.45f * v) + 0.03f * sinf(0.11f * u + 0.07f * v);
        const float tint[3] = {1.05f, 1.0f, 0.92f};
        for (int ch = 0; ch < 3; ++ch) rgb[ch] = rgb[ch] * (1.0f - a) + a * o * tint[ch];
      }
    });
  });
  ODNoiseProfile prof = whiteProfile(kSigma);
  ODParams p = temporalOnly(&prof, c.radius, c.quality);
  p.view = c.view;
  p.temporalStrength = c.temporalStrength;
  p.lumaStrength = c.luma;
  p.chromaStrength = c.chroma;
  ODSession* s = od_session_create(gCtx);
  const bool ok = renderFrame(s, noisy, cur, window(c.radius), p, true, out);
  od_session_destroy(s);
  *in = noisy[(size_t)cur];
  return ok;
}

static PinStats pinStatsOf(int ci, const Img& in, const Img& out) {
  PinStats st;
  memset(&st, 0, sizeof(st));
  const size_t n = (size_t)out.w * (size_t)out.h;
  for (size_t i = 0; i < n; ++i)
    for (int ch = 0; ch < 3; ++ch) {
      const double o = out.px[i * 4 + (size_t)ch], d = o - in.px[i * 4 + (size_t)ch];
      st.mean[ch] += o;
      st.rms[ch] += d * d;
    }
  for (int ch = 0; ch < 3; ++ch) {
    st.mean[ch] /= (double)n;
    st.rms[ch] = sqrt(st.rms[ch] / (double)n);
  }
  for (int k = 0; k < kPinSamples; ++k) {
    int x, y;
    pinSamplePos(ci, k, out.w, out.h, &x, &y);
    for (int ch = 0; ch < 3; ++ch) st.sample[k][ch] = out.px[((size_t)y * (size_t)out.w + (size_t)x) * 4 + (size_t)ch];
  }
  return st;
}

static void testPin() {
  gTest = "pin";
  const char* writePath = getenv("OD_PIN_WRITE");
  const char* dumpDir = getenv("OD_PIN_DUMP");
  const char* compareDir = getenv("OD_PIN_COMPARE");
  FILE* wf = nullptr;
  if (writePath) {
    wf = fopen(writePath, "w");
    if (!wf) {
      CHECK(false, "cannot write %s", writePath);
      return;
    }
    fprintf(wf, "// SPDX-License-Identifier: MIT\n"
                "// Reference statistics of the temporal stage for the `pin` test of test_temporal.mm,\n"
                "// which says what they are and how this file is regenerated. Generated, do not edit.\n");
  }
  const char* mode = getenv("OD_GRAIN_PROFILE_V2");
  const bool v2 = mode && strcmp(mode, "1") == 0;
  const PinRef* reference = v2 ? kPinRefV2 : kPinRef;
  const size_t count = v2 ? sizeof(kPinRefV2) / sizeof(kPinRefV2[0]) : sizeof(kPinRef) / sizeof(kPinRef[0]);
  const bool haveRef = count == (size_t)kPinCaseCount;
  if (!wf) CHECK(haveRef, "temporal reference has %zu cases, the test has %d", count, kPinCaseCount);
  double worstPixel = 0, worstStat = 0;
  for (int ci = 0; ci < kPinCaseCount; ++ci) {
    const PinCase& c = kPinCases[ci];
    Img in, out;
    if (!pinRender(ci, &in, &out)) continue;
    const PinStats st = pinStatsOf(ci, in, out);
    char path[1024];
    if (dumpDir) {
      snprintf(path, sizeof(path), "%s/%s.raw", dumpDir, c.name);
      FILE* f = fopen(path, "wb");
      if (f) {
        fwrite(out.px.data(), sizeof(float), out.px.size(), f);
        fclose(f);
      }
    }
    if (compareDir) {
      snprintf(path, sizeof(path), "%s/%s.raw", compareDir, c.name);
      FILE* f = fopen(path, "rb");
      std::vector<float> ref(out.px.size());
      if (f && fread(ref.data(), sizeof(float), ref.size(), f) == ref.size()) {
        double mx = 0, ss = 0, sd = 0;
        size_t differ = 0, above = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
          const double d = fabs((double)out.px[i] - (double)ref[i]);
          sd += (double)out.px[i] - (double)ref[i];
          mx = std::max(mx, d);
          ss += d * d;
          differ += d != 0;
          above += d > kPinPixelTol;
        }
        printf("pin compare %-22s max %.3e  rms %.3e  mean %+.2e  values differing %.3f%%  above %.0e: %zu\n", c.name,
               mx, sqrt(ss / (double)ref.size()), sd / (double)ref.size(), 100.0 * (double)differ / (double)ref.size(),
               kPinPixelTol, above);
      } else {
        printf("pin compare %-22s no reference frame in %s\n", c.name, compareDir);
      }
      if (f) fclose(f);
    }
    if (wf) {
      fprintf(wf, "{\"%s\", {{%.9g, %.9g, %.9g}, {%.9g, %.9g, %.9g}, {", c.name, st.mean[0], st.mean[1], st.mean[2],
              st.rms[0], st.rms[1], st.rms[2]);
      for (int k = 0; k < kPinSamples; ++k)
        fprintf(wf, "%s{%.9gf, %.9gf, %.9gf},", k % 4 == 0 ? "\n    " : " ", st.sample[k][0], st.sample[k][1],
                st.sample[k][2]);
      fprintf(wf, "}}},\n");
      continue;
    }
    if (!haveRef) continue;
    const PinRef& r = reference[ci];
    CHECK(strcmp(r.name, c.name) == 0, "pin_reference.inc: case %d is %s, expected %s", ci, r.name, c.name);
    double dPixel = 0, dStat = 0;
    for (int k = 0; k < kPinSamples; ++k)
      for (int ch = 0; ch < 3; ++ch) dPixel = std::max(dPixel, fabs((double)st.sample[k][ch] - (double)r.st.sample[k][ch]));
    for (int ch = 0; ch < 3; ++ch) {
      dStat = std::max(dStat, fabs(st.mean[ch] - r.st.mean[ch]));
      dStat = std::max(dStat, fabs(st.rms[ch] - r.st.rms[ch]));
    }
    const double tolPixel = c.view == OD_VIEW_OUTPUT ? kPinPixelTol : kPinViewTol;
    const double tolStat = c.quality ? kPinStatTolFull : kPinStatTolPlayback;
    printf("pin %-22s pixels within %.2e (tolerance %.0e), statistics within %.2e (tolerance %.0e)\n", c.name, dPixel,
           tolPixel, dStat, tolStat);
    CHECK(dPixel <= tolPixel, "%s: a pinned pixel moved by %.3e (tolerance %.1e)", c.name, dPixel, tolPixel);
    CHECK(dStat <= tolStat, "%s: a whole-frame statistic moved by %.3e (tolerance %.1e)", c.name, dStat, tolStat);
    worstPixel = std::max(worstPixel, dPixel / tolPixel);
    worstStat = std::max(worstStat, dStat / tolStat);
  }
  if (wf) {
    fclose(wf);
    printf("pin: wrote %s (%d cases)\n", writePath, kPinCaseCount);
  } else {
    printf("pin: %d cases; the largest difference is %.0f%% of its tolerance for the pixels, %.0f%% for the statistics\n",
           kPinCaseCount, 100.0 * worstPixel, 100.0 * worstStat);
  }
}

int main(int argc, char** argv) {
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
    char err[1024];
    gCtx = od_context_create((__bridge void*)gDevice, nullptr, err, sizeof(err));
    if (!gCtx) {
      fprintf(stderr, "od_context_create: %s\n", err);
      return 2;
    }
    std::string only = argc > 1 ? argv[1] : "";
    auto want = [&](const char* n) { return only.empty() || only == n; };
    if (want("static")) testStatic();
    if (want("pan")) {
      testPan("pan", 512, 384, 1.37f, 0.61f, 0.52, 0.56);
      testPan("pan-integer", 512, 384, 3.0f, -1.0f, 0.50, 0.50);
    }
    if (want("bigmotion")) testPan("bigmotion", 1920, 1080, 24.3f, 6.6f, 0.56, 0.60);
    if (want("occlusion")) {
      testOcclusion("occlusion-strong", 0.30f, 14.0f);
      testOcclusion("occlusion-faint", 0.035f, 14.0f);
      testOcclusion("occlusion-slow", 0.12f, 3.0f);
      testOcclusion("occlusion-superwhite", 6.0f, 14.0f);  // an object at 6.3: values far above 1 are legal
      testOcclusion("occlusion-chroma", 0.0f, 14.0f, true);  // differs from the background in colour only
    }
    if (want("thin_lines")) testThinLines();
    if (want("residual_guards") || only == "residual_extended") {
      gResidualModel = getenv("OD_TEST_LEGACY_GUARDS") == nullptr;
      const std::vector<int> radii = only == "residual_extended" ? std::vector<int>{1,3,5} : std::vector<int>{2};
      for (int r : radii) {
        testThinLines(r);
        testOcclusion("residual-slow", .12f, 3.0f, false, r);
        testOcclusion("residual-fast", .30f, 14.0f, false, r);
        testOcclusion("residual-chroma", 0.0f, 14.0f, true, r);
        testOcclusion("residual-superwhite", 6.0f, 14.0f, false, r);
      }
      gResidualModel = false;
    }
    if (want("cut")) testCut();
    if (want("repeat")) testRepeat();
    if (want("repeat_chroma")) testRepeatChroma();
    if (want("lighting")) testLighting();
    if (want("noise_level")) testNoiseLevel();
    if (want("cache")) testCache();
    if (want("edges")) testEdges();
    if (want("view")) testView();
    if (want("darkflat")) testDarkFlat();
    if (want("floor")) testFloor();
    if (want("mottle")) testMottle();
    if (want("pin")) testPin();
    od_context_destroy(gCtx);
    printf("temporal tests: %d checks, %d failures\n", gChecks, gFailures);
    return gFailures ? 1 : 0;
  }
}
