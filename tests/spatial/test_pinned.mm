// SPDX-License-Identifier: MIT
// Pins spatial, sharpening and automatic-profile output on deterministic synthetic frames.
// Cases compare output samples and whole-frame statistics with the fixed legacy and finest-
// grain reference headers.
// The tolerances account for half-float rounding. Keep references unchanged when the
// intended picture is unchanged.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "opendenoise/opendenoise.h"

namespace {

int gFailures = 0;
int gChecks = 0;

uint32_t hash32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

// Unit Gaussian, deterministic in (x, y, c, seed).
float gauss(uint32_t x, uint32_t y, uint32_t c, uint32_t seed) {
  uint32_t h1 = hash32(x * 73856093u ^ y * 19349663u ^ c * 83492791u ^ seed * 2654435761u);
  uint32_t h2 = hash32(h1 ^ 0x9E3779B9u);
  float u1 = ((float)(h1 >> 8) + 1.0f) / 16777217.0f;
  float u2 = (float)(h2 >> 8) / 16777216.0f;
  return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

// Planar Y, Cb, Cr (the working space).
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

void yccToBgra(const Img& im, std::vector<float>& bgra) {
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
Img bgraToYcc(const std::vector<float>& bgra, int w, int h) {
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

// ---- the scene ---------------------------------------------------------------
// A frame with everything the stages treat differently: a luminance ramp (the
// profile depends on luminance), gratings of 3, 6 and 12 pixels, hard luma and
// chroma edges, a patch below black and one above white, coloured areas, and
// noise that grows with luminance. `u` is the scale (1 = 512x288): the scene
// keeps its pixel-scale detail and grows in area. Frame t is frame 0 moved by
// (1.25, 0.5) pixels per frame, with its own noise.
float noiseSigma(float y, float s0) { return s0 * (0.5f + std::min(1.0f, std::max(0.0f, y))); }

void cleanAt(float fx, float fy, int W, int H, float* out) {
  const float nx = fx / (float)W, ny = fy / (float)H;
  float Y = 0.04f + 0.8f * nx;
  float cb = 0.06f * sinf(4.0f * ny) * (1.0f - nx), cr = 0.05f * cosf(3.0f * nx + 1.0f);
  if (ny < 0.25f) {
    // gratings, wavelength 3, 6, 12 pixels, at four angles
    const int band = std::min(3, (int)(nx * 4.0f));
    const float wl = band == 0 ? 3.0f : band == 1 ? 6.0f : band == 2 ? 12.0f : 6.0f;
    const float ang = 0.6f * (float)band;
    Y += 0.05f * sinf(6.2831853f * (fx * cosf(ang) + fy * sinf(ang)) / wl);
  } else if (ny < 0.5f) {
    // hard luma steps, a thin line, a patch below black and one above white
    if (fmodf(fx, 97.0f) < 40.0f) Y += 0.25f;
    if (fabsf(fmodf(fx + 2.0f * fy, 131.0f) - 60.0f) < 0.75f) Y -= 0.2f;
    if (nx > 0.05f && nx < 0.12f && ny > 0.3f && ny < 0.45f) Y = -0.06f;
    if (nx > 0.85f && nx < 0.95f && ny > 0.3f && ny < 0.45f) Y = 1.35f;
  } else if (ny < 0.75f) {
    // chroma-only edges and saturated patches
    if (fmodf(fx, 113.0f) < 50.0f) cb += 0.12f;
    if (fmodf(fx + fy, 89.0f) < 30.0f) cr -= 0.10f;
  } else {
    // soft shading (what NL-means turns into patches)
    Y += 0.08f * sinf(0.045f * fx) * sinf(0.06f * fy);
  }
  out[0] = Y;
  out[1] = cb;
  out[2] = cr;
}

Img sceneFrame(int W, int H, int t, float s0, bool withNoise) {
  Img im(W, H);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      float v[3];
      cleanAt((float)x + 1.25f * (float)t, (float)y + 0.5f * (float)t, W, H, v);
      const float s = withNoise ? noiseSigma(v[0], s0) : 0.0f;
      const uint32_t seed = 1000u + (uint32_t)t;
      im.at(0, x, y) = v[0] + s * gauss((uint32_t)x, (uint32_t)y, 0, seed);
      im.at(1, x, y) = v[1] + 0.6f * s * gauss((uint32_t)x, (uint32_t)y, 1, seed);
      im.at(2, x, y) = v[2] + 0.6f * s * gauss((uint32_t)x, (uint32_t)y, 2, seed);
    }
  }
  return im;
}

// The profile of that noise (white in every channel).
ODNoiseProfile sceneProfile(float s0) {
  ODNoiseProfile p;
  memset(&p, 0, sizeof(p));
  const float gain[OD_BANDS] = {0.8908f, 0.2007f, 0.0855f, 0.0412f};
  for (int c = 0; c < 3; ++c)
    for (int l = 0; l < OD_LEVELS; ++l) {
      const float s = noiseSigma((float)l / (float)(OD_LEVELS - 1), s0) * (c ? 0.6f : 1.0f);
      for (int b = 0; b < OD_BANDS; ++b) p.sigma[c][b][l] = s * gain[b];
      p.sigmaFull[c][l] = 0.92f * s;
    }
  p.confidence = 1;
  p.valid = 1;
  return p;
}

// ---- the cases ---------------------------------------------------------------
enum Kind { kSpatial, kPipeline, kProfileFrame };
struct Case {
  const char* name;
  Kind kind;
  int w, h;
  int quality;
  float luma, chroma, spatial, sharpen;
  float noise;           // s0 of the scene
  const char* countEnv;  // OD_SPATIAL_TEST_COUNT for this case's session, or nullptr
};

const Case kCases[] = {
    // spatial (and sharpen) alone: temporal off, the scene's own profile
    {"spatial_q0_default", kSpatial, 512, 288, 0, 0.6f, 1.0f, 0.5f, 0.0f, 0.02f, nullptr},
    {"spatial_q1_default", kSpatial, 512, 288, 1, 0.6f, 1.0f, 0.5f, 0.0f, 0.02f, nullptr},
    {"spatial_q0_full_heavy", kSpatial, 512, 288, 0, 1.0f, 1.0f, 1.0f, 0.0f, 0.05f, nullptr},
    {"spatial_q1_full_heavy", kSpatial, 512, 288, 1, 1.0f, 1.0f, 1.0f, 0.0f, 0.05f, nullptr},
    {"spatial_q1_luma_only", kSpatial, 512, 288, 1, 1.0f, 0.0f, 0.5f, 0.0f, 0.02f, nullptr},
    {"spatial_q0_chroma_only", kSpatial, 512, 288, 0, 0.0f, 1.0f, 0.5f, 0.0f, 0.02f, nullptr},
    {"spatial_q0_count_split", kSpatial, 512, 288, 0, 1.0f, 1.0f, 0.5f, 0.0f, 0.02f, "split:4"},
    {"spatial_q1_count_4", kSpatial, 512, 288, 1, 1.0f, 1.0f, 0.5f, 0.0f, 0.02f, "4"},
    {"sharpen_q0_alone", kSpatial, 512, 288, 0, 0.6f, 1.0f, 0.0f, 1.0f, 0.01f, nullptr},
    {"sharpen_q1_alone", kSpatial, 512, 288, 1, 0.6f, 1.0f, 0.0f, 1.0f, 0.01f, nullptr},
    {"spatial_sharpen_q0", kSpatial, 512, 288, 0, 0.6f, 1.0f, 0.5f, 0.5f, 0.02f, nullptr},
    {"spatial_sharpen_q1_count", kSpatial, 512, 288, 1, 0.6f, 1.0f, 0.5f, 0.5f, 0.02f, "split:3"},
    {"odd_size_q1", kSpatial, 333, 187, 1, 1.0f, 1.0f, 0.5f, 0.3f, 0.02f, nullptr},
    {"odd_size_q0", kSpatial, 333, 187, 0, 1.0f, 1.0f, 0.5f, 0.3f, 0.02f, nullptr},
    {"tiny_q0", kSpatial, 37, 21, 0, 1.0f, 1.0f, 0.5f, 0.5f, 0.02f, nullptr},
    // the whole pipeline with the automatic profile: frames 0..5 rendered in
    // order (blocking, so the profile is deterministic), the output of frame 5
    {"pipeline_q0", kPipeline, 512, 288, 0, 0.6f, 1.0f, 0.5f, 0.0f, 0.02f, nullptr},
    {"pipeline_q1_sharpen", kPipeline, 512, 288, 1, 0.6f, 1.0f, 0.5f, 0.3f, 0.02f, nullptr},
    // the profiler alone on one frame
    {"profile_frame", kProfileFrame, 512, 288, 1, 0, 0, 0, 0, 0.02f, nullptr},
    {"profile_frame_heavy", kProfileFrame, 640, 360, 1, 0, 0, 0, 0, 0.05f, nullptr},
};
const int kCaseCount = (int)(sizeof(kCases) / sizeof(kCases[0]));

constexpr int kImageStats = 12 + 16;
constexpr int kProfileStats = 16;
constexpr int kMaxStats = kImageStats + kProfileStats;

#include "pinned_reference.h"  // legacy default
#include "pinned_reference_v2.h"  // deliberate opt-in picture

// Half-float intermediate rounding and equivalent operation ordering allow small differences
// from the fixed reference.
constexpr double kRelTol = 2e-3;
constexpr double kAbsTol = 2e-6;
// Rounding also decides the spatial stage's one yes-or-no rule (is this pixel
// an outlier of the noise?) for a pixel in a hundred thousand or so, and such
// a pixel then moves by up to a few hundredths. One of them changing by 0.01
// moves a statistic over n pixels by about this much / n: nothing for a
// frame, visible for the 45-pixel grid cells of the tiny case.
constexpr double kPixelTol = 0.0127;

// ---- running -----------------------------------------------------------------
id<MTLDevice> gDevice;
ODContext* gCtx;

struct Frame {
  id<MTLBuffer> buf;
  ODFrame f;
};
Frame makeFrame(const std::vector<float>& bgra, int w, int h, int64_t key) {
  Frame fr;
  fr.buf = [gDevice newBufferWithLength:bgra.size() * 4 options:MTLResourceStorageModeShared];
  memcpy(fr.buf.contents, bgra.data(), bgra.size() * 4);
  fr.f.mtlBuffer = (__bridge void*)fr.buf;
  fr.f.width = (uint32_t)w;
  fr.f.height = (uint32_t)h;
  fr.f.rowBytes = (uint32_t)w * 16;
  fr.f.format = OD_PIXEL_BGRA32F;
  fr.f.frameKey = key;
  return fr;
}

// `n` receives the number of pixels behind each statistic.
void imageStats(const Img& in, const Img& out, double* s, double* n) {
  const int w = in.w, h = in.h;
  for (int c = 0; c < 3; ++c) {
    double mean = 0, d2 = 0, g2 = 0, b2 = 0;
    size_t gn = 0, bn = 0;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const double o = out.at(c, x, y), d = o - in.at(c, x, y);
        mean += o;
        d2 += d * d;
        if (x + 1 < w) {
          const double g = out.at(c, x + 1, y) - o;
          g2 += g * g;
          ++gn;
        }
        if (y + 1 < h) {
          const double g = out.at(c, x, y + 1) - o;
          g2 += g * g;
          ++gn;
        }
        if (x < 2 || y < 2 || x >= w - 2 || y >= h - 2) {
          b2 += d * d;
          ++bn;
        }
      }
    }
    const double all = (double)w * h;
    s[c * 4 + 0] = mean / all;
    s[c * 4 + 1] = std::sqrt(d2 / all);
    s[c * 4 + 2] = std::sqrt(g2 / (double)std::max<size_t>(gn, 1));
    s[c * 4 + 3] = std::sqrt(b2 / (double)std::max<size_t>(bn, 1));
    n[c * 4 + 0] = n[c * 4 + 1] = n[c * 4 + 2] = all;
    n[c * 4 + 3] = (double)std::max<size_t>(bn, 1);
  }
  for (int gy = 0; gy < 4; ++gy) {
    for (int gx = 0; gx < 4; ++gx) {
      double d2 = 0;
      size_t cell = 0;
      for (int y = gy * h / 4; y < (gy + 1) * h / 4; ++y)
        for (int x = gx * w / 4; x < (gx + 1) * w / 4; ++x) {
          const double d = (double)out.at(0, x, y) - in.at(0, x, y);
          d2 += d * d;
          ++cell;
        }
      s[12 + gy * 4 + gx] = std::sqrt(d2 / (double)std::max<size_t>(cell, 1));
      n[12 + gy * 4 + gx] = (double)std::max<size_t>(cell, 1);
    }
  }
}

void profileStats(const ODNoiseProfile& p, double* s) {
  for (int c = 0; c < 3; ++c) {
    for (int b = 0; b <= OD_BANDS; ++b) {
      double a = 0;
      for (int l = 0; l < OD_LEVELS; ++l) a += std::log(std::max(1e-12, (double)(b < OD_BANDS ? p.sigma[c][b][l] : p.sigmaFull[c][l])));
      s[c * 5 + b] = std::exp(a / OD_LEVELS);
    }
  }
  s[15] = p.valid ? p.confidence : -1.0;
}

// Runs one case. Returns the number of statistics written to `stats`, or -1.
// pixels[i] is the number of pixels behind stats[i] (0 for profile numbers).
int runCase(const Case& c, int scale, double* stats, double* pixels, std::vector<float>* rawOut) {
  const int w = c.w * scale, h = c.h * scale;
  if (c.countEnv) setenv("OD_SPATIAL_TEST_COUNT", c.countEnv, 1);
  else unsetenv("OD_SPATIAL_TEST_COUNT");
  ODSession* s = od_session_create(gCtx);
  unsetenv("OD_SPATIAL_TEST_COUNT");
  if (!s) return -1;
  int n = -1;
  @autoreleasepool {
    std::vector<float> bgra;
    if (c.kind == kProfileFrame) {
      yccToBgra(sceneFrame(w, h, 0, c.noise, true), bgra);
      Frame f = makeFrame(bgra, w, h, OD_NO_KEY);
      ODNoiseProfile p;
      memset(&p, 0, sizeof(p));
      if (od_profile_frame(s, &f.f, &p) != 0) {
        fprintf(stderr, "FAIL [%s] od_profile_frame: %s\n", c.name, od_last_error(s));
      } else {
        profileStats(p, stats);
        n = kProfileStats;
        if (rawOut) {
          rawOut->assign(sizeof(p) / sizeof(float), 0.0f);
          memcpy(rawOut->data(), &p, sizeof(p));
        }
      }
    } else {
      const ODNoiseProfile prof = sceneProfile(c.noise);
      ODParams p = od_params_default();
      p.quality = c.quality;
      p.lumaStrength = c.luma;
      p.chromaStrength = c.chroma;
      p.spatialStrength = c.spatial;
      p.sharpen = c.sharpen;
      p.waitUntilCompleted = 1;
      const bool pipeline = c.kind == kPipeline;
      p.temporalRadius = pipeline ? 2 : 0;
      p.manualProfile = pipeline ? nullptr : &prof;
      const int frames = pipeline ? 8 : 1, last = pipeline ? 5 : 0;
      std::vector<Frame> fr;
      std::vector<Img> src;
      for (int t = 0; t < frames; ++t) {
        src.push_back(sceneFrame(w, h, t, c.noise, true));
        yccToBgra(src.back(), bgra);
        fr.push_back(makeFrame(bgra, w, h, pipeline ? 100 + t : OD_NO_KEY));
      }
      Frame out = makeFrame(bgra, w, h, OD_NO_KEY);
      ODStats st;
      bool ok = true;
      for (int t = 0; t <= last && ok; ++t) {
        ODFrame win[5];
        int off[5], cnt = 0;
        win[cnt] = fr[(size_t)t].f;
        off[cnt++] = 0;
        for (int d = 1; d <= p.temporalRadius; ++d) {
          for (int sgn = -1; sgn <= 1; sgn += 2) {
            const int u = t + sgn * d;
            if (u < 0 || u >= frames) continue;
            win[cnt] = fr[(size_t)u].f;
            off[cnt++] = sgn * d;
          }
        }
        memset(out.buf.contents, 0, out.buf.length);
        if (od_render(s, win, off, cnt, &p, &out.f, &st) != 0) {
          fprintf(stderr, "FAIL [%s] od_render: %s\n", c.name, od_last_error(s));
          ok = false;
        }
      }
      if (ok) {
        std::vector<float> o((size_t)w * h * 4);
        memcpy(o.data(), out.buf.contents, o.size() * 4);
        bool finite = true;
        for (float v : o) finite = finite && std::isfinite(v);
        if (!finite) {
          fprintf(stderr, "FAIL [%s] NaN or Inf in the output\n", c.name);
        } else {
          imageStats(src[(size_t)last], bgraToYcc(o, w, h), stats, pixels);
          n = kImageStats;
          if (pipeline) {
            profileStats(st.profile, stats + kImageStats);
            n += kProfileStats;
          }
          if (rawOut) rawOut->swap(o);
        }
      }
    }
  }
  od_session_destroy(s);
  return n;
}

const char* statName(const Case& c, int i, char* buf, size_t len) {
  static const char* ch[3] = {"Y", "Cb", "Cr"};
  static const char* what[4] = {"mean", "rms removed", "rms gradient", "rms removed at the border"};
  if (c.kind != kProfileFrame && i < 12) {
    snprintf(buf, len, "%s %s", ch[i / 4], what[i % 4]);
  } else if (c.kind != kProfileFrame && i < kImageStats) {
    snprintf(buf, len, "Y rms removed, grid cell %d,%d", (i - 12) % 4, (i - 12) / 4);
  } else {
    const int k = c.kind == kProfileFrame ? i : i - kImageStats;
    if (k == 15) snprintf(buf, len, "profile confidence");
    else if (k % 5 == 4) snprintf(buf, len, "profile %s full band", ch[k / 5]);
    else snprintf(buf, len, "profile %s band %d", ch[k / 5], k % 5);
  }
  return buf;
}

}  // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    // Both grain modes are exercised by `make test` (test.mk sets the
    // session override, OD_GRAIN_PROFILE_V2=0 and =1, so zeroed and default
    // ODParams render alike); run directly, the default mode (finest grain).
    setenv("OD_GRAIN_PROFILE_V2", "1", 0);
    const char* mode = std::getenv("OD_GRAIN_PROFILE_V2");
    const bool v2 = mode && std::strcmp(mode, "1") == 0;
    const int referenceCount = v2 ? kPinnedCountV2 : kPinnedCount;
    const auto* reference = v2 ? kPinnedV2 : kPinned;
    bool print = false;
    std::string dumpDir;
    int scale = 1;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--print") print = true;
      else if (a == "--dump" && i + 1 < argc) dumpDir = argv[++i];
      else if (a == "--scale" && i + 1 < argc) scale = atoi(argv[++i]);
      else {
        fprintf(stderr, "usage: test_pinned [--print | --dump DIR [--scale N]]\n");
        return 2;
      }
    }
    if (scale < 1 || scale > 8 || (scale != 1 && dumpDir.empty())) {
      fprintf(stderr, "test_pinned: --scale is 1..8 and only goes with --dump\n");
      return 2;
    }
    gDevice = MTLCreateSystemDefaultDevice();
    if (!gDevice) {
      fprintf(stderr, "test_pinned: no Metal device\n");
      return 2;
    }
    char err[512] = "";
    gCtx = od_context_create((__bridge void*)gDevice, nullptr, err, sizeof(err));
    if (!gCtx) {
      fprintf(stderr, "test_pinned: od_context_create: %s\n", err);
      return 2;
    }
    if (print) {
      printf("// SPDX-License-Identifier: MIT\n");
      printf("// Reference statistics for tests/spatial/test_pinned.mm, one row per case of\n");
      printf("// kCases, written by `test_pinned --print`. Do not edit by hand.\n");
      printf("static const int kPinnedCount%s = %d;\n", v2 ? "V2" : "", kCaseCount);
      printf("static const double kPinned%s[][kMaxStats] = {\n", v2 ? "V2" : "");
    }
    for (int k = 0; k < kCaseCount; ++k) {
      const Case& c = kCases[k];
      double stats[kMaxStats] = {0}, pixels[kMaxStats] = {0};
      std::vector<float> raw;
      const int n = runCase(c, scale, stats, pixels, dumpDir.empty() ? nullptr : &raw);
      ++gChecks;
      if (n < 0) {
        ++gFailures;
        continue;
      }
      if (print) {
        printf("    {  // %s\n     ", c.name);
        for (int i = 0; i < n; ++i) printf(" %.9g,%s", stats[i], (i % 4 == 3 && i + 1 < n) ? "\n     " : "");
        printf("\n    },\n");
        continue;
      }
      if (!dumpDir.empty()) {
        const std::string path = dumpDir + "/" + c.name + ".raw";
        FILE* f = fopen(path.c_str(), "wb");
        if (!f || fwrite(raw.data(), 4, raw.size(), f) != raw.size()) {
          fprintf(stderr, "test_pinned: cannot write %s\n", path.c_str());
          return 2;
        }
        fclose(f);
        if (scale != 1) continue;
      }
      if (k >= referenceCount) {
        ++gFailures;
        fprintf(stderr, "FAIL [%s] no reference row: regenerate pinned_reference.h\n", c.name);
        continue;
      }
      double worst = 0;
      for (int i = 0; i < n; ++i) {
        ++gChecks;
        const double ref = reference[k][i];
        const double tol = kRelTol * std::fabs(ref) + kAbsTol + (pixels[i] > 0 ? kPixelTol / pixels[i] : 0.0);
        const double d = std::fabs(stats[i] - ref);
        worst = std::max(worst, d / tol);
        if (!(d <= tol)) {
          ++gFailures;
          char name[64];
          fprintf(stderr, "FAIL [%s] %s: %.9g, reference %.9g (off by %.2e, allowed %.2e)\n", c.name,
                  statName(c, i, name, sizeof(name)), stats[i], ref, d, tol);
        }
      }
      if (getenv("OD_PINNED_VERBOSE")) fprintf(stderr, "%-28s worst deviation %.3f of the tolerance\n", c.name, worst);
    }
    if (print) printf("};\n");
    od_context_destroy(gCtx);
    if (print || (!dumpDir.empty() && scale != 1)) return gFailures ? 1 : 0;
    printf("test_pinned: %d checks, %d failures\n", gChecks, gFailures);
    return gFailures ? 1 : 0;
  }
}
