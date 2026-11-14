// SPDX-License-Identifier: MIT
// Dark monochrome footage: R = G = B everywhere and every value
// below 0.5 (low-light black and white, a dark clip under a saturation-0 grade upstream).
// Its working chroma rounds to exactly zero in every tile (the conversion's error is below
// half the smallest float16 step there), so until the zero-energy profile change the
// automatic profile found nothing and the frame passed through untouched. It must profile (Y
// only) and be denoised, and stay gray. Bounded synthetic buffers: five 640x384 frames, 32f
// and 16f hosts, codes of a 10-bit and of an 8-bit decode (not dyadic values).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "opendenoise/opendenoise.h"

static int fails = 0;
#define CHECK(x)                                           \
  do {                                                     \
    if (!(x)) {                                            \
      fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); \
      ++fails;                                             \
    }                                                      \
  } while (0)

int main() {
  @autoreleasepool {
    unsetenv("OD_COARSE_CHROMA");
    unsetenv("OD_SPATIAL_TUNE");
    constexpr int W = 640, H = 384, N = 5;
    constexpr float kSigma = 0.02f;
    auto dev = MTLCreateSystemDefaultDevice();
    char err[512] = {};
    ODContext* ctx = od_context_create((__bridge void*)dev, nullptr, err, sizeof(err));
    if (!ctx) {
      fprintf(stderr, "%s\n", err);
      return 1;
    }
    // Clean picture: a dark ramp with a few flat patches, max 0.42.
    std::vector<float> clean(W * H);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        float v = 0.04f + 0.30f * x / W + 0.04f * y / H;
        if ((x / 96 + y / 96) % 3 == 0) v = 0.08f + 0.05f * ((x / 96) % 4);
        clean[y * W + x] = v;
      }
    for (int levels : {1023, 255}) {
      // Noisy frames, quantised as a decode delivers them, gray.
      std::vector<std::vector<float>> noisy(N, std::vector<float>(W * H));
      float maxValue = 0;
      for (int t = 0; t < N; ++t) {
        std::mt19937 rng(9000 + 17 * t + levels);
        std::normal_distribution<float> normal(0, kSigma);
        for (int i = 0; i < W * H; ++i) {
          float v = std::min(0.49f, std::max(0.0f, clean[i] + normal(rng)));
          noisy[t][i] = std::round(v * levels) / levels;
          maxValue = std::max(maxValue, noisy[t][i]);
        }
      }
      for (ODPixelFormat format : {OD_PIXEL_BGRA32F, OD_PIXEL_BGRA16F}) {
        const bool half = format == OD_PIXEL_BGRA16F;
        const size_t px = half ? 8 : 16, bytes = (size_t)W * H * px;
        id<MTLBuffer> in[N], out = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        for (int t = 0; t < N; ++t) {
          in[t] = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
          for (int i = 0; i < W * H; ++i)
            for (int c = 0; c < 4; ++c) {
              const float v = c == 3 ? 1.0f : noisy[t][i];
              if (half)
                ((__fp16*)in[t].contents)[i * 4 + c] = (__fp16)v;
              else
                ((float*)in[t].contents)[i * 4 + c] = v;
            }
        }
        auto value = [&](id<MTLBuffer> b, int i, int c) {
          return half ? (float)((const __fp16*)b.contents)[i * 4 + c] : ((const float*)b.contents)[i * 4 + c];
        };
        ODSession* s = od_session_create(ctx);
        // The profile of one frame: Y only.
        ODFrame one = {(__bridge void*)in[2], W, H, (uint32_t)(W * px), format, 3};
        ODNoiseProfile prof = {};
        const int prc = od_profile_frame(s, &one, &prof);
        float ySigma = 0, cMax = 0;
        for (int l = 0; l < OD_LEVELS; ++l) {
          ySigma = std::max(ySigma, prof.sigmaFull[0][l]);
          cMax = std::max({cMax, prof.sigmaFull[1][l], prof.sigmaFull[2][l]});
        }
        // The middle frame with its +/-2 window and default parameters, as
        // the plug-in sends it (keys, per-frame check).
        od_session_reset(s);
        ODFrame fr[N];
        int offs[N];
        const int order[N] = {0, -1, 1, -2, 2};
        for (int k = 0; k < N; ++k) {
          fr[k] = {(__bridge void*)in[2 + order[k]], W, H, (uint32_t)(W * px), format, 1 + 2 + order[k]};
          offs[k] = order[k];
        }
        ODParams p = od_params_default();
        p.waitUntilCompleted = 1;
        p.verifyFrameCache = 1;
        ODFrame dst = {(__bridge void*)out, W, H, (uint32_t)(W * px), format, OD_NO_KEY};
        ODStats st = {};
        const int rc = od_render(s, fr, offs, N, &p, &dst, &st);
        double inErr = 0, outErr = 0, gray = 0;
        size_t changed = 0;
        for (int y = 8; y < H - 8; ++y)
          for (int x = 8; x < W - 8; ++x) {
            const int i = y * W + x;
            const float g = value(out, i, 1), src = value(in[2], i, 1);
            inErr += std::pow(src - clean[i], 2);
            outErr += std::pow(g - clean[i], 2);
            changed += g != src;
            gray = std::max({gray, (double)std::fabs(value(out, i, 0) - g), (double)std::fabs(value(out, i, 2) - g)});
          }
        const double n = (double)(W - 16) * (H - 16);
        const double inRms = std::sqrt(inErr / n), outRms = std::sqrt(outErr / n);
        printf("gray %d levels %s (max %.3f): profile rc %d valid %u Y %.5f chroma %.2g; render rc %d valid %u, "
               "rms error %.5f -> %.5f (%.1f dB gain), changed %.0f %%, max |R-G|,|B-G| %.2g\n",
               levels, half ? "16f" : "32f", maxValue, prc, prof.valid, ySigma, cMax, rc, st.profile.valid, inRms,
               outRms, 20 * std::log10(inRms / std::max(outRms, 1e-9)), 100.0 * changed / n, gray);
        CHECK(maxValue < 0.5f);
        CHECK(prc == 0 && prof.valid);
        CHECK(ySigma > 0.6f * kSigma && ySigma < 1.4f * kSigma);  // the noise of the frame
        CHECK(cMax < 1e-5f);                                       // and no chroma noise
        CHECK(rc == 0 && st.profile.valid);
        CHECK(changed > n / 2);           // not passed through
        CHECK(outRms < 0.6 * inRms);      // denoised
        CHECK(gray < (half ? 2e-3 : 1e-4));  // still gray (16f: about one float16 step at 0.5)
        od_session_destroy(s);
      }
    }
    od_context_destroy(ctx);
    printf("gray: %d failures\n", fails);
    return fails ? 1 : 0;
  }
}
