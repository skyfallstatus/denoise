// SPDX-License-Identifier: MIT
// End-to-end rejection/qualification and manual-profile
// fallback. Bounded synthetic buffers only; three supplied frames, no application or media.
// Then the API v9 switch: ODParams.coarseChroma renders exactly what the diagnostic switch
// OD_COARSE_CHROMA=1 renders, and the environment overrides it either way.
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
    setenv("OD_COARSE_CHROMA", "1", 1);
    constexpr int W = 768, H = 768, N = 3, M = 48;
    auto dev = MTLCreateSystemDefaultDevice();
    char err[512] = {};
    ODContext* ctx = od_context_create((__bridge void*)dev, nullptr, err, sizeof(err));
    if (!ctx) {
      fprintf(stderr, "%s\n", err);
      return 1;
    }
    ODSession* session = od_session_create(ctx);
    id<MTLBuffer> in[N], out = [dev newBufferWithLength:W * H * 16 options:MTLResourceStorageModeShared];
    for (int t = 0; t < N; ++t)
      in[t] = [dev newBufferWithLength:W * H * 16 options:MTLResourceStorageModeShared];
    std::mt19937 rng(7321);
    std::normal_distribution<float> normal(0, 1);
    float coef[N][2][M], kx[M], ky[M];
    for (int j = 0; j < M; ++j) {
      float a = j * 2.399963f, f = 1.0f / (32 + j % 16 * 6);
      kx[j] = 6.283185f * f * cos(a);
      ky[j] = 6.283185f * f * sin(a);
    }
    for (int c = 0; c < 2; ++c)
      for (int j = 0; j < M; ++j) {
        coef[0][c][j] = normal(rng);
        for (int t = 1; t < N; ++t) coef[t][c][j] = .8f * coef[t - 1][c][j] + .6f * normal(rng);
      }
    ODFrame frames[N];
    int offs[N] = {0, -2, 2};
    ODFrame output = {(__bridge void*)out, W, H, W * 16, OD_PIXEL_BGRA32F, OD_NO_KEY};
    ODParams p = od_params_default();
    // The grain mode `advanced` 0 gives (API v10: every v6, v9 and v10 field
    // neutral), so that renders with advanced 0 and 1 compare.
    p.grainMode = OD_GRAIN_WHOLE;
    p.waitUntilCompleted = 1;
    p.noFrameCache = 1;
    auto fill = [&](int mode) {
      for (int t = 0; t < N; ++t) {
        float* buf = (float*)in[t].contents;
        for (int y = 0; y < H; ++y)
          for (int x = 0; x < W; ++x) {
            float yy = .3f + .12f * x / W + .03f * y / H, cb = .02f + .03f * x / W, cr = .04f - .02f * y / H;
            if (mode != 0) {
              float noise[2] = {};
              for (int c = 0; c < 2; ++c)
                for (int j = 0; j < M; ++j)
                  noise[c] += coef[mode == 2 ? 0 : t][c][j] * sin(kx[j] * x + ky[j] * y + c * 2.1f + j);
              cb += .004f * noise[0] / sqrt(M * .5f);
              cr += .004f * noise[1] / sqrt(M * .5f);
            }
            yy += .002f * normal(rng);
            cb += .001f * normal(rng);
            cr += .001f * normal(rng);
            size_t i = ((size_t)y * W + x) * 4;
            buf[i] = yy + 1.8556f * cb;
            buf[i + 1] = yy - .187324f * cb - .468124f * cr;
            buf[i + 2] = yy + 1.5748f * cr;
            buf[i + 3] = 1;
          }
        frames[t] = {(__bridge void*)in[t], W, H, W * 16, OD_PIXEL_BGRA32F, t + 1};
      }
    };
    for (int mode = 0; mode < 3; ++mode) {
      fill(mode);
      od_session_reset(session);
      ODStats st = {};
      CHECK(od_render(session, frames, offs, N, &p, &output, &st) == 0);
      int qualified = 0;
      for (int b = 0; b < OD_COARSE_BANDS; ++b)
        for (int l = 0; l < OD_LEVELS; ++l) qualified += st.profile.coarseConfidence[b][l] >= .5f;
      printf("coarse mode %d: qualified %d\n", mode, qualified);
      if (mode == 1)
        CHECK(qualified > 0);
      else
        CHECK(qualified == 0);
      // Exact zero-confidence fallback, including a supplied nonzero sigma.
      ODNoiseProfile manual = st.profile;
      memset(manual.coarseConfidence, 0, sizeof(manual.coarseConfidence));
      p.manualProfile = &manual;
      CHECK(od_render(session, frames, offs, N, &p, &output, nullptr) == 0);
      std::vector<float> a(W * H * 4);
      memcpy(a.data(), out.contents, a.size() * 4);
      for (auto& c : manual.coarseSigma)
        for (auto& b : c)
          for (float& v : b) v = .1f;
      CHECK(od_render(session, frames, offs, N, &p, &output, nullptr) == 0);
      CHECK(memcmp(a.data(), out.contents, a.size() * 4) == 0);
      if (mode == 1) {
        p.manualProfile = &st.profile;
        CHECK(od_render(session, frames, offs, N, &p, &output, nullptr) == 0);
        const float* after = (const float*)out.contents;
        double beforeMSE = 0, afterMSE = 0, maxY = 0;
        for (int y = 32; y < H - 32; ++y)
          for (int x = 32; x < W - 32; ++x) {
            size_t i = ((size_t)y * W + x) * 4;
            auto yc = [](const float* v, float* c) {
              c[0] = .0722f * v[0] + .7152f * v[1] + .2126f * v[2];
              c[1] = (v[0] - c[0]) / 1.8556f;
              c[2] = (v[2] - c[0]) / 1.5748f;
            };
            float ca[3], cb[3];
            yc(a.data() + i, ca);
            yc(after + i, cb);
            maxY = std::max(maxY, double(std::fabs(ca[0] - cb[0])));
            float truth[2] = {.02f + .03f * x / W, .04f - .02f * y / H};
            for (int c = 1; c < 3; ++c) {
              beforeMSE += std::pow(ca[c] - truth[c - 1], 2);
              afterMSE += std::pow(cb[c] - truth[c - 1], 2);
            }
          }
        printf("coarse residual MSE ratio %.4f, luma max delta %.9f\n", afterMSE / beforeMSE, maxY);
        CHECK(afterMSE < beforeMSE * .98);
        CHECK(maxY < 2e-7);
        std::vector<float> enabled(W * H * 4);
        memcpy(enabled.data(), out.contents, enabled.size() * 4);
        p.advanced = 0;
        CHECK(od_render(session, frames, offs, N, &p, &output, nullptr) == 0);
        CHECK(memcmp(enabled.data(), out.contents, enabled.size() * 4) == 0);
        p.advanced = 1;
        p.manualProfile = &manual;
      }
      // Manual garbage is never admitted to the optional spatial path.
      manual.coarseConfidence[0][2] = NAN;
      CHECK(od_render(session, frames, offs, N, &p, &output, nullptr) == 0);
      CHECK(memcmp(a.data(), out.contents, a.size() * 4) == 0);
      p.manualProfile = nullptr;
    }
    // API v9: the switch as a parameter. Frames whose coarse chroma noise
    // qualifies (mode 1); every render from a reset session, blocking.
    {
      fill(1);
      auto render = [&](const char* env, uint32_t on, uint32_t advanced, int* qualified) {
        if (env) setenv("OD_COARSE_CHROMA", env, 1);
        else unsetenv("OD_COARSE_CHROMA");
        ODParams q = od_params_default();
        q.grainMode = OD_GRAIN_WHOLE;  // neutral, as `advanced` 0 gives it
        q.waitUntilCompleted = 1;
        q.noFrameCache = 1;
        q.coarseChroma = on;
        q.advanced = advanced;
        od_session_reset(session);
        ODStats st = {};
        CHECK(od_render(session, frames, offs, N, &q, &output, &st) == 0);
        *qualified = 0;
        for (int b = 0; b < OD_COARSE_BANDS; ++b)
          for (int l = 0; l < OD_LEVELS; ++l) *qualified += st.profile.coarseConfidence[b][l] > 0;
        std::vector<float> v(W * H * 4);
        memcpy(v.data(), out.contents, v.size() * 4);
        return v;
      };
      int qEnv = 0, qParam = 0, qOff = 0, qEnvOff = 0, qAdv0 = 0;
      const std::vector<float> byEnv = render("1", 0, 1, &qEnv);
      const std::vector<float> byParam = render(nullptr, 1, 1, &qParam);
      const std::vector<float> off = render(nullptr, 0, 1, &qOff);
      const std::vector<float> envOff = render("0", 1, 1, &qEnvOff);
      const std::vector<float> adv0 = render(nullptr, 1, 0, &qAdv0);
      printf("coarse param: estimate knots env %d param %d off %d env0 %d advanced0 %d\n", qEnv, qParam, qOff, qEnvOff,
             qAdv0);
      CHECK(qEnv > 0 && qParam == qEnv);
      CHECK(byParam == byEnv);        // the parameter is the research switch, bit for bit
      CHECK(byParam != off);          // and it acts on these frames
      CHECK(qOff == 0 && qEnvOff == 0 && qAdv0 == 0);  // off: no estimate in the profile
      CHECK(envOff == off);           // OD_COARSE_CHROMA=0 overrides the parameter
      CHECK(adv0 == off);             // advanced = 0: every v6/v9/v10 field neutral
      unsetenv("OD_COARSE_CHROMA");

      // The switch changed on a session that already measured these keys
      // with it off: a parked frame, or a replayed range, when the editor
      // ticks the box (Premiere's new instance shares the session). The
      // estimate is made only inside a measurement and a key is measured
      // once, so the change must measure again; a render that may wait does,
      // and returns the corrected frame. Keys and the per-frame check as the
      // plug-in sends them.
      auto keyed = [&](uint32_t on, ODStats* st) {
        ODParams q = od_params_default();
        q.waitUntilCompleted = 1;
        q.verifyFrameCache = 1;
        q.coarseChroma = on;
        CHECK(od_render(session, frames, offs, N, &q, &output, st) == 0);
        std::vector<float> v(W * H * 4);
        memcpy(v.data(), out.contents, v.size() * 4);
        return v;
      };
      auto knots = [](const ODStats& st) {
        int n = 0;
        for (int b = 0; b < OD_COARSE_BANDS; ++b)
          for (int l = 0; l < OD_LEVELS; ++l) n += st.profile.coarseConfidence[b][l] > 0;
        return n;
      };
      auto sameCoarse = [](const ODNoiseProfile& a, const ODNoiseProfile& b) {
        return !memcmp(a.coarseSigma, b.coarseSigma, sizeof(a.coarseSigma)) &&
               !memcmp(a.coarseConfidence, b.coarseConfidence, sizeof(a.coarseConfidence)) &&
               !memcmp(a.coarseCoverage, b.coarseCoverage, sizeof(a.coarseCoverage)) &&
               !memcmp(a.coarseCorrelation, b.coarseCorrelation, sizeof(a.coarseCorrelation));
      };
      od_session_reset(session);
      ODStats freshOn = {};
      const std::vector<float> freshPicture = keyed(1, &freshOn);
      od_session_reset(session);
      ODStats st = {};
      const std::vector<float> parkedOff = keyed(0, &st);
      keyed(0, &st);
      const int offKnots = knots(st);
      const std::vector<float> switchedOn = keyed(1, &st);
      const int onKnots = knots(st);
      const bool coarseMatches = sameCoarse(st.profile, freshOn.profile);
      keyed(1, &st);
      const int againKnots = knots(st);
      const std::vector<float> switchedOff = keyed(0, &st);
      printf("coarse toggle: knots off %d, switched on %d (fresh %d), again %d; estimate as fresh %d, "
             "picture as fresh %d, back off as off %d\n",
             offKnots, onKnots, knots(freshOn), againKnots, coarseMatches, switchedOn == freshPicture,
             switchedOff == parkedOff);
      CHECK(offKnots == 0);
      CHECK(onKnots > 0 && onKnots == knots(freshOn));
      CHECK(coarseMatches);             // the estimate of this window, as a fresh session makes it
      CHECK(switchedOn != parkedOff);   // the first render after the change is corrected,
      CHECK(switchedOn == freshPicture);  // as a session that had it on from the start
      CHECK(againKnots == onKnots);
      CHECK(switchedOff == parkedOff);  // and switching back off restores the picture
    }
    od_session_destroy(session);
    od_context_destroy(ctx);
    printf("coarse: %d failures\n", fails);
    return fails ? 1 : 0;
  }
}
