// SPDX-License-Identifier: MIT
// bands step: the editor's amount per frequency band (ODParams.bandAmount,
// API v6), applied once by the pipeline after the last denoising stage and
// before sharpen (stage.h section 1). Shader: core/shaders/bands_amount.metal.
//
// The removed signal (image - original) is split into the octaves the spatial
// stage filters, with the same tent pyramid: band 0 (about 1-2 px), 1 (2-4
// px), 2 (4-8 px) and everything coarser, which takes the band-3 amount (the
// profile has no band beyond 3; the spatial stage's levels 4 and 5 are
// extrapolated from it). Each is scaled by its amount, luma and chroma
// separately, and the result is added back to the original. Alpha is the
// image's own, so the 1/sqrt(count) the spatial stage leaves there for sharpen
// survives.
//
// Cost (3840x2160): one half-size pass reading the image and the original, two smaller down-
// sampling passes, two collapse passes at half size and below, one full-size pass reading
// both again and writing the result: 0.54 ms (1920x1080: 0.10 ms),, "Advanced controls and
// the bands step". Nothing of it runs while every amount is 100 % (the pipeline does not
// call encode) in legacy mode. V2 also uses this step to retain fine luma grain; neutral
// band amounts need just the half-size downsample and full-size reconstruction.
#include "stage.h"

#include <algorithm>
#include <cstring>

namespace od {
namespace {

constexpr int kLevels = 3;           // G_1..G_3; G_3 holds band 3 and everything coarser
constexpr uint32_t kMinTopSize = 8;  // no level smaller than this (as the spatial stage)

struct BandsUniforms {
  float w[4];          // xyz: weights of this level's octave (Y, Cb, Cr)
  float wTop[4];       // xyz: weights of the top level
  float grain[4];      // global luma amount, grain floor, reserved
  float sigma[OD_LEVELS];  // unadjusted full-band luma sigma
  uint32_t flags[4];   // see BandsU in bands_amount.metal
};
static_assert(sizeof(BandsUniforms) == 96, "BandsUniforms must match BandsU");

class Bands final : public BandsStage {
 public:
  id<MTLComputePipelineState> downFirst_, down_, collapse_, final_;

  id<MTLTexture> encode(RenderEnv& env, id<MTLTexture> image) override {
    const ODParams& p = env.params;
    bool copyY = true, copyC = true;
    for (int b = 0; b < OD_BANDS; ++b) {
      copyY = copyY && p.bandAmount[0][b] == 1.0f;
      copyC = copyC && p.bandAmount[1][b] == 1.0f;
    }
    const bool fineGrain = env.grainProfileV2 && p.lumaStrength > 0.0f && env.grainFloor() > 0.0f;
    const bool amountsNeutral = copyY && copyC;
    if (fineGrain) copyY = false;
    if (copyY && copyC) return image;  // every amount 100 %: nothing to do
    auto weights = [&](int b, float* w) {
      w[0] = p.bandAmount[0][b];
      w[1] = w[2] = p.bandAmount[1][b];
      w[3] = 0.0f;
    };

    uint32_t lw[kLevels + 1], lh[kLevels + 1];
    lw[0] = env.width;
    lh[0] = env.height;
    int K = 0;  // top level; 0 = the frame is too small for a pyramid
    while (K < (amountsNeutral ? 1 : kLevels)) {
      const uint32_t nw = (lw[K] + 1) / 2, nh = (lh[K] + 1) / 2;
      if (std::min(nw, nh) < kMinTopSize) break;
      ++K;
      lw[K] = nw;
      lh[K] = nh;
    }

    BandsUniforms u;
    memset(&u, 0, sizeof(u));
    weights(OD_BANDS - 1, u.wTop);
    u.grain[0] = p.lumaStrength;
    u.grain[1] = fineGrain ? env.grainFloor() : 0.0f;
    for (int l = 0; l < OD_LEVELS; ++l) u.sigma[l] = lumaSigmaFull(env.profile, l);
    u.flags[1] = copyY ? 1u : 0u;
    u.flags[2] = copyC ? 1u : 0u;
    id<MTLTexture> orig = env.frames[0].tex;

    // ---- pyramid of the removed signal ----
    id<MTLTexture> G[kLevels + 1] = {};
    for (int k = 1; k <= K; ++k) {
      G[k] = env.temp(kWorkingFormat, lw[k], lh[k]);
      if (!G[k]) return nil;
      id<MTLComputeCommandEncoder> enc = env.compute();
      if (k == 1) {
        [enc setTexture:image atIndex:0];
        [enc setTexture:orig atIndex:1];
        [enc setTexture:G[1] atIndex:2];
        env.dispatch(downFirst_, lw[1], lh[1]);
      } else {
        [enc setTexture:G[k - 1] atIndex:0];
        [enc setTexture:G[k] atIndex:1];
        env.dispatch(down_, lw[k], lh[k]);
      }
    }

    // ---- collapse, coarse to fine: R_k for k = K-1 .. 1 ----
    id<MTLTexture> R = nil;  // R_(k+1); nil while that is the top (R_top = wTop G_top)
    for (int k = K - 1; k >= 1; --k) {
      id<MTLTexture> r = env.temp(kWorkingFormat, lw[k], lh[k]);
      if (!r) return nil;
      weights(k, u.w);
      u.flags[0] = R ? 0u : 1u;
      id<MTLComputeCommandEncoder> enc = env.compute();
      [enc setTexture:G[k] atIndex:0];
      [enc setTexture:G[k + 1] atIndex:1];
      [enc setTexture:(R ? R : G[k + 1]) atIndex:2];
      [enc setTexture:r atIndex:3];
      [enc setBytes:&u length:sizeof(u) atIndex:0];
      env.dispatch(collapse_, lw[k], lh[k]);
      if (R) env.recycle(R);
      env.recycle(G[k + 1]);
      G[k + 1] = nil;
      R = r;
    }

    // ---- full size ----
    id<MTLTexture> out = env.tempWorking();
    if (!out) return nil;
    weights(0, u.w);
    u.flags[0] = (K >= 1 && !R) ? 1u : 0u;
    u.flags[3] = K == 0 ? 1u : 0u;
    id<MTLComputeCommandEncoder> enc = env.compute();
    [enc setTexture:image atIndex:0];
    [enc setTexture:orig atIndex:1];
    [enc setTexture:(G[1] ? G[1] : image) atIndex:2];
    [enc setTexture:(R ? R : (G[1] ? G[1] : image)) atIndex:3];
    [enc setTexture:out atIndex:4];
    [enc setBytes:&u length:sizeof(u) atIndex:0];
    env.dispatch(final_, env.width, env.height);
    if (R) env.recycle(R);
    if (G[1]) env.recycle(G[1]);
    return out;
  }
};

}  // namespace

std::unique_ptr<BandsStage> makeBandsStage(Context& ctx, std::string* err) {
  auto s = std::make_unique<Bands>();
  s->downFirst_ = ctx.pipeline("bands_down_first", err);
  s->down_ = ctx.pipeline("bands_down", err);
  s->collapse_ = ctx.pipeline("bands_collapse", err);
  s->final_ = ctx.pipeline("bands_final", err);
  if (!s->downFirst_ || !s->down_ || !s->collapse_ || !s->final_) return nullptr;
  return s;
}

}  // namespace od
