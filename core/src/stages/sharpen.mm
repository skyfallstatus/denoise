// SPDX-License-Identifier: MIT
// sharpen stage: cored unsharp masking on a-trous bands 0 and 1
// of the luma, above the remaining noise floor only, with an overshoot limit. Contract:
// stage.h section 3. Shader: core/shaders/sharpen_usm.metal. Tests: tests/spatial/.
#include "stage.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace od {
namespace {

constexpr float kSigmaQ = 2e-4f;      // threshold floor: half-float quantisation
constexpr float kCore = 2.5f;         // coring starts at kCore x the remaining noise, full gain at 3x that.
                                      // (notes: 1.5; measured on the noisy test clips, sharpen = 1, no
                                      // denoising: flat-area error +6 % at 1.5, under +2 % at 2.5)
constexpr float kGain0 = 1.5f;        // band 0 gain at sharpen = 1
constexpr float kGain1 = 0.75f;       // band 1 gain at sharpen = 1
constexpr float kOvershoot = 0.10f;   // allowed overshoot, share of the local 3x3 range (bounded;
                                      // lowered: 0.25 is a visible halo on hard edges at sharpen = 1),
                                      // of a range of at most 1: so never more than 0.1 in absolute terms
                                      // (sharpen_usm.metal, sharpen_limit)
constexpr float kSpatialLeft = 0.5f;  // share of the noise the spatial stage leaves (cautious:
                                      // measured 0.25 to 0.4 on levels 0 and 1,)

struct SharpenUniforms {
  float thr[OD_LEVELS][4];  // x, y: kCore x the profiled sigma of bands 0, 1 (before any denoising),
                              // z: full-band luma sigma, unscaled (the grain floor's lookup)
  float gain[4];              // x, y: band gains, z: overshoot allowance, w: threshold floor
  float left[4];              // x: luma amount applied by the denoising stages (0 = none ran),
                              // y: share of the noise the spatial stage leaves (1 = it did not run),
                              // z: 1 = the image's alpha holds 1/sqrt(count) (the spatial stage ran),
                              // w: the grain floor the denoising stages capped x with (stage.h)
  uint32_t flags[4];           // x: fine-only grain retention (V2)
};

class Sharpen final : public SharpenStage {
 public:
  id<MTLComputePipelineState> b3_, b3d_, apply_, fast_;
  id<MTLTexture> one_;  // 1x1 r16f holding 1.0: "no count map"

  id<MTLTexture> encode(RenderEnv& env, id<MTLTexture> image, id<MTLTexture> count) override {
    const ODParams& p = env.params;
    // Without a profile the noise floor is unknown and sharpening would
    // amplify the noise: leave the picture alone.
    if (!env.profile.valid || p.sharpen <= 0) return image;
    const uint32_t W = env.width, H = env.height;

    // What is left of the luma noise after the stages that ran, per pixel in
    // the shader:  left = (1 - a) + a * rho * s,  a = luma amount,
    // rho = 1/sqrt(count) from the temporal stage's map (1 where nothing was
    // merged, or when it did not run), s = kSpatialLeft if spatial ran.
    const bool denoised = (env.temporalRuns || env.spatialRuns) && p.lumaStrength > 0;
    SharpenUniforms u;
    memset(&u, 0, sizeof(u));
    u.flags[0] = env.grainProfileV2 ? 1u : 0u;
    // Editor's corrections (API v6), each applied only away from its neutral
    // value: the luma noise of bands 0 and 1 (Luma detail and Noise profile
    // levels) and Sharpen noise floor on the coring threshold, Sharpen fine
    // and medium on the gains, Sharpen halo limit on the overshoot.
    const float adj[2] = {env.noiseAdjust(0, 0), env.noiseAdjust(0, 1)};
    for (int l = 0; l < OD_LEVELS; ++l) {
      for (int b = 0; b < 2; ++b) {
        u.thr[l][b] = kCore * env.profile.sigma[0][b][l] * env.sigmaScale();
        if (adj[b] != 1.0f) u.thr[l][b] *= adj[b];
        if (p.sharpenCoring != 1.0f) u.thr[l][b] *= p.sharpenCoring;
      }
      u.thr[l][2] = lumaSigmaFull(env.profile, l);
    }
    u.gain[0] = kGain0 * p.sharpen;
    u.gain[1] = kGain1 * p.sharpen;
    if (p.sharpenBand[0] != 1.0f) u.gain[0] *= p.sharpenBand[0];
    if (p.sharpenBand[1] != 1.0f) u.gain[1] *= p.sharpenBand[1];
    u.gain[2] = kOvershoot;
    if (p.sharpenHalo != 1.0f) u.gain[2] = kOvershoot * p.sharpenHalo;
    u.gain[3] = kSigmaQ;
    u.left[0] = denoised ? p.lumaStrength : 0.0f;
    // The bands step left part of the removed noise in place: assume the
    // smaller of the fine and medium luma amounts everywhere (more noise left,
    // so less of it is sharpened). Playback quality sharpens band 0 only.
    if (denoised && env.bandsRun) {
      const float band = p.quality == 0 ? p.bandAmount[0][0] : std::min(p.bandAmount[0][0], p.bandAmount[0][1]);
      u.left[0] = p.lumaStrength * band;
    }
    u.left[3] = denoised ? env.grainFloor() : 0.0f;  // the amount was capped by the floor (stage.h)
    u.left[1] = env.spatialRuns ? kSpatialLeft : 1.0f;
    // The spatial stage leaves 1/sqrt(count) in the alpha of its result (1
    // when the temporal stage gave no map): one fetch and one full-size
    // texture less than sampling the map here. `image` is that result
    // whenever the spatial stage ran: with a valid profile, which is checked
    // above, it never passes its input through.
    u.left[2] = env.spatialRuns ? 1.0f : 0.0f;
    id<MTLTexture> cnt = (env.temporalRuns && count) ? count : one_;

    id<MTLTexture> out = env.tempWorking();
    if (!out) return nil;
    if (p.quality == 0) {
      id<MTLComputeCommandEncoder> enc = env.compute("sharpen_apply_fast");
      [enc setTexture:image atIndex:0];
      [enc setTexture:out atIndex:1];
      [enc setTexture:cnt atIndex:2];
      [enc setBytes:&u length:sizeof(u) atIndex:0];
      env.dispatch(fast_, W, H);
      return out;
    }
    id<MTLTexture> c1 = env.temp(MTLPixelFormatR16Float, W, H);
    id<MTLTexture> c2 = env.temp(MTLPixelFormatR16Float, W, H);
    if (!c1 || !c2) return nil;
    id<MTLComputeCommandEncoder> enc = env.compute("sharpen_b3");
    [enc setTexture:image atIndex:0];
    [enc setTexture:c1 atIndex:1];
    env.dispatch(b3_, W, H);
    enc = env.compute("sharpen_b3_dilated");
    [enc setTexture:c1 atIndex:0];
    [enc setTexture:c2 atIndex:1];
    env.dispatch(b3d_, W, H);
    enc = env.compute("sharpen_apply");
    [enc setTexture:image atIndex:0];
    [enc setTexture:c1 atIndex:1];
    [enc setTexture:c2 atIndex:2];
    [enc setTexture:out atIndex:3];
    [enc setTexture:cnt atIndex:4];
    [enc setBytes:&u length:sizeof(u) atIndex:0];
    env.dispatch(apply_, W, H);
    env.recycle(c1);
    env.recycle(c2);
    return out;
  }
};

}  // namespace

std::unique_ptr<SharpenStage> makeSharpenStage(Context& ctx, std::string* err) {
  auto s = std::make_unique<Sharpen>();
  s->b3_ = ctx.pipeline("sharpen_b3", err);
  s->b3d_ = ctx.pipeline("sharpen_b3_dilated", err);
  s->apply_ = ctx.pipeline("sharpen_apply", err);
  s->fast_ = ctx.pipeline("sharpen_apply_fast", err);
  if (!s->b3_ || !s->b3d_ || !s->apply_ || !s->fast_) return nullptr;
  MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                                               width:1
                                                                              height:1
                                                                           mipmapped:NO];
  d.usage = MTLTextureUsageShaderRead;
  d.storageMode = MTLStorageModeShared;
  s->one_ = [ctx.device() newTextureWithDescriptor:d];
  if (!s->one_) {
    if (err) *err = "sharpen: could not create the 1x1 count texture";
    return nullptr;
  }
  const uint16_t oneHalf = 0x3C00;
  [s->one_ replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&oneHalf bytesPerRow:2];
  return s;
}

}  // namespace od
