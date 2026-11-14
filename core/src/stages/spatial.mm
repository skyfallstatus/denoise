// SPDX-License-Identifier: MIT
// Spatial stage: coarse-to-fine pyramid non-local means, luma-guided chroma.
// Contract: stage.h; shaders: spatial_nlm.metal and spatial_residual.metal.
// Build a tent pyramid G_0..G_K (at most five reductions), then reconstruct:
// X_k = G_k + U(Dn_(k+1) - G_(k+1)); Dn_k = NLM(X_k).
// Luma and chroma run at every level. Level 0 applies the channel amounts
// against the original frame and passes the residual-noise factor to sharpening.
// NLM uses noise-normalized 3x3 patches, unit weight up to q=2, then an
// exponential falloff. Chroma thresholds are scaled by 1.5 and use a quarter
// of the luma distance as their guide. Speck removal treats isolated dark and
// bright excursions with different upper bounds, preserving very bright points.
// Version 1 uses effective frame count plus a coarse-chroma persistence map.
// Version 2 calibrates remaining noise separately per channel and scale from
// raw and merged energies. Grain retention is applied according to grainMode.
// Arithmetic is mixed float/half; normalized patch terms use half precision.
#include "stage.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace od {
namespace {

// ---- tuning -------------
constexpr int kMaxCoarse = 5;   // coarse levels below the frame: G_1..G_5
constexpr int kMinTopSize = 8;  // do not build a level smaller than this

// Start values; OD_SPATIAL_TUNE="name=value,..." overrides them for tuning
// runs (read once per session, never set in production).
struct Tune {
  float etaY = 0.3f;       // weight fall-off beyond the knee, luma
  float etaC = 0.5f;       // ... chroma
  float knee = 2.0f;       // q up to which a candidate gets full weight (q is 1 +- 0.47 for noise alone)
  float guide = 0.25f;     // weight of the luma distance in the chroma weight
  float tcap = 2.0f;       // cap of one pixel's term in the luma q (2 = a pair 6 sigma apart).
                           // Luma only: capping chroma cost chroma PSNR at low noise.
  float outlierLo = 4.0f;  // a pixel between these many sigma off the mean of its 8 neighbours
  float outlier = 12.0f;   // is treated as an outlier of the noise (speck); brighter than its
                           // neighbours by more is kept (a star, a specular, a distant light)
  float outlierDark = 24.0f;  // the same limit for a pixel DARKER than its neighbours. Shot noise in log
                             // footage has a long dark tail (-12 to -18 sigma on the heavy test clips),
                             // kept as black dots at luma 100 % with a 12-sigma limit; 24 removes them all
                             // there, as does no limit at all. A finite limit keeps the stage continuous
                             // as the strength goes to 0 (every pixel then lies beyond it).
  float keep = kGrainFloor;  // luma grain sigma never removed (stage.h; the override is for tuning runs only:
                             // the temporal and sharpen stages keep using kGrainFloor). ODParams.grainFloor
                             // multiplies it (env.grainFloor() elsewhere)
  float lumaLevels = 6;    // luma is filtered on levels 0..lumaLevels-1
  float q0search = 1;      // search radius on level 0 at quality 0
  float ty[kMaxCoarse + 1] = {1, 1, 1, 1, 1, 1};  // luma threshold scale per level
  float tc[kMaxCoarse + 1] = {1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f};  // chroma threshold scale per level
  float fy[kMaxCoarse + 1] = {1, 1, 1, 1, 0, 0};  // share of rho the luma threshold follows per level
  // Chroma noise that persists between frames (spatial_chroma_persist): the level it is
  // measured on (0 = never), the finest level whose chroma threshold uses it, and the chroma
  // structure, in noise variances, over which it fades out (from cgate0 to cgate1).
  // diagnostic-only, opt-in. Never changes the public API or defaults.
  float stable = 0, fine = 1, softspeck = 0, diagnostic = 0;
  bool stableGiven = false, fineGiven = false;  // the tune named `stable` / `fine`
  float cpersist = 3;
  float cfrom = 2;
  float cgate0 = 3, cgate1 = 6;

  void parse(const char* e) {
    struct {
      const char* name;
      float* v;
    } tab[] = {{"etaY", &etaY}, {"etaC", &etaC}, {"guide", &guide}, {"keep", &keep}, {"knee", &knee}, {"cap", &tcap}, {"outlier", &outlier}, {"outlierLo", &outlierLo}, {"outlierDark", &outlierDark},
               {"lumaLevels", &lumaLevels}, {"q0search", &q0search},
               {"y0", &ty[0]}, {"y1", &ty[1]}, {"y2", &ty[2]}, {"y3", &ty[3]}, {"y4", &ty[4]}, {"y5", &ty[5]},
               {"c0", &tc[0]}, {"c1", &tc[1]}, {"c2", &tc[2]}, {"c3", &tc[3]}, {"c4", &tc[4]}, {"c5", &tc[5]},
               {"f0", &fy[0]}, {"f1", &fy[1]}, {"f2", &fy[2]}, {"f3", &fy[3]}, {"f4", &fy[4]}, {"f5", &fy[5]},
               {"stable", &stable}, {"fine", &fine}, {"softspeck", &softspeck}, {"diagnostic", &diagnostic}, {"cpersist", &cpersist}, {"cfrom", &cfrom}, {"cgate0", &cgate0}, {"cgate1", &cgate1}};
    while (e && *e) {
      const char* eq = strchr(e, '=');
      if (!eq) break;
      for (auto& t : tab) {
        if (strlen(t.name) == (size_t)(eq - e) && strncmp(t.name, e, (size_t)(eq - e)) == 0) {
          *t.v = (float)atof(eq + 1);
          stableGiven = stableGiven || t.v == &stable;
          fineGiven = fineGiven || t.v == &fine;
        }
      }
      e = strchr(eq, ',');
      if (e) ++e;
    }
  }
};

// Tent pyramid noise relative to the profile's a-trous bands, white noise. L_k for k = 0..3;
// levels 4 and 5 are relative to band 3.
constexpr float kL[4] = {1.04f, 1.31f, 1.35f, 1.36f};
constexpr float kG[4] = {1.12f, 1.56f, 1.66f, 1.68f};  // G_k as the TOP level, k = 0..3
constexpr float kL4 = 0.65f, kG4 = 0.83f, kG5 = 0.41f;
constexpr float kWhiteRatio32 = 0.482f;  // band3 / band2 for white noise

// Grain stability (ODParams.grainStability, API v9): the slider is the strength s of the
// band-selective reference, with the fine share f fixed at the diagnostic setting "band 75".
// The reference is linear in s, so the slider cross-fades the luma grain put back by (1 -
// amount) from the input's (0) to band 75's (1). Measured monotone in flicker and detail on
// pan_1080p_heavy and face_1080p_heavy.
constexpr float kGrainStabilityFine = 0.75f;

struct SpatialUniforms {
  float sig[OD_LEVELS][4];
  float amount[4];
  float shape[4];
  float robust[4];
  uint32_t flags[4];
  float research[4];
};

struct PersistUniforms {
  float sig[OD_LEVELS][4];  // yz = chroma noise sigma of the measured level per knot
  float gate[4];
  uint32_t flags[4];
};

class Spatial final : public SpatialStage {
 public:
  id<MTLComputePipelineState> downFirst_, down_, prepare_, nlm5_, nlm3_, testCount_, downFirstT_, chromaPersist_, nlm5p_;
  // Grain stability's reference (made with the stage: a user control).
  id<MTLComputePipelineState> stableRef_;
  id<MTLComputePipelineState> residualRatio_ = nil, residual5_ = nil, residual3_ = nil;
  id<MTLComputePipelineState> temporalBlend_ = nil;
  // diagnostic-only (OD_SPATIAL_TUNE softspeck, diagnostic): made only for sessions that
  // opted in.
  id<MTLComputePipelineState> countView_, nlm5r_, nlm3r_, nlm5pr_;
  // OD_SPATIAL_FUSED=1 (opt-in): the finest level's prepare and NL-means as
  // one kernel staging a 32x8 tile in threadgroup memory (spatial_fused3/5),
  // bit for bit the picture of the two passes. Off by default.
  id<MTLComputePipelineState> fused3_ = nil, fused5_ = nil;
  bool fused_ = [] { const char* p = getenv("OD_SPATIAL_FUSED"); return p && atoi(p) != 0; }();
  // Coarse chroma cleanup (ODParams.coarseChroma, API v9; OD_COARSE_CHROMA
  // overrides it, read per render): the qualified coarse chroma correction
  // after the ordinary result (coarseCorrection).
  id<MTLComputePipelineState> coarseFilter_, coarseApply_;
  id<MTLTexture> one_;  // 1x1 r16f holding 1.0: "no count map"
  float testCount_n_ = 0;
  bool testCountSplit_ = false;
  Tune t_;

  // Noise sigma of pyramid level k (top = this level is the Gaussian top) for
  // channel c at knot l, before strength scaling.
  static float levelSigma(const ODNoiseProfile& p, int c, int k, bool top, int l) {
    if (k <= 3) {
      if (top) {
        if (k == 0) return p.sigmaFull[c][l] > 0 ? p.sigmaFull[c][l] : kG[0] * p.sigma[c][0][l];
        return kG[k] * p.sigma[c][k][l];
      }
      return kL[k] * p.sigma[c][k][l];
    }
    // Beyond the profile: extrapolate from band 3, steeper when the measured
    // spectrum is redder than white (low-frequency chroma blotches). Checked
    // against the measured level noise of the test clips (spatialq
    // --print-profile): within 20 % for L_4 and G_5.
    const float b3 = p.sigma[c][3][l], b2 = p.sigma[c][2][l];
    float r = b2 > 1e-9f ? (b3 / b2) / kWhiteRatio32 : 1.0f;
    r = std::min(std::max(r, 1.0f), 2.0f);
    if (k == 4) return (top ? kG4 : kL4) * b3 * r;
    return kG5 * b3 * r;
  }

  id<MTLTexture> encode(RenderEnv& env, id<MTLTexture> image, id<MTLTexture> count) override {
    return encodeWithPyramid(env, image, count, nullptr);
  }

  // A partial-strength render can reuse the calibration's raw pyramid for
  // its spatial-only endpoint. It is local to this render, never history.
  id<MTLTexture> encodeWithPyramid(RenderEnv& env, id<MTLTexture> image, id<MTLTexture> count,
                                   id<MTLTexture> const* basePyramid) {
    const ODParams& p = env.params;
    if (!env.profile.valid) return image;  // nothing to go on: pass through
    const uint32_t W = env.width, H = env.height;
    id<MTLTexture> orig = env.frames[0].tex;

    if (env.grainProfileV2) {
      // Keep the private spatial tuning override meaningful after moving
      // retention into the bands step. The regular parameter is unchanged.
      if (t_.keep != kGrainFloor) env.fineGrainOverride = std::max(0.0f, t_.keep * p.grainFloor);
      // A diagnostic map is not a denoised picture to decompose.
      if (t_.diagnostic > 0) env.bandsRun = false;
    }

    if (t_.diagnostic == 1) return image;  // full temporal result, before amounts
    if (t_.diagnostic == 3 && count) {
      id<MTLTexture> v = env.temp(kWorkingFormat, W, H);
      if (!v) return nil;
      id<MTLComputeCommandEncoder> enc = env.compute();
      [enc setTexture:count atIndex:0]; [enc setTexture:v atIndex:1];
      env.dispatch(countView_, W, H);
      return v;  // count / OD_MAX_FRAMES as neutral luma (raw output recommended)
    }
    // The reference is only substituted at the FINAL amount blend; the chroma-persistence
    // measurement must still see the original input. Its strength and fine share: Grain
    // stability (ODParams.grainStability), unless OD_SPATIAL_TUNE names `stable` (diagnostic
    // runs: then its stable and fine, fine 1 unless given, exactly as before the control
    // existed).
    id<MTLTexture> blendRef = orig;
    const float stable = t_.stableGiven ? t_.stable : p.grainStability;
    const float fine = t_.stableGiven || t_.fineGiven ? t_.fine : kGrainStabilityFine;
    if (stable > 0 && count && p.lumaStrength > 0) {
      blendRef = env.temp(kWorkingFormat, W, H);
      if (!blendRef) return nil;
      // Fade the reference substitution to zero at either public strength's zero endpoint.
      // At the tested defaults (.8,.5), this multiplier is exactly one, so the measured
      // variant is unchanged.
      const float enabled = std::min(p.lumaStrength / 0.8f, 1.0f) * std::min(p.spatialStrength / 0.5f, 1.0f);
      const float u[4] = {enabled * std::clamp(stable, 0.0f, 1.0f), std::clamp(fine, 0.0f, 1.0f), 0, 0};
      id<MTLComputeCommandEncoder> enc = env.compute();
      [enc setTexture:orig atIndex:0]; [enc setTexture:image atIndex:1];
      [enc setTexture:blendRef atIndex:2]; [enc setBytes:u length:sizeof(u) atIndex:0];
      env.dispatch(stableRef_, W, H);
    }

    const bool ownCount = count != nil;  // the temporal stage's map, not the test aid's
    const bool measured = ownCount && p.temporalNoiseModel == 1;
    const bool blendTemporal = measured && p.temporalStrength < 1.0f && t_.diagnostic == 0;
    // The test aid can replace the base endpoint's count. In that case its
    // pyramid must be built normally with that count, rather than rho = 1.
    const bool reuseRaw = blendTemporal && testCount_n_ <= 0;
    if (testCount_n_ > 0 && !count) {
      id<MTLTexture> t = env.temp(MTLPixelFormatR16Float, W, H);
      if (!t) return nil;
      float u[4] = {testCount_n_, testCountSplit_ ? 1.0f : 0.0f, 0, 0};
      id<MTLComputeCommandEncoder> enc = env.compute((testCount_).label.UTF8String);
      [enc setTexture:t atIndex:0];
      [enc setBytes:u length:sizeof(u) atIndex:0];
      env.dispatch(testCount_, W, H);
      count = t;
    }
    id<MTLTexture> cnt = count ? count : one_;

    // ---- pyramid ----
    int K = 0;
    uint32_t lw[kMaxCoarse + 1], lh[kMaxCoarse + 1];
    lw[0] = W;
    lh[0] = H;
    while (K < kMaxCoarse) {
      uint32_t nw = (lw[K] + 1) / 2, nh = (lh[K] + 1) / 2;
      if (std::min(nw, nh) < (uint32_t)kMinTopSize) break;
      ++K;
      lw[K] = nw;
      lh[K] = nh;
    }
    // Chroma noise that persists between frames (see below): T_1 is made
    // with G_1.
    const int m = (int)t_.cpersist;
    const int applyFrom = std::max(0, (int)t_.cfrom);
    const bool usePersist = !measured && ownCount && p.chromaStrength > 0 && m >= 1 && m <= K;
    id<MTLTexture> T[kMaxCoarse + 1] = {};
    id<MTLTexture> G[kMaxCoarse + 1];
    G[0] = image;
    for (int k = 1; k <= K; ++k) {
      if (basePyramid) {
        G[k] = basePyramid[k];
        continue;
      }
      G[k] = env.temp(kWorkingFormat, lw[k], lh[k]);
      if (!G[k]) return nil;
      if (k == 1 && usePersist) {
        T[1] = env.temp(MTLPixelFormatRG16Float, lw[1], lh[1]);
        if (!T[1]) return nil;
      }
      id<MTLComputeCommandEncoder> enc = env.compute((k == 1 ? (usePersist ? downFirstT_ : downFirst_) : down_).label.UTF8String);
      [enc setTexture:G[k - 1] atIndex:0];
      if (k == 1) {
        [enc setTexture:cnt atIndex:1];
        [enc setTexture:G[k] atIndex:2];
        if (usePersist) {
          [enc setTexture:orig atIndex:3];
          [enc setTexture:T[1] atIndex:4];
        }
        env.dispatch(usePersist ? downFirstT_ : downFirst_, lw[k], lh[k]);
      } else {
        [enc setTexture:G[k] atIndex:1];
        env.dispatch(down_, lw[k], lh[k]);
      }
    }

    // Paired raw/merged energies, independently per band and channel. The
    // maps are built before coarse-to-fine reconstruction modifies anything.
    id<MTLTexture> residual[kMaxCoarse + 1] = {};
    id<MTLTexture> raw[kMaxCoarse + 1] = {orig};
    if (measured) {
      for (int k = 1; k <= K; ++k) {
        raw[k] = env.temp(kWorkingFormat, lw[k], lh[k]);
        if (!raw[k]) return nil;
        auto enc = env.compute("spatial_raw_down");
        [enc setTexture:raw[k-1] atIndex:0];
        if (reuseRaw && k == 1) {
          // The calibration consumes RGB only. Store rho = 1 in alpha so
          // the base endpoint sees exactly spatial_down_first(orig, one).
          [enc setTexture:one_ atIndex:1]; [enc setTexture:raw[k] atIndex:2];
          env.dispatch(downFirst_, lw[k], lh[k]);
        } else {
          [enc setTexture:raw[k] atIndex:1];
          env.dispatch(down_, lw[k], lh[k]);
        }
      }
      for (int k = 0; k <= K; ++k) {
        const bool top = k == K;
        struct ResidualUniforms { float sig[OD_LEVELS][4]; uint32_t flags[4]; } ru = {};
        ru.flags[0] = top;
        for (int l = 0; l < OD_LEVELS; ++l)
          for (int c = 0; c < 3; ++c)
            ru.sig[l][c] = levelSigma(env.profile, c, k, top, l) * env.sigmaScale();
        residual[k] = env.temp(kWorkingFormat, (lw[k]+7)/8, (lh[k]+7)/8);
        if (!residual[k]) return nil;
        auto enc = env.compute(residualRatio_.label.UTF8String);
        [enc setTexture:raw[k] atIndex:0]; [enc setTexture:raw[top ? k : k+1] atIndex:1];
        [enc setTexture:G[k] atIndex:2]; [enc setTexture:G[top ? k : k+1] atIndex:3];
        [enc setTexture:residual[k] atIndex:4]; [enc setTexture:cnt atIndex:5];
        [enc setBytes:&ru length:sizeof(ru) atIndex:0];
        [enc setComputePipelineState:residualRatio_];
        [enc dispatchThreadgroups:MTLSizeMake((lw[k]+7)/8, (lh[k]+7)/8, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
      }
      if (!reuseRaw) for (int k = 1; k <= K; ++k) env.recycle(raw[k]);
    }

    // ---- chroma noise that persists between frames ----
    // T_k: pyramid of (original - merged), chroma only, up to level m + 1;
    // P: the share of the chroma noise the merge did not average away
    // although the count map says it did, measured on level m
    // (spatial_chroma_persist). Only with the temporal stage's own count map:
    // without a merge there is nothing to measure.
    id<MTLTexture> P = nil;
    if (usePersist) {
      const int top = std::min(m + 1, K);
      for (int k = 2; k <= top; ++k) {
        T[k] = env.temp(MTLPixelFormatRG16Float, lw[k], lh[k]);
        if (!T[k]) return nil;
        id<MTLComputeCommandEncoder> enc = env.compute((down_).label.UTF8String);
        [enc setTexture:T[k - 1] atIndex:0];
        [enc setTexture:T[k] atIndex:1];
        env.dispatch(down_, lw[k], lh[k]);
        if (k - 1 < m) env.recycle(T[k - 1]);
      }
      PersistUniforms pu;
      memset(&pu, 0, sizeof(pu));
      for (int l = 0; l < OD_LEVELS; ++l)
        for (int c = 1; c < 3; ++c) {
          pu.sig[l][c] = levelSigma(env.profile, c, m, m == K, l) * env.sigmaScale();
          // The editor's correction of this band's chroma noise, as noiseLevel.
          const float adj = env.noiseAdjust(c, std::min(m, 3));
          if (adj != 1.0f) pu.sig[l][c] *= adj;
        }
      pu.flags[0] = m == K ? 1u : 0u;
      pu.gate[0] = t_.cgate0;
      pu.gate[1] = t_.cgate1;
      P = env.temp(MTLPixelFormatRG16Float, lw[m], lh[m]);
      if (!P) return nil;
      id<MTLComputeCommandEncoder> enc = env.compute((chromaPersist_).label.UTF8String);
      [enc setTexture:T[m] atIndex:0];
      [enc setTexture:(m == K ? T[m] : T[m + 1]) atIndex:1];
      [enc setTexture:G[m] atIndex:2];
      [enc setTexture:(m == K ? G[m] : G[m + 1]) atIndex:3];
      [enc setTexture:P atIndex:4];
      [enc setBytes:&pu length:sizeof(pu) atIndex:0];
      env.dispatch(chromaPersist_, lw[m], lh[m]);
      env.recycle(T[m]);
      if (m < K) env.recycle(T[m + 1]);
    }

    // ---- coarse to fine ----
    const float tau = 2.0f * p.spatialStrength * env.sigmaScale();
    id<MTLTexture> dnCoarse = nil;
    for (int k = K; k >= 0; --k) {
      const bool top = (k == K);
      const bool final = (k == 0);
      const bool persist = P && k >= applyFrom;
      SpatialUniforms u;
      memset(&u, 0, sizeof(u));
      for (int l = 0; l < OD_LEVELS; ++l) {
        for (int c = 0; c < 3; ++c) {
          float s = levelSigma(env.profile, c, k, top, l) * tau * (c ? t_.tc[k] : t_.ty[k]);
          // The editor's correction per band and channel (stage.h section 3),
          // after levelSigma, whose extrapolation to levels 4 and 5 reads the
          // measured profile. A one-level pyramid's top is the full band.
          const float adj = env.noiseAdjust(c, top && k == 0 ? -1 : std::min(k, 3));
          if (adj != 1.0f) s *= adj;
          u.sig[l][c] = std::max(s, 1e-7f);
        }
        u.sig[l][3] = lumaSigmaFull(env.profile, l);
      }
      u.amount[0] = p.lumaStrength;
      u.amount[1] = p.chromaStrength;
      u.amount[2] = t_.keep;
      if (p.grainFloor != 1.0f) u.amount[2] = t_.keep * p.grainFloor;
      if (env.grainProfileV2) u.amount[2] = 0.0f;  // fine grain retained by the bands step
      u.amount[3] = t_.outlierDark / std::sqrt(18.0f);
      u.shape[0] = 1.0f / t_.etaY;
      u.shape[1] = 1.0f / t_.etaC;
      u.shape[2] = t_.guide;
      u.shape[3] = t_.knee;
      u.robust[0] = t_.fy[k];
      u.robust[1] = t_.outlierLo / std::sqrt(18.0f);
      // Remove specks off (ODParams.speckRemoval 0): no pixel falls in the
      // outlier window. Finite: the unit is compiled with fast math.
      if (!p.speckRemoval) u.robust[1] = 1e30f;
      u.robust[2] = t_.outlier / std::sqrt(18.0f);
      u.robust[3] = t_.tcap;
      u.research[0] = std::clamp(t_.softspeck, 0.0f, 1.0f);
      u.research[1] = t_.diagnostic;
      u.flags[0] = final ? 1u : 0u;
      u.flags[1] = (k < (int)t_.lumaLevels && p.lumaStrength > 0) ? 1u : 0u;
      u.flags[2] = p.chromaStrength > 0 ? 1u : 0u;
      u.flags[3] = ownCount ? 1u : 0u;  // the result's alpha then carries rho for the sharpen stage

      // X_k. The top level of a real pyramid is G_K itself (its alpha already holds rho);
      // level 0 always goes through prepare to pick up rho. The diagnostic specializations
      // (softspeck, diagnostics) have no fused instance: they take the separate passes. The
      // final blend reference is blendRef (the input, or the opt-in stable reference).
      const bool research = t_.softspeck > 0 || t_.diagnostic >= 2;
      if (fused_ && final && !top && !persist && !research && !measured) {
        id<MTLTexture> dn = env.temp(kWorkingFormat, lw[k], lh[k]);
        if (!dn) return nil;
        id<MTLComputePipelineState> ps = (p.quality == 0 && t_.q0search < 2) ? fused3_ : fused5_;
        id<MTLComputeCommandEncoder> enc = env.compute(ps.label.UTF8String);
        [enc setComputePipelineState:ps];
        [enc setTexture:G[k] atIndex:0]; [enc setTexture:blendRef atIndex:1]; [enc setTexture:dn atIndex:2];
        [enc setTexture:dnCoarse atIndex:4]; [enc setTexture:G[k+1] atIndex:5]; [enc setTexture:cnt atIndex:6];
        [enc setBytes:&u length:sizeof(u) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((lw[k]+31)/32, (lh[k]+7)/8, 1) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
        env.recycle(dnCoarse); env.recycle(G[k+1]);
        dnCoarse = dn;
        continue;
      }
      id<MTLTexture> X = G[k];
      if (!top || final) {
        X = env.temp(kWorkingFormat, lw[k], lh[k]);
        if (!X) return nil;
        id<MTLComputeCommandEncoder> enc = env.compute((prepare_).label.UTF8String);
        [enc setTexture:G[k] atIndex:0];
        // No coarser level (tiny frame): any texture twice gives a zero correction.
        [enc setTexture:(top ? G[k] : dnCoarse) atIndex:1];
        [enc setTexture:(top ? G[k] : G[k + 1]) atIndex:2];
        [enc setTexture:cnt atIndex:3];
        [enc setTexture:X atIndex:4];
        [enc setBytes:&u length:sizeof(u) atIndex:0];
        env.dispatch(prepare_, lw[k], lh[k]);
        if (!top) {
          env.recycle(dnCoarse);
          env.recycle(G[k + 1]);
        }
      }

      id<MTLTexture> dn = env.temp(kWorkingFormat, lw[k], lh[k]);
      if (!dn) return nil;
      const bool small = (p.quality == 0 && k == 0 && t_.q0search < 2);
      id<MTLComputeCommandEncoder> enc = env.compute((measured ? (small ? residual3_ : residual5_) : persist ? nlm5p_ : small ? nlm3_ : nlm5_).label.UTF8String);
      [enc setTexture:X atIndex:0];
      [enc setTexture:(final ? blendRef : X) atIndex:1];
      [enc setTexture:dn atIndex:2];
      if (persist) [enc setTexture:P atIndex:3];
      if (measured) [enc setTexture:residual[k] atIndex:3];
      [enc setBytes:&u length:sizeof(u) atIndex:0];
      env.dispatch(measured ? (small ? residual3_ : residual5_) : research ? (persist ? nlm5pr_ : small ? nlm3r_ : nlm5r_) :
                              (persist ? nlm5p_ : small ? nlm3_ : nlm5_), lw[k], lh[k]);
      if (X != G[k]) env.recycle(X);
      if (measured) env.recycle(residual[k]);
      dnCoarse = dn;
    }
    id<MTLTexture> result = coarseCorrection(env, dnCoarse);
    if (result && blendTemporal) {
      // Strength is a line between the spatial-only and full temporal/spatial
      // endpoints, never a change of rejection or a second threshold curve.
      // No second spatial evaluation at the normal full-strength setting.
      id<MTLTexture> base = encodeWithPyramid(env, orig, nil, reuseRaw ? raw : nullptr);
      if (!base) return nil;
      id<MTLTexture> blended = env.tempWorking();
      if (!blended) return nil;
      auto enc = env.compute(temporalBlend_.label.UTF8String);
      [enc setTexture:base atIndex:0]; [enc setTexture:result atIndex:1];
      [enc setTexture:blended atIndex:2];
      [enc setBytes:&p.temporalStrength length:sizeof(float) atIndex:0];
      env.dispatch(temporalBlend_, W, H);
      env.recycle(base); env.recycle(result);
      result = blended;
    }
    return result;
  }

  id<MTLTexture> coarseCorrection(RenderEnv& env, id<MTLTexture> base) {
    if (env.params.chromaStrength <= 0 || !coarseChromaOn(env.params)) return base;
    bool active = false;
    int K = 5;
    for (int b = 0; b < OD_COARSE_BANDS; ++b)
      for (int l = 0; l < OD_LEVELS; ++l)
        if (coarseQualified(env.profile, b, l)) {
          active = true;
          if (b >= 5) K = std::max(K, b + 1);
        }
    if (!active) return base;
    struct U {
      float sig[OD_LEVELS][4];
    };
    id<MTLTexture> g[8] = {base};
    uint32_t w[8] = {env.width}, h[8] = {env.height};
    for (int k = 1; k <= K; ++k) {
      w[k] = (w[k - 1] + 1) / 2;
      h[k] = (h[k - 1] + 1) / 2;
      g[k] = env.temp(kWorkingFormat, w[k], h[k]);
      if (!g[k]) return nil;
      auto enc = env.compute();
      [enc setTexture:g[k - 1] atIndex:0];
      [enc setTexture:g[k] atIndex:1];
      env.dispatch(down_, w[k], h[k]);
      if (k > 1 && k < 4) env.recycle(g[k - 1]);
    }
    id<MTLTexture> parent = nil;
    U apply = {};
    for (int k = K; k >= 3; --k) {
      U u = {};
      const int b = k == K ? k - 1 : k - 3;
      for (int l = 0; l < OD_LEVELS; ++l) {
        u.sig[l][0] = std::max(2e-4f, levelSigma(env.profile, 0, std::min(k, 5), k == K, l) * 2);
        if (!coarseQualified(env.profile, b, l)) continue;
        u.sig[l][3] = 1;
        for (int c = 1; c < 3; ++c)
          u.sig[l][c] = env.profile.coarseSigma[c - 1][b][l] * (3 * env.params.spatialStrength) *
                        env.sigmaScale() * env.noiseAdjust(c, 3);
      }
      if (k == 3) apply = u;
      auto d = env.temp(MTLPixelFormatRG16Float, w[k], h[k]);
      if (!d) return nil;
      auto enc = env.compute();
      uint32_t top = k == K;
      [enc setTexture:g[k] atIndex:0];
      [enc setTexture:top ? g[k] : parent atIndex:1];
      [enc setTexture:d atIndex:2];
      [enc setBytes:&u length:sizeof(u) atIndex:0];
      [enc setBytes:&top length:sizeof(top) atIndex:1];
      env.dispatch(coarseFilter_, w[k], h[k]);
      if (parent) env.recycle(parent);
      parent = d;
    }
    // Fine reconstruction guard uses fine-band sigma, not the coarse one.
    for (int l = 0; l < OD_LEVELS; ++l) {
      apply.sig[l][0] = std::max(2e-4f, env.profile.sigma[0][0][l] * 3);
      for (int c = 1; c < 3; ++c) apply.sig[l][c] = std::max(2e-4f, env.profile.sigma[c][0][l] * 3);
    }
    auto dst = env.temp(kWorkingFormat, env.width, env.height);
    if (!dst) return nil;
    auto enc = env.compute();
    float amount = env.params.chromaStrength;
    [enc setTexture:base atIndex:0];
    [enc setTexture:parent atIndex:1];
    [enc setTexture:g[3] atIndex:2];
    [enc setTexture:dst atIndex:3];
    [enc setBytes:&apply length:sizeof(apply) atIndex:0];
    [enc setBytes:&amount length:sizeof(amount) atIndex:1];
    env.dispatch(coarseApply_, env.width, env.height);
    for (int k = 3; k <= K; ++k) env.recycle(g[k]);
    env.recycle(parent);
    return dst;
  }
};

}  // namespace

std::unique_ptr<SpatialStage> makeSpatialStage(Context& ctx, std::string* err) {
  auto s = std::make_unique<Spatial>();
  if (const char* e = getenv("OD_SPATIAL_TUNE")) s->t_.parse(e);
  if (s->fused_) {  // made only for sessions that opted in
    s->fused3_ = ctx.pipeline("spatial_fused3", err);
    s->fused5_ = ctx.pipeline("spatial_fused5", err);
    if (!s->fused3_ || !s->fused5_) return nullptr;
  }
  s->downFirst_ = ctx.pipeline("spatial_down_first", err);
  s->down_ = ctx.pipeline("spatial_down", err);
  s->prepare_ = ctx.pipeline("spatial_prepare", err);
  s->nlm5_ = ctx.pipeline("spatial_nlm5", err);
  s->nlm3_ = ctx.pipeline("spatial_nlm3", err);
  s->testCount_ = ctx.pipeline("spatial_test_count", err);
  s->downFirstT_ = ctx.pipeline("spatial_down_first_t", err);
  s->chromaPersist_ = ctx.pipeline("spatial_chroma_persist", err);
  s->coarseFilter_ = ctx.pipeline("spatial_coarse_filter", err);
  s->coarseApply_ = ctx.pipeline("spatial_coarse_apply", err);
  s->nlm5p_ = ctx.pipeline("spatial_nlm5_persist", err);
  s->stableRef_ = ctx.pipeline("spatial_stable_reference", err);
  s->residualRatio_ = ctx.pipeline("spatial_residual_ratio", err);
  s->residual5_ = ctx.pipeline("spatial_nlm5_residual", err);
  s->residual3_ = ctx.pipeline("spatial_nlm3_residual", err);
  s->temporalBlend_ = ctx.pipeline("spatial_temporal_blend", err);
  if (!s->residualRatio_ || !s->residual5_ || !s->residual3_ || !s->temporalBlend_) return nullptr;
  if (!s->downFirst_ || !s->down_ || !s->prepare_ || !s->nlm5_ || !s->nlm3_ || !s->testCount_ || !s->downFirstT_ ||
      !s->chromaPersist_ || !s->nlm5p_ || !s->coarseFilter_ || !s->coarseApply_ || !s->stableRef_)
    return nullptr;
  // Default sessions do not create the experimental pipeline states.
  if (s->t_.softspeck > 0 || s->t_.diagnostic >= 2) {
    s->nlm5r_ = ctx.pipeline("spatial_nlm5_research", err);
    s->nlm3r_ = ctx.pipeline("spatial_nlm3_research", err);
    s->nlm5pr_ = ctx.pipeline("spatial_nlm5_persist_research", err);
    if (!s->nlm5r_ || !s->nlm3r_ || !s->nlm5pr_) return nullptr;
  }
  if (s->t_.diagnostic == 3) {
    s->countView_ = ctx.pipeline("spatial_count_view", err);
    if (!s->countView_) return nullptr;
  }

  MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                                               width:1
                                                                              height:1
                                                                           mipmapped:NO];
  d.usage = MTLTextureUsageShaderRead;
  d.storageMode = MTLStorageModeShared;
  s->one_ = [ctx.device() newTextureWithDescriptor:d];
  if (!s->one_) {
    if (err) *err = "spatial: could not create the 1x1 count texture";
    return nullptr;
  }
  const uint16_t oneHalf = 0x3C00;
  [s->one_ replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&oneHalf bytesPerRow:2];

  // Test aid: OD_SPATIAL_TEST_COUNT=4 (count 4 everywhere) or =split:4 (left
  // half 1, right half 4) stands in for the temporal stage's count map when
  // it gives none. Read once; unset in production.
  if (const char* e = getenv("OD_SPATIAL_TEST_COUNT")) {
    if (strncmp(e, "split:", 6) == 0) {
      s->testCountSplit_ = true;
      e += 6;
    }
    s->testCount_n_ = (float)atof(e);
  }
  return s;
}

}  // namespace od
