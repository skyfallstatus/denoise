// SPDX-License-Identifier: MIT
// Open Denoise stage interface. THE CONTRACT between the pipeline (core/src/session.mm)
// and the stage implementations in core/src/stages/. Read all of it before
// writing a stage.
//
// ---------------------------------------------------------------------------
// 1. What a render looks like
// ---------------------------------------------------------------------------
// od_render() runs, in this order, on ONE command buffer (normally):
//
//   ingest    every frame of the window: host BGRA MTLBuffer -> working texture.
//             Cached in the session by frameKey.
//   profile   ProfileStage: measure noise on the current frame (asynchronous,
//             see section 5). Skipped when a manual profile is given.
//   temporal  TemporalStage::encode  -> image (+ optional per-pixel count)
//   spatial   SpatialStage::encode   -> image
//   bands     BandsStage::encode     -> image   (band amounts, or finest-grain mode)
//   sharpen   SharpenStage::encode   -> image
//   output    out = source pixel + toRGB(final image - ingested current frame)
//
// The pipeline, not the stage, decides whether a stage runs:
//   temporal  skipped if temporalRadius == 0, temporalStrength == 0, there is
//             no neighbour, or lumaStrength and chromaStrength are both 0
//   spatial   skipped if spatialStrength == 0, or luma and chroma strength
//             are both 0
//   sharpen   skipped if sharpen == 0
//   profile   skipped if a valid manual profile is given, or if none of the
//             three stages above runs
//   bands     runs if a band amount is below 1, temporal or spatial runs and
//             returned a new image, and the view is not Motion or Profile
// In the finest-grain mode (ODParams.grainMode = OD_GRAIN_FINEST, chosen per
// render; env.grainProfileV2) bands also runs for a positive luma amount and
// a positive grain floor.
// The decisions for this render are in env.temporalRuns / spatialRuns /
// sharpenRuns / bandsRun (debugSkipStages and the debug views included). A stage that
// needs to know what else runs reads those and nothing else.
//
// Amounts (params.lumaStrength / chromaStrength) are applied ONCE, by the last denoising
// stage that runs, as a blend against a reference: out = reference + amount * (denoised -
// reference) The reference is env.frames[0].tex (the input), with one exception: when
// spatial runs on a temporal result and ODParams.grainStability > 0, spatial's final LUMA
// blend uses R' = R - s * ((1 - fine) * (R - T) + fine * B(R - T)) R the input, T the
// temporal result, B a [1 2 1]/4 low-pass, s the control faded to 0 at either strength's
// zero (spatial_stable_reference). Chroma, the band amounts and a temporal-only render still
// blend against the input. So temporal applies them only when env.spatialRuns is false, and
// spatial always applies them. Sharpen never does. The luma amount is capped by the grain
// floor wherever it is applied (kGrainFloor below, od_luma_amount in common.metal), so the
// picture does not jump when spatialStrength reaches 0. The cap is computed from the input's
// sigma but applied to the reference: with Grain stability, R' holds less grain than the
// input, so the grain left can fall below kGrainFloor, by the ratio of R''s noise to the
// input's. The band amounts (params.bandAmount, API v6) are applied once too, by the
// pipeline's bands step (BandsStage below), after the last denoising stage and before
// sharpen, on image - env.frames[0].tex; the step keeps the alpha of its input. It runs only
// when a band amount is below 1 and temporal or spatial returned a new image. Stages never
// apply them. In the finest-grain mode (env.grainProfileV2), the last denoiser applies the
// uncapped luma amount. Bands caps only the finest tent-pyramid octave of image - input,
// using effectiveAmount / globalAmount as its extra weight; mid/coarse luma keeps the
// requested amount. Sharpen follows the cap only for its finest band. With neutral band
// amounts this needs only one half-size downsample and one full-size reconstruction. The
// full-band sigma calibration is retained; it is not a guarantee that measured output noise
// equals kGrainFloor. When everything is skipped the pipeline does not ingest at all (the
// frame is copied, or left alone when rendering in place). So a stage's encode is only
// called when it has work to do, and a stage does NOT have to be an exact identity at
// strength 0. It must, however, behave continuously as its strength goes to 0.
//
// Because output writes "source + delta", any pixel a stage returns unchanged
// (bit-equal to the ingested working pixel) comes out bit-identical to the
// host's pixel. Alpha always comes from the source; stages may ignore the A
// channel of the working textures entirely (leave whatever you like in it).
//
// ---------------------------------------------------------------------------
// 2. Working space
// ---------------------------------------------------------------------------
// Working textures are MTLPixelFormatRGBA16Float, full frame size, holding
// (Y, Cb, Cr, A): see core/shaders/common.metal for the exact matrix and the
// helpers od_rgb_to_ycc / od_ycc_to_rgb. Values are not clamped to [0,1]:
// negative values and super-whites are normal and must be preserved. Ingest
// guarantees there is no NaN and no Inf; a stage must not create any (guard
// divisions). The working format is 16f for 16f AND 32f hosts: the delta
// output makes a 32f working copy unnecessary (see output_buffer.metal).
// Stage-internal textures can use any format (r16Float luma pyramids, rg16Float
// motion vectors, ...).
//
// --------------------------------------------------------------------------- 3. Rules for
// encode --------------------------------------------------------------------------- *
// Append work to the render's command buffer through `env.compute` (shared compute encoder)
// or `env.blit` (shared blit encoder). Do not create your own encoders or command buffers,
// do not call endEncoding, commit or waitUntilCompleted. If you really need a private
// encoder, call env.endEncoding first, create it on env.commandBuffer, and end it before
// returning. Never keep env, its command buffer or its encoders beyond the call. * Input
// textures (WorkFrame::tex, the image handed to you, cached slots that are already filled)
// are READ-ONLY. They may be cache entries shared with other renders. Write only into
// textures you got from env.temp or into a slot from env.frameTexture that reported
// needsFill. * Return either the input image itself (nothing to do) or a texture obtained
// from env.temp with format RGBA16Float and the frame size. * Never allocate textures with
// the device directly on the render path: use env.temp (pooled, reused across renders) or
// env.frameTexture (cached per frame). Small MTLBuffers created once in the constructor are
// fine. * On failure call env.fail("...") and return false/nil. The render then returns an
// error; nothing is written to the output. * A stage object belongs to one session. Calls on
// a session are serialised, so encode needs no locking. Exception: ProfileStage::resolve
// (section 5). * Get pipeline states in the constructor: ctx.pipeline("temporal_merge",
// &err). Kernel names are global, prefix them with the stage name. A nil pipeline must make
// the factory fail (return nullptr with *err set); every od_render on that session then
// fails with that text (od_last_error). * Uniforms: [enc setBytes:length:atIndex:] with a
// struct that matches the shader's layout (mind float3/float4 alignment; plain uint/float
// fields and 16-byte multiples are the safe choice). * Dispatch with env.dispatch(pso,
// width, height): it picks the threadgroup size and uses dispatchThreads (non-uniform), so
// kernels need no bounds check, though one is harmless. * Honour env.params.quality: 0 =
// playback, 1 = full. * env.params.noiseLevel is a multiplier on every sigma of the profile;
// the pipeline does NOT pre-multiply. Use env.sigmaScale. Where a threshold decides what is
// noise, also multiply band b of channel c by env.noiseAdjust(c, b) (API v6, an editor's
// correction per band and channel), and only when it is not 1; motion search and cut
// statistics do not take it. Read the grain floor through env.grainFloor, never kGrainFloor
// directly. Every advanced value has a neutral default, and at it the stage must run exactly
// the arithmetic it ran before (guard each multiplication with "!= neutral"): the picture of
// existing projects does not change by a bit. * env.profile.valid can be 0 (first frames
// before the estimate exists, or the profile stage failed). Then do something safe: pass
// through.
//
// ---------------------------------------------------------------------------
// 4. Per-frame cached textures (session LRU)
// ---------------------------------------------------------------------------
// Premiere renders frames independently and in any order, and each render gets
// the whole temporal window again. Work that depends on ONE frame only (luma
// pyramid, down-scaled copies, per-frame feature maps) should be cached:
//
//   bool fill = false;
//   id<MTLTexture> pyr = env.frameTexture(frame, OD_SLOT('t','p','y','r'),
//                                         MTLPixelFormatR16Float, w, h, levels, &fill);
//   if (fill) { ...encode the work that writes pyr... }
//
// The texture lives in the session's LRU entry of that frame (keyed by
// frameKey + dimensions + host format), so the next render that has the same
// frame in its window as a NEIGHBOUR gets it back with fill == false. The
// current frame (env.frames[0]) is always ingested afresh and its slots are
// emptied first, so for it fill is always true: the output must never depend
// on a cached copy the host may have invalidated without changing the key. Frames without a key
// (OD_NO_KEY) get a plain temporary and fill == true every time: the same
// code path works. Results that depend on two frames (motion vectors towards
// the current frame) or on params/profile are NOT cacheable this way; use
// env.temp(). A slot's content must depend only on the frame's pixels, never
// on params (quality included): if you need a quality-dependent variant, use a
// different slot id for it.
//
// ---------------------------------------------------------------------------
// 5. Profile: asynchronous by design
// ---------------------------------------------------------------------------
// A noise estimate needs numbers back on the CPU, and waiting for the GPU
// inside od_render would stall Premiere's render thread. So:
//   * encodeMeasure() appends GPU work that reduces the frame to a SMALL
//     MTLBuffer of statistics (shared storage, allocated by the stage).
//   * When the command buffer has completed, the pipeline calls resolve() to
//     turn those statistics into an ODNoiseProfile, then smooth() to fold it
//     into the session's auto-profile slot. In the asynchronous case both run
//     on the session's own serial queue, in render order, not on Metal's
//     completion thread (resolve costs a few ms of CPU and the command queue
//     may be the host's).
//   * The render in progress uses the slot as it was BEFORE this frame's
//     measurement (a profile from the previous frames). Only the first
//     renders of a frame size and the decisive renders below block: the
//     pipeline commits ingest + measure, waits, resolves, and continues with
//     a fresh command buffer. A size blocks until it has an estimate of
//     confidence kUsableConfidence or more, for at most kColdWaits renders
//     (session.mm), so footage that never gives one (black, no flat area)
//     does not stall every render.
//   * params.realtime (the host is playing): nothing waits for the queue,
//     which is the host's and may hold several frames of its work. The
//     first estimate of a size is waited for at most kRealtimeWaitMs, once;
//     decisive estimates are resolved asynchronously like any other.
//   * With params.waitUntilCompleted the pipeline resolves before returning,
//     so results are deterministic in tests and exports.
//   * The pipeline does not re-measure a frameKey it measured recently.
//   * New footage under one session: on the first frame after a cut the
//     temporal stage detected (TemporalResult.gate, read back), and on a jump
//     to frames the session has not rendered recently (a render with
//     neighbours none of which, nor the frame itself, was rendered
//     recently), the pipeline measures the frame, blocks for the estimate
//     (unless realtime) and passes it to smooth() as decisive, so a far
//     estimate restarts the ring at once.
//
// ---------------------------------------------------------------------------
// 6. Debug views (ODParams.view)
// ---------------------------------------------------------------------------
//   OD_VIEW_NOISE    handled by output (0.5 + gain * (in - out)), nothing to do.
//   OD_VIEW_MOTION   TemporalStage writes a visualisation, as working-space
//                      Y,Cb,Cr, into its result image. The pipeline then skips
//                      spatial and sharpen and outputs the image as is.
//   OD_VIEW_PROFILE  ProfileStage::encodeView returns a working-space image;
//                      temporal, spatial and sharpen are skipped.
#ifndef OD_STAGE_H
#define OD_STAGE_H

#import <Metal/Metal.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "opendenoise/opendenoise.h"

#include "../context.h"

namespace od {

class Session;
struct KernelTraceState;

// The grain mode (ODParams.grainMode, API v10) is chosen per render. Tests may
// force it for a whole session: OD_GRAIN_PROFILE_V2 in the environment when
// the session is created, "1" = OD_GRAIN_FINEST, "0" = OD_GRAIN_WHOLE, -1 here
// = not set (the render's own field decides). Read once per session.
inline int grainModeOverride() {
  const char* e = std::getenv("OD_GRAIN_PROFILE_V2");
  if (!e) return -1;
  if (std::strcmp(e, "1") == 0) return OD_GRAIN_FINEST;
  if (std::strcmp(e, "0") == 0) return OD_GRAIN_WHOLE;
  return -1;
}
// The session's automatic profiles: one per grain mode (ProfileStage::smooth).
constexpr int kGrainModes = 2;

constexpr MTLPixelFormat kWorkingFormat = MTLPixelFormatRGBA16Float;

// Legacy luma grain floor: the amount cap aims to retain noise at
// this sigma, so a clean gradient still carries dither-level noise when it is
// quantised later (0.77 of a 10-bit code, 0.19 of an 8-bit code: it protects
// 10-bit deliveries, not 8-bit ones). The effective luma amount is
//   min(amount, 1 - kGrainFloor / sigmaFull(Y))
// with the profile's unscaled full-band luma sigma (lumaSigmaFull), so where
// the noise is under about twice the floor the slider has no effect above
// that cap. Applied by the stage that applies the amounts (temporal or
// spatial), and followed by sharpen when it estimates the noise left.
// The finest-grain mode keeps this calibration but applies the cap to the
// finest octave only.
constexpr float kGrainFloor = 7.5e-4f;
// Neutral value of ODParams.temporalEdgeGate: the share of a coarser merge
// level's rejection handed to the finer one (temporal.mm, kGateCoarse).
constexpr float kEdgeGateNeutral = 0.75f;
// An automatic noise estimate below this confidence is not trusted to stand
// for the footage on its own (profile.mm, fitConfidence: no sharp noise floor,
// most levels dropped as texture). It does not restart the session's profile
// at a change of footage (ProfileStage::smooth), and a frame size whose
// profile is below it has not finished its cold start: its next renders block
// for another estimate, up to kColdWaits of them (session.mm).
constexpr float kUsableConfidence = 0.3f;
// The coarse chroma estimate and correction of this render (API v9 ODParams.coarseChroma;
// the diagnostic switch OD_COARSE_CHROMA, read per call, overrides it: "1" on, any other
// value off). Off: no extra measurement or pass, and the API v8 profile fields stay zero.
inline bool coarseChromaOn(const ODParams& p) {
  if (const char* e = getenv("OD_COARSE_CHROMA")) return strcmp(e, "1") == 0;
  return p.coarseChroma != 0;
}
// API v8 coarse chroma fields (filled only when coarseChromaOn; coarse_profile.h).
// Coarse fields are independent of the original a-trous fit. Consumers must
// require coverage and confidence in BOTH colour channels, and treat invalid
// manual values as unavailable. No history or independent-frame credit here.
inline bool coarseQualified(const ODNoiseProfile& p, int b, int l) {
  if (b < 0 || b >= OD_COARSE_BANDS || l < 0 || l >= OD_LEVELS) return false;
  const float q = p.coarseConfidence[b][l], a = p.coarseCoverage[b][l];
  if (!std::isfinite(q) || q < .5f || q > 1 || !std::isfinite(a) || a <= 0 || a > 1) return false;
  for (int c = 0; c < 2; ++c)
    if (!std::isfinite(p.coarseSigma[c][b][l]) || p.coarseSigma[c][b][l] <= 0 ||
        p.coarseSigma[c][b][l] > 1) return false;
  return true;
}

// The profile's full-band luma sigma at knot l, for the grain floor. A profile
// without sigmaFull (older manual ones) falls back to band 0 times the
// full-band/band-0 ratio of white noise in the spatial stage's pyramid.
inline float lumaSigmaFull(const ODNoiseProfile& p, int l) {
  return p.sigmaFull[0][l] > 0 ? p.sigmaFull[0][l] : 1.12f * p.sigma[0][0][l];
}

// Slot id for RenderEnv::frameTexture: four characters, first one = stage
// initial (p, t, s, h) so ids never collide between stages.
#define OD_SLOT(a, b, c, d) \
  ((uint32_t)(uint8_t)(a) << 24 | (uint32_t)(uint8_t)(b) << 16 | (uint32_t)(uint8_t)(c) << 8 | (uint32_t)(uint8_t)(d))

// One frame of the temporal window, already ingested.
struct WorkFrame {
  id<MTLTexture> tex = nil;        // RGBA16Float Y,Cb,Cr,A, width x height. Read-only.
  int offset = 0;                  // frame offset relative to the current frame
  int64_t frameKey = OD_NO_KEY;  // OD_NO_KEY = not cacheable
  int cacheIndex = -1;             // internal (session LRU entry), do not touch
};

// Everything a stage may use during one render. Created by the pipeline, valid
// only for the duration of the encode() call it is passed to.
class RenderEnv {
 public:
  // ---- frame set -------------------------------------------------------
  uint32_t width = 0;   // frame size in pixels; all WorkFrames have this size
  uint32_t height = 0;
  // During profiling, v13 temporalNoiseModel selects the fixed analysis
  // subset; during temporal filtering it selects only the requested radius.
  // frames[0] is the current frame (offset 0). The others are the neighbours
  // the host supplied, restricted to |offset| <= params.temporalRadius, with
  // duplicates removed, sorted by |offset| ascending (nearest first; for equal
  // distance the past frame comes first). Missing neighbours are simply
  // absent: frameCount can be anything from 1 to 2*radius+1 and the window
  // can be one-sided (start/end of a clip).
  const WorkFrame* frames = nullptr;
  int frameCount = 0;

  // ---- parameters ------------------------------------------------------
  ODParams params;         // a copy; manualProfile must not be used by
                             // stages: read `profile` instead. The advanced
                             // fields (API v6) are sanitised: in range, and
                             // neutral when the caller left `advanced` at 0
  ODNoiseProfile profile;  // profile to use for this render (manual, or the
                             // session's smoothed auto profile). valid may be 0.
  bool grainProfileV2 = false;  // this render's grain mode is OD_GRAIN_FINEST (session.mm)
  float fineGrainOverride = -1.0f;  // spatial diagnostic tune `keep`, finest-grain mode only
  float sigmaScale() const { return params.noiseLevel; }
  // The editor's correction of channel c's noise in profile band b (b < 0:
  // the channel factor alone, for a full-band sigma), on top of sigmaScale().
  // 1 at the defaults: multiply by it only when it is not 1, so the default
  // path runs exactly the arithmetic it ran before (section 3).
  float noiseAdjust(int c, int b) const {
    const float ch = params.channelNoiseLevel[c];
    return b < 0 ? ch : ch * params.bandNoiseLevel[c ? 1 : 0][b < OD_BANDS ? b : OD_BANDS - 1];
  }
  // The luma grain floor of this render (kGrainFloor at the defaults).
  float grainFloor() const {
    if (grainProfileV2 && fineGrainOverride >= 0.0f) return fineGrainOverride;
    return params.grainFloor != 1.0f ? kGrainFloor * params.grainFloor : kGrainFloor;
  }
  // Which of the middle stages run in this render (stage.h section 1), set by
  // the pipeline before the first encode().
  bool temporalRuns = false;
  bool spatialRuns = false;
  bool sharpenRuns = false;
  // The bands step runs after the last denoising stage (finest grain, or a band amount below
  // 1, section 1). Only sharpen reads it, to estimate the noise left.
  bool bandsRun = false;

  // ---- GPU objects -----------------------------------------------------
  Context& ctx() const { return *ctx_; }
  id<MTLDevice> device() const { return ctx_->device(); }
  // The command buffer of this render. It can differ between stages (debug
  // timing mode), so never cache it.
  id<MTLCommandBuffer> commandBuffer() const { return cb_; }
  // Shared encoders. compute() ends an open blit encoder and vice versa; the
  // pipeline ends whatever is open. Consecutive compute passes from all stages
  // therefore share one encoder.
  id<MTLComputeCommandEncoder> compute(const char* traceLabel = nullptr);
  id<MTLBlitCommandEncoder> blit();
  void endEncoding();
  // setComputePipelineState + dispatchThreads over a w x h grid with a good
  // threadgroup size. Bind textures/buffers/bytes on env.compute() first.
  void dispatch(id<MTLComputePipelineState> pso, uint32_t w, uint32_t h);

  // ---- textures --------------------------------------------------------
  // Temporary texture from the session pool: private storage, usage
  // shaderRead|shaderWrite, contents undefined. Valid until the render ends;
  // it then goes back to the pool automatically. `mipLevels` > 1 allocates a
  // mip chain (fill it yourself or with env.blit() generateMipmaps).
  id<MTLTexture> temp(MTLPixelFormat format, uint32_t w, uint32_t h, uint32_t mipLevels = 1);
  // Shortcut: a working-format texture of the frame size.
  id<MTLTexture> tempWorking() { return temp(kWorkingFormat, width, height, 1); }
  // Optional: hand a temp() texture back early so a LATER pass of this render
  // can reuse the memory. Only call it after the last pass that touches the
  // texture has been encoded. Never recycle inputs, results or slot textures.
  void recycle(id<MTLTexture> tex);
  // Per-frame cached texture, see section 4. `frame` must be one of
  // env.frames. *needsFill is true when the caller must fill it in this render.
  // A slot is identified by (frame, slot id); asking for it again with another
  // format or size replaces it (needsFill = true).
  id<MTLTexture> frameTexture(const WorkFrame& frame, uint32_t slot, MTLPixelFormat format, uint32_t w,
                              uint32_t h, uint32_t mipLevels, bool* needsFill);

  // ---- errors ----------------------------------------------------------
  // printf-style. The first failure wins; the render returns non-zero and
  // od_last_error() returns the text.
  void fail(const char* fmt, ...) __attribute__((format(printf, 2, 3)));
  bool failed() const { return failed_; }

 private:
  friend class Session;
  Context* ctx_ = nullptr;
  Session* session_ = nullptr;
  id<MTLCommandBuffer> cb_ = nil;
  id<MTLComputeCommandEncoder> compute_ = nil;
  id<MTLBlitCommandEncoder> blit_ = nil;
  unsigned computeCount_ = 0, blitCount_ = 0;
  bool failed_ = false;
  std::shared_ptr<KernelTraceState> traceState_;
};

// ---------------------------------------------------------------------------
// profile
// ---------------------------------------------------------------------------
struct ProfileMeasurement {
  // Statistics written by the GPU work encodeMeasure() appended. Must be
  // MTLStorageModeShared so resolve() can read .contents. Allocate a fresh
  // (or pooled-by-you, thread-safely) buffer per measurement: several
  // measurements can be in flight at once. nil = "nothing was measured this
  // time" (allowed; resolve is then not called).
  id<MTLBuffer> data = nil;
  uint32_t width = 0, height = 0;  // frame size, set by the pipeline
  int64_t frameKey = OD_NO_KEY;  // set by the pipeline
  // Set by the pipeline before encodeMeasure(). force: measure this frame
  // whatever the throttle says. decisive: the footage may have changed here
  // (a detected cut, a jump to frames never rendered): smooth() replaces the
  // ring with this estimate if it is far from the slot, instead of waiting
  // for a second far estimate in a row.
  bool force = false;
  bool decisive = false;
  bool deterministic = false; // no history-dependent throttle or fit hysteresis
  // The neighbour the temporal estimator may pair the frame with: +1 or -1,
  // or 0 for the nearest one (-1 first). +1 on the first frame after a cut,
  // whose previous frame is another shot.
  int pairOffset = 0;
  std::shared_ptr<void> user;      // anything else resolve() needs
};

// Lightweight observational metadata, never read by a filter. Coordinates
// are candidates of the spatial profile fit. Protected by the session profile
// lock for ring metadata and the profiler's own lock for resolved tile data.
struct ProfileAnalysisInfo {
  uint32_t ringCount = 0, measuredKnots = 0, heldKnots = 0, tileCount = 0;
  uint32_t xy[128][2] = {};
  float energy[128][2] = {}; // profile fine/medium luma tile variances
};

class ProfileStage {
 public:
  virtual ~ProfileStage() = default;
  // Append GPU work that measures `cur` (the current frame) into m.data.
  // Return false after env.fail() on error. A throttle that depends on the
  // session's aggregate follows that of the render's grain mode
  // (env.grainProfileV2).
  virtual bool encodeMeasure(RenderEnv& env, const WorkFrame& cur, ProfileMeasurement& m) = 0;
  // CPU side, called once the command buffer has completed. MAY RUN ON
  // ANOTHER THREAD (the session's serial profile queue, never Metal's own
  // completion thread), concurrently with a later encodeMeasure() on the same
  // object: use only `m` and immutable or atomic members.
  // Fill *out (sigma in working-space units, see opendenoise.h) and return true; or
  // return false if the frame gave no usable estimate (slot is left alone).
  // *knots (may be null): bit l set = luminance knot l was measured on this
  // frame; the values at the other knots are interpolated or extrapolated.
  virtual bool resolve(const ProfileMeasurement& m, ODNoiseProfile* out, uint32_t* knots) = 0;
  // Fold a fresh estimate into the session's auto-profile slots, one per
  // grain mode (ODParams.grainMode): slots[m] is mode m's, nullptr = leave
  // that mode alone (its slot is held, Hold profile). Each mode keeps its
  // own history of the same estimates, so a render of either mode finds a
  // profile aggregated its way, and a session used in one mode only behaves
  // exactly as if the other did not exist. Called under the session's
  // profile lock. The default (session.mm) replaces an empty slot and
  // otherwise blends by confidence: slot += w * (fresh - slot) with
  // w = 0.25 * fresh.confidence / max(slot.confidence, fresh.confidence),
  // the same for both modes. knots: what resolve() reported (the default
  // ignores it). decisive: see ProfileMeasurement (the default ignores it).
  virtual void smooth(const ODNoiseProfile& fresh, uint32_t knots, ODNoiseProfile* const slots[kGrainModes],
                      bool decisive);
  // od_session_reset(): the session's slot has been emptied. Called on the
  // render thread, never during an encode. Forget everything about the
  // footage (throttles, warm-up) so the next render measures. Estimates of
  // earlier renders that are still in flight are dropped by the session.
  virtual void reset() {}
  virtual ProfileAnalysisInfo analysisInfo(int mode) { return {}; }
  // OD_VIEW_PROFILE: return a working-space image showing the regions used,
  // or nil to show the normal output.
  virtual id<MTLTexture> encodeView(RenderEnv& env, const WorkFrame& cur) {
    (void)env;
    (void)cur;
    return nil;
  }
};

// ---------------------------------------------------------------------------
// temporal
// ---------------------------------------------------------------------------
struct TemporalResult {
  id<MTLTexture> image = nil;  // working format, frame size. Required.
  // Optional, for the spatial stage: MTLPixelFormatR16Float, frame size, the
  // effective number of frames averaged at each pixel, N >= 1, defined so that
  // the remaining noise variance is about sigma^2 / N (1 = nothing merged,
  // e.g. occlusion). nil means 1 everywhere.
  id<MTLTexture> count = nil;
  int framesUsed = 1;  // frames that took part (goes to ODStats.framesUsed)
  // Optional, for the pipeline: MTLPixelFormatRG16Float, 1 x (frameCount - 1),
  // row a for env.frames[a + 1]: x = the neighbour's global weight, y = the
  // scene-cut value (0..1; a cut is handed on to the neighbours beyond it).
  // The pipeline reads it back to restart the automatic profile on the first
  // frame after a cut. nil = no cut detection.
  id<MTLTexture> gate = nil;
};

class TemporalStage {
 public:
  virtual ~TemporalStage() = default;
  // Merge env.frames[1..] into env.frames[0]. Only called with
  // env.frameCount >= 2. params.temporalStrength sets how readily a neighbour
  // is accepted. The luma / chroma amounts are applied here only when
  // env.spatialRuns is false (section 1); otherwise the full merge is handed
  // on and the spatial stage applies them.
  virtual bool encode(RenderEnv& env, TemporalResult* out) = 0;
};

// ---------------------------------------------------------------------------
// spatial
// ---------------------------------------------------------------------------
class SpatialStage {
 public:
  virtual ~SpatialStage() = default;
  // `image` is the temporal result, or env.frames[0].tex when temporal did not
  // run. `count` is TemporalResult::count or nil. Returns the filtered image,
  // or nil after env.fail().
  virtual id<MTLTexture> encode(RenderEnv& env, id<MTLTexture> image, id<MTLTexture> count) = 0;
};

// ---------------------------------------------------------------------------
// sharpen
// ---------------------------------------------------------------------------
class SharpenStage {
 public:
  virtual ~SharpenStage() = default;
  // `image` is the denoised image so far; env.frames[0].tex is the original.
  // `count` is TemporalResult::count (R16Float, frame size) when the temporal
  // stage ran and gave one, else nil (= 1 everywhere): where few frames were
  // merged more noise is left, and the coring threshold must follow.
  // Returns the sharpened image, or nil after env.fail().
  virtual id<MTLTexture> encode(RenderEnv& env, id<MTLTexture> image, id<MTLTexture> count) = 0;
};

// ---------------------------------------------------------------------------
// bands (a pipeline step, not an ODStage: its time counts under spatial)
// ---------------------------------------------------------------------------
class BandsStage {
 public:
  virtual ~BandsStage() = default;
  // `image` is the denoised image (working format, frame size), after the
  // last denoising stage. Returns
  //   original + sum_b w[c][b] * L_b(image - original),   alpha from image,
  // with L_b the tent-pyramid octaves (bands 0..3 of the profile; band 3
  // takes everything coarser) and w = params.bandAmount (luma for Y, chroma
  // for Cb and Cr); a channel group whose four weights are all 1 is copied
  // from `image`. nil after env.fail(). Only called when a weight is below 1.
  virtual id<MTLTexture> encode(RenderEnv& env, id<MTLTexture> image) = 0;
};

// Factories, one per stage, defined in the stage's .mm file. Called once per
// session. Return nullptr and set *err (naming the shader) on failure.
std::unique_ptr<ProfileStage> makeProfileStage(Context& ctx, std::string* err);
std::unique_ptr<TemporalStage> makeTemporalStage(Context& ctx, std::string* err);
std::unique_ptr<SpatialStage> makeSpatialStage(Context& ctx, std::string* err);
std::unique_ptr<SharpenStage> makeSharpenStage(Context& ctx, std::string* err);
std::unique_ptr<BandsStage> makeBandsStage(Context& ctx, std::string* err);

}  // namespace od

#endif  // OD_STAGE_H
