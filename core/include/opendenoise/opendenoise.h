// SPDX-License-Identifier: MIT
// Open Denoise — open-source video noise reduction core (Metal).
// Public C API. Usable from C, C++, Objective-C(++).
//
// Rules for changing this header: append fields at the END of structs, give
// every new field a documented default whose zero value reproduces the old
// behaviour (a zero-initialised struct renders as before; od_params_default()
// may pick another default on purpose, said with the field, as API v10's
// grainMode does), and bump OD_API_VERSION.
#ifndef OPENDENOISE_H
#define OPENDENOISE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OD_API_VERSION 13

#define OD_MAX_RADIUS 5   // max temporal radius (frames each side)
#define OD_MAX_FRAMES (2 * OD_MAX_RADIUS + 1)
#define OD_PROFILE_RADIUS 2 // v13 analysis window, independent of merge settings
#define OD_COARSE_BANDS 7 // direct tent L3..L6, detrended G5..G7
#define OD_BANDS 4        // spatial frequency bands, 0 = finest
#define OD_LEVELS 8       // luminance knots, uniform over Y in [0,1]
#define OD_NO_KEY INT64_MIN

// Pipeline stages, in execution order (API v2). Used by ODStats.stageMs and
// as bit positions in ODParams.debugSkipStages.
typedef enum {
  OD_STAGE_INGEST = 0,
  OD_STAGE_PROFILE = 1,
  OD_STAGE_TEMPORAL = 2,
  OD_STAGE_SPATIAL = 3,
  OD_STAGE_SHARPEN = 4,
  OD_STAGE_OUTPUT = 5,
  OD_STAGE_COUNT = 6
} ODStage;

// ODParams.debugFlags bits (API v2). All off by default.
#define OD_DEBUG_STAGE_TIMINGS 1u  // one command buffer per stage, each waited
                                     // for; fills ODStats.stageMs. Slow: for
                                     // measurement only. Implies waiting.
#define OD_DEBUG_IO_BLIT       2u  // ingest/output through blit + intermediate
                                     // texture instead of reading the MTLBuffer
                                     // in the kernel (measurement only)
#define OD_DEBUG_IO_VIEW       4u  // ingest/output through a linear texture
                                     // view of the MTLBuffer (measurement only;
                                     // fails if the buffer cannot be viewed)
#define OD_DEBUG_NO_ANALYSIS 16u // diagnostics: omit analysis work for interleaved A/B
#define OD_DEBUG_FORCE_PROFILE 8u  // measure the noise profile even when no
                                     // stage that needs it runs (measurement
                                     // of the profile stage alone)

typedef struct ODContext ODContext;  // one per MTLDevice; thread-safe
typedef struct ODSession ODSession;  // one per effect instance; owns caches;
                                         // calls on one session must be serialised

typedef enum {
  OD_PIXEL_BGRA32F = 0,  // 4 x float32 per pixel, memory order B,G,R,A
  OD_PIXEL_BGRA16F = 1,  // 4 x float16 per pixel, memory order B,G,R,A
} ODPixelFormat;

// A frame living in an id<MTLBuffer>. This is what Premiere hands a GPU filter
// on Metal. Rows are rowBytes apart; row 0 is the first row in memory.
typedef struct {
  void*    mtlBuffer;   // id<MTLBuffer>, unretained
  uint32_t width;
  uint32_t height;
  uint32_t rowBytes;
  ODPixelFormat format;
  int64_t  frameKey;    // stable identity of this frame (e.g. a hash of its
                        // time in the sequence). Tells the automatic profile
                        // which frames it has measured and where the footage
                        // jumps or cuts, and, when the caller vouches that a
                        // key names one content (ODParams.verifyFrameCache
                        // 0), lets the session reuse per-frame work between
                        // renders when the frame comes back as a neighbour
                        // (frames[0] is always processed afresh). OD_NO_KEY:
                        // none of that. A host that cannot tell when the
                        // content behind a key changes sets
                        // ODParams.verifyFrameCache (or noFrameCache).
} ODFrame;

// Noise model in Y, Cb, Cr working space. sigma[c][b][l] is the standard deviation of
// channel c (0=Y, 1=Cb, 2=Cr), B3 a-trous detail band b = c_b - c_(b+1), at luminance knot
// l. sigmaFull[c][l] measures c_0 - c_4; these correlated bands must not be combined by root
// sum of squares. Knots lie at Y = l / (OD_LEVELS - 1), with linear interpolation and
// clamping outside [0,1].
typedef struct {
  float    sigma[3][OD_BANDS][OD_LEVELS];
  float    sigmaFull[3][OD_LEVELS];
  float    confidence;  // 0..1, how trustworthy the estimate is
  uint32_t valid;       // 0 = empty profile
  // --- API v8 --- Optional direct coarse chroma measurements, filled only when the
  // diagnostic switch OD_COARSE_CHROMA=1 is set. Zero = unavailable, exactly the pre-v8
  // picture. No extrapolation across unsupported knots. Sigma is per-frame (never divided by
  // the number of correlated frames). Invalid optional values are ignored by the coarse
  // consumer. Rebuild clients for v8: the appended data changes the size of ODNoiseProfile
  // and of ODStats, which holds one.
  float coarseSigma[2][OD_COARSE_BANDS][OD_LEVELS]; // Cb, Cr; working units
  float coarseConfidence[OD_COARSE_BANDS][OD_LEVELS]; // 0..1; qualification >= .5
  float coarseCoverage[OD_COARSE_BANDS][OD_LEVELS]; // accepted fraction of raster
  float coarseCorrelation[OD_COARSE_BANDS][OD_LEVELS]; // mean pair correlation
} ODNoiseProfile;

typedef enum {
  OD_VIEW_OUTPUT = 0,
  OD_VIEW_NOISE = 1,    // what was removed, amplified around mid-grey
  OD_VIEW_MOTION = 2,   // motion / temporal weight visualisation
  OD_VIEW_PROFILE = 3,  // region(s) used for auto profiling
  OD_VIEW_SPLIT = 4,    // input on the left half, result on the right, 1 px divider
  OD_VIEW_ANALYSIS = 5, // result, with the analysis panel drawn over the lower left
} ODView;

// ODParams.grainMode (API v10): where the luma grain floor applies, and how
// the automatic profile treats luminance knots a frame did not measure.
typedef enum {
  OD_GRAIN_WHOLE = 0,   // the picture of API v9, bit for bit: the floor caps
                        // the luma amount of every size of noise; the
                        // automatic profile averages its 24-estimate ring at
                        // every knot, interpolated ones included
  OD_GRAIN_FINEST = 1,  // the floor keeps only the finest grain (the finest
                        // octave of the removed signal, about 1-2 px); medium and coarse
                        // luma noise go by lumaStrength over its whole range. A knot no
                        // estimate of the ring measured holds its last measured value
} ODGrainMode;

typedef struct {
  float lumaStrength;      // 0..1   default 0.8  amount of luma noise removed
  float chromaStrength;    // 0..1   default 1.0  amount of chroma noise removed
  int   temporalRadius;    // 0..OD_MAX_RADIUS  default 2; 0 = spatial only
  float temporalStrength;  // 0..1   default 1.0  how readily a neighbour is
                           // accepted into the merge; 0 = no temporal merge
  float spatialStrength;   // 0..1   default 0.5  spatial filter threshold
                           // (0.5 = thresholds at the profiled noise level);
                           // 0 = no spatial filter
  float sharpen;           // 0..1   default 0.0  detail above the noise floor
  float noiseLevel;        // multiplier on the profiled sigma, default 1.0;
                           // 0..1000 (NaN or negative = 1.0)
  int   quality;           // 0 = playback (fastest), 1 = full, default 1.
                           // 0: one motion refinement step instead of two, a
                           // 3x3 instead of 5x5 search on the finest spatial
                           // level, sharpen on band 0 only, and the noise
                           // profile measured on one frame in four
  int   view;              // ODView, default OD_VIEW_OUTPUT
  int   waitUntilCompleted;// 1 = block until the GPU work is done, default 0
  const ODNoiseProfile* manualProfile; // NULL = auto profile (default). Used
                           // when valid != 0; od_render fails if any sigma or
                           // sigmaFull is not finite, and clamps them to
                           // [0, 1e6] (confidence to [0, 1])
  // --- API v2 ---
  uint32_t debugFlags;      // OD_DEBUG_* bits, default 0
  uint32_t debugSkipStages; // bit (1u << ODStage) set = skip that stage.
                            // Only profile, temporal, spatial and sharpen can be
                            // skipped; default 0 = run everything
  // --- API v3 ---
  uint32_t noFrameCache;    // 1 = take nothing from the frame cache and put
                            // nothing in it: every frame is ingested afresh,
                            // and frameKey only identifies frames for the
                            // automatic profile. For hosts that cannot tell
                            // when the content behind a key changed (an effect
                            // below this one, a proxy toggle). Default 0
  // --- API v4 ---
  uint32_t realtime;        // 1 = the host is playing back and a stall drops
                            // frames: with waitUntilCompleted = 0, od_render
                            // does not wait for the GPU to run the work queued
                            // ahead of its own on the host's command queue.
                            // The automatic profile's first estimate for a
                            // frame size is waited for at most a few
                            // milliseconds (until it lands the frame borrows
                            // another size's profile, or passes through), and
                            // its restarts at cuts and jumps are measured
                            // without blocking. Not a bound on the call's
                            // time: obtaining a command buffer blocks while
                            // the queue has none free, and a call waits for
                            // room in the device's memory budget (at most
                            // 4 ms) and for another call on the same session
                            // to finish (measured: ODStats.commandBufferMs
                            // and lockWaitMs). 0 (default) = block for those
                            // estimates, as an export wants: every frame from
                            // the first uses a measured profile
  // --- API v5 ---
  uint32_t verifyFrameCache;// 1 = (with noFrameCache 0) the keys do not
                           // promise the pixels behind them (an effect below this one, a
                           // proxy toggle): nothing derived from a frame is reused between
                           // renders, every frame of the window is ingested from the host's
                           // buffer and its pyramid rebuilt: the pixels are those of
                           // noFrameCache. The picture is noFrameCache's bit for bit with a
                           // manual profile, and under the automatic profile while no frame
                           // comes back after its entry went (forward playback, steps back
                           // inside the window): a key that returns after its entry was
                           // evicted is measured again (nothing can tell whether its pixels
                           // are still those the profile learnt from), where noFrameCache,
                           // which keeps no entries, still counts it among its last 64
                           // measured keys, so from the next render the profiles and the
                           // pictures differ (tests/core verified_revisit: a scrub back, rms
                           // up to 1.7 10-bit codes). The keys still serve the automatic
                           // profile, and the session keeps samples of each keyed frame (one
                           // pixel per 8x8 block, 2 MB at 3840x2160, in the context's
                           // budget): new pixels under the key of the current frame make the
                           // next render measure the noise profile decisively (a sampled
                           // heuristic for the profile only). 0 (default) with
                           // keys: the caller vouches that a key names one content (a host
                           // with a trustworthy content generation mixes it into the key)
                           // and the frame cache reuses ingested neighbours and their
                           // pyramids --- API v6: advanced controls --- Shot-by-shot
                           // corrections on top of the controls above. Every one has a
                           // neutral value that renders exactly the picture of API v5, bit
                           // for bit, with no extra GPU work.
  uint32_t advanced;       // 1 = the fields below are used. 0 = every one of
                           // them takes its neutral value, so a
                           // zero-initialised ODParams renders as before.
                           // od_params_default() sets 1 and the neutral values
  float bandNoiseLevel[2][OD_BANDS]; // [0 luma, 1 chroma (Cb and Cr)][band, 0 =
                           // finest; 3 also covers everything coarser]:
                           // multiplier on that band's profiled sigma, on top
                           // of noiseLevel and channelNoiseLevel, wherever a
                           // threshold decides what is noise (temporal merge,
                           // spatial filter, sharpen coring; not motion search
                           // or cut detection). 0.25..4, neutral 1
  float channelNoiseLevel[3]; // Y, Cb, Cr: multiplier on that channel's
                           // profiled sigma, every band. 0.25..4, neutral 1
  float bandAmount[2][OD_BANDS]; // [luma, chroma][band]: share of the removed
                           // signal of that band that stays removed, after lumaStrength /
                           // chromaStrength (they multiply). 0..1, neutral 1. Any value
                           // below 1 adds the bands step
  float grainFloor;        // multiplier on the luma grain floor (sigma
                           // 7.5e-4). 0 = no floor. 0..10, neutral 1
  uint32_t speckRemoval;   // 1 = the spatial filter treats single pixels far
                           // off their neighbours as noise (neutral); 0 = off
  float temporalMotionThreshold; // multiplier on the merge's full-rejection
                           // threshold. 0.5..2, neutral 1
  float temporalEdgeGate;  // share of a coarser level's rejection the merge
                           // applies to the finer one. 0..1, neutral 0.75
  float sharpenBand[2];    // multipliers on the sharpen gains of bands 0 and
                           // 1. 0..2.5, neutral 1
  float sharpenCoring;     // multiplier on the sharpen coring threshold.
                           // 0.5..3, neutral 1
  float sharpenHalo;       // multiplier on the sharpen overshoot allowance.
                           // 0..5, neutral 1
  uint32_t profileHold;    // 1 = once the automatic profile of a frame size is
                           // established (8 estimates, usable confidence), stop measuring
                           // and keep it: no drift, no restart at cuts or jumps. Neutral 0.
                           // Kept in memory only (od_session_reset clears it) --- API v9:
                           // experimental controls --- Both neutral at 0, so a zero-
                           // initialised ODParams renders as before; like the v6 fields they
                           // take their neutral value when `advanced` is 0.
  float grainStability;    // 0..1, neutral 0. The final luma blend puts back
                           // (1 - effective luma amount) of a reference; at 0 that is the
                           // input (raw grain). Above 0 the reference takes s =
                           // grainStability of the residual input - temporal merge out of
                           // it, 25 % of the fine part and all of the part under a [1 2 1]/4
                           // tent: R' = R - s (0.25 (R-T) + 0.75 B(R-T)). At 1 it is the
                           // diagnostic setting OD_SPATIAL_TUNE="stable=1,fine=0.75". Luma
                           // only; needs a temporal merge (radius > 0) and a luma amount
                           // below 1, and fades out towards lumaStrength 0 and
                           // spatialStrength 0. One extra pass when it acts. OD_SPATIAL_TUNE
                           // with a `stable` key overrides it
  uint32_t coarseChroma;   // 1 = measure the confidence-qualified coarse
                           // chroma noise (ODNoiseProfile API v8 fields) from the supplied
                           // +/-2 window and correct coarse Cb/Cr blotches where it
                           // qualifies, as the diagnostic switch OD_COARSE_CHROMA=1. Neutral
                           // 0. OD_COARSE_CHROMA in the environment overrides it ("1" on,
                           // any other value off) --- API v10 ---
  uint32_t grainMode;      // ODGrainMode. 0 (a zero-initialised ODParams, or
                           // `advanced` 0) = OD_GRAIN_WHOLE, the picture of API v9 bit for
                           // bit; od_params_default sets OD_GRAIN_FINEST, the default of a
                           // newly applied effect (the plug-in's "Minimum grain applies
                           // to"). Any other value = WHOLE. Chosen per render; the session
                           // keeps one automatic profile per mode, both fed by the same
                           // estimates. For tests, OD_GRAIN_PROFILE_V2 in the environment
                           // when a session is created overrides it for that session ("1"
                           // FINEST, "0" WHOLE) --- API v13 ---
  uint32_t chromaRecovery; // 1 = recover an automatic chroma fit collapsed by
                           // quantized/constant chroma tiles. Neutral/default
                           // 0 preserves API v10 bit for bit. Only a collapsed
                           // channel is refitted; luma and healthy channels
                           // retain the legacy fit. Manual profiles unchanged.
                           // Set to 1 only for new effects; rebuild all clients.
  uint32_t temporalNoiseModel; // 1 = estimate remaining noise from per-band,
                           // per-channel merge correction energy, rather than
                           // the nominal count. temporalStrength interpolates
                           // spatial-only and full temporal/spatial endpoints
                           // (with no spatial filter, input and guarded merge).
                           // Partial strength runs spatial twice; 1 only once.
                           // 0 (default, also advanced=0) = legacy picture,
                           // bit for bit. Other values take 0. New effects
                           // may opt in; saved effects must retain 0. Clients
                           // must be rebuilt: ODParams has grown.
                           // With 1, callers supply neighbours through at least
                           // OD_PROFILE_RADIUS even at radius/strength zero
                           // (subject to clip bounds). Only the requested
                           // temporalRadius is merged. Profiling uses the
                           // fixed +/-OD_PROFILE_RADIUS subset; missing frames
                           // remain missing, never fabricated.
  // Out of range values are clamped to the ranges above; a value that is not
  // finite takes its neutral value. ODStats.profile reports the profile as
  // measured, without these multipliers.
} ODParams;

// API v13: immutable description of the last successfully completed render.
// No pointers, ownership, or settings changes. Noise uses 876 video codes
// from black to white. profile[c][size][knot] uses the profile's a-trous
// bands; full[c][knot] is its full-band noise, not a sum of the four bands.
typedef enum {
  OD_ANALYSIS_AUTOMATIC = 0, OD_ANALYSIS_LEARNING = 1,
  OD_ANALYSIS_SETTLED = 2, OD_ANALYSIS_FROZEN = 3, OD_ANALYSIS_CAPTURED = 4
} ODProfileState;
typedef struct {
  uint32_t valid;          // 1 = a completed render; profileValid is separate
  uint32_t profileValid;
  uint64_t renderId;
  int64_t frameKey;
  uint32_t width, height;
  ODProfileState state;
  uint32_t learningFrames; // 0..8 distinct estimates used, capped at 8
  uint32_t capturedFrames; // 0 = caller supplied no capture provenance
  float profile[3][OD_BANDS][OD_LEVELS];
  float full[3][OD_LEVELS];
  uint8_t measured[OD_LEVELS]; // recent ring supports knot; 0 = held/interpolated
  uint8_t held[OD_LEVELS]; // 1 = previously measured but absent from recent ring
  float histogram[OD_LEVELS]; // area fractions, bins [l/8,(l+1)/8); sampled raster
  float effectiveLuma[OD_LEVELS]; // finest amount after Keep grain and fine limit
  float temporalSpatialRatio; // fine stddev(frame difference)/sqrt(2)/spatial stddev
  uint32_t temporalValid;  // 0 = no adjacent frame or too few flat tiles
  float confidence;       // profile evidence summary, NOT a probability
  uint32_t flatTiles;     // current tiles surviving revalidation, at most 128
  uint32_t removalValid;  // bit c*4+b: measured input energy, enough flat tiles
  // 10*log10(input energy/output energy), signed, clipped to [-30,60].
  // Measured on the profile's flat tiles, revalidated on this frame, with
  // four Haar octaves (1-2, 2-4, 4-8, 8+ px). Includes sharpen. This is an
  // observation of textured footage, not a clean-reference accuracy score.
  float removedDb[3][OD_BANDS];
  float removedOverallDb[3]; // ratio of summed octave energies (not mean dB)
  float inputEnergy[3][OD_BANDS]; // working-unit energy, for weighted summaries
  float outputEnergy[3][OD_BANDS];
} ODAnalysis;

// Returns 1 and copies a snapshot, 0 (and zeroes out) before completion or
// after reset or during brief publication contention, -1 for NULL arguments.
// Never waits for GPU work or the publication lock. A blocking
// render publishes before returning; asynchronous renders publish at GPU
// completion, may lag by a frame, and stale pre-reset completions are dropped.
int od_session_last_analysis(ODSession*, ODAnalysis*);
// Two ASCII strings, always NUL terminated, at most 31 bytes. NULL analysis
// means no analysis yet; NULL params uses defaults. Never changes settings.
void od_analysis_status(const ODAnalysis*, const ODParams*, char row1[32], char row2[32]);

typedef enum { OD_DRAW_LINE, OD_DRAW_DOT, OD_DRAW_RECT, OD_DRAW_TEXT } ODDrawKind;
typedef struct {
  ODDrawKind kind;
  float x, y, w, h; // top-left coordinates; LINE ends at (w,h); DOT radius=w
  float rgba[4];    // unpremultiplied sRGB
  float stroke;    // width; DOT/RECT 0 = filled, positive = outline
  float dash;      // LINE: 0 = solid, positive = dash length (gap equals dash)
  float fontSize;  // TEXT; y is top, h is line height
  char text[96];   // UTF-8, owned; renderer uses a system sans-serif font
} ODDrawCommand;
#define OD_DRAW_CAPACITY 320
typedef struct {
  uint32_t count;
  float width, height;
  ODDrawCommand commands[OD_DRAW_CAPACITY];
} ODDrawList;
// Same drawing for Drawbot and the picture. No allocation, GPU work, or host
// dependency. Returns command count (0 for NULL list/nonpositive dimensions).
int od_analysis_layout(const ODAnalysis*, const ODParams*, ODDrawList*, float w, float h);
// Pixel box of the in-picture panel, also used by the exactness test. Origin
// is top left. Four outputs must be non-NULL. Small frames clip the panel.
void od_analysis_box(uint32_t frameWidth, uint32_t frameHeight,
                     uint32_t* x, uint32_t* y, uint32_t* w, uint32_t* h);

// One render's GPU work, reported once it has completed (ODStats.completed,
// API v7).
#define OD_MAX_COMPLETIONS 8
typedef struct {
  uint64_t renderId;       // ODStats.renderId of that render
  double   gpuMs;          // GPU time of its last command buffer (of all of
                           // them when the render waited)
  double   queueMs;        // that command buffer: commit -> GPU start
  double   latencyMs;      // commit -> completed
  int      failed;         // 1 = it failed on the GPU: the output is not valid
} ODCompletion;

typedef struct {
  double gpuMs;            // GPU time of this render when waitUntilCompleted
                           // is set; otherwise that of the most recent render
                           // that has completed (0 if none yet)
  double cpuMs;            // CPU wall time spent inside od_render
  int    framesUsed;       // frames handed to the temporal merge, the current
                           // one included (1 = no temporal merge). Neighbours
                           // rejected on the GPU (another shot, exact repeats,
                           // occlusions) are still counted
  int    cacheHits;        // per-frame cache hits in this render
  ODNoiseProfile profile;// profile used for this render
  // --- API v2 ---
  double stageMs[OD_STAGE_COUNT]; // GPU ms per stage; only filled with
                                    // OD_DEBUG_STAGE_TIMINGS, else 0
  // --- API v5 ---
  double waitMs;           // CPU ms this call spent waiting for the GPU
                           // (blocking renders, the profile's estimates)
  double gpuQueueMs;       // the most recent render that has completed: ms
                           // from its commit until the GPU started it (work
                           // queued ahead of it on the command queue); 0 if
                           // none yet
  double gpuLatencyMs;     // the same render: ms from commit to completion
  int    verifiedFrames;   // always 0: neighbours taken from the frame cache
                           // after a GPU check, which ODParams.verifyFrameCache
                           // no longer does (kept for the layout)
  // --- API v7 ---
  double   lockWaitMs;     // ms this call waited before it started: for
                           // room in the device's memory budget (renders in
                           // progress on the device, od_session_create), then
                           // for the session: another call on the same
                           // session in progress (or the memory policy
                           // trimming it)
  double   commandBufferMs;// ms this call spent obtaining command buffers
                           // from the queue. Metal blocks there while the
                           // queue has no free command buffer (its capacity
                           // taken by work queued ahead, the host's and
                           // ours); not part of waitMs, and a realtime render
                           // can wait here too
  uint64_t renderId;       // this call's id: unique in the process and
                           // increasing; 0 if its arguments were rejected
  int      inFlight;       // asynchronous work of this session not finished
                           // when the call returned (renders that did not
                           // wait, this one included, and the noise
                           // estimates queued after them)
  int      asyncFailures;  // renders of this session that did not wait and
                           // were found failed on the GPU since the previous
                           // call: their od_render returned 0 but their
                           // output is not valid. od_first_async_error()
                           // gives the first message
  int      completedCount; // entries filled in completed[]
  int      completedDropped; // completions lost since the previous call
                           // because more than 64 waited to be reported
  ODCompletion completed[OD_MAX_COMPLETIONS]; // renders of this session whose
                           // GPU work completed since the previous call, each
                           // reported exactly once, oldest first (a render
                           // that waited reports itself here, in its own
                           // call); more than OD_MAX_COMPLETIONS wait for the
                           // next call
} ODStats;

// device: id<MTLDevice>. queue: id<MTLCommandQueue> or NULL (one is created).
// On failure returns NULL and writes a message to err (if errLen > 0).
ODContext* od_context_create(void* device, void* queue, char* err, size_t errLen);
void         od_context_destroy(ODContext*);

// Sessions of one context share its pool of scratch textures; every context on one device
// shares one byte budget over the pools, every session's frame cache and the renders in
// progress (1/8 of the device's recommended working set, 1 to 8 GiB; the environment
// variable OD_MEMORY_MB, read when a context is created, sets it, 0 = no limit; contexts
// made with another setting have a budget of their own). A render waits at its start while
// the renders in progress on the device leave no room for it (a realtime one at most 4 ms,
// others at most 2 s), and is never refused for the budget. A session idle for
// OD_IDLE_TRIM_MS (default 3000; 0 = never) gives its frame cache back, the session used
// last after five times that; the automatic profile is kept.
ODSession* od_session_create(ODContext*);
void         od_session_destroy(ODSession*);
void         od_session_reset(ODSession*);   // drop caches and auto profile

ODParams   od_params_default(void);

// Denoise frames[0] into *out. frames[1..frameCount-1] are temporal
// neighbours; offsets[i] is the frame offset of frames[i] relative to
// frames[0] (offsets[0] == 0), in any order. Missing neighbours are fine:
// pass fewer frames. out may alias frames[0] (in-place). All frames must
// share width, height and format with out.
// Returns 0 on success, non-zero on error (message via od_last_error), also
// when the GPU could not be reached (no command buffer or no encoder): then
// *out is not written. With waitUntilCompleted = 0, 0 means the work was
// queued; if the GPU later fails it, the next call reports it
// (ODStats.asyncFailures, od_first_async_error).
int od_render(ODSession*, const ODFrame* frames, const int* offsets,
                int frameCount, const ODParams*, const ODFrame* out,
                ODStats* statsOrNull);

// Estimate a noise profile from one frame without rendering.
int od_profile_frame(ODSession*, const ODFrame* frame, ODNoiseProfile* outProfile);
// API v13: parameter-aware Capture measurement. Reads only advanced and
// chromaRecovery; all other fields (including manualProfile) are ignored.
// NULL params selects the legacy estimator. Existing od_profile_frame and
// serialized/manual profiles retain their previous meaning.
int od_profile_frame_with_params(ODSession*, const ODFrame*, const ODParams*, ODNoiseProfile*);

const char* od_last_error(ODSession*);
const char* od_version(void);

// API v7. The first error of a render of this session that did not wait and
// failed on the GPU after od_render had returned 0 (ODStats.asyncFailures);
// "" if none. Kept until the session is destroyed: later renders, successful
// or not, and od_session_reset do not clear it.
const char* od_first_async_error(ODSession*);

// API v7. Waits at most timeoutMs until no asynchronous work of any session
// of the process is pending: the completion of renders that did not wait,
// the noise estimates queued after them, and GPU memory queued for release,
// including that of sessions and contexts already destroyed. For a host's
// global teardown, before its plug-in may be unloaded. Returns the number of
// items still pending (0 = idle).
int od_wait_idle(double timeoutMs);

#ifdef __cplusplus
}
#endif
#endif  // OPENDENOISE_H
