// SPDX-License-Identifier: MIT
// Adaptive quality entry point for callers that KNOW render intent. Existing ODParams/ODStats ABI and
// od_render semantics are unchanged. Rebuild clients to use this extension.
#ifndef OD_QUALITY_H
#define OD_QUALITY_H
#include "opendenoise.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int automatic;           // 0: respect params.quality; 1: govern certified realtime only
  double frameDurationMs;   // sequence interval, NOT source/media interval; 0 unknown
  double timelineTimeMs;    // sequence timestamp; NaN if unavailable
  double callerWaitMs;      // optional wrapper lock wait before entering core
} ODQualityRequest;

typedef struct {
  uint64_t renderId, runId;
  int quality;              // effective quality for THIS render (0 or 1)
  int switches;             // downgrades this run: at most one
  double budgetMs;          // min(20 ms, 60% of sequence interval)
  double emaMs;             // completed warm calls at the selected quality
} ODQualityDecision;

typedef struct {
  uint64_t renderId, runId;
  int quality, failed;
  double callToFinishMs;    // this call's entry (incl. callerWaitMs) -> GPU finish
  double gpuMs;             // last command buffer GPU duration
  double budgetMs;
} ODQualityCompletion;

// The caller MUST establish playback independently of request cadence, idle
// gaps or missing requests: cached/out-of-order exports can produce all three.
// Premiere's GPU filter ABI does not supply that guarantee. Do not set realtime
// from those heuristics. odbench --realtime explicitly supplies playback intent.
// Auto export (realtime == 0, or a blocking render) is always Full. Explicit
// quality is unchanged, including explicit Playback on export. No radius change.
int od_render_quality(ODSession*, const ODFrame*, const int* offsets, int frameCount,
                      const ODParams*, const ODFrame* out, ODStats*,
                      const ODQualityRequest*, ODQualityDecision*);
// Optional diagnostics: drain up to capacity completions exactly once. Stores
// at most 256; overflow is counted in *dropped (reset on read). Does not wait.
// The governor consumes EVERY completion even if diagnostics overflow.
int od_quality_poll(ODSession*, ODQualityCompletion*, int capacity, uint64_t* dropped);
#ifdef __cplusplus
}
#endif
#endif
