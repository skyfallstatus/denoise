// SPDX-License-Identifier: MIT
// Opt-in profiling API. ODParams and the advanced-controls ABI are unchanged.
#ifndef OD_PROFILING_H
#define OD_PROFILING_H
#include "opendenoise.h"
#ifdef __cplusplus
extern "C" {
#endif
// Like od_render, but always measure and resolve the current frame plus the
// nearest +/-1 neighbour in the active temporal window before filtering. No history, cross-size
// fallback, throttle or unverified cache. Can wait even with realtime=1.
// manualProfile skips measurement. Empty estimates pass through as usual.
// Identity is supplied pixels, offsets, dimensions and settings, NOT keys.
int od_render_deterministic(ODSession*, const ODFrame*, const int* offsets,
                            int frameCount, const ODParams*, const ODFrame* out,
                            ODStats* statsOrNull);
// Aggregate 1..32 valid profiles of a FIXED sample set for Capture. Componentwise
// median in variance; even counts average the middle variances. Order invariant.
// Invalid input clears out and fails. Confidence is the median input confidence,
// not an error bar. Caller selects representative frames, preventing mixed shots.
int od_profile_aggregate(const ODNoiseProfile*, int count, ODNoiseProfile* out);
// Numerical payload for an arbitrary parameter / host blob: v1, little endian,
// fixed length, CRC32, no pointers/struct padding. Provenance belongs in a host
// envelope. Reject unknown version, length, CRC and nonfinite/negative values.
size_t od_profile_blob_size(void);
int od_profile_encode(const ODNoiseProfile*, void* bytes, size_t size);
int od_profile_decode(const void* bytes, size_t size, ODNoiseProfile* out);
#ifdef __cplusplus
}
#endif
#endif
