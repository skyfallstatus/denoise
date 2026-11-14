// SPDX-License-Identifier: MIT
// Internal debug entry points for tests and odbench. Not part of the public
// API; may change without notice.
#ifndef OD_DEBUG_H
#define OD_DEBUG_H

#include "opendenoise/opendenoise.h"

#ifdef __cplusplus
extern "C" {
#endif

// Compiles every embedded shader unit and kernel. 0 on success; otherwise the
// message (naming the shader file) is written to err.
int od_debug_compile_all(ODContext*, char* err, size_t errLen);

// Ingests `frame` and reads the working texture back: outYCCA receives
// width*height*4 floats (Y, Cb, Cr, A), rows tightly packed. Blocks.
int od_debug_read_working(ODSession*, const ODFrame* frame, float* outYCCA);

// The session's frame cache, and the idle texture pool of its context (shared
// by all the context's sessions): textures idle in it, their bytes, and the
// textures the context has created so far. Any pointer may be NULL.
void od_debug_cache_info(ODSession*, int* cacheEntries, uint64_t* cacheBytes, int* poolTextures,
                           uint64_t* poolBytes, uint64_t* texturesCreated);

// GPU memory of a context (core/src/memory.h): bytes of every texture it holds
// (scratch in use, idle pool, all sessions' caches), its budget (0 = no
// limit), and the idle pool. Any pointer may be NULL.
void od_debug_memory_info(ODContext*, uint64_t* allocatedBytes, uint64_t* budgetBytes, int* poolTextures,
                            uint64_t* poolBytes);

// Textures a context lets go of are freed on a queue of its own. Waits until
// every release requested so far has happened (for tests that read
// MTLDevice.currentAllocatedSize).
void od_debug_memory_drain(ODContext*);

// Waits until every asynchronous render of the session has completed and its
// noise estimate, if it took one, has been folded into the automatic profile.
// 0 when done, 1 after a 10 s timeout.
int od_debug_session_drain(ODSession*);

// Completions of the session's renders that no od_render call has handed
// over yet (ODStats.completed), oldest first, at most `max`: taken, so a
// later render does not report them again. For a caller that renders no
// more (odbench's hazard count at the end of a run, after
// od_debug_session_drain). Returns the number written.
int od_debug_take_completions(ODSession*, ODCompletion* out, int max);

// Renders of the session that blocked for their own noise estimate (the
// first render of a frame size, or new footage: a cut or a jump).
uint64_t od_debug_profile_waits(ODSession*);

// Bytes the session's last render held: its scratch and the cache entries it
// used (what the device's admission of renders estimates a render from).
uint64_t od_debug_session_held(ODSession*);

// GPU memory of the device a context is on, over every context made on it
// with the same budget setting (core/src/memory.h, DeviceLedger).
typedef struct {
  uint64_t budgetBytes;        // the device's budget, 0 = none
  uint64_t logicalBytes;       // what the core holds there now: pools, caches, samples, scratch
  uint64_t logicalPeakBytes;   // its largest value
  uint64_t physicalBytes;      // MTLDevice.currentAllocatedSize now: the whole process on that device,
                               // the host's buffers and resources still referenced by work in flight included
  uint64_t physicalPeakBytes;  // its largest value seen at a creation or at the end of a render
  int contexts;                // contexts sharing the budget
  int rendersActive;           // renders in progress (admitted, not ended)
  int rendersActiveMax;
  uint64_t admissionWaits;     // renders that waited for room at their start
  uint64_t admissionsForced;   // of those, admitted at their deadline without room
  double admissionWaitMs;      // time spent waiting, in all
  uint64_t overruns;           // creations that left the logical total beyond the budget
  uint64_t overrunPeakBytes;   // the most bytes beyond it
} ODDeviceMemory;
void od_debug_device_memory(ODContext*, ODDeviceMemory* out);
// Peaks and counters start again from the present values.
void od_debug_device_memory_reset_peaks(ODContext*);

#ifdef __cplusplus
}
#endif
#endif
