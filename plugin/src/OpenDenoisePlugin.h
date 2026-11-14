// SPDX-License-Identifier: MIT
// Open Denoise — definitions shared by the CPU entry (OpenDenoiseCPU.mm), the GPU
// filter (OpenDenoiseGPU.mm) and the PiPL (OpenDenoise.r, via the Makefile).
#ifndef OPENDENOISE_PLUGIN_H
#define OPENDENOISE_PLUGIN_H

// ---- Identity. Never change the match name: projects refer to it. ----------
#define OD_EFFECT_NAME      "Denoise"
#define OD_EFFECT_MATCHNAME "Open Denoise"
#define OD_EFFECT_CATEGORY  "Denoise"

#define OD_MAJOR_VERSION 0
#define OD_MINOR_VERSION 1
#define OD_BUG_VERSION   0
// PF_VERSION(0, 1, 0, PF_Stage_DEVELOP, 0); the Makefile passes the same
// number to Rez as OD_PIPL_VERSION. OpenDenoiseCPU.mm static_asserts they match.
#define OD_PIPL_VERSION_VALUE 0x00008000

// ---- Global out-flags (must equal the PiPL; see OpenDenoise.r) ---------------------
// out_flags:
//   PF_OutFlag_WIDE_TIME_INPUT  (1<<1)  we check out the input layer at other
//                                       times (temporal neighbours).
//   PF_OutFlag_DEEP_COLOR_AWARE (1<<25) we handle 16-bit worlds (After Effects).
//   PF_OutFlag_SEND_UPDATE_PARAMS_UI (1<<26) cosmetic capture status refresh.
// Deliberately NOT set:
//   NON_PARAM_VARY: the output only changes over time because the input does,
//     which WIDE_TIME_INPUT already tells the host. Setting it would stop the
//     host caching a still.
//   PIX_INDEPENDENT: the filter is spatial.
//   USE_OUTPUT_EXTENT / I_EXPAND_BUFFER / WORKS_IN_PLACE: we always process and
//     return the full frame at its own size.
// out_flags2:
//   PF_OutFlag2_PRESERVES_FULLY_OPAQUE_PIXELS (1<<8)  alpha is passed through.
//   PF_OutFlag2_SUPPORTS_THREADED_RENDERING   (1<<27) no writable sequence
//     data; all shared state is behind our own mutexes.
//   PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG (1<<3) each topic follows
//     its own PF_ParamFlag_START_COLLAPSED; without it every topic starts
//     collapsed (AE_Effect.h). All current topics start collapsed.
// Deliberately NOT set: SUPPORTS_SMART_RENDER / FLOAT_COLOR_AWARE (32-bit float
//   in After Effects needs SmartFX, not implemented; in Premiere 32f comes
//   through the pixel format suite and needs no flag), AUTOMATIC_WIDE_TIME_INPUT
//   (SmartFX only).
#define OD_OUT_FLAGS_VALUE  0x06000002
#define OD_OUT_FLAGS2_VALUE 0x08000108

// Parameter disk IDs are permanent: never renumber or reuse them. Saved values use these
// IDs; the panel order is kParamOrder and the PF index is PfIndex(id). Display names and
// panel positions are independent of saved identity.
enum {
  OD_P_INPUT = 0,
  OD_P_LUMA,          // 1 Luma strength, float slider 0..100 %, default 80
  OD_P_CHROMA,        // 2 Colour strength [Chroma strength], float slider 0..100 %, default 100
  OD_P_TEMPORAL_RADIUS,   // 3 Frames each side [Temporal radius], integer slider 0..5, default 2
  OD_P_TEMPORAL_STRENGTH, // 4 Temporal strength, float slider 0..100 %, default 100
  OD_P_SPATIAL,       // 5 Spatial strength, float slider 0..100 %, default 50
  OD_P_SHARPEN,       // 6 Sharpen, float slider 0..100 %, default 0
  OD_P_NOISE_LEVEL,   // 7 Noise level, float slider 25..400 %, default 100
  OD_P_QUALITY,       // 8 Quality, popup: 1 Auto, 2 Fast (also used when exporting) [Playback], 3 Full
  OD_P_VIEW,          // 9 View, popup: 1 Output, 2 Noise removed, 3 Motion, 4 Profile area
  // ---- Advanced controls (core API v6). ----
  // A topic's start and end entries are parameters too: they take a PF index
  // and a disk ID. Every slider below is a float slider in percent.
  OD_P_TOPIC_LUMA,             // 10 topic "Luma by size" [Luma detail]
  OD_P_LUMA_LEVEL_FINE,        // 11 Luma fine noise level [Luma fine level], 25..400 %, default 100
  OD_P_LUMA_LEVEL_MEDIUM,      // 12
  OD_P_LUMA_LEVEL_COARSE,      // 13
  OD_P_LUMA_LEVEL_VCOARSE,     // 14
  OD_P_LUMA_AMOUNT_FINE,       // 15 Luma fine strength [Luma fine amount], 0..100 %, default 100
  OD_P_LUMA_AMOUNT_MEDIUM,     // 16
  OD_P_LUMA_AMOUNT_COARSE,     // 17
  OD_P_LUMA_AMOUNT_VCOARSE,    // 18
  OD_P_GRAIN_FLOOR,            // 19 Minimum grain [Luma grain floor], 0..1000 %, default 100
  OD_P_REMOVE_SPECKS,          // 20 checkbox, default on, not animatable
  OD_P_TOPIC_LUMA_END,         // 21
  OD_P_TOPIC_CHROMA,           // 22 topic "Colour by size" [Chroma detail]
  OD_P_CHROMA_LEVEL_FINE,      // 23 Colour fine noise level [Chroma fine level], 25..400 %, default 100
  OD_P_CHROMA_LEVEL_MEDIUM,    // 24
  OD_P_CHROMA_LEVEL_COARSE,    // 25
  OD_P_CHROMA_LEVEL_VCOARSE,   // 26
  OD_P_CHROMA_AMOUNT_FINE,     // 27 Colour fine strength [Chroma fine amount], 0..100 %, default 100
  OD_P_CHROMA_AMOUNT_MEDIUM,   // 28
  OD_P_CHROMA_AMOUNT_COARSE,   // 29
  OD_P_CHROMA_AMOUNT_VCOARSE,  // 30
  OD_P_TOPIC_CHROMA_END,       // 31
  OD_P_TOPIC_TEMPORAL,         // 32 topic "Temporal"
  OD_P_MOTION_THRESHOLD,       // 33 50..200 %, default 100
  OD_P_EDGE_GATE,              // 34 edge ghost protection, 0..100 %, default 75
  OD_P_TOPIC_TEMPORAL_END,     // 35
  OD_P_TOPIC_SHARPEN,          // 36 topic "Sharpening" [Sharpen detail]
  OD_P_SHARPEN_FINE,           // 37 0..250 %, default 100
  OD_P_SHARPEN_MEDIUM,         // 38 0..250 %, default 100
  OD_P_SHARPEN_NOISE_FLOOR,    // 39 50..300 %, default 100
  OD_P_SHARPEN_HALO,           // 40 0..500 %, default 100
  OD_P_TOPIC_SHARPEN_END,      // 41
  OD_P_TOPIC_PROFILE,          // 42 topic "Noise profile"
  OD_P_HOLD_PROFILE,           // 43 Freeze (this session) [Hold profile], checkbox, default off
  OD_P_NOISE_Y,                // 44 luma noise level, 25..400 %, default 100
  OD_P_NOISE_CB,               // 45
  OD_P_NOISE_CR,               // 46
  OD_P_TOPIC_PROFILE_END,      // 47
  // ---- Experimental controls (core API v9). ----
  OD_P_TOPIC_EXPERIMENTAL,     // 48 topic "Experimental"
  OD_P_GRAIN_STABILITY,        // 49 float slider 0..100 %, default 0, animatable
  OD_P_COARSE_CHROMA,          // 50 Coarse colour cleanup [Coarse chroma cleanup], checkbox, default off
  OD_P_TOPIC_EXPERIMENTAL_END, // 51
  // ---- Saved noise profile: serialized payload and identity. ----
  OD_P_PROFILE_POLICY = 52,    // Profile policy, popup: 1 Automatic, 2 Use captured; supervised, not animatable
  OD_P_PROFILE_CAPTURE,        // 53 Capture profile, button "Capture / Save", supervised
  OD_P_PROFILE_REVISION,       // 54 capture revision, hidden integer slider
  OD_P_PROFILE_PAYLOAD,        // 55 captured profile, hidden arbitrary data (832-byte envelope)
  OD_P_PROFILE_DIGEST0,        // 56 first of eight hidden signed 32-bit words of the payload's SHA-256
  OD_P_PROFILE_DIGEST_LAST = OD_P_PROFILE_DIGEST0 + 7,  // 63
  OD_P_PROFILE_INFO,           // 64 Profile info, button "Refresh status", supervised, no saved data
  // ---- Effect Controls v2. ----
  OD_P_LOOK = 65,              // popup "Look", supervised, not animatable; the render never reads it
  OD_P_TOPIC_EXPERT = 66,      // topic "Expert"
  OD_P_TOPIC_EXPERT_END = 67,
  // ---- Grain mode: core API v10 ODParams.grainMode. ----
  OD_P_GRAIN_MODE = 68,        // popup "Minimum grain applies to": 1 Finest grain only (new effects),
                               // 2 Whole picture (previous) (projects saved before it existed);
                               // supervised (look set), not animatable
  OD_P_CAPTURE_REQUEST_LO = 69, // transient render-capture request, never a saved profile
  OD_P_CAPTURE_REQUEST_HI = 70,
  OD_P_CONTROL_SET = 71,
  OD_P_DB_LOOK = 72,
  OD_P_STATUS_NOISE = 73,
  OD_P_STATUS_HELP = 74,
  OD_P_DB_LUMA = 75,
  OD_P_DB_CHROMA = 76,
  OD_P_DB_GRAIN = 77,
  OD_P_DB_SPATIAL = 78,
  OD_P_DB_FRAMES = 79,
  OD_P_DB_PROFILE = 80,
  OD_P_DB_MOTION = 81,
  OD_P_DB_LUMA_LIMIT = 82,       // 82..85, fine to large
  OD_P_DB_CHROMA_LIMIT = 86,     // 86..89
  OD_P_DB_SHARPEN = 90,
  OD_P_DB_CHANNEL = 91,         // 91..93, Y / Cb / Cr
  OD_P_DB_LUMA_CORRECTION = 94,  // 94..97
  OD_P_DB_CHROMA_CORRECTION = 98,// 98..101
  OD_P_SWITCH_DB = 102,
  OD_P_ANALYSIS_KEY_LO = 103,
  OD_P_ANALYSIS_KEY_HI = 104,
  OD_P_PROFILE_PANEL = 105,     // retired UI row; permanently hidden, disk ID never reused
  OD_P_PROCESSING = 106,
  OD_P_TOPIC_PREVIOUS = 107,    // reserved even when the fallback never ships
  OD_P_TOPIC_PREVIOUS_END = 108,
};
enum {
  OD_ID_NEXT_FREE = 109,
};

#define OD_DFLT_LUMA 80
#define OD_DFLT_CHROMA 100
#define OD_DFLT_TEMPORAL_RADIUS 2
#define OD_DFLT_TEMPORAL_STRENGTH 100
#define OD_DFLT_SPATIAL 50
#define OD_DFLT_SHARPEN 0
#define OD_DFLT_NOISE_LEVEL 100
#define OD_NOISE_LEVEL_MIN 25
#define OD_NOISE_LEVEL_MAX 400
#define OD_DFLT_QUALITY 1  // Auto: the core's full quality when the host asks for a
                             // high-quality render (Premiere 27: playback and export alike),
                             // its playback quality otherwise Quality items, 1-based (the
                             // GPU side gets them 0-based). Never reordered or removed:
                             // projects store the index. Item 2 is the core's playback
                             // quality everywhere, exports included, hence its name.
#define OD_QUALITY_AUTO 1
#define OD_QUALITY_FAST 2
#define OD_QUALITY_FULL 3
#define OD_QUALITY_ITEMS "Auto|Fast (also used when exporting)|Full"
#define OD_DFLT_VIEW 1     // Output
#define OD_NUM_VIEWS 5
#define OD_VIEW_ITEMS "Result|Removed noise|Motion protection|Measured areas|Before / after split"
// Minimum grain applies to (disk ID 68, core ODParams.grainMode): where the grain floor
// (Minimum grain) applies. Finest grain only: the floor keeps the finest grain (about 1-2
// px) and Luma strength removes medium and coarse luma noise over its whole range, with the
// profile's measured-knot aggregate. Whole picture (previous): the picture of every build
// before it, bit for bit. A newly applied or reset effect takes Finest grain only
// (dephault); a project saved before the control existed takes Whole picture
// (PF_ParamFlag_USE_VALUE_FOR_OLD_PROJECTS with value = Whole picture), so it renders
// exactly as it did. Items are never reordered or removed.
#define OD_GRAIN_MODE_FINEST 1
#define OD_GRAIN_MODE_WHOLE 2
#define OD_NUM_GRAIN_MODES 2
#define OD_GRAIN_MODE_ITEMS "Finest grain only|Whole picture (previous)"
#define OD_DFLT_GRAIN_MODE OD_GRAIN_MODE_FINEST
#define OD_GRAIN_MODE_FOR_OLD_PROJECTS OD_GRAIN_MODE_WHOLE
// Look (disk ID 65): a menu that writes the look set's values when chosen
// (OpenDenoiseCPU.mm, UserChangedParam); the render never reads it. Items
// are never reordered or removed; a new Look is appended after Custom.
#define OD_LOOK_NATURAL 1
#define OD_LOOK_CLEANER 2
#define OD_LOOK_PHONE 3
#define OD_LOOK_PRESERVE 4
#define OD_LOOK_CUSTOM 5
#define OD_NUM_LOOKS 5
#define OD_LOOK_ITEMS "Natural|Cleaner|Phone|Preserve detail|Custom"
#define OD_DFLT_LOOK OD_LOOK_NATURAL         // a new or reset effect
#define OD_LOOK_FOR_OLD_PROJECTS OD_LOOK_CUSTOM  // a project saved before Look existed
                                                 // (PF_ParamFlag_USE_VALUE_FOR_OLD_PROJECTS)

// New units get new IDs. Previous IDs retain their saved meanings.
#define OD_CONTROLS_PREVIOUS 1
#define OD_CONTROLS_DB 2
#define OD_DB_LOOK_NATURAL 1
#define OD_DB_LOOK_CLEANER 2
#define OD_DB_LOOK_TEXTURE 3
#define OD_DB_LOOK_PHONE 4
#define OD_DB_LOOK_COLOUR 5
#define OD_DB_LOOK_CUSTOM 6
#define OD_DB_LOOK_ITEMS "Natural|Cleaner|Keep texture|Phone|Colour only|Custom"
#define OD_FRAMES_ITEMS "1 frame (off)|3 frames (1 each side)|5 frames (2 each side)|7 frames (3 each side)|9 frames (4 each side)|11 frames (5 each side)"

// Advanced controls: defaults are the neutral values (today's picture, bit
// for bit), and ranges, in percent of the UI.
#define OD_DFLT_BAND_LEVEL 100
#define OD_BAND_LEVEL_MIN 25
#define OD_BAND_LEVEL_MAX 400
#define OD_DFLT_BAND_AMOUNT 100
#define OD_DFLT_GRAIN_FLOOR 100
#define OD_GRAIN_FLOOR_MAX 1000
#define OD_DFLT_REMOVE_SPECKS 1
#define OD_DFLT_MOTION_THRESHOLD 100
#define OD_MOTION_THRESHOLD_MIN 50
#define OD_MOTION_THRESHOLD_MAX 200
#define OD_DFLT_EDGE_GATE 75
#define OD_DFLT_SHARPEN_BAND 100
#define OD_SHARPEN_BAND_MAX 250
#define OD_DFLT_SHARPEN_NOISE_FLOOR 100
#define OD_SHARPEN_NOISE_FLOOR_MIN 50
#define OD_SHARPEN_NOISE_FLOOR_MAX 300
#define OD_DFLT_SHARPEN_HALO 100
#define OD_SHARPEN_HALO_MAX 500
#define OD_DFLT_HOLD_PROFILE 0
#define OD_PROFILE_AUTOMATIC 1
#define OD_PROFILE_CAPTURED 2
#define OD_DFLT_CHANNEL_NOISE 100
// Experimental controls (core API v9): off at their defaults (today's picture).
#define OD_DFLT_GRAIN_STABILITY 0
#define OD_DFLT_COARSE_CHROMA 0

#if defined(__cplusplus) && !defined(OD_REZ)
#include <stdint.h>
#include <array>
#include <string>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>
#include "opendenoise/opendenoise.h"

namespace odplug {

// ---- Effect Controls order -------------------------------------------------
// The integration changes ONLY this constant if the host ignores dynamic hiding.
inline constexpr bool kUseDynamicVisibility = true;
// Fallback for hosts that show empty retired topics or hide adjacent controls:
// rebuild with OD_HIDE_UNUSED_TOPICS=0 for the previous flat topic layout.
#ifndef OD_HIDE_UNUSED_TOPICS
#define OD_HIDE_UNUSED_TOPICS 1
#endif
inline constexpr bool kUseMinimalPanel = kUseDynamicVisibility && OD_HIDE_UNUSED_TOPICS;
constexpr bool IsUnusedTopic(int id) {
  if (!kUseMinimalPanel) return false;
  for (int t : {10,21,22,31,32,35,36,41,42,47,48,51}) if (id == t) return true;
  return false;
}
inline constexpr int kPreviousOnly[] = {65, 1, 2, 19, 5, 3, 4, 7, 33, 15, 16, 17, 18, 27, 28, 29, 30, 39, 44, 45, 46, 11, 12, 13, 14, 23, 24, 25, 26};
constexpr bool IsPreviousOnly(int id) {
  for (int p : kPreviousOnly) if (p == id) return true;
  return false;
}
constexpr bool IsDBOnly(int id) { return id == 72 || (id >= 75 && id <= 101); }
// Each dB twin immediately follows its previous control. IDs, not positions,
// identify saved values. Capture storage stays a contiguous block.
inline constexpr std::array<int, 106> kInterleavedOrder = {
    65, 72, 102, 73, 74, 1, 75, 2, 76, 19, 77, 5,
    78, 3, 79, 9, 8, 42, 105, 52, 53, 54, 55, 56,
    57, 58, 59, 60, 61, 62, 63, 64, 43, 7, 80, 47,
    32, 4, 33, 81, 34, 35, 10, 15, 82, 16, 83, 17,
    84, 18, 85, 20, 21, 22, 27, 86, 28, 87, 29, 88,
    30, 89, 31, 36, 6, 37, 38, 39, 90, 40, 41, 66,
    68, 106, 44, 91, 45, 92, 46, 93, 11, 94, 12, 95,
    13, 96, 14, 97, 23, 98, 24, 99, 25, 100, 26, 101,
    67, 48, 49, 50, 51, 69, 70, 71, 103, 104,
};
constexpr std::array<int, 108> MakeFallbackOrder() {
  std::array<int, 108> order{};
  int n = 0;
  for (int id : kInterleavedOrder) if (!IsPreviousOnly(id)) order[n++] = id;
  order[n++] = OD_P_TOPIC_PREVIOUS;
  for (int id : kInterleavedOrder) if (IsPreviousOnly(id)) order[n++] = id;
  order[n++] = OD_P_TOPIC_PREVIOUS_END;
  return order;
}
inline constexpr auto kFallbackOrder = MakeFallbackOrder();
// Six main controls for dB; legacy twins occupy the same slots. Tuning is in
// one topic. Retired pairs are empty and outside Advanced: never nested.
inline constexpr std::array<int, 106> kMinimalOrder = {
    65,72,1,75,2,76,19,77,9,74,66,102,5,78,3,79,8,
    52,53,54,55,56,57,58,59,60,61,62,63,64,43,7,80,
    4,33,81,34,15,82,16,83,17,84,18,85,20,
    27,86,28,87,29,88,30,89,6,37,38,39,90,40,
    44,91,45,92,46,93,11,94,12,95,13,96,14,97,
    23,98,24,99,25,100,26,101,68,106,49,50,67,
    73,105,69,70,71,103,104,42,47,32,35,10,21,22,31,36,41,48,51,
};
// std::array sizes differ: choose in a constexpr function (not operator ?:).
template<bool dynamic> constexpr auto SelectedOrder() {
  if constexpr (dynamic) return kUseMinimalPanel ? kMinimalOrder : kInterleavedOrder;
  else return kFallbackOrder;
}
inline constexpr auto kParamOrder = SelectedOrder<kUseDynamicVisibility>();
inline constexpr int kNumOrdered = (int)kParamOrder.size();
inline constexpr int kNumParams = 1 + kNumOrdered;
inline constexpr int kTopicIds[] = {10,21,22,31,32,35,36,41,42,47,48,51,66,67,107,108};
inline constexpr int kNumTopicEntries = kUseDynamicVisibility ? 14 : 16;
constexpr bool IsTopicEntry(int id) {
  for (int t : kTopicIds) if (t == id) return true;
  return false;
}
constexpr int PfIndex(int id) {
  for (int i = 0; i < kNumOrdered; ++i) if (kParamOrder[i] == id) return i + 1;
  return -1;
}
constexpr int IdAtPf(int pf) { return pf >= 1 && pf <= kNumOrdered ? kParamOrder[pf - 1] : 0; }
constexpr int TopicEntriesBefore(int pf) {
  int n = 0;
  for (int p = 1; p < pf && p <= kNumOrdered; ++p) n += IsTopicEntry(IdAtPf(p)) ? 1 : 0;
  return n;
}
template<size_t N> constexpr bool ParamOrderIsValid(const std::array<int, N>& order) {
  bool seen[OD_ID_NEXT_FREE]{};
  int open = 0;
  for (int id : order) {
    if (id <= 0 || id >= OD_ID_NEXT_FREE || seen[id]) return false;
    seen[id] = true;
    for (int i = 0; i < 16; i += 2) {
      if (id == kTopicIds[i]) { if (open) return false; open = kTopicIds[i+1]; }
      if (id == kTopicIds[i+1]) { if (open != id) return false; open = 0; }
    }
  }
  for (int id = 1; id <= (int)N; ++id) if (!seen[id]) return false;
  return open == 0;
}
constexpr bool TwinOrderIsValid() {
  constexpr int twins[][2] = {{65,72},{1,75},{2,76},{19,77},{5,78},{3,79},{7,80},{33,81},{15,82},{16,83},{17,84},{18,85},{27,86},{28,87},{29,88},{30,89},{39,90},{44,91},{45,92},{46,93},{11,94},{12,95},{13,96},{14,97},{23,98},{24,99},{25,100},{26,101}};
  for (auto& pair : twins) {
    bool found = false;
    for (size_t i = 1; i < kInterleavedOrder.size(); ++i)
      found |= kInterleavedOrder[i-1] == pair[0] && kInterleavedOrder[i] == pair[1];
    if (!found) return false;
  }
  return true;
}
constexpr bool FallbackIsGrouped() {
  bool inside = false;
  for (int id : kFallbackOrder) {
    if (id == OD_P_TOPIC_PREVIOUS) inside = true;
    else if (id == OD_P_TOPIC_PREVIOUS_END) inside = false;
    else if (IsPreviousOnly(id) != inside) return false;
  }
  return kFallbackOrder.back() == OD_P_TOPIC_PREVIOUS_END;
}
constexpr bool CaptureBlockIsComplete() {
  for (int id = 52; id <= 64; ++id) if (PfIndex(id) != PfIndex(52) + id - 52) return false;
  return true;
}
static_assert(ParamOrderIsValid(kMinimalOrder) && ParamOrderIsValid(kInterleavedOrder) && ParamOrderIsValid(kFallbackOrder));
static_assert(TwinOrderIsValid() && FallbackIsGrouped() && CaptureBlockIsComplete());
static_assert(kNumParams == (kUseDynamicVisibility ? 107 : 109));

// Names are ASCII, <=31 bytes. Valid ranges can extend beyond drag ranges.
struct DBSlider { int id; const char* name; double lo, hi, dragLo, dragHi, dflt; };
inline constexpr DBSlider kDBSliders[] = {
    {75, "Luma reduction (dB)", 0,30,0,30,14},
    {76, "Colour reduction (dB)", 0,30,0,30,30},
    {77, "Keep grain (10-bit)", 0,6.57,0,3,0.657},
    {78, "Noise threshold (dB)", -18,6,-12,6,0},
    {80, "Profile correction (dB)", -12.05,12.05,-12,12,0},
    {81, "Motion threshold (dB)", -6.02,6.02,-6.02,6.02,0},
    {82, "Luma fine 1-2 px limit (dB)", 0,30,0,30,30},
    {83, "Luma medium 2-4 px limit (dB)", 0,30,0,30,30},
    {84, "Luma coarse 4-8 px limit (dB)", 0,30,0,30,30},
    {85, "Luma large 8+ px limit (dB)", 0,30,0,30,30},
    {86, "Colour fine 1-2 px limit (dB)", 0,30,0,30,30},
    {87, "Colour medium 2-4 px limit (dB)", 0,30,0,30,30},
    {88, "Colour coarse 4-8 px limit (dB)", 0,30,0,30,30},
    {89, "Colour large 8+ px limit (dB)", 0,30,0,30,30},
    {90, "Sharpen threshold (dB)", -6.02,9.54,-6.02,9.54,0},
    {91, "Luma correction (dB)", -12,12,-12,12,0},
    {92, "Cb correction (dB)", -12,12,-12,12,0},
    {93, "Cr correction (dB)", -12,12,-12,12,0},
    {94, "Luma fine correction (dB)", -12,12,-12,12,0},
    {95, "Luma medium correction (dB)", -12,12,-12,12,0},
    {96, "Luma coarse correction (dB)", -12,12,-12,12,0},
    {97, "Luma large correction (dB)", -12,12,-12,12,0},
    {98, "Colour fine correction (dB)", -12,12,-12,12,0},
    {99, "Colour medium correction (dB)", -12,12,-12,12,0},
    {100, "Colour coarse correction (dB)", -12,12,-12,12,0},
    {101, "Colour large correction (dB)", -12,12,-12,12,0},
};
constexpr const DBSlider* DBSliderFor(int id) {
  for (const auto& s : kDBSliders) if (s.id == id) return &s;
  return nullptr;
}
inline constexpr int kDBLookSet[] = {75,76,77,68,78,82,83,84,85,86,87,88,89,94,95,96,97,98,99,100,101};
inline constexpr int kDBLookSetSize = 21;
constexpr bool IsDBLookSetControl(int id) {
  for (int c : kDBLookSet) if (c == id) return true;
  return false;
}
double DBLookRecipeValue(int look, int id);
int MatchingDBLook(const double values[kDBLookSetSize]);

// ---- Look (disk ID 65) -----------------------------------------------------------
// The look set: the controls a Look writes when chosen, and whose manual
// change re-labels the Look (design, sections 3.3 and 4). Float sliders, and
// one popup, Minimum grain applies to (its 1-based item).
inline constexpr int kLookSet[] = {
    OD_P_LUMA,              OD_P_CHROMA,             OD_P_GRAIN_FLOOR,        OD_P_GRAIN_MODE,
    OD_P_LUMA_AMOUNT_FINE,  OD_P_LUMA_AMOUNT_MEDIUM, OD_P_LUMA_AMOUNT_COARSE,  OD_P_LUMA_AMOUNT_VCOARSE,
    OD_P_CHROMA_AMOUNT_FINE, OD_P_CHROMA_AMOUNT_MEDIUM, OD_P_CHROMA_AMOUNT_COARSE, OD_P_CHROMA_AMOUNT_VCOARSE,
    OD_P_LUMA_LEVEL_FINE,   OD_P_LUMA_LEVEL_MEDIUM,  OD_P_LUMA_LEVEL_COARSE,   OD_P_LUMA_LEVEL_VCOARSE,
    OD_P_CHROMA_LEVEL_FINE, OD_P_CHROMA_LEVEL_MEDIUM, OD_P_CHROMA_LEVEL_COARSE, OD_P_CHROMA_LEVEL_VCOARSE};
inline constexpr int kLookSetSize = (int)(sizeof kLookSet / sizeof kLookSet[0]);
constexpr bool IsLookSetControl(int id) {
  for (int c : kLookSet)
    if (c == id) return true;
  return false;
}
// A recipe value matches the control within this many percent (the UI shows
// tenths; a popup's item is an integer).
inline constexpr double kLookTolerance = 0.005;
// The value Look `look` (OD_LOOK_NATURAL .. OD_LOOK_PRESERVE) gives control
// `id` of the look set, in UI percent (a popup: its 1-based item); NaN for
// Custom, another Look or a control outside the look set.
double LookRecipeValue(int look, int id);
// The first Look, in menu order, whose recipe all of `values` match
// (values[i] is the current value of kLookSet[i]); OD_LOOK_CUSTOM if none.
int MatchingLook(const double values[kLookSetSize]);

// Raw UI values, as the host stores them.
struct UIParams {
  double luma = OD_DFLT_LUMA, chroma = OD_DFLT_CHROMA;
  double temporalRadius = OD_DFLT_TEMPORAL_RADIUS;
  double temporalStrength = OD_DFLT_TEMPORAL_STRENGTH;
  double spatial = OD_DFLT_SPATIAL, sharpen = OD_DFLT_SHARPEN;
  double noiseLevel = OD_DFLT_NOISE_LEVEL;
  int quality = OD_DFLT_QUALITY;  // 1-based popup index
  int view = OD_DFLT_VIEW;        // 1-based popup index
  // Advanced controls (percent; checkboxes 0 / 1). Index 0..3 = fine,
  // medium, coarse, very coarse.
  double lumaLevel[4] = {OD_DFLT_BAND_LEVEL, OD_DFLT_BAND_LEVEL, OD_DFLT_BAND_LEVEL, OD_DFLT_BAND_LEVEL};
  double lumaAmount[4] = {OD_DFLT_BAND_AMOUNT, OD_DFLT_BAND_AMOUNT, OD_DFLT_BAND_AMOUNT, OD_DFLT_BAND_AMOUNT};
  double grainFloor = OD_DFLT_GRAIN_FLOOR;
  double removeSpecks = OD_DFLT_REMOVE_SPECKS;
  double chromaLevel[4] = {OD_DFLT_BAND_LEVEL, OD_DFLT_BAND_LEVEL, OD_DFLT_BAND_LEVEL, OD_DFLT_BAND_LEVEL};
  double chromaAmount[4] = {OD_DFLT_BAND_AMOUNT, OD_DFLT_BAND_AMOUNT, OD_DFLT_BAND_AMOUNT, OD_DFLT_BAND_AMOUNT};
  double motionThreshold = OD_DFLT_MOTION_THRESHOLD;
  double edgeGate = OD_DFLT_EDGE_GATE;
  double sharpenFine = OD_DFLT_SHARPEN_BAND, sharpenMedium = OD_DFLT_SHARPEN_BAND;
  double sharpenNoiseFloor = OD_DFLT_SHARPEN_NOISE_FLOOR;
  double sharpenHalo = OD_DFLT_SHARPEN_HALO;
  double holdProfile = OD_DFLT_HOLD_PROFILE;
  double noiseY = OD_DFLT_CHANNEL_NOISE, noiseCb = OD_DFLT_CHANNEL_NOISE, noiseCr = OD_DFLT_CHANNEL_NOISE;
  // Experimental controls (percent; checkbox 0 / 1).
  double grainStability = OD_DFLT_GRAIN_STABILITY;
  double coarseChroma = OD_DFLT_COARSE_CHROMA;
  // Minimum grain applies to: 1-based popup item (OD_GRAIN_MODE_*).
  double grainMode = OD_DFLT_GRAIN_MODE;
  double controlSet = OD_CONTROLS_DB, processing = 2;
  double dbLuma = 14, dbChroma = 30, dbGrain = 0.657, dbSpatial = 0, dbFrames = 3;
  double dbProfile = 0, dbMotion = 0, dbSharpen = 0;
  double dbLumaLimit[4] = {30,30,30,30}, dbChromaLimit[4] = {30,30,30,30};
  double dbChannel[3] = {}, dbLumaCorrection[4] = {}, dbChromaCorrection[4] = {};
};

// Where each advanced or experimental control lives in UIParams, by disk ID
// (nullptr for the first nine controls, the topic entries, Look and IDs this
// build does not have). One table for both reads: the CPU path
// (params[PfIndex(id)]) and the GPU path (GetParam at its GPU index).
double* AdvancedField(UIParams& ui, int id);
// True for the checkboxes among the advanced and experimental controls.
bool IsAdvancedCheckbox(int id);
// True for the popups among them (Minimum grain applies to): UIParams holds
// the 1-based item, the CPU path reads u.pd.value, the GPU path an integer
// that Premiere hands over 0-based.
bool IsAdvancedPopup(int id);

int AdvancedPopupChoices(int id);
// Access to both sets for effective-set checks and transactional conversion.
double* RenderField(UIParams& ui, int id);
int EffectiveControlSet(const UIParams& ui);
// Empty reason on success. No writes to the caller on refusal; UI code must
// also prove absence of keyframes before committing this conversion.
bool ConvertToDB(const UIParams& before, UIParams& after, std::string& reason);
void ApplyProcessingVersion(ODParams& params, int version);
// Version 2 profiles a fixed window even with temporal merging off.
int DependencyRadius(const ODParams& params);

// UI values -> core parameters (clamped). hostWantsFull is what Quality =
// Auto resolves to: true when the host asked for a high-quality render.
// There is nothing to pass for a reduced playback resolution: the core works
// on the frame it is given and keeps an automatic profile per frame size.
// Passes frame keys that do not promise the pixels behind them
// (verifyFrameCache, unless KeysNameContent()): the core reuses no frame
// between renders and uses the keys for its automatic profile
// (OpenDenoiseGPU.mm, keyFor). Sets realtime when the host
// did not ask for a high-quality render and the call does not wait anyway
// (playback: the core does not wait for the GPU to run the host's queued
// work; it can still block obtaining a command buffer from a full queue, or
// on the session's lock, both measured in ODStats).
ODParams ToCoreParams(const UIParams& ui, bool hostWantsFull, bool waitUntilCompleted);
// The hook for a host that can tell when the pixels behind a time change: true once the
// frame keys carry a content generation the host guarantees (none in Premiere 26.x: false).
// The core then reuses ingested neighbours and their pyramids (verifyFrameCache 0), 0.1 to
// 0.9 ms per frame at radius 2.
bool KeysNameContent();
int ClampRadius(double uiRadius);

// Identity of one frame for the core (ODFrame.frameKey): its time in the
// sequence, the clip's position in the sequence (the track item's start, a
// constant of the instance), the field and the frame size mixed into 64 bits.
// Never OD_NO_KEY.
int64_t MakeFrameKey(int64_t sequenceTime, int64_t clipBase, uint32_t field, uint32_t w, uint32_t h);

// The track item an effect sits on, read from its clip node at creation: its track and its
// start in sequence time (ticks). Unknown for nodes without an owner clip (adjustment
// layers, as reported) or without the properties; `known` needs the start, `trackKnown` the
// track as well (the session identity needs both, the frame keys only the start). `timing`
// hashes what maps the item's frames to source frames (its media node's in-point, offset and
// clip ID, the clip's speed, reverse and time remapping): a slip or a speed change keeps the
// item's start, so without it the same frame keys came back for other pixels and the
// automatic profile kept the old footage's noise. 0 when none of it could be read.
// `duration` is EffectNode::EffectDuration, which splits the sessions of effects whose item
// is unknown. `speed` is the source frames per sequence frame (clip speed times the sequence
// frame duration over the media's), 0 = unknown.
struct TrackItem {
  bool known = false;
  bool trackKnown = false;
  int32_t trackID = 0;
  int64_t start = 0;
  uint64_t timing = 0;
  int64_t duration = 0;
  double clipSpeed = 0;           // ClipNode/MediaNode::ClipSpeed (1 = 100 %), 0 = unknown
  int64_t mediaTicksPerFrame = 0; // MediaNode::StreamFrameRate, 0 = unknown
};

// The constant part of every frame key of an instance: the item's start
// and its timing hash (0 without an item).
int64_t KeyBase(const TrackItem& item);

// Sequence frames between two requested neighbours: 1, or round(1 / r) when
// a sequence frame shows r < 1 source frames (slow motion, or a 25p clip in
// a 50p sequence: with frame sampling, neighbours one sequence frame apart
// were copies of each other and of the current frame). At most 8.
int NeighbourStep(const TrackItem& item, int64_t sequenceTicksPerFrame);

// The registry identity of an effect on a track item (AcquireSharedSession): its timeline,
// Premiere's runtime instance ID, and the track and start of the track item. The runtime ID
// alone is not enough: copies of one effect pasted onto several clips of a sequence share it
// (observed in Premiere 26.5.2: 4 IDs for the 52 instances of 13 sequences), and one session
// for all of them mixed their noise profiles. 0 = a private session, destroyed with its
// instance: no runtime ID, or an owner that is not fully known (no owner clip, as reported
// for adjustment layers; no track or start). A session shared on a partial identity mixed
// the profiles of copies that share an ID, so ownership must be complete to share.
uint64_t EffectIdentity(int32_t timelineID, uint32_t runtimeID, const TrackItem& item);

// The software path (OpenDenoiseCPU.mm) in Premiere: what PF_UtilitySuite
// tells about the clip an effect instance renders, read on every render.
// `known` only when every field was read. There is no track ID on this path.
struct ClipOwner {
  bool known = false;
  int64_t timelineID = 0;
  int64_t trackItemStart = 0, clipStart = 0, clipDuration = 0;
  double clipSpeed = 0;
};
// The session key of a software render: Premiere's filter instance ID with
// the clip it renders. The instance ID alone is shared by pasted copies (as
// on the GPU side). 0 = no complete identity: the render starts from a reset
// session (stateless), never from another clip's.
uint64_t ClipIdentity(bool haveInstanceID, long instanceID, const ClipOwner& owner);

// A time range, in the unit of the times it is compared with. Unknown: no
// bounds.
struct TimeBounds {
  bool known = false;
  int64_t start = 0, end = 0;  // [start, end)
};

// Temporal offsets to request around a frame, nearest first: -1,+1,-2,+2...
// `time` and `ticksPerFrame` in the unit of `bounds`; offset k is requested
// at time + k * step * ticksPerFrame (NeighbourStep). With known bounds an
// offset whose time falls outside them is dropped,
// and so is one before `earliest` whatever the bounds (the GPU path passes 0:
// sequence time starts at 0, and Premiere hands over nothing for a time
// before it, which left the frame count short and the render spatial only).
std::vector<int> NeighbourOffsets(int radius, int64_t time, int64_t ticksPerFrame,
                                  const TimeBounds& bounds, int64_t earliest = INT64_MIN, int step = 1);

// One ODContext per (MTLDevice, MTLCommandQueue) pair, created on first use
// and kept until ReleaseAllContexts(). device/queue are unretained ids.
ODContext* ContextFor(void* mtlDevice, void* mtlQueueOrNull, char* err, size_t errLen);
void ReleaseAllContexts();  // only destroys contexts that have no live sessions

// An ODSession plus the lock that serialises it.
class Session {
 public:
  static Session* Create(ODContext* ctx);
  ~Session();
  // Locks, renders, unlocks. Returns od_render's result; on failure copies
  // od_last_error into err. lockWaitMs (may be NULL): ms spent waiting for
  // this lock (another render of the shared session in progress) plus the
  // core's own (ODStats.lockWaitMs). asyncError (may be NULL, size errLen):
  // od_first_async_error, copied under the lock.
  // Only dormant bridge tests opt in. Production compiles out render-record
  // copies and the extra stats request entirely.
  template<bool recordAnalysis = false>
  int Render(const ODFrame* frames, const int* offsets, int count, const ODParams* p,
             const ODFrame* out, ODStats* stats, char* err, size_t errLen, double* lockWaitMs = nullptr,
             char* asyncError = nullptr, uintptr_t analysisSource = 0);
  int Profile(const ODFrame&, ODNoiseProfile&, char* error, size_t errorSize, const ODParams* = nullptr);
  bool TryAnalysis(ODAnalysis&, ODParams&, uintptr_t source);
  void Reset();
 private:
  Session() {}
  ODContext* ctx_ = nullptr;
  ODSession* session_ = nullptr;
  std::mutex mutex_;
  struct AnalysisRecord { uint64_t id = 0; uintptr_t source = 0; ODParams params{}; };
  std::array<AnalysisRecord,16> analysisRecords_{};
};

// ---- Sessions shared by successive GPU instances ----------------------------
// Premiere replaces a GPU filter instance whenever a parameter changes
// (PrSDKGPUFilter.h). The session (automatic noise profile, texture pool)
// belongs to the effect on its track item, not to one set of parameters, so
// instances find it here by an identity that outlives them (see
// OpenDenoiseGPU.mm). Reference-counted by the instances; after the last one
// is released the session stays parked for kParkMs (OD_PARK_MS overrides)
// so that the replacement instance picks it up, then it is destroyed. At most
// kMaxParked sessions are parked at once. identity 0 = a private session,
// destroyed with its instance.
enum { kParkMsDefault = 5000, kMaxParked = 4 };
std::shared_ptr<Session> AcquireSharedSession(ODContext* ctx, uint64_t identity);
void ReleaseSharedSession(ODContext* ctx, uint64_t identity);
// Plug-in shutdown: destroys every parked session and stops the expiry timer.
// Sessions still held by an instance are left alone.
void ShutdownSharedSessions();

// ---- Module lifetime ----------------------------------------------------------
// The host may unload this bundle after its global teardown (xGPUFilterEntry
// with inStartup = 0, PF_Cmd_GLOBAL_SETDOWN). Work the core queued (Metal
// completion handlers, noise estimates, texture releases) runs code of this
// image; retaining its data does not keep the code mapped. The SDK guide
// (3.3.x, "Mac 64-Bit and Cocoa") says a bundle that uses Cocoa must not be
// unloaded and should retain itself. So:
// PinImage: at the first startup, the image takes a reference to itself that
//   it never gives back (dlopen of its own path): the host's dlclose then
//   leaves it mapped.
// GlobalTeardown: at each global teardown, destroys parked sessions, waits a
//   bounded time (kTeardownWaitMsDefault; OD_TEARDOWN_MS overrides) for the
//   asynchronous work of every session (od_wait_idle), then releases the
//   contexts no session holds. Logged. Never on an instance's disposal.
// gpuSessions: also destroy the parked sessions of the GPU registry (the GPU
// entry's shutdown; the CPU side's setdown leaves them to it).
enum { kTeardownWaitMsDefault = 2000 };
void PinImage();
void GlobalTeardown(const char* who, bool gpuSessions);

void Log(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
// Logs at most once per `flag` (a static std::atomic<bool> at the call site).
#define OD_LOG_ONCE(flag, ...)                     \
  do {                                             \
    if (!(flag).exchange(true)) ::odplug::Log(__VA_ARGS__); \
  } while (0)
// For an event that can repeat on every frame (a render that fails): counts
// it in `counter` (a static std::atomic<uint64_t> at the call site) and logs
// the first 16 occurrences, then the 32nd, 64th, 128th ... with their number,
// so that a run shows that it kept happening and how often without one line
// per frame. (A failing render used to be logged once per process, or on
// every frame.) `fmt` must be a string literal.
inline uint64_t LogOccurrence(std::atomic<uint64_t>& counter) {
  const uint64_t k = counter.fetch_add(1) + 1;
  return k <= 16 || (k & (k - 1)) == 0 ? k : 0;
}
#define OD_LOG_COUNTED(counter, fmt, ...)                                                                    \
  do {                                                                                                       \
    if (const uint64_t od_occurrence_ = ::odplug::LogOccurrence(counter))                                    \
      ::odplug::Log(fmt " (occurrence %llu)", ##__VA_ARGS__, (unsigned long long)od_occurrence_);           \
  } while (0)

}  // namespace odplug
#endif  // __cplusplus

#endif  // OPENDENOISE_PLUGIN_H
