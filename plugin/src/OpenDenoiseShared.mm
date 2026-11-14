// SPDX-License-Identifier: MIT
// Open Denoise — state and helpers shared by the CPU and GPU entry points:
// parameter mapping, neighbour list, context registry, locked session, and
// the registry that shares a session between successive GPU instances.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include <dlfcn.h>
#include <os/log.h>
#include <stdarg.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <utility>

#include "OpenDenoisePlugin.h"
#include "opendenoise/profiling.h"

namespace odplug {

void Log(const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  static os_log_t log = os_log_create("dev.opendenoise.plugin", "plugin");
  os_log_error(log, "%{public}s", buf);
  if (getenv("OD_LOG_STDERR")) fprintf(stderr, "[Denoise] %s\n", buf);
}

static inline float Clamp01(double v) {
  if (!(v == v)) return 0.f;  // NaN
  return (float)std::min(1.0, std::max(0.0, v));
}

int ClampRadius(double r) {
  if (!(r == r)) return 0;
  int v = (int)llround(r);
  return std::min(OD_MAX_RADIUS, std::max(0, v));
}

static uint64_t Mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

uint64_t EffectIdentity(int32_t timelineID, uint32_t runtimeID, const TrackItem& item) {
  if (runtimeID == 0 || !item.known || !item.trackKnown) return 0;
  uint64_t k = Mix64(((uint64_t)(uint32_t)timelineID << 32) | runtimeID);
  k = Mix64(k ^ (uint64_t)(uint32_t)item.trackID);
  k = Mix64(k ^ (uint64_t)item.start);
  return k ? k : 1;
}

uint64_t ClipIdentity(bool haveInstanceID, long instanceID, const ClipOwner& o) {
  if (!haveInstanceID || !o.known) return 0;
  uint64_t k = Mix64((uint64_t)instanceID ^ 0x43505521ull);
  k = Mix64(k ^ (uint64_t)o.timelineID);
  k = Mix64(k ^ (uint64_t)o.trackItemStart);
  k = Mix64(k ^ (uint64_t)o.clipStart);
  k = Mix64(k ^ (uint64_t)o.clipDuration);
  uint64_t speedBits = 0;
  memcpy(&speedBits, &o.clipSpeed, sizeof speedBits);
  k = Mix64(k ^ speedBits);
  return k ? k : 1;
}

int64_t KeyBase(const TrackItem& item) {
  if (!item.known) return 0;
  if (item.timing == 0) return item.start;
  return (int64_t)Mix64((uint64_t)item.start ^ Mix64(item.timing));
}

int NeighbourStep(const TrackItem& item, int64_t seqTpf) {
  const double speed = std::fabs(item.clipSpeed);
  if (!(speed > 0) || item.mediaTicksPerFrame <= 0 || seqTpf <= 0) return 1;
  // Plausible frame rates only (1 to 1000 fps), else the properties are not
  // what we think they are.
  const double tps = 254016000000.0;
  if (item.mediaTicksPerFrame < (int64_t)(tps / 1000) || item.mediaTicksPerFrame > (int64_t)tps) return 1;
  const double r = speed * (double)seqTpf / (double)item.mediaTicksPerFrame;  // source frames per sequence frame
  if (!(r > 0) || r > 0.75) return 1;
  return (int)std::min(8.0, std::max(1.0, std::round(1.0 / r)));
}

int64_t MakeFrameKey(int64_t time, int64_t base, uint32_t field, uint32_t w, uint32_t h) {
  auto mix = Mix64;
  uint64_t k = mix((uint64_t)time);
  k = mix(k ^ (uint64_t)base);
  k = mix(k ^ (((uint64_t)w << 32) | h));
  k = mix(k ^ field);
  const int64_t key = (int64_t)k;
  return key == OD_NO_KEY ? 0 : key;
}

double* AdvancedField(UIParams& ui, int id) {
  if (id >= 82 && id <= 85) return &ui.dbLumaLimit[id-82];
  if (id >= 86 && id <= 89) return &ui.dbChromaLimit[id-86];
  if (id >= 91 && id <= 93) return &ui.dbChannel[id-91];
  if (id >= 94 && id <= 97) return &ui.dbLumaCorrection[id-94];
  if (id >= 98 && id <= 101) return &ui.dbChromaCorrection[id-98];
  switch (id) {
    case OD_P_CONTROL_SET: return &ui.controlSet;
    case OD_P_PROCESSING: return &ui.processing;
    case OD_P_DB_LUMA: return &ui.dbLuma;
    case OD_P_DB_CHROMA: return &ui.dbChroma;
    case OD_P_DB_GRAIN: return &ui.dbGrain;
    case OD_P_DB_SPATIAL: return &ui.dbSpatial;
    case OD_P_DB_FRAMES: return &ui.dbFrames;
    case OD_P_DB_PROFILE: return &ui.dbProfile;
    case OD_P_DB_MOTION: return &ui.dbMotion;
    case OD_P_DB_SHARPEN: return &ui.dbSharpen;
    case OD_P_LUMA_LEVEL_FINE: case OD_P_LUMA_LEVEL_MEDIUM: case OD_P_LUMA_LEVEL_COARSE: case OD_P_LUMA_LEVEL_VCOARSE:
      return &ui.lumaLevel[id - OD_P_LUMA_LEVEL_FINE];
    case OD_P_LUMA_AMOUNT_FINE: case OD_P_LUMA_AMOUNT_MEDIUM: case OD_P_LUMA_AMOUNT_COARSE: case OD_P_LUMA_AMOUNT_VCOARSE:
      return &ui.lumaAmount[id - OD_P_LUMA_AMOUNT_FINE];
    case OD_P_GRAIN_FLOOR: return &ui.grainFloor;
    case OD_P_REMOVE_SPECKS: return &ui.removeSpecks;
    case OD_P_CHROMA_LEVEL_FINE: case OD_P_CHROMA_LEVEL_MEDIUM: case OD_P_CHROMA_LEVEL_COARSE:
    case OD_P_CHROMA_LEVEL_VCOARSE:
      return &ui.chromaLevel[id - OD_P_CHROMA_LEVEL_FINE];
    case OD_P_CHROMA_AMOUNT_FINE: case OD_P_CHROMA_AMOUNT_MEDIUM: case OD_P_CHROMA_AMOUNT_COARSE:
    case OD_P_CHROMA_AMOUNT_VCOARSE:
      return &ui.chromaAmount[id - OD_P_CHROMA_AMOUNT_FINE];
    case OD_P_MOTION_THRESHOLD: return &ui.motionThreshold;
    case OD_P_EDGE_GATE: return &ui.edgeGate;
    case OD_P_SHARPEN_FINE: return &ui.sharpenFine;
    case OD_P_SHARPEN_MEDIUM: return &ui.sharpenMedium;
    case OD_P_SHARPEN_NOISE_FLOOR: return &ui.sharpenNoiseFloor;
    case OD_P_SHARPEN_HALO: return &ui.sharpenHalo;
    case OD_P_HOLD_PROFILE: return &ui.holdProfile;
    case OD_P_NOISE_Y: return &ui.noiseY;
    case OD_P_NOISE_CB: return &ui.noiseCb;
    case OD_P_NOISE_CR: return &ui.noiseCr;
    case OD_P_GRAIN_STABILITY: return &ui.grainStability;
    case OD_P_COARSE_CHROMA: return &ui.coarseChroma;
    case OD_P_GRAIN_MODE: return &ui.grainMode;
    default: return nullptr;
  }
}

bool IsAdvancedCheckbox(int id) {
  return id == OD_P_REMOVE_SPECKS || id == OD_P_HOLD_PROFILE || id == OD_P_COARSE_CHROMA;
}

int AdvancedPopupChoices(int id) {
  if (id == OD_P_DB_FRAMES) return 6;
  if (id == OD_P_GRAIN_MODE || id == OD_P_CONTROL_SET || id == OD_P_PROCESSING) return 2;
  return 0;
}
bool IsAdvancedPopup(int id) { return AdvancedPopupChoices(id) != 0; }

double* RenderField(UIParams& ui, int id) {
  switch (id) {
    case OD_P_LUMA: return &ui.luma;
    case OD_P_CHROMA: return &ui.chroma;
    case OD_P_TEMPORAL_RADIUS: return &ui.temporalRadius;
    case OD_P_TEMPORAL_STRENGTH: return &ui.temporalStrength;
    case OD_P_SPATIAL: return &ui.spatial;
    case OD_P_NOISE_LEVEL: return &ui.noiseLevel;
    default: return AdvancedField(ui, id);
  }
}

int EffectiveControlSet(const UIParams& values) {
  if (values.controlSet == OD_CONTROLS_PREVIOUS) return OD_CONTROLS_PREVIOUS;
  UIParams ui = values, defaults;
  for (int id = 75; id <= 101; ++id)
    if (*RenderField(ui,id) != *RenderField(defaults,id) &&
        *RenderField(ui,id) != double(float(*RenderField(defaults,id)))) return OD_CONTROLS_DB;
  for (int id : kPreviousOnly) {
    const double* v = RenderField(ui, id); // Look is not a render control
    if (v && *v != *RenderField(defaults, id)) return OD_CONTROLS_PREVIOUS;
  }
  return OD_CONTROLS_DB;
}

double DBLookRecipeValue(int look, int id) {
  if (look < 1 || look >= OD_DB_LOOK_CUSTOM || !IsDBLookSetControl(id)) return std::nan("");
  UIParams d;
  double v = *RenderField(d, id);
  if ((look == OD_DB_LOOK_CLEANER || look == OD_DB_LOOK_PHONE) && id == 77) v = 0;
  if (look == OD_DB_LOOK_TEXTURE && id == 75) v = 10.5;
  if (look == OD_DB_LOOK_PHONE && (id == 95 || id == 96)) v = 3.5;
  if (look == OD_DB_LOOK_COLOUR && id == 75) v = 0;
  return v;
}
int MatchingDBLook(const double values[kDBLookSetSize]) {
  for (int look = 1; look < OD_DB_LOOK_CUSTOM; ++look) {
    bool match = true;
    for (int i = 0; i < kDBLookSetSize; ++i)
      match &= std::isfinite(values[i]) && std::fabs(values[i] - DBLookRecipeValue(look,kDBLookSet[i])) <= 1e-6;
    if (match) return look;
  }
  return OD_DB_LOOK_CUSTOM;
}

static double Bounded(double v, double lo, double hi, double dflt = 0) {
  return std::isfinite(v) ? std::clamp(v, lo, hi) : dflt;
}
static double Reduction(double db) {
  db = Bounded(db, 0, 30);
  return db >= 30 ? 1.0 : 1.0 - std::pow(0.2, db / 14.0);
}
static double Gain(double db, double lo, double hi) {
  return std::pow(10.0, Bounded(db, lo, hi) / 20.0);
}
static double InverseReduction(double a) {
  return a == 1 ? 30 : 14 * std::log1p(-a) / std::log(0.2);
}

bool ConvertToDB(const UIParams& before, UIParams& after, std::string& reason) {
  reason.clear();
  if (before.temporalStrength != 100) { reason = "Temporal strength must be 100%"; return false; }
  if (before.spatial < 6.3) { reason = "Spatial strength below 6.3%"; return false; }
  UIParams v = before, defaults;
  v.dbLuma = InverseReduction(before.luma/100);
  v.dbChroma = InverseReduction(before.chroma/100);
  v.dbGrain = before.grainFloor * 0.657 / 100;
  v.dbSpatial = 20 * std::log10(2 * before.spatial/100);
  v.dbFrames = before.temporalRadius + 1;
  v.dbProfile = 20 * std::log10(before.noiseLevel/100);
  v.dbMotion = 20 * std::log10(before.motionThreshold/100);
  v.dbSharpen = 20 * std::log10(before.sharpenNoiseFloor/100);
  for (int b = 0; b < 4; ++b) {
    v.dbLumaLimit[b] = before.lumaAmount[b] == 100 || before.luma == 0 ? 30 :
                         InverseReduction(before.luma/100 * before.lumaAmount[b]/100);
    v.dbChromaLimit[b] = before.chromaAmount[b] == 100 || before.chroma == 0 ? 30 :
                           InverseReduction(before.chroma/100 * before.chromaAmount[b]/100);
    v.dbLumaCorrection[b] = 20 * std::log10(before.lumaLevel[b]/100);
    v.dbChromaCorrection[b] = 20 * std::log10(before.chromaLevel[b]/100);
  }
  v.dbChannel[0] = 20 * std::log10(before.noiseY/100);
  v.dbChannel[1] = 20 * std::log10(before.noiseCb/100);
  v.dbChannel[2] = 20 * std::log10(before.noiseCr/100);
  // The 30 dB endpoint and the narrower dB ranges leave real gaps. Refuse
  // instead of clamping: switching is not permission to change the picture.
  for (const auto& spec : kDBSliders) {
    const double value = *RenderField(v, spec.id);
    if (!std::isfinite(value) || value < spec.lo || value > spec.hi) {
      reason = "Value outside dB range: see Help";
      return false;
    }
  }
  if (!std::isfinite(v.dbFrames) || v.dbFrames < 1 || v.dbFrames > 6 || std::floor(v.dbFrames) != v.dbFrames) {
    reason = "Frames value cannot convert"; return false;
  }
  for (int id : kPreviousOnly) {
    double* field = RenderField(v,id);
    if (field) *field = *RenderField(defaults,id);
  }
  v.controlSet = OD_CONTROLS_DB;
  after = v; // Processing and all shared controls are preserved.
  return true;
}

void ApplyProcessingVersion(ODParams& p, int version) {
  p.chromaRecovery = version == 2 ? 1u : 0u;
  // keep all version-gated fields in this one place. ToCoreParams already
  // enables advanced controls for both versions; the core defaults remain neutral so saved
  // Version 1 effects retain their exact arithmetic.
  p.temporalNoiseModel = version == 2 ? 1u : 0u;
}

int DependencyRadius(const ODParams& p) {
  if (p.advanced && p.temporalNoiseModel == 1)
    return std::max(p.temporalRadius, OD_PROFILE_RADIUS);
  return p.temporalStrength > 0 ? p.temporalRadius : 0;
}

// ---- Look recipes ------------------------- Natural is every look-set control at its
// default: the picture of a newly applied effect. The other Looks change a few controls and
// leave the rest at Natural's values, so switching between Looks is deterministic. A project
// saved before Look existed holds Whole picture (previous), which no Look sets: it reads
// Custom, whatever its sliders say.
double LookRecipeValue(int look, int id) {
  if (look < OD_LOOK_NATURAL || look > OD_LOOK_PRESERVE || !IsLookSetControl(id)) return std::nan("");
  double v;  // Natural
  switch (id) {
    case OD_P_LUMA: v = OD_DFLT_LUMA; break;
    case OD_P_CHROMA: v = OD_DFLT_CHROMA; break;
    case OD_P_GRAIN_FLOOR: v = OD_DFLT_GRAIN_FLOOR; break;
    case OD_P_GRAIN_MODE: v = OD_DFLT_GRAIN_MODE; break;
    case OD_P_LUMA_AMOUNT_FINE: case OD_P_LUMA_AMOUNT_MEDIUM: case OD_P_LUMA_AMOUNT_COARSE:
    case OD_P_LUMA_AMOUNT_VCOARSE: case OD_P_CHROMA_AMOUNT_FINE: case OD_P_CHROMA_AMOUNT_MEDIUM:
    case OD_P_CHROMA_AMOUNT_COARSE: case OD_P_CHROMA_AMOUNT_VCOARSE:
      v = OD_DFLT_BAND_AMOUNT;
      break;
    default: v = OD_DFLT_BAND_LEVEL; break;  // the noise levels by size
  }
  switch (look) {
    case OD_LOOK_CLEANER:
      // No grain floor: Luma strength (80 %) acts at every brightness, which
      // is what cleans mid-tones and skin on real log footage; Luma itself
      // stays at 80 % (higher is what makes skin waxy on heavy noise).
      if (id == OD_P_GRAIN_FLOOR) v = 0;
      break;
    case OD_LOOK_PHONE:
      // As Cleaner, and the correlated 2-8 px luma mottle of phone footage
      // counts as noise: the medium and coarse luma noise levels at 150 %.
      if (id == OD_P_GRAIN_FLOOR) v = 0;
      if (id == OD_P_LUMA_LEVEL_MEDIUM || id == OD_P_LUMA_LEVEL_COARSE) v = 150;
      break;
    case OD_LOOK_PRESERVE:
      // Less luma noise taken out (70 % instead of 80 %): more fine texture and grain kept,
      // at a flicker cost. On the synthetic set it kept as much detail as cutting the finest
      // band alone (Luma fine strength 80 %), with less flicker and less noise in flat
      // areas.
      if (id == OD_P_LUMA) v = 70;
      break;
    default: break;
  }
  return v;
}

int MatchingLook(const double values[kLookSetSize]) {
  for (int look = OD_LOOK_NATURAL; look <= OD_LOOK_PRESERVE; ++look) {
    bool all = true;
    for (int i = 0; i < kLookSetSize && all; ++i) {
      const double v = values[i];
      all = v == v && std::fabs(v - LookRecipeValue(look, kLookSet[i])) <= kLookTolerance;
    }
    if (all) return look;
  }
  return OD_LOOK_CUSTOM;
}

// An advanced slider: percent -> core multiplier, clamped to the UI range; a
// NaN takes the default. At the defaults every division is exact (100 / 100,
// 75 / 100), so the core sees its neutral values.
static float Percent(double v, double lo, double hi, double dflt) {
  if (!(v == v)) v = dflt;
  return (float)(std::min(hi, std::max(lo, v)) / 100.0);
}
static uint32_t Checkbox(double v, double dflt) {
  if (!(v == v)) v = dflt;
  return v != 0 ? 1u : 0u;
}

bool KeysNameContent() {
  // Premiere 26.x hands a GPU filter no content generation for its frames.
  // Should one appear, mix it into MakeFrameKey and return true here.
  return false;
}

ODParams ToCoreParams(const UIParams& ui, bool hostWantsFull, bool wait) {
  ODParams p = od_params_default();
  p.lumaStrength = Clamp01(ui.luma / 100.0);
  p.chromaStrength = Clamp01(ui.chroma / 100.0);
  p.temporalRadius = ClampRadius(ui.temporalRadius);
  p.temporalStrength = Clamp01(ui.temporalStrength / 100.0);
  p.spatialStrength = Clamp01(ui.spatial / 100.0);
  p.sharpen = Clamp01(ui.sharpen / 100.0);
  double nl = ui.noiseLevel == ui.noiseLevel ? ui.noiseLevel : OD_DFLT_NOISE_LEVEL;
  nl = std::min((double)OD_NOISE_LEVEL_MAX, std::max((double)OD_NOISE_LEVEL_MIN, nl));
  p.noiseLevel = (float)(nl / 100.0);
  if (ui.quality == OD_QUALITY_FULL) p.quality = 1;
  else if (ui.quality == OD_QUALITY_FAST) p.quality = 0;  // exports too: its name says so
  else p.quality = hostWantsFull ? 1 : 0;  // Auto, and anything unexpected
  // Unshipped diagnostic items and corrupt saved values fall back to Result.
  // Check before subtracting, including INT_MIN from a malformed project.
  p.view = (ui.view >= 1 && ui.view <= OD_NUM_VIEWS) ? ui.view - 1 : OD_VIEW_OUTPUT;
  p.waitUntilCompleted = wait ? 1 : 0;
  // Playback (the host did not ask for a high-quality render): a stall drops
  // frames, so the core does not wait for the GPU to run Premiere's queue,
  // whose work ahead of ours can be several frames (it can still block
  // obtaining a command buffer from a full queue: ODStats.commandBufferMs). An export blocks for the profile's first
  // estimates and restarts, so every frame from the first uses a measured one.
  p.realtime = (!hostWantsFull && !wait) ? 1 : 0;
  p.manualProfile = nullptr;
  // Nothing tells the plug-in when the pixels behind a frame's time change (an effect below
  // this one, source settings, a proxy toggle), so the core reuses nothing between renders
  // (verifyFrameCache): every frame of the window is ingested on every render, exports and
  // playback alike. A sampled check before reusing a neighbour let changes between the
  // samples through, and a check of every pixel costs what the ingest costs. The keys tell
  // the automatic profile which frames it has measured, where the footage cuts or jumps, and
  // (through samples of the current frame) when the pixels under a key changed.
  // OD_FRAME_CACHE=0 in the host's environment drops that last part too (noFrameCache): a
  // switch for diagnosing a problem in the host without a rebuild.
  static const bool cacheOff = [] {
    const char* e = getenv("OD_FRAME_CACHE");
    return e && e[0] == '0';
  }();
  p.noFrameCache = cacheOff ? 1 : 0;
  p.verifyFrameCache = KeysNameContent() ? 0 : 1;
  // Advanced controls (API v6). A project saved without them gets their
  // defaults from the host, which are the core's neutral values.
  p.advanced = 1;
  for (int b = 0; b < OD_BANDS; ++b) {
    p.bandNoiseLevel[0][b] = Percent(ui.lumaLevel[b], OD_BAND_LEVEL_MIN, OD_BAND_LEVEL_MAX, OD_DFLT_BAND_LEVEL);
    p.bandNoiseLevel[1][b] = Percent(ui.chromaLevel[b], OD_BAND_LEVEL_MIN, OD_BAND_LEVEL_MAX, OD_DFLT_BAND_LEVEL);
    p.bandAmount[0][b] = Percent(ui.lumaAmount[b], 0, 100, OD_DFLT_BAND_AMOUNT);
    p.bandAmount[1][b] = Percent(ui.chromaAmount[b], 0, 100, OD_DFLT_BAND_AMOUNT);
  }
  p.channelNoiseLevel[0] = Percent(ui.noiseY, OD_BAND_LEVEL_MIN, OD_BAND_LEVEL_MAX, OD_DFLT_CHANNEL_NOISE);
  p.channelNoiseLevel[1] = Percent(ui.noiseCb, OD_BAND_LEVEL_MIN, OD_BAND_LEVEL_MAX, OD_DFLT_CHANNEL_NOISE);
  p.channelNoiseLevel[2] = Percent(ui.noiseCr, OD_BAND_LEVEL_MIN, OD_BAND_LEVEL_MAX, OD_DFLT_CHANNEL_NOISE);
  p.grainFloor = Percent(ui.grainFloor, 0, OD_GRAIN_FLOOR_MAX, OD_DFLT_GRAIN_FLOOR);
  p.speckRemoval = Checkbox(ui.removeSpecks, OD_DFLT_REMOVE_SPECKS);
  p.temporalMotionThreshold =
      Percent(ui.motionThreshold, OD_MOTION_THRESHOLD_MIN, OD_MOTION_THRESHOLD_MAX, OD_DFLT_MOTION_THRESHOLD);
  p.temporalEdgeGate = Percent(ui.edgeGate, 0, 100, OD_DFLT_EDGE_GATE);
  p.sharpenBand[0] = Percent(ui.sharpenFine, 0, OD_SHARPEN_BAND_MAX, OD_DFLT_SHARPEN_BAND);
  p.sharpenBand[1] = Percent(ui.sharpenMedium, 0, OD_SHARPEN_BAND_MAX, OD_DFLT_SHARPEN_BAND);
  p.sharpenCoring = Percent(ui.sharpenNoiseFloor, OD_SHARPEN_NOISE_FLOOR_MIN, OD_SHARPEN_NOISE_FLOOR_MAX,
                            OD_DFLT_SHARPEN_NOISE_FLOOR);
  p.sharpenHalo = Percent(ui.sharpenHalo, 0, OD_SHARPEN_HALO_MAX, OD_DFLT_SHARPEN_HALO);
  p.profileHold = Checkbox(ui.holdProfile, OD_DFLT_HOLD_PROFILE);
  // Experimental controls (API v9): 0 % and off are the core's neutral values.
  p.grainStability = Percent(ui.grainStability, 0, 100, OD_DFLT_GRAIN_STABILITY);
  p.coarseChroma = Checkbox(ui.coarseChroma, OD_DFLT_COARSE_CHROMA);
  // Minimum grain applies to (API v10). Whole picture (previous) is the
  // picture of every build before it, and the value of a project saved
  // before the control existed; anything else, a NaN included, is the
  // default, Finest grain only.
  const double gm = ui.grainMode == ui.grainMode ? ui.grainMode : OD_DFLT_GRAIN_MODE;
  p.grainMode = std::lround(gm) == OD_GRAIN_MODE_WHOLE ? OD_GRAIN_WHOLE : OD_GRAIN_FINEST;
  if (EffectiveControlSet(ui) == OD_CONTROLS_DB) {
    const double luma = Reduction(ui.dbLuma), chroma = Reduction(ui.dbChroma);
    p.lumaStrength = (float)luma;
    p.chromaStrength = (float)chroma;
    p.grainFloor = (float)(Bounded(ui.dbGrain,0,6.57,0.657) / 0.657);
    const double threshold = Bounded(ui.dbSpatial,-18,6);
    p.spatialStrength = threshold <= -18 ? 0 : (float)(0.5 * Gain(threshold,-18,6));
    p.temporalRadius = ClampRadius(Bounded(ui.dbFrames,1,6,3)-1);
    p.temporalStrength = 1;
    p.noiseLevel = (float)Gain(ui.dbProfile,-12.05,12.05);
    p.temporalMotionThreshold = (float)Gain(ui.dbMotion,-6.02,6.02);
    p.sharpenCoring = (float)Gain(ui.dbSharpen,-6.02,9.54);
    for (int b = 0; b < 4; ++b) {
      p.bandAmount[0][b] = luma > 0 ? (float)std::min(1.0,Reduction(ui.dbLumaLimit[b])/luma) : 1;
      p.bandAmount[1][b] = chroma > 0 ? (float)std::min(1.0,Reduction(ui.dbChromaLimit[b])/chroma) : 1;
      p.bandNoiseLevel[0][b] = (float)Gain(ui.dbLumaCorrection[b],-12,12);
      p.bandNoiseLevel[1][b] = (float)Gain(ui.dbChromaCorrection[b],-12,12);
    }
    for (int c = 0; c < 3; ++c) p.channelNoiseLevel[c] = (float)Gain(ui.dbChannel[c],-12,12);
  }
  ApplyProcessingVersion(p, ui.processing == 1 ? 1 : 2);
  return p;
}

std::vector<int> NeighbourOffsets(int radius, int64_t time, int64_t tpf, const TimeBounds& bounds, int64_t earliest,
                                  int step) {
  std::vector<int> out;
  radius = std::min(OD_MAX_RADIUS, std::max(0, radius));
  if (tpf <= 0) return out;
  step = std::max(1, step);
  for (int d = 1; d <= radius; ++d) {
    for (int s = -1; s <= 1; s += 2) {
      const int64_t t = time + (int64_t)(s * d) * step * tpf;
      if (t < earliest) continue;
      if (bounds.known && (t < bounds.start || t >= bounds.end)) continue;
      out.push_back(s * d);
    }
  }
  return out;
}

// ---- contexts ---------------------------------------------------------------

namespace {
struct ContextEntry {
  ODContext* ctx = nullptr;
  id<MTLDevice> device = nil;      // retained: keeps the key pointers valid
  id<MTLCommandQueue> queue = nil;
  int sessions = 0;
};
std::mutex gContextMutex;
std::map<std::pair<void*, void*>, ContextEntry>& Contexts() {
  static auto* m = new std::map<std::pair<void*, void*>, ContextEntry>();  // never destroyed:
  return *m;                           // hosts may call us during static teardown
}
}  // namespace

ODContext* ContextFor(void* device, void* queue, char* err, size_t errLen) {
  if (err && errLen) err[0] = 0;
  if (!device) {
    if (err && errLen) snprintf(err, errLen, "no Metal device");
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(gContextMutex);
  auto key = std::make_pair(device, queue);
  auto it = Contexts().find(key);
  if (it != Contexts().end()) return it->second.ctx;
  ODContext* ctx = od_context_create(device, queue, err, errLen);
  if (!ctx) return nullptr;
  ContextEntry e;
  e.ctx = ctx;
  e.device = (__bridge id<MTLDevice>)device;
  e.queue = queue ? (__bridge id<MTLCommandQueue>)queue : nil;
  Contexts()[key] = e;
  return ctx;
}

static void AdjustSessionCount(ODContext* ctx, int delta) {
  std::lock_guard<std::mutex> lock(gContextMutex);
  for (auto& kv : Contexts())
    if (kv.second.ctx == ctx) kv.second.sessions += delta;
}

void ReleaseAllContexts() {
  std::lock_guard<std::mutex> lock(gContextMutex);
  for (auto it = Contexts().begin(); it != Contexts().end();) {
    if (it->second.sessions <= 0) {
      od_context_destroy(it->second.ctx);
      it = Contexts().erase(it);
    } else {
      ++it;  // a live instance still uses it; leak rather than crash
    }
  }
}

// ---- sessions ---------------------------------------------------------------

Session* Session::Create(ODContext* ctx) {
  if (!ctx) return nullptr;
  ODSession* s = od_session_create(ctx);
  if (!s) return nullptr;
  Session* out = new Session();
  out->ctx_ = ctx;
  out->session_ = s;
  AdjustSessionCount(ctx, +1);
  return out;
}

Session::~Session() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (session_) od_session_destroy(session_);
  session_ = nullptr;
  AdjustSessionCount(ctx_, -1);
}

template<bool recordAnalysis>
int Session::Render(const ODFrame* frames, const int* offsets, int count, const ODParams* p,
                    const ODFrame* out, ODStats* stats, char* err, size_t errLen, double* lockWaitMs,
                    char* asyncError, uintptr_t analysisSource) {
  // OD_HOST_TRACE (diagnostics): one log line per render with this wrapper's
  // lock wait and the time inside the core.
  static const bool trace = getenv("OD_HOST_TRACE") != nullptr;
  const uint64_t from = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  std::lock_guard<std::mutex> lock(mutex_);
  const uint64_t acquired = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
  const double waited = (double)(acquired - from) / 1e6;
  if (err && errLen) err[0] = 0;
  int rc;
  ODStats localStats{};
  if constexpr (recordAnalysis) { if (!stats) stats = &localStats; }
  @autoreleasepool {
    rc = od_render(session_, frames, offsets, count, p, out, stats);
    if constexpr (recordAnalysis) { if (rc == 0 && stats->renderId) {
      auto& record = analysisRecords_[stats->renderId % analysisRecords_.size()];
      record = {stats->renderId,analysisSource,p ? *p : od_params_default()};
      record.params.manualProfile = nullptr;
    } else analysisRecords_ = {}; }
    if (rc != 0 && err && errLen) {
      const char* m = od_last_error(session_);
      snprintf(err, errLen, "%s", m ? m : "unknown error");
    }
    if (asyncError && errLen && stats && stats->asyncFailures > 0) {
      const char* m = od_first_async_error(session_);
      snprintf(asyncError, errLen, "%s", m ? m : "");
    }
  }
  if (lockWaitMs) *lockWaitMs = waited + (stats ? stats->lockWaitMs : 0);
  if (trace) {
    const double inside = (double)(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - acquired) / 1e6;
    Log("PLUGIN_HOST,lock_ms=%.6f,render_ms=%.6f,total_ms=%.6f,rc=%d", waited, inside, waited + inside, rc);
  }
  return rc;
}

template int Session::Render<false>(const ODFrame*, const int*, int, const ODParams*,
    const ODFrame*, ODStats*, char*, size_t, double*, char*, uintptr_t);
template int Session::Render<true>(const ODFrame*, const int*, int, const ODParams*,
    const ODFrame*, ODStats*, char*, size_t, double*, char*, uintptr_t);

int Session::Profile(const ODFrame& frame, ODNoiseProfile& profile, char* error, size_t size, const ODParams* params) {
  std::lock_guard<std::mutex> lock(mutex_);
  const int rc = od_profile_frame_with_params(session_, &frame, params, &profile);
  if (error && size) snprintf(error, size, "%s", rc ? od_last_error(session_) : "");
  return rc;
}

bool Session::TryAnalysis(ODAnalysis& a, ODParams& p, uintptr_t source) {
  a = {};
  std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock() || od_session_last_analysis(session_, &a) != 1) { a = {}; return false; }
  const auto& record = analysisRecords_[a.renderId % analysisRecords_.size()];
  if (record.id != a.renderId || source != record.source) { a = {}; return false; }
  p = record.params;
  return true;
}

void Session::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  analysisRecords_ = {};
  od_session_reset(session_);
}

// ---- sessions shared by successive GPU instances ----------------------------

namespace {
struct SharedEntry {
  std::shared_ptr<Session> session;
  int users = 0;
  double parkedAt = 0;  // ms, valid while users == 0
};
using SharedKey = std::pair<ODContext*, uint64_t>;
struct SharedRegistry {
  std::mutex mutex;
  std::map<SharedKey, SharedEntry> entries;
  dispatch_queue_t queue = nil;   // the expiry timer's serial queue
  dispatch_source_t timer = nil;  // nil until first needed, and after shutdown
};
SharedRegistry& Registry() {
  static auto* r = new SharedRegistry();  // never destroyed, like Contexts()
  return *r;
}

double NowMs() { return (double)clock_gettime_nsec_np(CLOCK_MONOTONIC) / 1e6; }

double ParkMs() {
  static const double ms = [] {
    const char* e = getenv("OD_PARK_MS");
    char* end = nullptr;
    const double v = e ? strtod(e, &end) : -1;
    return (e && end != e && v >= 0 && v <= 600000) ? v : (double)kParkMsDefault;
  }();
  return ms;
}

void ExpireParked();

// Called with r.mutex held. Arms the timer for the earliest expiry, if any.
void ArmTimerLocked(SharedRegistry& r) {
  double earliest = -1;
  for (auto& kv : r.entries)
    if (kv.second.users == 0 && (earliest < 0 || kv.second.parkedAt < earliest)) earliest = kv.second.parkedAt;
  if (earliest < 0) return;
  if (!r.timer) {
    if (!r.queue) r.queue = dispatch_queue_create("dev.opendenoise.plugin.park", DISPATCH_QUEUE_SERIAL);
    r.timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, r.queue);
    if (!r.timer) return;  // parked sessions then go at the next release or at shutdown
    dispatch_source_set_event_handler(r.timer, ^{ ExpireParked(); });
    dispatch_resume(r.timer);
  }
  const double wait = std::max(0.0, earliest + ParkMs() - NowMs()) + 5;  // ms; a little late, never early
  dispatch_source_set_timer(r.timer, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(wait * 1e6)), DISPATCH_TIME_FOREVER,
                            (uint64_t)20 * NSEC_PER_MSEC);
}

// Called with r.mutex held: moves out the parked sessions that should go, so
// that the caller destroys them after unlocking (od_session_destroy may wait
// for the GPU).
void CollectLocked(SharedRegistry& r, bool all, std::vector<std::shared_ptr<Session>>* out) {
  const double now = NowMs();
  for (auto it = r.entries.begin(); it != r.entries.end();) {
    if (it->second.users == 0 && (all || now - it->second.parkedAt >= ParkMs())) {
      out->push_back(std::move(it->second.session));
      it = r.entries.erase(it);
    } else {
      ++it;
    }
  }
  if (all) return;
  // Bound the number parked: the oldest go first.
  for (;;) {
    int parked = 0;
    auto oldest = r.entries.end();
    for (auto it = r.entries.begin(); it != r.entries.end(); ++it) {
      if (it->second.users != 0) continue;
      ++parked;
      if (oldest == r.entries.end() || it->second.parkedAt < oldest->second.parkedAt) oldest = it;
    }
    if (parked <= kMaxParked) break;
    out->push_back(std::move(oldest->second.session));
    r.entries.erase(oldest);
  }
}

void ExpireParked() {
  SharedRegistry& r = Registry();
  std::vector<std::shared_ptr<Session>> dead;
  {
    std::lock_guard<std::mutex> lock(r.mutex);
    CollectLocked(r, false, &dead);
    if (r.timer) ArmTimerLocked(r);
  }
  dead.clear();  // destroyed here, outside the lock
}
}  // namespace

std::shared_ptr<Session> AcquireSharedSession(ODContext* ctx, uint64_t identity) {
  if (!ctx) return nullptr;
  if (identity == 0) return std::shared_ptr<Session>(Session::Create(ctx));
  SharedRegistry& r = Registry();
  std::lock_guard<std::mutex> lock(r.mutex);
  SharedEntry& e = r.entries[SharedKey(ctx, identity)];
  if (!e.session) {
    // An exception (allocation) must not leave an empty entry behind.
    try {
      e.session = std::shared_ptr<Session>(Session::Create(ctx));
    } catch (...) {
      r.entries.erase(SharedKey(ctx, identity));
      throw;
    }
    if (!e.session) {
      r.entries.erase(SharedKey(ctx, identity));
      return nullptr;
    }
  }
  ++e.users;
  return e.session;
}

void ReleaseSharedSession(ODContext* ctx, uint64_t identity) {
  if (!ctx || identity == 0) return;
  SharedRegistry& r = Registry();
  std::vector<std::shared_ptr<Session>> dead;
  {
    std::lock_guard<std::mutex> lock(r.mutex);
    auto it = r.entries.find(SharedKey(ctx, identity));
    if (it == r.entries.end() || it->second.users <= 0) return;
    if (--it->second.users == 0) {
      it->second.parkedAt = NowMs();
      CollectLocked(r, false, &dead);
      ArmTimerLocked(r);
    }
  }
  dead.clear();
}

void ShutdownSharedSessions() {
  SharedRegistry& r = Registry();
  std::vector<std::shared_ptr<Session>> dead;
  dispatch_queue_t queue = nil;
  {
    std::lock_guard<std::mutex> lock(r.mutex);
    CollectLocked(r, true, &dead);
    if (r.timer) {
      dispatch_source_cancel(r.timer);
      r.timer = nil;
    }
    queue = r.queue;
  }
  // Let a handler that was already running finish before the host may unload
  // us. (It finds nothing to do: the entries are gone and the timer cancelled.)
  if (queue) dispatch_sync(queue, ^{});
  dead.clear();
}

// ---- module lifetime ----------------------------------------------------------

void PinImage() {
  static std::once_flag once;
  std::call_once(once, [] {
    Dl_info info = {};
    if (!dladdr((const void*)&PinImage, &info) || !info.dli_fname) {
      Log("could not find this image's path: it is not pinned against unloading");
      return;
    }
    // A reference that is never given back: the host's own dlclose (or
    // CFBundleUnloadExecutable) then leaves the image mapped. (RTLD_NODELETE
    // alone does not do it for an image already loaded: on macOS 27 the
    // image went once its count reached 0; plugin/tests/harness.mm,
    // TestUnload.)
    if (!dlopen(info.dli_fname, RTLD_NOLOAD | RTLD_NODELETE | RTLD_LAZY | RTLD_LOCAL))
      Log("could not pin this image against unloading: %s", dlerror());
  });
}

void GlobalTeardown(const char* who, bool gpuSessions) {
  static const double waitMs = [] {
    const char* e = getenv("OD_TEARDOWN_MS");
    char* end = nullptr;
    const double v = e ? strtod(e, &end) : -1;
    return (e && end != e && v >= 0 && v <= 60000) ? v : (double)kTeardownWaitMsDefault;
  }();
  const double t0 = NowMs();
  if (gpuSessions) ShutdownSharedSessions();  // parked sessions first, or their contexts stay
  const int left = od_wait_idle(waitMs);
  ReleaseAllContexts();
  Log("%s: global teardown: %.1f ms waiting for asynchronous work, %d item%s left%s", who, NowMs() - t0, left,
      left == 1 ? "" : "s", left ? " (the image is pinned: they may still run)" : "");
}

}  // namespace odplug
