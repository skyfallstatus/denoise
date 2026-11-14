// SPDX-License-Identifier: MIT
// odbench — command-line harness for the Open Denoise core.
//
// Feeds od_render the way Premiere would: every output frame gets its
// temporal neighbours, with frameKeys, each frame in its own MTLBuffer. Frames
// come from a synthetic generator, a raw BGRA float32 file, or any video
// through an ffmpeg pipe. Reports GPU / wall / CPU time per frame and can
// write the result as raw float or through ffmpeg.
//
// Memory is bounded and stated at start-up: 2*radius+1 host frames, one output
// frame or one in-place scratch frame, two float staging frames (plus one
// clean base image for the synthetic source); with --async, --in-flight more
// of each kind the run reuses. Frames are streamed; nothing else is
// held. A deadline (--deadline, default 600 s) kills the run with SIGALRM, and
// the run stops if the process footprint passes --mem-limit-gb (default 20,
// at most 24).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <ImageIO/ImageIO.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

#include <mach/mach.h>
#include <mach/mach_time.h>

#include "opendenoise/opendenoise.h"
#include "opendenoise/quality.h"
#include "opendenoise/profiling.h"

#include "debug.h"

namespace {

const char* kFfmpeg = "/opt/homebrew/bin/ffmpeg";
const char* kFfprobe = "/opt/homebrew/bin/ffprobe";

struct Options {
  // input
  std::string raw, video;
  std::string videoMatrix = "auto";  // --video-matrix: auto, bt709, bt601 or ffmpeg (decideMatrix)
  uint32_t width = 0, height = 0;  // 0 = default (1920x1080) or native video size
  float noise = 0.03f;
  float motion = 2.0f;
  uint32_t seed = 1;
  int start = 0;
  // run
  ODPixelFormat format = OD_PIXEL_BGRA32F;
  int frames = 60;
  int warmup = 5;
  uint32_t rowPad = 0;
  bool randomAccess = false, noCache = false, inPlace = false, async = false, perFrame = false;
  bool stageTimings = false;
  bool deterministicProfile = false;
  std::string profileLoad, profileSave;
  bool preload = false;
  bool autoQuality = false;
  int inFlight = 3;       // --async: renders that may be in flight before their buffers are reused
  int holdGpuMs = 0;      // --async debug: the queue is held from the first render (see usage)
  bool unfenced = false;  // --async debug: no fences, ring and scratch at their old sizes
  bool fenceElsewhere = false;  // --async debug: fences committed on a queue of their own (they order nothing)
  double intervalMs = 0;  // > 0: one render every intervalMs (real-time pacing)
  int sessions = 1;       // effect instances sharing the context
  int renders = 0;        // 0 = one per frame (times --sessions when sequential)
  double memLimitGB = 20;
  int idleMs = 0;         // after the run: watch memory while nothing renders
  bool proresStandard = false;
  int freezeProfile = -1;
  std::string profileLog, pairTune, analysisLog, analysisPNG;
  bool analysisPrint = false, analysisAB = false;
  bool manual = false;
  float manualSigma = 0.02f;
  ODParams params = od_params_default();
  // output
  std::string outRaw, outVideo, json, label;
  double fps = 25;
  int deadline = 600;
};

[[noreturn]] void die(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void die(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "odbench: ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
  exit(2);
}

void usage() {
  printf(
      "usage: odbench [options]\n"
      "\n"
      "Input (default: synthetic)\n"
      "  --synthetic            moving test scene with Gaussian noise (default)\n"
      "  --noise SIGMA          synthetic noise sigma, 0 = clean reference (default 0.03)\n"
      "  --motion PX            synthetic horizontal motion in pixels per frame (default 2)\n"
      "  --seed N               synthetic / random-access seed (default 1)\n"
      "  --raw FILE             raw frames, BGRA float32, tightly packed; needs --size\n"
      "  --video FILE           any video, decoded by ffmpeg (16-bit RGBA -> float)\n"
      "  --video-matrix auto|bt709|bt601|ffmpeg\n"
      "                         Y'CbCr matrix of the decode (default auto: the stream's tag; untagged\n"
      "                         HD/UHD as BT.709, as Premiere reads it; ffmpeg: ffmpeg's own reading,\n"
      "                         untagged as BT.601)\n"
      "  --start N              skip N source frames (raw / video)\n"
      "  --size WxH             frame size (default 1920x1080; video: the size ffmpeg delivers, in display\n"
      "                         orientation (a rotated phone clip comes out upright), or scaled to this)\n"
      "\n"
      "Run\n"
      "  --format 32f|16f       host pixel format (default 32f)\n"
      "  --frames N             output frames to render (default 60)\n"
      "  --warmup N             leading frames left out of the statistics (default 5)\n"
      "  --row-pad BYTES        extra bytes per row in the host buffers (multiple of 4)\n"
      "  --random-access        render the frames in a shuffled order (not with --video)\n"
      "  --no-cache             pass OD_NO_KEY: no per-frame caching, no keys for the profile\n"
      "  --no-frame-cache       keys for the profile, nothing cached (ODParams.noFrameCache): what\n"
      "                         the plug-in passed until API v5\n"
      "  --verify-cache         keys that do not promise the pixels (ODParams.verifyFrameCache): every\n"
      "                         frame ingested, samples kept for the profile: what the plug-in passes\n"
      "  --in-place             output into the current frame's buffer\n"
      "  --async                waitUntilCompleted = 0; reports throughput only\n"
      "  --realtime             with --async: ODParams.realtime, what the plug-in passes while the host plays\n"
      "                         (no call waits for the GPU; a size's first profile is waited for a few ms)\n"
      "  --in-flight N          with --async: renders that may be in flight at once (default 3). Every host\n"
      "                         buffer the CPU writes or reads again waits for the render that last used it\n"
      "                         (a fence per buffer); the ring, the in-place scratch and the outputs get N\n"
      "                         spare buffers so that the waits are rare, and outputs are read back late\n"
      "  --hold-gpu-ms T        with --async, a check: the queue is held from the first render until odbench\n"
      "                         would wait for the GPU, or T ms at most, so the CPU runs ahead of it; the\n"
      "                         report counts \"hazards\", CPU writes into a host buffer before the GPU\n"
      "                         work of the render that last used it had ended, judged by that render's\n"
      "                         reported completion (ODStats.completed), not by the fences\n"
      "  --debug-unfenced       with --async: no fences, ring and scratch at their old sizes (what odbench did\n"
      "                         before 12 Nov 2026); for the test that shows what the fences prevent\n"
      "  --debug-fence-elsewhere  with --async: each render's fence goes on a command queue of its own, so it\n"
      "                         orders nothing (a broken fence); for the test that the hazard count does\n"
      "                         not take the fences' word for it\n"
      "  --preload              load every source frame into host buffers first, so the renders run\n"
      "                         back to back with no frame generation in between (the GPU stays\n"
      "                         busy, as in a host that is playing). Memory: frames x frame size.\n"
      "  --only-stage LIST      run ingest/output plus only these of profile,temporal,spatial,sharpen\n"
      "                         (\"none\" = ingest + output only: the pass-through floor)\n"
      "  --skip-stage LIST      skip these stages\n"
      "  --deterministic-profile  measure from this window only; blocking profile resolve\n"
      "  --profile-load FILE   read a versioned captured profile payload\n"
      "  --profile-save FILE   capture first current frame and save/use its profile\n"
      "  --stage-timings        per-stage GPU time (separate command buffers; slower overall)\n"
      "  --interval-ms F        start one render every F ms (real-time pacing, e.g. 40 = 25 fps); the\n"
      "                         wait is outside the measured wall time. Default 0 = back to back\n"
      "  --sessions N           N sessions (effect instances) on one context. Sequential: every frame is\n"
      "                         rendered on each session in turn (stacked tracks). --random-access: each\n"
      "                         render picks a random session and a random frame (scrubbing a timeline\n"
      "                         of N clips). Reports process and GPU memory\n"
      "  --renders N            renders in total (default: one per frame, times N sessions if sequential)\n"
      "  --idle-ms T            after the last render, watch memory for T ms while nothing renders\n"
      "  --mem-limit-gb G       stop the run when the process footprint passes G GB (default 20, max 24)\n"
      "  --io direct|blit|view  buffer<->texture route (default direct; others are for measurement)\n"
      "\n"
      "Params (defaults = od_params_default)\n"
      "  --luma F  --chroma F  --radius N  --temporal F  --spatial F  --sharpen F\n"
      "  --noise-level F  --quality 0|1|auto  --view output|noise|motion|profile|split|analysis\n"
      "  --manual-sigma S       manual noise profile: that of white Gaussian noise of sigma S on R, G\n"
      "                         and B (what --noise S adds), instead of the automatic profile\n"
      "  --identity             all strengths 0 (the copy fast path)\n"
      "  --param NAME=VALUE     an advanced control (API v6), as the core takes it (multipliers, 0/1):\n"
      "                         nlY0..nlY3 nlC0..nlC3 (band noise level), amY0..amY3 amC0..amC3 (band\n"
      "                         amount), cnY cnCb cnCr (channel noise level), floor, speck, mthr, egate,\n"
      "                         shf, shm, shcore, shhalo, hold; API v9: gstab (grain stability 0..1),\n"
      "                         cchroma (coarse chroma cleanup 0/1); API v10: grain (0 = the grain floor\n"
      "                         over every size and the API v9 profile, 1 = finest grain only, the\n"
      "                         default); advanced=0 = the API v5 caller.\n"
      "                         Repeatable\n"
      "\n"
      "Output\n"
      "  --prores-standard      encode .mov with Apple ProRes 422, p210le, BT.709 tv\n"
      "  --pair-tune TEXT        two sessions: baseline then TEXT spatial tune on each frame\n"
      "  --temporal-noise-model N  0 legacy (default), 1 measured merge correction\n"
      "  --freeze-profile N     freeze the profile after frame N (blocking, single session)\n"
      "  --chroma-recovery      recover collapsed automatic chroma fits (API v13)\n"

      "  --analysis             print analysis JSON and both status rows per frame (blocking)\n"
      "  --analysis-log FILE    write the same snapshots as JSON lines (blocking)\n"
      "  --analysis-png FILE    render Analysis view; save the final frame as PNG\n"
      "  --analysis-ab          interleave two renders of each frame, analysis off/on\n"
      "  --no-analysis          omit analysis work (interleaved timing baseline)\n"
      "  --profile-log FILE     write per-render noise profiles as JSON lines\n"
      "  --out-raw FILE         write the result as raw BGRA float32\n"
      "  --out FILE             write through ffmpeg: .mov = ProRes 4444, name with %%05d.png = 16-bit PNGs\n"
      "  --fps F                frame rate for --out (default 25)\n"
      "  --per-frame            print one line per frame\n"
      "  --json FILE            write the report as JSON (\"-\" = stdout, text report goes to stderr)\n"
      "  --label TEXT           free text copied into the report\n"
      "  --deadline SEC         hard time limit for the whole run (default 600)\n");
}

// Host time in seconds: the clock of MTLCommandBuffer.GPUStartTime/EndTime,
// which the core's ODCompletion latencies are measured on (the hazard count).
double hostSeconds() {
  static const double scale = [] {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    return (double)tb.numer / (double)tb.denom * 1e-9;
  }();
  return (double)mach_absolute_time() * scale;
}

double nowMs() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

std::string shellQuote(const std::string& s) {
  std::string q = "'";
  for (char c : s) {
    if (c == '\'') q += "'\\''";
    else q += c;
  }
  return q + "'";
}

// --param NAME=VALUE: one advanced field of ODParams (API v6, v9). The names
// are those of the stub core's record (plugin/stub/od_stub.mm).
bool setAdvanced(ODParams* p, const std::string& kv) {
  const size_t eq = kv.find('=');
  if (eq == std::string::npos) return false;
  const std::string n = kv.substr(0, eq);
  const double v = atof(kv.c_str() + eq + 1);
  if (n.size() == 4 && (n.compare(0, 3, "nlY") == 0 || n.compare(0, 3, "nlC") == 0 || n.compare(0, 3, "amY") == 0 ||
                        n.compare(0, 3, "amC") == 0) && n[3] >= '0' && n[3] < '0' + OD_BANDS) {
    const int g = n[2] == 'C' ? 1 : 0, b = n[3] - '0';
    (n[0] == 'n' ? p->bandNoiseLevel : p->bandAmount)[g][b] = (float)v;
    return true;
  }
  if (n == "cnY") p->channelNoiseLevel[0] = (float)v;
  else if (n == "cnCb") p->channelNoiseLevel[1] = (float)v;
  else if (n == "cnCr") p->channelNoiseLevel[2] = (float)v;
  else if (n == "floor") p->grainFloor = (float)v;
  else if (n == "speck") p->speckRemoval = v != 0 ? 1u : 0u;
  else if (n == "mthr") p->temporalMotionThreshold = (float)v;
  else if (n == "egate") p->temporalEdgeGate = (float)v;
  else if (n == "shf") p->sharpenBand[0] = (float)v;
  else if (n == "shm") p->sharpenBand[1] = (float)v;
  else if (n == "shcore") p->sharpenCoring = (float)v;
  else if (n == "shhalo") p->sharpenHalo = (float)v;
  else if (n == "hold") p->profileHold = v != 0 ? 1u : 0u;
  else if (n == "gstab") p->grainStability = (float)v;
  else if (n == "cchroma") p->coarseChroma = v != 0 ? 1u : 0u;
  else if (n == "grain") p->grainMode = v != 0 ? (uint32_t)OD_GRAIN_FINEST : (uint32_t)OD_GRAIN_WHOLE;
  else if (n == "advanced") p->advanced = v != 0 ? 1u : 0u;
  else return false;
  return true;
}

uint32_t stageMask(const std::string& list) {
  uint32_t m = 0;
  size_t pos = 0;
  while (pos <= list.size()) {
    size_t e = list.find(',', pos);
    if (e == std::string::npos) e = list.size();
    std::string n = list.substr(pos, e - pos);
    if (n == "profile") m |= 1u << OD_STAGE_PROFILE;
    else if (n == "temporal") m |= 1u << OD_STAGE_TEMPORAL;
    else if (n == "spatial") m |= 1u << OD_STAGE_SPATIAL;
    else if (n == "sharpen") m |= 1u << OD_STAGE_SHARPEN;
    else if (n == "none" || n == "io" || n.empty()) {
    } else die("unknown stage '%s' (profile, temporal, spatial, sharpen, none)", n.c_str());
    pos = e + 1;
  }
  return m;
}

const uint32_t kMiddle = (1u << OD_STAGE_PROFILE) | (1u << OD_STAGE_TEMPORAL) | (1u << OD_STAGE_SPATIAL) |
                         (1u << OD_STAGE_SHARPEN);
const char* kStageNames[OD_STAGE_COUNT] = {"ingest", "profile", "temporal", "spatial", "sharpen", "output"};

Options parse(int argc, char** argv) {
  Options o;
  auto need = [&](int& i) -> const char* {
    if (i + 1 >= argc) die("%s needs a value", argv[i]);
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      usage();
      exit(0);
    } else if (a == "--synthetic") {
    } else if (a == "--noise") o.noise = (float)atof(need(i));
    else if (a == "--motion") o.motion = (float)atof(need(i));
    else if (a == "--seed") o.seed = (uint32_t)atoi(need(i));
    else if (a == "--raw") o.raw = need(i);
    else if (a == "--video") o.video = need(i);
    else if (a == "--video-matrix") {
      const std::string matrix = need(i);
      if (matrix != "auto" && matrix != "bt709" && matrix != "bt601" && matrix != "ffmpeg")
        die("--video-matrix wants auto, bt709, bt601 or ffmpeg");
      o.videoMatrix = matrix;
    }
    else if (a == "--start") o.start = atoi(need(i));
    else if (a == "--size") {
      if (sscanf(need(i), "%ux%u", &o.width, &o.height) != 2 || !o.width || !o.height) die("--size wants WxH");
    } else if (a == "--format") {
      std::string f = need(i);
      if (f == "32f") o.format = OD_PIXEL_BGRA32F;
      else if (f == "16f") o.format = OD_PIXEL_BGRA16F;
      else die("--format wants 32f or 16f");
    } else if (a == "--frames") o.frames = atoi(need(i));
    else if (a == "--warmup") o.warmup = atoi(need(i));
    else if (a == "--row-pad") o.rowPad = (uint32_t)atoi(need(i));
    else if (a == "--random-access") o.randomAccess = true;
    else if (a == "--no-cache") o.noCache = true;
    else if (a == "--no-frame-cache") o.params.noFrameCache = 1;
    else if (a == "--verify-cache") {
      o.params.noFrameCache = 0;
      o.params.verifyFrameCache = 1;
    }
    else if (a == "--in-place") o.inPlace = true;
    else if (a == "--async") o.async = true;
    else if (a == "--realtime") o.params.realtime = 1;
    else if (a == "--per-frame") o.perFrame = true;
    else if (a == "--only-stage") {
      const uint32_t keep = stageMask(need(i));
      o.params.debugSkipStages = kMiddle & ~keep;
      // The profile stage is otherwise skipped when nothing would use it.
      if (keep & (1u << OD_STAGE_PROFILE)) o.params.debugFlags |= OD_DEBUG_FORCE_PROFILE;
    }
    else if (a == "--preload") o.preload = true;
    else if (a == "--in-flight") o.inFlight = atoi(need(i));
    else if (a == "--hold-gpu-ms") o.holdGpuMs = atoi(need(i));
    else if (a == "--debug-unfenced") o.unfenced = true;
    else if (a == "--debug-fence-elsewhere") o.fenceElsewhere = true;
    else if (a == "--interval-ms") o.intervalMs = atof(need(i));
    else if (a == "--sessions") o.sessions = atoi(need(i));
    else if (a == "--renders") o.renders = atoi(need(i));
    else if (a == "--idle-ms") o.idleMs = atoi(need(i));
    else if (a == "--mem-limit-gb") o.memLimitGB = atof(need(i));
    else if (a == "--skip-stage") o.params.debugSkipStages |= stageMask(need(i));
    else if (a == "--deterministic-profile") o.deterministicProfile = true;
    else if (a == "--profile-load") o.profileLoad = need(i);
    else if (a == "--profile-save") o.profileSave = need(i);
    else if (a == "--stage-timings") o.stageTimings = true;
    else if (a == "--io") {
      std::string m = need(i);
      o.params.debugFlags &= ~(OD_DEBUG_IO_BLIT | OD_DEBUG_IO_VIEW);
      if (m == "blit") o.params.debugFlags |= OD_DEBUG_IO_BLIT;
      else if (m == "view") o.params.debugFlags |= OD_DEBUG_IO_VIEW;
      else if (m != "direct") die("--io wants direct, blit or view");
    } else if (a == "--luma") o.params.lumaStrength = (float)atof(need(i));
    else if (a == "--chroma") o.params.chromaStrength = (float)atof(need(i));
    else if (a == "--radius") o.params.temporalRadius = atoi(need(i));
    else if (a == "--temporal") o.params.temporalStrength = (float)atof(need(i));
    else if (a == "--temporal-noise-model") o.params.temporalNoiseModel = (uint32_t)atoi(need(i));
    else if (a == "--spatial") o.params.spatialStrength = (float)atof(need(i));
    else if (a == "--sharpen") o.params.sharpen = (float)atof(need(i));
    else if (a == "--noise-level") o.params.noiseLevel = (float)atof(need(i));
    else if (a == "--quality") {
      std::string q = need(i); o.autoQuality = q == "auto";
      if (!o.autoQuality && q != "0" && q != "1") die("--quality must be 0, 1 or auto");
      o.params.quality = q == "0" ? 0 : 1;
    }
    else if (a == "--view") {
      std::string v = need(i);
      if (v == "output") o.params.view = OD_VIEW_OUTPUT;
      else if (v == "noise") o.params.view = OD_VIEW_NOISE;
      else if (v == "motion") o.params.view = OD_VIEW_MOTION;
      else if (v == "profile") o.params.view = OD_VIEW_PROFILE;
      else if (v == "split") o.params.view = OD_VIEW_SPLIT;
      else if (v == "analysis") o.params.view = OD_VIEW_ANALYSIS;
      else die("unknown view");
    } else if (a == "--manual-sigma") {
      o.manual = true;
      o.manualSigma = (float)atof(need(i));
    } else if (a == "--identity") {
      o.params.lumaStrength = o.params.chromaStrength = o.params.sharpen = 0;
    } else if (a == "--param") {
      if (!setAdvanced(&o.params, need(i))) die("--param: unknown NAME=VALUE '%s' (try --help)", argv[i]);
    } else if (a == "--prores-standard") o.proresStandard = true;
    else if (a == "--pair-tune") o.pairTune = need(i);
    else if (a == "--freeze-profile") o.freezeProfile = atoi(need(i));
    else if (a == "--chroma-recovery") o.params.chromaRecovery = 1;

    else if (a == "--analysis-ab") { o.analysisAB = true; o.sessions = 2; }
    else if (a == "--analysis") o.analysisPrint = true;
    else if (a == "--analysis-log") o.analysisLog = need(i);
    else if (a == "--analysis-png") { o.analysisPNG = need(i); o.params.view = OD_VIEW_ANALYSIS; }
    else if (a == "--no-analysis") o.params.debugFlags |= OD_DEBUG_NO_ANALYSIS;
    else if (a == "--profile-log") o.profileLog = need(i);
    else if (a == "--out-raw") o.outRaw = need(i);
    else if (a == "--out") o.outVideo = need(i);
    else if (a == "--fps") o.fps = atof(need(i));
    else if (a == "--json") o.json = need(i);
    else if (a == "--label") o.label = need(i);
    else if (a == "--deadline") o.deadline = atoi(need(i));
    else die("unknown option %s (try --help)", a.c_str());
  }
  if (!o.raw.empty() && !o.video.empty()) die("--raw and --video are exclusive");
  if (!o.raw.empty() && !o.width) die("--raw needs --size");
  if (o.randomAccess && !o.video.empty()) die("--random-access needs a seekable source (synthetic or --raw)");
  if (o.frames < 1) die("--frames must be at least 1");
  if (o.analysisAB && (o.async || o.randomAccess || o.sessions != 2 || !o.pairTune.empty())) die("--analysis-ab requires two sequential blocking sessions");
  if (o.async && (o.analysisPrint || !o.analysisLog.empty())) die("per-frame analysis logs require blocking renders");
  if (!o.analysisPNG.empty() && (o.sessions != 1 || o.randomAccess)) die("--analysis-png requires one sequential session");
  if (o.rowPad % 4) die("--row-pad must be a multiple of 4");
  if (o.params.temporalRadius < 0 || o.params.temporalRadius > OD_MAX_RADIUS)
    die("--radius must be 0..%d", OD_MAX_RADIUS);
  if (o.deadline < 1) die("--deadline must be positive");
  if (o.autoQuality && o.params.realtime && (!o.async || !o.preload || o.inPlace || o.stageTimings || o.intervalMs <= 0 ||
                     !o.outRaw.empty() || !o.outVideo.empty() || o.deterministicProfile))
    die("--quality auto --realtime requires --async --preload, positive --interval-ms, no in-place/stage timings/output/deterministic profile");
  if (o.autoQuality && o.deterministicProfile) die("--quality auto cannot use --deterministic-profile");
  if (o.intervalMs < 0 || o.intervalMs > 10000) die("--interval-ms must be 0..10000");
  if (o.sessions < 1 || o.sessions > 256) die("--sessions must be 1..256");
  if (o.renders < 0) die("--renders must be positive");
  if (o.idleMs < 0) die("--idle-ms must be positive");
  if (o.inFlight < 1 || o.inFlight > 16) die("--in-flight must be 1..16");
  if (o.holdGpuMs < 0 || o.holdGpuMs > 10000) die("--hold-gpu-ms must be 0..10000");
  if (o.holdGpuMs > 0 && !o.async) die("--hold-gpu-ms needs --async (a blocking render would wait for the held queue)");
  if (o.holdGpuMs > 0 && o.stageTimings) die("--hold-gpu-ms does not go with --stage-timings (every stage waits)");
  if (o.unfenced && !o.async) die("--debug-unfenced needs --async");
  if (o.fenceElsewhere && !o.async) die("--debug-fence-elsewhere needs --async");
  if (o.fenceElsewhere && o.unfenced) die("--debug-fence-elsewhere and --debug-unfenced exclude each other");
  if (!(o.memLimitGB > 0 && o.memLimitGB <= 24)) die("--mem-limit-gb must be in (0, 24]");
  if (o.sessions > 1 && (!o.outRaw.empty() || !o.outVideo.empty())) die("--sessions does not write output");
  if (o.sessions > 1 && o.randomAccess && !o.video.empty()) die("--random-access needs a seekable source");
  if (o.warmup < 0) o.warmup = 0;
  return o;
}

// ---------------------------------------------------------------------------
// Frame sources. All deliver tightly packed BGRA float32.
// ---------------------------------------------------------------------------
struct Source {
  uint32_t width = 0, height = 0;
  int count = -1;  // -1 = unknown until the end is reached
  virtual ~Source() {}
  virtual bool seekable() const = 0;
  // Returns false at end of input.
  virtual bool read(int index, float* bgra) = 0;
};

inline uint32_t hash32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

// A clean scene (soft gradients, edges, fine texture, saturated patches) that
// translates horizontally, plus independent Gaussian noise per frame and
// channel. Deterministic in (seed, index): --noise 0 gives the clean
// reference of the same sequence.
struct SyntheticSource : Source {
  std::vector<float> base;  // RGB, width x height
  float noise, motion;
  uint32_t seed;
  SyntheticSource(uint32_t w, uint32_t h, int n, float noiseSigma, float motionPx, uint32_t sd)
      : noise(noiseSigma), motion(motionPx), seed(sd) {
    width = w;
    height = h;
    count = n;
    base.resize((size_t)w * h * 3);
    float* b = base.data();
    dispatch_apply(h, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t y) {
      for (uint32_t x = 0; x < w; ++x) {
        float u = (float)x / (float)w, v = (float)y / (float)h;
        float lum = 0.08f + 0.75f * (0.5f + 0.5f * sinf(6.2832f * (u * 1.5f + 0.3f * v))) * (0.35f + 0.65f * v);
        // hard-edged bars and a checker region
        if (v > 0.55f && v < 0.75f) lum = ((int)(u * 24.0f) & 1) ? 0.72f : 0.18f;
        if (v > 0.80f) lum = (((x / 6) ^ ((uint32_t)y / 6)) & 1) ? 0.55f : 0.35f;
        // fine texture
        float tex = 0.04f * sinf(0.9f * (float)x) * sinf(0.7f * (float)y);
        float r = lum + tex + 0.20f * sinf(6.2832f * u * 3.0f) * (v < 0.3f);
        float g = lum + tex;
        float bl = lum + tex + 0.20f * cosf(6.2832f * v * 2.0f) * (u > 0.6f);
        float* p = b + ((size_t)y * w + x) * 3;
        p[0] = r;
        p[1] = g;
        p[2] = bl;
      }
    });
  }
  bool seekable() const override { return true; }
  bool read(int index, float* bgra) override {
    if (index < 0 || index >= count) return false;
    const uint32_t w = width, h = height;
    const float* b = base.data();
    const float sigma = noise;
    const uint32_t salt = hash32(seed * 0x9E3779B9u + (uint32_t)index);
    int shift = (int)lroundf(motion * (float)index) % (int)w;
    if (shift < 0) shift += (int)w;
    dispatch_apply(h, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t y) {
      uint32_t s = hash32(salt ^ ((uint32_t)y * 0x85EBCA6Bu)) | 1u;
      float* o = bgra + y * (size_t)w * 4;
      const float* row = b + y * (size_t)w * 3;
      for (uint32_t x = 0; x < w; ++x) {
        uint32_t sx = x + (uint32_t)shift;
        if (sx >= w) sx -= w;
        const float* p = row + (size_t)sx * 3;
        float n[3] = {0, 0, 0};
        if (sigma > 0) {
          for (int c = 0; c < 3; ++c) {
            // sum of four 16-bit uniforms: close to Gaussian, variance 1/3
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            uint32_t a = s;
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            float t = (float)((a & 0xffff) + (a >> 16) + (s & 0xffff) + (s >> 16)) * (1.0f / 65535.0f) - 2.0f;
            n[c] = t * 1.7320508f * sigma;
          }
        }
        o[x * 4 + 0] = p[2] + n[2];
        o[x * 4 + 1] = p[1] + n[1];
        o[x * 4 + 2] = p[0] + n[0];
        o[x * 4 + 3] = 1.0f;
      }
    });
    return true;
  }
};

struct RawSource : Source {
  FILE* f = nullptr;
  int start;
  RawSource(const std::string& path, uint32_t w, uint32_t h, int startFrame) : start(startFrame) {
    width = w;
    height = h;
    f = fopen(path.c_str(), "rb");
    if (!f) die("cannot open %s", path.c_str());
    fseeko(f, 0, SEEK_END);
    off_t size = ftello(f);
    count = (int)(size / ((off_t)w * h * 16)) - start;
    if (count < 1) die("%s holds no %ux%u frame after --start", path.c_str(), w, h);
  }
  ~RawSource() override {
    if (f) fclose(f);
  }
  bool seekable() const override { return true; }
  bool read(int index, float* bgra) override {
    if (index < 0 || index >= count) return false;
    const size_t bytes = (size_t)width * height * 16;
    if (fseeko(f, (off_t)(index + start) * (off_t)bytes, SEEK_SET) != 0) return false;
    return fread(bgra, 1, bytes, f) == bytes;
  }
};

// Size of the frames ffmpeg delivers for `path` before any scaling. That is
// not the stored size ffprobe reports as the stream's width and height:
// ffmpeg turns the frames by the file's display matrix (a portrait phone clip
// is stored 3840x2160 with a quarter turn and comes out 2160x3840; a half
// turn or a flip keeps the size) and applies container cropping. So one
// frame is decoded here with the same input options as the pipe and written
// as a PAM image, whose header holds the size.
bool deliveredSize(const std::string& path, uint32_t& w, uint32_t& h) {
  const std::string cmd = std::string(kFfmpeg) + " -v error -nostdin -i " + shellQuote(path) +
                          " -an -frames:v 1 -f image2pipe -c:v pam -pix_fmt gray -";
  FILE* p = popen(cmd.c_str(), "r");
  if (!p) return false;
  w = h = 0;
  bool header = false;
  char line[256];
  if (fgets(line, sizeof(line), p) && strncmp(line, "P7", 2) == 0) {
    while (fgets(line, sizeof(line), p)) {
      if (strncmp(line, "ENDHDR", 6) == 0) {
        header = true;
        break;
      }
      sscanf(line, "WIDTH %u", &w);
      sscanf(line, "HEIGHT %u", &h);
    }
  }
  char sink[65536];  // the pixels, so that ffmpeg finishes writing
  while (fread(sink, 1, sizeof(sink), p) > 0) {
  }
  return pclose(p) == 0 && header && w && h;
}

// The Y'CbCr matrix the decode converts with (--video-matrix), as the name of ffmpeg's scale
// in_color_matrix, or "" to leave the choice to ffmpeg. ffmpeg converts a tagged stream with
// its tag and an untagged one as BT.601 whatever its size. Premiere reads untagged HD and
// UHD camera files as BT.709, so `auto` does too: a stream whose colour space is untagged
// and whose stored size is HD or more (width >= 1280 or height >= 720) is read as BT.709; a
// tagged stream, or a smaller one, as ffmpeg reads it. `ffmpeg` is ffmpeg's own reading
// (the explicit decoder-default mode); bt709 and bt601 force a matrix on every stream. *why
// says which case applied (for the report).
std::string decideMatrix(const std::string& path, const std::string& opt, std::string* why) {
  if (opt == "bt709" || opt == "bt601") {
    *why = "forced";
    return opt;
  }
  if (opt == "ffmpeg") {
    *why = "ffmpeg";
    return "";
  }
  const std::string cmd = std::string(kFfprobe) +
                          " -v error -select_streams v:0 -show_entries stream=width,height,color_space"
                          " -of default=noprint_wrappers=1 " + shellQuote(path);
  FILE* p = popen(cmd.c_str(), "r");
  if (!p) die("could not start ffprobe");
  unsigned w = 0, h = 0;
  char space[64] = "";
  char line[256];
  while (fgets(line, sizeof(line), p)) {
    sscanf(line, "width=%u", &w);
    sscanf(line, "height=%u", &h);
    sscanf(line, "color_space=%63s", space);
  }
  if (pclose(p) != 0 || !w || !h) die("ffprobe could not read the colour space of %s", path.c_str());
  const std::string cs = space;
  if (!cs.empty() && cs != "unknown" && cs != "unspecified" && cs != "reserved") {
    *why = "tag " + cs;
    return "";
  }
  if (w >= 1280 || h >= 720) {
    *why = "untagged HD";
    return "bt709";
  }
  *why = "untagged SD";
  return "";
}

struct VideoSource : Source {
  FILE* pipe = nullptr;
  int next = 0;
  std::vector<uint16_t> tmp;
  // w = 0: the size ffmpeg delivers (display orientation); otherwise frames
  // are scaled to w x h after ffmpeg has turned them.
  VideoSource(const std::string& path, uint32_t w, uint32_t h, int startFrame, const std::string& matrix) {
    if (access(kFfmpeg, X_OK) != 0) die("%s not found", kFfmpeg);
    if (!w && !deliveredSize(path, w, h)) die("ffmpeg could not decode a frame of %s to read its size", path.c_str());
    width = w;
    height = h;
    char vf[160];
    snprintf(vf, sizeof(vf), "select=gte(n\\,%d),scale=%u:%u:flags=bicubic%s%s", startFrame, w, h,
             matrix.empty() ? "" : ":in_color_matrix=", matrix.c_str());
    // ffmpeg's autorotation (on by default) runs before -vf, so the scale
    // above sees the frames in display orientation.
    std::string cmd = std::string(kFfmpeg) + " -v error -nostdin -i " + shellQuote(path) + " -vf " + shellQuote(vf) +
                      " -fps_mode passthrough -f rawvideo -pix_fmt rgba64le -";
    pipe = popen(cmd.c_str(), "r");
    if (!pipe) die("could not start ffmpeg");
    tmp.resize((size_t)w * h * 4);
  }
  ~VideoSource() override {
    if (pipe) pclose(pipe);
  }
  bool seekable() const override { return false; }
  bool read(int index, float* bgra) override {
    if (index != next) die("internal: video source read out of order (%d, expected %d)", index, next);
    const size_t n = (size_t)width * height * 4;
    if (fread(tmp.data(), 2, n, pipe) != n) return false;
    ++next;
    const uint16_t* s = tmp.data();
    const uint32_t w = width;
    dispatch_apply(height, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t y) {
      const uint16_t* r = s + y * (size_t)w * 4;
      float* o = bgra + y * (size_t)w * 4;
      for (uint32_t x = 0; x < w; ++x) {
        o[x * 4 + 0] = (float)r[x * 4 + 2] * (1.0f / 65535.0f);
        o[x * 4 + 1] = (float)r[x * 4 + 1] * (1.0f / 65535.0f);
        o[x * 4 + 2] = (float)r[x * 4 + 0] * (1.0f / 65535.0f);
        o[x * 4 + 3] = (float)r[x * 4 + 3] * (1.0f / 65535.0f);
      }
    });
    return true;
  }
};

// ---------------------------------------------------------------------------
// Host buffers
// ---------------------------------------------------------------------------
struct HostBuffer {
  id<MTLBuffer> buffer = nil;
  int index = INT32_MIN;  // source frame held, INT32_MIN = free
  // --async: an empty command buffer committed on the queue after the last
  // render that read or wrote this buffer. The queue completes command
  // buffers in order, so once it has completed that render has too.
  id<MTLCommandBuffer> fence = nil;
  uint64_t lastUse = 0;  // render that last used it (0 = never): the ring reuses the oldest
  uint64_t lastRender = 0;  // ODStats.renderId of that render (0 = none), for the hazard count
};

void packToHost(const float* bgra, id<MTLBuffer> buf, uint32_t w, uint32_t h, uint32_t rowBytes, ODPixelFormat fmt) {
  uint8_t* dst = (uint8_t*)buf.contents;
  dispatch_apply(h, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t y) {
    const float* s = bgra + y * (size_t)w * 4;
    uint8_t* d = dst + y * (size_t)rowBytes;
    if (fmt == OD_PIXEL_BGRA32F) {
      memcpy(d, s, (size_t)w * 16);
    } else {
      __fp16* hd = (__fp16*)d;
      for (uint32_t i = 0; i < w * 4; ++i) hd[i] = (__fp16)s[i];
    }
  });
}

void unpackFromHost(id<MTLBuffer> buf, float* bgra, uint32_t w, uint32_t h, uint32_t rowBytes, ODPixelFormat fmt) {
  const uint8_t* src = (const uint8_t*)buf.contents;
  dispatch_apply(h, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t y) {
    const uint8_t* s = src + y * (size_t)rowBytes;
    float* d = bgra + y * (size_t)w * 4;
    if (fmt == OD_PIXEL_BGRA32F) {
      memcpy(d, s, (size_t)w * 16);
    } else {
      const __fp16* hs = (const __fp16*)s;
      for (uint32_t i = 0; i < w * 4; ++i) d[i] = (float)hs[i];
    }
  });
}

struct Summary {
  double mean = 0, median = 0, p95 = 0, min = 0, max = 0;
};

Summary summarise(std::vector<double> v) {
  Summary s;
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  double sum = 0;
  for (double x : v) sum += x;
  s.mean = sum / (double)v.size();
  s.median = v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
  size_t i95 = (size_t)std::ceil(0.95 * (double)v.size());
  s.p95 = v[std::min(v.size() - 1, i95 > 0 ? i95 - 1 : 0)];
  s.min = v.front();
  s.max = v.back();
  return s;
}

std::string jsonEscape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') {
      o += '\\';
      o += c;
    } else if ((unsigned char)c < 0x20) {
      char b[8];
      snprintf(b, sizeof(b), "\\u%04x", c);
      o += b;
    } else {
      o += c;
    }
  }
  return o;
}

// Physical footprint of this process (what Activity Monitor calls Memory),
// GPU allocations in shared storage included.
uint64_t physFootprint() {
  task_vm_info_data_t info;
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) return 0;
  return info.phys_footprint;
}

struct MemSample {
  double ms = 0;  // since the end of the run (idle samples)
  uint64_t footprint = 0, device = 0, cacheBytes = 0;
  int cacheEntries = 0;
};

struct FrameRecord {
  ODQualityDecision decision = {};
  double callToFinishMs = 0, startLateMs = 0;
  int index;
  int session;
  double gpuMs, wallMs, cpuMs, waitMs, queueMs, latencyMs;
  int cacheHits, framesUsed;
  uint64_t profileHash;
  double stageMs[OD_STAGE_COUNT];
};

}  // namespace

// Analysis output is evidence, never an input to denoising.
static void printAnalysis(FILE* f, const ODAnalysis& a, const ODParams& p) {
  char row1[32],row2[32]; od_analysis_status(&a,&p,row1,row2);
  fprintf(f,"{\"renderId\":%llu,\"frameKey\":%lld,\"valid\":%u,\"state\":%u,\"learning\":%u,\"confidence\":%.6g,\"flatTiles\":%u,\"temporalValid\":%u,\"temporalSpatialRatio\":%.6g,\"removalValid\":%u,\"row1\":\"%s\",\"row2\":\"%s\"",
    (unsigned long long)a.renderId,(long long)a.frameKey,a.valid,a.state,a.learningFrames,a.confidence,a.flatTiles,a.temporalValid,a.temporalSpatialRatio,a.removalValid,row1,row2);
  auto array=[&](const char* key,const float* v,int n){fprintf(f,",\"%s\":[",key);for(int i=0;i<n;++i)fprintf(f,"%s%.8g",i?",":"",v[i]);fprintf(f,"]");};
  array("profile",&a.profile[0][0][0],96);array("full",&a.full[0][0],24);array("histogram",a.histogram,8);
  array("effectiveLuma",a.effectiveLuma,8);array("removedDb",&a.removedDb[0][0],12);array("removedOverallDb",a.removedOverallDb,3);
  array("inputEnergy",&a.inputEnergy[0][0],12);array("outputEnergy",&a.outputEnergy[0][0],12);
  fprintf(f,",\"measured\":[");for(int i=0;i<8;++i)fprintf(f,"%s%u",i?",":"",a.measured[i]);fprintf(f,"]}\n");fflush(f);
}
static void analysisPNG(const std::string& path,const float* bgra,uint32_t w,uint32_t h) {
  std::vector<uint8_t> rgba((size_t)w*h*4);
  for(size_t i=0;i<(size_t)w*h;++i)for(int c=0;c<4;++c)rgba[i*4+c]=c==3?255:(uint8_t)std::lround(std::clamp(bgra[i*4+2-c],0.f,1.f)*255);
  CGColorSpaceRef cs=CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  CGContextRef cg=CGBitmapContextCreate(rgba.data(),w,h,8,w*4,cs,kCGImageAlphaNoneSkipLast|kCGBitmapByteOrder32Big);
  CGImageRef image=CGBitmapContextCreateImage(cg);
  CFURLRef url=CFURLCreateFromFileSystemRepresentation(nullptr,(const UInt8*)path.c_str(),path.size(),false);
  CGImageDestinationRef dest=CGImageDestinationCreateWithURL(url,CFSTR("public.png"),1,nullptr);
  if(!dest||!image)die("cannot create analysis PNG");CGImageDestinationAddImage(dest,image,nullptr);
  if(!CGImageDestinationFinalize(dest))die("cannot save analysis PNG");
  CFRelease(dest);CFRelease(url);CGImageRelease(image);CGContextRelease(cg);CGColorSpaceRelease(cs);
}

int main(int argc, char** argv) {
  @autoreleasepool {
    Options o = parse(argc, argv);
    alarm((unsigned)o.deadline);  // hard stop: SIGALRM terminates the process

    std::string cmdline;
    for (int i = 0; i < argc; ++i) cmdline += (i ? " " : "") + std::string(argv[i]);

    // Text report goes to stderr when JSON goes to stdout.
    FILE* txt = o.json == "-" ? stderr : stdout;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) die("no Metal device");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandQueue> fenceQueue = o.fenceElsewhere ? [device newCommandQueue] : nil;
    char err[2048] = {0};
    ODContext* ctx = od_context_create((__bridge void*)device, (__bridge void*)queue, err, sizeof(err));
    if (!ctx) die("od_context_create: %s", err);
    std::vector<ODSession*> sessions((size_t)o.sessions, nullptr);
    const char* originalTune = getenv("OD_SPATIAL_TUNE");
    const bool hadTune = originalTune != nullptr;
    const std::string savedTune = originalTune ? originalTune : "";
    if (!o.pairTune.empty() && (o.sessions != 2 || o.randomAccess || o.async))
      die("--pair-tune requires --sessions 2, sequential blocking");
    int si = 0;
    for (ODSession*& s : sessions) {
      if (!o.pairTune.empty()) {
        if (si == 0) unsetenv("OD_SPATIAL_TUNE");
        else setenv("OD_SPATIAL_TUNE", o.pairTune.c_str(), 1);
      }
      ++si;
      s = od_session_create(ctx);
      if (!s) die("od_session_create failed");
    }
    if (!o.pairTune.empty()) {
      if (hadTune) setenv("OD_SPATIAL_TUNE", savedTune.c_str(), 1);
      else unsetenv("OD_SPATIAL_TUNE");
    }

    // ---- source ----
    std::unique_ptr<Source> src;
    const char* srcName = "synthetic";
    std::string matrixUsed, matrixWhy;  // --video only (decideMatrix)
    if (!o.video.empty()) {
      matrixUsed = decideMatrix(o.video, o.videoMatrix, &matrixWhy);
      src.reset(new VideoSource(o.video, o.width, o.height, o.start, matrixUsed));
      srcName = "video";
    } else if (!o.raw.empty()) {
      src.reset(new RawSource(o.raw, o.width, o.height, o.start));
      srcName = "raw";
    } else {
      if (!o.width) {
        o.width = 1920;
        o.height = 1080;
      }
      src.reset(new SyntheticSource(o.width, o.height, o.frames, o.noise, o.motion, o.seed));
    }
    const uint32_t W = src->width, H = src->height;
    const uint32_t bpp = o.format == OD_PIXEL_BGRA16F ? 8 : 16;
    const uint32_t rowBytes = W * bpp + o.rowPad;
    const size_t hostBytes = (size_t)rowBytes * H;
    const size_t stagingBytes = (size_t)W * H * 16;
    const int R = o.params.temporalNoiseModel == 1 ? std::max(o.params.temporalRadius, OD_PROFILE_RADIUS) : o.params.temporalRadius;
    int total = o.frames;
    if (src->count >= 0 && src->count < total) total = src->count;

    const bool writesOutput = !o.analysisPNG.empty() || !o.outRaw.empty() || !o.outVideo.empty();
    // --async: od_render returns before the GPU has read the window's host
    // buffers or written the output. Whenever the CPU writes or reads such a
    // buffer again (a ring slot refilled, the in-place scratch, an output read
    // back), it first waits for the render that last used it: each render is
    // followed by a fence (HostBuffer::fence). `depth` spare buffers in each
    // ring keep those waits for buffers `depth` renders old. Nothing is reused
    // with preloaded frames, out of place and no output written (the benchmark
    // matrix's case): no fences, as before.
    const bool tracked = o.async && (!o.preload || o.inPlace || writesOutput);
    const int depth = o.async ? o.inFlight : 0;
    const int spare = o.unfenced ? 0 : depth;
    const int ringSize = o.preload ? std::max(total, 2 * R + 1) : 2 * R + 1 + spare;
    const int nScratch = o.inPlace ? 1 + spare : 0;
    const int nOut = o.inPlace ? 0 : 1 + (writesOutput ? depth : 0);
    const int nHost = ringSize + nScratch + nOut;
    if (o.preload && (double)hostBytes * nHost > 24.0 * 1073741824.0)
      die("--preload would hold %.1f GB of frames (limit 24 GB): use fewer --frames",
          (double)hostBytes * nHost / 1073741824.0);
    const double memMB = ((double)hostBytes * nHost + (double)stagingBytes * 2 +
                          (o.video.empty() && o.raw.empty() ? (double)W * H * 12 : 0) +
                          (!o.video.empty() ? (double)W * H * 8 : 0)) / 1048576.0;
    fprintf(txt, "odbench: %s %ux%u %s, %d frames (warm-up %d), radius %d, device %s, Denoise %s\n", srcName, W, H,
            o.format == OD_PIXEL_BGRA16F ? "16f" : "32f", total, o.warmup, R, device.name.UTF8String,
            od_version());
    if (!o.video.empty())
      fprintf(txt, "  video decode: %s (%s)\n", matrixUsed.empty() ? "ffmpeg's matrix" : matrixUsed.c_str(),
              matrixWhy.c_str());
    fprintf(txt, "odbench: host memory held by the harness: %.0f MB (%d window frames + %d out + %d scratch + staging)\n",
            memMB, ringSize, nOut, nScratch);
    if (o.sessions > 1 || o.intervalMs > 0)
      fprintf(txt, "odbench: %d session(s), %s, %s; stops above a %.1f GB process footprint\n", o.sessions,
              o.sessions > 1 ? (o.randomAccess ? "random session and frame per render" : "each frame on every session")
                             : (o.randomAccess ? "random access" : "sequential"),
              o.intervalMs > 0 ? "paced" : "back to back", o.memLimitGB);

    // ring: the window's source frames; outRing: outputs (out of place);
    // scratchRing: the in-place copies of the current frame, which the render
    // overwrites with its output.
    std::vector<HostBuffer> ring((size_t)ringSize), outRing((size_t)nOut), scratchRing((size_t)nScratch);
    for (std::vector<HostBuffer>* v : {&ring, &outRing, &scratchRing}) {
      for (HostBuffer& hb : *v) {
        hb.buffer = [device newBufferWithLength:hostBytes options:MTLResourceStorageModeShared];
        if (!hb.buffer) die("could not allocate a host buffer");
      }
    }
    std::vector<float> staging((size_t)W * H * 4);

    // ---- outputs ----
    if (o.freezeProfile >= 0 && (o.async || o.sessions != 1)) die("--freeze-profile requires blocking single session");
    ODNoiseProfile frozenProfile = {};
    FILE* profileLog = o.profileLog.empty() ? nullptr : fopen(o.profileLog.c_str(), "w");
    if (!o.profileLog.empty() && !profileLog) die("cannot open profile log");
    FILE* analysisLog = o.analysisLog.empty() ? nullptr : fopen(o.analysisLog.c_str(), "w");
    if (!o.analysisLog.empty() && !analysisLog) die("cannot open analysis log");
    FILE* rawOut = nullptr;
    FILE* ffOut = nullptr;
    std::vector<uint16_t> out16;
    if (!o.outRaw.empty()) {
      rawOut = fopen(o.outRaw.c_str(), "wb");
      if (!rawOut) die("cannot write %s", o.outRaw.c_str());
    }
    if (!o.outVideo.empty()) {
      if (o.randomAccess) die("--out needs frames in order; use --out-raw with --random-access");
      const std::string& p = o.outVideo;
      auto endsWith = [&](const char* e) { return p.size() >= strlen(e) && p.compare(p.size() - strlen(e), strlen(e), e) == 0; };
      char in[256];
      snprintf(in, sizeof(in), " -v error -nostdin -y -f rawvideo -pix_fmt rgba64le -s %ux%u -r %g -i - ", W, H, o.fps);
      std::string codec;
      if (endsWith(".mov")) {
        codec = "-vf scale=out_color_matrix=bt709:out_range=tv -c:v prores_ks -profile:v 4 -pix_fmt yuv444p10le "
                "-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv ";
        if (o.proresStandard)
          codec = "-vf scale=out_color_matrix=bt709:out_range=tv -c:v prores_videotoolbox -profile:v standard -pix_fmt p210le "
                  "-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv ";
      } else if (endsWith(".png")) {
        codec = "-pix_fmt rgb48be ";
      }
      std::string cmd = std::string(kFfmpeg) + in + codec + shellQuote(p);
      ffOut = popen(cmd.c_str(), "w");
      if (!ffOut) die("could not start ffmpeg for %s", p.c_str());
      out16.resize((size_t)W * H * 4);
    }

    // ---- params ----
    ODNoiseProfile manual;
    ODParams params = o.params;
    params.waitUntilCompleted = o.async ? 0 : 1;
    if (o.stageTimings) params.debugFlags |= OD_DEBUG_STAGE_TIMINGS;
    if (o.manual) {
      memset(&manual, 0, sizeof(manual));
      // White noise of sigma S on R, G and B: its sigma in Y, Cb, Cr (the working-space
      // matrix in core/shaders/common.metal), times the share of white noise each a-trous
      // band holds. sigmaFull is the sigma of c_0 - c_4, 0.9987 of the pixel sigma; the
      // bands are correlated, so it is not the root sum of squares of the band gains
      // (0.918).
      const float chan[3] = {0.7497f, 0.6416f, 0.6770f};
      const float bandGain[OD_BANDS] = {0.8908f, 0.2007f, 0.0855f, 0.0412f};
      const float fullGain = 0.9987f;
      for (int c = 0; c < 3; ++c) {
        for (int b = 0; b < OD_BANDS; ++b)
          for (int l = 0; l < OD_LEVELS; ++l) manual.sigma[c][b][l] = o.manualSigma * chan[c] * bandGain[b];
        for (int l = 0; l < OD_LEVELS; ++l) manual.sigmaFull[c][l] = o.manualSigma * chan[c] * fullGain;
      }
      manual.confidence = 1;
      manual.valid = 1;
      params.manualProfile = &manual;
    }

    if (!o.profileLoad.empty()) {
      std::vector<uint8_t> data(od_profile_blob_size());
      FILE* f=fopen(o.profileLoad.c_str(),"rb");
      if(!f) die("profile-load failed");
      bool ok=fread(data.data(),1,data.size(),f)==data.size() && fgetc(f)==EOF;
      fclose(f);
      if(!ok || od_profile_decode(data.data(),data.size(),&manual)) die("invalid profile payload");
      params.manualProfile=&manual;
    }
    bool profileSaved=false;
    // ---- render order: (frame, session) ----
    std::vector<int> order, orderSession;
    uint32_t rng = hash32(o.seed + 12345u) | 1u;
    auto nextRandom = [&]() {
      rng ^= rng << 13;
      rng ^= rng >> 17;
      rng ^= rng << 5;
      return rng;
    };
    if (o.sessions == 1) {
      order.resize((size_t)total);
      for (int i = 0; i < total; ++i) order[(size_t)i] = i;
      if (o.randomAccess) {
        for (int i = total - 1; i > 0; --i) std::swap(order[(size_t)i], order[(size_t)(nextRandom() % (uint32_t)(i + 1))]);
      }
      orderSession.assign(order.size(), 0);
    } else if (!o.randomAccess) {
      for (int i = 0; i < total; ++i) {
        for (int k = 0; k < o.sessions; ++k) {
          order.push_back(i);
          orderSession.push_back(o.analysisAB && (i & 1) ? 1-k : k);
        }
      }
    } else {
      const int n = o.renders > 0 ? o.renders : total;
      for (int i = 0; i < n; ++i) {
        order.push_back((int)(nextRandom() % (uint32_t)total));
        orderSession.push_back((int)(nextRandom() % (uint32_t)o.sessions));
      }
    }
    if (o.renders > 0 && !order.empty() && (int)order.size() != o.renders) {
      // Cycle or truncate to the requested number of renders.
      std::vector<int> f, k;
      for (int i = 0; i < o.renders; ++i) {
        f.push_back(order[(size_t)i % order.size()]);
        k.push_back(orderSession[(size_t)i % order.size()]);
      }
      order.swap(f);
      orderSession.swap(k);
    }
    if (!o.outVideo.empty() && o.renders > 0) die("--out needs one render per frame, in order");

    // The queue held at the start (--hold-gpu-ms): an empty command buffer that
    // waits for an event, signalled when odbench first waits for the GPU, at
    // the end of the run, or after holdGpuMs at the latest. Every render queues
    // behind it, so the CPU runs ahead of the GPU as far as odbench lets it.
    id<MTLSharedEvent> gate = nil;
    bool gateOpen = true;
    auto openGate = [&]() {
      if (!gateOpen) {
        gate.signaledValue = 1;
        gateOpen = true;
      }
    };
    // Hazards: CPU writes into a host buffer before the GPU work of the render that last
    // used it had ended. Judged against that render's own completion, never against the
    // fences: od_render reports each render's completion once (ODStats.completed, API v7),
    // with its latency from commit to the end of its GPU work. The commit happened inside
    // the od_render call, after `start` (host seconds before the call), so the GPU work
    // ended at or after start + latency: a write before that is a hazard for certain. (A
    // write between that bound and the real end is not seen: the window is the od_render
    // call's CPU time before its commit.) A write whose render has not reported yet waits in
    // `unresolved`; the end of the run takes the completions no render handed over
    // (od_debug_take_completions), and what stays unresolved (completions lost) is reported
    // apart.
    long hazards = 0;
    long bufferWaits = 0;  // waits on a host buffer's fence before writing into it
    double bufferWaitMs = 0;
    std::map<uint64_t, double> renderStart;    // renderId -> host seconds before its od_render call
    std::map<uint64_t, double> renderEndedBy;  // renderId -> its GPU work ended at or after this
    struct WriteCheck {
      uint64_t render;
      double at;
    };
    std::vector<WriteCheck> unresolved;
    auto judge = [&](uint64_t render, double at) {
      auto it = renderEndedBy.find(render);
      if (it == renderEndedBy.end()) {
        unresolved.push_back({render, at});
      } else if (at < it->second) {
        ++hazards;
      }
    };
    auto noteCompletions = [&](const ODCompletion* c, int n) {
      for (int i = 0; i < n; ++i) {
        auto it = renderStart.find(c[i].renderId);
        if (it == renderStart.end()) continue;
        renderEndedBy[c[i].renderId] = it->second + c[i].latencyMs / 1000.0;
        renderStart.erase(it);
      }
      if (n == 0 || unresolved.empty()) return;
      std::vector<WriteCheck> still;
      for (const WriteCheck& w : unresolved) {
        if (renderEndedBy.count(w.render)) judge(w.render, w.at);
        else still.push_back(w);
      }
      unresolved.swap(still);
    };
    auto waitFor = [&](id<MTLCommandBuffer> cb) {
      if (!cb || cb.status >= MTLCommandBufferStatusCompleted) return false;
      openGate();
      [cb waitUntilCompleted];
      return true;
    };
    // Before the CPU writes into a host buffer the GPU may still be using:
    // wait for its fence (--debug-unfenced: do not), then count a hazard if
    // the render that used it is still not done.
    auto beforeWrite = [&](HostBuffer& hb) {
      if (!o.unfenced) {
        const double t = nowMs();
        if (waitFor(hb.fence)) {
          ++bufferWaits;
          bufferWaitMs += nowMs() - t;
        }
      }
      if (hb.lastRender) judge(hb.lastRender, hostSeconds());
      hb.fence = nil;
    };

    // Loads source frame `index` into the ring unless it is there already.
    // Buffers holding frames outside [lo, hi] are the ones reused.
    auto ensure = [&](int index, int lo, int hi) -> HostBuffer* {
      if (index < 0 || index >= total) return nullptr;
      HostBuffer* freeSlot = nullptr;
      for (HostBuffer& hb : ring) {
        if (hb.index == index) return &hb;
        // The free slot used longest ago: with --async its render is the
        // likeliest to be done.
        if ((hb.index < lo || hb.index > hi) && (!freeSlot || hb.lastUse < freeSlot->lastUse)) freeSlot = &hb;
      }
      if (!freeSlot) die("internal: no free host buffer");
      if (!src->read(index, staging.data())) {
        total = index;  // end of input reached earlier than --frames
        freeSlot->index = INT32_MIN;
        return nullptr;
      }
      beforeWrite(*freeSlot);
      packToHost(staging.data(), freeSlot->buffer, W, H, rowBytes, o.format);
      freeSlot->index = index;
      return freeSlot;
    };

    if (o.preload) {
      const int want = total;
      for (int i = 0; i < want && i < total; ++i) ensure(i, 0, want - 1);
    }

    // Outputs are read back in render order. Blocking: right after the render.
    // --async: when their buffer is needed again or at the end, after their
    // fence, so writing the output does not serialise the queue.
    struct Pending {
      HostBuffer* slot;
      int index;
    };
    std::deque<Pending> pending;
    auto readBack = [&](const Pending& pd) {
      waitFor(pd.slot->fence);
      unpackFromHost(pd.slot->buffer, staging.data(), W, H, rowBytes, o.format);
      if (!o.analysisPNG.empty()) analysisPNG(o.analysisPNG, staging.data(), W, H);
      if (rawOut) {
        if (o.randomAccess) fseeko(rawOut, (off_t)pd.index * (off_t)stagingBytes, SEEK_SET);
        if (fwrite(staging.data(), 1, stagingBytes, rawOut) != stagingBytes) die("write to %s failed", o.outRaw.c_str());
      }
      if (ffOut) {
        const float* s = staging.data();
        uint16_t* d = out16.data();
        dispatch_apply(H, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t y) {
          const float* sr = s + y * (size_t)W * 4;
          uint16_t* dr = d + y * (size_t)W * 4;
          for (uint32_t x = 0; x < W; ++x) {
            const int map[4] = {2, 1, 0, 3};
            for (int c = 0; c < 4; ++c) {
              float v = sr[x * 4 + map[c]];
              v = v < 0 ? 0 : (v > 1 ? 1 : v);
              dr[x * 4 + c] = (uint16_t)lrintf(v * 65535.0f);
            }
          }
        });
        if (fwrite(d, 2, out16.size(), ffOut) != out16.size()) die("pipe to ffmpeg failed");
      }
    };
    // Reads back every pending output up to the one in `slot` (all if null).
    auto flushPending = [&](HostBuffer* slot) {
      if (slot && std::none_of(pending.begin(), pending.end(), [&](const Pending& p) { return p.slot == slot; })) return;
      while (!pending.empty()) {
        const Pending pd = pending.front();
        pending.pop_front();
        readBack(pd);
        if (pd.slot == slot) break;
      }
    };
    int outTurn = 0;  // which output / scratch buffer the next render uses

    std::vector<FrameRecord> records;
    struct Completion { int session; ODQualityCompletion c; };
    std::vector<Completion> completions;
    auto collectQuality = [&](ODSession* session, int si) {
      ODQualityCompletion batch[256]; uint64_t dropped = 0;
      const int n = od_quality_poll(session, batch, 256, &dropped);
      if (dropped) die("quality completion telemetry overflow");
      for (int i = 0; i < n; ++i) {
        if (batch[i].failed) die("GPU quality completion failed");
        completions.push_back({si, batch[i]});
      }
    };
    records.reserve(order.size());
    double tRenderTotal = 0;
    int rendered = 0;
    // Memory: baseline once the harness holds its frames, before any render.
    auto sampleMemory = [&](double ms) {
      MemSample m;
      m.ms = ms;
      m.footprint = physFootprint();
      m.device = (uint64_t)device.currentAllocatedSize;
      for (ODSession* s : sessions) {
        int e = 0;
        uint64_t b = 0;
        od_debug_cache_info(s, &e, &b, nullptr, nullptr, nullptr);
        m.cacheEntries += e;
        m.cacheBytes += b;
      }
      return m;
    };
    const MemSample mem0 = sampleMemory(0);
    MemSample memPeak = mem0, memEnd = mem0;
    std::vector<MemSample> memIdle;
    bool memStopped = false;
    const uint64_t memLimit = (uint64_t)(o.memLimitGB * 1073741824.0);
    double nextTick = 0;
    if (o.holdGpuMs > 0) {
      gate = [device newSharedEvent];
      id<MTLCommandBuffer> hold = [queue commandBuffer];
      [hold encodeWaitForEvent:gate value:1];
      [hold commit];
      gateOpen = false;
      id<MTLSharedEvent> g = gate;
      dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)o.holdGpuMs * 1000000),
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                       g.signaledValue = 1;
                     });
    }
    const double tAll0 = nowMs();

    for (int n = 0; n < (int)order.size() && !memStopped; ++n) {
      @autoreleasepool {
        const int idx = order[(size_t)n];
        ODSession* session = sessions[(size_t)orderSession[(size_t)n]];
        if (idx >= total) continue;
        const int lo = idx - R, hi = idx + R;
        ODFrame frames[OD_MAX_FRAMES];
        int offsets[OD_MAX_FRAMES];
        int count = 0;
        // Sequential sources must be read in increasing order: fill the
        // window from the oldest frame up, then put the current frame first.
        HostBuffer* held[OD_MAX_FRAMES];
        int heldOff[OD_MAX_FRAMES];
        int nh = 0;
        for (int j = lo; j <= hi; ++j) {
          HostBuffer* hb = ensure(j, lo, hi);
          if (!hb) continue;
          held[nh] = hb;
          heldOff[nh++] = j - idx;
        }
        if (idx >= total) continue;  // the current frame itself hit end of input
        auto addFrame = [&](id<MTLBuffer> b, int off, int srcIndex) {
          ODFrame f;
          f.mtlBuffer = (__bridge void*)b;
          f.width = W;
          f.height = H;
          f.rowBytes = rowBytes;
          f.format = o.format;
          f.frameKey = o.noCache ? OD_NO_KEY : (int64_t)srcIndex * 1000;
          frames[count] = f;
          offsets[count++] = off;
        };
        id<MTLBuffer> curBuf = nil;
        for (int k = 0; k < nh; ++k) {
          if (heldOff[k] == 0) curBuf = held[k]->buffer;
        }
        if (!curBuf) continue;
        // Where the output goes: the next scratch (in place) or output buffer,
        // once its previous output has been read back.
        HostBuffer* target = o.inPlace ? &scratchRing[(size_t)(outTurn % nScratch)] : &outRing[(size_t)(outTurn % nOut)];
        ++outTurn;
        flushPending(target);
        if (o.inPlace) {
          beforeWrite(*target);
          memcpy(target->buffer.contents, curBuf.contents, hostBytes);
          curBuf = target->buffer;
        }
        addFrame(curBuf, 0, idx);
        for (int k = 0; k < nh; ++k) {
          if (heldOff[k] != 0) addFrame(held[k]->buffer, heldOff[k], idx + heldOff[k]);
        }
        ODFrame out = frames[0];
        out.mtlBuffer = (__bridge void*)target->buffer;
        out.frameKey = OD_NO_KEY;

        // An absolute clock retains late starts instead of hiding overload.
        double scheduled = nowMs();
        if (o.intervalMs > 0) {
          if (nextTick == 0) nextTick = scheduled;
          scheduled = nextTick;
          const double delay = scheduled - nowMs();
          if (delay > 0) usleep((useconds_t)(delay * 1000.0));
          nextTick += o.intervalMs;
        }
        ODQualityRequest qr = {o.autoQuality, o.intervalMs,
                              idx * (o.intervalMs > 0 ? o.intervalMs : 1000.0 / o.fps), 0};
        ODQualityDecision qd = {};
        if (o.analysisAB) { if (session == sessions[0]) params.debugFlags |= OD_DEBUG_NO_ANALYSIS; else params.debugFlags &= ~OD_DEBUG_NO_ANALYSIS; }
        ODStats st;
        const double startHost = hostSeconds();
        if (frozenProfile.valid) params.manualProfile = &frozenProfile;
        const double t0 = nowMs();
        if (!o.profileSave.empty() && !profileSaved) {
          if(od_profile_frame(session,&frames[0],&manual)) die("capture failed: %s",od_last_error(session));
          std::vector<uint8_t> data(od_profile_blob_size());
          if(od_profile_encode(&manual,data.data(),data.size())) die("profile encode failed");
          FILE* f=fopen(o.profileSave.c_str(),"wb");
          if(!f) die("profile-save failed");
          bool ok=fwrite(data.data(),1,data.size(),f)==data.size();
          ok=fclose(f)==0 && ok;
          if(!ok) die("profile-save incomplete");
          params.manualProfile=&manual; profileSaved=true;
        }
        int rc = o.deterministicProfile ? od_render_deterministic(session, frames, offsets, count, &params, &out, &st)
            : od_render_quality(session, frames, offsets, count, &params, &out, &st, &qr, &qd);
        const double wall = nowMs() - t0;
        if (rc != 0) die("od_render failed on frame %d: %s", idx, od_last_error(session));
        if (o.freezeProfile >= 0 && n == o.freezeProfile && st.profile.valid) frozenProfile = st.profile;
        if (o.analysisPrint || analysisLog) { ODAnalysis a={}; od_session_last_analysis(session,&a); if(o.analysisPrint)printAnalysis(stdout,a,params); if(analysisLog)printAnalysis(analysisLog,a,params); }
        if (profileLog) {
          fprintf(profileLog, "{\"frame\":%d,\"valid\":%d,\"confidence\":%.9g,\"sigma\":[", idx, st.profile.valid, st.profile.confidence);
          for (int c = 0; c < 3; ++c) for (int b = 0; b < OD_BANDS; ++b) for (int l = 0; l < OD_LEVELS; ++l)
            fprintf(profileLog, "%s%.9g", c || b || l ? "," : "", st.profile.sigma[c][b][l]);
          fprintf(profileLog, "],\"full\":[");
          for (int c = 0; c < 3; ++c) for (int l = 0; l < OD_LEVELS; ++l)
            fprintf(profileLog, "%s%.9g", c || l ? "," : "", st.profile.sigmaFull[c][l]);
          fprintf(profileLog, "]}\n"); fflush(profileLog);
        }
        tRenderTotal += wall;
        ++rendered;
        for (int k = 0; k < nh; ++k) held[k]->lastUse = (uint64_t)rendered;
        target->lastUse = (uint64_t)rendered;
        if (tracked) {
          // The hazard count's bookkeeping (only where buffers are reused
          // while renders may be in flight; the benchmark matrix's runs do
          // not have it).
          renderStart[st.renderId] = startHost;
          noteCompletions(st.completed, st.completedCount);
          for (int k = 0; k < nh; ++k) held[k]->lastRender = st.renderId;
          target->lastRender = st.renderId;
        }
        if (tracked && (rendered & 1023) == 0) {
          // Bounded bookkeeping: renders no host buffer and no pending check
          // refers to any more (ids only grow).
          uint64_t oldest = st.renderId;
          for (const std::vector<HostBuffer>* v : {&ring, &outRing, &scratchRing})
            for (const HostBuffer& hb : *v)
              if (hb.lastRender) oldest = std::min(oldest, hb.lastRender);
          for (const WriteCheck& w : unresolved) oldest = std::min(oldest, w.render);
          renderEndedBy.erase(renderEndedBy.begin(), renderEndedBy.lower_bound(oldest));
          renderStart.erase(renderStart.begin(), renderStart.lower_bound(oldest));  // completions lost
        }
        if (tracked) {
          id<MTLCommandBuffer> fence = [(o.fenceElsewhere ? fenceQueue : queue) commandBuffer];
          [fence commit];
          for (int k = 0; k < nh; ++k) held[k]->fence = fence;
          target->fence = fence;
        }
        if (writesOutput) {
          pending.push_back({target, idx});
          if (!o.async) flushPending(nullptr);
        }

        if (o.sessions > 1 || o.idleMs > 0) {
          const MemSample m = sampleMemory(0);
          if (m.footprint > memPeak.footprint) memPeak.footprint = m.footprint;
          if (m.device > memPeak.device) memPeak.device = m.device;
          if (m.cacheBytes > memPeak.cacheBytes) memPeak.cacheBytes = m.cacheBytes;
          if (m.cacheEntries > memPeak.cacheEntries) memPeak.cacheEntries = m.cacheEntries;
          if (m.footprint > memLimit) {
            fprintf(txt, "odbench: process footprint %.2f GB passed the %.1f GB limit after %d renders: run stopped\n",
                    m.footprint / 1073741824.0, o.memLimitGB, n + 1);
            memStopped = true;
          }
        } else if (physFootprint() > memLimit) {
          fprintf(txt, "odbench: process footprint passed the %.1f GB limit after %d renders: run stopped\n",
                  o.memLimitGB, n + 1);
          memStopped = true;
        }

        collectQuality(session, orderSession[(size_t)n]);
        FrameRecord r;
        r.decision = qd; r.startLateMs = std::max(0.0, t0 - scheduled);
        r.index = idx;
        r.session = orderSession[(size_t)n];
        r.gpuMs = st.gpuMs;
        r.wallMs = wall;
        r.cpuMs = st.cpuMs;
        r.waitMs = st.waitMs; r.queueMs = st.gpuQueueMs; r.latencyMs = st.gpuLatencyMs;
        r.cacheHits = st.cacheHits;
        r.framesUsed = st.framesUsed;
        // Fingerprint the profile actually used, including confidence and
        // validity. This distinguishes transport/math changes from profile
        // history when comparing otherwise identical pictures.
        r.profileHash = 14695981039346656037ull;
        const auto* profileBytes = reinterpret_cast<const unsigned char*>(&st.profile);
        for (size_t b = 0; b < sizeof(st.profile); ++b)
          r.profileHash = (r.profileHash ^ profileBytes[b]) * 1099511628211ull;
        memcpy(r.stageMs, st.stageMs, sizeof(r.stageMs));
        records.push_back(r);
        if (o.perFrame) {
          if (o.sessions > 1)
            fprintf(txt, "session %3d  ", r.session);
          fprintf(txt, "frame %5d  gpu %7.3f ms  wall %7.3f ms  cpu %6.3f ms  window %2d  hits %2d  used %2d%s\n", idx,
                  st.gpuMs, wall, st.cpuMs, count, st.cacheHits, st.framesUsed, n < o.warmup ? "  (warm-up)" : "");
        }

      }
    }
    flushPending(nullptr);
    openGate();
    // Drain the queue so asynchronous work is included in the total.
    {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      [cb commit];
      [cb waitUntilCompleted];
    }
    if (!o.deterministicProfile) {
      for (size_t si = 0; si < sessions.size(); ++si) {
        if (od_debug_session_drain(sessions[si])) die("quality completion drain timed out");
        collectQuality(sessions[si], (int)si);
      }
      if (completions.size() != records.size()) die("missing GPU quality completion records");
      std::map<std::pair<int, uint64_t>, ODQualityCompletion> byId;
      for (const auto& c : completions)
        if (!byId.emplace(std::make_pair(c.session, c.c.renderId), c.c).second) die("duplicate quality completion");
      for (auto& r : records) {
        const auto c = byId.find({r.session, r.decision.renderId});
        if (c == byId.end()) die("unmatched quality completion");
        r.callToFinishMs = c->second.callToFinishMs; r.gpuMs = c->second.gpuMs;
      }
    }
    const double tAll = nowMs() - tAll0;
    // The completions no render handed over (the last renders', and every
    // one held behind --hold-gpu-ms without fences), for the writes still
    // waiting for theirs.
    if (!unresolved.empty()) {
      for (ODSession* s : sessions) {
        od_debug_session_drain(s);
        ODCompletion c[OD_MAX_COMPLETIONS];
        int n;
        while ((n = od_debug_take_completions(s, c, OD_MAX_COMPLETIONS)) > 0) noteCompletions(c, n);
      }
    }
    const long hazardsUnresolved = (long)unresolved.size();
    memEnd = sampleMemory(0);
    if (o.idleMs > 0) {
      // Nothing renders now: what does the core give back, and when?
      const double t1 = nowMs();
      for (;;) {
        const double el = nowMs() - t1;
        if (el >= o.idleMs) break;
        usleep((useconds_t)(std::min(500.0, o.idleMs - el) * 1000.0));
        memIdle.push_back(sampleMemory(nowMs() - t1));
      }
    }
    if (analysisLog) fclose(analysisLog);
    if (rawOut) fclose(rawOut);
    if (ffOut && pclose(ffOut) != 0) die("ffmpeg reported an error writing %s", o.outVideo.c_str());

    // ---- report ----
    std::vector<double> gpu, wall, cpu;
    std::vector<double> stage[OD_STAGE_COUNT];
    long hits = 0;
    for (size_t i = 0; i < records.size(); ++i) {
      if ((int)i < o.warmup && records.size() > (size_t)o.warmup) continue;
      gpu.push_back(records[i].gpuMs);
      wall.push_back(records[i].wallMs);
      cpu.push_back(records[i].cpuMs);
      hits += records[i].cacheHits;
      for (int s = 0; s < OD_STAGE_COUNT; ++s) stage[s].push_back(records[i].stageMs[s]);
    }
    if (records.empty()) die("no frame was rendered");
    Summary sg = summarise(gpu), sw = summarise(wall), sc = summarise(cpu);
    int entries = 0, poolTex = 0;
    uint64_t cacheBytes = 0, poolBytes = 0, created = 0;
    od_debug_cache_info(sessions[0], &entries, &cacheBytes, &poolTex, &poolBytes, &created);

    fprintf(txt, "\nframes measured: %zu of %d rendered%s\n", gpu.size(), rendered,
            o.async ? "  (async: per-frame GPU time is that of the render matched by ID; wall = CPU call only)" : "");
    fprintf(txt, "            %9s %9s %9s %9s %9s\n", "mean", "median", "p95", "min", "max");
    fprintf(txt, "GPU ms      %9.3f %9.3f %9.3f %9.3f %9.3f\n", sg.mean, sg.median, sg.p95, sg.min, sg.max);
    fprintf(txt, "wall ms     %9.3f %9.3f %9.3f %9.3f %9.3f\n", sw.mean, sw.median, sw.p95, sw.min, sw.max);
    fprintf(txt, "CPU ms      %9.3f %9.3f %9.3f %9.3f %9.3f\n", sc.mean, sc.median, sc.p95, sc.min, sc.max);
    // Frames per second come from wall time only: blocking, 1000 / mean wall
    // ms; asynchronous, renders / elapsed time (frame generation included
    // unless --preload; the pacing rate when --interval-ms is set). 1000 / GPU
    // ms is not a frame rate: it is the ceiling if the GPU did nothing else.
    const double fpsWall = o.async ? (tAll > 0 ? 1000.0 * rendered / tAll : 0) : (sw.mean > 0 ? 1000.0 / sw.mean : 0);
    const double gpuBound = sg.mean > 0 ? 1000.0 / sg.mean : 0;
    fprintf(txt, "fps         %9.1f (%s)\n", fpsWall,
            o.intervalMs > 0 ? (o.async ? "paced: the pacing rate" : "from wall time; paced")
                             : (o.async ? "async throughput, frame generation included unless --preload"
                                        : "1000 / mean wall ms"));
    fprintf(txt, "1000/GPU ms %9.1f (GPU-bound ceiling, not a frame rate)\n", gpuBound);
    if (o.sessions > 1 || o.idleMs > 0) {
      const double G = 1073741824.0;
      auto line = [&](const char* what, const MemSample& m) {
        fprintf(txt, "  %-22s footprint %6.2f GB (+%5.2f)  GPU allocated %6.2f GB (+%5.2f)  caches %3d frames %5.2f GB\n",
                what, m.footprint / G, ((double)m.footprint - (double)mem0.footprint) / G, m.device / G,
                ((double)m.device - (double)mem0.device) / G, m.cacheEntries, m.cacheBytes / G);
      };
      fprintf(txt, "memory (process footprint and MTLDevice.currentAllocatedSize; + = above the harness baseline):\n");
      line("before the first render", mem0);
      line("peak", memPeak);
      line("after the last render", memEnd);
      for (const MemSample& m : memIdle) {
        char what[64];
        snprintf(what, sizeof(what), "idle %.1f s", m.ms / 1000.0);
        line(what, m);
      }
      ODDeviceMemory dm = {};
      od_debug_device_memory(ctx, &dm);
      fprintf(txt, "  device ledger: budget %.2f GB, logical peak %.2f GB (what the core held), physical peak %.2f GB "
                   "(MTLDevice, whole process); %d renders at once at most, %llu waited for room (%llu at the "
                   "deadline), %llu creations beyond the budget (by %.2f GB at most)\n",
              dm.budgetBytes / G, dm.logicalPeakBytes / G, dm.physicalPeakBytes / G, dm.rendersActiveMax,
              (unsigned long long)dm.admissionWaits, (unsigned long long)dm.admissionsForced,
              (unsigned long long)dm.overruns, dm.overrunPeakBytes / G);
    }
    if (o.stageTimings) {
      fprintf(txt, "per stage GPU ms (mean / median / p95):\n");
      for (int s = 0; s < OD_STAGE_COUNT; ++s) {
        Summary ss = summarise(stage[s]);
        fprintf(txt, "  %-9s %9.3f %9.3f %9.3f\n", kStageNames[s], ss.mean, ss.median, ss.p95);
      }
    }
    if (tracked)
      fprintf(txt, "async: %ld hazard(s) (CPU writes into a host buffer before the GPU work of the render that last "
                   "used it had ended, by that render's reported completion; %ld not judged: completion never "
                   "reported), %ld wait(s) on host-buffer fences (%.1f ms)%s%s\n", hazards, hazardsUnresolved,
              bufferWaits, bufferWaitMs, o.holdGpuMs > 0 ? ", queue held at the start" : "",
              o.unfenced ? ", UNFENCED (debug)" : o.fenceElsewhere ? ", FENCES ON ANOTHER QUEUE (debug)" : "");
    fprintf(txt, "cache hits/frame %.2f; cache %d frames, %.0f MB; pool %d idle textures, %.0f MB; %llu textures created\n",
            gpu.empty() ? 0.0 : (double)hits / (double)gpu.size(), entries, cacheBytes / 1048576.0, poolTex,
            poolBytes / 1048576.0, (unsigned long long)created);

    if (!o.json.empty()) {
      FILE* jf = o.json == "-" ? stdout : fopen(o.json.c_str(), "w");
      if (!jf) die("cannot write %s", o.json.c_str());
      auto sum = [&](const char* name, const Summary& s, const char* tail) {
        fprintf(jf, "  \"%s\": {\"mean\": %.4f, \"median\": %.4f, \"p95\": %.4f, \"min\": %.4f, \"max\": %.4f}%s\n", name,
                s.mean, s.median, s.p95, s.min, s.max, tail);
      };
      fprintf(jf, "{\n");
      fprintf(jf, "  \"label\": \"%s\",\n  \"cmdline\": \"%s\",\n", jsonEscape(o.label).c_str(), jsonEscape(cmdline).c_str());
      fprintf(jf, "  \"device\": \"%s\",\n  \"odVersion\": \"%s\",\n", jsonEscape(device.name.UTF8String).c_str(),
              od_version());
      fprintf(jf, "  \"source\": \"%s\", \"width\": %u, \"height\": %u, \"format\": \"%s\", \"rowBytes\": %u,\n", srcName,
              W, H, o.format == OD_PIXEL_BGRA16F ? "16f" : "32f", rowBytes);
      if (!o.video.empty())
        fprintf(jf, "  \"videoMatrix\": \"%s\", \"videoMatrixWhy\": \"%s\",\n",
                matrixUsed.empty() ? "ffmpeg" : matrixUsed.c_str(), jsonEscape(matrixWhy).c_str());
      fprintf(jf, "  \"framesRendered\": %d, \"framesMeasured\": %zu, \"warmup\": %d,\n", rendered, gpu.size(), o.warmup);
      fprintf(jf, "  \"mode\": {\"randomAccess\": %s, \"noCache\": %s, \"noFrameCache\": %s, \"verifyFrameCache\": %s, \"inPlace\": %s, \"async\": %s, \"stageTimings\": %s, \"preload\": %s, \"intervalMs\": %g, \"sessions\": %d},\n",
              o.randomAccess ? "true" : "false", o.noCache ? "true" : "false", o.params.noFrameCache ? "true" : "false",
              o.params.verifyFrameCache ? "true" : "false", o.inPlace ? "true" : "false",
              o.async ? "true" : "false", o.stageTimings ? "true" : "false", o.preload ? "true" : "false",
              o.intervalMs, o.sessions);
      fprintf(jf,
              "  \"params\": {\"lumaStrength\": %g, \"chromaStrength\": %g, \"temporalRadius\": %d, "
              "\"temporalStrength\": %g, \"spatialStrength\": %g, \"sharpen\": %g, \"noiseLevel\": %g, "
              "\"quality\": %d, \"view\": %d, \"manualSigma\": %g, \"debugFlags\": %u, \"debugSkipStages\": %u, "
              "\"realtime\": %u, \"temporalNoiseModel\": %u},\n",
              params.lumaStrength, params.chromaStrength, params.temporalRadius, params.temporalStrength,
              params.spatialStrength, params.sharpen, params.noiseLevel, params.quality, params.view,
              o.manual ? o.manualSigma : -1.0, params.debugFlags, params.debugSkipStages, params.realtime,
              params.temporalNoiseModel);
      sum("gpuMs", sg, ",");
      sum("wallMs", sw, ",");
      sum("cpuMs", sc, ",");
      fprintf(jf, "  \"realtime\": %s,\n", params.realtime ? "true" : "false");
      fprintf(jf, "  \"fpsWall\": %.3f, \"gpuBoundRate\": %.3f, \"totalMs\": %.3f, \"renderMs\": %.3f,\n", fpsWall,
              gpuBound, tAll, tRenderTotal);
      auto memJson = [&](const MemSample& m) {
        char b[256];
        snprintf(b, sizeof(b), "{\"ms\": %.0f, \"footprint\": %llu, \"device\": %llu, \"cacheEntries\": %d, \"cacheBytes\": %llu}",
                 m.ms, (unsigned long long)m.footprint, (unsigned long long)m.device, m.cacheEntries,
                 (unsigned long long)m.cacheBytes);
        return std::string(b);
      };
      fprintf(jf, "  \"memory\": {\"stoppedAtLimit\": %s, \"limitBytes\": %llu, \"baseline\": %s, \"peak\": %s, \"end\": %s, \"idle\": [",
              memStopped ? "true" : "false", (unsigned long long)memLimit, memJson(mem0).c_str(),
              memJson(memPeak).c_str(), memJson(memEnd).c_str());
      for (size_t i = 0; i < memIdle.size(); ++i) fprintf(jf, "%s%s", i ? ", " : "", memJson(memIdle[i]).c_str());
      ODDeviceMemory dm = {};
      od_debug_device_memory(ctx, &dm);
      fprintf(jf, "], \"ledger\": {\"budget\": %llu, \"logicalPeak\": %llu, \"physicalPeak\": %llu, \"rendersActiveMax\": %d, "
                  "\"admissionWaits\": %llu, \"admissionsForced\": %llu, \"overruns\": %llu, \"overrunPeak\": %llu}},\n",
              (unsigned long long)dm.budgetBytes, (unsigned long long)dm.logicalPeakBytes,
              (unsigned long long)dm.physicalPeakBytes, dm.rendersActiveMax, (unsigned long long)dm.admissionWaits,
              (unsigned long long)dm.admissionsForced, (unsigned long long)dm.overruns,
              (unsigned long long)dm.overrunPeakBytes);
      fprintf(jf, "  \"stageMs\": {");
      for (int s = 0; s < OD_STAGE_COUNT; ++s) {
        Summary ss = summarise(stage[s]);
        fprintf(jf, "%s\"%s\": {\"mean\": %.4f, \"median\": %.4f, \"p95\": %.4f}", s ? ", " : "", kStageNames[s], ss.mean,
                ss.median, ss.p95);
      }
      fprintf(jf, "},\n");
      fprintf(jf, "  \"async\": {\"tracked\": %s, \"inFlight\": %d, \"holdGpuMs\": %d, \"unfenced\": %s, "
                  "\"fenceElsewhere\": %s, \"hazards\": %ld, \"hazardsUnresolved\": %ld, \"bufferWaits\": %ld, "
                  "\"bufferWaitMs\": %.3f},\n",
              tracked ? "true" : "false", depth, o.holdGpuMs, o.unfenced ? "true" : "false",
              o.fenceElsewhere ? "true" : "false", hazards, hazardsUnresolved, bufferWaits, bufferWaitMs);
      fprintf(jf, "  \"cache\": {\"hitsPerFrame\": %.3f, \"entries\": %d, \"bytes\": %llu, \"poolTextures\": %d, "
                  "\"poolBytes\": %llu, \"texturesCreated\": %llu},\n",
              gpu.empty() ? 0.0 : (double)hits / (double)gpu.size(), entries, (unsigned long long)cacheBytes, poolTex,
              (unsigned long long)poolBytes, (unsigned long long)created);
      fprintf(jf, "  \"perFrame\": [\n");
      for (size_t i = 0; i < records.size(); ++i) {
        const FrameRecord& r = records[i];
        fprintf(jf, "    {\"frame\": %d, \"session\": %d, \"gpuMs\": %.4f, \"wallMs\": %.4f, \"cpuMs\": %.4f, \"waitMs\": %.4f, \"gpuQueueMs\": %.4f, \"gpuLatencyMs\": %.4f, \"cacheHits\": %d, \"framesUsed\": %d, \"profileHash\": \"%016llx\", "
                    "\"renderId\": %llu, \"runId\": %llu, \"quality\": %d, \"switches\": %d, \"budgetMs\": %.4f, \"emaMs\": %.4f, \"callToFinishMs\": %.4f, \"startLateMs\": %.4f}%s\n",
                r.index, r.session, r.gpuMs, r.wallMs, r.cpuMs, r.waitMs, r.queueMs, r.latencyMs, r.cacheHits, r.framesUsed,
                (unsigned long long)r.profileHash, (unsigned long long)r.decision.renderId,
                (unsigned long long)r.decision.runId, r.decision.quality, r.decision.switches,
                r.decision.budgetMs, r.decision.emaMs, r.callToFinishMs, r.startLateMs,
                i + 1 < records.size() ? "," : "");
      }
      fprintf(jf, "  ]\n}\n");
      if (jf != stdout) fclose(jf);
    }

    src.reset();
    for (ODSession* s : sessions) od_session_destroy(s);
    od_context_destroy(ctx);
    return 0;
  }
}
