// SPDX-License-Identifier: MIT
// temporal stage: motion-compensated robust merge of the
// temporal window. Contract: stage.h sections 3, 4 and 6. Shaders:
// core/shaders/temporal_{inc,motion,merge}.metal.
//
// What a render does (n = neighbours, at most 2 * OD_MAX_RADIUS):
//   1. Luma tent pyramid P_1..P_NP of every frame of the window, cached per
//      frame through env.frameTexture (slots 'tp1'..'tp5'): the only work that
//      depends on one frame alone.
//   2. Motion, every neighbour in the same dispatches: hierarchical block
//      matching on the pyramid (top level K: full search +-4; below, down to
//      level 2: parent seeds and +-1), then damped Lucas-Kanade on the level-1
//      grid (one step at quality 0, two at quality 1). Result: one vector per
//      8x8 pixels, in pixels (rg16f).
//   3. Per neighbour, a global gate computed on the GPU (the stage itself never
//      waits for it; the pipeline reads the cut values back after the render
//      to restart the automatic profile at a cut): 0 when
//      the neighbour is another shot (most textured blocks have no match even
//      after motion search, brightness changes excluded) or a bit-exact repeat
//      of the next frame inwards (it carries no new information): Y, Cb and
//      Cr compared texel by texel at full resolution, for the neighbours whose
//      coarse luma in the cut statistics is already identical.
//   4. Merge, four neighbours at a time: warp and difference against the
//      current frame, tent pyramid of the differences, then per level a
//      noise-normalised, locally averaged energy test per neighbour and band;
//      weighted mean; collapse. Level 0 writes the merged frame and the
//      effective frame count.
//
// OD_TEMPORAL_TRACE=N prints the GPU time of each kernel group.
#include "stage.h"
#include "../kernel_trace.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

namespace od {
namespace {

constexpr int kMaxNeighbours = 2 * OD_MAX_RADIUS;
constexpr int kMaxLevels = 6;  // pyramid levels 0..5
constexpr float kBeta = 0.5f;   // motion: seed penalty per pixel, in units of the level's noise variance
constexpr float kMu = 64.0f;    // motion: Lucas-Kanade damping, in units of the level-1 noise variance
constexpr float kVote = 16.0f;  // cut statistics: a block votes above this many noise variances
constexpr float kVoteNoise = 4.0f;  // cut statistics: noise allowance of the mismatch test, in noise variances
// Motion matching penalizes nonzero vectors by kZero noise variances to avoid selecting
// random motion in flat noisy areas.
constexpr float kZero = 0.5f;
constexpr float kT0 = 2.0f;     // merge: averaged normalised energy below this is noise (expectation 1)
constexpr float kT1 = 5.0f;     // merge: above this the neighbour is rejected in that band
// The finest level doubles luma's full-acceptance threshold, while leaving full rejection
// and chroma thresholds unchanged. This accepts isolated noise excursions without averaging
// thin moving lines into their background.
constexpr float kLevel0 = 2.0f;
constexpr float kLevel0Chroma = 1.0f;
// merge: a band energy between these rejects the neighbour fully at that
// level and at every finer one (a coarse rejection is otherwise handed down
// at kGateCoarse). Far beyond noise, so it changes nothing but very bright
// edges out of place, which the looser finest level would let through.
constexpr float kStrong0 = 30.0f, kStrong1 = 90.0f;
constexpr float kGateCoarse = kEdgeGateNeutral;  // merge: share of the coarser level's rejection handed to the finer one
// merge: the least gate an editor's Edge ghost protection gives
// (ODParams.temporalEdgeGate 0). Not 0: the shader hands a coarser level's
// strong mismatch down only while the gate is on (temporal_merge.metal), and
// that hand-down keeps bright edges out of place from ghosting.
constexpr float kMinEdgeGate = 1e-6f;
// merge, playback quality: the deltas are stored with an offset of about this
// many sigmas. 0 would store them as they are.
constexpr float kDeltaBias = 6.0f;

// Tent-pyramid noise relative to the a-trous band of the same octave.
constexpr float kL[4] = {1.04f, 1.31f, 1.35f, 1.36f};                  // L_0..L_3 against bands 0..3
constexpr float kG[kMaxLevels] = {1.0f, 1.56f, 1.66f, 1.68f, 0.83f, 0.41f};  // G_k against band min(k, 3)

// Start values above; OD_TEMPORAL_TUNE="name=value,..." overrides them for
// tuning runs (read once per stage, never set in production).
struct Tune {
  float t0 = kT0, t1 = kT1, level0 = kLevel0, strong0 = kStrong0, strong1 = kStrong1, gate = kGateCoarse;
  float level0c = kLevel0Chroma;
  float zero = kZero, beta = kBeta, mu = kMu;
  float taps = 5, lk0 = 1, lk1 = 2;  // warp taps (1, 5, 9); Lucas-Kanade steps at quality 0 and 1 (1 or more)
  void parse(const char* e) {
    struct {
      const char* name;
      float* v;
    } tab[] = {{"t0", &t0},     {"t1", &t1},     {"level0", &level0}, {"strong0", &strong0}, {"strong1", &strong1},
               {"gate", &gate}, {"zero", &zero}, {"beta", &beta},     {"mu", &mu},           {"taps", &taps},
               {"lk0", &lk0},   {"lk1", &lk1}, {"level0c", &level0c}};
    while (e && *e) {
      const char* eq = strchr(e, '=');
      if (!eq) break;
      for (auto& t : tab)
        if (strlen(t.name) == (size_t)(eq - e) && strncmp(t.name, e, (size_t)(eq - e)) == 0) *t.v = (float)atof(eq + 1);
      e = strchr(eq, ',');
      if (e) ++e;
    }
    lk0 = std::max(lk0, 1.0f);
    lk1 = std::max(lk1, 1.0f);
  }
};

struct MotionU {
  float knots[OD_LEVELS];
  float penK, muK, voteK, noiseK;
  float zeroK;
  uint32_t pad0, pad1, pad2;
  float inScale, outScale, minVotes;
  uint32_t flags;
  uint32_t first, rows, inRows, count;
  float seedScale[kMaxNeighbours];
  int32_t inner[kMaxNeighbours];
};

struct WarpU {
  float size[2];
  uint32_t taps, n, first, rows;
  float bias;
  uint32_t pad0;
};

struct MergeU {
  float knots[8][4];
  float mask[4];
  float t0, t1, gateK;
  uint32_t hasPrev, isTop, lumUp;
  float aY, aC;
  uint32_t first;
  float bias, biasUp;
  float keep;
  float s0, s1;
  float t0c, t1c;
};

struct ViewU {
  float invCount;
  uint32_t rows, pad0, pad1;
};

// The layouts of the shader structs (TMotionU, TWarpU, TMergeU, TViewU).
static_assert(sizeof(MotionU) == 176 && sizeof(WarpU) == 32 && sizeof(MergeU) == 208 && sizeof(ViewU) == 16,
              "uniform structs must match core/shaders/temporal_*.metal");

struct Size {
  uint32_t w = 0, h = 0;
};

// ---------------------------------------------------------------------------
// OD_TEMPORAL_TRACE=N (debugging aid, off by default): every group of kernels
// gets its own compute pass with GPU timestamps at its start and end, and at
// exit the process prints the GPU time per group and render, leaving out the
// first N renders. Apple GPUs sample counters at pass boundaries only, so this
// is the one way to time kernels inside a real render. Passes overlap on the
// GPU (measured), so each group is given its share of the time line (an
// instant with k passes running counts 1/k for each; the shares add up to the
// stage) next to the plain sum of its pass durations. Medians over the
// renders, because anything else using the GPU only ever adds time.
// ---------------------------------------------------------------------------
class Trace {
 public:
  static int skip() {
    static const int n = [] {
      const char* e = getenv("OD_TEMPORAL_TRACE");
      return e ? std::max(0, atoi(e)) : -1;
    }();
    return n;
  }

  // Starts the pass of group `label` (a string literal) and returns its
  // encoder; without tracing that is the render's shared encoder.
  id<MTLComputeCommandEncoder> begin(RenderEnv& env, const char* label) {
    if (skip() < 0) return env.compute();
    end();
    if (!buffer_) {
      id<MTLCounterSet> set = nil;
      for (id<MTLCounterSet> s in env.device().counterSets)
        if ([s.name isEqualToString:MTLCommonCounterSetTimestamp]) set = s;
      if (!set || ![env.device() supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) return env.compute();
      MTLCounterSampleBufferDescriptor* d = [MTLCounterSampleBufferDescriptor new];
      d.counterSet = set;
      d.storageMode = MTLStorageModeShared;
      d.sampleCount = kMaxPasses * 2;
      buffer_ = [env.device() newCounterSampleBufferWithDescriptor:d error:nil];
      if (!buffer_) return env.compute();
    }
    if (labels_.size() >= kMaxPasses) return env.compute();
    env.endEncoding();
    MTLComputePassDescriptor* pd = [MTLComputePassDescriptor computePassDescriptor];
    pd.sampleBufferAttachments[0].sampleBuffer = buffer_;
    pd.sampleBufferAttachments[0].startOfEncoderSampleIndex = labels_.size() * 2;
    pd.sampleBufferAttachments[0].endOfEncoderSampleIndex = labels_.size() * 2 + 1;
    labels_.push_back(label);
    pass_ = [env.commandBuffer() computeCommandEncoderWithDescriptor:pd];
    return pass_;
  }

  // Ends the open pass, if any. An encode() that fails must still do this.
  void abandon() {
    end();
    labels_.clear();
  }

  // Ends the last pass; the samples are read when the command buffer is done.
  void finish(RenderEnv& env) {
    end();
    if (!buffer_ || labels_.empty()) return;
    id<MTLCounterSampleBuffer> buffer = buffer_;
    std::vector<const char*> labels = std::move(labels_);
    buffer_ = nil;
    labels_.clear();
    [env.commandBuffer() addCompletedHandler:^(id<MTLCommandBuffer>) {
      const size_t n = labels.size();
      NSData* data = [buffer resolveCounterRange:NSMakeRange(0, n * 2)];
      if (!data || data.length < n * 2 * sizeof(MTLCounterResultTimestamp)) return;
      const MTLCounterResultTimestamp* t = (const MTLCounterResultTimestamp*)data.bytes;
      std::vector<uint64_t> cuts;
      for (size_t i = 0; i < n * 2; ++i) {
        if (t[i].timestamp == MTLCounterErrorValue) return;
        cuts.push_back(t[i].timestamp);
      }
      std::sort(cuts.begin(), cuts.end());
      std::map<std::string, Row> render;
      for (size_t c = 0; c + 1 < cuts.size(); ++c) {
        int running = 0;
        for (size_t i = 0; i < n; ++i) running += t[2 * i].timestamp <= cuts[c] && t[2 * i + 1].timestamp >= cuts[c + 1];
        const double dt = (double)(cuts[c + 1] - cuts[c]);
        if (!running) render["(idle)"].share += dt;
        for (size_t i = 0; i < n; ++i)
          if (t[2 * i].timestamp <= cuts[c] && t[2 * i + 1].timestamp >= cuts[c + 1]) render[labels[i]].share += dt / running;
      }
      std::vector<size_t> order(n);
      for (size_t i = 0; i < n; ++i) order[i] = i;
      std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return t[2 * a + 1].timestamp < t[2 * b + 1].timestamp; });
      uint64_t prevEnd = 0;
      for (size_t k = 0; k < n; ++k) {
        const size_t i = order[k];
        render[labels[i]].serial += (double)(t[2 * i + 1].timestamp - std::max(t[2 * i].timestamp, prevEnd));
        prevEnd = t[2 * i + 1].timestamp;
      }
      for (size_t i = 0; i < n; ++i) {
        render[labels[i]].sum += (double)(t[2 * i + 1].timestamp - t[2 * i].timestamp);
        render[labels[i]].passes += 1;
      }
      render["(stage)"].serial = render["(stage)"].share = render["(stage)"].sum = (double)(cuts.back() - cuts.front());
      Totals& g = totals();
      std::lock_guard<std::mutex> lock(g.mutex);
      if (g.renders++ < skip()) return;
      for (auto& e : render) g.rows[e.first].push_back(e.second);
    }];
  }

 private:
  static constexpr size_t kMaxPasses = 512;
  struct Row {
    double share = 0, serial = 0, sum = 0, passes = 0;
  };
  struct Totals {
    std::mutex mutex;
    std::map<std::string, std::vector<Row>> rows;
    int renders = 0;
  };
  static Totals& totals() {
    static Totals* g = [] {
      atexit([] {
        Totals& t = totals();
        std::lock_guard<std::mutex> lock(t.mutex);
        if (t.rows.empty()) return;
        fprintf(stderr, "temporal trace, %d renders: GPU ms per render (median share of the time line, median sum of "
                        "the pass durations), passes per render\n", t.renders - skip());
        for (auto& e : t.rows) {
          auto median = [&](double Row::* m) {
            std::vector<double> v;
            for (const Row& r : e.second) v.push_back(r.*m);
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
          };
          fprintf(stderr, "  %-20s %7.3f %7.3f %7.3f  %4.0f\n", e.first.c_str(), median(&Row::serial) * 1e-6,
                  median(&Row::share) * 1e-6, median(&Row::sum) * 1e-6, median(&Row::passes));
        }
      });
      return new Totals;
    }();
    return *g;
  }

  void end() {
    if (pass_) [pass_ endEncoding];
    pass_ = nil;
  }

  id<MTLCounterSampleBuffer> buffer_ = nil;
  id<MTLComputeCommandEncoder> pass_ = nil;
  std::vector<const char*> labels_;
};

class Temporal final : public TemporalStage {
 public:
  bool init(Context& ctx, std::string* err) {
    tune_.parse(getenv("OD_TEMPORAL_TUNE"));
    struct {
      id<MTLComputePipelineState> __strong* pso;
      const char* name;
    } list[] = {
        {&pyrDown_, "temporal_pyr_down"},     {&bmTopCost_, "temporal_bm_top_cost"}, {&bmTopPick_, "temporal_bm_top_pick"},
        {&bmLevel_, "temporal_bm_level"},
        {&lk_, "temporal_lk"},              {&cutStats_, "temporal_cutstats"},
        {&reduce_, "temporal_reduce"},       {&gate_, "temporal_gate"},          {&warp_, "temporal_warp"},
        {&down_, "temporal_down"},           {&energy_, "temporal_energy"},      {&mergeAcc_, "temporal_merge_acc"},
        {&mergeFin_, "temporal_merge_fin"},  {&mergeFin0_, "temporal_merge_fin0"}, {&viewMotion_, "temporal_view_motion"},
        {&repeatArm_, "temporal_repeat_arm"}, {&repeat_, "temporal_repeat"},
    };
    for (auto& e : list) {
      *e.pso = ctx.pipeline(e.name, err);
      if (!*e.pso) return false;
    }
    // The exact-repeat test's arguments, candidates and flags
    // (temporal_motion.metal). The session's renders run in order on one
    // queue, so one buffer per stage serves them all.
    repeatBuf_ = [ctx.device() newBufferWithLength:kRepeatBytes options:MTLResourceStorageModePrivate];
    if (!repeatBuf_) {
      if (err) *err = "temporal: could not allocate the repeat test's buffer";
      return false;
    }
    return true;
  }

  bool encode(RenderEnv& env, TemporalResult* out) override;

 private:
  id<MTLComputePipelineState> pyrDown_, bmTopCost_, bmTopPick_, bmLevel_, lk_, cutStats_, reduce_, gate_;
  id<MTLComputePipelineState> warp_, down_, energy_, mergeAcc_, mergeFin_, mergeFin0_, viewMotion_;
  id<MTLComputePipelineState> repeatArm_, repeat_;
  static constexpr NSUInteger kRepeatBytes = 128;  // temporal_motion.metal, "Exact repeats"
  id<MTLBuffer> repeatBuf_ = nil;
  Trace trace_;
  Tune tune_;
};

bool Temporal::encode(RenderEnv& env, TemporalResult* out) {
  out->image = env.frames[0].tex;
  out->count = nil;
  out->framesUsed = 1;
  const ODParams& p = env.params;
  const int nb = std::min(env.frameCount - 1, kMaxNeighbours);
  // Nothing safe to do without a noise level; tiny frames have no pyramid.
  if (nb < 1 || !env.profile.valid || env.width < 32 || env.height < 32) return true;

  // ---- geometry ---------------------------------------------------------
  Size sz[kMaxLevels];
  sz[0] = {env.width, env.height};
  for (int k = 1; k < kMaxLevels; ++k) sz[k] = {(sz[k - 1].w + 1) / 2, (sz[k - 1].h + 1) / 2};
  const uint32_t minDim = std::min(env.width, env.height);
  int K = 1;  // top motion level
  for (int k = 2; k <= 5; ++k)
    if ((minDim >> k) >= 64) K = k;
  int MT = 1;  // top merge level
  for (int b = 2; b <= 4; ++b)
    if (std::min(sz[b].w, sz[b].h) >= 8) MT = b;
  const int NP = std::max(K, MT);
  auto grid = [&](int k) { return Size{(sz[k].w + 3) / 4, (sz[k].h + 3) / 4}; };
  const bool full = p.quality != 0;
  const int kFine = std::min(K, 2);  // finest block-matching level
  const int kStat = std::max(kFine, std::min(K, 3));  // level of the cut statistics

  // ---- noise ------------------------------------------------------------
  const float sScale = env.sigmaScale();
  auto band = [&](int c, int b, int l) { return std::max(env.profile.sigma[c][b][l] * sScale, 1e-5f); };
  // The same, with the editor's per band and channel correction (ODParams
  // bandNoiseLevel / channelNoiseLevel), for the merge's noise thresholds
  // only: motion search and the cut statistics keep `band`.
  auto bandMerge = [&](int c, int b, int l) {
    float s = env.profile.sigma[c][b][l] * sScale;
    const float adj = env.noiseAdjust(c, b);
    if (adj != 1.0f) s *= adj;
    return std::max(s, 1e-5f);
  };
  // Luma noise of pyramid level k at each luminance knot. The motion kernels
  // look it up at each block's own luminance (temporal_motion.metal, t_var).
  auto levelKnots = [&](MotionU& m, int k) {
    for (int l = 0; l < OD_LEVELS; ++l) m.knots[l] = kG[k] * band(0, std::min(k, 3), l);
  };
  const float t = p.temporalNoiseModel ? 1.0f : std::max(p.temporalStrength, 1e-4f);
  // Amounts are applied once, by the last denoising stage that runs
  // (stage.h section 1): here only when spatial will not run.
  const float aY = env.spatialRuns ? 1.0f : p.lumaStrength;
  const float aC = env.spatialRuns ? 1.0f : p.chromaStrength;

  struct Closer {
    Trace& trace;
    ~Closer() { trace.abandon(); }
  } closer{trace_};
  id<MTLComputeCommandEncoder> enc = env.compute();
  auto group = [&](const char* label) { enc = kernelTraceEnabled() ? env.compute(label) : trace_.begin(env, label); };
  auto run = [&](id<MTLComputePipelineState> pso, uint32_t w, uint32_t h) {
    [enc setComputePipelineState:pso];
    [enc dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:Context::threadgroupFor(pso)];
  };
  static const char* const kLevelLabel[4][kMaxLevels] = {
      {"bm level 0", "bm level 1", "bm level 2", "bm level 3", "bm level 4", "bm level 5"},
      {"down 0", "down 1", "down 2", "down 3", "down 4", "down 5"},
      {"energy 0", "energy 1", "energy 2", "energy 3", "energy 4", "energy 5"},
      {"merge 0", "merge 1", "merge 2", "merge 3", "merge 4", "merge 5"}};
  group("pyramid");

  // ---- 1. luma pyramids (cached per frame) --------------------------------
  id<MTLTexture> pyr[OD_MAX_FRAMES][kMaxLevels];
  for (int f = 0; f <= nb; ++f) {
    pyr[f][0] = env.frames[f].tex;
    for (int k = 1; k <= NP; ++k) {
      bool fill = false;
      id<MTLTexture> tex = env.frameTexture(env.frames[f], OD_SLOT('t', 'p', '0' + k, ' '), MTLPixelFormatR16Float,
                                            sz[k].w, sz[k].h, 1, &fill);
      if (!tex) return false;
      pyr[f][k] = tex;
      if (fill) {
        if (kernelTraceEnabled()) group("temporal_pyr_down");
        [enc setTexture:pyr[f][k - 1] atIndex:0];
        [enc setTexture:tex atIndex:1];
        run(pyrDown_, sz[k].w, sz[k].h);
      }
    }
  }

  // ---- 2 + 3. motion and gate, all neighbours per dispatch ------------------
  // Neighbour i (1..nb) has slot a = i - 1 in the atlases (temporal_motion.metal).
  MotionU u = {};
  for (int i = 1; i <= nb; ++i) {
    const int off = env.frames[i].offset;
    // The next frame inwards on the same side (frames are sorted nearest first).
    int inner = 0;
    for (int j = 1; j < i; ++j) {
      const int oj = env.frames[j].offset;
      if ((oj > 0) == (off > 0) && std::abs(oj) < std::abs(off) &&
          (inner == 0 || std::abs(oj) > std::abs(env.frames[inner].offset)))
        inner = j;
    }
    u.inner[i - 1] = inner - 1;
    u.seedScale[i - 1] = inner ? (float)off / (float)env.frames[inner].offset : 0.0f;
  }
  u.count = (uint32_t)nb;
  const Tune& T = tune_;
  u.penK = T.beta;
  u.muK = T.mu;
  u.voteK = kVote;
  u.noiseK = kVoteNoise;
  u.zeroK = T.zero;
  auto bindLevel = [&](int k) {
    for (int f = 0; f <= kMaxNeighbours; ++f) [enc setTexture:pyr[std::min(f, nb)][k] atIndex:(NSUInteger)f];
  };
  // Binds input and output at 11 and 12 and dispatches over w x h x neighbours.
  auto runAll = [&](id<MTLComputePipelineState> pso, id<MTLTexture> in, id<MTLTexture> outTex, uint32_t w, uint32_t h,
                    uint32_t count) {
    [enc setTexture:in atIndex:11];
    [enc setTexture:outTex atIndex:12];
    [enc setBytes:&u length:sizeof(u) atIndex:0];
    [enc setComputePipelineState:pso];
    MTLSize tg = Context::threadgroupFor(pso);
    tg.depth = 1;
    [enc dispatchThreads:MTLSizeMake(w, h, count) threadsPerThreadgroup:tg];
  };
  auto atlas = [&](MTLPixelFormat f, Size g) { return env.temp(f, g.w, g.h * (uint32_t)nb); };

  id<MTLTexture> flow[kMaxLevels] = {nil};  // vectors of every neighbour at level k
  // top level: full search, in batches whose inner neighbours are already done
  {
    const Size g = grid(K);
    flow[K] = atlas(MTLPixelFormatRG16Float, g);
    if (!flow[K]) return false;
    levelKnots(u, K);
    u.rows = g.h;
    for (int a0 = 0; a0 < nb;) {
      int a1 = a0 + 1;
      while (a1 < nb && u.inner[a1] < a0) ++a1;
      const uint32_t count = (uint32_t)(a1 - a0);
      id<MTLTexture> cost = env.temp(MTLPixelFormatRGBA32Float, g.w * 9, g.h * 9 * count);
      if (!cost) return false;
      u.first = (uint32_t)a0;
      group("bm top cost");
      bindLevel(K);
      runAll(bmTopCost_, flow[K], cost, g.w * 9, g.h * 9, count);
      group("bm top pick");
      runAll(bmTopPick_, cost, flow[K], g.w, g.h, count);
      env.recycle(cost);
      a0 = a1;
    }
    u.first = 0;
  }
  for (int k = K - 1; k >= kFine; --k) {
    const Size g = grid(k);
    flow[k] = atlas(MTLPixelFormatRG16Float, g);
    if (!flow[k]) return false;
    levelKnots(u, k);
    u.rows = g.h;
    u.inRows = grid(k + 1).h;
    group(kLevelLabel[0][k]);
    bindLevel(k);
    runAll(bmLevel_, flow[k + 1], flow[k], g.w, g.h, (uint32_t)nb);
  }
  // gate: block statistics at level kStat, summed down to one texel per neighbour
  id<MTLTexture> gate = nil;
  {
    Size g = grid(kStat);
    id<MTLTexture> stats = atlas(MTLPixelFormatRGBA32Float, g);
    if (!stats) return false;
    levelKnots(u, kStat);
    u.minVotes = std::max(8.0f, 0.02f * (float)(g.w * g.h));
    u.rows = g.h;
    group("cut stats");
    bindLevel(kStat);
    runAll(cutStats_, flow[kStat], stats, g.w, g.h, (uint32_t)nb);
    group("cut reduce");
    while (g.w > 1 || g.h > 1) {
      const Size r = {(g.w + 7) / 8, (g.h + 7) / 8};
      id<MTLTexture> next = atlas(MTLPixelFormatRGBA32Float, r);
      if (!next) return false;
      u.inRows = g.h;
      u.rows = r.h;
      if (kernelTraceEnabled()) group("temporal_reduce");
      runAll(reduce_, stats, next, r.w, r.h, (uint32_t)nb);
      env.recycle(stats);
      stats = next;
      g = r;
    }
    // Neighbours whose statistics-level luma equals the inner neighbour's
    // are compared at full resolution, every channel, before the gate
    // calls them repeats (temporal_motion.metal, "Exact repeats").
    group("repeat");
    {
      MTLSize tg = Context::threadgroupFor(repeat_);
      tg.depth = 1;
      struct {
        uint32_t groupsX, groupsY, pad0, pad1;
      } r = {(uint32_t)(((env.width + 3) / 4 + tg.width - 1) / tg.width),  // kRepTexels = 4 along x
             (uint32_t)((env.height + tg.height - 1) / tg.height), 0, 0};
      [enc setTexture:stats atIndex:11];
      [enc setBytes:&u length:sizeof(u) atIndex:0];
      [enc setBuffer:repeatBuf_ offset:0 atIndex:1];
      [enc setBytes:&r length:sizeof(r) atIndex:2];
      run(repeatArm_, 1, 1);
      for (int f = 0; f <= kMaxNeighbours; ++f)
        [enc setTexture:env.frames[std::min(f, nb)].tex atIndex:(NSUInteger)f];
      [enc setComputePipelineState:repeat_];
      [enc dispatchThreadgroupsWithIndirectBuffer:repeatBuf_ indirectBufferOffset:0 threadsPerThreadgroup:tg];
    }
    group("cut gate");
    gate = env.temp(MTLPixelFormatRG16Float, 1, (uint32_t)nb);
    if (!gate) return false;
    if (kernelTraceEnabled()) group("temporal_gate");
    [enc setBuffer:repeatBuf_ offset:0 atIndex:1];
    runAll(gate_, stats, gate, 1, 1, 1);
    env.recycle(stats);
  }
  // sub-pixel refinement on the level-1 grid
  id<MTLTexture> flowV = nil;  // final vectors, level-1 grid, full-res pixels
  const Size g1 = grid(1);
  {
    levelKnots(u, 1);
    id<MTLTexture> src = flow[kFine];
    const bool upsample = kFine > 1;  // vectors come from the level-2 grid
    const int steps = (int)(full ? T.lk1 : T.lk0);
    group("lk");
    bindLevel(1);
    for (int s = 0; s < steps; ++s) {
      id<MTLTexture> dst = atlas(MTLPixelFormatRG16Float, g1);
      if (!dst) return false;
      const bool first = s == 0, last = s == steps - 1;
      u.flags = (first && upsample) ? 1u : 0u;
      u.inScale = (first && upsample) ? 2.0f : 1.0f;
      u.outScale = last ? 2.0f : 1.0f;
      u.rows = g1.h;
      u.inRows = first ? grid(kFine).h : g1.h;
      if (kernelTraceEnabled()) { group("temporal_lk"); bindLevel(1); }
      runAll(lk_, src, dst, g1.w, g1.h, (uint32_t)nb);
      if (!first) env.recycle(src);
      src = dst;
    }
    flowV = src;
  }
  for (int k = kFine; k <= K; ++k) env.recycle(flow[k]);

  // ---- 4. merge -----------------------------------------------------------
  // Playback quality stores delta + bias, halved from level to level like the
  // noise. A power of two, so that a delta
  // of exactly 0 stays exactly 0 through the pyramid.
  float dBias = 0.0f;
  if (!full && kDeltaBias > 0.0f) {
    float sFull = 0.0f;
    for (int c = 0; c < 3; ++c)
      for (int l = 0; l < OD_LEVELS; ++l) sFull = std::max(sFull, env.profile.sigmaFull[c][l] * sScale);
    dBias = std::exp2(std::round(std::log2(std::max(kDeltaBias * sFull, 1e-6f))));
  }
  MergeU mu[kMaxLevels] = {};
  const float mThr = p.temporalMotionThreshold;
  // Edge ghost protection (ODParams.temporalEdgeGate); the tuning override
  // stands while it is at its neutral value.
  const float edgeGate = p.temporalEdgeGate != kGateCoarse ? std::max(p.temporalEdgeGate, kMinEdgeGate) : T.gate;
  for (int b = 0; b <= MT; ++b) {
    const bool top = b == MT;
    const int pb = std::min(b, 3);
    const float k = top ? kG[b] : kL[b];
    for (int l = 0; l < OD_LEVELS; ++l)
      for (int c = 0; c < 3; ++c) mu[b].knots[l][c] = k * bandMerge(c, pb, l);
    // Level 0: luma's t0 only is raised (kLevel0), chroma keeps the
    // thresholds of the other levels (kLevel0Chroma = 1).
    const float tl = b == 0 ? T.level0 : 1.0f;
    const float tc = b == 0 ? T.level0c : 1.0f;
    mu[b].t0 = T.t0 * t * tl;
    // Motion threshold (ODParams.temporalMotionThreshold) moves full
    // rejection only, of luma and chroma; t1 stays at least 0.5 t above t0
    // (2.5 t against 2 t at 0.5; 4.5 t against 4 t at luma's level 0) and the
    // strong-mismatch range above t1.
    float t1n = T.t1;
    if (mThr != 1.0f) t1n *= mThr;
    mu[b].t1 = std::max(t1n, T.t0 * tl + 0.5f) * t;
    mu[b].t0c = T.t0 * t * tc;
    mu[b].t1c = std::max(t1n, T.t0 * tc + 0.5f) * tc * t;
    mu[b].s0 = std::max(T.strong0, mu[b].t1 / t) * t;
    mu[b].s1 = std::max(T.strong1, mu[b].s0 / t + 1.0f) * t;
    mu[b].gateK = (b <= MT - 2) ? edgeGate : 0.0f;  // the top level does not gate
    mu[b].isTop = top ? 1u : 0u;
    mu[b].lumUp = b == 0 ? 1u : 0u;
    const float blend = p.temporalNoiseModel && !env.spatialRuns ? p.temporalStrength : 1.0f;
    mu[b].aY = aY * blend;
    mu[b].aC = aC * blend;
    mu[b].bias = std::ldexp(dBias, -b);
    mu[b].biasUp = std::ldexp(dBias, -(b + 1));
  }
  // The grain floor goes with the amounts (stage.h): level 0 applies it when
  // the spatial stage does not run.
  if (!env.spatialRuns) {
    mu[0].keep = env.grainProfileV2 ? 0.0f : env.grainFloor();
    for (int l = 0; l < OD_LEVELS; ++l) mu[0].knots[l][3] = lumaSigmaFull(env.profile, l);
  }

  id<MTLTexture> acc1[kMaxLevels] = {nil}, acc2[kMaxLevels] = {nil};
  id<MTLTexture> dOut[kMaxLevels] = {nil};
  id<MTLTexture> image = nil, count = nil;
  const int groups = (nb + 3) / 4;
  for (int g = 0; g < groups; ++g) {
    const int first = 1 + g * 4;
    const int n = std::min(4, nb - g * 4);
    const bool last = g == groups - 1;
    // One colour channel per neighbour (temporal_merge.metal, "Layout").
    const MTLPixelFormat fHalf = n == 1 ? MTLPixelFormatR16Float : n == 2 ? MTLPixelFormatRG16Float : MTLPixelFormatRGBA16Float;
    const MTLPixelFormat fValid = n == 1 ? MTLPixelFormatR8Unorm : n == 2 ? MTLPixelFormatRG8Unorm : MTLPixelFormatRGBA8Unorm;
    struct Level {
      id<MTLTexture> y, cb, cr, v;
    } lv[kMaxLevels];
    for (int b = 0; b <= MT; ++b) {
      lv[b].y = env.temp(fHalf, sz[b].w, sz[b].h);
      lv[b].cb = env.temp(fHalf, sz[b].w, sz[b].h);
      lv[b].cr = env.temp(fHalf, sz[b].w, sz[b].h);
      lv[b].v = env.temp(fValid, sz[b].w, sz[b].h);
      if (!lv[b].y || !lv[b].cb || !lv[b].cr || !lv[b].v) return false;
    }
    auto slot = [&](int j) { return first + (j < n ? j : 0); };  // unused texture slots still need a texture

    // warp: the deltas of the whole group at level 0
    {
      const WarpU wu = {{(float)env.width, (float)env.height}, (uint32_t)T.taps, (uint32_t)n, (uint32_t)(first - 1), g1.h, dBias, 0};
      group("warp");
      [enc setTexture:env.frames[0].tex atIndex:0];
      for (int j = 0; j < 4; ++j) [enc setTexture:env.frames[slot(j)].tex atIndex:(NSUInteger)(1 + j)];
      [enc setTexture:flowV atIndex:5];
      [enc setTexture:lv[0].y atIndex:6];
      [enc setTexture:lv[0].cb atIndex:7];
      [enc setTexture:lv[0].cr atIndex:8];
      [enc setTexture:lv[0].v atIndex:9];
      [enc setBytes:&wu length:sizeof(wu) atIndex:0];
      run(warp_, env.width, env.height);
    }

    MergeU u[kMaxLevels];
    for (int b = 0; b <= MT; ++b) {
      u[b] = mu[b];
      for (int j = 0; j < 4; ++j) u[b].mask[j] = j < n ? 1.0f : 0.0f;
      u[b].hasPrev = g > 0 ? 1u : 0u;
      u[b].first = (uint32_t)(first - 1);
    }

    // the delta pyramid
    for (int b = 1; b <= MT; ++b) {
      group(kLevelLabel[1][b]);
      [enc setTexture:lv[b - 1].y atIndex:0];
      [enc setTexture:lv[b - 1].cb atIndex:1];
      [enc setTexture:lv[b - 1].cr atIndex:2];
      [enc setTexture:lv[b - 1].v atIndex:3];
      [enc setTexture:lv[b].y atIndex:4];
      [enc setTexture:lv[b].cb atIndex:5];
      [enc setTexture:lv[b].cr atIndex:6];
      [enc setTexture:lv[b].v atIndex:7];
      [enc setBytes:&u[b - 1] length:sizeof(MergeU) atIndex:0];  // of the level that is read
      run(down_, sz[b].w, sz[b].h);
    }

    // per level, coarse to fine: energy, then merge
    id<MTLTexture> rPrev = nil;
    for (int b = MT; b >= 0; --b) {
      const Level& l = lv[b];
      const Level& up = lv[b < MT ? b + 1 : b];  // not read at the top level
      id<MTLTexture> eY = env.temp(fHalf, sz[b].w, sz[b].h);
      id<MTLTexture> eC = env.temp(fHalf, sz[b].w, sz[b].h);
      if (!eY || !eC) return false;
      group(kLevelLabel[2][b]);
      [enc setTexture:l.y atIndex:0];
      [enc setTexture:l.cb atIndex:1];
      [enc setTexture:l.cr atIndex:2];
      [enc setTexture:up.y atIndex:3];
      [enc setTexture:up.cb atIndex:4];
      [enc setTexture:up.cr atIndex:5];
      [enc setTexture:pyr[0][b == 0 ? 1 : b] atIndex:6];
      [enc setTexture:eY atIndex:7];
      [enc setTexture:eC atIndex:8];
      [enc setBytes:&u[b] length:sizeof(MergeU) atIndex:0];
      run(energy_, sz[b].w, sz[b].h);

      // At radius >= 3 the accumulator and final merge share a level.
      // Keep kernel attribution separate from the temporal timeline view.
      group(kernelTraceEnabled() ? (!last ? "temporal_merge_acc" : b > 0 ? "temporal_merge_fin" : "temporal_merge_fin0")
                                 : kLevelLabel[3][b]);
      [enc setTexture:l.y atIndex:0];
      [enc setTexture:l.cb atIndex:1];
      [enc setTexture:l.cr atIndex:2];
      [enc setTexture:l.v atIndex:3];
      [enc setTexture:up.y atIndex:4];
      [enc setTexture:up.cb atIndex:5];
      [enc setTexture:up.cr atIndex:6];
      [enc setTexture:eY atIndex:7];
      [enc setTexture:eC atIndex:8];
      [enc setTexture:rPrev ? rPrev : eY atIndex:9];
      [enc setTexture:gate atIndex:10];
      [enc setTexture:acc1[b] ? acc1[b] : eY atIndex:11];
      [enc setTexture:acc2[b] ? acc2[b] : eY atIndex:12];
      [enc setBytes:&u[b] length:sizeof(MergeU) atIndex:0];
      id<MTLTexture> rOut = nil;
      if (b > 0 || !last) {
        rOut = env.temp(fHalf, sz[b].w, sz[b].h);
        if (!rOut) return false;
      }
      if (!last) {
        id<MTLTexture> n1 = env.temp(kWorkingFormat, sz[b].w, sz[b].h);
        id<MTLTexture> n2 = env.temp(kWorkingFormat, sz[b].w, sz[b].h);
        if (!n1 || !n2) return false;
        [enc setTexture:n1 atIndex:13];
        [enc setTexture:n2 atIndex:14];
        [enc setTexture:rOut atIndex:15];
        run(mergeAcc_, sz[b].w, sz[b].h);
        if (acc1[b]) env.recycle(acc1[b]);
        if (acc2[b]) env.recycle(acc2[b]);
        acc1[b] = n1;
        acc2[b] = n2;
      } else if (b > 0) {
        dOut[b] = env.temp(kWorkingFormat, sz[b].w, sz[b].h);
        if (!dOut[b]) return false;
        [enc setTexture:b < MT ? dOut[b + 1] : eY atIndex:13];
        [enc setTexture:dOut[b] atIndex:14];
        [enc setTexture:rOut atIndex:15];
        run(mergeFin_, sz[b].w, sz[b].h);
      } else {
        image = env.tempWorking();
        count = env.temp(MTLPixelFormatR16Float, env.width, env.height);
        if (!image || !count) return false;
        [enc setTexture:dOut[1] atIndex:13];
        [enc setTexture:env.frames[0].tex atIndex:14];
        [enc setTexture:image atIndex:15];
        [enc setTexture:count atIndex:16];
        run(mergeFin0_, env.width, env.height);
      }
      env.recycle(eY);
      env.recycle(eC);
      if (rPrev) env.recycle(rPrev);
      rPrev = rOut;
      if (last) {
        if (acc1[b]) env.recycle(acc1[b]);
        if (acc2[b]) env.recycle(acc2[b]);
        if (b < MT && dOut[b + 1]) env.recycle(dOut[b + 1]);
      }
    }
    if (rPrev) env.recycle(rPrev);
    for (int b = 0; b <= MT; ++b) {
      env.recycle(lv[b].y);
      env.recycle(lv[b].cb);
      env.recycle(lv[b].cr);
      env.recycle(lv[b].v);
    }
  }

  if (p.view == OD_VIEW_MOTION) {
    id<MTLTexture> view = env.tempWorking();
    if (!view) return false;
    const ViewU vu = {1.0f / (float)nb, g1.h, 0, 0};
    group("view");
    [enc setTexture:count atIndex:0];
    [enc setTexture:flowV atIndex:1];
    [enc setTexture:view atIndex:2];
    [enc setBytes:&vu length:sizeof(vu) atIndex:0];
    run(viewMotion_, env.width, env.height);
    image = view;
  }

  trace_.finish(env);
  out->image = image;
  out->count = count;
  out->gate = gate;  // read back by the pipeline (cut detection for the profile)
  out->framesUsed = nb + 1;
  return !env.failed();
}


}  // namespace

std::unique_ptr<TemporalStage> makeTemporalStage(Context& ctx, std::string* err) {
  auto s = std::make_unique<Temporal>();
  if (!s->init(ctx, err)) return nullptr;
  return s;
}

}  // namespace od
