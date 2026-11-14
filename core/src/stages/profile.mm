// SPDX-License-Identifier: MIT
// Automatic noise profile. Contract: stage.h; shaders: profile_measure.metal.
// The GPU measures 32x32 tiles: Y, Cb and Cr energy in four B3 a-trous
// bands, full-band energy, luminance, gradient coherence and dead pixels.
// The CPU fits a noise floor over brightness, rejecting structure and clipping.
// Spatial and frame-difference estimates are combined; the difference assumes
// independent noise and is checked against the spatial estimate for chroma.
// Fine bands anchor the coarse-band ratios to avoid fitting soft structure.
// The fit uses up to 43 brightness nodes, projected onto eight uniform knots.
// A confidence-weighted interquartile mean of log sigma over 24 estimates
// smooths the result. Finest-grain mode retains previously measured knots when
// no estimate in the ring supports them; whole-picture mode aggregates all knots.
// New-shot detection and explicit resets restart the aggregate.
// Playback throttles stable profiling; Full measures every frame.
// The optional direct coarse-chroma estimator is defined in coarse_profile.h.
#include "stage.h"

#import <CoreFoundation/CoreFoundation.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace od {
namespace {

constexpr uint32_t kTile = 32;          // tile edge in pixels (fixed by the shader)
constexpr uint32_t kRecordFloats = 24;  // PROFILE_RECORD_FLOATS: one record (see the shader)
constexpr uint32_t kTileFloats = 48;    // PROFILE_TILE_FLOATS: spatial record + temporal record
constexpr int kL = OD_LEVELS;
constexpr int kFeat = 15;  // 12 band energies (c * 4 + b) + 3 full-band (12 + c)

// ---- tuning constants ------------------------------------------------------
constexpr float kZeroFracMax = 0.02f;   // more dead pixels than this: clipped tile (or duplicated frame)
constexpr float kCoherenceMax = 0.25f;  // gradient coherence above this: oriented structure
// Effective degrees of freedom of a tile's band energy for white noise,
// measured from the spread of the tile energies on a uniform grey frame
// (tests/profile, OD_PROFILE_CALIBRATE=1 OD_PROFILE_DEBUG=3: P16/P50 =
// 0.954, 0.900, 0.808, 0.676 for bands 0..3).
constexpr double kNu[4] = {950.0, 190.0, 46.0, 15.0};
constexpr double kClipSigmas = 3.0;  // one-sided clip, in standard deviations of a tile energy
constexpr int kIterations = 3;
constexpr double kChromaClip = 2.5;    // chroma tiles above this multiple of the model variance are structure
constexpr double kChromaRedder = 2.0;  // largest sigma ratio of chroma band b to band b - 1 (b = 2, 3); luma 1.25
constexpr double kMaxRise = 2.5;       // largest believable rise of sigma per knot interval of luminance
constexpr double kMaxRatioRise = 1.4;  // the same for the ratio of one band's sigma to its reference band's
constexpr double kMaxRiseDark = 4.0;   // the same towards black below kDarkEnd (log footage: up to 3.6 per
                                       // interval where read noise dominates, S-Log3 and LogC alike)
constexpr double kDarkEnd = 0.40;      // luminance below which a node may rise that steeply towards black
constexpr double kLocalRange = 1.1;    // a knot with nodes of its own stays within this factor of them
constexpr int kMinNodeTiles = 24;        // tiles a node needs for the initial ranking
constexpr double kMinNodeWeight = 20.0;  // candidate weight a luminance node needs to be fitted (finest grid)
constexpr int kMinTiles = 32;            // usable tiles a frame needs for any estimate
constexpr int kGroupTiles = 120;         // tiles of one luminance group used per node estimate
constexpr float kSigmaFloor = 1e-6f;
// Recovery resolution tests, in normalized working-space variance. These
// correspond to 0.032 / 0.051 full-range ten-bit codes of sigma. They do
// not impose a noise floor: constant channels still remain constant.
constexpr float kChromaResolvedVar = 1e-9f;
constexpr double kChromaCollapsedVar = 2.5e-9;
constexpr float kLowConfidenceScale = 0.4f;  // sigma multiplier at confidence <= 0.1 (1 from 0.5 up)

struct Grid {
  uint32_t originX = 0, originY = 0, stride = 32, tilesW = 0, tilesH = 0;
  uint32_t count() const { return tilesW * tilesH; }
};

// Tiles stay 32 px away from the frame border (c_4 reaches 30 px, and hosts
// like to leave junk at the edges). Smaller frames get overlapping tiles so
// every luminance level still has enough of them.
Grid gridFor(uint32_t w, uint32_t h) {
  Grid g;
  const uint32_t margin = (w >= 512 && h >= 512) ? 32 : 0;
  g.originX = g.originY = margin;
  g.stride = ((uint64_t)w * h >= 4000000ull) ? 32 : 16;
  if (w >= 2 * margin + kTile && h >= 2 * margin + kTile) {
    g.tilesW = (w - 2 * margin - kTile) / g.stride + 1;
    g.tilesH = (h - 2 * margin - kTile) / g.stride + 1;
  }
  return g;
}

struct TileParamsGPU {
  uint32_t originX, originY, stride, tilesW, tilesH, base, pad0, pad1;
};
struct ViewParamsGPU {
  uint32_t originX, originY, stride, tilesW, tilesH, haveProfile;
  float thr0, thr1, coherenceMax, deadMax, pad0, pad1;
  float sig0[kL];
  float sig1[kL];
};

// Statistics buffers are about 1.4 MB; reuse them. One pool per context
// (Context::shared), not per session: an idle session then holds none, and
// the whole process at most four. Thread-safe: a buffer comes back from
// whichever thread drops the last copy of a measurement.
class BufferPool {
 public:
  explicit BufferPool(id<MTLDevice> device) : device_(device) {}
  id<MTLBuffer> acquire(NSUInteger length) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (size_t i = 0; i < idle_.size(); ++i) {
        if (idle_[i].length >= length && idle_[i].length <= 2 * length) {
          id<MTLBuffer> b = idle_[i];
          idle_.erase(idle_.begin() + (long)i);
          return b;
        }
      }
    }
    return [device_ newBufferWithLength:length options:MTLResourceStorageModeShared];
  }
  void release(id<MTLBuffer> b) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (idle_.size() >= 4) idle_.erase(idle_.begin());
    idle_.push_back(b);
  }

 private:
  id<MTLDevice> device_;
  std::mutex mutex_;
  std::vector<id<MTLBuffer>> idle_;
};

#include "coarse_profile.h"

struct MeasureInfo {
  CoarseMeasurement coarse;
  Grid grid;
  bool pair = false;
  bool chromaRecovery = false;
  std::shared_ptr<BufferPool> pool;
  id<MTLBuffer> buffer = nil;
  ~MeasureInfo() {
    if (pool && buffer) pool->release(buffer);
  }
};

// OD_PROFILE_DEBUG=1 prints both fits of every measurement to stderr, 2 adds
// the luminance nodes, 3 the measured degrees of freedom, 4 the luma nodes
// before the texture test (dropRisers), 5 every projection onto the knots.
int debugLevel() {
  static const int level = [] {
    const char* e = getenv("OD_PROFILE_DEBUG");
    return e ? atoi(e) : 0;
  }();
  return level;
}
bool debugEnabled() { return debugLevel() > 0; }
// OD_PROFILE_MODE=S|T restricts resolve() to the spatial or the temporal
// estimator and makes smooth() pass every estimate straight through, so the
// tests can look at one estimator and one frame at a time.
char debugMode() {
  static const char mode = [] {
    const char* e = getenv("OD_PROFILE_MODE");
    return e ? *e : '\0';
  }();
  return mode;
}

// ---- CPU estimation --------------------------------------------------------
// The floor is fitted on a fine luminance grid (43 nodes, 6 per knot
// interval) and projected onto the 8 knots afterwards. Fitting the 8 knots
// directly is biased low: noise can change by a factor 2 inside one knot
// interval (measured on the S-Log3 test clips), and a floor search then
// settles on the quietest part of the interval and clips the rest away as if
// it were texture.
// When a frame has too few flat tiles for that (small frames), the grid is
// coarsened: 6, 2, 1 nodes per interval, and finally one level for all.
constexpr int kMaxSub = 6;
constexpr int kMaxNodes = kMaxSub * (kL - 1) + 1;

struct Tile {
  float y;
  uint32_t index = 0; // original GPU tile, for read-only analysis
  float e[kFeat];
  float kurt;  // kurtosis of d_0.Y in the tile (3 for Gaussian noise)
};

struct Fit {
  bool ok = false;
  int sub = kMaxSub;                // nodes per knot interval used for this fit
  int nodes = kMaxNodes;            // sub * (kL - 1) + 1
  double var[kFeat][kMaxNodes];     // noise variance at the nodes
  double weight[kFeat][kMaxNodes];  // candidate weight behind each node (0 = interpolated)
  double knotVar[kFeat][kL];     // projected onto the knots
  bool knot[kL] = {};            // knot has luma candidates within half a knot interval
  int usable = 0;
  int candidates = 0;
  int validKnots = 0;
  double lowSpread0 = 0, lowSpread1 = 0;    // 1 - P16/P50 of the candidates' normalised energy
  double floorRatio0 = 0, floorRatio1 = 0;  // P10/P2.5 of all usable tiles' normalised energy
  double candFraction = 0;
  double kurtosis = 0;  // median over the candidates
  double droppedFraction = 0;  // share of the luma node weight dropped as texture
  bool oneLevel = false;       // fitted on one level for every luminance: no knot measured as such
};

// Runs f(0) .. f(n - 1) on the calling thread and on workers, and returns
// when all are done. The fit is a few milliseconds of independent work per
// feature, and it can be the render thread that waits for it (first frame of
// a session, blocking renders): spread over the cores it takes a quarter of
// the time and gives the same numbers, as every task has its own scratch
// memory and writes its own part of the result.
template <class F>
void parallelFor(size_t n, const F& f) {
  if (n <= 1) {
    if (n == 1) f(0);
    return;
  }
  const F* fp = &f;
  dispatch_apply(n, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
    (*fp)(i);
  });
}

double medianOf(std::vector<double>& v) {
  if (v.empty()) return 0;
  size_t k = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + (long)k, v.end());
  double hi = v[k];
  if (v.size() % 2) return hi;
  double lo = *std::max_element(v.begin(), v.begin() + (long)k);
  return 0.5 * (lo + hi);
}

double quantileOf(std::vector<double>& v, double q) {
  if (v.empty()) return 0;
  double pos = q * (double)(v.size() - 1);
  size_t k = (size_t)pos;
  std::nth_element(v.begin(), v.begin() + (long)k, v.end());
  double a = v[k];
  if (k + 1 >= v.size()) return a;
  double b = *std::min_element(v.begin() + (long)k + 1, v.end());
  return a + (pos - (double)k) * (b - a);
}

// Weighted quantile of (value, weight) pairs: the smallest value whose
// cumulative weight reaches q * total. By selection, not by sorting: this
// runs per node, per feature and per iteration, possibly on the render
// thread, and sorting made a fit take 8 ms instead of 2. Reorders v.
double weightedQuantile(std::vector<std::pair<double, double>>& v, double total, double q) {
  if (v.empty()) return 1.0;
  size_t lo = 0, hi = v.size();  // the answer is in [lo, hi)
  double target = q * total;     // cumulative weight wanted, counted from lo
  while (hi - lo > 8) {
    // median-of-three pivot
    const size_t mid = lo + (hi - lo) / 2;
    double a = v[lo].first, b = v[mid].first, c = v[hi - 1].first;
    const double pivot = std::max(std::min(a, b), std::min(std::max(a, b), c));
    auto midIt = std::partition(v.begin() + (long)lo, v.begin() + (long)hi,
                                [pivot](const std::pair<double, double>& p) { return p.first < pivot; });
    const size_t m = (size_t)(midIt - v.begin());
    double wl = 0;
    for (size_t k = lo; k < m; ++k) wl += v[k].second;
    if (wl >= target) {
      if (m == lo) break;  // (cannot happen: wl == 0 < target unless target <= 0)
      hi = m;
    } else {
      // values equal to the pivot: move them to the front of the right part
      auto eqIt = std::partition(midIt, v.begin() + (long)hi,
                                 [pivot](const std::pair<double, double>& p) { return p.first <= pivot; });
      double we = 0;
      for (auto it = midIt; it != eqIt; ++it) we += it->second;
      if (wl + we >= target) return pivot;
      target -= wl + we;
      lo = (size_t)(eqIt - v.begin());
      if (lo >= hi) return pivot;
    }
  }
  std::sort(v.begin() + (long)lo, v.begin() + (long)hi);
  double accw = 0;
  for (size_t k = lo; k < hi; ++k) {
    accw += v[k].second;
    if (accw >= target) return v[k].first;
  }
  return v[hi - 1].first;
}

// Quantile of a chi-square mean with nu degrees of freedom, relative to its
// mean (Wilson-Hilferty). z = standard normal quantile: 0 for the median,
// -0.6745 for the lower quartile.
double chiQuantile(double nu, double z) {
  double t = 1.0 - 2.0 / (9.0 * nu) + z * std::sqrt(2.0 / (9.0 * nu));
  return t * t * t;
}

int bandOf(int f) { return f < 12 ? f % 4 : -1; }
int channelOf(int f) { return f < 12 ? f / 4 : f - 12; }

// Fill nodes without their own estimate: linear in log(variance) between
// valid nodes, constant beyond the outermost one. Never zero.
void fillNodes(double* var, const bool* valid, int nodes) {
  int first = -1, last = -1;
  for (int l = 0; l < nodes; ++l) {
    if (valid[l]) {
      if (first < 0) first = l;
      last = l;
    }
  }
  if (first < 0) return;
  for (int l = 0; l < first; ++l) var[l] = var[first];
  for (int l = last + 1; l < nodes; ++l) var[l] = var[last];
  int a = first;
  for (int l = first + 1; l <= last; ++l) {
    if (!valid[l]) continue;
    for (int k = a + 1; k < l; ++k) {
      double t = (double)(k - a) / (double)(l - a);
      var[k] = std::exp(std::log(std::max(var[a], 1e-30)) * (1 - t) + std::log(std::max(var[l], 1e-30)) * t);
    }
    a = l;
  }
}

inline void nodeOf(float y, int nodes, int* l0, double* w) {
  double t = std::min(1.0, std::max(0.0, (double)y)) * (nodes - 1);
  int i = std::min((int)t, nodes - 2);
  *l0 = i;
  *w = t - i;
}

// Model variance at luminance (l0, w): sigma is linear between nodes, as it
// is between knots for the consumers (od_sigma_at).
inline double modelAt(const double* var, int l0, double w) {
  double s = std::sqrt(var[l0]) * (1 - w) + std::sqrt(var[l0 + 1]) * w;
  return std::max(s * s, 1e-30);
}

// Least-squares projection of the node sigmas onto the 8 knots (piecewise
// linear in sigma, relative errors, weighted by the candidates behind each
// node). Two small penalties make it well posed and decide what happens
// where the frame has no flat tile: between supported stretches the curve
// is interpolated, beyond them it levels off. Never zero.
void projectToKnots(const double* var, const double* weight, int sub, double* knotVar) {
  const int nodes = sub * (kL - 1) + 1;
  double wsum = 0, smean = 0;
  int used = 0;
  for (int j = 0; j < nodes; ++j) {
    if (weight[j] <= 0) continue;
    wsum += weight[j];
    smean += weight[j] * std::sqrt(var[j]);
    ++used;
  }
  if (wsum <= 0) {
    // nothing supported: keep whatever the nodes hold (constant fill)
    for (int k = 0; k < kL; ++k) knotVar[k] = var[k * sub];
    return;
  }
  smean /= wsum;
  // Which knot intervals hold data.
  bool supported[kL - 1] = {};
  for (int j = 0; j < nodes; ++j) {
    if (weight[j] <= 0) continue;
    if (j % sub == 0) {
      if (j / sub < kL - 1) supported[j / sub] = true;
      if (j / sub > 0) supported[j / sub - 1] = true;
    } else {
      supported[j / sub] = true;
    }
  }
  // Every knot stays within a factor of a reference: by default 1.6 of the
  // nearest node that has data (a line fitted to a few nodes at one end of an
  // interval can swing far at the other end). A knot with data of its own
  // (nodes next to it carrying a fifth of the mean node weight or more)
  // stays within kLocalRange of their weighted mean. A
  // straight line in sigma cannot follow a curve that falls steeply and then
  // levels off inside one interval (the toe of log footage): the fit bent
  // the knot at the flat end down to follow the steep part, and every knot
  // beyond the data levelled off at that value. On a real S-Log3 clip that
  // put knot 2 and all above it at 0.55 of the noise the nodes there read.
  // Held to its own nodes, the knot is right and the interval in between
  // errs high, towards a little more smoothing there. A knot outside its
  // range is pinned to the edge of it and the curve solved again, so the
  // knots beyond the data level off at the pinned value.
  double ref[kL], range[kL];
  for (int k = 0; k < kL; ++k) {
    int best = -1;
    for (int j = 0; j < nodes; ++j)
      if (weight[j] > 0 && (best < 0 || std::abs(j - k * sub) < std::abs(best - k * sub))) best = j;
    ref[k] = std::sqrt(var[best]);
    range[k] = 1.6;
    if (sub < 2) continue;
    // The nodes next to the knot (a sixth of an interval either side on the
    // finest grid): a wider window reaches across the step where the toe
    // meets the log segment and reads the knot low.
    const int reach = std::max(1, sub / 6);
    double lw = 0, ly = 0;
    for (int j = 0; j < nodes; ++j) {
      const int d = std::abs(j - k * sub);
      if (weight[j] <= 0 || d > reach) continue;
      const double h = weight[j] * (1.0 - d / (reach + 1.0));
      lw += h;
      ly += h * 0.5 * std::log(std::max(var[j], 1e-30));
    }
    if (lw >= 0.2 * wsum / used) {
      ref[k] = std::exp(ly / lw);
      range[k] = kLocalRange;
    }
  }
  double s[kL];
  double robust[kMaxNodes];
  for (int j = 0; j < nodes; ++j) robust[j] = 1.0;
  bool pinned[kL] = {};
  double pinTo[kL] = {};
  const double scale = 1.0 / (smean * smean);
  // Iteratively reweighted: a thinly populated node that sits far off the
  // curve of its neighbours (a few textured tiles) must not bend it.
  for (int round = 0; round < 7; ++round) {
    double A[kL][kL] = {}, b[kL] = {};
    for (int j = 0; j < nodes; ++j) {
      if (weight[j] <= 0) continue;
      const double sj = std::sqrt(var[j]);
      const int i = std::min(j / sub, kL - 2);
      const double f = (double)(j - i * sub) / sub;
      const double w = robust[j] * weight[j] / wsum / (sj * sj);
      const double h0 = 1 - f, h1 = f;
      A[i][i] += w * h0 * h0;
      A[i][i + 1] += w * h0 * h1;
      A[i + 1][i] += w * h0 * h1;
      A[i + 1][i + 1] += w * h1 * h1;
      b[i] += w * sj * h0;
      b[i + 1] += w * sj * h1;
    }
    // Slope penalty: weak where there is data, stiff where there is none (the
    // curve levels off instead of continuing a slope into the unknown).
    // Curvature penalty: only across two intervals that both hold data.
    for (int k = 0; k + 1 < kL; ++k) {  // (s[k+1] - s[k])^2
      const double l1 = (supported[k] ? 2e-3 : 1e2) * scale;
      A[k][k] += l1;
      A[k + 1][k + 1] += l1;
      A[k][k + 1] -= l1;
      A[k + 1][k] -= l1;
    }
    for (int k = 1; k + 1 < kL; ++k) {  // (s[k-1] - 2 s[k] + s[k+1])^2
      if (!supported[k - 1] || !supported[k]) continue;
      const double l2 = 1e-2 * scale;
      const double c[3] = {1, -2, 1};
      for (int a = 0; a < 3; ++a)
        for (int d = 0; d < 3; ++d) A[k - 1 + a][k - 1 + d] += l2 * c[a] * c[d];
    }
    for (int k = 0; k < kL; ++k) {  // pins, far stiffer than anything else
      if (!pinned[k]) continue;
      const double lp = 1e4 * scale;
      A[k][k] += lp;
      b[k] += lp * pinTo[k];
    }
    // Gaussian elimination with partial pivoting (8 x 8, symmetric positive definite).
    for (int c = 0; c < kL; ++c) {
      int piv = c;
      for (int r = c + 1; r < kL; ++r)
        if (std::fabs(A[r][c]) > std::fabs(A[piv][c])) piv = r;
      if (piv != c) {
        for (int k = 0; k < kL; ++k) std::swap(A[c][k], A[piv][k]);
        std::swap(b[c], b[piv]);
      }
      const double d = A[c][c];
      if (std::fabs(d) < 1e-300) continue;
      for (int r = c + 1; r < kL; ++r) {
        const double m = A[r][c] / d;
        for (int k = c; k < kL; ++k) A[r][k] -= m * A[c][k];
        b[r] -= m * b[c];
      }
    }
    for (int c = kL - 1; c >= 0; --c) {
      double v = b[c];
      for (int k = c + 1; k < kL; ++k) v -= A[c][k] * s[k];
      s[c] = std::fabs(A[c][c]) > 1e-300 ? v / A[c][c] : smean;
    }
    // Cauchy weights on the relative residual, tighter above the curve than
    // below it: what is left of texture can only lift a node.
    for (int j = 0; j < nodes; ++j) {
      if (weight[j] <= 0) continue;
      const double sj = std::sqrt(var[j]);
      const int i = std::min(j / sub, kL - 2);
      const double f = (double)(j - i * sub) / sub;
      const double fitv = std::max(s[i] * (1 - f) + s[i + 1] * f, 1e-12);
      const double r = std::log(sj / fitv) / (sj > fitv ? 0.15 : 0.40);
      robust[j] = 1.0 / (1.0 + r * r);
    }
    // After the fifth round (the robust weights have settled), pin the knots
    // that left their range and solve again; once pinned, a knot stays
    // pinned. Without a pin the result is that of the fifth round.
    if (round >= 4) {
      bool any = false;
      for (int k = 0; k < kL; ++k) {
        if (pinned[k] || range[k] >= 1.6) continue;
        if (!std::isfinite(s[k]) || s[k] < ref[k] / range[k] || s[k] > ref[k] * range[k]) {
          pinned[k] = any = true;
          pinTo[k] = std::isfinite(s[k]) && s[k] > ref[k] ? ref[k] * range[k] : ref[k] / range[k];
        }
      }
      if (!any) break;
    }
  }
  for (int k = 0; k < kL; ++k) {
    const double v = std::min(std::max(std::isfinite(s[k]) ? s[k] : ref[k], ref[k] / range[k]), ref[k] * range[k]);
    knotVar[k] = v * v;
  }
  if (debugLevel() >= 5) {
    fprintf(stderr, "[profile] project:");
    for (int k = 0; k < kL; ++k)
      fprintf(stderr, " %.5f(ref %.5f x%.2f%s)", std::sqrt(knotVar[k]), ref[k], range[k], pinned[k] ? " pin" : "");
    fprintf(stderr, "\n");
  }
}

bool fitFloorAt(const std::vector<Tile>& tiles, int sub, bool oneLevel, Fit* fit, bool recover = false) {
  const int n = (int)tiles.size();
  const int nodes = sub * (kL - 1) + 1;
  *fit = Fit();
  fit->sub = sub;
  fit->nodes = nodes;
  fit->usable = n;
  if (n < kMinTiles) return false;
  // Coarser grids are the fallback for frames with few flat tiles: ask less.
  const double minWeight = oneLevel ? 3.0 : sub >= kMaxSub ? kMinNodeWeight : sub == 2 ? 12.0 : 8.0;
  const int minStart = oneLevel ? 8 : sub >= kMaxSub ? kMinNodeTiles : 12;

  // One-sided clip per band: this many standard deviations of a tile energy.
  double clip[4];
  for (int b = 0; b < 4; ++b) clip[b] = 1.0 + kClipSigmas * std::sqrt(2.0 / kNu[b]);

  // ---- start: per node, the tiles lowest in both fine luma bands ----
  std::vector<int> byNode[kMaxNodes];
  for (int i = 0; i < n; ++i) {
    int l = (int)std::lround(std::min(1.0f, std::max(0.0f, tiles[(size_t)i].y)) * (nodes - 1));
    byNode[l].push_back(i);
  }
  bool valid[kMaxNodes] = {};
  std::vector<int> score((size_t)n, 0);
  auto startFrom = [&](const std::vector<int>& idx, int l) {
    const int m = (int)idx.size();
    std::vector<int> r0(idx), r1(idx);
    std::sort(r0.begin(), r0.end(), [&](int a, int b) { return tiles[(size_t)a].e[0] < tiles[(size_t)b].e[0]; });
    std::sort(r1.begin(), r1.end(), [&](int a, int b) { return tiles[(size_t)a].e[1] < tiles[(size_t)b].e[1]; });
    for (int k = 0; k < m; ++k) score[(size_t)r0[(size_t)k]] = k;
    for (int k = 0; k < m; ++k) score[(size_t)r1[(size_t)k]] = std::max(score[(size_t)r1[(size_t)k]], k);
    std::vector<int> order(idx);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return score[(size_t)a] < score[(size_t)b]; });
    int take = 0;
    while (take < m && score[(size_t)order[(size_t)take]] < (int)(0.3 * m)) ++take;
    take = std::max(take, std::min(m, 8));
    std::vector<double> v;
    for (int f = 0; f < kFeat; ++f) {
      v.clear();
      for (int k = 0; k < take; ++k) {
        const Tile& t = tiles[(size_t)order[(size_t)k]];
        if (recover && channelOf(f) > 0 && t.e[channelOf(f) * 4] < kChromaResolvedVar) continue;
        v.push_back(t.e[f]);
      }
      fit->var[f][l] = std::max(medianOf(v), 1e-30);
    }
  };
  // The nodes' tile lists are disjoint, so are the entries of `score` and
  // the columns of fit->var they write.
  int startNodes[kMaxNodes];
  int started = 0;
  for (int l = 0; l < nodes; ++l) {
    if ((int)byNode[l].size() >= minStart) {
      startNodes[started++] = l;
      valid[l] = true;
    }
  }
  parallelFor((size_t)started, [&](size_t k) { startFrom(byNode[startNodes[k]], startNodes[k]); });
  if (!started) {
    // Tiles spread too thinly over the luminance range: one start value for all.
    std::vector<int> all((size_t)n);
    for (int i = 0; i < n; ++i) all[(size_t)i] = i;
    startFrom(all, 0);
    for (int f = 0; f < kFeat; ++f)
      for (int l = 1; l < nodes; ++l) fit->var[f][l] = fit->var[f][0];
  } else {
    for (int f = 0; f < kFeat; ++f) fillNodes(fit->var[f], valid, nodes);
  }

  // ---- iterate: clip against the model from above, refit ----
  std::vector<double> ratio((size_t)n * kFeat);
  std::vector<int> l0s((size_t)n);
  std::vector<double> ws((size_t)n);
  std::vector<uint8_t> lumaCand((size_t)n);
  for (int i = 0; i < n; ++i) nodeOf(tiles[(size_t)i].y, nodes, &l0s[(size_t)i], &ws[(size_t)i]);
  for (int f = 0; f < kFeat; ++f)
    for (int l = 0; l < nodes; ++l) fit->weight[f][l] = 0;
  // Normalised energies of the fine bands (the only ones the selection
  // uses) against the current node model, and the luma candidates.
  auto classify = [&]() {
    static const int fine[6] = {0, 1, 4, 5, 8, 9};
    double sig[6][kMaxNodes];
    for (int k = 0; k < 6; ++k)
      for (int l = 0; l < nodes; ++l) sig[k][l] = std::sqrt(fit->var[fine[k]][l]);
    int cand = 0;
    for (int i = 0; i < n; ++i) {
      const Tile& t = tiles[(size_t)i];
      double* r = &ratio[(size_t)i * kFeat];
      const int l0 = l0s[(size_t)i];
      const double w = ws[(size_t)i];
      for (int k = 0; k < 6; ++k) {
        const double m = sig[k][l0] * (1 - w) + sig[k][l0 + 1] * w;
        r[fine[k]] = t.e[fine[k]] / std::max(m * m, 1e-30);
      }
      lumaCand[(size_t)i] = r[0] < clip[0] && r[1] < clip[1];
      cand += lumaCand[(size_t)i];
    }
    fit->candidates = cand;
  };
  // Tiles grouped by their lower node, so that the tiles with weight at node
  // l are a contiguous run (groups l - reach .. l + reach - 1).
  std::vector<int> order((size_t)n);
  int groupStart[kMaxNodes + 1] = {};
  for (int i = 0; i < n; ++i) ++groupStart[l0s[(size_t)i] + 1];
  for (int l = 0; l < nodes; ++l) groupStart[l + 1] += groupStart[l];
  {
    int fill[kMaxNodes];
    for (int l = 0; l < nodes; ++l) fill[l] = groupStart[l];
    for (int i = 0; i < n; ++i) order[(size_t)fill[l0s[(size_t)i]]++] = i;
  }
  // A crowded luminance level does not need all its tiles: beyond
  // kGroupTiles per group every step-th one is used, with its weight
  // multiplied by the step. (A 4K frame of sky puts thousands of tiles in a
  // few groups; the fit then cost 4 ms instead of 1.5, on a thread that can
  // be the render thread.) Sparse levels keep every tile.
  int groupStep[kMaxNodes];
  for (int l = 0; l < nodes; ++l) groupStep[l] = std::max(1, (groupStart[l + 1] - groupStart[l] + kGroupTiles - 1) / kGroupTiles);
  // Scratch memory of one feature's fit: every feature is fitted by its own
  // task (parallelFor).
  struct Scratch {
    std::vector<double> val;                        // per tile, < 0 = not used for this feature
    std::vector<std::pair<double, double>> pairs;  // (value, weight) of the tiles around one node
    bool valid[kMaxNodes];
  };
  std::vector<Scratch> scratchOf((size_t)kFeat);
  // (value, weight) of the tiles around node l, weight = 1 - distance / reach
  // in nodes, into sc->pairs; returns the weight sum.
  auto gather = [&](Scratch* sc, int l, int reach) {
    sc->pairs.clear();
    double wsum = 0;
    for (int g = std::max(0, l - reach); g <= std::min(nodes - 2, l + reach - 1); ++g) {
      const int step = groupStep[g];
      for (int k = groupStart[g]; k < groupStart[g + 1]; k += step) {
        const int i = order[(size_t)k];
        if (sc->val[(size_t)i] < 0) continue;
        const double w = step * (1.0 - std::fabs(g + ws[(size_t)i] - l) / reach);
        if (w <= 1e-9) continue;
        sc->pairs.emplace_back(sc->val[(size_t)i], w);
        wsum += w;
      }
    }
    return wsum;
  };
  static const int kFine[6] = {0, 1, 4, 5, 8, 9};  // bands 0 and 1 of Y, Cb, Cr; the others come afterwards
  for (int it = 0; it < kIterations; ++it) {
    classify();
    parallelFor(6, [&](size_t k) {
      const int f = kFine[k];
      const int c = channelOf(f), b = bandOf(f);
      Scratch* sc = &scratchOf[(size_t)f];
      sc->val.resize((size_t)n);
      // Luma: the median of the candidates. Chroma: the candidates are the
      // tiles that are flat in luma, minus gross chroma structure, and the
      // estimate is their 40th percentile. Chroma noise depends on the colour
      // as well as on the luminance, so tiles of one luminance do not share
      // one level; a tight clip like the luma one would settle on the
      // quietest colour (measured: 15-25 % low on the sky clips).
      // Recovery: codec-flat islands are not samples of the noisy population.
      // Use absolute tile energy, not an update relative to the collapsed
      // model (which otherwise traps even nonzero observations near zero).
      // Luma selection and the robust lower quantile still reject structure.
      const double z = c > 0 ? -0.2533 : 0.0;
      const double q = c > 0 ? 0.40 : 0.50;
      const double corr = chiQuantile(kNu[b], z);
      for (int i = 0; i < n; ++i) {
        const double* r = &ratio[(size_t)i * kFeat];
        const bool use = lumaCand[(size_t)i] && (c == 0 ||
            ((!recover || tiles[(size_t)i].e[c * 4] >= kChromaResolvedVar) &&
             (recover || (r[c * 4] < kChromaClip && r[c * 4 + 1] < kChromaClip))));
        sc->val[(size_t)i] = use ? (recover && c > 0 ? tiles[(size_t)i].e[f] : r[f]) : -1.0;
      }
      for (int l = 0; l < nodes; ++l) {
        const double wsum = gather(sc, l, 1);
        sc->valid[l] = wsum >= minWeight;
        fit->weight[f][l] = sc->valid[l] ? wsum : 0;
        if (sc->valid[l])
          fit->var[f][l] = std::max((recover && c > 0 ? 1.0 : fit->var[f][l]) *
                                  weightedQuantile(sc->pairs, wsum, q) / corr, 1e-30);
      }
      fillNodes(fit->var[f], sc->valid, nodes);
    });
  }

  // ---- levels that only occur on textured surfaces ----
  // The floor of such a level is the texture, not the noise. It gives itself
  // away because noise is a smooth function of luminance:
  //   * band 0 of a level cannot be more than kMaxRise per knot interval
  //     above any other level;
  //   * the SHAPE of the spectrum changes even less: the ratio of band 1 to
  //     band 0 (and of bands 2, 3 to band 1, of the full band to band 0)
  //     cannot be more than kMaxRatioRise per interval above another level's.
  //     This catches soft texture, which leaves band 0 alone.
  // Such nodes are dropped and the curve is interpolated across them. A
  // steeper genuine rise is cut off too, which errs towards too little
  // denoising.
  // Except towards black, in the lower part of the range (darkRise: band 0
  // of each channel): log-encoded footage has its noise rise steeply into the toe of
  // the curve, faster than kMaxRise. A node below kDarkEnd may stand up to
  // kMaxRiseDark per knot interval above any brighter node, and that much
  // within the first interval (the step where the toe meets the log segment
  // is sharp: 1.75 times within a sixth of an interval on a real S-Log3
  // clip). With the old bound every node under Y 0.19 of that clip was
  // dropped as texture (three quarters of its flat tiles) and the shadows
  // got the noise of the mid-tones, half of their own.
  auto dropRisers = [&](const double* v, double* weight, double maxRise, bool darkRise) {
    bool drop[kMaxNodes] = {};
    double rise[kMaxNodes], riseDark[kMaxNodes], sv[kMaxNodes];
    for (int d = 0; d < nodes; ++d) {
      rise[d] = std::pow(maxRise, d / (double)sub);
      riseDark[d] = std::max(rise[d], std::pow(kMaxRiseDark, std::max(1.0, d / (double)sub)));
    }
    for (int j = 0; j < nodes; ++j) sv[j] = std::sqrt(v[j]);
    const bool darkNode[2] = {false, darkRise};
    for (int j = 0; j < nodes; ++j) {
      if (weight[j] <= 0) continue;
      const bool dark = darkNode[(double)j / (nodes - 1) < kDarkEnd];
      double bound = 1e300;
      for (int i = 0; i < nodes; ++i) {
        if (i == j || weight[i] <= 0) continue;
        bound = std::min(bound, sv[i] * (dark && i > j ? riseDark : rise)[std::abs(j - i)]);
      }
      drop[j] = sv[j] > 1.15 * bound;
    }
    double dropped = 0;
    for (int j = 0; j < nodes; ++j) {
      if (!drop[j]) continue;
      dropped += weight[j];
      weight[j] = 0;
    }
    return dropped;
  };
  // Drops the risers of feature f relative to feature `ref`, and rebuilds f
  // at every node without data as ref times the interpolated ratio.
  auto tieToReference = [&](int f, int ref) {
    double rel[kMaxNodes];
    bool keep[kMaxNodes];
    for (int l = 0; l < nodes; ++l) {
      if (fit->weight[ref][l] <= 0) fit->weight[f][l] = 0;  // no reference, no ratio
      rel[l] = fit->var[f][l] / std::max(fit->var[ref][l], 1e-30);
    }
    // Quantization gives a discontinuous chroma floor across colours and
    // brightnesses. A quiet node cannot veto all the other colour nodes.
    if (!recover || channelOf(f) == 0) dropRisers(rel, fit->weight[f], kMaxRatioRise, false);
    bool any = false;
    for (int l = 0; l < nodes; ++l) any = (keep[l] = fit->weight[f][l] > 0) || any;
    if (!any) return;
    fillNodes(rel, keep, nodes);
    for (int l = 0; l < nodes; ++l)
      if (!keep[l]) fit->var[f][l] = fit->var[ref][l] * rel[l];
  };
  double total = 0;
  for (int l = 0; l < nodes; ++l) total += fit->weight[0][l];
  if (debugLevel() >= 4) {
    fprintf(stderr, "[profile] pre-drop nodes Y b0 / b1 (sigma:weight):");
    for (int l = 0; l < nodes; ++l)
      if (fit->weight[0][l] > 0)
        fprintf(stderr, " %.3f=%.5f/%.5f:%.0f", (double)l / (nodes - 1), std::sqrt(fit->var[0][l]),
                std::sqrt(fit->var[1][l]), fit->weight[0][l]);
    fprintf(stderr, "\n");
  }
  double dropped = 0;
  for (int c = 0; c < 3; ++c) {
    const double d = recover && c > 0 ? 0 : dropRisers(fit->var[c * 4], fit->weight[c * 4], kMaxRise, true);
    if (c == 0) dropped += d;
    bool keep[kMaxNodes];
    for (int l = 0; l < nodes; ++l) keep[l] = fit->weight[c * 4][l] > 0;
    fillNodes(fit->var[c * 4], keep, nodes);
    double before = 0, after = 0;
    for (int l = 0; l < nodes; ++l) before += fit->weight[c * 4 + 1][l];
    tieToReference(c * 4 + 1, c * 4);
    for (int l = 0; l < nodes; ++l) after += fit->weight[c * 4 + 1][l];
    if (c == 0) dropped = std::max(dropped, before - after);
  }
  fit->droppedFraction = total > 0 ? std::min(1.0, dropped / total) : 0;
  classify();  // fine-band ratios against the final node model

  // ---- bands 2 and 3, full band ----
  // A tile that is flat in the fine bands very often still holds soft
  // structure (shading, a defocused background, an edge 20 px outside the
  // tile), and that structure is far above the coarse-band noise, which is
  // only 4-9 % of the pixel noise. So these bands take the lower quartile of
  // the candidates instead of the median (right as long as a quarter of them
  // are clean), corrected with the chi-square quantile. A quartile needs more
  // tiles than a median, so each node pools half a knot interval either side;
  // inside that window the band is assumed to follow the luminance curve of
  // the same channel's band 1, which is already known.
  // (Each task reads the fine bands, which are final by now, and writes its
  // own feature.)
  static const int kCoarse[9] = {2, 3, 6, 7, 10, 11, 12, 13, 14};
  parallelFor(9, [&](size_t k) {
    const int f = kCoarse[k];
    const int c = channelOf(f), b = bandOf(f);
    Scratch* sc = &scratchOf[(size_t)f];
    sc->val.resize((size_t)n);
    // Full band: relative to band 0 (same pixels, so the ratio of a clean
    // tile hardly varies; anything coarse adds to it). Bands 2, 3: relative
    // to band 1.
    const int ref = b < 0 ? c * 4 : c * 4 + 1;
    const int reach = std::max(1, sub / 2);
    double sref[kMaxNodes];
    for (int l = 0; l < nodes; ++l) sref[l] = std::sqrt(fit->var[ref][l]);
    for (int i = 0; i < n; ++i) {
      const double* r = &ratio[(size_t)i * kFeat];
      bool use = r[0] < clip[0] && r[1] < clip[1];
      if (c > 0) use = use && (!recover || tiles[(size_t)i].e[c * 4] >= kChromaResolvedVar) &&
          (recover || (r[c * 4] < kChromaClip && r[c * 4 + 1] < kChromaClip));
      const Tile& t = tiles[(size_t)i];
      if (!use) {
        sc->val[(size_t)i] = -1.0;
      } else if (b < 0) {
        sc->val[(size_t)i] = t.e[f] / std::max((double)t.e[ref], 1e-30);
      } else {
        const double m = sref[l0s[(size_t)i]] * (1 - ws[(size_t)i]) + sref[l0s[(size_t)i] + 1] * ws[(size_t)i];
        sc->val[(size_t)i] = t.e[f] / std::max(m * m, 1e-30);
      }
    }
    for (int l = 0; l < nodes; ++l) {
      const double wsum = gather(sc, l, reach);
      // only where the reference band itself is measured at this level
      sc->valid[l] = wsum >= minWeight * reach && fit->weight[ref][l] > 0;
      fit->weight[f][l] = sc->valid[l] ? wsum / reach : 0;
      if (!sc->valid[l]) continue;
      double rel;
      if (b < 0) {
        // The ratio of a clean tile is tight (+-3 %), so its lower third is
        // 1.5 % under its centre.
        rel = weightedQuantile(sc->pairs, wsum, 0.33) * 1.015;
      } else {
        // Lower quartile and lower decile, each corrected with its chi-square
        // quantile: equal for noise; where the candidates' spread is wider
        // than noise alone (structure), the decile is nearer the truth.
        const double q25 = weightedQuantile(sc->pairs, wsum, 0.25) / chiQuantile(kNu[b], -0.6745);
        const double q10 = weightedQuantile(sc->pairs, wsum, 0.10) / chiQuantile(kNu[b], -1.2816);
        rel = std::min(q25, q10);
      }
      fit->var[f][l] = std::max(fit->var[ref][l] * rel, 1e-30);
    }
    tieToReference(f, ref);
  });
  classify();  // candidates and ratios against the final node model

  // ---- onto the knots ----
  bool any = false;
  for (int l = 0; l < nodes; ++l) any = any || fit->weight[0][l] > 0;
  if (!any) return false;
  // Band 0 of each channel directly; every other feature as a ratio to its
  // reference band (band 1 and the full band to band 0, bands 2 and 3 to
  // band 1). The ratio changes slowly with luminance, so where a feature has
  // no data of its own it is still tied to a curve that has, and the full
  // band cannot end up below band 0.
  for (int c = 0; c < 3; ++c) projectToKnots(fit->var[c * 4], fit->weight[c * 4], sub, fit->knotVar[c * 4]);
  for (int f : {1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 13, 14}) {
    const int c = channelOf(f), b = bandOf(f);
    const int ref = (b < 0 || b == 1) ? c * 4 : c * 4 + 1;
    double rel[kMaxNodes], relKnot[kL];
    for (int l = 0; l < nodes; ++l) rel[l] = fit->var[f][l] / std::max(fit->var[ref][l], 1e-30);
    projectToKnots(rel, fit->weight[f], sub, relKnot);
    for (int k = 0; k < kL; ++k) fit->knotVar[f][k] = fit->knotVar[ref][k] * (b < 0 ? std::max(relKnot[k], 1.0) : relKnot[k]);
  }
  // Prior on the spectrum: noise is not much redder than 1/f, i.e. the sigma
  // of band 2 does not exceed that of band 1 by more than a quarter, nor band
  // 3 that of band 2 (the real night clips have ratios of 0.5-0.6; white
  // noise 0.43 and 0.48). Soft structure in the "flat" tiles is, and this is
  // the last line against it. It also limits footage upscaled 3x or more,
  // whose noise does sit in bands 2-3: that then gets too little denoising
  // in those bands, not too much.
  // Chroma is allowed up to twice the band below (kChromaRedder): blotchy
  // chroma noise is redder than 1/f (processed phone footage: band 2 at 1.4
  // times band 1; correlated chroma noise 16 px across at 4K: band 3 at 2.7
  // times band 2, read as 1.25 with the luma limit), and the chroma
  // candidates are luma-flat tiles without gross chroma structure already.
  for (int c = 0; c < 3; ++c) {
    for (int l = 0; l < kL; ++l) {
      double sum = std::sqrt(fit->knotVar[c * 4][l]) + std::sqrt(fit->knotVar[c * 4 + 1][l]);
      const double redder = c == 0 ? 1.25 : kChromaRedder;
      for (int b = 2; b < 4; ++b) {
        double& v = fit->knotVar[c * 4 + b][l];
        v = std::min(v, redder * redder * fit->knotVar[c * 4 + b - 1][l]);
        sum += std::sqrt(v);
      }
      // c_0 - c_4 is the sum of the four bands, so its sigma cannot exceed
      // the sum of theirs whatever their correlation.
      double& full = fit->knotVar[12 + c][l];
      full = std::min(full, sum * sum);
    }
  }
  fit->validKnots = 0;
  for (int k = 0; k < kL; ++k) {
    bool have = false;
    for (int l = std::max(0, k * sub - sub / 2); l <= std::min(nodes - 1, k * sub + sub / 2); ++l)
      have = have || fit->weight[0][l] > 0;
    fit->knot[k] = have;
    fit->validKnots += have;
  }

  // ---- diagnostics ----
  std::vector<double> c0, c1, a0, a1, ku;
  for (int i = 0; i < n; ++i) {
    const double* r = &ratio[(size_t)i * kFeat];
    a0.push_back(r[0]);
    a1.push_back(r[1]);
    if (lumaCand[(size_t)i]) {
      c0.push_back(r[0]);
      c1.push_back(r[1]);
      ku.push_back(tiles[(size_t)i].kurt);
    }
  }
  auto lowSpread = [](std::vector<double>& v) {
    if (v.size() < 8) return 1.0;
    double med = quantileOf(v, 0.5), p16 = quantileOf(v, 0.16);
    return med > 0 ? 1.0 - p16 / med : 1.0;
  };
  auto floorRatio = [](std::vector<double>& v) {
    double lo = quantileOf(v, 0.025), hi = quantileOf(v, 0.10);
    return lo > 0 ? hi / lo : 99.0;
  };
  parallelFor(5, [&](size_t k) {
    switch (k) {
      case 0: fit->lowSpread0 = lowSpread(c0); break;
      case 1: fit->lowSpread1 = lowSpread(c1); break;
      case 2: fit->floorRatio0 = floorRatio(a0); break;
      case 3: fit->floorRatio1 = floorRatio(a1); break;
      default: fit->kurtosis = medianOf(ku); break;
    }
  });
  fit->candFraction = (double)fit->candidates / (double)n;
  if (debugLevel() >= 3) {
    // Spread of the tile energies among the candidates, for calibrating kNu:
    // P16/P50 and P84/P50 of the normalised energy of each luma band.
    fprintf(stderr, "[profile] tile energy P16/P50, P84/P50 (Y b0..b3):");
    for (int f = 0; f < 4; ++f) {
      std::vector<double> v;
      for (int i = 0; i < n; ++i)
        if (lumaCand[(size_t)i]) v.push_back(tiles[(size_t)i].e[f] / modelAt(fit->var[f], l0s[(size_t)i], ws[(size_t)i]));
      const double med = quantileOf(v, 0.5);
      fprintf(stderr, "  %.3f %.3f", quantileOf(v, 0.16) / med, quantileOf(v, 0.84) / med);
    }
    fprintf(stderr, "\n");
  }
  fit->ok = true;
  return true;
}

// A fit costs about 0.3 ms per thousand tiles, and resolve can run on the
// render thread (first frame, waitUntilCompleted). Frames up to 4K have at
// most 8000 tiles; beyond kMaxFitTiles (8K frames) a regular subset is kept.
constexpr size_t kMaxFitTiles = 10000;
void thin(std::vector<Tile>* tiles) {
  const size_t n = tiles->size();
  if (n <= kMaxFitTiles) return;
  // Bresenham-style: evenly spread over the list (the list is in raster
  // order, and an irrational-looking step avoids keeping whole columns).
  std::vector<Tile> kept;
  kept.reserve(kMaxFitTiles);
  for (size_t k = 0; k < kMaxFitTiles; ++k) kept.push_back((*tiles)[k * n / kMaxFitTiles]);
  tiles->swap(kept);
}

// The finest luminance grid that the frame's flat tiles can support.
bool fitFloorLuma(std::vector<Tile>& tiles, Fit* fit, bool recover = false) {
  for (int sub : {kMaxSub, 2, 1}) {
    if (!fitFloorAt(tiles, sub, false, fit, recover)) continue;
    double w = 0;
    for (int l = 0; l < fit->nodes; ++l) w += fit->weight[0][l];
    if (w >= 0.5 * fit->candidates) return true;  // most flat tiles sit at a level that could be fitted
  }
  // One level for everything: the luminance dependence is given up.
  for (Tile& t : tiles) t.y = 0.5f;
  const bool ok = fitFloorAt(tiles, 1, true, fit, recover);
  fit->oneLevel = true;
  return ok;
}

// Degenerate luma: use the strongest measured colour channel to select the
// noise floor, retaining the original luminance coordinate. Do not synthesize
// noise in a constant channel. Normal colour footage takes the unchanged path.
bool fitFloor(std::vector<Tile>& tiles, Fit* fit, bool recover = false) {
  double energy[3] = {};
  for (const Tile& t : tiles) for (int c=0;c<3;++c) energy[c] += t.e[c*4];
  int reference = 0;
  if (energy[0] < 1e-20) reference = energy[1] >= energy[2] ? 1 : 2;
  auto swapTiles = [&]() {
    if (!reference) return;
    for (Tile& t : tiles) {
      for (int b=0;b<4;++b) std::swap(t.e[b],t.e[reference*4+b]);
      std::swap(t.e[12],t.e[12+reference]);
    }
  };
  swapTiles();
  bool ok = fitFloorLuma(tiles, fit);
  if (ok && recover && !reference) {
    bool collapsed[3] = {};
    for (int c = 1; c < 3; ++c) {
      double peak = 0;
      for (int l = 0; l < kL; ++l) peak = std::max(peak, fit->knotVar[12+c][l]);
      // A near-zero fitted channel alongside resolvable tile variation is
      // the quantization-collapse signature. Healthy fits take no new path.
      int resolved = 0;
      for (const Tile& t : tiles) resolved += t.e[c*4] >= kChromaResolvedVar;
      collapsed[c] = peak < kChromaCollapsedVar && resolved >= kMinTiles && resolved > int(tiles.size()/4);
    }
    if (collapsed[1] || collapsed[2]) {
      Fit retry;
      std::vector<Tile> copy = tiles;
      if (fitFloorLuma(copy, &retry, true)) for (int c = 1; c < 3; ++c) if (collapsed[c]) {
        for (int b = 0; b < 5; ++b) {
          const int f = b < 4 ? c*4+b : 12+c;
          for (int l = 0; l < kL; ++l) fit->knotVar[f][l] = retry.knotVar[f][l];
        }
      }
    }
  }
  swapTiles();
  if (reference) for (int b=0;b<5;++b) {
    const int f=b<4?b:12, g=b<4?reference*4+b:12+reference;
    for (int n=0;n<kMaxNodes;++n) {
      std::swap(fit->var[f][n],fit->var[g][n]);
      std::swap(fit->weight[f][n],fit->weight[g][n]);
    }
    for (int l=0;l<kL;++l) std::swap(fit->knotVar[f][l],fit->knotVar[g][l]);
  }
  return ok;
}

// The knots a fit measured (bit l = knot l had luma candidates near it). A
// one-level fit measured none of them as such.
uint32_t knotMask(const Fit& f) {
  if (!f.ok || f.oneLevel) return 0;
  uint32_t m = 0;
  for (int l = 0; l < kL; ++l)
    if (f.knot[l]) m |= 1u << l;
  return m;
}

inline float smoothstep(float a, float b, float x) {
  float t = std::min(1.0f, std::max(0.0f, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

// How much a fit looks like a real noise floor, 0..1.
float fitConfidence(const Fit& f) {
  if (!f.ok) return 0;
  // Enough tiles on the floor: 150, or a fifth of the tiles on a small frame.
  const float need = std::min(150.0f, std::max(12.0f, 0.2f * (float)f.usable));
  const float n = std::min(1.0f, (float)f.candidates / need);
  // A noise floor is a sharp lower edge: among all usable tiles, the 10th
  // percentile of the normalised fine-band energy is only a little above the
  // 2.5th. Measured: 1.03-1.04 (band 0) and 1.07-1.08 (band 1) on synthetic
  // flat frames, 1.10-1.18 on the real night clips (their noise level also
  // varies a little inside a luminance node). A texture continuum has no
  // such edge: 1.3 and more.
  const float s0 = 1.0f - smoothstep(1.22f, 1.50f, (float)f.floorRatio0);
  const float s1 = 1.0f - smoothstep(1.25f, 1.60f, (float)f.floorRatio1);
  // Levels whose floor had to be thrown away as texture.
  const float d = 1.0f - 0.5f * (float)f.droppedFraction;
  // (The kurtosis of the finest band was tried as a texture detector and
  // dropped: 3.0 on synthetic Gaussian noise and 3.1-3.5 on the real night
  // clips, but 5.8-6.7 on the heavy-noise dark test clips, as high as on
  // texture. It is still printed by OD_PROFILE_DEBUG.)
  return n * std::min(s0, s1) * d;
}

void fitToProfile(const Fit& f, float confidence, ODNoiseProfile* out) {
  *out = ODNoiseProfile{};
  for (int c = 0; c < 3; ++c) {
    for (int l = 0; l < kL; ++l) {
      for (int b = 0; b < OD_BANDS; ++b)
        out->sigma[c][b][l] = std::max(kSigmaFloor, (float)std::sqrt(f.knotVar[c * 4 + b][l]));
      out->sigmaFull[c][l] = std::max(kSigmaFloor, (float)std::sqrt(f.knotVar[12 + c][l]));
    }
  }
  out->confidence = std::min(1.0f, std::max(0.0f, confidence));
  out->valid = 1;
}

// Spatial and temporal estimate into one profile.
//   * Each overestimates when its assumption fails (spatial: the "flat" tiles
//     carry texture; temporal: something moved) and neither underestimates,
//     so where they disagree the lower one is right. On the four real night
//     clips they agree within about 5 % wherever both have tiles.
//   * The exception is footage whose noise is correlated between frames
//     (blended frames, heavy inter-frame compression): there the temporal
//     estimate is too low. It cannot be told apart from a textured static
//     shot, so the temporal estimate is followed downwards only as far as its
//     own confidence goes. Erring low means too little denoising, which the
//     noise-level control can correct; erring high destroys detail.
//   * Where they agree (within 10 %), their confidence-weighted geometric
//     mean; in between, a smooth transition.
//   * A luminance level only one of them has tiles for is taken from that one.
void combineEstimates(const Fit& fs, float cs, const Fit& ft, float ct, ODNoiseProfile* out) {
  if (!ft.ok) {
    fitToProfile(fs, cs, out);
    return;
  }
  if (!fs.ok) {
    fitToProfile(ft, ct, out);
    return;
  }
  *out = ODNoiseProfile{};
  const double ws = std::max(cs, 0.05f), wt = std::max(ct, 0.05f);
  const double follow = std::min(1.0, ct / 0.6);
  double agree = 0;
  int agreeN = 0;
  // Chroma: a temporal estimate below the spatial one is also what chroma
  // noise that persists between frames gives (in-camera noise reduction,
  // codecs; measured on a phone clip: Cb full band 0.00149 temporal against
  // 0.00207 spatial in the shadows, luma band 0 within 2 %), and the temporal
  // stage cannot remove that noise: the spatial stage needs its per-frame
  // level. So where the fine luma bands of the two fits agree (the flat
  // tiles really are flat), chroma follows the spatial estimate; where the
  // temporal luma is lower too (texture in the "flat" tiles), it follows the
  // lower one as luma does.
  double lumaD = 0;
  int lumaN = 0;
  for (int f : {0, 1}) {
    for (int l = 0; l < kL; ++l) {
      if (!fs.knot[l] || !ft.knot[l]) continue;
      lumaD += 0.5 * std::log(std::max(fs.knotVar[f][l], 1e-30) / std::max(ft.knotVar[f][l], 1e-30));
      ++lumaN;
    }
  }
  const double trustS = lumaN ? 1.0 - smoothstep(0.05f, 0.25f, (float)(lumaD / lumaN)) : 0.0;
  for (int f = 0; f < kFeat; ++f) {
    const int c = channelOf(f), b = bandOf(f);
    for (int l = 0; l < kL; ++l) {
      const double ls = 0.5 * std::log(std::max(fs.knotVar[f][l], 1e-30));
      const double lt = 0.5 * std::log(std::max(ft.knotVar[f][l], 1e-30));
      double v;
      if (fs.knot[l] && !ft.knot[l]) {
        v = ls;
      } else if (ft.knot[l] && !fs.knot[l] && ct >= 0.3f) {
        v = lt;
      } else {
        const double d = ls - lt;  // > 0: temporal is lower
        const double avg = (ws * ls + wt * lt) / (ws + wt);
        const double low = d > 0 ? ls - follow * d * (c > 0 ? 1.0 - trustS : 1.0) : ls;
        const double u = smoothstep(0.095f, 0.30f, (float)std::fabs(d));  // 10 % .. 35 %
        v = (1 - u) * avg + u * low;
        if (c == 0 && (b == 0 || b == 1) && fs.knot[l] && ft.knot[l]) {
          agree += d;  // > 0: temporal lower
          ++agreeN;
        }
      }
      const float sig = std::max(kSigmaFloor, (float)std::exp(v));
      if (b >= 0) out->sigma[c][b][l] = sig;
      else out->sigmaFull[c][l] = sig;
    }
  }
  // Confidence. The temporal estimate is evidence only where it is LOWER
  // than the spatial one: on a moving textured scene both see the same
  // texture and agree, which proves nothing. So: the spatial confidence,
  // moving to the temporal one as the temporal estimate drops below the
  // spatial one (fine luma bands, levels both have tiles for).
  const float lower = agreeN > 0 ? smoothstep(0.05f, 0.25f, (float)(agree / agreeN)) : 0.0f;
  const float conf = cs + lower * (ct - cs);
  out->confidence = std::min(1.0f, std::max(0.0f, conf));
  out->valid = 1;
}

// How far the temporal estimate is below the spatial one: mean of
// log(spatial / temporal) over the levels both have tiles for, largest of
// the five luma features (the coarse bands and the full band are where soft
// structure inflates the spatial estimate first).
double temporalLower(const Fit& fs, const Fit& ft) {
  double worst = 0;
  for (int f : {0, 1, 2, 3, 12}) {
    double d = 0;
    int n = 0;
    for (int l = 0; l < kL; ++l) {
      if (!fs.knot[l] || !ft.knot[l]) continue;
      d += 0.5 * std::log(std::max(fs.knotVar[f][l], 1e-30) / std::max(ft.knotVar[f][l], 1e-30));
      ++n;
    }
    if (n) worst = std::max(worst, d / n);
  }
  return worst;
}

// Share of the temporal fit's tiles that sit at luminance levels the spatial
// fit has no flat tile for.
double temporalExtra(const Fit& fs, const Fit& ft) {
  double total = 0, extra = 0;
  for (int l = 0; l < ft.nodes; ++l) {
    const int k = std::min(kL - 1, (l + ft.sub / 2) / ft.sub);  // nearest knot
    total += ft.weight[0][l];
    if (ft.knot[k] && !fs.knot[k]) extra += ft.weight[0][l];
  }
  return total > 0 ? extra / total : 0.0;
}

void debugFit(const char* name, const Fit& f, float conf) {
  fprintf(stderr,
          "[profile] %s ok=%d usable=%d cand=%d (%.2f) grid %d knots=%d dropped %.2f lowSpread %.3f/%.3f floorRatio "
          "%.3f/%.3f kurt %.2f conf=%.2f\n",
          name, f.ok, f.usable, f.candidates, f.candFraction, f.sub, f.validKnots, f.droppedFraction, f.lowSpread0,
          f.lowSpread1, f.floorRatio0, f.floorRatio1, f.kurtosis, conf);
  if (!f.ok) return;
  for (int c = 0; c < 3; ++c) {
    for (int b = 0; b < 5; ++b) {
      if (c > 0 && b > 0 && b < 4) continue;
      fprintf(stderr, "[profile] %s %s b%c:", name, c == 0 ? "Y " : c == 1 ? "Cb" : "Cr", b < 4 ? '0' + b : 'F');
      for (int l = 0; l < kL; ++l)
        fprintf(stderr, " %8.5f%c", std::sqrt(f.knotVar[b < 4 ? c * 4 + b : 12 + c][l]), (c == 0 && f.knot[l]) ? '*' : ' ');
      fprintf(stderr, "\n");
    }
  }
  if (debugLevel() >= 2) {
    for (int feat : {0, 3}) {
      fprintf(stderr, "[profile] %s nodes Y b%d (sigma:weight):", name, feat);
      for (int l = 0; l < f.nodes; ++l)
        if (f.weight[feat][l] > 0)
          fprintf(stderr, " %.3f=%.5f:%.0f", (double)l / (f.nodes - 1), std::sqrt(f.var[feat][l]), f.weight[feat][l]);
      fprintf(stderr, "\n");
    }
  }
}

// ---- session aggregate -----------------------------------------------------
constexpr size_t kRing = 24;            // per-frame estimates kept
constexpr double kChangeFactor = 1.35;  // a fresh estimate this far from the slot counts as "different"
constexpr int kChangeStreak = 2;        // this many in a row, far on the median of the measured knots (one
                                        // more if far only on their upper third): new footage, start
                                        // again. A render uses the
                                        // slot as it was before its own frame was measured, so the first
                                        // kChangeStreak frames of new footage under one session (an
                                        // adjustment layer over a cut, a gain step) are filtered with the
                                        // old profile: 2 is the least that still ignores one odd frame.
                                        // At a cut the temporal stage detected, and on a jump, the
                                        // session marks the estimate decisive and one is enough
                                        // (smooth()).

// Weight of a ring entry in the slot.
double ringWeight(const ODNoiseProfile& p) {
  const double c = std::max(0.02f, p.confidence);
  return c * c;
}

// Confidence-weighted interquartile mean (the middle half by weight).
double robustMean(std::vector<std::pair<double, double>>& v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  double total = 0;
  for (auto& p : v) total += p.second;
  if (v.size() < 4 || total <= 0) {
    double a = 0;
    for (auto& p : v) a += p.first * p.second;
    return total > 0 ? a / total : v[0].first;
  }
  const double lo = 0.25 * total, hi = 0.75 * total;
  double acc = 0, sum = 0, wsum = 0;
  for (auto& p : v) {
    const double a = acc, b = acc + p.second;
    const double w = std::max(0.0, std::min(b, hi) - std::max(a, lo));
    sum += w * p.first;
    wsum += w;
    acc = b;
  }
  return wsum > 0 ? sum / wsum : v[v.size() / 2].first;
}

// How far a fresh estimate is from the slot, over the luminance knots both measured (fine
// luma bands 0 and 1). A knot without tiles holds a copy of (or an interpolation between)
// the measured ones, so when the highest measured knot of one frame lacks tiles, every knot
// above it jumps to the value below; compared over all eight knots, that alone made
// unchanged footage look like new footage. Knots: fresh's and the slot's masks; with no knot
// in common, fresh's measured knots, then all. Two figures: *median, the median |log ratio|,
// and the return value, the |log ratio| a third of the compared values exceed. A real change
// of noise can be confined to some levels (the shadows of a night street, where the noise
// rose 1.6 times while the mid-tones did not), which the median over the measured knots
// misses; the upper third sees it, and also sees one knot's estimate scatter from frame to
// frame, so smooth asks it for a longer streak. *direction: +1 if fresh is higher where it
// is far (the sum of the log ratios at or above the upper-third value), -1 if lower.
double profileDistance(const ODNoiseProfile& fresh, uint32_t freshKnots, const ODNoiseProfile& slot,
                       uint32_t slotKnots, double* median, int* direction) {
  uint32_t use = freshKnots & slotKnots;
  if (!use) use = freshKnots;
  if (!use) use = slotKnots;
  if (!use) use = (1u << kL) - 1;
  std::vector<double> d, sgn;
  for (int band = 0; band < 2; ++band)
    for (int l = 0; l < kL; ++l) {
      if (!(use & (1u << l))) continue;
      const double r =
          std::log(std::max(fresh.sigma[0][band][l], kSigmaFloor) / std::max(slot.sigma[0][band][l], kSigmaFloor));
      d.push_back(std::fabs(r));
      sgn.push_back(r);
    }
  std::vector<double> sorted = d;
  std::sort(sorted.begin(), sorted.end());
  // The value a third of them exceed: the (n - ceil(n / 3) + 1)-th smallest
  // (2 of 2, 3 of 4, 5 of 6, 11 of 16).
  const size_t n = sorted.size();
  const double q = sorted[n - (n + 2) / 3];
  if (median) *median = medianOf(sorted);
  if (direction) {
    double sum = 0;
    for (size_t k = 0; k < d.size(); ++k)
      if (d[k] >= q) sum += sgn[k];
    *direction = sum >= 0 ? 1 : -1;
  }
  return q;
}

void aggregate(const std::vector<ODNoiseProfile>& ring, const std::vector<uint32_t>& knots,
               ODNoiseProfile* slot, bool supportedKnots) {
  std::vector<std::pair<double, double>> v;
  // An absent knot is not an observation. Hold the previous value when
  // the whole ring has no support; smooth seeds a new shot from its first fit.
  ODNoiseProfile out = supportedKnots ? *slot : ODNoiseProfile{};
  for (int c = 0; c < 3; ++c) {
    for (int l = 0; l < kL; ++l) {
      for (int b = 0; b <= OD_BANDS; ++b) {
        v.clear();
        for (size_t i = 0; i < ring.size(); ++i) {
          if (supportedKnots && !(knots[i] & (1u << l))) continue;
          const ODNoiseProfile& p = ring[i];
          v.emplace_back(std::log(std::max(b < OD_BANDS ? p.sigma[c][b][l] : p.sigmaFull[c][l], kSigmaFloor)),
                         ringWeight(p));
        }
        if (v.empty()) continue;
        const float sig = std::max(kSigmaFloor, (float)std::exp(robustMean(v)));
        if (b < OD_BANDS) out.sigma[c][b][l] = sig;
        else out.sigmaFull[c][l] = sig;
      }
    }
  }
  v.clear();
  for (const ODNoiseProfile& p : ring) v.emplace_back(p.confidence, ringWeight(p));
  out.confidence = std::min(1.0f, std::max(0.0f, (float)robustMean(v)));
  out.valid = 1;
  *slot = out;
}

// The knots the slot holds as measured: those that estimates carrying at
// least a quarter of the ring's weight measured (knots[i], resolve()). What
// profileDistance compares a fresh estimate with.
uint32_t measuredKnots(const std::vector<ODNoiseProfile>& ring, const std::vector<uint32_t>& knots) {
  const size_t n = std::min(ring.size(), knots.size());
  double total = 0;
  for (size_t i = 0; i < n; ++i) total += ringWeight(ring[i]);
  uint32_t m = 0;
  for (int l = 0; l < kL; ++l) {
    double have = 0;
    for (size_t i = 0; i < n; ++i)
      if ((knots[i] >> l) & 1u) have += ringWeight(ring[i]);
    if (have > 0 && have >= 0.25 * total) m |= 1u << l;
  }
  return m;
}

// ---- stage -----------------------------------------------------------------
class Profiler final : public ProfileStage {
 public:
  // Session aggregates (agg_ below), one per grain mode (ODParams.grainMode), under the
  // session's profile lock except the atomics, which the throttle reads.
  struct Aggregate {
    std::vector<ODNoiseProfile> ring;
    std::vector<uint32_t> ringKnots;  // resolve()'s knot mask of each ring entry
    uint32_t slotKnots = 0;           // current measured mask, or accumulated support (finest grain)
    int streak = 0;      // far estimates in a row, in one direction (upper-third distance)
    int wideStreak = 0;  // the same, far on the median too
    int streakDir = 0;   // direction of the far estimates in the current streak
    std::atomic<int> ringCount{0};
    std::atomic<bool> changing{false};
    std::atomic<bool> resetPending{false};
  };
  Profiler(Context& ctx, std::string* err)
      : pool_(std::static_pointer_cast<BufferPool>(ctx.shared("profile.statistics-buffers", [&ctx] {
          return std::static_pointer_cast<void>(std::make_shared<BufferPool>(ctx.device()));
        }))) {
    const char* names[] = {"profile_l0", "profile_diff", "profile_l0_diff", "profile_l1",  "profile_b3",
                           "profile_l3", "profile_sum2", "profile_tiles",   "profile_view"};
    id<MTLComputePipelineState> __strong* psos[] = {&psoL0_, &psoDiff_, &psoL0Diff_, &psoL1_,  &psoB3_,
                                                    &psoL3_, &psoSum2_, &psoTiles_,  &psoView_};
    coarseDown_ = ctx.pipeline("profile_coarse_down", err);
    coarseStats_ = ctx.pipeline("profile_coarse_stats", err);
    for (int k = 0; k < 9; ++k) {
      *psos[k] = ctx.pipeline(names[k], err);
      if (!*psos[k]) break;  // psoView_ stays nil: the factory fails with *err
    }
  }
  ProfileAnalysisInfo analysisInfo(int mode) override {
    std::lock_guard<std::mutex> lock(analysisMutex_);
    ProfileAnalysisInfo info = analysis_;
    const Aggregate& a = agg_[mode == OD_GRAIN_FINEST ? OD_GRAIN_FINEST : OD_GRAIN_WHOLE];
    info.ringCount = a.ringCount.load();
    if (info.ringCount) info.measuredKnots = measuredKnots(a.ring, a.ringKnots);
    info.heldKnots = mode == OD_GRAIN_FINEST ? a.slotKnots & ~info.measuredKnots : 0;
    return info;
  }
  bool ok() const { return psoView_ != nil && coarseDown_ && coarseStats_; }

  bool encodeMeasure(RenderEnv& env, const WorkFrame& cur, ProfileMeasurement& m) override {
    m.data = nil;
    const Grid g = gridFor(env.width, env.height);
    if (g.count() < (uint32_t)kMinTiles) return true;  // frame too small to say anything
    // One stage per frame size (the session's profile tracks), so the size
    // never changes under it.
    if (!m.deterministic && !shouldMeasure(env, cur) && !m.force) return true;
    // Nearest neighbour in time, if the render has one (frames are sorted by
    // |offset|): the temporal estimator.
    id<MTLTexture> prev = nil;
    if (env.frameCount >= 2 && env.frames && env.frames[0].tex == cur.tex) {
      if (m.pairOffset == 0 && std::abs(env.frames[1].offset) == 1) prev = env.frames[1].tex;
      for (int k = 1; m.pairOffset != 0 && k < env.frameCount; ++k)
        if (env.frames[k].offset == m.pairOffset) prev = env.frames[k].tex;
    }
    // The temporal chain costs more than the spatial one. It runs on every
    // measurement only while it is telling us something (the spatial fit is
    // weak, the temporal estimate is lower, or it covers luminance levels the
    // spatial one has no flat tile for: see resolve()); otherwise on one
    // measurement in four, only to notice when that changes. The rule is the
    // same at every quality so that playback and export aggregate the same
    // kind of estimates.
    if (!m.deterministic) {
      if (prev && !temporalUseful_.load() && (measureCount_ % kTemporalEvery) != 0) prev = nil;
      ++measureCount_;
    }
    auto info = std::make_shared<MeasureInfo>();
    info->grid = g;
    info->pair = prev != nil;
    info->chromaRecovery = env.params.chromaRecovery != 0;
    info->pool = pool_;
    info->buffer = pool_->acquire((NSUInteger)g.count() * kTileFloats * sizeof(float));
    if (!info->buffer) {
      env.fail("profile: could not allocate the statistics buffer");
      return false;
    }
    if (!encodeStats(env, cur.tex, prev, info->buffer, g)) return false;
    if (!encodeCoarse(env, cur, m, info->coarse)) return false;
    m.data = info->buffer;
    m.user = info;
    return true;
  }

  // od_session_reset(): the slot is empty again. Measure the next frames
  // unconditionally and start the ring afresh with the first new estimate.
  void reset() override {
    { std::lock_guard<std::mutex> lock(analysisMutex_); analysis_ = {}; }
    warmup_ = kWarmup;
    measureCount_ = 0;
    for (Aggregate& a : agg_) {
      a.ringCount.store(0);
      a.changing.store(false);
      a.resetPending.store(true);
    }
    temporalUseful_.store(true);
  }

  // Playback (quality 0) does not need a measurement on every frame: once
  // the ring holds a few estimates, one frame in kEvery is enough to follow
  // the footage, and once the ring is full (the slot is then an average over
  // kRing estimates and barely moves) one in kSteadyEvery, a subset of those.
  // The others cost nothing. Which frames are measured depends on the frame
  // key, not on the order they arrive in. A change of footage is still
  // noticed by the first measurement after it, and from then on every frame
  // is measured until the slot has settled again (Aggregate::changing, of the
// render's grain mode). Full quality
  // measures every frame.
  bool shouldMeasure(RenderEnv& env, const WorkFrame& cur) {
    const double now = CFAbsoluteTimeGetCurrent();
    // After a pause, measure the next frames unconditionally: the stage is
    // not told about od_session_reset(), and an idle instance is where it
    // happens.
    if (now - lastCall_ > 0.5) warmup_ = kWarmup;
    lastCall_ = now;
    if (env.params.quality != 0) return true;
    if (warmup_ > 0) {
      --warmup_;
      return true;
    }
    // The aggregate of this render's grain mode: what a session used in one
    // mode only would have counted.
    const Aggregate& a = agg_[env.grainProfileV2 ? OD_GRAIN_FINEST : OD_GRAIN_WHOLE];
    const int have = a.ringCount.load();
    if (have < kThrottleAfter || a.changing.load() || a.resetPending.load()) return true;
    uint64_t h = cur.frameKey != OD_NO_KEY ? (uint64_t)cur.frameKey : counter_++;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33;
    return h % (have >= (int)kRing ? kSteadyEvery : kEvery) == 0;
  }

  bool resolve(const ProfileMeasurement& m, ODNoiseProfile* out, uint32_t* knots) override {
    if (knots) *knots = 0;
    const MeasureInfo* info = (const MeasureInfo*)m.user.get();
    if (!info || !m.data) return false;
    const double t0 = debugEnabled() ? CFAbsoluteTimeGetCurrent() : 0;
    const Grid& g = info->grid;
    const uint32_t n = g.count();
    if ((uint64_t)m.data.length < (uint64_t)n * kTileFloats * sizeof(float)) return false;
    const float* rec = (const float*)m.data.contents;

    std::vector<Tile> spatial, temporal;
    spatial.reserve(n);
    if (info->pair) temporal.reserve(n);
    auto finiteRecord = [&](const float* r) {
      for (uint32_t k = 0; k < (info->pair ? kTileFloats : kRecordFloats); ++k)
        if (!std::isfinite(r[k])) return false;
      return true;
    };
    // Degenerate footage, decided over the whole frame so that ordinary footage takes
    // exactly the path it always took: - luma without any detail in any tile (constant luma
    // under chroma noise): every tile is dead by the luma test (record 19), so the frame
    // uses the all-channel count (record 21); - a feature that is exactly zero in every
    // usable tile (the chroma of monochrome footage, a constant channel) is a valid zero,
    // not missing evidence: such tiles are kept, and that feature's sigma is 0.
    bool lumaFlat = true;
    for (uint32_t i = 0; i < n && lumaFlat; ++i) {
      const float* r = rec + (size_t)i * kTileFloats;
      if (!finiteRecord(r)) continue;
      for (int f : {0, 1, 2, 3, 12}) lumaFlat = lumaFlat && r[1 + f] == 0;
    }
    const int deadAt = lumaFlat ? 21 : 19;
    bool zeroS[kFeat], zeroT[kFeat];
    int usableS = 0, usableT = 0;
    for (int f = 0; f < kFeat; ++f) zeroS[f] = zeroT[f] = true;
    for (uint32_t i = 0; i < n; ++i) {
      const float* r = rec + (size_t)i * kTileFloats;
      const float* rt = r + kRecordFloats;
      if (!finiteRecord(r) || r[deadAt] > kZeroFracMax) continue;
      ++usableS;
      for (int f = 0; f < kFeat; ++f) zeroS[f] = zeroS[f] && r[1 + f] == 0;
      if (info->pair && rt[19] <= kZeroFracMax) {
        ++usableT;
        for (int f = 0; f < kFeat; ++f) zeroT[f] = zeroT[f] && rt[1 + f] == 0;
      }
    }
    for (int f = 0; f < kFeat; ++f) {
      zeroS[f] = zeroS[f] && usableS > 0;
      zeroT[f] = zeroT[f] && usableT > 0;
    }
    for (uint32_t i = 0; i < n; ++i) {
      const float* r = rec + (size_t)i * kTileFloats;
      const float* rt = r + kRecordFloats;
      if (!finiteRecord(r)) continue;
      if (r[deadAt] > kZeroFracMax) continue;  // clipped or synthetic flat
      Tile t;
      t.index = i;
      t.y = r[0];
      // Spatial: no oriented structure.
      const double tr = (double)r[16] + (double)r[17];
      const double dd = (double)r[16] - (double)r[17];
      const double coh = tr > 0 ? (dd * dd + 4.0 * (double)r[18] * (double)r[18]) / (tr * tr) : 0.0;
      if (coh <= kCoherenceMax) {
        bool pos = true;
        for (int f = 0; f < kFeat; ++f) {
          t.e[f] = r[1 + f];
          pos = pos && (t.e[f] > 0 || (zeroS[f] && t.e[f] == 0));
        }
        t.kurt = pos && r[1] > 0 ? r[20] / (r[1] * r[1]) : 0;
        if (pos) spatial.push_back(t);
      }
      // Temporal: Var(cur - prev) = 2 sigma^2. Exact duplicates are no use.
      if (info->pair && rt[19] <= kZeroFracMax) {
        bool pos = true;
        for (int f = 0; f < kFeat; ++f) {
          t.e[f] = 0.5f * rt[1 + f];
          pos = pos && (t.e[f] > 0 || (zeroT[f] && t.e[f] == 0));
        }
        t.kurt = pos && rt[1] > 0 ? rt[20] / (rt[1] * rt[1]) : 0;
        if (pos) temporal.push_back(t);
      }
    }

    thin(&spatial);
    thin(&temporal);
    // The two fits are independent: run them side by side (this is CPU time
    // on a Metal completion thread, or on the render thread for the first
    // frame and with waitUntilCompleted).
    const double t1 = debugEnabled() ? CFAbsoluteTimeGetCurrent() : 0;
    Fit fits[2];
    std::vector<Tile>* sets[2] = {&spatial, &temporal};
    Fit* fitp = fits;
    std::vector<Tile>** setp = sets;
    dispatch_apply(info->pair ? 2 : 1, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t k) {
      fitFloor(*setp[k], &fitp[k], info->chromaRecovery);
    });
    const Fit& fs = fits[0];
    const Fit& ft = fits[1];
    if (debugEnabled())
      fprintf(stderr, "[profile] cpu: collect %.2f ms, fits %.2f ms\n", 1000 * (t1 - t0),
              1000 * (CFAbsoluteTimeGetCurrent() - t1));
    const float cs = fitConfidence(fs), ct = fitConfidence(ft);
    if (debugEnabled()) {
      fprintf(stderr, "[profile] frame key %lld %ux%u tiles %u pair %d\n", (long long)m.frameKey, m.width, m.height, n,
              (int)info->pair);
      debugFit("S", fs, cs);
      if (info->pair) debugFit("T", ft, ct);
    }
    if (!fs.ok && !ft.ok) return false;
    // Is the temporal estimate telling us something the spatial one does not
    // (with some hysteresis)? If not, this estimate is the spatial one alone,
    // exactly as on the measurements that skip the temporal chain: the
    // session must not aggregate a mixture of two kinds of estimate.
    bool useTemporal = ft.ok;
    if (info->pair) {
      const double thr = m.deterministic || temporalUseful_.load() ? 0.05 : 0.10;
      useTemporal = !fs.ok || cs < 0.7f || (ft.ok && (temporalLower(fs, ft) > thr || temporalExtra(fs, ft) > thr));
      if (!m.deterministic) temporalUseful_.store(useTemporal);
    }
    // OD_PROFILE_MODE=S or T: one estimator only, for tests/profile.
    const char mode = debugMode();
    uint32_t mask = 0;
    if ((mode == 'S' || (!mode && !useTemporal)) && fs.ok) {
      fitToProfile(fs, cs, out);
      mask = knotMask(fs);
    } else if (mode == 'T' && ft.ok) {
      fitToProfile(ft, ct, out);
      mask = knotMask(ft);
    } else {
      combineEstimates(fs, cs, ft, ct, out);
      // combineEstimates takes a knot from the temporal fit alone only when
      // its confidence is 0.3 or more.
      mask = knotMask(fs) | (ft.ok && (ct >= 0.3f || !fs.ok) ? knotMask(ft) : 0u);
    }
    if (knots) *knots = mask;
    {
      ProfileAnalysisInfo next; next.measuredKnots = mask;
      std::vector<const Tile*> flat;
      if (fs.ok) for (const Tile& t : spatial) {
        const double pos = std::clamp((double)t.y, 0.0, 1.0) * (fs.nodes - 1);
        const int lo = std::min(fs.nodes - 2, (int)pos);
        const double f = pos - lo;
        bool use = true;
        for (int b = 0; b < 2; ++b) {
          const double v = fs.var[b][lo] * (1-f) + fs.var[b][lo+1] * f;
          use &= t.e[b] <= v * (1 + kClipSigmas * std::sqrt(2.0 / kNu[b]));
        }
        if (use) flat.push_back(&t);
      }
      next.tileCount = std::min<size_t>(128, flat.size());
      for (uint32_t k = 0; k < next.tileCount; ++k) {
        const Tile& t = *flat[k * flat.size() / next.tileCount];
        next.xy[k][0] = g.originX + (t.index % g.tilesW) * g.stride;
        next.xy[k][1] = g.originY + (t.index / g.tilesW) * g.stride;
        next.energy[k][0] = t.e[0]; next.energy[k][1] = t.e[1];
      }
      std::lock_guard<std::mutex> lock(analysisMutex_);
      analysis_ = next;
    }
    // No trustworthy floor (no flat tile, nothing static): whatever was
    // measured is an upper bound, texture included. Too little denoising on
    // such a frame is invisible (the texture masks the noise), too much eats
    // the texture, so the estimate is scaled down as the confidence drops.
    // In the session these estimates also carry almost no weight next to
    // better frames of the same clip.
    const float shrink = kLowConfidenceScale + (1.0f - kLowConfidenceScale) * smoothstep(0.1f, 0.5f, out->confidence);
    if (shrink < 1.0f && !mode) {
      for (int c = 0; c < 3; ++c) {
        for (int l = 0; l < kL; ++l) {
          for (int b = 0; b < OD_BANDS; ++b) out->sigma[c][b][l] = std::max(kSigmaFloor, out->sigma[c][b][l] * shrink);
          out->sigmaFull[c][l] = std::max(kSigmaFloor, out->sigmaFull[c][l] * shrink);
        }
      }
    }
    // A true zero-energy feature is valid (monochrome, constant channels).
    // Floors protect internal divisions/logs, not the exported noise model.
    for (int f = 0; f < kFeat; ++f) if (zeroS[f]) {
      const int c = channelOf(f), b = bandOf(f);
      for (int l = 0; l < kL; ++l) {
        if (b >= 0) out->sigma[c][b][l] = 0;
        else out->sigmaFull[c][l] = 0;
      }
    }
    const bool coarseTiming = getenv("OD_COARSE_TIMING") != nullptr;
    const double coarseStart = coarseTiming ? CFAbsoluteTimeGetCurrent() : 0;
    resolveCoarse(info->coarse, out);
    if (coarseTiming && info->coarse.data)
      fprintf(stderr, "[coarse-cost] key=%lld bytes=%lu cpu_ms=%.6f\n", (long long)m.frameKey,
              (unsigned long)info->coarse.data.length, 1000 * (CFAbsoluteTimeGetCurrent() - coarseStart));
    if (debugEnabled()) {
      fprintf(stderr, "[profile] => confidence %.2f, resolve %.2f ms, knots 0x%02x, Y b0:", out->confidence,
              1000 * (CFAbsoluteTimeGetCurrent() - t0), mask);
      for (int l = 0; l < kL; ++l) fprintf(stderr, " %.5f", out->sigma[0][0][l]);
      fprintf(stderr, "\n");
    }
    return true;
  }

  // Called under the session's profile lock (stage.h), so the rings need no
  // lock of their own. One aggregate per grain mode, fed the same estimates
  // (stage.h, smooth): OD_GRAIN_WHOLE's is a function of the set of
  // estimates in its ring, the aggregate of API v9 unchanged; OD_GRAIN_FINEST's
  // holds the knots its ring did not measure (supported knots), so it also
  // depends on the last supported history.
  void smooth(const ODNoiseProfile& fresh, uint32_t knots, ODNoiseProfile* const slots[kGrainModes],
              bool decisive) override {
    if (!fresh.valid) return;
    for (int m = 0; m < kGrainModes; ++m) {
      if (!slots[m]) continue;  // held (Hold profile): this mode's history stays as it is
      if (debugMode()) {
        *slots[m] = fresh;
        continue;
      }
      smoothMode(agg_[m], m == OD_GRAIN_FINEST, fresh, knots, slots[m], decisive);
    }
  }

  // One mode's aggregate. supportedKnots false is the session aggregate of
  // API v9, line for line.
  static void smoothMode(Aggregate& a, bool supportedKnots, const ODNoiseProfile& fresh, uint32_t knots,
                         ODNoiseProfile* slot, bool decisive) {
    if (!slot->valid || a.resetPending.exchange(false) ||
        (supportedKnots && slot->confidence < kUsableConfidence && fresh.confidence >= kUsableConfidence)) {
      // First estimate, or od_session_reset(); finest grain: also the first
      // usable estimate after unusable ones.
      a.ring.clear();
      a.ringKnots.clear();
      a.slotKnots = 0;
      a.streak = 0;
      a.wideStreak = 0;
      a.streakDir = 0;
    }
    if (!a.ring.empty()) {
      // New footage under the same session (an adjustment layer over a cut):
      // several estimates in a row far from the slot, in the same direction.
      // One odd frame does nothing, it is just one of 24 in a trimmed mean,
      // and neither do estimates that scatter either side of it. Where the
      // session saw a cut or a jump (decisive), one far estimate is enough.
      const bool usable = fresh.confidence >= kUsableConfidence || slot->confidence < kUsableConfidence;
      int dir = 0;
      double median = 0;
      const double upper = profileDistance(fresh, knots, *slot, a.slotKnots, &median, &dir);
      const bool differs = usable && upper > std::log(kChangeFactor);
      const bool widely = usable && median > std::log(kChangeFactor);
      if (!differs) {
        a.streak = 0;
        a.wideStreak = 0;
      } else if (a.streak > 0 && dir != a.streakDir) {
        a.streak = 1;  // far the other way: starts a streak of its own
        a.wideStreak = widely ? 1 : 0;
      } else {
        ++a.streak;
        a.wideStreak = widely ? a.wideStreak + 1 : 0;
      }
      a.streakDir = differs ? dir : 0;
      if (debugEnabled())
        fprintf(stderr, "[profile] smooth%s: conf %.2f knots 0x%02x slot 0x%02x distance %.3f (median %.3f) dir %+d "
                "streak %d/%d%s\n", supportedKnots ? " (finest grain)" : "", fresh.confidence, knots, a.slotKnots,
                upper, median, dir, a.streak, a.wideStreak, decisive ? " decisive" : "");
      if (differs && decisive) {
        a.ring.clear();
        a.ringKnots.clear();
        if (supportedKnots) a.slotKnots = 0;
        a.streak = 0;
        a.wideStreak = 0;
      } else if (a.wideStreak >= kChangeStreak || a.streak >= kChangeStreak + 1) {
        // keep the other new ones
        // The new shot must not inherit unmeasured knots from the old one.
        if (supportedKnots) {
          *slot = fresh;
          a.slotKnots = 0;
        }
        const size_t keep = (size_t)a.streak - 1;  // the other new ones
        a.ring.erase(a.ring.begin(), a.ring.end() - (long)std::min(keep, a.ring.size()));
        a.ringKnots.erase(a.ringKnots.begin(), a.ringKnots.end() - (long)std::min(keep, a.ringKnots.size()));
        a.streak = 0;
        a.wideStreak = 0;
      }
    }
    if (a.ring.size() >= kRing) {
      a.ring.erase(a.ring.begin());
      a.ringKnots.erase(a.ringKnots.begin());
    }
    if (supportedKnots && a.ring.empty()) *slot = fresh;  // cold start/reset/cut fallback
    a.ring.push_back(fresh);
    a.ringKnots.push_back(knots);
    aggregate(a.ring, a.ringKnots, slot, supportedKnots);
    // These are qualified by the current supplied window. Never accumulate
    // overlapping windows as independent observations or retain an old coarse
    // decision after a window with no evidence (cuts/jumps included).
    memcpy(slot->coarseSigma, fresh.coarseSigma, sizeof(slot->coarseSigma));
    memcpy(slot->coarseConfidence, fresh.coarseConfidence, sizeof(slot->coarseConfidence));
    memcpy(slot->coarseCoverage, fresh.coarseCoverage, sizeof(slot->coarseCoverage));
    memcpy(slot->coarseCorrelation, fresh.coarseCorrelation, sizeof(slot->coarseCorrelation));
    // Finest grain: held measurements remain evidence for comparison when
    // that tone returns.
    const uint32_t measured = measuredKnots(a.ring, a.ringKnots);
    a.slotKnots = supportedKnots ? a.slotKnots | measured : measured;
    a.ringCount.store((int)a.ring.size());
    a.changing.store(a.streak > 0);
  }

  id<MTLTexture> encodeView(RenderEnv& env, const WorkFrame& cur) override {
    const Grid g = gridFor(env.width, env.height);
    const NSUInteger need = std::max<NSUInteger>(16, (NSUInteger)g.count() * kTileFloats * sizeof(float));
    if (!viewStats_ || viewStats_.length < need) {
      viewStats_ = [env.device() newBufferWithLength:need options:MTLResourceStorageModePrivate];
      if (!viewStats_) {
        env.fail("profile: could not allocate the view statistics buffer");
        return nil;
      }
    }
    if (g.count() > 0 && !encodeStats(env, cur.tex, nil, viewStats_, g)) return nil;
    id<MTLTexture> dst = env.tempWorking();
    if (!dst) return nil;
    ViewParamsGPU p = {};
    p.originX = g.originX;
    p.originY = g.originY;
    p.stride = g.stride;
    p.tilesW = g.tilesW;
    p.tilesH = g.tilesH;
    p.haveProfile = env.profile.valid ? 1 : 0;
    p.thr0 = (float)(1.0 + kClipSigmas * std::sqrt(2.0 / kNu[0]));
    p.thr1 = (float)(1.0 + kClipSigmas * std::sqrt(2.0 / kNu[1]));
    p.coherenceMax = kCoherenceMax;
    p.deadMax = kZeroFracMax;
    for (int l = 0; l < kL; ++l) {
      p.sig0[l] = env.profile.sigma[0][0][l];
      p.sig1[l] = env.profile.sigma[0][1][l];
    }
    id<MTLComputeCommandEncoder> enc = env.compute((psoView_).label.UTF8String);
    [enc setTexture:cur.tex atIndex:0];
    [enc setTexture:dst atIndex:1];
    [enc setBuffer:viewStats_ offset:0 atIndex:0];
    [enc setBytes:&p length:sizeof(p) atIndex:1];
    env.dispatch(psoView_, env.width, env.height);
    return dst;
  }

 private:
  std::mutex analysisMutex_;
  ProfileAnalysisInfo analysis_;
  id<MTLComputePipelineState> coarseDown_ = nil, coarseStats_ = nil;
  bool encodeCoarse(RenderEnv& env, const WorkFrame& cur, const ProfileMeasurement& pm,
                    CoarseMeasurement& m) {
    if (!coarseChromaOn(env.params)) return true;
    if (env.frameCount < 3 || env.frames[0].tex != cur.tex || std::min(env.width, env.height) < 256)
      return true;
    // Widest available pair in the +/-2 window, or three one-sided frames
    // at an endpoint. Honour a decisive pair direction: never look across a
    // cut the session has already excluded for this measurement.
    int a = -1, b = -1;
    for (int i = 1; i < env.frameCount; ++i) {
      int o = env.frames[i].offset;
      if (!o || std::abs(o) > 2 || (pm.pairOffset && o * pm.pairOffset < 0)) continue;
      if (a < 0 || o < env.frames[a].offset) a = i;
      if (b < 0 || o > env.frames[b].offset) b = i;
    }
    if (a < 0 || b < 0 || a == b || env.frames[a].tex == env.frames[b].tex || env.frames[a].tex == cur.tex ||
        env.frames[b].tex == cur.tex)
      return true;
    uint32_t w[8], h[8];
    w[0] = env.width;
    h[0] = env.height;
    for (int k = 1; k < 8; ++k) {
      w[k] = (w[k - 1] + 1) / 2;
      h[k] = (h[k - 1] + 1) / 2;
    }
    uint32_t total = 0;
    for (int b = 0; b < 7; ++b) {
      int k = b < 4 ? b + 3 : b + 1;
      m.w[b] = w[k];
      m.h[b] = h[k];
      m.base[b] = total;
      total += std::max(0, (int(w[k]) - 4) / 16) * std::max(0, (int(h[k]) - 4) / 16);
    }
    if (!total) return true;
    m.data = [env.device() newBufferWithLength:(NSUInteger)total * 20 * sizeof(float)
                                     options:MTLResourceStorageModeShared];
    if (!m.data) {
      env.fail("profile: coarse statistics allocation failed");
      return false;
    }
    m.frames = 3;
    id<MTLTexture> inputs[3] = {cur.tex, env.frames[a].tex, env.frames[b].tex};
    id<MTLTexture> pyr[3][8] = {};
    for (int t = 0; t < 3; ++t) pyr[t][0] = inputs[t];
    for (int k = 1; k < 8; ++k) {
      auto e = env.compute();
      for (int t = 0; t < 3; ++t) {
        pyr[t][k] = env.temp(kWorkingFormat, w[k], h[k]);
        if (!pyr[t][k]) return false;
        [e setTexture:pyr[t][k-1] atIndex:t];
        [e setTexture:pyr[t][k] atIndex:t+3];
      }
      [e setComputePipelineState:coarseDown_];
      // Keep independent tent samples in 256-thread groups. Max-size groups
      // reduce occupancy on this texture-bound kernel. This changes no samples.
      [e dispatchThreads:MTLSizeMake(w[k],h[k],3) threadsPerThreadgroup:MTLSizeMake(16,16,1)];
      if (k == 2 || k == 3)
        for (int t = 0; t < 3; ++t) env.recycle(pyr[t][k-1]);
    }
    for (int b = 0; b < 7; ++b) {
      const int k = b < 4 ? b + 3 : b + 1;
      const uint32_t nx = std::max(0, (int(w[k]) - 4) / 16), ny = std::max(0, (int(h[k]) - 4) / 16);
      if (!nx || !ny) continue;
      auto e = env.compute();
      for (int t = 0; t < 3; ++t) {
        [e setTexture:pyr[t][k] atIndex:2*t];
        [e setTexture:pyr[t][b < 4 ? k+1 : k] atIndex:2*t+1];
      }
      uint32_t u[4] = {m.base[b], b < 4 ? 0u : 1u, nx, 0};
      [e setBuffer:m.data offset:0 atIndex:0];
      [e setBytes:u length:sizeof(u) atIndex:1];
      [e setComputePipelineState:coarseStats_];
      [e dispatchThreadgroups:MTLSizeMake(nx, ny, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    for (int t = 0; t < 3; ++t)
      for (int k = 3; k < 8; ++k) env.recycle(pyr[t][k]);
    return !env.failed();
  }
  // l0, l1, b3, l3, sum2 x 2 and tiles for one image: the frame, or the frame difference. Every
  // texture it takes is handed back to the pool for the next chain.
  bool encodeChain(RenderEnv& env, id<MTLTexture> image, bool isDiff, id<MTLBuffer> stats, const Grid& g,
                   uint32_t base) {
    const uint32_t wh = (env.width + 1) / 2, hh = (env.height + 1) / 2;
    const MTLPixelFormat fmt32 = MTLPixelFormatRGBA32Float;
    id<MTLTexture> c1 = env.temp(fmt32, wh, hh), e0 = env.temp(fmt32, wh, hh);
    if (!c1 || !e0) return false;
    {
      id<MTLComputeCommandEncoder> enc = env.compute((isDiff ? psoL0Diff_ : psoL0_).label.UTF8String);
      [enc setTexture:image atIndex:0];
      [enc setTexture:c1 atIndex:1];
      [enc setTexture:e0 atIndex:2];
      env.dispatch(isDiff ? psoL0Diff_ : psoL0_, wh, hh);
    }
    const uint32_t wq = ((uint32_t)c1.width + 1) / 2, hq = ((uint32_t)c1.height + 1) / 2;
    const uint32_t we = (wq + 1) / 2, he = (hq + 1) / 2;  // 8-pixel blocks
    const uint32_t wb = (we + 1) / 2, hb = (he + 1) / 2;  // 16-pixel blocks
    const MTLPixelFormat fmt = MTLPixelFormatRGBA32Float;
    id<MTLTexture> c2 = env.temp(fmt, wq, hq), c3 = env.temp(fmt, wq, hq);
    // Sums of E0, E1, J, V2, V3, V4 per cell, per 8-pixel and per 16-pixel block.
    id<MTLTexture> a[6], e[6], b[6];
    bool have = c2 && c3;
    for (int k = 0; k < 6; ++k) have = (a[k] = env.temp(fmt, wq, hq)) != nil && have;
    for (int k = 0; k < 6; ++k) have = (e[k] = env.temp(fmt, we, he)) != nil && have;
    for (int k = 0; k < 6; ++k) have = (b[k] = env.temp(fmt, wb, hb)) != nil && have;
    if (!have) return false;

    id<MTLComputeCommandEncoder> enc = env.compute((psoL1_).label.UTF8String);
    [enc setTexture:c1 atIndex:0];
    [enc setTexture:c2 atIndex:1];
    [enc setTexture:a[1] atIndex:2];
    [enc setTexture:a[2] atIndex:3];
    [enc setTexture:e0 atIndex:4];
    [enc setTexture:a[0] atIndex:5];
    env.dispatch(psoL1_, wq, hq);

    enc = env.compute((psoB3_).label.UTF8String);
    [enc setTexture:c2 atIndex:0];
    [enc setTexture:c3 atIndex:1];
    env.dispatch(psoB3_, wq, hq);

    enc = env.compute((psoL3_).label.UTF8String);
    [enc setTexture:c3 atIndex:0];
    [enc setTexture:c2 atIndex:1];
    [enc setTexture:c1 atIndex:2];
    [enc setTexture:image atIndex:3];
    [enc setTexture:a[3] atIndex:4];
    [enc setTexture:a[4] atIndex:5];
    [enc setTexture:a[5] atIndex:6];
    env.dispatch(psoL3_, wq, hq);

    enc = env.compute((psoSum2_).label.UTF8String);
    for (int k = 0; k < 6; ++k) [enc setTexture:a[k] atIndex:(NSUInteger)k];
    for (int k = 0; k < 6; ++k) [enc setTexture:e[k] atIndex:6 + (NSUInteger)k];
    env.dispatch(psoSum2_, we, he);
    enc = env.compute((psoSum2_).label.UTF8String);
    for (int k = 0; k < 6; ++k) [enc setTexture:e[k] atIndex:(NSUInteger)k];
    for (int k = 0; k < 6; ++k) [enc setTexture:b[k] atIndex:6 + (NSUInteger)k];
    env.dispatch(psoSum2_, wb, hb);

    enc = env.compute((psoTiles_).label.UTF8String);
    TileParamsGPU tp = {g.originX, g.originY, g.stride, g.tilesW, g.tilesH, base, 0, 0};
    for (int k = 0; k < 6; ++k) [enc setTexture:b[k] atIndex:(NSUInteger)k];
    [enc setBuffer:stats offset:0 atIndex:0];
    [enc setBytes:&tp length:sizeof(tp) atIndex:1];
    env.dispatch(psoTiles_, g.tilesW, g.tilesH);

    env.recycle(c1);
    env.recycle(e0);
    env.recycle(c2);
    env.recycle(c3);
    for (int k = 0; k < 6; ++k) env.recycle(a[k]);
    for (int k = 0; k < 6; ++k) env.recycle(e[k]);
    for (int k = 0; k < 6; ++k) env.recycle(b[k]);
    return !env.failed();
  }

  bool encodeStats(RenderEnv& env, id<MTLTexture> cur, id<MTLTexture> prev, id<MTLBuffer> stats, const Grid& g) {
    if (!encodeChain(env, cur, false, stats, g, 0)) return false;
    if (prev) {
      id<MTLTexture> diff = env.temp(MTLPixelFormatRGBA16Float, env.width, env.height);
      if (!diff) return false;
      id<MTLComputeCommandEncoder> enc = env.compute((psoDiff_).label.UTF8String);
      [enc setTexture:cur atIndex:0];
      [enc setTexture:prev atIndex:1];
      [enc setTexture:diff atIndex:2];
      env.dispatch(psoDiff_, env.width, env.height);
      if (!encodeChain(env, diff, true, stats, g, kRecordFloats)) return false;
      env.recycle(diff);
    }
    return !env.failed();
  }

  std::shared_ptr<BufferPool> pool_;
  id<MTLComputePipelineState> psoL0_ = nil, psoDiff_ = nil, psoL0Diff_ = nil, psoL1_ = nil, psoB3_ = nil,
                              psoL3_ = nil, psoSum2_ = nil, psoTiles_ = nil, psoView_ = nil;
  id<MTLBuffer> viewStats_ = nil;  // render thread only

  // Throttle (render thread, except the atomics written by smooth()).
  static constexpr int kWarmup = 4;         // unconditional measurements after a pause or a size change
  static constexpr int kThrottleAfter = 8;  // estimates in the ring before throttling starts
  static constexpr uint64_t kEvery = 4;
  static constexpr uint64_t kSteadyEvery = 8;  // a multiple of kEvery: with a full ring
  double lastCall_ = 0;
  int warmup_ = kWarmup;
  uint64_t counter_ = 0;
  static constexpr uint64_t kTemporalEvery = 4;
  uint64_t measureCount_ = 0;
  std::atomic<bool> temporalUseful_{true};  // written by resolve()

  Aggregate agg_[kGrainModes];
};

}  // namespace

std::unique_ptr<ProfileStage> makeProfileStage(Context& ctx, std::string* err) {
  auto p = std::make_unique<Profiler>(ctx, err);
  if (!p->ok()) return nullptr;
  return p;
}

}  // namespace od
