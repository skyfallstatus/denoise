// SPDX-License-Identifier: MIT
// Ground truth for the noise profiler tests: the true
// ODNoiseProfile of a noisy frame whose clean version is known. CPU, every pixel, the band
// definition of (undecimated B3 a-trous, mirrored borders), in the working space of
// core/shaders/common.metal.
#ifndef OD_PROFILE_TRUTH_H
#define OD_PROFILE_TRUTH_H

#include <dispatch/dispatch.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "opendenoise/opendenoise.h"

namespace odtest {

// The truth is kept in fine luminance bins (6 per knot interval) rather than
// at the 8 knots: noise can change by a factor 2 between two knots, and a
// frame rarely populates a whole knot interval evenly, so "the sigma at a
// knot" is not well defined by pooling. What matters is what a consumer gets
// when it interpolates the profile at the luminance of a pixel; that is what
// RatioStats compares, bin by bin.
constexpr int kTruthBins = 6 * (OD_LEVELS - 1);

struct Truth {
  double sigma[3][OD_BANDS + 1][kTruthBins];  // [.][OD_BANDS] = full band (c_0 - c_4)
  double share[kTruthBins];                     // share of the frame's pixels in each bin
  double meanY[kTruthBins];                     // their mean luminance
};

inline void toYcc(const float* bgra, size_t n, std::vector<float>* y, std::vector<float>* cb, std::vector<float>* cr) {
  y->resize(n);
  cb->resize(n);
  cr->resize(n);
  for (size_t i = 0; i < n; ++i) {
    const float b = bgra[i * 4], g = bgra[i * 4 + 1], r = bgra[i * 4 + 2];
    const float yy = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    (*y)[i] = yy;
    (*cb)[i] = (b - yy) * (1.0f / 1.8556f);
    (*cr)[i] = (r - yy) * (1.0f / 1.5748f);
  }
}

inline int mirror(int i, int n) {
  if (i < 0) i = -i;
  if (i >= n) i = 2 * n - 2 - i;
  return std::min(std::max(i, 0), n - 1);
}

// c_(k+1) = B3 with taps 2^k apart, separable.
inline void b3(const std::vector<float>& src, std::vector<float>* dst, uint32_t w, uint32_t h, int step) {
  std::vector<float> tmp(src.size());
  dst->resize(src.size());
  const float* s = src.data();
  float* t = tmp.data();
  float* d = dst->data();
  dispatch_apply(h, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t y) {
    for (uint32_t x = 0; x < w; ++x) {
      const float* row = s + y * w;
      t[y * w + x] = (row[mirror((int)x - 2 * step, (int)w)] + row[mirror((int)x + 2 * step, (int)w)]) * 0.0625f +
                     (row[mirror((int)x - step, (int)w)] + row[mirror((int)x + step, (int)w)]) * 0.25f + row[x] * 0.375f;
    }
  });
  dispatch_apply(h, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t y) {
    const float* r0 = t + (size_t)mirror((int)y - 2 * step, (int)h) * w;
    const float* r1 = t + (size_t)mirror((int)y - step, (int)h) * w;
    const float* r2 = t + y * w;
    const float* r3 = t + (size_t)mirror((int)y + step, (int)h) * w;
    const float* r4 = t + (size_t)mirror((int)y + 2 * step, (int)h) * w;
    for (uint32_t x = 0; x < w; ++x) d[y * w + x] = (r0[x] + r4[x]) * 0.0625f + (r1[x] + r3[x]) * 0.25f + r2[x] * 0.375f;
  });
}

// noisy, clean: BGRA float, w*h*4. The luminance of a pixel is c_4.Y of the
// NOISY frame (what the profiler and the consumers see). `margin` pixels at
// the border are left out (mirroring changes the coarse bands there).
inline void measureTruth(const float* noisy, const float* clean, uint32_t w, uint32_t h, Truth* out, int margin = 32) {
  const size_t n = (size_t)w * h;
  std::vector<float> ny, ncb, ncr, cy, ccb, ccr;
  toYcc(noisy, n, &ny, &ncb, &ncr);
  toYcc(clean, n, &cy, &ccb, &ccr);
  // luminance: c_4 of noisy Y
  std::vector<float> lum = ny, tmp;
  for (int k = 0; k < 4; ++k) {
    b3(lum, &tmp, w, h, 1 << k);
    lum.swap(tmp);
  }
  static double acc[3][OD_BANDS + 1][kTruthBins];
  double wsum[kTruthBins] = {}, ysum[kTruthBins] = {};
  for (auto& ch : acc)
    for (auto& band : ch)
      for (double& v : band) v = 0;
  std::vector<float>* nz[3] = {&ny, &ncb, &ncr};
  std::vector<float>* cz[3] = {&cy, &ccb, &ccr};
  if ((int)w <= 2 * margin + 8 || (int)h <= 2 * margin + 8) margin = 0;
  for (int c = 0; c < 3; ++c) {
    std::vector<float> c0(n), ck, cn;
    for (size_t i = 0; i < n; ++i) c0[i] = (*nz[c])[i] - (*cz[c])[i];
    ck = c0;
    for (int b = 0; b <= OD_BANDS; ++b) {
      const bool full = b == OD_BANDS;
      if (!full) b3(ck, &cn, w, h, 1 << b);
      // after the last band, ck holds c_4: full band = c_0 - c_4
      for (uint32_t y = (uint32_t)margin; y < h - (uint32_t)margin; ++y) {
        for (uint32_t x = (uint32_t)margin; x < w - (uint32_t)margin; ++x) {
          const size_t i = (size_t)y * w + x;
          const double d = full ? (double)c0[i] - ck[i] : (double)ck[i] - cn[i];
          const double yl = std::min(1.0, std::max(0.0, (double)lum[i]));
          const int bin = std::min(kTruthBins - 1, (int)(yl * kTruthBins));
          acc[c][b][bin] += d * d;
          if (c == 0 && b == 0) {
            wsum[bin] += 1;
            ysum[bin] += yl;
          }
        }
      }
      if (!full) ck.swap(cn);
    }
  }
  double total = 0;
  for (int l = 0; l < kTruthBins; ++l) total += wsum[l];
  for (int l = 0; l < kTruthBins; ++l) {
    out->share[l] = total > 0 ? wsum[l] / total : 0;
    out->meanY[l] = wsum[l] > 0 ? ysum[l] / wsum[l] : (l + 0.5) / kTruthBins;
    for (int c = 0; c < 3; ++c)
      for (int b = 0; b <= OD_BANDS; ++b) out->sigma[c][b][l] = wsum[l] > 0 ? std::sqrt(acc[c][b][l] / wsum[l]) : 0;
  }
}

// Linear interpolation between knots, as od_sigma_at in common.metal.
inline double profileAt(const float* knots, double y) {
  const double t = std::min(1.0, std::max(0.0, y)) * (OD_LEVELS - 1);
  const int i = std::min((int)t, OD_LEVELS - 2);
  return knots[i] + (t - i) * (knots[i + 1] - knots[i]);
}

// Bins holding at least minShare of the frame, one line per bin: luminance,
// share, then the true sigma of Y b0..b3, full, and Cb / Cr b0 and full.
inline void printTruth(const Truth& t, double minShare = 0.005) {
  printf("truth (bins with >= %.1f %% of the frame)\n", 100 * minShare);
  printf("     Y    share    Y b0     Y b1     Y b2     Y b3     Y bF    Cb b0    Cb bF    Cr b0    Cr bF\n");
  for (int l = 0; l < kTruthBins; ++l) {
    if (t.share[l] < minShare) continue;
    printf("  %.3f %5.1f%% %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f %8.5f\n", t.meanY[l], 100 * t.share[l],
           t.sigma[0][0][l], t.sigma[0][1][l], t.sigma[0][2][l], t.sigma[0][3][l], t.sigma[0][4][l], t.sigma[1][0][l],
           t.sigma[1][4][l], t.sigma[2][0][l], t.sigma[2][4][l]);
  }
}

// Collects (estimate interpolated at the bin's luminance) / (true sigma of
// the bin), weighted by the bin's share of the frame, per channel and band.
struct RatioStats {
  std::vector<std::pair<double, double>> r[3][OD_BANDS + 1];  // ratio, weight
  void add(const ODNoiseProfile& p, const Truth& t, double minShare = 0.005, double maxY = 2.0) {
    for (int l = 0; l < kTruthBins; ++l) {
      if (t.share[l] < minShare || t.meanY[l] > maxY) continue;
      for (int c = 0; c < 3; ++c) {
        for (int b = 0; b <= OD_BANDS; ++b) {
          const double est = profileAt(b < OD_BANDS ? p.sigma[c][b] : p.sigmaFull[c], t.meanY[l]);
          // (a bin without noise, e.g. a blown-out area, has no ratio)
          if (t.sigma[c][b][l] > 1e-6 && est > 0) r[c][b].emplace_back(est / t.sigma[c][b][l], t.share[l]);
        }
      }
    }
  }
  // Weighted quantile.
  static double q(std::vector<std::pair<double, double>> v, double f) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    double total = 0, acc = 0;
    for (auto& e : v) total += e.second;
    for (auto& e : v) {
      acc += e.second;
      if (acc >= f * total) return e.first;
    }
    return v.back().first;
  }
  double median(int c, int b) const { return q(r[c][b], 0.5); }
  double lo(int c, int b) const { return q(r[c][b], 0.05); }
  double hi(int c, int b) const { return q(r[c][b], 0.95); }
  void print(const char* title) const {
    printf("%s: pixel-weighted median [5 %% .. 95 %%] of estimate / truth, %zu bins\n", title, r[0][0].size());
    for (int c = 0; c < 3; ++c) {
      printf("  %s", c == 0 ? "Y " : c == 1 ? "Cb" : "Cr");
      for (int b = 0; b <= OD_BANDS; ++b)
        printf("  b%c %.2f [%.2f..%.2f]", b < OD_BANDS ? (char)('0' + b) : 'F', median(c, b), lo(c, b), hi(c, b));
      printf("\n");
    }
  }
};

}  // namespace odtest

#endif
