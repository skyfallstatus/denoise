// SPDX-License-Identifier: MIT
// Original double-precision sampled reference, retained ONLY in the tests.
// Independent oracle for the GPU sufficient-statistics implementation.
struct CoarseReferenceMeasurement {
  id<MTLBuffer> data = nil;
  uint32_t w[7] = {}, h[7] = {}, base[3][7] = {};
  int frames = 0;
};

// Non-overlapping 16x16 regions, with an orthonormal quadratic surface removed.
// Orthogonal fits remove constant colour, ramps and illumination curvature.
// Dynamic residual correlation, not the raw colour, must be between .35 and
// .96 in BOTH chroma channels. Static colour and independent noise fail.
static void resolveCoarseReference(const CoarseReferenceMeasurement& m, ODNoiseProfile* p) {
  if (!m.data || m.frames != 3) return;
  constexpr int N = 16, NN = N * N, F = 6;
  double basis[NN][F], norm[F] = {};
  const double mean2 = (N * N - 1) / 12.0;
  for (int y = 0; y < N; ++y)
    for (int x = 0; x < N; ++x) {
      int i = y * N + x;
      double u = x - (N - 1) * .5, v = y - (N - 1) * .5;
      double a[F] = {1, u, v, u * u - mean2, v * v - mean2, u * v};
      for (int f = 0; f < F; ++f) {
        basis[i][f] = a[f];
        norm[f] += a[f] * a[f];
      }
    }
  for (int i = 0; i < NN; ++i)
    for (int f = 0; f < F; ++f) basis[i][f] /= std::sqrt(norm[f]);
  const float* data = (const float*)m.data.contents;
  for (int b = 0; b < 7; ++b) {
    const int level = b < 4 ? b + 3 : b + 1;
    struct Region {
      double y, sig[2], rho, weight;
    };
    std::vector<Region> regions;
    const int w = m.w[b], h = m.h[b];
    // A two-sample boundary excludes clamped pyramid samples.
    for (int oy = 2; oy + N <= h - 2; oy += N)
      for (int ox = 2; ox + N <= w - 2; ox += N) {
        double r[3][3][NN], energy[3][3] = {}, lum = 0;
        bool finite = true;
        for (int t = 0; t < 3; ++t)
          for (int c = 0; c < 3; ++c) {
            double coef[F] = {};
            for (int y = 0; y < N; ++y)
              for (int x = 0; x < N; ++x) {
                int i = y * N + x;
                const float* q = data + 4 * (m.base[t][b] + (oy + y) * w + ox + x);
                double v = q[c];
                finite = finite && std::isfinite(v) && std::fabs(v) < 16;
                r[t][c][i] = v;
                if (t == 0 && c == 0) lum += q[3] / NN;
                for (int f = 0; f < F; ++f) coef[f] += v * basis[i][f];
              }
            for (int i = 0; i < NN; ++i) {
              for (int f = 0; f < F; ++f) r[t][c][i] -= coef[f] * basis[i][f];
              energy[t][c] += r[t][c][i] * r[t][c][i] / (NN - F);
            }
          }
        if (!finite || !std::isfinite(lum) || lum < .02 || lum > .98) continue;
        double rho[2], sig[2];
        bool ok = true;
        for (int c = 1; c < 3; ++c) {
          double e = (energy[0][c] + energy[1][c] + energy[2][c]) / 3;
          if (e < 1e-10) {
            ok = false;
            break;
          }
          double cross = 0;
          for (int t = 0; t < 3; ++t)
            for (int u = t + 1; u < 3; ++u) {
              double dot = 0;
              for (int i = 0; i < NN; ++i) dot += r[t][c][i] * r[u][c][i] / (NN - F);
              cross += dot / std::sqrt(std::max(energy[t][c] * energy[u][c], 1e-30)) / 3;
            }
          double fourth = 0;
          for (int i = 0; i < NN; ++i) fourth += std::pow(r[0][c][i], 4) / NN;
          // Kurtosis rejects sparse saturated features; a changing edge also
          // has high residual luma or strong colour-channel coherence below.
          if (cross < .35 || cross > .96 || fourth / (energy[0][c] * energy[0][c] + 1e-30) > 5.5) ok = false;
          rho[c - 1] = std::clamp(cross, 0.0, 1.0);
          sig[c - 1] = std::sqrt(e);
        }
        if (!ok) continue;
        double cc = 0;
        for (int i = 0; i < NN; ++i) cc += r[0][1][i] * r[0][2][i] / (NN - F);
        cc /= std::sqrt(energy[0][1] * energy[0][2] + 1e-30);
        if (std::fabs(cc) > .85 || std::fabs(rho[0] - rho[1]) > .30) continue;
        const double yl = std::clamp(lum * 7, 0.0, 7.0);
        int l = std::min(int(yl), 6);
        double a = yl - l;
        auto at = [&](int c, int band) {
          return (1 - a) * p->sigma[c][band][l] + a * p->sigma[c][band][l + 1];
        };
        const double sy = at(0, 3) * 1.36 * std::ldexp(1.0, 3 - level);
        if (std::sqrt(energy[0][0]) > std::max(5 * sy, .0015)) continue;
        // Evidence of excess relative to the OLD model, before any slider.
        for (int c = 0; c < 2; ++c) {
          double s3 = at(c + 1, 3), s2 = at(c + 1, 2),
                 slope = std::clamp(s3 / std::max(s2, 1e-9) / .482, 1.0, 2.0);
          double old = b == 0
                           ? 1.36 * s3
                           : (b == 1 ? .65 : (b < 4 ? .32 : .41) * std::ldexp(1.0, 5 - level)) * s3 * slope;
          if (sig[c] < 1.20 * old) ok = false;
        }
        if (ok) regions.push_back({lum, {sig[0], sig[1]}, .5 * (rho[0] + rho[1]), 1.0});
      }
    for (int l = 0; l < OD_LEVELS; ++l) {
      std::vector<double> s[2], corr;
      double weight = 0;
      for (const auto& r : regions) {
        double wt = std::max(0.0, 1.0 - std::fabs(r.y * 7 - l));
        if (wt < .2) continue;
        weight += wt;
        s[0].push_back(r.sig[0]);
        s[1].push_back(r.sig[1]);
        corr.push_back(r.rho);
      }
      const double coverage = weight * NN / std::max(w * h, 1);
      p->coarseCoverage[b][l] = coverage;
      // Spatial support only. N_eff=3/(1+2*rho), not three independent
      // observations. We conservatively give NO confidence bonus for N_eff.
      if (weight < 2 || s[0].size() < 3) continue;
      auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
      };
      double med[2] = {median(s[0]), median(s[1])}, rho = median(corr);
      bool agree = true;
      for (int c = 0; c < 2; ++c) {
        std::sort(s[c].begin(), s[c].end());
        if (s[c][s[c].size() * 3 / 4] > 2 * s[c][s[c].size() / 4]) agree = false;
      }
      if (!agree) continue;
      float confidence = std::min(1.0, weight / 4.0) * std::min(1.0, coverage / .025);
      p->coarseConfidence[b][l] = confidence;
      p->coarseCorrelation[b][l] = rho;
      for (int c = 0; c < 2; ++c) p->coarseSigma[c][b][l] = med[c];
      if (getenv("OD_COARSE_DEBUG"))
        fprintf(stderr,
                "[coarse] b=%d l=%d regions=%zu weight=%.2f coverage=%.4f confidence=%.3f rho=%.3f "
                "sigma=%.7f,%.7f\n",
                b, l, s[0].size(), weight, coverage, confidence, rho, med[0], med[1]);
    }
  }
}
