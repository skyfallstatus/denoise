// SPDX-License-Identifier: MIT
// Private implementation of the direct coarse profile. Included by profile.mm.
// Independent from the a-trous floor fit to keep rejection bit-exact.
struct CoarseMeasurement {
  id<MTLBuffer> data = nil;
  uint32_t w[7] = {}, h[7] = {}, base[7] = {};
  int frames = 0;
};

// Non-overlapping 16x16 regions, with an orthonormal quadratic surface removed.
// Orthogonal fits remove constant colour, ramps and illumination curvature.
// Dynamic residual correlation, not the raw colour, must be between .35 and
// .96 in BOTH chroma channels. Static colour and independent noise fail.
static void resolveCoarse(const CoarseMeasurement& m, ODNoiseProfile* p) {
  if (!m.data || m.frames != 3) return;
  constexpr int N = 16, NN = N * N;
  const float* data = (const float*)m.data.contents;
  dispatch_apply(7, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t band) {
    const int b = (int)band;
    const int level = b < 4 ? b + 3 : b + 1;
    struct Region {
      double y, sig[2], rho;
    };
    std::vector<Region> regions;
    const int w = m.w[b], h = m.h[b];
    // One 20-float sufficient-statistics record per non-overlapping region.
    // GPU removes the same orthonormal quadratic before reducing moments;
    // subtracting two large raw moments here would lose low-noise precision.
    const int nx = std::max(0, (w - 4) / N), ny = std::max(0, (h - 4) / N);
    regions.reserve(nx * ny);
    for (int i = 0; i < nx * ny; ++i) {
      const float* q = data + 20 * (m.base[b] + i);
      double energy[3][3];
      for (int t = 0; t < 3; ++t)
        for (int c = 0; c < 3; ++c) energy[t][c] = q[t * 3 + c];
      const double lum = q[18];
      bool finite = q[19] == 0;
      for (int k = 0; k < 19; ++k) finite = finite && std::isfinite(q[k]);
      if (!finite || lum < .02 || lum > .98) continue;
      double rho[2], sig[2];
      bool ok = true;
      for (int c = 1; c < 3; ++c) {
        double e = (energy[0][c] + energy[1][c] + energy[2][c]) / 3;
        if (e < 1e-10) {
          ok = false;
          break;
        }
        double cross = 0;
        int j = 9 + (c - 1) * 3;
        for (int t = 0; t < 3; ++t)
          for (int u = t + 1; u < 3; ++u)
            cross += q[j++] / std::sqrt(std::max(energy[t][c] * energy[u][c], 1e-30)) / 3;
        if (cross < .35 || cross > .96 || q[15 + c] / (energy[0][c] * energy[0][c] + 1e-30) > 5.5)
          ok = false;
        rho[c - 1] = std::clamp(cross, 0.0, 1.0);
        sig[c - 1] = std::sqrt(e);
      }
      if (!ok) continue;
      double cc = q[15] / std::sqrt(energy[0][1] * energy[0][2] + 1e-30);
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
      if (ok) regions.push_back({lum, {sig[0], sig[1]}, .5 * (rho[0] + rho[1])});
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
  });
}
