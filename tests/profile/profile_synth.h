// SPDX-License-Identifier: MIT
// Synthetic frames with known noise for the profiler tests: a clean scene in
// working space (Y, Cb, Cr) plus noise whose standard deviation depends on
// luminance, differs per channel and is not white. Everything is generated in
// float and converted to BGRA with the inverse of the matrix in
// core/shaders/common.metal. The truth is then measured from (noisy - clean)
// by profile_truth.h, so it is exact for the realisation, whatever the
// spectrum.
#ifndef OD_PROFILE_SYNTH_H
#define OD_PROFILE_SYNTH_H

#include <dispatch/dispatch.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace odtest {

inline uint32_t hash32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

// Standard normal, deterministic in (seed, index).
inline float gauss(uint32_t seed, uint32_t i) {
  const uint32_t a = hash32(seed ^ (i * 0x9E3779B9u));
  const uint32_t b = hash32(a ^ 0x85EBCA6Bu);
  const float u1 = ((float)(a >> 8) + 0.5f) * (1.0f / 16777216.0f);
  const float u2 = ((float)(b >> 8) + 0.5f) * (1.0f / 16777216.0f);
  return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

inline void whiteField(std::vector<float>* f, uint32_t w, uint32_t h, uint32_t seed) {
  f->resize((size_t)w * h);
  float* p = f->data();
  dispatch_apply(h, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t y) {
    for (uint32_t x = 0; x < w; ++x) p[y * w + x] = gauss(seed, (uint32_t)(y * w + x));
  });
}

// Smooth low-pass: separable [1 2 1]/4 with tap spacings 1, 2, 4, ...
// (`levels` of them, so the result is smooth over about 2^levels pixels),
// wrapping around, then rescaled to unit variance.
inline void blurUnit(std::vector<float>* f, uint32_t w, uint32_t h, int levels) {
  std::vector<float> tmp(f->size());
  for (int p = 0; p < levels; ++p) {
    const uint32_t spacing = 1u << p;
    float* s = f->data();
    float* t = tmp.data();
    dispatch_apply(h, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t y) {
      for (uint32_t x = 0; x < w; ++x) {
        const uint32_t xm = (x + w - spacing % w) % w, xp = (x + spacing) % w;
        t[y * w + x] = 0.25f * (s[y * w + xm] + s[y * w + xp]) + 0.5f * s[y * w + x];
      }
    });
    dispatch_apply(h, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t y) {
      const size_t ym = (y + h - spacing % h) % h, yp = (y + spacing) % h;
      for (uint32_t x = 0; x < w; ++x) s[y * w + x] = 0.25f * (t[ym * w + x] + t[yp * w + x]) + 0.5f * t[y * w + x];
    });
  }
  double m2 = 0;
  for (float v : *f) m2 += (double)v * v;
  const float k = (float)(1.0 / std::sqrt(m2 / (double)f->size()));
  for (float& v : *f) v *= k;
}

enum class Scene {
  Grey,      // one uniform grey: calibration
  Patches,   // flat patches at many luminance levels, nothing else
  Mixed,     // flat patches + ramps + edges + regions of fine and coarse texture
  Textured,  // texture of widely varying strength everywhere: no flat tile at all
  DarkClip,  // mostly black with clipped (exactly constant) areas and a few lit flat patches
  LogNight,  // night footage in log: 40 % of the frame at Y 0.09-0.13 (the toe), a third where the noise
             // falls (0.15-0.30), flat patches; two levels only on texture, one of them below the toe patches
};

struct Ycc {
  std::vector<float> y, cb, cr;
};

// The clean scene. `shift` translates it horizontally (motion between frames).
inline void makeScene(Scene scene, uint32_t w, uint32_t h, uint32_t seed, int shift, Ycc* out) {
  const size_t n = (size_t)w * h;
  out->y.assign(n, 0.0f);
  out->cb.assign(n, 0.0f);
  out->cr.assign(n, 0.0f);
  std::vector<float> fine, coarse;
  whiteField(&fine, w, h, seed * 7 + 1);
  blurUnit(&fine, w, h, 1);  // isotropic fine texture, energy in bands 0 and 1
  whiteField(&coarse, w, h, seed * 7 + 2);
  blurUnit(&coarse, w, h, 4);  // soft blobs about 16 px across, energy in bands 2 and 3
  std::vector<float> broad;
  whiteField(&broad, w, h, seed * 7 + 3);
  blurUnit(&broad, w, h, 6);  // varies over about 64 px: regions, larger than a tile
  const float* br = broad.data();
  float* Y = out->y.data();
  float* Cb = out->cb.data();
  float* Cr = out->cr.data();
  const float* fn = fine.data();
  const float* cs = coarse.data();
  dispatch_apply(h, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t yy) {
    for (uint32_t x = 0; x < w; ++x) {
      const uint32_t sx = (uint32_t)(((int)x + shift) % (int)w + (int)w) % w;
      const size_t si = yy * w + sx;
      const float u = (float)sx / (float)w, v = (float)yy / (float)h;
      // 8 x 6 grid of patches; luminance steps over columns and rows
      const int px = std::min(7, (int)(u * 8)), py = std::min(5, (int)(v * 6));
      const float level = 0.04f + 0.92f * ((float)((px * 5 + py * 3) % 16) / 15.0f);
      float y = level, cb = 0.03f * (float)((px % 3) - 1), cr = 0.03f * (float)((py % 3) - 1);
      if (scene == Scene::Grey) {
        y = 0.5f;
        cb = cr = 0.0f;
      } else if (scene == Scene::Mixed) {
        const int kind = (px + 2 * py) % 6;
        if (kind == 1) y += 0.10f * (u * 8 - (float)px - 0.5f);                  // linear ramp
        if (kind == 2) y += 0.05f * fn[si];                                      // fine texture
        if (kind == 3) y += 0.03f * cs[si];                                      // soft blobs
        if (kind == 4) y += ((int)(u * 160) & 1) ? 0.06f : -0.06f;               // stripes (oriented)
        if (kind == 5) y += 0.03f * fn[si] * (0.2f + 2.0f * fabsf(cs[si]));      // texture of varying strength
        if (kind == 2 || kind == 5) cb += 0.01f * fn[si];
      } else if (scene == Scene::Textured) {
        // Log-normal strength that changes from region to region, so tiles
        // differ and there is no sharp lower edge (a sharp one cannot be told
        // from a noise floor: see the uniform_texture test).
        const float strength = 0.03f * expf(0.8f * br[si]);
        y = 0.15f + 0.6f * v + strength * fn[si] + 0.04f * cs[si];
        cb = 0.02f * cs[si] + 0.3f * strength * fn[si];
        cr = -0.02f * cs[si];
      } else if (scene == Scene::LogNight) {
        // 48 patches in a fixed shuffle: 20 in the toe (Y 0.09-0.13), 16 where
        // the noise falls (Y 0.15-0.30, 0.01 apart), 8 brighter ones (2 of
        // them textured), and two levels that occur only on texture: Y 0.075
        // (below every flat toe patch; texture 2 to 8 times the noise there)
        // and Y 0.48, two patches each.
        const int perm = ((px * 6 + py) * 29) % 48;
        static const float hi[8] = {0.33f, 0.38f, 0.44f, 0.52f, 0.60f, 0.68f, 0.76f, 0.85f};
        if (perm < 20) y = 0.09f + 0.04f * (float)(perm % 10) / 9.0f;
        else if (perm < 36) y = 0.15f + 0.01f * (float)(perm - 20);
        else if (perm < 44) y = hi[perm - 36];
        else if (perm < 46) y = 0.075f + 0.02f * fn[si] * (0.4f + 1.2f * fabsf(cs[si]));
        else y = 0.48f + 0.04f * fn[si] * (0.4f + 1.2f * fabsf(cs[si]));
        if (perm == 37 || perm == 41) y += 0.04f * fn[si];
        cb = 0.01f * (float)((px % 3) - 1);
        cr = 0.01f * (float)((py % 3) - 1);
      } else if (scene == Scene::DarkClip) {
        const int kind = (px + 3 * py) % 5;
        y = kind == 0 ? 0.0f : kind == 1 ? 0.02f : kind == 2 ? 0.06f : kind == 3 ? 0.35f : 1.0f;
        cb = cr = 0.0f;
      }
      Y[yy * w + x] = y;
      Cb[yy * w + x] = cb;
      Cr[yy * w + x] = cr;
    }
  });
}

struct NoiseModel {
  float sigmaY = 0.01f;    // at mid grey
  float sigmaC = 0.006f;    // Cb, and Cr unless sigmaCr is set
  float sigmaCr = -1.0f;    // Cr when >= 0 (Cb and Cr noise differ on real cameras)
  float shadowGain = 1.5f; // sigma multiplied by (1 + shadowGain * (1 - Y)^2): more noise in the shadows
  float white = 1.0f;      // mix of white, medium and coarse noise (amplitudes; renormalised)
  float medium = 0.0f;
  float coarse = 0.0f;
  bool clipBlack = false;  // values below 0 are clipped to 0 after the noise is added (dead blacks)
  // The toe of a log curve (S-Log3 and the like): below toeEnd the sigma is
  // toeGain times the base; above, it falls by toeFall per knot interval
  // (1/7 of Y) over a width of toeSteep, then by toeFall2 per interval, until
  // it reaches the base. 0 = off. Multiplies shadowGain's.
  float toeGain = 0.0f;
  float toeEnd = 0.16f;
  float toeFall = 3.5f;
  float toeSteep = 1.0f;
  float toeFall2 = 3.5f;
};

// noisy = clean + noise. Returns the frame as BGRA float (w*h*4), plus the
// clean frame the same way.
inline void makeFrame(const Ycc& clean, uint32_t w, uint32_t h, const NoiseModel& m, uint32_t noiseSeed,
                      std::vector<float>* noisyBgra, std::vector<float>* cleanBgra) {
  const size_t n = (size_t)w * h;
  std::vector<float> field[3];
  const float norm = 1.0f / std::sqrt(m.white * m.white + m.medium * m.medium + m.coarse * m.coarse);
  for (int c = 0; c < 3; ++c) {
    whiteField(&field[c], w, h, noiseSeed * 31 + (uint32_t)c * 3 + 1);
    if (m.medium > 0 || m.coarse > 0) {
      std::vector<float> a, b;
      whiteField(&a, w, h, noiseSeed * 31 + (uint32_t)c * 3 + 2);
      blurUnit(&a, w, h, 2);
      whiteField(&b, w, h, noiseSeed * 31 + (uint32_t)c * 3 + 3);
      blurUnit(&b, w, h, 4);
      for (size_t i = 0; i < n; ++i) field[c][i] = norm * (m.white * field[c][i] + m.medium * a[i] + m.coarse * b[i]);
    }
  }
  noisyBgra->resize(n * 4);
  cleanBgra->resize(n * 4);
  for (size_t i = 0; i < n; ++i) {
    const float y = clean.y[i], cb = clean.cb[i], cr = clean.cr[i];
    const float d = 1.0f - std::min(1.0f, std::max(0.0f, y));
    float g = (1.0f + m.shadowGain * d * d) / (1.0f + m.shadowGain * 0.25f);
    if (m.toeGain > 0) {
      const float yc = std::min(1.0f, std::max(0.0f, y));
      const float a = std::min(std::max(yc - m.toeEnd, 0.0f), m.toeSteep), b = std::max(yc - m.toeEnd - m.toeSteep, 0.0f);
      g *= std::max(1.0f, m.toeGain * std::pow(m.toeFall, -a * 7.0f) * std::pow(m.toeFall2, -b * 7.0f));
    }
    float ny = y + m.sigmaY * g * field[0][i];
    const float ncb = cb + m.sigmaC * g * field[1][i];
    const float ncr = cr + (m.sigmaCr >= 0 ? m.sigmaCr : m.sigmaC) * g * field[2][i];
    float ncbv = ncb, ncrv = ncr;
    if (m.clipBlack && ny < 0) ny = 0;
    if (y >= 1.0f) {  // blown-out area: exactly constant, no noise at all
      ny = y;
      ncbv = cb;
      ncrv = cr;
    }
    for (int k = 0; k < 2; ++k) {
      const float yy = k ? y : ny, bb = k ? cb : ncbv, rr = k ? cr : ncrv;
      const float r = yy + 1.5748f * rr, b = yy + 1.8556f * bb;
      const float gch = (yy - 0.2126f * r - 0.0722f * b) * (1.0f / 0.7152f);
      float* o = (k ? cleanBgra->data() : noisyBgra->data()) + i * 4;
      o[0] = b;
      o[1] = gch;
      o[2] = r;
      o[3] = 1.0f;
    }
  }
}

}  // namespace odtest

#endif
