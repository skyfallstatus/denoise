// SPDX-License-Identifier: MIT
// Missing luminance support must not masquerade as a lower noise measurement
// in the finest-grain mode's profile (ODParams.grainMode = OD_GRAIN_FINEST),
// while the whole-picture mode's profile (OD_GRAIN_WHOLE, the aggregate of
// API v9) stays what it was. The stage keeps one aggregate per mode, fed the
// same estimates (stage.h, ProfileStage::smooth).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "context.h"
#include "stages/stage.h"

static ODNoiseProfile sample(float dark, float mid) {
  ODNoiseProfile p = {};
  p.valid = 1; p.confidence = 1;
  for (int c = 0; c < 3; ++c)
    for (int l = 0; l < OD_LEVELS; ++l) {
      p.sigmaFull[c][l] = (l < 2 ? dark : mid) * (c + 1);
      for (int b = 0; b < OD_BANDS; ++b)
        p.sigma[c][b][l] = p.sigmaFull[c][l] / (b + 1);
    }
  return p;
}
int main() {
  @autoreleasepool {
    auto dev = MTLCreateSystemDefaultDevice();
    od::Context ctx(dev, [dev newCommandQueue]);
    std::string err;
    auto stage = od::makeProfileStage(ctx, &err);
    if (!stage) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    int failures = 0, checks = 0;
    auto check = [&](bool ok, const char* what) {
      ++checks; if (!ok) { ++failures; fprintf(stderr, "FAIL %s\n", what); }
    };
    ODNoiseProfile whole = {}, fine = {};
    ODNoiseProfile* const both[od::kGrainModes] = {&whole, &fine};
    auto all = sample(.009f, .003f);
    for (int i = 0; i < 24; ++i) stage->smooth(all, 0xff, both, false);
    // A foreground occluder hides shadows for longer than the entire ring.
    // The fit fills those knots with midtone values, but measured only 2..7.
    auto covered = sample(.003f, .003f);
    for (int i = 0; i < 48; ++i) {
      stage->smooth(covered, 0xfc, both, false);
      for (int c = 0; c < 3; ++c) for (int l = 0; l < 2; ++l) {
        check(std::fabs(fine.sigmaFull[c][l] - all.sigmaFull[c][l]) < 1e-7f, "unmeasured full-band shadow changed");
        for (int b = 0; b < OD_BANDS; ++b)
          check(std::fabs(fine.sigma[c][b][l] - all.sigma[c][b][l]) < 1e-7f, "unmeasured band shadow changed");
      }
    }
    // The whole-picture mode keeps the aggregate of API v9: once the ring
    // holds only covered frames, its shadows are their interpolated values.
    for (int c = 0; c < 3; ++c)
      for (int l = 0; l < 2; ++l)
        check(std::fabs(whole.sigmaFull[c][l] - covered.sigmaFull[c][l]) < 1e-6f,
              "whole-picture mode no longer follows its ring at unmeasured knots");
    // A held mode (nullptr) is left alone while the other one moves on.
    {
      const ODNoiseProfile before = fine;
      ODNoiseProfile* const wholeOnly[od::kGrainModes] = {&whole, nullptr};
      auto brighter = sample(.003f, .004f);
      // A whole ring of them (a few would be trimmed by the robust mean).
      for (int i = 0; i < 24; ++i) stage->smooth(brighter, 0xfc, wholeOnly, false);
      check(memcmp(&before, &fine, sizeof fine) == 0, "a held mode's slot changed");
      check(std::fabs(whole.sigmaFull[0][4] - .004f) < 1e-6f, "the other mode did not move on");
    }
    // Explicit reset and decisive new footage must discard held shadows.
    stage->reset();
    stage->smooth(covered, 0xfc, both, false);
    check(std::fabs(fine.sigmaFull[0][0] - .003f) < 1e-7f, "reset retained previous shot");
    check(std::fabs(whole.sigmaFull[0][0] - .003f) < 1e-7f, "reset retained previous shot (whole picture)");
    stage->smooth(all, 0xff, both, true);
    check(std::fabs(fine.sigmaFull[0][0] - .009f) < 1e-7f, "remeasured shadow did not acquire");
    auto next = sample(.001f, .001f);
    stage->smooth(next, 0xfc, both, true);
    check(std::fabs(fine.sigmaFull[0][0] - .001f) < 1e-7f, "cut retained previous shot's unsupported knot");
    printf("knots: %d checks, %d failures\n", checks, failures);
    return failures != 0;
  }
}
