// SPDX-License-Identifier: MIT
// Pure decision logic: timestamps and completed latencies are injected for tests.
// Protected by Session::Shared::mutex; no GPU work, allocation, or host waits.
#ifndef OD_QUALITY_GOVERNOR_H
#define OD_QUALITY_GOVERNOR_H
#include <algorithm>
#include <cmath>
#include <cstdint>
#include "opendenoise/quality.h"
namespace od {
class QualityGovernor {
 public:
  void reset() { ++epoch_; active_ = false; configured_ = false; }
  static double budget(double interval) { return std::min(20.0, 0.60 * interval); }
  ODQualityDecision begin(const ODQualityRequest& r, bool realtime, int explicitQuality,
                          uint64_t configuration, double now) {
    const bool autoPlay = r.automatic && realtime;
    const bool newConfig = !configured_ || configuration != configuration_;
    const double delta = r.timelineTimeMs - lastTime_;
    const bool hostRate = validInterval(r.frameDurationMs);
    if (newConfig) {
      configuration_ = configuration; configured_ = true;
      interval_ = hostRate ? r.frameDurationMs : 40;
      pendingInterval_ = 0; pendingCount_ = 0;
      clearEstimates(); quality_ = 1; active_ = false;
    }
    bool rateChanged = false;
    if (hostRate) {
      rateChanged = std::abs(interval_ - r.frameDurationMs) > interval_ * .05;
      interval_ = r.frameDurationMs;
    } else if (active_ && validInterval(delta)) {
      // Two consistent sequence steps reject isolated seeks/skipped requests.
      // If sequence timestamps are absent, use consistent call-arrival steps.
      inferInterval(delta, rateChanged);
    } else if (active_ && !std::isfinite(r.timelineTimeMs)) {
      inferInterval(now - lastCall_, rateChanged);
    }
    const bool discontinuity = active_ &&
        (now - lastCall_ > std::max(250.0, 4 * interval_) ||
         (std::isfinite(delta) && (delta <= 0 || delta > 4 * interval_)));
    const bool newRun = autoPlay && (!active_ || discontinuity || rateChanged);
    if (newRun) {
      ++epoch_; switches_ = 0; over_ = 0; completed_ = 0;
      // A cheaper, sustained run is evidence to retry Full, only at a boundary.
      // Never upgrade on a frame of continuous playback. Retain the learned
      // fallback across short stop/start and scrub requests.
      if (quality_ == 0 && ((low_ >= 24 && now - downgradedAt_ >= 2000) ||
                            (rateChanged && ema_[1] > 0 && ema_[1] < .85 * budget(interval_))))
        quality_ = 1;
      low_ = 0;
    }
    if (!autoPlay && active_) ++epoch_; // old asynchronous work cannot train a new run
    active_ = autoPlay; lastCall_ = now; lastTime_ = r.timelineTimeMs;
    const int selected = !r.automatic ? (explicitQuality ? 1 : 0) : autoPlay ? quality_ : 1;
    ODQualityDecision d = {++serial_, epoch_, selected, autoPlay ? switches_ : 0,
                           autoPlay ? budget(interval_) : 0, ema_[selected]};
    return d;
  }
  void complete(const ODQualityCompletion& c, double now) {
    if (!active_ || c.runId != epoch_ || c.quality != quality_ || c.failed ||
        !std::isfinite(c.callToFinishMs) || c.callToFinishMs <= 0) return;
    // Initial pipeline allocation/profile cold start is not a steady-state cost.
    // It remains in completion telemetry and deadline counts.
    if (++completed_ <= 4) return;
    double& e = ema_[quality_];
    e = e > 0 ? e + .20 * (c.callToFinishMs - e) : c.callToFinishMs;
    const double b = budget(interval_);
    if (quality_ == 1) {
      over_ = e > b && c.callToFinishMs > b ? over_ + 1 : 0;
      if (completed_ >= 12 && over_ >= 6) {
        quality_ = 0; ++switches_; over_ = 0; completed_ = 0;
        ema_[0] = 0; low_ = 0; downgradedAt_ = now;
      }
    } else {
      low_ = e < .60 * b && c.callToFinishMs < .70 * b ? low_ + 1 : 0;
    }
  }
 private:
  static bool validInterval(double v) { return std::isfinite(v) && v >= 4 && v <= 100; }
  void clearEstimates() { ema_[0] = ema_[1] = 0; low_ = over_ = completed_ = 0; }
  void inferInterval(double step, bool& changed) {
    if (!validInterval(step)) { pendingCount_ = 0; return; }
    if (std::abs(step - pendingInterval_) < step * .03) ++pendingCount_;
    else { pendingInterval_ = step; pendingCount_ = 1; }
    if (pendingCount_ >= 2) {
      changed = std::abs(interval_ - step) > interval_ * .05;
      interval_ = step;
    }
  }
  uint64_t serial_ = 0, epoch_ = 0, configuration_ = 0;
  bool active_ = false, configured_ = false;
  int quality_ = 1, switches_ = 0, completed_ = 0, over_ = 0, low_ = 0, pendingCount_ = 0;
  double interval_ = 40, pendingInterval_ = 0;
  double lastCall_ = 0, lastTime_ = NAN, downgradedAt_ = 0, ema_[2] = {};
};
} // namespace od
#endif
