// SPDX-License-Identifier: MIT
#pragma once
#include "AE_Effect.h"
#include "SPBasic.h"
#include "opendenoise/opendenoise.h"
#include <memory>
#include <cstdint>
namespace odplug {
class Session;
class AnalysisLease;
struct AnalysisOwner {
  bool startKnown = false;
  int64_t start = 0; // rational seconds: start / scale
  uint64_t scale = 0;
  bool trackKnown = false;
  int32_t track = 0;
};
std::shared_ptr<AnalysisLease> AcquireAnalysis(SPBasicSuite*, int32_t runtime);
// Runtime ID is the bridge key. Only recent, known track/start disagreements
// conflict; nodes, timelines and transient UI references are not identities.
// source is only a software render-record guard, never a bridge key.
void PublishAnalysis(const std::shared_ptr<AnalysisLease>&, const AnalysisOwner&,
                     uintptr_t source, bool gpu, Session*, uint32_t capturedFrames);
void AnalysisLifecycle(PF_Cmd, PF_InData*);
bool ReadAnalysisForUI(PF_InData*, ODAnalysis&, ODParams&,
                       std::shared_ptr<AnalysisLease>* readLease = nullptr);
// Call only after a status row or graph actually displayed the read snapshot.
void NoteUIAnalysis(const std::shared_ptr<AnalysisLease>&);
// UI-owned serialized capture fallback; never supplies live removal numbers.
bool ReadSavedAnalysis(PF_InData*, PF_ParamDef*[], ODAnalysis&);
} // namespace odplug
