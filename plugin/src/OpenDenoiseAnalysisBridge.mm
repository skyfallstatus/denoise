// SPDX-License-Identifier: MIT
// Opaque Effect Data first, runtime ID fallback. Nothing is serialized and
// no frame buffers, worker threads or host callbacks are retained.
#include "AEConfig.h"
#include "AE_EffectCB.h"
#include "PrSDKAESupport.h"
#include "PrSDKOpaqueEffectDataSuite.h"
#include "OpenDenoisePlugin.h"
#include "OpenDenoiseCapture.h"
#include "OpenDenoiseAnalysisBridge.h"
#include <algorithm>
#include <chrono>
#include <os/log.h>
#include <map>
#include <mutex>

namespace odplug {
namespace {
using AnalysisClock = std::chrono::steady_clock;
using AnalysisTime = AnalysisClock::time_point;
constexpr auto kConflictWindow = std::chrono::seconds(2);
struct State {
  std::mutex mutex;
  bool closed = false, conflictLogged = false;
  std::atomic<bool> uiLogged{false};
  AnalysisOwner owner;
  AnalysisTime startSeen{}, trackSeen{}, conflictUntil{};
  uint32_t capturedFrames = 0;
  ODAnalysis snapshot{};
  ODParams params{};
};
// Keep a bounded record per runtime ID, including its once-only log flags.
// No session, frame buffer or host pointer is held in the bank.
std::mutex bankMutex;
std::map<int32_t, std::shared_ptr<State>> bank;
constexpr size_t kMaxIdentities = 512;
std::shared_ptr<State> StateFor(int32_t id) {
  std::unique_lock<std::mutex> lock(bankMutex, std::try_to_lock);
  if (!lock.owns_lock()) return {};
  auto it = bank.find(id);
  if (it != bank.end()) return it->second;
  if (bank.size() == kMaxIdentities) return {};
  auto s = std::make_shared<State>(); bank.emplace(id,s); return s;
}
struct Opaque { std::shared_ptr<State> state; };
bool UIIdentity(PF_InData* in, int32_t& id) {
  if (!in || in->appl_id != kAppID_Premiere || !in->pica_basicP || !in->effect_ref) return false;
  const void* raw = nullptr;
  if (in->pica_basicP->AcquireSuite(kPFUtilitySuite,kPFUtilitySuiteVersion4,&raw) || !raw) return false;
  auto u = static_cast<const PF_UtilitySuite4*>(raw);
  A_long instance = 0;
  const bool have = u->GetFilterInstanceID && !u->GetFilterInstanceID(in->effect_ref,&instance);
  id = instance; // zero and signed IDs are valid when the host returned success
  in->pica_basicP->ReleaseSuite(kPFUtilitySuite,kPFUtilitySuiteVersion4);
  return have;
}
void ObserveOwner(State& s, const AnalysisOwner& owner, AnalysisTime now, int32_t id) {
  const bool startKnown = owner.startKnown && owner.scale;
  const bool startConflict = startKnown && s.owner.startKnown && now-s.startSeen < kConflictWindow &&
      (__int128)owner.start*s.owner.scale != (__int128)s.owner.start*owner.scale;
  const bool trackConflict = owner.trackKnown && s.owner.trackKnown && now-s.trackSeen < kConflictWindow &&
      owner.track != s.owner.track;
  if (startConflict || trackConflict) {
    s.conflictUntil = std::max(s.conflictUntil,now+kConflictWindow);
    if (!s.conflictLogged) {
      Log("Analysis conflict: id=%d track=%d start=%d; withholding UI for 2s after last conflict",
          int(id), int(trackConflict), int(startConflict));
      s.conflictLogged = true;
    }
  }
  // Missing metadata is not evidence of a collision. It must not erase the
  // recent known track (software has none) or start used by the next publish.
  if (startKnown) {
    s.owner.startKnown = true; s.owner.start = owner.start; s.owner.scale = owner.scale; s.startSeen = now;
  }
  if (owner.trackKnown) {
    s.owner.trackKnown = true; s.owner.track = owner.track; s.trackSeen = now;
  }
}
} // namespace
class AnalysisLease {
 public:
  SPBasicSuite* basic = nullptr;
  const PrSDKOpaqueEffectDataSuite* api = nullptr;
  int32_t id = 0;
  Opaque* opaque = nullptr;
  std::shared_ptr<State> fallback;
  ~AnalysisLease() {
    if (opaque) {
      OpaqueEffectDataType* dispose = nullptr;
      if (!api->ReleaseOpaqueEffectData(id,&dispose) && dispose) delete reinterpret_cast<Opaque*>(dispose);
    }
    if (api) basic->ReleaseSuite(kPrSDKOpaqueEffectDataSuite,kPrSDKOpaqueEffectDataSuiteVersion);
  }
  std::shared_ptr<State> state() const { return opaque ? opaque->state : fallback; }
};
std::shared_ptr<AnalysisLease> AcquireAnalysis(SPBasicSuite* basic, int32_t id) {
  auto s = StateFor(id);
  if (!s) return {};
  auto lease = std::make_shared<AnalysisLease>();
  lease->basic = basic; lease->id = id; lease->fallback = s;
  const void* raw = nullptr;
  if (!basic || basic->AcquireSuite(kPrSDKOpaqueEffectDataSuite,kPrSDKOpaqueEffectDataSuiteVersion,&raw) || !raw)
    return lease;
  lease->api = static_cast<const PrSDKOpaqueEffectDataSuite*>(raw);
  if (!lease->api->AcquireOpaqueEffectData || !lease->api->RegisterOpaqueEffectData || !lease->api->ReleaseOpaqueEffectData)
    return lease;
  OpaqueEffectDataType* p = nullptr;
  if (lease->api->AcquireOpaqueEffectData(id,&p)) return lease;
  if (!p) {
    auto made = std::make_unique<Opaque>(); made->state = s;
    p = reinterpret_cast<OpaqueEffectDataType*>(made.get());
    if (lease->api->RegisterOpaqueEffectData(id,&p) || !p) return lease;
    if (p == reinterpret_cast<OpaqueEffectDataType*>(made.get())) made.release();
  }
  lease->opaque = reinterpret_cast<Opaque*>(p);
  return lease;
}
namespace {
// Explicit time keeps expiry tests deterministic without sleeping or changing
// the process clock. Production calls always use the monotonic clock.
void PublishAnalysisAt(const std::shared_ptr<AnalysisLease>& lease, const AnalysisOwner& owner,
                       uintptr_t source, bool gpu, Session* session, uint32_t frames, AnalysisTime now) {
  if (!lease) return;
  auto s = lease->state();
  std::unique_lock<std::mutex> lock(s->mutex,std::try_to_lock);
  if (!lock.owns_lock() || s->closed) return;
  ObserveOwner(*s,owner,now,lease->id);
  // Keep the latest candidate even during a conflict: after the conflict
  // expires, a UI refresh can show it without forcing another render.
  // The software source check remains essential if another CPU callback has
  // overwritten the shared Session between Render and TryAnalysis.
  ODAnalysis a{}; ODParams p{};
  if (session && (gpu || source)) session->TryAnalysis(a,p,gpu ? 0 : source);
  s->snapshot = a; s->params = p; s->capturedFrames = frames;
}
bool ReadAnalysisAt(PF_InData* in, ODAnalysis& a, ODParams& p,
                    std::shared_ptr<AnalysisLease>* readLease, AnalysisTime now) {
  a = {};
  if (readLease) readLease->reset();
  int32_t id;
  if (!UIIdentity(in,id)) return false;
  auto lease = AcquireAnalysis(in->pica_basicP,id);
  if (!lease) return false;
  auto s = lease->state();
  std::unique_lock<std::mutex> lock(s->mutex,std::try_to_lock);
  if (!lock.owns_lock() || s->closed || now < s->conflictUntil || !s->snapshot.valid) return false;
  a = s->snapshot; p = s->params;
  if (a.state == OD_ANALYSIS_CAPTURED) a.capturedFrames = s->capturedFrames;
  if (readLease) *readLease = std::move(lease);
  return true;
}
} // namespace
void PublishAnalysis(const std::shared_ptr<AnalysisLease>& lease, const AnalysisOwner& owner,
                     uintptr_t source, bool gpu, Session* session, uint32_t frames) {
  PublishAnalysisAt(lease,owner,source,gpu,session,frames,AnalysisClock::now());
}
void AnalysisLifecycle(PF_Cmd cmd, PF_InData* in) {
  if (cmd == PF_Cmd_GLOBAL_SETDOWN) {
    std::lock_guard<std::mutex> lock(bankMutex);
    for (auto& e:bank) {
      std::lock_guard<std::mutex> slot(e.second->mutex);
      e.second->closed = true; e.second->snapshot = {};
    }
    // Existing leases stay closed; a later host session starts empty.
    bank.clear();
    return;
  }
  if (cmd != PF_Cmd_SEQUENCE_SETDOWN && cmd != PF_Cmd_SEQUENCE_RESETUP) return;
  int32_t id;
  if (!UIIdentity(in,id)) return;
  auto lease = AcquireAnalysis(in->pica_basicP,id);
  if (!lease) return;
  auto s = lease->state();
  std::unique_lock<std::mutex> lock(s->mutex,std::try_to_lock);
  if (lock.owns_lock()) s->snapshot = {};
  // A sequence callback invalidates old data, not the runtime ID. Render/UI
  // references can change for the same instance; the next publish recovers.
}
bool ReadAnalysisForUI(PF_InData* in, ODAnalysis& a, ODParams& p,
                       std::shared_ptr<AnalysisLease>* readLease) {
  return ReadAnalysisAt(in,a,p,readLease,AnalysisClock::now());
}
void NoteUIAnalysis(const std::shared_ptr<AnalysisLease>& lease) {
  if (!lease || lease->state()->uiLogged.exchange(true)) return;
  const char* source = lease->opaque ? "opaque" : "bank";
  static os_log_t log = os_log_create("dev.opendenoise.plugin", "plugin");
  os_log_info(log,"UI analysis: id=%d source=%{public}s",int(lease->id),source);
  if (getenv("OD_LOG_STDERR"))
    fprintf(stderr,"[Denoise] UI analysis: id=%d source=%s\n",int(lease->id),source);
}
bool ReadSavedAnalysis(PF_InData* in, PF_ParamDef* params[], ODAnalysis& a) {
  if (!params || !params[PfIndex(OD_P_PROFILE_POLICY)] ||
      params[PfIndex(OD_P_PROFILE_POLICY)]->u.pd.value != OD_PROFILE_CAPTURED) return false;
  CaptureBytes bytes; Capture c;
  if (ReadCaptureHandle(in,params[PfIndex(OD_P_PROFILE_PAYLOAD)]->u.arb_d.value,bytes) ||
      !DecodeCapture(bytes,c) || !c.count || c.revision != uint32_t(params[PfIndex(OD_P_PROFILE_REVISION)]->u.sd.value)) return false;
  const auto digest = DigestCapture(bytes);
  for (int i=0;i<8;++i) if (digest[i] != uint32_t(params[PfIndex(OD_P_PROFILE_DIGEST0+i)]->u.sd.value)) return false;
  a = {}; a.valid = a.profileValid = 1; a.state = OD_ANALYSIS_CAPTURED;
  a.capturedFrames = c.count; a.width = c.width; a.height = c.height; a.confidence = c.profile.confidence;
  for (int ch=0;ch<3;++ch) for (int l=0;l<OD_LEVELS;++l) {
    a.full[ch][l] = 876*c.profile.sigmaFull[ch][l];
    for (int b=0;b<OD_BANDS;++b) a.profile[ch][b][l] = 876*c.profile.sigma[ch][b][l];
  }
  // No histogram, measured-knot provenance or removed-dB observation invented.
  return true;
}
} // namespace odplug
