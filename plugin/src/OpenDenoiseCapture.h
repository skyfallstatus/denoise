// SPDX-License-Identifier: MIT
// Project-owned, pointer-free profile; process-local immutable GPU transport.
#pragma once
#include <array>
#include <memory>
#include <string>
#include "opendenoise/profiling.h"

namespace odplug {
constexpr size_t kCaptureBytes = 832;
using CaptureBytes = std::array<uint8_t, kCaptureBytes>;
using CaptureDigest = std::array<uint32_t, 8>;
struct Capture {
  ODNoiseProfile profile{};
  uint32_t width = 0, height = 0, timeScale = 0, timeStep = 0;
  uint32_t count = 0, revision = 0, parNum = 1, parDen = 1;
  std::array<int64_t, 32> times{}; // checked-out layer times, NOT source frame IDs
};
struct CaptureSnapshot {
  Capture capture;
  CaptureBytes bytes;
  CaptureDigest digest;
};
bool EncodeCapture(const Capture&, CaptureBytes&);
bool DecodeCapture(const CaptureBytes&, Capture&);
CaptureDigest DigestCapture(const CaptureBytes&);
std::string CaptureStatus(const Capture&);
std::string CaptureDigestText(const CaptureDigest&);
uint64_t CaptureBridgeID(); // process/module diagnostic identity, not a saved key
void LogCaptureBridge(const char* event, bool keys = false);
// Live arbitrary handles and GPU instances pin immutable snapshots. A bounded
// recent-publication bank also spans temporary host-handle disposal between
// Save/UNFLATTEN and CreateInstance. No runtime ID or GPU allocation is a key.
std::shared_ptr<const CaptureSnapshot> FindCapture(const CaptureDigest&, uint32_t revision);

// Pending requests are bounded, process-local and never a persistence source.
// A random two-word parameter token routes UI -> CPU/GPU render clones without
// relying on the runtime ID (which Premiere may reuse for pasted effects).
class Session;
struct CaptureProgress {
  bool active = false, ready = false, failed = false;
  Capture capture;
  std::string action, detail;
};
uint64_t ArmCapture(const CaptureBytes& base, uint32_t revision,
                    uint32_t width, uint32_t height, uint32_t parNum, uint32_t parDen);
CaptureProgress PendingCapture(uint64_t token);
// DRAW must not wait for a render measuring a frame. False means a bank/request
// lock is busy; true returns a copy only when the current saved base matches.
bool TryCaptureForUI(uint64_t token, const CaptureBytes& base, CaptureProgress&);
void CancelCapture(uint64_t token);
bool AcknowledgeCaptureFailure(uint64_t token);
void FailCapture(uint64_t token, const char* reason);
bool CapturePending(uint64_t token);
// Called before filtering: no pixel or host buffer is retained. owner identifies
// the render stream; a second stream cannot contribute to the same request.
void ObserveCapture(uint64_t token, uint64_t owner, Session&, void* queue,
                    const ODFrame*, const int64_t* times, int count,
                    uint32_t timeScale, uint32_t timeStep, bool fullProgressive, const ODParams* params = nullptr);
bool CaptureBaseMatches(uint64_t token, const CaptureBytes&);

// SDK types must be included before this header at the host entry points.
PF_Err CaptureArbitrary(PF_InData*, PF_ArbParamsExtra*);
PF_Err ReadCaptureHandle(PF_InData*, PF_Handle, CaptureBytes&);
PF_Err NewCaptureHandle(PF_InData*, const CaptureBytes&, PF_Handle*, const char* origin = "new");
// UI selector only. Writes the already-owned fixed-size handle in place;
// never disposes a borrowed parameter value or edits a render's snapshot.
PF_Err WriteCaptureHandle(PF_InData*, PF_Handle, const CaptureBytes&, const char* origin = "ui-save");
} // namespace odplug
