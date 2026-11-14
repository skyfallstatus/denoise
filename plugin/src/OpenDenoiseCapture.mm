// SPDX-License-Identifier: MIT
// Adobe owns arbitrary handles. We own only immutable, content-addressed copies.
#import <Metal/Metal.h>
#include "AEConfig.h"
#include "PrSDKTypes.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "OpenDenoisePlugin.h"
#include "OpenDenoiseCapture.h"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <cstdio>
#include <set>
#include <cstdlib>
#include <time.h>
#include <unistd.h>

namespace odplug {
namespace {
uint32_t Get32(const uint8_t* p) {
  return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
void Put32(uint8_t* p, uint32_t x) { for (int i = 0; i < 4; ++i) p[i] = uint8_t(x >> (8*i)); }
uint32_t CRC(const uint8_t* p, size_t n) {
  uint32_t c = ~0u;
  for (size_t i = 0; i < n; ++i) {
    c ^= p[i];
    for (int j = 0; j < 8; ++j) c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1)));
  }
  return ~c;
}
// No Adobe calls under this lock (arbitrary callbacks can be re-entrant).
std::mutex gCaptureMutex;
std::map<PF_Handle, std::shared_ptr<const CaptureSnapshot>> gHandlePins;
// A host can keep flat bytes, dispose every temporary arbitrary handle, and
// only then construct the GPU instance. Handle lifetime is not value lifetime.
// Retain the most recent 1024 distinct publications for this module/process;
// at most 3 MiB of snapshot data, plus fixed shared_ptr slots. Live handles and
// existing GPU instances remain pinned independently of this bounded bank.
constexpr size_t kRecentCaptures = 1024;
static_assert(sizeof(CaptureSnapshot) <= 3072, "bounded saved-profile transport");
std::array<std::shared_ptr<const CaptureSnapshot>, kRecentCaptures> gRecentCaptures;
size_t gRecentNext = 0;
std::shared_ptr<const CaptureSnapshot> Snapshot(const CaptureBytes& bytes) {
  Capture c;
  if (!DecodeCapture(bytes, c) || !c.count) return {};
  return std::make_shared<const CaptureSnapshot>(CaptureSnapshot{c, bytes, DigestCapture(bytes)});
}
bool Pin(PF_Handle h, std::shared_ptr<const CaptureSnapshot> s) {
  std::lock_guard<std::mutex> lock(gCaptureMutex);
  if (!s) { gHandlePins.erase(h); return true; }
  auto collision = [&](const auto& other) {
    return other && other->digest == s->digest && other->bytes != s->bytes;
  };
  for (const auto& entry : gHandlePins) if (collision(entry.second)) return false;
  size_t same = gRecentCaptures.size();
  for (size_t i = 0; i < gRecentCaptures.size(); ++i) {
    if (collision(gRecentCaptures[i])) return false;
    if (gRecentCaptures[i] && gRecentCaptures[i]->bytes == s->bytes) same = i;
  }
  // The only potentially allocating operation precedes all publication.
  gHandlePins[h] = s;
  if (same == gRecentCaptures.size()) {
    gRecentCaptures[gRecentNext] = std::move(s);
    gRecentNext = (gRecentNext + 1) % gRecentCaptures.size();
  }
  return true;
}
bool HaveHandles(PF_InData* in) {
  return in && in->utils && in->utils->host_new_handle && in->utils->host_dispose_handle &&
      in->utils->host_get_handle_size && in->utils->host_lock_handle && in->utils->host_unlock_handle;
}
constexpr PF_Err kBad = PF_Err_INTERNAL_STRUCT_DAMAGED;
}

// No render edits a host parameter. The UI alone commits a completed request.
namespace {
using PixelDigest = std::array<uint8_t, CC_SHA256_DIGEST_LENGTH>;
struct Pending {
  std::mutex mutex;
  uint64_t renderOwner = 0, created = 0, workNs = 0;
  CaptureBytes base{};
  CaptureProgress progress;
  int64_t anchor = 0;
  bool anchored = false, failureSeen = false;
  uint32_t chromaRecovery = 0;
  ODPixelFormat format = OD_PIXEL_BGRA32F;
  std::map<int64_t, ODNoiseProfile> samples;
  std::set<int64_t> attempted;
  std::set<PixelDigest> digests;
};
std::mutex gPendingMutex;
std::map<uint64_t, std::shared_ptr<Pending>> gPending;
uint64_t CaptureNow() { return clock_gettime_nsec_np(CLOCK_MONOTONIC); }
std::shared_ptr<Pending> Request(uint64_t token) {
  if (!token) return {};
  std::lock_guard<std::mutex> lock(gPendingMutex);
  auto it = gPending.find(token);
  return it == gPending.end() ? nullptr : it->second;
}
void Failed(Pending& p, const char* why) {
  p.progress.failed = true;
  p.progress.ready = false;
  p.progress.action = "Capture failed - retry";
  p.progress.detail = why;
  Log("Capture: failed: %s; previous saved profile kept", why);
}
bool Live(Pending& p) {
  if (p.progress.failed || p.progress.ready) return false;
  if (CaptureNow() - p.created > 120000000000ull) Failed(p, "Timed out; keep playhead nearby and retry");
  if (p.workNs > 30000000000ull) Failed(p, "Measurement exceeded 30 seconds");
  return !p.progress.failed;
}
}

uint64_t ArmCapture(const CaptureBytes& base, uint32_t revision,
                    uint32_t width, uint32_t height, uint32_t parNum, uint32_t parDen) {
  auto p = std::make_shared<Pending>();
  p->base = base; p->created = CaptureNow();
  p->progress.active = true;
  p->progress.action = "Armed - step/play nearby";
  p->progress.detail = "Await render; click again to save";
  auto& c = p->progress.capture;
  c.width = width; c.height = height; c.revision = revision;
  c.parNum = parNum; c.parDen = parDen;
  std::lock_guard<std::mutex> lock(gPendingMutex);
  // Lazy expiry, no timer/task; requests own only small CPU statistics.
  for (auto it = gPending.begin(); it != gPending.end();) {
    if (CaptureNow() - it->second->created > 600000000000ull)
      it = gPending.erase(it);
    else ++it;
  }
  if (gPending.size() >= 16) return 0;
  uint64_t token = 0;
  do { arc4random_buf(&token, sizeof token); } while (!token || gPending.count(token));
  gPending.emplace(token, p);
  Log("Capture: armed for %ux%u; no USER_CHANGED_PARAM checkout; next full progressive render, then click Capture to save", width, height);
  return token;
}
CaptureProgress PendingCapture(uint64_t token) {
  auto p = Request(token);
  if (!p) return {};
  std::lock_guard<std::mutex> lock(p->mutex);
  Live(*p);
  return p->progress;
}
bool TryCaptureForUI(uint64_t token, const CaptureBytes& base, CaptureProgress& result) {
  result = {};
  if (!token) return true;
  std::shared_ptr<Pending> p;
  {
    std::unique_lock<std::mutex> lock(gPendingMutex, std::try_to_lock);
    if (!lock.owns_lock()) return false;
    const auto it = gPending.find(token);
    if (it == gPending.end()) return true;
    p = it->second;
  }
  std::unique_lock<std::mutex> lock(p->mutex, std::try_to_lock);
  if (!lock.owns_lock()) return false;
  if (p->base == base) {
    Live(*p);
    result = p->progress;
  }
  return true;
}
bool AcknowledgeCaptureFailure(uint64_t token) {
  auto p = Request(token);
  if (!p) return false;
  std::lock_guard<std::mutex> lock(p->mutex);
  const bool seen = p->failureSeen;
  p->failureSeen = true;
  return seen;
}
void CancelCapture(uint64_t token) {
  std::lock_guard<std::mutex> lock(gPendingMutex);
  gPending.erase(token);
}
void FailCapture(uint64_t token, const char* reason) {
  auto p = Request(token);
  if (!p) return;
  std::lock_guard<std::mutex> lock(p->mutex);
  if (Live(*p)) Failed(*p, reason);
}
bool CapturePending(uint64_t token) {
  auto p = Request(token);
  if (!p) return false;
  std::lock_guard<std::mutex> lock(p->mutex);
  return Live(*p);
}
bool CaptureBaseMatches(uint64_t token, const CaptureBytes& base) {
  auto p = Request(token);
  if (!p) return false;
  std::lock_guard<std::mutex> lock(p->mutex);
  return p->base == base;
}

void ObserveCapture(uint64_t token, uint64_t owner, Session& session, void* queue,
                    const ODFrame* frames, const int64_t* times, int count,
                    uint32_t scale, uint32_t step, bool fullProgressive, const ODParams* params) {
  auto p = Request(token);
  if (!p) return;
  std::lock_guard<std::mutex> lock(p->mutex);
  if (!Live(*p)) return;
  auto& c = p->progress.capture;
  if (!fullProgressive || !count || frames[0].width != c.width || frames[0].height != c.height) {
    p->progress.detail = "Await Full progressive render";
    Log("Capture: skip preview; need %ux%u full progressive", c.width, c.height);
    return;
  }
  if (!scale || !step) { Failed(*p, "Invalid render time base"); return; }
  if (!count || !owner) { Failed(*p, "Cannot identify render frames"); return; }
  const uint32_t recovery = params && params->advanced && params->chromaRecovery == 1 ? 1u : 0u;
  if (p->anchored && p->chromaRecovery != recovery) {
    Failed(*p, "Processing changed; capture again"); return;
  }
  if (p->anchored && (p->renderOwner != owner || c.timeScale != scale || c.timeStep != step)) {
    Failed(*p, "Render source/timing changed; retry"); return;
  }
  if (!p->anchored) {
    p->anchor = times[0]; p->renderOwner = owner; p->anchored = true; p->chromaRecovery = recovery;
    c.timeScale = scale; c.timeStep = step; p->format = frames[0].format;
    Log("Capture: window anchor=%lld scale=%u step=%u offsets=-4..4", (long long)p->anchor, scale, step);
  }
  // Reuse one allocation even if autoreleased command buffers retain their
  // resources until the render pool drains. Every blit completes before reuse.
  id<MTLBuffer> staging = nil;
  for (int i = 0; i < count && Live(*p); ++i) {
    const int64_t time = times[i];
    const long double delta = (long double)time - p->anchor;
    if (delta < -4.0L*step || delta > 4.0L*step || time < 0) continue;
    if ((time-p->anchor) % step || !p->attempted.insert(time).second) continue;
    const uint64_t start = CaptureNow();
    const ODFrame& f = frames[i];
    const uint32_t bpp = f.format == OD_PIXEL_BGRA32F ? 16 : 8;
    Log("Capture sample: render time=%lld/%u world=%ux%u rowbytes=%u format=%s checkout=n/a",
        (long long)time, scale, f.width, f.height, f.rowBytes, bpp == 16 ? "BGRA32F" : "BGRA16F");
    if (f.width != c.width || f.height != c.height || uint64_t(f.width)*f.height > 16777216 ||
        f.rowBytes < uint64_t(f.width)*bpp || f.format != p->format || !f.mtlBuffer) {
      Failed(*p, "Render size/format changed; use Full"); return;
    }
    id<MTLBuffer> source = (__bridge id<MTLBuffer>)f.mtlBuffer;
    if (source.length < uint64_t(f.height-1)*f.rowBytes + uint64_t(f.width)*bpp) {
      Failed(*p, "Capture buffer is too short"); return;
    }
    id<MTLBuffer> readable = source;
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    // Hash visible pixels only, never padding. Wait for prior host writes on
    // its queue; private buffers need a bounded packed staging copy.
    uint32_t hashStride = f.rowBytes;
    if (q) {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      if (!cb) { Failed(*p, "Cannot synchronize capture input"); return; }
      if (source.storageMode == MTLStorageModePrivate) {
        if (!staging) staging = [source.device newBufferWithLength:size_t(f.width)*f.height*bpp options:MTLResourceStorageModeShared];
        readable = staging;
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        if (!readable || !blit) { Failed(*p, "Cannot read capture input"); return; }
        hashStride = f.width*bpp;
        for (uint32_t y = 0; y < f.height; ++y)
          [blit copyFromBuffer:source sourceOffset:size_t(y)*f.rowBytes toBuffer:readable
             destinationOffset:size_t(y)*hashStride size:hashStride];
        [blit endEncoding];
      }
      [cb commit]; [cb waitUntilCompleted];
      if (cb.status == MTLCommandBufferStatusError) { Failed(*p, "Capture input synchronization failed"); return; }
    }
    if (!readable.contents) { Failed(*p, "Capture input has no readable pixels"); return; }
    CC_SHA256_CTX hash; CC_SHA256_Init(&hash);
    for (uint32_t y = 0; y < f.height; ++y)
      CC_SHA256_Update(&hash, (const char*)readable.contents + size_t(y)*hashStride, f.width*bpp);
    PixelDigest digest; CC_SHA256_Final(digest.data(), &hash);
    const bool duplicate = !p->digests.insert(digest).second;
    Log("Capture sample: time=%lld digest-duplicate=%d%s", (long long)time, duplicate, duplicate ? " skip" : "");
    if (!duplicate) {
      ODNoiseProfile fit{}; char error[256]{};
      ODFrame measured = f; measured.frameKey = OD_NO_KEY;
      const int rc = session.Profile(measured, fit, error, sizeof error, params);
      Log("Capture sample: time=%lld fit_rc=%d valid=%u confidence=%.6f accepted=%d reason=%s",
          (long long)time, rc, fit.valid, fit.confidence, !rc && fit.valid && fit.confidence >= .3f, error);
      if (!rc && fit.valid && fit.confidence >= .3f) p->samples.emplace(time, fit);
      else if (rc && !strstr(error, "no usable noise estimate")) { Failed(*p, "Profile measurement failed (see log)"); return; }
    }
    p->workNs += CaptureNow()-start;
  }
  if (!Live(*p)) return;
  c.count = (uint32_t)p->samples.size();
  if (c.count >= 3) {
    ODNoiseProfile measured[9]{}; unsigned i = 0;
    for (const auto& sample : p->samples) { c.times[i] = sample.first; measured[i++] = sample.second; }
    CaptureBytes bytes;
    if (od_profile_aggregate(measured, c.count, &c.profile) || !EncodeCapture(c, bytes)) {
      Failed(*p, "Capture aggregation failed"); return;
    }
    p->progress.ready = true;
    p->progress.action = "Ready - click to save";
    char detail[64]; snprintf(detail, sizeof detail, "%u frames; fit %.0f%%", c.count, 100*c.profile.confidence);
    p->progress.detail = detail;
    Log("Capture: ready (NOT saved): %s", CaptureStatus(c).c_str());
  } else if (p->attempted.size() >= 9) {
    Failed(*p, "Need 3 distinct usable frames; retry");
  } else {
    char detail[64]; snprintf(detail, sizeof detail, "%u/3 frames; step nearby", c.count);
    p->progress.detail = detail;
  }
}

bool DecodeCapture(const CaptureBytes& b, Capture& out) {
  out = Capture{};
  if (std::all_of(b.begin(), b.end(), [](uint8_t c) { return c == 0; })) return true;
  if (memcmp(b.data(), "ODCP", 4) || Get32(&b[4]) != 1 || Get32(&b[8]) != b.size() ||
      Get32(&b[12]) != 1 || Get32(&b[828]) != CRC(b.data(), 828)) return false;
  Capture c;
  c.width = Get32(&b[16]); c.height = Get32(&b[20]);
  c.timeScale = Get32(&b[24]); c.timeStep = Get32(&b[28]);
  c.count = Get32(&b[32]); c.revision = Get32(&b[36]);
  c.parNum = Get32(&b[296]); c.parDen = Get32(&b[300]);
  if (!c.width || !c.height || uint64_t(c.width)*c.height > 16777216 ||
      !c.timeScale || !c.timeStep || c.count < 3 || c.count > 32 ||
      !c.revision || c.revision > INT32_MAX || !c.parNum || !c.parDen) return false;
  for (unsigned i = 0; i < 32; ++i) {
    const uint64_t bits = uint64_t(Get32(&b[40+i*8])) | uint64_t(Get32(&b[44+i*8])) << 32;
    memcpy(&c.times[i], &bits, sizeof bits);
    if ((i < c.count && i && c.times[i] <= c.times[i-1]) || (i >= c.count && bits)) return false;
  }
  for (int i = 808; i < 828; ++i) if (b[i]) return false;
  if (od_profile_blob_size() != 504 || od_profile_decode(&b[304], 504, &c.profile) ||
      c.profile.confidence < 0.3f) return false;
  out = c;
  return true;
}
bool EncodeCapture(const Capture& c, CaptureBytes& b) {
  b.fill(0);
  if (!c.count) return true;
  memcpy(b.data(), "ODCP", 4);
  Put32(&b[4], 1); Put32(&b[8], (uint32_t)b.size()); Put32(&b[12], 1);
  Put32(&b[16], c.width); Put32(&b[20], c.height); Put32(&b[24], c.timeScale);
  Put32(&b[28], c.timeStep); Put32(&b[32], c.count); Put32(&b[36], c.revision);
  if (c.count > 32) return false;
  for (unsigned i = 0; i < c.count; ++i) {
    const uint64_t t = uint64_t(c.times[i]);
    Put32(&b[40+i*8], uint32_t(t)); Put32(&b[44+i*8], uint32_t(t >> 32));
  }
  Put32(&b[296], c.parNum); Put32(&b[300], c.parDen);
  if (od_profile_blob_size() != 504 || od_profile_encode(&c.profile, &b[304], 504)) return false;
  Put32(&b[828], CRC(b.data(), 828));
  Capture check;
  return DecodeCapture(b, check);
}
CaptureDigest DigestCapture(const CaptureBytes& b) {
  uint8_t digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(b.data(), (CC_LONG)b.size(), digest);
  CaptureDigest d;
  for (int i = 0; i < 8; ++i) d[i] = Get32(digest + i*4);
  return d;
}
std::string CaptureStatus(const Capture& c) {
  if (!c.count) return "No captured profile. Automatic keeps learning; Freeze lasts this session.";
  std::string s = "Captured layer frames ";
  char v[160];
  for (unsigned i = 0; i < c.count; ++i) {
    // Exact rational frame positions when current time is not frame-aligned.
    if (c.times[i] % c.timeStep)
      snprintf(v, sizeof v, "%s%lld/%u", i ? "," : "", (long long)c.times[i], c.timeStep);
    else snprintf(v, sizeof v, "%s%lld", i ? "," : "", (long long)(c.times[i]/c.timeStep));
    s += v;
  }
  snprintf(v, sizeof v, ". Confidence %.0f%% (fit, not accuracy). %ux%u. Revision %u. Source/colour unchecked.",
           100*c.profile.confidence, c.width, c.height, c.revision);
  return s + v;
}
std::string CaptureDigestText(const CaptureDigest& d) {
  char text[65];
  // Conventional SHA-256 byte order; scalars store each 4-byte word LE.
  for (size_t i = 0; i < d.size(); ++i)
    for (int j = 0; j < 4; ++j) snprintf(text + 8*i + 2*j, 3, "%02x", uint8_t(d[i] >> (8*j)));
  return text;
}
uint64_t CaptureBridgeID() {
  static const uint64_t id = [] { uint64_t v; arc4random_buf(&v, sizeof v); return v; }();
  return id;
}
void LogCaptureBridge(const char* event, bool keys) {
  // Copy only diagnostic keys under the lock, then log outside it. Bounded
  // output even for a large project; exact requested keys are logged by GPU.
  struct Key { CaptureDigest digest; uint32_t revision; bool pinned; };
  std::array<Key, 32> entries{};
  size_t shown = 0, pins = 0, retained = 0;
  {
    std::lock_guard<std::mutex> lock(gCaptureMutex);
    pins = gHandlePins.size();
    auto add = [&](const auto& s, bool pinned) {
      if (!keys || shown == entries.size()) return;
      for (size_t i = 0; i < shown; ++i) if (entries[i].digest == s->digest) return;
      entries[shown++] = {s->digest, s->capture.revision, pinned};
    };
    for (const auto& p : gHandlePins) add(p.second, true);
    for (const auto& s : gRecentCaptures) if (s) { ++retained; add(s, false); }
  }
  Log("Capture bridge: event=%s pid=%d bridge=%016llx handle_pins=%zu retained=%zu capacity=%zu keys_shown=%zu (max 32 unique)",
      event, getpid(), (unsigned long long)CaptureBridgeID(), pins, retained, kRecentCaptures, shown);
  for (size_t i = 0; i < shown; ++i)
    Log("Capture bridge key: bridge=%016llx revision=%u sha256=%s source=%s",
        (unsigned long long)CaptureBridgeID(), entries[i].revision, CaptureDigestText(entries[i].digest).c_str(),
        entries[i].pinned ? "handle" : "retained");
}
std::shared_ptr<const CaptureSnapshot> FindCapture(const CaptureDigest& d, uint32_t revision) {
  std::lock_guard<std::mutex> lock(gCaptureMutex);
  std::shared_ptr<const CaptureSnapshot> found;
  for (const auto& entry : gHandlePins) {
    const auto& s = entry.second;
    if (s->digest != d || s->capture.revision != revision) continue;
    if (found && found->bytes != s->bytes) return {}; // even a digest collision fails closed
    found = s;
  }
  for (const auto& s : gRecentCaptures) {
    if (!s || s->digest != d || s->capture.revision != revision) continue;
    if (found && found->bytes != s->bytes) return {};
    found = s;
  }
  return found;
}
PF_Err ReadCaptureHandle(PF_InData* in_data, PF_Handle h, CaptureBytes& b) {
  b.fill(0);
  if (!h) return PF_Err_NONE; // documented default for COPY(NULL)
  if (!HaveHandles(in_data) || PF_GET_HANDLE_SIZE(h) != b.size()) return kBad;
  const void* p = PF_LOCK_HANDLE(h);
  if (!p) return PF_Err_OUT_OF_MEMORY;
  memcpy(b.data(), p, b.size());
  PF_UNLOCK_HANDLE(h);
  Capture c;
  return DecodeCapture(b, c) ? PF_Err_NONE : kBad;
}
PF_Err WriteCaptureHandle(PF_InData* in_data, PF_Handle h, const CaptureBytes& b, const char* origin) {
  Capture c;
  if (!HaveHandles(in_data) || !h || PF_GET_HANDLE_SIZE(h) != b.size() || !DecodeCapture(b, c)) return kBad;
  auto snapshot = Snapshot(b); // allocate before touching the host's value
  const auto digestText = snapshot ? CaptureDigestText(snapshot->digest) : std::string{};
  void* p = PF_LOCK_HANDLE(h);
  if (!p) return PF_Err_OUT_OF_MEMORY; // keep the previous pin and bytes
  // Pin can allocate. On failure unlock, keeping both old bytes and old pin.
  // No Adobe call is made while the bridge mutex is held.
  try {
    if (!Pin(h, snapshot)) { PF_UNLOCK_HANDLE(h); return kBad; }
  } catch (...) { PF_UNLOCK_HANDLE(h); throw; }
  memcpy(p, b.data(), b.size());
  PF_UNLOCK_HANDLE(h);
  if (snapshot) {
    Log("Capture bridge publish: origin=%s pid=%d bridge=%016llx revision=%u sha256=%s",
        origin, getpid(), (unsigned long long)CaptureBridgeID(), c.revision, digestText.c_str());
    LogCaptureBridge("published");
  }
  return PF_Err_NONE;
}
PF_Err NewCaptureHandle(PF_InData* in_data, const CaptureBytes& b, PF_Handle* out, const char* origin) {
  if (!out) return kBad;
  *out = nullptr;
  Capture c;
  if (!HaveHandles(in_data) || !DecodeCapture(b, c)) return kBad;
  PF_Handle h = PF_NEW_HANDLE(b.size());
  if (!h) return PF_Err_OUT_OF_MEMORY;
  try {
    PF_Err err = WriteCaptureHandle(in_data, h, b, origin);
    if (err) { Pin(h, {}); PF_DISPOSE_HANDLE(h); return err; }
  } catch (...) { Pin(h, {}); PF_DISPOSE_HANDLE(h); throw; }
  *out = h;
  return PF_Err_NONE;
}
PF_Err CaptureArbitrary(PF_InData* in_data, PF_ArbParamsExtra* e) {
  if (!e || e->id != OD_P_PROFILE_PAYLOAD || !HaveHandles(in_data)) return kBad;
  CaptureBytes b{};
  auto copy = [&](PF_Handle src, PF_Handle* dst, const char* origin) {
    if (dst) *dst = nullptr;
    PF_Err err = ReadCaptureHandle(in_data, src, b);
    return err ? err : NewCaptureHandle(in_data, b, dst, origin);
  };
  switch (e->which_function) {
    case PF_Arbitrary_NEW_FUNC: return NewCaptureHandle(in_data, b, e->u.new_func_params.arbPH);
    case PF_Arbitrary_DISPOSE_FUNC: {
      auto h = e->u.dispose_func_params.arbH;
      Pin(h, {});
      if (h) PF_DISPOSE_HANDLE(h);
      LogCaptureBridge("dispose-handle");
      return PF_Err_NONE;
    }
    case PF_Arbitrary_COPY_FUNC: return copy(e->u.copy_func_params.src_arbH, e->u.copy_func_params.dst_arbPH, "copy");
    case PF_Arbitrary_INTERP_FUNC: // discrete, even if a host ignores CANNOT_TIME_VARY
      return copy(e->u.interp_func_params.tF < 1 ? e->u.interp_func_params.left_arbH :
                  e->u.interp_func_params.right_arbH, e->u.interp_func_params.interpPH, "interp");
    case PF_Arbitrary_FLAT_SIZE_FUNC:
      *e->u.flat_size_func_params.flat_data_sizePLu = (A_u_long)b.size(); return PF_Err_NONE;
    case PF_Arbitrary_FLATTEN_FUNC: {
      auto& p = e->u.flatten_func_params;
      if (!p.flat_dataPV || p.buf_sizeLu != b.size()) return kBad;
      PF_Err err = ReadCaptureHandle(in_data, p.arbH, b);
      if (!err) memcpy(p.flat_dataPV, b.data(), b.size());
      return err;
    }
    case PF_Arbitrary_UNFLATTEN_FUNC: {
      auto& p = e->u.unflatten_func_params;
      if (!p.arbPH) return kBad;
      *p.arbPH = nullptr;
      if (!p.flat_dataPV || p.buf_sizeLu != b.size()) return kBad;
      memcpy(b.data(), p.flat_dataPV, b.size());
      return NewCaptureHandle(in_data, b, p.arbPH, "unflatten");
    }
    case PF_Arbitrary_COMPARE_FUNC: {
      auto& p = e->u.compare_func_params;
      CaptureBytes other;
      PF_Err err = ReadCaptureHandle(in_data, p.a_arbH, b);
      if (!err) err = ReadCaptureHandle(in_data, p.b_arbH, other);
      *p.compareP = !err && b == other ? PF_ArbCompare_EQUAL : PF_ArbCompare_NOT_EQUAL;
      return err;
    }
    case PF_Arbitrary_PRINT_SIZE_FUNC:
      *e->u.print_size_func_params.print_sizePLu = (A_u_long)(b.size()*2+1); return PF_Err_NONE;
    case PF_Arbitrary_PRINT_FUNC: {
      auto& p = e->u.print_func_params;
      if (!p.print_bufferPC || p.print_sizeLu < b.size()*2+1) return kBad;
      PF_Err err = ReadCaptureHandle(in_data, p.arbH, b);
      if (err) return err;
      const char* hex = "0123456789abcdef";
      for (size_t i = 0; i < b.size(); ++i) {
        p.print_bufferPC[2*i] = hex[b[i] >> 4]; p.print_bufferPC[2*i+1] = hex[b[i] & 15];
      }
      p.print_bufferPC[b.size()*2] = 0; return PF_Err_NONE;
    }
    case PF_Arbitrary_SCAN_FUNC: {
      auto& p = e->u.scan_func_params;
      if (!p.arbPH) return kBad;
      *p.arbPH = nullptr;
      if (!p.bufPC || (p.bytes_to_scanLu != b.size()*2 && p.bytes_to_scanLu != b.size()*2+1))
        return PF_Err_CANNOT_PARSE_KEYFRAME_TEXT;
      auto nibble = [](char c) { return c >= '0' && c <= '9' ? c-'0' : c >= 'a' && c <= 'f' ? c-'a'+10 : -1; };
      for (size_t i = 0; i < b.size(); ++i) {
        int a = nibble(p.bufPC[2*i]), d = nibble(p.bufPC[2*i+1]);
        if (a < 0 || d < 0) return PF_Err_CANNOT_PARSE_KEYFRAME_TEXT;
        b[i] = uint8_t(a*16+d);
      }
      if (p.bytes_to_scanLu == b.size()*2+1 && p.bufPC[b.size()*2]) return PF_Err_CANNOT_PARSE_KEYFRAME_TEXT;
      Capture c;
      if (!DecodeCapture(b, c)) return PF_Err_CANNOT_PARSE_KEYFRAME_TEXT;
      return NewCaptureHandle(in_data, b, p.arbPH, "scan");
    }
    default: return kBad;
  }
}
} // namespace odplug
