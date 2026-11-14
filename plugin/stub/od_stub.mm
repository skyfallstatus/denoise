// SPDX-License-Identifier: MIT
// Stand-in for libopendenoise.a: implements core/include/opendenoise/opendenoise.h as a
// pass-through copy so the plug-in shell can be tested without the real core.
// Selected with `make OD_CORE=stub` (see plugin/Makefile).
//
// Test hooks, all through environment variables (the bundle may export only
// EffectMain and xGPUFilterEntry, so there is no other way in):
//   OD_STUB_FAIL=1         every od_render fails (the plug-in's pass-through path)
//   OD_STUB_TRACE=1        one human-readable line per render on stderr
//   OD_STUB_RECORD=<file>  appends one machine-readable line per session create,
//                          reset and destroy, context destroy and render (the
//                          parameters, offsets and frame keys the plug-in passed)
//   OD_STUB_FILL=1         the output is the fixed pattern StubFillValue() below,
//                          whatever the input: the harness checks the plug-in's
//                          conversion back to the host's format against it
//   OD_STUB_CHECK=<tol>    every frame handed in (current and neighbours) is
//                          compared with StubCheckValue() below within <tol>;
//                          the output is all 1.0 if every value matched, all 0.0
//                          otherwise: the harness checks the conversion from
//                          the host's format through it
// The two patterns are repeated in plugin/tests/harness.mm.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <atomic>
#include <mutex>
#include <string>
#include "opendenoise/opendenoise.h"

// Records name sessions and contexts by a serial number, never reused (an
// address can be, after a destroy).
static std::atomic<unsigned> gSerial{0};
struct ODContext {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  unsigned serial = ++gSerial;
};
struct ODSession {
  ODContext* ctx;
  std::string error;
  unsigned serial = ++gSerial;
};

namespace {

// Output of OD_STUB_FILL, channel c (B, G, R, A) of row y.
float StubFillValue(int c, uint32_t y) {
  static const float base[4] = {0.25f, 0.5f, 0.75f, 1.0f};
  return c < 3 ? base[c] - (float)(y % 8) / 256.f : base[3];
}
// What OD_STUB_CHECK expects in channel c (B, G, R, A) of pixel (x, y).
float StubCheckValue(int c, uint32_t x, uint32_t y) {
  return (float)((x * 7 + y * 13 + (uint32_t)c * 61) % 239) / 256.f + 0.03f;
}

std::mutex gRecordMutex;
void Record(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void Record(const char* fmt, ...) {
  const char* path = getenv("OD_STUB_RECORD");
  if (!path || !*path) return;
  std::lock_guard<std::mutex> lock(gRecordMutex);
  FILE* f = fopen(path, "a");
  if (!f) return;
  va_list ap;
  va_start(ap, fmt);
  vfprintf(f, fmt, ap);
  va_end(ap);
  fputc('\n', f);
  fclose(f);
}

float ReadValue(const ODFrame& f, uint32_t x, uint32_t y, int c) {
  const char* row = (const char*)((__bridge id<MTLBuffer>)f.mtlBuffer).contents + (size_t)y * f.rowBytes;
  if (f.format == OD_PIXEL_BGRA32F) return ((const float*)row)[x * 4 + c];
  return (float)((const __fp16*)row)[x * 4 + c];
}
void WriteValue(const ODFrame& f, uint32_t x, uint32_t y, int c, float v) {
  char* row = (char*)((__bridge id<MTLBuffer>)f.mtlBuffer).contents + (size_t)y * f.rowBytes;
  if (f.format == OD_PIXEL_BGRA32F) ((float*)row)[x * 4 + c] = v;
  else ((__fp16*)row)[x * 4 + c] = (__fp16)v;
}

}  // namespace

ODContext* od_context_create(void* device, void* queue, char* err, size_t errLen) {
  if (!device) {
    if (err && errLen) snprintf(err, errLen, "stub: no device");
    return nullptr;
  }
  ODContext* c = new ODContext();
  c->device = (__bridge id<MTLDevice>)device;
  c->queue = queue ? (__bridge id<MTLCommandQueue>)queue : [c->device newCommandQueue];
  return c;
}
void od_context_destroy(ODContext* c) {
  Record("ctxdestroy c=%u", c->serial);
  delete c;
}

ODSession* od_session_create(ODContext* c) {
  if (!c) return nullptr;
  ODSession* s = new ODSession();
  s->ctx = c;
  Record("create s=%u c=%u", s->serial, c->serial);
  return s;
}
void od_session_destroy(ODSession* s) {
  Record("destroy s=%u", s->serial);
  delete s;
}
void od_session_reset(ODSession* s) { Record("reset s=%u", s->serial); }

ODParams od_params_default(void) {
  ODParams p;
  memset(&p, 0, sizeof p);
  p.lumaStrength = 0.8f; p.chromaStrength = 1.0f; p.temporalRadius = 2; p.temporalStrength = 1.0f;
  p.spatialStrength = 0.5f; p.sharpen = 0.0f; p.noiseLevel = 1.0f; p.quality = 1;
  p.view = OD_VIEW_OUTPUT; p.waitUntilCompleted = 0; p.manualProfile = nullptr;
  p.advanced = 1;  // API v6: neutral values, as the real core
  for (int g = 0; g < 2; ++g)
    for (int b = 0; b < OD_BANDS; ++b) p.bandNoiseLevel[g][b] = p.bandAmount[g][b] = 1.0f;
  p.channelNoiseLevel[0] = p.channelNoiseLevel[1] = p.channelNoiseLevel[2] = 1.0f;
  p.grainFloor = 1.0f; p.speckRemoval = 1; p.temporalMotionThreshold = 1.0f; p.temporalEdgeGate = 0.75f;
  p.sharpenBand[0] = p.sharpenBand[1] = 1.0f; p.sharpenCoring = 1.0f; p.sharpenHalo = 1.0f; p.profileHold = 0;
  p.grainStability = 0.0f; p.coarseChroma = 0;  // API v9: off, as the real core
  p.grainMode = OD_GRAIN_FINEST;  // API v10: finest grain only, as the real core
  return p;
}

int od_render(ODSession* s, const ODFrame* frames, const int* offsets, int frameCount,
                const ODParams* params, const ODFrame* out, ODStats* stats) {
  if (!s) return 1;
  if (!frames || !offsets || frameCount < 1 || frameCount > OD_MAX_FRAMES || !params || !out ||
      !frames[0].mtlBuffer || !out->mtlBuffer) {
    s->error = "stub: bad arguments";
    return 2;
  }
  for (int i = 0; i < frameCount; ++i) {
    if (frames[i].width != out->width || frames[i].height != out->height || frames[i].format != out->format) {
      s->error = "stub: frame size or format mismatch";
      return 3;
    }
  }
  if (getenv("OD_STUB_RECORD")) {
    ODParams expected = od_params_default();
    std::string mismatches;
#define PIN_FIELD(f) if (memcmp(&params->f,&expected.f,sizeof expected.f)) mismatches += #f ",";
    PIN_FIELD(lumaStrength);
    PIN_FIELD(chromaStrength);
    PIN_FIELD(temporalRadius);
    PIN_FIELD(temporalStrength);
    PIN_FIELD(spatialStrength);
    PIN_FIELD(sharpen);
    PIN_FIELD(noiseLevel);
    PIN_FIELD(quality);
    PIN_FIELD(view);
    PIN_FIELD(manualProfile);
    PIN_FIELD(debugFlags);
    PIN_FIELD(debugSkipStages);
    PIN_FIELD(noFrameCache);
    PIN_FIELD(realtime);
    PIN_FIELD(advanced);
    PIN_FIELD(bandNoiseLevel);
    PIN_FIELD(channelNoiseLevel);
    PIN_FIELD(bandAmount);
    PIN_FIELD(grainFloor);
    PIN_FIELD(speckRemoval);
    PIN_FIELD(temporalMotionThreshold);
    PIN_FIELD(temporalEdgeGate);
    PIN_FIELD(sharpenBand);
    PIN_FIELD(sharpenCoring);
    PIN_FIELD(sharpenHalo);
    PIN_FIELD(profileHold);
    PIN_FIELD(grainStability);
    PIN_FIELD(coarseChroma);
    PIN_FIELD(grainMode);
#undef PIN_FIELD
    // These two host execution-policy fields intentionally differ from the
    // core standalone defaults; the harness checks each separately.
    std::string off, keys;
    char buf[64];
    for (int i = 0; i < frameCount; ++i) {
      snprintf(buf, sizeof buf, "%s%d", i ? "," : "", offsets[i]);
      off += buf;
      if (frames[i].frameKey == OD_NO_KEY) snprintf(buf, sizeof buf, "%snone", i ? "," : "");
      else snprintf(buf, sizeof buf, "%s%llx", i ? "," : "", (unsigned long long)frames[i].frameKey);
      keys += buf;
    }
    // API v6 fields: nlY0..3 / nlC0..3 band noise levels, amY0..3 / amC0..3
    // band amounts, cnY/cnCb/cnCr channel noise levels, then one key each;
    // API v9: gstab (grain stability), cchroma (coarse chroma cleanup);
    // API v10: grain (grain mode).
    std::string adv;
    for (int g = 0; g < 2; ++g)
      for (int b = 0; b < OD_BANDS; ++b) {
        snprintf(buf, sizeof buf, " nl%s%d=%.6g am%s%d=%.6g", g ? "C" : "Y", b, params->bandNoiseLevel[g][b],
                 g ? "C" : "Y", b, params->bandAmount[g][b]);
        adv += buf;
      }
    char adv2[512];
    snprintf(adv2, sizeof adv2, " cnY=%.6g cnCb=%.6g cnCr=%.6g floor=%.6g speck=%u mthr=%.6g egate=%.6g shf=%.6g shm=%.6g "
             "shcore=%.6g shhalo=%.6g hold=%u gstab=%.6g cchroma=%u grain=%u", params->channelNoiseLevel[0],
             params->channelNoiseLevel[1], params->channelNoiseLevel[2], params->grainFloor, params->speckRemoval,
             params->temporalMotionThreshold, params->temporalEdgeGate, params->sharpenBand[0], params->sharpenBand[1],
             params->sharpenCoring, params->sharpenHalo, params->profileHold, params->grainStability,
             params->coarseChroma, params->grainMode);
    adv += adv2;
    Record("render s=%u n=%d w=%u h=%u fmt=%d off=%s keys=%s luma=%.6g chroma=%.6g radius=%d tstr=%.6g "
           "spatial=%.6g sharpen=%.6g noise=%.6g quality=%d view=%d wait=%d manual=%d nocache=%u realtime=%u verify=%u "
           "adv=%u%s",
           s->serial, frameCount, out->width, out->height, (int)out->format, off.c_str(), keys.c_str(),
           params->lumaStrength, params->chromaStrength, params->temporalRadius, params->temporalStrength,
           params->spatialStrength, params->sharpen, params->noiseLevel, params->quality, params->view,
           params->waitUntilCompleted, params->manualProfile ? 1 : 0, params->noFrameCache, params->realtime,
           params->verifyFrameCache, params->advanced, adv.c_str());
    Record("defaultpin mismatches=%s wait=%d verify=%u recovery=%u temporalModel=%u", mismatches.empty() ? "none" : mismatches.c_str(),
           params->waitUntilCompleted,params->verifyFrameCache,params->chromaRecovery,params->temporalNoiseModel);

  }
  // Test hook: OD_STUB_FAIL=1 makes every render fail, to exercise the
  // plug-in's pass-through path.
  if (getenv("OD_STUB_FAIL")) {
    s->error = "stub: forced failure";
    return 4;
  }
  if (getenv("OD_STUB_TRACE")) {
    fprintf(stderr, "[stub] render %ux%u fmt=%d frames=%d offsets:", out->width, out->height, (int)out->format, frameCount);
    for (int i = 0; i < frameCount; ++i) fprintf(stderr, " %+d", offsets[i]);
    fprintf(stderr, " key0=%s q=%d view=%d luma=%.2f noise=%.2f wait=%d\n", frames[0].frameKey == OD_NO_KEY ? "none" : "set",
            params->quality, params->view, params->lumaStrength, params->noiseLevel, params->waitUntilCompleted);
  }
  if (stats) {
    memset(stats, 0, sizeof *stats);
    stats->framesUsed = frameCount;
  }
  // The pattern hooks work on the CPU side of shared buffers, synchronously.
  if (const char* tolText = getenv("OD_STUB_CHECK")) {
    const float tol = (float)atof(tolText);
    bool match = true;
    for (int i = 0; i < frameCount && match; ++i)
      for (uint32_t y = 0; y < out->height && match; ++y)
        for (uint32_t x = 0; x < out->width && match; ++x)
          for (int c = 0; c < 4; ++c)
            if (!(fabsf(ReadValue(frames[i], x, y, c) - StubCheckValue(c, x, y)) <= tol)) {
              match = false;
              break;
            }
    for (uint32_t y = 0; y < out->height; ++y)
      for (uint32_t x = 0; x < out->width; ++x)
        for (int c = 0; c < 4; ++c) WriteValue(*out, x, y, c, match ? 1.f : 0.f);
    return 0;
  }
  if (getenv("OD_STUB_FILL")) {
    for (uint32_t y = 0; y < out->height; ++y)
      for (uint32_t x = 0; x < out->width; ++x)
        for (int c = 0; c < 4; ++c) WriteValue(*out, x, y, c, StubFillValue(c, y));
    return 0;
  }
  if (frames[0].mtlBuffer == out->mtlBuffer) return 0;  // in place: nothing to do
  @autoreleasepool {
    id<MTLBuffer> src = (__bridge id<MTLBuffer>)frames[0].mtlBuffer;
    id<MTLBuffer> dst = (__bridge id<MTLBuffer>)out->mtlBuffer;
    const NSUInteger bpp = out->format == OD_PIXEL_BGRA32F ? 16 : 8;
    id<MTLCommandBuffer> cb = [s->ctx->queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    for (uint32_t y = 0; y < out->height; ++y) {
      [blit copyFromBuffer:src sourceOffset:(NSUInteger)y * frames[0].rowBytes
                  toBuffer:dst destinationOffset:(NSUInteger)y * out->rowBytes
                      size:out->width * bpp];
    }
    [blit endEncoding];
    [cb commit];
    if (params->waitUntilCompleted) {
      [cb waitUntilCompleted];
      if (cb.status == MTLCommandBufferStatusError) {
        s->error = "stub: command buffer failed";
        return 5;
      }
    }
  }
  return 0;
}

int od_profile_frame(ODSession* s, const ODFrame*, ODNoiseProfile* outProfile) {
  if (!s || !outProfile) return 1;
  memset(outProfile, 0, sizeof *outProfile);
  return 0;
}

const char* od_last_error(ODSession* s) { return s ? s->error.c_str() : "stub: no session"; }
const char* od_first_async_error(ODSession*) { return ""; }
int od_wait_idle(double) { return 0; }  // nothing asynchronous of its own
const char* od_version(void) { return "stub-0"; }

int od_profile_frame_with_params(ODSession* s, const ODFrame* f, const ODParams* p, ODNoiseProfile* out) {
  Record("profile recovery=%u",p ? p->chromaRecovery : 0);
  return od_profile_frame(s,f,out);
}
int od_session_last_analysis(ODSession*, ODAnalysis* a) { if (!a) return -1; *a = {}; return 0; }
