// SPDX-License-Identifier: MIT
// Correctness tests for the Open Denoise core: round-trip identity, cache
// correctness, in-place output, missing neighbours, argument checking, pool
// and cache bounds, ingest/output arithmetic, the context's memory policy
// (shared scratch, byte budget, trim off the render thread), a queue that
// gives no command buffer, garbage noise figures from a caller, and the frame
// cache checked on the GPU against the host's buffers (verifyFrameCache).
//
// These tests must keep passing when the stub stages are replaced by real
// ones, so identity is only asserted where the contract promises it:
// strengths at zero, or the middle stages switched off with debugSkipStages.
// Memory: the largest test holds seven 1920x1080 16f frames and seven
// sessions' caches at that size (< 1.5 GB); memory_device holds eleven
// 1920x1080 16f host buffers and, before its fix, up to six renders' working
// sets at once (about 1.7 GB).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include "opendenoise/opendenoise.h"

#include "context.h"
#include "debug.h"

static int gFailures = 0;
static int gChecks = 0;
static const char* gTest = "";

#define CHECK(cond, ...)                                 \
  do {                                                   \
    ++gChecks;                                           \
    if (!(cond)) {                                       \
      ++gFailures;                                       \
      fprintf(stderr, "FAIL [%s] %s:%d: ", gTest, __FILE__, __LINE__); \
      fprintf(stderr, __VA_ARGS__);                      \
      fprintf(stderr, "\n");                             \
    }                                                    \
  } while (0)

static ODNoiseProfile whiteProfile(float s);

static const uint32_t kSkipMiddle = (1u << OD_STAGE_PROFILE) | (1u << OD_STAGE_TEMPORAL) |
                                    (1u << OD_STAGE_SPATIAL) | (1u << OD_STAGE_SHARPEN);

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;
static ODContext* gCtx;

struct Buf {
  id<MTLBuffer> buffer;
  uint32_t width, height, rowBytes;
  ODPixelFormat format;
  uint32_t bpp() const { return format == OD_PIXEL_BGRA16F ? 8 : 16; }
  ODFrame frame(int64_t key = OD_NO_KEY) const {
    ODFrame f;
    f.mtlBuffer = (__bridge void*)buffer;
    f.width = width;
    f.height = height;
    f.rowBytes = rowBytes;
    f.format = format;
    f.frameKey = key;
    return f;
  }
  uint8_t* bytes() const { return (uint8_t*)buffer.contents; }
};

static uint32_t hash32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

// Deterministic content: a smooth part plus noise, salted with awkward values
// (negatives, super-whites, zeros, tiny numbers, and for 32f values beyond the
// half range).
static float testValue(uint32_t x, uint32_t y, uint32_t c, uint32_t seed, bool is32) {
  uint32_t h = hash32(x * 73856093u ^ y * 19349663u ^ c * 83492791u ^ seed * 2654435761u);
  float u = (float)(h & 0xffffff) / 16777215.0f;
  float v = 0.3f + 0.3f * sinf(0.05f * x + 0.5f * c + 0.3f * seed) * cosf(0.07f * y) + 0.1f * (u - 0.5f);
  if (c == 3) v = 0.25f + 0.75f * u;  // alpha
  switch (h >> 27) {
    case 0: return -0.37f * u;
    case 1: return 1.0f + 3.0f * u;
    case 2: return 0.0f;
    case 3: return 1e-6f * u;
    case 4: return is32 ? 250000.0f * u : 30000.0f * u;
    case 5: return -1234.5f * u;
    default: return v;
  }
}

static Buf makeBuf(uint32_t w, uint32_t h, ODPixelFormat fmt, uint32_t rowPad) {
  Buf b;
  b.width = w;
  b.height = h;
  b.format = fmt;
  b.rowBytes = w * b.bpp() + rowPad;
  b.buffer = [gDevice newBufferWithLength:(NSUInteger)b.rowBytes * h options:MTLResourceStorageModeShared];
  memset(b.bytes(), 0xA5, (size_t)b.rowBytes * h);
  return b;
}

// The values are stored in the buffer's format. beyondHalf: 32f buffers also
// get values beyond the half range (pass-through tests); off for tests where
// the stages must do real work. It changes the values, never the storage
// (testFixtures).
static void fillBuf(const Buf& b, uint32_t seed, bool beyondHalf = true) {
  const bool is32 = b.format == OD_PIXEL_BGRA32F;
  const bool wide = is32 && beyondHalf;
  for (uint32_t y = 0; y < b.height; ++y) {
    uint8_t* row = b.bytes() + (size_t)y * b.rowBytes;
    for (uint32_t x = 0; x < b.width; ++x) {
      for (uint32_t c = 0; c < 4; ++c) {
        float v = testValue(x, y, c, seed, wide);
        if (is32) {
          ((float*)row)[x * 4 + c] = v;
        } else {
          ((__fp16*)row)[x * 4 + c] = (__fp16)v;
        }
      }
    }
  }
}

static float pixelAt(const Buf& b, uint32_t x, uint32_t y, uint32_t c) {
  const uint8_t* row = b.bytes() + (size_t)y * b.rowBytes;
  return b.format == OD_PIXEL_BGRA32F ? ((const float*)row)[x * 4 + c] : (float)((const __fp16*)row)[x * 4 + c];
}

// Number of pixels whose bytes differ between a and b (same size and format).
static size_t diffPixels(const Buf& a, const Buf& b) {
  size_t n = 0;
  const uint32_t bpp = a.bpp();
  for (uint32_t y = 0; y < a.height; ++y) {
    const uint8_t* ra = a.bytes() + (size_t)y * a.rowBytes;
    const uint8_t* rb = b.bytes() + (size_t)y * b.rowBytes;
    if (memcmp(ra, rb, (size_t)a.width * bpp) == 0) continue;
    for (uint32_t x = 0; x < a.width; ++x) n += memcmp(ra + x * bpp, rb + x * bpp, bpp) != 0;
  }
  return n;
}

// Row padding must never be written.
static size_t dirtyPadding(const Buf& b) {
  size_t n = 0;
  const uint32_t used = b.width * b.bpp();
  for (uint32_t y = 0; y < b.height; ++y) {
    const uint8_t* row = b.bytes() + (size_t)y * b.rowBytes;
    for (uint32_t i = used; i < b.rowBytes; ++i) n += row[i] != 0xA5;
  }
  return n;
}

static void waitForQueue() {
  id<MTLCommandBuffer> cb = [gQueue commandBuffer];
  [cb commit];
  [cb waitUntilCompleted];
}

static ODParams syncParams() {
  ODParams p = od_params_default();
  p.waitUntilCompleted = 1;
  return p;
}

static const char* fmtName(ODPixelFormat f) { return f == OD_PIXEL_BGRA32F ? "32f" : "16f"; }

// ---------------------------------------------------------------------------

// The fixtures themselves: fillBuf stores what testValue gives in the buffer's
// own format, whatever beyondHalf says (beyondHalf only decides whether a 32f
// buffer also gets values beyond the half range), and leaves the row padding
// alone. It once chose the writer by beyondHalf: a 32f buffer filled with
// beyondHalf = false got half floats in the first half of every row and the
// 0xA5 sentinel in the rest, so the cache and in-place tests that use it ran
// on malformed values instead of a normal 32f picture.
static void testFixtures() {
  gTest = "fixtures";
  for (ODPixelFormat fmt : {OD_PIXEL_BGRA32F, OD_PIXEL_BGRA16F}) {
    for (bool beyond : {true, false}) {
      Buf b = makeBuf(37, 5, fmt, 12);
      fillBuf(b, 3, beyond);
      const bool is32 = fmt == OD_PIXEL_BGRA32F;
      size_t wrong = 0, sentinel = 0, overHalf = 0;
      for (uint32_t y = 0; y < b.height; ++y) {
        const uint8_t* row = b.bytes() + (size_t)y * b.rowBytes;
        for (uint32_t x = 0; x < b.width; ++x) {
          for (uint32_t c = 0; c < 4; ++c) {
            const float v = testValue(x, y, c, 3, is32 && beyond);
            const float want = is32 ? v : (float)(__fp16)v;
            const float got = pixelAt(b, x, y, c);
            wrong += !(got == want);
            overHalf += std::fabs(got) > 65504.0f;
            const uint8_t* cell = row + (size_t)(x * 4 + c) * (is32 ? 4 : 2);
            sentinel += cell[0] == 0xA5 && cell[1] == 0xA5 && (!is32 || (cell[2] == 0xA5 && cell[3] == 0xA5));
          }
        }
      }
      const char* what = beyond ? "beyondHalf" : "no beyondHalf";
      CHECK(wrong == 0, "%s, %s: %zu stored values differ from testValue in the buffer's format", fmtName(fmt), what,
            wrong);
      CHECK(sentinel == 0, "%s, %s: %zu values still hold the 0xA5 sentinel", fmtName(fmt), what, sentinel);
      CHECK(dirtyPadding(b) == 0, "%s, %s: row padding written", fmtName(fmt), what);
      if (is32 && beyond)
        CHECK(overHalf > 0, "32f, beyondHalf: no value beyond the half range");
      else
        CHECK(overHalf == 0, "%s, %s: %zu values beyond the half range", fmtName(fmt), what, overHalf);
    }
  }
}

static void testCompileAll() {
  gTest = "compile_all";
  char err[2048] = {0};
  int rc = od_debug_compile_all(gCtx, err, sizeof(err));
  CHECK(rc == 0, "shader compile failed: %s", err);
}

static void testRoundTrip() {
  gTest = "round_trip";
  const uint32_t sizes[][2] = {{1, 1}, {3, 2}, {17, 9}, {333, 127}, {1921, 1081}};
  const uint32_t pads[] = {0, 12, 64};
  ODSession* s = od_session_create(gCtx);
  for (ODPixelFormat fmt : {OD_PIXEL_BGRA32F, OD_PIXEL_BGRA16F}) {
    for (auto& sz : sizes) {
      for (uint32_t pad : pads) {
        // Real path (ingest + output), middle stages off; and the three I/O
        // variants. Mode 3 is the strength-0 path with a different out stride.
        for (int mode = 0; mode < 5; ++mode) {
          @autoreleasepool {
            Buf in = makeBuf(sz[0], sz[1], fmt, pad);
            Buf out = makeBuf(sz[0], sz[1], fmt, mode == 3 ? pad + 16 : (mode == 4 ? 0 : pad));
            fillBuf(in, 7 + pad);
            ODParams p = syncParams();
            p.debugSkipStages = kSkipMiddle;
            if (mode == 1) p.debugFlags = OD_DEBUG_IO_BLIT;
            if (mode == 2) p.debugFlags = OD_DEBUG_IO_VIEW;
            if (mode >= 3) {
              p.debugSkipStages = 0;
              p.lumaStrength = p.chromaStrength = p.sharpen = 0;
            }
            ODFrame fi = in.frame(mode == 0 ? 42 : OD_NO_KEY), fo = out.frame();
            int off = 0;
            ODStats st;
            int rc = od_render(s, &fi, &off, 1, &p, &fo, &st);
            if ((mode == 1 || mode == 2) && rc != 0) {
              // Blits and linear texture views need an aligned stride; refusing is fine.
              CHECK(strlen(od_last_error(s)) > 0, "view mode failed without a message");
              continue;
            }
            CHECK(rc == 0, "%s %ux%u pad %u mode %d: render failed: %s", fmtName(fmt), sz[0], sz[1], pad, mode,
                  od_last_error(s));
            if (rc != 0) continue;
            size_t d = diffPixels(in, out);
            CHECK(d == 0, "%s %ux%u pad %u mode %d: %zu pixels differ from the input", fmtName(fmt), sz[0], sz[1],
                  pad, mode, d);
            if (mode != 1 && mode != 2) {
              size_t dp = dirtyPadding(out);
              CHECK(dp == 0, "%s %ux%u pad %u mode %d: %zu padding bytes overwritten", fmtName(fmt), sz[0], sz[1],
                    pad, mode, dp);
            }
          }
        }
      }
    }
  }
  od_session_destroy(s);
}

static void testSpecialValues() {
  gTest = "special_values";
  // NaN, Inf and -0 must survive the real path bit for bit (32f).
  ODSession* s = od_session_create(gCtx);
  Buf in = makeBuf(8, 2, OD_PIXEL_BGRA32F, 0), out = makeBuf(8, 2, OD_PIXEL_BGRA32F, 0);
  fillBuf(in, 1);
  float* f = (float*)in.bytes();
  f[0] = NAN;
  f[5] = INFINITY;
  f[10] = -INFINITY;
  f[13] = -0.0f;
  f[16] = 3.0e38f;
  f[21] = 1e-42f;  // denormal
  ODParams p = syncParams();
  p.debugSkipStages = kSkipMiddle;
  ODFrame fi = in.frame(), fo = out.frame();
  int off = 0;
  CHECK(od_render(s, &fi, &off, 1, &p, &fo, nullptr) == 0, "render failed: %s", od_last_error(s));
  CHECK(memcmp(in.bytes(), out.bytes(), 8 * 2 * 16) == 0, "special values did not round-trip bit-exactly");
  od_session_destroy(s);
}

static void testIngestValues() {
  gTest = "ingest_values";
  ODSession* s = od_session_create(gCtx);
  for (ODPixelFormat fmt : {OD_PIXEL_BGRA32F, OD_PIXEL_BGRA16F}) {
    Buf in = makeBuf(61, 37, fmt, 20);
    fillBuf(in, 3);
    if (fmt == OD_PIXEL_BGRA32F) ((float*)in.bytes())[4] = NAN;  // pixel (1,0), B
    std::vector<float> w((size_t)in.width * in.height * 4);
    ODFrame fi = in.frame();
    int rc = od_debug_read_working(s, &fi, w.data());
    CHECK(rc == 0, "read_working failed: %s", od_last_error(s));
    if (rc != 0) continue;
    size_t bad = 0;
    double worst = 0;
    for (uint32_t y = 0; y < in.height; ++y) {
      for (uint32_t x = 0; x < in.width; ++x) {
        double B = pixelAt(in, x, y, 0), G = pixelAt(in, x, y, 1), R = pixelAt(in, x, y, 2), A = pixelAt(in, x, y, 3);
        double Y = 0.2126 * R + 0.7152 * G + 0.0722 * B;
        double ref[4] = {Y, (B - Y) / 1.8556, (R - Y) / 1.5748, A};
        if (std::isnan(Y)) continue;  // NaN pixel: checked below
        const float* got = &w[((size_t)y * in.width + x) * 4];
        for (int c = 0; c < 4; ++c) {
          double r = ref[c];
          if (std::isnan(r)) r = 0;  // scrubbed
          r = std::fmin(65504.0, std::fmax(-65504.0, r));
          // half has 11 significant bits: relative 2^-11, plus float32 slack
          // from cancellation in the colour differences.
          double mag = std::fabs(R) + std::fabs(G) + std::fabs(B) + std::fabs(A);
          double tol = std::fabs(r) * 6e-4 + mag * 2e-7 + 1e-7;
          double e = std::fabs((double)got[c] - r);
          if (!(e <= tol)) {
            ++bad;
            if (bad <= 3)
              fprintf(stderr, "  %s (%u,%u) ch %d: got %g want %g\n", fmtName(fmt), x, y, c, got[c], r);
          }
          if (std::fabs(r) > 1e-3) worst = std::fmax(worst, e / std::fabs(r));
        }
      }
    }
    if (fmt == OD_PIXEL_BGRA32F)
      CHECK(w[4] == 0 && w[5] == 0 && w[6] == 0, "NaN pixel not scrubbed to 0: %g %g %g", w[4], w[5], w[6]);
    CHECK(bad == 0, "%s: %zu working-space values out of tolerance", fmtName(fmt), bad);
    CHECK(worst < 6e-4, "%s: worst relative error %g", fmtName(fmt), worst);
  }
  od_session_destroy(s);
}

// Drives the output kernel directly with a known non-zero delta, so the
// "source + toRGB(final - ingested)" arithmetic is checked even while the
// stages are stubs.
static void testOutputDelta() {
  gTest = "output_delta";
  const uint32_t W = 19, H = 7;
  od::Context& ctx = *gCtx->impl;
  for (ODPixelFormat fmt : {OD_PIXEL_BGRA32F, OD_PIXEL_BGRA16F}) {
    const bool is32 = fmt == OD_PIXEL_BGRA32F;
    std::string err;
    id<MTLComputePipelineState> pso = ctx.pipeline(is32 ? "output_bgra32f" : "output_bgra16f", &err);
    CHECK(pso != nil, "pipeline: %s", err.c_str());
    if (!pso) continue;
    Buf in = makeBuf(W, H, fmt, 24), out = makeBuf(W, H, fmt, 8);
    fillBuf(in, 11);
    MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                 width:W
                                                                                height:H
                                                                             mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead;
    d.storageMode = MTLStorageModeShared;
    id<MTLTexture> t0 = [gDevice newTextureWithDescriptor:d], t1 = [gDevice newTextureWithDescriptor:d];
    std::vector<float> a((size_t)W * H * 4), b((size_t)W * H * 4);
    for (uint32_t y = 0; y < H; ++y) {
      for (uint32_t x = 0; x < W; ++x) {
        for (uint32_t c = 0; c < 4; ++c) {
          size_t i = ((size_t)y * W + x) * 4 + c;
          a[i] = testValue(x, y, c, 5, false) * 0.01f;
          // every third pixel untouched, the others moved by a known delta
          float delta = (x % 3 == 0) ? 0.0f : 0.02f * (float)((int)((x + y + c) % 7) - 3);
          b[i] = a[i] + delta;
        }
      }
    }
    MTLRegion r = MTLRegionMake2D(0, 0, W, H);
    [t0 replaceRegion:r mipmapLevel:0 withBytes:a.data() bytesPerRow:W * 16];
    [t1 replaceRegion:r mipmapLevel:0 withBytes:b.data() bytesPerRow:W * 16];
    struct {
      uint32_t srcStride, dstStride, mode;
      float gain;
    } u = {in.rowBytes / (is32 ? 4u : 2u), out.rowBytes / (is32 ? 4u : 2u), 0, 4.0f};
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:in.buffer offset:0 atIndex:0];
    [enc setBuffer:out.buffer offset:0 atIndex:1];
    [enc setBytes:&u length:sizeof(u) atIndex:2];
    [enc setTexture:t0 atIndex:0];
    [enc setTexture:t1 atIndex:1];
    [enc dispatchThreads:MTLSizeMake(W, H, 1) threadsPerThreadgroup:od::Context::threadgroupFor(pso)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    size_t bad = 0, untouchedBad = 0;
    for (uint32_t y = 0; y < H; ++y) {
      for (uint32_t x = 0; x < W; ++x) {
        size_t i = ((size_t)y * W + x) * 4;
        double dY = (double)b[i] - a[i], dCb = (double)b[i + 1] - a[i + 1], dCr = (double)b[i + 2] - a[i + 2];
        double dR = dY + 1.5748 * dCr, dB = dY + 1.8556 * dCb;
        double dG = (dY - 0.2126 * dR - 0.0722 * dB) / 0.7152;
        double want[4] = {pixelAt(in, x, y, 0) + dB, pixelAt(in, x, y, 1) + dG, pixelAt(in, x, y, 2) + dR,
                          pixelAt(in, x, y, 3)};
        if (x % 3 == 0) {
          const uint32_t bpp = in.bpp();
          untouchedBad += memcmp(in.bytes() + (size_t)y * in.rowBytes + x * bpp,
                                 out.bytes() + (size_t)y * out.rowBytes + x * bpp, bpp) != 0;
          continue;
        }
        for (int c = 0; c < 4; ++c) {
          double got = pixelAt(out, x, y, c);
          double tol = is32 ? (std::fabs(want[c]) * 1e-6 + 1e-6) : (std::fabs(want[c]) * 6e-4 + 1e-6);
          if (c == 3) tol = 0;  // alpha is copied
          if (!(std::fabs(got - want[c]) <= tol)) {
            if (++bad <= 3) fprintf(stderr, "  %s (%u,%u) ch %d: got %g want %g\n", fmtName(fmt), x, y, c, got, want[c]);
          }
        }
      }
    }
    CHECK(bad == 0, "%s: %zu output values wrong", fmtName(fmt), bad);
    CHECK(untouchedBad == 0, "%s: %zu untouched pixels not bit-identical", fmtName(fmt), untouchedBad);
    CHECK(dirtyPadding(out) == 0, "%s: padding overwritten", fmtName(fmt));
  }
}

// 16f output: a finite result beyond the half range is clamped to +-65504,
// never written as infinity (the untouched-pixel path copies bits and skips
// the clamp, so the output kernel is driven directly with a real delta).
static void testOutputClamp16() {
  gTest = "output_clamp16";
  const uint32_t W = 4, H = 1;
  od::Context& ctx = *gCtx->impl;
  std::string err;
  id<MTLComputePipelineState> pso = ctx.pipeline("output_bgra16f", &err);
  CHECK(pso != nil, "pipeline: %s", err.c_str());
  if (!pso) return;
  Buf in = makeBuf(W, H, OD_PIXEL_BGRA16F, 0), out = makeBuf(W, H, OD_PIXEL_BGRA16F, 0);
  // source B,G,R and the luma delta per pixel: +60000 + 30000, -60000 - 30000,
  // +60000 - 30000 (no clamp), -60000 + 30000 (no clamp).
  const float src[W] = {60000, -60000, 60000, -60000}, dy[W] = {30000, -30000, -30000, 30000};
  const float want[W] = {65504, -65504, 30000, -30000};
  for (uint32_t x = 0; x < W; ++x) {
    __fp16* px = (__fp16*)in.bytes() + x * 4;
    px[0] = px[1] = px[2] = (__fp16)src[x];
    px[3] = (__fp16)1.0f;
  }
  MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                               width:W
                                                                              height:H
                                                                           mipmapped:NO];
  d.usage = MTLTextureUsageShaderRead;
  d.storageMode = MTLStorageModeShared;
  id<MTLTexture> t0 = [gDevice newTextureWithDescriptor:d], t1 = [gDevice newTextureWithDescriptor:d];
  std::vector<float> a((size_t)W * 4, 0.0f), b((size_t)W * 4, 0.0f);
  for (uint32_t x = 0; x < W; ++x) b[x * 4] = dy[x];  // Y only: the same delta on R, G and B
  [t0 replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0 withBytes:a.data() bytesPerRow:W * 16];
  [t1 replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0 withBytes:b.data() bytesPerRow:W * 16];
  struct {
    uint32_t srcStride, dstStride, mode;
    float gain;
  } u = {in.rowBytes / 2u, out.rowBytes / 2u, 0, 4.0f};
  id<MTLCommandBuffer> cb = [gQueue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso];
  [enc setBuffer:in.buffer offset:0 atIndex:0];
  [enc setBuffer:out.buffer offset:0 atIndex:1];
  [enc setBytes:&u length:sizeof(u) atIndex:2];
  [enc setTexture:t0 atIndex:0];
  [enc setTexture:t1 atIndex:1];
  [enc dispatchThreads:MTLSizeMake(W, H, 1) threadsPerThreadgroup:od::Context::threadgroupFor(pso)];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  for (uint32_t x = 0; x < W; ++x)
    for (uint32_t c = 0; c < 3; ++c) {
      const float got = pixelAt(out, x, 0, c);
      CHECK(std::isfinite(got) && fabsf(got - want[x]) <= fabsf(want[x]) * 1e-3f,
            "pixel %u channel %u: source %g + delta %g gave %g, expected %g", x, c, src[x], dy[x], got, want[x]);
    }
}

static void testNoiseView() {
  gTest = "noise_view";
  ODSession* s = od_session_create(gCtx);
  Buf in = makeBuf(40, 23, OD_PIXEL_BGRA32F, 0), out = makeBuf(40, 23, OD_PIXEL_BGRA32F, 0);
  fillBuf(in, 2);
  ODParams p = syncParams();
  p.debugSkipStages = kSkipMiddle;
  p.view = OD_VIEW_NOISE;
  ODFrame fi = in.frame(), fo = out.frame();
  int off = 0;
  CHECK(od_render(s, &fi, &off, 1, &p, &fo, nullptr) == 0, "render failed: %s", od_last_error(s));
  size_t bad = 0;
  for (uint32_t y = 0; y < in.height; ++y)
    for (uint32_t x = 0; x < in.width; ++x) {
      for (int c = 0; c < 3; ++c) bad += pixelAt(out, x, y, c) != 0.5f;
      bad += pixelAt(out, x, y, 3) != pixelAt(in, x, y, 3);
    }
  CHECK(bad == 0, "noise view of an untouched frame should be 0.5 grey with source alpha (%zu wrong)", bad);
  od_session_destroy(s);
}

// Renders a sequence the way a host would and returns every output frame.
static bool renderSequence(ODSession* s, const std::vector<Buf>& src, int radius, bool keyed, bool inPlace,
                           const std::vector<int>& orderOfFrames, std::vector<Buf>* outs, int* totalHits,
                           const ODParams& base) {
  const int n = (int)src.size();
  outs->assign((size_t)n, Buf{});
  for (int idx : orderOfFrames) {
    @autoreleasepool {
      ODFrame frames[OD_MAX_FRAMES];
      int offsets[OD_MAX_FRAMES];
      int count = 0;
      Buf cur = src[(size_t)idx];
      if (inPlace) {
        cur = makeBuf(cur.width, cur.height, cur.format, cur.rowBytes - cur.width * cur.bpp());
        memcpy(cur.bytes(), src[(size_t)idx].bytes(), (size_t)cur.rowBytes * cur.height);
      }
      frames[count] = cur.frame(keyed ? 1000 + idx : OD_NO_KEY);
      offsets[count++] = 0;
      // neighbours in a deliberately odd order: +1, -1, +2, -2, ...
      for (int d = 1; d <= radius; ++d) {
        for (int sgn : {1, -1}) {
          int j = idx + sgn * d;
          if (j < 0 || j >= n) continue;  // missing at the ends
          frames[count] = src[(size_t)j].frame(keyed ? 1000 + j : OD_NO_KEY);
          offsets[count++] = sgn * d;
        }
      }
      Buf out = inPlace ? cur : makeBuf(cur.width, cur.height, cur.format, 0);
      ODFrame fo = out.frame();
      ODStats st;
      if (od_render(s, frames, offsets, count, &base, &fo, &st) != 0) {
        CHECK(false, "frame %d: render failed: %s", idx, od_last_error(s));
        return false;
      }
      if (totalHits) *totalHits += st.cacheHits;
      // With a profile and the stages on, a render must change the frame: a
      // pass-through would satisfy every comparison between two renders.
      if (base.manualProfile && !base.debugSkipStages) {
        CHECK(st.profile.valid == 1, "frame %d: no profile", idx);
        CHECK(diffPixels(src[(size_t)idx], out) > 0, "frame %d: the render left the frame unchanged", idx);
      }
      (*outs)[(size_t)idx] = out;
    }
  }
  return true;
}

static void testCacheAndInPlace() {
  gTest = "cache_inplace";
  const int N = 7, R = 2;
  for (ODPixelFormat fmt : {OD_PIXEL_BGRA32F, OD_PIXEL_BGRA16F}) {
    std::vector<Buf> src;
    for (int i = 0; i < N; ++i) {
      src.push_back(makeBuf(301, 173, fmt, 16));
      fillBuf(src.back(), 100 + i, false);
    }
    std::vector<int> seq, shuffled = {3, 0, 6, 2, 5, 1, 4, 3, 3, 0};
    for (int i = 0; i < N; ++i) seq.push_back(i);
    const ODNoiseProfile prof = whiteProfile(0.03f);
    ODParams p = syncParams();
    p.temporalRadius = R;
    // A fixed profile, so the stages run whatever the automatic profile makes
    // of this content; renderSequence checks that every render changed the
    // frame, so the comparisons below are not between two pass-throughs.
    p.manualProfile = &prof;

    ODSession* a = od_session_create(gCtx);
    ODSession* b = od_session_create(gCtx);
    ODSession* c = od_session_create(gCtx);
    std::vector<Buf> oa, ob, oc;
    int hitsA = 0, hitsB = 0;
    bool ok = renderSequence(a, src, R, true, false, seq, &oa, &hitsA, p) &&
              renderSequence(b, src, R, false, false, seq, &ob, &hitsB, p) &&
              renderSequence(c, src, R, true, true, seq, &oc, nullptr, p);
    if (ok) {
      size_t dCache = 0, dInPlace = 0;
      for (int i = 0; i < N; ++i) {
        dCache += diffPixels(oa[(size_t)i], ob[(size_t)i]);
        dInPlace += diffPixels(oa[(size_t)i], oc[(size_t)i]);
      }
      CHECK(dCache == 0, "%s: cached and uncached renders differ in %zu pixels", fmtName(fmt), dCache);
      CHECK(dInPlace == 0, "%s: in-place and out-of-place renders differ in %zu pixels", fmtName(fmt), dInPlace);
      CHECK(hitsA > 0, "%s: keyed sequence had no cache hits", fmtName(fmt));
      CHECK(hitsB == 0, "%s: unkeyed sequence reported %d cache hits", fmtName(fmt), hitsB);
    }
    // Random access on the warm session must give the same frames as a cold,
    // uncached session rendering in the same order.
    od_session_reset(a);
    od_session_reset(b);
    std::vector<Buf> ra, rb;
    if (renderSequence(a, src, R, true, false, shuffled, &ra, nullptr, p) &&
        renderSequence(b, src, R, false, false, shuffled, &rb, nullptr, p)) {
      size_t d = 0;
      for (int i = 0; i < N; ++i) d += diffPixels(ra[(size_t)i], rb[(size_t)i]);
      CHECK(d == 0, "%s: random-access cached vs uncached differ in %zu pixels", fmtName(fmt), d);
    }
    // With the middle stages off the in-place result is the input itself.
    ODParams ps = p;
    ps.debugSkipStages = kSkipMiddle;
    std::vector<Buf> oi;
    if (renderSequence(c, src, R, true, true, seq, &oi, nullptr, ps)) {
      size_t d = 0;
      for (int i = 0; i < N; ++i) d += diffPixels(src[(size_t)i], oi[(size_t)i]) + dirtyPadding(oi[(size_t)i]);
      CHECK(d == 0, "%s: in-place pass-through changed %zu pixels/padding bytes", fmtName(fmt), d);
    }
    od_session_destroy(a);
    od_session_destroy(b);
    od_session_destroy(c);
  }
}

static void testMissingNeighbours() {
  gTest = "missing_neighbours";
  ODSession* s = od_session_create(gCtx);
  std::vector<Buf> src;
  for (int i = 0; i < 4; ++i) {
    src.push_back(makeBuf(130, 70, OD_PIXEL_BGRA16F, 0));
    fillBuf(src.back(), 50 + i);
  }
  Buf out = makeBuf(130, 70, OD_PIXEL_BGRA16F, 0);
  ODFrame fo = out.frame();
  const ODNoiseProfile prof = whiteProfile(0.03f);
  ODParams p = syncParams();
  p.temporalRadius = 3;
  p.manualProfile = &prof;
  struct Case {
    std::vector<int> idx, off;
    int used;  // frames that reach the temporal merge
  } cases[] = {
      {{0}, {0}, 1},                       // alone
      {{0, 1, 2, 3}, {0, 1, 2, 3}, 4},     // future only (clip start)
      {{3, 2, 1, 0}, {0, -1, -2, -3}, 4},  // past only (clip end)
      {{1, 3, 0}, {0, 2, -1}, 3},          // gaps, unordered
      {{1, 2, 2, 0}, {0, 1, 1, -1}, 3},    // duplicate offset
      {{0, 1, 2}, {0, 5, -4}, 1},          // all neighbours beyond the radius
  };
  for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ++ci) {
    for (int pass = 0; pass < 2; ++pass) {
      ODFrame frames[OD_MAX_FRAMES];
      int n = (int)cases[ci].idx.size();
      for (int k = 0; k < n; ++k) frames[k] = src[(size_t)cases[ci].idx[(size_t)k]].frame(cases[ci].idx[(size_t)k]);
      ODParams q = p;
      if (pass == 1) q.debugSkipStages = kSkipMiddle;
      ODStats st;
      int rc = od_render(s, frames, cases[ci].off.data(), n, &q, &fo, &st);
      CHECK(rc == 0, "case %zu pass %d: render failed: %s", ci, pass, od_last_error(s));
      const int want = pass == 0 ? cases[ci].used : 1;
      CHECK(st.framesUsed == want, "case %zu pass %d: framesUsed %d, expected %d", ci, pass, st.framesUsed, want);
      if (rc == 0 && pass == 0) {
        const size_t d = diffPixels(src[(size_t)cases[ci].idx[0]], out);
        CHECK(d > 0, "case %zu: the render left the frame unchanged", ci);
      }
      if (rc == 0 && pass == 1) {
        size_t d = diffPixels(src[(size_t)cases[ci].idx[0]], out);
        CHECK(d == 0, "case %zu: pass-through differs in %zu pixels", ci, d);
      }
    }
  }
  od_session_destroy(s);
}

static void testAsync() {
  gTest = "async";
  ODSession* s = od_session_create(gCtx);
  Buf in = makeBuf(640, 360, OD_PIXEL_BGRA32F, 0), out = makeBuf(640, 360, OD_PIXEL_BGRA32F, 0);
  fillBuf(in, 9);
  ODParams p = od_params_default();  // waitUntilCompleted = 0
  p.debugSkipStages = kSkipMiddle;
  ODFrame fi = in.frame(77), fo = out.frame();
  int off = 0;
  ODStats st;
  for (int i = 0; i < 3; ++i) {
    CHECK(od_render(s, &fi, &off, 1, &p, &fo, &st) == 0, "render failed: %s", od_last_error(s));
    waitForQueue();
  }
  CHECK(diffPixels(in, out) == 0, "asynchronous render did not reproduce the input");
  CHECK(st.gpuMs > 0, "gpuMs of the previous completed render should be reported (got %g)", st.gpuMs);
  od_session_destroy(s);
  // Destroying a session with work in flight must be safe.
  s = od_session_create(gCtx);
  od_render(s, &fi, &off, 1, &p, &fo, nullptr);
  od_session_destroy(s);
  waitForQueue();
  CHECK(diffPixels(in, out) == 0, "render in flight at session destroy was lost");
}

static void testErrors() {
  gTest = "errors";
  ODSession* s = od_session_create(gCtx);
  Buf in = makeBuf(32, 16, OD_PIXEL_BGRA32F, 0), out = makeBuf(32, 16, OD_PIXEL_BGRA32F, 0);
  Buf small = makeBuf(16, 16, OD_PIXEL_BGRA32F, 0), half = makeBuf(32, 16, OD_PIXEL_BGRA16F, 0);
  fillBuf(in, 1);
  ODParams p = syncParams();
  ODFrame fi = in.frame(), fo = out.frame();
  int off0 = 0, off1 = 1;
  auto expectFail = [&](int rc, const char* what) {
    CHECK(rc != 0, "%s: expected an error", what);
    CHECK(strlen(od_last_error(s)) > 0, "%s: no error message", what);
  };
  expectFail(od_render(s, nullptr, &off0, 1, &p, &fo, nullptr), "NULL frames");
  expectFail(od_render(s, &fi, &off0, 0, &p, &fo, nullptr), "frameCount 0");
  expectFail(od_render(s, &fi, &off0, OD_MAX_FRAMES + 1, &p, &fo, nullptr), "frameCount too large");
  expectFail(od_render(s, &fi, &off1, 1, &p, &fo, nullptr), "offsets[0] != 0");
  ODFrame fs = small.frame(), fh = half.frame();
  expectFail(od_render(s, &fi, &off0, 1, &p, &fs, nullptr), "size mismatch");
  expectFail(od_render(s, &fi, &off0, 1, &p, &fh, nullptr), "format mismatch");
  ODFrame bad = fi;
  bad.rowBytes = 32 * 16 - 4;
  expectFail(od_render(s, &bad, &off0, 1, &p, &fo, nullptr), "rowBytes too small");
  bad = fi;
  bad.height = 17;
  expectFail(od_render(s, &bad, &off0, 1, &p, &fo, nullptr), "buffer too small");
  bad = fi;
  bad.mtlBuffer = nullptr;
  expectFail(od_render(s, &bad, &off0, 1, &p, &fo, nullptr), "NULL buffer");
  ODFrame two[2] = {fi, fo};
  int offs[2] = {0, 1};
  expectFail(od_render(s, two, offs, 2, &p, &fo, nullptr), "out aliases a neighbour");
  // A good render afterwards works and clears the error.
  CHECK(od_render(s, &fi, &off0, 1, &p, &fo, nullptr) == 0, "render after errors failed: %s", od_last_error(s));
  CHECK(strlen(od_last_error(s)) == 0, "error not cleared after a successful render");
  // Extreme / nonsense params are sanitised, not fatal.
  ODParams wild = syncParams();
  wild.lumaStrength = NAN;
  wild.chromaStrength = 17;
  wild.temporalRadius = 99;
  wild.spatialStrength = -3;
  wild.noiseLevel = -1;
  CHECK(od_render(s, &fi, &off0, 1, &wild, &fo, nullptr) == 0, "wild params failed: %s", od_last_error(s));
  od_session_destroy(s);

  char err[256] = {0};
  CHECK(od_context_create(nullptr, nullptr, err, sizeof(err)) == nullptr && strlen(err) > 0,
        "context_create(NULL) should fail with a message");
  CHECK(strlen(od_version()) > 0, "version string empty");
  ODParams d = od_params_default();
  CHECK(d.lumaStrength == 0.8f && d.chromaStrength == 1.0f && d.temporalRadius == 2 && d.temporalStrength == 1.0f &&
            d.spatialStrength == 0.5f && d.sharpen == 0.0f && d.noiseLevel == 1.0f && d.quality == 1 &&
            d.view == OD_VIEW_OUTPUT && d.waitUntilCompleted == 0 && d.manualProfile == nullptr &&
            d.debugFlags == 0 && d.debugSkipStages == 0,
        "od_params_default does not match opendenoise.h");
}

static void testBounds() {
  gTest = "bounds";
  // 200 distinct keyed frames through one session: the cache stays bounded
  // and the pool stops creating textures once warm.
  ODSession* s = od_session_create(gCtx);
  const int N = 200, R = 2;
  std::vector<Buf> ring;
  for (int i = 0; i < 2 * R + 1; ++i) {
    ring.push_back(makeBuf(256, 144, OD_PIXEL_BGRA16F, 0));
    fillBuf(ring.back(), (uint32_t)i);
  }
  Buf out = makeBuf(256, 144, OD_PIXEL_BGRA16F, 0);
  ODFrame fo = out.frame();
  const ODNoiseProfile prof = whiteProfile(0.03f);
  ODParams p = syncParams();
  p.manualProfile = &prof;
  uint64_t createdAt50 = 0, createdInterior = 0, created = 0, cacheBytes = 0;
  int entries = 0, maxEntries = 0;
  size_t unchanged = 0;
  for (int i = 0; i < N; ++i) {
    ODFrame frames[OD_MAX_FRAMES];
    int offsets[OD_MAX_FRAMES];
    int n = 0;
    for (int d = 0; d <= 2 * R; ++d) {
      int o = d == 0 ? 0 : (d % 2 ? (d + 1) / 2 : -d / 2);
      int j = i + o;
      if (j < 0 || j >= N) continue;
      frames[n] = ring[(size_t)(j % (2 * R + 1))].frame(j);
      offsets[n++] = o;
    }
    if (od_render(s, frames, offsets, n, &p, &fo, nullptr) != 0) {
      CHECK(false, "frame %d failed: %s", i, od_last_error(s));
      break;
    }
    if (i % 25 == 0) unchanged += diffPixels(ring[(size_t)(i % (2 * R + 1))], out) == 0;
    od_debug_cache_info(s, &entries, &cacheBytes, nullptr, nullptr, &created);
    if (entries > maxEntries) maxEntries = entries;
    if (i == 50) createdAt50 = created;
    if (i == N - R - 1) createdInterior = created;
  }
  CHECK(unchanged == 0, "%zu sampled renders left the frame unchanged", unchanged);
  CHECK(maxEntries <= 2 * R + 3, "cache grew to %d entries (playback reuses the last windows: %d)", maxEntries,
        2 * R + 3);
  CHECK(maxEntries >= 2 * R + 1, "cache holds only %d entries, expected at least the window", maxEntries);
  // Steady state on the interior frames; the last R frames have one-sided
  // windows and may need a texture set of their own (stated bound).
  CHECK(createdInterior == createdAt50, "pool kept allocating: %llu textures at frame 50, %llu at frame %d",
        (unsigned long long)createdAt50, (unsigned long long)createdInterior, N - R - 1);
  CHECK(created - createdInterior <= 64, "the one-sided windows at the end created %llu textures",
        (unsigned long long)(created - createdInterior));
  od_session_reset(s);
  od_debug_cache_info(s, &entries, &cacheBytes, nullptr, nullptr, nullptr);
  CHECK(entries == 0 && cacheBytes == 0, "reset left %d entries / %llu bytes", entries, (unsigned long long)cacheBytes);
  od_session_destroy(s);
}

static void testThreads() {
  gTest = "threads";
  // One context, several sessions on several threads.
  const int T = 4;
  __block int failures = 0;
  NSLock* lock = [NSLock new];
  dispatch_group_t g = dispatch_group_create();
  for (int t = 0; t < T; ++t) {
    dispatch_group_async(g, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
      @autoreleasepool {
        ODSession* s = od_session_create(gCtx);
        Buf in = makeBuf(200 + 7 * (uint32_t)t, 100, t % 2 ? OD_PIXEL_BGRA16F : OD_PIXEL_BGRA32F, 0);
        Buf out = makeBuf(in.width, in.height, in.format, 0);
        fillBuf(in, (uint32_t)t);
        ODParams p = syncParams();
        p.debugSkipStages = kSkipMiddle;
        ODFrame fi = in.frame(t), fo = out.frame();
        int off = 0, bad = 0;
        for (int i = 0; i < 25; ++i) {
          if (od_render(s, &fi, &off, 1, &p, &fo, nullptr) != 0 || diffPixels(in, out) != 0) ++bad;
        }
        od_session_destroy(s);
        [lock lock];
        failures += bad;
        [lock unlock];
      }
    });
  }
  dispatch_group_wait(g, DISPATCH_TIME_FOREVER);
  CHECK(failures == 0, "%d renders failed or differed across threads", failures);

  // The same with the stages running and the bands step (a band amount at
  // 50 %, API v6): its temporaries come from the shared pool; every render of
  // a thread gives the same bytes.
  __block int bandFailures = 0;
  for (int t = 0; t < T; ++t) {
    dispatch_group_async(g, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
      @autoreleasepool {
        ODSession* s = od_session_create(gCtx);
        Buf in = makeBuf(200 + 7 * (uint32_t)t, 100, t % 2 ? OD_PIXEL_BGRA16F : OD_PIXEL_BGRA32F, 0);
        Buf out = makeBuf(in.width, in.height, in.format, 0);
        Buf first = makeBuf(in.width, in.height, in.format, 0);
        fillBuf(in, 40 + (uint32_t)t, /*beyondHalf*/ false);
        const ODNoiseProfile prof = whiteProfile(0.03f);
        ODParams p = syncParams();
        p.manualProfile = &prof;
        p.bandAmount[0][0] = 0.5f;
        p.bandAmount[1][2] = 0.5f;
        ODFrame fi = in.frame(), fo = out.frame();
        int off = 0, bad = 0;
        for (int i = 0; i < 25; ++i) {
          if (od_render(s, &fi, &off, 1, &p, &fo, nullptr) != 0) {
            ++bad;
            continue;
          }
          if (i == 0) memcpy(first.bytes(), out.bytes(), (size_t)out.rowBytes * out.height);
          else if (diffPixels(first, out) != 0) ++bad;
        }
        if (diffPixels(in, first) == 0) ++bad;  // it must have denoised
        od_session_destroy(s);
        [lock lock];
        bandFailures += bad;
        [lock unlock];
      }
    });
  }
  dispatch_group_wait(g, DISPATCH_TIME_FOREVER);
  CHECK(bandFailures == 0, "%d renders with the bands step failed or were not repeatable across threads", bandFailures);
}


// ---------------------------------------------------------------------------
// Memory policy (core/src/memory.h), command-buffer failures, caller input.
// ---------------------------------------------------------------------------

// White Gaussian noise of pixel sigma s on R, G and B, as a profile.
static ODNoiseProfile whiteProfile(float s) {
  ODNoiseProfile p;
  memset(&p, 0, sizeof(p));
  const float chan[3] = {0.7497f, 0.6416f, 0.6770f};
  const float band[OD_BANDS] = {0.8908f, 0.2007f, 0.0855f, 0.0412f};
  for (int c = 0; c < 3; ++c) {
    for (int b = 0; b < OD_BANDS; ++b)
      for (int l = 0; l < OD_LEVELS; ++l) p.sigma[c][b][l] = s * chan[c] * band[b];
    for (int l = 0; l < OD_LEVELS; ++l) p.sigmaFull[c][l] = s * chan[c] * 0.9987f;
  }
  p.confidence = 1;
  p.valid = 1;
  return p;
}

// A context of its own with the given memory settings (NULL = default).
static ODContext* makeContext(const char* memoryMB, const char* idleTrimMs, void* queue = nullptr) {
  if (memoryMB) setenv("OD_MEMORY_MB", memoryMB, 1);
  if (idleTrimMs) setenv("OD_IDLE_TRIM_MS", idleTrimMs, 1);
  char err[1024] = {0};
  ODContext* c = od_context_create((__bridge void*)gDevice, queue ? queue : (__bridge void*)gQueue, err, sizeof(err));
  unsetenv("OD_MEMORY_MB");
  unsetenv("OD_IDLE_TRIM_MS");
  if (!c) fprintf(stderr, "od_context_create: %s\n", err);
  return c;
}

static uint64_t deviceBytes() { return (uint64_t)gDevice.currentAllocatedSize; }

// Renders frame `idx` of `src` (keys keyBase + i) with its window of radius R.
static int renderWindow(ODSession* s, const std::vector<Buf>& src, int idx, int R, int64_t keyBase, const ODParams& p,
                        const Buf& out, ODStats* st = nullptr) {
  ODFrame frames[OD_MAX_FRAMES];
  int offsets[OD_MAX_FRAMES];
  int n = 0;
  frames[n] = src[(size_t)idx].frame(keyBase + idx);
  offsets[n++] = 0;
  for (int d = 1; d <= R; ++d) {
    for (int sgn : {-1, 1}) {
      const int j = idx + sgn * d;
      if (j < 0 || j >= (int)src.size()) continue;
      frames[n] = src[(size_t)j].frame(keyBase + j);
      offsets[n++] = sgn * d;
    }
  }
  ODFrame fo = out.frame();
  return od_render(s, frames, offsets, n, &p, &fo, st);
}

static uint64_t cacheBytesOf(ODSession* s, int* entries = nullptr) {
  uint64_t b = 0;
  od_debug_cache_info(s, entries, &b, nullptr, nullptr, nullptr);
  return b;
}

// Many effect instances on one context. (1) The scratch memory of a render is
// one working set for the whole context, not one per session. (2) A byte
// budget (OD_MEMORY_MB) bounds everything the context holds, whatever the
// number of sessions, and evictions under it never change a picture.
static void memoryShared(bool bands) {
  const int N = 8, R = 2;
  const uint32_t w = 640, h = 360;
  std::vector<Buf> src;
  for (int i = 0; i < 2 * R + 1; ++i) {
    src.push_back(makeBuf(w, h, OD_PIXEL_BGRA16F, 0));
    fillBuf(src.back(), 300 + (uint32_t)i);
  }
  std::vector<Buf> outA, outB;
  for (int k = 0; k < N; ++k) {
    outA.push_back(makeBuf(w, h, OD_PIXEL_BGRA16F, 0));
    outB.push_back(makeBuf(w, h, OD_PIXEL_BGRA16F, 0));
  }
  const ODNoiseProfile prof = whiteProfile(0.03f);
  ODParams p = syncParams();
  p.manualProfile = &prof;  // the stages run, and no statistics buffers are made
  p.quality = 0;
  if (bands) p.bandAmount[0][1] = 0.5f;  // the bands step's temporaries go through the pool and the budget
  printf("  %s\n", bands ? "with the bands step (a band amount at 50 %):" : "defaults:");

  // (1) No budget, no trim.
  ODContext* ctx = makeContext("0", "0");
  if (!ctx) return;
  std::vector<ODSession*> ss;
  for (int k = 0; k < N; ++k) ss.push_back(od_session_create(ctx));
  waitForQueue();
  od_debug_memory_drain(gCtx);  // earlier tests' sessions are freed on gCtx's queue
  const uint64_t dev0 = deviceBytes();
  CHECK(renderWindow(ss[0], src, R, R, 1000, p, outA[0]) == 0, "render failed: %s", od_last_error(ss[0]));
  od_debug_memory_drain(ctx);
  const uint64_t one = deviceBytes() - dev0, cache1 = cacheBytesOf(ss[0]);
  const double scratch1 = (double)one - (double)cache1;
  for (int k = 1; k < N; ++k)
    CHECK(renderWindow(ss[(size_t)k], src, R, R, 1000, p, outA[(size_t)k]) == 0, "render failed: %s",
          od_last_error(ss[(size_t)k]));
  od_debug_memory_drain(ctx);
  uint64_t cacheN = 0, allocated = 0;
  for (ODSession* s : ss) cacheN += cacheBytesOf(s);
  od_debug_memory_info(ctx, &allocated, nullptr, nullptr, nullptr);
  const uint64_t all = deviceBytes() - dev0;
  const double scratchN = (double)all - (double)cacheN;
  printf("  %d sessions, one render each at %ux%u: device +%.1f MB (caches %.1f MB, scratch %.1f MB; one session: "
         "cache %.1f MB, scratch %.1f MB)\n",
         N, w, h, all / 1048576.0, cacheN / 1048576.0, scratchN / 1048576.0, cache1 / 1048576.0, scratch1 / 1048576.0);
  CHECK(scratch1 > 0 && scratchN < 1.5 * scratch1,
        "scratch is per session: %.1f MB for %d sessions against %.1f MB for one", scratchN / 1048576.0, N,
        scratch1 / 1048576.0);
  CHECK(fabs((double)allocated - (double)all) < 4.0 * 1048576.0,
        "the context counts %.1f MB, the device %.1f MB", allocated / 1048576.0, all / 1048576.0);
  for (ODSession* s : ss) od_session_destroy(s);
  ss.clear();
  od_debug_memory_drain(ctx);
  CHECK(deviceBytes() - dev0 < 2 * 1048576 + (uint64_t)scratch1 * 2,
        "destroying the sessions left %.1f MB", (deviceBytes() - dev0) / 1048576.0);
  od_context_destroy(ctx);

  // (2) A budget that holds one render's working set and two windows.
  const uint64_t budget = (uint64_t)(scratch1 + 2.0 * (double)cache1) + (1u << 20);
  char mb[32];
  snprintf(mb, sizeof(mb), "%llu", (unsigned long long)(budget >> 20));
  const uint64_t budgetBytes = (budget >> 20) << 20;
  ctx = makeContext(mb, "0");
  if (!ctx) return;
  for (int k = 0; k < N; ++k) ss.push_back(od_session_create(ctx));
  od_debug_memory_drain(gCtx);
  const uint64_t devB = deviceBytes();
  uint64_t worst = 0, worstDevice = 0, reported = 0;
  size_t differ = 0;
  for (int round = 0; round < 2; ++round) {
    for (int k = 0; k < N; ++k) {
      if (renderWindow(ss[(size_t)k], src, R, R, 1000, p, outB[(size_t)k]) != 0) {
        CHECK(false, "budgeted render failed: %s", od_last_error(ss[(size_t)k]));
        continue;
      }
      differ += diffPixels(outA[(size_t)k], outB[(size_t)k]);
      od_debug_memory_info(ctx, &allocated, &reported, nullptr, nullptr);
      worst = std::max(worst, allocated);
      od_debug_memory_drain(ctx);
      worstDevice = std::max(worstDevice, deviceBytes() - devB);
    }
  }
  cacheN = 0;
  for (ODSession* s : ss) cacheN += cacheBytesOf(s);
  printf("  budget %.1f MB: context held at most %.1f MB (device +%.1f MB), caches %.1f MB at the end\n",
         budgetBytes / 1048576.0, worst / 1048576.0, worstDevice / 1048576.0, cacheN / 1048576.0);
  CHECK(reported == budgetBytes, "OD_MEMORY_MB gave a budget of %llu bytes", (unsigned long long)reported);
  CHECK(worst <= budgetBytes + (1u << 20), "the context went to %.1f MB with a budget of %.1f MB", worst / 1048576.0,
        budgetBytes / 1048576.0);
  CHECK(worstDevice <= budgetBytes + 2 * (1u << 20), "the device went to +%.1f MB with a budget of %.1f MB",
        worstDevice / 1048576.0, budgetBytes / 1048576.0);
  CHECK(differ == 0, "renders under the budget differ from unbudgeted ones in %zu pixels", differ);
  for (ODSession* s : ss) od_session_destroy(s);
  od_context_destroy(ctx);
}
static void testMemoryShared() {
  gTest = "memory_shared";
  memoryShared(false);
  memoryShared(true);
}

// Several contexts on one device, each on its own command queue (the plug-in makes one per
// queue Premiere hands it), share one budget. (1) Sequential renders on three contexts under
// one budget: the device's total stays within it, and the pictures are those of unbudgeted
// renders. (2) Six threads render at once on three contexts, blocking (their working sets
// held for the GPU time): renders are admitted while their working sets fit together, so the
// device's total stays within the budget. (3) Overrun policy: a budget below one render's
// working set; renders still run, one at a time, and the overrun is counted. The total is
// sampled on the side through od_debug_memory_info, which the core had before this change,
// and read from the device's ledger.
static void testMemoryDevice() {
  gTest = "memory_device";
  const int C = 3, R = 2, F = 2 * R + 1;
  const uint32_t w = 1920, h = 1080;
  const uint64_t MB = 1ull << 20;
  std::vector<Buf> src;
  for (int i = 0; i < F; ++i) {
    src.push_back(makeBuf(w, h, OD_PIXEL_BGRA16F, 0));
    fillBuf(src.back(), 500 + (uint32_t)i);
  }
  const ODNoiseProfile prof = whiteProfile(0.03f);
  ODParams p = syncParams();
  p.manualProfile = &prof;
  p.temporalRadius = R;
  p.quality = 0;  // keys and the plain frame cache: caches the budget must evict
  // The reference picture, one render's working set and one window's cache,
  // without a budget.
  Buf refOut = makeBuf(w, h, OD_PIXEL_BGRA16F, 0);
  uint64_t held = 0, cache1 = 0;
  {
    ODContext* c = makeContext("0", "0");
    if (!c) return;
    ODSession* s = od_session_create(c);
    CHECK(renderWindow(s, src, R, R, 4000, p, refOut) == 0, "reference render failed: %s", od_last_error(s));
    held = od_debug_session_held(s);
    cache1 = cacheBytesOf(s);
    od_session_destroy(s);
    od_context_destroy(c);
  }
  printf("  %ux%u radius %d: one render holds %.0f MB, its window's cache %.0f MB\n", w, h, R, held / (double)MB,
         cache1 / (double)MB);
  CHECK(held > cache1 && cache1 > 0, "held %llu, cache %llu: the test says nothing", (unsigned long long)held,
        (unsigned long long)cache1);
  std::vector<id<MTLCommandQueue>> queues;
  for (int c = 0; c < C; ++c) queues.push_back([gDevice newCommandQueue]);
  auto makeContexts = [&](uint64_t budget, std::vector<ODContext*>* out) {
    char mb[32];
    snprintf(mb, sizeof(mb), "%llu", (unsigned long long)(budget / MB));
    for (int c = 0; c < C; ++c) out->push_back(makeContext(mb, "0", (__bridge void*)queues[(size_t)c]));
    for (ODContext* c : *out)
      if (!c) return false;
    return true;
  };
  auto total = [&](const std::vector<ODContext*>& ctxs) {
    uint64_t sum = 0;
    for (ODContext* c : ctxs) {
      uint64_t a = 0;
      od_debug_memory_info(c, &a, nullptr, nullptr, nullptr);
      sum += a;
    }
    return sum;
  };

  // (1) Sequential, three sessions per context, every window rendered twice.
  {
    const uint64_t budget = ((held + 2 * cache1) / MB + 1) * MB;
    std::vector<ODContext*> ctxs;
    if (!makeContexts(budget, &ctxs)) return;
    od_debug_device_memory_reset_peaks(ctxs[0]);
    std::vector<ODSession*> ss;
    for (int k = 0; k < 3 * C; ++k) ss.push_back(od_session_create(ctxs[(size_t)k % C]));
    Buf out = makeBuf(w, h, OD_PIXEL_BGRA16F, 0);
    uint64_t worst = 0;
    size_t differ = 0;
    int failed = 0;
    for (int round = 0; round < 2; ++round)
      for (ODSession* s : ss) {
        if (renderWindow(s, src, R, R, 4000, p, out) != 0) ++failed;
        differ += diffPixels(out, refOut);
        worst = std::max(worst, total(ctxs));
      }
    ODDeviceMemory dm = {};
    od_debug_device_memory(ctxs[0], &dm);
    printf("  sequential, %d contexts on their own queues, budget %.0f MB: the contexts held at most %.0f MB in "
           "all (ledger peak %.0f MB, %d contexts on it)\n",
           C, budget / (double)MB, worst / (double)MB, dm.logicalPeakBytes / (double)MB, dm.contexts);
    CHECK(failed == 0, "%d sequential renders failed", failed);
    CHECK(dm.contexts == C && dm.budgetBytes == budget, "the ledger has %d contexts and a budget of %llu", dm.contexts,
          (unsigned long long)dm.budgetBytes);
    CHECK(worst <= budget + MB, "%d contexts held %.0f MB in all with a budget of %.0f MB for the device", C,
          worst / (double)MB, budget / (double)MB);
    CHECK(dm.logicalPeakBytes <= budget + MB, "ledger peak %.0f MB over a budget of %.0f MB",
          dm.logicalPeakBytes / (double)MB, budget / (double)MB);
    CHECK(differ == 0, "renders under one budget across contexts differ from unbudgeted ones in %zu pixels", differ);
    for (ODSession* s : ss) od_session_destroy(s);
    for (ODContext* c : ctxs) od_context_destroy(c);
  }

  // (2) and (3): T threads at once, each on a session of its own, spread over
  // the contexts; a side thread samples the contexts' total.
  auto concurrent = [&](const char* what, uint64_t budget, int T, int renders, bool expectOverrun) {
    std::vector<ODContext*> ctxs;
    if (!makeContexts(budget, &ctxs)) return;
    od_debug_device_memory_reset_peaks(ctxs[0]);
    std::atomic<bool> done{false};
    std::atomic<uint64_t> worst{0};
    std::thread sampler([&] {
      while (!done.load()) {
        const uint64_t t = total(ctxs);
        uint64_t cur = worst.load();
        while (t > cur && !worst.compare_exchange_weak(cur, t)) {
        }
        usleep(100);
      }
    });
    std::atomic<int> failed{0};
    std::atomic<size_t> differ{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < T; ++t) {
      threads.emplace_back([&, t] {
        @autoreleasepool {
          ODSession* s = od_session_create(ctxs[(size_t)t % C]);
          Buf out = makeBuf(w, h, OD_PIXEL_BGRA16F, 0);
          for (int i = 0; i < renders; ++i) {
            if (renderWindow(s, src, R, R, 4000, p, out) != 0) ++failed;
            differ += diffPixels(out, refOut);
          }
          od_session_destroy(s);
        }
      });
    }
    for (std::thread& th : threads) th.join();
    done.store(true);
    sampler.join();
    ODDeviceMemory dm = {};
    od_debug_device_memory(ctxs[0], &dm);
    printf("  %s: %d threads on %d contexts, budget %.0f MB: the contexts held at most %.0f MB in all (ledger peak "
           "%.0f MB, physical peak %.0f MB); %d renders at once at most, %llu waited (%.1f ms in all, %llu at "
           "the deadline), %llu creations beyond the budget (by %.0f MB at most)\n",
           what, T, C, budget / (double)MB, worst.load() / (double)MB, dm.logicalPeakBytes / (double)MB,
           dm.physicalPeakBytes / (double)MB, dm.rendersActiveMax, (unsigned long long)dm.admissionWaits,
           dm.admissionWaitMs, (unsigned long long)dm.admissionsForced, (unsigned long long)dm.overruns,
           dm.overrunPeakBytes / (double)MB);
    CHECK(failed.load() == 0, "%s: %d renders failed", what, failed.load());
    CHECK(differ.load() == 0, "%s: renders differ from the reference in %zu pixels", what, differ.load());
    CHECK(dm.admissionsForced == 0, "%s: %llu renders were admitted at their deadline", what,
          (unsigned long long)dm.admissionsForced);
    if (!expectOverrun) {
      const int fit = (int)(budget / held);
      CHECK(worst.load() <= budget + MB && dm.logicalPeakBytes <= budget + MB,
            "%s: %d concurrent renders held %.0f MB (ledger %.0f MB) on the device with a budget of %.0f MB", what,
            T, worst.load() / (double)MB, dm.logicalPeakBytes / (double)MB, budget / (double)MB);
      CHECK(dm.rendersActiveMax >= 1 && dm.rendersActiveMax <= std::max(1, fit),
            "%s: %d renders in progress at once, %d working sets fit the budget", what, dm.rendersActiveMax, fit);
      // Within the alignment of a texture or two (what makeRoom counts as
      // freed is the pooled size, creations ask the device's size).
      CHECK(dm.overrunPeakBytes <= MB, "%s: %llu creations beyond the budget, by %.1f MB at most", what,
            (unsigned long long)dm.overruns, dm.overrunPeakBytes / (double)MB);
    } else {
      // One render alone needs more than the budget: it runs, alone.
      CHECK(dm.rendersActiveMax == 1, "%s: %d renders in progress at once with a budget below one working set", what,
            dm.rendersActiveMax);
      CHECK(dm.overruns > 0 && dm.overrunPeakBytes > 0, "%s: the overrun was not counted", what);
      CHECK(worst.load() <= held + held / 4 && dm.logicalPeakBytes <= held + held / 4,
            "%s: the device held %.0f MB (ledger %.0f MB), more than one render's %.0f MB", what,
            worst.load() / (double)MB, dm.logicalPeakBytes / (double)MB, held / (double)MB);
    }
    for (ODContext* c : ctxs) od_context_destroy(c);
  };
  concurrent("concurrent", ((5 * held / 2) / MB + 1) * MB, 6, 6, false);
  concurrent("overrun policy", (held / 2 / MB) * MB, 3, 4, true);
}

// The cache keeps what playback reuses: the windows of the last three renders.
// Two interleaved key streams through one session (the two fields of
// interlaced material) keep their hits.
static void testCacheWindow() {
  gTest = "cache_window";
  const int R = 2, F = 16;
  const uint32_t w = 256, h = 144;
  std::vector<Buf> a, b;
  for (int i = 0; i < F; ++i) {
    a.push_back(makeBuf(w, h, OD_PIXEL_BGRA16F, 0));
    fillBuf(a.back(), 400 + (uint32_t)i);
    b.push_back(makeBuf(w, h, OD_PIXEL_BGRA16F, 0));
    fillBuf(b.back(), 600 + (uint32_t)i);
  }
  Buf out = makeBuf(w, h, OD_PIXEL_BGRA16F, 0);
  const ODNoiseProfile prof = whiteProfile(0.03f);
  ODParams p = syncParams();
  p.manualProfile = &prof;
  ODSession* one = od_session_create(gCtx);
  ODSession* two = od_session_create(gCtx);
  int steadyOne = -1, minTwo = 1 << 30, maxEntries = 0;
  for (int i = R; i < F - R; ++i) {
    ODStats st;
    CHECK(renderWindow(one, a, i, R, 1000, p, out, &st) == 0, "render failed: %s", od_last_error(one));
    if (i >= R + 3) steadyOne = st.cacheHits;
    ODStats sa, sb;
    CHECK(renderWindow(two, a, i, R, 1000, p, out, &sa) == 0, "render failed: %s", od_last_error(two));
    CHECK(renderWindow(two, b, i, R, 5000, p, out, &sb) == 0, "render failed: %s", od_last_error(two));
    if (i >= R + 3) minTwo = std::min(minTwo, std::min(sa.cacheHits, sb.cacheHits));
    int e = 0;
    cacheBytesOf(one, &e);
    maxEntries = std::max(maxEntries, e);
  }
  printf("  sequential: %d cache hits per render, at most %d entries; two interleaved streams: at least %d hits\n",
         steadyOne, maxEntries, minTwo);
  CHECK(maxEntries == 2 * R + 3, "sequential playback keeps %d entries, expected %d (the last windows)", maxEntries,
        2 * R + 3);
  CHECK(steadyOne > 2 * R, "sequential playback has only %d hits per render", steadyOne);
  CHECK(minTwo == steadyOne, "interleaved streams lose hits: %d against %d for one stream", minTwo, steadyOne);
  od_session_destroy(one);
  od_session_destroy(two);
}

// The idle trim frees memory on its own queue, never inside another
// session's od_render: a render that comes along when several idle sessions
// are due must not stall (it used to free their textures itself, under the
// session list's lock: tens of milliseconds for a few sessions).
static void testTrimOffRenderThread() {
  gTest = "trim_off_render_thread";
  const int N = 6, R = 2, F = 2 * R + 8;
  const uint32_t w = 1920, h = 1080;
  std::vector<Buf> src;
  for (int i = 0; i < F; ++i) {
    src.push_back(makeBuf(w, h, OD_PIXEL_BGRA16F, 0));
    fillBuf(src.back(), 700 + (uint32_t)i);
  }
  Buf out = makeBuf(w, h, OD_PIXEL_BGRA16F, 0);
  const ODNoiseProfile prof = whiteProfile(0.03f);
  ODParams p = syncParams();
  p.manualProfile = &prof;
  p.quality = 0;
  // A timing check: one retry, so a scheduling hiccup on a busy machine does
  // not fail it, while a trim on the render thread (5 ms and more here, every
  // time) does.
  bool passed = false;
  for (int attempt = 0; attempt < 2 && !passed; ++attempt) {
    ODContext* ctx = makeContext("0", "600");
    if (!ctx) return;
    std::vector<ODSession*> idle;
    for (int k = 0; k < N; ++k) idle.push_back(od_session_create(ctx));
    ODSession* play = od_session_create(ctx);
    CHECK(renderWindow(play, src, R, R, 9000, p, out) == 0, "render failed: %s", od_last_error(play));  // warm
    // Six instances render a few frames each, well within one idle period...
    for (ODSession* s : idle)
      for (int i = R; i < F - R; ++i)
        CHECK(renderWindow(s, src, i, R, 1000, p, out) == 0, "render failed: %s", od_last_error(s));
    uint64_t held = 0;
    od_debug_memory_info(ctx, &held, nullptr, nullptr, nullptr);
    // ...then another one plays asynchronously (encode only) while they come
    // due for the trim.
    ODParams pa = p;
    pa.waitUntilCompleted = 0;
    std::vector<double> ms;
    for (int i = 0; i < 120; ++i) {
      @autoreleasepool {
        const double r0 = CFAbsoluteTimeGetCurrent();
        CHECK(renderWindow(play, src, R + (i % 3), R, 9000, pa, out) == 0, "render failed: %s", od_last_error(play));
        ms.push_back(1000.0 * (CFAbsoluteTimeGetCurrent() - r0));
        usleep(12000);
      }
    }
    waitForQueue();
    od_debug_memory_drain(ctx);
    int trimmed = 0;
    for (ODSession* s : idle) {
      int e = 0;
      cacheBytesOf(s, &e);
      trimmed += e == 0;
    }
    std::sort(ms.begin(), ms.end());
    const double median = ms[ms.size() / 2], worst = ms.back(), limit = 3.0 + 10.0 * median;
    printf("  %d idle sessions (%.0f MB) trimmed while another one played: od_render median %.2f ms, max %.2f ms "
           "(limit %.1f)\n",
           trimmed, held / 1048576.0, median, worst, limit);
    CHECK(trimmed == N, "only %d of %d idle sessions were trimmed", trimmed, N);
    passed = worst < limit;
    for (ODSession* s : idle) od_session_destroy(s);
    od_session_destroy(play);
    od_context_destroy(ctx);
  }
  CHECK(passed, "od_render stalled while idle sessions were trimmed (twice)");
}

// A command queue that gives no command buffer on the Nth call.
@interface FaultyQueue : NSProxy
@property(nonatomic) int failAt;  // 1-based call number that returns nil; 0 = never
@property(nonatomic) int calls;
- (instancetype)initWithQueue:(id<MTLCommandQueue>)queue;
@end

@implementation FaultyQueue {
  id<MTLCommandQueue> _queue;
}
- (instancetype)initWithQueue:(id<MTLCommandQueue>)queue {
  _queue = queue;
  return self;
}
- (id<MTLCommandBuffer>)commandBuffer {
  ++_calls;
  if (_failAt && _calls == _failAt) return nil;
  return [_queue commandBuffer];
}
- (NSMethodSignature*)methodSignatureForSelector:(SEL)sel {
  return [(NSObject*)_queue methodSignatureForSelector:sel];
}
- (void)forwardInvocation:(NSInvocation*)inv {
  [inv invokeWithTarget:_queue];
}
@end

// No command buffer from the queue (device lost): every entry point fails
// with a message, keeps no cache entry it could not fill, and the next render
// is exactly what a fresh session gives.
static void testNoCommandBuffer() {
  gTest = "no_command_buffer";
  const int R = 2;
  const uint32_t w = 320, h = 180;
  std::vector<Buf> src;
  for (int i = 0; i < 2 * R + 2; ++i) {
    src.push_back(makeBuf(w, h, OD_PIXEL_BGRA32F, 0));
    fillBuf(src.back(), 800 + (uint32_t)i);
  }
  Buf out = makeBuf(w, h, OD_PIXEL_BGRA32F, 0), ref = makeBuf(w, h, OD_PIXEL_BGRA32F, 0);
  const ODNoiseProfile prof = whiteProfile(0.03f);
  FaultyQueue* fq = [[FaultyQueue alloc] initWithQueue:gQueue];
  ODContext* ctx = makeContext(nullptr, "0", (__bridge void*)fq);
  if (!ctx) return;
  struct Case {
    const char* name;
    int wait, timing, failAt;  // failAt: which commandBuffer call of the render gives nil
    bool manual;
  } cases[] = {
      {"blocking", 1, 0, 1, true},
      {"asynchronous", 0, 0, 1, true},
      {"timing mode, first buffer", 1, 1, 1, true},
      {"timing mode, second buffer", 1, 1, 2, true},
      {"cold automatic profile, first buffer", 1, 0, 1, false},
      {"cold automatic profile, buffer after the blocking estimate", 0, 0, 2, false},
  };
  for (const Case& c : cases) {
    ODSession* s = od_session_create(ctx);
    ODParams p = syncParams();
    p.waitUntilCompleted = c.wait;
    if (c.timing) p.debugFlags |= OD_DEBUG_STAGE_TIMINGS;
    if (c.manual) p.manualProfile = &prof;
    // A healthy render first, so the failing one finds its neighbours cached.
    fq.failAt = 0;
    if (c.manual) CHECK(renderWindow(s, src, R, R, 1000, p, out) == 0, "%s: healthy render failed", c.name);
    waitForQueue();
    fq.calls = 0;
    fq.failAt = c.failAt;
    memset(out.bytes(), 0x5A, (size_t)out.rowBytes * h);
    const int rc = renderWindow(s, src, R + 1, R, 1000, p, out);
    fq.failAt = 0;
    int entries = -1;
    cacheBytesOf(s, &entries);
    CHECK(rc != 0, "%s: render with no command buffer returned 0", c.name);
    CHECK(strlen(od_last_error(s)) > 0, "%s: no error message", c.name);
    CHECK(entries == 0, "%s: %d cache entries kept after the failure", c.name, entries);
    // The next render on the same session matches a fresh session's.
    ODParams q = p;
    q.waitUntilCompleted = 1;
    q.debugFlags = 0;
    CHECK(renderWindow(s, src, R + 1, R, 1000, q, out) == 0, "%s: render after the failure failed: %s", c.name,
          od_last_error(s));
    ODSession* fresh = od_session_create(gCtx);
    CHECK(renderWindow(fresh, src, R + 1, R, 1000, q, ref) == 0, "%s: fresh render failed", c.name);
    const size_t d = diffPixels(out, ref);
    CHECK(d == 0, "%s: the render after the failure differs from a fresh session's in %zu pixels", c.name, d);
    od_session_destroy(fresh);
    od_session_destroy(s);
  }
  // The other entry points.
  ODSession* s = od_session_create(ctx);
  ODFrame f = src[0].frame(5);
  ODNoiseProfile np;
  std::vector<float> ycc((size_t)w * h * 4);
  fq.calls = 0;
  fq.failAt = 1;
  CHECK(od_profile_frame(s, &f, &np) != 0 && strlen(od_last_error(s)) > 0,
        "od_profile_frame with no command buffer did not fail");
  fq.calls = 0;
  fq.failAt = 1;
  CHECK(od_debug_read_working(s, &f, ycc.data()) != 0, "read-back with no command buffer did not fail");
  fq.failAt = 0;
  od_session_destroy(s);
  od_context_destroy(ctx);
}

// Garbage noise figures from a caller (a corrupt preset): od_render refuses a
// profile with non-finite values, clamps the rest, and never writes a
// non-finite or saturated frame.
static void testGarbageProfile() {
  gTest = "garbage_profile";
  const int R = 2;
  const uint32_t w = 256, h = 144;
  const ODNoiseProfile good = whiteProfile(0.03f);
  auto filled = [](float v) {
    ODNoiseProfile p;
    for (int c = 0; c < 3; ++c) {
      for (int b = 0; b < OD_BANDS; ++b)
        for (int l = 0; l < OD_LEVELS; ++l) p.sigma[c][b][l] = v;
      for (int l = 0; l < OD_LEVELS; ++l) p.sigmaFull[c][l] = v;
    }
    p.confidence = 1;
    p.valid = 1;
    return p;
  };
  struct Case {
    const char* name;
    ODNoiseProfile prof;
    float noiseLevel;
    bool refused;  // expected: od_render returns an error
  };
  std::vector<Case> cases;
  cases.push_back({"all NaN", filled(NAN), 1, true});
  cases.push_back({"all +Inf", filled(INFINITY), 1, true});
  cases.push_back({"all 1e30", filled(1e30f), 1, false});
  cases.push_back({"all 3e38", filled(3e38f), 1, false});
  cases.push_back({"all -1", filled(-1), 1, false});
  Case one = {"one NaN knot", good, 1, true};
  one.prof.sigma[0][0][3] = NAN;
  cases.push_back(one);
  Case inf = {"one Inf knot", good, 1, true};
  inf.prof.sigma[1][2][5] = INFINITY;
  cases.push_back(inf);
  Case full = {"one NaN sigmaFull", good, 1, true};
  full.prof.sigmaFull[0][2] = NAN;
  cases.push_back(full);
  cases.push_back({"noiseLevel 1e25", good, 1e25f, false});
  cases.push_back({"noiseLevel 3e38", good, 3e38f, false});
  cases.push_back({"good profile (control)", good, 1, false});
  for (ODPixelFormat fmt : {OD_PIXEL_BGRA32F, OD_PIXEL_BGRA16F}) {
    // Values within the half range in both formats: 32f values beyond it are
    // a separate case (the temporal stage, not the caller's numbers).
    std::vector<Buf> src;
    for (int i = 0; i < 2 * R + 1; ++i) {
      src.push_back(makeBuf(w, h, fmt, 0));
      const Buf& b = src.back();
      for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
          for (uint32_t c = 0; c < 4; ++c) {
            const float v = testValue(x, y, c, 900 + (uint32_t)i, false);
            uint8_t* row = b.bytes() + (size_t)y * b.rowBytes;
            if (fmt == OD_PIXEL_BGRA32F) ((float*)row)[x * 4 + c] = v;
            else ((__fp16*)row)[x * 4 + c] = (__fp16)v;
          }
    }
    Buf out = makeBuf(w, h, fmt, 0);
    ODSession* s = od_session_create(gCtx);
    for (const Case& c : cases) {
      ODParams p = syncParams();
      p.manualProfile = &c.prof;
      p.noiseLevel = c.noiseLevel;
      memset(out.bytes(), 0, (size_t)out.rowBytes * h);
      ODStats st;
      const int rc = renderWindow(s, src, R, R, 1000, p, out, &st);
      if (c.refused) {
        CHECK(rc != 0 && strlen(od_last_error(s)) > 0, "%s %s: accepted a profile with non-finite values",
              fmtName(fmt), c.name);
        continue;
      }
      CHECK(rc == 0, "%s %s: render failed: %s", fmtName(fmt), c.name, od_last_error(s));
      size_t bad = 0;
      for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
          for (uint32_t ch = 0; ch < 3; ++ch) {
            const float v = pixelAt(out, x, y, ch);
            bad += !std::isfinite(v) || (fmt == OD_PIXEL_BGRA16F && fabsf(v) >= 65504.0f);
          }
      CHECK(bad == 0, "%s %s: %zu colour values are non-finite or saturated", fmtName(fmt), c.name, bad);
      bool reported = std::isfinite(st.profile.sigma[0][0][0]) && st.profile.sigma[0][0][0] >= 0 &&
                      st.profile.sigma[0][0][0] <= 1e6f;
      CHECK(reported, "%s %s: ODStats reports sigma %g, not the sanitised value", fmtName(fmt), c.name,
            st.profile.sigma[0][0][0]);
    }
    od_session_destroy(s);
  }
}

// Values at or beyond the half range (32f super-whites up to 250000, 16f
// values of +-65504) must not turn into NaN or infinity anywhere in the
// pipeline. A difference of two clamped working values is up to twice the half
// maximum: stored in a half texture it became infinity, and the next pyramid
// pass spread NaN over the whole frame (every colour value of a 256x144 frame
// with a manual profile, before the fix).
static void testBeyondHalf() {
  gTest = "beyond_half";
  const int R = 2;
  const uint32_t w = 256, h = 144;
  const ODNoiseProfile prof = whiteProfile(0.03f);
  for (ODPixelFormat fmt : {OD_PIXEL_BGRA32F, OD_PIXEL_BGRA16F}) {
    std::vector<Buf> src;
    for (int i = 0; i < 2 * R + 1; ++i) {
      src.push_back(makeBuf(w, h, fmt, 0));
      fillBuf(src.back(), 900 + (uint32_t)i, true);
      if (fmt == OD_PIXEL_BGRA16F) {
        // The half extremes, on alternate frames so the deltas between frames
        // reach twice the half maximum.
        const float ext = (i % 2) ? 65504.0f : -65504.0f;
        for (uint32_t y = 0; y < h; y += 3)
          for (uint32_t x = (uint32_t)i; x < w; x += 5)
            for (uint32_t c = 0; c < 3; ++c)
              ((__fp16*)(src.back().bytes() + (size_t)y * src.back().rowBytes))[x * 4 + c] = (__fp16)ext;
      }
    }
    Buf out = makeBuf(w, h, fmt, 0);
    struct Mode {
      const char* name;
      bool manual;
      uint32_t skip;
      int quality;
      float sharpen;
    } modes[] = {
        {"auto profile", false, 0, 1, 0},
        {"manual, all stages q1", true, 0, 1, 0},
        {"manual, all stages q0", true, 0, 0, 0},
        {"manual, temporal only", true, 1u << OD_STAGE_SPATIAL, 1, 0},
        {"manual, spatial only", true, 1u << OD_STAGE_TEMPORAL, 1, 0},
        {"manual, sharpen", true, 0, 1, 0.8f},
        {"auto, spatial only", false, 1u << OD_STAGE_TEMPORAL, 1, 0},
    };
    for (const Mode& m : modes) {
      ODSession* s = od_session_create(gCtx);
      ODParams p = syncParams();
      if (m.manual) p.manualProfile = &prof;
      p.debugSkipStages = m.skip;
      p.quality = m.quality;
      p.sharpen = m.sharpen;
      memset(out.bytes(), 0, (size_t)out.rowBytes * h);
      const int rc = renderWindow(s, src, R, R, 5000, p, out);
      CHECK(rc == 0, "%s %s: render failed: %s", fmtName(fmt), m.name, od_last_error(s));
      size_t bad = 0;
      for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
          for (uint32_t ch = 0; ch < 3; ++ch) bad += !std::isfinite(pixelAt(out, x, y, ch));
      CHECK(bad == 0, "%s %s: %zu of %u colour values are not finite", fmtName(fmt), m.name, bad, w * h * 3);
      od_session_destroy(s);
    }
  }
}

// A static scene with noise of its own in every frame (sigma 0.03), so that
// the temporal merge takes the neighbours in: a stale neighbour then shows in
// the picture.
static void fillNoisy(const Buf& b, uint32_t frameSeed, float lift, uint32_t x0 = 0, uint32_t y0 = 0,
                      uint32_t x1 = 0xffffffffu, uint32_t y1 = 0xffffffffu) {
  for (uint32_t y = 0; y < b.height; ++y) {
    uint8_t* row = b.bytes() + (size_t)y * b.rowBytes;
    for (uint32_t x = 0; x < b.width; ++x) {
      const bool in = x >= x0 && x < x1 && y >= y0 && y < y1;
      for (uint32_t c = 0; c < 4; ++c) {
        float v = 1.0f;
        if (c < 3) {
          float g = 0;  // about unit variance: sum of 4 uniforms, centred and scaled
          for (uint32_t k = 0; k < 4; ++k)
            g += (float)(hash32(x * 73856093u ^ y * 19349663u ^ (c * 4 + k) * 83492791u ^ frameSeed * 2654435761u) &
                         0xffff) / 65535.0f;
          g = (g - 2.0f) * 1.7320508f;
          v = 0.4f + 0.2f * sinf(0.05f * x + 0.7f * c) * cosf(0.07f * y) + 0.03f * g + (in ? lift : 0.0f);
        }
        if (b.format == OD_PIXEL_BGRA32F) ((float*)row)[x * 4 + c] = v;
        else ((__fp16*)row)[x * 4 + c] = (__fp16)v;
      }
    }
  }
}

// ODParams.verifyFrameCache (what the plug-in passes): the keys do not
// promise the pixels, so no frame is reused. (1) Unchanged frames give
// exactly the picture of a session that ingests everything afresh, with no
// cache hit. (2) New pixels behind the same keys, a sub-sigma change (+0.25
// sigma) of a whole neighbour, of a 16x16 patch of it or of a 1-pixel line
// of it: the picture equals a fresh session's, also when the render does not
// wait. The control, the plain frame cache (the caller vouches for its keys),
// keeps the stale neighbour and differs. (cache_stale: changes that avoid
// the samples the cache was once checked against.)
static void testVerifiedCache() {
  gTest = "verified_cache";
  const int N = 7, R = 2;
  const uint32_t w = 301, h = 173;
  const ODNoiseProfile prof = whiteProfile(0.03f);
  for (ODPixelFormat fmt : {OD_PIXEL_BGRA32F, OD_PIXEL_BGRA16F}) {
    std::vector<Buf> src;
    for (int i = 0; i < N; ++i) {
      src.push_back(makeBuf(w, h, fmt, 16));
      fillNoisy(src.back(), 300 + (uint32_t)i, 0);
    }
    ODParams fresh = syncParams();
    fresh.temporalRadius = R;
    fresh.manualProfile = &prof;
    fresh.noFrameCache = 1;
    ODParams verify = fresh;
    verify.noFrameCache = 0;
    verify.verifyFrameCache = 1;

    // (1) sequential playback
    ODSession* a = od_session_create(gCtx);
    ODSession* b = od_session_create(gCtx);
    size_t d = 0;
    int hits = 0, verified = 0;
    Buf oa = makeBuf(w, h, fmt, 0), ob = makeBuf(w, h, fmt, 0);
    for (int i = 0; i < N; ++i) {
      ODStats st;
      CHECK(renderWindow(a, src, i, R, 7000, verify, oa, &st) == 0, "%s: render failed: %s", fmtName(fmt),
            od_last_error(a));
      hits += st.cacheHits;
      verified += st.verifiedFrames;
      CHECK(renderWindow(b, src, i, R, 7000, fresh, ob) == 0, "%s: render failed", fmtName(fmt));
      d += diffPixels(oa, ob);
    }
    CHECK(d == 0, "%s: verifying renders and fresh ingest differ in %zu pixels", fmtName(fmt), d);
    CHECK(verified == 0 && hits == 0, "%s: a verifying session reused frames (%d hits, %d verified)", fmtName(fmt),
          hits, verified);
    od_session_destroy(a);
    od_session_destroy(b);

    // (2) the same keys with new pixels behind neighbour +1
    struct Change {
      const char* what;
      uint32_t x0, y0, x1, y1;
      bool wait;
    } changes[] = {
        {"whole frame", 0, 0, w, h, true},
        {"16x16 patch", 141, 77, 157, 93, true},
        // Thin lines: one sample per 8x8 block at a free position let 1-pixel
        // lines of hundreds of pixels through (these two, among others).
        // Stratified positions catch any such line longer than 70 pixels.
        {"1-pixel row, 250 px", 51, 74, 301, 75, true},
        {"1-pixel column, 157 px", 100, 8, 101, 165, true},
        {"whole frame, render not waited for", 0, 0, w, h, false},
    };
    for (const Change& c : changes) {
      for (int mode = 0; mode < 2; ++mode) {  // 0 = verified cache, 1 = plain frame cache (control)
        std::vector<Buf> cur;
        for (int i = 0; i < N; ++i) {
          cur.push_back(makeBuf(w, h, fmt, 16));
          memcpy(cur.back().bytes(), src[(size_t)i].bytes(), (size_t)cur.back().rowBytes * h);
        }
        ODParams p = verify;
        if (mode == 1) p.verifyFrameCache = 0;
        ODSession* s = od_session_create(gCtx);
        Buf o1 = makeBuf(w, h, fmt, 0), o2 = makeBuf(w, h, fmt, 0), ref = makeBuf(w, h, fmt, 0);
        CHECK(renderWindow(s, cur, 3, R, 9000, p, o1) == 0, "%s %s: first render failed", fmtName(fmt), c.what);
        fillNoisy(cur[4], 300 + 4, 0.25f * 0.03f, c.x0, c.y0, c.x1, c.y1);  // same key
        ODParams p2 = p;
        p2.waitUntilCompleted = c.wait ? 1 : 0;
        ODStats st;
        CHECK(renderWindow(s, cur, 3, R, 9000, p2, o2, &st) == 0, "%s %s: render after the change failed",
              fmtName(fmt), c.what);
        if (!c.wait) waitForQueue();
        ODSession* f = od_session_create(gCtx);
        CHECK(renderWindow(f, cur, 3, R, 9000, fresh, ref) == 0, "%s %s: fresh render failed", fmtName(fmt), c.what);
        const size_t diff = diffPixels(o2, ref);
        const size_t moved = diffPixels(o1, o2);
        if (mode == 0) {
          CHECK(diff == 0, "%s %s: %zu pixels differ from a fresh session: a stale neighbour was used", fmtName(fmt),
                c.what, diff);
          CHECK(st.verifiedFrames == 0 && st.cacheHits == 0, "%s %s: %d hits, %d verified: a frame was reused",
                fmtName(fmt), c.what, st.cacheHits, st.verifiedFrames);
          CHECK(moved > 0, "%s %s: the change did not reach the picture (the test says nothing)", fmtName(fmt),
                c.what);
        } else {
          CHECK(diff > 0, "%s %s: the plain frame cache did not reuse the old neighbour (the control says nothing)",
                fmtName(fmt), c.what);
        }
        od_session_destroy(s);
        od_session_destroy(f);
      }
    }
  }
}

// Fixed sample positions used by stale-cache regression fixtures.
static void sampledPosition(uint32_t bx, uint32_t by, uint32_t w, uint32_t h, uint32_t* px, uint32_t* py) {
  auto mix = [](uint32_t v, uint32_t k) {
    uint32_t x = v * k;
    x ^= x >> 15;
    x *= 0x2C1B3C6Du;
    x ^= x >> 12;
    return x;
  };
  static const uint32_t perm[8] = {3u, 7u, 0u, 4u, 6u, 1u, 5u, 2u};
  const uint32_t hy = mix(by, 0x85EBCA77u), hx = mix(bx, 0x9E3779B1u);
  const uint32_t oy = perm[(bx + hy) & 7u] ^ ((hy >> 3) & 7u);
  const uint32_t ox = perm[(by + hx) & 7u] ^ ((hx >> 3) & 7u);
  *px = std::min(bx * 8u + ox, w - 1u);
  *py = std::min(by * 8u + oy, h - 1u);
}

// A changed neighbour must never be reused, even when the change avoids every cache sample.
// Compare verified-cache renders against fresh ingestion after either changing all unsampled
// pixels by 0.25 sigma or changing nine unsampled pixels by 0.04 in RGB.
static void testCacheStale() {
  gTest = "cache_stale";
  const int N = 7, R = 2;
  const uint32_t w = 301, h = 173;
  const float sigma = 0.03f;
  const ODNoiseProfile prof = whiteProfile(sigma);
  for (ODPixelFormat fmt : {OD_PIXEL_BGRA32F, OD_PIXEL_BGRA16F}) {
    std::vector<uint8_t> sampled((size_t)w * h, 0);
    for (uint32_t by = 0; by < (h + 7) / 8; ++by)
      for (uint32_t bx = 0; bx < (w + 7) / 8; ++bx) {
        uint32_t x, y;
        sampledPosition(bx, by, w, h, &x, &y);
        sampled[(size_t)y * w + x] = 1;
      }
    struct Case {
      const char* who;
      bool everywhere;  // true: every unsampled pixel (+0.25 sigma); false: nine of them (+0.04)
    } cases[] = {{"+0.25 sigma except at the samples", true}, {"nine unsampled pixels +0.04", false}};
    for (const Case& c : cases) {
      std::vector<Buf> cur;
      for (int i = 0; i < N; ++i) {
        cur.push_back(makeBuf(w, h, fmt, 16));
        fillNoisy(cur.back(), 300 + (uint32_t)i, 0);
      }
      ODParams p = syncParams();
      p.temporalRadius = R;
      p.manualProfile = &prof;
      p.noFrameCache = 0;
      p.verifyFrameCache = 1;
      ODParams fresh = p;
      fresh.noFrameCache = 1;
      ODSession* s = od_session_create(gCtx);
      Buf o1 = makeBuf(w, h, fmt, 0), o2 = makeBuf(w, h, fmt, 0), ref = makeBuf(w, h, fmt, 0);
      CHECK(renderWindow(s, cur, 3, R, 9100, p, o1) == 0, "%s %s: warm render failed: %s", fmtName(fmt), c.who,
            od_last_error(s));
      // New pixels behind neighbour +1's key.
      const Buf& nb = cur[4];
      size_t moved = 0;
      int placed = 0;
      for (uint32_t y = 0; y < h; ++y) {
        uint8_t* row = nb.bytes() + (size_t)y * nb.rowBytes;
        for (uint32_t x = 0; x < w; ++x) {
          if (sampled[(size_t)y * w + x]) continue;
          float d = 0;
          if (c.everywhere) {
            d = 0.25f * sigma;
          } else if (placed < 9 && x >= 40 && y >= 40 && (x * 7 + y * 3) % 97 == 0) {
            d = 0.04f;
            ++placed;
          }
          if (d == 0) continue;
          for (uint32_t ch = 0; ch < 3; ++ch) {
            if (fmt == OD_PIXEL_BGRA32F) ((float*)row)[x * 4 + ch] += d;
            else ((__fp16*)row)[x * 4 + ch] = (__fp16)((float)((__fp16*)row)[x * 4 + ch] + d);
          }
          ++moved;
        }
      }
      ODStats st = {};
      CHECK(renderWindow(s, cur, 3, R, 9100, p, o2, &st) == 0, "%s %s: render after the change failed",
            fmtName(fmt), c.who);
      ODSession* f = od_session_create(gCtx);
      CHECK(renderWindow(f, cur, 3, R, 9100, fresh, ref) == 0, "%s %s: fresh render failed", fmtName(fmt), c.who);
      const size_t stale = diffPixels(o2, ref), changed = diffPixels(o1, ref);
      double rms = 0, mx = 0;
      for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
          for (uint32_t ch = 0; ch < 3; ++ch) {
            const double d = (double)pixelAt(o2, x, y, ch) - (double)pixelAt(ref, x, y, ch);
            rms += d * d;
            mx = std::max(mx, fabs(d));
          }
      rms = sqrt(rms / (3.0 * w * h));
      printf("  %s %s: %zu neighbour pixels moved; cached vs fresh %zu pixels differ (rms %.6f, max %.6f); fresh "
             "vs before the change %zu\n",
             fmtName(fmt), c.who, moved, stale, rms, mx, changed);
      CHECK(changed > 0, "%s %s: the change did not reach a fresh render's picture (the test says nothing)",
            fmtName(fmt), c.who);
      CHECK(stale == 0, "%s %s: %zu pixels differ from a fresh session: a stale neighbour was used", fmtName(fmt),
            c.who, stale);
      od_session_destroy(s);
      od_session_destroy(f);
    }
  }
}

// ODParams.verifyFrameCache against noFrameCache: the same pixels, and when the picture is
// the same. The render order of a scrub: frames 2 to 16 forward, then back to 4, 5, 6, whose
// entries the cache evicted long before (kKeepRenders). A key that comes back after its
// entry went is measured again by the automatic profile under verifyFrameCache (nothing can
// tell whether the host still hands over the pixels it learnt from), while noFrameCache,
// which keeps no entries, still counts it among its last 64 measured keys: from the render
// after the return the profiles, and so the pictures, differ a little. Equal at every
// render: with a manual profile, and under the automatic profile while no frame comes back
// after its entry went (forward playback, and steps back within the window).
static void testVerifiedRevisit() {
  gTest = "verified_revisit";
  const int N = 20, R = 2;
  const uint32_t w = 320, h = 180;
  const ODNoiseProfile prof = whiteProfile(0.03f);
  std::vector<Buf> src;
  for (int i = 0; i < N; ++i) {
    src.push_back(makeBuf(w, h, OD_PIXEL_BGRA32F, 0));
    fillNoisy(src.back(), 700 + (uint32_t)i, 0);
  }
  std::vector<int> scrub, forward;
  for (int i = 2; i <= 16; ++i) scrub.push_back(i);
  const int back = (int)scrub.size();  // the render that returns to frame 4
  for (int i : {4, 5, 6, 7, 8}) scrub.push_back(i);
  for (int i = 2; i <= 12; ++i) forward.push_back(i);
  for (int i : {11, 10, 12, 13, 14, 13, 15}) forward.push_back(i);  // back within the window: entries still there
  // First render at which the two sessions differ (-1 = none), the largest
  // difference at a pixel and the largest rms over a render, in 10-bit code
  // values.
  double rmsCodes = 0;
  auto run = [&](const std::vector<int>& order, bool manual, double* maxCodes) {
    ODParams fresh = syncParams();
    fresh.temporalRadius = R;
    if (manual) fresh.manualProfile = &prof;
    fresh.noFrameCache = 1;
    ODParams verify = fresh;
    verify.noFrameCache = 0;
    verify.verifyFrameCache = 1;
    ODSession* a = od_session_create(gCtx);
    ODSession* b = od_session_create(gCtx);
    Buf oa = makeBuf(w, h, OD_PIXEL_BGRA32F, 0), ob = makeBuf(w, h, OD_PIXEL_BGRA32F, 0);
    int first = -1;
    *maxCodes = 0;
    rmsCodes = 0;
    for (size_t k = 0; k < order.size(); ++k) {
      CHECK(renderWindow(a, src, order[k], R, 4000, verify, oa) == 0, "render failed: %s", od_last_error(a));
      CHECK(renderWindow(b, src, order[k], R, 4000, fresh, ob) == 0, "render failed: %s", od_last_error(b));
      const size_t d = diffPixels(oa, ob);
      if (d == 0) continue;
      if (first < 0) first = (int)k;
      double mx = 0, ss = 0;
      for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
          for (uint32_t c = 0; c < 3; ++c) {
            const double e = 1023.0 * fabs((double)pixelAt(oa, x, y, c) - pixelAt(ob, x, y, c));
            mx = std::max(mx, e);
            ss += e * e;
          }
      *maxCodes = std::max(*maxCodes, mx);
      rmsCodes = std::max(rmsCodes, sqrt(ss / (3.0 * w * h)));
      if (getenv("OD_TEST_VERBOSE"))
        printf("    render %zu (frame %d): %zu pixels differ, max %.2f, rms %.4f 10-bit codes\n", k, order[k], d, mx,
               sqrt(ss / (3.0 * w * h)));
    }
    od_session_destroy(a);
    od_session_destroy(b);
    return first;
  };
  double codes = 0;
  int first = run(scrub, /*manual*/ true, &codes);
  CHECK(first < 0, "manual profile, a scrub back to evicted frames: verifyFrameCache and noFrameCache differ from "
        "render %d (max %.2f 10-bit codes): the pixels used are not those of noFrameCache", first, codes);
  first = run(forward, false, &codes);
  CHECK(first < 0, "automatic profile, forward with steps back inside the window: the pictures differ from render "
        "%d (max %.2f 10-bit codes)", first, codes);
  first = run(scrub, false, &codes);
  printf("  automatic profile, scrub back to frames whose entries went: first difference at render %d (the return is "
         "render %d), up to %.2f 10-bit codes at a pixel, rms up to %.2f\n", first, back, codes, rmsCodes);
  CHECK(first > back, "automatic profile, a scrub back to evicted frames: the first difference is at render %d, "
        "expected after the return (render %d): %s", first, back,
        first < 0 ? "none at all; if noFrameCache now forgets keys too, update the verifyFrameCache note in "
                    "opendenoise.h and this test"
                  : "before the frame came back, so something else than the profile's memory of keys differs");
  CHECK(rmsCodes < 3, "automatic profile, a scrub back: rms %.2f 10-bit codes apart (max %.2f; expected the "
        "difference of two profiles, not of two pictures)", rmsCodes, codes);
}

int main(int argc, char** argv) {
  @autoreleasepool {
    gDevice = MTLCreateSystemDefaultDevice();
    if (!gDevice) {
      fprintf(stderr, "no Metal device\n");
      return 2;
    }
    gQueue = [gDevice newCommandQueue];
    char err[2048] = {0};
    gCtx = od_context_create((__bridge void*)gDevice, (__bridge void*)gQueue, err, sizeof(err));
    if (!gCtx) {
      fprintf(stderr, "od_context_create failed: %s\n", err);
      return 2;
    }
    struct {
      const char* name;
      void (*fn)();
    } tests[] = {
        {"fixtures", testFixtures},
        {"compile_all", testCompileAll},       {"round_trip", testRoundTrip},
        {"special_values", testSpecialValues}, {"ingest_values", testIngestValues},
        {"output_delta", testOutputDelta},     {"noise_view", testNoiseView},
        {"output_clamp16", testOutputClamp16},
        {"cache_inplace", testCacheAndInPlace}, {"missing_neighbours", testMissingNeighbours},
        {"async", testAsync},                  {"errors", testErrors},
        {"bounds", testBounds},                {"threads", testThreads},
        {"memory_shared", testMemoryShared},   {"cache_window", testCacheWindow},
        {"memory_device", testMemoryDevice},
        {"trim_off_render_thread", testTrimOffRenderThread},
        {"no_command_buffer", testNoCommandBuffer},
        {"garbage_profile", testGarbageProfile},
        {"beyond_half", testBeyondHalf},
        {"verified_cache", testVerifiedCache},
        {"cache_stale", testCacheStale},
        {"verified_revisit", testVerifiedRevisit},
    };
    for (auto& t : tests) {
      if (argc > 1 && strcmp(argv[1], t.name) != 0) continue;  // one test by name
      int before = gFailures;
      @autoreleasepool {
        t.fn();
      }
      printf("%-24s %s\n", t.name, gFailures == before ? "ok" : "FAILED");
    }
    od_context_destroy(gCtx);
    printf("%d checks, %d failures (device: %s, Denoise %s)\n", gChecks, gFailures, gDevice.name.UTF8String,
           od_version());
    return gFailures ? 1 : 0;
  }
}
