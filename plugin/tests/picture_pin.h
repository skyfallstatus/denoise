// SPDX-License-Identifier: MIT
// Exact picture pins for both plug-in paths using the real core.
// The synthetic scene has a luminance ramp, fine texture and noise varying with brightness.
// Four consecutive frames per case exercise software checkouts and GPU neighbours, including
// both parameter layouts.
// Old-project and newly applied effect cases have independent fixed references in
// picture_pin_ref.h. OD_PIN_RECORD_ONLY is a reference-generation aid; normal runs compare
// every output hash.
#include <unistd.h>

#include <map>
#include <string>
#include <vector>

#ifndef OD_PIN_RECORD_ONLY
static void PinPrepareCPU(CPUCall& c, const PF_EffectWorld& in, A_long t, intptr_t ref,
                          const std::map<int, double>& ui, bool oldProject) {
  PrepareCPU(c, true, in, t, ref, ui, oldProject);
}
static void PinOldProjectGPU() { OldProjectGPU(); }
#else
static void PinPrepareCPU(CPUCall& c, const PF_EffectWorld& in, A_long t, intptr_t ref,
                          const std::map<int, double>& ui, bool) {
  PrepareCPU(c, true, in, t, ref, ui);
}
static void PinOldProjectGPU() {}
#endif

static const int kPinW = 384, kPinH = 216, kPinFrames = 4, kPinFirst = 20;
static const char* const kPinDevice = "Apple M5 Max";

// The scene at frame t, float BGRA, rowFloats floats per row.
static void PinFill(float* px, int w, int h, int rowFloats, int t) {
  Rng r(9100u + (unsigned)t);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      double Y = 0.04 + 0.86 * x / (w - 1);
      if (y < h / 3) Y += ((x / 2 + y / 2) % 2 ? 0.012 : -0.012);  // fine texture, top third
      if (y >= 2 * h / 3) Y += 0.05 * std::sin(x * 0.07 + t * 0.3);  // slow structure, bottom third
      const double sigma = 0.0012 + 0.006 * Y;
      float* p = px + (size_t)y * rowFloats + (size_t)x * 4;
      p[0] = (float)(Y + 0.02 + sigma * r.gauss());  // B
      p[1] = (float)(Y + sigma * r.gauss());         // G
      p[2] = (float)(Y - 0.01 + sigma * r.gauss());  // R
      p[3] = 1.f;
    }
}
static uint64_t PinHash(const void* data, size_t n) {
  uint64_t h = 0xcbf29ce484222325ull;
  const unsigned char* b = (const unsigned char*)data;
  for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 0x100000001b3ull;
  return h;
}

struct PinCase {
  const char* name;
  bool oldProject;
  std::map<int, double> ui;
  bool newerLayout;  // uses a control absent from the legacy layout
};
static std::vector<PinCase> PinCases() {
  // One control of each kind and group of the legacy build, off its default.
  const std::map<int, double> custom = {
      {1, 90},   {2, 85},   {4, 90},   {5, 60},   {6, 15},   {7, 120},  {8, 3},
      {19, 150}, {15, 85},  {25, 130}, {33, 120}, {34, 60},  {37, 150}, {44, 110}};
  std::vector<PinCase> v = {{"old-defaults", true, {}, false},
                            {"old-custom", true, custom, false},
                            {"new-defaults", false, {}, false},
                            {"new-custom", false, custom, false}};
#ifndef OD_PIN_RECORD_ONLY
  v.push_back({"new-whole", false, {{OD_P_GRAIN_MODE, OD_GRAIN_MODE_WHOLE}}, true});
#endif
  return v;
}
// A control's GPU value, typed as Premiere hands it, by disk ID (the
// the legacy build controls and Minimum grain applies to).
static void PinSetGPU(int id, double v) {
  if (id == 3 || id == 8 || id == 9 || id == 68) SetI(id, (int)v);
  else if (id == 20 || id == 43 || id == 50) SetB(id, v != 0);
  else SetF(id, v);
}

// Recorded hashes: case/path/frame. Empty in the recording builds.
struct PinRef {
  const char* key;
  uint64_t hash;
};
#ifndef OD_PIN_RECORD_ONLY
#include "picture_pin_ref.h"  // kPinRefs
#else
static const PinRef kPinRefs[] = {{"", 0}};
#endif
static bool PinRefOf(const std::string& key, uint64_t* h) {
  for (const PinRef& r : kPinRefs)
    if (key == r.key) {
      *h = r.hash;
      return true;
    }
  return false;
}

static void TestPicturePin(EffectMainFn effectMain, PrGPUFilterEntry entry) {
  if (!gReal) return;
  printf("Picture pin (real core, %dx%d, frames %d..%d, radius 2)\n", kPinW, kPinH, kPinFirst,
         kPinFirst + kPinFrames - 1);
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  const bool sameDevice = dev && std::string(dev.name.UTF8String) == kPinDevice;
  std::map<std::string, uint64_t> got;
  const std::vector<PinCase> cases = PinCases();

  // ---- software path ----
  World in = MakeWorld(kPinW, kPinH, 16, 0, false, false, 1);
  World out = MakeWorld(kPinW, kPinH, 16, 0, false, false, 2);
  SetWorldFormat(in.w.data, PrPixelFormat_BGRA_4444_32f);
  SetWorldFormat(out.w.data, PrPixelFormat_BGRA_4444_32f);
  gCPU.failCheckouts = false;
  gCPU.source = {in.w, PrPixelFormat_BGRA_4444_32f, [](A_long t, PF_EffectWorld& w) {
                   PinFill((float*)w.data, w.width, w.height, (int)(w.rowbytes / 4), (int)(t / 1001));
                 }};
  for (size_t k = 0; k < cases.size(); ++k) {
    const PinCase& pc = cases[k];
    for (int f = 0; f < kPinFrames; ++f) {
      const int t = kPinFirst + f;
      PinFill((float*)in.w.data, kPinW, kPinH, kPinW * 4, t);
      CPUCall call;
      PinPrepareCPU(call, in.w, (A_long)t * 1001, (intptr_t)(7100 + k), pc.ui, pc.oldProject);
      call.defs[PfOf(106)].u.pd.value = 1; // historical picture pins always Processing Version 1
      const PF_Err e = effectMain(PF_Cmd_RENDER, &call.in_data, &call.out_data, call.ptrs.data(), &out.w, nullptr);
      CHECK(e == PF_Err_NONE, "pin %s cpu frame %d: err %d", pc.name, t, (int)e);
      got[std::string(pc.name) + "/cpu/" + std::to_string(f)] = PinHash(out.store.data(), out.store.size());
    }
  }
  gCPU.source = {};
  gCPU.neighbourStore.clear();

  // ---- GPU path ----
  gGPU.device = dev;
  gGPU.queue = [dev newCommandQueue];
  GPUEnv g;
  g.util.getSPBasicSuite = GetBasic;
  g.suites.utilFuncs = &g.util;
  PrGPUFilterInfo info = {};
  csSDK_int32 index = 0;
  CHECK(entry(PrSDKGPUFilterInterfaceVersion, &index, kPrTrue, &g.suites, &g.filter, &info) == 0, "pin: GPU startup");
  const int offsets[4] = {-1, 1, -2, 2};  // the order the plug-in asks for them, nearest first
  // Every case with the topic entries counted; the custom ones once more
  // without (the other GPU numbering), under a name of their own ("~nt"),
  // compared with the same recording: the picture does not depend on it.
  std::vector<std::pair<size_t, bool>> runs;
  for (size_t k = 0; k < cases.size(); ++k) runs.push_back({k, true});
#ifndef OD_PIN_RECORD_ONLY
  for (size_t k = 0; k < cases.size(); ++k)
    if (!cases[k].ui.empty() && !cases[k].newerLayout) runs.push_back({k, false});
#endif
  for (size_t run = 0; run < runs.size(); ++run) {
    const size_t k = runs[run].first;
    const bool withTopics = runs[run].second;
    const PinCase& pc = cases[k];
    const std::string name = std::string(pc.name) + (withTopics ? "" : "~nt");
    UseLayout(withTopics, withTopics ? kCountTopics : kCountNoTopics);
    for (const auto& kv : pc.ui) PinSetGPU(kv.first, kv.second);
    if (pc.oldProject) PinOldProjectGPU();
    SetI(106,1); // historical picture pins, unchanged reference hashes
    SetNode(9500 + (long long)run, 1000, 1100, 1000, 1100);  // a session of its own
    PrGPUFilterInstance inst = {};
    if (!Create(g, inst)) {
      CHECK(false, "pin %s: GPU instance declined", name.c_str());
      continue;
    }
    for (int f = 0; f < kPinFrames; ++f) {
      const int t = kPinFirst + f;
      FakePPix cur = MakePPix(PrPixelFormat_GPU_BGRA_4444_32f, kPinW, kPinH, 0, 1);
      FakePPix dst = MakePPix(PrPixelFormat_GPU_BGRA_4444_32f, kPinW, kPinH, 0, 2);
      PinFill((float*)cur.buffer.contents, kPinW, kPinH, kPinW * 4, t);
      FakePPix nb[4];
      PPixHand frames[5] = {(PPixHand)&cur};
      for (int i = 0; i < 4; ++i) {
        nb[i] = MakePPix(PrPixelFormat_GPU_BGRA_4444_32f, kPinW, kPinH, 0, 3);
        PinFill((float*)nb[i].buffer.contents, kPinW, kPinH, kPinW * 4, t + offsets[i]);
        frames[i + 1] = (PPixHand)&nb[i];
      }
      PrGPUFilterRenderParams rp = {};
      rp.inClipTime = (4000 + t) * kTPF;
      rp.inSequenceTime = (1000 + t) * kTPF;
      rp.inRenderTicksPerFrame = kTPF;
      rp.inQuality = kPrRenderQuality_High;
      rp.inDownsampleFactorX = rp.inDownsampleFactorY = 1;
      PPixHand oh = (PPixHand)&dst;
      const prSuiteError e = g.filter.Render(&inst, &rp, frames, 5, &oh);
      WaitForGPU();
      // The automatic profile resolves this frame's estimate on the
      // session's queue after the GPU is done; leave it time, so that the
      // next render reads the same profile on every run.
      usleep(150000);
      CHECK(e == suiteError_NoError, "pin %s gpu frame %d: err 0x%x", name.c_str(), t, (unsigned)e);
      got[name + "/gpu/" + std::to_string(f)] = PinHash(dst.buffer.contents, (size_t)dst.rowBytes * kPinH);
    }
    g.filter.DisposeInstance(&inst);
  }
  index = 0;
  entry(PrSDKGPUFilterInterfaceVersion, &index, kPrFalse, &g.suites, &g.filter, &info);
  UseLayout(true, kCountTopics);

  // ---- report and compare ----
  int same = 0, compared = 0;
  for (const auto& kv : got) {
    printf("  pin %-22s %016llx\n", kv.first.c_str(), (unsigned long long)kv.second);
    std::string key = kv.first;
    // A new effect set to Whole picture (previous) renders a the legacy build
    // project's picture: compared with old-defaults' recording.
    if (key.compare(0, 10, "new-whole/") == 0) key = "old-defaults/" + key.substr(10);
    const size_t nt = key.find("~nt/");
    if (nt != std::string::npos) key.erase(nt, 3);  // the other GPU numbering, the same picture
    uint64_t want = 0;
    if (!sameDevice || !PinRefOf(key, &want)) continue;
    ++compared;
    same += kv.second == want;
    CHECK(kv.second == want, "pin %s: %016llx, recorded %016llx (%s)", kv.first.c_str(),
          (unsigned long long)kv.second, (unsigned long long)want,
          key.compare(0, 4, "old-") == 0 ? "a project saved by 5584635 must render as 5584635 did"
                                         : "a new effect must render the finest-grain reference picture");
  }
#ifndef OD_PIN_RECORD_ONLY
  if (sameDevice)
    CHECK(compared == (int)got.size(), "pin: %d of %zu outputs have a recorded hash", compared, got.size());
#endif
  if (sameDevice)
    printf("  picture pin: %d of %d outputs identical to the recorded builds\n", same, compared);
  else
    printf("  picture pin: recorded on %s, this is %s: hashes printed, not compared\n", kPinDevice,
           dev ? dev.name.UTF8String : "?");
}
