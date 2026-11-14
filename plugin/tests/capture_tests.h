// SPDX-License-Identifier: MIT
// Included by the fake-host harness after its CPU/GPU fixtures.
// All Adobe persistence is exercised through EffectMain, not through internal helpers.
// Parameters are addressed by disk ID: CPU slots through PfOf, GPU slots through
// SetI/SetF/SegIndex, which take disk IDs too.
#include "AE_EffectCB.h"
#include <array>
#include <CommonCrypto/CommonDigest.h>

static std::map<PF_Handle, size_t> gCaptureHandles;
static int gFailCaptureLockAt = 0;
static PF_Handle CaptureNew(A_u_longlong n) {
  if (n > 65536) return nullptr;
  auto h = (PF_Handle)malloc(sizeof(void*));
  if (!h) return nullptr;
  *h = (char*)calloc(1, (size_t)n);
  if (!*h) { free(h); return nullptr; }
  gCaptureHandles[h] = (size_t)n;
  return h;
}
static void* CaptureLock(PF_Handle h) {
  if (gFailCaptureLockAt > 0 && --gFailCaptureLockAt == 0) return nullptr;
  return gCaptureHandles.count(h) ? *h : nullptr;
}
static void CaptureUnlock(PF_Handle) {}
static A_u_longlong CaptureSize(PF_Handle h) { return gCaptureHandles.count(h) ? gCaptureHandles.at(h) : 0; }
static void CaptureDispose(PF_Handle h) {
  CHECK(gCaptureHandles.erase(h) == 1, "arbitrary handle disposed exactly once");
  free(*h); free(h);
}
static PF_UtilCallbacks CaptureUtils() {
  PF_UtilCallbacks u{};
  u.host_new_handle = CaptureNew; u.host_lock_handle = CaptureLock; u.host_unlock_handle = CaptureUnlock;
  u.host_dispose_handle = CaptureDispose; u.host_get_handle_size = CaptureSize;
  return u;
}
static PF_Err Arb(EffectMainFn fn, CPUCall& c, PF_ArbParamsExtra& e) {
  e.id = 55;
  return fn(PF_Cmd_ARBITRARY_CALLBACK, &c.in_data, &c.out_data, nullptr, nullptr, &e);
}
static void DropCapture(EffectMainFn fn, CPUCall& c, PF_Handle h) {
  PF_ArbParamsExtra e{}; e.which_function = PF_Arbitrary_DISPOSE_FUNC; e.u.dispose_func_params.arbH = h;
  CHECK(Arb(fn, c, e) == 0, "dispose capture");
}
struct SavedCapture {
  std::array<uint8_t, 832> bytes{};
  std::array<int32_t, 10> scalars{}; // policy, revision, all eight SHA-256 words
};
static SavedCapture FlattenCapture(EffectMainFn fn, CPUCall& c) {
  SavedCapture s;
  s.scalars[0] = c.defs[PfOf(52)].u.pd.value; s.scalars[1] = c.defs[PfOf(54)].u.sd.value;
  for (int i = 0; i < 8; ++i) s.scalars[2+i] = c.defs[PfOf(56 + i)].u.sd.value;
  PF_ArbParamsExtra e{}; e.which_function = PF_Arbitrary_FLATTEN_FUNC;
  e.u.flatten_func_params.arbH = c.defs[PfOf(55)].u.arb_d.value;
  e.u.flatten_func_params.flat_dataPV = s.bytes.data(); e.u.flatten_func_params.buf_sizeLu = s.bytes.size();
  CHECK(Arb(fn, c, e) == 0, "flatten saved capture");
  return s;
}
static PF_Err RestoreCapture(EffectMainFn fn, CPUCall& c, const SavedCapture& s) {
  PF_ArbParamsExtra e{}; e.which_function = PF_Arbitrary_UNFLATTEN_FUNC;
  e.u.unflatten_func_params.flat_dataPV = s.bytes.data(); e.u.unflatten_func_params.buf_sizeLu = s.bytes.size();
  e.u.unflatten_func_params.arbPH = &c.defs[PfOf(55)].u.arb_d.value;
  PF_Err err = Arb(fn, c, e);
  if (!err) {
    c.defs[PfOf(52)].u.pd.value = s.scalars[0]; c.defs[PfOf(54)].u.sd.value = s.scalars[1];
    for (int i = 0; i < 8; ++i) c.defs[PfOf(56 + i)].u.sd.value = s.scalars[2+i];
  }
  return err;
}
static void CaptureGPUParams(const SavedCapture& s, bool withTopics = true) {
  UseLayout(withTopics, withTopics ? kCountTopics : kCountNoTopics);
  SetI(OD_P_TEMPORAL_RADIUS, 0);
  SetI(52, s.scalars[0]); SetI(54, s.scalars[1]);
  for (int i = 0; i < 8; ++i) SetI(56+i, s.scalars[2+i]);
}
// Model the committed USER_CHANGED_PARAM values directly. In particular this
// does NOT call COPY/FLATTEN/UNFLATTEN to accidentally hydrate the bridge first.
static void CaptureGPUFromUI(const CPUCall& c, bool withTopics, bool float64) {
  UseLayout(withTopics, withTopics ? kCountTopics : kCountNoTopics);
  SetI(OD_P_TEMPORAL_RADIUS, 0);
  SetI(52, c.defs[PfOf(52)].u.pd.value);
  CHECK(SegIndex(54) == (odplug::kUseMinimalPanel ? (withTopics ? 19 : 18) : odplug::kUseDynamicVisibility ? (withTopics ? 21 : 20) : (withTopics ? 15 : 14)), "revision GPU numbering");
  for (int id : {54,56,57,58,59,60,61,62,63}) {
    if (id != 54) CHECK(SegIndex(id) == id-(odplug::kUseMinimalPanel ? 35 : odplug::kUseDynamicVisibility ? 33 : 39)-(withTopics ? 0 : 1), "digest GPU numbering for %d", id);
    const int32_t value = c.defs[PfOf(id)].u.sd.value;
    if (float64) SetF(id, double(value)); else SetI(id, value);
  }
}
using PixelDigest = std::array<uint8_t, CC_SHA256_DIGEST_LENGTH>;
static PixelDigest PixelHash(const void* p, size_t n) {
  PixelDigest d{}; CC_SHA256(p, (CC_LONG)n, d.data()); return d;
}
// This mode starts after exec: no registered capture, sessions or host handles
// inherited. It models renderer/AME process restoration, not Adobe itself.
static int CaptureReopenChild(EffectMainFn fn, PrGPUFilterEntry entry, const char* path) {
  SavedCapture s; PixelDigest expected;
  FILE* f = fopen(path,"rb");
  if (!f) return 2;
  bool valid = fread(&s,1,sizeof s,f)==sizeof s && fread(expected.data(),1,expected.size(),f)==expected.size() && fgetc(f)==EOF;
  fclose(f);
  if (!valid) return 2;
  PF_InData setup{}; PF_OutData setupOut{};
  setup.appl_id = kAppID_Premiere; setup.pica_basicP = &gBasic; setup.inter.add_param = FakeAddParam;
  fn(PF_Cmd_GLOBAL_SETUP,&setup,&setupOut,nullptr,nullptr,nullptr);
  fn(PF_Cmd_PARAMS_SETUP,&setup,&setupOut,nullptr,nullptr,nullptr);
  World in = MakeWorld(320,192,16,0,false,false,1), out = MakeWorld(320,192,16,0,false,false,2);
  FillNoisy((float*)in.w.data,320*192,1234);
  SetWorldFormat(in.w.data,PrPixelFormat_BGRA_4444_32f); SetWorldFormat(out.w.data,PrPixelFormat_BGRA_4444_32f);
  auto utils = CaptureUtils(); CPUCall call;
  PrepareCPU(call,true,in.w,20*1001,880,{{OD_P_TEMPORAL_RADIUS,0}}); call.in_data.utils=&utils;
  gGPU.device=MTLCreateSystemDefaultDevice(); gGPU.queue=[gGPU.device newCommandQueue];
  GPUEnv g; g.util.getSPBasicSuite=GetBasic; g.suites.utilFuncs=&g.util;
  PrGPUFilterInfo info{}; csSDK_int32 index=0;
  CHECK(entry(PrSDKGPUFilterInterfaceVersion,&index,kPrTrue,&g.suites,&g.filter,&info)==0,"fresh-process GPU startup");
  CaptureGPUParams(s); SetNode(9901,1000,1100,1000,1100); PrGPUFilterInstance inst{};
  CHECK(!Create(g,inst),"fresh process really has no saved snapshot before UNFLATTEN");
  CHECK(RestoreCapture(fn,call,s)==0,"fresh-process unflatten");
  call.defs[PfOf(69)].u.sd.value=77; call.defs[PfOf(70)].u.sd.value=88; // stale armed request, no registry
  CHECK(fn(PF_Cmd_RENDER,&call.in_data,&call.out_data,call.ptrs.data(),&out.w,nullptr)==0,"fresh-process CPU");
  CHECK(PixelHash(out.store.data(),out.store.size())==expected,"fresh-process CPU bytes match saved reference");
  // A host may retain the flattened value and dispose its temporary arb
  // handle before asking the GPU to create an instance.
  DropCapture(fn,call,call.defs[PfOf(55)].u.arb_d.value);
  call.defs[PfOf(55)].u.arb_d.value=nullptr;
  for (bool topics : {true,false}) for (bool doubles : {false,true}) {
    CaptureGPUFromUI(call,topics,doubles); SetI(69,77); SetI(70,88); inst={};
    CHECK(Create(g,inst),"fresh-process GPU restored from arbitrary callback");
    if(inst.ioPrivatePluginData) {
      auto input=MakePPix(PrPixelFormat_GPU_BGRA_4444_32f,320,192,0,1);
      auto output=MakePPix(PrPixelFormat_GPU_BGRA_4444_32f,320,192,0,1);
      memcpy(input.buffer.contents,in.w.data,320*192*16);
      PrGPUFilterRenderParams rp{}; rp.inClipTime=4050*kTPF; rp.inSequenceTime=1050*kTPF;
      rp.inRenderTicksPerFrame=kTPF; rp.inQuality=kPrRenderQuality_High;
      rp.inDownsampleFactorX=rp.inDownsampleFactorY=1;
      PPixHand ih=(PPixHand)&input, oh=(PPixHand)&output;
      CHECK(g.filter.Render(&inst,&rp,&ih,1,&oh)==0,"fresh-process GPU render"); WaitForGPU();
      CHECK(PixelHash(output.buffer.contents,320*192*16)==expected,"fresh-process GPU bytes match saved reference");
      g.filter.DisposeInstance(&inst);
    }
  }
  index=0; entry(PrSDKGPUFilterInterfaceVersion,&index,kPrFalse,&g.suites,&g.filter,&info);
  fn(PF_Cmd_GLOBAL_SETDOWN,&setup,&setupOut,nullptr,nullptr,nullptr);
  printf("  capture fresh-process reload: %d checks, %d failed\n",gChecks.load(),gFailures.load());
  return gFailures ? 1 : 0;
}
// Premiere-like UI callbacks reject frame checkouts; normal renders supply
// them. Capture/save must not call checkout at either UI step.
static PF_Err CaptureClick(EffectMainFn fn, CPUCall& c, int id=53) {
  const bool old = gCPU.failCheckouts;
  const int before = gCPU.checkouts;
  gCPU.failCheckouts = true;
  PF_UserChangedParamExtra e{}; e.param_index = PfOf(id);
  const auto result = fn(PF_Cmd_USER_CHANGED_PARAM,&c.in_data,&c.out_data,c.ptrs.data(),nullptr,&e);
  gCPU.failCheckouts = old;
  CHECK(gCPU.checkouts == before,"UI capture/info made no forbidden checkout");
  return result;
}
static PF_Err CaptureCPUWindow(EffectMainFn fn, CPUCall& c, PF_EffectWorld& out) {
  auto err = CaptureClick(fn,c);
  if (err) return err;
  const auto radius = c.defs[PfOf(3)].u.sd.value;
  c.defs[PfOf(3)].u.sd.value = 4;
  err = fn(PF_Cmd_RENDER,&c.in_data,&c.out_data,c.ptrs.data(),&out,nullptr);
  c.defs[PfOf(3)].u.sd.value = radius;
  if (err) return err;
  return CaptureClick(fn,c);
}
#include "reserved_param_tests.h"

static void TestRenderCapturePaths(EffectMainFn fn, GPUEnv& g) {
  const int W=320,H=192;
  auto utils=CaptureUtils();
  World in=MakeWorld(W,H,16,0,false,false,1), out=MakeWorld(W,H,16,0,false,false,2);
  FillNoisy((float*)in.w.data,W*H,1234);
  SetWorldFormat(in.w.data,PrPixelFormat_BGRA_4444_32f); SetWorldFormat(out.w.data,PrPixelFormat_BGRA_4444_32f);
  CPUCall c; PrepareCPU(c,true,in.w,20*1001,8801,{{OD_P_TEMPORAL_RADIUS,0}});
  c.in_data.utils=&utils; c.in_data.field=PF_Field_FRAME;
  c.in_data.pixel_aspect_ratio.num=c.in_data.pixel_aspect_ratio.den=1;
  gCPU.noOwnerInfo=true; // cold reference on both CPU calls
  CHECK(fn(PF_Cmd_RENDER,&c.in_data,&c.out_data,c.ptrs.data(),&out.w,nullptr)==0,"empty Automatic CPU reference");
  auto autoPixels=out.store;
  c.defs[PfOf(52)].u.pd.value=2;
  CHECK(fn(PF_Cmd_RENDER,&c.in_data,&c.out_data,c.ptrs.data(),&out.w,nullptr)==0 && out.store==autoPixels,
        "Use captured without capture is byte-identical Automatic on CPU");
  gCPU.noOwnerInfo=false;
  CHECK(CaptureClick(fn,c,52)==0 && gCaptureUI[std::make_pair(c.in_data.effect_ref,PfOf(64))]=="No capture - Automatic",
        "empty captured policy visibly explains Automatic fallback");
  UseLayout(true,kCountTopics); SetI(3,0);
  SetNode(9010,1000,1100,1000,1100);
  PrGPUFilterRenderParams rp{}; rp.inSequenceTime=1050*kTPF; rp.inClipTime=4050*kTPF;
  rp.inRenderTicksPerFrame=kTPF; rp.inQuality=kPrRenderQuality_High;
  rp.inDownsampleFactorX=rp.inDownsampleFactorY=1;
  auto gi=MakePPix(PrPixelFormat_GPU_BGRA_4444_32f,W,H,0,1);
  auto go=MakePPix(PrPixelFormat_GPU_BGRA_4444_32f,W,H,0,1);
  memcpy(gi.buffer.contents,in.w.data,W*H*16);
  PPixHand ih=(PPixHand)&gi,oh=(PPixHand)&go;
  PixelDigest automatic{};
  for (bool empty : {false,true}) {
    SetI(52,empty?2:1); SetNode(empty?9011:9010,1000,1100,1000,1100);
    PrGPUFilterInstance inst{};
    CHECK(Create(g,inst),"empty Use captured and Automatic both accept GPU");
    if (!inst.ioPrivatePluginData) continue;
    CHECK(g.filter.Render(&inst,&rp,&ih,1,&oh)==0,"empty GPU output is valid"); WaitForGPU();
    auto hash=PixelHash(go.buffer.contents,W*H*16);
    if (!empty) automatic=hash;
    else CHECK(hash==automatic,"empty Use captured GPU bytes equal Automatic");
    g.filter.DisposeInstance(&inst);
  }
  // Radius zero: repeated time, repeated content, then two more distinct
  // frames. The UI host refuses all UI checkouts, including Save.
  CHECK(CaptureClick(fn,c)==0,"arm GPU request");
  UseLayout(true,kCountTopics); SetI(3,0); SetI(52,2);
  SetI(69,c.defs[PfOf(69)].u.sd.value); SetI(70,c.defs[PfOf(70)].u.sd.value);
  SetNode(9012,1000,1100,1000,1100);
  PrGPUFilterInstance inst{};
  CHECK(Create(g,inst),"armed GPU instance accepted with no saved profile");
  if (inst.ioPrivatePluginData) {
    rp.inDownsampleFactorX=rp.inDownsampleFactorY=2;
    CHECK(g.filter.Render(&inst,&rp,&ih,1,&oh)==0,"low-res speculative render still succeeds"); WaitForGPU();
    CHECK(CaptureClick(fn,c,64)==0 && strstr(c.out_data.return_msg,"Await Full"),"low-resolution preview waits for Full");
    rp.inDownsampleFactorX=rp.inDownsampleFactorY=1;
    // Private host buffer: capture must stage pixels on the host queue rather
    // than dereference an inaccessible contents pointer.
    id<MTLBuffer> sharedInput=gi.buffer;
    id<MTLBuffer> privateInput=[gGPU.device newBufferWithLength:sharedInput.length options:MTLResourceStorageModePrivate];
    auto cb=[gGPU.queue commandBuffer]; auto blit=[cb blitCommandEncoder];
    [blit copyFromBuffer:sharedInput sourceOffset:0 toBuffer:privateInput destinationOffset:0 size:sharedInput.length];
    [blit endEncoding]; [cb commit];
    gi.buffer=privateInput;
    // Hold this request inside its measurement. A five-second watchdog
    // always releases the GPU; all threads are joined before this scope ends.
    // A legacy draw event must be ignored even while capture is busy.
    id<MTLSharedEvent> gate=[gGPU.device newSharedEvent];
    id<MTLCommandBuffer> hold=[gGPU.queue commandBuffer];
    CHECK(gate && hold,"create bounded capture UI contention gate");
    if (gate && hold) {
      dispatch_semaphore_t release=dispatch_semaphore_create(0);
      std::atomic<bool> timedOut{false};
      std::thread watchdog([&] {
        if (dispatch_semaphore_wait(release,dispatch_time(DISPATCH_TIME_NOW,5*NSEC_PER_SEC))) timedOut=true;
        gate.signaledValue=1;
      });
      [hold encodeWaitForEvent:gate value:1]; [hold commit];
      const auto logStart=PluginLogSize();
      prSuiteError renderError=suiteError_NoError;
      std::thread render([&] { @autoreleasepool { renderError=g.filter.Render(&inst,&rp,&ih,1,&oh); } });
      bool measuring=false;
      const double deadline=MonoMs()+2000;
      do {
        measuring=PluginLogFrom(logStart).find("Capture sample: render time=")!=std::string::npos;
        if (!measuring) usleep(1000);
      } while (!measuring && MonoMs()<deadline);
      CHECK(measuring,"capture owns its measurement lock before retired UI event");
      PF_Context context{}; context.w_type=PF_Window_EFFECT; auto contextP=&context;
      PF_EventExtra event{}; event.contextH=&contextP; event.e_type=PF_Event_DRAW;
      event.effect_win.index=PfOf(105); event.effect_win.area=PF_EA_CONTROL;
      const auto before=c.defs;
      CHECK(fn(PF_Cmd_EVENT,&c.in_data,&c.out_data,c.ptrs.data(),nullptr,&event)==0 && !event.evt_out_flags &&
            !memcmp(before.data(),c.defs.data(),before.size()*sizeof(PF_ParamDef)),
            "retired row ignores draw event during capture without parameter writes");
      CHECK(gate.signaledValue==0 && !timedOut,"ignored event returns before the capture GPU gate opens");
      dispatch_semaphore_signal(release);
      watchdog.join(); render.join();
      CHECK(renderError==0,"private GPU input capture with retired UI event");
    } else {
      CHECK(g.filter.Render(&inst,&rp,&ih,1,&oh)==0,"private GPU input capture");
    }
    WaitForGPU();
    gi.buffer=sharedInput;
    for (int offset : {0,0,1,2,3}) {
      rp.inSequenceTime=(1050+offset)*kTPF; rp.inClipTime=(4050+offset)*kTPF;
      if (offset>=2) FillNoisy((float*)gi.buffer.contents,W*H,1234+offset);
      CHECK(g.filter.Render(&inst,&rp,&ih,1,&oh)==0,"armed GPU normal render"); WaitForGPU();
      CHECK(CaptureClick(fn,c,64)==0,"refresh GPU progress");
      if (offset==1) CHECK(strstr(c.out_data.return_msg,"1/3"),"pixel duplicate never counts as a second frame");
    }
    CHECK(strstr(c.out_data.return_msg,"Ready"),"three GPU renders produce ready capture");
    CHECK(!c.defs[PfOf(55)].u.arb_d.value && c.defs[PfOf(54)].u.sd.value==0,"GPU never mutates saved payload");
    auto before=c.defs;
    CHECK(fn(PF_Cmd_UPDATE_PARAMS_UI,&c.in_data,&c.out_data,c.ptrs.data(),nullptr,nullptr)==0 &&
          !memcmp(before.data(),c.defs.data(),before.size()*sizeof(PF_ParamDef)),"UPDATE_PARAMS_UI is strictly cosmetic");
    // The UI ref may change when the host reconstructs its UI instance; the
    // parameter token, not that pointer, must reconnect it to the request.
    c.in_data.effect_ref=(PF_ProgPtr)8802;
    CHECK(CaptureClick(fn,c)==0 && c.defs[PfOf(54)].u.sd.value==1 && c.defs[PfOf(55)].u.arb_d.value,
          "GPU request saves after UI instance reconstruction");
    CHECK(strstr(c.out_data.return_msg,"4050,4052,4053"),"GPU saved times are exact, sorted and unique");
    CHECK(!c.defs[PfOf(69)].u.sd.value && !c.defs[PfOf(70)].u.sd.value,"save consumes transient token");
    g.filter.DisposeInstance(&inst);
    // Save -> new instance, with no intervening arbitrary callback. Test the
    // host's two exact scalar representations through independent GPU indices.
    for (bool topics : {true,false}) for (bool doubles : {false,true}) {
      CaptureGPUFromUI(c,topics,doubles);
      PrGPUFilterInstance savedInstance{};
      const auto logStart = PluginLogSize();
      CHECK(Create(g,savedInstance),"USER_CHANGED_PARAM save immediately usable on GPU (topics=%d doubles=%d)",topics,doubles);
      if (!savedInstance.ioPrivatePluginData) continue;
      const auto diagnostic = PluginLogFrom(logStart);
      CHECK(diagnostic.find("result=bridge-hit") != std::string::npos &&
            diagnostic.find(doubles ? "type=Float64(6)" : "type=Int32(3)") != std::string::npos &&
            diagnostic.find("disk=54 pf="+std::to_string(PfOf(54))+" gpu="+std::to_string(SegIndex(54))) != std::string::npos,
            "diagnostics identify successful key, type and GPU mapping");
      CHECK(g.filter.Render(&savedInstance,&rp,&ih,1,&oh)==0,"same-session saved GPU render"); WaitForGPU();
      g.filter.DisposeInstance(&savedInstance);
    }
    // Reject representations that could round/change identity. A valid
    // snapshot in the bridge must never excuse bad scalar transport.
    for (double bad : {1.25, double(NAN), double(INFINITY), -double(INFINITY), 2147483648.0, -2147483649.0}) {
      CaptureGPUFromUI(c,true,true); SetF(56,bad);
      PrGPUFilterInstance invalid{};
      const auto from = PluginLogSize();
      CHECK(!Create(g,invalid),"non-integral/non-finite/out-of-range digest rejected");
      const auto log = PluginLogFrom(from);
      CHECK(log.find("result=invalid-scalar") != std::string::npos &&
            log.find("disk=56 pf="+std::to_string(PfOf(56))+" gpu="+std::to_string(SegIndex(56))+" type=Float64(6)") != std::string::npos &&
            log.find("Capture bridge key:") != std::string::npos,
            "failed scalar logs raw value and present bridge keys");
      if(invalid.ioPrivatePluginData) g.filter.DisposeInstance(&invalid);
    }
    for (bool narrow : {false,true}) {
      CaptureGPUFromUI(c,true,false);
      PrParam bad{};
      if (narrow) { bad.mType=kPrParamType_Float32; bad.mFloat32=float(c.defs[PfOf(56)].u.sd.value); }
      else { bad.mType=kPrParamType_Int64; bad.mInt64=INT64_MAX; }
      gGPU.params[SegIndex(56)]=bad;
      PrGPUFilterInstance invalid{};
      CHECK(!Create(g,invalid),"Float32 and overflowing Int64 identity rejected");
      if(invalid.ioPrivatePluginData) g.filter.DisposeInstance(&invalid);
    }
    CaptureGPUFromUI(c,true,false); gGPU.failParamIndex=SegIndex(54);
    PrGPUFilterInstance invalid{};
    CHECK(!Create(g,invalid),"failed revision GetParam rejects captured GPU creation");
    if(invalid.ioPrivatePluginData) g.filter.DisposeInstance(&invalid);
    gGPU.failParamIndex=-1;
    CaptureGPUFromUI(c,true,false);
    bool negativeWideWord=false;
    for(int id : {54,56,57,58,59,60,61,62,63}) {
      PrParam p{}; p.mType=kPrParamType_Int64; p.mInt64=c.defs[PfOf(id)].u.sd.value;
      negativeWideWord |= p.mInt64 < -16777216;
      gGPU.params[SegIndex(id)]=p;
    }
    CHECK(negativeWideWord,"fixture exercises negative full-width digest words");
    CHECK(Create(g,invalid),"exact Int64 signed words accepted without truncation");
    if(invalid.ioPrivatePluginData) g.filter.DisposeInstance(&invalid);
  }
  if (c.defs[PfOf(55)].u.arb_d.value) {
    const auto saved=FlattenCapture(fn,c);
    // A normal GPU temporal window provides five profiles in one render.
    CHECK(CaptureClick(fn,c)==0,"arm GPU neighbour window");
    CaptureGPUParams(saved); SetI(3,2);
    SetI(69,c.defs[PfOf(69)].u.sd.value); SetI(70,c.defs[PfOf(70)].u.sd.value);
    SetNode(9013,1000,1100,1000,1100);
    inst={}; CHECK(Create(g,inst),"valid saved capture stays GPU eligible while recapturing");
    if (inst.ioPrivatePluginData) {
      std::array<FakePPix,4> ns;
      std::array<PPixHand,5> fs{ih};
      for(int i=0;i<4;++i) {
        ns[i]=MakePPix(PrPixelFormat_GPU_BGRA_4444_32f,W,H,0,1);
        FillNoisy((float*)ns[i].buffer.contents,W*H,4500+i);
        auto privateNeighbour=[gGPU.device newBufferWithLength:ns[i].buffer.length options:MTLResourceStorageModePrivate];
        auto cb=[gGPU.queue commandBuffer]; auto blit=[cb blitCommandEncoder];
        [blit copyFromBuffer:ns[i].buffer sourceOffset:0 toBuffer:privateNeighbour destinationOffset:0 size:ns[i].buffer.length];
        [blit endEncoding]; [cb commit];
        ns[i].buffer=privateNeighbour; fs[i+1]=(PPixHand)&ns[i];
      }
      CHECK(g.filter.Render(&inst,&rp,fs.data(),5,&oh)==0,"normal GPU neighbours measured before in-place filtering"); WaitForGPU();
      // The existing handle is read once, then locked again for the write.
      // Failure on that second lock must keep the original transaction/pin.
      const auto priorScalars = c.defs;
      gFailCaptureLockAt=2;
      CHECK(CaptureClick(fn,c)!=0,"failed Save lock reports an error");
      CHECK(!memcmp(c.defs.data(),priorScalars.data(),c.defs.size()*sizeof(PF_ParamDef)) &&
            !memcmp(*c.defs[PfOf(55)].u.arb_d.value,saved.bytes.data(),saved.bytes.size()),
            "failed Save leaves all parameters and payload unchanged");
      CHECK(CaptureClick(fn,c)==0 && c.defs[PfOf(54)].u.sd.value==2 && strstr(c.out_data.return_msg,"4051,4052,4053,4054,4055"),
            "five GPU neighbours saved as one revision");
      g.filter.DisposeInstance(&inst);
    }
    // Black samples and all failing normal checkouts cannot replace a saved
    // profile. Then repeated identical samples exhaust the nine-frame window.
    auto retained=FlattenCapture(fn,c);
    memset(in.w.data,0,W*H*16);
    gCPU.source={in.w,PrPixelFormat_BGRA_4444_32f,nullptr};
    CHECK(CaptureCPUWindow(fn,c,out.w)==0,"unusable-frame capture reports failure without render error");
    CHECK(FlattenCapture(fn,c).bytes==retained.bytes && strstr(c.out_data.return_msg,"Need 3 distinct"),
          "low-confidence and duplicate samples retain saved profile and visible failure reason");
    DropCapture(fn,c,c.defs[PfOf(55)].u.arb_d.value);
  }
  gCPU.source={};
  // Premiere also delivers half-float, padded, in-place GPU worlds.
  CPUCall half; PrepareCPU(half,true,in.w,20*1001,8803,{{OD_P_TEMPORAL_RADIUS,2}});
  half.in_data.utils=&utils; half.in_data.field=PF_Field_FRAME;
  CHECK(CaptureClick(fn,half)==0,"arm 16f GPU capture");
  UseLayout(true,kCountTopics);
  SetI(69,half.defs[PfOf(69)].u.sd.value); SetI(70,half.defs[PfOf(70)].u.sd.value);
  SetNode(9014,1000,1100,1000,1100);
  inst={}; CHECK(Create(g,inst),"16f capture instance");
  if(inst.ioPrivatePluginData) {
    std::array<FakePPix,5> hs; std::array<PPixHand,5> fs;
    std::vector<float> pixels(W*H*4);
    for(int i=0;i<5;++i) {
      hs[i]=MakePPix(PrPixelFormat_GPU_BGRA_4444_16f,W,H,16,1);
      FillNoisy(pixels.data(),W*H,6000+i);
      for(int y=0;y<H;++y) for(int x=0;x<W*4;++x)
        ((__fp16*)((char*)hs[i].buffer.contents+y*hs[i].rowBytes))[x]=(__fp16)pixels[y*W*4+x];
      fs[i]=(PPixHand)&hs[i];
    }
    CHECK(g.filter.Render(&inst,&rp,fs.data(),5,&fs[0])==0,"16f padded in-place capture render"); WaitForGPU();
    CHECK(CaptureClick(fn,half)==0 && half.defs[PfOf(54)].u.sd.value==1 && half.defs[PfOf(55)].u.arb_d.value,
          "16f current and neighbours saved");
    if(half.defs[PfOf(55)].u.arb_d.value) DropCapture(fn,half,half.defs[PfOf(55)].u.arb_d.value);
    g.filter.DisposeInstance(&inst);
  }
  printf("  capture paths: empty GPU/CPU Automatic parity, GPU accumulation/neighbours, duplicate and preview rejection, cosmetic UI\n");
}
// Exercise the bank's actual hard bound, including a live handle older than
// the bank. Only small statistics blobs are created; no frames or threads.
static void TestCaptureRetention(EffectMainFn fn, GPUEnv& g, CPUCall& call, const SavedCapture& seed) {
  auto revision = [](SavedCapture s, uint32_t rev) {
    auto put = [&](size_t at, uint32_t v) { for(int j=0;j<4;++j) s.bytes[at+j]=uint8_t(v>>(8*j)); };
    put(36,rev);
    uint32_t crc=~0u;
    for(size_t i=0;i<828;++i) {
      crc ^= s.bytes[i];
      for(int j=0;j<8;++j) crc=(crc>>1)^(0xedb88320u&(0u-(crc&1)));
    }
    put(828,~crc);
    uint8_t hash[32]; CC_SHA256(s.bytes.data(),s.bytes.size(),hash);
    s.scalars[1]=int32_t(rev);
    for(int i=0;i<8;++i) memcpy(&s.scalars[2+i],hash+4*i,4); // arm64 host is little-endian
    return s;
  };
  auto pinned = revision(seed,10000), first = revision(seed,10001), last=first;
  CHECK(RestoreCapture(fn,call,pinned)==0,"retention: pin old live handle");
  const auto live=call.defs[PfOf(55)].u.arb_d.value;
  call.defs[PfOf(55)].u.arb_d.value=nullptr;
  for(uint32_t i=0;i<1025;++i) {
    last=revision(seed,10001+i);
    CHECK(RestoreCapture(fn,call,last)==0,"retention: publish bounded fixture");
    if(call.defs[PfOf(55)].u.arb_d.value) DropCapture(fn,call,call.defs[PfOf(55)].u.arb_d.value);
    call.defs[PfOf(55)].u.arb_d.value=nullptr;
  }
  PrGPUFilterInstance inst{};
  CaptureGPUParams(last);
  auto from=PluginLogSize();
  CHECK(Create(g,inst),"retention: newest snapshot survives all temporary handles");
  CHECK(PluginLogFrom(from).find("handle_pins=1 retained=1024 capacity=1024")!=std::string::npos,
        "transport remains at its declared capacity");
  if(inst.ioPrivatePluginData) g.filter.DisposeInstance(&inst);
  inst={}; CaptureGPUParams(first);
  CHECK(!Create(g,inst),"retention: evicted data declines rather than substituting another capture");
  if(inst.ioPrivatePluginData) g.filter.DisposeInstance(&inst);
  inst={}; CaptureGPUParams(pinned);
  CHECK(Create(g,inst),"retention: live handle survives bank eviction");
  if(inst.ioPrivatePluginData) g.filter.DisposeInstance(&inst);
  DropCapture(fn,call,live);
  CHECK(RestoreCapture(fn,call,first)==0,"retention: authoritative unflatten restores evicted value");
  DropCapture(fn,call,call.defs[PfOf(55)].u.arb_d.value); call.defs[PfOf(55)].u.arb_d.value=nullptr;
  inst={}; CaptureGPUParams(first);
  CHECK(Create(g,inst),"retention: restored value GPU eligible again");
  if(inst.ioPrivatePluginData) g.filter.DisposeInstance(&inst);
}
static void TestCapture(EffectMainFn fn, PrGPUFilterEntry entry, const char* self, const char* plugin) {
  if (!gReal) return; // fitting and actual CPU/GPU pixels use the real core
  printf("Persisted capture callbacks and rendering\n");
  const int W = 320, H = 192;
  World in = MakeWorld(W, H, 16, 0, false, false, 1);
  World out = MakeWorld(W, H, 16, 0, false, false, 2);
  FillNoisy((float*)in.w.data, size_t(W)*H, 1234);
  SetWorldFormat(in.w.data, PrPixelFormat_BGRA_4444_32f);
  SetWorldFormat(out.w.data, PrPixelFormat_BGRA_4444_32f);
  PF_UtilCallbacks utils = CaptureUtils();
  CPUCall a;
  PrepareCPU(a, true, in.w, 20*1001, 880, {{OD_P_TEMPORAL_RADIUS, 0}});
  a.in_data.utils = &utils;
  a.in_data.field = PF_Field_FRAME;
  a.in_data.pixel_aspect_ratio.num = a.in_data.pixel_aspect_ratio.den = 1;
  gCPU.source = {in.w, PrPixelFormat_BGRA_4444_32f, [](A_long t, PF_EffectWorld& w) {
    FillNoisy((float*)w.data, size_t(w.width)*w.height, 1234u+uint32_t(t));
  }};
  gCPU.failCheckouts = false;
  ResetCheckoutStats();
  PF_UserChangedParamExtra change{}; change.param_index = PfOf(53);
  CHECK(CaptureClick(fn,a)==0,"arm capture action");
  CHECK(a.defs[PfOf(52)].u.pd.value==1 && a.defs[PfOf(54)].u.sd.value==0 && !a.defs[PfOf(55)].u.arb_d.value,
        "arming leaves saved policy/revision/payload untouched");
  CHECK(gCaptureUI[std::make_pair(a.in_data.effect_ref,PfOf(53))].find("Armed")!=std::string::npos,
        "visible armed label through UI suite");
  CHECK(gCPU.checkouts==0,"Premiere failing UI checkout never attempted");
  a.defs[PfOf(3)].u.sd.value=4;
  CHECK(fn(PF_Cmd_RENDER,&a.in_data,&a.out_data,a.ptrs.data(),&out.w,nullptr)==0,"armed CPU render");
  a.defs[PfOf(3)].u.sd.value=0;
  CHECK(!a.defs[PfOf(55)].u.arb_d.value && a.defs[PfOf(54)].u.sd.value==0,"render never commits saved state");
  CHECK(fn(PF_Cmd_UPDATE_PARAMS_UI,&a.in_data,&a.out_data,a.ptrs.data(),nullptr,nullptr)==0,
        "cosmetic progress refresh");
  CHECK(gCaptureUI[std::make_pair(a.in_data.effect_ref,PfOf(53))].find("Ready")!=std::string::npos,
        "visible ready label");
  CHECK(!a.defs[PfOf(55)].u.arb_d.value,"UI refresh cannot save parameters");
  CHECK(CaptureClick(fn,a)==0,"explicit save action");
  CHECK(a.defs[PfOf(52)].u.pd.value == 2 && a.defs[PfOf(54)].u.sd.value == 1 && a.defs[PfOf(55)].u.arb_d.value,
        "capture commits policy, revision and payload: %s", a.out_data.return_msg);
  CHECK(gCPU.checkouts == 8 && gCPU.checkins == 8, "normal render checkouts/checkins %d/%d", gCPU.checkouts, gCPU.checkins);
  CHECK(gCaptureUI[std::make_pair(a.in_data.effect_ref,PfOf(53))].find("fit")!=std::string::npos &&
        gCaptureUI[std::make_pair(a.in_data.effect_ref,PfOf(64))].find("16")!=std::string::npos,
        "visible saved confidence and frame positions");
  if (!a.defs[PfOf(55)].u.arb_d.value) return;
  CHECK(strstr(a.out_data.return_msg, "16,17,18,19,20,21,22,23,24") && strstr(a.out_data.return_msg, "Confidence"),
        "exact captured frames and confidence shown: %s", a.out_data.return_msg);
  for (int pf : {52,54,55,56,57,58,59,60,61,62,63})
    CHECK(a.defs[PfOf(pf)].uu.change_flags & PF_ChangeFlag_CHANGED_VALUE, "param %d in one undo transaction", pf);
  SavedCapture saved = FlattenCapture(fn, a);
  CHECK(fn(PF_Cmd_RENDER, &a.in_data, &a.out_data, a.ptrs.data(), &out.w, nullptr) == 0, "captured CPU render");
  const auto reference = out.store;
  const auto expected = PixelHash(reference.data(),reference.size());
  const std::string fixturePath = gRecordPath + ".capture.bin";
  FILE* fixture = fopen(fixturePath.c_str(),"wb");
  bool fixtureWritten = fixture && fwrite(&saved,1,sizeof saved,fixture)==sizeof saved &&
      fwrite(expected.data(),1,expected.size(),fixture)==expected.size();
  if(fixture) fclose(fixture);
  CHECK(fixtureWritten,"bounded capture fixture written");

  // Failed and cancelled captures preserve all saved bytes and revision.
  gCPU.failCheckouts = true;
  CHECK(CaptureCPUWindow(fn,a,out.w)==0,"failed checkout capture remains pending");
  CHECK(FlattenCapture(fn, a).bytes == saved.bytes && a.defs[PfOf(54)].u.sd.value == 1, "insufficient evidence keeps capture");
  gCPU.failCheckouts = false;

  a.in_data.inter.abort = [](PF_ProgPtr) -> PF_Err { return PF_Interrupt_CANCEL; };
  CHECK(fn(PF_Cmd_USER_CHANGED_PARAM,&a.in_data,&a.out_data,a.ptrs.data(),nullptr,&change)==PF_Interrupt_CANCEL,
        "capture cancellation propagates");
  a.in_data.inter.abort = nullptr;
  CHECK(FlattenCapture(fn,a).bytes==saved.bytes,"cancel leaves previous capture intact");

  PF_Handle copy = nullptr;
  PF_ArbParamsExtra e{}; e.which_function = PF_Arbitrary_COPY_FUNC;
  e.u.copy_func_params.src_arbH = a.defs[PfOf(55)].u.arb_d.value; e.u.copy_func_params.dst_arbPH = &copy;
  CHECK(Arb(fn, a, e) == 0 && copy && copy != a.defs[PfOf(55)].u.arb_d.value && *copy != *a.defs[PfOf(55)].u.arb_d.value,
        "copy owns an independent handle and allocation");
  e = {}; e.which_function = PF_Arbitrary_COMPARE_FUNC;
  PF_ArbCompareResult cmp = PF_ArbCompare_NOT_EQUAL;
  e.u.compare_func_params = {nullptr, a.defs[PfOf(55)].u.arb_d.value, copy, &cmp};
  CHECK(Arb(fn, a, e) == 0 && cmp == PF_ArbCompare_EQUAL, "canonical compare");
  DropCapture(fn, a, copy);
  // Discrete interpolation, text copy/paste, NEW, FLAT_SIZE and COPY(NULL).
  e = {}; e.which_function = PF_Arbitrary_INTERP_FUNC;
  e.u.interp_func_params = {nullptr,a.defs[PfOf(55)].u.arb_d.value,nullptr,0.75,&copy};
  CHECK(Arb(fn,a,e)==0 && copy,"discrete interpolation copies left before t=1");
  DropCapture(fn,a,copy);
  A_u_long size = 0;
  e = {}; e.which_function = PF_Arbitrary_FLAT_SIZE_FUNC;
  e.u.flat_size_func_params = {nullptr,a.defs[PfOf(55)].u.arb_d.value,&size};
  CHECK(Arb(fn,a,e)==0 && size==832,"bounded flatten size");
  e = {}; e.which_function = PF_Arbitrary_PRINT_SIZE_FUNC;
  e.u.print_size_func_params = {nullptr,a.defs[PfOf(55)].u.arb_d.value,&size};
  CHECK(Arb(fn,a,e)==0 && size==1665,"bounded print size");
  std::array<char,1665> printed{};
  e = {}; e.which_function = PF_Arbitrary_PRINT_FUNC;
  e.u.print_func_params = {nullptr,PF_ArbPrint_NONE,a.defs[PfOf(55)].u.arb_d.value,size,printed.data()};
  CHECK(Arb(fn,a,e)==0,"print capture");
  e = {}; e.which_function = PF_Arbitrary_SCAN_FUNC;
  e.u.scan_func_params = {nullptr,printed.data(),size,&copy};
  CHECK(Arb(fn,a,e)==0 && copy,"scan capture");
  CHECK(!memcmp(*copy,saved.bytes.data(),saved.bytes.size()),"text round trip exact"); DropCapture(fn,a,copy);
  e = {}; e.which_function = PF_Arbitrary_COPY_FUNC;
  e.u.copy_func_params = {nullptr,nullptr,&copy};
  CHECK(Arb(fn,a,e)==0 && copy,"copy null supplies empty default"); DropCapture(fn,a,copy);
  e = {}; e.which_function = PF_Arbitrary_NEW_FUNC; e.u.new_func_params = {nullptr,&copy};
  CHECK(Arb(fn,a,e)==0 && copy,"new default handle"); DropCapture(fn,a,copy);

  // Every serialized byte is covered, including provenance and core CRC.
  int rejected = 0;
  for (size_t i = 0; i < saved.bytes.size(); ++i) {
    auto bad = saved; bad.bytes[i] ^= 1;
    PF_Handle invalid = nullptr;
    e = {}; e.which_function = PF_Arbitrary_UNFLATTEN_FUNC;
    e.u.unflatten_func_params = {nullptr, (A_u_long)bad.bytes.size(), bad.bytes.data(), &invalid};
    if (Arb(fn, a, e) && !invalid) ++rejected;
  }
  CHECK(rejected == 832, "corruption rejected at %d/832 bytes", rejected);
  for (A_u_long n : {0u, 831u, 833u, 65536u, 0xffffffffu}) {
    PF_Handle invalid = nullptr;
    e = {}; e.which_function = PF_Arbitrary_UNFLATTEN_FUNC;
    e.u.unflatten_func_params = {nullptr, n, saved.bytes.data(), &invalid};
    CHECK(Arb(fn, a, e) && !invalid, "bad length %u rejected before reading", n);
  }

  // Host handle lifetime is NOT process lifetime. The later exec test is the
  // genuinely cold case; the same-process handoff must retain the snapshot.
  DropCapture(fn, a, a.defs[PfOf(55)].u.arb_d.value); a.defs[PfOf(55)].u.arb_d.value = nullptr;
  CHECK(gCaptureHandles.empty(), "all capture handles disposed before reload");
  gGPU.device = MTLCreateSystemDefaultDevice(); gGPU.queue = [gGPU.device newCommandQueue];
  GPUEnv g; g.util.getSPBasicSuite = GetBasic; g.suites.utilFuncs = &g.util;
  PrGPUFilterInfo info{}; csSDK_int32 index = 0;
  CHECK(entry(PrSDKGPUFilterInterfaceVersion, &index, kPrTrue, &g.suites, &g.filter, &info) == 0, "capture GPU startup");
  TestRenderCapturePaths(fn,g);
  gCPU.source = {in.w, PrPixelFormat_BGRA_4444_32f, [](A_long t, PF_EffectWorld& w) {
    FillNoisy((float*)w.data, size_t(w.width)*w.height, 1234u+uint32_t(t));
  }};
  SetNode(9901, 1000, 1100, 1000, 1100);
  CaptureGPUParams(saved);
  PrGPUFilterInstance inst{};
  CHECK(Create(g, inst), "new GPU instance after host handle disposal retains saved snapshot");
  if (inst.ioPrivatePluginData) g.filter.DisposeInstance(&inst);
  inst = {};
  CHECK(RestoreCapture(fn, a, saved) == 0, "saved arbitrary data reload");
  CHECK(FlattenCapture(fn, a).bytes == saved.bytes, "save/load bytes identical");
  CHECK(fn(PF_Cmd_RENDER, &a.in_data, &a.out_data, a.ptrs.data(), &out.w, nullptr) == 0 && out.store == reference,
        "reloaded CPU output byte identical");
  CHECK(Create(g, inst), "reloaded capture accepted on GPU");
  if (inst.ioPrivatePluginData) {
    FakePPix gpuIn = MakePPix(PrPixelFormat_GPU_BGRA_4444_32f, W, H, 0, 1);
    FakePPix gpuOut = MakePPix(PrPixelFormat_GPU_BGRA_4444_32f, W, H, 0, 1);
    memcpy(gpuIn.buffer.contents, in.w.data, size_t(W)*H*16);
    PrGPUFilterRenderParams rp{}; rp.inSequenceTime = 1050*kTPF; rp.inClipTime = 4050*kTPF;
    rp.inRenderTicksPerFrame = kTPF; rp.inQuality = kPrRenderQuality_High;
    rp.inDownsampleFactorX = rp.inDownsampleFactorY = 1;
    PPixHand frame = (PPixHand)&gpuIn, output = (PPixHand)&gpuOut;
    CHECK(g.filter.Render(&inst, &rp, &frame, 1, &output) == 0, "captured GPU render");
    WaitForGPU();
    CHECK(!memcmp(gpuOut.buffer.contents, reference.data(), reference.size()), "captured CPU/GPU bytes identical");
    // The other GPU numbering (topic entries not counted): the saved-profile block sits
    // inside the Noise profile topic, so its GPU indices differ between the two; the same
    // capture must be found.
    {
      CaptureGPUParams(saved, /*withTopics*/ false);
      PrGPUFilterInstance nt{};
      CHECK(Create(g, nt), "reloaded capture accepted on GPU without topics");
      if (nt.ioPrivatePluginData) {
        CHECK(g.filter.Render(&nt, &rp, &frame, 1, &output) == 0, "captured GPU render without topics");
        WaitForGPU();
        CHECK(!memcmp(gpuOut.buffer.contents, reference.data(), reference.size()),
              "captured CPU/GPU bytes identical without topics");
        g.filter.DisposeInstance(&nt);
      }
      CaptureGPUParams(saved);
    }
    gGPU.throwInGetParam = true;
    CHECK(g.filter.Render(&inst,&rp,&frame,1,&output)!=0,"captured exception returns an error, never successful bypass");
    gGPU.throwInGetParam = false;
    gpuIn.width = W/2;
    CHECK(g.filter.Render(&inst,&rp,&frame,1,&output)!=0,"captured raster mismatch errors");
    gpuIn.width = W;
    gGPU.failParamIndex = SegIndex(OD_P_GRAIN_FLOOR);
    CHECK(g.filter.Render(&inst,&rp,&frame,1,&output)!=0,"captured render never substitutes default advanced controls");
    gGPU.failParamIndex = -1;
    // Full +/-2 temporal window with identical pixels on both host paths.
    a.defs[PfOf(OD_P_TEMPORAL_RADIUS)].u.sd.value = 2;
    CHECK(fn(PF_Cmd_RENDER,&a.in_data,&a.out_data,a.ptrs.data(),&out.w,nullptr)==0,"captured CPU radius 2");
    SetI(OD_P_TEMPORAL_RADIUS,2);
    std::array<FakePPix,4> neighbours;
    std::array<PPixHand,5> frames{frame};
    const int offsets[4]={-1,1,-2,2};
    for(int i=0;i<4;++i) {
      neighbours[i]=MakePPix(PrPixelFormat_GPU_BGRA_4444_32f,W,H,0,1);
      FillNoisy((float*)neighbours[i].buffer.contents,size_t(W)*H,1234u+uint32_t((20+offsets[i])*1001));
      frames[i+1]=(PPixHand)&neighbours[i];
    }
    CHECK(g.filter.Render(&inst,&rp,frames.data(),5,&output)==0,"captured GPU radius 2"); WaitForGPU();
    CHECK(!memcmp(gpuOut.buffer.contents,out.store.data(),out.store.size()),"captured CPU/GPU radius 2 bytes identical");
    a.defs[PfOf(OD_P_TEMPORAL_RADIUS)].u.sd.value=0; SetI(OD_P_TEMPORAL_RADIUS,0);
    // The GPU pins a value, not a handle. Host disposal cannot invalidate it.
    DropCapture(fn, a, a.defs[PfOf(55)].u.arb_d.value); a.defs[PfOf(55)].u.arb_d.value = nullptr;
    CHECK(g.filter.Render(&inst, &rp, &frame, 1, &output) == 0, "GPU snapshot outlives disposed host handle");
    WaitForGPU();
    CHECK(!memcmp(gpuOut.buffer.contents, reference.data(), reference.size()), "immutable GPU output after disposal");
    SetI(54, saved.scalars[1]+1);
    CHECK(g.filter.Render(&inst, &rp, &frame, 1, &output) != 0, "changed/missing revision errors, never substitutes auto");
    g.filter.DisposeInstance(&inst);
  }
  if (a.defs[PfOf(55)].u.arb_d.value) DropCapture(fn, a, a.defs[PfOf(55)].u.arb_d.value);
  a.defs[PfOf(55)].u.arb_d.value = nullptr;
  CHECK(RestoreCapture(fn, a, saved) == 0, "restore undo source");
  // Second clip deliberately shares Premiere's runtime ID and revision 1.
  CPUCall b; PrepareCPU(b, true, in.w, 20*1001, 881, {{OD_P_TEMPORAL_RADIUS,0}});
  b.in_data.utils = &utils; b.in_data.field = PF_Field_FRAME;
  gCPU.source.gen = [](A_long t, PF_EffectWorld& w) {
    FillNoisy((float*)w.data, size_t(w.width)*w.height, 9000u+uint32_t(t));
    float* p = (float*)w.data;
    for (int i = 0; i < w.width*w.height*4; ++i) if (i%4 != 3) p[i] = 0.5f+(p[i]-0.5f)*0.25f;
  };
  CHECK(CaptureCPUWindow(fn,b,out.w)==0 &&
        b.defs[PfOf(55)].u.arb_d.value, "second clip captures");
  if (b.defs[PfOf(55)].u.arb_d.value) {
    auto second = FlattenCapture(fn,b);
    CHECK(second.scalars[1] == saved.scalars[1] && second.bytes != saved.bytes, "two independent revision-one captures");
    std::vector<char> secondPixels;
    for (auto* call : {&b, &a, &b, &a}) {
      CHECK(fn(PF_Cmd_RENDER, &call->in_data, &call->out_data, call->ptrs.data(), &out.w, nullptr) == 0, "two clip render");
      CHECK((out.store == reference) == (call == &a), "two clips never exchange captured profiles");
      if(call==&b) secondPixels=out.store;
    }
    // Same node/runtime ID/owner/session and revision; only the full digest
    // distinguishes these immutable settings. Repeat in alternating order.
    PrGPUFilterInstance ia{},ib{};
    CaptureGPUParams(saved); bool haveA=Create(g,ia);
    CaptureGPUParams(second); bool haveB=Create(g,ib);
    CHECK(haveA && haveB,"two captures with colliding host identities create independently");
    if(haveA && haveB) {
      auto gi=MakePPix(PrPixelFormat_GPU_BGRA_4444_32f,W,H,0,1);
      auto go=MakePPix(PrPixelFormat_GPU_BGRA_4444_32f,W,H,0,1);
      memcpy(gi.buffer.contents,in.w.data,size_t(W)*H*16);
      PrGPUFilterRenderParams rp{}; rp.inClipTime=4050*kTPF; rp.inSequenceTime=1050*kTPF;
      rp.inRenderTicksPerFrame=kTPF; rp.inQuality=kPrRenderQuality_High;
      rp.inDownsampleFactorX=rp.inDownsampleFactorY=1;
      PPixHand ih=(PPixHand)&gi,oh=(PPixHand)&go;
      for(bool useB : {false,true,false,true}) {
        CaptureGPUParams(useB?second:saved);
        CHECK(g.filter.Render(useB?&ib:&ia,&rp,&ih,1,&oh)==0,"two captured GPU render"); WaitForGPU();
        const auto& pixels=useB?secondPixels:reference;
        CHECK(!memcmp(go.buffer.contents,pixels.data(),pixels.size()),"same-runtime captures stay independent on GPU");
      }
      SetI(56,0); CHECK(!Create(g,inst),"incorrect digest declines GPU creation");
      CaptureGPUParams(saved); SetF(56,1); CHECK(!Create(g,inst),"wrong exact Float64 digest declined");
    }
    if(haveA) g.filter.DisposeInstance(&ia);
    if(haveB) g.filter.DisposeInstance(&ib);
    // Undo/redo is a host restoration of the entire parameter transaction.
    DropCapture(fn,b,b.defs[PfOf(55)].u.arb_d.value); b.defs[PfOf(55)].u.arb_d.value = nullptr;
    CHECK(RestoreCapture(fn,b,saved) == 0, "undo restores all captured state");
    CHECK(fn(PF_Cmd_RENDER,&b.in_data,&b.out_data,b.ptrs.data(),&out.w,nullptr) == 0 && out.store == reference, "undo output");
    DropCapture(fn,b,b.defs[PfOf(55)].u.arb_d.value); b.defs[PfOf(55)].u.arb_d.value = nullptr;
    CHECK(RestoreCapture(fn,b,second) == 0, "redo restores all captured state");
    CHECK(fn(PF_Cmd_RENDER,&b.in_data,&b.out_data,b.ptrs.data(),&out.w,nullptr) == 0 && out.store != reference, "redo output");
    DropCapture(fn,b,b.defs[PfOf(55)].u.arb_d.value);
  }
  DropCapture(fn,a,a.defs[PfOf(55)].u.arb_d.value);
  a.defs[PfOf(55)].u.arb_d.value=nullptr;
  CHECK(gCaptureHandles.empty(), "no arbitrary handles leaked");
  TestCaptureRetention(fn,g,a,saved);
  CHECK(gCaptureHandles.empty(), "retention test leaked no host handles");
  index = 0; entry(PrSDKGPUFilterInterfaceVersion,&index,kPrFalse,&g.suites,&g.filter,&info);
  PF_OutData shutdown{};
  fn(PF_Cmd_GLOBAL_SETDOWN,&a.in_data,&shutdown,nullptr,nullptr,nullptr);
  gCPU.source = {}; gCPU.neighbourStore.clear();
  if(fixtureWritten) {
    std::vector<char*> args={(char*)self,(char*)plugin,(char*)"--real",(char*)"--capture-load",(char*)fixturePath.c_str(),nullptr};
    pid_t pid=0; const int rc=posix_spawn(&pid,self,nullptr,nullptr,args.data(),environ);
    CHECK(rc==0,"fresh-process capture test spawned");
    if(!rc) {
      int status=0; waitpid(pid,&status,0);
      CHECK(WIFEXITED(status) && WEXITSTATUS(status)==0,"fresh-process capture reload exited successfully (status %d)",status);
    }
  }
  printf("  capture: 9 frames; 832/832 corrupt-byte rejections; save/load, immutable copy, undo/redo, CPU/GPU parity\n");
}
