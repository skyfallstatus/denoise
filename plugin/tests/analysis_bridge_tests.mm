// SPDX-License-Identifier: MIT
// Real bridge/core, fake host identities. No Adobe application required.
#include <cstdio>
#include <unistd.h>
#include <thread>
#include <future>
#include "../src/OpenDenoiseShared.mm"
#include "../src/OpenDenoiseAnalysisBridge.mm"
using namespace odplug;
static int failures=0, checks=0;
static void check(bool ok,const char* what) { ++checks; if (!ok) { ++failures; printf("FAIL %s\n",what); } }
static int32_t runtime=0, timeline=7; static A_long start=30;
static bool opaqueEnabled=true, ownerKnown=true, idKnown=true;
static std::map<int32_t,std::pair<OpaqueEffectDataType*,int>> host;
static int suites=0;
static PF_UtilitySuite4 utility = [] { PF_UtilitySuite4 u{};
  u.GetFilterInstanceID=[](PF_ProgPtr,A_long* p)->PF_Err{*p=runtime;return idKnown?0:1;};
  u.GetContainingTimelineID=[](PF_ProgPtr,PrTimelineID* p)->PF_Err{*p=timeline;return ownerKnown?0:1;};
  u.GetTrackItemStart=[](PF_ProgPtr,A_long* p)->PF_Err{*p=start;return ownerKnown?0:1;}; return u; }();
static PrSDKOpaqueEffectDataSuite opaque = {
  [](int32_t id,OpaqueEffectDataType** p)->PF_Err { auto it=host.find(id);*p=it==host.end()?nullptr:it->second.first;if(*p)++it->second.second;return 0; },
  [](int32_t id,OpaqueEffectDataType** p)->PF_Err { auto& e=host[id];if(!e.first)e.first=*p;*p=e.first;++e.second;return 0; },
  [](int32_t id,OpaqueEffectDataType** p)->PF_Err { auto it=host.find(id);if(it==host.end())return 1;*p=nullptr;if(!--it->second.second){*p=it->second.first;host.erase(it);}return 0; }
};
static SPBasicSuite basic = [] { SPBasicSuite b{};
  b.AcquireSuite=[](const char* name,int32_t,const void** p)->SPErr {
    *p=!strcmp(name,kPFUtilitySuite)?(const void*)&utility:
       opaqueEnabled&&!strcmp(name,kPrSDKOpaqueEffectDataSuite)?(const void*)&opaque:nullptr;
    if (*p) ++suites; return *p?0:kSPBadParameterError; };
  b.ReleaseSuite=[](const char*,int32_t)->SPErr{--suites;return 0;}; return b; }();
int main() { alarm(60); @autoreleasepool {
  FILE* log=tmpfile(); const int savedStderr=dup(STDERR_FILENO);
  if(!log || savedStderr<0)return 1;
  setenv("OD_LOG_STDERR","1",1);dup2(fileno(log),STDERR_FILENO);
  auto logCount=[&](const std::string& needle) {
    fflush(stderr);rewind(log);std::string text;char buf[1024];
    while(fgets(buf,sizeof buf,log))text+=buf;
    fseek(log,0,SEEK_END);int n=0;
    for(size_t pos=0;(pos=text.find(needle,pos))!=std::string::npos;pos+=needle.size())++n;
    return n;
  };
  char error[256]{};
  id<MTLDevice> device=MTLCreateSystemDefaultDevice();
  auto ctx=ContextFor((__bridge void*)device,nullptr,error,sizeof error);
  check(ctx!=nullptr,"real core context"); if(!ctx)return 1;
  auto session=std::shared_ptr<Session>(Session::Create(ctx));
  auto inbuf=[device newBufferWithLength:320*192*16 options:MTLResourceStorageModeShared];
  auto outbuf=[device newBufferWithLength:320*192*16 options:MTLResourceStorageModeShared];
  auto pixels=(float*)inbuf.contents;
  for(int i=0;i<320*192;++i){for(int c=0;c<3;++c)pixels[4*i+c]=.2f+float((i*73+c*17)%127)*.00005f;pixels[4*i+3]=1;}
  ODFrame f{};f.mtlBuffer=(__bridge void*)inbuf;f.width=320;f.height=192;f.rowBytes=320*16;f.format=OD_PIXEL_BGRA32F;f.frameKey=OD_NO_KEY;
  ODFrame dest=f;dest.mtlBuffer=(__bridge void*)outbuf;
  ODParams p=od_params_default();p.waitUntilCompleted=1;p.temporalRadius=0;
  int offset=0; ODAnalysis a{};ODParams read{};
  PF_InData ui{};ui.appl_id=kAppID_Premiere;ui.effect_ref=(PF_ProgPtr)101;ui.pica_basicP=&basic;ui.time_scale=30;
  auto render=[&](uintptr_t ref) { check(session->Render<true>(&f,&offset,1,&p,&dest,nullptr,error,sizeof error,nullptr,nullptr,ref)==0,"bounded real render"); };
  auto now=AnalysisClock::now();
  auto publish=[&](const auto& lease,AnalysisOwner owner,uintptr_t source,bool gpu) {
    PublishAnalysisAt(lease,owner,source,gpu,session.get(),0,now);
  };
  auto readUI=[&] { return ReadAnalysisAt(&ui,a,read,nullptr,now); };
  const AnalysisOwner gpuOwner{true,1,1,true,2};
  const AnalysisOwner cpuOwner{true,1,1,false,0};
  for(bool useOpaque:{true,false}) for(bool gpu:{true,false}) {
    opaqueEnabled=useOpaque; runtime=100+10*useOpaque+gpu;
    const int originalID=runtime;
    const uintptr_t source=gpu?123:101;
    auto lease=AcquireAnalysis(&basic,runtime);
    check(lease && bool(lease->opaque)==useOpaque,"Opaque primary / runtime fallback selected");
    check(!readUI(),"empty before a render");
    p.lumaStrength=.25f; render(gpu?0:source);
    // A render before the UI's first sequence callback is not discarded.
    publish(lease,gpu?gpuOwner:cpuOwner,source,gpu);
    AnalysisLifecycle(PF_Cmd_SEQUENCE_SETUP,&ui);
    check(readUI()&&a.valid&&read.lumaStrength==.25f,"GPU/software data available under its runtime ID");
    const auto original=a;
    check(!lease->state()->uiLogged,"snapshot reads alone do not log first display");
    std::shared_ptr<AnalysisLease> shown;
    check(ReadAnalysisAt(&ui,a,read,&shown,now),"read returns actual transport for display logging");
    NoteUIAnalysis(shown);NoteUIAnalysis(shown);
    const std::string marker="UI analysis: id="+std::to_string(runtime)+" source="+(useOpaque?"opaque":"bank");
    check(logCount(marker)==1,"first-display log names actual transport once");
    shown.reset();
    check(a.width==320 && a.height==192,"snapshot belongs to the expected render");
    char r1[32],r2[32];od_analysis_status(&a,&read,r1,r2);check(strlen(r1)<=31&&strlen(r2)<=31,"status rows bounded");
    // Copied effects have distinct runtime IDs even at the same track/start.
    runtime=originalID+1000;
    auto copyLease=AcquireAnalysis(&basic,runtime);
    check(!readUI(),"unrendered copy has no inherited analysis");
    p.lumaStrength=.75f;render(gpu?0:102);
    publish(copyLease,gpu?gpuOwner:cpuOwner,gpu?456:102,gpu);
    check(readUI()&&read.lumaStrength==.75f,"unique copy reads its own data");
    runtime=originalID;
    check(readUI()&&read.lumaStrength==.25f&&a.renderId==original.renderId&&
          !memcmp(a.full,original.full,sizeof a.full),"selecting original restores only its own snapshot");
    // PF callback wrappers and their timeline IDs churn for the same effect.
    ui.effect_ref=(PF_ProgPtr)103;timeline=999;start=99;ownerKnown=false;
    check(readUI(),"UI ref/timeline/start metadata does not replace runtime identity");
    ownerKnown=true;timeline=7;start=30;ui.effect_ref=(PF_ProgPtr)101;
    render(gpu?0:104);
    publish(lease,gpu?gpuOwner:cpuOwner,gpu?789:104,gpu);
    check(readUI()&&read.lumaStrength==.75f,"GPU node / CPU callback churn updates same instance");
    // Exercise route changes while an opaque lease is still alive.
    opaqueEnabled=!useOpaque;
    check(readUI()&&read.lumaStrength==.75f,"Opaque and bank agree during route changes");
    std::shared_ptr<AnalysisLease> changedRoute;
    check(ReadAnalysisAt(&ui,a,read,&changedRoute,now),"read through changed route");
    NoteUIAnalysis(changedRoute);
    check(logCount("UI analysis: id="+std::to_string(runtime)+" source=")==1,"route churn never repeats first-display log");
    changedRoute.reset();
    opaqueEnabled=useOpaque;
  }
  opaqueEnabled=true;p.lumaStrength=.5f;
  for(int conflictKind=0;conflictKind<3;++conflictKind) {
    runtime=200+conflictKind;auto lease=AcquireAnalysis(&basic,runtime);
    const bool gpu=conflictKind!=2;
    render(gpu?0:101);
    auto owner=gpu?gpuOwner:cpuOwner;
    publish(lease,owner,gpu?123:101,gpu);
    check(readUI(),"valid before conflict");
    now+=std::chrono::milliseconds(100);
    if(conflictKind==0) owner.track=3; else owner.start=2;
    publish(lease,owner,gpu?124:101,gpu);
    check(!readUI()&&!a.valid,"real recent track/start conflict fails closed");
    // Missing evidence and repeated UI reads must not clear a conflict.
    publish(lease,{},gpu?125:101,gpu);
    check(!readUI(),"unknown owner cannot clear recent conflict");
    now+=kConflictWindow-std::chrono::milliseconds(1);
    check(!readUI(),"conflict remains for the full recent window");
    publish(lease,owner,gpu?125:101,gpu);
    now+=std::chrono::milliseconds(1);
    check(readUI(),"conflict ages out on Update without another render");
    // A legitimate move after the recent window is not another conflict.
    now+=kConflictWindow;owner.start=10;owner.track=4;
    publish(lease,owner,gpu?126:101,gpu);
    check(readUI(),"track/start changes after idle are accepted");
    // Recurring conflicts still suppress data; logging is once per ID.
    now+=std::chrono::milliseconds(10);owner.start=11;
    publish(lease,owner,gpu?127:101,gpu);
    check(!readUI()&&lease->state()->conflictLogged,"recurring conflict still fails closed");
    now+=std::chrono::seconds(1);owner.start=12;
    publish(lease,owner,gpu?128:101,gpu);
    now+=std::chrono::seconds(1);
    check(!readUI(),"continued disagreements extend suppression past the initial deadline");
    now+=std::chrono::seconds(1);
    check(readUI(),"recovery is two seconds after the last disagreement");
    check(logCount("Analysis conflict: id="+std::to_string(runtime)+" ")==1,"recurring conflict logs once per ID");
  }
  runtime=210;auto mixed=AcquireAnalysis(&basic,runtime);
  render(0);publish(mixed,gpuOwner,123,true);check(readUI(),"initial GPU publication");
  render(101);publish(mixed,{true,30,30,false,0},101,false);
  check(readUI(),"equal rational start and CPU/GPU switch do not conflict");
  render(0);publish(mixed,{true,254016000000LL,254016000000ULL,true,3},456,true);
  check(!readUI(),"software publication without track cannot erase recent GPU track conflict");
  now+=kConflictWindow;
  publish(mixed,{true,INT64_MAX,UINT64_MAX,true,3},456,true);
  check(readUI(),"64-bit rational start accepted after expiry without overflow");
  now+=std::chrono::milliseconds(1);
  publish(mixed,{true,INT64_MAX-1,UINT64_MAX,true,3},789,true);
  check(!readUI(),"one-tick rational start difference is a real conflict");
  mixed.reset();
  for(int32_t id:{0,INT32_MIN,INT32_MAX,220}) {
    runtime=id;auto lease=AcquireAnalysis(&basic,runtime);
    render(0);publish(lease,{},123,true);
    check(readUI(),"missing owner metadata is not a conflict; successful signed/zero IDs work");
    idKnown=false;check(!readUI(),"failed runtime ID lookup cannot expose data");idKnown=true;
  }
  runtime=230;auto race=AcquireAnalysis(&basic,runtime);
  render(101);publish(race,cpuOwner,101,false);check(readUI(),"software source before race");
  render(102);
  check(readUI(),"published copy survives another software callback rendering");
  publish(race,cpuOwner,101,false);
  check(!readUI(),"CPU source mismatch cannot publish another callback's render record");
  publish(race,cpuOwner,102,false);
  check(readUI(),"matching CPU render record recovers without poisoning the ID");
  race.reset();
  CaptureBytes base{};int64_t time=0;
  auto token=ArmCapture(base,1,320,192,1,1);
  ObserveCapture(token,101,*session,nullptr,&f,&time,1,30,1,true,&p);
  check(!PendingCapture(token).failed,"capture accepts initial Processing mode");
  p.chromaRecovery=1;time=1;
  ObserveCapture(token,101,*session,nullptr,&f,&time,1,30,1,true,&p);
  check(PendingCapture(token).failed && PendingCapture(token).detail=="Processing changed; capture again",
        "capture never mixes Processing estimators in one saved profile");
  CancelCapture(token);p.chromaRecovery=0;
  runtime=300;auto lease=AcquireAnalysis(&basic,runtime);
  render(101);PublishAnalysis(lease,cpuOwner,101,false,session.get(),0);
  auto state=lease->state();
  std::promise<void> locked,release;auto stop=release.get_future();
  std::thread worker([&]{std::lock_guard<std::mutex> guard(state->mutex);locked.set_value();stop.wait();});
  locked.get_future().wait();check(!ReadAnalysisForUI(&ui,a,read),"UI read returns while publication lock held");release.set_value();worker.join();
  check(ReadAnalysisForUI(&ui,a,read),"read recovers after contention");
  AnalysisLifecycle(PF_Cmd_SEQUENCE_SETDOWN,&ui);
  check(!ReadAnalysisForUI(&ui,a,read),"sequence setdown invalidates stale data");
  PublishAnalysis(lease,cpuOwner,101,false,session.get(),0);
  check(ReadAnalysisForUI(&ui,a,read),"sequence callbacks do not retire runtime IDs permanently");
  AnalysisLifecycle(PF_Cmd_SEQUENCE_RESETUP,&ui);
  check(!ReadAnalysisForUI(&ui,a,read),"sequence resetup also invalidates stale data");
  PublishAnalysis(lease,cpuOwner,101,false,session.get(),0);
  check(ReadAnalysisForUI(&ui,a,read),"fresh publication recovers after resetup");
  lease.reset();state.reset();
  for(int i=2000;i<2600;++i) { auto l=AcquireAnalysis(&basic,i); }
  check(!AcquireAnalysis(&basic,9999),"identity bank remains bounded");
  check(host.empty()&&suites==0,"all opaque references and suite leases balanced");
  runtime=300;auto closing=AcquireAnalysis(&basic,runtime);
  AnalysisLifecycle(PF_Cmd_GLOBAL_SETDOWN,&ui);
  PublishAnalysis(closing,cpuOwner,101,false,session.get(),0);
  check(closing->state()->closed&&!closing->state()->snapshot.valid,"outstanding leases cannot republish after global teardown");
  closing.reset();
  check(bank.empty(),"global teardown clears bank for the next host session");
  runtime=301;auto reopened=AcquireAnalysis(&basic,runtime);
  PublishAnalysis(reopened,cpuOwner,101,false,session.get(),0);
  check(ReadAnalysisForUI(&ui,a,read),"new host session can publish after global teardown");
  reopened.reset();check(host.empty()&&suites==0,"teardown/reopen keeps opaque and suite leases balanced");
  AnalysisLifecycle(PF_Cmd_GLOBAL_SETDOWN,&ui);
  session.reset();GlobalTeardown("bridge-test",true);
  printf("analysis bridge: %d checks, %d failures (Opaque, bank, unique IDs, expiring conflicts, churn, CPU races, bounds)\n",checks,failures);
  fflush(stderr);dup2(savedStderr,STDERR_FILENO);close(savedStderr);fclose(log);
  return failures?1:0;
} }
