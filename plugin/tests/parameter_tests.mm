// SPDX-License-Identifier: MIT
// Shared mapping against the real core's defaults, including every ABI field.
#include <cstdio>
#include <unistd.h>
#include <cstring>
#include <cmath>
#include <climits>
#include "../src/OpenDenoiseShared.mm"
int main() {
  using namespace odplug;
  alarm(30);
  unsigned checks=0, failed=0;
  auto check=[&](bool ok,const char* what){++checks;if(!ok){++failed;printf("FAIL %s\n",what);}};
  for(int controls:{1,2}) for(int version:{1,2}) for(bool cpu:{true,false}) {
    UIParams u;u.controlSet=controls;u.processing=version;
    const auto result=ToCoreParams(u,true,cpu);
    for(int view:{6,7,INT_MAX,0,-1,INT_MIN}) {
      u.view=view; const auto fallback=ToCoreParams(u,true,cpu);
      check(!memcmp(&result,&fallback,sizeof result),"invalid View maps every render field to Result without rewriting storage");
      check(u.view==view,"saved View remains untouched");
    }
    for(int view=1;view<=5;++view) {
      u.view=view;check(ToCoreParams(u,true,cpu).view==view-1,"five visible Views retain their core indices");
    }
  }
  for(int version:{1,2}) for(bool cpu:{true,false}) {
    UIParams u;u.processing=version;
    // Adobe's stored float default, then a typed double of the same value.
    for(double grain:{double(.657f),.657}) {
      u.dbGrain=grain;
      ODParams p=ToCoreParams(u,true,cpu), expected=od_params_default();
      expected.chromaRecovery = version == 2 ? 1u : 0u;
      expected.temporalNoiseModel = version == 2 ? 1u : 0u;
      check(p.verifyFrameCache==1 && p.waitUntilCompleted==int(cpu),"host execution policy");
      // Existing cache verification and CPU readback policy are intentionally
      // not od_params_default(): compare them above, then compare EVERY byte.
      p.verifyFrameCache=expected.verifyFrameCache;p.waitUntilCompleted=expected.waitUntilCompleted;
      check(!memcmp(&p,&expected,sizeof p),"new dB default equals real od_params_default in every field");
    }
  }
  // Preset/paste detection must accept both float and double default storage.
  for(double grain:{double(.657f),.657}) {
    UIParams u;u.dbGrain=grain;u.luma=61;
    check(EffectiveControlSet(u)==1,"old preset detected");
    u.dbLuma=9;check(EffectiveControlSet(u)==2,"dB edit wins");
    u.controlSet=1;check(EffectiveControlSet(u)==1,"explicit Previous wins");
  }
  // Safe conversions preserve every effective picture parameter (float ULP
  // rounding only); selected zero-strength band amounts are render-neutral.
  for(int i=0;i<50;++i) {
    UIParams old;old.controlSet=1;old.processing=1;
    old.luma=20+i;old.chroma=30+i;old.grainFloor=i*8;old.spatial=10+i;
    old.motionThreshold=60+i;old.noiseLevel=40+i*4;old.sharpenNoiseFloor=70+i;
    old.temporalRadius=i%6;
    for(int b=0;b<4;++b) {
      old.lumaAmount[b]=20+i;old.chromaAmount[b]=30+i;
      old.lumaLevel[b]=40+i*3+b;old.chromaLevel[b]=60+i*2+b;
    }
    old.noiseY=60+i;old.noiseCb=70+i;old.noiseCr=80+i;
    UIParams after;std::string reason;
    check(ConvertToDB(old,after,reason),"representable conversion succeeds");
    check(after.processing==1,"conversion preserves processing");
    const ODParams a=ToCoreParams(old,true,false),b=ToCoreParams(after,true,false);
#define SAME(f) check(a.f==b.f,#f " exact conversion")
    SAME(lumaStrength);SAME(chromaStrength);SAME(temporalRadius);SAME(temporalStrength);
    SAME(spatialStrength);SAME(noiseLevel);SAME(grainFloor);SAME(temporalMotionThreshold);SAME(sharpenCoring);
    for(int g=0;g<2;++g) for(int band=0;band<4;++band) {
      check(std::fabs(a.bandAmount[g][band]-b.bandAmount[g][band])<=1e-7,"band amount float rounding");
      check(a.bandNoiseLevel[g][band]==b.bandNoiseLevel[g][band],"size correction exact conversion");
    }
    for(int c=0;c<3;++c) check(a.channelNoiseLevel[c]==b.channelNoiseLevel[c],"channel correction exact conversion");
#undef SAME
  }
  printf("real-core parameter pin: %u checks, %u failed; Processing 2 differs only by chromaRecovery and temporalNoiseModel\n",checks,failed);
  return failed?1:0;
}
