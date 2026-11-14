// SPDX-License-Identifier: MIT
// Single-frame/seek overlays: <=32 MiB host buffers and a 512 MiB core budget.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "opendenoise/opendenoise.h"
#include "analysis_layout.h"
#include "debug.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>
#include <unistd.h>

static int checks=0,failures=0;
#define CHECK(x) do{++checks;if(!(x)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);++failures;}}while(0)

static ODNoiseProfile profile(){
  ODNoiseProfile p{};p.valid=1;p.confidence=1;
  const float bands[]={.8908f,.2007f,.0855f,.0412f};
  for(int c=0;c<3;++c)for(int l=0;l<8;++l){p.sigmaFull[c][l]=.008f;
    for(int b=0;b<4;++b)p.sigma[c][b][l]=.008f*bands[b];}
  return p;
}

// Compare a fresh/current-frame overlay with a reference session that has
// already rendered exactly that frame. The old one-frame lag passes the
// reference but fails the cold render and the first render after each seek.
static void checkFrame(ODSession* session,ODFrame in,ODFrame out,ODFrame expected,ODParams p){
  ODStats stats{};int offset=0;
  CHECK(od_render(session,&in,&offset,1,&p,&out,&stats)==0);
  CHECK(!od_debug_session_drain(session));
  ODAnalysis a{};CHECK(od_session_last_analysis(session,&a)==1);
  CHECK(a.renderId==stats.renderId&&a.frameKey==in.frameKey);
  CHECK(a.profileValid==!(p.debugSkipStages&(1u<<OD_STAGE_PROFILE)));
  CHECK(a.width==in.width&&a.height==in.height);
  float histogram=0;for(float h:a.histogram)histogram+=h;CHECK(std::abs(histogram-1)<1e-5);
  auto got=(__bridge id<MTLBuffer>)out.mtlBuffer;
  auto want=(__bridge id<MTLBuffer>)expected.mtlBuffer;
  CHECK(!memcmp(got.contents,want.contents,size_t(in.rowBytes)*in.height));
}

static void numbers(){
  ODAnalysis a{};a.valid=a.profileValid=1;a.removalValid=4095;
  auto p=od_params_default();ODDrawList list{};
  auto cell=[&](int c,int b){const float xs[]={88,140,186,247};
    for(uint32_t i=0;i<list.count;++i){const auto& d=list.commands[i];
      if(d.kind==OD_DRAW_TEXT&&d.x==xs[b]&&d.y==305+27*c)return d.text;}
    return "missing";
  };
  for(float v:{-0.f,-.001f,-.049f,0.f,.049f,-.051f,-1.25f,1.25f}){
    for(int b=0;b<4;++b){a.removedDb[0][b]=v;
      for(int c=1;c<3;++c){a.inputEnergy[c][b]=1;a.outputEnergy[c][b]=std::pow(10.f,-v/10);}}
    od::analysisViewLayout(&a,&p,&list,360,400);
    for(int c=0;c<2;++c)for(int b=0;b<4;++b){CHECK(strcmp(cell(c,b),"-0.0"));
      if(std::abs(v)<.05f)CHECK(!strcmp(cell(c,b),"0.0"));
      if(v==-.051f)CHECK(!strcmp(cell(c,b),"-0.1"));}
  }
  for(uint32_t mask:{0u,15u,255u,3855u}){a.removalValid=mask;od::analysisViewLayout(&a,&p,&list,360,400);
    for(int b=0;b<4;++b)CHECK(!strcmp(cell(1,b),"--"));}
  a.removalValid=4095;
  for(float v:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}){
    a.removedDb[0][0]=v;a.inputEnergy[1][0]=v;
    od::analysisViewLayout(&a,&p,&list,360,400);CHECK(!strcmp(cell(0,0),"--"));CHECK(!strcmp(cell(1,0),"--"));
  }
  a.inputEnergy[1][0]=0;od::analysisViewLayout(&a,&p,&list,360,400);CHECK(!strcmp(cell(1,0),"--"));
  a.inputEnergy[1][0]=1;a.outputEnergy[2][0]=-1;od::analysisViewLayout(&a,&p,&list,360,400);CHECK(!strcmp(cell(1,0),"--"));
  a.outputEnergy[1][0]=a.outputEnergy[2][0]=0;
  od::analysisViewLayout(&a,&p,&list,360,400);CHECK(!strcmp(cell(1,0),"60.0"));
}

int main(){alarm(180);setenv("OD_MEMORY_MB","512",1);@autoreleasepool{
  numbers();
  auto dev=MTLCreateSystemDefaultDevice();auto queue=[dev newCommandQueue];
  auto ctx=od_context_create((__bridge void*)dev,(__bridge void*)queue,nullptr,0);CHECK(ctx);if(!ctx)return 1;
  const uint32_t W=512,H=512;
  for(int format:{0,1})for(bool inplace:{false,true})for(int mode:{0,1,2}){@autoreleasepool{
    // Mode 0: automatic, first render only, no prior session or profile.
    // Mode 1: fixed profile, unrelated seek, and changed pixels under one key.
    // Mode 2: unavailable profile, compact empty-state pixels.
    uint32_t bpp=format?8:16,stride=W*bpp+32;size_t bytes=size_t(stride)*H;
    auto src=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    auto dst=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    auto ref=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    CHECK(src&&dst&&ref);if(!src||!dst||!ref)return 1;
    auto s=od_session_create(ctx);CHECK(s);if(!s)return 1;
    auto p=od_params_default();auto manual=profile();p.temporalRadius=0;p.verifyFrameCache=1;
    p.view=OD_VIEW_ANALYSIS;p.realtime=1;p.waitUntilCompleted=0;
    if(mode==1)p.manualProfile=&manual;
    if(mode==2)p.debugSkipStages=1u<<OD_STAGE_PROFILE;
    for(int step=0;step<(mode==1?3:1);++step){
      std::mt19937 rng(99+step);std::normal_distribution<float> n(0,.008f/(step+1));
      memset(src.contents,0x35,bytes);memset(dst.contents,0x35,bytes);memset(ref.contents,0x35,bytes);
      for(uint32_t y=0;y<H;++y)for(uint32_t x=0;x<W;++x)for(int c=0;c<4;++c){
        const float v=c==3?.75f:.15f+step*.3f+n(rng);
        if(format)((__fp16*)((char*)src.contents+y*stride))[x*4+c]=v;
        else((float*)((char*)src.contents+y*stride))[x*4+c]=v;
      }
      ODFrame in{(__bridge void*)src,W,H,stride,(ODPixelFormat)format,step?900:1},out=in,expected=in;
      out.mtlBuffer=(__bridge void*)(inplace?src:dst);expected.mtlBuffer=(__bridge void*)ref;
      auto result=od_session_create(ctx);CHECK(result);auto rp=p;rp.view=OD_VIEW_OUTPUT;rp.waitUntilCompleted=1;int offset=0;
      CHECK(!od_render(result,&in,&offset,1,&rp,&expected,nullptr));
      std::vector<uint8_t> plain(bytes);memcpy(plain.data(),ref.contents,bytes);
      rp.view=OD_VIEW_ANALYSIS;CHECK(!od_render(result,&in,&offset,1,&rp,&expected,nullptr));
      od_session_destroy(result);
      checkFrame(s,in,out,expected,p);
      if(mode==2){uint32_t bx,by,bw,bh;od::analysisViewBox(nullptr,W,H,&bx,&by,&bw,&bh);
        auto got=(const uint8_t*)(inplace?src:dst).contents;size_t outside=0,inside=0;
        for(uint32_t y=0;y<H;++y)for(uint32_t x=0;x<W;++x){size_t i=size_t(y)*stride+x*bpp;
          bool changed=memcmp(got+i,plain.data()+i,bpp)!=0;
          if(x<bx||x>=bx+bw||y<by||y>=by+bh)outside+=changed;else inside+=changed;}
        CHECK(!outside&&inside>0);
      }
    }
    od_session_destroy(s);
  }}
  od_context_destroy(ctx);printf("analysis current frame: %d checks, %d failures\n",checks,failures);return failures?1:0;
}}
