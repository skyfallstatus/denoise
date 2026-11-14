// SPDX-License-Identifier: MIT
// Window-local stability and scene-colour rejection. 64 frames, five buffers;
// fixed seed, <160 MiB of host storage, no media and no recursive image history.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "opendenoise/opendenoise.h"
#include "debug.h"

int main() {
  @autoreleasepool {
    setenv("OD_COARSE_CHROMA","1",1);
    constexpr int W=768,H=768,T=64,M=32,P=W*H;
    auto dev=MTLCreateSystemDefaultDevice(); char err[512]={};
    auto ctx=od_context_create((__bridge void*)dev,nullptr,err,sizeof(err));
    if(!ctx) { fprintf(stderr,"%s\n",err); return 1; }
    auto session=od_session_create(ctx);
    id<MTLBuffer> buffers[5],output=[dev newBufferWithLength:P*16 options:MTLResourceStorageModeShared];
    if(!output) return 2;
    for(auto& b:buffers) { b=[dev newBufferWithLength:P*16 options:MTLResourceStorageModeShared]; if(!b) return 2; }
    std::mt19937 rng(73121); std::normal_distribution<float> normal(0,1);
    std::vector<float> wave(M*P),coeff(T*2*M);
    for(int j=0;j<M;++j) {
      float a=j*2.399963f,k=6.283185f/(32+(j%16)*6);
      for(int y=0;y<H;++y) for(int x=0;x<W;++x) wave[j*P+y*W+x]=sin(k*(cos(a)*x+sin(a)*y)+j);
    }
    for(int t=0;t<T;++t) for(int c=0;c<2;++c) for(int j=0;j<M;++j)
      coeff[(t*2+c)*M+j]=t?.90f*coeff[((t-1)*2+c)*M+j]+sqrt(1-.9f*.9f)*normal(rng):normal(rng);
    auto fill=[&](id<MTLBuffer> buffer,int t,int mode) {
      float* dst=(float*)buffer.contents;
      std::mt19937 local(9741+t); std::normal_distribution<float> n(0,1);
      for(int y=0;y<H;++y) for(int x=0;x<W;++x) {
        int i=y*W+x;
        float yy=.4f+.03f*x/W, cb=.02f+.01f*y/H,cr=.04f+.01f*x/W;
        if(mode==0 || (mode==3 && t<32)) {
          float z[2]={};
          for(int c=0;c<2;++c) for(int j=0;j<M;++j) z[c]+=coeff[(t*2+c)*M+j]*wave[j*P+i];
          cb+=.004f*z[0]/sqrt(M*.5f);cr+=.004f*z[1]/sqrt(M*.5f);
        }
        if(mode==1 || (mode==3 && t>=32)) {
          float u=(x-300-t*2)/250.f,v=(y-350+t)/230.f;
          float light=.05f*exp(-.5f*(u*u+v*v));cb+=light;cr-=.7f*light;
        }
        if(mode==2) {
          int mx=180+t*3,my=360+t/4;
          if(abs(x-mx)<4 && abs(y-my)<4) { cb=-.18f;cr=.24f; }
        }
        yy+=.002f*n(local);cb+=.001f*n(local);cr+=.001f*n(local);
        dst[4*i]=yy+1.8556f*cb;dst[4*i+1]=yy-.187324f*cb-.468124f*cr;
        dst[4*i+2]=yy+1.5748f*cr;dst[4*i+3]=1;
      }
    };
    ODFrame out={(__bridge void*)output,W,H,W*16,OD_PIXEL_BGRA32F,OD_NO_KEY};
    ODParams p=od_params_default();p.waitUntilCompleted=1;p.noFrameCache=1;
    // Profile only: fresh measurement per window; no history-dependent sigma
    // from the existing a-trous ring enters the equivalence assertions.
    p.debugFlags=OD_DEBUG_FORCE_PROFILE;p.debugSkipStages=(1u<<OD_STAGE_SPATIAL);
    int fails=0;
    for(int mode=0;mode<4;++mode) {
      std::vector<ODNoiseProfile> profiles(T);
      int counts[56]={},toggles[56]={};bool last[56]={};
      for(int t=2;t<T-2;++t) {
        ODFrame frames[5]; int offsets[5]={0,-1,1,-2,2};
        for(int j=0;j<5;++j) {
          fill(buffers[j],t+offsets[j],mode);
          frames[j]={(__bridge void*)buffers[j],W,H,W*16,OD_PIXEL_BGRA32F,t+offsets[j]+1};
        }
        od_session_reset(session);ODStats st={};
        if(od_render(session,frames,offsets,5,&p,&out,&st)) return 3;
        profiles[t]=st.profile;
        for(int b=0;b<7;++b) for(int l=0;l<8;++l) {
          int i=b*8+l;bool on=st.profile.coarseConfidence[b][l]>=.5f;
          counts[i]+=on;if(t>2 && last[i]!=on) ++toggles[i];last[i]=on;
        }
      }
      int active=0,switches=0;for(int i=0;i<56;++i){ active+=counts[i];switches+=toggles[i]; }
      printf("coarse time mode %d: active knot-frames %d, transitions %d; L3 knot3 %d/60, transitions %d\n",mode,active,switches,counts[3],toggles[3]);
      if(mode==0 && (counts[3]!=60 || toggles[3])) ++fails;
      if((mode==1 || mode==2) && active) ++fails;
      if(mode==3) for(int t=35;t<62;++t) for(auto& band:profiles[t].coarseConfidence) for(float q:band) if(q>=.5f) ++fails;
      // Same windows in scrambled order and repeats, with intervening resets.
      for(int t:{29,7,52,29,7}) {
        ODFrame frames[5];int offsets[5]={0,-1,1,-2,2};
        for(int j=0;j<5;++j) {fill(buffers[j],t+offsets[j],mode);frames[j]={(__bridge void*)buffers[j],W,H,W*16,OD_PIXEL_BGRA32F,t+offsets[j]+1};}
        od_session_reset(session);ODStats st={};
        if(od_render(session,frames,offsets,5,&p,&out,&st)) return 3;
        if(memcmp(st.profile.coarseSigma,profiles[t].coarseSigma,sizeof(st.profile.coarseSigma)) ||
           memcmp(st.profile.coarseConfidence,profiles[t].coarseConfidence,sizeof(st.profile.coarseConfidence))) ++fails;
      }
    }
    od_session_destroy(session);od_context_destroy(ctx);
    printf("coarse time: %d failures\n",fails);return fails?1:0;
  }
}
