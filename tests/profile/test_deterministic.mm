// SPDX-License-Identifier: MIT
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "opendenoise/opendenoise.h"
#ifndef OD_PROFILE_BASELINE
#include "opendenoise/profiling.h"
#else
#define od_render_deterministic od_render
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
static int checks=0, failures=0;
#define CHECK(c) do {++checks; if(!(c)){++failures;fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c);}}while(0)
int main() { @autoreleasepool {
  auto dev=MTLCreateSystemDefaultDevice(); auto q=[dev newCommandQueue]; char err[512];
  auto ctx=od_context_create((__bridge void*)dev,(__bridge void*)q,err,sizeof(err)); if(!ctx) return 2;
  constexpr int W=640,H=384,N=8; const size_t bytes=W*H*16;
  auto buffer=[&](){return [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];};
  auto frame=[&](id<MTLBuffer> b,int key){return ODFrame{(__bridge void*)b,W,H,W*16,OD_PIXEL_BGRA32F,key};};
  auto fill=[&](id<MTLBuffer> b,float sigma,int seed,bool mono){
    std::mt19937 gen(seed);std::normal_distribution<float> normal(0,sigma);auto p=(float*)b.contents;
    for(int y=0;y<H;++y)for(int x=0;x<W;++x){int i=(y*W+x)*4; float level=0.10f+0.65f*(x/80)/7;float n=normal(gen);if(mono){level=0.25f;n=std::round(n*1024.0f)/1024.0f;}
      for(int c=0;c<3;++c)p[i+c]=level+(mono?n:normal(gen));p[i+3]=1;}
  };
  id<MTLBuffer> src[N];for(int i=0;i<N;++i){src[i]=buffer();fill(src[i],0.004f*(i+1),100+i,false);}
  auto out=buffer();auto dst=frame(out,-1);int offs[3]={0,-1,1};
  ODParams p=od_params_default();p.waitUntilCompleted=1;p.noFrameCache=1;p.temporalRadius=0;
  auto s=od_session_create(ctx);auto mono=buffer();fill(mono,0.02f,42,true);auto mf=frame(mono,55);ODNoiseProfile mp={};
  int rc=od_profile_frame(s,&mf,&mp);printf("monochrome rc=%d valid=%u Y=%.9g Cb=%.9g Cr=%.9g\n",rc,mp.valid,mp.sigma[0][0][2],mp.sigma[1][0][2],mp.sigma[2][0][2]);
  CHECK(rc==0&&mp.valid);CHECK(mp.sigma[0][0][2]>0.015&&mp.sigma[0][0][2]<0.021);CHECK(mp.sigmaFull[1][2]<1e-5&&mp.sigmaFull[2][2]<1e-5);
  // Chroma-only noise: Y is constant after ingest, Cb varies, Cr is zero.
  std::mt19937 cg(22);std::normal_distribution<float> cn(0,0.015f);auto cp=(float*)mono.contents;
  for(int i=0;i<W*H;++i){float cb=cn(cg);cp[i*4]=0.4f+1.8556f*cb;cp[i*4+1]=0.4f-(0.0722f*1.8556f/0.7152f)*cb;cp[i*4+2]=0.4f;cp[i*4+3]=1;}
  rc=od_profile_frame(s,&mf,&mp);
  printf("chroma_only rc=%d valid=%u Y=%.9g Cb=%.9g Cr=%.9g\n",rc,mp.valid,mp.sigma[0][0][2],mp.sigma[1][0][2],mp.sigma[2][0][2]);
  CHECK(rc==0&&mp.valid);CHECK(mp.sigma[1][0][2]>0.010&&mp.sigma[1][0][2]<0.017);CHECK(mp.sigma[0][0][2]<1e-5);
  od_session_reset(s);ODStats old={},changed={},fresh={};auto f=frame(src[0],7);
  for(int i=0;i<5;++i)CHECK(od_render_deterministic(s,&f,offs,1,&p,&dst,&old)==0);
  f=frame(src[7],7);CHECK(od_render_deterministic(s,&f,offs,1,&p,&dst,&changed)==0);
  std::vector<uint8_t> a(bytes);memcpy(a.data(),out.contents,bytes);auto freshSession=od_session_create(ctx);
  CHECK(od_render_deterministic(freshSession,&f,offs,1,&p,&dst,&fresh)==0);
  printf("same_key old=%.9g changed=%.9g fresh=%.9g ratio=%.6f bytes_equal=%d\n",old.profile.sigma[0][0][2],changed.profile.sigma[0][0][2],fresh.profile.sigma[0][0][2],changed.profile.sigma[0][0][2]/fresh.profile.sigma[0][0][2],!memcmp(a.data(),out.contents,bytes));
  CHECK(!memcmp(a.data(),out.contents,bytes));CHECK(!memcmp(&changed.profile,&fresh.profile,sizeof(ODNoiseProfile)));
  od_session_destroy(freshSession);
  for(int radius:{0,1})for(int quality:{0,1}) {
    p.temporalRadius=radius;p.quality=quality;
    std::vector<std::vector<uint8_t>> reference(N,std::vector<uint8_t>(bytes));ODNoiseProfile profiles[N];
    for(int pass=0;pass<3;++pass){od_session_reset(s);int permutation[3][N]={{0,1,2,3,4,5,6,7},{7,6,5,4,3,2,1,0},{3,0,6,1,7,4,2,5}};
      size_t mismatched=0,profileMismatch=0;
      id<MTLBuffer> outputs[N];ODNoiseProfile current[N];for(int i=0;i<N;++i)outputs[i]=buffer();
      for(int j=0;j<N;++j){int i=permutation[pass][j];ODFrame fr[3]={frame(src[i],100+i),frame(src[(i+N-1)%N],100+(i+N-1)%N),frame(src[(i+1)%N],100+(i+1)%N)};ODStats st={};
        p.waitUntilCompleted=pass!=2;p.realtime=pass==2;
        auto target=frame(outputs[i],-1);
        CHECK(od_render_deterministic(s,fr,offs,3,&p,&target,&st)==0);
        current[i]=st.profile;
      }
      auto done=[q commandBuffer];[done commit];[done waitUntilCompleted];
      for(int i=0;i<N;++i) {
        if(pass==0){memcpy(reference[i].data(),outputs[i].contents,bytes);profiles[i]=current[i];}
        else {mismatched+=memcmp(reference[i].data(),outputs[i].contents,bytes)!=0;profileMismatch+=memcmp(&profiles[i],&current[i],sizeof(current[i]))!=0;}
      }
      printf("order radius=%d quality=%d pass=%d output_mismatch=%zu profile_mismatch=%zu\n",radius,quality,pass,mismatched,profileMismatch);
      CHECK(mismatched==0&&profileMismatch==0);
    }
  }
#ifndef OD_PROFILE_BASELINE
  ODNoiseProfile inputs[3]={old.profile,changed.profile,fresh.profile},agg={},reverse={};
  CHECK(od_profile_aggregate(inputs,3,&agg)==0);std::reverse(inputs,inputs+3);CHECK(od_profile_aggregate(inputs,3,&reverse)==0);CHECK(!memcmp(&agg,&reverse,sizeof(agg)));
  std::vector<uint8_t> blob(od_profile_blob_size());CHECK(od_profile_encode(&agg,blob.data(),blob.size())==0);
  ODNoiseProfile decoded={};CHECK(od_profile_decode(blob.data(),blob.size(),&decoded)==0);CHECK(!memcmp(&agg,&decoded,sizeof(agg)));
  // A decoded capture drives a fresh session identically to the original.
  p.waitUntilCompleted=1;p.realtime=0;p.temporalRadius=0;p.manualProfile=&agg;
  CHECK(od_render_deterministic(s,&f,offs,1,&p,&dst,nullptr)==0);
  memcpy(a.data(),out.contents,bytes);freshSession=od_session_create(ctx);p.manualProfile=&decoded;
  CHECK(od_render_deterministic(freshSession,&f,offs,1,&p,&dst,nullptr)==0);
  CHECK(!memcmp(a.data(),out.contents,bytes));od_session_destroy(freshSession);p.manualProfile=nullptr;
  for(size_t i=0;i<blob.size();++i){blob[i]^=1;CHECK(od_profile_decode(blob.data(),blob.size(),&decoded)!=0&&!decoded.valid);blob[i]^=1;}
  CHECK(od_profile_decode(blob.data(),blob.size()-1,&decoded)!=0);
  inputs[0].sigma[0][0][0]=NAN;CHECK(od_profile_encode(&inputs[0],blob.data(),blob.size())!=0);
  CHECK(od_profile_aggregate(inputs,3,&decoded)!=0&&!decoded.valid);
#endif
  od_session_destroy(s);od_context_destroy(ctx);printf("%d checks, %d failures\n",checks,failures);return failures?1:0;
}}
