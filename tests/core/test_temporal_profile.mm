// SPDX-License-Identifier: MIT
// The analysis window is independent of temporal strength and merge radius.
// Bounded: 24 host frames at 640x384 (< 95 MiB), one session at a time.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "opendenoise/opendenoise.h"
#include <vector>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <algorithm>
static unsigned hash(unsigned x) {x^=x>>16;x*=0x7feb352d;x^=x>>15;x*=0x846ca68b;return x^(x>>16);}
int main() { @autoreleasepool {
  auto device=MTLCreateSystemDefaultDevice();char error[1024]={};
  auto ctx=od_context_create((__bridge void*)device,nullptr,error,sizeof(error));
  if(!ctx) {fprintf(stderr,"%s\n",error);return 2;}
  constexpr int W=640,H=384,N=24;
  std::vector<id<MTLBuffer>> frames;
  for(int t=0;t<N;++t) {
    auto b=[device newBufferWithLength:W*H*16 options:MTLResourceStorageModeShared];
    float* a=(float*)b.contents;
    for(int y=0;y<H;++y)for(int x=0;x<W;++x)for(int c=0;c<4;++c) {
      float z=float(hash((y*W+x)*31+c*719+t*17981)&65535)/65535.f-.5f;
      a[(y*W+x)*4+c]=c==3? .7f : .25f+.08f*sinf(x*.08f+t*.02f)*sinf(y*.04f)+(t>=12?.09f:.04f)*z;
    }
    frames.push_back(b);
  }
  auto out=[device newBufferWithLength:W*H*16 options:MTLResourceStorageModeShared];
  int failures=0,checks=0;
  // Integration: the recovered chroma estimator must see that same fixed
  // window. Compare each estimator with its own baseline, never with the
  // other estimator's picture.
  for(unsigned recovery:{0u,1u}) {
  std::vector<ODNoiseProfile> reference;
  for(int radius : {0,1,2,3,5})for(float strength : {0.f,.0001f,.25f,1.f}) {
    auto s=od_session_create(ctx); ODParams p=od_params_default();
    p.temporalNoiseModel=1;p.chromaRecovery=recovery;
    p.temporalRadius=radius;p.temporalStrength=strength;p.waitUntilCompleted=1;
    const int delivered=std::max(radius,OD_PROFILE_RADIUS);
    for(int t=5;t<19;++t) {
      ODFrame f[OD_MAX_FRAMES]={};int offsets[OD_MAX_FRAMES]={};int n=0;
      for(int k:{0,-1,1,-2,2,-3,3,-4,4,-5,5})if(std::abs(k)<=delivered) {
        f[n]={(__bridge void*)frames[t+k],W,H,W*16,OD_PIXEL_BGRA32F,t+k};offsets[n++]=k;
      }
      ODFrame o=f[0];o.mtlBuffer=(__bridge void*)out;ODStats st={};
      ++checks;if(od_render(s,f,offsets,n,&p,&o,&st)) {++failures;fprintf(stderr,"render %s\n",od_last_error(s));break;}
      if(radius==0&&strength==0)reference.push_back(st.profile);
      else {++checks;if(memcmp(&st.profile,&reference[t-5],sizeof(st.profile))) {++failures;fprintf(stderr,"profile changed r%d s%g frame%d\n",radius,strength,t);}}
    }
    od_session_destroy(s);
  }
  }
  od_context_destroy(ctx);
  printf("temporal profile: %d checks, %d failures\n",checks,failures);return failures?1:0;
}}
