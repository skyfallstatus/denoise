// SPDX-License-Identifier: MIT
// Quantized subsampled chroma with codec-flat islands must not zero the
// whole colour profile. Three 768-square frames, under 50 MiB of images.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "opendenoise/opendenoise.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
static int fails=0;
#define CHECK(x) do { if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);++fails;} } while(0)
int main(){ @autoreleasepool {
 constexpr int W=768,H=768,N=3;
 auto dev=MTLCreateSystemDefaultDevice();char err[512]={};
 auto ctx=od_context_create((__bridge void*)dev,nullptr,err,sizeof(err));
 if(!ctx){fprintf(stderr,"%s\n",err);return 1;}
 auto ses=od_session_create(ctx);
 id<MTLBuffer> in[N],out=[dev newBufferWithLength:W*H*16 options:MTLResourceStorageModeShared];
 ODFrame fr[N];int offs[N]={0,-1,1};
 ODFrame dst={(__bridge void*)out,W,H,W*16,OD_PIXEL_BGRA32F,OD_NO_KEY};
 std::mt19937 rng(91832);std::normal_distribution<float> norm(0,1);
 std::vector<float> cn[2];for(auto& v:cn)v.resize(W*H/4);
 auto fill=[&](int mode){
  for(int t=0;t<N;++t){
   in[t]=[dev newBufferWithLength:W*H*16 options:MTLResourceStorageModeShared];
   for(auto& v:cn)for(auto& x:v)x=std::round(.003f*norm(rng)*224)/224;
   float* p=(float*)in[t].contents;
   for(int y=0;y<H;++y)for(int x=0;x<W;++x){
    float yy=.12f+.45f*x/W+.008f*norm(rng),c[2];
    for(int ch=0;ch<2;++ch){
     int xx=x/2,yyi=y/2;float fx=(x%2)*.5f,fy=(y%2)*.5f;
     auto at=[&](int a,int b){return cn[ch][std::min(b,H/2-1)*(W/2)+std::min(a,W/2-1)];};
     c[ch]=(1-fy)*((1-fx)*at(xx,yyi)+fx*at(xx+1,yyi))+fy*((1-fx)*at(xx,yyi+1)+fx*at(xx+1,yyi+1));
     if(mode==0 && ((ch==0 && x>W/3 && x<W/2)||(ch==1 && x<W/5)))c[ch]=1e-7f*norm(rng);
     if(mode==2)c[ch]=0;
    }
    // Equal-luma saturated marks: their colour edges must not bleed.
    if(y>H*3/4 && y<H*3/4+16 && x>W/2 && x<W/2+16){c[0]+=.18f;c[1]-=.12f;}
    int i=(y*W+x)*4;p[i]=yy+1.8556f*c[0];p[i+1]=yy-.187324f*c[0]-.468124f*c[1];p[i+2]=yy+1.5748f*c[1];p[i+3]=1;
   }
   fr[t]={(__bridge void*)in[t],W,H,W*16,OD_PIXEL_BGRA32F,t+1};
  }
 };
 auto render=[&](ODParams p, bool reset, ODStats* st){
  if(reset)od_session_reset(ses);
  CHECK(od_render(ses,fr,offs,N,&p,&dst,st)==0);
  std::vector<float> r(W*H*4);memcpy(r.data(),out.contents,r.size()*4);return r;
 };
 ODParams p=od_params_default();p.waitUntilCompleted=1;p.verifyFrameCache=1;
 CHECK(p.chromaRecovery==0);
 for(int mode=0;mode<3;++mode){
  fill(mode);p.chromaRecovery=0;ODStats a={},b={};auto old=render(p,true,&a);
  p.chromaRecovery=1;auto on=render(p,true,&b);
  printf("mode %d Cb full legacy/recovered %.7f %.7f; Cr %.7f %.7f\n",mode,a.profile.sigmaFull[1][2],b.profile.sigmaFull[1][2],a.profile.sigmaFull[2][2],b.profile.sigmaFull[2][2]);
  if(mode==0){
   CHECK(b.profile.sigmaFull[1][2]>a.profile.sigmaFull[1][2]*5);
   CHECK(b.profile.sigmaFull[1][2]>.0004f);
   double mse[2]={},markDelta=0,bias[2][2]={}; int count[2]={};
   for(int y=32;y<H-32;++y)for(int x=32;x<W-32;++x){
    int i=(y*W+x)*4;
    auto cb=[&](const std::vector<float>& v){float l=.0722f*v[i]+.7152f*v[i+1]+.2126f*v[i+2];return (v[i]-l)/1.8556f;};
    if(y<H/2 && x>W/2){mse[0]+=cb(old)*cb(old);mse[1]+=cb(on)*cb(on);}
    if(y>H*3/4-2&&y<H*3/4+18&&x>W/2-2&&x<W/2+18){
      bool inside=y>H*3/4&&y<H*3/4+16&&x>W/2&&x<W/2+16;
      float truth=inside?.18f:0.f; int g=inside?0:1;
      bias[g][0]+=cb(old)-truth;bias[g][1]+=cb(on)-truth;++count[g];
      markDelta=std::max(markDelta,double(std::abs(cb(old)-cb(on))));
    }
   }
   printf("Cb MSE ratio %.5f; max mark/edge delta %.5f codes\n",mse[1]/mse[0],markDelta*1023);
   CHECK(mse[1]<mse[0]*.85);
   for(int g=0;g<2;++g){
    printf("mark region %d mean error old/new %.5f %.5f codes\n",g,1023*bias[g][0]/count[g],1023*bias[g][1]/count[g]);
    CHECK(std::abs(bias[g][1]/count[g])<.001f);
    CHECK(std::abs((bias[g][1]-bias[g][0])/count[g])<.1f/1023);
   }
   // Toggle on a shared session; both modes must retain their own profile.
   p.chromaRecovery=0;auto off=render(p,false,nullptr);CHECK(off==old);
   p.chromaRecovery=1;auto again=render(p,false,nullptr);CHECK(again==on);
   ODNoiseProfile captureOld={},captureNew={},captureNeutral={};
   CHECK(od_profile_frame(ses,&fr[0],&captureOld)==0);
   CHECK(od_profile_frame_with_params(ses,&fr[0],&p,&captureNew)==0);
   CHECK(od_profile_frame_with_params(ses,&fr[0],nullptr,&captureNeutral)==0);
   CHECK(memcmp(&captureOld,&captureNeutral,sizeof(captureOld))==0);
   CHECK(captureNew.sigmaFull[1][2]>captureOld.sigmaFull[1][2]*5);
   p.manualProfile=&a.profile;auto manualOn=render(p,true,nullptr);
   p.chromaRecovery=0;CHECK(render(p,true,nullptr)==manualOn);p.manualProfile=nullptr;
   p.grainMode=OD_GRAIN_WHOLE;p.advanced=0;p.chromaRecovery=1;
   auto neutral=render(p,true,nullptr);
   p.chromaRecovery=0;CHECK(render(p,true,nullptr)==neutral);
   p.advanced=1;p.grainMode=OD_GRAIN_FINEST;
  }else CHECK(old==on);
  p.chromaRecovery=99;CHECK(render(p,true,nullptr)==old);
 }
 od_session_destroy(ses);od_context_destroy(ctx);
 printf("chroma recovery failures: %d\n",fails);return fails?1:0;
}}
