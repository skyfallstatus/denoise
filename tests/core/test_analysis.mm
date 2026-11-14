// SPDX-License-Identifier: MIT
// Bounded analysis tests: <128 MB, no files, no host applications.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "opendenoise/opendenoise.h"
#include "context.h"
#include "stages/stage.h"
#include "debug.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
static int failures=0;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);++failures;}}while(0)
static ODNoiseProfile profile(float n=.008f){ODNoiseProfile p={};p.valid=1;p.confidence=1;const float bands[]={.8908f,.2007f,.0855f,.0412f};for(int c=0;c<3;++c)for(int l=0;l<8;++l){p.sigmaFull[c][l]=n;for(int b=0;b<4;++b)p.sigma[c][b][l]=n*bands[b];}return p;}
static void fill(float* dst,int w,int h,int kind,int seed){
  std::mt19937 rng(seed);std::normal_distribution<float> normal(0,.008f);std::vector<float> noise(w*h);
  for(auto& v:noise)v=normal(rng);
  for(int y=0;y<h;++y)for(int x=0;x<w;++x){float n=noise[y*w+x];if(kind==1){n=0;for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx)n+=noise[std::clamp(y+dy,0,h-1)*w+std::clamp(x+dx,0,w-1)]/5;}
    float Y=.4f+(kind==2?0:n),cb=kind==2?n:0,cr=kind==2?normal(rng):0;
    dst[(y*w+x)*4+0]=Y+1.8556f*cb;dst[(y*w+x)*4+1]=Y-.187324f*cb-.468124f*cr;dst[(y*w+x)*4+2]=Y+1.5748f*cr;dst[(y*w+x)*4+3]=.75f;
  }
}
// Independent CPU energy oracle: decode the actual host pixels, then apply
// a plain double-precision Haar transform. Does not use shader helpers.
static void energies(const uint8_t* data,uint32_t stride,int fmt,uint32_t W,uint32_t H,double out[3][4]) {
  for(int tile=0;tile<128;++tile){uint32_t x0=(tile%16)*(W-32)/15,y0=(tile/16)*(H-32)/7;
    for(int c=0;c<3;++c){double a[1024];
      for(int y=0;y<32;++y)for(int x=0;x<32;++x){const uint8_t* px=data+(y0+y)*stride+(x0+x)*(fmt?8:16);
        double b=fmt?((const __fp16*)px)[0]:((const float*)px)[0],g=fmt?((const __fp16*)px)[1]:((const float*)px)[1],r=fmt?((const __fp16*)px)[2]:((const float*)px)[2];
        double Y=.2126*r+.7152*g+.0722*b;a[y*32+x]=c==0?Y:c==1?(b-Y)/1.8556:(r-Y)/1.5748;
      }
      for(int band=0,side=32;band<4;++band,side/=2){double next[256]={},sum=0;int half=side/2;
        for(int y=0;y<half;++y)for(int x=0;x<half;++x){double aa=a[y*2*side+x*2],b=a[y*2*side+x*2+1],cc=a[(y*2+1)*side+x*2],d=a[(y*2+1)*side+x*2+1];
          double h=(aa-b+cc-d)/4,v=(aa+b-cc-d)/4,z=(aa-b-cc+d)/4;sum+=h*h+v*v+z*z;next[y*half+x]=(aa+b+cc+d)/4;
        }out[c][band]+=sum/(half*half*128);std::copy(next,next+half*half,a);
      }
    }
  }
}

int main(){@autoreleasepool{
  auto dev=MTLCreateSystemDefaultDevice();auto q=[dev newCommandQueue];auto ctx=od_context_create((__bridge void*)dev,(__bridge void*)q,nullptr,0);
  CHECK(ctx);if(!ctx)return 1;
  const int w=512,h=512;auto in=[dev newBufferWithLength:w*h*16 options:MTLResourceStorageModeShared];auto out=[dev newBufferWithLength:w*h*16 options:MTLResourceStorageModeShared];
  ODFrame f={(__bridge void*)in,w,h,w*16,OD_PIXEL_BGRA32F,0},o=f;o.mtlBuffer=(__bridge void*)out;int off=0;
  for(int kind=0;kind<3;++kind){auto s=od_session_create(ctx);ODAnalysis a={};CHECK(od_session_last_analysis(s,&a)==0);auto p=od_params_default();p.waitUntilCompleted=1;p.temporalRadius=0;p.grainFloor=0;
    for(int i=0;i<9;++i){fill((float*)in.contents,w,h,kind,i+1);f.frameKey=i;CHECK(!od_render(s,&f,&off,1,&p,&o,nullptr));}
    CHECK(od_session_last_analysis(s,&a)==1);char r1[32],r2[32];od_analysis_status(&a,&p,r1,r2);printf("fixture %d: %s / %s, tiles=%u, removal=%.2f\n",kind,r1,r2,a.flatTiles,a.removedOverallDb[0]);
    CHECK(strstr(r1,kind==0?"fine":kind==1?"blotchy":"colour"));CHECK(a.flatTiles>=8);CHECK(a.valid&&a.profileValid);CHECK(a.learningFrames==8);CHECK(a.state==OD_ANALYSIS_SETTLED);
    float hist=0;for(float v:a.histogram)hist+=v;CHECK(std::abs(hist-1)<1e-5);if(kind==0)CHECK(a.removedDb[0][0]>5);
    if(kind==0){p.profileHold=1;f.frameKey=10;CHECK(!od_render(s,&f,&off,1,&p,&o,nullptr));CHECK(od_session_last_analysis(s,&a)==1);CHECK(a.state==OD_ANALYSIS_FROZEN);}
    od_session_reset(s);CHECK(od_session_last_analysis(s,&a)==0);od_session_destroy(s);
  }
  // A shadow occluded for the entire recent ring keeps its old profile value
  // but loses its filled-dot claim. No image or algorithm is modified here.
  {std::string err;auto c=ctx->impl;auto stage=od::makeProfileStage(*c,&err);CHECK(stage);
    ODNoiseProfile whole={},fine={},fresh=profile();ODNoiseProfile* slots[]={&whole,&fine};
    for(int k=0;k<24;++k)stage->smooth(fresh,3,slots,false);float shadow=fine.sigmaFull[0][0];
    auto i=stage->analysisInfo(OD_GRAIN_FINEST);CHECK((i.measuredKnots&3)==3);
    for(int k=0;k<32;++k)stage->smooth(fresh,0xf0,slots,false);
    i=stage->analysisInfo(OD_GRAIN_FINEST);CHECK((i.measuredKnots&3)==0);CHECK((i.heldKnots&3)==3);CHECK(fine.sigmaFull[0][0]==shadow);
  }
  // Exact views, both precisions, padded rows, odd width, in-place too.
  for(int format=0;format<2;++format){uint32_t W=513,H=257,bpp=format?8:16,stride=W*bpp+32;size_t bytes=stride*H;
    auto src=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];auto dst=[dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    std::vector<float> values(W*H*4);fill(values.data(),W,H,0,23);memset(src.contents,0x35,bytes);
    for(uint32_t y=0;y<H;++y)for(uint32_t x=0;x<W*4;++x){if(format)((__fp16*)((char*)src.contents+y*stride))[x]=values[y*W*4+x];else((float*)((char*)src.contents+y*stride))[x]=values[y*W*4+x];}
    std::vector<uint8_t> original(bytes);memcpy(original.data(),src.contents,bytes);
    ODFrame fi={(__bridge void*)src,W,H,stride,(ODPixelFormat)format,0},fo=fi;fo.mtlBuffer=(__bridge void*)dst;
    auto s=od_session_create(ctx);auto p=od_params_default();auto manual=profile();p.manualProfile=&manual;p.waitUntilCompleted=1;p.temporalRadius=0;
    p.debugFlags=OD_DEBUG_NO_ANALYSIS;CHECK(!od_render(s,&fi,&off,1,&p,&fo,nullptr));std::vector<uint8_t> ref(bytes);memcpy(ref.data(),dst.contents,bytes);
    p.debugFlags=0;CHECK(!od_render(s,&fi,&off,1,&p,&fo,nullptr));for(uint32_t y=0;y<H;++y)CHECK(!memcmp(ref.data()+y*stride,(char*)dst.contents+y*stride,W*bpp));
    ODAnalysis measured={};CHECK(od_session_last_analysis(s,&measured)==1);CHECK(measured.flatTiles==128);
    double inputE[3][4]={},outputE[3][4]={};energies(original.data(),stride,format,W,H,inputE);energies((const uint8_t*)dst.contents,stride,format,W,H,outputE);
    for(int b=0;b<4;++b){double db=10*std::log10(inputE[0][b]/outputE[0][b]);CHECK(std::abs(measured.removedDb[0][b]-db)<.02);}
    for(bool inplace:{false,true})for(int view:{OD_VIEW_SPLIT,OD_VIEW_ANALYSIS}){memcpy(src.contents,original.data(),bytes);fo.mtlBuffer=(__bridge void*)(inplace?src:dst);p.view=view;CHECK(!od_render(s,&fi,&off,1,&p,&fo,nullptr));
      uint32_t bx,by,bw,bh;od_analysis_box(W,H,&bx,&by,&bw,&bh);auto data=(uint8_t*)(inplace?src:dst).contents;
      for(uint32_t y=0;y<H;++y)for(uint32_t x=0;x<W;++x){size_t k=y*stride+x*bpp;
        if(view==OD_VIEW_SPLIT){if(x<W/2)CHECK(!memcmp(data+k,original.data()+k,bpp));else if(x>W/2)CHECK(!memcmp(data+k,ref.data()+k,bpp));}
        else if(x<bx||x>=bx+bw||y<by||y>=by+bh)CHECK(!memcmp(data+k,ref.data()+k,bpp));
        CHECK(!memcmp(data+k+3*bpp/4,original.data()+k+3*bpp/4,bpp/4));
      }
    }
    p.view=0;p.waitUntilCompleted=0;memcpy(src.contents,original.data(),bytes);fo.mtlBuffer=(__bridge void*)dst;CHECK(!od_render(s,&fi,&off,1,&p,&fo,nullptr));CHECK(!od_debug_session_drain(s));ODAnalysis a;CHECK(od_session_last_analysis(s,&a)==1);CHECK(a.state==OD_ANALYSIS_CAPTURED);
    od_session_reset(s);CHECK(od_session_last_analysis(s,&a)==0);od_session_destroy(s);
  }
  // Row priority, finite byte limits, no scientific or host jargon in drawing.
  {ODAnalysis a={};a.valid=a.profileValid=1;a.confidence=1;a.flatTiles=128;a.state=OD_ANALYSIS_SETTLED;a.removalValid=4095;
    for(int l=0;l<8;++l){a.measured[l]=1;a.histogram[l]=.125f;a.effectiveLuma[l]=.8f;for(int c=0;c<3;++c){a.full[c][l]=2;for(int b=0;b<4;++b)a.profile[c][b][l]=b? .4f:2;}}
    a.removedOverallDb[0]=14;auto p=od_params_default();char r1[32],r2[32];od_analysis_status(nullptr,&p,r1,r2);CHECK(!strcmp(r1,"Noise: measuring..."));
    a.state=OD_ANALYSIS_LEARNING;a.learningFrames=3;a.confidence=0;od_analysis_status(&a,&p,r1,r2);CHECK(!strcmp(r2,"Learning noise (3 of 8)"));
    a.state=OD_ANALYSIS_SETTLED;od_analysis_status(&a,&p,r1,r2);CHECK(!strcmp(r2,"Unsure: few flat areas"));
    a.confidence=1;a.measured[0]=a.measured[1]=0;a.held[0]=a.held[1]=1;od_analysis_status(&a,&p,r1,r2);CHECK(!strcmp(r2,"Shadows unseen: kept last"));
    a.measured[0]=a.measured[1]=1;a.temporalValid=1;a.temporalSpatialRatio=.5f;
    const auto analysisBefore=a; const auto paramsBefore=p;
    od_analysis_status(&a,&p,r1,r2);CHECK(!strcmp(r2,"Phone-like noise: try Phone"));
    CHECK(strlen(r2)<=31 && !memcmp(&a,&analysisBefore,sizeof a) && !memcmp(&p,&paramsBefore,sizeof p));
    a.temporalSpatialRatio=.7f;od_analysis_status(&a,&p,r1,r2);CHECK(strcmp(r2,"Phone-like noise: try Phone"));
    a.temporalValid=0;for(auto& v:a.effectiveLuma)v=0;od_analysis_status(&a,&p,r1,r2);CHECK(!strcmp(r2,"Luma held by Keep grain"));
    for(auto& v:a.effectiveLuma)v=.8f;a.removedOverallDb[0]=5;od_analysis_status(&a,&p,r1,r2);CHECK(!strcmp(r2,"Luma at filter limit"));
    a.removedOverallDb[0]=14;p.view=OD_VIEW_SPLIT;od_analysis_status(&a,&p,r1,r2);CHECK(!strcmp(r2,"View: exports this view"));
    p.view=0;od_analysis_status(&a,&p,r1,r2);CHECK(strstr(r2,"Removed: luma"));CHECK(strlen(r1)<=31&&strlen(r2)<=31);
    ODDrawList list;CHECK(od_analysis_layout(&a,&p,&list,640,420)>0);CHECK(list.count<OD_DRAW_CAPACITY);for(uint32_t k=0;k<list.count;++k)CHECK(!strstr(list.commands[k].text,"sigma"));
  }
  od_context_destroy(ctx);printf("analysis: %d failures\n",failures);return failures?1:0;
}}
