// SPDX-License-Identifier: MIT
// Compare the reduced GPU record with the original double-precision projection.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "context.h"
#include "stages/stage.h"
#include "stages/coarse_profile.h"
#include "coarse_reference.h"

int main() {
  @autoreleasepool {
    auto dev=MTLCreateSystemDefaultDevice();
    auto queue=[dev newCommandQueue];
    od::Context ctx(dev,queue);
    std::string err;
    auto pso=ctx.pipeline("profile_coarse_stats",&err);
    if (!pso) { fprintf(stderr,"%s\n",err.c_str()); return 1; }
    constexpr int W=133,H=103,NN=W*H, NX=(W-4)/16,NY=(H-4)/16;
    int fails=0, decisions=0;
    double maxSigma=0,maxConf=0;
    for (int mode=0; mode<8; ++mode) {
      CoarseMeasurement m; CoarseReferenceMeasurement ref;
      m.frames=ref.frames=3;
      m.data=[dev newBufferWithLength:7*NX*NY*20*4 options:MTLResourceStorageModeShared];
      ref.data=[dev newBufferWithLength:3*7*NN*16 options:MTLResourceStorageModeShared];
      auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:W height:H mipmapped:NO];
      desc.storageMode=MTLStorageModeShared; desc.usage=MTLTextureUsageShaderRead;
      id<MTLTexture> tex[3],zero=[dev newTextureWithDescriptor:desc];
      std::vector<float> z(NN*4,0);
      [zero replaceRegion:MTLRegionMake2D(0,0,W,H) mipmapLevel:0 withBytes:z.data() bytesPerRow:W*16];
      std::mt19937 rng(16381); std::normal_distribution<float> normal(0,1);
      std::vector<float> noise(NN*3),values(NN*4);
      for (auto& n:noise) n=normal(rng);
      for (int t=0;t<3;++t) {
        tex[t]=[dev newTextureWithDescriptor:desc];
        for (int y=0;y<H;++y) for(int x=0;x<W;++x) {
          int i=y*W+x;
          for(int c=0;c<3;++c) {
            float u=x/float(W),v=y/float(H);
            float n=normal(rng), rho=mode==1?0:mode==2?1:.8f;
            float amp=mode==3?1e-5f:.003f;
            values[i*4+c]=(c==0?.4f:.03f)+.02f*u+.04f*v*v + (c==0?.0001f:amp)*(rho*noise[i*3+c]+sqrt(1-rho*rho)*n);
            if(mode==4 && x%16==8 && y%16==8 && c>0) values[i*4+c]+=.1f;
            if(mode==5 && x==7 && y==7) values[i*4+c]=NAN;
            if(mode==6 && x==7 && y==7) values[i*4+c]=16;
            if(mode==7 && c>0) values[i*4+c]+=.6f;
          }
          values[i*4+3]=values[i*4];
        }
        [tex[t] replaceRegion:MTLRegionMake2D(0,0,W,H) mipmapLevel:0 withBytes:values.data() bytesPerRow:W*16];
        for(int b=0;b<7;++b) {
          ref.base[t][b]=(t*7+b)*NN;
          memcpy((float*)ref.data.contents+ref.base[t][b]*4,values.data(),NN*16);
        }
      }
      auto cb=[queue commandBuffer]; auto e=[cb computeCommandEncoder];
      [e setComputePipelineState:pso];
      for(int b=0;b<7;++b) {
        m.w[b]=ref.w[b]=W; m.h[b]=ref.h[b]=H; m.base[b]=b*NX*NY;
        for(int t=0;t<3;++t) { [e setTexture:tex[t] atIndex:2*t]; [e setTexture:zero atIndex:2*t+1]; }
        uint32_t cfg[4]={m.base[b],b<4?0u:1u,NX,0};
        [e setBuffer:m.data offset:0 atIndex:0]; [e setBytes:cfg length:16 atIndex:1];
        [e dispatchThreadgroups:MTLSizeMake(NX,NY,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      }
      [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
      if(cb.status==MTLCommandBufferStatusError) return 2;
      ODNoiseProfile a={},b={};
      for(int c=0;c<3;++c) for(int band=0;band<4;++band) for(int l=0;l<8;++l) a.sigma[c][band][l]=1e-5;
      b=a;
      resolveCoarse(m,&a); resolveCoarseReference(ref,&b);
      for(int band=0;band<7;++band) for(int l=0;l<8;++l) {
        bool same=od::coarseQualified(a,band,l)==od::coarseQualified(b,band,l); decisions+=same;
        if(!same) ++fails;
        double d=fabs(a.coarseConfidence[band][l]-b.coarseConfidence[band][l]); maxConf=std::max(maxConf,d);
        if(d>2e-6) ++fails;
        for(int c=0;c<2;++c) {
          double d=fabs(a.coarseSigma[c][band][l]-b.coarseSigma[c][band][l]);
          maxSigma=std::max(maxSigma,d);
          if(d>1e-8+5e-6*b.coarseSigma[c][band][l]) ++fails;
        }
      }
    }
    printf("coarse GPU/reference: decisions %d/448, max sigma delta %.9g, max confidence delta %.9g, failures %d\n",decisions,maxSigma,maxConf,fails);
    return fails?1:0;
  }
}
