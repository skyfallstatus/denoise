// SPDX-License-Identifier: MIT
// Exact calibration-map equality against the scalar API v11 reference.
// Covers partial tiles, top-level demeaning, varying counts and profile
// knots, negative/super-white values and constant/structured inputs.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>
#include "context.h"
#include "opendenoise/opendenoise.h"
#include "residual_reference.h"

static uint32_t hash(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; return x ^ (x >> 16);
}
static id<MTLTexture> texture(id<MTLDevice> dev, uint32_t w, uint32_t h, int channels, uint32_t seed, int kind) {
  auto d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:channels == 1 ? MTLPixelFormatR16Float : MTLPixelFormatRGBA16Float
                                                           width:w height:h mipmapped:NO];
  d.storageMode = MTLStorageModeShared;
  d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  auto t = [dev newTextureWithDescriptor:d];
  if (!t) return nil;
  std::vector<_Float16> v((size_t)w*h*channels);
  for (size_t i = 0; i < v.size(); ++i) {
    float z = (hash(uint32_t(i) + seed*104729u) >> 8) * (1.0f/16777216.0f);
    float f = channels == 1 ? 1 + 10*z : kind == 0 ? .3f : kind == 1 ? (z-.5f)*.1f + (i%4 == 0 ? .3f : 0) :
              kind == 2 ? (z-.25f)*2 : (z-.5f)*131008;
    v[i] = (_Float16)f;
  }
  [t replaceRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0 withBytes:v.data() bytesPerRow:w*channels*2];
  return t;
}
int main() {
  @autoreleasepool {
    static_assert(OD_LEVELS == 8, "reference's knot count");
    auto dev = MTLCreateSystemDefaultDevice();
    auto queue = [dev newCommandQueue];
    if (!dev || !queue) return 2;
    od::Context ctx(dev, queue);
    std::string error;
    auto opt = ctx.pipeline("spatial_residual_ratio", &error);
    auto options = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) options.mathMode = MTLMathModeSafe;
    else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      options.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    NSError* err = nil;
    auto lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kResidualReference] options:options error:&err];
    auto ref = lib ? [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"spatial_residual_ratio"] error:&err] : nil;
    if (!ref || !opt) { fprintf(stderr,"compile: %s %s\n",error.c_str(),err.description.UTF8String); return 2; }
    int cases = 0, failures = 0;
    const uint32_t sizes[][2] = {{1,1},{7,9},{8,8},{9,15},{31,17},{193,129},{641,359},{1920,1080}};
    for (const auto& size : sizes) for (int top : {0,1}) for (int kind : {0,1,2,3}) for (int scale : {0,1,2}) {
      @autoreleasepool {
        uint32_t w=size[0],h=size[1],rw=(w+7)/8,rh=(h+7)/8;
        id<MTLTexture> tex[6] = {texture(dev,w,h,4,7,kind),texture(dev,(w+1)/2,(h+1)/2,4,8,kind),
          texture(dev,w,h,4,9,kind),texture(dev,(w+1)/2,(h+1)/2,4,10,kind),nil,
          texture(dev,scale ? w*scale : 1,scale ? h*scale : 1,1,11,kind)};
        auto a=texture(dev,rw,rh,4,0,0),b=texture(dev,rw,rh,4,0,0);
        for (int i : {0,1,2,3,5}) if (!tex[i]) return 2;
        if (!a || !b) return 2;
        struct U {float sigma[OD_LEVELS][4]; uint32_t flags[4];} u={};
        u.flags[0]=top;
        for (int l=0;l<OD_LEVELS;++l) for(int c=0;c<3;++c) u.sigma[l][c]=.001f*(1+l*2+c*3);
        auto cb=[queue commandBuffer];
        auto enc=[cb computeCommandEncoder];
        if (!cb || !enc) return 2;
        for (int pass=0;pass<2;++pass) {
          tex[4]=pass ? b : a;
          for(int i=0;i<6;++i) [enc setTexture:tex[i] atIndex:i];
          [enc setBytes:&u length:sizeof(u) atIndex:0];
          [enc setComputePipelineState:pass ? opt : ref];
          if (pass) [enc dispatchThreadgroups:MTLSizeMake(rw,rh,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
          else [enc dispatchThreads:MTLSizeMake(rw,rh,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
        }
        [enc endEncoding];[cb commit];[cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {fprintf(stderr,"GPU error: %s\n",cb.error.description.UTF8String);return 2;}
        std::vector<uint16_t> av((size_t)rw*rh*4),bv(av.size());
        [a getBytes:av.data() bytesPerRow:rw*8 fromRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0];
        [b getBytes:bv.data() bytesPerRow:rw*8 fromRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0];
        ++cases;
        if (av != bv) {++failures;fprintf(stderr,"FAIL %ux%u top=%d kind=%d scale=%d\n",w,h,top,kind,scale);}
      }
    }
    printf("test_residual_exact: %d cases, %d failures\n",cases,failures);
    return failures ? 1 : 0;
  }
}
