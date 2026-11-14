// SPDX-License-Identifier: MIT
#include "analysis_internal.h"
#include "analysis_layout.h"
#import <CoreGraphics/CoreGraphics.h>
#import <CoreText/CoreText.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace od {
std::unique_ptr<AnalysisStage> AnalysisStage::make(Context& c,std::string* err){
  auto s=std::unique_ptr<AnalysisStage>(new AnalysisStage);
  s->tiles_=c.pipeline("analysis_tiles",err);if(!s->tiles_)return nullptr;
  s->histogram_=c.pipeline("analysis_histogram",err);s->reduce_=c.pipeline("analysis_reduce",err);
  s->overlay32_=c.pipeline("analysis_overlay32",err);s->overlay16_=c.pipeline("analysis_overlay16",err);
  return s->histogram_&&s->reduce_&&s->overlay32_&&s->overlay16_?std::move(s):nullptr;
}
AnalysisReadback AnalysisStage::encode(RenderEnv& env,const ODFrame& source,id<MTLTexture> result,const ProfileAnalysisInfo& info,
                                     uint64_t renderId,bool frozen,bool captured){
  AnalysisReadback r;ODAnalysis& a=r.snapshot;
  a.valid=1;a.profileValid=env.profile.valid;a.renderId=renderId;a.frameKey=env.frames[0].frameKey;
  a.width=env.width;a.height=env.height;a.confidence=env.profile.confidence;
  a.learningFrames=std::min(8u,info.ringCount);
  a.state=captured?OD_ANALYSIS_CAPTURED:frozen?OD_ANALYSIS_FROZEN:info.ringCount>=8?OD_ANALYSIS_SETTLED:
    info.ringCount?OD_ANALYSIS_LEARNING:OD_ANALYSIS_AUTOMATIC;
  for(int l=0;l<8;++l){a.measured[l]=!captured&&((info.measuredKnots>>l)&1);
    a.held[l]=!captured&&((info.heldKnots>>l)&1);
    float sig=lumaSigmaFull(env.profile,l),amount=env.params.lumaStrength;
    float effective=std::min(amount,std::max(0.f,1-env.grainFloor()/std::max(sig,1e-10f)));
    a.effectiveLuma[l]=effective*env.params.bandAmount[0][0];
    for(int c=0;c<3;++c){a.full[c][l]=876*env.profile.sigmaFull[c][l];for(int b=0;b<4;++b)a.profile[c][b][l]=876*env.profile.sigma[c][b][l];}
  }
  struct Tile {uint32_t x,y;float fineMax,mediumMax;};
  Tile ts[128]={};uint32_t count=info.tileCount;
  for(uint32_t k=0;k<count;++k){ts[k]={info.xy[k][0],info.xy[k][1],std::max(1e-12f,3*info.energy[k][0]),std::max(1e-12f,12*info.energy[k][1])};}
  // Captured profiles have no tile provenance in ODNoiseProfile. Requalify
  // a bounded subset of the profiler's 32px raster against its noise model.
  if(captured&&env.width>=32&&env.height>=32){count=128;
    for(uint32_t k=0;k<count;++k){uint32_t x=(k%16)*(env.width-32)/15,y=(k/16)*(env.height-32)/7;
      float mx=0,mm=0;for(int l=0;l<8;++l){mx=std::max(mx,env.profile.sigma[0][0][l]);mm=std::max(mm,env.profile.sigma[0][1][l]);}
      ts[k]={x,y,std::max(1e-12f,3*mx*mx),std::max(1e-12f,12*mm*mm)};
    }
  }
  id<MTLTexture> prev=env.frames[0].tex;uint32_t pair=0;
  for(int k=1;k<env.frameCount;++k)if(std::abs(env.frames[k].offset)==1){prev=env.frames[k].tex;pair=1;break;}
  a.temporalValid=pair;
  r.data=[env.device() newBufferWithLength:40*sizeof(float) options:MTLResourceStorageModeShared];
  r.scratch=[env.device() newBufferWithLength:(4096+256)*sizeof(float) options:MTLResourceStorageModePrivate];
  r.tiles=[env.device() newBufferWithBytes:ts length:sizeof(ts) options:MTLResourceStorageModeShared];
  if(!r.data||!r.scratch||!r.tiles){env.fail("analysis: allocation failed");return r;}
  struct {uint32_t tiles,pair,width,height,stride,isHalf;} u={count,pair,env.width,env.height,source.rowBytes,source.format==OD_PIXEL_BGRA16F?1u:0u};
  auto e=env.compute("analysis_tiles");[e setComputePipelineState:tiles_];
  [e setTexture:env.frames[0].tex atIndex:0];[e setTexture:result atIndex:1];[e setTexture:prev atIndex:2];
  [e setBuffer:(__bridge id<MTLBuffer>)source.mtlBuffer offset:0 atIndex:3];
  [e setBuffer:r.tiles offset:0 atIndex:0];[e setBuffer:r.scratch offset:0 atIndex:1];[e setBytes:&u length:sizeof(u) atIndex:2];
  if(count)[e dispatchThreadgroups:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  e=env.compute("analysis_histogram");[e setComputePipelineState:histogram_];[e setTexture:env.frames[0].tex atIndex:0];[e setBuffer:r.scratch offset:0 atIndex:0];
  [e dispatchThreadgroups:MTLSizeMake(32,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  e=env.compute("analysis_reduce");[e setBuffer:r.scratch offset:0 atIndex:0];[e setBuffer:r.data offset:0 atIndex:1];[e setBytes:&u length:sizeof(u) atIndex:2];env.dispatch(reduce_,40,1);
  return r;
}
ODAnalysis AnalysisStage::resolve(const AnalysisReadback& r){
  ODAnalysis a=r.snapshot;if(!r.data)return a;
  const float* v=(const float*)r.data.contents;a.flatTiles=(uint32_t)std::max(0.f,v[24]);
  for(int l=0;l<8;++l)a.histogram[l]=v[32+l];
  auto db=[](double i,double o){return i>1e-18?(float)std::clamp(10*std::log10(i/std::max(o,1e-30)),-30.,60.):0.f;};
  for(int c=0;c<3;++c){double in=0,out=0;for(int b=0;b<4;++b){float i=v[c*4+b],o=v[12+c*4+b];
    a.inputEnergy[c][b]=a.flatTiles?i/a.flatTiles:0;a.outputEnergy[c][b]=a.flatTiles?o/a.flatTiles:0;
    if(a.flatTiles>=8&&i>1e-18){a.removalValid|=1u<<(c*4+b);a.removedDb[c][b]=db(i,o);}in+=i;out+=o;
  }a.removedOverallDb[c]=db(in,out);}
  a.temporalValid=a.temporalValid&&a.flatTiles>=8&&v[25]>1e-18;
  a.temporalSpatialRatio=a.temporalValid?std::sqrt(std::max(0.f,v[26]/v[25])):0;
  return a;
}
bool AnalysisStage::overlay(RenderEnv& env,const ODFrame& out,const ODAnalysis* a){
  uint32_t x,y,w,h;analysisViewBox(a,out.width,out.height,&x,&y,&w,&h);if(!w||!h)return true;
  // This CPU rasterisation exists only in Analysis view. Core and Drawbot
  // share the drawing vocabulary; the in-picture layout reflows for Fit.
  auto list=std::make_unique<ODDrawList>();analysisViewLayout(a,&env.params,list.get(),w,h);
  std::vector<uint8_t> pixels((size_t)w*h*4);
  CGColorSpaceRef cs=CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  CGContextRef cg=CGBitmapContextCreate(pixels.data(),w,h,8,w*4,cs,kCGImageAlphaPremultipliedLast|kCGBitmapByteOrder32Big);
  CGColorSpaceRelease(cs);if(!cg){env.fail("analysis: cannot create drawing context");return false;}
  CGContextTranslateCTM(cg,0,h);CGContextScaleCTM(cg,1,-1);
  for(uint32_t k=0;k<list->count;++k){const auto& d=list->commands[k];
    CGContextSetRGBFillColor(cg,d.rgba[0],d.rgba[1],d.rgba[2],d.rgba[3]);CGContextSetRGBStrokeColor(cg,d.rgba[0],d.rgba[1],d.rgba[2],d.rgba[3]);CGContextSetLineWidth(cg,d.stroke);
    CGFloat dash=d.dash;CGContextSetLineDash(cg,0,&dash,d.dash>0?1:0);
    if(d.kind==OD_DRAW_LINE){CGContextBeginPath(cg);CGContextMoveToPoint(cg,d.x,d.y);CGContextAddLineToPoint(cg,d.w,d.h);CGContextStrokePath(cg);}
    else if(d.kind==OD_DRAW_RECT){CGRect b=CGRectMake(d.x,d.y,d.w,d.h);if(d.stroke>0)CGContextStrokeRect(cg,b);else CGContextFillRect(cg,b);}
    else if(d.kind==OD_DRAW_DOT){CGRect b=CGRectMake(d.x-d.w,d.y-d.w,d.w*2,d.w*2);if(d.stroke>0)CGContextStrokeEllipseInRect(cg,b);else CGContextFillEllipseInRect(cg,b);}
    else {CFStringRef str=CFStringCreateWithCString(nullptr,d.text,kCFStringEncodingUTF8);CTFontRef font=CTFontCreateWithName(CFSTR("Helvetica"),d.fontSize,nullptr);
      const void* keys[]={kCTFontAttributeName,kCTForegroundColorFromContextAttributeName};const void* vals[]={font,kCFBooleanTrue};
      CFDictionaryRef attrs=CFDictionaryCreate(nullptr,keys,vals,2,&kCFTypeDictionaryKeyCallBacks,&kCFTypeDictionaryValueCallBacks);
      CFAttributedStringRef as=CFAttributedStringCreate(nullptr,str,attrs);CTLineRef line=CTLineCreateWithAttributedString(as);
      CGContextSaveGState(cg);CGContextTranslateCTM(cg,d.x,d.y+CTFontGetAscent(font));CGContextScaleCTM(cg,1,-1);CGContextSetTextMatrix(cg,CGAffineTransformIdentity);CGContextSetTextPosition(cg,0,0);CTLineDraw(line,cg);CGContextRestoreGState(cg);
      CFRelease(line);CFRelease(as);CFRelease(attrs);CFRelease(font);CFRelease(str);
    }
  }
  CGContextRelease(cg);
  auto desc=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:w height:h mipmapped:NO];desc.storageMode=MTLStorageModeShared;desc.usage=MTLTextureUsageShaderRead;
  id<MTLTexture> texture=[env.device() newTextureWithDescriptor:desc];if(!texture){env.fail("analysis: texture allocation failed");return false;}
  [texture replaceRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:0 withBytes:pixels.data() bytesPerRow:w*4];
  struct{uint32_t stride,x,y,w,h;}u={out.rowBytes/(out.format==OD_PIXEL_BGRA16F?2u:4u),x,y,w,h};
  auto e=env.compute("analysis_overlay");[e setBuffer:(__bridge id<MTLBuffer>)out.mtlBuffer offset:0 atIndex:0];[e setBytes:&u length:sizeof(u) atIndex:1];[e setTexture:texture atIndex:0];env.dispatch(out.format==OD_PIXEL_BGRA16F?overlay16_:overlay32_,w,h);
  return !env.failed();
}
}
