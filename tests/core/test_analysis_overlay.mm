// SPDX-License-Identifier: MIT
// Actual overlay at HD/UHD, both orientations/precisions. Fixed <=4K rasters,
// <=600 MiB host buffers per case plus a 1024 MiB core budget. No host apps.
// Optional caller-selected PNG output; four full frames and four 500px Fits.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreText/CoreText.h>
#import <ImageIO/ImageIO.h>
#include "opendenoise/opendenoise.h"
#include "analysis_layout.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
#include <unistd.h>
static int failures=0,checks=0;
#define CHECK(x) do{++checks;if(!(x)){fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);++failures;}}while(0)
static bool savePNG(const std::string& path,CGImageRef image) {
  CFURLRef url=CFURLCreateFromFileSystemRepresentation(nullptr,(const UInt8*)path.c_str(),path.size(),false);
  CGImageDestinationRef dest=CGImageDestinationCreateWithURL(url,CFSTR("public.png"),1,nullptr);
  bool ok=false;if(dest&&image){CGImageDestinationAddImage(dest,image,nullptr);ok=CGImageDestinationFinalize(dest);}
  if(dest)CFRelease(dest);if(url)CFRelease(url);return ok;
}
static void artifacts(const char* folder,const uint8_t* bgra,uint32_t W,uint32_t H,uint32_t stride) {
  std::filesystem::create_directories(folder);
  std::vector<uint8_t> rgba(size_t(W)*H*4);
  for(uint32_t y=0;y<H;++y)for(uint32_t x=0;x<W;++x){auto p=(const float*)(bgra+y*stride)+x*4;
    for(int c=0;c<3;++c)rgba[(size_t(y)*W+x)*4+c]=std::lround(std::clamp(p[2-c],0.f,1.f)*255);
    rgba[(size_t(y)*W+x)*4+3]=255;
  }
  auto cs=CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  auto cg=CGBitmapContextCreate(rgba.data(),W,H,8,W*4,cs,kCGImageAlphaNoneSkipLast|kCGBitmapByteOrder32Big);
  CHECK(cg);if(!cg){CGColorSpaceRelease(cs);return;}
  auto image=CGBitmapContextCreateImage(cg);
  const std::string base=std::string(folder)+"/analysis-"+std::to_string(W)+"x"+std::to_string(H);
  CHECK(savePNG(base+".png",image));
  const uint32_t fitH=std::lround(double(H)*500/W);
  auto fit=CGBitmapContextCreate(nullptr,500,fitH,8,500*4,cs,kCGImageAlphaNoneSkipLast|kCGBitmapByteOrder32Big);
  CHECK(fit);if(fit){CGContextSetInterpolationQuality(fit,kCGInterpolationHigh);CGContextDrawImage(fit,CGRectMake(0,0,500,fitH),image);
    auto preview=CGBitmapContextCreateImage(fit);CHECK(savePNG(base+"-fit500.png",preview));CGImageRelease(preview);CGContextRelease(fit);}
  CGImageRelease(image);CGContextRelease(cg);CGColorSpaceRelease(cs);
}
int main(){alarm(120);setenv("OD_MEMORY_MB","1024",1);@autoreleasepool {
  auto device=MTLCreateSystemDefaultDevice();auto queue=[device newCommandQueue];
  auto ctx=od_context_create((__bridge void*)device,(__bridge void*)queue,nullptr,0);CHECK(ctx);if(!ctx)return 1;
  const uint32_t sizes[][2]={{1920,1080},{1080,1920},{3840,2160},{2160,3840}};
  // Empty overlays keep the same readable type scale, but their rectangle
  // fits only the two lines. Native Effect Controls retains its own layout.
  for(const auto& size:sizes){
    uint32_t fx,fy,fw,fh;od_analysis_box(size[0],size[1],&fx,&fy,&fw,&fh);
    ODAnalysis missing{};missing.valid=1;
    for(const ODAnalysis* a:{(const ODAnalysis*)nullptr,(const ODAnalysis*)&missing}){
      uint32_t x,y,w,h;od::analysisViewBox(a,size[0],size[1],&x,&y,&w,&h);
      CHECK(x==fx&&y+h==fy+fh&&w<fw&&h<fh/4);
      ODDrawList list{};CHECK(od::analysisViewLayout(a,nullptr,&list,w,h)==4);
      CHECK(std::abs(list.commands[0].w-w)<.001f&&std::abs(list.commands[0].h-h)<.001f);
      for(uint32_t i=2;i<4;++i){const auto& d=list.commands[i];
        CHECK(d.kind==OD_DRAW_TEXT);CHECK(!strstr(d.text,"Play or click"));
        auto text=CFStringCreateWithCString(nullptr,d.text,kCFStringEncodingUTF8);
        auto font=CTFontCreateWithName(CFSTR("Helvetica"),d.fontSize,nullptr);
        const void* keys[]={kCTFontAttributeName};const void* vals[]={font};
        auto attrs=CFDictionaryCreate(nullptr,keys,vals,1,&kCFTypeDictionaryKeyCallBacks,&kCFTypeDictionaryValueCallBacks);
        auto str=CFAttributedStringCreate(nullptr,text,attrs);auto line=CTLineCreateWithAttributedString(str);
        CHECK(d.x+CTLineGetTypographicBounds(line,nullptr,nullptr,nullptr)<w);
        CHECK(d.y+CTFontGetAscent(font)+CTFontGetDescent(font)<h);
        CHECK(d.fontSize*500/size[0]>=9);
        CFRelease(line);CFRelease(str);CFRelease(attrs);CFRelease(font);CFRelease(text);
      }
      printf("empty %ux%u box=%u,%u %ux%u\n",size[0],size[1],x,y,w,h);
    }
  }
  for(const auto& size:sizes)for(int format:{0,1}){@autoreleasepool {
    const uint32_t W=size[0],H=size[1],bpp=format?8:16,stride=W*bpp+32;const size_t bytes=size_t(stride)*H;
    CHECK(bytes*4<600ULL*1024*1024);if(bytes*4>=600ULL*1024*1024)return 1;
    auto input=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    auto output=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];CHECK(input&&output);if(!input||!output)return 1;
    memset(input.contents,0x35,bytes);memset(output.contents,0x35,bytes);
    for(uint32_t y=0;y<H;++y)for(uint32_t x=0;x<W;++x){uint32_t r=(x*73+y*193)^((x+1)*(y+13)*977);
      float noise=(float(r%127)/126-.5f)*.035f;
      float px[]={.10f+.45f*x/W+noise,.10f+.45f*y/H+noise,.15f+.35f*x/W+noise,.25f+.5f*y/H};
      for(int c=0;c<4;++c)if(format)((__fp16*)((char*)input.contents+y*stride))[x*4+c]=px[c];else((float*)((char*)input.contents+y*stride))[x*4+c]=px[c];
    }
    std::vector<uint8_t> original(bytes),reference(bytes);memcpy(original.data(),input.contents,bytes);
    ODNoiseProfile manual{};manual.valid=1;manual.confidence=1;const float bands[]={.8908f,.2007f,.0855f,.0412f};
    for(int c=0;c<3;++c)for(int l=0;l<8;++l){float n=(c?.004f:.008f)*(1+.5f*(7-l)/7);manual.sigmaFull[c][l]=n;
      for(int b=0;b<4;++b)manual.sigma[c][b][l]=n*bands[b];}
    auto p=od_params_default();p.manualProfile=&manual;p.waitUntilCompleted=1;p.temporalRadius=0;
    auto session=od_session_create(ctx);CHECK(session);if(!session)return 1;
    ODFrame in{(__bridge void*)input,W,H,stride,(ODPixelFormat)format,OD_NO_KEY},out=in;out.mtlBuffer=(__bridge void*)output;int offset=0;
    CHECK(od_render(session,&in,&offset,1,&p,&out,nullptr)==0);memcpy(reference.data(),output.contents,bytes);
    ODAnalysis a{};CHECK(od_session_last_analysis(session,&a)==1&&a.profileValid);
    uint32_t bx,by,bw,bh;od_analysis_box(W,H,&bx,&by,&bw,&bh);
    CHECK(bx>=W*.05&&by>=H*.05&&bx+bw<=W*.95&&by+bh<=H*.95);
    CHECK(std::abs(double(bw)/W-(W<H?.70:.40))<.001);
    auto list=std::make_unique<ODDrawList>();CHECK(od::analysisViewLayout(&a,&p,list.get(),bw,bh)>0);
    float smallest=1000;std::vector<CGRect> textBoxes;
    for(uint32_t i=0;i<list->count;++i){const auto& d=list->commands[i];if(d.kind!=OD_DRAW_TEXT||!d.text[0])continue;
      smallest=std::min(smallest,d.fontSize*500/W);
      auto text=CFStringCreateWithCString(nullptr,d.text,kCFStringEncodingUTF8);auto font=CTFontCreateWithName(CFSTR("Helvetica"),d.fontSize,nullptr);
      const void* keys[]={kCTFontAttributeName};const void* values[]={font};
      auto attrs=CFDictionaryCreate(nullptr,keys,values,1,&kCFTypeDictionaryKeyCallBacks,&kCFTypeDictionaryValueCallBacks);
      auto str=CFAttributedStringCreate(nullptr,text,attrs);auto line=CTLineCreateWithAttributedString(str);
      const double textWidth=CTLineGetTypographicBounds(line,nullptr,nullptr,nullptr);
      CHECK(d.x+textWidth<=bw+1);
      for(const auto& box:textBoxes)if(std::abs(box.origin.y-d.y)<.01)
        CHECK(d.x>=box.origin.x+box.size.width+1 || box.origin.x>=d.x+textWidth+1);
      textBoxes.push_back(CGRectMake(d.x,d.y,textWidth,d.fontSize));
      CHECK(d.y+CTFontGetAscent(font)+CTFontGetDescent(font)<=bh+1);
      CFRelease(line);CFRelease(str);CFRelease(attrs);CFRelease(font);CFRelease(text);
    }
    CHECK(smallest>=9);
    for(bool inplace:{false,true}) {
      memcpy(input.contents,original.data(),bytes);out.mtlBuffer=(__bridge void*)(inplace?input:output);p.view=OD_VIEW_ANALYSIS;
      CHECK(od_render(session,&in,&offset,1,&p,&out,nullptr)==0);
      const auto data=(const uint8_t*)(inplace?input:output).contents;
      uint64_t outsideBad=0,alphaBad=0,padBad=0,changedInside=0;
      for(uint32_t y=0;y<H;++y){for(uint32_t x=0;x<W;++x){const size_t k=size_t(y)*stride+x*bpp;
        bool changed=memcmp(data+k,reference.data()+k,bpp)!=0;
        if(x<bx||x>=bx+bw||y<by||y>=by+bh)outsideBad+=changed;else changedInside+=changed;
        alphaBad+=memcmp(data+k+3*bpp/4,original.data()+k+3*bpp/4,bpp/4)!=0;
      }padBad+=memcmp(data+size_t(y)*stride+W*bpp,original.data()+size_t(y)*stride+W*bpp,stride-W*bpp)!=0;}
      CHECK(!outsideBad&&!alphaBad&&!padBad&&changedInside>0);
      printf("overlay %ux%u %s %s box=%u,%u %ux%u outside=%llu alpha=%llu padding=%llu min-fit-font=%.2fpx\n",W,H,format?"16f":"32f",inplace?"in-place":"separate",bx,by,bw,bh,(unsigned long long)outsideBad,(unsigned long long)alphaBad,(unsigned long long)padBad,smallest);
      if(!format&&!inplace)if(const char* folder=getenv("OD_ANALYSIS_ARTIFACT_DIR"))artifacts(folder,data,W,H,stride);
    }
    od_session_destroy(session);
  }}
  for(auto size:std::vector<std::pair<uint32_t,uint32_t>>{{0,0},{1,1},{17,9},{9,17},{4096,2160},{2160,4096}}){
    uint32_t x,y,w,h;od_analysis_box(size.first,size.second,&x,&y,&w,&h);CHECK(x+w<=size.first&&y+h<=size.second);
  }
  od_context_destroy(ctx);printf("analysis overlay: %d checks, %d failures\n",checks,failures);return failures?1:0;
}}
