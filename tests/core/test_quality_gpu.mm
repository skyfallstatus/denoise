// SPDX-License-Identifier: MIT
// Completion identity, queue latency and explicit output equivalence on real Metal.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstring>
#include "opendenoise/quality.h"
#include "debug.h"
static int failures=0, checks=0;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x); } } while(0)
int main() { @autoreleasepool {
  id<MTLDevice> device=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> queue=[device newCommandQueue];
  char error[256]; ODContext* ctx=od_context_create((__bridge void*)device,(__bridge void*)queue,error,sizeof error);
  if(!ctx) { fprintf(stderr,"%s\n",error); return 1; }
  ODSession* session=od_session_create(ctx);
  const int bytes=64*64*16;
  id<MTLBuffer> input=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> output=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  for(int i=0;i<bytes/4;++i)((float*)input.contents)[i]=float(i%53)/64;
  ODFrame f={(__bridge void*)input,64,64,64*16,OD_PIXEL_BGRA32F,1}, out=f; out.mtlBuffer=(__bridge void*)output;
  ODParams p=od_params_default(); p.lumaStrength=p.chromaStrength=p.sharpen=0; p.temporalRadius=0;
  p.waitUntilCompleted=1;
  int off=0; ODStats st={}; ODQualityDecision d={}; ODQualityRequest r={1,40,0,0};
  CHECK(od_render_quality(session,&f,&off,1,&p,&out,&st,&r,&d)==0);
  CHECK(d.quality==1 && d.budgetMs==0 && memcmp(input.contents,output.contents,bytes)==0);
  ODQualityCompletion c[16]; uint64_t dropped=0;
  CHECK(od_quality_poll(session,c,16,&dropped)==1 && !dropped && c[0].renderId==d.renderId);
  CHECK(od_quality_poll(session,c,16,&dropped)==0); // once, not most-recent repeats
  p.waitUntilCompleted=0; p.realtime=1; r.callerWaitMs=7;
  id<MTLSharedEvent> event=[device newSharedEvent];
  id<MTLCommandBuffer> hold=[queue commandBuffer]; [hold encodeWaitForEvent:event value:1]; [hold commit];
  // A bounded external queue delay. Release even if the render fails.
  dispatch_after(dispatch_time(DISPATCH_TIME_NOW,50*NSEC_PER_MSEC),dispatch_get_global_queue(QOS_CLASS_DEFAULT,0), ^{event.signaledValue=1;});
  r.timelineTimeMs=40;
  CHECK(od_render_quality(session,&f,&off,1,&p,&out,&st,&r,&d)==0);
  CHECK(od_debug_session_drain(session)==0);
  CHECK(od_quality_poll(session,c,16,&dropped)==1);
  CHECK(c[0].renderId==d.renderId && !c[0].failed && c[0].callToFinishMs>40);
  CHECK(c[0].callToFinishMs>c[0].gpuMs+7); // includes queue plus wrapper admission
  CHECK(memcmp(input.contents,output.contents,bytes)==0);
  r.automatic=0; p.quality=0; p.realtime=0; p.waitUntilCompleted=1;
  CHECK(od_render_quality(session,&f,&off,1,&p,&out,&st,&r,&d)==0 && d.quality==0);
  CHECK(od_quality_poll(session,c,16,&dropped)==1 && c[0].quality==0);
  od_session_reset(session); r.automatic=1; p.realtime=1; p.waitUntilCompleted=0;
  CHECK(od_render_quality(session,&f,&off,1,&p,&out,&st,&r,&d)==0 && d.quality==1);
  CHECK(od_debug_session_drain(session)==0);
  od_session_destroy(session); od_context_destroy(ctx);
  printf("quality GPU: %d checks, %d failures\n",checks,failures);
  return failures ? 1 : 0;
} }
