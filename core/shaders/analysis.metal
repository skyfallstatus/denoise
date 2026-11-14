// SPDX-License-Identifier: MIT
// Bounded observation only: up to 128 profile tiles, 32x32 pixels each.
// A separable Haar decomposition is calculated in threadgroup memory. Its
// three detail orientations are summed at each of four octave sizes. No
// image is written; the CPU reads just 40 floats after GPU completion.
struct AnalysisTile { uint2 xy; float fineMax, mediumMax; };
struct AnalysisUniforms { uint tiles, pair, width, height, stride, isHalf; };
static float4 haar_energy(float4 a,float4 b,float4 c,float4 d) {
  float4 h=(a-b+c-d)*.25f,v=(a+b-c-d)*.25f,z=(a-b-c+d)*.25f;
  return h*h+v*v+z*z;
}
static float4 group_sum(float4 v, threadgroup float4* sums,uint tid,uint lane,uint group) {
  v=simd_sum(v);if(lane==0)sums[group]=v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  v=tid<8?sums[tid]:float4(0);v=simd_sum(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return v;
}
kernel void analysis_tiles(texture2d<float,access::read> src [[texture(0)]],
                            texture2d<float,access::read> result [[texture(1)]],
                            texture2d<float,access::read> prev [[texture(2)]],
                            const device AnalysisTile* tiles [[buffer(0)]],
                            device float* partial [[buffer(1)]],
                            constant AnalysisUniforms& u [[buffer(2)]],
                            const device uchar* host [[buffer(3)]],
                            uint tile [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float4 si[256],so[256],sd[256],sums[8];
  threadgroup float4 ei[4],eo[4],ed[4];
  const AnalysisTile t=tiles[tile];
  float4 va[4],vb[4],vc[4];
  uint2 base=t.xy+uint2(tid%16,tid/16)*2;
  for(uint k=0;k<4;++k){uint2 q=min(base+uint2(k%2,k/2),uint2(u.width-1,u.height-1));
    va[k]=src.read(q);vb[k]=result.read(q);vc[k]=u.pair?(va[k]-prev.read(q))*.70710678f:float4(0);
    // Reconstruct the actual host Result arithmetic before output overwrites
    // an in-place source. Includes 16f output rounding and 32f source precision.
    float3 rgb=u.isHalf?float3(((const device packed_half4*)(host+q.y*u.stride))[q.x].xyz):float3(((const device packed_float4*)(host+q.y*u.stride))[q.x].xyz);
    float3 outRGB=rgb+od_ycc_to_rgb(vb[k].xyz-va[k].xyz).zyx;
    if(u.isHalf)outRGB=float3(half3(clamp(outRGB,-OD_HALF_MAX,OD_HALF_MAX)));
    bool valid=all(isfinite(rgb))&&all(isfinite(outRGB));
    va[k].xyz=valid?od_rgb_to_ycc(rgb.zyx):float3(NAN);
    vb[k].xyz=valid?od_rgb_to_ycc(outRGB.zyx):float3(NAN);
    va[k].w=vb[k].w=vc[k].w=0;
  }
  float4 e=group_sum(haar_energy(va[0],va[1],va[2],va[3]),sums,tid,lane,sg);
  if(tid==0)ei[0]=e/256;
  e=group_sum(haar_energy(vb[0],vb[1],vb[2],vb[3]),sums,tid,lane,sg);if(tid==0)eo[0]=e/256;
  e=group_sum(haar_energy(vc[0],vc[1],vc[2],vc[3]),sums,tid,lane,sg);if(tid==0)ed[0]=e/256;
  si[tid]=(va[0]+va[1]+va[2]+va[3])*.25f;so[tid]=(vb[0]+vb[1]+vb[2]+vb[3])*.25f;sd[tid]=(vc[0]+vc[1]+vc[2]+vc[3])*.25f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint band=1;band<4;++band){uint side=16>>band,n=side*side;
    for(uint k=0;k<4;++k){uint q=(tid/side*2+k/2)*(side*2)+(tid%side*2+k%2);
      va[k]=tid<n?si[q]:float4(0);vb[k]=tid<n?so[q]:float4(0);vc[k]=tid<n?sd[q]:float4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    e=group_sum(haar_energy(va[0],va[1],va[2],va[3]),sums,tid,lane,sg);if(tid==0)ei[band]=e/n;
    e=group_sum(haar_energy(vb[0],vb[1],vb[2],vb[3]),sums,tid,lane,sg);if(tid==0)eo[band]=e/n;
    e=group_sum(haar_energy(vc[0],vc[1],vc[2],vc[3]),sums,tid,lane,sg);if(tid==0)ed[band]=e/n;
    if(tid<n){si[tid]=(va[0]+va[1]+va[2]+va[3])*.25f;so[tid]=(vb[0]+vb[1]+vb[2]+vb[3])*.25f;sd[tid]=(vc[0]+vc[1]+vc[2]+vc[3])*.25f;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if(tid==0){bool good=ei[0].x<=t.fineMax && ei[1].x<=t.mediumMax;
    for(uint b=0;b<4;++b)good=good&&all(isfinite(ei[b]))&&all(isfinite(eo[b]));
    for(uint c=0;c<3;++c)for(uint b=0;b<4;++b){partial[tile*32+c*4+b]=good?ei[b][c]:0;partial[tile*32+12+c*4+b]=good?eo[b][c]:0;}
    partial[tile*32+24]=good?1:0;
    // Finest octave isolates grain from static shading/soft structure.
    float input=ei[0].x,diff=ed[0].x;
    partial[tile*32+25]=good&&u.pair?input:0;partial[tile*32+26]=good&&u.pair?diff:0;
    for(uint k=27;k<32;++k)partial[tile*32+k]=0;
  }
}
kernel void analysis_histogram(texture2d<float,access::read> src [[texture(0)]],
                                device float* partial [[buffer(0)]],
                                uint tid [[thread_index_in_threadgroup]],uint block [[threadgroup_position_in_grid]]) {
  threadgroup atomic_uint bins[8];if(tid<8)atomic_store_explicit(&bins[tid],0,memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint k=block*256+tid;
  uint2 q=uint2((k%128+.5f)*src.get_width()/128,(k/128+.5f)*src.get_height()/64);
  float y=src.read(min(q,uint2(src.get_width()-1,src.get_height()-1))).x;
  uint bin=min(7u,uint(clamp(y,0.f,1.f)*8));atomic_fetch_add_explicit(&bins[bin],1,memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(tid<8)partial[4096+block*8+tid]=atomic_load_explicit(&bins[tid],memory_order_relaxed);
}
kernel void analysis_reduce(const device float* partial [[buffer(0)]],device float* result [[buffer(1)]],
                             constant AnalysisUniforms& u [[buffer(2)]],uint k [[thread_position_in_grid]]) {
  if(k>=40)return;float sum=0;
  if(k<32){for(uint i=0;i<u.tiles;++i)sum+=partial[i*32+k];}
  else {for(uint i=0;i<32;++i)sum+=partial[4096+i*8+k-32];sum/=8192;}
  result[k]=sum;
}
struct AnalysisOverlay { uint stride,x,y,w,h; };
kernel void analysis_overlay32(device float* dst [[buffer(0)]],constant AnalysisOverlay& u [[buffer(1)]],
                                texture2d<float,access::read> panel [[texture(0)]],uint2 q [[thread_position_in_grid]]) {
  if(q.x>=u.w||q.y>=u.h)return;float4 c=panel.read(q);uint i=(q.y+u.y)*u.stride+(q.x+u.x)*4;
  for(uint k=0;k<3;++k)dst[i+k]=c[2-k]+(1-c.w)*dst[i+k];
}
kernel void analysis_overlay16(device half* dst [[buffer(0)]],constant AnalysisOverlay& u [[buffer(1)]],
                                texture2d<float,access::read> panel [[texture(0)]],uint2 q [[thread_position_in_grid]]) {
  if(q.x>=u.w||q.y>=u.h)return;float4 c=panel.read(q);uint i=(q.y+u.y)*u.stride+(q.x+u.x)*4;
  for(uint k=0;k<3;++k)dst[i+k]=half(c[2-k]+(1-c.w)*float(dst[i+k]));
}
