// SPDX-License-Identifier: MIT
#include "opendenoise/profiling.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
namespace {
constexpr size_t kValues = 3 * (OD_BANDS + 1) * OD_LEVELS + 1;
constexpr size_t kBytes = 16 + 4 * kValues + 4;
void put32(uint8_t* p, uint32_t v) { for (int i=0;i<4;++i) p[i]=uint8_t(v>>(8*i)); }
uint32_t get32(const uint8_t* p) { uint32_t v=0; for(int i=0;i<4;++i) v|=uint32_t(p[i])<<(8*i); return v; }
uint32_t crc(const uint8_t* p,size_t n) {
  uint32_t c=~0u;
  for(size_t i=0;i<n;++i) { c^=p[i]; for(int b=0;b<8;++b) c=(c>>1)^((0u-(c&1u))&0xedb88320u); }
  return ~c;
}
std::vector<float> values(const ODNoiseProfile& p) {
  std::vector<float> v;
  for(int c=0;c<3;++c) {
    for(int b=0;b<OD_BANDS;++b) for(int l=0;l<OD_LEVELS;++l) v.push_back(p.sigma[c][b][l]);
    for(int l=0;l<OD_LEVELS;++l) v.push_back(p.sigmaFull[c][l]);
  }
  v.push_back(p.confidence); return v;
}
ODNoiseProfile profile(const std::vector<float>& v) {
  ODNoiseProfile p={}; size_t i=0;
  for(int c=0;c<3;++c) {
    for(int b=0;b<OD_BANDS;++b) for(int l=0;l<OD_LEVELS;++l) p.sigma[c][b][l]=v[i++];
    for(int l=0;l<OD_LEVELS;++l) p.sigmaFull[c][l]=v[i++];
  }
  p.confidence=v[i]; p.valid=1; return p;
}
bool valid(const ODNoiseProfile& p) {
  if(!p.valid || p.confidence<0 || p.confidence>1) return false;
  for(float v:values(p)) if(!std::isfinite(v)||v<0||v>1e6f) return false;
  return true;
}
}
size_t od_profile_blob_size(void) { return kBytes; }
int od_profile_encode(const ODNoiseProfile* p,void* bytes,size_t size) {
  if(!p||!bytes||size!=kBytes||!valid(*p)) return 1;
  auto v=values(*p); auto d=static_cast<uint8_t*>(bytes);
  memcpy(d,"ODNP",4); put32(d+4,1); put32(d+8,kBytes); put32(d+12,1); // basis=YCC atrous4 uniform8
  for(size_t i=0;i<v.size();++i) { uint32_t u; memcpy(&u,&v[i],4); put32(d+16+4*i,u); }
  put32(d+kBytes-4,crc(d,kBytes-4)); return 0;
}
int od_profile_decode(const void* bytes,size_t size,ODNoiseProfile* out) {
  if(!out) return 1;
  *out={}; if(!bytes||size!=kBytes) return 1;
  auto d=static_cast<const uint8_t*>(bytes);
  if(memcmp(d,"ODNP",4)||get32(d+4)!=1||get32(d+8)!=kBytes||get32(d+12)!=1||get32(d+kBytes-4)!=crc(d,kBytes-4)) return 1;
  std::vector<float> v(kValues);
  for(size_t i=0;i<v.size();++i) { uint32_t u=get32(d+16+4*i); memcpy(&v[i],&u,4); }
  auto p=profile(v); if(!valid(p)) return 1; *out=p; return 0;
}
int od_profile_aggregate(const ODNoiseProfile* in,int count,ODNoiseProfile* out) {
  if(!out) return 1;
  if(!in||count<1||count>32) { *out={}; return 1; }
  std::vector<std::vector<float>> rows;
  for(int i=0;i<count;++i) {
    if(!valid(in[i])) { *out={}; return 1; }
    rows.push_back(values(in[i]));
  }
  std::vector<float> v(kValues);
  for(size_t f=0;f<kValues;++f) {
    std::vector<double> a;
    for(auto& row:rows) a.push_back(f+1<kValues ? double(row[f])*row[f] : row[f]);
    std::sort(a.begin(),a.end());
    double median=(a[(count-1)/2]+a[count/2])*0.5;
    v[f]=float(f+1<kValues?std::sqrt(median):median);
  }
  *out=profile(v); return 0;
}
