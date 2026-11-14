// SPDX-License-Identifier: MIT
// Host-independent wording and drawing. Observations never change settings.
#include "analysis_layout.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace {
constexpr float kEmptyWidth = 276.f, kEmptyHeight = 88.f;
bool hasProfile(const ODAnalysis* a) { return a && a->valid && a->profileValid; }
float finite(float x) { return std::isfinite(x) ? std::max(0.f,x) : 0.f; }
float at(const float* knots, float y) {
  float t = std::clamp(y,0.f,1.f)*7;
  int l = std::min(6,(int)t);
  return finite(knots[l])*(1-(t-l)) + finite(knots[l+1])*(t-l);
}
float weighted(const ODAnalysis& a, const float* v) {
  float sum=0, area=0;
  for(int l=0;l<8;++l) { float h=finite(a.histogram[l]); sum+=h*at(v,(l+.5f)/8); area+=h; }
  return area>0 ? sum/area : 0;
}
// Profile white-noise a-trous ratio: 0.20066 / 0.89080 = 0.22526.
struct Words { const char *level, *kind, *where; };
Words words(const ODAnalysis& a) {
  float y=weighted(a,a.full[0]);
  float col=std::hypot(weighted(a,a.full[1]),weighted(a,a.full[2]))/std::sqrt(2.f);
  float ratio=weighted(a,a.profile[0][1])/std::max(weighted(a,a.profile[0][0]),1e-6f)/.22526f;
  Words w={y<.4f?"faint":y<.8f?"light":y<1.6f?"medium":y<3.2f?"heavy":"severe",
    col>std::max(.05f,1.5f*y)?"colour":ratio<1.25f?"fine":ratio<=2.7f?"coarse":"blotchy","overall"};
  float ranges[3]={},total=0;
  for(int l=0;l<8;++l) {
    float b=(l+.5f)/8, s=at(a.full[0],b), e=s*s*finite(a.histogram[l]);
    // Colour-only scenes still have a meaningful location.
    if(col>1.5f*y) {float cb=at(a.full[1],b),cr=at(a.full[2],b);e=.5f*(cb*cb+cr*cr)*finite(a.histogram[l]);}
    float lo=l/8.f,hi=(l+1)/8.f;
    const float cuts[]={0,.25f,.65f,1};
    for(int r=0;r<3;++r)ranges[r]+=e*std::max(0.f,std::min(hi,cuts[r+1])-std::max(lo,cuts[r]))*8;
    total+=e;
  }
  for(int r=0;r<3;++r) if(ranges[r]>.5f*total) w.where=r==0?"shadows":r==1?"mids":"brights";
  return w;
}
float asked(float a) { a=std::clamp(finite(a),0.f,1.f);return a>=1?30.f:std::min(30.f,14.f*std::log1p(-a)/std::log(.2f)); }
float colourRemoved(const ODAnalysis& a) {
  double i=0,o=0;
  for(int c=1;c<3;++c)for(int b=0;b<4;++b){i+=finite(a.inputEnergy[c][b]);o+=finite(a.outputEnergy[c][b]);}
  return i>0?(float)std::clamp(10*std::log10(i/std::max(o,1e-30)),-30.,60.):0;
}
void copy(char* dst,const char* s){if(dst)std::snprintf(dst,32,"%s",s);}
}

void od_analysis_status(const ODAnalysis* a,const ODParams* p0,char row1[32],char row2[32]) {
  const ODParams def=od_params_default(); const ODParams& p=p0?*p0:def;
  if(!a||!a->valid||!a->profileValid){copy(row1,"Noise: measuring...");copy(row2,p.view!=OD_VIEW_OUTPUT?"View: exports this view":"Help and status");return;}
  auto w=words(*a);
  if(row1)std::snprintf(row1,32,"Noise: %s, %s, %s",w.level,w.kind,w.where);
  float heldArea=0,area=0;
  for(int l=0;l<8;++l){float h=finite(a->histogram[l]);area+=h;if(at(a->effectiveLuma,(l+.5f)/8)<=.5f*p.lumaStrength)heldArea+=h;}
  float colFine=0;
  for(int c=1;c<3;++c)for(int b=0;b<2;++b)colFine+=weighted(*a,a->profile[c][b]);
  // Priority is the product contract, including the view warning below limits.
  if(a->state==OD_ANALYSIS_LEARNING){if(row2)std::snprintf(row2,32,"Learning noise (%u of 8)",std::min(8u,a->learningFrames));}
  else if(a->confidence<.3f||a->flatTiles<8)copy(row2,"Unsure: few flat areas");
  else if(a->state!=OD_ANALYSIS_CAPTURED&&!a->measured[0]&&!a->measured[1]&&
          (a->held[0]||a->held[1]))copy(row2,"Shadows unseen: kept last");
  else if(p.chromaStrength>0&&colFine<.10f)copy(row2,"Colour noise: none found");
  else if(a->temporalValid&&a->temporalSpatialRatio<.7f)copy(row2,"Phone-like noise: try Phone");
  else if(p.lumaStrength>0&&(!p.advanced||p.grainFloor>0)&&heldArea>.5f*area)copy(row2,"Luma held by Keep grain");
  else if(p.lumaStrength>0&&weighted(*a,a->full[0])>=.1f&&(a->removalValid&15)==15&&asked(p.lumaStrength)-a->removedOverallDb[0]>4)copy(row2,"Luma at filter limit");
  else if(p.view!=OD_VIEW_OUTPUT)copy(row2,"View: exports this view");
  else if(row2)std::snprintf(row2,32,"Removed: luma %.0f dB, col %.0f dB",
       std::round(std::clamp(a->removedOverallDb[0],-30.f,60.f)) + 0.f, std::round(std::clamp(colourRemoved(*a),-30.f,60.f)) + 0.f);
}

void od_analysis_box(uint32_t fw,uint32_t fh,uint32_t* x,uint32_t* y,uint32_t* w,uint32_t* h){
  if(!x||!y||!w||!h)return;
  *x=*y=*w=*h=0;if(!fw||!fh)return;
  const uint32_t mx=std::min(fw/2,uint32_t((uint64_t(fw)+19)/20));
  const uint32_t my=std::min(fh/2,uint32_t((uint64_t(fh)+19)/20));
  const uint32_t availableW=fw-2*mx,availableH=fh-2*my;
  // The shorter side controls scale: 70% of portrait width, 40% of 16:9
  // landscape width. Fit the 360x400 reflow inside 5% margins on both axes.
  const double desired=std::min(fw,fh)*(fw<fh?.70:(.40*16/9));
  const double width=std::min({desired,double(availableW),availableH*.9});
  *w=uint32_t(std::lround(width));*h=std::min(availableH,uint32_t(std::lround(*w/ .9)));
  *x=mx;*y=fh-my-*h;
}

void od::analysisViewBox(const ODAnalysis* a,uint32_t fw,uint32_t fh,
                         uint32_t* x,uint32_t* y,uint32_t* w,uint32_t* h) {
  od_analysis_box(fw,fh,x,y,w,h);
  if(!x||!y||!w||!h||hasProfile(a))return;
  const float scale=std::min(*w/360.f,*h/400.f);
  const uint32_t height=std::min(*h,uint32_t(std::lround(kEmptyHeight*scale)));
  *y+=*h-height;*h=height;
  *w=std::min(*w,uint32_t(std::lround(kEmptyWidth*scale)));
}

static int Layout(const ODAnalysis* a,const ODParams* p0,ODDrawList* list,float w,float h,bool overlay){
  if(!list)return 0;*list={};if(!(w>0&&h>0))return 0;list->width=w;list->height=h;
  const ODParams def=od_params_default();const ODParams& p=p0?*p0:def;
  const bool shortPanel = !overlay && h <= 210 && w <= 600;
  const bool have=hasProfile(a);
  const float designWidth = overlay ? (have ? 360.f : kEmptyWidth) : 640.f;
  const float designHeight = overlay ? (have ? 400.f : kEmptyHeight) : shortPanel ? 240.f : 420.f;
  const float sx=w/designWidth,sy=h/designHeight,fs=std::min(sx,sy);
  const bool compact = overlay || w < 520 || shortPanel;
  const float white[]={.94f,.96f,.98f,1},muted[]={.62f,.69f,.76f,1},orange[]={1,.66f,.29f,1},red[]={1,.44f,.27f,1},mint[]={.40f,.86f,.73f,1},grid[]={.22f,.22f,.22f,1},bg[]={32.f/255,32.f/255,32.f/255,1};
  auto add=[&](ODDrawKind k,float x,float y,float cw,float ch,const float* col,float stroke=0,float dash=0,float font=0,const char* t=""){
    if(list->count>=OD_DRAW_CAPACITY)return;
    auto& d=list->commands[list->count++];d.kind=k;d.x=x*sx;d.y=y*sy;
    d.w=cw*sx;d.h=ch*sy;d.stroke=stroke*fs;d.dash=dash*fs;d.fontSize=font ? (overlay ? std::max(17.f,font)*fs : std::max(compact ? 9.f : 0.f,font*fs)) : 0;
    std::copy(col,col+4,d.rgba);std::snprintf(d.text,sizeof(d.text),"%s",t);
  };
  auto text=[&](float x,float y,const char* t,const float* col= nullptr,float size=14){add(OD_DRAW_TEXT,x,y,0,size*1.25f,col?col:white,0,0,size,t);};
  auto line=[&](float x,float y,float ex,float ey,const float* col,float stroke=1,float dash=0){add(OD_DRAW_LINE,x,y,ex,ey,col,stroke,dash);};
  add(OD_DRAW_RECT,0,0,designWidth,designHeight,bg);add(OD_DRAW_RECT,.5f,.5f,designWidth-1,designHeight-1,grid,1);
  char s[96],row1[32],row2[32];od_analysis_status(a,&p,row1,row2);
  if(have){text(overlay?12:20,overlay?10:shortPanel?10:16,"NOISE FOUND",white,overlay?22:shortPanel?13:17);auto v=words(*a);std::snprintf(s,sizeof(s),"%s / %s%s / %s",v.level,v.kind,std::strcmp(v.kind,"fine")==0?" grain":"",v.where);if (a->renderId && (overlay || !compact)) text(overlay?12:158,overlay?39:18,s,white,overlay?18:16);
    if(a->state==OD_ANALYSIS_CAPTURED){if(a->capturedFrames)std::snprintf(s,sizeof(s),"saved profile / %u frames",a->capturedFrames);else std::snprintf(s,sizeof(s),"saved profile");}
    else if(a->state==OD_ANALYSIS_FROZEN)std::snprintf(s,sizeof(s),"frozen");
    else if(a->state==OD_ANALYSIS_LEARNING)std::snprintf(s,sizeof(s),"automatic / learning %u/8",a->learningFrames);
    else std::snprintf(s,sizeof(s),"automatic / %s",a->state==OD_ANALYSIS_SETTLED?"settled":"measuring");
    text(overlay?12:shortPanel?180:20,overlay?64:shortPanel?10:43,s,muted,shortPanel?12:13);
  }else {
    text(overlay?12:20,18,overlay?"Noise: unavailable":"Noise: measuring...",muted,overlay?22:17);
    text(overlay?12:20,overlay?54:48,overlay?"No usable noise estimate":"Play or click Update",muted,14);
    return (int)list->count;
  }
  const float legend=overlay?92:shortPanel?34:73;
  text(overlay?12:20,legend,"Noise / 10-bit codes",muted,12);
  text(overlay?236:357,legend,"Y",white,12);line(overlay?221:340,legend+8,overlay?231:351,legend+8,white,1.6f);
  text(overlay?280:409,legend,"Cb",orange,12);line(overlay?264:387,legend+8,overlay?275:403,legend+8,orange,1.6f,4);
  text(overlay?325:469,legend,"Cr",red,12);line(overlay?309:447,legend+8,overlay?320:463,legend+8,red,1.6f,1.5f);
  if(shortPanel){line(518,legend+8,530,legend+8,mint,1,4);text(536,legend,"Grain",mint,11);}
  if (!compact) {
    add(OD_DRAW_DOT,516,80,2.5f,0,white);text(525,73,"measured",muted,11);
    add(OD_DRAW_DOT,586,80,2.5f,0,white,1);text(594,73,"held",muted,11);
  }
  float high=8;
  if(have)for(int c=0;c<3;++c)for(int l=0;l<8;++l)while(high<64&&a->full[c][l]>high)high*=2;
  const float low=.125f,top=overlay?122:shortPanel?58:102,bottom=overlay?214:shortPanel?128:242,left=overlay?40:51,right=overlay?346:613;
  auto yy=[&](float v){return bottom-(std::log2(std::clamp(finite(v),low,high)/low)/std::log2(high/low))*(bottom-top);};
  for(float v=.25f;v<=high;v*=(overlay||shortPanel)?4:2){float y=yy(v);line(left,y,right,y,grid);std::snprintf(s,sizeof(s),"%g",v);text(overlay?8:v<1?16:25,y-(overlay?9:6),s,muted,11);}
  if(have){float maxH=0;for(float v:a->histogram)maxH=std::max(maxH,finite(v));
    const float hist[]={.30f,.37f,.44f,.48f};
    for(int l=0;l<8;++l){float bar=maxH>0?finite(a->histogram[l])/maxH*29:0;add(OD_DRAW_RECT,left+l*(right-left)/8+1,bottom-bar,(right-left)/8-2,bar,hist);}
    for(int c=0;c<3;++c){const float* col=c==0?white:c==1?orange:red;float dash=c==0?0:c==1?5:1.5f;
      for(int l=0;l<7;++l)line(left+l*(right-left)/7,yy(a->full[c][l]),left+(l+1)*(right-left)/7,yy(a->full[c][l+1]),col,1.8f,dash);
      for(int l=0;l<8;++l)add(OD_DRAW_DOT,left+l*(right-left)/7,yy(a->full[c][l]),3.f,0,col,a->measured[l]?0:1.3f);
    }
  }
  float keep=.657f*(p.advanced?std::clamp(finite(p.grainFloor),0.f,10.f):1.f);
  if(keep>0){line(left,yy(keep),right,yy(keep),mint,1,5);std::snprintf(s,sizeof(s),"Keep grain %.2f",keep);if(!shortPanel)text(overlay?196:454,overlay?250:270,s,mint,12);}
  const float axis=overlay?224:shortPanel?137:251, table=overlay?279:shortPanel?163:301;
  const float row=overlay?305:shortPanel?180:339, rowStep=overlay?27:shortPanel?20:24;
  text(overlay?40:51,axis,"Shadows",muted,12);text(overlay?166:250,axis,"Mid-tones",muted,12);text(overlay?274:536,axis,"Highlights",muted,12);
  if(!shortPanel&&!overlay)text(51,270,"Brightness of this frame",muted,11);
  if(overlay)text(12,250,"REMOVED / dB",white,17);
  line(overlay?12:20,table-8,overlay?348:620,table-8,grid);text(overlay?12:20,table+3,overlay?"":compact?"dB removed":"REMOVED / dB",white,13);
  const float xs[]={overlay?88.f:181.f,overlay?140.f:281.f,overlay?186.f:381.f,overlay?247.f:481.f};const char* sizes[]={"Fine",overlay?"Med.":"Medium","Coarse","Large"};const char* px[]={"1-2 px","2-4 px","4-8 px","8+ px"};
  for(int b=0;b<4;++b){text(xs[b],table,sizes[b],muted,12);if(!shortPanel&&!overlay)text(xs[b],317,px[b],muted,10);}
  text(overlay?306:565,table,"Asked",muted,12);text(overlay?12:20,row,"Luma",white,overlay?18:14);text(overlay?12:20,row+rowStep,"Colour",orange,overlay?18:14);
  for(int c=0;c<2;++c)for(int b=0;b<4;++b){bool valid=have&&(a->removalValid&(1u<<(c*4+b)));
    float v=0;if(valid){v=a->removedDb[0][b];if(c){double ei=a->inputEnergy[1][b]+a->inputEnergy[2][b],eo=a->outputEnergy[1][b]+a->outputEnergy[2][b];v=ei>0?std::clamp(10*std::log10(ei/std::max(eo,1e-30)),-30.,60.):0;}}
    if(overlay){
      // Colour combines both channels, so both must be measurable. Reject
      // missing/invalid energies rather than displaying an invented zero.
      if(c){valid=valid&&(a->removalValid&(1u<<(8+b)));
        for(int ch=1;ch<3;++ch)valid=valid&&std::isfinite(a->inputEnergy[ch][b])&&a->inputEnergy[ch][b]>0&&
          std::isfinite(a->outputEnergy[ch][b])&&a->outputEnergy[ch][b]>=0;}
      valid=valid&&std::isfinite(v);
      // Normalize only values whose one-decimal representation is negative
      // zero. Preserve real negative measurements and all other rounding.
      if(std::abs(v)<.05f)v=0.f;
    }
    if(valid)std::snprintf(s,sizeof(s),"%.1f",v);else std::snprintf(s,sizeof(s),"--");text(xs[b],row+rowStep*c,s,c?orange:white,overlay?18:15);
  }
  for(int c=0;c<2;++c){float d=asked(c?p.chromaStrength:p.lumaStrength);std::snprintf(s,sizeof(s),d>=30?"30 / all":"%.0f",d);text(overlay?306:565,row+rowStep*c,s,c?orange:white,13);}
  const float footer=overlay?373:shortPanel?221:399;
  line(overlay?12:20,overlay?363:shortPanel?217:390,overlay?348:620,overlay?363:shortPanel?217:390,grid);text(overlay?12:20,footer,a->state==OD_ANALYSIS_CAPTURED && !a->renderId ? "Saved profile; removal not measured" : row2,mint,12);
  return (int)list->count;
}

int od_analysis_layout(const ODAnalysis* a,const ODParams* p,ODDrawList* list,float w,float h) {
  return Layout(a,p,list,w,h,false);
}
int od::analysisViewLayout(const ODAnalysis* a,const ODParams* p,ODDrawList* list,float w,float h) {
  return Layout(a,p,list,w,h,true);
}
