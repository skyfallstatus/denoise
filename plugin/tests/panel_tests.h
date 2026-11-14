// SPDX-License-Identifier: MIT
// independent recipe, UI-state, migration, and real entry-point
// pins.
static double PanelValue(const PF_ParamDef& d) {
  return d.param_type == kPU ? d.u.pd.value : d.param_type == kSL ? d.u.sd.value :
         d.param_type == kCB ? d.u.bd.value : d.u.fs_d.value;
}
static void PanelSet(CPUCall& c,int id,double v) {
  auto& d=c.defs[PfOf(id)];
  if(d.param_type==kPU) d.u.pd.value=(A_long)v;
  else if(d.param_type==kSL) d.u.sd.value=(A_long)v;
  else if(d.param_type==kCB) d.u.bd.value=v!=0;
  else d.u.fs_d.value=v;
}
static std::set<int> PanelChange(EffectMainFn fn, CPUCall& c,int id) {
  gPanelSampleParams[c.in_data.effect_ref]=c.defs;
  for(auto& d:c.defs) d.uu.change_flags=0;
  PF_UserChangedParamExtra e{}; e.param_index=PfOf(id);
  CHECK(fn(PF_Cmd_USER_CHANGED_PARAM,&c.in_data,&c.out_data,c.ptrs.data(),nullptr,&e)==0,"panel change %d",id);
  std::set<int> written;
  for(int i=1;i<(int)c.defs.size();++i) if(c.defs[i].uu.change_flags & PF_ChangeFlag_CHANGED_VALUE) written.insert(kExpect[i-1].id);
  return written;
}
static void PanelRefresh(EffectMainFn fn, CPUCall& c) {
  gPanelSampleParams[c.in_data.effect_ref]=c.defs;
  for(auto& d:c.defs) d.uu.change_flags=0;
  auto before=c.defs;
  CHECK(fn(PF_Cmd_UPDATE_PARAMS_UI,&c.in_data,&c.out_data,c.ptrs.data(),nullptr,nullptr)==0,"panel refresh");
  CHECK(!memcmp(before.data(),c.defs.data(),sizeof(PF_ParamDef)*c.defs.size()),"UI refresh never writes values, flags or parameter definitions");
}
static void TestPanelUI(EffectMainFn fn) {
  World w=MakeWorld(8,8,16,0,false,false,1);
  CPUCall c; PrepareCPU(c,true,w.w,0,19001); c.in_data.quality=PF_Quality_HI;
  auto flags=[&](int id) {
    const auto it=gPanelUI.find({c.in_data.effect_ref,PfOf(id)});
    return it==gPanelUI.end()?c.defs[PfOf(id)].ui_flags:it->second.ui_flags;
  };
  auto label=[&](int id) {
    const auto it=gCaptureUI.find({c.in_data.effect_ref,PfOf(id)});
    return it==gCaptureUI.end()?std::string(c.defs[PfOf(id)].PF_DEF_NAME):it->second;
  };
  for(int id:{75,76}) {
    const auto& slider=c.defs[PfOf(id)].u.fs_d;
    CHECK(slider.useExponent && slider.exponent==1.5f,"mild reduction travel only, id %d",id);
    CHECK(slider.valid_min==0 && slider.valid_max==30 && slider.slider_min==0 && slider.slider_max==30,
          "literal dB storage/ranges unchanged, id %d",id);
  }
  PanelRefresh(fn,c);
  CHECK(label(73)=="Noise: measuring..." && (flags(73)&PF_PUI_NO_ECW_UI),"retired row fixed and hidden");
  CHECK(!gPanelUI.count({c.in_data.effect_ref,PfOf(73)}) && !gPanelUI.count({c.in_data.effect_ref,PfOf(74)}),
        "retired row and Help never receive UI mutations");
  for(int id:{71,103,104,105}) CHECK(flags(id)&PF_PUI_NO_ECW_UI,"hidden storage %d",id);
  CHECK(c.defs[PfOf(66)].flags & PF_ParamFlag_START_COLLAPSED,"Advanced starts folded");
  if (odplug::kUseMinimalPanel) {
    std::vector<int> main;
    for (const auto& e:kExpect) {
      if(e.id==66) break;
      if (!(flags(e.id)&(PF_PUI_NO_ECW_UI|PF_PUI_INVISIBLE))) main.push_back(e.id);
    }
    CHECK((main==std::vector<int>{72,75,76,77,9,74}),"exact six-control main panel in order");
    for(int id:{5,78,3,79,8,52,53,64,43,7,80,4,33,81,34,82,89,6,37,90,40,91,101,68,106,49,50,102})
      CHECK(PfOf(id)>PfOf(66) && PfOf(id)<PfOf(67),"tuning %d inside Advanced",id);
    for(int id:{10,21,22,31,32,35,36,41,42,47,48,51})
      CHECK((flags(id)&PF_PUI_NO_ECW_UI) && (flags(id)&PF_PUI_INVISIBLE),"retired topic %d hidden",id);
  } else {
    for(int id:{10,22,32,36,42,48,66})
      CHECK(!(flags(id)&(PF_PUI_NO_ECW_UI|PF_PUI_INVISIBLE)),"fallback topic %d visible",id);
  }
  CHECK(!strcmp(c.defs[PfOf(73)].u.button_d.u.PF_DEF_NAMESPTR,"Update") &&
        !strcmp(c.defs[PfOf(74)].u.button_d.u.PF_DEF_NAMESPTR,"Open guide"),"fixed button captions");
  CHECK(!strcmp(c.defs[PfOf(72)].u.pd.u.PF_DEF_NAMESPTR,"Natural|Cleaner|Keep texture|Phone|Colour only|Custom"),"dB Look items");
  CHECK(c.defs[PfOf(9)].u.pd.num_choices==5 &&
        !strcmp(c.defs[PfOf(9)].u.pd.u.PF_DEF_NAMESPTR,"Result|Removed noise|Motion protection|Measured areas|Before / after split"),
        "View has exactly five items in the original order");
  for(int view:{6,7,INT_MAX,0,-1,INT_MIN}) {
    PanelSet(c,9,view); PanelRefresh(fn,c);
    CHECK(label(74)=="Help","invalid saved View renders Result without export warning");
    CHECK(PanelValue(c.defs[PfOf(9)])==view,"invalid saved View is not rewritten by UI refresh");
  }
  PanelSet(c,9,1); PanelRefresh(fn,c);
  CHECK(label(74)=="Help","Help is fixed before rendering");
  if(odplug::kUseDynamicVisibility) {
    CHECK((flags(1)&PF_PUI_INVISIBLE) && !(flags(75)&PF_PUI_INVISIBLE),"new effect shows dB controls");
    CHECK(flags(102)&PF_PUI_INVISIBLE,"Switch hidden for dB set");
  }
  CHECK((flags(37)&PF_PUI_DISABLED) && (flags(38)&PF_PUI_DISABLED) && (flags(90)&PF_PUI_DISABLED),"sharpen off greying");
  PanelSet(c,6,20); PanelSet(c,8,2); PanelSet(c,79,1); PanelSet(c,75,0); PanelSet(c,76,0);
  PanelRefresh(fn,c);
  for(int id:{38,81,34,49,77,82,83,84,85,86,87,88,89,50}) CHECK(flags(id)&PF_PUI_DISABLED,"grey control %d",id);
  CHECK(label(74)=="Help","inactive controls never rename Help");
  CHECK(!(flags(37)&PF_PUI_DISABLED) && !(flags(90)&PF_PUI_DISABLED),"sharpen fine/threshold enabled in Fast");
  PanelSet(c,8,3); PanelSet(c,79,3); PanelSet(c,75,14); PanelSet(c,76,30); PanelSet(c,9,5);
  PanelRefresh(fn,c);
  for(int id:{37,38,90,81,34,49,77,82,86,50}) CHECK(!(flags(id)&PF_PUI_DISABLED),"re-enable control %d",id);
  CHECK(label(74)=="Help","View never renames Help");
  // Effective Previous after an old preset is pasted onto a new instance.
  PanelSet(c,1,67); PanelRefresh(fn,c);
  if(odplug::kUseDynamicVisibility) CHECK(!(flags(1)&PF_PUI_INVISIBLE) && (flags(75)&PF_PUI_INVISIBLE),"old preset effective set");
  PanelSet(c,75,10); PanelRefresh(fn,c);
  if(odplug::kUseDynamicVisibility) CHECK((flags(1)&PF_PUI_INVISIBLE) && !(flags(75)&PF_PUI_INVISIBLE),"non-default dB wins explicit dB set");
  PanelSet(c,71,1); PanelRefresh(fn,c);
  if(odplug::kUseDynamicVisibility) CHECK(!(flags(1)&PF_PUI_INVISIBLE) && (flags(75)&PF_PUI_INVISIBLE),"explicit Previous wins");
  CHECK((flags(105)&PF_PUI_NO_ECW_UI) && (flags(105)&PF_PUI_INVISIBLE),"retired row stays hidden for Previous controls");

  // dB Looks: recipe set is specified independently. Shared and unrelated
  // controls, including Processing, may never change with a Look.
  const std::set<int> lookSet={75,76,77,68,78,82,83,84,85,86,87,88,89,94,95,96,97,98,99,100,101};
  PrepareCPU(c,true,w.w,0,19001); c.in_data.quality=PF_Quality_HI;
  for(int look:{2,3,4,5,1}) {
    PanelSet(c,72,look);
    const auto before=c.defs;
    const auto changed=PanelChange(fn,c,72);
    for(int id:changed) CHECK(lookSet.count(id),"Look only writes recipe control %d",id);
    for(const auto& e:kExpect) {
      double want=PanelValue(before[PfOf(e.id)]);
      if(lookSet.count(e.id)) {
        want=e.dflt;
        if((look==2 || look==4) && e.id==77) want=0;
        if(look==3 && e.id==75) want=10.5;
        if(look==4 && (e.id==95 || e.id==96)) want=3.5;
        if(look==5 && e.id==75) want=0;
      }
      if(e.type==kFS || e.type==kPU || e.type==kSL || e.type==kCB)
        CHECK(std::fabs(PanelValue(c.defs[PfOf(e.id)])-want)<1e-7,"Look %d control %d",look,e.id);
    }
  }
  PanelSet(c,75,11); CHECK(PanelChange(fn,c,75)==std::set<int>{72} && PanelValue(c.defs[PfOf(72)])==6,"manual dB change -> Custom");
  PanelSet(c,75,14); CHECK(PanelChange(fn,c,75)==std::set<int>{72} && PanelValue(c.defs[PfOf(72)])==1,"manual return -> Natural");
  PanelSet(c,72,6); CHECK(PanelChange(fn,c,72).empty(),"Custom writes nothing");

  // Conversion: full transaction followed by a refusal matrix. No edits on
  // refusal, including when the host cannot prove absence of keyframes.
  PrepareCPU(c,true,w.w,0,19001,{{71,1},{106,1},{1,70},{2,80},{19,200},{5,60},{3,4},{15,50},{12,150},{68,2}});
  const auto old=c.defs;
  auto written=PanelChange(fn,c,102);
  CHECK(written.count(71) && !written.count(106) && PanelValue(c.defs[PfOf(106)])==1,"Switch one transaction preserves Processing");
  CHECK(PanelValue(c.defs[PfOf(71)])==2 && PanelValue(c.defs[PfOf(79)])==5,"Switch selects converted dB controls");
  CHECK(std::fabs(PanelValue(c.defs[PfOf(75)])-14*std::log(0.3)/std::log(0.2))<1e-12,"inverse amount");
  CHECK(std::fabs(PanelValue(c.defs[PfOf(82)])-14*std::log(0.65)/std::log(0.2))<1e-12,"inverse band product");
  for(int id:{1,2,19,5,3,15,12}) CHECK(PanelValue(c.defs[PfOf(id)])==kExpect[PfOf(id)-1].dflt,"previous control %d reset",id);
  CHECK(PanelValue(c.defs[PfOf(68)])==2,"shared grain mode preserved");
  // Emulate host undo of exactly the flagged values: the whole old state returns.
  for(int id:written) c.defs[PfOf(id)] = old[PfOf(id)];
  for(int i=1;i<(int)c.defs.size();++i) if(kExpect[i-1].type==kFS || kExpect[i-1].type==kPU || kExpect[i-1].type==kSL)
    CHECK(PanelValue(c.defs[i])==PanelValue(old[i]),"Switch undo restores control %d",kExpect[i-1].id);
  for(int test=0;test<8;++test) {
    PrepareCPU(c,true,w.w,0,19001,{{71,1}});
    if(test==0) PanelSet(c,4,90);
    if(test==1) PanelSet(c,5,5);
    if(test==2) gPanelKeys[PfOf(1)]=1;
    if(test==3) gPanelKeys[PfOf(3)]=3;
    if(test==4) { gPanelKeysFail=true; gPanelSampleFail=true; }
    if(test==5) PanelSet(c,1,99);
    if(test==6) PanelSet(c,44,400);
    if(test==7) PanelSet(c,5,100);
    const auto before=c.defs;
    CHECK(PanelChange(fn,c,102).empty(),"refusal %d writes no values",test);
    for(int id:{1,4,5,44,71,75,106}) CHECK(PanelValue(c.defs[PfOf(id)])==PanelValue(before[PfOf(id)]),"refusal preserves %d",id);
    const std::string reason=c.out_data.return_msg;
    CHECK(reason.find("kept old")!=std::string::npos || reason=="Animated: keyframes kept" || reason.find("must be")!=std::string::npos ||
          reason.find("below")!=std::string::npos || reason.find("range")!=std::string::npos,"refusal returned to host: %s",reason.c_str());
    CHECK(label(74)=="Help","Switch refusal never renames Help");
    gPanelSamples=0;
    PanelRefresh(fn,c);
    CHECK(label(74)=="Help" && gPanelSamples==0,"UI refresh never rechecks conversion or renames Help");
    gPanelKeys.clear(); gPanelKeysFail=false; gPanelSampleFail=false;
  }
  // Premiere 27's keyframe API fails: static values must still convert.
  // Equal endpoints with an interior excursion must refuse without writes.
  for(bool animated:{false,true}) {
    PrepareCPU(c,true,w.w,0,19001,{{71,1},{106,1}});
    c.in_data.total_time=300*1001;
    gPanelKeysFail=true;gPanelSamples=gPanelSampleCheckins=0;gPanelSampleTimes.clear();
    gPanelSampleValue=[animated](int pf,A_long t,double v) {
      return animated && pf==PfOf(1) && t>100*1001 && t<200*1001 ? v-10 : v;
    };
    const long logStart=PluginLogSize();const auto before=c.defs;
    const auto changed=PanelChange(fn,c,102);
    CHECK(gPanelSamples>0 && gPanelSamples==gPanelSampleCheckins,"sampling checks every value back in");
    CHECK(PluginLogFrom(logStart).find("animation check=parameter sampling")!=std::string::npos,"Switch logs sampling fallback");
    if(animated) {
      CHECK(changed.empty() && std::string(c.out_data.return_msg)=="Animated: keyframes kept","interior animation refuses conversion");
      for(int id:{1,3,4,71,75,79,106}) CHECK(PanelValue(c.defs[PfOf(id)])==PanelValue(before[PfOf(id)]),"sampling refusal preserves %d",id);
      PanelRefresh(fn,c);CHECK(label(74)=="Help","sampled animation refusal leaves Help fixed");
    } else {
      CHECK(changed.count(71) && PanelValue(c.defs[PfOf(71)])==2 && PanelValue(c.defs[PfOf(106)])==1,
            "API failure + static samples switches units and preserves Processing");
      const std::set<A_long> times(gPanelSampleTimes.begin(),gPanelSampleTimes.end());
      CHECK(times.size()==18 && *times.begin()==0 && *times.rbegin()==299*1001,
            "fallback samples clip start/end and sixteen interior times");
    }
    gPanelKeysFail=false;gPanelSampleValue={};
  }
  // If PF total_time is unavailable, the Premiere duration is in frames.
  PrepareCPU(c,true,w.w,0,19001,{{71,1}});gPanelKeysFail=true;gPanelSampleTimes.clear();
  CHECK(PanelChange(fn,c,102).count(71),"utility duration fallback permits static Switch");
  CHECK(!gPanelSampleTimes.empty() && *std::max_element(gPanelSampleTimes.begin(),gPanelSampleTimes.end())==299*1001,
        "utility duration converted from frames to PF time units");
  gPanelKeysFail=false;gPanelSampleParams.clear();
}

struct DBCase { int id; double value; const char* field; double expected; };
static std::vector<DBCase> DBMappingCases() {
  auto amount=[](double x){return x==30?1:1-std::pow(.2,x/14);};
  auto gain=[](double x){return std::pow(10.,x/20);};
  std::vector<DBCase> c={{75,0,"luma",0},{75,7,"luma",amount(7)},{76,0,"chroma",0},{76,9,"chroma",amount(9)},
    {77,1.314,"floor",2},{78,-6,"spatial",.5*gain(-6)},{78,-18,"spatial",0},
    {80,6,"noise",gain(6)},{81,-3,"mthr",gain(-3)},{90,3,"shcore",gain(3)},
    {9,1,"view",0},{9,2,"view",1},{9,3,"view",2},{9,4,"view",3},{9,5,"view",4},
    {9,6,"view",0},{9,7,"view",0},{9,INT_MAX,"view",0},{9,0,"view",0},{9,-1,"view",0}};
  // Stable storage for generated field names.
  static const char* fields[]={"amY0","amY1","amY2","amY3","amC0","amC1","amC2","amC3",
    "cnY","cnCb","cnCr","nlY0","nlY1","nlY2","nlY3","nlC0","nlC1","nlC2","nlC3"};
  for(int b=0;b<8;++b) c.push_back({82+b,2.+b,fields[b],amount(2.+b)/(b<4?.8:1.)});
  for(int i=0;i<11;++i) c.push_back({91+i,-5.+i,fields[8+i],gain(-5.+i)});
  for(int i=1;i<=6;++i) c.push_back({79,double(i),"radius",double(i-1)});
  return c;
}
static void CheckDefaultPin(const std::vector<Rec>& records, int wait, int version) {
  const auto* p=LastOf(records,"defaultpin");
  CHECK(p && (*p)["mismatches"]=="none","new effect every picture field == od_params_default(): %s",p?(*p)["mismatches"].c_str():"missing");
  CHECK(p && p->num("recovery")== (version==2),"Processing gates chroma recovery");
  CHECK(p && p->num("temporalModel")== (version==2),"Processing gates the temporal noise model");
  CHECK(p && p->num("wait")==wait && p->num("verify")==1,"host execution-policy fields pinned separately");
}

// Exercise the actual host callbacks, including the dependency list passed
// to Render. Version 1's old requests are pinned alongside the new policy.
static void TestProcessingDependencies(EffectMainFn fn, GPUEnv* gpu, const World& in, World& out) {
  for(int version:{1,2}) for(int controls:{1,2}) for(int radius=0;radius<=5;++radius)
    for(int strength:{0,50,100}) {
      if(controls==2 && strength!=100) continue; // dB controls use full temporal strength
      const int request=version==2 ? std::max(radius,2) : strength==0 ? 0 : radius;
      const std::map<int,double> values={{106,double(version)},{71,double(controls)},
          {3,double(radius)},{79,double(radius+1)},{4,double(strength)}};
      auto expected=[&](int frame) {
        std::vector<int> offsets;
        for(int d=1;d<=request;++d) {
          if(frame-d>=1000) offsets.push_back(-d);
          if(frame+d<1100) offsets.push_back(d);
        }
        return offsets;
      };
      auto coreOffsets=[](const std::vector<int>& offsets) {
        std::string s="0";for(int d:offsets)s+=","+std::to_string(d);return s;
      };
      if(!gpu) {
        CPUCall c;PrepareCPU(c,true,in.w,20*1001,19100,values);
        ResetCheckoutStats();Take();
        CHECK(CallRender(fn,c,out.w)==0,"CPU dependency version %d set %d radius %d strength %d",version,controls,radius,strength);
        std::vector<A_long> times;for(int d:expected(1050)) times.push_back((20+d)*1001);
        CHECK(gCPU.checkoutTimes==times && gCPU.checkins==int(times.size()),
              "CPU exact checkout/checkin policy version %d set %d radius %d strength %d",version,controls,radius,strength);
        const auto records=Take();const auto* r=LastOf(records,"render");
        CHECK(r && (*r)["off"]==coreOffsets(expected(1050)),"CPU supplied offsets equal dependency policy");
        CHECK(r && r->num("radius")==radius && std::fabs(r->num("tstr")-strength/100.)<1e-6,
              "profiling window never enlarges the requested merge");
        continue;
      }
      DefaultGPUParams();
      for(const auto& v:values) {
        const int type=kExpect[PfOf(v.first)-1].type;
        if(type==kPU || type==kSL) SetI(v.first,(int)v.second); else SetF(v.first,v.second);
      }
      PrGPUFilterInstance inst{};CHECK(Create(*gpu,inst),"GPU dependency matrix instance");
      if(!inst.ioPrivatePluginData) continue;
      for(int frame:{1000,1050,1099}) {
        const auto offsets=expected(frame);
        CHECK(DepsAt(*gpu,inst,frame,frame)==offsets,
              "GPU exact dependencies version %d set %d radius %d strength %d frame %d",version,controls,radius,strength,frame);
        Rec r=RecordedRender(*gpu,inst,frame,kPrRenderQuality_High,1+(int)offsets.size());
        CHECK(r["off"]==coreOffsets(offsets),"GPU render uses exactly its dependency plan");
        CHECK(r.num("radius")==radius && std::fabs(r.num("tstr")-strength/100.)<1e-6,
              "GPU profiling window never enlarges the requested merge");
      }
      gpu->filter.DisposeInstance(&inst);
    }
  printf("  Processing dependency matrix: %s, both versions/control sets, radii 0..5, strengths 0/50/100\n",gpu?"GPU + clip bounds":"CPU");
}
static void TestPanel(EffectMainFn fn, PrGPUFilterEntry entry) {
  TestPanelUI(fn);
  if(gReal) return; // exact render arguments are captured by the independent stub
  World in=MakeWorld(64,64,16,0,false,false,1),out=MakeWorld(64,64,16,0,false,false,2);
  SetWorldFormat(in.w.data,PrPixelFormat_BGRA_4444_32f); SetWorldFormat(out.w.data,PrPixelFormat_BGRA_4444_32f);
  gCPU.source={in.w,PrPixelFormat_BGRA_4444_32f,nullptr}; gCPU.failCheckouts=false;
  TestProcessingDependencies(fn,nullptr,in,out);
  std::map<std::pair<int,double>,Rec> cpu;
  for(int version:{1,2}) {
    CPUCall c; PrepareCPU(c,true,in.w,20*1001,19000,{{106,double(version)}});
    Take(); CHECK(CallRender(fn,c,out.w)==0,"default CPU version %d",version); CheckDefaultPin(Take(),1,version);
  }
  for(const auto& test:DBMappingCases()) {
    CPUCall c; PrepareCPU(c,true,in.w,20*1001,19000,{{test.id,test.value}});
    Take(); CHECK(CallRender(fn,c,out.w)==0,"dB CPU %d",test.id);
    const auto records=Take(); const auto* r=LastOf(records,"render");
    CHECK(r && std::fabs(r->num(test.field)-test.expected)<1e-5,"dB CPU %d -> %s",test.id,test.field);
    if(r) cpu[{test.id,test.value}]=*r;
  }
  GPUEnv g; g.util.getSPBasicSuite=GetBasic;g.suites.utilFuncs=&g.util;
  gGPU.device=MTLCreateSystemDefaultDevice();gGPU.queue=[gGPU.device newCommandQueue];
  PrGPUFilterInfo info{};csSDK_int32 index=0;
  CHECK(entry(PrSDKGPUFilterInterfaceVersion,&index,kPrTrue,&g.suites,&g.filter,&info)==0,"dB GPU startup");
  for(bool topics:{true,false}) for(bool input:{false,true}) {
    UseLayout(topics,(topics?kCountTopics:kCountNoTopics)+(input?1:0));
    SetNode(19000+(topics?4:0)+(input?2:0),1000,1100,1000,1100);
    for(int version:{1,2}) {
      SetI(106,version);PrGPUFilterInstance inst{};CHECK(Create(g,inst),"default GPU instance");
      if(!inst.ioPrivatePluginData) continue;
      Take(); RecordedRender(g,inst,1050,kPrRenderQuality_High);CheckDefaultPin(Take(),0,version);
      g.filter.DisposeInstance(&inst);
    }
    for(const auto& test:DBMappingCases()) {
      DefaultGPUParams();
      if(kExpect[PfOf(test.id)-1].type==kPU) SetI(test.id,(int)test.value);else SetF(test.id,test.value);
      PrGPUFilterInstance inst{};CHECK(Create(g,inst),"dB GPU instance %d",test.id);if(!inst.ioPrivatePluginData) continue;
      const auto deps=DepsAt(g,inst,1050,1050);
      const int radius=test.id==79?std::max(2,(int)test.expected):2;
      CHECK((int)deps.size()==radius*2,"dB dependencies include the fixed profiling window, id %d",test.id);
      Rec r=RecordedRender(g,inst,1050,kPrRenderQuality_High,1+radius*2);
      CHECK(std::fabs(r.num(test.field)-test.expected)<1e-5,"dB GPU %d -> %s",test.id,test.field);
      for(auto& kv:AdvancedDefaults()) CHECK(r[kv.first.c_str()]==cpu[std::make_pair(test.id,test.value)][kv.first.c_str()],"CPU/GPU dB parity id %d field %s",test.id,kv.first.c_str());
      g.filter.DisposeInstance(&inst);
    }
    TestProcessingDependencies(fn,&g,in,out);
  }
  index=0;entry(PrSDKGPUFilterInterfaceVersion,&index,kPrFalse,&g.suites,&g.filter,&info);
  gCPU.source={};UseLayout(true,kCountTopics);
  PF_InData teardown{}; PF_OutData teardownOut{};
  fn(PF_Cmd_GLOBAL_SETDOWN,&teardown,&teardownOut,nullptr,nullptr,nullptr);
  printf("  processing default identity: Version 1 unchanged, Version 2 chroma recovery + temporal noise model only; CPU + all four GPU counts; every picture field exact\n");
}
