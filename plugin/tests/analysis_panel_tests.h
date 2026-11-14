// SPDX-License-Identifier: MIT
// Integration at the loaded EffectMain boundary: a real completed render,
// fixed Help and a hidden retired row, before and after CPU/GPU rendering.
static void CheckFixedPanel(const CPUCall& c) {
  CHECK((c.defs[PfOf(73)].ui_flags&PF_PUI_NO_ECW_UI) &&
        (c.defs[PfOf(73)].ui_flags&PF_PUI_INVISIBLE),"retired row permanently hidden");
  CHECK(!strcmp(c.defs[PfOf(74)].PF_DEF_NAME,"Help") &&
        !strcmp(c.defs[PfOf(74)].u.button_d.u.PF_DEF_NAMESPTR,"Open guide") &&
        !(c.defs[PfOf(74)].ui_flags&(PF_PUI_NO_ECW_UI|PF_PUI_INVISIBLE|PF_PUI_DISABLED)),"Help fixed and available");
  for(int id:{73,74}) CHECK(!gPanelUI.count({c.in_data.effect_ref,PfOf(id)}),"no UI mutation for fixed row %d",id);
}
static void TestAnalysisPanel(EffectMainFn fn) {
  if (!gReal) return;
  World input=MakeWorld(320,192,16,0,false,false,1), output=MakeWorld(320,192,16,0,false,false,2);
  FillNoisy((float*)input.w.data,320*192,2189);
  SetWorldFormat(input.w.data,PrPixelFormat_BGRA_4444_32f);
  SetWorldFormat(output.w.data,PrPixelFormat_BGRA_4444_32f);
  CPUCall a;PrepareCPU(a,true,input.w,10*1001,64001,{{79,1},{106,1}});
  const auto original=a.defs;
  const long logStart=PluginLogSize();
  PanelRefresh(fn,a);
  CheckFixedPanel(a);
  for(int i=0;i<3;++i) {
    a.in_data.current_time=(10+i)*1001;
    FillNoisy((float*)input.w.data,320*192,2189+i);
    CHECK(CallRender(fn,a,output.w)==0,"analysis CPU real render");
  }
  CHECK(PanelChange(fn,a,73).empty(),"retired button cannot change values");
  CheckFixedPanel(a);
  for(int pf=1;pf<odplug::kNumParams;++pf) if(kExpect[pf-1].type!=kAR)
    CHECK(PanelValue(a.defs[pf])==PanelValue(original[pf]),"UI preserves control %d",pf);
  // A copied effect has a fresh ID, including at an identical start.
  CPUCall copy;PrepareCPU(copy,true,input.w,12*1001,64002,{{79,1},{106,1}});
  CPUHost::Owner owner;OwnerOf(a.in_data.effect_ref,&owner);gCPU.ownerOf[copy.in_data.effect_ref]=owner;
  PanelRefresh(fn,copy);PanelRefresh(fn,a);
  CheckFixedPanel(copy); CheckFixedPanel(a);
  CHECK(CallRender(fn,copy,output.w)==0,"render independent software copy");
  PanelRefresh(fn,copy);
  CheckFixedPanel(copy);
  // Only a real publication disagreement blocks both readers of the same ID.
  gCPU.instanceIdOf[copy.in_data.effect_ref]=64001;
  gCPU.ownerOf[copy.in_data.effect_ref].itemStart+=copy.in_data.time_scale;
  CHECK(CallRender(fn,copy,output.w)==0,"render same-ID conflict at another start");
  PanelRefresh(fn,copy);PanelRefresh(fn,a);
  CheckFixedPanel(copy); CheckFixedPanel(a);
  CHECK(PluginLogFrom(logStart).find("UI analysis:")==std::string::npos &&
        PluginLogFrom(logStart).find("Analysis conflict:")==std::string::npos,"CPU never publishes UI analysis");
  gCPU.instanceIdOf.erase(copy.in_data.effect_ref);gCPU.ownerOf.erase(copy.in_data.effect_ref);
  printf("  fixed panel: real CPU renders, copies and owner changes\n");
}

static void TestGPUAnalysisPanel(EffectMainFn fn,PrGPUFilterEntry entry) {
  if (!gReal) return;
  gIsPremiere=true;gGPU.device=MTLCreateSystemDefaultDevice();gGPU.queue=[gGPU.device newCommandQueue];
  GPUEnv g;g.util.getSPBasicSuite=GetBasic;g.suites.utilFuncs=&g.util;
  PrGPUFilterInfo info{};csSDK_int32 index=0;
  CHECK(entry(PrSDKGPUFilterInterfaceVersion,&index,kPrTrue,&g.suites,&g.filter,&info)==0,"analysis GPU startup");
  UseLayout(true,kCountTopics);SetI(79,1);SetI(106,1);
  World world=MakeWorld(320,192,16,0,false,false,1);
  CPUCall a,b;
  PrepareCPU(a,true,world.w,1001,65001,{{79,1},{106,1}});
  PrepareCPU(b,true,world.w,1001,65002,{{79,1},{106,1}});
  const auto original=a.defs;
  auto input=MakePPix(PrPixelFormat_GPU_BGRA_4444_32f,320,192,0,1);
  auto output=MakePPix(PrPixelFormat_GPU_BGRA_4444_32f,320,192,0,2);
  PPixHand ih=(PPixHand)&input,oh=(PPixHand)&output;
  PrGPUFilterRenderParams rp{};rp.inRenderTicksPerFrame=kTPF;rp.inQuality=kPrRenderQuality_High;
  rp.inDownsampleFactorX=rp.inDownsampleFactorY=1;rp.inRenderFieldType=prFieldsNone;
  auto render=[&](int id,int node,int timeline,int track,int start,float amplitude) {
    SetNode(id,start,start+100,start,start+100);
    SetProp(kOwnerNode,kVideoSegmentProperty_Clip_TrackID,track);
    // Rebuild the host node while keeping the runtime instance ID.
    SetProp(node,kVideoSegmentProperty_Effect_RuntimeInstanceID,id);
    SetProp(node,kVideoSegmentProperty_Effect_EffectDuration,100*kTPF);
    PrGPUFilterInstance inst{};inst.piSuites=&g.suites;inst.inNodeID=node;inst.inTimelineID=timeline;
    CHECK(g.filter.CreateInstance(&inst)==0&&inst.ioPrivatePluginData,"analysis GPU instance created");
    if(!inst.ioPrivatePluginData)return;
    for(int i=0;i<4;++i) {
      auto px=(float*)input.buffer.contents;FillNoisy(px,320*192,2301+i);
      for(int j=0;j<320*192;++j)for(int c=0;c<3;++c)px[4*j+c]=.2f+amplitude*(px[4*j+c]-.2f);
      rp.inSequenceTime=(start+10+i)*kTPF;rp.inClipTime=(10+i)*kTPF;
      CHECK(g.filter.Render(&inst,&rp,&ih,1,&oh)==0,"real GPU analysis render");WaitForGPU();
    }
    g.filter.DisposeInstance(&inst);
  };
  const long logStart=PluginLogSize();
  PanelRefresh(fn,a); CheckFixedPanel(a);
  render(65001,kEffectNode,3,2,1000,1);
  CHECK(PanelChange(fn,a,73).empty(),"retired button is inert after GPU render");
  CheckFixedPanel(a);
  PanelRefresh(fn,b); CheckFixedPanel(b);
  render(65002,kEffectNode+10,77,3,1000,.04f);
  PanelRefresh(fn,b);PanelRefresh(fn,a);CheckFixedPanel(b);CheckFixedPanel(a);
  render(65001,kEffectNode+20,99,2,1000,1);PanelRefresh(fn,a);CheckFixedPanel(a);
  render(65001,kEffectNode+30,103,4,1000,1);PanelRefresh(fn,a);CheckFixedPanel(a);
  render(65001,kEffectNode+40,104,2,1000,1);PanelRefresh(fn,a);CheckFixedPanel(a);
  const auto log=PluginLogFrom(logStart);
  CHECK(log.find("UI analysis:")==std::string::npos && log.find("Analysis conflict:")==std::string::npos,
        "GPU never publishes UI analysis");
  for(int pf=1;pf<odplug::kNumParams;++pf) if(kExpect[pf-1].type!=kAR)
    CHECK(PanelValue(a.defs[pf])==PanelValue(original[pf]),"GPU analysis UI preserves stored control %d",pf);
  index=0;entry(PrSDKGPUFilterInterfaceVersion,&index,kPrFalse,&g.suites,&g.filter,&info);
  printf("  fixed GPU panel: copies, node/timeline churn and owner changes\n");
}
