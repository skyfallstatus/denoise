// SPDX-License-Identifier: MIT
// Retired disk ID 105 still round-trips through EffectMain, without custom UI.
static void TestReservedParam(EffectMainFn fn) {
  auto utils=CaptureUtils(); World in=MakeWorld(32,32,16,0,false,false,1);
  CPUCall c; PrepareCPU(c,true,in.w,0,8899); c.in_data.utils=&utils;
  const auto& row=c.defs[PfOf(105)];
  CHECK(row.param_type==PF_Param_ARBITRARY_DATA &&
        (row.ui_flags & (PF_PUI_NO_ECW_UI | PF_PUI_INVISIBLE)) == (PF_PUI_NO_ECW_UI | PF_PUI_INVISIBLE) &&
        !(row.ui_flags & (PF_PUI_CONTROL | PF_PUI_TOPIC)) && row.ui_width==0 && row.ui_height==0,
        "ID 105 remains registered and permanently hidden without drawing flags or size");
  CHECK(gRegisteredUI==0,"no custom UI registered");
  PF_Context ctx{}; ctx.w_type=PF_Window_EFFECT; auto ctxP=&ctx;
  for(auto type:{PF_Event_DRAW,PF_Event_DO_CLICK,PF_Event_DRAG,PF_Event_ADJUST_CURSOR}) {
    PF_EventExtra e{}; e.contextH=&ctxP; e.e_type=type;
    e.effect_win.index=PfOf(105); e.effect_win.area=PF_EA_CONTROL;
    const auto before=e; const auto defs=c.defs;
    const int checkouts=gCPU.checkouts, acquisitions=gAcquireAttempts.load();
    c.out_data={};
    CHECK(fn(PF_Cmd_EVENT,&c.in_data,&c.out_data,c.ptrs.data(),nullptr,&e)==0,"retired row event is benign");
    CHECK(!memcmp(&e,&before,sizeof e) && !c.out_data.out_flags,"no event handling or UI refresh");
    CHECK(gCPU.checkouts==checkouts && gAcquireAttempts==acquisitions &&
          !memcmp(defs.data(),c.defs.data(),defs.size()*sizeof(PF_ParamDef)),
          "no drawing, suite access, checkout, value or undo side effects");
  }

  auto arb=[&](PF_ArbParamsExtra& a) { a.id=105; return fn(PF_Cmd_ARBITRARY_CALLBACK,&c.in_data,&c.out_data,nullptr,nullptr,&a); };
  PF_Handle a=nullptr,b=nullptr,d=nullptr; unsigned char flat=0; A_u_long size=0;
  const auto live=gCaptureHandles.size();
  PF_ArbParamsExtra x{}; x.which_function=PF_Arbitrary_NEW_FUNC; x.u.new_func_params.arbPH=&a;
  CHECK(arb(x)==0 && a,"profile row arbitrary NEW");
  x={};x.which_function=PF_Arbitrary_COPY_FUNC;x.u.copy_func_params.src_arbH=nullptr;x.u.copy_func_params.dst_arbPH=&b;
  CHECK(arb(x)==0 && b,"profile row COPY(NULL)");
  x={};x.which_function=PF_Arbitrary_FLAT_SIZE_FUNC;x.u.flat_size_func_params.flat_data_sizePLu=&size;
  CHECK(arb(x)==0 && size==1,"profile row has no profile/pointer in its one-byte payload");
  x={};x.which_function=PF_Arbitrary_FLATTEN_FUNC;x.u.flatten_func_params.arbH=a;x.u.flatten_func_params.flat_dataPV=&flat;x.u.flatten_func_params.buf_sizeLu=1;
  CHECK(arb(x)==0 && flat==1,"profile row flatten version");
  x={};x.which_function=PF_Arbitrary_UNFLATTEN_FUNC;x.u.unflatten_func_params.flat_dataPV=&flat;x.u.unflatten_func_params.buf_sizeLu=1;x.u.unflatten_func_params.arbPH=&d;
  CHECK(arb(x)==0 && d,"profile row unflatten");
  flat=2; PF_Handle rejected=nullptr; x.u.unflatten_func_params.arbPH=&rejected;
  CHECK(arb(x)!=0 && !rejected,"unknown UI payload version rejected");
  PF_ArbCompareResult equal=PF_ArbCompare_NOT_EQUAL;
  x={};x.which_function=PF_Arbitrary_COMPARE_FUNC;x.u.compare_func_params.a_arbH=a;x.u.compare_func_params.b_arbH=d;x.u.compare_func_params.compareP=&equal;
  CHECK(arb(x)==0 && equal==PF_ArbCompare_EQUAL,"cosmetic row never changes picture identity");
  for(auto h:{a,b,d}) { x={};x.which_function=PF_Arbitrary_DISPOSE_FUNC;x.u.dispose_func_params.arbH=h;CHECK(arb(x)==0,"dispose UI payload"); }
  CHECK(gCaptureHandles.size()==live,"all UI resources released");
  printf("  retired ID 105: hidden, no UI registration/events/suite calls, original one-byte persistence intact\n");
}
