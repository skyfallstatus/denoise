// SPDX-License-Identifier: MIT
// UI parameter state, included inside the CPU entry namespace.
// Never called by a render. UPDATE_PARAMS_UI changes only copies of UI state.

bool HasSavedCapture(PF_InData* in_data, PF_ParamDef* params[]) {
  CaptureBytes bytes; Capture c;
  if (ReadCaptureHandle(in_data,params[PfIndex(55)]->u.arb_d.value,bytes) || !DecodeCapture(bytes,c) || !c.count)
    return false;
  if (c.revision != uint32_t(params[PfIndex(54)]->u.sd.value)) return false;
  const auto digest = DigestCapture(bytes);
  for (int i=0;i<8;++i) if (digest[i] != uint32_t(params[PfIndex(56+i)]->u.sd.value)) return false;
  return true;
}

bool PanelDisabled(int id, const ODParams& p, bool captured) {
  if (p.sharpen == 0 && (id == 37 || id == 38 || id == 39 || id == 90 || id == 40)) return true;
  if (p.quality == 0 && id == 38) return true;
  if (p.temporalRadius == 0 && (id == 33 || id == 81 || id == 34 || id == 49)) return true;
  if (captured && id == 43) return true;
  if (p.lumaStrength == 0 && (id == 19 || id == 77 || (id >= 15 && id <= 18) ||
                            (id >= 82 && id <= 85) || id == 49)) return true;
  if (p.chromaStrength == 0 && ((id >= 27 && id <= 30) || (id >= 86 && id <= 89) || id == 50)) return true;
  return false;
}

std::string DBConversionReason(PF_InData*, PF_ParamDef*[], UIParams&, bool logMethod = false);

PF_Err UpdatePanelUI(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], const char* refusal = nullptr) {
  if (!params) return PF_Err_NONE;
  // Click feedback stays a host message, never a parameter name.
  if (refusal && *refusal) snprintf(out_data->return_msg,sizeof out_data->return_msg,"%s",refusal);
  const UIParams values = ReadParams(params);
  const bool previous = EffectiveControlSet(values) == OD_CONTROLS_PREVIOUS;
  // Auto follows the host's requested quality; never infer playback intent.
  const ODParams p = ToCoreParams(values,in_data->quality == PF_Quality_HI,true);
  const bool captured = params[PfIndex(52)]->u.pd.value == OD_PROFILE_CAPTURED && HasSavedCapture(in_data,params);
  Suite<PF_ParamUtilsSuite3> suite(in_data,kPFParamUtilsSuite,kPFParamUtilsSuiteVersion3);
  if (!suite || !suite->PF_UpdateParamUI) return PF_Err_NONE;
  PF_Err firstError = PF_Err_NONE;
  for (int id : kParamOrder) {
    // These registrations are fixed for the lifetime of the panel. In
    // particular, neither the retired row nor Help is ever renamed.
    if (id == OD_P_STATUS_NOISE || id == OD_P_STATUS_HELP || IsUnusedTopic(id)) continue;
    PF_ParamDef copy = *params[PfIndex(id)];
    // Values, keyframes, original flags and change_flags stay untouched.
    copy.uu.change_flags = PF_ChangeFlag_NONE;
    if (kUseDynamicVisibility && (IsPreviousOnly(id) || IsDBOnly(id))) {
      const bool visible = IsPreviousOnly(id) ? previous : !previous;
      if (visible) copy.ui_flags &= ~PF_PUI_INVISIBLE;
      else copy.ui_flags |= PF_PUI_INVISIBLE;
    }
    if (id == OD_P_SWITCH_DB) {
      if (previous) copy.ui_flags &= ~PF_PUI_INVISIBLE;
      else if (kUseDynamicVisibility) copy.ui_flags |= PF_PUI_INVISIBLE;
      // The fallback must not depend on the hide flag being honoured.
      if (!previous) copy.ui_flags |= PF_PUI_DISABLED;
      else copy.ui_flags &= ~PF_PUI_DISABLED;
    } else if (PanelDisabled(id,p,captured)) copy.ui_flags |= PF_PUI_DISABLED;
    else copy.ui_flags &= ~PF_PUI_DISABLED;
    if (id == OD_P_GRAIN_MODE)
      snprintf(copy.PF_DEF_NAME,sizeof copy.PF_DEF_NAME,"%s",previous ? "Minimum grain applies to" : "Keep grain applies to");
    if (id == OD_P_PROFILE_POLICY)
      snprintf(copy.PF_DEF_NAME,sizeof copy.PF_DEF_NAME,"%s",previous ? "Profile policy" : "Profile");
    // Capture rows have their own freshly computed names; do not overwrite.
    if (id == OD_P_PROFILE_CAPTURE || id == OD_P_PROFILE_INFO) continue;
    const PF_Err e = suite->PF_UpdateParamUI(in_data->effect_ref,PfIndex(id),&copy);
    if (!firstError && e) firstError = e;
  }
  return firstError;
}

void SetPanelValue(PF_ParamDef* d, double value) {
  double old = d->param_type == PF_Param_POPUP ? d->u.pd.value :
               d->param_type == PF_Param_SLIDER ? d->u.sd.value : d->u.fs_d.value;
  if (old == value) return;
  if (d->param_type == PF_Param_POPUP) d->u.pd.value = (A_long)value;
  else if (d->param_type == PF_Param_SLIDER) d->u.sd.value = (A_long)value;
  else d->u.fs_d.value = value;
  d->uu.change_flags |= PF_ChangeFlag_CHANGED_VALUE;
}

// Premiere may not implement PF_GetKeyframeCount. This bounded fallback is
// deliberately a sampling test, not proof that no key exists between samples.
// It checks values only, never pixels, and changes nothing before all pass.
std::string SamplePreviousAnimation(PF_InData* in_data, PF_ParamDef* params[], bool logMethod) {
  if (!in_data->inter.checkout_param || !in_data->inter.checkin_param || !in_data->time_scale)
    return "Keyframes unknown: kept old";
  const int64_t step = std::max<A_long>(1, in_data->time_step > 0 ? in_data->time_step : in_data->local_time_step);
  // PF time is effect-sequence time starting at zero, not timeline ticks.
  // PF total_time is in time_scale units; the Premiere utility's fallback
  // duration is in frames and must be multiplied by the PF frame step.
  int64_t duration = in_data->total_time;
  if (duration <= 0 && IsPremiere(in_data) && (in_data->time_step > 0 || in_data->local_time_step > 0)) {
    Suite<PF_UtilitySuite4> util(in_data,kPFUtilitySuite,kPFUtilitySuiteVersion4);
    A_long frames = 0;
    if (util && util->GetClipDuration && !util->GetClipDuration(in_data->effect_ref,&frames) && frames > 0)
      duration = int64_t(frames)*step;
  }
  if (duration <= 0 || duration > INT32_MAX) return "Keyframes unknown: kept old";
  const int64_t lastFrame = (duration-1)/step;
  std::vector<A_long> times;
  // Start, last frame, and sixteen interior points, deduplicated for shorts.
  for (int i=0;i<18;++i) {
    const A_long t = A_long((lastFrame*i/17)*step);
    if (times.empty() || times.back()!=t) times.push_back(t);
  }
  if (logMethod) Log("Switch to dB: animation check=parameter sampling, samples=%zu, first=%d last=%d scale=%u",
                     times.size(),int(times.front()),int(times.back()),unsigned(in_data->time_scale));
  auto value=[](const PF_ParamDef& d,double& v) {
    switch(d.param_type) {
      case PF_Param_FLOAT_SLIDER: v=d.u.fs_d.value; break;
      case PF_Param_SLIDER: v=d.u.sd.value; break;
      case PF_Param_POPUP: v=d.u.pd.value; break;
      case PF_Param_CHECKBOX: v=d.u.bd.value; break;
      default: return false;
    }
    return std::isfinite(v);
  };
  for (int id:kPreviousOnly) {
    if (id==OD_P_LOOK) continue;
    const int pf=PfIndex(id); double current=0;
    if (!params[pf] || !value(*params[pf],current)) return "Keyframes unknown: kept old";
    for (A_long t:times) {
      PF_ParamDef sample{};
      const PF_Err err=PF_CHECKOUT_PARAM(in_data,pf,t,A_long(step),in_data->time_scale,&sample);
      double sampled=0;
      const bool known=!err && sample.param_type==params[pf]->param_type && value(sample,sampled);
      // Check in even a failed checkout, as in the render's Checkout guard.
      const PF_Err checked=PF_CHECKIN_PARAM(in_data,&sample);
      if (!known || checked) return "Keyframes unknown: kept old";
      if (sampled!=current) return "Animated: keyframes kept";
    }
  }
  return {};
}

std::string DBConversionReason(PF_InData* in_data, PF_ParamDef* params[], UIParams& after, bool logMethod) {
  const UIParams before=ReadParams(params);
  if (EffectiveControlSet(before) != OD_CONTROLS_PREVIOUS) return {};
  std::string reason;
  if (!ConvertToDB(before,after,reason)) return reason;
  Suite<PF_ParamUtilsSuite3> suite(in_data,kPFParamUtilsSuite,kPFParamUtilsSuiteVersion3);
  bool sample = !suite || !suite->PF_GetKeyframeCount;
  if (!sample) for (int id : kPreviousOnly) {
    if (id == OD_P_LOOK) continue;
    PF_KeyIndex count = PF_KeyIndex_NONE;
    if (suite->PF_GetKeyframeCount(in_data->effect_ref,PfIndex(id),&count) != PF_Err_NONE) {
      sample = true; continue;
    }
    // A successful count still protects a lone key or equal-valued keys.
    if (count != PF_KeyIndex_NONE && count != 0) {
      if (logMethod) Log("Switch to dB: animation check=keyframe count, animated");
      return "Animated: keyframes kept";
    }
  }
  if (sample) {
    if (logMethod) Log("Switch to dB: keyframe API unavailable, animation check=parameter sampling");
    return SamplePreviousAnimation(in_data,params,logMethod);
  }
  if (logMethod) Log("Switch to dB: animation check=keyframe count, static");
  return {};
}

std::string SwitchToDB(PF_InData* in_data, PF_ParamDef* params[]) {
  if (EffectiveControlSet(ReadParams(params)) != OD_CONTROLS_PREVIOUS) return {};
  UIParams after;
  const std::string reason = DBConversionReason(in_data,params,after,true);
  if (!reason.empty()) return reason;
  // Commit only after all checks succeed. All flags belong to this single
  // USER_CHANGED_PARAM transaction, as for Look. Processing is never written.
  for (int id=75;id<=101;++id) SetPanelValue(params[PfIndex(id)],*RenderField(after,id));
  for (int id : kPreviousOnly) {
    if (id == OD_P_LOOK) SetPanelValue(params[PfIndex(id)],OD_DFLT_LOOK);
    else SetPanelValue(params[PfIndex(id)],*RenderField(after,id));
  }
  SetPanelValue(params[PfIndex(OD_P_CONTROL_SET)],OD_CONTROLS_DB);
  double v[kDBLookSetSize];
  for (int i=0;i<kDBLookSetSize;++i) v[i] = *RenderField(after,kDBLookSet[i]);
  SetPanelValue(params[PfIndex(OD_P_DB_LOOK)],MatchingDBLook(v));
  return {};
}

bool OpenBundledHelp() {
  Dl_info image{};
  if (!dladdr((const void*)&OpenBundledHelp,&image) || !image.dli_fname) return false;
  NSString* binary = [NSString stringWithUTF8String:image.dli_fname];
  NSString* contents = [[binary stringByDeletingLastPathComponent] stringByDeletingLastPathComponent];
  NSURL* url = [NSURL fileURLWithPath:[contents stringByAppendingPathComponent:@"Resources/help/index.html"]];
  if (![[NSFileManager defaultManager] fileExistsAtPath:url.path]) return false;
  // No host objects captured across the dispatch. NSWorkspace is a UI API.
  if ([NSThread isMainThread]) return [[NSWorkspace sharedWorkspace] openURL:url];
  dispatch_async(dispatch_get_main_queue(), ^{ [[NSWorkspace sharedWorkspace] openURL:url]; });
  return true;
}

PF_Err PanelUserChanged(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_UserChangedParamExtra* e) {
  std::string reason;
  if (e && IdAtPf(e->param_index) == OD_P_SWITCH_DB) reason = SwitchToDB(in_data,params);
  if (e && IdAtPf(e->param_index) == OD_P_STATUS_HELP && !OpenBundledHelp()) reason = "Help unavailable in this bundle";
  return UpdatePanelUI(in_data,out_data,params,reason.c_str());
}
