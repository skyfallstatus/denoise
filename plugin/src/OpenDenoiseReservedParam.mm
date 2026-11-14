// SPDX-License-Identifier: MIT
// Persistence only for retired disk ID 105. Never draw or handle UI events.
#include "AEConfig.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "OpenDenoisePlugin.h"
#include "OpenDenoiseReservedParam.h"
#include <cstring>

namespace odplug {
namespace { constexpr PF_Err bad = PF_Err_INTERNAL_STRUCT_DAMAGED; }
// The original one-byte version is not profile data and cannot affect a render.
// Retain every persistence operation, including COPY(NULL), for saved projects.
PF_Err ProfilePanelArbitrary(PF_InData* in_data, PF_ArbParamsExtra* e) {
  if (!e || e->id != OD_P_PROFILE_PANEL) return bad;
  auto fresh = [&](PF_Handle* dst) -> PF_Err {
    if (!dst) return bad;
    *dst = nullptr;
    if (!in_data || !in_data->utils || !in_data->utils->host_new_handle || !in_data->utils->host_lock_handle ||
        !in_data->utils->host_unlock_handle || !in_data->utils->host_dispose_handle) return bad;
    auto h = PF_NEW_HANDLE(1);
    if (!h) return PF_Err_OUT_OF_MEMORY;
    auto p = static_cast<unsigned char*>(PF_LOCK_HANDLE(h));
    if (!p) { PF_DISPOSE_HANDLE(h); return PF_Err_OUT_OF_MEMORY; }
    *p = 1; PF_UNLOCK_HANDLE(h); *dst = h;
    return PF_Err_NONE;
  };
  switch(e->which_function) {
    case PF_Arbitrary_NEW_FUNC: return fresh(e->u.new_func_params.arbPH);
    case PF_Arbitrary_COPY_FUNC: return fresh(e->u.copy_func_params.dst_arbPH);
    case PF_Arbitrary_INTERP_FUNC: return fresh(e->u.interp_func_params.interpPH);
    case PF_Arbitrary_DISPOSE_FUNC:
      if (e->u.dispose_func_params.arbH) {
        if (!in_data || !in_data->utils || !in_data->utils->host_dispose_handle) return bad;
        PF_DISPOSE_HANDLE(e->u.dispose_func_params.arbH);
      }
      return PF_Err_NONE;
    case PF_Arbitrary_FLAT_SIZE_FUNC:
      if (!e->u.flat_size_func_params.flat_data_sizePLu) return bad;
      *e->u.flat_size_func_params.flat_data_sizePLu = 1; return PF_Err_NONE;
    case PF_Arbitrary_FLATTEN_FUNC:
      if (!e->u.flatten_func_params.flat_dataPV || e->u.flatten_func_params.buf_sizeLu != 1) return bad;
      *static_cast<unsigned char*>(e->u.flatten_func_params.flat_dataPV) = 1; return PF_Err_NONE;
    case PF_Arbitrary_UNFLATTEN_FUNC:
      if (!e->u.unflatten_func_params.flat_dataPV || e->u.unflatten_func_params.buf_sizeLu != 1 ||
          *static_cast<const unsigned char*>(e->u.unflatten_func_params.flat_dataPV) != 1) return bad;
      return fresh(e->u.unflatten_func_params.arbPH);
    case PF_Arbitrary_COMPARE_FUNC:
      if (!e->u.compare_func_params.compareP) return bad;
      *e->u.compare_func_params.compareP = PF_ArbCompare_EQUAL; return PF_Err_NONE;
    case PF_Arbitrary_PRINT_SIZE_FUNC:
      if (!e->u.print_size_func_params.print_sizePLu) return bad;
      *e->u.print_size_func_params.print_sizePLu = 2; return PF_Err_NONE;
    case PF_Arbitrary_PRINT_FUNC:
      if (!e->u.print_func_params.print_bufferPC || e->u.print_func_params.print_sizeLu < 2) return bad;
      memcpy(e->u.print_func_params.print_bufferPC,"1",2); return PF_Err_NONE;
    case PF_Arbitrary_SCAN_FUNC:
      if (!e->u.scan_func_params.bufPC ||
          (e->u.scan_func_params.bytes_to_scanLu != 1 && e->u.scan_func_params.bytes_to_scanLu != 2) ||
          e->u.scan_func_params.bufPC[0] != '1' ||
          (e->u.scan_func_params.bytes_to_scanLu == 2 && e->u.scan_func_params.bufPC[1])) return PF_Err_CANNOT_PARSE_KEYFRAME_TEXT;
      return fresh(e->u.scan_func_params.arbPH);
    default: return bad;
  }
}
} // namespace odplug
