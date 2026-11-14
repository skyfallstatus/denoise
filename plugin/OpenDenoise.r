// SPDX-License-Identifier: MIT
// Open Denoise — PiPL resource. Compiled with Rez (see plugin/Makefile).
//
// The out-flag values here MUST equal what GlobalSetup() in src/OpenDenoiseCPU.mm
// puts in out_data->out_flags / out_flags2 (After Effects refuses to load an
// effect whose PiPL and code disagree). They are passed in by the Makefile
// from src/OpenDenoisePlugin.h so there is a single source of truth.
#include "AEConfig.h"
#include "AE_EffectVers.h"
#include "AE_General.r"

#ifndef OD_OUT_FLAGS
#error "OD_OUT_FLAGS / OD_OUT_FLAGS2 / OD_PIPL_VERSION must be defined by the Makefile"
#endif

resource 'PiPL' (16000) {
	{
		Kind { AEEffect },
		Name { "Denoise" },
		Category { "Denoise" },
		CodeMacARM64 { "EffectMain" },
		AE_PiPL_Version { 2, 0 },
		AE_Effect_Spec_Version { PF_PLUG_IN_VERSION, PF_PLUG_IN_SUBVERS },
		AE_Effect_Version { OD_PIPL_VERSION },
		AE_Effect_Info_Flags { 0 },
		AE_Effect_Global_OutFlags { OD_OUT_FLAGS },
		AE_Effect_Global_OutFlags_2 { OD_OUT_FLAGS2 },
		/* Stable forever: projects and presets refer to the effect by this. */
		AE_Effect_Match_Name { "Open Denoise" },
		AE_Reserved_Info { 8 }
	}
};
