// SPDX-License-Identifier: MIT
#pragma once
#include "opendenoise/opendenoise.h"
namespace od {
// Reflow for an in-picture panel seen at Fit; native Effect Controls retain
// od_analysis_layout's existing compact layout and literal pixel font sizes.
int analysisViewLayout(const ODAnalysis*, const ODParams*, ODDrawList*, float width, float height);
// The public box is the full data panel; an unavailable analysis uses a
// smaller two-line box at the same text scale and bottom-left safe margin.
void analysisViewBox(const ODAnalysis*, uint32_t frameWidth, uint32_t frameHeight,
                     uint32_t* x, uint32_t* y, uint32_t* width, uint32_t* height);
}
