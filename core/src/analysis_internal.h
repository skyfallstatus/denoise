// SPDX-License-Identifier: MIT
#ifndef OD_ANALYSIS_INTERNAL_H
#define OD_ANALYSIS_INTERNAL_H
#include "stages/stage.h"
namespace od {
struct AnalysisReadback {
  ODAnalysis snapshot = {};
  id<MTLBuffer> data = nil;
  id<MTLBuffer> scratch = nil;
  id<MTLBuffer> tiles = nil;
};
class AnalysisStage {
 public:
  static std::unique_ptr<AnalysisStage> make(Context&, std::string*);
  AnalysisReadback encode(RenderEnv&, const ODFrame& source, id<MTLTexture> result, const ProfileAnalysisInfo&, uint64_t,
                          bool frozen, bool captured);
  static ODAnalysis resolve(const AnalysisReadback&);
  bool overlay(RenderEnv&, const ODFrame&, const ODAnalysis*);
 private:
  id<MTLComputePipelineState> tiles_, histogram_, reduce_, overlay32_, overlay16_;
};
}
#endif
