// SPDX-License-Identifier: MIT
// No timing sleeps: controlled completion latencies exercise policy and epochs.
#include <cstdio>
#include <cmath>
#include "quality_governor.h"
static int checks = 0, failures = 0;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); } } while (0)
struct Run {
  od::QualityGovernor g;
  double wall = 0, timeline = 0, interval = 40;
  ODQualityRequest r = {1, 40, 0, 0};
  ODQualityDecision frame(double latency, uint64_t config = 1, bool realtime = true, int q = 0) {
    wall += interval; timeline += interval; r.timelineTimeMs = timeline;
    auto d = g.begin(r, realtime, q, config, wall);
    g.complete({d.renderId,d.runId,d.quality,0,latency,latency,d.budgetMs},wall+latency);
    return d;
  }
};
int main() {
  CHECK(od::QualityGovernor::budget(40) == 20);
  CHECK(od::QualityGovernor::budget(20) == 12);
  CHECK(od::QualityGovernor::budget(1000.0/60) == 10);
  { Run t; for (int i=0;i<300;++i) { auto d=t.frame(14); CHECK(d.quality==1 && d.switches==0); } }
  { Run t; t.r.frameDurationMs=t.interval=20;
    for (int i=0;i<60;++i) t.frame(15);
    auto d=t.frame(9); CHECK(d.quality==0 && d.switches==1 && d.budgetMs==12);
    for (int i=0;i<400;++i) CHECK(t.frame(5).quality==0); // no mid-run recovery pumping
    t.wall+=500; d=t.frame(5); CHECK(d.quality==1 && d.switches==0); // later run, ample headroom
  }
  { Run t; t.r.frameDurationMs=t.interval=20;
    // Alternating outliers / safe frames cannot satisfy sustained overload.
    for (int i=0;i<300;++i) CHECK(t.frame(i%2 ? 16 : 8).quality==1);
    for (int i=0;i<30;++i) t.frame(16);
    CHECK(t.frame(8).quality==0);
    t.wall+=500; CHECK(t.frame(8).quality==0); // short stop/start doesn't retry
    // Export remains Full even after playback overload; explicit 0 stays 0.
    CHECK(t.frame(100,1,false).quality==1);
    t.r.automatic=0; CHECK(t.frame(100,1,false,0).quality==0);
    CHECK(t.frame(100,1,true,1).quality==1);
  }
  { Run t; t.r.frameDurationMs=t.interval=20;
    for (int i=0;i<60;++i)t.frame(15);
    CHECK(t.frame(9).quality==0);
    t.r.frameDurationMs=t.interval=40; CHECK(t.frame(14).quality==1); // 25 fps fits measured Full
  }
  { Run t; t.r.frameDurationMs=0; t.interval=20;
    for (int i=0;i<5;++i) t.frame(8);
    CHECK(t.frame(8).budgetMs==12);
    t.timeline+=1000; CHECK(t.frame(8).budgetMs==12); // seeks aren't sequence rate
    t.r.frameDurationMs=NAN; CHECK(std::isfinite(t.frame(8).budgetMs));
  }
  { Run t; auto d=t.frame(8); t.g.reset(); auto fresh=t.frame(8);
    for(int i=0;i<100;++i)t.g.complete({d.renderId,d.runId,1,0,200,200,20},t.wall);
    CHECK(t.frame(8).quality==1 && fresh.runId!=d.runId);
    for(int i=0;i<60;++i)t.frame(30);
    CHECK(t.frame(8).quality==0);
    CHECK(t.frame(8,2).quality==1); // raster/settings isolate history
  }
  { Run t; for(int i=0;i<4;++i) t.frame(200); // cold starts are visible but not training
    for(int i=0;i<60;++i)CHECK(t.frame(8).quality==1);
    auto d=t.frame(8);
    for(int i=0;i<50;++i) {
      t.g.complete({d.renderId,d.runId,1,1,200,200,20},t.wall); // failure isn't cost
      t.g.complete({d.renderId,d.runId,1,0,NAN,0,20},t.wall);
    }
    CHECK(t.frame(8).quality==1);
  }
  { od::QualityGovernor g; ODQualityRequest r={1,0,NAN,0}; ODQualityDecision d={};
    for(int i=0;i<10;++i)d=g.begin(r,true,0,1,i*20);
    CHECK(d.budgetMs==12); // final fallback: call-arrival cadence
    r.frameDurationMs=40; d=g.begin(r,true,0,1,200); CHECK(d.budgetMs==20); // host wins
  }
  { Run t; t.r.frameDurationMs=t.interval=20;
    // Explicitly identified playback: observed 11.5 ms GPU + 41 ms queued
    // cannot fit. Changing only timeline steps does NOT create that intent.
    for (int i=0;i<80;++i) { t.timeline += i%6==0 ? 20 : 0; t.frame(52.5); }
    auto d=t.frame(8); CHECK(d.quality==0 && d.switches==1);
    for (int i=0;i<80;++i) CHECK(t.frame(7.7).quality==0);
  }
  { Run t;
    for(int i=0;i<80;++i) CHECK(t.frame(11.6).quality==1); // 25p, fits 20 ms allowance
  }
  { Run t; t.r.frameDurationMs=20; t.interval=30;
    for(int i=0;i<80;++i) CHECK(t.frame(52.5,1,false).quality==1); // slow GPU-bound export
    for(int i=0;i<80;++i) {
      t.timeline += i%6==0 ? 20 : 0;
      CHECK(t.frame(52.5,1,false).quality==1); // omitted/cached frames are not playback intent
    }
  }
  printf("quality governor: %d checks, %d failures\n",checks,failures);
  return failures ? 1 : 0;
}
