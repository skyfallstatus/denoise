# SPDX-License-Identifier: MIT
# Denoise — plain Makefile, Command Line Tools only (no Xcode, no `metal`).
#
#   make core        $(BUILD)/libopendenoise.a
#   make odbench     $(BUILD)/odbench
#   make test        build and run every suite (tests/*/test.mk), and the plug-in
#                    harness against the real core when the Adobe SDKs are found
#   make plugin      $(BUILD)/Denoise.plugin linked with libopendenoise.a (plugin/Makefile;
#                    needs PREMIERE_SDK and AE_SDK)
#   make plugin-test plug-in harness + bundle verifier
#   make clean       removes $(BUILD)
#
# Every target honours BUILD=<dir> (default build).
# SDK paths belong in untracked local.mk or the environment.
#
# Sources are found by wildcard: a new .mm/.cpp under core/src or core/src/*,
# or a new .metal under core/shaders, is picked up without editing this file.
# Shaders are embedded as source strings by scripts/embed_shaders.py (rules
# for shader files are documented there) and compiled at run time.

-include local.mk
.DEFAULT_GOAL := all

BUILD ?= build
CXX ?= clang++
AR ?= ar
PYTHON ?= /usr/bin/python3
MACOSX_DEPLOYMENT_TARGET ?= 14.0
export MACOSX_DEPLOYMENT_TARGET

OPT ?= -O2
CXXFLAGS ?=
OD_CXXFLAGS := -arch arm64 -std=c++17 $(OPT) -g -fobjc-arc -Wall -Wextra -Wno-unused-parameter \
                 -Icore/include -Icore/src $(CXXFLAGS)
OD_LDFLAGS := -arch arm64 -framework Metal -framework Foundation -framework CoreGraphics -framework CoreText $(LDFLAGS)

OBJ := $(BUILD)/obj
GEN := $(BUILD)/gen

CORE_SRCS := $(wildcard core/src/*.mm core/src/*.cpp core/src/*/*.mm core/src/*/*.cpp)
CORE_OBJS := $(patsubst %,$(OBJ)/%.o,$(CORE_SRCS)) $(OBJ)/gen/od_shaders.cpp.o
SHADERS := $(sort $(wildcard core/shaders/*.metal))

BENCH_SRCS := $(wildcard tools/odbench/*.mm tools/odbench/*.cpp)
BENCH_OBJS := $(patsubst %,$(OBJ)/%.o,$(BENCH_SRCS))

TEST_SRCS := $(wildcard tests/core/*.mm)
TEST_BINS := $(patsubst tests/core/%.mm,$(BUILD)/tests/%,$(TEST_SRCS))

# Deadline for anything `make test` runs (seconds). There is no `timeout`
# binary on the reference machine.
TEST_DEADLINE ?= 300
DEADLINE = perl -e 'alarm shift; exec @ARGV or die "exec: $$!"' $(TEST_DEADLINE)

.PHONY: all core odbench test test-core test-build plugin plugin-test sdk-check clean FORCE
all: core odbench

core: $(BUILD)/libopendenoise.a
odbench: $(BUILD)/odbench

$(BUILD)/libopendenoise.a: $(CORE_OBJS)
	@mkdir -p $(dir $@)
	@rm -f $@
	$(AR) rcs $@ $^

# The list of shader files, rewritten only when it changes, so adding or
# deleting a .metal file re-runs the embed step.
$(GEN)/shaders.list: FORCE
	@mkdir -p $(GEN)
	@echo "$(SHADERS)" > $@.new
	@if cmp -s $@.new $@; then rm -f $@.new; else mv $@.new $@; fi

$(GEN)/od_shaders.cpp: $(SHADERS) $(GEN)/shaders.list scripts/embed_shaders.py
	@mkdir -p $(GEN)
	$(PYTHON) scripts/embed_shaders.py $@ core/shaders

$(OBJ)/gen/od_shaders.cpp.o: $(GEN)/od_shaders.cpp core/src/shaders.h
	@mkdir -p $(dir $@)
	$(CXX) $(OD_CXXFLAGS) -c $< -o $@

$(OBJ)/%.mm.o: %.mm
	@mkdir -p $(dir $@)
	$(CXX) $(OD_CXXFLAGS) -MMD -MP -c $< -o $@

$(OBJ)/%.cpp.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(OD_CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/odbench: $(BENCH_OBJS) $(BUILD)/libopendenoise.a
	@mkdir -p $(dir $@)
	$(CXX) $(BENCH_OBJS) $(BUILD)/libopendenoise.a $(OD_LDFLAGS) -framework ImageIO -o $@

$(BUILD)/tests/%: $(OBJ)/tests/core/%.mm.o $(BUILD)/libopendenoise.a
	@mkdir -p $(dir $@)
	$(CXX) $< $(BUILD)/libopendenoise.a $(OD_LDFLAGS) -o $@

# Optional PNG evidence from the bounded in-picture overlay test.
$(BUILD)/tests/test_analysis_overlay: OD_LDFLAGS += -framework ImageIO

test-core: $(TEST_BINS)
	@set -e; for t in $(TEST_BINS); do echo "== $$t"; $(DEADLINE) $$t; done

# Checks of this Makefile itself (scripts/test_build.sh): one plug-in
# sub-make per invocation, the SDK message. Dry runs only, a few seconds.
test-build:
	@$(DEADLINE) scripts/test_build.sh

# Other test suites hook in with tests/<suite>/test.mk, adding prerequisites
# to `test` (they can use BUILD, DEADLINE, $(BUILD)/libopendenoise.a, $(BUILD)/odbench).
test: test-core test-build
-include $(wildcard tests/*/test.mk)

# The plug-in has its own Makefile, run with make -C plugin. It gets OD_ROOT
# and an absolute BUILD, which it turns back into paths relative to plugin/
# (make cannot quote target names, and the repository path may contain
# spaces), and the SDK locations, which are only used quoted in recipes.
# OD_CORE=lib links the libopendenoise.a built here; OD_CORE=stub builds the
# shell around a pass-through core.
OD_CORE ?= lib
# The two Adobe SDK folders (README, "Build"). No default:
# pass them on the command line, export them, or set them in local.mk.
PREMIERE_SDK ?=
AE_SDK ?=
PLUGIN_MAKE = $(MAKE) -C plugin OD_ROOT="$(CURDIR)" BUILD="$(abspath $(BUILD))" OD_CORE=$(OD_CORE) \
	PREMIERE_SDK="$(PREMIERE_SDK)" AE_SDK="$(AE_SDK)"

HAVE_SDKS := $(shell test -f "$(PREMIERE_SDK)/Examples/Headers/PrSDKGPUFilter.h" && test -f "$(AE_SDK)/Examples/Headers/AE_Effect.h" && echo yes)

sdk-check:
	@test -f "$(PREMIERE_SDK)/Examples/Headers/PrSDKGPUFilter.h" || { \
	  echo "PREMIERE_SDK not set or not found ('$(PREMIERE_SDK)'): pass PREMIERE_SDK=\"/path/to/Premiere Pro 26.0 C++ SDK\" (the folder with Examples/Headers/PrSDKGPUFilter.h), export it, or set it in local.mk; see README, \"Build\"" >&2; exit 1; }
	@test -f "$(AE_SDK)/Examples/Headers/AE_Effect.h" || { \
	  echo "AE_SDK not set or not found ('$(AE_SDK)'): pass AE_SDK=\"/path/to/AdobeAfterEffectsSDK_26.5_MacOS\" (the folder with Examples/Headers/AE_Effect.h), export it, or set it in local.mk; see README, \"Build\"" >&2; exit 1; }

# One plug-in sub-make per invocation. `plugin` and `plugin-test` each run
# `make -C plugin`, and two of those linking, running Rez and signing the same
# bundle at once fail under -j. So when the harness runs in the same invocation
# (`make plugin plugin-test`, or `make test plugin` with the SDKs found), the
# harness's sub-make builds the bundle and `plugin` only waits for it.
# The `+` marks the sub-make as recursive: it shares the -j job slots, and a
# dry run (make -n) runs it too, which scripts/test_build.sh relies on.
PLUGIN_BY_HARNESS := $(if $(filter plugin-test,$(MAKECMDGOALS)),yes,$(if $(and $(filter test,$(MAKECMDGOALS)),$(HAVE_SDKS)),yes))
ifeq ($(PLUGIN_BY_HARNESS),yes)
plugin: plugin-test
	@:
else
plugin: core sdk-check
	@+$(PLUGIN_MAKE) plugin
endif

plugin-test: core sdk-check
	@+$(PLUGIN_MAKE) plugin-test plugin-verify

# `make test` runs the plug-in harness too when both SDKs are present.
ifeq ($(HAVE_SDKS),yes)
test: plugin-test
else
test: no-sdk-note
.PHONY: no-sdk-note
no-sdk-note:
	@echo "note: Adobe SDKs not found (PREMIERE_SDK, AE_SDK): the plug-in harness was not run"
endif

clean:
	rm -rf "$(BUILD)"

-include $(CORE_OBJS:.o=.d) $(BENCH_OBJS:.o=.d) $(patsubst %,$(OBJ)/%.d,$(TEST_SRCS))
.SECONDARY:
