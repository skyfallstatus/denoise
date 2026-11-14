# SPDX-License-Identifier: MIT
# Denoise — Premiere Pro plug-in bundle. Command Line Tools only.
#
#   make -C plugin BUILD=build/<name> plugin        -> <BUILD>/Denoise.plugin
#   make -C plugin BUILD=build/<name> plugin-test   -> builds and runs the harness
#   make -C plugin BUILD=build/<name> plugin-verify -> structure / PiPL / symbols / signature
#
# Variables:
#   BUILD          build directory; relative paths are relative to the repo root.
#                  An absolute BUILD under OD_ROOT (what the top-level Makefile
#                  passes) is made relative, so that the repository may sit in
#                  a folder whose path contains spaces. A BUILD outside it must
#                  not contain spaces (make cannot quote target names).
#   OD_ROOT        the repository root, as the top-level Makefile passes it
#   PREMIERE_SDK   "Premiere Pro 26.0 C++ SDK" folder (contains Examples/)
#   AE_SDK         "AdobeAfterEffectsSDK_26.5_MacOS" folder (contains Examples/)
#                  No defaults: the command line, the environment, or the
#                  untracked local.mk at the repo root (as for the top level)
#   OD_CORE      lib (default: link OD_LIB, the real core) | stub (plugin/stub,
#                  a pass-through core for testing the shell alone)
#   OD_LIB       path of libopendenoise.a when OD_CORE=lib (default <BUILD>/libopendenoise.a;
#                  build it with `make core` at the top level, or use `make plugin`
#                  there, which does both)
#   OD_LIB_LDFLAGS  extra link flags the real core needs (frameworks)

# Every path that make uses as a target or prerequisite is relative to this
# directory (`make -C plugin`), never absolute: make splits target names at
# spaces and has no quoting for them. Paths in recipes are quoted.
MAKEFILE_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))
ROOT := $(if $(filter .,$(MAKEFILE_DIR)),..,$(MAKEFILE_DIR)/..)
PLUGIN_DIR := $(ROOT)/plugin

BUILD ?= build
# BUILD_REL: relative to the repo root (or absolute outside it).
# BUILD_DIR: the same directory as seen from here.
BUILD_REL := $(BUILD)
ifneq ($(OD_ROOT),)
BUILD_REL := $(subst $(OD_ROOT)/,,$(BUILD))
endif
ifeq ($(filter /%,$(BUILD_REL)),)
BUILD_DIR := $(ROOT)/$(BUILD_REL)
else
BUILD_DIR := $(BUILD_REL)
endif
ifneq ($(words $(BUILD_DIR)),1)
$(error BUILD "$(BUILD)" contains a space and is not inside the repository: use a build directory without spaces, or one inside the repository)
endif

-include $(ROOT)/local.mk
PREMIERE_SDK ?=
AE_SDK ?=

OD_CORE ?= lib
OD_LIB ?= $(BUILD_DIR)/libopendenoise.a
OD_LIB_LDFLAGS ?= -framework CoreGraphics -framework CoreText

MIN_MACOS ?= 14.0
OD_HIDE_UNUSED_TOPICS ?= 1
VERSION := 0.1.0

CXX := clang++
SDKROOT_PATH := $(shell xcrun --show-sdk-path)

OBJ := $(BUILD_DIR)/plugin-obj
BUNDLE := $(BUILD_DIR)/Denoise.plugin
BIN := $(BUNDLE)/Contents/MacOS/Denoise
RSRC := $(BUNDLE)/Contents/Resources/Denoise.rsrc
HARNESS := $(BUILD_DIR)/opendenoise-plugin-harness
BRIDGE_TEST := $(BUILD_DIR)/opendenoise-analysis-bridge-tests
PARAM_TEST := $(BUILD_DIR)/opendenoise-parameter-tests
HELP := $(BUNDLE)/Contents/Resources/help/index.html

# Header order follows the SDK_ProcAmp Xcode project: Premiere headers first,
# then the After Effects ones.
INCLUDES := -I"$(ROOT)/core/include" -I"$(PLUGIN_DIR)/src" \
  -I"$(PREMIERE_SDK)/Examples/Headers" \
  -I"$(AE_SDK)/Examples/Headers" -I"$(AE_SDK)/Examples/Headers/SP" \
  -I"$(AE_SDK)/Examples/Util" -I"$(AE_SDK)/Examples/Resources"

CXXFLAGS := -arch arm64 -std=c++17 -O2 -g -fobjc-arc -fvisibility=hidden -fvisibility-inlines-hidden \
  -mmacosx-version-min=$(MIN_MACOS) -isysroot "$(SDKROOT_PATH)" \
  -DOD_HIDE_UNUSED_TOPICS=$(OD_HIDE_UNUSED_TOPICS) -Wall -Wextra -Wno-unused-parameter -Wno-four-char-constants -Wno-multichar -Wno-pragma-pack \
  -Wno-missing-field-initializers -Wno-unused-function $(INCLUDES)

LDFLAGS := -arch arm64 -bundle -mmacosx-version-min=$(MIN_MACOS) -isysroot "$(SDKROOT_PATH)" \
  -framework Metal -framework Foundation -framework AppKit \
  -Wl,-exported_symbol,_EffectMain -Wl,-exported_symbol,_xGPUFilterEntry -Wl,-dead_strip

PLUGIN_SRCS := OpenDenoiseCPU.mm OpenDenoiseGPU.mm OpenDenoiseShared.mm OpenDenoiseCapture.mm OpenDenoiseReservedParam.mm OpenDenoiseAnalysisBridge.mm
PLUGIN_OBJS := $(PLUGIN_SRCS:%.mm=$(OBJ)/%.o)

ifeq ($(OD_CORE),stub)
CORE_OBJS := $(OBJ)/od_stub.o $(OBJ)/profile_blob.o $(OBJ)/analysis.o

$(OBJ)/analysis.o: $(ROOT)/core/src/analysis.cpp $(ROOT)/core/src/analysis_layout.h
	@mkdir -p "$(dir $@)"
	$(CXX) $(CXXFLAGS) -c "$<" -o "$@"

$(OBJ)/profile_blob.o: $(ROOT)/core/src/profile_blob.cpp
	@mkdir -p "$(dir $@)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c "$<" -o "$@"
CORE_LINK :=
else ifeq ($(OD_CORE),lib)
CORE_OBJS :=
CORE_LINK := "$(OD_LIB)" $(OD_LIB_LDFLAGS)
else
$(error OD_CORE must be 'stub' or 'lib')
endif

# Single source of truth for the numbers the PiPL must share with the code.
PIPL_DEFS := $(shell awk '/^\#define OD_OUT_FLAGS_VALUE/{printf "-d OD_OUT_FLAGS=%s ",$$3} /^\#define OD_OUT_FLAGS2_VALUE/{printf "-d OD_OUT_FLAGS2=%s ",$$3} /^\#define OD_PIPL_VERSION_VALUE/{printf "-d OD_PIPL_VERSION=%s ",$$3}' "$(PLUGIN_DIR)/src/OpenDenoisePlugin.h")

.PHONY: plugin plugin-test plugin-verify plugin-clean check-sdks
plugin: check-sdks $(BUNDLE)/Contents/_CodeSignature/CodeResources

check-sdks:
	@test -f "$(PREMIERE_SDK)/Examples/Headers/PrSDKGPUFilter.h" || { echo "PREMIERE_SDK not set or not found ('$(PREMIERE_SDK)'): pass PREMIERE_SDK=\"/path/to/Premiere Pro 26.0 C++ SDK\", export it, or set it in local.mk at the repo root" >&2; exit 1; }
	@test -f "$(AE_SDK)/Examples/Headers/AE_Effect.h" || { echo "AE_SDK not set or not found ('$(AE_SDK)'): pass AE_SDK=\"/path/to/AdobeAfterEffectsSDK_26.5_MacOS\", export it, or set it in local.mk at the repo root" >&2; exit 1; }
	@test "$(OD_CORE)" = stub || test -f "$(OD_LIB)" || { echo "OD_CORE=lib but no $(OD_LIB): run 'make core' (or 'make plugin') at the top level first"; exit 1; }

$(OBJ)/%.o: $(PLUGIN_DIR)/src/%.mm $(PLUGIN_DIR)/src/OpenDenoisePlugin.h $(PLUGIN_DIR)/src/RenderStats.h $(PLUGIN_DIR)/src/OpenDenoiseCapture.h $(PLUGIN_DIR)/src/OpenDenoiseReservedParam.h $(PLUGIN_DIR)/src/OpenDenoisePanelUI.h $(PLUGIN_DIR)/src/OpenDenoiseAnalysisBridge.h \
  $(ROOT)/core/include/opendenoise/opendenoise.h
	@mkdir -p "$(OBJ)"
	$(CXX) $(CXXFLAGS) -c "$<" -o "$@"

$(OBJ)/od_stub.o: $(PLUGIN_DIR)/stub/od_stub.mm $(ROOT)/core/include/opendenoise/opendenoise.h
	@mkdir -p "$(OBJ)"
	$(CXX) $(CXXFLAGS) -c "$<" -o "$@"

# A rebuilt libopendenoise.a relinks (an OD_LIB path with spaces cannot be a
# prerequisite: then every build relinks). Any relink removes the signature
# first, so the bundle is always signed again, whatever the timestamps.
ifeq ($(OD_CORE),lib)
LIB_PREREQ := $(if $(filter 1,$(words $(OD_LIB))),$(OD_LIB),FORCE)
ifeq ($(words $(OD_LIB)),1)
$(OD_LIB):
	@echo "OD_CORE=lib but no $(OD_LIB): run 'make core' (or 'make plugin') at the top level first"; exit 1
endif
endif
$(BIN): $(PLUGIN_OBJS) $(CORE_OBJS) $(LIB_PREREQ)
	@mkdir -p "$(dir $@)"
	@rm -rf "$(BUNDLE)/Contents/_CodeSignature"
	$(CXX) $(LDFLAGS) $(PLUGIN_OBJS) $(CORE_OBJS) $(CORE_LINK) -o "$@"

# PiPL -> data-fork resource file at Contents/Resources/<name>.rsrc, which is
# where Xcode's Rez build phase puts it for the SDK samples.
$(RSRC): $(PLUGIN_DIR)/OpenDenoise.r $(PLUGIN_DIR)/src/OpenDenoisePlugin.h
	@mkdir -p "$(dir $@)"
	Rez -o "$@" -useDF -d AE_OS_MAC=1 -d __MACH__ $(PIPL_DEFS) \
	  -i "$(AE_SDK)/Examples/Headers" -i "$(AE_SDK)/Examples/Headers/SP" -i "$(AE_SDK)/Examples/Resources" \
	  -isysroot "$(SDKROOT_PATH)" "$(PLUGIN_DIR)/OpenDenoise.r"

$(BUNDLE)/Contents/Info.plist: $(PLUGIN_DIR)/Info.plist $(PLUGIN_DIR)/Makefile
	@mkdir -p "$(dir $@)"
	sed -e 's/@VERSION@/$(VERSION)/g' -e 's/@MIN_MACOS@/$(MIN_MACOS)/g' "$<" > "$@"
	plutil -lint "$@" >/dev/null

$(BUNDLE)/Contents/PkgInfo:
	@mkdir -p "$(dir $@)"
	printf 'eFKTFXTC' > "$@"

$(HELP): $(PLUGIN_DIR)/resources/help/index.html
	@mkdir -p "$(dir $@)"
	cp "$<" "$@"

# Ad-hoc signature, last: it seals the binary, Info.plist and resources.
$(BUNDLE)/Contents/_CodeSignature/CodeResources: $(BIN) $(RSRC) $(HELP) $(BUNDLE)/Contents/Info.plist $(BUNDLE)/Contents/PkgInfo
	codesign --force --sign - "$(BUNDLE)"
	@echo "built $(BUNDLE) (core: $(OD_CORE))"

$(HARNESS): $(PLUGIN_DIR)/tests/harness.mm $(PLUGIN_DIR)/tests/capture_tests.h $(PLUGIN_DIR)/tests/panel_tests.h $(PLUGIN_DIR)/tests/analysis_panel_tests.h $(PLUGIN_DIR)/tests/reserved_param_tests.h $(PLUGIN_DIR)/tests/picture_pin.h $(PLUGIN_DIR)/tests/picture_pin_ref.h $(PLUGIN_DIR)/src/OpenDenoisePlugin.h $(PLUGIN_DIR)/src/RenderStats.h $(PLUGIN_DIR)/src/OpenDenoiseCapture.h \
  $(ROOT)/core/include/opendenoise/opendenoise.h
	@mkdir -p "$(dir $@)"
	$(CXX) -arch arm64 -std=c++17 -O1 -g -fobjc-arc -mmacosx-version-min=$(MIN_MACOS) -isysroot "$(SDKROOT_PATH)" \
	  -DOD_HIDE_UNUSED_TOPICS=$(OD_HIDE_UNUSED_TOPICS) -Wall -Wno-four-char-constants -Wno-multichar -Wno-pragma-pack -Wno-missing-field-initializers -Wno-unused-function $(INCLUDES) \
	  -framework Metal -framework Foundation "$<" -o "$@"

# The harness has its own 60 s alarm; its largest frames are 640x360.
# Real core: --real (the harness checks that noise is removed, that the
# neighbours count, colour levels and alpha). Then the same harness against
# the stub core, built in <BUILD>/stub so that it never replaces the real
# bundle: the pixel conversions exactly, the parameters, offsets and keys the
# plug-in passes, the session registry, and pass-through when every core call
# fails. Then the install script's host check and a build from a path with
# spaces (both fast; see the scripts).
STUB_BUILD := $(BUILD_REL)/stub
TEST_DEADLINE ?= 300
DEADLINE = perl -e 'alarm shift; exec @ARGV or die "exec: $$!"' $(TEST_DEADLINE)
ifeq ($(OD_CORE),lib)
$(PARAM_TEST): $(PLUGIN_DIR)/tests/parameter_tests.mm $(PLUGIN_DIR)/src/OpenDenoiseShared.mm $(PLUGIN_DIR)/src/OpenDenoisePlugin.h $(LIB_PREREQ)
	$(CXX) $(CXXFLAGS) "$<" $(CORE_LINK) -framework Metal -framework Foundation -o "$@"

$(BRIDGE_TEST): $(PLUGIN_DIR)/tests/analysis_bridge_tests.mm $(PLUGIN_DIR)/src/OpenDenoiseAnalysisBridge.mm $(PLUGIN_DIR)/src/OpenDenoiseAnalysisBridge.h $(PLUGIN_DIR)/src/OpenDenoiseShared.mm $(OBJ)/OpenDenoiseCapture.o $(LIB_PREREQ)
	$(CXX) $(CXXFLAGS) "$<" $(OBJ)/OpenDenoiseCapture.o $(CORE_LINK) -framework Metal -framework Foundation -o "$@"

plugin-test: plugin $(HARNESS) $(PARAM_TEST) $(BRIDGE_TEST)
	"$(BRIDGE_TEST)"
	"$(PARAM_TEST)"
	"$(HARNESS)" "$(BIN)" --real
	@$(MAKE) --no-print-directory BUILD="$(STUB_BUILD)" OD_ROOT= OD_CORE=stub \
	  PREMIERE_SDK="$(PREMIERE_SDK)" AE_SDK="$(AE_SDK)" plugin-test
	$(DEADLINE) "$(PLUGIN_DIR)/tests/install_test.sh" "$(ROOT)/scripts/install.sh" "$(BUNDLE)"
	$(DEADLINE) "$(PLUGIN_DIR)/tests/space_path_test.sh" "$(ROOT)" "$(BUILD_DIR)/space-path-test" \
	  "$(PREMIERE_SDK)" "$(AE_SDK)"
else
plugin-test: plugin $(HARNESS)
	"$(HARNESS)" "$(BIN)"
	OD_STUB_FAIL=1 "$(HARNESS)" "$(BIN)" --expect-passthrough
endif

plugin-verify: plugin
	"$(PLUGIN_DIR)/tests/verify_bundle.sh" "$(BUNDLE)"

plugin-clean:
	rm -rf "$(OBJ)" "$(BUNDLE)" "$(HARNESS)" "$(HARNESS).dSYM" "$(HARNESS).record" "$(HARNESS).fail.record" "$(HARNESS).unload.record" "$(HARNESS).record.log" "$(HARNESS).fail.record.log" \
	  "$(BUILD_DIR)/stub" "$(BUILD_DIR)/space-path-test"

FORCE:
