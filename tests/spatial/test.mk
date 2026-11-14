# SPDX-License-Identifier: MIT
# Spatial / sharpen stage tests. Included by the top-level Makefile
# (-include tests/*/test.mk); uses BUILD, OBJ, CXX, OD_CXXFLAGS,
# OD_LDFLAGS and DEADLINE from there.
#
#   make test-spatial   correctness tests (synthetic frames, no media needed)
#   make test-pinned    the picture of spatial, sharpen and profile against
#                       statistics recorded before they were optimised

SPATIAL_TEST := $(BUILD)/tests/test_spatial
PINNED_TEST := $(BUILD)/tests/test_pinned

$(SPATIAL_TEST): $(OBJ)/tests/spatial/test_spatial.mm.o $(BUILD)/libopendenoise.a
	@mkdir -p $(dir $@)
	$(CXX) $< $(BUILD)/libopendenoise.a $(OD_LDFLAGS) -o $@

$(PINNED_TEST): $(OBJ)/tests/spatial/test_pinned.mm.o $(BUILD)/libopendenoise.a
	@mkdir -p $(dir $@)
	$(CXX) $< $(BUILD)/libopendenoise.a $(OD_LDFLAGS) -o $@

.PHONY: test-spatial test-pinned
test-spatial: $(SPATIAL_TEST)
	@echo "== $(SPATIAL_TEST) (legacy)"; $(DEADLINE) env OD_GRAIN_PROFILE_V2=0 $(SPATIAL_TEST)
	@echo "== $(SPATIAL_TEST) (finest grain)"; $(DEADLINE) env OD_GRAIN_PROFILE_V2=1 $(SPATIAL_TEST)
test-pinned: $(PINNED_TEST)
	@echo "== $(PINNED_TEST) (legacy)"; $(DEADLINE) env OD_GRAIN_PROFILE_V2=0 $(PINNED_TEST)
	@echo "== $(PINNED_TEST) (finest grain)"; $(DEADLINE) env OD_GRAIN_PROFILE_V2=1 $(PINNED_TEST)

test: test-spatial test-pinned

-include $(OBJ)/tests/spatial/test_spatial.mm.d $(OBJ)/tests/spatial/test_pinned.mm.d

# Pin the ordered reduction against the original scalar calibration kernel.
RESIDUAL_EXACT_TEST := $(BUILD)/tests/test_residual_exact
$(RESIDUAL_EXACT_TEST): $(OBJ)/tests/spatial/test_residual_exact.mm.o $(BUILD)/libopendenoise.a
	@mkdir -p $(dir $@)
	$(CXX) $< $(BUILD)/libopendenoise.a $(OD_LDFLAGS) -o $@
.PHONY: test-residual-exact
test-residual-exact: $(RESIDUAL_EXACT_TEST)
	$(DEADLINE) $(RESIDUAL_EXACT_TEST)
test: test-residual-exact
-include $(OBJ)/tests/spatial/test_residual_exact.mm.d
