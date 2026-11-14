# SPDX-License-Identifier: MIT
# Noise profiler tests. Included by the top-level Makefile.
#
#   make test-profile    synthetic ground-truth tests (no media needed), part of `make test`

$(BUILD)/tests/test_profile: $(OBJ)/tests/profile/test_profile.mm.o $(BUILD)/libopendenoise.a
	@mkdir -p $(dir $@)
	$(CXX) $< $(BUILD)/libopendenoise.a $(OD_LDFLAGS) -o $@

.PHONY: test-profile
test-profile: $(BUILD)/tests/test_profile
	@echo "== $<"; $(DEADLINE) $<

test: test-profile

-include $(OBJ)/tests/profile/test_profile.mm.d

$(BUILD)/tests/test_deterministic: $(OBJ)/tests/profile/test_deterministic.mm.o $(BUILD)/libopendenoise.a
	@mkdir -p $(dir $@)
	$(CXX) $< $(BUILD)/libopendenoise.a $(OD_LDFLAGS) -o $@
.PHONY: test-deterministic
test-deterministic: $(BUILD)/tests/test_deterministic
	@$(DEADLINE) $<
test: test-deterministic
-include $(OBJ)/tests/profile/test_deterministic.mm.d

$(BUILD)/tests/test_coarse: $(OBJ)/tests/profile/test_coarse.mm.o $(BUILD)/libopendenoise.a
	@mkdir -p $(dir $@)
	$(CXX) $< $(BUILD)/libopendenoise.a $(OD_LDFLAGS) -o $@
.PHONY: test-coarse
test-coarse: $(BUILD)/tests/test_coarse
	@echo "== $<"; $(DEADLINE) $<
test: test-coarse
-include $(OBJ)/tests/profile/test_coarse.mm.d

$(BUILD)/tests/test_coarse_stats: $(OBJ)/tests/profile/test_coarse_stats.mm.o $(BUILD)/libopendenoise.a
	@mkdir -p $(dir $@)
	$(CXX) $< $(BUILD)/libopendenoise.a $(OD_LDFLAGS) -o $@
.PHONY: test-coarse-stats
test-coarse-stats: $(BUILD)/tests/test_coarse_stats
	@echo "== $<"; $(DEADLINE) $<
test: test-coarse-stats
-include $(OBJ)/tests/profile/test_coarse_stats.mm.d

$(BUILD)/tests/test_coarse_time: $(OBJ)/tests/profile/test_coarse_time.mm.o $(BUILD)/libopendenoise.a
	@mkdir -p $(dir $@)
	$(CXX) $< $(BUILD)/libopendenoise.a $(OD_LDFLAGS) -o $@
.PHONY: test-coarse-time
test-coarse-time: $(BUILD)/tests/test_coarse_time
	@echo "== $<"; $(DEADLINE) $<
test: test-coarse-time
-include $(OBJ)/tests/profile/test_coarse_time.mm.d

$(BUILD)/tests/test_gray: $(OBJ)/tests/profile/test_gray.mm.o $(BUILD)/libopendenoise.a
	@mkdir -p $(dir $@)
	$(CXX) $< $(BUILD)/libopendenoise.a $(OD_LDFLAGS) -o $@
.PHONY: test-gray
test-gray: $(BUILD)/tests/test_gray
	@echo "== $<"; $(DEADLINE) $<
test: test-gray
-include $(OBJ)/tests/profile/test_gray.mm.d

$(BUILD)/tests/test_knots: $(OBJ)/tests/profile/test_knots.mm.o $(BUILD)/libopendenoise.a
	@mkdir -p $(dir $@)
	$(CXX) $< $(BUILD)/libopendenoise.a $(OD_LDFLAGS) -o $@
.PHONY: test-knots
test-knots: $(BUILD)/tests/test_knots
	@$(DEADLINE) $<
test: test-knots
-include $(OBJ)/tests/profile/test_knots.mm.d

$(BUILD)/tests/test_chroma_recovery: $(OBJ)/tests/profile/test_chroma_recovery.mm.o $(BUILD)/libopendenoise.a
	@mkdir -p $(dir $@)
	$(CXX) $< $(BUILD)/libopendenoise.a $(OD_LDFLAGS) -o $@
.PHONY: test-chroma-recovery
test-chroma-recovery: $(BUILD)/tests/test_chroma_recovery
	@$(DEADLINE) $<
test: test-chroma-recovery
-include $(OBJ)/tests/profile/test_chroma_recovery.mm.d
