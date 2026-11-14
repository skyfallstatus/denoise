# SPDX-License-Identifier: MIT
# Temporal stage tests. Included by the top-level Makefile.
#
#   make test-temporal     synthetic correctness tests (part of `make test`)

TEMPORAL_TEST := $(BUILD)/tests/temporal/test_temporal

$(TEMPORAL_TEST): $(OBJ)/tests/temporal/test_temporal.mm.o $(BUILD)/libopendenoise.a
	@mkdir -p $(dir $@)
	$(CXX) $< $(BUILD)/libopendenoise.a $(OD_LDFLAGS) -o $@

.PHONY: test-temporal
test-temporal: $(TEMPORAL_TEST)
	@echo "== $(TEMPORAL_TEST) (legacy)"; $(DEADLINE) env OD_GRAIN_PROFILE_V2=0 $(TEMPORAL_TEST)
	@echo "== $(TEMPORAL_TEST) (finest grain)"; $(DEADLINE) env OD_GRAIN_PROFILE_V2=1 $(TEMPORAL_TEST)

test: test-temporal

-include $(OBJ)/tests/temporal/test_temporal.mm.d
