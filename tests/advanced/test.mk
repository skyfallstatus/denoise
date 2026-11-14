# SPDX-License-Identifier: MIT
# Advanced controls (API v6) through od_render: the defaults, advanced = 0 and
# NaN are one render, pinned bit for bit as output hashes of the core before
# API v6; each control moves the picture the way opendenoise.h says, in every
# stage that reads it; the bands step, Hold profile, and out-of-range input.
# Included by the top-level Makefile.
#
#   make test-advanced    synthetic frames, no media needed; part of `make test`

ADVANCED_TEST := $(BUILD)/tests/test_advanced

$(ADVANCED_TEST): $(OBJ)/tests/advanced/test_advanced.mm.o $(BUILD)/libopendenoise.a
	@mkdir -p $(dir $@)
	$(CXX) $< $(BUILD)/libopendenoise.a $(OD_LDFLAGS) -o $@

.PHONY: test-advanced
test-advanced: $(ADVANCED_TEST)
	@echo "== $(ADVANCED_TEST) (legacy)"; $(DEADLINE) env OD_GRAIN_PROFILE_V2=0 $(ADVANCED_TEST)
	@echo "== $(ADVANCED_TEST) (finest grain)"; $(DEADLINE) env OD_GRAIN_PROFILE_V2=1 $(ADVANCED_TEST)

test: test-advanced

-include $(OBJ)/tests/advanced/test_advanced.mm.d
