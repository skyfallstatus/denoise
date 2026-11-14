# SPDX-License-Identifier: MIT
# Integration tests: the real stages together through od_render. Included by
# the top-level Makefile.
#
#   make test-pipeline    synthetic frames, no media needed; part of `make test`

PIPELINE_TEST := $(BUILD)/tests/test_pipeline

$(PIPELINE_TEST): $(OBJ)/tests/pipeline/test_pipeline.mm.o $(BUILD)/libopendenoise.a
	@mkdir -p $(dir $@)
	$(CXX) $< $(BUILD)/libopendenoise.a $(OD_LDFLAGS) -o $@

.PHONY: test-pipeline
test-pipeline: $(PIPELINE_TEST)
	@echo "== $(PIPELINE_TEST)"; $(DEADLINE) $(PIPELINE_TEST)

test: test-pipeline

-include $(OBJ)/tests/pipeline/test_pipeline.mm.d
