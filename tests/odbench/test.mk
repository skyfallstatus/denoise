# SPDX-License-Identifier: MIT
# odbench decoding and asynchronous rendering tests.
#
#   make test-odbench    rotated clips it makes itself (no test media; skips
#                        without ffmpeg), the decode matrix of untagged HD and
#                        SD clips and tagged ones, and --async under a held queue
#                        (no host buffer overwritten while the GPU uses it);
#                        part of `make test`

.PHONY: test-odbench
test-odbench: $(BUILD)/odbench
	@echo "== tests/odbench/test_video_size.py"
	@$(DEADLINE) $(PYTHON) tests/odbench/test_video_size.py "$(BUILD)/odbench"
	@echo "== tests/odbench/test_video_matrix.py"
	@$(DEADLINE) $(PYTHON) tests/odbench/test_video_matrix.py "$(BUILD)/odbench"
	@echo "== tests/odbench/test_async.py"
	@$(DEADLINE) $(PYTHON) tests/odbench/test_async.py "$(BUILD)/odbench"

test: test-odbench
