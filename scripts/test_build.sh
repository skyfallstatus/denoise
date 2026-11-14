#!/bin/bash
# SPDX-License-Identifier: MIT
# Checks of the top-level Makefile, run by `make test` (target test-build).
#
# Dry runs (make -n) against a fake SDK folder that holds only the two header
# files the Makefile probes for, so nothing is compiled, linked or signed and
# the real SDKs are not needed:
#   - every combination of plugin goals starts exactly one plug-in sub-make
#     (two at once link, run Rez and sign the same bundle and fail under -j);
#   - `make sdk-check` with no SDK paths fails and says how to set them.
# Usage: scripts/test_build.sh [path/to/Makefile]   (default: the repo's)
set -u
root="$(cd "$(dirname "$0")/.." && pwd)"
mk="${1:-$root/Makefile}"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/od-test-build.XXXXXX")" || exit 2
trap 'rm -rf "$tmp"' EXIT
tmp="$(cd "$tmp" && pwd -P)" || exit 2   # no // or symlinks: make prints normalised paths
mkdir -p "$tmp/pr/Examples/Headers" "$tmp/ae/Examples/Headers"
: > "$tmp/pr/Examples/Headers/PrSDKGPUFilter.h"
: > "$tmp/ae/Examples/Headers/AE_Effect.h"

# This target tree is only printed by make -n, never created. Keep it relative
# and space-free: GNU make cannot parse target paths below a TMPDIR with spaces.
# The fake SDKs still live under TMPDIR and are quoted as recipe arguments.
dry_build="build/test-build-dry-$$"
checks=0
fails=0
check() {   # check <description> <condition-exit-status>
  checks=$((checks + 1))
  if [ "$2" -eq 0 ]; then echo "ok   $1"; else echo "FAIL $1"; fails=$((fails + 1)); fi
}

submakes() {   # plug-in sub-makes, and signing steps of the real bundle, a dry run would start
  local out
  out=$(cd "$root" && make -n -j8 -f "$mk" BUILD="$dry_build" PREMIERE_SDK="$tmp/pr" AE_SDK="$tmp/ae" "$@" 2>&1)
  local n s
  n=$(printf '%s\n' "$out" | /usr/bin/grep -c -e ' -C plugin ')
  # Only <BUILD>/Denoise.plugin counts: the harness may sign other bundles
  # (a stub-core copy in a subfolder) in sub-makes of its own.
  s=$(printf '%s\n' "$out" | /usr/bin/grep -e 'codesign --force' | /usr/bin/grep -c -F -e "$dry_build/Denoise.plugin\"")
  echo "$n $s"
}

for goals in "plugin" "plugin-test" "plugin plugin-test" "plugin-test plugin" "test plugin" "plugin test"; do
  # shellcheck disable=SC2086
  set -- $(submakes $goals)
  check "make -j4 $goals: one plug-in sub-make (got $1) and one codesign (got $2)" \
    $([ "$1" -eq 1 ] && [ "$2" -eq 1 ]; echo $?)
done

out=$(cd "$root" && make -s -f "$mk" sdk-check PREMIERE_SDK= AE_SDK= 2>&1); rc=$?
check "sdk-check without SDK paths fails (rc $rc)" $([ $rc -ne 0 ]; echo $?)
case "$out" in *PREMIERE_SDK*local.mk*) r=0 ;; *) r=1 ;; esac
check "sdk-check says how to set PREMIERE_SDK: $(printf '%s' "$out" | head -1 | cut -c1-90)..." $r
out=$(cd "$root" && make -s -f "$mk" sdk-check PREMIERE_SDK="$tmp/pr" AE_SDK=/nonexistent 2>&1); rc=$?
case "$out" in *AE_SDK*local.mk*) r=0 ;; *) r=1 ;; esac
check "sdk-check names a missing AE_SDK (rc $rc)" $([ $rc -ne 0 ] && [ $r -eq 0 ]; echo $?)
out=$(cd "$root" && make -s -f "$mk" sdk-check PREMIERE_SDK="$tmp/pr" AE_SDK="$tmp/ae" 2>&1); rc=$?
check "sdk-check passes with both folders (rc $rc)" $rc

echo "test_build: $checks checks, $fails failures"
[ $fails -eq 0 ]
