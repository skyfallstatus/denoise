#!/bin/zsh
# SPDX-License-Identifier: MIT
# The plug-in must build from a repository whose path contains spaces (make
# splits target names at spaces). Copies plugin/ and core/include/ to
# "<work>/space path/open denoise/" and runs plugin/Makefile there:
#   - dry runs of `plugin-test plugin-verify`, invoked as `make -C plugin
#     BUILD=<relative>` and as the top-level Makefile invokes it (absolute
#     BUILD and OD_ROOT), against a placeholder core library;
#   - a real build of the bundle with the stub core, and its signature check.
# Usage: space_path_test.sh <repo root> <work dir> <PREMIERE_SDK> <AE_SDK>
set -u
root="${1:?usage: space_path_test.sh <repo root> <work dir> <PREMIERE_SDK> <AE_SDK>}"
work="${2:?}"; psdk="${3:?}"; aesdk="${4:?}"
root="${root:A}"; work="${work:A}"
copy="$work/space path/open denoise"
rm -rf "$work"
mkdir -p "$copy/core/src" || exit 1
cp -R "$root/plugin" "$copy/plugin" && cp -R "$root/core/include" "$copy/core/include" || { print "FAIL  copy"; exit 1; }
cp "$root/core/src/profile_blob.cpp" "$root/core/src/analysis.cpp" "$root/core/src/analysis_layout.h" "$copy/core/src/" || exit 1
for b in rel top; do mkdir -p "$copy/build/$b" && : > "$copy/build/$b/libopendenoise.a"; done

fail=0
run() {  # label, command...
  local label="$1"; shift
  local log="$work/${label//[^A-Za-z0-9]/_}.log"
  "$@" >"$log" 2>&1
  local rc=$?
  if (( rc == 0 )) && ! grep -qE 'No rule to make target|overriding (commands|recipe)|\*\*\*' "$log"; then
    print -r -- "ok    space path: $label"
  else
    print -r -- "FAIL  space path: $label (exit $rc, log $log)"
    tail -n 5 "$log"
    fail=1
  fi
}

run "make -C plugin, relative BUILD (dry run)" \
  make -n -C "$copy/plugin" BUILD=build/rel PREMIERE_SDK="$psdk" AE_SDK="$aesdk" plugin-test plugin-verify
run "as the top-level Makefile calls it (dry run)" \
  make -n -C "$copy/plugin" OD_ROOT="$copy" BUILD="$copy/build/top" OD_CORE=lib \
  PREMIERE_SDK="$psdk" AE_SDK="$aesdk" plugin-test plugin-verify
run "stub-core bundle builds" \
  make -C "$copy/plugin" OD_ROOT="$copy" BUILD="$copy/build/stub" OD_CORE=stub \
  PREMIERE_SDK="$psdk" AE_SDK="$aesdk" plugin
run "stub-core bundle signature verifies" codesign --verify --strict "$copy/build/stub/Denoise.plugin"

if (( fail )); then print "space path checks FAILED"; exit 1; fi
rm -rf "$work"
print "space path checks passed"
