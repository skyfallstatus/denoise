#!/bin/zsh
# SPDX-License-Identifier: MIT
# Checks scripts/install.sh without installing anything in the system folder
# or invoking real sudo: --help prints the header and no code,
# and the running-host check sees Premiere Pro, Premiere Pro (Beta), Media
# Encoder, After Effects (also its nested aerendercore) and the After Effects
# render engine, but not an unrelated Adobe application. A host is simulated
# by an idle process whose command line carries such a path (pgrep -f matches
# whole command lines), and recognised by its pid in the script's output, so
# a real host running at the same time does not confuse the test. These host
# checks use --dry-run. Replacement tests use a copy of the script with a
# temporary destination and a guarded mock sudo; the production destination
# has no environment override.
# Usage: install_test.sh <install.sh> <Denoise.plugin>
set -u
inst="${1:?usage: install_test.sh <install.sh> <Denoise.plugin>}"
bundle="${2:?usage: install_test.sh <install.sh> <Denoise.plugin>}"
bundle="${bundle:A}"
fail=0
ok()  { print -r -- "ok    install.sh: $1"; }
bad() { print -r -- "FAIL  install.sh: $1"; fail=1; }

pids=()
work=""
cleanup() {
  (( ${#pids} )) && kill $pids 2>/dev/null
  [[ -n "$work" ]] && rm -rf "$work"
  return 0
}
trap cleanup EXIT
trap 'exit 130' INT TERM

out="$("$inst" --help 2>&1)"; rc=$?
if [[ $rc == 0 && "$out" == *"--dry-run"* && "$out" != *"set -eu"* && "$out" != *setopt* && "$out" != *SPDX* ]]; then
  ok "--help prints the header only"
else
  bad "--help (exit $rc): $out"
fi

# Starts an idle process with `exe` in its command line; its pid goes to
# $fake_pid. (Not in a command substitution: that would wait for the process.)
fake_pid=0
fake_host() {
  perl -e 'sleep 20' -- "$1" </dev/null >/dev/null 2>&1 &
  fake_pid=$!
  pids+=($fake_pid)
  local i
  for i in {1..100}; do  # until exec has replaced the forked shell
    [[ "$(ps -o args= -p $fake_pid 2>/dev/null)" == *"$1"* ]] && break
    perl -e 'select(undef, undef, undef, 0.02)'
  done
}

check() {  # label, executable path, 1 = must be detected / 0 = must not be
  local label="$1" exe="$2" expect="$3"
  fake_host "$exe"
  local pid=$fake_pid
  local out rc
  out="$("$inst" --dry-run "$bundle" 2>&1)"; rc=$?
  if (( expect )); then
    if [[ $rc == 1 && "$out" == *"pid $pid "* && "$out" == *"pass --force"* ]]; then ok "detects $label"
    else bad "detects $label (exit $rc): $out"; fi
    out="$("$inst" --dry-run --force "$bundle" 2>&1)"; rc=$?
    if [[ $rc == 0 && "$out" == *"pid $pid "* && "$out" == *"dry run: would"* ]]; then ok "--force goes past $label"
    else bad "--force with $label (exit $rc): $out"; fi
  else
    if [[ "$out" != *"pid $pid "* ]]; then ok "ignores $label"
    else bad "ignores $label: $out"; fi
  fi
  kill $pid 2>/dev/null
  wait $pid 2>/dev/null
  pids=(${pids:#$pid})
  return 0
}

A=/Applications/odtest
check "Premiere Pro 2026" "$A/Adobe Premiere Pro 2026/Adobe Premiere Pro 2026.app/Contents/MacOS/Adobe Premiere Pro 2026" 1
check "Premiere Pro (Beta)" "$A/Adobe Premiere Pro (Beta)/Adobe Premiere Pro (Beta).app/Contents/MacOS/Adobe Premiere Pro (Beta)" 1
check "Media Encoder 2026" "$A/Adobe Media Encoder 2026/Adobe Media Encoder 2026.app/Contents/MacOS/Adobe Media Encoder 2026" 1
check "After Effects 2026" "$A/Adobe After Effects 2026/Adobe After Effects 2026.app/Contents/MacOS/After Effects" 1
check "aerendercore" "$A/Adobe After Effects 2026/Adobe After Effects 2026.app/Contents/aerendercore.app/Contents/MacOS/aerendercore" 1
check "After Effects Render Engine" "$A/Adobe After Effects Render Engine 2026/Adobe After Effects Render Engine 2026.app/Contents/MacOS/After Effects Render Engine" 1
check "Bridge (not a MediaCore host)" "$A/Adobe Bridge 2026/Adobe Bridge 2026.app/Contents/MacOS/Adobe Bridge 2026" 0
check "a document named like Premiere" "$A/Adobe Premiere Pro 2026 notes.txt" 0

# Exercise the real copy/verify/remove/swap sequence, with every mutation
# confined to a temporary folder beside the build. No root access is used.
work="$(mktemp -d "${bundle:h}/install-test.XXXXXX")" || exit 2
export OD_INSTALL_TEST_DEST="$work/MediaCore"
export OD_INSTALL_TEST_SOURCE="$bundle"
export OD_INSTALL_TEST_LOG="$work/commands.log"
mkdir -p "$work/bin" || exit 2
mock_inst="$work/install.sh"
sed 's|^DEST_DIR=.*|DEST_DIR="$OD_INSTALL_TEST_DEST"|' "$inst" > "$mock_inst" || exit 2
grep -qx 'DEST_DIR="$OD_INSTALL_TEST_DEST"' "$mock_inst" || exit 2
cat > "$work/bin/sudo" <<'MOCK'
#!/bin/zsh
# SPDX-License-Identifier: MIT
set -eu
# Never forward to real sudo. Reject commands or paths outside this fixture.
[[ "$1" == -n ]] || exit 90
shift
op="$1"; shift
[[ "$op" == true && $# == 0 ]] && exit 0
case "$op" in mkdir|rm|ditto|xattr|codesign|mv) ;; *) exit 91 ;; esac
for arg in "$@"; do
  case "$arg" in
    -p|-rf|-dr|--verify|--strict|com.apple.quarantine) ;;
    "$OD_INSTALL_TEST_SOURCE"|"$OD_INSTALL_TEST_DEST"|"$OD_INSTALL_TEST_DEST/"*) ;;
    *) print -u2 -- "mock refused path: $arg"; exit 92 ;;
  esac
done
print -r -- "$op $*" >> "$OD_INSTALL_TEST_LOG"
case "$op" in
  mkdir) /bin/mkdir "$@" ;;
  rm)
    if [[ "${OD_INSTALL_TEST_FAIL_REMOVE:-0}" == 1 && "$2" == "$OD_INSTALL_TEST_DEST/OpenDenoise.plugin" ]]; then exit 43; fi
    /bin/rm "$@" ;;
  ditto) /usr/bin/ditto "$@" ;;
  xattr) exit 0 ;;
  codesign)
    [[ "${OD_INSTALL_TEST_FAIL_VERIFY:-0}" == 1 ]] && exit 42
    /usr/bin/codesign "$@" ;;
  mv)
    # The legacy registration must be gone before the new bundle is published.
    [[ ! -e "$OD_INSTALL_TEST_DEST/OpenDenoise.plugin" && ! -L "$OD_INSTALL_TEST_DEST/OpenDenoise.plugin" ]] || exit 93
    /bin/mv "$@" ;;
esac
MOCK
cat > "$work/bin/pgrep" <<'MOCK'
#!/bin/zsh
# SPDX-License-Identifier: MIT
if [[ "${OD_INSTALL_TEST_HOST:-0}" == 1 ]]; then print 999999; exit 0; fi
exit 1
MOCK
chmod +x "$mock_inst" "$work/bin/sudo" "$work/bin/pgrep" || exit 2

legacy="$OD_INSTALL_TEST_DEST/OpenDenoise.plugin"
dest="$OD_INSTALL_TEST_DEST/Denoise.plugin"
reset_dest() {
  rm -rf "$OD_INSTALL_TEST_DEST"
  mkdir -p "$OD_INSTALL_TEST_DEST/Unrelated.plugin"
  print keep > "$OD_INSTALL_TEST_DEST/Unrelated.plugin/marker"
  : > "$OD_INSTALL_TEST_LOG"
}
run_mock() {
  out="$(PATH="$work/bin:$PATH" "$mock_inst" "$@" "$bundle" </dev/null 2>&1)"; rc=$?
}
for mode in clean legacy current both legacy-symlink; do
  reset_dest
  case "$mode" in legacy|both) mkdir -p "$legacy"; print old > "$legacy/marker" ;; esac
  case "$mode" in current|both) mkdir -p "$dest"; print old > "$dest/marker" ;; esac
  [[ "$mode" == legacy-symlink ]] && ln -s "$work/missing" "$legacy"
  run_mock
  if [[ $rc == 0 && ! -e "$legacy" && ! -L "$legacy" && ! -e "$dest/marker" ]] &&
     cmp -s "$bundle/Contents/MacOS/Denoise" "$dest/Contents/MacOS/Denoise" &&
     codesign --verify --strict "$dest" 2>/dev/null &&
     [[ "$(cat "$OD_INSTALL_TEST_DEST/Unrelated.plugin/marker")" == keep ]]; then
    ok "$mode install leaves only Denoise and preserves unrelated bundles"
  else bad "$mode install (exit $rc): $out"; fi
  case "$mode" in
    legacy|both|legacy-symlink)
      if [[ "$out" == *"removed legacy $legacy (same effect match name)"* ]]; then ok "$mode reports legacy removal"
      else bad "$mode omitted removal report: $out"; fi ;;
    *)
      if [[ "$out" != *"removed legacy"* ]]; then ok "$mode reports no spurious legacy removal"
      else bad "$mode claims legacy removal: $out"; fi ;;
  esac
done

# Release layout: install.sh and Denoise.plugin side by side, no argument.
mkdir -p "$work/release" || exit 2
cp "$inst" "$work/release/install.sh" && ditto "$bundle" "$work/release/Denoise.plugin" || exit 2
out="$(PATH="$work/bin:$PATH" OD_INSTALL_TEST_HOST=0 "$work/release/install.sh" --dry-run </dev/null 2>&1)"; rc=$?
if [[ $rc == 0 && "$out" == *"source: ${work:A}/release/Denoise.plugin "* ]]; then ok "finds the bundle next to the script"
else bad "release layout (exit $rc): $out"; fi

for mode in dry-run host-refusal failed-verification failed-removal; do
  reset_dest
  mkdir -p "$legacy" "$dest"
  print old > "$legacy/marker"; print current > "$dest/marker"
  export OD_INSTALL_TEST_HOST=0 OD_INSTALL_TEST_FAIL_VERIFY=0 OD_INSTALL_TEST_FAIL_REMOVE=0
  case "$mode" in
    dry-run) run_mock --dry-run ;;
    host-refusal) OD_INSTALL_TEST_HOST=1; run_mock ;;
    failed-verification) OD_INSTALL_TEST_FAIL_VERIFY=1; run_mock ;;
    failed-removal) OD_INSTALL_TEST_FAIL_REMOVE=1; run_mock ;;
  esac
  if [[ -f "$legacy/marker" && -f "$dest/marker" && ! -e "$dest/Contents" ]] &&
     [[ "$(cat "$legacy/marker")" == old && "$(cat "$dest/marker")" == current ]] &&
     ! grep -q '^mv ' "$OD_INSTALL_TEST_LOG"; then ok "$mode preserves both existing bundles"
  else bad "$mode changed a bundle: $out"; fi
  if [[ "$mode" == dry-run ]]; then
    if [[ $rc == 0 && ! -s "$OD_INSTALL_TEST_LOG" && "$out" == *"would remove legacy $legacy"* ]]; then
      ok "dry run reports legacy cleanup without privilege commands"
    else bad "dry run (exit $rc): $out"; fi
  elif [[ "$mode" == host-refusal && $rc == 1 && ! -s "$OD_INSTALL_TEST_LOG" && "$out" == *"pass --force"* ]]; then
    ok "host refusal stops before privilege commands"
  elif [[ "$mode" == failed-verification && $rc == 1 && "$out" == *"copied bundle does not verify; nothing was installed"* ]]; then
    ok "staging verification failure stops installation"
  elif [[ "$mode" == failed-removal && $rc == 43 && "$out" != *"installed $dest"* ]]; then
    ok "legacy removal failure stops installation"
  else bad "$mode unexpected result (exit $rc): $out"; fi
done

if (( fail )); then print "install.sh checks FAILED"; exit 1; fi
print "install.sh checks passed"
