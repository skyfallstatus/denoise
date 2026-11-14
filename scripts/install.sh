#!/bin/zsh
# SPDX-License-Identifier: MIT
# Install Denoise.plugin into Adobe's shared MediaCore plug-ins folder.
#
#   print -r -- "$PASSWORD" | scripts/install.sh [--force] [--dry-run] [path/to/Denoise.plugin]
#
# - The destination needs root. The sudo password is read from stdin (one
#   line) and handed to `sudo -S`; it is never printed or put on a command
#   line. If sudo already has a valid ticket, stdin may be empty.
# - Refuses to run while an Adobe host that loads MediaCore plug-ins is open
#   (Premiere Pro, Premiere Pro (Beta), Media Encoder, After Effects and its
#   render engines: they scan plug-ins at start-up and may have the old binary
#   mapped) unless --force is given. The hosts found are listed.
# - Replaces <MediaCore>/Denoise.plugin and removes the former
#   <MediaCore>/OpenDenoise.plugin, which registers the same effect identity.
#   Prints every step it takes.
# - Default bundle: Denoise.plugin next to this script (the release download),
#   otherwise build/Denoise.plugin in the repo this script lives in.

set -eu
setopt pipefail

DEST_DIR="/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore"
NAME="Denoise.plugin"
LEGACY_NAME="OpenDenoise.plugin"
ROOT="${0:A:h:h}"

force=0
dry=0
bundle=""
for arg in "$@"; do
  case "$arg" in
    --force) force=1 ;;
    --dry-run) dry=1 ;;
    -h|--help) awk 'NR == 1 || /SPDX-License-Identifier/ { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; exit 0 ;;
    -*) print -u2 -r -- "install.sh: unknown option $arg"; exit 2 ;;
    *) bundle="$arg" ;;
  esac
done
if [[ -z "$bundle" ]]; then
  if [[ -d "${0:A:h}/$NAME" ]]; then bundle="${0:A:h}/$NAME"; else bundle="$ROOT/build/$NAME"; fi
fi
bundle="${bundle:A}"

say() { print -r -- "install.sh: $*"; }
die() { print -u2 -r -- "install.sh: $*"; exit 1; }

# ---- what are we installing? -------------------------------------------------
[[ -d "$bundle" && "${bundle:t}" == "$NAME" ]] || die "not a $NAME bundle: $bundle (build it with: make plugin)"
[[ -f "$bundle/Contents/MacOS/Denoise" ]] || die "bundle has no Contents/MacOS/Denoise"
[[ -s "$bundle/Contents/Resources/Denoise.rsrc" ]] || die "bundle has no Contents/Resources/Denoise.rsrc (PiPL)"
codesign --verify --strict "$bundle" 2>/dev/null || die "bundle signature does not verify; rebuild it"
version="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$bundle/Contents/Info.plist" 2>/dev/null || print '?')"
say "source: $bundle (version $version, $(du -sk "$bundle" | awk '{print $1}') KB)"

# ---- is an Adobe host running? -----------------------------------------------
# Executables inside "Adobe Premiere Pro 2026.app", "Adobe Premiere Pro
# (Beta).app", "Adobe Media Encoder 2026.app", "Adobe After Effects 2026.app"
# (also its nested aerendercore.app) and "Adobe After Effects Render Engine
# 2026.app". pgrep -f matches whole command lines, so a process that merely has
# such a path among its arguments also counts; --force is the way past that.
HOST_RE='Adobe (Premiere Pro|Media Encoder|After Effects)[^/]*\.app/Contents/([^/]+\.app/Contents/)?MacOS/'
host_pids=("${(@f)$(pgrep -f -- "$HOST_RE" 2>/dev/null || true)}")
host_pids=(${host_pids:#})   # drop the empty element when nothing matched
if (( ${#host_pids} )); then
  say "an Adobe host that loads MediaCore plug-ins is running:"
  for pid in $host_pids; do
    # The executable only (ps -o comm=), not the arguments: they can hold
    # tokens and URLs that have no place in a log.
    print -r -- "  pid $pid $(ps -o comm= -p "$pid" 2>/dev/null || print '?')"
  done
  if (( force )); then
    say "continuing because of --force (restart those hosts to load the new build)"
  else
    die "quit them first, or pass --force."
  fi
else
  say "no Adobe host that loads MediaCore plug-ins is running"
fi

dest="$DEST_DIR/$NAME"
legacy="$DEST_DIR/$LEGACY_NAME"
staging="$DEST_DIR/.$NAME.installing.$$"

if (( dry )); then
  say "dry run: would create $DEST_DIR if missing"
  [[ -e "$dest" ]] && say "dry run: would replace existing $dest" || say "dry run: would create $dest"
  if [[ -e "$legacy" || -L "$legacy" ]]; then
    say "dry run: would remove legacy $legacy (same effect match name)"
  fi
  exit 0
fi

# ---- sudo with the password from stdin ---------------------------------------
password=""
if ! sudo -n true 2>/dev/null; then
  if [[ -t 0 ]]; then
    print -n -u2 "install.sh: sudo password: "
    # Interrupted at the prompt: echo back on, and stop (not carry on with an
    # empty password).
    trap 'stty echo 2>/dev/null; print -u2 ""; exit 130' INT TERM HUP
    stty -echo; IFS= read -r password || true; stty echo; print -u2 ""
    trap - INT TERM HUP
  else
    IFS= read -r password || true
  fi
  [[ -n "$password" ]] || die "sudo needs a password and none was given on stdin"
  print -r -- "$password" | sudo -S -p '' -v 2>/dev/null || die "sudo rejected the password"
fi
as_root() {
  if [[ -n "$password" ]]; then print -r -- "$password" | sudo -S -p '' "$@"; else sudo -n "$@"; fi
}
cleanup() { as_root rm -rf "$staging" 2>/dev/null || true; }
trap cleanup EXIT

# ---- copy, then swap ---------------------------------------------------------
as_root mkdir -p "$DEST_DIR"
say "destination folder: $DEST_DIR"
as_root rm -rf "$staging"
as_root ditto "$bundle" "$staging"                      # keeps the signature intact
as_root xattr -dr com.apple.quarantine "$staging" 2>/dev/null || true
as_root codesign --verify --strict "$staging" || die "copied bundle does not verify; nothing was installed"
# Do this only after the new copy verifies, and before publishing it: both
# bundles register the permanent match name "Open Denoise".
if [[ -e "$legacy" || -L "$legacy" ]]; then
  as_root rm -rf "$legacy"
  say "removed legacy $legacy (same effect match name)"
fi
if [[ -e "$dest" ]]; then
  as_root rm -rf "$dest"
  say "removed previous $dest"
fi
as_root mv "$staging" "$dest"
say "installed $dest"
say "binary: $(shasum -a 256 "$dest/Contents/MacOS/Denoise" | awk '{print $1}')"
say "done. Start (or restart) Premiere Pro, Media Encoder or After Effects; the effect is Video Effects > Denoise > Denoise."
