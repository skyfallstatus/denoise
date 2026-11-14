#!/bin/zsh
# SPDX-License-Identifier: MIT
# Static checks on a built Denoise.plugin: layout, Info.plist, PiPL (via DeRez),
# exported entry points, architecture, signature. Read-only.
# Usage: verify_bundle.sh <path to Denoise.plugin>
set -u
B="${1:?usage: verify_bundle.sh <Denoise.plugin>}"
fail=0
ok()  { print -r -- "ok    $1"; }
bad() { print -r -- "FAIL  $1"; fail=1; }
check() { if eval "$2" >/dev/null 2>&1; then ok "$1"; else bad "$1"; fi }

check "bundle directory Denoise.plugin"       '[[ -d "$B" && "${B:t}" == Denoise.plugin ]]'
check "Contents/Info.plist lints"             'plutil -lint "$B/Contents/Info.plist"'
check "Contents/PkgInfo = eFKTFXTC"           '[[ "$(cat "$B/Contents/PkgInfo")" == eFKTFXTC ]]'
check "Contents/MacOS/Denoise exists"         '[[ -f "$B/Contents/MacOS/Denoise" ]]'
check "Contents/Resources/Denoise.rsrc"       '[[ -s "$B/Contents/Resources/Denoise.rsrc" ]]'

pl() { /usr/libexec/PlistBuddy -c "Print :$1" "$B/Contents/Info.plist" 2>/dev/null; }
check "CFBundlePackageType eFKT"              '[[ "$(pl CFBundlePackageType)" == eFKT ]]'
check "CFBundleSignature FXTC"                '[[ "$(pl CFBundleSignature)" == FXTC ]]'
check "CFBundleExecutable Denoise"            '[[ "$(pl CFBundleExecutable)" == Denoise ]]'
check "CFBundleName Denoise"                  '[[ "$(pl CFBundleName)" == Denoise ]]'
check "CFBundleIdentifier unchanged"          '[[ "$(pl CFBundleIdentifier)" == dev.opendenoise.denoise ]]'
check "copyright names Denoise"               '[[ "$(pl NSHumanReadableCopyright)" == "MIT License. Denoise contributors." ]]'

check "Mach-O bundle, arm64"                  'file "$B/Contents/MacOS/Denoise" | grep -q "Mach-O 64-bit bundle arm64"'
syms="$(nm -gU "$B/Contents/MacOS/Denoise" | awk '{print $3}' | sort | tr '\n' ' ')"
print -r -- "      exported symbols: $syms"
check "exports exactly EffectMain + xGPUFilterEntry" '[[ "$syms" == "_EffectMain _xGPUFilterEntry " ]]'
check "no unresolved od_ symbols"             '! nm -u "$B/Contents/MacOS/Denoise" | grep -Eq "^[[:space:]]*_od_"'

# PiPL: decode the resource fork-less file back and look at each property.
pipl="$(DeRez -useDF "$B/Contents/Resources/Denoise.rsrc" 2>/dev/null)"
hex="$(print -r -- "$pipl" | LC_ALL=C sed -n 's/^[[:space:]]*\$"\([0-9A-F ]*\)".*/\1/p' | tr -d ' \n')"
check "one PiPL resource, id 16000"           '[[ "$(print -r -- "$pipl" | grep -c "^data ")" == 1 ]] && print -r -- "$pipl" | grep -q "data .PiPL. (16000)"'
h() { print -rn -- "$1" | xxd -p | tr -d '\n' | tr a-f A-F; }   # ascii -> hex
prop() { [[ "$hex" == *"$(h 8BIM)$(h "$1")"* ]]; }
check "PiPL kind = eFKT"                      '[[ "$hex" == *"$(h 8BIMkind)0000000000000004$(h eFKT)"* ]]'
check "PiPL name = Denoise"                   '[[ "$hex" == *"$(h 8BIMname)000000000000000807$(h Denoise)"* ]]'
check "PiPL category = Denoise"               '[[ "$hex" == *"$(h 8BIMcatg)000000000000000807$(h Denoise)"* ]]'
check "PiPL arm64 entry = EffectMain"         '[[ "$hex" == *"$(h 8BIMma64)000000000000000B0A$(h EffectMain)"* ]]'
check "PiPL has no Intel entry"               '! prop mi64'
# Literal compatibility pin: NEVER derive this from the current display name
# or OD_EFFECT_MATCHNAME. Existing projects refer to AE.Open Denoise.
check "PiPL permanent match name = Open Denoise" '[[ "$hex" == *"$(h 8BIMeMNA)000000000000000D0C$(h "Open Denoise")"* ]]'
check "PiPL out flags  = 0x06000002 (no custom UI)" '[[ "$hex" == *"$(h 8BIMeGLO)000000000000000406000002"* ]]'
check "PiPL out flags2 = 0x08000108"          '[[ "$hex" == *"$(h 8BIMeGL2)000000000000000408000108"* ]]'
check "PiPL effect version = 0x00008000"      '[[ "$hex" == *"$(h 8BIMeVER)000000000000000400008000"* ]]'
check "PiPL PiPL version 2.0"                 '[[ "$hex" == *"$(h 8BIMePVR)000000000000000400020000"* ]]'
check "PiPL spec version present"             'prop eSVR'

check "code signature valid (ad hoc)"         'codesign --verify --strict "$B"'
check "signature is ad hoc"                   'codesign -dv "$B" 2>&1 | grep -q "Signature=adhoc"'
check "rsrc is sealed by the signature"       'grep -q "Resources/Denoise.rsrc" "$B/Contents/_CodeSignature/CodeResources"'

check "bundled offline help exists" '[[ -s "$B/Contents/Resources/help/index.html" ]]'
check "help has the control guide" 'grep -q "Denoise control guide" "$B/Contents/Resources/help/index.html"'
check "help sealed by signature" 'grep -q "Resources/help/index.html" "$B/Contents/_CodeSignature/CodeResources"'

if (( fail )); then print "bundle verification FAILED"; exit 1; fi
print "bundle verification passed"
