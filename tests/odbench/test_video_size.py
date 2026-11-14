#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""odbench --video reads rotated clips upright, at the size ffmpeg delivers.

A phone clip shot in portrait is stored landscape (say 3840x2160) with a
display matrix that turns it a quarter turn; ffmpeg turns the frames it
decodes, so they come out 2160x3840. odbench used to take the stored size from
ffprobe and scale the turned frames back to it, which squeezed the picture
unless --size was passed by hand.

Run by `make test` (tests/odbench/test.mk) with the system Python; no test
media. It makes its own clips: a 128x72 test pattern (PNG in .mov, lossless
RGB, 6 frames) and remuxed copies with a display rotation of 90, -90 and 180
degrees, then checks, for each:
  - odbench reports the display size (72x128 for a quarter turn, 128x72
    otherwise) without --size;
  - with --identity (an exact copy), odbench's --out-raw frames equal ffmpeg's
    own decode of the clip (autorotated, nothing else), pixel for pixel, and
    the 180-degree clip's frames are the plain clip's turned by half a turn
    (the size alone cannot show that turn);
  - --size stays an explicit override: --size 36x64 on the 90-degree clip
    equals ffmpeg's decode scaled to 36x64 after turning.

Everything goes to a new folder under $TMPDIR, removed at the end. Every file
ffmpeg writes is capped at 8 MB (RLIMIT_FSIZE); odbench writes its frames into
a pipe this test reads with an 8 MB cap (see odbench() for why not a file);
every child has a timeout, and the test stops as soon as a clip fails to be
made. Skips (exit 0)
when the ffmpeg that odbench calls is missing, or when that ffmpeg is too old
to write a display rotation (-display_rotation, FFmpeg 7).

Usage: tests/odbench/test_video_size.py [path/to/odbench]
"""
import array
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# The binaries odbench calls (kFfmpeg in tools/odbench/odbench.mm).
FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
W, H, FRAMES = 128, 72, 6
FILE_LIMIT = 8 << 20  # bytes, per file ffmpeg writes (RLIMIT_FSIZE)
OUT_LIMIT = 8 << 20  # bytes of frames read from one odbench run
TOTAL_LIMIT = 16 << 20  # bytes, everything in the test folder
CHECKS = 0
FAILS = 0

def check(desc, ok):
    global CHECKS, FAILS
    CHECKS += 1
    if not ok:
        FAILS += 1
    print("%s %s" % ("ok  " if ok else "FAIL", desc))
    return ok


def cap_files():
    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_LIMIT, FILE_LIMIT))


def run(cmd, timeout=60, **kw):
    return subprocess.run(cmd, capture_output=True, timeout=timeout, preexec_fn=cap_files, **kw)


def folder_bytes(d):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs)


def ffmpeg_frames(clip, vf=None):
    """ffmpeg's own decode of the clip (autorotation on, as by default) as
    16-bit RGBA, and the frame size it delivered (from a PAM header)."""
    head = run([FFMPEG, "-v", "error", "-nostdin", "-i", clip, "-frames:v", "1", "-f", "image2pipe", "-c:v",
                "pam", "-pix_fmt", "gray", *(["-vf", vf] if vf else []), "-"]).stdout
    end = head.find(b"ENDHDR")
    tok = head[:end].decode("ascii", "replace").split() if end > 0 else []
    size = (int(tok[tok.index("WIDTH") + 1]), int(tok[tok.index("HEIGHT") + 1])) if "HEIGHT" in tok else None
    p = run([FFMPEG, "-v", "error", "-nostdin", "-i", clip, *(["-vf", vf] if vf else []), "-fps_mode",
             "passthrough", "-f", "rawvideo", "-pix_fmt", "rgba64le", "-"])
    if p.returncode != 0:
        raise RuntimeError("ffmpeg could not decode %s: %s" % (clip, p.stderr.decode()[-300:]))
    a = array.array("H")
    a.frombytes(p.stdout)
    if sys.byteorder != "little":
        a.byteswap()
    return size, a


def odbench(binary, clip, out_dir, tag, extra=()):
    """Runs odbench with --identity (copy) on the clip. Returns its JSON report,
    its output frames (BGRA float32) and "", or None, None and the error.

    No file-size rlimit here: Metal appends to its shader cache files, which
    are far larger than any cap this test could set, whenever a library is
    not cached yet (SIGXFSZ). So the frames go into a pipe that this test
    reads with a cap of OUT_LIMIT bytes, killing odbench past it; only the
    JSON report (a few kB) is a file."""
    rep = os.path.join(out_dir, tag + ".json")
    r, w = os.pipe()
    p = subprocess.Popen([binary, "--video", clip, "--frames", str(FRAMES), "--warmup", "0", "--identity",
                          "--out-raw", "/dev/fd/%d" % w, "--json", rep, "--deadline", "60", "--mem-limit-gb", "4",
                          *extra], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, pass_fds=(w,))
    os.close(w)
    raw = bytearray()
    with os.fdopen(r, "rb") as fh:
        while True:
            chunk = fh.read(1 << 16)
            if not chunk:
                break
            raw += chunk
            if len(raw) > OUT_LIMIT:
                p.kill()
                break
    try:
        err = p.communicate(timeout=90)[1].decode()
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        return None, None, "(no exit after 90 s)"
    if len(raw) > OUT_LIMIT:
        return None, None, "(wrote more than %d bytes of frames: stopped)" % OUT_LIMIT
    if p.returncode != 0:
        return None, None, "(exit %d) %s" % (p.returncode, err[-300:])
    f = array.array("f")
    f.frombytes(bytes(raw[:len(raw) // 4 * 4]))
    if sys.byteorder != "little":
        f.byteswap()
    with open(rep) as fh:
        return json.load(fh), f, ""


def same_pixels(bgra, rgba):
    """odbench's float BGRA against ffmpeg's 16-bit RGBA: every value within a
    quarter code (one code is 1/65535)."""
    if len(bgra) != len(rgba):
        return False, "%d values from odbench, %d from ffmpeg" % (len(bgra), len(rgba))
    tol = 0.25 / 65535.0
    worst = 0.0
    for i in range(0, len(rgba), 4):
        for c, s in ((0, 2), (1, 1), (2, 0), (3, 3)):
            d = abs(bgra[i + c] - rgba[i + s] / 65535.0)
            if d > worst:
                worst = d
    return worst <= tol, "largest difference %.2f codes" % (worst * 65535.0)


def half_turn(rgba, w, h):
    """The frames of a 16-bit RGBA array turned by half a turn (each frame's
    pixels in reverse order)."""
    out = array.array("H", rgba)
    n = w * h * 4
    for f in range(len(rgba) // n):
        for p in range(w * h):
            q = f * n + (w * h - 1 - p) * 4
            out[f * n + p * 4:f * n + p * 4 + 4] = rgba[q:q + 4]
    return out


def main():
    binary = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "odbench"))
    if not os.access(FFMPEG, os.X_OK):
        print("skip: %s not found (odbench --video needs it)" % FFMPEG)
        return 0
    if not os.access(binary, os.X_OK):
        print("FAIL odbench not found at %s (make odbench)" % binary)
        return 1
    tmp = tempfile.mkdtemp(prefix="od-test-video-size.")
    try:
        plain = os.path.join(tmp, "plain.mov")
        p = run([FFMPEG, "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i", "testsrc2=size=%dx%d:rate=25" % (W, H),
                 "-frames:v", str(FRAMES), "-c:v", "png", "-pix_fmt", "rgb24", plain])
        if p.returncode != 0 or not os.path.isfile(plain) or not 0 < os.path.getsize(plain) <= FILE_LIMIT:
            print("FAIL could not make the test clip: %s" % p.stderr.decode()[-300:])
            return 1
        clips = {"none": plain}
        for deg in (90, -90, 180):
            out = os.path.join(tmp, "rot%d.mov" % deg)
            p = run([FFMPEG, "-v", "error", "-nostdin", "-y", "-display_rotation", str(deg), "-i", plain,
                     "-c", "copy", out])
            err = p.stderr.decode()
            if p.returncode != 0 and "display_rotation" in err:
                print("skip: this ffmpeg cannot write a display rotation (-display_rotation): %s" % err.strip()[-200:])
                return 0
            if p.returncode != 0 or not os.path.isfile(out) or not 0 < os.path.getsize(out) <= FILE_LIMIT:
                print("FAIL could not make the rotated clip (%d degrees): %s" % (deg, err[-300:]))
                return 1
            clips[deg] = out
        if os.access(FFPROBE, os.X_OK):
            p = run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                     "stream=width,height:stream_side_data=rotation", "-of", "json", clips[90]])
            s = json.loads(p.stdout or b"{}").get("streams", [{}])[0]
            rot = [d.get("rotation") for d in s.get("side_data_list", []) if "rotation" in d]
            if not check("the 90-degree clip is stored %dx%d with a quarter-turn display matrix (ffprobe: %sx%s, %s)"
                         % (W, H, s.get("width"), s.get("height"), rot),
                         (s.get("width"), s.get("height")) == (W, H) and rot and abs(abs(rot[0]) - 90) < 1):
                return 1

        for key, clip in clips.items():
            quarter = key in (90, -90)
            want = (H, W) if quarter else (W, H)
            size, ref = ffmpeg_frames(clip)
            check("rotation %s: ffmpeg delivers %dx%d (expected %dx%d)" % (key, size[0], size[1], want[0], want[1]),
                  size == want)
            rep, got, err = odbench(binary, clip, tmp, "od_%s" % key)
            if not check("rotation %s: odbench runs without --size %s" % (key, err), rep is not None):
                continue
            check("rotation %s: odbench reports %sx%s, the display size" % (key, rep["width"], rep["height"]),
                  (rep["width"], rep["height"]) == want and rep["framesRendered"] == FRAMES)
            ok, how = same_pixels(got, ref)
            check("rotation %s: odbench's frames are ffmpeg's upright frames (%s)" % (key, how), ok)
            if key == "none":
                plain = ref
            elif key == 180:
                # Not vacuous: the half turn keeps the size, so check the
                # frames really were turned.
                ok, how = same_pixels(got, half_turn(plain, W, H))
                check("rotation 180: odbench's frames are the plain clip's turned by half a turn (%s)" % how, ok)

        # --size is still an explicit override, applied after the turn.
        size, ref = ffmpeg_frames(clips[90], "scale=36:64:flags=bicubic")
        rep, got, err = odbench(binary, clips[90], tmp, "od_size", ["--size", "36x64"])
        if check("--size 36x64 on the 90-degree clip runs %s" % err, rep is not None):
            check("--size 36x64: odbench reports %sx%s" % (rep["width"], rep["height"]),
                  (rep["width"], rep["height"]) == (36, 64) and size == (36, 64))
            ok, how = same_pixels(got, ref)
            check("--size 36x64: frames equal ffmpeg's, turned then scaled (%s)" % how, ok)

        used = folder_bytes(tmp)
        check("test folder holds %d bytes (limit %d)" % (used, TOTAL_LIMIT), used <= TOTAL_LIMIT)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("test_video_size: %d checks, %d failures" % (CHECKS, FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
