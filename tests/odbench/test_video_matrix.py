#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""odbench --video reads an untagged HD clip as BT.709, as Premiere does.

ffmpeg converts an untagged Y'CbCr stream to RGB as BT.601 whatever its size;
Premiere reads untagged HD and UHD camera files as BT.709. odbench's --video-matrix auto (the default) follows the
stream's tag and reads an untagged stream of HD size or more as BT.709;
`ffmpeg` keeps ffmpeg's own reading, bt709 and bt601 force a matrix.

Run by `make test` (tests/odbench/test.mk) with the system Python; no test
media. It makes 1280x16 (HD by width) and 640x16 (SD) Y'CbCr clips, untagged
and full range like the real camera clips, and one tagged BT.470BG, then
checks with --identity (an exact copy) that:
  - the untagged HD clip comes out as ffmpeg's decode with in_color_matrix
    bt709, pixel for pixel, and the report says "bt709", "untagged HD";
  - with --video-matrix ffmpeg it comes out as ffmpeg's default decode (which
    differs: the clip can tell the two apart);
  - the untagged SD clip and the tagged clip come out as ffmpeg reads them;
  - --video-matrix bt601 forces BT.601.

Bounds and helpers are those of test_video_size.py (8 MB per file ffmpeg
writes, 8 MB of frames per odbench run, timeouts on every child).

Usage: tests/odbench/test_video_matrix.py [path/to/odbench]
"""
import importlib.util
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("test_video_size", os.path.join(HERE, "test_video_size.py"))
vs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vs)

UNTAG = "setparams=colorspace=unknown:color_primaries=unknown:color_trc=unknown:range=pc"


def make(path, args):
    p = vs.run([vs.FFMPEG, "-v", "error", "-nostdin", "-y", *args, "-c:v", "ffv1", "-pix_fmt", "yuv422p10le", path])
    if p.returncode != 0 or not os.path.isfile(path) or not 0 < os.path.getsize(path) <= vs.FILE_LIMIT:
        print("FAIL could not make %s: %s" % (os.path.basename(path), p.stderr.decode()[-300:]))
        sys.exit(1)


def main():
    binary = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(vs.ROOT, "build", "odbench"))
    if not os.access(vs.FFMPEG, os.X_OK):
        print("skip: %s not found (odbench --video needs it)" % vs.FFMPEG)
        return 0
    if not os.access(binary, os.X_OK):
        print("FAIL odbench not found at %s (make odbench)" % binary)
        return 1
    tmp = tempfile.mkdtemp(prefix="od-test-video-matrix.")
    try:
        hd, sd, tagged = (os.path.join(tmp, n) for n in ("hd.mkv", "sd.mkv", "tagged.mkv"))
        src = ["-f", "lavfi", "-i", "testsrc2=size=1280x16:rate=25", "-frames:v", str(vs.FRAMES)]
        make(hd, src + ["-vf", "scale=out_color_matrix=bt709:out_range=pc," + UNTAG, "-color_range", "pc"])
        make(sd, ["-i", hd, "-vf", "crop=640:16:0:0," + UNTAG, "-color_range", "pc"])
        make(tagged, ["-i", hd, "-vf", "setparams=colorspace=bt470bg", "-colorspace", "bt470bg",
                      "-color_range", "pc"])
        cases = [  # clip, odbench options, ffmpeg -vf that must give the same pixels, report matrix, why
            ("untagged HD, auto", hd, [], "scale=in_color_matrix=bt709", "bt709", "untagged HD"),
            ("untagged HD, --video-matrix ffmpeg", hd, ["--video-matrix", "ffmpeg"], None, "ffmpeg", "ffmpeg"),
            ("untagged HD, --video-matrix bt601", hd, ["--video-matrix", "bt601"], "scale=in_color_matrix=bt601",
             "bt601", "forced"),
            ("untagged SD, auto", sd, [], None, "ffmpeg", "untagged SD"),
            ("tagged BT.470BG, auto", tagged, [], None, "ffmpeg", "tag bt470bg"),
        ]
        for k, (desc, clip, extra, vf, matrix, why) in enumerate(cases):
            rep, frames, err = vs.odbench(binary, clip, tmp, "m%d" % k, extra)
            if rep is None:
                vs.check("%s: odbench ran %s" % (desc, err), False)
                continue
            vs.check("%s: report says %s (%s)" % (desc, rep.get("videoMatrix"), rep.get("videoMatrixWhy")),
                     rep.get("videoMatrix") == matrix and rep.get("videoMatrixWhy") == why)
            _, want = vs.ffmpeg_frames(clip, vf)
            ok, how = vs.same_pixels(frames, want)
            vs.check("%s: pixels equal ffmpeg's decode%s (%s)" % (desc, " with " + vf if vf else "", how), ok)
        # The clip must tell BT.601 from BT.709, or the checks above say nothing.
        _, a = vs.ffmpeg_frames(hd, "scale=in_color_matrix=bt709")
        _, b = vs.ffmpeg_frames(hd)
        vs.check("the HD clip decodes differently as BT.709 and as BT.601", a != b)
        vs.check("test folder holds %d bytes (limit %d)" % (vs.folder_bytes(tmp), vs.TOTAL_LIMIT),
                 vs.folder_bytes(tmp) <= vs.TOTAL_LIMIT)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("test_video_matrix: %d checks, %d failures" % (vs.CHECKS, vs.FAILS))
    return 1 if vs.FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
