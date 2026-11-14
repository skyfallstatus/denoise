#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""odbench --async never writes into a host buffer the GPU is still using.

With --async, od_render returns before the GPU has read the window's host
buffers or written the output. odbench used to refill a ring slot (without
--preload) and copy the next frame into its single in-place scratch buffer
(--in-place) with nothing to say the earlier render was done, so a render
could read the next frame's pixels instead of its own.

The check: --hold-gpu-ms holds odbench's queue from the first render until
odbench first waits for the GPU, so the CPU runs as far ahead as odbench
lets it, and the report counts "hazards": CPU writes into a host buffer
before the GPU work of the render that last used it had ended, judged by
that render's completion as od_render reports it (ODStats.completed), not
by the fences. Each async mode must give 0 hazards, none left unjudged,
and, written with --out-raw (read back late, after each render's fence),
exactly the bytes of the same run rendered blocking. --debug-unfenced (no
fences, the ring and scratch at their old sizes: what odbench did before)
and --debug-fence-elsewhere (fences on a queue of their own, which order
nothing) must show hazards, the no-output modes included, and, without
--preload, different bytes: the check is not vacuous, and does not take
the fences' word for it.

Runs with --verify-cache (the plug-in path: every cached neighbour is
checked against its host buffer, so every window buffer is read by every
render) and a manual profile (deterministic output). Run by `make test`
(tests/odbench/test.mk) with the system Python; no media, no files besides
the JSON reports (frames come through a pipe, capped at 8 MB per run), every
run with a timeout and odbench's own --deadline.

Usage: tests/odbench/test_async.py [path/to/odbench]
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_LIMIT = 8 << 20
COMMON = ["--size", "128x72", "--frames", "16", "--warmup", "0", "--radius", "2", "--manual-sigma", "0.03",
          "--verify-cache", "--deadline", "60", "--mem-limit-gb", "4"]
HOLD = ["--hold-gpu-ms", "3000"]
CHECKS = 0
FAILS = 0


def check(desc, ok):
    global CHECKS, FAILS
    CHECKS += 1
    if not ok:
        FAILS += 1
    print("%s %s" % ("ok  " if ok else "FAIL", desc), flush=True)
    return ok


def odbench(binary, tmp, tag, extra, output=True):
    """Runs odbench; returns (JSON report, sha256 of the --out-raw bytes or
    None, number of bytes, error text)."""
    rep = os.path.join(tmp, tag + ".json")
    args = [binary, *COMMON, *extra, "--json", rep]
    r = w = None
    if output:
        r, w = os.pipe()
        args += ["--out-raw", "/dev/fd/%d" % w]
    p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, pass_fds=(w,) if output else ())
    digest, n = None, 0
    if output:
        os.close(w)
        h = hashlib.sha256()
        with os.fdopen(r, "rb") as fh:
            while True:
                chunk = fh.read(1 << 16)
                if not chunk:
                    break
                n += len(chunk)
                if n > OUT_LIMIT:
                    p.kill()
                    break
                h.update(chunk)
        digest = h.hexdigest()
    try:
        err = p.communicate(timeout=90)[1].decode(errors="replace")
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        return None, None, n, "(no exit after 90 s)"
    if n > OUT_LIMIT:
        return None, None, n, "(more than %d bytes of frames: stopped)" % OUT_LIMIT
    if p.returncode != 0:
        return None, None, n, "(exit %d) %s" % (p.returncode, err.strip()[-300:])
    with open(rep) as fh:
        return json.load(fh), digest, n, ""


def main():
    binary = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "odbench"))
    if not os.access(binary, os.X_OK):
        print("FAIL odbench not found at %s (make odbench)" % binary)
        return 1
    tmp = tempfile.mkdtemp(prefix="od-test-async.")
    try:
        ref, want, n, err = odbench(binary, tmp, "blocking", [])
        if not check("blocking reference: %d bytes of frames %s" % (n, err), ref is not None and n == 16 * 128 * 72 * 16):
            return 1
        _, inplace, _, err = odbench(binary, tmp, "blocking_inplace", ["--in-place"])
        check("blocking --in-place: the same bytes %s" % err, inplace == want)

        cases = [
            ("--async", ["--async"], True),
            ("--async --in-place", ["--async", "--in-place"], True),
            ("--async --preload", ["--async", "--preload"], True),
            ("--async --in-place --preload", ["--async", "--in-place", "--preload"], True),
            ("--async, no output", ["--async"], False),
            ("--async --in-place --preload, no output", ["--async", "--in-place", "--preload"], False),
            ("--async --in-flight 1", ["--async", "--in-flight", "1"], True),
        ]
        for name, extra, output in cases:
            rep, got, n, err = odbench(binary, tmp, "c%d" % CHECKS, extra + HOLD, output)
            if not check("%s, queue held: runs %s" % (name, err), rep is not None):
                continue
            a = rep.get("async", {})
            check("%s: %s hazard(s), %s unresolved, %s wait(s) on host-buffer fences"
                  % (name, a.get("hazards"), a.get("hazardsUnresolved"), a.get("bufferWaits")),
                  a.get("hazards") == 0 and a.get("hazardsUnresolved") == 0)
            if output:
                check("%s: the blocking run's bytes" % name, got == want)
        # The CPU really did run ahead of the held queue: it had to wait.
        rep, _, _, err = odbench(binary, tmp, "waits", ["--async"] + HOLD, False)
        waits = (rep or {}).get("async", {}).get("bufferWaits")
        check("--async without --preload, queue held: the CPU waited on fences %s times %s" % (waits, err),
              isinstance(waits, int) and waits > 0)

        # Not vacuous: without the fences the same runs overwrite buffers in use.
        rep, got, _, err = odbench(binary, tmp, "unfenced", ["--async", "--debug-unfenced"] + HOLD)
        hz = (rep or {}).get("async", {}).get("hazards")
        check("--async --debug-unfenced, queue held: %s hazard(s) %s" % (hz, err), isinstance(hz, int) and hz > 0)
        check("--async --debug-unfenced: bytes differ from the blocking run (the overwrite reached the output)",
              rep is not None and got != want)
        rep, _, _, err = odbench(binary, tmp, "unfenced_inplace",
                                 ["--async", "--in-place", "--preload", "--debug-unfenced"] + HOLD, False)
        hz = (rep or {}).get("async", {}).get("hazards")
        check("--async --in-place --preload --debug-unfenced: %s hazard(s) %s" % (hz, err), isinstance(hz, int) and hz > 0)

        # Fences on another queue do not protect these buffers. Completion
        # accounting must detect the overwrite, with or without output.
        for name, extra, output in [
            ("--async --debug-fence-elsewhere, no output", ["--async"], False),
            ("--async --in-place --preload --debug-fence-elsewhere, no output", ["--async", "--in-place", "--preload"],
             False),
            ("--async --debug-fence-elsewhere", ["--async"], True),
        ]:
            rep, got, _, err = odbench(binary, tmp, "c%d" % CHECKS, extra + ["--debug-fence-elsewhere"] + HOLD, output)
            a = (rep or {}).get("async", {})
            hz = a.get("hazards")
            check("%s, queue held: %s hazard(s), %s unresolved %s" % (name, hz, a.get("hazardsUnresolved"), err),
                  isinstance(hz, int) and hz > 0 and a.get("hazardsUnresolved") == 0)
            if output:
                check("%s: bytes differ from the blocking run (the overwrite reached the output)" % name,
                      rep is not None and got != want)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("test_async: %d checks, %d failures" % (CHECKS, FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
