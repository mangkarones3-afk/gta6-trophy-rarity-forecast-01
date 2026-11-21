#!/usr/bin/env python3
"""Validate main.py on the kit practice cases p00 and p01.

For each case: run the program on a copy of the input, check the output
against the exact contract (one "trophies" key, 55 rows, ids matching the
case exactly, five finite non-decreasing quantiles in [0, 100], file size
<= 65536 bytes), run it a second time and require byte-identical output
(sha256), and measure the wall-clock runtime per case.

Exit code 0 iff every check passes.
"""

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KIT_INPUT = os.path.expanduser(
    "~/workspace/makefun/gta6-trophy-rarity-forecast-01-kit/"
    "gta6-trophy-rarity-forecast-01-kit/input")
CASES = ("p00", "p01")


def run_once(case_id, in_src, work):
    in_dir = os.path.join(work, "in")
    out_dir = os.path.join(work, "out")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(in_src, os.path.join(in_dir, case_id + ".in"))
    env = dict(os.environ)
    env["MAKEFUN_INPUT_DIR"] = in_dir
    env["MAKEFUN_OUTPUT_DIR"] = out_dir
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, os.path.join(ROOT, "main.py")],
                          env=env, capture_output=True, text=True, timeout=60)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if proc.returncode != 0:
        raise RuntimeError("main.py exited %d: %s" % (proc.returncode,
                                                      proc.stderr[-400:]))
    out_path = os.path.join(out_dir, case_id + ".out")
    with open(out_path, "rb") as fh:
        blob = fh.read()
    return blob, elapsed_ms


def check_output(case, blob):
    problems = []
    if len(blob) > 65536:
        problems.append("output %d bytes > 65536" % len(blob))
    try:
        data = json.loads(blob.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        return ["output is not UTF-8 JSON: %s" % exc]
    if set(data.keys()) != {"trophies"}:
        problems.append("top-level keys %s != {trophies}" % sorted(data.keys()))
        return problems
    rows = data["trophies"]
    want_ids = [t["id"] for t in case["trophies"]]
    got_ids = [r.get("id") for r in rows]
    if got_ids != want_ids:
        problems.append("row ids do not match case trophies in order")
    if len(set(got_ids)) != len(got_ids):
        problems.append("duplicate trophy ids")
    if len(rows) != 55:
        problems.append("row count %d != 55" % len(rows))
    for r in rows:
        if set(r.keys()) != {"id", "q"}:
            problems.append("row %s has keys %s" % (r.get("id"), sorted(r.keys())))
            break
        q = r["q"]
        if len(q) != 5:
            problems.append("row %s has %d quantiles" % (r["id"], len(q)))
            break
        for v in q:
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                problems.append("row %s has non-finite quantile %r" % (r["id"], v))
                break
            if v < 0.0 or v > 100.0:
                problems.append("row %s quantile %r outside [0,100]" % (r["id"], v))
                break
        if any(q[k] < q[k - 1] for k in range(1, 5)):
            problems.append("row %s quantiles decrease: %s" % (r["id"], q))
            break
    return problems


def main():
    ok = True
    for case_id in CASES:
        src = os.path.join(KIT_INPUT, case_id + ".in")
        with open(src, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        with tempfile.TemporaryDirectory() as w1, \
                tempfile.TemporaryDirectory() as w2:
            blob1, ms1 = run_once(case_id, src, w1)
            blob2, ms2 = run_once(case_id, src, w2)
        problems = check_output(case, blob1)
        h1 = hashlib.sha256(blob1).hexdigest()
        h2 = hashlib.sha256(blob2).hexdigest()
        deterministic = (blob1 == blob2)
        if not deterministic:
            problems.append("two runs differ (sha256 %s vs %s)" % (h1, h2))
        status = "PASS" if not problems else "FAIL"
        if problems:
            ok = False
        print("%s %s: schema %s, sha256 run1 %s, run2 %s, deterministic %s, "
              "runtime %.1f ms / %.1f ms, size %d bytes"
              % (case_id, status, "ok" if not problems else "BAD", h1, h2,
                 deterministic, ms1, ms2, len(blob1)))
        for p in problems:
            print("   - " + p)
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
