#!/usr/bin/env python3
"""Find unstable CPU cores: run the same calculation on one logical CPU at a time.

Some Intel 13th and 14th generation CPUs have cores that make errors under load. Programs then crash
(segmentation fault) or give wrong numbers at random. This script runs a fixed calculation on each logical
CPU in turn, pinned to that CPU. A good CPU always gives the same answer. A bad CPU crashes or gives a
different answer.

    python scripts/find_bad_cores.py                 # 60 s per CPU
    python scripts/find_bad_cores.py --seconds 120   # longer test finds more errors
    python scripts/find_bad_cores.py --cpus 0-7      # test only some CPUs

At the end it prints a taskset command that uses only the CPUs that passed. Only the Python standard library
is used, so it runs in any Python 3 environment. Linux only.
"""

from __future__ import annotations

import argparse
import collections
import os
import subprocess
import sys
import time

WORK = r'''
import hashlib, os, sys, time, zlib
os.sched_setaffinity(0, [{CPU}])
end = time.time() + {SECONDS}
digest = None
rounds = 0
while time.time() < end:
    h = hashlib.sha256()
    # Floating-point vector arithmetic, like the code that crashed.
    x, y, z = 0.1, 0.2, 0.3
    for i in range(200000):
        x, y, z = (y * 1.0000001 + z * 0.5) % 7.0, (z * 0.9999999 - x * 0.25) % 5.0, (x * 1.5 + y) % 3.0
    h.update(repr((x, y, z)).encode())
    # Integer arithmetic and memory: big numbers and compression.
    n = 1
    for i in range(1, 3000):
        n = (n * (i | 1) + i) % (1 << 4096)
    h.update(str(n).encode())
    data = bytes((i * 7919) % 251 for i in range(300000))
    h.update(zlib.compress(data, 9))
    d = h.hexdigest()
    if digest is None:
        digest = d
    elif d != digest:
        print("MISMATCH", rounds, flush=True)
        sys.exit(3)
    rounds += 1
print("DIGEST", digest, rounds, flush=True)
'''


def parse_cpus(text: str) -> list[int]:
    cpus = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(part))
    return cpus


def as_ranges(cpus: list[int]) -> str:
    out, start, prev = [], None, None
    for c in sorted(cpus):
        if start is None:
            start = prev = c
        elif c == prev + 1:
            prev = c
        else:
            out.append(f"{start}-{prev}" if prev > start else str(start))
            start = prev = c
    if start is not None:
        out.append(f"{start}-{prev}" if prev > start else str(start))
    return ",".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=60, help="test time per CPU (default 60)")
    parser.add_argument("--cpus", default=None, help="CPUs to test, for example 0-7,16 (default: all)")
    args = parser.parse_args()

    cpus = parse_cpus(args.cpus) if args.cpus else sorted(os.sched_getaffinity(0))
    print(f"Testing {len(cpus)} CPUs, {args.seconds:g} s each, about {len(cpus) * args.seconds / 60:.0f} min.")
    print("Close other heavy programs first. Press Ctrl+C to stop.\n")

    results = {}
    for cpu in cpus:
        start = time.time()
        proc = subprocess.run([sys.executable, "-c", WORK.replace("{CPU}", str(cpu)).replace("{SECONDS}", str(args.seconds))],
                              capture_output=True, text=True)
        out = proc.stdout.strip().splitlines()
        last = out[-1] if out else ""
        if proc.returncode < 0:
            results[cpu] = ("CRASHED", f"killed by signal {-proc.returncode}")
        elif proc.returncode == 3 or last.startswith("MISMATCH"):
            results[cpu] = ("WRONG", "gave a different answer in a later round")
        elif proc.returncode != 0 or not last.startswith("DIGEST"):
            results[cpu] = ("CRASHED", (proc.stderr.strip().splitlines() or ["exit code %d" % proc.returncode])[-1])
        else:
            _, digest, rounds = last.split()
            results[cpu] = ("DIGEST", digest, int(rounds))
        r = results[cpu]
        shown = f"ok ({r[2]} rounds)" if r[0] == "DIGEST" else f"{r[0]}: {r[1]}"
        print(f"CPU {cpu:3d}: {shown}   [{time.time() - start:.0f} s]", flush=True)

    # The answer that most CPUs gave is the correct one.
    digests = collections.Counter(r[1] for r in results.values() if r[0] == "DIGEST")
    good_digest = digests.most_common(1)[0][0] if digests else None
    good, bad = [], []
    for cpu, r in results.items():
        if r[0] == "DIGEST" and r[1] == good_digest:
            good.append(cpu)
        else:
            bad.append(cpu)
            if r[0] == "DIGEST":
                results[cpu] = ("WRONG", "answer differs from most other CPUs")

    print()
    if not bad:
        print("No bad CPU found. A longer test (--seconds 300) can still find rare errors.")
    else:
        print("Bad CPUs: " + ", ".join(f"{c} ({results[c][0]})" for c in sorted(bad)))
        if good:
            print("Run programs only on the good CPUs, for example:")
            print(f"    taskset -c {as_ranges(good)} python -m mdagent start")
        else:
            print("No CPU passed. Check the output above: the script itself may have failed.")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
