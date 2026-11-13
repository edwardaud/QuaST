#!/usr/bin/env python3
"""Check that the statistics history ASPECT restores from a checkpoint is the
same as the one the source run wrote.

`run_E6_long/statistics` starts at step 0 although the run resumed at step 3716:
ASPECT restores the whole statistics table from the checkpoint and appends.  This
verifies that the restored part (steps 0..3715) equals `run_E6/statistics` to the
last printed digit in every column except the last one (the visualization file
name, which changes because the run has a different output directory).

Usage: python3 check_e6_restore.py [run_E6/statistics] [run_E6_long/statistics]
"""
import sys

A = sys.argv[1] if len(sys.argv) > 1 else "run_E6/statistics"
B = sys.argv[2] if len(sys.argv) > 2 else "run_E6_long/statistics"
N = 3716


def rows(path, n):
    out = []
    for line in open(path, errors="replace"):
        if line.startswith("#") or not line.strip():
            continue
        out.append(line.split()[:20])        # drop the visualization file name
        if len(out) == n:
            break
    return out


a, b = rows(A, N), rows(B, N)
print(f"{A}: {len(a)} rows;  {B}: {len(b)} rows (first {N} data rows)")
assert len(a) == len(b) == N
bad = 0
worst = 0.0
for i, (ra, rb) in enumerate(zip(a, b)):
    if ra[:3] != rb[:3]:
        bad += 1
        continue
    for x, y in zip(ra[3:], rb[3:]):
        d = abs(float(x) - float(y))
        worst = max(worst, d)
        if d != 0.0:
            bad += 1
print(f"rows differing in any of columns 1-20: {bad}; max |difference| = {worst:.3e}")


def col(path, idx, n):
    return [float(r[idx]) for r in rows(path, n)]


for idx, name in ((18, "Nu_bot (CBF)"), (19, "Nu_top"), (10, "V_rms")):
    x, y = col(A, idx, N), col(B, idx, N)
    print(f"  {name:12s}: {sum(1 for p, q in zip(x, y) if p != q)} of {N} rows differ")
