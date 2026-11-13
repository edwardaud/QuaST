#!/usr/bin/env python3
"""Calibrate the sd definition on run_E_long (L5 bottom wall) and our run before
the E6_long series exists: which convention reproduces the 4.36 % / 0.37 %
pair quoted in section 17.7 (sd of Nu_bot over t_nd = [1e-3, 2.16e-3])?

Candidates:
  native   pstdev of the samples that fall in the window (one per step for
           ASPECT `statistics`, one per 5 steps for our `frame_meta`, one per
           VTU frame for the gradient family)
  grid     pstdev of n equally spaced (linear-interpolated) samples, n = the
           native count, so the spacing equals the native cadence
  grid2000 the section 10.1 / 14.3 convention: 2000 equally spaced samples
"""
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import long_window as lw

HERE = os.path.dirname(os.path.abspath(__file__))
WINS = [(1e-3, 2.16e-3), (1e-3, 3e-3)]


def sd_native(rows, key, lo, hi):
    sel = [r[key] for r in rows if lo <= r["t"] <= hi]
    if len(sel) < 3:
        return None, len(sel)
    return st.pstdev(sel), len(sel)


def sd_grid(rows, key, lo, hi, n=None):
    sel = [r for r in rows if lo <= r["t"] <= hi]
    if len(sel) < 3:
        return None, len(sel)
    nn = n or len(sel)
    vals = [lw.interp(rows, key, lo + (hi - lo) * i / (nn - 1)) for i in range(nn)]
    if any(v is None for v in vals):
        return None, len(sel)
    return st.pstdev(vals), len(sel)


def main():
    asp = lw.read_aspect_stats([os.path.join(HERE, "run_E_long", "statistics")])
    grad = lw.read_grad_csv([os.path.join(lw.REPO, "out_julia", "agent",
                                         "asp_long_q2_series.csv")])
    ours = lw.read_frames(lw.DEF_FRAMES)
    print(f"E_long stats {len(asp)} rows, {asp[0]['t']:.3e}..{asp[-1]['t']:.3e}, "
          f"median cadence {st.median([b['t']-a['t'] for a,b in zip(asp,asp[1:])]):.3e}")
    print(f"E_long grad  {len(grad)} frames, {grad[0]['t']:.3e}..{grad[-1]['t']:.3e}")
    print(f"ours         {len(ours)} frames, {ours[0]['t']:.3e}..{ours[-1]['t']:.3e}, "
          f"median cadence {st.median([b['t']-a['t'] for a,b in zip(ours,ours[1:10])]):.3e}")

    for lo, hi in WINS:
        print()
        print(f"=== window [{lo:.2e}, {hi:.2e}] ===")
        hdr = f"{'series':22s} {'key':7s} {'n':>5s} {'mean':>10s} {'sd':>9s} {'sd %':>7s}"
        print(hdr)
        for name, rows in (("E_long CBF", asp), ("E_long grad", grad), ("ours", ours)):
            for key in ("nu_bot", "nu_top", "vrms"):
                m = lw.dense_mean(rows, key, lo, hi) if name != "E_long grad" \
                    else lw.frame_mean(rows, key, lo, hi)[0]
                s, n = sd_native(rows, key, lo, hi)
                sg, _ = sd_grid(rows, key, lo, hi)
                s2, _ = sd_grid(rows, key, lo, hi, 2000)
                def f(x, w=9, p=4):
                    return " " * w if x is None else f"{x:{w}.{p}f}"
                rel = 100.0 * s / abs(m) if (s is not None and m) else None
                sel = [r[key] for r in rows if lo <= r["t"] <= hi]
                mn = st.mean(sel) if sel else None
                reln = 100.0 * s / abs(mn) if (s is not None and mn) else None
                print(f"{name:22s} {key:7s} {n:5d} {m:10.4f} "
                      f"nat {f(s)} grid {f(sg)} g2000 {f(s2)} | "
                      f"{'   -  ' if rel is None else f'{rel:6.2f}%'} "
                      f"| natmean {mn:9.4f} sdnat/natmean "
                      f"{'   -  ' if reln is None else f'{reln:6.2f}%'}")
        print("  (native sd of the *smoothed* curve is not the question; 'sd %' is "
              "native sd / |dense mean|)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
