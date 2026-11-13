#!/usr/bin/env python3
"""Section 12 comparison: does one extra bottom-boundary-layer refinement level
in ASPECT (`run_E6`) move its Nu_bot toward the Julia card?

Three data sets, all read-only:

  run_E/statistics      baseline ASPECT E (h_bottom = 1/32 d, level 5)
  run_E6/statistics     E with the bottom two level-5 rows refined once more
                        (h_bottom = 1/64 d, level 6, y <= 0.0625)
  out_julia/case_realistic_E/frames/frame_meta.csv   our card (pointwise Nu)

Reports the window means over t_nd = 1e-6 .. 1e-3 and the sub-windows of
section 10.1, for Nu_top and Nu_bot, plus the E6-E and E6-ours differences.
The conversion is the one of compare_E.py / nu_bot_check.py:
Nu = Q*d/(Lx*k*dT) with k = 3.3 W/(m K), t_nd = t*kappa/d^2.

Usage:
  python3 e6_compare.py            # all three data sets
  python3 e6_compare.py --no-ours  # ASPECT E vs E6 only
"""
import argparse
import csv
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))

D = 2.9e6
DT = 1000.0
LX = 2.5 * D
T0 = D * D / 1.0e-6
NORM = LX * 3.3 * DT / D
LO, HI = 1.0e-6, 1.0e-3


def read_stats(path):
    rows = []
    for line in open(path, errors="replace"):
        if line.startswith("#") or not line.strip():
            continue
        v = line.split()
        try:
            rows.append(dict(step=int(v[0]), t=float(v[1]) / T0,
                             dt=float(v[2]) / T0, cells=int(float(v[3])),
                             vrms=float(v[10]) * D / 1.0e-6,
                             vmax=float(v[11]) * D / 1.0e-6,
                             nu_bot=-float(v[18]) / NORM, nu_top=float(v[19]) / NORM,
                             cbf_ok=int(v[0]) >= 2))
        except (ValueError, IndexError):
            continue
    return [r for r in rows if r["cbf_ok"]]


def read_frames(path):
    out = []
    for r in csv.DictReader(open(path)):
        try:
            out.append(dict(t=float(r["t"]), nu_top=float(r["Nu_top"]),
                            nu_bot=float(r["Nu_bot"])))
        except (TypeError, ValueError, KeyError):
            continue
    return out


def interp(rows, key, t):
    if not rows or t < rows[0]["t"] or t > rows[-1]["t"]:
        return None
    for a, b in zip(rows, rows[1:]):
        if a["t"] <= t <= b["t"]:
            if b["t"] == a["t"]:
                return a[key]
            w = (t - a["t"]) / (b["t"] - a["t"])
            return a[key] + w * (b[key] - a[key])
    return rows[-1][key]


def mean_at(rows, key, lo, hi, n=2000):
    """Mean over lo..hi; returns (mean, covered fraction) so a run that has not
    reached `hi` yet still gives a (marked) partial number."""
    ts = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    vals = [interp(rows, key, t) for t in ts]
    ok = [v for v in vals if v is not None]
    return (st.mean(ok) if ok else float("nan")), len(ok) / len(vals)


def fmt(v, cov):
    return f"{v:10.5f}" if cov > 0.999 else f"{v:9.4f}*"


def read_pvd(path):
    """Frame times from an ASPECT solution.pvd (index -> t)."""
    out = {}
    for m in re.finditer(r'timestep="([\d.eE+-]+)"[^>]*file="solution/solution-(\d+)\.pvtu"',
                         open(path, errors="replace").read()):
        out[int(m.group(2))] = float(m.group(1))
    return out


def nearest(rows, t):
    return min(rows, key=lambda r: abs(r["t"] - t))


def frame_table(E, E6, ours, frames):
    ta = read_pvd(os.path.join(HERE, "run_E", "solution.pvd"))
    t6 = read_pvd(os.path.join(HERE, "run_E6", "solution.pvd"))
    print("=== per-frame Nu_bot at ASPECT's own frame times (CBF, statistics) ===")
    print(f"{'frame':5s} {'t_nd':>10s} {'E':>9s} {'E6':>9s} {'d(E6-E)':>9s} "
          f"{'%':>7s} {'ours':>9s} {'E6-ours%':>9s} {'E-ours%':>8s} {'Vrms E':>8s} {'E6':>8s}")
    for i in frames:
        if i not in ta or i not in t6:
            print(f"{i:5d}  (frame not written yet in one of the runs)")
            continue
        ra, r6 = nearest(E, ta[i] / T0), nearest(E6, t6[i] / T0)
        o = interp(ours, "nu_bot", ta[i] / T0)
        line = (f"{i:5d} {ta[i]/T0:10.3e} {ra['nu_bot']:9.4f} {r6['nu_bot']:9.4f} "
                f"{r6['nu_bot']-ra['nu_bot']:+9.4f} {100*(r6['nu_bot']-ra['nu_bot'])/abs(ra['nu_bot']):+6.2f}%")
        if o is not None:
            line += (f" {o:9.4f} {100*(r6['nu_bot']-o)/abs(o):+8.2f}% "
                     f"{100*(ra['nu_bot']-o)/abs(o):+7.2f}%")
        line += f" {ra['vrms']:8.0f} {r6['vrms']:8.0f}"
        print(line)
    print()


def cost_table(E, E6):
    """Step / dt / cells at matched t_nd, and the CFL constant c = dt_nd*vmax."""
    print("=== cost: step, dt_nd, cells, c = dt_nd*vmax at matched t_nd ===")
    print(f"{'t_nd':>10s} {'E step':>7s} {'E6 step':>8s} {'E cells':>8s} {'E6 cells':>9s} "
          f"{'E dt_nd':>9s} {'E6 dt_nd':>10s} {'c_E':>9s} {'c_E6':>9s}")
    for t in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3):
        ra, r6 = nearest(E, t), nearest(E6, t)
        if abs(ra["t"] - t) / t > 0.3 or abs(r6["t"] - t) / t > 0.3:
            continue
        print(f"{t:10.1e} {ra['step']:7d} {r6['step']:8d} {ra['cells']:8d} {r6['cells']:9d} "
              f"{ra['dt']:9.2e} {r6['dt']:10.2e} {ra['dt']*ra['vmax']:9.3e} "
              f"{r6['dt']*r6['vmax']:9.3e}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", default=os.path.join(HERE, "run_E", "statistics"))
    ap.add_argument("--E6", default=os.path.join(HERE, "run_E6", "statistics"))
    ap.add_argument("--frames", default=os.path.join(
        REPO, "out_julia", "case_realistic_E", "frames", "frame_meta.csv"))
    ap.add_argument("--no-ours", action="store_true")
    ap.add_argument("--frame-table", nargs="*", type=int, default=[8, 12, 15, 16],
                    help="ASPECT frame indices for the per-frame Nu_bot table")
    a = ap.parse_args()

    E = read_stats(a.E)
    E6 = read_stats(a.E6)
    print(f"E : {len(E)} steps, t_nd {E[0]['t']:.3e} .. {E[-1]['t']:.3e}")
    print(f"E6: {len(E6)} steps, t_nd {E6[0]['t']:.3e} .. {E6[-1]['t']:.3e}")
    ours = None if a.no_ours else read_frames(a.frames)
    if ours:
        print(f"O : {len(ours)} frames, t_nd {ours[0]['t']:.3e} .. {ours[-1]['t']:.3e}")
    print()

    def table(lo, hi, title):
        print(f"=== {title}  t_nd = {lo:.0e} .. {hi:.0e} ===")
        head = f"{'quantity':7s} {'E':>10s} {'E6':>10s} {'d(E6-E)':>10s} {'d(E6-E)%':>9s}"
        if ours:
            head += f" {'ours':>10s} {'E6-ours':>9s} {'E6-ours%':>9s} {'E-ours%':>8s}"
        print(head)
        for k in ("nu_top", "nu_bot"):
            e, ce = mean_at(E, k, lo, hi)
            e6, c6 = mean_at(E6, k, lo, hi)
            d = e6 - e
            line = f"{k:7s} {fmt(e, ce)} {fmt(e6, c6)} {d:+10.5f} {100*d/abs(e):+8.3f}%"
            if ours:
                o, co = mean_at(ours, k, lo, hi)
                line += (f" {fmt(o, co)} {100*(e6-o)/abs(o):+8.3f}% "
                         f"{100*(e-o)/abs(o):+7.3f}%")
            print(line)
        if c6 <= 0.999 or ce <= 0.999:
            print(f"  (* partial: E covers {ce*100:.1f}%, E6 covers {c6*100:.1f}% of the window)")
        print()

    table(LO, HI, "whole window")
    edges = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    print("=== Nu_bot by sub-window ===")
    head = f"{'window':21s} {'E':>9s} {'E6':>9s} {'d(E6-E)':>9s}"
    if ours:
        head += f" {'ours':>9s} {'E6-ours%':>9s}"
    print(head)
    for lo, hi in zip(edges[:-1], edges[1:]):
        e, ce = mean_at(E, "nu_bot", lo, hi)
        e6, c6 = mean_at(E6, "nu_bot", lo, hi)
        line = f"[{lo:.0e},{hi:.0e}]  {e:9.4f} {e6:9.4f} {e6-e:+9.4f}"
        if ours:
            o, _ = mean_at(ours, "nu_bot", lo, hi)
            line += f" {o:9.4f} {100*(e6-o)/abs(o):+8.2f}%"
        if c6 <= 0.999:
            line += "  (partial)"
        print(line)
    print()
    if ours:
        print("Note: E6-ours% is the number to compare with the -6.11% of section 10.1;")
        print("E-ours% is the same quantity for the baseline (reproduces -6.11%).")
        print()
        frame_table(E, E6, ours, a.frame_table)
        cost_table(E, E6)


if __name__ == "__main__":
    main()
