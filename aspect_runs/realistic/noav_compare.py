#!/usr/bin/env python3
"""Section 15 comparison: does switching ASPECT's artificial viscosity off
(`set beta = 0`, run_noav) move its wall heat flux toward the Julia card?

Data sets (all read-only):

  run_E/statistics         baseline (cR = 0.11, beta = 0.052: ASPECT defaults)
  run_noav/statistics      the same card with `set beta = 0`
  out_julia/case_realistic_E/frames/frame_meta.csv   our card (pointwise Nu)

Same conversion and the same 2000-sample linear interpolation as
compare_E.py / nu_bot_check.py / e6_compare.py:
    t_nd = t * kappa / d^2,  Nu = Q*d/(Lx*k*dT),  k = 3.3 W/(m K),  NORM = 8250 W

Usage:
  python3 noav_compare.py
  python3 noav_compare.py --base run_E/statistics --new run_noav/statistics
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
                             tmax=float(v[14]) / DT - 0.5,
                             tmin=float(v[15]) / DT - 0.5,
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
                            nu_bot=float(r["Nu_bot"]),
                            vrms=float(r["Vrms"]), tmax=float(r["Tmax"])))
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
    ts = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    vals = [interp(rows, key, t) for t in ts]
    ok = [v for v in vals if v is not None]
    return (st.mean(ok) if ok else float("nan")), len(ok) / len(vals)


def read_pvd(path):
    out = {}
    for m in re.finditer(r'timestep="([\d.eE+-]+)"[^>]*file="solution/solution-(\d+)\.pvtu"',
                         open(path, errors="replace").read()):
        out[int(m.group(2))] = float(m.group(1))
    return out


def nearest(rows, t):
    return min(rows, key=lambda r: abs(r["t"] - t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(HERE, "run_E", "statistics"))
    ap.add_argument("--new", default=os.path.join(HERE, "run_noav", "statistics"))
    ap.add_argument("--frames", default=os.path.join(
        REPO, "out_julia", "case_realistic_E", "frames", "frame_meta.csv"))
    ap.add_argument("--frame-table", nargs="*", type=int, default=[8, 12, 15, 16])
    a = ap.parse_args()

    E = read_stats(a.base)
    N = read_stats(a.new)
    ours = read_frames(a.frames)
    print(f"base: {len(E)} steps, t_nd {E[0]['t']:.3e} .. {E[-1]['t']:.3e}")
    print(f"new : {len(N)} steps, t_nd {N[0]['t']:.3e} .. {N[-1]['t']:.3e}")
    print(f"ours: {len(ours)} frames, t_nd {ours[0]['t']:.3e} .. {ours[-1]['t']:.3e}")
    print()

    def table(lo, hi, title):
        print(f"=== {title}  t_nd = {lo:.0e} .. {hi:.0e} ===")
        print(f"{'quantity':7s} {'base':>10s} {'new':>10s} {'d(new-base)':>12s} {'d%':>8s} "
              f"{'ours':>10s} {'new-ours%':>10s} {'base-ours%':>11s}")
        for k in ("nu_top", "nu_bot", "vrms", "tmax", "tmin"):
            e, ce = mean_at(E, k, lo, hi)
            n, cn = mean_at(N, k, lo, hi)
            o, co = mean_at(ours, k, lo, hi) if k in ("nu_top", "nu_bot", "vrms", "tmax") else (float("nan"), 0.0)
            print(f"{k:7s} {e:10.5f} {n:10.5f} {n-e:+12.5f} {100*(n-e)/abs(e):+7.3f}% "
                  f"{o:10.5f} {100*(n-o)/abs(o):+9.2f}% {100*(e-o)/abs(o):+10.2f}%")
        if min(ce, cn, co) <= 0.999:
            print(f"  (* partial: base {ce*100:.1f}%, new {cn*100:.1f}%, ours {co*100:.1f}%)")
        print()

    table(LO, HI, "whole window")
    table(3e-4, 1e-3, "late sub-window (70 % of the linear time)")

    edges = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    print("=== Nu_bot by sub-window ===")
    print(f"{'window':21s} {'base':>9s} {'new':>9s} {'d(new-base)':>12s} {'d%':>8s} "
          f"{'ours':>9s} {'new-ours%':>10s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        e, ce = mean_at(E, "nu_bot", lo, hi)
        n, cn = mean_at(N, "nu_bot", lo, hi)
        o, _ = mean_at(ours, "nu_bot", lo, hi)
        print(f"[{lo:.0e},{hi:.0e}]  {e:9.4f} {n:9.4f} {n-e:+12.4f} "
              f"{100*(n-e)/abs(e):+7.2f}% {o:9.4f} {100*(n-o)/abs(o):+9.2f}%")
    print()

    # per-frame Nu_bot at the new run's own frame times
    try:
        tn = read_pvd(os.path.join(HERE, "run_noav", "solution.pvd"))
        te = read_pvd(os.path.join(HERE, "run_E", "solution.pvd"))
    except OSError:
        tn = te = {}
    if tn:
        print("=== per-frame Nu_bot at run_noav's own frame times (CBF) ===")
        print(f"{'frame':5s} {'t_nd':>10s} {'base':>9s} {'new':>9s} {'d':>9s} {'%':>7s} "
              f"{'ours':>9s} {'new-ours%':>10s} {'Vrms base':>9s} {'new':>9s}")
        for i in a.frame_table:
            if i not in tn:
                continue
            t = tn[i] / T0
            rb = nearest(E, t)
            rn = nearest(N, t)
            o = interp(ours, "nu_bot", t)
            print(f"{i:5d} {t:10.3e} {rb['nu_bot']:9.4f} {rn['nu_bot']:9.4f} "
                  f"{rn['nu_bot']-rb['nu_bot']:+9.4f} "
                  f"{100*(rn['nu_bot']-rb['nu_bot'])/abs(rb['nu_bot']):+6.2f}% "
                  f"{(o if o is not None else float('nan')):9.4f} "
                  f"{(100*(rn['nu_bot']-o)/abs(o) if o else float('nan')):+9.2f}% "
                  f"{rb['vrms']:9.0f} {rn['vrms']:9.0f}")
        print()

    print("=== cost at matched t_nd (step, dt_nd, cells, c = dt_nd*vmax) ===")
    print(f"{'t_nd':>10s} {'base step':>9s} {'new step':>8s} {'base cells':>10s} "
          f"{'new cells':>9s} {'base dt':>9s} {'new dt':>9s} {'c base':>9s} {'c new':>9s}")
    for t in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3):
        rb, rn = nearest(E, t), nearest(N, t)
        if abs(rb["t"] - t) / t > 0.4 or abs(rn["t"] - t) / t > 0.4:
            continue
        print(f"{t:10.1e} {rb['step']:9d} {rn['step']:8d} {rb['cells']:10d} "
              f"{rn['cells']:9d} {rb['dt']:9.2e} {rn['dt']:9.2e} "
              f"{rb['dt']*rb['vmax']:9.3e} {rn['dt']*rn['vmax']:9.3e}")
    print()
    print(f"total steps base {E[-1]['step']} .. {E[-1]['step']}, new {N[-1]['step']}")
    print(f"final cells base {E[-1]['cells']} new {N[-1]['cells']}")


if __name__ == "__main__":
    main()
