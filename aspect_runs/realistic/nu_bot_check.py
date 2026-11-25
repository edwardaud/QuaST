#!/usr/bin/env python3
"""Where does the Nu_bot difference between ASPECT's `realistic_E` and our
`case_realistic_E` come from?

Three measurements, all on data the two runs already wrote:

  --windows   the window means of Nu_top/Nu_bot/Vrms/Tmax over t_nd = 1e-6..1e-3
              and the same split into sub-windows (where does the difference
              live?), plus the distribution of Nu_bot.
  --vtu       ASPECT's *gradient* wall flux, extracted from its own VTU output
              (Q2 nodal T, 2x2 patches -> the one-sided 3-point derivative),
              compared with (a) the `heat flux statistics` column (a Gresho
              CBF, not a gradient) and (b) our pointwise Nu_bot at the same t.
  --walls     ASPECT's bottom/top wall cell geometry (h, hx) from the VTU.

Why --vtu matters: ASPECT's `heat flux statistics` is a consistent-boundary-flux
(CBF) reaction of the *BDF2 time step*, see
`source/postprocess/heat_flux_statistics.cc` and
`source/postprocess/heat_flux_map.cc:41` (assembly) / `:319` (integration).
Its own gradient flux is not written by any postprocessor, so this script
reconstructs it; that separates "definition" from "solution" differences.

The VTU mesh is deal.II's DataOut: every Q2 cell is written as a 2x2 block of
bilinear quads whose vertices are the Q2 support points (0, h/2, h).  For a
wall node x there are therefore points at y = 0, h/2, h and

    dT/dy(x,0) = (-3 T(x,0) + 4 T(x,h/2) - T(x,h)) / h

is the exact y-derivative of the quadratic interpolant -- the same family as
our own one-sided GLL wall derivative.  Checked against the analytic initial
state (Nu_bot = -4.000, Nu_top = +6.000 exactly) and against ASPECT's own CBF
column (agrees to <0.7% for t_nd > 2e-4).

Usage:
  python3 nu_bot_check.py                 # all three
  python3 nu_bot_check.py --windows --vtu
  python3 nu_bot_check.py --vtu --frames PATH  --stats PATH  --solution DIR

Defaults: run_E/statistics, run_E/solution, out_julia/case_realistic_E/frames.
"""
import argparse
import glob
import math
import os
import re
import statistics as st
from collections import Counter, defaultdict

import numpy as np
import meshio

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))

D = 2.9e6                     # box depth [m]
DT = 1000.0                   # temperature drop [K]
LX = 2.5 * D                  # box length [m]
T0 = D * D / 1.0e-6           # time scale [s]
NORM = LX * 3.3 * DT / D      # W per unit of Nu (= 8250)

LO, HI = 1.0e-6, 1.0e-3       # the comparison window in t_nd


# --------------------------------------------------------------------- inputs --
def read_stats(path):
    """ASPECT `statistics` rows: t_nd, Nu_top, Nu_bot, Vrms_nd, Tmax_nd."""
    names, rows = {}, []
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        if line.startswith("#"):
            m = re.match(r"#\s*(\d+):\s*(.*)", line)
            if m:
                names[int(m.group(1))] = m.group(2).strip()
            continue
        if not line.strip():
            continue
        v = line.split()
        try:
            t = float(v[1])
            rows.append(dict(step=int(v[0]), t=t / T0,
                             dt=float(v[2]) / T0,
                             vrms=float(v[10]) * D / 1.0e-6,
                             tmax=float(v[14]) / DT - 0.5,
                             nu_bot=-float(v[18]) / NORM,
                             nu_top=float(v[19]) / NORM,
                             cbf_ok=int(v[0]) >= 2))
        except (ValueError, IndexError):
            continue
    return rows


def read_frames(path):
    """Our `frame_meta.csv` (`t` is already t_nd)."""
    import csv
    out = []
    for r in csv.DictReader(open(path)):
        try:
            out.append(dict(step=int(r["step"]), t=float(r["t"]),
                            nu_top=float(r["Nu_top"]), nu_bot=float(r["Nu_bot"]),
                            vrms=float(r["Vrms"]), tmax=float(r["Tmax"])))
        except (TypeError, ValueError, KeyError):
            continue
    return out


def interp(rows, key, t):
    """Linear interpolation of `key` in t; None outside the range."""
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
    return st.mean([interp(rows, key, t) for t in ts])


# ------------------------------------------------------------------ --windows --
def windows(stats_path, frames_path):
    asp = [r for r in read_stats(stats_path) if r["cbf_ok"]]
    ours = read_frames(frames_path)
    print("=== window means, t_nd = 1e-6 .. 1e-3 (uniform in time, 2000 samples) ===")
    print(f"{'quantity':8s} {'ASPECT':>10s} {'ours':>10s} {'diff':>8s}")
    for k in ("nu_top", "nu_bot", "vrms", "tmax"):
        a = mean_at(asp, k, LO, HI)
        o = mean_at(ours, k, LO, HI)
        print(f"{k:8s} {a:10.5g} {o:10.5g} {100*(a-o)/abs(o):+7.2f}%")
    print()
    print("=== Nu_bot by sub-window (uniform in each) ===")
    edges = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    for lo, hi in zip(edges[:-1], edges[1:]):
        a = mean_at(asp, "nu_bot", lo, hi)
        o = mean_at(ours, "nu_bot", lo, hi)
        print(f"[{lo:.0e},{hi:.0e}]  A={a:9.4f}  O={o:9.4f}  {100*(a-o)/abs(o):+7.2f}%")
    print()
    print("=== distribution of Nu_bot over the window (raw samples) ===")

    def pct(v, p):
        s = sorted(v)
        return s[min(len(s) - 1, int(p / 100 * (len(s) - 1)))]

    for name, rows in (("A", asp), ("O", ours)):
        v = [r["nu_bot"] for r in rows if LO <= r["t"] <= HI]
        print(f"  {name}: n={len(v):5d} min={min(v):8.3f} p10={pct(v,10):8.3f} "
              f"med={pct(v,50):8.3f} p90={pct(v,90):8.3f} max={max(v):8.3f} "
              f"mean={st.mean(v):8.4f} sd={st.pstdev(v):6.3f}")


# ---------------------------------------------------------------------- --vtu --
def vtu_frames(sol_dir):
    out = {}
    for p in sorted(glob.glob(os.path.join(sol_dir, "solution-*.vtu"))):
        m = re.search(r"solution-(\d+)\.(\d+)\.vtu$", os.path.basename(p))
        if m:
            out.setdefault(int(m.group(1)), []).append(p)
    return [(k, sorted(v)) for k, v in sorted(out.items())]


def vtu_collect(pieces):
    """(x, y, T) over all rank pieces of one frame, deduplicated on (x, y)."""
    xs, ys, Ts = [], [], []
    for p in pieces:
        m = meshio.read(p)
        xs.append(m.points[:, 0]); ys.append(m.points[:, 1]); Ts.append(m.point_data["T"])
    x = np.concatenate(xs); y = np.concatenate(ys); T = np.concatenate(Ts)
    key = np.round(np.stack([x / D, y / D], axis=1), 9)
    _, idx = np.unique(key, axis=0, return_index=True)
    return x[idx], y[idx], T[idx]


def wall_flux(x, y, T, wall):
    """(Nu, wall-node x/D, wall cell heights h/D) from the Q2 patch structure."""
    ymax = y.max()
    sel = np.abs(y) < 1e-6 * D if wall == "bottom" else np.abs(y - ymax) < 1e-6 * D
    xw = np.unique(x[sel])
    d = np.full(xw.size, np.nan)
    hs = np.full(xw.size, np.nan)
    for i, xv in enumerate(xw):
        col = np.abs(x - xv) < 1e-6 * D
        yy, tt = y[col], T[col]
        o = np.argsort(yy)
        if wall == "top":
            o = o[::-1]
        yy, tt = yy[o], tt[o]
        keep = []
        for j in range(len(yy)):
            if not keep or abs(yy[j] - yy[keep[-1]]) > 1e-9 * D:
                keep.append(j)
            if len(keep) == 3:
                break
        if len(keep) < 3:
            continue
        y0, y1, y2 = yy[keep]
        h = abs(y0 - y2)
        if abs(abs(y0 - y1) - h / 2.0) > 1e-3 * h:
            continue
        val = (-3.0 * tt[keep[0]] + 4.0 * tt[keep[1]] - tt[keep[2]]) / h
        d[i] = -val if wall == "top" else val
        hs[i] = h / D
    ok = np.isfinite(d)
    integ = float(np.sum(0.5 * (d[ok][1:] + d[ok][:-1]) * np.diff(xw[ok]))) if ok.sum() > 1 else 0.0
    return -(integ / LX) * D / DT, xw[ok] / D, hs[ok]


def vtu_report(sol_dir, frames_path, stats_path):
    stats = read_stats(stats_path)
    ours = read_frames(frames_path)
    tabs = sorted(stats, key=lambda r: r["t"])

    def stat_at(t, key):
        best = min(tabs, key=lambda r: abs(r["t"] - t))
        return best[key]

    rows = []
    print("=== ASPECT gradient flux (VTU) vs its CBF statistic vs our pointwise Nu_bot ===")
    print(f"{'t_nd':>9} {'step':>5} {'A_grad':>9} {'A_cbf':>9} {'O':>8} {'A_grad-O':>9} {'%':>7} "
          f"{'h_wall/d':>8} {'n_wall':>6}")
    for idx, pieces in vtu_frames(sol_dir):
        x, y, T = vtu_collect(pieces)
        nb, dx, hs = wall_flux(x, y, T, "bottom")
        nt, _, _ = wall_flux(x, y, T, "top")
        # the frame time: TIME lives in the FieldData
        t_s = None
        for p in pieces:
            m = meshio.read(p)
            if "TIME" in m.field_data:
                t_s = float(np.asarray(m.field_data["TIME"]).ravel()[0])
                break
        t_nd = (t_s / T0) if t_s is not None else tabs[min(idx, len(tabs) - 1)]["t"]
        cb = stat_at(t_nd, "nu_bot")
        cbt = stat_at(t_nd, "nu_top")
        o = interp(ours, "nu_bot", t_nd)
        d = 100 * (nb - o) / abs(o) if (o not in (None, 0)) else float("nan")
        hm = float(np.median(hs)) if hs.size else float("nan")
        print(f"{t_nd:9.3e} {idx:5d} {nb:9.3f} {cb:9.3f} "
              f"{(o if o is not None else float('nan')):8.3f} {nb-(o or 0):9.3f} {d:7.2f} "
              f"{hm:8.4f} {dx.size:6d}")
        rows.append((t_nd, idx, nb, cb, nt, cbt, hm, dx.size))

    # <Nu_bot> converted to the gradient family
    ts = np.linspace(LO, HI, 4000)
    cbf = np.array([interp(tabs, "nu_bot", t) for t in ts])
    wr = np.array([interp(ours, "nu_bot", t) for t in ts])
    gt = np.array([r[0] for r in rows if r[0] > 0])
    gd = np.array([r[3] - r[2] for r in rows if r[0] > 0])
    diff = np.interp(ts, gt, gd, left=gd[0], right=gd[-1])
    print()
    print(f"<Nu_bot> ASPECT CBF      = {cbf.mean():+9.4f}")
    print(f"<Nu_bot> ASPECT gradient = {cbf.mean()-diff.mean():+9.4f}  (<cbf-grad> = {diff.mean():+.4f})")
    print(f"<Nu_bot> ours            = {wr.mean():+9.4f}")
    print(f"gap CBF  vs ours = {100*(cbf.mean()-wr.mean())/abs(wr.mean()):+.2f}%")
    print(f"gap GRAD vs ours = {100*((cbf.mean()-diff.mean())-wr.mean())/abs(wr.mean()):+.2f}%")
    return rows


def walls_report(sol_dir):
    print("=== ASPECT wall cell geometry (from the VTU patches) ===")
    for idx, pieces in vtu_frames(sol_dir):
        if idx not in (0, 1, 2, 4, 8, 12, 16):
            continue
        x, y, T = vtu_collect(pieces)
        for wall in ("bottom", "top"):
            ymax = y.max()
            sel = np.abs(y) < 1e-6 * D if wall == "bottom" else np.abs(y - ymax) < 1e-6 * D
            xw = np.unique(x[sel])
            dx = np.diff(xw) / D
            print(f"  frame {idx:2d} {wall:6s}: node spacing {dx.min():.4f}..{dx.max():.4f} d "
                  f"(hx = 2x), histogram {dict(Counter([round(v, 5) for v in dx]))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", action="store_true")
    ap.add_argument("--vtu", action="store_true")
    ap.add_argument("--walls", action="store_true")
    ap.add_argument("--stats", default=os.path.join(HERE, "run_E", "statistics"))
    ap.add_argument("--solution", default=os.path.join(HERE, "run_E", "solution"))
    ap.add_argument("--frames", default=os.path.join(
        REPO, "out_julia", "case_realistic_E", "frames", "frame_meta.csv"))
    a = ap.parse_args()
    if not (a.windows or a.vtu or a.walls):
        a.windows = a.vtu = a.walls = True
    if a.windows:
        windows(a.stats, a.frames)
        print()
    if a.vtu:
        vtu_report(a.solution, a.frames, a.stats)
        print()
    if a.walls:
        walls_report(a.solution)


if __name__ == "__main__":
    main()
