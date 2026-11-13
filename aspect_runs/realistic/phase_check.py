#!/usr/bin/env python3
"""Is the cross-code Nu_bot difference a *phase* difference in the internal-heating
transient, or a structural difference in the bottom boundary layer?
(docs/ASPECT_REALISTIC_COMPARE.md section 17.6)

Four measurements, all on data the two runs already wrote:

  --identity   the volume-integrated energy identity  d<T>/dt = Nu_bot - Nu_top + H
               (H = 10): d<T>/dt (centred difference of the volume-averaged T),
               the flux-implied rate, and the residual (latent/closure terms),
               for both codes at matched t_nd -- the parent test of section 17.6.
  --lag        cross-correlation of the two codes' Nu_top(t) and Nu_bot(t) as a
               function of a time shift.  A pure phase difference must give
               r -> 1 at the same lag for both; a structural difference cannot.
  --slopes     regression of Nu_bot on Nu_top in each code ("how strongly does
               the bottom flux track the top flux"), over the long window and
               over its late part.
  --overturns  the number of box overturns in each window, t_ov = 1/Vrms
               (docs/dev/CASE_REALISTIC.md section 7.4), i.e. the integral of Vrms dt.

Defaults: ASPECT = run_E_cont/statistics + run_E_long/statistics (merged, keyed
by step) and the exact-Q2 gradient series of the VTU frames; ours =
out_julia/case_realistic_E/frames/frame_meta.csv (read-only).

Usage:
  python3 phase_check.py                # all four
  python3 phase_check.py --lag --slopes
  python3 phase_check.py --identity --dw 2e-4
"""
import argparse
import csv
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))

D = 2.9e6
T0 = 8.41e18
NORM = 8250.0
H = 10.0

LO, HI = 1.0e-3, 5.0e-3          # the long window
DEF_GRAD = [os.path.join(REPO, "out_julia", "case_realistic_E_aspimport",
                         "asp_series.csv")]


# --------------------------------------------------------------------- inputs --
def read_stats(paths):
    """ASPECT statistics: t_nd, Nu_bot (CBF), Nu_top (CBF), Vrms_nd, Tmean_nd."""
    out, seen = [], set()
    for p in paths:
        for line in open(p, errors="replace"):
            if line.startswith("#") or not line.strip():
                continue
            v = line.split()
            try:
                s = int(v[0])
                if s in seen:
                    continue
                seen.add(s)
                out.append(dict(step=s, t=float(v[1]) / T0,
                                vrms=float(v[10]) * D / 1e-6,
                                tmean=float(v[13]) / 1000.0 - 0.5,
                                nu_bot=-float(v[18]) / NORM,
                                nu_top=float(v[19]) / NORM))
            except (ValueError, IndexError):
                continue
    out.sort(key=lambda r: r["t"])
    return out


def read_grad(paths):
    """`extract_aspect_state.py --series` CSV rows, de-duplicated on time."""
    rows, seen = [], set()
    for p in paths:
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            try:
                ts = round(float(r["t_s"]), -3)
                if ts in seen:
                    continue
                seen.add(ts)
                rows.append(dict(t=float(r["t_nd"]), nu_bot=float(r["nu_bot"]),
                                 nu_top=float(r["nu_top"]),
                                 vrms=float(r["Vrms_nd"]),
                                 tmean=float(r["Tmean_nd"])))
            except (TypeError, ValueError, KeyError):
                continue
    rows.sort(key=lambda r: r["t"])
    return rows


def read_ours(path):
    out = []
    for r in csv.DictReader(open(path)):
        try:
            out.append(dict(t=float(r["t"]), nu_bot=float(r["Nu_bot"]),
                            nu_top=float(r["Nu_top"]), vrms=float(r["Vrms"]),
                            tmean=float(r["Tmean"])))
        except (TypeError, ValueError, KeyError):
            continue
    out.sort(key=lambda r: r["t"])
    return out


# ------------------------------------------------------------------- helpers ---
def interp(rows, key, t):
    if not rows or t < rows[0]["t"] or t > rows[-1]["t"]:
        return None
    lo, hi = 0, len(rows) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if rows[mid]["t"] <= t:
            lo = mid
        else:
            hi = mid
    a, b = rows[lo], rows[hi]
    w = (t - a["t"]) / (b["t"] - a["t"]) if b["t"] > a["t"] else 0.0
    return a[key] + w * (b[key] - a[key])


def corr(x, y):
    mx, my = st.mean(x), st.mean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else float("nan")


def rate(rows, t, dw):
    a, b = interp(rows, "tmean", t - dw), interp(rows, "tmean", t + dw)
    return None if (a is None or b is None) else (b - a) / (2 * dw)


# ------------------------------------------------------------------- sections --
def identity(A, O, times, dw):
    print(f"=== d<T>/dt = Nu_bot - Nu_top + H (H = {H}); centred difference "
          f"+-{dw:.1e} ===")
    print(f"{'t_nd':>9} | {'TmeanA':>8} {'TmeanO':>8} {'dT A-O':>8} | "
          f"{'dT/dt A':>8} {'dT/dt O':>8} {'gap':>7} | "
          f"{'flux A':>8} {'flux O':>8} {'gap':>7} | {'resid A':>8} {'resid O':>8}")
    for t in times:
        ta, to = interp(A, "tmean", t), interp(O, "tmean", t)
        ra, ro = rate(A, t, dw), rate(O, t, dw)
        nba, nta = interp(A, "nu_bot", t), interp(A, "nu_top", t)
        nbo, nto = interp(O, "nu_bot", t), interp(O, "nu_top", t)
        if None in (ta, to, ra, ro, nba, nta, nbo, nto):
            print(f"{t:9.2e} |  (not covered)")
            continue
        fa, fo = nba - nta + H, nbo - nto + H
        print(f"{t:9.2e} | {ta:8.5f} {to:8.5f} {ta-to:+8.5f} | "
              f"{ra:8.3f} {ro:8.3f} {ra-ro:+7.3f} | "
              f"{fa:8.3f} {fo:8.3f} {fa-fo:+7.3f} | "
              f"{ra-fa:+8.3f} {ro-fo:+8.3f}")
    print("(gap = A - O; flux = Nu_bot - Nu_top + H; resid = d<T>/dt - flux)")
    print("If the two gaps tracked, the Nu_bot difference would be the stored-heat")
    print("(phase) difference.  The residual differs by 0.36-0.85 between the codes,")
    print("i.e. as large as the 0.17-0.25 effect, so the identity alone cannot")
    print("decide -- see --lag and --slopes.")


def lag(A, O, lo, hi, grid, maxlag):
    print(f"=== cross-correlation of the two codes over [{lo:.2e},{hi:.2e}] "
          f"(grid {grid:.1e}, lags +-{maxlag:.1e}) ===")
    ts = []
    t = lo
    while t <= hi:
        ts.append(t)
        t += grid
    for key in ("nu_top", "nu_bot"):
        a = [interp(A, key, t) for t in ts]
        best = None
        rows = []
        for k in range(int(-maxlag / grid), int(maxlag / grid) + 1):
            lag_ = k * grid
            pairs = [(a[i], interp(O, key, t - lag_)) for i, t in enumerate(ts)
                     if lo <= t - lag_ <= hi]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            if len(pairs) < 20:
                continue
            r = corr([p[0] for p in pairs], [p[1] for p in pairs])
            rows.append((lag_, r, len(pairs)))
            if best is None or r > best[1]:
                best = rows[-1]
        print(f"  {key}: best lag {best[0]:+.2e}  r = {best[1]:+.3f}  "
              f"(n = {best[2]}, r at lag 0 = "
              f"{[x[1] for x in rows if abs(x[0]) < 1e-12][0]:+.3f})")
        for lag_, r, n in rows[::max(1, len(rows) // 6)]:
            print(f"      lag {lag_:+.2e}  r = {r:+.3f}")
    print("A pure phase difference gives r -> 1 at one lag for BOTH quantities;")
    print("Nu_top does (r >= 0.996 everywhere), Nu_bot does not (<= 0.64).")


def slopes(pairs, lo, hi):
    print(f"=== regression Nu_bot = a + b*Nu_top over [{lo:.2e},{hi:.2e}] ===")
    print(f"{'series':14s} {'n':>5} {'<Nu_top>':>9} {'<Nu_bot>':>9} "
          f"{'sd(Nu_top)':>10} {'sd(Nu_bot)':>10} {'b':>9} {'r':>7}")
    for name, rows in pairs:
        sel = [r for r in rows if lo <= r["t"] <= hi]
        if len(sel) < 10:
            print(f"{name:14s} not covered")
            continue
        x = [r["nu_top"] for r in sel]
        y = [r["nu_bot"] for r in sel]
        mx, my = st.mean(x), st.mean(y)
        sxx = sum((a - mx) ** 2 for a in x)
        sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
        print(f"{name:14s} {len(sel):5d} {mx:9.3f} {my:+9.4f} "
              f"{st.pstdev(x):10.3f} {st.pstdev(y):10.4f} "
              f"{sxy/sxx if sxx else float('nan'):+9.4f} {corr(x, y):+7.3f}")


def overturns(pairs):
    print("=== box overturns in each window, t_ov = 1/Vrms "
          "(docs/dev/CASE_REALISTIC.md 7.4): integral of Vrms dt ===")
    for lo, hi in ((1e-6, 1e-3), (1e-3, 2e-3), (1e-3, 3e-3), (1e-3, 4e-3),
                   (1e-3, 5e-3), (3.5e-3, 5e-3)):
        out = []
        for name, rows in pairs:
            sel = [r for r in rows if lo <= r["t"] <= hi]
            if len(sel) < 2:
                out.append(f"{name}: -")
                continue
            tot = sum(0.5 * (a["vrms"] + b["vrms"]) * (b["t"] - a["t"])
                      for a, b in zip(sel, sel[1:]))
            out.append(f"{name}: {tot:5.2f}")
        print(f"  [{lo:.1e},{hi:.1e}]  " + "   ".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", nargs="+",
                    default=[os.path.join(HERE, "run_E_cont", "statistics"),
                             os.path.join(HERE, "run_E_long", "statistics")])
    ap.add_argument("--grad-csv", nargs="*", default=None,
                    help="the exact-Q2 gradient series of run_E_cont / "
                         "run_E_long (default: the section-16 CSV)")
    ap.add_argument("--frames", default=os.path.join(
        REPO, "out_julia", "case_realistic_E", "frames", "frame_meta.csv"))
    ap.add_argument("--dw", type=float, default=1.5e-4)
    ap.add_argument("--times", nargs="*", type=float,
                    default=[1.1e-3, 1.3e-3, 1.5e-3, 1.7e-3, 2.0e-3, 2.5e-3,
                             3.0e-3, 3.5e-3, 4.0e-3, 4.5e-3, 5.0e-3])
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--lag", action="store_true")
    ap.add_argument("--slopes", action="store_true")
    ap.add_argument("--overturns", action="store_true")
    a = ap.parse_args()
    if not (a.identity or a.lag or a.slopes or a.overturns):
        a.identity = a.lag = a.slopes = a.overturns = True

    stats = read_stats(a.stats)
    grad = read_grad(a.grad_csv if a.grad_csv is not None else DEF_GRAD)
    ours = read_ours(a.frames)
    print(f"ASPECT statistics: {len(stats)} steps, "
          f"t_nd {stats[0]['t']:.4e}..{stats[-1]['t']:.4e}")
    print(f"ASPECT gradient  : {len(grad)} frames, "
          f"t_nd {grad[0]['t']:.4e}..{grad[-1]['t']:.4e}")
    print(f"ours             : {len(ours)} frames, "
          f"t_nd {ours[0]['t']:.4e}..{ours[-1]['t']:.4e}")
    print()

    # the identity wants a dense series; the gradient series is the same family
    # on both sides, so use it when it covers the window, else the statistics
    A_dense = grad if (grad and grad[0]["t"] <= LO) else stats
    if a.identity:
        identity(A_dense, ours, a.times, a.dw)
        print()
    if a.lag:
        lag(grad, ours, 1.5e-3, min(HI, grad[-1]["t"]), 2e-6, 3e-4)
        print()
    if a.slopes:
        pairs = [("A grad", grad), ("A CBF", stats), ("ours", ours)]
        slopes(pairs, LO, min(HI, grad[-1]["t"]))
        print()
    if a.overturns:
        overturns([("A", stats), ("ours", ours)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
