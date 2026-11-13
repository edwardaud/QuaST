#!/usr/bin/env python3
"""Section 18: does refining ASPECT's bottom wall from level 5 to level 6 (the
E6 configuration) reduce the *variability* of Nu_bot, and does it move the mean
toward our run?

Reference numbers from section 17.7, over t_nd = [1e-3, 2.16e-3]:

    ASPECT (L5 wall, run_E_long)   sd(Nu_bot) = 4.36 % of its own mean
    ours                           sd(Nu_bot) = 0.37 %
    ASPECT / ours                         Vrms sd only 1.6x

Convention used here for "sd %" (it reproduces those three numbers exactly):
the standard deviation and the mean are both taken over the *native samples* of
the series inside the window -- one sample per time step for ASPECT's
`statistics`, one per 5 steps for our `frame_meta`, one per VTU frame for the
gradient family.  A second column divides by the equal-time (dense, 2000-point)
mean, which weights the window uniformly instead of weighting by 1/dt; the two
differ by <= 1 % here except for Vrms, where ASPECT takes more steps per unit
time exactly when Vrms is high (see the `Vrms` note in section 18).

Both wall-functional families are reported:
  CBF       ASPECT `statistics` column 19/20 (Gresho CBF); ours = point value
            (<= 0.05 % apart, section 17.2)
  gradient  ASPECT VTU exact Q2 wall derivative (`extract_aspect_state.py
            --series`); ours = `frame_meta.csv` `Nu_bot` (same family)

Usage:
  python3 e6_long_compare.py                       # all tables
  python3 e6_long_compare.py --windows 1e-3:2.16e-3,1e-3:3e-3
  python3 e6_long_compare.py --overlap             # determinism of the resume
"""
import argparse
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import long_window as lw

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = lw.REPO

STATS = {
    "L5": os.path.join(HERE, "run_E_long", "statistics"),
    "L6": os.path.join(HERE, "run_E6_long", "statistics"),
}
GRAD = {
    "L5": os.path.join(REPO, "out_julia", "agent", "asp_long_q2_series.csv"),
    "L6": os.path.join(REPO, "out_julia", "agent", "asp_e6long_q2_series.csv"),
}
OURS = os.path.join(REPO, "out_julia", "case_realistic_E", "frames", "frame_meta.csv")
DEF_WIN = "1e-3:2.16e-3,1e-3:3e-3"
OUR_CAD = 1.68e-6        # our frame cadence in t_nd (5 Julia steps x 3.36e-7)
T_CK = 6.388867502739e15 / lw.T0   # the E6 checkpoint run_E6_long resumes from


# ------------------------------------------------------------------ statistics --
def native(rows, key, lo, hi):
    """(n, mean, sd) over the samples that fall inside [lo, hi]."""
    sel = [r[key] for r in rows if lo <= r["t"] <= hi]
    if len(sel) < 3:
        return len(sel), None, None
    return len(sel), st.mean(sel), st.pstdev(sel)


def dense(rows, key, lo, hi, n=2000):
    return lw.dense_mean(rows, key, lo, hi, n)


def frame(rows, key, lo, hi):
    return lw.frame_mean(rows, key, lo, hi)[0]


def pct(a, b):
    return lw.pct(a, b)


def sd_pct(rows, key, lo, hi):
    """(n, mean, sd, sd% of native mean, sd% of dense mean)."""
    n, m, s = native(rows, key, lo, hi)
    if s is None:
        return n, None, None, None, None
    d = dense(rows, key, lo, hi)
    return n, m, s, 100.0 * s / abs(m), (100.0 * s / abs(d) if d else None)


def sd_at_cadence(rows, key, lo, hi, cad):
    """pstdev of samples resampled on an equal-time grid of spacing `cad`."""
    n = int(round((hi - lo) / cad)) + 1
    if n < 5:
        return None, n
    vals = [lw.interp(rows, key, lo + (hi - lo) * i / (n - 1)) for i in range(n)]
    if any(v is None for v in vals):
        return None, n
    return st.pstdev(vals), n


def trend_resid(rows, key, lo, hi, width):
    """Split the in-window variance into a smooth (moving-average) part and the
    residual, on an equal-time grid.  `width` is the averaging window in t_nd."""
    sel = [r for r in rows if lo <= r["t"] <= hi]
    if len(sel) < 10:
        return None
    n = len(sel)
    ts = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    vs = [lw.interp(rows, key, t) for t in ts]
    if any(v is None for v in vs):
        return None
    half = max(1, int(round(0.5 * width / ((hi - lo) / (n - 1)))))
    out = []
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        out.append(sum(vs[a:b]) / (b - a))
    tr = st.pstdev(out)
    rs = st.pstdev([v - w for v, w in zip(vs, out)])
    return st.pstdev(vs), tr, rs


def gap(rows, key, t, other):
    """Instantaneous 100*(A-O)/|O| at t (both series interpolated)."""
    a = lw.interp(rows, key, t)
    b = lw.interp(other, key, t)
    if a is None or b is None:
        return None
    return 100.0 * (a - b) / abs(b)


# ------------------------------------------------------------------------ main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default=DEF_WIN)
    ap.add_argument("--ours-windows", default="1e-3:2.16e-3,1e-3:3e-3,1e-3:5e-3")
    ap.add_argument("--trace", default="1e-3 1.2e-3 1.5e-3 2e-3 2.5e-3 3e-3 3.5e-3 4e-3 5e-3")
    ap.add_argument("--overlap", action="store_true")
    ap.add_argument("--no-trend", action="store_true")
    ap.add_argument("--sampled", nargs=3, type=float, default=None,
                    metavar=("LO", "HI", "N"),
                    help="also print the two ASPECT series and ours at N equally "
                         "spaced times on [LO, HI] (CBF family)")
    a = ap.parse_args()
    wins = lw.parse_windows(a.windows)

    stats = {k: lw.read_aspect_stats([p]) for k, p in STATS.items() if os.path.isfile(p)}
    grad = {k: lw.read_grad_csv([p]) for k, p in GRAD.items() if os.path.isfile(p)}
    ours = lw.read_frames(OURS)

    print("=" * 108)
    print("inputs")
    for k, rows in stats.items():
        if rows:
            cad = st.median([b["t"] - a2["t"] for a2, b in zip(rows, rows[1:])])
            print(f"  ASPECT CBF {k}: {len(rows):5d} steps, t_nd {rows[0]['t']:.4e} .. "
                  f"{rows[-1]['t']:.4e}, median dt_nd {cad:.3e}")
    for k, rows in grad.items():
        if rows:
            cad = st.median([b["t"] - a2["t"] for a2, b in zip(rows, rows[1:])])
            print(f"  ASPECT grad {k}: {len(rows):5d} frames, t_nd {rows[0]['t']:.4e} .. "
                  f"{rows[-1]['t']:.4e}, median dt_nd {cad:.3e}")
    cad = st.median([b["t"] - a2["t"] for a2, b in zip(ours, ours[1:])])
    print(f"  ours       : {len(ours):5d} frames, t_nd {ours[0]['t']:.4e} .. "
          f"{ours[-1]['t']:.4e}, median dt_nd {cad:.3e}")

    if a.overlap:
        e6 = lw.read_aspect_stats([os.path.join(HERE, "run_E6", "statistics")])
        cur = stats.get("L6", [])
        print()
        print("=" * 108)
        print("determinism of the E6 resume: run_E6_long vs run_E6 on their overlap")
        lo = max(cur[0]["t"] if cur else 0, e6[0]["t"] if e6 else 0, T_CK)
        hi = min(cur[-1]["t"] if cur else 0, e6[-1]["t"] if e6 else 0)
        if hi <= lo:
            print("  *** no overlap yet ***")
        else:
            print(f"  overlap t_nd = {lo:.6e} .. {hi:.6e}")
            for k in ("nu_bot", "nu_top", "vrms", "tmax"):
                ma, mb = dense(cur, k, lo, hi), dense(e6, k, lo, hi)
                print(f"  {k:8s} run_E6_long = {ma:12.6f}   run_E6 = {mb:12.6f}   "
                      f"diff = {mb - ma:+.3e} ({pct(mb, ma):+.5f} %)")
            m = {r["step"]: r for r in cur}
            pairs = [(r["step"], abs(m[r["step"]]["nu_bot"] - r["nu_bot"]))
                     for r in e6 if r["step"] in m and lo <= r["t"] <= hi]
            if pairs:
                bad = [(s, d) for s, d in pairs if d > 1e-6]
                print(f"  row by row: {len(pairs)} steps, max abs(dNu_bot) = "
                      f"{max(d for _, d in pairs):.3e}; steps differing by > 1e-6: "
                      f"{len(bad)}" + (f" {bad[:3]}" if bad else ""))
                print(f"  max abs(dNu_top) = "
                      f"{max(abs(m[r['step']]['nu_top'] - r['nu_top']) for r in e6 if r['step'] in m and lo <= r['t'] <= hi):.3e}")

    # ---------------------------------------------------------------- table 1 --
    print()
    print("=" * 108)
    print("Table 18.A  level of the bottom wall vs Nu_bot / Nu_top / Vrms variability")
    print("  sd% nat     = sd / mean over the native samples (the section 17.7 convention)")
    print("  sd% dense   = sd / equal-time window mean")
    print("  sd% common  = sd of the series resampled at OUR frame cadence, / dense mean")
    print("                (makes ASPECT's per-step series comparable with ours)")
    hdr = (f"{'window':>17} {'run':>4} {'key':>7} {'n':>5} {'mean(nat)':>10} "
           f"{'mean(dense)':>11} {'sd':>8} {'sd% nat':>8} {'sd% dense':>9} "
           f"{'sd(common)':>10} {'sd% com':>8}")
    print(hdr)
    print("-" * len(hdr))

    def f(x, w=8, p=4):
        return " " * w if x is None else f"{x:{w}.{p}f}"

    for lo, hi in wins:
        for key in ("nu_bot", "nu_top", "vrms"):
            for tag in ("L5", "L6"):
                rows = stats.get(tag)
                if not rows:
                    continue
                n, m, s, rn, rd = sd_pct(rows, key, lo, hi)
                d = dense(rows, key, lo, hi)
                sc, nc = sd_at_cadence(rows, key, lo, hi, OUR_CAD)
                cover = "  [ends %.2e]" % rows[-1]["t"] if hi > rows[-1]["t"] else ""
                print(f"[{lo:.2e},{hi:.2e}] {tag:>4} {key:>7} {n:5d} {f(m,10)} "
                      f"{f(d,11)} {f(s)} {f(rn,8,2)} {f(rd,9,2)} {f(sc,10)} "
                      f"{f(100.0 * sc / abs(d) if (sc and d) else None,8,2)}" + cover)
            n, m, s, rn, rd = sd_pct(ours, key, lo, hi)
            d = dense(ours, key, lo, hi)
            sc, _ = sd_at_cadence(ours, key, lo, hi, OUR_CAD)
            print(f"[{lo:.2e},{hi:.2e}] {'O':>4} {key:>7} {n:5d} {f(m,10)} "
                  f"{f(d,11)} {f(s)} {f(rn,8,2)} {f(rd,9,2)} {f(sc,10)} "
                  f"{f(100.0 * sc / abs(d) if (sc and d) else None,8,2)}")

    # same for the gradient family (frame cadence)
    print()
    hdr = (f"{'window':>17} {'run':>4} {'key':>7} {'n':>5} {'mean':>10} {'sd':>8} "
           f"{'sd%':>7}   (gradient family, frame cadence)")
    print(hdr)
    print("-" * len(hdr))
    for lo, hi in wins:
        for key in ("nu_bot", "nu_top", "vrms"):
            for tag in ("L5", "L6"):
                rows = grad.get(tag)
                if not rows:
                    continue
                n, m, s = native(rows, key, lo, hi)
                cover = "  [ends %.2e]" % rows[-1]["t"] if hi > rows[-1]["t"] else ""
                print(f"[{lo:.2e},{hi:.2e}] {tag:>4} {key:>7} {n:5d} {f(m,10)} {f(s)} "
                      f"{f(100.0 * s / abs(m) if s else None,7,2)}" + cover)
            n, m, s = native(ours, key, lo, hi)
            print(f"[{lo:.2e},{hi:.2e}] {'O':>4} {key:>7} {n:5d} {f(m,10)} {f(s)} "
                  f"{f(100.0 * s / abs(m) if s else None,7,2)}")

    # ---------------------------------------------------------------- table 2 --
    print()
    print("=" * 108)
    print("Table 18.B  window means and the cross-code gap (both families)")
    hdr = (f"{'window':>17} | {'A_L5 cbf':>9} {'A_L6 cbf':>9} {'A_L5 gr':>9} {'A_L6 gr':>9} "
           f"{'O':>9} | {'gapL5 cbf':>9} {'gapL6 cbf':>9} {'gapL5 gr':>9} {'gapL6 gr':>9}")
    print(hdr)
    print("-" * len(hdr))
    for lo, hi in wins:
        rows5, rows6 = stats.get("L5"), stats.get("L6")
        g5, g6 = grad.get("L5"), grad.get("L6")
        o = dense(ours, "nu_bot", lo, hi)
        ab5 = dense(rows5, "nu_bot", lo, hi) if rows5 else None
        ab6 = dense(rows6, "nu_bot", lo, hi) if rows6 else None
        gb5 = frame(g5, "nu_bot", lo, hi) if g5 else None
        gb6 = frame(g6, "nu_bot", lo, hi) if g6 else None
        print(f"[{lo:.2e},{hi:.2e}] | {f(ab5,9)} {f(ab6,9)} {f(gb5,9)} {f(gb6,9)} {f(o,9)} | "
              f"{f(pct(ab5,o),9,2)} {f(pct(ab6,o),9,2)} {f(pct(gb5,o),9,2)} {f(pct(gb6,o),9,2)}")

    # Nu_top / Vrms means, the "total transport" check
    print()
    print("Table 18.B2  Nu_top and Vrms window means (gradient family for A)")
    print(f"{'window':>17} | {'Nu_top L5':>9} {'Nu_top L6':>9} {'Nu_top O':>9} | "
          f"{'Vrms L5':>8} {'Vrms L6':>8} {'Vrms O':>8} | {'gapNT L5':>8} {'gapNT L6':>8}")
    for lo, hi in wins:
        rows5, rows6 = stats.get("L5"), stats.get("L6")
        g5, g6 = grad.get("L5"), grad.get("L6")
        n5 = frame(g5, "nu_top", lo, hi) if g5 else dense(rows5, "nu_top", lo, hi, )
        n6 = frame(g6, "nu_top", lo, hi) if g6 else (dense(rows6, "nu_top", lo, hi) if rows6 else None)
        no = dense(ours, "nu_top", lo, hi)
        v5 = dense(rows5, "vrms", lo, hi) if rows5 else None
        v6 = dense(rows6, "vrms", lo, hi) if rows6 else None
        vo = dense(ours, "vrms", lo, hi)
        print(f"[{lo:.2e},{hi:.2e}] | {f(n5,9)} {f(n6,9)} {f(no,9)} | "
              f"{f(v5,8,1)} {f(v6,8,1)} {f(vo,8,1)} | {f(pct(n5,no),8,2)} {f(pct(n6,no),8,2)}")

    # ---------------------------------------------------------------- table 3 --
    if not a.no_trend:
        print()
        print("=" * 108)
        print("Table 18.C  variance decomposition of the CBF Nu_bot series: "
              "smooth trend vs residual")
        print("  trend = centred moving average of width 1e-4 t_nd (8 % of the "
              "[1e-3,2.16e-3] window) and 3e-4 (robustness)")
        print(f"{'window':>17} {'run':>4} | {'sd tot':>8} {'sd tr(1e-4)':>11} "
              f"{'sd res(1e-4)':>12} | {'sd tr(3e-4)':>11} {'sd res(3e-4)':>12} | "
              f"{'sd% tot':>8} {'res%':>6}")
        for lo, hi in wins:
            for tag in ("L5", "L6", "O"):
                rows = stats.get(tag) if tag != "O" else ours
                if not rows:
                    continue
                tr1 = trend_resid(rows, "nu_bot", lo, hi, 1e-4 * (hi - lo) / 1.16e-3)
                tr3 = trend_resid(rows, "nu_bot", lo, hi, 3e-4 * (hi - lo) / 1.16e-3)
                if tr1 is None or tr3 is None:
                    continue
                tot, t1, r1 = tr1
                _, t3, r3 = tr3
                m = native(rows, "nu_bot", lo, hi)[1]
                print(f"[{lo:.2e},{hi:.2e}] {tag:>4} | {f(tot)} {f(t1,11)} {f(r1,12)} | "
                      f"{f(t3,11)} {f(r3,12)} | {f(100*tot/abs(m),8,2)} "
                      f"{f(100*r1/abs(m),6,2)}")

    # ---------------------------------------------------------------- table 4 --
    print()
    print("=" * 108)
    print("Table 18.D  instantaneous Nu_bot gap 100*(A-O)/|O|, CBF family")
    print(f"{'t_nd':>9} | {'A_L5':>9} {'A_L6':>9} {'O':>9} | {'gap L5':>8} {'gap L6':>8} | "
          f"{'d(gap)':>8}")
    for t in [float(x) for x in a.trace.split()]:
        o = lw.interp(ours, "nu_bot", t)
        a5 = lw.interp(stats["L5"], "nu_bot", t) if stats.get("L5") else None
        a6 = lw.interp(stats["L6"], "nu_bot", t) if stats.get("L6") else None
        g5 = pct(a5, o)
        g6 = pct(a6, o)
        print(f"{t:9.2e} | {f(a5,9)} {f(a6,9)} {f(o,9)} | "
              f"{'   -  ' if g5 != g5 else format(g5,'+8.2f')} "
              f"{'   -  ' if g6 != g6 else format(g6,'+8.2f')} | "
              f"{'   -  ' if (g5 != g5 or g6 != g6) else format(g6-g5,'+8.2f')}")

    # running cumulative mean gap (section 17.5 style)
    print()
    print("Table 18.E  running cumulative mean over [1e-3, t_hi], CBF family")
    print(f"{'t_hi':>9} | {'A_L5':>9} {'A_L6':>9} {'O':>9} | {'gap L5':>8} {'gap L6':>8} | "
          f"{'d(gap)':>8}")
    for t in [float(x) for x in a.trace.split()]:
        if t <= 1e-3:
            continue
        o = dense(ours, "nu_bot", 1e-3, t)
        a5 = dense(stats["L5"], "nu_bot", 1e-3, t) if stats.get("L5") else None
        a6 = dense(stats["L6"], "nu_bot", 1e-3, t) if stats.get("L6") else None
        g5, g6 = pct(a5, o), pct(a6, o)
        print(f"{t:9.2e} | {f(a5,9)} {f(a6,9)} {f(o,9)} | "
              f"{'   -  ' if g6 != g6 else format(g5,'+8.2f')} "
              f"{'   -  ' if g6 != g6 else format(g6,'+8.2f')} | "
              f"{'   -  ' if g6 != g6 else format(g6-g5,'+8.2f')}")

    # ---------------------------------------------------------------- table 6 --
    if a.sampled:
        lo, hi, n = a.sampled
        n = int(n)
        print()
        print("=" * 108)
        print(f"Table 18.H  the two ASPECT trajectories, sampled on [{lo:.3e},{hi:.3e}] "
              f"(CBF family)")
        print(f"{'t_nd':>10} | {'A_L5 Nu_bot':>11} {'A_L6 Nu_bot':>11} {'O Nu_bot':>9} | "
              f"{'L6-L5':>8} | {'NT L5':>7} {'NT L6':>7} | {'Vrms L5':>8} {'Vrms L6':>8}")
        for i in range(n):
            t = lo + (hi - lo) * i / max(1, n - 1)
            v5 = lw.interp(stats["L5"], "nu_bot", t) if stats.get("L5") else None
            v6 = lw.interp(stats["L6"], "nu_bot", t) if stats.get("L6") else None
            vo = lw.interp(ours, "nu_bot", t)
            n5 = lw.interp(stats["L5"], "nu_top", t) if stats.get("L5") else None
            n6 = lw.interp(stats["L6"], "nu_top", t) if stats.get("L6") else None
            r5 = lw.interp(stats["L5"], "vrms", t) if stats.get("L5") else None
            r6 = lw.interp(stats["L6"], "vrms", t) if stats.get("L6") else None
            print(f"{t:10.4e} | {f(v5,11)} {f(v6,11)} {f(vo,9)} | "
                  f"{'   -  ' if (v5 is None or v6 is None) else format(v6-v5,'+8.4f')} | "
                  f"{f(n5,7,3)} {f(n6,7,3)} | {f(r5,8,0)} {f(r6,8,0)}")

    # ---------------------------------------------------------------- table 7 --
    if stats.get("L6") and grad.get("L6"):
        print()
        print("=" * 108)
        print("Table 18.F  what fraction of the L5->ours distance does the wall "
              "refinement cover?")
        print("  frac mean = (m6 - m5) / (m_O - m5);  frac sd = (sd5 - sd6) / (sd5 - sdO)")
        for lo, hi in wins:
            if hi > stats["L6"][-1]["t"]:
                print(f"[{lo:.2e},{hi:.2e}] : L6 series ends at "
                      f"{stats['L6'][-1]['t']:.3e} -- not covered")
                continue
            m5 = dense(stats["L5"], "nu_bot", lo, hi)
            m6 = dense(stats["L6"], "nu_bot", lo, hi)
            mo = dense(ours, "nu_bot", lo, hi)
            g5, g6 = pct(m5, mo), pct(m6, mo)
            s5 = sd_pct(stats["L5"], "nu_bot", lo, hi)[2]
            s6 = sd_pct(stats["L6"], "nu_bot", lo, hi)[2]
            so = sd_pct(ours, "nu_bot", lo, hi)[2]
            print(f"[{lo:.2e},{hi:.2e}] mean gap CBF: L5 {g5:+.2f}% -> L6 {g6:+.2f}%  "
                  f"({g5 - g6:+.2f} pp of the gap closed); "
                  f"frac mean {(m6 - m5) / (mo - m5) * 100 if mo != m5 else float('nan'):+.1f}%")
            print(f"                 sd: L5 {s5:.4f} -> L6 {s6:.4f} (ours {so:.4f}); "
                  f"frac sd {(s5 - s6) / (s5 - so) * 100 if s5 != so else float('nan'):+.1f}%")
        # instantaneous swing of the gap on [1e-3, 3e-3]
        lo, hi = 1e-3, min(3e-3, stats["L6"][-1]["t"])
        if hi > lo:
            gs5, gs6, ts = [], [], [lo + (hi - lo) * i / 400 for i in range(401)]
            for t in ts:
                g5 = gap(stats["L5"], "nu_bot", t, ours)
                g6 = gap(stats["L6"], "nu_bot", t, ours)
                if g5 is not None and g6 is not None:
                    gs5.append(g5)
                    gs6.append(g6)
            print(f"instantaneous gap over [{lo:.2e},{hi:.2e}]: "
                  f"L5 {min(gs5):+.2f} .. {max(gs5):+.2f} (swing {max(gs5)-min(gs5):.2f} pp); "
                  f"L6 {min(gs6):+.2f} .. {max(gs6):+.2f} (swing {max(gs6)-min(gs6):.2f} pp)")
            print(f"  swing reduction: {(max(gs5)-min(gs5)) - (max(gs6)-min(gs6)):.2f} pp "
                  f"({((max(gs5)-min(gs5)) - (max(gs6)-min(gs6))) / (max(gs5)-min(gs5)) * 100:+.1f} %)")
            print(f"  mean |gap| over the window: L5 {st.mean([abs(g) for g in gs5]):.2f} pp, "
                  f"L6 {st.mean([abs(g) for g in gs6]):.2f} pp")

        # the L6 - L5 difference series itself: smooth shift or noise reduction?
        print()
        print("Table 18.G  the difference series Nu_bot(L6) - Nu_bot(L5) on "
              "[1e-3, t_end] (CBF family)")
        for lo, hi in wins:
            if hi > stats["L6"][-1]["t"]:
                continue
            d5 = [r["t"] for r in stats["L5"] if lo <= r["t"] <= hi]
            diffs, tgrid = [], [lo + (hi - lo) * i / 800 for i in range(801)]
            for t in tgrid:
                v5 = lw.interp(stats["L5"], "nu_bot", t)
                v6 = lw.interp(stats["L6"], "nu_bot", t)
                if v5 is None or v6 is None:
                    continue
                diffs.append(v6 - v5)
            if not diffs:
                continue
            m5d = dense(stats["L5"], "nu_bot", lo, hi)
            m = st.mean(diffs)
            s = st.pstdev(diffs)
            print(f"[{lo:.2e},{hi:.2e}] mean(L6-L5) = {m:+.4f} "
                  f"({100.0 * m / abs(m5d):+.2f} % of Nu_bot,L5), "
                  f"sd = {s:.4f} ({100.0 * s / abs(m5d):.2f} %), "
                  f"min {min(diffs):+.4f} max {max(diffs):+.4f}")
            # is the difference smooth in time? correlation with its own 1e-4 trend
            n = len(diffs)
            half = max(1, int(round(0.5 * 1e-4 / ((hi - lo) / (n - 1)))))
            tr = [sum(diffs[max(0, i-half):min(n, i+half+1)]) /
                  (min(n, i+half+1) - max(0, i-half)) for i in range(n)]
            print(f"          sd of its 1e-4 moving average = {st.pstdev(tr):.4f}, "
                  f"sd of the residual = {st.pstdev([b-a2 for a2,b in zip(tr,diffs)]):.4f} "
                  f"(residual/trend = "
                  f"{st.pstdev([b-a2 for a2,b in zip(tr,diffs)])/st.pstdev(tr):.2f})")
    # ---------------------------------------------------------------- table 8 --
    lo, hi = 1e-3, min(3e-3, stats["L6"][-1]["t"]) if stats.get("L6") else 1e-3
    if stats.get("L6") and hi > lo:
        print()
        print("=" * 108)
        print("Table 18.I  is the L5->L6 change a phase shift or a structural change?")
        n = 1001
        ts = [lo + (hi - lo) * i / (n - 1) for i in range(n)]

        def ser(rows, key):
            return [lw.interp(rows, key, t) for t in ts]

        def corr(a, b):
            ma, mb = st.mean(a), st.mean(b)
            sa = (sum((x - ma) ** 2 for x in a)) ** 0.5
            sb = (sum((x - mb) ** 2 for x in b)) ** 0.5
            return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)

        for key in ("nu_bot", "nu_top"):
            a5 = ser(stats["L5"], key)
            a6 = ser(stats["L6"], key)
            print(f"  {key}: zero-lag r(L5,L6) = {corr(a5, a6):.4f}")
            best = (None, -2.0)
            for k in range(-150, 151, 5):
                if k == 0:
                    continue
                sh = int(round(k * 2e-6 / ((hi - lo) / (n - 1))))
                if abs(sh) >= n // 3:
                    continue
                if sh > 0:
                    r = corr(a5[sh:], a6[:-sh])
                else:
                    r = corr(a5[:sh], a6[-sh:])
                if r > best[1]:
                    best = (k * 2e-6, r)
            print(f"           best lag {best[0]:+.2e} t_nd with r = {best[1]:.4f}")

        # top-bottom coupling slope, the section 17.6 statistic
        for tag, rows in (("L5", stats["L5"]), ("L6", stats["L6"]), ("O", ours)):
            x = ser(rows, "nu_top")
            y = ser(rows, "nu_bot")
            mx, my = st.mean(x), st.mean(y)
            sxx = sum((v - mx) ** 2 for v in x)
            sxy = sum((v - mx) * (w - my) for v, w in zip(x, y))
            slope = sxy / sxx
            r = corr(x, y)
            print(f"  dNu_bot/dNu_top on [{lo:.1e},{hi:.1e}]: {tag:2s} slope {slope:+.4f}  "
                  f"r = {r:+.3f}  sd(Nu_bot) = {st.pstdev(y):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
