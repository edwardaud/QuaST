#!/usr/bin/env python3
"""Section 16: the same-state experiment (import ASPECT's state, run both codes).

Reads
  * ASPECT's continuation diagnostics (`run_E_cont/statistics`, CBF family), and
    optionally the Q2-exact gradient-family series produced by
    `extract_aspect_state.py --series` (the same wall functional as QuaST),
  * our run started from ASPECT's state (`case_realistic_E_aspimport`),
  * our natural from-t=0 trajectory (`case_realistic_E`, for reference),
converts everything to t_nd / Nu / Vrms_nd with the section-2 scale set, and
prints

  1. the common-state values (the interpolation check),
  2. the trajectory table at N matched t_nd,
  3. the relative separation s(t) = (ours - asp)/|asp| per quantity, the time to
     reach 1 / 2 / 5 % and the exponential growth rate (Lyapunov-like) of the
     separation over the pre-saturation stretch,
  4. the same for our *natural* run against ASPECT, i.e. the -6 % window gap that
     the experiment is about: how large is it at t0 (the common state), when does
     it reach its window mean.

Usage
  python3 same_state_compare.py                       # defaults below
  python3 same_state_compare.py --t0 9.5862e-4 --n 14
"""
import argparse
import csv
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

# scale set, docs/dev/VALIDATION.md section 2.1
D = 2.9e6
KAPPA = 1.0e-6
DT = 1000.0
LX = 2.5 * D
K_TH = 3.3
NORM = LX * K_TH * DT / D          # 8250 W per unit Nu
T0 = D * D / KAPPA                 # 8.41e18 s


def read_aspect(path):
    """ASPECT `statistics` -> list of dicts, columns found by header."""
    names, rows = {}, []
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                m = re.match(r"#\s*(\d+):\s*(.*)", line)
                if m:
                    names[int(m.group(1))] = m.group(2).strip()
                continue
            if line:
                rows.append(line.split())

    def col(tag):
        for i, name in names.items():
            if name.startswith(tag):
                return i - 1
        raise KeyError(tag)

    c = dict(step=col("Time step number"), t=col("Time (seconds)"),
             dt=col("Time step size"), cells=col("Number of mesh cells"),
             vrms=col("RMS velocity"), vmax=col("Max. velocity"),
             tmax=col("Maximal temperature"), tmean=col("Average temperature"),
             qtop=col('Outward heat flux through boundary with indicator 3'),
             qbot=col('Outward heat flux through boundary with indicator 2'))
    out = []
    for v in rows:
        try:
            out.append(dict(
                step=int(v[c["step"]]), t_nd=float(v[c["t"]]) / T0,
                dt_nd=float(v[c["dt"]]) / T0, cells=float(v[c["cells"]]),
                vrms=float(v[c["vrms"]]) * D / KAPPA,
                vmax=float(v[c["vmax"]]) * D / KAPPA,
                tmax=float(v[c["tmax"]]) / DT - 0.5,
                tmean=float(v[c["tmean"]]) / DT - 0.5,
                nu_top=float(v[c["qtop"]]) / NORM,
                nu_bot=-float(v[c["qbot"]]) / NORM))
        except (ValueError, IndexError):
            continue
    return out


def read_our_frames(path):
    """Our frame_meta.csv; `t` is already t_nd."""
    out = []
    with open(path, errors="replace") as f:
        header = f.readline().strip().split(",")
        for line in f:
            p = line.strip().split(",")
            if len(p) != len(header):
                continue
            r = dict(zip(header, p))
            try:
                out.append(dict(step=int(r["step"]), t_nd=float(r["t"]),
                                dt_nd=float(r["dt"]),
                                nu_top=float(r["Nu_top"]),
                                nu_bot=float(r["Nu_bot"]),
                                vrms=float(r["Vrms"]),
                                tmax=float(r["Tmax"]),
                                tmean=float(r["Tmean"]),
                                leaves=int(r["leaves"])))
            except (ValueError, KeyError):
                continue
    return out


def read_series_csv(path):
    """The `--series` output of extract_aspect_state.py (gradient family)."""
    out = []
    with open(path, errors="replace") as f:
        for r in csv.DictReader(f):
            out.append(dict(frame=int(r["frame"]), t_nd=float(r["t_nd"]),
                            nu_bot=float(r["nu_bot"]), nu_top=float(r["nu_top"]),
                            vrms=float(r["Vrms_nd"]), tmax=float(r["Tmax_nd"]),
                            tmean=float(r["Tmean_nd"]),
                            cells=int(r["ncells"])))
    return out


def interp(rows, key, tgrid, field):
    t = np.array([r[key] for r in rows])
    v = np.array([r[field] for r in rows])
    o = np.argsort(t)
    return np.interp(tgrid, t[o], v[o])


def report_separation(t, s, label, t0, vmax0, min_span=3.0e-5):
    """Times to reach |s| = 1/2/5 % and the exponential rate of the *growth*.

    `s` is the relative separation (ours - asp)/|asp|.  `s(t0)` is the offset at
    the common state (the interpolation/family floor); the dynamics is the growth
    of `s - s(t0)`, which is what the growth fit measures.
    """
    s0 = np.interp(t0, t, s)
    print(f"\n  {label}: s(t0) = {100*s0:+.4f} %")
    for thr in (0.01, 0.02, 0.05, 0.10):
        hit = None
        for i in range(len(t)):
            if t[i] < t0:
                continue
            if abs(s[i]) >= thr:
                hit = t[i]
                break
        if hit is None:
            print(f"    |s| = {100*thr:4.1f} %: not reached by t_nd = {t[-1]:.4e} "
                  f"(max |s| = {100*np.abs(s[t >= t0]).max():.2f} %)")
        else:
            dt = hit - t0
            print(f"    |s| = {100*thr:4.1f} %: t_nd = {hit:.4e}  "
                  f"(+{dt:.3e} = {dt/vmax0:.2f} box-crossing times)")
    m = (t > t0) & (np.abs(s - s0) > 1e-4) & (np.abs(s) < 0.2)
    span = t[-1] - t0
    if m.sum() >= 5 and span >= min_span:
        A = np.polyfit(t[m] - t0, np.log(np.abs(s[m] - s0)), 1)
        lam = A[0]
        if lam > 0:
            print(f"    growth fit over {int(m.sum())} samples (|s|<20 %): "
                  f"d ln|s-s0|/dt = {lam:.3e}  ->  e-fold time {1/lam:.3e} t_nd = "
                  f"{1/lam/vmax0:.2f} box-crossing times")
            print(f"    from the measured s(t0) = {100*s0:+.4f} % to |s| = 6 %: "
                  f"{(np.log(0.06/abs(s0)))/lam:.3e} t_nd")
        else:
            print(f"    growth fit: not exponential (d ln|s-s0|/dt = {lam:.2e} <= 0, "
                  "the separation is non-monotone); use the linear slope below")
    else:
        print(f"    growth fit: needs >= 5 samples and a {min_span:.1e} span "
              f"(have {m.sum()} samples, span {span:.2e}); use --n to add samples")
    return s0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asp-stats", default=os.path.join(
        HERE, "run_E_cont", "statistics"))
    ap.add_argument("--asp-series", default=os.path.join(
        REPO, "out_julia", "case_realistic_E_aspimport", "asp_series.csv"))
    ap.add_argument("--ours", default=os.path.join(
        REPO, "out_julia", "case_realistic_E_aspimport", "frames", "frame_meta.csv"))
    ap.add_argument("--natural", default=os.path.join(
        REPO, "out_julia", "case_realistic_E", "frames", "frame_meta.csv"))
    ap.add_argument("--t0", type=float, default=None,
                    help="common state; default: first frame of the imported run")
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--series-only", action="store_true")
    ap.add_argument("--markdown", action="store_true",
                    help="print the two trajectory tables as markdown")
    a = ap.parse_args()

    asp = read_aspect(a.asp_stats)
    ours = read_our_frames(a.ours)
    nat = read_our_frames(a.natural) if os.path.isfile(a.natural) else []
    ser = read_series_csv(a.asp_series) if os.path.isfile(a.asp_series) else []

    t0 = a.t0 if a.t0 is not None else ours[0]["t_nd"]
    t_end = min(ours[-1]["t_nd"], asp[-1]["t_nd"])
    print(f"common state t0 = {t0:.6e} (t = {t0*T0:.4e} s)")
    print(f"ASPECT:  {len(asp)} steps, t_nd {asp[0]['t_nd']:.4e} .. {asp[-1]['t_nd']:.4e}")
    if ser:
        print(f"ASPECT Q2 series: {len(ser)} frames, t_nd {ser[0]['t_nd']:.4e} .. {ser[-1]['t_nd']:.4e}")
    print(f"ours:    {len(ours)} frames, t_nd {ours[0]['t_nd']:.4e} .. {ours[-1]['t_nd']:.4e}")
    print(f"common window: {t0:.4e} .. {t_end:.4e}  (span {t_end - t0:.3e})")

    tgrid = np.linspace(t0, t_end, a.n)
    # exact ASPECT values at t0 (nearest step/frame) for the interpolation check
    i0 = int(np.argmin([abs(r["t_nd"] - t0) for r in asp]))
    print(f"\n--- common state (ASPECT step {asp[i0]['step']}, "
          f"t_nd = {asp[i0]['t_nd']:.6e}) ---")
    print(f"  ASPECT statistics (CBF): Nu_top = {asp[i0]['nu_top']:+.6f}  "
          f"Nu_bot = {asp[i0]['nu_bot']:+.6f}  Vrms = {asp[i0]['vrms']:.2f}  "
          f"Tmax = {asp[i0]['tmax']:.5f}  vmax = {asp[i0]['vmax']:.1f}  "
          f"cells = {asp[i0]['cells']:.0f}")
    if ser:
        j0 = int(np.argmin([abs(r["t_nd"] - t0) for r in ser]))
        print(f"  ASPECT Q2 gradient family: Nu_top = {ser[j0]['nu_top']:+.6f}  "
              f"Nu_bot = {ser[j0]['nu_bot']:+.6f}  Vrms = {ser[j0]['vrms']:.2f}  "
              f"Tmax = {ser[j0]['tmax']:.5f}  (frame {ser[j0]['frame']})")
    o0 = ours[0]
    print(f"  ours (imported state, step {o0['step']}): Nu_top = {o0['nu_top']:+.6f}  "
          f"Nu_bot = {o0['nu_bot']:+.6f}  Vrms = {o0['vrms']:.2f}  Tmax = {o0['tmax']:.5f}")
    print("  (the import script prints the same numbers *before* the first step; "
          "the row above is after one dt = %.2e)" % o0["dt_nd"])

    # ---- trajectory table --------------------------------------------------
    hdr = (f"\n{'t_nd':>11} | {'Nu_bot A':>9} {'Nu_bot o':>9} {'s%':>7} | "
           f"{'Nu_top A':>8} {'Nu_top o':>8} {'s%':>6} | "
           f"{'Vrms A':>7} {'Vrms o':>7} {'s%':>6} | {'Nu_bot nat':>10} {'s%':>6}")
    print(hdr)
    print("-" * len(hdr))
    nu_b_asp = interp(asp, "t_nd", tgrid, "nu_bot")
    nu_t_asp = interp(asp, "t_nd", tgrid, "nu_top")
    vr_asp = interp(asp, "t_nd", tgrid, "vrms")
    nu_b_o = interp(ours, "t_nd", tgrid, "nu_bot")
    nu_t_o = interp(ours, "t_nd", tgrid, "nu_top")
    vr_o = interp(ours, "t_nd", tgrid, "vrms")
    for i in range(a.n):
        line = (f"{tgrid[i]:11.4e} | {nu_b_asp[i]:9.5f} {nu_b_o[i]:9.5f} "
                f"{100*(nu_b_o[i]-nu_b_asp[i])/abs(nu_b_asp[i]):7.2f} | "
                f"{nu_t_asp[i]:8.5f} {nu_t_o[i]:8.5f} "
                f"{100*(nu_t_o[i]-nu_t_asp[i])/abs(nu_t_asp[i]):6.2f} | "
                f"{vr_asp[i]:7.1f} {vr_o[i]:7.1f} "
                f"{100*(vr_o[i]-vr_asp[i])/abs(vr_asp[i]):6.2f}")
        if nat and tgrid[i] <= nat[-1]["t_nd"]:
            nb = float(np.interp(tgrid[i], [r["t_nd"] for r in nat],
                                 [r["nu_bot"] for r in nat]))
            line += f" | {nb:10.5f} {100*(nb-nu_b_asp[i])/abs(nu_b_asp[i]):6.2f}"
        print(line)

    # ---- separations -------------------------------------------------------
    if a.markdown:
        def pct(x, y):
            return 100 * (x - y) / abs(y)
        print("\n<!-- table A: CBF family + natural run -->")
        print("| `t_nd` | `Nu_bot` A | `Nu_bot` o | `s%` | `Nu_top` A | `Nu_top` o | "
              "`s%` | `Vrms` A | `Vrms` o | `s%` | `Nu_bot` nat | `s%` |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i in range(a.n):
            nb = np.nan
            s_nat = np.nan
            if nat and tgrid[i] <= nat[-1]["t_nd"]:
                nb = float(np.interp(tgrid[i], [r["t_nd"] for r in nat],
                                     [r["nu_bot"] for r in nat]))
                s_nat = pct(nb, nu_b_asp[i])
            print(f"| {tgrid[i]:.4e} | {nu_b_asp[i]:.5f} | {nu_b_o[i]:.5f} | "
                  f"{pct(nu_b_o[i], nu_b_asp[i]):+.2f} | {nu_t_asp[i]:.5f} | "
                  f"{nu_t_o[i]:.5f} | {pct(nu_t_o[i], nu_t_asp[i]):+.2f} | "
                  f"{vr_asp[i]:.1f} | {vr_o[i]:.1f} | {pct(vr_o[i], vr_asp[i]):+.2f} | "
                  f"{nb:.5f} | {s_nat:+.2f} |")
        if ser:
            print("\n<!-- table B: same wall-functional family (Q2 gradient) -->")
            print("| `t_nd` | `Nu_bot` A (grad) | `Nu_bot` o | `s%` | "
                  "`Nu_top` A (grad) | `Nu_top` o | `s%` |")
            print("|---|---|---|---|---|---|---|")
            gb = interp(ser, "t_nd", tgrid, "nu_bot")
            gt = interp(ser, "t_nd", tgrid, "nu_top")
            for i in range(a.n):
                print(f"| {tgrid[i]:.4e} | {gb[i]:.5f} | {nu_b_o[i]:.5f} | "
                      f"{pct(nu_b_o[i], gb[i]):+.3f} | {gt[i]:.5f} | {nu_t_o[i]:.5f} | "
                      f"{pct(nu_t_o[i], gt[i]):+.3f} |")
        return

    vmax0 = float(np.interp(t0, [r["t_nd"] for r in asp],
                            [r["vmax"] for r in asp]))
    print(f"\n  box-crossing time at the common state: 1/vmax = {1/vmax0:.3e} t_nd "
          f"(vmax = {vmax0:.0f})")
    s_nub = (nu_b_o - nu_b_asp) / np.abs(nu_b_asp)
    s_nut = (nu_t_o - nu_t_asp) / np.abs(nu_t_asp)
    s_vr = (vr_o - vr_asp) / np.abs(vr_asp)
    report_separation(tgrid, s_nub, "Nu_bot (CBF family both sides)", t0, vmax0)
    report_separation(tgrid, s_nut, "Nu_top (CBF family both sides)", t0, vmax0)
    report_separation(tgrid, s_vr, "Vrms", t0, vmax0)

    if ser:
        s_nub_g = ((nu_b_o - interp(ser, "t_nd", tgrid, "nu_bot")) /
                   np.abs(interp(ser, "t_nd", tgrid, "nu_bot")))
        print("\n  --- same wall-functional family (ASPECT Q2 gradient vs our "
              "pointwise) ---")
        report_separation(tgrid, s_nub_g, "Nu_bot (gradient family both sides)",
                          t0, vmax0)

    # ---- measured drift rates and their extrapolation to the full window ----
    print("\n  --- drift rate of the separation (linear fit over the common "
          "span, % per 1e-4 t_nd) ---")
    for name, s in (("Nu_bot CBF", s_nub), ("Nu_top CBF", s_nut),
                    ("Vrms", s_vr)):
        A = np.polyfit(tgrid - t0, 100 * s, 1)
        print(f"    {name:12s}: slope {A[0]*1e-4:+7.3f} %/1e-4   "
              f"over the full 1e-3 window: {A[0]*1e-3:+.2f} %")
    if ser:
        A = np.polyfit(tgrid - t0, 100 * s_nub_g, 1)
        print(f"    {'Nu_bot grad':12s}: slope {A[0]*1e-4:+7.3f} %/1e-4   "
              f"over the full 1e-3 window: {A[0]*1e-3:+.2f} %")

    if nat:
        print("\n  --- reference: our natural from-t=0 run against ASPECT ---")
        m = (tgrid >= nat[0]["t_nd"]) & (tgrid <= nat[-1]["t_nd"])
        if m.any():
            nb = np.interp(tgrid[m], [r["t_nd"] for r in nat],
                           [r["nu_bot"] for r in nat])
            s = (nb - nu_b_asp[m]) / np.abs(nu_b_asp[m])
            print(f"    Nu_bot separation over the same window: "
                  f"{100*s.min():+.2f} % .. {100*s.max():+.2f} %")
            print(f"    ASPECT Nu_bot over the window: {nu_b_asp[m].min():.5f} .. "
                  f"{nu_b_asp[m].max():.5f}; ours (natural): {nb.min():.5f} .. "
                  f"{nb.max():.5f}")

    # window means over the common window (the -6 % quantity)
    print("\n  --- window means over the common span ---")
    for name, aa, oo in (("Nu_bot", nu_b_asp, nu_b_o), ("Nu_top", nu_t_asp, nu_t_o),
                         ("Vrms", vr_asp, vr_o)):
        print(f"    {name:7s}: ASPECT {aa.mean():10.5f}  ours(imported) "
              f"{oo.mean():10.5f}  diff {100*(oo.mean()-aa.mean())/abs(aa.mean()):+7.2f} %")
    if ser:
        nu_b_g = interp(ser, "t_nd", tgrid, "nu_bot")
        print(f"    Nu_bot gradient family: ASPECT {nu_b_g.mean():10.5f}  ours "
              f"{nu_b_o.mean():10.5f}  diff "
              f"{100*(nu_b_o.mean()-nu_b_g.mean())/abs(nu_b_g.mean()):+7.2f} %")


if __name__ == "__main__":
    main()
