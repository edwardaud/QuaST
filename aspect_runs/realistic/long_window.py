#!/usr/bin/env python3
"""Long-time-average cross-code comparison of Nu_bot / Nu_top / Vrms / Tmax
(docs/dev/VALIDATION.md section 17).

Why this exists: section 16 showed that the -6.11 % (CBF) / -8.44 % (gradient)
*whole-window* Nu_bot gap over t_nd = [1e-6, 1e-3] is a slow drift along the
trajectory, not a same-state discretisation difference, and that the sign of the
accumulated gap flips with the window position.  A quantity whose sign flips
with the window is not a stationary cross-code bias, so the honest question is
what the gap is in the *long-time mean* over windows far away from the startup
transient:

    [1e-3, 3e-3]   [1e-3, 4e-3]   [2e-3, 4e-3]   [1e-3, 5e-3]

Two wall-functional families are reported for both codes:

  CBF family       ASPECT `statistics` column 19/20 (Gresho CBF of the BDF2 time
                   step) vs our pointwise wall derivative (our own time-step CBF
                   `semquad_nu_cbf`/`nu_cbf_compare.jl` agrees with the pointwise
                   value to 0.045 % at the bottom wall, section 13.1, so the
                   family conversion for us is <= 0.05 %).
  gradient family  ASPECT VTU, exact Q2 wall derivative (`extract_aspect_state.py
                   --series`; the same family as our one-sided GLL derivative --
                   section 16.3 measured the two to 0.003 % on one state) vs our
                   `frame_meta.csv` `Nu_bot`.

Inputs (all read-only):
  aspect_runs/realistic/run_E_long/statistics      dense (every step) CBF series
  aspect_runs/realistic/run_E_long/solution/       VTU frames -> --grad-csv
  out_julia/case_realistic_E/frames/frame_meta.csv our dense (every 5 steps) series

Usage:
  # multi-window table (both families), plus the block-split standard errors
  python3 long_window.py --windows 1e-3:3e-3,1e-3:4e-3,2e-3:4e-3,1e-3:5e-3

  # the running gap: cumulative mean over [t0, t] for a list of t
  python3 long_window.py --running-t0 1e-3 --running-to 1e-3 1.5e-3 2e-3 2.5e-3 3e-3

  # cross-check: the segment run_E_cont already covered, reproduced by run_E_long
  python3 long_window.py --check-overlap run_E_cont/statistics

Exit status is 0; windows that the data do not cover yet are printed as
`not covered` instead of being silently truncated.
"""
import argparse
import csv
import glob
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))

D = 2.9e6                     # box depth [m]
DT = 1000.0                   # temperature drop [K]
LX = 2.5 * D                  # box length [m]
T0 = D * D / 1.0e-6           # time scale [s] = 8.41e18
NORM = LX * 3.3 * DT / D      # W per unit of Nu (= 8250)

DEF_STATS = os.path.join(HERE, "run_E_long", "statistics")
DEF_FRAMES = os.path.join(REPO, "out_julia", "case_realistic_E", "frames",
                          "frame_meta.csv")


# --------------------------------------------------------------------- inputs --
def read_aspect_stats(paths):
    """ASPECT `statistics` rows (columns found by header, not by index)."""
    names, rows, seen = {}, [], set()
    for path in paths:
        names = {}
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
                step = int(v[0])
                if step in seen:            # a later file wins (resume overlap)
                    continue
                seen.add(step)
                rows.append(dict(
                    step=step,
                    t=float(v[1]) / T0,
                    dt=float(v[2]) / T0,
                    vrms=float(v[10]) * D / 1.0e-6,
                    tmax=float(v[14]) / DT - 0.5,
                    tmean=float(v[13]) / DT - 0.5,
                    nu_bot=-float(v[18]) / NORM,
                    nu_top=float(v[19]) / NORM,
                    cbf_ok=step >= 2,
                    src=os.path.basename(os.path.dirname(path))))
            except (ValueError, IndexError):
                continue
    rows.sort(key=lambda r: r["t"])
    return [r for r in rows if r["cbf_ok"]]


def read_grad_csv(paths):
    """`extract_aspect_state.py --series` CSV rows (Q2 gradient family).

    Frames from different runs are keyed by their time, not by their index: the
    frame *counter* restarts from the checkpoint, so run_E_cont and run_E_long
    both begin at `solution-00017` while running at 2.5e13 s and 5e13 s cadence.
    """
    rows, seen = [], set()
    for path in paths:
        for r in csv.DictReader(open(path)):
            try:
                ts = round(float(r["t_s"]), -3)
                if ts in seen:
                    continue
                seen.add(ts)
                rows.append(dict(frame=int(r["frame"]), t=float(r["t_nd"]),
                                 t_s=float(r["t_s"]),
                                 nu_bot=float(r["nu_bot"]),
                                 nu_top=float(r["nu_top"]),
                                 vrms=float(r["Vrms_nd"]),
                                 tmax=float(r["Tmax_nd"]),
                                 tmean=float(r["Tmean_nd"]),
                                 ncells=int(r["ncells"])))
            except (TypeError, ValueError, KeyError):
                continue
    rows.sort(key=lambda r: r["t"])
    return rows


def read_frames(path):
    """Our `frame_meta.csv`; `t` is already t_nd."""
    out = []
    for r in csv.DictReader(open(path)):
        try:
            out.append(dict(step=int(r["step"]), t=float(r["t"]),
                            nu_top=float(r["Nu_top"]), nu_bot=float(r["Nu_bot"]),
                            vrms=float(r["Vrms"]), tmax=float(r["Tmax"]),
                            tmean=float(r["Tmean"]), leaves=int(r["leaves"]),
                            dt=float(r["dt"])))
        except (TypeError, ValueError, KeyError):
            continue
    out.sort(key=lambda r: r["t"])
    return out


# ------------------------------------------------------------------- helpers --
def interp(rows, key, t):
    """Linear interpolation in t; None outside the range."""
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
    if b["t"] == a["t"]:
        return a[key]
    w = (t - a["t"]) / (b["t"] - a["t"])
    return a[key] + w * (b[key] - a[key])


def dense_mean(rows, key, lo, hi, n=2000):
    """Uniform-in-time window mean (the convention of sections 10.1 / 14.3)."""
    vals = [interp(rows, key, lo + (hi - lo) * i / (n - 1)) for i in range(n)]
    if any(v is None for v in vals):
        return None
    return st.mean(vals)


def frame_mean(rows, key, lo, hi):
    """Time-weighted mean over the frames inside [lo, hi] (equal weights when
    the frames are equally spaced; returns (mean, nframes, sd)).  Returns
    (None, n, None) when the frames do not *cover* the window: an average of
    the frames that happen to exist would silently compare different windows."""
    sel = [r for r in rows if lo <= r["t"] <= hi]
    if len(sel) < 3:
        return None, len(sel), None
    pad = 0.02 * (hi - lo)
    if sel[0]["t"] > lo + pad or sel[-1]["t"] < hi - pad:
        return None, len(sel), None
    # trapezoid weights over the frame times, renormalised on the window
    ts = [r["t"] for r in sel]
    vs = [r[key] for r in sel]
    w = [0.0] * len(sel)
    for i in range(len(sel) - 1):
        h = ts[i + 1] - ts[i]
        w[i] += 0.5 * h
        w[i + 1] += 0.5 * h
    tw = sum(w)
    mean = sum(wi * vi for wi, vi in zip(w, vs)) / tw
    return mean, len(sel), st.pstdev(vs)


def block_se(rows, key, lo, hi, k=8, n=400):
    """Standard error of the window mean from a k-block split of the window
    (captures the fact that the samples are temporally correlated)."""
    means = []
    for i in range(k):
        a = lo + (hi - lo) * i / k
        b = lo + (hi - lo) * (i + 1) / k
        m = dense_mean(rows, key, a, b, n)
        if m is not None:
            means.append(m)
    if len(means) < 3:
        return None, len(means)
    return st.stdev(means) / (len(means) ** 0.5), len(means)


def pct(a, b):
    if a is None or b is None or b == 0:
        return float("nan")
    return 100.0 * (a - b) / abs(b)


def parse_windows(spec):
    out = []
    for part in spec.split(","):
        if not part.strip():
            continue
        a, b = part.split(":")
        out.append((float(a), float(b)))
    return out


# ---------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asp-stats", nargs="+", default=[DEF_STATS])
    ap.add_argument("--grad-csv", nargs="*", default=None)
    ap.add_argument("--frames", default=DEF_FRAMES)
    ap.add_argument("--windows", default=None)
    ap.add_argument("--running-t0", type=float, default=None)
    ap.add_argument("--running-to", nargs="*", type=float, default=None)
    ap.add_argument("--check-overlap", default=None,
                    help="a second ASPECT statistics file to compare on the "
                         "overlap with --asp-stats (determinism of the resume)")
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--markdown", action="store_true",
                    help="print the multi-window and running-gap tables as "
                         "markdown, ready to paste into section 17")
    a = ap.parse_args()

    asp = read_aspect_stats(a.asp_stats)
    ours = read_frames(a.frames)
    grad = read_grad_csv(a.grad_csv) if a.grad_csv else []

    print("=" * 100)
    print("inputs")
    print(f"  ASPECT CBF  {a.asp_stats}")
    if asp:
        print(f"              {len(asp)} steps, step {asp[0]['step']}..{asp[-1]['step']}, "
              f"t_nd = {asp[0]['t']:.6e} .. {asp[-1]['t']:.6e}, "
              f"last dt_nd = {asp[-1]['dt']:.3e}")
    print(f"  ours        {a.frames}")
    if ours:
        print(f"              {len(ours)} frames, step {ours[0]['step']}..{ours[-1]['step']}, "
              f"t_nd = {ours[0]['t']:.6e} .. {ours[-1]['t']:.6e}, "
              f"dt_nd = {ours[-1]['dt']:.3e}")
    if grad:
        print(f"  ASPECT grad {len(grad)} VTU frames, "
              f"t_nd = {grad[0]['t']:.6e} .. {grad[-1]['t']:.6e}")
        dts = [b["t"] - aa["t"] for aa, b in zip(grad, grad[1:])]
        if dts:
            print(f"              frame spacing dt_nd = {st.median(dts):.3e} "
                  f"({1/st.median(dts):.0f} frames per unit t_nd)")

    if a.check_overlap:
        other = read_aspect_stats([a.check_overlap])
        print()
        print("=" * 100)
        print(f"resume determinism: {a.check_overlap} vs --asp-stats on their overlap")
        lo = max(asp[0]["t"] if asp else 0, other[0]["t"] if other else 0)
        hi = min(asp[-1]["t"] if asp else 0, other[-1]["t"] if other else 0)
        if hi <= lo:
            print("  *** no overlap ***")
        else:
            print(f"  overlap t_nd = {lo:.6e} .. {hi:.6e}")
            for k in ("nu_bot", "nu_top", "vrms", "tmax"):
                ma = dense_mean(asp, k, lo, hi)
                mb = dense_mean(other, k, lo, hi)
                print(f"  {k:8s} run_E_long = {ma:12.6f}   other = {mb:12.6f}   "
                      f"diff = {mb - ma:+.3e} ({pct(mb, ma):+.4f} %)")

    if a.windows:
        wins = parse_windows(a.windows)
        print()
        print("=" * 100)
        print("window means, both wall-functional families")
        hdr = (f"{'window':>18} {'n_A':>6} {'n_g':>4} {'n_O':>5} | "
               f"{'A_cbf':>9} {'A_grad':>9} {'O':>9} | {'CBF gap':>8} {'GRAD gap':>9} | "
               f"{'Nu_top A_cbf':>12} {'Nu_top A_grad':>13} {'Nu_top O':>9} | "
               f"{'Vrms A':>7} {'Vrms O':>7} {'Tmax A':>7} {'Tmax O':>7}")
        print(hdr)
        print("-" * len(hdr))
        for lo, hi in wins:
            avail = ""
            if asp and hi > asp[-1]["t"]:
                avail = f"  [ASPECT series ends at {asp[-1]['t']:.4e}]"
            if ours and hi > ours[-1]["t"]:
                avail += f"  [our frames end at {ours[-1]['t']:.4e}]"
            na = sum(1 for r in asp if lo <= r["t"] <= hi)
            ng = sum(1 for r in grad if lo <= r["t"] <= hi)
            no = sum(1 for r in ours if lo <= r["t"] <= hi)
            ab = dense_mean(asp, "nu_bot", lo, hi)
            an = dense_mean(asp, "nu_top", lo, hi)
            av = dense_mean(asp, "vrms", lo, hi)
            at = dense_mean(asp, "tmax", lo, hi)
            gb, gn, gv, gt = (frame_mean(grad, "nu_bot", lo, hi)[0],
                              frame_mean(grad, "nu_top", lo, hi)[0],
                              frame_mean(grad, "vrms", lo, hi)[0],
                              frame_mean(grad, "tmax", lo, hi)[0])
            ob = dense_mean(ours, "nu_bot", lo, hi)
            on = dense_mean(ours, "nu_top", lo, hi)
            ov = dense_mean(ours, "vrms", lo, hi)
            ot = dense_mean(ours, "tmax", lo, hi)

            def f(x, fmt="%9.4f"):
                return "        -" if x is None else fmt % x

            print(f"[{lo:.1e},{hi:.1e}] {na:6d} {ng:4d} {no:5d} | "
                  f"{f(ab)} {f(gb)} {f(ob)} | {pct(ab, ob):+7.2f}% {pct(gb, ob):+8.2f}% | "
                  f"{f(an,'%12.4f')} {f(gn,'%13.4f')} {f(on)} | "
                  f"{f(av,'%7.1f')} {f(ov,'%7.1f')} {f(at,'%7.4f')} {f(ot,'%7.4f')}"
                  + avail)
        print()
        print("block-split standard errors (k = %d blocks of the window)" % a.blocks)
        hdr2 = f"{'window':>18} | {'SE A_cbf':>9} {'SE A_grad':>10} {'SE O':>8} | {'SE CBF gap':>11} {'SE GRAD gap':>12}"
        print(hdr2)
        for lo, hi in wins:
            sa, _ = block_se(asp, "nu_bot", lo, hi, a.blocks)
            sb, _ = block_se(ours, "nu_bot", lo, hi, a.blocks)
            # the gradient family has few frames: block SE from its raw scatter
            selg = [r["nu_bot"] for r in grad if lo <= r["t"] <= hi]
            sg = (st.stdev(selg) / (len(selg) ** 0.5)) if len(selg) > 3 else None

            def s(x, fmt="%9.4f"):
                return "        -" if x is None else fmt % x

            def comb(*xs):
                xs = [x for x in xs if x is not None]
                return (sum(x * x for x in xs)) ** 0.5 if xs else None

            print(f"[{lo:.1e},{hi:.1e}] | {s(sa)} {s(sg,'%10.4f')} {s(sb)} | "
                  f"{s(comb(sa, sb),'%11.4f')} {s(comb(sg, sb),'%12.4f')}")

        if a.markdown:
            print()
            print("| window | steps A | frames grad | frames O | "
                  "`Nu_bot` A (CBF) | `Nu_bot` A (grad) | `Nu_bot` O | "
                  "gap CBF | gap grad | `Nu_top` A (CBF) | `Nu_top` A (grad) | "
                  "`Nu_top` O | `Vrms` A | `Vrms` O | `Tmax` A | `Tmax` O |")
            print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

            def m(x, fmt="%.4f"):
                return "—" if x is None else fmt % x

            for lo, hi in wins:
                na = sum(1 for r in asp if lo <= r["t"] <= hi)
                ng = sum(1 for r in grad if lo <= r["t"] <= hi)
                no = sum(1 for r in ours if lo <= r["t"] <= hi)
                ab = dense_mean(asp, "nu_bot", lo, hi)
                gb = frame_mean(grad, "nu_bot", lo, hi)[0]
                ob = dense_mean(ours, "nu_bot", lo, hi)
                an = dense_mean(asp, "nu_top", lo, hi)
                gn = frame_mean(grad, "nu_top", lo, hi)[0]
                on = dense_mean(ours, "nu_top", lo, hi)
                av = dense_mean(asp, "vrms", lo, hi)
                ov = dense_mean(ours, "vrms", lo, hi)
                at = dense_mean(asp, "tmax", lo, hi)
                ot = dense_mean(ours, "tmax", lo, hi)
                print(f"| [{lo:.1e}, {hi:.1e}] | {na} | {ng} | {no} | "
                      f"{m(ab)} | {m(gb)} | {m(ob)} | "
                      f"{m(pct(ab, ob), '%+.2f %%')} | {m(pct(gb, ob), '%+.2f %%')} | "
                      f"{m(an)} | {m(gn)} | {m(on)} | "
                      f"{m(av, '%.1f')} | {m(ov, '%.1f')} | {m(at)} | {m(ot)} |")

    if a.running_t0 is not None and a.running_to:
        t0 = a.running_t0
        print()
        print("=" * 100)
        print(f"running gap: cumulative mean over [{t0:.1e}, t]")
        hdr = (f"{'t_hi':>10} | {'A_cbf':>9} {'A_grad':>9} {'O':>9} | "
               f"{'CBF gap':>8} {'GRAD gap':>9} | {'CBF-O':>9} {'GRAD-O':>9}")
        print(hdr)
        print("-" * len(hdr))
        for hi in a.running_to:
            ab = dense_mean(asp, "nu_bot", t0, hi)
            gb = frame_mean(grad, "nu_bot", t0, hi)[0]
            ob = dense_mean(ours, "nu_bot", t0, hi)
            print(f"{hi:10.4e} | {('        -' if ab is None else '%9.4f' % ab)} "
                  f"{('        -' if gb is None else '%9.4f' % gb)} "
                  f"{('        -' if ob is None else '%9.4f' % ob)} | "
                  f"{('      -' if ab is None else '%+7.2f%%' % pct(ab, ob))} "
                  f"{('      -' if gb is None else '%+8.2f%%' % pct(gb, ob))} | "
                  f"{('       -' if ab is None else '%+9.4f' % (ab - ob))} "
                  f"{('       -' if gb is None else '%+9.4f' % (gb - ob))}")
        if a.markdown:
            print()
            print("| `t_hi` | `Nu_bot` A (CBF) | `Nu_bot` A (grad) | `Nu_bot` O | "
                  "gap CBF | gap grad | CBF − O | grad − O |")
            print("|---|---|---|---|---|---|---|---|")
            for hi in a.running_to:
                ab = dense_mean(asp, "nu_bot", t0, hi)
                gb = frame_mean(grad, "nu_bot", t0, hi)[0]
                ob = dense_mean(ours, "nu_bot", t0, hi)
                print(f"| {hi:.2e} | "
                      f"{'—' if ab is None else '%.4f' % ab} | "
                      f"{'—' if gb is None else '%.4f' % gb} | "
                      f"{'—' if ob is None else '%.4f' % ob} | "
                      f"{'—' if ab is None else '%+.2f %%' % pct(ab, ob)} | "
                      f"{'—' if gb is None else '%+.2f %%' % pct(gb, ob)} | "
                      f"{'—' if ab is None else '%+.4f' % (ab - ob)} | "
                      f"{'—' if gb is None else '%+.4f' % (gb - ob)} |")
    print()
    print("gaps are 100*(A-O)/|O| with A = ASPECT, O = our run; 'A_cbf' = the "
          "`statistics`\nheat-flux column, 'A_grad' = the exact Q2 wall derivative "
          "from the VTU frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
