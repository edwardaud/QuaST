#!/usr/bin/env python3
"""Compare the ASPECT `realistic_E` run with QuaST's `case_realistic_E` run.

The two runs are the *same card*: 2.5 x 1 Boussinesq box, T_top = 0 / T_bot = 1,
H = 10, Ra = 1e5, composite isostress diffusion+dislocation rheology with the
two phase transitions (latent heat per transition), AMR to level 7, run from
t = 0.  ASPECT is dimensional, our solver is nondimensional, so the ASPECT
diagnostics are converted with the scale set below:

    t_nd    = t_ASPECT * kappa / d^2          (d = 2.9e6 m, kappa = 1e-6 m^2/s)
    Vrms_nd = Vrms_ASPECT * d / kappa
    T_nd    = T_ASPECT / dT - 0.5             (dT = 1000 K, eta_T0 = 0.5)
    Nu_top  = Q_top * d / (Lx * k * dT)
    Nu_bot  = -Q_bot * d / (Lx * k * dT)      (Q = outward heat flux in W)

Our run is *already* nondimensional: its frame_meta.csv column `t` is t_nd and
its Nu_top / Nu_bot / Vrms are the same nondimensional quantities, so no
conversion is applied to it.

Output: a table of Nu_top, Nu_bot, Vrms (and Tmax) at matched t_nd, with both
sides linearly interpolated onto the matched times.

Usage:
  python3 compare_E.py                        # defaults, geometric time grid
  python3 compare_E.py --times 2e-6 5e-6 1e-5
  python3 compare_E.py --n 20                 # number of matched times
  python3 compare_E.py --statistics PATH --frames PATH

Defaults:
  statistics = <this dir>/run_E/statistics
  frames     = <repo>/out_julia/case_realistic_E/frames/frame_meta.csv
"""

import math
import os
import re
import sys

# --- the scale set ----------
D = 2.9e6                 # box depth [m]
KAPPA = 1.0e-6            # thermal diffusivity [m^2/s]
DT = 1000.0               # temperature drop [K]
L_X = 2.5 * D             # box length [m]
K_COND = 3.3              # kappa cp rho0 [W/(m K)]
NORM = L_X * K_COND * DT / D      # W per unit of Nu (= 8250)
T0 = D * D / KAPPA        # 8.41e18 s, the time scale

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STATS = os.path.join(HERE, "run_E", "statistics")
DEFAULT_FRAMES = os.path.join(
    HERE, "..", "..", "out_julia", "case_realistic_E", "frames", "frame_meta.csv")


# ---------------------------------------------------------------------------
def read_aspect(path):
    """ASPECT `statistics`: one dict per row, columns found by their header."""
    names = {}
    rows = []
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

    c_step = col("Time step number")
    c_t = col("Time (seconds)")
    c_dt = col("Time step size")
    c_cells = col("Number of mesh cells")
    c_vrms = col("RMS velocity")
    c_vmax = col("Max. velocity")
    c_tmax = col("Maximal temperature")
    c_qtop = col('Outward heat flux through boundary with indicator 3 ("top")')
    c_qbot = col('Outward heat flux through boundary with indicator 2 ("bottom")')

    out = []
    for v in rows:
        try:
            t_s = float(v[c_t])
            step = int(v[c_step])
            out.append(dict(
                step=step, t=t_s / T0, t_s=t_s, dt=float(v[c_dt]),
                cells=float(v[c_cells]), vrms=float(v[c_vrms]) * D / KAPPA,
                vmax=float(v[c_vmax]) * D / KAPPA,
                nu_top=float(v[c_qtop]) / NORM,
                nu_bot=-float(v[c_qbot]) / NORM,
                tmax=float(v[c_tmax]) / DT - 0.5,
                # the CBF bottom flux of the first step is a postprocessing
                # artefact
                cbf_ok=(step >= 2)))
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
                out.append(dict(step=int(r["step"]), t=float(r["t"]),
                                nu_top=float(r["Nu_top"]), nu_bot=float(r["Nu_bot"]),
                                vrms=float(r["Vrms"]), tmax=float(r["Tmax"]),
                                leaves=int(r.get("leaves", 0))))
            except (ValueError, KeyError):
                continue
    return out


def bracket(series, t):
    """Index pair (lo, hi) with t within [t_lo, t_hi], or (None, None)."""
    if not series:
        return None, None
    if t < series[0]["t"] or t > series[-1]["t"]:
        return None, None
    if t == series[-1]["t"]:
        return len(series) - 1, len(series) - 1
    lo, hi = 0, len(series) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if series[mid]["t"] <= t:
            lo = mid
        else:
            hi = mid
    return lo, hi


def interp(series, key, t):
    """Linear interpolation in t_nd; None when t is outside the series."""
    lo, hi = bracket(series, t)
    if lo is None:
        return None
    if lo == hi:
        return series[lo][key]
    a, b = series[lo], series[hi]
    w = (t - a["t"]) / (b["t"] - a["t"])
    return a[key] + w * (b[key] - a[key])


def nearest(series, t):
    lo, hi = bracket(series, t)
    if lo is None:
        return None
    if lo == hi:
        return series[lo]
    return series[lo] if abs(series[lo]["t"] - t) <= abs(series[hi]["t"] - t) else series[hi]


def pct(a, b):
    """100 * (a - b)/|b|, or nan."""
    if a is None or b is None or b == 0:
        return float("nan")
    return 100.0 * (a - b) / abs(b)


def main(argv):
    stats_path = DEFAULT_STATS
    frames_path = DEFAULT_FRAMES
    times = []
    n_grid = 12
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--statistics":
            stats_path = argv[i + 1]; i += 2
        elif a == "--frames":
            frames_path = argv[i + 1]; i += 2
        elif a == "--times":
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                times.append(float(argv[i])); i += 1
        elif a == "--n":
            n_grid = int(argv[i + 1]); i += 2
        elif a in ("-h", "--help"):
            print(__doc__); return 0
        else:
            raise SystemExit("unknown argument: " + a)

    if not os.path.exists(stats_path):
        raise SystemExit("no ASPECT statistics file: " + stats_path)
    if not os.path.exists(frames_path):
        raise SystemExit("no frame_meta.csv: " + frames_path)

    aspect = read_aspect(stats_path)
    ours = read_our_frames(frames_path)
    if not aspect or not ours:
        raise SystemExit("empty input")

    print(f"ASPECT : {stats_path}")
    print(f"         {len(aspect)} rows, step {aspect[0]['step']}..{aspect[-1]['step']}, "
          f"t_nd = {aspect[0]['t']:.4e} .. {aspect[-1]['t']:.4e} "
          f"({aspect[-1]['t_s']:.3e} s), last dt_nd = {aspect[-1]['dt'] / T0:.3e}")
    dt_ours = float("nan")
    if len(ours) > 1 and ours[-1]["step"] != ours[-2]["step"]:
        dt_ours = ((ours[-1]["t"] - ours[-2]["t"]) /
                   (ours[-1]["step"] - ours[-2]["step"]))
    print(f"ours   : {frames_path}")
    print(f"         {len(ours)} frames, step {ours[0]['step']}..{ours[-1]['step']}, "
          f"t_nd = {ours[0]['t']:.4e} .. {ours[-1]['t']:.4e}, "
          f"last dt_nd = {dt_ours:.3e}")
    lo = max(aspect[0]["t"], ours[0]["t"])
    hi = min(aspect[-1]["t"], ours[-1]["t"])
    if hi <= lo:
        print(f"\n*** no overlap: ASPECT ends at t_nd = {aspect[-1]['t']:.3e}, "
              f"ours starts at {ours[0]['t']:.3e} ***")
        return 0
    print(f"overlap: t_nd = {lo:.4e} .. {hi:.4e} "
          f"({hi / lo:.1f}x); our frames every 5 steps, ASPECT every step")

    if not times:
        if hi > lo * 1.0001:
            times = [math.exp(math.log(lo) + k / (n_grid - 1) * math.log(hi / lo))
                     for k in range(n_grid)]
        else:
            times = [lo]
    times = [t for t in times if lo * (1 - 1e-9) <= t <= hi * (1 + 1e-9)]
    if not times:
        print("\n*** none of the requested times is inside the overlap ***")
        return 0
    times = [min(max(t, lo), hi) for t in times]

    print()
    head = (f"{'t_nd':>10} {'t_A(s)':>11} {'step':>5} "
            f"{'Nu_top_A':>9} {'Nu_top_O':>9} {'dNu_top%':>8} "
            f"{'Nu_bot_A':>9} {'Nu_bot_O':>9} {'dNu_bot%':>8} "
            f"{'Vrms_A':>9} {'Vrms_O':>9} {'dVrms%':>7} "
            f"{'Tmax_A':>7} {'Tmax_O':>7} {'cells':>6}")
    print(head)
    print("-" * len(head))
    for t in times:
        ra = nearest(aspect, t)
        nu_top_a = interp(aspect, "nu_top", t)
        nu_bot_a = interp(aspect, "nu_bot", t)
        vrms_a = interp(aspect, "vrms", t)
        tmax_a = interp(aspect, "tmax", t)
        nu_top_o = interp(ours, "nu_top", t)
        nu_bot_o = interp(ours, "nu_bot", t)
        vrms_o = interp(ours, "vrms", t)
        tmax_o = interp(ours, "tmax", t)

        def f(x, fmt):
            return "-" if x is None else fmt % x

        mark = "" if (ra is None or ra["cbf_ok"]) else "*"
        print(f"{t:10.3e} {ra['t_s'] if ra else float('nan'):11.4e} "
              f"{ra['step'] if ra else -1:5d} "
              f"{f(nu_top_a, '%9.5f')} {f(nu_top_o, '%9.5f')} {pct(nu_top_a, nu_top_o):8.3f} "
              f"{f(nu_bot_a, '%9.4f')}{mark:<1} "
              f"{f(nu_bot_o, '%9.4f')} {pct(nu_bot_a, nu_bot_o):8.2f} "
              f"{f(vrms_a, '%9.4e')} {f(vrms_o, '%9.4e')} {pct(vrms_a, vrms_o):7.2f} "
              f"{f(tmax_a, '%7.4f')} {f(tmax_o, '%7.4f')} "
              f"{ra['cells'] if ra else float('nan'):6.0f}")

    print("\n(A = ASPECT converted to our units, O = our run; dX% = 100*(A-O)/|O|)")
    print("* = ASPECT step 0/1, whose CBF bottom flux is a first-step artefact "
          "(section 4);\n    the first ASPECT step is excluded from any comparison.")
    print("Caveat: the flow is an oscillating transient and the two codes are not "
          "phase locked;\n        Vrms and Nu_bot at a single t_nd are sensitive to a "
          "shift of ~1e-6 in t_nd.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
