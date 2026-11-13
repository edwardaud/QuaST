#!/usr/bin/env python3
"""Compare the ASPECT realistic-case run with MantleSEM's A run.

Reads
  * ASPECT's `statistics` file (one row per time step), and
  * MantleSEM's `out_julia/case_realistic/run.log` (one `step` line per step),

converts the ASPECT diagnostics to the nondimensional units of the MantleSEM
case and writes a table at matched nondimensional times.

Conversions (docs/ASPECT_REALISTIC_COMPARE.md, section 2):
  t_nd      = t_ASPECT * kappa / d^2
  Vrms_nd   = Vrms_ASPECT * d / kappa
  T_nd      = T_ASPECT / dT - eta_T0          (eta_T0 = 0.5)
  Nu_top    = Q_top * d / (Lx * k * dT)
  Nu_bot    = -Q_bot * d / (Lx * k * dT)

The heat-flux postprocessor reports the *outward* flux in W (in 2-D: W per unit
length), so the bottom flux enters with a minus sign to match MantleSEM's
Nu_bot = -dT/dz|_bot.

Usage:
  python3 compare.py <aspect_statistics> <mantlesem_run.log> [t_nd ...]
"""

import re
import sys

D = 2.9e6
KAPPA = 1.0e-6
DT = 1000.0
L_X = 2.5 * D
K_COND = 3.3          # kappa * cp * rho0 = 1e-6 * 1000 * 3300
NORM = L_X * K_COND * DT / D      # = 8250 W per unit of Nu


def read_aspect(path):
    """Return a list of dicts with t, Vrms, Tmin/Tavg/Tmax, Nu_top, Nu_bot."""
    names = {}
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                m = re.match(r"#\s*(\d+):\s*(.*)", line)
                if m:
                    names[int(m.group(1))] = m.group(2).strip()
                continue
            if not line:
                continue
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
    c_tmin = col("Minimal temperature")
    c_tavg = col("Average temperature")
    c_tmax = col("Maximal temperature")
    c_qtop = col('Outward heat flux through boundary with indicator 3 ("top")')
    c_qbot = col('Outward heat flux through boundary with indicator 2 ("bottom")')
    out = []
    for v in rows:
        try:
            out.append(dict(
                step=int(v[c_step]),
                t=float(v[c_t]),
                dt=float(v[c_dt]),
                cells=float(v[c_cells]),
                vrms=float(v[c_vrms]),
                tmin=float(v[c_tmin]),
                tavg=float(v[c_tavg]),
                tmax=float(v[c_tmax]),
                qtop=float(v[c_qtop]),
                qbot=float(v[c_qbot]),
            ))
        except (ValueError, KeyError, IndexError):
            continue
    return out


def read_ours(path):
    pat = re.compile(
        r"step\s+(\d+)\s+t=([\d.eE+-]+)\s+dt=([\d.eE+-]+)\s+"
        r"Nu=([\d.eE+-]+)/([\d.eE+-]+)\s+Vrms=([\d.eE+-]+)")
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                out.append(dict(step=int(m.group(1)), t=float(m.group(2)),
                                dt=float(m.group(3)), nu_top=float(m.group(4)),
                                nu_bot=float(m.group(5)), vrms=float(m.group(6))))
    return out


def read_frames(path):
    """Our frame_meta.csv (every 5th step): t, Tmax, Nu, Vrms."""
    out = []
    with open(path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            p = line.strip().split(",")
            if len(p) != len(header):
                continue
            r = dict(zip(header, p))
            out.append(dict(t=float(r["t"]), step=int(r["step"]),
                            tmax=float(r["Tmax"]), tmin=float(r["Tmin"]),
                            nu_top=float(r["Nu_top"]), nu_bot=float(r["Nu_bot"]),
                            vrms=float(r["Vrms"])))
    return out


def interp(series, key, t):
    """Linear interpolation of `key` at time t (series sorted by t)."""
    if t <= series[0]["t"]:
        return series[0][key]
    if t >= series[-1]["t"]:
        return series[-1][key]
    lo, hi = 0, len(series) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if series[mid]["t"] <= t:
            lo = mid
        else:
            hi = mid
    a, b = series[lo], series[hi]
    w = (t - a["t"]) / (b["t"] - a["t"])
    return a[key] + w * (b[key] - a[key])


def main():
    stats = read_aspect(sys.argv[1])
    ours = read_ours(sys.argv[2])
    frames = []
    import os
    fpath = os.path.join(os.path.dirname(os.path.abspath(sys.argv[2])), "frames", "frame_meta.csv")
    if os.path.exists(fpath):
        frames = read_frames(fpath)
    times = [float(x) for x in sys.argv[3:]]
    if not times:
        times = [0.0, 2e-6, 5e-6, 1e-5, 1.5e-5, 2e-5, 2.5e-5, 2.8e-5]
    print(f"ASPECT rows: {len(stats)}  (t = {stats[0]['t']:.3e} .. {stats[-1]['t']:.3e} s)")
    print(f"ours   rows: {len(ours)}  (t = {ours[0]['t']:.3e} .. {ours[-1]['t']:.3e})")
    head = (f"{'t_nd':>10} {'ASPECT t(s)':>12} {'step':>5} {'Nu_top_A':>9} {'Nu_top_O':>9} "
            f"{'Nu_bot_A':>9} {'Nu_bot_O':>9} {'Vrms_A':>9} {'Vrms_O':>9} "
            f"{'Tmax_A':>7} {'Tmax_O':>7}")
    print(head)
    for t_nd in times:
        t_a = t_nd * D * D / KAPPA
        # nearest ASPECT row
        rec = min(stats, key=lambda r: abs(r["t"] - t_a))
        if abs(rec["t"] - t_a) > 0.02 * max(t_a, 1e-30) and t_a > 0:
            rec = None
        if rec is None:
            continue
        nu_top_a = rec["qtop"] / NORM
        nu_bot_a = -rec["qbot"] / NORM
        vrms_a = rec["vrms"] * D / KAPPA
        tmax_a = rec["tmax"] / DT - 0.5
        nu_top_o = interp(ours, "nu_top", t_nd)
        nu_bot_o = interp(ours, "nu_bot", t_nd)
        vrms_o = interp(ours, "vrms", t_nd)
        tmax_o = interp(frames, "tmax", t_nd) if frames else float("nan")
        print(f"{t_nd:10.3e} {rec['t']:12.4e} {rec['step']:5d} "
              f"{nu_top_a:9.5f} {nu_top_o:9.5f} {nu_bot_a:9.4f} {nu_bot_o:9.4f} "
              f"{vrms_a:9.4e} {vrms_o:9.4e} {tmax_a:7.4f} {tmax_o:7.4f} "
              f"  (cells={rec['cells']:.0f})")


if __name__ == "__main__":
    main()
