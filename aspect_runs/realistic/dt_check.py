#!/usr/bin/env python3
"""Side check for section 18: E vs E6 time-step and velocity history at matched
t_nd.  Section 12.3 claims the extra bottom level did not change the time step
(`c = dt_nd * v_max` identical at 1e-5 .. 3e-4).  The median dt_nd of the two
statistics files differs by a factor ~2.5, so re-measure it directly.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import long_window as lw

HERE = os.path.dirname(os.path.abspath(__file__))
D = 2.9e6
KAPPA = 1e-6


def read(path):
    rows = []
    for line in open(path, errors="replace"):
        if line.startswith("#") or not line.strip():
            continue
        v = line.split()
        try:
            rows.append(dict(step=int(v[0]), t=float(v[1]) / lw.T0,
                             dt=float(v[2]) / lw.T0,
                             cells=int(v[3]),
                             vrms=float(v[10]) * D / KAPPA,
                             vmax=float(v[11]) * D / KAPPA,
                             nubot=-float(v[18]) / lw.NORM))
        except (ValueError, IndexError):
            continue
    return rows


def at(rows, t):
    for r in rows:
        if r["t"] >= t:
            return r
    return None


e = read(os.path.join(HERE, "run_E", "statistics"))
e6 = read(os.path.join(HERE, "run_E6", "statistics"))
print(f"{'t_nd':>9} | {'E: dt_nd':>10} {'vmax':>8} {'vrms':>7} {'cells':>6} {'c=dt*vmax':>10} | "
      f"{'E6: dt_nd':>10} {'vmax':>8} {'vrms':>7} {'cells':>6} {'c=dt*vmax':>10} | "
      f"{'dt E/E6':>8} {'vmax E6/E':>9}")
for t in (1e-5, 3e-5, 1e-4, 3e-4, 5e-4, 7e-4, 9e-4):
    a, b = at(e, t), at(e6, t)
    if not a or not b:
        continue
    print(f"{t:9.1e} | {a['dt']:10.3e} {a['vmax']:8.0f} {a['vrms']:7.1f} {a['cells']:6d} "
          f"{a['dt']*a['vmax']:10.3e} | {b['dt']:10.3e} {b['vmax']:8.0f} {b['vrms']:7.1f} "
          f"{b['cells']:6d} {b['dt']*b['vmax']:10.3e} | {a['dt']/b['dt']:8.3f} "
          f"{b['vmax']/a['vmax']:9.3f}")

for name, rows in (("E", e), ("E6", e6)):
    c = [r["dt"] * r["vmax"] for r in rows[1:] if r["dt"] > 0]
    cs = sorted(c)
    print(f"{name:3s} c = dt_nd*vmax_nd: min {cs[0]:.4e} median {cs[len(cs)//2]:.4e} "
          f"max {cs[-1]:.4e};  dt_nd median "
          f"{sorted(r['dt'] for r in rows if r['dt']>0)[len(rows)//2]:.3e}")
