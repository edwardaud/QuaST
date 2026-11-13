#!/usr/bin/env python3
"""Cost projection for `realistic_E6_long.prm` (section 18): how many steps and
how much wall clock does it take to carry the E6 configuration (bottom band at
level 6) from its own checkpoint to t_nd = 3e-3?

Data (all measured, nothing assumed):
  run_E6/statistics        E6 per-step `dt` and `vmax` up to t_nd = 1e-3
  run_E6.log               E6 step times (wall clock) -> s/step
  run_E_long/statistics    the L5 run's own `dt(t_nd)`, measured to 5e-3
  run_E_long.log           its wall clock -> s/step

Model: ASPECT's step is CFL-limited, dt_nd * vmax_nd = c (nearly constant), so
the number of steps to cover a window is N = int dtau / dt_nd(tau).  E6's own
dt(tau) is known to 1e-3; beyond that the *shape* is taken from run_E_long
(same case, one level less at the wall, measured all the way to 5e-3) and the
*scale* from the E6/E_long ratio measured on the overlap 7.6e-4 .. 1e-3.

Usage: python3 project_e6.py [--target 3e-3]
"""
import argparse
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
D = 2.9e6
KAPPA = 1e-6
T0 = D * D / KAPPA          # 8.41e18 s


def read_stats(path):
    names = {}
    rows = []
    for line in open(path, errors="replace"):
        s = line.strip()
        if s.startswith("#"):
            m = re.match(r"#\s*(\d+):\s*(.*)", s)
            if m:
                names[int(m.group(1))] = m.group(2).strip()
            continue
        if not s:
            continue
        v = s.split()
        try:
            rows.append(dict(step=int(v[0]), t=float(v[1]) / T0,
                             dt=float(v[2]) / T0,
                             vrms=float(v[10]) * D / KAPPA,
                             vmax=float(v[11]) * D / KAPPA))
        except (ValueError, IndexError):
            continue
    return rows


def read_log(path):
    pat = re.compile(r"\*\*\* Timestep (\d+):\s+t=([\d.eE+-]+) seconds")
    out = []
    for line in open(path, errors="replace"):
        m = pat.search(line)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
    return out


def interp(rows, key, t):
    if t < rows[0]["t"] or t > rows[-1]["t"]:
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


def steps_between(rows, t0, t1, n=20000):
    """N = int_{t0}^{t1} dtau/dt_nd(tau), dt given by the rows' own `dt` column."""
    tot = 0.0
    prev_t = t0
    prev_dt = interp(rows, "dt", t0)
    for i in range(1, n + 1):
        t = t0 + (t1 - t0) * i / n
        dt = interp(rows, "dt", t)
        if dt is None:
            return None
        tot += 0.5 * (1.0 / prev_dt + 1.0 / dt) * (t - prev_t)
        prev_t, prev_dt = t, dt
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=3e-3)
    a = ap.parse_args()

    e6 = read_stats(os.path.join(HERE, "run_E6", "statistics"))
    el = read_stats(os.path.join(HERE, "run_E_long", "statistics"))
    log6 = read_log(os.path.join(HERE, "run_E6.log"))
    logl = read_log(os.path.join(HERE, "run_E_long.log"))

    # checkpoint of E6 = the state run_E6_long resumes from
    t_ck = 6.38887e15 / T0
    print(f"E6 statistics: {len(e6)} steps, t_nd = {e6[0]['t']:.4e} .. {e6[-1]['t']:.4e}")
    print(f"E_long        : {len(el)} steps, t_nd = {el[0]['t']:.4e} .. {el[-1]['t']:.4e}")
    print(f"E6 checkpoint : step 3716, t = 6.38887e15 s = t_nd {t_ck:.4e}")

    # --- s/step, measured from the logs ------------------------------------
    def wall(log, path):
        if len(log) < 2:
            return None
        t_first = os.path.getmtime(path)
        return None
    sp6 = 4860.0 / 4930        # section 12.4: 18:20:05 -> 19:41:05, 4930 steps
    spl = 7590.0 / 9063        # section 17.1: 04:22:57 -> 06:29, 9063 steps
    print(f"s/step measured: E6 {sp6:.3f} (81 min / 4930), E_long {spl:.3f} (126.5 min / 9063)")

    # --- dt ratio on the overlap ------------------------------------------
    lo, hi = t_ck, min(e6[-1]["t"], el[-1]["t"])
    ratios = []
    for r in e6:
        if lo <= r["t"] <= hi:
            d = interp(el, "dt", r["t"])
            if d:
                ratios.append(r["dt"] / d)
    print(f"\ndt ratio E6/E_long on [{lo:.4e},{hi:.4e}]: n={len(ratios)}, "
          f"median {st.median(ratios):.4f}, min {min(ratios):.4f}, max {max(ratios):.4f}")

    # --- c = dt_nd * vmax -------------------------------------------------
    for name, rows in (("E6", e6), ("E_long", el)):
        cs = [r["dt"] * r["vmax"] for r in rows if r["dt"] > 0]
        print(f"c = dt_nd*vmax {name:6s}: median {st.median(cs):.4e} "
              f"(last 50 steps {st.median(cs[-50:]):.4e})")

    # --- step counts -------------------------------------------------------
    n_el_ck = sum(1 for r in el if lo <= r["t"] <= a.target)
    n_el_1e3 = sum(1 for r in el if lo <= r["t"] <= 1e-3)
    print(f"\nE_long measured steps on [{lo:.4e},{a.target:.1e}]  = {n_el_ck}")
    print(f"E_long measured steps on [{lo:.4e},1e-3]              = {n_el_1e3}")
    print(f"E6     measured steps on [{lo:.4e},1e-3]              = "
          f"{sum(1 for r in e6 if lo <= r['t'] <= 1e-3)}")

    med = st.median(ratios)
    print("\nprojection to t_nd = %.3e (t = %.4e s):" % (a.target, a.target * T0))
    for label, dt_scale in (("dt = E_long's own dt * median ratio", med),
                            ("dt = E6's own dt (no scaling, lower bound on N)", 1.0)):
        # rows with dt scaled: emulate by dividing the step size
        rows = [dict(t=r["t"], dt=r["dt"] * dt_scale) for r in el]
        n = steps_between(rows, t_ck, a.target)
        for spr, nm in ((sp6, "E6 s/step"), (spl, "E_long s/step")):
            print(f"  {label:46s}: N = {n:7.0f} steps -> "
                  f"{n * spr / 3600.0:5.2f} h at {nm} {spr:.3f}")
    print("\n(reference: E_long actually took %d steps and %.2f h for the same window)"
          % (n_el_ck, n_el_ck * spl / 3600.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
