"""Project the ASPECT realistic_E run's cost to t_nd = 1e-3 (t = 8.41e15 s).

Data:
  aspect_runs/realistic/run_E.log        (per-step "*** Timestep N: t=... dt=...")
  aspect_runs/realistic/run_E/statistics (per-step Vrms, vmax, cells)
  out_julia/case_realistic/frames/frame_meta.csv  (our A run: u_max(t_nd), 0..3.4e-2)
  out_julia/case_realistic_E/frames/frame_meta.csv (our E run, so far)

Model: ASPECT's time step is CFL-limited, dt = c/u_max with c = dt*u_max nearly
constant (it only changes when AMR changes the smallest cell).  So the number of
steps to advance dt is u_max(t)/c dt, and c is calibrated on the last steps of
the actual run.  u_max(t) is taken from our A run (same case, 10000 steps to
t_nd = 3.4e-2), which brackets the whole target window.
"""
import argparse
import os
import re
import statistics
import sys

D = 2.9e6
KAPPA = 1e-6
T0 = D * D / KAPPA          # 8.41e18 s
U0 = KAPPA / D              # 3.4483e-13 m/s
TARGET = 1.0e-3

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def parse_log(path):
    pat = re.compile(r"\*\*\* Timestep (\d+):\s+t=([\d.eE+-]+) seconds,\s+dt=([\d.eE+-]+) seconds")
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                out.append(dict(step=int(m.group(1)), t=float(m.group(2)),
                                dt=float(m.group(3))))
    return out


def parse_stats(path):
    names, rows = {}, []
    with open(path, errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("#"):
                m = re.match(r"#\s*(\d+):\s*(.*)", s)
                if m:
                    names[int(m.group(1))] = m.group(2).strip()
                continue
            if s:
                rows.append(s.split())
    def col(tag):
        for i, n in names.items():
            if n.startswith(tag):
                return i - 1
        raise KeyError(tag)
    c_step, c_vrms, c_vmax = col("Time step number"), col("RMS velocity"), col("Max. velocity")
    c_cells = col("Number of mesh cells")
    out = []
    for v in rows:
        try:
            out.append(dict(step=int(v[c_step]),
                            vrms=float(v[c_vrms]) * D / KAPPA,
                            vmax=float(v[c_vmax]) * D / KAPPA,
                            cells=float(v[c_cells])))
        except (ValueError, IndexError):
            continue
    return out


def parse_frames(path):
    out = []
    with open(path, errors="replace") as f:
        header = f.readline().strip().split(",")
        for line in f:
            p = line.strip().split(",")
            if len(p) != len(header):
                continue
            r = dict(zip(header, p))
            try:
                out.append((float(r["t"]), float(r["umax"]), float(r["Vrms"])))
            except (ValueError, KeyError):
                pass
    out.sort()
    return out


def sample(curve, t):
    """log-linear interpolation of umax in t."""
    if t <= curve[0][0]:
        return curve[0][1]
    if t >= curve[-1][0]:
        return curve[-1][1]
    lo, hi = 0, len(curve) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if curve[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    (t0, u0, _), (t1, u1, _) = curve[lo], curve[hi]
    if u0 <= 0 or u1 <= 0:
        return u0 + (u1 - u0) * (t - t0) / (t1 - t0)
    w = (t - t0) / (t1 - t0)
    return u0 * (u1 / u0) ** w


def integrate(curve, t_start, t_end):
    """integral of umax dt on a log grid with 4000 points per decade."""
    n = 400
    # log grid in t plus the endpoints
    lo, hi = t_start, t_end
    import math
    pts = [lo + (hi - lo) * k / n for k in range(n + 1)] if hi / lo < 1.5 else \
          [math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * k / n) for k in range(n + 1)]
    total = 0.0
    prev_t, prev_u = pts[0], sample(curve, pts[0])
    for t in pts[1:]:
        u = sample(curve, t)
        total += 0.5 * (prev_u + u) * (t - prev_t)
        prev_t, prev_u = t, u
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(HERE, "run_E.log"),
                    help="ASPECT log (default: run_E.log)")
    ap.add_argument("--stats", default=os.path.join(HERE, "run_E", "statistics"),
                    help="ASPECT statistics file (default: run_E/statistics)")
    ap.add_argument("--target", type=float, default=TARGET,
                    help="the t_nd the projection is to (default: 1e-3; "
                         "section 17 uses 5e-3 for run_E_long)")
    args = ap.parse_args()
    t_target = args.target
    log = parse_log(args.log)
    stats = parse_stats(args.stats)
    a = parse_frames(os.path.join(REPO, "out_julia", "case_realistic", "frames", "frame_meta.csv"))
    e_path = os.path.join(REPO, "out_julia", "case_realistic_E", "frames", "frame_meta.csv")
    e = parse_frames(e_path) if os.path.exists(e_path) else []

    by_step = {s["step"]: s for s in stats}
    merged = [dict(s, tnd=s["t"] / T0, dtnd=s["dt"] / T0) for s in log
              if s["step"] in by_step]
    for s in merged:
        s.update(by_step[s["step"]])
        s["c"] = s["dtnd"] * s["vmax"]
    last = merged[-1]
    c_recent = statistics.median([s["c"] for s in merged[-20:]])
    print(f"ASPECT: {len(merged)} steps, last step {last['step']}, "
          f"t_nd = {last['tnd']:.4e}, dt_nd = {last['dtnd']:.3e}, "
          f"Vrms = {last['vrms']:.0f}, vmax = {last['vmax']:.0f}, "
          f"cells = {last['cells']:.0f}")
    print(f"c = dt_nd * vmax: last = {last['c']:.3e}, median(last 20) = {c_recent:.3e}, "
          f"min/max over run = {min(s['c'] for s in merged):.3e}/"
          f"{max(s['c'] for s in merged):.3e}")
    if e:
        print(f"our E : {len(e)} frames, t_nd = {e[0][0]:.3e} .. {e[-1][0]:.3e}, "
              f"umax = {e[-1][1]:.0f}, Vrms = {e[-1][2]:.0f}")
    print(f"our A : t_nd = {a[0][0]:.3e} .. {a[-1][0]:.3e}, "
          f"umax peak = {max(x[1] for x in a):.0f} at t_nd = "
          f"{max(a, key=lambda x: x[1])[0]:.2e}, last umax = {a[-1][1]:.0f}")

    t_start = last["tnd"]
    for curve_name, curve in (("A curve", a), ("E curve where available", e + [x for x in a if x[0] > e[-1][0]] if e else a)):
        I = integrate(curve, t_start, t_target)
        for c in (c_recent, min(s["c"] for s in merged[-50:]), max(s["c"] for s in merged[-50:])):
            n = I / c
            print(f"  {curve_name:26s} c={c:.3e}: integral u dt = {I:.4e} -> "
                  f"{n:.0f} steps to t_nd=1e-3, total {last['step'] + n:.0f}")

    # sensitivity: what if the flow decays like our A run (umax -> 1e3) vs stays
    print("\numax(t_nd) from our A run at selected times:")
    for t in (2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3):
        print(f"  t_nd={t:.1e}: umax={sample(a, t):8.0f}  Vrms={sample([(x[0], x[2], x[2]) for x in a], t):8.0f}")


if __name__ == "__main__":
    sys.exit(main())
