#!/usr/bin/env python3
"""Ensemble statistics of the ASPECT `realistic_E` family (section 14).

Members: `run_E` (member 0, the original card) plus `run_ens1..4` (the same
card with a 0.2-0.6 % perturbation of one initial-condition seed amplitude; see
`make_ens.py`).  For every member the same quantities are computed as in
`compare_E.py` / `nu_bot_check.py --windows`:

  t_nd    = t * kappa / d^2
  Vrms_nd = Vrms * d / kappa
  T_nd    = T / dT - 0.5
  Nu_top  =  Q_top * d / (Lx * k * dT)
  Nu_bot  = -Q_bot * d / (Lx * k * dT)

window means are uniform-in-time averages of the linearly interpolated series
(2000 samples) over t_nd = 1e-6..1e-3, plus the late sub-window 3e-4..1e-3.
Our `case_realistic_E` frame_meta is read with the same interpolation, so the
percentile of our run can be computed against the ASPECT ensemble.

Usage:
  python3 ensemble_E.py                     # all members found in run_*/statistics
  python3 ensemble_E.py --members 0 1 2 3 4
  python3 ensemble_E.py --json out.json
"""
import argparse
import csv
import json
import math
import os
import re
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))

D = 2.9e6
KAPPA = 1.0e-6
DT = 1000.0
LX = 2.5 * D
K_COND = 3.3
NORM = LX * K_COND * DT / D          # 8250 W per unit Nu
T0 = D * D / KAPPA                   # 8.41e18 s

LO, HI = 1.0e-6, 1.0e-3              # comparison window
LATE_LO, LATE_HI = 3.0e-4, 1.0e-3    # late sub-window (70 % of linear time)

OURS = os.path.join(REPO, "out_julia", "case_realistic_E", "frames", "frame_meta.csv")

PERTURB = {
    0: "unperturbed (realistic_E.prm)",
    1: "A1 0.05 -> 0.0501 (+0.2 %)",
    2: "A1 0.05 -> 0.0502 (+0.4 %)",
    3: "A1 0.05 -> 0.0503 (+0.6 %)",
    4: "A2 0.02 -> 0.0201 (+0.5 %)",
}


# --------------------------------------------------------------------- inputs --
def read_stats(path):
    """ASPECT `statistics`, columns found by their header (as compare_E.py)."""
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
    c_cells = col("Number of mesh cells")
    c_vrms = col("RMS velocity")
    c_tmax = col("Maximal temperature")
    c_qbot = col('Outward heat flux through boundary with indicator 2 ("bottom")')
    c_qtop = col('Outward heat flux through boundary with indicator 3 ("top")')

    out = []
    for v in rows:
        try:
            step = int(v[c_step])
            out.append(dict(step=step, t=float(v[c_t]) / T0,
                            cells=float(v[c_cells]),
                            vrms=float(v[c_vrms]) * D / KAPPA,
                            tmax=float(v[c_tmax]) / DT - 0.5,
                            nu_bot=-float(v[c_qbot]) / NORM,
                            nu_top=float(v[c_qtop]) / NORM,
                            # ASPECT's step-0/1 CBF is a postprocessing artefact
                            cbf_ok=step >= 2))
        except (ValueError, IndexError):
            continue
    return out


def read_ours(path):
    out = []
    for r in csv.DictReader(open(path)):
        try:
            out.append(dict(step=int(r["step"]), t=float(r["t"]),
                            nu_top=float(r["Nu_top"]), nu_bot=float(r["Nu_bot"]),
                            vrms=float(r["Vrms"]), tmax=float(r["Tmax"]),
                            cells=float(r.get("leaves", 0))))
        except (TypeError, ValueError, KeyError):
            continue
    return out


def interp(rows, key, t):
    if not rows or t < rows[0]["t"] or t > rows[-1]["t"]:
        return None
    lo, hi = 0, len(rows) - 1
    if t == rows[-1]["t"]:
        return rows[-1][key]
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


def mean_at(rows, key, lo, hi, n=2000):
    """Uniform-in-time mean of the interpolated series (nu_bot_check.py)."""
    ts = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    vals = [interp(rows, key, t) for t in ts]
    if any(x is None for x in vals):
        return None
    return st.mean(vals)


def member_path(i):
    return os.path.join(HERE, "run_E" if i == 0 else "run_ens%d" % i, "statistics")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--json")
    ap.add_argument("--csv")
    a = ap.parse_args(argv)

    keys = ("nu_top", "nu_bot", "vrms", "tmax")
    lat = ("nu_top", "nu_bot")

    members = []
    for i in a.members:
        p = member_path(i)
        if not os.path.exists(p):
            print("skip member %d: no %s" % (i, p))
            continue
        rows = [r for r in read_stats(p) if r["cbf_ok"]]
        if not rows or rows[-1]["t"] < HI * 0.999:
            print("skip member %d: ends at t_nd=%.3e" % (i, rows[-1]["t"] if rows else -1))
            continue
        rec = dict(member=i, perturbation=PERTURB.get(i, ""), t_end=rows[-1]["t"],
                   steps=rows[-1]["step"], cells=rows[-1]["cells"])
        for k in keys:
            rec["win_" + k] = mean_at(rows, k, LO, HI)
        for k in lat:
            rec["late_" + k] = mean_at(rows, k, LATE_LO, LATE_HI)
        rec["_rows"] = rows
        members.append(rec)

    ours = read_ours(OURS)
    our = {}
    for k in keys:
        our["win_" + k] = mean_at(ours, k, LO, HI)
    for k in lat:
        our["late_" + k] = mean_at(ours, k, LATE_LO, LATE_HI)

    # -------- table ---------------------------------------------------------
    labels = dict(nu_top="Nu_top", nu_bot="Nu_bot", vrms="Vrms", tmax="Tmax")
    print("=== members: window means t_nd = 1e-6..1e-3 (2000 uniform samples) ===")
    head = ("m  %-34s %9s %9s %9s %8s | %9s %9s" %
            ("perturbation", "Nu_top", "Nu_bot", "Vrms", "Tmax",
             "late Nu_bot", "late Nu_top"))
    print(head)
    print("-" * len(head))
    for r in members:
        print("%d  %-34s %9.5f %9.4f %9.1f %8.4f | %9.4f %9.5f" %
              (r["member"], r["perturbation"], r["win_nu_top"], r["win_nu_bot"],
               r["win_vrms"], r["win_tmax"], r["late_nu_bot"], r["late_nu_top"]))
    print("%-37s %9.5f %9.4f %9.1f %8.4f | %9.4f %9.5f" %
          ("ours (case_realistic_E)", our["win_nu_top"], our["win_nu_bot"],
           our["win_vrms"], our["win_tmax"], our["late_nu_bot"], our["late_nu_top"]))

    # -------- ensemble statistics ------------------------------------------
    stats = {}
    for tag, sel in (("all", members), ("perturbed", [r for r in members if r["member"] != 0])):
        if len(sel) < 2:
            continue
        stats[tag] = {}
        for k in tuple(keys) + tuple("late_" + x for x in lat):
            field = ("win_" + k) if k in keys else k
            v = [r[field] for r in sel]
            stats[tag][k] = dict(n=len(v), mean=st.mean(v), sd=st.stdev(v),
                                 min=min(v), max=max(v),
                                 rng=max(v) - min(v))
    print()
    for tag in ("all", "perturbed"):
        if tag not in stats:
            continue
        print("=== ensemble statistics (%s, n=%d) ===" % (tag, stats[tag]["nu_bot"]["n"]))
        print("%-12s %10s %10s %10s %10s %10s %10s" %
              ("quantity", "mean", "sd", "sd/|mean|%", "min", "max", "range"))
        for k in ("nu_top", "nu_bot", "vrms", "tmax", "late_nu_top", "late_nu_bot"):
            s = stats[tag][k]
            print("%-12s %10.5g %10.4g %10.3f %10.5g %10.5g %10.4g" %
                  (k, s["mean"], s["sd"], 100 * s["sd"] / abs(s["mean"]),
                   s["min"], s["max"], s["rng"]))
        print()

    # -------- percentile of our run ----------------------------------------
    if "all" not in stats:
        print("\n(need at least two members for ensemble statistics)")
        return 0
    print()
    print("=== our run vs the ensemble ===")
    print("%-10s %10s %10s %8s %8s %10s %8s %10s" %
          ("quantity", "ours", "ens mean", "z", "pct", "ens min", "ens max", "gap%"))
    pctls = {}
    for k in ("nu_top", "nu_bot", "vrms", "tmax", "late_nu_bot", "late_nu_top"):
        field = ("win_" + k) if k in keys else k
        s = stats["all"][k]
        v = our[field]
        z = (v - s["mean"]) / s["sd"] if s["sd"] > 0 else float("nan")
        # percentile: fraction of members at or below ours (ties count half)
        vals = [r[field] for r in members]
        below = sum(1 for x in vals if x < v)
        eq = sum(1 for x in vals if x == v)
        pct = 100.0 * (below + 0.5 * eq) / len(members)
        gap = 100.0 * (s["mean"] - v) / abs(v)
        pctls[k] = dict(ours=v, mean=s["mean"], sd=s["sd"], z=z, percentile=pct,
                        min=s["min"], max=s["max"], gap_pct=gap)
        print("%-10s %10.5g %10.5g %8.2f %8.1f %10.5g %8.5g %8.2f" %
              (k, v, s["mean"], z, pct, s["min"], s["max"], gap))

    # -------- per-member gap to ours ---------------------------------------
    print()
    print("=== per-member gap to our run (%; 100*(member-ours)/|ours|) ===")
    print("%-6s %10s %10s %10s %10s" % ("member", "Nu_top", "Nu_bot", "late Nu_bot", "Tmax"))
    for r in members:
        print("%-6d %10.3f %10.2f %10.2f %10.2f" %
              (r["member"],
               100 * (r["win_nu_top"] - our["win_nu_top"]) / abs(our["win_nu_top"]),
               100 * (r["win_nu_bot"] - our["win_nu_bot"]) / abs(our["win_nu_bot"]),
               100 * (r["late_nu_bot"] - our["late_nu_bot"]) / abs(our["late_nu_bot"]),
               100 * (r["win_tmax"] - our["win_tmax"]) / abs(our["win_tmax"])))

    # -------- divergence of the trajectories --------------------------------
    print()
    print("=== fixed-time spread of Vrms across the ASPECT ensemble ===")
    print("%-10s %10s %10s %10s %10s" % ("t_nd", "mean", "sd", "sd/mean %", "range"))
    tprobe = [1e-6, 1e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3]
    div = []
    for t in tprobe:
        v = [interp(r["_rows"], "vrms", t) for r in members]
        if any(x is None for x in v):
            continue
        m, s = st.mean(v), st.stdev(v)
        div.append(dict(t=t, mean=m, sd=s, rng=max(v) - min(v)))
        print("%-10.1e %10.1f %10.1f %10.2f %10.1f" % (t, m, s, 100 * s / m, max(v) - min(v)))
        del v

    # -------- perturbation growth: member m vs member 0 ---------------------
    growth = []
    if len(members) > 1:
        ref = members[0]["_rows"]
        print()
        print("=== perturbation growth relative to member 0 (run_E) ===")
        print("%-10s %-24s %12s %12s %12s %12s %8s %8s" %
              ("t_nd", "member", "dVrms/Vrms", "dNu_bot", "dNu_top", "dTmax",
               "cells0", "cellsm"))
        tprobe2 = [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 1.5e-4, 2e-4,
                   3e-4, 4e-4, 5e-4, 7e-4, 1e-3]
        for r in members[1:]:
            for t in tprobe2:
                if t > r["_rows"][-1]["t"] or t > ref[-1]["t"]:
                    continue
                r0 = [interp(ref, k, t) for k in ("vrms", "nu_bot", "nu_top", "tmax", "cells")]
                rm = [interp(r["_rows"], k, t) for k in ("vrms", "nu_bot", "nu_top", "tmax", "cells")]
                if any(x is None for x in r0 + rm):
                    continue
                rec = dict(t=t, member=r["member"], dVrms_rel=(rm[0] - r0[0]) / r0[0],
                           dNu_bot=rm[1] - r0[1], dNu_top=rm[2] - r0[2],
                           dTmax=rm[3] - r0[3], cells0=r0[4], cellsm=rm[4])
                growth.append(rec)
                print("%-10.1e %-24s %12.3e %12.3e %12.3e %12.3e %8.0f %8.0f" %
                      (t, "m%d" % r["member"], rec["dVrms_rel"], rec["dNu_bot"],
                       rec["dNu_top"], rec["dTmax"], r0[4], rm[4]))

    out = dict(members=[{k: v for k, v in r.items() if k != "_rows"} for r in members],
               ours=our, stats=stats, percentiles=pctls, divergence=div,
               growth=growth)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
        print("\nwrote", a.json)
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["member", "perturbation", "Nu_top", "Nu_bot", "Vrms", "Tmax",
                         "late_Nu_bot", "late_Nu_top", "steps", "t_end"])
            for r in members:
                wr.writerow([r["member"], r["perturbation"], r["win_nu_top"],
                             r["win_nu_bot"], r["win_vrms"], r["win_tmax"],
                             r["late_nu_bot"], r["late_nu_top"], r["steps"], r["t_end"]])
        print("wrote", a.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
