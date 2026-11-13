#!/usr/bin/env python3
"""Interior artificial-viscosity statistics of an ASPECT run's VTU frames
(docs/dev/VALIDATION.md section 15).

The VTU array is `artificial_viscosity` (units W/(m K), plugin
`source/postprocess/visualization/artificial_viscosity.cc:35,45`), i.e. the same
quantity that enters

    kappa_eff = max(k, alpha_AV)        heat_flux_map.cc:203-212
    max(k, alpha_AV)                    assemblers/advection.cc:172-184

so it is directly comparable with k = 3.3 W/(m K).  deal.II writes cell data as
node data (one value per patch vertex), so the statistics below are over the
deduplicated VTU vertices; that is the same set the doc's section 12 tables use.

Reported per frame: max, median, mean, fraction > k, and the same restricted to
the "interior" rows y/d > 0.09 (above the bottom boundary layer) and to the
bottom rows y/d <= 0.05.  A run that has no such field (e.g. run_E, whose
`List of output variables` is `material properties` only) is reported as absent.

Usage:
  python3 av_stats.py --solution run_E6/solution
  python3 av_stats.py --solution run_noav/solution --k 3.3
  python3 av_stats.py --solution run_E6/solution --solution2 run_noav/solution
"""
import argparse
import glob
import os

import numpy as np
import meshio

D = 2.9e6


def frames(sol_dir):
    out = {}
    for p in sorted(glob.glob(os.path.join(sol_dir, "solution-*.vtu"))):
        base = os.path.basename(p)
        idx = int(base.split("-")[1].split(".")[0])
        out.setdefault(idx, []).append(p)
    return sorted(out.items())


def collect(pieces, field):
    pts, vals, t = [], [], None
    for p in pieces:
        m = meshio.read(p)
        if field not in m.point_data:
            return None, None, None
        pts.append(m.points)
        vals.append(np.asarray(m.point_data[field]).ravel())
        if t is None and "TIME" in m.field_data:
            t = float(np.asarray(m.field_data["TIME"]).ravel()[0])
    pts = np.concatenate(pts)
    vals = np.concatenate(vals)
    key = np.round(np.stack([pts[:, 0] / D, pts[:, 1] / D], axis=1), 9)
    _, first = np.unique(key, axis=0, return_index=True)
    return pts[first], vals[first], t


def stats(v):
    if v.size == 0:
        return dict(n=0)
    return dict(n=int(v.size), max=float(v.max()), med=float(np.median(v)),
                mean=float(v.mean()), f_gt_k=float(np.mean(v > 3.3)),
                f_gt_1=float(np.mean(v > 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--solution2", default=None)
    ap.add_argument("--k", type=float, default=3.3)
    ap.add_argument("--frames", nargs="*", type=int, default=None)
    a = ap.parse_args()

    global K
    K = a.k
    for label, sol in (("run", a.solution), ("run2", a.solution2)):
        if sol is None:
            continue
        fl = frames(sol)
        print(f"=== {label}: {sol}  ({len(fl)} frames) ===")
        print(f"{'frame':>5} {'t_nd':>10} {'n':>6} {'max':>9} {'med':>9} {'mean':>9} "
              f"{'>k':>7} {'>1':>7} | {'int max':>9} {'int med':>9} {'int >k':>7} | "
              f"{'y<=.05 max':>11} {'y<=.05 >k':>10}")
        summary = []
        for idx, pieces in fl:
            if a.frames and idx not in a.frames:
                continue
            pts, av, t = collect(pieces, "artificial_viscosity")
            if pts is None:
                print(f"{idx:5d}  artificial_viscosity: ABSENT")
                continue
            y = pts[:, 1] / D
            s = stats(av)
            si = stats(av[y > 0.09])
            sb = stats(av[y <= 0.05])
            print(f"{idx:5d} {(t/8.41e18 if t is not None else float('nan')):10.3e} {s['n']:6d} "
                  f"{s['max']:9.4g} {s['med']:9.4g} {s['mean']:9.4g} "
                  f"{s['f_gt_k']*100:6.2f}% {s['f_gt_1']*100:6.2f}% | "
                  f"{si.get('max', float('nan')):9.4g} {si.get('med', float('nan')):9.4g} "
                  f"{si.get('f_gt_k', 0)*100:6.2f}% | "
                  f"{sb.get('max', float('nan')):11.4g} {sb.get('f_gt_k', 0)*100:9.2f}%")
            summary.append((t, s, si, sb))
        if summary:
            print()
            print("  over the %d frames, as  [min, median, max]  of the per-frame value:" % len(summary))
            spec = [("max (whole box)          ", lambda s: s[1]["max"]),
                    ("median (whole box)       ", lambda s: s[1]["med"]),
                    ("mean (whole box)         ", lambda s: s[1]["mean"]),
                    ("frac > k  [%]            ", lambda s: s[1]["f_gt_k"]),
                    ("frac > 1 W/(m K)  [%]    ", lambda s: s[1]["f_gt_1"]),
                    ("interior (y>0.09) max    ", lambda s: s[2]["max"]),
                    ("interior (y>0.09) median ", lambda s: s[2]["med"]),
                    ("interior (y>0.09) frac>k ", lambda s: s[2]["f_gt_k"]),
                    ("y<=0.05 max              ", lambda s: s[3]["max"]),
                    ("y<=0.05 frac > k         ", lambda s: s[3]["f_gt_k"])]
            for name, sel in spec:
                v = sorted(sel(x) for x in summary)
                print(f"    {name} {v[0]:10.4g} {v[len(v)//2]:10.4g} {v[-1]:10.4g}")
        print()


if __name__ == "__main__":
    main()
