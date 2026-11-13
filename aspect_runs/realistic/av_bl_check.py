#!/usr/bin/env python3
"""Bottom-boundary-layer inspection of ASPECT VTU frames: what fields are
written, is there an artificial-viscosity (`kappa_eff`) field, how big is it,
and what is the temperature profile in y_nd <= 0.05.

Motivation (docs/dev/VALIDATION.md section 12): ASPECT's boundary
heat flux is a Gresho CBF whose volume integral uses

    kappa_eff = max(k, alpha_AV)        (source/postprocess/heat_flux_map.cc:207-212)

so artificial viscosity in the bottom boundary layer could inflate Nu_bot.
`run_E` was written with `List of output variables = material properties`
only, i.e. no alpha_AV field; `realistic_E6.prm` adds the visualization
plugin `artificial viscosity` (VTU array `artificial_viscosity`, units
W/(m K), source/postprocess/visualization/artificial_viscosity.cc).

deal.II's DataOut writes every Q2 cell as a 2x2 block of bilinear quads, so
the patch height dy is half a cell and the cell level is L = -log2(2*dy/d).
The CBF bottom-face integral only runs over the cells that touch y = 0, so
the `y/d = 0` row below is the one that can enter Nu_bot; the rows above are
reported for context.

Usage:
  python3 av_bl_check.py --solution aspect_runs/realistic/run_E6/solution --frames 0 8 16
  python3 av_bl_check.py --solution aspect_runs/realistic/run_E/solution --frames 0
"""
import argparse
import glob
import os

import numpy as np
import meshio

D = 2.9e6
K = 3.3          # ASPECT thermal conductivity, W/(m K)
LX = 2.5 * D


def read_frame(sol_dir, idx):
    pieces = sorted(glob.glob(os.path.join(sol_dir, f"solution-{idx:05d}.*.vtu")))
    if not pieces:
        raise SystemExit(f"no pieces for frame {idx} in {sol_dir}")
    P, point, cells, t = [], {}, [], None
    for p in pieces:
        m = meshio.read(p)
        P.append(m.points)
        for k, v in m.point_data.items():
            point.setdefault(k, []).append(np.asarray(v).ravel())
        cells.append(m.points[m.cells[0].data])
        if t is None and "TIME" in m.field_data:
            t = float(np.asarray(m.field_data["TIME"]).ravel()[0])
    return np.concatenate(P), {k: np.concatenate(v) for k, v in point.items()}, cells, t


def x_average(x, T):
    o = np.argsort(x)
    x, T = x[o], T[o]
    cov = x[-1] - x[0]
    if cov < 0.9 * LX:
        return None
    return float(np.trapz(T, x) / cov) if hasattr(np, "trapz") else float(np.trapezoid(T, x) / cov)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--frames", nargs="+", type=int, default=[0])
    ap.add_argument("--k", type=float, default=K)
    a = ap.parse_args()

    for idx in a.frames:
        pts, pd, cells, t = read_frame(a.solution, idx)
        print(f"=== frame {idx:5d}  t = {t:.6g} s   t_nd = {t/8.41e18:.4e} ===")
        print("  point fields:", ", ".join(sorted(pd.keys())))
        print("  cell  fields: (none; deal.II writes cell data as node data)")

        # unique vertices (DataOut duplicates points per cell, so dedup for profiles)
        key = np.round(np.stack([pts[:, 0] / D, pts[:, 1] / D], axis=1), 9)
        uniq, first = np.unique(key, axis=0, return_index=True)
        x, y = pts[first, 0], pts[first, 1]
        T = pd["T"][first]
        av_p = pd.get("artificial_viscosity")

        # ---- mesh: patch geometry / inferred cell levels -------------------
        P = np.concatenate(cells)
        dy = P[:, :, 1].max(axis=1) - P[:, :, 1].min(axis=1)
        yc = P[:, :, 1].mean(axis=1)
        lev = np.round(-np.log2(2.0 * dy / D)).astype(int)
        print(f"  patches (4 per Q2 cell): {len(dy)}; levels {sorted(set(lev.tolist()))}")
        for name, sel in (("bottom row (y=0, the CBF cells)", np.abs(yc) < 1e-6 * D),
                          ("rows y<=0.05", (yc > 1e-6 * D) & (yc <= 0.05 * D))):
            if not sel.any():
                continue
            un, cnt = np.unique(lev[sel], return_counts=True)
            hs = {int(k): round(float(np.median(dy[sel][lev[sel] == k])) / D * 2, 6) for k in un}
            print(f"  {name}: n={int(sel.sum())}  level histogram "
                  f"{ {int(k): int(v) for k, v in zip(un, cnt)} }  h/d per level {hs}")

        # ---- artificial viscosity -----------------------------------------
        if av_p is not None:
            av = av_p[first]
            keyy = np.round(y / D, 9)
            for yv in np.unique(keyy):
                if yv > 0.0501:
                    break
                sel = keyy == yv
                if not sel.any():
                    continue
                v = av[sel]
                print(f"    alpha_AV y/d={yv:8.6f}: n={sel.sum():3d} min={v.min():.4g} "
                      f"med={np.median(v):.4g} max={v.max():.4g} W/(m K) -> "
                      f"kappa_eff/k median={np.median(np.maximum(1.0, v / a.k)):.4f} "
                      f"max={np.maximum(1.0, v / a.k).max():.4f}")
            for lo, hi in ((0.02, 0.05), (0.05, 0.09), (0.09, 0.2)):
                sel = (y / D >= lo) & (y / D < hi)
                if sel.any():
                    v = av[sel]
                    print(f"    y/d [{lo:.2f},{hi:.2f}): n={sel.sum():4d} alpha_AV med={np.median(v):.4g} "
                          f"max={v.max():.4g} -> kappa_eff/k med={np.median(np.maximum(1.0, v/a.k)):.3f} "
                          f"max={np.maximum(1.0, v/a.k).max():.3f}")
            print(f"    whole box: alpha_AV min={av.min():.4g} max={av.max():.4g} W/(m K) -> "
                  f"max kappa_eff/k = {np.maximum(1.0, av/a.k).max():.4f}")
            xmax = x[np.argmax(av)] / D
            ymax_ = y[np.argmax(av)] / D
            print(f"    alpha_AV max at (x/d, y/d) = ({xmax:.3f}, {ymax_:.3f})")
        else:
            print("  artificial_viscosity: ABSENT (run_E cannot answer candidate (b))")

        # ---- T profile in the bottom BL -----------------------------------
        print("  x-averaged T in the bottom BL (T_nd = T/1000 - 0.5):")
        keyy = np.round(y / D, 9)
        for yv in np.unique(keyy):
            if yv > 0.0501:
                break
            sel = keyy == yv
            Tx = x_average(x[sel], T[sel])
            if Tx is None:
                continue
            print(f"    y/d = {yv:8.6f}   T = {Tx:9.3f} K   T_nd = {Tx/1000 - 0.5:.6f}   n_x = {sel.sum()}")
        print()


if __name__ == "__main__":
    main()
