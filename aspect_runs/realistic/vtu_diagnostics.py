#!/usr/bin/env python3
"""Boundary diagnostics of the ASPECT realistic-case run from the VTU output.

ASPECT's `heat flux statistics` postprocessor uses the consistent boundary
flux (CBF) method, which in the first time steps (and during fast transients)
carries a visible non-conduction contribution (measured: Nu_bot = -26 at step 1
of the level-5 run).  This script recomputes Nu from the temperature field
itself:

    Nu_top = -dT/dz|_top * d/dT
    Nu_bot = -dT/dz|_bot * d/dT

with a one-sided quadratic derivative through the three nodes normal to the
boundary (arbitrary node spacing, so AMR meshes are handled), averaged over x.
This is the same definition as MantleSEM's `nu_walls` (x-average of the wall
derivative), evaluated on the ASPECT field.

All MPI pieces of a VTU output are read (`solution-NNNNN.*.vtu`).

Usage:
  python3 vtu_diagnostics.py <solution dir>
"""

import glob
import os
import re
import sys

import numpy as np
import meshio

D = 2.9e6
DT = 1000.0


def lagrange_derivative_at_first(z, t):
    """f'(z0) for the quadratic through (z0,t0),(z1,t1),(z2,t2), arbitrary z."""
    z0, z1, z2 = z
    t0, t1, t2 = t
    l0 = (2 * z0 - z1 - z2) / ((z0 - z1) * (z0 - z2))
    l1 = (z0 - z2) / ((z1 - z0) * (z1 - z2))
    l2 = (z0 - z1) / ((z2 - z0) * (z2 - z1))
    return t0 * l0 + t1 * l1 + t2 * l2


def boundary_gradient(P, T, x, which, tol=1e-3):
    col = np.where(np.abs(P[:, 0] - x) < tol)[0]
    col = col[np.argsort(P[col, 1])]
    z = P[col, 1]
    t = T[col]
    # drop duplicate z values (hanging nodes / piece boundaries)
    keep = np.concatenate(([True], np.diff(z) > 1.0))
    z, t = z[keep], t[keep]
    if len(z) < 3:
        return np.nan
    if which == "bottom":
        return lagrange_derivative_at_first(z[:3], t[:3])
    return -lagrange_derivative_at_first((D - z[-3:])[::-1], t[-3:][::-1])


def read_frames(sol_dir):
    """Return {frame index: (time, points, T)} merged over all MPI pieces."""
    frames = {}
    pat = re.compile(r"solution-(\d+)\.(\d+)\.vtu$")
    for f in sorted(glob.glob(os.path.join(sol_dir, "solution-*.vtu"))):
        m = pat.search(os.path.basename(f))
        if not m:
            continue
        k = int(m.group(1))
        mesh = meshio.read(f)
        pts = mesh.points[:, :2]
        T = mesh.point_data["T"]
        t = float(np.asarray(mesh.field_data["TIME"]).ravel()[0])
        frames.setdefault(k, [t, [], []])
        frames[k][1].append(pts)
        frames[k][2].append(T)
    out = {}
    for k, (t, pts_list, T_list) in frames.items():
        P = np.vstack(pts_list)
        T = np.concatenate(T_list)
        key = np.round(P, 6)
        uniq, idx = np.unique(key, axis=0, return_index=True)
        out[k] = (t, P[idx], T[idx])
    return out


def main():
    sol_dir = sys.argv[1]
    frames = read_frames(sol_dir)
    print(f"{'frame':>5} {'t(s)':>12} {'t_nd':>10} {'Nu_top':>9} {'Nu_bot':>9} "
          f"{'Tmin':>8} {'Tmax':>8} {'Tmean':>8}")
    for k in sorted(frames):
        t, P, T = frames[k]
        xs = np.unique(P[:, 0])
        # 17 columns are enough for the x-average
        idx = np.linspace(0, len(xs) - 1, min(17, len(xs))).astype(int)
        gb = [boundary_gradient(P, T, xs[i], "bottom") for i in idx]
        gt = [boundary_gradient(P, T, xs[i], "top") for i in idx]
        nu_bot = -np.nanmean(gb) * D / DT
        nu_top = -np.nanmean(gt) * D / DT
        t_nd = t * 1e-6 / D ** 2
        print(f"{k:5d} {t:12.4e} {t_nd:10.3e} {nu_top:9.5f} {nu_bot:9.4f} "
              f"{T.min():8.2f} {T.max():8.2f} {T.mean():8.2f}")


if __name__ == "__main__":
    main()
