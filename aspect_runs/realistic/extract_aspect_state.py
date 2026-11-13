#!/usr/bin/env python3
"""Extract one ASPECT VTU frame as the *exact* Q2 field, with diagnostics.

Why not just hand the patch values to the Julia side: deal.II's DataOut writes
every Q2 cell as a 2x2 block of **bilinear** patches, so the file's own
interpolation is bilinear while the field is quadratic.  On the bottom boundary
layer that costs O(h) in the wall gradient: with patch height 1/32 the bilinear
wall slope is off by ~1e3 K per unit y_nd against a wall gradient of 4e3 K/unit,
i.e. ~10-25 % of the quantity the comparison is about -- far too lossy.

The 3x3 patch vertices of one Q2 cell *are* its nine Q2 nodes (4 vertices,
4 edge midpoints, cell centre), so the quadratic is recoverable exactly: group
the four patches of a cell by (level, ix//2, iy//2), read the nine node values
by integer index arithmetic, and evaluate the tensor-product Q2 Lagrange basis.

ASPECT mesh geometry (realistic.prm): 1x1 base cell, `Initial global
refinement = 5`, so level l has h_y = d/2^l, h_x = Lx/2^l (2.5:1 elongated);
deal.II DataOut patches are half a cell in each direction.

Modes
  --frame N --out F.h5     write the nine nodal values per cell (+ checks)
  --series A B --csv F     per-frame diagnostics for frames A..B, no h5
  --probe N                diagnostics of frame N only

Output (HDF5, frame mode):
  /cells/level /cells/ix /cells/iy         int32  Q2 cell level, grid index
  /cells/x0 /cells/y0 /cells/hx /cells/hy  float  cell box (m)
  /cells/T  /cells/ux /cells/uz            float  (ncells, 9) Q2 nodes (K, m/s)
  attrs: time (s), d, kappa, frame, source, npatch, npoint, ncell
Node order: n = 3*j + i at (x0 + i*hx/2, y0 + j*hy/2), i,j = 0,1,2.

The diagnostics are the *same wall-functional family as MantleSEM*: the Q2
normal derivative at the wall, Nu = -<dT/dy>/dT for both walls (so the
conductive state gives Nu_top = +6, Nu_bot = -4, as `semquad_nu_walls` does).
ASPECT's own `statistics` reports the Gresho CBF family instead, which differs
from this one by <= 0.7 % per frame after t_nd > 3e-4
(docs/dev/VALIDATION.md section 13.4); the CBF values stay in the
statistics file and both are quoted in section 16.

Usage:
  python extract_aspect_state.py --solution aspect_runs/realistic/run_E/solution \
      --frame 16 --out /tmp/mspec/asp_frame16.h5
  python extract_aspect_state.py --solution aspect_runs/realistic/run_E_cont/solution \
      --series 17 200 --csv out_julia/case_realistic_E_aspimport/asp_series.csv
"""
import argparse
import glob
import os

import h5py
import meshio
import numpy as np

D = 2.9e6          # box height (m), the scale-set length
KAPPA = 1.0e-6     # thermal diffusivity (m^2/s)
KTH = 3.3          # thermal conductivity W/(m K)  (ASPECT realistic.prm)
LX = 2.5 * D
T0 = 8.41e18       # d^2/kappa, the nondimensionalisation of time (s)
DT = 1000.0        # temperature drop (K)

# 4-point Gauss for the bulk integrals, 3-point for the (quadratic) wall traces
GQ, GW = np.polynomial.legendre.leggauss(4)
Q3, W3 = np.polynomial.legendre.leggauss(3)


def _lag(t, i):
    if i == 0:
        return 0.5 * t * (t - 1.0)
    if i == 1:
        return 1.0 - t * t
    return 0.5 * t * (t + 1.0)


def q2_basis(xi, eta):
    """(3,3,npts) tensor-product Q2 weights, node (i, j)."""
    L = np.stack([_lag(xi, 0), _lag(xi, 1), _lag(xi, 2)])
    M = np.stack([_lag(eta, 0), _lag(eta, 1), _lag(eta, 2)])
    return L, M


def read_patches(sol_dir, idx):
    pieces = sorted(glob.glob(os.path.join(sol_dir, f"solution-{idx:05d}.*.vtu")))
    if not pieces:
        return None
    P, T, U, C, t = [], [], [], [], None
    for p in pieces:
        m = meshio.read(p)
        conn = m.cells[0].data                     # (n,4) quad patches
        assert conn.shape[1] == 4, conn.shape
        P.append(m.points)
        T.append(np.asarray(m.point_data["T"]).ravel())
        U.append(np.asarray(m.point_data["velocity"]))
        C.append(conn)
        if t is None and "TIME" in m.field_data:
            t = float(np.asarray(m.field_data["TIME"]).ravel()[0])
    return P, T, U, C, t


def build_cells(P, T, U, C):
    """Group bilinear patches into Q2 cells; return per-cell 3x3 node values.

    The nine Q2 nodes are recovered by *integer* index arithmetic from the patch
    connectivity, not by coordinate matching: the VTU stores the point
    coordinates in float32 (measured: 0.125 m jitter on coordinates of ~7e6 m),
    so coordinate keys are unreliable, while the patch index (ix//2, iy//2) is
    exact and the 2x2 patch grid puts the local node indices at
    (ix%2) + a, (iy%2) + b for patch-corner a,b in {0,1}.
    """
    LOp, HIp = [], []
    for p, c in zip(P, C):
        q = p[c]
        LOp.append(q.min(axis=1))
        HIp.append(q.max(axis=1))
    lo = np.concatenate(LOp)
    hi = np.concatenate(HIp)
    hx_p = hi[:, 0] - lo[:, 0]
    hy_p = hi[:, 1] - lo[:, 1]
    pl = np.rint(np.log2(D / hy_p)).astype(np.int64)
    assert np.allclose(hy_p, D / 2.0**pl, rtol=1e-4, atol=1e-3), "hy off the ladder"
    assert np.allclose(hx_p, LX / 2.0**pl, rtol=1e-4, atol=1e-3), "hx/hy not 2.5:1"
    ix_p = np.rint(lo[:, 0] / (LX / 2.0**pl)).astype(np.int64)
    iy_p = np.rint(lo[:, 1] / (D / 2.0**pl)).astype(np.int64)
    assert np.all(ix_p >= 0) and np.all(ix_p < 2**pl), "patch x index out of range"
    assert np.all(iy_p >= 0) and np.all(iy_p < 2**pl), "patch y index out of range"

    flat = []
    k0 = 0
    for p, T, U, c in zip(P, T, U, C):
        for k in range(len(c)):
            flat.append(dict(pl=int(pl[k0 + k]), ix=int(ix_p[k0 + k]),
                             iy=int(iy_p[k0 + k]), conn=c[k], p=p, T=T, U=U))
        k0 += len(c)

    groups = {}
    for fp in flat:
        groups.setdefault((fp["pl"] - 1, fp["ix"] // 2, fp["iy"] // 2), []).append(fp)
    for kk, g in groups.items():
        if len(g) != 4:
            raise SystemExit(f"Q2 cell {kk} has {len(g)} patches, expected 4 "
                             "(the VTU is not the 2x2 DataOut pattern)")

    cells = sorted(groups)
    n = len(cells)
    out = dict(level=np.zeros(n, np.int32), ix=np.zeros(n, np.int32),
               iy=np.zeros(n, np.int32), x0=np.zeros(n), y0=np.zeros(n),
               hx=np.zeros(n), hy=np.zeros(n), T=np.zeros((n, 9)),
               ux=np.zeros((n, 9)), uz=np.zeros((n, 9)))
    filled = np.zeros((n, 9), np.int64)
    worst_dup = 0.0
    for c, (l, i, j) in enumerate(cells):
        h_y = D / 2.0**l
        h_x = LX / 2.0**l
        out["level"][c], out["ix"][c], out["iy"][c] = l, i, j
        out["x0"][c], out["y0"][c] = i * h_x, j * h_y
        out["hx"][c], out["hy"][c] = h_x, h_y
        for fp in groups[(l, i, j)]:
            a0 = fp["ix"] % 2
            b0 = fp["iy"] % 2
            for cc, (da, db) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
                gi = fp["conn"][cc]
                a, b = a0 + da, b0 + db
                if not (0 <= a <= 2 and 0 <= b <= 2):
                    raise SystemExit(f"patch corner maps to node ({a},{b}) outside 3x3")
                m = 3 * b + a
                vT = float(fp["T"][gi])
                vx = float(fp["U"][gi, 0])
                vz = float(fp["U"][gi, 1])
                if filled[c, m]:
                    worst_dup = max(worst_dup, abs(vT - out["T"][c, m]),
                                    abs(vx - out["ux"][c, m]), abs(vz - out["uz"][c, m]))
                else:
                    out["T"][c, m], out["ux"][c, m], out["uz"][c, m] = vT, vx, vz
                    filled[c, m] = 1
    if not np.all(filled == 1):
        raise SystemExit(f"{(filled==0).sum()} of {9*n} Q2 nodes never assigned")
    out["dup_err"] = worst_dup
    return out


def frame_diagnostics(cl):
    """Vectorised bulk + wall diagnostics of the exact Q2 field.

    Nu uses the same family as MantleSEM: the Q2 normal derivative at the wall,
    Nu = -<dT/dy_nd>/dT (conductive state: Nu_top = +6, Nu_bot = -4).
    """
    n = len(cl["level"])
    Tn = cl["T"].reshape(n, 3, 3)        # [c, b(y), a(x)]
    Uxn = cl["ux"].reshape(n, 3, 3)
    Uzn = cl["uz"].reshape(n, 3, 3)
    L = np.stack([_lag(GQ, 0), _lag(GQ, 1), _lag(GQ, 2)])      # (3, nq)
    M = L
    # T, ux, uz at the 4x4 Gauss grid: (n, nq, nq)
    Tv = np.einsum("cba,aq,bp->cqp", Tn, L, M)
    Uxv = np.einsum("cba,aq,bp->cqp", Uxn, L, M)
    Uzv = np.einsum("cba,aq,bp->cqp", Uzn, L, M)
    wq = (np.outer(GW, GW)[None, :, :] *
          (0.25 * cl["hx"] * cl["hy"])[:, None, None])   # (n, nq, nq)
    area = wq.sum()
    vol = float(area)
    Tmax = float(Tv.max())
    Tmean = float((wq * Tv).sum() / vol)
    Vrms = float(np.sqrt((wq * (Uxv**2 + Uzv**2)).sum() / vol))
    # wall fluxes (bottom row y0 = 0, top row y0 + hy = d); the 3-point Gauss
    # rule integrates the quadratic wall trace exactly
    hb = cl["hy"]
    Lw = np.stack([_lag(Q3, 0), _lag(Q3, 1), _lag(Q3, 2)], axis=1)   # (nq, 3)
    nu_bot = nu_top = 0.0
    bot = np.where(np.abs(cl["y0"]) < 1e-9)[0]
    if len(bot):
        dTdy = (-3.0 * Tn[bot, 0, :] + 4.0 * Tn[bot, 1, :] - Tn[bot, 2, :]) / hb[bot, None]
        val = np.einsum("qa,ca->cq", Lw, dTdy)
        nu_bot = float(np.sum(W3[None, :] * val * (0.5 * cl["hx"][bot])[:, None]))
    top = np.where(np.abs(cl["y0"] + cl["hy"] - D) < 1e-9)[0]
    if len(top):
        dTdy = (3.0 * Tn[top, 2, :] - 4.0 * Tn[top, 1, :] + Tn[top, 0, :]) / hb[top, None]
        val = np.einsum("qa,ca->cq", Lw, dTdy)
        nu_top = float(np.sum(W3[None, :] * val * (0.5 * cl["hx"][top])[:, None]))
    # Nu = -(d/dT) <dT/dy> with <-> x-average over Lx: the x-integral above is
    # already in meters, so divide by Lx and scale by d/DT
    nu_b = -nu_bot / LX * D / DT
    nu_t = -nu_top / LX * D / DT
    return dict(nu_bot=nu_b, nu_top=nu_t, Vrms_nd=Vrms * D / KAPPA,
                Tmax_nd=Tmax / DT - 0.5, Tmean_nd=Tmean / DT - 0.5,
                Tmax_K=Tmax, Tmean_K=Tmean, area=vol / (LX * D))


def check_reconstruction(cl):
    """Evaluating all nine weights at a node must return that node's value."""
    err = 0.0
    Tn = cl["T"].reshape(len(cl["level"]), 3, 3)
    for c in range(len(cl["level"])):
        for b in range(3):
            for a in range(3):
                xi = -1.0 + a
                et = -1.0 + b
                L, M = q2_basis(np.array([xi]), np.array([et]))
                v = float(np.einsum("ba,a,b->", Tn[c], L[:, 0], M[:, 0]))
                err = max(err, abs(v - Tn[c, b, a]))
    return err


def load(sol_dir, idx):
    r = read_patches(sol_dir, idx)
    if r is None:
        return None
    P, T, U, C, t = r
    cl = build_cells(P, T, U, C)
    return cl, t, sum(len(p) for p in P), sum(len(c) for c in C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--series", nargs=2, type=int, default=None,
                    metavar=("FIRST", "LAST"))
    ap.add_argument("--csv", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    if a.series:
        lo, hi = a.series
        rows = ["frame,t_s,t_nd,ncells,level_min,level_max,nu_bot,nu_top,"
                "Vrms_nd,Tmax_nd,Tmean_nd,Tmax_K,Tmean_K"]
        for idx in range(lo, hi + 1):
            r = load(a.solution, idx)
            if r is None:
                continue
            cl, t, npatch_pts, npatch = r
            d = frame_diagnostics(cl)
            rows.append(f"{idx},{t:.9e},{t/T0:.9e},{len(cl['level'])},"
                        f"{int(cl['level'].min())},{int(cl['level'].max())},"
                        f"{d['nu_bot']:.9f},{d['nu_top']:.9f},{d['Vrms_nd']:.6f},"
                        f"{d['Tmax_nd']:.8f},{d['Tmean_nd']:.8f},"
                        f"{d['Tmax_K']:.4f},{d['Tmean_K']:.4f}")
            if not a.quiet:
                print(rows[-1], flush=True)
        out = a.csv or os.path.join(a.solution, "q2_series.csv")
        with open(out, "w") as f:
            f.write("\n".join(rows) + "\n")
        print(f"wrote {out} ({len(rows)-1} frames)")
        return

    if a.frame is None:
        raise SystemExit("--frame or --series is required")
    r = load(a.solution, a.frame)
    if r is None:
        raise SystemExit(f"no pieces for frame {a.frame} in {a.solution}")
    cl, t, npatch_pts, npatch = r
    levs = sorted(set(cl["level"].tolist()))
    print(f"frame {a.frame}: {npatch_pts} patch points, {npatch} patches, "
          f"t = {t:.6g} s = {t/T0:.6e} t0")
    print(f"  Q2 cells: {len(cl['level'])}; levels {levs}; cells/level " +
          str({int(l): int((cl["level"] == l).sum()) for l in levs}))
    print(f"  node duplicates agree to {cl['dup_err']:.3e} (K or m/s)")
    print(f"  check1 reconstruction at the 9 nodes: max |dT| = "
          f"{check_reconstruction(cl):.3e} K")
    d = frame_diagnostics(cl)
    print(f"  check2 bulk: area/box = {d['area']:.10f}  Tmax = {d['Tmax_K']:.3f} K  "
          f"Tmean = {d['Tmean_K']:.4f} K  Vrms = {d['Vrms_nd']:.2f} nd")
    print(f"  check2 wall gradient: Nu_bot = {d['nu_bot']:+.5f}  "
          f"Nu_top = {d['nu_top']:+.5f}   (conductive reference: -4 / +6)")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with h5py.File(a.out, "w") as f:
            g = f.create_group("cells")
            for k in ("level", "ix", "iy", "x0", "y0", "hx", "hy", "T", "ux", "uz"):
                g[k] = cl[k]
            f.attrs["time"] = float(t)
            f.attrs["d"] = D
            f.attrs["kappa"] = KAPPA
            f.attrs["kth"] = KTH
            f.attrs["frame"] = a.frame
            f.attrs["source"] = os.path.abspath(a.solution)
            f.attrs["npatch"] = npatch
            f.attrs["npoint"] = npatch_pts
            f.attrs["ncell"] = int(len(cl["level"]))
        print(f"  wrote {a.out}  ({len(cl['level'])} cells)")


if __name__ == "__main__":
    main()
