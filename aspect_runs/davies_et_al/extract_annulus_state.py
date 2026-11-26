#!/usr/bin/env python3
"""Extract one ASPECT annulus VTU frame as the exact Q2 field on a (r, phi) grid.

This is the annulus twin of `aspect_runs/realistic/extract_aspect_state.py`
(the box extractor).  ASPECT's DataOut writes every Q2 cell as a 2 x 2 block of
**bilinear** sub-quads on the cell's nine Q2 nodes, so the file's own
interpolation is bilinear while the field is quadratic.  On this card the
relevant quantity is the wall radial derivative: with the cell height 1/32 the
bilinear wall slope would be off by ~1e2 per unit r against a wall gradient of
~8, i.e. several percent of the Nusselt number.  The 3x3 patch vertices of one
Q2 cell *are* its nine Q2 nodes (4 corners, 4 edge midpoints, the centre), so
the quadratic is recoverable exactly.  The extractor gathers the nine nodal
values of every cell, de-duplicates the periodically shared copies, and rebuilds
the structured nodal grid

    r_i = R0 + i * (R1 - R0)/(2 * cells_r),      i = 0 .. 2*cells_r
    phi_j = j * 2*pi/(2 * cells_t),              j = 0 .. 2*cells_t - 1

(the seam is one column: phi = 2*pi is phi = 0).  For the Davies et al. (2022)
case 2.2 refinement-4 mesh (192 tangential x 16 radial = 3 072 Q2 cells) that is
33 x 384 = 12 672 nodes, the same node cloud QuaST's 16 x 192 Q2-Q1-Q2 ring
has.

What is verified while extracting (the measured numbers are printed and the
worst ones stored as HDF5 attributes, so a downstream record can quote them):

  * every VTU point lands on a grid node: the radial and angular recovery
    residuals are printed (the point coordinates are Float32, so the floor is
    the Float32 representation error, ~1e-7);
  * every grid node is covered and the copies of a shared node agree (the file
    writes the same node once per incident cell; measured spread 0);
  * the seam column is single-valued: the nodes at phi = 2*pi are stored as
    phi = 0, the grid covers the circle exactly once, and the one-sided angular
    derivatives of the recovered Q2 field agree across the seam (reported as the
    seam derivative jump -- a diagnostic, not a tolerance);
  * the temperature and both velocity components are finite, and the velocity
    really is two-dimensional (components 2/3 nonzero; the out-of-plane one is
    zero).

Output (HDF5, one file per frame):

    /r      float64 (nr)          radial node coordinates (fitted uniform grid)
    /phi    float64 (nphi)        angular node coordinates (nominal uniform grid)
    /T      float64 (nr, nphi)    total nondimensional temperature
    /ux     float64 (nr, nphi)    Cartesian x velocity
    /uy     float64 (nr, nphi)    Cartesian y velocity
    attrs  R0, R1 (fitted grid end radii), cells_r, cells_t, t, frame,
           source_vtu_sha256, plus the provenance/diagnostic attributes listed
           in `main` (source_vtu, n_vtu, n_point, n_cell, recovery_r,
           recovery_phi, duplicate_spread, seam_derivative_jump, r_fit_error,
           field_nu_top, field_nu_bot, field_vrms, field_meanT, field_tmax,
           and -- with --stats -- aspect_t/nu_top/nu_bot/vrms/meanT).

Units: the Davies et al. (2022) / ASPECT benchmark card is **nondimensional**
(d = R1 - R0 = 1 is the length unit, kappa = 1, dT = 1), so t, T and u are
already in the solver's units.  No conversion is applied (contrast the box
extractor, whose card is dimensional).

The wall diagnostics use the *family-A* definition of `semquad_nu_walls` (the
one-sided Q2 radial derivative at the wall divided by the per-wall conductive
reference `-dTbar/dr(r_wall) = 1/(r_wall |ln f|)`), so they are directly
comparable with the imported state's Nusselt pair; ASPECT's own `statistics` file
reports the *family-C* consistent-boundary-flux value instead, which differs by
several percent at this resolution (that difference is the point of the
comparison record, not an error).

A note on the parser duplication.  `scripts/annulus_video.py` already reads
these VTUs (`class AspectSource`); the renderer's reader is not a module, it
lives inside a 1 400-line script whose conventions (dedup by rounded
coordinate, per-cell quad grouping) are tuned for rendering, and the batch that
delivered it is frozen.  So this extractor duplicates the small amount of VTU
reading it needs instead of refactoring the renderer (which would require
re-rendering the delivered v3 video to prove the output unchanged).  The
duplication is checked: `--cross-check` imports `AspectSource` itself and
verifies that this extractor's recovered (T, u) at every node equals the
renderer's unique point cloud **bit for bit** on the same frame.

Usage:
  python extract_annulus_state.py --vtu <dir or piece.vtu> [--frame N] \
      --out state_t3.h5 [--stats logs/aspect/case22_statistics] [--cross-check]
  python extract_annulus_state.py --vtu <dir> --list

`--vtu` may be an ASPECT output directory (containing `solution/`), a
`solution/` directory, or a single `.vtu` piece (all pieces of that frame are
then read).
"""
import argparse
import glob
import hashlib
import os
import re
import sys

import h5py
import numpy as np

DEFAULT_R0 = 1.22
DEFAULT_R1 = 2.22

# 4-point Gauss for the bulk integrals (|u|^2 * r is degree 5 per direction),
# 3-point Gauss for the quadratic wall traces and their radial derivatives.
GQ, GW = np.polynomial.legendre.leggauss(4)
Q3, W3 = np.polynomial.legendre.leggauss(3)

VERSION = "annulus_state_v1"


# ----------------------------------------------------------------------------
# small Q2 helpers (the nine nodes of a cell: n = 3*b + a, a along r, b along
# phi; same convention as the box extractor)
# ----------------------------------------------------------------------------
def lag(t, i):
    """One-dimensional Q2 Lagrange basis on nodes -1, 0, 1."""
    if i == 0:
        return 0.5 * t * (t - 1.0)
    if i == 1:
        return 1.0 - t * t
    return 0.5 * t * (t + 1.0)


def lag_stack(t):
    """(npts, 3) matrix of the three Q2 basis values."""
    t = np.asarray(t, dtype=np.float64)
    return np.stack([lag(t, 0), lag(t, 1), lag(t, 2)], axis=-1)


# ----------------------------------------------------------------------------
# frame discovery / reading
# ----------------------------------------------------------------------------
PIECE_RE = re.compile(r"solution-(\d+)\.(\d+)\.vtu$")
TIME_RE = re.compile(r'Name="TIME"[^>]*>([^<]+)<')


def resolve_solution_dir(path):
    """Return (solution_dir, root_dir) from a piece, a solution dir or an
    output dir."""
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return os.path.dirname(path), os.path.dirname(os.path.dirname(path))
    sol = os.path.join(path, "solution")
    if os.path.isdir(sol):
        return sol, path
    return path, os.path.dirname(path)


def frame_table(sol_dir):
    """[(frame, t, [piece paths])] sorted by frame index; TIME parsed from the
    XML header of the first piece only (cheap: no meshio read)."""
    pieces = {}
    for f in sorted(glob.glob(os.path.join(sol_dir, "solution-*"))):
        m = PIECE_RE.search(os.path.basename(f))
        if m:
            pieces.setdefault(int(m.group(1)), []).append(f)
    rows = []
    for idx in sorted(pieces):
        t = None
        with open(pieces[idx][0], "r", errors="replace") as fh:
            head = fh.read(8192)
        m = TIME_RE.search(head)
        if m:
            t = float(m.group(1))
        rows.append((idx, t, pieces[idx]))
    return rows


def pick_frame(sol_dir, frame):
    rows = frame_table(sol_dir)
    if not rows:
        raise SystemExit("no solution-*.vtu pieces under %s" % sol_dir)
    if frame is None:
        idx, t, paths = rows[-1]
    else:
        hit = [r for r in rows if r[0] == frame]
        if not hit:
            raise SystemExit("frame %d is not in %s (available: %s .. %s)"
                             % (frame, sol_dir, rows[0][0], rows[-1][0]))
        idx, t, paths = hit[0]
    return idx, t, paths, len(rows)


def read_frame(paths):
    """Merge the pieces of one frame: (points (n,2), T (n,), U (n,2), t, ncomp)."""
    import meshio
    P, T, U = [], [], []
    t = None
    ncomp = None
    for p in paths:
        mesh = meshio.read(p)
        if mesh.field_data and "TIME" in mesh.field_data:
            ti = float(np.asarray(mesh.field_data["TIME"]).ravel()[0])
            if t is not None and abs(ti - t) > 1.0e-12:
                raise SystemExit("pieces of one frame disagree on TIME: "
                                 "%r vs %r" % (t, ti))
            t = ti
        P.append(np.asarray(mesh.points[:, :2], dtype=np.float64))
        T.append(np.asarray(mesh.point_data["T"], dtype=np.float64).ravel())
        vel = np.asarray(mesh.point_data["velocity"], dtype=np.float64)
        if vel.ndim != 2 or vel.shape[1] < 2:
            raise SystemExit("%s: `velocity` has shape %r, expected (n, >=2)"
                             % (p, vel.shape))
        if ncomp is None:
            ncomp = vel.shape[1]
        elif ncomp != vel.shape[1]:
            raise SystemExit("pieces disagree on the velocity components")
        U.append(vel[:, :2])
    return (np.vstack(P), np.concatenate(T), np.vstack(U), t, ncomp)


def sha256_of_files(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# structured-grid recovery
# ----------------------------------------------------------------------------
def cluster_radii(r, r0, r1):
    """Cluster the point radii into the radial node layers.

    The layers are `(nr-1)` uniform gaps ~ (R1-R0)/(nr-1) apart, while the
    Float32 coordinate noise is ~1e-7; a gap threshold of 1e-3 of the radius
    span therefore separates the layers robustly for any sane radial
    resolution.  Returns (nr, r_layer (nr,), dr, index per point (n,)).  The
    node spacing is taken from the card radii `r0`/`r1`, not from the measured
    span: the stored grid is the card grid (see `recover_grid`)."""
    lo, hi = float(r.min()), float(r.max())
    span = hi - lo
    thr = max(1.0e-6, 1.0e-3 * span)
    order = np.argsort(r)
    rs = r[order]
    starts = np.concatenate(([0], np.nonzero(np.diff(rs) > thr)[0] + 1))
    ends = np.concatenate((starts[1:], [len(rs)]))
    layer_mean = np.array([rs[a:b].mean() for a, b in zip(starts, ends)])
    nr = len(layer_mean)
    # node spacing from the card's radii (validated below against the measured
    # layers): the stored grid is the card grid, not the Float32-noised one
    dr = (r1 - r0) / (nr - 1)
    idx = np.rint((r - layer_mean[0]) / dr).astype(np.int64)
    return nr, layer_mean, dr, idx


def recover_grid(P, T, U, r0_expected, r1_expected):
    """Recover the (nr, nphi) structured nodal grid from a merged point cloud.

    Returns a dict with the grid arrays, the measured recovery errors and the
    per-point grid indices (used by `--cross-check`).
    """
    r = np.hypot(P[:, 0], P[:, 1])
    phi = np.mod(np.arctan2(P[:, 1], P[:, 0]), 2.0 * np.pi)
    nr, r_layer, dr, i = cluster_radii(r, r0_expected, r1_expected)
    if nr < 3 or nr % 2 != 1:
        raise SystemExit("recovered %d radial layers; the annulus Q2 grid has "
                         "an odd number (2*cells_r + 1)" % nr)
    cells_r = (nr - 1) // 2
    # The volume file writes nine points per Q2 cell, so the cell count is
    # n_point/9 and the angular cell count follows from the radial one.
    if len(P) % 9 != 0:
        raise SystemExit("%d VTU points is not 9 per Q2 cell (the DataOut "
                         "pattern changed?)" % len(P))
    n_cell = len(P) // 9
    if n_cell % cells_r != 0:
        raise SystemExit("cell count %d is not divisible by cells_r=%d"
                         % (n_cell, cells_r))
    cells_t = n_cell // cells_r
    nphi = 2 * cells_t
    # radius span check against the target card before relying on the indices
    r_meas = r_layer
    span_err = max(abs(r_meas[0] - r0_expected), abs(r_meas[-1] - r1_expected))
    if span_err > 1.0e-4:
        raise SystemExit(
            "recovered radius span [%.6f, %.6f] does not match the annulus "
            "card [%g, %g] (err %.3e); --r0/--r1 set the expected geometry"
            % (r_meas[0], r_meas[-1], r0_expected, r1_expected, span_err))
    # radial residual against the *card* ladder (the Float32 floor).  The state
    # file stores the card grid itself, not the noisy measured coordinates:
    # the VTU coordinates are Float32, so a least-squares fit of the measured
    # layer radii would only reproduce them to ~1e-8; the card's uniform grid
    # (validated by the measured residuals being at the Float32 floor) is the
    # grid the physics actually has, and the same one QuaST's ring builds.
    r_card = r0_expected + np.arange(nr) * dr
    res_r = np.abs(r - r_card[i])
    # angular index with the periodic wrap; the seam column is phi = 0
    dphi = 2.0 * np.pi / nphi
    j = np.rint(phi / dphi).astype(np.int64) % nphi
    res_phi = np.abs(phi - j * dphi)
    res_phi = np.minimum(res_phi, 2.0 * np.pi - res_phi)
    if res_r.max() > 0.25 * dr or res_phi.max() > 0.25 * dphi:
        raise SystemExit(
            "point cloud does not land on the (r, phi) grid: max radial "
            "residual %.3e (dr=%.3e), max angular residual %.3e (dphi=%.3e)"
            % (res_r.max(), dr, res_phi.max(), dphi))
    # gather the nodal values: every grid node must be covered once or more,
    # and the copies of a shared node must agree.
    key = i * nphi + j
    uniq, counts = np.unique(key, return_counts=True)
    expected = nr * nphi
    if len(uniq) != expected:
        raise SystemExit("recovered %d grid nodes, expected %d = %d x %d"
                         % (len(uniq), expected, nr, nphi))
    Tn = np.full((nr, nphi), np.nan)
    Uxn = np.full((nr, nphi), np.nan)
    Uyn = np.full((nr, nphi), np.nan)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    T_ord = T[order]
    U_ord = U[order]
    starts = np.searchsorted(ks, uniq)
    ends = np.searchsorted(ks, uniq, side="right")
    dup_spread = 0.0
    for u, a, b in zip(uniq, starts, ends):
        ii, jj = divmod(int(u), nphi)
        Tn[ii, jj] = T_ord[a:b].mean()
        Uxn[ii, jj] = U_ord[a:b, 0].mean()
        Uyn[ii, jj] = U_ord[a:b, 1].mean()
        if b - a > 1:
            dup_spread = max(dup_spread, float(np.ptp(T_ord[a:b])),
                             float(np.ptp(U_ord[a:b, 0])),
                             float(np.ptp(U_ord[a:b, 1])))
    if not (np.all(np.isfinite(Tn)) and np.all(np.isfinite(Uxn))
            and np.all(np.isfinite(Uyn))):
        raise SystemExit("the recovered grid has uncovered nodes")
    # nominal card grid stored in the state file (see above): the measured
    # layer radii are checked against it, not used as the grid.
    r_grid = r_card.copy()
    r_fit_err = float(np.abs(r_meas - r_grid).max())
    phi_grid = np.arange(nphi) * dphi
    # seam: the angular Q2 field must be one-sided-smooth across phi = 0/2pi.
    # The value is single-valued by construction (one stored column); the
    # derivative jump is the measured diagnostic.  The Q2 derivative of a cell
    # at its right end (eta = +1) is (0.5 f[-1] - 2 f[0] + 1.5 f[+1]) and at its
    # left end (eta = -1) is (-1.5 f[-1] + 2 f[0] - 0.5 f[+1]), with the
    # physical node spacing dphi between the three nodes of a cell.
    T0 = Tn[:, 0]
    T1 = Tn[:, 1]
    T2 = Tn[:, 2]
    # derivative leaving the seam into the first cell (left end of cell 0)
    dTdphi_after = (-1.5 * T0 + 2.0 * T1 - 0.5 * T2) / dphi
    Tn1 = Tn[:, nphi - 1]
    Tn2 = Tn[:, nphi - 2]
    # derivative arriving at the seam from the last cell (right end of cell
    # cells_t-1, whose right node IS the seam column by the periodic wrap)
    dTdphi_before = (0.5 * Tn2 - 2.0 * Tn1 + 1.5 * T0) / dphi
    seam_jump = float(np.abs(dTdphi_before - dTdphi_after).max())
    return dict(
        r=r_grid, phi=phi_grid, T=Tn, ux=Uxn, uy=Uyn,
        nr=nr, nphi=nphi, cells_r=cells_r, cells_t=cells_t,
        i=i, j=j, r_pts=r, phi_pts=phi, n_point=len(P), n_cell=n_cell,
        recovery_r=float(res_r.max()), recovery_phi=float(res_phi.max()),
        duplicate_spread=float(dup_spread),
        r_fit_error=r_fit_err,
        r_measured=(float(r_meas[0]), float(r_meas[-1])),
        seam_derivative_jump=seam_jump,
        n_unique_nodes=len(uniq),
        multiplicity={int(k): int(v) for k, v in
                      zip(*np.unique(counts, return_counts=True))},
        ncomp=int(U.shape[1]),
    )


# ----------------------------------------------------------------------------
# diagnostics on the exact structured Q2 field (family A, the solver's)
# ----------------------------------------------------------------------------
def _cells3(field, nphi):
    """(nr, nphi) node grid -> (cells_r, cells_t, 3, 3) per-cell node values,
    node (a, b) at the radial position 2k+a and the angular column 2m+b
    (periodic in the angular direction)."""
    nr = field.shape[0]
    nc_r, nc_p = (nr - 1) // 2, nphi // 2
    out = np.empty((nc_r, nc_p, 3, 3))       # [cell_r, cell_phi, node_r, node_phi]
    cols = (2 * np.arange(nc_p)[:, None] + np.arange(3)[None, :]) % nphi
    for k in range(nc_r):
        # sub[a, m, b]: node radial a of cell k, node angular b of cell m
        sub = field[2 * k:2 * k + 3][:, cols]
        out[k] = sub.transpose(1, 0, 2)
    return out


def field_diagnostics(grid, r0, r1):
    """Family-A wall Nusselt pair, V_rms, mean/max T of the recovered Q2 field.

    The quadrature is exact for the integrands: 4x4 Gauss for the bulk (the Q2
    field squared times the linear polar Jacobian r), 3-point Gauss for the
    quadratic wall traces.
    """
    nr, nphi = grid["nr"], grid["nphi"]
    cells_r, cells_t = grid["cells_r"], grid["cells_t"]
    h_r = (r1 - r0) / (2.0 * cells_r)      # radial node spacing (cell = 2*h_r)
    cell_r = 2.0 * h_r                     # radial cell width
    h_p = 2.0 * np.pi / cells_t            # angular cell width
    Tc = _cells3(grid["T"], nphi)
    Uxc = _cells3(grid["ux"], nphi)
    Uyc = _cells3(grid["uy"], nphi)
    L = lag_stack(GQ)                                   # (nq, 3)
    Tv = np.einsum("kmpq,ap,bq->kmab", Tc, L, L)
    Uxv = np.einsum("kmpq,ap,bq->kmab", Uxc, L, L)
    Uyv = np.einsum("kmpq,ap,bq->kmab", Uyc, L, L)
    # physical quadrature weight r dr dphi through the separable polar map
    rq = (r0 + np.arange(cells_r)[:, None] * cell_r
          + 0.5 * cell_r * (1.0 + GQ)[None, :])           # (ncr, nq)
    wq = 0.5 * cell_r * 0.5 * h_p * np.outer(GW, GW)       # (nq, nq)
    w3 = (np.broadcast_to(wq, (cells_r, cells_t, *wq.shape))
          * rq[:, None, :, None])
    area = float(w3.sum())
    Tmean = float((w3 * Tv).sum() / area)
    Vrms = float(np.sqrt((w3 * (Uxv ** 2 + Uyv ** 2)).sum() / area))
    Tmax = float(Tv.max())
    Tmin = float(Tv.min())
    lnr = np.log(r1 / r0)
    Lg = lag_stack(Q3)
    # wall rows: radial node 0 (inner) and nr-1 (outer); the radial one-sided
    # Q2 derivative at the wall, then the angular 3-point Gauss trace.
    # `integ` is `int dT/dr dphi` over the wall circle (the r factor of the
    # arc-length element cancels in the per-wall average, see ANNULUS_REFS).
    flux = {}
    for tag, rows in (("inner", [0, 1, 2]), ("outer", [nr - 1, nr - 2, nr - 3])):
        # the three radial nodes of the wall-adjacent cell, for every angular
        # node of every angular cell: (cells_t, 3 radial, 3 angular)
        cols = (2 * np.arange(cells_t)[:, None] + np.arange(3)[None, :]) % nphi
        sub = grid["T"][rows][:, cols]          # (3 radial, cells_t, 3 angular)
        v = sub.transpose(1, 0, 2)
        dTdr = np.empty((cells_t, 3))
        if tag == "inner":
            # one-sided Q2 derivative at the inner wall (nodes at r = r0,
            # r0 + h_r, r0 + 2 h_r; cell width = 2 h_r)
            dTdr = (-3.0 * v[:, 0, :] + 4.0 * v[:, 1, :] - v[:, 2, :]) / cell_r
        else:
            # reversed order: rows[0] = R1, rows[1] = R1 - h_r, rows[2] = R1-2h_r
            dTdr = (3.0 * v[:, 0, :] - 4.0 * v[:, 1, :] + v[:, 2, :]) / cell_r
        integ = float(np.sum(W3[None, :]
                             * np.einsum("ma,qa->mq", dTdr, Lg)
                             * (0.5 * h_p), axis=1).sum())
        flux[tag] = integ
    # Nu_wall = < -dT/dr >_wall / (-dTbar/dr(r_wall)) = -I * r_wall * lnr/(2 pi)
    nu_bot = -flux["inner"] * r0 * lnr / (2.0 * np.pi)
    nu_top = -flux["outer"] * r1 * lnr / (2.0 * np.pi)
    return dict(nu_top=float(nu_top), nu_bot=float(nu_bot),
                vrms=Vrms, meanT=Tmean, tmax=Tmax, tmin=Tmin, area=area)


# ----------------------------------------------------------------------------
# ASPECT raw statistics row
# ----------------------------------------------------------------------------
def parse_statistics_row(path, frame, t_hdr, r0=DEFAULT_R0, r1=DEFAULT_R1):
    """The ASPECT statistics row of one visualization output, converted to the
    benchmark's nondimensional Nu pair, V_rms and mean T.

    Two ways to find the row, in order:

      1. the statistics table records the visualization file name
         (`.../solution-NNNNN`) per output; the row whose name ends in
         `solution-%05d` is *this* frame's row.  Its `Time (seconds)` column is
         the exact output time: the VTU's own `TIME` field is Float32 (about
         six significant digits), so for a frame at t = 2.002962604 it says
         `2.00296` -- the statistics row is the more accurate time and the
         extractor stores it as the state's `t`.
      2. otherwise the row nearest to the VTU header time (within 1e-4
         relative) is used, and the VTU time is kept.

    The conversion is the one the renderer uses: for the nondimensional Davies
    card the outward heat flux column is `Q = int k dT/dn ds`, so the family-C
    Nusselt numbers are `Nu_top = Q_top |ln f|/(2 pi)` and
    `Nu_bot = -Q_bot |ln f|/(2 pi)`.  `ln f` is `|ln(R0/R1)|` as in
    ANNULUS_REFS.
    """
    names, rows = {}, []
    with open(path, errors="replace") as fh:
        for line in fh:
            s = line.strip()
            m = re.match(r"#\s*(\d+):\s*(.*)", s)
            if m:
                names[int(m.group(1))] = m.group(2).strip()
            elif s:
                rows.append(s.split())
    if not names or not rows:
        return None

    def col(tag):
        for k, name in sorted(names.items()):
            if name.startswith(tag):
                return k - 1
        raise KeyError(tag)

    try:
        c_t = col("Time (seconds)")
        c_v = col("RMS velocity")
        c_tavg = col("Average temperature")
        c_qb = col("Outward heat flux through boundary with indicator 0")
        c_qt = col("Outward heat flux through boundary with indicator 1")
    except KeyError as e:
        return None
    c_viz = None
    for k, name in sorted(names.items()):
        if name.startswith("Visualization file name"):
            c_viz = k - 1
            break
    best = None
    tag = "solution-%05d" % frame
    for v in rows:
        try:
            tt = float(v[c_t])
        except (ValueError, IndexError):
            continue
        if c_viz is not None and len(v) > c_viz:
            if v[c_viz].strip('"').endswith(tag):
                best = (0.0, v, "visualization-name")
                break
        d = abs(tt - t_hdr)
        if best is None or d < best[0]:
            best = (d, v, "nearest-time")
    if best is None:
        return None
    d, v, how = best
    if how == "nearest-time" and d > 1.0e-4 * max(1.0, abs(t_hdr)):
        return None
    lnf = abs(np.log(r0 / r1))
    qb, qt = float(v[c_qb]), float(v[c_qt])
    return dict(aspect_t=float(v[c_t]), aspect_nu_top=qt * lnf / (2 * np.pi),
                aspect_nu_bot=-qb * lnf / (2 * np.pi),
                aspect_vrms=float(v[c_v]),
                aspect_meanT=float(v[c_tavg]), aspect_dt=d,
                aspect_time_source=how)


# ----------------------------------------------------------------------------
# cross-check against the renderer's parser
# ----------------------------------------------------------------------------
def cross_check(grid, root_dir, frame):
    """Compare this extractor's node values with `annulus_video.AspectSource`.

    Both read the same meshio point cloud; this verifies that the renderer's
    de-duplication and this extractor's grid reconstruction agree *bit for bit*
    on the frame's nodal T and velocity, i.e. the duplicated VTU reading is not
    a second, divergent parser.
    """
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(repo, "scripts"))
    try:
        import annulus_video
    except Exception as exc:                        # pragma: no cover
        print("  cross-check: SKIPPED (cannot import scripts/annulus_video.py: "
              "%r)" % (exc,))
        return None
    src = annulus_video.AspectSource(root_dir)
    pos = [k for k, f in enumerate(src.frames) if f["idx"] == frame]
    if not pos:
        print("  cross-check: SKIPPED (renderer does not list frame %d)" % frame)
        return None
    f = src.read(pos[0])
    r = np.hypot(f["x"], f["y"])
    phi = np.mod(np.arctan2(f["y"], f["x"]), 2.0 * np.pi)
    nr, nphi = grid["nr"], grid["nphi"]
    dr = (grid["r"][-1] - grid["r"][0]) / (nr - 1)
    i = np.rint((r - grid["r"][0]) / dr).astype(np.int64)
    dphi = 2.0 * np.pi / nphi
    j = np.rint(phi / dphi).astype(np.int64) % nphi
    dT = np.abs(f["T"] - grid["T"][i, j])
    dX = np.abs(f["x"] - grid["r"][i] * np.cos(grid["phi"][j]))
    dY = np.abs(f["y"] - grid["r"][i] * np.sin(grid["phi"][j]))
    speed = np.hypot(grid["ux"][i, j], grid["uy"][i, j])
    dU = np.abs(f["speed"] - speed)
    print("  cross-check vs scripts/annulus_video.AspectSource: %d unique "
          "renderer points; max|dT|=%.3e max|d|u||=%.3e max|dx|=%.3e "
          "max|dy|=%.3e  n_quads=%d" %
          (len(r), dT.max(), dU.max(), dX.max(), dY.max(), f["n_quads"]))
    return dict(n=len(r), dT=float(dT.max()), dU=float(dU.max()),
                dx=float(dX.max()), dy=float(dY.max()))


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vtu", default=None,
                    help="ASPECT output dir, solution/ dir, or one .vtu piece")
    ap.add_argument("--frame", type=int, default=None,
                    help="frame index (NNNNN); default: the last frame")
    ap.add_argument("--out", default=None, help="output HDF5 state file")
    ap.add_argument("--r0", type=float, default=DEFAULT_R0,
                    help="expected inner radius (default %g)" % DEFAULT_R0)
    ap.add_argument("--r1", type=float, default=DEFAULT_R1,
                    help="expected outer radius (default %g)" % DEFAULT_R1)
    ap.add_argument("--stats", default=None,
                    help="raw ASPECT statistics table: attach the row at the "
                         "frame's time")
    ap.add_argument("--cross-check", action="store_true",
                    help="compare against scripts/annulus_video.AspectSource")
    ap.add_argument("--list", action="store_true",
                    help="list the frames of --vtu and exit")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(sys.argv[1:] if argv is None else argv)
    if a.vtu is None:
        raise SystemExit("--vtu is required (an output dir, a solution/ dir, "
                         "or one .vtu piece)")
    sol_dir, root_dir = resolve_solution_dir(a.vtu)
    rows = frame_table(sol_dir)
    if not rows:
        raise SystemExit("no solution-*.vtu pieces under %s" % sol_dir)
    if a.list:
        print("%s: %d frames under %s" % (root_dir, len(rows), sol_dir))
        for idx, t, paths in rows:
            print("  frame %05d  t=%s  pieces=%d  bytes=%d"
                  % (idx, "%.9f" % t if t is not None else "n/a", len(paths),
                     sum(os.path.getsize(p) for p in paths)))
        return 0
    if a.out is None:
        raise SystemExit("--out is required (unless --list)")
    frame, t_hdr, paths, nframes = pick_frame(sol_dir, a.frame)
    if a.stats is None:
        for cand in (os.path.join(root_dir, "statistics"),
                     os.path.join(os.path.dirname(root_dir), "statistics")):
            if os.path.isfile(cand):
                a.stats = cand
                break
    if not a.quiet:
        print("frame %05d of %d: %d VTU pieces, t header = %r"
              % (frame, nframes, len(paths), t_hdr))
    P, T, U, t, ncomp = read_frame(paths)
    if t is None:
        t = t_hdr
    t = float(t)
    if not a.quiet:
        print("  %d points, %d VTU pieces, velocity components %d"
              % (len(P), len(paths), ncomp))
    for name, arr in (("T", T), ("ux", U[:, 0]), ("uy", U[:, 1])):
        if not np.all(np.isfinite(arr)):
            raise SystemExit("frame %d: %s has non-finite values" % (frame, name))
    if np.any(U[:, 0] != 0.0) and np.any(U[:, 1] != 0.0):
        pass
    else:
        raise SystemExit("frame %d: the velocity is not two-dimensional (one "
                         "component is identically zero)" % frame)
    if ncomp > 2:
        # the third component of the 2-D spherical-shell velocity must vanish
        import meshio as _meshio                           # noqa: F401
        for p in paths:
            mesh = _meshio.read(p)
            vel = np.asarray(mesh.point_data["velocity"], dtype=np.float64)
            if vel.shape[1] > 2 and np.any(vel[:, 2] != 0.0):
                raise SystemExit("%s: the out-of-plane velocity component is "
                                 "nonzero" % p)
    grid = recover_grid(P, T, U, a.r0, a.r1)
    nr, nphi = grid["nr"], grid["nphi"]
    if not a.quiet:
        print("  recovered grid: nr=%d nphi=%d = %d nodes; cells_r=%d "
              "cells_t=%d; point multiplicity %s"
              % (nr, nphi, grid["n_unique_nodes"], grid["cells_r"],
                 grid["cells_t"], grid["multiplicity"]))
        print("  recovery error: r=%.3e (dr=%.3e)  phi=%.3e (dphi=%.3e)  "
              "duplicate spread=%.3e  r ladder fit=%.3e"
              % (grid["recovery_r"],
                 (a.r1 - a.r0) / (2 * grid["cells_r"]), grid["recovery_phi"],
                 2 * np.pi / nphi, grid["duplicate_spread"],
                 grid["r_fit_error"]))
        print("  r measured span [%.9f, %.9f] (card [%g, %g]); seam "
              "derivative jump %.3e" % (grid["r_measured"][0],
                                        grid["r_measured"][1], a.r0, a.r1,
                                        grid["seam_derivative_jump"]))
        print("  T range [%.9f, %.9f]; velocity |u| max %.6e"
              % (grid["T"].min(), grid["T"].max(),
                 np.hypot(grid["ux"], grid["uy"]).max()))
    # radial/angular walls: free-slip normal velocity should be Float32 round-off
    ph = grid["phi"]
    for tag, row, rw in (("inner", 0, a.r0), ("outer", nr - 1, a.r1)):
        ur = (grid["ux"][row] * np.cos(ph)
              + grid["uy"][row] * np.sin(ph)) / rw
        ut = (-grid["ux"][row] * np.sin(ph)
              + grid["uy"][row] * np.cos(ph)) / rw
        if not a.quiet:
            print("  %s wall: max|u_r|=%.3e (free-slip normal velocity) "
                  "max|u_t|=%.3e" % (tag, np.abs(ur).max(), np.abs(ut).max()))
    diag = field_diagnostics(grid, grid["r"][0], grid["r"][-1])
    stats = parse_statistics_row(a.stats, frame, t, a.r0, a.r1) if a.stats else None
    if stats is not None and abs(stats["aspect_t"] - t) > 0.0:
        # the VTU TIME field is Float32 (~6 significant digits); the statistics
        # row carries the exact output time.  Store the exact one.
        if not a.quiet:
            print("  frame time refined from the statistics row: %.9g -> "
                  "%.12g (VTU TIME is Float32)" % (t, stats["aspect_t"]))
        t = stats["aspect_t"]
    if not a.quiet:
        print("  field diagnostics (family A, exact Q2): Nu_top=%.9f "
              "Nu_bot=%.9f Vrms=%.9f meanT=%.12f Tmax=%.9f"
              % (diag["nu_top"], diag["nu_bot"], diag["vrms"], diag["meanT"],
                 diag["tmax"]))
    if stats:
        print("  ASPECT statistics row at t=%.9f (dt=%.1e): Nu_top=%.9f "
              "Nu_bot=%.9f Vrms=%.9f meanT=%.12f"
              % (stats["aspect_t"], stats["aspect_dt"],
                 stats["aspect_nu_top"], stats["aspect_nu_bot"],
                 stats["aspect_vrms"], stats["aspect_meanT"]))
        print("  relative differences (family A field vs ASPECT statistics, "
              "family C for Nu): Nu_top %+.4f%%  Nu_bot %+.4f%%  Vrms "
              "%+.3e  meanT %+.3e" %
              (100 * (diag["nu_top"] - stats["aspect_nu_top"])
               / stats["aspect_nu_top"],
               100 * (diag["nu_bot"] - stats["aspect_nu_bot"])
               / stats["aspect_nu_bot"],
               (diag["vrms"] - stats["aspect_vrms"]) / stats["aspect_vrms"],
               (diag["meanT"] - stats["aspect_meanT"])
               / abs(stats["aspect_meanT"])))
    cc = cross_check(grid, root_dir, frame) if a.cross_check else None
    if cc and (cc["dT"] != 0.0 or cc["dU"] != 0.0):
        raise SystemExit("cross-check FAILED: the renderer parser and this "
                         "extractor disagree (dT=%.3e, d|u|=%.3e)"
                         % (cc["dT"], cc["dU"]))
    sha = sha256_of_files(paths)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with h5py.File(a.out, "w") as f:
        f["r"] = grid["r"]
        f["phi"] = grid["phi"]
        f["T"] = grid["T"]
        f["ux"] = grid["ux"]
        f["uy"] = grid["uy"]
        f.attrs["format"] = VERSION
        f.attrs["R0"] = float(grid["r"][0])
        f.attrs["R1"] = float(grid["r"][-1])
        f.attrs["cells_r"] = int(grid["cells_r"])
        f.attrs["cells_t"] = int(grid["cells_t"])
        f.attrs["t"] = t
        f.attrs["frame"] = int(frame)
        f.attrs["source_vtu_sha256"] = sha
        f.attrs["source_vtu"] = os.path.join(sol_dir, "solution-%05d.*.vtu"
                                             % frame)
        f.attrs["n_vtu"] = int(len(paths))
        f.attrs["n_point"] = int(grid["n_point"])
        f.attrs["n_cell"] = int(grid["n_cell"])
        f.attrs["recovery_r"] = float(grid["recovery_r"])
        f.attrs["recovery_phi"] = float(grid["recovery_phi"])
        f.attrs["duplicate_spread"] = float(grid["duplicate_spread"])
        f.attrs["r_fit_error"] = float(grid["r_fit_error"])
        f.attrs["r0_measured"] = float(grid["r_measured"][0])
        f.attrs["r1_measured"] = float(grid["r_measured"][1])
        f.attrs["seam_derivative_jump"] = float(grid["seam_derivative_jump"])
        f.attrs["field_nu_top"] = float(diag["nu_top"])
        f.attrs["field_nu_bot"] = float(diag["nu_bot"])
        f.attrs["field_vrms"] = float(diag["vrms"])
        f.attrs["field_meanT"] = float(diag["meanT"])
        f.attrs["field_tmax"] = float(diag["tmax"])
        f.attrs["field_tmin"] = float(diag["tmin"])
        f.attrs["velocity_components"] = int(ncomp)
        if stats:
            for k, v in stats.items():
                if isinstance(v, str):
                    f.attrs[k] = v
                else:
                    f.attrs[k] = float(v)
        if cc:
            f.attrs["crosscheck_n"] = int(cc["n"])
            f.attrs["crosscheck_max_dT"] = float(cc["dT"])
            f.attrs["crosscheck_max_dspeed"] = float(cc["dU"])
        f.attrs["extractor"] = "aspect_runs/davies_et_al/extract_annulus_state.py"
    if not a.quiet:
        print("  wrote %s  (%d x %d nodes, t=%.9f, sha256(vtu)=%s)"
              % (a.out, nr, nphi, t, sha[:16] + "..."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
