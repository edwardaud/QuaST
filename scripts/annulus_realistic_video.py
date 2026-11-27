#!/usr/bin/env python3
"""Production renderer for the realistic-card-on-the-annulus QuaST-vs-ASPECT clip.

The figure is the canvas of ``docs/assets/quast_vs_aspect_realistic.gif`` (the
owner's contract), drawn by the reference-style layout of
``scripts/annulus_video.py`` (imported, not copied): 1200 px wide, exactly 7
blank rows above the panels, 22 px left margin, four 279 px panels with 13 px
gaps, 6 px between the two rows, the three 9 px colour bars under columns 1-3
and the discrete level legend under column 4, the row names inside the
temperature panels in bold white Computer Modern with a thin dark outline and
every panel's numbers on a 60 % white plate in its lower-right corner.  There is
no header, no title and no per-frame time text: the card and the frame
provenance belong to this renderer's JSON sidecar and to the README caption.

Columns (both rows share one colour scale per column)
-----------------------------------------------------
1. total temperature ``T`` (``inferno``);
2. the angular-mean anomaly ``theta = T - <T>_phi(r)`` on the common polar grid
   (``RdBu_r``, symmetric);
3. ``log10(eta/eta_ref)`` of the card's composite-isostress law (``magma_r``
   with the box clip's power-law colour mapping, gamma = 0.30: the colour is
   warped toward the soft end, the bar labels the true range, dark = stiff =
   cold).  QuaST's value is the solver's own per-node ``log10(eta/eta_ref)``
   field (``frame_NNNNN_eta.bin``); ASPECT's is its saved nodal ``viscosity``
   converted with eta_nd = eta/eta0 (the plugin clamps it to the card's
   [eta_min, eta_max], so the field is usable -- see ``--check``);
4. the adaptive mesh coloured by tree level with white cell boundaries and the
   discrete level legend.

The temperature panel's plate carries ``V_rms``, ``mean T`` and ``T_max`` (the
card has no Nusselt pair: for H = 10 on the ring the per-wall normalisation is
undefined by construction); the anomaly panel its min/max; the viscosity panel
``log10 eta`` min/max; the mesh panel the leaf/cell count and the level range.

Card: R0 = 1.22, R1 = 2.22 (gap d = 1), nominal Ra = 1e5 at the composite
calibration point, composite isostress diffusion+dislocation rheology with the
box card's prefactors, H = 10, two phase transitions (z0 = 0.87/0.78) with
per-phase latent heat, free-slip ring, AMR settings copied from the box card
(QuaST) / the original ASPECT AMR block.  The measured ``Ra_interior`` is read
from the QuaST run's final ``ANNULUS_REALISTIC_RESULT`` line.

Inputs
------
``--quast DIR``   a ``profile = :state`` frame directory (``# annulus_frames_v3``
                  of ``examples/annulus_realistic_demo.jl``): ``frame_%05d.bin``
                  (int64 nnodes, then Float64 x, y, theta, T, ux, uy),
                  ``frame_%05d_eta.bin`` (Float32 log10(eta/eta_ref) at the same
                  nodes), ``frame_%05d_mesh.csv`` (level,x0,y0,hx,hy leaves in
                  logical coordinates) and ``frame_meta.csv``.
``--aspect DIR ..`` ASPECT output directory (or its ``solution/``): the
                  ``solution-NNNNN.PPPP.vtu`` pieces, its ``solution.pvd`` and
                  ``statistics``.  Repeated (and comma-separated) directories are
                  merged in time and de-duplicated, so the two pilot parts can be
                  passed together.  Dimensional shell card -> nondimensional
                  scheme: r_nd = r/d, T_nd = (T - 500)/1000,
                  u_nd = u*d/kappa, eta_nd = eta/eta0.

Field rasterisation
-------------------
Both codes are Q2 on quadtree leaves, so both are evaluated **exactly** with the
Q2 tensor basis: the leaves of a given tree level tile a fixed dyadic lattice in
logical coordinates, so every level is rasterised in one vectorised pass and the
levels are painted from coarse to fine (the finest cell containing each sample
wins).  That gives the fields on the common 400 x 2048 polar grid and the mesh
panel as a level/cell-id raster (one white 1-px cell boundary at the 2x master,
exactly like the reference's 0.42 pt white outlines).  The alternative scipy
Delaunay interpolation of the node cloud is kept only as a cross-check in
``--check`` (measured agreement is reported there): the direct Q2 evaluation is
both faster and error-free.

Outputs
-------
``--out STEM.gif`` (or ``.mp4``): the clip at 1200 px, encoded from 2400 px
masters with Lanczos downsampling; ``--also-poster`` the last frame as a 1200 px
PNG; ``--also-mp4`` the H.264 clip (1200 px and the 2400 px master sibling);
``--sidecar PATH`` the JSON plan (card, AMR settings, Ra, window, per-frame
t/step/delta, scales, eta/law provenance, level-inference validation, no-Nusselt
note, caption); ``--stills N`` / ``--stills-at t1,t2,..`` spot frames as 1200 px
PNGs instead of the clip.

Modes
-----
``--check`` (alias ``--test``) validates the inputs and prints the tables
(inventory, conversions, ASPECT level inference vs its statistics, viscosity
provenance, QuaST Q2-vs-Delaunay, time alignment) without rendering.
``--check-frames N`` sets how many ASPECT frames the expensive checks use.

Example (production, once both runs are complete)
-------------------------------------------------
    /home/szh/dedalus-env/bin/python scripts/annulus_realistic_video.py \
        --quast out_julia/annulus_realistic/quast_prod \
        --aspect out_julia/annulus_realistic/aspect_prod \
        --t0 1e-6 --t1 3.2e-4 --frames 120 \
        --out out_julia/annulus_realistic/render/realistic.gif \
        --also-poster --also-mp4 \
        --sidecar out_julia/annulus_realistic/render/realistic.json

Example (pilot test, the data this renderer was developed on)
-------------------------------------------------------------
    ... --quast out_julia/annulus_realistic/quast_frames_direct120 \
        --aspect /mnt/c/temp/quast_aspect/realistic_annulus/output-annulus-realistic-pilot \
                 /mnt/c/temp/quast_aspect/realistic_annulus/output-annulus-realistic-pilot2 \
        --t0 1e-6 --t1 3e-5 --frames 20 ...
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.colors import (BoundaryNorm, FuncNorm, ListedColormap,  # noqa: E402
                               Normalize, to_rgba)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import annulus_video as av                                        # noqa: E402
try:
    import eta_panel as ep                                        # noqa: E402
except Exception:                                                 # pragma: no cover
    ep = None

# ---------------------------------------------------------------- the card ---
R0, R1 = 1.22, 2.22                    # ring, gap d = R1 - R0 = 1 (physics unit)
D_PHYS = 2.9e6                         # ASPECT length unit, m
KAPPA = 1.0e-6                         # ASPECT thermal diffusivity, m^2/s
DELTA_T = 1000.0                       # temperature scale, K
T_SURF = 500.0                         # outer-wall temperature, K
ETA0 = 2.368635e23                     # ASPECT reference viscosity, Pa s
T_ND = D_PHYS * D_PHYS / KAPPA         # 8.41e18 s, the nondimensional time unit
U_ND = D_PHYS / KAPPA                  # 2.9e12 s/m
RA_NOMINAL = 1.0e5                     # nominal Ra at the calibration point
ETA_GAMMA = 0.30                       # the box clip's power-law colour warp
QUAST_COARSE_NX = 8                    # level-3 ring: 8 radial x 80 angular
QUAST_COARSE_NY = 80
QUAST_LEVEL0 = 3
ASPECT_COARSE_ANG = 12                 # ASPECT's coarse shell: 12 x 1
ETA_MIN_ND, ETA_MAX_ND = 1.0e-4, 1.0e4  # the card's (and plugin's) clamp
T_MESH_LW = 1                          # white boundary width, master pixels

TAU = 2.0 * math.pi
LEVEL_COLOURS = av.REF_LEVEL_COLORS    # the reference mesh panel's palette


def log(msg):
    print("[annulus_realistic] " + msg, flush=True)


def sig(x, n=4):
    return av.sigfig(x, n) if x is not None else "n/a"


# ------------------------------------------------------------------- text ---
def plate_text_inset(ins, font_scale=1.0):
    """Temperature-panel plate: V_rms, mean T and T_max (no Nusselt pair).

    At `font_scale` > 1.1 the three numbers would overflow the 279 px panel, so
    they are split over two lines (the plate grows upward); the default 1.0 is
    the one-line reference layout.
    """
    def num(x, fmt):
        return "n/a" if x is None else fmt % x
    vrms = ("n/a" if ins.get("Vrms") is None
            else r"%s" % av.ref_group(ins["Vrms"]))
    one = (r"$V_{\mathrm{rms}} = %s$   $\mathrm{mean}\,T = %s$   "
           r"$T_{\mathrm{max}} = %s$"
           % (vrms, num(ins.get("meanT"), "%.4f"),
              num(ins.get("Tmax"), "%.4f")))
    if font_scale <= 1.1:
        return one
    return ((r"$V_{\mathrm{rms}} = %s$   $\mathrm{mean}\,T = %s$" + "\n" +
             r"$T_{\mathrm{max}} = %s$")
            % (vrms, num(ins.get("meanT"), "%.4f"),
               num(ins.get("Tmax"), "%.4f")))


def eta_norm(vmin, vmax, gamma=ETA_GAMMA):
    """Power-law colour norm over the TRUE log10 eta range (box-clip mapping)."""
    lo, hi = float(vmin), float(vmax)
    span = max(hi - lo, 1.0e-9)

    def forward(x):
        return np.clip((np.asarray(x) - lo) / span, 0.0, 1.0) ** gamma

    def inverse(y):
        return lo + span * np.clip(np.asarray(y), 0.0, 1.0) ** (1.0 / gamma)

    return FuncNorm((forward, inverse), vmin=lo, vmax=hi)


def level_legend(lev_min, lev_max):
    nlev = lev_max - lev_min + 1
    palette = LEVEL_COLOURS[:nlev] if nlev <= len(LEVEL_COLOURS) else None
    if palette is not None:
        base = [to_rgba(c) for c in palette]
    else:
        base = plt.get_cmap("turbo")(np.linspace(0.08, 0.92, nlev))
    cmap = ListedColormap(base)
    norm = BoundaryNorm(np.arange(lev_min - 0.5, lev_max + 1.5, 1.0), nlev)
    return dict(cmap=cmap, norm=norm, ticks=list(range(lev_min, lev_max + 1)),
                title="AMR mesh, colour = level (%d-%d)" % (lev_min, lev_max))


# ------------------------------------------------------ Q2 lattice machinery -
RING3 = np.array([0, 1, 2, 5, 8, 7, 6, 3])     # 3x3 row-major lattice boundary


def basis3(x):
    """Quadratic Lagrange basis on the three nodes 0, 1/2, 1 of [0, 1]."""
    x = np.asarray(x, dtype="float64")
    return ((1.0 - x) * (1.0 - 2.0 * x), 4.0 * x * (1.0 - x),
            x * (2.0 * x - 1.0))


def q2_eval_levels(levels, r_pts, phi_pts, names=None):
    """Exact Q2 evaluation of every field on the given polar points.

    `levels` is a coarse-to-fine list of dicts with `level`, `n_rad`, `n_ang`,
    `idx` (int32 (n_rad, n_ang), cell id or -1) and `values` (a dict of
    (ncells, 3, 3) arrays, radial basis index 0 = inner).  The finest cell
    containing each point wins (the passes are painted coarse -> fine).
    Returns (fields dict, level map, cell map).
    """
    n = len(r_pts)
    all_names = sorted(levels[0]["values"]) if levels else []
    names = all_names if names is None else list(names)
    out = {k: np.full(n, np.nan) for k in names}
    lev_map = np.full(n, -1, dtype=np.int16)
    cell_map = np.full(n, -1, dtype=np.int64)
    S = (np.asarray(r_pts, dtype="float64") - R0) / (R1 - R0)
    Th = np.mod(np.asarray(phi_pts, dtype="float64"), TAU) / TAU
    for lev in levels:
        nr, na = int(lev["n_rad"]), int(lev["n_ang"])
        idx = lev["idx"]
        u = S * nr
        iv = np.floor(u).astype(np.int64)
        np.clip(iv, 0, nr - 1, out=iv)
        fu = np.clip(u - iv, 0.0, 1.0)
        v = Th * na
        jv = np.floor(v).astype(np.int64)
        jv %= na
        fv = np.clip(v - np.floor(v), 0.0, 1.0)
        cid = idx[iv, jv]
        valid = cid >= 0
        if not valid.any():
            continue
        bu = basis3(fu[valid])
        bv = basis3(fv[valid])
        inds = cid[valid]
        for name in names:
            V = lev["values"][name][inds]
            res = V[:, 0, 0] * bu[0] * bv[0]
            for a in range(3):
                for b in range(3):
                    if a == 0 and b == 0:
                        continue
                    res += V[:, a, b] * bu[a] * bv[b]
            out[name][valid] = res
        lev_map[valid] = lev["level"]
        cell_map[valid] = (np.int64(lev["level"]) * 100000000
                           + iv[valid] * na + jv[valid])
    return out, lev_map, cell_map


def raster_mesh_panel(levels, level_cmap, level_norm, panel_n, r_box,
                      px_per_unit=None, min_cell_px=0.67, supersample=1,
                      line_px=None):
    """RGBA image of the mesh panel: level fill + white cell boundaries.

    Both codes' leaves at one level tile a dyadic lattice in logical (S, T)
    coordinates, so the panel is painted per level on the panel's own pixel
    grid (the finest level wins) and the boundaries are the pixels where the
    level or the cell id differs from a 4-neighbour.  The boundary is
    `line_px` master pixels wide (default 1); the caller scales it with the
    master scale so the delivered outline weight stays the reference's
    ~0.5 px at 1200, and `supersample` renders the raster at that multiple of
    the master panel and Lanczos-downsamples it.

    A cell smaller than `min_cell_px` *delivered* (1200 px canvas) pixels is
    colour-filled without its own outline: at the delivered panel size a
    level-7 cell of this card is ~0.5 px across, so outlining all ~160 000 of
    them would fill the mesh panel white and hide the level colour.  The
    level-boundary contours are always drawn, so the refinement regions stay
    readable; panels whose cells are larger keep the reference's white grid.
    The threshold and the per-level decision are reported in the sidecar.
    """
    if px_per_unit is None:
        px_per_unit = panel_n / (2.0 * r_box)
    ss = max(1, int(supersample))
    n = int(round(panel_n * ss))
    if line_px is None:
        line_px = ss
    line_px = max(1, int(round(line_px * ss)))
    xs = -r_box + (np.arange(n) + 0.5) * (2.0 * r_box / n)
    X, Y = np.meshgrid(xs, xs)                     # row = y, col = x
    r = np.hypot(X, Y).astype("float32")
    phi = np.empty_like(r)
    np.arctan2(Y.astype("float32"), X.astype("float32"), out=phi)
    del X, Y
    np.mod(phi, np.float32(TAU), out=phi)
    inside = (r >= R0) & (r <= R1)
    S = ((r - np.float32(R0)) / np.float32(R1 - R0)).astype("float32")
    Th = (phi / np.float32(TAU)).astype("float32")
    del phi, r
    lev_map = np.full((n, n), -1, dtype=np.int16)
    cid_map = np.full((n, n), -1, dtype=np.int32)
    drawable = []
    for lev in levels:
        nr, na = int(lev["n_rad"]), int(lev["n_ang"])
        size_px = min(px_per_unit * (R1 - R0) / nr,
                      px_per_unit * R0 * TAU / na)
        drawable.append(bool(size_px >= min_cell_px))
        u = S * nr
        iv = np.floor(u).astype(np.int64)
        np.clip(iv, 0, nr - 1, out=iv)
        v = Th * na
        jv = np.floor(v).astype(np.int64)
        jv %= na
        cid = lev["idx"][iv, jv]
        valid = cid >= 0
        if not valid.any():
            continue
        lev_map[valid] = lev["level"]
        cid_map[valid] = (np.int32(lev["level"]) * 100000000
                          + iv[valid].astype(np.int32) * na
                          + jv[valid].astype(np.int32))
    inside &= (lev_map >= 0)
    nlev = int(lev_map.max()) + 1
    colours = np.full((max(nlev, 1), 3), 255, dtype=np.uint8)
    for lev in levels:
        colours[lev["level"]] = np.asarray(
            level_cmap(level_norm(lev["level"]))[:3], dtype="float64") * 255.0
    img = np.full((n, n, 3), 255, dtype=np.uint8)
    img[inside] = colours[lev_map[inside]]
    del colours, S, Th

    def differing(M):
        e = np.zeros(M.shape, dtype=bool)
        d = M[:-1, :] != M[1:, :]
        e[:-1, :] |= d
        e[1:, :] |= d
        d = M[:, :-1] != M[:, 1:]
        e[:, :-1] |= d
        e[:, 1:] |= d
        return e

    # always draw the level contours; draw a cell's own white outline only if
    # that level's cells are at least `min_cell_px` delivered pixels across
    draw_at = np.zeros((n, n), dtype=bool)
    for lev, ok in zip(levels, drawable):
        if ok:
            draw_at |= (lev_map == lev["level"])
    edge = inside & (differing(lev_map)
                     | (differing(cid_map) & draw_at))
    del draw_at, cid_map, lev_map
    # widen the 1-px boundary to the requested delivered line width (a
    # 4-neighbour dilation; keeps the outline weight scale-independent)
    for _ in range(line_px - 1):
        nxt = edge.copy()
        nxt[1:, :] |= edge[:-1, :]
        nxt[:-1, :] |= edge[1:, :]
        nxt[:, 1:] |= edge[:, :-1]
        nxt[:, :-1] |= edge[:, 1:]
        edge = nxt
    img[edge] = 255
    from PIL import Image
    out = Image.fromarray(img).resize((panel_n, panel_n), Image.LANCZOS)
    return np.asarray(out, dtype=np.uint8), drawable


# ================================================================ QuaST row ==
class QuastRealistic:
    """A ``profile = :state`` realistic-annulus QuaST frame directory (v3)."""

    kind = "quast"

    def __init__(self, path):
        path = os.path.abspath(path)
        if not os.path.isfile(os.path.join(path, "frame_meta.csv")):
            alt = os.path.join(path, "frames")
            if os.path.isfile(os.path.join(alt, "frame_meta.csv")):
                path = alt
        self.dir = path
        metapath = os.path.join(path, "frame_meta.csv")
        if not os.path.isfile(metapath):
            raise SystemExit("%s: no frame_meta.csv" % path)
        header, rows = [], []
        with open(metapath, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    header.append(s)
                    continue
                rows.append(s)
        self.meta = dict(version=None, nx=QUAST_COARSE_NX, ny=QUAST_COARSE_NY,
                         R0=R0, R1=R1, Phi=TAU, nr=None, nphi=None)
        for h in header:
            m = re.match(r"#\s*annulus_frames_v(\d+)", h)
            if m:
                self.meta["version"] = int(m.group(1))
            m = re.match(r"#\s*mesh\s+(.*)", h)
            if m:
                for tok in m.group(1).split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        self.meta[k] = (int(v) if k in ("nx", "ny")
                                        else float(v))
            m = re.match(r"#\s*polar_grid\s+(.*)", h)
            if m:
                for tok in m.group(1).split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        if k in ("nr", "nphi"):
                            self.meta[k] = int(v)
        if self.meta["version"] != 3:
            raise SystemExit(
                "%s: frame stream version %r, the realistic clip needs the v3 "
                "state profile (# annulus_frames_v3)" % (metapath,
                                                        self.meta["version"]))
        if not rows or not rows[0].startswith("frame,"):
            raise SystemExit("%s: no CSV header row" % metapath)
        cols = rows[0].split(",")
        self.meta["columns"] = cols
        need = ["frame", "step", "t", "Vrms", "meanT", "Tmin", "Tmax", "nnodes",
                "leaves", "level_min", "level_max", "dt", "umax", "eta_interior"]
        missing = [k for k in need if k not in cols]
        if missing:
            raise SystemExit("%s: v3 frame_meta is missing columns %s"
                             % (metapath, missing))
        self.frames = []
        for line in rows[1:]:
            v = line.split(",")
            if len(v) != len(cols):
                continue
            d = {}
            for k, x in zip(cols, v):
                try:
                    d[k] = (int(float(x)) if k in (
                        "frame", "step", "nnodes", "leaves", "level_min",
                        "level_max") else float(x))
                except ValueError:
                    d[k] = None
            f = d.get("frame")
            if f is None:
                continue
            n = int(d["nnodes"])
            binp = os.path.join(self.dir, "frame_%05d.bin" % f)
            etp = os.path.join(self.dir, "frame_%05d_eta.bin" % f)
            msp = os.path.join(self.dir, "frame_%05d_mesh.csv" % f)
            if not os.path.isfile(binp) or os.path.getsize(binp) != 8 + 6 * 8 * n:
                continue
            if not os.path.isfile(etp) or os.path.getsize(etp) != 4 * n:
                continue
            if not os.path.isfile(msp):
                continue
            d["bin"] = binp
            d["eta_bin"] = etp
            d["mesh_csv"] = msp
            self.frames.append(d)
        self.frames.sort(key=lambda f: f["t"])
        self.times = [f["t"] for f in self.frames]
        if not self.frames:
            raise SystemExit("%s: no complete v3 frames (bin + eta + mesh)"
                             % path)
        self._cache = {}
        self._mesh = {}
        self._levels = {}

    def __len__(self):
        return len(self.frames)

    def describe(self, i):
        f = self.frames[i]
        return "frame %d, step %d" % (f["frame"], f["step"])

    def read(self, i):
        if i in self._cache:
            return self._cache[i]
        f = self.frames[i]
        with open(f["bin"], "rb") as fh:
            n = int(np.fromfile(fh, dtype=np.int64, count=1)[0])
            raw = np.fromfile(fh, dtype=np.float64)
        if raw.size != 6 * n or n != int(f["nnodes"]):
            raise SystemExit("%s: expected %d nodes x 6 arrays, got %d floats"
                             % (f["bin"], n, raw.size))
        a = raw.reshape(6, n)
        eta = np.fromfile(f["eta_bin"], dtype=np.float32).astype("float64")
        out = dict(x=a[0], y=a[1], theta=a[2], T=a[3], ux=a[4], uy=a[5],
                   logeta=eta, n=n)
        if len(self._cache) > 6:
            self._cache.clear()
        self._cache[i] = out
        return out

    def mesh(self, i):
        if i in self._mesh:
            return self._mesh[i]
        f = self.frames[i]
        lev, x0, y0, hx, hy = [], [], [], [], []
        with open(f["mesh_csv"], "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                v = s.split(",")
                if len(v) != 5:
                    continue
                lev.append(int(v[0]))
                x0.append(float(v[1]))
                y0.append(float(v[2]))
                hx.append(float(v[3]))
                hy.append(float(v[4]))
        if not lev:
            raise SystemExit("%s: empty mesh dump" % f["mesh_csv"])
        m = dict(lev=np.asarray(lev, dtype="int64"),
                 x0=np.asarray(x0), y0=np.asarray(y0),
                 hx=np.asarray(hx), hy=np.asarray(hy))
        if int(f["leaves"]) != len(lev):
            raise SystemExit("%s: %d leaves on disk, meta says %d"
                             % (f["mesh_csv"], len(lev), int(f["leaves"])))
        if (int(m["lev"].min()) != int(f["level_min"])
                or int(m["lev"].max()) != int(f["level_max"])):
            raise SystemExit("%s: levels %d..%d on disk, meta says %d..%d"
                             % (f["mesh_csv"], m["lev"].min(), m["lev"].max(),
                                f["level_min"], f["level_max"]))
        self._mesh[i] = m
        return m

    def q2_levels(self, i):
        """Per-level Q2 cell lattices (exact evaluation of the solver's field)."""
        if i in self._levels:
            return self._levels[i]
        from scipy.spatial import cKDTree
        f = self.read(i)
        m = self.mesh(i)
        tree = cKDTree(np.column_stack([f["x"], f["y"]]))
        S3 = np.array([0.0, 0.5, 1.0])
        levels = []
        for L in sorted(set(int(x) for x in np.unique(m["lev"]))):
            nr = QUAST_COARSE_NX * 2 ** (L - QUAST_LEVEL0)
            na = QUAST_COARSE_NY * 2 ** (L - QUAST_LEVEL0)
            sel = np.where(m["lev"] == L)[0]
            ir = np.round(m["x0"][sel] * nr).astype(np.int64)
            ja = np.round(m["y0"][sel] * na).astype(np.int64) % na
            dev = max(np.abs(m["x0"][sel] * nr - ir).max(),
                      np.abs(m["y0"][sel] * na - ja).max())
            if dev > 1.0e-6:
                raise SystemExit("%s: leaves of level %d are not on the "
                                 "level's %d x %d lattice (dev %.2e)"
                                 % (f["mesh_csv"], L, nr, na, dev))
            idx = np.full((nr, na), -1, dtype=np.int32)
            idx[ir, ja] = np.arange(len(sel), dtype=np.int32)
            # the nine Q2 nodes of every leaf, exactly where the solver put them
            Sn = (m["x0"][sel][:, None, None]
                  + m["hx"][sel][:, None, None] * S3[None, :, None])
            Tn = (m["y0"][sel][:, None, None]
                  + m["hy"][sel][:, None, None] * S3[None, None, :])
            Rn = R0 + (R1 - R0) * Sn
            Pn = TAU * Tn
            xy = np.stack([Rn * np.cos(Pn), Rn * np.sin(Pn)],
                          axis=-1).reshape(-1, 2)
            dist, node = tree.query(xy)
            if dist.max() > 1.0e-9:
                raise SystemExit(
                    "%s: leaf Q2 node missing from the node cloud (max "
                    "distance %.2e); this frame stream disagrees with its own "
                    "mesh dump" % (f["mesh_csv"], float(dist.max())))
            node = node.reshape(len(sel), 3, 3)
            levels.append(dict(
                level=L, n_rad=nr, n_ang=na, idx=idx, cells=len(sel),
                values=dict(T=f["T"][node], logeta=f["logeta"][node])))
        self._levels.clear()
        self._levels[i] = levels
        return levels

    def mesh_meta(self, i):
        f = self.frames[i]
        return dict(leaves=int(f["leaves"]), cells=int(f["leaves"]),
                    level_min=int(f["level_min"]),
                    level_max=int(f["level_max"]),
                    nodes=int(f["nnodes"]))

    def nodal_extent(self, i):
        """True range of the nodal fields (the colour-scale reference)."""
        f = self.read(i)
        return dict(T=(float(f["T"].min()), float(f["T"].max())),
                    logeta=(float(f["logeta"].min()),
                            float(f["logeta"].max())))

    def inset(self, i):
        f = self.frames[i]
        return dict(Vrms=float(f["Vrms"]), meanT=float(f["meanT"]),
                    Tmax=float(f["Tmax"]), eta_interior=float(f["eta_interior"]),
                    source="frame_meta.csv", describe="t=%.6g" % f["t"])


# =============================================================== ASPECT row ==
def aspect_geometry(P):
    """Cells' tree level and dyadic lattice position from their nine Q2 nodes.

    `P` is (n, 9, 2) nondimensional (r*cos phi, r*sin phi) with the 3x3 node
    lattice in row-major order (radial index 0 = inner edge).  The level is
    inferred from the polar rectangle's area and mid radius,

        A / r_mid = (2 pi / 12) / 4**L   ->   L = 0.5 log2((2 pi/12)/(A/r_mid)),

    independent of the card's refinement settings; the dyadic radial/angular
    bounds and the 3x3 node lattice positions then validate the inference.
    Returns dict(level, kr, kphi, i_idx, j_idx, dev) for all cells.
    """
    r = np.hypot(P[..., 0], P[..., 1])                       # (n, 9)
    ph = np.mod(np.arctan2(P[..., 1], P[..., 0]), TAU)
    phm = np.arctan2(np.sin(ph).mean(axis=1), np.cos(ph).mean(axis=1))
    dph = (ph - phm[:, None] + math.pi) % TAU - math.pi      # (n, 9)
    ringp = P[:, RING3, :]
    x, y = ringp[..., 0], ringp[..., 1]
    area = 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1)
                              - y * np.roll(x, -1, axis=1), axis=1))
    rmin = r.min(axis=1)
    rmax = r.max(axis=1)
    rmid = 0.5 * (rmin + rmax)
    lf = 0.5 * np.log2((TAU / ASPECT_COARSE_ANG)
                       / np.maximum(area / np.maximum(rmid, 1.0e-30), 1e-300))
    lev = np.round(lf).astype(np.int64)
    if np.abs(lf - lev).max() > 0.05:
        raise SystemExit("ASPECT level inference is ambiguous (max |Lf-L| "
                         "%.2f) -- unexpected cell geometry" % np.abs(lf - lev).max())
    kr_f = (rmin - R0) * 2.0 ** lev
    dphmin = dph.min(axis=1)
    dphmax = dph.max(axis=1)
    ph_lo = phm + dphmin
    kphi_f = (ph_lo % TAU) / TAU * (ASPECT_COARSE_ANG * 2.0 ** lev)
    dev = max(float(np.abs(kr_f - np.round(kr_f)).max()),
              float(np.abs(kphi_f - np.round(kphi_f)).max()),
              float(np.abs(lf - lev).max()))
    i_idx = np.round(2.0 * (r - rmin[:, None])
                     / np.maximum(rmax - rmin, 1e-30)[:, None]).astype(np.int64)
    j_idx = np.round(2.0 * (dph - dphmin[:, None])
                     / np.maximum(dphmax - dphmin, 1e-30)[:, None]).astype(np.int64)
    ok = (np.sort(i_idx * 3 + j_idx, axis=1)
          == np.arange(9)[None, :]).all()
    if not ok:
        raise SystemExit("ASPECT Q2 3x3 node lattice not recovered")
    return dict(level=lev, kr=np.round(kr_f).astype(np.int64),
                kphi=np.round(kphi_f).astype(np.int64) % (ASPECT_COARSE_ANG
                                                           * 2 ** lev),
                i_idx=i_idx, j_idx=j_idx, dev=dev)


def read_pvd_times(path):
    """[(t_s, relative file name)] from an ASPECT solution.pvd."""
    txt = open(path, "r", encoding="utf-8", errors="replace").read()
    ent = re.findall(r'timestep="([^"]+)"[^>]*file="([^"]+)"', txt)
    out = []
    for t, f in ent:
        try:
            out.append((float(t), f))
        except ValueError:
            continue
    return out


def read_aspect_statistics(path):
    """Rows of an ASPECT statistics table -> list of dicts (SI units)."""
    names, raw = {}, []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            m = re.match(r"#\s*(\d+):\s*(.*)", s)
            if m:
                names[int(m.group(1))] = m.group(2).strip()
            elif s:
                raw.append(s.split())
    idx = {name: k - 1 for k, name in names.items()}
    if "Time (seconds)" not in idx:
        return []
    out = []
    for v in raw:
        try:
            d = dict(t=float(v[idx["Time (seconds)"]]))
            for key, col in (
                    ("cells", "Number of mesh cells"),
                    ("vrms", "RMS velocity (m/s)"),
                    ("umax", "Max. velocity (m/s)"),
                    ("Tmin", "Minimal temperature (K)"),
                    ("Tavg", "Average temperature (K)"),
                    ("Tmax", "Maximal temperature (K)"),
                    ("Tavg_nd", "Average nondimensional temperature (K)")):
                d[key] = float(v[idx[col]]) if col in idx else None
            d["cells"] = int(d["cells"])
            out.append(d)
        except (ValueError, IndexError, KeyError):
            continue
    out.sort(key=lambda d: d["t"])
    return out


class AspectRealistic:
    """One or more ``realistic composite`` ASPECT annulus output directories."""

    kind = "aspect"

    def __init__(self, dirs, stats=None):
        self.dirs = [os.path.abspath(d) for d in dirs]
        frames = []
        stats_rows = []
        for d in self.dirs:
            sol = os.path.join(d, "solution")
            if not os.path.isdir(sol):
                sol = d
            pvd = os.path.join(d, "solution.pvd")
            entries = []
            if os.path.isfile(pvd):
                entries = read_pvd_times(pvd)
            if entries:
                for t_s, rel in entries:
                    stem = os.path.splitext(os.path.basename(rel))[0]
                    pieces = sorted(glob.glob(os.path.join(
                        sol, stem + ".*.vtu")))
                    if not pieces:
                        continue
                    frames.append(dict(t_s=t_s, t=t_s / T_ND, sol_dir=sol,
                                       stem=stem, pieces=pieces))
            else:
                pat = re.compile(r"solution-(\d+)\.(\d+)\.vtu$")
                groups = {}
                for p in sorted(glob.glob(os.path.join(sol,
                                                       "solution-*.vtu"))):
                    m = pat.search(os.path.basename(p))
                    if m:
                        groups.setdefault(int(m.group(1)), []).append(p)
                for idx in sorted(groups):
                    import meshio
                    mesh = meshio.read(groups[idx][0])
                    if "TIME" not in mesh.field_data:
                        raise SystemExit("no TIME field in %s"
                                         % groups[idx][0])
                    t_s = float(np.asarray(mesh.field_data["TIME"]).ravel()[0])
                    frames.append(dict(t_s=t_s, t=t_s / T_ND, sol_dir=sol,
                                       stem="solution-%05d" % idx,
                                       pieces=groups[idx]))
            # statistics
            for cand in ([stats] if stats else []) + [
                    os.path.join(d, "statistics"),
                    os.path.join(sol, "statistics"),
                    os.path.join(os.path.dirname(sol), "statistics")]:
                if cand and os.path.isfile(cand):
                    rows = read_aspect_statistics(cand)
                    if rows:
                        stats_rows.extend(rows)
                        break
        # de-duplicate frames by time (the pilot's two parts overlap)
        frames.sort(key=lambda f: f["t"])
        uniq = []
        for f in frames:
            if uniq and abs(f["t"] - uniq[-1]["t"]) <= 1.0e-12 * max(1.0, f["t"]):
                continue
            uniq.append(f)
        self.frames = uniq
        if not self.frames:
            raise SystemExit("no ASPECT vtu frames under %s" % ", ".join(dirs))
        self.times = [f["t"] for f in self.frames]
        stats_rows.sort(key=lambda r: r["t"])
        su = []
        for r in stats_rows:
            if su and abs(r["t"] - su[-1]["t"]) <= 1.0e-6:
                continue
            su.append(r)
        self.stats = su
        self.stats_paths = []
        for d in self.dirs:
            for cand in (os.path.join(d, "statistics"),
                         os.path.join(d, "solution", "statistics")):
                if os.path.isfile(cand):
                    self.stats_paths.append(cand)
                    break
        self.stats_path = self.stats_paths[0] if self.stats_paths else None
        self._cache = {}
        self._levels = {}

    def __len__(self):
        return len(self.frames)

    def describe(self, i):
        return "solution %s" % self.frames[i]["stem"]

    # -- one VTU output set -> per-cell Q2 data -----------------------------
    def read(self, i):
        import meshio
        if i in self._cache:
            return self._cache[i]
        fr = self.frames[i]
        cells = dict(level=[], kr=[], kphi=[], T=[], logeta=[], u=[])
        pieces = 0
        dev = 0.0
        for p in fr["pieces"]:
            mesh = meshio.read(p)
            pieces += 1
            pd = mesh.point_data
            if "T" not in pd or "viscosity" not in pd:
                raise SystemExit("%s: missing T/viscosity point data" % p)
            P = np.asarray(mesh.points[:, :2], dtype="float64") / D_PHYS
            T = (np.asarray(pd["T"], dtype="float64") - T_SURF) / DELTA_T
            eta = np.asarray(pd["viscosity"], dtype="float64") / ETA0
            logeta = np.log10(np.maximum(eta, 1.0e-30))
            quads = None
            for block in mesh.cells:
                if block.type == "quad":
                    quads = np.asarray(block.data)
                    break
            if quads is None or len(quads) % 4 != 0:
                raise SystemExit("%s: no 2x2 quad groups" % p)
            n = len(quads) // 4
            ids16 = quads.reshape(n, 16)
            s = np.sort(ids16, axis=1)
            base = s[:, 0]
            if ((s - base[:, None]) > 8).any():
                raise SystemExit("%s: VTU cells are not 9 consecutive nodes"
                                 % p)
            ids9 = base[:, None] + np.arange(9, dtype=s.dtype)[None, :]
            Pc = P[ids9]
            g = aspect_geometry(Pc)
            T9 = T[ids9]
            E9 = logeta[ids9]
            # scatter the nine node values onto the 3x3 (radial, angular) grid:
            # sorting by flat = 3*i + j and reshaping is row-major scatter
            ncs = len(ids9)
            flat = np.sort(g["i_idx"] * 3 + g["j_idx"], axis=1)
            if not (flat == np.arange(9)[None, :]).all():
                raise SystemExit("%s: Q2 3x3 node lattice not recovered" % p)
            order = np.argsort(g["i_idx"] * 3 + g["j_idx"], axis=1)
            Tb = np.take_along_axis(T9, order, axis=1).reshape(ncs, 3, 3)
            Eb = np.take_along_axis(E9, order, axis=1).reshape(ncs, 3, 3)
            cells["level"].append(g["level"])
            cells["kr"].append(g["kr"])
            cells["kphi"].append(g["kphi"])
            cells["T"].append(Tb)
            cells["logeta"].append(Eb)
            dev = max(dev, float(g["dev"]))
            if "velocity" in pd:
                vel = np.asarray(pd["velocity"], dtype="float64")[:, :2]
                v = vel[ids9] * U_ND
                # 3x3 blocks of the two velocity components, for the --check
                # composite-law cross-check only
                idx = np.broadcast_to(order[:, :, None], (ncs, 9, 2))
                ub = np.take_along_axis(v, idx, axis=1).reshape(ncs, 3, 3, 2)
                cells["u"].append(ub)
            else:
                cells["u"].append(None)
        out = dict(t=fr["t"], level=np.concatenate(cells["level"]),
                   kr=np.concatenate(cells["kr"]),
                   kphi=np.concatenate(cells["kphi"]),
                   T=np.concatenate(cells["T"]),
                   logeta=np.concatenate(cells["logeta"]),
                   pieces=pieces, dev=dev)
        if cells["u"][0] is not None:
            out["u"] = np.concatenate(cells["u"])
        else:
            out["u"] = None
        if len(self._cache) > 2:
            self._cache.clear()
        self._cache[i] = out
        return out

    def q2_levels(self, i):
        if i in self._levels:
            return self._levels[i]
        f = self.read(i)
        levels = []
        for L in sorted(set(int(x) for x in np.unique(f["level"]))):
            nr = 2 ** L
            na = ASPECT_COARSE_ANG * 2 ** L
            sel = np.where(f["level"] == L)[0]
            idx = np.full((nr, na), -1, dtype=np.int32)
            idx[f["kr"][sel], f["kphi"][sel]] = np.arange(len(sel),
                                                          dtype=np.int32)
            levels.append(dict(
                level=L, n_rad=nr, n_ang=na, idx=idx, cells=len(sel),
                values=dict(T=f["T"][sel], logeta=f["logeta"][sel])))
        self._levels.clear()
        self._levels[i] = levels
        return levels

    def mesh_meta(self, i):
        f = self.read(i)
        lev = f["level"]
        return dict(leaves=None, cells=int(len(lev)),
                    level_min=int(lev.min()), level_max=int(lev.max()),
                    pieces=int(f["pieces"]))

    def nodal_extent(self, i):
        f = self.read(i)
        return dict(T=(float(f["T"].min()), float(f["T"].max())),
                    logeta=(float(f["logeta"].min()),
                            float(f["logeta"].max())))

    def stat_at(self, t):
        if not self.stats:
            return {}
        times = [r["t"] for r in self.stats]
        j = av.nearest_index(times, t * T_ND)
        r = dict(self.stats[j])
        r["delta_s"] = r["t"] - t * T_ND
        return r

    def inset(self, i):
        t = self.frames[i]["t"]
        s = self.stat_at(t)
        if not s:
            return dict(Vrms=None, meanT=None, Tmax=None, source=None,
                        describe="t=%.6g" % t)
        tmax = (s["Tmax"] - T_SURF) / DELTA_T if s.get("Tmax") is not None else None
        return dict(Vrms=(s["vrms"] * U_ND if s.get("vrms") is not None else None),
                    meanT=s.get("Tavg_nd"), Tmax=tmax, cells=s.get("cells"),
                    source="statistics", describe="t=%.6g" % t)

    def predicted_cells(self, i):
        s = self.stat_at(self.frames[i]["t"])
        return s.get("cells") if s else None


# ----------------------------------------------------------- validation -----
def validate_aspect_levels(src, i):
    """Level histogram of a VTU frame against the statistics cell count."""
    f = src.read(i)
    lev = f["level"]
    hist = {int(k): int(v) for k, v in zip(*np.unique(lev, return_counts=True))}
    pred = src.predicted_cells(i)
    return dict(cells=int(len(lev)), stats_cells=pred,
                match=(pred is None or pred == len(lev)), hist=hist,
                level_min=int(lev.min()), level_max=int(lev.max()),
                frac_dev=float(f["dev"]))


def validate_eta_saved(src, i, nsamp=1200, seed=3):
    """Cross-check ASPECT's saved viscosity against the card's composite law.

    The card's QuaST law is ported in scripts/eta_panel.py (geometry-free; the
    depth argument becomes h = (r - R0)/d).  The strain rate at each sampled
    cell centre comes from the Q2 velocity gradient (the VTU carries the
    velocity on the same nine nodes), so this is a genuine independent check of
    the saved field, not a fit.
    """
    if ep is None:
        return None
    f = src.read(i)
    if f.get("u") is None:
        return None
    n = len(f["level"])
    rng = np.random.default_rng(seed)
    sel = rng.choice(n, size=min(nsamp, n), replace=False)
    T = f["T"][sel]
    E = f["logeta"][sel]
    U = f["u"][sel]
    lev = f["level"][sel]
    # centre-node values and the Q2 derivative at the element centre
    Tc = T[:, 1, 1]
    Ec = E[:, 1, 1]
    dL = np.array([-1.0, 0.0, 1.0])       # dL0..dL2 at x = 1/2
    Lc = np.array([0.0, 1.0, 0.0])        # L0..L2 at x = 1/2
    # d/dxi and d/deta of each velocity component (rows: 3x3 node grid)
    dv_dxi = np.einsum("a,sabk,b->sk", dL, U, Lc)
    dv_deta = np.einsum("a,sabk,b->sk", Lc, U, dL)
    # radial and angular extent of each cell (from the inferred polar bounds)
    nr = 2.0 ** lev
    dr = (R1 - R0) / nr
    dphi = TAU / (ASPECT_COARSE_ANG * 2.0 ** lev)
    dv_dr = dv_dxi / dr[:, None]
    dv_dphi = dv_deta / dphi[:, None]
    rr = R0 + (f["kr"][sel] + 0.5) * dr
    pp = (f["kphi"][sel] + 0.5) * dphi
    cs, sn = np.cos(pp), np.sin(pp)
    # Cartesian gradient from polar components: u_x = u_r cos - u_phi sin ...
    # Here velocities are stored as (u_x, u_y) already, so chain-rule the
    # Cartesian components directly.
    dux_dx = dv_dr[:, 0] * cs - dv_dphi[:, 0] * sn / rr
    dux_dy = dv_dr[:, 0] * sn + dv_dphi[:, 0] * cs / rr
    duy_dx = dv_dr[:, 1] * cs - dv_dphi[:, 1] * sn / rr
    duy_dy = dv_dr[:, 1] * sn + dv_dphi[:, 1] * cs / rr
    exx, eyy = dux_dx, duy_dy
    exy = 0.5 * (dux_dy + duy_dx)
    edot = np.sqrt(0.5 * (exx ** 2 + eyy ** 2 + 2.0 * exy ** 2))
    h = (rr - R0) / (R1 - R0)
    eta_law = ep.eta_composite(Tc, edot, h)
    d = np.abs(np.log10(np.maximum(eta_law, 1e-30)) - np.log10(
        np.maximum(10.0 ** Ec, 1e-30)))
    return dict(n=int(len(sel)), median=float(np.median(d)),
                p90=float(np.percentile(d, 90)), max=float(d.max()),
                frac_clamped=float(np.mean(np.abs(Ec - math.log10(ETA_MAX_ND))
                                           < 1e-6)),
                eta_law_min=float(eta_law.min()),
                eta_law_max=float(eta_law.max()))


def validate_quast_q2(src, i, npts=3000, seed=5):
    """Q2 direct evaluation against scipy-Delaunay interpolation of the cloud."""
    rng = np.random.default_rng(seed)
    S = rng.random(npts)
    T = rng.random(npts)
    r = R0 + (R1 - R0) * S
    phi = TAU * T
    f = src.read(i)
    levels = src.q2_levels(i)
    vals, _, _ = q2_eval_levels(levels, r, phi)
    xy = np.column_stack([r * np.cos(phi), r * np.sin(phi)])
    interp = av.PolarInterpolator(xy)
    interp.fit(f["x"], f["y"])
    Td = interp.apply(f["T"])
    d = np.abs(vals["T"] - Td)
    return dict(frame=int(src.frames[i]["frame"]), pts=int(npts),
                max=float(d.max()), median=float(np.median(d)),
                p99=float(np.percentile(d, 99)))


def validate_aspect_fields(src, i, npts=1500, seed=11):
    """ASPECT Q2 evaluation and the cell-centred fallback vs a Delaunay check.

    The VTU nodal cloud is de-duplicated and triangulated (scipy), and the two
    renderer candidates are compared against that independent linear
    interpolation on the same state: the Q2 tensor evaluation the renderer uses
    and the per-cell constant the task allows as a fallback.
    """
    import meshio
    import scipy.sparse as sp
    from scipy.spatial import Delaunay
    f = src.read(i)
    levels = src.q2_levels(i)
    rng = np.random.default_rng(seed)
    S = rng.random(npts)
    T = rng.random(npts)
    r = R0 + (R1 - R0) * S
    phi = TAU * T
    xy = np.column_stack([r * np.cos(phi), r * np.sin(phi)])
    vals, _, _ = q2_eval_levels(levels, r, phi)
    # constant-per-cell candidate: repeat the centre node on the 3x3 grid
    lev_const = [dict(lev, values=dict(
        T=np.repeat(np.repeat(lev["values"]["T"][:, 1, 1][:, None, None],
                              3, axis=1), 3, axis=2)))
        for lev in levels]
    cst, _, _ = q2_eval_levels(lev_const, r, phi, names=["T"])
    P, Tn, En = [], [], []
    for p in src.frames[i]["pieces"]:
        mesh = meshio.read(p)
        P.append(np.asarray(mesh.points[:, :2], dtype="float64") / D_PHYS)
        Tn.append((np.asarray(mesh.point_data["T"]) - T_SURF) / DELTA_T)
        En.append(np.log10(np.maximum(
            np.asarray(mesh.point_data["viscosity"]) / ETA0, 1e-30)))
    P = np.vstack(P)
    Tn = np.concatenate(Tn)
    En = np.concatenate(En)
    uid = np.unique(np.round(P, 12), axis=0, return_index=True)[1]
    P, Tn, En = P[uid], Tn[uid], En[uid]
    tri = Delaunay(P)
    si = tri.find_simplex(xy)
    good = si >= 0
    if not good.all():
        xy, vals, cst = xy[good], dict(
            T=vals["T"][good], logeta=vals["logeta"][good]), dict(
            T=cst["T"][good])
        si = tri.find_simplex(xy)
    X = tri.transform[si, :2]
    Y = xy - tri.transform[si, 2]
    bb = np.einsum("ijk,ik->ij", X, Y)
    bary = np.column_stack([bb, 1.0 - bb.sum(axis=1)])
    W = sp.csr_matrix((bary.ravel(), (np.repeat(np.arange(len(si)), 3),
                                      tri.simplices[si].ravel())),
                      shape=(len(si), len(P)))
    out = dict(frame=int(i), t=float(src.times[i]), pts=int(len(si)),
               nodes=int(len(P)))
    for name, cand in (("T", vals["T"]), ("logeta", vals["logeta"])):
        ref = W @ (Tn if name == "T" else En)
        d = np.abs(cand - ref)
        out[name] = dict(max=float(d.max()), median=float(np.median(d)),
                         p99=float(np.percentile(d, 99)))
    dc = np.abs(cst["T"] - (W @ Tn))
    out["cell_constant_T"] = dict(max=float(dc.max()),
                                  median=float(np.median(dc)),
                                  p99=float(np.percentile(dc, 99)))
    return out


# ------------------------------------------------------------------ plan -----
def make_plan(sources, labels, t0, t1, nframes):
    if nframes < 1:
        raise SystemExit("--frames must be >= 1")
    if nframes == 1:
        targets = [float(t0)]
    else:
        targets = list(np.linspace(float(t0), float(t1), int(nframes)))
    plan = []
    for k, t in enumerate(targets):
        picks = []
        for j, s in enumerate(sources):
            i = av.nearest_index(s.times, t)
            picks.append(dict(row=j, label=labels[j], index=i, t=s.times[i],
                              delta=s.times[i] - t))
        plan.append(dict(k=k, target=float(t), picks=picks))
    print("[annulus_realistic] window t = [%s, %s], %d target times"
          % (sig(t0), sig(t1), len(plan)), flush=True)
    span = float(t1) - float(t0)
    for j, (lab, s) in enumerate(zip(labels, sources)):
        deltas = np.array([abs(fr["picks"][j]["delta"]) for fr in plan])
        used = len({fr["picks"][j]["index"] for fr in plan})
        print("  %-8s %5d frames  t = %.6g .. %.6g  %d distinct shown  "
              "max |delta| = %.2e (%.3f%% of the window, %.3g s)"
              % (lab, len(s), s.times[0], s.times[-1], used,
                 deltas.max(), 100.0 * deltas.max() / span if span > 0 else 0.0,
                 deltas.max() * T_ND), flush=True)
        if used < len(plan):
            print("     NOTE: %d target times repeat this row's nearest frame"
                  % (len(plan) - used), flush=True)
    return plan


def plan_table(plan, sources, labels, max_rows=40):
    """A compact per-target alignment table (all rows if short, else sampled)."""
    n = len(plan)
    show = list(range(n)) if n <= max_rows else \
        list(range(0, 8)) + list(range(8, n - 8, max(1, (n - 16) // 12))) + \
        list(range(n - 8, n))
    print("\n  target t          %-26s %-26s" % (labels[0], labels[1]))
    for k in show:
        fr = plan[k]
        cells = []
        for j, p in enumerate(fr["picks"]):
            s = sources[j]
            cells.append("%s t=%.4e d=%+.1e"
                         % (s.describe(p["index"]).split(",")[0],
                            p["t"], p["delta"]))
        print("  %4d %-12.4e  %-27s %-27s"
              % (k, fr["target"], cells[0], cells[1]))


# ------------------------------------------------------------- renderer -----
def field_grid(src, i, nr, nphi, names=("T", "logeta")):
    """The fields of frame i on the common polar analysis grid."""
    levels = src.q2_levels(i)
    r = R0 + (np.arange(nr) + 0.5) * (R1 - R0) / nr
    phi = (np.arange(nphi) + 0.5) * TAU / nphi
    Pg, Rg = np.meshgrid(phi, r, indexing="ij")
    vals, _, _ = q2_eval_levels(levels, Rg.ravel(), Pg.ravel(), names=names)
    out = dict(t=src.frames[i]["t"])
    for k in names:
        out[k] = vals[k].reshape(nphi, nr)
    if "T" in out:
        out["theta"] = out["T"] - np.nanmean(out["T"], axis=0, keepdims=True)
    return out


def scan_scales(sources, labels, plan, nr, nphi):
    t0 = time.time()
    t_lo, t_hi = np.inf, -np.inf
    th_lo, th_hi = np.inf, -np.inf
    e_lo, e_hi = np.inf, -np.inf
    lev_lo, lev_hi = None, None
    for j, s in enumerate(sources):
        for fr in plan:
            i = fr["picks"][j]["index"]
            ext = s.nodal_extent(i)          # exact nodal range (scale anchor)
            t_lo = min(t_lo, ext["T"][0])
            t_hi = max(t_hi, ext["T"][1])
            e_lo = min(e_lo, ext["logeta"][0])
            e_hi = max(e_hi, ext["logeta"][1])
            fg = field_grid(s, i, nr, nphi, names=("T",))
            th_lo = min(th_lo, float(np.nanmin(fg["theta"])))
            th_hi = max(th_hi, float(np.nanmax(fg["theta"])))
            m = s.mesh_meta(i)
            lev_lo = (m["level_min"] if lev_lo is None
                      else min(lev_lo, m["level_min"]))
            lev_hi = (m["level_max"] if lev_hi is None
                      else max(lev_hi, m["level_max"]))
        print("[annulus_realistic] scanned %d frames of %r for the scales "
              "(%.1f s)" % (len(plan), labels[j], time.time() - t0), flush=True)
    th_peak = av.ceil2(max(abs(th_lo), abs(th_hi)))
    return dict(T=(av.floor2(t_lo), av.ceil2(t_hi)),
                theta=(-th_peak, th_peak),
                eta=(av.floor2(e_lo), av.ceil2(e_hi)),
                levels=(int(lev_lo), int(lev_hi)),
                raw=dict(T=(t_lo, t_hi), theta=(th_lo, th_hi),
                         eta=(e_lo, e_hi)))


def build_row_axes(geo, labels, norms, cmaps, level_legend, row_plate,
                   font_scale=1.0, plate_alpha=av.REF_PLATE_ALPHA):
    return av.build_reference_figure(
        labels, norms, cmaps, geo, level_legend,
        plate_inset=row_plate, font_scale=font_scale,
        plate_alpha=plate_alpha,
        bar_titles=[r"$T$", r"$\theta = T - \langle T\rangle_\phi(r)$",
                    r"$\log_{10}\eta/\eta_{\mathrm{ref}}$"])


def render_frame(fig, rows, sources, plan, k, nr, nphi, level_cmap, level_norm,
                 panel_n):
    fr = plan[k]
    cache = FIG.setdefault("_frame_cache", {})
    for j, (row, pick) in enumerate(zip(rows, fr["picks"])):
        src = sources[j]
        i = pick["index"]
        key = (j, i)
        if key not in cache:
            fg = field_grid(src, i, nr, nphi)
            levels = src.q2_levels(i)
            mesh_rgba, mesh_drawable = raster_mesh_panel(
                levels, level_cmap, level_norm, panel_n, FIG["r_box"],
                px_per_unit=FIG["delivered_px_per_unit"],
                min_cell_px=FIG["mesh_min_cell_px"],
                supersample=FIG["mesh_supersample"],
                line_px=max(1, int(round(0.5 * FIG["scale"]))))
            cache[key] = (fg, src.mesh_meta(i), src.inset(i), mesh_rgba,
                          mesh_drawable)
            if len(cache) > 6:
                for old in list(cache)[:-6]:
                    cache.pop(old, None)
        fg, mesh_meta, ins, mesh_rgba, mesh_drawable = cache[key]
        FIG.setdefault("mesh_drawable", {})
        FIG["mesh_drawable"][ROW_NAMES[j]] = mesh_drawable
        row.update(fr["target"], pick, src, fg, ins,
                   mesh=mesh_meta, mesh_image=mesh_rgba)


ROW_NAMES = ["QuaST", "ASPECT"]


FIG = {}


def main(argv=None):
    global FIG
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quast", metavar="DIR",
                    default="out_julia/annulus_realistic/quast_prod",
                    help="QuaST realistic frame directory (v3 state profile)")
    ap.add_argument("--aspect", nargs="+", metavar="DIR",
                    default=["out_julia/annulus_realistic/aspect_prod"],
                    help="ASPECT output directory/directories (repeated or "
                         "comma-separated; merged and de-duplicated in time)")
    ap.add_argument("--aspect-stats", default=None, metavar="CSV",
                    help="override the ASPECT statistics file")
    ap.add_argument("--t0", type=float, default=1.0e-6)
    ap.add_argument("--t1", type=float, default=3.2e-4)
    ap.add_argument("--frames", type=int, default=120,
                    help="number of target times in [t0, t1] (default 120)")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--out", default="out_julia/annulus_realistic/render/"
                                     "realistic.gif")
    ap.add_argument("--also-poster", action="store_true",
                    help="write the last frame as a 1200 px PNG")
    ap.add_argument("--also-mp4", action="store_true",
                    help="write the H.264 clips (1200 px and 2400 px)")
    ap.add_argument("--sidecar", default=None, metavar="JSON")
    ap.add_argument("--stills", type=int, default=0, metavar="N",
                    help="render N evenly spaced spot frames as 1200 px PNGs "
                         "instead of the clip")
    ap.add_argument("--stills-at", default=None, metavar="T1,T2,...",
                    help="render spot frames at these times instead")
    ap.add_argument("--check", "--test", dest="check", action="store_true",
                    help="validate the inputs and print the tables; render "
                         "nothing")
    ap.add_argument("--check-frames", type=int, default=3,
                    help="ASPECT frames used for the expensive checks (default 3)")
    ap.add_argument("--quast-log", default=None,
                    help="QuaST run log for the card/AMR/Ra_interior record "
                         "(default: <quast>/../quast_prod.log if present)")
    ap.add_argument("--aspect-prm", default=None,
                    help="ASPECT parameter file for the AMR record (default: "
                         "<aspect>/original.prm)")
    ap.add_argument("--plate-alpha", type=float, default=0.75,
                    help="opacity of the in-panel white plates (default 0.75; "
                         "the reference uses 0.60, but over the bright T panel "
                         "0.75 keeps the small plate text readable)")
    ap.add_argument("--font-scale", type=float, default=1.0,
                    help="multiply every text size of the reference layout "
                         "(default 1.0 = the measured reference sizes; the "
                         "final annulus clip uses 1.25 for legibility, with "
                         "the T plate split over two lines)")
    ap.add_argument("--master-scale", type=int, default=2,
                    help="render the master at this multiple of 1200 px and "
                         "Lanczos-downsample (default 2)")
    ap.add_argument("--polar-nr", type=int, default=av.REF_POLAR_NR)
    ap.add_argument("--polar-nphi", type=int, default=av.REF_POLAR_NPHI)
    ap.add_argument("--eta-vmin", type=float, default=None)
    ap.add_argument("--eta-vmax", type=float, default=None)
    ap.add_argument("--theta-max", type=float, default=None)
    ap.add_argument("--mesh-min-cell-px", type=float, default=0.67,
                    help="delivered-pixel cell size below which a cell's own "
                         "white outline is omitted; level contours are always "
                         "drawn (default 0.67 delivered px)")
    ap.add_argument("--mesh-supersample", type=int, default=1,
                    help="render the mesh panel at this multiple of the master "
                         "panel and Lanczos-downsample it (default 1, the "
                         "reference's own white-outline weight)")
    ap.add_argument("--tmin", type=float, default=None)
    ap.add_argument("--tmax", type=float, default=None)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--ffmpeg", default=None)
    ap.add_argument("--gif-width", type=int, default=None)
    ap.add_argument("--gif-colors", type=int, default=256)
    ap.add_argument("--gif-max-mb", type=float, default=8.0)
    ap.add_argument("--gif-dither", choices=("none", "floyd"), default="none",
                    help="GIF palette dithering; 'none' (default) keeps the "
                         "dense mesh panels compressible (no visible gain at "
                         "1200 px, ~2.2x smaller file)")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--no-poster", action="store_true")
    ap.add_argument("--rm-pngs", action="store_true")
    ap.add_argument("--quast-provenance", default=None,
                    help="'pilot' or 'production' (default: inferred)")
    ap.add_argument("--aspect-provenance", default=None)
    args = ap.parse_args(argv)
    t_start = time.time()

    # -- sources
    quast = QuastRealistic(args.quast)
    aspect_dirs = []
    for a in args.aspect:
        aspect_dirs.extend([x for x in a.split(",") if x.strip()])
    for d in aspect_dirs:
        if not os.path.isdir(d):
            raise SystemExit("--aspect %r is not a directory" % d)
    aspect = AspectRealistic(aspect_dirs, stats=args.aspect_stats)
    sources = [quast, aspect]
    labels = ["QuaST", "ASPECT"]
    print("[annulus_realistic] QuaST %4d frames  t = %.6g .. %.6g  (%s)"
          % (len(quast), quast.times[0], quast.times[-1], quast.dir), flush=True)
    print("[annulus_realistic] ASPECT %4d frames  t = %.6g .. %.6g  (%s)"
          % (len(aspect), aspect.times[0], aspect.times[-1],
             ", ".join(aspect.dirs)), flush=True)
    if quast.meta.get("R0") is not None and (
            abs(quast.meta["R0"] - R0) > 1e-9
            or abs(quast.meta["R1"] - R1) > 1e-9):
        raise SystemExit("QuaST frame metadata R0/R1 = %r/%r, the card is "
                         "%r/%r" % (quast.meta["R0"], quast.meta["R1"], R0, R1))

    # -- plan
    plan = make_plan(sources, labels, args.t0, args.t1, args.frames)
    if args.check:
        return check_mode(args, sources, labels, plan)

    # -- shared scales
    nr, nphi = args.polar_nr, args.polar_nphi
    print("[annulus_realistic] common polar grid: %d radial x %d angular cell "
          "centres" % (nr, nphi), flush=True)
    scales = scan_scales(sources, labels, plan, nr, nphi)
    t_lo = args.tmin if args.tmin is not None else scales["T"][0]
    t_hi = args.tmax if args.tmax is not None else scales["T"][1]
    th_peak = (args.theta_max if args.theta_max is not None
               else scales["theta"][1])
    e_lo = args.eta_vmin if args.eta_vmin is not None else scales["eta"][0]
    e_hi = args.eta_vmax if args.eta_vmax is not None else scales["eta"][1]
    norms = [Normalize(vmin=t_lo, vmax=t_hi),
             Normalize(vmin=-th_peak, vmax=th_peak),
             eta_norm(e_lo, e_hi)]
    cmaps = [plt.get_cmap("inferno"), plt.get_cmap("RdBu_r"),
             plt.get_cmap("magma_r")]
    lev_lo, lev_hi = scales["levels"]
    legend = level_legend(lev_lo, lev_hi)
    print("[annulus_realistic] shared scales: T = [%s, %s], theta = +-[%s], "
          "log10 eta = [%s, %s] (power law gamma = %.2f), levels %d-%d"
          % (sig(t_lo), sig(t_hi), sig(th_peak), sig(e_lo), sig(e_hi),
             ETA_GAMMA, lev_lo, lev_hi), flush=True)

    # -- canvas (the reference-style layout of annulus_video.py)
    geo = av.reference_layout(len(sources), scale=args.master_scale)
    geo["r_box"] = R1 * av.REF_PANEL / float(av.REF_PANEL - 1)
    geo["sampler"] = av.PanelSampler(
        geo["panel_w"], geo["r_box"], nr, nphi, r0=R0, r1=R1,
        factor=geo["scale"])
    geo["mesh_min_cell_px"] = float(args.mesh_min_cell_px)
    geo["mesh_supersample"] = int(args.mesh_supersample)
    geo["delivered_px_per_unit"] = geo["panel_w"] / (2.0 * geo["r_box"])
    FIG = geo
    print("[annulus_realistic] reference style: canvas %d x %d px, panels %d "
          "px, master %d x %d px (%dx)"
          % (geo["px_w"], geo["px_h"], geo["panel_w"], geo["master_px_w"],
             geo["master_px_h"], geo["scale"]), flush=True)
    fig, rows = build_row_axes(geo, labels, norms, cmaps, legend,
                               lambda ins: plate_text_inset(ins, args.font_scale),
                               font_scale=args.font_scale,
                               plate_alpha=args.plate_alpha)
    save_dpi = av.DPI * geo["scale"]
    panel_n = int(round(geo["panel_w"] * geo["scale"]))
    level_cmap, level_norm = legend["cmap"], legend["norm"]

    # -- output paths
    out = os.path.abspath(args.out)
    base, ext = os.path.splitext(out)
    stem = base if ext.lower() in (".gif", ".mp4", ".png") else out
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    frames_dir = stem + "_frames"

    def render_kth(k, path):
        render_frame(fig, rows, sources, plan, k, nr, nphi, level_cmap,
                     level_norm, panel_n)
        fig.savefig(path, dpi=save_dpi, facecolor=fig.get_facecolor())

    # -- stills mode
    if args.stills or args.stills_at:
        stills_dir = stem + "_stills"
        os.makedirs(stills_dir, exist_ok=True)
        if args.stills_at:
            times = [float(x) for x in args.stills_at.split(",") if x.strip()]
            ks = []
            for t in times:
                ks.append(int(np.argmin([abs(fr["target"] - t) for fr in plan])))
        else:
            n = max(1, int(args.stills))
            n = min(n, len(plan))
            ks = list(np.unique(np.round(np.linspace(0, len(plan) - 1, n)
                                         ).astype(int)))
        for k in ks:
            fr = plan[k]
            tmp = os.path.join(stills_dir, "_master_f%05d.png" % k)
            render_kth(k, tmp)
            png = os.path.join(stills_dir, "still_t%.4e.png"
                               % fr["target"])
            av.downsample_png(tmp, png, geo["px_w"], blank_top=av.REF_TOP)
            os.remove(tmp)
            print("[annulus_realistic] still %s" % png, flush=True)
        plt.close(fig)
        print("[annulus_realistic] %d stills in %s (%.1f s)"
              % (len(ks), stills_dir, time.time() - t_start), flush=True)
        return 0

    # -- full clip
    os.makedirs(frames_dir, exist_ok=True)
    for old in glob.glob(os.path.join(frames_dir, "f_*.png")):
        os.remove(old)
    t_render = time.time()
    for k, fr in enumerate(plan):
        path = os.path.join(frames_dir, "f_%05d.png" % k)
        render_kth(k, path)
        if k % 10 == 0 or k == len(plan) - 1:
            print("[annulus_realistic] frame %d/%d %s (%.1f s)"
                  % (k + 1, len(plan), os.path.basename(path),
                     time.time() - t_render), flush=True)
    plt.close(fig)
    render_s = time.time() - t_render

    # -- poster
    poster = None
    last_png = os.path.join(frames_dir, "f_%05d.png" % (len(plan) - 1))
    if args.also_poster and not args.no_poster:
        poster = stem + "_poster.png"
        av.downsample_png(last_png, poster, geo["px_w"],
                          blank_top=av.REF_TOP)
        print("[annulus_realistic] poster %s (%d px, Lanczos from the %d px "
              "master)" % (poster, geo["px_w"], geo["master_px_w"]), flush=True)

    # -- mp4
    mp4 = mp4_2x = None
    ffmpeg = None
    if args.also_mp4:
        ffmpeg = av.find_ffmpeg(args.ffmpeg)
        mp4 = stem + ".mp4"
        cmd, dt_enc = av.encode(
            ffmpeg, frames_dir, "f_%05d.png", args.fps, mp4, crf=args.crf,
            preset=args.preset,
            vf="scale=%d:-2:flags=lanczos" % geo["px_w"])
        mp4_2x = stem + "_2x.mp4"
        av.encode(ffmpeg, frames_dir, "f_%05d.png", args.fps, mp4_2x,
                  crf=args.crf, preset=args.preset)
        print("[annulus_realistic] encoded %s and %s (%.1f s)"
              % (mp4, mp4_2x, dt_enc), flush=True)

    # -- gif
    gif = None
    gif_info = None
    if not args.no_gif:
        gif = stem + ".gif"
        t_gif = time.time()
        gif_info = av.write_gif(
            sorted(glob.glob(os.path.join(frames_dir, "f_*.png"))),
            gif, args.fps, width_cap=args.gif_width, colors=args.gif_colors,
            max_bytes=args.gif_max_mb * 1e6, downsample_to=geo["px_w"],
            dither=args.gif_dither,
            blank_top=av.REF_TOP)
        print("[annulus_realistic] gif %s  %d px  %d colours  %.2f MB "
              "(%.1f s)" % (gif, gif_info[0], gif_info[1],
                            gif_info[2] / 1e6, time.time() - t_gif), flush=True)

    # -- sidecar
    mesh_draw = dict((k, list(v)) for k, v in FIG.get("mesh_drawable", {}).items())
    sidecar = build_sidecar(args, sources, labels, plan, scales,
                            dict(T=(t_lo, t_hi), theta=(-th_peak, th_peak),
                                 eta=(e_lo, e_hi)), legend, geo,
                            dict(out=out, gif=gif, mp4=mp4, mp4_2x=mp4_2x,
                                 poster=poster, frames_dir=frames_dir,
                                 gif_info=gif_info,
                                 render_s=render_s,
                                 total_s=time.time() - t_start,
                                 ffmpeg=ffmpeg, mesh_draw=mesh_draw))
    sidecar_path = args.sidecar or (stem + ".json")
    os.makedirs(os.path.dirname(os.path.abspath(sidecar_path)) or ".",
                exist_ok=True)
    with open(sidecar_path, "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=1)
    print("[annulus_realistic] sidecar %s" % sidecar_path, flush=True)
    if args.rm_pngs:
        import shutil
        shutil.rmtree(frames_dir, ignore_errors=True)
    print("[annulus_realistic] done in %.1f s" % (time.time() - t_start),
          flush=True)
    return 0


# ------------------------------------------------------------- sidecar ------
def parse_quast_log(path):
    info = dict(path=path, header=[], result={})
    if not path or not os.path.isfile(path):
        info["missing"] = True
        return info
    keymap = {"card:": "card", "tree:": "tree", "latent coefficients:": "latent",
              "AMR:": "amr", "time:": "time", "pilot window:": "window",
              "checkpoint:": "checkpoint", "resume:": "resume"}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s.startswith("#"):
                if s.startswith("ANNULUS_REALISTIC_RESULT"):
                    for tok in s.split()[1:]:
                        if "=" in tok:
                            k, v = tok.split("=", 1)
                            try:
                                info["result"][k] = (int(v) if re.fullmatch(
                                    r"[-+]?\d+", v) else float(v))
                            except ValueError:
                                info["result"][k] = v
                continue
            body = s.lstrip("# ").strip()
            info["header"].append(s)
            for key, name in keymap.items():
                if body.startswith(key) or s.startswith("# " + key):
                    info[name] = body
                    break
    return info


def parse_aspect_prm(path):
    info = dict(path=path, missing=True)
    if not path or not os.path.isfile(path):
        return info
    wanted = {
        "End time", "Output directory", "Time between graphical output",
        "Initial global refinement", "Initial adaptive refinement",
        "Time steps between mesh refinement", "Strategy",
        "Refinement fraction", "Coarsening fraction",
        "Minimum refinement level", "Maximum refinement level",
        "Linear solver tolerance", "Temperature solver tolerance",
        "Maximum time step", "CFL number", "Model name",
        "List of model names",
        "Tangential velocity boundary indicators",
        "Fixed temperature boundary indicators",
        "Radiogenic heating rate", "Formulation", "Opening angle",
        "Initial temperature model", "Gravity model", "Geometry model",
    }
    info["missing"] = False
    info["settings"] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.split("#", 1)[0].strip()
            m = re.match(r"set\s+(.+?)\s*=\s*(.+)$", s)
            if m and m.group(1).strip() in wanted:
                info["settings"][m.group(1).strip()] = m.group(2).strip()
    return info


def build_sidecar(args, sources, labels, plan, scales, used_scales, legend,
                  geo, outputs):
    quast, aspect = sources
    qlog = args.quast_log
    if qlog is None:
        cand = os.path.join(os.path.dirname(os.path.dirname(quast.dir)),
                            "quast_prod.log")
        qlog = cand if os.path.isfile(cand) else None
    qinfo = parse_quast_log(qlog)
    if args.aspect_prm:
        aprm = parse_aspect_prm(args.aspect_prm)
    else:
        cand = os.path.join(aspect.dirs[0], "original.prm")
        aprm = parse_aspect_prm(cand if os.path.isfile(cand) else None)
    aspect_prms = []
    for d in aspect.dirs:
        cand = os.path.join(d, "original.prm")
        if os.path.isfile(cand):
            aspect_prms.append(parse_aspect_prm(cand))
    ra_interior = qinfo.get("result", {}).get("Ra_interior")
    eta_min = qinfo.get("result", {}).get("eta_min")
    eta_max = qinfo.get("result", {}).get("eta_max")
    eta_contrast = qinfo.get("result", {}).get("eta_contrast")
    qprov = args.quast_provenance or (
        "pilot" if "pilot" in quast.dir or "direct" in quast.dir
        else "production")
    aprov = args.aspect_provenance or (
        "pilot" if "pilot" in aspect.dirs[0] else "production")

    targets = []
    for fr in plan:
        picks = []
        for j, p in enumerate(fr["picks"]):
            s = sources[j]
            d = dict(row=labels[j], index=int(p["index"]),
                     t=float(p["t"]), delta=float(p["delta"]),
                     describe=s.describe(p["index"]),
                     inset=s.inset(p["index"]))
            if isinstance(s, QuastRealistic):
                f = s.frames[p["index"]]
                d.update(frame=int(f["frame"]), step=int(f["step"]))
            else:
                d.update(frame=s.frames[p["index"]]["stem"])
            picks.append(d)
        targets.append(dict(k=int(fr["k"]), t=float(fr["target"]), picks=picks))

    layout = dict(
        style="reference: the canvas of docs/assets/quast_vs_aspect_realistic.gif",
        width=int(geo["px_w"]), height=int(geo["px_h"]),
        master_width=int(geo["master_px_w"]),
        master_height=int(geo["master_px_h"]),
        supersampling="master rendered at %dx the 1200 px canvas and "
                      "Lanczos-downsampled to %d px for the GIF, poster and "
                      "the 1200 px mp4" % (geo["scale"], geo["px_w"]),
        top_margin_px=av.REF_TOP, panel_px=int(geo["panel_w"]),
        panel_x=[int(x) for x in geo["x0"]],
        panel_top=[int(t) for t in geo["panel_top"]],
        panel_gap_px=int(av.REF_X0[1] - (av.REF_X0[0] + av.REF_PANEL)),
        right_margin_px=int(av.REF_W - (av.REF_X0[3] + av.REF_PANEL)),
        row_gap_px=int(av.REF_ROW_GAP), bar_gap_px=int(av.REF_BAR_GAP),
        bar_px=int(av.REF_BAR_H), bar_label_y=int(geo["title_y"]),
        columns=["T", "theta_ang", "log10_eta_over_eta_ref", "mesh(level)"],
        header=("none: nothing above the panels, no title and no per-frame "
                "time text; the card and the frame provenance are in this "
                "sidecar and the README caption"),
        row_labels="QuaST / ASPECT inside the temperature panels, bold white "
                   "Computer Modern with a thin dark outline",
        plates="lower-right 60 % white plates: V_rms / mean T / T_max; theta "
               "min/max; log10 eta min/max; leaves-or-cells / level range")
    sidecar = dict(
        figure="the realistic mantle card on the closed 2-D annulus, QuaST "
               "(top) vs ASPECT (bottom)",
        card=dict(
            geometry="PolarAnnulus(R0=1.22, R1=2.22, Phi=2*pi; seam=:periodic); "
                     "gap d = 1 is the physics length unit, d = 2.9e6 m "
                     "(ASPECT)",
            boundaries="free slip on both circles, T = 1 (1500 K) inner, "
                       "T = 0 (500 K) outer, remove net rotation (ASPECT)",
            physics="H = 10, Boussinesq, nominal Ra = 1e5 at the composite "
                    "calibration point; composite isostress diffusion+"
                    "dislocation rheology (A_d=1.6240183119e5, A_s="
                    "2.6796419238e-1, E_d=12, E_s=17, n=3, eta_T0=0.5, "
                    "eps_min=1e-12, eta_min=1e-4, eta_max=1e4), two phase "
                    "transitions z0 = 0.87 (+0.032, B=1.8182) and z0 = 0.78 "
                    "(-0.032, B=3.0303) with per-phase latent heat "
                    "(L=0.11), no yielding/weakening",
            initial_condition="0.05 sin(pi h) cos(2 phi) + "
                              "0.02 sin(2 pi h) cos(4 phi) on top of the "
                              "annular H-inclusive conductive base, "
                              "h=(r-R0)/d, phi=atan2(y,x)",
            measured=dict(Ra_interior=ra_interior,
                          source="ANNULUS_REALISTIC_RESULT, final state of "
                                 "the QuaST run" if ra_interior else None,
                          eta_min=eta_min, eta_max=eta_max,
                          eta_contrast=eta_contrast),
            amr_quast=qinfo.get("amr"),
            amr_aspect=aprm.get("settings"),
            material_model_aspect=aprm.get("settings", {}).get("Model name"),
            quast_log=qinfo.get("path"),
            aspect_prm=aprm.get("path"),
            aspect_prms=[dict(path=p.get("path"),
                              settings=p.get("settings"))
                         for p in aspect_prms]),
        no_nusselt="The per-wall Nusselt pair is undefined on this card by "
                   "design (H = 10 makes the H-inclusive base conductive flux "
                   "negative at the inner circle, -dTbar/dr(R0) = -4.3058), so "
                   "the QuaST arm runs nu_undefined=:nan and the frame stream "
                   "has no Nusselt column.  The figure therefore shows "
                   "V_rms / mean T / T_max instead of a Nusselt pair; this is "
                   "a deliberate property of the card, not missing data.",
        conversions=dict(
            length="r_nd = r / d, d = 2.9e6 m",
            temperature="T_nd = (T - 500 K) / 1000 K",
            velocity="u_nd = u * d / kappa, d/kappa = 2.9e12 s",
            viscosity="eta_nd = eta / eta0, eta0 = 2.368635e23 Pa s",
            time="t_nd = t * kappa / d^2 = t / 8.41e18 s",
            source="aspect_runs/annulus_realistic_demo/annulus_realistic_pilot"
                   ".prm (dimensional shell card)"),
        inputs=dict(
            quast=dict(path=quast.dir, kind="annulus_frames_v3 state profile",
                       nframes=len(quast), t_range=[quast.times[0],
                                                    quast.times[-1]],
                       meta_version=quast.meta.get("version"),
                       mesh=dict(nx=quast.meta.get("nx"), ny=quast.meta.get("ny"),
                                 R0=quast.meta.get("R0"),
                                 R1=quast.meta.get("R1")),
                       field_eta="frame_%05d_eta.bin, the solver's own nodal "
                                 "log10(eta/eta_ref)",
                       provenance=qprov),
            aspect=dict(paths=aspect.dirs, nframes=len(aspect),
                        t_range=[aspect.times[0], aspect.times[-1]],
                        statistics=aspect.stats_paths,
                        nstatistics=len(aspect.stats),
                        field_eta="the saved nodal `viscosity`, converted with "
                                  "eta_nd = eta/eta0 (the realistic composite "
                                  "plugin clamps it to the card's "
                                  "[eta_min, eta_max], so it is usable; "
                                  "cross-checked against the card's composite "
                                  "law in --check)",
                        field_T="point_data['T'], converted with "
                                "T_nd=(T-500)/1000",
                        provenance=aprov)),
        layout=layout,
        theta_definition=dict(
            formula="theta = T - <T>_phi(r)",
            grid="common polar grid of %d radial x %d angular cell centres "
                 "(%d points), identical for both codes"
                 % (args.polar_nr, args.polar_nphi,
                    args.polar_nr * args.polar_nphi),
            note="<T>_phi(r) is the angular mean at each radius on the common "
                 "grid: a diagnostic of the state, not a solver variable; the "
                 "annular analogue of the box clip's horizontal-mean anomaly"),
        rasterisation=dict(
            method="exact Q2 tensor-basis evaluation on each code's own leaves: "
                   "the leaves of one tree level tile a dyadic lattice in "
                   "logical (S, T) coordinates, so each level is evaluated in "
                   "one vectorised pass and painted coarse->fine; the finest "
                   "cell containing each polar-grid point wins",
            quast="frame_%05d_mesh.csv leaves (logical S,T) + the solver's Q2 "
                  "node cloud; each leaf's nine Q2 nodes are looked up in the "
                  "node cloud (max distance < 1e-9) and the Q2 basis is "
                  "evaluated exactly",
            aspect="each VTU parent cell is the group of four linear sub-quads "
                   "on nine consecutive Q2 nodes; the level is inferred from "
                   "the cell area and mid radius, and the nine node values are "
                   "evaluated with the Q2 tensor basis on the cell's dyadic "
                   "polar rectangle",
            mesh_panel="level/cell-id raster on the panel's own pixel grid "
                       "(one white boundary pixel at the 2x master, i.e. the "
                       "reference's white 0.42 pt cell outlines)",
            mesh_boundaries=dict(
                policy="every cell's boundary is drawn in white while the cell "
                       "is at least %.2f delivered (1200 px canvas) pixels "
                       "across, at the reference's ~0.5 delivered px outline "
                       "weight (scale-independent); below that a cell is "
                       "colour-filled and only the level-boundary contours are "
                       "drawn (a level-7 ASPECT cell of this card is ~0.5 px "
                       "across at 279 px, so outlining all ~160 000 of them "
                       "would fill the mesh panel white)"
                       % args.mesh_min_cell_px,
                per_row=outputs.get("mesh_draw", {}),
                note="the per-row list is the drawable flag of each level, "
                     "coarsest first; the level legend is shared by both rows")),
        mesh_definition=dict(
            quast="per-frame leaves of QuaST's balanced tree, coloured by tree "
                  "level; the coarse ring is 8 radial x 80 angular at level 3",
            aspect="per-frame active cells, coloured by the tree level "
                   "inferred from the cell area; the coarse shell is 12 x 1, "
                   "the card's Initial global refinement = 5 (level 5)",
            level_calibration="a tree level is each code's own: on this card "
                              "QuaST's levels 5-7 are (320x32, 640x64, "
                              "1280x128) leaves and ASPECT's 5-7 are "
                              "(384x32, 768x64, 1536x128) cells, i.e. the same "
                              "refinement scale within ~20 % in the range the "
                              "figure shows; the level numbers are not frozen "
                              "against each other",
            levels=dict(shared_min=int(scales["levels"][0]),
                        shared_max=int(scales["levels"][1]),
                        ticks=list(range(int(scales["levels"][0]),
                                         int(scales["levels"][1]) + 1)))),
        scales=dict(
            T=dict(vmin=float(used_scales["T"][0]),
                   vmax=float(used_scales["T"][1]), cmap="inferno",
                   measured=[float(scales["raw"]["T"][0]),
                             float(scales["raw"]["T"][1])]),
            theta=dict(vmin=float(used_scales["theta"][0]),
                       vmax=float(used_scales["theta"][1]), cmap="RdBu_r",
                       measured=[float(scales["raw"]["theta"][0]),
                                 float(scales["raw"]["theta"][1])]),
            log10_eta=dict(vmin=float(used_scales["eta"][0]),
                           vmax=float(used_scales["eta"][1]), cmap="magma_r",
                           gamma=ETA_GAMMA,
                           norm="power law (box-clip mapping): colour = "
                                "((log10 eta - vmin)/(vmax - vmin))**0.30; "
                                "the bar labels the true log10 values, "
                                "dark = stiff = cold",
                           measured=[float(scales["raw"]["eta"][0]),
                                     float(scales["raw"]["eta"][1])]),
            level=dict(vmin=int(scales["levels"][0]),
                       vmax=int(scales["levels"][1]),
                       colours="discrete, one colour per tree level")
        ),
        eta_definition=dict(
            quantity="log10(eta/eta_ref) with eta_ref = 1 (nondimensional) = "
                     "eta0 = 2.368635e23 Pa s",
            quast="the solver's composite-isostress nodal field written per "
                  "frame (frame_%05d_eta.bin)",
            aspect="the saved nodal `viscosity` of ASPECT's realistic "
                   "composite material model (ASPECT's diffusion-dislocation "
                   "partition with the card's prefactors), eta_nd = eta/eta0; "
                   "the plugin clamps it to [eta_min, eta_max] of the card",
            note="each row shows the material state its own solver used; the "
                 "two laws are the same isostress diffusion+dislocation "
                 "family calibrated at the same card point, not a fitted "
                 "reproduction"),
        window=dict(t0=float(args.t0), t1=float(args.t1),
                    nframes=int(len(plan)), fps=float(args.fps),
                    frame_dt=(float(plan[-1]["target"] - plan[0]["target"])
                              / max(len(plan) - 1, 1))),
        matching=dict(
            rule="one target list in time; each row shows the nearest available "
                 "frame (no time interpolation), the offset delta = t_frame - "
                 "t_target is recorded per target",
            quast_frame_cadence="one frame every 5 steps plus one step past "
                                "every AMR cycle (and the final state)",
            aspect_frame_cadence=aprm.get("settings", {}).get(
                "Time between graphical output"),
            max_abs_delta={labels[j]: float(max(
                abs(fr["picks"][j]["delta"]) for fr in plan))
                for j in range(len(sources))},
            distinct_frames={labels[j]: len({fr["picks"][j]["index"]
                                             for fr in plan})
                             for j in range(len(sources))},
            targets=targets),
        outputs=dict(
            gif=outputs["gif"], mp4=outputs["mp4"], mp4_2x=outputs["mp4_2x"],
            poster=outputs["poster"], frames_dir=outputs["frames_dir"],
            gif_info=(None if outputs["gif_info"] is None else
                      dict(width=outputs["gif_info"][0],
                           colours=outputs["gif_info"][1],
                           bytes=outputs["gif_info"][2])),
            render_s=outputs["render_s"], total_s=outputs["total_s"]),
        command=" ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
        environment=dict(numpy=np.__version__,
                         matplotlib=matplotlib.__version__,
                         python=sys.version.split()[0]),
        caption=(
            "The realistic mantle card (H = 10, nominal Ra = 1e5 at the "
            "composite calibration point; measured Ra_interior = %s at the "
            "QuaST run's final state) on the closed 2-D annulus, run by QuaST "
            "(top) and by ASPECT (bottom): total temperature, its anomaly "
            "about the angular mean at each radius, log10(eta/eta_ref) of the "
            "composite-isostress rheology on the power-law colour mapping "
            "(gamma = 0.30, bar labels the true range, dark = stiff = cold), "
            "and each code's own adaptive mesh coloured by tree level.  Both "
            "rows share one colour scale per column.  The card has no "
            "Nusselt pair by design (H = 10 makes the per-wall normalisation "
            "undefined), so the temperature plate carries V_rms, mean T and "
            "T_max instead.  No header or card text is drawn: the card, the "
            "measured numbers and the provenance are in the JSON sidecar and "
            "in this caption."
            % (sig(ra_interior) if ra_interior is not None else "not recorded")),
        notes=[
            "The two mesh panels compare two independent adaptive "
            "discretizations: QuaST's gradient-jump indicator with a wall "
            "minimum level vs ASPECT's temperature strategy (20 % refine / "
            "0.5 % coarsen); the cell counts and level ranges are each code's "
            "own and are printed on the plates.",
            "Each row shows its own nearest frame in time (the offset delta is "
            "recorded per target in `matching.targets`); there is no time "
            "interpolation, only spatial rasterisation.",
            "Mesh-panel boundary policy: every cell outline is drawn while the "
            "cell is at least %.2f delivered px across; a level-7 ASPECT cell "
            "of this card is ~0.5 px across at 279 px, so for such sub-pixel "
            "levels the panel is colour-filled with the level-contour outlines "
            "only (outlining all ~160 000 cells at the delivered size would "
            "fill the panel white).  Pass --mesh-min-cell-px 0.3 to draw every "
            "outline instead." % args.mesh_min_cell_px,
            "This is a demonstration of the same-card comparison, not an "
            "accuracy claim: the two codes advance the transient with "
            "different integrators and time-step sequences.",
            "The ASPECT row's viscosity is the saved material field, not a "
            "reconstruction; --check cross-checks it against the card's "
            "composite law with the Q2 strain rate and reports the "
            "difference.",
        ],
    )
    sidecar["notes"] = [n for n in sidecar["notes"]]
    return sidecar


# --------------------------------------------------------------- --check ----
def check_mode(args, sources, labels, plan):
    quast, aspect = sources
    ok = True
    print("\n== input inventory " + "=" * 64)
    qmeta = quast.meta
    qT = [f["Tmax"] for f in quast.frames]
    qTmin = [f["Tmin"] for f in quast.frames]
    qlev = [(f["level_min"], f["level_max"]) for f in quast.frames]
    qleaves = [f["leaves"] for f in quast.frames]
    print("QuaST  %s" % quast.dir)
    print("  frames %4d   t = %.6g .. %.6g   steps %d .. %d   nodes %d .. %d"
          % (len(quast), quast.times[0], quast.times[-1],
             quast.frames[0]["step"], quast.frames[-1]["step"],
             min(f["nnodes"] for f in quast.frames),
             max(f["nnodes"] for f in quast.frames)))
    print("  leaves %6d .. %6d   level %d-%d   T = [%.6f, %.6f]"
          % (min(qleaves), max(qleaves), min(x[0] for x in qlev),
             max(x[1] for x in qlev), min(qTmin), max(qT)))
    # eta range over every frame (cheap: the eta bins are small)
    t0 = time.time()
    e_lo, e_hi = np.inf, -np.inf
    for f in quast.frames:
        e = np.fromfile(f["eta_bin"], dtype=np.float32)
        e_lo = min(e_lo, float(e.min()))
        e_hi = max(e_hi, float(e.max()))
    print("  log10(eta/eta_ref) over all frames = [%.4f, %.4f]   (%.1f s)"
          % (e_lo, e_hi, time.time() - t0))
    if e_lo < math.log10(ETA_MIN_ND) - 1.0e-3 or \
            e_hi > math.log10(ETA_MAX_ND) + 1.0e-3:
        print("  WARNING: QuaST eta leaves the card's clamp [%g, %g]"
              % (math.log10(ETA_MIN_ND), math.log10(ETA_MAX_ND)))
        ok = False
    print("ASPECT %s" % ", ".join(aspect.dirs))
    print("  frames %4d   t = %.6g .. %.6g   statistics rows %d (%s)"
          % (len(aspect), aspect.times[0], aspect.times[-1], len(aspect.stats),
             aspect.stats_path))
    cells_stats = []
    for i in range(len(aspect)):
        s = aspect.stat_at(aspect.times[i])
        if s:
            cells_stats.append(s["cells"])
    if cells_stats:
        print("  statistics cells %d .. %d   V_rms_nd %.3g .. %.3g   "
              "Tmax_nd %.4f .. %.4f"
              % (min(cells_stats), max(cells_stats),
                 min(s["vrms"] * U_ND for s in aspect.stats),
                 max(s["vrms"] * U_ND for s in aspect.stats),
                 min((s["Tmax"] - T_SURF) / DELTA_T for s in aspect.stats),
                 max((s["Tmax"] - T_SURF) / DELTA_T for s in aspect.stats)))

    print("\n== conversions " + "=" * 66)
    print("  d = %.6g m   kappa = %.6g m^2/s   t_nd = t / %.6g s   "
          "u_nd = u * %.6g s/m   eta0 = %.6g Pa s"
          % (D_PHYS, KAPPA, T_ND, U_ND, ETA0))
    print("  T_nd = (T - %.0f K) / %.0f K   r_nd = r / d   eta_nd = eta / eta0"
          % (T_SURF, DELTA_T))
    print("  target window t_nd = [%.6g, %.6g] = [%.4g, %.4g] s"
          % (args.t0, args.t1, args.t0 * T_ND, args.t1 * T_ND))

    # -- QuaST Q2 vs Delaunay (one mid frame)
    print("\n== QuaST rasterisation: direct Q2 vs scipy Delaunay " + "=" * 22)
    mid = min(len(quast) - 1, len(quast) // 2)
    v = validate_quast_q2(quast, mid)
    print("  frame %d (step %d): %d random points, Q2 vs Delaunay |dT|: "
          "max %.2e, median %.2e, p99 %.2e"
          % (v["frame"], quast.frames[mid]["step"], v["pts"], v["max"],
             v["median"], v["p99"]))
    print("  (the renderer uses the exact Q2 evaluation; the Delaunay route "
          "smooths at most one colour step and is kept only as this check)")

    print("\n== ASPECT rasterisation: Q2 vs scipy Delaunay " + "=" * 26)
    v = validate_aspect_fields(aspect, 0)
    print("  frame t_nd=%.4e, %d de-duplicated Q2 nodes: Q2 vs Delaunay "
          "on %d points" % (v["t"], v["nodes"], v["pts"]))
    for name in ("T", "logeta"):
        print("    %-6s |d| max %.2e, median %.2e, p99 %.2e"
              % (name, v[name]["max"], v[name]["median"], v[name]["p99"]))
    print("    cell-centred constant T vs Delaunay: max %.2e, median %.2e, "
          "p99 %.2e  (the fallback the card does not need)"
          % (v["cell_constant_T"]["max"], v["cell_constant_T"]["median"],
             v["cell_constant_T"]["p99"]))

    # -- ASPECT level inference + eta provenance
    print("\n== ASPECT level inference vs statistics " + "=" * 36)
    nf = max(1, min(args.check_frames, len(aspect)))
    picks = sorted(set(int(x) for x in np.linspace(0, len(aspect) - 1, nf)))
    bad = 0
    for i in picks:
        r = validate_aspect_levels(aspect, i)
        s = aspect.stat_at(aspect.times[i])
        print("  t_nd=%.4e  cells(VTU)=%7d stats=%s match=%s  "
              "levels %d-%d  frac_dev=%.1e"
              % (aspect.times[i], r["cells"],
                 "n/a" if r["stats_cells"] is None else r["stats_cells"],
                 r["match"], r["level_min"], r["level_max"], r["frac_dev"]))
        print("      histogram %s" % r["hist"])
        if not r["match"]:
            bad += 1
        if r["frac_dev"] > 1.0e-3:
            bad += 1
            print("      WARNING: dyadic lattice deviation %.2e" % r["frac_dev"])
    if bad:
        ok = False

    # -- ASPECT saved eta vs the card's composite law
    print("\n== ASPECT viscosity provenance " + "=" * 44)
    i = picks[len(picks) // 2]
    f = aspect.read(i)
    print("  frame t_nd=%.4e: saved log10 eta range [%.4f, %.4f]"
          % (aspect.times[i], float(f["logeta"].min()),
             float(f["logeta"].max())))
    if np.abs(f["logeta"] - math.log10(ETA_MAX_ND)).min() < 1e-6:
        print("  (some nodes sit on the card's upper viscosity clamp %.0e)"
              % ETA_MAX_ND)
    v = validate_eta_saved(aspect, i)
    if v:
        print("  composite-law cross-check on %d cell centres: "
              "|dlog10 eta| median %.2e, p90 %.2e, max %.2e; %.1f %% of the "
              "samples on the upper clamp"
              % (v["n"], v["median"], v["p90"], v["max"],
                 100.0 * v["frac_clamped"]))
        print("  law eta_nd range on the sample [%.3e, %.3e]; the renderer "
              "uses the saved field (each solver's actual state)" %
              (v["eta_law_min"], v["eta_law_max"]))

    # -- field ranges over the shown frames
    print("\n== shown-frame field ranges " + "=" * 48)
    scales = scan_scales(sources, labels, plan, args.polar_nr, args.polar_nphi)
    print("  T = [%.5f, %.5f] -> scale [%s, %s]"
          % (scales["raw"]["T"][0], scales["raw"]["T"][1],
             sig(scales["T"][0]), sig(scales["T"][1])))
    print("  theta = [%.4f, %.4f] -> symmetric +-%s"
          % (scales["raw"]["theta"][0], scales["raw"]["theta"][1],
             sig(scales["theta"][1])))
    print("  log10 eta = [%.4f, %.4f] -> scale [%s, %s]"
          % (scales["raw"]["eta"][0], scales["raw"]["eta"][1],
             sig(scales["eta"][0]), sig(scales["eta"][1])))
    print("  levels %d-%d over the shown frames" % scales["levels"])

    # -- time alignment
    print("\n== time alignment " + "=" * 58)
    plan_table(plan, sources, labels)
    for j, s in enumerate(sources):
        d = np.array([abs(fr["picks"][j]["delta"]) for fr in plan])
        print("  %-8s max |delta| = %.3e (%.4f%% of the window = %.3g s), "
              "distinct frames %d/%d"
              % (labels[j], d.max(), 100.0 * d.max()
                 / max(args.t1 - args.t0, 1e-30), d.max() * T_ND,
                 len({fr["picks"][j]["index"] for fr in plan}), len(plan)))

    # -- provenance
    print("\n== provenance / sidecar " + "=" * 52)
    qlog = args.quast_log
    if qlog is None:
        cand = os.path.join(os.path.dirname(os.path.dirname(quast.dir)),
                            "quast_prod.log")
        qlog = cand if os.path.isfile(cand) else None
    qinfo = parse_quast_log(qlog)
    if qinfo.get("missing"):
        print("QuaST log: not found -- the sidecar will record the card and "
              "the AMR settings from the frame metadata only")
        print("  (pass --quast-log PATH after the run finishes to record the "
              "measured Ra_interior)")
    else:
        print("QuaST log: %s" % qinfo["path"])
        for key in ("card", "tree", "amr", "time"):
            if key in qinfo:
                print("  # %s" % qinfo[key])
        res = qinfo.get("result", {})
        if res:
            print("  final: t=%.6e steps=%s leaves_max=%s level_seen=%s "
                  "Ra_interior=%.4e eta_min=%.3e eta_max=%.3e"
                  % (res.get("t", float("nan")), res.get("steps"),
                     res.get("leaves_max"), res.get("level_seen"),
                     res.get("Ra_interior", float("nan")),
                     res.get("eta_min", float("nan")),
                     res.get("eta_max", float("nan"))))
    prm = args.aspect_prm or os.path.join(aspect.dirs[0], "original.prm")
    if os.path.isfile(prm):
        ap = parse_aspect_prm(prm)
        print("ASPECT prm: %s" % prm)
        for k in ("Model name", "Initial global refinement",
                  "Initial adaptive refinement",
                  "Time steps between mesh refinement", "Strategy",
                  "Refinement fraction", "Coarsening fraction",
                  "Minimum refinement level", "Time between graphical output",
                  "List of model names"):
            if k in ap["settings"]:
                print("  %-38s = %s" % (k, ap["settings"][k]))
    else:
        print("ASPECT prm: not found (%s)" % prm)
    print("\n== check %s " % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
