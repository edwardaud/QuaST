#!/usr/bin/env python3
"""Two-row QuaST-vs-ASPECT comparison video for the 2-D cylindrical annulus.

The card is Davies et al. (2022) case 2.1 / the ASPECT documentation benchmark:
the closed periodic ring R0 = 1.22, R1 = 2.22 (gap d = 1), isoviscous eta = 1,
Ra = 1e4, Boussinesq H = 0, free slip on both circles, T = 1 at the inner wall
and 0 at the outer, and the ASPECT `spherical hexagonal perturbation` (angular
mode 4) as the initial condition.  Both codes run the *same* 3 072-cell mesh --
QuaST 16 radial x 192 angular Q2-Q1-Q2 quadtree leaves, ASPECT
`Initial global refinement = 4` = 192 tangential x 16 radial Q2-Q1-Q2 cells --
and both are sampled every 0.05 time units from t = 0 to t = 2 (41 frames).

    python3 scripts/annulus_video.py \
        --row "QuaST=quast:out_julia/annulus_video/quaST_frames" \
        --row "ASPECT=aspect:out_julia/annulus_video/aspect_frames" \
        --aspect-stats out_julia/annulus_video/logs/aspect_statistics.csv \
        --out out_julia/annulus_video/quast_vs_aspect_annulus.mp4

Layout (stated so the result is reproducible)
---------------------------------------------
Two rows (QuaST top, ASPECT bottom), four square panels per row:

  1. total temperature T, shared `inferno` scale over every row and frame;
  2. the angular-mean anomaly  theta_ang = T - <T>_phi(r), shared symmetric
     diverging (`RdBu_r`) scale.  <T>_phi(r) is the mean over phi at each
     radius on ONE common polar grid (200 radial x 1024 angular cell centres,
     read from the QuaST frame metadata, pinned identically for both codes), so
     the two rows carry the identical definition; it is a diagnostic of the
     state, not a solver variable;
  3. log10 |u| on a shared scale (replaces the reference box GIF's viscosity
     column -- this card is isoviscous, eta = 1 everywhere, so a viscosity panel
     would be a constant colour and carry no information);
  4. the mesh: QuaST element boundaries reconstructed from the uniform
     nx x ny ring, ASPECT cell boundaries from the VTU connectivity, both
     labelled with their cell count.  Both are the same 3 072-cell uniform mesh.

Each row also prints the time of the frame it *actually* shows (no synthetic
interpolation in time: the nearest available frame inside +-0.05/2 is used) and
an inset box with that frame's Nu_top, Nu_bot, V_rms and mean T.  QuaST's values
come from its frame metadata, ASPECT's from its `statistics` file.  The two
Nusselt columns are each code's own wall-flux family -- QuaST quotes the
one-sided GLL wall gradient (family A of the validation page) and ASPECT quotes
its `heat flux statistics` consistent boundary flux -- so a several-per-cent
difference in Nu between the rows at this mesh is a measurement definition, not
a different state; the velocity and mean temperature columns agree closely.

The disk fills a square panel, so the master is ~1200 x 630 px (two bands of
square panels), not the wide 2.5:1 box shape of
`docs/assets/quast_vs_aspect_realistic.gif`; the difference is the geometry, not
a layout change.

Frame sources
-------------
`quast:<dir>`   a frame directory from `examples/annulus_benchmark.jl` with
                `ANN_FRAMES` set: `frame_%05d.bin` (int64 nnodes, then Float64
                x, y, theta, T, |u|), `frame_meta.csv` whose '#' header carries
                the uniform mesh (nx, ny, R0, R1, Phi) and the analysis polar
                grid (nr, nphi), plus one CSV row per frame with
                frame,step,t,Nu_T,Nu_B,Vrms,meanT,Tmin,Tmax,nnodes.
`aspect:<dir>`  an ASPECT output directory (or its `solution/` directory):
                `solution-NNNNN.PPPP.vtu` pieces.  The VTU of this image writes
                each Q2 cell as a 2 x 2 group of linear quads on the cell's nine
                Q2 nodes (measured: 3 072 cells -> 12 288 quads, 27 648 points,
                12 672 unique after de-duplication); the renderer rebuilds the
                3 072 parent cells from those groups for the mesh panel and
                interpolates the unique nodal T and |velocity| onto the shared
                polar grid.  The card is non-dimensional, so no unit conversion
                is applied.

Colour scales are one scale per column, measured over a scan of every shown
frame of every row and rounded outward (`floor2`/`ceil2`, two significant
digits), exactly as `three_row_video.py` does; `--vmin/--vmax`,
`--dtheta-max`, `--umin/--umax` override.  The scale actually used is written
into the `.json` sidecar and printed on the frame's colour bars.

Outputs
-------
<out>.mp4            H.264, yuv420p, --crf 18, --preset slow
<out>.gif            Pillow GIF, optimized, --fps 8-10, <= ~8 MB (auto-fitted)
<out>_poster.png     the final frame as a PNG
<out>.json           the plan: rows, window, target times, per-row matches,
                     scales, per-frame diagnostics, commands, timings
<out>_frames/*.png   the individual frames (kept for re-encoding)

Continuation-video overrides (added for the case-2.2 video; the defaults are the
case-2.1 card and reproduce the original output byte for byte):
`--header`/`--header2` replace the two header lines drawn above the rows, and
`--sidecar-extra FILE` merges a JSON object into the `.json` sidecar (top-level
keys replace the defaults) so a continuation video can state its card,
protocol and caveats.  `--case` already overrides the case name in the sidecar.

AMR demonstration mode (`--amr`, the figure
`docs/assets/quast_vs_aspect_annulus_demo.gif`)
---------------------------------------------------------------
`--amr` switches the last two columns and the mesh handling:

  3. `log10 eta = -3 T` (the material law is eta = 1000^(-T) = 10^(-3T), so the
     viscosity field is the total temperature field; no eta samples are written
     to the frame stream and none is needed), shared `magma_r` scale;
  4. each row's **adaptive mesh**, redrawn every frame from that frame's own
     tree and coloured by tree level on one shared discrete scale: QuaST's
     leaves are read from `frame_%05d_mesh.csv` (logical coordinates, mapped
     with the card's own map), ASPECT's parent cells are rebuilt from the VTU
     Q2 groups and each cell's level is inferred from its area and radius as
     `L = 0.5 log2((2 pi/12)/(A/r_mid))` (verified against `Initial global
     refinement`).  Both codes are 12-angular x 1-radial coarse annulus
     quadtrees, so a level means the same refinement scale on both, but the
     meshes themselves are independent.

The AMR mode needs a v2 QuaST frame stream (`ANN_MESH_DUMP=1`: per-frame mesh
CSV and the leaves/level_min/level_max/dt columns) and any ASPECT VTU output.
The row insets (Nu_top, Nu_bot, V_rms, mean T), the nearest-frame time matching,
the common polar grid and the shared field scales are the same code as the
benchmark videos; without `--amr` nothing in this mode runs.
"""
import argparse
import bisect
import csv
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.collections import LineCollection            # noqa: E402
from matplotlib.cm import ScalarMappable                     # noqa: E402
from matplotlib.colors import (BoundaryNorm, ListedColormap, Normalize,  # noqa: E402
                               to_rgba)

# --- the case geometry (the card; the frame metadata repeats it) ------------
R0 = 1.22
R1 = 2.22
# --- style ------------------------------------------------------------------
PX_W = 1200                # master width in pixels (the reference GIF's width)
PAD_X = 44
GAP = 22
N_COLS = 4
PANEL_TOP = 40             # two header lines above the first band
ROW_GAP = 10
DPI = 96
BG = "white"
FG = "#111111"
DIM = "#3f3f3f"
HEADER = "#555555"
SPINE = "#9a9a9a"
MESH_COL = "#1f4e79"
T_CMAP = "inferno"
THETA_CMAP = "RdBu_r"
SPEED_CMAP = "viridis"
# AMR demonstration mode (`--amr`): the third column is log10 eta = -3 T, drawn
# with the viscosity colormap of the realistic box video; the fourth column is
# the adaptive mesh coloured by tree level.
ETA_CMAP = "magma_r"
LEVEL_CMAP = "turbo"       # fallback sampling when more levels than colours
# Discrete, high-contrast level colours (the 'colour = level' legend has to stay
# readable when 5 000+ cells are drawn as thin lines in a 261 px panel).
LEVEL_COLORS = ["#2166ac", "#00b0d8", "#39a83c", "#f3d000", "#f07c00",
                "#b2182b", "#7a0177", "#4d004b"]
SPEED_FLOOR = 1.0e-8       # |u| below this is drawn at the bottom of the scale
# The theta anomaly is defined on this grid when the QuaST metadata does not
# declare one (the harness records its own ANN_FRAME_NR/NPHI, default 200/1024).
NR_DEFAULT = 200
NPHI_DEFAULT = 1024


# --------------------------------------------------------------------- helpers
def floor2(x):
    """Round down to two significant digits."""
    if x == 0.0 or not np.isfinite(x):
        return 0.0
    e = math.floor(math.log10(abs(x))) - 1
    return math.floor(x / 10.0 ** e) * 10.0 ** e


def ceil2(x):
    """Round up to two significant digits."""
    if x == 0.0 or not np.isfinite(x):
        return 1.0
    e = math.floor(math.log10(abs(x))) - 1
    return math.ceil(x / 10.0 ** e) * 10.0 ** e


def sigfig(x, sig=4):
    """Short fixed-format string for a label."""
    if x == 0:
        return "0"
    if abs(x) >= 1e-2:
        return "%.*g" % (sig, x)
    return "%.2e" % x


def sci(x, sig=3):
    """A number as mathtext: a real exponent once it needs one, else 3 digits."""
    if x is None:
        return "n/a"
    if x == 0.0:
        return r"$0$"
    if abs(x) >= 1.0e3 or abs(x) < 1.0e-2:
        mant, exp = ("%.*e" % (sig - 1, x)).split("e")
        if "." in mant:
            mant = mant.rstrip("0").rstrip(".")
        return r"$%s\times10^{%d}$" % (mant, int(exp))
    return r"$%.*g$" % (sig, x)


def nearest_index(times, t):
    """Index of the element of the sorted list `times` nearest to t."""
    i = bisect.bisect_left(times, t)
    if i == 0:
        return 0
    if i >= len(times):
        return len(times) - 1
    return i if abs(times[i] - t) < abs(times[i - 1] - t) else i - 1


def time_delta_text(d):
    """+/- exponent label for a frame-time offset."""
    if d == 0.0:
        return "0"
    return "%+.1e" % d


def read_csv_rows(path):
    """Rows of a plain CSV as dicts, or []."""
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []


def aspect_cell_level(corners):
    """Tree level of an ASPECT annular cell from its four corner points.

    A cell of tree level `L` on the 360-degree spherical shell spans
    `dr = d/2^L` radially and `dphi = 2 pi/(12*2^L)` angularly (the coarse
    shell is 12 cells around and one radial cell, which is what
    `Initial global refinement = 3` reports as 96 x 8).  Its physical area is
    `A = r_mid dr dphi` with `r_mid = (r_min + r_max)/2`, so

        A / r_mid = dr dphi = (2 pi/12) / 4^L,
        L = 0.5 * log2((2 pi / 12) / (A / r_mid)).

    The inference is per cell and needs no knowledge of the card's refinement
    settings; `--aspect-level-base` is only used to check it (the script reports
    the inferred level of the coarsest cell and compares it with the initial
    global refinement).  Returns None for a degenerate cell.
    """
    x = np.asarray(corners[:, 0], dtype="float64")
    y = np.asarray(corners[:, 1], dtype="float64")
    a = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    r = np.hypot(x, y)
    r_mid = 0.5 * (float(r.min()) + float(r.max()))
    if not (a > 0.0) or not (r_mid > 0.0):
        return None
    a_norm = a / r_mid
    ref = 2.0 * math.pi / 12.0
    lv = 0.5 * math.log2(ref / a_norm)
    return int(round(lv))


# ---------------------------------------------------------------- QuaST frames
class QuastSource:
    """A frame directory written by examples/annulus_benchmark.jl (ANN_FRAMES)."""

    kind = "quast"

    def __init__(self, path):
        path = os.path.abspath(path)
        self.dir = path
        if not os.path.isfile(os.path.join(path, "frame_meta.csv")) \
                and os.path.isfile(os.path.join(path, "frames",
                                                "frame_meta.csv")):
            self.dir = os.path.join(path, "frames")
        metapath = os.path.join(self.dir, "frame_meta.csv")
        self.meta = dict(nx=None, ny=None, R0=R0, R1=R1, Phi=2.0 * math.pi,
                         nr=NR_DEFAULT, nphi=NPHI_DEFAULT, version=None)
        rows = []
        header = []
        with open(metapath, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("#"):
                    header.append(s)
                    continue
                if not s:
                    continue
                rows.append(s)
        # The v2 stream (ANN_MESH_DUMP=1, the demo preset) writes one
        # frame_%05d_mesh.csv per frame and appends leaves/level_min/level_max/dt
        # to the meta rows; v1 streams (the benchmark videos) are unchanged.
        self.mesh_dump = any("mesh_dump" in h for h in header)
        for s in header:
            m = re.match(r"#\s*annulus_frames_v(\d+)", s)
            if m:
                self.meta["version"] = int(m.group(1))
            m = re.match(r"#\s*mesh\s+(.*)", s)
            if m:
                for tok in m.group(1).split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        self.meta[k] = int(v) if k in ("nx", "ny") else float(v)
            m = re.match(r"#\s*polar_grid\s+(.*)", s)
            if m:
                for tok in m.group(1).split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        self.meta[k] = int(v)
        if self.meta["nx"] is None or self.meta["ny"] is None:
            raise SystemExit("%s: frame_meta.csv has no '# mesh nx=... ny=...' "
                             "header; is this a frame directory from "
                             "examples/annulus_benchmark.jl with ANN_FRAMES?"
                             % metapath)
        fieldnames = ["frame", "step", "t", "Nu_T", "Nu_B", "Vrms", "meanT",
                      "Tmin", "Tmax", "nnodes"]
        # column names from the CSV header row (v1 and v2 have the same first
        # ten columns; v2 appends four)
        if rows and rows[0].startswith("frame,"):
            cols = rows[0].split(",")
            data_rows = rows[1:]
        else:
            cols = fieldnames
            data_rows = rows
        idx = {name: i for i, name in enumerate(cols)}
        has_extras = all(k in idx for k in
                         ("leaves", "level_min", "level_max", "dt"))
        self.frames = []
        for line in data_rows:
            v = line.split(",")
            if len(v) < len(fieldnames):
                continue
            try:
                f = dict(frame=int(v[0]), step=int(v[1]),
                         t=float(v[2]), Nu_T=float(v[3]), Nu_B=float(v[4]),
                         Vrms=float(v[5]), meanT=float(v[6]),
                         Tmin=float(v[7]), Tmax=float(v[8]),
                         nnodes=int(v[9]))
                if has_extras:
                    f.update(leaves=int(v[idx["leaves"]]),
                             level_min=int(v[idx["level_min"]]),
                             level_max=int(v[idx["level_max"]]),
                             dt=float(v[idx["dt"]]))
            except (ValueError, KeyError):
                continue
            binp = os.path.join(self.dir, "frame_%05d.bin" % f["frame"])
            if not (os.path.isfile(binp) and
                    os.path.getsize(binp) == 8 + 5 * 8 * f["nnodes"]):
                continue
            if self.mesh_dump:
                meshp = os.path.join(self.dir,
                                     "frame_%05d_mesh.csv" % f["frame"])
                if not os.path.isfile(meshp):
                    continue
                f["mesh_path"] = meshp
            self.frames.append(f)
        self.frames.sort(key=lambda f: f["t"])
        self.times = [f["t"] for f in self.frames]
        self._cache = {}
        self._interp = None
        self._interp_xy = None

    def __len__(self):
        return len(self.frames)

    def describe(self, i):
        f = self.frames[i]
        return "frame %d, step %d" % (f["frame"], f["step"])

    def meta_row(self, i):
        return self.frames[i]

    def read(self, i):
        """dict(x, y, theta, T, speed) for frame i (cached, small)."""
        if i in self._cache:
            return self._cache[i]
        f = self.frames[i]
        path = os.path.join(self.dir, "frame_%05d.bin" % f["frame"])
        with open(path, "rb") as fh:
            n = int(np.fromfile(fh, dtype=np.int64, count=1)[0])
            raw = np.fromfile(fh, dtype=np.float64)
        if raw.size != 5 * n or n != f["nnodes"]:
            raise SystemExit("%s: expected %d nodes x 5 arrays, got %d floats"
                             % (path, n, raw.size))
        a = raw.reshape(5, n)
        out = dict(x=a[0], y=a[1], theta=a[2], T=a[3], speed=a[4])
        if len(self._cache) > 6:
            self._cache.clear()
        self._cache[i] = out
        return out

    def polar_grid(self):
        return int(self.meta["nr"]), int(self.meta["nphi"])

    def build_interp(self, f0, targets):
        self._interp = PolarInterpolator(targets)
        self._interp.fit(f0["x"], f0["y"])
        self._interp_xy = (f0["x"], f0["y"])

    def field_grid(self, i, targets):
        f = self.read(i)
        nr, nphi = self.polar_grid()
        if self._interp is None or not (
                len(f["x"]) == len(self._interp_xy[0]) and
                np.array_equal(f["x"], self._interp_xy[0])):
            self.build_interp(f, targets)
        T = self._interp.apply(f["T"]).reshape(nphi, nr)
        sp = self._interp.apply(f["speed"]).reshape(nphi, nr)
        return dict(t=self.frames[i]["t"], T=T, speed=sp,
                    row=self.frames[i])

    def mesh_segments(self):
        """Element boundaries of the uniform nx x ny ring (exact arcs/lines)."""
        nx, ny = int(self.meta["nx"]), int(self.meta["ny"])
        r0, r1 = float(self.meta["R0"]), float(self.meta["R1"])
        segs = []
        n_arc = 720
        phi = np.linspace(0.0, 2.0 * math.pi, n_arc + 1)
        for k in range(nx + 1):
            rk = r0 + (r1 - r0) * k / nx
            pts = np.column_stack([rk * np.cos(phi), rk * np.sin(phi)])
            segs.append(np.stack([pts[:-1], pts[1:]], axis=1))
        pj = 2.0 * math.pi * np.arange(ny) / ny
        for p in pj:
            c, s = math.cos(p), math.sin(p)
            segs.append(np.array([[[r0 * c, r0 * s], [r1 * c, r1 * s]]]))
        return np.concatenate(segs)

    def mesh_note(self):
        return "%d x %d = %s cells" % (
            int(self.meta["nx"]), int(self.meta["ny"]),
            "{:,}".format(int(self.meta["nx"]) * int(self.meta["ny"])))

    # -- the per-frame adaptive mesh (ANN_MESH_DUMP=1 / v2 stream) -----------
    def mesh_frame(self, i):
        """The adaptive mesh of frame i, coloured by tree level.

        Returns dict(segments (n,2,2), levels (n,), leaves, level_min,
        level_max, note).  The dump is in logical coordinates (x = radial S,
        y = angular T); `segments` are the four logical edges of every leaf
        mapped to physical space with the card's own map
        (x,y) = (r cos(Phi T), r sin(Phi T)), r = R0 + (R1-R0) S.  A curved
        cell edge is therefore a chord on the ring (the logical cell is a
        quad); the binary node cloud and the panels use the interpolation, so
        only the overlay is chorded.
        """
        f = self.frames[i]
        path = f.get("mesh_path")
        if path is None:
            raise SystemExit("row %r: frame %d has no per-frame mesh dump "
                             "(run the example with ANN_MESH_DUMP=1)"
                             % (self.kind, f["frame"]))
        lev, x0, y0, hx, hy = [], [], [], [], []
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
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
            raise SystemExit("%s: empty mesh dump" % path)
        lev = np.asarray(lev, dtype="int64")
        x0 = np.asarray(x0, dtype="float64")
        y0 = np.asarray(y0, dtype="float64")
        hx = np.asarray(hx, dtype="float64")
        hy = np.asarray(hy, dtype="float64")
        if f.get("leaves") is not None and int(f["leaves"]) != len(lev):
            raise SystemExit("%s: %d leaves on disk, meta row says %d"
                             % (path, len(lev), int(f["leaves"])))
        if f.get("level_min") is not None and (
                int(lev.min()) != int(f["level_min"]) or
                int(lev.max()) != int(f["level_max"])):
            raise SystemExit("%s: levels %d..%d on disk, meta row says %d..%d"
                             % (path, int(lev.min()), int(lev.max()),
                                int(f["level_min"]), int(f["level_max"])))
        r0 = float(self.meta["R0"])
        r1 = float(self.meta["R1"])
        phi_max = float(self.meta["Phi"])
        S = np.stack([x0, x0 + hx, x0 + hx, x0, x0], axis=1)
        T = np.stack([y0, y0, y0 + hy, y0 + hy, y0], axis=1)
        R = r0 + (r1 - r0) * S
        P = phi_max * T
        X = R * np.cos(P)
        Y = R * np.sin(P)
        pts = np.stack([X, Y], axis=2)                     # (n, 5, 2)
        segs = np.stack([pts[:, :-1, :], pts[:, 1:, :]],
                        axis=2).reshape(-1, 2, 2)
        levels = np.repeat(lev, 4)
        lmin, lmax = int(lev.min()), int(lev.max())
        return dict(segments=segs, levels=levels, leaves=len(lev),
                    level_min=lmin, level_max=lmax,
                    note="%s leaves, level %d-%d"
                         % ("{:,}".format(len(lev)), lmin, lmax))

    def inset(self, i):
        f = self.frames[i]
        return dict(Nu_top=f["Nu_T"], Nu_bot=f["Nu_B"], Vrms=f["Vrms"],
                    meanT=f["meanT"], source="frame_meta.csv",
                    describe="t=%.6f" % f["t"])


# ------------------------------------------------------------------ ASPECT VTU
class AspectSource:
    """An ASPECT output directory: solution-NNNNN.PPPP.vtu pieces + statistics."""

    kind = "aspect"

    def __init__(self, path, r0=R0, r1=R1, stats=None):
        path = os.path.abspath(path)
        sol = os.path.join(path, "solution")
        self.root = path
        self.sol_dir = sol if os.path.isdir(sol) else path
        self.r0, self.r1 = float(r0), float(r1)
        self.pieces = {}
        pat = re.compile(r"solution-(\d+)\.(\d+)\.vtu$")
        for f in sorted(glob.glob(os.path.join(self.sol_dir, "solution-*"))):
            m = pat.search(os.path.basename(f))
            if m:
                self.pieces.setdefault(int(m.group(1)), []).append(f)
        if not self.pieces:
            raise SystemExit("no solution-NNNNN.PPPP.vtu pieces under %s"
                             % self.sol_dir)
        self.index = sorted(self.pieces)
        self.frames = []
        for idx in self.index:
            t = self._read_time(self.pieces[idx][0])
            self.frames.append(dict(t=t, idx=idx,
                                    npieces=len(self.pieces[idx])))
        self.frames.sort(key=lambda f: f["t"])
        self.times = [f["t"] for f in self.frames]
        self.stats = self._read_stats(stats)
        self._cache = {}
        self._interp = None
        self._interp_xy = None
        self._mesh = None
        self._mesh_cache = {}

    # -- time pass: the TIME field only -------------------------------------
    def _read_time(self, path):
        import meshio
        mesh = meshio.read(path)
        if "TIME" not in mesh.field_data:
            raise SystemExit("no TIME field in %s -- cannot order the frames"
                             % path)
        return float(np.asarray(mesh.field_data["TIME"]).ravel()[0])

    def __len__(self):
        return len(self.frames)

    def describe(self, i):
        return "frame %05d" % self.frames[i]["idx"]

    # -- statistics ---------------------------------------------------------
    def _read_stats(self, stats):
        """Per-output-time rows: t, Nu_top, Nu_bot, Vrms, meanT."""
        csvpath = None
        if stats:
            csvpath = os.path.abspath(stats)
        else:
            for cand in (os.path.join(self.sol_dir, "statistics"),
                         os.path.join(os.path.dirname(self.sol_dir),
                                      "statistics")):
                if os.path.isfile(cand):
                    csvpath = cand
                    break
        if csvpath is None:
            print("[annulus_video] ASPECT: no statistics file found; the inset "
                  "will show n/a", flush=True)
            return []
        rows = read_csv_rows(csvpath)
        if rows and "Nu_top" in rows[0]:
            out = [dict(t=float(r["t"]), Nu_top=float(r["Nu_top"]),
                        Nu_bot=float(r["Nu_bot"]), Vrms=float(r["Vrms"]),
                        meanT=float(r["meanT"]))
                   for r in rows if r.get("t")]
            print("[annulus_video] ASPECT diagnostics: %d rows from %s"
                  % (len(out), csvpath), flush=True)
            return out
        # fall back to the raw ASPECT statistics table
        out = self._parse_raw_statistics(csvpath)
        print("[annulus_video] ASPECT diagnostics: %d rows parsed from the raw "
              "statistics table %s" % (len(out), csvpath), flush=True)
        return out

    def _parse_raw_statistics(self, path):
        names, raw = {}, []
        try:
            with open(path, errors="replace") as fh:
                for line in fh:
                    s = line.strip()
                    m = re.match(r"#\s*(\d+):\s*(.*)", s)
                    if m:
                        names[int(m.group(1))] = m.group(2).strip()
                    elif s:
                        raw.append(s.split())
        except OSError:
            return []

        def col(tag):
            for k, name in sorted(names.items()):
                if name.startswith(tag):
                    return k - 1
            raise KeyError(tag)
        try:
            c_t = col("Time (seconds)")
            c_vrms = col("RMS velocity")
            c_qbot = col('Outward heat flux through boundary with indicator '
                         '0 ("bottom")')
            c_qtop = col('Outward heat flux through boundary with indicator '
                         '1 ("top")')
        except KeyError:
            return []
        try:
            c_tavg = col("Average temperature")
        except KeyError:
            try:
                c_tavg = col("Average nondimensional temperature")
            except KeyError:
                return []
        lnf = abs(math.log(self.r0 / self.r1))
        out = []
        for v in raw:
            try:
                qtop = float(v[c_qtop])
                qbot = float(v[c_qbot])
                out.append(dict(t=float(v[c_t]),
                                Nu_top=qtop * lnf / (2.0 * math.pi),
                                Nu_bot=-qbot * lnf / (2.0 * math.pi),
                                Vrms=float(v[c_vrms]),
                                meanT=float(v[c_tavg])))
            except (ValueError, IndexError):
                continue
        return out

    def stat_at(self, t):
        if not self.stats:
            return {}
        times = [r["t"] for r in self.stats]
        i = nearest_index(times, t)
        r = dict(self.stats[i])
        r["delta"] = r["t"] - t
        return r

    # -- field pass ---------------------------------------------------------
    def read(self, i):
        """dict(t, x, y, T, speed, quads, Porig) for one output time.

        All pieces are merged; the Q2 nodes that the VTU writes once per cell
        are de-duplicated (bit-identical duplicates on this output, measured),
        and the original per-cell point list is kept for the mesh panel, whose
        quads index it.
        """
        if i in self._cache:
            return self._cache[i]
        import meshio
        f = self.frames[i]
        P, T, V, quads = [], [], [], []
        npts = 0
        for p in self.pieces[f["idx"]]:
            mesh = meshio.read(p)
            P.append(np.asarray(mesh.points[:, :2], dtype="float64"))
            T.append(np.asarray(mesh.point_data["T"], dtype="float64"))
            vel = np.asarray(mesh.point_data["velocity"], dtype="float64")
            V.append(np.hypot(vel[:, 0], vel[:, 1]))
            for block in mesh.cells:
                if block.type == "quad":
                    quads.append(np.asarray(block.data) + npts)
            npts += len(P[-1])
        Porig = np.vstack(P)
        Torig = np.concatenate(T)
        sporig = np.concatenate(V)
        q = np.concatenate(quads) if quads else np.zeros((0, 4), dtype=np.int64)
        # de-duplicate the per-cell copies of the shared Q2 nodes (exact)
        key = np.round(Porig, 9)
        uid = np.unique(key, axis=0, return_index=True)[1]
        out = dict(t=f["t"], x=Porig[uid, 0], y=Porig[uid, 1],
                   T=Torig[uid], speed=sporig[uid], quads=q, Porig=Porig,
                   n_unique=len(uid), n_quads=len(q))
        if len(self._cache) > 4:
            self._cache.clear()
        self._cache[i] = out
        return out

    def polar_grid(self):
        return NR_DEFAULT, NPHI_DEFAULT

    def build_interp(self, f0, targets):
        self._interp = PolarInterpolator(targets)
        self._interp.fit(f0["x"], f0["y"])
        self._interp_xy = (f0["x"], f0["y"])

    def field_grid(self, i, targets):
        nr, nphi = self.polar_grid()
        f = self.read(i)
        if self._interp is None or not (
                len(f["x"]) == len(self._interp_xy[0]) and
                np.array_equal(f["x"], self._interp_xy[0])):
            self.build_interp(f, targets)
        T = self._interp.apply(f["T"]).reshape(nphi, nr)
        sp = self._interp.apply(f["speed"]).reshape(nphi, nr)
        return dict(t=f["t"], T=T, speed=sp, row=f)

    def cells(self, i):
        """Parent-cell corners (n,4,2): groups of 4 VTU quads on 9 Q2 nodes."""
        raise NotImplementedError  # see mesh_segments, which keeps the Q2 ring

    def mesh_segments(self):
        """Parent-cell boundaries, following the Q2 mid-edge nodes.

        The VTU writes each Q2 cell as four linear sub-quads on a 3x3 block of
        its nine nodes; consecutive quads come in groups of four (measured:
        3 072 groups for 12 288 quads, each group on exactly nine consecutive
        points).  The outer ring of the 3x3 block is the cell boundary, and it
        passes through the Q2 mid-edge nodes, so the curved cell edge is
        followed rather than chorded.
        """
        f = self.read(0)
        q = f["quads"]
        Porig = f["Porig"]
        if len(q) % 4 != 0:
            return np.zeros((0, 2, 2)), len(q)
        groups = q.reshape(-1, 4, 4)
        segs = []
        n_bad = 0
        for g in groups:
            ids = np.unique(g.ravel())
            if len(ids) != 9:
                n_bad += 1
                continue
            base = int(ids.min())
            # 3x3 local lattice, row-major (measured from the VTU quads):
            #   0 1 2 / 3 4 5 / 6 7 8
            ring = np.array([base + 0, base + 1, base + 2, base + 5,
                             base + 8, base + 7, base + 6, base + 3, base + 0])
            segs.append(np.stack([Porig[ring[:-1]], Porig[ring[1:]]], axis=1))
        if not segs:
            return np.zeros((0, 2, 2)), 0
        return np.concatenate(segs), n_bad

    def mesh_frame(self, i):
        """Parent-cell boundaries of frame i, coloured by inferred tree level.

        The level of each parent cell is inferred from its area and its radius
        (`aspect_cell_level`); the reconstruction follows the Q2 mid-edge nodes
        exactly like `mesh_segments`.  The result is cached: the AMR demo reads
        every shown frame once for the level legend and once more for the
        fields, and the VTU piece loading is the expensive part.
        """
        if i in self._mesh_cache:
            return self._mesh_cache[i]
        fr = self.read(i)
        q = fr["quads"]
        Porig = fr["Porig"]
        if len(q) == 0 or len(q) % 4 != 0:
            raise SystemExit("row %r: %d VTU quads is not a multiple of 4"
                             % (self.kind, len(q)))
        groups = q.reshape(-1, 4, 4)
        segs, levels = [], []
        n_bad = 0
        for g in groups:
            ids = np.unique(g.ravel())
            if len(ids) != 9:
                n_bad += 1
                continue
            base = int(ids.min())
            ring = np.array([base + 0, base + 1, base + 2, base + 5,
                             base + 8, base + 7, base + 6, base + 3, base + 0])
            segs.append(np.stack([Porig[ring[:-1]], Porig[ring[1:]]], axis=1))
            lv = aspect_cell_level(
                Porig[[base + 0, base + 2, base + 8, base + 6]])
            levels.append(0 if lv is None else int(lv))
        if not segs:
            raise SystemExit("row %r: no parent cells rebuilt from the VTU"
                             % self.kind)
        lev = np.asarray(levels, dtype="int64")
        out = dict(segments=np.concatenate(segs),
                   levels=np.repeat(lev, 8), cells=len(lev),
                   level_min=int(lev.min()), level_max=int(lev.max()),
                   bad=int(n_bad),
                   note="%s cells, level %d-%d"
                        % ("{:,}".format(len(lev)), int(lev.min()),
                           int(lev.max())))
        if len(self._mesh_cache) > 90:
            self._mesh_cache.clear()
        self._mesh_cache[i] = out
        return out

    def mesh_note(self):
        s = self.stat_at(self.times[0]) if self.stats else {}
        cells = s.get("cells") if s else None
        n = 3072 if cells is None else int(cells)
        return "%s cells (uniform)" % "{:,}".format(n)

    def inset(self, i):
        t = self.frames[i]["t"]
        s = self.stat_at(t)
        if not s:
            return dict(Nu_top=None, Nu_bot=None, Vrms=None, meanT=None,
                        source=None, describe="t=%.6f" % t)
        return dict(Nu_top=s["Nu_top"], Nu_bot=s["Nu_bot"], Vrms=s["Vrms"],
                    meanT=s["meanT"], source="statistics",
                    describe="stat row t=%.6f" % s["t"])


# --------------------------------------------------------------- interpolation
class PolarInterpolator:
    """Fixed Delaunay interpolation from a node cloud to the common polar grid.

    The node positions never move for either code (no AMR on this card), so the
    barycentric weights are built once and reused for every frame: the per-frame
    cost is one sparse matrix-vector product per field.
    """

    def __init__(self, targets):
        self.targets = targets
        self.W = None

    def fit(self, x, y):
        from scipy.spatial import Delaunay
        import scipy.sparse as sp
        pts = np.column_stack([x, y])
        tri = Delaunay(pts)
        s = tri.find_simplex(self.targets)
        if np.any(s < 0):
            bad = int(np.sum(s < 0))
            raise SystemExit("%d of %d polar-grid points fall outside the node "
                             "cloud's triangulation" % (bad,
                                                         len(self.targets)))
        X = tri.transform[s, :2]
        Y = self.targets - tri.transform[s, 2]
        b = np.einsum("ijk,ik->ij", X, Y)
        bary = np.column_stack([b, 1.0 - b.sum(axis=1)])
        verts = tri.simplices[s]
        rows = np.repeat(np.arange(len(s)), 3)
        cols = verts.ravel()
        data = bary.ravel()
        self.W = sp.csr_matrix((data, (rows, cols)),
                               shape=(len(s), len(pts)))
        return self

    def apply(self, values):
        return self.W @ np.asarray(values, dtype="float64")


class PanelSampler:
    """Fixed bilinear sampler from the polar analysis grid to one square panel.

    The disk is drawn as an RGBA image over the panel's bounding square (the
    panel is square because the domain is a disk), at `factor` times the panel
    resolution so the curved boundaries stay smooth when matplotlib downsamples
    it.  The sample indices and weights depend only on the geometry, so they are
    built once and every frame costs four gathers per field.  Pixels outside the
    ring are transparent (the panel background is white).
    """

    def __init__(self, panel_px, r_box, nr, nphi, r0=R0, r1=R1, factor=2):
        self.n = int(panel_px * factor)
        xs = -r_box + (np.arange(self.n) + 0.5) * (2.0 * r_box / self.n)
        X, Y = np.meshgrid(xs, xs)          # row = y, col = x, origin lower
        r = np.hypot(X, Y)
        p = np.mod(np.arctan2(Y, X), 2.0 * math.pi)
        self.mask = (r >= r0) & (r <= r1)
        dr = (r1 - r0) / nr
        dphi = 2.0 * math.pi / nphi
        fi = (r - r0) / dr - 0.5
        fj = p / dphi - 0.5
        i0 = np.floor(fi).astype(np.int64)
        j0 = np.floor(fj).astype(np.int64)
        wi = fi - i0
        wj = fj - j0
        i0c = np.clip(i0, 0, nr - 1)
        i1c = np.clip(i0 + 1, 0, nr - 1)
        j0m = np.mod(j0, nphi)
        j1m = np.mod(j0 + 1, nphi)
        base0 = j0m * nr
        base1 = j1m * nr
        self.idx = (base0 + i0c, base0 + i1c, base1 + i0c, base1 + i1c)
        w00 = (1.0 - wi) * (1.0 - wj)
        w01 = (1.0 - wi) * wj
        w10 = wi * (1.0 - wj)
        w11 = wi * wj
        self.w = (w00, w01, w10, w11)
        self.extent = (-r_box, r_box, -r_box, r_box)

    def rgba(self, C, cmap, norm):
        """RGBA image of the (nphi, nr) field C on the panel's pixel grid."""
        flat = np.asarray(C, dtype="float64").ravel()
        v = (self.w[0] * flat[self.idx[0]] + self.w[1] * flat[self.idx[1]]
             + self.w[2] * flat[self.idx[2]] + self.w[3] * flat[self.idx[3]])
        out = cmap(norm(v))
        out[..., 3] = np.where(self.mask, 1.0, 0.0)
        return out


# ---------------------------------------------------------------- frame plan
def build_plan(sources, labels, frame_dt, t0=None, t1=None):
    starts = [s.times[0] for s in sources]
    ends = [s.times[-1] for s in sources]
    if t1 is None:
        last = min(ends)
        t1 = math.floor(last / frame_dt + 1.0e-9) * frame_dt
    if t0 is None:
        t0 = 0.0
    if not (t1 >= t0 - 1.0e-12):
        raise SystemExit("empty window [%g, %g]" % (t0, t1))
    n = int(round((t1 - t0) / frame_dt)) + 1
    targets = [t0 + k * frame_dt for k in range(n)]
    tol = 0.5 * frame_dt + 1.0e-9
    plan = []
    for k, t in enumerate(targets):
        picks = []
        for j, s in enumerate(sources):
            i = nearest_index(s.times, t)
            delta = s.times[i] - t
            if abs(delta) > tol:
                raise SystemExit(
                    "row %r has no frame within %.3f of t=%.6f (nearest is %s "
                    "at t=%.6f, offset %+.3e); refusing to invent one"
                    % (labels[j], tol, t, s.describe(i), s.times[i], delta))
            picks.append(dict(row=j, label=labels[j], index=i, t=s.times[i],
                              delta=delta))
        plan.append(dict(k=k, target=t, picks=picks))
    print("[annulus_video] window t = [%s, %s], %d target times every %g"
          % (sigfig(t0), sigfig(t1), len(plan), frame_dt), flush=True)
    span = t1 - t0
    for j, (lab, s) in enumerate(zip(labels, sources)):
        deltas = [abs(fr["picks"][j]["delta"]) for fr in plan]
        used = len({fr["picks"][j]["index"] for fr in plan})
        print("  %-10s %4d frames, t = %.6f .. %.6f, %d distinct shown, "
              "max offset %.2e (%.2f%% of the window)"
              % (lab, len(s), s.times[0], s.times[-1], used, max(deltas),
                 100.0 * max(deltas) / span if span > 0 else 0.0), flush=True)
        if used < len(plan):
            print("     NOTE: %d target times repeat this row's nearest frame"
                  % (len(plan) - used), flush=True)
    return t0, t1, plan


# ------------------------------------------------------------------- drawing
def rect_px(x_px, y_top_px, w_px, h_px, px_h):
    """Figure-fraction rect for an axes placed by pixel coordinates from top."""
    return [x_px / PX_W, (px_h - y_top_px - h_px) / px_h,
            w_px / PX_W, h_px / px_h]


def text_px(fig, x_px, y_top_px, px_h, **kw):
    return fig.text(x_px / PX_W, 1.0 - y_top_px / px_h, **kw)


def add_field_panel(fig, ax):
    ax.set_axis_off()
    ax.set_xlim(-R1 * 1.03, R1 * 1.03)
    ax.set_ylim(-R1 * 1.03, R1 * 1.03)
    ax.set_aspect("equal")


def add_inset(fig, ax, x, y, fontsize=8.0):
    return fig.text(
        x, y, "", transform=ax.transAxes, color=FG, fontsize=fontsize,
        fontweight="bold", va="bottom", ha="left", linespacing=1.35,
        bbox=dict(facecolor="white", alpha=0.82, edgecolor="none",
                  boxstyle="round,pad=0.30"))


class Row:
    """One labelled row: T, theta, log10|u| and mesh panels plus their text."""

    def __init__(self, fig, geo, band, label, norms, cmaps):
        self.geo = geo
        px_h = geo["px_h"]
        w = geo["panel_w"]
        top = geo["panel_top"][band]
        self.norms = norms
        self.cmaps = cmaps
        # AMR mode (--amr): the third column is log10 eta = -3 T and the mesh
        # panel is redrawn every frame, coloured by tree level.  The default
        # mode (the benchmark videos) is untouched.
        self.amr = bool(geo.get("amr", False))
        self.axes = []
        for k in range(N_COLS):
            ax = fig.add_axes(rect_px(geo["x0"][k], top, w, w, px_h))
            add_field_panel(fig, ax)
            ax.set_facecolor(BG)
            self.axes.append(ax)
        self.ax_t, self.ax_th, self.ax_u, self.ax_m = self.axes
        self.ims = [None] * 3
        self.mesh_line = None
        self.sampler = geo["sampler"]
        self.label_text = fig.text(
            0.012, 0.988, label, transform=self.ax_t.transAxes, color=FG,
            fontsize=13.5, fontweight="bold", va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none",
                      boxstyle="round,pad=0.26"))
        # the time label sits in the top right corner of the T panel: the target
        # and the frame actually shown, so the matching is visible
        self.time_text = fig.text(
            0.986, 0.986, "", transform=self.ax_t.transAxes, color=FG,
            fontsize=10.0, fontweight="bold", va="top", ha="right",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none",
                      boxstyle="round,pad=0.20"))
        self.time_sub = fig.text(
            0.986, 0.938, "", transform=self.ax_t.transAxes, color=DIM,
            fontsize=7.4, va="top", ha="right",
            bbox=dict(facecolor="white", alpha=0.72, edgecolor="none",
                      boxstyle="round,pad=0.16"))
        self.nu_text = add_inset(fig, self.ax_t, 0.02, 0.02, 7.5)
        self.th_text = add_inset(fig, self.ax_th, 0.98, 0.02, 7.5)
        self.th_text.set_ha("right")
        self.u_text = add_inset(fig, self.ax_u, 0.98, 0.02, 7.5)
        self.u_text.set_ha("right")
        self.mesh_text = add_inset(fig, self.ax_m, 0.98, 0.02, 7.5)
        self.mesh_text.set_ha("right")

    def set_mesh(self, segments, color, lw=0.22, alpha=0.75):
        if self.mesh_line is not None:
            self.mesh_line.remove()
            self.mesh_line = None
        self.mesh_line = LineCollection(
            np.asarray(segments), colors=color, linewidths=lw, alpha=alpha,
            zorder=3)
        self.ax_m.add_collection(self.mesh_line)

    def set_mesh_colored(self, mesh, cmap, norm, lw=0.22, alpha=0.95):
        """Replace the mesh overlay with the adaptive cells coloured by level."""
        if self.mesh_line is not None:
            self.mesh_line.remove()
            self.mesh_line = None
        colors = cmap(norm(mesh["levels"]))
        self.mesh_line = LineCollection(
            np.asarray(mesh["segments"]), colors=colors, linewidths=lw,
            alpha=alpha, zorder=3)
        self.ax_m.add_collection(self.mesh_line)

    def update(self, target, pick, src, fg, ins, mesh=None,
               level_cmap=None, level_norm=None):
        """Draw the frame of row `pick['row']` at its actual time."""
        third = fg["logeta"] if self.amr else fg["logspeed"]
        fields = (fg["T"], fg["theta"], third)
        for k, (ax, cmap, norm, C) in enumerate(zip(
                self.axes[:3], self.cmaps, self.norms, fields)):
            img = self.sampler.rgba(C, cmap, norm)
            if self.ims[k] is None:
                self.ims[k] = ax.imshow(
                    img, extent=self.sampler.extent, origin="lower",
                    interpolation="bilinear", zorder=1)
            else:
                self.ims[k].set_data(img)
        if mesh is not None and level_cmap is not None:
            self.set_mesh_colored(mesh, level_cmap, level_norm)
        # time: the frame actually shown; the target as its offset
        self.time_text.set_text(r"$t = %.4f$" % fg["t"])
        d = fg["t"] - target
        if abs(d) > 5.0e-7:
            self.time_sub.set_text(r"$\mathrm{target}\ %.4f$  $\Delta\ %s$"
                                   % (target, time_delta_text(d)))
        else:
            self.time_sub.set_text(r"$\mathrm{target}\ %.4f$" % target)
        # per-row diagnostics of the shown frame
        def num(v, fmt="%.4f"):
            return "n/a" if v is None else fmt % v
        self.nu_text.set_text(
            r"$\mathrm{Nu}_{\mathrm{top}} = %s$  "
            r"$\mathrm{Nu}_{\mathrm{bot}} = %s$"
            "\n" r"$V_{\mathrm{rms}} = %s$  $\langle T\rangle = %s$"
            % (num(ins.get("Nu_top")), num(ins.get("Nu_bot")),
               num(ins.get("Vrms"), "%.4f"), num(ins.get("meanT"), "%.5f")))
        thmin, thmax = float(np.nanmin(fg["theta"])), float(np.nanmax(fg["theta"]))
        self.th_text.set_text(
            r"$\theta_{\min} = %+.3f$  $\theta_{\max} = %+.3f$"
            % (thmin, thmax))
        if self.amr:
            self.u_text.set_text(
                r"$\log_{10}\eta_{\min} = %.2f$  "
                r"$\log_{10}\eta_{\max} = %.2f$"
                % (float(np.nanmin(fg["logeta"])),
                   float(np.nanmax(fg["logeta"]))))
        else:
            self.u_text.set_text(
                r"$\log_{10}|u|_{\min} = %.2f$  "
                r"$\log_{10}|u|_{\max} = %.2f$"
                % (float(np.nanmin(fg["logspeed"])),
                   float(np.nanmax(fg["logspeed"]))))
        self.mesh_text.set_text(mesh["note"] if mesh is not None
                                else src.mesh_note())


def layout(px_w, n_rows):
    col_w = (px_w - 2 * PAD_X - (N_COLS - 1) * GAP) // N_COLS
    panel_h = col_w
    panel_top = [PANEL_TOP + r * (panel_h + ROW_GAP) for r in range(n_rows)]
    panels_end = PANEL_TOP + n_rows * panel_h + (n_rows - 1) * ROW_GAP
    cbar_top = panels_end + 16
    cbar_h = 13
    title_y = cbar_top + cbar_h + 28          # clear of the bar's tick labels
    px_h = title_y + 24
    px_h += px_h % 2                          # H.264 wants an even height
    x0 = [PAD_X + k * (col_w + GAP) for k in range(N_COLS)]
    return dict(px_w=px_w, px_h=px_h, col_w=col_w, panel_w=panel_h,
                panel_top=panel_top, x0=x0,
                cbar_top=cbar_top, cbar_h=cbar_h, title_y=title_y)


def build_figure(labels, norms, cmaps, geo, header, header2, titles=None,
                 mesh_title=None, level_legend=None):
    fig = plt.figure(figsize=(geo["px_w"] / DPI, geo["px_h"] / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG)
    rows = [Row(fig, geo, b, lab, norms, cmaps)
            for b, lab in enumerate(labels)]
    # one horizontal colour bar per field column, on a shared baseline
    if titles is None:
        titles = [r"$T$", r"$\theta = T - \langle T\rangle_\phi(r)$",
                  r"$\log_{10}|u|$, $|u|$ floor $10^{%d}$"
                  % int(round(math.log10(SPEED_FLOOR)))]
    if mesh_title is None:
        mesh_title = "uniform 3 072-cell mesh\n(element boundaries)"
    for k in range(3):
        cax = fig.add_axes(rect_px(geo["x0"][k], geo["cbar_top"],
                                   geo["panel_w"], geo["cbar_h"],
                                   geo["px_h"]))
        cb = fig.colorbar(ScalarMappable(norm=norms[k], cmap=cmaps[k]),
                          cax=cax, orientation="horizontal")
        cb.ax.tick_params(colors=DIM, labelsize=9, length=2.5, pad=2)
        cb.outline.set_edgecolor(SPINE)
        cb.outline.set_linewidth(0.7)
        ticks = geo["ticks"][k]
        cb.set_ticks(ticks)
        cb.set_ticklabels([r"$%.3g$" % t for t in ticks])
        text_px(fig, geo["x0"][k] + geo["panel_w"] / 2.0, geo["title_y"],
                geo["px_h"], s=titles[k], color=FG, fontsize=11.5,
                va="center", ha="center")
    if level_legend is not None:
        # AMR mode: the fourth column's colour bar is the discrete level scale,
        # shared by both rows (the level of a cell is the same tree-level scale
        # on both codes: 12 angular x 1 radial coarse cells).
        cax = fig.add_axes(rect_px(geo["x0"][3], geo["cbar_top"],
                                   geo["panel_w"], geo["cbar_h"],
                                   geo["px_h"]))
        cb = fig.colorbar(
            ScalarMappable(norm=level_legend["norm"],
                           cmap=level_legend["cmap"]),
            cax=cax, orientation="horizontal", spacing="uniform")
        cb.ax.tick_params(colors=DIM, labelsize=9, length=2.5, pad=2)
        cb.outline.set_edgecolor(SPINE)
        cb.outline.set_linewidth(0.7)
        cb.set_ticks(level_legend["ticks"])
        cb.set_ticklabels([str(t) for t in level_legend["ticks"]])
        text_px(fig, geo["x0"][3] + geo["panel_w"] / 2.0, geo["title_y"],
                geo["px_h"], s=level_legend["title"], color=FG,
                fontsize=11.5, va="center", ha="center")
    else:
        text_px(fig, geo["x0"][3] + geo["panel_w"] / 2.0,
                geo["title_y"] - 6, geo["px_h"], s=mesh_title,
                color=DIM, fontsize=8.0, va="center", ha="center",
                linespacing=1.2)
    # two header lines: the case, then the field/functional definitions.  The
    # AMR demo's first header is longer than the benchmark card's, so it is
    # drawn one point smaller to stay inside the 1200 px master (the default
    # path keeps 11.5).
    text_px(fig, PAD_X, 12, geo["px_h"], s=header,
            color=FG, fontsize=(10.0 if level_legend is not None else 11.5),
            va="center", ha="left")
    text_px(fig, PAD_X, 28, geo["px_h"], s=header2, color=HEADER,
            fontsize=8.5, va="center", ha="left")
    return fig, rows


# ------------------------------------------------------------------- encoder
def find_ffmpeg(explicit=None):
    cands = []
    if explicit:
        cands.append(explicit)
    env = os.environ.get("QUAST_FFMPEG")
    if env:
        cands.append(env)
    cands.append(shutil.which("ffmpeg") or "")
    try:
        import imageio_ffmpeg
        cands.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise SystemExit("no ffmpeg found; put it on PATH, set QUAST_FFMPEG, or "
                     "pass --ffmpeg PATH")


def host_path_for_ffmpeg(ffmpeg, path):
    """Paths for the encoder: a Windows ffmpeg.exe cannot open WSL /mnt paths.

    The repository scripts are expected to run on either side of the WSL
    boundary; when the encoder found is a Windows executable reached through
    /mnt, the frame pattern and the output path are converted with `wslpath -w`.
    A native ffmpeg gets the paths unchanged.
    """
    if not path:
        return path
    is_win_exe = ffmpeg.endswith(".exe") or ffmpeg.startswith("/mnt/")
    if is_win_exe and path.startswith("/mnt/"):
        try:
            win = subprocess.check_output(["wslpath", "-w", path],
                                          text=True).strip()
            if win:
                return win
        except Exception:
            pass
    return path


def encode(ffmpeg, frames_dir, pattern, fps, out, crf=18, preset="slow"):
    inp = host_path_for_ffmpeg(ffmpeg, os.path.join(frames_dir, pattern))
    outp = host_path_for_ffmpeg(ffmpeg, out)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
           "-i", inp,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
           "-preset", preset, "-r", str(fps), "-movflags", "+faststart", outp]
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("ffmpeg failed:\n%s\n%s" % (" ".join(cmd), r.stderr))
    return cmd, time.time() - t


def write_gif(pngs, out, fps, width_cap=None, colors=256, max_bytes=8.0e6):
    """Pillow GIF with one adaptive palette from a sample of the frames.

    The palette is shared by every frame (built from a montage of evenly spaced
    frames) so the colours do not flicker between frames; `optimize=True` then
    delta-compresses the frames.  If the result is over `max_bytes`, the width
    and the palette are stepped down and the GIF is rewritten, and the chosen
    parameters are returned.
    """
    from PIL import Image
    frames = [Image.open(p).convert("RGB") for p in pngs]
    w0, h0 = frames[0].size
    montage_step = max(1, len(frames) // 8)
    attempts = []
    if width_cap:
        attempts.append((width_cap, colors))
    for w in (w0, int(w0 * 0.86), int(w0 * 0.72)):
        for c in (colors, 160, 128):
            if (w, c) not in attempts:
                attempts.append((w, c))
    last = None
    for w, c in attempts:
        h = int(round(h0 * w / float(w0)))
        h += h % 2
        small = [fr.resize((w, h), Image.LANCZOS) for fr in frames]
        mont = Image.new("RGB", (w, h * len(small[::montage_step])))
        for k, fr in enumerate(small[::montage_step]):
            mont.paste(fr, (0, k * h))
        pal = mont.quantize(colors=c, method=Image.MEDIANCUT)
        qf = [fr.quantize(palette=pal, dither=Image.FLOYDSTEINBERG)
              for fr in small]
        qf[0].save(out, save_all=True, append_images=qf[1:],
                   duration=int(round(1000.0 / fps)), loop=0, optimize=True)
        size = os.path.getsize(out)
        last = (w, c, size)
        print("[annulus_video] gif attempt: %d px, %d colours -> %.2f MB"
              % (w, c, size / 1e6), flush=True)
        if size <= max_bytes:
            return last
    print("[annulus_video] gif did not fit %.1f MB (best %.2f MB at %d px, %d "
          "colours)" % (max_bytes / 1e6, last[2] / 1e6, last[0], last[1]),
          flush=True)
    return last


# ---------------------------------------------------------------------- main
def main(argv=None):
    global R0, R1
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row", action="append", required=True,
                    metavar="LABEL=KIND:PATH",
                    help="one row, repeated; KIND is quast or aspect, rows "
                         "are stacked top to bottom")
    ap.add_argument("--aspect-stats", default=None,
                    help="CSV from extract_aspect_statistics.py (default: "
                         "auto-detect the ASPECT statistics file)")
    ap.add_argument("--r0", type=float, default=R0,
                    help="inner radius of the card (default %g)" % R0)
    ap.add_argument("--r1", type=float, default=R1,
                    help="outer radius of the card (default %g)" % R1)
    ap.add_argument("--frame-dt", type=float, default=0.05,
                    help="target frame interval in time units (default 0.05)")
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--out", default="out_julia/annulus_video/"
                                    "quast_vs_aspect_annulus.mp4")
    ap.add_argument("--case", default="Davies et al. (2022) case 2.1")
    ap.add_argument("--header", default=None,
                    help="override the first header line (default: the case-2.1 "
                         "card text)")
    ap.add_argument("--header2", default=None,
                    help="override the second header line (default: the case-2.1 "
                         "window text)")
    ap.add_argument("--sidecar-extra", default=None, metavar="JSON",
                    help="JSON object merged into the .json sidecar (top-level "
                         "keys replace the defaults); used to record the "
                         "continuation protocol and the case-2.2 caveats of a "
                         "video whose window/rows differ from the default card")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--dtheta-max", type=float, default=None)
    ap.add_argument("--umin", type=float, default=None)
    ap.add_argument("--umax", type=float, default=None)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--ffmpeg", default=None)
    ap.add_argument("--gif-width", type=int, default=None,
                    help="cap the GIF width (default: try the master width first)")
    ap.add_argument("--gif-colors", type=int, default=256)
    ap.add_argument("--gif-max-mb", type=float, default=8.0)
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--amr", action="store_true",
                    help="AMR demonstration layout: the third column is "
                         "log10 eta = -3 T (eta = 1000^(-T)) and the fourth is "
                         "the adaptive mesh coloured by tree level, redrawn "
                         "every frame.  Needs QuaST frames written with "
                         "ANN_MESH_DUMP=1 and ASPECT VTU frames.")
    ap.add_argument("--aspect-level-base", type=int, default=3,
                    help="the initial global refinement of the ASPECT card "
                         "(default 3); used only to check the level inferred "
                         "per cell by aspect_cell_level")
    ap.add_argument("--still", default=None, metavar="PATH",
                    help="write one PNG at --still-index instead of the video")
    ap.add_argument("--still-index", type=int, default=-1)
    ap.add_argument("--list", action="store_true",
                    help="print the frame inventory of every row and exit")
    ap.add_argument("--rm-pngs", action="store_true")
    args = ap.parse_args(argv)

    R0, R1 = args.r0, args.r1

    # ---- rows
    labels, sources = [], []
    for spec in args.row:
        m = re.match(r"^(?:(?P<label>.*)=)?(?P<kind>[A-Za-z_]+):(?P<path>.+)$",
                     spec, re.S)
        if not m:
            raise SystemExit("row spec %r is not [LABEL=]KIND:PATH" % spec)
        kind = m.group("kind").strip().lower()
        path = m.group("path").strip()
        label = (m.group("label") or "").strip() \
            or os.path.basename(os.path.normpath(path))
        if kind == "quast":
            src = QuastSource(path)
        elif kind == "aspect":
            src = AspectSource(path, r0=args.r0, r1=args.r1,
                               stats=args.aspect_stats)
        else:
            raise SystemExit("unknown row kind %r (use quast: or aspect:)" % kind)
        labels.append(label)
        sources.append(src)
    for lab, s in zip(labels, sources):
        print("[annulus_video] row %-8s %-8s %4d frames  t = %.6f .. %.6f"
              % (lab, s.kind, len(s), s.times[0], s.times[-1]), flush=True)
        if len(s) == 0:
            raise SystemExit("row %r has no frames" % lab)
    if args.list:
        for lab, s in zip(labels, sources):
            print("\n== %s (%s)" % (lab, s.kind))
            for i, f in enumerate(s.frames):
                print("   %4d  t=%14.9f  %s" % (i, f["t"], s.describe(i)))
        return 0
    if args.amr:
        for lab, s in zip(labels, sources):
            if isinstance(s, QuastSource) and not s.mesh_dump:
                raise SystemExit(
                    "row %r: --amr needs QuaST frames written with "
                    "ANN_MESH_DUMP=1 (frame_meta.csv v2 + one "
                    "frame_NNNNN_mesh.csv per frame); this directory is a v1 "
                    "frame stream" % lab)
            if isinstance(s, QuastSource) and \
                    "level_min" not in s.frames[0]:
                raise SystemExit("row %r: the v2 frame_meta table has no "
                                 "level_min column" % lab)

    # ---- sanity: every row's node cloud sits on the card's ring, and the
    # QuaST metadata's geometry agrees with the radii the renderer was given
    for lab, s in zip(labels, sources):
        f0 = s.read(0)
        rr = np.hypot(f0["x"], f0["y"])
        if isinstance(s, QuastSource):
            if not (abs(s.meta["R0"] - R0) < 1e-9
                    and abs(s.meta["R1"] - R1) < 1e-9):
                raise SystemExit(
                    "row %r: frame metadata geometry R0=%r R1=%r disagrees "
                    "with --r0/--r1 %r/%r"
                    % (lab, s.meta["R0"], s.meta["R1"], R0, R1))
        print("[annulus_video] %-8s node cloud: %d nodes, r = [%.9f, %.9f]"
              % (lab, len(rr), float(rr.min()), float(rr.max())), flush=True)
        if abs(float(rr.min()) - R0) > 1e-3 or abs(float(rr.max()) - R1) > 1e-3:
            raise SystemExit(
                "row %r: node radii [%g, %g] are not the card's [%g, %g]"
                % (lab, float(rr.min()), float(rr.max()), R0, R1))

    # ---- the common analysis polar grid (one definition for both codes)
    nr, nphi = NR_DEFAULT, NPHI_DEFAULT
    for s in sources:
        if isinstance(s, QuastSource):
            nr, nphi = s.polar_grid()
            break
    r = R0 + (np.arange(nr) + 0.5) * (R1 - R0) / nr
    phi = (np.arange(nphi) + 0.5) * 2.0 * math.pi / nphi
    # target order is (phi, r): rows are angular, columns radial, matching the
    # analysis grids' shape (nphi, nr) and the pcolormesh cell layout
    Pg, Rg = np.meshgrid(phi, r, indexing="ij")
    targets = np.column_stack([(Rg * np.cos(Pg)).ravel(),
                               (Rg * np.sin(Pg)).ravel()])
    print("[annulus_video] common polar grid: %d radial x %d angular cell "
          "centres, r = [%g, %g]" % (nr, nphi, r[0], r[-1]), flush=True)

    def grid_for(src, i):
        fg = src.field_grid(i, targets)
        T = fg["T"]
        fg["theta"] = T - T.mean(axis=0, keepdims=True)
        fg["logspeed"] = np.log10(np.maximum(fg["speed"], SPEED_FLOOR))
        if args.amr:
            # eta = 1000^(-T) = 10^(-3 T), so log10 eta = -3 T exactly from the
            # total temperature the panels already show.  No eta samples are
            # needed in the frame stream (and none is written).
            fg["logeta"] = -3.0 * T
        return fg

    t0, t1, plan = build_plan(sources, labels, args.frame_dt,
                              t0=args.t0, t1=args.t1)

    # ASPECT inset provenance: each inset uses the statistics row nearest to
    # the VTU time actually shown; report the largest mismatch (the statistics
    # file has one row per timestep, so it should be ~0)
    stat_match = {}
    for j, (lab, s) in enumerate(zip(labels, sources)):
        if not (isinstance(s, AspectSource) and s.stats):
            continue
        dmax = 0.0
        for fr in plan:
            st = s.stat_at(fr["picks"][j]["t"])
            dmax = max(dmax, abs(st.get("delta", 0.0)))
        stat_match[lab] = dmax
        print("[annulus_video] %-8s statistics rows: max |t_stats - t_frame| "
              "over the shown frames = %.2e" % (lab, dmax), flush=True)
        if dmax > 5.0e-3:
            print("[annulus_video] WARNING: statistics/frame time mismatch > 5e-3",
                  flush=True)

    # ---- shared scales: one per column, over every shown frame of every row
    t_scan = time.time()
    t_lo, t_hi = [], []
    th_abs = 0.0
    u_lo, u_hi = [], []
    for j, s in enumerate(sources):
        for fr in plan:
            i = fr["picks"][j]["index"]
            fg = grid_for(s, i)
            raw = s.read(i)
            t_lo.append(float(np.nanmin(raw["T"])))
            t_hi.append(float(np.nanmax(raw["T"])))
            th_abs = max(th_abs, float(np.nanmax(np.abs(fg["theta"]))))
            u_lo.append(float(fg["logspeed"].min()))
            u_hi.append(float(fg["logspeed"].max()))
        print("[annulus_video] scanned %d frames of %r for the scales"
              % (len(plan), labels[j]), flush=True)
    vmin = args.vmin if args.vmin is not None else floor2(min(t_lo))
    vmax = args.vmax if args.vmax is not None else ceil2(max(t_hi))
    th_peak = (args.dtheta_max if args.dtheta_max is not None
               else ceil2(th_abs))
    u_vmin = args.umin if args.umin is not None else floor2(min(u_lo))
    u_vmax = args.umax if args.umax is not None else ceil2(max(u_hi))
    # AMR mode: the third column is log10 eta = -3 T (eta = 1000^(-T)) instead
    # of log10|u|; its scale is the same shared scan, mapped through -3T.  eta
    # is at most 1 (T >= 0), so the top of the scale never goes above 0.
    eta_vmin = floor2(-3.0 * max(t_hi))
    eta_vmax = min(0.0, ceil2(-3.0 * min(t_lo)))
    if args.amr:
        norms = [Normalize(vmin=vmin, vmax=vmax),
                 Normalize(vmin=-th_peak, vmax=th_peak),
                 Normalize(vmin=eta_vmin, vmax=eta_vmax)]
        cmaps = [plt.get_cmap(T_CMAP), plt.get_cmap(THETA_CMAP),
                 plt.get_cmap(ETA_CMAP)]
    else:
        norms = [Normalize(vmin=vmin, vmax=vmax),
                 Normalize(vmin=-th_peak, vmax=th_peak),
                 Normalize(vmin=u_vmin, vmax=u_vmax)]
        cmaps = [plt.get_cmap(T_CMAP), plt.get_cmap(THETA_CMAP),
                 plt.get_cmap(SPEED_CMAP)]
    if args.amr:
        print("[annulus_video] shared scales: T = [%s, %s], theta = +-[%s], "
              "log10 eta = [%s, %s] (eta = 1000^(-T) = 10^(-3T), evaluated "
              "from the total temperature)"
              % (sigfig(vmin), sigfig(vmax), sigfig(th_peak), sigfig(eta_vmin),
                 sigfig(eta_vmax)), flush=True)
    else:
        print("[annulus_video] shared scales: T = [%s, %s], theta = +-[%s], "
              "log10|u| = [%s, %s] (|u| floor %g)"
              % (sigfig(vmin), sigfig(vmax), sigfig(th_peak), sigfig(u_vmin),
                 sigfig(u_vmax), SPEED_FLOOR), flush=True)
    print("[annulus_video] scale scan took %.1f s" % (time.time() - t_scan),
          flush=True)

    # ---- AMR mode: the shared level legend and the per-row cell counts
    level_cmap = None
    level_norm = None
    level_legend = None
    cells_by_row = {}
    if args.amr:
        level_seen = set()
        for j, (lab, s) in enumerate(zip(labels, sources)):
            cells = []
            cmin, cmax = None, None
            for fr in plan:
                i = fr["picks"][j]["index"]
                if isinstance(s, QuastSource):
                    f = s.frames[i]
                    lo, hi = int(f["level_min"]), int(f["level_max"])
                    cells.append(int(f["leaves"]))
                else:
                    m = s.mesh_frame(i)
                    lo, hi = m["level_min"], m["level_max"]
                    cells.append(int(m["cells"]))
                level_seen.update(range(lo, hi + 1))
                cmin = lo if cmin is None else min(cmin, lo)
                cmax = hi if cmax is None else max(cmax, hi)
            cells_by_row[lab] = dict(cells_min=min(cells),
                                     cells_max=max(cells),
                                     level_min=int(cmin),
                                     level_max=int(cmax))
            print("[annulus_video] %-8s adaptive mesh over the shown frames: "
                  "%d .. %d cells, level %d-%d"
                  % (lab, min(cells), max(cells), cmin, cmax), flush=True)
            if isinstance(s, AspectSource) and cmin != args.aspect_level_base:
                print("[annulus_video] WARNING: %-8s coarsest inferred level "
                      "%d != Initial global refinement %d"
                      % (lab, cmin, args.aspect_level_base), flush=True)
        lev_min, lev_max = min(level_seen), max(level_seen)
        nlev = lev_max - lev_min + 1
        # discrete level colours, shared by both rows (the level of a cell is
        # the same tree-level scale on both codes: a 12-angular x 1-radial
        # coarse annulus refined to level L)
        if nlev <= len(LEVEL_COLORS):
            base = [to_rgba(c) for c in LEVEL_COLORS[:nlev]]
        else:
            base = plt.get_cmap(LEVEL_CMAP)(np.linspace(0.08, 0.92, nlev))
        level_cmap = ListedColormap(base)
        level_norm = BoundaryNorm(np.arange(lev_min - 0.5, lev_max + 1.5, 1.0),
                                  nlev)
        level_legend = dict(
            cmap=level_cmap, norm=level_norm,
            ticks=list(range(lev_min, lev_max + 1)),
            title="AMR mesh, colour = level (%d-%d)" % (lev_min, lev_max))
        print("[annulus_video] shared level legend: %d..%d (%d levels)"
              % (lev_min, lev_max, nlev), flush=True)

    # ---- geometry of the master
    geo = layout(PX_W, len(sources))
    r_box = R1 * 1.03
    geo["r_box"] = r_box
    geo["sampler"] = PanelSampler(geo["panel_w"], r_box, nr, nphi, r0=R0,
                                  r1=R1, factor=2)
    geo["amr"] = bool(args.amr)
    geo["ticks"] = [
        [vmin, (vmin + vmax) / 2.0, vmax],
        [-th_peak, 0.0, th_peak],
        ([eta_vmin, (eta_vmin + eta_vmax) / 2.0, eta_vmax] if args.amr
         else [u_vmin, (u_vmin + u_vmax) / 2.0, u_vmax]),
    ]
    if args.amr:
        header = ("%s  |  annulus Ra = 1e3 at the reference viscosity 1 "
                  "(local hot-boundary Ra = 1e6), eta = 1000^(-T)  |  AMR on "
                  "both codes  |  QuaST (top) vs ASPECT (bottom)" % args.case)
        header2 = ("t = %s .. %s, %d frames @ %g fps  |  theta = T - angular "
                   "mean at r on a common %d x %d polar grid  |  log10 eta = "
                   "-3 T (from the same total T)  |  mesh colour = tree level"
                   % (sigfig(t0), sigfig(t1), len(plan), args.fps, nr, nphi))
    else:
        header = ("%s  |  isoviscous Ra = 1e4 annulus (R0 = %g, R1 = %g)  |  "
                  "matched 3 072-cell mesh  |  QuaST (top) vs ASPECT (bottom)"
                  % (args.case, R0, R1))
        header2 = ("t = 0 .. 2, %d frames @ %g fps  |  theta = T - angular mean "
                   "at r on a common %d x %d polar grid  |  Nu = each code's own "
                   "wall-flux family (QuaST GLL gradient vs ASPECT flux "
                   "statistics)"
                   % (len(plan), args.fps, nr, nphi))
    # Case-2.2 / continuation videos reuse the renderer with their own one-line
    # header overrides (the default strings are the case-2.1 card's).
    if args.header is not None:
        header = args.header
    if args.header2 is not None:
        header2 = args.header2
    print("[annulus_video] master %d x %d px, panels %d px"
          % (geo["px_w"], geo["px_h"], geo["panel_w"]), flush=True)
    if args.amr:
        amr_titles = [r"$T$", r"$\theta = T - \langle T\rangle_\phi(r)$",
                      r"$\log_{10}\eta = -3T$  ($\eta = 1000^{-T}$)"]
    else:
        amr_titles = None
    fig, rows = build_figure(labels, norms, cmaps, geo, header, header2,
                             titles=amr_titles,
                             level_legend=level_legend if args.amr else None)

    # ---- the meshes.  The benchmark videos draw one fixed mesh per row; the
    # AMR demo redraws each row's mesh every frame from the frame's own tree.
    mesh_info = {}
    if args.amr:
        for lab in labels:
            c = cells_by_row[lab]
            mesh_info[lab] = dict(
                kind="adaptive quadtree, coloured by tree level per frame",
                cells_min=c["cells_min"], cells_max=c["cells_max"],
                level_min=c["level_min"], level_max=c["level_max"])
            print("[annulus_video] %s mesh: %s" % (lab, mesh_info[lab]),
                  flush=True)
    else:
        for j, (lab, s) in enumerate(zip(labels, sources)):
            t_mesh = time.time()
            if isinstance(s, QuastSource):
                segs = s.mesh_segments()
                if len(segs) == 0:
                    raise SystemExit("row %r: no mesh segments" % lab)
                rows[j].set_mesh(segs, MESH_COL, lw=0.20, alpha=0.70)
                mesh_info[lab] = dict(kind="uniform ring, element boundaries "
                                           "reconstructed from nx x ny",
                                      nx=int(s.meta["nx"]), ny=int(s.meta["ny"]),
                                      cells=int(s.meta["nx"]) * int(s.meta["ny"]),
                                      segments=int(len(segs)))
            else:
                segs, n_bad = s.mesh_segments()
                if len(segs) == 0:
                    raise SystemExit("row %r: no mesh segments" % lab)
                rows[j].set_mesh(segs, MESH_COL, lw=0.20, alpha=0.70)
                mesh_info[lab] = dict(kind="parent 3 072 cells rebuilt from the "
                                           "VTU 2x2 Q2 groups",
                                      cells=3072, segments=int(len(segs)),
                                      ungrouped=int(n_bad))
            print("[annulus_video] %s mesh: %s (%.1f s)"
                  % (lab, mesh_info[lab], time.time() - t_mesh), flush=True)

    # ---- render
    out = os.path.abspath(args.out)
    stem, _ = os.path.splitext(out)
    frames_dir = stem + "_frames"
    os.makedirs(frames_dir, exist_ok=True)
    for old in glob.glob(os.path.join(frames_dir, "f_*.png")):
        os.remove(old)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    t_render = time.time()
    still_index = args.still_index if args.still_index >= 0 else len(plan) // 2
    fgs = {}
    for k, fr in enumerate(plan):
        for j, (row, pick) in enumerate(zip(rows, fr["picks"])):
            src = sources[j]
            key = (j, pick["index"])
            if key not in fgs:
                if len(fgs) > 12:
                    fgs.clear()
                fgs[key] = grid_for(src, pick["index"])
            fg = fgs[key]
            mesh = None
            if args.amr:
                mesh = src.mesh_frame(pick["index"])
            row.update(fr["target"], pick, src, fg, src.inset(pick["index"]),
                       mesh=mesh, level_cmap=level_cmap,
                       level_norm=level_norm)
        if args.still:
            if k == still_index:
                fig.savefig(args.still, dpi=DPI, facecolor=fig.get_facecolor())
                print("[annulus_video] still -> %s" % args.still, flush=True)
                plt.close(fig)
                return 0
        else:
            path = os.path.join(frames_dir, "f_%05d.png" % k)
            fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
            if k % 5 == 0 or k == len(plan) - 1:
                print("[annulus_video] frame %d/%d  %s  (%.1f s)"
                      % (k + 1, len(plan), os.path.basename(path),
                         time.time() - t_render), flush=True)
    plt.close(fig)
    render_s = time.time() - t_render

    # ---- poster
    poster = stem + "_poster.png"
    shutil.copyfile(os.path.join(frames_dir, "f_%05d.png" % (len(plan) - 1)),
                    poster)

    # ---- mp4
    ffmpeg = find_ffmpeg(args.ffmpeg)
    t_enc = time.time()
    cmd, dt_enc = encode(ffmpeg, frames_dir, "f_%05d.png", args.fps, out,
                         crf=args.crf, preset=args.preset)
    print("[annulus_video] encoded %s (%.1f s) with %s"
          % (out, dt_enc, ffmpeg), flush=True)

    # ---- gif
    gif = None
    gif_info = None
    if not args.no_gif:
        gif = stem + ".gif"
        t_gif = time.time()
        gif_info = write_gif(sorted(glob.glob(os.path.join(frames_dir,
                                                           "f_*.png"))),
                             gif, args.fps, width_cap=args.gif_width,
                             colors=args.gif_colors,
                             max_bytes=args.gif_max_mb * 1e6)
        print("[annulus_video] gif %s  %d px  %d colours  %.2f MB (%.1f s)"
              % (gif, gif_info[0], gif_info[1], gif_info[2] / 1e6,
                 time.time() - t_gif), flush=True)

    # ---- sidecar
    rows_meta = []
    for lab, s in zip(labels, sources):
        d = dict(label=lab, kind=s.kind,
                 path=(s.dir if isinstance(s, QuastSource) else s.sol_dir),
                 nframes=len(s), t_range=[s.times[0], s.times[-1]])
        if isinstance(s, QuastSource):
            d.update(frame_meta=os.path.join(s.dir, "frame_meta.csv"),
                     mesh=s.meta)
            d["nu_definition"] = "one-sided GLL wall gradient (family A)"
        else:
            d.update(statistics="%d rows" % len(s.stats),
                     nu_definition="ASPECT heat flux statistics (consistent "
                                   "boundary flux, family C)")
            d["points_per_vtu"] = s.read(0)["n_unique"]
        rows_meta.append(d)
    sidecar = dict(
        case=args.case,
        card=dict(geometry="PolarAnnulus(R0=%.17g, R1=%.17g, Phi=2*pi; "
                           "seam=:periodic)" % (R0, R1),
                  Ra=1e4, eta=1.0, H=0.0,
                  walls="free slip on both circles",
                  temperatures="T = 1 inner, T = 0 outer",
                  initial_condition="ASPECT spherical hexagonal perturbation, "
                                    "angular mode 4: s = (R1-r)/(R1-R0), "
                                    "T = s + 0.2 s (1-s) cos(4 atan2(y,x))",
                  mesh="QuaST 16 x 192 Q2-Q1-Q2 leaves vs ASPECT Initial "
                       "global refinement = 4 (192 x 16 Q2-Q1-Q2 cells)",
                  time="t = 0 -> 2, BDF2, CFL 10, one Picard iteration/step"),
        layout=dict(width=geo["px_w"], height=geo["px_h"],
                    panel_px=geo["panel_w"], rows=len(sources),
                    columns=["T", "theta_ang", "log10|u|", "mesh"],
                    panel_aspect="square (the disk bounding box)",
                    aspect_ratio_note=(
                        "The disk fills a square panel, so the master is about "
                        "%d x %d px (ratio %.2f:1), not the 2.5:1 field of the "
                        "docs/assets/quast_vs_aspect_realistic.gif box video; "
                        "the difference is the annular geometry, and no field "
                        "is stretched: each panel is square."
                        % (geo["px_w"], geo["px_h"],
                           geo["px_w"] / float(geo["px_h"])))),
        theta_definition=dict(
            formula="theta_ang = T - <T>_phi(r)",
            grid="common polar grid of %d radial x %d angular cell centres "
                 "(%d points) evaluated identically for both codes"
                 % (nr, nphi, nr * nphi),
            interpolation="linear (Delaunay barycentric) from each code's own "
                          "Q2 node cloud; interpolation error O(h^2) ~ 5e-4 "
                          "on T, below the colour resolution",
            note="T = semquad_base_T(r) + theta on QuaST's nodes; ASPECT's T "
                 "is its full temperature field.  <T>_phi(r) is a diagnostic "
                 "of the state, not a solver variable."),
        mesh_definition=dict(
            quast="element boundaries of the uniform 16 x 192 ring: 17 "
                  "concentric circles and 192 radial lines (analytic)",
            aspect="3 072 parent cell boundaries rebuilt from the VTU's 2x2 "
                   "groups of linear quads (each group is one Q2 cell on nine "
                   "nodes); the Q2 mid-edge nodes are kept, so the curved cell "
                   "edges are followed",
            common="both 3 072 uniform cells on the same 16 radial x 192 "
                   "angular layout"),
        speed_definition="log10 of speed = hypot(u_x, u_y); shared scale; "
                         "speed below %g drawn at the floor" % SPEED_FLOOR,
        statistics_match=dict((k, v) for k, v in stat_match.items()),
        scales=dict(T=dict(vmin=vmin, vmax=vmax, cmap=T_CMAP),
                    theta=dict(vmin=-th_peak, vmax=th_peak, cmap=THETA_CMAP),
                    log10_speed=dict(vmin=u_vmin, vmax=u_vmax,
                                     cmap=SPEED_CMAP, floor=SPEED_FLOOR)),
        fps=args.fps, nframes=len(plan), duration_s=len(plan) / args.fps,
        frame_dt=args.frame_dt, t0=t0, t1=t1,
        matching=dict(rule="nearest frame time within +-frame_dt/2, no "
                           "interpolation in time",
                      targets=[dict(k=fr["k"], t=fr["target"],
                                    picks=[dict(
                                        row=labels[p["row"]],
                                        index=p["index"],
                                        t=p["t"],
                                        delta=p["delta"],
                                        describe=sources[p["row"]].describe(
                                            p["index"]),
                                        inset=sources[p["row"]].inset(
                                            p["index"]))
                                        for p in fr["picks"]])
                               for fr in plan]),
        rows=rows_meta, mesh=mesh_info,
        outputs=dict(mp4=out, poster=poster, gif=gif,
                     gif_info=(None if gif_info is None else
                               dict(width=gif_info[0], colors=gif_info[1],
                                    bytes=gif_info[2])),
                     frames_dir=frames_dir,
                     png_pattern="f_%05d.png"),
        ffmpeg=ffmpeg, encode_cmd=cmd, encode_s=dt_enc, render_s=render_s,
        command=" ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
        caption=(
            "The same Davies et al. (2022) case-2.1 annulus (Ra = 1e4, "
            "isoviscous, free slip) run by QuaST (top) and ASPECT (bottom) on "
            "the same 3 072-cell mesh from the same initial condition: total "
            "temperature, its angular-mean anomaly, log10 |u| (the viscosity "
            "panel of the box comparison is replaced because eta = 1 "
            "everywhere here), and the element boundaries.  Each row prints "
            "the time of the frame it shows and its own Nu_top, Nu_bot, V_rms "
            "and mean T."),
        notes=[
            "The first target (t = 0) is matched to QuaST's first frame "
            "(t = 1e-3, the state after its first Stokes solve and first "
            "energy step) and ASPECT's t = 0 output (its timestep-0 state: "
            "initial temperature plus initial Stokes velocity).  The velocity "
            "is the same quasi-static state (V_rms 10.0584 vs 10.0582), while "
            "QuaST's one 1e-3 step has already advected the perturbation "
            "(theta extremes 0.061 vs 0.050); QuaST's driver exposes no "
            "pre-step state, so no initial-condition frame is written for it.",
            "Nu_top/Nu_bot are the two codes' own wall-flux functionals, not "
            "the same family: QuaST's one-sided GLL wall gradient and "
            "ASPECT's consistent boundary flux differ by a few per cent on "
            "this 3 072-cell mesh (see docs/en/validation.md, cross-code "
            "section); V_rms and mean T agree closely.",
            "During the spin-up the two rows' insets differ visibly (e.g. at "
            "target t = 0.15: Nu_top 5.08 vs 4.73, mean T 0.3868 vs 0.3890), "
            "because the two codes advance the transient with different "
            "timestep sequences and integrators (QuaST BDF2 at dt ~ 5.3e-3, "
            "ASPECT at dt ~ 3e-3) and the transient is not a published target; "
            "from t ~ 0.4 both rows show the same steady state and the final "
            "frame agrees to 1e-5 in V_rms/mean T with identical theta "
            "extremes.",
            "ASPECT's visualization output writes each Q2 cell as four linear "
            "sub-quads on the cell's nine Q2 nodes; the renderer reconstructs "
            "the 3 072 parent cells for the mesh panel.",
        ],
        dev=dict(numpy=np.__version__,
                 matplotlib=matplotlib.__version__,
                 python=sys.version.split()[0]),
    )
    # AMR demonstration mode records its own column/scale/mesh provenance.  The
    # card, caption and caveats are supplied from the tracked JSON
    # (`--sidecar-extra`) so the numbers in them stay traceable to the run logs;
    # the keys below are the measured facts the renderer itself knows.
    if args.amr:
        sidecar["layout"]["columns"] = ["T", "theta_ang", "log10_eta",
                                        "mesh(level)"]
        sidecar.pop("speed_definition", None)
        sidecar["scales"] = dict(
            T=dict(vmin=vmin, vmax=vmax, cmap=T_CMAP),
            theta=dict(vmin=-th_peak, vmax=th_peak, cmap=THETA_CMAP),
            log10_eta=dict(vmin=eta_vmin, vmax=eta_vmax, cmap=ETA_CMAP,
                           definition="log10 eta = -3 T exactly (eta = "
                                      "1000^(-T) = 10^(-3T)), evaluated from "
                                      "the same total temperature field"),
            level=dict(vmin=lev_min, vmax=lev_max,
                       colors="discrete %s, one colour per tree level"
                              % LEVEL_CMAP))
        sidecar["eta_definition"] = dict(
            law="eta = 1000^(-T) = 10^(-3 T); ASPECT `VoT` material model "
                "(image plugin), QuaST eta_law = :expT with deta_T = 1000",
            panel="log10 eta = -3 T on the common polar grid, computed from "
                  "the same total temperature T as the first column; the frame "
                  "stream writes no viscosity samples",
            note="the panel displays the material law's viscosity, it is not "
                 "an independent measurement")
        sidecar["mesh_definition"] = dict(
            quast="per-frame leaves of the balanced AMR tree, read from "
                  "frame_%05d_mesh.csv (logical coordinates level,x0,y0,hx,hy; "
                  "r = R0 + (R1-R0) x, phi = Phi y), drawn as quads",
            aspect="per-frame parent cells rebuilt from the VTU's 2x2 groups of "
                   "linear quads (one Q2 cell on nine nodes); each cell's tree "
                   "level is inferred from its area and radius, "
                   "L = 0.5 log2((2 pi/12)/(A/r_mid))",
            levels="both codes are 12-angular x 1-radial coarse annulus "
                   "quadtrees, so a level means the same physical refinement "
                   "scale on both; the two codes still run independent adaptive "
                   "meshes with different cell counts and level ranges",
            common="the mesh panel is redrawn from that frame's own tree; the "
                   "mesh is not interpolated in time (nearest frame, like the "
                   "fields)")
        sidecar["levels"] = dict(
            shared_min=lev_min, shared_max=lev_max,
            ticks=list(range(lev_min, lev_max + 1)),
            rule="QuaST: tree level of the leaf (from the mesh dump).  ASPECT: "
                 "inferred per parent cell from A/r_mid; the coarsest inferred "
                 "level is checked against Initial global refinement = %d."
                 % args.aspect_level_base,
            by_row=cells_by_row,
            legend="one discrete colour per level on a shared scale")
        sidecar["cells_by_row"] = cells_by_row
        sidecar["caption"] = (
            "A purpose-configured annulus demonstration of the adaptive path "
            "(Ra = 1e3 at the reference viscosity 1, eta = 1000^(-T), AMR on "
            "both codes), run by QuaST (top) and ASPECT (bottom): total "
            "temperature, its angular-mean anomaly, log10 eta = -3 T, and each "
            "code's own adaptive mesh coloured by tree level.  This is a "
            "demonstration, not a validation; the two meshes are independent.")
        sidecar["notes"] = [
            "The two codes' meshes are different (QuaST's gradient-jump "
            "indicator + wall minimum level; ASPECT's temperature strategy), so "
            "the mesh panels compare two independent adaptive discretizations, "
            "not one mesh.",
            "This is a demonstration of the AMR path, not an accuracy claim and "
            "not the published VoT benchmark case.",
        ]
    # A continuation video reuses the renderer but records its own card, window,
    # provenance and caveats: `--sidecar-extra` merges top-level keys into the
    # sidecar (a key it contains replaces the default; keys it does not contain
    # keep the measured values above).
    if args.sidecar_extra:
        with open(args.sidecar_extra, "r", encoding="utf-8") as fh:
            extra = json.load(fh)
        if not isinstance(extra, dict):
            raise SystemExit("--sidecar-extra must contain a JSON object")
        sidecar.update(extra)
        print("[annulus_video] sidecar extra: %d keys from %s"
              % (len(extra), args.sidecar_extra), flush=True)
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=1)
    size = os.path.getsize(out)
    print("[annulus_video] %s  %.2f MB  %d frames @ %g fps = %.2f s"
          % (out, size / 1e6, len(plan), args.fps, len(plan) / args.fps),
          flush=True)
    print("[annulus_video] poster %s" % poster, flush=True)
    if gif:
        print("[annulus_video] gif %s  %.2f MB"
              % (gif, os.path.getsize(gif) / 1e6), flush=True)
    print("[annulus_video] sidecar %s" % (stem + ".json"), flush=True)
    if args.rm_pngs:
        shutil.rmtree(frames_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)



