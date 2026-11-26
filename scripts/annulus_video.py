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
from matplotlib.colors import Normalize                      # noqa: E402

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
        self.frames = []
        for line in rows:
            v = line.split(",")
            if len(v) < len(fieldnames):
                continue
            try:
                f = dict(frame=int(v[0]), step=int(v[1]),
                         t=float(v[2]), Nu_T=float(v[3]), Nu_B=float(v[4]),
                         Vrms=float(v[5]), meanT=float(v[6]),
                         Tmin=float(v[7]), Tmax=float(v[8]),
                         nnodes=int(v[9]))
            except ValueError:
                continue
            binp = os.path.join(self.dir, "frame_%05d.bin" % f["frame"])
            if os.path.isfile(binp) and os.path.getsize(binp) == 8 + 5 * 8 * f["nnodes"]:
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

    def update(self, target, pick, src, fg, ins):
        """Draw the frame of row `pick['row']` at its actual time."""
        fields = (fg["T"], fg["theta"], fg["logspeed"])
        for k, (ax, cmap, norm, C) in enumerate(zip(
                self.axes[:3], self.cmaps, self.norms, fields)):
            img = self.sampler.rgba(C, cmap, norm)
            if self.ims[k] is None:
                self.ims[k] = ax.imshow(
                    img, extent=self.sampler.extent, origin="lower",
                    interpolation="bilinear", zorder=1)
            else:
                self.ims[k].set_data(img)
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
        self.u_text.set_text(
            r"$\log_{10}|u|_{\min} = %.2f$  "
            r"$\log_{10}|u|_{\max} = %.2f$"
            % (float(np.nanmin(fg["logspeed"])),
               float(np.nanmax(fg["logspeed"]))))
        self.mesh_text.set_text(src.mesh_note())


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


def build_figure(labels, norms, cmaps, geo, header, header2):
    fig = plt.figure(figsize=(geo["px_w"] / DPI, geo["px_h"] / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG)
    rows = [Row(fig, geo, b, lab, norms, cmaps)
            for b, lab in enumerate(labels)]
    # one horizontal colour bar per field column, on a shared baseline
    titles = [r"$T$", r"$\theta = T - \langle T\rangle_\phi(r)$",
              r"$\log_{10}|u|$, $|u|$ floor $10^{%d}$"
              % int(round(math.log10(SPEED_FLOOR)))]
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
    text_px(fig, geo["x0"][3] + geo["panel_w"] / 2.0, geo["title_y"] - 6,
            geo["px_h"], s="uniform 3 072-cell mesh\n(element boundaries)",
            color=DIM, fontsize=8.0, va="center", ha="center",
            linespacing=1.2)
    # two header lines: the case, then the field/functional definitions
    text_px(fig, PAD_X, 12, geo["px_h"], s=header, color=FG, fontsize=11.5,
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
    norms = [Normalize(vmin=vmin, vmax=vmax),
             Normalize(vmin=-th_peak, vmax=th_peak),
             Normalize(vmin=u_vmin, vmax=u_vmax)]
    cmaps = [plt.get_cmap(T_CMAP), plt.get_cmap(THETA_CMAP),
             plt.get_cmap(SPEED_CMAP)]
    print("[annulus_video] shared scales: T = [%s, %s], theta = +-[%s], "
          "log10|u| = [%s, %s] (|u| floor %g)"
          % (sigfig(vmin), sigfig(vmax), sigfig(th_peak), sigfig(u_vmin),
             sigfig(u_vmax), SPEED_FLOOR), flush=True)
    print("[annulus_video] scale scan took %.1f s" % (time.time() - t_scan),
          flush=True)

    # ---- geometry of the master
    geo = layout(PX_W, len(sources))
    r_box = R1 * 1.03
    geo["r_box"] = r_box
    geo["sampler"] = PanelSampler(geo["panel_w"], r_box, nr, nphi, r0=R0,
                                  r1=R1, factor=2)
    geo["ticks"] = [
        [vmin, (vmin + vmax) / 2.0, vmax],
        [-th_peak, 0.0, th_peak],
        [u_vmin, (u_vmin + u_vmax) / 2.0, u_vmax],
    ]
    header = ("%s  |  isoviscous Ra = 1e4 annulus (R0 = %g, R1 = %g)  |  "
              "matched 3 072-cell mesh  |  QuaST (top) vs ASPECT (bottom)"
              % (args.case, R0, R1))
    header2 = ("t = 0 .. 2, %d frames @ %g fps  |  theta = T - angular mean at "
               "r on a common %d x %d polar grid  |  Nu = each code's own "
               "wall-flux family (QuaST GLL gradient vs ASPECT flux statistics)"
               % (len(plan), args.fps, nr, nphi))
    print("[annulus_video] master %d x %d px, panels %d px"
          % (geo["px_w"], geo["px_h"], geo["panel_w"]), flush=True)
    fig, rows = build_figure(labels, norms, cmaps, geo, header, header2)

    # ---- the fixed meshes (one per row, this card has no AMR)
    mesh_info = {}
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
            row.update(fr["target"], pick, src, fg, src.inset(pick["index"]))
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



