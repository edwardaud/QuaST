#!/usr/bin/env python3
"""Stack one or more runs of the same convection case into one labelled video.

One row per run: the video is a verification comparison of the same 2.5 x 1
case solved by (top) our sparse direct solver, (middle) our matrix-free
multigrid, (bottom) ASPECT.  Every row is drawn at the *same* pixel size, with
a *shared* temperature colour scale, so a difference between the rows is a
difference in the solution and not in the plot.  The number of rows comes from
the number of `--row` arguments: N rows give a 1920 x (N*720) master, so a
single arm can be rendered alone without empty bands.

    python3 scripts/three_row_video.py \
        --row "sparse direct (MUMPS)=our:out_julia/verify100_direct" \
        --row "matrix-free MG=our:out_julia/verify100_mg" \
        --row "ASPECT=aspect:aspect_runs/realistic/run_verify100" \
        --t0 1e-5 --t1 1e-4 --frames 60 --fps 15 \
        --out out_julia/verify100_render/verify100_3row.mp4

    # one arm only: 1920 x 720
    python3 scripts/three_row_video.py \
        --row "matrix-free MG=our:out_julia/verify100_mg" --fps 10 \
        --out out_julia/verify100_render/mg_only_1row.mp4

Frame sources
-------------
`our:<path>`     a run root with `frames/` (or the `frames/` directory itself):
                 `frame_NNNNN_T.bin` + `frame_NNNNN_mesh.csv` + `frame_meta.csv`.
                 Read with live_render's readers -- the same ng x ng float64
                 (x fastest, `[z,x]`) and the same `level,x0,y0,h` mesh boxes.
`aspect:<path>`  an ASPECT output directory (or its `solution/` directory):
                 `solution-NNNNN.PPPP.vtu` pieces.  All pieces of one output are
                 merged, duplicate MPI-interface nodes dropped, and the vertex
                 field `T` interpolated onto a regular grid (default 640 x 256).
                 Dimensional input is detected and converted with the scale set of
                 docs/dev/VALIDATION.md section 2.1 (D = 2.9e6 m,
                 kappa = 1e-6 m^2/s, dT = 1000 K):
                     t_nd = t_s * kappa / D^2,  T_nd = T_K / dT - 0.5,
                     x_nd = x / D,              z_nd = z / D.
                 (the box is 2.5 D long, so x_nd runs 0 .. 2.5, as on our grid)

Layout (stated so that the video is reproducible)
------------------------------------------------
One 720 px band per row: N rows -> 1920 x (N*720) px, so the three-arm video is
1920 x 2160 (chosen over 1920 x 1080, which would give each of three rows only
360 px of height) and a single arm is 1920 x 720.  Inside a band the temperature
field is 1650 x 660 px (exactly 2.5:1, the box aspect), a label line names the
solver on the left and prints the time on the right, and a single shared colour
bar runs down the right edge over exactly the used height.  The top row's label
line also carries the case name, the window and the frame rate.  A 1920 x 1080
`_1080p.mp4` can be produced as well (`--also-1080`).

Colour scale rule
-----------------
ONE scale for the whole video and for all of its rows:
    vmin = min(0, floor2(min T over the sampled frames of every row))
    vmax = max(1, ceil2(max T over the sampled frames of every row))
with floor2/ceil2 rounding outward to two significant digits.  The extremes are
measured on at most `--scan` frames per row, evenly spaced over the window.  The
scale therefore cannot drift between frames and never clips a sampled value;
`--vmin/--vmax` override it, and the scale actually used is written into the
`.json` sidecar and drawn as the colour bar's two end ticks.  Colormap
`inferno`, as in live_render.py.

Frame alignment rule
--------------------
The three sequences are forced to the same frame count N, chosen as the number
of frames the LEAST-sampled row has inside the window (so the video never shows
a row more finely than that row can support); `--frames` overrides N.  The N
target times are then the frame times of that least-sampled row, decimated in
index to N, so every one of its frames appears exactly once and the denser rows
are decimated onto it.  Each row shows the frame whose time is NEAREST to the
target -- no interpolation of the fields in time, so nothing in the video is
synthetic -- and a row with fewer frames than N (only possible under an explicit
`--frames`) repeats its nearest frame.  The compromise is visible, not hidden:
every row prints the time of the frame it actually shows and its offset from the
target, and the `.json` sidecar lists every (target, frame, offset) triple.
`--time-grid linear|log` replaces the reference cadence with an even grid over
the window, if a constant frame interval is wanted instead.
Outputs
-------
<out>.mp4             H.264, yuv420p, -crf 18, -preset slow
<out>_1080p.mp4       optional 1920 x 1080 stack
<out>_poster.png      the final frame stack as a PNG
<out>.json            the plan: rows, window, target times, per-row matches,
                      shared scale, encoder command
<out>_frames/*.png    the individual stacks (kept for re-encoding)
"""
import argparse
import bisect
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
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.collections import LineCollection    # noqa: E402
from matplotlib.colors import FuncNorm, Normalize, to_rgba       # noqa: E402
from matplotlib.cm import ScalarMappable             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import live_render as lr                             # noqa: E402  (frame readers)

# --- the ASPECT scale set of docs/dev/VALIDATION.md section 2.1 ---
D = 2.9e6                 # box depth [m]; the box is 2.5 D long
KAPPA = 1.0e-6            # thermal diffusivity [m^2/s]
DT = 1000.0               # temperature drop [K]
T0 = D * D / KAPPA        # time scale [s]  (8.41e18)

# --- pixel layout: one 1920 x 720 band per row, so N rows -> 1920 x (N*720) --
PX_W = 1920
PX_ROW = 720
PX_FIELD_W = 1650         # exact 2.5:1 with PX_FIELD_H
PX_FIELD_H = 660
PAD_LEFT = 46             # room for the z tick labels
FIELD_TOP = 36            # label line sits in the 36 px above the field
CBAR_X = 1730
CBAR_W = 52
DPI = 96                  # px = inch * 96; 3 rows -> figsize 20 x 22.5 in
MESH_STYLE = dict(colors="cyan", linewidths=0.18, alpha=0.35)
BG = "#0d0d0d"
FG = "#e8e8e8"
DIM = "#a8a8a8"


def canvas_px(n_rows, row_h=None):
    """Master canvas for n_rows bands: 1920 x (n_rows * row_h)."""
    return PX_W, n_rows * (PX_ROW if row_h is None else row_h)


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


def sigfig(x):
    """Short fixed-format string for an axis label."""
    if x == 0:
        return "0"
    if abs(x) >= 1e-2:
        return "%.4g" % x
    return "%.3e" % x


def nearest_index(times, t):
    """Index of the element of the sorted list `times` nearest to t."""
    i = bisect.bisect_left(times, t)
    if i == 0:
        return 0
    if i >= len(times):
        return len(times) - 1
    return i if abs(times[i] - t) < abs(times[i - 1] - t) else i - 1


def bracket_index(times, t):
    """Index pair (lo, hi) with times[lo] <= t <= times[hi]; equal at the ends."""
    i = bisect.bisect_left(times, t)
    if i <= 0:
        return 0, 0
    if i >= len(times):
        return len(times) - 1, len(times) - 1
    if times[i] == t:
        return i, i
    return i - 1, i


def median_step(times):
    if len(times) < 2:
        return 0.0
    d = np.diff(np.asarray(times))
    d = d[d > 0]
    return float(np.median(d)) if d.size else 0.0


def time_text(target, pick, src):
    """The right-hand label of one row: target time, shown frame, offset."""
    return ("t* %s  ·  row %s (%s)  ·  Δ %+.2e"
            % (sigfig(target), sigfig(pick["t"]), src.describe(pick["index"]),
               pick["delta"]))


# ------------------------------------------------------------------ our frames
class OurSource:
    """A QuaST run: frames/frame_NNNNN_*.bin|csv, metadata in frame_meta.csv."""

    kind = "our"

    def __init__(self, path, lx):
        path = os.path.abspath(path)
        fdir = os.path.join(path, "frames")
        if not os.path.isdir(fdir):
            fdir = path
        self.frames_dir = fdir
        self.lx = lx
        self.rows = lr.read_meta(os.path.join(fdir, "frame_meta.csv"))
        self.frames = []
        if self.rows:
            for r in self.rows:
                ng = int(r["ng"])
                binp = os.path.join(fdir, "frame_%05d_T.bin" % int(r["step"]))
                if os.path.isfile(binp) and os.path.getsize(binp) == ng * ng * 8:
                    self.frames.append(dict(t=float(r["t"]), ng=ng,
                                            step=int(r["step"])))
        else:                                   # no metadata: infer from the files
            print("[three_row] NO frame_meta.csv in %s -- inferring t=step"
                  % fdir, flush=True)
            for binp in sorted(glob.glob(os.path.join(fdir, "frame_*_T.bin"))):
                m = re.search(r"frame_(\d+)_T\.bin$", binp)
                if not m:
                    continue
                size = os.path.getsize(binp)
                ng = int(round(math.sqrt(size / 8.0)))
                if ng * ng * 8 != size:
                    continue
                step = int(m.group(1))
                self.frames.append(dict(t=float(step), ng=ng, step=step))
        self.frames.sort(key=lambda f: f["t"])
        self.times = [f["t"] for f in self.frames]
        self.meta = {int(r["step"]): r for r in self.rows}
        self._T = {}
        self._cells = {}

    def __len__(self):
        return len(self.frames)

    def describe(self, i):
        return "step %d" % self.frames[i]["step"]

    def _field(self, i):
        if i in self._T:
            return self._T[i]
        f = self.frames[i]
        tag = "frame_%05d" % f["step"]
        T = lr.read_frame_bin(os.path.join(self.frames_dir, tag + "_T.bin"), f["ng"])
        if T is None:
            raise SystemExit("frame %d is incomplete: %s" % (f["step"], tag))
        boxes = lr.read_mesh(os.path.join(self.frames_dir, tag + "_mesh.csv"))
        segs = lr.mesh_segments(boxes) if boxes else None
        out = (T, segs)
        if len(self._T) > 6:
            self._T.clear()
        self._T[i] = out
        return out

    def draw(self, i):
        """(T on the ng x ng grid, mesh segments or None)."""
        return self._field(i)

    def raw_range(self, i):
        T, _ = self._field(i)
        return float(np.nanmin(T)), float(np.nanmax(T))

    def cells(self, i):
        """Cell corners (n,4,2) and edge lengths (n,) -- one quad per leaf."""
        if i in self._cells:
            return self._cells[i]
        f = self.frames[i]
        boxes = lr.read_mesh(os.path.join(self.frames_dir,
                                          "frame_%05d_mesh.csv" % f["step"]),
                             with_levels=True)
        if not boxes:
            out = (np.zeros((0, 4, 2)), np.zeros(0))
        else:
            b = np.asarray(boxes, dtype="float64")      # level, x0, y0, h
            x0, y0, h = b[:, 1], b[:, 2], b[:, 3]
            x1, y1 = x0 + h, y0 + h
            quads = np.stack([np.stack([x0, y0], 1), np.stack([x1, y0], 1),
                              np.stack([x1, y1], 1), np.stack([x0, y1], 1)], 1)
            out = (quads, h)
        if len(self._cells) > 6:
            self._cells.clear()
        self._cells[i] = out
        return out

    def status(self, i, h_ref):
        _, h = self.cells(i)
        lv = level_of(h, h_ref) if len(h) else np.zeros(0, dtype=int)
        row = self.meta.get(self.frames[i]["step"], {})
        return dict(kind="leaves", n=int(len(h)),
                    lo=int(lv.min()) if len(lv) else 0,
                    hi=int(lv.max()) if len(lv) else 0,
                    nu=row.get("Nu_top"), vrms=row.get("Vrms"),
                    tmax=row.get("Tmax"), source="frame_meta.csv")


# --------------------------------------------------------------- ASPECT frames
class AspectSource:
    """An ASPECT output directory: solution-NNNNN.PPPP.vtu pieces, merged."""

    kind = "aspect"

    def __init__(self, path, lx, grid=(640, 256), temp="auto", nd="auto",
                 mesh="edges"):
        path = os.path.abspath(path)
        sol = os.path.join(path, "solution")
        self.sol_dir = sol if os.path.isdir(sol) else path
        self.lx = lx
        self.nx, self.nz = int(grid[0]), int(grid[1])
        self.temp_mode = temp
        self.nd_mode = nd
        self.mesh_mode = mesh
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
        self.frames = [None] * len(self.index)      # filled by _read_meta
        for k, idx in enumerate(self.index):
            self.frames[k] = dict(t=float("nan"), idx=idx, npieces=len(self.pieces[idx]))
        self._read_times()
        self.frames.sort(key=lambda f: f["t"])
        self.times = [f["t"] for f in self.frames]
        self._cache = {}
        self._cache_order = []
        self._range = {}
        self._conv = {}
        self._cells = {}
        self._stats = None

    # -- time pass: only the TIME field, cheap (still reads the pieces) ------
    def _read_times(self):
        import meshio
        for f in self.frames:
            mesh = meshio.read(self.pieces[f["idx"]][0])
            if "TIME" not in mesh.field_data:
                raise SystemExit("no TIME field in %s -- cannot order the frames"
                                 % self.pieces[f["idx"]][0])
            t_s = float(np.asarray(mesh.field_data["TIME"]).ravel()[0])
            f["t_s"] = t_s
            f["t"] = t_s / T0

    def __len__(self):
        return len(self.frames)

    def describe(self, i):
        return "frame %05d" % self.frames[i]["idx"]

    # -- field pass: merge all pieces, dedupe, interpolate to a regular grid --
    def _field(self, i):
        if i in self._cache:
            return self._cache[i]
        import meshio
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        f = self.frames[i]
        pts, vals, segs = [], [], []
        for p in self.pieces[f["idx"]]:
            mesh = meshio.read(p)
            P = np.asarray(mesh.points[:, :2], dtype="float64")
            T = np.asarray(mesh.point_data["T"], dtype="float64")
            pts.append(P)
            vals.append(T)
            if self.mesh_mode == "edges":
                for block in mesh.cells:
                    if block.type != "quad":
                        continue
                    q = np.asarray(block.data)
                    for k in range(4):
                        a = P[q[:, k]]
                        b = P[q[:, (k + 1) % 4]]
                        segs.append(np.stack([a, b], axis=1))
        P = np.vstack(pts)
        T = np.concatenate(vals)
        self._decide(P, T)
        if self._conv["nd"]:
            P = P / D                       # x_nd = x/D in [0, 2.5], z_nd in [0, 1]
        if self._conv["kelvin"]:
            T = T / DT - 0.5
        if segs:
            segs = np.concatenate(segs, axis=0)
            if self._conv["nd"]:
                segs = segs / D
        else:
            segs = None

        key = np.round(P, 9)
        uid = np.unique(key, axis=0, return_index=True)[1]
        Pu, Tu = P[uid], T[uid]
        xi = np.linspace(0.0, self.lx, self.nx)
        zi = np.linspace(0.0, 1.0, self.nz)
        X, Z = np.meshgrid(xi, zi)
        Tg = LinearNDInterpolator(Pu, Tu)(X, Z)
        nan = ~np.isfinite(Tg)
        if nan.any():                       # convex-hull rim: nearest fill
            Tg[nan] = NearestNDInterpolator(Pu, Tu)(X[nan], Z[nan])
        out = (Tg, segs, float(np.nanmin(T)), float(np.nanmax(T)))
        if len(self._cache_order) >= 3:
            self._cache.pop(self._cache_order.pop(0), None)
        self._cache[i] = out
        self._cache_order.append(i)
        return out

    def _decide(self, P, T):
        """Decide once whether ASPECT's output is dimensional (loudly)."""
        if "kelvin" in self._conv:
            return
        use_dim = (float(np.nanmax(np.abs(P))) > 5.0
                   if self.nd_mode == "auto" else self.nd_mode == "yes")
        if self.temp_mode == "auto":
            kelvin = float(np.nanmax(T)) > 10.0
        else:
            kelvin = self.temp_mode == "kelvin"
        self._conv["nd"] = use_dim
        self._conv["kelvin"] = kelvin
        print("[three_row] ASPECT units: coordinates %s, temperature %s"
              % ("dimensional -> x/D, z/D" if use_dim else "nondimensional",
                 "kelvin -> T/dT - 0.5" if kelvin else "nondimensional"),
              flush=True)

    def draw(self, i):
        Tg, segs, _, _ = self._field(i)
        return Tg, segs

    def raw_range(self, i):
        """min/max of the raw vertex T (no interpolation), for the shared scale."""
        if i in self._range:
            return self._range[i]
        import meshio
        vals, P = [], []
        for p in self.pieces[self.frames[i]["idx"]]:
            mesh = meshio.read(p)
            vals.append(np.asarray(mesh.point_data["T"], dtype="float64"))
            P.append(np.asarray(mesh.points[:, :2], dtype="float64"))
        T = np.concatenate(vals)
        self._decide(np.vstack(P), T)
        if self._conv["kelvin"]:
            T = T / DT - 0.5
        r = (float(np.nanmin(T)), float(np.nanmax(T)))
        self._range[i] = r
        return r

    def cells(self, i):
        """Cell corners (n,4,2) and edge lengths (n,) on the ASPECT mesh."""
        if i in self._cells:
            return self._cells[i]
        import meshio
        f = self.frames[i]
        quads, edges = [], []
        for p in self.pieces[f["idx"]]:
            mesh = meshio.read(p)
            P = np.asarray(mesh.points[:, :2], dtype="float64")
            if "nd" not in self._conv:
                self._decide(P, np.asarray(mesh.point_data["T"], dtype="float64"))
            if self._conv["nd"]:
                P = P / D
            for block in mesh.cells:
                if block.type != "quad":
                    continue
                Q = P[np.asarray(block.data)]
                quads.append(Q)
                edges.append(np.hypot(Q[:, 1, 0] - Q[:, 0, 0],
                                      Q[:, 1, 1] - Q[:, 0, 1]))
        out = (np.concatenate(quads) if quads else np.zeros((0, 4, 2)),
               np.concatenate(edges) if edges else np.zeros(0))
        if len(self._cells) > 3:
            self._cells.clear()
        self._cells[i] = out
        return out

    def _stat_at(self, t):
        """The statistics row nearest to t_nd, or {} if the file is unreadable."""
        if self._stats is None:
            path = os.path.join(os.path.dirname(self.sol_dir), "statistics")
            if not os.path.isfile(path):
                path = os.path.join(self.sol_dir, "statistics")
            self._stats = read_aspect_statistics(path)
            if self._stats:
                print("[three_row] ASPECT diagnostics: %d rows from %s"
                      % (len(self._stats), path), flush=True)
            else:
                print("[three_row] ASPECT diagnostics: no statistics file, "
                      "the status line will show the cell count only", flush=True)
        if not self._stats:
            return {}
        return self._stats[nearest_index([r["t"] for r in self._stats], t)]

    def status(self, i, h_ref):
        _, h = self.cells(i)
        lv = level_of(h, h_ref) if len(h) else np.zeros(0, dtype=int)
        st = self._stat_at(self.frames[i]["t"])
        return dict(kind="cells", n=int(len(h)),
                    lo=int(lv.min()) if len(lv) else 0,
                    hi=int(lv.max()) if len(lv) else 0,
                    nu=st.get("nu_top"), vrms=st.get("vrms"),
                    tmax=st.get("tmax"),
                    source="statistics" if st else None)


def make_source(kind, path, args):
    if kind == "our":
        return OurSource(path, args.lx)
    if kind == "aspect":
        return AspectSource(path, args.lx, grid=(args.aspect_grid[0],
                                                 args.aspect_grid[1]),
                            temp=args.aspect_temp, nd=args.aspect_nd,
                            mesh=args.aspect_mesh)
    raise SystemExit("unknown row kind %r (use our: or aspect:)" % kind)


def parse_row(spec, args):
    """`label=kind:path` or `kind:path` -> (label, source).

    The label may contain ':' and '=' (the last '=' before `kind:` splits), so
    `--row "run 3 = ASPECT check=aspect:<dir>"` works; a *path* must not
    contain '='.
    """
    m = re.match(r"^(?:(?P<label>.*)=)?(?P<kind>[A-Za-z_]+):(?P<path>.+)$",
                 spec, re.S)
    if not m:
        raise SystemExit("row spec %r is not [LABEL=]KIND:PATH" % spec)
    kind = m.group("kind").strip().lower()
    path = m.group("path").strip()
    label = (m.group("label") or "").strip()
    src = make_source(kind, path, args)
    if not label:
        label = os.path.basename(os.path.normpath(path))
    return label, src


# ---------------------------------------------------------------- frame plan
def build_plan(sources, labels, args):
    """Common target times and, per row, the nearest available frame."""
    starts = [s.times[0] for s in sources if s.times]
    ends = [s.times[-1] for s in sources if s.times]
    if len(starts) != len(sources):
        raise SystemExit("a row has no frames at all")
    t0 = args.t0 if args.t0 is not None else max(starts)
    t1 = args.t1 if args.t1 is not None else min(ends)
    if not (t1 > t0):
        raise SystemExit("empty window [%g, %g]" % (t0, t1))
    inside = []
    for lab, s in zip(labels, sources):
        idx = [i for i, t in enumerate(s.times) if t0 <= t <= t1]
        if not idx:
            raise SystemExit(
                "row %r has no frame in [%g, %g] (it covers %g .. %g)"
                % (lab, t0, t1, s.times[0], s.times[-1]))
        inside.append(idx)
    n_auto = min(len(idx) for idx in inside)
    n = args.frames if args.frames else n_auto
    if n > args.max_frames:
        print("[three_row] %d frames > --max-frames %d, clipping"
              % (n, args.max_frames), flush=True)
        n = args.max_frames
    n = max(1, n)
    if args.time_grid == "frames":
        # the cadence comes from the row with the FEWEST frames in the window
        # (the one that limits the common frame count), decimated in index to n
        j_ref = min(range(len(sources)), key=lambda j: len(inside[j]))
        ref = inside[j_ref]
        take = np.round(np.linspace(0.0, len(ref) - 1.0, n)).astype(int)
        targets = [sources[j_ref].times[ref[k]] for k in take]
        if n > len(ref):
            print("[three_row] %d targets over %d frames of %r: repeats by "
                  "design" % (n, len(ref), labels[j_ref]), flush=True)
    elif args.time_grid == "log":
        targets = list(np.exp(np.linspace(math.log(t0), math.log(t1), n)))
    else:
        targets = list(np.linspace(t0, t1, n))

    plan = []
    for k, t in enumerate(targets):
        picks = []
        for j, (lab, s) in enumerate(zip(labels, sources)):
            i = nearest_index(s.times, t)
            # never leave the window: clamp to the frames that are actually in it
            if s.times[i] < t0:
                i = inside[j][0]
            elif s.times[i] > t1:
                i = inside[j][-1]
            lo, hi = bracket_index(s.times, t)
            lo = min(max(lo, inside[j][0]), inside[j][-1])
            hi = min(max(hi, inside[j][0]), inside[j][-1])
            if hi < lo:
                lo, hi = hi, lo
            w = 0.0
            if hi > lo and s.times[hi] > s.times[lo]:
                w = (t - s.times[lo]) / (s.times[hi] - s.times[lo])
                w = min(1.0, max(0.0, w))
            picks.append(dict(row=j, label=lab, index=i, t=s.times[i],
                              delta=s.times[i] - t, lo=lo, hi=hi, w=w))
        plan.append(dict(k=k, target=t, picks=picks))

    print("[three_row] window t_nd = [%s, %s], %d target times (%s), "
          "auto count %d" % (sigfig(t0), sigfig(t1), n, args.time_grid, n_auto),
          flush=True)
    span = t1 - t0
    for j, (lab, s, idx) in enumerate(zip(labels, sources, inside)):
        step = median_step([s.times[i] for i in idx])
        deltas = [abs(fr["picks"][j]["delta"]) for fr in plan]
        used = len({fr["picks"][j]["index"] for fr in plan})
        reps = n - used
        print("  %-26s %6d frames in window, median dt %9.3e, %d distinct "
              "frames shown, max offset %9.3e (%.1f%% of the window)"
              % (lab, len(idx), step, used, max(deltas),
                 100.0 * max(deltas) / span if span > 0 else 0.0), flush=True)
        if reps > 0:
            print("     NOTE: %d of %d target times repeat this row's nearest "
                  "frame (it has only %d frames in the window)"
                  % (reps, n, len(idx)), flush=True)
        elif span > 0 and max(deltas) > 0.02 * span:
            print("     NOTE: max offset is > 2%% of the window", flush=True)
    return t0, t1, plan


# ------------------------------------------------------------------- drawing
class RowView:
    def __init__(self, fig, band, px_h, label, shared_norm, cmap):
        self.label = label
        top_px = band * PX_ROW
        bottom_px = px_h - (top_px + FIELD_TOP + PX_FIELD_H)
        ax = fig.add_axes([PAD_LEFT / PX_W, bottom_px / px_h,
                           PX_FIELD_W / PX_W, PX_FIELD_H / px_h])
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color("#555555")
        ax.tick_params(colors=DIM, labelsize=11, length=3)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        self.ax = ax
        self.im = None
        self.lc = None
        # label line: number + solver on the left, time on the right
        self.label_text = fig.text(
            PAD_LEFT / PX_W, 1.0 - (top_px + 17) / px_h,
            ("%d  %s" % (band + 1, label))[:38], color=FG, fontsize=17,
            fontweight="bold", va="center", ha="left")
        self.time_text = fig.text(
            (PAD_LEFT + PX_FIELD_W) / PX_W, 1.0 - (top_px + 18) / px_h,
            "", color=DIM, fontsize=12, va="center", ha="right")
        self.norm = shared_norm
        self.cmap = cmap

    def update(self, T, segs, lx, time_text, mesh=True):
        if self.im is None:
            # first frame: build the image with live_render's own drawing code
            self.im = lr.draw_temperature(self.ax, T, None, lx, 1.0)
            self.ax.set_xlabel("")
            self.ax.set_ylabel("")
            self.ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
            self.ax.set_yticks([0.0, 0.5, 1.0])
            self.ax.set_xlim(0.0, lx)
            self.ax.set_ylim(0.0, 1.0)
        else:
            self.im.set_data(T)
        self.im.set_cmap(self.cmap)
        self.im.set_clim(self.norm.vmin, self.norm.vmax)
        if segs is not None and mesh:
            if self.lc is None:
                self.lc = LineCollection(segs, **MESH_STYLE)
                self.lc.set_zorder(2)
                self.ax.add_collection(self.lc)
            else:
                self.lc.set_segments(segs)
        elif self.lc is not None:
            self.lc.set_segments([])
        self.time_text.set_text(time_text)


def build_figure(labels, norm, cmap, header, worst_time):
    px_w, px_h = canvas_px(len(labels))
    fig = plt.figure(figsize=(px_w / DPI, px_h / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG)
    views = []
    for band, lab in enumerate(labels):
        views.append(RowView(fig, band, px_h, lab, norm, cmap))
    # one shared colour bar down the right edge, over the used height; its
    # end ticks are the exact vmin/vmax, so the frame carries the scale values
    top = FIELD_TOP
    bottom = px_h - ((len(labels) - 1) * PX_ROW + FIELD_TOP + PX_FIELD_H)
    cax = fig.add_axes([CBAR_X / PX_W, bottom / px_h, CBAR_W / PX_W,
                        (px_h - top - bottom) / px_h])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(colors=DIM, labelsize=13)
    cb.outline.set_edgecolor("#555555")
    ticks = [norm.vmin] + [t for t in (0.5, 1.0, 1.5)
                           if norm.vmin < t < norm.vmax] + [norm.vmax]
    cb.set_ticks(ticks)
    cb.set_ticklabels(["%.3g" % t for t in ticks])
    cax.set_title("T shared", color=FG, fontsize=12, pad=6)

    # the case / window / rate line rides on the top row's label line, between
    # the row label (left) and the time (right); measure instead of guessing, so
    # an unlucky label or time string cannot produce an overlap
    for v, wt in zip(views, worst_time):
        v.time_text.set_text(wt)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    hx = max(426.0, PAD_LEFT + views[0].label_text.get_window_extent(rend).width
             + 16.0)
    right = PAD_LEFT + PX_FIELD_W
    avail = right - views[0].time_text.get_window_extent(rend).width - 16.0 - hx
    text = header[:120]
    htxt = fig.text(hx / PX_W, 1.0 - 18.0 / px_h, text, color="#9a9a9a",
                    fontsize=11.5, va="center", ha="left")
    while len(text) > 16 and htxt.get_window_extent(rend).width > avail:
        text = text[:-2]
        htxt.set_text(text + " …")
    for v in views:
        v.time_text.set_text("")
    return fig, views


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
    raise SystemExit("no ffmpeg found; put it on PATH, set QUAST_FFMPEG, or pass --ffmpeg PATH")


def sibling_1080_vf(px_h, pad="0x0d0d0d"):
    """The -vf of the optional 1920 x 1080 sibling.

    Three rows (a 2160 px master) keep the exact 2:1 downscale this flag has
    always produced, byte for byte.  Any other height is fitted into 1920 x 1080
    and padded with the frame background, so the 2.5:1 field is never stretched:
    a one-row 1920 x 720 master becomes 1920 x 1080 with 180 px bands above and
    below, not a 1.5x vertical stretch.
    """
    if px_h == 3 * PX_ROW:
        return "scale=1920:1080:flags=lanczos,setsar=1"
    return ("scale=1920:1080:force_original_aspect_ratio=decrease:flags=lanczos,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=%s,setsar=1" % pad)


def encode(ffmpeg, frames_dir, pattern, fps, out, crf=18, preset="slow",
           vf=None):
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
           "-i", os.path.join(frames_dir, pattern)]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
            "-preset", preset, "-r", str(fps), "-movflags", "+faststart", out]
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("ffmpeg failed:\n%s\n%s" % (" ".join(cmd), r.stderr))
    return cmd, time.time() - t


# ------------------------------------------------------- two-column mode
# The 2-column layout adds a panel per row that shows the mesh only -- every
# cell outlined and coloured by the refinement level implied by its edge length
# -- so the grid structure can be read without the colour field.  It is drawn on
# white, which is why it carries its own theme instead of the dark BG/FG/DIM
# constants of the single-column layout (those are left untouched, so the
# one-column videos stay byte-identical).
PX_COL = 825                 # one column; two of them are PX_FIELD_W wide
PX_ROW_2COL = 560            # band height in the 2-column mode (content + margin)
LABEL_TOP = 44               # band top -> panel top, px
STATUS_BLOCK = 96            # px reserved under the panels for the status lines
LEVEL_COLOURS = ["#1f4e79", "#2e75b6", "#548235", "#bf9000",
                 "#c55a11", "#c00000", "#7030a0", "#1a1a1a"]
MAX_LEVEL = len(LEVEL_COLOURS) - 1
WHITE = dict(bg="#ffffff", fg="#111111", dim="#3f3f3f", spine="#8c8c8c",
             header="#5a5a5a", overlay="#6a6a6a", cbar_edge="#8c8c8c")

# --- ASPECT diagnostics scale set, docs/dev/VALIDATION.md 2.1 ------
K_COND = 3.3                     # kappa cp rho0 [W/(m K)]
NORM_NU = 2.5 * K_COND * DT      # W per unit of Nu (= 8250)


def read_aspect_statistics(path):
    """Rows of an ASPECT `statistics` file in our units, [] if unreadable.

    The columns are found by their header names, exactly as
    aspect_runs/realistic/compare_E.py does; the conversions are the scale set
    of docs/dev/VALIDATION.md section 2.1.
    """
    names, rows = {}, []
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                s = line.strip()
                m = re.match(r"#\s*(\d+):\s*(.*)", s)
                if m:
                    names[int(m.group(1))] = m.group(2).strip()
                elif s:
                    rows.append(s.split())
    except OSError:
        return []

    def col(tag):
        for i, name in names.items():
            if name.startswith(tag):
                return i - 1
        raise KeyError(tag)

    try:
        c_t = col("Time (seconds)")
        c_cells = col("Number of mesh cells")
        c_vrms = col("RMS velocity")
        c_qtop = col('Outward heat flux through boundary with indicator 3 ("top")')
        c_tmax = col("Maximal temperature")
    except KeyError:
        return []
    out = []
    for v in rows:
        try:
            out.append(dict(t=float(v[c_t]) / T0,
                            cells=float(v[c_cells]),
                            vrms=float(v[c_vrms]) * D / KAPPA,
                            nu_top=float(v[c_qtop]) / NORM_NU,
                            tmax=float(v[c_tmax]) / DT - 0.5))
        except (ValueError, IndexError):
            continue
    return out


def level_of(h, h_ref):
    """Refinement level implied by a cell edge: L0 = h_ref, each level halves h."""
    h = np.maximum(np.asarray(h, dtype="float64"), 1e-12)
    lv = np.round(np.log2(float(h_ref) / h)).astype(int)
    return np.clip(lv, 0, MAX_LEVEL)


def quad_segments(quads):
    """(n,4,2) cell corners -> (4n,2,2) edges, four consecutive per cell."""
    a = np.asarray(quads, dtype="float64")
    b = np.roll(a, -1, axis=1)
    return np.stack([a, b], axis=2).reshape(-1, 2, 2)


def status_text(st):
    """Two lines: counts and level range, then whatever diagnostics we have."""
    line1 = ["%s %s" % (st["kind"], "{:,}".format(int(st["n"]))),
             ("levels L%d-L%d" % (st["lo"], st["hi"])) if st["hi"] > st["lo"]
             else "level L%d" % st["lo"]]
    if st.get("mesh_note"):
        line1.append(st["mesh_note"])
    line2 = []
    if st.get("nu") is not None:
        line2.append("Nu_top %.2f" % st["nu"])
    if st.get("vrms") is not None:
        line2.append("Vrms %.3g" % st["vrms"])
    if st.get("tmax") is not None:
        line2.append("Tmax %.3f" % st["tmax"])
    if st.get("source"):
        line2.append("from %s" % st["source"])
    return "   |   ".join(line1) + "\n" + "   |   ".join(line2)


class TwoColRow:
    """One row: T + mesh on the left, mesh coloured by level on the right."""

    def __init__(self, fig, band, px_h, row_h, label, norm, cmap, panel_h, lx,
                 off):
        self.norm = norm
        self.cmap = cmap
        self.lx = lx
        self.im = None
        self.lc_l = None
        self.lc_r = None
        top = band * row_h + off
        bottom_px = px_h - (top + LABEL_TOP + panel_h)
        rect_l = [PAD_LEFT / PX_W, bottom_px / px_h, PX_COL / PX_W,
                  panel_h / px_h]
        rect_r = [(PAD_LEFT + PX_COL) / PX_W, bottom_px / px_h, PX_COL / PX_W,
                  panel_h / px_h]
        self.ax_l = fig.add_axes(rect_l)
        self.ax_r = fig.add_axes(rect_r)
        for ax in (self.ax_l, self.ax_r):
            ax.set_facecolor(WHITE["bg"])
            for sp in ax.spines.values():
                sp.set_color(WHITE["spine"])
            ax.tick_params(colors=WHITE["dim"], labelsize=10, length=3)
            ax.set_xlim(0.0, lx)
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
            ax.set_yticks([0.0, 0.5, 1.0])
        self.ax_r.tick_params(labelleft=False)      # one y axis is enough
        if band != 0:
            self.ax_l.tick_params(labelbottom=False)
            self.ax_r.tick_params(labelbottom=False)
        self.label_text = fig.text(
            PAD_LEFT / PX_W, 1.0 - (top + 24) / px_h,
            ("%d  %s" % (band + 1, label))[:38], color=WHITE["fg"],
            fontsize=17, fontweight="bold", va="center", ha="left")
        self.time_text = fig.text(
            (PAD_LEFT + PX_FIELD_W) / PX_W, 1.0 - (top + 24) / px_h, "",
            color=WHITE["dim"], fontsize=12, va="center", ha="right")
        self.status_text = fig.text(
            PAD_LEFT / PX_W, 1.0 - (top + LABEL_TOP + panel_h + 40) / px_h, "",
            color=WHITE["fg"], fontsize=12.5, va="top", ha="left",
            linespacing=1.45)

    def update(self, T, quads, h, h_ref, time_str, status_str):
        if self.im is None:
            self.im = lr.draw_temperature(self.ax_l, T, None, self.lx, 1.0)
            self.im.set_cmap(self.cmap)
            self.ax_l.set_xlabel("")
            self.ax_l.set_ylabel("")
        else:
            self.im.set_data(T)
        self.im.set_clim(self.norm.vmin, self.norm.vmax)
        segs = quad_segments(quads)
        if self.lc_l is None:
            self.lc_l = LineCollection(segs, colors=WHITE["overlay"],
                                       linewidths=0.12, alpha=0.35, zorder=2)
            self.ax_l.add_collection(self.lc_l)
        else:
            self.lc_l.set_segments(segs)
        lv = level_of(h, h_ref) if len(h) else np.zeros(0, dtype=int)
        cols = np.repeat(np.array([to_rgba(LEVEL_COLOURS[k]) for k in lv],
                                  dtype="float64").reshape(-1, 4), 4, axis=0)
        if self.lc_r is None:
            self.lc_r = LineCollection(segs, linewidths=0.45, zorder=2)
            self.lc_r.set_color(cols)
            self.ax_r.add_collection(self.lc_r)
        else:
            self.lc_r.set_segments(segs)
            self.lc_r.set_color(cols)
        self.time_text.set_text(time_str)
        self.status_text.set_text(status_str)


def build_two_col_figure(labels, norm, cmap, header, px_h, panel_h,
                         level_max, h_ref, lx):
    fig = plt.figure(figsize=(PX_W / DPI, px_h / DPI), dpi=DPI)
    fig.patch.set_facecolor(WHITE["bg"])
    # a 2.5:1 panel is 330 px tall, so the band is reduced to the content block
    # plus a small margin and the block is centred in what is left
    row_h = px_h // len(labels)
    content_h = LABEL_TOP + panel_h + STATUS_BLOCK
    off = max(0, (row_h - content_h) // 2)
    views = [TwoColRow(fig, b, px_h, row_h, lab, norm, cmap, panel_h, lx, off)
             for b, lab in enumerate(labels)]
    # shared temperature bar, only over the height the panels actually use
    top = off + LABEL_TOP
    bottom = px_h - ((len(labels) - 1) * row_h + off + LABEL_TOP + panel_h)
    cax = fig.add_axes([CBAR_X / PX_W, bottom / px_h, CBAR_W / PX_W,
                        (px_h - top - bottom) / px_h])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.ax.tick_params(colors=WHITE["dim"], labelsize=13)
    cb.outline.set_edgecolor(WHITE["cbar_edge"])
    ticks = [norm.vmin] + [t for t in (0.5, 1.0, 1.5)
                           if norm.vmin < t < norm.vmax] + [norm.vmax]
    cb.set_ticks(ticks)
    cb.set_ticklabels(["%.3g" % t for t in ticks])
    cax.set_title("T shared", color=WHITE["fg"], fontsize=12, pad=6)

    # header on its own line in the top margin, so a long case name or a long
    # blend label can never run into each other
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    hx = PAD_LEFT
    avail = PX_W - 2 * PAD_LEFT - 20.0
    text = header[:150]
    htxt = fig.text(hx / PX_W, 1.0 - 18.0 / px_h, text,
                    color=WHITE["header"], fontsize=11.5, va="center",
                    ha="left")
    while len(text) > 16 and htxt.get_window_extent(rend).width > avail:
        text = text[:-2]
        htxt.set_text(text + " …")
    for v in views:
        v.time_text.set_text("")

    # discrete level legend, beside the status line of the top row, under the
    # mesh column it explains (y is measured from the canvas top, so convert)
    item = 94.0
    total = (level_max + 1) * item
    x0 = PAD_LEFT + PX_COL + PX_COL / 2.0 - total / 2.0
    y = px_h - (off + LABEL_TOP + panel_h + 46.0)
    for k in range(level_max + 1):
        x = x0 + k * item
        fig.add_artist(plt.Rectangle((x / PX_W, (y - 6.5) / px_h),
                                     34.0 / PX_W, 13.0 / px_h,
                                     transform=fig.transFigure,
                                     facecolor=LEVEL_COLOURS[k],
                                     edgecolor=WHITE["spine"], linewidth=0.6))
        fig.text((x + 40.0) / PX_W, y / px_h, "L%d" % k, color=WHITE["fg"],
                 fontsize=11, va="center", ha="left")
    fig.text((PAD_LEFT + PX_COL + PX_COL / 2.0) / PX_W, (y - 28.0) / px_h,
             "mesh only, coloured by refinement level: L0 = %.0f km, each "
             "level halves the cell edge" % (h_ref * D / 1000.0),
             color=WHITE["dim"], fontsize=10.5, va="center", ha="center")
    return fig, views


def sample_temperature(src, pick, interp):
    """(T, mesh frame index, note) for one row at one target time.

    `nearest` shows the nearest frame; `linear` blends the two bracketing frames
    element-wise, which is only possible for the temperature field -- the mesh
    column always shows the nearest frame and says so.
    """
    lo, hi, w = pick["lo"], pick["hi"], pick["w"]
    if interp != "linear" or hi == lo:
        i = pick["index"] if interp != "linear" else (lo if w < 0.5 else hi)
        return src.draw(i)[0], i, None
    T0 = src.draw(lo)[0]
    T1 = src.draw(hi)[0]
    mesh_i = lo if w < 0.5 else hi
    note = "mesh: nearest frame, %s" % src.describe(mesh_i)
    if T0.shape != T1.shape:                    # cannot blend different grids
        return T0, mesh_i, note
    return (1.0 - w) * T0 + w * T1, mesh_i, note


def time_label(target, pick, src, interp):
    """The right-hand label: the shown frame, or the two frames of a blend."""
    if interp != "linear" or pick["hi"] == pick["lo"]:
        return time_text(target, pick, src)
    lo, hi, w = pick["lo"], pick["hi"], pick["w"]
    return ("t* %s  ·  blend %s (%s) to %s (%s), w %.2f"
            % (sigfig(target), sigfig(src.times[lo]), src.describe(lo),
               sigfig(src.times[hi]), src.describe(hi), w))


# ------------------------------------------------- clean three-column layout
# What the README shows.  Two rows of three bare panels: the temperature field,
# the temperature anomaly about the horizontal mean, and the mesh coloured by
# refinement level.  The panels carry no axes, ticks or labels; the only
# furniture is one bar under each of the first two columns and the level legend
# under the third, all on one line.  Every number and symbol goes through
# mathtext with the Computer Modern faces.
CLEAN = dict(pad_x=44, pad_top=16, gap=26, band_gap=14,
             cbar_h=16, cbar_label=26, bottom=10, title=20)
PANEL_LABEL_POS = (0.014, 0.962)     # solver name, axes coordinates
PANEL_TEXT_POS = (0.985, 0.032)      # diagnostics, axes coordinates
THETA_CMAP = "RdBu_r"
ETA_CMAP = "magma_r"
ETA_GAMMA = 0.30                     # expands the soft end of the eta scale
ETA_RANGE = (-3.0, 4.0)              # log10(eta/eta0), fixed for the clip


def use_science_style():
    """SciencePlots' look without a TeX installation.

    The `science` style asks for `text.usetex`, and there is no LaTeX on this
    machine, so it is applied and then TeX is switched off: mathtext with the
    Computer Modern font set gives the same faces for the symbols that carry
    subscripts or exponents.
    """
    try:
        import scienceplots  # noqa: F401  (registers the style)
        plt.style.use(["science"])
    except Exception:
        pass
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "standard",     # the style asks for "tight"
        "savefig.pad_inches": 0.0,
    })


def clean_geometry(px_w, n_rows, lx, n_cols=3):
    """Pixel rectangles for the clean layout at canvas width `px_w`."""
    g = CLEAN
    col_w = (px_w - 2 * g["pad_x"] - (n_cols - 1) * g["gap"]) // n_cols
    panel_h = int(round(col_w / float(lx)))
    band_h = panel_h + g["band_gap"]
    strip_h = g["cbar_h"] + g["cbar_label"] + g["title"] + g["bottom"]
    px_h = g["pad_top"] + n_rows * band_h + strip_h
    px_h += px_h % 2                        # h264 wants an even height
    geo = dict(g)
    geo.update(px_w=px_w, px_h=px_h, col_w=col_w, panel_h=panel_h,
               band_h=band_h, strip_h=strip_h, n_cols=n_cols, lx=lx,
               x0=[g["pad_x"] + k * (col_w + g["gap"]) for k in range(n_cols)])
    return geo


def sci(x, sig=3):
    """A number as mathtext: a real exponent once it needs one, else 3 digits."""
    if x is None:
        return None
    if x == 0.0:
        return r"$0$"
    if abs(x) >= 1.0e3 or abs(x) < 1.0e-2:
        mant, exp = ("%.*e" % (sig - 1, x)).split("e")
        if "." in mant:
            mant = mant.rstrip("0").rstrip(".")
        return r"$%s\times10^{%d}$" % (mant, int(exp))
    return r"$%.*g$" % (sig, x)


def eta_norm():
    """A power-law norm over ETA_RANGE: full labels, most colour on the interior.

    The viscosity spans nine decades but 95 % of the domain sits between -3 and
    1; a linear scale spends almost all its colour on the stiff lid.  Forward and
    inverse are both given so the colourbar can place its ticks.
    """
    lo, hi = ETA_RANGE
    g = ETA_GAMMA
    forward = lambda x: np.clip((np.asarray(x) - lo) / (hi - lo), 0.0, 1.0) ** g
    inverse = lambda y: lo + (hi - lo) * np.clip(y, 0.0, 1.0) ** (1.0 / g)
    return FuncNorm((forward, inverse), vmin=lo, vmax=hi)


def temperature_anomaly(T):
    """T minus its horizontal mean, averaged over x (rows of T are y)."""
    return T - T.mean(axis=1, keepdims=True)


def status_line2(st):
    """Field diagnostics in mathtext, for the lower right of the T panel."""
    parts = []
    if st.get("nu") is not None:
        parts.append(r"$\mathrm{Nu}_{\mathrm{top}} = %.2f$" % st["nu"])
    if st.get("vrms") is not None:
        parts.append(r"$V_{\mathrm{rms}} = %s$" % sci(st["vrms"]).strip("$"))
    if st.get("tmax") is not None:
        parts.append(r"$T_{\max} = %.3f$" % st["tmax"])
    return "    ".join(parts)


def status_line1(st):
    """Mesh counts in mathtext, for the lower right of the mesh panel."""
    n = "{:,}".format(int(st["n"])).replace(",", r"\,")
    kind = r"\mathrm{%s}" % st["kind"]
    if st["hi"] > st["lo"]:
        lv = r"$\mathrm{levels}\ %d$–$%d$" % (st["lo"], st["hi"])
    else:
        lv = r"$\mathrm{level}\ %d$" % st["lo"]
    out = r"$%s = %s$    " % (kind, n) + lv
    if st.get("mesh_note"):
        out += "    " + st["mesh_note"]
    return out


def status_line3(dT):
    """Extremes of the anomaly shown, for the lower right of its panel."""
    return (r"$\theta_{\min} = %.2f$    $\theta_{\max} = %+.2f$"
            % (float(np.nanmin(dT)), float(np.nanmax(dT))))


def status_line4(dE):
    """Viscosity extremes in the panel's own units, for its lower right."""
    return (r"$\log_{10}\eta_{\min} = %.2f$    $\log_{10}\eta_{\max} = %.2f$"
            % (float(np.nanmin(dE)), float(np.nanmax(dE))))


class CleanRow:
    """One bare row: T, its anomaly, and the level-coloured mesh."""

    def __init__(self, fig, geo, band, label, norm, cmap, norm_theta,
                 norm_eta=None):
        g = geo
        self.norm, self.norm_theta, self.cmap = norm, norm_theta, cmap
        self.norm_eta = norm_eta
        self.im = self.im_t = self.im_e = self.lc_l = self.lc_r = None
        y_top = g["px_h"] - (g["pad_top"] + band * g["band_h"])
        y_bottom = y_top - g["panel_h"]
        h = g["panel_h"] / g["px_h"]
        axes = []
        for k in range(g["n_cols"]):
            ax = fig.add_axes([g["x0"][k] / g["px_w"], y_bottom / g["px_h"],
                               g["col_w"] / g["px_w"], h])
            ax.set_axis_off()             # no frame, no ticks, no labels
            ax.set_xlim(0.0, g["lx"])
            ax.set_ylim(0.0, 1.0)
            axes.append(ax)
        self.ax_t, self.ax_m = axes[0], axes[1]
        self.ax_e = axes[2] if len(axes) == 4 else None
        self.ax_mesh = axes[-1]
        self.label = fig.text(
            PANEL_LABEL_POS[0], PANEL_LABEL_POS[1], label,
            transform=self.ax_t.transAxes, color="white", fontsize=19,
            fontweight="bold", va="top", ha="left")
        self.t_left = fig.text(
            PANEL_TEXT_POS[0], PANEL_TEXT_POS[1], "",
            transform=self.ax_t.transAxes, color="#111111", fontsize=12.5,
            fontweight="bold", va="bottom", ha="right",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none",
                      boxstyle="round,pad=0.36"))
        self.t_mid = fig.text(
            PANEL_TEXT_POS[0], PANEL_TEXT_POS[1], "",
            transform=self.ax_m.transAxes, color="#111111", fontsize=12.5,
            fontweight="bold", va="bottom", ha="right",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none",
                      boxstyle="round,pad=0.36"))
        self.t_eta = None
        if self.ax_e is not None:
            self.t_eta = fig.text(
                PANEL_TEXT_POS[0], PANEL_TEXT_POS[1], "",
                transform=self.ax_e.transAxes, color="#111111", fontsize=12.5,
                fontweight="bold", va="bottom", ha="right",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none",
                          boxstyle="round,pad=0.36"))
        self.t_right = fig.text(
            PANEL_TEXT_POS[0], PANEL_TEXT_POS[1], "",
            transform=self.ax_mesh.transAxes, color="#111111", fontsize=12.5,
            fontweight="bold", va="bottom", ha="right",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none",
                      boxstyle="round,pad=0.36"))

    def update(self, T, dT, quads, h, h_ref, text_left, text_mid, text_right,
               dE=None, text_eta=""):
        if self.im is None:
            self.im = lr.draw_temperature(self.ax_t, T, None, self.ax_t.get_xlim()[1], 1.0)
            self.im.set_cmap(self.cmap)
            self.im_t = lr.draw_temperature(self.ax_m, dT, None,
                                            self.ax_m.get_xlim()[1], 1.0)
            self.im_t.set_cmap(THETA_CMAP)
            if self.ax_e is not None and dE is not None:
                self.im_e = lr.draw_temperature(self.ax_e, dE, None,
                                                self.ax_e.get_xlim()[1], 1.0)
                self.im_e.set_cmap(ETA_CMAP)
        else:
            self.im.set_data(T)
            self.im_t.set_data(dT)
            if self.im_e is not None and dE is not None:
                self.im_e.set_data(dE)
        self.im.set_clim(self.norm.vmin, self.norm.vmax)
        self.im_t.set_clim(self.norm_theta.vmin, self.norm_theta.vmax)
        if self.im_e is not None:
            self.im_e.set_norm(self.norm_eta)
        segs = quad_segments(quads)
        if self.lc_l is None:
            self.lc_l = LineCollection(segs, colors=WHITE["overlay"],
                                       linewidths=0.12, alpha=0.35, zorder=2)
            self.ax_t.add_collection(self.lc_l)
        else:
            self.lc_l.set_segments(segs)
        lv = level_of(h, h_ref) if len(h) else np.zeros(0, dtype=int)
        cols = np.repeat(np.array([to_rgba(LEVEL_COLOURS[k]) for k in lv],
                                  dtype="float64").reshape(-1, 4), 4, axis=0)
        if self.lc_r is None:
            self.lc_r = LineCollection(segs, linewidths=0.45, zorder=2)
            self.lc_r.set_color(cols)
            self.ax_mesh.add_collection(self.lc_r)
        else:
            self.lc_r.set_segments(segs)
            self.lc_r.set_color(cols)
        self.t_left.set_text(text_left)
        self.t_mid.set_text(text_mid)
        self.t_right.set_text(text_right)
        if self.t_eta is not None:
            self.t_eta.set_text(text_eta)


def build_clean_figure(labels, norm, cmap, geo, level_max, norm_theta,
                       norm_eta=None):
    use_science_style()
    fig = plt.figure(figsize=(geo["px_w"] / DPI, geo["px_h"] / DPI), dpi=DPI)
    fig.patch.set_facecolor("white")
    rows = [CleanRow(fig, geo, b, lab, norm, cmap, norm_theta, norm_eta)
            for b, lab in enumerate(labels)]
    g = geo
    strip_top = g["px_h"] - (g["pad_top"] + len(labels) * g["band_h"])
    bar_y = (strip_top - g["cbar_h"]) / g["px_h"]
    bar_h = g["cbar_h"] / g["px_h"]
    title_y = (strip_top - g["cbar_h"] - g["cbar_label"] - 6.0) / g["px_h"]

    def bar(k, mappable, norm_local, ticks, title):
        cax = fig.add_axes([g["x0"][k] / g["px_w"], bar_y,
                            g["col_w"] / g["px_w"], bar_h])
        cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        cb.ax.tick_params(colors="#333333", labelsize=11, length=2.5, pad=2)
        cb.outline.set_edgecolor(WHITE["cbar_edge"])
        cb.outline.set_linewidth(0.8)
        cb.set_ticks(ticks)
        cb.set_ticklabels([r"$%.3g$" % t for t in ticks])
        fig.text((g["x0"][k] + g["col_w"] / 2.0) / g["px_w"], title_y, title,
                 color="#222222", fontsize=12, va="center", ha="center")

    ticks_t = [norm.vmin] + [t for t in (0.5, 1.0, 1.5)
                             if norm.vmin < t < norm.vmax] + [norm.vmax]
    bar(0, ScalarMappable(norm=norm, cmap=cmap), norm, ticks_t, r"$T$")

    lim = max(abs(norm_theta.vmin), abs(norm_theta.vmax))
    tt = [-0.8 * lim, -0.4 * lim, 0.0, 0.4 * lim, 0.8 * lim]
    bar(1, ScalarMappable(norm=norm_theta, cmap=THETA_CMAP), norm_theta, tt,
        r"$\theta = T - \langle T\rangle_x$")

    if g["n_cols"] == 4:
        lo, hi = ETA_RANGE
        ticks_e = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
        bar(2, ScalarMappable(norm=norm_eta, cmap=ETA_CMAP),
            None, ticks_e, r"$\log_{10}\eta$")

    # each entry is a swatch plus its label, so centring has to account for the
    # label width; measure one entry once and lay the row out from that
    swatch, gap = 26.0, 5.0
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    probe = fig.text(0.0, 0.0, r"$L_{%d}$" % level_max, fontsize=11)
    label_w = probe.get_window_extent(rend).width
    probe.remove()
    item = swatch + gap + label_w + 26.0        # swatch, label, breathing room
    total = (level_max + 1) * item - 26.0
    x0 = g["x0"][-1] + g["col_w"] / 2.0 - total / 2.0
    y = strip_top - g["cbar_h"] / 2.0
    for k in range(level_max + 1):
        x = x0 + k * item
        fig.add_artist(plt.Rectangle((x / g["px_w"], (y - 5.5) / g["px_h"]),
                                     swatch / g["px_w"], 11.0 / g["px_h"],
                                     transform=fig.transFigure,
                                     facecolor=LEVEL_COLOURS[k],
                                     edgecolor=WHITE["spine"], linewidth=0.6))
        fig.text((x + swatch + gap) / g["px_w"], y / g["px_h"],
                 r"$L_{%d}$" % k, color="#222222", fontsize=11,
                 va="center", ha="left")
    return fig, rows, geo


def eta_image(src, pick, args):
    """The viscosity image for one row at one target."""
    import eta_panel
    idx = pick["index"]
    if src.kind == "our":
        fr = src.frames[idx]
        ng = int(fr["ng"])
        return eta_panel.our_eta_grid(src.frames_dir, int(fr["step"]), src.lx,
                                      ng, ng)
    return eta_panel.aspect_eta_grid(src.sol_dir, src.index[idx], src.lx,
                                     src.nx, src.nz)


def render_two_col(sources, labels, plan, norm, cmap, args, frames_dir, case, t0, t1,
                   px_h=None):
    """The clean layout: write frames_dir/f_%05d.png, or one still."""
    lx = args.lx
    h_ref, h_min = 0.0, float("inf")
    for j, src in enumerate(sources):
        if args.interp == "linear":
            idxs = sorted({k for fr in plan
                           for k in (fr["picks"][j]["lo"], fr["picks"][j]["hi"])})
        else:
            idxs = sorted({fr["picks"][j]["index"] for fr in plan})
        if len(idxs) > 4:
            idxs = [idxs[k] for k in np.linspace(0, len(idxs) - 1, 4).astype(int)]
        for i in idxs:
            _, h = src.cells(i)
            if len(h):
                h_ref = max(h_ref, float(np.nanmax(h)))
                h_min = min(h_min, float(np.nanmin(h)))
    if not (h_ref > 0.0):
        h_ref = 0.0625
    if not np.isfinite(h_min):
        h_min = h_ref
    level_max = int(level_of(np.array([h_min]), h_ref)[0])

    # one anomaly range for the whole clip, from a sample of the plan
    sample_vals = []
    for k in range(0, len(plan), max(1, len(plan) // 12)):
        for j, src in enumerate(sources):
            T, _, _ = sample_temperature(src, plan[k]["picks"][j], args.interp)
            d = temperature_anomaly(T)
            sample_vals.append(np.abs(d).ravel()[::7])
    vals = np.concatenate(sample_vals) if sample_vals else np.array([0.1])
    peak = float(np.percentile(vals, 99.0))
    peak = max(0.05, np.ceil(peak * 20.0) / 20.0)
    norm_theta = Normalize(vmin=-peak, vmax=peak)
    print("[three_row] anomaly fixed at +-%.2f, T at %.3f..%.3f"
          % (peak, norm.vmin, norm.vmax), flush=True)

    n_cols = 4 if args.eta else 3
    geo = clean_geometry(args.px_w, len(labels), lx, n_cols=n_cols)
    print("[three_row] clean layout: %d x %d px, columns %d px, panel %d px, "
          "levels 0..%d" % (geo["px_w"], geo["px_h"], geo["col_w"],
                            geo["panel_h"], level_max), flush=True)
    norm_eta = eta_norm() if args.eta else None
    fig, rows, geo = build_clean_figure(labels, norm, cmap, geo, level_max,
                                        norm_theta, norm_eta)
    t_start = time.time()
    for k, fr in enumerate(plan):
        for j, (row, pick) in enumerate(zip(rows, fr["picks"])):
            src = sources[j]
            T, mesh_i, note = sample_temperature(src, pick, args.interp)
            quads, h = src.cells(mesh_i)
            st = src.status(mesh_i, h_ref)
            if note:
                st = dict(st, mesh_note=note)
            dT = temperature_anomaly(T)
            dE = None
            if args.eta:
                dE = eta_image(src, pick, args)
            row.update(T, dT, quads, h, h_ref, status_line2(st),
                       status_line3(dT), status_line1(st), dE,
                       status_line4(dE) if dE is not None else "")
        if args.still:
            if k == args.still_index:
                fig.savefig(args.still, dpi=DPI, facecolor=fig.get_facecolor())
                print("[three_row] still -> %s" % args.still, flush=True)
                plt.close(fig)
                raise SystemExit(0)
        else:
            path = os.path.join(frames_dir, "f_%05d.png" % k)
            fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
            if not args.quiet and (k % 10 == 0 or k == len(plan) - 1):
                print("[three_row] frame %d/%d  %s  (%.1f s elapsed)"
                      % (k + 1, len(plan), os.path.basename(path),
                         time.time() - t_start), flush=True)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row", action="append", required=True,
                    metavar="LABEL=KIND:PATH",
                    help="one row, repeated; rows are stacked top to bottom, "
                         "any number works (3 for the verification video, 1 "
                         "for a single arm)")
    ap.add_argument("--t0", type=float, default=None, help="window start [t_nd]")
    ap.add_argument("--t1", type=float, default=None, help="window end [t_nd]")
    ap.add_argument("--frames", type=int, default=0,
                    help="target frame count (default: the row with fewest frames)")
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--time-grid", choices=["frames", "linear", "log"],
                    default="frames",
                    help="where the common target times come from: the frame "
                         "times of the least-sampled row (default), or an even "
                         "grid over the window, linear or logarithmic in t")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--out", default="out_julia/verify100_render/three_row.mp4")
    ap.add_argument("--case", default=None,
                    help="case name for the header line (default: the output stem)")
    ap.add_argument("--lx", type=float, default=2.5, help="box width")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--scan", type=int, default=80,
                    help="frames per row sampled when measuring the shared scale")
    ap.add_argument("--aspect-grid", type=int, nargs=2, default=(640, 256),
                    metavar=("NX", "NZ"))
    ap.add_argument("--aspect-temp", choices=["auto", "kelvin", "nondim"],
                    default="auto")
    ap.add_argument("--aspect-nd", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--aspect-mesh", choices=["edges", "none"], default="edges")
    ap.add_argument("--interp", choices=["nearest", "linear"], default="nearest",
                    help="nearest: show the nearest frame of each row (default, "
                         "the one-column behaviour); linear: blend the two "
                         "bracketing temperature fields, mesh from the nearest "
                         "frame (two-column mode)")
    ap.add_argument("--eta", action="store_true",
                    help="add the log10(viscosity) panel between the anomaly and "
                         "the mesh, rebuilt from each code's own state")
    ap.add_argument("--px-w", type=int, default=1920,
                    help="canvas width in pixels (default 1920); a 3-column "
                         "layout at 2400 keeps the panels legible after a "
                         "downscale to a README GIF")
    ap.add_argument("--layout", choices=["clean", "classic"], default="clean",
                    help="the 2-column layout was replaced by the three-panel "
                         "clean one; the flag is accepted for old commands")
    ap.add_argument("--still", default=None, metavar="PATH",
                    help="write one PNG at --still-index instead of the frames")
    ap.add_argument("--still-index", type=int, default=-1,
                    help="which target to draw for --still (default: the middle)")
    ap.add_argument("--columns", type=int, choices=[1, 2], default=1,
                    help="1: one panel per row (temperature + mesh); 2: a second "
                         "panel per row with the mesh only, coloured by refinement "
                         "level, on a white canvas")
    ap.add_argument("--also-1080", action="store_true",
                    help="also write a 1920x1080 sibling: an exact 2:1 "
                         "downscale for 3 rows, otherwise a fit-and-pad that "
                         "keeps the 2.5:1 field geometry")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--ffmpeg", default=None)
    ap.add_argument("--rm-pngs", action="store_true",
                    help="delete the per-frame PNGs after encoding")
    ap.add_argument("--list", action="store_true",
                    help="print the frame inventory of every row and exit")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    labels, sources = [], []
    for spec in args.row:
        lab, src = parse_row(spec, args)
        labels.append(lab)
        sources.append(src)
    if not sources:
        raise SystemExit("no --row given")
    for lab, s in zip(labels, sources):
        if len(s) == 0:
            where = s.frames_dir if isinstance(s, OurSource) else s.sol_dir
            raise SystemExit("row %r has no readable frames under %s"
                             % (lab, where))

    for lab, s in zip(labels, sources):
        print("[three_row] row %-28s %s  %4d frames  t_nd %s .. %s"
              % (lab, s.kind, len(s), sigfig(s.times[0]), sigfig(s.times[-1])),
              flush=True)
    if args.list:
        for lab, s in zip(labels, sources):
            print("\n== %s (%s)" % (lab, s.kind))
            for i, f in enumerate(s.frames):
                print("   %4d  t_nd=%12.5e  %s" % (i, f["t"], s.describe(i)))
        return 0

    t0, t1, plan = build_plan(sources, labels, args)

    # ---- shared colour scale: scan every row, one range over the whole window
    lo, hi = [], []
    n_scan = max(2, args.scan)
    for j, (lab, s) in enumerate(zip(labels, sources)):
        if args.interp == "linear":
            idxs = sorted({k for fr in plan
                           for k in (fr["picks"][j]["lo"], fr["picks"][j]["hi"])})
        else:
            idxs = sorted({fr["picks"][j]["index"] for fr in plan})
        if len(idxs) > n_scan:
            take = np.linspace(0, len(idxs) - 1, n_scan).astype(int)
            idxs = [idxs[k] for k in sorted(set(take))]
        for i in idxs:
            a, b = s.raw_range(i)
            lo.append(a)
            hi.append(b)
        print("[three_row] scanned %d frames of %r for the scale"
              % (len(idxs), lab), flush=True)
    vmin = args.vmin if args.vmin is not None else min(0.0, floor2(min(lo)))
    vmax = args.vmax if args.vmax is not None else max(1.0, ceil2(max(hi)))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("inferno")
    print("[three_row] shared colour scale T = [%s, %s]" % (sigfig(vmin),
                                                            sigfig(vmax)),
          flush=True)

    # ---- render
    out = os.path.abspath(args.out)
    stem, _ = os.path.splitext(out)
    frames_dir = stem + "_frames"
    os.makedirs(frames_dir, exist_ok=True)
    for old in glob.glob(os.path.join(frames_dir, "f_*.png")):
        os.remove(old)
    if args.interp == "linear" and args.columns != 2:
        print("[three_row] --interp linear only applies to --columns 2; the "
              "one-column layout stays nearest-frame", flush=True)
    px_w, px_h = canvas_px(len(labels),
                            PX_ROW_2COL if args.columns == 2 else PX_ROW)
    case = args.case or os.path.basename(stem)
    if args.still and args.still_index < 0:
        args.still_index = len(plan) // 2 if "plan" in dir() else 0
    if args.columns == 2:
        render_two_col(sources, labels, plan, norm, cmap, args, frames_dir,
                       case, t0, t1, px_h)
    else:
        header = ("%s  |  t = [%s, %s]  |  %d frames @ %g fps"
                  % (case, sigfig(t0), sigfig(t1), len(plan), args.fps))
        print("[three_row] header: %s" % header, flush=True)
        worst = [max((time_text(fr["target"], fr["picks"][j], s)
                      for fr in plan),
                     key=len) for j, s in enumerate(sources)]
        fig, views = build_figure(labels, norm, cmap, header, worst)

        t_start = time.time()
        for k, fr in enumerate(plan):
            for j, (view, pick) in enumerate(zip(views, fr["picks"])):
                src = sources[j]
                T, segs = src.draw(pick["index"])
                view.update(T, segs, args.lx,
                            time_text(fr["target"], pick, src),
                            mesh=(segs is not None))
            path = os.path.join(frames_dir, "f_%05d.png" % k)
            fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
            if not args.quiet and (k % 10 == 0 or k == len(plan) - 1):
                print("[three_row] frame %d/%d  %s  (%.1f s elapsed)"
                      % (k + 1, len(plan), os.path.basename(path),
                         time.time() - t_start), flush=True)
        plt.close(fig)

    # ---- poster = the final frame stack
    poster = stem + "_poster.png"
    shutil.copyfile(os.path.join(frames_dir, "f_%05d.png" % (len(plan) - 1)),
                    poster)

    # ---- encode
    ffmpeg = find_ffmpeg(args.ffmpeg)
    cmd, dt = encode(ffmpeg, frames_dir, "f_%05d.png", args.fps, out,
                     crf=args.crf, preset=args.preset)
    print("[three_row] encoded %s (%.1f s) with %s" % (out, dt, ffmpeg), flush=True)
    alt = None
    if args.also_1080:
        alt = stem + "_1080p.mp4"
        encode(ffmpeg, frames_dir, "f_%05d.png", args.fps, alt, crf=args.crf,
               preset=args.preset,
               vf=sibling_1080_vf(px_h,
                                  pad="0xffffff" if args.columns == 2
                                  else "0x0d0d0d"))
        print("[three_row] encoded %s" % alt, flush=True)

    # ---- sidecar with the plan (provenance)
    meta = dict(
        case=case, out=out, poster=poster, frames_dir=frames_dir,
        width=px_w, height=px_h, row_px=[PX_ROW] * len(labels),
        columns=args.columns,
        field_px=[PX_FIELD_W, PX_FIELD_H], fps=args.fps, nframes=len(plan),
        duration_s=len(plan) / args.fps, window=[t0, t1],
        time_grid=args.time_grid, interp=args.interp,
        scale=dict(vmin=vmin, vmax=vmax, cmap="inferno"),
        rows=[dict(label=lab, kind=s.kind,
                   path=(s.frames_dir if isinstance(s, OurSource) else s.sol_dir),
                   nframes=len(s), t_range=[s.times[0], s.times[-1]])
              for lab, s in zip(labels, sources)],
        targets=[dict(k=fr["k"], t=fr["target"],
                      picks=[dict(row=labels[p["row"]], index=p["index"],
                                  t=p["t"], delta=p["delta"], lo=p["lo"],
                                  hi=p["hi"], w=p["w"],
                                  describe=sources[p["row"]].describe(p["index"]),
                                  describe_lo=sources[p["row"]].describe(p["lo"]),
                                  describe_hi=sources[p["row"]].describe(p["hi"]))
                             for p in fr["picks"]])
                 for fr in plan],
        ffmpeg=ffmpeg, encode_cmd=cmd, alt_1080p=alt,
        png_pattern="f_%05d.png", dpi=DPI)
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    size = os.path.getsize(out)
    print("[three_row] %s  %.1f MB  %d frames @ %g fps = %.2f s"
          % (out, size / 1e6, len(plan), args.fps, len(plan) / args.fps),
          flush=True)
    if alt:
        print("[three_row] %s  %.1f MB" % (alt, os.path.getsize(alt) / 1e6),
              flush=True)
    print("[three_row] poster %s" % poster, flush=True)
    if args.rm_pngs:
        shutil.rmtree(frames_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
