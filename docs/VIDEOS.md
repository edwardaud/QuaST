# VIDEOS.md: the comparison clips and how they are made

`scripts/three_row_video.py` turns runs of the same 2.5 x 1 convection case into
one labelled video: one row per run, the same pixel size, one temperature colour
scale, one common frame list. A difference between two rows is then a difference
in the solution, because the plot no longer has room to differ. The clips answer
one question with the eye: do the sparse direct solver, the matrix-free multigrid
and ASPECT produce the same evolution?

**Run artifacts.** The `out_julia/...` paths in this document name the development tree's
render directories and run directories, and the `aspect_runs/realistic/output-*` paths name
ASPECT run directories from the same tree. The clips and the runs are not part of this
repository; section 5 records how they were made, and its commands need those directories
present before they will run.

| file | role |
|---|---|
| `scripts/three_row_video.py` | renderer, planner and encoder |
| `scripts/live_render.py` | the frame readers reused by the renderer |
| `out_julia/verify100_render/*.mp4` | the clips (gitignored) |
| `out_julia/verify100_render/*.json` | each clip's plan and provenance |

This file replaces the earlier `VIDEOS.md`; the plumbing numbers that file
carried for the 2026-11-12 test runs are not repeated here.

## 1. What the clips are for

### 1.1 The question

There are three evolutions to compare. Two are ours: the same case solved with
`linear=:direct` and with `linear=:mg`. The third is ASPECT's answer to the same
case. The two of our runs are controlled: `julia/scripts/verify100_case.sh` writes
every AMR knob out explicitly and sets `CASE_STOKES_SCALING=equilibrated` for both
arms, so the only difference between those rows is `CASE_LINEAR`. Both start from
scratch at t = 0 on the production G card, not from a saved checkpoint. A single
frame cannot settle it: two fields can look alike at one time and differ in when
the plumes start, how many there are, or where the mesh spends its cells.

### 1.2 What to look at

Watch three things, in this order.

1. The onset. In the verify7000 window the cold lid dips at frame 10 of the
   100-frame clip (target t = 3.322e-5): one downwelling at x = 1.25, in both
   rows. By frame 30 (t = 9.767e-5) four cold sheets are sinking inside the box.
   An onset that differs by one or two frames between rows is visible.

2. The plume count and the x positions. At the last frame of the 100-frame clip
   the MG field (step 2171, t = 2.8896e-4) has seven downwellings crossing
   z = 0.75, at x = 0.41, 0.65, 0.93, 1.25, 1.58, 1.85, 2.11. The ASPECT field at
   t = 2.8802e-4 also has seven, at x = 0.356, 0.579, 0.955, 1.256, 1.549, 1.921,
   2.140. The count matches; the largest position difference is 0.071 in x, under
   3% of the 2.5 box width. Both lists come from the same scan: a horizontal cut
   at z = 0.75, smoothed over five grid cells, local minima at least 0.02 below
   the mean.

3. The mesh panels, in the two-column clips. Our tree puts its finest cells (L5)
   along the lid and down each plume stem. ASPECT's forest puts its finest cells
   (L4) in vertical bands through the stems and leaves the lid at L2 to L3. Two
   codes can agree on the field and spend their cells differently; this panel is
   where that shows.

The single-row clip answers a fourth question, what the case does after the
comparison window: the seven downwellings of t = 2.89e-4 merge to five by 8.0e-4
and four by 1.141e-3 (same z = 0.75 cut). A missing plume, an onset shifted by
more than a frame, or a region refined in one row and not the other is the kind of
difference these clips exist to expose. A colour that means a different cell size
in the two rows is a renderer bug, and sections 3.1 and 3.4 rule it out.

Every frame prints what it shows: the row label names the run, the same line prints
the target `t*`, the frame actually displayed, its step and the offset `delta`, and
the two-column status block adds the leaf or cell count, the level range and
`Nu_top`, `Vrms`, `Tmax` with their source. At the last frame of the 100-frame clip
those read `leaves 9,985`, `Vrms 484` for MG and `cells 25,972`, `Vrms 523` for
ASPECT, 3.1e-5 apart in t; the offset is printed next to them.

## 2. The pipeline

A row is `--row "LABEL=KIND:PATH"`; `our` takes a run root or its `frames/`,
`aspect` an ASPECT directory or its `solution/`; N is the number of rows.

### 2.1 Our frames

A run root holds `frames/` with three files per frame, written by
`julia/scripts/case_realistic_2d.jl`:

| file | content |
|---|---|
| `frame_meta.csv` | one row per frame, 19 columns: `frame, step, t, dt, N, leaves, nlev, Tmin, Tmax, Tmean, Nu_top, Nu_bot, Vrms, umax, eta_min, eta_max, eta_clip_frac, wall_s, ng` |
| `frame_NNNNN_T.bin` | `ng x ng` little-endian float64, x fastest, so the array is `[z, x]`; `ng = 256`, 524,288 bytes |
| `frame_NNNNN_mesh.csv` | one line per leaf, `level,x0,y0,h` |

The grid is the 256 x 256 frame grid over x in [0, 2.5] and z in [0, 1]; the AMR
mesh is separate, and the leaf boxes are what the cyan overlay draws. The renderer
reads `step`, `t`, `ng` and the three diagnostics by column name. The verify100
arms run 500 steps with frame stride 5, so they write 100 frames (steps 1, 6, ...,
496) plus the final state at step 500: 101 frames, not uniformly spaced in t
because the timestep changes while the stride does not.

`read_frame_bin` checks the file size against `ng*ng*8` and returns `None` for a
frame still being written; `OurSource` drops those at load time, so a video
pointed at a live run skips the newest frame instead of drawing it half written.

### 2.2 ASPECT's output

An ASPECT output directory holds `solution/` with, for each output index, one
`solution-NNNNN.PPPP.vtu` per MPI rank: six pieces in the verify7000 and verify100
runs, plus one `.pvtu` and one `.visit` per index. The pattern is
`solution-(\d+)\.(\d+)\.vtu$`, so the six `.vtu` pieces are read and the `.pvtu`,
a manifest over those same pieces, is skipped.

Once per index: read every piece with `meshio` (`points[:, :2]` as x, z, the point
field `T`, the `TIME` field from `field_data`); concatenate and drop duplicate
interface nodes with `np.unique` on coordinates rounded to 1e-9; detect units once
(`max|P| > 5` means dimensional coordinates, `max T > 10` means kelvin);
interpolate onto a regular grid with a linear Delaunay interpolant and fill the
convex-hull rim from the nearest node.

The conversions are `x_nd = x/D`, `z_nd = z/D`, `T_nd = T/dT - 0.5` and
`t_nd = t_s * kappa/D^2`, with `D = 2.9e6 m`, `kappa = 1e-6 m^2/s`, `dT = 1000 K`,
`T0 = D^2/kappa`; `--aspect-nd` and `--aspect-temp` override the detection. A
missing `TIME` field is a hard error: without it the order of the frames would be
a guess.

The sampling grid is `640 x 256` (`--aspect-grid`), so one sample is 0.0039 of the
box wide, finer than the finest ASPECT cell here (0.0098) and coarser than the
1650 x 660 display area. The quad cells feed the cyan overlay and the level
colouring. Diagnostics come from the run's `statistics` file, columns found by
header name, nearest row in time, with `Nu_top = flux/(2.5 * 3.3 * 1000)`,
`Vrms * D/kappa` and `Tmax/dT - 0.5`.

### 2.3 The readers are live_render's

The renderer imports `live_render` and calls its `read_meta`, `read_frame_bin`,
`read_mesh` and `mesh_segments`, and builds the temperature image with
`lr.draw_temperature` (same `inferno` colormap, same 0 to 2.5 by 0 to 1 extent,
same cyan overlay). After the first frame it overrides the colour limits with
`set_clim` for the shared scale of section 3.1.

That reuse is why the video and the live page cannot disagree about the format:
one parser for the metadata, one for the binary, one for the mesh. The renderer
adds one argument to the reader, `read_mesh(..., with_levels=True)`, returning
`(level, x0, y0, h)` for the level column; the default return is unchanged.

## 3. The rules that make the comparison honest

### 3.1 One colour scale

One scale for every frame of every row:

    vmin = min(0, floor2(min T over the sampled frames of every row))
    vmax = max(1, ceil2(max T over the sampled frames of every row))

`floor2` and `ceil2` round outward to two significant digits, so the scale cannot
clip a sampled value. The extremes are measured on the raw field: for an ASPECT
row that is the vertex `T` before the interpolation onto the display grid. At most
`--scan` frames per row are sampled (default 80, evenly spaced); with
`--interp linear` both bracketing frames of every target are included.

The scale is printed at start, drawn as the colour bar's end ticks, and stored in
the `.json`; `--vmin/--vmax` override it. All five clips recorded `[0, 1.9]`. One
scale means a shade is one temperature at every time and in every row; a per-frame
autoscale would turn the growth of the plumes into a colour drift.

### 3.2 One time axis

* N is the number of frames the least-sampled row has inside the window, unless
  `--frames` sets it (`--max-frames`, default 400, caps it).
* With `--time-grid frames` (the default) the targets are that row's frame times,
  decimated in index to N. Every frame of the reference row appears once and the
  denser rows are decimated onto it.
* `--time-grid linear|log` replaces the cadence with an even (linear) or
  logarithmic grid over the window. The 100-frame clip uses `linear`: under a
  plain `frames` cadence its 100 targets would repeat, because the reference row
  has 32 frames there.
* Each row shows the frame nearest the target, clamped to the frames inside the
  window. The offset is printed on the row label and listed in the `.json` for
  every target and every row.

The clamp matters at both ends: a target past a row's last frame shows that frame
and carries the offset. The run prints a `NOTE` when a row repeats more than one
target or when its worst offset exceeds 2% of the window. In `verify100_3row` the
ASPECT cadence is the reference: 97 targets, while each of our rows shows 54
distinct frames, worst offset 2.449e-6, 7.7% of the window.

### 3.3 `--interp linear`: blending temperature, never the mesh

With `--interp linear` (two-column mode only) a row blends the two frames that
bracket the target:

    w = (t - t_lo) / (t_hi - t_lo)
    T = (1 - w) * T_lo + w * T_hi

The fields of one row share a grid across frames (256 x 256 for ours, 640 x 256
for the ASPECT sampling), so the blend is an element-wise weighted sum with no
resampling. The mesh column cannot be blended: the leaf set changes between frames
and there is no element-wise correspondence between two AMR trees. It always shows
the nearest frame and names it in the status line (`mesh: nearest frame, step
2171`); the row label switches to `blend t_lo (step A) to t_hi (step B), w 0.42`.
The shared scale is measured over both bracketing frames, so a blend cannot exceed
it. The one-column mode ignores `--interp` and stays nearest-frame; the script
prints a note if the flag is passed with `--columns 1`.

### 3.4 One level convention

Both mesh panels colour every cell by `L = round(log2(h_ref / h))`, with `h` the
cell edge and `h_ref` the coarsest cell over the sampled frames of all rows (up to
four plan frames per row). `h_ref` is 0.125 in these clips, printed in the legend
as `L0 = 362 km`, and each level halves the edge. The palette is a fixed list of
eight colours and the index is clipped to `[0, 7]`.

The input is the cell edge. Our `frame_mesh.csv` carries a `level` column, our own
tree's AMR level counted from our base grid; ASPECT's numbering starts from its
base. The two numbers are not comparable; the edge length is: a cell of 0.0625 is
L1 whichever code wrote it, so a colour means the same physical cell size in both
rows. For ours `h` comes from the leaf box; for ASPECT it is the quad's first edge.

Measured in the 100-frame clip: ours spans L1 to L5 (0.0625 to 0.0039, 11 km at
the finest), ASPECT L2 to L4 (0.039 to 0.0098, 28 km at the finest).

### 3.5 One 1080p rule

The master is `1920 x N*720`; `--also-1080` writes 1920 x 1080 and never stretches
the 2.5:1 field. Three rows (2160 px) get the exact 2:1 downscale,
`scale=1920:1080:flags=lanczos,setsar=1`. Any other height is fitted with
`force_original_aspect_ratio=decrease` and padded to 1920 x 1080 with the frame
background: `0x0d0d0d` for the dark one-column theme, `0xffffff` for the
two-column theme. A one-row 1920 x 720 master gains 180 px bands above and below;
a two-row 1920 x 1440 master becomes 1440 x 1080 with side bars.

## 4. The clips

All five are in `out_julia/verify100_render/`, H.264 High, `yuv420p`,
`-crf 18 -preset slow`, 10 fps. Sizes are decimal MB (bytes / 1e6).

| clip | rows | window (t_nd) | frames | duration | master | size |
|---|---|---|---|---|---|---|
| `mg7000_1row.mp4` | MG | 1.000e-6 to 1.1411e-3 | 75 | 7.5 s | 1920 x 720 | 2.50 MB |
| `mg_vs_aspect_2row.mp4` | MG, ASPECT | 1.000e-6 to 3.200e-4 | 32 | 3.2 s | 1920 x 1440 | 2.24 MB |
| `mg_vs_aspect_2row_2col.mp4` | MG, ASPECT | 1.000e-6 to 3.200e-4 | 32 | 3.2 s | 1920 x 1440 | 1.73 MB |
| `mg_vs_aspect_2row_2col_100f.mp4` | MG, ASPECT | 1.000e-6 to 3.200e-4 | 100 | 10.0 s | 1920 x 1120 | 2.56 MB |
| `verify100_3row.mp4` | direct, MG, ASPECT | 1.000e-6 to 3.2886e-5 | 97 | 9.7 s | 1920 x 2160 | 3.33 MB |

The 1080p siblings are 2.52, 1.34, 1.14, 2.60 and 1.77 MB; only the three-row one
is a true downscale, the rest are fit-and-pad per section 3.5.

### 4.1 `mg7000_1row.mp4`

The matrix-free arm alone, t = 1e-6 to 1.141e-3, 75 frames, all exact (with one
row the cadence is its own). Canvas 1920 x 720, field 1650 x 660, header
`verify7000 MG: 5100 steps from t=0`. It shows the whole run: a stratified start
with a faint wobble, one central plume, seven downwellings by 2.89e-4, then five
at 8.0e-4 and four at 1.141e-3 as the cells merge. No comparison row; this is the
reference for what the case does.

### 4.2 `mg_vs_aspect_2row.mp4`

Same MG run against ASPECT (`aspect_runs/realistic/output-verify7000`), one
column, 32 targets from the MG cadence, MG exact on all 32; ASPECT shows 30
distinct frames, worst offset 2.21e-6 (0.69%). The last target is MG 2.8896e-4
against ASPECT 2.8802e-4. Each row is a 1650 x 660 field with the cyan overlay and
the shared bar. The plume set forms and sinks, both rows agree on the count and on
the shape of every cell, and the ASPECT row is visibly smoother (the 640 x 256
sampling). Both rows end within 1e-6: this is the apples-to-apples clip.

### 4.3 `mg_vs_aspect_2row_2col.mp4`

The same 32 targets in two-column mode: left panel T plus a thin grey mesh, right
panel the mesh alone coloured by level, white canvas, 1920 x 1440, 1.73 MB. It was
rendered before commit 2273668 tightened the band, so its rows are 720 px and the
master is 1440 px; rerunning the section 5 command today gives 1920 x 1120 (band
560 px). It adds the refinement comparison: our panel is large L1/L2 cells with L4
haloes and L5 strips on the lid and stems; ASPECT's is L3 through the interior, L2
in the top rows, L4 bands down the stems. Same seven plumes, different trees.

### 4.4 `mg_vs_aspect_2row_2col_100f.mp4`

The smooth version of 4.3: 100 targets on an even grid, 10 s at 10 fps,
1920 x 1120, 2.56 MB. MG has 32 real frames in the window, ASPECT 100. With
`--interp linear`, 89 targets fall strictly between two MG frames and are blended;
the first sits on MG's first frame; the last 10 lie past MG's coverage (2.8896e-4)
and repeat it, the last offset 3.10e-5 (9.7% of the window). ASPECT blends on 98,
worst offset 2.21e-6 (0.69%). The count goes one at frame 10, four at frame 30,
seven at the end. The mesh column's step stays put for stretches and then jumps,
because geometry is never blended.

### 4.5 `verify100_3row.mp4`

The three-arm clip: sparse direct, matrix-free MG and ASPECT from t = 0. Window
1.000e-6 to 3.2886e-5 (the intersection); ASPECT is the least-sampled row, so 97
targets come from its cadence. Our two runs write frames every 5 steps (1, 6, ...,
496, plus 500), so their rows step in lockstep; the frame times differ only in the
last digits (direct and ASPECT end at 3.2893e-5, MG at 3.2886e-5). Each shows 54
distinct frames (43 repeats, worst offset 2.449e-6). By the last target
(t = 3.258e-5) a single cold downwelling sits at x = 1.25 in all three rows. Our
overlays show the tree refined on the plume and the lid; ASPECT's mesh looks
uniform here.

Also in the directory: the earlier plumbing clips `test_g20_3row.mp4`,
`test_aspect_3row.mp4` and `mg_only*.mp4`, not part of this catalogue.

## 5. Reproduction

Use the interpreter that has numpy, matplotlib, scipy and meshio; the system
`python3` on this machine has none of them.

    PY=/home/szh/dedalus-env/bin/python

One row, the whole run, 1920 x 720:

    $PY scripts/three_row_video.py \
      --row "matrix-free MG=our:out_julia/verify7000_mg" \
      --fps 10 \
      --case "verify7000 MG: 5100 steps from t=0" \
      --out out_julia/verify100_render/mg7000_1row.mp4 --also-1080

Two rows, one column, window 1e-6 to 3.2e-4:

    $PY scripts/three_row_video.py \
      --row "matrix-free MG=our:out_julia/verify7000_mg" \
      --row "ASPECT=aspect:aspect_runs/realistic/output-verify7000" \
      --t0 1e-6 --t1 3.2e-4 --fps 10 \
      --case "verify7000: MG vs ASPECT, 0 to 3.2e-4" \
      --out out_julia/verify100_render/mg_vs_aspect_2row.mp4 --also-1080

Two columns, the same 32 targets:

    $PY scripts/three_row_video.py \
      --row "matrix-free MG (verify7000)=our:out_julia/verify7000_mg" \
      --row "ASPECT (verify7000)=aspect:aspect_runs/realistic/output-verify7000" \
      --t0 1e-6 --t1 3.2e-4 --columns 2 --fps 10 \
      --case "verify7000: MG vs ASPECT, two columns" \
      --out out_julia/verify100_render/mg_vs_aspect_2row_2col.mp4 --also-1080

Two columns, 100 frames, temperature blended between frames:

    $PY scripts/three_row_video.py \
      --row "matrix-free MG (verify7000)=our:out_julia/verify7000_mg" \
      --row "ASPECT (verify7000)=aspect:aspect_runs/realistic/output-verify7000" \
      --t0 1e-6 --t1 3.2e-4 --frames 100 --fps 10 \
      --columns 2 --interp linear --time-grid linear \
      --case "verify7000: MG vs ASPECT, two columns" \
      --out out_julia/verify100_render/mg_vs_aspect_2row_2col_100f.mp4 --also-1080

The three-row clip is the same call with three `--row` arguments, no `--t0/--t1`
(the window is the intersection) and `--fps 10`; its rows are in the `.json`. A
window with no frame of some row stops the script and prints what each row covers.
`--list` prints every row's frame inventory and exits.

The encoder is found by `find_ffmpeg`: `--ffmpeg PATH`, then
`shutil.which("ffmpeg")`, then `imageio_ffmpeg.get_ffmpeg_exe()`, then a hard-coded
path in the dedalus env and a Windows path. The recorded clips used
`/home/szh/dedalus-env/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2`.
The command is `-framerate FPS -i f_%05d.png [-vf VF] -c:v libx264 -pix_fmt yuv420p
-crf 18 -preset slow -r FPS -movflags +faststart`, with the `-vf` of section 3.5
for the 1080p sibling.

Each run writes:

| output | content |
|---|---|
| `<stem>.mp4` | the master stack |
| `<stem>_1080p.mp4` | the 1920 x 1080 sibling, with `--also-1080` |
| `<stem>_poster.png` | the last frame stack, at master size |
| `<stem>.json` | rows, window, targets, every pick and offset, scale, ffmpeg, encode command |
| `<stem>_frames/f_%05d.png` | the stacks at 96 dpi, kept for re-encoding |

`--rm-pngs` deletes the last one after encoding. Nothing is written inside a run
directory: the renderer only reads `frames/` or `solution/`.

## 6. Limits

The thing to know before quoting the 100-frame clip: it contains 32 real MG frames.
Everything between them is a linear blend, and the mesh column never blends, so the
field is smooth while the geometry still jumps at frame boundaries. A feature
shorter than one MG frame interval (about 1e-5 in this window) can be missed. For
the same reason the last 10 targets repeat MG's last frame while ASPECT keeps
moving; the final second compares fields 3.1e-5 apart, 9.7% of the window.

In the three-row clip the resampling goes the other way: ASPECT sets the 97
targets and each of our rows repeats 43 of them. At 10 fps that reads as a
stutter; the step number on the label tells you when.

ASPECT is read only as `.vtu` pieces. The `.pvtu` of the same index is skipped, so
a run that wrote only `.pvtu` files would not render, and the `TIME` field is
mandatory. If ASPECT's naming or field names change, the pattern in
`AspectSource.__init__` has to change with them.

The ASPECT temperature is sampled on 640 x 256 before it is drawn. That grid is
finer than ASPECT's finest cell here (0.0098) and coarser than the display area,
so the ASPECT row is smoother than the MG row by construction, and interpolation
can shift a plume position slightly. `--aspect-grid` raises the sampling at a cost
in time and memory; the differences in section 1.2 are small, and the measured
numbers there are the evidence, with the clips as the visual check. Only
temperature is drawn; velocity, viscosity and `Nu(t)` stay in `frame_meta.csv`,
the run `history.csv` and the comparison scripts.

To produce frames for a new clip, set `CASE_OUT` explicitly to a directory you are
willing to overwrite. The card defaults to `out_julia/case_realistic` when the
variable is unset, and that directory held a spin-up's record; a two-step smoke
launched without the variable wrote over its `history.csv`, its first frames and
its frame metadata on 2026-11-12. `verify100_case.sh` always sets `CASE_OUT` and
refuses an existing output directory. A denser video needs more real frames: rerun
the case with `CASE_FRAME=1` (or another stride) in a fresh `CASE_OUT` and point
the renderer at it; the renderer only reads, so it is safe against any finished
run.
