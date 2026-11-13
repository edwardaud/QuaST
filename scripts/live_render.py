#!/usr/bin/env python3
"""Live renderer for the realistic 2-D mantle-convection case.

Polls the Julia run's output directory and renders every new frame to PNG:

    <root>/frames/frame_meta.csv        one row per frame (written by the driver)
    <root>/frames/frame_NNNNN_T.bin     T on an ng x ng grid, float64, x fastest
    <root>/frames/frame_NNNNN_mesh.csv  level,x0,y0,h  (one line per leaf)

Outputs (all under <root>):

    render/frame_NNNNN.png    temperature + mesh overlay
    render/panel_NNNNN.png    same, next to Nu(t)/Vrms(t)
    live.gif                  the last frames, rebuilt every N frames
    live.html                 self-refreshing page showing the newest panel

The renderer only ever *reads* the solver's files and writes into its own
subdirectories, so it cannot slow the run down; a frame whose binary is not yet
completely written is skipped and picked up on the next poll.

    /home/szh/dedalus-env/bin/python scripts/live_render.py \
        --dir out_julia/case_realistic --interval 2 --gif-every 10

No third-party dependency beyond numpy/matplotlib/Pillow.
"""
import argparse
import csv
import glob
import io
import json
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.collections import LineCollection    # noqa: E402
from PIL import Image                                # noqa: E402

META_COLS = ["frame", "step", "t", "dt", "N", "leaves", "nlev", "Tmin", "Tmax",
             "Tmean", "Nu_top", "Nu_bot", "Vrms", "umax", "eta_min", "eta_max",
             "eta_clip_frac", "wall_s", "ng"]


# --------------------------------------------------------------------------- io
def read_meta(path):
    """Rows of frame_meta.csv as dicts; tolerant of a partially written tail."""
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", newline="") as fh:
            text = fh.read()
    except OSError:
        return []
    if not text:
        return []
    lines = text.splitlines()
    if not text.endswith("\n"):
        lines = lines[:-1]              # drop the incomplete last line
    if not lines:
        return []
    header = [h.strip() for h in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) != len(header):
            continue
        try:
            row = {k: float(v) for k, v in zip(header, parts)}
        except ValueError:
            continue
        row["step"] = int(row["step"])
        row["ng"] = int(row["ng"])
        rows.append(row)
    return rows


def read_frame_bin(path, ng):
    """(ng,ng) float64 array A[z,x], or None while the file is incomplete."""
    need = ng * ng * 8
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    if size != need:
        return None
    try:
        raw = np.fromfile(path, dtype="<f8")
    except OSError:
        return None
    if raw.size != ng * ng:
        return None
    return raw.reshape(ng, ng)          # x varies fastest -> [z, x]


def read_mesh(path, with_levels=False):
    """Leaf boxes (x0, y0, h), or (level, x0, y0, h) if with_levels; else None.

    The default stays exactly as it was, because every caller parses the tuples
    positionally; `with_levels=True` is for callers that colour by refinement
    level (scripts/three_row_video.py's mesh-only column).
    """
    if not os.path.isfile(path):
        return None
    boxes = []
    try:
        with open(path, "r") as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) != 4:
                    continue
                try:
                    if with_levels:
                        boxes.append((int(float(parts[0])), float(parts[1]),
                                      float(parts[2]), float(parts[3])))
                    else:
                        boxes.append((float(parts[1]), float(parts[2]),
                                      float(parts[3])))
                except ValueError:
                    continue
    except OSError:
        return None
    return boxes or None


# ----------------------------------------------------------------------- plot
def mesh_segments(boxes):
    segs = []
    for (x0, y0, h) in boxes:
        x1, y1 = x0 + h, y0 + h
        segs.append([(x0, y0), (x1, y0)])
        segs.append([(x1, y0), (x1, y1)])
        segs.append([(x1, y1), (x0, y1)])
        segs.append([(x0, y1), (x0, y0)])
    return segs


def draw_temperature(ax, T, boxes, lx, tmax_scale):
    im = ax.imshow(T, origin="lower", extent=(0.0, lx, 0.0, 1.0),
                   cmap="inferno", vmin=0.0, vmax=tmax_scale,
                   interpolation="bilinear", aspect="equal", zorder=1)
    if boxes:
        lc = LineCollection(mesh_segments(boxes), colors="cyan",
                            linewidths=0.18, alpha=0.35, zorder=2)
        ax.add_collection(lc)
    ax.set_xlim(0.0, lx)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    return im


def render_frame(row, root, outdir, lx=2.5, dpi=110, tmax_scale=2.0,
                 history=None):
    ng = row["ng"]
    step = row["step"]
    tag = "frame_%05d" % step
    T = read_frame_bin(os.path.join(root, "frames", tag + "_T.bin"), ng)
    if T is None:
        return None
    boxes = read_mesh(os.path.join(root, "frames", tag + "_mesh.csv"))

    fig = plt.figure(figsize=(11.0, 4.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.15])
    ax = fig.add_subplot(gs[0, 0])
    im = draw_temperature(ax, T, boxes, lx, tmax_scale)
    ax.set_title("T(x,z)  step %d  t=%.3e  leaves=%d  N=%d"
                 % (step, row["t"], int(row["leaves"]), int(row["N"])),
                 fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=7)

    ax2 = fig.add_subplot(gs[0, 1])
    hist = [r for r in (history or []) if r["Vrms"] > 0.0 and r["step"] <= step]
    if len(hist) >= 2:
        tt = np.array([r["t"] for r in hist])
        nut = np.array([r["Nu_top"] for r in hist])
        nub = np.array([r["Nu_bot"] for r in hist])
        vr = np.array([r["Vrms"] for r in hist])
        ax2.plot(tt, nut, "-", color="tab:red", lw=1.0, label="Nu_top")
        ax2.plot(tt, nub, "--", color="tab:orange", lw=1.0, label="Nu_bot")
        ax2.set_xlabel("t", fontsize=8)
        ax2.set_ylabel("Nu", fontsize=8, color="tab:red")
        ax2.tick_params(labelsize=7)
        ax2b = ax2.twinx()
        ax2b.plot(tt, vr, "-", color="tab:blue", lw=1.0, label="Vrms")
        ax2b.set_ylabel("Vrms", fontsize=8, color="tab:blue")
        ax2b.tick_params(labelsize=7)
        ax2b.set_yscale("log")
        lines = ax2.get_lines() + ax2b.get_lines()
        ax2.legend(lines, [ln.get_label() for ln in lines], fontsize=6,
                   loc="best")
        ax2.set_title("Nu(t), Vrms(t)", fontsize=9)
    else:
        ax2.text(0.5, 0.5, "waiting for\ntime series", ha="center",
                 va="center", fontsize=9, transform=ax2.transAxes)
        ax2.set_axis_off()

    info = ("eta in [%.2e, %.2e]  clipped %.2f%%  umax %.2e  wall %.0f s"
            % (row["eta_min"], row["eta_max"], 100.0 * row["eta_clip_frac"],
               row["umax"], row["wall_s"]))
    fig.suptitle("realistic 2-D case — " + info, fontsize=8)

    fpath = os.path.join(outdir, tag + ".png")
    ppath = os.path.join(outdir, "panel_%05d.png" % step)
    fig.savefig(fpath, dpi=dpi)
    fig.savefig(ppath, dpi=dpi)
    plt.close(fig)
    return fpath, ppath


def rebuild_gif(outdir, gif_path, max_frames=120, width=560, duration=180):
    panels = sorted(glob.glob(os.path.join(outdir, "panel_*.png")))
    if len(panels) < 2:
        return False
    panels = panels[-max_frames:]
    frames = []
    for p in panels:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        h = max(1, int(round(im.height * width / im.width)))
        frames.append(im.resize((width, h), Image.BILINEAR))
    if len(frames) < 2:
        return False
    tmp = gif_path + ".tmp"
    frames[0].save(tmp, format="GIF", save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True)
    os.replace(tmp, gif_path)
    return True


def write_html(root, outdir, rows, newest_panel, nframes):
    """Self-refreshing page; written atomically so a browser never sees half."""
    rel = os.path.relpath(newest_panel, root).replace(os.sep, "/") \
        if newest_panel else None
    tail = rows[-8:]
    trs = []
    for r in tail:
        trs.append("<tr><td>%d</td><td>%.4e</td><td>%.5f</td><td>%.5f</td>"
                   "<td>%.4e</td><td>%d</td><td>%.1f</td></tr>"
                   % (r["step"], r["t"], r["Nu_top"], r["Nu_bot"], r["Vrms"],
                      int(r["leaves"]), r["wall_s"]))
    img = ('<img src="%s" style="max-width:100%%;border:1px solid #888">'
           % rel) if rel else "<p><i>no frame yet</i></p>"
    html = """<!doctype html>
<html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="6">
<title>mantle-spectral — live</title>
<style>
 body{font-family:system-ui,sans-serif;margin:12px;background:#111;color:#ddd}
 table{border-collapse:collapse;font-size:12px;margin-top:8px}
 td,th{border:1px solid #444;padding:2px 6px;text-align:right}
 a{color:#7cf}
</style></head><body>
<h3>realistic 2-D mantle convection — live (%d frames rendered)</h3>
%s
<p><a href="live.gif">live.gif</a> &nbsp; <a href="frames/frame_meta.csv">frame_meta.csv</a>
&nbsp; <a href="history.csv">history.csv</a></p>
<table><tr><th>step</th><th>t</th><th>Nu_top</th><th>Nu_bot</th><th>Vrms</th>
<th>leaves</th><th>wall s</th></tr>
%s
</table>
<p style="color:#777;font-size:11px">page auto-refreshes every 6 s; newest
panel: %s</p>
</body></html>
""" % (nframes, img, "\n".join(trs), rel or "-")
    tmp = os.path.join(root, "live.html.tmp")
    with open(tmp, "w") as fh:
        fh.write(html)
    os.replace(tmp, os.path.join(root, "live.html"))


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="out_julia/case_realistic",
                    help="run output root (contains frames/)")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="seconds between polls")
    ap.add_argument("--gif-every", type=int, default=10,
                    help="rebuild live.gif every N rendered frames")
    ap.add_argument("--max-per-poll", type=int, default=2,
                    help="frames rendered per poll (keeps the loop light)")
    ap.add_argument("--lx", type=float, default=2.5, help="box width")
    ap.add_argument("--tmax", type=float, default=2.0,
                    help="fixed top of the colour scale")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--once", action="store_true",
                    help="render everything available, then exit")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.dir)
    outdir = os.path.join(root, "render")
    os.makedirs(outdir, exist_ok=True)
    metapath = os.path.join(root, "frames", "frame_meta.csv")
    gif_path = os.path.join(root, "live.gif")
    done = set()
    nframes = 0
    fresh = 0
    last_msg = 0.0
    if not args.quiet:
        print("[live_render] watching %s" % metapath, flush=True)
    while True:
        rows = read_meta(metapath)
        todolist = [r for r in rows if r["step"] not in done]
        todolist = todolist[:args.max_per_poll]
        rendered = None
        for r in todolist:
            try:
                out = render_frame(r, root, outdir, lx=args.lx,
                                   dpi=args.dpi, tmax_scale=args.tmax,
                                   history=rows)
            except Exception as exc:              # never die on one frame
                print("[live_render] frame %d failed: %r" % (r["step"], exc),
                      flush=True)
                done.add(r["step"])
                continue
            if out is None:
                continue                           # not completely written yet
            done.add(r["step"])
            nframes += 1
            fresh += 1
            rendered = out[1]
            if not args.quiet:
                print("[live_render] frame %d -> %s"
                      % (r["step"], os.path.basename(out[1])), flush=True)
        if rows and (rendered is not None or nframes == 0):
            newest = rendered or sorted(
                glob.glob(os.path.join(outdir, "panel_*.png")))
            newest = newest[-1] if isinstance(newest, list) and newest else newest
            write_html(root, outdir, rows, newest, nframes)
        if fresh >= args.gif_every and rows:
            try:
                if rebuild_gif(outdir, gif_path):
                    if not args.quiet:
                        print("[live_render] live.gif rebuilt", flush=True)
            except Exception as exc:
                print("[live_render] gif failed: %r" % (exc,), flush=True)
            fresh = 0
        if args.once:
            break
        if not args.quiet and time.time() - last_msg > 60.0:
            last_msg = time.time()
            print("[live_render] %d frames rendered, %d pending"
                  % (nframes, max(0, len(rows) - len(done))), flush=True)
        time.sleep(max(0.2, args.interval))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
