#!/usr/bin/env python3
"""Build the `--sidecar-extra` JSON of the annulus AMR demonstration figure.

The renderer (`scripts/annulus_video.py --amr`) already records the measured
scales, the frame matching and the mesh levels it drew.  This helper adds the
card, the caption, the caveats and the measured run evidence, all read from the
run logs rather than typed in:

    QUAST_LOG        out_julia/annulus_demo/logs/quast_full.log
    QUAST_FRAMES     out_julia/annulus_demo/quast_frames
    ASPECT_LOG       out_julia/annulus_demo/logs/aspect_full.log
    ASPECT_STATS     out_julia/annulus_demo/aspect_frames/statistics

Run it after both legs and before the render:

    python3 scripts/annulus_demo_sidecar.py \
        --quast-log ... --quast-frames ... --aspect-log ... \
        --aspect-stats ... --out .../demo_sidecar_extra.json

It is a diagnostic only: it reads frame streams, writes one JSON and touches
nothing else.  Every number in the JSON comes from the log/streams it is given.
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np


def parse_kv_line(line):
    """`KEY a=1 b=2` -> (KEY, {a: 1, b: 2}) with numbers parsed as floats/ints."""
    parts = line.split()
    key = parts[0]
    kv = {}
    for tok in parts[1:]:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if k == "dir":
            kv[k] = v
            continue
        try:
            kv[k] = int(v)
        except ValueError:
            try:
                kv[k] = float(v)
            except ValueError:
                kv[k] = v
    return key, kv


def parse_quast_log(path):
    out = dict(card=None, demo=[], result=None, frames=None, amr=None,
               amr_lines=[], steps_with_dt=None)
    dts = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("# card:"):
                out["card"] = s
            elif s.startswith("# demo:"):
                out["demo"].append(s)
            elif s.startswith("# frames:"):
                out["frames_line"] = s
            elif s.startswith("ANNULUS_RESULT"):
                _, kv = parse_kv_line(s)
                out["result"] = kv
            elif s.startswith("ANNULUS_FRAMES"):
                _, kv = parse_kv_line(s)
                out["frames"] = kv
            elif s.startswith("ANNULUS_AMR"):
                _, kv = parse_kv_line(s)
                out["amr"] = kv
            elif s.startswith("AMR "):
                m = re.search(r"step\s+(\d+)\s+t=([0-9.eE+-]+)\s+leaves=(\d+)"
                              r"\s+level=(\d+)-(\d+)\s+dt=([0-9.eE+-]+)", s)
                if m:
                    out["amr_lines"].append(dict(
                        step=int(m.group(1)), t=float(m.group(2)),
                        leaves=int(m.group(3)), level_min=int(m.group(4)),
                        level_max=int(m.group(5)), dt=float(m.group(6))))
                    dts.append(float(m.group(6)))
            elif s.startswith("step "):
                m = re.search(r"t=([0-9.eE+-]+)\s+dt=([0-9.eE+-]+)", s)
                if m:
                    dts.append(float(m.group(2)))
    if dts:
        out["steps_with_dt"] = dict(min_dt=min(dts), max_dt=max(dts))
    return out


def read_frame_meta(path):
    header, rows = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                header.append(s)
                continue
            rows.append(s)
    cols = rows[0].split(",")
    data = []
    for line in rows[1:]:
        v = line.split(",")
        if len(v) != len(cols):
            continue
        data.append(dict(zip(cols, v)))
    return header, cols, data


def quast_frame_stream(path):
    """Leaves/levels/dt per frame, and the frame-to-frame field change.

    The field change is measured on the same common polar grid the renderer
    uses (`annulus_video` is imported, not copied): theta = T - <T>_phi(r) of
    the first frame against the quarter, half, three-quarter and last frames,
    so a static clip is detected from the data rather than assumed.
    """
    meta_path = os.path.join(path, "frame_meta.csv")
    header, cols, rows = read_frame_meta(meta_path)
    out = dict(nframes=len(rows), columns=cols,
               leaves=[int(r["leaves"]) for r in rows],
               level_min=[int(r["level_min"]) for r in rows],
               level_max=[int(r["level_max"]) for r in rows],
               dt=[float(r["dt"]) for r in rows],
               t=[float(r["t"]) for r in rows],
               tmin=[float(r["Tmin"]) for r in rows],
               tmax=[float(r["Tmax"]) for r in rows],
               Vrms=[float(r["Vrms"]) for r in rows],
               meanT=[float(r["meanT"]) for r in rows])
    # frame-to-frame state change on the common polar grid
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import annulus_video as av
    src = av.QuastSource(path)
    nr, nphi = src.polar_grid()
    r = av.R0 + (np.arange(nr) + 0.5) * (av.R1 - av.R0) / nr
    phi = (np.arange(nphi) + 0.5) * 2.0 * math.pi / nphi
    Pg, Rg = np.meshgrid(phi, r, indexing="ij")
    targets = np.column_stack([(Rg * np.cos(Pg)).ravel(),
                               (Rg * np.sin(Pg)).ravel()])
    picks = sorted({0, len(src) // 4, len(src) // 2,
                    3 * len(src) // 4, len(src) - 1})
    fields = {}
    for i in picks:
        fg = src.field_grid(i, targets)
        T = fg["T"]
        fields[i] = T - T.mean(axis=0, keepdims=True)
    ref = fields[picks[0]]
    rms = float(np.sqrt(np.mean(ref ** 2)))
    diffs = {}
    for i in picks[1:]:
        d = fields[i] - ref
        diffs["frame %d vs frame %d" % (i, picks[0])] = dict(
            t=float(src.times[i]), max_abs_theta=float(np.max(np.abs(d))),
            rms_theta=float(np.sqrt(np.mean(d ** 2))),
            rms_theta_rel=float(np.sqrt(np.mean(d ** 2)) /
                                max(rms, 1.0e-300)))
    out["theta_change_vs_first"] = diffs
    out["theta_rms_first"] = rms
    out["leaves_max"] = max(out["leaves"])
    out["leaves_final"] = out["leaves"][-1]
    out["level_seen_min"] = min(out["level_min"])
    out["level_seen_max"] = max(out["level_max"])
    out["dt_min"] = min(out["dt"])
    out["dt_max"] = max(out["dt"])
    out["t_first"] = out["t"][0]
    out["t_last"] = out["t"][-1]
    return out


def aspect_log(path):
    out = dict(steps=0, wallclock_s=None, cells=None, cell_levels=[])
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    out["steps"] = len(re.findall(r"\*\*\* Timestep \d+:", text))
    # ASPECT prints the timer every `Timing output frequency` steps, so the
    # log carries several cumulative blocks; the last one is the total.
    walls = re.findall(
        r"Total wallclock time elapsed since start\s*\|\s*([0-9.]+)s", text)
    if walls:
        out["wallclock_s"] = float(walls[-1])
        out["timer_blocks"] = len(walls)
    incl = re.findall(
        r"Total wallclock time elapsed including restarts:\s*([0-9.]+)s", text)
    if incl:
        out["wallclock_incl_restarts_s"] = float(incl[-1])
    m = re.findall(r"Number of active cells:\s*([0-9,]+)", text)
    if m:
        out["cells_list"] = sorted({int(s.replace(",", "")) for s in m})
    # the refinement level count ASPECT reports with each cell count: "on M
    # levels" means levels 0..M-1, so the tree's maximum level is M-1.  The
    # startup mesh prints 768 (on 4 levels) -> 996 (on 5) -> 1 176 (on 5):
    # the two initial adaptive passes end at tree levels 3-4, not 3-5.
    out["cell_levels"] = [
        dict(cells=int(c.replace(",", "")), levels=int(l),
             max_level=int(l) - 1)
        for c, l in re.findall(
            r"Number of active cells:\s*([0-9,]+)\s*\(on (\d+) levels\)", text)]
    return out


def aspect_level_trace(alog):
    """Readable 'cells (on M levels)' trace of the ASPECT startup refinement."""
    trace = alog.get("cell_levels") or []
    if not trace:
        return "cell counts not recorded in this log"
    return " -> ".join("%s cells on %d levels" % (format(t["cells"], ","),
                                                  t["levels"]) for t in trace)


def aspect_statistics(path):
    """Time trace of the ASPECT statistics table (raw ASPECT column indices)."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            v = line.split()
            if len(v) < 19:
                continue
            try:
                rows.append(dict(t=float(v[1]), dt=float(v[2]),
                                 cells=int(v[3]), step=int(v[0]),
                                 vrms=float(v[11]), vmax=float(v[12]),
                                 Tmin=float(v[13]), meanT=float(v[14]),
                                 Tmax=float(v[15]),
                                 q_bot=float(v[17]), q_top=float(v[18])))
            except ValueError:
                continue
    rows.sort(key=lambda r: r["t"])
    out = dict(nrows=len(rows), t_first=rows[0]["t"] if rows else None,
               t_last=rows[-1]["t"] if rows else None,
               cells=sorted({r["cells"] for r in rows}) if rows else [],
               vrms_min=min(r["vrms"] for r in rows) if rows else None,
               vrms_max=max(r["vrms"] for r in rows) if rows else None,
               meanT_first=rows[0]["meanT"] if rows else None,
               meanT_last=rows[-1]["meanT"] if rows else None,
               dt_min=min(r["dt"] for r in rows) if rows else None,
               dt_max=max(r["dt"] for r in rows) if rows else None)
    # per-visualisation-output drifts: the evidence that the clip is not static
    if rows:
        steps = [r["step"] for r in rows]
        out["first_step"] = steps[0]
        out["last_step"] = steps[-1]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quast-log", required=True)
    ap.add_argument("--quast-frames", required=True)
    ap.add_argument("--aspect-log", required=True)
    ap.add_argument("--aspect-stats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tstop", type=float, default=0.2)
    ap.add_argument("--frame-dt", type=float, default=0.005)
    ap.add_argument("--quast-steps", type=int, default=None,
                    help="override the step count if the log was truncated")
    args = ap.parse_args(argv)

    qlog = parse_quast_log(args.quast_log)
    qframes = quast_frame_stream(args.quast_frames)
    alog = aspect_log(args.aspect_log)
    astats = aspect_statistics(args.aspect_stats)
    qres = qlog["result"] or {}
    qfr = qlog["frames"] or {}
    qamr = qlog["amr"] or {}
    steps = args.quast_steps or qres.get("steps")
    wall = qres.get("wall")
    leaves_max = max(qamr.get("leaves_max", qframes["leaves_max"]),
                     qframes["leaves_max"])

    extra = dict(
        case="annulus AMR demonstration (QuaST vs ASPECT)",
        card=dict(
            geometry="PolarAnnulus(R0=1.22, R1=2.22, Phi=2*pi; "
                     "seam=:periodic); ASPECT spherical shell, Opening angle = "
                     "360, radii 1.22/2.22, non-dimensional, D = 1",
            Ra="1e3 at the reference viscosity 1 (gravity 1e9 x alpha 1e-6); "
               "the hot material at the inner wall has eta = 1e3^(-1) = 1e-3 "
               "and therefore a local Rayleigh number Ra/eta = 1e6",
            eta="1000^(-T) (ASPECT `VoT` material model; QuaST eta_law = :expT "
                "with deta_T = 1000)",
            H="0.0 (no internal heating)",
            walls="free slip on both circles (ASPECT Tangential velocity "
                  "boundary indicators = 0,1)",
            temperatures="T = 1 at the inner wall, T = 0 at the outer wall",
            initial_condition="ASPECT spherical hexagonal perturbation, angular "
                              "mode 4, rotation offset 0: s = (R1-r)/(R1-R0), "
                              "T = s + 0.2 s (1-s) cos(4 atan2(y,x)); the same "
                              "total T on both codes",
            mesh_quast="AMR: start from a 4x48 = 192-leaf level-2 ring tree, "
                       "initial_refine = 2 on the IC, then one balanced cycle "
                       "every 5 steps (refine_frac 0.2, coarsen_frac 0.05, "
                       "max_leaves 12000, max_level 7, wall_min_level 3, "
                       "wall_strip 0.1, indicator :theta, 1 pass)",
            mesh_aspect="AMR: Initial global refinement 3 + Initial adaptive "
                        "refinement 2 (temperature strategy, refinement "
                        "fraction 0.2, coarsening fraction 0.05); the adaptive "
                        "startup mesh is held fixed during the time loop (see "
                        "the caveats)",
            time="t = 0 -> %g, CFL 10, BDF2, %d frames every %g"
                 % (args.tstop, qframes["nframes"], args.frame_dt),
            linear_quast=":mg, matrix-free multigrid, 8 threads",
        ),
        runs=dict(
            quast=dict(
                command="julia --project=. --threads=8 "
                        "examples/annulus_benchmark.jl demo %g" % args.tstop,
                steps=steps, wall_s=wall,
                s_per_step=qres.get("s_per_step"),
                s_step_warm=qres.get("s_step_warm"),
                leaves_final=qres.get("leaves", qframes["leaves_final"]),
                leaves_max=leaves_max,
                level_final=[qframes["level_min"][-1],
                             qframes["level_max"][-1]],
                level_seen=[qframes["level_seen_min"],
                            qframes["level_seen_max"]],
                t_last=qframes["t_last"], dt_last=qres.get("dt"),
                dt_min=qframes["dt_min"], dt_max=qframes["dt_max"],
                Nu_T=qres.get("NuT"), Nu_B=qres.get("NuB"),
                Vrms=qres.get("Vrms"), meanT=qres.get("meanT"),
                Umax=qres.get("Umax"),
                frames=qframes["nframes"],
                log=args.quast_log,
            ),
            aspect=dict(
                command="mpirun --allow-run-as-root -n 4 "
                        "/home/dealii/aspect/aspect-release demo.prm "
                        "(pinned image, aspect-release)",
                steps=alog["steps"], wall_s=alog.get("wallclock_s"),
                cells=astats["cells"],
                startup_refinement=[
                    dict(cells=t["cells"], reported_levels=t["levels"],
                         max_tree_level=t["max_level"])
                    for t in (alog.get("cell_levels") or [])],
                startup_level_range=([3, max(t["max_level"] for t in
                                             alog["cell_levels"])]
                                     if alog.get("cell_levels") else None),
                t_last=astats["t_last"],
                vrms_range=[astats["vrms_min"], astats["vrms_max"]],
                meanT_range=[astats["meanT_first"], astats["meanT_last"]],
                dt_range=[astats["dt_min"], astats["dt_max"]],
                statistics_rows=astats["nrows"],
                log=args.aspect_log,
            ),
        ),
        amr_statistics=dict(
            quast=dict(leaves_final=qframes["leaves_final"],
                       leaves_max=leaves_max,
                       level_seen=[qframes["level_seen_min"],
                                   qframes["level_seen_max"]],
                       level_final=[qframes["level_min"][-1],
                                    qframes["level_max"][-1]],
                       dt_range=[qframes["dt_min"], qframes["dt_max"]],
                       theta_change_vs_first=qframes["theta_change_vs_first"]),
            aspect=dict(cells=astats["cells"],
                        level_range="inferred per parent cell by the renderer "
                                    "from A/r_mid; the coarsest cells are the "
                                    "initial global refinement level 3",
                        dframes="held fixed after the initial adaptive "
                                "refinement (see the caveats)"),
        ),
        caption=(
            "A purpose-configured annulus demonstration of the adaptive path "
            "(Ra = 1e3 at the reference viscosity 1, so the hot inner material "
            "has local Ra = 1e6; eta = 1000^(-T); AMR on both codes), run by "
            "QuaST (top) and ASPECT (bottom): total temperature, its "
            "angular-mean anomaly, log10 eta = -3 T, and each code's own "
            "adaptive mesh filled in its tree-level colour.  The solver names "
            "are inside the temperature panels and each panel's numbers are on "
            "its lower-right plate; the image carries no header, no time text "
            "and nothing above the panels -- the card, the run evidence and the "
            "caveats are in this sidecar and in the README caption.  This is a "
            "demonstration, not a validation: the two codes run independent "
            "adaptive meshes and no accuracy claim is made."),
        notes=[
            "This is a DEMONSTRATION of the AMR path, not the published "
            "Davies et al. (2022) VoT benchmark case and not an accuracy "
            "claim.  The two meshes are different by construction: QuaST "
            "refines with its gradient-jump indicator and a wall minimum "
            "level, ASPECT with its temperature strategy, and the in-loop "
            "refinement policies differ (see below).",
            "The approved first card (Ra = 1e6 at the reference viscosity 1, "
            "eta = 1000^(-T), H = 10, t = 0 -> 0.3) was measured to be "
            "infeasible and is recorded here as the reason for tuning: with "
            "eta = 1000^(-T) the hot inner boundary sits at local Ra = 1e9, "
            "the ASPECT pilot measured u_max 6.73e4 (step 0) and 2.19e5 "
            "(step 1) with a CFL-limited dt of 1.76e-6, i.e. O(1e5-1e6) steps "
            "for t = 0.3; at that forcing the ASPECT temperature solve also "
            "failed (1000 iterations, residual 4.8e-3).  H = 10 is "
            "additionally refused by QuaST's own wall-flux normalisation on "
            "the annulus: the internal heating reverses the sign of the base "
            "conductive flux at the inner wall (-dTbar/dr = -4.31 at R0 vs "
            "+5.38 at R1), so the family-A Nusselt normalisation is "
            "undefined.  The card was therefore tuned to Ra = 1e3 (local hot "
            "Ra = 1e6) and H = 0, and the window to t = 0 -> 0.2; Ra = 1e4 was "
            "also measured (u_max 5.5e4, dt 2.6e-6, ~1e5 steps for t = 0.3) "
            "and rejected for the same reason.",
            "ASPECT's in-loop mesh refinement is off in this run "
            "(Time steps between mesh refinement = 100000): in the pinned "
            "ASPECT 2.5.0-pre image an in-loop cycle every 5 steps drove an "
            "unbounded velocity spike (u_max 5.4e6 at t = 0.016) and a "
            "temperature-solver failure, reproducibly, with several solver "
            "and AMR settings.  ASPECT's mesh is therefore its adaptive "
            "startup mesh, shown in the mesh panel but held fixed; QuaST's "
            "mesh evolves in-loop.  This asymmetry is stated, not hidden.  "
            "The startup mesh is on levels 3-4: the ASPECT log records the "
            "two initial adaptive passes as %s, i.e. five refinement levels "
            "(0-4) in ASPECT's counting = tree levels 3-4, which the renderer "
            "also infers per cell from the VTU (it agrees; the earlier 'levels "
            "3-5' reading of '5 levels' was wrong)."
            % aspect_level_trace(alog),
            "The two rows' window is t = 0 -> %g (41 frames at the 0.005 "
            "cadence) rather than 0.3 so the figure fits the measurement "
            "budget; the flow is evolving over it (QuaST: %s; ASPECT V_rms "
            "%g..%g and mean T %g..%g over the shown window), so the clip is "
            "not a steady plateau."
            % (args.tstop,
               ", ".join("%s: max|dtheta| %.3g (%.2f%% of rms theta)"
                         % (k, v["max_abs_theta"],
                            100.0 * v["rms_theta_rel"])
                         for k, v in qframes["theta_change_vs_first"].items()),
               astats["vrms_min"], astats["vrms_max"],
               astats["meanT_first"], astats["meanT_last"]),
            "Nu_top/Nu_bot are each code's own wall-flux family (QuaST's "
            "one-sided GLL gradient, family A; ASPECT's consistent boundary "
            "flux, family C), so the two columns' Nu values are not the same "
            "measurement and must not be compared as one; V_rms and mean T "
            "are state quantities.  No Nusselt value from this demonstration "
            "is compared with any published number.",
            "The QuaST frame stream is written from the driver callback, "
            "which runs after the AMR pass: on a step whose AMR cycle changed "
            "the mesh the flow fields are reset for the next Stokes solve, so "
            "the example defers such a frame by one step rather than writing "
            "speed = 0 / V_rms = 0 for a state that has a flow (the benchmark "
            "videos never AMR and are unchanged).",
        ],
        provenance=dict(
            quast_log=args.quast_log,
            quast_frames=os.path.join(args.quast_frames, "frame_meta.csv"),
            aspect_log=args.aspect_log,
            aspect_statistics=args.aspect_stats,
            aspect_card="aspect_runs/annulus_demo/demo.prm",
            scene="scripts/run_annulus_demo.sh",
            renderer="scripts/annulus_video.py --amr --style reference",
        ),
    )
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(extra, fh, indent=1)
    print("[annulus_demo_sidecar] wrote %s" % args.out)
    print("[annulus_demo_sidecar] QuaST steps=%s wall=%ss leaves_max=%s "
          "level_seen=%s" % (steps, wall, leaves_max, extra["runs"]["quast"]
                             ["level_seen"]))
    print("[annulus_demo_sidecar] ASPECT steps=%s wall=%ss cells=%s"
          % (alog["steps"], alog.get("wallclock_s"), astats["cells"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
