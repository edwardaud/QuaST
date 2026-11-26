#!/usr/bin/env bash
# Reproduce the annulus AMR demonstration run (the QuaST side) and the figure.
#
#     bash scripts/run_annulus_demo.sh
#
# The card is the `demo` preset of examples/annulus_benchmark.jl (see its header)
# with the tuned physics below: Ra = 1e3 at the reference viscosity 1,
# eta = 1000^(-T), H = 0 (no internal heating), the ASPECT mode-4 initial
# condition, free-slip periodic ring, t = 0 -> 0.2, CFL 10, BDF2, 41 frames
# every 0.005, and AMR on: initial_refine = 2 on the IC, then one balanced cycle
# every 5 steps (refine_frac 0.2, coarsen_frac 0.05, max_leaves 12000,
# max_level 7, wall_min_level 3, wall_strip 0.1, indicator :theta),
# linear = :mg (matrix-free multigrid, 8 threads).
#
# Why the card departs from the first proposal (all measured, see the campaign
# record docs/dev/BUG_FIX_LOG.md and the figure's JSON sidecar):
#   * Ra = 1e6 with eta = 1000^(-T) (the first proposal) is infeasible: the hot
#     inner boundary has local Ra = 1e9, the CFL-limited dt is ~2e-6 (ASPECT
#     measured) and the run needs O(1e5-1e6) steps for t = 0.3;
#   * H = 10 is refused by QuaST's own wall-flux normalisation on the annulus:
#     the internal heating reverses the sign of the base conductive flux at the
#     inner wall (-dTbar/dr = -4.31 at R0 vs +5.38 at R1), so the family-A
#     Nusselt normalisation is undefined; H = 0 is used and documented;
#   * Ra = 1e3 is the largest Ra whose CFL-limited trajectory fits the budget
#     and stays visibly time-dependent (Ra = 1e4 was measured at u_max ~5e4 and
#     dt ~2.5e-6, i.e. ~1e5 steps for t = 0.3);
#   * t = 0.2 with a 0.005 frame cadence keeps 41 frames inside the time box.
#
# The ASPECT side is aspect_runs/annulus_demo/demo.prm and is run from Windows
# PowerShell with the pinned image (docker is not available inside WSL):
#
#   docker run --rm --entrypoint /bin/bash \
#     -v C:/temp/quast_aspect/annulus_demo:/work -w /work \
#     gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac \
#     -c "mpirun --allow-run-as-root -n 4 /home/dealii/aspect/aspect-release demo.prm"
#
# then copy output-annulus-demo/solution into OUT/aspect_frames/solution and
# the statistics file to OUT/aspect_frames/statistics.  See
# aspect_runs/annulus_demo/README.md for the exact commands and measured costs.
#
# Environment overrides: QUAST_ROOT, QUAST_JULIA, QUAST_OUT (as the other
# scripts in this directory), plus ANN_DEMO_TSTOP, ANN_DEMO_FRAME_DT,
# ANN_DEMO_MAX_LEAVES, ANN_DEMO_RA, ANN_DEMO_H, ANN_DEMO_DETA_T,
# QUAST_THREADS and no render (RENDER=0).
set -u
QUAST_ROOT=${QUAST_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}
QUAST_JULIA=${QUAST_JULIA:-julia}
QUAST_OUT=${QUAST_OUT:-$QUAST_ROOT/out_julia}
cd -- "$QUAST_ROOT" || exit 1

OUT="$QUAST_OUT/annulus_demo"
LOG="$OUT/logs"
mkdir -p "$LOG"
RA=${ANN_DEMO_RA:-1e3}
H=${ANN_DEMO_H:-0}
DETA=${ANN_DEMO_DETA_T:-1000}
TSTOP=${ANN_DEMO_TSTOP:-0.2}
FDT=${ANN_DEMO_FRAME_DT:-0.005}
MAXLEAVES=${ANN_DEMO_MAX_LEAVES:-12000}
THREADS=${QUAST_THREADS:-8}

echo "=== annulus demo QuaST leg: Ra=$RA H=$H deta_T=$DETA tstop=$TSTOP frame_dt=$FDT max_leaves=$MAXLEAVES"
ANN_RA="$RA" ANN_H="$H" ANN_DETA_T="$DETA" \
ANN_FRAMES="$OUT/quast_frames" ANN_FRAME_DT="$FDT" ANN_MESH_DUMP=1 \
ANN_CFL=10 ANN_MAX_LEAVES="$MAXLEAVES" ANN_MG_THREADS="$THREADS" ANN_REPORT=37 \
"$QUAST_JULIA" --project=. --threads="$THREADS" \
  examples/annulus_benchmark.jl demo "$TSTOP" > "$LOG/quast.log" 2>&1
echo "=== QuaST exit=$? log=$LOG/quast.log"
grep -E '^ANNULUS_(RESULT|FRAMES|AMR)' "$LOG/quast.log" | tail -3

if [ "${RENDER:-1}" = "1" ]; then
  echo "=== sidecar (reads the run log, the frame stream and the ASPECT outputs)"
  ${QUAST_PY:-python3} "$QUAST_ROOT/scripts/annulus_demo_sidecar.py" \
    --quast-log "$LOG/quast.log" \
    --quast-frames "$OUT/quast_frames" \
    --aspect-log "$LOG/aspect_full.log" \
    --aspect-stats "$OUT/aspect_frames/statistics" \
    --tstop "$TSTOP" --frame-dt "$FDT" \
    --out "$LOG/demo_sidecar_extra.json"
  echo "=== render (the ASPECT frames must already be in $OUT/aspect_frames)"
  ${QUAST_PY:-python3} "$QUAST_ROOT/scripts/annulus_video.py" \
    --row "QuaST=quast:$OUT/quast_frames" \
    --row "ASPECT=aspect:$OUT/aspect_frames" \
    --aspect-stats "$OUT/aspect_frames/statistics" \
    --amr --frame-dt "$FDT" --fps 10 \
    --case "annulus AMR demonstration" \
    --header "annulus AMR demonstration (not a benchmark)  |  Ra = 1e3 (eta_ref = 1, inner local 1e6), eta = 1000^(-T), H = 0  |  QuaST vs ASPECT" \
    --header2 "t = 0 .. $TSTOP, 41 frames @ 10 fps  |  theta = T - angular mean at r on a common 200 x 1024 polar grid  |  log10 eta = -3T from the same total T  |  mesh colour = tree level" \
    --sidecar-extra "$LOG/demo_sidecar_extra.json" \
    --out "$OUT/quast_vs_aspect_annulus_demo.mp4" \
    ${QUAST_FFMPEG:+--ffmpeg "$QUAST_FFMPEG"}
  echo "=== render exit=$?"
fi
