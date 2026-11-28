#!/usr/bin/env bash
# run_annulus_realistic_demo.sh -- reproduce docs/assets/quast_vs_aspect_annulus_demo.gif
#
# The clip is the realistic 2.5 x 1 mantle card on the closed 2-D annulus
# (H = 10, nominal Ra = 1e5 at the composite calibration point, composite
# isostress rheology, two phase transitions with per-phase latent heat, free
# slip), run by QuaST and by ASPECT on their own adaptive meshes for
# t_nd = 1e-6 .. 1.2e-4.  QuaST runs the direct solve (its matrix-free MG arm
# does not reach the box card's 1e-3 true-residual guard on this ring; see the
# guided guide section 1.10 and the pilot record), ASPECT runs the original box
# clip's seven-line AMR block verbatim on a spherical shell.
#
# The stages:
#
#   quast    the QuaST leg (examples/annulus_realistic_demo.jl; production
#            theta indicator, refine_frac 0.2, 30k leaves, cap-aware max-level
#            fallback on by default, direct; ~6.6 h on 8 threads to
#            t_nd = 1.2e-4; HDF5 checkpoint every 100 steps, ANN_REAL_RESTART
#            resumes).  The fallback removes the measured 28-frame theta/0.2
#            stall at the level cap; the solver keyword itself is opt-in.
#   aspect   the ASPECT leg: docker is on the Windows host, so this script
#            *prints* the two commands (plugin build, production run) instead of
#            running them from WSL; see aspect_runs/annulus_realistic_demo/README.md
#   render   the figure from the two finished run directories
#
#   bash scripts/run_annulus_realistic_demo.sh quast
#   bash scripts/run_annulus_realistic_demo.sh aspect      # prints the commands
#   bash scripts/run_annulus_realistic_demo.sh render
#
# Environment (all overridable):
#   QUAST_ROOT      repository root        (default: the parent of this script)
#   QUAST_JULIA     julia executable       (default: julia on PATH)
#   QUAST_OUT       run-output root        (default: <root>/out_julia/annulus_realistic)
#   QUAST_THREADS   Julia threads          (default: 8)
#   ASPECT_OUT      the finished ASPECT output directory, i.e. the Windows mount
#                   that holds output-annulus-realistic-prod (default:
#                   <QUAST_OUT>/aspect_prod; the docker run below mounts the
#                   Windows E: path, so pass it explicitly for a real run)
#   PY              the Python with numpy/matplotlib/scipy/meshio
#
# The tracked figure was produced with the settings below; the QuaST leg is
# deterministic and the ASPECT leg is deterministic for a fixed MPI rank count
# (the recorded run used 8 ranks).  Render 120 frames over the window; the GIF
# encoder uses --gif-dither none (the dense mesh panels compress ~2x better
# without dithering and the difference is not visible at 1200 px).
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
JULIA=${QUAST_JULIA:-julia}
OUT=${QUAST_OUT:-$ROOT/out_julia/annulus_realistic}
QUAST_DIR=${QUAST_DIR:-$OUT/quast_theta_cap30}
ASPECT_DIR=${ASPECT_OUT:-$OUT/aspect_prod}
PY=${PY:-/home/szh/dedalus-env/bin/python}
STAGE=${1:-all}

run_quast() {
    mkdir -p "$OUT"
    if [ -e "$QUAST_DIR" ] && [ -z "${ANN_REAL_RESTART:-}" ]; then
        echo "run_annulus_realistic_demo.sh: refusing, $QUAST_DIR exists (use ANN_REAL_RESTART to resume)" >&2
        exit 3
    fi
    echo "== QuaST leg: $QUAST_DIR"
    ANN_REAL_OUT=$QUAST_DIR \
    ANN_REAL_STEPS=60000 \
    ANN_REAL_TSTOP=1.2e-4 \
    ANN_REAL_FRAME=5 \
    ANN_REAL_REPORT=50 \
    ANN_REAL_LINEAR=direct \
    ANN_REAL_MAXLEAVES=30000 \
    ANN_REAL_INDICATOR=theta \
    ANN_REAL_REFINE_FRAC=0.2 \
    ANN_REAL_CAP_AWARE=1 \
    ANN_REAL_CHECKPOINT=$OUT/quast_theta_cap30.h5 \
    ANN_REAL_CHECKPOINT_EVERY=100 \
    "$JULIA" --project="$ROOT" --threads="${QUAST_THREADS:-8}" \
        "$ROOT/examples/annulus_realistic_demo.jl" | tee "$OUT/quast_prod.log"
}

print_aspect() {
    cat <<'EOF'
== ASPECT leg (run from Windows PowerShell; see aspect_runs/annulus_realistic_demo/README.md)

The image is pinned; the plugin is built once from the tracked source and the
card is aspect_runs/annulus_realistic_demo/annulus_realistic_prod.prm.  The
recorded run used 8 MPI ranks and wrote its output (~3 GB) to the Windows E:
mount; the second mount below is where the renderer reads it from WSL.

# 1) stage the tracked plugin sources at an ASCII host path, build both variants
$img = "gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac"
docker run --rm --entrypoint /bin/bash -v C:/temp/quast_aspect/realistic_annulus:/work -w /work $img `
  -c "mkdir -p build && cd build && cmake -DAspect_DIR=/home/dealii/aspect/build .. && make -j8"

# 2) run the production card (detached container so it survives the shell)
docker run -d --name quast_aspect_prod --entrypoint /bin/bash `
  -v C:/temp/quast_aspect/realistic_annulus:/work `
  -v E:/temp/quast_aspect_prod:/out -w /out $img `
  -c 'start=$(date +%s); mpirun --allow-run-as-root -n 8 /home/dealii/aspect/aspect-release /work/annulus_realistic_prod.prm > /out/logs/prod.log 2>&1; rc=$?; end=$(date +%s); echo "exit=$rc wall=$((end-start))s" >> /out/logs/prod.log'

# 3) the renderer reads /out/output-annulus-realistic-prod (from WSL:
#    /mnt/e/temp/quast_aspect_prod/output-annulus-realistic-prod)
EOF
}

run_render() {
    if [ ! -d "$ASPECT_DIR" ]; then
        echo "run_annulus_realistic_demo.sh: ASPECT output not found at $ASPECT_DIR" >&2
        echo "  set ASPECT_OUT to the finished ASPECT run directory" >&2
        exit 4
    fi
    mkdir -p "$OUT/render"
    echo "== render: $OUT/render/theta_cap30.gif"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" \
        "$ROOT/scripts/annulus_realistic_video.py" \
        --quast "$QUAST_DIR" --aspect "$ASPECT_DIR" \
        --quast-log "$OUT/quast_theta_cap30.log" \
        --t0 1e-6 --t1 1.2e-4 --frames 120 --fps 10 \
        --master-scale 4 --polar-nr 800 --polar-nphi 4096 \
        --mesh-supersample 2 --mesh-min-cell-px 1.5 \
        --font-scale 1.25 --plate-alpha 0.75 --gif-dither none \
        --out "$OUT/render/theta_cap30.gif" \
        --also-poster --also-mp4 \
        --sidecar "$OUT/render/theta_cap30.json"
}

case "$STAGE" in
    quast)  run_quast ;;
    aspect) print_aspect ;;
    render) run_render ;;
    all)    print_aspect; run_render ;;
    *) echo "usage: $0 [quast|aspect|render|all]" >&2; exit 2 ;;
esac
