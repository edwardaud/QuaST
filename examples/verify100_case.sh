#!/usr/bin/env bash
# verify100_case.sh -- launch one arm of the three-arm verification run.
#
# All arms start FROM SCRATCH at t = 0 from the card's own initial condition
# (NOT from the G resume checkpoint), on the production G card:
#   box 2.5x1, composite isostress rheology, two phase transitions with
#   per-phase latent heat, H = 10, Ra = 1e5, free slip, amr=balanced,
#   amr_coarsen_gate=1, symmetry=none, 10000 leaves, wall rule 5/0.1.
# 500 steps at the card's frame stride 5 write 100 cadence frames
# (steps 1, 6, ..., 496) plus the final-state frame at step 500.
#
#   bash examples/verify100_case.sh mg      [steps]  -> out_julia/verify100_mg
#   bash examples/verify100_case.sh direct  [steps]  -> out_julia/verify100_direct
#
# The log is out_julia/verify100_<arm>.log.  `CASE_OUT` is always set
# explicitly (docs/dev/CASE_REALISTIC.md section 5), and the arm refuses to reuse
# an existing output directory.  `VERIFY100_FOREGROUND=1` runs Julia in the
# foreground (used under a supervisor that keeps the WSL session alive);
# otherwise the process is detached with setsid.
set -euo pipefail

ARM=${1:?usage: verify100_case.sh <mg|direct|auto> [steps]}
STEPS=${2:-500}
case "$ARM" in
    mg|direct|auto) ;;
    *) echo "verify100_case.sh: unknown arm '$ARM' (mg|direct|auto)" >&2; exit 2 ;;
esac

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
JL=${JL:-/home/szh/opt/julia-1.12.7/bin/julia}
RUN_ID=${RUN_ID:-verify100}
OUT="out_julia/${RUN_ID}_$ARM"
LOG="out_julia/${RUN_ID}_$ARM.log"

if [ -e "$OUT" ]; then
    echo "verify100_case.sh: refusing, $OUT already exists" >&2
    echo "  (CASE_OUT is always explicit: docs/dev/CASE_REALISTIC.md section 5)" >&2
    exit 3
fi
if pgrep -f "case_realistic_2d.jl spinup" >/dev/null 2>&1; then
    echo "verify100_case.sh: refusing, a case_realistic_2d spinup is already running" >&2
    exit 4
fi

# The production G card.  CASE_STOKES_SCALING=equilibrated is the documented
# solver-side saddle equilibration (docs/dev/BUG_LEDGER.md); it only
# engages on a direct solve whose unscaled health check fails, so it is a
# no-op on the solves that pass and is set for both arms so that the only
# difference between them is CASE_LINEAR.  Every AMR knob is written out
# explicitly, including the ones whose card defaults already equal the
# production value (CASE_HYST=1, CASE_EV=0.11), so the direct and MG arms
# cannot drift apart through a changed default.
export CASE_OUT="$OUT" CASE_STEPS="$STEPS" CASE_MAXLEAVES=10000 \
    CASE_COARSEN=0.05 CASE_LINEAR="$ARM" \
    CASE_LATENT=per_phase CASE_MAXLEV=7 CASE_WALLMIN=5 CASE_WALLSTRIP=0.1 \
    CASE_HYST=1 CASE_AMR_GATE=1 CASE_SYMMETRY=none \
    CASE_EV=0.11 CASE_STOKES_SCALING=equilibrated CASE_FRAME="${CASE_FRAME:-0}"

JL_ARGS=(--project=. --threads=8 examples/case_realistic_2d.jl spinup)

if [ "${VERIFY100_FOREGROUND:-0}" = "1" ]; then
    exec "$JL" "${JL_ARGS[@]}" > "$LOG" 2>&1
fi

setsid nohup "$JL" "${JL_ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
echo "launched $ARM arm: pid=$! CASE_OUT=$OUT steps=$STEPS log=$LOG"
