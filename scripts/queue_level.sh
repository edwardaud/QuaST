#!/usr/bin/env bash
# Queue a finer-level run behind the one already running for the same case.
#
#     bash scripts/queue_level.sh <case> <level> <wait_for_level> [tstop] [threads]
#
# It polls the coarser log until the run has written its final "[case] steps="
# line, then starts the finer level in the same session.
set -u
case=${1:?case}
level=${2:?level}
wait_level=${3:?wait level}
tstop=${4:-0.2}
threads=${5:-4}

# QUAST_ROOT / QUAST_JULIA / QUAST_OUT: repository root, Julia executable, run
# output root (defaults: this script's repository, `julia` from PATH,
# out_julia/ under the root). scripts/README.md documents them.
QUAST_ROOT=${QUAST_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}
QUAST_JULIA=${QUAST_JULIA:-julia}
QUAST_OUT=${QUAST_OUT:-$QUAST_ROOT/out_julia}
cd -- "$QUAST_ROOT" || exit 1
wait_log="$QUAST_OUT/benchmarks/${case}_L${wait_level}.log"

while ! grep -q "^\[${case}\] steps=" "$wait_log" 2>/dev/null; do
  sleep 30
done

bash "$QUAST_ROOT/scripts/run_one.sh" "$case" "$level" "$tstop" "$threads"
