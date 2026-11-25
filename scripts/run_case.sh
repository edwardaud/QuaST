#!/usr/bin/env bash
# Start one case at a given level in its own tmux session.
#
#     bash scripts/run_case.sh <case> <level> [tstop] [threads]
set -u
case=${1:?case}
level=${2:?level}
tstop=${3:-0.2}
threads=${4:-3}

# QUAST_ROOT / QUAST_JULIA / QUAST_OUT: repository root, Julia executable, run
# output root (defaults: this script's repository, `julia` from PATH,
# out_julia/ under the root). scripts/README.md documents them.
QUAST_ROOT=${QUAST_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}
QUAST_JULIA=${QUAST_JULIA:-julia}
QUAST_OUT=${QUAST_OUT:-$QUAST_ROOT/out_julia}
cd -- "$QUAST_ROOT" || exit 1
tmux new-session -d -s "b${case}L${level}" \
  "QUAST_ROOT='$QUAST_ROOT' QUAST_JULIA='$QUAST_JULIA' QUAST_OUT='$QUAST_OUT' \
   bash '$QUAST_ROOT/scripts/run_one.sh' $case $level $tstop $threads; sleep 600"
echo "started b${case}L${level}"
