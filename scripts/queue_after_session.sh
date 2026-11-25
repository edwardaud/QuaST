#!/usr/bin/env bash
# Start a case once another tmux session has finished.
#
#     bash scripts/queue_after_session.sh <target session> <case> <level> [tstop] [threads]
set -u
target=${1:?target session}
case=${2:?case}
level=${3:?level}
tstop=${4:-0.2}
threads=${5:-3}

# QUAST_ROOT / QUAST_JULIA / QUAST_OUT: repository root, Julia executable, run
# output root (defaults: this script's repository, `julia` from PATH,
# out_julia/ under the root). scripts/README.md documents them.
QUAST_ROOT=${QUAST_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}
QUAST_JULIA=${QUAST_JULIA:-julia}
QUAST_OUT=${QUAST_OUT:-$QUAST_ROOT/out_julia}
cd -- "$QUAST_ROOT" || exit 1
while tmux has-session -t "$target" 2>/dev/null; do
  sleep 30
done
bash "$QUAST_ROOT/scripts/run_one.sh" "$case" "$level" "$tstop" "$threads"
