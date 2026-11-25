#!/usr/bin/env bash
# Run the five Blankenbach cases, one tmux session each, at the level given.
#
#     bash scripts/run_parallel.sh [level] [tstop] [threads]
#
# Every case starts from ASPECT's own ascii initial condition, so a run needs
# tens of steps rather than tens of thousands.
set -u
level=${1:-3}
tstop=${2:-0.2}
threads=${3:-3}

# QUAST_ROOT / QUAST_JULIA / QUAST_OUT: repository root, Julia executable, run
# output root (defaults: this script's repository, `julia` from PATH,
# out_julia/ under the root). scripts/README.md documents them.
QUAST_ROOT=${QUAST_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}
QUAST_JULIA=${QUAST_JULIA:-julia}
QUAST_OUT=${QUAST_OUT:-$QUAST_ROOT/out_julia}
cd -- "$QUAST_ROOT" || exit 1

tmux kill-server 2>/dev/null || true
sleep 1

for case in 1a 1b 1c 2a 2b; do
  tmux new-session -d -s "b$case" \
    "QUAST_ROOT='$QUAST_ROOT' QUAST_JULIA='$QUAST_JULIA' QUAST_OUT='$QUAST_OUT' \
     bash '$QUAST_ROOT/scripts/run_one.sh' $case $level $tstop $threads; sleep 600"
  echo "started b$case ($case level $level, $threads threads)"
done

sleep 5
tmux ls
