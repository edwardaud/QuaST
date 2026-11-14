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

cd /mnt/e/研究生资料/project/QuaST || exit 1

tmux kill-server 2>/dev/null || true
sleep 1

for case in 1a 1b 1c 2a 2b; do
  tmux new-session -d -s "b$case" \
    "bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh $case $level $tstop $threads; sleep 600"
  echo "started b$case ($case level $level, $threads threads)"
done

sleep 5
tmux ls
