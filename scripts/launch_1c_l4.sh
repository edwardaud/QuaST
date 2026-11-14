#!/usr/bin/env bash
# Start the level-4 run for 1c now: its level-3 error is the largest of the
# family (7.8 percent), so it needs the refinement most.
set -u
cd /mnt/e/研究生资料/project/QuaST || exit 1
tmux new-session -d -s "b1cL4" \
  "bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh 1c 4 0.2 4; sleep 600"
sleep 2
tmux ls
