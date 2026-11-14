#!/usr/bin/env bash
# 2a level 4 with the wall rule on: walls at level 5, interior at level 4.
set -u
cd /mnt/e/研究生资料/project/QuaST || exit 1
tmux kill-session -t b2aL4 2>/dev/null || true
tmux new-session -d -s b2aL4w \
  "CASE_WALLMIN=5 CASE_INITREF=1 bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh 2a 4 0.2 4; sleep 600"
sleep 3
tmux ls
