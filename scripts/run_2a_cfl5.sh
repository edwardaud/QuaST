#!/usr/bin/env bash
# 2a the way the benchmark cards do it: CFL 5, wall-refined level 3, run to the
# end time instead of stopping on the per-step criterion.
set -u
cd /mnt/e/研究生资料/project/QuaST || exit 1
tmux kill-session -t b2acf5 2>/dev/null || true
tmux new-session -d -s b2acf5 \
  "CASE_TAG=cf5 CASE_WALLMIN=4 CASE_INITREF=1 CASE_CFL=5.0 CASE_STEADY=Inf bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh 2a 3 0.2 4; sleep 600"
sleep 3
tmux ls
