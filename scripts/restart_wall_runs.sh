#!/usr/bin/env bash
# Restart the two wall-refined runs with the fixed diagnostics: the old callback
# mixed the pre-adaptation space with the post-adaptation tree, so its Nu was
# meaningless even though the step records were right.
set -u
cd /mnt/e/研究生资料/project/QuaST || exit 1

tmux kill-session -t w2a 2>/dev/null || true
tmux kill-session -t w2b 2>/dev/null || true
pkill -f 'benchmark_cases.jl 2a 4' 2>/dev/null || true
pkill -f 'benchmark_cases.jl 2b 4' 2>/dev/null || true
sleep 3

tmux new-session -d -s w2a \
  "CASE_TAG=wall CASE_WALLMIN=5 CASE_INITREF=1 bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh 2a 4 0.2 4; sleep 600"
tmux new-session -d -s w2b \
  "CASE_TAG=wall CASE_WALLMIN=5 CASE_INITREF=1 bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh 2b 4 0.2 3; sleep 600"

sleep 3
tmux ls
