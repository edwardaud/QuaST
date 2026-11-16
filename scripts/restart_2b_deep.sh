#!/usr/bin/env bash
# Stop 2a and 2b, then restart 2b with the wall configuration that actually
# reaches a percent-level Nu: interior level 4, walls at level 5.
set -u
cd /mnt/e/研究生资料/project/QuaST || exit 1

tmux kill-session -t l2a 2>/dev/null || true
tmux kill-session -t l2b 2>/dev/null || true
pkill -f 'benchmark_cases.jl 2a 3' 2>/dev/null || true
pkill -f 'benchmark_cases.jl 2b 3' 2>/dev/null || true
sleep 3

tmux new-session -d -s b2bdeep \
  "CASE_TAG=wall5 CASE_WALLMIN=5 CASE_INITREF=1 CASE_STEADY=Inf bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh 2b 4 0.2 4; sleep 600"

sleep 3
tmux ls
free -g | sed -n 2p
