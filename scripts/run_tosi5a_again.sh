#!/usr/bin/env bash
# Re-run Tosi case 5a with the fixed cycle analysis and the trajectory saved.
set -u
cd /mnt/e/研究生资料/project/QuaST || exit 1
mkdir -p out_julia/benchmarks/ckpt_tosi5a
tmux kill-session -t t5c 2>/dev/null || true
tmux new-session -d -s t5c \
  "CASE_TAG=ra1000b CASE_RA=1000 CASE_STEADY=Inf CASE_CKPT_DIR=/mnt/e/研究生资料/project/QuaST/out_julia/benchmarks/ckpt_tosi5a bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh tosi5a 3 1.0 3; sleep 600"
sleep 3
tmux ls
