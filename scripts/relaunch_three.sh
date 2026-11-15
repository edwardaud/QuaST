#!/usr/bin/env bash
# Three things at once:
#   1. 2a and 2b with interior level 3 and walls at level 4, the cheaper wall
#      refinement that has a chance of reaching a steady state.
#   2. Tosi case 1 re-run to t = 1 with the steady criterion off, because its
#      per-step change falls below 1e-6 long before the state is steady.
#   3. Nothing for Tosi 5a yet: it waits for case 1 to calibrate Ra.
set -u
cd /mnt/e/研究生资料/project/QuaST || exit 1

for s in w2a w2b t1; do tmux kill-session -t "$s" 2>/dev/null || true; done
sleep 2

tmux new-session -d -s w3a \
  "CASE_TAG=wall3 CASE_WALLMIN=4 CASE_INITREF=1 bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh 2a 3 0.2 4; sleep 600"
tmux new-session -d -s w3b \
  "CASE_TAG=wall3 CASE_WALLMIN=4 CASE_INITREF=1 bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh 2b 3 0.2 3; sleep 600"
tmux new-session -d -s t1b \
  "CASE_TAG=case1 CASE_STEADY=Inf CASE_CKPT_DIR=/mnt/e/研究生资料/project/QuaST/out_julia/benchmarks/ckpt_tosi1 bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh tosi1 3 1.0 3; sleep 600"

sleep 3
tmux ls
