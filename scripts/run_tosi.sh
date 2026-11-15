#!/usr/bin/env bash
# The two Tosi benchmarks, now that the analytic initial condition is passed as
# an anomaly.  Both start from the paper's T = (1-z) + 0.01 cos(pi x) sin(pi z)
# on a 32 x 32 mesh, which is ASPECT's refinement 5, and run to t = 1.
set -u
cd /mnt/e/研究生资料/project/QuaST || exit 1

tmux kill-session -t t1 2>/dev/null || true
tmux kill-session -t t5a 2>/dev/null || true

tmux new-session -d -s t1 \
  "CASE_TAG=case1 bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh tosi1 3 1.0 2; sleep 600"
tmux new-session -d -s t5a \
  "CASE_TAG=case5a bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh tosi5a 3 1.0 2; sleep 600"

sleep 3
tmux ls
