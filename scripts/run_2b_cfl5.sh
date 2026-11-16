#!/usr/bin/env bash
# 2b the way ASPECT runs it: CFL 5, run to t = 1, no adaptive refinement during
# the run.  The steady state does not depend on the step size, so the small CFL
# the driver defaults to bought nothing but wall time.
set -u
cd /mnt/e/研究生资料/project/QuaST || exit 1
tmux kill-session -t b2bcf5 2>/dev/null || true
tmux kill-session -t b2bdeep2 2>/dev/null || true
tmux new-session -d -s b2bcf5 \
  "CASE_TAG=cf5 CASE_WALLMIN=5 CASE_INITREF=1 CASE_CFL=5.0 CASE_STEADY=Inf bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh 2b 4 1.0 4; sleep 600"
tmux new-session -d -s b2bl3cf5 \
  "CASE_TAG=cf5 CASE_WALLMIN=4 CASE_INITREF=1 CASE_CFL=5.0 CASE_STEADY=Inf bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh 2b 3 1.0 3; sleep 600"
sleep 3
tmux ls
