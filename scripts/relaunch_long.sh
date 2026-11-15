#!/usr/bin/env bash
# Three long runs, all with the premature steady-state stop disabled:
#   1. Tosi case 1 at Ra = 1000, the calibration run.  eta0 stays at 0.1, so
#      Ra/(eta/eta0) really is ten times what the paper's pairing gives.
#   2. 2a, interior 3 with walls at 4, to t = 0.3.
#   3. 2b, interior 3 with walls at 4, to t = 0.5.
set -u
cd /mnt/e/研究生资料/project/QuaST || exit 1

for s in t1b w3a w3b; do tmux kill-session -t "$s" 2>/dev/null || true; done
sleep 2

tmux new-session -d -s tra1000 \
  "CASE_TAG=ra1000 CASE_RA=1000 CASE_STEADY=Inf bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh tosi1 3 1.0 3; sleep 600"
tmux new-session -d -s l2a \
  "CASE_TAG=long3 CASE_WALLMIN=4 CASE_INITREF=1 CASE_STEADY=Inf bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh 2a 3 0.3 4; sleep 600"
tmux new-session -d -s l2b \
  "CASE_TAG=long3 CASE_WALLMIN=4 CASE_INITREF=1 CASE_STEADY=Inf bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh 2b 3 0.5 3; sleep 600"

sleep 3
tmux ls
