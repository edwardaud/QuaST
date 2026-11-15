#!/usr/bin/env bash
# Tosi case 5a, now at the calibrated Ra = 1000.
set -u
cd /mnt/e/研究生资料/project/QuaST || exit 1
tmux kill-session -t t5a 2>/dev/null || true
tmux new-session -d -s t5b \
  "CASE_TAG=ra1000 CASE_RA=1000 CASE_STEADY=Inf bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh tosi5a 3 1.0 3; sleep 600"
sleep 3
tmux ls
