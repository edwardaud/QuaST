#!/usr/bin/env bash
# Tosi case 1 with the walls refined: interior level 3, wall strip and wall rows
# at level 4 (1 600 leaves against 1 024 uniform).  This is the cheap probe of
# whether the +10.6 % bottom flux at level 3 is a wall-resolution effect.
set -u
cd /mnt/e/研究生资料/project/QuaST || exit 1
tmux kill-session -t t1w 2>/dev/null || true
tmux new-session -d -s t1w \
  "CASE_TAG=ra1000wall4 CASE_RA=1000 CASE_STEADY=Inf CASE_WALLMIN=4 CASE_INITREF=1 bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh tosi1 3 1.0 3; sleep 600"
sleep 3
tmux ls
