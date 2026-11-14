#!/usr/bin/env bash
# Wall-refined runs for 2a and 2b: interior at level 4, walls forced to level 5.
# The uniform 2b level 4 run keeps going; this one is tagged so the two logs do
# not collide.
set -u
cd /mnt/e/研究生资料/project/QuaST || exit 1

tmux kill-session -t b2bL4w 2>/dev/null || true
tmux new-session -d -s b2bL4w \
  "CASE_TAG=wall CASE_WALLMIN=5 CASE_INITREF=1 bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh 2b 4 0.2 3; sleep 600"
echo "started 2b level 4, wall refined"

sleep 3
tmux ls
