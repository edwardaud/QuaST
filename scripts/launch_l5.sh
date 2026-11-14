#!/usr/bin/env bash
# Queue level-5 runs for the cases whose level-4 run is in flight.
set -u
cd /mnt/e/研究生资料/project/QuaST || exit 1

for case in 1a 1b; do
  tmux new-session -d -s "q${case}L5" \
    "bash /mnt/e/研究生资料/project/QuaST/scripts/queue_level.sh $case 5 4 0.2 4; sleep 600"
  echo "queued ${case} level 5 behind level 4"
done

sleep 3
tmux ls
