#!/usr/bin/env bash
# Queue the rest of the benchmark campaign behind the runs in flight.
#
#   - 1c, 2a, 2b level 4 behind their level 3 (2a, 2b) or level 4 (1c) run
#   - 1c level 5 behind its level 4
#   - Tosi case 1 behind 1a level 5, which is the run closest to finishing
#
# The steady answer of a benchmark does not depend on the time step, so every
# queued run uses CFL 2 instead of 0.5; ASPECT runs these cases with CFL 5.
set -u
cd /mnt/e/研究生资料/project/QuaST || exit 1

queue_level () {  # case level wait_level
  local case=$1 level=$2 wait_level=$3
  tmux new-session -d -s "q${case}L${level}" \
    "CASE_CFL=2.0 bash /mnt/e/研究生资料/project/QuaST/scripts/queue_level.sh $case $level $wait_level 0.2 3; sleep 600"
  echo "queued $case level $level behind level $wait_level"
}

queue_after () {  # target_session case level
  local target=$1 case=$2 level=$3
  tmux new-session -d -s "q${case}L${level}" \
    "CASE_CFL=2.0 bash /mnt/e/研究生资料/project/QuaST/scripts/queue_after_session.sh $target $case $level 0.2 3; sleep 600"
  echo "queued $case level $level behind session $target"
}

queue_level 1c 5 4
queue_level 2a 4 3
queue_level 2b 4 3
queue_after q1aL5 tosi1 3

sleep 3
tmux ls
