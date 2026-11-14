#!/usr/bin/env bash
# Launch the Blankenbach cases, one tmux session each.
#
# Each case starts from ASPECT's own `ascii data` initial condition, which is
# already close to the steady state, so a run needs tens of steps rather than
# tens of thousands.
set -u
cd /mnt/e/研究生资料/project/QuaST || exit 1

tmux kill-server 2>/dev/null || true
sleep 1

launch () {  # name case level tstop threads
  local name=$1 case=$2 level=$3 tstop=$4 threads=${5:-4}
  tmux new-session -d -s "$name" \
    "bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh $case $level $tstop $threads; sleep 600"
  echo "started $name ($case level $level, $threads threads)"
}

launch b1a 1a 4 0.2 4
launch b1b 1b 4 0.2 4
launch b1c 1c 4 0.2 4
launch b2a 2a 4 0.2 4

sleep 5
tmux ls
