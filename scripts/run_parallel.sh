#!/usr/bin/env bash
# Launch the four Blankenbach cases that were still open, one tmux session each,
# four threads per session so the machine's sixteen cores are all used.
set -u
cd /mnt/e/研究生资料/project/QuaST || exit 1

tmux kill-server 2>/dev/null || true
sleep 1

launch () {  # name case level tstop
  local name=$1 case=$2 level=$3 tstop=$4
  tmux new-session -d -s "$name" \
    "bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh $case $level $tstop 4; sleep 7200"
  echo "started $name ($case level $level)"
}

launch b1a 1a 4 1.0
launch b2a 2a 4 1.0
launch b1b 1b 4 1.0
launch b1c 1c 3 1.0

sleep 5
tmux ls
