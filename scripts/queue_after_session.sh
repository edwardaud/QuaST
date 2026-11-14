#!/usr/bin/env bash
# Start a case once another tmux session has finished.
#
#     bash scripts/queue_after_session.sh <target session> <case> <level> [tstop] [threads]
set -u
target=${1:?target session}
case=${2:?case}
level=${3:?level}
tstop=${4:-0.2}
threads=${5:-3}

cd /mnt/e/研究生资料/project/QuaST || exit 1
while tmux has-session -t "$target" 2>/dev/null; do
  sleep 30
done
bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh "$case" "$level" "$tstop" "$threads"
