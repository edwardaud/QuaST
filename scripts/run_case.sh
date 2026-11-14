#!/usr/bin/env bash
# Start one case at a given level in its own tmux session.
#
#     bash scripts/run_case.sh <case> <level> [tstop] [threads]
set -u
case=${1:?case}
level=${2:?level}
tstop=${3:-0.2}
threads=${4:-3}
cd /mnt/e/研究生资料/project/QuaST || exit 1
tmux new-session -d -s "b${case}L${level}" \
  "bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh $case $level $tstop $threads; sleep 600"
echo "started b${case}L${level}"
