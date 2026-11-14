#!/usr/bin/env bash
# Queue a finer-level run behind the one already running for the same case.
#
#     bash scripts/queue_level.sh <case> <level> <wait_for_level> [tstop] [threads]
#
# It polls the coarser log until the run has written its final "[case] steps="
# line, then starts the finer level in the same session.
set -u
case=${1:?case}
level=${2:?level}
wait_level=${3:?wait level}
tstop=${4:-0.2}
threads=${5:-4}

cd /mnt/e/研究生资料/project/QuaST || exit 1
wait_log="out_julia/benchmarks/${case}_L${wait_level}.log"

while ! grep -q "^\[${case}\] steps=" "$wait_log" 2>/dev/null; do
  sleep 30
done

bash /mnt/e/研究生资料/project/QuaST/scripts/run_one.sh "$case" "$level" "$tstop" "$threads"
