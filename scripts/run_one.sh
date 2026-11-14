#!/usr/bin/env bash
# Run one benchmark case in its own process, so several can share the machine.
#
#     bash scripts/run_one.sh <case> <level> <tstop> <threads>
#
# The log lands in out_julia/benchmarks/<case>_L<level>.log.
set -u
case=${1:?case}
level=${2:?level}
tstop=${3:?tstop}
threads=${4:-4}

cd /mnt/e/研究生资料/project/QuaST || exit 1
J=/home/szh/opt/julia-1.12.7/bin/julia
mkdir -p out_julia/benchmarks
# CASE_TAG separates two runs of the same case and level, e.g. a wall-refined
# run from the uniform one it is meant to be compared with.
tag=${CASE_TAG:+_$CASE_TAG}
log="out_julia/benchmarks/${case}_L${level}${tag}.log"
echo "=== $case level $level tstop $tstop threads $threads tag ${CASE_TAG:-none} start $(date '+%H:%M:%S')"
$J --project=. --threads="$threads" examples/benchmark_cases.jl "$case" "$level" "$tstop" \
   > "$log" 2>&1
tail -2 "$log"
echo "=== $case done $(date '+%H:%M:%S')"
