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

# Machine-independent paths.  Each comes from the environment when set, so a
# checkout at another path and a Julia that is not on PATH both work without
# editing this file.  The defaults are the repository this script lives in,
# `julia` from PATH, and the git-ignored out_julia/ tree under the repository.
# scripts/README.md documents the three variables.
QUAST_ROOT=${QUAST_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}
QUAST_JULIA=${QUAST_JULIA:-julia}
QUAST_OUT=${QUAST_OUT:-$QUAST_ROOT/out_julia}
cd -- "$QUAST_ROOT" || exit 1
J=$QUAST_JULIA
mkdir -p "$QUAST_OUT/benchmarks"
# CASE_TAG separates two runs of the same case and level, e.g. a wall-refined
# run from the uniform one it is meant to be compared with.
tag=${CASE_TAG:+_$CASE_TAG}
log="$QUAST_OUT/benchmarks/${case}_L${level}${tag}.log"
echo "=== $case level $level tstop $tstop threads $threads tag ${CASE_TAG:-none} start $(date '+%H:%M:%S')"
$J --project=. --threads="$threads" examples/benchmark_cases.jl "$case" "$level" "$tstop" \
   > "$log" 2>&1
tail -2 "$log"
echo "=== $case done $(date '+%H:%M:%S')"
