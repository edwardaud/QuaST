#!/usr/bin/env bash
# Run the Blankenbach cases the validation table still lists as open. CFL-driven
# time step, so each case advances far enough to reach its steady state.
set -u
cd /mnt/e/研究生资料/project/QuaST || exit 1
J=/home/szh/opt/julia-1.12.7/bin/julia
OUT=out_julia/benchmarks
mkdir -p "$OUT"

run () {  # case level tstop
  local case=$1 level=$2 tstop=$3
  echo "=== $case level $level tstop $tstop start $(date '+%H:%M:%S')"
  $J --project=. --threads=8 examples/benchmark_cases.jl "$case" "$level" "$tstop" \
     > "$OUT/${case}_L${level}.log" 2>&1
  tail -3 "$OUT/${case}_L${level}.log"
  echo "=== $case done $(date '+%H:%M:%S')"
}

run 1a 4 1.0
run 2a 4 1.0
run 1b 4 1.0
run 1c 3 1.0
echo "=== queue finished $(date '+%H:%M:%S')"
