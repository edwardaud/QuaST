#!/usr/bin/env bash
# Reproduce the validation campaign: every benchmark row in the validation page.
#
#     bash scripts/run_benchmarks.sh [threads]
#
# Each case runs in its own tmux session, so closing the terminal does not kill
# it.  The settings are the ones the benchmark cards use:
#
#   - Blankenbach: CFL 5 (base_case2b.prm), the paper's initial condition, and no
#     adaptive refinement during the run;
#   - 2a and 2b additionally use the wall rule, since Nu is a one-sided gradient
#     inside the boundary layer;
#   - Tosi: Ra = 1000 with eta0 = 0.1, which is the pairing that reproduces
#     Table 2, and the steady-per-step criterion off, because a Tosi flow evolves
#     on a timescale far longer than one step.
#
# Logs land in $QUAST_OUT/benchmarks (default: out_julia/benchmarks);
# `scripts/benchmark_table.jl` turns them into the table rows.
#
# Portability: QUAST_ROOT (repository root), QUAST_JULIA (the Julia executable)
# and QUAST_OUT (the run-output root) are taken from the environment when set;
# the defaults are the repository this script lives in, `julia` from PATH and
# out_julia/ under the repository.  scripts/README.md documents the three.
set -u
QUAST_ROOT=${QUAST_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}
QUAST_JULIA=${QUAST_JULIA:-julia}
QUAST_OUT=${QUAST_OUT:-$QUAST_ROOT/out_julia}
cd -- "$QUAST_ROOT" || exit 1

JOB=${1:-4}   # threads per case

launch () {   # session case level tstop wall cfl extra-env...
  local sess=$1 case=$2 level=$3 tstop=$4 wall=$5 cfl=$6; shift 6
  tmux kill-session -t "$sess" 2>/dev/null || true
  tmux new-session -d -s "$sess" \
    "QUAST_ROOT='$QUAST_ROOT' QUAST_JULIA='$QUAST_JULIA' QUAST_OUT='$QUAST_OUT' \
     CASE_WALLMIN=$wall CASE_CFL=$cfl CASE_INITREF=$([ "$wall" -gt 0 ] && echo 1 || echo 0) CASE_STEADY=Inf $* \
     bash '$QUAST_ROOT/scripts/run_one.sh' $case $level $tstop $JOB; sleep 600"
  printf 'started %-10s %s level %s, wall %s, CFL %s\n' "$sess" "$case" "$level" "$wall" "$cfl"
}

# Blankenbach family: 1a, 1b, 1c at two levels each; 2a and 2b with walls.
launch b1a3 1a 3 0.2  0 5.0
launch b1a5 1a 5 0.2  0 5.0
launch b1b3 1b 3 0.2  0 5.0
launch b1b5 1b 5 0.2  0 5.0
launch b1c3 1c 3 0.2  0 5.0
launch b1c5 1c 5 0.2  0 5.0
launch b2a  2a 3 0.2  4 5.0
launch b2b  2b 4 1.0  5 5.0

# Tosi family at the calibrated Rayleigh number.
launch t1  tosi1  3 1.0 0 0.5 CASE_RA=1000 "CASE_CKPT_DIR=$QUAST_OUT/benchmarks/ckpt_tosi1"
launch t5a tosi5a 3 1.0 0 0.5 CASE_RA=1000 "CASE_CKPT_DIR=$QUAST_OUT/benchmarks/ckpt_tosi5a"

sleep 3
tmux ls
