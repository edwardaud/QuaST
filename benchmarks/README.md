# Benchmarks

Performance instruments behind the numbers in `results/`. They time the solver on production
states and on uniform test trees, and print their own tables or one `RESULT key=value` line.
Run them from the repository root with the project instantiated, for example

```bash
julia --project=. --threads=8 benchmarks/bench_direct_vs_mg.jl F
```

Most states come from files under `out_julia/`, which is not tracked: the production
checkpoints `out_julia/_s7/case/pre_step6131.h5` (state `F`) and
`out_julia/s340/ab_none/resume.h5` (state `P1`), and the blob frames in
`out_julia/fresh_blob_kelly_1024/`. The scripts that build their own uniform trees say so
below. Every script copies its checkpoint to a temporary directory and opens it read-only.

| Script | What it measures |
|---|---|
| `results/` | Provenance records behind published numbers, one file per number; the convention is in `results/README.md`. |
| `bench_auto_crossover.jl` | Same-session A/B of `linear=:direct` with `stokes_scaling=:equilibrated` and `linear=:mg` with the shipped defaults, on the state classes the old selector table names: uniform `L2` to `L6`, the blob frame-19 trees, the 25-frame blob trajectory, the `mid` contrast-50 trees, `L<k>p2` and the `F` and `P1` states. Takes one or more state names. |
| `bench_bfbt_leafdiag.jl` | Builds the BFBT weight both ways (the lumped restriction and the exact leaf-local diagonal), times the pressure block and one FGMRES solve with each, and compares both against the equilibrated direct solve. Takes `F` or `P1`. |
| `bench_card_mg_tol.jl` | Runs the card's solver at `mg_tol` = 1e-10, 1e-8, 1e-6 and 1e-5 on one state and reports iterations, true residual, warm solve time, the auto decision, and the difference against the 1e-10 run and the direct solve. Takes `F`, `P1` or `C`. |
| `bench_direct_vs_mg.jl` | Head-to-head of the direct and matrix-free multigrid paths in one process on one checkpoint: assembly, setup, factorisation, iteration, cold and warm solve times, and peak resident memory. Takes `F` or `P1` and a repeat count. |
| `bench_mg_apply.jl` | Per-level timing of the matrix-free velocity apply `Cu' K (Cu v)`, its three phases, per-leaf and per-dof costs, allocations per apply, the V-cycle, and a memory-bandwidth calibration. `STATE=F` or `STATE=P1`. |
| `bench_mg_defaults.jl` | Cost and accuracy of the matrix-free multigrid default variants: coarsening factor 1.0 against 2.0, and `:auto`, `:taper` and `:uniform` smoothing. Takes `<case> <variant> [reps]`, with `F`, `P1`, `l5`, `l6` or `blob19` as the case. |
| `bench_mg_forks.jl` | Fork and join accounting of the per-call and consolidated V-cycles, per-level timings, the empty-fork cost, and with `BENCH_SOLVE=1` the warm step and full solve for both paths, compared bitwise. `BENCH_CASE=F` or `P1`. |
| `bench_mg_level_cache.jl` | Rebuilds versus reuses the per-level multigrid hierarchy cache over a 30-step AMR trajectory, with per-cycle build times, the end-to-end step, and a bitwise comparison of `theta`, `ux` and `uz`. `BENCH_CASE=F` or `P1`. |
| `bench_mg_reduced_assembly.jl` | Setup and solve cost of the matrix-free path before and after the reduced assembly; `BENCH_NEW=0` runs the same script against the pre-change tree. `BENCH_CASE=F` or `P1`. |
| `bench_mg_slice.jl` | Cache and working-set measurement for the cache-blocked leaf kernel: the reported cache sizes, the block's live bytes, per-level kernel times, the flushed-cache control, and the bandwidth calibration. `STATE=F` or `STATE=P1`. |
| `bench_s1_solve.jl` | Cost of the two `chol(S1)` triangular solves in the w-BFBT pressure block: the setup split, hot, allocating and flushed solves, multi-right-hand-side fits, alternate factorisation kernels, and the structure of `F1`. Takes `F` or `P1`. |
| `profile_mg_current.jl` | One warm matrix-free multigrid solve on a production state, split into assembly, hierarchy build, w-BFBT build, FGMRES iteration and the per-iteration components, with the true-residual history. `MG_PROFILE_CASE=F` or `P1`. |
| `profile_mg_vcycle.jl` | Component-level profile of one V-cycle and one full `linear=:mg` saddle solve on the `F` state, with the replica loops checked bitwise against the production functions. `MG_PROFILE_CKPT` overrides the checkpoint. |

None of these scripts records a runtime in its header; they print their own timings, so the
table has no runtime column. The environment variables each script reads are listed in its
header, and the measurement conditions are recorded per number in `results/`.
