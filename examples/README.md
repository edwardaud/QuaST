# Examples

Runnable cases and checks. Run them from the repository root after
`julia --project=. -e 'using Pkg; Pkg.instantiate()'`, as
`julia --project=. examples/<script>` plus whatever arguments the script header lists.
`docs/examples.md` walks through the main runs and gives their measured runtimes.

Run output is written under `out_julia/`, which git ignores.

| Script | What it does |
|---|---|
| `amr_coarsen_gate_probe.jl` | Coarsening-gate probe for the balanced AMR policy (docs/dev/AMR.md §4.1). |
| `blankenbach_2b.jl` | Blankenbach 1989 case 2b, ASPECT's setup (benchmarks/blankenbach/base_case2b.prm): |
| `blankenbach_2b_cbf_ev.jl` | Blankenbach 2b: what does the consistent boundary flux (CBF) actually say on |
| `blankenbach_2b_stretch.jl` | Blankenbach 2b on the STRETCHED single-root base (the original route C grid): |
| `blankenbach_2b_time.jl` | Blankenbach (1989) 2b, **time-dependent**, on the square-cell grid (5x2 base). |
| `blankenbach_2b_wall.jl` | Blankenbach 2b on route C with **wall refinement** instead of uniform h-refinement: |
| `calib_realistic_2d.jl` | Calibration of the composite (diffusion + dislocation) prefactors for the |
| `case_realistic_2d.jl` | ============================================================================= |
| `import_aspect_state.jl` | ============================================================================= |
| `nu_cbf_compare.jl` | ============================================================================= |
| `nusselt_audit.jl` | ============================================================================= |
| `nusselt_wall_functional_test.jl` | What are the two wall functionals, exactly? |
| `run_blob_kelly_1024.jl` | Produce the 25-frame blob baseline of docs/dev/VALIDATION.md §2.3. |
| `validate_aspect_mapping.jl` | Validate the dimensional mapping of the QuaST `:composite` rheology onto |
| `verify100_case.sh` | verify100_case.sh -- launch one arm of the three-arm verification run. |
| `verify100_compare.jl` | verify100_compare.jl -- end-state comparison of the direct and MG arms, and |
| `verify100_ic_check.jl` | verify100_ic_check.jl -- is the ASPECT card's t=0 state the same function as |
