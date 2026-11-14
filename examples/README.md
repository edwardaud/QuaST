# Examples

Runnable cases and checks. Run them from the repository root after
`julia --project=. -e 'using Pkg; Pkg.instantiate()'`, with the form
`julia --project=. examples/<script>` plus the arguments the script's header lists.
`docs/examples.md` walks through the four main runs and gives their measured eight-thread
runtimes.

Several scripts read a checkpoint of the realistic card. Run output and checkpoints are not
tracked (`out_julia/` is ignored by git), so reproduce a state with the card first, or pass
your own file as the argument. `examples/data/` holds the ASCII initial temperature that
`blankenbach_2b_cbf_ev.jl` reads.

| Script | What it computes |
|---|---|
| `amr_coarsen_gate_probe.jl` | Replays the balanced AMR pass on a checkpoint tree with the cycle, wall rule and leaf budget timed separately, and reports quartet merges, merges that the post-merge balance splits again, foreign splits, the residual leaf count and the 2:1 violations; its `supply` mode reports the coarsening supply that the 2:1 pre-check would keep. Usage: `trace <ckpt.h5>` or `supply <ckpt.h5>`. |
| `blankenbach_2b.jl` | Blankenbach 2b on square cells (5 x 2 base) with the time-dependent driver and a steady-state stop; prints wall `Nu_top` and `Vrms` against the 6.9299 / 171.755 reference. Usage: `<level> <tstop> [ev]`. |
| `blankenbach_2b_cbf_ev.jl` | Runs 2b from the stored ASPECT initial temperature and measures whether the consistent boundary flux changes when the stored entropy-viscosity field is removed; the two values must be bitwise equal. Usage: `[level] [tstop]`. |
| `blankenbach_2b_stretch.jl` | 2b on a stretched single-root grid (`2^L` by `2^L` cells with `h_x = 2.5 h_y`), with optional wall refinement. Usage: `[level] [wall] [tstop]`. |
| `blankenbach_2b_time.jl` | 2b as four time-dependent runs with the viscosity coefficients scaled by 0.25, 0.5, 0.75 and 1.0, each stage starting from the previous temperature field. |
| `blankenbach_2b_wall.jl` | 2b with wall refinement in the thermal boundary layers: level 4 (80 x 32) plus a level-6 strip of width 0.1 at both walls. Usage: `[base] [wall] [tstop]`. |
| `blankenbach_ra_scan.jl` | Steady solves at `Ra` = 3e3, 1e4 and 3e4 on an 8 x 8 `p = 4` mesh; prints `Nu` and `Vrms` for each. |
| `calib_realistic_2d.jl` | Derives the composite prefactors: measures the reference strain rate from a constant-viscosity precursor, solves the two calibration conditions for `comp_Ad` and `comp_As`, and prints the parameter card. |
| `case_realistic_2d.jl` | The production 2.5 x 1 card with modes `smoke`, `short`, `spinup`, `freeze`, `wallrefine`, `resume` and `symseed`; writes `history.csv`, frames and HDF5 checkpoints. The header gives `smoke` as about two minutes. |
| `highra_run.jl` | High-Rayleigh-number, high-viscosity-contrast runner on a tanh-graded mesh, with optional internal heating; writes `out_julia/highra_<tag>_frames.h5`, a log and a history CSV. Usage: `[nxe] [Ra] [deta] [tstop] ...`. |
| `import_aspect_state.jl` | Builds a QuaST checkpoint from an ASPECT state file written by `aspect_runs/realistic/extract_aspect_state.py`: evaluates the Q2 field at the QuaST nodes, builds the card's tree with the card's own AMR rule, and stores the state with a template checkpoint's configuration. Usage: `--state ... --template ... --out ...`. |
| `nu_cbf_compare.jl` | On one checkpoint state, compares the pointwise wall flux, the steady consistent boundary flux, the exact Dirichlet reaction of the time step that produced the state, and the heat-budget terms; `--sweep` repeats the comparison at `h` = 1/16, 1/32 and 1/64. |
| `nusselt_audit.jl` | Measures every term of the discrete energy identity on a fixed state and the residual of each candidate Nusselt definition against it. Usage: `[options] ckpt [ckpt...]`. |
| `nusselt_wall_functional_test.jl` | Applies the pointwise and consistent-boundary-flux wall functionals to exact polynomials on the production tree and compares them with the exact wall flux. Needs a checkpoint argument. |
| `run_2b.jl` | Steady 2b with four continuation stages on a 40 x 16 `p = 4` mesh; prints `Nu_top`, `Nu_bot` and `Vrms` per stage. Usage: `[nxe] [nze] [stage...]`. |
| `run_blob_kelly_1024.jl` | Produces the 25-frame blob regression: 126 steps at `Ra = 1e8` on a 1024-leaf tree with Kelly AMR every 5 steps, per-frame SHA-256 rows, and the final `Tmax25` and `theta` digest. `BLOB_OUT` selects the output directory. |
| `run_tosi_case1.jl` | Steady Tosi Case 1 on an 8 x 8 `p = 4` mesh; prints `Nu_top`, `Nu_bot` and `Vrms`. |
| `run_tosi_case2_diag16.jl` | Steady Tosi Case 2 on a 16 x 16 `p = 4` mesh with the mean temperature, surface RMS velocity and surface maximum velocity. |
| `run_tosi_case3_diag16.jl` | The same diagnostics for Tosi Case 3 on a 16 x 16 `p = 4` mesh. |
| `validate_aspect_mapping.jl` | Evaluates the QuaST composite isostress law and ASPECT's diffusion-dislocation law on a grid of temperature, strain rate and depth, and prints the relative difference. |
| `verify100_case.sh` | Launches one arm (`mg`, `direct` or `auto`) of the 500-step three-arm verification from the card's `t = 0` state, with every card knob written out; refuses an output directory that already exists. Usage: `bash examples/verify100_case.sh <arm> [steps]`. |
| `verify100_compare.jl` | Compares the end states of two arms on the common 256 x 256 frame grid, with the end-state diagnostics and the step at which the fields start to separate. Usage: `<dirA> <dirB> [production]`. |
| `verify100_ic_check.jl` | Checks that the QuaST card and `aspect_runs/realistic/realistic.prm` define the same `t = 0` temperature and share every constant; the exit code is 0 if all checks pass. Usage: `[prm path]`. |

Only `case_realistic_2d.jl` gives a runtime in its header (about two minutes for `smoke`);
the other scripts do not, so this table has no runtime column. `docs/examples.md` records the
measured times for the four main runs.
