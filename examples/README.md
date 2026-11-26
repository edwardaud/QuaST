# Examples

Runnable cases and checks. Run them from the repository root after
`julia --project=. -e 'using Pkg; Pkg.instantiate()'`, as
`julia --project=. examples/<script>` plus whatever arguments the script header lists.
`docs/en/examples.md` walks through the main runs and gives their measured runtimes.

Run output is written under `out_julia/`, which git ignores.

| Script | What it does |
|---|---|
| `data/` | The tracked Blankenbach initial-condition fields; CC0-1.0, see `data/README.md`. |
| `case_realistic_2d.jl` | The production case: 2.5 x 1 box, composite rheology, two phase transitions with latent heat, adaptive mesh. Modes `smoke`, `short`, `spinup`, `resume`, `freeze`, `wallrefine`, `symseed`. |
| `channel_flow_2d.jl` | The regional channel demonstration: a prescribed normal-velocity inflow, no-slip channel walls and an `:open` outflow carrying the exact traction, compared against plane Poiseuille. |
| `subduction_2d.jl` | The regional subduction-style case: a 2 x 1 kinematic channel with the three per-wall mechanisms, the exact Couette-Poiseuille validation, and modes `validate`, `rheology`, `phases`, `full`, `converge`. |
| `benchmark_cases.jl` | Every published benchmark in one entry point: Blankenbach 1a, 1b, 1c, 2a, 2b and Tosi cases 1 and 5a, each at a chosen level. |
| `annulus_benchmark.jl` | The published 2-D cylindrical-annulus benchmark (Davies et al. 2022 / ASPECT `Ra = 1e5` free-slip ring): the paper card and initial condition, the stated steady-state criterion, the diagnostics (`Nu_T`, `Nu_B`, `V_RMS`, `<T>`, the wall-flux split), checkpoint/resume, the steady-relaxation mode used at the published resolutions, and the comparison line against the published values. Usage and the measured rungs are in `docs/en/examples.md`. |
| `annulus_benchmark.jl demo` | The purpose-configured annulus AMR **demonstration**, not a benchmark: the `demo` token (instead of `<nx>x<ny>`) sets Ra = 1e3 at the reference viscosity 1, eta = 1000^(-T), H = 0, the ASPECT mode-4 IC, frames every 0.005 with a per-frame mesh dump, and the balanced AMR cycle every 5 steps (the ENV knobs `ANN_RA`, `ANN_H`, `ANN_DETA_T`, `ANN_AMR_EVERY`, `ANN_MAX_LEAVES`, ... override every preset). It is the QuaST side of `docs/assets/quast_vs_aspect_annulus_demo.gif`; `scripts/run_annulus_demo.sh` runs it and renders the figure. |
| `run_blob_kelly_1024.jl` | The 25-frame `Ra = 1e8` blob baseline, with per-frame SHA-256 rows. |
| `calib_realistic_2d.jl` | Derivation of the composite rheology prefactors that `case_realistic_2d.jl` carries as literals. |
| `import_aspect_state.jl` | Build a QuaST checkpoint from an ASPECT state, for the same-state comparison. |
| `validate_aspect_mapping.jl` | Check the dimensional mapping of the `:composite` law onto ASPECT's `diffusion dislocation` model. |
| `verify100_case.sh` | Launch one arm of the three-arm run (direct, multigrid, production) that shows the two linear paths agree. |
| `verify100_compare.jl` | Compare the end states of those arms frame by frame. |
| `verify100_ic_check.jl` | Check that the ASPECT card and the QuaST card start from the same function at `t = 0`. |
| `nusselt_audit.jl` | Arbitration between the three wall-flux definitions against the scheme's own discrete budget. |
| `nusselt_wall_functional_test.jl` | The algebra relating the pointwise and consistent-boundary-flux functionals on one wall element. |
| `nu_cbf_compare.jl` | The earlier pointwise-versus-CBF comparison on the realistic card, kept as the provenance of that study. |
| `amr_coarsen_gate_probe.jl` | Coarsening-gate probe for the balanced AMR policy. |

The benchmark variants that used to live here (`blankenbach_2b_time.jl`,
`_stretch.jl`, `_wall.jl`, `_cbf_ev.jl`, and `blankenbach_2b.jl`) are gone.
`benchmark_cases.jl` covers all of them: it starts from ASPECT's own initial
field, takes the wall rule and the level as arguments, and reports the wall,
CBF and energy families.
