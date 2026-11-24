# Driver

The time loop and the nonlinear solvers that advance one step. `run.jl` is
`run_semquad_simulation`, the production entry point; `maxwell_state.jl`,
`maxwell_reports.jl` and `eta_ref_report.jl` hold the viscoelastic driver state
and the two viscosity diagnostics it reports; `picard.jl` holds the lagging,
averaging and relaxation helpers; `step_newton.jl` and `newton_tangent.jl`
implement the optional per-step Newton (a research path: it needs
`linear = :direct` and is slower than the production Picard + multigrid; see the
header of `step_newton.jl`). Physics and linear algebra are elsewhere: the driver
only composes them.

| File | What it is |
|---|---|
| `amr_step.jl` | The AMR block of the time loop, extracted from `run.jl` as `_amr_step!`. |
| `diagnostics_row.jl` | The per-step diagnostics row of the run history. |
| `eta_ref_report.jl` | The `eta_ref` convention report: normalised field, interior mean, `Ra_interior`, physical units. |
| `mass.jl` | The TALA/EOS density functions, the compressibility block and the eta-range diagnostic. |
| `maxwell_reports.jl` | The Maxwell diagnostics: the co-rotation validity band and the viscosity regime records. |
| `maxwell_state.jl` | The driver-side Maxwell stress state: nodal material viscosity and the one-step objective update. |
| `newton_tangent.jl` | The matrix-free tangent-vector product of the per-step Newton. |
| `picard.jl` | Lagged viscosity, strain-rate averaging and adaptive Picard relaxation. |
| `picard_control.jl` | The field blend, the divergence test and the adaptive relaxation. |
| `strain_rate.jl` | The nodal strain-rate and spin fields of the SEM velocity. |
| `restart_guard.jl` | The `run_config` builder and the restart guard of `run_semquad_simulation`. |
| `run.jl` | `run_semquad_simulation`: the time loop, output history and per-step profile. |
| `run_diagnostics.jl` | The driver-owned snapshot of the solver and AMR diagnostic out-parameters. |
| `step_newton.jl` | Globalization of the per-time-step nonlinear solve. |

Stage 0 cut 4 lifts the mapped-geometry refusal: with cuts 1–4 the assembled
Stokes operator, the energy assembly and every per-step diagnostics path are
routed through the geometry, so `run_semquad_simulation(...; geom=
AffineBox/CurvedBox)` runs a **fixed-mesh, `linear=:direct`, Picard** time loop.
What remains box-only is refused up front, naming the owning cut: adaptive
transfer/indicator/AMR and the `:x` symmetry mirror (cut 6), the matrix-free/MG
solver (cut 5), the strain-rate reconstructions of `driver/strain_rate.jl` (cut 6,
which every strain-rate-dependent law, yielding, weakening and the Maxwell
stress advance need), the per-step Newton path (its mapped tangent has no gate
yet), and the Cartesian-`y` reference-state / density-hook / phase-buoyancy
physics wiring (the annulus stage re-evaluates those closures at `height`).  The
history row of a mapped run is the mapped diagnostics: G270 runs the driver on
both validation geometries with heating off/on, the CBF and exact-residual
columns on, and the latent-heat closure path, and closes the time-step CBF
energy budget identity of `docs/dev/VALIDATION.md` §4.1 to the solver's floor.

