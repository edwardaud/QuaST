# Driver

The time loop and the nonlinear solvers that advance one step or seek a steady
state. `run.jl` is `run_semquad_simulation`, the production entry point;
`maxwell_state.jl`, `maxwell_reports.jl` and `eta_ref_report.jl` hold the
viscoelastic driver state and the two viscosity diagnostics it reports;
`picard.jl` holds the lagging, averaging and relaxation helpers; `newton.jl`,
`tosi_steady.jl`, `dp_steady.jl` and `step_newton.jl` implement the analytic
tangent maps for eta(T), the Tosi law and Drucker–Prager, steady and per-step;
`steady_entry.jl` dispatches between them. Physics and linear algebra are
elsewhere: the driver only composes them.

| File | What it is |
|---|---|
| `dp_steady.jl` | Steady Newton for the Drucker–Prager rheology, including the pressure derivative. |
| `eta_ref_report.jl` | The `eta_ref` convention report: normalised field, interior mean, `Ra_interior`, physical units. |
| `mass.jl` | The TALA/EOS density functions, the compressibility block and the eta-range diagnostic. |
| `maxwell_reports.jl` | The Maxwell diagnostics: the co-rotation validity band and the viscosity regime records. |
| `newton.jl` | Analytic tangent-linear Newton for the steady quadtree step map. |
| `maxwell_state.jl` | The driver-side Maxwell stress state: nodal material viscosity and the one-step objective update. |
| `picard.jl` | Lagged viscosity, strain-rate averaging and adaptive Picard relaxation. |
| `picard_control.jl` | The field blend, the divergence test and the adaptive relaxation. |
| `strain_rate.jl` | The nodal strain-rate and spin fields of the SEM velocity. |
| `restart_guard.jl` | The `run_config` builder and the restart guard of `run_semquad_simulation`. |
| `run.jl` | `run_semquad_simulation`: the time loop, output history and per-step profile. |
| `steady_entry.jl` | Uniform steady-state entry point dispatching the three Newton paths. |
| `step_newton.jl` | Globalization of the per-time-step nonlinear solve. |
| `tosi_steady.jl` | Steady Newton for the Tosi (2015) viscoplastic rheology. |
