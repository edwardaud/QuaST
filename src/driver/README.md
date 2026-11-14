# Driver

The time loop and the nonlinear solvers that advance one step or seek a steady
state. `run.jl` is `run_semquad_simulation`, the production entry point;
`picard.jl` holds the lagging, averaging and relaxation helpers; `newton.jl`,
`tosi_steady.jl`, `dp_steady.jl` and `step_newton.jl` implement the analytic
tangent maps for eta(T), the Tosi law and Drucker–Prager, steady and per-step;
`steady_entry.jl` dispatches between them. Physics and linear algebra are
elsewhere: the driver only composes them.

| File | What it is |
|---|---|
| `dp_steady.jl` | Steady Newton for the Drucker–Prager rheology, including the pressure derivative. |
| `mass.jl` | The TALA/EOS density functions, the compressibility block and the eta-range diagnostic. |
| `newton.jl` | Analytic tangent-linear Newton for the steady quadtree step map. |
| `picard.jl` | Lagged viscosity, strain-rate averaging and adaptive Picard relaxation. |
| `run.jl` | `run_semquad_simulation`: the time loop, output history and per-step profile. |
| `steady_entry.jl` | Uniform steady-state entry point dispatching the three Newton paths. |
| `step_newton.jl` | Globalization of the per-time-step nonlinear solve. |
| `tosi_steady.jl` | Steady Newton for the Tosi (2015) viscoplastic rheology. |
