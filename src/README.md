# Source

The QuaST package itself. `QuaST.jl` is the only entry point: it fixes the
include order (only definition-time references constrain it) and the exported
surface, and every other file here is loaded by it. Nothing under `src/` runs on
its own. The directories are layers of one module, not separate packages, and
the calls between them run one way: `stokes/` and `physics/` use `core/`,
`mg/` and `driver/` use `stokes/`, and the driver composes the rest. Gates live
in `test/`, runnable cases in `examples/`, timings in `benchmarks/`.

| File or directory | What it holds |
|---|---|
| `QuaST.jl` | Module entry point: the include order and the exported surface. |
| `core/` | Quadrature, reference element, the quadtree, the Q_p space and constraints. |
| `physics/` | Material laws, the parameter card and the energy equation. |
| `stokes/` | Taylor–Hood Stokes assembly and the linear solves. |
| `mg/` | Geometric multigrid and Krylov for the Stokes saddle. |
| `amr/` | The balanced AMR policy, the indicator and the AMR pass. |
| `driver/` | The time loop, Picard/Newton and the steady entries. |
| `io/` | HDF5 checkpoint and restart. |
| `diagnostics/` | Nusselt numbers, Vrms, CBF and entropy viscosity. |
