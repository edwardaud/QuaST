# Stokes

The linear algebra of the Taylor–Hood Stokes saddle: the `SemStokes` state, the
reference element matrices and the variable-viscosity leaf assembly, the
matrix-free velocity operator, the pressure operators, the sparse direct solve,
Schur-CG and the multigrid entry point, plus the `linear=:auto` selector. This
is the linear algebra, not the physics: the viscosity field and the forcing come
from `physics/`, and the hierarchy and Krylov machinery are in `mg/`.

| File | What it is |
|---|---|
| `apply.jl` | The consolidated per-level fork/join used by the matrix-free applies and V-cycle sweeps. |
| `assembly.jl` | Reference Q_p/Q_q element matrices and the sparse pressure mass/stiffness. |
| `assembly_var.jl` | Leaf-parallel variable-viscosity assembly with deterministic block merge. |
| `diag.jl` | The matrix-free diagonal and the affine-constraint helpers. |
| `direct.jl` | The sparse direct saddle: stale-factor cache, iterative refinement and shift ladder. |
| `kop.jl` | The matrix-free velocity stiffness `K*v` and its precomputed eta field. |
| `operators.jl` | The quadrature divergence `B` and the compressibility block `Pcm`. |
| `selector.jl` | `linear=:auto`: choose direct or MG from saddle size and viscosity contrast. |
| `solve_const.jl` | Constant-viscosity Stokes solve and point evaluation. |
| `solve_var.jl` | Variable-viscosity Stokes solve: direct, Schur-CG, MG, lagged and auto. |
| `solve_var_core.jl` | The core variable-viscosity Stokes solve (every linear path). |
| `solve_var_fallback.jl` | The `linear = :auto` safety net that redoes an unacceptable MG solve. |
| `state.jl` | `SemStokes`: Q_p velocity, Q_{p-1} pressure, constraints and residual records. |
