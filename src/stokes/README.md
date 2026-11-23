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

## The operator/multigrid cycle is deliberate

`kop.jl` and `apply.jl` (this directory) call into `mg/` -- `mg/levelcache.jl`
for the cached level data and `mg/vcycle.jl` for the preconditioner hook -- and
`mg/` calls back into the operator to apply it on every level.  An architecture
audit will flag that as a cycle between the operator layer and the solver layer.
It is the structure of **matrix-free multigrid**: the preconditioner is a
V-cycle *over the operator*, so it cannot be expressed through an interface that
hides the operator's per-level kernels without giving up the whole point of
being matrix-free.  deal.II's `MatrixFreeOperators`/`MGTransfer` pair and
ASPECT's `StokesMatrixFree`/`StokesPreconditioner` split have the same shape.

The cycle is therefore accepted, with two rules that keep it from spreading:
the calls out of this directory are confined to the four files named above, and
`solve_var_core.jl` remains the only place that decides *which* linear path runs.

## Diagnostics are published through module state

The `LAST_SEMSTOKES_*`, `LAST_AUTO_FALLBACK` and `SEMSTOKES_*` counters in
`state.jl`, `selector.jl` and `direct.jl` are **out-parameters**: the solver
writes what it did and the driver reads it to log, to guard the restart and to
fail loudly.  No arithmetic result of a solve reads them back.  The rule is that
anything in that family is write-once per call and read only by the caller after
the call; a new value that has to influence the next solve belongs in an argument,
not in this family.
