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

## Geometry: the box path is a specialization, the mapped path a second kernel

Stage 0 cut 2 (docs/dev/GEOM_DESIGN.md §3, docs/dev/GEOM_PROGRESS.md) routed the
**assembled** Stokes operator through the geometry layer (`core/geometry.jl`):
the leaf kernel of `assembly_var.jl`, the load/traction face integral, the
divergence `B`/`Pcm` of `operators.jl`, the pressure mass/stiffness of
`assembly.jl` and the fixed-dof/net-flux/perimeter passes of
`solve_var_core.jl` now dispatch on the space's `geom`.  The dispatch rule is
the project's bitwise rule:

* the `CartesianBox` methods are the historical bodies, character for
  character (`2.0/hx` folded into the gradient, `0.25*hx*hy`, `0.5*h*gl_w`, the
  Cartesian wall assignment tables).  They must not be routed through
  `J = diag(hx/2, hy/2)`; the nine-hash dump (`GEOM_DESIGN` §4.3 F1–F7) is the
  gate;
* the mapped methods are new bodies that read the precomputed leaf/face metric
  (`quad_point`, `quad_measure`, `basis_grad`, `face_point`, `face_weight`,
  `face_normal`, `face_tangent`).  They are proved by G267 (affine exactness,
  traction and mapping independence) and G268 (the mapped kernel fed the box
  map reproduces the box operator);
* the mapped assembly is reachable only through `make_semstokes(qt, p;
  geom=…)`; cut 2 accepted `linear=:direct` only, and the driver refused mapped
  cards until the energy assembly/diagnostics cuts landed.

Stage 0 cut 5 (`docs/dev/GEOM_PROGRESS.md`) routes the **matrix-free** operator
and the multigrid layer through the geometry.  The rule is the same two-body
rule, with the historical matrix-free kernel kept as a separate family:

* `kop.jl` defines `AbstractSemQuadKOp` with two concrete operators:
  `SemQuadKOp` (the historical box operator: seven per-leaf scalars, today's
  kernels) and `SemQuadKOpMapped` (the mapped operator: the cached
  `det J`/`J^{-T}`/quadrature-point arrays of `core/geometry.jl`).  The box
  operator's construction and arithmetic are unchanged; its cache key
  (`SEMQUAD_KOP_CACHE`) gains the geometry stamp, so two geometries on one tree
  can never share an operator (`_semquad_kop_key` exposes the key for G271);
* `apply.jl` adds the mapped kernel bodies (`_semquad_kn_mapped!` and the
  leaf-blocked `_semquad_kn_blk_mapped!`/`..._store!` twins, used by the
  threaded/region V-cycle), which do the same four tensor-product contractions
  on the raw reference derivatives, mix them through `J^{-T}` at every
  quadrature point (the design's four weighted reference-space pieces,
  `GEOM_DESIGN` §2.6) and then run the box kernel's four accumulations.  The
  gather/scatter, the eta sampling/lane table, the threaded block partition and
  the scatter order are shared, so the mapped threaded V-cycle is bitwise the
  serial one (G271 (e));
* `diag.jl` adds the mapped element diagonal and element stiffness the MG
  hierarchy needs (`_semquad_diag_full!`, `_semquad_leaf_stiffness!`), so the
  Jacobi diagonal is the exact mapped `diag(Cu' K Cu)` and the coarsest dense
  operator is materialised from the mapped apply;
* the mapped scratch (three extra weighted-gradient planes in `SemQuadKWork`/
  `SemQuadKScratch`) is allocated **only** when the operator is mapped
  (`SemQuadKWork(op)` checks the operator type), so a box hierarchy build/apply
  allocates exactly the pre-cut-5 scratch.

A mapped space now supports `linear=:direct`, `:mg` (sparse and matrix-free
operators) and `:auto` (the selector's eta-contrast sampler covers the mapped
domain); `:schur_cg` and `:lagged` stay box-only (reference-matrix / stale-LU
gauge assumptions) and are refused with an actionable message.  G271 is the
mapped operator/hierarchy gate: mapped sparse `K` vs the mapped matrix-free
apply (≤1e-12), mapped `:mg` vs mapped `:direct` (≤1e-6), two geometries on one
tree giving two operators/caches/hierarchies and a zero-allocation warm apply,
the geometry-forgotten-key collision and its measured silent failure, the
logical coarse levels, and the driver's mapped linear paths.

## Diagnostics are published through module state

The `LAST_SEMSTOKES_*`, `LAST_AUTO_FALLBACK` and `SEMSTOKES_*` counters in
`state.jl`, `selector.jl` and `direct.jl` are **out-parameters**: the solver
writes what it did and the driver reads it to log, to guard the restart and to
fail loudly.  No arithmetic result of a solve reads them back.  The rule is that
anything in that family is write-once per call and read only by the caller after
the call; a new value that has to influence the next solve belongs in an argument,
not in this family.
