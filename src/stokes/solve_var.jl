"""
    solve_semquad_stokes_var(ss, qt, eta_fun, fx, fz, uDx, uDz)

Variable-viscosity quadtree Taylor-Hood Stokes solve.  The weak form is

    int 2 eta(x,y) eps(u):eps(v) - p div(v) - q div(u) = int f.v,

with `eta_fun(x,y)` evaluated at the Gauss points, affine Dirichlet data
`u = uD` on the domain boundary, the same hanging-node constraints as the
scalar space (lifted to both velocity components), and one pinned pressure dof.
The element matrices are assembled with `make_fedata(p, p-1)` Gauss quadrature,
matching the uniform-mesh `assemble_stokes` convention.  With
`linear=:schur_cg`, `schur_precond=:mass` uses the pressure mass matrix
(spectrally equivalent to the Schur complement) and `schur_maxiter` caps the
PCG iterations; `LAST_SEMSTOKES_RESIDUAL` records the true reduced-saddle
residual and `LAST_SEMSTOKES_SCHUR_ITERS` the iteration count.

`linear=:auto` (S6, production) resolves to `:direct` or `:mg` **before the
assembly** from the current saddle, so nobody has to remember the rule:
`saddle_dofs = 2*Nu + Np` (node counts) and the viscosity contrast sampled on
the actual `eta_fun` (`semquad_eta_contrast`, fixed 33x33 grid), fed to
[`semquad_choose_linear`](@ref).  An MG decision also forces
`mg_operator=:matrixfree` and `mg_threads=min(8, Threads.nthreads())`
(`semquad_auto_mg_threads`); `schur_precond` must be `:mass` (the default) or
the selector falls back to `:direct`.  The decision and its reason are in
`LAST_SEMSTOKES_AUTO_CHOICE`.  The rule is re-calibrated at `c7e0f55` after the
MG optimisation campaign (docs/dev/MG.md§16): any contrast takes MG from
20k dofs, contrast <= 100 from 2k, direct below that -- the old contrast-10
rung is gone because the measurements cannot tell the two low bands apart
(`benchmarks/bench_auto_crossover.jl` is the A/B; docs/dev/SOLVER.md §4.2/§4.4
carry the table and the old, pre-campaign numbers, which are superseded).
An `:auto` solve that resolves to `:direct` produces the **bitwise** direct
result; an `:auto` solve that resolves to `:mg` produces the same answer as an
explicit `linear=:mg` call (<=1e-10; the MG solve is not bitwise vs direct).
`solve_semquad_stokes_var` itself keeps `:direct` as its default (the
`run_semquad_simulation` default became `:auto` on 2026-11-13); pass `:auto`
explicitly here to use the selector.

`linear=:mg` is the scalable path for a **uniform** quadtree: geometric
multigrid for the velocity block (level coarsening, rediscretised eta) and a
block-triangular right preconditioner for GMRES.  `mg_cycles` V-cycles per
preconditioner application (default 1 since S3/MF-4: at p=4 L4 one V-cycle per
outer iteration needs 26 outer iterations but only 26 V-cycles in total, while
two need 21 but 42 -- measured 0.14 s vs 0.23 s for the whole saddle solve),
`mg_tol` the outer tolerance (1e-10).

`geo_u`/`geo_coef` add the strain-rate Jacobian `Kgeo` to the left-hand side
(Newton JVP; G156), and `dp_u`/`dp_pfield`/`dp_coef` add the pressure-derivative
block `(Kp)_{i,a} = int 2 (deta/dp) eps(u):eps(v_i) phi_a` returned as `Kp`
(`2N x np`) and `Kp_red = Cu' Kp Cp` (G163).  The Drucker-Prager saddle is then
non-symmetric: the momentum block is `Bt + Kp_red` while the constraint rows
have no pressure dependence.  Both options are only implemented for
`deviatoric=false`, and both default to "off", which keeps the assembly bit
identical to the historical one.
`elastic_stress` adds the 2b elastic force term: `nothing` (the default) leaves
the load vector untouched; a `(txx, tyy, txy)` tuple of nodal Q2 stress
components adds `-int tau : eps(v)` to the velocity rows, i.e. the weak form of
`div(tau)` as a body force.  The driver passes `a*tau^n`, the part of the
previous step's deviatoric stress not already carried by the effective viscosity
(`maxwell_force_stress`).  The same quadrature loop and basis as the `2 eta
eps(u):eps(v)` stiffness are used, so a uniform stress assembles to no load and
the term is exact for the Q2 interpolant of the nodal field (G247).

The direct branch checks its reduced residual and retries with a `1e-12 *
max|diag(S)|` diagonal shift when the unshifted LU is unstable (wall-refined
free-slip trees; `LAST_SEMSTOKES_REGULARIZED` records which path ran -- the flag
is only touched by the `:direct` branch).  `stokes_accept_tol` (default `Inf`)
turns that single retry into the shift ladder of
[`_shift_ladder_solve`](@ref): with a finite tolerance -- the production driver
passes its `stokes_tol` -- a rung that still misses it escalates to `1e-10` and
then `1e-8` relative shift, and a residual that survives the last rung is
polished with a few iterative-refinement steps against the unshifted operator.
`Inf` accepts the first rung unconditionally, which is the historical behaviour
bit for bit.  `stokes_lost_tol` (default `1.0`) is the second trigger of the
ladder: an unshifted relative residual above it means the factorisation did not
solve the system at all, and the diagonal shift then has to be chosen at the
scale of the near-null cluster rather than far above it
([`SEMSTOKES_TINY_SHIFTS`](@ref); docs/dev/CASE_REALISTIC.md §13.10).  `Inf`
disables that phase, and it never fires on a solve whose unshifted residual is
below `1.0`, so every answer that was already usable is bitwise unchanged.
The rung that produced the returned solution is in
`LAST_SEMSTOKES_SHIFT`, the refinement count in `LAST_SEMSTOKES_REFINE_ITERS`,
the per-rung residual history in `LAST_SEMSTOKES_LADDER`, and the two saddle
blocks separately in `LAST_SEMSTOKES_RESID_SPLIT` (docs/dev/CASE_REALISTIC.md §13).
`stokes_scaling` (default `:none`) selects the symmetric equilibration of the
reduced saddle used by the direct branch (`:equilibrated`;
docs/dev/BUG_LEDGER.md).  The scaling is an exact congruence
`S -> D*S*D`, `rhs -> D*rhs` with
`D = diag(1/sqrt(diag Kred), 2 max|Kred| / max(|Bt|,|Bs|))` and `x = D*y`: it
removes the viscosity and element-size scales from the operator, so the
wall-strip pressure family is lifted far above the round-off floor (the
measured smallest eigenvalue of the scaled operator is a momentum mode at
`4.2e-11` relative, instead of the unscaled `1.7e-14` pressure mode) and the
*unshifted* LU resolves it (measured 6.0e-10 on the AMR tree and 3.6e-8 on the
uniform-L5 control).  It is engaged only when
the historical unscaled unshifted solve misses its `1e-8`/`accept_tol` health
check, so a healthy solve is bitwise identical even with the option on; the
scaled answer is accepted when its true residual is within the 1e-6 accuracy
goal and is reported with `LAST_SEMSTOKES_LADDER_PHASE[] = :equilibrated` and
zero shift.  Otherwise the historical unscaled shift ladder runs unchanged as
the fallback.  `:none` is the historical path bit for bit, and the scaling is
never applied to the `:mg`/`:lagged`/`:schur_cg` paths.  The number of solves
answered by the scaled path is counted in `LAST_SEMSTOKES_SCALED[]` (cumulative;
not reset by a solve).  `run_semquad_simulation` exposes the keyword as
`stokes_scaling`, forwards it to every solve and stores it in the checkpoint's
`run_config` (docs/dev/BUG_LEDGER.md).
`mg_pressure_mode` selects the Schur approximation: `:auto` (default) uses the
1/eta-weighted pressure mass for contrast <= 100 and the three-factor w-BFBT
above it; `:mass`, `:stiffness`, `:lsc` and `:bfbt` force one.
`mg_pressure_precond` is the public selector of the pressure-block
approximate inverse: `:auto` (default) keeps the `mg_pressure_mode` resolution
above, `:bfbt` is the three-factor w-BFBT, `:mass` the *unweighted* pressure
mass matrix, `:scaled_mass` the `mg_pressure_weight`-scaled one (`:inv_eta` by
default), `:pcd` pressure convection-diffusion (`A_p^{-1} F_p M_p^{-1}`; with no
advection term in the momentum equation `C_p = 0`, so it reduces to the mass
inverse at three times the cost) and `:lsc` the least-squares commutator.  On
the 1024-leaf blob tree (16384x contrast) the measured outer iteration counts
are 39 (`:bfbt`) / 68 (`:scaled_mass`) / 239 (`:lsc`) / not converged
(`:mass`, `:pcd`), which is why `:auto` keeps BFBT there (docs/dev/SOLVER.md §13).
`mg_pressure_scale=0` means the mode's default (1 for mass, 0.1 for LSC,
0.4 for BFBT; measured scale curve in docs/dev/SOLVER.md §13).  `mg_restart`/`mg_maxit` control GMRES (default 100/500).  The outer loop is
**flexible** GMRES (FGMRES, `fgmres_restarted`) and stops on the **true**
reduced-saddle residual, never on the preconditioned one.
`mg_pressure_cache=true` (S3 stage A) caches the pressure-block factors across
Stokes solves, keyed by the tree fingerprint and invalidated when the sampled
viscosity moves by more than `mg_pressure_eta_tol` (default **1.0** since
S5-1; S3 used 1e-2 and then 0.5).  The BFBT build costs ~0.2-1.1 s on the
1024-leaf blob tree, and on the measured threshold curve (docs/dev/SOLVER.md
§13.3) the outer iteration count is flat from 1e-12 to `Inf` (39.7 -> 39.9),
so the threshold only trades build time against staleness: at **1.0** every
tree block is built exactly once (25 builds instead of 41 in the 25-frame blob
run), at the cost of accepting up to 100% eta drift inside a tree block.
`mg_pressure_eta_tol=Inf` is supported and means "never rebuild on eta drift"
(the tree fingerprint still forces a rebuild on every AMR change); it is only
safe for smoothly evolving viscosity -- **no phase change and no yield onset
inside a tree block** -- because nothing then bounds the staleness (§19.1).
`mg_operator=:matrixfree` (the default since S3/MF-4) builds every hierarchy
level from the leaf-local matrix-free apply of MF-1 and stores no velocity
operator matrix anywhere (44-79x faster hierarchy build, 1.26-1.48x faster
V-cycle at p=4); `:sparse` keeps the historical rediscretised/Galerkin `A[k]`.
`mg_vel_solver=:vcycle` (the default since S3/MF-6) solves the SPD velocity
block with the historical fixed `mg_cycles` V-cycles -- a *linear*
preconditioner, so the flexible outer GMRES reduces to ordinary GMRES there.
`:cg` is the MF-4 scheme that solves the velocity block inside the outer
iteration with CG on `Kred` preconditioned by the hierarchy (the
deal.II/ASPECT FGMRES + CG+MG scheme), controlled by `mg_vel_tol`/`mg_vel_maxit`
(1e-2/10).  Measured on the 1024-leaf blob tree: `:vcycle` needs 40 outer
iterations and 40 V-cycles, `:cg` needs 25 outer and 84 inner-CG (=V-cycle)
iterations, i.e. `:vcycle` is 33% faster per Stokes solve at 13 ms/V-cycle,
which is why it is the default; `:cg` remains available and is exercised by
G177.
`mg_smoother=:auto` uses damped Jacobi (faster per apply; the saddle iteration
count is dominated by the pressure block, and SGS did not reduce it on 2b's
16384x contrast).  `:sgs` is available for experiments; it gives a much better
velocity-block V-cycle but costs 2-3x per apply.  A Chebyshev smoother was
implemented and measured during the tuning pass; it diverges on the two
production trees (the reduced operator has near-null modes outside the
polynomial's window), so it is not shipped (docs/dev/MG.md).
`mg_coarse=:auto` rediscretises on uniform trees and uses the exact Galerkin
operator `P' A P` on non-uniform ones (same iteration count to a few percent,
but no coarse re-assembly per Picard iteration).
`mg_coarsen_layers`/`mg_coarsen_factor` control the h-coarsening of the
hierarchy (`_mg_coarsen_tree`): `1` is the one-tree-layer-per-step rule and
`mg_coarsen_factor` keeps merging until a step coarsens by at least that factor.
`mg_coarsen_factor=nothing` (the default) resolves to **1.5** on the
matrix-free path and `1.0` on the sparse path.  Measured on both production
states: 2.0 removes F's near-duplicate level (8 -> 6 levels, 70 -> 81
iterations, 5.42 -> 3.36 s at one thread) but on P1 it over-coarsens (8 -> 6
levels, 61 -> 177 iterations, 5.45 -> 8.03 s); 1.5 gets the F win (8 -> 6) and
keeps P1's 40 776-dof level (8 -> 7, 65 iterations with the top-heavy 3+3
schedule, 5.22 s).  Uniform and blob trees are a bitwise no-op at 1.5.
`mg_coarse_direct_max=n>0` stops the hierarchy at the finest level with at most
`n` dofs and factorises that level directly instead of coarsening further (also
measured: it costs more than it saves on the blob tree, default 0 = off).
`mg_smoothing=:auto` (the default) uses `mg_pre`/`mg_post` sweeps on every level
(`:uniform`, the historical 2+2) unless the hierarchy is top-heavy (its finest
coarsening step keeps more than half of the fine dofs): at p=2 the default then
rises to uniform 3+3, while at p>=4 it keeps the measured front-loaded
`:taper`.  `mg_omega` (`nothing` = the schedule's own default: 2/3 for
`:uniform`, 0.55 for `:taper`) overrides the damping.
Explicit `mg_pre`/`mg_post` vectors (coarse first) select any per-level
schedule; `mg_smoothing=:taper` keeps the historical `1, 2, 2, 4, 4, 8, ...`
shape with omega=0.55.  Measured: the 6-level F hierarchy is best at uniform
2+2 (81 iterations), the 7-level P1 hierarchy at uniform 3+3 (65); the old
taper gives 99 / 90 (docs/dev/MG.md).
A non-uniform tree is handled by hierarchical coarsening.

`mg_threads=1` (default) keeps the whole `linear=:mg` path serial and bitwise as
before.  `mg_threads=N>1` (S5-2) threads the **leaf-local** work of the
matrix-free V-cycle (the `semquad_apply_K!` kernel, the damped-Jacobi sweeps and
the residual evaluation) and of the outer matrix-free operator, using `N` fixed
contiguous leaf blocks and one scratch buffer per block; the scatter-add across
leaves that share boundary nodes stays serial and runs in leaf order, so the
threaded V-cycle and the whole solve are **bitwise identical** to the serial
ones (G190) -- the thread count changes the wall clock, not a single bit.  Only
`linear=:mg, mg_operator=:matrixfree` is threaded (the sparse path keeps its
serial `mul!`); a request larger than `Threads.nthreads()` is clamped with a
warning and a single-threaded process warns once and stays serial.  Measured on
uniform 4096-leaf (132k velocity dof) trees: 2.6x V-cycle and 1.9x whole-solve
speedup at 8 threads; at 1024 leaves the V-cycle gain is smaller (1.6x) and the
whole solve does not improve (docs/dev/SOLVER.md §19.2).

`assemble_threads=1` (default) assembles the Stokes leaves serially, exactly as
before (bitwise).  `assemble_threads=N>1` cuts the leaf list into `N` fixed
contiguous blocks, gives each block its own COO buffers, assembles the blocks
with `Threads.@threads` over the Julia thread pool and merges the block buffers
in block-index order.  The result is deterministic run to run.  Since block
order equals leaf order, the concatenated COO lists are the serial ones and
K/Kgeo/Kp/B are bitwise identical; only the load vector is assembled as a sum
of block sums, so `fv`/`fr`/the solution can differ at round-off (G181 checks
both the matrices and `fv` against serial to <=1e-15 relative).  If Julia has
only one thread, requesting >1 warns once and falls back to serial.

`linear=:auto` is additionally **safe**: if the selector picks `:mg` but the
solve is not acceptable -- the true reduced-saddle residual is above
`auto_fallback_tol` (keyword, default `1e-5`; `run_semquad_simulation` passes
its `stokes_tol`), or the MG path throws -- the same solve is redone with the
sparse direct path instead of aborting the run.  The event is recorded in
`LAST_AUTO_FALLBACK` (`count`, `last_reason`), warned once per process, and
`LAST_SEMSTOKES_AUTO_CHOICE` still reports the auto decision (`:mg`).  Explicit
`linear=:mg` is untouched and still fails loudly, and an `:auto` solve that
chose `:direct` is bitwise unchanged.  Measured trigger: the 8192-leaf
realistic case (7498 leaves, `saddle_dofs` ~72k), where the matrix-free MG
returned relres 4.33 before the exact constrained and BFBT diagonals; the
direct fallback recovered the solve.  After those fixes the same class of tree
converges (the card comment in `examples/case_realistic_2d.jl` and
docs/dev/SOLVER.md §4.4 carry the measurement), so the fallback remains a safety
net, not the expected path.

`assemble_only` selects what the call is allowed to build:

* `false` (default) -- solve;
* `true` -- assemble the full system and return it (`K`, `Kred`, `Bs`, `Bt`,
  `fr`, `gr`, ...), as every caller/Newton/`return_system` caller expects;
* `:mg` -- the **reduced matrix-free MG assembly**: return the same
  `Cu`/`Cp`/`B`/`Bc`/`Bm`/`fv`/`u0`/`p0`/`Bs`/`Bt` but with `K === nothing` and
  `Kred === nothing`, and without evaluating the viscosity closure inside the
  leaf assembly (it only feeds `K`, which is not formed).  `fr` uses the
  matrix-free `K u0` apply, i.e. the same value the production solve uses.
  This is what `_semquad_mg_geometry` needs to build the hierarchy: it reads
  `Cu` only.  `SEMSTOKES_K_ASSEMBLIES` counts every formation of the sparse
  `K`; an `assemble_only=:mg` call does not move it.  The mode requires
  `linear=:mg, mg_operator=:matrixfree` and is bitwise for every object it
  returns (docs/dev/MG.md, G220).

**Boundary-condition keywords.** The two positional closures `uDx`/`uDz` are
the historical global Dirichlet data: every wall's *normal* component is taken
from `uDx` on an x-wall (`x = const`) and from `uDz` on a y-wall
(`y = const`), and a `:rigid` wall takes its *tangential* component from the
other closure.  The keywords refine that value per wall, in the order
`(xminus, xplus, yminus, yplus)`:

* `wall_bc` (`:same`/`:free`/`:rigid`/`:open`) selects a wall's condition;
  `:same` follows `bc`.  The normal component is Dirichlet on a `:free` and on a
  `:rigid` wall; an `:open` wall (stage 3) has **no** Dirichlet component on
  either component -- the natural traction condition applies, optionally with
  the `wall_t`/`wall_tn` data.
* `wall_u` is the wall's *tangential* velocity, and only exists on a `:rigid`
  wall; a value on a (resolved) `:free` or `:open` wall is an `ArgumentError`,
  because the tangential component there is free and the value would be silently
  ignored (the pre-G258 behaviour).
* `wall_un` is the wall's *normal* velocity: `u_x` on an x-wall and `u_z` on a
  y-wall, in the positive coordinate direction -- the value of the wall's
  normal component, exactly like the legacy closures, **not** the outward flux
  `u·n`.  `nothing` falls back to the global closure for that wall, which
  keeps the historical path bit-identical (G258).  A value on an `:open` wall is
  an `ArgumentError` (there is no Dirichlet component to refine).
* `wall_t` is the wall's *tangential traction*, added to the Stokes RHS as
  `∫ tbar·v dS` on the **unreduced** load vector (so `Cu'` projects it onto the
  admissible test space; it is never Dirichlet data).  `tbar = σ·n` is the
  **total** Cauchy traction and the value is its global component in the
  positive coordinate direction -- `t_x` on a y-wall and `t_z` on an x-wall,
  the same convention as `wall_u`/`wall_un`, so there is no outward-normal sign
  table to get wrong.  On an axis-aligned wall the tangential component is
  pressure-free (`t_x = σ_xy`, `t_z = σ_zy`), so this is the exact
  `2ηε(u)·n` shear traction.  It requires a wall whose tangential component is
  free (a `:free` or `:open` wall); on a `:rigid` wall the tangential velocity
  is Dirichlet, so a traction there would be dropped by the constraints -- a
  silent no-op, refused with an `ArgumentError` (docs/dev/BC_DESIGN.md §3.2).
  `nothing` is zero traction, so the default path is bit-identical (G259/G260).
* `wall_tn` is the wall's *normal traction* (stage 3), the same data term for
  the wall's normal component: the global `t_x` on an x-wall and `t_z` on a
  y-wall, no outward-normal sign table.  It carries the pressure
  (`t_n = -p + 2ηε_nn`), so it is accepted only on an `:open` wall (free normal
  velocity) and it is the data that fixes the pressure level: because
  `tbar = σ·n` uses the **outward** normal, a pressure `p0` on an open boundary
  is `wall_tn = -p0` on the top wall and `+p0` on the bottom wall.  It is the
  **dynamic** pressure traction, not a lithostatic value.  `nothing` is zero
  traction (the natural condition), so the default path is bit-identical
  (G261/G262/G263).

**Open walls and the pressure gauge.** The nodal pressure pin is exact only
while the normal velocity is Dirichlet on the whole boundary (the constant
pressure is then in the null space of the reduced gradient, and the dropped
continuity row is a linear combination of the others).  With any `:open` wall
that is no longer true, so the gauge is derived: the solver pins one pressure
dof **iff no wall is `:open`** and otherwise leaves the pressure unpinned; the
resolved decision is `LAST_SEMSTOKES_PRESSURE_GAUGE` (`:pin` / `:none`).  The
open configurations the solver accepts are the well-posed ones: at least one
x-wall and one y-wall must keep a Dirichlet normal component (otherwise a rigid
translation is left admissible -- refused), and all four walls `:open` (the pure
traction problem, which needs a velocity gauge and net force/torque-compatible
data) is refused.  `linear=:direct` (with its shift ladder), `linear=:mg` and
`linear=:auto` are verified without the pin; `:schur_cg`/`:lagged` are refused
with an open wall because those paths assume the closed-boundary gauge.

**Net-flux compatibility.** When no wall is `:open` the prescribed normal data
must satisfy `∫∂Ω u·n dS = 0`.  An unbalanced configuration (e.g. a through-flow
prescribed on one x-wall only) raises an `ArgumentError` that reports the
measured net flux, instead of solving a discrete problem whose continuity
equations are inconsistent.  With any `:open` wall the free normal velocity
absorbs the imbalance, so the check is skipped.  A corner dof belongs to two
walls and the walls are written in the order above, so a later wall owns the
corner: in particular, a rigid top/bottom wall's tangential value overrides the
side walls' normal value at its two corners (docs/dev/BC_DESIGN.md §1.4;
G258 (d)).

**Mapped geometry (stage 0 cut 2).**  When `ss` was built with a non-box
geometry (`make_semstokes(qt, p; geom=AffineBox(...))`), the assembled Stokes
operator is routed through the geometry's Jacobian: the leaf kernel takes
`(x, y, w, gx, gz)` from the leaf metric, the traction load integrates
`t.n`/`t.tau` with the mapped outward normal and positive tangent, the
fixed-dof loop evaluates the Dirichlet closures at the mapped wall-node
positions, and the net-flux check/perimeter use the mapped face measure.  Cut 2
accepts `linear=:direct` only (`:mg`, `:schur_cg` and `:lagged` still run the
historical box kernels / box reference-matrix assumptions and are refused with
an actionable message; `:mg` is cut 5), `:rigid`/`:open` walls only (a `:free`
wall or `wall_un` prescribe a *global component*, which is the wall-normal
direction only on an axis-aligned wall; the general normal-velocity row is the
annulus BC stage), and it refuses `geo_u`/`dp_u`/`elastic_stress` and the
Cartesian density multiplier (their leaves are box-only until a later cut).
`wall_t`/`wall_tn` on a mapped geometry are the wall-local components `t.tau`
and `t.n` of the total traction (GEOM_DESIGN §6.3), not the box's global
components.  The box path is unchanged and bitwise: the box methods are the
historical bodies (GEOM_DESIGN §4), and the mapped kernel is a separate body
(G267 exactness, G268 cross-path consistency).
"""
function solve_semquad_stokes_var(ss::SemStokes, qt::Quadtree,
                                  eta_fun::Function,
                                  fx::Function, fz::Function,
                                  uDx::Function, uDz::Function; kwargs...)
    # `linear=:auto` safety net (docstring above).  `assemble_only` never solves,
    # so it has no residual to judge; explicit paths are forwarded untouched.
    is_auto = get(kwargs, :linear, :direct) === :auto
    has_ftol = haskey(kwargs, :auto_fallback_tol)
    ftol = has_ftol ? Float64(kwargs[:auto_fallback_tol]) : 1.0e-5
    core = has_ftol ?
           (; (k => v for (k, v) in pairs(kwargs)
               if k !== :auto_fallback_tol)...) :
           kwargs
    if !is_auto
        return _solve_semquad_stokes_var_core(ss, qt, eta_fun, fx, fz, uDx, uDz;
                                              core...)
    end
    # Reset the selector record so a throw *before* the resolution (argument
    # validation) cannot be mistaken for an MG failure of this solve.
    LAST_SEMSTOKES_AUTO_CHOICE[] = _SEMSTOKES_AUTO_CHOICE_NONE
    res = try
        _solve_semquad_stokes_var_core(ss, qt, eta_fun, fx, fz, uDx, uDz; core...)
    catch err
        err isa InterruptException && rethrow()
        ch = LAST_SEMSTOKES_AUTO_CHOICE[]
        ch.choice === :mg || rethrow()
        return _solve_semquad_stokes_var_fallback(
            ss, qt, eta_fun, fx, fz, uDx, uDz, core, ch,
            "the MG solve threw $(typeof(err)): $(sprint(showerror, err))")
    end
    get(core, :assemble_only, false) !== false && return res
    ch = LAST_SEMSTOKES_AUTO_CHOICE[]
    if ch.choice === :mg && !(LAST_SEMSTOKES_RESIDUAL[] <= ftol)  # also NaN
        return _solve_semquad_stokes_var_fallback(
            ss, qt, eta_fun, fx, fz, uDx, uDz, core, ch,
            @sprintf("true residual %.3e > auto_fallback_tol %.1e (saddle_dofs=%d, contrast=%.4g)", LAST_SEMSTOKES_RESIDUAL[],
                     ftol, ch.saddle_dofs, ch.contrast))
    end
    return res
end

# Redo an unacceptable `linear=:auto` MG solve with the explicit `:direct` path
