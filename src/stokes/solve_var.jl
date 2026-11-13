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
  `fr`, `gr`, ...), as every DWR/Newton/`return_system` caller expects;
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
# (same arguments, same state, so the answer is the direct answer) and count it.
function _solve_semquad_stokes_var_fallback(ss::SemStokes, qt::Quadtree,
                                            eta_fun::Function, fx::Function,
                                            fz::Function, uDx::Function,
                                            uDz::Function, core, ch, why::String)
    prev = LAST_AUTO_FALLBACK[]
    LAST_AUTO_FALLBACK[] = (count=prev.count + 1, last_reason=why)
    @warn "linear=:auto resolved to :mg but that solve was not acceptable; " *
          "redoing it with linear=:direct ($why).  Explicit linear=:mg is not " *
          "affected; LAST_AUTO_FALLBACK[] counts these solves." maxlog=1
    direct = (; (k => v for (k, v) in pairs(core) if k !== :linear)...,
                 linear=:direct)
    res = _solve_semquad_stokes_var_core(ss, qt, eta_fun, fx, fz, uDx, uDz;
                                         direct...)
    # Report the selector's decision; the counter records which path answered.
    LAST_SEMSTOKES_AUTO_CHOICE[] = ch
    return res
end

function _solve_semquad_stokes_var_core(ss::SemStokes, qt::Quadtree,
                                  eta_fun::Function,
                                  fx::Function, fz::Function,
                                  uDx::Function, uDz::Function;
                                  mass_rho::Union{Nothing,Function}=nothing,
                                  mass_form::Symbol=:anelastic,
                                  mass_drho::Union{Nothing,Function}=nothing,
                                  compress_block::Union{Nothing,AbstractMatrix}=nothing,
                                  deviatoric::Bool=false,
                                  geo_u::Union{Nothing,Vector{Float64}}=nothing,
                                  geo_coef::Union{Nothing,Function}=nothing,
                                  geo_spd_safety::Float64=0.0,
                                  dp_u::Union{Nothing,Vector{Float64}}=nothing,
                                  dp_pfield::Union{Nothing,Vector{Float64}}=nothing,
                                  dp_coef::Union{Nothing,Function}=nothing,
                                  bc::Symbol=:rigid, linear::Symbol=:direct,
                                  stokes_accept_tol::Float64=Inf,
                                  stokes_lost_tol::Float64=SEMSTOKES_LOST_TOL,
                                  stokes_scaling::Symbol=:none,
                                  schur_maxiter::Int=200, schur_precond::Symbol=:mass,
                                  mg_cycles::Int=1, mg_tol::Float64=1.0e-10,
                                  mg_pressure_scale::Float64=0.0,
                                  mg_pressure_weight::Symbol=:inv_eta,
                                  mg_pressure_mode::Symbol=:auto,
                                  mg_pressure_precond::Symbol=:auto,
                                  mg_pressure_cache::Bool=true,
                                  mg_pressure_eta_tol::Float64=SEMQUAD_PRESSURE_DEFAULT_ETA_TOL,
                                  mg_pressure_bfbt_d::Symbol=:auto,
                                  mg_pressure_bfbt_s2::Symbol=:free,
                                  mg_pressure_bfbt_s1::Symbol=:chol,
                                  mg_pressure_bfbt_s1_solve::Symbol=:chol,
                                  mg_pressure_bfbt_s1_tol::Float64=1.0e-2,
                                  mg_pressure_bfbt_s1_maxit::Int=50,
                                  mg_pressure_bfbt_s1_minwork::Int=0,
                                  mg_pressure_bfbt_s1_chunk::Int=0,
                                  mg_smoother::Symbol=:auto,
                                  mg_coarse::Symbol=:auto,
                                  mg_coarsen_layers::Int=1,
                                  mg_coarsen_factor::Union{Nothing,Float64}=nothing,
                                  mg_coarse_direct_max::Int=0,
                                  mg_smoothing::Symbol=:auto,
                                  mg_pre::Union{Integer,AbstractVector{<:Integer}}=2,
                                  mg_post::Union{Integer,AbstractVector{<:Integer}}=2,
                                  mg_omega::Union{Nothing,Float64}=nothing,
                                  mg_operator::Symbol=:matrixfree,
                                  mg_vel_solver::Symbol=:vcycle,
                                  mg_vel_tol::Float64=1.0e-1,
                                  mg_vel_maxit::Int=10,
                                  mg_restart::Int=100, mg_maxit::Int=500,
                                  mg_threads::Int=1,
                                  mg_diag::Symbol=:constrained,
                                  assemble_threads::Int=1,
                                  return_system::Bool=false,
                                  assemble_only::Union{Bool,Symbol}=false)
    bc in (:free, :rigid) || throw(ArgumentError("bc must be :free or :rigid"))
    stokes_scaling in SEMSTOKES_SCALINGS ||
        throw(ArgumentError("stokes_scaling must be :none or :equilibrated"))
    (geo_u === nothing) == (geo_coef === nothing) ||
        throw(ArgumentError("geo_u and geo_coef must be given together"))
    if geo_u !== nothing
        length(geo_u) == 2 * ss.su.N ||
            throw(ArgumentError("geo_u has $(length(geo_u)) entries, expected $(2 * ss.su.N)"))
        deviatoric &&
            throw(ArgumentError("the geometric strain-rate Jacobian is only " *
                                "implemented for deviatoric=false"))
    end
    # Pressure-derivative (Drucker-Prager) Jacobian block `Kp`: the momentum
    # residual is `K(eta(T, edot(u), p)) u - B'p - f`, so its pressure
    # derivative is
    #     (Kp)_{i,a} = int 2 (deta/dp) eps(u):eps(v_i) phi_a
    # in the velocity rows and nothing in the constraint rows.  The resulting
    # saddle system is non-symmetric (velocity rows <-> pressure columns), which
    # the direct LU handles.  `dp_u` is the velocity whose strain rate enters
    # `deta/dp`, `dp_pfield` the nodal pressure field, `dp_coef(x,y,edot,p)` the
    # derivative itself (e.g. `semquad_deta_dp` wrapped with the temperature).
    if dp_u !== nothing || dp_pfield !== nothing || dp_coef !== nothing
        (dp_u !== nothing && dp_pfield !== nothing && dp_coef !== nothing) ||
            throw(ArgumentError("dp_u, dp_pfield and dp_coef must be given " *
                                "together"))
        length(dp_u) == 2 * ss.su.N ||
            throw(ArgumentError("dp_u has $(length(dp_u)) entries, expected $(2 * ss.su.N)"))
        length(dp_pfield) == ss.sp.N ||
            throw(ArgumentError("dp_pfield has $(length(dp_pfield)) entries, " *
                                "expected $(ss.sp.N)"))
        deviatoric &&
            throw(ArgumentError("the pressure-derivative Jacobian block is " *
                                "only implemented for deviatoric=false"))
    end
    linear in (:direct, :schur_cg, :mg, :lagged, :auto) ||
        throw(ArgumentError("linear must be :direct, :schur_cg, :mg, :lagged " *
                            "or :auto"))
    (assemble_only === false || assemble_only === true || assemble_only === :mg) ||
        throw(ArgumentError("assemble_only must be false (solve), true (assemble " *
                            "everything) or :mg (the reduced matrix-free MG " *
                            "geometry: no velocity stiffness K, no Kred)"))
    mg_cycles >= 1 || throw(ArgumentError("mg_cycles must be >= 1"))
    mg_tol > 0.0 || throw(ArgumentError("mg_tol must be > 0"))
    mg_pressure_scale >= 0.0 ||
        throw(ArgumentError("mg_pressure_scale must be >= 0 (0 = auto)"))
    mg_pressure_weight in (:inv_eta, :eta, :none) ||
        throw(ArgumentError("mg_pressure_weight must be :inv_eta, :eta or :none"))
    mg_pressure_mode in (:auto, :mass, :stiffness, :lsc, :bfbt) ||
        throw(ArgumentError("mg_pressure_mode must be :auto, :mass, :stiffness, :lsc or :bfbt"))
    mg_pressure_precond in (:auto, :bfbt, :mass, :scaled_mass, :pcd, :lsc) ||
        throw(ArgumentError("mg_pressure_precond must be :auto, :bfbt, :mass, " *
                            ":scaled_mass, :pcd or :lsc"))
    mg_pressure_eta_tol > 0.0 ||
        throw(ArgumentError("mg_pressure_eta_tol must be > 0 (Inf = never " *
                            "rebuild on eta drift; a tree change still rebuilds)"))
    mg_pressure_bfbt_d in (:auto, :leaf, :exact, :matrix) ||
        throw(ArgumentError("mg_pressure_bfbt_d must be :auto, :leaf, :exact " *
                            "or :matrix"))
    mg_pressure_bfbt_s2 in (:free, :matrix) ||
        throw(ArgumentError("mg_pressure_bfbt_s2 must be :free or :matrix"))
    mg_pressure_bfbt_s1 in (:chol, :mass) ||
        throw(ArgumentError("mg_pressure_bfbt_s1 must be :chol or :mass"))
    mg_pressure_bfbt_s1_solve in (:chol, :level, :cg_diag, :cg_ichol) ||
        throw(ArgumentError("mg_pressure_bfbt_s1_solve must be :chol, :level, " *
                            ":cg_diag or :cg_ichol"))
    mg_pressure_bfbt_s1_tol > 0.0 ||
        throw(ArgumentError("mg_pressure_bfbt_s1_tol must be > 0"))
    mg_pressure_bfbt_s1_maxit >= 1 ||
        throw(ArgumentError("mg_pressure_bfbt_s1_maxit must be >= 1"))
    mg_smoother in (:auto, :jacobi, :sgs) ||
        throw(ArgumentError("mg_smoother must be :auto, :jacobi or :sgs"))
    mg_coarse in (:auto, :rediscretize, :galerkin) ||
        throw(ArgumentError("mg_coarse must be :auto, :rediscretize or :galerkin"))
    mg_coarsen_layers >= 1 ||
        throw(ArgumentError("mg_coarsen_layers must be >= 1"))
    (mg_coarsen_factor === nothing ||
     (mg_coarsen_factor >= 1.0 && isfinite(mg_coarsen_factor))) ||
        throw(ArgumentError("mg_coarsen_factor must be >= 1 (or `nothing` = " *
                            "the matrix-free production default 1.5)"))
    mg_coarse_direct_max >= 0 ||
        throw(ArgumentError("mg_coarse_direct_max must be >= 0"))
    mg_smoothing in (:auto, :uniform, :taper) ||
        throw(ArgumentError("mg_smoothing must be :auto, :uniform or :taper"))
    _mg_sw_ok(v) = v isa Integer ? v >= 0 : all(x -> x >= 0, v)
    _mg_sw_ok(mg_pre) || throw(ArgumentError("mg_pre must be >= 0"))
    _mg_sw_ok(mg_post) || throw(ArgumentError("mg_post must be >= 0"))
    (mg_omega === nothing || 0.0 < mg_omega < 1.0) ||
        throw(ArgumentError("mg_omega must be in (0,1) or `nothing`"))
    mg_operator in (:sparse, :matrixfree) ||
        throw(ArgumentError("mg_operator must be :sparse or :matrixfree"))
    # Production default of the h-coarsening step.  `coarsen_factor=2.0` was
    # measured on the F tree (1.5x end to end, docs/dev/MG.md) but on
    # P1 it collapses the hierarchy from 8 to 6 levels and costs 177 outer
    # iterations instead of 61 (8.03 s against 5.45 s, measured at 1e-6).  The
    # shipped default is therefore 1.5: it still removes F's 1.7 %-coarsening
    # near-duplicate level (F: 8 -> 6 levels) but keeps P1's 40 776-dof level
    # (P1: 8 -> 7 levels), and wins on both states (F 5.42 -> 3.36 s,
    # P1 5.45 -> 5.12 s at one thread).  `mg_coarsen_factor=1.0` restores the
    # historical rule and `2.0` remains selectable; uniform/blob trees are a
    # bitwise no-op at 1.5 (their steps already coarsen by >=2x).
    coarsen_factor = mg_coarsen_factor === nothing ?
                     (mg_operator === :matrixfree ? 1.5 : 1.0) :
                     mg_coarsen_factor
    mg_vel_solver in (:cg, :vcycle) ||
        throw(ArgumentError("mg_vel_solver must be :cg or :vcycle"))
    mg_vel_tol > 0.0 || throw(ArgumentError("mg_vel_tol must be > 0"))
    mg_vel_maxit >= 1 || throw(ArgumentError("mg_vel_maxit must be >= 1"))
    mg_restart >= 5 || throw(ArgumentError("mg_restart must be >= 5"))
    mg_maxit >= 1 || throw(ArgumentError("mg_maxit must be >= 1"))
    mg_threads >= 1 || throw(ArgumentError("mg_threads must be >= 1"))
    mg_diag in (:constrained, :lumped) ||
        throw(ArgumentError("mg_diag must be :constrained or :lumped"))
    mass_form in (:anelastic, :rho_div, :boussinesq) ||
        throw(ArgumentError("mass_form must be :anelastic, :rho_div or :boussinesq"))
    schur_precond in (:none, :mass) ||
        throw(ArgumentError("schur_precond must be :none or :mass"))
    assemble_threads >= 1 ||
        throw(ArgumentError("assemble_threads must be >= 1"))
    # S6: resolve `linear=:auto` **before** the assembly, so an MG decision
    # never pays for or stores the sparse velocity operator.  The decision uses
    # only O(1) node counts and the fixed 33x33-grid eta sample (measured cost
    # in docs/dev/SOLVER.md §20); an explicit `:direct`/`:mg` call skips this
    # block entirely and runs exactly as before (bitwise for `:direct`).
    if linear === :auto
        auto_choice, auto_threads = _semquad_resolve_auto(ss, eta_fun;
                                                          schur_precond=schur_precond)
        linear = auto_choice
        if linear === :mg
            # The selector is calibrated on the matrix-free operator, and
            # `mg_threads=min(8, Threads.nthreads())` (S5-2 measured the gain).
            mg_operator = :matrixfree
            mg_threads = auto_threads
        end
    else
        # An `assemble_only` call never solves, so it must not overwrite the
        # `:auto` decision of the last real solve: the MG hierarchy builder and
        # the DWR/Newton probes assemble with `linear=:direct, assemble_only=true`
        # *inside* an auto MG solve, and recording `:none` there would hide the
        # selector's decision (it broke the auto diagnostics until S7).
        assemble_only !== false ||
            (LAST_SEMSTOKES_AUTO_CHOICE[] =
                 (choice=:none,
                  reason=get(_SEMSTOKES_AUTO_OFF_REASONS, linear,
                             "linear not :auto (auto not requested)"),
                  saddle_dofs=0, contrast=1.0, samples=0, sample_ms=0.0,
                  mg_threads=0))
    end
    # BFBT weight: the exact restricted diagonal
    # `diag(Cu' M_u(sqrt(eta)) Cu)` is the shipped default on the matrix-free
    # path (docs/dev/MG.md).  It is a zero-per-apply-cost change
    # that removes 10 outer iterations on `F` (81 -> 71) and leaves `P1` at 65;
    # its leaf-local construction is cheaper than the historical lumped `:leaf`
    # diagonal at 8 threads and the same at 1 (both are dominated by the
    # leaf/quadrature-point eta evaluations).  `:leaf` is kept as the A/B and
    # `:matrix` as the sparse reference for G226; the sparse-operator MG path
    # keeps `:leaf` because it is not the measured production path.  Resolved
    # after the `linear=:auto` block, which may switch `mg_operator`.
    bfbt_d = mg_pressure_bfbt_d === :auto ?
             (mg_operator === :matrixfree ? :exact : :leaf) :
             mg_pressure_bfbt_d
    # `assemble_only=:mg` is the *reduced* assembly of the matrix-free MG
    # geometry: the caller (the hierarchy builder) only reads `Cu`, so the
    # velocity stiffness `K`, the COO triplets behind it and `Kred` are not
    # formed at all.  It is defined only for the matrix-free MG path (the sparse
    # operator path and the direct path need `K`/`Kred`); it is opt-in and
    # returns the same `Cu`/`Cp`/`B`/`fv` bit for bit.
    kfree = assemble_only === :mg
    kfree && return_system &&
        throw(ArgumentError("assemble_only=:mg does not form K/Kred, so it is " *
                            "incompatible with return_system=true"))
    kfree && !(linear === :mg && mg_operator === :matrixfree) &&
        throw(ArgumentError("assemble_only=:mg is the reduced matrix-free MG " *
                            "assembly; it requires linear=:mg with " *
                            "mg_operator=:matrixfree (got linear=$linear, " *
                            "mg_operator=$mg_operator)"))
    LAST_SEMSTOKES_VEL_CG_ITERS[] = 0
    p = ss.p; q = ss.q
    su = ss.su; sp = ss.sp
    nu = su.N; np = sp.N
    Nu2 = 2 * nu
    fd = make_fedata(p, q)
    n = p + 1
    nv = n * n
    npq = (q + 1)^2
    # S3 stage C: `linear=:mg, mg_operator=:matrixfree` never needs the sparse
    # velocity operator.  The right-hand side and the outer FGMRES operator are
    # built from the leaf-local apply (`semquad_apply_K!` / `_apply_Kred!`), so
    # the `2n^2 x 2n^2` dense leaf blocks and their `4 nv^2` triplets are never
    # formed.  Everything that still needs `K` (return_system, the Newton
    # geometric/pressure-derivative blocks) keeps it, and so does a plain
    # `assemble_only=true`; only the reduced `assemble_only=:mg` geometry call
    # joins the solve path in the `K`-free branch.  On that branch the leaf
    # assembly also skips the viscosity closure (its `etaq` only enters `K` and
    # the two Jacobian blocks), which is 0.15 s of the F setup at one thread --
    # see `SemQuadAsmCfg.mf_path`.
    mf_path = linear === :mg && mg_operator === :matrixfree &&
              (assemble_only === false || kfree) &&
              !return_system && geo_u === nothing && dp_u === nothing
    pleaves = Set(sp.leaves)
    # Deviatoric stress 2 eta (eps - (1/3) div(u) I): the volumetric part of
    # 2 eta eps:eps(v) loses 2/3 eta div(u) div(v).  ASPECT's ALA always uses
    # this form (manual eq. 20); without it a compressible (div u = Di u_z)
    # flow is too stiff.  `deviatoric=false` keeps the historical operator.
    a2 = deviatoric ? 4.0 / 3.0 : 2.0
    # S4 stage 1: leaf assembly.  `assemble_threads=1` (default) is the
    # historical serial path, bit for bit; >1 cuts the leaves into fixed
    # contiguous blocks, assembles them in parallel and merges the block COO
    # buffers in block order.  The threaded result is deterministic run to run
    # but is *not* bitwise equal to the serial one (different summation order),
    # which is why the default stays 1.
    asm_cfg = SemQuadAsmCfg(a2, deviatoric, mf_path, geo_u, geo_coef,
                            geo_spd_safety, dp_u, dp_pfield, dp_coef)
    asm = _semquad_assemble_blocks(
        _semquad_asm_thread_count(assemble_threads), asm_cfg, su, sp,
        eta_fun, fx, fz, fd, pleaves)
    IK = asm.IK; JK = asm.JK; VK = asm.VK
    IB = asm.IB; JB = asm.JB; VB = asm.VB
    IG = asm.IG; JG = asm.JG; VG = asm.VG
    IP = asm.IP; JP = asm.JP; VP = asm.VP
    fv = asm.fv

    if mf_path
        K = nothing
    else
        SEMSTOKES_K_ASSEMBLIES[] += 1
        K = sparse(IK, JK, VK, Nu2, Nu2)
    end
    Kgeo = geo_u === nothing ? nothing : sparse(IG, JG, VG, Nu2, Nu2)
    Kp = dp_u === nothing ? nothing : sparse(IP, JP, VP, Nu2, np)
    Bref = sparse(IB, JB, VB, np, Nu2)
    # Mass constraint with a stratified reference density.  ASPECT's ALA mass
    # equation is div(rho u) = 0, whose Galerkin form is
    #     int q (rho div u + u_z drho/dy) dV = 0,
    # **not** `int q rho div u = 0`: the latter only enforces div u = 0 and, with
    # upwellings, breaks the exact cancellation between the advection of the
    # reference state and the adiabatic heating term (King 2010 then cools and
    # its energy budget does not close).  The momentum pressure term stays the
    # unweighted -int p div v, so Bm = Bref while Bc carries the density.
    # `mass_form=:rho_div` keeps the historical single rho-weighted matrix for
    # both blocks so the old behaviour stays regression-testable.
    # `mass_form=:boussinesq` is the **TALA** continuity equation `div u = 0`:
    # the reference density enters the buoyancy only, not the mass conservation
    # (ASPECT's `truncated anelastic liquid approximation`).  It is the
    # apples-to-apples TALA counterpart of the ALA form for the same reference
    # state, which is what docs/dev/PHYSICS.md §10.3 measures; it is off unless
    # requested, so the default path is unchanged.
    Bm = Bref
    Bc = Bref
    if mass_rho !== nothing && mass_form !== :boussinesq
        if mass_form === :anelastic
            # derivative of the density profile at the quadrature points
            drho = mass_drho === nothing ? (y -> begin
                hd = 1.0e-6
                yp = min(y + hd, 1.0); ym = max(y - hd, 0.0)
                (mass_rho(yp) - mass_rho(ym)) / (yp - ym)
            end) : mass_drho
            Bc = semquad_stokes_B(ss, qt; rho_fun=mass_rho, drho_fun=drho)
        else
            Bc = semquad_stokes_B(ss, qt; rho_fun=mass_rho)
            Bm = Bc
        end
        linear === :schur_cg && mass_form === :anelastic &&
            throw(ArgumentError("mass_form=:anelastic makes the saddle system " *
                                "nonsymmetric (Bm != Bc'); use linear=:direct"))
    end

    # affine Dirichlet data on the domain boundary
    fixed = Dict{Int,Float64}()
    for lf in su.leaves
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = su.elem[lf]
        for dir in (:xminus, :xplus, :yminus, :yplus)
            has_no_face_neighbor(qt, lf, dir) || continue
            if dir === :xminus || dir === :xplus
                iF = dir === :xminus ? 1 : n
                xf = dir === :xplus ? x0 + hx : x0
                for j in 1:n
                    yv = y0 + 0.5 * (su.xi[j] + 1.0) * hy
                    g = uids[(j - 1) * n + iF]
                    fixed[g] = uDx(xf, yv)
                    bc === :rigid && (fixed[nu + g] = uDz(xf, yv))
                end
            else
                jF = dir === :yminus ? 1 : n
                yf = dir === :yplus ? y0 + hy : y0
                for i in 1:n
                    xv = x0 + 0.5 * (su.xi[i] + 1.0) * hx
                    g = uids[(jF - 1) * n + i]
                    fixed[nu + g] = uDz(xv, yf)
                    bc === :rigid && (fixed[g] = uDx(xv, yf))
                end
            end
        end
    end
    cons_u = semquad_constraints(su, qt)
    cons2 = Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
    for (c, (ms, ws)) in cons_u
        cons2[c] = (ms, ws)
        cons2[nu + c] = (nu .+ ms, ws)
    end
    u0, Cu, _ = _affine_constraint(Nu2, cons2, fixed)
    cons_p = semquad_constraints(sp, qt)
    pf = Dict{Int,Float64}()
    pfree0 = setdiff(1:np, collect(keys(cons_p)))
    isempty(pfree0) || (pf[first(pfree0)] = 0.0)
    p0, Cp, _ = _affine_constraint(np, cons_p, pf)

    CpB = Cp' * Bc * Cu
    # S3 stage C: on the fully matrix-free MG path there is no sparse `K` and no
    # `Kred = Cu' K Cu`.  `fr = Cu' (fv - K u0 + Bm' p0)` needs `K u0` on the
    # *full* velocity vector, which the leaf-local apply gives directly, and the
    # outer operator becomes `Cu' K (Cu v)` (`_semquad_apply_Kred!`).
    mfop = nothing
    mfapplyK = nothing
    mfpar = nothing
    if mf_path
        mfop = semquad_kop(ss, qt; deviatoric=deviatoric)
        # sampled once, then laid out for the lane kernel (docs/dev/MG.md
        # §15): the leaf-major field plus its block-major lane table
        mfetav = semquad_eta_lanes(mfop, semquad_eta_field(mfop, eta_fun))
        mfwork = SemQuadKWork(mfop.n, mfop.nq, Nu2)
        Ku0 = zeros(Float64, Nu2)
        _semquad_apply_K_raw!(Ku0, mfop, mfetav, u0, mfwork)
        fr = Cu' * (fv - Ku0 + Bm' * p0)
        # S5-2: the outer operator (FGMRES residual and the matrix-free BFBT
        # `S2`) uses the same threaded leaf blocks as the V-cycle.
        nt_outer = _semquad_mg_thread_count(mg_threads)
        if nt_outer > 1
            par_outer = SemQuadKPar(mfop, nt_outer)
            mfpar = par_outer
            mfapplyK = (out, v) -> _semquad_apply_Kred_threaded!(out, mfop, Cu,
                                                                 mfetav, v,
                                                                 mfwork,
                                                                 par_outer)
        else
            mfapplyK = (out, v) -> _semquad_apply_Kred!(out, mfop, Cu, mfetav,
                                                        v, mfwork)
        end
    else
        fr = Cu' * (fv - K * u0 + Bm' * p0)
    end
    Kred = mf_path ? nothing :
           Kgeo === nothing ? Cu' * K * Cu : Cu' * (K + Kgeo) * Cu
    Kp_red = Kp === nothing ? nothing : Cu' * Kp * Cp
    gr = Cp' * (Bc * u0)
    nf = size(Cu, 2); nfp = size(Cp, 2)
    Bs = -CpB
    Bt = Bm === Bc ? sparse(transpose(Bs)) : -(Cu' * Bm' * Cp)
    # Compressibility: in route C's (dynamic) pressure gauge the ALA density
    # perturbation -rho_ad Di gamma (p - p_ad) contributes +rho_ad Di gamma p
    # to the *upward* momentum.  The sign is not free: with ASPECT's own
    # temperature field this sign reproduces ASPECT's velocity to six digits
    # (Vrms 24.7657 vs 24.7657, max|u| 46.858 vs 46.858), while the opposite
    # sign gives Vrms 26.94 (the historical 8% King error).  See
    # ROUTE_C_STATUS.md for the operator-agreement test.
    if compress_block !== nothing
        Bmm = sparse(transpose(Bm)) - compress_block
        Bt = -(Cu' * Bmm * Cp)
    end
    # Pressure-block preconditioner for the saddle solve.  It is built after
    # Kred/Bs/Bt (and after the compressibility update of Bt) because the LSC
    # variant needs diag(Kred) and the final Bt.
    Fp = nothing
    if linear === :schur_cg && schur_precond === :mass
        Mfull = assemble_semquad_pressure_mass(sp)
        Mred = Cp' * Mfull * Cp
        Fp = cholesky(Symmetric(Mred))
        issuccess(Fp) ||
            throw(ArgumentError("pressure mass preconditioner is not SPD"))
    elseif linear === :mg && !kfree
        # Pressure-block approximate inverse (mass / LSC / PCD / three-factor
        # w-BFBT); shared with the MG-accelerated Newton solver, see
        # mg/pressure.jl.  `mg_pressure_precond` is the public selector
        # (:bfbt | :mass | :scaled_mass | :pcd | :lsc); its default `:auto`
        # keeps the historical `mg_pressure_mode` resolution, which picks the
        # three-factor w-BFBT above 100x viscosity contrast -- the measured
        # winner on the blob tree (docs/dev/SOLVER.md §13).  Not built for the
        # reduced `assemble_only=:mg` geometry call, which never uses it.
        pmode_arg = mg_pressure_mode
        pweight_arg = mg_pressure_weight
        if mg_pressure_precond !== :auto
            pmode_arg, pweight_arg =
                _semquad_pressure_precond_mode(mg_pressure_precond,
                                               mg_pressure_weight)
        end
        # S3 stage A: cached by (tree fingerprint, nf, nfp) and reused while the
        # sampled viscosity moves by at most `mg_pressure_eta_tol` (the outer
        # FGMRES is flexible, so a stale Shat^{-1} is a valid preconditioner).
        Fp, pscale = semquad_mg_pressure_precond(ss, Cu, Cp, Bs, Bt, Kred,
                                                 eta_fun;
                                                 mode=pmode_arg,
                                                 scale=mg_pressure_scale,
                                                 weight=pweight_arg,
                                                 key=_saddle_key(qt, p, bc, nf, nfp),
                                                 eta_tol=mg_pressure_eta_tol,
                                                 cache=mg_pressure_cache,
                                                 bfbt_d=bfbt_d,
                                                 bfbt_s2=mg_pressure_bfbt_s2,
                                                 bfbt_s1=mg_pressure_bfbt_s1,
                                                 bfbt_s1_solve=mg_pressure_bfbt_s1_solve,
                                                 bfbt_s1_tol=mg_pressure_bfbt_s1_tol,
                                                 bfbt_s1_maxit=mg_pressure_bfbt_s1_maxit,
                                                 bfbt_s1_minwork=mg_pressure_bfbt_s1_minwork,
                                                 bfbt_s1_chunk=mg_pressure_bfbt_s1_chunk,
                                                 nthreads=_semquad_mg_thread_count(mg_threads),
                                                 applyK=mfapplyK)
    end
    if assemble_only !== false
        return (K=K, B=Bref, Bc=Bc, Bm=Bm, Cu=Cu, Cp=Cp, u0=u0, p0=p0, fv=fv,
                fr=fr, gr=gr, Kred=Kred, Bs=Bs, Bt=Bt, Kgeo=Kgeo, Kp=Kp,
                Kp_red=Kp_red)
    end
    nit = 0
    # Every solve resets the direct-path shift record, so it describes this
    # solve only (a `:mg`/`:schur_cg`/`:lagged` solve reports "no shift").
    LAST_SEMSTOKES_SHIFT[] = 0.0
    LAST_SEMSTOKES_REFINE_ITERS[] = 0
    LAST_SEMSTOKES_LADDER[] = nothing
    LAST_SEMSTOKES_LADDER_PHASE[] = nothing
    if linear === :schur_cg
        # Saddle [Kred (-CpB'); (-CpB) 0] with Bs = -CpB, Bt = Bs':
        #   Kred u + Bt p = fr,   Bs u = gr,
        # solved by a matrix-free Schur CG on S = Bs Kred^{-1} Bt.
        # A constrained/hanging velocity mode can leave a zero eigenvalue in
        # Kred after AMR; regularize with a 1e-12 relative diagonal shift so the
        # sparse LU is defined.  The shift is below the G44 exactness tolerance.
        dmax = maximum(abs, diag(Kred))
        Kred = Kred + (1.0e-12 * max(dmax, 1.0)) * I
        # Row/column equilibration: wall-refined AMR trees have h_min/h_max
        # ~1e-4 and Kred condition numbers ~1e12; the scaled LU is much better
        # conditioned and removes the artificial large velocity from the shift.
        d = sqrt.(max.(diag(Kred), 1.0e-14 * dmax))
        Dinv = spdiagm(0 => 1.0 ./ d)
        Kscaled = Dinv * Kred * Dinv
        Fv = lu(Kscaled)
        ufree = Dinv * (Fv \ (Dinv * fr))
        r = Bs * ufree - gr
        pfree = zeros(Float64, nfp)
        z = Fp === nothing ? copy(r) : Fp \ r
        pp = copy(z)
        rho = dot(r, z)
        bnorm = max(dot(r, r), 1.0e-300)
        for it in 1:schur_maxiter
            nit = it
            Ap = Bs * (Dinv * (Fv \ (Dinv * (Bt * pp))))
            den = dot(pp, Ap)
            den == 0.0 && break
            alpha = rho / den
            pfree .+= alpha .* pp
            r .-= alpha .* Ap
            rs = dot(r, r)
            rs <= 1.0e-18 * bnorm && break
            z = Fp === nothing ? copy(r) : Fp \ r
            rho_new = dot(r, z)
            pp .= z .+ (rho_new / rho) .* pp
            rho = rho_new
        end
        ufree = Dinv * (Fv \ (Dinv * (fr - Bt * pfree)))
    elseif linear === :mg
        # Scalable path: geometric MG for the velocity block plus a pressure
        # Schur block, used as a block-triangular right preconditioner for
        # flexible GMRES on the reduced saddle (S3/MF-4).  The velocity block
        # is solved by the fixed `mg_cycles` V-cycles
        # (`mg_vel_solver=:vcycle`, the default since MF-6, 33% faster than the
        # inner-CG variant on the 1024-leaf blob tree); `mg_vel_solver=:cg`
        # keeps the MF-4 scheme, CG on the SPD `Kred` preconditioned by the
        # hierarchy (the deal.II/ASPECT FGMRES+CGMG form).  The criterion is
        # the *true* reduced-saddle residual (see `fgmres_restarted`), never
        # the preconditioned one.
        uniform = length(unique(lf[1] for lf in qt.leaves)) == 1
        smoother = mg_smoother === :auto ? :jacobi : mg_smoother
        coarse = mg_coarse === :auto ? (uniform ? :rediscretize : :galerkin) :
                 mg_coarse
        # S3/MF-2: the matrix-free hierarchy always coarsens by
        # rediscretisation (see make_semquad_mg).
        mg_operator === :matrixfree && (coarse = :rediscretize)
        mg = make_semquad_mg(ss, qt, p, bc, deviatoric, eta_fun, Kred;
                             cycles=mg_cycles, smoother=smoother,
                             coarse_operator=coarse, operator=mg_operator,
                             coarsen_layers=mg_coarsen_layers,
                             coarsen_factor=coarsen_factor,
                             coarse_direct_max=mg_coarse_direct_max,
                             smoothing=mg_smoothing, pre=mg_pre,
                             post=mg_post, omega=mg_omega,
                             mg_threads=mg_threads, mg_diag=mg_diag,
                             fine_Cu=Cu,
                             # On the matrix-free path the solve already built
                             # the finest level's leaf-local operator and its
                             # eta field; reuse both instead of resampling the
                             # viscosity a third time (0.17 s at one thread).
                             fine_mfop=mf_path ? mfop : nothing,
                             fine_mfeta=mf_path ? mfetav : nothing)
        Fp === nothing &&
            throw(ArgumentError("linear=:mg requires the pressure mass " *
                                "preconditioner (schur_precond=:mass)"))
        nftot = nf + nfp
        sol, nit, relres, cgiters =
            semquad_mg_saddle_solve(Kred, Bs, Bt, Fp, pscale, vcat(fr, gr), mg;
                                    tol=mg_tol, maxit=mg_maxit,
                                    restart=mg_restart,
                                    vel_solver=mg_vel_solver,
                                    vel_tol=mg_vel_tol,
                                    vel_maxit=mg_vel_maxit,
                                    applyK=mfapplyK)
        LAST_SEMSTOKES_VEL_CG_ITERS[] = cgiters
        ufree = sol[1:nf]
        pfree = sol[(nf + 1):nftot]
        relres > 1.0e-5 &&
            @warn "MG-FGMRES true relres=$relres exceeds 1e-5 at level " *
                  "$(mg.geo.L)"
    elseif linear === :lagged
        S = [Kred Bt; Bs spzeros(nfp, nfp)]
        rhs = vcat(fr, gr)
        # S2: reuse the previous LU as a preconditioner and polish with iterative
        # refinement; refactor only on a cache miss / stall.  The saddle depends
        # on the viscosity field and the tree, not on dt, so between two AMR
        # cycles the stale factor is a good preconditioner.
        key = _saddle_key(qt, ss.su.p, bc, nf, nfp)
        sol, rn = _lagged_solve!(S, rhs, key; tol=1.0e-8, maxit=25)
        LAST_SEMSTOKES_REGULARIZED[] = false
        if rn > 1.0e-8
            dmax = maximum(abs, diag(S)) + 1.0
            Ssh = S + (1.0e-12 * dmax) * I
            sol, rn = _lagged_solve!(Ssh, rhs, key; tol=1.0e-8, maxit=6,
                                     force_refactor=true)
            LAST_SEMSTOKES_REGULARIZED[] = true
        end
        ufree = sol[1:nf]
        pfree = sol[nf+1:end]
    else
        S = [Kred Bt; Bs spzeros(nfp, nfp)]
        rhs = vcat(fr, gr)
        # Wall-refined trees with free-slip walls make the *unshifted* sparse LU
        # unstable: on the 2b base=3/4 wall trees the factorisation has a growth
        # factor ~1e15 (max |U_ii| ~ 1e13-1e16 against a 2.9e-5 smallest pivot)
        # and the backward residual is ~1e3, while the same system solved with a
        # 1e-12 relative diagonal shift -- exactly what `_semquad_newton_saddle`
        # and the :schur_cg path already use -- is exact (residual 3e-11) and
        # gives the same velocity to 1e-6.  The near-null mode is a pressure mode
        # of the level-6 wall strip (|Bs'q|/|q| ~ 5e-3, velocity part 2e-4).
        #
        # `_shift_ladder_solve` keeps that single retry as its first rung and
        # only escalates (1e-10, 1e-8) while the residual misses
        # `stokes_accept_tol`, the driver's Stokes gate; the last resort is a few
        # iterative-refinement steps with the same factorisation.  The default
        # `accept_tol = Inf` stops after the first rung, so the well-conditioned
        # systems (uniform meshes, L2-L4 2b, the blob) stay bit-identical.
        # docs/dev/CASE_REALISTIC.md §13: on the 8192-leaf realistic tree the 1e-12
        # rung alone leaves 1.3e-3 (1.3x the card's 1e-3) and the driver aborted
        # the long run at step 6130.  §13.10: when the unshifted solve is *lost*
        # (`stokes_lost_tol`, default 1.0) the ladder tries the tiny shifts
        # first, because the shift's bias is `shift*|x|` and the near-null
        # cluster of the level-5 wall strip sits ~170x below the 1e-12 rung
        # (measured: 4.7e7 unshifted -> 5.2e-7 at 1e-14 on the step-6600 state,
        # while the 1e-12 rung leaves 1.1e-4).
        rden0 = max(norm(fr), norm(gr), 1.0)
        rel = function (x)
            rv = norm(Kred * x[1:nf] + Bt * x[(nf + 1):(nf + nfp)] - fr) / rden0
            rd = norm(Bs * x[1:nf] - gr) / rden0
            return max(rv, rd)
        end
        if stokes_scaling === :equilibrated
            # docs/dev/BUG_LEDGER.md: solver-side symmetric equilibration
            # of the same saddle.  First the historical unscaled unshifted solve,
            # exactly the ladder's rule 1: a solve that is healthy there is
            # returned untouched, so healthy paths stay bitwise identical even
            # with the option on, and the scaling is only paid for on a solve
            # that would otherwise enter the shift ladder.
            x_un = S \ rhs
            r_un = rel(x_un)
            r_un = isfinite(r_un) ? r_un : Inf
            if !(r_un > 1.0e-8) && !(r_un > stokes_accept_tol)
                sol, sh, nref, rungs = x_un, 0.0, 0, nothing
                LAST_SEMSTOKES_LADDER_PHASE[] = :healthy
            else
                # The congruence S -> D*S*D, rhs -> D*rhs leaves the solution
                # unchanged (`x = D*y`); it changes only the arithmetic of the
                # factorisation.  In the scaled operator the wall-strip pressure
                # family is lifted far above the round-off floor (the measured
                # smallest eigenvalue is a momentum mode at 4.2e-11 relative,
                # not the unscaled 1.7e-14 pressure mode), so the *unshifted* LU
                # resolves it: 6.0e-10 (AMR 7641) and 3.6e-8 (uniform L5,
                # round-off limited by the null component) against 3.7e-7 /
                # 1.2e-6 for the unscaled tiny-shift stopgap.  The 1e-6 gate is
                # the stated accuracy goal; if the scaled solve misses it, the
                # historical unscaled ladder is the unchanged fallback.
                sol_scaled, _ = _semquad_stokes_equil_solve(S, Kred, Bt, Bs, rhs)
                r_sc = rel(sol_scaled)
                r_sc = isfinite(r_sc) ? r_sc : Inf
                if r_sc <= 1.0e-6 && r_sc <= stokes_accept_tol
                    sol, sh, nref = sol_scaled, 0.0, 0
                    rungs = Any[(0.0, r_un), (:equilibrated, r_sc)]
                    LAST_SEMSTOKES_LADDER_PHASE[] = :equilibrated
                    LAST_SEMSTOKES_SCALED[] += 1
                else
                    sol, _, sh, nref, rungs =
                        _shift_ladder_solve(S, rhs; accept_tol=stokes_accept_tol,
                                            lost_tol=stokes_lost_tol,
                                            bad_tol=1.0e-8, resid=rel)
                end
            end
        else
            sol, _, sh, nref, rungs =
                _shift_ladder_solve(S, rhs; accept_tol=stokes_accept_tol,
                                    lost_tol=stokes_lost_tol,
                                    bad_tol=1.0e-8, resid=rel)
        end
        LAST_SEMSTOKES_REGULARIZED[] = sh > 0.0
        LAST_SEMSTOKES_SHIFT[] = sh
        LAST_SEMSTOKES_REFINE_ITERS[] = nref
        LAST_SEMSTOKES_LADDER[] = rungs
        ufree = sol[1:nf]
        pfree = sol[nf+1:end]
    end
    # True reduced-saddle residual of the system that was actually solved
    # (Schur uses the regularized Kred).  This is what distinguishes a
    # converged solve from a singular/ill-conditioned direct solve.
    r1 = if mf_path
        Ku = zeros(Float64, nf)
        mfapplyK(Ku, ufree)
        Ku .+ Bt * pfree .- fr
    else
        Kred * ufree + Bt * pfree - fr
    end
    r2 = Bs * ufree - gr
    rden = max(norm(fr), norm(gr), 1.0)
    LAST_SEMSTOKES_RESIDUAL[] = max(norm(r1), norm(r2)) / rden
    LAST_SEMSTOKES_RESID_SPLIT[] = (vel=norm(r1) / rden, div=norm(r2) / rden)
    LAST_SEMSTOKES_SCHUR_ITERS[] = nit
    u = u0 + Cu * ufree
    pv = p0 + Cp * pfree
    if return_system
        return u, pv, fixed, cons2, cons_p, K, Bref, Cu, Cp, u0, p0, fv
    end
    return u, pv, fixed, cons2, cons_p
end

