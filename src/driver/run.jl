# Per-step wall-clock profile collected by
# `run_semquad_simulation(...; profile_steps=true)`: a vector of named tuples
# `(step, t, leaves, N, stokes, work, energy, diag, amr, callback, total)` in
# seconds.  `nothing` unless profiling was requested; the driver/bench scripts
# read it to attribute a production run's wall clock without re-running it.
const SEMQUAD_STEP_PROFILE = Ref{Any}(nothing)

# Per-step nonlinear-solve record collected by
# `run_semquad_simulation(...; pic_diag=true)`: one named tuple per step with
# `(step, mode, iters, changes, rejects, relax, clamp_frac, stokes_res,
#  newton_res, fallback)` — `iters` is the number of nonlinear iterations that
# changed the state (Picard sweeps or Newton iterations), `changes` their
# relative theta update, `clamp_frac` the fraction of eta evaluations that hit
# `eta_min`/`eta_max` in the last Stokes assembly, `stokes_res` the Stokes
# relative residual, `fallback` whether the Newton failed and Picard took over.
# `nothing` unless diagnostics were requested.
const SEMQUAD_PIC_DIAG = Ref{Any}(nothing)

# Counters used by the `:composite` viscosity closure when `pic_diag` is on:
# `_SEMQUAD_ETA_CALLS` counts the eta evaluations of the last Stokes assembly
# (+ the work/dissipation evaluation) and `_SEMQUAD_ETA_CLAMP` how many of them
# hit `eta_min`/`eta_max`.  They are scratch state of the driver, reset at the
# start of every nonlinear iteration; they never enter an arithmetic expression.
const _SEMQUAD_ETA_CLAMP = Ref(0)
const _SEMQUAD_ETA_CALLS = Ref(0)

@inline function _count_eta_clamp(clipped::Bool)
    clipped && (_SEMQUAD_ETA_CLAMP[] += 1)
    _SEMQUAD_ETA_CALLS[] += 1
    return nothing
end

"""
    run_semquad_simulation(qt, pr, theta0; p=2, tstop, dt0, amr_every=0,
                           refine_frac=0.2, coarsen_frac=0.0, maxsteps=10000,
                           advection=:galerkin, cfl=0.0, verbose=false)

Run the 2-D Boussinesq convection loop on a quadtree.  `theta0` is the
perturbation on the conductive base `Tbar` and must vanish on the top/bottom
walls (the energy solve imposes homogeneous Dirichlet data there); the sides are
insulating, and the velocity is free-slip.  `eta_law` may be `:expT` (temperature
only) or `:expTz` (temperature + depth, ASPECT nondimensional law); `:expTz`
requires `eta_b`/`eta_c`.  `amr_every > 0` triggers one
indicator/refine/coarsen/transfer cycle every `amr_every` steps, using the
balanced policy of [`amr_cycle_balanced!`](@ref) (the only adaptive policy;
`refine_frac`/`coarsen_frac` are its cumulative-error thresholds).
`amr_passes > 1` reruns the
indicator/refine/transfer cycle on the same time level (a numerical budget;
default 1).  `max_leaves` is a hard post-balance leaf budget: the cycle pays for
a refinement by merging the lowest-error eligible quartet, and any over-budget
remainder is merged by the budget pass (default: no cap), so a comparison
against a static mesh can be made at a fixed DOF budget.  `max_level` caps
refinement at that quadtree level
(default: no cap); coarsening is unaffected.  `wall_min_level > 0` forces
leaves within `wall_strip` of the bottom/top walls back to at least that level
after every AMR cycle (the numerical boundary-refinement criterion; default 0).
`symmetry=:x` marks mirror leaves together (for symmetric benchmark/IC tests;
default `:none`).  `transfer_mode=:l2` uses the global L2 projection for
coarsening instead of pointwise nodal restriction (default `:nodal` for
backward-compatible tests).  `initial_refine > 0` runs that many
balanced AMR cycles on `theta0` *before* the first time step;
when `theta_ic(x,y)` is given, the analytic initial condition is re-sampled on
each refined tree instead of transferring the coarse initial field.  This is
the clean-protocol path for high-Ra runs: the boundary layer/plume is resolved
at t=0, rather than being inherited from a coarse static spin-up (default 0).  With `maxpic>1` the flow and energy are Picard-iterated;
`pic_relax` in (0,1] under-relaxes the theta update (`pic_relax=1` is the
original unrelaxed iteration).  With `cfl>0`
the step is updated from the current velocity and mesh (5% growth per step,
free shrink, hard CFL override); with `cfl=0` it stays at `dt0`.  If
`stokes_tol` is finite, the run errors out whenever the Stokes relative
residual exceeds it (default `Inf` keeps the old behaviour); set it to e.g.
`1e-6` for high-Ra production and raise `schur_maxiter` until it passes.
If `cfl_tol` is finite, the run errors out whenever `umax*dt/hmin` exceeds it
(a wall-CFL guard for level comparisons; default `Inf` keeps the old
behaviour).  `linear=:mg` is forwarded with `mg_operator`/`mg_vel_solver` and
the V-cycle schedule keywords `mg_smoothing` (`:auto`/`:uniform`/`:taper`) and
`mg_omega` (`nothing` = the schedule's own default: 2/3 uniform, 0.55 taper);
before S3 stage 2 the driver only forwarded the first two, so a driver-level
rollback of the tapered schedule (`mg_smoothing=:uniform, mg_omega=2/3`) was
impossible.  `linear=:auto` (S6) is forwarded unchanged and resolved inside
each `solve_semquad_stokes_var` call from the current tree size and the
viscosity contrast of the eta field actually used (`semquad_choose_linear`;
MG additionally forces `mg_operator=:matrixfree` and
`mg_threads=min(8, Threads.nthreads())`).  **`linear=:auto` is the default**
since 2026-11-13 (docs/dev/SOLVER.md §4.4): a driver call with no `linear` gets the
measured faster path per solve, and an `:auto` solve that resolves to `:direct`
is bitwise identical to the explicit `:direct` path (G191).  `linear=:direct`
and `linear=:mg` remain selectable; `:direct` is the only bitwise-reproducible
path (G150), so a bitwise reference must pin it explicitly, and
`pic_mode=:newton` requires it explicitly.  The decision and its reason are
exposed as `LAST_SEMSTOKES_AUTO_CHOICE` for logging (docs/dev/SOLVER.md §20).
`mg_pressure_eta_tol` (default
`SEMQUAD_PRESSURE_DEFAULT_ETA_TOL` = 1.0) is the pressure-block cache
threshold of §13.3/§19.1: the factors are rebuilt when the tree changes or the
sampled viscosity drifts by more than that relative amount (`Inf` = only on a
tree change; only for smoothly evolving viscosity, no phase change / no yield
onset).  `mg_threads > 1` forwards the S5-2 threaded matrix-free V-cycle
(`solve_semquad_stokes_var`'s `mg_threads`; bitwise identical to the serial
V-cycle, docs/dev/SOLVER.md §19.2).  The MG and solver keyword defaults
reproduce the previous behaviour exactly; only the `linear` default moved (to
`:auto`).
`stokes_scaling` (default `:none`) forwards the direct-path symmetric
equilibration of docs/dev/BUG_LEDGER.md to every Stokes solve
(`:equilibrated` enables it; it engages only when the historical unscaled
unshifted solve fails its health check, so a healthy solve stays bitwise
identical).  It is stored in the checkpoint's `run_config` and compared on
restart, because it can change the answer of a failing solve: a checkpoint
written before the keyword existed has no `stokes_scaling` key and is read as
`none`, so switching it on for an existing checkpoint chain is refused unless
that config string is rewritten.  `LAST_SEMSTOKES_SCALED[]` counts the solves
it answered.
`bc` is taken from `pr.bc` (`:free` free-slip or `:rigid`
no-slip, both walls).  `checkpoint_path`/`checkpoint_every` write an HDF5
checkpoint of the full state (quadtree and its maxlevel, theta, theta_prev, t,
dt, dt_prev, step, p) every `checkpoint_every` steps and once at the end; the
`AmrCooldown` hysteresis state (the two
cooldown dicts and the cycle counter) is written too, so a resume restores the
AMR cycle exactly.  `restart_from=<file>` replaces the mesh, p and initial
state with that state before the first step, so a restarted run is
bit-comparable with an uninterrupted one (`tstop`/`maxsteps` stay absolute); a
checkpoint without the cooldown data (every old file) loads an empty cooldown,
which is the historical restart behaviour.  The trajectory-affecting
settings (AMR cadence and fractions, wall/hotspot rules, advection, EV, Picard
controls, cfl, dtmax/dtmin) are stored with the state; a restart raises
`ArgumentError` listing every setting that differs from the live run, so a
resumed run cannot silently continue with another configuration.  The *physics*
is versioned separately from its parameter values:
`run_config["physics_revision"]` records `PHYSICS_REVISION` (the single source
of truth is `src/physics/revision.jl`, whose comment is the dated changelog),
because a commit can change the behaviour of unchanged parameters (958af5e
projected the Maxwell stress update onto the deviatoric strain rate, 8c27a02
added the co-rotational term).  A checkpoint whose stored revision differs is
refused, since its stress history was produced by the other update.  A
checkpoint with no `physics_revision` key (every file written before the marker
existed) is resumed with a warning that says whether it carries a Maxwell stress
history -- the only state the revision change can affect -- because refusing
would invalidate every historical checkpoint, including the `shear_modulus = 0`
ones whose physics is unchanged.  The cooldown
is state, not configuration, and is not part of that comparison.
`phases` adds the phase-transition buoyancy and the latent heat (docs/dev/PHYSICS.md
§7).  The latent coefficients come from `Params.latent_mode`: `:global` (the
default) applies `pr.latent_L` to every transition — the historical behaviour,
bit for bit — and `:per_phase` uses each `PhaseTransition.latent` (signed, see
`phase_latent_coefficient`), which is the physically correct form because the
sign of the latent term follows the Clapeyron slope (§7.8).  `latent_scheme =
:implicit` (default) evaluates the apparent heat capacity
`1 + sum_i latent_i*dGamma_i/dT` and the advective latent source at the current
iterate of the step, `:lagged` at the start-of-step temperature.  The choice only
exists inside an iterated step: with the default `maxpic = 1` there is no
iterate, so both settings give the same single lagged sweep (bitwise identical,
G199), which is first order in the latent coupling.  `maxpic > 1` and
`pic_mode = :newton` make the step (and with it the latent coupling) implicit
and second order; the measured orders, cost and the benchmark/production
recommendation are in docs/dev/PHYSICS.md §13.  `weak_chi0` is the initial value of
the strain-weakening field
(docs/dev/PHYSICS.md §9; the field is active only when the `Params` weakening
parameters are set); it is clamped to [0,1] and the field is checkpointed with
the state.
`profile_steps=true` fills `SEMQUAD_STEP_PROFILE` with one named tuple per step
(`stokes`, `work`, `energy`, `diag`, `amr`, `callback`, `total`, in seconds,
plus `step`/`t`/`leaves`/`N`); it only adds `time_ns()` calls around the
existing sections and changes nothing numerically.
`extdiag=false` (the default) keeps every output byte-identical.  `extdiag=true`
adds one history column, `resid_A` = `R/A`, the exact residual of the quoted
pointwise Nusselt pair against the scheme's own identity `(ID1)`
(`semquad_exact_residual`, docs/dev/VALIDATION.md §2.3/§3.2).  The historical
`closure`/`closure_nolatent` columns are **not** changed by it: they are the
historical formula, which equals `-dH/dt` plus three bookkeeping terms at a
transient state and is not a flux error (docs/dev/VALIDATION.md §4).  The extra
column changes `history.csv`'s layout, so
`examples/case_realistic_2d.jl` only accepts `CASE_EXTDIAG=1` in a new
output directory.
`corotation_guard=true` (the default) checks the **validity band** of the
linearised co-rotational (Jaumann) term, `1 + dt*ad_W` (G254).  The term is
valid where `2|W|dt << 1` (elastic regime) and `2*tau_M*|W|` is not large
(fast-relaxation regime), i.e. where `2|W|*min(dt, tau_M)` is small
(`MAXWELL_COROTATION_BOUND` = 0.1 quantifies it); outside that band the
rotation is not faithful even though the scheme remains consistent.  With
`shear_modulus > 0` the driver reports the run-maximum over the domain of
`2|W|dt`, of `2*tau_M*|W|` and of the per-step rotation
`2|W|*min(dt, tau_M)` once per run at `Info` level, and warns once -- at the
first step that exceeds the bound -- with the measured value, the bound and
what it means.  The run continues: this is an accuracy band, not a stability
limit, so refusing would abort a long run for a quantified accuracy loss;
reduce `dt` (`dtmax`/`cfl`) to get back inside.  The keyword is diagnostic
only: the maxima are accumulated from values the stress update already samples,
no arithmetic result depends on them, and a run with the guard off is
bit-identical (G254).  With `shear_modulus == 0` nothing is sampled or logged.
`regime_report=true` (the default) reports the **Maxwell regime** when
`shear_modulus > 0` (G255): once at the first step that has a stress update and
once at the end of the run, the material viscosity range and contrast, the
effective viscosity range and contrast at that step's `dt` (the 2a closure
`eta_eff = 1/(1/eta + 1/(G dt))`), and the range of `G*dt/eta`.  The record ends
with what the numbers mean: `G*dt` caps the effective viscosity from above, so
the contrast the Stokes solve actually sees can be smaller than the material one
by decades -- a viscoelastic card is a *softer, different* problem than its
`shear_modulus = 0` twin, not the same problem plus memory.  This is diagnostic
only: the ranges are computed from the nodal viscosity field and step the stress
update already uses, `regime_report=false` silences both records and leaves a
run bit-identical (G255), and with `shear_modulus == 0` nothing is sampled or
logged.  It is a separate record from the co-rotation validity report above.
`Params.eta_ref` (default 1.0) is the **Rayleigh-number reference viscosity** in
the units the material law returns (G257): the momentum equation is solved with
`eta_law/eta_ref` wherever the law's viscosity enters -- the same wrapper chain
that applies `eta_multiplier` and the Maxwell closure -- so the run's convention
is `Ra = rho g alpha dT d^3/(kappa eta_ref)`.  Whenever `eta_ref != 1` the
driver reports the convention once at the first flow solve and once at the end
of the run: `Ra`, `eta_ref`, the normalised field `eta/eta_ref` range and
contrast, `eta_interior` (the arithmetic volume mean over the interior band
`0.25 <= z <= 0.75`, the middle half of the layer) and
`Ra_interior = Ra/eta_interior`.  `eta_ref_scale` (default `nothing`) is the
card's dimensional scale set as a `NamedTuple` with positive finite
`rho, g, alpha, dT, d, kappa`; when given, the report also prints
`eta_0 = rho g alpha dT d^3/(Ra kappa)` in Pa s -- the physical viscosity `Ra` is
defined with -- and the physical interior viscosity.  The report is diagnostic
only (it measures the closure the solve used; no arithmetic result depends on
it), and with `eta_ref = 1.0` it logs nothing, so the historical runs stay
bit-identical.  The normalisation is applied by the Picard time-stepping path
only: `pic_mode = :newton` and `steady_newton` build their viscosity from the raw
law and refuse `eta_ref != 1` instead of solving a different problem silently.
Returns a named tuple with the final state and history rows
`[t, dt, Nu_top, Nu_bot, Vrms, meanT, N, nleaves]` (plus the opt-in columns
described above; `history_extra` names them).  `txx`/`tyy`/`txy` are the nodal
Maxwell stress-history components (`nothing` unless `pr.shear_modulus > 0`;
docs/dev/PHASE2.md 2b): state that is advanced once per step, transferred on AMR
and checkpointed.  The tensor is the **deviatoric** stress: the update is driven
by `ehat = eps - (1/3) tr(eps) I` (G249), so the out-of-plane normal stress is
implied by trace zero, `tau_zz = -(tau_xx + tau_yy)`, rather than stored -- the
2-D momentum equation never contains it.  With `shear_modulus > 0` the previous
step's stress also drives the momentum through the 2b elastic force term
`div(a tau^n)` (2b step 2; see `maxwell_force_stress`), so its history feeds the
velocity trajectory.  With `shear_modulus == 0` no elastic path exists at all and
the run is bit-identical to the pre-2b one.
"""
function run_semquad_simulation(qt::Quadtree, pr::Params, theta0::Vector{Float64};
                                p::Int=2, tstop::Float64=1.0e-3,
                                dt0::Float64=1.0e-4, amr_every::Int=0,
                                refine_frac::Float64=0.2,
                                amr_passes::Int=1,
                                coarsen_frac::Float64=0.0,
                                max_leaves::Int=typemax(Int),
                                max_level::Int=typemax(Int),
                                wall_min_level::Int=0, wall_strip::Float64=0.1,
                                symmetry::Symbol=:none,
                                amr_hysteresis::Int=1,
                                amr_coarsen_gate::Bool=true,
                                # :theta (default, historical), :velocity, or
                                # :both (normalised sum); see docs/dev/PHASE1.md
                                # 1.1 and src/amr/indicator.jl
                                amr_indicator::Symbol=:theta,
                                transfer_mode::Symbol=:nodal,
                                initial_refine::Int=0,
                                theta_ic::Union{Nothing,Function}=nothing,
                                maxsteps::Int=10000,
                                advection::Symbol=:galerkin,
                                ev::Float64=0.0, ev_cmax::Float64=0.052,
                                ev_alpha::Int=2,
                                callback::Union{Nothing,Function}=nothing,
                                hotspot_weight::Float64=0.0, hotspot_frac::Float64=0.5,
                                hotspot_min_level::Int=0, hotspot_mode::Symbol=:theta,
                                hotspot_zmin::Float64=0.0, zero_side_eta::Bool=false, side_strip::Float64=0.05,
                                top_wall_level::Int=0, top_wall_frac::Float64=0.5, cfl::Float64=0.0, dtmax::Float64=Inf,
                                dt_growth::Float64=1.05,
                                dtmin::Float64=1.0e-14,
                                maxpic::Int=1, pic_tol::Float64=1.0e-6,
                                pic_relax::Float64=1.0,
                                pic_adapt::Bool=false,
                                pic_relax_min::Float64=0.05,
                                pic_eta_relax::Float64=1.0,
                                pic_u_relax::Float64=1.0,
                                pic_div_window::Int=0,
                                pic_div_factor::Float64=10.0,
                                pic_mode::Symbol=:picard,
                                pic_reject::Bool=false,
                                pic_reject_clamp::Float64=0.25,
                                pic_reject_res::Float64=10.0,
                                pic_reject_max::Int=4,
                                pic_diag::Bool=false,
                                newton_maxit::Int=15,
                                newton_krylov::Int=80,
                                newton_restart::Int=30,
                                newton_ls::Int=8,
                                newton_geo_safety::Float64=0.0,
                                linear::Symbol=:auto, schur_maxiter::Int=200,
                                schur_precond::Symbol=:mass,
                                mg_operator::Symbol=:matrixfree,
                                mg_vel_solver::Symbol=:vcycle,
                                mg_smoothing::Symbol=:auto,
                                mg_omega::Union{Nothing,Float64}=nothing,
                                mg_pressure_eta_tol::Float64=SEMQUAD_PRESSURE_DEFAULT_ETA_TOL,
                                mg_threads::Int=1,
                                # Saddle tolerance of the `linear=:mg` solve.
                                # The default is `solve_semquad_stokes_var`'s,
                                # so a run that does not set it is unchanged;
                                # the trajectory benchmark
                                # (`benchmarks/bench_mg_level_cache.jl`) uses
                                # the documented 1e-6.  Inert on every other
                                # `linear` setting.
                                mg_tol::Float64=1.0e-10,
                                stokes_scaling::Symbol=:none,
                                profile_steps::Bool=false,
                                mass_form::Symbol=:anelastic,
                                phases::Vector{PhaseTransition}=PhaseTransition[],
                                latent_scheme::Symbol=:implicit,
                                weak_chi0::Float64=0.0,
                                nu_cbf::Bool=false,
                                extdiag::Bool=false,
                                # G254: report the validity band of the
                                # linearised co-rotational term and warn when
                                # the run leaves it.  Diagnostic only (no
                                # arithmetic result depends on it); silently
                                # does nothing when `shear_modulus = 0`.
                                corotation_guard::Bool=true,
                                # G255: report the material vs effective
                                # viscosity regime (the contrast collapse of the
                                # `eta_eff = 1/(1/eta + 1/(G dt))` cap) once at
                                # the first stress step and once at the end.
                                # Diagnostic only: `false` silences both records
                                # and leaves the run bit-identical; silently
                                # does nothing when `shear_modulus = 0`.
                                regime_report::Bool=true,
                                # G257: the card's dimensional scale set, used
                                # only by the eta_ref report to print the
                                # reference viscosity in Pa s.  `nothing` (the
                                # default) keeps the report dimensionless.
                                eta_ref_scale::Union{Nothing,NamedTuple}=nothing,
                                stokes_tol::Float64=Inf,
                                cfl_tol::Float64=Inf,
                                tavg_start::Float64=Inf,
                                steady_tol::Float64=Inf,
                                steady_window::Int=5,
                                tmin::Float64=0.0,
                                restart_from::String="",
                                checkpoint_path::String="",
                                checkpoint_every::Int=0,
                                checkpoint_meta::AbstractDict=Dict{String,Any}(),
                                steady_newton::Symbol=:none,
                                verbose::Bool=false)
    pr.eta_law in (:expT, :expTz, :arrhenius, :tosi, :composite) ||
        throw(ArgumentError("run_semquad_simulation requires eta_law=:expT, " *
                            ":expTz, :arrhenius, :tosi or :composite"))
    steady_window >= 1 ||
        throw(ArgumentError("steady_window must be >= 1"))
    # G257: the optional dimensional scale set of the eta_ref report.  It is
    # only multiplied into physical viscosities (Pa s), but a malformed one
    # would silently print a NaN/Inf, so validate it up front.
    if eta_ref_scale !== nothing
        (eta_ref_scale isa NamedTuple &&
         all(k -> haskey(eta_ref_scale, k) && isfinite(eta_ref_scale[k]) &&
                  eta_ref_scale[k] > 0.0,
             (:rho, :g, :alpha, :dT, :d, :kappa))) ||
            throw(ArgumentError("eta_ref_scale must be `nothing` or a NamedTuple " *
                                "with positive finite entries rho, g, alpha, dT, " *
                                "d, kappa (the dimensional scale set of the " *
                                "card, used to report the reference viscosity in " *
                                "Pa s)"))
    end
    mass_form in (:anelastic, :boussinesq) ||
        throw(ArgumentError("mass_form must be :anelastic or :boussinesq"))
    # Forwarded verbatim to `solve_semquad_stokes_var`; validated here so a typo
    # fails before the setup work instead of inside the first Stokes solve.
    linear in (:direct, :schur_cg, :mg, :lagged, :auto) ||
        throw(ArgumentError("linear must be :direct, :schur_cg, :mg, :lagged " *
                            "or :auto"))
    # Same contract as `linear`: validated here, so a typo fails before the
    # setup work instead of inside the first Stokes solve.
    stokes_scaling in SEMSTOKES_SCALINGS ||
        throw(ArgumentError("stokes_scaling must be :none or :equilibrated"))
    mg_smoothing in (:auto, :uniform, :taper) ||
        throw(ArgumentError("mg_smoothing must be :auto, :uniform or :taper"))
    (mg_omega === nothing || 0.0 < mg_omega < 1.0) ||
        throw(ArgumentError("mg_omega must be in (0,1) or `nothing`"))
    tavg_start >= 0.0 ||
        throw(ArgumentError("tavg_start must be >= 0"))
    0.0 < pic_relax <= 1.0 ||
        throw(ArgumentError("pic_relax must be in (0, 1]"))
    0.0 <= pic_eta_relax <= 1.0 ||
        throw(ArgumentError("pic_eta_relax must be in [0, 1]"))
    0.0 < pic_u_relax <= 1.0 ||
        throw(ArgumentError("pic_u_relax must be in (0, 1]"))
    if pr.eta_law === :tosi && maxpic > 1
        # Measured: with the strain-rate weakening of the Tosi law, iterating
        # the Picard loop inside one step grows u until the CFL bound (which is
        # computed at iteration 1) is violated and the energy solve diverges.
        # Even pic_u_relax = 0.3 does not prevent it.  The lagged scheme
        # (maxpic = 1) reproduces the reference trajectory to 5-8%, so require
        # it explicitly instead of diverging silently.
        throw(ArgumentError("eta_law=:tosi requires maxpic=1 (lagged viscosity); " *
                            "iterating within a step diverges"))
    end
    pic_div_window >= 0 ||
        throw(ArgumentError("pic_div_window must be >= 0"))
    pic_div_factor > 1.0 ||
        throw(ArgumentError("pic_div_factor must be > 1"))
    pic_mode in (:picard, :newton) ||
        throw(ArgumentError("pic_mode must be :picard or :newton"))
    # The per-step Newton path builds its viscosity from the raw law plus the 2a
    # viscoelastic softening; it does not apply `Params.eta_multiplier` (a weak
    # zone, or any other multiplicative modification of the law).  A run that asks
    # for both would therefore solve a *different* viscosity than the Picard path
    # with the same card -- a wrong answer with no error.  Refuse instead.  The
    # steady-state Newton entry points have the same gap; they are not reachable
    # from here and are out of scope.
    (pic_mode !== :newton || pr.eta_multiplier === nothing) ||
        throw(ArgumentError("pic_mode=:newton does not apply Params.eta_multiplier " *
                            "(the per-step Newton builds its viscosity from the raw " *
                            "law plus the viscoelastic softening only), so a run with " *
                            "both would differ from the Picard path with the same " *
                            "card.  Use pic_mode=:picard, or drop eta_multiplier."))
    # G257: same gap for the Rayleigh-number reference viscosity.  The Newton
    # path builds its viscosity from the raw law; applying eta_ref only on the
    # Picard path would make the two paths solve different problems.
    (pic_mode !== :newton || pr.eta_ref == 1.0) ||
        throw(ArgumentError("pic_mode=:newton does not apply Params.eta_ref " *
                            "(the per-step Newton builds its viscosity from the " *
                            "raw law, without the eta/eta_ref normalisation), so " *
                            "a run with both would differ from the Picard path " *
                            "with the same card.  Use pic_mode=:picard, or set " *
                            "eta_ref = 1.0."))
    # The steady-state hand-off at the end of the run has the same gap: the
    # steady Newton entry points do not apply `Params.eta_ref` (that path is out
    # of scope and untouched).  Refuse instead of solving a different problem.
    (steady_newton === :none || pr.eta_ref == 1.0) ||
        throw(ArgumentError("steady_newton=$(steady_newton) does not apply " *
                            "Params.eta_ref: the steady-state entry points build " *
                            "their viscosity from the raw law.  Set eta_ref = " *
                            "1.0, or run the time-stepping path without the " *
                            "steady hand-off."))
    latent_scheme in (:implicit, :lagged) ||
        throw(ArgumentError("latent_scheme must be :implicit or :lagged"))
    0.0 <= weak_chi0 <= 1.0 ||
        throw(ArgumentError("weak_chi0 must be in [0, 1]"))
    pic_reject_max >= 0 ||
        throw(ArgumentError("pic_reject_max must be >= 0"))
    0.0 <= pic_reject_clamp < 1.0 ||
        throw(ArgumentError("pic_reject_clamp must be in [0, 1)"))
    pic_reject_res > 1.0 ||
        throw(ArgumentError("pic_reject_res must be > 1"))
    newton_maxit >= 1 || throw(ArgumentError("newton_maxit must be >= 1"))
    newton_krylov >= 1 || throw(ArgumentError("newton_krylov must be >= 1"))
    newton_restart >= 1 || throw(ArgumentError("newton_restart must be >= 1"))
    newton_ls >= 1 || throw(ArgumentError("newton_ls must be >= 1"))
    newton_geo_safety >= 0.0 ||
        throw(ArgumentError("newton_geo_safety must be >= 0"))
    if pic_mode === :newton
        # The per-step Newton's Jacobian is the Boussinesq system with a
        # temperature-, strain-rate- and (optionally) pressure- and
        # weakening-dependent viscosity, the phase-transition buoyancy and the
        # latent heat (apparent heat capacity + advective source).  The
        # Drucker-Prager plastic branch is supported through its `Kp` pressure
        # block and the `deta/dchi` weakening block (docs/dev/PHYSICS.md §8, §9);
        # the TALA/ALA reference state (weighted buoyancy), compressible
        # momentum and the CBF diagnostics still add terms it does not
        # linearise, so they are rejected instead of silently degrading the
        # Newton to a Picard iteration.
        pr.refstate === nothing ||
            throw(ArgumentError("pic_mode=:newton does not support a " *
                                "reference state (TALA/ALA buoyancy)"))
        pr.compress_p &&
            throw(ArgumentError("pic_mode=:newton does not support the " *
                                "compressible momentum term"))
        nu_cbf &&
            throw(ArgumentError("pic_mode=:newton does not support nu_cbf " *
                                "(the EV field of the step is internal)"))
        linear === :direct ||
            throw(ArgumentError("pic_mode=:newton requires linear=:direct " *
                                "explicitly: the library default is :auto, " *
                                "which may resolve to :mg, while Newton needs " *
                                "one direct saddle factorisation per " *
                                "iteration.  Pass linear=:direct."))
    end
    max_leaves >= 1 || throw(ArgumentError("max_leaves must be >= 1"))
    initial_refine >= 0 || throw(ArgumentError("initial_refine must be >= 0"))
    max_level >= 1 || throw(ArgumentError("max_level must be >= 1"))
    wall_min_level >= 0 || throw(ArgumentError("wall_min_level must be >= 0"))
    0.0 < wall_strip < 0.5 || throw(ArgumentError("wall_strip must be in (0, 0.5)"))
    amr_indicator in (:theta, :velocity, :both) ||
        throw(ArgumentError("amr_indicator must be :theta, :velocity or :both"))
    symmetry in (:none, :x) ||
        throw(ArgumentError("symmetry must be :none or :x"))
    amr_hysteresis >= 0 ||
        throw(ArgumentError("amr_hysteresis must be >= 0"))
    transfer_mode in (:nodal, :l2) ||
        throw(ArgumentError("transfer_mode must be :nodal or :l2"))
    checkpoint_every >= 0 ||
        throw(ArgumentError("checkpoint_every must be >= 0"))
    # Settings that change the trajectory.  They are stored in the checkpoint
    # and validated on restart, so a resumed run cannot silently continue with
    # a different AMR/physics/time-integration setup.
    # A decaying heating rate needs a time-dependent path: the steady solvers and
    # the per-step Newton tangent assemble the source in their own places, and a
    # steady state of a decaying source does not exist.  Refusing is better than
    # integrating a source the residual does not see.
    if isfinite(pr.H_half_life)
        (pic_mode === :picard && mass_form in (:boussinesq, :anelastic)) ||
            throw(ArgumentError("H_half_life requires pic_mode=:picard; the " *
                                "steady and Newton paths hold H constant"))
    end

    # Every setting that changes the trajectory is recorded in one dictionary:
    # the checkpoint stores it and the restart guard compares it.  The builder
    # and the guard live in `restart_guard.jl`; the keywords they read are
    # forwarded here.
    run_config = _run_config(pr;
                             amr_every=amr_every, refine_frac=refine_frac,
                             coarsen_frac=coarsen_frac, amr_passes=amr_passes,
                             max_leaves=max_leaves, max_level=max_level,
                             wall_min_level=wall_min_level,
                             wall_strip=wall_strip, symmetry=symmetry,
                             transfer_mode=transfer_mode,
                             amr_hysteresis=amr_hysteresis,
                             amr_coarsen_gate=amr_coarsen_gate,
                             amr_indicator=amr_indicator,
                             hotspot_weight=hotspot_weight,
                             hotspot_frac=hotspot_frac,
                             hotspot_min_level=hotspot_min_level,
                             hotspot_mode=hotspot_mode,
                             hotspot_zmin=hotspot_zmin,
                             zero_side_eta=zero_side_eta,
                             side_strip=side_strip,
                             top_wall_level=top_wall_level,
                             top_wall_frac=top_wall_frac,
                             advection=advection, ev=ev, ev_cmax=ev_cmax,
                             ev_alpha=ev_alpha, maxpic=maxpic,
                             pic_tol=pic_tol, pic_relax=pic_relax,
                             pic_mode=pic_mode, pic_reject=pic_reject,
                             cfl=cfl, dtmax=dtmax, dtmin=dtmin,
                             mass_form=mass_form,
                             stokes_scaling=stokes_scaling,
                             latent_scheme=latent_scheme, phases=phases,
                             weak_chi0=weak_chi0)
    # The restart guard: load the checkpoint, refuse another physics revision,
    # warn for a key-less file, compare the configuration and check theta; it
    # returns the resumed state (the arguments unchanged when there is none).
    guard = _restart_guard(qt, p, theta0, run_config, restart_from)
    restart = guard.restart
    qt = guard.qt
    p = guard.p
    theta0 = guard.theta0
    ss = make_semstokes(qt, p)
    sq = ss.su
    length(theta0) == sq.N ||
        throw(ArgumentError("theta0 has $(length(theta0)) entries, mesh needs $(sq.N)"))
    # theta0 is the perturbation T-Tbar, so its wall value is Twall - Tbar(wall).
    # For the Boussinesq base (Twall_top=0, Twall_bot=1, Tbar(1)=0, Tbar(0)=1)
    # this is zero, and a nonzero theta0 at a wall is a mistake (it creates a
    # spurious boundary layer and silently invalidates adaptive-vs-static
    # comparisons).  Keep that guard.  With a reference state the wall value is
    # generally nonzero and a cold start (theta0 = 0 there) is a legitimate
    # initial condition, so only the zero-target case is enforced.
    for g in 1:sq.N
        y = sq.coords[g][2]
        wall = y <= 1.0e-14 ? pr.Twall_bot :
               (y >= 1.0 - 1.0e-14 ? pr.Twall_top : NaN)
        isnan(wall) && continue
        target = wall - semquad_base_T(pr, y)
        if target == 0.0 && abs(theta0[g]) > 1.0e-12
            throw(ArgumentError("theta0 is the perturbation T-Tbar and must " *
                                "vanish on the top/bottom walls; node $g at " *
                                "y=$y has theta0=$(theta0[g])"))
        end
    end
    theta = copy(theta0)
    if restart === nothing
        # `initial_refine` runs the balanced cycle -- the only adaptive policy --
        # on the initial condition, with the indicator and the wall-row protect
        # set the first in-loop pass builds in `semquad_amr_passes!`.  The
        # cooldown is reset here so a fresh run never inherits the module global
        # of an earlier run in the same process; the loop below resets it again
        # for a restart-free start, so the pre-refinement state is not carried
        # into the time loop (each pass still pays its hysteresis, which is what
        # an equal in-loop pass sequence would do).
        SEMQUAD_AMR_COOLDOWN[] = AmrCooldown()
        for pass in 1:initial_refine
        length(qt.leaves) < max_leaves || break
        # At t = 0 the flow is at rest, so the velocity the in-loop pass receives
        # from the Stokes solve is identically zero.  Pass that zero field
        # explicitly instead of the undefined `ux`/`uz` the old code read here:
        # `:velocity` then sees an all-zero indicator and the cycle returns
        # without refining (there is no velocity error to resolve yet), and
        # `:both` reduces to `:theta` *exactly* (jt + 0), so its initial mesh is
        # the theta one.  That reduction is correct behaviour at rest, not a
        # missing refinement; `run.jl`'s time loop passes the real velocity.
        if amr_indicator === :theta
            eta = semquad_gradient_jump(sq, qt, theta)
        else
            ux0 = zeros(Float64, sq.N)
            eta = amr_indicator === :velocity ?
                  semquad_velocity_jump(sq, qt, ux0, ux0) :
                  semquad_combined_jump(sq, qt, theta, ux0, ux0)
        end
        protect = Set{NTuple{3,Int}}()
        for lf in qt.leaves
            if lf[3] == 0 || lf[3] == qt.ny0 * (1 << lf[1]) - 1
                push!(protect, lf)
            end
        end
        amr_cycle_balanced!(qt, eta; refine_frac=refine_frac,
                            coarsen_frac=coarsen_frac,
                            max_level=max_level, max_leaves=max_leaves,
                            symmetry=symmetry, protect=protect,
                            wall_min_level=wall_min_level,
                            wall_strip=wall_strip,
                            cooldown=SEMQUAD_AMR_COOLDOWN[],
                            hysteresis_cycles=amr_hysteresis,
                            coarsen_gate=amr_coarsen_gate)
        refine_wall_to_min!(qt, wall_min_level; strip=wall_strip,
                            symmetry=symmetry)
        # Hard leaf budget.  The cycle itself does not coarsen a quartet that
        # its Dörfler cut did not select, and 2:1 balancing can still add
        # leaves, so merge the lowest-error complete quartets here (the removed
        # classic budget pass called the fraction cycle with zero fractions,
        # whose guard gave up as soon as the first leaf in indicator order could
        # not be merged).
        if symmetry === :x
            mirror_close!(qt; max_level=max_level)
            balance!(qt; symmetry=:x)
        end
        eta_cap = copy(eta)
        for lf in qt.leaves
            get!(eta_cap, lf, 0.0)
        end
        merged, budget_ok = semquad_enforce_leaf_budget!(
            qt, eta_cap; max_leaves=max_leaves, protect=Set{NTuple{3,Int}}(),
            symmetry=symmetry)
        merged > 0 && verbose &&
            @printf("    AMR budget: merged %d quartets -> %d leaves\n", merged,
                    length(qt.leaves))
        budget_ok || @warn "leaf budget not reachable: $(length(qt.leaves)) leaves > max_leaves=$max_leaves (all remaining quartets are protected or unbalanced)"
        ss2 = make_semstokes(qt, p)
        if theta_ic === nothing
            theta = transfer_semquad(sq, theta, ss2.su; mode=transfer_mode,
                                     zero_walls=true, qt_new=qt)
        else
            theta = [Float64(theta_ic(x, y)) for (x, y) in ss2.su.coords]
            for g in 1:ss2.su.N
                y = ss2.su.coords[g][2]
                (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (theta[g] = 0.0)
            end
            all(isfinite, theta) ||
                throw(ArgumentError("theta_ic returned a non-finite value"))
        end
        ss = ss2
        sq = ss.su
        # `:auto` may resolve to MG on the new tree, so drop old hierarchies too.
        # Only the per-tree entries are dropped: the per-level cache (G221)
        # keeps the levels whose own leaf set survived the cycle, which is what
        # makes an AMR cycle rebuild only the levels the mesh actually changed.
        (linear === :mg || linear === :auto) && _semquad_mg_drop_tree_entries!()
        end
    end
    hist = Vector{Vector{Float64}}()
    if restart === nothing
        theta_prev = copy(theta)
        dt = dt0
        dt_prev = dt
        t = 0.0
        step = 0
        ux = zeros(Float64, sq.N)
        uz = zeros(Float64, sq.N)
        pv = zeros(Float64, ss.sp.N)
    else
        theta = copy(restart.theta)
        theta_prev = copy(restart.theta_prev)
        dt = restart.dt
        dt_prev = restart.dt_prev
        t = restart.t
        step = restart.step
        ux = copy(restart.ux)
        uz = copy(restart.uz)
        pv = copy(restart.pv)
        # A final checkpoint written after the end-of-run diagnostic re-solve
        # carries a velocity that belongs to the final mesh, not to the
        # trajectory: the uninterrupted run would start the next step from the
        # zeros the AMR pass left (the same zeros an in-loop checkpoint stores).
        # The marker is absent on old files and on in-loop checkpoints, so this
        # is a no-op for every historical restart.
        if hasproperty(restart, :velocity_stale) && restart.velocity_stale
            fill!(ux, 0.0); fill!(uz, 0.0); fill!(pv, 0.0)
        end
    end
    # Elementwise EV field of the last energy assembly (nothing when EV is off or
    # the checkpoint predates it); restored for diagnostics, then overwritten by
    # the next step.
    ev_field = (restart !== nothing && ev > 0.0) ? restart.ev : nothing
    velocity_stale = true
    # Hysteresis state of the balanced AMR cycle.  A fresh run starts with an
    # empty cooldown, so it is deterministic.  A checkpoint carries the cooldown
    # (`save_semquad_state`'s optional datasets), and it is restored verbatim so
    # the first AMR cycle after a resume is the one the uninterrupted run would
    # have executed; an old checkpoint without the data loads an empty cooldown,
    # which is exactly the historical restart behaviour (docs/dev/BUG_LEDGER.md
    # §10.5).  The cooldown is state, not configuration: it is not in
    # `run_config` and the restart guard must not compare it.
    SEMQUAD_AMR_COOLDOWN[] =
        (restart !== nothing && restart.cooldown !== nothing) ?
        restart.cooldown : AmrCooldown()

    # P1: running time averages (t >= tavg_start) and steady-state detection.
    avg_n = 0
    avg_sut = 0.0; avg_sub = 0.0; avg_sv = 0.0; avg_st = 0.0
    avg_sut2 = 0.0; avg_sub2 = 0.0; avg_sv2 = 0.0; avg_st2 = 0.0
    steady_count = 0
    Nu_prev = NaN

    yielding = yield_active(pr)
    tosi = pr.eta_law === :tosi
    composite = pr.eta_law === :composite
    # Strain-weakening field chi (docs/dev/PHYSICS.md §9); `nothing` when
    # `weak_active(pr)` is false, in which case every weakening code path is
    # skipped and the run is bit-identical to the pre-weakening behaviour.
    # `chi` is the start-of-step level chi^n, `chi_prev` is chi^(n-1) and
    # `chi_iter` the running nonlinear iterate (the one eta sees).
    weak = weak_active(pr)
    chi = nothing
    chi_prev = nothing
    chi_iter = nothing
    if weak
        if restart !== nothing && restart.weak !== nothing
            length(restart.weak) == sq.N ||
                throw(ArgumentError("checkpoint weakening field has " *
                                    "$(length(restart.weak)) entries, mesh " *
                                    "needs $(sq.N)"))
            chi = clamp.(copy(restart.weak), 0.0, 1.0)
            chi_prev = restart.weak_prev === nothing ? copy(chi) :
                       clamp.(copy(restart.weak_prev), 0.0, 1.0)
            length(chi_prev) == sq.N ||
                throw(ArgumentError("checkpoint previous weakening field has " *
                                    "$(length(chi_prev)) entries, mesh needs " *
                                    "$(sq.N)"))
        else
            chi = fill(weak_chi0, sq.N)
            chi_prev = copy(chi)
        end
        chi_iter = copy(chi)
    end
    # Maxwell stress history (2b): three Q2 nodal tensor components `txx`,
    # `tyy`, `txy` on the same scalar layout as `chi`.  `nothing` when
    # `pr.shear_modulus <= 0` (the default), in which case every stress-history
    # code path is skipped and the run is bit-identical to the pre-2b behaviour.
    # They are the *deviatoric* stress: `_advance_maxwell_stress!` projects the
    # strain rate, and the out-of-plane normal stress is implied by trace zero
    # (`tau_zz = -(tau_xx + tau_yy)`), not stored -- no assembly reads it.  They
    # are advanced once per step, and the previous step's value drives the
    # momentum solve through the elastic force term `div(a tau^n)` (2b step 2);
    # they are carried through AMR and the checkpoint exactly like `chi`.
    stress = pr.shear_modulus > 0.0
    txx = nothing
    tyy = nothing
    txy = nothing
    if stress
        # A checkpoint written before the stress history existed has none of the
        # three datasets, so `restart.txx` is `nothing`: the history restarts
        # from zero (the only self-consistent choice -- a file without it cannot
        # supply it).  Every file this version writes carries all three.
        any_stress = restart !== nothing &&
                     (restart.txx !== nothing || restart.tyy !== nothing ||
                      restart.txy !== nothing)
        if any_stress
            (restart.txx !== nothing && restart.tyy !== nothing &&
             restart.txy !== nothing) ||
                throw(ArgumentError("checkpoint has a partial Maxwell stress " *
                                    "state: txx/tyy/txy must all be present"))
            length(restart.txx) == sq.N ||
                throw(ArgumentError("checkpoint txx stress field has " *
                                    "$(length(restart.txx)) entries, mesh " *
                                    "needs $(sq.N)"))
            length(restart.tyy) == sq.N ||
                throw(ArgumentError("checkpoint tyy stress field has " *
                                    "$(length(restart.tyy)) entries, mesh " *
                                    "needs $(sq.N)"))
            length(restart.txy) == sq.N ||
                throw(ArgumentError("checkpoint txy stress field has " *
                                    "$(length(restart.txy)) entries, mesh " *
                                    "needs $(sq.N)"))
            txx = copy(restart.txx)
            tyy = copy(restart.tyy)
            txy = copy(restart.txy)
        else
            txx = zeros(Float64, sq.N)
            tyy = zeros(Float64, sq.N)
            txy = zeros(Float64, sq.N)
        end
    end
    edot_field = nothing
    eta_for_energy = nothing
    W_last = 0.0; U_last = 0.0; bal_last = 0.0
    # Timings of the last `solve_flow!` call for `profile_steps=true`; the Refs
    # keep the closure box-free whether or not profiling is on.
    stokes_ns = Ref(0); work_ns = Ref(0)
    # Viscosity closure of the state the last `solve_flow!` used.  Diagnostics
    # only: the Stokes-failure line reports its eta range (docs/dev/CASE_REALISTIC.md
    # §13).  `nothing` before the first solve.
    eta_probe = Ref{Any}(nothing)
    # The same closure *before* the 2a Maxwell effective-viscosity softening.
    # The stress history (2b step 1) advances with the material viscosity, so
    # its Maxwell time is eta/G and not the softened eta_eff/G; `_advance`
    # samples this closure at the nodes after the step has converged.  Only set
    # when `shear_modulus > 0` (the state is `nothing` otherwise).
    eta_base_probe = Ref{Any}(nothing)
    # Co-rotation validity band (G254): run maxima of the three per-step
    # rotations `_advance_maxwell_stress!` returns for the steps that advanced
    # the stress, plus the bookkeeping of the one warning per run.  Pure
    # reporting state: no numeric result reads it, and with `shear_modulus = 0`
    # the stress path never runs (`_advance` is not called), so a G = 0 run pays
    # nothing and logs nothing.
    corot_band = (two_w_dt=0.0, two_taum_w=0.0, resolved_rotation=0.0)
    corot_band_steps = 0
    corot_band_warned = false
    # Maxwell regime (G255): the material/effective viscosity ranges of the last
    # stress step, so the end-of-run record repeats the same measurement as the
    # first-step one.  `nothing` until a stress step has run; with
    # `shear_modulus = 0` (or `regime_report = false`) nothing is computed, so a
    # G = 0 run pays nothing and logs nothing.
    regime_last = nothing
    regime_reported = false
    # eta_ref convention (G257): one report at the first flow solve and one at the
    # end of the run, each a pure measurement of the closure the solve used.  With
    # `Params.eta_ref == 1.0` (the historical convention) nothing is sampled and
    # nothing is logged, so the default path stays bit-identical.
    eta_ref_reported = Ref(false)

    function solve_flow!(; stokes_scaling::Symbol=:none)
        leaves = Set(sq.leaves)
        th = theta
        Tfun = (x, y) -> _semquad_eval_at(sq, th, x, y, leaves)
        # With yielding on, the plastic branch needs the strain rate and the
        # pressure of the previous iterate (lagged Picard).  `pic_eta_relax < 1`
        # additionally averages the nodal strain-rate field over the Picard
        # iterations (ASPECT-style damping); the default 1.0 keeps the pointwise
        # evaluation and is bit-identical to the pre-averaging behaviour.
        local eta_fun
        if tosi
            # Tosi et al. (2015) law: needs T, the strain rate and the height.
            if pic_eta_relax >= 1.0
                eta_fun = (x, y) -> begin
                    T = Tbar(y, pr.H) + _semquad_eval_at(sq, th, x, y, leaves)
                    ed = semquad_strain_rate_at(sq, ux, uz, x, y, leaves)
                    semquad_eta_tosi(T, ed, y, pr.tosi)[1]
                end
            else
                ed_new = semquad_strain_rate_field(sq, ux, uz)
                edot_field = edot_field === nothing ? ed_new :
                             blend_field(edot_field, ed_new, pic_eta_relax)
                eta_fun = (x, y) -> begin
                    T = Tbar(y, pr.H) + _semquad_eval_at(sq, th, x, y, leaves)
                    ed = _semquad_eval_at(sq, edot_field, x, y, leaves)
                    semquad_eta_tosi(T, ed, y, pr.tosi)[1]
                end
            end
        elseif yielding
            if pic_eta_relax >= 1.0
                eta_fun = (x, y) -> semquad_eta_at(sq, th, ux, uz, pv, ss.sp,
                                                   chi_iter, x, y, pr, leaves)
            else
                ed_new = semquad_strain_rate_field(sq, ux, uz)
                edot_field = edot_field === nothing ? ed_new :
                             blend_field(edot_field, ed_new, pic_eta_relax)
                eta_fun = (x, y) -> semquad_eta_at(sq, th, edot_field, pv,
                                                   ss.sp, chi_iter, x, y, pr,
                                                   leaves)
            end
        elseif composite
            # Diffusion + dislocation creep: the law needs the strain rate, so
            # it is evaluated from the current (lagged) velocity exactly like the
            # :tosi branch; `pic_eta_relax < 1` averages the nodal strain-rate
            # field over the Picard iterations (ASPECT-style damping).
            if pic_eta_relax >= 1.0
                eta_fun = (x, y) -> begin
                    T = Tbar(y, pr.H) + _semquad_eval_at(sq, th, x, y, leaves)
                    ed = semquad_strain_rate_at(sq, ux, uz, x, y, leaves)
                    e, cl = eta_composite_at(T, ed, y, pr)
                    pic_diag && _count_eta_clamp(cl)
                    e
                end
            else
                ed_new = semquad_strain_rate_field(sq, ux, uz)
                edot_field = edot_field === nothing ? ed_new :
                             blend_field(edot_field, ed_new, pic_eta_relax)
                eta_fun = (x, y) -> begin
                    T = Tbar(y, pr.H) + _semquad_eval_at(sq, th, x, y, leaves)
                    ed = _semquad_eval_at(sq, edot_field, x, y, leaves)
                    e, cl = eta_composite_at(T, ed, y, pr)
                    pic_diag && _count_eta_clamp(cl)
                    e
                end
            end
        else
            eta_fun = (x, y) -> semquad_eta_at(sq, th, x, y, pr, leaves)
        end
        # Weak-zone hook: one multiplicative factor on whatever the law returned,
        # so no law has to know about it.  `nothing` (the default) leaves the
        # closure untouched, bit for bit.
        if pr.eta_multiplier !== nothing
            base_eta = eta_fun
            mulf = pr.eta_multiplier
            eta_fun = (x, y) -> base_eta(x, y) * mulf(x, y)
        end
        # G257: the Rayleigh-number reference viscosity.  `eta_ref` is the
        # viscosity the Ra is defined with, in the units the law returns, so the
        # flow solve uses eta/eta_ref in the same wrapper chain as the weak-zone
        # multiplier and the Maxwell closure below: rescaling the law by c and
        # eta_ref by c cancels exactly, while rescaling the law alone does not.
        # `eta_ref == 1.0` (the default) skips the wrapper, so every historical
        # run stays bit-identical.
        if pr.eta_ref != 1.0
            base_ref = eta_fun
            eta_ref_value = pr.eta_ref
            eta_fun = (x, y) -> base_ref(x, y) / eta_ref_value
        end
        # Maxwell relaxation in the effective-viscosity form.  The step enters, so
        # the closure is rebuilt every step; `shear_modulus <= 0` (the default)
        # skips it and leaves the law bit-identical.
        elastic_stress = nothing
        if pr.shear_modulus > 0.0
            # The stress history (2b step 1) relaxes on the *material* Maxwell
            # time eta/G, so keep the law before the softening below.
            eta_base_probe[] = eta_fun
            base_ve = eta_fun
            Gv = pr.shear_modulus
            eta_fun = (x, y) -> begin
                e = base_ve(x, y)
                1.0 / (1.0 / e + 1.0 / (Gv * dt))
            end
            # 2b step 2: the elastic force term of *this* solve.  The updated
            # Maxwell stress is
            #
            #     tau^{n+1} = 2 eta_eff ehat^{n+1} + a R(tau^n),
            #         R(tau) = tau + dt (W tau - tau W),  a = eta/(eta + G dt),
            #
            # with `ehat` the deviatoric strain rate (G249) and `R` the
            # co-rotational (Jaumann) rotation (G250); the first term is already
            # in the operator, and the explicit load assembled here is
            # `a tau^n`, i.e. without the rotation increment -- the deliberate
            # one-step lag documented on `maxwell_force_stress`.  `txx` still
            # holds `tau^n` here: the advance happens after this step's
            # momentum, so a same-step update can never feed back (G247 (d)).
            # The material viscosity is sampled from the same closure the stress
            # advance uses at the end of the step (`eta_base_probe[]`).
            elastic_stress = maxwell_force_stress(
                txx, tyy, txy, _nodal_eta(base_ve, sq), dt, Gv)
        end
        eta_for_energy = eta_fun
        # Buoyancy.  With a reference state this is the TALA/ALA form
        # rho(y) alpha(y) (T - Tad) instead of the full T: the hydrostatic part of
        # the base state is then balanced exactly, so theta = 0 with u = 0 is a
        # genuine fixed point (and the depth-dependent terms of the reference
        # state do not drive a spurious flow).  Without a reference state the
        # expression is unchanged, bit for bit.
        # Density hook (G248): `rho_multiplier` multiplies the density where the
        # buoyancy uses it; `nothing` keeps the historical closure and never
        # evaluates it.  A uniform `c` is exactly `Ra -> Ra*c`.
        rmul = pr.rho_multiplier
        fz = if pr.refstate === nothing
            rmul === nothing ? ((x, y) -> pr.Ra * Tfun(x, y)) :
                ((x, y) -> pr.Ra * rmul(x, y) * Tfun(x, y))
        else
            rs = pr.refstate
            # Tfun is already the perturbation theta = T - Tad, so the TALA
            # buoyancy rho alpha (T - Tad) is rho alpha theta; subtracting Tad
            # again here would be a double count.
            rmul === nothing ?
                ((x, y) -> pr.Ra * semquad_rho(rs, y) * semquad_alpha(rs, y) *
                           Tfun(x, y)) :
                ((x, y) -> pr.Ra * semquad_rho(rs, y) * semquad_alpha(rs, y) *
                           rmul(x, y) * Tfun(x, y))
        end
        # Phase transitions: the anomaly with respect to the base state adds a
        # buoyancy term; with no phases `fz` is used as is (bit identical).
        if !isempty(phases)
            fz_thermal = fz
            fz = (x, y) -> fz_thermal(x, y) +
                           semquad_phase_buoyancy(pr, phases, y, Tfun(x, y))
        end
        # Keep the viscosity closure of the state this solve *used*, so the
        # Stokes-failure diagnostic can report the eta range without rebuilding
        # it from a later state (the error is raised after `solve_flow!`).
        eta_probe[] = eta_fun
        tp = profile_steps ? time_ns() : 0
        u, pp, _, _, _ = solve_semquad_stokes_var(ss, qt, eta_fun,
                                                  (x, y) -> 0.0, fz,
                                                  (x, y) -> 0.0, (x, y) -> 0.0;
                                                  mass_rho=mass_rho_fun(pr),
                                                  mass_form=mass_form,
                                                  mass_drho=mass_drho_fun(pr),
                                                  mass_rho_mult=pr.rho_multiplier,
                                                  compress_block=compress_block_fun(ss, qt, pr),
                                                  deviatoric=pr.deviatoric,
                                                  elastic_stress=elastic_stress,
                                                  bc=pr.bc, linear=linear,
                                                  wall_bc=pr.wall_bc,
                                                  wall_u=pr.wall_u,
                                                  stokes_scaling=stokes_scaling,
                                                  auto_fallback_tol=stokes_tol,
                                                  stokes_accept_tol=stokes_tol,
                                                  schur_maxiter=schur_maxiter,
                                                  schur_precond=schur_precond,
                                                  mg_operator=mg_operator,
                                                  mg_vel_solver=mg_vel_solver,
                                                  mg_smoothing=mg_smoothing,
                                                  mg_omega=mg_omega,
                                                  mg_pressure_eta_tol=mg_pressure_eta_tol,
                                                  mg_threads=mg_threads,
                                                  mg_tol=mg_tol)
        profile_steps && (stokes_ns[] = time_ns() - tp)
        # Under-relax the velocity across Picard iterations.  The strain-rate
        # dependent laws feed eps -> eta -> u back into each iteration, and with
        # pic_u_relax = 1 an iterated solve can diverge; blending the new solve
        # with the previous iterate damps that loop.  1.0 keeps the old
        # behaviour bit for bit.
        if pic_u_relax >= 1.0
            for g in 1:sq.N
                ux[g] = u[g]
                uz[g] = u[sq.N + g]
            end
        else
            for g in 1:sq.N
                ux[g] = (1.0 - pic_u_relax) * ux[g] + pic_u_relax * u[g]
                uz[g] = (1.0 - pic_u_relax) * uz[g] + pic_u_relax * u[sq.N + g]
            end
        end
        for g in 1:ss.sp.N
            pv[g] = pp[g]
        end
        velocity_stale = false
        # G257: the eta_ref convention, reported once at the first flow solve.
        # Gated on `eta_ref != 1.0`, so every default-convention run logs exactly
        # what it logged before.  The measurement samples the closure the solve
        # used (post-solve velocity, start-of-step temperature); the pic_diag eta
        # counters are saved and restored so it cannot change the reported clamp
        # fraction of the run.
        if pr.eta_ref != 1.0 && !eta_ref_reported[]
            eta_ref_reported[] = true
            calls_before = _SEMQUAD_ETA_CALLS[]
            clamp_before = _SEMQUAD_ETA_CLAMP[]
            stats = _eta_ref_field_stats(sq, qt, eta_fun, pr)
            _SEMQUAD_ETA_CALLS[] = calls_before
            _SEMQUAD_ETA_CLAMP[] = clamp_before
            @info _eta_ref_message(pr, stats, step + 1, t, false,
                                   eta_ref_scale)
        end
        # Work and dissipation must be evaluated with the *same* temperature and
        # viscosity the Stokes solve used; using the end-of-step theta would mix
        # a new temperature with an old velocity and break Ra*W = U.
        tp = profile_steps ? time_ns() : 0
        W_last, U_last = semquad_work_dissipation(sq, qt, pr, th, ux, uz, eta_fun;
                                                  rho_multiplier=pr.rho_multiplier)
        profile_steps && (work_ns[] = time_ns() - tp)
        bal_last = (pr.Ra * W_last - U_last) /
                   max(abs(pr.Ra * W_last), abs(U_last), 1.0e-300)
        return nothing
    end

    # Viscosity factory for `pic_mode=:newton`: given a state `(theta, ufull)`
    # (with `ufull = vcat(ux, uz)`) it returns
    # `(eta_fun, deta_T_fun, deta_eps_fun)` evaluated at that state, plus the
    # optional Drucker-Prager entries `deta_dp_fun(x,y,edot,p)` (4th) and
    # `deta_dchi_fun(x,y,edot,p)` (5th) when a plastic branch / weakening field
    # is active (docs/dev/PHYSICS.md §8.4, §9.4).  This is the only
    # rheology-dependent part of the per-step Newton (`semquad_step_newton`),
    # so the solver itself is rheology-agnostic.
    # `deta_eps_fun(x, y, edot)` receives the strain rate that the assembler
    # computed from the base velocity, so `Kgeo` sees exactly the field the
    # viscosity was evaluated with.  `pfull` is the current pressure iterate
    # (needed by the Drucker-Prager yield stress); the 2-argument call is kept
    # for a state whose pressure is not available (then p = 0 at the quadrature
    # points, which is only meaningful for a pressure-independent yield).
    # `newton_make_eta_law` is the raw law; the public factory below adds the 2a
    # softening when `shear_modulus > 0` (the Newton residual must be assembled
    # with `eta_eff`, like the Picard sweeps it continues from).
    function newton_make_eta_law(th::Vector{Float64}, ufull::Vector{Float64},
                                 pfull::Union{Nothing,Vector{Float64}}=nothing)
        lvs = Set(sq.leaves)
        Nn = sq.N
        uxv = ufull[1:Nn]
        uzv = ufull[(Nn + 1):(2Nn)]
        T_at(x, y) = Tbar(y, pr.H) + _semquad_eval_at(sq, th, x, y, lvs)
        ed_at(x, y) = semquad_strain_rate_at(sq, uxv, uzv, x, y, lvs)
        if pr.eta_law === :composite
            eta_f = (x, y) -> eta_composite_at(T_at(x, y), ed_at(x, y), y, pr)[1]
            dT_f = (x, y) -> deta_composite_at(T_at(x, y), ed_at(x, y), y, pr)[1]
            de_f = (x, y, ed) -> deta_composite_at(T_at(x, y), ed, y, pr)[2]
            return eta_f, dT_f, de_f
        elseif pr.eta_law === :tosi
            eta_f = (x, y) -> semquad_eta_tosi(T_at(x, y), ed_at(x, y), y,
                                               pr.tosi)[1]
            dT_f = (x, y) -> semquad_deta_tosi(T_at(x, y), ed_at(x, y), y,
                                               pr.tosi)[2]
            de_f = (x, y, ed) -> semquad_deta_tosi(T_at(x, y), ed, y, pr.tosi)[1]
            return eta_f, dT_f, de_f
        elseif yield_active(pr)
            p_at = (x, y) -> pfull === nothing ? 0.0 :
                             _semquad_eval_at(ss.sp, pfull, x, y, lvs)
            c_at = chi === nothing ? ((x, y) -> 0.0) :
                    ((x, y) -> _semquad_eval_at(sq, chi, x, y, lvs))
            eta_f = (x, y) -> semquad_visco_plastic_eta(T_at(x, y), y,
                                                        ed_at(x, y), p_at(x, y),
                                                        c_at(x, y), pr)
            dT_f = (x, y) -> semquad_dp_deta_T(T_at(x, y), y, ed_at(x, y),
                                               p_at(x, y), c_at(x, y), pr)
            de_f = (x, y, ed) -> semquad_deta_dp_eps(T_at(x, y), y, ed,
                                                     p_at(x, y), c_at(x, y), pr)
            dp_f = (x, y, ed, pq) -> semquad_deta_dp(T_at(x, y), y, ed, pq,
                                                     c_at(x, y), pr)
            # The chi-derivative block is assembled on the *state's* pressure
            # (the closure above), so it takes only (x, y, edot).
            dchi_f = (x, y, ed) -> semquad_deta_chi(T_at(x, y), y, ed,
                                                    p_at(x, y), c_at(x, y), pr)
            return eta_f, dT_f, de_f, dp_f, dchi_f
        end
        eta_f = (x, y) -> eta_at(T_at(x, y), y, pr)[1]
        dT_f = (x, y) -> deta_dT_at(T_at(x, y), y, pr)
        return eta_f, dT_f, nothing
    end

    # 2a softening for the Newton path: `eta_eff = 1/(1/eta + 1/(G dt))`, with the
    # chain rule on every derivative closure (see `_ve_soften_eta_factory`).
    # `G = 0` returns the raw law unchanged, bit for bit.
    function newton_make_eta(th::Vector{Float64}, ufull::Vector{Float64},
                             pfull::Union{Nothing,Vector{Float64}}=nothing)
        mk = newton_make_eta_law(th, ufull, pfull)
        return _ve_soften_eta_factory(mk, pr.shear_modulus, dt)
    end

    SEMQUAD_STEP_PROFILE[] = profile_steps ? Any[] : nothing
    SEMQUAD_PIC_DIAG[] = pic_diag ? Any[] : nothing
    # Fresh `:auto` bookkeeping for this run.  `LAST_SEMSTOKES_AUTO_CHOICE` is a
    # process-global Ref, and a run that performs no Stokes solve (maxsteps=0)
    # would otherwise leave a *previous* run's `:mg` decision visible to the
    # `steady_newton` hand-off below and to the caller's logging.  `:none` means
    # "no auto solve happened in this run", which the steady entry maps to
    # `:direct`; the first real solve overwrites it with its own decision.
    linear === :auto &&
        (LAST_SEMSTOKES_AUTO_CHOICE[] = _SEMSTOKES_AUTO_CHOICE_NONE)
    # `pic_reject` needs the previous iterate's flow state; the buffers are
    # re-allocated after an AMR cycle changes `sq.N`.
    pic_reject_any = pic_reject && maxpic > 1
    ux_saved = Float64[]
    uz_saved = Float64[]
    pv_saved = Float64[]
    # Declared in the function scope because it is assigned inside the Picard
    # loop but read by the step diagnostics afterwards (a `while` body is its
    # own scope in Julia); only the opt-in `extdiag` diagnostic reads it.
    lat_theta_used = nothing

    while step < maxsteps && t < tstop
        tp_step = profile_steps ? time_ns() : 0
        tprof = profile_steps ? zeros(Float64, 6) : nothing
        theta_old = copy(theta)
        theta_iter = copy(theta)
        # The weakening iterate starts at the start-of-step level chi^n: the
        # first flow solve of the step uses it (operator splitting; docs/
        # RHEOLOGY.md §9.3).  For maxpic > 1 the chi transport is then iterated
        # inside the Picard loop, so the fixed point is self-consistent.
        chi_iter = weak ? copy(chi) : nothing
        pic_changes = Float64[]
        # per-iteration bookkeeping for the two globalization options
        n_reject = 0
        prev_clamp = NaN
        prev_res = NaN
        step_clamp = NaN
        step_res = NaN
        newton_iters = 0
        newton_fallback = false
        newton_res = NaN
        newton_done = false
        it = 0
        while it < maxpic
            it += 1
            theta = theta_iter
            # `pic_reject` restores the previous iterate's flow state when a
            # trial update is rejected, so it must be saved before `solve_flow!`
            # overwrites it.
            if pic_reject_any
                if length(ux_saved) != length(ux)
                    ux_saved = zeros(Float64, length(ux))
                    uz_saved = zeros(Float64, length(uz))
                    pv_saved = zeros(Float64, length(pv))
                end
                copyto!(ux_saved, ux)
                copyto!(uz_saved, uz)
                copyto!(pv_saved, pv)
            end
            pic_diag && (_SEMQUAD_ETA_CLAMP[] = 0; _SEMQUAD_ETA_CALLS[] = 0)
            solve_flow!(stokes_scaling=stokes_scaling)
            if tprof !== nothing
                tprof[1] += stokes_ns[] / 1.0e9
                tprof[2] += work_ns[] / 1.0e9
            end
            if isfinite(stokes_tol) && LAST_SEMSTOKES_RESIDUAL[] > stokes_tol
                # The direct path has already escalated its shift ladder and
                # refined (src/stokes/, docs/dev/CASE_REALISTIC.md
                # §13); this error is the last resort, so report the numbers that
                # identify the state.  Written to *stderr*, the same unbuffered
                # stream as the error, so it is adjacent to it in a redirected
                # log (the stdout step records are block-buffered, §11.1).
                sp = LAST_SEMSTOKES_RESID_SPLIT[]
                er = eta_probe[] === nothing ? nothing :
                     semquad_eta_range(sq, eta_probe[], pr)
                @printf(stderr,
                        "[case] STOKES-FAIL step=%d res=%.6e (vel=%.3e div=%.3e) stokes_tol=%.1e shift=%.0e refine=%d leaves=%d N=%d saddle=%d eta=[%.3e,%.3e] eta_clip=%.3f%%\n",
                        step, LAST_SEMSTOKES_RESIDUAL[], sp.vel, sp.div,
                        stokes_tol, LAST_SEMSTOKES_SHIFT[],
                        LAST_SEMSTOKES_REFINE_ITERS[], length(qt.leaves), sq.N,
                        2 * sq.N + ss.sp.N,
                        er === nothing ? NaN : er.min,
                        er === nothing ? NaN : er.max,
                        er === nothing ? NaN : 100.0 * er.nclip / er.n)
                lad = LAST_SEMSTOKES_LADDER[]
                if lad !== nothing
                    @printf(stderr, "[case] STOKES-FAIL ladder: %s\n",
                            join([r[1] === :refine ? @sprintf("refine:%d:%.3e", LAST_SEMSTOKES_REFINE_ITERS[], r[2]) :
                                   r[1] === :equilibrated ? @sprintf("equil:%.3e", r[2]) :
                                   @sprintf("%.0e:%.3e", r[1], r[2]) for r in lad], " "))
                end
                flush(stderr)
                # Name the fix per path: `schur_maxiter` only caps the
                # `:schur_cg` iteration and is inert on the `:auto`/`:mg`
                # paths, so the old catch-all text misdirected an auto-MG
                # failure.  `linear` here is still the driver's kwarg, i.e.
                # `:auto` when the selector made the choice.
                fix = linear === :auto ?
                      "the solve auto-selected " *
                      "$(LAST_SEMSTOKES_AUTO_CHOICE[].choice); pass " *
                      "linear=:direct explicitly to force the direct shift ladder" :
                      linear === :mg ?
                      "raise mg_tol/mg_maxit or pass linear=:direct explicitly" :
                      "increase schur_maxiter or use linear=:direct"
                error("Stokes residual $(LAST_SEMSTOKES_RESIDUAL[]) exceeds " *
                      "stokes_tol=$stokes_tol at step $step; $fix (the direct " *
                      "shift ladder and its refinement already ran, " *
                      "docs/dev/CASE_REALISTIC.md §13)")
            end
            verbose && linear === :schur_cg &&
                @printf("    Stokes res=%.2e nit=%d\n",
                        LAST_SEMSTOKES_RESIDUAL[], LAST_SEMSTOKES_SCHUR_ITERS[])
            if it == 1 && cfl > 0.0
                # Update dt on the mesh and velocity of the current step.  This
                # must happen here (not after the AMR block): the AMR block
                # rebuilds the mesh and zeroes ux/uz, so an end-of-step update
                # would see umax=0 and dt could only shrink, never grow.  At
                # step 0 the initial Stokes solve can already give |u|~1e8 at
                # Ra=1e8, so dt0 is cut to the CFL value; later steps follow the
                # same 5%-growth / free-shrink rule as the uniform driver
                # (timestep.jl), with a hard override when the clamped dt would
                # still exceed the CFL limit.
                umax = maximum(hypot.(ux, uz))
                if umax > 0.0
                    hmin = minimum(min(leaf_hx(qt, lf[1]), leaf_hy(qt, lf[1])) for lf in qt.leaves)
                    dt_cfl = cfl * hmin / umax
                    if step == 0
                        dt = min(dt, dt_cfl)
                    else
                        dt_target = min(dtmax, dt_cfl)
                        dt = min(max(dt_target, 0.95 * dt), dt_growth * dt)
                        dt > dt_cfl && (dt = dt_cfl)
                    end
                    dt = min(dtmax, max(dtmin, dt))
                end
            end
            ev_field = ev > 0.0 ? zeros(Float64, length(qt.leaves)) : nothing
            if pic_mode === :newton && it == 1
                # Globalization (ii): damped Newton on the coupled per-step
                # residual, started from the Stokes solution of this first
                # sweep.  The analytic tangents are built by
                # `semquad_step_newton` from the viscosity factory.
                # 2b step 2: the Newton's momentum residual carries the same
                # lagged elastic force `a tau^n` as the Picard sweep it starts
                # from (`tau = txx/tyy/txy` still holds the previous step's
                # stress).  The material viscosity of the start state sets `a`,
                # and it is frozen with the load for the step, exactly like the
                # load itself; the stiffness inside the Newton uses the softened
                # `newton_make_eta`, so this is the same backward-Euler split.
                ve_elastic = if pr.shear_modulus > 0.0
                    eta_mat = newton_make_eta_law(theta_iter, vcat(ux, uz),
                                                  pv)[1]
                    maxwell_force_stress(txx, tyy, txy,
                                         _nodal_eta(eta_mat, sq), dt,
                                         pr.shear_modulus)
                else
                    nothing
                end
                nrs = semquad_step_newton(ss, qt, pr, theta_prev, theta_iter,
                                          ux, uz, pv, newton_make_eta;
                                          dt=dt, dt_prev=dt_prev,
                                          firststep=(step == 0),
                                          advection=advection, ev=ev,
                                          ev_cmax=ev_cmax, ev_alpha=ev_alpha,
                                          ev_timestep=step, tol=pic_tol,
                                          maxit=newton_maxit,
                                          krylov=newton_krylov,
                                          restart=newton_restart,
                                          ls_halvings=newton_ls,
                                          geo_spd_safety=newton_geo_safety,
                                          phases=phases,
                                          latent_scheme=latent_scheme,
                                          elastic_stress=ve_elastic,
                                          chi=chi_iter, chi_n=chi,
                                          chi_nm1=chi_prev,
                                          verbose=verbose)
                newton_iters = nrs.iters
                newton_res = nrs.res
                if nrs.converged
                    theta_iter = nrs.theta
                    ux .= nrs.ux
                    uz .= nrs.uz
                    pv .= nrs.pv
                    weak && (chi_iter = nrs.chi)
                    nrs.ev_field === nothing || (ev_field = copy(nrs.ev_field))
                    push!(pic_changes, nrs.change)
                    step_clamp = pic_diag && _SEMQUAD_ETA_CALLS[] > 0 ?
                                 _SEMQUAD_ETA_CLAMP[] / _SEMQUAD_ETA_CALLS[] :
                                 NaN
                    step_res = LAST_SEMSTOKES_RESIDUAL[]
                    newton_done = true
                    break
                end
                # Globalization (iii): the Newton failed, so Picard takes over
                # from the start-of-step state (ux/uz/pv are untouched — the
                # Newton works on copies — and `it` stays 1).
                newton_fallback = true
                verbose && @printf("    Newton failed (change=%.3e res=%.3e); falling back to Picard\n", nrs.change, nrs.res)
            end
            tp = profile_steps ? time_ns() : 0
            # `theta_n` is the **start-of-step** temperature, not the current
            # iterate: the BDF2/backward-Euler mass term of the step must use
            # `theta^n`, so the fixed point of the Picard loop is the implicit
            # step itself.  Passing `theta_iter` here (the historical code)
            # makes the fixed point a *different* scheme -- for the first step
            # the mass terms cancel identically and the iteration converges to
            # the steady energy equation, for later steps to a backward-Euler
            # step with dt/c.  `theta_iter == theta_old` on the first iteration,
            # so `maxpic = 1` is bit-identical to the historical behaviour.
            # `lat_theta_used` is the temperature at which this sweep evaluates
            # the latent apparent capacity; the optional exact-residual
            # diagnostic (`extdiag`) rebuilds the operator with the same value.
            # Radiogenic decay: the source is evaluated at the end of the step,
            # which is the time level the BDF2 residual sits on.  `t` is the
            # start-of-step time and is advanced at the end of the loop.
            heat_scale = isfinite(pr.H_half_life) ?
                exp(-log(2.0) * (t + dt) / pr.H_half_life) : 1.0
            lat_theta_used = latent_scheme === :lagged ? nothing : theta_iter
            theta_new = solve_semquad_energy(sq, qt, pr, theta_old, theta_prev,
                                             ux, uz, dt, dt_prev, step == 0;
                                             advection=advection,
                                             ev=ev, ev_cmax=ev_cmax,
                                             ev_alpha=ev_alpha,
                                             ev_timestep=step,
                                             ev_out=ev_field,
                                             eta_fun=eta_for_energy,
                                             phases=phases,
                                             latent_theta=lat_theta_used,
                                             heating_scale=heat_scale)
            tprof !== nothing && (tprof[3] += (time_ns() - tp) / 1.0e9)
            change = norm(theta_new - theta_iter) / max(norm(theta_new), 1.0e-30)
            # `pic_reject` can undo this sweep; the chi iterate of the state the
            # flow was solved with must then be restored too.
            chi_iter_saved = weak ? copy(chi_iter) : nothing
            # Strain weakening (docs/dev/PHYSICS.md §9.3): one implicit BDF2
            # transport with the velocity of this sweep and the *start-of-step*
            # levels chi^n/chi^(n-1); the yielding indicator sees the current
            # iterate chi_iter (the field the flow was solved with).  The update
            # is under-relaxed like theta so the Picard fixed point is the same.
            if weak
                chi_target = solve_semquad_weakening(
                    sq, ss.sp, qt, pr, chi, chi_prev, theta_new, ux, uz, pv,
                    dt, dt_prev, step == 0; advection=advection,
                    chi_src=chi_iter)
                chi_change = norm(chi_target - chi_iter) /
                             max(norm(chi_target), 1.0e-30)
                chi_iter = (1.0 - pic_relax) .* chi_iter .+
                           pic_relax .* chi_target
                change = max(change, chi_change)
            end
            verbose && @printf("    Picard it=%d change=%.3e\n", it, change)
            push!(pic_changes, change)
            # Clamp fraction and Stokes residual of the state this iteration
            # started from (the eta field is the one `solve_flow!` assembled).
            step_clamp = pic_diag && _SEMQUAD_ETA_CALLS[] > 0 ?
                         _SEMQUAD_ETA_CLAMP[] / _SEMQUAD_ETA_CALLS[] : NaN
            step_res = LAST_SEMSTOKES_RESIDUAL[]
            if pic_div_window > 0 &&
               picard_diverging(pic_changes; window=pic_div_window,
                                factor=pic_div_factor)
                error("Picard iteration is diverging at step $(step) " *
                      "(change=$(change)); reduce pic_relax or use a smaller " *
                      "dt / eta relaxation (pic_eta_relax)")
            end
            if pic_reject_any && it >= 2 && n_reject < pic_reject_max &&
               pic_relax > pic_relax_min
                # Globalization (i): trial-state rejection.  The eta field of the
                # state this iteration started from is clamp-dominated (an
                # absolute fraction above `pic_reject_clamp` that is also worse
                # than the previous iteration) or the Stokes relative residual
                # grew by more than `pic_reject_res`; the update is then
                # rejected, the previous iterate's flow is restored and the
                # iteration is redone with a halved relaxation.
                clamp_bad = isfinite(step_clamp) &&
                            step_clamp > pic_reject_clamp &&
                            step_clamp > prev_clamp
                res_bad = isfinite(prev_res) && prev_res > 0.0 &&
                          step_res > pic_reject_res * prev_res
                if clamp_bad || res_bad
                    copyto!(ux, ux_saved)
                    copyto!(uz, uz_saved)
                    copyto!(pv, pv_saved)
                    weak && (chi_iter = chi_iter_saved)
                    pop!(pic_changes)
                    pic_relax = max(pic_relax_min, 0.5 * pic_relax)
                    n_reject += 1
                    it -= 1
                    verbose && @printf("    Picard reject (clamp=%.3f res=%.1e) -> pic_relax=%.3g\n", step_clamp, step_res, pic_relax)
                    continue
                end
            end
            prev_clamp = step_clamp
            prev_res = step_res
            theta_iter = (1.0 - pic_relax) * theta_iter + pic_relax * theta_new
            if pic_adapt && length(pic_changes) >= 2
                pic_relax = picard_adapt_relax(pic_relax, pic_changes[end - 1],
                                               pic_changes[end];
                                               rmin=pic_relax_min)
            end
            change < pic_tol && break
        end
        if pic_diag
            push!(SEMQUAD_PIC_DIAG[],
                  (step=step, mode=newton_done ? :newton : :picard,
                   iters=newton_done ? newton_iters : length(pic_changes),
                   changes=copy(pic_changes), rejects=n_reject,
                   relax=pic_relax, clamp_frac=step_clamp,
                   stokes_res=step_res, newton_res=newton_res,
                   fallback=newton_fallback))
        end
        theta = theta_iter
        if weak
            # The converged step supplies the new BDF2 levels: chi^n -> chi.
            chi_prev = chi
            chi = clamp.(chi_iter, 0.0, 1.0)
        end
        # Maxwell stress history (2b): one backward-Euler step with the velocity
        # of the step that just converged, on the mesh it was solved on (the AMR
        # block below transfers the result, exactly like `chi`).  The material
        # viscosity is sampled at the nodes from the closure of the last flow
        # solve, before the 2a effective-viscosity softening, so tau_M = eta/G is
        # the material Maxwell time.  `dt` is this step's step (updated by the
        # CFL block at `it == 1`), the same one the energy solve used.
        #
        # This is deliberately *after* the converged momentum solve: the step's
        # Stokes solve used `a tau^{n-1}` (the elastic force assembled in
        # `solve_flow!`), and the result here is `tau^n`, which only the *next*
        # step's momentum will see.  Advancing before the momentum solve (or
        # re-solving after this line) would feed a same-step elastic stress back
        # into its own momentum; G247 (d) reconstructs both sides of the step and
        # would catch it.  The energy solve of a Picard sweep happens before this
        # line, inside the iteration, but the energy equation reads neither tau
        # nor the elastic load, so the momentum order is what the coupling needs.
        if stress && eta_base_probe[] !== nothing
            eta_nodes = _nodal_eta(eta_base_probe[], sq)
            band = _advance_maxwell_stress!(txx, tyy, txy, sq, ux, uz, dt,
                                            eta_nodes, pr.shear_modulus)
            # G255: the Maxwell regime, reported once at the first step that has
            # a stress update.  The stress update has just built the *material*
            # nodal viscosity and this step's `dt`, which are exactly the two
            # inputs of the 2a closure `eta_eff = 1/(1/eta + 1/(G dt))`, so the
            # measurement is taken on the field the solve saw and at the step it
            # used.  Diagnostic only: `regime_report = false` skips it and leaves
            # the run bit-identical.
            if regime_report
                regime_last = maxwell_regime(eta_nodes, dt, pr.shear_modulus)
                if !regime_reported
                    regime_reported = true
                    @info _maxwell_regime_message(regime_last, step + 1,
                                                  t + dt, dt,
                                                  pr.shear_modulus, false)
                end
            end
            # G254: accumulate the validity-band maxima and warn once, at the
            # first step that leaves the band.  `band` is `nothing` only for
            # G <= 0, which the `stress` guard already excludes; the explicit
            # check keeps this a pure diagnostic either way.
            if corotation_guard && band !== nothing
                corot_band = (two_w_dt=max(corot_band.two_w_dt, band.two_w_dt),
                              two_taum_w=max(corot_band.two_taum_w,
                                             band.two_taum_w),
                              resolved_rotation=max(corot_band.resolved_rotation,
                                                    band.resolved_rotation))
                corot_band_steps += 1
                if !corot_band_warned &&
                   band.resolved_rotation > MAXWELL_COROTATION_BOUND
                    corot_band_warned = true
                    @warn _corotation_band_warning(step + 1, t + dt,
                                                   band.two_w_dt,
                                                   band.two_taum_w,
                                                   band.resolved_rotation)
                end
            end
        end
        # Keep a handle on the second BDF2 level of the step that just finished:
        # `theta_prev` is overwritten below, and the opt-in exact-residual
        # diagnostic needs theta^(n-1) (no copy; the array is not mutated).
        theta_nm1_used = theta_prev
        theta_prev = theta_old
        t += dt
        step += 1

        if isfinite(cfl_tol)
            hmin = minimum(min(leaf_hx(qt, lf[1]), leaf_hy(qt, lf[1])) for lf in qt.leaves)
            umax = maximum(hypot.(ux, uz))
            cfl_step = umax * dt / hmin
            if cfl_step > cfl_tol
                error("wall CFL $(cfl_step) exceeds cfl_tol=$cfl_tol at " *
                      "step $step (dt=$dt, hmin=$hmin, umax=$umax); reduce dt")
            end
        end

        tp = profile_steps ? time_ns() : 0
        diag = _step_diagnostics(sq, qt, pr, theta, theta_prev, theta_nm1_used,
                                 ux, uz, dt, dt_prev, step, t, ev_field, phases,
                                 advection, eta_for_energy, W_last, U_last,
                                 bal_last; extdiag=extdiag, nu_cbf=nu_cbf,
                                 lat_theta_used=lat_theta_used)
        row = diag.row
        Nut = diag.Nut; Nub = diag.Nub; Vr = diag.Vr
        Ta = diag.Ta; closure = diag.closure
        push!(hist, row)
        tprof !== nothing && (tprof[4] += (time_ns() - tp) / 1.0e9)
        verbose && @printf("  step %4d t=%.3e dt=%.2e Nu=%.5f/%.5f Vrms=%.4e N=%d leaves=%d closure=%.3e\n",
                           step, t, dt, Nut, Nub, Vr, sq.N, length(qt.leaves), closure)

        # Time averages over t >= tavg_start (the diagnostic the earlier structured-grid layer has via
        # `tavg_start`); reported in the result as `avg`.
        if t >= tavg_start
            avg_n += 1
            avg_sut += Nut; avg_sub += Nub; avg_sv += Vr; avg_st += Ta
            avg_sut2 += Nut^2; avg_sub2 += Nub^2
            avg_sv2 += Vr^2; avg_st2 += Ta^2
        end

        # The AMR decision must depend only on `step`, not on whether this is the
        # final step of the run: otherwise a checkpoint written at the last step
        # stores a pre-AMR mesh, and a restart from it is not equivalent to the
        # uninterrupted run (which would have refined that step).
        if amr_every > 0 && step % amr_every == 0
            tp = profile_steps ? time_ns() : 0
            old_sq = sq
            ss, theta, theta_prev, nr = semquad_amr_passes!(
                ss, qt, theta, theta_prev;
                refine_frac=refine_frac, coarsen_frac=coarsen_frac,
                passes=amr_passes, max_level=max_level,
                max_leaves=max_leaves, wall_min_level=wall_min_level,
                wall_strip=wall_strip, symmetry=symmetry,
                transfer_mode=transfer_mode,
                hotspot_weight=hotspot_weight, hotspot_frac=hotspot_frac,
                hotspot_min_level=hotspot_min_level,
                hotspot_mode=hotspot_mode, hotspot_zmin=hotspot_zmin, zero_side_eta=zero_side_eta, side_strip=side_strip,
                top_wall_level=top_wall_level, top_wall_frac=top_wall_frac,
                amr_hysteresis=amr_hysteresis,
                amr_coarsen_gate=amr_coarsen_gate,
                # The in-loop indicator runs on the velocity the step just
                # solved, which is on the current mesh (`sq`).  `:theta`
                # ignores these; `:velocity`/`:both` need them or
                # `semquad_amr_passes!` would have to guess (the historical bug
                # was reading an undefined local and throwing on the first AMR
                # cycle -- G234 pins a real `amr_every=1` velocity cycle).
                ux=ux, uz=uz,
                indicator=amr_indicator)
            sq = ss.su
            # The weakening field lives on the scalar space, so it follows the
            # same transfer as theta (no wall Dirichlet data: `zero_walls=false`).
            if weak
                chi = transfer_semquad(old_sq, chi, sq; mode=transfer_mode,
                                       zero_walls=false, qt_new=qt)
                chi_prev = transfer_semquad(old_sq, chi_prev, sq;
                                            mode=transfer_mode,
                                            zero_walls=false, qt_new=qt)
                chi_iter = copy(chi)
            end
            # The Maxwell stress history is a set of scalar fields on the same
            # space, so it follows the same transfer as `chi` (2b step 1).
            if stress
                txx = transfer_semquad(old_sq, txx, sq; mode=transfer_mode,
                                       zero_walls=false, qt_new=qt)
                tyy = transfer_semquad(old_sq, tyy, sq; mode=transfer_mode,
                                       zero_walls=false, qt_new=qt)
                txy = transfer_semquad(old_sq, txy, sq; mode=transfer_mode,
                                       zero_walls=false, qt_new=qt)
            end
            ux = zeros(Float64, sq.N)
            uz = zeros(Float64, sq.N)
            pv = zeros(Float64, ss.sp.N)
            edot_field = nothing      # the mesh changed: the old field is stale
            ev_field = nothing        # same for the elementwise EV field
            # old tree hierarchies (`:auto` may resolve to MG on the new tree).
            # `_semquad_mg_drop_tree_entries!` keeps the per-level cache (G221):
            # a coarse level whose leaf set did not change survives the cycle.
            (linear === :mg || linear === :auto) && _semquad_mg_drop_tree_entries!()
            velocity_stale = true
            tprof !== nothing && (tprof[5] += (time_ns() - tp) / 1.0e9)
            verbose && @printf("    AMR: N=%d leaves=%d maxjump=%d refined=%d coarsened=%d merged=%d\n",
                               sq.N, length(qt.leaves), max_face_level_jump(qt), nr,
                               LAST_AMR_STATS[].coarsened, LAST_AMR_STATS[].merged)
            if verbose
                # Balanced-cycle diagnostics (docs/dev/AMR.md §4.1/§4):
                # the recirculation counter and the coarsening-gate ledger.
                d = LAST_AMR_BALANCED[]
                @printf("      balanced: gate_refused=%d extended=%d recirc=%d post=%+d tau_h=%.3e eligible=%d\n",
                        d.gate_refused, d.gate_extended, d.recirc, d.post,
                        d.tau_h, d.eligible)
            end
        end

        # The step that just finished used `dt`; that is also the BDF2
        # `dt_prev` of the next step.  It must be updated *before* the
        # checkpoint write: an in-loop checkpoint (the production cadence,
        # e.g. every 200 steps) otherwise stores the previous step's `dt` as
        # `dt_prev`, and a resume would re-run the next BDF2 step with a stale
        # time step even when every other field is exact.  `dt` itself was
        # updated at the CFL block above (`it == 1`), i.e. it is this step's dt.
        dt_prev = dt

        if callback !== nothing
            tp = profile_steps ? time_ns() : 0
            callback((step=step, t=t, dt=dt, theta=copy(theta),
                      ux=copy(ux), uz=copy(uz), sq=sq, qt=qt))
            tprof !== nothing && (tprof[6] += (time_ns() - tp) / 1.0e9)
        end

        if !isempty(checkpoint_path) && checkpoint_every > 0 &&
           step % checkpoint_every == 0
            save_semquad_state(checkpoint_path, qt, theta, theta_prev, ux, uz, pv;
                               t=t, dt=dt, dt_prev=dt_prev, step=step, p=p,
                               config=run_config, meta=checkpoint_meta,
                               ev=ev_field, weak=chi, weak_prev=chi_prev,
                               txx=txx, tyy=tyy, txy=txy,
                               cooldown=SEMQUAD_AMR_COOLDOWN[])
        end

        # Steady-state detection, same criterion as the uniform driver: the
        # relative theta change and |dNu| must both stay below steady_tol for
        # `steady_window` consecutive steps (only after t > tmin).
        if isfinite(steady_tol)
            change_t = norm(theta - theta_prev) / max(norm(theta), 1.0e-30)
            dnu = isfinite(Nu_prev) ? abs(Nut - Nu_prev) : Inf
            if t > tmin && change_t < steady_tol &&
               dnu < steady_tol * max(1.0, abs(Nut))
                steady_count += 1
                steady_count >= steady_window && break
            else
                steady_count = 0
            end
            Nu_prev = Nut
        end

        if tprof !== nothing
            push!(SEMQUAD_STEP_PROFILE[],
                  (step=step, t=t, dt=dt, leaves=length(qt.leaves), N=sq.N,
                   stokes=tprof[1], work=tprof[2], energy=tprof[3],
                   diag=tprof[4], amr=tprof[5], callback=tprof[6],
                   total=(time_ns() - tp_step) / 1.0e9))
        end
    end

    # The last AMR cycle resets ux/uz to zero on the new mesh; re-solve so the
    # returned velocity (and any CFL/vorticity diagnostic) belongs to the
    # returned theta/mesh instead of being a zero field.
    # The final checkpoint must describe the *trajectory* state even though the
    # returned velocity is re-solved below: capture the stale marker first and
    # write it into the file, so a restart zeroes the diagnostic velocity again
    # (see `save_semquad_state(velocity_stale=...)`).
    final_velocity_stale = velocity_stale
    if velocity_stale && step > 0
        solve_flow!(stokes_scaling=stokes_scaling)
    end

    # G257: the eta_ref convention, once more at the end of the run, on the
    # closure of the last flow solve (the final re-solve above has refreshed it if
    # the last step was an AMR cycle).  The developed state's interior Rayleigh
    # number is the quantity a reader lines a card up with a paper against.  The
    # first report's flag also suppresses this one for a run that solved nothing.
    if pr.eta_ref != 1.0 && eta_ref_reported[] && eta_probe[] !== nothing
        stats = _eta_ref_field_stats(sq, qt, eta_probe[], pr)
        @info _eta_ref_message(pr, stats, step, t, true, eta_ref_scale)
    end

    # Always leave a final checkpoint so a run that hits tstop/maxsteps can be
    # continued even when the last step is not a multiple of checkpoint_every.
    if !isempty(checkpoint_path) && step > 0
        save_semquad_state(checkpoint_path, qt, theta, theta_prev, ux, uz, pv;
                           t=t, dt=dt, dt_prev=dt_prev, step=step, p=p,
                           config=run_config, meta=checkpoint_meta,
                           ev=ev_field, weak=chi, weak_prev=chi_prev,
                           txx=txx, tyy=tyy, txy=txy,
                           cooldown=SEMQUAD_AMR_COOLDOWN[],
                           velocity_stale=final_velocity_stale)
    end

    # G255: report the Maxwell regime once more at the end of the run, with the
    # last stress step's material field and `dt` (the same measurement the
    # first-step record made, so a card can be diagnosed from either end of its
    # log).  Separate from the co-rotation report below: that one is an accuracy
    # band, this one is what viscosity field the solve actually saw.  A run with
    # `shear_modulus = 0`, and a `regime_report = false` run, log nothing here.
    if stress && regime_report && regime_last !== nothing
        @info _maxwell_regime_message(regime_last, step, t, dt,
                                      pr.shear_modulus, true)
    end

    # G254: report the co-rotation validity band once per run, after every
    # stress step, so a run that stays inside the band still states its
    # measured headroom (the warning above is only for leaving it).  A run with
    # `shear_modulus = 0` never advances the stress, so it logs nothing here.
    if stress && corotation_guard && corot_band_steps > 0
        @info _corotation_band_report(corot_band_steps, corot_band.two_w_dt,
                                      corot_band.two_taum_w,
                                      corot_band.resolved_rotation,
                                      corot_band_warned)
    end

    avg = avg_n > 0 ? (n=avg_n, t_start=tavg_start,
                       Nu_top=avg_sut / avg_n, Nu_bot=avg_sub / avg_n,
                       Vrms=avg_sv / avg_n, meanT=avg_st / avg_n,
                       Nu_top_std=sqrt(max(avg_sut2 / avg_n - (avg_sut / avg_n)^2, 0.0)),
                       Nu_bot_std=sqrt(max(avg_sub2 / avg_n - (avg_sub / avg_n)^2, 0.0)),
                       Vrms_std=sqrt(max(avg_sv2 / avg_n - (avg_sv / avg_n)^2, 0.0)),
                       meanT_std=sqrt(max(avg_st2 / avg_n - (avg_st / avg_n)^2, 0.0))) : nothing
    # Optional steady Newton on the final state.  `run_semquad_simulation`
    # always time-steps; `steady_newton=:auto|:map|:tosi|:dp` additionally calls
    # `semquad_steady_newton` on the returned state (the DP/Tosi variants need a
    # developed theta, which the time loop provides).  The default `:none`
    # leaves the historical behaviour bit-identical.
    steady = nothing
    if steady_newton !== :none
        # Steady Newton's inner solvers accept `:direct`/`:mg` only; an
        # `:auto` driver (the default since 2026-11-13) hands them the path its
        # last Stokes solve resolved to.  The record is reset at run start, so
        # `:none` here means no solve ran in this run and `:direct` is the
        # fallback that keeps the historical behaviour.
        steady_linear = linear
        if steady_linear === :auto
            c = LAST_SEMSTOKES_AUTO_CHOICE[].choice
            steady_linear = c === :mg ? :mg : :direct
        end
        steady = semquad_steady_newton(ss.su, qt, ss, pr, theta;
                                       kind=steady_newton, dt=dt,
                                       advection=advection, linear=steady_linear)
    end
    # Names of the columns that were appended after the 18 base ones, in row
    # order, so a writer can emit a header without duplicating the conditions
    # (examples/case_realistic_2d.jl).  The default run has none.
    hist_extra = String[]
    semquad_latent_active(pr, phases) && push!(hist_extra, "closure_nolatent")
    nu_cbf && append!(hist_extra, ["Nu_cbf_top", "Nu_cbf_bot"])
    extdiag && push!(hist_extra, "resid_A")
    return (theta=theta, t=t, steps=step, history=hist, history_extra=hist_extra,
            qt=qt, ss=ss,
            ux=ux, uz=uz, pv=pv, avg=avg, steady=steady,
            weak=weak, chi=chi, chi_prev=chi_prev,
            # Maxwell stress history (2b): `nothing` unless
            # `shear_modulus > 0`.  Advanced once per step and, with
            # `shear_modulus > 0`, the previous step's value drives the next
            # Stokes solve through the elastic force term.
            txx=txx, tyy=tyy, txy=txy)
end

