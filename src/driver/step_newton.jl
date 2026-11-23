# Globalization of the *per-time-step* nonlinear solve of the quadtree driver.
#
# The driver's step is the fixed point of one Picard sweep
#
#   u(theta)  = Stokes(eta(T(theta), eps(u_lagged))),      (momentum + mass)
#   theta_new = A_E(u)^{-1} rhs(theta, u),                (BDF2 / backward Euler)
#
# with `A_E = (a/dt) M + K + C(u)` and `rhs = (b/dt) M theta + ... + F(u)` (see
# `solve_semquad_energy`).  `pic_mode = :picard` iterates that map; it diverges
# for a strongly strain-rate dependent viscosity because the map's spectral
# radius exceeds one (measured on the composite law: Vrms 19.8 -> 23.2 -> 43.7
# over three iterations, docs/dev/PHYSICS.md).
#
# This file adds the alternative `pic_mode = :newton`: a damped Newton-Krylov
# solve of the *coupled* per-step residual
#
#     r_S = Cu' (fv - K u - Bm' p)              momentum  (eta = eta(T, eps(u)))
#     r_C = gr - Bs uhat                        mass
#     r_E = Cfree' (A_E(u) theta - rhs(theta,u)) energy
#
# with the analytic Jacobian
#
#     d r_S / du = -(Kred + Kgeo)               Kgeo from deta/deps (G156 wiring)
#     d r_S / dth = Cu' (Ra Mv dth - K(deta_T) u)
#     d r_E / du  = Cfree' (C(du) theta - F(du))
#     d r_E / dth = Cfree' (A_E - (b/dt) M) Cfree
#
# i.e. the same blocks as the steady coupled Newton (`semquad_coupled.jl`) but
# with the time-step energy operator and with the strain-rate tangent Kgeo that
# a temperature-only law does not need.  The globalization is a backtracking
# line search on the block-scaled residual that halves on every increase, and a
# Status: research option, not the production path.  The production configuration
# is Picard with a matrix-free multigrid Stokes solve (`pic_mode = :picard`,
# `linear = :auto`), and no shipped card sets `pic_mode = :newton`.  A Newton
# iteration assembles a Jacobian and factorises its saddle directly, so the driver
# refuses `linear = :mg` and also refuses a reference state (TALA/ALA), the
# compressible momentum term and `nu_cbf` -- the analytic Jacobian does not
# linearise them.  Measured: 18 Picard against 7 Newton iterations over three
# steps at Ra = 1e6, but 1.87x slower per iteration at 64 leaves and up to 12x at
# L5, because each iteration pays a direct factorisation instead of a multigrid
# cycle (docs/dev/SOLVER.md sections 3.2 and 4.6).  A failed Newton falls back to
# Picard and `pic_diag` records it.
# Picard fallback is taken by the *driver* when this function reports failure.
#
# The strain-rate dependence is supplied by the caller through `make_eta`, so
# this file contains no rheology: `make_eta(theta, ufull)` returns
# `(eta_fun, deta_T_fun, deta_eps_fun)` evaluated at the given state
# (`deta_eps_fun = nothing` for a law that does not depend on the strain rate).

"""
    _semquad_step_residual_ctx(ss, qt, pr, theta_nm1, theta_n, theta, ufull,
                               pfull,
                               make_eta, leaves; dt, dt_prev, firststep,
                               advection, ev, ev_cmax, ev_alpha, ev_timestep,
                               geo_spd_safety, phases, latent_scheme)

Assemble the Stokes system and the energy operator of one time step at the state
`(theta, ufull, pfull)` and return the residual `(r_S, r_C, r_E)` together with
everything the Jacobian-vector product and the preconditioner need.  Internal
helper of [`semquad_step_newton`](@ref).

`phases` adds the phase-transition buoyancy to the momentum residual and the
latent heat (apparent heat capacity) to the energy operator.  With
`latent_scheme = :implicit` (default) the capacity/source are evaluated at the
current iterate `theta` (an extra nonlinearity the Jacobian linearises);
`:lagged` freezes them at the start-of-step `theta_n`, so they are linear in the
iterate.  `:lagged` is first order in the latent coupling and is meant as a
diagnostic: the measured orders and the benchmark recommendation are in
docs/dev/PHYSICS.md §13.
"""
function _semquad_step_residual_ctx(ss::SemStokes, qt::Quadtree, pr::Params,
                                    theta_nm1::Vector{Float64},
                                    theta_n::Vector{Float64},
                                    theta::Vector{Float64},
                                    ufull::Vector{Float64},
                                    pfull::Vector{Float64},
                                    make_eta::Function,
                                    leaves::Set{NTuple{3,Int}};
                                    dt::Float64, dt_prev::Float64,
                                    firststep::Bool,
                                    advection::Symbol=:galerkin,
                                    ev::Float64=0.0, ev_cmax::Float64=0.052,
                                    ev_alpha::Int=2, ev_timestep::Int=0,
                                    geo_spd_safety::Float64=0.0,
                                    phases::Vector{PhaseTransition}=PhaseTransition[],
                                    latent_scheme::Symbol=:implicit,
                                    chi::Union{Nothing,Vector{Float64}}=nothing,
                                    chi_n::Union{Nothing,Vector{Float64}}=nothing,
                                    chi_nm1::Union{Nothing,Vector{Float64}}=nothing,
                                    elastic_stress::Union{Nothing,NTuple{3,Vector{Float64}}}=nothing)
    latent_scheme in (:implicit, :lagged) ||
        throw(ArgumentError("latent_scheme must be :implicit or :lagged"))
    sq = ss.su
    N = sq.N
    z2 = (x, y) -> 0.0
    # `make_eta(theta, ufull, pfull[, chi])` returns `(eta_fun, deta_T_fun,
    # deta_eps_fun)` and optionally the Drucker-Prager pressure derivative as a
    # fourth entry `deta_dp_fun(x, y, edot, p)` (docs/dev/PHYSICS.md §8.4) and the
    # weakening derivative as a fifth `deta_dchi_fun(x, y, edot)` (§9.4).  A
    # `make_eta` that does not accept the strain-weakening field is called with
    # three arguments, so the historical contract still works.
    mk = applicable(make_eta, theta, ufull, pfull, chi) ?
         make_eta(theta, ufull, pfull, chi) : make_eta(theta, ufull, pfull)
    length(mk) >= 3 ||
        throw(ArgumentError("make_eta must return at least " *
                            "(eta_fun, deta_T_fun, deta_eps_fun)"))
    eta_fun = mk[1]; deta_T_fun = mk[2]; deta_eps_fun = mk[3]
    dp_fun = length(mk) >= 4 ? mk[4] : nothing
    dchi_fun = length(mk) >= 5 ? mk[5] : nothing
    # Buoyancy load with the density hook (G248): `theta_buoyancy_load` returns
    # the historical `(x, y) -> Ra * theta` closure when
    # `Params.rho_multiplier === nothing` (bit for bit) and multiplies the
    # density when it is set.
    th_at = (x, y) -> QuaST._semquad_eval_at(sq, theta, x, y, leaves)
    fz = if isempty(phases)
        theta_buoyancy_load(pr, th_at)
    else
        # θ-buoyancy plus the phase anomaly, exactly the driver's `fz`
        # (`semquad_phase_buoyancy` subtracts the base-state fraction).
        m = pr.rho_multiplier
        m === nothing ?
            ((x, y) -> begin
                 th = th_at(x, y)
                 pr.Ra * th + semquad_phase_buoyancy(pr, phases, y, th)
             end) :
            ((x, y) -> begin
                 th = th_at(x, y)
                 pr.Ra * m(x, y) * th +
                     semquad_phase_buoyancy(pr, phases, y, th)
             end)
    end
    sys = dp_fun === nothing ?
          solve_semquad_stokes_var(ss, qt, eta_fun, z2, fz, z2, z2;
                                   bc=pr.bc, linear=:direct,
                                   deviatoric=pr.deviatoric,
                                   elastic_stress=elastic_stress,
                                   assemble_only=true,
                                   geo_u=deta_eps_fun === nothing ? nothing : ufull,
                                   geo_coef=deta_eps_fun,
                                   geo_spd_safety=geo_spd_safety) :
          solve_semquad_stokes_var(ss, qt, eta_fun, z2, fz, z2, z2;
                                   bc=pr.bc, linear=:direct,
                                   deviatoric=pr.deviatoric,
                                   elastic_stress=elastic_stress,
                                   assemble_only=true,
                                   geo_u=deta_eps_fun === nothing ? nothing : ufull,
                                   geo_coef=deta_eps_fun,
                                   geo_spd_safety=geo_spd_safety,
                                   dp_u=ufull, dp_pfield=pfull, dp_coef=dp_fun)
    latent_implicit = semquad_latent_active(pr, phases) &&
                      latent_scheme === :implicit
    parts = solve_semquad_energy(sq, qt, pr, theta_n, theta_nm1,
                                 ufull[1:N], ufull[(N + 1):(2N)], dt, dt_prev,
                                 firststep; advection=advection, ev=ev,
                                 ev_cmax=ev_cmax, ev_alpha=ev_alpha,
                                 ev_timestep=ev_timestep, eta_fun=eta_fun,
                                 phases=phases,
                                 latent_theta=latent_implicit ? theta : nothing,
                                 return_parts=true)
    nf = size(sys.Cu, 2)
    nfp = size(sys.Cp, 2)
    nt = size(parts.Cfree, 2)
    uhat = sys.Cu' * (ufull - sys.u0)
    phat = sys.Cp' * (pfull - sys.p0)
    # `sys.Kred` carries the geometric tangent when `geo_u` was given; the
    # residual must use the true stiffness, so rebuild the unreduced product.
    Kred_true = sys.Kgeo === nothing ? sys.Kred : sys.Cu' * sys.K * sys.Cu
    rS = sys.fr - Kred_true * uhat - sys.Bt * phat
    rC = sys.gr - sys.Bs * uhat
    rE = parts.Cfree' * (parts.A * theta - parts.rhs)
    # `parts.Ar = Cfree' A_E Cfree` is exactly the `d r_E / d theta` block (the
    # mass term of `rhs` belongs to the start-of-step level, so it does not
    # depend on the iterate), and it also defines the Picard-equivalent change.
    FAr = lu(parts.Ar)
    # ---- strain weakening (docs/dev/PHYSICS.md §9) ---------------------------
    # `chi` is the current iterate of the weakening field, `chi_n`/`chi_nm1`
    # the two BDF2 levels (frozen during the step).  With `chi === nothing`
    # every line below is skipped and the system is exactly the 3-block one.
    weak = chi !== nothing
    partsW = nothing
    rW = nothing
    FArW = nothing
    Kchi_red = nothing
    wW = 0.0
    if weak
        (chi_n === nothing || chi_nm1 === nothing) &&
            throw(ArgumentError("the weakening residual needs the two " *
                                "start-of-step levels chi_n and chi_nm1"))
        partsW = solve_semquad_weakening(sq, ss.sp, qt, pr, chi_n, chi_nm1,
                                         theta, ufull[1:N],
                                         ufull[(N + 1):(2N)], pfull, dt,
                                         dt_prev, firststep;
                                         advection=advection, chi_src=chi,
                                         return_parts=true)
        rW = partsW.Cfree' * (partsW.A * chi - partsW.rhs)
        FArW = lu(partsW.Ar)
        wW = max(norm(partsW.Cfree' * partsW.rhs), 1.0e-30)
        if dchi_fun !== nothing
            Kchi = assemble_semquad_deta_chi(sq, qt, dchi_fun, ufull)
            Kchi_red = sys.Cu' * Kchi * partsW.Cfree
        end
    end
    return (sys=sys, parts=parts, eta_fun=eta_fun, deta_T_fun=deta_T_fun,
            deta_eps_fun=deta_eps_fun, dp_fun=dp_fun, dchi_fun=dchi_fun,
            nf=nf, nfp=nfp, nt=nt, FAr=FAr,
            Kred_true=Kred_true, rS=rS, rC=rC, rE=rE, qt=qt,
            phases=phases, theta_n=theta_n, theta_nm1=theta_nm1,
            latent_implicit=latent_implicit,
            weak=weak, chi=chi, partsW=partsW, rW=rW, FArW=FArW,
            Kchi_red=Kchi_red, wW=wW,
            nw=weak ? size(partsW.Cfree, 2) : 0)
end

"""
    _semquad_step_change(ctx, theta)

The driver's own convergence measure for one step,
`||theta_new - theta|| / max(||theta_new||, 1e-30)` with
`theta_new = A_E^{-1}(rhs)` — identical to the quantity the Picard loop tests
against `pic_tol`, and evaluated from the stored residual so that the Newton and
Picard iteration counts are compared on the same scale.  With a weakening field
the change is the maximum of the temperature and `chi` updates and the residual
gains the `r_W` block.  Returns `(change, scaled_residual)`.
"""
function _semquad_step_change(ctx, theta::Vector{Float64}, wS::Float64,
                              wC::Float64, wE::Float64)
    rEfull = ctx.parts.A * theta - ctx.parts.rhs
    dth = -ctx.parts.Cfree * (ctx.FAr \ (ctx.parts.Cfree' * rEfull))
    change = norm(dth) / max(norm(theta + dth), 1.0e-30)
    res = sqrt((norm(ctx.rS) / max(wS, 1.0e-30))^2 +
               (norm(ctx.rC) / max(wC, 1.0e-30))^2 +
               (norm(ctx.rE) / max(wE, 1.0e-30))^2)
    if ctx.weak
        rWfull = ctx.partsW.A * ctx.chi - ctx.partsW.rhs
        dchi = -ctx.partsW.Cfree *
               (ctx.FArW \ (ctx.partsW.Cfree' * rWfull))
        change = max(change,
                     norm(dchi) / max(norm(ctx.chi + dchi), 1.0e-30))
        res = sqrt(res^2 + (norm(ctx.rW) / max(ctx.wW, 1.0e-30))^2)
    end
    return change, res
end

"""
    _semquad_step_saddle(Kred, Bt, Bs; b=nothing, Kp=nothing, rtol=1e-8,
                         verbose=false, it=0)

Direct LU of the reduced saddle `[Kred Bt; Bs 0]` (or `[Kred Bt+Kp; Bs 0]` when
the Drucker-Prager pressure-derivative block `Kp` is given) with an **adaptive**
relative diagonal shift: the shift is grown through `1e-12 ... 1e-1` until the
factorisation solves `b` (the momentum/continuity residual of the current
iterate) to `rtol` in the relative residual.  Without the check a near-singular
`Kred + Kgeo` gives a factorisation with huge entries that is accepted silently
and destroys the Newton direction (measured: the Krylov iteration then stalls at
a relative residual of 0.1-0.9).  Returns a `solve(b)` closure.
"""
function _semquad_step_saddle(Kred, Bt, Bs; b::Union{Nothing,Vector{Float64}}=nothing,
                              Kp=nothing, rtol::Float64=1.0e-8,
                              verbose::Bool=false,
                              it::Int=0)
    nf = size(Kred, 1)
    nfp = size(Bs, 1)
    Bt_eff = Kp === nothing ? Bt : Bt + Kp
    S = [Kred Bt_eff; Bs spzeros(Float64, nfp, nfp)]
    dmax = maximum(abs, diag(S)) + 1.0
    nb = b === nothing ? 0.0 : norm(b)
    best = nothing
    for rel in (1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1)
        Ss = S + (rel * dmax) * I
        Slu = try
            lu(Ss)
        catch err
            err isa SingularException || rethrow()
            nothing
        end
        Slu === nothing && continue
        best = (Slu, rel)
        if b === nothing || nb == 0.0
            return (solve = x -> Slu \ x, shift=rel)
        end
        xx = Slu \ b
        # residual of the *shifted* system: `S` alone would report the shift
        # itself as an error and send the loop to the largest shift
        rr = norm(Ss * xx - b) / nb
        if rr <= rtol
            verbose && rel > 1.0e-12 &&
                @printf("  step-newton %2d: saddle shift %.0e (relres %.1e)\n", it, rel, rr)
            return (solve = x -> Slu \ x, shift=rel)
        end
    end
    best === nothing &&
        error("per-step Newton: direct LU failed for every relative shift")
    verbose && @printf("  step-newton %2d: saddle shift %.0e is inaccurate\n", it, best[2])
    return (solve = x -> best[1] \ x, shift=best[2])
end

"""
    _semquad_step_phase_tangent_mass(sq, qt, pr, phases, theta)

Weighted mass matrix `int Ra (1 - sum_phase B dGamma/dT) phi_i phi_j` of the
phase-buoyancy derivative in `theta` (`semquad_phase_buoyancy`'s anomaly term
moves the transition with the temperature).  `nothing` for an empty phase list.
"""
function _semquad_step_phase_tangent_mass(sq::SemQuad, qt::Quadtree, pr::Params,
                                          phases::Vector{PhaseTransition},
                                          theta::Vector{Float64})
    isempty(phases) && return nothing
    leaves = Set(qt.leaves)
    wfun = (x, y) -> begin
        T = semquad_base_T(pr, y) +
            QuaST._semquad_eval_at(sq, theta, x, y, leaves)
        s = 1.0
        for pt in phases
            s -= pt.B * phase_dgamma_dT(pt, y, T)
        end
        pr.Ra * s
    end
    return assemble_semquad_pressure_mass(sq; weight=wfun)
end

"""
    semquad_step_jvp(ctx, theta, ufull, x, Mv; advection, ev, ...)

Analytic Jacobian-vector product of the per-step residual at the state that
produced `ctx` (`x = [xu; xp; xth]` in the reduced unknowns).  Exposed so that
the finite-difference check of the gate (`t_g193`) exercises exactly the matrix
the Newton uses.

The blocks are
`d r_S/d(u,p,theta) = (-Kred-Kgeo)*xu - Bt*xp + Cu'(Ra(1 - sum B Gamma_T) Mv dtheta - K(deta_T)u)`
(the weighted `Mv` includes the phase-buoyancy derivative when `phases` is
non-empty), `d r_C = -Bs*xu`,
`d r_E = Cfree'(C(du)theta - F(du)) + Cfree' A_E dtheta + Cfree'(latent tangent)`.

The latent tangent (`latent_implicit`) is the directional derivative of the
apparent-heat-capacity operator and the `-L Gamma_z u_z` source in the
temperature direction: see `assemble_semquad_energy_latent_jvp`.  `Mv_phase` may
be supplied by the caller (the Newton builds it once per iteration); otherwise
it is assembled here from `theta` and `ctx.phases`.
"""
function semquad_step_jvp(ctx, theta::Vector{Float64}, ufull::Vector{Float64},
                          x::Vector{Float64}, Mv; N::Int,
                          sq::SemQuad, q::Int, pr::Params,
                          advection::Symbol=:galerkin, ev::Float64=0.0,
                          ev_cmax::Float64=0.052, ev_alpha::Int=2,
                          ev_timestep::Int=0,
                          Kred::Union{Nothing,Any}=nothing,
                          Mv_phase::Union{Nothing,Any}=nothing,
                          latent_tangent::Bool=true)
    nf, nfp = ctx.nf, ctx.nfp
    Cu, Cp, Cfree = ctx.sys.Cu, ctx.sys.Cp, ctx.parts.Cfree
    Kred_jac = Kred === nothing ? ctx.sys.Kred : Kred
    # Drucker-Prager: the momentum residual depends on p through the yield
    # stress, so `d r_S/dp = -(Bt + Kp_red)`.  Without a plastic branch
    # `Kp_red === nothing` and the expression is the historical `Bt`.
    Bt_jac = ctx.sys.Kp_red === nothing ? ctx.sys.Bt :
             ctx.sys.Bt + ctx.sys.Kp_red
    xu = x[1:nf]
    xp = x[(nf + 1):(nf + nfp)]
    xth = x[(nf + nfp + 1):(nf + nfp + ctx.nt)]
    dth = Cfree * xth
    du = Cu * xu
    if isempty(ctx.phases)
        dfull = vcat(zeros(Float64, N), pr.Ra * (Mv * dth))
    else
        # d(f_z)/d(theta) = Ra (1 - sum B Gamma_T): the phase anomaly moves the
        # transition with T.  This is the *weighted* mass matrix of the same
        # `Mv` the constant-buoyancy tangent uses.
        wph = Mv_phase === nothing ?
              _semquad_step_phase_tangent_mass(sq, ctx.qt, pr, ctx.phases,
                                               theta) : Mv_phase
        dfull = vcat(zeros(Float64, N), wph * dth)
    end
    dKu = _semquad_jvp_ku(sq, q, pr, theta, dth, ufull;
                          deta_T_fun=ctx.deta_T_fun)
    yS = Cu' * (dfull - dKu) - Kred_jac * xu - Bt_jac * xp
    yC = -ctx.sys.Bs * xu
    _, _, Cd, Fdu = assemble_semquad_energy(
        sq, ctx.qt, pr, du[1:N], du[(N + 1):(2N)];
        advection=advection, theta=theta, ev=ev, ev_cmax=ev_cmax,
        ev_alpha=ev_alpha, ev_timestep=ev_timestep,
        ev_field=ctx.parts.ev_field, phases=ctx.phases)
    yE = Cfree' * (Cd * theta - Fdu) + Cfree' * (ctx.parts.A * dth)
    if ctx.latent_implicit && latent_tangent
        Md, Cdl, Fdl = assemble_semquad_energy_latent_jvp(
            sq, ctx.qt, pr, ctx.phases, dth, theta, ufull[1:N],
            ufull[(N + 1):(2N)]; advection=advection)
        ca, cb, cc = ctx.parts.ca, ctx.parts.cb, ctx.parts.cc
        yE = yE .+ Cfree' * (ca * (Md * theta) -
                             cb * (Md * ctx.theta_n) +
                             cc * (Md * ctx.theta_nm1) +
                             Cdl * theta - Fdl)
    end
    if ctx.weak
        # Strain weakening (docs/dev/PHYSICS.md §9.4): the chi block is appended
        # after [u; p; theta].  `d r_S/d chi = -Cu' Kchi Cfree_W` is exact;
        # `d r_W/d chi = Cfree_W' A chi` is exact; `d r_W/du` is the advection
        # of chi by the velocity perturbation (the yielding indicator of the
        # source is lagged, which is the only inexact term).
        nw = ctx.nw
        xw = x[(nf + nfp + ctx.nt + 1):end]
        dchi = ctx.partsW.Cfree * xw
        ctx.Kchi_red === nothing || (yS = yS .- ctx.Kchi_red * xw)
        yW = ctx.partsW.Cfree' * (ctx.partsW.A * dchi) +
             semquad_weakening_jvp(ctx.partsW, sq, ctx.qt, pr, du[1:N],
                                   du[(N + 1):(2N)], ctx.chi;
                                   advection=advection)
        return vcat(yS, yC, yE, yW)
    end
    return vcat(yS, yC, yE)
end

"""
    semquad_step_residual(ctx, theta)

The full per-step residual `[rS; rC; rE]` of the state that produced `ctx`
(pure function of the state, exposed for the finite-difference check).
"""
semquad_step_residual(ctx, theta::Vector{Float64}) = vcat(ctx.rS, ctx.rC, ctx.rE)

"""
    semquad_step_newton(ss, qt, pr, theta_nm1, theta_n, ux, uz, pv, make_eta; ...)

Damped Newton-Krylov solve of the nonlinear system of **one time step** of the
the quadtree driver (not a steady-state solve): the root of `(r_S, r_C, r_E)` above,
i.e. exactly the fixed point that the `maxpic` Picard loop converges to.
`theta_nm1` is `theta^(n-1)` (frozen) and `theta_n` is the **start-of-step**
temperature `theta^n`, which is also the initial iterate (the driver calls this
right after the first Picard sweep).

`make_eta(theta, ufull, pfull[, chi])` must return
`(eta_fun, deta_T_fun, deta_eps_fun)`; `eta_fun(x,y)` is the viscosity,
`deta_T_fun(x,y)` its temperature derivative and `deta_eps_fun(x,y,edot)` its
strain-rate derivative (`nothing` when the law has none) at the state it was
built from.  With a Drucker-Prager yield the tuple may carry the pressure and
weakening derivatives as 4th/5th entries (docs/dev/PHYSICS.md §8.4, §9.4).

With `chi !== nothing` the strain-weakening field is a fourth unknown block
(`chi_n`/`chi_nm1` are its two start-of-step BDF2 levels) and the residual is
`(r_S, r_C, r_E, r_W)`; the returned named tuple then also carries `chi`.

The iteration is Newton-GMRES with a block-triangular preconditioner (the
`Kred + Kgeo` saddle plus the energy operator), a backtracking line search that
halves `alpha` on every increase of the block-scaled residual (up to
`ls_halvings` times), and it stops when BOTH the Picard-equivalent `change`
(identical to the driver's `pic_tol` measure) and the scaled residual are below
`tol`.  Returns `(theta, ux, uz, pv, converged, iters, history, change, res,
alpha, ev_field)`; the driver falls back to Picard when `converged == false`.

Two loop guards (docs/dev/BUG_LEDGER.md §S16): `krylov_stagnation` stops a GMRES
attempt whose restart cycle bought less than that factor (the direction is then
rejected exactly as a `rel > 1e-4` failure, and on the healthy case the guard
never fires, so the iterates are unchanged), and `gmres_restarted` now honours
`maxit` inside the cycle instead of overshooting it by up to `restart - 1`.

Only the direct saddle solve is supported (`linear=:direct`); the Newton is a
per-step solver on small/medium trees, and the direct factorisation is what
makes one Jacobian assembly per iteration affordable.
"""
function semquad_step_newton(ss::SemStokes, qt::Quadtree, pr::Params,
                             theta_nm1::Vector{Float64},
                             theta_n::Vector{Float64},
                             ux::Vector{Float64}, uz::Vector{Float64},
                             pv::Vector{Float64},
                             make_eta::Function;
                             dt::Float64, dt_prev::Float64,
                             firststep::Bool=false,
                             advection::Symbol=:galerkin,
                             ev::Float64=0.0, ev_cmax::Float64=0.052,
                             ev_alpha::Int=2, ev_timestep::Int=0,
                             tol::Float64=1.0e-6, maxit::Int=15,
                             krylov::Int=80, restart::Int=30,
                             krylov_stagnation::Float64=1.2,
                             ls_halvings::Int=8,
                             geo_spd_safety::Float64=0.0,
                             phases::Vector{PhaseTransition}=PhaseTransition[],
                             latent_scheme::Symbol=:implicit,
                             chi::Union{Nothing,Vector{Float64}}=nothing,
                             chi_n::Union{Nothing,Vector{Float64}}=nothing,
                             chi_nm1::Union{Nothing,Vector{Float64}}=nothing,
                             elastic_stress::Union{Nothing,NTuple{3,Vector{Float64}}}=nothing,
                             verbose::Bool=false)
    sq = ss.su
    N = sq.N
    leaves = Set(qt.leaves)
    Mv = buoyancy_mass_matrix(sq, pr)
    theta = copy(theta_n)
    ufull = vcat(copy(ux), copy(uz))
    pfull = copy(pv)
    z2 = (x, y) -> 0.0
    # Strain weakening: `chi` is the initial iterate, `chi_n`/`chi_nm1` the two
    # frozen BDF2 levels.  When `chi === nothing` the solve is the historical
    # (theta, u, p) system bit for bit.  The iterate is not clamped here (the
    # driver projects the stored field); `semquad_weak_factor` clamps its
    # argument, so the momentum never sees an out-of-range factor.
    chi_v = chi === nothing ? nothing : copy(chi)
    chi_n === nothing && (chi_n = chi_v)
    chi_nm1 === nothing && (chi_nm1 = chi_v)
    ctx(th, uf, pf, cv) = _semquad_step_residual_ctx(
        ss, qt, pr, theta_nm1, theta_n, th, uf, pf, make_eta, leaves; dt=dt,
        dt_prev=dt_prev, firststep=firststep, advection=advection, ev=ev,
        ev_cmax=ev_cmax, ev_alpha=ev_alpha, ev_timestep=ev_timestep,
        geo_spd_safety=geo_spd_safety, phases=phases,
        latent_scheme=latent_scheme, chi=cv, chi_n=chi_n, chi_nm1=chi_nm1,
        elastic_stress=elastic_stress)
    c = ctx(theta, ufull, pfull, chi_v)
    wS = max(norm(c.sys.fr), 1.0e-30)
    wC = max(norm(c.sys.Bs) * norm(c.sys.Cu' * ufull), 1.0e-30)
    wE = max(norm(c.parts.Cfree' * c.parts.rhs), 1.0e-30)
    chg, res = _semquad_step_change(c, theta, wS, wC, wE)
    hist = Float64[chg]
    converged = (chg <= tol) && (res <= tol)
    verbose && @printf("  step-newton  0: change=%.3e res=%.3e (rS=%.2e rC=%.2e rE=%.2e)\n", chg, res, norm(c.rS) / wS, norm(c.rC) / wC, norm(c.rE) / wE)
    it = 0
    alpha_last = 1.0
    while !converged && it < maxit
        it += 1
        nf, nfp, nt = c.nf, c.nfp, c.nt
        nw = c.nw
        Cu, Cp, Cfree = c.sys.Cu, c.sys.Cp, c.parts.Cfree
        Bt = c.sys.Bt
        Bs = c.sys.Bs
        bS0 = vcat(c.rS, c.rC)
        # weakening row of a velocity direction: the advection of chi by `du`
        weak_coupling(du) = c.weak ?
            semquad_weakening_jvp(c.partsW, sq, qt, pr, du[1:N],
                                  du[(N + 1):(2N)], c.chi;
                                  advection=advection) :
            zeros(Float64, nw)
        # energy row of a velocity direction: one energy assembly, no Stokes solve
        energy_coupling(du) = begin
            _, _, Cd, Fdu = assemble_semquad_energy(
                sq, qt, pr, du[1:N], du[(N + 1):(2N)];
                advection=advection, theta=theta, ev=ev, ev_cmax=ev_cmax,
                ev_alpha=ev_alpha, ev_timestep=ev_timestep,
                ev_field=c.parts.ev_field, phases=phases)
            Cfree' * (Cd * theta - Fdu)
        end
        # The phase-buoyancy tangent is independent of the Kgeo continuation, so
        # build it once per Newton iteration instead of once per Krylov vector.
        Mv_phase = _semquad_step_phase_tangent_mass(sq, qt, pr, phases, theta)
        rhsN = c.weak ? -vcat(c.rS, c.rC, c.rE, c.rW) : -vcat(c.rS, c.rC, c.rE)
        # ---- Kgeo continuation + backtracking line search -------------------
        # The exact strain-rate tangent `Kgeo` can make the velocity block
        # `Kred + Kgeo` indefinite/near singular for a strongly weakening law
        # (the geometric term partially cancels `2 eta`), and the Newton
        # direction then stops being a descent direction.  The scaling `s` of
        # `Kgeo` is therefore a *damping parameter*: try the exact tangent
        # first, and if its linear solve fails or its line search rejects every
        # step, halve it (s -> 0 is the Picard-type tangent of the coupled
        # system, which always gives a descent direction here).
        accepted = false
        th_t = theta; uf_t = ufull; pf_t = pfull; cw_t = chi_v
        c_t = c; chg_t = chg; res_t = res
        alpha = 1.0
        rel = NaN
        nit = 0
        s_last = 0.0
        for s in (1.0, 0.5, 0.25, 0.1, 0.0)
            Kred_s = s == 1.0 ? c.sys.Kred :
                     c.sys.Cu' * (c.sys.K + s * c.sys.Kgeo) * c.sys.Cu
            solver = _semquad_step_saddle(Kred_s, Bt, Bs; b=bS0,
                                          Kp=c.sys.Kp_red,
                                          verbose=verbose, it=it)
            jac = x -> semquad_step_jvp(c, theta, ufull, x, Mv; N=N, sq=sq,
                                        q=ss.q, pr=pr, advection=advection,
                                        ev=ev, ev_cmax=ev_cmax,
                                        ev_alpha=ev_alpha,
                                        ev_timestep=ev_timestep, Kred=Kred_s,
                                        Mv_phase=Mv_phase)
            precond = b -> begin
                z = zeros(Float64, nf + nfp + nt + nw)
                sol = solver.solve(vcat(-b[1:nf], -b[(nf + 1):(nf + nfp)]))
                z[1:nf] = sol[1:nf]
                z[(nf + 1):(nf + nfp)] = sol[(nf + 1):end]
                rhsE = b[(nf + nfp + 1):(nf + nfp + nt)] .-
                       energy_coupling(Cu * z[1:nf])
                z[(nf + nfp + 1):(nf + nfp + nt)] = c.FAr \ rhsE
                if c.weak
                    rhsW = b[(nf + nfp + nt + 1):end] .-
                           weak_coupling(Cu * z[1:nf])
                    z[(nf + nfp + nt + 1):end] = c.FArW \ rhsW
                end
                return z
            end
            dx, rel, nit = QuaST.gmres_restarted(jac, rhsN; precond=precond,
                                                     tol=1.0e-8, maxit=krylov,
                                                     restart=restart,
                                                     stagnation=krylov_stagnation)
            rel > 1.0e-4 && (verbose && @printf("  step-newton %2d: kgeo=%.2g linear rel=%.1e (%d krylov vectors) -> damp\n", it, s, rel, nit); continue)
            alpha = 1.0
            for _ in 1:ls_halvings
                th1 = theta .+ alpha .* (Cfree * dx[(nf + nfp + 1):(nf + nfp + nt)])
                uf1 = ufull .+ alpha .* (Cu * dx[1:nf])
                pf1 = pfull .+ alpha .* (Cp * dx[(nf + 1):(nf + nfp)])
                cw1 = c.weak ?
                      chi_v .+ alpha .* (c.partsW.Cfree *
                                         dx[(nf + nfp + nt + 1):end]) :
                      nothing
                c1 = ctx(th1, uf1, pf1, cw1)
                chg1, res1 = _semquad_step_change(c1, th1, wS, wC, wE)
                if res1 < (1.0 - 1.0e-4 * alpha) * res
                    th_t, uf_t, pf_t = th1, uf1, pf1
                    cw_t = cw1
                    c_t, chg_t, res_t = c1, chg1, res1
                    accepted = true
                    s_last = s
                    break
                end
                alpha *= 0.5
            end
            accepted && break
        end
        if !accepted
            verbose && @printf("  step-newton %2d: line search failed (res=%.3e, gmres rel=%.1e)\n", it, res, rel)
            break
        end
        theta, ufull, pfull, chi_v = th_t, uf_t, pf_t, cw_t
        c, chg, res = c_t, chg_t, res_t
        alpha_last = alpha
        push!(hist, chg)
        converged = (chg <= tol) && (res <= tol)
        verbose && @printf("  step-newton %2d: change=%.3e res=%.3e (rS=%.2e rC=%.2e rE=%.2e) alpha=%.2g kgeo=%.2g gmres=%d rel=%.1e\n", it, chg, res, norm(c.rS) / wS, norm(c.rC) / wC, norm(c.rE) / wE, alpha, s_last, nit, rel)
    end
    return (theta=theta, ux=ufull[1:N], uz=ufull[(N + 1):(2N)], pv=pfull,
            chi=chi_v,
            converged=converged, iters=it, history=hist, change=chg, res=res,
            alpha=alpha_last, ev_field=c.parts.ev_field)
end
