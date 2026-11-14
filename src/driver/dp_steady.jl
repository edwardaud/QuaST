# Steady Newton for the Drucker-Prager visco-plastic rheology on this solver.
#
# Unlike the Tosi (2015) law, the DP viscosity depends on the *dynamic pressure*
# through the yield stress:
#
#     tau_y = min(c cos(phi) + p sin(phi), tau_max)
#     eta_pl = tau_y / (2 edot),      eta = clip(min(eta_visc(T), eta_pl))
#
# so the steady map is a function of (theta, u, p): at frozen theta the inner
# problem couples u and p (both come out of the Stokes solve, and p feeds back
# into eta).  `semquad_dp_inner` solves that by Picard iteration; the
# linearisation then has three derivative terms
#
#     delta-eta = (deta/dT) dtheta + (deta/dedot) d eps(u) + (deta/dp) dp
#     delta-K u = K(deta_T) u + Kgeo du + Kp dp
#
# where `Kgeo` is the strain-rate operator (as in G156, with the DP plastic
# derivative `-eta/edot`) and `Kp` is the non-symmetric velocity-row <->
# pressure-column block of G163.  The saddle of the Newton step is therefore
#
#     [ Kred + Kgeo    Bt + Kp_red ] [ du ]   [ ... ]
#     [ Bs             0          ] [ dp ] = [  0  ]
#
# which is non-symmetric; the direct LU handles it (G163).  In the fully
# yielded limit the plastic tangent is singular (`d(2 eta edot)/d edot = 0`), so
# `dp_kgeo=false` (freeze the plastic branch) is offered as a robust variant.

"""
    semquad_dp_stokes_newton(sq, qt, ss, pr, theta; u_init, p_init, tol, maxit,
                             spd_safety, linear, verbose)

Newton solve of the Drucker-Prager Stokes problem at frozen `theta`.

The unknowns are the reduced velocity and pressure `(uhat, phat)`; the residual
is the *true* one

    r_S = fr - Kred(theta,u,p) uhat - Bt phat,   r_C = gr - Bs uhat

(assembled with the actual viscosity, not with the Jacobian operator), and the
Jacobian is the DP saddle

    S = [ Kred + Kgeo   Bt + Kp_red ;  Bs  0 ]

with `Kgeo` from `deta/dedot` and `Kp` from `deta/dp`.  `Kgeo` is scaled by
ASPECT's SPD factor (`spd_safety`, default 0.9), which is what keeps the
velocity block well posed in the fully yielded limit where the raw plastic
tangent `-eta/edot` would cancel the viscous modulus exactly.  The step solves
`S dx = r` (the residual Jacobian is `-S`) and is damped by a line search on the
block-scaled residual norm.  This is the "inner (u,p) iteration" of the DP map
Newton, done as a Newton instead of a Picard loop (the Picard loop does not
converge for yield-stress feedback; see docs/dev/SOLVER.md §1.11).

Returns `(u, p, converged, iters, history, resid)`.
"""
function semquad_dp_stokes_newton(sq::SemQuad, qt::Quadtree, ss::SemStokes,
                                  pr::Params, theta::Vector{Float64};
                                  u_init::Union{Nothing,Vector{Float64}}=nothing,
                                  p_init::Union{Nothing,Vector{Float64}}=nothing,
                                  tol::Float64=1.0e-8, maxit::Int=20,
                                  spd_safety::Float64=0.0,
                                  picard_warmup::Int=2,
                                  linear::Symbol=:direct,
                                  ptc::Bool=false, ptc_delta0::Float64=0.0,
                                  ptc_eta::Float64=1.0e-2,
                                  ptc_delta_max::Float64=1.0e8,
                                  ptc_growth::Float64=10.0,
                                  verbose::Bool=false)
    N = sq.N
    np = ss.sp.N
    leaves = Set(qt.leaves)
    pleaves = Set(ss.sp.leaves)
    z2 = (x, y) -> 0.0
    evalth(th, x, y) = QuaST._semquad_eval_at(sq, th, x, y, leaves)
    # --- start state: the isoviscous (viscous-branch) Stokes solution ---------
    u = zeros(Float64, 2N)
    p = zeros(Float64, np)
    if u_init !== nothing
        u .= u_init
    else
        eta_v = (x, y) -> eta_at(Tbar(y, pr.H) + evalth(theta, x, y), y, pr)[1]
        fz = (x, y) -> pr.Ra * evalth(theta, x, y)
        u, p, _, _, _ = solve_semquad_stokes_var(ss, qt, eta_v, z2, fz, z2, z2;
                                                 bc=pr.bc, linear=:direct,
                                                 deviatoric=pr.deviatoric)
    end
    p_init !== nothing && (p .= p_init)
    # --- residual / Jacobian at a state --------------------------------------
    function stokes_eta(u, p)
        (x, y) -> begin
            T = Tbar(y, pr.H) + evalth(theta, x, y)
            ed = semquad_strain_rate_at(sq, u[1:N], u[(N + 1):(2N)], x, y,
                                        leaves)
            pq = QuaST._semquad_eval_at(ss.sp, p, x, y, pleaves)
            semquad_visco_plastic_eta(T, y, ed, pq, pr)
        end
    end
    function residual(u, p)
        eta_fun = stokes_eta(u, p)
        fz_r = (x, y) -> pr.Ra * evalth(theta, x, y)
        sys = solve_semquad_stokes_var(ss, qt, eta_fun, z2, fz_r, z2, z2;
                                       bc=pr.bc, linear=:direct,
                                       deviatoric=pr.deviatoric,
                                       assemble_only=true)
        uh = sys.Cu' * (u - sys.u0)
        ph = sys.Cp' * (p - sys.p0)
        rS = sys.fr - sys.Kred * uh - sys.Bt * ph
        rC = sys.gr - sys.Bs * uh
        return vcat(rS, rC), sys
    end
    function jacobian(u, p)
        eta_fun = stokes_eta(u, p)
        deta_eps = (x, y, ed) -> begin
            T = Tbar(y, pr.H) + evalth(theta, x, y)
            pq = QuaST._semquad_eval_at(ss.sp, p, x, y, pleaves)
            semquad_deta_dp_eps(T, y, ed, pq, pr)
        end
        deta_pc = (x, y, ed, pq) -> begin
            T = Tbar(y, pr.H) + evalth(theta, x, y)
            semquad_deta_dp(T, y, ed, pq, pr)
        end
        sysJ = solve_semquad_stokes_var(ss, qt, eta_fun, z2, z2, z2, z2;
                                        bc=pr.bc, linear=:direct,
                                        deviatoric=pr.deviatoric,
                                        assemble_only=true, geo_u=u,
                                        geo_coef=deta_eps,
                                        geo_spd_safety=spd_safety, dp_u=u,
                                        dp_pfield=p, dp_coef=deta_pc)
        if ptc_on
            # pseudo-transient continuation: [K + M/delta, Bt; Bs 0] with the
            # *velocity* mass matrix only (a pseudo-time term on the pressure
            # would destroy exact incompressibility)
            sysJ = merge(sysJ, (Kred=sysJ.Kred .+ (1.0 / ptc_delta) .* Mred_ptc,))
        end
        solver = _semquad_newton_saddle(ss, qt, ss.p, pr.bc, pr.deviatoric,
                                        eta_fun, sysJ; linear=linear)
        return solver, sysJ
    end
    # --- Picard warm-up (ASPECT's "pre-Newton nonlinear iterations") ---------
    # The viscous start can be orders of magnitude away from the yielded
    # solution; a few lagged Picard sweeps bring the state into the Newton
    # basin before the SPD-stabilised Newton takes over.
    fz = (x, y) -> pr.Ra * evalth(theta, x, y)
    for _ in 1:picard_warmup
        eta_fun = stokes_eta(u, p)
        u, p, _, _, _ = solve_semquad_stokes_var(ss, qt, eta_fun, z2, fz, z2,
                                                 z2; bc=pr.bc, linear=:direct,
                                                 deviatoric=pr.deviatoric)
    end
    # block scales: relative to the initial residual of each block, with a floor
    # tied to the physical magnitudes so a block that starts at zero (e.g. the
    # mass residual of a Stokes-consistent start) does not blow the norm up later
    r0, sys0 = residual(u, p)
    nf = size(sys0.Kred, 1)
    nfp = size(sys0.Bs, 1)
    wS = max(norm(r0[1:nf]), 1.0e-8 * max(norm(sys0.fr), 1e-30))
    wC = max(norm(r0[(nf + 1):end]),
             1.0e-8 * max(norm(sys0.Bs) * norm(sys0.Cu' * u), 1e-30))
    scaled(r) = sqrt((norm(r[1:nf]) / wS)^2 + (norm(r[(nf + 1):end]) / wC)^2)
    # --- pseudo-transient continuation (PTC) --------------------------------
    # Standard residual-free form (Mulder & van Leer 1985; Kelley & Keyes 1998;
    # Coffey, Kelley & Keyes 2003): regularise the step by the velocity mass
    # matrix, [J + (1/delta) M_u] dx = r, and update delta with the
    # switched-evolution-relaxation rule delta <- delta*|r_k|/|r_{k+1}| (capped).
    # Once the scaled residual has dropped by `ptc_eta`, PTC is switched off and
    # the iteration is pure Newton.
    ptc_on = ptc
    ptc_delta = ptc_delta0
    if ptc && ptc_delta <= 0.0
        ptc_delta = 1.0 / max(scaled(r0), 1.0e-8)
    end
    Mred_ptc = ptc ? (sys0.Cu' * assemble_semquad_velocity_mass(ss.su) *
                      sys0.Cu) : nothing
    ptc_r0 = scaled(r0)
    r = r0
    hist = Float64[scaled(r)]
    converged = scaled(r) <= tol
    it = 0
    while !converged && it < maxit
        it += 1
        nr = scaled(r)
        solver, _ = jacobian(u, p)
        dx = solver.solve(r)
        alpha = 1.0
        accepted = false
        for _ in 1:12
            ut = u .+ alpha .* (sys0.Cu * dx[1:nf])
            pt = p .+ alpha .* (sys0.Cp * dx[(nf + 1):(nf + nfp)])
            rt, _ = residual(ut, pt)
            if scaled(rt) < (1.0 - 1.0e-4 * alpha) * nr
                u, p, r = ut, pt, rt
                accepted = true
                break
            end
            alpha *= 0.5
        end
        push!(hist, scaled(r))
        verbose && @printf("  dp stokes %2d: scaled=%.3e alpha=%.3g ptc=%.1e\n",
                           it, nr, alpha, ptc_delta)
        accepted || break
        converged = scaled(r) <= tol
        if ptc_on
            if scaled(r) > 0.0
                ptc_delta = min(ptc_delta * (nr / scaled(r)),
                                ptc_delta * ptc_growth, ptc_delta_max)
            end
            scaled(r) <= ptc_eta * ptc_r0 && (ptc_on = false)
        end
    end
    return (u=u, p=p, converged=converged, iters=it, history=hist,
            resid=hist[end])
end

"""
    semquad_dp_inner(sq, qt, ss, pr, theta; tol, maxit, u_init, p_init, linear)

Solve the Drucker-Prager inner problem at frozen `theta`: alternate the Stokes
solve with the viscosity evaluated at the current strain rate *and pressure*
until the nodal viscosity field changes by less than `tol` (relative).  Returns
`(u, p, eta, converged, iters, history)`.
"""
function semquad_dp_inner(sq::SemQuad, qt::Quadtree, ss::SemStokes,
                          pr::Params, theta::Vector{Float64};
                          tol::Float64=1.0e-3, maxit::Int=30,
                          u_init::Union{Nothing,Vector{Float64}}=nothing,
                          p_init::Union{Nothing,Vector{Float64}}=nothing,
                          eta_relax::Float64=1.0,
                          method::Symbol=:newton,
                          spd_safety::Float64=0.0,
                          linear::Symbol=:direct,
                          schur_maxiter::Int=200,
                          schur_precond::Symbol=:none)
    N = sq.N
    sp = ss.sp
    leaves = Set(qt.leaves)
    pleaves = Set(sp.leaves)
    if method === :newton
        # The yield-stress feedback makes the Picard fixed point converge very
        # slowly (or not at all); ASPECT solves the same inner problem with a
        # (SPD-stabilised) Newton, which is what `semquad_dp_stokes_newton`
        # does.  The Picard branch is kept for comparison.
        r = semquad_dp_stokes_newton(sq, qt, ss, pr, theta; u_init=u_init,
                                     p_init=p_init, tol=tol, maxit=maxit,
                                     spd_safety=spd_safety, linear=linear)
        eta_field = [semquad_visco_plastic_eta(
                         Tbar(y, pr.H) + QuaST._semquad_eval_at(sq, theta, x, y, leaves),
                         y,
                         semquad_strain_rate_at(sq, r.u[1:N], r.u[(N + 1):(2N)], x, y, leaves),
                         QuaST._semquad_eval_at(sp, r.p, x, y, pleaves),
                         pr) for (x, y) in sq.coords]
        return (u=r.u, p=r.p, eta=eta_field, converged=r.converged,
                iters=r.iters, history=r.history)
    end
    ux = zeros(Float64, N)
    uz = zeros(Float64, N)
    if u_init !== nothing
        ux .= u_init[1:N]
        uz .= u_init[(N + 1):(2N)]
    end
    pv = p_init === nothing ? zeros(Float64, sp.N) : copy(p_init)
    eta_field = nothing
    edot_field = nothing
    u = nothing
    pp = nothing
    hist = Float64[]
    converged = false
    it = 0
    while it < maxit
        it += 1
        # Nodal strain-rate field, optionally under-relaxed.  The DP plastic
        # viscosity is ~1/edot, so a lagged Picard iteration has a strong
        # feedback; `eta_relax < 1` damps it exactly like the driver's
        # `pic_eta_relax` path (G123).
        ed_new = semquad_strain_rate_field(sq, ux, uz)
        edot_field = (edot_field === nothing || eta_relax >= 1.0) ? ed_new :
                     blend_field(edot_field, ed_new, eta_relax)
        eta_fun = (x, y) -> begin
            T = Tbar(y, pr.H) + QuaST._semquad_eval_at(sq, theta, x, y,
                                                           leaves)
            ed = QuaST._semquad_eval_at(sq, edot_field, x, y, leaves)
            pq = QuaST._semquad_eval_at(sp, pv, x, y, pleaves)
            semquad_visco_plastic_eta(T, y, ed, pq, pr)
        end
        u_new, p_new, _, _, _ = solve_semquad_stokes_var(
            ss, qt, eta_fun, (x, y) -> 0.0,
            (x, y) -> pr.Ra * QuaST._semquad_eval_at(sq, theta, x, y,
                                                         leaves),
            (x, y) -> 0.0, (x, y) -> 0.0;
            bc=pr.bc, linear=linear, deviatoric=pr.deviatoric,
            schur_maxiter=schur_maxiter, schur_precond=schur_precond)
        ux .= u_new[1:N]
        uz .= u_new[(N + 1):(2N)]
        pv .= p_new
        u = u_new
        pp = p_new
        eta_new = [eta_fun(x, y) for (x, y) in sq.coords]
        if eta_field === nothing
            push!(hist, NaN)
        else
            ch = norm(eta_new .- eta_field) / max(norm(eta_new), 1.0e-300)
            push!(hist, ch)
            if ch < tol
                eta_field = eta_new
                converged = true
                break
            end
        end
        eta_field = eta_new
    end
    return (u=u, p=pp, eta=eta_field, converged=converged, iters=it,
            history=hist)
end

"""
    semquad_dp_steady_step(sq, qt, ss, pr, theta; dt, dp_tol, dp_maxit, ...)

One backward-Euler energy step with `u, p` from `semquad_dp_inner` at frozen
`theta` (the DP analogue of `semquad_steady_step`), plus the Jacobian saddle of
the map: `Kgeo` (strain-rate derivative) and `Kp_red` (pressure derivative) are
assembled at the base state and reused by every JVP.

`dp_kgeo=false` drops the plastic strain-rate tangent (robust variant for fully
yielded states); `dp_kp=false` drops the pressure block (for comparisons).
"""
function semquad_dp_steady_step(sq::SemQuad, qt::Quadtree, ss::SemStokes,
                                pr::Params, theta::Vector{Float64};
                                dt::Float64=1.0e-2,
                                advection::Symbol=:galerkin,
                                dp_tol::Float64=1.0e-3,
                                dp_maxit::Int=30,
                                dp_inner::Symbol=:newton,
                                spd_safety::Float64=0.0,
                                u_init::Union{Nothing,Vector{Float64}}=nothing,
                                p_init::Union{Nothing,Vector{Float64}}=nothing,
                                linear::Symbol=:direct,
                                dp_kgeo::Bool=true, dp_kp::Bool=true)
    theta = copy(theta)
    N = sq.N
    for g in 1:N
        y = sq.coords[g][2]
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (theta[g] = 0.0)
    end
    inner = semquad_dp_inner(sq, qt, ss, pr, theta; tol=dp_tol, maxit=dp_maxit,
                             u_init=u_init, p_init=p_init, linear=linear,
                             spd_safety=spd_safety, method=dp_inner)
    parts = semquad_steady_parts(sq, qt, pr, theta, inner.u; dt=dt,
                                 advection=advection)
    ur = parts.Ar \ (parts.Cfree' * (parts.rhs - parts.A * parts.u0))
    theta_new = parts.u0 + parts.Cfree * ur
    leaves = Set(qt.leaves)
    pleaves = Set(ss.sp.leaves)
    evalth(th, x, y) = QuaST._semquad_eval_at(sq, th, x, y, leaves)
    u = inner.u
    pv = inner.p
    deta_eps = (x, y, ed) -> begin
        dp_kgeo || return 0.0
        T = Tbar(y, pr.H) + evalth(theta, x, y)
        pq = QuaST._semquad_eval_at(ss.sp, pv, x, y, pleaves)
        semquad_deta_dp_eps(T, y, ed, pq, pr)
    end
    deta_p = (x, y, ed, pq) -> begin
        dp_kp || return 0.0
        T = Tbar(y, pr.H) + evalth(theta, x, y)
        semquad_deta_dp(T, y, ed, pq, pr)
    end
    eta_fun = (x, y) -> begin
        T = Tbar(y, pr.H) + evalth(theta, x, y)
        ed = semquad_strain_rate_at(sq, u[1:N], u[(N + 1):(2N)], x, y, leaves)
        pq = QuaST._semquad_eval_at(ss.sp, pv, x, y, pleaves)
        semquad_visco_plastic_eta(T, y, ed, pq, pr)
    end
    z2 = (x, y) -> 0.0
    sys_jac = solve_semquad_stokes_var(ss, qt, eta_fun, z2, z2, z2, z2;
                                       bc=pr.bc, linear=:direct,
                                       deviatoric=pr.deviatoric,
                                       assemble_only=true, geo_u=u,
                                       geo_coef=deta_eps,
                                       geo_spd_safety=spd_safety, dp_u=u,
                                       dp_pfield=pv, dp_coef=deta_p)
    solver = _semquad_newton_saddle(ss, qt, ss.p, pr.bc, pr.deviatoric, eta_fun,
                                    sys_jac; linear=linear)
    return (theta=theta_new, u=inner.u, p=inner.p, eta=inner.eta,
            parts=parts, inner_iters=inner.iters,
            inner_converged=inner.converged, dt=dt, advection=advection,
            sys_jac=sys_jac, solver=solver, dp_kgeo=dp_kgeo, dp_kp=dp_kp)
end

"""
    semquad_dp_steady_jv(sq, qt, ss, pr, theta, v, step; dt, advection)

Analytic tangent-linear JVP for the DP steady map.  The saddle carries `Kgeo`
and `Kp_red` from the base step; the explicit temperature term is assembled as
a perturbed stiffness applied to the base velocity.  The pressure perturbation
enters through the saddle (the `Kp_red` block), not as an extra right-hand side.
"""
function semquad_dp_steady_jv(sq::SemQuad, qt::Quadtree, ss::SemStokes,
                              pr::Params, theta::Vector{Float64},
                              v::Vector{Float64}, step;
                              dt::Float64=step.dt,
                              advection::Symbol=step.advection)
    N = sq.N
    leaves = Set(qt.leaves)
    pleaves = Set(ss.sp.leaves)
    evalth(th, x, y) = QuaST._semquad_eval_at(sq, th, x, y, leaves)
    u = step.u
    pv = step.p
    deta_T = (x, y) -> begin
        T = Tbar(y, pr.H) + evalth(theta, x, y)
        ed = semquad_strain_rate_at(sq, u[1:N], u[(N + 1):(2N)], x, y, leaves)
        pq = QuaST._semquad_eval_at(ss.sp, pv, x, y, pleaves)
        semquad_dp_deta_T(T, y, ed, pq, pr) * evalth(v, x, y)
    end
    sys = step.sys_jac
    z2 = (x, y) -> 0.0
    sd = solve_semquad_stokes_var(ss, qt, deta_T, z2, z2, z2, z2;
                                  bc=pr.bc, linear=:direct,
                                  deviatoric=false, assemble_only=true)
    dKu = sd.K * u
    nf = size(sys.Kred, 1)
    nfp = size(sys.Bs, 1)
    Mv = assemble_semquad_pressure_mass(ss.su)
    df = vcat(zeros(Float64, N), pr.Ra * (Mv * v))
    rhs = sys.Cu' * (df - dKu)
    sol = step.solver.solve(vcat(rhs, zeros(Float64, nfp)))
    du = sys.Cu * sol[1:nf]
    _, _, Cd, Fd = assemble_semquad_energy(sq, qt, pr, du[1:N],
                                           du[(N + 1):(2N)];
                                           advection=advection,
                                           theta=step.theta)
    drhs = (1.0 / dt) * (step.parts.M * v)
    br = step.parts.Cfree' * (drhs + Fd - Cd * step.theta)
    return step.parts.Cfree * (step.parts.Ar \ br)
end

"""
    semquad_dp_newton(sq, qt, ss, pr, theta0; ...)

Damped Newton-Krylov solve of the DP steady map `G_DP(theta) = theta`; the JVP
is `semquad_dp_steady_jv` (Kgeo + Kp saddle).  Mirrors `semquad_tosi_newton`,
including the optional pre-Picard sweeps.
"""
function semquad_dp_newton(sq::SemQuad, qt::Quadtree, ss::SemStokes,
                           pr::Params, theta0::Vector{Float64};
                           dt::Float64=1.0e-2, tol::Float64=1.0e-8,
                           maxit::Int=15, krylov::Int=40,
                           restart::Int=20, verbose::Bool=false,
                           advection::Symbol=:galerkin,
                           picard_steps::Int=0,
                           dp_tol::Float64=1.0e-3,
                           dp_maxit::Int=30,
                           dp_kgeo::Bool=true, dp_kp::Bool=true,
                           linear::Symbol=:direct)
    theta = copy(theta0)
    N = sq.N
    for g in 1:N
        y = sq.coords[g][2]
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (theta[g] = 0.0)
    end
    step = semquad_dp_steady_step(sq, qt, ss, pr, theta; dt=dt,
                                  advection=advection, dp_tol=dp_tol,
                                  dp_maxit=dp_maxit, linear=linear,
                                  dp_kgeo=dp_kgeo, dp_kp=dp_kp)
    for k in 1:picard_steps
        theta = step.theta
        step = semquad_dp_steady_step(sq, qt, ss, pr, theta; dt=dt,
                                      advection=advection, dp_tol=dp_tol,
                                      dp_maxit=dp_maxit, u_init=step.u,
                                      p_init=step.p, linear=linear,
                                      dp_kgeo=dp_kgeo, dp_kp=dp_kp)
        verbose && @printf("  picard %2d: |F| = %.3e (inner %d)\n", k,
                           norm(step.theta - theta), step.inner_iters)
    end
    F = step.theta - theta
    hist = [norm(F)]
    converged = norm(F) <= tol
    it = 0
    while !converged && it < maxit
        it += 1
        nr = norm(F)
        Jop = v -> semquad_dp_steady_jv(sq, qt, ss, pr, theta, v, step;
                                        dt=dt, advection=advection) - v
        dx, rel, nit = gmres_restarted(Jop, -F; tol=1.0e-8, maxit=krylov,
                                       restart=restart)
        alpha = 1.0
        accepted = false
        for _ in 1:12
            theta_t = theta .+ alpha .* dx
            step_t = semquad_dp_steady_step(sq, qt, ss, pr, theta_t; dt=dt,
                                            advection=advection,
                                            dp_tol=dp_tol,
                                            dp_maxit=dp_maxit,
                                            u_init=step.u, p_init=step.p,
                                            linear=linear, dp_kgeo=dp_kgeo,
                                            dp_kp=dp_kp)
            F_t = step_t.theta - theta_t
            if norm(F_t) < (1.0 - 1.0e-4 * alpha) * nr
                theta = theta_t
                step = step_t
                F = F_t
                accepted = true
                break
            end
            alpha *= 0.5
        end
        push!(hist, norm(F))
        verbose && @printf("  newton %2d: |F| = %.3e (alpha = %.2g, gmres %d, rel %.1e)\n",
                           it, norm(F), alpha, nit, rel)
        accepted || break
        converged = norm(F) <= tol
    end
    return (theta=theta, step=step, converged=converged, iters=it,
            history=hist)
end
