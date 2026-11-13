# Steady Newton for the Tosi (2015) viscoplastic rheology on route C.
#
# The Tosi viscosity depends on temperature *and* on the strain rate,
#
#   eta = eta(T, eps_II),   eps_II = sqrt(0.5 edot:edot),
#
# so the steady map is no longer a function of theta alone: for a fixed theta the
# velocity must satisfy the Stokes equations with the viscosity evaluated at its
# own strain rate.  `semquad_tosi_inner` solves that inner problem by Picard
# iteration (Stokes <-> strain rate) at frozen theta; `semquad_tosi_steady_step`
# then advances the energy equation, exactly like `semquad_steady_step` does for
# eta(T).
#
# The tangent linearisation of the inner Stokes problem has two parts:
#
#   delta-eta = (deta/dT) delta-theta  +  (deta/deps) delta-eps(u)
#   delta-K u = K(deta_T) u + Kgeo delta-u
#
# The first term is assembled explicitly as a perturbed stiffness; the second is
# the geometric operator `Kgeo` already validated by G156
# (`solve_semquad_stokes_var(...; geo_u=u, geo_coef=deta_deps)` returns
# `Kred = Cu'(K + Kgeo)Cu`).  The energy linearisation is unchanged.
#
# Gate: G161 compares the Newton fixed point with the endpoint of a route-C Tosi
# time-stepping run (Case 1a parameters).

"""
    semquad_tosi_inner(sq, qt, ss, pr, theta; tol=1e-3, maxit=20, u_init=nothing)

Solve the Tosi inner problem at frozen `theta`: alternate the Stokes solve with
the viscosity evaluated at the current strain rate until the nodal viscosity
field changes by less than `tol` in relative norm.  Returns
`(u, p, eta, converged, iters, history)` where `eta` is the viscosity field at
the velocity nodes.
"""
function semquad_tosi_inner(sq::SemQuad, qt::Quadtree, ss::SemStokes,
                            pr::Params, theta::Vector{Float64};
                            tol::Float64=1.0e-3, maxit::Int=20,
                            u_init::Union{Nothing,Vector{Float64}}=nothing,
                            linear::Symbol=:direct,
                            schur_maxiter::Int=200,
                            schur_precond::Symbol=:none,
                            pic_eta_relax::Float64=1.0)
    N = sq.N
    leaves = Set(qt.leaves)
    ux = zeros(Float64, N)
    uz = zeros(Float64, N)
    if u_init !== nothing
        ux .= u_init[1:N]
        uz .= u_init[(N + 1):(2N)]
    end
    eta_field = nothing
    u = nothing
    pp = nothing
    hist = Float64[]
    converged = false
    it = 0
    while it < maxit
        it += 1
        eta_fun = (x, y) -> begin
            T = Tbar(y, pr.H) + MantleSEM._semquad_eval_at(sq, theta, x, y,
                                                           leaves)
            ed = semquad_strain_rate_at(sq, ux, uz, x, y, leaves)
            semquad_eta_tosi(T, ed, y, pr.tosi)[1]
        end
        u_new, p_new, _, _, _ = solve_semquad_stokes_var(
            ss, qt, eta_fun, (x, y) -> 0.0,
            (x, y) -> pr.Ra * MantleSEM._semquad_eval_at(sq, theta, x, y,
                                                         leaves),
            (x, y) -> 0.0, (x, y) -> 0.0;
            bc=pr.bc, linear=linear, deviatoric=pr.deviatoric,
            schur_maxiter=schur_maxiter, schur_precond=schur_precond)
        if pic_eta_relax < 1.0 && u !== nothing
            for g in 1:N
                ux[g] = (1.0 - pic_eta_relax) * ux[g] + pic_eta_relax * u_new[g]
                uz[g] = (1.0 - pic_eta_relax) * uz[g] +
                        pic_eta_relax * u_new[N + g]
            end
        else
            ux .= u_new[1:N]
            uz .= u_new[(N + 1):(2N)]
        end
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
    semquad_tosi_steady_step(sq, qt, ss, pr, theta; dt, tosi_tol, tosi_maxit, ...)

One backward-Euler energy step with `u` from `semquad_tosi_inner` at frozen
`theta` (the Tosi analogue of `semquad_steady_step`).  Returns the same fields
as the eta(T) step plus `eta`, `inner_iters` and `inner_converged`.
"""
function semquad_tosi_steady_step(sq::SemQuad, qt::Quadtree, ss::SemStokes,
                                  pr::Params, theta::Vector{Float64};
                                  dt::Float64=1.0e-2,
                                  advection::Symbol=:galerkin,
                                  tosi_tol::Float64=1.0e-3,
                                  tosi_maxit::Int=20,
                                  u_init::Union{Nothing,Vector{Float64}}=nothing,
                                  linear::Symbol=:direct)
    theta = copy(theta)
    N = sq.N
    for g in 1:N
        y = sq.coords[g][2]
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (theta[g] = 0.0)
    end
    inner = semquad_tosi_inner(sq, qt, ss, pr, theta; tol=tosi_tol,
                               maxit=tosi_maxit, u_init=u_init, linear=linear)
    parts = semquad_steady_parts(sq, qt, pr, theta, inner.u; dt=dt,
                                 advection=advection)
    ur = parts.Ar \ (parts.Cfree' * (parts.rhs - parts.A * parts.u0))
    theta_new = parts.u0 + parts.Cfree * ur
    # Jacobian Stokes operator (with Kgeo) assembled once per step and reused by
    # every JVP of the Newton Krylov solve.  Rebuilding it per JVP made the
    # Tosi Newton cost one saddle LU per Krylov iteration.
    leaves = Set(qt.leaves)
    evalth(th, x, y) = MantleSEM._semquad_eval_at(sq, th, x, y, leaves)
    u = inner.u
    deta_eps = (x, y, ed) -> begin
        T = Tbar(y, pr.H) + evalth(theta, x, y)
        semquad_deta_tosi(T, ed, y, pr.tosi)[1]
    end
    eta_fun = (x, y) -> begin
        T = Tbar(y, pr.H) + evalth(theta, x, y)
        ed = semquad_strain_rate_at(sq, u[1:sq.N], u[(sq.N + 1):(2sq.N)], x, y,
                                    leaves)
        semquad_eta_tosi(T, ed, y, pr.tosi)[1]
    end
    z2 = (x, y) -> 0.0
    sys_jac = solve_semquad_stokes_var(ss, qt, eta_fun, z2, z2, z2, z2;
                                       bc=pr.bc, linear=:direct,
                                       deviatoric=pr.deviatoric,
                                       assemble_only=true, geo_u=u,
                                       geo_coef=deta_eps)
    solver = _semquad_newton_saddle(ss, qt, ss.p, pr.bc, pr.deviatoric, eta_fun,
                                    sys_jac; linear=linear)
    return (theta=theta_new, u=inner.u, p=inner.p, eta=inner.eta,
            parts=parts, inner_iters=inner.iters,
            inner_converged=inner.converged, dt=dt, advection=advection,
            sys_jac=sys_jac, solver=solver)
end

"""
    semquad_tosi_steady_jv(sq, qt, ss, pr, theta, v, step; dt, advection)

Analytic tangent-linear JVP for the Tosi steady map.  The Stokes operator
includes `Kgeo` (the `deta/deps` chain rule) and the explicit `deta/dT` term is
assembled as a perturbed stiffness applied to the base velocity.
"""
function semquad_tosi_steady_jv(sq::SemQuad, qt::Quadtree, ss::SemStokes,
                                pr::Params, theta::Vector{Float64},
                                v::Vector{Float64}, step;
                                dt::Float64=step.dt,
                                advection::Symbol=step.advection)
    N = sq.N
    leaves = Set(qt.leaves)
    evalth(th, x, y) = MantleSEM._semquad_eval_at(sq, th, x, y, leaves)
    u = step.u
    # Explicit temperature dependence: deta_T = (deta/dT) delta-theta.
    deta_T = (x, y) -> begin
        T = Tbar(y, pr.H) + evalth(theta, x, y)
        ed = semquad_strain_rate_at(sq, u[1:N], u[(N + 1):(2N)], x, y, leaves)
        _, deta_dT = semquad_deta_tosi(T, ed, y, pr.tosi)
        deta_dT * evalth(v, x, y)
    end
    # The Kgeo-augmented Stokes system and its saddle solver were assembled once
    # by `semquad_tosi_steady_step`; reuse them for every JVP.
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
    semquad_tosi_newton(sq, qt, ss, pr, theta0; ...)

Damped Newton-Krylov solve of `G_Tosi(theta) = theta` with the analytic Tosi
JVP.  `picard_steps` warm-up sweeps use `semquad_tosi_steady_step` (which itself
solves the inner strain-rate problem), mirroring `semquad_newton_steady`.
"""
function semquad_tosi_newton(sq::SemQuad, qt::Quadtree, ss::SemStokes,
                             pr::Params, theta0::Vector{Float64};
                             dt::Float64=1.0e-2, tol::Float64=1.0e-8,
                             maxit::Int=15, krylov::Int=40,
                             restart::Int=20, verbose::Bool=false,
                             advection::Symbol=:galerkin,
                             picard_steps::Int=0,
                             tosi_tol::Float64=1.0e-3,
                             tosi_maxit::Int=20,
                             linear::Symbol=:direct)
    theta = copy(theta0)
    N = sq.N
    for g in 1:N
        y = sq.coords[g][2]
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (theta[g] = 0.0)
    end
    step = semquad_tosi_steady_step(sq, qt, ss, pr, theta; dt=dt,
                                    advection=advection, tosi_tol=tosi_tol,
                                    tosi_maxit=tosi_maxit, linear=linear)
    for k in 1:picard_steps
        theta = step.theta
        step = semquad_tosi_steady_step(sq, qt, ss, pr, theta; dt=dt,
                                        advection=advection,
                                        tosi_tol=tosi_tol,
                                        tosi_maxit=tosi_maxit,
                                        u_init=step.u, linear=linear)
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
        Jop = v -> semquad_tosi_steady_jv(sq, qt, ss, pr, theta, v, step;
                                          dt=dt, advection=advection) - v
        dx, rel, nit = gmres_restarted(Jop, -F; tol=1.0e-8, maxit=krylov,
                                       restart=restart)
        alpha = 1.0
        accepted = false
        for _ in 1:12
            theta_t = theta .+ alpha .* dx
            step_t = semquad_tosi_steady_step(sq, qt, ss, pr, theta_t; dt=dt,
                                              advection=advection,
                                              tosi_tol=tosi_tol,
                                              tosi_maxit=tosi_maxit,
                                              u_init=step.u, linear=linear)
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
