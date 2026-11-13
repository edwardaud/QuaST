# Analytic tangent-linear Newton for the steady route-C step map.
#
# The map is
#
#   G(theta) = one backward-Euler energy step with
#              u = Stokes(eta(theta)),  theta_prev = theta,
#
# so a fixed point `G(theta) = theta` is a steady solution of the coupled
# Boussinesq system.  `theta` is a perturbation field, i.e. the physical
# temperature is `T = Tbar(y,H) + theta`.
#
# The Jacobian-vector product is the exact tangent linearisation for a
# temperature-dependent viscosity (`:const`, `:expT`, `:expTz`, `:arrhenius`):
#
#   delta-eta   = deta/dT * delta-theta
#   delta-K u   = K(delta-eta) u
#   [K B'; B 0] [delta-u; delta-p] = [delta-f - delta-K u; 0]
#   E delta-theta+ = (1/dt) M delta-theta + F(delta-u) - C(delta-u) theta+
#   E = (1/dt) M + K_T + C(u)
#
# and `J_F v = delta-theta+ - v`.  This is the tangent-linear JVP used inside
# GMRES; it avoids the finite-difference noise that makes the FD/JFNK version
# stall at ~1e-3 in the strong-viscosity case (see ROUTE_C_STATUS §4.9).
#
# Restrictions (checked up front): Boussinesq temperature-dependent viscosity,
# no viscoplastic/yielding rheology, no entropy viscosity (ev=0), no phase
# changes or heating terms that depend on theta, and a direct saddle solve.
# These are exactly the 2b benchmark's settings.  Extending the map to EV or
# strain-rate-dependent viscosity requires their derivatives as well.

"""
    semquad_steady_parts(sq, qt, pr, theta, u; dt, advection)

Assemble `M, K, C, F` and the constrained energy operator for one backward-Euler
step at `theta_prev = theta`.
"""
function semquad_steady_parts(sq::SemQuad, qt::Quadtree, pr::Params,
                              theta::Vector{Float64}, u::Vector{Float64};
                              dt::Float64=1.0e-2,
                              advection::Symbol=:galerkin,
                              ev::Float64=0.0, ev_cmax::Float64=0.052,
                              ev_alpha::Int=2, ev_timestep::Int=10)
    N = sq.N
    ev_field = nothing
    if ev > 0.0
        evg = semquad_ev_globals(sq, qt, pr, theta, theta,
                                 u[1:N], u[(N + 1):(2N)], 0.0)
        ev_field = zeros(Float64, length(qt.leaves))
        M, K, C, F = assemble_semquad_energy(sq, qt, pr, u[1:N],
                                             u[(N + 1):(2N)];
                                             advection=advection, theta=theta,
                                             ev=ev, ev_cmax=ev_cmax,
                                             ev_alpha=ev_alpha,
                                             ev_timestep=ev_timestep,
                                             evg=evg, theta_prev=theta,
                                             dt=dt, ev_out=ev_field)
    else
        M, K, C, F = assemble_semquad_energy(sq, qt, pr, u[1:N],
                                             u[(N + 1):(2N)];
                                             advection=advection, theta=theta)
    end
    fixed = Dict{Int,Float64}()
    for g in 1:N
        y = sq.coords[g][2]
        y <= 1.0e-14 && (fixed[g] = pr.Twall_bot - semquad_base_T(pr, y))
        y >= 1.0 - 1.0e-14 && (fixed[g] = pr.Twall_top - semquad_base_T(pr, y))
    end
    cons = semquad_constraints(sq, qt)
    u0, Cfree, _ = _affine_constraint(N, cons, fixed)
    A = (1.0 / dt) * M + K + C
    rhs = (1.0 / dt) * (M * theta) + F
    Ar = Cfree' * A * Cfree
    return (M=M, K=K, C=C, F=F, A=A, rhs=rhs, u0=u0, Cfree=Cfree,
            Ar=Ar, fixed=fixed, dt=dt, ev_field=ev_field)
end

"""
    _semquad_newton_saddle(ss, qt, p, bc, deviatoric, eta_fun, sys; linear=:direct,
                           tol=1e-10, maxit=500, restart=100, ...)

Build the reduced-saddle solver reused by the steady step and its JVP.

`linear=:direct` reproduces the historical path: the monolithic saddle
`[Kred Bt; Bs 0]` is shifted by `1e-12 * max|diag|` and factorised once with
UMFPACK; every JVP right-hand side is then a back-substitution.

`linear=:mg` builds Route C's geometric multigrid for `Kred` and the
`semquad_mg_pressure_precond` Schur block once, and solves each right-hand side
with right-preconditioned GMRES (the same preconditioner as the transient
`linear=:mg` path).  This removes the direct factorisation from the Newton
iteration, which is what limits the wall-refined 2b Newton solve.

Returns a named tuple with `solve(b)` plus the factorisation/hierarchy.
"""
function _semquad_newton_saddle(ss::SemStokes, qt::Quadtree, p::Int, bc::Symbol,
                                deviatoric::Bool, eta_fun::Function, sys;
                                linear::Symbol=:direct,
                                tol::Float64=1.0e-8, maxit::Int=500,
                                restart::Int=100,
                                mg_cycles::Int=1, mg_smoother::Symbol=:auto,
                                mg_coarse::Symbol=:auto,
                                mg_operator::Symbol=:matrixfree,
                                mg_vel_solver::Symbol=:vcycle,
                                mg_vel_tol::Float64=1.0e-2,
                                mg_vel_maxit::Int=10,
                                mg_pressure_mode::Symbol=:auto,
                                mg_pressure_scale::Float64=0.0)
    nf = size(sys.Kred, 1)
    nfp = size(sys.Bs, 1)
    # Drucker-Prager: the momentum row gains the pressure-derivative block
    # Kp_red (velocity<->pressure coupling), so Bt becomes sys.Bt + Kp_red and
    # the saddle is non-symmetric.  `sys.Kp_red === nothing` keeps the historical
    # (symmetric) system bit for bit.
    Bt = sys.Kp_red === nothing ? sys.Bt : sys.Bt + sys.Kp_red
    if linear === :direct
        S = [sys.Kred Bt; sys.Bs spzeros(Float64, nfp, nfp)]
        dmax = maximum(abs, diag(S)) + 1.0
        # The historical shift is 1e-12*max|diag(S)| and is tried first, so a
        # healthy system stays bit-identical.  A near-singular Kred+Kgeo (strong
        # viscosity gradient, e.g. 2b at L4) can still give UMFPACK a zero pivot;
        # the relative shift then grows until the factorisation succeeds.  This
        # is the "adaptive shift" that the L4 negative result in
        # ROUTE_C_STATUS §1.14 asked for.
        Slu = nothing
        used = 0.0
        for rel in (1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2)
            try
                Slu = lu(S + (rel * dmax) * I)
                used = rel
                break
            catch err
                err isa SingularException || rethrow()
            end
        end
        Slu === nothing &&
            error("_semquad_newton_saddle: direct LU failed for every shift up " *
                  "to 1e-4*max|diag| (n=$(nf + nfp))")
        return (solve = b -> Slu \ b, kind=:direct, nf=nf, nfp=nfp, Slu=Slu,
                shift=used)
    elseif linear === :mg
        # The direct path solves the shifted saddle S + 1e-12*max|diag(S)| I.
        # Apply the same relative shift to Kred: Newton's JVP right-hand sides
        # are arbitrary reduced vectors and can contain a component along a
        # near-null velocity mode.  Without it the inner GMRES was observed to
        # diverge (relres 1e5-1e7); with it, residual failures are confined to
        # the line-search trial states, whose viscosity can be pathological and
        # which the search rejects anyway.
        dmax = maximum(abs, diag(sys.Kred)) + 1.0
        Kred = sys.Kred + (1.0e-12 * dmax) * I
        uniform = length(unique(lf[1] for lf in qt.leaves)) == 1
        smoother = mg_smoother === :auto ? :jacobi : mg_smoother
        coarse = mg_coarse === :auto ?
                 (uniform ? :rediscretize : :galerkin) : mg_coarse
        # S3/MF-4: the matrix-free hierarchy is the default here too (it stores
        # no velocity operator and builds 44-79x faster); the sparse path stays
        # available as `mg_operator=:sparse`.
        mg_operator === :matrixfree && (coarse = :rediscretize)
        mg = make_semquad_mg(ss, qt, p, bc, deviatoric, eta_fun, Kred;
                             cycles=mg_cycles, smoother=smoother,
                             coarse_operator=coarse, operator=mg_operator)
        # NB: with a Kp block the pressure Schur approximation is built from the
        # non-symmetric Bt; the MG path is a best-effort option for the DP
        # Newton (the direct path is the default there).
        Fp, pscale = semquad_mg_pressure_precond(ss, sys.Cu, sys.Cp, sys.Bs,
                                                 Bt, Kred, eta_fun;
                                                 mode=mg_pressure_mode,
                                                 scale=mg_pressure_scale)
        Bs = sys.Bs
        # A warm start from the previous JVP solution was measured to *hurt*
        # (the successive Krylov directions are not correlated enough: L4 JVP
        # 0.49 s cold vs 0.63 s warm-started), so every solve starts from zero.
        # Using one block-preconditioner application as the JVP solve (inexact
        # Newton) was also tried and diverged (|F| 5.4, Vrms 646): the JVP must
        # be solved to ~1e-8 for the outer Newton GMRES to work.
        # The saddle itself is solved exactly as in the transient path
        # (`semquad_mg_saddle_solve`: FGMRES outside, CG+MG inside for the SPD
        # velocity block, three-factor BFBT/mass pressure block).
        solve = function (b)
            sol, nit, relres, _ = semquad_mg_saddle_solve(Kred, Bs, Bt, Fp,
                                                          pscale,
                                                          Vector{Float64}(b),
                                                          mg; tol=tol,
                                                          maxit=maxit,
                                                          restart=restart,
                                                          vel_solver=mg_vel_solver,
                                                          vel_tol=mg_vel_tol,
                                                          vel_maxit=mg_vel_maxit)
            # Near convergence the JVP right-hand sides become tiny (|b| ~ 1e-9
            # of the base step); a relative-residual warning would then fire on
            # round-off although the absolute error is at the direct solve's
            # level.  Only warn when the RHS is a meaningful fraction of the
            # physical [fr; gr].
            bnorm = norm(vcat(sys.fr, sys.gr))
            if relres > 1.0e-5 && norm(b) > 1.0e-8 * max(bnorm, 1.0)
                @warn "Newton MG saddle FGMRES relres=$relres (|b|=$(norm(b)))"
            end
            return sol
        end
        return (solve=solve, kind=:mg, nf=nf, nfp=nfp, mg=mg, Fp=Fp,
                pscale=pscale)
    end
    throw(ArgumentError("Newton linear must be :direct or :mg"))
end

"""
    semquad_steady_step(sq, qt, ss, pr, theta; dt=1e-2, advection=:galerkin)

Evaluate `G(theta)`: Stokes with `eta(theta)`, then one backward-Euler energy
step with `theta_prev = theta`.  Returns a named tuple with `theta`, `u`, `p`,
the assembled parts, and the reduced saddle solver (reused by the JVP).
"""
function semquad_steady_step(sq::SemQuad, qt::Quadtree, ss::SemStokes,
                             pr::Params, theta::Vector{Float64};
                             dt::Float64=1.0e-2,
                             advection::Symbol=:galerkin,
                             ev::Float64=0.0, ev_cmax::Float64=0.052,
                             ev_alpha::Int=2, ev_timestep::Int=10,
                             linear::Symbol=:direct,
                             mg_cycles::Int=1, mg_smoother::Symbol=:auto,
                             mg_coarse::Symbol=:auto,
                             mg_operator::Symbol=:matrixfree,
                             mg_vel_solver::Symbol=:vcycle,
                             mg_vel_tol::Float64=1.0e-1,
                             mg_vel_maxit::Int=10,
                             mg_pressure_mode::Symbol=:auto,
                             mg_pressure_scale::Float64=0.0,
                             newton_tol::Float64=1.0e-8,
                             newton_maxit::Int=500,
                             newton_restart::Int=100)
    theta = copy(theta)
    N = sq.N
    # homogeneous wall values: theta is a perturbation with zero walls
    for g in 1:N
        y = sq.coords[g][2]
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (theta[g] = 0.0)
    end
    leaves = Set(qt.leaves)
    eta = (x, y) -> semquad_eta_at(sq, theta, x, y, pr, leaves)
    fz = (x, y) -> pr.Ra * MantleSEM._semquad_eval_at(sq, theta, x, y, leaves)
    z2 = (x, y) -> 0.0
    sys = solve_semquad_stokes_var(ss, qt, eta, z2, fz, z2, z2;
                                   bc=pr.bc, linear=:direct,
                                   deviatoric=false, assemble_only=true)
    nf = size(sys.Kred, 1)
    nfp = size(sys.Bs, 1)
    solver = _semquad_newton_saddle(ss, qt, ss.p, pr.bc, false, eta, sys;
                                    linear=linear, tol=newton_tol,
                                    maxit=newton_maxit, restart=newton_restart,
                                    mg_cycles=mg_cycles, mg_smoother=mg_smoother,
                                    mg_coarse=mg_coarse,
                               mg_operator=mg_operator,
                               mg_vel_solver=mg_vel_solver,
                               mg_vel_tol=mg_vel_tol,
                               mg_vel_maxit=mg_vel_maxit,
                                    mg_pressure_mode=mg_pressure_mode,
                                    mg_pressure_scale=mg_pressure_scale)
    sol = solver.solve(vcat(sys.fr, sys.gr))
    u = sys.u0 + sys.Cu * sol[1:nf]
    pv = sys.p0 + sys.Cp * sol[(nf + 1):(nf + nfp)]
    parts = semquad_steady_parts(sq, qt, pr, theta, u; dt=dt,
                                 advection=advection, ev=ev,
                                 ev_cmax=ev_cmax, ev_alpha=ev_alpha,
                                 ev_timestep=ev_timestep)
    ur = parts.Ar \ (parts.Cfree' * (parts.rhs - parts.A * parts.u0))
    theta_new = parts.u0 + parts.Cfree * ur
    Mv = assemble_semquad_pressure_mass(ss.su)
    return (theta=theta_new, u=u, p=pv, parts=parts, sys=sys, solver=solver,
            Mv=Mv, nf=nf, nfp=nfp, dt=dt, advection=advection,
            ev=ev, ev_cmax=ev_cmax, ev_alpha=ev_alpha,
            ev_timestep=ev_timestep)
end

"""
    _semquad_jvp_ku(sq, q, pr, theta, v, u)

Matrix-free `K(deta) * u` for the Newton JVP, with
`deta = (deta/dT) * delta-theta`.

The JVP only needs the action of the perturbed velocity stiffness on the base
velocity, but the historical path built the whole global K/B/constraint system
(`assemble_only=true`) just to use it.  This routine evaluates the temperature
and its perturbation at each element's own quadrature points from the local
nodal values (no tree search), builds the same element stiffness as the
production assembler (`deviatoric=false`, same quadrature), and applies it to
the element-local base velocity.  The cost is O(nleaves * (2*nv)^2) instead of
a global sparse assembly, and it matches the assembled `K * u` to round-off.
"""
function _semquad_jvp_ku(sq::SemQuad, q::Int, pr::Params,
                         theta::Vector{Float64}, v::Vector{Float64},
                         u::Vector{Float64};
                         deta_T_fun::Union{Nothing,Function}=nothing)
    N = sq.N
    p = sq.p
    n = p + 1
    nv = n * n
    fd = make_fedata(p, q)
    phi = zeros(Float64, nv)
    gx = zeros(Float64, nv)
    gz = zeros(Float64, nv)
    Kloc = zeros(Float64, 2nv, 2nv)
    uloc = zeros(Float64, 2nv)
    outl = zeros(Float64, 2nv)
    out = zeros(Float64, 2 * N)
    a2 = 2.0
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = sq.elem[lf]
        fill!(Kloc, 0.0)
        for qy in 1:fd.nq, qx in 1:fd.nq
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            Tq = 0.0; dTq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                phi[k] = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                gx[k] = fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / hx)
                gz[k] = fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hy)
                Tq += theta[ids[k]] * phi[k]
                dTq += v[ids[k]] * phi[k]
            end
            T = Tbar(y, pr.H) + Tq
            # `deta_T_fun` overrides the built-in d eta/d T of the temperature
            # laws (the composite/strain-rate-dependent laws need their own
            # derivative, which also has to see the strain rate).
            deta = deta_T_fun === nothing ? deta_dT_at(T, y, pr) :
                   deta_T_fun(x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx, y)
            etaq = deta * dTq
            for k in 1:nv, l in 1:nv
                Kloc[k, l] += etaq * (a2 * gx[k] * gx[l] + gz[k] * gz[l]) * w
                Kloc[nv + k, nv + l] += etaq * (a2 * gz[k] * gz[l] +
                                                gx[k] * gx[l]) * w
                Kloc[k, nv + l] += etaq * (gz[k] * gx[l]) * w
                Kloc[nv + k, l] += etaq * (gx[k] * gz[l]) * w
            end
        end
        for k in 1:nv
            uloc[k] = u[ids[k]]
            uloc[nv + k] = u[N + ids[k]]
        end
        mul!(outl, Kloc, uloc)
        for k in 1:nv
            out[ids[k]] += outl[k]
            out[N + ids[k]] += outl[nv + k]
        end
    end
    return out
end

"""
    _semquad_ev_directional(sq, qt, pr, theta, v, u, du, step; h=1e-6)

Directional derivative of the elementwise entropy-viscosity field along
`(v, du)`:
`(nu(theta + h v, u + h du) - nu(theta - h v, u - h du)) / (2h)`.
The EV field depends on `theta` (through the entropy residual and the global
variation) and on `u` (through the residual's advection term and `umax`), so
both directions are perturbed.  Cost: two O(leaves) field evaluations.
"""
function _semquad_ev_directional(sq::SemQuad, qt::Quadtree, pr::Params,
                                 theta::Vector{Float64}, v::Vector{Float64},
                                 u::Vector{Float64}, du::Vector{Float64},
                                 step; h::Float64=1.0e-6)
    kw = (dt=step.dt, ev=step.ev, ev_cmax=step.ev_cmax,
          ev_alpha=step.ev_alpha, ev_timestep=step.ev_timestep)
    N = sq.N
    up = u .+ h .* du
    um = u .- h .* du
    fp = semquad_ev_field(sq, qt, pr, theta .+ h .* v, up[1:N],
                          up[(N + 1):(2N)]; kw...)
    fm = semquad_ev_field(sq, qt, pr, theta .- h .* v, um[1:N],
                          um[(N + 1):(2N)]; kw...)
    return (fp .- fm) ./ (2h)
end

"""
    _semquad_ev_diffusion(sq, q, theta, dev)

Matrix-free `D_ev(dev) theta`, where the EV stiffness is
`sum_e dev_e * int_e grad(phi_k) . grad(phi_l)` (the `(1 + nu_ev)` factor of
`assemble_semquad_energy` minus the physical `1`).  The product is returned in
the full (unconstrained) space, like the assembled `K`; the caller applies
`Cfree'`.
"""
function _semquad_ev_diffusion(sq::SemQuad, q::Int, theta::Vector{Float64},
                               dev::Vector{Float64})
    N = sq.N
    p = sq.p
    n = p + 1
    nv = n * n
    fd = make_fedata(p, q)
    out = zeros(Float64, N)
    gx = zeros(Float64, nv)
    gz = zeros(Float64, nv)
    for (ileaf, lf) in enumerate(sq.leaves)
        nu_e = dev[ileaf]
        nu_e == 0.0 && continue
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            gthx = 0.0; gthz = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                gx[k] = fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / hx)
                gz[k] = fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hy)
                gthx += theta[ids[k]] * gx[k]
                gthz += theta[ids[k]] * gz[k]
            end
            c = nu_e * w
            for k in 1:nv
                out[ids[k]] += c * (gthx * gx[k] + gthz * gz[k])
            end
        end
    end
    return out
end

"""
    semquad_steady_jv(sq, qt, ss, pr, theta, v, step; dt, advection, ev_tangent)

Analytic tangent-linear Jacobian-vector product `J_G(theta) v` for the steady
map evaluated in `step = semquad_steady_step(...)`.

With `ev_tangent=true` the *entropy-viscosity field* is differentiated as well:
`G` recomputes `nu_ev(theta, u)` on every evaluation, so freezing it (the
historical `ev_tangent=false` behaviour) makes the Newton a fixed-point
iteration in that dependence (linear convergence, ~0.6 per step on 2b;
measured JVP-vs-FD error 15.7% on an EV state, vs 3.6e-8 with the tangent).
`semquad_newton_steady` passes `ev_tangent=true` by default.  The directional
derivative is obtained by a central difference of the field along `(v, du)`
(two O(leaves) evaluations) and enters the energy linearisation as
`-D_ev(d nu) theta` in the residual.
"""
function semquad_steady_jv(sq::SemQuad, qt::Quadtree, ss::SemStokes,
                           pr::Params, theta::Vector{Float64},
                           v::Vector{Float64}, step; dt::Float64=step.dt,
                           advection::Symbol=step.advection,
                           ev_tangent::Bool=false)
    N = sq.N
    # Matrix-free K(deta) u with the perturbed temperature evaluated from the
    # element's own nodal values (no tree search).  The historical path built
    # the full assemble_only Stokes system just to use `sd.K * step.u`, which
    # was 2.8 s of the 10.1 s L6-uniform JVP.
    dKu = _semquad_jvp_ku(sq, ss.q, pr, theta, v, step.u)
    df = vcat(zeros(Float64, N), pr.Ra * (step.Mv * v))
    rhs = step.sys.Cu' * (df - dKu)
    sol = step.solver.solve(vcat(rhs, zeros(Float64, step.nfp)))
    du = step.sys.Cu * sol[1:step.nf]
    _, _, Cd, Fd = if step.parts.ev_field === nothing
        assemble_semquad_energy(sq, qt, pr, du[1:N], du[(N + 1):(2N)];
                                advection=advection, theta=step.theta)
    else
        # EV is frozen at the base step by default; `ev_tangent` adds its
        # derivative below.  The fixed point is self-consistent either way
        # because `semquad_steady_step` recomputes the field.
        assemble_semquad_energy(sq, qt, pr, du[1:N], du[(N + 1):(2N)];
                                advection=advection, theta=step.theta,
                                ev=step.ev, ev_cmax=step.ev_cmax,
                                ev_alpha=step.ev_alpha,
                                ev_timestep=step.ev_timestep,
                                ev_field=step.parts.ev_field)
    end
    drhs = (1.0 / dt) * (step.parts.M * v)
    dev_term = zeros(Float64, N)
    if ev_tangent && step.parts.ev_field !== nothing
        dev = _semquad_ev_directional(sq, qt, pr, theta, v, step.u, du, step)
        if any(!=(0.0), dev)
            dev_term = _semquad_ev_diffusion(sq, ss.q, step.theta, dev)
        end
    end
    br = step.parts.Cfree' * (drhs + Fd - Cd * step.theta - dev_term)
    return step.parts.Cfree * (step.parts.Ar \ br)
end

"""
    semquad_newton_steady(sq, qt, ss, pr, theta0; dt=1e-2, tol=1e-8, maxit=10,
                          krylov=40, restart=20, verbose=false)

Damped Newton-Krylov solve of `G(theta) = theta` with the analytic tangent JVP.
Returns `(theta, step, converged, iters, history)`.
"""
function semquad_newton_steady(sq::SemQuad, qt::Quadtree, ss::SemStokes,
                               pr::Params, theta0::Vector{Float64};
                               dt::Float64=1.0e-2, tol::Float64=1.0e-8,
                               maxit::Int=10, krylov::Int=40,
                               restart::Int=20, verbose::Bool=false,
                               advection::Symbol=:galerkin,
                               ev::Float64=0.0, ev_cmax::Float64=0.052,
                               ev_alpha::Int=2, ev_timestep::Int=10,
                               ev_tangent::Bool=true,
                               picard_steps::Int=0,
                               linear::Symbol=:direct,
                               mg_cycles::Int=1, mg_smoother::Symbol=:auto,
                               mg_coarse::Symbol=:auto,
                               mg_operator::Symbol=:matrixfree,
                               mg_vel_solver::Symbol=:vcycle,
                               mg_vel_tol::Float64=1.0e-1,
                               mg_vel_maxit::Int=10,
                               mg_pressure_mode::Symbol=:auto,
                               mg_pressure_scale::Float64=0.0,
                               newton_tol::Float64=1.0e-8,
                               newton_maxit::Int=500,
                               newton_restart::Int=100)
    theta = copy(theta0)
    N = sq.N
    for g in 1:N
        y = sq.coords[g][2]
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (theta[g] = 0.0)
    end
    step = semquad_steady_step(sq, qt, ss, pr, theta; dt=dt,
                               advection=advection, ev=ev,
                               ev_cmax=ev_cmax, ev_alpha=ev_alpha,
                               ev_timestep=ev_timestep,
                               linear=linear, mg_cycles=mg_cycles,
                               mg_smoother=mg_smoother,
                               mg_coarse=mg_coarse,
                               mg_operator=mg_operator,
                               mg_vel_solver=mg_vel_solver,
                               mg_vel_tol=mg_vel_tol,
                               mg_vel_maxit=mg_vel_maxit,
                               mg_pressure_mode=mg_pressure_mode,
                               mg_pressure_scale=mg_pressure_scale,
                               newton_tol=newton_tol,
                               newton_maxit=newton_maxit,
                               newton_restart=newton_restart)
    # Pre-Newton Picard: the Newton linearisation is only local; on a cold
    # high-resolution state the line search can otherwise be forced to alpha
    # ~1e-2 (ASPECT also runs up to ten Picard iterations before its Newton).
    for k in 1:picard_steps
        theta = step.theta
        step = semquad_steady_step(sq, qt, ss, pr, theta; dt=dt,
                                   advection=advection, ev=ev,
                                   ev_cmax=ev_cmax, ev_alpha=ev_alpha,
                                   ev_timestep=ev_timestep,
                                   linear=linear, mg_cycles=mg_cycles,
                                   mg_smoother=mg_smoother,
                                   mg_coarse=mg_coarse,
                               mg_operator=mg_operator,
                               mg_vel_solver=mg_vel_solver,
                               mg_vel_tol=mg_vel_tol,
                               mg_vel_maxit=mg_vel_maxit,
                                   mg_pressure_mode=mg_pressure_mode,
                                   mg_pressure_scale=mg_pressure_scale,
                                   newton_tol=newton_tol,
                                   newton_maxit=newton_maxit,
                                   newton_restart=newton_restart)
        verbose && @printf("  picard %2d: |F| = %.3e\n", k,
                           norm(step.theta - theta))
    end
    F = step.theta - theta
    hist = [norm(F)]
    converged = norm(F) <= tol
    it = 0
    while !converged && it < maxit
        it += 1
        nr = norm(F)
        Jop = v -> semquad_steady_jv(sq, qt, ss, pr, theta, v, step;
                                     dt=dt, advection=advection,
                                     ev_tangent=ev_tangent) - v
        dx, rel, nit = gmres_restarted(Jop, -F; tol=1.0e-8,
                                       maxit=krylov, restart=restart)
        alpha = 1.0
        accepted = false
        for _ in 1:12
            theta_t = theta .+ alpha .* dx
            step_t = semquad_steady_step(sq, qt, ss, pr, theta_t; dt=dt,
                                         advection=advection, ev=ev,
                                         ev_cmax=ev_cmax,
                                         ev_alpha=ev_alpha,
                                         ev_timestep=ev_timestep,
                                         linear=linear, mg_cycles=mg_cycles,
                                         mg_smoother=mg_smoother,
                                         mg_coarse=mg_coarse,
                               mg_operator=mg_operator,
                               mg_vel_solver=mg_vel_solver,
                               mg_vel_tol=mg_vel_tol,
                               mg_vel_maxit=mg_vel_maxit,
                                         mg_pressure_mode=mg_pressure_mode,
                                         mg_pressure_scale=mg_pressure_scale,
                                         newton_tol=newton_tol,
                                         newton_maxit=newton_maxit,
                                         newton_restart=newton_restart)
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
