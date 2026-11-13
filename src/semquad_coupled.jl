# Coupled (u, p, theta) steady Newton for the route C semi-quad solver.
#
# The steady state of the backward-Euler map is the root of
#
#     r_S = fr - Kred(th) uhat - Bt phat                       (momentum)
#     r_C = gr - Bs uhat                                    (incompressibility)
#     r_E = Cfree' ( K(th) th + C(u) th - F(u) )            (energy)
#
# (the `K, C, F, M` of `assemble_semquad_energy`, which is where the entropy
# viscosity enters).  Differentiating the energy row gives the two couplings
#
#     d r_E / d u    = Cfree' ( C(du) th - F(du) )      (velocity -> energy)
#     d r_E / d th   = (1/dt) Cfree' M Cfree
#                      + Cfree' (K + C(u)) Cfree
#                      + Cfree' Cu' K_T Cu Cfree      (viscosity -> energy)
#                      - D_ev(d nu) th                (EV tangent, G166)
#
# and the Stokes rows couple to theta through the buoyancy and `K(th)`:
#
#     d r_S / d th = Cu' ( Ra Mv dth - K_T dth ) .
#
# The earlier "coupled Newton is ill-conditioned" result was a sign error in
# `d r_E / d u` (see ROUTE_C_STATUS §1.4); with the sign above the method
# converges quadratically.  This file is the production version of
# `examples/coupled_newton_proto.jl`, with
#   * EV support: the residual uses the EV-modified operators of
#     `semquad_steady_parts`, and the Jacobian gets the `-D_ev(d nu) th` term
#     (G166) when `ev_tangent=true`;
#   * no duplicate assembly in the preconditioner: only the energy row of the
#     Stokes coupling is evaluated (and only when `precond_coupling=true`);
#   * a switchable Stokes block (`stokes_block=:direct|:mg|:auto`) for the
#     preconditioner and the Jacobian.

"""
    semquad_coupled_newton(sq, qt, ss, pr, theta0; dt, ev, ...)

Damped Newton-Krylov solve of the coupled steady system `(r_S, r_C, r_E) = 0`
for `(u, p, theta)`, with block-scaled residual, a line search on the scaled
norm and a block-triangular preconditioner (Stokes saddle + energy operator with
the `d r_E / d u` coupling).  Returns
`(theta, u, p, converged, iters, history)`.

Keywords: `dt` (backward-Euler step of the map whose fixed point is the steady
state), `ev`/`ev_cmax`/`ev_alpha`/`ev_timestep` (entropy viscosity),
`ev_tangent` (include the EV field derivative in the energy block, G166),
`advection`, `tol`/`maxit`, `krylov`/`restart` (outer GMRES),
`stokes_block` (`:direct`, `:mg` or `:auto` -> `:mg` for `nf > 5000`),
`precond_coupling` (include the energy<->Stokes coupling in the preconditioner).
"""
function semquad_coupled_newton(sq::SemQuad, qt::Quadtree, ss::SemStokes,
                                pr::Params, theta0::Vector{Float64};
                                dt::Float64=1.0e-2,
                                ev::Float64=0.0, ev_cmax::Float64=0.052,
                                ev_alpha::Int=2, ev_timestep::Int=10,
                                ev_tangent::Bool=false,
                                ev_tangent_damp::Float64=1.0,
                                ev_tangent_ramp::Float64=0.0,
                                advection::Symbol=:galerkin,
                                tol::Float64=1.0e-8, maxit::Int=15,
                                krylov::Int=300, restart::Int=100,
                                stokes_block::Symbol=:auto,
                                precond_coupling::Bool=true,
                                verbose::Bool=false)
    N = sq.N
    leaves = Set(qt.leaves)
    z2 = (x, y) -> 0.0
    th = copy(theta0)
    for g in 1:N
        y = sq.coords[g][2]
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (th[g] = 0.0)
    end
    sys_at(thv) = begin
        eta = (x, y) -> semquad_eta_at(sq, thv, x, y, pr, leaves)
        fz = (x, y) -> pr.Ra * QuaST._semquad_eval_at(sq, thv, x, y, leaves)
        solve_semquad_stokes_var(ss, qt, eta, z2, fz, z2, z2; bc=pr.bc,
                                 linear=:direct, deviatoric=pr.deviatoric,
                                 assemble_only=true)
    end
    parts_at(thv, u) = semquad_steady_parts(sq, qt, pr, thv, u; dt=dt,
                                            advection=advection, ev=ev,
                                            ev_cmax=ev_cmax, ev_alpha=ev_alpha,
                                            ev_timestep=ev_timestep)
    function resid(thv, u, p)
        sys = sys_at(thv)
        parts = parts_at(thv, u)
        uh = sys.Cu' * (u - sys.u0)
        ph = sys.Cp' * (p - sys.p0)
        rS = sys.fr - sys.Kred * uh - sys.Bt * ph
        rC = sys.gr - sys.Bs * uh
        rE = parts.Cfree' * (parts.K * thv + parts.C * thv - parts.F)
        return vcat(rS, rC, rE), sys, parts
    end
    # --- base step gives the geometry, the projection and the scales ---------
    base = semquad_steady_step(sq, qt, ss, pr, theta0; dt=dt, ev=ev,
                               ev_cmax=ev_cmax, ev_alpha=ev_alpha,
                               ev_timestep=ev_timestep, advection=advection)
    nf, nfp = base.nf, base.nfp
    u = copy(base.u)
    p = copy(base.p)
    nt = size(base.parts.Cfree, 2)
    Mv = base.Mv
    wS = max(norm(base.sys.fr), 1.0e-30)
    wC = max(norm(base.sys.Bs) * norm(base.sys.Cu' * u), 1.0e-30)
    wE = max(norm(base.parts.Cfree' * base.parts.F), 1.0e-30)
    scaled(r) = sqrt((norm(r[1:nf]) / wS)^2 +
                     (norm(r[(nf + 1):(nf + nfp)]) / wC)^2 +
                     (norm(r[(nf + nfp + 1):end]) / wE)^2)
    sb = stokes_block === :auto ? (nf > 5000 ? :mg : :direct) : stokes_block
    # energy->Stokes coupling needs the theta-perturbed stiffness applied to u;
    # built once per Newton step from the base system.
    it = 0
    converged = false
    hist = Float64[]
    last_jac = nothing
    last_sys = nothing
    last_parts = nothing
    while it < maxit
        it += 1
        rc, sysc, partsc = resid(th, u, p)
        last_sys = sysc
        last_parts = partsc
        nr = scaled(rc)
        # ASPECT-style derivative ramp: switch the EV derivative on only once
        # the residual has dropped below `ev_tangent_ramp * |r_0|`.
        damp_ev = ev_tangent_damp
        if ev_tangent_ramp > 0.0 && ev > 0.0
            rref = isempty(hist) ? nr : hist[1]
            damp_ev *= max(0.0, 1.0 - nr / (ev_tangent_ramp * rref))
        end
        Cu, Cp, Cfree = sysc.Cu, sysc.Cp, partsc.Cfree
        Asteady = partsc.Ar - (1.0 / dt) * (Cfree' * partsc.M * Cfree)
        eta_c = (x, y) -> semquad_eta_at(sq, th, x, y, pr, leaves)
        solver = QuaST._semquad_newton_saddle(ss, qt, ss.p, pr.bc,
                                                  pr.deviatoric, eta_c, sysc;
                                                  linear=sb)
        # energy row of a Stokes direction only: one energy assembly, no wasted
        # Stokes evaluation (this is what the prototype recomputed in full).
        function energy_coupling(du)
            _, _, Cdu, Fdu = assemble_semquad_energy(
                sq, qt, pr, du[1:N], du[(N + 1):(2N)];
                advection=advection, theta=th, ev=ev, ev_cmax=ev_cmax,
                ev_alpha=ev_alpha, ev_timestep=ev_timestep,
                ev_field=partsc.ev_field)
            return Cfree' * (Cdu * th - Fdu)
        end
        function jac(x)
            xu = x[1:nf]
            xp = x[(nf + 1):(nf + nfp)]
            xty = x[(nf + nfp + 1):end]
            dth = Cfree * xty
            du = Cu * xu
            yS = -sysc.Kred * xu - sysc.Bt * xp
            dfull = vcat(zeros(Float64, N), pr.Ra * (Mv * dth))
            yS .+= sysc.Cu' * dfull
            yS .-= sysc.Cu' * QuaST._semquad_jvp_ku(sq, ss.q, pr, th, dth, u)
            yC = -sysc.Bs * xu
            # d r_E . (du, dtheta) = Cfree'( C(du) th - F(du) )
            #                        + Cfree'(K + C(u)) Cfree dtheta
            _, _, Cdu, Fdu = assemble_semquad_energy(
                sq, qt, pr, du[1:N], du[(N + 1):(2N)];
                advection=advection, theta=th, ev=ev, ev_cmax=ev_cmax,
                ev_alpha=ev_alpha, ev_timestep=ev_timestep,
                ev_field=partsc.ev_field)
            yE = Cfree' * (Cdu * th - Fdu) + Asteady * xty
            if ev > 0.0 && ev_tangent && damp_ev != 0.0
                dev = QuaST._semquad_ev_directional(sq, qt, pr, th, dth, u,
                                                        du, base)
                yE .-= damp_ev .* (Cfree' *
                      QuaST._semquad_ev_diffusion(sq, ss.q, th, dev))
            end
            return vcat(yS, yC, yE)
        end
        last_jac = jac
        function precond(b)
            z = zeros(Float64, nf + nfp + nt)
            sol = solver.solve(vcat(-b[1:nf], -b[(nf + 1):(nf + nfp)]))
            z[1:nf] = sol[1:nf]
            z[(nf + 1):(nf + nfp)] = sol[(nf + 1):end]
            rhsE = b[(nf + nfp + 1):end]
            if precond_coupling
                du = Cu * z[1:nf]
                rhsE = rhsE .- energy_coupling(du)
            end
            z[(nf + nfp + 1):end] = partsc.Ar \ rhsE
            return z
        end
        push!(hist, nr)
        dx, rel, nit = QuaST.gmres_restarted(jac, -rc; precond=precond,
                                                 tol=1.0e-10, maxit=krylov,
                                                 restart=restart)
        alpha = 1.0
        accepted = false
        for _ in 1:12
            ut = u .+ alpha .* (Cu * dx[1:nf])
            pt = p .+ alpha .* (Cp * dx[(nf + 1):(nf + nfp)])
            tht = th .+ alpha .* (Cfree * dx[(nf + nfp + 1):end])
            rt, _, _ = resid(tht, ut, pt)
            if scaled(rt) < (1.0 - 1.0e-4 * alpha) * nr
                th, u, p = tht, ut, pt
                accepted = true
                break
            end
            alpha *= 0.5
        end
        verbose && @printf("  coupled %2d: scaled=%.3e alpha=%.3g gmres=%d rel=%.1e\n",
                           it, nr, alpha, nit, rel)
        accepted || break
        converged = scaled(resid(th, u, p)[1]) <= tol
        converged && break
    end
    return (theta=th, u=u, p=p, converged=converged, iters=it, history=hist,
            nf=nf, nfp=nfp, nt=nt, stokes_block=sb, jac=last_jac, resid=resid,
            scaled=scaled, sys=last_sys, parts=last_parts)
end
