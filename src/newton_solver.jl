# Newton-Krylov-free (dense LU) Stokes solver with viscosity continuation and
# smooth log-space caps, used by run_simulation(nonlinear=:newton).

function _pack_state(st, dm::DofMap, m::Mesh)
    nv = m.Nscalar
    ux = zeros(dm.nux); uz = zeros(dm.nuz); pp = zeros(dm.npres)
    for g in 1:nv
        a = dm.red_ux[g]; a > 0 && (ux[a] = st.ux[g])
        b = dm.red_uz[g]; b > 0 && (uz[b] = st.uz[g])
    end
    for g in 1:m.Npres
        a = dm.red_p[g]; a > 0 && (pp[a] = st.p[g])
    end
    return vcat(ux, uz, pp)
end

function _unpack_state(x, dm::DofMap, m::Mesh)
    nv = m.Nscalar
    ux = zeros(nv); uz = zeros(nv); pp = zeros(m.Npres)
    for g in 1:nv
        a = dm.red_ux[g]; a > 0 && (ux[g] = x[a])
        b = dm.red_uz[g]; b > 0 && (uz[g] = x[dm.nux + b])
    end
    for g in 1:m.Npres
        a = dm.red_p[g]; a > 0 && (pp[g] = x[dm.nux + dm.nuz + a])
    end
    return ux, uz, pp
end

"""
    solve_stokes_newton(m, fd, pr, theta, dm; rheology, refstate, st0)

Newton solve of the nonlinear Stokes system with a five-stage viscosity
continuation and smooth log-space caps.  `st0` (optional) provides the initial
guess; returns a `StokesSolution`.
"""
function solve_stokes_newton(m::Mesh, fd::FEData, pr::Params, theta::Vector{Float64},
                             dm::DofMap; rheology::RheologyModel,
                             refstate=nothing, st0=nothing, maxit::Int=6,
                             verbose::Bool=false)
    nvel = dm.nux + dm.nuz
    np = dm.npres
    n = nvel + np
    if st0 === nothing
        s0 = solve_stokes(m, fd, pr, theta, dm; refstate=refstate,
                          velocity_space=:taylor_hood)
        x = vcat(s0.ux[dm.red_ux .> 0], s0.uz[dm.red_uz .> 0], zeros(np))
    else
        x = _pack_state(st0, dm, m)
    end
    function residual(xx, rh)
        ux, uz, _ = _unpack_state(xx, dm, m)
        s = assemble_stokes(m, fd, pr, theta, dm; refstate=refstate, rheology=rh,
                            u_prev=(ux, uz), smooth_caps=true)
        return s.A * xx - s.f
    end
    function jacobian(xx, rh)
        ux, uz, _ = _unpack_state(xx, dm, m)
        return assemble_stokes(m, fd, pr, theta, dm; refstate=refstate, rheology=rh,
                               u_prev=(ux, uz), newton=true).A
    end
    stages = [(1e20, min(1e24, rheology.etamax)), (1e18, min(1e26, rheology.etamax)),
              (1e17, min(1e27, rheology.etamax)), (rheology.etamin, rheology.etamax)]
    F = residual(x, rheology)
    for (lo, hi) in stages
        rh = RheologyModel(rheology.creep, rheology.yield, rheology.ref_eps,
                           rheology.min_eps, lo, hi, rheology.Tsurf, rheology.dT,
                           rheology.p0, rheology.eps_scale, rheology.eta0)
        F = residual(x, rh)
        for it in 1:maxit
            J = jacobian(x, rh)
            accepted = false
            base = norm(J) / n
            for lam in (0.0, 1e-4, 1e-2, 1.0, 1e2, 1e4) .* base
                Jd = lam == 0.0 ? J : (J + lam * I)
                dx = -(Jd \ F)
                alpha = 1.0
                for ls in 1:16
                    xn = x .+ alpha .* dx
                    Fn = residual(xn, rh)
                    if norm(Fn) < norm(F)
                        x = xn; F = Fn; accepted = true
                        break
                    end
                    alpha *= 0.5
                end
                accepted && break
            end
            accepted || break
            norm(F) < 1e-8 && break
        end
        verbose && @info "newton stage" lo hi norm(F)
    end
    ux, uz, pp = _unpack_state(x, dm, m)
    sf = assemble_stokes(m, fd, pr, theta, dm; refstate=refstate, rheology=rheology,
                         u_prev=(ux, uz), smooth_caps=true)
    return StokesSolution(ux, uz, pp, norm(F), sf.eta_min, sf.eta_max, sf.nclip, nothing)
end



