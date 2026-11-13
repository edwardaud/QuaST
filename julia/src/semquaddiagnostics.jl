# Quadtree SEM diagnostics: wall Nusselt numbers, Vrms, mean temperature and the
# energy Nusselt number.  These are the acceptance quantities for the adaptive
# extreme-case runs, so they must match the uniform-mesh definitions in
# physics.jl on exactly representable fields.

"""
    semquad_nu_walls(sq, qt, theta, pr)

Wall Nusselt numbers from the one-sided GLL derivative at the top/bottom
leaves, integrated over x:

    Nu_top = 1 + H/2 - <dtheta/dy>_top,
    Nu_bot = 1 - H/2 - <dtheta/dy>_bot.
"""
function semquad_nu_walls(sq::SemQuad, qt::Quadtree, theta::Vector{Float64},
                          pr::Params)
    p = sq.p
    n = p + 1
    st = 0.0; sb = 0.0; wt = 0.0; wb = 0.0
    for lf in sq.leaves
        h = leaf_hy(sq, lf[1])
        hx = leaf_hx(sq, lf[1])
        y0 = lf[3] * h
        ids = sq.elem[lf]
        if y0 + h >= 1.0 - 1.0e-14
            for i in 1:n
                d = 0.0
                for j in 1:n
                    d += sq.D[n, j] * theta[ids[(j - 1) * n + i]]
                end
                d *= 2.0 / h
                w = 0.5 * hx * sq.w[i]
                st += w * d
                wt += w
            end
        end
        if y0 <= 1.0e-14
            for i in 1:n
                d = 0.0
                for j in 1:n
                    d += sq.D[1, j] * theta[ids[(j - 1) * n + i]]
                end
                d *= 2.0 / h
                w = 0.5 * hx * sq.w[i]
                sb += w * d
                wb += w
            end
        end
    end
    # Nu = -(dT/dy)_wall / ((T_bot - T_top)/1) with T = base + theta, so the
    # conductive reference flux is the base-state gradient at the wall (for the
    # Boussinesq base this is 1 +- H/2, so the old expressions are recovered).
    top_ref = -semquad_base_gradient(pr, 1.0)
    bot_ref = -semquad_base_gradient(pr, 0.0)
    return top_ref - st / wt, bot_ref - sb / wb
end

"""
    semquad_base_advection_integral(sq, qt, pr, ux, uz)

Volume integral `int u_z dTbar/dz`.  For a divergence-free velocity with zero
normal flow on all walls this is zero (integrate by parts:
`int u.grad(Tbar) = -int Tbar div u = 0`).  A nonzero value measures how much
the discrete velocity violates that identity; it is the source term of the mean
theta/energy budget and the leading candidate for the H=5/AMR closure error.
"""
function semquad_base_advection_integral(sq::SemQuad, qt::Quadtree, pr::Params,
                                         ux::Vector{Float64}, uz::Vector{Float64})
    p = sq.p
    n = p + 1
    fd = make_fedata(p, p - 1)
    s = 0.0
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        y0 = lf[3] * hy
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            uzq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                uzq += uz[ids[k]] * fd.phi_v[i, qx] * fd.phi_v[j, qy]
            end
            s += w * uzq * dTbar(y, pr.H)
        end
    end
    return s
end

"""
    semquad_theta2_divu_integral(sq, qt, theta, ux, uz)

Volume integral `int theta^2 div u`.  The conservative/skew advection conserves
the L2 norm only when `div u = 0`; this moment is the leading divergence-induced
growth term of the theta amplitude.  It is a diagnostic, not a gate on a
physical run.
"""
function semquad_theta2_divu_integral(sq::SemQuad, qt::Quadtree,
                                      theta::Vector{Float64},
                                      ux::Vector{Float64}, uz::Vector{Float64})
    p = sq.p
    n = p + 1
    fd = make_fedata(p, p - 1)
    s = 0.0
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            thq = 0.0; dux = 0.0; duz = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                thq += theta[ids[k]] * fd.phi_v[i, qx] * fd.phi_v[j, qy]
                dux += ux[ids[k]] * fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / hx)
                duz += uz[ids[k]] * fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hy)
            end
            s += w * thq * thq * (dux + duz)
        end
    end
    return s
end

"""
    semquad_max_divergence(sq, qt, ux, uz)

Maximum pointwise `|div u|` at the GLL nodes of the quadtree.  Q2-Q1 enforces
the divergence only weakly (`int q div u = 0` for the pressure test space), so
this is not expected to be zero; it is the diagnostic needed to judge whether
the conservative advection form `-C'` (which assumes `div u = 0`) is consistent
with the discrete velocity.
"""
function semquad_max_divergence(sq::SemQuad, qt::Quadtree,
                                ux::Vector{Float64}, uz::Vector{Float64})
    p = sq.p
    n = p + 1
    worst = 0.0
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        ids = sq.elem[lf]
        for j in 1:n, i in 1:n
            dux = 0.0; duz = 0.0
            for a in 1:n
                dux += sq.D[i, a] * ux[ids[(j - 1) * n + a]]
                duz += sq.D[j, a] * uz[ids[(a - 1) * n + i]]
            end
            dux *= 2.0 / hx
            duz *= 2.0 / hy
            worst = max(worst, abs(dux + duz))
        end
    end
    return worst
end

"""
    semquad_surface_velocity(sq, qt, ux, uz; wall=:top)

RMS and maximum of the velocity magnitude along a horizontal wall, integrated
with the wall-adjacent GLL rule (the same one `semquad_nu_walls` uses):

    u_surf^RMS = sqrt( int |u|^2 dx / int dx ),   u_surf^max = max |u|.

These are two of the diagnostics reported in Tosi et al. (2015), Table 2.
"""
function semquad_surface_velocity(sq::SemQuad, qt::Quadtree,
                                  ux::Vector{Float64}, uz::Vector{Float64};
                                  wall::Symbol=:top)
    wall in (:top, :bottom) ||
        throw(ArgumentError("wall must be :top or :bottom"))
    p = sq.p
    n = p + 1
    jF = wall === :top ? n : 1
    s2 = 0.0; wsum = 0.0; vmax = 0.0
    for lf in sq.leaves
        h = leaf_hy(sq, lf[1])
        hx = leaf_hx(sq, lf[1])
        y0 = lf[3] * h
        on_wall = wall === :top ? (y0 + h >= 1.0 - 1.0e-14) : (y0 <= 1.0e-14)
        on_wall || continue
        ids = sq.elem[lf]
        for i in 1:n
            w = 0.5 * hx * sq.w[i]
            g = ids[(jF - 1) * n + i]
            v = hypot(ux[g], uz[g])
            s2 += w * v * v
            wsum += w
            vmax = max(vmax, v)
        end
    end
    return sqrt(s2 / wsum), vmax
end

"""
    semquad_work_dissipation(sq, qt, pr, theta, ux, uz, eta_fun)

Volume integrals of the buoyancy work and the viscous dissipation,

    W = int rho_bar T u_z dV,    T = Tbar + theta
    U = int 2 eta edot:edot dV,

both in the nondimensional units of route C (Ra multiplies the buoyancy, so the
mechanical balance of the continuous problem is `Ra * W = U`).  `eta_fun(x,y)`
must return the viscosity used by the Stokes solve at that point.

These are the `W` and `U` entries of Tosi et al. (2015), Table 2, where the
dissipation is tabulated as `U/Ra` so that the two numbers are comparable.
"""
function semquad_work_dissipation(sq::SemQuad, qt::Quadtree, pr::Params,
                                  theta::Vector{Float64},
                                  ux::Vector{Float64}, uz::Vector{Float64},
                                  eta_fun::Function)
    p = sq.p
    n = p + 1
    fd = make_fedata(p, p - 1)
    W = 0.0
    U = 0.0
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            thq = 0.0; uzq = 0.0
            dux_dx = 0.0; dux_dy = 0.0; duz_dx = 0.0; duz_dy = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                thq += theta[ids[k]] * b
                uzq += uz[ids[k]] * b
                dux_dx += ux[ids[k]] * fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / hx)
                dux_dy += ux[ids[k]] * fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hy)
                duz_dx += uz[ids[k]] * fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / hx)
                duz_dy += uz[ids[k]] * fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hy)
            end
            # Full temperature T = Tbar(y) + theta (Tbar is the reference
            # adiabat when one is set), weighted by the reference density: the
            # ALA buoyancy work is int rho(y) T u_z dV, and with div(rho u)=0 the
            # adiabatic part integrates to zero, so this is also int rho theta
            # u_z.  Without a reference state both reduce to int T u_z.
            T = semquad_base_T(pr, y) + thq
            # `semquad_rho` reads a `Union{Nothing,Function}` field, so the
            # ternary is typed `Any` without the assertion (the else branch is
            # `rs.rho_fun(z)`); `W` then becomes `Any` and every quadrature
            # point boxes `W += ...` (measured: 5.04 MB per 5 blob steps,
            # `--track-allocation=user`, semquaddiagnostics.jl:244).
            rho_w = (pr.refstate === nothing ? 1.0 :
                     semquad_rho(pr.refstate, y))::Float64
            W += w * rho_w * T * uzq
            exx = dux_dx
            ezz = duz_dy
            exz = 0.5 * (dux_dy + duz_dx)
            eed = exx^2 + ezz^2 + 2.0 * exz^2       # edot:edot
            if pr.deviatoric
                # ASPECT's 3-D deviatoric projection used on a 2-D flow:
                # eps_d:eps_d = eps:eps - (1/3) div(u)^2
                eed -= (1.0 / 3.0) * (exx + ezz)^2
            end
            eta = eta_fun(x, y)
            U += w * 2.0 * eta * eed
        end
    end
    # volume means: identity on the unit square (area == 1 exactly) and keeps
    # the Tosi table comparison and Ra*W = U meaningful for Lx != 1
    A = total_area(qt)
    return W / A, U / A
end

"""Volume RMS velocity on the quadtree (Gauss quadrature, same as `vrms`)."""
function semquad_vrms(sq::SemQuad, qt::Quadtree, ux::Vector{Float64},
                      uz::Vector{Float64})
    p = sq.p
    n = p + 1
    fd = make_fedata(p, p - 1)
    s = 0.0
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            uxq = 0.0; uzq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                uxq += ux[ids[k]] * b
                uzq += uz[ids[k]] * b
            end
            s += w * (uxq * uxq + uzq * uzq)
        end
    end
    # volume mean (the domain area is exactly 1 on the unit square, so this is
    # bit identical there)
    return sqrt(s / total_area(qt))
end

"""Volume mean of the full temperature T = Tbar + theta (Gauss quadrature)."""
function semquad_mean_T(sq::SemQuad, qt::Quadtree, theta::Vector{Float64},
                        pr::Params)
    p = sq.p
    n = p + 1
    fd = make_fedata(p, p - 1)
    s = 0.0
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        y0 = lf[3] * hy
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            thq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                thq += theta[ids[k]] * fd.phi_v[i, qx] * fd.phi_v[j, qy]
            end
            s += w * (semquad_base_T(pr, y) + thq)
        end
    end
    return s / total_area(qt)
end

"""Energy Nusselt number 1 + H/2 + <u_z theta> on the quadtree."""
function semquad_nu_energy(sq::SemQuad, qt::Quadtree, pr::Params,
                           theta::Vector{Float64}, ux::Vector{Float64},
                           uz::Vector{Float64})
    p = sq.p
    n = p + 1
    fd = make_fedata(p, p - 1)
    s = 0.0
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            uzq = 0.0; thq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                uzq += uz[ids[k]] * b
                thq += theta[ids[k]] * b
            end
            s += w * uzq * thq
        end
    end
    return 1.0 + 0.5 * pr.H + s / total_area(qt)
end

"""
    semquad_heat_budget(sq, qt, pr, theta; phases=PhaseTransition[],
                        ux=nothing, uz=nothing, eta_fun=nothing)

Volume integrals of the terms of the latent-heat energy budget, on the same
quadrature the energy assembly uses (Gauss for the volumes, the wall GLL rule of
`semquad_nu_walls` for the fluxes).  With the phase function
`Gamma(y,T)` of `phase.jl` and
`h = theta + sum_i latent_i*(Gamma_i(y,T) - Gamma_i(y,Tbar))` (the sensible +
latent anomaly relative to the conductive base state; `latent_i` is
`pr.latent_L` in `:global` mode or `PhaseTransition.latent` in `:per_phase`
mode, docs/dev/PHYSICS.md §7.8), the semi-discrete energy equation integrated over
the box is

    d/dt int h = Q + S,        Q = int_wall dtheta/dn ds,

so a finite difference of `enthalpy` between two states must equal the time
integral of `flux + heating` (see `t_g194_latent_heat.jl`).

Returns a named tuple with

| field | meaning |
|---|---|
| `theta_int` | `int theta dV` (sensible heat; `Tbar` is steady so its integral is a constant) |
| `latent_int` | `int sum_i latent_i*(Gamma_i(y,T) - Gamma_i(y,Tbar)) dV` (latent enthalpy anomaly) |
| `enthalpy` | `theta_int + latent_int` |
| `flux_top`, `flux_bot` | `int_wall dtheta/dy dx` on each wall |
| `flux` | `flux_top - flux_bot` = `int_wall dtheta/dn ds`, the net outward normal derivative (the physical heat flux is its negative, `-k dtheta/dn`) |
| `heating` | `int S dV` from `rho_heating`/adiabatic/dissipation (`0` when those are off) |
| `base_adv` | `int u_z dTbar/dz dV` (zero for a divergence-free discrete velocity; the leading budget error term) |
| `latent_src` | `int sum_i latent_i*dGamma_i/dz u_z dV`, the volume integral of the latent source `-latent_i*dGamma_i/dz*u_z` of the assembled equation (sign flipped: this is the term on the flux side) |
| `latent_adv` | `int sum_i latent_i*dGamma_i/dT (u.grad theta) dV`, the apparent-capacity correction to the advective transport |
| `latent_dgdt` | `latent_src + latent_adv` = `int sum_i latent_i DGamma_i/Dt dV` at a steady state (it is the latent contribution to the steady flux balance) |
| `u_grad_theta` | `int u.grad theta dV` (zero for a divergence-free discrete velocity) |

`ux, uz` are needed for `base_adv`, for the adiabatic/dissipation heating and for
the latent terms of the flux balance (`latent_src`/`latent_adv`/`latent_dgdt`/
`u_grad_theta`, which are `nothing` when the velocity is not supplied); they are
zero when the latent term is inactive.  `eta_fun` is only needed for the
dissipation heating, and throws when it is missing while
`heating_dissipation` is on.

The steady energy balance the closure diagnostic uses is, per unit area,

    Nut - Nub = H - (base_adv + latent_src + latent_adv)/A,

so `latent_dgdt/A` is the latent contribution to the flux balance (docs/
RHEOLOGY.md §7.3, `docs/dev/CASE_REALISTIC.md` §7.2).  It is **not**
`latent_int`: that is an enthalpy, and its time derivative is the transient
term, not a steady source.
"""
function semquad_heat_budget(sq::SemQuad, qt::Quadtree, pr::Params,
                             theta::Vector{Float64};
                             phases::Vector{PhaseTransition}=PhaseTransition[],
                             ux::Union{Nothing,Vector{Float64}}=nothing,
                             uz::Union{Nothing,Vector{Float64}}=nothing,
                             eta_fun::Union{Nothing,Function}=nothing)
    p = sq.p
    n = p + 1
    fd = make_fedata(p, p - 1)
    L = pr.latent_L
    use_lat = semquad_latent_active(pr, phases)
    per_phase = pr.latent_mode === :per_phase
    lcoef = use_lat ? semquad_latent_coefficients(pr, phases) : Float64[]
    theta_int = 0.0
    latent_int = 0.0
    latent_src = 0.0
    latent_adv = 0.0
    u_grad_theta = 0.0
    heating = 0.0
    base_adv = 0.0
    # The latent terms of the steady flux balance need the velocity; the
    # enthalpy integrals do not.  When `ux`/`uz` are not supplied the three
    # velocity-dependent latent fields are returned as `nothing` (a loud
    # absence rather than a silent zero) so every pre-existing caller that only
    # wants `latent_int` keeps working unchanged.
    latent_bal = use_lat && ux !== nothing && uz !== nothing
    need_u = pr.heating_adiabatic || (uz !== nothing && ux !== nothing)
    need_u && (ux === nothing || uz === nothing) &&
        throw(ArgumentError("semquad_heat_budget needs ux and uz for the " *
                            "adiabatic heating / base advection"))
    if pr.heating_dissipation && eta_fun === nothing
        throw(ArgumentError("semquad_heat_budget needs eta_fun when " *
                            "heating_dissipation is on"))
    end
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            thq = 0.0; uzq = 0.0; uxq = 0.0
            dth_dx = 0.0; dth_dy = 0.0
            dux_dx = 0.0; dux_dy = 0.0; duz_dx = 0.0; duz_dy = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                thq += theta[ids[k]] * b
                if need_u
                    uzq += uz[ids[k]] * b
                end
                if latent_bal
                    uxq += ux[ids[k]] * b
                    dth_dx += theta[ids[k]] * fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / hx)
                    dth_dy += theta[ids[k]] * fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hy)
                end
                if pr.heating_dissipation
                    dux_dx += ux[ids[k]] * fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / hx)
                    dux_dy += ux[ids[k]] * fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hy)
                    duz_dx += uz[ids[k]] * fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / hx)
                    duz_dy += uz[ids[k]] * fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hy)
                end
            end
            theta_int += w * thq
            T = semquad_base_T(pr, y) + thq
            if use_lat
                # `sum_i latent_i (Gamma_i - Gamma_i(Tbar))`: the same per-
                # transition coefficients the energy operator uses (§7.8), so
                # the budget closes for `:per_phase` as it does for `:global`.
                # The same loop accumulates the two terms of the *steady*
                # balance: the latent source `latent_i Gamma_z,i u_z` and the
                # apparent-capacity advective correction
                # `latent_i Gamma_T,i u.grad(theta)` (§7.3).
                g = 0.0
                gz_u = 0.0
                gt_ug = 0.0
                ugrad = latent_bal ? (uxq * dth_dx + uzq * dth_dy) : 0.0
                if per_phase
                    for i in eachindex(phases)
                        pt = phases[i]
                        lc = lcoef[i]
                        Tb = semquad_base_T(pr, y)
                        g += lc * (phase_gamma(pt, y, T) - phase_gamma(pt, y, Tb))
                        if latent_bal
                            gz_u += lc * phase_dgamma_dz(pt, y, T) * uzq
                            gt_ug += lc * phase_dgamma_dT(pt, y, T) * ugrad
                        end
                    end
                    latent_int += w * g
                else
                    for pt in phases
                        g += phase_gamma(pt, y, T) -
                             phase_gamma(pt, y, semquad_base_T(pr, y))
                        if latent_bal
                            gz_u += L * phase_dgamma_dz(pt, y, T) * uzq
                            gt_ug += L * phase_dgamma_dT(pt, y, T) * ugrad
                        end
                    end
                    latent_int += w * L * g
                end
                if latent_bal
                    latent_src += w * gz_u
                    latent_adv += w * gt_ug
                end
            end
            if pr.rho_heating
                heating += w * (-T * pr.H)
            end
            if pr.Di > 0.0 && (pr.heating_adiabatic || pr.heating_dissipation)
                if pr.heating_adiabatic
                    rho_ad = (pr.refstate === nothing ? 1.0 :
                              semquad_rho(pr.refstate, y))::Float64
                    heating += w * (-pr.Di * rho_ad * T * uzq)
                end
                if pr.heating_dissipation
                    exz = 0.5 * (dux_dy + duz_dx)
                    eed = dux_dx^2 + duz_dy^2 + 2.0 * exz^2
                    if pr.deviatoric
                        eed -= (1.0 / 3.0) * (dux_dx + duz_dy)^2
                    end
                    heating += w * (pr.Di / pr.Ra) * 2.0 * eta_fun(x, y) * eed
                end
            end
            if need_u
                base_adv += w * uzq * semquad_base_gradient(pr, y)
            end
            use_lat && (u_grad_theta += w * ugrad)
        end
    end
    # wall fluxes, the same one-sided GLL derivative rule as semquad_nu_walls
    st = 0.0; sb = 0.0
    for lf in sq.leaves
        h = leaf_hy(sq, lf[1])
        hx = leaf_hx(sq, lf[1])
        y0 = lf[3] * h
        ids = sq.elem[lf]
        if y0 + h >= 1.0 - 1.0e-14
            for i in 1:n
                d = 0.0
                for j in 1:n
                    d += sq.D[n, j] * theta[ids[(j - 1) * n + i]]
                end
                st += 0.5 * hx * sq.w[i] * d * (2.0 / h)
            end
        end
        if y0 <= 1.0e-14
            for i in 1:n
                d = 0.0
                for j in 1:n
                    d += sq.D[1, j] * theta[ids[(j - 1) * n + i]]
                end
                sb += 0.5 * hx * sq.w[i] * d * (2.0 / h)
            end
        end
    end
    return (theta_int=theta_int, latent_int=latent_int,
            enthalpy=theta_int + latent_int,
            flux_top=st, flux_bot=sb, flux=st - sb,
            heating=heating, base_adv=base_adv,
            latent_src=(use_lat ? (latent_bal ? latent_src : nothing) : 0.0),
            latent_adv=(use_lat ? (latent_bal ? latent_adv : nothing) : 0.0),
            latent_dgdt=(use_lat ? (latent_bal ? latent_src + latent_adv : nothing) : 0.0),
            u_grad_theta=(use_lat ? (latent_bal ? u_grad_theta : nothing) : 0.0))
end

"""
    semquad_exact_residual(sq, qt, pr, theta, theta_n, theta_nm1, ux, uz,
                           dt, dt_prev, firststep, ev_field; advection=:skew,
                           phases=PhaseTransition[], latent_theta=nothing,
                           eta_fun=nothing)

The **exact** residual of the quoted pointwise Nusselt pair against the scheme's
own discrete identity (docs/dev/VALIDATION.md §2.3, eq. `(ID1)`):

    R = F_pt - (dH/dt + sum(C theta) + Ibase + latent_src)
    F_pt = flux_top - flux_bot,   dH/dt = 1'(ca M theta - cb M theta_n + cc M theta_nm1)

where `M, K, C` are rebuilt with the *same* operator the step used (the stored
elementwise EV field and the same apparent-capacity temperature) and
`ca/cb/cc` are the BDF2 coefficients of that step.  `R/A` is in Nusselt units.
It vanishes (to the linear solver's floor) for the time-step CBF and is `O(h)`
for the pointwise pair, so a non-zero `R/A` at a transient state is a property
of the pointwise family, **not** an error of the run
(docs/dev/VALIDATION.md §3.2); the historical `closure` column is a different
quantity and does not close for any definition (NUSSELT_AUDIT §4).

`theta_n`/`theta_nm1` are the two previous BDF2 levels of the step that produced
`theta`; `ev_field` is the elementwise EV field of that step (`nothing` when
`ev == 0`), so the operator is the one that produced the state and no EV field
is recomputed.  `latent_theta` is the temperature at which the step evaluated
the apparent capacity (`nothing` means `theta_n`, the lagged/default choice);
pass it explicitly when `maxpic > 1`, where it is the last iterate.

Returns a named tuple with `R`, `R_over_A = R / total_area(qt)` and the terms of
the identity (`flux`, `dHdt`, `sumC`, `Ibase`, `latent_src`, `rhs`).  This is a
pure diagnostic: no solver path calls it unless the caller asks for it (the
driver's opt-in `extdiag=true`, docs/dev/CASE_REALISTIC.md §7.2).
"""
function semquad_exact_residual(sq::SemQuad, qt::Quadtree, pr::Params,
                                theta::Vector{Float64},
                                theta_n::Vector{Float64},
                                theta_nm1::Vector{Float64},
                                ux::Vector{Float64}, uz::Vector{Float64},
                                dt::Float64, dt_prev::Float64,
                                firststep::Bool,
                                ev_field::Union{Nothing,Vector{Float64}};
                                advection::Symbol=:skew,
                                phases::Vector{PhaseTransition}=PhaseTransition[],
                                latent_theta::Union{Nothing,Vector{Float64}}=nothing,
                                eta_fun::Union{Nothing,Function}=nothing)
    rho_ref = (pr.rho_ref_weight && pr.refstate !== nothing) ?
              (y -> semquad_rho(pr.refstate, y) * pr.refstate.cp) : nothing
    M, K, C, F = assemble_semquad_energy(sq, qt, pr, ux, uz;
                                         advection=advection,
                                         theta=theta_n,
                                         theta_lat=latent_theta,
                                         phases=phases,
                                         rho_heating=pr.rho_heating,
                                         rho_cp=pr.rho_cp,
                                         heating_adiabatic=pr.heating_adiabatic,
                                         heating_dissipation=pr.heating_dissipation,
                                         eta_fun=eta_fun, rho_ref=rho_ref,
                                         ev=0.0, ev_field=ev_field)
    # The BDF2 mass coefficients of the step that produced `theta` (the first
    # step is backward Euler, cc == 0).
    if firststep
        ca = 1.0 / dt; cb = 1.0 / dt; cc = 0.0
    else
        w = dt / dt_prev
        ca = (2.0 * w + 1.0) / (1.0 + w) / dt
        cb = (1.0 + w) / dt
        cc = w * w / (1.0 + w) / dt
    end
    dHdt = sum(M * (ca .* theta .- cb .* theta_n .+ cc .* theta_nm1))
    sumC = sum(C * theta)
    bud = semquad_heat_budget(sq, qt, pr, theta; phases=phases, ux=ux, uz=uz,
                              eta_fun=eta_fun)
    Ibase = semquad_base_advection_integral(sq, qt, pr, ux, uz)
    latent_src = bud.latent_src === nothing ? 0.0 : bud.latent_src
    rhs = dHdt + sumC + Ibase + latent_src
    area = total_area(qt)
    R = bud.flux - rhs
    return (R=R, R_over_A=R / area, flux=bud.flux, dHdt=dHdt, sumC=sumC,
            Ibase=Ibase, latent_src=latent_src, rhs=rhs, area=area)
end

"""
    semquad_level_stats(qt; strip=0.1)

Leaf-level histogram plus the finest level in the wall strips (`y <= strip` or
`y >= 1-strip`) and in the interior.  This is the cheap "is the AMR actually
tracking the boundary layer?" diagnostic: for a wall-refined tree
`wall_max > interior_max`, for a uniform tree they are equal.
"""
function semquad_level_stats(qt::Quadtree; strip::Float64=0.1)
    hist = Dict{Int,Int}()
    wall_max = 0
    interior_max = 0
    for lf in qt.leaves
        l, ix, iy = lf
        h = leaf_hy(qt, l)
        y0 = iy * h
        hist[l] = get(hist, l, 0) + 1
        if y0 + h <= strip || y0 >= 1.0 - strip
            wall_max = max(wall_max, l)
        else
            interior_max = max(interior_max, l)
        end
    end
    return (histogram=hist, wall_max=wall_max, interior_max=interior_max,
            n=length(qt.leaves))
end

