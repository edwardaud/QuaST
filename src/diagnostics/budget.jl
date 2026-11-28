# Energy budget diagnostics: work/dissipation, heat budget with the
# latent/adiabatic terms (box + mapped twins).
"""
    semquad_work_dissipation(sq, qt, pr, theta, ux, uz, eta_fun)

Volume integrals of the buoyancy work and the viscous dissipation,

    W = int rho_bar T u_z dV,    T = Tbar + theta
    U = int 2 eta edot:edot dV,

both in the nondimensional units of this solver (Ra multiplies the buoyancy, so the
mechanical balance of the continuous problem is `Ra * W = U`).  `eta_fun(x,y)`
must return the viscosity used by the Stokes solve at that point.
`rho_multiplier` is the density hook (`Params.rho_multiplier`): when given, the
density of `W` is `rho_bar(x,y) = rho(y) * rho_multiplier(x,y)`, the same
density the buoyancy load uses, so `Ra*W = U` stays the balance of the solve;
`nothing` (default) leaves the integral bit-identical.

These are the `W` and `U` entries of Tosi et al. (2015), Table 2, where the
dissipation is tabulated as `U/Ra` so that the two numbers are comparable.
"""
function semquad_work_dissipation(sq::SemQuad, qt::Quadtree, pr::Params,
                                  theta::Vector{Float64},
                                  ux::Vector{Float64}, uz::Vector{Float64},
                                  eta_fun::Function;
                                  rho_multiplier::Union{Nothing,Function}=nothing)
    g = sq.geom
    g isa CartesianBox ||
        return _semquad_work_dissipation_mapped(g, sq, qt, pr, theta, ux, uz,
                                                eta_fun;
                                                rho_multiplier=rho_multiplier)
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
            # Density hook: the buoyancy work must use the density the load used
            # (a uniform multiplier c then gives W -> c*W, keeping Ra*W = U).
            rho_multiplier === nothing || (rho_w *= rho_multiplier(x, y))
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
mode), the semi-discrete energy equation integrated over
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
RHEOLOGY.md §7.3).  It is **not**
`latent_int`: that is an enthalpy, and its time derivative is the transient
term, not a steady source.
"""
function semquad_heat_budget(sq::SemQuad, qt::Quadtree, pr::Params,
                             theta::Vector{Float64};
                             phases::Vector{PhaseTransition}=PhaseTransition[],
                             ux::Union{Nothing,Vector{Float64}}=nothing,
                             uz::Union{Nothing,Vector{Float64}}=nothing,
                             eta_fun::Union{Nothing,Function}=nothing)
    g = sq.geom
    g isa CartesianBox ||
        return _semquad_heat_budget_mapped(g, sq, qt, pr, theta; phases=phases,
                                           ux=ux, uz=uz, eta_fun=eta_fun)
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

function _semquad_work_dissipation_mapped(g::AbstractGeometry, sq::SemQuad,
                                          qt::Quadtree, pr::Params,
                                          theta::Vector{Float64},
                                          ux::Vector{Float64},
                                          uz::Vector{Float64},
                                          eta_fun::Function;
                                          rho_multiplier::Union{Nothing,Function}=nothing)
    p = sq.p
    n = p + 1
    fd = make_fedata(p, p - 1)
    W = 0.0
    U = 0.0
    for lf in sq.leaves
        m = leaf_metric(g, sq, lf, fd)
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = quad_measure(m, fd, qx, qy)
            x, y = quad_point(m, sq, lf, fd, qx, qy)
            h = height(g, x, y)
            thq = 0.0
            uxq = 0.0
            uzq = 0.0
            dux_dx = 0.0
            dux_dy = 0.0
            duz_dx = 0.0
            duz_dy = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                thq += theta[ids[k]] * b
                uxq += ux[ids[k]] * b
                uzq += uz[ids[k]] * b
                gx, gz = basis_grad(m, fd, qx, qy, i, j)
                dux_dx += ux[ids[k]] * gx
                dux_dy += ux[ids[k]] * gz
                duz_dx += uz[ids[k]] * gx
                duz_dy += uz[ids[k]] * gz
            end
            # The buoyancy work is the work of the buoyancy force along the
            # velocity, i.e. `rho T (u . grad h)` with `grad h` the upward
            # direction (GEOM_DESIGN §2.7/§6.4): the mapped `u_z`.
            # Stage 0.5b1: the geometry's conductive base state (radial log/H on
            # the annulus; the historical quadratic on the validation maps).
            T = semquad_base_T(pr, g, x, y) + thq
            rho_w = (pr.refstate === nothing ? 1.0 :
                     semquad_rho(pr.refstate, h))::Float64
            rho_multiplier === nothing || (rho_w *= rho_multiplier(x, y))
            ugh = upward_velocity(g, x, y, uxq, uzq)
            W += w * rho_w * T * ugh
            exx = dux_dx
            ezz = duz_dy
            exz = 0.5 * (dux_dy + duz_dx)
            eed = exx^2 + ezz^2 + 2.0 * exz^2
            if pr.deviatoric
                eed -= (1.0 / 3.0) * (exx + ezz)^2
            end
            eta = eta_fun(x, y)
            U += w * 2.0 * eta * eed
        end
    end
    A = total_measure(g, sq, fd)
    return W / A, U / A
end

function _semquad_heat_budget_mapped(g::AbstractGeometry, sq::SemQuad,
                                     qt::Quadtree, pr::Params,
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
        m = leaf_metric(g, sq, lf, fd)
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = quad_measure(m, fd, qx, qy)
            x, y = quad_point(m, sq, lf, fd, qx, qy)
            h = height(g, x, y)
            thq = 0.0
            uxq = 0.0
            uzq = 0.0
            dth_dx = 0.0
            dth_dy = 0.0
            dux_dx = 0.0
            dux_dy = 0.0
            duz_dx = 0.0
            duz_dy = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                thq += theta[ids[k]] * b
                if need_u
                    # both components: `upward_velocity` needs the full `u`.
                    uxq += ux[ids[k]] * b
                    uzq += uz[ids[k]] * b
                end
                if latent_bal
                    gx, gz = basis_grad(m, fd, qx, qy, i, j)
                    dth_dx += theta[ids[k]] * gx
                    dth_dy += theta[ids[k]] * gz
                end
                if pr.heating_dissipation
                    gx, gz = basis_grad(m, fd, qx, qy, i, j)
                    dux_dx += ux[ids[k]] * gx
                    dux_dy += ux[ids[k]] * gz
                    duz_dx += uz[ids[k]] * gx
                    duz_dy += uz[ids[k]] * gz
                end
            end
            # directional velocity of the mapped geometry (GEOM_DESIGN §6.4).
            ugh = need_u ? upward_velocity(g, x, y, uxq, uzq) : 0.0
            theta_int += w * thq
            # Stage 0.5b1: the geometry's conductive base state (radial log/H on
            # the annulus; the historical quadratic-in-`h` on the validation
            # maps), not the Cartesian profile evaluated at `h`.
            T = semquad_base_T(pr, g, x, y) + thq
            Tb = semquad_base_T(pr, g, x, y)
            if use_lat
                gg = 0.0
                gz_u = 0.0
                gt_ug = 0.0
                ugrad = latent_bal ? (uxq * dth_dx + uzq * dth_dy) : 0.0
                if per_phase
                    for i in eachindex(phases)
                        pt = phases[i]
                        lc = lcoef[i]
                        gg += lc * (phase_gamma(pt, h, T) - phase_gamma(pt, h, Tb))
                        if latent_bal
                            gz_u += lc * phase_dgamma_dz(pt, h, T) * ugh
                            gt_ug += lc * phase_dgamma_dT(pt, h, T) * ugrad
                        end
                    end
                    latent_int += w * gg
                else
                    for pt in phases
                        gg += phase_gamma(pt, h, T) - phase_gamma(pt, h, Tb)
                        if latent_bal
                            gz_u += L * phase_dgamma_dz(pt, h, T) * ugh
                            gt_ug += L * phase_dgamma_dT(pt, h, T) * ugrad
                        end
                    end
                    latent_int += w * L * gg
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
                              semquad_rho(pr.refstate, h))::Float64
                    heating += w * (-pr.Di * rho_ad * T * ugh)
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
                base_adv += w * ugh * semquad_base_gradient(pr, g, x, y)
            end
            use_lat && (u_grad_theta += w * ugrad)
        end
    end
    # Wall fluxes with the mapped outward normal, in the box's "positive upward"
    # convention: `flux_top = int dtheta/dn_top`, `flux_bot = -int dtheta/dn_bot`.
    sideb, sidet = nu_wall_sides(g)
    f_top, _ = _semquad_wall_flux(g, sq, theta, sidet)
    f_bot, _ = _semquad_wall_flux(g, sq, theta, sideb)
    st = f_top
    sb = -f_bot
    return (theta_int=theta_int, latent_int=latent_int,
            enthalpy=theta_int + latent_int,
            flux_top=st, flux_bot=sb, flux=st - sb,
            heating=heating, base_adv=base_adv,
            latent_src=(use_lat ? (latent_bal ? latent_src : nothing) : 0.0),
            latent_adv=(use_lat ? (latent_bal ? latent_adv : nothing) : 0.0),
            latent_dgdt=(use_lat ? (latent_bal ? latent_src + latent_adv : nothing) : 0.0),
            u_grad_theta=(use_lat ? (latent_bal ? u_grad_theta : nothing) : 0.0))
end

