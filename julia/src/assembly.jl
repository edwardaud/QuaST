const QUAD_DIAG = Ref{Any}(nothing)
# Sparse assembly of the Stokes saddle-point system and of the energy operator.
#
# Everything is assembled on the reduced (free) degrees of freedom: fixed
# velocity rows/columns and the pinned pressure row are skipped, so the linear
# system is nonsingular by construction and no post-hoc row zeroing is needed.
# The true residual of every solve is computed in `direct.jl`.

struct StokesSystem
    A::SparseMatrixCSC{Float64,Int}
    f::Vector{Float64}
    eta_min::Float64
    eta_max::Float64
    nclip::Int
end

"""
Assemble ``[A B^T; B 0]`` and the buoyancy right-hand side on the reduced DOFs.

The viscosity is evaluated at the Gauss points of each element from the local
temperature ``T = Tbar(z) + sum_k theta_k phi_k``.  It is never resampled or
projected onto another grid.
"""
# Boussinesq buoyancy switch: `:full` uses Ra*T (the hydrostatic part is left
# to the pressure), `:anomaly` uses Ra*(T-Tbar).  ASPECT's dynamic-pressure
# formulation solves for the perturbation, so the default is the anomaly form;
# `assemble_stokes` sets this switch from its `buoyancy_form` argument.
const FULL_T_BODY = Ref(false)

function assemble_stokes(m::Mesh, fd::FEData, pr::Params,
                         theta::Vector{Float64}, dm::DofMap;
                         refstate::Union{Nothing,RefState}=nothing,
                         tala::Bool=true,
                         p_prev::Union{Nothing,Vector{Float64}}=nothing,
                         mass_form::Symbol=:rho,
                         buoyancy_form::Symbol=:anomaly,
                         phases::Vector{PhaseTransition}=PhaseTransition[],
                         rheology::Union{Nothing,RheologyModel}=nothing,
                         newton::Bool=false,
                         smooth_caps::Bool=false,
                         u_prev::Union{Nothing,Tuple{Vector{Float64},Vector{Float64}}}=nothing,
                         tosi::Union{Nothing,TosiParams}=nothing,
                         T_offset::Float64=0.0, ala_test::Bool=false)
    FULL_T_BODY[] = (buoyancy_form === :full)
    n = dm.ntotal
    nv = (m.p + 1)^2
    np = (m.q + 1)^2
    est = m.nxe * m.nze * (4 * nv * nv + 2 * np * nv)
    I = Int[]
    J = Int[]
    V = Float64[]
    sizehint!(I, est)
    sizehint!(J, est)
    sizehint!(V, est)
    f = zeros(Float64, n)

    Axx = zeros(Float64, nv, nv)
    Auxz = zeros(Float64, nv, nv)
    Auzux = zeros(Float64, nv, nv)
    Azz = zeros(Float64, nv, nv)
    Bx = zeros(Float64, np, nv)
    Bz = zeros(Float64, np, nv)
    fz = zeros(Float64, nv)
    th = zeros(Float64, nv)
    phik = zeros(Float64, nv)
    gxk = zeros(Float64, nv)
    gzk = zeros(Float64, nv)
    psik = zeros(Float64, np)

    p = m.p
    q = m.q
    hx = m.hx
    pv = fd.phi_v
    dpv = fd.dphi_v
    pp = fd.phi_p
    dpp = fd.dphi_p
    np = (m.q + 1)^2
    pp = fd.phi_p
    eta_lo = Inf
    eta_hi = -Inf
    nclip = 0

    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        ze0 = z_left(m, ez)
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            gp = element_pres_gids(m, ex, ez)
            fill!(Axx, 0.0)
            fill!(Auxz, 0.0)
            fill!(Auzux, 0.0)
            fill!(Azz, 0.0)
            fill!(Bx, 0.0)
            fill!(Bz, 0.0)
            fill!(fz, 0.0)
            for k in 1:nv
                th[k] = theta[g[k]]
            end
            for qy in 1:fd.nq
                z = ze0 + 0.5 * (fd.gl_x[qy] + 1.0) * hze
                wy = fd.gl_w[qy]
                for qx in 1:fd.nq
                    x = ex * hx + 0.5 * (fd.gl_x[qx] + 1.0) * hx
                    rho_q = 1.0; alpha_q = 1.0; Tad_q = 0.0
                    rho_n = 1.0; alpha_n = 1.0
                    if refstate !== nothing
                        rho_q, alpha_q, _, _, beta_q, Tad_q, pad_q = refstate_at(refstate, z)
                        rho_n = rho_q / refstate.rho0
                        alpha_n = alpha_q / refstate.alpha0
                    end
                    Tloc = refstate === nothing ? Tbar(z, pr.H) : Tbar(z, pr.H) + refstate.Tad[end]
                    k = 0
                    for j in 1:p+1
                        py = pv[j, qy]
                        for i in 1:p+1
                            k += 1
                            b = pv[i, qx] * py
                            Tloc += th[k] * b
                            phik[k] = b
                            gxk[k] = dpv[i, qx] * py * (2.0 / hx)
                            gzk[k] = pv[i, qx] * dpv[j, qy] * (2.0 / hze)
                        end
                    end
                    Tphys = Tloc + T_offset
                    edot = rheology === nothing ? 0.0 : rheology.ref_eps / rheology.eps_scale
                    p_use = refstate === nothing ? 0.0 : refstate_at(refstate, z)[7]
                    exd_u = 0.0; ezd_u = 0.0; exz_u = 0.0; dxd = 0.0; dzd = 0.0
                    if (rheology !== nothing || tosi !== nothing) && u_prev !== nothing
                        duxdx = 0.0; duxdz = 0.0; duzdx = 0.0; duzdz = 0.0
                        for kk in 1:nv
                            duxdx += u_prev[1][g[kk]] * gxk[kk]
                            duxdz += u_prev[1][g[kk]] * gzk[kk]
                            duzdx += u_prev[2][g[kk]] * gxk[kk]
                            duzdz += u_prev[2][g[kk]] * gzk[kk]
                        end
                        tr = duxdx + duzdz
                        exd_u = duxdx
                        ezd_u = duzdz
                        exz_u = 0.5 * (duxdz + duzdx)
                        dxd = duxdx - 0.5 * tr
                        dzd = duzdz - 0.5 * tr
                        rawed = sqrt(max(0.5 * (dxd*dxd + dzd*dzd +
                                               2.0*exz_u*exz_u), 0.0))
                        edot = rawed > 0.0 ?
                               (rheology === nothing ? rawed : max(rawed, rheology.min_eps / rheology.eps_scale)) :
                               (rheology === nothing ? 0.0 : rheology.ref_eps / rheology.eps_scale)
                    end
                    if rheology !== nothing && p_prev !== nothing
                        pq_p = 0.0
                        l = 0
                        for b in 1:(q+1), a in 1:(q+1)
                            l += 1
                            pq_p += pp[a,qx] * pp[b,qy] * p_prev[gp[l]]
                        end
                        p_use = pq_p
                    end
                    deta_tosi = 0.0
                    eta, clipped = if tosi !== nothing
                        et, dt = eta_tosi(Tloc, edot, 1.0 - z, tosi)
                        deta_tosi = dt
                        (et, false)
                    elseif rheology !== nothing
                        if u_prev === nothing
                            (1.0, false)
                        else
                            (eta_rheology_nondim(rheology, edot, p_use, Tphys), false)
                        end
                    elseif pr.eta_custom === nothing
                        eta_at(Tphys, z, pr)
                    else
                        (Float64(pr.eta_custom(x, z)), false)
                    end
                    eta = eta::Float64
                    clipped = clipped::Bool
                    cap_factor = 1.0
                    spd_alpha = 1.0
                    eta_raw_nd = 0.0
                    deta_raw_nd = 0.0
                    if (newton || smooth_caps) && rheology !== nothing && u_prev !== nothing
                        eps_phys = edot * rheology.eps_scale
                        T_phys = rheology.Tsurf + rheology.dT * Tloc
                        p_phys = p_use * rheology.p0
                        if rheology.yield.c < 1e30
                            eta_phys, deta_phys = eta_vp_smooth_stable(eps_phys, p_phys, T_phys,
                                                                           rheology.creep, rheology.yield)
                        else
                            eta_phys = eta_visco_plastic_raw(eps_phys, p_phys, T_phys,
                                                             rheology.creep, rheology.yield)
                            deta_phys = deta_deps_dislocation(eps_phys, p_phys, T_phys,
                                                              rheology.creep)
                        end
                        eta_raw_nd = eta_phys / rheology.eta0
                        deta_raw_nd = deta_phys / rheology.eta0
                        eta, cap_factor = smooth_cap_log(eta_raw_nd,
                                                         rheology.etamin / rheology.eta0,
                                                         rheology.etamax / rheology.eta0, 20.0)
                        spd_alpha = spd_factor(eta, deta_raw_nd, edot, exd_u, ezd_u, exz_u,
                                               dxd, dzd, exz_u; safety=0.9)
                    end
                    if QUAD_DIAG[] isa Vector
                        push!(QUAD_DIAG[], (qx, qy, z, eta, edot, dxd, dzd, exz_u, eta_raw_nd, deta_raw_nd, cap_factor, p_use))
                    end
                    eta_lo = min(eta_lo, eta)
                    eta_hi = max(eta_hi, eta)
                    clipped && (nclip += 1)

                    l = 0
                    for b in 1:q+1
                        py = pp[b, qy]
                        for a in 1:q+1
                            l += 1
                            psik[l] = pp[a, qx] * py
                        end
                    end
                    w = fd.gl_w[qx] * wy * hx * hze / 4.0
                    pq = 0.0
                    if refstate !== nothing && p_prev !== nothing
                        for l in 1:np
                            pq += psik[l] * p_prev[gp[l]]
                        end
                    end

                    # A blocks
                    geo_coef = 0.0
                    if newton && tosi !== nothing && edot > 0.0
                        geo_coef = deta_tosi / edot
                    elseif newton && rheology !== nothing && u_prev !== nothing && edot > 0.0
                        n_exp = rheology.creep.n
                        geo_coef = eta * cap_factor * deta_raw_nd / (eta_raw_nd * edot) * spd_alpha
                    end
                    for kk in 1:nv
                        gx_k = gxk[kk]
                        gz_k = gzk[kk]
                        t2 = 2.0 * eta
                        Tux_k = exd_u * gx_k + exz_u * gz_k
                        Tuz_k = ezd_u * gz_k + exz_u * gx_k
                        for ll in 1:nv
                            Axx[kk, ll] += t2 * ((ala_test ? 2.0 / 3.0 : 1.0) * gx_k * gxk[ll] + 0.5 * gz_k * gzk[ll]) * w
                            Azz[kk, ll] += t2 * ((ala_test ? 2.0 / 3.0 : 1.0) * gz_k * gzk[ll] + 0.5 * gx_k * gxk[ll]) * w
                            Auxz[kk, ll] += (ala_test ? 1.0 / 3.0 : 1.0) * eta * gz_k * gxk[ll] * w
                            Auzux[kk, ll] += (ala_test ? 1.0 / 3.0 : 1.0) * eta * gx_k * gzk[ll] * w
                            if geo_coef != 0.0
                                Dux_l = 0.5 * (dxd - dzd) * gxk[ll] + exz_u * gzk[ll]
                                Duz_l = 0.5 * (dzd - dxd) * gzk[ll] + exz_u * gxk[ll]
                                Axx[kk, ll] += geo_coef * Tux_k * Dux_l * w
                                Auxz[kk, ll] += geo_coef * Tux_k * Duz_l * w
                                Auzux[kk, ll] += geo_coef * Tuz_k * Dux_l * w
                                Azz[kk, ll] += geo_coef * Tuz_k * Duz_l * w
                            end
                        end
                    end
                    # B blocks
                    # ASPECT assembles the ALA/reference-density mass equation as
                    #   -div u - (1/rho) drho_d(depth) (ghat . u) = 0
                    # (source/simulator/assemblers/stokes.cc:600 for the explicit
                    #  RHS form, and :636-638 for the implicit matrix form, used
                    #  when the reference density profile is implicit).
                    # With z pointing up: drho_d(depth) = -drho/dz and ghat_z = -1,
                    # so the u_z coefficient is -pressure_scaling * (drho/dz)/rho.
                    # For King (rho = exp(Di*depth)) this gives div u = +Di u_z,
                    # i.e. upwelling material expands.  Getting this sign wrong
                    # forces div u = -Di u_z and makes div(rho u) = -2 Di rho u_z.
                    drho_ratio = 0.0
                    if (mass_form === :aspect || (refstate !== nothing && mass_form === :rho)) && refstate !== nothing
                        drho_ratio = refstate_drho(refstate, z) / rho_q
                    end
                    for ll in 1:np
                        ps = psik[ll]
                        for kk in 1:nv
                            if (mass_form === :aspect || (refstate !== nothing && mass_form === :rho))
                                Bx[ll, kk] += -ps * gxk[kk] * w
                                Bz[ll, kk] += -ps * gzk[kk] * w -
                                              ps * drho_ratio * phik[kk] * w
                            else
                                Bx[ll, kk] += -ps * rho_n * gxk[kk] * w
                                Bz[ll, kk] += -ps * rho_n * gzk[kk] * w
                            end
                        end
                    end
                    # buoyancy
                    phase_force = 0.0
                    if refstate !== nothing && buoyancy_form !== :direct
                        for pt in phases
                            dG = phase_gamma(pt, z, Tphys) - phase_gamma(pt, z, Tad_q)
                            phase_force += pr.Ra * pt.B * dG *
                                           (1.0 - refstate.alpha0 * alpha_n * (Tphys - Tad_q))
                        end
                    end
                    for kk in 1:nv
                        if refstate === nothing
                            fz[kk] += pr.Ra * (FULL_T_BODY[] ? Tloc : (Tloc - Tbar(z, pr.H))) * phik[kk] * w
                        else
                            # TALA buoyancy: -rho_bar alpha_bar T' g, g downward;
                            # upward force is +Ra rho_bar alpha_bar (T - Tad)
                            if buoyancy_form === :direct
                                dens_n = 1.0 / refstate.alpha0
                                for pt in phases
                                    dens_n += pt.B * phase_gamma(pt, z, Tphys)
                                end
                                fz[kk] += pr.Ra * dens_n *
                                          (1.0 - refstate.alpha0 * alpha_n * (Tphys - Tad_q)) *
                                          phik[kk] * w
                            elseif buoyancy_form === :full
                                # ALA: the static buoyancy rho_ad*g is balanced by grad(p_ad).
                                # With a dynamic pressure we keep only the perturbation
                                # alpha*theta (theta = Tloc - [Tbar + Tad(top)]).
                                # stokes.cc:374-379 uses the full rho*g only together with p_ad.
                                th_q = Tloc - (Tbar(z, pr.H) + refstate.Tad[end])
                                fz[kk] += pr.Ra * rho_n * alpha_n * th_q * phik[kk] * w
                            else
                                fz[kk] += pr.Ra * rho_n * alpha_n * (Tphys - Tad_q) * phik[kk] * w
                            end
                            # phase buoyancy: heavy phase pulls downward
                            fz[kk] -= phase_force * phik[kk] * w
                            if p_prev !== nothing
                                # ALA beta_T p' term: -Ra rho_bar beta_T p' g
                                fz[kk] -= rho_n * (beta_q / refstate.beta0) * pq * phik[kk] * w
                            end
                        end
                    end
                end
            end

            # scatter: u_x block
            for kk in 1:nv
                rk = dm.red_ux[g[kk]]
                rk == 0 && continue
                for ll in 1:nv
                    rl = dm.red_ux[g[ll]]
                    rl == 0 && continue
                    v = Axx[kk, ll]
                    v != 0.0 && (push!(I, rk); push!(J, rl); push!(V, v))
                end
            end
            # u_z block
            for kk in 1:nv
                rk = dm.red_uz[g[kk]]
                rk == 0 && continue
                row = dm.nux + rk
                for ll in 1:nv
                    rl = dm.red_uz[g[ll]]
                    rl == 0 && continue
                    v = Azz[kk, ll]
                    v != 0.0 && (push!(I, row); push!(J, dm.nux + rl); push!(V, v))
                end
            end
            # u_x-u_z coupling
            for kk in 1:nv
                rk = dm.red_ux[g[kk]]
                rk == 0 && continue
                for ll in 1:nv
                    rl = dm.red_uz[g[ll]]
                    rl == 0 && continue
                    v = Auxz[kk, ll]
                    v != 0.0 && (push!(I, rk); push!(J, dm.nux + rl); push!(V, v))
                end
            end
            for kk in 1:nv
                rk = dm.red_uz[g[kk]]
                rk == 0 && continue
                row = dm.nux + rk
                for ll in 1:nv
                    rl = dm.red_ux[g[ll]]
                    rl == 0 && continue
                    v = Auzux[kk, ll]
                    v != 0.0 && (push!(I, row); push!(J, rl); push!(V, v))
                end
            end
            # B blocks (pressure rows)
            for ll in 1:np
                rp = dm.red_p[gp[ll]]
                rp == 0 && continue
                row = dm.nux + dm.nuz + rp
                for kk in 1:nv
                    rk = dm.red_ux[g[kk]]
                    rk > 0 && (v = Bx[ll, kk]; v != 0.0 && (push!(I, row); push!(J, rk); push!(V, v)))
                    rz = dm.red_uz[g[kk]]
                    rz > 0 && (v = Bz[ll, kk]; v != 0.0 && (push!(I, row); push!(J, dm.nux + rz); push!(V, v)))
                end
            end
            # B^T block (velocity rows, pressure columns)
            for ll in 1:np
                rp = dm.red_p[gp[ll]]
                rp == 0 && continue
                col = dm.nux + dm.nuz + rp
                for kk in 1:nv
                    rk = dm.red_ux[g[kk]]
                    rk > 0 && (v = Bx[ll, kk]; v != 0.0 && (push!(I, rk); push!(J, col); push!(V, v)))
                    rz = dm.red_uz[g[kk]]
                    rz > 0 && (v = Bz[ll, kk]; v != 0.0 && (push!(I, dm.nux + rz); push!(J, col); push!(V, v)))
                end
            end
            # buoyancy right-hand side (z component only)
            for kk in 1:nv
                rz = dm.red_uz[g[kk]]
                rz > 0 && (f[dm.nux + rz] += fz[kk])
            end
        end
    end
    A = sparse(I, J, V, n, n)
    return StokesSystem(A, f, eta_lo, eta_hi, nclip)
end

"""
Assemble the discrete divergence ``B`` (all pressure rows, reduced velocity
columns) for the inf-sup / null-space gate.  The pinned pressure row is *not*
removed here; rank deficiency one is the constant-pressure mode.
"""
function assemble_B(m::Mesh, fd::FEData, dm::DofMap)
    nv = (m.p + 1)^2
    np = (m.q + 1)^2
    ncol = dm.nux + dm.nuz
    I = Int[]
    J = Int[]
    V = Float64[]
    sizehint!(I, m.nxe * m.nze * 2 * np * nv)
    sizehint!(J, m.nxe * m.nze * 2 * np * nv)
    sizehint!(V, m.nxe * m.nze * 2 * np * nv)
    phik = zeros(Float64, nv)
    gxk = zeros(Float64, nv)
    gzk = zeros(Float64, nv)
    Bxl = zeros(Float64, np, nv)
    Bzl = zeros(Float64, np, nv)
    p = m.p
    q = m.q
    pv = fd.phi_v
    dpv = fd.dphi_v
    pp = fd.phi_p
    dpp = fd.dphi_p
    np = (m.q + 1)^2
    pp = fd.phi_p
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            gp = element_pres_gids(m, ex, ez)
            fill!(Bxl, 0.0)
            fill!(Bzl, 0.0)
            for qy in 1:fd.nq
                for qx in 1:fd.nq
                    k = 0
                    for j in 1:p+1
                        py = pv[j, qy]
                        for i in 1:p+1
                            k += 1
                            phik[k] = pv[i, qx] * py
                            gxk[k] = dpv[i, qx] * py * (2.0 / m.hx)
                            gzk[k] = pv[i, qx] * dpv[j, qy] * (2.0 / hze)
                        end
                    end
                    w = fd.gl_w[qx] * fd.gl_w[qy] * m.hx * hze / 4.0
                    l = 0
                    for b in 1:q+1
                        py = pp[b, qy]
                        for a in 1:q+1
                            l += 1
                            ps = pp[a, qx] * py
                            for kk in 1:nv
                                Bxl[l, kk] += -ps * gxk[kk] * w
                                Bzl[l, kk] += -ps * gzk[kk] * w
                            end
                        end
                    end
                end
            end
            for l in 1:np
                row = gp[l]
                for kk in 1:nv
                    rk = dm.red_ux[g[kk]]
                    rk > 0 && (v = Bxl[l, kk]; v != 0.0 && (push!(I, row); push!(J, rk); push!(V, v)))
                    rz = dm.red_uz[g[kk]]
                    rz > 0 && (v = Bzl[l, kk]; v != 0.0 && (push!(I, row); push!(J, dm.nux + rz); push!(V, v)))
                end
            end
        end
    end
    return sparse(I, J, V, m.Npres, ncol)
end

"""
Divergence-penalty matrix ``D_ij = int (div phi_i)(div phi_j)`` on the reduced
velocity DOFs.  Adding ``gamma D`` to the Stokes operator enforces div u = 0
more strongly than the weak continuity rows alone.
"""
function assemble_div_penalty(m::Mesh, fd::FEData, dm::DofMap)
    nv = (m.p + 1)^2
    n = dm.ntotal
    I = Int[]; J = Int[]; V = Float64[]
    gxk = zeros(Float64, nv)
    gzk = zeros(Float64, nv)
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            for qy in 1:fd.nq
                for qx in 1:fd.nq
                    k = 0
                    for j in 1:m.p+1
                        for i in 1:m.p+1
                            k += 1
                            gxk[k] = fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / m.hx)
                            gzk[k] = fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hze)
                        end
                    end
                    w = fd.gl_w[qx] * fd.gl_w[qy] * m.hx * hze / 4.0
                    for kk in 1:nv
                        rxk = dm.red_ux[g[kk]]
                        rzk = dm.red_uz[g[kk]]
                        for ll in 1:nv
                            rxl = dm.red_ux[g[ll]]
                            rzl = dm.red_uz[g[ll]]
                            if rxk > 0 && rxl > 0
                                v = gxk[kk] * gxk[ll] * w
                                v != 0.0 && (push!(I, rxk); push!(J, rxl); push!(V, v))
                            end
                            if rxk > 0 && rzl > 0
                                v = gxk[kk] * gzk[ll] * w
                                v != 0.0 && (push!(I, rxk); push!(J, rzl); push!(V, v))
                            end
                            if rzk > 0 && rxl > 0
                                v = gzk[kk] * gxk[ll] * w
                                v != 0.0 && (push!(I, rzk); push!(J, rxl); push!(V, v))
                            end
                            if rzk > 0 && rzl > 0
                                v = gzk[kk] * gzk[ll] * w
                                v != 0.0 && (push!(I, rzk); push!(J, rzl); push!(V, v))
                            end
                        end
                    end
                end
            end
        end
    end
    return sparse(I, J, V, n, n)
end

struct EnergyMatrices
    M::SparseMatrixCSC{Float64,Int}
    K::SparseMatrixCSC{Float64,Int}
    C::SparseMatrixCSC{Float64,Int}
    S::SparseMatrixCSC{Float64,Int}
    F::Vector{Float64}
end

"""Assemble a custom body force ``(fx,fz)`` on the reduced velocity DOFs."""
function assemble_body_force(m::Mesh, fd::FEData, dm::DofMap,
                             fx::Function, fz::Function)
    n = dm.ntotal
    f = zeros(Float64, n)
    nv = (m.p + 1)^2
    phik = zeros(Float64, nv)
    p = m.p
    pv = fd.phi_v
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        ze0 = z_left(m, ez)
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            for qy in 1:fd.nq
                z = ze0 + 0.5 * (fd.gl_x[qy] + 1.0) * hze
                for qx in 1:fd.nq
                    x = ex * m.hx + 0.5 * (fd.gl_x[qx] + 1.0) * m.hx
                    k = 0
                    for j in 1:p+1
                        py = pv[j, qy]
                        for i in 1:p+1
                            k += 1
                            phik[k] = pv[i, qx] * py
                        end
                    end
                    w = fd.gl_w[qx] * fd.gl_w[qy] * m.hx * hze / 4.0
                    fxv = fx(x, z)
                    fzv = fz(x, z)
                    for kk in 1:nv
                        r = dm.red_ux[g[kk]]
                        r > 0 && (f[r] += fxv * phik[kk] * w)
                        r = dm.red_uz[g[kk]]
                        r > 0 && (f[dm.nux + r] += fzv * phik[kk] * w)
                    end
                end
            end
        end
    end
    return f
end

"""
Independent quadrature evaluation of the Stokes bilinear form
``int 2 eta(T) eps(u):eps(v)``.  Used by G4 to compare against ``v' A u`` from
the assembled matrix, without sharing the local-matrix code path.
"""
function bilinear_stokes(m::Mesh, fd::FEData, pr::Params, theta::Vector{Float64},
                         ux::Vector{Float64}, uz::Vector{Float64},
                         vx::Vector{Float64}, vz::Vector{Float64})
    nv = (m.p + 1)^2
    phik = zeros(Float64, nv)
    gxk = zeros(Float64, nv)
    gzk = zeros(Float64, nv)
    p = m.p
    pv = fd.phi_v
    dpv = fd.dphi_v
    pp = fd.phi_p
    dpp = fd.dphi_p
    np = (m.q + 1)^2
    s = 0.0
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        ze0 = z_left(m, ez)
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            for qy in 1:fd.nq
                z = ze0 + 0.5 * (fd.gl_x[qy] + 1.0) * hze
                for qx in 1:fd.nq
                    x = ex * m.hx + 0.5 * (fd.gl_x[qx] + 1.0) * m.hx
                    Tloc = Tbar(z, pr.H)
                    k = 0
                    for j in 1:p+1
                        py = pv[j, qy]
                        for i in 1:p+1
                            k += 1
                            b = pv[i, qx] * py
                            phik[k] = b
                            gxk[k] = dpv[i, qx] * py * (2.0 / m.hx)
                            gzk[k] = pv[i, qx] * dpv[j, qy] * (2.0 / hze)
                            Tloc += theta[g[k]] * b
                        end
                    end
                    Tphys = Tloc
                    eta = pr.eta_custom === nothing ? eta_at(Tphys, z, pr)[1] :
                          Float64(pr.eta_custom(x, z))
                    uxq = 0.0; uzq = 0.0; vxq = 0.0; vzq = 0.0
                    dux_dx = 0.0; dux_dz = 0.0; duz_dx = 0.0; duz_dz = 0.0
                    dvx_dx = 0.0; dvx_dz = 0.0; dvz_dx = 0.0; dvz_dz = 0.0
                    for kk in 1:nv
                        b = phik[kk]
                        gx = gxk[kk]
                        gz = gzk[kk]
                        uxk = ux[g[kk]]; uzk = uz[g[kk]]
                        vxk = vx[g[kk]]; vzk = vz[g[kk]]
                        uxq += uxk * b; uzq += uzk * b
                        vxq += vxk * b; vzq += vzk * b
                        dux_dx += uxk * gx; dux_dz += uxk * gz
                        duz_dx += uzk * gx; duz_dz += uzk * gz
                        dvx_dx += vxk * gx; dvx_dz += vxk * gz
                        dvz_dx += vzk * gx; dvz_dz += vzk * gz
                    end
                    exx_u = dux_dx
                    ezz_u = duz_dz
                    exz_u = 0.5 * (dux_dz + duz_dx)
                    exx_v = dvx_dx
                    ezz_v = dvz_dz
                    exz_v = 0.5 * (dvx_dz + dvz_dx)
                    w = fd.gl_w[qx] * fd.gl_w[qy] * m.hx * hze / 4.0
                    s += 2.0 * eta * (exx_u * exx_v + ezz_u * ezz_v +
                                      2.0 * exz_u * exz_v) * w
                end
            end
        end
    end
    return s
end

"""
Assemble mass, stiffness, advection and SUPG matrices plus the base-state
right-hand side, all on the free theta DOFs (homogeneous Dirichlet at z=0,1).

The SUPG coefficient is ``tau_e = (h_e/(2|u|))(coth Pe - 1/Pe)`` with
``Pe = |u| h_e / 2`` and ``h_e = sqrt(hx hz)``.  It is multiplied by
``pr.supg``; ``pr.supg = 0`` switches the term off exactly.
"""
function assemble_energy(m::Mesh, fd::FEData, pr::Params,
                         ux::Vector{Float64}, uz::Vector{Float64},
                         dmt::ThetaMap; advection::Symbol=:skew,
                         velq::Union{Nothing,QuadVelocity}=nothing,
                         ev::Float64=0.0,
                         ev_cmax::Float64=0.5,
                         ev_alpha::Int=2,
                         ev_gamma::Float64=0.0,
                         ev_timestep::Int=0,
                         ev_dt_ratio::Float64=1.0,
                         ev_length_scale::Float64=0.0,
                         theta_now::Union{Nothing,Vector{Float64}}=nothing,
                         theta_prev::Union{Nothing,Vector{Float64}}=nothing,
                         dt::Float64=0.0,
                         ux_ev::Union{Nothing,Vector{Float64}}=nothing,
                         uz_ev::Union{Nothing,Vector{Float64}}=nothing,
                         nu_const::Float64=0.0,
                         nu_fun::Union{Nothing,Function}=nothing,
                         ev_bnd_neighbor::Bool=false,
                         ev_bnd_own::Bool=false,
                         heating_adiabatic::Bool=false,
                         heating_dissipation::Bool=false,
                         Di::Float64=0.0,
                         refstate::Union{Nothing,RefState}=nothing,
                         phases::Vector{PhaseTransition}=PhaseTransition[],
                         latent_heat::Bool=true,
                         p_prev::Union{Nothing,Vector{Float64}}=nothing,
                         ev_diag::Union{Nothing,Vector{Float64}}=nothing, T_offset::Float64=0.0, ala_test::Bool=false, diag::Union{Nothing,Dict{Symbol,Float64}}=nothing)
    em = _energy_from_local(m, fd, pr, ux, uz, dmt; velq=velq, ev=ev,
                            ev_cmax=ev_cmax, ev_alpha=ev_alpha,
                            ev_gamma=ev_gamma, ev_timestep=ev_timestep,
                            ev_dt_ratio=ev_dt_ratio,
                            ev_length_scale=ev_length_scale,
                            theta_now=theta_now,
                            theta_prev=theta_prev, dt=dt,
                            ux_ev=ux_ev, uz_ev=uz_ev, nu_const=nu_const,
                            nu_fun=nu_fun, ev_bnd_neighbor=ev_bnd_neighbor,
                            ev_bnd_own=ev_bnd_own,
                            heating_adiabatic=heating_adiabatic,
                            heating_dissipation=heating_dissipation, Di=Di,
                            refstate=refstate,
                            phases=phases,
                            latent_heat=latent_heat,
                            p_prev=p_prev,
                            ev_diag=ev_diag, T_offset=T_offset, diag=diag)
    if advection === :galerkin
        return em
    elseif advection === :conservative
        # weak form of div(u theta): -C^T (boundary terms vanish)
        return EnergyMatrices(em.M, em.K, -sparse(transpose(em.C)), em.S, em.F)
    elseif advection === :skew
        # 0.5(C-C^T): theta^T C theta = 0 even if div u != 0
        Ct = sparse(transpose(em.C))
        return EnergyMatrices(em.M, em.K, 0.5 * (em.C - Ct), em.S, em.F)
    else
        throw(ArgumentError("advection must be :galerkin, :skew or :conservative"))
    end
end

# ---------------------------------------------------------------------------
# Entropy viscosity, ASPECT form
# ---------------------------------------------------------------------------
# Sources (ASPECT 2.5.0-pre, main 340c0d018):
#   source/simulator/entropy_viscosity.cc
#     :37-110    get_entropy_variation
#     :120-238   compute_viscosity
#     :246-448   get_artificial_viscosity (incl. Dirichlet boundary rule)
#     :538-571   material inputs at the mean of old/old_old, ALA density
#                replacement by the adiabatic density
#   source/simulator/assemblers/advection.cc:261-301  temperature residual
#   source/simulator/assemblers/interface.cc:609-621  advection prefactor = rho cp
#   source/simulator/helper_functions.cc:474-506      u_infty (nodal |u|)
#   source/simulator/helper_functions.cc:714-786      extrapolated field range
#   source/simulator/assemblers/advection.cc:162-182  diffusion = max(k, nu_art)
#
# Definitions used below (all from the source, none tuned):
#   h_K            cell diameter, deal.II cell->diameter(); for a rectangle the
#                  diagonal sqrt(hx^2 + hz^2)
#   max_prefactor  max_q (rho*cp) in the cell                    (:609-621)
#   max_conduct.   max_q k in the cell
#   u_infty        max over velocity nodes of |u_old|
#   field range    range of the extrapolated T:
#                  (1+dt/dt_old) T_old - (dt/dt_old) T_oldold, or T_old when
#                  timestep_number <= 1
#   average_field  0.5*(T_min + T_max)          (entropy_viscosity.cc:278-280)
#   entropy var.   max(max e - <e>, <e> - min e) with e = (T_mid - average)^2,
#                  <e> = int e dV / int dV, T_mid = (T_old + T_oldold)/2
#   residual       |rho cp (dT/dt + u.grad T) - k lap T - gamma|, multiplied by
#                  |T_mid - average| when alpha = 2
#   nu_max         beta * max_prefactor * max_velocity * h_K
#   nu_e alpha=2   cR h_K^2 max_residual / entropy_variation
#   nu_e alpha=1   cR h_K Omega max_velocity max_residual /
#                  (u_infty field_variation)
#   nu             min(nu_max, nu_e); nu_max for timestep_number <= 1 or when
#                  the entropy/field variation vanishes
#   boundary       a cell touching a Dirichlet-T boundary whose velocity BC is
#                  tangential or zero gets nu = 0 (all our walls are free-slip
#                  or rigid, so this is the top and bottom cell row)

struct EVGlobals
    minT::Float64
    maxT::Float64
    avgT::Float64
    entropy_variation::Float64
    avg_entropy::Float64
    field_variation::Float64
    u_infty::Float64
end

"""
    _ev_globals(m, fd, pr, refstate, theta_now, theta_prev, ux_mid, uz_mid,
                dt_ratio, T_offset)

Domain-wide quantities required by ASPECT's entropy viscosity
(`entropy_viscosity.cc:276-281`).  `dt_ratio = dt_new/dt_old` is used only for
the extrapolated field range; pass 0 to use the old field (ASPECT does this for
`timestep_number <= 1`).
"""
function _ev_globals(m::Mesh, fd::FEData, pr::Params,
                     refstate::Union{Nothing,RefState},
                     theta_now::Vector{Float64},
                     theta_prev::Union{Nothing,Vector{Float64}},
                     ux_mid::Vector{Float64}, uz_mid::Vector{Float64},
                     dt_ratio::Float64, T_offset::Float64)
    xs, zs = node_coords(m, fd.gll_v)
    Tbase = (refstate === nothing ? 0.0 : refstate.Tad[end]) + T_offset
    minT = Inf; maxT = -Inf
    uinf = 0.0
    @inbounds for iz in eachindex(zs)
        z = zs[iz]
        tb = Tbar(z, pr.H) + Tbase
        for ix in eachindex(xs)
            g = (iz - 1) * m.nvx + ix
            tn = tb + theta_now[g]
            t = if dt_ratio > 0.0 && theta_prev !== nothing
                (1.0 + dt_ratio) * tn - dt_ratio * (tb + theta_prev[g])
            else
                tn
            end
            minT = min(minT, t); maxT = max(maxT, t)
            uinf = max(uinf, hypot(ux_mid[g], uz_mid[g]))
        end
    end
    avgT = 0.5 * (minT + maxT)
    # entropy variation at the Gauss points of the temperature element
    eint = 0.0; area = 0.0; emin = Inf; emax = -Inf
    for ez in 0:m.nze-1
        hze = z_size(m, ez); z0 = z_left(m, ez)
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            for qy in 1:fd.nq, qx in 1:fd.nq
                z = z0 + 0.5 * (fd.gl_x[qy] + 1.0) * hze
                thn = 0.0; thp = 0.0
                for j in 1:m.p+1, i in 1:m.p+1
                    kk = (j - 1) * (m.p + 1) + i
                    b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                    thn += theta_now[g[kk]] * b
                    if theta_prev !== nothing
                        thp += theta_prev[g[kk]] * b
                    end
                end
                Tmid = Tbar(z, pr.H) + Tbase + 0.5 * (thn + thp)
                e = (Tmid - avgT)^2
                w = fd.gl_w[qx] * fd.gl_w[qy] * m.hx * hze / 4.0
                eint += w * e; area += w
                emin = min(emin, e); emax = max(emax, e)
            end
        end
    end
    avge = area > 0.0 ? eint / area : 0.0
    entropy_variation = max(emax - avge, avge - emin)
    return EVGlobals(minT, maxT, avgT, entropy_variation, avge,
                     maxT - minT, uinf)
end

"""
    _ev_cell_stats(m, fd, pr, refstate, theta_now, theta_prev, ux_mid, uz_mid,
                   dt, ez, ex, g; kwargs...)

Cell-wise maximum residual, maximum velocity, maximum rho*cp and maximum
conductivity, exactly as `AdvectionSystem::compute_residual`
(assemblers/advection.cc:261-301) and `compute_viscosity`
(entropy_viscosity.cc:120-238).

`dt` is the OLD timestep (ASPECT's `old_time_step`).  The velocity used is the
mean of the old and old-old velocities.
"""
function _ev_cell_stats(m::Mesh, fd::FEData, pr::Params,
                        refstate::Union{Nothing,RefState},
                        theta_now::Vector{Float64},
                        theta_prev::Union{Nothing,Vector{Float64}},
                        ux_mid::Vector{Float64}, uz_mid::Vector{Float64},
                        dt::Float64, ez::Int, ex::Int, g::EVGlobals;
                        T_offset::Float64=0.0, Di::Float64=0.0,
                        heating_adiabatic::Bool=false,
                        heating_dissipation::Bool=false,
                        alpha::Int=2, gamma_stab::Float64=0.0,
                        velq::Union{Nothing,QuadVelocity}=nothing)
    p = m.p
    hx = m.hx
    hze = z_size(m, ez)
    z0 = z_left(m, ez)
    pv = fd.phi_v
    dpv = fd.dphi_v
    Tbase = (refstate === nothing ? 0.0 : refstate.Tad[end]) + T_offset
    gg = element_vel_gids(m, ex, ez)
    he = sqrt(hx * hx + hze * hze)
    Rmax = 0.0
    umax = 0.0
    max_pref = 0.0
    max_k = 0.0
    for qy in 1:fd.nq
        z = z0 + 0.5 * (fd.gl_x[qy] + 1.0) * hze
        gz_base = dTbar(z, pr.H)
        rho_n = 1.0; cp_n = 1.0; alpha_n = 1.0; kdiff = 1.0
        if refstate !== nothing
            rq, aq, cq, kq, _, _, _ = refstate_at(refstate, z)
            rho_n = rq / refstate.rho0
            cp_n = cq / refstate.cp0
            alpha_n = aq / refstate.alpha0
            kdiff = kq / refstate.k0
        end
        rho_cp = rho_n * cp_n
        max_pref = max(max_pref, rho_cp)
        max_k = max(max_k, kdiff)
        for qx in 1:fd.nq
            _, _, d2x = lagrange_d2(fd.gll_v, fd.gl_x[qx])
            _, _, d2y = lagrange_d2(fd.gll_v, fd.gl_x[qy])
            uxq = 0.0; uzq = 0.0
            thn = 0.0; thp = 0.0
            gthx = 0.0; gthz = 0.0; lapth = 0.0
            gthxp = 0.0; gthzp = 0.0; lapp = 0.0
            duxdx = 0.0; duxdz = 0.0; duzdx = 0.0; duzdz = 0.0
            k = 0
            for j in 1:p+1
                py = pv[j, qy]
                for i in 1:p+1
                    k += 1
                    b = pv[i, qx] * py
                    gxk = dpv[i, qx] * py * (2.0 / hx)
                    gzk = pv[i, qx] * dpv[j, qy] * (2.0 / hze)
                    uxq += ux_mid[gg[k]] * b
                    uzq += uz_mid[gg[k]] * b
                    duxdx += ux_mid[gg[k]] * gxk
                    duxdz += ux_mid[gg[k]] * gzk
                    duzdx += uz_mid[gg[k]] * gxk
                    duzdz += uz_mid[gg[k]] * gzk
                    thn += theta_now[gg[k]] * b
                    gthx += theta_now[gg[k]] * gxk
                    gthz += theta_now[gg[k]] * gzk
                    lapth += theta_now[gg[k]] *
                             (d2x[i] * py * (2.0 / hx)^2 +
                              pv[i, qx] * d2y[j] * (2.0 / hze)^2)
                    if theta_prev !== nothing
                        thp += theta_prev[gg[k]] * b
                        gthxp += theta_prev[gg[k]] * gxk
                        gthzp += theta_prev[gg[k]] * gzk
                        lapp += theta_prev[gg[k]] *
                                (d2x[i] * py * (2.0 / hx)^2 +
                                 pv[i, qx] * d2y[j] * (2.0 / hze)^2)
                    end
                end
            end
            if velq !== nothing
                uxq = velq.ux[ex+1, ez+1, qx, qy]
                uzq = velq.uz[ex+1, ez+1, qx, qy]
            end
            Tmid = Tbar(z, pr.H) + Tbase + 0.5 * (thn + thp)
            # heating source at the mean state, same formulas as the assembly
            gam_ad = 0.0
            if heating_adiabatic
                gam_ad = -Di * rho_n * alpha_n * Tmid * uzq
            end
            gam_sh = 0.0
            if heating_dissipation
                divu = duxdx + duzdz
                exz = 0.5 * (duxdz + duzdx)
                eps2 = refstate === nothing ?
                       (duxdx^2 + duzdz^2 + 0.5 * (duxdz + duzdx)^2) :
                       ((duxdx - divu / 3)^2 + (duzdz - divu / 3)^2 + 2 * exz^2)
                gam_sh = (Di / pr.Ra) * 2.0 * eta_at(Tmid, z, pr)[1] * eps2
            end
            dTdt = dt > 0.0 ? (thn - thp) / dt : 0.0
            adv = 0.5 * uxq * (gthx + gthxp) + 0.5 * uzq * (gthz + gthzp) +
                  uzq * gz_base
            lap = 0.5 * (lapth + lapp)
            R = abs(rho_cp * (dTdt + adv) - kdiff * lap - (gam_ad + gam_sh))
            alpha == 2 && (R *= abs(Tmid - g.avgT))
            Rmax = max(Rmax, R)
            # max_velocity in the cell: |u| + gamma * ||eps(u)|| * h_K
            eps_norm = sqrt(duxdx^2 + duzdz^2 + 2 * (0.5 * (duxdz + duzdx))^2)
            umax = max(umax, hypot(uxq, uzq) + gamma_stab * eps_norm * he)
        end
    end
    return Rmax, umax, max_pref, max_k
end

"""
    _ev_nu_cell(m, fd, pr, refstate, theta_now, theta_prev, ux_mid, uz_mid,
                dt, ez, ex, evg; ev, ev_cmax, ...) -> (nu_e, nu_max, nu)

Entropy viscosity of one cell, following `compute_viscosity`
(entropy_viscosity.cc:120-238) and the Dirichlet boundary rule (:371-447).
Returns the entropy-viscosity value `nu_e`, the upper bound `nu_max` and the
active value `nu` (min of the two, or `nu_max` on the first two steps; zero in
Dirichlet-temperature cells with tangential/zero velocity).
"""
function _ev_nu_cell(m::Mesh, fd::FEData, pr::Params,
                     refstate::Union{Nothing,RefState},
                     theta_now::Vector{Float64},
                     theta_prev::Union{Nothing,Vector{Float64}},
                     ux_mid::Vector{Float64}, uz_mid::Vector{Float64},
                     dt::Float64, ez::Int, ex::Int, evg::EVGlobals;
                     ev::Float64, ev_cmax::Float64, ev_alpha::Int=2,
                     ev_gamma::Float64=0.0, ev_timestep::Int=0,
                     ev_length::Float64=1.0, T_offset::Float64=0.0,
                     Di::Float64=0.0, heating_adiabatic::Bool=false,
                     heating_dissipation::Bool=false,
                     velq::Union{Nothing,QuadVelocity}=nothing,
                     nu_const::Float64=0.0, nu_fun::Union{Nothing,Function}=nothing)
    hx = m.hx
    hze = z_size(m, ez)
    he = sqrt(hx * hx + hze * hze)
    Rmax, umax_e, max_pref, max_k = _ev_cell_stats(
        m, fd, pr, refstate, theta_now, theta_prev, ux_mid, uz_mid,
        dt, ez, ex, evg; T_offset=T_offset, Di=Di,
        heating_adiabatic=heating_adiabatic,
        heating_dissipation=heating_dissipation,
        alpha=ev_alpha, gamma_stab=ev_gamma, velq=velq)
    nu_max = ev_cmax * max_pref * umax_e * he
    if abs(evg.u_infty) < 1.0e-50
        nu = ev_cmax * max_k / ev_length * he
        return 0.0, nu_max, nu
    elseif ev_timestep <= 1 ||
           abs(evg.entropy_variation) < 1.0e-50 ||
           abs(evg.field_variation) < 1.0e-50
        nu = nu_max
        return 0.0, nu_max, nu
    end
    if ev_alpha == 2
        nu_e = ev * he * he * Rmax / evg.entropy_variation
    else
        nu_e = ev * he * ev_length * umax_e * Rmax /
               (evg.u_infty * evg.field_variation)
    end
    nu = min(nu_max, nu_e)
    if nu_const > 0.0
        nu = nu_const
    elseif nu_fun !== nothing
        nu = nu_fun(z_left(m, ez) + 0.5 * hze)
    end
    # Dirichlet-T cells whose velocity BC is tangential or zero carry no
    # entropy viscosity (entropy_viscosity.cc:371-447).
    if ez == 0 || ez == m.nze - 1
        nu = 0.0
    end
    return nu_e, nu_max, nu
end


const DENSITY_WARNED = Ref(false)

function _energy_from_local(m::Mesh, fd::FEData, pr::Params,
                            ux::Vector{Float64}, uz::Vector{Float64},
                            dmt::ThetaMap;
                            velq::Union{Nothing,QuadVelocity}=nothing,
                            ev::Float64=0.0,
                            ev_cmax::Float64=0.5,
                            ev_alpha::Int=2,
                            ev_gamma::Float64=0.0,
                            ev_timestep::Int=0,
                            ev_dt_ratio::Float64=1.0,
                            ev_length_scale::Float64=0.0,
                            theta_now::Union{Nothing,Vector{Float64}}=nothing,
                            theta_prev::Union{Nothing,Vector{Float64}}=nothing,
                            dt::Float64=0.0,
                            ux_ev::Union{Nothing,Vector{Float64}}=nothing,
                            uz_ev::Union{Nothing,Vector{Float64}}=nothing,
                            nu_const::Float64=0.0,
                            nu_fun::Union{Nothing,Function}=nothing,
                            ev_bnd_neighbor::Bool=false,
                            ev_bnd_own::Bool=false,
                            heating_adiabatic::Bool=false,
                            heating_dissipation::Bool=false,
                            Di::Float64=0.0,
                            refstate::Union{Nothing,RefState}=nothing,
                            phases::Vector{PhaseTransition}=PhaseTransition[],
                            latent_heat::Bool=true,
                            p_prev::Union{Nothing,Vector{Float64}}=nothing,
                            ev_diag::Union{Nothing,Vector{Float64}}=nothing, T_offset::Float64=0.0, ala_test::Bool=false, diag::Union{Nothing,Dict{Symbol,Float64}}=nothing)
    n = dmt.n
    # Domain-wide entropy-viscosity quantities, computed once per assembly from
    # the old/old-old solutions exactly as ASPECT does.  The stabilization is
    # off unless cR > 0: this keeps every pre-existing run unchanged and makes
    # `ev` the single switch (ASPECT itself has the method on by default, with
    # cR=0.11 and beta=0.052; the King ala.prm uses cR=0.033, beta=0.005).
    evg = nothing
    if ev > 0.0 && theta_now !== nothing && dt > 0.0
        evg = _ev_globals(m, fd, pr, refstate, theta_now, theta_prev,
                          ux_ev === nothing ? ux : ux_ev,
                          uz_ev === nothing ? uz : uz_ev,
                          ev_dt_ratio, T_offset)
    end
    ev_length = ev_length_scale > 0.0 ? ev_length_scale : max(m.Lx, 1.0)
    nv = (m.p + 1)^2
    est = m.nxe * m.nze * 4 * nv * nv
    IM = Int[]; JM = Int[]; VM = Float64[]
    IK = Int[]; JK = Int[]; VK = Float64[]
    IC = Int[]; JC = Int[]; VC = Float64[]
    IS = Int[]; JS = Int[]; VS = Float64[]
    for arr in (IM, JM, IK, JK, IC, JC, IS, JS)
        sizehint!(arr, est)
    end
    for arr in (VM, VK, VC, VS)
        sizehint!(arr, est)
    end
    F = zeros(Float64, n)
    M = zeros(Float64, nv, nv)
    K = zeros(Float64, nv, nv)
    C = zeros(Float64, nv, nv)
    S = zeros(Float64, nv, nv)
    Fl = zeros(Float64, nv)
    phik = zeros(Float64, nv)
    gxk = zeros(Float64, nv)
    gzk = zeros(Float64, nv)
    p = m.p
    hx = m.hx
    pv = fd.phi_v
    dpv = fd.dphi_v
    pp = fd.phi_p
    dpp = fd.dphi_p
    np = (m.q + 1)^2
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        ze0 = z_left(m, ez)
        # cell diameter, deal.II cell->diameter() for a rectangle
        he = sqrt(hx * hx + hze * hze)
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            gp = element_pres_gids(m, ex, ez)
            nu_h_e = 0.0
            nu_e_e = 0.0
            nu_max_e = 0.0
            if evg !== nothing
                nu_e_e, nu_max_e, nu_h_e = _ev_nu_cell(
                    m, fd, pr, refstate, theta_now, theta_prev,
                    ux_ev === nothing ? ux : ux_ev,
                    uz_ev === nothing ? uz : uz_ev,
                    dt, ez, ex, evg;
                    ev=ev, ev_cmax=ev_cmax, ev_alpha=ev_alpha,
                    ev_gamma=ev_gamma, ev_timestep=ev_timestep,
                    ev_length=ev_length, T_offset=T_offset, Di=Di,
                    heating_adiabatic=heating_adiabatic,
                    heating_dissipation=heating_dissipation,
                    velq=velq, nu_const=nu_const, nu_fun=nu_fun)
                if ev_diag !== nothing
                    ev_diag[1] = max(ev_diag[1], nu_e_e)
                    ev_diag[2] = max(ev_diag[2], nu_max_e)
                    ev_diag[3] = max(ev_diag[3], nu_h_e)
                    ev_diag[4] += nu_e_e
                    ev_diag[5] += 1.0
                end
            elseif nu_const > 0.0
                nu_h_e = nu_const
            elseif nu_fun !== nothing
                nu_h_e = nu_fun(z_left(m, ez) + 0.5 * hze)
            end
            fill!(M, 0.0); fill!(K, 0.0); fill!(C, 0.0); fill!(S, 0.0); fill!(Fl, 0.0)
            for qy in 1:fd.nq
                z = ze0 + 0.5 * (fd.gl_x[qy] + 1.0) * hze
                gz_base = dTbar(z, pr.H)
                for qx in 1:fd.nq
                    _, _, d2x = lagrange_d2(fd.gll_v, fd.gl_x[qx])
                    _, _, d2y = lagrange_d2(fd.gll_v, fd.gl_x[qy])
                    uxq = 0.0
                    uzq = 0.0
                    duxdx = 0.0; duxdz = 0.0; duzdx = 0.0; duzdz = 0.0
                    thn = 0.0
                    thp = 0.0
                    gthx = 0.0
                    gthz = 0.0
                    lapth = 0.0
                    k = 0
                    for j in 1:p+1
                        py = pv[j, qy]
                        for i in 1:p+1
                            k += 1
                            b = pv[i, qx] * py
                            phik[k] = b
                            gxk[k] = dpv[i, qx] * py * (2.0 / hx)
                            gzk[k] = pv[i, qx] * dpv[j, qy] * (2.0 / hze)
                            uxq += ux[g[k]] * b
                            uzq += uz[g[k]] * b
                            duxdx += ux[g[k]] * gxk[k]
                            duxdz += ux[g[k]] * gzk[k]
                            duzdx += uz[g[k]] * gxk[k]
                            duzdz += uz[g[k]] * gzk[k]
                            if theta_now !== nothing
                                thn += theta_now[g[k]] * b
                                gthx += theta_now[g[k]] * gxk[k]
                                gthz += theta_now[g[k]] * gzk[k]
                                lapth += theta_now[g[k]] *
                                         (d2x[i] * py * (2.0 / hx)^2 +
                                          pv[i, qx] * d2y[j] * (2.0 / hze)^2)
                                if theta_prev !== nothing
                                    thp += theta_prev[g[k]] * b
                                end
                            end
                        end
                    end
                    if velq !== nothing
                        uxq = velq.ux[ex+1, ez+1, qx, qy]
                        uzq = velq.uz[ex+1, ez+1, qx, qy]
                    end
                    umag = sqrt(uxq * uxq + uzq * uzq)
                    nu_h = nu_h_e
                    Pe = umag * he / 2.0
                    tau = 0.0
                    if pr.supg != 0.0
                        if Pe < 1e-4
                            tau = he * he / 12.0
                        else
                            tau = he / (2.0 * umag) * (coth(Pe) - 1.0 / Pe)
                        end
                        tau *= pr.supg
                    end
                    w = fd.gl_w[qx] * fd.gl_w[qy] * hx * hze / 4.0
                    rho_q = 1.0; cp_q = 1.0; alpha_q = 1.0; Tad_q = 0.0
                    rho_n = 1.0; cp_n = 1.0; alpha_n = 1.0
                    if refstate !== nothing
                        rho_q, alpha_q, cp_q, k_q, _, Tad_q, pad_q = refstate_at(refstate, z)
                        rho_n = rho_q / refstate.rho0
                        cp_n = cp_q / refstate.cp0
                        alpha_n = alpha_q / refstate.alpha0
                    end
                    kdiff = refstate === nothing ? 1.0 : k_q / refstate.k0
                    # Total temperature as ASPECT evaluates it: the physical T
                    # includes the adiabatic surface value Tad(top) (the initial
                    # condition and the Dirichlet values are Tbar + Tad(top) +
                    # theta).  Omitting Tad(top) here made the adiabatic heating
                    # alpha*rho*T*(u.g) about 9% too weak for King.
                    Tloc_h = Tbar(z, pr.H) + thn + T_offset +
                             (refstate === nothing ? 0.0 : refstate.Tad[end])
                    # ALA energy equation: ASPECT replaces the material-model
                    # density by the ADIABATIC REFERENCE DENSITY before the
                    # temperature assembler and the heating models run
                    #   source/simulator/assembly.cc:896-904  (dispatching on
                    #   Parameters::Formulation::TemperatureEquation::
                    #    reference_density_profile, set for ALA in
                    #    source/simulator/parameters.cc:1591-1595)
                    # so rho_cp = rho_ad * cp and the adiabatic heating uses
                    # rho_ad.  The full material density (which can even become
                    # negative through the Di*gamma*(p-p_ad) term) is only used
                    # in the momentum equation.  Using the full density here
                    # was a misreading of advection.cc:107-114 and made the
                    # discrete advection inconsistent with div(rho_ad u)=0.
                    wc = refstate === nothing ? 1.0 : rho_n * cp_n
                    lat_cp = 0.0; lat_src = 0.0
                    if refstate !== nothing && latent_heat
                        pqx = 0.0; pqz = 0.0
                        if p_prev !== nothing
                            l = 0
                            for b in 1:(m.q+1), a in 1:(m.q+1)
                                l += 1
                                ps = pp[a,qx] * pp[b,qy]
                                pqx += ps * dpp[a,qx] * (2.0/hx) * p_prev[gp[l]]
                                pqz += ps * pp[a,qx] * dpp[b,qy] * (2.0/hze) * p_prev[gp[l]]
                            end
                        end
                        up = uxq * pqx + uzq * pqz
                        for pt in phases
                            gz = phase_dgamma_dz(pt, z, Tloc_h)
                            s = pt.slope
                            lat_cp += -Tloc_h * gz * pt.B * s * s * Di * rho_n
                            lat_src += (Di / pr.Ra) * Tloc_h * gz * s * pt.B *
                                       refstate.alpha0 / rho_n * up
                        end
                    end
                    wc += lat_cp
                    gam_ad = 0.0
                    if heating_adiabatic
                        # The heating models run after assembly.cc:896-904, so the
                        # simplified adiabatic heating alpha*rho*T*(u.g) also sees
                        # the reference density (adiabatic_heating.cc:54-58).
                        rho_heat = refstate === nothing ? 1.0 : rho_n
                        gam_ad = -Di * rho_heat * alpha_n * Tloc_h * uzq
                    end
                    divu = duxdx + duzdz
                    exz = 0.5 * (duxdz + duzdx)
                    eps2 = refstate === nothing ?
                           (duxdx*duxdx + duzdz*duzdz + 0.5*(duxdz+duzdx)^2) :
                           ((duxdx - divu/3)^2 + (duzdz - divu/3)^2 + 2*exz^2)
                    gam_sh = heating_dissipation ?
                             (Di / pr.Ra) * 2.0 * eta_at(Tloc_h, z, pr)[1] * eps2 : 0.0
                    gam_tot = gam_ad + gam_sh
                    cond_ref = 0.0
                    base_term = -wc * uzq * gz_base
                    if diag !== nothing
                        diag[:gam] = get(diag, :gam, 0.0) + gam_tot * w
                        diag[:base] = get(diag, :base, 0.0) + base_term * w
                        diag[:cond] = get(diag, :cond, 0.0) + cond_ref * w
                        diag[:wcT] = get(diag, :wcT, 0.0) + wc * Tloc_h * w
                    end
                    for kk in 1:nv
                        ph_k = phik[kk]
                        gx_k = gxk[kk]
                        gz_k = gzk[kk]
                        adv_k = uxq * gx_k + uzq * gz_k
                        for ll in 1:nv
                            M[kk, ll] += wc * ph_k * phik[ll] * w
                            # ASPECT uses max(conductivity, artificial_viscosity)
                            # as the diffusion constant (advection.cc:162-182).
                            # The former experimental wc*0.5*|u|*h term is gone:
                            # it was 50-100x larger than ASPECT's entropy
                            # viscosity (whose own output shows nu_art <= 5.5e-3
                            # against k = 1) and biased Nu low.
                            diff_coef = refstate === nothing ? max(1.0, nu_h) :
                                        max(kdiff, nu_h)
                            K[kk, ll] += diff_coef *
                                         (gx_k * gxk[ll] + gz_k * gzk[ll]) * w
                            adv_l = uxq * gxk[ll] + uzq * gzk[ll]
                            C[kk, ll] += wc * ph_k * adv_l * w
                            if tau != 0.0
                                S[kk, ll] += wc * tau * adv_k * adv_l * w
                            end
                        end
                        Fl[kk] += (base_term * ph_k + gam_tot * ph_k + lat_src * ph_k + cond_ref * ph_k) * w
                    end
                end
            end
            for kk in 1:nv
                rk = dmt.red[g[kk]]
                rk == 0 && continue
                F[rk] += Fl[kk]
                for ll in 1:nv
                    rl = dmt.red[g[ll]]
                    rl == 0 && continue
                    v = M[kk, ll]
                    v != 0.0 && (push!(IM, rk); push!(JM, rl); push!(VM, v))
                    v = K[kk, ll]
                    v != 0.0 && (push!(IK, rk); push!(JK, rl); push!(VK, v))
                    v = C[kk, ll]
                    v != 0.0 && (push!(IC, rk); push!(JC, rl); push!(VC, v))
                    v = S[kk, ll]
                    v != 0.0 && (push!(IS, rk); push!(JS, rl); push!(VS, v))
                end
            end
        end
    end
    Msp = sparse(IM, JM, VM, n, n)
    Ksp = sparse(IK, JK, VK, n, n)
    Csp = sparse(IC, JC, VC, n, n)
    Ssp = sparse(IS, JS, VS, n, n)
    return EnergyMatrices(Msp, Ksp, Csp, Ssp, F)
end






































"""
    assemble_velocity_mass(m, fd, dm)

Lumped-consistent velocity mass matrix (Q_p) for the pseudo-transient term.
Pressure rows/columns are zero.
"""
function assemble_velocity_mass(m::Mesh, fd::FEData, dm::DofMap)
    n = dm.ntotal
    I = Int[]; J = Int[]; V = Float64[]
    nv = (m.p + 1)^2
    phik = zeros(Float64, nv)
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            for qy in 1:fd.nq, qx in 1:fd.nq
                k = 0
                for j in 1:m.p+1
                    py = fd.phi_v[j, qy]
                    for i in 1:m.p+1
                        k += 1
                        phik[k] = fd.phi_v[i, qx] * py
                    end
                end
                w = fd.gl_w[qx] * fd.gl_w[qy] * m.hx * hze / 4.0
                for kk in 1:nv
                    a = dm.red_ux[g[kk]]; b = dm.red_uz[g[kk]]
                    for ll in 1:nv
                        val = phik[kk] * phik[ll] * w
                        if a > 0
                            c = dm.red_ux[g[ll]]
                            c > 0 && (push!(I, a); push!(J, c); push!(V, val))
                        end
                        if b > 0
                            c = dm.red_uz[g[ll]]
                            c > 0 && (push!(I, dm.nux + b); push!(J, dm.nux + c); push!(V, val))
                        end
                    end
                end
            end
        end
    end
    return sparse(I, J, V, n, n)
end














































