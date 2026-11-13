# Divergence cleaning / velocity projection.
#
# The Q4-Q3 Stokes velocity is only weakly divergence-free.  In the strong
# variable-viscosity boundary layers of Blankenbach 2a the pointwise
# max|div u| can be O(10^2), which destroys the discrete energy balance.
#
# This module solves the Neumann problem
#
#     Delta phi = div u,   dphi/dn = 0,
#
# with phi in the pressure space Q_q, then updates u <- u - grad phi.
# One phi degree of freedom is pinned.  The correction is evaluated at the
# velocity GLL nodes from the local pressure basis (averaged on shared nodes),
# and the fixed normal-velocity components are re-zeroed afterwards.

function assemble_pressure_stiffness(m::Mesh, fd::FEData)
    np = (m.q + 1)^2
    I = Int[]
    J = Int[]
    V = Float64[]
    Kl = zeros(Float64, np, np)
    q = m.q
    hx = m.hx
    pp = fd.phi_p
    dpp = fd.dphi_p
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        for ex in 0:m.nxe-1
            gp = element_pres_gids(m, ex, ez)
            fill!(Kl, 0.0)
            for qy in 1:fd.nq
                for qx in 1:fd.nq
                    w = fd.gl_w[qx] * fd.gl_w[qy] * hx * hze / 4.0
                    l = 0
                    for b in 1:q+1
                        for a in 1:q+1
                            l += 1
                            ps = pp[a, qx] * pp[b, qy]
                            gx = dpp[a, qx] * pp[b, qy] * (2.0 / hx)
                            gz = pp[a, qx] * dpp[b, qy] * (2.0 / hze)
                            for l2 in 1:np
                                b2 = (l2 - 1) ÷ (q + 1) + 1
                                a2 = mod(l2 - 1, q + 1) + 1
                                ps2 = pp[a2, qx] * pp[b2, qy]
                                gx2 = dpp[a2, qx] * pp[b2, qy] * (2.0 / hx)
                                gz2 = pp[a2, qx] * dpp[b2, qy] * (2.0 / hze)
                                Kl[l, l2] += (gx * gx2 + gz * gz2) * w
                            end
                        end
                    end
                end
            end
            for l in 1:np
                for l2 in 1:np
                    v = Kl[l, l2]
                    v != 0.0 && (push!(I, gp[l]); push!(J, gp[l2]); push!(V, v))
                end
            end
        end
    end
    return sparse(I, J, V, m.Npres, m.Npres)
end

"""
Project the velocity onto (approximately) divergence-free fields.

Returns `(uxp, uzp, div_before, div_after)`.
"""
function project_velocity(m::Mesh, fd::FEData, dm::DofMap,
                          ux::Vector{Float64}, uz::Vector{Float64};
                          Kp::Union{Nothing,SparseMatrixCSC{Float64,Int}}=nothing,
                          verbose::Bool=false)
    K = Kp === nothing ? assemble_pressure_stiffness(m, fd) : Kp
    np = (m.q + 1)^2
    q = m.q
    hx = m.hx
    pp = fd.phi_p
    dpp = fd.dphi_p
    pv = fd.phi_v
    dpv = fd.dphi_v

    # ---- RHS -int psi div u ------------------------------------------------
    b = zeros(Float64, m.Npres)
    gxk = zeros(Float64, np)
    gzk = zeros(Float64, np)
    psik = zeros(Float64, np)
    nv = (m.p + 1)^2
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            gp = element_pres_gids(m, ex, ez)
            for qy in 1:fd.nq
                for qx in 1:fd.nq
                    # div u at the quad point
                    divu = 0.0
                    k = 0
                    for j in 1:m.p+1
                        py = pv[j, qy]
                        for i in 1:m.p+1
                            k += 1
                            gx = dpv[i, qx] * py * (2.0 / hx)
                            gz = pv[i, qx] * dpv[j, qy] * (2.0 / hze)
                            divu += ux[g[k]] * gx + uz[g[k]] * gz
                        end
                    end
                    w = fd.gl_w[qx] * fd.gl_w[qy] * hx * hze / 4.0
                    l = 0
                    for bb in 1:q+1
                        for aa in 1:q+1
                            l += 1
                            psi = pp[aa, qx] * pp[bb, qy]
                            b[gp[l]] += -psi * divu * w
                        end
                    end
                end
            end
        end
    end

    # ---- solve K phi = b with one pinned pressure dof ----------------------
    pin = dm.pin
    free = [i for i in 1:m.Npres if i != pin]
    phi = zeros(Float64, m.Npres)
    Kred = K[free, free]
    bred = b[free]
    phired = Kred \ bred
    phi[free] = phired

    # ---- correction at the velocity nodes ---------------------------------
    # pressure basis values/derivatives at the velocity GLL nodes
    Lx = zeros(Float64, m.p + 1, q + 1)
    Lxp = zeros(Float64, m.p + 1, q + 1)
    for i in 1:m.p+1
        L, dL = lagrange_eval(fd.gll_p, fd.gll_v.nodes[i])
        Lx[i, :] .= L
        Lxp[i, :] .= dL
    end
    corrx = zeros(Float64, m.Nscalar)
    corrz = zeros(Float64, m.Nscalar)
    cnt = zeros(Int, m.Nscalar)
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            gp = element_pres_gids(m, ex, ez)
            for j in 1:m.p+1
                for i in 1:m.p+1
                    gpx = 0.0
                    gpz = 0.0
                    l = 0
                    for bb in 1:q+1
                        for aa in 1:q+1
                            l += 1
                            ph = phi[gp[l]]
                            gpx += ph * (2.0 / hx) * Lxp[i, aa] * Lx[j, bb]
                            gpz += ph * (2.0 / hze) * Lx[i, aa] * Lxp[j, bb]
                        end
                    end
                    k = (j - 1) * (m.p + 1) + i
                    gg = g[k]
                    corrx[gg] += gpx
                    corrz[gg] += gpz
                    cnt[gg] += 1
                end
            end
        end
    end
    uxp = copy(ux)
    uzp = copy(uz)
    @inbounds for g in 1:m.Nscalar
        c = cnt[g]
        c == 0 && continue
        uxp[g] -= corrx[g] / c
        uzp[g] -= corrz[g] / c
    end
    # restore the free-slip / rigid constraints
    @inbounds for g in 1:m.Nscalar
        dm.red_ux[g] == 0 && (uxp[g] = 0.0)
        dm.red_uz[g] == 0 && (uzp[g] = 0.0)
    end
    db = max_divergence(m, fd, ux, uz)
    da = max_divergence(m, fd, uxp, uzp)
    verbose && @printf("  projection: max|div| %.2e -> %.2e\n", db, da)
    return uxp, uzp, db, da
end

"""Pressure mass matrix: integral phi_i phi_j (Q_q space)."""
function assemble_pressure_mass(m::Mesh, fd::FEData)
    np = (m.q + 1)^2
    I = Int[]; J = Int[]; V = Float64[]
    Kl = zeros(Float64, np, np)
    q = m.q; hx = m.hx
    pp = fd.phi_p
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        for ex in 0:m.nxe-1
            gp = element_pres_gids(m, ex, ez)
            fill!(Kl, 0.0)
            for qy in 1:fd.nq, qx in 1:fd.nq
                w = fd.gl_w[qx] * fd.gl_w[qy] * hx * hze / 4.0
                for l in 1:np
                    b = div(l - 1, q + 1) + 1
                    a = mod(l - 1, q + 1) + 1
                    ps = pp[a, qx] * pp[b, qy]
                    for l2 in 1:np
                        b2 = div(l2 - 1, q + 1) + 1
                        a2 = mod(l2 - 1, q + 1) + 1
                        Kl[l, l2] += ps * pp[a2, qx] * pp[b2, qy] * w
                    end
                end
            end
            for l in 1:np, l2 in 1:np
                v = Kl[l, l2]
                v == 0.0 && continue
                push!(I, gp[l]); push!(J, gp[l2]); push!(V, v)
            end
        end
    end
    return sparse(I, J, V, m.Npres, m.Npres)
end
"""Viscosity-weighted pressure mass matrix: integral (1/eta) phi_i phi_j."""
function assemble_pressure_mass_visc(m::Mesh, fd::FEData, pr::Params, theta::Vector{Float64})
    np = (m.q + 1)^2
    nv = (m.p + 1)^2
    I = Int[]; J = Int[]; V = Float64[]
    Kl = zeros(Float64, np, np)
    q = m.q; p = m.p; hx = m.hx
    pp = fd.phi_p; pv = fd.phi_v
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez); ze0 = z_left(m, ez)
        for ex in 0:m.nxe-1
            gp = element_pres_gids(m, ex, ez)
            gv = element_vel_gids(m, ex, ez)
            fill!(Kl, 0.0)
            for qy in 1:fd.nq, qx in 1:fd.nq
                z = ze0 + 0.5 * (fd.gl_x[qy] + 1.0) * hze
                Tloc = Tbar(z, pr.H)
                for j in 1:p+1, i in 1:p+1
                    k = (j - 1) * (p + 1) + i
                    Tloc += theta[gv[k]] * pv[i, qx] * pv[j, qy]
                end
                eta, _ = eta_of_T(Tloc, pr)
                w = fd.gl_w[qx] * fd.gl_w[qy] * hx * hze / 4.0 / eta
                for l in 1:np
                    b = div(l - 1, q + 1) + 1
                    a = mod(l - 1, q + 1) + 1
                    ps = pp[a, qx] * pp[b, qy]
                    for l2 in 1:np
                        b2 = div(l2 - 1, q + 1) + 1
                        a2 = mod(l2 - 1, q + 1) + 1
                        Kl[l, l2] += ps * pp[a2, qx] * pp[b2, qy] * w
                    end
                end
            end
            for l in 1:np, l2 in 1:np
                v = Kl[l, l2]
                v == 0.0 && continue
                push!(I, gp[l]); push!(J, gp[l2]); push!(V, v)
            end
        end
    end
    return sparse(I, J, V, m.Npres, m.Npres)
end