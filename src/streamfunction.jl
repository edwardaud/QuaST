# Divergence-free velocity recovery via a high-order streamfunction.
#
# For free-slip walls, psi = 0 on the whole boundary is the correct
# streamfunction condition: it gives u.n = 0 on each wall while leaving the
# tangential component free.  We solve
#
#     -Laplacian psi = omega,   omega = d_x u_z - d_z u_x,
#
# with psi in Q_s (s = p+1) and homogeneous Dirichlet data, then set
#
#     u_x = d_z psi,   u_z = -d_x psi.
#
# Since both derivatives are taken on the same Q_s polynomial, div u = 0
# identically (in exact arithmetic) at the quadrature points.

struct ScalarSpace
    s::Int
    nx::Int
    nz::Int
    N::Int
    rule::GLLRule
    nq::Int
    gl_x::Vector{Float64}
    gl_w::Vector{Float64}
    phi::Matrix{Float64}
    dphi::Matrix{Float64}
end

"""Velocity values at every element quadrature point (corrected or raw)."""
struct QuadVelocity
    ux::Array{Float64,4}   # [ex+1, ez+1, qx, qy]
    uz::Array{Float64,4}
end

function make_scalar_space(m::Mesh, s::Int; nq::Int=0)
    s >= 1 || throw(ArgumentError("s must be >= 1"))
    nq == 0 && (nq = s + 2)
    rule = gll_rule(s)
    x, w = gauss_legendre(nq)
    phi = zeros(Float64, s + 1, nq)
    dphi = zeros(Float64, s + 1, nq)
    for k in 1:nq
        L, dL = lagrange_eval(rule, x[k])
        phi[:, k] .= L
        dphi[:, k] .= dL
    end
    return ScalarSpace(s, m.nxe * s + 1, m.nze * s + 1,
                       (m.nxe * s + 1) * (m.nze * s + 1),
                       rule, nq, x, w, phi, dphi)
end

function element_scalar_gids(m::Mesh, ss::ScalarSpace, ex::Int, ez::Int)
    n = (ss.s + 1)^2
    g = Vector{Int}(undef, n)
    k = 0
    for j in 1:ss.s+1
        iz = ez * ss.s + j
        for i in 1:ss.s+1
            k += 1
            g[k] = (iz - 1) * ss.nx + (ex * ss.s + i)
        end
    end
    return g
end

"""Dirichlet map for psi = 0 on all four walls."""
function make_psi_map(ss::ScalarSpace)
    red = zeros(Int, ss.N)
    n = 0
    for iz in 1:ss.nz
        for ix in 1:ss.nx
            (ix == 1 || ix == ss.nx || iz == 1 || iz == ss.nz) && continue
            n += 1
            red[(iz - 1) * ss.nx + ix] = n
        end
    end
    return red, n
end

"""
Recover a divergence-free velocity from the Stokes velocity by solving the
streamfunction Poisson problem on Q_{p+1}.  Returns
`(ux_new, uz_new, psi, div_before, div_after)`.
"""
function streamfunction_velocity(m::Mesh, fd::FEData, dm::DofMap,
                                 ux::Vector{Float64}, uz::Vector{Float64};
                                 s::Int=0, verbose::Bool=false)
    ss = make_scalar_space(m, s == 0 ? m.p + 1 : s)
    red, nfree = make_psi_map(ss)
    sdeg = ss.s
    nv = (m.p + 1)^2
    ns = (sdeg + 1)^2
    I = Int[]; J = Int[]; V = Float64[]
    est = m.nxe * m.nze * ns * ns
    sizehint!(I, est); sizehint!(J, est); sizehint!(V, est)
    rhs = zeros(Float64, ss.N)
    Kl = zeros(Float64, ns, ns)
    phil = zeros(Float64, ns)
    gxl = zeros(Float64, ns)
    gzl = zeros(Float64, ns)
    hx = m.hx
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        ze0 = z_left(m, ez)
        for ex in 0:m.nxe-1
            gu = element_vel_gids(m, ex, ez)
            gs = element_scalar_gids(m, ss, ex, ez)
            fill!(Kl, 0.0)
            for qy in 1:ss.nq
                z = ze0 + 0.5 * (ss.gl_x[qy] + 1.0) * hze
                for qx in 1:ss.nq
                    x = ex * hx + 0.5 * (ss.gl_x[qx] + 1.0) * hx
                    # vorticity omega = d_x u_z - d_z u_x
                    dux_dz = 0.0
                    duz_dx = 0.0
                    k = 0
                    for j in 1:m.p+1
                        for i in 1:m.p+1
                            k += 1
                            dux_dz += ux[gu[k]] * fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hze)
                            duz_dx += uz[gu[k]] * fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / hx)
                        end
                    end
                    omega = duz_dx - dux_dz
                    # scalar basis
                    k = 0
                    for j in 1:sdeg+1
                        py = ss.phi[j, qy]
                        dpy = ss.dphi[j, qy]
                        for i in 1:sdeg+1
                            k += 1
                            phil[k] = ss.phi[i, qx] * py
                            gxl[k] = ss.dphi[i, qx] * py * (2.0 / hx)
                            gzl[k] = ss.phi[i, qx] * dpy * (2.0 / hze)
                        end
                    end
                    w = ss.gl_w[qx] * ss.gl_w[qy] * hx * hze / 4.0
                    for kk in 1:ns
                        rhs[gs[kk]] += omega * phil[kk] * w
                        for ll in 1:ns
                            Kl[kk, ll] += (gxl[kk] * gxl[ll] + gzl[kk] * gzl[ll]) * w
                        end
                    end
                end
            end
            for kk in 1:ns
                for ll in 1:ns
                    v = Kl[kk, ll]
                    v != 0.0 && (push!(I, gs[kk]); push!(J, gs[ll]); push!(V, v))
                end
            end
        end
    end
    K = sparse(I, J, V, ss.N, ss.N)
    free = [i for i in 1:ss.N if red[i] > 0]
    Kred = K[free, free]
    bred = rhs[free]
    psired = Kred \ bred
    psi = zeros(Float64, ss.N)
    psi[free] = psired

    # evaluate u = curl psi at the velocity nodes
    Lv = zeros(Float64, m.p + 1, sdeg + 1)
    dLv = zeros(Float64, m.p + 1, sdeg + 1)
    for i in 1:m.p+1
        L, dL = lagrange_eval(ss.rule, fd.gll_v.nodes[i])
        Lv[i, :] .= L
        dLv[i, :] .= dL
    end
    uxn = zeros(Float64, m.Nscalar)
    uzn = zeros(Float64, m.Nscalar)
    cnt = zeros(Int, m.Nscalar)
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        for ex in 0:m.nxe-1
            gu = element_vel_gids(m, ex, ez)
            gs = element_scalar_gids(m, ss, ex, ez)
            for j in 1:m.p+1
                for i in 1:m.p+1
                    uxq = 0.0
                    uzq = 0.0
                    l = 0
                    for bb in 1:sdeg+1
                        for aa in 1:sdeg+1
                            l += 1
                            ph = psi[gs[l]]
                            uxq += ph * Lv[i, aa] * dLv[j, bb] * (2.0 / hze)
                            uzq -= ph * dLv[i, aa] * Lv[j, bb] * (2.0 / hx)
                        end
                    end
                    k = (j - 1) * (m.p + 1) + i
                    gg = gu[k]
                    uxn[gg] += uxq
                    uzn[gg] += uzq
                    cnt[gg] += 1
                end
            end
        end
    end
    ux_new = zeros(Float64, m.Nscalar)
    uz_new = zeros(Float64, m.Nscalar)
    @inbounds for g in 1:m.Nscalar
        c = cnt[g]
        c == 0 && continue
        ux_new[g] = uxn[g] / c
        uz_new[g] = uzn[g] / c
    end
    @inbounds for g in 1:m.Nscalar
        dm.red_ux[g] == 0 && (ux_new[g] = 0.0)
        dm.red_uz[g] == 0 && (uz_new[g] = 0.0)
    end
    db = max_divergence(m, fd, ux, uz)
    da = max_divergence(m, fd, ux_new, uz_new)
    verbose && @printf("  streamfunction: max|div| %.2e -> %.2e  npsi=%d\n",
                       db, da, nfree)
    return ux_new, uz_new, psi, db, da
end

"""
Same streamfunction recovery, but evaluated at the **quadrature points** and
returned as a `QuadVelocity`.  `u = (d_z psi, -d_x psi)` is discretely
divergence-free and has zero normal velocity on all walls because psi=0 there.
Use this for the energy advection when the weak `B u=0` is not enough.
"""
function streamfunction_quads(m::Mesh, fd::FEData, dm::DofMap,
                              ux::Vector{Float64}, uz::Vector{Float64};
                              s::Int=0, verbose::Bool=false)
    ss = make_scalar_space(m, s == 0 ? m.p + 1 : s)
    red, nfree = make_psi_map(ss)
    sdeg = ss.s
    ns = (sdeg + 1)^2
    I = Int[]; J = Int[]; V = Float64[]
    est = m.nxe * m.nze * ns * ns
    sizehint!(I, est); sizehint!(J, est); sizehint!(V, est)
    rhs = zeros(Float64, ss.N)
    Kl = zeros(Float64, ns, ns)
    phil = zeros(Float64, ns)
    gxl = zeros(Float64, ns)
    gzl = zeros(Float64, ns)
    hx = m.hx
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        ze0 = z_left(m, ez)
        for ex in 0:m.nxe-1
            gu = element_vel_gids(m, ex, ez)
            gs = element_scalar_gids(m, ss, ex, ez)
            fill!(Kl, 0.0)
            for qy in 1:ss.nq
                for qx in 1:ss.nq
                    dux_dz = 0.0
                    duz_dx = 0.0
                    k = 0
                    for j in 1:m.p+1
                        for i in 1:m.p+1
                            k += 1
                            dux_dz += ux[gu[k]] * fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hze)
                            duz_dx += uz[gu[k]] * fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / hx)
                        end
                    end
                    omega = duz_dx - dux_dz
                    k = 0
                    for j in 1:sdeg+1
                        py = ss.phi[j, qy]
                        dpy = ss.dphi[j, qy]
                        for i in 1:sdeg+1
                            k += 1
                            phil[k] = ss.phi[i, qx] * py
                            gxl[k] = ss.dphi[i, qx] * py * (2.0 / hx)
                            gzl[k] = ss.phi[i, qx] * dpy * (2.0 / hze)
                        end
                    end
                    w = ss.gl_w[qx] * ss.gl_w[qy] * hx * hze / 4.0
                    for kk in 1:ns
                        rhs[gs[kk]] += omega * phil[kk] * w
                        for ll in 1:ns
                            Kl[kk, ll] += (gxl[kk] * gxl[ll] + gzl[kk] * gzl[ll]) * w
                        end
                    end
                end
            end
            for kk in 1:ns
                for ll in 1:ns
                    v = Kl[kk, ll]
                    v != 0.0 && (push!(I, gs[kk]); push!(J, gs[ll]); push!(V, v))
                end
            end
        end
    end
    K = sparse(I, J, V, ss.N, ss.N)
    free = [i for i in 1:ss.N if red[i] > 0]
    psi = zeros(Float64, ss.N)
    psi[free] = K[free, free] \ rhs[free]

    uxq = Array{Float64,4}(undef, m.nxe, m.nze, ss.nq, ss.nq)
    uzq = Array{Float64,4}(undef, m.nxe, m.nze, ss.nq, ss.nq)
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        for ex in 0:m.nxe-1
            gs = element_scalar_gids(m, ss, ex, ez)
            for qy in 1:ss.nq
                for qx in 1:ss.nq
                    gpx = 0.0
                    gpz = 0.0
                    l = 0
                    for bb in 1:sdeg+1
                        for aa in 1:sdeg+1
                            l += 1
                            ph = psi[gs[l]]
                            gpx += ph * ss.dphi[aa, qx] * ss.phi[bb, qy] * (2.0 / hx)
                            gpz += ph * ss.phi[aa, qx] * ss.dphi[bb, qy] * (2.0 / hze)
                        end
                    end
                    uxq[ex+1, ez+1, qx, qy] = gpz
                    uzq[ex+1, ez+1, qx, qy] = -gpx
                end
            end
        end
    end
    db = max_divergence(m, fd, ux, uz)
    verbose && @printf("  streamfunction-quads: max|div| raw %.2e  npsi=%d\n", db, nfree)
    return QuadVelocity(uxq, uzq), db
end

"""
Volume RMS velocity and max speed from a `QuadVelocity`, using the same Gauss
quadrature as the energy assembly.  Use this when the physical velocity is the
divergence-free projection `u_psi`, not the raw Stokes field.
"""
function quad_stats(m::Mesh, fd::FEData, velq::QuadVelocity)
    s = 0.0
    umax = 0.0
    nqx = size(velq.ux, 3)
    nqy = size(velq.ux, 4)
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        for ex in 0:m.nxe-1
            for qy in 1:nqy
                for qx in 1:nqx
                    ux = velq.ux[ex+1, ez+1, qx, qy]
                    uz = velq.uz[ex+1, ez+1, qx, qy]
                    w = fd.gl_w[qx] * fd.gl_w[qy] * m.hx * hze / 4.0
                    s += (ux * ux + uz * uz) * w
                    umax = max(umax, hypot(ux, uz))
                end
            end
        end
    end
    return sqrt(s / m.Lx), umax
end

"""
Base-state advection integral ``I = int u_z dTbar/dz dV``.

For H=0, dTbar=-1 and I=0 by continuity; for H!=0, dTbar is not constant and
the steady global energy balance is

    Nu_top - Nu_bot = H - I,

not H.  This term is essential for the H=5 mixed-heating configuration.
"""
function base_advection_quad(m::Mesh, fd::FEData, pr::Params, velq::QuadVelocity)
    I = 0.0
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        ze0 = z_left(m, ez)
        for ex in 0:m.nxe-1
            for qy in 1:fd.nq
                z = ze0 + 0.5 * (fd.gl_x[qy] + 1.0) * hze
                gz = dTbar(z, pr.H)
                for qx in 1:fd.nq
                    w = fd.gl_w[qx] * fd.gl_w[qy] * m.hx * hze / 4.0
                    I += velq.uz[ex+1, ez+1, qx, qy] * gz * w
                end
            end
        end
    end
    return I
end

"""
Q_{p+1} potential projection: solve the Neumann problem

    Laplacian phi = div u,   dphi/dn = 0,

on Q_{p+1}, then u <- u - grad phi.  grad phi is Q_p and can represent the
pointwise divergence residual that the Q_q pressure projection cannot see.
Returns `(ux_new, uz_new, phi, div_before, div_after)`.
"""
function potential_projection(m::Mesh, fd::FEData, dm::DofMap,
                              ux::Vector{Float64}, uz::Vector{Float64};
                              s::Int=0, verbose::Bool=false)
    ss = make_scalar_space(m, s == 0 ? m.p + 1 : s)
    sdeg = ss.s
    nv = (m.p + 1)^2
    ns = (sdeg + 1)^2
    I = Int[]; J = Int[]; V = Float64[]
    est = m.nxe * m.nze * ns * ns
    sizehint!(I, est); sizehint!(J, est); sizehint!(V, est)
    b = zeros(Float64, ss.N)
    Kl = zeros(Float64, ns, ns)
    phil = zeros(Float64, ns)
    gxl = zeros(Float64, ns)
    gzl = zeros(Float64, ns)
    hx = m.hx
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        ze0 = z_left(m, ez)
        for ex in 0:m.nxe-1
            gu = element_vel_gids(m, ex, ez)
            gs = element_scalar_gids(m, ss, ex, ez)
            fill!(Kl, 0.0)
            for qy in 1:ss.nq
                z = ze0 + 0.5 * (ss.gl_x[qy] + 1.0) * hze
                for qx in 1:ss.nq
                    # div u
                    divu = 0.0
                    k = 0
                    for j in 1:m.p+1
                        py = fd.phi_v[j, qy]
                        for i in 1:m.p+1
                            k += 1
                            gx = fd.dphi_v[i, qx] * py * (2.0 / hx)
                            gz = fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hze)
                            divu += ux[gu[k]] * gx + uz[gu[k]] * gz
                        end
                    end
                    k = 0
                    for j in 1:sdeg+1
                        py = ss.phi[j, qy]
                        dpy = ss.dphi[j, qy]
                        for i in 1:sdeg+1
                            k += 1
                            phil[k] = ss.phi[i, qx] * py
                            gxl[k] = ss.dphi[i, qx] * py * (2.0 / hx)
                            gzl[k] = ss.phi[i, qx] * dpy * (2.0 / hze)
                        end
                    end
                    w = ss.gl_w[qx] * ss.gl_w[qy] * hx * hze / 4.0
                    for kk in 1:ns
                        b[gs[kk]] += -phil[kk] * divu * w
                        for ll in 1:ns
                            Kl[kk, ll] += (gxl[kk] * gxl[ll] + gzl[kk] * gzl[ll]) * w
                        end
                    end
                end
            end
            for kk in 1:ns
                for ll in 1:ns
                    v = Kl[kk, ll]
                    v != 0.0 && (push!(I, gs[kk]); push!(J, gs[ll]); push!(V, v))
                end
            end
        end
    end
    K = sparse(I, J, V, ss.N, ss.N)
    pin = 1
    free = [i for i in 1:ss.N if i != pin]
    phired = K[free, free] \ b[free]
    phi = zeros(Float64, ss.N)
    phi[free] = phired

    # grad phi at the velocity nodes
    Lv = zeros(Float64, m.p + 1, sdeg + 1)
    dLv = zeros(Float64, m.p + 1, sdeg + 1)
    for i in 1:m.p+1
        L, dL = lagrange_eval(ss.rule, fd.gll_v.nodes[i])
        Lv[i, :] .= L
        dLv[i, :] .= dL
    end
    corx = zeros(Float64, m.Nscalar)
    corz = zeros(Float64, m.Nscalar)
    cnt = zeros(Int, m.Nscalar)
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        for ex in 0:m.nxe-1
            gu = element_vel_gids(m, ex, ez)
            gs = element_scalar_gids(m, ss, ex, ez)
            for j in 1:m.p+1
                for i in 1:m.p+1
                    gpx = 0.0
                    gpz = 0.0
                    l = 0
                    for bb in 1:sdeg+1
                        for aa in 1:sdeg+1
                            l += 1
                            ph = phi[gs[l]]
                            gpx += ph * dLv[i, aa] * Lv[j, bb] * (2.0 / hx)
                            gpz += ph * Lv[i, aa] * dLv[j, bb] * (2.0 / hze)
                        end
                    end
                    k = (j - 1) * (m.p + 1) + i
                    gg = gu[k]
                    corx[gg] += gpx
                    corz[gg] += gpz
                    cnt[gg] += 1
                end
            end
        end
    end
    ux_new = copy(ux)
    uz_new = copy(uz)
    @inbounds for g in 1:m.Nscalar
        c = cnt[g]
        c == 0 && continue
        ux_new[g] -= corx[g] / c
        uz_new[g] -= corz[g] / c
    end
    @inbounds for g in 1:m.Nscalar
        dm.red_ux[g] == 0 && (ux_new[g] = 0.0)
        dm.red_uz[g] == 0 && (uz_new[g] = 0.0)
    end
    db = max_divergence(m, fd, ux, uz)
    da = max_divergence(m, fd, ux_new, uz_new)
    verbose && @printf("  potential(Q%d): max|div| %.2e -> %.2e\n", sdeg, db, da)
    return ux_new, uz_new, phi, db, da
end

"""
Q5 potential correction evaluated AT THE QUADRATURE POINTS.

This is the version that can actually be used by the energy assembly: no nodal
averaging is involved, so `u - grad phi` is pointwise divergence-free at the
Gauss points (up to the Q5 solve).  `ux,uz` are only used to form `div u`.
"""
function potential_correction(m::Mesh, fd::FEData, dm::DofMap,
                              ux::Vector{Float64}, uz::Vector{Float64};
                              s::Int=0, verbose::Bool=false)
    ss = make_scalar_space(m, s == 0 ? m.p + 1 : s)
    sdeg = ss.s
    ns = (sdeg + 1)^2
    I = Int[]; J = Int[]; V = Float64[]
    est = m.nxe * m.nze * ns * ns
    sizehint!(I, est); sizehint!(J, est); sizehint!(V, est)
    b = zeros(Float64, ss.N)
    Kl = zeros(Float64, ns, ns)
    phil = zeros(Float64, ns)
    gxl = zeros(Float64, ns)
    gzl = zeros(Float64, ns)
    hx = m.hx
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        ze0 = z_left(m, ez)
        for ex in 0:m.nxe-1
            gu = element_vel_gids(m, ex, ez)
            gs = element_scalar_gids(m, ss, ex, ez)
            fill!(Kl, 0.0)
            for qy in 1:ss.nq
                z = ze0 + 0.5 * (ss.gl_x[qy] + 1.0) * hze
                for qx in 1:ss.nq
                    divu = 0.0
                    k = 0
                    for j in 1:m.p+1
                        py = fd.phi_v[j, qy]
                        for i in 1:m.p+1
                            k += 1
                            gx = fd.dphi_v[i, qx] * py * (2.0 / hx)
                            gz = fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hze)
                            divu += ux[gu[k]] * gx + uz[gu[k]] * gz
                        end
                    end
                    k = 0
                    for j in 1:sdeg+1
                        py = ss.phi[j, qy]
                        dpy = ss.dphi[j, qy]
                        for i in 1:sdeg+1
                            k += 1
                            phil[k] = ss.phi[i, qx] * py
                            gxl[k] = ss.dphi[i, qx] * py * (2.0 / hx)
                            gzl[k] = ss.phi[i, qx] * dpy * (2.0 / hze)
                        end
                    end
                    w = ss.gl_w[qx] * ss.gl_w[qy] * hx * hze / 4.0
                    for kk in 1:ns
                        b[gs[kk]] += -phil[kk] * divu * w
                        for ll in 1:ns
                            Kl[kk, ll] += (gxl[kk] * gxl[ll] + gzl[kk] * gzl[ll]) * w
                        end
                    end
                end
            end
            for kk in 1:ns
                for ll in 1:ns
                    v = Kl[kk, ll]
                    v != 0.0 && (push!(I, gs[kk]); push!(J, gs[ll]); push!(V, v))
                end
            end
        end
    end
    K = sparse(I, J, V, ss.N, ss.N)
    pin = 1
    free = [i for i in 1:ss.N if i != pin]
    phi = zeros(Float64, ss.N)
    phi[free] = K[free, free] \ b[free]

    uxq = Array{Float64,4}(undef, m.nxe, m.nze, ss.nq, ss.nq)
    uzq = Array{Float64,4}(undef, m.nxe, m.nze, ss.nq, ss.nq)
    hx = m.hx
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        for ex in 0:m.nxe-1
            gu = element_vel_gids(m, ex, ez)
            gs = element_scalar_gids(m, ss, ex, ez)
            for qy in 1:ss.nq
                for qx in 1:ss.nq
                    uraw = 0.0
                    urawz = 0.0
                    k = 0
                    for j in 1:m.p+1
                        py = fd.phi_v[j, qy]
                        for i in 1:m.p+1
                            k += 1
                            bb = fd.phi_v[i, qx] * py
                            uraw += ux[gu[k]] * bb
                            urawz += uz[gu[k]] * bb
                        end
                    end
                    gpx = 0.0
                    gpz = 0.0
                    l = 0
                    for bb in 1:sdeg+1
                        for aa in 1:sdeg+1
                            l += 1
                            ph = phi[gs[l]]
                            gpx += ph * ss.dphi[aa, qx] * ss.phi[bb, qy] * (2.0 / hx)
                            gpz += ph * ss.phi[aa, qx] * ss.dphi[bb, qy] * (2.0 / hze)
                        end
                    end
                    uxq[ex+1, ez+1, qx, qy] = uraw - gpx
                    uzq[ex+1, ez+1, qx, qy] = urawz - gpz
                end
            end
        end
    end
    db = max_divergence(m, fd, ux, uz)
    verbose && @printf("  potential-correction(Q%d): raw max|div| %.2e (quadrature values ready)\n",
                       sdeg, db)
    return QuadVelocity(uxq, uzq), db
end
