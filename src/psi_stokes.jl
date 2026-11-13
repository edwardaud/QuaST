# Streamfunction (psi) velocity space: exact pointwise divergence-free velocity.
#
# u = curl psi = (d_z psi, -d_x psi),  psi in Q_{p+1},  psi = 0 on all walls.
# The wall condition gives u.n = 0; free-slip tangential traction is the natural
# boundary condition of the weak form.  Pressure is eliminated because the test
# space is divergence-free:
#
#   a(psi, phi) = int 2 eta eps(curl psi) : eps(curl phi) dV
#   b(phi)      = - Ra int T d_x phi dV
#
# With psi = phi_i(x) phi_j(z):
#   eps_xx = d2xz psi,  eps_zz = -d2xz psi,
#   eps_xz = 0.5 (d2z psi - d2x psi),
# so the local kernel is 4 eta (d2xz_i d2xz_j + eps_xz_i eps_xz_j).

function assemble_psi_stokes(m::Mesh, fd::FEData, pr::Params,
                             theta::Vector{Float64};
                             s::Int=0, verbose::Bool=false,
                             return_system::Bool=false)
    sdeg0 = s == 0 ? m.p + 1 : s
    ss = make_scalar_space(m, sdeg0; nq=sdeg0 + 4)
    red, nfree = make_psi_map(ss)
    sdeg = ss.s
    ns = (sdeg + 1)^2
    nv = (m.p + 1)^2
    hx = m.hx
    # second derivatives of the 1D Q_{p+1} basis at the quadrature points
    d2x = zeros(Float64, sdeg + 1, ss.nq)
    for q in 1:ss.nq
        _, _, d2 = lagrange_d2(ss.rule, ss.gl_x[q])
        d2x[:, q] .= d2
    end
    # Q4 temperature basis evaluated at the Q_{p+1} quadrature points
    thphi = zeros(Float64, m.p + 1, ss.nq)
    for q in 1:ss.nq
        L, _ = lagrange_eval(fd.gll_v, ss.gl_x[q])
        thphi[:, q] .= L
    end
    I = Int[]; J = Int[]; V = Float64[]
    est = m.nxe * m.nze * ns * ns
    sizehint!(I, est); sizehint!(J, est); sizehint!(V, est)
    b = zeros(Float64, ss.N)
    Kl = zeros(Float64, ns, ns)
    phil = zeros(Float64, ns)
    gxl = zeros(Float64, ns)
    gzl = zeros(Float64, ns)
    d2xzl = zeros(Float64, ns)
    d2xl = zeros(Float64, ns)
    d2zl = zeros(Float64, ns)
    exzl = zeros(Float64, ns)
    eta_min = Inf
    eta_max = -Inf
    nclip = 0
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
                    Tloc = Tbar(z, pr.H)
                    kv = 0
                    for j in 1:m.p+1
                        for i in 1:m.p+1
                            kv += 1
                            Tloc += theta[gu[kv]] * thphi[i, qx] * thphi[j, qy]
                        end
                    end
                    eta, clipped = eta_of_T(Tloc, pr)
                    eta_min = min(eta_min, eta)
                    eta_max = max(eta_max, eta)
                    clipped && (nclip += 1)
                    k = 0
                    for j in 1:sdeg+1
                        py = ss.phi[j, qy]
                        dpy = ss.dphi[j, qy]
                        d2py = d2x[j, qy]
                        for i in 1:sdeg+1
                            k += 1
                            phil[k] = ss.phi[i, qx] * py
                            gxl[k] = ss.dphi[i, qx] * py * (2.0 / hx)
                            gzl[k] = ss.phi[i, qx] * dpy * (2.0 / hze)
                            d2xzl[k] = ss.dphi[i, qx] * dpy * (2.0 / hx) * (2.0 / hze)
                            d2xl[k] = d2x[i, qx] * py * (2.0 / hx)^2
                            d2zl[k] = ss.phi[i, qx] * d2py * (2.0 / hze)^2
                            exzl[k] = 0.5 * (d2zl[k] - d2xl[k])
                        end
                    end
                    w = ss.gl_w[qx] * ss.gl_w[qy] * hx * hze / 4.0
                    for kk in 1:ns
                        b[gs[kk]] += -pr.Ra * Tloc * gxl[kk] * w
                        for ll in 1:ns
                            Kl[kk, ll] += 4.0 * eta *
                                (d2xzl[kk] * d2xzl[ll] + exzl[kk] * exzl[ll]) * w
                        end
                    end
                end
            end
            for kk in 1:ns
                rk = red[gs[kk]]
                rk == 0 && continue
                for ll in 1:ns
                    rl = red[gs[ll]]
                    rl == 0 && continue
                    v = Kl[kk, ll]
                    v != 0.0 && (push!(I, rk); push!(J, rl); push!(V, v))
                end
            end
        end
    end
    K = sparse(I, J, V, nfree, nfree)
    K = K + sparse(1:nfree, 1:nfree, fill(1.0e-12, nfree), nfree, nfree)
    free = [i for i in 1:ss.N if red[i] > 0]
    if return_system
        return K, b, free, ss, eta_min, eta_max, nclip
    end
    psired = K \ b[free]
    psi = zeros(Float64, ss.N)
    psi[free] = psired
    res = norm(K * psired - b[free]) / max(norm(b[free]), 1.0e-30)

    # exact curl at every Gauss point
    uxq = Array{Float64,4}(undef, m.nxe, m.nze, ss.nq, ss.nq)
    uzq = Array{Float64,4}(undef, m.nxe, m.nze, ss.nq, ss.nq)
    for ez in 0:m.nze-1
        hze = z_size(m, ez)
        for ex in 0:m.nxe-1
            gs = element_scalar_gids(m, ss, ex, ez)
            for qy in 1:ss.nq
                for qx in 1:ss.nq
                    gpx = 0.0
                    gpz = 0.0
                    k = 0
                    for j in 1:sdeg+1
                        py = ss.phi[j, qy]
                        dpy = ss.dphi[j, qy]
                        for i in 1:sdeg+1
                            k += 1
                            ph = psi[gs[k]]
                            gpx += ph * ss.dphi[i, qx] * py * (2.0 / hx)
                            gpz += ph * ss.phi[i, qx] * dpy * (2.0 / hze)
                        end
                    end
                    uxq[ex+1, ez+1, qx, qy] = gpz
                    uzq[ex+1, ez+1, qx, qy] = -gpx
                end
            end
        end
    end
    velq = QuadVelocity(uxq, uzq)

    # nodal values at the Q_p velocity nodes (for diagnostics/legacy paths)
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
    for ez in 0:m.nze-1
        hze = z_size(m, ez)
        for ex in 0:m.nxe-1
            gu = element_vel_gids(m, ex, ez)
            gs = element_scalar_gids(m, ss, ex, ez)
            for j in 1:m.p+1
                for i in 1:m.p+1
                    ux = 0.0
                    uz = 0.0
                    l = 0
                    for bb in 1:sdeg+1
                        for aa in 1:sdeg+1
                            l += 1
                            ph = psi[gs[l]]
                            ux += ph * Lv[i, aa] * dLv[j, bb] * (2.0 / hze)
                            uz -= ph * dLv[i, aa] * Lv[j, bb] * (2.0 / hx)
                        end
                    end
                    g = gu[(j - 1) * (m.p + 1) + i]
                    uxn[g] += ux
                    uzn[g] += uz
                    cnt[g] += 1
                end
            end
        end
    end
    ux = zeros(Float64, m.Nscalar)
    uz = zeros(Float64, m.Nscalar)
    for g in 1:m.Nscalar
        cnt[g] == 0 && continue
        ux[g] = uxn[g] / cnt[g]
        uz[g] = uzn[g] / cnt[g]
    end
    verbose && @printf("  psi-Stokes: nfree=%d res=%.2e eta=[%.3e,%.3e] clip=%d\n",
                       nfree, res, eta_min, eta_max, nclip)
    return StokesSolution(ux, uz, zeros(Float64, m.Npres), res,
                          eta_min, eta_max, nclip, velq)
end

function solve_stokes_psi(m::Mesh, fd::FEData, pr::Params,
                          theta::Vector{Float64}, dm::DofMap;
                          res_bar::Float64=1.0e-8, linear::Symbol=:direct)
    pr.bc === :free ||
        throw(ArgumentError("psi velocity space currently supports free-slip only"))
    return assemble_psi_stokes(m, fd, pr, theta)
end

"""
Mixed psi-omega system of two second-order H1 equations:

    -div(eta grad omega) = -Ra d_x T,   omega=0 on walls
    -Laplacian psi       = -omega,      psi=0 on walls
    u = curl psi.

This avoids the non-conforming C0 fourth-order operator.  The present vorticity
equation keeps the dominant eta(T) diffusion term; the grad(eta) cross terms of
the variable-viscosity curl are not yet included (next stage).
"""
function assemble_psi_omega_stokes(m::Mesh, fd::FEData, pr::Params,
                                   theta::Vector{Float64};
                                   s::Int=0, verbose::Bool=false,
                                   return_system::Bool=false,
                                   beta_full::Float64=0.0,
                                   nstress::Int=0,
                                   stress_relax::Float64=1.0)
    sdeg0 = s == 0 ? m.p + 1 : s
    ss = make_scalar_space(m, sdeg0; nq=sdeg0 + 4)
    red, nfree = make_psi_map(ss)
    sdeg = ss.s
    ns = (sdeg + 1)^2
    thphi = zeros(Float64, m.p + 1, ss.nq)
    thdphi = zeros(Float64, m.p + 1, ss.nq)
    for q in 1:ss.nq
        L, dL = lagrange_eval(fd.gll_v, ss.gl_x[q])
        thphi[:, q] .= L
        thdphi[:, q] .= dL
    end
    d2q = zeros(Float64, sdeg + 1, ss.nq)
    for q in 1:ss.nq
        _, _, d2 = lagrange_d2(ss.rule, ss.gl_x[q])
        d2q[:, q] .= d2
    end
    I1 = Int[]; J1 = Int[]; V1 = Float64[]
    I2 = Int[]; J2 = Int[]; V2 = Float64[]
    IM = Int[]; JM = Int[]; VM = Float64[]
    bomega = zeros(Float64, ss.N)
    K1l = zeros(Float64, ns, ns)
    K2l = zeros(Float64, ns, ns)
    Ml = zeros(Float64, ns, ns)
    phil = zeros(Float64, ns)
    gxl = zeros(Float64, ns)
    gzl = zeros(Float64, ns)
    d2xzl = zeros(Float64, ns)
    d2xl = zeros(Float64, ns)
    d2zl = zeros(Float64, ns)
    exzl = zeros(Float64, ns)
    eta_min = Inf; eta_max = -Inf; nclip = 0
    for ez in 0:m.nze-1
        hze = z_size(m, ez); ze0 = z_left(m, ez)
        for ex in 0:m.nxe-1
            gu = element_vel_gids(m, ex, ez)
            gs = element_scalar_gids(m, ss, ex, ez)
            fill!(K1l, 0.0); fill!(K2l, 0.0); fill!(Ml, 0.0)
            for qy in 1:ss.nq
                z = ze0 + 0.5 * (ss.gl_x[qy] + 1.0) * hze
                for qx in 1:ss.nq
                    Tloc = Tbar(z, pr.H)
                    dTdx = 0.0
                    dTdz = dTbar(z, pr.H)
                    kv = 0
                    for j in 1:m.p+1
                        for i in 1:m.p+1
                            kv += 1
                            Tloc += theta[gu[kv]] * thphi[i, qx] * thphi[j, qy]
                            dTdx += theta[gu[kv]] * thdphi[i, qx] * thphi[j, qy] * (2.0 / m.hx)
                            dTdz += theta[gu[kv]] * thphi[i, qx] * thdphi[j, qy] * (2.0 / hze)
                        end
                    end
                    eta, clipped = eta_of_T(Tloc, pr)
                    eta_min = min(eta_min, eta); eta_max = max(eta_max, eta)
                    clipped && (nclip += 1)
                    etap = pr.eta_law === :expT ? -log(pr.deta_T) * eta : 0.0
                    detadx = etap * dTdx
                    detadz = etap * dTdz
                    k = 0
                    for j in 1:sdeg+1
                        py = ss.phi[j, qy]; dpy = ss.dphi[j, qy]
                        for i in 1:sdeg+1
                            k += 1
                            phil[k] = ss.phi[i, qx] * py
                            gxl[k] = ss.dphi[i, qx] * py * (2.0 / m.hx)
                            gzl[k] = ss.phi[i, qx] * dpy * (2.0 / hze)
                            d2xzl[k] = ss.dphi[i, qx] * dpy * (2.0 / m.hx) * (2.0 / hze)
                            d2xl[k] = d2q[i, qx] * py * (2.0 / m.hx)^2
                            d2zl[k] = ss.phi[i, qx] * d2q[j, qy] * (2.0 / hze)^2
                            exzl[k] = 0.5 * (d2zl[k] - d2xl[k])
                        end
                    end
                    w = ss.gl_w[qx] * ss.gl_w[qy] * m.hx * hze / 4.0
                    for kk in 1:ns
                        bomega[gs[kk]] += -pr.Ra * Tloc * gxl[kk] * w
                        for ll in 1:ns
                            gg = gxl[kk] * gxl[ll] + gzl[kk] * gzl[ll]
                            exx_l = d2xzl[ll]
                            exz_l = exzl[ll]
                            gx_l = 2.0 * (detadx * exx_l + detadz * exz_l)
                            gz_l = 2.0 * (detadx * exz_l - detadz * exx_l)
                            extra = gx_l * gzl[kk] - gz_l * gxl[kk]
                            K1l[kk, ll] += (eta * gg -
                                (detadx * gxl[ll] + detadz * gzl[ll]) * phil[kk] +
                                beta_full * extra) * w
                            K2l[kk, ll] += gg * w
                            Ml[kk, ll] += phil[kk] * phil[ll] * w
                        end
                    end
                end
            end
            for kk in 1:ns
                rk = red[gs[kk]]; rk == 0 && continue
                for ll in 1:ns
                    rl = red[gs[ll]]; rl == 0 && continue
                    v1 = K1l[kk, ll]
                    v1 != 0.0 && (push!(I1, rk); push!(J1, rl); push!(V1, v1))
                    v2 = K2l[kk, ll]
                    v2 != 0.0 && (push!(I2, rk); push!(J2, rl); push!(V2, v2))
                    vm = Ml[kk, ll]
                    vm != 0.0 && (push!(IM, rk); push!(JM, rl); push!(VM, vm))
                end
            end
        end
    end
    reg = sparse(1:nfree, 1:nfree, fill(1.0e-12, nfree), nfree, nfree)
    K1 = sparse(I1, J1, V1, nfree, nfree) + reg
    K2 = sparse(I2, J2, V2, nfree, nfree) + reg
    M = sparse(IM, JM, VM, nfree, nfree)
    free = [i for i in 1:ss.N if red[i] > 0]
    if return_system
        return K1, K2, M, bomega, free, ss, eta_min, eta_max, nclip
    end
    omred = K1 \ bomega[free]
    omega = zeros(Float64, ss.N)
    omega[free] = omred
    bpsi = -(M * omred)
    psired = K2 \ bpsi
    psi = zeros(Float64, ss.N)
    psi[free] = psired
    res1 = norm(K1 * omred - bomega[free]) / max(norm(bomega[free]), 1.0e-30)
    res2 = norm(K2 * psired - bpsi) / max(norm(bpsi), 1.0e-30)
    res = max(res1, res2)

    # Lagged mixed-stress Picard: g = 2 grad(eta):eps(u) from the previous psi,
    # and -curl(g) is put only on the RHS (no d2 x test in K1).
    bcorr_prev = zeros(Float64, ss.N)
    for it in 1:nstress
        bcorr = zeros(Float64, ss.N)
        for ez in 0:m.nze-1
            hze = z_size(m, ez); ze0 = z_left(m, ez)
            for ex in 0:m.nxe-1
                gu = element_vel_gids(m, ex, ez)
                gs = element_scalar_gids(m, ss, ex, ez)
                for qy in 1:ss.nq
                    z = ze0 + 0.5 * (ss.gl_x[qy] + 1.0) * hze
                    for qx in 1:ss.nq
                        Tloc = Tbar(z, pr.H)
                        dTdx = 0.0
                        dTdz = dTbar(z, pr.H)
                        kv = 0
                        for j in 1:m.p+1
                            for i in 1:m.p+1
                                kv += 1
                                Tloc += theta[gu[kv]] * thphi[i, qx] * thphi[j, qy]
                                dTdx += theta[gu[kv]] * thdphi[i, qx] * thphi[j, qy] * (2.0 / m.hx)
                                dTdz += theta[gu[kv]] * thphi[i, qx] * thdphi[j, qy] * (2.0 / hze)
                            end
                        end
                        eta, _ = eta_of_T(Tloc, pr)
                        etap = pr.eta_law === :expT ? -log(pr.deta_T) * eta : 0.0
                        detadx = etap * dTdx
                        detadz = etap * dTdz
                        d2xz = 0.0; d2x = 0.0; d2z = 0.0
                        k = 0
                        for j in 1:sdeg+1
                            for i in 1:sdeg+1
                                k += 1
                                ph = psi[gs[k]]
                                d2xz += ph * ss.dphi[i, qx] * ss.dphi[j, qy] *
                                        (2.0 / m.hx) * (2.0 / hze)
                                d2x += ph * d2q[i, qx] * ss.phi[j, qy] * (2.0 / m.hx)^2
                                d2z += ph * ss.phi[i, qx] * d2q[j, qy] * (2.0 / hze)^2
                            end
                        end
                        exx = d2xz
                        exz = 0.5 * (d2z - d2x)
                        gx = 2.0 * (detadx * exx + detadz * exz)
                        gz = 2.0 * (detadx * exz - detadz * exx)
                        w = ss.gl_w[qx] * ss.gl_w[qy] * m.hx * hze / 4.0
                        k = 0
                        for j in 1:sdeg+1
                            for i in 1:sdeg+1
                                k += 1
                                gxk = ss.dphi[i, qx] * ss.phi[j, qy] * (2.0 / m.hx)
                                gzk = ss.phi[i, qx] * ss.dphi[j, qy] * (2.0 / hze)
                                bcorr[gs[k]] += (gx * gzk - gz * gxk) * w
                            end
                        end
                    end
                end
            end
        end
        if it == 1
            bcorr .*= stress_relax
        else
            bcorr .= stress_relax .* bcorr .+ (1.0 - stress_relax) .* bcorr_prev
        end
        bcorr_prev .= bcorr
        omred = K1 \ (bomega[free] + bcorr[free])
        omega = zeros(Float64, ss.N)
        omega[free] = omred
        bpsi = -(M * omred)
        psired = K2 \ bpsi
        psi = zeros(Float64, ss.N)
        psi[free] = psired
        res1 = norm(K1 * omred - bomega[free] - bcorr[free]) /
               max(norm(bomega[free] + bcorr[free]), 1.0e-30)
        res2 = norm(K2 * psired - bpsi) / max(norm(bpsi), 1.0e-30)
        res = max(res1, res2)
    end

    uxq = Array{Float64,4}(undef, m.nxe, m.nze, ss.nq, ss.nq)
    uzq = Array{Float64,4}(undef, m.nxe, m.nze, ss.nq, ss.nq)
    for ez in 0:m.nze-1
        hze = z_size(m, ez)
        for ex in 0:m.nxe-1
            gs = element_scalar_gids(m, ss, ex, ez)
            for qy in 1:ss.nq
                for qx in 1:ss.nq
                    gpx = 0.0; gpz = 0.0; k = 0
                    for j in 1:sdeg+1
                        py = ss.phi[j, qy]; dpy = ss.dphi[j, qy]
                        for i in 1:sdeg+1
                            k += 1
                            ph = psi[gs[k]]
                            gpx += ph * ss.dphi[i, qx] * py * (2.0 / m.hx)
                            gpz += ph * ss.phi[i, qx] * dpy * (2.0 / hze)
                        end
                    end
                    uxq[ex+1, ez+1, qx, qy] = gpz
                    uzq[ex+1, ez+1, qx, qy] = -gpx
                end
            end
        end
    end
    velq = QuadVelocity(uxq, uzq)
    Lv = zeros(Float64, m.p + 1, sdeg + 1)
    dLv = zeros(Float64, m.p + 1, sdeg + 1)
    for i in 1:m.p+1
        L, dL = lagrange_eval(ss.rule, fd.gll_v.nodes[i])
        Lv[i, :] .= L; dLv[i, :] .= dL
    end
    uxn = zeros(Float64, m.Nscalar); uzn = zeros(Float64, m.Nscalar)
    cnt = zeros(Int, m.Nscalar)
    for ez in 0:m.nze-1
        hze = z_size(m, ez)
        for ex in 0:m.nxe-1
            gu = element_vel_gids(m, ex, ez)
            gs = element_scalar_gids(m, ss, ex, ez)
            for j in 1:m.p+1
                for i in 1:m.p+1
                    ux = 0.0; uz = 0.0; l = 0
                    for bb in 1:sdeg+1
                        for aa in 1:sdeg+1
                            l += 1
                            ph = psi[gs[l]]
                            ux += ph * Lv[i, aa] * dLv[j, bb] * (2.0 / hze)
                            uz -= ph * dLv[i, aa] * Lv[j, bb] * (2.0 / m.hx)
                        end
                    end
                    g = gu[(j - 1) * (m.p + 1) + i]
                    uxn[g] += ux; uzn[g] += uz; cnt[g] += 1
                end
            end
        end
    end
    ux = zeros(Float64, m.Nscalar); uz = zeros(Float64, m.Nscalar)
    for g in 1:m.Nscalar
        cnt[g] == 0 && continue
        ux[g] = uxn[g] / cnt[g]; uz[g] = uzn[g] / cnt[g]
    end
    verbose && @printf("  psi-omega: nfree=%d res=%.2e/%.2e eta=[%.3e,%.3e]\n",
                       nfree, res1, res2, eta_min, eta_max)
    return StokesSolution(ux, uz, zeros(Float64, m.Npres), res,
                          eta_min, eta_max, nclip, velq)
end

function solve_stokes_psi_omega(m::Mesh, fd::FEData, pr::Params,
                                theta::Vector{Float64}, dm::DofMap;
                                res_bar::Float64=1.0e-8, linear::Symbol=:direct,
                                beta_full::Float64=0.0, nstress::Int=0,
                                stress_relax::Float64=1.0)
    pr.bc === :free ||
        throw(ArgumentError("psi-omega velocity space supports free-slip only"))
    return assemble_psi_omega_stokes(m, fd, pr, theta;
                                     beta_full=beta_full, nstress=nstress,
                                     stress_relax=stress_relax)
end

