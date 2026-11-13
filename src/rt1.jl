# RT1 on a reference quad, V = Q_{2,1} x Q_{1,2}.
#
# 12 DOFs: 2 normal-flux moments per edge + 4 interior moments.
# Edge functionals (outward normal):
#   e1 left   x=-1, ux, q=(1, eta)
#   e2 right  x=+1, ux, q=(1, eta)
#   e3 bottom z=-1, uz, q=(1, xi)
#   e4 top    z=+1, uz, q=(1, xi)
# Interior (area) moments: int ux*xi, int ux*eta, int uz*xi, int uz*eta.
#
# The basis is obtained by inverting the 12x12 matrix of these functionals
# acting on the tensor monomials.

const RT1_UX_EXP = ((0,0),(1,0),(2,0),(0,1),(1,1),(2,1))
const RT1_UZ_EXP = ((0,0),(1,0),(0,1),(0,2),(1,1),(1,2))

@inline function _mono(a::Int, b::Int, xi::Float64, eta::Float64)
    return (xi^a) * (eta^b)
end

struct RT1Basis
    cux::Matrix{Float64}  # 6 x 12 coefficients
    cuz::Matrix{Float64}  # 6 x 12
end

"""Build the RT1 reference basis by inverting the moment matrix."""
function build_rt1_basis()
    gx, gw = gauss_legendre(3)
    M = zeros(Float64, 12, 12)
    # edge functionals
    for (f, e, k) in ((1,1,0),(2,1,1),(3,2,0),(4,2,1),
                      (5,3,0),(6,3,1),(7,4,0),(8,4,1))
        sgn = e == 1 || e == 3 ? -1.0 : 1.0
        for (col, (a,b)) in enumerate(RT1_UX_EXP)
            integ = 0.0
            if e == 1 || e == 2
                xfix = e == 1 ? -1.0 : 1.0
                for (t, w) in zip(gx, gw)
                    q = k == 0 ? 1.0 : t
                    integ += _mono(a, b, xfix, t) * q * w
                end
                M[f, col] += sgn * integ
            end
        end
        for (col, (a,b)) in enumerate(RT1_UZ_EXP)
            integ = 0.0
            if e == 3 || e == 4
                yfix = e == 3 ? -1.0 : 1.0
                for (t, w) in zip(gx, gw)
                    q = k == 0 ? 1.0 : t
                    integ += _mono(a, b, t, yfix) * q * w
                end
                M[f, 6 + col] += sgn * integ
            end
        end
    end
    # interior moments 9..12, chosen to separate the edge-determined modes
    for (qx, wx) in zip(gx, gw)
        for (qy, wy) in zip(gx, gw)
            w = wx * wy
            q9 = (1.0 - qx^2)
            q10 = qy * (1.0 - qx^2)
            q11 = (1.0 - qy^2)
            q12 = qx * (1.0 - qy^2)
            for (col, (a,b)) in enumerate(RT1_UX_EXP)
                ux = _mono(a, b, qx, qy)
                M[9, col] += ux * q9 * w
                M[10, col] += ux * q10 * w
            end
            for (col, (a,b)) in enumerate(RT1_UZ_EXP)
                uz = _mono(a, b, qx, qy)
                M[11, 6 + col] += uz * q11 * w
                M[12, 6 + col] += uz * q12 * w
            end
        end
    end
    C = inv(M)
    return RT1Basis(C[1:6, :], C[7:12, :])
end

const RT1_REF = build_rt1_basis()

"""RT1 basis values and reference derivatives at (xi, eta)."""
function rt1_basis(xi::Float64, eta::Float64)
    ux = zeros(Float64, 12); uz = zeros(Float64, 12)
    ux_xi = zeros(Float64, 12); ux_eta = zeros(Float64, 12)
    uz_xi = zeros(Float64, 12); uz_eta = zeros(Float64, 12)
    for j in 1:12
        for (k, (a, b)) in enumerate(RT1_UX_EXP)
            ux[j] += RT1_REF.cux[k, j] * _mono(a, b, xi, eta)
            a > 0 && (ux_xi[j] += RT1_REF.cux[k, j] * a * _mono(a-1, b, xi, eta))
            b > 0 && (ux_eta[j] += RT1_REF.cux[k, j] * b * _mono(a, b-1, xi, eta))
        end
        for (k, (a, b)) in enumerate(RT1_UZ_EXP)
            uz[j] += RT1_REF.cuz[k, j] * _mono(a, b, xi, eta)
            a > 0 && (uz_xi[j] += RT1_REF.cuz[k, j] * a * _mono(a-1, b, xi, eta))
            b > 0 && (uz_eta[j] += RT1_REF.cuz[k, j] * b * _mono(a, b-1, xi, eta))
        end
    end
    return ux, uz, ux_xi, ux_eta, uz_xi, uz_eta
end

"""Edge/moment functional of RT1 basis `j`: int_edge (u.n) q ds on reference edge e."""
function rt1_edge_moment(e::Int, k::Int, j::Int)
    gx, gw = gauss_legendre(3)
    integ = 0.0
    for (t, w) in zip(gx, gw)
        xi, eta = e == 1 ? (-1.0, t) :
                  e == 2 ? (1.0, t) :
                  e == 3 ? (t, -1.0) : (t, 1.0)
        ux, uz, _, _, _, _ = rt1_basis(xi, eta)
        nx, nz = rt0_edge_normal(e)
        q = k == 0 ? 1.0 : t
        integ += (ux[j] * nx + uz[j] * nz) * q * w
    end
    return integ
end

# ---------------------------------------------------------------------------
# Global RT1 map: 2 edge moments per edge + 4 interior DOFs per element.
# ---------------------------------------------------------------------------

struct RT1Map
    rt0::RT0Map
    red_edge::Vector{Int}   # length 2*nedge
    nfree_edge::Int
    red_int::Vector{Int}    # length 4*ne, all free
    ne::Int
    nv::Int
end

function make_rt1_map(m::Mesh, bc::Symbol=:free)
    r0 = make_rt0_map(m, bc)
    red_edge = zeros(Int, 2 * r0.nedge)
    nf = 0
    for id in 1:r0.nedge
        r0.red[id] == 0 && continue
        nf += 1; red_edge[2*(id-1)+1] = nf
        nf += 1; red_edge[2*(id-1)+2] = nf
    end
    ne = m.nxe * m.nze
    red_int = collect(1:(4*ne))
    return RT1Map(r0, red_edge, nf, red_int, ne, nf + 4*ne)
end

"""
RT1-DG0 Stokes assembly.  Piola mapping on axis-aligned rectangles:
  ux = (2/hz) ux_hat,  uz = (2/hx) uz_hat,
  d_x ux = (4/(hx hz)) d_xi ux_hat, etc.
"""
function assemble_rt1_stokes(m::Mesh, fd::FEData, pr::Params,
                             theta::Vector{Float64};
                             bc::Symbol=:free, verbose::Bool=false,
                             return_system::Bool=false,
                             stab::Float64=0.0,
                             body_fx::Union{Nothing,Function}=nothing,
                             body_fz::Union{Nothing,Function}=nothing,
                             pressure_order::Int=1,
                             tangent_penalty::Float64=100.0,
                             hydrostatic::Bool=true)
    rmap = make_rt1_map(m, bc)
    nv = rmap.nv
    ne = rmap.ne
    npl = pressure_order == 1 ? 3 : 6
    pred = zeros(Int, npl*ne)
    np = 0
    for idx in 0:(npl*ne-1)
        idx == 0 && continue
        np += 1
        pred[idx+1] = np
    end
    ntot = nv + np
    II = Int[]; JJ = Int[]; VV = Float64[]
    f = zeros(Float64, ntot)
    etasum = zeros(Float64, ne); earea = zeros(Float64, ne)
    eta_min = Inf; eta_max = -Inf; nclip = 0
    hx = m.hx
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez); z0 = z_left(m, ez); z1 = z0 + hze
        for ex in 0:m.nxe-1
            x0 = ex * hx; x1 = x0 + hx
            pe = (ez*m.nxe + ex) + 1
            eids = (rmap.rt0.idv[ex+1, ez+1], rmap.rt0.idv[ex+2, ez+1],
                    rmap.rt0.idh[ex+1, ez+1], rmap.rt0.idh[ex+1, ez+2])
            signs = rt0_edge_signs()
            cols = zeros(Int, 12); fac = ones(Float64, 12)
            for e in 1:4
                id = eids[e]
                for k in 0:1
                    j = 2*(e-1) + k + 1
                    gdof = 2*(id-1) + k + 1
                    cols[j] = rmap.red_edge[gdof]
                    fac[j] = signs[e]
                end
            end
            for j in 9:12
                cols[j] = rmap.nfree_edge + (ez*m.nxe + ex)*4 + (j - 8)
            end
            area = hx * hze
            for qy in 1:fd.nq
                eta = fd.gl_x[qy]
                z = z0 + 0.5 * (eta + 1.0) * hze
                for qx in 1:fd.nq
                    xi = fd.gl_x[qx]
                    x = x0 + 0.5 * (xi + 1.0) * hx
                    Tloc = Tbar(z, pr.H)
                    thloc = 0.0
                    kv = 0
                    for j in 1:m.p+1
                        for i in 1:m.p+1
                            kv += 1
                            thv = theta[(ez*m.p+j-1)*m.nvx + (ex*m.p+i)] *
                                  fd.phi_v[i, qx] * fd.phi_v[j, qy]
                            Tloc += thv
                            thloc += thv
                        end
                    end
                    etaT, clipped = eta_of_T(Tloc, pr)
                    eta_min = min(eta_min, etaT); eta_max = max(eta_max, etaT)
                    clipped && (nclip += 1)
                    w = fd.gl_w[qx] * fd.gl_w[qy] * area / 4.0
                    etasum[pe] += etaT * w
                    earea[pe] = area
                    ux, uz, uxi, ueta, zxi, zeta = rt1_basis(xi, eta)
                    uxPh = (2.0 / hze) .* ux
                    uzPh = (2.0 / hx) .* uz
                    dxx = (4.0 / (hx*hze)) .* uxi
                    dxz = (4.0 / (hze*hze)) .* ueta
                    dzx = (4.0 / (hx*hx)) .* zxi
                    dzz = (4.0 / (hx*hze)) .* zeta
                    exx = dxx
                    ezz = dzz
                    exz = 0.5 .* (dxz .+ dzx)
                    divv = dxx .+ dzz
                    for i in 1:12
                        cols[i] == 0 && continue
                        for j in 1:12
                            cols[j] == 0 && continue
                            val = 2.0 * etaT *
                                (exx[i]*exx[j] + ezz[i]*ezz[j] + 2.0*exz[i]*exz[j]) * w
                            val += stab * (uxPh[i]*uxPh[j] + uzPh[i]*uzPh[j]) * w
                            push!(II, cols[i]); push!(JJ, cols[j])
                            push!(VV, fac[i]*fac[j]*val)
                        end
                        if body_fz === nothing
                            body_T = hydrostatic ? thloc : Tloc
                            f[cols[i]] += fac[i] * pr.Ra * body_T * uzPh[i] * w
                        else
                            bfx = body_fx === nothing ? 0.0 : body_fx(x, z)
                            f[cols[i]] += fac[i] * (bfx * uxPh[i] + body_fz(x, z) * uzPh[i]) * w
                        end
                    end
                    xc = x0 + 0.5 * hx
                    zc = z0 + 0.5 * hze
                    for k in 0:(npl-1)
                        idx = npl*(pe-1) + k + 1
                        pred[idx] == 0 && continue
                        dx = (x - xc) / hx
                        dz = (z - zc) / hze
                        pk = k == 0 ? 1.0 :
                             k == 1 ? dx :
                             k == 2 ? dz :
                             k == 3 ? dx*dx :
                             k == 4 ? dx*dz : dz*dz
                        row = nv + pred[idx]
                        for i in 1:12
                            cols[i] == 0 && continue
                            val = -fac[i] * divv[i] * pk * w
                            push!(II, row); push!(JJ, cols[i]); push!(VV, val)
                            push!(II, cols[i]); push!(JJ, row); push!(VV, val)
                        end
                    end
                end
            end
        end
    end
    etae = etasum ./ max.(earea, 1.0e-30)
    if tangent_penalty > 0
        function local_cf(exx, ezz)
            eids = (rmap.rt0.idv[exx+1, ezz+1], rmap.rt0.idv[exx+2, ezz+1],
                    rmap.rt0.idh[exx+1, ezz+1], rmap.rt0.idh[exx+1, ezz+2])
            cls = zeros(Int, 12); fc = ones(Float64, 12)
            sg = rt0_edge_signs()
            for e in 1:4, k in 0:1
                j = 2*(e-1)+k+1
                gd = 2*(eids[e]-1)+k+1
                cls[j] = rmap.red_edge[gd]; fc[j] = sg[e]
            end
            for j in 9:12
                cls[j] = rmap.nfree_edge + (ezz*m.nxe+exx)*4 + (j-8)
            end
            return cls, fc
        end
        gpx, gpw = gauss_legendre(3)
        # vertical faces: tangential uz
        for ezz in 0:m.nze-1
            hzf = z_size(m, ezz)
            for exx in 1:m.nxe-1
                cL, fL = local_cf(exx-1, ezz)
                cR, fR = local_cf(exx, ezz)
                etaf = 0.5 * (etae[ezz*m.nxe+exx] + etae[ezz*m.nxe+exx+1])
                for (tt, ww) in zip(gpx, gpw)
                    uxL, uzL, uxiL, uetaL, zxiL, zetaL = rt1_basis(1.0, tt)
                    uxR, uzR, uxiR, uetaR, zxiR, zetaR = rt1_basis(-1.0, tt)
                    tL = (2.0/hx) .* uzL; tR = (2.0/hx) .* uzR
                    exzL = 0.5 .* ((4.0/hzf^2) .* uetaL .+ (4.0/hx^2) .* zxiL)
                    exzR = 0.5 .* ((4.0/hzf^2) .* uetaR .+ (4.0/hx^2) .* zxiR)
                    gF = 0.5 .* vcat(fL .* exzL, .-fR .* exzR)
                    vF = vcat(fL .* tL, .-fR .* tR)
                    cF = vcat(cL, cR)
                    wf = ww * hzf / 2.0
                    pen = tangent_penalty * etaf / hzf
                    for I in 1:24
                        cF[I] == 0 && continue
                        for J in 1:24
                            cF[J] == 0 && continue
                            val = -gF[I] * vF[J] - vF[I] * gF[J] + pen * vF[I] * vF[J]
                            push!(II, cF[I]); push!(JJ, cF[J])
                            push!(VV, val * wf)
                        end
                    end
                end
            end
        end
        # horizontal faces: tangential ux (different hz on the two sides)
        for exx in 0:m.nxe-1
            for ezz in 1:m.nze-1
                hzb = z_size(m, ezz-1); hzt = z_size(m, ezz)
                cB, fB = local_cf(exx, ezz-1)
                cT, fT = local_cf(exx, ezz)
                etaf = 0.5 * (etae[(ezz-1)*m.nxe+exx+1] + etae[ezz*m.nxe+exx+1])
                for (tt, ww) in zip(gpx, gpw)
                    uxB, uzB, uxiB, uetaB, zxiB, zetaB = rt1_basis(tt, 1.0)
                    uxT, uzT, uxiT, uetaT, zxiT, zetaT = rt1_basis(tt, -1.0)
                    tB = (2.0/hzb) .* uxB; tT = (2.0/hzt) .* uxT
                    exzB = 0.5 .* ((4.0/hzb^2) .* uetaB .+ (4.0/hx^2) .* zxiB)
                    exzT = 0.5 .* ((4.0/hzt^2) .* uetaT .+ (4.0/hx^2) .* zxiT)
                    gF = 0.5 .* vcat(fB .* exzB, .-fT .* exzT)
                    vF = vcat(fB .* tB, .-fT .* tT)
                    cF = vcat(cB, cT)
                    wf = ww * hx / 2.0
                    pen = tangent_penalty * etaf / hx
                    for I in 1:24
                        cF[I] == 0 && continue
                        for J in 1:24
                            cF[J] == 0 && continue
                            val = -gF[I] * vF[J] - vF[I] * gF[J] + pen * vF[I] * vF[J]
                            push!(II, cF[I]); push!(JJ, cF[J])
                            push!(VV, val * wf)
                        end
                    end
                end
            end
        end
    end
    A = sparse(II, JJ, VV, ntot, ntot)
    A = A + sparse(1:ntot, 1:ntot, fill(1.0e-12, ntot), ntot, ntot)
    rhs = vcat(f[1:nv], zeros(Float64, np))
    if return_system
        return A, rhs, rmap
    end
    dscale = 1.0 ./ sqrt.(max.(vec(sum(abs.(A), dims=1)), 1.0e-30))
    D = sparse(1:ntot, 1:ntot, dscale, ntot, ntot)
    Ae = D * A * D
    be = dscale .* rhs
    ye = Ae \ be
    sol = dscale .* ye
    res = norm(A * sol - rhs) / max(norm(rhs), 1.0e-30)
    U = sol[1:nv]

    uxq = Array{Float64,4}(undef, m.nxe, m.nze, fd.nq, fd.nq)
    uzq = Array{Float64,4}(undef, m.nxe, m.nze, fd.nq, fd.nq)
    for ez in 0:m.nze-1
        hze = z_size(m, ez); z0 = z_left(m, ez)
        for ex in 0:m.nxe-1
            x0 = ex * hx
            eids = (rmap.rt0.idv[ex+1, ez+1], rmap.rt0.idv[ex+2, ez+1],
                    rmap.rt0.idh[ex+1, ez+1], rmap.rt0.idh[ex+1, ez+2])
            signs = rt0_edge_signs()
            cols = zeros(Int, 12); fac = ones(Float64, 12)
            for e in 1:4, k in 0:1
                j = 2*(e-1)+k+1
                gdof = 2*(eids[e]-1)+k+1
                cols[j] = rmap.red_edge[gdof]; fac[j] = signs[e]
            end
            for j in 9:12
                cols[j] = rmap.nfree_edge + (ez*m.nxe+ex)*4 + (j-8)
            end
            Ul = zeros(Float64, 12)
            for j in 1:12
                cols[j] == 0 && continue
                Ul[j] = fac[j] * U[cols[j]]
            end
            for qy in 1:fd.nq
                eta = fd.gl_x[qy]; z = z0 + 0.5*(eta+1.0)*hze
                for qx in 1:fd.nq
                    xi = fd.gl_x[qx]; x = x0 + 0.5*(xi+1.0)*hx
                    ux, uz, _, _, _, _ = rt1_basis(xi, eta)
                    uxq[ex+1, ez+1, qx, qy] = sum(Ul[j]*(2.0/hze)*ux[j] for j in 1:12)
                    uzq[ex+1, ez+1, qx, qy] = sum(Ul[j]*(2.0/hx)*uz[j] for j in 1:12)
                end
            end
        end
    end
    velq = QuadVelocity(uxq, uzq)
    ux = zeros(Float64, m.Nscalar); uz = zeros(Float64, m.Nscalar)
    for ez in 0:m.nze-1
        hze = z_size(m, ez); z0 = z_left(m, ez)
        for ex in 0:m.nxe-1
            x0 = ex * hx
            eids = (rmap.rt0.idv[ex+1, ez+1], rmap.rt0.idv[ex+2, ez+1],
                    rmap.rt0.idh[ex+1, ez+1], rmap.rt0.idh[ex+1, ez+2])
            signs = rt0_edge_signs()
            cols = zeros(Int, 12); fac = ones(Float64, 12)
            for e in 1:4, k in 0:1
                j = 2*(e-1)+k+1
                gdof = 2*(eids[e]-1)+k+1
                cols[j] = rmap.red_edge[gdof]; fac[j] = signs[e]
            end
            for j in 9:12
                cols[j] = rmap.nfree_edge + (ez*m.nxe+ex)*4 + (j-8)
            end
            Ul = zeros(Float64, 12)
            for j in 1:12
                cols[j] == 0 && continue
                Ul[j] = fac[j] * U[cols[j]]
            end
            for j in 1:m.p+1
                eta = fd.gll_v.nodes[j]; z = z0 + 0.5*(eta+1)*hze
                for i in 1:m.p+1
                    xi = fd.gll_v.nodes[i]; x = x0 + 0.5*(xi+1)*hx
                    uxb, uzb, _, _, _, _ = rt1_basis(xi, eta)
                    g = (ez*m.p+j-1)*m.nvx + (ex*m.p+i)
                    ux[g] = sum(Ul[jj]*(2.0/hze)*uxb[jj] for jj in 1:12)
                    uz[g] = sum(Ul[jj]*(2.0/hx)*uzb[jj] for jj in 1:12)
                end
            end
        end
    end
    verbose && @printf("  rt1-Stokes: nv=%d np=%d res=%.2e eta=[%.3e,%.3e]\n",
                       nv, np, res, eta_min, eta_max)
    return StokesSolution(ux, uz, zeros(Float64, m.Npres), res,
                          eta_min, eta_max, nclip, velq)
end

function solve_stokes_rt1(m::Mesh, fd::FEData, pr::Params,
                          theta::Vector{Float64}, dm::DofMap;
                          res_bar::Float64=1.0e-8, linear::Symbol=:direct)
    pr.bc === :free ||
        throw(ArgumentError("RT1 velocity space currently supports free-slip only"))
    return assemble_rt1_stokes(m, fd, pr, theta; bc=pr.bc)
end







