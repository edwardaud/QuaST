# 2D discontinuous-Galerkin prototype (stage B): steady advection on a uniform
# tensor mesh.
#
# Solves  u . grad T = 0  with upwind face fluxes and inflow Dirichlet data.
# The face is a line, so each face term carries the 1D GLL mass matrix along
# the face (the 2D analogue of the point face in dg1d.jl).  This module is the
# first half of the energy-equation prototype; the SIPG diffusion operator is
# added next and must reuse the same face machinery.
#
# References: Hesthaven & Warburton (2008), Nodal Discontinuous Galerkin
# Methods, ch. 6-7 (upwind fluxes); the SIPG diffusion fluxes follow Arnold et
# al. (2002).

struct DG2D
    nex::Int
    ney::Int
    p::Int
    hx::Float64
    hy::Float64
    xi::Vector{Float64}
    w::Vector{Float64}
    D::Matrix{Float64}
end

function make_dg2d(nex::Int, ney::Int, p::Int)
    nex >= 1 && ney >= 1 && p >= 1 || throw(ArgumentError("nex, ney, p must be >= 1"))
    r = gll_rule(p)
    n = p + 1
    D = zeros(Float64, n, n)
    for i in 1:n
        _, dL = lagrange_eval(r, r.nodes[i])
        for j in 1:n
            D[i, j] = dL[j]
        end
    end
    DG2D(nex, ney, p, 1.0 / nex, 1.0 / ney, copy(r.nodes), copy(r.weights), D)
end

@inline _elem(dg::DG2D, ex::Int, ey::Int) = (ey - 1) * dg.nex + ex
@inline _base(dg::DG2D, ex::Int, ey::Int) = (_elem(dg, ex, ey) - 1) * (dg.p + 1)^2
@inline _lid(dg::DG2D, i::Int, j::Int) = (j - 1) * (dg.p + 1) + i
@inline _gid(dg::DG2D, ex::Int, ey::Int, i::Int, j::Int) = _base(dg, ex, ey) + _lid(dg, i, j)
@inline _xnode(dg::DG2D, ex::Int, i::Int) = (ex - 1) * dg.hx + 0.5 * (dg.xi[i] + 1.0) * dg.hx
@inline _ynode(dg::DG2D, ey::Int, j::Int) = (ey - 1) * dg.hy + 0.5 * (dg.xi[j] + 1.0) * dg.hy

"""
    assemble_dg2d_advection(dg, ux, uy, inflow)

Return `(A, b)` for the steady upwind DG discretisation of `u . grad T = 0`
with the inflow value `inflow(x,y)` on boundaries where `u.n < 0`.  `A` is
non-symmetric (upwind) as expected.
"""
function assemble_dg2d_advection(dg::DG2D, ux::Float64, uy::Float64,
                                 inflow::Function)
    n = dg.p + 1
    N = dg.nex * dg.ney * n^2
    A = zeros(Float64, N, N)
    b = zeros(Float64, N)
    w = dg.w
    M1 = Diagonal(w)
    # weak form: -int_K T (u . grad v); the face terms below are +int (u.n) That v
    DW = dg.D' * M1
    Kx = kron(M1, DW)            # d/dx on the (i fast, j slow) ordering
    Ky = kron(DW, M1)            # d/dy
    Ke = -ux * 0.5 * dg.hy .* Kx .- uy * 0.5 * dg.hx .* Ky
    for ey in 1:dg.ney, ex in 1:dg.nex
        base = _base(dg, ex, ey)
        idx = base .+ (1:n^2)
        A[idx, idx] .+= Ke
    end
    # vertical faces (normal +x): upwind on the L/R trace
    hy2 = 0.5 * dg.hy
    for ey in 1:dg.ney, ex in 1:(dg.nex-1)
        for j in 1:n
            wj = hy2 * w[j]
            gL = _gid(dg, ex, ey, n, j)
            gR = _gid(dg, ex + 1, ey, 1, j)
            if ux >= 0.0
                A[gL, gL] += ux * wj
                A[gR, gL] -= ux * wj
            else
                A[gL, gR] += ux * wj
                A[gR, gR] -= ux * wj
            end
        end
    end
    # horizontal faces (normal +y)
    hx2 = 0.5 * dg.hx
    for ey in 1:(dg.ney-1), ex in 1:dg.nex
        for i in 1:n
            wi = hx2 * w[i]
            gB = _gid(dg, ex, ey, i, n)
            gT = _gid(dg, ex, ey + 1, i, 1)
            if uy >= 0.0
                A[gB, gB] += uy * wi
                A[gT, gB] -= uy * wi
            else
                A[gB, gT] += uy * wi
                A[gT, gT] -= uy * wi
            end
        end
    end
    # domain boundaries
    for ey in 1:dg.ney
        # left (x=0), outward normal -x: inflow when ux > 0
        for j in 1:n
            g = _gid(dg, 1, ey, 1, j)
            wn = hy2 * w[j]
            if ux >= 0.0
                b[g] += ux * wn * inflow(0.0, _ynode(dg, ey, j))
            else
                A[g, g] += -ux * wn
            end
        end
        # right (x=1), outward normal +x: inflow when ux < 0
        for j in 1:n
            g = _gid(dg, dg.nex, ey, n, j)
            wn = hy2 * w[j]
            if ux < 0.0
                b[g] += -ux * wn * inflow(1.0, _ynode(dg, ey, j))
            else
                A[g, g] += ux * wn
            end
        end
    end
    for ex in 1:dg.nex
        # bottom (y=0), outward normal -y: inflow when uy > 0
        for i in 1:n
            g = _gid(dg, ex, 1, i, 1)
            wn = hx2 * w[i]
            if uy >= 0.0
                b[g] += uy * wn * inflow(_xnode(dg, ex, i), 0.0)
            else
                A[g, g] += -uy * wn
            end
        end
        # top (y=1), outward normal +y: inflow when uy < 0
        for i in 1:n
            g = _gid(dg, ex, dg.ney, i, n)
            wn = hx2 * w[i]
            if uy < 0.0
                b[g] += -uy * wn * inflow(_xnode(dg, ex, i), 1.0)
            else
                A[g, g] += uy * wn
            end
        end
    end
    return A, b
end

"""Evaluate a DG2D field at (x,y) by locating the element and summing the basis."""
function dg2d_eval(dg::DG2D, u::Vector{Float64}, x::Float64, y::Float64)
    ex = clamp(floor(Int, x / dg.hx) + 1, 1, dg.nex)
    ey = clamp(floor(Int, y / dg.hy) + 1, 1, dg.ney)
    xi = 2.0 * (x - (ex - 1) * dg.hx) / dg.hx - 1.0
    eta = 2.0 * (y - (ey - 1) * dg.hy) / dg.hy - 1.0
    Lx, _ = lagrange_eval(gll_rule(dg.p), xi)
    Ly, _ = lagrange_eval(gll_rule(dg.p), eta)
    val = 0.0
    for j in 1:(dg.p+1), i in 1:(dg.p+1)
        val += u[_gid(dg, ex, ey, i, j)] * Lx[i] * Ly[j]
    end
    return val
end




"""2D SIPG Poisson: -kappa laplace(T)=f with Dirichlet data on all four sides."""
function assemble_dg2d_poisson(dg::DG2D, f::Function, uD::Function;
                               kappa::Float64=1.0, sigma_C::Float64=50.0)
    n = dg.p + 1
    N = dg.nex * dg.ney * n^2
    A = zeros(Float64, N, N)
    b = zeros(Float64, N)
    w = dg.w
    W = Diagonal(w)
    DW = dg.D' * W * dg.D
    sig = sigma_C * (dg.p + 1)^2 / min(dg.hx, dg.hy)
    Ke = kappa .* ((dg.hy / dg.hx) .* kron(W, DW) .+ (dg.hx / dg.hy) .* kron(DW, W))
    for ey in 1:dg.ney, ex in 1:dg.nex
        idx = _base(dg, ex, ey) .+ (1:n^2)
        A[idx, idx] .+= Ke
        for j in 1:n, i in 1:n
            x = _xnode(dg, ex, i); y = _ynode(dg, ey, j)
            b[_gid(dg, ex, ey, i, j)] += 0.25 * dg.hx * dg.hy * w[i] * w[j] * f(x, y)
        end
    end
    hy2 = 0.5 * dg.hy
    for ey in 1:dg.ney, ex in 1:(dg.nex-1)
        cL = (2.0 / dg.hx) .* dg.D[n, :]
        cR = (2.0 / dg.hx) .* dg.D[1, :]
        for j in 1:n
            wf = hy2 * w[j]
            iL = _gid(dg, ex, ey, n, j)
            iR = _gid(dg, ex + 1, ey, 1, j)
            for i in 1:n
                gLi = _gid(dg, ex, ey, i, j)
                gRi = _gid(dg, ex + 1, ey, i, j)
                A[iL, gLi] += -0.5 * wf * cL[i]
                A[iL, gRi] += -0.5 * wf * cR[i]
                A[iR, gLi] += +0.5 * wf * cL[i]
                A[iR, gRi] += +0.5 * wf * cR[i]
                A[gLi, iL] += -0.5 * wf * cL[i]
                A[gLi, iR] += +0.5 * wf * cL[i]
                A[gRi, iL] += -0.5 * wf * cR[i]
                A[gRi, iR] += +0.5 * wf * cR[i]
            end
            A[iL, iL] += sig * wf
            A[iL, iR] -= sig * wf
            A[iR, iL] -= sig * wf
            A[iR, iR] += sig * wf
        end
    end
    hx2 = 0.5 * dg.hx
    for ey in 1:(dg.ney-1), ex in 1:dg.nex
        cB = (2.0 / dg.hy) .* dg.D[n, :]
        cT = (2.0 / dg.hy) .* dg.D[1, :]
        for i in 1:n
            wf = hx2 * w[i]
            iB = _gid(dg, ex, ey, i, n)
            iT = _gid(dg, ex, ey + 1, i, 1)
            for j in 1:n
                gBj = _gid(dg, ex, ey, i, j)
                gTj = _gid(dg, ex, ey + 1, i, j)
                A[iB, gBj] += -0.5 * wf * cB[j]
                A[iB, gTj] += -0.5 * wf * cT[j]
                A[iT, gBj] += +0.5 * wf * cB[j]
                A[iT, gTj] += +0.5 * wf * cT[j]
                A[gBj, iB] += -0.5 * wf * cB[j]
                A[gBj, iT] += +0.5 * wf * cB[j]
                A[gTj, iB] += -0.5 * wf * cT[j]
                A[gTj, iT] += +0.5 * wf * cT[j]
            end
            A[iB, iB] += sig * wf
            A[iB, iT] -= sig * wf
            A[iT, iB] -= sig * wf
            A[iT, iT] += sig * wf
        end
    end
    for ey in 1:dg.ney
        c = (2.0 / dg.hx) .* dg.D[1, :]
        for j in 1:n
            wf = hy2 * w[j]
            iF = _gid(dg, 1, ey, 1, j)
            Td = uD(0.0, _ynode(dg, ey, j))
            for i in 1:n
                gi = _gid(dg, 1, ey, i, j)
                A[iF, gi] += wf * c[i]
                A[gi, iF] += wf * c[i]
                b[gi] += wf * c[i] * Td
            end
            A[iF, iF] += sig * wf
            b[iF] += sig * wf * Td
        end
        c = (2.0 / dg.hx) .* dg.D[n, :]
        for j in 1:n
            wf = hy2 * w[j]
            iF = _gid(dg, dg.nex, ey, n, j)
            Td = uD(1.0, _ynode(dg, ey, j))
            for i in 1:n
                gi = _gid(dg, dg.nex, ey, i, j)
                A[iF, gi] += -wf * c[i]
                A[gi, iF] += -wf * c[i]
                b[gi] += -wf * c[i] * Td
            end
            A[iF, iF] += sig * wf
            b[iF] += sig * wf * Td
        end
    end
    for ex in 1:dg.nex
        c = (2.0 / dg.hy) .* dg.D[1, :]
        for i in 1:n
            wf = hx2 * w[i]
            iF = _gid(dg, ex, 1, i, 1)
            Td = uD(_xnode(dg, ex, i), 0.0)
            for j in 1:n
                gj = _gid(dg, ex, 1, i, j)
                A[iF, gj] += wf * c[j]
                A[gj, iF] += wf * c[j]
                b[gj] += wf * c[j] * Td
            end
            A[iF, iF] += sig * wf
            b[iF] += sig * wf * Td
        end
        c = (2.0 / dg.hy) .* dg.D[n, :]
        for i in 1:n
            wf = hx2 * w[i]
            iF = _gid(dg, ex, dg.ney, i, n)
            Td = uD(_xnode(dg, ex, i), 1.0)
            for j in 1:n
                gj = _gid(dg, ex, dg.ney, i, j)
                A[iF, gj] += -wf * c[j]
                A[gj, iF] += -wf * c[j]
                b[gj] += -wf * c[j] * Td
            end
            A[iF, iF] += sig * wf
            b[iF] += sig * wf * Td
        end
    end
    return A, b
end
