# 1D discontinuous-Galerkin prototype (stage B groundwork).
#
# Solves the SIPG discretisation of  -kappa u'' = f  on [0,1] with Dirichlet
# boundary conditions.  The scheme is the 1D version of the 2D DG-SEM energy
# prototype: element-wise Q_p, upwind/penalty fluxes at faces, Nitsche
# (SIPG) Dirichlet boundaries.  References: Arnold, Brezzi, Cockburn & Marini
# (2002), "Unified analysis of discontinuous Galerkin methods for elliptic
# problems", SIAM J. Numer. Anal. 39, 1749-1779; Hesthaven & Warburton (2008),
# Nodal Discontinuous Galerkin Methods, ch. 7.
#
# The penalty is sigma = C (p+1)^2 / h with C >= 1 (Arnold et al. 2002).  The
# default C = 50 is the smallest value in {1,5,10,20,50} that gives the optimal
# order p+1 for p = 1,2,3 in the manufactured-solution test; it is calibrated by
# that test, not chosen by feel.

struct DG1D
    ne::Int
    p::Int
    h::Float64
    xi::Vector{Float64}     # GLL nodes on [-1,1]
    w::Vector{Float64}      # GLL weights
    D::Matrix{Float64}      # differentiation matrix, D[i,j] = L_j'(xi_i)
end

"""Build a 1D DG mesh with `ne` elements and polynomial degree `p`."""
function make_dg1d(ne::Int, p::Int)
    ne >= 1 && p >= 1 || throw(ArgumentError("ne and p must be >= 1"))
    rule = gll_rule(p)
    n = p + 1
    D = zeros(Float64, n, n)
    for i in 1:n
        _, dL = lagrange_eval(rule, rule.nodes[i])
        for j in 1:n
            D[i, j] = dL[j]
        end
    end
    DG1D(ne, p, 1.0 / ne, copy(rule.nodes), copy(rule.weights), D)
end

"""
    assemble_dg1d_poisson(dg, f, uD; kappa=1.0, sigma_C=5.0)

Return `(A, b)` for the SIPG discretisation of `-kappa u'' = f` with
`u(0)=uD(0)`, `u(1)=uD(1)`.  `A` is symmetric.
"""
function assemble_dg1d_poisson(dg::DG1D, f::Function, uD::Function;
                               kappa::Float64=1.0, sigma_C::Float64=50.0)
    ne, p, h = dg.ne, dg.p, dg.h
    n = p + 1
    N = ne * n
    A = zeros(Float64, N, N)
    b = zeros(Float64, N)
    W = dg.w
    sig = sigma_C * (p + 1)^2 / h
    Ke = (2.0 / h) * kappa .* (dg.D' * Diagonal(W) * dg.D)
    # volume terms and load
    for e in 1:ne
        idx = (e-1)*n .+ (1:n)
        A[idx, idx] .+= Ke
        x0 = (e - 1) * h
        for q in 1:n
            xq = x0 + 0.5 * (dg.xi[q] + 1.0) * h
            b[idx[q]] += 0.5 * h * W[q] * f(xq)
        end
    end
    # interior faces (SIPG): -{du.n}[v] - {dv.n}[u] + sigma [u][v]
    for e in 1:(ne-1)
        iL = (e - 1) * n + n
        iR = e * n + 1
        idxL = (e - 1) * n .+ (1:n)
        idxR = e * n .+ (1:n)
        cL = (2.0 / h) .* dg.D[n, :]
        cR = (2.0 / h) .* dg.D[1, :]
        # term 1: -{du.n}[v] with the SAME normal on both sides
        A[iL, idxL] .+= -0.5 .* kappa .* cL
        A[iL, idxR] .+= -0.5 .* kappa .* cR
        A[iR, idxL] .+= +0.5 .* kappa .* cL
        A[iR, idxR] .+= +0.5 .* kappa .* cR
        # term 2: -{dv.n}[u]; v'(face) involves all basis functions of the
        # element, so this contributes to every row and only to columns iL/iR
        A[idxL, iL] .+= -0.5 .* kappa .* cL
        A[idxL, iR] .+= +0.5 .* kappa .* cL
        A[idxR, iL] .+= -0.5 .* kappa .* cR
        A[idxR, iR] .+= +0.5 .* kappa .* cR
        # penalty
        A[iL, iL] += sig
        A[iL, iR] -= sig
        A[iR, iL] -= sig
        A[iR, iR] += sig
    end
    # Dirichlet boundaries (Nitsche)
    cL = (2.0 / h) .* dg.D[1, :]       # derivative at x=0 (element 1, node 1)
    u0 = uD(0.0)
    for i in 1:n
        A[i, 1] += kappa * cL[i]        # -int v' n u  with n=-1
        b[i] += kappa * cL[i] * u0
    end
    for j in 1:n
        A[1, j] += kappa * cL[j]        # -int u' n v  with n=-1
    end
    A[1, 1] += sig
    b[1] += sig * u0
    cR = (2.0 / h) .* dg.D[n, :]       # derivative at x=1 (last element, node n)
    iN = N
    u1 = uD(1.0)
    for i in 1:n
        A[iN - n + i, iN] += -kappa * cR[i]   # -int v' n u, n=+1
        b[iN - n + i] += -kappa * cR[i] * u1
    end
    for j in 1:n
        A[iN, iN - n + j] += -kappa * cR[j]   # -int u' n v, n=+1
    end
    A[iN, iN] += sig
    b[iN] += sig * u1
    return A, b
end

"""Solve the SIPG Poisson problem and return the nodal solution."""
function solve_dg1d_poisson(dg::DG1D, f::Function, uD::Function;
                            kappa::Float64=1.0, sigma_C::Float64=50.0)
    A, b = assemble_dg1d_poisson(dg, f, uD; kappa=kappa, sigma_C=sigma_C)
    return A \ b
end

"""L2 error of a DG solution against an exact function.

The norm is integrated with a GLL rule of degree p+5 (exact for the degree-2p
integrand), not with the p+1 solution nodes: the latter under-integrates the
error for p >= 2 and would fake a lower convergence order.
"""
function dg1d_l2_error(dg::DG1D, u::Vector{Float64}, exact::Function)
    n = dg.p + 1
    h = dg.h
    basis = gll_rule(dg.p)
    quad = gll_rule(dg.p + 5)
    err2 = 0.0
    for e in 1:dg.ne
        x0 = (e - 1) * h
        for q in eachindex(quad.nodes)
            xiq = quad.nodes[q]
            L, _ = lagrange_eval(basis, xiq)
            uh = 0.0
            for i in 1:n
                uh += L[i] * u[(e-1)*n + i]
            end
            xq = x0 + 0.5 * (xiq + 1.0) * h
            err2 += 0.5 * h * quad.weights[q] * (uh - exact(xq))^2
        end
    end
    return sqrt(err2)
end
