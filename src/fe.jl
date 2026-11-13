# Reference-element data: GLL nodes for the nodal basis, Gauss-Legendre points
# for quadrature, and the 1-D Lagrange values/derivatives at those points.
#
# Using Gauss-Legendre (not GLL) quadrature avoids the under-integration of the
# variable-viscosity and advection terms; there is no mass lumping in this code,
# so nothing is gained by GLL quadrature.

struct FEData
    p::Int
    q::Int
    nq::Int
    gll_v::GLLRule
    gll_p::GLLRule
    gl_x::Vector{Float64}
    gl_w::Vector{Float64}
    phi_v::Matrix{Float64}    # (p+1) x nq
    dphi_v::Matrix{Float64}
    phi_p::Matrix{Float64}    # (q+1) x nq
    dphi_p::Matrix{Float64}
end

function make_fedata(p::Int, q::Int; nq::Int=0)
    p >= 2 || throw(ArgumentError("p must be >= 2"))
    q >= 1 || throw(ArgumentError("q must be >= 1"))
    nq == 0 && (nq = p + 2)
    gv = gll_rule(p)
    gp = gll_rule(q)
    gx, gw = gauss_legendre(nq)
    phiv = zeros(Float64, p + 1, nq)
    dphiv = zeros(Float64, p + 1, nq)
    phip = zeros(Float64, q + 1, nq)
    dphip = zeros(Float64, q + 1, nq)
    @inbounds for k in 1:nq
        L, dL = lagrange_eval(gv, gx[k])
        phiv[:, k] .= L
        dphiv[:, k] .= dL
        Lp, dLp = lagrange_eval(gp, gx[k])
        phip[:, k] .= Lp
        dphip[:, k] .= dLp
    end
    return FEData(p, q, nq, gv, gp, gx, gw, phiv, dphiv, phip, dphip)
end
