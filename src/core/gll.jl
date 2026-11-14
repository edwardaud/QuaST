# GLL nodes/weights, barycentric Lagrange basis, Gauss-Legendre quadrature.
#
# GLL nodes are the roots of (1-x^2) P'_p(x).  The interior roots are found by
# Newton on P'_p from the Chebyshev-Lobatto initial guesses; the endpoints are
# exact.  Weights use w_i = 2 / (p (p+1) P_p(x_i)^2).  This is independent of
# the quadrature used for assembly (Gauss-Legendre, see `gauss_legendre`).

struct GLLRule
    p::Int
    nodes::Vector{Float64}
    weights::Vector{Float64}
    bary::Vector{Float64}
end

"""Return P_p(x), P'_p(x), P''_p(x) by the three-term recurrence."""
function legendre_with_derivs(p::Int, x::Float64)
    p < 0 && throw(ArgumentError("p must be >= 0, got $p"))
    if p == 0
        return 1.0, 0.0, 0.0
    elseif p == 1
        return x, 1.0, 0.0
    end
    P0, P1 = 1.0, x
    dP0, dP1 = 0.0, 1.0
    d2P0, d2P1 = 0.0, 0.0
    @inbounds for n in 1:p-1
        c1 = (2.0 * n + 1.0) / (n + 1.0)
        c2 = n / (n + 1.0)
        P2 = c1 * x * P1 - c2 * P0
        dP2 = c1 * (P1 + x * dP1) - c2 * dP0
        d2P2 = c1 * (2.0 * dP1 + x * d2P1) - c2 * d2P0
        P0, P1 = P1, P2
        dP0, dP1 = dP1, dP2
        d2P0, d2P1 = d2P1, d2P2
    end
    return P1, dP1, d2P1
end

function barycentric_weights(x::Vector{Float64})
    n = length(x)
    b = ones(Float64, n)
    @inbounds for i in 1:n
        prod = 1.0
        for j in 1:n
            j == i && continue
            prod *= (x[i] - x[j])
        end
        b[i] = 1.0 / prod
    end
    return b
end

"""GLL rule of degree p (p+1 nodes on [-1,1])."""
function gll_rule(p::Int)
    p >= 1 || throw(ArgumentError("p must be >= 1, got $p"))
    if p == 1
        x = [-1.0, 1.0]
        w = [1.0, 1.0]
    else
        x = zeros(Float64, p + 1)
        x[1] = -1.0
        x[end] = 1.0
        @inbounds for i in 1:p-1
            xi = cos(pi * i / p)
            for _ in 1:60
                _, dP, d2P = legendre_with_derivs(p, xi)
                delta = dP / d2P
                xi -= delta
                abs(delta) < 1e-15 && break
            end
            x[p + 1 - i] = xi
        end
        sort!(x)
        x[1] = -1.0
        x[end] = 1.0
        w = zeros(Float64, p + 1)
        @inbounds for i in 1:p+1
            P, _, _ = legendre_with_derivs(p, x[i])
            w[i] = 2.0 / (p * (p + 1) * P * P)
        end
    end
    return GLLRule(p, x, w, barycentric_weights(x))
end

"""
    lagrange_eval!(L, dL, rule, x)

Out-parameter form of [`lagrange_eval`](@ref): fills `L` (basis values) and
`dL` (first derivatives) at `x` from the barycentric formula.  The arithmetic
and the summation order are identical to `lagrange_eval`, so for `L`/`dL`
initialised to zero the two forms are bitwise equal; the buffers are fully
overwritten in both branches (the exact-node-hit branch first clears `L`,
because the historical form returns a fresh zero vector with a single 1).

Used by the hot assembly/transfer loops, which reuse one buffer per call site
instead of allocating two `n`-vectors per evaluation.
"""
@inline function lagrange_eval!(L::AbstractVector{Float64},
                               dL::AbstractVector{Float64},
                               rule::GLLRule, x::Float64)
    n = rule.p + 1
    # exact node hit: L_i = delta, dL_i = sum_j (b_j/b_i)/(x_i-x_j)
    @inbounds for i in 1:n
        if abs(x - rule.nodes[i]) < 1e-14
            fill!(L, 0.0)
            L[i] = 1.0
            s = 0.0
            for j in 1:n
                j == i && continue
                dL[j] = (rule.bary[j] / rule.bary[i]) /
                        (rule.nodes[i] - rule.nodes[j])
                s += dL[j]
            end
            dL[i] = -s
            return L, dL
        end
    end
    t = 0.0
    dt = 0.0
    @inbounds for j in 1:n
        dx = x - rule.nodes[j]
        bj = rule.bary[j]
        t += bj / dx
        dt += -bj / (dx * dx)
    end
    @inbounds for i in 1:n
        dx = x - rule.nodes[i]
        Li = (rule.bary[i] / dx) / t
        L[i] = Li
        dL[i] = (-rule.bary[i] / (dx * dx)) / t - Li * (dt / t)
    end
    return L, dL
end

"""Lagrange basis values and derivatives at an arbitrary point x in [-1,1]."""
function lagrange_eval(rule::GLLRule, x::Float64)
    n = rule.p + 1
    L = zeros(Float64, n)
    dL = zeros(Float64, n)
    return lagrange_eval!(L, dL, rule, x)
end


"""
Lagrange basis values, first and second derivatives at an arbitrary point.
Uses the barycentric formula, which is stable for the GLL basis.

  L_i'/L_i = -1/(x-x_i) - S'/S
  L_i''    = L_i [ (L_i'/L_i)^2 + 1/(x-x_i)^2 - S''/S + (S'/S)^2 ]

with S = sum_j b_j/(x-x_j).
"""
function lagrange_d2(rule::GLLRule, x::Float64)
    n = rule.p + 1
    L = zeros(Float64, n)
    dL = zeros(Float64, n)
    d2 = zeros(Float64, n)
    return lagrange_d2!(L, dL, d2, rule, x)
end

"""
    lagrange_d2!(L, dL, d2, rule, x)

Out-parameter form of [`lagrange_d2`](@ref); same arithmetic and order, so it
is bitwise equal to the allocating form when the buffers start at zero.  The
exact-node-hit branch leaves `d2` at zero (the historical form returns a fresh
zero vector there), so the buffer is cleared first.
"""
@inline function lagrange_d2!(L::AbstractVector{Float64},
                              dL::AbstractVector{Float64},
                              d2::AbstractVector{Float64},
                              rule::GLLRule, x::Float64)
    n = rule.p + 1
    # exact node hit: use the existing stable first-derivative formula
    @inbounds for i in 1:n
        if abs(x - rule.nodes[i]) < 1e-14
            fill!(L, 0.0)
            fill!(d2, 0.0)
            L[i] = 1.0
            s = 0.0
            for j in 1:n
                j == i && continue
                dL[j] = (rule.bary[j] / rule.bary[i]) /
                        (rule.nodes[i] - rule.nodes[j])
                s += dL[j]
            end
            dL[i] = -s
            return L, dL, d2
        end
    end
    S = 0.0
    Sp = 0.0
    Spp = 0.0
    @inbounds for j in 1:n
        dx = x - rule.nodes[j]
        bj = rule.bary[j]
        S += bj / dx
        Sp -= bj / (dx * dx)
        Spp += 2.0 * bj / (dx * dx * dx)
    end
    @inbounds for i in 1:n
        dx = x - rule.nodes[i]
        Li = (rule.bary[i] / dx) / S
        ratio = -1.0 / dx - Sp / S
        L[i] = Li
        dL[i] = Li * ratio
        d2[i] = Li * (ratio^2 + 1.0 / (dx * dx) - Spp / S + (Sp / S)^2)
    end
    return L, dL, d2
end

"""Gauss-Legendre nodes and weights on [-1,1] via the Golub-Welsch algorithm."""
function gauss_legendre(n::Int)
    n >= 1 || throw(ArgumentError("n must be >= 1, got $n"))
    if n == 1
        return [0.0], [2.0]
    end
    beta = [j / sqrt(4.0 * j * j - 1.0) for j in 1:n-1]
    T = SymTridiagonal(zeros(Float64, n), beta)
    E = eigen(T)
    x = copy(E.values)
    w = 2.0 .* (E.vectors[1, :] .^ 2)
    idx = sortperm(x)
    return x[idx], w[idx]
end

# ---------------------------------------------------------------------------
# Cached rules and the values-only Lagrange evaluation used by the point-
# evaluation path (`_semquad_eval_at`, the viscosity sampling of every
# quadrature point, and the 65536-point frame dump).
#
# `gll_rule(p)` is a pure function of `p` but runs a Newton iteration per node
# and allocates three vectors *per call*: `--track-allocation=user` attributed
# 36.4 MB (x/z/w) plus 32 MB (Legendre derivative scratch) per 5 blob steps to
# it, and `lagrange_eval` another 106 MB, because `_semquad_eval_at` rebuilt
# the rule at every point.  The rules for the degrees this code uses are built
# once here and shared read-only: every caller treats a `GLLRule` as immutable
# (nothing in `src/` or `test/` assigns to `.nodes`/`.weights`/`.bary`), so the
# shared data is indistinguishable from a fresh copy, bit for bit.
const _GLL_RULES = ntuple(p -> gll_rule(p), 12)

@inline function gll_rule_cached(p::Int)
    1 <= p <= 12 && return _GLL_RULES[p]
    return gll_rule(p)          # rare degrees: historical behaviour
end

"""
    lagrange_values(rule, x)

Lagrange basis values at `x` without the first derivatives.  Same barycentric
formula and same summation order as `lagrange_eval`, so the values are bitwise
equal to its `L`; it only skips the derivative branch (`dt`, `dL`).
"""
function lagrange_values(rule::GLLRule, x::Float64)
    n = rule.p + 1
    L = zeros(Float64, n)
    return lagrange_values!(L, rule, x)
end

"""
    lagrange_values!(L, rule, x)

Out-parameter form of [`lagrange_values`](@ref): fills `L` and returns it.  Same
formula and summation order, so the values are bitwise equal to the allocating
form; the exact-node-hit branch clears `L` first (the historical form returns a
fresh zero vector with a single 1).
"""
@inline function lagrange_values!(L::AbstractVector{Float64}, rule::GLLRule,
                                  x::Float64)
    n = rule.p + 1
    @inbounds for i in 1:n
        if abs(x - rule.nodes[i]) < 1e-14
            fill!(L, 0.0)
            L[i] = 1.0
            return L
        end
    end
    t = 0.0
    @inbounds for j in 1:n
        t += rule.bary[j] / (x - rule.nodes[j])
    end
    @inbounds for i in 1:n
        L[i] = (rule.bary[i] / (x - rule.nodes[i])) / t
    end
    return L
end
