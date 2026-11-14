# Point evaluation of a quadtree SEM field: leaf search and the reusable Lagrange workspace.
"""Evaluate a quadtree SEM field at (x,y)."""
function semquad_eval(sq::SemQuad, u::Vector{Float64}, x::Float64, y::Float64)
    n = sq.p + 1
    rule = gll_rule(sq.p)
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        if x >= x0 - 1.0e-14 && x <= x0 + hx + 1.0e-14 &&
           y >= y0 - 1.0e-14 && y <= y0 + hy + 1.0e-14
            xi = 2.0 * (x - x0) / hx - 1.0
            eta = 2.0 * (y - y0) / hy - 1.0
            Lb, _ = lagrange_eval(rule, xi)
            Lc, _ = lagrange_eval(rule, eta)
            ids = sq.elem[lf]
            v = 0.0
            for j in 1:n, i in 1:n
                v += u[ids[(j - 1) * n + i]] * Lb[i] * Lc[j]
            end
            return v
        end
    end
    throw(ArgumentError("point ($x,$y) not in the tree"))
end

"""
    _semquad_leaf_at(sq, x, y, leaves)

Locate the leaf containing `(x,y)` by descending the quadtree from the root.
`leaves` is the set of `sq.leaves`, passed in to avoid rebuilding it per point.
"""
function _semquad_leaf_at(sq::SemQuad, x::Float64, y::Float64,
                          leaves::Set{NTuple{3,Int}})
    # Start from the *actual* level-0 base cell, not (0,0): a multi-root tree
    # (cells_x/cells_y > 1) has several base leaves, and checking only (0,0,0)
    # would return the first base cell for every point.  Refined trees hid this
    # because they have no level-0 leaves; once a coarse region survives
    # coarsening (e.g. an MG coarse level, or AMR with coarsening) every point
    # outside base cell 0 was evaluated with the wrong element.
    l = 0
    ix = clamp(floor(Int, x / leaf_hx(sq, 0)), 0, sq.nx0 - 1)
    iy = clamp(floor(Int, y / leaf_hy(sq, 0)), 0, sq.ny0 - 1)
    while true
        lf = (l, ix, iy)
        lf in leaves && return lf
        l += 1
        l <= 62 || throw(ArgumentError("point ($x,$y) not in the tree"))
        ix = clamp(floor(Int, x / leaf_hx(sq, l)), 0, sq.nx0 * (1 << l) - 1)
        iy = clamp(floor(Int, y / leaf_hy(sq, l)), 0, sq.ny0 * (1 << l) - 1)
    end
end

"""
    SemQuadEvalWork(n)

Two `n`-vector buffers for the out-parameter point evaluation
([`_semquad_eval_at!`](@ref), [`_semquad_eval_at_leaf!`](@ref)).  One per loop:
the transfer reuses it over every new node and the energy assembly over every
quadrature point, instead of allocating two `lagrange_values` results per call
(`--track-allocation=user` charged 109.9 MB to `lagrange_values` per 4 AMR
cycles on the 10 000-leaf state, essentially all of it the nodal transfer).
"""
struct SemQuadEvalWork
    Lb::Vector{Float64}
    Lc::Vector{Float64}
end

SemQuadEvalWork(n::Int) = SemQuadEvalWork(zeros(Float64, n), zeros(Float64, n))

"""
    _semquad_eval_at_leaf!(w, sq, u, x, y, lf)

Evaluate the field `u` at `(x,y)` knowing that the point lies in leaf `lf`
(the energy assembly's quadrature points and the transfer's per-leaf loop do).
Identical arithmetic to [`_semquad_eval_at`](@ref); only the point search is
skipped.  The caller is responsible for the leaf being the one the search
would return (true for every interior point).
"""
function _semquad_eval_at_leaf!(w::SemQuadEvalWork, sq::SemQuad,
                                u::Vector{Float64}, x::Float64, y::Float64,
                                lf::NTuple{3,Int})
    n = sq.p + 1
    hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
    x0 = lf[2] * hx; y0 = lf[3] * hy
    xi = 2.0 * (x - x0) / hx - 1.0
    eta = 2.0 * (y - y0) / hy - 1.0
    rule = gll_rule_cached(sq.p)
    lagrange_values!(w.Lb, rule, xi)
    lagrange_values!(w.Lc, rule, eta)
    ids = sq.elem[lf]
    v = 0.0
    for j in 1:n, i in 1:n
        v += u[ids[(j - 1) * n + i]] * w.Lb[i] * w.Lc[j]
    end
    return v
end

"""Evaluate a quadtree SEM field at (x,y) with an O(maxlevel) point search."""
function _semquad_eval_at(sq::SemQuad, u::Vector{Float64},
                          x::Float64, y::Float64,
                          leaves::Set{NTuple{3,Int}})
    return _semquad_eval_at!(SemQuadEvalWork(sq.p + 1), sq, u, x, y, leaves)
end

"""Out-parameter form of [`_semquad_eval_at`](@ref); same value bit for bit."""
function _semquad_eval_at!(w::SemQuadEvalWork, sq::SemQuad,
                           u::Vector{Float64}, x::Float64, y::Float64,
                           leaves::Set{NTuple{3,Int}})
    lf = _semquad_leaf_at(sq, x, y, leaves)
    return _semquad_eval_at_leaf!(w, sq, u, x, y, lf)
end

"""
    _semquad_grad_at(sq, u, x, y, leaves)

Gradient `(du/dx, du/dy)` of a quadtree SEM field at `(x,y)`, using the element
own GLL derivative matrices and the physical scaling `2/h`.
"""
function _semquad_grad_at(sq::SemQuad, u::Vector{Float64},
                          x::Float64, y::Float64,
                          leaves::Set{NTuple{3,Int}})
    lf = _semquad_leaf_at(sq, x, y, leaves)
    n = sq.p + 1
    hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
    x0 = lf[2] * hx; y0 = lf[3] * hy
    xi = 2.0 * (x - x0) / hx - 1.0
    eta = 2.0 * (y - y0) / hy - 1.0
    rule = gll_rule_cached(sq.p)
    Lb, dLb = lagrange_eval(rule, xi)
    Lc, dLc = lagrange_eval(rule, eta)
    ids = sq.elem[lf]
    dx = 0.0; dy = 0.0
    for j in 1:n, i in 1:n
        v = u[ids[(j - 1) * n + i]]
        dx += v * dLb[i] * Lc[j]
        dy += v * Lb[i] * dLc[j]
    end
    return dx * 2.0 / hx, dy * 2.0 / hy
end

