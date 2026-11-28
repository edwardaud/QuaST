# Geometry mapping layer: logical coordinates, maps, Jacobians and leaf location.
# Extracted from core/geometry.jl; the geometry types are included before this
# file, and every cross-slice reference below resolves backwards.
# ---------------------------------------------------------------------------
# 3. Logical coordinates: global logical (S,T), leaf reference (xi,eta)
# ---------------------------------------------------------------------------
#
# Logical coordinates are the canonical identity of a point (GEOM_DESIGN §2.1
# rule 4): node dedup, leaf location and constraint/prolongation positions are
# computed from `(level, ix, iy, node-index)`, and only then is the map applied.
# The box path keeps its own historical expressions (E2/E3/E11), which are
# mathematically the same map; the helpers here are what the mapped path uses.

@inline leaf_nside_x(lay, l::Int) = lay.nx0 * (1 << l)
@inline leaf_nside_y(lay, l::Int) = lay.ny0 * (1 << l)

"""
    global_logical(lay, lf, xi, eta) -> (S, T)

Global logical position in `[0,1]^2` of the leaf reference point `(xi,eta)` of
leaf `lf = (level, ix, iy)`.  `lay` is the `Quadtree` (or a `SemQuad`, which
carries the same layout fields) the leaf belongs to.
"""
@inline function global_logical(lay, lf::NTuple{3,Int},
                                xi::Float64, eta::Float64)
    l = lf[1]
    S = (lf[2] + 0.5 * (xi + 1.0)) / leaf_nside_x(lay, l)
    T = (lf[3] + 0.5 * (eta + 1.0)) / leaf_nside_y(lay, l)
    return (S, T)
end

"""
    leaf_ref(lay, lf, S, T) -> (xi, eta)

Leaf reference coordinates of the global logical point `(S,T)` known to lie in
leaf `lf`; the inverse of [`global_logical`](@ref).
"""
@inline function leaf_ref(lay, lf::NTuple{3,Int},
                          S::Float64, T::Float64)
    l = lf[1]
    xi = 2.0 * (S * leaf_nside_x(lay, l) - lf[2]) - 1.0
    eta = 2.0 * (T * leaf_nside_y(lay, l) - lf[3]) - 1.0
    return (xi, eta)
end

"""
    _logical_node_key(qt, lf, i, j, rule) -> (Snum, Tnum)

Canonical logical identity of the SEM node `(i,j)` of leaf `lf`, as the two
**scaled global-logical numerators**

    Snum = (2*ix + xi_i + 1) * 2^(maxlevel-l),   Tnum likewise,

with `xi_i` the `i`-th GLL node.  The node's global logical coordinate is
`S = Snum / (2*nx0*2^maxlevel)` (`T` likewise), so the pair identifies the point
exactly and, crucially, **two leaves that share the node produce the same pair
bit for bit**: a shared node is a leaf corner on both sides, or a face node with
the same reference index, and both cases evaluate the same exact expression
(`xi = +-1` gives an integer numerator; a shared face-interior node uses the same
`xi_j` and the same level fraction; the even-order mid-edge coincidence
`xi_i = 0` is the integer `2*ix+1`).  No rounding tolerance is involved, which is
what removes the latent box assumption of the historical `round(x/tol)` node
hash.  The identity is geometric and therefore independent of the polynomial
order.

The scale `2^(maxlevel-l)` puts a node at level `l` on the common finest grid
`maxlevel`, so leaves at different levels compare directly.

**Stage 0.5c1 (the periodic seam).**  On a `seam = :periodic` tree the logical
`T` direction is a circle, so the node at `T = 1` *is* the node at `T = 0`.  The
`T = 1` numerator is exactly the denominator `2*ny0*2^maxlevel` (the last column's
`+1` node is `(2*(nside-1) + 2)*scale = 2*nside*scale`), so canonicalising
`Tnum == den_y` to `0` identifies the two sides **bit for bit** and needs no
tolerance: the wrapped node then also takes the `T = 0` physical coordinate
(`logical_map(geom, ..., 0)`), which is the coordinate of the seam curve the map
closes.  The `Snum` of the two images is the same exact expression as for any
shared interior node, so the seam nodes merge by the same rule.  A non-periodic
tree takes the historical body bit for bit.
"""
@inline function _logical_node_key(qt::Quadtree, lf::NTuple{3,Int},
                                   i::Int, j::Int, rule)
    l = lf[1]
    scale = Float64(1 << (qt.maxlevel - l))
    Snum = (2.0 * lf[2] + (rule.nodes[i] + 1.0)) * scale
    Tnum = (2.0 * lf[3] + (rule.nodes[j] + 1.0)) * scale
    if qt.seam === :periodic && Tnum == _logical_den_y(qt)
        Tnum = 0.0
    end
    return (Snum, Tnum)
end

# denominators of the canonical logical coordinate recovered from a key
@inline _logical_den_x(qt::Quadtree) = 2.0 * qt.nx0 * Float64(1 << qt.maxlevel)
@inline _logical_den_y(qt::Quadtree) = 2.0 * qt.ny0 * Float64(1 << qt.maxlevel)

"""
    locate_leaf_logical(lay, S, T, leaves) -> lf

Locate the leaf containing the global logical point `(S,T)` by an
O(maxlevel) index descent from the level-0 base cells.  `leaves` is the set of
leaf keys (passed in to avoid rebuilding it per point).  This is the only leaf
search of the mapped path: it never looks at a physical coordinate, so a curved
face is located by construction rather than by a rounded position test.
"""
function locate_leaf_logical(lay, S::Float64, T::Float64,
                             leaves::Set{NTuple{3,Int}})
    l = 0
    ix = clamp(floor(Int, S * lay.nx0), 0, lay.nx0 - 1)
    iy = clamp(floor(Int, T * lay.ny0), 0, lay.ny0 - 1)
    while true
        lf = (l, ix, iy)
        lf in leaves && return lf
        l += 1
        l <= 62 || throw(ArgumentError("logical point ($S,$T) not in the tree"))
        ix = clamp(floor(Int, S * leaf_nside_x(lay, l)), 0,
                   leaf_nside_x(lay, l) - 1)
        iy = clamp(floor(Int, T * leaf_nside_y(lay, l)), 0,
                   leaf_nside_y(lay, l) - 1)
    end
end

# ---------------------------------------------------------------------------
# 4. The map, its inverse, the Jacobian
# ---------------------------------------------------------------------------

"""
    logical_map(g, S, T) -> (x, y)

The global map from the **global logical** coordinate `(S,T) in [0,1] x [0,1]`
to physical coordinates.  `S`/`T` are the normalised logical coordinates
`element_map` builds from the leaf reference point; the geometry performs its
own physical scaling, and for the box that scaling is `x = g.Lx*S` — i.e. a
caller must pass `S in [0,1]`, never a physical-like `Lx*S` (the stage-0.5
audit's D1 defect, `src/mg/pressure.jl` `semquad_eta_contrast`).  This is the
only place a geometry defines its map; every mapped point in the solver goes
through it (directly or through `element_map`).
"""
@inline function logical_map(g::CartesianBox, S::Float64, T::Float64)
    return (g.Lx * S, T)
end
@inline function logical_map(g::AffineBox, S::Float64, T::Float64)
    A = g.A
    u = g.Lx * S
    return (A[1] * u + A[2] * T + g.b[1],
            A[3] * u + A[4] * T + g.b[2])
end
@inline function logical_map(g::CurvedBox, S::Float64, T::Float64)
    return (g.Lx * S, T * (1.0 + g.a * sinpi(S)))
end

"""
    element_map(g, lay, lf, xi, eta) -> (x, y)

Physical coordinates of the leaf reference point `(xi,eta)` in leaf `lf`.
The `CartesianBox` method is the historical formula, character for character
(GEOM_DESIGN E3): `x = x0 + 0.5*(xi+1.0)*hx`, `y = y0 + 0.5*(eta+1.0)*hy`.
Every mapped method is `logical_map(g, global_logical(lay, lf, xi, eta)...)`.
"""
@inline function element_map(g::CartesianBox, lay, lf::NTuple{3,Int},
                             xi::Float64, eta::Float64)
    lm = leaf_metric(g, lay, lf)
    return (lm.x0 + 0.5 * (xi + 1.0) * lm.hx,
            lm.y0 + 0.5 * (eta + 1.0) * lm.hy)
end
@inline function element_map(g::AbstractGeometry, lay, lf::NTuple{3,Int},
                             xi::Float64, eta::Float64)
    S, T = global_logical(lay, lf, xi, eta)
    return logical_map(g, S, T)
end

"""
    element_inverse(g, lay, x, y) -> (S, T)          # global logical
    element_inverse(g, lay, lf, x, y) -> (xi, eta)   # leaf reference

The inverse map.  The 5-argument form is the leaf-reference one the point
evaluation/transfer use (the caller knows the leaf); the 4-argument form is the
global logical one used to *find* the leaf on the mapped path.
"""
@inline element_inverse(::CartesianBox, lay, x::Float64, y::Float64) =
    (x / lay.Lx, y)
@inline function element_inverse(g::AffineBox, lay, x::Float64, y::Float64)
    A = g.A
    d = A[1] * A[4] - A[2] * A[3]
    px = x - g.b[1]
    py = y - g.b[2]
    u = (A[4] * px - A[2] * py) / d
    v = (-A[3] * px + A[1] * py) / d
    return (u / g.Lx, v)
end
@inline function element_inverse(g::CurvedBox, lay, x::Float64, y::Float64)
    S = x / g.Lx
    return (S, y / (1.0 + g.a * sinpi(S)))
end

# leaf-reference form: box uses the historical E11 expression (the point is
# known to be in the leaf); mapped geometries invert globally then go to the
# leaf reference coordinates.
@inline function element_inverse(g::CartesianBox, lay, lf::NTuple{3,Int},
                                 x::Float64, y::Float64)
    lm = leaf_metric(g, lay, lf)
    return (2.0 * (x - lm.x0) / lm.hx - 1.0,
            2.0 * (y - lm.y0) / lm.hy - 1.0)
end
@inline function element_inverse(g::AbstractGeometry, lay, lf::NTuple{3,Int},
                                 x::Float64, y::Float64)
    S, T = element_inverse(g, lay, x, y)
    return leaf_ref(lay, lf, S, T)
end

"""
    element_jacobian(g, lay, lf, xi, eta) -> (J, detJ)

The element Jacobian `J = (x_xi, x_eta, y_xi, y_eta)` of the map from the leaf
reference square and its determinant.  Affine geometries (`CartesianBox`,
`AffineBox`) have a constant `J` per leaf; `CurvedBox` varies inside the leaf.
Evaluated in cold paths and in the metric build, never inside a quadrature loop.

For the box, `detJ` is the stored volume factor `0.25*hx*hy` (GEOM_DESIGN E4),
not `J[1]*J[4] - J[2]*J[3]`: the box volume weight must be the historical
operand bit for bit, and `J` is the historical derivative `0.5*hx`/`0.5*hy`.
"""
@inline function element_jacobian(g::CartesianBox, lay, lf::NTuple{3,Int},
                                  xi::Float64, eta::Float64)
    lm = leaf_metric(g, lay, lf)
    return ((0.5 * lm.hx, 0.0, 0.0, 0.5 * lm.hy), lm.wfac)
end
@inline function element_jacobian(g::AffineBox, lay, lf::NTuple{3,Int},
                                  xi::Float64, eta::Float64)
    l = lf[1]
    ds = g.Lx * (1.0 / leaf_nside_x(lay, l)) * 0.5   # dx/dxi factor, logical x
    dt = (1.0 / leaf_nside_y(lay, l)) * 0.5          # dT/deta factor
    A = g.A
    a = A[1] * ds
    b = A[2] * dt
    c = A[3] * ds
    d = A[4] * dt
    return ((a, b, c, d), a * d - b * c)
end
@inline function element_jacobian(g::CurvedBox, lay, lf::NTuple{3,Int},
                                  xi::Float64, eta::Float64)
    S, T = global_logical(lay, lf, xi, eta)
    l = lf[1]
    dS = (1.0 / leaf_nside_x(lay, l)) * 0.5
    dT = (1.0 / leaf_nside_y(lay, l)) * 0.5
    dd = 1.0 + g.a * sinpi(S)
    a = g.Lx * dS                    # x_xi
    b = 0.0                          # x_eta
    c = T * (g.a * pi * cospi(S)) * dS   # y_xi
    d = dd * dT                      # y_eta
    return ((a, b, c, d), a * d - b * c)
end

@inline function _jacobian_inverse_from_jac(J::NTuple{4,Float64}, detJ::Float64)
    invdet = 1.0 / detJ
    # J = ((x_xi, x_eta), (y_xi, y_eta)) = (a, b, c, d);  J^{-T} =
    # ((d, -c), (-b, a)) / detJ, i.e. (xi_x, eta_x, xi_y, eta_y).  Then
    # (u_x, u_y) = (Jinv11*u_xi + Jinv12*u_eta, Jinv21*u_xi + Jinv22*u_eta),
    # which is the chain rule the mapped kernels use.
    return (J[4] * invdet, -J[3] * invdet, -J[2] * invdet, J[1] * invdet)
end

"""
    jacobian_inverse(g, lay, lf, xi, eta) -> (Jinv11, Jinv12, Jinv21, Jinv22)

`J^{-T}` of [`element_jacobian`](@ref), in row-major component order
`(xi_x, eta_x, xi_y, eta_y)`.  The `CartesianBox` method returns the historical
reciprocals `(2.0/hx, 0.0, 0.0, 2.0/hy)` directly (GEOM_DESIGN E5/E7): computing
them from `J` and `detJ` would change the bit pattern of a box derivative.
"""
@inline function jacobian_inverse(g::CartesianBox, lay, lf::NTuple{3,Int},
                                  xi::Float64, eta::Float64)
    lm = leaf_metric(g, lay, lf)
    return (lm.ihx, 0.0, 0.0, lm.ihy)
end
@inline function jacobian_inverse(g::AbstractGeometry, lay, lf::NTuple{3,Int},
                                  xi::Float64, eta::Float64)
    J, detJ = element_jacobian(g, lay, lf, xi, eta)
    return _jacobian_inverse_from_jac(J, detJ)
end

"""
    locate_leaf(g, lay, x, y, leaves) -> lf

The leaf containing the physical point `(x,y)`.  The box method is the current
floor search on the physical coordinate (GEOM_DESIGN E1/E11); mapped geometries
invert the global map and descend in logical coordinates
([`locate_leaf_logical`](@ref)), which never rounds a physical coordinate.
"""
function locate_leaf(g::CartesianBox, lay, x::Float64, y::Float64,
                     leaves::Set{NTuple{3,Int}})
    # The historical floor search, character for character: `x / leaf_hx`, not
    # `x * nside / Lx` (the two differ in the last bits on a boundary).
    l = 0
    ix = clamp(floor(Int, x / leaf_hx(lay, 0)), 0, lay.nx0 - 1)
    iy = clamp(floor(Int, y / leaf_hy(lay, 0)), 0, lay.ny0 - 1)
    while true
        lf = (l, ix, iy)
        lf in leaves && return lf
        l += 1
        l <= 62 || throw(ArgumentError("point ($x,$y) not in the tree"))
        ix = clamp(floor(Int, x / leaf_hx(lay, l)), 0,
                   lay.nx0 * (1 << l) - 1)
        iy = clamp(floor(Int, y / leaf_hy(lay, l)), 0,
                   lay.ny0 * (1 << l) - 1)
    end
end
function locate_leaf(g::AbstractGeometry, lay, x::Float64, y::Float64,
                     leaves::Set{NTuple{3,Int}})
    S, T = element_inverse(g, lay, x, y)
    return locate_leaf_logical(lay, S, T, leaves)
end

