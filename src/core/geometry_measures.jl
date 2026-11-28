# Geometry measures: physical height/depth, gradient, Laplacian, gravity,
# quadrature weights, leaf/boundary measures, sizes and wall-node helpers.
# Included after the geometry metrics file; box/affine/curved methods stay
# verbatim and the mapped dispatch resolves at runtime.
# 7. Physical height / depth, its gradient, gravity
# ---------------------------------------------------------------------------
#
# These are the bridge from "the physics is a function of the vertical
# coordinate y" to "the physics is a function of the physical height h(x,y)"
# (GEOM_DESIGN §2.7).  On the box the accessors are exactly today's y / 1-y /
# (0,1) / 0 / (0,-1): the box path keeps its `y` until the annulus stage wires
# the closures, so nothing changes bitwise.  On a mapped geometry the same
# physical law is evaluated at `height(g,x,y)`; `grad_height` is the unit "up"
# direction only when |grad h| = 1 (true for the box and for the annulus, where
# h = r - R0; not for a general validation geometry, so gravity normalises).

@inline height(::CartesianBox, x::Float64, y::Float64) = y
@inline depth(::CartesianBox, x::Float64, y::Float64) = 1.0 - y
@inline grad_height(::CartesianBox, x::Float64, y::Float64) = (0.0, 1.0)
@inline height_laplacian(::CartesianBox, x::Float64, y::Float64) = 0.0
@inline gravity(::CartesianBox, x::Float64, y::Float64) = (0.0, -1.0)
@inline gravity_magnitude(::CartesianBox, x::Float64, y::Float64) = 1.0

@inline function _affine_det(g::AffineBox)
    return g.A[1] * g.A[4] - g.A[2] * g.A[3]
end
@inline function grad_height(g::AffineBox, x::Float64, y::Float64)
    d = _affine_det(g)
    return (-g.A[3] / d, g.A[1] / d)
end
@inline function height(g::AffineBox, x::Float64, y::Float64)
    d = _affine_det(g)
    return (-g.A[3] * (x - g.b[1]) + g.A[1] * (y - g.b[2])) / d
end
@inline depth(g::AffineBox, x::Float64, y::Float64) = 1.0 - height(g, x, y)
@inline height_laplacian(::AffineBox, x::Float64, y::Float64) = 0.0
@inline function gravity(g::AffineBox, x::Float64, y::Float64)
    hx, hy = grad_height(g, x, y)
    n = hypot(hx, hy)
    return (-hx / n, -hy / n)
end
@inline gravity_magnitude(::AffineBox, x::Float64, y::Float64) = 1.0

@inline function _curved_d(g::CurvedBox, S::Float64)
    return 1.0 + g.a * sinpi(S)
end
@inline function _curved_dp(g::CurvedBox, S::Float64)
    return g.a * pi * cospi(S)
end
@inline function _curved_dpp(g::CurvedBox, S::Float64)
    return -g.a * pi * pi * sinpi(S)
end
@inline function height(g::CurvedBox, x::Float64, y::Float64)
    S = x / g.Lx
    return y / _curved_d(g, S)
end
@inline depth(g::CurvedBox, x::Float64, y::Float64) = 1.0 - height(g, x, y)
@inline function grad_height(g::CurvedBox, x::Float64, y::Float64)
    S = x / g.Lx
    d = _curved_d(g, S)
    dp = _curved_dp(g, S)
    T = y / d
    return (-T * dp / (d * g.Lx), 1.0 / d)
end
@inline function height_laplacian(g::CurvedBox, x::Float64, y::Float64)
    S = x / g.Lx
    d = _curved_d(g, S)
    dp = _curved_dp(g, S)
    dpp = _curved_dpp(g, S)
    T = y / d
    return T * (-dpp / d + 2.0 * (dp / d)^2) / (g.Lx * g.Lx)
end
@inline function gravity(g::CurvedBox, x::Float64, y::Float64)
    hx, hy = grad_height(g, x, y)
    n = hypot(hx, hy)
    return (-hx / n, -hy / n)
end
@inline gravity_magnitude(::CurvedBox, x::Float64, y::Float64) = 1.0

"""
    upward_velocity(g, x, y, ux, uz) -> Float64

The velocity component along the physical height gradient `u . grad h` — the
generalisation of the Cartesian `u_z` used by every directional physics term
(base-state advection, adiabatic heating, latent source; `GEOM_DESIGN` §2.7,
§6.4).  The box method is the historical factor `uz` itself, an accessor with no
arithmetic (so a box kernel that used it would be unchanged bit for bit); the
box energy body does not read it (it keeps its `uzq` statements verbatim), the
box method exists so the interface is total and so a future wiring change is a
one-line dispatch.  Mapped: `ux*grad_height[1] + uz*grad_height[2]`; for the
annulus (`|grad h| = 1`, radial) this is the radial velocity.
"""
@inline upward_velocity(::CartesianBox, x::Float64, y::Float64,
                        ux::Float64, uz::Float64) = uz
@inline function upward_velocity(g::AbstractGeometry, x::Float64, y::Float64,
                                 ux::Float64, uz::Float64)
    ghx, ghy = grad_height(g, x, y)
    return ux * ghx + uz * ghy
end

"""
    measure_weight(g, x, y) -> Float64

Optional positive physical weight multiplying `det J` and every face Jacobian
(GEOM_DESIGN §2.3): `1.0` on every plane geometry, so the kernels ignore it in
cut 1.  An axisymmetric shell reduction would return `2*pi*r` (or `r^2*sin
theta`) without a kernel change.  Not read by the box methods, which are the
historical expressions.
"""
@inline measure_weight(::AbstractGeometry, x::Float64, y::Float64) = 1.0

# ---------------------------------------------------------------------------
# 7b. Physical measures (diagnostics; cut 4)
# ---------------------------------------------------------------------------
#
# The diagnostics of cut 4 are the first consumers of the layer's measures
# (GEOM_DESIGN §2.3, rows 462-471; the cut-1 record deferred them).  The box
# methods are the historical expressions, bit for bit:
#   * `leaf_measure` is the E13 leaf term `leaf_hx*leaf_hy` and `total_measure`
#     the E13 sum `total_area(qt)`, so a diagnostics normalisation cannot move a
#     single bit on the box path;
#   * `boundary_measure` is the E8 GLL wall sum `0.5*h*sq.w[i]`, exactly the
#     `wt`/`wb` normaliser `semquad_nu_walls` accumulates internally;
#   * `leaf_vertical_size` is E15's `leaf_hy` (the history `hmin`) and
#     `leaf_min_size` E15's `min(leaf_hx, leaf_hy)` (the CFL `hmin`).
# The mapped methods are the general integrals: `quad_measure` over the leaf for
# the volume, the face Jacobian at the wall GLL nodes for the wall, the physical
# extents for the sizes.

"""
    leaf_measure(g, lay, lf) -> Float64

Physical 2-D measure of one leaf.  Box method: the historical E13 leaf term
`leaf_hx*leaf_hy` (the operand `total_area` sums).  Mapped method: the integral
of `det J` over the leaf, i.e. `sum quad_measure` on an 8-point Gauss rule
(exact for the affine validation geometries and at machine precision for
`CurvedBox`'s smooth `det J`).  The mapped diagnostics that need the same
discrete measure as their own quadrature use [`total_measure`](@ref) with the
diagnostics' rule instead.
"""
@inline leaf_measure(g::CartesianBox, lay, lf::NTuple{3,Int}) =
    leaf_hx(lay, lf[1]) * leaf_hy(lay, lf[1])

"""
    leaf_measure(g::AbstractGeometry, lay, lf, fd::FEData) -> Float64

The mapped leaf measure integrated with the caller's quadrature `fd` (the fully
weighted `quad_measure`), so a mean computed with that rule divides by the
measure of the same rule.
"""
function leaf_measure(g::AbstractGeometry, lay, lf::NTuple{3,Int}, fd::FEData)
    m = leaf_metric(g, lay, lf, fd)
    s = 0.0
    @inbounds for qy in 1:fd.nq, qx in 1:fd.nq
        s += quad_measure(m, fd, qx, qy)
    end
    return s
end
function leaf_measure(g::AbstractGeometry, lay, lf::NTuple{3,Int})
    return leaf_measure(g, lay, lf, make_fedata(2, 1; nq=8))
end

"""
    total_measure(g, lay[, fd]) -> Float64

Physical measure of the whole domain (area in 2-D).  The box method is the
historical E13 `total_area(qt) = sum(leaf_hx*leaf_hy)` over the tree's leaf set,
**bit for bit** (that is the expression every box diagnostics divides by).
A mapped geometry integrates `quad_measure`
over the leaves; with the caller's `fd` it is the measure of the caller's
quadrature rule, which is what a mapped mean must divide by (so a constant field
has the constant as its mean up to round-off).

`total_measure(g::CartesianBox, sq::SemQuad)` sums the E13 leaf terms in the
space's sorted leaf order instead of the tree's hash-set order.  It exists for
API completeness only: the box diagnostics use the `Quadtree` method so that
their normalisation is the historical `total_area` bit for bit.
"""
function total_measure(g::CartesianBox, lay)
    # `lay` is a `Quadtree` on the historical path: E13's sum, bit for bit.
    lay isa Quadtree && return total_area(lay)
    # A `SemQuad`: sum the same leaf terms in the space's sorted leaf order.
    s = 0.0
    for lf in lay.leaves
        s += leaf_measure(g, lay, lf)
    end
    return s
end
function total_measure(g::AbstractGeometry, lay, fd::FEData)
    s = 0.0
    for lf in lay.leaves
        s += leaf_measure(g, lay, lf, fd)
    end
    return s
end
function total_measure(g::AbstractGeometry, lay)
    return total_measure(g, lay, make_fedata(lay.p, lay.p - 1))
end

"""
    boundary_measure(g, sq, side) -> Float64

Physical length of one logical boundary side, integrated with the **GLL wall
rule** the box wall bodies use (E8).  The box method reproduces the historical
`sum 0.5*hx*sq.w[i]` (resp. `0.5*hy*sq.w[j]`) accumulation, so the mapped
`semquad_nu_walls`/`semquad_nu_cbf` normalisers are the box ones on the box map.
The mapped method integrates the face Jacobian at the same GLL nodes; this is a
cold measure (the diagnostics call it once per side).
"""
function boundary_measure(g::CartesianBox, sq, side::Symbol)
    n = sq.p + 1
    s = 0.0
    if side === :xminus || side === :xplus
        xplus = side === :xplus
        for lf in sq.leaves
            (xplus ? lf[2] == leaf_nside_x(sq, lf[1]) - 1 : lf[2] == 0) || continue
            hy = leaf_hy(sq, lf[1])
            @inbounds for j in 1:n
                s += 0.5 * hy * sq.w[j]
            end
        end
    elseif side === :yminus || side === :yplus
        yplus = side === :yplus
        for lf in sq.leaves
            (yplus ? lf[3] == leaf_nside_y(sq, lf[1]) - 1 : lf[3] == 0) || continue
            hx = leaf_hx(sq, lf[1])
            @inbounds for i in 1:n
                s += 0.5 * hx * sq.w[i]
            end
        end
    else
        throw(ArgumentError("unknown face side $side"))
    end
    return s
end
function boundary_measure(g::AbstractGeometry, sq, side::Symbol)
    n = sq.p + 1
    s = 0.0
    for lf in sq.leaves
        _on_logical_wall(lf, side, sq) || continue
        fm = _mapped_face_metric(g, sq, lf, side, sq.xi)
        @inbounds for q in 1:n
            s += face_jacobian(fm, q) * sq.w[q]
        end
    end
    return s
end

"""
    leaf_min_size(g, lay, lf) -> Float64

Physical element size for the CFL/hmin path (GEOM_DESIGN E15).  Box method:
the historical `min(leaf_hx, leaf_hy)` bit for bit.  Mapped method: the smaller
of the two physical edge extents at the leaf centre, `min(2|J e_xi|, 2|J e_eta|)`
(the reference extent is 2 in each direction; for the box map this is exactly
`min(hx, hy)`): on the annulus the angular cell
size grows with `r`, so the CFL minimum is at the inner boundary.
"""
@inline leaf_min_size(g::CartesianBox, lay, lf::NTuple{3,Int}) =
    min(leaf_hx(lay, lf[1]), leaf_hy(lay, lf[1]))
@inline function leaf_min_size(g::AbstractGeometry, lay, lf::NTuple{3,Int})
    J, _ = element_jacobian(g, lay, lf, 0.0, 0.0)
    return min(2.0 * hypot(J[1], J[3]), 2.0 * hypot(J[2], J[4]))
end

"""
    leaf_vertical_size(g, lay, lf) -> Float64

Physical extent of one leaf **along the height direction** (max `height` minus
min `height` over the leaf corners: for a linear `h` that is the projection of
the element onto `grad h`; for the box it is the leaf's vertical size).  Box
method: the historical E15 `leaf_hy` (the history `hmin`), bit for bit.
"""
@inline leaf_vertical_size(g::CartesianBox, lay, lf::NTuple{3,Int}) =
    leaf_hy(lay, lf[1])
function leaf_vertical_size(g::AbstractGeometry, lay, lf::NTuple{3,Int})
    h11 = _height_at(g, lay, lf, -1.0, -1.0)
    h21 = _height_at(g, lay, lf, 1.0, -1.0)
    h22 = _height_at(g, lay, lf, 1.0, 1.0)
    h12 = _height_at(g, lay, lf, -1.0, 1.0)
    hlo = min(h11, h21, h22, h12)
    hhi = max(h11, h21, h22, h12)
    return hhi - hlo
end
@inline function _height_at(g::AbstractGeometry, lay, lf::NTuple{3,Int},
                            xi::Float64, eta::Float64)
    x, y = element_map(g, lay, lf, xi, eta)
    return height(g, x, y)
end

"""
    leaf_normal_size(g, lay, lf, side) -> Float64

Physical extent of one leaf **along the outward normal of the logical face
`side`** — the `h` of the `(h/2) * int [du/dn]^2 ds` AMR indicator and of every
other face-normal length scale (GEOM_DESIGN §2.3, §6.8: "the distance between
opposite faces along the normal").

* Box method: the historical face-normal element size `leaf_hx` on the x-sides
  and `leaf_hy` on the y-sides (E15's normal scale; the box indicator's `h`).
* Mapped method: the distance between the two opposite faces at the face
  midpoint, which for a leaf reference coordinate `s` is `2 / |grad s|` (the
  reference interval has length 2 in `s`).  For the box map this is
  `2 / |2/hx| = hx` (resp. `hy`) up to the reciprocal round-off; it is the
  quantity whose factor-`(R1/R0)` variation across the annulus the physical
  indicator must see (GEOM_DESIGN §6.8).  At the face midpoint, with
  `Jinv = (xi_x, eta_x, xi_y, eta_y)` the rows are `grad xi = (inv1, inv3)` and
  `grad eta = (inv2, inv4)`.
"""
@inline function leaf_normal_size(g::CartesianBox, lay, lf::NTuple{3,Int},
                                  side::Symbol)
    if side === :xminus || side === :xplus
        return leaf_hx(lay, lf[1])
    elseif side === :yminus || side === :yplus
        return leaf_hy(lay, lf[1])
    end
    throw(ArgumentError("unknown face side $side"))
end
@inline function leaf_normal_size(g::AbstractGeometry, lay, lf::NTuple{3,Int},
                                  side::Symbol)
    is_x = side === :xminus || side === :xplus
    is_x || side === :yminus || side === :yplus ||
        throw(ArgumentError("unknown face side $side"))
    sgn = (side === :xplus || side === :yplus) ? 1.0 : -1.0
    xi, eta = is_x ? (sgn, 0.0) : (0.0, sgn)
    inv = jacobian_inverse(g, lay, lf, xi, eta)
    nrm = is_x ? hypot(inv[1], inv[3]) : hypot(inv[2], inv[4])
    return 2.0 / nrm
end

"""
    minimum_leaf_size(g, qt) -> Float64

Minimum [`leaf_min_size`](@ref) over the tree: the CFL `hmin` of
`driver/run.jl`.  The box method is the historical E15 expression
`minimum(min(leaf_hx, leaf_hy) for lf in qt.leaves)`, bit for bit.
"""
@inline minimum_leaf_size(g::CartesianBox, qt::Quadtree) =
    minimum(min(leaf_hx(qt, lf[1]), leaf_hy(qt, lf[1])) for lf in qt.leaves)
function minimum_leaf_size(g::AbstractGeometry, qt::Quadtree)
    hmin = Inf
    for lf in qt.leaves
        h = leaf_min_size(g, qt, lf)
        h < hmin && (hmin = h)
    end
    return hmin
end

"""
    minimum_leaf_height(g, qt) -> Float64

Minimum [`leaf_vertical_size`](@ref) over the tree: the history `hmin` of
`driver/diagnostics_row.jl`.  The box method is the historical E15
`minimum(leaf_hy(sq, lf[1]) for lf in qt.leaves)`, bit for bit.
"""
@inline minimum_leaf_height(g::CartesianBox, qt::Quadtree) =
    minimum(leaf_hy(qt, lf[1]) for lf in qt.leaves)
function minimum_leaf_height(g::AbstractGeometry, qt::Quadtree)
    hmin = Inf
    for lf in qt.leaves
        h = leaf_vertical_size(g, qt, lf)
        h < hmin && (hmin = h)
    end
    return hmin
end

"""
    nu_wall_sides(g) -> (bottom_side, top_side)

The two logical sides whose wall heat flux defines the Nusselt pair, ordered
`(bottom/hot, top/cold)`.  All stage-0 geometries are the box (or its images),
so the default is `(:yminus, :yplus)`: their `height` increases with logical
`y` and the box Nu definition integrates the two `y`-walls.  The annulus
(stage 0.5) overrides this to the two circles (`xminus`/`xplus`, GEOM_DESIGN
§6.3/§6.6), which is why the diagnostics ask the geometry instead of
hard-coding the logical sides.
"""
@inline nu_wall_sides(::AbstractGeometry) = (:yminus, :yplus)

"""
    _nusselt_area(sq, qt) -> Float64

The normalisation "area" of the Nusselt-unit diagnostics: the `R/A` of
[`semquad_exact_residual`](@ref), the wall flux normalisation of
[`semquad_nu_cbf`](@ref) and the latent `closure` term of the history row.  The
box method is the historical `total_area(qt)` bit for bit (the box height is 1,
so the area equals the wall length the quantities divide by); a mapped geometry
uses the top wall's physical measure, the annulus generalisation of "Lx"
(GEOM_DESIGN §6.6).
"""
@inline function _nusselt_area(sq, qt::Quadtree)
    g = sq.geom
    g isa CartesianBox && return total_area(qt)
    return boundary_measure(g, sq, nu_wall_sides(g)[2])
end

# --- logical wall helpers used by the mapped diagnostics (cold paths) -------

"""
    _on_logical_wall(lf, side, lay) -> Bool

Whether the leaf `lf` has a face on the domain boundary logical `side`.  This is
the **logical** wall classification the box bodies also use (`lf[3] == 0`,
`lf[3] == nside-1`, ...; GEOM_DESIGN E16): the physical row tests of the box
diagnostics are equivalent to it for the axis-aligned box, and it stays correct
for every mapped geometry.
"""
@inline function _on_logical_wall(lf::NTuple{3,Int}, side::Symbol, lay)
    if side === :xminus
        return lf[2] == 0
    elseif side === :xplus
        return lf[2] == leaf_nside_x(lay, lf[1]) - 1
    elseif side === :yminus
        return lf[3] == 0
    elseif side === :yplus
        return lf[3] == leaf_nside_y(lay, lf[1]) - 1
    end
    throw(ArgumentError("unknown face side $side"))
end

"""
    _wall_ref_point(side, t) -> (xi, eta)

Leaf reference coordinates of the point at face parameter `t in [-1,1]` on
logical `side`.
"""
@inline function _wall_ref_point(side::Symbol, t::Float64)
    side === :xminus && return (-1.0, t)
    side === :xplus && return (1.0, t)
    side === :yminus && return (t, -1.0)
    side === :yplus && return (t, 1.0)
    throw(ArgumentError("unknown face side $side"))
end

"""
    _logical_wall_node(ids, n, side, q) -> Int

Global id of the `q`-th wall-node of a leaf with element ids `ids` on logical
`side`, in the order the box wall bodies use (moving face index `i`/`j`).
"""
@inline function _logical_wall_node(ids::Vector{Int}, n::Int, side::Symbol,
                                    q::Int)
    if side === :xminus || side === :xplus
        i = side === :xplus ? n : 1
        return ids[(q - 1) * n + i]
    end
    j = side === :yplus ? n : 1
    return ids[(j - 1) * n + q]
end

