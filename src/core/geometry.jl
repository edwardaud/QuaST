# Geometry layer residual: physical measures, topology and the PolarAnnulus
# geometry.  The types, map/Jacobian and metrics files are included before this
# file; all remaining cross-file references resolve backwards in include order.
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

# ---------------------------------------------------------------------------
# 8. Logical boundary registry
# ---------------------------------------------------------------------------

"""
    LogicalBoundary

One logical side of the domain and its physical meaning: `side` is the logical
side (`:xminus`, `:xplus`, `:yminus`, `:yplus`), `kind` its physical type
(`:wall`, `:periodic`, `:symmetry`) and `tag` the name downstream layers use
(`:left`, `:right`, `:bottom`, `:top`; the annulus adds `:inner`, `:outer`,
`:seam_lo`, `:seam_hi`).  A BC or diagnostics layer reads the geometry's
registry instead of assuming Cartesian walls.
"""
struct LogicalBoundary
    side::Symbol
    kind::Symbol
    tag::Symbol
end

"""
    boundaries(g) -> NTuple{4,LogicalBoundary}

The logical boundary registry, in `(:xminus, :xplus, :yminus, :yplus)` order.
All three stage-0 geometries are four-walled boxes (the affine/curved images of
the box); the annulus will register circles and seam sides instead.
"""
function boundaries(::CartesianBox)
    return (LogicalBoundary(:xminus, :wall, :left),
            LogicalBoundary(:xplus, :wall, :right),
            LogicalBoundary(:yminus, :wall, :bottom),
            LogicalBoundary(:yplus, :wall, :top))
end
function boundaries(::AffineBox)
    return (LogicalBoundary(:xminus, :wall, :left),
            LogicalBoundary(:xplus, :wall, :right),
            LogicalBoundary(:yminus, :wall, :bottom),
            LogicalBoundary(:yplus, :wall, :top))
end
function boundaries(::CurvedBox)
    return (LogicalBoundary(:xminus, :wall, :left),
            LogicalBoundary(:xplus, :wall, :right),
            LogicalBoundary(:yminus, :wall, :bottom),
            LogicalBoundary(:yplus, :wall, :top))
end

"""
    perimeter(g::CartesianBox) -> Float64

Analytic perimeter `2*(Lx + 1)` of the box (GEOM_DESIGN E12).  The box path must
return the historical expression.
"""
@inline perimeter(g::CartesianBox) = 2.0 * (g.Lx + 1.0)

"""
    perimeter(g::AbstractGeometry, lay, bmask) -> Float64

Physical perimeter of a mapped geometry, integrated from the face Jacobian over
every domain-boundary face (`bmask` is the serial boundary mask of
[`_semquad_boundary_face_mask`](@ref); entry `4*(it-1) + side` non-zero means
that leaf face is on `∂Ω`).  This is the mapped `perimeter` of GEOM_DESIGN
§2.3/E12: the box path keeps its analytic `2*(Lx+1)` (a mapped method for a box
would integrate a different Float64), so the two are separate methods.

It is a cold measure — used by the net-flux compatibility threshold in the
solve, never in an operator kernel — so it may build its own face quadrature;
an 8-point Gauss rule integrates a smooth face Jacobian accurately and the
value only scales a round-off-level threshold.
"""
function perimeter(g::AbstractGeometry, lay, bmask::Vector{UInt8})
    fd = make_fedata(2, 1; nq=8)
    tot = 0.0
    @inbounds for it in eachindex(lay.leaves)
        lf = lay.leaves[it]
        b = 4 * (it - 1)
        for (s, side) in enumerate(LOGICAL_SIDES)
            bmask[b + s] != 0x00 || continue
            fm = face_metric(g, lay, lf, side, fd)
            for q in 1:fd.nq
                tot += face_jacobian(fm, q) * fd.gl_w[q]
            end
        end
    end
    return tot
end


# ---------------------------------------------------------------------------
# 9. Topology dispatch
# ---------------------------------------------------------------------------

"""
    face_pairs(qt, g::AbstractGeometry) -> Vector{FacePair}

The face-pair topology of the tree on the geometry.  The box is served by the
historical ace_pairs(qt) (unchanged, GEOM_DESIGN §3 cut 1 keeps the box
body); every mapped geometry by ace_pairs_logical(qt), which is built from
the flat neighbour table with index arithmetic and never tests a physical
interval (GEOM_DESIGN §1.5(b)).
"""
face_pairs(qt::Quadtree, ::CartesianBox) = face_pairs(qt)
face_pairs(qt::Quadtree, ::AbstractGeometry) = face_pairs_logical(qt)

# ---------------------------------------------------------------------------
# 9b. The tree/geometry seam agreement (stage 0.5c1)
# ---------------------------------------------------------------------------

"""
    _geom_periodic(g::AbstractGeometry) -> Bool

Whether the geometry closes the logical `T` direction into a ring.  Only the
polar annulus can (`seam = :periodic`, and then `Phi = 2*pi` by construction);
every box-like geometry is bounded, and so is the annulus *wedge* (`seam =
:wall`, `Phi < 2*pi`, or even the cut ring `Phi = 2*pi` whose two seam images
coincide but are not glued).  The generic answer is `false`; the `PolarAnnulus`
method is in section 10 with the type.
"""
@inline _geom_periodic(::AbstractGeometry) = false

"""
    check_seam_compat(qt::Quadtree, g::AbstractGeometry)

Refuse a tree/geometry pair whose seam topologies disagree.  The periodic seam
is a **tree** feature (`Quadtree(...; seam = :periodic)`) while the map that
closes it is a **geometry** feature (`PolarAnnulus(...; seam = :periodic)`); one
without the other is not a ring and would silently run a half-glued topology:

  * a periodic tree with a bounded geometry would identify `T = 0`/`T = 1` nodes
    of a domain whose two sides are *not* the same curve (a wedge, a box);
  * a bounded tree with a periodic geometry would run the ring map with a cut:
    the seam faces are walls (free-slip would pin `u.n = 0` on a radial cut of
    what should be the same point), which is the silent non-periodic failure the
    stage-0.5a record warned about.

Called by [`make_semquad`](@ref) (so every space construction, including the MG
coarse levels, is checked) and by the driver before setup.
"""
function check_seam_compat(qt::Quadtree, g::AbstractGeometry)
    tree_periodic = qt.seam === :periodic
    geom_periodic = _geom_periodic(g)
    tree_periodic == geom_periodic && return nothing
    if tree_periodic
        throw(ArgumentError("the tree has seam = :periodic but the geometry " *
                            "($(geometry_name(g))) does not close: a periodic " *
                            "seam identifies the T = 0 and T = 1 logical sides, " *
                            "which is only a physical identification under a map " *
                            "that closes (PolarAnnulus with Phi = 2*pi and " *
                            "seam = :periodic).  Build the tree with " *
                            "Quadtree(...; seam = :none) for a wedge/box, or " *
                            "the geometry with seam = :periodic for a ring."))
    end
    throw(ArgumentError("the geometry ($(geometry_name(g))) closes its angular " *
                        "direction (seam = :periodic) but the tree has " *
                        "seam = :none: the seam faces would be treated as walls " *
                        "(a bounded tree does not identify T = 0 with T = 1), so " *
                        "the solve would silently be non-periodic.  Build the " *
                        "tree with Quadtree(...; seam = :periodic)."))
end

# ---------------------------------------------------------------------------
# 10. Stage 0.5a: the 2-D polar annulus (`PolarAnnulus`)
# ---------------------------------------------------------------------------
#
# All `PolarAnnulus` code lives in this section: the type and its constructor,
# every geometry method, the canonical identity and the cache stamp.  Nothing
# above is edited, so the box/affine/curved paths cannot see the annulus (the
# nine state-dump hashes are untouched) and the generic mapped bodies of
# sections 4-9 serve it unchanged.  This is the geometry half of stage 0.5; the
# radial physics wiring (base state, buoyancy, curved-wall boundary rows,
# diagnostics overrides, driver knobs) is stage 0.5b and touches none of this.
#
# MAP (GEOM_DESIGN §6.1, ANNULUS_REFS §3).  The canonical logical coordinates
# are the unit square `(S,T) in [0,1]^2`; the physical point is
#
#     (x,y) = (r cos phi, r sin phi),   r = R0 + d*S,  phi = Phi*T,  d = R1-R0,
#
# i.e. the composition `(S,T) -> (r,phi) -> (r cos phi, r sin phi)`.  With the
# benchmark length unit `d = 1` (every published geometry) this is the brief's
# `r = R0 + S`.  The tree's `Lx` never enters the annulus map: `global_logical`
# always returns `S in [0,1]`, so the radial range covers the whole logical
# x-range of the tree (build the tree with the default `Lx = 1`; the annulus
# does not read it, unlike `AffineBox`/`CurvedBox` which carry their own `Lx`).
#
# JACOBIAN (GEOM_DESIGN §6.2).  With
#
#     J_ST = d(x,y)/d(r,phi) = [[cos phi, -r sin phi],
#                               [sin phi,  r cos phi]]
#     d(r,phi)/d(S,T)       = diag(d, Phi)
#
# the global logical Jacobian and its inverse are
#
#     J_{S,T} = J_ST diag(d, Phi) = [[d cos phi, -Phi r sin phi],
#                                    [d sin phi,  Phi r cos phi]]
#     det J_{S,T} = d Phi r > 0            (orientation preserving; = Phi r for d=1)
#     J^{-1}_{S,T} = [[cos phi/d,        sin phi/d      ],
#                     [-sin phi/(Phi r), cos phi/(Phi r)]]
#     J^{-T}      = [[cos phi/d, -sin phi/(Phi r)],
#                    [sin phi/d,  cos phi/(Phi r)]]      (the cut-1 convention
#                                                          (xi_x, eta_x, xi_y, eta_y))
#
# The leaf reference Jacobian is the chain with the logical steps
# `dS/dxi = 1/(2 nside_x)` and `dT/deta = 1/(2 nside_y)`:
#
#     J_leaf      = J_{S,T} diag(1/(2 nside_x), 1/(2 nside_y))
#     det J_leaf  = Phi r d / (4 nside_x nside_y) = Phi r h_S h_T / 4,
#
# with `h_S = d/nside_x` the physical radial leaf extent and `h_T = 1/nside_y`
# the logical angular extent (the physical arc length is `Phi r h_T`).  The
# inverse `J^{-T}` is the *generic* `AbstractGeometry` method, i.e. derived from
# the same `J`/`det J` every mapped geometry uses (GEOM_DESIGN §2.3): the
# analytic expression above is what that derivation produces, and G273 pins
# both it and the chain-rule convention.
#
# SECOND DERIVATIVES (the cut-3 `element_hessian` surface).  Write `a = dS/dxi`,
# `b = dT/deta`; `r` is affine in `xi` and `phi` affine in `eta`, so
#
#     x_{xi xi} = 0,  x_{xi eta} = -Phi d a b sin phi,  x_{eta eta} = -Phi^2 r b^2 cos phi
#     y_{xi xi} = 0,  y_{xi eta} = +Phi d a b cos phi,  y_{eta eta} = -Phi^2 r b^2 sin phi
#
# A dropped term here is invisible on the box and is exactly the F4 defect of
# the cut-3 record; G273's coordinate-Laplacian identities measure it (the
# annulus is the geometry where the map-curvature term `C` cannot be dropped).
#
# MEASURES (GEOM_DESIGN §6.6).  The generic mapped bodies are consumed unchanged:
# `quad_measure = det J_leaf * gl_w * gl_w`, and integrating `det J` gives
#
#     total_measure = int_0^1 int_0^1 d Phi (R0 + d S) dS dT
#                   = d Phi (R0 + d/2) = Phi/2 (R1^2 - R0^2),
#
# the plane-annulus area.  `measure_weight` stays the abstract default `1.0`:
# the benchmarks are the 2-D annulus in the plane (ANNULUS_REFS §3, "area
# integrals, arc-length boundary integrals, heat flows per unit cylinder
# length"), not a 3-D axisymmetric shell reduction; a `2*pi*r` weight would be a
# different benchmark and a different Nu normalisation.  The wall measure is the
# arc length: `boundary_measure(:xminus) = Phi R0`, `(:xplus) = Phi R1`; the
# wedge's radial sides have length `d` each.

"""
    PolarAnnulus(R0, R1, Φ; seam=:wall) <: AbstractGeometry

Stage 0.5a geometry: the 2-D polar annulus `R0 <= r <= R1`, `0 <= phi <= Φ`,
a wedge of the plane with an inner and an outer circular wall and (for the
wedge) two radial seam sides.  This is the 2-D Cartesian annulus of
(the benchmark's "2-D cylindrical shell"), not a
shell of a 3-D sphere.

The logical coordinates are `S in [0,1]` radial and `T in [0,1]` angular, and
the map is

    (x, y) = (r cos(Φ T), r sin(Φ T)),    r = R0 + (R1 - R0) S.

The published benchmark geometry is `R0 = 1.22`, `R1 = 2.22` (`R0 = 0.55` in
units of the gap), i.e. `R1 - R0 = 1`; a general thickness is supported by the
map and every metric/measure, but the physics closures are functions of the
**nondimensional height** in `[0,1]`, so the gap `R1 - R0` is the length unit of
the reductions (`height = r - R0`; `depth = 1 - height`).  The constructor does
not require `R1 - R0 == 1` exactly: the published radii give `R1 - R0 =
1 + 2^-52` in `Float64`, so an exact equality would reject the benchmark.

`height = hypot(x,y) - R0` and `gravity = -e_r` with `|g| = 1` (radial,
constant magnitude, inward: ANNULUS_REFS §5.1).  The logical boundaries are

| side | physical | tag |
|---|---|---|
| `xminus` | inner circle `r = R0`, outward normal `-e_r` | `:inner` |
| `xplus`  | outer circle `r = R1`, outward normal `+e_r` | `:outer` |
| `yminus` | radial side `phi = 0`, outward normal `-e_phi(0)` | `:seam_lo` |
| `yplus`  | radial side `phi = Φ`, outward normal `+e_phi(Φ)` | `:seam_hi` |

and the Nusselt pair is the two circles (`nu_wall_sides = (:xminus, :xplus)`).

**The seam (stage 0.5c1: wedge or ring).**  `seam = :wall` is the wedge (or
the cut ring): the two radial sides are ordinary logical walls (free-slip radial
cuts; the boundary registry tags them `(:seam_lo, :seam_hi)`).  `seam =
:periodic` is the closed 360-degree ring: the `T = 0` and `T = 1` images are the
same physical curve, the two seam sides are **not** walls, and the registry
reports them as `kind = :periodic` with `tag = :seam_lo`/`:seam_hi`.  The seam is
a **quadtree topology** feature and lives on the tree
(`Quadtree(...; seam = :periodic)`: cross-seam neighbour pairs, the node
identity across the seam, the 2:1 balance across it, periodic transfer/MG/AMR);
the geometry contributes only the map that closes and the registry entry.  The
two halves must agree -- a ring geometry on a bounded tree, or a periodic tree on
a wedge, is refused by [`check_seam_compat`](@ref) -- and the ring constructor
requires the opening angle to close **exactly**:

* `seam = :periodic` requires `Φ == 2*pi` (the `Float64` value).  Any other angle
  leaves a gap between the two seam images, so identifying them would be a
  topology that the map does not realise; the constructor refuses with the
  measured angle and tells the caller to pass `2*pi` (or to use the wedge).
  There is deliberately no tolerance-and-snap: the geometry identity and the
  stamp carry `Φ` verbatim, so snapping would change a parameter the checkpoint
  records.  A caller whose `Φ` is one ulp off `2*pi` gets the refusal and the
  exact value to write;
* `Φ == 2*pi` with `seam = :wall` stays constructible as a *cut ring* (all four
  sides are walls and the two seam images coincide to round-off, but they are
  not glued: `perimeter` counts both radial sides and the seam faces are walls).
  This is useful for metric tests and is not the periodic benchmark.

Validation: `R0 > 0`, `R1 > R0`, `0 < Φ <= 2*pi`, all finite and the map
orientation-preserving (`(R1 - R0) * Φ` finite and positive).
"""
struct PolarAnnulus <: AbstractGeometry
    R0::Float64
    R1::Float64
    Φ::Float64
    seam::Symbol

    function PolarAnnulus(R0::Float64, R1::Float64, Φ::Float64, seam::Symbol)
        isfinite(R0) && isfinite(R1) && isfinite(Φ) ||
            throw(ArgumentError("PolarAnnulus: R0, R1 and Phi must be finite " *
                                "(got R0 = $R0, R1 = $R1, Phi = $Φ)"))
        R0 > 0.0 ||
            throw(ArgumentError("PolarAnnulus: the inner radius must be positive " *
                                "(R0 = $R0); r = 0 is not in the domain"))
        R1 > R0 ||
            throw(ArgumentError("PolarAnnulus: R1 must be strictly greater than R0 " *
                                "(got R0 = $R0, R1 = $R1)"))
        0.0 < Φ <= 2.0 * pi ||
            throw(ArgumentError("PolarAnnulus: Phi must be in (0, 2pi] -- a " *
                                "quarter/half/three-quarter/full wedge in radians " *
                                "(got Phi = $Φ)"))
        d = R1 - R0
        (isfinite(d) && d > 0.0 && isfinite(d * Φ)) ||
            throw(ArgumentError("PolarAnnulus: the map must be finite and " *
                                "orientation-preserving, det J = Phi*(R1-R0)*r > 0 " *
                                "for r > 0 (got (R1-R0)*Phi = $(d * Φ))"))
        if seam === :periodic
            Φ == 2.0 * pi ||
                throw(ArgumentError("PolarAnnulus: seam = :periodic is the closed " *
                                    "360-degree ring and requires the opening angle " *
                                    "to close exactly (Phi == 2*pi); got Phi = $Φ, " *
                                    "which leaves a gap of $(2.0 * pi - Φ) rad " *
                                    "between the two seam images, so identifying " *
                                    "them would not be the map.  Pass Phi = 2*pi " *
                                    "for the ring, or seam = :wall for a wedge (a " *
                                    "cut ring at Phi = 2*pi with seam = :wall is " *
                                    "the bounded topology)."))
        elseif seam !== :wall
            throw(ArgumentError("PolarAnnulus: seam must be :wall or :periodic " *
                                "(got :$seam)"))
        end
        return new(R0, R1, Φ, seam)
    end
end

"""
    _geom_periodic(g::PolarAnnulus) -> Bool

The `PolarAnnulus` method of [`_geom_periodic`](@ref): `true` only for the ring
(`seam = :periodic`, which the constructor restricts to `Phi = 2*pi`).  The
wedge and the cut ring (`Phi = 2*pi`, `seam = :wall`) are bounded.
"""
@inline _geom_periodic(g::PolarAnnulus) = g.seam === :periodic

PolarAnnulus(R0::Real, R1::Real, Φ::Real; seam::Symbol=:wall) =
    PolarAnnulus(Float64(R0), Float64(R1), Float64(Φ), seam)
PolarAnnulus(R0::Real, R1::Real, Φ::Real, seam::Symbol) =
    PolarAnnulus(Float64(R0), Float64(R1), Float64(Φ), seam)

# --- identity (checkpoint config, restart guard, cache stamp) ---------------

geometry_name(::PolarAnnulus) = :PolarAnnulus

"""
    geometry_identity(g::PolarAnnulus) -> String

Canonical identity of the annulus: the two radii, the wedge angle and the seam
mode.  Every shape parameter and the seam are included, so the restart guard
compares the ring/wedge decision as well as the shape (a future ring must not
resume a wedge checkpoint); the string is stable and round-tripping
(`_geom_number`).
"""
function geometry_identity(g::PolarAnnulus)
    return string("PolarAnnulus(R0=", _geom_number(g.R0),
                  ",R1=", _geom_number(g.R1),
                  ",Phi=", _geom_number(g.Φ),
                  ",seam=:", g.seam, ")")
end

# The cache stamp of the metric/curvature/face cache (cut 5): the same
# discipline as the other geometries -- a type tag plus the raw Float64 bits of
# every shape parameter and the seam.  Separate from `geometry_identity` because
# it must not allocate on a cache lookup.
function geometry_stamp(g::PolarAnnulus)
    return hash((UInt8(4), reinterpret(UInt64, g.R0), reinterpret(UInt64, g.R1),
                 reinterpret(UInt64, g.Φ), UInt8(g.seam === :wall ? 1 : 2)))
end

# --- the map, its analytic inverse, the Jacobian, the Hessian --------------

@inline function logical_map(g::PolarAnnulus, S::Float64, T::Float64)
    r = g.R0 + (g.R1 - g.R0) * S
    phi = g.Φ * T
    return (r * cos(phi), r * sin(phi))
end

"""
    element_inverse(g::PolarAnnulus, lay, x, y) -> (S, T)

The analytic inverse of the annulus map:

    r = hypot(x, y),   S = (r - R0)/(R1 - R0),
    phi = atan(y, x) wrapped into [0, 2pi),   T = phi/Φ.

Defined for every point with `r > 0` (the origin is never in the domain because
`R0 > 0`); a point outside the wedge's angular sector gives `T` outside
`[0,1]`, which is the analytic continuation `locate_leaf` clamps exactly as for
every mapped geometry.  The `atan` branch matters for `Φ > pi`: a point with
`phi in (pi, 2pi)` has `atan(y, x) < 0` and must be wrapped before dividing by
`Φ` (G273 feeds such points).
"""
@inline function element_inverse(g::PolarAnnulus, lay, x::Float64, y::Float64)
    r = hypot(x, y)
    S = (r - g.R0) / (g.R1 - g.R0)
    phi = atan(y, x)
    phi < 0.0 && (phi += 2.0 * pi)
    return (S, phi / g.Φ)
end

"""
    element_jacobian(g::PolarAnnulus, lay, lf, xi, eta) -> (J, detJ)

The leaf-reference Jacobian `J = (x_xi, x_eta, y_xi, y_eta)` of the annulus map
and its determinant.  Evaluated in cold paths and in the metric build (never in
a quadrature loop); the derivation is in the section header.  `detJ` is formed
from the returned `J` entries (`a*d - b*c`), so it is the determinant of exactly
the matrix the kernels differentiate with, and it is positive: `Phi r d h_S h_T`
with `h_S = 1/(2 nside_x)`, `h_T = 1/(2 nside_y)`.
"""
@inline function element_jacobian(g::PolarAnnulus, lay, lf::NTuple{3,Int},
                                  xi::Float64, eta::Float64)
    S, T = global_logical(lay, lf, xi, eta)
    d = g.R1 - g.R0
    r = g.R0 + d * S
    phi = g.Φ * T
    l = lf[1]
    hS = 0.5 / leaf_nside_x(lay, l)      # dS/dxi
    hT = 0.5 / leaf_nside_y(lay, l)      # dT/deta
    cphi = cos(phi)
    sphi = sin(phi)
    a = cphi * d * hS                    # x_xi
    b = -g.Φ * r * sphi * hT             # x_eta
    c = sphi * d * hS                    # y_xi
    e = g.Φ * r * cphi * hT              # y_eta
    return ((a, b, c, e), a * e - b * c)
end

"""
    element_hessian(g::PolarAnnulus, lay, lf, xi, eta) -> (x11, x12, x22, y11, y12, y22)

The second derivatives of the annulus map with respect to the leaf reference
coordinates (the `element_hessian` interface of cut 3, consumed by
`map_curvature`/`lap_phi`).  The derivation is in the section header; the two
entries that are identically zero (`x_{xi xi}` and `y_{xi xi}`, because `r` is
affine in `xi` and `phi` affine in `eta`) are returned as `0.0` exactly.
"""
@inline function element_hessian(g::PolarAnnulus, lay, lf::NTuple{3,Int},
                                 xi::Float64, eta::Float64)
    S, T = global_logical(lay, lf, xi, eta)
    d = g.R1 - g.R0
    r = g.R0 + d * S
    phi = g.Φ * T
    l = lf[1]
    hS = 0.5 / leaf_nside_x(lay, l)
    hT = 0.5 / leaf_nside_y(lay, l)
    f2 = g.Φ * g.Φ * r * hT * hT
    cphi = cos(phi)
    sphi = sin(phi)
    # (x_xi xi, x_xi eta, x_eta eta, y_xi xi, y_xi eta, y_eta eta)
    return (0.0, -g.Φ * d * hS * hT * sphi, -f2 * cphi,
            0.0, g.Φ * d * hS * hT * cphi, -f2 * sphi)
end

# --- physical height, depth, gravity ---------------------------------------

@inline function height(g::PolarAnnulus, x::Float64, y::Float64)
    return hypot(x, y) - g.R0
end
@inline depth(g::PolarAnnulus, x::Float64, y::Float64) = 1.0 - height(g, x, y)
@inline function grad_height(g::PolarAnnulus, x::Float64, y::Float64)
    r = hypot(x, y)
    return (x / r, y / r)                # e_r
end
@inline function height_laplacian(g::PolarAnnulus, x::Float64, y::Float64)
    return 1.0 / hypot(x, y)             # laplacian r = 1/r in 2-D
end
@inline function gravity(g::PolarAnnulus, x::Float64, y::Float64)
    r = hypot(x, y)
    return (-x / r, -y / r)              # -e_r, constant unit magnitude
end
@inline gravity_magnitude(::PolarAnnulus, x::Float64, y::Float64) = 1.0

# --- boundary registry -----------------------------------------------------

"""
    boundaries(g::PolarAnnulus) -> NTuple{4,LogicalBoundary}

The registry: the two circles are `:inner`/`:outer` walls and the two radial
sides are `:seam_lo`/`:seam_hi`.  The seam sides are `:wall` for the wedge (and
the cut ring) and **`:periodic` for the ring** (stage 0.5c1), where they are not
domain boundary at all: the wall loops never see them because the periodic tree
reports no face without a neighbour on the seam sides
([`_semquad_boundary_face_mask`](@ref) is built from
[`has_no_face_neighbor`](@ref)), and a consumer that reads the registry instead
of the per-face mask must skip a `:periodic` side rather than apply a wall
condition to it.  The tag names stay `:seam_lo`/`:seam_hi` for both topologies,
so the wedge metadata is unchanged.
"""
function boundaries(g::PolarAnnulus)
    sk = g.seam === :periodic ? :periodic : :wall
    return (LogicalBoundary(:xminus, :wall, :inner),
            LogicalBoundary(:xplus, :wall, :outer),
            LogicalBoundary(:yminus, sk, :seam_lo),
            LogicalBoundary(:yplus, sk, :seam_hi))
end

"""
    nu_wall_sides(g::PolarAnnulus) -> (:xminus, :xplus)

The annulus Nusselt pair is the two circles (inner/hot bottom, outer/cold top),
which is why the diagnostics ask the geometry instead of hard-coding the two
logical `y` sides (GEOM_DESIGN §6.3/§6.6).
"""
@inline nu_wall_sides(::PolarAnnulus) = (:xminus, :xplus)

# --- inference budget for the geometry API (stage 0.5a finding) -------------
#
# `SemQuad.geom` is deliberately non-parametric (cut 1: the struct is NOT
# parametric and the field is `AbstractGeometry`), so a cold call site that only
# knows the abstract type asks inference to resolve a generic function that now
# has one method per concrete subtype.  Julia's default `max_methods = 3` makes
# inference give up (return `Any`) as soon as a *fourth* concrete geometry
# exists.  The box path's arithmetic does not change, but its inference does,
# and two existing pins saw exactly that when `PolarAnnulus` was added:
#
#   * `_semquad_kop_key` (`stokes/kop.jl`) calls `geometry_stamp(sq.geom)`; with
#     `Any` the 9-field key tuple is heap-allocated, so the warm public mapped
#     apply allocates 128 B (G271 (c) measured exactly 128; the baseline is 0);
#   * `semquad_work_dissipation` (`diagnostics/global.jl`) takes
#     `g = sq.geom` and its mapped fallback calls `height(g, x, y)`; with `Any`
#     the box branch's merged return type widens to `Tuple{Any, Float64}` and
#     the `@inferred` pin fails (G188).
#
# The fix is a compiler *budget*, not a code path: raise `max_methods` to 4 for
# the geometry API functions whose method count is one-per-subtype.  This is
# `Base.Experimental.@max_methods` (a documented inference setting; it changes
# no body, no dispatch result and no allocation on a concrete call).  With the
# budget at 4, `_semquad_kop_key` infers `Tuple{..., UInt64}` and the warm
# mapped public apply is 0 B again, and `semquad_work_dissipation`'s box
# inference is `Tuple{Float64, Float64}` again; G188/G271 pass exactly as on the
# cut-6 tree.  A fifth production geometry would raise these to 5.
Base.Experimental.@max_methods 4 function logical_map end
Base.Experimental.@max_methods 4 function element_inverse end
Base.Experimental.@max_methods 4 function element_jacobian end
Base.Experimental.@max_methods 4 function element_hessian end
Base.Experimental.@max_methods 4 function height end
Base.Experimental.@max_methods 4 function depth end
Base.Experimental.@max_methods 4 function grad_height end
Base.Experimental.@max_methods 4 function height_laplacian end
Base.Experimental.@max_methods 4 function gravity end
Base.Experimental.@max_methods 4 function gravity_magnitude end
Base.Experimental.@max_methods 4 function boundaries end
Base.Experimental.@max_methods 4 function geometry_name end
Base.Experimental.@max_methods 4 function geometry_identity end
Base.Experimental.@max_methods 4 function geometry_stamp end
