# Geometry / mapping layer (stage 0, cut 1).
#
# The solver's physical domain is produced by one global map from the logical
# rectangle `[0,Lx] x [0,1]` (the quadtree's logical coordinates `(S,T)`), and a
# leaf-local affine map from the reference square `[-1,1]^2`.  This file owns
# both, plus everything the operators read from the geometry:
#
#   * the four types `CartesianBox` (production; the axis-aligned box with the
#     identity map), `AffineBox` and `CurvedBox` (validation geometries), all
#     dispatching from `AbstractGeometry`;
#   * the point map, its inverse, the per-element Jacobian, `det J` and `J^{-1}`;
#   * the leaf metric (the per-leaf scalars / per-quadrature-point arrays the
#     kernels read) and the hot-loop accessors `quad_weight`/`quad_point`/
#     `basis_grad`;
#   * the face metric (face Jacobian, outward unit normal, positive tangent) and
#     the hot-loop accessors `face_point`/`face_weight`/`face_normal`/
#     `face_tangent`;
#   * the logical boundary registry, the physical height/depth and gravity;
#   * the logical topology helpers `global_logical`, `leaf_ref`,
#     `locate_leaf_logical`, `locate_leaf`;
#   * `geometry_identity`, the canonical checkpoint signature.
#
# THE BOX PATH IS A SPECIALIZATION, NOT A ROUTE THROUGH THE GENERAL CODE.
# `CartesianBox` methods are the current solver's expressions, character for
# character (`docs/dev/GEOM_DESIGN.md` §4.2, rows E1-E16): `ihx = 2.0/hx`,
# `wfac = 0.25*hx*hy`, `x = x0 + 0.5*(xi+1)*hx`, `wt = 0.5*h*gl_w[q]`, ...
# The general Jacobian route is never reachable with a `CartesianBox`: the box
# `basis_grad` is `dphi*phi*ihx`, not `(J22*phi_xi - J21*phi_eta)/detJ`, and
# `quad_weight` returns the stored `wfac`, not a recomputed `det J`.  A general
# rewrite moves the nine state-dump hashes (GEOM_DESIGN §4.3 F1-F7), so do not
# "clean it up".
#
# The validation geometries are NOT reachable from any card in cut 1; they are
# the proof of the interface (gate G266) and the rehearsal for the annulus
# (`docs/dev/GEOM_DESIGN.md` §5, `docs/dev/GEOM_PROGRESS.md`).

const LOGICAL_SIDES = (:xminus, :xplus, :yminus, :yplus)

# ---------------------------------------------------------------------------
# 1. Types
# ---------------------------------------------------------------------------

"""
    AbstractGeometry

A global map from the logical rectangle `[0,Lx] x [0,1]` to the physical
domain, plus its derived metrics and boundary classification.  Immutable and
hashable (structural `==`/`hash` are the defaults for these structs): a
geometry is a cache key, and two geometries compare equal only when every shape
parameter is equal.

The interface a new geometry must implement (see the methods below):
`logical_map`, `element_jacobian`, `jacobian_inverse` (or derive it),
`height`/`depth`/`grad_height`/`height_laplacian`/`gravity`,
`boundaries`.  The quad/face hot-loop accessors are generic over the two metric
types and only need `element_jacobian` + `logical_map`.
"""
abstract type AbstractGeometry end

"""
    CartesianBox(Lx)
    CartesianBox(qt::Quadtree)

Production geometry: the axis-aligned box `[0,Lx] x [0,1]` under the identity
map `(x,y) = (Lx*S, T)`.  This is the historical geometry and its path is the
historical arithmetic; see the file header.  `Lx` is the physical width and the
height is 1 (`core/quadtree.jl`).
"""
struct CartesianBox <: AbstractGeometry
    Lx::Float64
    function CartesianBox(Lx::Float64)
        isfinite(Lx) && Lx > 0.0 ||
            throw(ArgumentError("CartesianBox: Lx must be positive and finite (got $Lx)"))
        return new(Lx)
    end
end
CartesianBox(qt::Quadtree) = CartesianBox(qt.Lx)

"""
    AffineBox(A, b, Lx)

Stage-0 **validation** geometry: the box under a general affine map

    (x,y) = A * (Lx*S, T) + b,

with `A = (A11, A12, A21, A22)` a 2x2 matrix and `b = (b1, b2)` a shift.  The
physical domain is the parallelogram image of `[0,Lx] x [0,1]`.  `det A > 0` is
required so the map preserves orientation and the outward normals point out.
This type exists to exercise the *general* mapped code on a geometry where a
polynomial manufactured solution is still exact (GEOM_DESIGN §5.2/G267); no
card selects it in cut 1.
"""
struct AffineBox <: AbstractGeometry
    A::NTuple{4,Float64}   # A11, A12, A21, A22
    b::NTuple{2,Float64}
    Lx::Float64
    function AffineBox(A::NTuple{4,Float64}, b::NTuple{2,Float64},
                       Lx::Float64)
        detA = A[1] * A[4] - A[2] * A[3]
        isfinite(detA) && detA > 0.0 ||
            throw(ArgumentError("AffineBox: det(A) = $detA must be positive " *
                                "and finite (the map must preserve orientation)"))
        isfinite(Lx) && Lx > 0.0 ||
            throw(ArgumentError("AffineBox: Lx must be positive and finite (got $Lx)"))
        all(isfinite, A) && all(isfinite, b) ||
            throw(ArgumentError("AffineBox: A and b must be finite"))
        return new(A, b, Lx)
    end
end
function AffineBox(A::NTuple{4,<:Real}, b::NTuple{2,<:Real}, Lx::Real)
    return AffineBox((Float64(A[1]), Float64(A[2]), Float64(A[3]), Float64(A[4])),
                     (Float64(b[1]), Float64(b[2])), Float64(Lx))
end
AffineBox(; A::NTuple{4,<:Real}=(1.0, 0.0, 0.0, 1.0),
          b::NTuple{2,<:Real}=(0.0, 0.0), Lx::Real=1.0) = AffineBox(A, b, Lx)

"""
    CurvedBox(a, Lx)

Stage-0 **validation** geometry: a smooth non-affine map with an analytic
inverse and one curved side,

    x = Lx*S,   y = T * (1 + a*sin(pi*S)),   |a| < 1,

so `det J = Lx*(1 + a*sin(pi*S)) > 0`.  The logical `yplus` wall is the curve
`y = 1 + a*sin(pi*x/Lx)`; the other three sides are straight.  Inverse:
`S = x/Lx`, `T = y/(1 + a*sin(pi*S))`.  Used by the non-affine manufactured
solution gate (GEOM_DESIGN §5.4/G269); no card selects it in cut 1.
"""
struct CurvedBox <: AbstractGeometry
    a::Float64
    Lx::Float64
    function CurvedBox(a::Float64, Lx::Float64)
        isfinite(a) && abs(a) < 1.0 ||
            throw(ArgumentError("CurvedBox: |a| < 1 is required (got a = $a)"))
        isfinite(Lx) && Lx > 0.0 ||
            throw(ArgumentError("CurvedBox: Lx must be positive and finite (got $Lx)"))
        return new(a, Lx)
    end
end
CurvedBox(a::Real, Lx::Real) = CurvedBox(Float64(a), Float64(Lx))
CurvedBox(; a::Real=0.0, Lx::Real=1.0) = CurvedBox(Float64(a), Float64(Lx))

# ---------------------------------------------------------------------------
# 2. Canonical identity (checkpoint config)
# ---------------------------------------------------------------------------

geometry_name(::CartesianBox) = :CartesianBox
geometry_name(::AffineBox) = :AffineBox
geometry_name(::CurvedBox) = :CurvedBox

# `string(Float64)` is the shortest round-tripping decimal form, so the record
# carries every parameter exactly and is stable across processes/machines.
_geom_number(x::Float64) = string(x)

"""
    geometry_identity(g::AbstractGeometry) -> String

The canonical identity of a geometry: its kind and every shape parameter, in a
fixed order, formatted so that equal geometries give equal strings and a
different geometry cannot share one.  This is what the checkpoint config
records (key `"geometry"`) and what the restart guard compares; it is also the
natural hash key for any geometry-dependent cache added by later cuts.
"""
function geometry_identity(g::CartesianBox)
    return string("CartesianBox(Lx=", _geom_number(g.Lx), ")")
end
function geometry_identity(g::AffineBox)
    A = g.A
    return string("AffineBox(A11=", _geom_number(A[1]),
                  ",A12=", _geom_number(A[2]),
                  ",A21=", _geom_number(A[3]),
                  ",A22=", _geom_number(A[4]),
                  ",b1=", _geom_number(g.b[1]),
                  ",b2=", _geom_number(g.b[2]),
                  ",Lx=", _geom_number(g.Lx), ")")
end
function geometry_identity(g::CurvedBox)
    return string("CurvedBox(a=", _geom_number(g.a),
                  ",Lx=", _geom_number(g.Lx), ")")
end

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
"""
@inline function _logical_node_key(qt::Quadtree, lf::NTuple{3,Int},
                                   i::Int, j::Int, rule)
    l = lf[1]
    scale = Float64(1 << (qt.maxlevel - l))
    Snum = (2.0 * lf[2] + (rule.nodes[i] + 1.0)) * scale
    Tnum = (2.0 * lf[3] + (rule.nodes[j] + 1.0)) * scale
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

The global map from logical `(S,T) in [0,Lx] x [0,1]` to physical coordinates.
This is the only place a geometry defines its map; every mapped point in the
solver goes through it (directly or through `element_map`).
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

# ---------------------------------------------------------------------------
# 5. Leaf metric and the quadrature-loop accessors
# ---------------------------------------------------------------------------

"""
    BoxLeafMetric

The box metric: the seven scalars every box kernel reads, built once per leaf
and constant across the leaf.  `ihx = 2.0/hx`, `ihy = 2.0/hy` (the literal
reciprocals the code forms today) and `wfac = 0.25*hx*hy` (the volume factor
`0.25*hx*hy` in its historical association).  See GEOM_DESIGN E4/E5.
"""
struct BoxLeafMetric
    x0::Float64
    y0::Float64
    hx::Float64
    hy::Float64
    ihx::Float64
    ihy::Float64
    wfac::Float64
end

@inline function leaf_metric(::CartesianBox, lay, lf::NTuple{3,Int})
    hx = leaf_hx(lay, lf[1])
    hy = leaf_hy(lay, lf[1])
    x0 = lf[2] * hx
    y0 = lf[3] * hy
    return BoxLeafMetric(x0, y0, hx, hy, 2.0 / hx, 2.0 / hy, 0.25 * hx * hy)
end

"""
    MappedLeafMetric

The mapped metric: flat leaf-major arrays over the `nq x nq` quadrature points,
built once per `(geometry, leaf, p, nq)` and only read in the kernel loops.
`jinv[1..4]` are the four `J^{-T}` components of [`jacobian_inverse`](@ref) in
the same order; `xq`/`yq` are the mapped quadrature points and `detJ` the
volume factor.  A mapped kernel indexes them flat (`(qy-1)*nq + qx`).
"""
struct MappedLeafMetric
    xq::Vector{Float64}
    yq::Vector{Float64}
    detJ::Vector{Float64}
    jinv::NTuple{4,Vector{Float64}}
    nq::Int
end

@inline _leaf_metric_index(m::MappedLeafMetric, qx::Int, qy::Int) =
    (qy - 1) * m.nq + qx

"""
    leaf_metric(g, lay, lf, fd) -> MappedLeafMetric

Build the mapped leaf metric of a non-box geometry at the `fd.nq x fd.nq`
quadrature points (the map, Jacobian and inverse are evaluated once per point;
the kernels only index the arrays).  The box path deliberately does not use
this: its methods are the seven `BoxLeafMetric` scalars.
"""
function leaf_metric(g::AbstractGeometry, lay, lf::NTuple{3,Int}, fd::FEData)
    nq = fd.nq
    n2 = nq * nq
    xq = Vector{Float64}(undef, n2)
    yq = Vector{Float64}(undef, n2)
    detJ = Vector{Float64}(undef, n2)
    j11 = Vector{Float64}(undef, n2)
    j12 = Vector{Float64}(undef, n2)
    j21 = Vector{Float64}(undef, n2)
    j22 = Vector{Float64}(undef, n2)
    @inbounds for qy in 1:nq, qx in 1:nq
        k = (qy - 1) * nq + qx
        xi = fd.gl_x[qx]
        eta = fd.gl_x[qy]
        S, T = global_logical(lay, lf, xi, eta)
        x, y = logical_map(g, S, T)
        J, dj = element_jacobian(g, lay, lf, xi, eta)
        inv = _jacobian_inverse_from_jac(J, dj)
        xq[k] = x
        yq[k] = y
        detJ[k] = dj
        j11[k] = inv[1]
        j12[k] = inv[2]
        j21[k] = inv[3]
        j22[k] = inv[4]
    end
    return MappedLeafMetric(xq, yq, detJ, (j11, j12, j21, j22), nq)
end

"""
    quad_weight(m, fd, qx, qy) -> Float64

Volume weight of quadrature point `(qx,qy)`: `det J` for a mapped metric, the
stored `wfac` for a box metric (which the caller multiplies by the rule
weights), i.e. the historical `0.25*hx*hy*gl_w[qx]*gl_w[qy]` (GEOM_DESIGN E4).
"""
@inline quad_weight(m::BoxLeafMetric, fd::FEData, qx::Int, qy::Int) =
    m.wfac * fd.gl_w[qx] * fd.gl_w[qy]
@inline quad_weight(m::MappedLeafMetric, fd::FEData, qx::Int, qy::Int) =
    m.detJ[_leaf_metric_index(m, qx, qy)]

"""
    quad_measure(m, fd, qx, qy) -> Float64

The **fully weighted** quadrature volume measure `|det J| * gl_w[qx] * gl_w[qy]`
— the single quantity a volume quadrature loop sums.  The box method is the
historical expression of GEOM_DESIGN E4, `wfac * gl_w[qx] * gl_w[qy]` with
`wfac = 0.25*hx*hy` (the same operand and association as
`0.25*hx*hy*gl_w[qx]*gl_w[qy]`).

*Cut-2 interface note.*  `quad_weight` above is **not** interchangeable between
the two metric types: its box method keeps the historical E4 form, which already
contains the rule weights, because G266 pins it bit for bit against
`0.25*hx*hy*gl_w`; its mapped method returns the bare `detJ`, as the cut-1
docstring states.  A generic kernel that must run on both metrics therefore
uses this accessor, which includes the rule weights on both sides.  The two
accessors are pinned separately: `quad_weight` by G266 (per-metric historical
form), `quad_measure` by the cut-2 gates (assembled operators).
"""
@inline quad_measure(m::BoxLeafMetric, fd::FEData, qx::Int, qy::Int) =
    m.wfac * fd.gl_w[qx] * fd.gl_w[qy]
@inline quad_measure(m::MappedLeafMetric, fd::FEData, qx::Int, qy::Int) =
    m.detJ[_leaf_metric_index(m, qx, qy)] * fd.gl_w[qx] * fd.gl_w[qy]

"""
    quad_point(m, lay, lf, fd, qx, qy) -> (x, y)

Physical quadrature point.  The box method is the historical point map
(GEOM_DESIGN E3); the mapped method returns the precomputed array entry.
"""
@inline function quad_point(m::BoxLeafMetric, lay, lf::NTuple{3,Int},
                            fd::FEData, qx::Int, qy::Int)
    return (m.x0 + 0.5 * (fd.gl_x[qx] + 1.0) * m.hx,
            m.y0 + 0.5 * (fd.gl_x[qy] + 1.0) * m.hy)
end
@inline function quad_point(m::MappedLeafMetric, lay, lf::NTuple{3,Int},
                            fd::FEData, qx::Int, qy::Int)
    k = _leaf_metric_index(m, qx, qy)
    return (m.xq[k], m.yq[k])
end

"""
    basis_grad(m, fd, qx, qy, i, j) -> (gx, gz)

Physical gradient of the tensor-product basis function `(i,j)` at the
quadrature point `(qx,qy)`.  The box method is `dphi*phi*ihx`,
`phi*dphi*ihy` in that exact association (GEOM_DESIGN E5); the mapped method is
the chain rule, `Jinv11*dphi*phi + Jinv12*phi*dphi` per component.
"""
@inline function basis_grad(m::BoxLeafMetric, fd::FEData, qx::Int, qy::Int,
                            i::Int, j::Int)
    return (fd.dphi_v[i, qx] * fd.phi_v[j, qy] * m.ihx,
            fd.phi_v[i, qx] * fd.dphi_v[j, qy] * m.ihy)
end
@inline function basis_grad(m::MappedLeafMetric, fd::FEData, qx::Int, qy::Int,
                            i::Int, j::Int)
    k = _leaf_metric_index(m, qx, qy)
    dxi = fd.dphi_v[i, qx] * fd.phi_v[j, qy]
    deta = fd.phi_v[i, qx] * fd.dphi_v[j, qy]
    gx = dxi * m.jinv[1][k] + deta * m.jinv[2][k]
    gz = dxi * m.jinv[3][k] + deta * m.jinv[4][k]
    return (gx, gz)
end

"""
    basis_grad_p(m, fd, qx, qy, i, j) -> (gx, gz)

Physical gradient of the **pressure** tensor-product basis function `(i,j)`
(the `fd.phi_p`/`fd.dphi_p` family, degree `fd.q`) at the quadrature point.
Needed because the pressure mass/stiffness of `stokes/assembly.jl` live on the
lower-degree pressure space; `basis_grad` is the velocity (degree `fd.p`)
family.  The box method is `dphi_p*phi_p*ihx` in the historical association; the
mapped method is the same chain rule as `basis_grad`.
"""
@inline function basis_grad_p(m::BoxLeafMetric, fd::FEData, qx::Int, qy::Int,
                              i::Int, j::Int)
    return (fd.dphi_p[i, qx] * fd.phi_p[j, qy] * m.ihx,
            fd.phi_p[i, qx] * fd.dphi_p[j, qy] * m.ihy)
end
@inline function basis_grad_p(m::MappedLeafMetric, fd::FEData, qx::Int,
                              qy::Int, i::Int, j::Int)
    k = _leaf_metric_index(m, qx, qy)
    dxi = fd.dphi_p[i, qx] * fd.phi_p[j, qy]
    deta = fd.phi_p[i, qx] * fd.dphi_p[j, qy]
    gx = dxi * m.jinv[1][k] + deta * m.jinv[2][k]
    gz = dxi * m.jinv[3][k] + deta * m.jinv[4][k]
    return (gx, gz)
end

# ---------------------------------------------------------------------------
# 5b. Map curvature: the second derivatives of the map (cut 3)
# ---------------------------------------------------------------------------
#
# The energy path is the only part of the solver that differentiates twice: the
# entropy-viscosity residual needs the physical Laplacian of the temperature
# perturbation, and the base-state source needs `lap(Tbar(h))` (`GEOM_DESIGN`
# §2.7/§4.3 F4/§8 risk 10).  The physical Laplacian of a function represented on
# the reference element is, with `Jinv = J^{-T}` (the `jacobian_inverse` order
# `(xi_x, eta_x, xi_y, eta_y)`) and `H_p` the Hessian of the physical coordinate
# `p` in the leaf reference coordinates,
#
#     lap f = sum_{j,k} M[j,k] f_{xi_j xi_k} + sum_j C[j] f_{xi_j}
#     M     = Jinv^T Jinv                       (the metric tensor)
#     C[j]  = -sum_p Jinv[p,j] * tr(M * H_p)    (the map-curvature term)
#
# The `C` term is the one a naive implementation drops: it vanishes for an
# affine map (`H_p = 0`) and for the axis-aligned box, so the box cannot see it,
# but on any map whose Jacobian varies it is the same order as the `M` term
# (the measured failure is in `test/t_g269_curved_energy.jl`, part (d)).  The
# cross metric `M[xi,eta]` is nonzero whenever `x_eta != 0` or `y_xi != 0`
# (a sheared or curved element), so the mixed reference second derivative
# `f_{xi eta}` cannot be dropped either.
#
# THE BOX METHOD IS THE HISTORICAL EXPRESSION, NOT THE TRUE BOX CURVATURE.
# The historical entropy-viscosity code uses `(2/h)^2` for **both** directions
# with the single `h = leaf_hy` (it assumes square leaves, `GEOM_DESIGN` §1.6
# item 1) and no cross/curvature term (F4).  The box path keeps that body
# verbatim, so its `map_curvature` method returns exactly those historical
# coefficients (`m11 = m22 = (2.0/h)^2`, `h = leaf_hy`, `m12 = c1 = c2 = 0`);
# a stretched box (`Lx != 1`) is therefore *not* reproduced by the mapped
# formula, deliberately, because the nine state-dump hashes are the stronger
# contract and the historical EV is what produced every recorded run.  The
# mapped method below is the true curvature.  The two coincide on square leaves
# (the measured cross-check is in G269 (c)).

@inline _lap_index(m, qx::Int, qy::Int) = (qy - 1) * m.nq + qx

"""
    element_hessian(g, lay, lf, xi, eta) -> (x11, x12, x22, y11, y12, y22)

Second derivatives of the map with respect to the **leaf reference**
coordinates `(xi, eta)`, in the order

    x_{xi xi}, x_{xi eta}, x_{eta eta}, y_{xi xi}, y_{xi eta}, y_{eta eta}.

Affine geometries (`CartesianBox`, `AffineBox`) have a constant Jacobian, so
this is identically zero; `CurvedBox` carries the analytic values.  A curved
geometry added later (the annulus) must implement this method — it is part of
the geometry interface from cut 3 on, because the energy path cannot build a
correct physical Laplacian without it.
"""
@inline element_hessian(::CartesianBox, lay, lf::NTuple{3,Int},
                        xi::Float64, eta::Float64) =
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
@inline element_hessian(::AffineBox, lay, lf::NTuple{3,Int},
                        xi::Float64, eta::Float64) =
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
@inline function element_hessian(g::CurvedBox, lay, lf::NTuple{3,Int},
                                 xi::Float64, eta::Float64)
    # x = Lx*S (affine in S, so x'' = 0); y = T*d(S) with
    #   y_xi  = T*d'(S)*dS,  y_xi xi = T*d''(S)*dS^2,  y_xi eta = d'(S)*dS*dT.
    S, T = global_logical(lay, lf, xi, eta)
    l = lf[1]
    dS = (1.0 / leaf_nside_x(lay, l)) * 0.5
    dT = (1.0 / leaf_nside_y(lay, l)) * 0.5
    dp = _curved_dp(g, S)
    dpp = _curved_dpp(g, S)
    return (0.0, 0.0, 0.0,
            T * dpp * dS * dS, dp * dS * dT, 0.0)
end

"""
    BoxMapCurvature

The box map-curvature coefficients: the historical entropy-viscosity values
`m11 = m22 = (2.0/h)^2` with the single `h = leaf_hy`, and zero elsewhere
(`GEOM_DESIGN` §1.6 item 1/F4).  Not the true box curvature; see the section
header.
"""
struct BoxMapCurvature
    m11::Float64   # coefficient of f_{xi xi}
    m12::Float64   # coefficient of f_{xi eta} (metric tensor entry; lap uses 2*m12)
    m22::Float64   # coefficient of f_{eta eta}
    c1::Float64    # coefficient of f_xi   (map-curvature term)
    c2::Float64    # coefficient of f_eta  (map-curvature term)
end

"""
    MappedMapCurvature

The mapped map-curvature coefficients, flat leaf-major arrays over the
`nq x nq` quadrature points (built once per leaf by `map_curvature`):
`m11/m12/m22` are the entries of the metric tensor `Jinv^T Jinv` and
`c1/c2` the map-curvature coefficients `C[xi]`, `C[eta]` above.
"""
struct MappedMapCurvature
    m11::Vector{Float64}
    m12::Vector{Float64}
    m22::Vector{Float64}
    c1::Vector{Float64}
    c2::Vector{Float64}
    nq::Int
end

"""
    map_curvature(::CartesianBox, lay, lf, fd) -> BoxMapCurvature
    map_curvature(g::AbstractGeometry, lay, lf, fd) -> MappedMapCurvature

Second-derivative coefficients of the physical Laplacian at the quadrature
points of leaf `lf`.  The `CartesianBox` method returns the **historical**
entropy-viscosity coefficients (`(2.0/h)^2` in both directions with
`h = leaf_hy`, no cross or curvature term), so a box kernel routed through it
would reproduce the historical `semquad_ev_cell_stats` arithmetic bit for bit;
the production box body is not routed through it (F4), it stays verbatim.  The
mapped method is the chain rule of the section header, evaluated from
`jacobian_inverse` and `element_hessian`:
`M11 = xi_x^2 + xi_y^2`, `M12 = xi_x eta_x + xi_y eta_y`, `M22 = eta_x^2 + eta_y^2`,
`C[xi] = -(xi_x tr(M H_x) + xi_y tr(M H_y))`, `C[eta] = -(eta_x tr(M H_x) + eta_y tr(M H_y))`
with `tr(M H_p) = M11 Hp11 + 2 M12 Hp12 + M22 Hp22`.
"""
@inline function map_curvature(::CartesianBox, lay, lf::NTuple{3,Int},
                               fd::FEData)
    h = leaf_hy(lay, lf[1])          # the historical single scale, not hx
    ihe2 = (2.0 / h)^2
    return BoxMapCurvature(ihe2, 0.0, ihe2, 0.0, 0.0)
end

function map_curvature(g::AbstractGeometry, lay, lf::NTuple{3,Int}, fd::FEData)
    nq = fd.nq
    n2 = nq * nq
    m11 = Vector{Float64}(undef, n2)
    m12 = Vector{Float64}(undef, n2)
    m22 = Vector{Float64}(undef, n2)
    c1 = Vector{Float64}(undef, n2)
    c2 = Vector{Float64}(undef, n2)
    @inbounds for qy in 1:nq, qx in 1:nq
        k = (qy - 1) * nq + qx
        xi = fd.gl_x[qx]
        eta = fd.gl_x[qy]
        inv = jacobian_inverse(g, lay, lf, xi, eta)
        xi_x, eta_x, xi_y, eta_y = inv
        m11k = xi_x * xi_x + xi_y * xi_y
        m12k = xi_x * eta_x + xi_y * eta_y
        m22k = eta_x * eta_x + eta_y * eta_y
        H = element_hessian(g, lay, lf, xi, eta)
        tHx = m11k * H[1] + 2.0 * m12k * H[2] + m22k * H[3]
        tHy = m11k * H[4] + 2.0 * m12k * H[5] + m22k * H[6]
        m11[k] = m11k
        m12[k] = m12k
        m22[k] = m22k
        c1[k] = -(xi_x * tHx + xi_y * tHy)
        c2[k] = -(eta_x * tHx + eta_y * tHy)
    end
    return MappedMapCurvature(m11, m12, m22, c1, c2, nq)
end

"""
    lap_phi(mc, fd, d2tab, qx, qy, i, j) -> Float64

Physical Laplacian of the tensor-product nodal basis function `(i,j)` at the
quadrature point `(qx,qy)`.  `d2tab[q][a]` is the second derivative of the 1-D
GLL basis function `a` at the 1-D quadrature point `q` (the table
`semquad_ev_cell_stats` builds from `lagrange_d2`).  The box method is the
historical entropy-viscosity term
`d2x[i] * phi[j] * (2.0/h)^2 + phi[i] * d2y[j] * (2.0/h)^2` in its exact
association; the mapped method applies the chain rule of the section header:

    m11 * f_{xi xi} + 2*m12 * f_{xi eta} + m22 * f_{eta eta}
        + c1 * f_xi + c2 * f_eta.
"""
@inline function lap_phi(mc::BoxMapCurvature, fd::FEData,
                         d2tab::Vector{Vector{Float64}}, qx::Int, qy::Int,
                         i::Int, j::Int)
    return d2tab[qx][i] * fd.phi_v[j, qy] * mc.m11 +
           fd.phi_v[i, qx] * d2tab[qy][j] * mc.m22
end
@inline function lap_phi(mc::MappedMapCurvature, fd::FEData,
                         d2tab::Vector{Vector{Float64}}, qx::Int, qy::Int,
                         i::Int, j::Int)
    k = _lap_index(mc, qx, qy)
    dxi = fd.dphi_v[i, qx] * fd.phi_v[j, qy]
    deta = fd.phi_v[i, qx] * fd.dphi_v[j, qy]
    d2xi = d2tab[qx][i] * fd.phi_v[j, qy]
    d2eta = fd.phi_v[i, qx] * d2tab[qy][j]
    return mc.m11[k] * d2xi +
           2.0 * mc.m12[k] * (fd.dphi_v[i, qx] * fd.dphi_v[j, qy]) +
           mc.m22[k] * d2eta + mc.c1[k] * dxi + mc.c2[k] * deta
end

# ---------------------------------------------------------------------------
# 6. Face metric and the face-loop accessors
# ---------------------------------------------------------------------------

"""
    BoxFaceMetric

The box face metric: `half` is the historical face factor `0.5*hy` on an
x-face / `0.5*hx` on a y-face (GEOM_DESIGN E8), and `(nx,ny)`/`(tx,ty)` are the
outward unit normal and the positive tangent (the direction of increasing
logical face parameter, GEOM_DESIGN §2.7).
"""
struct BoxFaceMetric
    half::Float64
    nx::Float64
    ny::Float64
    tx::Float64
    ty::Float64
end

"""
    face_metric(g, lay, lf, side, fd) -> BoxFaceMetric | MappedFaceMetric

Face metric of `side in (:xminus, :xplus, :yminus, :yplus)` for one leaf.  The
box method returns the historical factors; a mapped geometry precomputes, per
face quadrature point, the face Jacobian `ds/dq`, the outward unit normal and
the positive tangent.
"""
@inline function face_metric(::CartesianBox, lay, lf::NTuple{3,Int},
                             side::Symbol, fd::FEData)
    hx = leaf_hx(lay, lf[1])
    hy = leaf_hy(lay, lf[1])
    if side === :xminus
        return BoxFaceMetric(0.5 * hy, -1.0, 0.0, 0.0, 1.0)
    elseif side === :xplus
        return BoxFaceMetric(0.5 * hy, 1.0, 0.0, 0.0, 1.0)
    elseif side === :yminus
        return BoxFaceMetric(0.5 * hx, 0.0, -1.0, 1.0, 0.0)
    elseif side === :yplus
        return BoxFaceMetric(0.5 * hx, 0.0, 1.0, 1.0, 0.0)
    end
    throw(ArgumentError("unknown face side $side"))
end

"""
    MappedFaceMetric

The mapped face metric: arrays over the `nq` face quadrature points, one entry
per point: the face Jacobian `ds/dq`, the outward unit normal `(nx,ny)`, the
positive tangent `(tx,ty)` and the physical point `(xq,yq)`.
"""
struct MappedFaceMetric
    xq::Vector{Float64}
    yq::Vector{Float64}
    jac::Vector{Float64}
    nx::Vector{Float64}
    ny::Vector{Float64}
    tx::Vector{Float64}
    ty::Vector{Float64}
    nq::Int
end

function face_metric(g::AbstractGeometry, lay, lf::NTuple{3,Int},
                     side::Symbol, fd::FEData)
    return _mapped_face_metric(g, lay, lf, side, fd.gl_x)
end

"""
    _mapped_face_metric(g, lay, lf, side, qs) -> MappedFaceMetric

The mapped face metric evaluated at the **arbitrary** face parameters `qs`
(face parameter in `[-1,1]`; the physical point is the map image).  This is the
core of [`face_metric(g, lay, lf, side, fd)`](@ref), which feeds the Gauss
quadrature points of `fd`; the diagnostics call it with the **GLL nodes**
(`sq.xi`) because the box wall bodies integrate with the GLL rule (E8), and a
mapped wall integral must use the same rule to be comparable.
"""
function _mapped_face_metric(g::AbstractGeometry, lay, lf::NTuple{3,Int},
                             side::Symbol, qs::Vector{Float64})
    nq = length(qs)
    xq = Vector{Float64}(undef, nq)
    yq = Vector{Float64}(undef, nq)
    jac = Vector{Float64}(undef, nq)
    nx = Vector{Float64}(undef, nq)
    ny = Vector{Float64}(undef, nq)
    tx = Vector{Float64}(undef, nq)
    ty = Vector{Float64}(undef, nq)
    is_x = side === :xminus || side === :xplus
    is_x || side === :yminus || side === :yplus ||
        throw(ArgumentError("unknown face side $side"))
    sgn = (side === :xplus || side === :yplus) ? 1.0 : -1.0
    @inbounds for q in 1:nq
        qr = qs[q]
        xi, eta = is_x ? (sgn, qr) : (qr, sgn)
        J, _ = element_jacobian(g, lay, lf, xi, eta)
        xq[q], yq[q] = element_map(g, lay, lf, xi, eta)
        if is_x
            # face parameter eta; tangent (x_eta, y_eta), outward +-(y_eta,-x_eta)
            a = J[2]
            b = J[4]
            j = hypot(a, b)
            jac[q] = j
            tx[q] = a / j
            ty[q] = b / j
            nx[q] = sgn * b / j
            ny[q] = -sgn * a / j
        else
            # face parameter xi; tangent (x_xi, y_xi), outward +-(-y_xi,x_xi)
            a = J[1]
            b = J[3]
            j = hypot(a, b)
            jac[q] = j
            tx[q] = a / j
            ty[q] = b / j
            nx[q] = -sgn * b / j
            ny[q] = sgn * a / j
        end
    end
    return MappedFaceMetric(xq, yq, jac, nx, ny, tx, ty, nq)
end

@inline face_jacobian(fm::BoxFaceMetric) = fm.half
@inline face_jacobian(fm::MappedFaceMetric, q::Int) = fm.jac[q]

"""
    face_weight(fm, fd, q) -> Float64

Face quadrature weight `ds/dq * gl_w[q]`.  The box method `half*gl_w[q]` is the
historical `0.5*h*gl_w[q]` (GEOM_DESIGN E8) with `half = 0.5*h` stored.
"""
@inline face_weight(fm::BoxFaceMetric, fd::FEData, q::Int) =
    fm.half * fd.gl_w[q]
@inline face_weight(fm::MappedFaceMetric, fd::FEData, q::Int) =
    fm.jac[q] * fd.gl_w[q]

@inline face_normal(fm::BoxFaceMetric, ::Int=1) = (fm.nx, fm.ny)
@inline face_normal(fm::MappedFaceMetric, q::Int) = (fm.nx[q], fm.ny[q])
@inline face_tangent(fm::BoxFaceMetric, ::Int=1) = (fm.tx, fm.ty)
@inline face_tangent(fm::MappedFaceMetric, q::Int) = (fm.tx[q], fm.ty[q])

@inline face_point(fm::MappedFaceMetric, q::Int) = (fm.xq[q], fm.yq[q])

"""
    face_point(lm, side, fd, q) -> (x, y)
    face_point(g, lay, lf, side, fd, q) -> (x, y)

Physical face quadrature point.  The box method is the historical wall-node
expression (GEOM_DESIGN E10 for the fixed coordinate, E3 for the moving one);
the mapped method returns the precomputed array entry.
"""
@inline function face_point(lm::BoxLeafMetric, side::Symbol, fd::FEData,
                            q::Int)
    if side === :xminus
        return (lm.x0, lm.y0 + 0.5 * (fd.gl_x[q] + 1.0) * lm.hy)
    elseif side === :xplus
        return (lm.x0 + lm.hx, lm.y0 + 0.5 * (fd.gl_x[q] + 1.0) * lm.hy)
    elseif side === :yminus
        return (lm.x0 + 0.5 * (fd.gl_x[q] + 1.0) * lm.hx, lm.y0)
    elseif side === :yplus
        return (lm.x0 + 0.5 * (fd.gl_x[q] + 1.0) * lm.hx, lm.y0 + lm.hy)
    end
    throw(ArgumentError("unknown face side $side"))
end
@inline function face_point(g::CartesianBox, lay, lf::NTuple{3,Int},
                            side::Symbol, fd::FEData, q::Int)
    return face_point(leaf_metric(g, lay, lf), side, fd, q)
end
@inline function face_point(g::AbstractGeometry, lay, lf::NTuple{3,Int},
                            side::Symbol, fd::FEData, q::Int)
    return face_point(face_metric(g, lay, lf, side, fd), q)
end

# ---------------------------------------------------------------------------
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
**bit for bit** (that is the expression every box diagnostics divides by; see
`docs/dev/GEOM_DESIGN.md` E13).  A mapped geometry integrates `quad_measure`
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
`min(hx, hy)`).  `docs/dev/GEOM_DESIGN.md` §6.6: on the annulus the angular cell
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