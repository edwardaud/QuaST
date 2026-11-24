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
    nq = fd.nq
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
        qr = fd.gl_x[q]
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
    measure_weight(g, x, y) -> Float64

Optional positive physical weight multiplying `det J` and every face Jacobian
(GEOM_DESIGN §2.3): `1.0` on every plane geometry, so the kernels ignore it in
cut 1.  An axisymmetric shell reduction would return `2*pi*r` (or `r^2*sin
theta`) without a kernel change.  Not read by the box methods, which are the
historical expressions.
"""
@inline measure_weight(::AbstractGeometry, x::Float64, y::Float64) = 1.0

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

Analytic perimeter `2*(Lx + 1)` of the box (GEOM_DESIGN E12).  Mapped
geometries add their own once a consumer needs it (cut 4); the box path must
return the historical expression.
"""
@inline perimeter(g::CartesianBox) = 2.0 * (g.Lx + 1.0)


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