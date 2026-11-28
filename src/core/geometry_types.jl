# Geometry / mapping layer (stage 0, cut 1).
#
# The solver's physical domain is produced by one global map from the logical
# rectangle `[0,Lx] x [0,1]` (the quadtree's logical coordinates `(S,T)`), and a
# leaf-local affine map from the reference square `[-1,1]^2`.  This file owns
# both, plus everything the operators read from the geometry:
#
#   * the geometry types `CartesianBox` (production; the axis-aligned box with
#     the identity map), `AffineBox` and `CurvedBox` (validation geometries) and
#     `PolarAnnulus` (stage 0.5a, the 2-D polar annulus), all dispatching from
#     `AbstractGeometry`.  `PolarAnnulus` is defined with all of its methods in
#     section 10 at the end of this file, so the stage-0 bodies above are
#     untouched by it;
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
# character (rows E1-E16): `ihx = 2.0/hx`,
# `wfac = 0.25*hx*hy`, `x = x0 + 0.5*(xi+1)*hx`, `wt = 0.5*h*gl_w[q]`, ...
# The general Jacobian route is never reachable with a `CartesianBox`: the box
# `basis_grad` is `dphi*phi*ihx`, not `(J22*phi_xi - J21*phi_eta)/detJ`, and
# `quad_weight` returns the stored `wfac`, not a recomputed `det J`.  A general
# rewrite moves the nine state-dump hashes (GEOM_DESIGN §4.3 F1-F7), so do not
# "clean it up".
#
# The validation geometries are NOT reachable from any card in cut 1; they are
# the proof of the interface (gate G266) and the rehearsal for the annulus.

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

