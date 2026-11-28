# Geometry: the 2-D polar annulus (PolarAnnulus) and its geometry methods.
# This is the tail of the former geometry monolith; the @max_methods block at
# the end is intentionally last in include order.
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
