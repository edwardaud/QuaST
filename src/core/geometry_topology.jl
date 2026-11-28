# Geometry topology: logical boundary registry, perimeters, face-pair
# dispatch and seam compatibility.  Included after the measures file; the
# PolarAnnulus methods are in the next file.
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

