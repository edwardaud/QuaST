# Wall refinement to a minimum level.
#
# `refine_wall_to_min!` refines the leaves inside the wall strips until the strip
# reaches `min_level`, balancing after every sweep.  It is separate from the
# adaptation cycle: a card asks for it once, before the run, to place a stated
# number of cells inside the thermal boundary layers.  With `symmetry = :x` it
# refines the mirror partner with every wall leaf and uses the mirror-aware
# `balance!`, so the wall refinement cannot break the x-mirror invariant.

"""
    refine_wall_to_min!(qt, min_level; strip=0.1, symmetry=:none,
                        geom=CartesianBox(qt))

Refine leaves whose centers lie within `strip` of the bottom/top walls until
they reach at least `min_level`, then balance.  This is the numerical
counterpart of ASPECT's boundary refinement criterion / minimum refinement:
error-fraction coarsening must not remove the thermal boundary layer.

Stage 0 cut 6: the strip is a **physical distance from the Dirichlet walls**.
`geom` selects the criterion: the historical Cartesian `y` rows and centre
height on the box, the physical `height` accessor and the geometry's
`nu_wall_sides` on a mapped geometry (so the annulus's strips are its inner and
outer circles).  The strip test is shared with the balanced cycle
([`_amr_in_wall_strip`](@ref)), and its box method is the historical expression,
so the box path is bitwise unchanged.

`symmetry=:x` refines the mirror partner with every wall leaf and uses the
mirror-aware `balance!`, so the wall refinement cannot break the x-mirror
invariant (the wall criterion depends only on `iy`, so on a closed tree it is
symmetric by itself; the pairing matters for a tree that is already damaged).
It stays a **box-only** policy: a mapped geometry refuses it in the driver
because the logical x mirror is not a physical symmetry of a mapped mesh
(GEOM_DESIGN §6.5).
"""
function refine_wall_to_min!(qt::Quadtree, min_level::Int; strip::Float64=0.1,
                             symmetry::Symbol=:none,
                             geom::AbstractGeometry=CartesianBox(qt))
    min_level <= 0 && return 0
    0.0 < strip < 0.5 || throw(ArgumentError("strip must be in (0, 0.5)"))
    symmetry in (:none, :x) || throw(ArgumentError("symmetry must be :none or :x"))
    nref = 0
    for _ in 1:min_level
        changed = false
        for lf in collect(qt.leaves)
            if _amr_in_wall_strip(geom, qt, lf, strip) && lf[1] < min_level
                if symmetry === :x
                    ml = _mirror_leaf(qt, lf)
                    (ml == lf || ml in qt.leaves) || continue
                    refine!(qt, lf) || continue
                    nref += 1
                    changed = true
                    ml != lf && refine!(qt, ml) && (nref += 1)
                elseif refine!(qt, lf)
                    nref += 1
                    changed = true
                end
            end
        end
        changed || break
        balance!(qt; symmetry=symmetry)
    end
    return nref
end
