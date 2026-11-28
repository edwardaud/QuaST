# Physical wall-strip predicates for the AMR wall policy.
#
# Extracted from `amr/leaf_ops.jl` and placed in the geometry layer: these are
# physical-distance helpers (`height`, `element_map`, `nu_wall_sides`) used by
# the balanced cycle, the AMR pass and `refine_wall_to_min!`.  The box methods
# keep the historical Cartesian `y` arithmetic bit for bit.

# ---------------------------------------------------------------------------
# Wall strips on a mapped geometry (stage 0 cut 6, GEOM_DESIGN section 3 cut 6
# and section 6.5)
# ---------------------------------------------------------------------------
#
# The balanced cycle's wall strip, the AMR pass's wall protect set and
# `refine_wall_to_min!` are *physical-distance* criteria ("within `strip` of the
# Dirichlet walls").  On the box they are written in the Cartesian `y`; a mapped
# geometry expresses the same criterion in the physical `height` accessor, whose
# box value is `y` exactly (GEOM_DESIGN E17), and names the Dirichlet wall sides
# through `nu_wall_sides(g)` (the annulus registers its circles there instead of
# the logical `y` rows).  Both methods keep the historical arithmetic on the box
# so the box path is bitwise unchanged.

"""
    _amr_wall_leaf(g, lay, lf) -> Bool

`true` when the leaf touches one of the geometry's physical Dirichlet walls
(`nu_wall_sides(g)`) - the first/last wall rows the balanced cycle protects from
coarsening and `refine_wall_to_min!` enforces.  Box method: the historical
`lf[3] == 0 || lf[3] == ny0*(1<<l)-1`; a mapped geometry: the same logical-side
test on the sides the geometry names as its Nusselt walls.
"""
@inline function _amr_wall_leaf(::CartesianBox, lay, lf::NTuple{3,Int})
    return lf[3] == 0 || lf[3] == lay.ny0 * (1 << lf[1]) - 1
end
@inline function _amr_wall_leaf(g::AbstractGeometry, lay, lf::NTuple{3,Int})
    for side in nu_wall_sides(g)
        _on_logical_wall(lf, side, lay) && return true
    end
    return false
end

"""
    _amr_center_height(g, lay, lf) -> Float64

Physical height of the leaf centre: the historical `zc = (lf[3] + 0.5) *
leaf_hy` on the box; `height(g, element_map(g, lay, lf, 0, 0))` for a mapped
geometry (the map image of the leaf-centre logical point, which is the physical
height the strip criterion is expressed in).
"""
@inline function _amr_center_height(::CartesianBox, lay, lf::NTuple{3,Int})
    return (lf[3] + 0.5) * leaf_hy(lay, lf[1])
end
@inline function _amr_center_height(g::AbstractGeometry, lay, lf::NTuple{3,Int})
    x, y = element_map(g, lay, lf, 0.0, 0.0)
    return height(g, x, y)
end

"""
    _amr_in_wall_strip(g, lay, lf, strip) -> Bool

Whether `lf` is inside one of the two wall strips `[0, strip]` / `[1-strip, 1]`
of the physical height, or is a wall leaf.  Box method: the historical `zc`
test, bit for bit; mapped geometry: the physical height of the leaf centre and
the wall-row test of [`_amr_wall_leaf`](@ref).
"""
@inline function _amr_in_wall_strip(::CartesianBox, lay, lf::NTuple{3,Int},
                                    strip::Float64)
    lf[3] == 0 && return true
    lf[3] == lay.ny0 * (1 << lf[1]) - 1 && return true
    zc = (lf[3] + 0.5) * leaf_hy(lay, lf[1])
    return zc <= strip || zc >= 1.0 - strip
end
@inline function _amr_in_wall_strip(g::AbstractGeometry, lay, lf::NTuple{3,Int},
                                    strip::Float64)
    _amr_wall_leaf(g, lay, lf) && return true
    hc = _amr_center_height(g, lay, lf)
    return hc <= strip || hc >= 1.0 - strip
end
