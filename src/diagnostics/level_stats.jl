# Per-level AMR statistics (box and mapped tree methods).
"""
    semquad_level_stats(qt; strip=0.1)

Leaf-level histogram plus the finest level in the wall strips (`y <= strip` or
`y >= 1-strip`) and in the interior.  This is the cheap "is the AMR actually
tracking the boundary layer?" diagnostic: for a wall-refined tree
`wall_max > interior_max`, for a uniform tree they are equal.

This signature is the **box-tree** wall claim: the strips are the logical `y`
rows, which the quadtree does not know are physical walls.  A periodic-seam
tree (an annulus ring) is refused (stage-0.5 audit defect B3) — on one it
reports `wall_max = 1` where the mapped method reports `wall_max = 0` — and a
mapped geometry must use the geometry-aware
[`semquad_level_stats`](@ref)`(g, qt; strip)` method.  A bounded mapped tree (a
wedge) cannot be detected (the tree stores no geometry); the box signature is
documented as the logical-`y` claim for that case.
"""
function semquad_level_stats(qt::Quadtree; strip::Float64=0.1)
    qt.seam === :none ||
        throw(ArgumentError("semquad_level_stats(qt) is the box-tree wall " *
                            "claim (the strips are the logical y rows, which " *
                            "are not physical walls on a mapped topology) and " *
                            "a periodic-seam tree is not a box tree.  Use the " *
                            "geometry-aware method: " *
                            "semquad_level_stats(g, qt; strip=$strip) " *
                            "(stage-0.5 audit defect B3)."))
    hist = Dict{Int,Int}()
    wall_max = 0
    interior_max = 0
    for lf in qt.leaves
        l, ix, iy = lf
        h = leaf_hy(qt, l)
        y0 = iy * h
        hist[l] = get(hist, l, 0) + 1
        if y0 + h <= strip || y0 >= 1.0 - strip
            wall_max = max(wall_max, l)
        else
            interior_max = max(interior_max, l)
        end
    end
    return (histogram=hist, wall_max=wall_max, interior_max=interior_max,
            n=length(qt.leaves))
end

"""
    semquad_level_stats(g::AbstractGeometry, qt; strip=0.1)

Mapped method of [`semquad_level_stats`](@ref): a leaf is in a wall strip when
its **physical height** extent lies inside `h <= strip` (bottom) or
`h >= 1-strip` (top), tested on the four mapped leaf corners.  The box method
above keeps its historical physical-`y` leaf test (E16); this is the method a
mapped caller must use (the box signature refuses a periodic-seam tree with an
actionable message, stage-0.5 audit defect B3).
"""
function semquad_level_stats(g::AbstractGeometry, qt::Quadtree;
                             strip::Float64=0.1)
    hist = Dict{Int,Int}()
    wall_max = 0
    interior_max = 0
    for lf in qt.leaves
        l = lf[1]
        hist[l] = get(hist, l, 0) + 1
        hlo = Inf
        hhi = -Inf
        for (xi, eta) in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
            x, y = element_map(g, qt, lf, xi, eta)
            h = height(g, x, y)
            hlo = min(hlo, h)
            hhi = max(hhi, h)
        end
        if hhi <= strip || hlo >= 1.0 - strip
            wall_max = max(wall_max, l)
        else
            interior_max = max(interior_max, l)
        end
    end
    return (histogram=hist, wall_max=wall_max, interior_max=interior_max,
            n=length(qt.leaves))
end

