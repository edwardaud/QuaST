# Rebuilding the tree a checkpoint describes.
#
# The state file stores the leaf table, not a tree object, so a resume has to
# rebuild the quadtree the checkpoint was written on.  `_checkpoint_leaf_candidates`
# validates the candidate leaf set and `_rebuild_tree` reconstructs the tree from
# it, so the resumed mesh is the mesh the stored arrays belong to.

"""
    _checkpoint_leaf_candidates(leaves)

Candidate base-cell counts implied by the leaves that touch the right/top wall.
A leaf `(l, ix, iy)` on the right wall satisfies `ix + 1 == nx0 * 2^l`, so every
level whose largest `ix` is a right-wall leaf contributes `(max(ix)+1) / 2^l`.
Candidates are validated by rebuilding the tree and comparing the leaf set, so
a level with an interior extreme only produces a harmless candidate.
"""
function _checkpoint_leaf_candidates(leaves::Vector{NTuple{3,Int}})
    maxx = Dict{Int,Int}()
    maxy = Dict{Int,Int}()
    for (l, ix, iy) in leaves
        maxx[l] = max(get(maxx, l, -1), ix)
        maxy[l] = max(get(maxy, l, -1), iy)
    end
    cx = Set{Int}([1])
    cy = Set{Int}([1])
    for (l, mx) in maxx
        q = 1 << l
        (mx + 1) % q == 0 && push!(cx, (mx + 1) ÷ q)
    end
    for (l, my) in maxy
        q = 1 << l
        (my + 1) % q == 0 && push!(cy, (my + 1) ÷ q)
    end
    return sort(collect(cx)), sort(collect(cy))
end

"""
    _rebuild_tree(leaves, maxlev; nx0, ny0, Lx)

Rebuild a tree with the given base cells and return it only when its leaf set is
exactly `leaves`; otherwise return `nothing`.
"""
function _rebuild_tree(leaves::Vector{NTuple{3,Int}}, maxlev::Int;
                       nx0::Int, ny0::Int, Lx::Float64)
    qt = Quadtree(maxlev; Lx=Lx, cells_x=nx0, cells_y=ny0)
    for (l, ix, iy) in leaves
        node = (0, ix >> l, iy >> l)
        for lev in 1:l
            node in qt.leaves && refine!(qt, node)
            node = (lev, ix >> (l - lev), iy >> (l - lev))
        end
    end
    length(qt.leaves) == length(leaves) || return nothing
    Set(qt.leaves) == Set(leaves) || return nothing
    return qt
end
