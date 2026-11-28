# Leaf-level AMR operations shared by the adaptive cycle and the driver.
#
# This file contains the pair-strict refine/coarsen primitives.  The hard leaf
# budget lives in `amr/budget.jl`; the physical wall-strip predicates live in
# `core/wall_strip.jl`.  Nothing here knows about a policy -- the balanced
# cycle and `semquad_amr_passes!` are the only callers.
"""
    _refine_pair!(qt, lf, max_level, symmetry)

Refine `lf`, refusing above `max_level` and -- with `symmetry=:x` -- refusing a
one-sided split: the mirror partner must be a leaf below `max_level` too, and
both sides are split or neither.  In a mirror-closed tree the partner is always
a leaf, so the pair rule only rejects legacy asymmetric input (which
`mirror_close!` then repairs).  Returns the number of leaves added.
"""
function _refine_pair!(qt::Quadtree, lf::NTuple{3,Int}, max_level::Int,
                       symmetry::Symbol)
    lf[1] < max_level || return 0
    if symmetry === :x
        # pair-strict: never split one side of a mirror pair alone.  In a closed
        # tree the partner is always a leaf, so this only rejects legacy
        # asymmetric input (which `mirror_close!` then repairs).
        ml = _mirror_leaf(qt, lf)
        (ml == lf || (ml[1] < max_level && ml in qt.leaves)) || return 0
        refine!(qt, lf) || return 0
        n = 1
        if ml != lf && refine!(qt, ml)
            n += 1
        end
        return n
    end
    refine!(qt, lf) || return 0
    return 1
end

"""
    _coarsen_pair!(qt, lf, symmetry, protect)

Merge the sibling quartet of `lf` (and, with `symmetry=:x`, its mirror quartet).
Returns the number of quartets merged, 0 when the quartet is incomplete or any of
its four leaves is in `protect`.  With `:x` the merge is atomic over the mirror
pair: the mirror quartet has to be complete and unprotected *before* either side
is merged, so an initially mirror-symmetric mesh stays closed.
"""
function _coarsen_pair!(qt::Quadtree, lf::NTuple{3,Int}, symmetry::Symbol,
                        protect::Set{NTuple{3,Int}})
    # coarsen! merges the whole sibling quad; protect if any sibling is
    # protected, otherwise the protected leaf would disappear indirectly.
    bx = lf[2] - lf[2] % 2
    by = lf[3] - lf[3] % 2
    for dy in 0:1, dx in 0:1
        (lf[1], bx + dx, by + dy) in protect && return 0
    end
    ml = lf
    if symmetry === :x
        # pair-strict: the mirror quartet has to be complete and unprotected
        # *before* either side is merged, otherwise one side would be merged
        # alone and the tree loses the invariant (5 non-mirrored leaves per
        # one-sided merge).
        ml = _mirror_leaf(qt, lf)
        if ml != lf
            mbx = ml[2] - ml[2] % 2
            mby = ml[3] - ml[3] % 2
            for dy in 0:1, dx in 0:1
                k = (ml[1], mbx + dx, mby + dy)
                (k in qt.leaves && !(k in protect)) || return 0
            end
        end
    end
    coarsen!(qt, lf) || return 0
    n = 1
    if symmetry === :x && ml != lf && coarsen!(qt, ml)
        n += 1
    end
    return n
end
