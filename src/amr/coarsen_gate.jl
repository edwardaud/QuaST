# The 2:1-aware coarsening gate (deal.II `coarsening_allowed` port).
"""
    _coarsening_allowed(qt, kids, will_refine, granted) -> Bool

deal.II's `coarsening_allowed()` (`source/grid/tria.cc:10935-10996`, v9.4.0;
`docs/dev/AMR.md` §1.3) for one quartet: iterate the
quartet's **external** faces and deny the merge when a face neighbour would end
the pass more than one level finer than the merged parent (level
`kids[1][1] - 1`).

`will_refine` holds the leaves the refinement plan splits (`+1`: after the step
that neighbour is two levels finer than the parent, so the merge would be undone
by `balance!`), `granted` holds the parents whose quartet this plan already
accepted (`-1`: those leaves are gone before the refinement pass runs).  The
query is per boundary child, not `neighbors(qt, parent, dir)` -- the latter also
returns the parent's own children and a graded face returns up to
`2^(maxlevel-p)` leaves.

Returns `false` when a neighbour is two levels finer, so the caller may merge
only quartets whose parent is already 2:1-legal *after* the whole pass.

The neighbour lists come from the flat [`QuadFaceTable`](@ref) (`_face_slot`),
so the eight queries per quartet are array reads: this predicate runs once per
candidate quartet in every plan sweep and once per live-funded merge
(`docs/dev/AMR.md` §9).
"""
function _coarsening_allowed(qt::Quadtree, kids::NTuple{4,NTuple{3,Int}},
                             will_refine::Set{NTuple{3,Int}},
                             granted::Set{NTuple{3,Int}})
    k00, k01, k10, k11 = kids
    l = k00[1]                     # child level; the merged parent is l - 1
    p = _pq(k00)                   # `_pq(nb) == p`  <=>  `nb` is in the quartet
    # The four external faces, two boundary children each, in `_face_kids` order:
    # :xminus = kids[:, 1] = (k00, k01), :xplus = (k10, k11),
    # :yminus = kids[1, :] = (k00, k10), :yplus = (k01, k11).
    for (d, c1, c2) in ((2, k00, k01), (1, k10, k11),
                        (4, k00, k10), (3, k01, k11))
        for c in (c1, c2)
            id = _table_leaf_id!(qt, c)
            id == Int32(0) && continue
            tab = _table_ready!(qt, id)
            o, n = _face_slot(tab, id, d)
            @inbounds for k in 0:(n - 1)
                nid = tab.pool[o + k]
                nb = tab.keys[nid]
                pn = _pq(nb)
                pn == p && continue            # one of the quartet's own children
                eff = (nb in will_refine ? 1 : 0) - (pn in granted ? 1 : 0)
                Int(tab.lev[nid]) + eff <= l || return false
            end
        end
    end
    return true
end

# Matrix form (the layout `_quartet_kids` returns), kept for the G211 gate and
# for callers that build the quartet as a 2x2 array; delegates to the tuple
# method so there is exactly one implementation of the predicate.
function _coarsening_allowed(qt::Quadtree, kids::Matrix{NTuple{3,Int}},
                             will_refine::Set{NTuple{3,Int}},
                             granted::Set{NTuple{3,Int}})
    return _coarsening_allowed(qt,
                               (kids[1, 1], kids[2, 1], kids[1, 2], kids[2, 2]),
                               will_refine, granted)
end

"""
    _merge_allowed(qt, lf, will_refine, granted, symmetry) -> Bool

The gate for one quartet of the plan: the quartet and (under `symmetry=:x`) its
mirror quartet must both be complete and `_coarsening_allowed`.  Atomic over the
mirror pair, because `_coarsen_pair!` merges both or neither.
"""
function _merge_allowed(qt::Quadtree, lf::NTuple{3,Int},
                        will_refine::Set{NTuple{3,Int}},
                        granted::Set{NTuple{3,Int}}, symmetry::Symbol)
    kids = _kids4(lf)
    all(k -> k in qt.leaves, kids) || return false
    _coarsening_allowed(qt, kids, will_refine, granted) || return false
    if symmetry === :x
        ml = _mirror_leaf(qt, lf)
        if ml != lf
            mk = _kids4(ml)
            all(k -> k in qt.leaves, mk) || return false
            _coarsening_allowed(qt, mk, will_refine, granted) || return false
        end
    end
    return true
end

