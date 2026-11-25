# The flat face table of the quadtree.
#
# The topology queries -- neighbours at a direction, the 2:1 partner, the face
# slots of a leaf -- are answered from a flat array indexed by leaf id instead of a
# hash lookup, because the adaptive cycle asks them for every leaf on every pass.
# This file holds the table itself: the incremental maintenance `refine!` and
# `coarsen!` call, the O(n) rebuild, and the query accessors.  The tree type and
# its refinement primitives are in `quadtree.jl`.

# ---- the flat table: queries -------------------------------------------------

"""
    _push_touch_node!(out, tab, l, jx, jy, d, lmax)

Append the ids of the leaves of the `(l, jx, jy)` subtree that touch face
direction `d` of the node's box, in increasing coordinate along the face.

The caller guarantees that `(l, jx, jy)` is an **existing node**: its parent is
internal and the tree is complete, so every child is a leaf or an internal node.
The queried face is the node's near edge, which fixes which child can touch it:
`:xplus`/`:yplus` take the low child, `:xminus`/`:yminus` the high one; the
along-face children are visited in increasing `y` (vertical face) or `x`
(horizontal face), the order of the historical `_collect_face!` descent.
"""
function _push_touch_node!(out::Vector{Int32}, tab::QuadFaceTable,
                           l::Int, jx::Int, jy::Int, d::Int, lmax::Int)
    id = get(tab.ids, (l, jx, jy), Int32(0))
    if id != Int32(0)
        push!(out, id)
        return nothing
    end
    l < lmax || return nothing      # unreachable for a complete tree
    if d == 1 || d == 2             # vertical face: children ordered by y
        cx = d == 1 ? 2 * jx : 2 * jx + 1
        _push_touch_node!(out, tab, l + 1, cx, 2 * jy, d, lmax)
        _push_touch_node!(out, tab, l + 1, cx, 2 * jy + 1, d, lmax)
    else                            # horizontal face: children ordered by x
        cy = d == 3 ? 2 * jy : 2 * jy + 1
        _push_touch_node!(out, tab, l + 1, 2 * jx, cy, d, lmax)
        _push_touch_node!(out, tab, l + 1, 2 * jx + 1, cy, d, lmax)
    end
    return nothing
end

"""
    _push_face_ids!(out, tab, qt, leaf, d)

Append the ids of the face neighbours of `leaf` in direction `d` (1-based,
`FACE_DIRS` order) to `out` -- the same list, in the same order, as
`_neighbors_scan(qt, leaf, dir)`, but allocation-free.

`leaf` must be a leaf of the tree (`tab.ids` contains it).  The box one cell
away in direction `d` is covered by

  * a **coarser** leaf (found first by the ancestor walk below), and then it
    covers the whole face: one neighbour;
  * the **equal-level** cell, if that is a leaf: one neighbour;
  * the equal-level cell's **subtree**, if that is an internal node.

In a complete quadtree a cell exists iff none of its ancestors is a leaf, so "no
ancestor is a leaf" is exactly "the equal-level cell is an internal node".  The
leaf's own children are never in the list: they do not exist (a leaf has none),
and the equal-level cell is on the other side of the face.

**Stage 0.5c1, the periodic seam.**  On a `seam = :periodic` tree the `y`
direction is closed: the `:yplus` face of the last column and the `:yminus` face
of the first column are the same physical face.  The adjacent column across that
face is the *first* column (`:yplus` of the last column) or the *last* column
(`:yminus` of the first column) — never a column outside the layout — so the
wrap is a substitution of `jy` with an in-range index, not a modulo in the
hot path:

  * `:yplus` wraps `jy = nside -> 0`: the equal-level cell is the first column,
    whose ancestor at level `k` is the first column (`0 >> m == 0`);
  * `:yminus` wraps `jy = -1 -> nside - 1`: the equal-level cell is the last
    column, whose ancestor at level `k` is the last column
    (`(nside-1) >> m == nside(k) - 1`, exact because `nside` is a power of two
    multiple of the coarse count).

The subtree descent then reuses the bounded arithmetic (the wrapped column's
in-range index is already the near column of the cut), and a cross-seam pair of
*different* levels is found by the same ancestor walk.  The `x` direction is
never periodic.  A non-periodic tree takes the historical body bit for bit.
"""
function _push_face_ids!(out::Vector{Int32}, tab::QuadFaceTable, qt::Quadtree,
                         leaf::NTuple{3,Int}, d::Int)
    l, ix, iy = leaf
    local jx::Int, jy::Int
    if d == 1                       # :xplus
        jx = ix + 1; jy = iy
        jx < qt.nx0 * (1 << l) || return out
    elseif d == 2                   # :xminus
        jx = ix - 1; jy = iy
        ix > 0 || return out
    elseif d == 3                   # :yplus
        jx = ix; jy = iy + 1
        if jy >= qt.ny0 * (1 << l)
            qt.seam === :periodic || return out
            jy = 0                  # cross-seam: the first column of the sector
        end
    else                            # :yminus
        jx = ix; jy = iy - 1
        if iy == 0
            qt.seam === :periodic || return out
            jy = qt.ny0 * (1 << l) - 1     # cross-seam: the last column
        end
    end
    for k in 0:l
        id = get(tab.ids, (k, jx >> (l - k), jy >> (l - k)), Int32(0))
        if id != Int32(0)           # a coarser or equal-level leaf: whole face
            push!(out, id)
            return out
        end
    end
    # No ancestor of the equal-level cell is a leaf: it is an internal node.
    if d == 1 || d == 2
        cx = d == 1 ? 2 * jx : 2 * jx + 1
        _push_touch_node!(out, tab, l + 1, cx, 2 * jy, d, qt.maxlevel)
        _push_touch_node!(out, tab, l + 1, cx, 2 * jy + 1, d, qt.maxlevel)
    else
        cy = d == 3 ? 2 * jy : 2 * jy + 1
        _push_touch_node!(out, tab, l + 1, 2 * jx, cy, d, qt.maxlevel)
        _push_touch_node!(out, tab, l + 1, 2 * jx + 1, cy, d, qt.maxlevel)
    end
    return out
end

# ---- the flat table: storage -------------------------------------------------

# Write `len` ids from `scratch[from:from+len-1]` into `slot`, in place when its
# pool block is large enough, otherwise into a fresh block at the high-water
# mark (the old block becomes a hole; `_table_leaf_id!` rebuilds the table when
# the holes exceed half of the pool).
function _slot_put!(tab::QuadFaceTable, slot::Int, from::Int, len::Int)
    o = Int(tab.off[slot])
    if o != 0 && len <= Int(tab.scap[slot])
        @inbounds for k in 0:(len - 1)
            tab.pool[o + k] = tab.scratch[from + k]
        end
        tab.slen[slot] = Int32(len)
        return nothing
    end
    o != 0 && (tab.holes += tab.scap[slot])
    cap = max(len, 4)
    need = Int(tab.used) + cap - 1
    if need > length(tab.pool)
        resize!(tab.pool, max(need, 2 * length(tab.pool) + 64))
    end
    o = Int(tab.used)
    @inbounds for k in 0:(len - 1)
        tab.pool[o + k] = tab.scratch[from + k]
    end
    tab.off[slot] = Int32(o)
    tab.scap[slot] = Int32(cap)
    tab.slen[slot] = Int32(len)
    tab.used = Int32(o + cap)
    return nothing
end

"""
    _table_recompute!(tab, qt, id)

Recompute the four face lists of leaf `id` from the current leaf set and mark
them current.  No allocation once the scratch buffer and the pool are warm: the
lists go through `tab.scratch`, and every pointer is an `Int32` id.
"""
function _table_recompute!(tab::QuadFaceTable, qt::Quadtree, id::Int32)
    tab.gen[id] == tab.cur && return nothing
    lf = tab.keys[id]
    scratch = tab.scratch
    empty!(scratch)
    m1 = length(scratch) + 1
    _push_face_ids!(scratch, tab, qt, lf, 1)
    m2 = length(scratch) + 1
    _push_face_ids!(scratch, tab, qt, lf, 2)
    m3 = length(scratch) + 1
    _push_face_ids!(scratch, tab, qt, lf, 3)
    m4 = length(scratch) + 1
    _push_face_ids!(scratch, tab, qt, lf, 4)
    base = 4 * (Int(id) - 1)
    _slot_put!(tab, base + 1, m1, m2 - m1)
    _slot_put!(tab, base + 2, m2, m3 - m2)
    _slot_put!(tab, base + 3, m3, m4 - m3)
    _slot_put!(tab, base + 4, m4, length(scratch) + 1 - m4)
    tab.gen[id] = tab.cur
    return nothing
end

"""Rebuild the whole flat table from `qt.leaves`: one O(n) pass, used on the
first query of a tree (and after `_invalidate_faces!` or a pool compaction).
Ids are assigned in leaf-set iteration order; the neighbour lists themselves are
geometric, so the result does not depend on that order."""
function _table_build!(tab::QuadFaceTable, qt::Quadtree)
    empty!(tab.ids); empty!(tab.keys); empty!(tab.lev); empty!(tab.gen)
    empty!(tab.off); empty!(tab.slen); empty!(tab.scap); empty!(tab.freeids)
    empty!(tab.live)
    tab.used = Int32(1)
    tab.holes = Int32(0)
    tab.cur = tab.cur == typemax(Int32) ? Int32(1) : tab.cur + Int32(1)
    n = length(qt.leaves)
    sizehint!(tab.ids, n)
    sizehint!(tab.keys, n); sizehint!(tab.lev, n); sizehint!(tab.gen, n)
    sizehint!(tab.off, 4n); sizehint!(tab.slen, 4n); sizehint!(tab.scap, 4n)
    for lf in qt.leaves
        id = Int32(length(tab.keys) + 1)
        push!(tab.keys, lf)
        push!(tab.lev, Int8(lf[1]))
        push!(tab.gen, Int32(0))
        push!(tab.live, true)
        tab.ids[lf] = id
        for _ in 1:4
            push!(tab.off, Int32(0))
            push!(tab.slen, Int32(0))
            push!(tab.scap, Int32(0))
        end
    end
    for id in 1:n
        _table_recompute!(tab, qt, Int32(id))
    end
    tab.rebuilds += 1
    return tab
end

# Parent node of a leaf's quartet (`amr_balanced.jl`'s `_pq`, defined here
# because the local rebalance below needs it; one definition for the module).
@inline _pq(lf::NTuple{3,Int}) = (lf[1] - 1, lf[2] >> 1, lf[3] >> 1)

@inline function _table_ensure_built!(qt::Quadtree)
    tab = qt.faces.tab
    if tab.cur == Int32(0) || Int(tab.holes) > (Int(tab.used) >> 1)
        _table_build!(tab, qt)
    end
    return tab
end

@inline function _table_leaf_id!(qt::Quadtree, leaf::NTuple{3,Int})
    tab = _table_ensure_built!(qt)
    return get(tab.ids, leaf, Int32(0))
end

@inline function _table_ready!(qt::Quadtree, id::Int32)
    tab = qt.faces.tab
    tab.gen[id] == tab.cur || _table_recompute!(tab, qt, id)
    return tab
end

# `(offset, count)` of the neighbour ids of leaf `id` in direction `d`.
@inline function _face_slot(tab::QuadFaceTable, id::Int32, d::Int)
    slot = 4 * (Int(id) - 1) + d
    return Int(tab.off[slot]), Int(tab.slen[slot])
end

"True when a face neighbour of leaf `id` is two or more levels finer (the 2:1
violation test of `balance!`), read straight from the table."
@inline function _leaf21(tab::QuadFaceTable, id::Int32)
    l = Int(tab.lev[id]) + 2
    for d in 1:4
        o, n = _face_slot(tab, id, d)
        @inbounds for k in 0:(n - 1)
            Int(tab.lev[tab.pool[o + k]]) >= l && return true
        end
    end
    return false
end

"Fill `nbs` with the keys of the face neighbours of leaf `id` (all four faces,
`FACE_DIRS` order) and return it.  The caller owns `nbs` and reuses it."
@inline function _leaf_nbs!(nbs::Vector{NTuple{3,Int}}, tab::QuadFaceTable,
                            id::Int32)
    empty!(nbs)
    for d in 1:4
        o, n = _face_slot(tab, id, d)
        @inbounds for k in 0:(n - 1)
            push!(nbs, tab.keys[tab.pool[o + k]])
        end
    end
    return nbs
end

# ---- the flat table: incremental maintenance ---------------------------------

"""
    _faces_drop_leaf!(qt, leaf)

Remove `leaf` from the flat table, marking every leaf whose face list contains
it stale (a split replaces one face by two sub-faces; a merge does the reverse),
so the next query recomputes exactly those.  Must be called **before** the leaf
leaves `qt.leaves`: the recomputation of its own lists needs the current tree.
"""
function _faces_drop_leaf!(qt::Quadtree, leaf::NTuple{3,Int})
    tab = qt.faces.tab
    tab.cur == Int32(0) && return nothing
    id = get(tab.ids, leaf, Int32(0))
    if id == Int32(0)               # unknown leaf: rebuild rather than serve stale data
        tab.cur = Int32(0)
        return nothing
    end
    tab.gen[id] == tab.cur || _table_recompute!(tab, qt, id)
    for d in 1:4
        o, n = _face_slot(tab, id, d)
        @inbounds for k in 0:(n - 1)
            tab.gen[tab.pool[o + k]] = Int32(0)
        end
    end
    tab.gen[id] = Int32(0)
    tab.live[id] = false
    delete!(tab.ids, leaf)
    push!(tab.freeids, id)
    return nothing
end

"""
    _faces_add_leaf!(qt, leaf)

Register a new leaf in the flat table with an empty (invalid) entry; its four
lists are computed on the first query.  Ids freed by `_faces_drop_leaf!` are
reused, keeping the pool blocks they already own.
"""
function _faces_add_leaf!(qt::Quadtree, leaf::NTuple{3,Int})
    tab = qt.faces.tab
    tab.cur == Int32(0) && return nothing
    if haskey(tab.ids, leaf)        # already known: rebuild rather than corrupt
        tab.cur = Int32(0)
        return nothing
    end
    id = isempty(tab.freeids) ? Int32(length(tab.keys) + 1) : pop!(tab.freeids)
    if Int(id) == length(tab.keys) + 1
        push!(tab.keys, leaf)
        push!(tab.lev, Int8(leaf[1]))
        push!(tab.gen, Int32(0))
        push!(tab.live, true)
        for _ in 1:4
            push!(tab.off, Int32(0))
            push!(tab.slen, Int32(0))
            push!(tab.scap, Int32(0))
        end
    else
        tab.keys[id] = leaf
        tab.lev[id] = Int8(leaf[1])
        tab.gen[id] = Int32(0)
        tab.live[id] = true
        base = 4 * (Int(id) - 1)
        for d in 1:4
            tab.slen[base + d] = Int32(0)   # keep off/scap: the block is reused
        end
    end
    tab.ids[leaf] = id
    return nothing
end
