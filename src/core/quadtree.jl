# 2:1-balanced quadtree for the DG-SEM AMR prototype (stage B).
#
# Leaves are `(level, ix, iy)` with `ix,iy in 0:2^level-1`.  The tree is
# refined/coarsened by splitting/merging four siblings, and `balance!` enforces
# the 2:1 condition (face neighbors differ by at most one level).  DG does not
# need hanging-node constraints, so the tree only has to provide the leaf list
# and the face-neighbor lists (one or two leaves per face).
#
# This module is independent of the DG operators; the next step is to assemble
# the verified dg2d face terms on these leaves (intersections of coarse and
# fine faces are integrated with the fine-side quadrature).

const FACE_DIRS = (:xplus, :xminus, :yplus, :yminus)

# Direction symbol -> `1:4` (`FACE_DIRS` order), the slot order of
# [`QuadFaceTable`](@ref).  Hot loops index the integer directly.
@inline function _diridx(dir::Symbol)
    dir === :xplus && return 1
    dir === :xminus && return 2
    dir === :yplus && return 3
    dir === :yminus && return 4
    throw(ArgumentError("unknown direction $dir"))
end

# ---------------------------------------------------------------------------
# Flat face-neighbour table (docs/dev/AMR.md §9, stage A)
# ---------------------------------------------------------------------------
#
# deal.II keeps `TriaLevel::neighbors`, a flat table rebuilt once by
# `update_neighbors()` after the refinement pass (`tria_levels.h:116-148`,
# `tria.cc:2118-2289`); p4est keeps the flat `quad_to_quad`/`quad_to_face`
# arrays and answers `p4est_mesh_face_neighbor_next()` with array reads
# (`p4est_mesh.h:51-72`).  Both are O(1) per query and allocation-free.  Our
# `neighbors` used to allocate a fresh `Vector` and refill it through a
# recursive descent from the base cells on *every* query: 223.9 MB
# (`_collect_face!`) + 74.6 MB (`neighbors`) per 4 AMR cycles, 42 % of the whole
# pass (`docs/dev/AMR.md` §4).
#
# The table here is maintained **incrementally** instead of being rebuilt per
# tree version: `balance!` mutates the tree between the queries of one wave, and
# a per-version rebuild after every split is exactly the pattern that made the
# old cache re-query the whole tree (a split dropped all cached lists).  A split
# only changes the neighbour lists of the split node and of the leaves facing
# it, so `refine!`/`coarsen!` mark those leaves stale (one `Int32` write per
# leaf) and a query recomputes O(1) leaves.
#
# Layout:
#   * `ids` maps a leaf key to a dense id 1..n (one `Dict` lookup per query; the
#     id map is our design, the flat arrays are deal.II's/p4est's);
#   * the four face lists of leaf `id` are `(off, len)` slices of one `Int32`
#     pool of ids, slot = `4*(id-1)+d`, `d` in `1:4` = `FACE_DIRS` order;
#   * `lev[id]` is the leaf level, so the 2:1 test is an `Int8` compare and the
#     hot loops never build a key tuple;
#   * `gen[id] == cur` means the four slices are valid.
#
# A face list is a pure function of the leaf set, computed from the *topology*
# (index arithmetic) rather than from float coordinates, and it is exactly the
# list `_neighbors_scan` (the historical `_collect_face!` descent, kept as the
# reference oracle) returns, in the same order -- increasing coordinate along
# the face.  The adjacent equal-level cell is either a leaf (one neighbour), or
# covered by a coarser leaf (one neighbour covering the whole face), or it is an
# internal node, in which case the neighbours are the leaves of its subtree that
# touch the face.  In a complete quadtree the equal-level cell exists iff none
# of its ancestors is a leaf, so the coarse case is the ancestor walk in
# `_push_face_ids!` and the fine case is the `_push_touch_node!` descent.
mutable struct QuadFaceTable
    ids::Dict{NTuple{3,Int},Int32}   # live leaf -> id
    keys::Vector{NTuple{3,Int}}      # id -> leaf
    lev::Vector{Int8}                # id -> level
    gen::Vector{Int32}               # id -> validity token (== cur: slices current)
    cur::Int32                       # current token; 0 = table invalid
    off::Vector{Int32}               # slot -> pool offset (0 = empty)
    slen::Vector{Int32}              # slot -> number of neighbours
    scap::Vector{Int32}              # slot -> capacity of its pool block
    pool::Vector{Int32}              # neighbour ids
    used::Int32                      # pool high-water mark (next free entry)
    freeids::Vector{Int32}           # freed ids, reused by `_faces_add_leaf!`
    live::BitVector                  # id is the id of a live leaf
    holes::Int32                     # pool entries abandoned by relocations
    scratch::Vector{Int32}           # reusable buffer of `_table_recompute!`
    rebuilds::Int                    # full rebuilds (diagnostic)
end
QuadFaceTable() = QuadFaceTable(Dict{NTuple{3,Int},Int32}(), NTuple{3,Int}[],
                                Int8[], Int32[], Int32(0), Int32[], Int32[],
                                Int32[], Int32[], Int32(1), Int32[], falses(0),
                                Int32(0), Int32[], 0)

"""
    _bitvec_grow!(b::BitVector, n::Int)

Grow `b` to `n` bits, **zeroing** the new ones.  `resize!` on a `BitVector` only
zeroes the tail bits of the last old chunk; a garbage `true` in a fresh chunk
would silently drop a queue entry or a live id (measured: 7 of 150 random merges
missed a split before this loop).
"""
@inline function _bitvec_grow!(b::BitVector, n::Int)
    m = length(b)
    m >= n && return b
    resize!(b, n)
    for k in (m + 1):n
        b[k] = false
    end
    return b
end

"""
    FacePair

Concrete element type of [`face_pairs`](@ref).  The list used to be
`Vector{NamedTuple}` (an abstract element type): every consumer that read
`pf.leafL`/`pf.dir`/... paid a dynamic dispatch and a boxed field load, and
`push!` boxed each pair.  `--track-allocation=user` attributed 5.0 MB to the
`L, R, dir = pf.leafL, pf.leafR, pf.dir` destructuring and 2 x 1.25 MB to the
following `eta[L] += ...` lines per 4 AMR cycles on the 10 000-leaf state.
The values and the list order are unchanged; only the element type is concrete.
"""
const FacePair = NamedTuple{(:leafL, :leafR, :dir, :offset, :len),
                            Tuple{NTuple{3,Int},NTuple{3,Int},Symbol,Float64,Float64}}

"""Per-tree-version caches derived from the leaf set: the flat
[`QuadFaceTable`](@ref), the `face_pairs` list, and `stamp`, a counter bumped by
every `refine!`/`coarsen!`.  `refine!`/`coarsen!` keep the table up to date
incrementally and drop `pairs`; a `Quadtree` therefore never serves a stale face
list or pair (the mutation hooks mark exactly the leaves whose lists change).
`_invalidate_faces!` is the sledgehammer: it invalidates the table so the next
query rebuilds it from the leaf set."""
mutable struct QuadFaceCache
    stamp::Int
    pairs::Union{Nothing,Vector{FacePair}}
    tab::QuadFaceTable
end
QuadFaceCache() = QuadFaceCache(0, nothing, QuadFaceTable())

struct Quadtree
    maxlevel::Int
    Lx::Float64                 # physical width of the box; the height is always 1
    nx0::Int                    # base cells in x at level 0
    ny0::Int                    # base cells in y at level 0
    leaves::Set{NTuple{3,Int}}
    faces::QuadFaceCache
    function Quadtree(maxlevel::Int, Lx::Float64, nx0::Int, ny0::Int,
                      leaves::Set{NTuple{3,Int}})
        return new(maxlevel, Lx, nx0, ny0, leaves, QuadFaceCache())
    end
end

"""
    Quadtree(maxlevel; Lx=nothing, cells_x=1, cells_y=1)

`cells_x × cells_y` base cells over `[0,Lx]×[0,1]`, `Lx` defaulting to
`cells_x/cells_y`.  With `cells_x/cells_y = Lx` the leaves are **square**
(side `1/(cells_y*2^l)`); `cells_x=5, cells_y=2` is ASPECT's 2b geometry.  The
defaults `(1,1)` are the historical unit square, and `cells_x=cells_y=1, Lx≠1`
keeps the older coordinate stretching (anisotropic leaves).
"""
function Quadtree(maxlevel::Int; Lx::Union{Nothing,Float64}=nothing,
                  cells_x::Int=1, cells_y::Int=1)
    maxlevel >= 0 || throw(ArgumentError("maxlevel must be >= 0"))
    cells_x >= 1 || throw(ArgumentError("cells_x must be >= 1"))
    cells_y >= 1 || throw(ArgumentError("cells_y must be >= 1"))
    Lxb = Lx === nothing ? cells_x / cells_y : Float64(Lx)
    Lxb > 0.0 || throw(ArgumentError("Lx must be > 0"))
    if (cells_x != 1 || cells_y != 1) &&
       !isapprox(Lxb, cells_x / cells_y; rtol=1.0e-12)
        throw(ArgumentError("Lx=$(Lxb) is inconsistent with cells_x/cells_y = " *
                            "$(cells_x)/$(cells_y)"))
    end
    base = Set([(0, ix, iy) for iy in 0:(cells_y - 1) for ix in 0:(cells_x - 1)])
    Quadtree(maxlevel, Lxb, cells_x, cells_y, base)
end

# Bump the tree version of the derived caches (`face_pairs`; the flat table is
# kept current by the mutation hooks below).  `_invalidate_faces!` is the
# sledgehammer for callers that replace the leaf set wholesale (a checkpoint
# load, a tree copy in a test): it drops the table, so the next query rebuilds
# it from the leaf set.
@inline function _touch_faces!(qt::Quadtree)
    c = qt.faces
    c.stamp += 1
    c.pairs = nothing
    return nothing
end

@inline function _invalidate_faces!(qt::Quadtree)
    _touch_faces!(qt)
    qt.faces.tab.cur = Int32(0)
    return nothing
end

@inline nside(qt::Quadtree, l::Int) = 1 << l
@inline nside_x(qt::Quadtree, l::Int) = qt.nx0 * (1 << l)
@inline nside_y(qt::Quadtree, l::Int) = qt.ny0 * (1 << l)
# Physical leaf size; with cells_x/cells_y == Lx the two are identical.
@inline leaf_hx(qt::Quadtree, l::Int) = qt.Lx / (qt.nx0 * (1 << l))
@inline leaf_hy(qt::Quadtree, l::Int) = 1.0 / (qt.ny0 * (1 << l))
@inline leaf_h(qt::Quadtree, l::Int) = leaf_hy(qt, l)
@inline leaf_area(qt::Quadtree, l::Int) = leaf_hx(qt, l) * leaf_hy(qt, l)
total_area(qt::Quadtree) = sum(leaf_area(qt, l) for (l, _, _) in qt.leaves)

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
        jy < qt.ny0 * (1 << l) || return out
    else                            # :yminus
        jx = ix; jy = iy - 1
        iy > 0 || return out
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

"""Split one leaf into four children.  Returns false at `maxlevel`."""
function refine!(qt::Quadtree, leaf::NTuple{3,Int})
    l, ix, iy = leaf
    leaf in qt.leaves || return false
    l < qt.maxlevel || return false
    _faces_drop_leaf!(qt, leaf)
    delete!(qt.leaves, leaf)
    for dy in 0:1, dx in 0:1
        c = (l + 1, 2ix + dx, 2iy + dy)
        push!(qt.leaves, c)
        _faces_add_leaf!(qt, c)
    end
    _touch_faces!(qt)
    return true
end

"""Merge four siblings into their parent.  Returns false if they are not all leaves."""
function coarsen!(qt::Quadtree, leaf::NTuple{3,Int})
    l, ix, iy = leaf
    l > 0 || return false
    px, py = ix >> 1, iy >> 1
    kids = ((l, 2px, 2py), (l, 2px, 2py + 1), (l, 2px + 1, 2py),
            (l, 2px + 1, 2py + 1))
    all(k -> k in qt.leaves, kids) || return false
    for k in kids
        _faces_drop_leaf!(qt, k)
        delete!(qt.leaves, k)
    end
    p = (l - 1, px, py)
    push!(qt.leaves, p)
    _faces_add_leaf!(qt, p)
    _touch_faces!(qt)
    return true
end

"""
    _collect_face!(out, qt, l, ix, iy, vertical, coord, lo, hi)

Historical face-collection kernel: descend the node `(l, ix, iy)` and push every
leaf whose box touches the face segment `coord` inside `[lo, hi]`.  Kept as the
**reference oracle** of the flat table (`_neighbors_scan` below): it is the
implementation the table replaced, and G216 requires the two to agree leaf by
leaf on random trees (`docs/dev/AMR.md` §9).  It allocates nothing
itself but pushes into `out` for every hit, and it re-descends from the base
cells on every query -- that is the 223.9 MB of the audit.
"""
function _collect_face!(out::Vector{NTuple{3,Int}}, qt::Quadtree,
                        l::Int, ix::Int, iy::Int,
                        vertical::Bool, coord::Float64,
                        lo::Float64, hi::Float64)
    h = leaf_h(qt, l)
    x0 = ix * h; y0 = iy * h
    tol = 1.0e-12
    if vertical
        (x0 - tol <= coord <= x0 + h + tol) || return nothing
        (min(y0 + h, hi) - max(y0, lo) > tol) || return nothing
    else
        (y0 - tol <= coord <= y0 + h + tol) || return nothing
        (min(x0 + h, hi) - max(x0, lo) > tol) || return nothing
    end
    if (l, ix, iy) in qt.leaves
        push!(out, (l, ix, iy))
        return nothing
    end
    l >= qt.maxlevel && return nothing
    for dy in 0:1, dx in 0:1
        _collect_face!(out, qt, l + 1, 2ix + dx, 2iy + dy,
                       vertical, coord, lo, hi)
    end
    return nothing
end

"""
    _neighbors_scan(qt, leaf, dir)

The historical `neighbors` implementation, verbatim (float-coordinate descent
from the base cells).  Reference oracle for the flat table; **not** on any hot
path.  `neighbors` falls back to it for a key that is not a leaf of the tree, so
the two agree for every input.
"""
function _neighbors_scan(qt::Quadtree, leaf::NTuple{3,Int}, dir::Symbol)
    dir in FACE_DIRS || throw(ArgumentError("unknown direction $dir"))
    l, ix, iy = leaf
    h = leaf_h(qt, l)
    x0 = ix * h; y0 = iy * h
    out = NTuple{3,Int}[]
    # With several base cells (cells_x x cells_y) the face search has to start
    # from every base cell, not from a single root.
    for by in 0:(qt.ny0 - 1), bx in 0:(qt.nx0 - 1)
        if dir === :xplus
            _collect_face!(out, qt, 0, bx, by, true, x0 + h, y0, y0 + h)
        elseif dir === :xminus
            _collect_face!(out, qt, 0, bx, by, true, x0, y0, y0 + h)
        elseif dir === :yplus
            _collect_face!(out, qt, 0, bx, by, false, y0 + h, x0, x0 + h)
        else
            _collect_face!(out, qt, 0, bx, by, false, y0, x0, x0 + h)
        end
    end
    filter!(o -> o != leaf, out)
    return out
end

"""
    neighbors(qt, leaf, dir)

All leaves touching the given face of `leaf` (one, two, or more when the
neighbor is refined; a coarser leaf covers the whole face), in increasing
coordinate along the face.

This is the **compatibility wrapper** of the flat neighbour table: it returns a
fresh `Vector` per call (the historical contract: callers may keep or mutate it)
and is meant for code that iterates the vector.  The hot loops read the table
directly (`_table_leaf_id!` / `_face_slot` / `_leaf_nbs!` / `_leaf21`), which
allocates nothing per query -- one `Vector` per `(leaf, dir)` was 74.6 MB per 4
AMR cycles (`docs/dev/AMR.md` §4).  A key that is not a leaf of the
tree is answered by the historical `_neighbors_scan`.
"""
function neighbors(qt::Quadtree, leaf::NTuple{3,Int}, dir::Symbol)
    d = _diridx(dir)
    id = _table_leaf_id!(qt, leaf)
    id == Int32(0) && return _neighbors_scan(qt, leaf, dir)
    tab = _table_ready!(qt, id)
    o, n = _face_slot(tab, id, d)
    out = Vector{NTuple{3,Int}}(undef, n)
    @inbounds for k in 0:(n - 1)
        out[k + 1] = tab.keys[tab.pool[o + k]]
    end
    return out
end

"""
    has_no_face_neighbor(qt, leaf, dir)

`true` when the face of `leaf` in direction `dir` has no neighbour, i.e. it is a
domain boundary.  Allocation-free replacement for
`isempty(neighbors(qt, leaf, dir))`, which is how the assembly and the
constraints ask for the boundary faces (4 queries per leaf per assembly).
"""
@inline function has_no_face_neighbor(qt::Quadtree, leaf::NTuple{3,Int},
                                      dir::Symbol)
    d = _diridx(dir)
    id = _table_leaf_id!(qt, leaf)
    id == Int32(0) && return isempty(_neighbors_scan(qt, leaf, dir))
    tab = _table_ready!(qt, id)
    return Int(tab.slen[4 * (Int(id) - 1) + d]) == 0
end

# ---------------------------------------------------------------------------
# x-mirror symmetry (`symmetry=:x` in `amr_cycle_balanced!` / the budget pass)
# ---------------------------------------------------------------------------
#
# `symmetry=:x` keeps the leaf set closed under the reflection x -> Lx - x: for
# every leaf `(l, ix, iy)` its partner `(l, nx0*2^l - 1 - ix, iy)` is a leaf
# too.  Every *tree* operation must therefore act on mirror pairs.  The marking
# and the coarsening pass always did, but three operations used to act on one
# leaf at a time and slowly broke the invariant on a long run:
#
#   * `balance!` split a single leaf for the 2:1 condition,
#   * `_coarsen_pair!` merged only the `lf` quartet when the mirror quartet was
#     protected (`protect` is built from the theta field, so it is not
#     mirror-symmetric: measured 81 non-mirrored leaves from 0 in one pass),
#   * `semquad_enforce_leaf_budget!` and the removed classic cycle's hard-budget
#     guard did the same when the mirror quartet was protected or incomplete.
#
# A single one-sided merge leaves **five** non-mirrored leaves (the merged
# parent plus the four children of the still-split mirror parent), which is why
# the realistic checkpoint's count moves in fives (measured 120 at step 6130).
# The two operations that can run away over thousands of cycles (balance and the
# hard-budget merge) are pair-aware here, `_coarsen_pair!` refuses a pair it
# cannot merge on both sides, and `mirror_close!` repairs a tree that was
# damaged before the fix (`docs/dev/CASE_REALISTIC.md` §15).

"""Mirror of a leaf about x = Lx/2: `(l, nx0*2^l - 1 - ix, iy)`."""
@inline _mirror_leaf(qt::Quadtree, lf::NTuple{3,Int}) =
    (lf[1], nside_x(qt, lf[1]) - 1 - lf[2], lf[3])

"""
    mirror_defect(qt)

Number of leaves whose x-mirror partner is not a leaf of the tree.  Zero is the
`symmetry=:x` invariant; `mirror_close!` restores it, `balance!`/`amr_cycle_balanced!`
with `symmetry=:x` preserve it.
"""
function mirror_defect(qt::Quadtree)
    n = 0
    for lf in qt.leaves
        _mirror_leaf(qt, lf) in qt.leaves || (n += 1)
    end
    return n
end

"""The leaf covering the mirror box of `lf`: a leaf at a level below `lf[1]`
(the mirror box is coarser), the node at `lf[1]` itself (the mirror box is
subdivided, so the mirror is *finer*), or the mirror leaf itself."""
function _mirror_container(qt::Quadtree, lf::NTuple{3,Int})
    l, ix, iy = _mirror_leaf(qt, lf)
    node = (0, ix >> l, iy >> l)
    while !(node in qt.leaves)
        node[1] < l || return node      # the box itself is subdivided
        node = (node[1] + 1, ix >> (l - node[1] - 1), iy >> (l - node[1] - 1))
    end
    return node
end

"""The 2:1 test for one leaf (a face neighbor two levels finer).  Reads the flat
table; the historical scan is the fallback for a key that is not a leaf."""
function _needs_refine_21(qt::Quadtree, leaf::NTuple{3,Int})
    id = _table_leaf_id!(qt, leaf)
    if id == Int32(0)
        for dir in FACE_DIRS
            for nb in _neighbors_scan(qt, leaf, dir)
                nb[1] >= leaf[1] + 2 && return true
            end
        end
        return false
    end
    tab = _table_ready!(qt, id)
    return _leaf21(tab, id)
end

"""
    _mirror_protect(qt, protect)

Close a `protect` set under the x-mirror: under `symmetry=:x` a protected leaf
protects its mirror partner too.  Without this the pair-strict merges would be
refused by an *asymmetric* protect entry -- a protect set derived from a field
that is only approximately symmetric (the removed classic cycle's
`coarsen_theta_tol` guard), or a 30 % random protect set, refuses ~94 % of the
pairs, i.e. the tree stops coarsening at all.
Protecting the partner instead keeps the invariant *and* the coarsening
ability; it is the conservative direction (never coarsen a leaf whose mirror is
protected either).
"""
function _mirror_protect(qt::Quadtree, protect::Set{NTuple{3,Int}})
    isempty(protect) && return protect
    out = Set{NTuple{3,Int}}(protect)
    for lf in protect
        ml = _mirror_leaf(qt, lf)
        ml != lf && push!(out, ml)
    end
    return out
end

"""
    LAST_MIRROR_CLOSE

Diagnostics of the last [`mirror_close!`](@ref) pass: `(refined, ok, rounds,
capped)`.  `ok` is the invariant (`mirror_defect == 0`), `rounds` the number of
passes the repair needed, and `capped` whether it stopped at `maxiter` instead of
fixing the tree.  `mirror_close!` used to return `(refined, ok)` and every caller
that ignored `ok` (the AMR cycle does) could leave a non-mirror-closed tree
silently; the counter makes the cap visible, and `amr_cycle_balanced!` records it
in [`LAST_AMR_BALANCED`](@ref).
"""
const LAST_MIRROR_CLOSE = Ref((refined=0, ok=true, rounds=0, capped=false))

"""
    mirror_close!(qt; max_level=qt.maxlevel, maxiter=1000)

Repair pass for the `symmetry=:x` invariant.  While some leaf's mirror partner
is not a leaf, refine the **coarser** side of that mismatch (the coarse leaf
covering the mirror box, or the leaf itself when the mirror box is already
subdivided).  Refinement is monotone, so this terminates; it stops at
`max_level`, which bounds the repair but can leave a defect that no refinement
can fix (the second return value is then false).  The cost is bounded by the
size of the closed supertree: the result is the common refinement of the tree
and its mirror image, which has at most twice the leaves of a closed tree (it is
a refinement, so it can push the leaf count over `max_leaves`; the budget pass
runs after this and merges mirror pairs, so the final tree has at most the
budget plus the documented one-pair slack).

Returns `(refined, ok)` with `ok == (mirror_defect(qt) == 0)`.  The progress
measure is `refined` itself (every pass that changes anything splits at least
one leaf), so the `changed` flag cannot spin: a pass with `refined == 0` leaves
the tree bit-for-bit identical and the loop breaks.  `LAST_MIRROR_CLOSE` records
`rounds` and whether the `maxiter` cap fired.
"""
function mirror_close!(qt::Quadtree; max_level::Int=qt.maxlevel,
                       maxiter::Int=1000)
    refined = 0
    rounds = 0
    capped = false
    for _ in 1:maxiter
        rounds += 1
        changed = false
        for lf in sort(collect(qt.leaves))
            lf in qt.leaves || continue
            _mirror_leaf(qt, lf) in qt.leaves && continue
            c = _mirror_container(qt, lf)
            target = c[1] < lf[1] ? c : lf
            target[1] < max_level || continue
            if refine!(qt, target)
                refined += 1
                changed = true
            end
        end
        changed || break
        capped = rounds >= maxiter
    end
    ok = mirror_defect(qt) == 0
    LAST_MIRROR_CLOSE[] = (refined=refined, ok=ok, rounds=rounds, capped=capped)
    return refined, ok
end

"""Worklist 2:1 balance for `symmetry=:x`: whenever a leaf has to be split for
the 2:1 condition, its mirror partner is split with it (both or neither, so a
pair at `maxlevel` stays untouched).  Identical to `balance!` otherwise.

Both variants read the flat neighbour table (one `Int32` compare per neighbour
level, no per-query vector) and the mutations keep it current, so a wave costs
O(#wave) array reads instead of 4 queries per leaf through the historical
descent."""
function _balance_mirror_x!(qt::Quadtree; maxiter::Int=1000)
    wave = sort(collect(qt.leaves))
    next = Set{NTuple{3,Int}}()
    nbs = NTuple{3,Int}[]
    pair = NTuple{3,Int}[]
    for _ in 1:maxiter
        isempty(wave) && return true
        prev = wave
        empty!(next)
        for leaf in wave
            id = _table_leaf_id!(qt, leaf)
            id == Int32(0) && continue
            _leaf21(_table_ready!(qt, id), id) || continue
            empty!(pair)
            push!(pair, leaf)
            ml = _mirror_leaf(qt, leaf)
            if ml != leaf && ml in qt.leaves
                push!(pair, ml)     # a closed tree splits both sides or neither
            end
            for lf in pair
                lid = _table_leaf_id!(qt, lf)
                lid == Int32(0) && continue
                # Snapshot the old face neighbors before the split: together
                # with the new children they are exactly the leaves whose 2:1
                # status can have changed.  (The table is updated by the split,
                # so the snapshot has to be taken first.)
                _leaf_nbs!(nbs, _table_ready!(qt, lid), lid)
                l, ix, iy = lf
                if refine!(qt, lf)
                    for dy in 0:1, dx in 0:1
                        push!(next, (l + 1, 2ix + dx, 2iy + dy))
                    end
                else
                    push!(next, lf)     # maxlevel: cannot be split
                end
                for nb in nbs
                    push!(next, nb)
                end
            end
        end
        wave = sort(collect(next))
        wave == prev && break
    end
    return false
end

"""
    balance!(qt; maxiter=1000, symmetry=:none)

Refine leaves until every face neighbor differs by at most one level.  Returns
false if `maxiter` refinement waves were exhausted (only possible when a leaf at
`maxlevel` still has a two-level-finer neighbor, which no refinement can fix).

The 2:1 condition is local, so this is a worklist sweep instead of a full-tree
re-scan per round: the first wave checks every leaf once, and after a leaf is
split only its four children and its old face neighbors (the only leaves whose
neighbor lists change) are queued for the next wave.  The classical local
scheme (p4est `balance`, deal.II `prepare_coarsening_and_refinement`) does the
same; the fixpoint is the unique minimal 2:1 refinement of the input tree, so
the leaf set is identical to the full-scan version.

`symmetry=:x` splits the mirror partner of every leaf that has to be split, so
the tree stays closed under x -> Lx - x (see `mirror_defect`); `:none` is the
historical behaviour, unchanged.
"""
function balance!(qt::Quadtree; maxiter::Int=1000, symmetry::Symbol=:none)
    symmetry in (:none, :x) || throw(ArgumentError("symmetry must be :none or :x"))
    symmetry === :x && return _balance_mirror_x!(qt; maxiter=maxiter)
    wave = sort(collect(qt.leaves))
    next = Set{NTuple{3,Int}}()
    nbs = NTuple{3,Int}[]
    for _ in 1:maxiter
        isempty(wave) && return true
        prev = wave
        empty!(next)
        for leaf in wave
            id = _table_leaf_id!(qt, leaf)
            id == Int32(0) && continue
            _leaf21(_table_ready!(qt, id), id) || continue
            # Snapshot the old face neighbors before the split: together with
            # the new children they are exactly the leaves whose 2:1 status can
            # have changed.
            _leaf_nbs!(nbs, qt.faces.tab, id)
            l, ix, iy = leaf
            if refine!(qt, leaf)
                for dy in 0:1, dx in 0:1
                    push!(next, (l + 1, 2ix + dx, 2iy + dy))
                end
            else
                push!(next, leaf)   # maxlevel: cannot be split
            end
            for nb in nbs
                push!(next, nb)
            end
        end
        wave = sort(collect(next))
        # Fixed-point guard (same bug class as the step-95 leaf-budget spin,
        # docs/dev/SOLVER.md §16): the next wave is a pure function of the leaf
        # set and of `refine!`'s success, so `wave == prev` means *no* leaf was
        # split this pass and the following pass would repeat this one exactly.
        # A leaf that `refine!` refuses (it is at `maxlevel`) used to be
        # re-queued and retried `maxiter` times: 1000 full no-op passes.  The
        # comparison is O(#leaves) on an already-sorted vector and exits at the
        # first difference, i.e. it costs a handful of comparisons when the
        # tree is still making progress (measured below 0.1% of `balance!`).
        wave == prev && break
    end
    return false
end

"""Refine every leaf for which `pred(level, ix, iy)` is true, iterating until
no leaf qualifies (the predicate is re-evaluated on the new children)."""
function refine_where!(qt::Quadtree, pred::Function)
    for _ in 1:qt.maxlevel
        any_ref = false
        for leaf in collect(qt.leaves)
            leaf in qt.leaves || continue
            if pred(leaf...) && leaf[1] < qt.maxlevel
                refine!(qt, leaf)
                any_ref = true
            end
        end
        any_ref || break
        balance!(qt)
    end
    balance!(qt)
end

# ---------------------------------------------------------------------------
# Local (incremental) 2:1 rebalance
# ---------------------------------------------------------------------------
#
# `balance!` is a global fixed point: it sorts the whole leaf set and re-tests
# every leaf, which is 11-25 ms on a 10 000-leaf tree.  The hard budget pass
# calls it once per merge (14-29 times per cycle) to learn that the one new
# parent usually survives.
#
# The 2:1 condition is local, and the reference
# (`docs/dev/AMR.md`, verified there) makes the replacement
# exact:
#
#   **Lemma 1.** If `T` is 2:1 and `T'` replaces one complete quartet by its
#   parent `p`, then `p` is the *only* leaf of `T'` that is not 2:1.
#   **Lemma 2.** If a violating leaf `x` is split, the only leaves that can newly
#   violate are the face neighbours of `x` one level coarser than `x`.
#   **Theorem.** With those seeds the queue closes at exactly `balance!`'s leaf
#   set (the least 2:1 refinement), *provided the tree is 2:1 at entry*;
#   without that precondition there is an explicit counterexample
#   (`docs/dev/AMR.md` §4.1).
#
# So the pass keeps a FIFO of leaf ids (`queued` dedups, cleared on pop so a
# leaf tested clean can be re-tested after its neighbour changes), splits only
# actual violators, and stops when the queue is empty.  The entry precondition
# is *checked*, not assumed: a tree that enters unbalanced is repaired by
# seeding the queue with every violator (`all_violations=true`), which makes the
# pass compute the same closure as `balance!` from that tree too.

"""
    LocalBalance()

Reusable work-queue state of [`rebalance_local!`](@ref): the FIFO of leaf ids
(parallel to `QuadFaceTable.ids`), a `queued` bit per live id, a reusable
neighbour buffer, the parents of the leaves the last pass split, and the pass
counters (`ntests`, `nsplits`, `capped`, `unsplittable`).  One instance is
reused for the whole run ([`LOCAL_BALANCE`](@ref)), so a pass allocates nothing
once the vectors are warm.
"""
mutable struct LocalBalance
    queue::Vector{Int32}                 # leaf ids waiting to be tested
    queued::BitVector                    # 1 = id is in the queue
    head::Int                            # FIFO read cursor
    nbs::Vector{NTuple{3,Int}}           # reusable neighbour snapshot
    parents::Vector{NTuple{3,Int}}       # `_pq` of every leaf the pass split
    ntests::Int
    nsplits::Int
    unsplittable::Int
    capped::Bool
end
LocalBalance() = LocalBalance(Int32[], falses(0), 1, NTuple{3,Int}[],
                              NTuple{3,Int}[], 0, 0, 0, false)

"Default work-queue state of [`rebalance_local!`](@ref) (one per process, reused
by every pass: the vectors only grow to the largest tree seen)."
const LOCAL_BALANCE = LocalBalance()

# Structural cap of one pass: every pop either finds a stale entry or tests a
# live leaf, and only a split enqueues new work (<= 4 children + the <= 8 face
# neighbours), so a real closure uses far less than this.  Firing it sets
# `capped` instead of spinning -- the `balance!` `wave == prev` lesson.
@inline _local_test_cap(n::Int) = 64 * max(n, 64) + 1024

@inline function _lb_push!(st::LocalBalance, id::Int32, nid::Int)
    id == Int32(0) && return nothing
    _bitvec_grow!(st.queued, nid)
    st.queued[id] && return nothing
    st.queued[id] = true
    push!(st.queue, id)
    return nothing
end

"""
    _tree_is_21(qt) -> Bool

`true` when no leaf has a face neighbour two or more levels finer: the entry
precondition of [`rebalance_local!`](@ref).  O(#leaves) `Int8` compares on the
flat table (no neighbour lists are built), so the budget pass can afford it
once per pass.
"""
function _tree_is_21(qt::Quadtree)
    tab = _table_ensure_built!(qt)
    for id in Int32(1):Int32(length(tab.keys))
        tab.live[id] || continue
        _table_ready!(qt, id)
        _leaf21(tab, id) && return false
    end
    return true
end

# Split `lf` (a violating leaf) and enqueue its four children and its old face
# neighbours (Lemma 2: only `N(x)` at one level coarser can newly violate).
# `nbs` is the pre-split snapshot of `N(lf)`.
function _lb_split!(qt::Quadtree, st::LocalBalance, tab::QuadFaceTable,
                    lf::NTuple{3,Int})
    l, ix, iy = lf
    if !refine!(qt, lf)
        st.unsplittable += 1            # at `maxlevel`: no refinement can fix it
        return nothing
    end
    st.nsplits += 1
    push!(st.parents, _pq(lf))
    nid = length(tab.keys)
    for dy in 0:1, dx in 0:1
        cid = get(tab.ids, (l + 1, 2ix + dx, 2iy + dy), Int32(0))
        _lb_push!(st, cid, nid)
    end
    for nb in st.nbs
        _lb_push!(st, get(tab.ids, nb, Int32(0)), nid)
    end
    return nothing
end

"""
    rebalance_local!(qt, st=LOCAL_BALANCE; seeds=(), symmetry=:none,
                     all_violations=false) -> (nsplits, ok)

2:1-rebalance `qt` locally, starting from the leaves in `seeds` (and their face
neighbours): split every leaf that faces a neighbour two levels finer, enqueueing
the new children and the old neighbours of every split, until the queue is
empty.  Returns the number of splits and `ok = !capped && unsplittable == 0`.

The result is the leaf set of `balance!(qt)` **provided `qt` was 2:1 before the
mutation that produced the seeds** (`docs/dev/AMR.md` §4.1:
Lemmas 1-2 make the queue closed under the only two events that create a
violation, and the closure is the unique least 2:1 refinement).  For a tree that
is not 2:1 (a legacy checkpoint, a hand-built test tree) pass
`all_violations=true`: the queue is then seeded with every violator, which makes
the pass compute the same closure as the global scan.

Under `symmetry=:x` a split is pair-atomic: the mirror partner is split too
(when it is a leaf), `:x`-invariant trees stay invariant, and a partner at
`maxlevel` counts as `unsplittable` instead of leaving a one-sided split
silently -- the repair path for a damaged tree remains `mirror_close!`.

`st.parents` lists the parents of every leaf the pass split (per pass): the
budget pass blocks all of them, so a quartet the rebalance undid is never
re-merged (the S16 spin of `docs/dev/AMR.md` §3(e)).
"""
function rebalance_local!(qt::Quadtree, st::LocalBalance=LOCAL_BALANCE;
                          seeds::Tuple=(), symmetry::Symbol=:none,
                          all_violations::Bool=false)
    symmetry in (:none, :x) ||
        throw(ArgumentError("symmetry must be :none or :x"))
    tab = _table_ensure_built!(qt)
    nid = length(tab.keys)
    for id in st.queue
        st.queued[id] = false           # a capped pass leaves bits set
    end
    empty!(st.queue)
    empty!(st.parents)
    st.head = 1
    st.ntests = 0
    st.nsplits = 0
    st.unsplittable = 0
    st.capped = false
    # ---- seeds: each requested leaf, its face neighbours, and its mirror ----
    for lf in seeds
        id = get(tab.ids, lf, Int32(0))
        id == Int32(0) && continue
        _lb_push!(st, id, nid)
        _leaf_nbs!(st.nbs, _table_ready!(qt, id), id)
        for nb in st.nbs
            _lb_push!(st, get(tab.ids, nb, Int32(0)), nid)
        end
        if symmetry === :x
            ml = _mirror_leaf(qt, lf)
            if ml != lf
                mid = get(tab.ids, ml, Int32(0))
                if mid != Int32(0)
                    _lb_push!(st, mid, nid)
                    _leaf_nbs!(st.nbs, _table_ready!(qt, mid), mid)
                    for nb in st.nbs
                        _lb_push!(st, get(tab.ids, nb, Int32(0)), nid)
                    end
                end
            end
        end
    end
    # ---- entry precondition: seed every violator of a tree that is not 2:1 --
    if all_violations
        for id in Int32(1):Int32(nid)
            tab.live[id] || continue
            _table_ready!(qt, id)
            _leaf21(tab, id) && _lb_push!(st, id, nid)
        end
    end
    # ---- drain: split the violators, enqueue what a split can affect ---------
    cap = _local_test_cap(nid)
    while st.head <= length(st.queue)
        if st.ntests > cap
            st.capped = true
            break
        end
        id = st.queue[st.head]
        st.head += 1
        st.queued[id] && (st.queued[id] = false)
        id <= length(tab.keys) || continue
        tab.live[id] || continue        # stale entry: the leaf is gone
        lf = tab.keys[id]
        st.ntests += 1
        _table_ready!(qt, id)
        _leaf_nbs!(st.nbs, tab, id)     # snapshot BEFORE the split
        _leaf21(tab, id) || continue
        _lb_split!(qt, st, tab, lf)
        if symmetry === :x
            ml = _mirror_leaf(qt, lf)
            if ml != lf
                mid = get(tab.ids, ml, Int32(0))
                if mid != Int32(0)
                    _table_ready!(qt, mid)
                    _leaf_nbs!(st.nbs, tab, mid)
                    _lb_split!(qt, st, tab, ml)
                end
            end
        end
    end
    return st.nsplits, (!st.capped && st.unsplittable == 0)
end

"""Check the 2:1 condition on every face; returns the worst level difference."""
function max_face_level_jump(qt::Quadtree)
    worst = 0
    for leaf in qt.leaves
        id = _table_leaf_id!(qt, leaf)
        id == Int32(0) && continue
        tab = _table_ready!(qt, id)
        l = Int(tab.lev[id])
        for d in 1:4
            o, n = _face_slot(tab, id, d)
            @inbounds for k in 0:(n - 1)
                worst = max(worst, abs(Int(tab.lev[tab.pool[o + k]]) - l))
            end
        end
    end
    return worst
end

"""
    wall_graded_tree(base, wall_level; strip=0.1)

Uniform tree at `base`, then geometric wall refinement: a leaf at distance `d`
from the nearest wall is refined while its level is below
`min(wall_level, base + ceil(log2(strip/d)))`.  The strip therefore halves as
the level increases, so the number of wall leaves grows linearly with
`wall_level` rather than quadratically.  This is the quadtree analogue of the
tanh-graded static mesh used by the high-Ra runs.
"""
function wall_graded_tree(base::Int, wall_level::Int; strip::Float64=0.1)
    base >= 0 || throw(ArgumentError("base must be >= 0"))
    wall_level >= base || throw(ArgumentError("wall_level must be >= base"))
    0.0 < strip <= 0.5 || throw(ArgumentError("strip must be in (0, 0.5]"))
    qt = Quadtree(wall_level + 2)
    for _ in 1:base
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
    end
    # One pass per level, no balance! inside the loop: the target level is
    # 1-Lipschitz in the cell size, so the tree is already 2:1 (or needs only
    # one final balance).  Calling refine_where! here would call the O(N^2)
    # balance! after every level and hangs at level ~12.
    for _ in 1:(wall_level - base)
        changed = false
        for lf in collect(qt.leaves)
            l, ix, iy = lf
            h = leaf_h(qt, l)
            y0 = iy * h
            d = min(y0, 1.0 - (y0 + h))
            target = d <= 0.0 ? wall_level :
                     d < strip ? min(wall_level, base + ceil(Int, log2(strip / d))) : l
            if l < target
                refine!(qt, lf)
                changed = true
            end
        end
        changed || break
    end
    balance!(qt)
    return qt
end

"""Physical `(x0, y0, hx, hy)` of a leaf."""
function leaf_box(qt::Quadtree, leaf::NTuple{3,Int})
    l, ix, iy = leaf
    hx = leaf_hx(qt, l); hy = leaf_hy(qt, l)
    return (ix * hx, iy * hy, hx, hy)
end


"""
    face_pairs(qt)

Interior sub-faces of the balanced tree, each listed once.  The coarser leaf is
always `leafL` (`leafL[1] <= leafR[1]`); `dir` points from the coarse leaf to
the fine one, and the sub-face runs from `offset` to `offset+len` along the
face (y for x-faces, x for y-faces).  A 2:1 coarse face therefore appears as two
sub-faces, one per fine neighbor.

The list is cached per tree version (see `QuadFaceCache`): the indicator,
constraints and assembly all consume it for the same mesh, and it is a pure
function of the leaf set.  Callers must not mutate the returned vector.
"""
function face_pairs(qt::Quadtree)
    cached = qt.faces.pairs
    cached === nothing || return cached
    out = FacePair[]
    seen = Set{NTuple{7,Int}}()
    for leaf in sort(collect(qt.leaves)), d in 1:4
        id = _table_leaf_id!(qt, leaf)
        id == Int32(0) && continue
        tab = _table_ready!(qt, id)
        o, n = _face_slot(tab, id, d)
        dir = FACE_DIRS[d]
        @inbounds for k in 0:(n - 1)
            nb = tab.keys[tab.pool[o + k]]
            a, b = leaf, nb
            if a[1] > b[1]
                a, b = b, a
            elseif a[1] == b[1] && !(a < b)
                continue
            end
            hA = leaf_h(qt, a[1]); hB = leaf_h(qt, b[1])
            xa = a[2] * hA; ya = a[3] * hA
            xb = b[2] * hB; yb = b[3] * hB
            dd = :none
            # the fine interval must be contained in the coarse face interval
            yin = yb >= ya - 1e-12 && yb + hB <= ya + hA + 1e-12
            xin = xb >= xa - 1e-12 && xb + hB <= xa + hA + 1e-12
            if abs(xb - (xa + hA)) < 1e-12 && yin
                dd = :xplus
            elseif abs(xa - (xb + hB)) < 1e-12 && yin
                dd = :xminus
            elseif abs(yb - (ya + hA)) < 1e-12 && xin
                dd = :yplus
            elseif abs(ya - (yb + hB)) < 1e-12 && xin
                dd = :yminus
            end
            dd === :none && continue
            off = dd in (:xplus, :xminus) ? yb : xb
            key = (a[1], a[2], a[3], b[1], b[2], b[3], dd === :xplus ? 1 : dd === :xminus ? 2 : dd === :yplus ? 3 : 4)
            key in seen && continue
            push!(seen, key)
            push!(out, (leafL=a, leafR=b, dir=dd, offset=off, len=hB))
        end
    end
    qt.faces.pairs = out
    return out
end


