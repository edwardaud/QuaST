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
    # Stage 0.5c1: the periodic seam of the logical `y` direction.  The tree owns
    # the *topology*; the geometry only supplies a map that closes (`PolarAnnulus`
    # with `Phi = 2*pi`, docs/dev/GEOM_DESIGN.md 6.7).
    #
    #   * `:none` (default) -- the historical box topology: `y` is a bounded
    #     interval and the four logical sides are the domain boundary.
    #   * `:periodic` -- the `y` direction is closed.  The last angular column
    #     `iy = ny0*2^l - 1` and the first column `iy = 0` are face neighbours
    #     across the seam (`_push_face_ids!` wraps the row index), `balance!`
    #     therefore enforces the 2:1 condition *across* the seam, and the SEM
    #     nodes at `T = 0`/`T = 1` are one node (`_logical_node_key`
    #     canonicalises the `T = 1` numerator to 0).  The `y` direction needs at
    #     least two base cells: a single cell would be its own cross-seam
    #     neighbour.
    #
    # The flag is part of the tree because the face table, the balance and the
    # node identity are topology; it is *not* read by any box-path body (every
    # box tree has `seam === :none`), so the box arithmetic is untouched.
    seam::Symbol
    function Quadtree(maxlevel::Int, Lx::Float64, nx0::Int, ny0::Int,
                      leaves::Set{NTuple{3,Int}}, seam::Symbol=:none)
        seam === :periodic && ny0 < 2 &&
            throw(ArgumentError("a periodic seam needs at least two base cells " *
                                "in the closed direction (a single base cell " *
                                "would be its own cross-seam neighbour), got " *
                                "cells_y = $ny0"))
        return new(maxlevel, Lx, nx0, ny0, leaves, QuadFaceCache(), seam)
    end
end

"""
    Quadtree(maxlevel; Lx=nothing, cells_x=1, cells_y=1, seam=:none)

`cells_x × cells_y` base cells over `[0,Lx]×[0,1]`, `Lx` defaulting to
`cells_x/cells_y`.  With `cells_x/cells_y = Lx` the leaves are **square**
(side `1/(cells_y*2^l)`); `cells_x=5, cells_y=2` is ASPECT's 2b geometry.  The
defaults `(1,1)` are the historical unit square, and `cells_x=cells_y=1, Lx≠1`
keeps the older coordinate stretching (anisotropic leaves).

`seam` (stage 0.5c1) selects the logical-`y` topology: `:none` is the historical
box (and every box tree); `:periodic` closes the `y` direction (the ring of the
2-D annulus, `PolarAnnulus` with `Phi = 2*pi`) -- see the [`Quadtree`](@ref)
field comment.  It needs `cells_y >= 2`.
"""
function Quadtree(maxlevel::Int; Lx::Union{Nothing,Float64}=nothing,
                  cells_x::Int=1, cells_y::Int=1, seam::Symbol=:none)
    maxlevel >= 0 || throw(ArgumentError("maxlevel must be >= 0"))
    cells_x >= 1 || throw(ArgumentError("cells_x must be >= 1"))
    cells_y >= 1 || throw(ArgumentError("cells_y must be >= 1"))
    seam in (:none, :periodic) ||
        throw(ArgumentError("a Quadtree seam must be :none or :periodic " *
                            "(got :$seam)"))
    Lxb = Lx === nothing ? cells_x / cells_y : Float64(Lx)
    Lxb > 0.0 || throw(ArgumentError("Lx must be > 0"))
    if (cells_x != 1 || cells_y != 1) &&
       !isapprox(Lxb, cells_x / cells_y; rtol=1.0e-12)
        throw(ArgumentError("Lx=$(Lxb) is inconsistent with cells_x/cells_y = " *
                            "$(cells_x)/$(cells_y)"))
    end
    base = Set([(0, ix, iy) for iy in 0:(cells_y - 1) for ix in 0:(cells_x - 1)])
    Quadtree(maxlevel, Lxb, cells_x, cells_y, base, seam)
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

# The tree is split by concern: this file holds the type, its constructor and
# the refinement primitives (`refine!`, `coarsen!`), the neighbour queries and
# the wall-graded builders below; `quadtree_table.jl` holds the flat face table
# they are answered from, and `quadtree_balance.jl` the x-mirror symmetry and
# the 2:1 balancing policy.

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

Stage 0.5c1: on a `seam = :periodic` tree the face coordinate of a `y` side is
wrapped (`1 -> 0` for `:yplus`, `0 -> 1` for `:yminus`), which is exactly the
cross-seam neighbour set the flat table computes by index arithmetic; the two are
compared leaf by leaf in G276 (b).  The wrap is a coordinate substitution, so the
oracle stays independent of the table logic.
"""
function _neighbors_scan(qt::Quadtree, leaf::NTuple{3,Int}, dir::Symbol)
    dir in FACE_DIRS || throw(ArgumentError("unknown direction $dir"))
    l, ix, iy = leaf
    h = leaf_h(qt, l)
    x0 = ix * h; y0 = iy * h
    # Periodic seam: a face on the closing coordinate is the *other* side of the
    # sector, so the scan coordinate wraps by one logical period.
    xp = x0 + h
    yp = y0 + h
    if qt.seam === :periodic
        yp >= 1.0 && (yp = 0.0)
    end
    ym = (qt.seam === :periodic && y0 <= 0.0) ? 1.0 : y0
    out = NTuple{3,Int}[]
    # With several base cells (cells_x x cells_y) the face search has to start
    # from every base cell, not from a single root.
    for by in 0:(qt.ny0 - 1), bx in 0:(qt.nx0 - 1)
        if dir === :xplus
            _collect_face!(out, qt, 0, bx, by, true, xp, y0, y0 + h)
        elseif dir === :xminus
            _collect_face!(out, qt, 0, bx, by, true, x0, y0, y0 + h)
        elseif dir === :yplus
            _collect_face!(out, qt, 0, bx, by, false, yp, x0, x0 + h)
        else
            _collect_face!(out, qt, 0, bx, by, false, ym, x0, x0 + h)
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
    wall_graded_tree(base, wall_level; strip=0.1, geom=nothing)

Uniform tree at `base`, then geometric wall refinement: a leaf at distance `d`
from the nearest wall is refined while its level is below
`min(wall_level, base + ceil(log2(strip/d)))`.  The strip therefore halves as
the level increases, so the number of wall leaves grows linearly with
`wall_level` rather than quadratically.  This is the quadtree analogue of the
tanh-graded static mesh used by the high-Ra runs.

`geom` (stage 0 cut 6, GEOM_DESIGN §3 cut 6) makes the wall strip a **physical
distance from the walls in the height coordinate**: with `geom=nothing` or a
`CartesianBox` the body is the historical Cartesian-`y` one, bit for bit; with a
mapped geometry the distance is measured in `height(g,x,y)` (the layer
coordinate the physics uses), so the annulus strips are radial strips instead of
logical y rows.  Every stage-0 geometry has `height = logical T` by
construction, so the tree is the same for the box and the validation geometries;
the difference becomes physical for the annulus (`height = r - R0`).
"""
function wall_graded_tree(base::Int, wall_level::Int; strip::Float64=0.1,
                          geom=nothing)
    base >= 0 || throw(ArgumentError("base must be >= 0"))
    wall_level >= base || throw(ArgumentError("wall_level must be >= base"))
    0.0 < strip <= 0.5 || throw(ArgumentError("strip must be in (0, 0.5]"))
    (geom === nothing || geom isa CartesianBox) &&
        return _wall_graded_tree_box(base, wall_level; strip=strip)
    return _wall_graded_tree_height(geom, base, wall_level; strip=strip)
end

# Historical box body, verbatim: do not rewrite it.
function _wall_graded_tree_box(base::Int, wall_level::Int; strip::Float64=0.1)
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

# Mapped body (cut 6): the same geometric grading with the distance measured in
# the physical height (`_height_at` maps the leaf corners through the geometry).
function _wall_graded_tree_height(g, base::Int, wall_level::Int;
                                  strip::Float64=0.1)
    qt = Quadtree(wall_level + 2)
    for _ in 1:base
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
    end
    for _ in 1:(wall_level - base)
        changed = false
        for lf in collect(qt.leaves)
            l = lf[1]
            h11 = _height_at(g, qt, lf, -1.0, -1.0)
            h21 = _height_at(g, qt, lf, 1.0, -1.0)
            h22 = _height_at(g, qt, lf, 1.0, 1.0)
            h12 = _height_at(g, qt, lf, -1.0, 1.0)
            lo = min(h11, h21, h22, h12)
            hi = max(h11, h21, h22, h12)
            d = min(lo, 1.0 - hi)
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

This body is the **bounded-box** topology: its pair direction is decided from
physical face intervals, so a cross-seam pair of a `seam = :periodic` tree would
be silently dropped (the two sides are one period apart).  A periodic tree must
therefore go through [`face_pairs`](@ref)`(qt, geom)` /
[`face_pairs_logical`](@ref) (the mapped path); the accessor refuses instead of
serving an incomplete list.
"""
function face_pairs(qt::Quadtree)
    qt.seam === :periodic &&
        throw(ArgumentError("face_pairs(qt) is the bounded-box pair list and " *
                            "cannot describe a periodic seam: the cross-seam " *
                            "pair of a seam = :periodic tree is one logical " *
                            "period apart, so the physical-interval test would " *
                            "drop it.  Use face_pairs(qt, geom) (the mapped " *
                            "list, face_pairs_logical) on a periodic tree; a " *
                            "periodic tree only pairs with a periodic geometry."))
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

"""
    face_pairs_logical(qt)

Interior sub-faces of the balanced tree rebuilt **from the neighbour table**
with index arithmetic only (GEOM_DESIGN §1.5(b)/§4.4): this is the topology the
mapped path uses.  It returns the same [`FacePair`](@ref) list as
[`face_pairs`](@ref) — same pairs, same order, same `dir` — but it never tests a
physical interval: the direction comes from the leaf indices (`b[2] > a[2]` is
`:xplus`, etc.) and the containment is guaranteed by the flat neighbour table.
Because the pair order follows `sort(collect(qt.leaves))` and the table's
increasing-coordinate neighbour order, it equals the historical list exactly on
a box with equal logical and physical units.

`offset` and `len` are **logical** face coordinates here (the fine leaf's start
and extent in `S`/`T`, in `[0,1]`); the historical `face_pairs` stores physical
ones.  The two agree on a square-leaf single-root unit box; a mapped geometry
passes the physical face point/measure to its kernels instead of using these
(the annulus stage).  The current consumers (indicator, constraints, assembly)
read only `leafL`, `leafR` and `dir`, so no caller depends on the units.

Cut 5 caches the list per `(leaf-set fingerprint, layout)` in
[`LOGICAL_FACE_PAIR_CACHE`](@ref); the mapped constraints and every mapped
assembly reuse it, so the warm mapped path does not rebuild it (the cut-2 record
measured that rebuild at ~1.5 MB per call at 1024 leaves).  The key carries no
geometry because the list is topology, not metric.  **Stage 0.5c1** adds the
tree's `seam` flag to the key: a periodic tree has extra cross-seam pairs on the
same leaf set, so a periodic and a bounded tree must never share an entry.

On a `seam = :periodic` tree the two `y` direction tests are **circular**: the
angular index arithmetic is modulo the fine level's column count, so the coarse
last column and the fine first column (and the reverse) are neighbours.  The
`x` tests stay exact as before.  A wrapped pair keeps the same meaning of
`offset`/`len` (the fine leaf's own logical start and extent), which for the
wrapped side is the `T = 0` side of the sector.
"""
@inline function _flip_dir(dir::Symbol)
    dir === :xplus && return :xminus
    dir === :xminus && return :xplus
    dir === :yplus && return :yminus
    return :yplus
end

"""
    LOGICAL_FACE_PAIR_CACHE

Per-tree cache of [`face_pairs_logical`](@ref) (cut 5): the logical pair list is
a pure function of the leaf set, the base cells and (stage 0.5c1) the tree's
`seam` topology — it must not see the geometry
(GEOM_DESIGN §2.1 rule 4) — so it is keyed by the leaf-set fingerprint, the
layout and the seam, and a geometry change never invalidates it (there is
nothing geometric to invalidate).  Same read-only contract as the box
`face_pairs` cache: callers must not mutate the returned vector.

Invalidation rule: a new key is a new leaf set/layout/seam; entries are never
mutated; `empty!(LOGICAL_FACE_PAIR_CACHE)` is the only clearing operation.
"""
const LOGICAL_FACE_PAIR_CACHE = Dict{Any,Vector{FacePair}}()

"""Cache key of one logical face-pair list: leaf set + layout + seam topology."""
function _logical_face_pair_key(qt::Quadtree)
    return (_leafset_hash(qt.leaves), length(qt.leaves), qt.nx0, qt.ny0,
            qt.Lx, qt.seam)
end

function face_pairs_logical(qt::Quadtree)
    key = _logical_face_pair_key(qt)
    cached = get(LOGICAL_FACE_PAIR_CACHE, key, nothing)
    cached === nothing || return cached
    out = _build_face_pairs_logical(qt)
    LOGICAL_FACE_PAIR_CACHE[key] = out
    return out
end

function _build_face_pairs_logical(qt::Quadtree)
    periodic = qt.seam === :periodic
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
            ddir = a === leaf ? dir : _flip_dir(dir)
            # The direction must agree with the leaf *intervals*: compare the
            # coarse interval (scaled to the fine leaf's index units) with the
            # fine one.  Same level is the sh == 0 case.  No floating point.
            # On a periodic tree the y tests are circular (see the docstring).
            la = a[1]
            lb = b[1]
            sh = lb - la
            sh >= 0 || continue
            if ddir === :xplus
                (b[2] == (a[2] + 1) << sh) || continue
                b[3] >= (a[3] << sh) && b[3] + 1 <= ((a[3] + 1) << sh) || continue
            elseif ddir === :xminus
                ((b[2] + 1) == (a[2] << sh)) || continue
                b[3] >= (a[3] << sh) && b[3] + 1 <= ((a[3] + 1) << sh) || continue
            elseif periodic
                nside_r = qt.ny0 * (1 << lb)
                # cross-face test: circular in the closed y direction
                if ddir === :yplus
                    (b[3] == mod((a[3] + 1) << sh, nside_r)) || continue
                else
                    (mod(b[3] + 1 - mod(a[3] << sh, nside_r), nside_r) == 0) ||
                        continue
                end
                # along-face extent: x is bounded, so the historical test holds
                b[2] >= (a[2] << sh) && b[2] + 1 <= ((a[2] + 1) << sh) || continue
            elseif ddir === :yplus
                (b[3] == (a[3] + 1) << sh) || continue
                b[2] >= (a[2] << sh) && b[2] + 1 <= ((a[2] + 1) << sh) || continue
            else
                ((b[3] + 1) == (a[3] << sh)) || continue
                b[2] >= (a[2] << sh) && b[2] + 1 <= ((a[2] + 1) << sh) || continue
            end
            if ddir === :xplus || ddir === :xminus
                len = 1.0 / (qt.ny0 * (1 << lb))
                off = b[3] * len
            else
                len = 1.0 / (qt.nx0 * (1 << lb))
                off = b[2] * len
            end
            key = (a[1], a[2], a[3], b[1], b[2], b[3],
                   ddir === :xplus ? 1 : ddir === :xminus ? 2 :
                   ddir === :yplus ? 3 : 4)
            key in seen && continue
            push!(seen, key)
            push!(out, (leafL=a, leafR=b, dir=ddir, offset=off, len=len))
        end
    end
    return out
end


