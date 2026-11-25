# x-mirror symmetry and 2:1 balancing.
#
# `symmetry = :x` keeps the leaf set closed under the reflection `x -> Lx - x`;
# `balance!` restores the 2:1 condition with the mirror-aware refinement
# `rebalance_local!` uses.  Both are policy on top of the tree primitives in
# `quadtree.jl`: they read the face table in `quadtree_table.jl` and split leaves,
# they never change the storage.

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
# damaged before the fix.

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
        # Fixed-point guard (same bug class as the step-95 leaf-budget spin):
        # the next wave is a pure function of the leaf
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
# (verified there) makes the replacement
# exact:
#
#   **Lemma 1.** If `T` is 2:1 and `T'` replaces one complete quartet by its
#   parent `p`, then `p` is the *only* leaf of `T'` that is not 2:1.
#   **Lemma 2.** If a violating leaf `x` is split, the only leaves that can newly
#   violate are the face neighbours of `x` one level coarser than `x`.
#   **Theorem.** With those seeds the queue closes at exactly `balance!`'s leaf
#   set (the least 2:1 refinement), *provided the tree is 2:1 at entry*;
#   without that precondition there is an explicit counterexample.
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
mutation that produced the seeds**:
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
re-merged (the S16 spin).
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
