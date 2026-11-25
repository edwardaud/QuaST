# Leaf-level AMR operations shared by the adaptive cycle and the driver.
#
# This file is what remains of `dgquad.jl` after the classic AMR policy and the
# DG prototype (DG Poisson/advection operators, the DG gradient-jump indicator
# and the fraction-based `amr_cycle!`) were removed at `8c622df`: the pair-strict
# refine/coarsen primitives and the hard leaf-budget pass.  Nothing here knows
# about a policy -- `amr_cycle_balanced!` (`amr_balanced.jl`) and
# `semquad_amr_passes!` (`driver/amr_hooks.jl`) are the only callers.

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


# ---------------------------------------------------------------------------
# Hard leaf budget (`semquad_enforce_leaf_budget!`).
# ---------------------------------------------------------------------------

"""
    LAST_BUDGET_PASS

Diagnostics of the last [`semquad_enforce_leaf_budget!`](@ref) call:
`(merged, scans, guard, reached, capped, splits, tests, local_capped,
entry_21, unsplittable)`.  `scans` is the number of full leaf scans the pass
needed (one per merge, so `scans == merged` unless a merge was undone by the
rebalance), `guard` the loop-iteration count, `reached` whether the tree ended
at or below `max_leaves`, and `capped` whether the `guard > 100000` safety cap
fired instead of a candidate search ending.  The pass visits each sibling
quartet once per scan (`seenp` below), which is the same candidate order as the
historical full scan -- the first leaf of a quartet in set order -- at a quarter
of the lookups.

The last four fields are the local rebalance (`rebalance_local!`): `splits` the
leaves the 2:1 rebalance split over the whole pass,
the structural test cap fired, `entry_21` whether the tree was already 2:1 at
pass entry (see `rebalance_local!`: the equivalence with the global `balance!`
is proven under that precondition, so a `false` here means the first rebalance
seeded every violator), and `unsplittable` how many violating leaves were at
`maxlevel` (no refinement can fix those; they are counted, not re-queued).
"""
const LAST_BUDGET_PASS = Ref((merged=0, scans=0, guard=0, reached=true,
                              capped=false, splits=0, tests=0,
                              local_capped=false, entry_21=true,
                              unsplittable=0))

@inline function _record_budget_pass(merged::Int, scans::Int, guard::Int,
                                     reached::Bool, capped::Bool;
                                     splits::Int=0, tests::Int=0,
                                     local_capped::Bool=false,
                                     entry_21::Bool=true,
                                     unsplittable::Int=0)
    LAST_BUDGET_PASS[] = (merged=merged, scans=scans, guard=guard,
                          reached=reached, capped=capped, splits=splits,
                          tests=tests, local_capped=local_capped,
                          entry_21=entry_21, unsplittable=unsplittable)
    return merged, reached
end

"""
    _rebalance_after_merge!(qt, st, parent, mparent, symmetry, entry21)

The budget pass's per-merge 2:1 step: a **local** rebalance seeded at the merged
parent.  `parent` and `mparent` are the merged parents (equal without `:x`), which
are exactly the leaves a merge can make non-2:1 (Lemma 1); `rebalance_local!`
adds their face neighbours and, when `entry21 == false` (the tree was not 2:1 at
pass entry), every other violator.  Returns `st`.
"""
@inline function _rebalance_after_merge!(qt::Quadtree, st::LocalBalance,
                                         parent::NTuple{3,Int},
                                         mparent::NTuple{3,Int},
                                         symmetry::Symbol, entry21::Bool)
    if symmetry === :x && mparent != parent
        rebalance_local!(qt, st; seeds=(parent, mparent), symmetry=:x,
                         all_violations=!entry21)
    else
        rebalance_local!(qt, st; seeds=(parent,), symmetry=symmetry,
                         all_violations=!entry21)
    end
    return st
end

"""
    semquad_enforce_leaf_budget!(qt, eta; max_leaves, protect, symmetry)

Enforce a hard leaf budget by merging the **lowest-error complete sibling
quartets** until the tree has at most `max_leaves` leaves (or no mergeable
quartet is left), re-balancing after every merge.  Returns `(merged, reached)`.

This is the `coarsen_frac == 0` enforcement path: the adaptive cycle merges only
the quartets its Dörfler cut selects, so a refine-only tree (or one whose
coarsening is blocked) can stay above the budget after 2:1 balancing.  The
candidate is the *parent* of a complete quartet, chosen by the summed indicator,
so the merge always makes progress; a quartet whose merge is undone by the 2:1
rebalance is protected and never retried.

The rebalance after each merge is **local** (`rebalance_local!`): the
merge is the only event that can create a 2:1 violation (Lemma 1), so the queue
starts at the new parent and its face neighbours instead of re-scanning the
10 000-leaf tree.  The result is the same leaf set the global `balance!` would
produce -- the pass re-establishes the 2:1 invariant after every merge, so the
next merge starts from a 2:1 tree (the theorem's precondition); a tree that
enters unbalanced (`entry_21 == false`, e.g. a legacy checkpoint) makes the
first rebalance seed every violator, which is the same closure again.

Every parent the rebalance splits goes into `blocked`, not just the merge's own:
a split re-forms the quartet one level down, and re-merging it would spin.

With `symmetry=:x` the merge is **atomic over the mirror pair**: the candidate is
only eligible when the mirror quartet is complete and unprotected too, and both
quartets are merged (or neither).  A pair whose mirror is protected is refused,
so the invariant survives; the documented cost is that the pass stops with
`reached == false` while the tree is up to one pair (6 leaves, less when the
refused pairs overlap) above `max_leaves` -- the old behaviour merged one side
and left 5 non-mirrored leaves per one-sided merge.

`eta` is updated in place: the merged parent gets the summed indicator of its
children, so it can be merged again at the next level.
"""
function semquad_enforce_leaf_budget!(qt::Quadtree, eta::Dict{NTuple{3,Int},Float64};
                                      max_leaves::Int=typemax(Int),
                                      protect::Set{NTuple{3,Int}}=Set{NTuple{3,Int}}(),
                                      symmetry::Symbol=:none)
    max_leaves >= 1 || throw(ArgumentError("max_leaves must be >= 1"))
    symmetry in (:none, :x) || throw(ArgumentError("symmetry must be :none or :x"))
    protect = symmetry === :x ? _mirror_protect(qt, protect) : protect
    merged = 0
    blocked = Set{NTuple{3,Int}}()
    # One entry per sibling quartet per scan: `_budget_quartet` reconstructs the
    # quartet from the parent, so all four children give the identical candidate
    # and only the first one in set order can ever win (`cand.eta < bestv` is
    # strict).  Skipping the other three is therefore the same candidate order
    # and the same values, at a quarter of the lookups.
    seenp = Set{NTuple{3,Int}}()
    guard = 0
    scans = 0
    # Local rebalance state (reused across merges) + the entry precondition.
    st = LOCAL_BALANCE
    entry21 = _tree_is_21(qt)
    splits_total = 0
    tests_total = 0
    while length(qt.leaves) > max_leaves
        guard += 1
        if guard > 100000
            return _record_budget_pass(merged, scans, guard, false, true;
                                       splits=splits_total, tests=tests_total,
                                       entry_21=entry21)
        end
        best = nothing
        bestv = Inf
        scans += 1
        empty!(seenp)
        for lf in qt.leaves
            p = (lf[1] - 1, lf[2] >> 1, lf[3] >> 1)
            p in seenp && continue
            push!(seenp, p)
            cand = _budget_quartet(qt, lf, eta, protect, blocked)
            cand === nothing && continue
            if symmetry === :x
                kid = cand.kids[1]
                ml = _mirror_leaf(qt, kid)
                if ml != kid
                    # merge both sides or neither
                    _budget_quartet(qt, ml, eta, protect, blocked) === nothing &&
                        continue
                end
            end
            if cand.eta < bestv
                bestv = cand.eta
                best = cand.kids
            end
        end
        best === nothing && return _record_budget_pass(merged, scans, guard,
                                                       false, false;
                                                       splits=splits_total,
                                                       tests=tests_total,
                                                       entry_21=entry21)
        n0 = length(qt.leaves)
        kid = best[1]
        parent = (kid[1] - 1, kid[2] >> 1, kid[3] >> 1)
        coarsen!(qt, kid) || return _record_budget_pass(merged, scans, guard,
                                                        false, false;
                                                        splits=splits_total,
                                                        tests=tests_total,
                                                        entry_21=entry21)
        eta[parent] = bestv
        merged += 1
        mparent = parent
        if symmetry === :x
            ml = _mirror_leaf(qt, kid)
            if ml != kid
                mkid = (ml[1], ml[2] - ml[2] % 2, ml[3] - ml[3] % 2)
                mparent = (ml[1] - 1, ml[2] >> 1, ml[3] >> 1)
                mcand = _budget_quartet(qt, mkid, eta, protect, blocked)
                if mcand !== nothing && coarsen!(qt, mkid)
                    eta[mparent] = mcand.eta
                    merged += 1
                end
            end
        end
        # Local 2:1 rebalance of the merge's neighbourhood (the seeds are the
        # new parent, its mirror half, and their face neighbours -- expanded by
        # `rebalance_local!`).  `entry21` is the theorem's precondition.
        _rebalance_after_merge!(qt, st, parent, mparent, symmetry, entry21)
        splits_total += st.nsplits
        tests_total += st.ntests
        # after a completed rebalance the tree is 2:1 again (the precondition of
        # the next merge's local pass); a capped one is not, so the next pass
        # re-seeds every violator.
        entry21 = !st.capped
        if length(qt.leaves) >= n0
            # The merge was undone by the rebalance: do not retry this quartet.
            # The four `(k[1], k[2] >> 1, k[3] >> 1)` entries are the historical
            # code's, verbatim: `k[1]` is the *child* level, so they are not the
            # merged parent (`parent` is pushed separately) -- but such a key is
            # a real quartet parent one level down, so they block the quartet
            # that a re-merge would form there.  They are load-bearing for the
            # bitwise trajectory; removing
            # them makes the pass merge more (see the 6/60 counterexample class
            # in the same section).  Every cascade parent the rebalance splits
            # has an incomplete quartet, so it needs no entry here (the linear
            # candidate scan cannot re-merge it).
            push!(blocked, parent)
            for k in best
                push!(blocked, (k[1], k[2] >> 1, k[3] >> 1))
            end
            push!(blocked, mparent)
        end
    end
    return _record_budget_pass(merged, scans, guard,
                               length(qt.leaves) <= max_leaves, false;
                               splits=splits_total, tests=tests_total,
                               local_capped=st.capped, entry_21=entry21,
                               unsplittable=st.unsplittable)
end

"""One budget candidate: the complete, unprotected, unblocked sibling quartet of
`lf`, with its summed indicator.  `nothing` when the quartet cannot be merged.

Allocation-free: the historical version built the 2x2 quartet as a
`Matrix{NTuple{3,Int}}` (a heap array per leaf per scan, 0.33 s/cycle on the
10k-leaf operating tree).  The four keys are returned in the column-major order
of that matrix (`kids[1,1], kids[2,1], kids[1,2], kids[2,2]`), so `kids[1]` is
the same child and the indicator sum accumulates in the same order."""
function _budget_quartet(qt::Quadtree, lf::NTuple{3,Int},
                         eta::Dict{NTuple{3,Int},Float64},
                         protect::Set{NTuple{3,Int}},
                         blocked::Set{NTuple{3,Int}})
    l, ix, iy = lf
    l > 0 || return nothing
    px, py = ix >> 1, iy >> 1
    parent = (l - 1, px, py)
    parent in blocked && return nothing
    bx = 2px
    by = 2py
    k1 = (l, bx, by)
    k2 = (l, bx, by + 1)
    k3 = (l, bx + 1, by)
    k4 = (l, bx + 1, by + 1)
    (k1 in qt.leaves && k2 in qt.leaves && k3 in qt.leaves && k4 in qt.leaves) ||
        return nothing
    (k1 in protect || k2 in protect || k3 in protect || k4 in protect) &&
        return nothing
    s = max(get(eta, k1, 0.0), 0.0)
    s += max(get(eta, k2, 0.0), 0.0)
    s += max(get(eta, k3, 0.0), 0.0)
    s += max(get(eta, k4, 0.0), 0.0)
    return (kids=(k1, k2, k3, k4), eta=s)
end

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
