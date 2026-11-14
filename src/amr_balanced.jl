# The balanced AMR policy of docs/dev/AMR.md §3-§5.
#
# It is the only adaptive policy: `run_semquad_simulation` has used it since
# 2026-11-13 (docs/dev/AMR.md), and the `:classic` fraction + hard-budget path
# (formerly `amr_cycle!` in `dgquad.jl`) was removed at `8c622df`, so
# `semquad_amr_passes!` and the driver's `initial_refine` both call
# `amr_cycle_balanced!` unconditionally.
#
# Measured motivation (production run E, `out_julia/case_realistic_E`, step
# 18800: 9994/10000 leaves, 3720 AMR cycles of which 723 changed anything, last
# structural change at step 17225, delta_top thinning to 0.073 = h/delta 0.213):
#
#   (a) the absolute `coarsen_theta_tol = 1e-2` protects 9892 of 9994 leaves
#       (99.0 %); with the `symmetry=:x` mirror closure the protect set covers
#       9980 leaves and only **14** are coarsenable, so no sibling quartet can
#       ever be merged;
#   (b) the classic coarsen loop charges *every* walked leaf to its `cum_c`
#       budget, merged or not (the removed `amr_cycle!`), so the budget is
#       spent by refused leaves: measured 94.8 % by protected leaves, 5.2 % by
#       incomplete quartets, 0.0 % by mergeable ones;
#   (c) at 9994 leaves the hard budget guard `length + 10 > max_leaves` breaks
#       the refinement loop **before the first candidate** (0 of the 82
#       Dörfler refinements the policy asks for), and the driver additionally
#       pins `refine_frac = 0` at the cap;
#   (d) the 10 %-of-height wall-strip protect set is 2040 leaves (20.4 %) and
#       carries 60.6 % of the raw indicator.
#
# The `:balanced` path keeps 2:1 balance, the x-mirror invariant (G208) and the
# leaf budget, and replaces the four mechanisms:
#
#   (a') coarsening is *relative*: the Dörfler-from-below threshold
#        `tau_c = coarsen_frac * total` (cumulative, from below) is compared
#        against a **complete quartet's** `max(eta_child)`, never against a
#        single child's eta and never against an absolute tolerance;
#   (b') refused / protected quartets are not charged: a merge is only counted
#        when it happened;
#   (c') budget exchange: refinement is never blocked by `max_leaves`; the
#        lowest-sum eligible quartet is merged to fund three slots (six under
#        `:x`), and only when *no* eligible quartet is left does the refinement
#        loop stop, with `budget_limited = true`;
#   (d') the wall protect set is only the first/last leaf rows; the strip's
#        minimum level is guaranteed by `refine_wall_to_min!`, and the merge
#        side refuses a quartet at `level <= l_min` inside the strip (the
#        quarter that the wall rule would re-split on the same pass);
#   (e') hysteresis: a threshold deadband `tau_h = min(tau_c, 0.5*tau_r)` plus
#        an explicit cycle cooldown (`AmrCooldown`) so a quartet that was just
#        merged is not re-refined and a leaf that was just created by a split is
#        not immediately merged again.  Both are **our** choices: ASPECT has no
#        hysteresis (docs/dev/AMR.md §1.7) and no cell budget.
#
# The coarsen *supply* is demand-driven: `deficit = n0 + 6*#candidates -
# max_leaves` leaves have to disappear to fit the Dörfler refinement set, capped
# by the relative cell budget `coarsen_frac * n0`.  An unconditional
# "coarsen 5 % of the error every cycle" is a one-way coarsening, not a trade:
# measured on the E tree, one such pass merges 1112 quartets (9994 -> 6682
# leaves) and removes only 1.48 % of the indicator.

"""
    AmrCooldown()

Hysteresis state of `amr_cycle_balanced!`: two cooldown dicts, `no_refine`
(leaves produced by a merge; not refined again for a few cycles) and
`no_coarsen` (leaves produced by a split; not merged again for a few cycles),
plus `cycle`, the number of AMR cycles this state has been aged through.

`cycle` is *not* read by the policy: it is an identity/diagnostic counter that a
checkpoint can round-trip, so a restored cooldown can be shown to be the same
object (the decisions themselves depend only on the dicts).  A key is removed
when its counter reaches zero or when the leaf stops being a leaf, so the state
never refers to a stale mesh.  Pass the same object to every cycle of a run
(`SEMQUAD_AMR_COOLDOWN` is the driver's default); `save_semquad_state` persists
it as optional HDF5 state (docs/dev/BUG_LEDGER.md §10.5), and a checkpoint
without it loads an empty cooldown, which is the historical restart behaviour.
"""
mutable struct AmrCooldown
    no_refine::Dict{NTuple{3,Int},Int}
    no_coarsen::Dict{NTuple{3,Int},Int}
    cycle::Int
end
AmrCooldown() =
    AmrCooldown(Dict{NTuple{3,Int},Int}(), Dict{NTuple{3,Int},Int}(), 0)
# Backwards-compatible two-dict constructor (tests and probes built the object
# before `cycle` existed); the counter starts at zero.
AmrCooldown(no_refine::Dict{NTuple{3,Int},Int},
            no_coarsen::Dict{NTuple{3,Int},Int}) =
    AmrCooldown(no_refine, no_coarsen, 0)

"""
    _quartet_kids(lf)

The four children of the parent of the quartet `lf` belongs to (a 2x2 array
indexed `[dy+1, dx+1]`)."""
function _quartet_kids(lf::NTuple{3,Int})
    l = lf[1]
    bx = lf[2] - lf[2] % 2
    by = lf[3] - lf[3] % 2
    return [(l, bx + dx, by + dy) for dy in 0:1, dx in 0:1]
end

"Parent node of a leaf's quartet (`quadtree.jl`, where the local rebalance uses
it too; one definition for the module)."
# `_pq(lf) = (lf[1] - 1, lf[2] >> 1, lf[3] >> 1)` -- defined in quadtree.jl.

"`true` when `lf` is the `(dx=0, dy=0)` child of a quartet, i.e. the one whose
parent is `(lf[1]-1, lf[2]>>1, lf[3]>>1)` with `lf[2]` and `lf[3]` even: exactly
one leaf per sibling quartet, so a full-leaf scan can visit each quartet once."
@inline _quartet_anchor(lf::NTuple{3,Int}) = (lf[2] & 1) == 0 && (lf[3] & 1) == 0

"""
    _kids4(lf) -> NTuple{4,NTuple{3,Int}}

The four children of the parent of `lf`, as a **tuple** in the column-major order
of [`_quartet_kids`](@ref) (`kids[1,1], kids[2,1], kids[1,2], kids[2,2]`).  The
Matrix form allocates one array per call; the plan and the gate only need the
four keys, and every `all`/`sum`/`maximum` over the tuple visits them in the
same order as over the Matrix, so the arithmetic is identical.
"""
@inline function _kids4(lf::NTuple{3,Int})
    l = lf[1]
    bx = lf[2] - lf[2] % 2
    by = lf[3] - lf[3] % 2
    return ((l, bx, by), (l, bx, by + 1), (l, bx + 1, by), (l, bx + 1, by + 1))
end

"The two children on one external face of the parent box."
function _face_kids(kids::Matrix{NTuple{3,Int}}, dir::Symbol)
    dir === :xminus && return vec(kids[:, 1])
    dir === :xplus && return vec(kids[:, 2])
    dir === :yminus && return vec(kids[1, :])
    return vec(kids[2, :])
end

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

"""
    MAX_PLAN_SWEEPS, MAX_PLAN_ROUNDS

Structural caps of the plan walk (`docs/dev/AMR.md` §4).  Neither
is reached by a tree whose legal supply is at least `need` (measured on the
operating state: 2-4 sweeps and 1-2 rounds); they exist so a tree that cannot
supply the merge budget stops after a bounded amount of work and reports
`sweeps`/`rounds`/`capped` in [`LAST_AMR_BALANCED`](@ref) instead of walking the
whole leaf set again for every `target`-sized extension.
"""
const MAX_PLAN_SWEEPS = 64
const MAX_PLAN_ROUNDS = 16

"""
    _fund_live!(qt, asc, eta, protect, lmin, cooldown, will_refine, symmetry,
                tau_r, tau_h, granted) -> (merged_quartets, extended)

The live-gated funding merge of the `coarsen_gate` path: the plan is exhausted
but the refinement still does not fit, so scan the indicator order for a
complete quartet whose merge `balance!` will not undo (the already-applied merges
are in the tree, so no `granted` set is needed) and merge the first one found.

Top-level rather than a closure over the cycle: as a closure it captured (and
therefore boxed) `nc`, `gate_extended` and `tau_r`, so every read inside the scan
was an `Any` load (`docs/dev/AMR.md` §4).  `granted` is the caller's
reusable empty set.  Returns `(merged_quartets, extended, leaf)`, where `leaf` is
the quartet representative that was merged (the caller records it); `(0, false,
nothing)` when no legal quartet is left.
"""
function _fund_live!(qt::Quadtree, asc::Vector{NTuple{3,Int}},
                     eta::Dict{NTuple{3,Int},Float64},
                     protect::Set{NTuple{3,Int}}, lmin,
                     cooldown::Union{Nothing,AmrCooldown},
                     will_refine::Set{NTuple{3,Int}}, symmetry::Symbol,
                     tau_r::Float64, tau_h::Float64,
                     granted::Set{NTuple{3,Int}})
    for lf2 in asc
        lf2 in qt.leaves || continue
        lf2[1] > lmin(lf2) || continue
        (cooldown !== nothing && haskey(cooldown.no_coarsen, lf2)) && continue
        kids2 = _kids4(lf2)
        all(k -> k in qt.leaves, kids2) || continue
        any(k -> k in protect, kids2) && continue
        all(k -> k[1] > lmin(k), kids2) || continue
        maximum(k -> max(get(eta, k, 0.0), 0.0), kids2) < tau_r || continue
        _merge_allowed(qt, lf2, will_refine, granted, symmetry) || continue
        m2 = _coarsen_pair!(qt, lf2, symmetry, protect)
        m2 == 0 && continue
        return m2, maximum(k -> max(get(eta, k, 0.0), 0.0), kids2) > tau_h, lf2
    end
    return 0, false, nothing
end

"""
    _coarsen_plan(qt, eta, protect, lmin, cooldown, need, symmetry,
                  will_refine, tau_h, tau_r)

The coarsening plan of one `:balanced` cycle with `coarsen_gate=true`: a
`fix_coarsen_flags`-style selection (`tria.cc:14830-15032`) of the quartets to
merge, computed **before any mutation** so every decision sees the same tree.

1. **Candidates** are the complete, unprotected quartets at `level > l_min`, not
   in the `no_coarsen` cooldown, with **no child at or above the refine
   threshold** `tau_r` (ASPECT never coarsens a refine-flagged cell,
   `source/grid/grid_refinement.cc:199-211`).  They are ordered deadband first
   (`max(eta_child) <= tau_h`), then by summed indicator -- so the criterion's
   own choice is consumed first and the walk below only *extends* it.
2. **Grants** are decided finest level first (`tria.cc:15000-15005`), each with
   `_coarsening_allowed` against the grants already made, iterated to a fixed
   point: a coarser candidate that needed a finer quartet is retried after that
   quartet is granted (deal.II's outer flag loop, `:15023-15031`).  A granted
   quartet may only rely on decisions that are final, so a refusal is final for
   this level sweep.
3. **Supply**: the walk only extends past `tau_h` when the deadband's legal
   quartets cannot pay `need` leaves.  Measured (`docs/dev/AMR.md`
   §4.1): at the operating point 14 of the 15 deadband quartets are
   geometrically illegal while the tree has 1635 legal complete quartets, so a
   gate without that extension would refuse the whole exchange and freeze the
   mesh.

Returns `(order, refused, extended, sweeps, rounds, capped)`: `order` the
representatives to merge (all legal together, in summed-indicator order),
`refused` the deadband quartets the 2:1 gate rejected, `extended` the granted
quartets taken from above `tau_h`, `sweeps`/`rounds` the fixed-point sweeps and
extension rounds the walk used, and `capped` whether it stopped on one of the two
structural caps instead of converging (`docs/dev/AMR.md` §4: a
monotone measure plus a recorded cap, not an unbounded `while true`).

`asc` is the cycle's already-sorted leaf list (`(eta, leaf)` ascending); passing
it saves a second full sort of the leaf set per cycle and is the identical order.

The fixed point is `granted`: a quartet is granted only once, and every grant is
a member of `granted` that was not there before, so the inner sweep terminates
after at most `n` grants (`MAX_PLAN_SWEEPS` bounds it structurally).  The outer
round only grows `limit` towards `n`; `MAX_PLAN_ROUNDS` bounds the *walk* so a
tree whose legal supply is smaller than `need` cannot spend
`O(n^2/target)` lookups looking for it.
"""
function _coarsen_plan(qt::Quadtree, eta::Dict{NTuple{3,Int},Float64},
                       protect::Set{NTuple{3,Int}}, lmin,
                       cooldown::Union{Nothing,AmrCooldown}, need::Int,
                       symmetry::Symbol, will_refine::Set{NTuple{3,Int}},
                       tau_h::Float64, tau_r::Float64;
                       asc::Union{Nothing,Vector{NTuple{3,Int}}}=nothing)
    plan = NTuple{3,Int}[]
    refused = 0
    extended = 0
    sweeps = 0
    rounds = 0
    capped = false
    need > 0 || return (order=plan, refused=refused, extended=extended,
                        sweeps=sweeps, rounds=rounds, capped=capped)
    leaf = NTuple{3,Int}[]
    qsum = Float64[]
    maxe = Float64[]
    seen = Set{NTuple{3,Int}}()
    sorted = asc === nothing ?
             sort(collect(qt.leaves), by=lf -> (max(get(eta, lf, 0.0), 0.0), lf)) :
             asc
    for lf in sorted
        lf[1] > lmin(lf) || continue
        p = _pq(lf)
        p in seen && continue
        kids = _kids4(lf)
        all(k -> k in qt.leaves, kids) || continue
        any(k -> k in protect, kids) && continue
        all(k -> k[1] > lmin(k), kids) || continue
        if cooldown !== nothing
            any(k -> haskey(cooldown.no_coarsen, k), kids) && continue
        end
        m = maximum(k -> max(get(eta, k, 0.0), 0.0), kids)
        m < tau_r || continue          # never coarsen a refine-flagged cell
        if symmetry === :x
            ml = _mirror_leaf(qt, lf)
            if ml != lf
                mk = _kids4(ml)
                all(k -> k in qt.leaves && !(k in protect) &&
                         k[1] > lmin(k), mk) || continue
                maximum(k -> max(get(eta, k, 0.0), 0.0), mk) < tau_r || continue
            end
        end
        push!(seen, p)
        push!(leaf, lf)
        push!(qsum, sum(k -> max(get(eta, k, 0.0), 0.0), kids))
        push!(maxe, m)
    end
    n = length(leaf)
    n > 0 || return (order=plan, refused=refused, extended=extended,
                     sweeps=sweeps, rounds=rounds, capped=capped)
    # deadband first, then summed indicator: the criterion's own choice first
    perm = sortperm(1:n, by=i -> (maxe[i] > tau_h ? 1 : 0, qsum[i], leaf[i]))
    granted = Set{NTuple{3,Int}}()
    per = symmetry === :x ? 6 : 3
    target = max(1, cld(need, per))
    limit = min(n, target + 8)
    while true
        rounds += 1
        changed = true
        # The sweep order depends on `limit` only -- `perm`, `qsum` and `leaf`
        # are fixed while `granted` grows -- so it is computed once per round
        # instead of once per fixed-point sweep (same permutation: `sortperm` is
        # a pure function of its input).
        sweep = sortperm(1:limit,
                         by=i -> (-leaf[perm[i]][1], qsum[perm[i]],
                                  leaf[perm[i]]))
        while changed
            changed = false
            sweeps += 1
            # finest level first, then the criterion order (tria.cc:15000-15005)
            for i in sweep
                j = perm[i]
                lf = leaf[j]
                p = _pq(lf)
                p in granted && continue
                _merge_allowed(qt, lf, will_refine, granted, symmetry) || continue
                push!(granted, p)
                if symmetry === :x
                    ml = _mirror_leaf(qt, lf)
                    ml != lf && push!(granted, _pq(ml))
                end
                push!(plan, lf)
                changed = true
            end
            if sweeps >= MAX_PLAN_SWEEPS
                capped = true
                break
            end
        end
        capped && break
        per * length(plan) >= need && break
        limit >= n && break
        if rounds >= MAX_PLAN_ROUNDS
            capped = true
            break
        end
        limit = min(n, limit + max(target, 8))
    end
    sort!(plan, by=lf -> (max(get(eta, lf, 0.0), 0.0), lf))
    for i in 1:limit
        maxe[perm[i]] > tau_h && continue      # only deadband quartets count
        _pq(leaf[perm[i]]) in granted && continue
        refused += 1
    end
    for lf in plan
        maximum(k -> max(get(eta, k, 0.0), 0.0), _kids4(lf)) > tau_h &&
            (extended += 1)
    end
    return (order=plan, refused=refused, extended=extended,
            sweeps=sweeps, rounds=rounds, capped=capped)
end

"""Age the two cooldown dicts by one cycle and drop dead/expired keys.

A value of `n` means "suppressed for the next `n` cycles": the decrement happens
at the *start* of a cycle and a key is dropped when it goes negative, so an
entry stored with `hysteresis_cycles=1` in cycle `k` still suppresses the
operation in cycle `k+1` and expires at the start of cycle `k+2`.

The age is counted in `c.cycle` (incremented here, once per cycle), so a
checkpoint taken after cycle `k` restores a state whose next cycle is `k+1`.
"""
function _age_cooldown!(c::AmrCooldown, leaves::Set{NTuple{3,Int}})
    c.cycle += 1
    for d in (c.no_refine, c.no_coarsen)
        for k in collect(keys(d))
            if !(k in leaves)
                delete!(d, k)                 # the leaf is gone
            else
                d[k] -= 1
                d[k] < 0 && delete!(d, k)     # the cooldown expired
            end
        end
    end
    return c
end

"""Default hysteresis state of the production driver (`run_semquad_simulation`)."""
const SEMQUAD_AMR_COOLDOWN = Ref(AmrCooldown())

"""
Diagnostics of the last `amr_cycle_balanced!` cycle (named tuple):
`(refined, coarsened, exchanged, n, want, eligible, dropped, tau_r, tau_c,
tau_h, deadband, budget_limited, budget_infeasible, merged, recirc, post)`.
`exchanged` counts the leaves freed by the budget exchange, `want` the leaves
the Dörfler refinement set would need, `eligible` the complete quartets that
passed every gate, `budget_limited` that refinement stopped because no eligible
quartet was left, `budget_infeasible` that the tree is still above `max_leaves`
after the cycle (the caller's hard budget pass then merges the lowest-error
quartets).

`merged` is the number of quartets this cycle merged, `recirc` how many of those
parents the post-merge `balance!`/`mirror_close!` split again (**the
recirculation counter**; docs/dev/AMR.md §4.1), `post` the leaves the
post-merge balancing added, and `parents` the parent nodes themselves (one entry
per merged quartet, both halves under `symmetry=:x`), so a probe can attribute
the recirculation.  `nmerge`/`npost` are the leaf counts after the coarsening
pass and after the refinement pass, i.e. before the post-merge balancing, so
`npost - nmerge == 3*refined - 3*(merges done by the exchange)` is checkable.
`recirc == 0` means every merge survived the balance exactly as merged; with
`coarsen_gate=true` the gate promises it.

Loop/guard counters (docs/dev/AMR.md §4, docs/dev/AMR.md):
`plan_sweeps`/`plan_rounds` the fixed-point sweeps and extension rounds the
coarsening plan used, `plan_capped` whether it stopped on `MAX_PLAN_SWEEPS`/
`MAX_PLAN_ROUNDS` instead of converging, `fund_calls`/`fund_hits` how often the
live-gated funding merge was invoked and how often it found a legal quartet,
`balance_ok`/`mirror_ok` the success flags of the post-merge passes (both used to
be ignored), and `mirror_rounds` the passes `mirror_close!` needed.
"""
const LAST_AMR_BALANCED = Ref((refined=0, coarsened=0, exchanged=0, n=0, want=0,
                              eligible=0, dropped=0, tau_r=0.0, tau_c=0.0,
                              tau_h=0.0, deadband=false, budget_limited=false,
                              budget_infeasible=false, merged=0, recirc=0,
                              post=0, parents=NTuple{3,Int}[], nmerge=0,
                              npost=0, gate_refused=0, gate_extended=0,
                              plan_sweeps=0, plan_rounds=0, plan_capped=false,
                              fund_calls=0, fund_hits=0, balance_ok=true,
                              mirror_ok=true, mirror_rounds=0))

"""
    amr_cycle_balanced!(qt, eta; refine_frac=0.2, coarsen_frac=0.05,
                        hysteresis=0.5, max_level=typemax(Int),
                        max_leaves=typemax(Int), symmetry=:none,
                        protect=Set(), wall_min_level=0, wall_strip=0.1,
                        cooldown=nothing, hysteresis_cycles=1,
                        coarsen_cells=nothing, coarsen_gate=true)

One AMR cycle of the balanced policy (docs/dev/AMR.md §3-§5), the only adaptive
policy.  Returns `(refined, coarsened, exchanged)` in leaves, and records the
cycle's threshold/limit diagnostics in [`LAST_AMR_BALANCED`](@ref).

The thresholds are cumulative error fractions, computed exactly by sorting
(deal.II bisects the same quantity, docs/dev/AMR.md §1.3):

    tau_r = eta value at the refine_frac * total point from above
    tau_c = eta value at the coarsen_frac * total point from below
    tau_h = min(tau_c, hysteresis * tau_r)          # coarsen gate (deadband)

A leaf is refined when `eta >= tau_r`, `level < max_level` and it is not in the
`no_refine` cooldown.  A quartet is merged only when all four children are
leaves, none is protected, every child has `level > l_min` (`l_min` is
`wall_min_level` inside `wall_strip`, 0 elsewhere), `max(eta_child) <= tau_h`,
and the mirror quartet (`symmetry=:x`) passes the same test.  `coarsen_cells`
caps the leaves dropped per cycle (default `round(coarsen_frac * n)`).

`coarsen_gate` (default `true`) adds deal.II's `coarsening_allowed()` pre-check
(`docs/dev/AMR.md`, `docs/dev/AMR.md` §4): a
quartet is merged only when its parent's external faces keep the 2:1 rule after
the whole pass, counting the rest of the merge plan (`-1` level) and the refine
plan (`+1`).  The post-merge `balance!` then has nothing to split in the merged
region (`LAST_AMR_BALANCED[].recirc == 0`).  Because the deadband's lowest-error
quartets are usually the geometrically illegal ones (`docs/dev/AMR.md` §4.1
measured 14 of 15),
the plan extends the supply past `tau_h` -- but never past `tau_r`, so a
refine-flagged cell is still never coarsened -- and reports how many merges were
taken that way (`extended`).  `coarsen_gate=false` is the historical behaviour.
"""
function amr_cycle_balanced!(qt::Quadtree, eta::Dict{NTuple{3,Int},Float64};
                             refine_frac::Float64=0.2,
                             coarsen_frac::Float64=0.05,
                             hysteresis::Float64=0.5,
                             max_level::Int=typemax(Int),
                             max_leaves::Int=typemax(Int),
                             symmetry::Symbol=:none,
                             protect::Set{NTuple{3,Int}}=Set{NTuple{3,Int}}(),
                             wall_min_level::Int=0,
                             wall_strip::Float64=0.1,
                             cooldown::Union{Nothing,AmrCooldown}=nothing,
                             hysteresis_cycles::Int=1,
                             coarsen_cells::Union{Nothing,Int}=nothing,
                             coarsen_gate::Bool=true)
    symmetry in (:none, :x) ||
        throw(ArgumentError("symmetry must be :none or :x"))
    max_level >= 1 || throw(ArgumentError("max_level must be >= 1"))
    max_leaves >= 1 || throw(ArgumentError("max_leaves must be >= 1"))
    0.0 <= refine_frac <= 1.0 ||
        throw(ArgumentError("refine_frac must be in [0, 1]"))
    0.0 <= coarsen_frac <= 1.0 ||
        throw(ArgumentError("coarsen_frac must be in [0, 1]"))
    0.0 <= hysteresis <= 1.0 ||
        throw(ArgumentError("hysteresis must be in [0, 1]"))
    hysteresis_cycles >= 0 ||
        throw(ArgumentError("hysteresis_cycles must be >= 0"))
    wall_min_level >= 0 ||
        throw(ArgumentError("wall_min_level must be >= 0"))
    0.0 < wall_strip < 0.5 ||
        throw(ArgumentError("wall_strip must be in (0, 0.5)"))
    protect = symmetry === :x ? _mirror_protect(qt, protect) : protect
    cool = hysteresis_cycles > 0 ? cooldown : nothing
    cool === nothing || _age_cooldown!(cool, qt.leaves)
    n0 = length(qt.leaves)
    n0 > 0 || return 0, 0, 0
    total = 0.0
    for lf in qt.leaves
        total += max(get(eta, lf, 0.0), 0.0)
    end
    total > 0.0 || return 0, 0, 0
    per = symmetry === :x ? 6 : 3          # leaves added by one pair refinement

    eof(lf) = max(get(eta, lf, 0.0), 0.0)

    # --- minimum-refinement function `l_min(x)`: the wall strip floor ---------
    in_strip(lf) = begin
        lf[3] == 0 && return true
        lf[3] == qt.ny0 * (1 << lf[1]) - 1 && return true
        zc = (lf[3] + 0.5) * leaf_hy(qt, lf[1])
        return zc <= wall_strip || zc >= 1.0 - wall_strip
    end
    lmin(lf) = (wall_min_level > 0 && in_strip(lf)) ? wall_min_level : 0

    # --- (a') the two cumulative-error thresholds (exact, by sorting) --------
    asc = sort(collect(qt.leaves), by=lf -> (eof(lf), lf))
    desc = reverse(asc)     # `(eta, leaf)` descending == `(-eta, leaf)` ascending
    tau_r = 0.0
    if refine_frac > 0.0
        tgt_r = refine_frac * total
        cum = 0.0
        for lf in desc
            tau_r = eof(lf)
            cum += tau_r
            cum >= tgt_r && break
        end
    else
        tau_r = Inf
    end
    tau_c = 0.0
    if coarsen_frac > 0.0
        tgt_c = coarsen_frac * total
        cum = 0.0
        for lf in asc
            tau_c = eof(lf)
            cum += tau_c
            cum >= tgt_c && break
        end
    else
        tau_c = -Inf
    end
    tau_h = min(tau_c, hysteresis * tau_r)
    deadband = hysteresis * tau_r < tau_c

    # --- (c') the refinement demand (Dörfler from above) --------------------
    cand = NTuple{3,Int}[]
    want = 0
    for lf in desc
        lf[1] < max_level || continue
        eof(lf) >= tau_r || break        # descending: the rest is below tau_r
        (cool !== nothing && haskey(cool.no_refine, lf)) && continue
        push!(cand, lf)
        want += per
    end

    # --- the eligible quartets, lowest summed indicator first ---------------
    # `coarsen_frac` also caps the cell fraction: the whole pass may not drop
    # more than that many leaves (relative, like every other threshold here).
    capcells = coarsen_cells === nothing ? round(Int, coarsen_frac * n0) :
                                           coarsen_cells
    capcells = max(capcells, 0)
    deficit = max(0, n0 + want - max_leaves)
    need = min(deficit, capcells)
    # The refinement plan the loop below will actually split (the Dörfler prefix
    # up to `refine_frac * total`), needed by the coarsening gate: deal.II's
    # refine flags are an input to `coarsening_allowed`.
    will_refine = Set{NTuple{3,Int}}()
    if coarsen_gate
        cumw = 0.0
        for lf in cand
            cumw >= refine_frac * total && break
            push!(will_refine, lf)
            if symmetry === :x
                ml = _mirror_leaf(qt, lf)
                ml != lf && push!(will_refine, ml)
            end
            cumw += eof(lf)
            if symmetry === :x
                ml = _mirror_leaf(qt, lf)
                ml != lf && (cumw += eof(ml))
            end
        end
    end
    seen = Set{NTuple{3,Int}}()
    qsum = Float64[]
    qkid = NTuple{3,Int}[]
    gate_refused = 0
    gate_extended = 0
    plan_sweeps = 0
    plan_rounds = 0
    plan_capped = false
    if coarsen_gate
        # deal.II `fix_coarsen_flags` semantics: plan the merge set before any
        # mutation, 2:1-legality against the plan's own grants and the refine
        # plan; extend past the deadband only when its legal quartets cannot pay.
        plan = _coarsen_plan(qt, eta, protect, lmin, cool, need, symmetry,
                             will_refine, tau_h, tau_r; asc=asc)
        qkid = plan.order
        gate_refused = plan.refused
        plan_sweeps = plan.sweeps
        plan_rounds = plan.rounds
        plan_capped = plan.capped
        gate_extended = plan.extended
    elseif tau_h > -Inf
        for lf in asc
            lf[1] > lmin(lf) || continue              # level > l_min
            parent = (lf[1] - 1, lf[2] >> 1, lf[3] >> 1)
            parent in seen && continue
            bx = lf[2] - lf[2] % 2; by = lf[3] - lf[3] % 2
            kids = ((lf[1], bx + dx, by + dy) for dy in 0:1 for dx in 0:1)
            kids = collect(kids)
            all(k -> k in qt.leaves, kids) || continue
            any(k -> k in protect, kids) && continue
            all(k -> k[1] > lmin(k), kids) || continue
            m = maximum(eof, kids)
            m <= tau_h || continue
            if symmetry === :x
                ml = _mirror_leaf(qt, lf)
                if ml != lf
                    mbx = ml[2] - ml[2] % 2; mby = ml[3] - ml[3] % 2
                    mk = [(ml[1], mbx + dx, mby + dy) for dy in 0:1, dx in 0:1]
                    all(k -> k in qt.leaves && !(k in protect) &&
                             k[1] > lmin(k), mk) || continue
                    maximum(eof, mk) <= tau_h || continue
                end
            end
            push!(seen, parent)
            push!(qsum, sum(eof, kids))
            push!(qkid, lf)
        end
        perm = sortperm(qsum)
        qsum = qsum[perm]
        qkid = qkid[perm]
    end

    nc = 0
    nex = 0
    dropped = 0
    cursor = 1
    mergedp = NTuple{3,Int}[]      # parents this cycle created by a merge
    note_merged(lf) = begin
        p = (lf[1] - 1, lf[2] >> 1, lf[3] >> 1)
        push!(mergedp, p)
        cool === nothing && return nothing
        cool.no_refine[p] = hysteresis_cycles
        if symmetry === :x
            # `_coarsen_pair!` merges the mirror quartet too: its parent has to
            # be cooled as well, or the mirror half of the pair is re-refined on
            # the next cycle (measured: 12 of 20 merged parents, G210).
            pm = _mirror_leaf(qt, p)
            if pm != p
                cool.no_refine[pm] = hysteresis_cycles
                push!(mergedp, pm)
            end
        end
        return nothing
    end
    # --- (1) the coarsening pass: pay for the refinement in advance ----------
    # With `coarsen_gate=true` the whole plan is executed: its grants may rely on
    # each other (deal.II's flag set is executed as a whole), so stopping in the
    # middle could leave a granted quartet facing a neighbour whose merge did not
    # happen.  The plan is sized to `need`, so the over-shoot is at most one batch.
    while cursor <= length(qkid) && (coarsen_gate || dropped < need)
        lf = qkid[cursor]
        cursor += 1
        lf in qt.leaves || continue
        m = _coarsen_pair!(qt, lf, symmetry, protect)
        m == 0 && continue
        nc += m
        dropped += 3 * m
        note_merged(lf)
    end

    # Live-gated funding merge for the gate path: the plan is exhausted but the
    # refinement still does not fit (`deficit > capcells`).  `_fund_live!` scans
    # the indicator order for a complete quartet whose merge `balance!` will not
    # undo; the already-applied merges are in the tree, so the `granted` set it
    # sees is empty (one reusable set for the whole cycle).
    empty_granted = Set{NTuple{3,Int}}()
    fund_calls = 0
    fund_hits = 0

    # --- (2) the refinement pass, never blocked by the budget ---------------
    # `(c')` A refinement that does not fit is funded by merging the next
    # lowest-sum eligible quartet (3 slots, 6 under `:x`); the loop stops only
    # when no eligible quartet is left (`budget_limited`).
    nmerge = length(qt.leaves)
    nr = 0
    cum = 0.0
    target = refine_frac * total
    limited = false
    for lf in cand
        cum >= target && break
        lf in qt.leaves || continue
        lf[1] < max_level || continue
        (cool !== nothing && haskey(cool.no_refine, lf)) && continue
        while length(qt.leaves) + per > max_leaves
            progressed = false
            while cursor <= length(qkid)
                c = qkid[cursor]
                cursor += 1
                c in qt.leaves || continue
                m = _coarsen_pair!(qt, c, symmetry, protect)
                m == 0 && continue
                nc += m
                nex += 3 * m
                note_merged(c)
                progressed = true
                break
            end
            if !progressed && coarsen_gate
                fund_calls += 1
                m, ext, lff = _fund_live!(qt, asc, eta, protect, lmin, cool,
                                          will_refine, symmetry, tau_r, tau_h,
                                          empty_granted)
                if m > 0
                    nc += m
                    nex += 3 * m
                    ext && (gate_extended += m)
                    note_merged(lff)
                    progressed = true
                    fund_hits += 1
                end
            end
            if !progressed
                limited = true
                break
            end
        end
        limited && break
        n = _refine_pair!(qt, lf, max_level, symmetry)
        n == 0 && continue
        nr += n
        cum += eof(lf)
        if symmetry === :x
            ml = _mirror_leaf(qt, lf)
            ml != lf && (cum += eof(ml))
        end
        if cool !== nothing
            for dy in 0:1, dx in 0:1
                cool.no_coarsen[(lf[1] + 1, 2 * lf[2] + dx, 2 * lf[3] + dy)] =
                    hysteresis_cycles
            end
        end
    end

    n_post = length(qt.leaves)     # leaves before the post-merge balancing
    balance_ok = balance!(qt; symmetry=symmetry)
    mirror_rounds = 0
    mirror_ok = true
    if symmetry === :x
        _, mirror_ok = mirror_close!(qt; max_level=max_level)
        mirror_rounds = LAST_MIRROR_CLOSE[].rounds
        balance_ok = balance!(qt; symmetry=:x) && balance_ok
    end
    recirc = 0
    for p in mergedp
        p in qt.leaves || (recirc += 1)
    end
    LAST_AMR_BALANCED[] = (refined=nr, coarsened=nc, exchanged=nex,
                           n=length(qt.leaves), want=want,
                           eligible=length(qkid), dropped=dropped,
                           tau_r=tau_r, tau_c=tau_c, tau_h=tau_h,
                           deadband=deadband, budget_limited=limited,
                           budget_infeasible=length(qt.leaves) > max_leaves,
                           merged=length(mergedp), recirc=recirc,
                           post=length(qt.leaves) - n_post, parents=mergedp,
                           nmerge=nmerge, npost=n_post,
                           gate_refused=gate_refused,
                           gate_extended=gate_extended,
                           plan_sweeps=plan_sweeps, plan_rounds=plan_rounds,
                           plan_capped=plan_capped,
                           fund_calls=fund_calls, fund_hits=fund_hits,
                           balance_ok=balance_ok, mirror_ok=mirror_ok,
                           mirror_rounds=mirror_rounds)
    return nr, nc, nex
end
