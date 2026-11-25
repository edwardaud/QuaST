# Quartet geometry, the coarsening plan and the live-gated funding merge.
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
    MAX_PLAN_SWEEPS, MAX_PLAN_ROUNDS

Structural caps of the plan walk. Neither
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
was an `Any` load. `granted` is the caller's
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
   quartets cannot pay `need` leaves. Measured at the operating point 14 of the
  15 deadband quartets are
   geometrically illegal while the tree has 1635 legal complete quartets, so a
   gate without that extension would refuse the whole exchange and freeze the
   mesh.

Returns `(order, refused, extended, sweeps, rounds, capped)`: `order` the
representatives to merge (all legal together, in summed-indicator order),
`refused` the deadband quartets the 2:1 gate rejected, `extended` the granted
quartets taken from above `tau_h`, `sweeps`/`rounds` the fixed-point sweeps and
extension rounds the walk used, and `capped` whether it stopped on one of the two
structural caps instead of converging (a
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

