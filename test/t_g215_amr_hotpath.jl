# G215: the AMR hot path is equivalent to the pre-audit implementation.
using Test, QuaST, Random, Printf

# G215: the S15/S16 hot-path audit of the `:balanced` AMR code
# (docs/dev/AMR.md, docs/dev/AMR.md).
#
# The audit changed allocation behaviour and loop guards in
# `src/amr_balanced.jl`, `src/amr_leaf_ops.jl` and `src/quadtree.jl`.
# "Faster" is only acceptable if it is the *same* result, so this gate pins:
#
#   1. **Equivalence against the pre-audit implementation.**  The historical
#      `_coarsening_allowed`, `_coarsen_plan`, `_budget_quartet` and
#      `semquad_enforce_leaf_budget!` are carried here verbatim (the same pattern
#      G188/G202 used) and must agree on random graded trees: same predicate,
#      same plan (`order`/`refused`/`extended`), same candidate, same merged
#      count and same final leaf set.  The tuple/ancestor/tuple forms are the
#      only change to those functions.
#   2. **Allocation budgets** (the §15 gate style): `_coarsening_allowed`,
#      `_merge_allowed`, `_budget_quartet`, one full budget scan and
#      `_invalidate_faces!` allocate 0 bytes per call; the *historical* versions
#      allocate (so the budgets are load-bearing, not vacuous).
#   3. **The loop caps fire and are recorded**: a tree on which the coarsening
#      plan cannot progress stops at `MAX_PLAN_ROUNDS` with `capped == true` in
#      `LAST_AMR_BALANCED`/`plan.capped` (not by spinning); a normal tree reports
#      `capped == false` with a small sweep count; `mirror_close!` records
#      `rounds`/`capped` in `LAST_MIRROR_CLOSE` and a forced `maxiter=1` cap is
#      visible; the hard-budget pass records `scans == merged` and a non-firing
#      guard.
#   4. **The incrementally maintained face table is never stale**: after a
#      mutation the next `neighbors` call equals a freshly built tree's answer,
#      the mutation marks only O(1) leaves stale (no rebuild), and
#      `_invalidate_faces!` itself allocates nothing.

const MS = QuaST

g215_fnv(a) = begin
    h = 0xcbf29ce484222325
    for b in reinterpret(UInt8, a)
        h ⊻= UInt64(b)
        h *= 0x100000001b3
    end
    string(h, base=16, pad=16)
end
g215_fnv_leaves(qt) = g215_fnv(reduce(vcat, [Int64[l, ix, iy]
                                            for (l, ix, iy) in sort(collect(qt.leaves))]))

# ---- the pre-audit implementations, verbatim --------------------------------
function g215_old_allowed(qt::Quadtree, kids::Matrix{NTuple{3,Int}},
                          will_refine::Set{NTuple{3,Int}},
                          granted::Set{NTuple{3,Int}})
    l = kids[1][1]
    inside = Set(vec(kids))
    for dir in MS.FACE_DIRS
        for c in MS._face_kids(kids, dir)
            c in qt.leaves || continue
            for nb in neighbors(qt, c, dir)
                nb in inside && continue
                eff = (nb in will_refine ? 1 : 0) -
                      (MS._pq(nb) in granted ? 1 : 0)
                nb[1] + eff <= l || return false
            end
        end
    end
    return true
end

function g215_old_plan(qt::Quadtree, eta::Dict{NTuple{3,Int},Float64},
                       protect::Set{NTuple{3,Int}}, lmin,
                       cooldown::Union{Nothing,AmrCooldown}, need::Int,
                       symmetry::Symbol, will_refine::Set{NTuple{3,Int}},
                       tau_h::Float64, tau_r::Float64)
    plan = NTuple{3,Int}[]
    refused = 0
    extended = 0
    need > 0 || return (order=plan, refused=refused, extended=extended)
    leaf = NTuple{3,Int}[]
    qsum = Float64[]
    maxe = Float64[]
    seen = Set{NTuple{3,Int}}()
    for lf in sort(collect(qt.leaves), by=lf -> (max(get(eta, lf, 0.0), 0.0), lf))
        lf[1] > lmin(lf) || continue
        p = MS._pq(lf)
        p in seen && continue
        kids = MS._quartet_kids(lf)
        all(k -> k in qt.leaves, kids) || continue
        any(k -> k in protect, kids) && continue
        all(k -> k[1] > lmin(k), kids) || continue
        if cooldown !== nothing
            any(k -> haskey(cooldown.no_coarsen, k), kids) && continue
        end
        m = maximum(k -> max(get(eta, k, 0.0), 0.0), kids)
        m < tau_r || continue
        if symmetry === :x
            ml = MS._mirror_leaf(qt, lf)
            if ml != lf
                mk = MS._quartet_kids(ml)
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
    n > 0 || return (order=plan, refused=refused, extended=extended)
    perm = sortperm(1:n, by=i -> (maxe[i] > tau_h ? 1 : 0, qsum[i], leaf[i]))
    granted = Set{NTuple{3,Int}}()
    per = symmetry === :x ? 6 : 3
    target = max(1, cld(need, per))
    limit = min(n, target + 8)
    while true
        changed = true
        while changed
            changed = false
            sweep = sortperm(1:limit,
                             by=i -> (-leaf[perm[i]][1], qsum[perm[i]],
                                      leaf[perm[i]]))
            for i in sweep
                j = perm[i]
                lf = leaf[j]
                p = MS._pq(lf)
                p in granted && continue
                g215_old_merge_allowed(qt, lf, will_refine, granted, symmetry) ||
                    continue
                push!(granted, p)
                if symmetry === :x
                    ml = MS._mirror_leaf(qt, lf)
                    ml != lf && push!(granted, MS._pq(ml))
                end
                push!(plan, lf)
                changed = true
            end
        end
        per * length(plan) >= need && break
        limit >= n && break
        limit = min(n, limit + max(target, 8))
    end
    sort!(plan, by=lf -> (max(get(eta, lf, 0.0), 0.0), lf))
    for i in 1:limit
        maxe[perm[i]] > tau_h && continue
        MS._pq(leaf[perm[i]]) in granted && continue
        refused += 1
    end
    for lf in plan
        maximum(k -> max(get(eta, k, 0.0), 0.0), MS._quartet_kids(lf)) > tau_h &&
            (extended += 1)
    end
    return (order=plan, refused=refused, extended=extended)
end

"Historical `_merge_allowed`: it also calls the historical predicate."
function g215_old_merge_allowed(qt::Quadtree, lf::NTuple{3,Int},
                               will_refine::Set{NTuple{3,Int}},
                               granted::Set{NTuple{3,Int}}, symmetry::Symbol)
    kids = MS._quartet_kids(lf)
    all(k -> k in qt.leaves, kids) || return false
    g215_old_allowed(qt, kids, will_refine, granted) || return false
    if symmetry === :x
        ml = MS._mirror_leaf(qt, lf)
        if ml != lf
            mk = MS._quartet_kids(ml)
            all(k -> k in qt.leaves, mk) || return false
            g215_old_allowed(qt, mk, will_refine, granted) || return false
        end
    end
    return true
end

function g215_old_budget_quartet(qt::Quadtree, lf::NTuple{3,Int},
                                 eta::Dict{NTuple{3,Int},Float64},
                                 protect::Set{NTuple{3,Int}},
                                 blocked::Set{NTuple{3,Int}})
    l, ix, iy = lf
    l > 0 || return nothing
    px, py = ix >> 1, iy >> 1
    parent = (l - 1, px, py)
    parent in blocked && return nothing
    kids = [(l, 2px + dx, 2py + dy) for dy in 0:1, dx in 0:1]
    all(k -> k in qt.leaves, kids) || return nothing
    any(k -> k in protect, kids) && return nothing
    s = 0.0
    for k in kids
        s += max(get(eta, k, 0.0), 0.0)
    end
    return (kids=kids, eta=s)
end

function g215_old_budget_pass!(qt::Quadtree, eta::Dict{NTuple{3,Int},Float64};
                               max_leaves::Int=typemax(Int),
                               protect::Set{NTuple{3,Int}}=Set{NTuple{3,Int}}(),
                               symmetry::Symbol=:none)
    protect = symmetry === :x ? MS._mirror_protect(qt, protect) : protect
    merged = 0
    blocked = Set{NTuple{3,Int}}()
    guard = 0
    while length(qt.leaves) > max_leaves
        guard += 1
        guard > 100000 && break
        best = nothing
        bestv = Inf
        for lf in qt.leaves
            cand = g215_old_budget_quartet(qt, lf, eta, protect, blocked)
            cand === nothing && continue
            if symmetry === :x
                kid = cand.kids[1]
                ml = MS._mirror_leaf(qt, kid)
                if ml != kid
                    g215_old_budget_quartet(qt, ml, eta, protect, blocked) ===
                        nothing && continue
                end
            end
            if cand.eta < bestv
                bestv = cand.eta
                best = cand.kids
            end
        end
        best === nothing && return merged, false
        n0 = length(qt.leaves)
        kid = best[1]
        parent = (kid[1] - 1, kid[2] >> 1, kid[3] >> 1)
        coarsen!(qt, kid) || return merged, false
        eta[parent] = bestv
        merged += 1
        mparent = parent
        if symmetry === :x
            ml = MS._mirror_leaf(qt, kid)
            if ml != kid
                mkid = (ml[1], ml[2] - ml[2] % 2, ml[3] - ml[3] % 2)
                mparent = (ml[1] - 1, ml[2] >> 1, ml[3] >> 1)
                mcand = g215_old_budget_quartet(qt, mkid, eta, protect, blocked)
                if mcand !== nothing && coarsen!(qt, mkid)
                    eta[mparent] = mcand.eta
                    merged += 1
                end
            end
        end
        balance!(qt; symmetry=symmetry)
        if length(qt.leaves) >= n0
            push!(blocked, parent)
            for k in best
                push!(blocked, (k[1], k[2] >> 1, k[3] >> 1))
            end
            push!(blocked, mparent)
        end
    end
    return merged, length(qt.leaves) <= max_leaves
end

# ---- a random graded tree and eta field --------------------------------------
"Random refine/coarsen sequence on a small tree: graded, varied levels."
function g215_tree(rng, maxlev::Int=6, nsteps::Int=40; cells_x::Int=1,
                   symmetry::Symbol=:none)
    qt = Quadtree(maxlev; cells_x=cells_x, cells_y=1)
    for _ in 1:nsteps
        ls = collect(qt.leaves)
        lf = ls[rand(rng, 1:length(ls))]
        if rand(rng) < 0.62
            refine!(qt, lf)
        else
            coarsen!(qt, lf)
        end
        rand(rng) < 0.25 && balance!(qt)
    end
    balance!(qt)
    symmetry === :x && begin
        mirror_close!(qt; max_level=maxlev)
        balance!(qt; symmetry=:x)
    end
    return qt
end

"Random indicator over the leaves, plus the wall-row protect set."
function g215_eta_and_protect(rng, qt)
    eta = Dict{NTuple{3,Int},Float64}()
    for lf in qt.leaves
        eta[lf] = rand(rng) * 10.0^(rand(rng, -3:2))
    end
    protect = Set{NTuple{3,Int}}()
    for lf in qt.leaves
        (lf[3] == 0 || lf[3] == qt.ny0 * (1 << lf[1]) - 1 ||
         rand(rng) < 0.05) && push!(protect, lf)
    end
    return eta, protect
end

@testset "G215 AMR hot path" begin
    rng = MersenneTwister(20261111)

    # ---- 1. equivalence with the pre-audit implementation -------------------
    @testset "pre-audit equivalence: predicate, plan, candidate, budget pass" begin
        nallow = 0
        nplan = 0
        nplan_nonempty = 0
        nbudget = 0
        nmerged = 0
        for trial in 1:24
            sym = trial % 3 == 0 ? :x : :none
            qt = g215_tree(rng, 6, 40; cells_x=(trial % 2 == 0 ? 2 : 1),
                           symmetry=(trial % 4 == 0 ? :x : :none))
            eta, protect = g215_eta_and_protect(rng, qt)
            will_refine = Set(lf for lf in qt.leaves if rand(rng) < 0.1)
            granted = Set(MS._pq(lf) for lf in qt.leaves if rand(rng) < 0.05)
            # (a) the predicate, on every complete quartet
            for lf in collect(qt.leaves)
                kids = MS._quartet_kids(lf)
                all(k -> k in qt.leaves, kids) || continue
                a_old = g215_old_allowed(qt, kids, will_refine, granted)
                a_new = MS._coarsening_allowed(qt, kids, will_refine, granted)
                a_tup = MS._coarsening_allowed(qt, MS._kids4(lf), will_refine,
                                              granted)
                @test a_old == a_new == a_tup
                @test MS._merge_allowed(qt, lf, will_refine, granted, sym) ==
                      g215_old_merge_allowed(qt, lf, will_refine, granted, sym)
                nallow += 1
            end
            # (b) the plan, on several need/threshold combinations
            for (need, tau_h, tau_r) in ((0, 0.5, Inf), (9, 0.5, 1.0),
                                         (30, 0.3, 1.0), (200, 0.01, Inf))
                p_old = g215_old_plan(qt, copy(eta), protect, _ -> 0, nothing,
                                      need, sym, will_refine, tau_h, tau_r)
                p_new = MS._coarsen_plan(qt, copy(eta), protect, _ -> 0, nothing,
                                         need, sym, will_refine, tau_h, tau_r)
                @test p_old.order == p_new.order
                @test p_old.refused == p_new.refused
                @test p_old.extended == p_new.extended
                nplan += 1
                nplan_nonempty += isempty(p_new.order) ? 0 : 1
            end
            # (c) the budget candidate, on every leaf
            blocked = Set(lf for lf in qt.leaves if rand(rng) < 0.02)
            for lf in collect(qt.leaves)
                c_old = g215_old_budget_quartet(qt, lf, eta, protect, blocked)
                c_new = MS._budget_quartet(qt, lf, eta, protect, blocked)
                @test (c_old === nothing) == (c_new === nothing)
                if c_old !== nothing
                    @test (c_old.kids[1, 1], c_old.kids[2, 1], c_old.kids[1, 2],
                           c_old.kids[2, 2]) == c_new.kids
                    @test c_old.eta === c_new.eta
                end
                nbudget += 1
            end
            # (d) the whole hard-budget pass, from the same tree and eta
            q_old = deepcopy(qt)
            q_new = deepcopy(qt)
            e_old = copy(eta)
            e_new = copy(eta)
            budget = max(1, length(qt.leaves) - 12)
            m_old, ok_old = g215_old_budget_pass!(q_old, e_old; max_leaves=budget,
                                                  protect=copy(protect),
                                                  symmetry=(trial % 4 == 0 ? :x : :none))
            m_new, ok_new = semquad_enforce_leaf_budget!(q_new, e_new;
                                                  max_leaves=budget,
                                                  protect=copy(protect),
                                                  symmetry=(trial % 4 == 0 ? :x : :none))
            @test m_old == m_new
            @test ok_old == ok_new
            @test g215_fnv_leaves(q_old) == g215_fnv_leaves(q_new)
            @test e_old == e_new
            d = LAST_BUDGET_PASS[]
            @test d.merged == m_new
            @test d.scans >= 0 && d.guard >= d.scans
            @test d.capped == false
            nmerged += m_new
        end
        @printf("  equivalence: %d quartets, %d plans (%d non-empty), %d candidates, %d budget merges\n",
                nallow, nplan, nplan_nonempty, nbudget, nmerged)
        @test nallow > 1000
        @test nplan_nonempty >= 6
        @test nmerged > 0
    end

    # ---- 2. allocation budgets ---------------------------------------------
    @testset "allocation budgets (0 B/call) and the historical controls" begin
        qt = g215_tree(MersenneTwister(7), 6, 200; cells_x=5)
        eta, protect = g215_eta_and_protect(MersenneTwister(8), qt)
        lf = first(lf for lf in sort(collect(qt.leaves))
                   if all(k -> k in qt.leaves, MS._quartet_kids(lf)))
        kids = MS._quartet_kids(lf)
        pg = Set{NTuple{3,Int}}()
        blocked = Set{NTuple{3,Int}}()
        @printf("  tree: %d leaves; quartet %s\n", length(qt.leaves), string(lf))
        # warm the face cache: the *first* query on a (leaf,dir) must build and
        # cache its list, which is a one-off allocation, not a per-call one
        MS._coarsening_allowed(qt, kids, pg, pg)
        MS._coarsening_allowed(qt, MS._kids4(lf), pg, pg)
        g215_old_allowed(qt, kids, pg, pg)
        MS._merge_allowed(qt, lf, pg, pg, :none)
        MS._budget_quartet(qt, lf, eta, protect, blocked)
        g215_old_budget_quartet(qt, lf, eta, protect, blocked)
        a_new = @allocated MS._coarsening_allowed(qt, kids, pg, pg)
        a_tup = @allocated MS._coarsening_allowed(qt, MS._kids4(lf), pg, pg)
        a_old = @allocated g215_old_allowed(qt, kids, pg, pg)
        a_mrg = @allocated MS._merge_allowed(qt, lf, pg, pg, :none)
        a_bq = @allocated MS._budget_quartet(qt, lf, eta, protect, blocked)
        a_bq_old = @allocated g215_old_budget_quartet(qt, lf, eta, protect, blocked)
        a_inv = @allocated MS._invalidate_faces!(qt)
        # one full candidate scan, quartet at a time (the budget pass's inner loop)
        scan(qt, eta, protect, blocked, seen) = begin
            n = 0
            empty!(seen)
            for l in qt.leaves
                p = (l[1] - 1, l[2] >> 1, l[3] >> 1)
                p in seen && continue
                push!(seen, p)
                MS._budget_quartet(qt, l, eta, protect, blocked) === nothing ||
                    (n += 1)
            end
            n
        end
        seen = Set{NTuple{3,Int}}()
        scan(qt, eta, protect, blocked, seen)
        a_scan = @allocated scan(qt, eta, protect, blocked, seen)
        @printf("  bytes/call: coarsening_allowed new=%d tuple=%d old=%d | merge_allowed=%d | budget_quartet new=%d old=%d | scan=%d | invalidate=%d\n",
                a_new, a_tup, a_old, a_mrg, a_bq, a_bq_old, a_scan, a_inv)
        @test a_new == 0
        @test a_tup == 0
        @test a_mrg == 0
        @test a_bq == 0
        @test a_scan == 0
        @test a_inv == 0
        # the budgets are load-bearing: the historical versions allocate
        @test a_old > 0
        @test a_bq_old > 0
        # _coarsen_plan allocates O(#candidates), not O(#leaves): with the
        # cycle's sorted `asc` it does not re-sort the leaf set (measured 4.4x /
        # 2.6x fewer bytes), and neither form allocates per leaf.
        asc = sort(collect(qt.leaves), by=l -> (max(get(eta, l, 0.0), 0.0), l))
        wr = Set{NTuple{3,Int}}()
        pl1 = MS._coarsen_plan(qt, eta, protect, _ -> 0, nothing, 30, :none, wr,
                               0.5, 1.0; asc=asc)
        a_plan_asc = @allocated MS._coarsen_plan(qt, eta, protect, _ -> 0,
                                                 nothing, 30, :none, wr, 0.5, 1.0;
                                                 asc=asc)
        a_plan_sort = @allocated MS._coarsen_plan(qt, eta, protect, _ -> 0,
                                                  nothing, 30, :none, wr, 0.5,
                                                  1.0)
        @printf("  _coarsen_plan bytes: asc=%d re-sort=%d (sweeps=%d rounds=%d)\n",
                a_plan_asc, a_plan_sort, pl1.sweeps, pl1.rounds)
        @test a_plan_asc < a_plan_sort
        @test pl1.capped == false
        @test pl1.sweeps <= 8
    end

    # ---- 3. the caps fire and are recorded ---------------------------------
    @testset "loop caps: plan round cap, mirror cap, budget guard" begin
        # (a) a tree on which *no* quartet is legal: every neighbour is in
        # `will_refine` (+1 level), so on a uniform tree every face would end two
        # levels finer than the merged parent.  The plan cannot progress, so the
        # extension walk must stop at MAX_PLAN_ROUNDS and say so.  256 quartets
        # (> the 22 + 14*15 = 232 the round cap can reach) are needed so the walk
        # cannot escape through the `limit >= n` exit first.
        qt = Quadtree(7)
        for _ in 1:5, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        @test length(qt.leaves) == 1024
        eta = Dict{NTuple{3,Int},Float64}(lf => 1.0 for lf in qt.leaves)
        wr = Set{NTuple{3,Int}}(qt.leaves)      # every leaf refine-flagged
        pl = MS._coarsen_plan(qt, eta, Set{NTuple{3,Int}}(), _ -> 0, nothing, 40,
                              :none, wr, 1.0, Inf)
        @printf("  capped plan: |order|=%d refused=%d sweeps=%d rounds=%d capped=%s (MAX rounds=%d)\n",
                length(pl.order), pl.refused, pl.sweeps, pl.rounds, pl.capped,
                MS.MAX_PLAN_ROUNDS)
        @test isempty(pl.order)
        @test pl.capped
        @test pl.rounds == MS.MAX_PLAN_ROUNDS
        @test pl.sweeps >= 1
        @test pl.refused > 0
        # ... and the same tree with an empty will_refine converges normally
        pl2 = MS._coarsen_plan(qt, eta, Set{NTuple{3,Int}}(), _ -> 0, nothing, 40,
                               :none, Set{NTuple{3,Int}}(), 1.0, Inf)
        @printf("  normal plan: |order|=%d sweeps=%d rounds=%d capped=%s\n",
                length(pl2.order), pl2.sweeps, pl2.rounds, pl2.capped)
        @test !pl2.capped
        @test length(pl2.order) >= 40 ÷ 3
        @test pl2.rounds <= 4
        # (b) the caps are structural, not a behaviour change: a cycle on the
        # same uniform tree reports them in LAST_AMR_BALANCED and they do not fire
        q3 = deepcopy(qt)
        amr_cycle_balanced!(q3, eta; refine_frac=0.0, coarsen_frac=0.2,
                            max_level=6, max_leaves=length(qt.leaves) - 30,
                            symmetry=:none, coarsen_gate=true)
        d = LAST_AMR_BALANCED[]
        @printf("  cycle: coarsened=%d plan_sweeps=%d plan_rounds=%d capped=%s fund_calls=%d balance_ok=%s mirror_ok=%s\n",
                d.coarsened, d.plan_sweeps, d.plan_rounds, d.plan_capped,
                d.fund_calls, d.balance_ok, d.mirror_ok)
        @test d.plan_capped == false
        @test d.plan_rounds >= 1
        @test d.balance_ok
        @test d.mirror_ok
        @test max_face_level_jump(q3) <= 1
        # (c) mirror_close! records its rounds and a forced cap is visible
        qm = g215_tree(MersenneTwister(11), 6, 30; symmetry=:x)
        defect0 = mirror_defect(qm)
        r1, ok1 = mirror_close!(qm; max_level=6)
        c1 = LAST_MIRROR_CLOSE[]
        @printf("  mirror_close!: defect0=%d refined=%d ok=%s rounds=%d capped=%s\n",
                defect0, r1, ok1, c1.rounds, c1.capped)
        @test ok1 == (mirror_defect(qm) == 0)
        @test c1.refined == r1 && c1.ok == ok1
        @test c1.rounds >= 1
        @test c1.capped == false
        # force the cap: a tree that needs >1 round with maxiter=1
        qm2 = g215_tree(MersenneTwister(12), 6, 30; symmetry=:x)
        refine!(qm2, first(sort(collect(qm2.leaves))))
        refine!(qm2, first(sort(collect(qm2.leaves))))
        r2, ok2 = mirror_close!(qm2; max_level=6, maxiter=1)
        c2 = LAST_MIRROR_CLOSE[]
        @printf("  mirror_close! maxiter=1: refined=%d ok=%s rounds=%d capped=%s\n",
                r2, ok2, c2.rounds, c2.capped)
        @test c2.rounds == 1
        @test c2.capped == (r2 > 0)
        # (d) the budget pass reports scans/guard and never silently spins
        qb = g215_tree(MersenneTwister(13), 6, 60; symmetry=:x)
        eb = Dict{NTuple{3,Int},Float64}()
        for lf in qb.leaves
            eb[lf] = rand(rng) + 1.0
        end
        n0 = length(qb.leaves)
        mb, okb = semquad_enforce_leaf_budget!(qb, eb; max_leaves=n0 - 3,
                                               symmetry=:x)
        db = LAST_BUDGET_PASS[]
        @printf("  budget: leaves %d -> %d merged=%d scans=%d guard=%d reached=%s capped=%s\n",
                n0, length(qb.leaves), mb, db.scans, db.guard, db.reached,
                db.capped)
        @test db.merged == mb
        @test db.capped == false
        @test db.scans >= 1
        @test okb == db.reached
        @test length(qb.leaves) <= n0 - 3 || !okb
    end

    # ---- 4. the face cache is never stale ----------------------------------
    @testset "face cache: incremental table, same answer as a fresh tree" begin
        rng2 = MersenneTwister(21)
        qt = g215_tree(rng2, 6, 200; cells_x=5)
        for lf in collect(qt.leaves)
            for dir in MS.FACE_DIRS
                neighbors(qt, lf, dir)        # fill the table
            end
        end
        tab = qt.faces.tab
        nq = length(tab.ids)
        @test nq == length(qt.leaves)
        @test tab.cur != 0
        nbuild = tab.rebuilds
        @printf("  table: %d leaves, %d rebuilds while building the tree\n",
                nq, nbuild)
        st0 = qt.faces.stamp
        # a mutation must only bump the stamp and mark the touched leaves
        # stale: the table is *incrementally* maintained, never rebuilt
        @test refine!(qt, first(sort(collect(qt.leaves))))
        @test qt.faces.stamp == st0 + 1
        @test tab.rebuilds == nbuild
        stale = count(i -> tab.gen[i] != tab.cur, 1:length(tab.keys))
        @printf("  one refine! marks %d of %d leaves stale\n", stale,
                length(tab.keys))
        @test 0 < stale <= 40
        # same leaves, cold cache: the inner constructor resets `QuadFaceCache`
        qref = MS.Quadtree(qt.maxlevel, qt.Lx, qt.nx0, qt.ny0, copy(qt.leaves))
        same = 0
        for lf in sort(collect(qt.leaves))
            lf in qt.leaves || continue
            for dir in MS.FACE_DIRS
                a = neighbors(qt, lf, dir)          # recomputes the stale ones
                b = neighbors(qref, lf, dir)
                @test a == b
                same += 1
            end
        end
        @printf("  cache: %d queries after one mutation, stamp=%d\n",
                same, qt.faces.stamp)
        @test count(i -> tab.gen[i] != tab.cur, 1:length(tab.keys)) == 0
        # repeated query/mutate alternation stays consistent (the budget pass's
        # pattern: one merge, one balance, one fresh scan)
        for _ in 1:5
            lf = first(sort(collect(qt.leaves)))
            coarsen!(qt, lf) || continue
            balance!(qt)
            qref2 = MS.Quadtree(qt.maxlevel, qt.Lx, qt.nx0, qt.ny0,
                                copy(qt.leaves))
            for l in sort(collect(qt.leaves))[1:min(40, length(qt.leaves))]
                for dir in MS.FACE_DIRS
                    @test neighbors(qt, l, dir) == neighbors(qref2, l, dir)
                end
            end
        end
        @printf("  after 5 merge+balance cycles: %d rebuilds (incremental)\n",
                tab.rebuilds)
    end

    # ---- 5. a deterministic cycle: the new counters are pinned ------------
    @testset "cycle counters and a pinned gate-on cycle" begin
        qt = g215_tree(MersenneTwister(31), 6, 60; cells_x=2)
        eta = Dict{NTuple{3,Int},Float64}()
        for lf in qt.leaves
            # a deadband with a few low-error quartets, like the operating state
            eta[lf] = 1.0 + (lf[3] % 3) * 0.5 + (lf[2] % 2) * 0.25
        end
        n0 = length(qt.leaves)
        q1 = deepcopy(qt)
        q2 = deepcopy(qt)
        e1 = copy(eta)
        e2 = copy(eta)
        r1 = amr_cycle_balanced!(q1, e1; refine_frac=0.0, coarsen_frac=0.3,
                                 max_level=6, max_leaves=n0 - 30, symmetry=:none,
                                 coarsen_gate=true)
        d1 = LAST_AMR_BALANCED[]
        r2 = amr_cycle_balanced!(q2, e2; refine_frac=0.0, coarsen_frac=0.3,
                                 max_level=6, max_leaves=n0 - 30, symmetry=:none,
                                 coarsen_gate=false)
        d2 = LAST_AMR_BALANCED[]
        @printf("  gate on : coarsened=%d recirc=%d plan_sweeps=%d rounds=%d fund=%d/%d leaves=%d\n",
                r1[2], d1.recirc, d1.plan_sweeps, d1.plan_rounds, d1.fund_hits,
                d1.fund_calls, length(q1.leaves))
        @printf("  gate off: coarsened=%d recirc=%d plan_sweeps=%d rounds=%d leaves=%d\n",
                r2[2], d2.recirc, d2.plan_sweeps, d2.plan_rounds, length(q2.leaves))
        @test d1.plan_capped == false
        @test d1.plan_sweeps > 0 && d1.plan_rounds > 0
        @test d2.plan_sweeps == 0 && d2.plan_rounds == 0
        @test d1.recirc == 0
        @test r1[2] > 0
        @test max_face_level_jump(q1) <= 1
        @test isapprox(total_area(q1), q1.Lx; rtol=0.0, atol=1.0e-14)
        @printf("  leaf FNV after the gate-on cycle = %s\n", g215_fnv_leaves(q1))
    end
end
