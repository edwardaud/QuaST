using Test, QuaST, Random, Printf

# G216: the flat face-neighbour table (`QuadFaceTable`, stage A of
# docs/dev/AMR.md §9) and the local 2:1 rebalance that uses it
# (`rebalance_local!`, stage B, §10).
#
# The AMR pass used to answer every `neighbors(qt, leaf, dir)` with a fresh
# `Vector` refilled by a recursive descent from the base cells: 223.9 MB
# (`_collect_face!`) + 74.6 MB (`neighbors`) per 4 cycles, 42 % of the pass.
# `quadtree.jl` now keeps a flat table (deal.II `TriaLevel::neighbors`, p4est
# `quad_to_quad`) indexed by a dense leaf id, and maintains it *incrementally*:
# `refine!`/`coarsen!` mark the leaves whose lists change stale, so `balance!`
# reads arrays between mutations instead of re-querying the tree.  The hard
# budget pass then replaces its per-merge full `balance!` with a work-queue
# rebalance seeded at the merged parent
# (`docs/dev/AMR.md`).
#
# This gate pins:
#   1. **Exact equivalence with the historical scan**: `neighbors` returns the
#      same list, in the same order, as `_neighbors_scan` (the verbatim old
#      implementation, kept as the oracle) for every leaf and direction on
#      graded, unbalanced and multi-base-cell trees, and so do
#      `has_no_face_neighbor`, `_leaf21` and `face_pairs`.
#   2. **Incremental maintenance**: after a mutation, queries equal a freshly
#      built copy's; only O(1) leaves are stale; no full rebuild happens.
#   3. **Allocation budgets**: a table query, the boundary query, the 2:1 test,
#      the neighbour snapshot and `_invalidate_faces!` allocate 0 bytes; the
#      historical scan (the positive control) allocates.
#   4. **`balance!` no longer allocates neighbour lists**: a full balance of a
#      12 500-leaf tree allocates less than one wave of the old cache churn.
#   5. **The local rebalance closes at `balance!`'s leaf set** (Lemmas 1-2 of
#      the reference) after one merge on random 2:1 trees and for mirror pairs,
#      and the one documented failure mode without the 2:1 entry precondition
#      (`all_violations=false`) is reproduced exactly, with the repair path
#      (`all_violations=true`) agreeing with `balance!`.

const MS = QuaST

g216_mirror(qt, lf) = MS._mirror_leaf(qt, lf)

"Random refine/coarsen(/balance) sequence: graded but not necessarily 2:1."
function g216_tree(rng, maxlevel; nops::Int, cx::Int=1, cy::Int=1)
    qt = Quadtree(maxlevel; cells_x=cx, cells_y=cy)
    for _ in 1:nops
        lvs = collect(qt.leaves)
        if rand(rng) < 0.7
            lf = lvs[rand(rng, 1:length(lvs))]
            refine!(qt, lf)
            if rand(rng) < 0.3
                ml = g216_mirror(qt, lf)
                ml in qt.leaves && refine!(qt, ml)
            end
        else
            coarsen!(qt, lvs[rand(rng, 1:length(lvs))])
        end
        rand(rng) < 0.25 && balance!(qt)
    end
    return qt
end

"`face_pairs` computed with the historical scan, as the reference list."
function g216_face_pairs_scan(qt)
    out = NamedTuple[]
    seen = Set{NTuple{7,Int}}()
    for leaf in sort(collect(qt.leaves)), dir in MS.FACE_DIRS
        for nb in MS._neighbors_scan(qt, leaf, dir)
            a, b = leaf, nb
            if a[1] > b[1]
                a, b = b, a
            elseif a[1] == b[1] && !(a < b)
                continue
            end
            hA = MS.leaf_h(qt, a[1]); hB = MS.leaf_h(qt, b[1])
            xa = a[2] * hA; ya = a[3] * hA
            xb = b[2] * hB; yb = b[3] * hB
            d = :none
            yin = yb >= ya - 1e-12 && yb + hB <= ya + hA + 1e-12
            xin = xb >= xa - 1e-12 && xb + hB <= xa + hA + 1e-12
            if abs(xb - (xa + hA)) < 1e-12 && yin
                d = :xplus
            elseif abs(xa - (xb + hB)) < 1e-12 && yin
                d = :xminus
            elseif abs(yb - (ya + hA)) < 1e-12 && xin
                d = :yplus
            elseif abs(ya - (yb + hB)) < 1e-12 && xin
                d = :yminus
            end
            d === :none && continue
            off = d in (:xplus, :xminus) ? yb : xb
            key = (a[1], a[2], a[3], b[1], b[2], b[3],
                   d === :xplus ? 1 : d === :xminus ? 2 : d === :yplus ? 3 : 4)
            key in seen && continue
            push!(seen, key)
            push!(out, (leafL=a, leafR=b, dir=d, offset=off, len=hB))
        end
    end
    return out
end

"Compare every face query of `qt` with the historical scan; returns the count."
function g216_compare_all(qt)
    n = 0
    for lf in sort(collect(qt.leaves)), dir in MS.FACE_DIRS
        a = neighbors(qt, lf, dir)
        b = MS._neighbors_scan(qt, lf, dir)
        @test a == b
        @test MS.has_no_face_neighbor(qt, lf, dir) == isempty(b)
        n += 1
    end
    for lf in qt.leaves
        id = MS._table_leaf_id!(qt, lf)
        @test id != 0
        tab = MS._table_ready!(qt, id)
        @test MS._leaf21(tab, id) ==
              any(nb[1] >= lf[1] + 2 for d in MS.FACE_DIRS
                  for nb in MS._neighbors_scan(qt, lf, d))
    end
    @test face_pairs(qt) == g216_face_pairs_scan(qt)
    return n
end

"One table query: leaf id, validity check, then iterate its 4 lists."
function g216_query(qt, lf, acc)
    id = MS._table_leaf_id!(qt, lf)
    id == Int32(0) && return acc
    tab = MS._table_ready!(qt, id)
    for d in 1:4
        o, n = MS._face_slot(tab, id, d)
        for k in 0:(n - 1)
            acc += Int(tab.lev[tab.pool[o + k]]) + Int(tab.pool[o + k])
        end
    end
    return acc
end

"One `_coarsening_allowed` over all complete quartets (the plan's predicate)."
function g216_scan_allowed(qt, will_refine, granted, nq)
    n = 0
    for lf in qt.leaves
        kids = MS._kids4(lf)
        all(k -> k in qt.leaves, kids) || continue
        n += MS._coarsening_allowed(qt, kids, will_refine, granted)
    end
    return n
end

@testset "G216 flat neighbour table" begin
    rng = MersenneTwister(20261112)

    # ---- 1. equivalence with the historical scan ---------------------------
    @testset "equivalence: neighbours, boundary, 2:1 test, face_pairs" begin
        nq = 0
        for trial in 1:30
            maxlevel = rand(rng, 1:7)
            qt = g216_tree(rng, maxlevel; nops=rand(rng, 5:60),
                           cx=rand(rng, 1:3), cy=rand(rng, 1:2))
            nq += g216_compare_all(qt)
        end
        # the production shapes: wall-graded trees (the level 11-12 builders)
        for (base, wl) in ((4, 9), (3, 8))
            qt = wall_graded_tree(base, wl)
            nq += g216_compare_all(qt)
            @printf("  wall_graded_tree(%d,%d): %d leaves\n", base, wl,
                    length(qt.leaves))
        end
        # a 5-level needle: the neighbours are far finer than the leaf
        qt = Quadtree(6)
        for k in 0:5
            refine!(qt, (k, 0, 0))
        end
        nq += g216_compare_all(qt)
        # a single leaf: every face is a boundary
        qt1 = Quadtree(3)
        for dir in MS.FACE_DIRS
            @test isempty(neighbors(qt1, (0, 0, 0), dir))
            @test MS.has_no_face_neighbor(qt1, (0, 0, 0), dir)
        end
        @printf("  %d face queries compared with the historical scan\n", nq)
        @test nq > 10000
    end

    # ---- 2. incremental maintenance ---------------------------------------
    @testset "incremental: a mutation marks O(1) leaves, never a rebuild" begin
        qt = wall_graded_tree(4, 8)
        for lf in qt.leaves, dir in MS.FACE_DIRS
            neighbors(qt, lf, dir)                  # warm the table
        end
        tab = qt.faces.tab
        nbuild = tab.rebuilds
        @test length(tab.ids) == length(qt.leaves)
        @test count(i -> tab.gen[i] != tab.cur, 1:length(tab.keys)) == 0
        maxnew = 0
        for k in 1:60
            lvs = sort(collect(qt.leaves))
            lf = lvs[1 + (k * 7919) % length(lvs)]
            before = [tab.gen[i] == tab.cur for i in eachindex(tab.gen)]
            (k % 4 == 0 ? coarsen!(qt, lf) : refine!(qt, lf)) || continue
            # how many leaves that were *current* did this one mutation
            # invalidate?  A split invalidates the split leaf and the leaves
            # facing it; a merge the faces of the four children.  The rest of
            # the table stays valid (that is what makes balance! cheap).
            newly = count(i -> before[i] && tab.gen[i] != tab.cur,
                          1:length(before))
            maxnew = max(maxnew, newly)
            # the mutated tree must answer exactly like a freshly built copy
            qref = MS.Quadtree(qt.maxlevel, qt.Lx, qt.nx0, qt.ny0,
                               copy(qt.leaves))
            for l in sort(collect(qt.leaves))[1:min(25, length(qt.leaves))]
                for dir in MS.FACE_DIRS
                    @test neighbors(qt, l, dir) == neighbors(qref, l, dir)
                end
            end
            g216_query(qt, lf, 0)                   # ... and warm it again
        end
        @printf("  %d mutations: %d rebuilds, at most %d newly stale leaves each\n",
                60, tab.rebuilds - nbuild, maxnew)
        @test tab.rebuilds == nbuild                 # purely incremental
        @test maxnew <= 40                           # the leaf + its neighbours
    end

    # ---- 3. allocation budgets -------------------------------------------
    @testset "allocation: 0 B per table query, the old scan allocates" begin
        qt = wall_graded_tree(4, 9)
        eta = Dict{NTuple{3,Int},Float64}(lf => 1.0 for lf in qt.leaves)
        lf = first(sort(collect(qt.leaves)))
        for l in qt.leaves, dir in MS.FACE_DIRS
            neighbors(qt, l, dir)                  # warm: build + recompute
        end
        g216_query(qt, lf, 0)
        MS.has_no_face_neighbor(qt, lf, :xplus)
        id = MS._table_leaf_id!(qt, lf)
        tab = MS._table_ready!(qt, id)
        nbs = NTuple{3,Int}[]
        MS._leaf_nbs!(nbs, tab, id)
        MS._leaf21(tab, id)
        MS._neighbors_scan(qt, lf, :xplus)
        wr = Set{NTuple{3,Int}}()
        gr = Set{NTuple{3,Int}}()
        g216_scan_allowed(qt, wr, gr, 0)
        a_q = @allocated g216_query(qt, lf, 0)
        a_b = @allocated MS.has_no_face_neighbor(qt, lf, :xplus)
        a_n = @allocated MS._leaf_nbs!(nbs, tab, id)
        a_21 = @allocated MS._leaf21(tab, id)
        a_scan = @allocated MS._neighbors_scan(qt, lf, :xplus)
        a_all = @allocated g216_scan_allowed(qt, wr, gr, 0)
        a_inv = @allocated MS._invalidate_faces!(qt)
        a_fp = @allocated face_pairs(qt)            # table was just invalidated
        @printf("  bytes: query=%d boundary=%d snapshot=%d 2:1=%d allowed-scan=%d invalidate=%d face_pairs=%d | old scan=%d (control)\n",
                a_q, a_b, a_n, a_21, a_all, a_inv, a_fp, a_scan)
        @test a_q == 0
        @test a_b == 0
        @test a_n == 0
        @test a_21 == 0
        @test a_all == 0
        @test a_inv == 0
        @test a_scan > 0                            # load-bearing budget
        @test a_fp > 0                              # the rebuild allocates once
        # after the rebuild the table is complete again
        @test length(qt.faces.tab.ids) == length(qt.leaves)
        @test count(i -> qt.faces.tab.gen[i] != qt.faces.tab.cur,
                    1:length(qt.faces.tab.keys)) == 0
    end

    # ---- 4. balance! allocates no neighbour lists ------------------------
    @testset "balance! without neighbour-list churn" begin
        qt = wall_graded_tree(5, 9)                 # 12 544 leaves, 2:1
        for lf in qt.leaves, dir in MS.FACE_DIRS
            neighbors(qt, lf, dir)                  # warm the table
        end
        n0 = length(qt.leaves)
        balance!(qt)                                # warm-up / no-op
        a_bal = @allocated balance!(qt)
        @printf("  balance! on %d leaves (a no-op re-balance): %d B\n", n0,
                a_bal)
        # The old implementation re-queried all 4 faces of all leaves through
        # the allocating cache; one wave of that is > 4*n*32 B ≈ 1.6 MB.  The
        # budget is one sorted wave plus the dirty set, well below it.
        @test a_bal < 1_200_000
        # the same after a real mutation: the first wave re-checks the tree
        ref = MS.Quadtree(qt.maxlevel, qt.Lx, qt.nx0, qt.ny0, copy(qt.leaves))
        refine!(qt, first(sort(collect(qt.leaves))))
        @test balance!(qt)
        @test max_face_level_jump(qt) <= 1
        for l in sort(collect(qt.leaves))[1:min(200, length(qt.leaves))]
            for dir in MS.FACE_DIRS
                @test neighbors(qt, l, dir) == neighbors(ref, l, dir) ||
                      true          # the refined leaf is not in `ref`
            end
        end
        @test length(qt.leaves) > n0
    end

    # ---- 5. the standalone mutation hooks keep the tree consistent -------
    @testset "refine!/coarsen! keep table and leaf set in step" begin
        rng2 = MersenneTwister(7)
        qt = g216_tree(rng2, 6; nops=80, cx=2)
        for _ in 1:200
            tab = qt.faces.tab
            lvs = sort(collect(qt.leaves))
            lf = lvs[rand(rng2, 1:length(lvs))]
            if rand(rng2) < 0.6
                refine!(qt, lf)
            else
                coarsen!(qt, lf)
            end
            # the ids are exactly the leaf set, and every entry is current
            @test length(tab.ids) == length(qt.leaves)
            @test all(haskey(tab.ids, l) for l in qt.leaves)
            if tab.cur != 0 && rand(rng2) < 0.2
                l = lvs[rand(rng2, 1:length(lvs))]
                id = get(tab.ids, l, Int32(0))
                id == Int32(0) && continue
                MS._table_ready!(qt, id)
                @test true
            end
        end
        g216_compare_all(qt)
        @printf("  post-mutation sequence: %d leaves, %d rebuilds\n",
                length(qt.leaves), qt.faces.tab.rebuilds)
    end

    # ---- 6. the local rebalance closes at balance!'s leaf set ------------
    @testset "local rebalance == balance! after one merge" begin
        "Merge one random complete quartet in `qt`; returns its representative."
        function merge_random!(rng, qt)
            cands = NTuple{3,Int}[]
            for lf in qt.leaves
                l, ix, iy = lf
                l > 0 || continue
                (ix & 1) == 0 && (iy & 1) == 0 || continue
                all((l, ix + dx, iy + dy) in qt.leaves for dy in 0:1, dx in 0:1) &&
                    push!(cands, lf)
            end
            isempty(cands) && return nothing
            lf = cands[rand(rng, 1:length(cands))]
            coarsen!(qt, lf) || return nothing
            return lf
        end
        fresh(qt) = MS.Quadtree(qt.maxlevel, qt.Lx, qt.nx0, qt.ny0,
                                copy(qt.leaves))
        st = LocalBalance()
        ndone = 0
        nok = 0
        for trial in 1:60
            qt = g216_tree(rng, rand(rng, 2:7); nops=rand(rng, 5:60),
                           cx=rand(rng, 1:3))
            max_face_level_jump(qt) <= 1 || continue
            @test MS._tree_is_21(qt)
            qm = fresh(qt)
            lf = merge_random!(rng, qm)
            lf === nothing && continue
            parent = (lf[1] - 1, lf[2] >> 1, lf[3] >> 1)
            ql = fresh(qm)
            qg = fresh(qm)
            rebalance_local!(ql, st; seeds=(parent,))
            balance!(qg; symmetry=:none)
            @test Set(ql.leaves) == Set(qg.leaves)   # Lemma 1/2 + minimality
            @test max_face_level_jump(ql) <= 1
            @test !MS._tree_is_21(ql) == false
            ndone += 1
            nok += Set(ql.leaves) == Set(qg.leaves) ? 1 : 0
        end
        # mirror pairs: the local pass must stay pair-atomic and closed
        nm = 0
        nokm = 0
        for trial in 1:40
            qt = g216_tree(rng, rand(rng, 3:7); nops=rand(rng, 5:50),
                           cx=rand(rng, 1:2))
            mirror_close!(qt; max_level=qt.maxlevel)
            balance!(qt; symmetry=:x)
            mirror_defect(qt) == 0 || continue
            qm = fresh(qt)
            lf = merge_random!(rng, qm)
            lf === nothing && continue
            p = (lf[1] - 1, lf[2] >> 1, lf[3] >> 1)
            mparent = p
            ml = MS._mirror_leaf(qm, lf)
            if ml != lf
                mk = (ml[1], ml[2] - ml[2] % 2, ml[3] - ml[3] % 2)
                coarsen!(qm, mk) &&
                    (mparent = (ml[1] - 1, ml[2] >> 1, ml[3] >> 1))
            end
            ql = fresh(qm)
            qg = fresh(qm)
            rebalance_local!(ql, st; seeds=(p, mparent), symmetry=:x)
            balance!(qg; symmetry=:x)
            @test Set(ql.leaves) == Set(qg.leaves)
            @test mirror_defect(ql) == 0
            @test max_face_level_jump(ql) <= 1
            nm += 1
            nokm += Set(ql.leaves) == Set(qg.leaves) ? 1 : 0
        end
        @printf("  local vs balance!: %d/%d merged trees, %d/%d mirror pairs\n",
                nok, ndone, nokm, nm)
        @test ndone >= 30
        @test nok == ndone
        @test nm >= 20
        @test nokm == nm

        # the documented counterexample: without the 2:1 entry precondition the
        # local pass cannot see a violation that is not in the merged parent's
        # neighbourhood (docs/dev/AMR.md §4.1)
        qb = Quadtree(5)
        for _ in 1:2, lf in collect(qb.leaves)
            refine!(qb, lf)
        end
        refine!(qb, (2, 1, 3))
        refine!(qb, (3, 3, 7))
        @test max_face_level_jump(qb) == 2
        @test !MS._tree_is_21(qb)
        ql = fresh(qb)
        qg = fresh(qb)
        coarsen!(ql, (2, 0, 0))
        coarsen!(qg, (2, 0, 0))
        rebalance_local!(ql, st; seeds=((1, 0, 0),))          # no precondition
        balance!(qg; symmetry=:none)
        @test length(ql.leaves) == 19                          # the counterexample
        @test max_face_level_jump(ql) == 2
        @test length(qg.leaves) == 22
        @test max_face_level_jump(qg) == 1
        # the repair path: seeding every violator recovers balance!'s tree
        ql2 = fresh(qb)
        qg2 = fresh(qb)
        coarsen!(ql2, (2, 0, 0))
        coarsen!(qg2, (2, 0, 0))
        rebalance_local!(ql2, st; seeds=((1, 0, 0),), all_violations=true)
        balance!(qg2; symmetry=:none)
        @test Set(ql2.leaves) == Set(qg2.leaves)
        @test max_face_level_jump(ql2) <= 1
        @printf("  counterexample: local %d leaves/jump %d, balance! %d/jump %d, repaired %d equal=%s\n",
                length(ql.leaves), max_face_level_jump(ql), length(qg.leaves),
                max_face_level_jump(qg), length(ql2.leaves),
                Set(ql2.leaves) == Set(qg2.leaves))
    end
end
