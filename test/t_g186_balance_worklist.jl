using Test, QuaST, Random, Printf
using QuaST: FACE_DIRS

# G186: 2:1 balancing as a local worklist + face/neighbour caching per tree
# version (the step-95 AMR outlier fix).
#
# Background (docs/dev/SOLVER.md §16): on the Ra=1e8 blob production run the
# step that first hit `max_leaves=1024` (958 -> 1030 leaves after refinement)
# spent 31.5-31.9 s in one AMR cycle.  The cause was not a single slow balance
# pass but the *leaf-budget guard* of the then-current fraction cycle (the
# removed `amr_cycle!`): the lowest-error complete
# quartet was coarsened, 2:1 balancing re-split its parent (the parent has a
# one-level-finer neighbour), and the loop retried the same quartet 10000
# times, each retry a full-tree `balance!` (2 passes, ~8.2k `neighbors` calls
# and ~2.1e5 `_collect_face!` visits).  That is 31.8 s of pure no-op work.
#
# This gate pins the three properties the fix relies on:
#   (1) the worklist `balance!` produces the same leaf set as the old
#       full-scan implementation on random refinement/coarsening sequences;
#   (2) `neighbors`/`face_pairs` caches are invalidated by every tree
#       mutation (a queried-then-mutated tree equals a freshly built one);
#   (3) a cycle with no refine/coarsen fractions on an over-budget tree
#       terminates as a fixed point in milliseconds instead of 10000 rounds,
#       and the hard leaf-budget pass then merges the tree down.

# The old full-scan balance, kept here as the reference semantics.
function balance_full_scan!(qt::Quadtree; maxiter::Int=1000)
    for _ in 1:maxiter
        changed = false
        for leaf in sort(collect(qt.leaves))
            leaf in qt.leaves || continue
            for dir in FACE_DIRS
                bad = false
                for nb in neighbors(qt, leaf, dir)
                    if nb[1] >= leaf[1] + 2
                        bad = true
                        break
                    end
                end
                if bad
                    refine!(qt, leaf)
                    changed = true
                    break
                end
            end
        end
        changed || return true
    end
    return false
end

mirror(lf) = (lf[1], (1 << lf[1]) - 1 - lf[2], lf[3])

function copy_tree(qt::Quadtree)
    qc = Quadtree(qt.maxlevel; Lx=qt.Lx, cells_x=qt.nx0, cells_y=qt.ny0)
    empty!(qc.leaves)
    for lf in qt.leaves
        push!(qc.leaves, lf)
    end
    QuaST._invalidate_faces!(qc)
    return qc
end

"Random refine/coarsen(/balance) sequence; returns a leaf set, not necessarily 2:1."
function random_tree(rng, maxlevel; nops::Int, cx::Int=1, cy::Int=1)
    qt = Quadtree(maxlevel; cells_x=cx, cells_y=cy)
    for _ in 1:nops
        lvs = collect(qt.leaves)
        if rand(rng) < 0.7
            lf = lvs[rand(rng, 1:length(lvs))]
            refine!(qt, lf)
            if rand(rng) < 0.4
                ml = mirror(lf)
                rand(rng) < 0.8 && ml in qt.leaves && refine!(qt, ml)
            end
        else
            lf = lvs[rand(rng, 1:length(lvs))]
            coarsen!(qt, lf)
        end
        rand(rng) < 0.25 && balance!(qt)
    end
    return qt
end

"Cross-check every cached face query against a freshly built copy of the tree."
function cache_matches_fresh(qt::Quadtree)
    qref = copy_tree(qt)
    for lf in qt.leaves, dir in FACE_DIRS
        neighbors(qt, lf, dir) == neighbors(qref, lf, dir) || return false
    end
    return face_pairs(qt) == face_pairs(qref)
end

"""
The production step-95 shape: a level-6 block of 64x64 leaves (x,y < 0.5) inside
a uniform level-4 tree, so balancing puts a level-5 ring around it.  A complete
level-5 quartet in that ring has a level-4 parent adjacent to the level-6 block:
coarsening it is undone by `balance!` and would restore the tree exactly.
"""
function over_budget_tree()
    qt = Quadtree(7)
    for _ in 1:4, lf in collect(qt.leaves)              # uniform level 4: 256 leaves
        refine!(qt, lf)
    end
    for _ in 1:2                                        # lower-left quadrant -> level 6
        for lf in collect(qt.leaves)
            h = 1.0 / (1 << lf[1]); x0 = lf[2] * h; y0 = lf[3] * h
            if lf[1] < 6 && x0 + h <= 0.5 + 1e-12 && y0 + h <= 0.5 + 1e-12
                refine!(qt, lf)
            end
        end
    end
    balance!(qt)
    return qt
end

@testset "G186 balance worklist + face cache" begin
    # ---------------------------------------------------------------- (1)
    # worklist balance == full-scan balance on random trees
    rng = MersenneTwister(186)
    ntrial = 40
    for trial in 1:ntrial
        maxlevel = rand(rng, 2:7)
        cx, cy = rand(rng, 1:3), rand(rng, 1:2)
        qt = random_tree(rng, maxlevel; nops=rand(rng, 10:80), cx=cx, cy=cy)
        qa = copy_tree(qt)
        qb = copy_tree(qt)
        ok_a = balance!(qa)
        ok_b = balance_full_scan!(qb)
        @test ok_a == ok_b
        @test Set(qa.leaves) == Set(qb.leaves)
        @test max_face_level_jump(qa) <= 1
        # balance! is idempotent: a second call changes nothing
        na = length(qa.leaves)
        @test balance!(qa)
        @test length(qa.leaves) == na
        @test cache_matches_fresh(qa)
    end

    # ---------------------------------------------------------------- (2)
    # the face cache is dropped by refine!/coarsen!/balance! on *every*
    # mutation; a populated-then-mutated tree must agree with a fresh one.
    # Since the flat neighbour table (`QuadFaceTable`,
    # docs/dev/AMR.md §9) the invalidation is *per leaf*: a mutation
    # marks exactly the leaves whose neighbour lists change stale and a query
    # recomputes those, instead of dropping the whole table.  The contract is
    # unchanged and is what is asserted here: no stale list is ever served.
    qt = Quadtree(6)
    for _ in 1:4, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    for lf in qt.leaves, dir in FACE_DIRS     # populate the whole table
        neighbors(qt, lf, dir)
    end
    face_pairs(qt)
    tab = qt.faces.tab
    nq = length(tab.ids)
    @test nq == length(qt.leaves)
    @test all(tab.gen[tab.ids[lf]] == tab.cur for lf in qt.leaves)
    @test qt.faces.pairs !== nothing
    nbuild = tab.rebuilds
    stamp = qt.faces.stamp
    n0 = length(qt.leaves)
    # refine one leaf: stamp bumps, only the touched leaves go stale, queries
    # stay correct (and the first one recomputes those leaves)
    lf = sort(collect(qt.leaves))[1]
    @test refine!(qt, lf)
    @test qt.faces.stamp == stamp + 1
    @test qt.faces.pairs === nothing
    @test tab.rebuilds == nbuild            # incremental: no full rebuild
    stale = count(i -> tab.gen[i] != tab.cur, 1:length(tab.keys))
    @test 0 < stale <= 40                   # the touched leaf + its neighbours
    @test cache_matches_fresh(qt)
    @test count(i -> tab.gen[i] != tab.cur, 1:length(tab.keys)) == 0
    # coarsen a quartet: same contract
    stamp = qt.faces.stamp
    merged = false
    for cand in sort(collect(qt.leaves))
        if coarsen!(qt, cand)
            merged = true
            break
        end
    end
    @test merged
    @test qt.faces.stamp > stamp
    @test tab.rebuilds == nbuild
    @test cache_matches_fresh(qt)
    # a queried tree, a `deepcopy` of it, and a never-queried copy all agree
    qdc = deepcopy(qt)
    @test cache_matches_fresh(qdc)
    @test Set(qdc.leaves) == Set(qt.leaves)

    # ---------------------------------------------------------------- (3)
    # over-budget AMR cycle at a balancing fixed point: the old guard loop ran
    # its full 10000 iterations here (31.8 s on the production tree); the fixed
    # point must be detected in milliseconds.
    qt = over_budget_tree()
    n0 = length(qt.leaves)
    @test n0 > 1000
    @test max_face_level_jump(qt) <= 1
    eta = Dict{NTuple{3,Int},Float64}(lf => 1.0 for lf in qt.leaves)
    # the four level-5 children of the level-4 cell (4,3,8): its face at y=0.5
    # touches the level-6 block, so the merge is undone by balance
    stuck = [(5, 6, 16), (5, 7, 16), (5, 6, 17), (5, 7, 17)]
    for lf in stuck
        @test lf in qt.leaves
        eta[lf] = 0.0
    end
    q = copy_tree(qt)
    # JIT warm-up on a throwaway copy (the cycle is a no-op fixed point anyway)
    qw = copy_tree(qt)
    amr_cycle_balanced!(qw, eta; refine_frac=0.0, coarsen_frac=0.0,
                        max_level=7, max_leaves=n0 - 4, symmetry=:none)
    t0 = time()
    nr, nc, _ = amr_cycle_balanced!(q, eta; refine_frac=0.0, coarsen_frac=0.0,
                                    max_level=7, max_leaves=n0 - 4,
                                    symmetry=:none)
    elapsed = time() - t0
    @printf("G186: %d-leaf over-budget cycle %.4f s (nr=%d nc=%d)\n",
            n0, elapsed, nr, nc)
    @test elapsed < 2.0            # old code: 10000 no-op rounds, ~30 s here
    @test Set(q.leaves) == Set(qt.leaves)   # exact fixed point: tree restored
    @test max_face_level_jump(q) <= 1
    # the driver's blocked-set budget pass then merges the tree down
    eta_cap = copy(eta)
    for lf in q.leaves
        get!(eta_cap, lf, 0.0)
    end
    merged, ok = semquad_enforce_leaf_budget!(q, eta_cap; max_leaves=n0 - 4,
                                              symmetry=:none)
    @test ok
    @test merged > 0
    @test length(q.leaves) <= n0 - 4
    @test max_face_level_jump(q) <= 1
end
