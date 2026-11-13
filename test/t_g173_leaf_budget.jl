using Test, QuaST, LinearAlgebra, Printf

# G173: hard leaf budget under refine-only AMR (coarsen_frac = 0).
#
# With coarsen_frac = 0 the AMR cycle cannot coarsen, and 2:1 balancing can add
# leaves after the refinement, so a refine-only run used to stay far above
# `max_leaves` (measured on the Ra=1e8 blob case: 3016 leaves against
# max_leaves = 1600).  `semquad_enforce_leaf_budget!` now merges the
# lowest-error complete sibling quartets until the budget is met (or no quartet
# is left); this gate pins that down, plus the protect and symmetry rules and a
# driver-level integration check.

@testset "G173 hard leaf budget" begin
    function graded_tree(base::Int, hi::Int, nref::Int)
        qt = Quadtree(hi)
        for _ in 1:base, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        # refine the nref leaves nearest the origin up to `hi`
        for _ in 1:(hi - base)
            cand = sort([lf for lf in qt.leaves if lf[1] < hi],
                        by=lf -> (lf[2] + lf[3], lf))
            for lf in cand[1:min(nref, length(cand))]
                refine!(qt, lf)
            end
            balance!(qt)
        end
        return qt
    end

    # (1) budget reached, 2:1 preserved, finest leaves survive
    qt = graded_tree(3, 6, 4)
    n0 = length(qt.leaves)
    eta = Dict{NTuple{3,Int},Float64}()
    for lf in qt.leaves
        # error concentrated on the refined corner (small x+z), flat elsewhere
        eta[lf] = lf[1] >= 5 ? 1.0 : 0.01 * (1.0 + lf[2] + lf[3])
    end
    budget = n0 - 8
    merged, reached = semquad_enforce_leaf_budget!(qt, eta; max_leaves=budget,
                                                   symmetry=:none)
    @test merged > 0
    @test reached
    @test length(qt.leaves) <= budget
    @test max_face_level_jump(qt) <= 1
    @test any(lf -> lf[1] >= 5, qt.leaves)   # the high-error corner is kept

    # (2) no-op when the tree is already inside the budget
    qt2 = graded_tree(3, 4, 2)
    n2 = length(qt2.leaves)
    eta2 = Dict{NTuple{3,Int},Float64}(lf => 1.0 for lf in qt2.leaves)
    m2, ok2 = semquad_enforce_leaf_budget!(qt2, eta2; max_leaves=n2 + 10)
    @test m2 == 0
    @test ok2
    @test length(qt2.leaves) == n2

    # (3) protected quartets are never merged
    qt3 = graded_tree(2, 5, 4)
    n3 = length(qt3.leaves)
    eta3 = Dict{NTuple{3,Int},Float64}(lf => 0.0 for lf in qt3.leaves)
    protected = Set{NTuple{3,Int}}()
    for lf in sort(collect(qt3.leaves), by=lf -> eta3[lf])[1:4]
        push!(protected, lf)
    end
    _, ok3 = semquad_enforce_leaf_budget!(qt3, eta3; max_leaves=1,
                                          protect=protected, symmetry=:none)
    @test all(lf -> lf in qt3.leaves || lf[1] >= 1, protected)
    @test length(qt3.leaves) >= 1
    @test !ok3 || length(qt3.leaves) <= 1

    # (4) :x symmetry merges mirror quartets together -> even leaf reduction.
    # Build a mirror-symmetric tree first: the mirror of (l, ix, iy) in the unit
    # box is (l, 2^l - 1 - ix, iy).
    qt4 = Quadtree(5)
    for _ in 1:3, lf in collect(qt4.leaves)
        refine!(qt4, lf)
    end
    balance!(qt4)
    mirror(lf) = (lf[1], (1 << lf[1]) - 1 - lf[2], lf[3])
    for _ in 1:2
        cand = sort([lf for lf in qt4.leaves if lf[1] < 5], by=lf -> (lf[2] + lf[3], lf))
        for lf in cand[1:2]
            refine!(qt4, lf)
            ml = mirror(lf)
            ml in qt4.leaves && refine!(qt4, ml)
        end
        balance!(qt4)
    end
    n4 = length(qt4.leaves)
    eta4 = Dict{NTuple{3,Int},Float64}(lf => 1.0 for lf in qt4.leaves)
    merged4, _ = semquad_enforce_leaf_budget!(qt4, eta4; max_leaves=n4 - 16,
                                              symmetry=:x)
    @test merged4 % 2 == 0
    @test length(qt4.leaves) <= n4 - 16
    @test all(lf -> mirror(lf) in qt4.leaves, qt4.leaves)   # still mirror-symmetric

    # (5) driver integration: refine-only AMR keeps leaves <= max_leaves
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz, eta_b=9.70406052783923433,
                eta_c=4.15888308335967186, Twall_top=0.0, Twall_bot=1.0,
                bc=:free)
    qt5 = Quadtree(5)
    for _ in 1:2, lf in collect(qt5.leaves)
        refine!(qt5, lf)
    end
    balance!(qt5)
    ss5 = make_semstokes(qt5, 2)
    th5 = [0.1 * sinpi(y) * cos(2pi * x) for (x, y) in ss5.su.coords]
    for (g, (_, y)) in enumerate(ss5.su.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th5[g] = 0.0)
    end
    seen = Int[]
    cb = st -> push!(seen, length(st.qt.leaves))
    # This gate pins the classic hard max_leaves cap and its quartet-merge path.
    res = run_semquad_simulation(qt5, pr, th5; p=2, tstop=6.0e-4, dt0=1.0e-4,
                                 maxsteps=6, amr_every=1, coarsen_frac=0.0,
                                 refine_frac=0.5, marking=:error_fraction,
                                 amr_policy=:classic,
                                 max_level=5, max_leaves=40, linear=:direct,
                                 callback=cb, verbose=false)
    @test !isempty(seen)
    @test maximum(seen) <= 40
    @test length(res.qt.leaves) <= 40
    @test max_face_level_jump(res.qt) <= 1
    # (6) driver path that must actually MERGE: the budget starts below the
    # initial tree, so the over-budget merge inside semquad_amr_passes! runs
    # (this is the path that crashed with an undefined `verbose` when the
    # budget was first wired in).
    qt6 = Quadtree(5)
    for _ in 1:3, lf in collect(qt6.leaves)
        refine!(qt6, lf)
    end
    balance!(qt6)
    n6 = length(qt6.leaves)
    @test n6 > 32
    ss6 = make_semstokes(qt6, 2)
    th6 = [0.1 * sinpi(y) * cos(2pi * x) for (x, y) in ss6.su.coords]
    for (g, (_, y)) in enumerate(ss6.su.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th6[g] = 0.0)
    end
    seen6 = Int[]
    cb6 = st -> push!(seen6, length(st.qt.leaves))
    res6 = run_semquad_simulation(qt6, pr, th6; p=2, tstop=4.0e-4, dt0=1.0e-4,
                                  maxsteps=4, amr_every=1, coarsen_frac=0.0,
                                  refine_frac=0.5, marking=:error_fraction,
                                  amr_policy=:classic,
                                  max_level=5, max_leaves=32, linear=:direct,
                                  callback=cb6, verbose=false)
    @test !isempty(seen6)
    @test maximum(seen6) <= 32
    @test length(res6.qt.leaves) <= 32

    @printf("G173: budget enforcement ok (driver leaves=%d, peak=%d, merge path %d->%d)\n",
            length(res.qt.leaves), maximum(seen), n6, length(res6.qt.leaves))
end
