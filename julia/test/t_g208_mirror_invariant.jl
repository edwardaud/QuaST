using Test, MantleSEM, Random, Printf

# G208: the x-mirror invariant of `symmetry=:x`.
#
# `symmetry=:x` is supposed to keep the AMR forest closed under the reflection
# x -> Lx - x: for every leaf `(l, ix, iy)` the partner `(l, nx0*2^l - 1 - ix, iy)`
# must be a leaf.  The marking and the coarsening passes have always acted on
# mirror pairs, but until `docs/CASE_REALISTIC.md` §15 three operations did not:
#
#   * `_coarsen_pair!` merged only the `lf` quartet when the *mirror* quartet
#     was in `protect` (the `coarsen_theta_tol` guard is built from a field that
#     is only approximately mirror-symmetric).  This is the operation that
#     introduces the *first* asymmetry: measured 0 -> 81 non-mirrored leaves in
#     one coarsening pass on a 910-leaf tree.
#   * `semquad_enforce_leaf_budget!` and `amr_cycle!`'s hard-budget guard merged
#     a quartet whose mirror was protected or incomplete on one side only
#     (measured 0 -> 5 and 0 -> 10 on closed trees).
#   * `balance!` split a single leaf for the 2:1 condition; on a closed tree its
#     badness test is mirror-invariant so it did not create the first defect, but
#     it is now pair-strict so it cannot spread one either.
#
# One one-sided merge leaves **five** non-mirrored leaves: the new parent (its
# mirror, the mirror parent, is still split) plus the four children of the mirror
# parent (their mirrors, the lf children, are gone).  That is why the realistic
# checkpoint's count moves in fives and reached 120/125 by step 6130.
#
# The fix: `balance!(...; symmetry=:x)` splits mirror pairs, `_coarsen_pair!`
# and the budget pass merge both sides or neither, `protect` is closed under the
# mirror, and `mirror_close!` is a refinement-only repair pass that restores the
# invariant on a tree damaged before the fix.  `symmetry=:none` is untouched:
# the pins at the bottom are byte-identical to the pre-fix worktree.

# ---- helpers ---------------------------------------------------------------
mirror(lf::NTuple{3,Int}, nx0::Int) = (lf[1], nx0 * (1 << lf[1]) - 1 - lf[2], lf[3])
nonmirror(qt::Quadtree) = count(lf -> !(mirror(lf, qt.nx0) in qt.leaves), qt.leaves)

"""FNV-1a 64 over the raw bytes of a Float64 vector (no external dependency)."""
function _fnv(a)
    h = 0xcbf29ce484222325
    for b in reinterpret(UInt8, a)
        h ⊻= UInt64(b)
        h *= 0x100000001b3
    end
    return string(h, base=16, pad=16)
end
_fnvtree(qt) = _fnv(reinterpret(UInt8, reduce(vcat,
    [collect(Float64, (l, ix, iy)) for (l, ix, iy) in sort(collect(qt.leaves))])))

"""Uniform base tree refined `base` times (mirror-closed by construction)."""
function _seed_tree(cx::Int, cy::Int, base::Int, maxlev::Int)
    qt = Quadtree(maxlev; cells_x=cx, cells_y=cy, Lx=cx / cy)
    for _ in 1:base, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

"""Indicator-driven random AMR sequence.  `mode=:graded` puts the error near the
bottom wall (the realistic case's shape), `:uniform` is white noise."""
function _random_eta(qt::Quadtree, rng; mode::Symbol=:graded)
    eta = Dict{NTuple{3,Int},Float64}()
    for lf in qt.leaves
        eta[lf] = mode === :uniform ? rand(rng) :
                  rand(rng)^2 * exp(-3.0 * (lf[3] + 0.5) / (qt.ny0 * (1 << lf[1])))
    end
    return eta
end

"""A non-uniform but mirror-closed tree: paired refinements + balance."""
function _graded_closed_tree(cx::Int, cy::Int, maxlev::Int, rounds::Int, rng)
    qt = _seed_tree(cx, cy, 3, maxlev)
    for _ in 1:rounds
        cand = sort([lf for lf in qt.leaves if lf[1] < maxlev], by=lf -> rand(rng))
        for lf in cand[1:min(24, length(cand))]
            refine!(qt, lf)
            ml = mirror(lf, qt.nx0)
            ml in qt.leaves && refine!(qt, ml)
        end
        balance!(qt)
    end
    return qt
end

@testset "G208 mirror invariant under symmetry=:x" begin
    # ---- 1. `mirror_defect` and the five-defect arithmetic ------------------
    @testset "defect count" begin
        for (cx, cy) in ((1, 1), (1, 2), (5, 2), (2, 1))
            qt = _seed_tree(cx, cy, 2, 6)
            @test nonmirror(qt) == 0
            @test mirror_defect(qt) == 0
            # one one-sided split = 5 non-mirrored leaves (measured)
            lf = sort(collect(qt.leaves))[1]
            @test refine!(qt, lf)
            @test mirror_defect(qt) == 5
            @test mirror_defect(qt) == nonmirror(qt)
            # ... and `mirror_close!` repairs it by refinement only
            n0 = length(qt.leaves)
            added, ok = mirror_close!(qt; max_level=qt.maxlevel)
            @test ok
            @test mirror_defect(qt) == 0
            @test added > 0
            @test length(qt.leaves) > n0
            @test isapprox(total_area(qt), qt.Lx; rtol=0.0, atol=1.0e-14)
            balance!(qt; symmetry=:x)
            @test mirror_defect(qt) == 0
            @test max_face_level_jump(qt) <= 1
        end
        # the repair respects max_level: a defect it cannot represent is refused
        qt = _seed_tree(5, 2, 1, 2)
        lf = sort(collect(qt.leaves))[1]
        refine!(qt, lf)                        # level-2 leaves, level-1 mirror
        @test mirror_defect(qt) == 5
        added0, ok0 = mirror_close!(qt; max_level=1)
        @test added0 == 0
        @test !ok0                             # max_level 1 cannot represent it
        @test maximum(l -> l[1], collect(qt.leaves)) == 2
        added1, ok1 = mirror_close!(qt; max_level=2)
        @test ok1
        @test added1 == 1
        @test mirror_defect(qt) == 0
        @test maximum(l -> l[1], collect(qt.leaves)) <= 2
    end

    # ---- 2. `balance!` with `symmetry=:x` needs pair splits ----------------
    @testset "balance! splits mirror pairs" begin
        for (cx, cy) in ((1, 1), (5, 2))
            for seed in 1:3
                rng = MersenneTwister(seed)
                qt = _graded_closed_tree(cx, cy, 6, 3, rng)
                # paired refinements without balance -> a 2:1 violation
                cand = sort([lf for lf in qt.leaves if lf[1] < 6],
                            by=lf -> rand(rng))
                for lf in cand[1:6]
                    refine!(qt, lf)
                    ml = mirror(lf, qt.nx0)
                    ml in qt.leaves && refine!(qt, ml)
                end
                @test mirror_defect(qt) == 0
                @test max_face_level_jump(qt) > 1
                @test balance!(qt; symmetry=:x)
                @test max_face_level_jump(qt) <= 1
                @test mirror_defect(qt) == 0
                @test isapprox(total_area(qt), qt.Lx; rtol=0.0, atol=1.0e-14)
            end
        end
        @test_throws ArgumentError balance!(_seed_tree(1, 1, 1, 3); symmetry=:y)
    end

    # ---- 3. random AMR cycles: defect == 0 after every cycle ---------------
    # `amr_cycle!`'s own guard is a *best-effort* cap (its fixed point can stop
    # above `max_leaves` when 2:1 balancing re-splits the merged parent; measured
    # 1234 leaves against 1200 on this sequence, with or without the fix), so the
    # hard cap asserted here is the one the driver applies, i.e.
    # `amr_cycle!` followed by `semquad_enforce_leaf_budget!` in the same pass.
    @testset "random AMR cycles" begin
        for (cx, cy, maxlev) in ((1, 1, 7), (5, 2, 7))
            for seed in 1:3
                rng = MersenneTwister(seed)
                qt = _seed_tree(cx, cy, 3, maxlev)
                @test mirror_defect(qt) == 0
                for k in 1:15
                    eta = _random_eta(qt, rng)
                    amr_cycle!(qt, eta; refine_frac=0.2, coarsen_frac=0.05,
                               marking=:error_fraction, max_level=maxlev,
                               max_leaves=1200, symmetry=:x)
                    @test mirror_defect(qt) == 0
                    @test max_face_level_jump(qt) <= 1
                    eta_cap = copy(eta)
                    for lf in qt.leaves
                        get!(eta_cap, lf, 0.0)
                    end
                    merged, ok = semquad_enforce_leaf_budget!(qt, eta_cap;
                        max_leaves=1200, symmetry=:x)
                    @test ok
                    @test length(qt.leaves) <= 1200
                    @test merged % 2 == 0
                    @test mirror_defect(qt) == 0
                    @test max_face_level_jump(qt) <= 1
                end
            end
            # `marking=:level` / `:global` and the refine-only path too
            for marking in (:level, :global)
                rng = MersenneTwister(7)
                qt = _seed_tree(cx, cy, 3, maxlev)
                for k in 1:5
                    eta = _random_eta(qt, rng; mode=:uniform)
                    amr_cycle!(qt, eta; refine_frac=0.25, coarsen_frac=0.0,
                               marking=marking, max_level=maxlev,
                               max_leaves=900, symmetry=:x)
                    @test mirror_defect(qt) == 0
                    @test max_face_level_jump(qt) <= 1
                end
            end
        end
    end

    # ---- 4. the coarsening pass with an asymmetric protect set ------------
    # This is the operation that introduced the first asymmetry (0 -> 81).  A
    # 30 % random protect set is the adversarial limit of `coarsen_theta_tol`
    # (which fires on one side of a pair when the field is not exactly
    # symmetric).
    @testset "asymmetric protect set" begin
        for (cx, cy) in ((1, 1), (5, 2))
            rng = MersenneTwister(17)
            qt = _graded_closed_tree(cx, cy, 7, 4, rng)
            @test mirror_defect(qt) == 0
            protect = Set{NTuple{3,Int}}()
            for lf in sort(collect(qt.leaves))
                rand(rng) < 0.30 && push!(protect, lf)
            end
            eta = _random_eta(qt, rng)
            nr, nc = amr_cycle!(qt, eta; refine_frac=0.0, coarsen_frac=0.3,
                                marking=:error_fraction, max_level=7,
                                symmetry=:x, protect=protect)
            @test mirror_defect(qt) == 0       # was 81 / 35 before the fix
            @test max_face_level_jump(qt) <= 1
            @test nc > 0                       # coarsening is not simply dead
            # no protected leaf may disappear
            @test all(lf -> lf in qt.leaves, protect)
            # and the same for the mirror-aware budget pass
            n1 = length(qt.leaves)
            merged, ok = semquad_enforce_leaf_budget!(qt, eta;
                max_leaves=n1 - 12, protect=protect, symmetry=:x)
            @test mirror_defect(qt) == 0       # was 65 -> 76 before the fix
            @test max_face_level_jump(qt) <= 1
            @test merged % 2 == 0              # pair merges only
            @test all(lf -> lf in qt.leaves, protect)
            @test ok || length(qt.leaves) <= n1
        end
    end

    # ---- 5. the hard leaf budget -------------------------------------------
    @testset "leaf budget" begin
        # (a) a mild protect set: the budget is reached exactly, in pairs
        for (cx, cy) in ((1, 1), (5, 2))
            rng = MersenneTwister(31)
            qt = _graded_closed_tree(cx, cy, 7, 4, rng)
            eta = Dict{NTuple{3,Int},Float64}(lf => rand(rng) for lf in qt.leaves)
            protect = Set{NTuple{3,Int}}()
            for lf in sort(collect(qt.leaves))
                rand(rng) < 0.05 && push!(protect, lf)
            end
            budget = length(qt.leaves) - 24
            merged, ok = semquad_enforce_leaf_budget!(qt, eta;
                max_leaves=budget, protect=protect, symmetry=:x)
            @test ok
            @test length(qt.leaves) <= budget
            @test merged % 2 == 0
            @test mirror_defect(qt) == 0
            @test max_face_level_jump(qt) <= 1
        end
        # (b) the adversarial 30 % protect set: the invariant wins and the
        # refusal is bounded.  Documented tolerance: one pair (6 leaves) per
        # refused pair; measured worst case on this tree is +18.
        for (cx, cy) in ((1, 1), (5, 2))
            rng = MersenneTwister(37)
            qt = _graded_closed_tree(cx, cy, 7, 4, rng)
            eta = Dict{NTuple{3,Int},Float64}(lf => rand(rng) for lf in qt.leaves)
            protect = Set{NTuple{3,Int}}()
            for lf in sort(collect(qt.leaves))
                rand(rng) < 0.30 && push!(protect, lf)
            end
            n0 = length(qt.leaves)
            budget = n0 - 24
            merged, ok = semquad_enforce_leaf_budget!(qt, eta;
                max_leaves=budget, protect=protect, symmetry=:x)
            @test mirror_defect(qt) == 0
            @test max_face_level_jump(qt) <= 1
            @test merged % 2 == 0
            @test length(qt.leaves) <= budget + 18
            @test length(qt.leaves) <= n0
            @test ok || length(qt.leaves) <= budget + 18
        end
        # (c) `amr_cycle!`'s internal guard with :x keeps the invariant
        for (cx, cy) in ((1, 1), (5, 2))
            rng = MersenneTwister(41)
            qt = _graded_closed_tree(cx, cy, 7, 4, rng)
            eta = Dict{NTuple{3,Int},Float64}(lf => rand(rng) for lf in qt.leaves)
            protect = Set{NTuple{3,Int}}()
            for lf in sort(collect(qt.leaves))
                rand(rng) < 0.10 && push!(protect, lf)
            end
            n0 = length(qt.leaves)
            nr, nc = amr_cycle!(qt, eta; refine_frac=0.0, coarsen_frac=0.0,
                                marking=:error_fraction, max_level=7,
                                max_leaves=n0 - 24, symmetry=:x,
                                protect=protect)
            @test mirror_defect(qt) == 0
            @test max_face_level_jump(qt) <= 1
            @test nc % 2 == 0
            @test length(qt.leaves) <= n0
        end
    end

    # ---- 6. the wall-refined / wall-protected path -------------------------
    @testset "wall path" begin
        for (cx, cy) in ((1, 1), (5, 2))
            rng = MersenneTwister(3)
            qt = _seed_tree(cx, cy, 2, 7)
            for k in 1:4
                nref = refine_wall_to_min!(qt, 5; strip=0.1, symmetry=:x)
                @test mirror_defect(qt) == 0
                @test max_face_level_jump(qt) <= 1
                eta = _random_eta(qt, rng)
                protect = Set{NTuple{3,Int}}()
                for lf in qt.leaves
                    h = 1.0 / (qt.ny0 * (1 << lf[1]))
                    zc = (lf[3] + 0.5) * h
                    if lf[3] == 0 || lf[3] == qt.ny0 * (1 << lf[1]) - 1 ||
                       zc <= 0.1 || zc >= 0.9
                        eta[lf] = 0.0
                        push!(protect, lf)
                    end
                end
                merged, ok = semquad_enforce_leaf_budget!(qt, eta;
                    max_leaves=length(qt.leaves) - 8, protect=protect,
                    symmetry=:x)
                @test mirror_defect(qt) == 0
                @test max_face_level_jump(qt) <= 1
                @test merged % 2 == 0
                @test all(lf -> lf in qt.leaves, protect)
            end
        end
        # a legacy (damaged) tree is repaired by the first wall+AMR pass
        for (cx, cy) in ((1, 1), (5, 2))
            qt = _seed_tree(cx, cy, 2, 7)
            for lf in sort(collect(qt.leaves))[1:3]   # one-sided damage
                refine!(qt, lf)
            end
            @test mirror_defect(qt) > 0
            refine_wall_to_min!(qt, 4; strip=0.1, symmetry=:x)
            balance!(qt; symmetry=:x)
            added, ok = mirror_close!(qt; max_level=7)
            @test ok
            @test mirror_defect(qt) == 0
            @test max_face_level_jump(qt) <= 1
            @test isapprox(total_area(qt), qt.Lx; rtol=0.0, atol=1.0e-14)
        end
        @test_throws ArgumentError refine_wall_to_min!(_seed_tree(1, 1, 1, 3), 2;
                                                        symmetry=:y)
    end

    # ---- 7. the driver pass (`semquad_amr_passes!`) ------------------------
    @testset "driver pass" begin
        pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz,
                    eta_b=9.70406052783923433184,
                    eta_c=4.1588830833596718565,
                    Twall_top=0.0, Twall_bot=1.0, bc=:free)
        for (cx, cy) in ((1, 1), (5, 2))
            qt = Quadtree(5; cells_x=cx, cells_y=cy, Lx=cx / cy)
            for _ in 1:2, lf in collect(qt.leaves)
                refine!(qt, lf)
            end
            balance!(qt)
            ss = make_semstokes(qt, 2)
            th = [0.1 * sinpi(y) * cospi(x / qt.Lx) for (x, y) in ss.su.coords]
            for (g, (_, y)) in enumerate(ss.su.coords)
                (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th[g] = 0.0)
            end
            for k in 1:3
                ss, th, _, nr = semquad_amr_passes!(
                    ss, qt, th, copy(th); refine_frac=0.3, coarsen_frac=0.05,
                    marking=:error_fraction, max_level=5, max_leaves=400,
                    wall_min_level=3, wall_strip=0.1, symmetry=:x)
                @test mirror_defect(qt) == 0
                @test max_face_level_jump(qt) <= 1
                @test length(qt.leaves) <= 400 + 18
            end
        end
    end

    # ---- 8. `symmetry=:none` is byte-identical to the pre-fix tree --------
    # Values measured on the pristine HEAD 94e8bd3 worktree with
    # `out_julia/agent/g208_snapshot.jl`; the fix must not move a single bit.
    @testset "symmetry=:none is unchanged" begin
        # (a) `balance!` on a 5x2 tree with one-sided refinements
        qt = Quadtree(6; cells_x=5, cells_y=2, Lx=2.5)
        for _ in 1:3, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        for lf in sort(collect(qt.leaves))[1:4]
            refine!(qt, lf)
        end
        balance!(qt)
        @test length(qt.leaves) == 652
        @test _fnvtree(qt) == "35968c872ebeabf0"
        @test max_face_level_jump(qt) == 1

        # (b) `refine_wall_to_min!` on the default path
        qw = Quadtree(7; cells_x=5, cells_y=2, Lx=2.5)
        for _ in 1:2, lf in collect(qw.leaves)
            refine!(qw, lf)
        end
        balance!(qw)
        @test refine_wall_to_min!(qw, 5; strip=0.1) == 680
        @test length(qw.leaves) == 2320
        @test _fnvtree(qw) == "07812bd2ff43ed4d"

        # (c) a deterministic AMR sequence
        q = Quadtree(7; cells_x=5, cells_y=2, Lx=2.5)
        for _ in 1:3, lf in collect(q.leaves)
            refine!(q, lf)
        end
        balance!(q)
        rng = MersenneTwister(20261109)
        for k in 1:6
            eta = Dict{NTuple{3,Int},Float64}()
            for lf in q.leaves
                eta[lf] = rand(rng)^2 *
                          exp(-2.0 * (lf[3] + 0.5) / (q.ny0 * (1 << lf[1])))
            end
            amr_cycle!(q, eta; refine_frac=0.2, coarsen_frac=0.05,
                       marking=:error_fraction, max_level=7, max_leaves=900)
        end
        @test length(q.leaves) == 901
        @test _fnvtree(q) == "0de7757f838fd16f"
        @test max_face_level_jump(q) == 1

        # (d) a small driver run: theta/ux/uz/history bitwise
        pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz,
                    eta_b=9.70406052783923433184,
                    eta_c=4.1588830833596718565,
                    Twall_top=0.0, Twall_bot=1.0, bc=:free)
        qt5 = Quadtree(5; cells_x=5, cells_y=2, Lx=2.5)
        for _ in 1:2, lf in collect(qt5.leaves)
            refine!(qt5, lf)
        end
        balance!(qt5)
        ss5 = make_semstokes(qt5, 2)
        th5 = [0.1 * sinpi(y) * cospi(x / qt5.Lx) for (x, y) in ss5.su.coords]
        for (g, (_, y)) in enumerate(ss5.su.coords)
            (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th5[g] = 0.0)
        end
        # Classic tree pins: this driver path is the historical (classic) trajectory.
        res = run_semquad_simulation(qt5, pr, th5; p=2, tstop=8.0e-4,
                                     dt0=1.0e-4, maxsteps=5, amr_every=1,
                                     coarsen_frac=0.0, refine_frac=0.5,
                                     marking=:error_fraction, max_level=5,
                                     max_leaves=120, amr_policy=:classic,
                                     linear=:direct, verbose=false)
        @test res.steps == 5
        @test length(res.qt.leaves) == 118
        @test _fnv(reinterpret(UInt8, res.theta)) == "84de84a66ab9dd69"
        @test _fnv(reinterpret(UInt8, vcat(res.ux, res.uz))) == "6ea033597f291f32"
        @test _fnv(reinterpret(UInt8, reduce(vcat, res.history))) == "8fcbf706331b1542"
    end
end
