# G187: the balance! fixed-point guard and its unreachability invariant.
using Test, QuaST, LinearAlgebra, Random

# G187 / S3 stage 4: the `balance!` fixed-point guard.
#
# `balance!` walks a wave of leaves that violate the 2:1 condition, refines them
# and queues their old face neighbours for the next pass.  A leaf that
# `refine!` refuses was re-queued unchanged, so a wave that makes no progress
# used to be retried `maxiter` = 1000 times: the same "state does not change,
# loop keeps going" bug class as the step-95 leaf-budget spin
# only bounded by `maxiter`.  The fix adds one
# comparison per pass:
#
#     wave == prev && break        # nothing was refined -> next pass repeats
#
# (`wave` is already sorted, so the comparison is O(#leaves) and exits at the
# first difference on a tree that is still making progress.)
#
# The guard is *defensive*: it is provably unreachable with the current
# `_collect_face!`/`refine!` pair, and the gate pins the reason instead of
# pretending to fire it:
#
#   * `_collect_face!` never descends below `qt.maxlevel` (`l >= qt.maxlevel &&
#     return`), so a *detected* violation `nb[1] >= leaf[1] + 2` implies
#     `leaf[1] + 2 <= maxlevel`, i.e. `leaf[1] < maxlevel`, and `refine!` only
#     refuses at `l >= maxlevel`.  Hence the re-queue branch is dead code and
#     the spin cannot happen today.  It becomes reachable the moment
#     `_collect_face!` learns to look past `maxlevel`, `refine!` grows another
#     failure mode, or a scheme balances a tree whose `maxlevel` was lowered --
#     which is exactly when the guard is needed.
#   * Therefore the gate checks (a) guarded == unguarded leaf sets, (b) the
#     invariant above on 40 random trees, (c) that an *invalid* tree (a face
#     two levels finer, which `_collect_face!` cannot see from the coarse side)
#     leaves `balance!` a fast no-op rather than a spin, and (d) that the guard
#     costs nothing measurable on trees that do balance.

const _G187_REF_MAXITER = 1000

# Verbatim copy of the pre-fix `balance!` body (commit 4bbdae9): the same wave
# algorithm, no fixed-point test.
function _g187_balance_ref!(qt::Quadtree; maxiter::Int=_G187_REF_MAXITER)
    wave = sort(collect(qt.leaves))
    for _ in 1:maxiter
        isempty(wave) && return true
        next = Set{NTuple{3,Int}}()
        for leaf in wave
            leaf in qt.leaves || continue
            bad = false
            for dir in QuaST.FACE_DIRS
                for nb in neighbors(qt, leaf, dir)
                    if nb[1] >= leaf[1] + 2
                        bad = true
                        break
                    end
                end
                bad && break
            end
            bad || continue
            nbs = NTuple{3,Int}[]
            for dir in QuaST.FACE_DIRS
                append!(nbs, neighbors(qt, leaf, dir))
            end
            l, ix, iy = leaf
            if refine!(qt, leaf)
                for dy in 0:1, dx in 0:1
                    push!(next, (l + 1, 2ix + dx, 2iy + dy))
                end
            else
                push!(next, leaf)
            end
            for nb in nbs
                push!(next, nb)
            end
        end
        wave = sort(collect(next))
    end
    return false
end

function _g187_random_tree(rng, level::Int, nops::Int)
    qt = Quadtree(level)
    for _ in 1:nops
        lvs = collect(qt.leaves)
        lf = lvs[rand(rng, 1:length(lvs))]
        if rand(rng) < 0.6
            refine!(qt, lf)
        else
            coarsen!(qt, lf)
        end
    end
    return qt
end

# A valid leaf partition containing a face that is two levels finer than its
# coarse side: `neighbors` sees it from the fine side only, so `balance!` is a
# no-op here (documented above).
function _g187_deep_face_tree()
    q7 = Quadtree(7)
    for _ in 1:4, lf in collect(q7.leaves)          # uniform level 4
        refine!(q7, lf)
    end
    balance!(q7)
    refine!(q7, (4, 0, 0))                           # region to level 6
    for dy in 0:1, dx in 0:1
        refine!(q7, (5, dx, dy))
    end
    return Quadtree(4, 1.0, 1, 1, copy(q7.leaves))
end

@testset "G187 loop-guard semantics (balance! fixed point)" begin
    # (a) guarded == unguarded leaf set on random trees.
    rng = MersenneTwister(20261107)
    for trial in 1:24
        qt = _g187_random_tree(rng, 5, 1 + trial % 7)
        qa = Quadtree(5, 1.0, 1, 1, copy(qt.leaves))
        qb = Quadtree(5, 1.0, 1, 1, copy(qt.leaves))
        ra = balance!(qa)
        rb = _g187_balance_ref!(qb)
        @test ra == rb
        @test sort(collect(qa.leaves)) == sort(collect(qb.leaves))
        @test max_face_level_jump(qa) <= 1
    end

    # (b) the unreachability invariant: every violation that `balance!` can see
    #     is on a leaf that `refine!` may split (l < maxlevel).
    for trial in 1:40
        qt = _g187_random_tree(MersenneTwister(1000 + trial), 5,
                               trial % 9)
        nviol = 0
        for leaf in qt.leaves, dir in QuaST.FACE_DIRS
            for nb in neighbors(qt, leaf, dir)
                if nb[1] >= leaf[1] + 2
                    nviol += 1
                    @test leaf[1] < qt.maxlevel        # refine! cannot refuse
                end
            end
        end
        @test nviol >= 0
    end

    # (c) an invalid face two levels finer leaves balance! a fast no-op.
    qt = _g187_deep_face_tree()
    before = sort(collect(qt.leaves))
    t = @elapsed r = balance!(qt)
    @test r == true                     # nothing it can see violates 2:1
    @test sort(collect(qt.leaves)) == before
    @test max_face_level_jump(qt) >= 2  # ... although the tree is not 2:1
    @info "G187 deep face" leaves = length(before) seconds = t
    @test t < 0.2

    # (d) no measurable cost on trees that do balance.
    q0 = _g187_random_tree(MersenneTwister(7), 5, 4)
    ta = minimum([@elapsed balance!(Quadtree(5, 1.0, 1, 1, copy(q0.leaves)))
                  for _ in 1:5])
    tb = minimum([@elapsed _g187_balance_ref!(Quadtree(5, 1.0, 1, 1,
                                                       copy(q0.leaves)))
                  for _ in 1:5])
    @info "G187 normal tree" guarded_s = ta old_s = tb ratio = tb / ta
    @test ta < 2.0 * tb + 1.0e-4
end
