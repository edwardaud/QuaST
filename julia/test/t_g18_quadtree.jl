# 2:1 quadtree: area conservation, 2:1 balance, neighbor counts, coarsening.
using Test
using Random
using MantleSEM

@testset "G18 2:1 quadtree" begin
    qt = Quadtree(5)
    @test total_area(qt) ≈ 1.0
    @test refine!(qt, (0, 0, 0))
    @test length(qt.leaves) == 4
    @test total_area(qt) ≈ 1.0
    @test !refine!(qt, (3, 0, 0))             # a leaf not in the tree is ignored
    @test refine!(qt, (1, 0, 0))
    @test balance!(qt)
    @test max_face_level_jump(qt) <= 1
    @test total_area(qt) ≈ 1.0

    # neighbor counts: 1 or 2 on an interior face, 0 on the domain boundary
    leaf = (2, 0, 0)
    @test 1 <= length(neighbors(qt, leaf, :xplus)) <= 2
    corner = (1, 1, 0)
    @test 1 <= length(neighbors(qt, corner, :yplus)) <= 2
    right = (1, 1, 0)
    @test length(neighbors(qt, right, :xplus)) == 0

    # coarsening a complete set of siblings
    qt2 = Quadtree(4)
    refine!(qt2, (0, 0, 0))
    for dy in 0:1, dx in 0:1
        refine!(qt2, (1, dx, dy))
    end
    @test total_area(qt2) ≈ 1.0
    for py in 0:1, px in 0:1
        @test coarsen!(qt2, (2, 2px, 2py))   # merges the four siblings at once
    end
    @test length(qt2.leaves) == 4
    @test total_area(qt2) ≈ 1.0

    # random refine/coarsen cycles preserve the invariants
    rng = MersenneTwister(7)
    qt3 = Quadtree(6)
    for _ in 1:15
        for _ in 1:12
            leaves = collect(qt3.leaves)
            isempty(leaves) && break
            leaf = rand(rng, leaves)
            if rand(rng) < 0.7
                refine!(qt3, leaf)
            else
                coarsen!(qt3, leaf)
            end
        end
        @test balance!(qt3)
        @test total_area(qt3) ≈ 1.0
        @test max_face_level_jump(qt3) <= 1
    end

    # refine_where! honours the predicate
    qt4 = Quadtree(5)
    refine_where!(qt4, (l, ix, iy) -> l < 3 &&
                 abs((ix + 0.5) / (1 << l) - 0.5) < 0.3 &&
                 abs((iy + 0.5) / (1 << l) - 0.5) < 0.3)
    @test max_face_level_jump(qt4) <= 1
    @test total_area(qt4) ≈ 1.0
    @test any(l == 3 for (l, _, _) in qt4.leaves)
end


@testset "G18 face pairing" begin
    # uniform 4x4: 2 * (n-1) * n interior faces, each of length h
    qt = Quadtree(4)
    refine!(qt, (0, 0, 0))
    for dy in 0:1, dx in 0:1
        refine!(qt, (1, dx, dy))
    end
    @test length(qt.leaves) == 16
    pairs = face_pairs(qt)
    @test length(pairs) == 2 * 3 * 4
    @test all(p.len ≈ 0.25 for p in pairs)
    @test all(p.leafL[1] == p.leafR[1] for p in pairs)

    # one refined leaf: the coarse neighbor face is split into two sub-faces
    qt2 = Quadtree(4)
    refine!(qt2, (0, 0, 0))
    refine!(qt2, (1, 0, 0))
    @test balance!(qt2)
    pr = face_pairs(qt2)
    coarse = [p for p in pr if p.leafL[1] < p.leafR[1]]
    @test !isempty(coarse)
    @test all(p.leafL[1] <= p.leafR[1] for p in pr)
    @test all(p.len ≈ 2.0^(-p.leafR[1]) for p in pr)
    # every sub-face lies on the coarse element's face
    for p in coarse
        lL, ixL, iyL = p.leafL
        hL = 2.0^(-lL)
        if p.dir in (:xplus, :xminus)
            @test p.offset >= iyL * hL - 1e-12
            @test p.offset + p.len <= (iyL + 1) * hL + 1e-12
        else
            @test p.offset >= ixL * hL - 1e-12
            @test p.offset + p.len <= (ixL + 1) * hL + 1e-12
        end
    end
end


@testset "G18 mixed 2:1 pairing" begin
    qt = Quadtree(4)
    refine!(qt, (0, 0, 0))
    refine!(qt, (1, 0, 0))
    @test balance!(qt)
    pr = face_pairs(qt)
    # the coarse leaf (1,1,0) has two :xminus sub-faces covering [0,0.5]
    subs = [p for p in pr if p.leafL == (1, 1, 0) && p.dir == :xminus]
    @test length(subs) == 2
    @test sort([p.offset for p in subs]) ≈ [0.0, 0.25]
    @test all(p.len ≈ 0.25 for p in subs)
    # the coarse leaf (1,0,1) has two :yminus sub-faces covering [0,0.5]
    subs2 = [p for p in pr if p.leafL == (1, 0, 1) && p.dir == :yminus]
    @test length(subs2) == 2
    @test sort([p.offset for p in subs2]) ≈ [0.0, 0.25]
    # the fine-fine faces among the refined children must be present
    @test any(p.leafL == (2, 0, 0) && p.leafR == (2, 1, 0) for p in pr)
    @test any(p.leafL == (2, 0, 0) && p.leafR == (2, 0, 1) for p in pr)
    # every pair is normalized with the coarse side first
    @test all(p.leafL[1] <= p.leafR[1] for p in pr)
end
