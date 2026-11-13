# G62: AMR coarsening in the SEM cycle.
#
# The low-Ra long run over-refined the whole interior to level 5 with
# coarsen_frac=0.  This gate checks that coarsen_frac>0 actually merges
# complete sibling groups, decreases the leaf count, and preserves 2:1.
using Test
using MantleSEM

function uniform_tree(level::Int)
    qt = Quadtree(level + 2)
    for _ in 1:level
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
    end
    return qt
end

@testset "G62 AMR coarsening" begin
    qt = uniform_tree(3)
    sq = make_semquad(qt, 2)
    theta = [0.05 * sinpi(sq.coords[i][1]) * sinpi(sq.coords[i][2])
             for i in 1:sq.N]
    eta = semquad_gradient_jump(sq, qt, theta)
    n0 = length(qt.leaves)

    nr, nc = amr_cycle!(qt, eta; refine_frac=0.0, coarsen_frac=1.0)
    @test nc > 0
    @test length(qt.leaves) < n0
    @test max_face_level_jump(qt) <= 1

    qt2 = uniform_tree(3)
    sq2 = make_semquad(qt2, 2)
    eta2 = semquad_gradient_jump(sq2, qt2, theta)
    n2 = length(qt2.leaves)
    nr2, nc2 = amr_cycle!(qt2, eta2; refine_frac=0.0, coarsen_frac=0.0)
    @test nc2 == 0
    @test length(qt2.leaves) == n2
    @test max_face_level_jump(qt2) <= 1
end
