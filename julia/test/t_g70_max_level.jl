# G70: AMR max_level cap.
#
# The level7/8/9 self-convergence needs a controlled maximum level; without it
# dynamic refinement can overshoot the intended wall level by 2:1 balancing.
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

@testset "G70 AMR max_level" begin
    qt = uniform_tree(2)
    sq = make_semquad(qt, 2)
    theta = [0.05 * sinpi(sq.coords[i][1]) * sinpi(sq.coords[i][2])
             for i in 1:sq.N]
    eta = semquad_gradient_jump(sq, qt, theta)
    for _ in 1:4
        amr_cycle!(qt, eta; refine_frac=0.5, coarsen_frac=0.0, max_level=3)
        sq = make_semquad(qt, 2)
        theta = [0.05 * sinpi(sq.coords[i][1]) * sinpi(sq.coords[i][2]) for i in 1:sq.N]
        eta = semquad_gradient_jump(sq, qt, theta)
    end
    @test maximum(lf[1] for lf in qt.leaves) <= 3
    @test max_face_level_jump(qt) <= 1

    qt2 = uniform_tree(2)
    sq2 = make_semquad(qt2, 2)
    theta = [0.05 * sinpi(sq2.coords[i][1]) * sinpi(sq2.coords[i][2]) for i in 1:sq2.N]
    eta2 = semquad_gradient_jump(sq2, qt2, theta)
    for _ in 1:4
        amr_cycle!(qt2, eta2; refine_frac=0.5, coarsen_frac=0.0)
        sq2 = make_semquad(qt2, 2)
        theta = [0.05 * sinpi(sq2.coords[i][1]) * sinpi(sq2.coords[i][2]) for i in 1:sq2.N]
        eta2 = semquad_gradient_jump(sq2, qt2, theta)
    end
    @test maximum(lf[1] for lf in qt2.leaves) > 3

    @test_throws ArgumentError amr_cycle!(uniform_tree(2), eta;
                                          refine_frac=0.5, max_level=0)
end
