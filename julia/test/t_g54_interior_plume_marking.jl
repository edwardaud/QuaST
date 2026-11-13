# G54: AMR marking must track interior plumes, not only the wall.
#
# The extreme smoke's level histogram is wall_max=10, interior_max=4: the wall
# consumes almost all of the global top-20% budget.  This gate builds a wall-
# graded tree with an additional interior Gaussian hotspot and checks that the
# level-balanced marking (`marking=:level`) refines both the wall and the
# hotspot, preserves 2:1, and is accepted by the AMR cycle.
using Test
using MantleSEM

function center_level(qt)
    lv = 0
    for lf in qt.leaves
        h = 2.0^(-lf[1]); x0 = lf[2] * h; y0 = lf[3] * h
        if x0 < 0.6 && x0 + h > 0.4 && y0 < 0.6 && y0 + h > 0.4
            lv = max(lv, lf[1])
        end
    end
    return lv
end

@testset "G54 interior plume marking" begin
    qt0 = wall_graded_tree(2, 8; strip=0.15)
    sq = make_semquad(qt0, 2)
    theta = [0.10 * exp(-sq.coords[i][2] / 0.05) +
             0.20 * exp(-((sq.coords[i][1] - 0.5)^2 +
                          (sq.coords[i][2] - 0.5)^2) / 0.01)
             for i in 1:sq.N]
    eta = semquad_gradient_jump(sq, qt0, theta)
    c0 = center_level(qt0)
    w0 = maximum(lf[1] for lf in qt0.leaves if lf[3] == 0)

    for marking in (:global, :level)
        qt = deepcopy(qt0)
        nr, _ = amr_cycle!(qt, eta; refine_frac=0.2, marking=marking)
        @test nr > 0
        @test length(qt.leaves) > length(qt0.leaves)
        @test max_face_level_jump(qt) <= 1
        @test center_level(qt) > c0       # the hotspot is refined
        @test maximum(lf[1] for lf in qt.leaves if lf[3] == 0) > w0
    end
end
