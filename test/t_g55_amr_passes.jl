# G55: multi-pass AMR on the same time level.
#
# The extreme smoke's interior starts at level 3 and rises only one level per
# time step, while the wall is already level 9-10.  semquad_amr_passes! runs
# several indicator/refine/transfer cycles before the next energy step, so an
# interior plume can catch up within one time step.  This gate uses the same
# wall-layer + interior-Gaussian field as G54 and checks that 3 passes refine
# the hotspot further than 1 pass while preserving 2:1 and the field size.
using Test
using LinearAlgebra
using QuaST

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

@testset "G55 multi-pass AMR" begin
    qt0 = wall_graded_tree(2, 8; strip=0.15)
    ss0 = make_semstokes(qt0, 2)
    sq0 = ss0.su
    theta = [0.10 * exp(-sq0.coords[i][2] / 0.05) +
             0.20 * exp(-((sq0.coords[i][1] - 0.5)^2 +
                          (sq0.coords[i][2] - 0.5)^2) / 0.01)
             for i in 1:sq0.N]
    c0 = center_level(qt0)

    qt1 = deepcopy(qt0)
    ss1 = make_semstokes(qt1, 2)
    th1 = copy(theta)
    ss1, th1, _, nr1 = semquad_amr_passes!(ss1, qt1, th1, copy(th1);
                                           refine_frac=0.2, passes=1)
    c1 = center_level(qt1)

    qt3 = deepcopy(qt0)
    ss3 = make_semstokes(qt3, 2)
    th3 = copy(theta)
    ss3, th3, thp3, nr3 = semquad_amr_passes!(ss3, qt3, th3, copy(th3);
                                              refine_frac=0.2, passes=3)
    c3 = center_level(qt3)

    @test nr1 > 0
    @test nr3 > nr1
    @test c1 > c0
    @test c3 > c1
    @test c3 >= 5
    @test max_face_level_jump(qt3) <= 1
    @test length(th3) == ss3.su.N
    @test length(thp3) == ss3.su.N
    @test all(isfinite, th3)
    @test_throws ArgumentError semquad_amr_passes!(ss3, qt3, th3, thp3; passes=0)
end
