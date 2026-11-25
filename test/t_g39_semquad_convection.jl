# G39: coupled quadtree convection loop with periodic AMR.
#
# A small Rayleigh-Benard case at Ra=1e4, Delta eta=1e2, free slip, H=0.  The
# adaptive run (AMR every 5 steps) and the static run must stay finite, satisfy
# the 2:1 balance, and agree on Nu_top/Vrms at the same physical time.  This is
# the mechanical acceptance test for the coupled loop before the high-Ra run.
using Test
using LinearAlgebra
using QuaST

function uniform_tree(level::Int)
    qt = Quadtree(level + 3)
    for _ in 1:level
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
    end
    return qt
end

@testset "G39 coupled quadtree convection + AMR" begin
    L = 3
    qt_ad = uniform_tree(L)
    qt_st = uniform_tree(L)
    sq = make_semquad(qt_ad, 2)
    theta0 = [0.01 * cospi(sq.coords[i][1]) * sinpi(sq.coords[i][2]) for i in 1:sq.N]
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0e2, bc=:free)

    # Thresholds calibrated on the balanced policy.
    res_ad = run_semquad_simulation(qt_ad, pr, theta0;
                                    tstop=1.0e-3, dt0=1.0e-4,
                                    amr_every=5, refine_frac=0.2,
                                    verbose=false)
    res_st = run_semquad_simulation(qt_st, pr, theta0;
                                    tstop=1.0e-3, dt0=1.0e-4,
                                    amr_every=0, verbose=false)

    @test length(res_ad.history) == 10
    @test length(res_st.history) == 10
    @test all(isfinite, vcat(res_ad.history...))
    @test max_face_level_jump(res_ad.qt) <= 1
    @test length(res_ad.qt.leaves) >= length(qt_st.leaves)

    Nu_ad = res_ad.history[end][3]
    Nu_st = res_st.history[end][3]
    Vr_ad = res_ad.history[end][5]
    Vr_st = res_st.history[end][5]
    @test abs(Nu_ad - Nu_st) / max(abs(Nu_st), 1.0) < 0.2
    @test abs(Vr_ad - Vr_st) / max(abs(Vr_st), 1.0e-30) < 0.2
end
