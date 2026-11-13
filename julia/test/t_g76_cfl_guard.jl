# G76: wall-CFL guard for level comparisons.
#
# Different AMR levels have different hmin; comparing them at the same fixed dt
# is only meaningful if the wall CFL is below the threshold.  cfl_tol makes an
# invalid run fail instead of silently producing a time-under-resolved e89.
using Test
using MantleSEM

@testset "G76 wall CFL guard" begin
    qt = wall_graded_tree(2, 5; strip=0.15)
    sq = make_semquad(qt, 2)
    th0 = [0.01 * cospi(sq.coords[i][1]) * sinpi(sq.coords[i][2])
           for i in 1:sq.N]
    pr = Params(Ra=1.0e8, H=5.0, eta_law=:expT, deta_T=1.0e3, bc=:free,
                eta_min=1.0e-6, eta_max=1.0e6)

    @test_throws ErrorException run_semquad_simulation(deepcopy(qt), pr, th0;
                                                       p=2, tstop=1.0e-9,
                                                       dt0=1.0e-6, dtmax=1.0e-6,
                                                       cfl=0.0, maxsteps=1,
                                                       maxpic=1, amr_every=0,
                                                       linear=:direct,
                                                       cfl_tol=1.0e-3,
                                                       advection=:skew,
                                                       verbose=false)
    res = run_semquad_simulation(deepcopy(qt), pr, th0;
                                 p=2, tstop=1.0e-9, dt0=1.0e-12,
                                 dtmax=1.0e-12, cfl=0.0, maxsteps=1,
                                 maxpic=1, amr_every=0, linear=:direct,
                                 cfl_tol=1.0, advection=:skew, verbose=false)
    @test res.steps == 1
end
