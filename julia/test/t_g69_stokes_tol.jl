# G69: Stokes tolerance guard for high-Ra AMR.
#
# The Ra=1e8 dynamic runs with schur_maxiter=300 hit the iteration cap at
# residual 1e-3; with 2000 iterations the same step converges to 1e-9 in
# 572/664 iterations.  This gate checks that stokes_tol catches the
# unconverged case and that a larger schur_maxiter passes.
using Test
using MantleSEM

@testset "G69 Stokes tolerance" begin
    qt = wall_graded_tree(2, 6; strip=0.15)
    sq = make_semquad(qt, 2)
    theta0 = [0.01 * cospi(sq.coords[i][1]) * sinpi(sq.coords[i][2])
              for i in 1:sq.N]
    pr = Params(Ra=1.0e8, H=5.0, eta_law=:expT, deta_T=1.0e3, bc=:free,
                eta_min=1.0e-6, eta_max=1.0e6)

    @test_throws ErrorException run_semquad_simulation(deepcopy(qt), pr, theta0;
                                                       p=2, tstop=1.0e-9,
                                                       dt0=1.0e-10, cfl=0.0,
                                                       maxsteps=1, maxpic=1,
                                                       amr_every=0,
                                                       linear=:schur_cg,
                                                       schur_maxiter=5,
                                                       stokes_tol=1.0e-6,
                                                       advection=:skew,
                                                       verbose=false)

    res = run_semquad_simulation(deepcopy(qt), pr, theta0;
                                 p=2, tstop=1.0e-9, dt0=1.0e-10, cfl=0.0,
                                 maxsteps=1, maxpic=1, amr_every=0,
                                 linear=:schur_cg, schur_maxiter=1000,
                                 stokes_tol=1.0e-6, advection=:skew,
                                 verbose=false)
    @test res.steps == 1
    @test all(isfinite, res.history[end])
end
