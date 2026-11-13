# G56: the H=5 extreme smoke is stable with one Picard iteration.
#
# On the wall6 tree (N~2.7k), two steps with maxpic=1 give Vrms~7.9e6 and
# theta_max~0.024, while maxpic=3 amplifies to Vrms~1.8e7 and theta~0.11
# (the unrelaxed Picard fixed point is unstable here).  This gate locks the
# maxpic=1 baseline used for the H=5 extreme runs and checks that pic_relax is
# validated.
using Test
using LinearAlgebra
using QuaST

@testset "G56 H=5 Picard baseline" begin
    qt0 = wall_graded_tree(2, 6; strip=0.15)
    sq0 = make_semquad(qt0, 2)
    theta0 = [0.01 * cospi(sq0.coords[i][1]) * sinpi(sq0.coords[i][2]) for i in 1:sq0.N]
    pr = Params(Ra=1.0e8, H=5.0, eta_law=:expT, deta_T=1.0e3, bc=:free,
                eta_min=1.0e-6, eta_max=1.0e6)

    qt = deepcopy(qt0)
    # Policy-agnostic gate calibrated on classic; pin the policy explicitly.
    res = run_semquad_simulation(qt, pr, theta0;
                                 tstop=1.9e-9, dt0=1.0e-9, dtmax=1.0e-6,
                                 cfl=20.0, dtmin=1.0e-12,
                                 maxpic=1, pic_tol=1.0e-6, pic_relax=1.0,
                                 amr_every=1, refine_frac=0.2, coarsen_frac=0.0,
                                 amr_policy=:classic,
                                 maxsteps=2, advection=:conservative,
                                 linear=:schur_cg, schur_maxiter=300,
                                 verbose=false)
    @test res.steps == 2
    @test all(isfinite, res.history[end])
    @test res.history[end][5] < 1.2e7       # Vrms, un-amplified baseline
    @test maximum(abs, res.theta) < 0.04    # theta stays O(0.02), not O(0.1)

    @test_throws ArgumentError run_semquad_simulation(deepcopy(qt0), pr, theta0;
                                                      tstop=1.0e-9, dt0=1.0e-9,
                                                      maxpic=1, pic_relax=0.0,
                                                      maxsteps=1, amr_every=0,
                                                      linear=:direct)
end
