# G59: AMR DOF budget (max_leaves).
#
# The low-Ra H=5 AMR run refined 20% every step and grew from 1.3e4 to 1.7e5
# DOF in nine steps, so it could not be compared with a static reference at a
# fixed DOF budget.  max_leaves stops the AMR cycle once the tree reaches that
# number of leaves.
using Test
using QuaST

@testset "G59 AMR leaf budget" begin
    qt0 = wall_graded_tree(2, 4; strip=0.15)
    n0 = length(qt0.leaves)
    sq0 = make_semquad(qt0, 4)
    theta0 = [0.01 * sinpi(sq0.coords[i][1]) * sinpi(sq0.coords[i][2]) for i in 1:sq0.N]
    pr = Params(Ra=1.0e4, H=5.0, eta_law=:expT, deta_T=1.0, bc=:free)

    qt1 = deepcopy(qt0)
    # This gate pins the classic hard max_leaves cap (no budget exchange); keep it classic.
    res1 = run_semquad_simulation(qt1, pr, theta0;
                                  p=4, tstop=1.0, dt0=1.0e-9, dtmax=1.0e-9,
                                  cfl=0.0, maxsteps=1, maxpic=1,
                                  amr_every=1, refine_frac=0.2,
                                  amr_policy=:classic,
                                  max_leaves=n0, linear=:schur_cg,
                                  schur_maxiter=100, advection=:skew,
                                  verbose=false)
    @test length(res1.qt.leaves) == n0

    qt2 = deepcopy(qt0)
    res2 = run_semquad_simulation(qt2, pr, theta0;
                                  p=4, tstop=1.0, dt0=1.0e-9, dtmax=1.0e-9,
                                  cfl=0.0, maxsteps=1, maxpic=1,
                                  amr_every=1, refine_frac=0.2,
                                  amr_policy=:classic,
                                  max_leaves=n0 + 100, linear=:schur_cg,
                                  schur_maxiter=100, advection=:skew,
                                  verbose=false)
    @test length(res2.qt.leaves) > n0

    @test_throws ArgumentError run_semquad_simulation(deepcopy(qt0), pr, theta0;
                                                      p=4, tstop=1.0, dt0=1.0e-9,
                                                      dtmax=1.0e-9, cfl=0.0,
                                                      maxsteps=0, amr_every=0,
                                                      max_leaves=0,
                                                      linear=:schur_cg)
end
