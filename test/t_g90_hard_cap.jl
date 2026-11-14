# G90: max_leaves is a hard cap even after wall refinement.
#
# refine_wall_to_min! runs after the AMR cycle and used to push the tree over
# max_leaves (e.g. leaves=1666 > max_leaves=1600 in the guarded run); the hard
# leaf-budget pass after it is what keeps the cap.
using Test
using QuaST

@testset "G90 hard leaf cap after wall refinement" begin
    qt0 = wall_graded_tree(2, 4; strip=0.15)
    n0 = length(qt0.leaves)
    sq0 = make_semquad(qt0, 4)
    theta0 = [0.01 * sinpi(sq0.coords[i][1]) * sinpi(sq0.coords[i][2])
              for i in 1:sq0.N]
    pr = Params(Ra=1.0e4, H=5.0, eta_law=:expT, deta_T=1.0, bc=:free)

    # This gate pins the hard max_leaves cap.
    res = run_semquad_simulation(deepcopy(qt0), pr, theta0;
                                 p=4, tstop=1.0, dt0=1.0e-9, dtmax=1.0e-9,
                                 cfl=0.0, maxsteps=1, maxpic=1,
                                 amr_every=1, refine_frac=0.2,
                                 max_leaves=n0, wall_min_level=4,
                                 max_level=7, linear=:schur_cg,
                                 schur_maxiter=100, advection=:skew,
                                 verbose=false)
    @test length(res.qt.leaves) <= n0

    res2 = run_semquad_simulation(deepcopy(qt0), pr, theta0;
                                  p=4, tstop=1.0, dt0=1.0e-9, dtmax=1.0e-9,
                                  cfl=0.0, maxsteps=0, maxpic=1,
                                  initial_refine=1, refine_frac=0.2,
                                  max_leaves=n0, wall_min_level=4,
                                  max_level=7, linear=:schur_cg,
                                  schur_maxiter=100, advection=:skew,
                                  verbose=false)
    @test length(res2.qt.leaves) <= n0
end
