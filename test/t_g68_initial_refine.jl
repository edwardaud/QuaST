# G68: initial AMR pre-refinement with analytic re-sampling.
#
# Fixes the high-Ra protocol bug where the AMR run started from a coarse n16
# state: with initial_refine>0 and theta_ic, the tree is refined on the initial
# condition before the first step and theta is re-sampled analytically on every
# new tree (not transferred from the coarse field).
using Test
using QuaST

function uniform_tree(level::Int)
    qt = Quadtree(level + 2)
    for _ in 1:level
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
    end
    return qt
end

@testset "G68 initial AMR refinement" begin
    qt0 = wall_graded_tree(2, 4; strip=0.15)
    n0 = length(qt0.leaves)
    sq0 = make_semquad(qt0, 2)
    theta0 = [0.05 * sinpi(sq0.coords[i][1]) * sinpi(sq0.coords[i][2])
              for i in 1:sq0.N]
    pr = Params(Ra=1.0e7, H=5.0, eta_law=:expT, deta_T=1.0e3, bc=:free)

    ic = (x, y) -> 0.05 * sinpi(x) * sinpi(y)
    res = run_semquad_simulation(deepcopy(qt0), pr, theta0;
                                 p=2, tstop=1.0e-9, dt0=1.0e-9,
                                 maxsteps=0, initial_refine=2,
                                 refine_frac=0.5, coarsen_frac=0.0,
                                 theta_ic=ic, linear=:direct,
                                 verbose=false)
    @test length(res.qt.leaves) > n0
    @test res.ss.su.N > sq0.N
    @test res.steps == 0
    err_ic = maximum(abs(res.theta[g] - ic(x, y))
                     for (g, (x, y)) in enumerate(res.ss.su.coords))
    @test err_ic < 1.0e-12
    @test max_face_level_jump(res.qt) <= 1

    qt1 = deepcopy(qt0)
    sq1 = make_semquad(qt1, 2)
    th1 = [0.05 * sinpi(sq1.coords[i][1]) * sinpi(sq1.coords[i][2])
           for i in 1:sq1.N]
    res0 = run_semquad_simulation(qt1, pr, th1;
                                  p=2, tstop=1.0e-9, dt0=1.0e-9,
                                  maxsteps=0, initial_refine=0,
                                  linear=:direct, verbose=false)
    @test length(res0.qt.leaves) == length(qt0.leaves)
    @test res0.ss.su.N == sq1.N
    @test res0.theta ≈ th1

    res_t = run_semquad_simulation(deepcopy(qt0), pr, theta0;
                                   p=2, tstop=1.0e-9, dt0=1.0e-9,
                                   maxsteps=0, initial_refine=1,
                                   refine_frac=0.3, linear=:direct,
                                   verbose=false)
    @test length(res_t.qt.leaves) > n0
    @test all(isfinite, res_t.theta)

    @test_throws ArgumentError run_semquad_simulation(deepcopy(qt0), pr, theta0;
                                                      p=2, tstop=1.0e-9,
                                                      dt0=1.0e-9, maxsteps=0,
                                                      initial_refine=-1,
                                                      linear=:direct)
end
