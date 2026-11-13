# G45: the first energy step must already respect the CFL, not only dt0.
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

@testset "G45 initial-step CFL" begin
    qt = uniform_tree(2)
    sq = make_semquad(qt, 2)
    theta0 = [0.01 * cospi(sq.coords[i][1]) * sinpi(sq.coords[i][2]) for i in 1:sq.N]
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0e2, bc=:free)
    res = run_semquad_simulation(qt, pr, theta0;
                                 tstop=1.0e-4, dt0=1.0e-3,
                                 cfl=0.1, dtmax=1.0e-3, dtmin=1.0e-14,
                                 maxpic=1, amr_every=0, maxsteps=1,
                                 linear=:direct, verbose=false)
    @test length(res.history) == 1
    @test res.history[1][2] > 0.0
    @test res.history[1][2] < 1.0e-3
end
