# G40: CFL control and Picard options in the quadtree loop.
using Test
using MantleSEM

function uniform_tree(level::Int)
    qt = Quadtree(level + 3)
    for _ in 1:level
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
    end
    return qt
end

@testset "G40 quadtree CFL + Picard options" begin
    qt = uniform_tree(2)
    sq = make_semquad(qt, 2)
    theta0 = [0.01 * cospi(sq.coords[i][1]) * sinpi(sq.coords[i][2]) for i in 1:sq.N]
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0e2, bc=:free)
    res = run_semquad_simulation(qt, pr, theta0;
                                 tstop=1.0e-3, dt0=1.0e-5,
                                 cfl=20.0, dtmax=1.0e-4, dtmin=1.0e-14,
                                 maxpic=2, pic_tol=1.0e-4,
                                 amr_every=0, maxsteps=10, verbose=false)
    @test length(res.history) == 10
    @test all(isfinite, vcat(res.history...))
    @test all(h[2] > 0.0 for h in res.history)
end
