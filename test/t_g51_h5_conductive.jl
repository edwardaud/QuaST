# G51: the H>0 conductive base state must be steady.
#
# Tbar(z) = 1-z+(H/2)z(1-z) satisfies lap(Tbar) + H = 0, so theta = T-Tbar = 0
# with u = 0 is an exact steady solution.  Adding an explicit +H source to the
# theta equation (instead of relying on the +H already absorbed in Tbar) would
# make theta grow at rate H and shift Nu by H*dt; this gate catches that.
using Test
using LinearAlgebra
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

@testset "G51 H=5 steady conductive state" begin
    pr = Params(Ra=1.0e4, H=5.0, eta_law=:expT, deta_T=1.0, bc=:free)

    qt = uniform_tree(2)
    sq = make_semquad(qt, 2)
    th0 = zeros(Float64, sq.N)
    res = run_semquad_simulation(qt, pr, th0;
                                 tstop=1.0e-3, dt0=1.0e-4, cfl=0.0,
                                 maxpic=1, amr_every=0, maxsteps=1,
                                 linear=:direct, verbose=false)
    @test maximum(abs, res.theta) < 1.0e-12
    @test res.history[1][3] ≈ 3.5 atol=1.0e-12
    @test res.history[1][4] ≈ -1.5 atol=1.0e-12

    m = make_mesh(1.0, 8, 8, 2)
    fd = make_fedata(2, m.q)
    th = zeros(Float64, m.Nscalar)
    r = run_simulation(m, fd, pr; theta0=th, tstop=1.0e-3, dt0=1.0e-4,
                       cfl=0.0, maxsteps=1, maxpic=1, verbose=false)
    @test maximum(abs, r.theta) < 1.0e-12
    @test r.Nu_top ≈ 3.5 atol=1.0e-12
    @test r.Nu_bot ≈ -1.5 atol=1.0e-12
end
