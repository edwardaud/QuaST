# G240: a mixed per-wall condition survives an AMR cycle.
#
# G235 pins the mixed tuple on a fixed mesh and G236 pins the constraint set; this
# gate pins their *composition*, which is what production does: the tree changes
# under a run whose walls carry different conditions, and the velocity is transferred
# between two trees whose constraint sets differ.  That composition is exactly where
# a silent regression can hide, because each half is green on its own.

using Test, QuaST

@testset "G240 mixed walls through an AMR cycle" begin
    function _run(wall_bc)
        qt = Quadtree(2)
        for _ in 1:2, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        ss = make_semstokes(qt, 2)
        sq = ss.su
        th = [0.05 * sinpi(z) * cospi(2.0 * x / 2.5) for (x, z) in sq.coords]
        pr = Params(Ra=1.0e3, H=0.0, eta_law=:expT, deta_T=1.0,
                    bc=:rigid, wall_bc=wall_bc)
        res = run_semquad_simulation(qt, pr, th; p=2, maxsteps=4, tstop=4.0e-2,
                                     dt0=1.0e-2, dtmax=1.0e-2, dtmin=1.0e-18,
                                     amr_every=1, refine_frac=0.3, coarsen_frac=0.0,
                                     max_level=4, max_leaves=4000,
                                     wall_min_level=0, wall_strip=0.1,
                                     linear=:direct, maxpic=1, verbose=false,
                                     mass_form=:boussinesq)
        return res
    end

    uni = _run((:same, :same, :same, :same))          # all rigid, as bc says
    mix = _run((:rigid, :rigid, :free, :free))        # y-walls free

    # the run completed and stayed finite across the cycles
    @test mix.steps == 4
    @test all(isfinite, mix.theta)
    @test all(isfinite, mix.ux)
    @test all(isfinite, mix.uz)

    # the cycles fired: the final tree is not the starting one
    @test length(mix.qt.leaves) != 64

    # and the mixed condition still changes the answer after the transfer
    @test mix.ux != uni.ux
end
