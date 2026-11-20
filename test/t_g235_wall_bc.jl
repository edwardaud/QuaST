# G235: per-wall velocity conditions (phase 1, item 1.3, step 1).
using Test, QuaST, LinearAlgebra

# The tuple is (xminus, xplus, yminus, yplus) and `:same` follows `bc`, so the
# historical single-`bc` path must be bit-identical.  A mixed tuple is refused
# with anything but `linear=:direct` until the multigrid carries it.

@testset "G235 per-wall velocity conditions" begin
    @test Params().wall_bc == (:same, :same, :same, :same)
    @test Params(bc=:rigid).wall_bc == (:same, :same, :same, :same)
    @test Params(wall_bc=(:rigid, :free, :free, :rigid)).wall_bc ==
          (:rigid, :free, :free, :rigid)
    @test_throws ArgumentError Params(wall_bc=(:free, :free, :free, :bogus))
    @test_throws Exception Params(wall_bc=(:free, :free, :free))
    # a 3-tuple is rejected by the type, an unknown symbol by the check

    function _run(bc; wall_bc=(:same, :same, :same, :same), linear=:direct)
        qt = Quadtree(3)
        for _ in 1:3, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        sq = make_semquad(qt, 2)
        th = [0.05 * sinpi(z) * cospi(x) for (x, z) in sq.coords]
        pr = Params(Ra=1.0e3, H=0.0, eta_law=:expT, deta_T=1.0,
                    bc=bc, wall_bc=wall_bc)
        run_semquad_simulation(qt, pr, th; p=2, maxsteps=2, tstop=4.0e-2,
                               dt0=2.0e-2, dtmax=2.0e-2, dtmin=1.0e-18,
                               amr_every=0, ev=0.0, linear=linear, maxpic=1,
                               verbose=false, mass_form=:boussinesq)
    end

    # (a) the default tuple keeps `bc=:rigid` bit-identical
    a = _run(:rigid)
    b = _run(:rigid; wall_bc=(:same, :same, :same, :same))
    @test a.theta == b.theta
    @test a.ux == b.ux

    # (b) an all-:free tuple on a :rigid base is the :free case
    c = _run(:rigid; wall_bc=(:free, :free, :free, :free))
    f = _run(:free)
    @test c.ux == f.ux
    @test c.uz == f.uz

    # (c) a mixed tuple is a different problem, and it is refused on the multigrid
    d = _run(:rigid; wall_bc=(:rigid, :free, :free, :free))
    @test all(isfinite, d.ux)
    @test d.ux != a.ux
    @test d.ux != f.ux
    @test_throws ArgumentError _run(:rigid; wall_bc=(:rigid, :free, :free, :free),
                                    linear=:mg)
end
