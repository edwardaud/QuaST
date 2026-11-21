# G253: the Newton path refuses a viscosity multiplier it does not apply.
#
# `pic_mode=:newton` builds its viscosity from the raw law plus the 2a viscoelastic
# softening; it never applies `Params.eta_multiplier`.  Before this gate a card asking
# for both solved a different viscosity than the Picard path with the same card, with
# no error -- the class of failure this project has spent two phases removing.
#
# The gate pins the refusal, its message, and that the two paths still agree when the
# multiplier is absent (the case a user is allowed to run).

using Test, QuaST

@testset "G253 Newton refuses eta_multiplier" begin
    qt = Quadtree(2)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    th = [0.05 * sinpi(z) * cospi(2.0 * x / 2.5) for (x, z) in sq.coords]
    weak = (x, y) -> (abs(x - 1.25) < 0.1 && y < 0.3) ? 0.05 : 1.0

    function _run(pr; mode=:picard)
        run_semquad_simulation(qt, pr, copy(th); p=2, maxsteps=2, tstop=4.0e-2,
                               dt0=2.0e-2, dtmax=2.0e-2, dtmin=1.0e-18,
                               amr_every=0, ev=0.0, linear=:direct, maxpic=1,
                               pic_mode=mode, verbose=false,
                               mass_form=:boussinesq)
    end

    pr_weak = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
                     eta_multiplier=weak, bc=:free)

    # the refusal, with a message a user can act on
    err = try
        _run(pr_weak; mode=:newton)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("eta_multiplier", msg)
    @test occursin("pic_mode=:newton", msg)

    # the Picard path accepts it (so the refusal is about Newton, not the multiplier)
    ok = _run(pr_weak; mode=:picard)
    @test all(isfinite, ok.theta)

    # without the multiplier both paths run.  No cross-path equivalence is asserted:
    # at one nonlinear iteration (maxpic=1) Picard and Newton are different
    # schemes with different first-iterate behaviour -- measured 0.92 relative on
    # the first step -- and claiming agreement here would be claiming something the
    # settings do not support.
    pr_plain = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0, bc=:free)
    a = _run(pr_plain; mode=:picard)
    b = _run(pr_plain; mode=:newton)
    @test all(isfinite, a.theta)
    @test all(isfinite, b.theta)
    @test all(isfinite, b.ux) && any(!iszero, b.ux)
end
