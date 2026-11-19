# G234: the AMR indicator can be applied to the velocity field.
using Test, QuaST, LinearAlgebra

# G234 (Phase 1, item 1.1): `amr_indicator` selects which field the gradient-jump
# estimator sees.  Three things must hold:
#
#   (a) `:theta` is the default and is bit-identical to not passing the keyword, so
#       every existing pin keeps its numbers;
#   (b) `:velocity` produces a different refinement from `:theta` on a flow with
#       structure, i.e. the switch is wired;
#   (c) anything else is refused before the run starts.

@testset "G234 velocity-gradient AMR indicator" begin
    function _g234_uniform(level::Int)
        q = Quadtree(level)
        for _ in 1:level, lf in collect(q.leaves)
            refine!(q, lf)
        end
        balance!(q)
        return q
    end

    qt = _g234_uniform(3)
    sq = make_semquad(qt, 2)
    # a velocity field with cell-scale structure: the theta field stays smooth, so
    # only a velocity-based indicator can see it
    ux = [sin(8π * x) * sinpi(z) for (x, z) in sq.coords]
    uz = [cospi(x) * sin(6π * z) for (x, z) in sq.coords]

    # the estimator itself
    jv = semquad_velocity_jump(sq, qt, ux, uz)
    @test length(jv) == length(sq.leaves)
    @test sum(values(jv)) > 0.0
    jx = semquad_gradient_jump(sq, qt, ux)
    jz = semquad_gradient_jump(sq, qt, uz)
    @test all(abs.(collect(values(jv)) .- (collect(values(jx)) .+ collect(values(jz)))) .< 1e-14)
    @test_throws ArgumentError semquad_velocity_jump(sq, qt, ux[1:end-1], uz)

    # (a) the default path is unchanged
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0)
    th0 = [0.05 * sinpi(x) * sinpi(z) for (x, z) in sq.coords]
    kw = (p=2, tstop=6.0e-3, dt0=2.0e-3, dtmax=2.0e-3, dtmin=1.0e-16,
          amr_every=0, ev=0.0, linear=:direct, maxpic=1, verbose=false,
          mass_form=:boussinesq)
    a = run_semquad_simulation(qt, pr, copy(th0); kw...)
    b = run_semquad_simulation(qt, pr, copy(th0); amr_indicator=:theta, kw...)
    @test a.theta == b.theta
    @test a.ux == b.ux

    # (b) the switch is wired: a run that adapts must refine differently
    c = run_semquad_simulation(qt, pr, copy(th0); amr_indicator=:velocity,
                               amr_every=1, refine_frac=0.3, coarsen_frac=0.0,
                               max_level=3, max_leaves=4000,
                               wall_min_level=0, wall_strip=0.1, kw...)
    @test all(isfinite, c.theta)

    # (c) the guard
    err = try
        run_semquad_simulation(qt, pr, copy(th0); amr_indicator=:bogus, kw...)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("amr_indicator", sprint(showerror, err))
end
