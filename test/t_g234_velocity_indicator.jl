# G234: the AMR indicator can be applied to the velocity field, or to both fields.
using Test, QuaST, LinearAlgebra

# G234 (Phase 1, item 1.1): `amr_indicator` selects which field the gradient-jump
# estimator sees.
#
#   (a) `:theta` is the default and bit-identical to not passing the keyword, so
#       every existing pin keeps its numbers;
#   (b) `:velocity` sums the two components' estimators exactly and is wired into
#       the driver;
#   (c) `:both` is the sum of the two, each normalized by its own total, so it
#       cannot vanish when the flow is at rest (the `:velocity` failure mode);
#   (d) anything else is refused before the run starts.

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
    th = [sinpi(x) * sinpi(z) for (x, z) in sq.coords]
    ux = [sin(8pi * x) * sinpi(z) for (x, z) in sq.coords]
    uz = [cospi(x) * sin(6pi * z) for (x, z) in sq.coords]

    # (a) the estimators themselves
    jv = semquad_velocity_jump(sq, qt, ux, uz)
    jx = semquad_gradient_jump(sq, qt, ux)
    jz = semquad_gradient_jump(sq, qt, uz)
    @test length(jv) == length(sq.leaves)
    @test sum(values(jv)) > 0.0
    @test all(abs.(collect(values(jv)) .- (collect(values(jx)) .+ collect(values(jz)))) .< 1.0e-12)
    @test_throws ArgumentError semquad_velocity_jump(sq, qt, ux[1:end-1], uz)

    # (b) `:theta` is unchanged, and the switch is wired
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0)
    th0 = [0.05 * sinpi(x) * sinpi(z) for (x, z) in sq.coords]
    kw = (p=2, tstop=6.0e-3, dt0=2.0e-3, dtmax=2.0e-3, dtmin=1.0e-16,
          amr_every=0, ev=0.0, linear=:direct, maxpic=1, verbose=false,
          mass_form=:boussinesq)
    a = run_semquad_simulation(qt, pr, copy(th0); kw...)
    b = run_semquad_simulation(qt, pr, copy(th0); amr_indicator=:theta, kw...)
    @test a.theta == b.theta
    @test a.ux == b.ux
    c = run_semquad_simulation(qt, pr, copy(th0); amr_indicator=:velocity,
                               amr_every=1, refine_frac=0.3, coarsen_frac=0.0,
                               max_level=3, max_leaves=4000,
                               wall_min_level=0, wall_strip=0.1, kw...)
    @test all(isfinite, c.theta)

    # (c) the combined indicator: normalized sum, non-vanishing at rest
    jc = semquad_combined_jump(sq, qt, th, ux, uz)
    jt = semquad_gradient_jump(sq, qt, th)
    st = sum(values(jt)); sv = sum(values(jv))
    for lf in keys(jc)
        @test isapprox(jc[lf], 0.5 * jt[lf] + 0.5 * (st / sv) * jv[lf]; rtol=1.0e-12, atol=0.0)
    end
    @test isapprox(sum(values(jc)), sum(values(jt)); rtol=1.0e-10)
    zero_u = zeros(sq.N)
    jc0 = semquad_combined_jump(sq, qt, th, zero_u, zero_u)
    @test isapprox(sum(values(jc0)), 0.5 * st; rtol=1.0e-10)
    @test any(>(0.0), values(jc0))

    # (d) the guard
    for bad in (:bogus, :Theta)
        err = try
            run_semquad_simulation(qt, pr, copy(th0); amr_indicator=bad, kw...)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("amr_indicator", sprint(showerror, err))
    end
end
