using Test, MantleSEM

@testset "G118 Picard strain-rate averaging and divergence monitor" begin
    # (1) the nodal strain-rate field must match the pointwise evaluator for a
    # field whose invariant is constant: u = (a x + b y, c y)
    qt = Quadtree(4)
    for _ in 1:3, lf in collect(qt.leaves); refine!(qt, lf); end
    balance!(qt)
    sq = make_semquad(qt, 4)
    leaves = Set(qt.leaves)
    a, b, c = 0.3, -0.2, 0.7
    ux = [a * x + b * y for (x, y) in sq.coords]
    uz = [c * y for (x, y) in sq.coords]
    ed_field = semquad_strain_rate_field(sq, ux, uz)
    ed_exact = sqrt(0.5 * (a^2 + c^2 + 2.0 * (0.5 * b)^2))
    @test all(abs.(ed_field .- ed_exact) .< 1.0e-12)
    for (x, y) in ((0.2, 0.3), (0.8, 0.1), (0.55, 0.65))
        @test semquad_strain_rate_at(sq, ux, uz, x, y, leaves) ≈ ed_exact atol=1.0e-12
    end

    # (2) blending: relax = 1 returns the new field exactly (bit-identical path)
    p0 = [1.0, 2.0, 3.0]
    p1 = [3.0, 4.0, 5.0]
    @test blend_field(p0, p1, 1.0) === p1
    @test blend_field(p0, p1, 0.0) === p0
    @test blend_field(p0, p1, 0.5) == [2.0, 3.0, 4.0]
    @test_throws ArgumentError blend_field(p0, p1, 1.5)
    @test_throws ArgumentError blend_field(p0, [1.0, 2.0], 0.5)

    # (3) divergence monitor: growing sequence fires, decaying/steady do not
    growing = [1.0e-3, 1.0e-3, 1.0e-2, 1.0e-1, 1.0]
    @test picard_diverging(growing; window=3, factor=5.0)
    decaying = [1.0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4]
    @test !picard_diverging(decaying; window=3, factor=5.0)
    steady = fill(1.0e-4, 6)
    @test !picard_diverging(steady; window=3, factor=5.0)
    @test !picard_diverging([1.0, 2.0]; window=3, factor=5.0)   # too short
    @test_throws ArgumentError picard_diverging(growing; window=0)
    @test_throws ArgumentError picard_diverging(growing; window=2, factor=1.0)

    # (4) the nodal-field eta overload is a different code path: scaling the
    # field must change eta when the plastic branch is actually active
    pr = Params(Ra=1.0e6, H=0.0, eta_law=:arrhenius, eta_E=3.0, eta_T0=0.5,
                eta_min=1.0e-9, yield_c=1.0e-3)
    sp = make_semquad(qt, 1)                     # p-1 = 1 pressure space
    theta = [0.1 * sinpi(x) * sinpi(z) for (x, z) in sq.coords]
    pv = zeros(sp.N)
    e_base, e_big = 10.0, 100.0
    eta_small = semquad_eta_at(sq, theta, fill(e_big, sq.N), pv, sp,
                               0.5, 0.5, pr, leaves)
    eta_large = semquad_eta_at(sq, theta, fill(e_base, sq.N), pv, sp,
                               0.5, 0.5, pr, leaves)
    @test eta_small < eta_large                    # more strain rate -> weaker
    eta_pt = semquad_eta_at(sq, theta, ux, uz, pv, sp, 0.5, 0.5, pr, leaves)
    @test isfinite(eta_pt)

    # (5) driver: default pic_eta_relax=1 is bit-identical to the plain yielding
    # path; 0.5 takes the averaging branch, stays finite and gives a different
    # solution when yielding is active
    function run_case(; kw...)
        q = Quadtree(5)
        for _ in 1:3, lf in collect(q.leaves); refine!(q, lf); end
        balance!(q)
        ss = make_semstokes(q, 2)
        theta0 = [0.1 * sinpi(x) * sinpi(z) for (x, z) in ss.su.coords]
        prr = Params(Ra=1.0e6, H=0.0, eta_law=:arrhenius, eta_E=3.0,
                     eta_T0=0.5, eta_min=1.0e-9, yield_c=1.0e-3)
        dt = 1.0e-5
        run_semquad_simulation(q, prr, theta0;
            p=2, tstop=4 * dt, dt0=dt, dtmax=dt, dtmin=1.0e-16,
            amr_every=0, ev=0.0, advection=:skew, linear=:direct,
            maxpic=2, pic_tol=1.0e-12, verbose=false, kw...)
    end
    r_def = run_case()
    r_one = run_case(pic_eta_relax=1.0)
    @test maximum(abs, r_def.theta - r_one.theta) == 0.0
    @test r_def.history == r_one.history
    r_half = run_case(pic_eta_relax=0.5)
    @test all(isfinite, r_half.theta)
    @test all(isfinite, vcat(r_half.history...))
    @test maximum(abs, r_half.theta - r_def.theta) > 0.0   # path is active

    # (6) the monitor does not fire on a healthy run, and does fire when the
    # threshold is absurdly tight relative to the actual Picard sequence
    @test run_case(pic_div_window=3, pic_div_factor=50.0).steps == 4
    @test_throws ErrorException run_case(pic_div_window=1,
                                         pic_div_factor=1.0 + 1.0e-12,
                                         pic_relax=0.5)
end
