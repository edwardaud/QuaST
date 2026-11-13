using Test, MantleSEM

@testset "G121 surface velocity diagnostics" begin
    function build(qt)
        balance!(qt)
        return make_semquad(qt, 4), Set(qt.leaves)
    end

    # (1) uniform tangential velocity: RMS = max = 1
    qt = Quadtree(4)
    for _ in 1:3, lf in collect(qt.leaves); refine!(qt, lf); end
    sq, leaves = build(qt)
    ux = ones(sq.N); uz = zeros(sq.N)
    r, m = semquad_surface_velocity(sq, qt, ux, uz; wall=:top)
    @test r ≈ 1.0 atol=1e-14
    @test m ≈ 1.0 atol=1e-14

    # (2) ux = x^2: int_0^1 x^4 dx = 1/5 -> RMS = 1/sqrt(5), max = 1.
    # p=4 GLL quadrature is exact for degree 7, so this is an exact check.
    ux = [x^2 for (x, y) in sq.coords]
    uz = zeros(sq.N)
    r, m = semquad_surface_velocity(sq, qt, ux, uz; wall=:top)
    @test r ≈ 1.0 / sqrt(5.0) atol=1e-13
    @test m ≈ 1.0 atol=1e-13

    # (3) the same field on a 2:1 mesh (several wall leaves at different levels)
    qt2 = Quadtree(5)
    for _ in 1:2, lf in collect(qt2.leaves); refine!(qt2, lf); end
    refine!(qt2, (2, 1, 1)); refine!(qt2, (3, 3, 3))
    balance!(qt2)
    sq2 = make_semquad(qt2, 4)
    ux2 = [x^2 for (x, y) in sq2.coords]
    uz2 = zeros(sq2.N)
    r2, m2 = semquad_surface_velocity(sq2, qt2, ux2, uz2; wall=:top)
    @test r2 ≈ 1.0 / sqrt(5.0) atol=1e-13
    @test m2 ≈ 1.0 atol=1e-13

    # (4) a field localised at the bottom: ux = (1-y) x^2
    uxb = [(1.0 - y) * x^2 for (x, y) in sq.coords]
    rb, mb = semquad_surface_velocity(sq, qt, uxb, zeros(sq.N); wall=:bottom)
    @test rb ≈ 1.0 / sqrt(5.0) atol=1e-13
    @test mb ≈ 1.0 atol=1e-13
    rt, mt = semquad_surface_velocity(sq, qt, uxb, zeros(sq.N); wall=:top)
    @test rt ≈ 0.0 atol=1e-14
    @test mt ≈ 0.0 atol=1e-14

    # (5) guards
    @test_throws ArgumentError semquad_surface_velocity(sq, qt, ux, uz; wall=:left)

    # (6) driver: history carries the two columns and they match the diagnostic
    qt3 = Quadtree(5)
    for _ in 1:3, lf in collect(qt3.leaves); refine!(qt3, lf); end
    balance!(qt3)
    ss = make_semstokes(qt3, 2)
    theta0 = [0.05 * sinpi(x) * sinpi(z) for (x, z) in ss.su.coords]
    pr = Params(Ra=1.0e6, H=0.0, eta_law=:expTz,
                eta_b=log(16.0), eta_c=log(4.0), bc=:free)
    dt = 1.0e-5
    res = run_semquad_simulation(qt3, pr, theta0;
        p=2, tstop=3 * dt, dt0=dt, dtmax=dt, dtmin=1.0e-16,
        amr_every=0, ev=0.0, advection=:skew, linear=:direct,
        maxpic=1, pic_tol=1.0e-6, verbose=false)
    @test length(res.history[end]) == 18
    rs, rm = semquad_surface_velocity(res.ss.su, res.qt, res.ux, res.uz; wall=:top)
    @test res.history[end][14] == rs
    @test res.history[end][15] == rm
    # the surface maximum cannot exceed the global maximum
    @test rm <= maximum(hypot.(res.ux, res.uz)) + 1.0e-12
end
