# G130: the TALA/ALA mass constraint with a stratified density and the drho/dy term.
using Test, QuaST
using SparseArrays

@testset "G130 TALA/ALA mass constraint with a stratified density" begin
    # (1) constant density: the switch stays off, so the reference-element path is
    # used unchanged (bit-identical by construction).
    rs0 = SemRefState(Tsurf=0.0, alpha=1.0, cp=1.0, g=1.0, rho0=1.0)
    pr0 = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=2.0, refstate=rs0)
    @test mass_rho_fun(pr0) === nothing
    @test mass_rho_fun(Params(Ra=1.0e4, H=0.0)) === nothing

    # (2) a stratified density gives a density function
    rho = y -> exp(0.7 * (1.0 - y))
    drho = y -> -0.7 * exp(0.7 * (1.0 - y))
    rs = SemRefState(Tsurf=0.0, alpha=1.0, cp=1.0, g=1.0, rho0=1.0,
                     rho_fun=rho)
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=2.0, refstate=rs)
    rf = mass_rho_fun(pr)
    @test rf !== nothing
    @test rf(1.0) ≈ 1.0 rtol=1e-14
    @test rf(0.0) ≈ exp(0.7) rtol=1e-14

    qt = Quadtree(5)
    for _ in 1:3, lf in collect(qt.leaves); refine!(qt, lf); end
    balance!(qt)
    ss = make_semstokes(qt, 2)

    # (3) the drho term: constant density + zero derivative reproduces the
    # reference matrix, and the anelastic matrix differs from the rho-weighted
    # divergence by exactly int q phi drho/dy (checked column by column).
    B0 = semquad_stokes_B(ss, qt)
    @test maximum(abs, semquad_stokes_B(ss, qt; rho_fun=y -> 1.0,
                                        drho_fun=y -> 0.0) - B0) < 1e-12
    Brho = semquad_stokes_B(ss, qt; rho_fun=rho)
    Ban = semquad_stokes_B(ss, qt; rho_fun=rho, drho_fun=drho)
    @test maximum(abs, Ban - Brho) > 1e-3 * maximum(abs, Brho)

    # (4) default run (anelastic): finite, and the returned velocity satisfies
    # the *anelastic* discrete constraint, not just the rho-weighted one.
    theta0 = [0.05 * sinpi(x) * sinpi(z) for (x, z) in ss.su.coords]
    dt = 1.0e-5
    res = run_semquad_simulation(qt, pr, theta0;
        p=2, tstop=3 * dt, dt0=dt, dtmax=dt, dtmin=1.0e-16,
        amr_every=0, ev=0.0, advection=:skew, linear=:direct,
        maxpic=1, pic_tol=1.0e-6, verbose=false)
    @test all(isfinite, res.theta)
    u = vcat(res.ux, res.uz)
    @test maximum(abs, Ban * u) < 1.0e-5
    @test maximum(abs, Brho * u) > 1.0e-4    # the two constraints differ

    # (5) legacy :rho_div keeps the historical single rho-weighted matrix
    res2 = run_semquad_simulation(qt, pr, theta0;
        p=2, tstop=3 * dt, dt0=dt, dtmax=dt, dtmin=1.0e-16,
        amr_every=0, ev=0.0, advection=:skew, linear=:direct,
        maxpic=1, pic_tol=1.0e-6, verbose=false, mass_form=:rho_div)
    u2 = vcat(res2.ux, res2.uz)
    @test maximum(abs, Brho * u2) < 1.0e-5
    @test maximum(abs, Ban * u2) > 1.0e-4

    # (6) with rho = 1 the stratified route reduces to the constant one
    B1 = semquad_stokes_B(res.ss, res.qt; rho_fun=y -> 1.0)
    Bn = semquad_stokes_B(res.ss, res.qt)
    @test maximum(abs, B1 - Bn) < 1.0e-14

    # (7) a run without a reference state still uses div(u) = 0 and is finite
    res3 = run_semquad_simulation(qt, Params(Ra=1.0e4, H=0.0, eta_law=:expT,
                                             deta_T=2.0), theta0;
        p=2, tstop=3 * dt, dt0=dt, dtmax=dt, dtmin=1.0e-16,
        amr_every=0, ev=0.0, advection=:skew, linear=:direct,
        maxpic=1, pic_tol=1.0e-6, verbose=false)
    @test all(isfinite, res3.theta)
    @test maximum(abs, Bn * vcat(res3.ux, res3.uz)) <
          1.0e-8 * max(1.0, maximum(hypot.(res3.ux, res3.uz)))
end
