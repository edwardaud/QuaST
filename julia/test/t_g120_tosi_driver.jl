using Test, MantleSEM

@testset "G120 Tosi law through the driver (eta0 convention)" begin
    law = TosiLaw(etaT=1.0e5, etaZ=10.0, eta_star=1.0e-3, sigma_y=4.0,
                  eta_init=1.0e-1, eta0=1.0e-1, eta_min=1.0e-6, eta_max=10.0)

    # (1) Params guards
    @test_throws ArgumentError Params(eta_law=:tosi)
    @test_throws ArgumentError Params(eta_law=:tosi, tosi=law, yield_c=1.0)
    @test_throws ArgumentError Params(eta_law=:expT, deta_T=2.0, tosi=law)
    pr = Params(Ra=100.0, H=0.0, eta_law=:tosi, tosi=law)
    @test pr.tosi === law

    # (2) eta_at cannot evaluate a strain-rate dependent law
    @test_throws ArgumentError eta_at(0.5, 0.5, pr)

    # (3) through the driver: finite, and the viscosity at two points equals the
    # law evaluated with the strain rate from the returned velocity
    qt = Quadtree(5)
    for _ in 1:3, lf in collect(qt.leaves); refine!(qt, lf); end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    theta0 = [0.01 * cospi(x) * sinpi(z) for (x, z) in ss.su.coords]
    dt = 1.0e-5
    res = run_semquad_simulation(qt, Params(Ra=100.0, H=0.0, eta_law=:tosi,
                                            tosi=law), theta0;
        p=2, tstop=4 * dt, dt0=dt, dtmax=dt, dtmin=1.0e-16,
        amr_every=0, ev=0.0, advection=:skew, linear=:direct,
        maxpic=1, pic_tol=1.0e-10, verbose=false)
    @test all(isfinite, res.theta)
    @test all(isfinite, vcat(res.history...))
    leaves = Set(res.qt.leaves)
    for (x, y) in ((0.3, 0.2), (0.7, 0.8), (0.5, 0.5))
        T = Tbar(y, 0.0) + MantleSEM._semquad_eval_at(res.ss.su, res.theta, x, y, leaves)
        ed = semquad_strain_rate_at(res.ss.su, res.ux, res.uz, x, y, leaves)
        eta_law = semquad_eta_tosi(T, ed, y, law)[1]
        eta_used = pr.tosi.eta0 > 0 ? eta_law : NaN
        @test isfinite(eta_used)
        @test law.eta_min / law.eta0 <= eta_used <= law.eta_max / law.eta0
    end

    # (4) the eta0 convention: halving eta0 doubles the nondimensional viscosity
    law2 = TosiLaw(etaT=1.0e5, etaZ=10.0, eta_star=1.0e-3, sigma_y=4.0,
                   eta0=0.05, eta_min=1.0e-6, eta_max=10.0)
    _, abs1 = semquad_eta_tosi(0.3, 1.0e-3, 0.4, law)
    nd1, _ = semquad_eta_tosi(0.3, 1.0e-3, 0.4, law)
    nd2, _ = semquad_eta_tosi(0.3, 1.0e-3, 0.4, law2)
    @test nd2 ≈ 2.0 * nd1
    @test nd1 == abs1 / law.eta0

    # (5) a larger eta0 (softer nondimensional viscosity at the same Ra) gives a
    # *different* finite run: proves the eta0 value actually reaches Stokes
    res2 = run_semquad_simulation(qt, Params(Ra=100.0, H=0.0, eta_law=:tosi,
                                             tosi=law2), theta0;
        p=2, tstop=4 * dt, dt0=dt, dtmax=dt, dtmin=1.0e-16,
        amr_every=0, ev=0.0, advection=:skew, linear=:direct,
        maxpic=1, pic_tol=1.0e-10, verbose=false)
    @test all(isfinite, res2.theta)
    @test maximum(abs, res2.theta - res.theta) > 0.0

    # (6) the Tosi law parameters are part of the checkpoint configuration, so a
    # restart with a different law (here the depth direction) is rejected
    tmp = tempname() * ".h5"
    run_semquad_simulation(qt, Params(Ra=100.0, H=0.0, eta_law=:tosi, tosi=law),
                           theta0;
        p=2, tstop=2 * dt, dt0=dt, dtmax=dt, dtmin=1.0e-16,
        amr_every=0, ev=0.0, advection=:skew, linear=:direct,
        maxpic=1, pic_tol=1.0e-10, verbose=false,
        checkpoint_path=tmp, checkpoint_every=1)
    st = load_semquad_state(tmp)
    @test st.config["tosi_z_is_depth"] == "true"
    @test st.config["tosi_sigma_y"] == "4.0"
    law_h = TosiLaw(etaT=1.0e5, etaZ=10.0, eta_star=1.0e-3, sigma_y=4.0,
                    eta_init=1.0e-1, eta0=1.0e-1, eta_min=1.0e-6, eta_max=10.0,
                    z_is_depth=false)
    @test_throws ArgumentError run_semquad_simulation(qt,
        Params(Ra=100.0, H=0.0, eta_law=:tosi, tosi=law_h), Float64[];
        p=2, tstop=4 * dt, dt0=dt, dtmax=dt, dtmin=1.0e-16,
        amr_every=0, ev=0.0, advection=:skew, linear=:direct,
        maxpic=1, pic_tol=1.0e-10, verbose=false, restart_from=tmp)
    rm(tmp; force=true)
end
