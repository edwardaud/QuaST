# G127: TALA buoyancy and wall temperatures on the adiabat — zero anomaly force, physical wall values.
using Test, QuaST

@testset "G127 TALA buoyancy and wall temperatures" begin
    rs = SemRefState(Tsurf=0.2, alpha=0.5, cp=1.0, g=1.0, rho0=1.0)
    Tad_top = semquad_Tad(rs, 1.0)
    Tad_bot = semquad_Tad(rs, 0.0)

    function run_case(pr; theta_amp=0.0, nsteps=3)
        qt = Quadtree(5)
        for _ in 1:3, lf in collect(qt.leaves); refine!(qt, lf); end
        balance!(qt)
        ss = make_semstokes(qt, 2)
        theta0 = theta_amp == 0.0 ? zeros(ss.su.N) :
                 [theta_amp * sinpi(x) * sinpi(z) for (x, z) in ss.su.coords]
        dt = 1.0e-5
        run_semquad_simulation(qt, pr, theta0;
            p=2, tstop=nsteps * dt, dt0=dt, dtmax=dt, dtmin=1.0e-16,
            amr_every=0, ev=0.0, advection=:skew, linear=:direct,
            maxpic=1, pic_tol=1.0e-6, verbose=false)
    end

    # (1) walls on the adiabat and theta = 0: the TALA buoyancy
    # Ra rho(y) alpha(y) theta is exactly zero, so the flow stays at the level of
    # the adiabat-curvature response only.  That response is physical: the
    # exponential adiabat has lap(Tad) = (alpha g/cp)^2 Tad != 0 and the theta
    # equation carries it as a source, so a static adiabat is not a conduction
    # solution and theta = 0 is not an exact fixed point of the coupled system.
    pr1 = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=2.0,
                 refstate=rs, Twall_top=Tad_top, Twall_bot=Tad_bot)
    r = run_case(pr1; theta_amp=0.0)
    @test maximum(abs, r.ux) < 1.0e-5
    @test maximum(abs, r.uz) < 1.0e-5
    @test maximum(abs, r.theta) < 1.0e-4

    # (2) physical walls (0 and 1) with a reference state: theta_wall = Twall-Tad
    r2 = run_case(Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=2.0,
                         refstate=rs); theta_amp=0.0)
    wall_lo = Inf
    wall_hi = -Inf
    for (g, (x, y)) in enumerate(r2.ss.su.coords)
        if y <= 1.0e-14
            wall_lo = r2.theta[g]
        elseif y >= 1.0 - 1.0e-14
            wall_hi = r2.theta[g]
        end
    end
    @test wall_lo ≈ 1.0 - Tad_bot rtol=1e-10
    @test wall_hi ≈ 0.0 - Tad_top rtol=1e-10
    @test maximum(hypot.(r2.ux, r2.uz)) > 0.0

    # (3) without a reference state the wall values are zero (Tad = Tbar), so a
    # theta = 0 state is still a fixed point: the old path is unchanged
    r3 = run_case(Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=2.0); theta_amp=0.0)
    @test maximum(abs, r3.ux) < 1.0e-8
    @test maximum(abs, r3.theta) < 1.0e-10
end
