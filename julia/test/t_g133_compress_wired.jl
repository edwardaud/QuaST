using Test, MantleSEM

@testset "G133 compressibility term wired in" begin
    rs = SemRefState(Tsurf=0.091, alpha=1.0, cp=1.0, g=1.0, rho0=1.0,
                     rho_fun=y -> exp(1.0 * (1.0 - y)))

    function run_case(pr; nsteps=4)
        qt = Quadtree(5)
        for _ in 1:3, lf in collect(qt.leaves); refine!(qt, lf); end
        balance!(qt)
        ss = make_semstokes(qt, 2)
        theta0 = [0.05 * sinpi(x) * sinpi(z) for (x, z) in ss.su.coords]
        run_semquad_simulation(qt, pr, theta0;
            p=2, tstop=nsteps * 1.0e-5, dt0=1.0e-5, dtmax=1.0e-5, dtmin=1.0e-16,
            amr_every=0, ev=0.0, advection=:skew, linear=:direct,
            maxpic=1, pic_tol=1.0e-6, verbose=false)
    end

    # (1) default: the switch is off, so compress_block_fun returns nothing and
    # the saddle system is the old one -- bit-identical to before.
    off = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=2.0, refstate=rs,
                 Di=1.0, heating_adiabatic=true)
    qt = Quadtree(5)
    for _ in 1:3, lf in collect(qt.leaves); refine!(qt, lf); end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    @test compress_block_fun(ss, qt, off) === nothing

    # (2) with the switch on the block is assembled and has the expected shape
    on = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=2.0, refstate=rs,
                Di=1.0, heating_adiabatic=true, compress_p=true,
                compress_gamma=1.0)
    Pc = compress_block_fun(ss, qt, on)
    @test Pc !== nothing
    @test size(Pc) == (2 * ss.su.N, ss.sp.N)
    @test maximum(abs, Pc) > 0.0

    # (3) Di = 0 makes the term vanish -> the solution must equal the switch-off
    # run (continuous degeneration)
    zero_di = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=2.0, refstate=rs,
                     Di=0.0, heating_adiabatic=true, compress_p=true)
    @test maximum(abs, compress_block_fun(ss, qt, zero_di)) == 0.0

    # (4) through the driver: on vs off differ, and both stay finite
    r_off = run_case(off)
    r_on = run_case(on)
    @test all(isfinite, vcat(r_on.history...))
    @test maximum(abs, r_on.theta - r_off.theta) > 0.0
    @test maximum(abs, r_on.theta - r_off.theta) < 1.0   # no runaway
end
