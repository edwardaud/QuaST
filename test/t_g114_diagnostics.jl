using Test, QuaST

@testset "G114 SEM diagnostics: closure, averages, steady stop" begin
    function run_case(; nsteps::Int, H::Float64=0.0, theta_amp::Float64=0.0,
                      tavg_start::Float64=Inf, steady_tol::Float64=Inf,
                      steady_window::Int=3, tmin::Float64=0.0, maxsteps::Int=100)
        qt = Quadtree(5)
        for _ in 1:3, lf in collect(qt.leaves); refine!(qt, lf); end
        balance!(qt)
        ss = make_semstokes(qt, 2)
        theta0 = [theta_amp * sinpi(x) * sinpi(z) for (x, z) in ss.su.coords]
        pr = Params(Ra=1.0e6, H=H, eta_law=:expTz,
                    eta_b=log(16.0), eta_c=log(4.0), bc=:free)
        dt = 1.0e-5
        run_semquad_simulation(qt, pr, theta0;
            p=2, tstop=nsteps * dt, dt0=dt, dtmax=dt, dtmin=1.0e-16,
            amr_every=0, ev=0.0, advection=:skew, linear=:direct,
            maxpic=1, pic_tol=1.0e-6, verbose=false, maxsteps=maxsteps,
            tavg_start=tavg_start, steady_tol=steady_tol,
            steady_window=steady_window, tmin=tmin)
    end

    # (1) conductive state with H != 0: Nu_top = 1 + H/2, Nu_bot = 1 - H/2 and
    # the closure residual Nu_top - Nu_bot - (H - I) must vanish (u = 0 -> I = 0)
    rc = run_case(nsteps=1, H=2.0)
    @test rc.history[1][3] ≈ 2.0 atol=1.0e-12      # Nu_top
    @test rc.history[1][4] ≈ 0.0 atol=1.0e-12      # Nu_bot
    @test rc.history[1][13] ≈ 0.0 atol=1.0e-10     # closure

    # (2) history now carries the extra diagnostics
    r = run_case(nsteps=4, H=1.0, theta_amp=0.05)
    @test length(r.history[1]) == 18
    div_re = semquad_max_divergence(r.ss.su, r.qt, r.ux, r.uz)
    @test r.history[end][9] == div_re
    @test r.history[end][10] ≈ minimum(2.0^(-lf[1]) for lf in r.qt.leaves)
    umax = maximum(hypot.(r.ux, r.uz))
    @test r.history[end][12] ≈ umax * r.history[end][2] / r.history[end][10]

    # (3) time averages over t >= tavg_start
    ra = run_case(nsteps=4, H=1.0, theta_amp=0.05, tavg_start=0.0)
    @test ra.avg !== nothing
    @test ra.avg.n == 4
    @test ra.avg.Vrms ≈ sum(h[5] for h in ra.history) / 4
    @test ra.avg.Nu_top ≈ sum(h[3] for h in ra.history) / 4
    @test ra.avg.Nu_top_std >= 0.0
    # requesting averages from a time the run never reaches gives nothing
    @test run_case(nsteps=2, tavg_start=1.0).avg === nothing

    # (4) steady-state stop: a conductive initial state is stationary, so the
    # criterion must fire after steady_window consecutive good steps
    rs = run_case(nsteps=40, steady_tol=1.0e-10, steady_window=3, maxsteps=40)
    @test rs.steps == 4                     # step 1 has dNu = Inf, then 3 good
    @test rs.steps < 40
    # a genuinely evolving state must not be declared steady: with detection
    # enabled the run must take exactly as many steps as without it
    rn = run_case(nsteps=6, theta_amp=0.05, steady_tol=1.0e-12, steady_window=3)
    r_inf = run_case(nsteps=6, theta_amp=0.05, steady_tol=Inf)
    @test rn.steps == r_inf.steps
end
