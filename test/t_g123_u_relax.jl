# G123: lagged viscosity and velocity under-relaxation in the Tosi Picard loop.
using Test, QuaST

@testset "G123 lagged viscosity and velocity under-relaxation" begin
    # The Tosi law is strain-rate weakening.  Iterating the Picard loop inside a
    # step grows u until the CFL bound (fixed at iteration 1) is violated; the
    # driver now refuses maxpic > 1 for :tosi instead of diverging silently.
    law = TosiLaw(etaT=1.0e5, etaZ=10.0, eta_star=1.0e-3, sigma_y=4.0,
                  eta_init=1.0e-1, eta0=1.0, eta_min=1.0e-6, eta_max=10.0)

    function run_case(; maxpic=1, pic_u_relax=1.0, nsteps=30)
        pr = Params(Ra=100.0, H=0.0, eta_law=:tosi, tosi=law, bc=:free)
        qt = Quadtree(7)
        for _ in 1:4, lf in collect(qt.leaves); refine!(qt, lf); end
        balance!(qt)
        ss = make_semstokes(qt, 2)
        theta0 = [0.01 * cospi(x) * sinpi(z) for (x, z) in ss.su.coords]
        run_semquad_simulation(qt, pr, theta0;
            p=2, tstop=10.0, dt0=1.0e-4, dtmax=1.0e-3, dtmin=1.0e-9,
            cfl=1.0, amr_every=0, ev=0.0, advection=:skew, linear=:direct,
            maxpic=maxpic, pic_tol=1.0e-8, pic_u_relax=pic_u_relax,
            verbose=false, maxsteps=nsteps)
    end

    # (1) the diverging configuration is rejected, not run
    @test_throws ArgumentError run_case(maxpic=5)

    # (2) argument guards
    @test_throws ArgumentError run_case(pic_u_relax=0.0, nsteps=1)
    @test_throws ArgumentError run_case(pic_u_relax=1.5, nsteps=1)

    # (3) pic_u_relax = 1 is bit-identical to leaving it at the default
    a = run_case(maxpic=1)
    b = run_case(maxpic=1, pic_u_relax=1.0)
    @test a.history == b.history
    @test maximum(abs, a.theta - b.theta) == 0.0

    # (4) with the lagged scheme, damping is stable and finite (it changes the
    # answer, which is why it is not the default)
    c = run_case(maxpic=1, pic_u_relax=0.5)
    @test all(isfinite, vcat(c.history...))
    @test maximum(h[5] for h in c.history) < 200.0
    @test maximum(abs, c.theta - a.theta) > 0.0
end
