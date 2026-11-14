# G115: the Arrhenius viscosity law — closed form, T=1 normalisation and the clamps.
using Test, QuaST

@testset "G115 Arrhenius viscosity law" begin
    # eta = exp( E/(T + T0) - E/(1 + T0) ), normalised so eta(T=1) = 1.
    # T is the absolute temperature in [0,1] (T=1 bottom, T=0 top), so colder
    # material is stiffer: eta(0) > eta(1).
    E = 6.907755278982137          # ln(1000) scale of the exponent
    T0 = 0.5
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:arrhenius, eta_E=E, eta_T0=T0)

    # (1) normalisation and closed form
    @test eta_at(1.0, 0.0, pr)[1] == 1.0
    @test eta_at(0.0, 1.0, pr)[1] ≈ exp(E * (1.0 / T0 - 1.0 / (1.0 + T0)))
    for T in (0.0, 0.25, 0.5, 0.75, 1.0)
        @test eta_at(T, 0.5, pr)[1] ≈ exp(E * (1.0 / (T + T0) - 1.0 / (1.0 + T0)))
    end

    # (2) strictly increasing as T decreases (stiffer when cold)
    vals = [eta_at(T, 0.5, pr)[1] for T in 0.0:0.05:1.0]
    @test all(diff(vals) .< 0.0)

    # (3) E = 0 is exactly the constant limit, and the clip is applied
    pr0 = Params(Ra=1.0e4, H=0.0, eta_law=:arrhenius, eta_E=0.0, eta_T0=T0)
    @test all(eta_at(T, 0.5, pr0)[1] == 1.0 for T in 0.0:0.1:1.0)
    # eta(T=1) = 1 < eta_min = 10 -> clipped up
    prmin = Params(Ra=1.0e4, H=0.0, eta_law=:arrhenius, eta_E=E, eta_T0=T0,
                   eta_min=10.0)
    @test eta_at(1.0, 0.0, prmin)[1] == 10.0
    @test eta_at(1.0, 0.0, prmin)[2] == true
    # eta(T=0) ~ 1e4 > eta_max = 1e3 -> clipped down
    prmax = Params(Ra=1.0e4, H=0.0, eta_law=:arrhenius, eta_E=E, eta_T0=T0,
                   eta_max=1.0e3)
    @test eta_at(0.0, 1.0, prmax)[1] == 1.0e3
    @test eta_at(0.0, 1.0, prmax)[2] == true

    # (4) parameter guards
    @test_throws ArgumentError Params(eta_law=:arrhenius, eta_E=E, eta_T0=0.0)
    @test_throws ArgumentError Params(eta_law=:arrhenius, eta_E=-1.0, eta_T0=T0)

    # (5) through the driver: eta_E = 0 must reproduce the constant-viscosity
    # run bit for bit (both laws then give eta == 1 exactly)
    function run_law(law::Symbol)
        qt = Quadtree(5)
        for _ in 1:3, lf in collect(qt.leaves); refine!(qt, lf); end
        balance!(qt)
        ss = make_semstokes(qt, 2)
        theta0 = [0.05 * sinpi(x) * sinpi(z) for (x, z) in ss.su.coords]
        p = law === :arrhenius ?
            Params(Ra=1.0e6, H=0.0, eta_law=:arrhenius, eta_E=0.0, eta_T0=0.5) :
            Params(Ra=1.0e6, H=0.0, eta_law=:expT, deta_T=1.0)
        dt = 1.0e-5
        run_semquad_simulation(qt, p, theta0;
            p=2, tstop=4 * dt, dt0=dt, dtmax=dt, dtmin=1.0e-16,
            amr_every=0, ev=0.0, advection=:skew, linear=:direct,
            maxpic=1, pic_tol=1.0e-6, verbose=false)
    end
    ra = run_law(:arrhenius)
    rz = run_law(:expT)
    @test maximum(abs, ra.theta - rz.theta) == 0.0
    @test ra.history == rz.history

    # (6) a real Arrhenius run stays finite and produces a viscosity contrast
    qt = Quadtree(5)
    for _ in 1:3, lf in collect(qt.leaves); refine!(qt, lf); end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    theta0 = [0.1 * sinpi(x) * sinpi(z) for (x, z) in ss.su.coords]
    prr = Params(Ra=1.0e6, H=0.0, eta_law=:arrhenius, eta_E=E, eta_T0=T0)
    dt = 1.0e-5
    rr = run_semquad_simulation(qt, prr, theta0;
        p=2, tstop=3 * dt, dt0=dt, dtmax=dt, dtmin=1.0e-16,
        amr_every=0, ev=0.0, advection=:skew, linear=:direct,
        maxpic=1, pic_tol=1.0e-6, verbose=false)
    @test all(isfinite, rr.theta)
    @test all(isfinite, vcat(rr.history...))
    # viscosity really varies across the box
    leaves = Set(rr.qt.leaves)
    e_top = semquad_eta_at(rr.ss.su, rr.theta, 0.5, 0.999, prr, leaves)
    e_bot = semquad_eta_at(rr.ss.su, rr.theta, 0.5, 0.001, prr, leaves)
    @test e_top / e_bot > 10.0
end
