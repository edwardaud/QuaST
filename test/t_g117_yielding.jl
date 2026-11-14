# G117: Drucker-Prager yielding — the plastic cut-off with the taumax cap and eta_min floor.
using Test, QuaST

@testset "G117 Drucker-Prager yielding" begin
    # eta = min(eta_visc, tau_y/(2 edot)), tau_y = min(c cos(phi)+p sin(phi), tmax)
    base = Params(Ra=1.0e4, H=0.0, eta_law=:arrhenius, eta_E=6.907755278982137,
                  eta_T0=0.5, eta_min=1.0e-3, yield_c=1.0e30)

    # (1) a huge cohesion is inert: identical to the pure Arrhenius value
    pr0 = Params(Ra=1.0e4, H=0.0, eta_law=:arrhenius, eta_E=6.907755278982137,
                 eta_T0=0.5, eta_min=1.0e-3)
    @test yield_active(pr0) == false
    @test yield_active(base) == true
    for (T, edot, p) in ((1.0, 1.0e-3, 0.0), (0.5, 1.0e-2, 0.1), (0.0, 1.0, -0.2))
        eta_v = eta_at(T, 0.5, pr0)[1]
        @test semquad_visco_plastic_eta(T, 0.5, edot, p, base) == eta_v
    end

    # (2) pressure-independent yield (phi = 0): eta_pl = c/(2 edot)
    c = 0.5
    pr1 = Params(Ra=1.0e4, H=0.0, eta_law=:arrhenius, eta_E=0.0, eta_T0=0.5,
                 eta_min=1.0e-3, yield_c=c)
    @test semquad_yield_stress(123.0, pr1) == c        # no pressure dependence
    # edot large enough that c/(2 edot) < eta_visc = 1
    @test semquad_visco_plastic_eta(1.0, 0.5, 1.0, 0.0, pr1) ≈ c / 2.0
    # edot small: the viscous value wins
    @test semquad_visco_plastic_eta(1.0, 0.5, 1.0e-6, 0.0, pr1) == 1.0

    # (3) monotonicity in c, edot and p
    cvals = [0.1, 0.5, 1.0, 5.0]
    etas = [semquad_visco_plastic_eta(1.0, 0.5, 1.0, 0.0,
               Params(Ra=1.0e4, H=0.0, eta_law=:arrhenius, eta_E=0.0,
                      eta_T0=0.5, eta_min=1.0e-6, yield_c=cc)) for cc in cvals]
    @test all(diff(etas) .> 0.0)                       # more cohesion -> stiffer
    edots = [1.0e-2, 1.0e-1, 1.0, 10.0]
    etas_e = [semquad_visco_plastic_eta(1.0, 0.5, e, 0.0, pr1) for e in edots]
    @test all(diff(etas_e) .<= 0.0)                    # faster -> never stiffer
    @test etas_e[end] ≈ c / (2 * edots[end])           # yield-dominated at high rate
    pr3 = Params(Ra=1.0e4, H=0.0, eta_law=:arrhenius, eta_E=0.0, eta_T0=0.5,
                 eta_min=1.0e-9, yield_c=0.0, yield_phi=deg2rad(30.0))
    etas_p = [semquad_visco_plastic_eta(1.0, 0.5, 1.0, pv, pr3)
              for pv in (0.0, 1.0, 2.0, 4.0)]
    @test all(diff(etas_p) .> 0.0)                     # more pressure -> stiffer

    # (4) taumax caps the yield stress, and the eta_min floor is respected
    pr4 = Params(Ra=1.0e4, H=0.0, eta_law=:arrhenius, eta_E=0.0, eta_T0=0.5,
                 eta_min=1.0e-2, yield_c=1.0e30, yield_taumax=0.2)
    @test semquad_yield_stress(1.0e6, pr4) == 0.2
    @test semquad_visco_plastic_eta(1.0, 0.5, 1.0e3, 1.0e6, pr4) >= 1.0e-2

    # (5) yielding requires a finite eta_min
    @test_throws ArgumentError Params(eta_law=:arrhenius, eta_E=1.0, eta_T0=0.5,
                                      yield_c=1.0)
    # and rejects a bad friction angle
    @test_throws ArgumentError Params(yield_phi=2.0, eta_min=1.0e-3)

    # (6) through the driver: an inert yield stress is bit-identical to the
    # Arrhenius-only run; an active one stays finite and lowers Vrms
    function run_yield(prr)
        qt = Quadtree(5)
        for _ in 1:3, lf in collect(qt.leaves); refine!(qt, lf); end
        balance!(qt)
        ss = make_semstokes(qt, 2)
        theta0 = [0.1 * sinpi(x) * sinpi(z) for (x, z) in ss.su.coords]
        dt = 1.0e-5
        run_semquad_simulation(qt, prr, theta0;
            p=2, tstop=4 * dt, dt0=dt, dtmax=dt, dtmin=1.0e-16,
            amr_every=0, ev=0.0, advection=:skew, linear=:direct,
            maxpic=1, pic_tol=1.0e-6, verbose=false)
    end
    pr_inert = Params(Ra=1.0e6, H=0.0, eta_law=:arrhenius, eta_E=3.0,
                      eta_T0=0.5, eta_min=1.0e-3, yield_c=1.0e30)
    pr_plain = Params(Ra=1.0e6, H=0.0, eta_law=:arrhenius, eta_E=3.0,
                      eta_T0=0.5, eta_min=1.0e-3)
    a = run_yield(pr_inert)
    b = run_yield(pr_plain)
    @test maximum(abs, a.theta - b.theta) == 0.0
    @test a.history == b.history

    pr_y = Params(Ra=1.0e6, H=0.0, eta_law=:arrhenius, eta_E=3.0,
                  eta_T0=0.5, eta_min=1.0e-2, yield_c=0.05, yield_phi=0.0)
    ry = run_yield(pr_y)
    @test all(isfinite, ry.theta)
    @test all(isfinite, vcat(ry.history...))
    @test ry.avg === nothing
end
