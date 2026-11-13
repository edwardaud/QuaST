using Test, QuaST

@testset "G125 heating sources: dissipation and adiabatic" begin
    # Sum of the assembled source vector equals the volume integral of the
    # source, because sum_i phi_i = 1 at every point.  Two analytic states give
    # closed-form integrals, so this is an independent check of the new terms.
    qt = Quadtree(4)
    for _ in 1:2, lf in collect(qt.leaves); refine!(qt, lf); end
    balance!(qt)
    sq = make_semquad(qt, 3)
    n = sq.N
    zero_f = (x, y) -> 0.0
    eta1 = (x, y) -> 1.0

    # (1) Di = 0 (or heating off) must not change the source at all
    pr = Params(Ra=100.0, H=0.0)
    ux = [sinpi(x) * sinpi(y) for (x, y) in sq.coords]
    uz = [cospi(x) * sinpi(y) for (x, y) in sq.coords]
    th = [0.2 * sinpi(x) * sinpi(y) for (x, y) in sq.coords]
    _, _, _, F0 = assemble_semquad_energy(sq, qt, pr, ux, uz; theta=th)
    _, _, _, Fdi0 = assemble_semquad_energy(sq, qt, pr, ux, uz; theta=th,
                                            heating_dissipation=true,
                                            heating_adiabatic=true)
    @test Fdi0 == F0

    # (2) dissipation: u = (a y, 0) gives edot:edot = a^2/2, so with eta = 1
    #     int 2 eta edot:edot = a^2 and the source integral is (Di/Ra) a^2
    a = 0.7
    ux2 = [a * y for (x, y) in sq.coords]
    pr1 = Params(Ra=100.0, H=0.0, Di=2.5)
    _, _, _, F1 = assemble_semquad_energy(sq, qt, pr1, ux2, zeros(n);
                                          theta=zeros(n),
                                          heating_dissipation=true,
                                          eta_fun=eta1)
    _, _, _, F0b = assemble_semquad_energy(sq, qt, pr1, ux2, zeros(n);
                                           theta=zeros(n))
    @test sum(F1) - sum(F0b) ≈ (pr1.Di / pr1.Ra) * a^2 rtol=1e-12

    # (3) adiabatic: u = (0, b y), T = 1 - y (theta = 0) gives
    #     -Di int (1-y) b y dy = -Di * b/6
    b = 2.0
    uz2 = [b * y for (x, y) in sq.coords]
    _, _, _, F2 = assemble_semquad_energy(sq, qt, pr1, zeros(n), uz2;
                                          theta=zeros(n),
                                          heating_adiabatic=true)
    # baseline with the same velocity: the base advection -u_z dTbar must cancel
    _, _, _, F2b = assemble_semquad_energy(sq, qt, pr1, zeros(n), uz2;
                                           theta=zeros(n))
    @test sum(F2) - sum(F2b) ≈ -pr1.Di * b / 6.0 rtol=1e-12

    # (4) with theta != 0 the adiabatic integrand uses the full T = Tbar + theta
    th2 = [0.1 * x for (x, y) in sq.coords]      # linear in x: int over x = 0.05
    _, _, _, F3 = assemble_semquad_energy(sq, qt, pr1, zeros(n), [b * y for (x, y) in sq.coords];
                                          theta=th2, heating_adiabatic=true)
    # int_T (1-y+0.1x) b y = b/6 + b*0.1*0.5*int y dy = b/6 + b*0.1*0.25
    _, _, _, F3b = assemble_semquad_energy(sq, qt, pr1, zeros(n), uz2;
                                           theta=th2)
    @test sum(F3) - sum(F3b) ≈ -pr1.Di * (b / 6.0 + b * 0.1 * 0.25) rtol=1e-10

    # (5) through the driver: Di > 0 stays finite and changes the solution
    function run_di(prr)
        qt2 = Quadtree(5)
        for _ in 1:2, lf in collect(qt2.leaves); refine!(qt2, lf); end
        balance!(qt2)
        ss = make_semstokes(qt2, 2)
        theta0 = [0.05 * sinpi(x) * sinpi(z) for (x, z) in ss.su.coords]
        run_semquad_simulation(qt2, prr, theta0;
            p=2, tstop=3 * 1.0e-5, dt0=1.0e-5, dtmax=1.0e-5, dtmin=1.0e-16,
            amr_every=0, ev=0.0, advection=:skew, linear=:direct,
            maxpic=1, pic_tol=1.0e-6, verbose=false)
    end
    base = Params(Ra=1.0e6, H=0.0, eta_law=:expTz, eta_b=log(16.0), eta_c=log(4.0))
    hot = Params(Ra=1.0e6, H=0.0, eta_law=:expTz, eta_b=log(16.0), eta_c=log(4.0),
                 Di=1.0, heating_dissipation=true, heating_adiabatic=true)
    r0 = run_di(base)
    r1 = run_di(hot)
    @test all(isfinite, r1.theta)
    @test maximum(abs, r1.theta - r0.theta) > 0.0
end
