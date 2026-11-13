using Test, QuaST
using SparseArrays

@testset "G131 rho*cp weighting of the energy equation" begin
    qt = Quadtree(4)
    for _ in 1:2, lf in collect(qt.leaves); refine!(qt, lf); end
    balance!(qt)
    sq = make_semquad(qt, 3)
    n = sq.N
    ux = [sinpi(x) * sinpi(y) for (x, y) in sq.coords]
    uz = [cospi(x) * sinpi(y) for (x, y) in sq.coords]
    th = [0.1 * sinpi(x) * sinpi(y) for (x, y) in sq.coords]

    pr0 = Params(Ra=1.0e4, H=0.0)
    M0, K0, C0, F0 = assemble_semquad_energy(sq, qt, pr0, ux, uz; theta=th)

    # (1) the switch is off by default -> bit-identical
    rs = SemRefState(Tsurf=0.091, alpha=1.0, cp=1.0, g=1.0, rho0=1.0,
                     rho_fun=y -> exp(1.0 * (1.0 - y)))
    # With a reference state the *base state* already changes F (the adiabat
    # replaces the conductive profile), so the switch only has to leave the
    # matrices alone: an explicit rho = 1 must equal the default (off) path.
    pr_on = Params(Ra=1.0e4, H=0.0, refstate=rs)              # weight off
    M1, K1, C1, F1 = assemble_semquad_energy(sq, qt, pr_on, ux, uz; theta=th)
    M1b, K1b, C1b, F1b = assemble_semquad_energy(sq, qt, pr_on, ux, uz; theta=th,
                                                 rho_ref=y -> 1.0)
    @test M1b == M1 && K1b == K1 && C1b == C1 && F1b == F1

    # (2) with the weighting on, M and C are scaled pointwise by rho(y) cp
    rw = y -> exp(1.0 * (1.0 - y))
    M2, K2, C2, F2 = assemble_semquad_energy(sq, qt, pr_on, ux, uz; theta=th,
                                             rho_ref=rw)
    @test M2 != M0
    @test K2 == K0                              # diffusion is not weighted
    @test C2 != C0
    # the mass matrix with rho=1 must agree with the unweighted one
    M3, _, _, _ = assemble_semquad_energy(sq, qt, pr_on, ux, uz; theta=th,
                                          rho_ref=y -> 1.0)
    @test maximum(abs, M3 - M0) < 1.0e-14

    # (3) through the driver: the switch is off by default, and turning it on
    # changes the solution while staying finite
    function run_case(prr; nsteps=3)
        q2 = Quadtree(5)
        for _ in 1:3, lf in collect(q2.leaves); refine!(q2, lf); end
        balance!(q2)
        ss = make_semstokes(q2, 2)
        theta0 = [0.05 * sinpi(x) * sinpi(z) for (x, z) in ss.su.coords]
        run_semquad_simulation(q2, prr, theta0;
            p=2, tstop=nsteps * 1.0e-5, dt0=1.0e-5, dtmax=1.0e-5, dtmin=1.0e-16,
            amr_every=0, ev=0.0, advection=:skew, linear=:direct,
            maxpic=1, pic_tol=1.0e-6, verbose=false)
    end
    pa = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=2.0, refstate=rs)
    pb = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=2.0, refstate=rs,
                rho_ref_weight=true)
    ra = run_case(pa)
    rb = run_case(pb)
    @test all(isfinite, vcat(rb.history...))
    @test maximum(abs, ra.theta - rb.theta) > 0.0
end
