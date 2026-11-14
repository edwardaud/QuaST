# G122: work and viscous dissipation — closed-form integrals and Ra*W = U.
using Test, QuaST

@testset "G122 work and viscous dissipation" begin
    qt = Quadtree(4)
    for _ in 1:3, lf in collect(qt.leaves); refine!(qt, lf); end
    balance!(qt)
    sq = make_semquad(qt, 4)
    pr = Params(Ra=100.0, H=0.0, eta_law=:expTz, eta_b=1.0e-12, eta_c=0.0)
    eta1 = (x, y) -> 1.0
    n = sq.N
    leaves = Set(qt.leaves)

    # (1) conductive state (theta = 0, u = 0): both integrals vanish
    W, U = semquad_work_dissipation(sq, qt, pr, zeros(n), zeros(n), zeros(n), eta1)
    @test W == 0.0
    @test U == 0.0

    # (2) Couette-like shear u = (a y, 0): edot:edot = a^2/2, so with eta = 1
    #     U = int 2*1*(a^2/2) = a^2 and W = 0 (u_z = 0) exactly.
    a = 0.7
    ux = [a * y for (x, y) in sq.coords]
    uz = zeros(n)
    W, U = semquad_work_dissipation(sq, qt, pr, zeros(n), ux, uz, eta1)
    @test U ≈ a^2 atol=1e-12
    @test abs(W) < 1e-14

    # (3) vertical flow u = (0, b y) with theta = 0 (T = Tbar = 1 - y):
    #     W = int_0^1 b y (1-y) dy = b/6 exactly
    b = 2.5
    uz2 = [b * y for (x, y) in sq.coords]
    W2, U2 = semquad_work_dissipation(sq, qt, pr, zeros(n), zeros(n), uz2, eta1)
    @test W2 ≈ b / 6.0 atol=1e-12
    @test U2 ≈ 2.0 * b^2 atol=1e-12      # edot:edot = b^2 -> U = 2 b^2

    # (4) eta enters linearly: doubling eta doubles U and leaves W unchanged
    W3, U3 = semquad_work_dissipation(sq, qt, pr, zeros(n), ux, uz,
                                      (x, y) -> 2.0)
    @test U3 ≈ 2.0 * a^2 atol=1e-12
    @test W3 == W

    # (5) the physical balance Ra*W = U holds for a driven Stokes solution
    qt5 = Quadtree(5)
    for _ in 1:3, lf in collect(qt5.leaves); refine!(qt5, lf); end
    balance!(qt5)
    ss = make_semstokes(qt5, 2)
    theta0 = [0.05 * sinpi(x) * sinpi(z) for (x, z) in ss.su.coords]
    pr5 = Params(Ra=1.0e6, H=0.0, eta_law=:expTz,
                 eta_b=log(16.0), eta_c=log(4.0), bc=:free)
    dt = 1.0e-5
    res = run_semquad_simulation(qt5, pr5, theta0;
        p=2, tstop=2 * dt, dt0=dt, dtmax=dt, dtmin=1.0e-16,
        amr_every=0, ev=0.0, advection=:skew, linear=:direct,
        maxpic=1, pic_tol=1.0e-6, verbose=false)
    @test length(res.history[end]) == 18
    Wr = res.history[end][16]
    Ur = res.history[end][17]
    @test Wr > 0.0
    @test Ur > 0.0
    # discrete balance with a different quadrature rule: within 2%
    @test abs(res.history[end][18]) < 2.0e-2
    # the stored pair is internally consistent: recomputing with the same
    # velocity and the temperature that solve used would require that state, so
    # instead require the recorded balance to be the one reported
    @test res.history[end][18] == (pr5.Ra * Wr - Ur) / max(abs(pr5.Ra * Wr), abs(Ur), 1e-300)

    # (6) deviatoric projection (ASPECT's ALA): pure compression u=(a x, a y)
    # has eps:eps = 2a^2 and div u = 2a, so eps_d:eps_d = 2a^2 - (4/3)a^2 =
    # (2/3)a^2 and U = int 2 eta eps_d:eps_d = (4/3) a^2, versus 4 a^2 for the
    # full stress.
    prd = Params(Ra=100.0, H=0.0, eta_law=:expTz, eta_b=1.0e-12, eta_c=0.0,
                 deviatoric=true)
    ac = 0.3
    uxc = [ac * x for (x, y) in sq.coords]
    uzc = [ac * y for (x, y) in sq.coords]
    _, Uc = semquad_work_dissipation(sq, qt, prd, zeros(n), uxc, uzc, eta1)
    @test Uc ≈ (4.0 / 3.0) * ac^2 atol=1e-12
    _, Uf = semquad_work_dissipation(sq, qt, pr, zeros(n), uxc, uzc, eta1)
    @test Uf ≈ 4.0 * ac^2 atol=1e-12
end
