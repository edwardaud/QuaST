using Test, QuaST

@testset "G134 adiabatic heating carries rho_ad(y)" begin
    qt = Quadtree(4)
    for _ in 1:2, lf in collect(qt.leaves); refine!(qt, lf); end
    balance!(qt)
    sq = make_semquad(qt, 3)
    n = sq.N
    b = 2.0
    uz = [b * y for (x, y) in sq.coords]
    z0 = zeros(n)

    # (1) no reference state: the density factor is 1, so the closed form of
    # G125 must still hold exactly (-Di b/6 for T = 1 - y)
    pr0 = Params(Ra=100.0, H=0.0, Di=2.5, heating_adiabatic=true)
    _, _, _, F1 = assemble_semquad_energy(sq, qt, pr0, z0, uz; theta=z0,
                                          heating_adiabatic=true)
    _, _, _, F0 = assemble_semquad_energy(sq, qt, pr0, z0, uz; theta=z0)
    @test sum(F1) - sum(F0) ≈ -pr0.Di * b / 6.0 rtol=1e-12

    # (2) constant-density reference state: same value
    rs0 = SemRefState(Tsurf=0.5, alpha=1.0, cp=1.0, g=1.0, rho0=1.0)
    pr1 = Params(Ra=100.0, H=0.0, Di=2.5, heating_adiabatic=true, refstate=rs0)
    _, _, _, Fa = assemble_semquad_energy(sq, qt, pr1, z0, uz; theta=z0,
                                          heating_adiabatic=true)
    _, _, _, Fb = assemble_semquad_energy(sq, qt, pr1, z0, uz; theta=z0)
    # T = Tad(y) = 0.5 exp(1-y) with theta = 0, integrated by quadrature
    nq1 = 20000
    ref0 = 0.0
    for k in 1:nq1
        y = (k - 0.5) / nq1
        ref0 += (1.0 / nq1) * (-pr1.Di * 1.0 * semquad_Tad(rs0, y) * b * y)
    end
    @test sum(Fa) - sum(Fb) ≈ ref0 rtol=1e-6

    # (3) stratified rho: the source picks up rho_ad(y) = exp(1 - y)
    rs = SemRefState(Tsurf=0.5, alpha=1.0, cp=1.0, g=1.0, rho0=1.0,
                     rho_fun=y -> exp(1.0 * (1.0 - y)))
    pr2 = Params(Ra=100.0, H=0.0, Di=2.5, heating_adiabatic=true, refstate=rs)
    _, _, _, Fc = assemble_semquad_energy(sq, qt, pr2, z0, uz; theta=z0,
                                          heating_adiabatic=true)
    _, _, _, Fd = assemble_semquad_energy(sq, qt, pr2, z0, uz; theta=z0)
    # int_0^1 -Di * exp(1-y) * (1-y) * b y dy, integrated independently
    nq = 20000
    ref = 0.0
    for k in 1:nq
        y = (k - 0.5) / nq
        ref += (1.0 / nq) * (-pr2.Di * exp(1.0 - y) * semquad_Tad(rs, y) * b * y)
    end
    @test sum(Fc) - sum(Fd) ≈ ref rtol=1e-6
    # and it differs from the constant-density value by the expected factor
    @test sum(Fc) - sum(Fd) < 0.0
end
