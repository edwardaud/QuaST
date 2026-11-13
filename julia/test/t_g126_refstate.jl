using Test, MantleSEM

@testset "G126 reference state (P3.2 foundation)" begin
    # (1) the analytic adiabat satisfies dTad/dz = -alpha g Tad / cp
    rs = SemRefState(Tsurf=0.2, alpha=0.5, cp=1.0, g=1.0, rho0=1.0)
    for z in (0.0, 0.25, 0.5, 0.75, 1.0)
        @test semquad_dTad_dz(rs, z) ≈
              -rs.alpha * rs.g / rs.cp * semquad_Tad(rs, z) rtol=1e-14
    end
    @test semquad_Tad(rs, 1.0) ≈ 0.2 rtol=1e-14          # surface value
    @test semquad_Tad(rs, 0.0) > semquad_Tad(rs, 1.0)    # hotter at the bottom

    # (2) guards
    @test_throws ArgumentError SemRefState(cp=0.0)
    @test_throws ArgumentError SemRefState(alpha=-1.0)

    # (3) with no reference state the base gradient is exactly the old one
    pr0 = Params(Ra=100.0, H=2.0)
    @test semquad_base_gradient(pr0, 0.3) == -1.0 + 0.5 * 2.0 * (1.0 - 0.6)
    @test semquad_base_T(pr0, 0.3) == Tbar(0.3, 2.0)

    # (4) with a reference state it is the adiabat
    pr1 = Params(Ra=100.0, H=0.0, refstate=rs)
    @test semquad_base_gradient(pr1, 0.3) ≈ semquad_dTad_dz(rs, 0.3) rtol=1e-15
    @test semquad_base_T(pr1, 0.3) ≈ semquad_Tad(rs, 0.3) rtol=1e-15

    # (5) the energy assembly uses the adiabat: for a linear theta the base
    # source changes from int(-u_z dTbar) to int(-u_z dTad), and the difference
    # is the closed-form integral of the two gradients
    qt = Quadtree(4)
    for _ in 1:2, lf in collect(qt.leaves); refine!(qt, lf); end
    balance!(qt)
    sq = make_semquad(qt, 3)
    n = sq.N
    b = 2.0
    pr0h = Params(Ra=100.0, H=0.0)      # H = 0 so dTbar = -1 exactly
    uz = [b * y for (x, y) in sq.coords]
    _, _, _, Fa = assemble_semquad_energy(sq, qt, pr0h, zeros(n), uz; theta=zeros(n))
    _, _, _, Fb = assemble_semquad_energy(sq, qt, pr1, zeros(n), uz; theta=zeros(n))
    # int_0^1 -b y dTad/dz dz = b alpha g/cp * int y Tad dz, done analytically below
    rs2 = rs
    integ = 0.0
    nq = 4000
    for k in 1:nq
        z = (k - 0.5) / nq
        integ += (1.0 / nq) * (-b * z) * semquad_dTad_dz(rs2, z)
    end
    # Fb carries both the base advection and lap(Tad); check both pieces.
    lapint = (rs.alpha * rs.g / rs.cp) *
             (semquad_Tad(rs, 0.0) - semquad_Tad(rs, 1.0))
    @test sum(Fb) ≈ integ + lapint rtol=1e-3  # quadrature vs the assembly's rule
    @test sum(Fa) ≈ b / 2.0 rtol=1e-12        # int -b y dTbar = b/2

    # (5b) the base Laplacian: zero for the linear Boussinesq base, and
    # (alpha g/cp)^2 Tad for the adiabat.  The theta equation needs it because
    # lap(T) = lap(theta) + lap(Tad).
    @test semquad_base_laplacian(pr0h, 0.37) == 0.0
    zz = 0.37
    hh = 1.0e-3
    fd2 = (semquad_Tad(rs, zz + hh) - 2 * semquad_Tad(rs, zz) +
           semquad_Tad(rs, zz - hh)) / hh^2
    @test semquad_base_laplacian(pr1, zz) ≈ fd2 rtol=1e-5
    @test semquad_base_laplacian(pr1, zz) ≈
          (rs.alpha * rs.g / rs.cp)^2 * semquad_Tad(rs, zz) rtol=1e-14
    # with u = 0 and theta = 0 the only energy source is lap(Tad), and its
    # volume integral is (alpha g/cp) * (Tad(0) - Tad(1)) exactly
    _, _, _, Fc = assemble_semquad_energy(sq, qt, pr1, zeros(n), zeros(n);
                                          theta=zeros(n))
    @test sum(Fc) ≈ lapint rtol=1e-3

    # (6) Di, the heating switches, rho_heating/rho_cp, H and the reference
    # state are part of the checkpoint configuration
    tmp = tempname() * ".h5"
    qt2 = Quadtree(5)
    for _ in 1:2, lf in collect(qt2.leaves); refine!(qt2, lf); end
    balance!(qt2)
    ss = make_semstokes(qt2, 2)
    th0 = [0.05 * sinpi(x) * sinpi(z) for (x, z) in ss.su.coords]
    prc = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=2.0,
                 Di=0.3, heating_dissipation=true, refstate=rs)
    run_semquad_simulation(qt2, prc, th0;
        p=2, tstop=2.0e-6, dt0=1.0e-6, dtmax=1.0e-6, dtmin=1.0e-16,
        amr_every=0, ev=0.0, advection=:skew, linear=:direct,
        maxpic=1, pic_tol=1.0e-6, verbose=false,
        checkpoint_path=tmp, checkpoint_every=1)
    st = load_semquad_state(tmp)
    @test st.config["Di"] == "0.3"
    @test st.config["heating_dissipation"] == "true"
    @test st.config["refstate"] == "adiabat"
    @test st.config["ref_alpha"] == "0.5"
    pr_bad = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=2.0,
                    Di=0.4, heating_dissipation=true, refstate=rs)
    @test_throws ArgumentError run_semquad_simulation(qt2, pr_bad, Float64[];
        p=2, tstop=4.0e-6, dt0=1.0e-6, dtmax=1.0e-6, dtmin=1.0e-16,
        amr_every=0, ev=0.0, advection=:skew, linear=:direct,
        maxpic=1, pic_tol=1.0e-6, verbose=false, restart_from=tmp)
    rm(tmp; force=true)
end
