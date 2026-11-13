# Entropy-viscosity gates (ASPECT form).
#
# Every expected value below is derived from the formulas in
# source/simulator/entropy_viscosity.cc and
# source/simulator/assemblers/advection.cc, not from a previous run of this
# code.  The tests exercise the internal helpers directly so a formula change
# fails here before it can silently change a benchmark.
using Test
using LinearAlgebra
using QuaST

const MSEM = QuaST

function _ev_case(n=4)
    m = make_mesh(1.0, n, n, 4; zgrading=:uniform)
    fd = make_fedata(4, m.q)
    pr = Params(Ra=1.0e4, eta_const=1.0)
    return m, fd, pr
end

@testset "G11 EV global quantities (analytic)" begin
    m, fd, pr = _ev_case(4)
    theta = zeros(Float64, m.Nscalar)
    ux = fill(1.0, m.Nscalar)      # |u| = 1 everywhere
    uz = zeros(Float64, m.Nscalar)
    g = MSEM._ev_globals(m, fd, pr, nothing, theta, theta, ux, uz, 0.0, 0.0)
    # T = Tbar = 1-z on the unit box
    @test g.minT ≈ 0.0 atol=1e-12
    @test g.maxT ≈ 1.0 atol=1e-12
    @test g.avgT ≈ 0.5 atol=1e-12
    @test g.field_variation ≈ 1.0 atol=1e-12
    @test g.u_infty ≈ 1.0 atol=1e-12
    # e=(T-1/2)^2=(1/2-z)^2: <e>=1/12 exactly (Gauss integrates the quartic)
    @test g.avg_entropy ≈ 1.0 / 12.0 rtol=1e-12
    # The extrema are taken at the Gauss points, as in ASPECT, so compare with
    # an independent loop over the same points rather than with the continuous
    # maximum 1/4 (which is attained only on the boundary).
    emin = Inf; emax = -Inf
    for ez in 0:m.nze-1
        hze = z_size(m, ez); z0 = z_left(m, ez)
        for qy in 1:fd.nq
            z = z0 + 0.5 * (fd.gl_x[qy] + 1.0) * hze
            e = ((1.0 - z) - g.avgT)^2
            emin = min(emin, e); emax = max(emax, e)
        end
    end
    @test g.entropy_variation ≈ max(emax - g.avg_entropy,
                                    g.avg_entropy - emin) rtol=1e-12
    @test 0.0 < g.entropy_variation < 1.0 / 6.0
end

@testset "G11 EV residual formula" begin
    m, fd, pr = _ev_case(4)
    theta = zeros(Float64, m.Nscalar)
    w = 0.3
    ux = zeros(Float64, m.Nscalar)
    uz = fill(w, m.Nscalar)
    dt = 1.0e-3
    g = MSEM._ev_globals(m, fd, pr, nothing, theta, theta, ux, uz, 0.0, 0.0)
    ez, ex = 1, 1
    # heating off: R = |rho_cp*(u.grad T)|*|T-avg| = w*|T-1/2|
    # T=1-z at the Gauss points of the cell
    z0 = z_left(m, ez); hze = z_size(m, ez)
    R_exp = 0.0
    for qy in 1:fd.nq
        z = z0 + 0.5 * (fd.gl_x[qy] + 1.0) * hze
        R_exp = max(R_exp, w * abs((1.0 - z) - g.avgT))
    end
    Rmax, umax, pref, kmax = MSEM._ev_cell_stats(
        m, fd, pr, nothing, theta, theta, ux, uz, dt, ez, ex, g;
        heating_adiabatic=false, heating_dissipation=false, alpha=2)
    @test Rmax ≈ R_exp rtol=1e-12
    @test umax ≈ w rtol=1e-12
    @test pref ≈ 1.0 rtol=1e-12
    @test kmax ≈ 1.0 rtol=1e-12

    # adiabatic heating on: R = |(-w) - (-Di*T*w)|*|T-avg| = w*|Di*T-1|*|T-avg|
    Di = 1.0
    R_exp2 = 0.0
    for qy in 1:fd.nq
        z = z0 + 0.5 * (fd.gl_x[qy] + 1.0) * hze
        T = 1.0 - z
        R_exp2 = max(R_exp2, w * abs(Di * T - 1.0) * abs(T - g.avgT))
    end
    Rmax2, _, _, _ = MSEM._ev_cell_stats(
        m, fd, pr, nothing, theta, theta, ux, uz, dt, ez, ex, g;
        heating_adiabatic=true, heating_dissipation=false, Di=Di, alpha=2)
    @test Rmax2 ≈ R_exp2 rtol=1e-12
end

@testset "G11 EV nu_max / nu_e / boundary" begin
    m, fd, pr = _ev_case(4)
    theta = zeros(Float64, m.Nscalar)
    w = 0.3
    ux = zeros(Float64, m.Nscalar)
    uz = fill(w, m.Nscalar)
    dt = 1.0e-3
    g = MSEM._ev_globals(m, fd, pr, nothing, theta, theta, ux, uz, 0.0, 0.0)
    cR, beta = 0.033, 0.005
    ez, ex = 1, 1
    Rmax, umax, pref, _ = MSEM._ev_cell_stats(
        m, fd, pr, nothing, theta, theta, ux, uz, dt, ez, ex, g;
        heating_adiabatic=false, heating_dissipation=false, alpha=2)
    he = sqrt(m.hx^2 + z_size(m, ez)^2)
    nu_max_exp = beta * pref * umax * he
    nu_e_exp = cR * he^2 * Rmax / g.entropy_variation
    nu_e, nu_max, nu = MSEM._ev_nu_cell(
        m, fd, pr, nothing, theta, theta, ux, uz, dt, ez, ex, g;
        ev=cR, ev_cmax=beta, ev_alpha=2, ev_timestep=5, ev_length=1.0)
    @test nu_max ≈ nu_max_exp rtol=1e-12
    @test nu_e ≈ nu_e_exp rtol=1e-12
    @test nu ≈ min(nu_max_exp, nu_e_exp) rtol=1e-12

    # first two timesteps: nu = nu_max
    _, _, nu0 = MSEM._ev_nu_cell(
        m, fd, pr, nothing, theta, theta, ux, uz, dt, ez, ex, g;
        ev=cR, ev_cmax=beta, ev_alpha=2, ev_timestep=0, ev_length=1.0)
    @test nu0 ≈ nu_max_exp rtol=1e-12

    # Dirichlet-T boundary cells (ez=0 and ez=nze-1) get zero
    for ezbd in (0, m.nze - 1)
        _, _, nub = MSEM._ev_nu_cell(
            m, fd, pr, nothing, theta, theta, ux, uz, dt, ezbd, 0, g;
            ev=cR, ev_cmax=beta, ev_alpha=2, ev_timestep=5, ev_length=1.0)
        @test nub == 0.0
    end
end

@testset "G11 EV diffusion = max(k, nu)" begin
    m, fd, pr = _ev_case(4)
    dmt = make_theta_map(m)
    theta = zeros(Float64, m.Nscalar)
    ux = zeros(Float64, m.Nscalar)
    uz = zeros(Float64, m.Nscalar)
    kw = (; theta_now=theta, theta_prev=theta, dt=1.0e-3,
          heating_adiabatic=false, heating_dissipation=false)
    K0 = assemble_energy(m, fd, pr, ux, uz, dmt; ev=0.0, kw...).K
    # nu_const below k -> max(k,nu)=k, matrix unchanged
    K1 = assemble_energy(m, fd, pr, ux, uz, dmt; ev=0.0, nu_const=0.5, kw...).K
    @test K1 == K0
    # nu_const above k -> max(k,nu)=nu, matrix scaled exactly by nu/k
    K2 = assemble_energy(m, fd, pr, ux, uz, dmt; ev=0.0, nu_const=2.0, kw...).K
    @test K2 == 2.0 * K0
end

@testset "G11 EV active path reaches K" begin
    # Plumbing test only: a deliberately huge cR makes nu exceed k so the
    # entropy-viscosity branch (not the nu_const branch) must change K.
    m, fd, pr = _ev_case(4)
    dmt = make_theta_map(m)
    xs, zs = node_coords(m, fd.gll_v)
    theta = [0.05 * sinpi(xs[mod(k-1,m.nvx)+1]) * sinpi(zs[div(k-1,m.nvx)+1])
             for k in 1:m.Nscalar]
    w = 0.2
    ux = zeros(Float64, m.Nscalar)
    uz = fill(w, m.Nscalar)
    kw = (; theta_now=theta, theta_prev=theta, dt=1.0e-3,
          heating_adiabatic=false, heating_dissipation=false, ev_timestep=5)
    K0 = assemble_energy(m, fd, pr, ux, uz, dmt; ev=0.0, kw...).K
    KA = assemble_energy(m, fd, pr, ux, uz, dmt; ev=1.0e6, ev_cmax=100.0, kw...).K
    @test norm(KA - K0) > 0.0
    @test KA ≈ KA'          # diffusion operator stays symmetric
end

@testset "G11 EV alpha=1 branch" begin
    m, fd, pr = _ev_case(4)
    theta = zeros(Float64, m.Nscalar)
    w = 0.3
    ux = zeros(Float64, m.Nscalar)
    uz = fill(w, m.Nscalar)
    dt = 1.0e-3
    g = MSEM._ev_globals(m, fd, pr, nothing, theta, theta, ux, uz, 0.0, 0.0)
    ez, ex = 1, 1
    Rmax, umax, pref, _ = MSEM._ev_cell_stats(
        m, fd, pr, nothing, theta, theta, ux, uz, dt, ez, ex, g;
        heating_adiabatic=false, heating_dissipation=false, alpha=1)
    he = sqrt(m.hx^2 + z_size(m, ez)^2)
    cR, beta, Omega = 0.033, 0.005, 1.0
    nu_max_exp = beta * pref * umax * he
    nu_e_exp = cR * he * Omega * umax * Rmax / (g.u_infty * g.field_variation)
    nu_e, nu_max, nu = MSEM._ev_nu_cell(
        m, fd, pr, nothing, theta, theta, ux, uz, dt, ez, ex, g;
        ev=cR, ev_cmax=beta, ev_alpha=1, ev_timestep=5, ev_length=Omega)
    @test nu_max ≈ nu_max_exp rtol=1e-12
    @test nu_e ≈ nu_e_exp rtol=1e-12
    @test nu ≈ min(nu_max_exp, nu_e_exp) rtol=1e-12
end

@testset "G11 EV strain-rate term in max_velocity" begin
    m, fd, pr = _ev_case(4)
    xs, zs = node_coords(m, fd.gll_v)
    th = zeros(Float64, m.Nscalar)
    # u_x = w*z -> eps_xz = w/2, so ||eps|| = w/sqrt(2) everywhere
    w = 0.4
    ux = [w * zs[div(k-1,m.nvx)+1] for k in 1:m.Nscalar]
    uz = zeros(Float64, m.Nscalar)
    g = MSEM._ev_globals(m, fd, pr, nothing, th, th, ux, uz, 0.0, 0.0)
    ez, ex = 1, 1
    he = sqrt(m.hx^2 + z_size(m, ez)^2)
    gamma = 0.1
    _, umax, _, _ = MSEM._ev_cell_stats(
        m, fd, pr, nothing, th, th, ux, uz, 1.0e-3, ez, ex, g;
        heating_adiabatic=false, heating_dissipation=false, alpha=2,
        gamma_stab=gamma)
    # max over the Gauss points of |u| + gamma*||eps||*h
    umax_exp = 0.0
    z0 = z_left(m, ez); hze = z_size(m, ez)
    for qy in 1:fd.nq
        z = z0 + 0.5 * (fd.gl_x[qy] + 1.0) * hze
        umax_exp = max(umax_exp, abs(w * z) + gamma * (w / sqrt(2.0)) * he)
    end
    @test umax ≈ umax_exp rtol=1e-12
end

@testset "G11 EV zero-velocity branch" begin
    m, fd, pr = _ev_case(4)
    th = zeros(Float64, m.Nscalar)
    z = zeros(Float64, m.Nscalar)
    g = MSEM._ev_globals(m, fd, pr, nothing, th, th, z, z, 0.0, 0.0)
    @test g.u_infty == 0.0
    ez, ex = 1, 1
    he = sqrt(m.hx^2 + z_size(m, ez)^2)
    beta, kmax, Omega = 0.005, 1.0, 1.0
    _, _, nu = MSEM._ev_nu_cell(
        m, fd, pr, nothing, th, th, z, z, 1.0e-3, ez, ex, g;
        ev=0.033, ev_cmax=beta, ev_alpha=2, ev_timestep=5, ev_length=Omega)
    @test nu ≈ beta * kmax / Omega * he rtol=1e-12
end

@testset "G11 EV extrapolated field range" begin
    m, fd, pr = _ev_case(4)
    th_now = zeros(Float64, m.Nscalar)
    th_prev = fill(0.1, m.Nscalar)
    ux = ones(Float64, m.Nscalar)
    uz = zeros(Float64, m.Nscalar)
    ratio = 1.0
    g = MSEM._ev_globals(m, fd, pr, nothing, th_now, th_prev, ux, uz, ratio, 0.0)
    # T_extrap = (1+r) T_now - r T_prev = 2*(1-z) - (0.1 + 1-z) = 1-z-0.1
    @test g.minT ≈ -0.1 rtol=1e-12
    @test g.maxT ≈ 0.9 rtol=1e-12
    @test g.avgT ≈ 0.4 rtol=1e-12
    @test g.field_variation ≈ 1.0 rtol=1e-12
end

@testset "G11 EV residual uses the second derivative" begin
    # The EV residual needs lap(theta).  `lagrange_d2` returns (L, dL, d2), and
    # the SEM cell-stats code used to destructure `_, d2x = ...`, i.e. it put
    # dL into the Laplacian.  A quadratic field with u = 0 and theta_prev =
    # theta has dTdt = adv = 0 and lap = 2, so the alpha=1 residual is exactly 2;
    # with dL in place of d2 it is not.
    qt = Quadtree(2)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    sq = make_semquad(qt, 3)
    th = [x^2 for (x, y) in sq.coords]
    z = zeros(Float64, sq.N)
    pr = Params(Ra=1.0e4, H=0.0)
    evg = MSEM.semquad_ev_globals(sq, qt, pr, th, th, z, z, 1.0)
    lf = first(sq.leaves)
    Rmax, umax = MSEM.semquad_ev_cell_stats(sq, pr, th, th, z, z, 1.0e-3,
                                            lf, evg; alpha=1)
    @test umax == 0.0
    @test Rmax ≈ 2.0 atol=1e-12
end
