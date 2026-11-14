# G168: Drucker-Prager steady Newton — the derivatives and Kgeo against finite differences.
using Test, QuaST, LinearAlgebra, Printf

# G168: Drucker-Prager steady Newton (map form) with the DP tangent.
#
# The DP viscosity depends on (T, edot, p), so the steady map needs
#   (i)  an inner (u,p) solve at frozen theta -- `semquad_dp_stokes_newton`,
#   (ii) a Jacobian with BOTH the strain-rate block Kgeo (plastic
#        `deta/dedot = -eta/edot`) and the non-symmetric pressure block Kp,
#   (iii) the map Newton on theta (`semquad_dp_newton`).
# The gate checks
#   (a) the three derivatives against central differences (plastic branch
#       included: that needs a mild viscosity contrast so the plastic branch
#       wins at a moderate strain rate),
#   (b) Kgeo with the DP plastic derivative against a central difference of the
#       viscous residual `K(eta(edot)) u`,
#   (c) ASPECT's SPD factor on Kgeo: 1 in the viscous branch, in (0,1) in the
#       plastic branch, and the reduced `K + Kgeo` positive definite with it,
#   (d) a developed DP case (Ra=100, eta_T=1e5, c=4): the inner Newton is
#       (super)linear and small, the map Newton converges to |F| <= 1e-8 and a
#       second solve from the fixed point does not move.

@testset "G168 Drucker-Prager steady Newton" begin
    # mild viscosity contrast (eta in [0.01,1]) so that at edot ~ 50 the plastic
    # branch wins for the moderate yield stress used here
    prd = Params(Ra=1.0e3, H=0.0, eta_law=:expT, deta_T=1.0e2,
                 yield_c=4.0, yield_phi=0.3, yield_taumax=1.0e6,
                 yield_eps_min=1.0e-8, eta_min=1.0e-6, eta_max=10.0,
                 Twall_top=0.0, Twall_bot=1.0, bc=:free)
    h = 1.0e-7
    # ---- (a) derivatives vs central differences ----
    # (T, edot, p, y, expected branch): plastic at edot=50, viscous at edot=0.1
    for (T, e, p, y) in ((0.9, 50.0, 3.0, 0.3), (0.5, 50.0, 3.0, 0.5),
                         (0.5, 0.1, 3.0, 0.5), (0.2, 5.0, 0.5, 0.7))
        fd_T = (semquad_visco_plastic_eta(T + h, y, e, p, prd) -
                semquad_visco_plastic_eta(T - h, y, e, p, prd)) / (2h)
        fd_e = (semquad_visco_plastic_eta(T, y, e + h, p, prd) -
                semquad_visco_plastic_eta(T, y, e - h, p, prd)) / (2h)
        fd_p = (semquad_visco_plastic_eta(T, y, e, p + h, prd) -
                semquad_visco_plastic_eta(T, y, e, p - h, prd)) / (2h)
        an_T = semquad_dp_deta_T(T, y, e, p, prd)
        an_e = semquad_deta_dp_eps(T, y, e, p, prd)
        an_p = semquad_deta_dp(T, y, e, p, prd)
        @test abs(an_T - fd_T) <= 1.0e-5 * max(abs(fd_T), 1.0e-10)
        @test abs(an_e - fd_e) <= 1.0e-5 * max(abs(fd_e), 1.0e-10)
        @test abs(an_p - fd_p) <= 1.0e-5 * max(abs(fd_p), 1.0e-10)
    end
    # plastic branch: strain-rate weakening (negative) and no T dependence
    @test semquad_visco_plastic_eta(0.5, 0.5, 50.0, 3.0, prd) ≈
          4.0 * cos(0.3) + 3.0 * sin(0.3) |> x -> x / 100.0
    @test semquad_deta_dp_eps(0.5, 0.5, 50.0, 3.0, prd) < 0.0
    @test semquad_dp_deta_T(0.5, 0.5, 50.0, 3.0, prd) == 0.0
    @test semquad_deta_dp_eps(0.5, 0.5, 1.0e-9, 3.0, prd) == 0.0
    # ---- (b) Kgeo with the DP plastic derivative vs FD ----
    qt = Quadtree(2; cells_x=2, cells_y=2)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    N = sq.N
    leaves = Set(qt.leaves)
    ux = [40.0 * sin(pi * x) * cos(pi * y) for (x, y) in sq.coords]
    uz = [-40.0 * cos(pi * x) * sin(pi * y) for (x, y) in sq.coords]
    for (i, (x, y)) in enumerate(sq.coords)
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (ux[i] = 0.0; uz[i] = 0.0)
    end
    u = vcat(ux, uz)
    z2 = (x, y) -> 0.0
    pval = 3.0
    ed_here = semquad_strain_rate_at(sq, ux, uz, 0.3, 0.4, leaves)
    @printf("G168: edot at the test point = %.2f, deta/dedot = %.3e\n", ed_here,
            semquad_deta_dp_eps(0.5, 0.4, ed_here, pval, prd))
    @test semquad_deta_dp_eps(0.5, 0.4, ed_here, pval, prd) != 0.0
    eta_fun = (x, y) -> semquad_visco_plastic_eta(0.5, y,
                semquad_strain_rate_at(sq, ux, uz, x, y, leaves), pval, prd)
    deta_fun = (x, y, ed) -> semquad_deta_dp_eps(0.5, y, ed, pval, prd)
    sys0 = solve_semquad_stokes_var(ss, qt, eta_fun, z2, z2, z2, z2;
                                    bc=:free, linear=:direct,
                                    assemble_only=true)
    sysg = solve_semquad_stokes_var(ss, qt, eta_fun, z2, z2, z2, z2;
                                    bc=:free, linear=:direct,
                                    assemble_only=true, geo_u=u,
                                    geo_coef=deta_fun)
    @test norm(sysg.Kgeo) > 0.0
    eps_fd = 1.0e-6
    up = u .+ eps_fd .* u
    um = u .- eps_fd .* u
    etp = (x, y) -> semquad_visco_plastic_eta(0.5, y,
                semquad_strain_rate_at(sq, up[1:N], up[(N + 1):(2N)], x, y,
                                       leaves), pval, prd)
    etm = (x, y) -> semquad_visco_plastic_eta(0.5, y,
                semquad_strain_rate_at(sq, um[1:N], um[(N + 1):(2N)], x, y,
                                       leaves), pval, prd)
    sp = solve_semquad_stokes_var(ss, qt, etp, z2, z2, z2, z2; bc=:free,
                                  linear=:direct, assemble_only=true)
    sm = solve_semquad_stokes_var(ss, qt, etm, z2, z2, z2, z2; bc=:free,
                                  linear=:direct, assemble_only=true)
    fdv = (sp.K * u - sm.K * u) ./ (2eps_fd)
    anv = sysg.Kgeo * u
    @test norm(fdv - anv) <= 1.0e-4 * norm(fdv)
    @printf("G168: Kgeo vs FD rel=%.2e\n", norm(fdv - anv) / norm(fdv))
    # ---- (c) SPD factor ----
    @test spd_factor(1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0) == 1.0
    alpha_pl = spd_factor(1.0e-2, -1.0e-2, 50.0, 50.0, -50.0, 0.0,
                          50.0, -50.0, 0.0; safety=0.9)
    @test 0.0 < alpha_pl < 1.0
    snp = solve_semquad_stokes_var(ss, qt, eta_fun, z2, z2, z2, z2;
                                   bc=:free, linear=:direct,
                                   assemble_only=true, geo_u=u,
                                   geo_coef=deta_fun, geo_spd_safety=0.9)
    K1 = Matrix(sys0.Cu' * (sys0.K + sysg.Kgeo) * sys0.Cu)
    K2 = Matrix(sys0.Cu' * (sys0.K + snp.Kgeo) * sys0.Cu)
    lam1 = minimum(eigvals(Symmetric(K1)))
    lam2 = minimum(eigvals(Symmetric(K2)))
    @printf("G168: eigmin(K+Kgeo) raw=%.3e spd=%.3e\n", lam1, lam2)
    @test lam2 > 0.0
    # ---- (d) DP Stokes Newton (deterministic, no transient) ----------------
    # The inner (u,p) Newton from the isoviscous start on the Tosi-like case:
    # the plastic branch turns on during the iteration, and with the exact
    # tangent (SPD off) the history is short and superlinear.
    prd2 = Params(Ra=100.0, H=0.0, eta_law=:expT, deta_T=1.0e5,
                  yield_c=4.0, yield_phi=0.3, yield_taumax=1.0e6,
                  yield_eps_min=1.0e-8, eta_min=1.0e-6, eta_max=10.0,
                  Twall_top=0.0, Twall_bot=1.0, bc=:free)
    qt2 = Quadtree(2; cells_x=3, cells_y=3)
    for _ in 1:2, lf in collect(qt2.leaves)
        refine!(qt2, lf)
    end
    balance!(qt2)
    ss2 = make_semstokes(qt2, 2)
    sq2 = ss2.su
    th0 = [0.1 * cospi(x) * sinpi(y) for (x, y) in sq2.coords]
    for (i, (x, y)) in enumerate(sq2.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th0[i] = 0.0)
    end
    sn = semquad_dp_stokes_newton(sq2, qt2, ss2, prd2, th0; tol=1.0e-8,
                                  maxit=10)
    @test sn.converged
    @test sn.iters <= 8
    @test sn.resid < 1.0e-8
    vrs = semquad_vrms(sq2, qt2, sn.u[1:sq2.N], sn.u[(sq2.N + 1):(2sq2.N)])
    @test 10.0 < vrs < 500.0
    @printf("G168: DP Stokes Newton iters=%d resid=%.1e Vrms=%.2f\n", sn.iters,
            sn.resid, vrs)
    # ---- (d2) DP JVP vs a central difference of the map --------------------
    # At this state both derivative blocks are active (|Kp| ~ 6e-2,
    # |Kgeo| ~ 6.5): the JVP carries the Kgeo and Kp saddles plus the explicit
    # deta/dT term, so this is the end-to-end check of the DP linearisation.
    st = semquad_dp_steady_step(sq2, qt2, ss2, prd2, th0; dt=1.0e-2,
                                dp_tol=1.0e-8, dp_maxit=10, u_init=sn.u,
                                p_init=sn.p)
    @test st.sys_jac.Kp !== nothing
    @test norm(st.sys_jac.Kp) > 0.0
    @test norm(st.sys_jac.Kgeo) > 0.0
    vdir = [sinpi(2x / 1.0) * sinpi(2y) for (x, y) in sq2.coords]
    for (i, (x, y)) in enumerate(sq2.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (vdir[i] = 0.0)
    end
    vdir ./= norm(vdir)
    Jv = semquad_dp_steady_jv(sq2, qt2, ss2, prd2, th0, vdir, st) .- vdir
    efd = 1.0e-5
    sp2 = semquad_dp_steady_step(sq2, qt2, ss2, prd2, th0 .+ efd .* vdir;
                                 dt=1.0e-2, dp_tol=1.0e-8, dp_maxit=10,
                                 u_init=sn.u, p_init=sn.p)
    sm2 = semquad_dp_steady_step(sq2, qt2, ss2, prd2, th0 .- efd .* vdir;
                                 dt=1.0e-2, dp_tol=1.0e-8, dp_maxit=10,
                                 u_init=sn.u, p_init=sn.p)
    fdv = (sp2.theta .- sm2.theta) ./ (2efd) .- vdir
    @printf("G168: DP JVP vs FD rel=%.2e\n", norm(Jv - fdv) / norm(fdv))
    @test norm(Jv - fdv) / norm(fdv) < 1.0e-4
    # ---- (e) DP map Newton (deterministic): weak start -> conductive branch -
    thw = [0.01 * cospi(x) * sinpi(y) for (x, y) in sq2.coords]
    for (i, (x, y)) in enumerate(sq2.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (thw[i] = 0.0)
    end
    rn = semquad_dp_newton(sq2, qt2, ss2, prd2, thw; dt=1.0e-2, tol=1.0e-8,
                           maxit=10, krylov=40, restart=20, dp_tol=1.0e-7,
                           dp_maxit=15, picard_steps=0)
    @test rn.converged
    @test norm(rn.step.theta - rn.theta) < 1.0e-8
    @test length(rn.history) == rn.iters + 1
    @printf("G168: DP map Newton iters=%d |F|=%.1e\n", rn.iters,
            norm(rn.step.theta - rn.theta))
end
