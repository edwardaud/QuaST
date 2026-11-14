# G197: TALA self-consistency — the analytic state and the discrete momentum balance.
using Test, QuaST, LinearAlgebra, SparseArrays

# G197 -- TALA self-consistency of the route-C (semquad) path
# (docs/dev/PHYSICS.md section 10).
#
# The formulation that is verified here is the one the code implements:
#   momentum   -div(2 eta eps(u)) + grad p = Ra rho(y) alpha(y) theta   (anomaly)
#   mass       div u = 0            (TALA, mass_form=:boussinesq)
#              div(rho u) = 0       (ALA,  mass_form=:anelastic)
#   energy     dtheta/dt + u.grad theta = lap(theta) - u_z dTad/dz + lap(Tad)
# with theta = T - Tad and (rho, alpha) from the reference state.  The reference
# hydrostatic pressure is never discretised: it is subtracted analytically, which
# is what test (1) measures.
@testset "G197 TALA self-consistency" begin
    RA = 1.0e6
    P = 2
    tree(nref) = begin
        qt = Quadtree(nref)
        for _ in 1:nref, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        qt
    end
    rs = SemRefState(Tsurf=0.2, alpha=1.0, cp=1.0, g=1.0, rho0=1.0)
    Lam = rs.alpha * rs.g / rs.cp
    pr = Params(Ra=RA, H=0.0, eta_law=:expT, deta_T=2.0, refstate=rs,
                Twall_top=semquad_Tad(rs, 1.0), Twall_bot=semquad_Tad(rs, 0.0))

    # (0) the analytic state itself: dTad/dz = -Lam*Tad, lap(Tad) = Lam^2 Tad
    for y in (0.0, 0.3, 0.5, 0.9, 1.0)
        @test semquad_dTad_dz(rs, y) ≈ -Lam * semquad_Tad(rs, y) rtol=1e-14
        h = 1.0e-4
        fd = (semquad_Tad(rs, y + h) - 2 * semquad_Tad(rs, y) +
              semquad_Tad(rs, y - h)) / h^2
        @test semquad_base_laplacian(pr, y) ≈ fd rtol=1e-5
        @test semquad_base_laplacian(pr, y) ≈ Lam^2 * semquad_Tad(rs, y) rtol=1e-14
    end

    # (1a) TALA anomaly form at theta = 0: the assembled load vector, the
    # velocity and the pressure are all exactly zero.
    q3 = tree(3)
    ss = make_semstokes(q3, P)
    fz0 = (x, y) -> RA * QuaST.semquad_rho(rs, y) *
                    QuaST.semquad_alpha(rs, y) * 0.0
    u0, p0v = solve_semquad_stokes_var(ss, q3, (x, y) -> 1.0,
                                       (x, y) -> 0.0, fz0,
                                       (x, y) -> 0.0, (x, y) -> 0.0;
                                       linear=:direct, bc=:free)
    @test maximum(abs, u0) == 0.0
    @test maximum(abs, p0v) == 0.0
    @test LAST_SEMSTOKES_RESIDUAL[] == 0.0

    # (1b) a resting *conductive* column: theta_c = (1-z) - Tad.  The exact
    # discrete balance p'(y) = fz(y) is checked by evaluating the momentum
    # residual at (u = 0, p = int fz); it is the part of the force the Q1
    # pressure space cannot represent, and it converges at second order.
    fzc = (x, y) -> RA * QuaST.semquad_rho(rs, y) *
                    QuaST.semquad_alpha(rs, y) * ((1.0 - y) - semquad_Tad(rs, y))
    pref_c(y) = RA * (y - y^2 / 2 +
                      rs.Tsurf * (exp(Lam * (1.0 - y)) - exp(Lam)))
    function hydro_res(qt, fz, pf)
        s = make_semstokes(qt, P)
        u, pv, fixed, cons2, cons_p, K, Bref, Cu, Cp, uu0, pp0, fv =
            solve_semquad_stokes_var(s, qt, (x, y) -> 1.0, (x, y) -> 0.0, fz,
                                     (x, y) -> 0.0, (x, y) -> 0.0;
                                     return_system=true, linear=:direct, bc=:free)
        Bs = -Cp' * Bref * Cu
        Bt = sparse(transpose(Bs))
        fr = Cu' * (fv - K * uu0 + Bref' * pp0)
        pn = [pf(y) for (_, y) in s.sp.coords]
        r = Bt * (Cp' * pn) - fr
        (res=maximum(abs, r) / maximum(abs, fr), uinf=maximum(abs, u))
    end
    r16 = hydro_res(tree(2), fzc, pref_c)
    r64 = hydro_res(tree(3), fzc, pref_c)
    @test r16.res < 2.0e-2                     # measured 7.29e-3
    @test r64.res < 3.0e-3                     # measured 1.91e-3
    @test 2.5 < r16.res / r64.res < 5.5        # measured 3.83 -> h^1.94
    @test r16.uinf > 5 * r64.uinf               # measured 14.6x
    @test r64.uinf < 1.0                       # measured 0.42

    # (1c) counterfactual: the *full* hydrostatic force.  With a linear p_ref
    # (constant rho) the discrete balance is exact; with a stratified rho it is
    # only O(h^2), which is what the analytic subtraction removes.
    fz1 = (x, y) -> -RA
    pf1(y) = -RA * y
    r1 = hydro_res(tree(3), fz1, pf1)
    @test r1.res < 1.0e-12                     # measured 1.5e-14
    @test r1.uinf < 1.0e-6                     # measured 1.0e-10
    fze = (x, y) -> -RA * exp(1.0 - y)
    pfe(y) = RA * (exp(1.0 - y) - exp(1.0))
    re16 = hydro_res(tree(2), fze, pfe)
    re64 = hydro_res(tree(3), fze, pfe)
    @test 2.5 < re16.res / re64.res < 5.5      # measured 3.76 -> h^1.91

    # (2) energy drift over 10 steps with u = 0.
    # (2a) T = Tad: drift = 10 dt lap(Tad) (first order in dt), i.e. the static
    # adiabat is NOT a steady state; the adiabatic-curvature source drives it.
    prA = pr
    sq = make_semquad(q3, P)
    z = [y for (_, y) in sq.coords]
    inner = [i for i in eachindex(z) if z[i] >= 1e-12 && z[i] <= 1 - 1e-12]
    function drift10(qtp, prp, th0; dt=1.0e-5)
        s = make_semquad(qtp, P)
        th = copy(th0); thp = copy(th0)
        for k in 1:10
            thn = solve_semquad_energy(s, qtp, prp, th, thp, zeros(s.N),
                                       zeros(s.N), dt, dt, k == 1)
            thp = th; th = thn
        end
        th
    end
    th1 = drift10(q3, prA, zeros(sq.N); dt=1.0e-5)
    th2 = drift10(q3, prA, zeros(sq.N); dt=5.0e-6)
    @test maximum(abs, th1[inner]) ≈ 2 * maximum(abs, th2[inner]) rtol=0.1
    lapint = Lam * (semquad_Tad(rs, 0.0) - semquad_Tad(rs, 1.0))
    bud = semquad_heat_budget(sq, q3, prA, th1;
                              ux=zeros(sq.N), uz=zeros(sq.N))
    @test 0.90 < bud.theta_int / (10 * 1.0e-5 * lapint) < 1.02   # measured 0.962
    pred = [10 * 1.0e-5 * semquad_base_laplacian(prA, y) for y in z]
    @test maximum(abs, (th1 .- pred)[inner]) / maximum(abs, pred[inner]) < 0.25

    # (2b) the conductive profile T = 1 - z IS steady: theta_c = (1-z) - Tad,
    # drift/||theta|| = 9.3e-8 at 16 leaves, 2.5e-8 at 64 (second order).
    prB = Params(Ra=RA, H=0.0, eta_law=:expT, deta_T=2.0, refstate=rs,
                 Twall_top=0.0, Twall_bot=1.0)
    for (nref, tol) in ((2, 3.0e-7), (3, 1.0e-7))
        qc = tree(nref)
        sc = make_semquad(qc, P)
        zc = [y for (_, y) in sc.coords]
        th0c = [(1.0 - y) - semquad_Tad(rs, y) for y in zc]
        thc = drift10(qc, prB, th0c)
        @test maximum(abs, thc .- th0c) / maximum(abs, th0c) < tol
    end

    # (3) ALA vs TALA on Ra = 1e6, 64 leaves, composite rheology, 5 steps.
    # Same reference state, same buoyancy, only the continuity equation differs.
    rs2 = SemRefState(Tsurf=0.2, alpha=1.0, cp=1.0, g=1.0, rho0=1.0,
                      rho_fun=y -> exp(1.0 - y))
    prc = Params(Ra=RA, H=0.0, eta_law=:composite,
                 comp_Ed=log(1.0e4), comp_Es=12.0, comp_n=3.0, eta_T0=0.5,
                 comp_Ad=exp(log(1.0e4) / 1.5), comp_As=1.0e-2 * exp(12.0 / 4.5),
                 eta_min=1.0e-6, eta_max=1.0e4, refstate=rs2,
                 Twall_top=semquad_Tad(rs2, 1.0),
                 Twall_bot=semquad_Tad(rs2, 0.0))
    qt = q3
    function conv(pq, fm; dt=2.0e-5)
        s = make_semquad(pq, P)
        th = [0.05 * sinpi(x) * sinpi(y) for (x, y) in s.coords]
        for (i, (_, y)) in enumerate(s.coords)
            (y <= 1e-14 || y >= 1.0 - 1e-14) && (th[i] = 0.0)
        end
        run_semquad_simulation(pq, prc, th; p=P, tstop=5 * dt, dt0=dt, dtmax=dt,
                               dtmin=1.0e-16, amr_every=0, ev=0.0,
                               advection=:skew, linear=:direct, maxpic=1,
                               pic_tol=1.0e-6, mass_form=fm, verbose=false)
    end
    rT = conv(qt, :boussinesq)
    rA = conv(qt, :anelastic)
    @test all(isfinite, rT.theta) && all(isfinite, rA.theta)
    uT = vcat(rT.ux, rT.uz); uA = vcat(rA.ux, rA.uz)
    B0 = semquad_stokes_B(make_semstokes(qt, P), qt)
    BA = semquad_stokes_B(make_semstokes(qt, P), qt;
                          rho_fun=y -> exp(1.0 - y), drho_fun=y -> -exp(1.0 - y))
    @test maximum(abs, B0 * uT) < 1.0e-10      # TALA satisfies div u = 0
    @test maximum(abs, BA * uA) < 1.0e-8       # ALA satisfies div(rho u) = 0
    @test maximum(abs, BA * uT) > 0.1          # ... and they are different
    @test maximum(abs, B0 * uA) > 0.1
    dV = (rA.history[end][5] - rT.history[end][5]) / rT.history[end][5]
    @test -0.15 < dV < -0.03                   # measured -7.56e-2
    @test abs((rA.history[end][3] - rT.history[end][3]) /
              rT.history[end][3]) < 1.0e-5     # measured 2.0e-7 (Nu)
    du = uA - uT
    @test maximum(abs, du) > 10.0              # measured 26.0 (max|u| 180)

    # with rho == 1 the two forms agree to round-off, and :boussinesq with a
    # constant rho is bitwise the no-rho_fun run
    pr1 = Params(Ra=RA, H=0.0, eta_law=:expT, deta_T=1.0,
                 refstate=SemRefState(Tsurf=0.2, alpha=1.0, cp=1.0, g=1.0,
                                      rho0=1.0, rho_fun=y -> 1.0),
                 Twall_top=semquad_Tad(rs, 1.0), Twall_bot=semquad_Tad(rs, 0.0))
    prN = Params(Ra=RA, H=0.0, eta_law=:expT, deta_T=1.0, refstate=rs,
                 Twall_top=semquad_Tad(rs, 1.0), Twall_bot=semquad_Tad(rs, 0.0))
    function conv1(pq, pp, fm)
        s = make_semquad(pq, P)
        th = [0.05 * sinpi(x) * sinpi(y) for (x, y) in s.coords]
        for (i, (_, y)) in enumerate(s.coords)
            (y <= 1e-14 || y >= 1.0 - 1e-14) && (th[i] = 0.0)
        end
        run_semquad_simulation(pq, pp, th; p=P, tstop=1.0e-4, dt0=2.0e-5,
                               dtmax=2.0e-5, dtmin=1.0e-16, amr_every=0,
                               ev=0.0, advection=:skew, linear=:direct,
                               maxpic=1, pic_tol=1.0e-6, mass_form=fm,
                               verbose=false)
    end
    a1 = conv1(qt, pr1, :anelastic)
    a2 = conv1(qt, pr1, :boussinesq)
    a3 = conv1(qt, prN, :boussinesq)
    @test maximum(abs, a1.theta - a2.theta) < 1.0e-13   # measured 2.6e-16
    @test a2.theta == a3.theta && a2.ux == a3.ux && a2.uz == a3.uz

    # the parameter guard and the default (no reference state) path
    @test_throws ArgumentError run_semquad_simulation(qt, prc,
        zeros(make_semquad(qt, P).N); p=P, tstop=1.0e-6, mass_form=:bogus,
        verbose=false)
    @test_throws ArgumentError solve_semquad_stokes_var(make_semstokes(qt, P), qt,
        (x, y) -> 1.0, (x, y) -> 0.0, (x, y) -> 0.0, (x, y) -> 0.0,
        (x, y) -> 0.0; mass_form=:bogus, linear=:direct)
end
