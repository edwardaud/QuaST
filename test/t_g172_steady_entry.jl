# G172: the uniform steady-state entry point and Drucker-Prager in the time driver.
using Test, QuaST, LinearAlgebra, Printf

# G172: uniform steady-state entry + Drucker-Prager in the time driver.
#
# (a) `semquad_steady_newton` must pick the Newton variant from the rheology
#     (:map for eta(T), :dp when yielding is active, :tosi for a TosiLaw) and
#     return bit-identical results to the direct call;
# (b) the unsteady driver must run the Drucker-Prager yielding path: with the
#     plastic damper (G170) the state yields (a positive fraction of nodes on
#     the plastic branch) and the run stays bounded.  Without the damper the
#     same case is the runaway documented in G170/G171.

@testset "G172 steady entry and DP in the driver" begin
    qt = Quadtree(2; cells_x=3, cells_y=3)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    N = sq.N
    th = [0.1 * cospi(x) * sinpi(y) for (x, y) in sq.coords]
    for (i, (x, y)) in enumerate(sq.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th[i] = 0.0)
    end
    # ---- (a) dispatch ----
    pr_map = Params(Ra=1.0e4, H=0.0, eta_law=:expTz,
                    eta_b=9.70406052783923433184,
                    eta_c=4.1588830833596718565,
                    Twall_top=0.0, Twall_bot=1.0, bc=:free)
    r1 = semquad_steady_newton(sq, qt, ss, pr_map, th; dt=1.0e-2, tol=1.0e-8,
                               maxit=4, krylov=40, restart=20)
    r1b = semquad_newton_steady(sq, qt, ss, pr_map, th; dt=1.0e-2,
                                tol=1.0e-8, maxit=4, krylov=40, restart=20)
    @test r1.kind === :map
    @test r1.theta == r1b.theta
    pr_dp = Params(Ra=100.0, H=0.0, eta_law=:expT, deta_T=1.0e5,
                   yield_c=4.0, yield_phi=0.3, yield_taumax=1.0e6,
                   yield_eps_min=1.0e-8, eta_min=1.0e-6, eta_max=10.0,
                   Twall_top=0.0, Twall_bot=1.0, bc=:free)
    thw = 0.01 .* th
    r2 = semquad_steady_newton(sq, qt, ss, pr_dp, thw; dt=1.0e-2, tol=1.0e-8,
                               maxit=8, krylov=40, restart=20,
                               dp_tol=1.0e-7, dp_maxit=15)
    r2b = semquad_dp_newton(sq, qt, ss, pr_dp, thw; dt=1.0e-2, tol=1.0e-8,
                            maxit=8, krylov=40, restart=20, dp_tol=1.0e-7,
                            dp_maxit=15)
    @test r2.kind === :dp
    @test r2.theta == r2b.theta
    @test r2.converged
    # TosiLaw detection
    law = TosiLaw(etaT=1.0e5, etaZ=1.0, sigma_y=4.0, eta_star=0.0,
                  eta_min=1.0e-6, eta_max=10.0, eta_init=1.0,
                  z_is_depth=false, eta0=1.0)
    pr_t = Params(Ra=1.0e2, H=0.0, eta_law=:tosi, tosi=law,
                  eta_min=1.0e-6, eta_max=10.0,
                  Twall_top=0.0, Twall_bot=1.0, bc=:free)
    r3 = semquad_steady_newton(sq, qt, ss, pr_t, thw; dt=1.0e-2, tol=1.0e-8,
                               maxit=3, krylov=40, restart=20,
                               tosi_tol=1.0e-3, tosi_maxit=3)
    r3b = semquad_tosi_newton(sq, qt, ss, pr_t, thw; dt=1.0e-2, tol=1.0e-8,
                              maxit=3, krylov=40, restart=20,
                              tosi_tol=1.0e-3, tosi_maxit=3)
    @test r3.kind === :tosi
    @test r3.theta == r3b.theta
    @test_throws ArgumentError semquad_steady_newton(sq, qt, ss, pr_map, th;
                                                     kind=:nope)
    # ---- (b) DP yielding in the time driver ----
    prd = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0e3,
                 yield_c=1.0, yield_phi=0.3, yield_taumax=1.0e6,
                 yield_eps_min=1.0e-8, yield_damper=1.0e-2,
                 eta_min=1.0e-4, eta_max=10.0,
                 Twall_top=0.0, Twall_bot=1.0, bc=:free)
    th0 = [0.05 * cospi(x) * sinpi(y) for (x, y) in sq.coords]
    for (i, (x, y)) in enumerate(sq.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th0[i] = 0.0)
    end
    res = run_semquad_simulation(qt, prd, th0; p=2, tstop=2.0e-3, dt0=1.0e-6,
                                 dtmax=1.0e-4, cfl=2.0, maxsteps=400,
                                 verbose=false, ev=0.0, advection=:skew,
                                 maxpic=1, linear=:direct)
    @test res.steps > 5
    @test isfinite(semquad_vrms(sq, res.qt, res.ux, res.uz))
    # plastic fraction at the final state
    leaves = Set(res.qt.leaves)
    npl = 0
    for (x, y) in sq.coords
        ed = semquad_strain_rate_at(sq, res.ux, res.uz, x, y, leaves)
        T = Tbar(y, prd.H) + QuaST._semquad_eval_at(sq, res.theta, x, y,
                                                       leaves)
        pq = QuaST._semquad_eval_at(ss.sp, res.pv, x, y,
                                        Set(ss.sp.leaves))
        eta_v, _ = eta_at(T, y, prd)
        tau_y = semquad_yield_stress(pq, prd)
        tau_y / (2.0 * max(ed, prd.yield_eps_min)) < eta_v && (npl += 1)
    end
    frac = npl / length(sq.coords)
    vr = semquad_vrms(sq, res.qt, res.ux, res.uz)
    @printf("G172: driver DP: %d steps, plastic fraction=%.2f, Vrms=%.1f\n",
            res.steps, frac, vr)
    @test frac > 0.0
    @test vr < 1.0e5
    # ---- (c) the driver can hand off to the steady entry ----
    res2 = run_semquad_simulation(qt, prd, res.theta; p=2, tstop=1.5e-3,
                                  dt0=1.0e-6, dtmax=1.0e-4, cfl=2.0,
                                  maxsteps=200, verbose=false, ev=0.0,
                                  advection=:skew, maxpic=1, linear=:direct,
                                  steady_newton=:dp)
    @test res2.steady !== nothing
    @test res2.steady.kind === :dp
    @test isfinite(norm(res2.steady.theta))
    @printf("G172: driver steady_newton=:dp conv=%s it=%d\n",
            res2.steady.converged, res2.steady.iters)
end
