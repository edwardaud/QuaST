# G170: ASPECT's plastic damper for the Drucker-Prager law — the formula and its derivatives.
using Test, QuaST, LinearAlgebra, Printf

# G170: ASPECT's plastic damper for the Drucker-Prager law.
#
#     eta_pl_damped = (eta_d + eta_pl) / (1 + eta_d / eta_v)
#
# (`use_plastic_damper` in ASPECT `drucker_prager.cc`).  `eta_d = 0` must be bit
# identical to the historical hard `min(eta_v, eta_pl)` law; `eta_d > 0` raises
# the yielded viscosity and, because the denominator carries `eta_v(T)`, gives
# the yielded branch a temperature dependence.  The gate checks
#   (a) the viscosity law: no effect in the viscous branch, monotone increase in
#       eta_d in the yielded branch, and the damper never crosses eta_v;
#   (b) the three derivatives against central differences with the damper on;
#   (c) the point of the damper: a *cold* DP Stokes Newton that stalls without
#       it converges in a handful of iterations with it.

@testset "G170 plastic damper" begin
    base = (Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0e3, yield_c=1.0,
            yield_phi=0.3, yield_taumax=1.0e6, yield_eps_min=1.0e-8,
            eta_min=1.0e-4, eta_max=10.0, Twall_top=0.0, Twall_bot=1.0,
            bc=:free)
    pr0 = Params(; base..., yield_damper=0.0)
    prd = Params(; base..., yield_damper=1.0e-2)
    prD = Params(; base..., yield_damper=1.0e-1)
    # ---- (a) viscosity law ----
    # viscous branch (eta_pl > eta_v): the damper must not change anything
    Tv, ev = 0.9, 1.0e-2
    eta_v = eta_at(Tv, 0.2, pr0)[1]
    @test semquad_visco_plastic_eta(Tv, 0.2, ev, 3.0, pr0) ==
          semquad_visco_plastic_eta(Tv, 0.2, ev, 3.0, prd)
    @test semquad_visco_plastic_eta(Tv, 0.2, ev, 3.0, prd) == eta_v
    # yielded branch: monotone in eta_d, and always below eta_v
    for (T, e, p, y) in ((0.5, 20.0, 1.0, 0.5), (0.2, 5.0, 0.5, 0.7),
                         (0.4, 50.0, 3.0, 0.3))
        e0 = semquad_visco_plastic_eta(T, y, e, p, pr0)
        e1 = semquad_visco_plastic_eta(T, y, e, p, prd)
        e2 = semquad_visco_plastic_eta(T, y, e, p, prD)
        @test e1 >= e0
        @test e2 >= e1
        @test e2 < eta_at(T, y, pr0)[1] + 1.0e-15
    end
    # the damper formula itself
    ev_ = eta_at(0.4, 0.3, pr0)[1]
    ty = semquad_yield_stress(3.0, prd)
    epl = ty / (2.0 * 50.0)
    @test semquad_visco_plastic_eta(0.4, 0.3, 50.0, 3.0, prd) ≈
          (1.0e-2 + epl) / (1.0 + 1.0e-2 / ev_)
    # ---- (b) derivatives vs central differences with the damper ----
    h = 1.0e-7
    for (T, e, p, y) in ((0.5, 20.0, 1.0, 0.5), (0.2, 5.0, 0.5, 0.7),
                         (0.4, 50.0, 3.0, 0.3), (0.7, 20.0, 1.0, 0.5))
        fd_T = (semquad_visco_plastic_eta(T + h, y, e, p, prd) -
                semquad_visco_plastic_eta(T - h, y, e, p, prd)) / (2h)
        fd_e = (semquad_visco_plastic_eta(T, y, e + h, p, prd) -
                semquad_visco_plastic_eta(T, y, e - h, p, prd)) / (2h)
        fd_p = (semquad_visco_plastic_eta(T, y, e, p + h, prd) -
                semquad_visco_plastic_eta(T, y, e, p - h, prd)) / (2h)
        @test abs(semquad_dp_deta_T(T, y, e, p, prd) - fd_T) <=
              1.0e-5 * max(abs(fd_T), 1.0e-12)
        @test abs(semquad_deta_dp_eps(T, y, e, p, prd) - fd_e) <=
              1.0e-5 * max(abs(fd_e), 1.0e-12)
        @test abs(semquad_deta_dp(T, y, e, p, prd) - fd_p) <=
              1.0e-5 * max(abs(fd_p), 1.0e-12)
    end
    # the damper gives the yielded branch a T dependence of its own
    @test semquad_dp_deta_T(0.4, 0.3, 50.0, 3.0, pr0) == 0.0
    @test semquad_dp_deta_T(0.4, 0.3, 50.0, 3.0, prd) != 0.0
    # ---- (c) cold start of the DP Stokes Newton ----
    qt = Quadtree(2; cells_x=3, cells_y=3)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    th = [0.1 * cospi(x) * sinpi(y) for (x, y) in sq.coords]
    for (i, (x, y)) in enumerate(sq.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th[i] = 0.0)
    end
    s0 = semquad_dp_stokes_newton(sq, qt, ss, pr0, th; tol=1.0e-8, maxit=12)
    sd = semquad_dp_stokes_newton(sq, qt, ss, prd, th; tol=1.0e-8, maxit=12)
    v0 = semquad_vrms(sq, qt, s0.u[1:sq.N], s0.u[(sq.N + 1):(2sq.N)])
    vd = semquad_vrms(sq, qt, sd.u[1:sq.N], sd.u[(sq.N + 1):(2sq.N)])
    @printf("G170: cold start no damper: conv=%s it=%d resid=%.1e Vrms=%.1e\n",
            s0.converged, s0.iters, s0.resid, v0)
    @printf("G170: cold start damper:    conv=%s it=%d resid=%.1e Vrms=%.1e\n",
            sd.converged, sd.iters, sd.resid, vd)
    @test sd.converged
    @test sd.iters <= 10
    @test sd.resid < 1.0e-8
    @test 10.0 < vd < 1.0e4
    # without the damper the same cold start does not reach the tolerance
    @test !s0.converged
    @test s0.resid > 1.0e-3
end
