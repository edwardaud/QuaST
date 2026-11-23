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
    # The former section (c) drove the DP Stokes Newton from a cold start to
    # show the damper is what makes that solver converge.  The Newton was
    # removed with the steady-state solver, so the convergence claim has no
    # subject; the derivative checks above are the physics it depended on.
    # the damper gives the yielded branch a T dependence of its own
    @test semquad_dp_deta_T(0.4, 0.3, 50.0, 3.0, pr0) == 0.0
    @test semquad_dp_deta_T(0.4, 0.3, 50.0, 3.0, prd) != 0.0
end
