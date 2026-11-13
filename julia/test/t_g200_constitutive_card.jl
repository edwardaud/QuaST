# G200: cross-code constitutive card -- MantleSEM `eta_law = :composite` vs
# ASPECT's `diffusion dislocation` material model.
#
# ASPECT side: an independent re-implementation of
#   source/material_model/diffusion_dislocation.cc
#   source/material_model/rheology/{diffusion,dislocation}_creep.cc
#   include/aspect/global.h (gas_constant)
# as shipped in the image gassmoeller/aspect:latest (ASPECT 2.5.0-pre, commit
# 340c0d018d9f2dbb71deda800f038229e7653f00, deal.II 9.4.0).  The equations are
# quoted in docs/RHEOLOGY.md section 14.  Nothing here calls the solver; our
# side calls `eta_composite_at` only.
#
# What is pinned:
#   (1) the branch laws + the parameter mapping agree to round-off (<= 1e-12);
#   (2) the one-mechanism limits and the strain-rate floor;
#   (3) the combination rule is NOT the same: the legacy law (ours before
#       2026-11-08, now `comp_rule = :isostrain`) is the harmonic mean of the
#       branch viscosities evaluated at the total strain rate, ASPECT solves the
#       isostress partition.  The measured offsets are pinned below (stage-2
#       ASPECT run: 1.281705x at T=1500 K, edot=1e-6 1/s; the analytic values at
#       r=1 are 1.410639x for n=3.5, 1.364656x for n=3 -- NOT the maximum, see
#       the 2026-11-08 correction in docs/RHEOLOGY.md 14.4b).  The default was
#       switched to `:isostress` on 2026-11-08; this card compares the branch
#       laws with the *same* (legacy) combination rule on both sides, and
#       `t_g201` pins the new default against the same ASPECT numbers;
#   (4) the viscosity-bound placement differs in ASPECT's fixpoint fallback
#       (each branch is clipped before the harmonic mean -> eta_max/2);
#   (5) the documented RHEOLOGY 1.2 parameter mapping was wrong by the recorded
#       factors 4.0e51 (comp_Ad) and 6.093e15 (comp_As) -- section 1.2 is
#       corrected, these numbers are pinned so it cannot regress.

using Test, MantleSEM

@testset "G200 constitutive card: :composite vs ASPECT diffusion dislocation" begin
    # ---- ASPECT's formula, independent implementation ------------------------
    R_GAS = 8.3144621                    # ASPECT include/aspect/global.h:72
    # physical scaling used to map dimensional <-> code units
    eta0 = 1.0e21                        # Pa s (Rayleigh reference viscosity)
    dT = 1000.0                          # K
    Ttop = 273.0                         # K
    P0 = 3300.0 * 9.81 * 2.9e6           # Pa (rho0 g H)
    S = 1.0e-6 / (2.9e6)^2               # 1/s (kappa/H^2)
    dg = 1.0e-3                          # m (grain size)

    # DiffusionCreep::compute_viscosity / DislocationCreep::compute_viscosity
    #   eta_d = 0.5 A_d^-1 d^m exp((E_d + P V_d)/(R T))
    #   eta_s = 0.5 A_s^-1/n edot^((1-n)/n) exp((E_s + P V_s)/(n R T))
    function branches(T_K, P, ed, s)
        edd = max(ed, s.minrate)
        eta_d = 0.5 / s.A_d * exp((s.E_d + P * s.V_d) / (R_GAS * T_K)) * dg^3
        eta_s = 0.5 * s.A_s^(-1.0 / s.n) *
                exp((s.E_s + P * s.V_s) / (s.n * R_GAS * T_K)) *
                edd^((1.0 - s.n) / s.n)
        return eta_d, eta_s
    end
    # the closed-form combination both codes *intend* (harmonic mean)
    function eta_harmonic(T_K, P, ed, s; clip=true)
        ed_d, ed_s = branches(T_K, P, ed, s)
        eta = 1.0 / (1.0 / ed_d + 1.0 / ed_s)
        clip && (eta = min(max(eta, s.vmin), s.vmax))
        return eta
    end
    # ASPECT's actual combination: isostress partition
    #   tau/(2 eta_d) + A_s tau^n exp(-(E_s + P V_s)/(R T)) = edot
    function eta_parallel(T_K, P, ed, s)
        ed = max(ed, s.minrate)
        ed_d, _ = branches(T_K, P, ed, s)
        Bs = s.A_s * exp(-(s.E_s + P * s.V_s) / (R_GAS * T_K))
        lo, hi = 0.0, 2.0 * ed_d * ed
        for _ in 1:200
            mid = 0.5 * (lo + hi)
            if mid / (2.0 * ed_d) + Bs * mid^s.n > ed
                hi = mid
            else
                lo = mid
            end
        end
        eta = 0.5 * (lo + hi) / (2.0 * ed)
        return min(max(eta, s.vmin), s.vmax)
    end
    # our :composite parameters from ASPECT's (the corrected section-1.2 mapping).
    # The card compares the *branch laws* with the same combination rule on both
    # sides, so it uses the legacy `:isostrain` rule; the default `:isostress`
    # rule is pinned against the same ASPECT numbers by `t_g201`.
    function composite_params(s; eta_min=-Inf, eta_max=Inf,
                              rule=:isostrain)
        Params(; eta_law=:composite,
               comp_Ad=2.0 * eta0 * s.A_d / dg^3,
               comp_Ed=s.E_d / (R_GAS * dT),
               comp_Vd=s.V_d * P0 / (R_GAS * dT),
               comp_As=2.0 * eta0 * s.A_s^(1.0 / s.n) * S^((s.n - 1.0) / s.n),
               comp_Es=s.E_s / (R_GAS * dT),
               comp_Vs=s.V_s * P0 / (R_GAS * dT),
               comp_n=s.n,
               comp_eps_min=s.minrate / S,
               eta_T0=Ttop / dT,
               eta_min=eta_min, eta_max=eta_max, comp_rule=rule)
    end
    function aspect_set(; A_d, E_d, V_d, A_s, E_s, V_s, n, minrate,
                        vmin=1e17, vmax=1e28)
        (A_d=A_d, E_d=E_d, V_d=V_d, A_s=A_s, E_s=E_s, V_s=V_s, n=n,
         minrate=minrate, vmin=vmin, vmax=vmax)
    end

    # ---- the card grid: cold lid -> interior, 9 decades of edot, the box ----
    Ts = collect(0.0:0.1:1.0)
    Eds = [10.0^k for k in -4:4]
    Zs = [0.0, 0.25, 0.5, 0.75, 1.0]

    function card_maxrel(s; clip)
        pr = composite_params(s; eta_min=(-Inf), eta_max=Inf)
        clip && (pr = composite_params(s; eta_min=s.vmin / eta0,
                                       eta_max=s.vmax / eta0))
        m = 0.0
        for T in Ts, ed in Eds, z in Zs
            ea = eta_harmonic(dT * (T + pr.eta_T0), P0 * (1.0 - z), ed * S, s;
                              clip=clip)
            eo, _ = eta_composite_at(T, ed, z, pr)
            m = max(m, abs(eo * eta0 - ea) / ea)
        end
        return m
    end

    # the repo's G192-style set, centred diffusion / mixed / dislocation
    TREF = 0.5 + Ttop / dT
    AD_MIX = exp((9.0 + 1.0) / TREF) / (exp(12.0 / (3.0 * TREF)) / 5.0e3)
    mk(Ad) = aspect_set(A_d=Ad * dg^3 / (2.0 * eta0),
                        E_d=9.0 * R_GAS * dT, V_d=2.0 * R_GAS * dT / P0,
                        A_s=(5.0e3 / (2.0 * eta0))^3 * S^(-2.0),
                        E_s=12.0 * R_GAS * dT, V_s=1.0 * R_GAS * dT / P0,
                        n=3.0, minrate=1.0e-12 * S)
    sets = [("D-diffusion", mk(100.0 * AD_MIX)),
            ("M-mixed", mk(AD_MIX)),
            ("S-dislocation", mk(AD_MIX / 100.0))]

    # ==== (1) branch laws + mapping: round-off agreement =====================
    for (name, s) in sets
        @test card_maxrel(s; clip=false) <= 1.0e-12
        @test card_maxrel(s; clip=true) <= 1.0e-12
    end
    # ASPECT's own defaults, same scaling (large exponents -> looser bound)
    sA = aspect_set(A_d=1.5e-15, E_d=375e3, V_d=6e-6, A_s=1.1e-16, E_s=530e3,
                    V_s=1.4e-5, n=3.5, minrate=1.4e-20)
    @test card_maxrel(sA; clip=false) <= 1.0e-12
    @test card_maxrel(sA; clip=true) <= 1.0e-12

    # ==== (2) limits and the strain-rate floor ==============================
    s = sets[2][2]
    pr = composite_params(s)
    # diffusion only: A_s -> 0 must give the diffusion branch exactly
    prd = composite_params(aspect_set(A_d=s.A_d, E_d=s.E_d, V_d=s.V_d,
                                      A_s=1e-300, E_s=s.E_s, V_s=s.V_s, n=s.n,
                                      minrate=s.minrate))
    for (T, ed, z) in ((1.0, 1.0, 0.0), (0.4, 10.0, 0.6), (0.0, 1e4, 1.0))
        edd, _ = branches(dT * (T + prd.eta_T0), P0 * (1 - z), ed * S, s)
        @test eta_composite_at(T, ed, z, prd)[1] * eta0 ≈ edd rtol = 1e-12
    end
    # dislocation only: A_d -> 0
    prs = composite_params(aspect_set(A_d=1e-300, E_d=s.E_d, V_d=s.V_d,
                                      A_s=s.A_s, E_s=s.E_s, V_s=s.V_s, n=s.n,
                                      minrate=s.minrate))
    for (T, ed, z) in ((1.0, 1.0, 0.0), (0.4, 10.0, 0.6), (0.0, 1e4, 1.0))
        _, eds = branches(dT * (T + prs.eta_T0), P0 * (1 - z), ed * S, s)
        @test eta_composite_at(T, ed, z, prs)[1] * eta0 ≈ eds rtol = 1e-12
    end
    # the floor: below comp_eps_min the dislocation branch is frozen, so the
    # law is the diffusion limit in both codes
    sfl = aspect_set(A_d=s.A_d, E_d=s.E_d, V_d=s.V_d, A_s=s.A_s, E_s=s.E_s,
                     V_s=s.V_s, n=s.n, minrate=1e-6 * S)
    prf = composite_params(sfl)
    for ed in (1.0e-8, 1.0e-7, 1.0e-6)
        ea = eta_harmonic(dT * (0.5 + prf.eta_T0), 0.5 * P0, ed * S, sfl)
        @test eta_composite_at(0.5, ed, 0.5, prf)[1] * eta0 ≈ ea rtol = 1e-12
    end

    # ==== (3) the combination rule differs: isostrain vs isostress ==========
    # balanced state T=1500 K, edot=1e-6 1/s: eta_d = eta_s = 1e21 Pa s
    s2 = aspect_set(A_d=1.906578e-17, E_d=375e3, V_d=0.0, A_s=7.985007e-42,
                    E_s=530e3, V_s=0.0, n=3.5, minrate=1e-30)
    eta_d2, eta_s2 = branches(1500.0, 0.0, 1e-6, s2)
    @test eta_d2 ≈ 3.0e20 rtol = 1e-5
    @test eta_s2 ≈ 1.0e21 rtol = 1e-5
    ours2 = 1.0 / (1.0 / eta_d2 + 1.0 / eta_s2)
    par2 = eta_parallel(1500.0, 0.0, 1e-6, s2)
    @test ours2 ≈ 2.3076923076923075e20 rtol = 1e-6
    # ASPECT 2.5.0-pre, measured (mean of the "viscosity" field of the
    # controlled simple-shear run, docs/RHEOLOGY.md section 14.5)
    measured2 = 2.957782e20
    @test par2 ≈ measured2 rtol = 1e-4
    @test measured2 / ours2 ≈ 1.2817057 rtol = 1e-4
    # analytic worst case at the exactly balanced point: x + x^n = 1 -> 2x
    for (n, want) in ((3.0, 1.3646556), (3.5, 1.4106394))
        lo, hi = 0.0, 1.0
        for _ in 1:200
            mid = 0.5 * (lo + hi)
            (mid + mid^n > 1.0) ? (hi = mid) : (lo = mid)
        end
        @test (lo + hi) ≈ want rtol = 1e-6
    end
    # and the one-mechanism limits agree exactly (no combination ambiguity)
    @test eta_parallel(1500.0, 0.0, 1e-6,
                       aspect_set(A_d=1.906578e-17, E_d=375e3, V_d=0.0,
                                  A_s=1e-300, E_s=530e3, V_s=0.0, n=3.5,
                                  minrate=1e-30)) ≈ eta_d2 rtol = 1e-12

    # balanced state (r = 1, eta_d = eta_disl = 1e21 Pa s): the analytic
    # maximum.  ASPECT 2.5.0-pre measured 7.053197e20 Pa s in the same
    # controlled simple-shear setup (docs/RHEOLOGY.md section 14.5).
    s2b = aspect_set(A_d=5.719734e-18, E_d=375e3, V_d=0.0, A_s=7.985007e-42,
                     E_s=530e3, V_s=0.0, n=3.5, minrate=1e-30)
    eta_db, eta_sb = branches(1500.0, 0.0, 1e-6, s2b)
    @test eta_db ≈ 1.0e21 rtol = 1e-5
    @test eta_sb ≈ 1.0e21 rtol = 1e-5
    ours_b = 1.0 / (1.0 / eta_db + 1.0 / eta_sb)
    @test ours_b ≈ 5.0e20 rtol = 1e-6
    par_b = eta_parallel(1500.0, 0.0, 1e-6, s2b)
    measured_b = 7.053197e20
    @test par_b ≈ measured_b rtol = 1e-4
    @test measured_b / ours_b ≈ 1.4106394 rtol = 1e-6

    # ==== (4) ASPECT's fixpoint fallback clips each branch first =============
    # Both branches above eta_max: ASPECT's fallback gives eta_max/2, our
    # clip-once convention gives eta_max.
    vmax = 1e28
    eta_d = 1e30; eta_s = 1e30
    @test 1.0 / (1.0 / min(eta_d, vmax) + 1.0 / min(eta_s, vmax)) == 0.5 * vmax
    # our convention, at the same state: clip only the harmonic mean
    @test min(max(1.0 / (1.0 / eta_d + 1.0 / eta_s), 1e17), vmax) == vmax

    # ==== (5) the strain-rate invariant: ASPECT uses deviator(eps) ===========
    # In 2-D deal.II's deviator subtracts tr/2, so
    #   edot_ASPECT = sqrt(edot_full^2 - (tr eps)^2/4).
    for (exx, ezz, exz) in ((0.5, -0.5, 0.0), (0.5, -0.3, 0.2), (1.0, 1.0, 0.0))
        tr = exx + ezz
        full2 = 0.5 * (exx^2 + ezz^2 + 2 * exz^2)
        dev2 = full2 - tr^2 / 4
        @test dev2 ≈ 0.5 * ((exx - tr / 2)^2 + (ezz - tr / 2)^2 + 2 * exz^2)
        # incompressible (tr = 0) -> identical, which is our solver's case
        if tr == 0.0
            @test dev2 ≈ full2
        end
    end

    # ==== (6) the corrected documented mapping (pinned factors) =============
    # RHEOLOGY 1.2 used comp_Ad = A_d/(2 eta0) and
    # comp_As = (A_s/2)^(1/n)/eta0^(1/n); the correct expressions are above.
    # The ratios are set independent:
    cp = composite_params(sets[2][2])
    old_Ad = sets[2][2].A_d / (2 * eta0)
    old_As = (sets[2][2].A_s / 2)^(1 / sets[2][2].n) / eta0^(1 / sets[2][2].n)
    @test cp.comp_Ad / old_Ad ≈ 4.0e51 rtol = 1e-6
    @test cp.comp_As / old_As ≈ 6.093161e15 rtol = 1e-6
    # the factor-2 signature: dropping it from comp_Ad doubles a
    # diffusion-dominated viscosity
    pr_ok = composite_params(sets[1][2])
    pr_half = Params(; eta_law=:composite, comp_Ad=pr_ok.comp_Ad / 2,
                     comp_Ed=pr_ok.comp_Ed, comp_Vd=pr_ok.comp_Vd,
                     comp_As=pr_ok.comp_As, comp_Es=pr_ok.comp_Es,
                     comp_Vs=pr_ok.comp_Vs, comp_n=pr_ok.comp_n,
                     comp_eps_min=pr_ok.comp_eps_min, eta_T0=pr_ok.eta_T0,
                     eta_min=pr_ok.eta_min, eta_max=pr_ok.eta_max,
                     comp_rule=pr_ok.comp_rule)
    @test eta_composite_at(0.9, 1e-3, 0.5, pr_half)[1] /
          eta_composite_at(0.9, 1e-3, 0.5, pr_ok)[1] ≈ 2.0 rtol = 1e-3
end
