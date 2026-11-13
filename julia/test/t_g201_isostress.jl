# G201: the **default** `comp_rule = :isostress` composite law.
#
# Since 2026-11-08 (docs/RHEOLOGY.md §14.8) `eta_law = :composite` solves
# ASPECT's isostress partition at every quadrature point:
#
#     edot = tau/(2 eta_diff) + A_s exp(-(E_s+V_s(1-z))/(R T)) tau^n   ->  tau
#     eta  = tau/(2 edot)                        (then one eta_min/eta_max clip)
#
# and the legacy total-rate harmonic mean is still available as
# `comp_rule = :isostrain` (pinned by `t_g192`).  This gate pins:
#
#   (1) the two controlled ASPECT 2.5.0-pre runs of RHEOLOGY §14.5
#       (2.957782e20 Pa s at r = 0.3, 7.053197e20 Pa s at the balanced r = 1)
#       and the ratios to the legacy isostrain values;
#   (2) the scalar solve itself: residual, partition consistency, the
#       iteration cap/tolerance, the closed-form limits and the strain-rate
#       floor;
#   (3) the analytic tangents of §14.4b against central differences;
#   (4) the eta_min/eta_max clamp: continuous value, one-sided zero derivative;
#   (5) a 495-point card against an independent re-implementation of ASPECT's
#       algorithm (bisection on the stress), with the legacy offset for scale;
#   (6) the legacy `:isostrain` law reproduces its old closed form (this is the
#       reproducibility contract for the pre-switch results, in addition to the
#       bitwise driver hashes of out_julia/agent/g201_bitwise.jl);
#   (7) a default-rule time-stepping smoke run that differs from the legacy one.
using Test, MantleSEM

@testset "G201 isostress composite law (default)" begin
    Ad, Ed, Vd = 2.0e3, 9.0, 2.0
    As, Es, Vs = 5.0e3, 12.0, 1.0
    n, T0 = 3.0, 0.5
    mk(rule=:isostress; kw...) =
        Params(; Ra=1.0e6, H=0.0, eta_law=:composite,
               comp_Ad=Ad, comp_Ed=Ed, comp_Vd=Vd,
               comp_As=As, comp_Es=Es, comp_Vs=Vs, comp_n=n,
               eta_T0=T0, eta_min=-Inf, eta_max=Inf, comp_rule=rule, kw...)
    pr = mk()
    pr_i = mk(:isostrain)

    # ---- (0) the default, the guard, and the field's doc semantics ----------
    @test pr.comp_rule === :isostress
    @test Params(eta_law=:composite, comp_Ad=1.0, comp_As=0.0,
                 eta_T0=0.5).comp_rule === :isostress
    @test Params(eta_law=:composite, comp_Ad=1.0, comp_As=0.0, eta_T0=0.5,
                 comp_rule=:isostrain).comp_rule === :isostrain
    @test_throws ArgumentError Params(eta_law=:composite, comp_Ad=1.0,
                                      comp_As=0.0, eta_T0=0.5,
                                      comp_rule=:bogus)

    # ---- (1) the controlled ASPECT runs of §14.5 ---------------------------
    # The mapping and the state construction are those of t_g200.
    R_GAS = 8.3144621
    eta0 = 1.0e21
    dT = 1000.0
    Ttop = 273.0
    P0 = 3300.0 * 9.81 * 2.9e6
    S = 1.0e-6 / (2.9e6)^2
    dg = 1.0e-3
    function stage2(A_d, A_s, n2; rule=:isostress)
        Params(; eta_law=:composite,
               comp_Ad=2.0 * eta0 * A_d / dg^3, comp_Ed=375e3 / (R_GAS * dT),
               comp_Vd=0.0,
               comp_As=2.0 * eta0 * A_s^(1.0 / n2) * S^((n2 - 1.0) / n2),
               comp_Es=530e3 / (R_GAS * dT), comp_Vs=0.0, comp_n=n2,
               comp_eps_min=1e-30 / S, eta_T0=Ttop / dT,
               eta_min=1e17 / eta0, eta_max=1e28 / eta0, comp_rule=rule)
    end
    T1500 = 1500.0 / dT - Ttop / dT
    edot = 1e-6 / S
    # r = 0.3: eta_d = 3.0e20 Pa s, eta_disl = 1.0e21 Pa s
    pr03 = stage2(1.906578e-17, 7.985007e-42, 3.5)
    pr1 = stage2(5.719734e-18, 7.985007e-42, 3.5)
    e03, cl03 = eta_composite_at(T1500, edot, 0.0, pr03)
    e1, cl1 = eta_composite_at(T1500, edot, 0.0, pr1)
    measured03 = 2.957782e20            # ASPECT 2.5.0-pre, mean viscosity field
    measured1 = 7.053197e20
    @test e03 * eta0 ≈ measured03 rtol = 1.0e-5
    @test e1 * eta0 ≈ measured1 rtol = 1.0e-5
    @test !cl03 && !cl1
    # the legacy law is the measured offset: 1.2817057x and 1.4106394x
    leg03 = eta_composite_at(T1500, edot, 0.0,
                             stage2(1.906578e-17, 7.985007e-42, 3.5;
                                    rule=:isostrain))[1]
    leg1 = eta_composite_at(T1500, edot, 0.0,
                            stage2(5.719734e-18, 7.985007e-42, 3.5;
                                   rule=:isostrain))[1]
    @test e03 / leg03 ≈ 1.2817057 rtol = 1.0e-5
    @test e1 / leg1 ≈ 1.4106394 rtol = 1.0e-6
    # the two legacy values §14.5 records (the 7-digit A_d/A_s of the run
    # files give them to 1e-7 relative, hence the tolerance)
    @test leg03 * eta0 ≈ 2.3076923076923075e20 rtol = 1.0e-6
    @test leg1 * eta0 ≈ 5.0e20 rtol = 1.0e-6

    # ---- (2) the scalar solve ---------------------------------------------
    # residual and partition consistency away from the closed-form limits
    for (T, ed, z) in ((1.0, 1.0, 0.0), (0.6, 1.0e-2, 0.3), (0.3, 1.0e3, 0.8),
                       (0.85, 1.0e-4, 0.2), (0.2, 1.0e-6, 0.9))
        st = composite_isostress_state(T, ed, z, pr)
        @test st.converged
        @test 1 <= st.iters <= COMP_ISO_MAXIT
        k = st.inv_diff / st.inv_disl
        @test abs(k * st.x + st.x^n - 1.0) <= COMP_ISO_TOL
        ed_used = max(ed, pr.comp_eps_min)
        @test st.ed == ed_used
        @test st.ed_d + st.ed_s ≈ ed_used rtol = 1.0e-12
        # eta = tau/(2 ed) is what the partition implies: ed_d = inv_diff eta ed
        @test st.eta ≈ st.ed_d / (st.inv_diff * ed_used) rtol = 1.0e-12
        @test st.eta ≈ st.x / st.inv_disl rtol = 1.0e-15
        @test st.x > 0.0 && st.x <= 1.0
        # the value returned through the public entry point is the same
        # (both clamps are off here)
        @test eta_composite_at(T, ed, z, pr)[1] == st.eta
    end
    # the tolerance is really 1e-15 and the cap is not silently smaller
    @test COMP_ISO_TOL == 1.0e-15
    @test COMP_ISO_MAXIT >= 50
    # iteration count over the whole G201 card: bounded and always converged
    cards = (0.0:0.1:1.0, exp10.(range(-4, 4; length=17)), (0.0, 0.5, 1.0))
    maxit = 0
    nconv = 0
    ntot = 0
    for T in cards[1], ed in cards[2], z in cards[3]
        st = composite_isostress_state(T, ed, z, pr)
        maxit = max(maxit, st.iters)
        nconv += st.converged ? 1 : 0
        ntot += 1
    end
    @test nconv == ntot
    @test maxit <= 10

    # one-mechanism limits are closed form, iteration-free, and bitwise equal
    # to the legacy law (the two rules coincide there -- §14.4b)
    for (T, ed, z) in ((1.0, 1.0, 0.0), (0.3, 1.0e-3, 0.6), (0.0, 5.0, 1.0),
                       (0.6, 1.0e-2, 0.3))
        for kw in ((comp_As=0.0,), (comp_Ad=0.0,), (comp_n=1.0,))
            prls = mk(:isostress; kw...)
            prli = mk(:isostrain; kw...)
            stl = composite_isostress_state(T, ed, z, prls)
            @test stl.iters == 0
            @test stl.converged
            @test eta_composite_at(T, ed, z, prls)[1] ==
                  eta_composite_at(T, ed, z, prli)[1]
        end
    end
    # the n = 1 rule identity is exact in both value and derivative
    prn1s = mk(comp_n=1.0)
    prn1i = mk(:isostrain, comp_n=1.0)
    for (T, ed, z) in ((0.5, 0.3, 0.7), (0.9, 1e-3, 0.2), (0.1, 2.0, 0.9))
        @test eta_composite_at(T, ed, z, prn1s)[1] ==
              eta_composite_at(T, ed, z, prn1i)[1]
        @test deta_composite_at(T, ed, z, prn1s)[1] ≈
              deta_composite_at(T, ed, z, prn1i)[1] rtol = 1.0e-13
        @test deta_composite_at(T, ed, z, prn1s)[2] == 0.0
    end

    # strain-rate floor: below comp_eps_min the viscosity is frozen
    prfloor = mk(comp_eps_min=1.0e-2)
    e_floor = eta_composite_at(0.5, 1.0e-4, 0.5, prfloor)[1]
    @test eta_composite_at(0.5, 1.0e-5, 0.5, prfloor)[1] == e_floor
    @test eta_composite_at(0.5, 1.0e-3, 0.5, prfloor)[1] == e_floor
    @test deta_composite_at(0.5, 1.0e-4, 0.5, prfloor)[2] == 0.0
    @test deta_composite_at(0.5, 1.1e-2, 0.5, prfloor)[2] < 0.0

    # ---- (3) analytic tangents vs central differences ----------------------
    # §14.4b: deta_dT = -(ed_d Q_d + ed_s Q_s)/(2 edot R T^2 F_tau),
    #         deta_dedot = -(n-1) ed_s eta/(edot (ed_d + n ed_s))
    function fdcheck(T, ed, z; hT=1.0e-6, he=1.0e-7)
        # NOTE: a closure assigns to a captured variable, so every local here
        # gets a name that does not exist in the enclosing testset scope.
        dTv, dev = deta_composite_at(T, ed, z, pr)
        fdT = (eta_composite_at(T + hT, ed, z, pr)[1] -
               eta_composite_at(T - hT, ed, z, pr)[1]) / (2hT)
        fde = (eta_composite_at(T, ed * (1 + he), z, pr)[1] -
               eta_composite_at(T, ed * (1 - he), z, pr)[1]) / (2he * ed)
        return dTv, fdT, dev, fde
    end
    ed_cross = ((Ad / As) * exp(-(Ed + Vd * 0.5) / (0.5 + T0) +
                                (Es + Vs * 0.5) / (n * (0.5 + T0))))^(n / (n - 1))
    pts = ((1.0, 1.0, 0.0), (0.6, 1.0e-2, 0.3), (0.3, 1.0e3, 0.8),
           (0.5, ed_cross, 0.5), (0.85, 1.0e-4, 0.2))
    for (T, ed, z) in pts, hT in (1.0e-5, 1.0e-6)
        dTa, fdT, dea, fde = fdcheck(T, ed, z; hT=hT)
        @test dTa < 0.0 && dea < 0.0
        @test abs(dTa - fdT) <= 1.0e-6 * abs(fdT)
        @test abs(dea - fde) <= 1.0e-6 * abs(fde)
        # not a zero-vs-zero comparison
        @test abs(dTa) > 1.0e-6 * eta_composite_at(T, ed, z, pr)[1]
    end
    # the isostress tangent is *not* the legacy one where both mechanisms
    # contribute (otherwise the test above would not distinguish the rules)
    dTs, des = deta_composite_at(0.5, ed_cross, 0.5, pr)
    dTi, dei = deta_composite_at(0.5, ed_cross, 0.5, pr_i)
    @test abs(des - dei) > 1.0e-2 * abs(dei)
    @test abs(dTs - dTi) > 1.0e-2 * abs(dTi)
    # the T_cut guard and the clamp convention (one-sided zero)
    prcut = mk(comp_Tcut=0.6)
    @test deta_composite_at(0.0, 1.0, 0.5, prcut)[1] == 0.0
    @test deta_composite_at(0.6 - T0 + 0.05, 1.0, 0.5, prcut)[1] < 0.0
    @test eta_composite_at(0.0, 1.0, 0.5, prcut)[1] ==
          eta_composite_at(0.6 - T0, 1.0, 0.5, prcut)[1]

    # ---- (4) the clamp: continuous value, one-sided zero derivative --------
    eta_max = 1.0
    pr_c = mk(eta_max=eta_max)
    raw_eta(ed) = composite_isostress_state(0.5, ed, 0.5, pr_c).eta
    lo, hi = 1.0e-12, 1.0e6
    @test raw_eta(lo) > eta_max && raw_eta(hi) < eta_max
    for _ in 1:200
        mid = sqrt(lo * hi)
        raw_eta(mid) > eta_max ? (lo = mid) : (hi = mid)
    end
    ed_star = hi
    left = eta_composite_at(0.5, ed_star * (1 - 1.0e-6), 0.5, pr_c)
    right = eta_composite_at(0.5, ed_star * (1 + 1.0e-6), 0.5, pr_c)
    @test left[1] == eta_max && left[2]
    @test abs(right[1] - eta_max) <= 1.0e-4 * eta_max
    @test deta_composite_at(0.5, ed_star * (1 - 1.0e-3), 0.5, pr_c) == (0.0, 0.0)
    @test deta_composite_at(0.5, ed_star * (1 + 1.0e-3), 0.5, pr_c)[2] < 0.0
    # no jump across the crossing: the value is continuous at 1e-3 relative
    for rr in (1 - 1.0e-3, 1 + 1.0e-3)
        @test abs(eta_composite_at(0.5, ed_star * rr, 0.5, pr_c)[1] - eta_max) <=
              1.0e-3 * eta_max
    end
    # and the isostress crossing is *not* the isostrain crossing (a different
    # law has a different eta_max boundary): 1.397061e-3 vs 1.270208e-3
    pr_ci = mk(:isostrain, eta_max=eta_max)
    f_i(ed) = 1.0 / (composite_inv_eta(0.5, ed, 0.5, pr_ci)[1] +
                     composite_inv_eta(0.5, ed, 0.5, pr_ci)[2]) - eta_max
    lo, hi = 1.0e-12, 1.0e6
    for _ in 1:200
        mid = sqrt(lo * hi)
        f_i(mid) > 0.0 ? (lo = mid) : (hi = mid)
    end
    @test ed_star ≈ 1.397061e-3 rtol = 1.0e-5
    @test hi ≈ 1.270208e-3 rtol = 1.0e-5
    # bottom clamp
    pr_m = mk(eta_min=1.0e-3)
    @test eta_composite_at(1.0, 1.0e12, 0.0, pr_m) == (1.0e-3, true)
    @test deta_composite_at(1.0, 1.0e12, 0.0, pr_m) == (0.0, 0.0)

    # ---- (5) the 495-point card vs ASPECT's algorithm ----------------------
    # independent re-implementation (bisection on the stress), from t_g200
    function branches(T_K, P, ed, s)
        edd = max(ed, s.minrate)
        eta_d = 0.5 / s.A_d * exp((s.E_d + P * s.V_d) / (R_GAS * T_K)) * dg^3
        eta_s = 0.5 * s.A_s^(-1.0 / s.n) *
                exp((s.E_s + P * s.V_s) / (s.n * R_GAS * T_K)) *
                edd^((1.0 - s.n) / s.n)
        return eta_d, eta_s
    end
    function eta_parallel(T_K, P, ed, s)
        edd = max(ed, s.minrate)
        eta_d, _ = branches(T_K, P, edd, s)
        Bs = s.A_s * exp(-(s.E_s + P * s.V_s) / (R_GAS * T_K))
        lo2, hi2 = 0.0, 2.0 * eta_d * edd
        for _ in 1:200
            midd = 0.5 * (lo2 + hi2)
            midd / (2.0 * eta_d) + Bs * midd^s.n > edd ? (hi2 = midd) :
                                                         (lo2 = midd)
        end
        return min(max(0.5 * (lo2 + hi2) / (2.0 * edd), s.vmin), s.vmax)
    end
    function composite_params(s; rule)
        Params(; eta_law=:composite,
               comp_Ad=2.0 * eta0 * s.A_d / dg^3, comp_Ed=s.E_d / (R_GAS * dT),
               comp_Vd=s.V_d * P0 / (R_GAS * dT),
               comp_As=2.0 * eta0 * s.A_s^(1.0 / s.n) * S^((s.n - 1.0) / s.n),
               comp_Es=s.E_s / (R_GAS * dT),
               comp_Vs=s.V_s * P0 / (R_GAS * dT), comp_n=s.n,
               comp_eps_min=s.minrate / S, eta_T0=Ttop / dT,
               eta_min=s.vmin / eta0, eta_max=s.vmax / eta0, comp_rule=rule)
    end
    TREF = 0.5 + Ttop / dT
    AD_MIX = exp((9.0 + 1.0) / TREF) / (exp(12.0 / (3.0 * TREF)) / 5.0e3)
    mkaset(Ad2) = (A_d=Ad2 * dg^3 / (2.0 * eta0),
                   E_d=9.0 * R_GAS * dT, V_d=2.0 * R_GAS * dT / P0,
                   A_s=(5.0e3 / (2.0 * eta0))^3 * S^(-2.0),
                   E_s=12.0 * R_GAS * dT, V_s=1.0 * R_GAS * dT / P0,
                   n=3.0, minrate=1.0e-12 * S, vmin=1e17, vmax=1e28)
    aset = (A_d=1.5e-15, E_d=375e3, V_d=6e-6, A_s=1.1e-16, E_s=530e3,
            V_s=1.4e-5, n=3.5, minrate=1.4e-20, vmin=1e17, vmax=1e28)
    sets = [("D", mkaset(100.0 * AD_MIX)), ("M", mkaset(AD_MIX)),
            ("S", mkaset(AD_MIX / 100.0)), ("A", aset)]
    Ts = collect(0.0:0.1:1.0)
    Eds = [10.0^k for k in -4:4]
    Zs = [0.0, 0.25, 0.5, 0.75, 1.0]
    for (name, s) in sets
        pr_ss = composite_params(s; rule=:isostress)
        pr_si = composite_params(s; rule=:isostrain)
        mx = 0.0
        mxleg = 0.0
        nin = 0
        for T in Ts, ed in Eds, z in Zs
            ref = eta_parallel(dT * (T + pr_ss.eta_T0), P0 * (1.0 - z), ed * S, s)
            ours = eta_composite_at(T, ed, z, pr_ss)[1] * eta0
            leg = eta_composite_at(T, ed, z, pr_si)[1] * eta0
            if ref > 1.01 * s.vmin && ref < 0.99 * s.vmax &&
               ours > 1.01 * s.vmin && ours < 0.99 * s.vmax &&
               leg > 1.01 * s.vmin && leg < 0.99 * s.vmax
                mx = max(mx, abs(ours - ref) / ref)
                mxleg = max(mxleg, abs(leg - ref) / ref)
                nin += 1
            end
        end
        @test nin > 50
        @test mx <= 1.0e-12            # measured 6.6e-15 .. 8.9e-15
        @test mxleg > 1.0e-2           # the legacy offset is real (>= 28%)
    end

    # ---- (6) the legacy rule reproduces its closed form --------------------
    # This is the in-gate reproducibility contract of the pre-switch numbers;
    # the end-to-end bitwise hashes are in out_julia/agent/g201_bitwise.jl.
    for (T, ed, z) in ((1.0, 1.0, 0.0), (0.6, 1.0e-2, 0.3), (0.3, 1.0e3, 0.8),
                       (0.85, 1.0e-4, 0.2), (0.2, 1.0e-6, 0.9))
        id, isl, Tnd, ed_used = composite_inv_eta(T, ed, z, pr_i)
        @test Tnd == T + T0
        @test ed_used == max(ed, pr_i.comp_eps_min)
        @test eta_composite_at(T, ed, z, pr_i)[1] == 1.0 / (id + isl)
        # the analytic derivative of the harmonic mean, unchanged
        eta_i = 1.0 / (id + isl)
        want_T = -(eta_i * eta_i) * ((Ed + Vd * (1.0 - z)) * id +
                                     (Es + Vs * (1.0 - z)) / n * isl) / (Tnd * Tnd)
        @test deta_composite_at(T, ed, z, pr_i)[1] == want_T
    end
    # the one-mechanism variants of the legacy law are the old branches
    @test eta_composite_at(0.5, 1.0, 0.5, mk(:isostrain, comp_As=0.0))[1] ==
          1.0 / composite_inv_eta(0.5, 1.0, 0.5,
                                  mk(:isostrain, comp_As=0.0))[1]

    # ---- (7) default-rule driver smoke run --------------------------------
    qt = Quadtree(5)
    for _ in 1:3, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    th0 = [0.05 * sinpi(x) * sinpi(z) for (x, z) in ss.su.coords]
    function smoke(prv)
        run_semquad_simulation(qt, prv, copy(th0); p=2, tstop=3.0e-5,
                               dt0=1.0e-5, dtmax=1.0e-5, dtmin=1.0e-16,
                               amr_every=0, ev=0.0, advection=:skew,
                               linear=:direct, maxpic=1, pic_tol=1.0e-6,
                               verbose=false)
    end
    pr_run = Params(Ra=1.0e6, H=0.0, eta_law=:composite,
                    comp_Ad=exp(9.210340371976184 / (1.0 + T0)),
                    comp_Ed=9.210340371976184, comp_Vd=0.0,
                    comp_As=1.0e-2 * exp(12.0 / (n * (1.0 + T0))), comp_Es=12.0,
                    comp_Vs=0.0, comp_n=n, eta_T0=T0, eta_min=1.0e-6,
                    eta_max=1.0e4)                       # default :isostress
    r_new = smoke(pr_run)
    r_old = smoke(Params(; Ra=pr_run.Ra, H=0.0, eta_law=:composite,
                         comp_Ad=pr_run.comp_Ad, comp_Ed=pr_run.comp_Ed,
                         comp_As=pr_run.comp_As, comp_Es=pr_run.comp_Es,
                         comp_n=pr_run.comp_n, eta_T0=pr_run.eta_T0,
                         eta_min=pr_run.eta_min, eta_max=pr_run.eta_max,
                         comp_rule=:isostrain))
    @test all(isfinite, r_new.theta) && all(isfinite, vcat(r_new.history...))
    @test r_new.steps == 3
    @test isfinite(r_new.history[end][3]) && isfinite(r_new.history[end][5])
    # the model-form switch is visible in the time-stepping solution
    @test maximum(abs, r_new.theta - r_old.theta) > 0.0
end
