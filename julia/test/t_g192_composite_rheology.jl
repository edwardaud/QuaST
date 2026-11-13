# G192: composite diffusion + dislocation creep rheology (`eta_law=:composite`),
# **legacy `comp_rule = :isostrain` law**.
#
# Since the 2026-11-08 model-form switch (docs/dev/PHYSICS.md §14.8) the *default*
# combination rule is `:isostress` (ASPECT's law), and this gate pins the
# **legacy** `:isostrain` rule explicitly: it is kept for bitwise reproducibility
# of the pre-switch results and this file is their regression.  The
# `:isostrain` law (ASPECT's `diffusion dislocation` material model with the
# harmonic combination, ASPECT's eta = stress/(2 edot) factor 2 absorbed into
# the prefactors):
#
#     eta_eff  = 1 / (1/eta_diff + 1/eta_disl)
#     eta_diff = Ad^-1 exp((Ed + Vd (1-z)) / T_nd)
#     eta_disl = As^-1 exp((Es + Vs (1-z)) / (n T_nd)) eps_II^((1-n)/n)
#     T_nd     = max(T + T0, comp_Tcut)
#
# `z` is the height (0 bottom, 1 top; `1-z` is depth) and `eps_II` is the code's
# `sqrt(0.5 edot:edot)` strain-rate invariant.  This gate covers the formula
# (1), the harmonic-mean limits (2), the clamps and their continuity/kink
# behaviour (3), the temperature floor (4), the analytic derivatives against
# finite differences (5), the parameter guards (6), the cross-check against
# `:arrhenius` in the diffusion-only limit (7) and an end-to-end time-stepping
# smoke run on a small tree (8).  `t_g201` covers the default `:isostress` rule.
using Test, MantleSEM

@testset "G192 composite diffusion+dislocation rheology (isostrain)" begin
    Ad, Ed, Vd = 2.0e3, 9.0, 2.0
    As, Es, Vs = 5.0e3, 12.0, 1.0
    n, T0 = 3.0, 0.5
    mk(; kw...) = Params(; Ra=1.0e6, H=0.0, eta_law=:composite,
                         comp_Ad=Ad, comp_Ed=Ed, comp_Vd=Vd,
                         comp_As=As, comp_Es=Es, comp_Vs=Vs, comp_n=n,
                         eta_T0=T0, eta_min=1.0e-8, eta_max=1.0e8,
                         comp_rule=:isostrain, kw...)
    pr = mk()

    # closed forms used by the tests below
    eta_diff(T, z) = Ad^(-1.0) * exp((Ed + Vd * (1.0 - z)) / (T + T0))
    eta_disl(T, ed, z) = As^(-1.0) * exp((Es + Vs * (1.0 - z)) / (n * (T + T0))) *
                         ed^((1.0 - n) / n)

    # ---- (1) the formula: eta = 1/(1/eta_diff + 1/eta_disl) -----------------
    for (T, ed, z) in ((1.0, 1.0, 0.0), (0.5, 0.3, 0.7), (0.2, 2.0, 0.1),
                       (0.9, 1.0e-6, 0.5), (0.1, 1.0e3, 1.0))
        ed_d, ed_s = eta_diff(T, z), eta_disl(T, ed, z)
        want = 1.0 / (1.0 / ed_d + 1.0 / ed_s)
        got, clipped = eta_composite_at(T, ed, z, pr)
        @test got ≈ want rtol = 1.0e-13
        @test !clipped
        # harmonic mean is bounded by both branches
        @test got <= min(ed_d, ed_s) * (1.0 + 1.0e-13)
        # the inverse form is what the implementation carries
        inv_diff, inv_disl, Tnd, ed_used = composite_inv_eta(T, ed, z, pr)
        @test inv_diff ≈ 1.0 / ed_d rtol = 1.0e-13
        @test inv_disl ≈ 1.0 / ed_s rtol = 1.0e-13
        @test Tnd == T + T0
        @test ed_used == max(ed, pr.comp_eps_min)
    end

    # monotonicity: colder is stiffer, faster is weaker (n > 1)
    zc = 0.4
    Tvals = [eta_composite_at(T, 0.5, zc, pr)[1] for T in 0.0:0.05:1.0]
    @test all(diff(Tvals) .< 0.0)
    evals = [eta_composite_at(0.5, ed, zc, pr)[1] for ed in exp10.(range(-4, 3; length=30))]
    @test all(diff(evals) .< 0.0)
    # deeper is stiffer (V_d, V_s > 0)
    zvals = [eta_composite_at(0.5, 0.5, z, pr)[1] for z in 0.0:0.05:1.0]
    @test all(diff(zvals) .< 0.0)

    # ---- (2) limits: diffusion-only, dislocation-only ----------------------
    # A_s = 0 -> inv_disl = 0 -> exactly the diffusion branch
    pr_d = mk(comp_As=0.0)
    for (T, ed, z) in ((1.0, 1.0, 0.0), (0.3, 1.0e-3, 0.6), (0.0, 5.0, 1.0))
        @test eta_composite_at(T, ed, z, pr_d)[1] ≈ eta_diff(T, z) rtol = 1.0e-14
    end
    # A_d = 0 -> inv_diff = 0 -> exactly the dislocation branch
    pr_s = mk(comp_Ad=0.0)
    for (T, ed, z) in ((1.0, 1.0, 0.0), (0.3, 1.0e-3, 0.6), (0.0, 5.0, 1.0))
        @test eta_composite_at(T, ed, z, pr_s)[1] ≈ eta_disl(T, ed, z) rtol = 1.0e-12
    end
    # eps -> 0 is the diffusion limit (bounded viscosity, no Inf).  With the
    # default comp_eps_min = 1e-12 the residual dislocation term is
    # O(comp_eps_min^((n-1)/n)) ~ 1e-8, so use a tiny floor for the exact test.
    pr_lim = mk(comp_eps_min=1.0e-20)
    e0 = eta_composite_at(0.5, 0.0, 0.5, pr_lim)[1]
    @test e0 ≈ eta_diff(0.5, 0.5) rtol = 1.0e-9
    @test all(isfinite, eta_composite_at(0.5, 0.0, 0.5, pr_lim))

    # ---- (3) clamps: value continuous, derivative is the one-sided 0 (kink) -
    # eta at T=0.5, z=0.5 ranges over ~[0.13, 11] in edot, so cap it at 1.
    eta_max = 1.0
    pr_c = mk(eta_max=eta_max)
    # eta increases as edot decreases -> finds the eta_max crossing by bisection
    # on the *unclipped* value (the clipped one is identically eta_max below it)
    function f(ed)
        id, is, _, _ = composite_inv_eta(0.5, ed, 0.5, pr_c)
        return 1.0 / (id + is) - eta_max
    end
    lo, hi = 1.0e-12, 1.0e6          # f(lo) > 0 (clipped), f(hi) < 0 (free)
    @test f(lo) > 0.0 && f(hi) < 0.0
    for _ in 1:200
        mid = sqrt(lo * hi)
        if f(mid) > 0.0
            lo = mid
        else
            hi = mid
        end
    end
    ed_star = hi
    left = eta_composite_at(0.5, ed_star * (1.0 - 1.0e-6), 0.5, pr_c)
    right = eta_composite_at(0.5, ed_star * (1.0 + 1.0e-6), 0.5, pr_c)
    # no jump at the clamp: both sides are eta_max to round-off / O(1e-6) away
    @test left[1] == eta_max && left[2]
    @test abs(right[1] - eta_max) <= 1.0e-4 * eta_max
    # the unclipped value really is above eta_max there (so the clamp is active)
    @test f(ed_star * (1.0 + 1.0e-6)) < 0.0 && f(ed_star * (1.0 - 1.0e-6)) > 0.0
    @test eta_composite_at(0.5, ed_star * (1.0 - 1.0e-3), 0.5, mk())[1] > eta_max
    # the analytic derivative is the one-sided zero of the hard clamp; the
    # underlying law has a finite slope, i.e. the clamp is a *kink* (the first
    # derivative jumps) but not a jump in the value (documented in RHEOLOGY.md)
    @test deta_composite_at(0.5, ed_star * (1.0 - 1.0e-3), 0.5, pr_c) == (0.0, 0.0)
    d_free = deta_composite_at(0.5, ed_star * (1.0 + 1.0e-3), 0.5, pr_c)[2]
    @test d_free < 0.0
    # and in a neighbourhood that straddles the crossing the value has no jump:
    # |eta(ed) - eta_max| ~ |deta/dln ed| * 1e-3 at the crossing
    for r in (1.0 - 1.0e-3, 1.0 + 1.0e-3)
        @test abs(eta_composite_at(0.5, ed_star * r, 0.5, pr_c)[1] - eta_max) <=
              1.0e-3 * eta_max
    end
    # bottom clamp: eta_min, reached by raising the strain rate
    eta_min = 1.0e-3
    pr_m = mk(eta_min=eta_min)
    emin_hit = eta_composite_at(1.0, 1.0e12, 0.0, pr_m)
    @test emin_hit[1] == eta_min && emin_hit[2]
    @test deta_composite_at(1.0, 1.0e12, 0.0, pr_m) == (0.0, 0.0)
    # inside the clamps the derivative is nonzero (the kink is only at the bound)
    @test deta_composite_at(1.0, 1.0e-2, 0.0, pr_m)[2] < 0.0

    # ---- (4) T_cut: the exponent is floored, d eta/dT vanishes below it -----
    Tcut = 0.6
    pr_t = mk(comp_Tcut=Tcut)
    for ed in (1.0e-3, 1.0, 1.0e3)
        @test eta_composite_at(0.0, ed, 0.5, pr_t)[1] ==
              eta_composite_at(Tcut - T0, ed, 0.5, pr_t)[1]
        @test deta_composite_at(0.0, ed, 0.5, pr_t)[1] == 0.0
    end
    # the regularisation caps the cold-end viscosity (Vd, Vs > 0)
    Tlo = Tcut - T0 - 0.1
    @test eta_composite_at(Tlo, 1.0, 0.5, pr_t)[1] <=
          eta_composite_at(Tlo, 1.0, 0.5, mk())[1]
    # above the floor the derivative is alive again
    @test deta_composite_at(Tcut - T0 + 0.05, 1.0, 0.5, pr_t)[1] < 0.0

    # ---- (5) analytic derivatives vs central finite differences -------------
    pr_f = mk(eta_min=-Inf, eta_max=Inf)
    function fdcheck(T, ed, z; hT=1.0e-6, he=1.0e-7)
        dT, de = deta_composite_at(T, ed, z, pr_f)
        fdT = (eta_composite_at(T + hT, ed, z, pr_f)[1] -
               eta_composite_at(T - hT, ed, z, pr_f)[1]) / (2hT)
        fde = (eta_composite_at(T, ed * (1 + he), z, pr_f)[1] -
               eta_composite_at(T, ed * (1 - he), z, pr_f)[1]) / (2he * ed)
        return dT, fdT, de, fde
    end
    # sample points: hot diffusion-dominated, mixed, cold dislocation-dominated,
    # and the crossover strain rate where inv_diff == inv_disl
    Tcross = 0.5
    dzc = 1.0 - 0.5
    ed_cross = ((Ad / As) * exp(-(Ed + Vd * dzc) / (Tcross + T0) +
                                (Es + Vs * dzc) / (n * (Tcross + T0))))^(n / (n - 1))
    invd_c, invs_c, _, _ = composite_inv_eta(Tcross, ed_cross, 0.5, pr_f)
    @test invd_c ≈ invs_c rtol = 1.0e-10      # really the crossover
    pts = ((1.0, 1.0, 0.0), (0.6, 1.0e-2, 0.3), (0.3, 1.0e3, 0.8),
           (Tcross, ed_cross, 0.5), (0.85, 1.0e-4, 0.2))
    for (T, ed, z) in pts
        dT, fdT, de, fde = fdcheck(T, ed, z)
        @test dT < 0.0 && de < 0.0
        @test abs(dT - fdT) <= 1.0e-6 * abs(fdT)
        @test abs(de - fde) <= 1.0e-6 * abs(fde)
    end
    # the FD check must not be trivially zero-vs-zero: at every point the two
    # derivatives are a non-negligible fraction of the viscosity itself
    for (T, ed, z) in pts
        e = eta_composite_at(T, ed, z, pr_f)[1]
        dT, de = deta_composite_at(T, ed, z, pr_f)
        @test abs(dT) > 1.0e-6 * e
        @test abs(de) > 1.0e-12 * e / ed
    end
    # strain-rate floor: derivative is switched off below comp_eps_min
    pr_em = mk(comp_eps_min=1.0e-3, eta_min=-Inf, eta_max=Inf)
    @test deta_composite_at(0.5, 1.0e-4, 0.5, pr_em)[2] == 0.0
    @test deta_composite_at(0.5, 1.0e-2, 0.5, pr_em)[2] < 0.0

    # ---- (6) guards --------------------------------------------------------
    @test_throws ArgumentError Params(eta_law=:composite)                 # no prefactor
    @test_throws ArgumentError Params(eta_law=:composite, comp_Ad=-1.0, comp_As=1.0)
    @test_throws ArgumentError Params(eta_law=:composite, comp_Ad=1.0, comp_As=0.0,
                                      comp_n=0.0)
    @test_throws ArgumentError Params(eta_law=:composite, comp_Ad=1.0, comp_As=0.0,
                                      comp_Tcut=-1.0)
    @test_throws ArgumentError Params(eta_law=:composite, comp_Ad=1.0, comp_As=0.0,
                                      eta_T0=0.0)
    # the plastic branch is not part of this law
    @test_throws ArgumentError Params(eta_law=:composite, comp_Ad=1.0, comp_As=1.0,
                                      yield_c=0.1, eta_min=1.0e-6)
    # the strain-rate-independent entry points reject :composite
    @test_throws ArgumentError eta_at(0.5, 0.5, pr)
    @test_throws ArgumentError deta_dT_at(0.5, 0.5, pr)

    # ---- (7) diffusion-only limit == :arrhenius, end to end -----------------
    # eta_diff with Ad = exp(Ed/(1+T0)) and Vd = 0 is exactly
    # exp(Ed*(1/(T+T0) - 1/(1+T0))), i.e. the :arrhenius law normalised to
    # eta(T=1) = 1.
    Ed2 = 9.210340371976184                      # ln 1e4: a 1e4 contrast
    pr_a = Params(Ra=1.0e6, H=0.0, eta_law=:arrhenius, eta_E=Ed2, eta_T0=T0)
    pr_cd = Params(Ra=1.0e6, H=0.0, eta_law=:composite,
                   comp_Ad=exp(Ed2 / (1.0 + T0)), comp_Ed=Ed2, comp_Vd=0.0,
                   comp_As=0.0, comp_Es=0.0, comp_Vs=0.0, comp_n=3.0,
                   eta_T0=T0, comp_rule=:isostrain)
    for T in 0.0:0.1:1.0, z in (0.0, 0.5, 1.0)
        @test eta_composite_at(T, 1.0, z, pr_cd)[1] ≈ eta_at(T, z, pr_a)[1] rtol=1e-13
    end
    qt7 = Quadtree(5)
    for _ in 1:3, lf in collect(qt7.leaves); refine!(qt7, lf); end
    balance!(qt7)
    ss7 = make_semstokes(qt7, 2)
    th7 = [0.1 * sinpi(x) * sinpi(z) for (x, z) in ss7.su.coords]
    dt7 = 1.0e-5
    run7(p) = run_semquad_simulation(qt7, p, th7;
        p=2, tstop=2 * dt7, dt0=dt7, dtmax=dt7, dtmin=1.0e-16,
        amr_every=0, ev=0.0, advection=:skew, linear=:direct,
        maxpic=1, pic_tol=1.0e-6, verbose=false)
    ra = run7(pr_a)
    rc = run7(pr_cd)
    @test maximum(abs, ra.theta - rc.theta) <= 1.0e-10
    @test ra.history[end][3] ≈ rc.history[end][3] rtol = 1.0e-10

    # ---- (8) time-stepping smoke run (no steady Newton) ---------------------
    # Strong composite case: a 1e4 Arrhenius contrast in the diffusion branch
    # plus an active dislocation branch (n = 3), n = 3, on a 64-leaf tree.
    # `comp_rule = :isostrain` like the rest of this gate; the default
    # `:isostress` driver run is in `t_g201` section (7).
    pr_run = Params(Ra=1.0e6, H=0.0, eta_law=:composite,
                    comp_Ad=exp(Ed2 / (1.0 + T0)), comp_Ed=Ed2, comp_Vd=0.0,
                    comp_As=1.0e-2 * exp(12.0 / (n * (1.0 + T0))), comp_Es=12.0,
                    comp_Vs=0.0, comp_n=n, eta_T0=T0,
                    eta_min=1.0e-6, eta_max=1.0e4, comp_rule=:isostrain)
    qt = Quadtree(5)
    for _ in 1:3, lf in collect(qt.leaves); refine!(qt, lf); end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    theta0 = [0.05 * sinpi(x) * sinpi(z) for (x, z) in ss.su.coords]
    dts = 1.0e-5
    runs = map((1, 3)) do maxpic
        run_semquad_simulation(qt, pr_run, theta0;
            p=2, tstop=3 * dts, dt0=dts, dtmax=dts, dtmin=1.0e-16,
            amr_every=0, ev=0.0, advection=:skew, linear=:direct,
            maxpic=maxpic, pic_tol=1.0e-6, pic_relax=1.0, verbose=false)
    end
    for r in runs
        @test all(isfinite, r.theta)
        @test all(isfinite, vcat(r.history...))
        @test r.steps == 3
        # the law really produced a viscosity contrast in the run
        leaves = Set(r.qt.leaves)
        e_hot = eta_composite_at(Tbar(0.001, pr_run.H), 1.0, 0.001, pr_run)[1]
        e_cold = eta_composite_at(Tbar(0.999, pr_run.H), 1.0, 0.999, pr_run)[1]
        @test e_cold / e_hot > 1.0e3
        @test length(leaves) == length(r.qt.leaves)
    end
    # iterating inside the step changes the state (the Picard loop is active)
    @test maximum(abs, runs[1].theta - runs[2].theta) > 0.0
end
