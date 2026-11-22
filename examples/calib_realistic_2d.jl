#!/usr/bin/env julia
# Calibration of the composite (diffusion + dislocation) prefactors and — when
# the target contrast needs it — of the activation-energy scale, for the
# "realistic 2-D" statistical case (`case_realistic_2d.jl`,
# docs/dev/CASE_REALISTIC.md §3).  This script is the *derivation*; the case
# script carries the resulting numbers as literals, so the production run does
# not depend on it.
#
#   julia --project=. examples/calib_realistic_2d.jl
#
# What it does
# ------------
# 1. Measures the reference strain rate `edot_ref` from a constant-viscosity
#    precursor (eta = 1, Ra = 1e5, H = 10, same box/tree/IC as the case).  The
#    composite law needs a strain rate at which the contrast condition is
#    defined; using the RMS strain rate of the developed flow makes the
#    calibration representative of the run instead of an arbitrary O(1) number.
# 2. The **existing calibration**, printed unchanged so it stays auditable: at
#    the card's energies (E_d = 12, E_s = 17) solve the two conditions
#       (i)  eta_eff(T=0.5, z=0.5, edot_ref) = 1
#       (ii) eta_eff(T=0.1)/eta_eff(T=0.9) = 1e4        (the *old* window)
#    for the common scale and the branch ratio r = eta_diff/eta_disl at the
#    calibration point.  The ratio condition gives r = 0.129720 and the
#    prefactors the production card carries.
# 3. The **extended calibration** — the point of this script.  The card's own
#    field does not live in T in [0.1, 0.9]: its base state is
#        Tbar(z,H) = (1-z) + (H/2) z (1-z),  H = 10   (src/physics/params.jl)
#    so the cold end is the top wall T_top = 0 and the hot end is the base
#    state's maximum
#        T_bot = max_z Tbar(z,H) = (H+2)^2/(8H) = 1.8  at z* = (H-2)/(2H).
#    That maximum is derived here, not assumed, and printed together with the
#    bottom-wall value (1.0) and the mid-depth value (1.75) so the reader can
#    see which number is used and why.  The contrast condition therefore becomes
#        eta(T_top, z=0.5, edot_ref) / eta(T_bot, z=0.5, edot_ref) = TARGET
#    with TARGET = `CALIB_TARGET_CONTRAST` (default 1e4).  The realised contrast
#    of the current card over the full range is reported next to the mantle's
#    1e3 to 1e5 and a one-line verdict is printed.
# 4. The **activation energy is the unknown when the target is not reachable
#    with the card's energies.**  The two prefactors are still fixed by the
#    existing conditions: the value condition at (T=0.5, z=0.5) gives the common
#    scale, and the branch ratio r = 0.129720 that the existing [0.1, 0.9]
#    ratio condition fixed is `held`.  The choice is measured, not aesthetic:
#    re-solving the ratio over the wider range instead (with the card's
#    energies) gives r = 39.8, which hits the same thermal contrast by throwing
#    the law into the dislocation limit; on a developed field of the case
#    (`case_ev_0.11` step 7208, 9994 leaves) the realised contrast then rises to
#    1.6e6 (clamps on) instead of falling, because the cold lid's low strain
#    rate amplification becomes enormous.  Holding r keeps the mixture that the
#    project's physics note measured as effectively diffusion-dominated
#    (docs/dev/PHYSICS.md §2.3).  With the prefactors fixed,
#    the full-range contrast is a one-parameter function C(E_d) of the
#    activation energy, so:
#        E_s = (E_s0/E_d0) E_d        (the ASPECT 530/375 kJ/mol ratio is kept,
#                                      so only the overall energy scale moves)
#        solve  C(E_d) = TARGET       by bisection in E_d,
#        re-deriving (A_d, A_s) from the value condition and r at every
#        candidate E_d.  The two steps -- prefactors, activation energy -- are
#        iterated to convergence (the outer loop is the bisection; the observed
#        residual is <= 1e-12 relative).  When the target already holds at the
#        card's energies (e.g. because the range was set back to [0.1, 0.9]
#        with `CALIB_TTOP`/`CALIB_TBOT`), the solve keeps the card's numbers
#        exactly: the default target then reproduces the existing calibration.
# 5. Declares the **reference viscosity** the prefactors produce.  The value
#    condition is eta(T=0.5, z=0.5, edot_ref) = 1, so in the units the material
#    law returns the reference viscosity of this calibration is eta_ref = 1.
#    That is exactly what `Params.eta_ref` expects (the default, and the card's
#    CASE_ETA_REF default): Ra = rho g alpha dT d^3/(kappa eta_ref).  If the
#    prefactors are rescaled, eta_ref must be rescaled with them -- multiplying
#    comp_Ad and comp_As by 1/c multiplies every viscosity the law returns by c,
#    so the same physical problem needs eta_ref = c; with eta_ref left at 1 the
#    flow changes while Ra does not.
# 6. Verifies everything against `eta_composite_at` itself (not against the
#    hand-rolled algebra) and prints the variant's full parameter set and the
#    environment command line that runs the card with it:
#        CASE_COMP_ED=... CASE_COMP_ES=... CASE_COMP_AD=... CASE_COMP_AS=...
#
# No solver code is touched; this only calls exported functions.

using QuaST, Printf, LinearAlgebra
# A runtime format string (the long messages below are concatenated literals so
# they stay readable).  `@printf` requires a literal format at macro-expansion
# time, so the literal-only messages use `@printf` and the concatenated ones go
# through this helper.
printf(fmt::AbstractString, args...) = print(Printf.format(Printf.Format(fmt), args...))

# ------------------------------------------------------------ model card ----
const NEXP  = 3.0      # comp_n, stress exponent (mantle: 3-3.5)
const T0    = 0.5      # eta_T0, nondimensional absolute surface temperature
const ED0   = 12.0     # comp_Ed = E_d/(R dT), the production card's value
const ES0   = 17.0     # comp_Es = E_s/(R dT)  (= 1.42 E_d, the ASPECT ratio)
const ED_RATIO = ES0 / ED0   # E_s/E_d kept when the energy scale is solved
const ZC    = 0.5      # calibration height
const TC    = 0.5      # calibration temperature
const EDOT_MIN = 1.0e-12   # comp_eps_min

# The *old* contrast window, kept only as the condition that fixes the branch
# ratio (docs/dev/PHYSICS.md §2.3: "the ratio fixes the branch mixture").
const TLOW_REF  = 0.1
const THIGH_REF = 0.9
const RATIO_REF = 1.0e4

# ------------------------------------------------ the card's real range ----
# `CALIB_TARGET_CONTRAST` is the requested contrast over the full range.  The
# default 1e4 is the existing condition's value; with the default range the
# calibration therefore changes the card's energies (that is the point), while
# with the range set back to the old window it reproduces the card exactly.
const TARGET = parse(Float64, get(ENV, "CALIB_TARGET_CONTRAST", "1.0e4"))
const HHEAT  = 10.0    # internal-heating number of the case (theta equation)
# The mantle's viscosity-contrast range used throughout this project:
# 1e3 to 1e5 (docs/dev/HANDOVER.md §6 and §10.2 item 2, docs/dev/PHASE1.md
# §1.2 — the "7.8e6 against the mantle's 1e3-1e5" account).  The verdict below
# compares against it.
const MANTLE_LO = 1.0e3
const MANTLE_HI = 1.0e5

"""
    base_state_max(H)

Maximum over the layer of the card's conductive base state
`Tbar(z,H) = (1-z) + (H/2) z (1-z)`, i.e. the hottest temperature the card's
own base state reaches.  `dTbar/dz = -1 + (H/2)(1-2z)` vanishes at
`z* = (H-2)/(2H)` and `Tbar(z*) = (H+2)^2/(8H)`; for `H <= 2` the maximum is the
bottom wall value 1 and `z* <= 0` is outside the layer.
"""
function base_state_max(H::Float64)
    H <= 2.0 && return 1.0
    return (H + 2.0)^2 / (8.0 * H)
end

# `CALIB_TTOP`/`CALIB_TBOT` let a reader re-run the calibration over another
# range (in particular the old [0.1, 0.9], which must reproduce the card).
const TTOP = parse(Float64, get(ENV, "CALIB_TTOP", "0.0"))
const TBOT = begin
    raw = strip(get(ENV, "CALIB_TBOT", ""))
    t = isempty(raw) ? base_state_max(HHEAT) : parse(Float64, raw)
    (isfinite(t) && t > TTOP) ||
        error("CALIB_TBOT must be finite and above CALIB_TTOP=$(TTOP), got $(t)")
    t
end
# The derived (default) hot end, printed in the derivation block.
const TBOT_DERIVED = base_state_max(HHEAT)

# ------------------------------------------------------------- case geometry
const LX     = 2.5
const CX     = 5
const CY     = 2
const PMAX   = 7        # max_level (leaf size 1/256 at the finest level)
const PORD   = 2        # polynomial order
# The calibration is Ra-dependent: the reference strain rate it measures and the
# prefactors it solves for both follow the run's Ra.  `CALIB_RA` selects the case.
const RA     = parse(Float64, get(ENV, "CALIB_RA", "1.0e5"))

# Unit-branch viscosities at (T, z = ZC, edot = edot_ref), normalised to 1 at
# (TC, ZC): eta_d = exp(ED*(1/T_nd - 1/T_ndc)), eta_s likewise with ES/NEXP.
Tnd(T) = T + T0
const TNDC = TC + T0
eta_d_unit(T, ED) = exp(ED * (1.0 / Tnd(T) - 1.0 / TNDC))
eta_s_unit(T, ES) = exp((ES / NEXP) * (1.0 / Tnd(T) - 1.0 / TNDC))

"""
    eta_iso_unit(T, r, ED, ES)

Isostress effective viscosity of two branches with `eta_d = eta_d_unit(T,ED)` and
`eta_s = eta_s_unit(T,ES)/r`, using the code's scalar equation `k x + x^n = 1`
with `k = inv_diff/inv_disl` and `eta = x/inv_disl = x*eta_s`
(docs/dev/PHYSICS.md §14.8.1).  `r` is the diffusion-to-dislocation branch
ratio at the calibration point.
"""
function eta_iso_unit(T::Float64, r::Float64, ED::Float64, ES::Float64)
    ed = eta_d_unit(T, ED)
    es = eta_s_unit(T, ES) / r
    k = es / ed                       # = inv_diff/inv_disl
    # bracket [1/(1+k), 1] as in `composite_isostress_state`, bisection is
    # enough here (the value is re-verified with eta_composite_at below).
    lo = 1.0 / (1.0 + k); hi = 1.0
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        g = k * mid + mid^NEXP - 1.0
        g < 0.0 ? (lo = mid) : (hi = mid)
    end
    x = 0.5 * (lo + hi)
    return x * es
end

"""
    solve_branch_ratio(ED, ES, TA, TB, target)

Branch ratio `r = eta_diff/eta_disl` at the calibration point that makes
`eta(TA)/eta(TB) = target` (unit prefactors; a common scale cancels in the
ratio).  The contrast is *decreasing* in r (r -> 0 leaves pure diffusion,
`exp(E_d (1/T_nd(TA) - 1/T_nd(TB)))`; r -> Inf the pure-dislocation limit), so
the root is unique.  Returns `(r, window_lower, window_upper)`; `r === nothing`
when the target is outside the achievable window `[window_lower, window_upper]`.
"""
function solve_branch_ratio(ED::Float64, ES::Float64, TA::Float64, TB::Float64,
                            target::Float64)
    s = 1.0 / Tnd(TA) - 1.0 / Tnd(TB)
    window_lo = exp((ES / NEXP) * s)      # pure dislocation (r -> Inf)
    window_hi = exp(ED * s)               # pure diffusion   (r -> 0)
    (window_lo <= target <= window_hi) ||
        return (nothing, window_lo, window_hi)
    lo = log(1.0e-10); hi = log(1.0e10)
    for _ in 1:300
        mid = 0.5 * (lo + hi)
        f = eta_iso_unit(TA, exp(mid), ED, ES) /
            eta_iso_unit(TB, exp(mid), ED, ES)
        f > target ? (lo = mid) : (hi = mid)
    end
    return (exp(0.5 * (lo + hi)), window_lo, window_hi)
end

"""
    prefactors(r, ED, ES, edot_ref)

The two prefactors `(comp_Ad, comp_As)` that put the isostress viscosity at the
calibration point to 1 and the branch ratio at the calibration point to `r`.
The common scale is closed form: at `(TC, ZC)` the unit-branch viscosity is
`scale = eta_iso_unit(TC, r, ED, ES)`, and `comp_Ad = exp(E_d) scale` because
`eta_diff(TC) = 1/(comp_Ad exp(-E_d/T_ndc)) = scale`; with `x r = scale` the
dislocation prefactor follows as `comp_As = exp(E_s/n) edot_ref^(-q) x`,
`q = (n-1)/n`.
"""
function prefactors(r::Float64, ED::Float64, ES::Float64, edot_ref::Float64)
    scale = eta_iso_unit(TC, r, ED, ES)   # = x/r = eta_disl(TC) with unit prefactors
    xrt = scale * r                       # x of k x + x^n = 1 at the cal. point
    comp_Ad = exp(ED) * scale
    comp_As = exp(ES / NEXP) * edot_ref^(-(NEXP - 1.0) / NEXP) * xrt
    return comp_Ad, comp_As
end

"""
    card_params(ED, ES, AD, AS; eta_min=-Inf, eta_max=Inf)

The `Params` of the production card with the composite law at the given
parameters (all other card rows are the ones the calibration is derived for).
"""
function card_params(ED::Float64, ES::Float64, AD::Float64, AS::Float64;
                     eta_min::Float64=-Inf, eta_max::Float64=Inf)
    return Params(Ra=RA, H=HHEAT, eta_law=:composite, comp_rule=:isostress,
                  comp_n=NEXP, comp_Ed=ED, comp_Es=ES, comp_Vd=0.0,
                  comp_Vs=0.0, comp_Ad=AD, comp_As=AS, comp_eps_min=EDOT_MIN,
                  comp_Tcut=0.0, eta_T0=T0, eta_min=eta_min, eta_max=eta_max,
                  bc=:free)
end

"""
    realised_contrast(ED, ES, AD, AS, edot_ref; ta=TTOP, tb=TBOT)

The contrast the law actually realises over the card's temperature range at the
calibration height and rate: `eta(ta, z=0.5, edot_ref)/eta(tb, z=0.5, edot_ref)`,
through the public closure.  Bounds are off unless asked for, so the number is
the law's, not the clamp's.
"""
function realised_contrast(ED::Float64, ES::Float64, AD::Float64, AS::Float64,
                           edot_ref::Float64; ta::Float64=TTOP, tb::Float64=TBOT,
                           eta_min::Float64=-Inf, eta_max::Float64=Inf)
    pr = card_params(ED, ES, AD, AS; eta_min=eta_min, eta_max=eta_max)
    et = eta_composite_at(ta, edot_ref, ZC, pr)[1]
    eb = eta_composite_at(tb, edot_ref, ZC, pr)[1]
    return et / eb
end

"""
    solve_activation_energy(r, target, edot_ref)

Solve `C(E_d) = target` for the activation-energy scale, with `E_s = ED_RATIO E_d`
and the prefactors re-derived from the value condition and the held branch ratio
at every candidate (the two calibration steps, iterated as the outer bisection).
`C` is increasing in `E_d`; the root is bracketed over `E_d in [1e-2, 60]` and
bisected to `1e-13` relative.  Returns `(E_d, E_s, Ad, As, niter)`.
"""
function solve_activation_energy(r::Float64, target::Float64,
                                 edot_ref::Float64)
    (isfinite(target) && target > 1.0) ||
        error("CALIB_TARGET_CONTRAST must be finite and > 1, got $(target)")
    C(ED) = begin
        ES = ED_RATIO * ED
        AD, AS = prefactors(r, ED, ES, edot_ref)
        realised_contrast(ED, ES, AD, AS, edot_ref)
    end
    elo, ehi = 1.0e-2, 60.0
    clo, chi = C(elo), C(ehi)
    (clo <= target <= chi) ||
        error("target $(target) is outside the range reachable by scaling the " *
              "activation energy: C(E_d->0)=$(clo), C(E_d=$(ehi))=$(chi)")
    lo, hi = elo, ehi
    niter = 0
    while hi - lo > 1.0e-13 * hi
        niter += 1
        mid = 0.5 * (lo + hi)
        if C(mid) < target
            lo = mid
        else
            hi = mid
        end
    end
    ED = 0.5 * (lo + hi)
    ES = ED_RATIO * ED
    AD, AS = prefactors(r, ED, ES, edot_ref)
    return ED, ES, AD, AS, niter
end

# ---------------------------------------------------------------- precursor
function precursor_edot()
    qt = Quadtree(PMAX; Lx=LX, cells_x=CX, cells_y=CY)
    for _ in 1:3, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, PORD)
    sq = ss.su
    theta0 = [0.05 * sinpi(y) * cospi(2.0 * x / LX) for (x, y) in sq.coords]
    # `:const` is not accepted by the driver; `:expT` with deta_T = 1 IS the
    # constant law eta = exp(-log(1)*T) = 1 (bit-identical to eta_const = 1).
    pr = Params(Ra=RA, H=HHEAT, eta_law=:expT, deta_T=1.0, bc=:free)
    res = run_semquad_simulation(qt, pr, theta0; p=PORD, tstop=1.0e-2,
                                 dt0=1.0e-6, dtmax=2.0e-5, dtmin=1.0e-14,
                                 cfl=0.5, maxsteps=120, maxpic=1,
                                 advection=:skew, ev=0.11, ev_cmax=0.052,
                                 linear=:direct, stokes_tol=1.0e-6,
                                 max_leaves=length(qt.leaves), max_level=PMAX,
                                 verbose=false)
    ed = semquad_strain_rate_field(res.ss.su, res.ux, res.uz)
    # interior nodes only (the wall nodes carry the boundary-layer derivative,
    # which is not representative of the volume)
    interior = [abs(y - 0.5) < 0.5 - 1.0e-12 for (x, y) in res.ss.su.coords]
    edi = ed[interior]
    rms = sqrt(sum(edi .^ 2) / length(edi))
    hist = res.history
    last = hist[end]
    printf("precursor: steps=%d t=%.3e Vrms=%.4e max|u|=%.4e Nu_top=%.5f\n",
            res.steps, res.t, last[5], last[11], last[3])
    printf("precursor: edot_rms(interior)=%.6e  edot_median=%.6e  max=%.6e\n",
            rms, sort(edi)[cld(length(edi), 2)], maximum(edi))
    return rms
end

# The mantle-range verdict, one line (the requirement of this pass).
verdict(C::Float64) = MANTLE_LO <= C <= MANTLE_HI ?
    "WITHIN the mantle range" : "OUTSIDE the mantle range"

function main()
    printf("=== composite prefactor + activation-energy calibration ")
    printf("(docs/dev/CASE_REALISTIC.md §3) ===\n")
    printf("fixed: n=%.1f eta_T0=%.2f Vd=Vs=0 comp_rule=:isostress\n",
            NEXP, T0)
    printf("card energies: comp_Ed=%.2f comp_Es=%.2f (E_s/E_d=%.4f, " *
            "the ASPECT 530/375 kJ/mol pair)\n", ED0, ES0, ED_RATIO)
    # `CALIB_EDOT` overrides the precursor.  It exists because at high Ra the
    # *run* visits strain rates orders of magnitude above the constant-viscosity
    # precursor's, so calibrating at the precursor rate puts the law's knee far
    # below where the flow actually lives and the interior sits on `eta_min`
    # (docs/dev/PHASE1.md, the Ra = 1e8 account).
    edot_env = strip(get(ENV, "CALIB_EDOT", ""))
    edot_ref = isempty(edot_env) ? precursor_edot() : parse(Float64, edot_env)
    # round to 3 significant digits so the card is reproducible by hand
    edot_ref = round(edot_ref; sigdigits=3)
    printf("edot_ref (rounded) = %.6e\n", edot_ref)

    # ------------------------------------------------ the card's real range
    # Derived, printed, and used by the contrast condition below.  The number
    # the old window's value meant: 0.1..0.9 sat in the lid, while the base
    # state (and the field built on it) reaches 1.8 at its hottest.
    zstar = HHEAT > 2.0 ? (HHEAT - 2.0) / (2.0 * HHEAT) : 0.0
    printf("\n--- the card's own temperature range (base state, H=%.1f) ---\n",
            HHEAT)
    printf("  Tbar(z,H) = (1-z) + (H/2) z (1-z)   (src/physics/params.jl)\n")
    printf("  bottom wall z=0    Tbar = %.4f\n", Tbar(0.0, HHEAT))
    printf("  mid-depth  z=0.5   Tbar = %.4f   (the interior temperature of " *
            "docs/dev/PHASE1.md §1.2)\n", Tbar(0.5, HHEAT))
    printf("  maximum    z*=%.3f  Tbar = %.4f   <- derived hot end T_bot\n",
            zstar, TBOT_DERIVED)
    printf("  field census of a developed run: T in [0, 1.647] " *
            "(docs/dev/PHYSICS.md §2.3)\n")
    printf("  calibration range: T_top=%.4f -> T_bot=%.4f  (T_bot %s the " *
            "derived value)\n", TTOP, TBOT,
            TBOT == TBOT_DERIVED ? "is" : "overrides")
    printf("  target contrast over that range: %.6e\n", TARGET)

    # ---------------------------------- step 1: the card's own prefactors
    # The two prefactors come from the existing two conditions, exactly as
    # before: the value condition and the [0.1,0.9] ratio condition at the
    # card's energies.  That ratio condition is what fixes the branch ratio r;
    # r is held by the extended step below (see the header).
    r, rlo, rhi = solve_branch_ratio(ED0, ES0, TLOW_REF, THIGH_REF, RATIO_REF)
    r === nothing &&
        error("the existing [0.1,0.9] ratio condition is not reachable at the " *
              "card's energies; window [$(rlo), $(rhi)]")
    AD0, AS0 = prefactors(r, ED0, ES0, edot_ref)
    printf("\n=== existing calibration (card energies, old window) ===\n")
    printf("branch ratio r = eta_d/eta_s at (T=%.2f,z=%.2f,edot_ref) = %.6f\n",
            TC, ZC, r)
    printf("comp_Ad = %.10e\ncomp_As = %.10e  (edot_ref = %.6e)\n",
            AD0, AS0, edot_ref)
    e05 = eta_composite_at(TC, edot_ref, ZC, card_params(ED0, ES0, AD0, AS0))[1]
    c_oldwin = realised_contrast(ED0, ES0, AD0, AS0, edot_ref;
                                 ta=TLOW_REF, tb=THIGH_REF)
    printf("VERIFY eta(0.5,z=0.5,edot_ref) = %.15e  (target 1)\n", e05)
    printf("VERIFY eta(0.1)/eta(0.9)        = %.15e  (target %.0e)\n",
            c_oldwin, RATIO_REF)
    printf("VERIFY rel. deviations: %.3e / %.3e\n", abs(e05 - 1.0),
            abs(c_oldwin - RATIO_REF) / RATIO_REF)

    # ------------------------------------ the reference viscosity it declares
    # The value condition above is the *definition* of the law's reference
    # viscosity: the calibration forces eta = 1 at (T=0.5, z=0.5, edot_ref), and
    # `Params.eta_ref` is exactly that number in the law's units.  Say it in the
    # output, because it is what a card must carry when the law is rescaled.
    printf("\n=== the reference viscosity this calibration declares ===\n")
    printf("value condition: eta(T=%.2f, z=%.2f, edot_ref) = %.15e  (target 1)\n",
            TC, ZC, e05)
    printf("=> in the units the material law returns, the reference viscosity of\n")
    printf("   these prefactors is eta_ref = 1.0.  That is the value Params.eta_ref\n")
    printf("   expects (the library default) and the production card's CASE_ETA_REF\n")
    printf("   default.  Ra is defined as rho g alpha dT d^3/(kappa eta_ref), so a\n")
    printf("   card that rescales the law must rescale eta_ref with it:\n")
    printf("     multiplying comp_Ad and comp_As by 1/c multiplies every viscosity\n")
    printf("     the law returns by c (both creep branches and the isostress\n")
    printf("     solution scale together), so the same physical problem needs\n")
    printf("     eta_ref = c.\n")
    printf("   Leaving eta_ref = 1 with the rescaled law keeps Ra at its old value\n")
    printf("   while every physical viscosity -- and the interior Rayleigh number\n")
    printf("   Ra/eta_interior -- moves by c.  That is the foot-gun the declaration\n")
    printf("   closes.  The eta_min/eta_max bounds are absolute bounds in the law's\n")
    printf("   units (applied before the normalisation), so a pure rescaling must\n")
    printf("   rescale them too.\n")

    # ------------------------- step 2: the diagnostic, on the real range
    c_card = realised_contrast(ED0, ES0, AD0, AS0, edot_ref)   # full real range
    printf("\n=== realised contrast over the full range [%.3f, %.3f] ===\n",
            TTOP, TBOT)
    printf("  at z=%.2f, edot_ref=%.4e, clamps off:\n", ZC, edot_ref)
    printf("  current card : %.6e   (mantle range 1e3-1e5: " *
            "docs/dev/HANDOVER.md §10.2 item 2, docs/dev/PHASE1.md §1.2)\n",
            c_card)
    printf("  target       : %.6e\n", TARGET)
    printf("  VERDICT current card: %.6e %s\n", c_card, verdict(c_card))

    # ------------------- step 3: the activation-energy solve for the target
    if abs(c_card / TARGET - 1.0) <= 1.0e-9
        # The target is already met at the card's energies: keep the card's
        # numbers exactly.  This is the "unchanged behaviour" branch -- setting
        # the range back to [0.1, 0.9] lands here.
        EDv, ESv, ADv, ASv = ED0, ES0, AD0, AS0
        printf("\n=== variant: target already met at the card's energies; " *
                "energies unchanged ===\n")
    else
        EDv, ESv, ADv, ASv, niter = solve_activation_energy(r, TARGET, edot_ref)
        printf("\n=== variant: activation energy solved for the target ===\n")
        printf("E_d = %.9f  E_s = %.9f   (E_s/E_d = %.6f; %d bisection " *
                "iterations)\n", EDv, ESv, ESv / EDv, niter)
    end
    # verify the variant against the public closure, not the algebra
    pr = card_params(EDv, ESv, ADv, ASv)
    ev05 = eta_composite_at(TC, edot_ref, ZC, pr)[1]
    c_real = realised_contrast(EDv, ESv, ADv, ASv, edot_ref)
    c_old = realised_contrast(EDv, ESv, ADv, ASv, edot_ref;
                              ta=TLOW_REF, tb=THIGH_REF)
    c_cl = realised_contrast(EDv, ESv, ADv, ASv, edot_ref;
                             eta_min=1.0e-4, eta_max=1.0e4)
    printf("VERIFY eta(0.5,z=0.5,edot_ref) = %.15e  (target 1)\n", ev05)
    printf("VERIFY eta(T_top)/eta(T_bot)    = %.15e  (target %.6e)\n",
            c_real, TARGET)
    printf("VERIFY rel. deviations: %.3e / %.3e\n", abs(ev05 - 1.0),
            abs(c_real - TARGET) / TARGET)
    if EDv == ED0 && ESv == ES0
        printf("  (the range is the old [%.2f,%.2f] window; the held branch " *
                "ratio reproduces the card's calibration exactly)\n", TTOP, TBOT)
    else
        printf("  (the held branch ratio keeps the [0.1,0.9] contrast at %.4e, " *
                "not %.0e: the full range is the calibration's identity now)\n",
                c_old, RATIO_REF)
    end
    printf("  contrast with the card clamps [1e-4,1e4]: %.6e (equal to the " *
            "unclamped value when no bound is active)\n", c_cl)
    printf("  VERDICT variant: %.6e   %s\n", c_real, verdict(c_real))
    printf("\nvariant parameter set:\n")
    printf("  comp_Ed = %.9f\n  comp_Es = %.9f\n  comp_Ad = %.10e\n" *
            "  comp_As = %.10e\n  comp_n = %.1f  eta_T0 = %.2f  " *
            "comp_eps_min = 1.0e-12  comp_rule = :isostress\n",
            EDv, ESv, ADv, ASv, NEXP, T0)
    printf("  edot_ref = %.6e  z = %.2f  range = [%.3f, %.3f]  target = %.6e\n",
            edot_ref, ZC, TTOP, TBOT, TARGET)
    # Both parameter sets satisfy the same value condition (eta = 1 at the
    # calibration point), so both declare the same reference viscosity.
    printf("  eta_ref = 1.0 for the card and for the variant: the value condition\n")
    printf("  eta(%.2f, z=%.2f, edot_ref) = 1 sets the law's own reference, so no\n",
            TC, ZC)
    printf("  CASE_ETA_REF is needed for either parameter set.\n")
    printf("\nrun the card with the variant (environment, defaults unchanged):\n")
    printf("  CASE_COMP_ED=%.9f CASE_COMP_ES=%.9f CASE_COMP_AD=%.10e " *
            "CASE_COMP_AS=%.10e julia --project=. examples/case_realistic_2d.jl " *
            "smoke\n", EDv, ESv, ADv, ASv)
    printf("or (same numbers, the additive card knobs):\n")
    printf("  CASE_COMP_ED=%.9f CASE_COMP_ES=%.9f CASE_COMP_AD=%.10e " *
            "CASE_COMP_AS=%.10e\n", EDv, ESv, ADv, ASv)

    # -------------------------------------------------- the eta table
    printf("\n eta(T,z=0.5) at the calibration rate (variant, clamps off):\n")
    printf("   T     eta(edot_ref)  eta(edot/100)  eta(10*edot)   eta(1000*edot)\n")
    for T in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 1.75, 1.8, 2.0, 2.5)
        e(x, ed) = eta_composite_at(x, ed, ZC, pr)[1]
        printf("  %4.2f  %12.5e  %13.5e  %13.5e  %14.5e\n", T, e(T, edot_ref),
                e(T, edot_ref / 100), e(T, 10 * edot_ref),
                e(T, 1000 * edot_ref))
    end
    printf("\n same table for the current card (the production literals):\n")
    pr0 = card_params(ED0, ES0, AD0, AS0)
    for T in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 1.75, 1.8, 2.0, 2.5)
        e(x, ed) = eta_composite_at(x, ed, ZC, pr0)[1]
        printf("  %4.2f  %12.5e  %13.5e  %13.5e  %14.5e\n", T, e(T, edot_ref),
                e(T, edot_ref / 100), e(T, 10 * edot_ref),
                e(T, 1000 * edot_ref))
    end
    # ------------------------------------------------------------ clamp census
    for (lo, hi) in ((1.0e-4, 1.0e6), (1.0e-3, 1.0e5))
        prb = card_params(ED0, ES0, AD0, AS0; eta_min=lo, eta_max=hi)
        ncl = 0; ntot = 0
        for Tv in range(0.0, 2.6; length=261)
            _, cl = eta_composite_at(Tv, edot_ref, ZC, prb)
            ntot += 1; ncl += cl ? 1 : 0
        end
        printf("current-card clamp census z=0.5, edot_ref, eta in [%.0e,%.0e]: " *
                "%d/%d of T in [0,2.6] clipped\n", lo, hi, ncl, ntot)
    end
    printf("\ncard (unchanged literals): comp_Ad=%.10e comp_As=%.10e " *
            "comp_Ed=%.1f comp_Es=%.1f comp_n=%.1f\n", AD0, AS0, ED0, ES0, NEXP)
    printf("      eta_T0=%.2f comp_eps_min=1e-12 comp_rule=:isostress  " *
            "edot_ref=%.4e\n", T0, edot_ref)
    printf("variant (calibrated for the real range, target %.4e): " *
            "comp_Ad=%.10e comp_As=%.10e comp_Ed=%.6f comp_Es=%.6f\n",
            TARGET, ADv, ASv, EDv, ESv)

    # ---------------------------------------------------- physical magnitudes
    # The prefactors only mean something together with the viscosity they imply.
    # `eta0` is what Ra fixes (eta0 = rho g alpha dT d^3 / (Ra kappa)): with
    # eta_ref = 1 (the value condition above) eta0 is the *physical reference
    # viscosity* Ra is defined with, so eta(0.5) is eta0 by construction and the
    # number a paper would quote is the *interior* one: Ra_interior = Ra/eta_int.
    # Print all of them, and the clip census over the strain-rate range the run
    # actually visits, because a law that sits on a bound has no structure left
    # whatever its prefactors.
    scale = (rho=3300.0, g=9.81, alpha=3.0e-5, dT=1000.0, d=2.9e6, kappa=1.0e-6)
    eta0 = scale.rho * scale.g * scale.alpha * scale.dT * scale.d^3 /
           (RA * scale.kappa)
    pr_ref = card_params(ED0, ES0, AD0, AS0; eta_min=1.0e-4, eta_max=1.0e4)
    eta_int_nd, _ = eta_composite_at(1.75, edot_ref, 0.5, pr_ref)
    printf("\nphysical magnitudes (d=%.2e m, kappa=%.1e m^2/s, Ra=%.1e):\n",
            scale.d, scale.kappa, RA)
    printf("  eta_ref = 1.0 (the value condition above), so the physical\n")
    printf("  reference viscosity Ra is defined with is eta0 = rho g alpha dT d^3/\n")
    printf("  (Ra kappa) = %.3e Pa s   [mantle 1e21]\n", eta0)
    printf("  eta at the interior temperature T=1.75 = %.3e Pa s   " *
            "[mantle 1e19-1e20]\n", eta0 * eta_int_nd)
    printf("  Ra based on the interior viscosity = Ra/eta_int = %.3e\n",
            RA / eta_int_nd)
    printf("  eta at the stiff end (clip)            = %.3e Pa s   " *
            "[lithosphere 1e23-1e25]\n", eta0 * 1.0e4)
    printf("  contrast lid/interior                  = %.1e            " *
            "[mantle 1e3-1e5]\n", 1.0e4 / eta_int_nd)
    # split the census by which bound is hit: a stiff lid on eta_max is the
    # intended physics, a soft interior on eta_min means the law is saturated
    for f in (1.0, 1.0e2, 1.0e3, 1.0e4, 1.0e5)
        nmax = 0; nmin = 0; ntot = 0
        for i in 1:61
            T = 2.6 * (i - 1) / 60
            e, _ = eta_composite_at(T, f * edot_ref, 0.5, pr_ref)
            ntot += 1
            nmax += e >= 1.0e4 * (1.0 - 1.0e-12) ? 1 : 0
            nmin += e <= 1.0e-4 * (1.0 + 1.0e-12) ? 1 : 0
        end
        printf("  current-card clip census at edot = %.0e x edot_ref: max %d/%d, " *
                "min %d/%d\n", f, nmax, ntot, nmin, ntot)
    end
end

main()
