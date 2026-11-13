#!/usr/bin/env julia
# Calibration of the composite (diffusion + dislocation) prefactors for the
# "realistic 2-D" statistical case (`case_realistic_2d.jl`, docs/dev/CASE_REALISTIC.md
# §3).  This script is the *derivation*; the case script carries the resulting
# numbers as literals, so the production run does not depend on it.
#
#   julia --project=. examples/calib_realistic_2d.jl
#
# What it does
# ------------
# 1. Measures the reference strain rate `edot_ref` from a constant-viscosity
#    precursor (eta = 1, Ra = 1e5, H = 10, same box/tree/IC as the case).  The
#    composite law needs a strain rate at which "the viscosity contrast over
#    T in [0.1, 0.9] is 1e4" is defined; using the RMS strain rate of the
#    developed flow makes the calibration representative of the run instead of
#    an arbitrary O(1) number.
# 2. Solves the two calibration conditions for the prefactors (comp_Ad,
#    comp_As) at fixed E_d, E_s, n, eta_T0:
#       (i)  eta_eff(T=0.5, z=0.5, edot_ref) = 1
#       (ii) eta_eff(T=0.1, z=0.5, edot_ref) / eta_eff(T=0.9, z=0.5, edot_ref)
#            = 1e4
#    The ratio condition depends only on the *branch ratio*
#    r = eta_diff/eta_disl at the calibration point (a common rescaling of both
#    prefactors leaves every ratio invariant), so it is a 1-D root find in r;
#    the common scale then follows from (i) in closed form.
# 3. Verifies the result against `eta_composite_at` itself (not against the
#    hand-rolled algebra) and prints the whole parameter card.
#
# No solver code is touched; this only calls exported functions.

using QuaST, Printf, LinearAlgebra

# ---------------------------------------------------------------- model card
const NEXP  = 3.0      # comp_n, stress exponent (mantle: 3-3.5)
const T0    = 0.5      # eta_T0, nondimensional absolute surface temperature
const ED    = 12.0     # comp_Ed = E_d/(R dT)
const ES    = 17.0     # comp_Es = E_s/(R dT)  (= 1.42 E_d, the ASPECT ratio)
const ZC    = 0.5      # calibration height
const TC    = 0.5      # calibration temperature
const TLOW  = 0.1      # contrast window [TLOW, THIGH] at fixed z
const THIGH = 0.9
const CONTRAST = 1.0e4 # required eta(TLOW)/eta(THIGH) at z = ZC, edot_ref

# ------------------------------------------------------------- case geometry
const LX     = 2.5
const CX     = 5
const CY     = 2
const PMAX   = 7        # max_level (leaf size 1/256 at the finest level)
const PORD   = 2        # polynomial order
const HHEAT  = 10.0     # internal-heating number (source H of the theta eq.)
const RA     = 1.0e5

# Unit-branch viscosities at (T, z = ZC, edot = edot_ref), normalised to 1 at
# (TC, ZC): eta_d = exp(ED*(1/T_nd - 1/T_ndc)), eta_s likewise with ES/NEXP.
Tnd(T) = T + T0
const TNDC = TC + T0
eta_d_unit(T) = exp(ED * (1.0 / Tnd(T) - 1.0 / TNDC))
eta_s_unit(T) = exp((ES / NEXP) * (1.0 / Tnd(T) - 1.0 / TNDC))

"""
    eta_iso_unit(T, r)

Isostress effective viscosity of two branches with `eta_d = eta_d_unit(T)` and
`eta_s = eta_d_unit(T)/r`, using the code's scalar equation `k x + x^n = 1` with
`k = inv_diff/inv_disl` and `eta = x/inv_disl = x*eta_s` (docs/dev/PHYSICS.md
§14.8.1).  `r` is the diffusion-to-dislocation branch ratio at the calibration
point.
"""
function eta_iso_unit(T::Float64, r::Float64)
    ed = eta_d_unit(T)
    es = eta_s_unit(T) / r
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

contrast_unit(r) = eta_iso_unit(TLOW, r) / eta_iso_unit(THIGH, r)

function solve_branch_ratio()
    # contrast(r) is *decreasing* in r: r -> 0 gives the pure-diffusion
    # contrast exp(E_d (1/T_nd(TLOW) - 1/T_nd(THIGH))), r -> Inf the
    # pure-dislocation one exp((E_s/n)(...)).  Bisect in log r.
    flo = contrast_unit(1.0e-8)
    fhi = contrast_unit(1.0e8)
    (flo > CONTRAST > fhi) ||
        error("contrast window at the two branch limits is [$(fhi), $(flo)], " *
              "which does not contain $(CONTRAST); adjust E_d/E_s")
    lo = log(1.0e-8); hi = log(1.0e8)
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        f = contrast_unit(exp(mid))
        f > CONTRAST ? (lo = mid) : (hi = mid)
    end
    return exp(0.5 * (lo + hi))
end

# ------------------------------------------------------------------ precursor
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
    @printf("precursor: steps=%d t=%.3e Vrms=%.4e max|u|=%.4e Nu_top=%.5f\n",
            res.steps, res.t, last[5], last[11], last[3])
    @printf("precursor: edot_rms(interior)=%.6e  edot_median=%.6e  max=%.6e\n",
            rms, sort(edi)[cld(length(edi), 2)], maximum(edi))
    return rms
end

function main()
    @printf("=== composite prefactor calibration (docs/dev/CASE_REALISTIC.md §3) ===\n")
    @printf("fixed: n=%.1f eta_T0=%.2f comp_Ed=%.2f comp_Es=%.2f Vd=Vs=0\n",
            NEXP, T0, ED, ES)
    edot_ref = precursor_edot()
    # round to 3 significant digits so the card is reproducible by hand
    edot_ref = round(edot_ref; sigdigits=3)
    @printf("edot_ref (rounded) = %.6e\n", edot_ref)

    r = solve_branch_ratio()
    scale = eta_iso_unit(TC, r)          # = x/r = eta_disl(TC) with unit prefactors
    xrt = scale * r                      # x of k x + x^n = 1 at the cal. point
    # eta_d(TC)=1/r*x, eta_s(TC)=x (see the header derivation)
    comp_Ad = exp(ED) * scale
    comp_As = exp(ES / NEXP) * edot_ref^(-(NEXP - 1.0) / NEXP) * xrt
    @printf("\nbranch ratio r = eta_d/eta_s at (T=%.2f,z=%.2f,edot_ref) = %.6f\n",
            TC, ZC, r)
    @printf("comp_Ad = %.10e\ncomp_As = %.10e  (edot_ref = %.6e)\n",
            comp_Ad, comp_As, edot_ref)

    # ------------------------------------------------------------- verification
    pr = Params(Ra=RA, H=HHEAT, eta_law=:composite, comp_rule=:isostress,
                comp_n=NEXP, comp_Ed=ED, comp_Es=ES, comp_Vd=0.0, comp_Vs=0.0,
                comp_Ad=comp_Ad, comp_As=comp_As, comp_eps_min=1.0e-12,
                comp_Tcut=0.0, eta_T0=T0, eta_min=-Inf, eta_max=Inf)
    e = (T, ed) -> eta_composite_at(T, ed, ZC, pr)[1]
    e05 = e(TC, edot_ref)
    c = e(TLOW, edot_ref) / e(THIGH, edot_ref)
    @printf("\nVERIFY  eta(0.5, z=0.5, edot_ref) = %.15e   (target 1)\n", e05)
    @printf("VERIFY  eta(0.1)/eta(0.9)          = %.15e   (target 1e4)\n", c)
    @printf("VERIFY  rel. deviations: %.3e / %.3e\n", abs(e05 - 1.0),
            abs(c - 1.0e4) / 1.0e4)
    @printf("\n eta(T,z=0.5) at the calibration rate (clamps off):\n")
    @printf("   T     eta(edot_ref)  eta(edot/100)  eta(10*edot)   eta(1000*edot)\n")
    for T in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 1.75, 2.0, 2.5)
        @printf("  %4.2f  %12.5e  %13.5e  %13.5e  %14.5e\n", T, e(T, edot_ref),
                e(T, edot_ref / 100), e(T, 10 * edot_ref),
                e(T, 1000 * edot_ref))
    end
    @printf("\n mid-depth base temperature of the case: Tbar(0.5,H=%.0f)=%.4f\n",
            HHEAT, Tbar(0.5, HHEAT))
    # ------------------------------------------------------------ clamp census
    for (lo, hi) in ((1.0e-4, 1.0e6), (1.0e-3, 1.0e5))
        prb = Params(Ra=RA, H=HHEAT, eta_law=:composite, comp_rule=:isostress,
                     comp_n=NEXP, comp_Ed=ED, comp_Es=ES, comp_Ad=comp_Ad,
                     comp_As=comp_As, comp_eps_min=1.0e-12, eta_T0=T0,
                     eta_min=lo, eta_max=hi)
        ncl = 0; ntot = 0
        for Tv in range(0.0, 2.6; length=261)
            _, cl = eta_composite_at(Tv, edot_ref, ZC, prb)
            ntot += 1; ncl += cl ? 1 : 0
        end
        @printf("clamp census z=0.5, edot_ref, eta in [%.0e,%.0e]: %d/%d of T in [0,2.6] clipped\n",
                lo, hi, ncl, ntot)
    end
    @printf("\ncard: comp_Ad=%.10e comp_As=%.10e comp_Ed=%.1f comp_Es=%.1f comp_n=%.1f\n",
            comp_Ad, comp_As, ED, ES, NEXP)
    @printf("      eta_T0=%.2f comp_eps_min=1e-12 comp_rule=:isostress  edot_ref=%.4e\n",
            T0, edot_ref)
end

main()
