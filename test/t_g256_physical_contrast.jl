# G256: the card's realised viscosity contrast over the temperature range it
# actually uses, and the calibration's mantle-contrast variant.
#
# **The defect this gate pins.**  docs/dev/HANDOVER.md §10.2 item 2 records that
# the production card's constitutive contrast is 7.8e6 against the mantle's
# 1e3 to 1e5, and that "a reader could reasonably think the card is mantle-like
# when it is not".  The calibration (`examples/calib_realistic_2d.jl`) solved
#     (i)  eta(T=0.5, z=0.5, edot_ref) = 1
#     (ii) eta(0.1)/eta(0.9) = 1e4
# so its contrast condition was evaluated over a 0.8-wide window inside the lid.
# The card's *field* does not live there: the base state is
#     Tbar(z,H) = (1-z) + (H/2) z (1-z),  H = 10     (src/physics/params.jl),
# whose maximum is (H+2)^2/(8H) = 1.8 at z* = (H-2)/(2H), and a developed field
# spans T in [0, 1.647] (docs/dev/PHYSICS.md §2.3).  At the calibration rate the
# card's realised contrast over that real range is 1.957e6, and a field census
# of its step-1 state gives 1.0e7 (the lid sits on eta_max), not ~1e4.
#
# **The extension.**  The calibration derives the hot end from the base state
# (printing the derivation) and evaluates the contrast condition over the real
# range.  The two prefactors still come from the existing conditions -- the
# value condition and the branch ratio r = eta_d/eta_s = 0.129720 that the old
# [0.1,0.9] ratio condition fixed -- and, because the prefactors then have no
# freedom left, the activation-energy scale E_d (with E_s/E_d held at the ASPECT
# 530/375 kJ/mol ratio) is solved for the target contrast.  For the default
# target 1e4 over [0, 1.8] the calibration prints
#     comp_Ed = 6.973731528, comp_Es = 9.879452998,
#     comp_Ad = 1.0658847774e3, comp_As = 2.4961576276e-2,
# which is the variant this gate pins.  (Re-solving the *branch ratio* over the
# wider range instead reaches the same thermal contrast but not a mantle-range
# field contrast: it drives the law into the dislocation limit and the realised
# census rises to 1.6e6 on a developed field -- a measured negative, so the
# ratio is held.)
#
# **What is asserted, in four parts.**
#   (a) the hot end is derived from the card's own base state, and the variant
#       satisfies both calibration conditions at the reference point over the
#       new range, to a stated tolerance (1e-9 on the value condition, 1e-8 on
#       the contrast -- the solve itself closes to 2e-13); neither eta is on a
#       clamp, so the number is the law's;
#   (b) the production card's realised contrast over the SAME range is outside
#       the mantle range by more than a decade, so this is not a 0 == 0 gate:
#       the variant changes what the card realises.  With the range set back to
#       the old [0.1,0.9] window the card already meets the default target, which
#       is the calibration's "unchanged behaviour" branch;
#   (c) the card knobs `CASE_COMP_ED`/`CASE_COMP_ES`: read from the environment,
#       forwarded into `Params`, printed, and defaulting to the card's literals
#       12.0/17.0 with no transformation -- the default card is the same
#       parameter card as before (the pre-existing default trajectory is pinned
#       by the regression gates; here the *inputs* are pinned);
#   (d) a field census: a one-step run of the card's own geometry, seed, phases
#       and clamps with the variant parameters, eta evaluated over the velocity
#       nodes exactly as the card's frame writer does; the realised contrast
#       must land in [1e3, 1e5].  The same run with the card's literals is two
#       decades above, so the variant is what moves the census into range.
using Test, QuaST, Printf

# ---------------------------------------------------------------- constants
# examples/calib_realistic_2d.jl, default run; edot_ref rounded to 3 s.d.
const G256_EDOT_REF = 1.65e3
const G256_H        = 10.0
const G256_TTOP     = 0.0
const G256_TBOT     = 1.8        # max_z Tbar(z,H) = (H+2)^2/(8H)
const G256_ZC       = 0.5
const G256_TC       = 0.5
const G256_TARGET   = 1.0e4
# the production card's literals (unchanged by the knobs)
const G256_C_ED = 12.0
const G256_C_ES = 17.0
const G256_C_AD = 1.6240183119e5
const G256_C_AS = 2.6796419238e-1
# the variant the extended calibration solves for the default target
const G256_V_ED = 6.973731528
const G256_V_ES = 9.879452998
const G256_V_AD = 1.0658847774e3
const G256_V_AS = 2.4961576276e-2
# the mantle's viscosity-contrast range this project uses
# (docs/dev/HANDOVER.md §6 and §10.2 item 2, docs/dev/PHASE1.md §1.2)
const G256_MANTLE_LO = 1.0e3
const G256_MANTLE_HI = 1.0e5
# the old contrast window, kept because it is what fixes the branch ratio
const G256_TLOW_REF  = 0.1
const G256_THIGH_REF = 0.9
const G256_RATIO_REF = 1.0e4

"Maximum of the card's base state over the layer: the analytic value."
g256_base_state_max(H) = H <= 2.0 ? 1.0 : (H + 2.0)^2 / (8.0 * H)

"Card `Params` with the composite law at the given material parameters."
function g256_params(ED, ES, AD, AS; eta_min=-Inf, eta_max=Inf)
    return Params(Ra=1.0e5, H=G256_H, eta_law=:composite, comp_rule=:isostress,
                  comp_Ad=AD, comp_As=AS, comp_Ed=ED, comp_Es=ES,
                  comp_Vd=0.0, comp_Vs=0.0, comp_n=3.0, comp_Tcut=0.0,
                  comp_eps_min=1.0e-12, eta_T0=0.5, eta_min=eta_min,
                  eta_max=eta_max, bc=:free, latent_L=0.11,
                  latent_mode=:per_phase)
end

# The production card's own geometry, seed, phases and run kwargs
# (examples/case_realistic_2d.jl): the census below is the card's field.
g256_seed(x, y) = 0.05 * sinpi(y) * cospi(2.0 * x / 2.5) +
                  0.02 * sinpi(2.0 * y) * cospi(4.0 * x / 2.5)

function g256_tree()
    qt = Quadtree(7; Lx=2.5, cells_x=5, cells_y=2)
    for _ in 1:3, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

function g256_phases()
    ph = [PhaseTransition(0.87, 0.025, 0.032, 0.5, 180.0, 1.8181818181818181),
          PhaseTransition(0.78, 0.025, -0.032, 0.5, 300.0, 3.0303030303030303)]
    return per_phase_latent(Params(Ra=1.0e5, H=G256_H), ph;
                            rho_ref=3300.0, cp=1000.0, dT=1000.0, g=9.81,
                            d=2.9e6, T_surface=0.5)
end

function g256_run(pr; nsteps::Int)
    qt = g256_tree()
    ss = make_semstokes(qt, 2)
    theta0 = [g256_seed(x, y) for (x, y) in ss.su.coords]
    ph = g256_phases()
    return run_semquad_simulation(qt, pr, theta0;
        p=2, tstop=Inf, maxsteps=nsteps,
        dt0=1.0e-6, dtmax=5.0e-5, dtmin=1.0e-9, cfl=0.5, dt_growth=1.05,
        maxpic=1, pic_tol=1.0e-6, pic_mode=:picard, amr_every=5,
        refine_frac=0.2, coarsen_frac=0.05, max_leaves=10_000, max_level=7,
        symmetry=:none, wall_min_level=5, wall_strip=0.1, amr_hysteresis=1,
        amr_indicator=:theta, amr_coarsen_gate=true, transfer_mode=:nodal,
        advection=:skew, ev=0.11, ev_cmax=0.052, ev_alpha=2,
        linear=:auto, stokes_tol=1.0e-3, mg_tol=1.0e-8,
        stokes_scaling=:none, mass_form=:boussinesq, phases=ph,
        latent_scheme=:implicit, verbose=false)
end

"""
    g256_census(res, pr)

The card's frame-writer census on a returned state: `T = Tbar(y,H) + theta` at
the velocity nodes, `edot` from the returned velocity, `eta` through the public
closure with whatever bounds `pr` carries (the card's `[1e-4, 1e4]`).  Returns
the min/max eta, their ratio (the realised field contrast), the field's
temperature and strain-rate ranges, and the number of clipped nodes.
"""
function g256_census(res::NamedTuple, pr::Params)
    ss = make_semstokes(res.qt, 2)
    ed = semquad_strain_rate_field(ss.su, res.ux, res.uz)
    n = length(ed)
    T = Vector{Float64}(undef, n)
    eta = Vector{Float64}(undef, n)
    nclip = 0
    for g in 1:n
        x, y = ss.su.coords[g]
        T[g] = Tbar(y, pr.H) + res.theta[g]
        eta[g], cl = eta_composite_at(T[g], ed[g], y, pr)
        nclip += cl ? 1 : 0
    end
    return (; eta_min=minimum(eta), eta_max=maximum(eta),
            contrast=maximum(eta) / minimum(eta), Tmin=minimum(T),
            Tmax=maximum(T), edot_min=minimum(ed), edot_max=maximum(ed),
            nclip, N=n)
end

@testset "G256 physical contrast over the card's real temperature range" begin
    # ---- (a) the derived hot end, and the variant's calibration ----------
    # The range is derived from the card's own base state, not assumed: for
    # H = 10 the maximum is (H+2)^2/(8H) = 1.8 at z = 0.4.  The bottom *wall*
    # value (1.0) and the mid-depth value the old 7.8e6 census used (1.75) are
    # printed by the calibration alongside it; the hot end here is the maximum
    # the base state reaches, because that is the hottest material the field can
    # contain.
    @test g256_base_state_max(G256_H) == 1.8
    @test Tbar(0.0, G256_H) == 1.0
    @test Tbar(0.5, G256_H) == 1.75
    zs = range(0.0, 1.0; length=2001)          # z = 0.4 is on the grid
    @test isapprox(maximum(Tbar(z, G256_H) for z in zs), G256_TBOT; rtol=1e-13)

    pr_v = g256_params(G256_V_ED, G256_V_ES, G256_V_AD, G256_V_AS)
    eta05, clip05 = eta_composite_at(G256_TC, G256_EDOT_REF, G256_ZC, pr_v)
    eta_top, clip_top = eta_composite_at(G256_TTOP, G256_EDOT_REF, G256_ZC, pr_v)
    eta_bot, clip_bot = eta_composite_at(G256_TBOT, G256_EDOT_REF, G256_ZC, pr_v)
    @test !clip05 && !clip_top && !clip_bot
    @test isapprox(eta05, 1.0; rtol=1.0e-8)                     # condition (i)
    @test isapprox(eta_top / eta_bot, G256_TARGET; rtol=1.0e-8) # condition (ii)
    # the calibration's own residual is 2e-13; the pinned literals are its
    # printed 10 significant digits, which is what sets the 1e-8 tolerance
    @test abs(eta_top / eta_bot / G256_TARGET - 1.0) < 1.0e-9
    @printf("G256 (a) derived range [%.1f, %.1f]; eta(0.5)=%.15e, eta(top)=%.6e, eta(bot)=%.6e, contrast=%.15e\n",
            G256_TTOP, G256_TBOT, eta05, eta_top, eta_bot, eta_top / eta_bot)

    # ---- (b) the card's own contrast over the same range is NOT mantle-like -
    pr_c = g256_params(G256_C_ED, G256_C_ES, G256_C_AD, G256_C_AS)
    ct, _ = eta_composite_at(G256_TTOP, G256_EDOT_REF, G256_ZC, pr_c)
    cb, _ = eta_composite_at(G256_TBOT, G256_EDOT_REF, G256_ZC, pr_c)
    c_card = ct / cb
    @test isapprox(c_card, 1.957303e6; rtol=1.0e-5)  # the calibration's number
    @test c_card > G256_MANTLE_HI
    @test c_card > 10 * G256_MANTLE_HI               # outside by a decade+
    # ... and the card's OLD window still carries the old ratio condition, which
    # is the "unchanged behaviour" branch: with [0.1,0.9] the card already meets
    # the default target, so the extended calibration changes nothing there.
    c_old = eta_composite_at(G256_TLOW_REF, G256_EDOT_REF, G256_ZC, pr_c)[1] /
            eta_composite_at(G256_THIGH_REF, G256_EDOT_REF, G256_ZC, pr_c)[1]
    @test isapprox(c_old, G256_RATIO_REF; rtol=1.0e-9)
    @test abs(c_old / G256_TARGET - 1.0) <= 1.0e-9   # calibration's no-op trigger
    @printf("G256 (b) card contrast over the real range = %.6e (>%.0e); over [0.1,0.9] = %.6e\n",
            c_card, G256_MANTLE_HI, c_old)

    # ---- (c) the card knobs, and the default parameters are unchanged ------
    card = normpath(joinpath(@__DIR__, "..", "examples",
                             "case_realistic_2d.jl"))
    @test isfile(card)
    src = read(card, String)
    # read from the environment with the card's literals as the defaults, with
    # no arithmetic applied to them (so unset == the old parameter card)
    m_ed = match(r"const\s+COMP_ED\s*=\s*parse\(Float64,\s*" *
                 r"get\(ENV,\s*\"CASE_COMP_ED\",\s*\"([0-9.]+)\"\)\)", src)
    m_es = match(r"const\s+COMP_ES\s*=\s*parse\(Float64,\s*" *
                 r"get\(ENV,\s*\"CASE_COMP_ES\",\s*\"([0-9.]+)\"\)\)", src)
    @test m_ed !== nothing
    @test m_es !== nothing
    @test parse(Float64, m_ed.captures[1]) == G256_C_ED
    @test parse(Float64, m_es.captures[1]) == G256_C_ES
    # forwarded into Params, where the driver reads them
    @test occursin("comp_Ed=COMP_ED", src)
    @test occursin("comp_Es=COMP_ES", src)
    # printed with their value, so a log records which energies ran
    @test occursin("Ed=%.6f Es=%.6f", src)
    @test occursin(", COMP_ED, COMP_ES,", src)
    # documented as opt-in with the mantle-contrast variant
    @test occursin("CASE_COMP_ED", src)
    @test occursin("CASE_COMP_ES", src)
    @test occursin("mantle", src)
    # the library side the card relies on: the energies reach Params
    pr = Params(comp_Ed=G256_C_ED, comp_Es=G256_C_ES)
    @test pr.comp_Ed == G256_C_ED
    @test pr.comp_Es == G256_C_ES
    @test_throws ArgumentError Params(eta_law=:composite, comp_Ad=1.0,
                                      comp_Ed=-1.0)
    @test_throws ArgumentError Params(eta_law=:composite, comp_Ad=1.0,
                                      comp_Es=-1.0)

    # ---- (d) a field census: the variant lands in the mantle range ---------
    # The card's own geometry, seed, phases and clamps; one step produces the
    # state the card's own frame at step 1 samples.  The two arms differ only in
    # the four composite parameters.
    pr_v_card = g256_params(G256_V_ED, G256_V_ES, G256_V_AD, G256_V_AS;
                            eta_min=1.0e-4, eta_max=1.0e4)
    pr_c_card = g256_params(G256_C_ED, G256_C_ES, G256_C_AD, G256_C_AS;
                            eta_min=1.0e-4, eta_max=1.0e4)
    res_v = g256_run(pr_v_card; nsteps=1)
    res_c = g256_run(pr_c_card; nsteps=1)
    @test res_v.steps == 1 && res_c.steps == 1
    @test !isempty(res_v.history) && !isempty(res_c.history)
    cen_v = g256_census(res_v, pr_v_card)
    cen_c = g256_census(res_c, pr_c_card)
    @printf("G256 (d) variant census: eta in [%.4e, %.4e], contrast %.4e, T in [%.3f, %.3f], edot in [%.2e, %.2e], clipped %d/%d\n",
            cen_v.eta_min, cen_v.eta_max, cen_v.contrast, cen_v.Tmin,
            cen_v.Tmax, cen_v.edot_min, cen_v.edot_max, cen_v.nclip, cen_v.N)
    @printf("G256 (d) card census   : eta in [%.4e, %.4e], contrast %.4e, clipped %d/%d\n",
            cen_c.eta_min, cen_c.eta_max, cen_c.contrast,
            cen_c.nclip, cen_c.N)
    # the field really spans a range wider than the old calibration window
    @test cen_v.Tmin == 0.0
    @test cen_v.Tmax > 1.8          # the seed's anomaly reaches 1.8594
    @test cen_c.Tmax > 1.8
    @test cen_v.edot_max > G256_EDOT_REF
    # the variant's field contrast lands in the mantle range ...
    @test G256_MANTLE_LO <= cen_v.contrast <= G256_MANTLE_HI
    # ... while the production card's is two decades above it
    @test cen_c.contrast > 10 * G256_MANTLE_HI
    @test cen_v.contrast < cen_c.contrast / 100
    # and the two contrasts come from different physics, not the same numbers
    @test cen_v.nclip == 0           # the variant is not carried by a clamp
    @test cen_c.nclip > 0            # the card's lid sits on eta_max
    @test res_v.theta != res_c.theta
end
