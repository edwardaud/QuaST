# G257: the Rayleigh-number reference viscosity (`Params.eta_ref`), the
# convention that makes a card comparable with a paper.
#
# **The defect this gate pins.**  Every mature mantle code defines the Rayleigh
# number with an explicit reference viscosity and states it,
#
#     Ra = rho g alpha dT d^3 / (kappa eta_ref)
#
# (ASPECT works dimensionally and historically exposed a `Reference viscosity`
# for its scaling, https://aspect-documentation.readthedocs.io/en/latest/user/
# methods/dimensionalize/index.html; Schulz et al. 2020, GJI 220, 18-36 state
# their reference Rayleigh number with an eta_r that "represents just a
# viscosity that is thought to be characteristic for the system",
# https://academic.oup.com/gji/article/220/1/18/5571091).  This solver had the
# reference **implicit**: `Ra` is a parameter and the production card's composite
# law is calibrated so that eta = 1 at (T = 0.5, z = 0.5, edot_ref), so `eta_ref`
# was silently whatever the calibration produced.  The consequence is a real
# foot-gun: rescaling the law's prefactors changes every physical viscosity
# without changing `Ra`, because nothing ties the two together.
#
# **The convention.**  `Params.eta_ref` (default 1.0) is the viscosity the
# Rayleigh number is defined with, in the units the material law returns; the
# momentum solve uses `eta_law/eta_ref` wherever the law's viscosity enters (the
# same wrapper chain that applies `eta_multiplier` and the Maxwell closure).
# Rescaling the law by a constant and `eta_ref` by the same constant cancels
# exactly: the flow is unchanged.  Rescaling the law alone is the foot-gun.
#
# **What is asserted.**
#   (a) the card knob `CASE_ETA_REF` (default 1.0), its forwarding into `Params`,
#       its header line, the scale set that makes the report print Pa s, and the
#       calibration's declaration that the prefactors produce `eta_ref = 1`; plus
#       the library default and the parameter validation;
#   (b) the default is bit-identical: the pinned state below was captured at HEAD
#       `2d7ac42` (the commit before this gate) and again from the working tree
#       with `--threads=8`; the two are byte-identical.  The gate re-runs the
#       same card and requires exact equality of the history and of theta/ux/uz;
#       a run at the default convention emits no eta_ref report;
#   (c) the **invariance that defines the convention**: the composite law with
#       both prefactors divided by c and `eta_ref = c` reproduces the base run to
#       round-off, while the same scaled law with `eta_ref = 1` changes the flow
#       by O(1) (the foot-gun).  The clamps are absolute bounds applied inside
#       the law, so the clamped arm rescales them too; leaving them alone is
#       shown to break the invariance.  The same cancellation is checked through
#       the Maxwell effective-viscosity closure (`shear_modulus > 0`), i.e. the
#       whole wrapper chain, not just the raw law;
#   (d) the run report: exactly two `Info` records when `eta_ref != 1` (first
#       flow solve and end of run), whose numbers are a *direct* recomputation
#       from the state the run used (the same public closures, so the tolerance
#       is the printed precision), including `eta_0` in Pa s from the card's
#       scale set, and the stated interior-band definition (the middle half of
#       the layer).  `eta_ref = 1` emits no report at all;
#   (e) the path that does not apply the normalisation refuses the combination
#       instead of silently solving a different problem: `pic_mode = :newton`.
#
# The helper functions and constants are at file scope (not inside the testset):
# G255 documents why a closure inside a `@testset` is the wrong shape for
# helpers that compare against file-scope constants.
using Test, QuaST, Logging, Printf

# ---------------------------------------------------------------- scenario ----
# A 16-leaf uniform unit square (NOT one leaf: HANDOVER section 8).  Two fixed
# 1e-3 steps with the direct solver: the whole scenario costs milliseconds.
function g257_tree()
    qt = Quadtree(3)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

g257_seed(x, y) = 0.05 * sinpi(y) * cospi(2.0 * x) +
                  0.02 * sinpi(2.0 * y) * cospi(4.0 * x)

# The production card's composite-law parameters (case_realistic_2d.jl).
const G257_AD = 1.6240183119e5
const G257_AS = 2.6796419238e-1

"""
    g257_params_comp(; Ad, As, eta_min, eta_max, G, eta_ref) -> Params

The card's composite law on the gate's box.  The default has the clamps off
(`-Inf`/`Inf`), which is what makes the prefactor-scaling invariance exact: the
clamps are absolute bounds inside the law, so a clamped law needs its bounds
rescaled too (checked separately).  `Ra = 1e3` keeps the velocities readable.
"""
function g257_params_comp(; Ad::Float64=G257_AD, As::Float64=G257_AS,
                          eta_min::Float64=-Inf, eta_max::Float64=Inf,
                          G::Float64=0.0, eta_ref::Float64=1.0)
    return Params(Ra=1.0e3, H=0.0, eta_law=:composite, comp_rule=:isostress,
                  comp_Ad=Ad, comp_As=As,
                  comp_Ed=12.0, comp_Es=17.0, comp_Vd=0.0, comp_Vs=0.0,
                  comp_n=3.0, comp_Tcut=0.0, comp_eps_min=1.0e-12,
                  eta_T0=0.5, eta_min=eta_min, eta_max=eta_max,
                  bc=:free, shear_modulus=G, eta_ref=eta_ref)
end

"Temperature-dependent law for the pinned default-path reference."
function g257_params_ref(; eta_ref::Union{Nothing,Float64}=nothing)
    if eta_ref === nothing
        return Params(Ra=1.0e3, H=0.0, eta_law=:expT, deta_T=1.0e2, bc=:free)
    end
    return Params(Ra=1.0e3, H=0.0, eta_law=:expT, deta_T=1.0e2, bc=:free,
                  eta_ref=eta_ref)
end

"""
    g257_run(pr; nsteps=2, callback=nothing, eta_ref_scale=nothing)

Two fixed steps of the gate's box with the direct solver (`--threads=8` and
`--threads=1` produce the same bits; the pinned reference was captured at 8).
"""
function g257_run(pr::Params; nsteps::Int=2, callback=nothing,
                  eta_ref_scale=nothing)
    qt = g257_tree()
    ss = make_semstokes(qt, 2)
    theta0 = [g257_seed(x, y) for (x, y) in ss.su.coords]
    return run_semquad_simulation(qt, pr, theta0; p=2, tstop=Inf,
                                  maxsteps=nsteps, dt0=1.0e-3, dtmax=1.0e-3,
                                  dtmin=1.0e-30, cfl=0.0, maxpic=1,
                                  linear=:direct, stokes_tol=Inf,
                                  advection=:skew, callback=callback,
                                  eta_ref_scale=eta_ref_scale, verbose=false)
end

# Silence the run report in the flow-only arms; the report itself is asserted in
# section (d), and a silenced run is also what proves it changes no number.
g257_silent(f) = Logging.with_logger(f, Logging.NullLogger())

"L2-relative difference ||a-b||/||b|| (a max-relative metric is useless here:
the boundary velocity is exactly zero in one run and round-off in the other)."
l2rel(a, b) = sqrt(sum(abs2, a .- b)) / sqrt(sum(abs2, b))
g257_relhist(a, b) = l2rel(reduce(vcat, a), reduce(vcat, b))

g257_report_msgs(logs) = [l.message for l in logs
                          if l.level == Logging.Info &&
                             occursin("eta_ref convention", l.message)]

# The dimensional scale set the production card passes to the driver (the ASPECT
# scale set.
const G257_SCALE = (rho=3300.0, g=9.81, alpha=3.0e-5, dT=1000.0, d=2.9e6,
                    kappa=1.0e-6)

# ------------------------------------------------- pinned default-path state ----
# Captured at HEAD 2d7ac42 (pre-gate) AND from the working tree at the gate's
# commit, both with `--threads=8`; `diff` of the two dumps is empty.  The state
# is the complete trajectory evidence of the default (eta_ref = 1) path: two
# history rows of 18 columns and the three Q2 nodal fields.
const G257_REF_HISTORY = [0.001, 0.001, 1.0008440140668564, 1.013366376839702, 4.593806867056774, 0.5000095838796684, 81.0, 16.0, 19.402897508314027, 0.25, 11.886453733577627, 0.04754581493431051, -0.012522362772844669, 0.6086068677476874, 0.8527858111282665, 0.08478233693106102, 84.78233693106169, -7.87794009685751e-15, 0.002, 0.001, 1.0016968212573298, 1.0288001186591733, 5.199515551138152, 0.5000273484375652, 81.0, 16.0, 22.893377986270167, 0.25, 13.132678523938749, 0.05253071409575499, -0.027103297401843966, 0.6340557615686725, 0.8893880230371285, 0.10399950114560083, 103.99950114559995, 8.47189633255076e-15]
const G257_REF_THETA = [0.0, 0.0, 0.0, 0.03841712131392624, 0.01805496357572729, -0.014253096694257871, 0.0639406811291327, 0.03379495573912916, -0.016863208340668588, 0.0688192912084141, 0.04135144798833763, -0.009768973047320325, 0.05812037897754333, 0.04112125555507573, 0.0012931415545072845, 0.04008906042518444, 0.03510625693049961, 0.010573101549609661, 0.022705520744482656, 0.025479774417102023, 0.013910505318759368, 0.009615636789621428, 0.013408954015597751, 0.009595239798950233, 0.0, 0.0, 0.0, 0.0, 0.0, -0.022113437774068833, -0.018227894032945335, -0.035739891352780113, -0.03423761761927608, -0.04016326284758922, -0.046797915670718476, -0.038687069786049376, -0.055764408217903766, -0.03319566172022067, -0.05738878337534564, -0.024351034771996717, -0.04826235288636659, -0.012917984836738063, -0.027802592890949315, 0.0, 0.0, 0.0, 0.0, -0.022113437774068836, -0.014253096694257887, -0.03573989135278009, -0.016863208340668626, -0.040163262847589226, -0.00976897304732034, -0.0386870697860493, 0.0012931415545072756, -0.03319566172022068, 0.010573101549609665, -0.0243510347719967, 0.013910505318759361, -0.012917984836738054, 0.009595239798950246, 0.0, 0.0, 0.0, 0.0, 0.01805496357572732, 0.03841712131392628, 0.033794955739129225, 0.0639406811291327, 0.04135144798833768, 0.06881929120841415, 0.04112125555507574, 0.058120378977543345, 0.035106256930499666, 0.04008906042518447, 0.025479774417102034, 0.022705520744482687, 0.013408954015597782, 0.009615636789621443, 0.0, 0.0]
const G257_REF_UX = [0.0, -10.597327656067442, -11.800692159542484, 0.0, -5.169580678719629, -6.257396832107522, 0.0, 0.41214373751822275, -0.25165364731824247, 0.0, 2.699787535789351, 2.6566885602797106, 0.0, 2.9187338069378734, 3.2516333377099844, 0.0, 2.0377716826097476, 2.511298644767122, 0.0, 1.2310570662539166, 1.6725002356483836, 0.0, 0.7501708481157198, 1.0848668485006308, 0.0, 0.5969897331330648, 0.8893880230371121, -7.5107244517353005, 6.059791723710902e-14, -3.9696693072123255, 4.0934445568741036e-14, -0.27143798776061046, 1.6248129335735376e-14, 1.6998972448089293, -2.6018379257660286e-15, 2.137950012397063, -1.8770791809652658e-14, 1.7241716551947712, -1.8302485030875267e-16, 1.2174725060808835, -1.1323653479486891e-14, 0.8118396697584797, -3.555662336928075e-15, 0.6739271897675888, -2.795638493875013e-16, 7.510724451735375, 11.80069215954259, 3.969669307212384, 6.257396832107583, 0.2714379877606233, 0.2516536473182616, -1.6998972448089515, -2.656688560279745, -2.1379500123970776, -3.251633337710008, -1.724171655194795, -2.5112986447671255, -1.217472506080888, -1.6725002356483913, -0.8118396697584996, -1.0848668485006556, -0.6739271897676125, -0.8893880230371285, 10.59732765606749, 0.0, 5.169580678719661, 0.0, -0.4121437375182167, 0.0, -2.699787535789361, 0.0, -2.9187338069378836, 0.0, -2.0377716826097525, 0.0, -1.2310570662539222, 0.0, -0.7501708481157308, 0.0, -0.5969897331330734, 0.0]
const G257_REF_UZ = [0.0, 0.0, 0.0, 9.499488123024632, 5.299472576614657, -2.099687048816745, 13.132678523938695, 7.267515124415112, -2.3931672917006255, 10.241767467067895, 6.070509656462153, -1.4832887428978232, 6.491153324395156, 4.268675871120168, -0.5435153896150909, 3.550015282647924, 2.565276928824367, -0.05843186110137439, 1.7145601600926754, 1.3362169416244558, 0.10408270315224993, 0.6832512175909535, 0.541056065473873, 0.07605158933753595, 0.0, 0.0, 0.0, 0.0, 0.0, -5.343564441480409, -5.217034185716066, -7.272084072678015, -7.954917668334268, -6.115157301934769, -7.207146668684816, -4.254288003745536, -5.3926354166036115, -2.564487148970624, -3.4480741268003516, -1.34211666148666, -1.9211114972479821, -0.5415158506739002, -0.8280190528910979, 0.0, 0.0, 0.0, 0.0, -5.343564441480406, -2.099687048816741, -7.272084072678008, -2.393167291700609, -6.115157301934757, -1.4832887428978303, -4.2542880037455255, -0.5435153896150774, -2.5644871489706023, -0.058431861101362566, -1.3421166614866558, 0.10408270315225555, -0.5415158506739008, 0.07605158933753939, 0.0, 0.0, 0.0, 0.0, 5.299472576614683, 9.499488123024644, 7.2675151244151435, 13.132678523938749, 6.070509656462182, 10.241767467067938, 4.268675871120194, 6.49115332439519, 2.565276928824392, 3.5500152826479567, 1.3362169416244698, 1.7145601600927025, 0.5410560654738834, 0.6832512175909671, 0.0, 0.0]

@testset "G257 the eta_ref convention" begin
    # ================================================================ (a) ===
    # The card knob, the forward wiring, and the calibration's declaration.
    card = normpath(joinpath(@__DIR__, "..", "examples",
                             "case_realistic_2d.jl"))
    @test isfile(card)
    src = read(card, String)
    # read from the environment, default 1.0 (the historical convention)
    @test occursin("CASE_ETA_REF", src)
    @test occursin("get(ENV, \"CASE_ETA_REF\", \"1.0\")", src)
    @test occursin("isfinite(v) && v > 0.0", src)
    # forwarded into Params, where the driver reads it
    @test occursin("eta_ref=ETA_REF", src)
    # printed in the header line, with its argument (the card's convention)
    @test occursin("eta_ref=%.6g", src)
    @test occursin(", ETA_REF,", src)
    # the dimensional scale set for the Pa s report, passed to the driver
    @test occursin("const SCALE_SET", src)
    @test occursin("eta_ref_scale=SCALE_SET", src)
    # the frame census is in eta_ref units (the units the solve uses)
    @test occursin("v /= ETA_REF", src)
    # the calibration declares the reference its prefactors produce, and the
    # rescaling rule for a user
    calib = normpath(joinpath(@__DIR__, "..", "examples",
                              "calib_realistic_2d.jl"))
    @test isfile(calib)
    csrc = read(calib, String)
    @test occursin("the reference viscosity this calibration declares", csrc)
    @test occursin("eta_ref = 1.0", csrc)
    @test occursin("eta_ref = c", csrc)
    @test occursin("rescale", csrc)
    @test occursin("comp_Ad", csrc) && occursin("comp_As", csrc)

    # the library side: default 1.0, read-back, validation
    @test Params().eta_ref == 1.0
    @test Params(eta_ref=2.5).eta_ref == 2.5
    @test Params(eta_law=:expT, deta_T=2.0, eta_ref=1).eta_ref == 1.0
    @test_throws ArgumentError Params(eta_ref=0.0)
    @test_throws ArgumentError Params(eta_ref=-1.0)
    @test_throws ArgumentError Params(eta_ref=NaN)
    @test_throws ArgumentError Params(eta_ref=Inf)
    guard_src = read(normpath(joinpath(@__DIR__, "..", "src", "driver",
                                       "restart_guard.jl")), String)
    # a restart cannot switch eta_ref silently (old checkpoints skip the key)
    @test occursin("run_config[\"eta_ref\"] = pr.eta_ref", guard_src)

    # ================================================================ (b) ===
    # The default is bit-identical: the complete state of the pinned scenario.
    r_def = g257_silent(() -> g257_run(g257_params_ref()))
    @test r_def.steps == 2
    @test reduce(vcat, r_def.history) == G257_REF_HISTORY
    @test r_def.theta == G257_REF_THETA
    @test r_def.ux == G257_REF_UX
    @test r_def.uz == G257_REF_UZ
    # the pinned state is not a degenerate all-zero field
    @test maximum(abs, G257_REF_THETA) > 1.0e-2
    @test maximum(abs, G257_REF_UX) > 1.0
    # explicitly setting the default convention takes the same path
    r_one = g257_silent(() -> g257_run(g257_params_ref(eta_ref=1.0)))
    @test r_one.history == r_def.history
    @test r_one.theta == r_def.theta
    @test r_one.ux == r_def.ux && r_one.uz == r_def.uz
    # ... and the default convention emits no eta_ref report
    logs_one, _ = Test.collect_test_logs(min_level=Logging.Info) do
        g257_run(g257_params_ref(eta_ref=1.0))
    end
    @test isempty(g257_report_msgs(logs_one))
    @printf("G257 (b) pinned default state reproduced bit for bit (%d history values, %d nodes); eta_ref=1 silent\n",
            length(G257_REF_HISTORY), length(G257_REF_THETA))

    # ================================================================ (c) ===
    # The invariance: scaling the law's prefactors and eta_ref together cancels;
    # scaling the law alone is the foot-gun.
    c = 10.0
    base = g257_silent(() -> g257_run(g257_params_comp()))
    inv = g257_silent(() -> g257_run(g257_params_comp(Ad=G257_AD / c,
                                                      As=G257_AS / c,
                                                      eta_ref=c)))
    foot = g257_silent(() -> g257_run(g257_params_comp(Ad=G257_AD / c,
                                                       As=G257_AS / c,
                                                       eta_ref=1.0)))
    d_inv = (hist=g257_relhist(base.history, inv.history),
             theta=l2rel(base.theta, inv.theta),
             ux=l2rel(base.ux, inv.ux), uz=l2rel(base.uz, inv.uz))
    d_foot = (hist=g257_relhist(base.history, foot.history),
              theta=l2rel(base.theta, foot.theta),
              ux=l2rel(base.ux, foot.ux), uz=l2rel(base.uz, foot.uz))
    # the cancellation is exact up to round-off (measured ~1e-15)
    @test d_inv.hist < 1.0e-12
    @test d_inv.theta < 1.0e-12
    @test d_inv.ux < 1.0e-10
    @test d_inv.uz < 1.0e-10
    @test isapprox(inv.history[end][5], base.history[end][5]; rtol=1.0e-12)
    # the foot-gun: the same Ra, a law c times stiffer, a visibly different flow.
    # Measured at c = 10: hist 1.4e-1, ux 9.4 (the velocity is ~c times smaller,
    # so the L2-relative difference against it is large).
    @test d_foot.hist > 1.0e-2
    @test d_foot.theta > 1.0e-3
    @test d_foot.ux > 1.0
    @test d_foot.uz > 1.0
    @test d_foot.hist > 1.0e8 * d_inv.hist
    @test foot.history[end][5] < base.history[end][5] / 5.0
    @test foot.theta != base.theta
    @printf("G257 (c) invariance: hist %.3e, theta %.3e, ux %.3e, uz %.3e (round-off)\n",
            d_inv.hist, d_inv.theta, d_inv.ux, d_inv.uz)
    @printf("G257 (c) foot-gun  : hist %.3e, theta %.3e, ux %.3e, uz %.3e (c=%.0f)\n",
            d_foot.hist, d_foot.theta, d_foot.ux, d_foot.uz, c)

    # The clamps are absolute bounds inside the law (applied before the eta_ref
    # normalisation), so a pure law rescaling rescales them too.  A tight window
    # is used because the card's own [1e-4, 1e4] barely engages on this box
    # (eta spans 1.3e-3 to 3.1e3 there), which would make the arm vacuous.
    cl_lo, cl_hi = 0.5, 2.0
    pr_tight = g257_params_comp(eta_min=cl_lo, eta_max=cl_hi)
    cl_base = g257_silent(() -> g257_run(pr_tight))
    # non-vacuous: the hot interior sits on eta_min and the cold lid on eta_max
    @test eta_composite_at(1.75, 1.0e3, 0.5, pr_tight)[2]
    @test eta_composite_at(0.0, 1.0e3, 0.5, pr_tight)[2]
    # the clamps change the flow of this arm (the window is really active)
    @test g257_relhist(base.history, cl_base.history) > 1.0e-2
    cl_inv = g257_silent(() -> g257_run(g257_params_comp(Ad=G257_AD / c,
                                                         As=G257_AS / c,
                                                         eta_min=cl_lo * c,
                                                         eta_max=cl_hi * c,
                                                         eta_ref=c)))
    @test g257_relhist(cl_base.history, cl_inv.history) < 1.0e-12
    @test l2rel(cl_base.theta, cl_inv.theta) < 1.0e-12
    # leaving the bounds unscaled is NOT the invariance: the law is scaled but the
    # clip window is not, so the two runs solve different problems
    cl_wrong = g257_silent(() -> g257_run(g257_params_comp(Ad=G257_AD / c,
                                                           As=G257_AS / c,
                                                           eta_min=cl_lo,
                                                           eta_max=cl_hi,
                                                           eta_ref=c)))
    @test g257_relhist(cl_base.history, cl_wrong.history) > 1.0e-2
    @printf("G257 (c) clamped arm: clamps active (%.3e vs unclamped), scaled bounds invariant (%.3e), unscaled bounds break it (%.3e)\n",
            g257_relhist(base.history, cl_base.history),
            g257_relhist(cl_base.history, cl_inv.history),
            g257_relhist(cl_base.history, cl_wrong.history))

    # The cancellation goes through the whole wrapper chain, not only the raw
    # law: with the Maxwell closure active, eta_eff = 1/(1/eta + 1/(G dt)) and
    # the elastic force term both see the normalised viscosity, so scaling the
    # law and eta_ref together leaves the viscoelastic run untouched too.
    G = 10.0
    ve_base = g257_silent(() -> g257_run(g257_params_comp(G=G)))
    ve_inv = g257_silent(() -> g257_run(g257_params_comp(Ad=G257_AD / c,
                                                         As=G257_AS / c,
                                                         eta_ref=c, G=G)))
    @test ve_base.txx !== nothing && ve_inv.txx !== nothing
    @test g257_relhist(ve_base.history, ve_inv.history) < 1.0e-12
    @test l2rel(ve_base.theta, ve_inv.theta) < 1.0e-12
    @test l2rel(ve_base.ux, ve_inv.ux) < 1.0e-10
    @test l2rel(ve_base.txx, ve_inv.txx) < 1.0e-12
    @test l2rel(ve_base.txy, ve_inv.txy) < 1.0e-12
    @printf("G257 (c) viscoelastic arm: hist %.3e, txx %.3e (G=%.1f)\n",
            g257_relhist(ve_base.history, ve_inv.history),
            l2rel(ve_base.txx, ve_inv.txx), G)

    # ================================================================ (d) ===
    # The run report: two records when eta_ref != 1, on the reported partition,
    # with numbers that are a direct recomputation from the state the run used.
    pr_rep = Params(Ra=1.0e3, H=0.0, eta_law=:expTz, eta_b=log(1.0e4),
                    eta_c=0.0, eta_ref=5.0, bc=:free)
    caps = Any[]
    cb = s -> (push!(caps, (step=s.step, theta=copy(s.theta), sq=s.sq)); nothing)
    logs_rep, res_rep = Test.collect_test_logs(min_level=Logging.Info) do
        g257_run(pr_rep; callback=cb, eta_ref_scale=G257_SCALE)
    end
    msgs = g257_report_msgs(logs_rep)
    @test length(msgs) == 2                       # first solve + end of run
    @test occursin("first flow solve, step 1", msgs[1])
    @test occursin("end of run, step 2", msgs[2])
    @test res_rep.steps == 2
    # the recorded convention itself
    for m in msgs
        @test occursin(@sprintf("Ra = %.6e", pr_rep.Ra), m)
        @test occursin(@sprintf("eta_ref = %.6e", pr_rep.eta_ref), m)
        @test occursin("Ra = rho g alpha dT d^3/(kappa eta_ref)", m)
        @test occursin("eta/eta_ref", m)
        @test occursin("Ra_interior", m)
        @test occursin("interior band 0.25 <= z <= 0.75", m)
        @test occursin("volume mean", m)
        @test occursin("middle half", m)
        @test occursin("Quote Ra together with eta_ref", m)
        @test occursin("Pa s", m)
    end
    eta0 = G257_SCALE.rho * G257_SCALE.g * G257_SCALE.alpha *
           G257_SCALE.dT * G257_SCALE.d^3 / (pr_rep.Ra * G257_SCALE.kappa)
    unit = eta0 / pr_rep.eta_ref
    @test occursin(@sprintf("%.6e", eta0), msgs[1])
    @test occursin(@sprintf("%.6e", unit), msgs[1])
    # the first record is measured on the closure of the first flow solve: the
    # start-of-step theta (theta0) and the normalised law.  The gate rebuilds
    # theta0 from the same tree and seed (there is no AMR, so the spaces match).
    sq = res_rep.ss.su
    leaves = Set(res_rep.qt.leaves)
    qt_rep = g257_tree()
    theta0_rep = [g257_seed(x, y) for (x, y) in make_semstokes(qt_rep, 2).su.coords]
    eta1 = (x, y) -> semquad_eta_at(sq, theta0_rep, x, y, pr_rep,
                                    leaves) / pr_rep.eta_ref
    st1 = QuaST._eta_ref_field_stats(sq, res_rep.qt, eta1, pr_rep)
    @test st1.max > st1.min                          # not a constant field
    @test isapprox(st1.interior, QuaST._eta_interior_band_mean(sq, res_rep.qt,
                                                               eta1);
                   rtol=1.0e-14)
    for (field, val) in (("min", st1.min), ("max", st1.max),
                         ("contrast", st1.contrast),
                         ("interior", st1.interior))
        @test occursin(@sprintf("%.6e", val), msgs[1])
    end
    @test occursin(@sprintf("%.6e", pr_rep.Ra / st1.interior), msgs[1])
    @test occursin(@sprintf("%.6e", st1.interior * unit), msgs[1])
    # the end record is measured on the closure of the last solve, i.e. on the
    # theta at the start of step 2 (the callback state of step 1)
    th_last = caps[end - 1].theta
    eta2 = (x, y) -> semquad_eta_at(sq, th_last, x, y, pr_rep,
                                    leaves) / pr_rep.eta_ref
    st2 = QuaST._eta_ref_field_stats(sq, res_rep.qt, eta2, pr_rep)
    for (field, val) in (("min", st2.min), ("max", st2.max),
                         ("contrast", st2.contrast),
                         ("interior", st2.interior))
        @test occursin(@sprintf("%.6e", val), msgs[2])
    end
    @test occursin(@sprintf("%.6e", pr_rep.Ra / st2.interior), msgs[2])
    # the interior mean is between the field's min and max (a real mean)
    @test st1.min <= st1.interior <= st1.max
    @test st2.min <= st2.interior <= st2.max
    @printf("G257 (d) report fields: eta in [%.6e, %.6e], eta_interior %.6e, Ra_interior %.6e, eta0 %.6e Pa s\n",
            st1.min, st1.max, st1.interior, pr_rep.Ra / st1.interior, eta0)

    # the measurement helper itself, on closures with known volume means
    q = g257_tree()
    ssq = make_semstokes(q, 2)
    @test isapprox(QuaST._eta_interior_band_mean(ssq.su, q, (x, y) -> 2.0),
                   2.0; rtol=1.0e-14)
    @test isapprox(QuaST._eta_interior_band_mean(ssq.su, q, (x, y) -> y),
                   0.5; rtol=1.0e-14)       # band bounds align with leaf faces
    pr_const = Params(eta_law=:expT, deta_T=2.0)
    cst = QuaST._eta_ref_field_stats(ssq.su, q, (x, y) -> 3.0, pr_const)
    @test cst.min == 3.0 && cst.max == 3.0 && cst.contrast == 1.0
    @test isapprox(cst.interior, 3.0; rtol=1.0e-14)
    @test cst.zlo == 0.25 && cst.zhi == 0.75
    # the report does not appear at the default convention (no numbers change)
    pr_def_rep = Params(Ra=1.0e3, H=0.0, eta_law=:expTz, eta_b=log(1.0e4),
                        eta_c=0.0, eta_ref=1.0, bc=:free)
    logs_def, _ = Test.collect_test_logs(min_level=Logging.Info) do
        g257_run(pr_def_rep; eta_ref_scale=G257_SCALE)
    end
    @test isempty(g257_report_msgs(logs_def))

    # ================================================================ (e) ===
    # The paths that do not apply the normalisation refuse the combination.
    qt = g257_tree()
    ss = make_semstokes(qt, 2)
    th0 = zeros(ss.su.N)
    pr_n = Params(Ra=1.0e3, H=0.0, eta_law=:expT, deta_T=1.0e2,
                  eta_ref=2.0, bc=:free)
    err = try
        run_semquad_simulation(qt, pr_n, th0; p=2, maxsteps=1,
                               linear=:direct, pic_mode=:newton)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("eta_ref", err.msg)
    @test occursin("newton", lowercase(err.msg))
    # The steady-state hand-off was removed with the steady solver; the Newton
    # per-step path is the only remaining one that bypasses the normalisation.
    # a malformed scale set is rejected before any work
    @test_throws ArgumentError run_semquad_simulation(qt, pr_n, th0; maxsteps=1,
        linear=:direct,
        eta_ref_scale=(rho=1.0, g=1.0, alpha=1.0, dT=1.0, d=1.0))
    @test_throws ArgumentError run_semquad_simulation(qt, pr_n, th0; maxsteps=1,
        linear=:direct,
        eta_ref_scale=(rho=1.0, g=1.0, alpha=1.0, dT=1.0, d=1.0, kappa=-1.0))
end
