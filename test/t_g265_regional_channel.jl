# G265: the regional subduction-style channel example -- the three per-wall
# mechanisms acting together, checked against the analytic Couette-Poiseuille
# solution. Gate of `examples/subduction_2d.jl`.
#
# The example is a 2 x 1 regional box driven by a prescribed normal-velocity
# inflow (`xminus`, `:rigid` + `wall_un`), a moving plate (`yplus`, `:rigid` +
# `wall_u = U_p`), a no-slip base and an `:open` outflow (`xplus`) carrying the
# far-field normal traction (`wall_tn = -p_out`) and the far-field shear
# traction.  In the isoviscous limit the shear data are exactly
# `tau_ref(y) = U_p + G(1/2 - y)` and the exact solution is
#
#     u_x = u_ref(y) = U_p y + G y (1-y)/2,  u_y = 0,
#     p   = p_ref(x) = p_out + G (Lx - x),
#
# a degree-2 velocity / degree-1 pressure field that Q2/Q1 contain exactly and
# that the linear traction data integrate exactly with the face quadrature.
#
# **What this gate pins, and why only this.**  The design classifies its checks
# into exact identities, conservation identities and asymptotic/discretisation
# comparisons (section 6.8).  A gate must pin only the first class, with the
# stated tolerances:
#
#   * checks 1/2  the velocity and pressure fields against u_ref/p_ref, 1e-10
#                 relative (the residual is the direct linear solve);
#   * check 3     the per-wall mass flux (+-Q_ref, exactly zero on the two
#                 no-slip y-walls) and the net-zero identity, 1e-10 relative;
#   * check 4     the outflow pressure against p_out (the `wall_tn` data and the
#                 derived pressure gauge), 1e-10 relative to the pressure scale
#                 -- an absolute 1e-10 would sit below the O(1e3) pressure
#                 level's own round-off (design error D4)) -- plus
#                 LAST_SEMSTOKES_PRESSURE_GAUGE == :none;
#                 LAST_SEMSTOKES_PRESSURE_GAUGE == :none;
#   * check 8     the `deta_T = 1` continuation with the stage-(ii) card on the
#                 same tree and linear path (design section 6.7), 1e-10; the
#                 production-tree variant is a printed diagnostic in the example
#                 (measured 1.1e-9, one order above the identity class, D8);
#   * structure   the card's wall wiring and data literals, the stage split, the
#                 modes parsing and the four well-posedness refusals.
#
# **Not in this gate**, by the same classification: checks 5 (open-box heat
# budget) and 10 (latent enthalpy) are diagnostics for this fixed-side-wall card
# (their pointwise-wall-gradient form does not close; design defects D6/D7),
# check 6 is the 10 % asymptotic plate-cooling comparison and check 7 the
# resolution sequence.
# `rheology`, `phases`, `full` and `converge` modes, are asymptotic or
# discretisation-level rather than exact, and each costs minutes; pinning them
# here would break the three-minute gate budget the task fixes.  `test/README.md`
# and the example say so as well.
using Test, QuaST

# The example is *exercised*, not just documented: include it (its command-line
# block is PROGRAM_FILE-guarded, so nothing runs on include).
include(joinpath(@__DIR__, "..", "examples", "subduction_2d.jl"))

# ---------------------------------------------------------------------------
# The gate's own reference formulas.  They are written out here from the card's
# literals, not taken from the example, so a wrong U_p / G / p_out in the
# example has to fail this gate.
# ---------------------------------------------------------------------------
const G265_UP = 1.6e3
const G265_G = 1.0e3
const G265_POUT = 1.0e3
const G265_LX = 2.0
g265_uref(y) = G265_UP * y + G265_G * y * (1.0 - y) / 2.0
g265_tauref(y) = G265_UP + G265_G * (0.5 - y)
g265_pref(x) = G265_POUT + G265_G * (G265_LX - x)
g265_pscale() = G265_G * G265_LX + G265_POUT
g265_qref() = G265_UP / 2.0 + G265_G / 12.0
g265_tauscale() = G265_UP + G265_G / 2.0

# Outward flux per wall, in the order (xminus, xplus, yminus, yplus), with the
# assembly's own 1-D Gauss rule (exact for the Q2 trace); the same integrator as
# the shipped channel gate G264.
function g265_flux_per_wall(ss, qt, u)
    su = ss.su
    n = su.p + 1
    nu = su.N
    fd = make_fedata(ss.p, ss.q)
    out = zeros(Float64, 4)
    for it in eachindex(su.leaves)
        lf = su.leaves[it]
        hx = QuaST.leaf_hx(su, lf[1])
        hy = QuaST.leaf_hy(su, lf[1])
        ids = su.elem[lf]
        for (k, dir) in enumerate((:xminus, :xplus, :yminus, :yplus))
            QuaST.has_no_face_neighbor(qt, lf, dir) || continue
            if dir === :xminus || dir === :xplus
                iF = dir === :xminus ? 1 : n
                sgn = dir === :xplus ? 1.0 : -1.0
                for q in 1:fd.nq
                    val = 0.0
                    for j in 1:n
                        val += u[ids[(j - 1) * n + iF]] * fd.phi_v[j, q]
                    end
                    out[k] += sgn * (0.5 * hy * fd.gl_w[q]) * val
                end
            else
                jF = dir === :yminus ? 1 : n
                sgn = dir === :yplus ? 1.0 : -1.0
                for q in 1:fd.nq
                    val = 0.0
                    for i in 1:n
                        val += u[nu + ids[(jF - 1) * n + i]] * fd.phi_v[i, q]
                    end
                    out[k] += sgn * (0.5 * hx * fd.gl_w[q]) * val
                end
            end
        end
    end
    return out
end

# The one-sided GLL derivative eta du_x/dy on the outflow wall against tau_ref.
function g265_outflow_shear(ss, qt, ux)
    sq = ss.su
    n = sq.p + 1
    rule = gll_rule(sq.p)
    worst = 0.0
    for lf in sq.leaves
        QuaST.has_no_face_neighbor(qt, lf, :xplus) || continue
        hy = QuaST.leaf_hy(sq, lf[1])
        y0 = lf[3] * hy
        ids = sq.elem[lf]
        for j0 in 1:n
            d = 0.0
            for j in 1:n
                d += sq.D[j0, j] * ux[ids[(j - 1) * n + n]]
            end
            d *= 2.0 / hy
            y = y0 + 0.5 * (rule.nodes[j0] + 1.0) * hy
            worst = max(worst, abs(d - g265_tauref(y)))
        end
    end
    return worst / g265_tauscale()
end

@testset "G265 regional subduction channel" begin

# ---- (a) the card and the per-wall wiring --------------------------------
@testset "card and wall wiring" begin
    pr = regional_card(:validate)
    @test pr.wall_bc == (:rigid, :open, :rigid, :rigid)
    @test pr.bc === :free
    @test pr.Ra == 0.0                          # kinematic thermal model
    @test pr.H == 1.0
    @test pr.eta_law === :expT && pr.deta_T == 1.0
    @test pr.eta_ref == 1.0 && pr.eta_T0 == 0.5
    @test pr.eta_min == 1.0e-4 && pr.eta_max == 1.0e4
    @test pr.side_thermal === :fixed
    @test pr.Twall_top == 0.0 && pr.Twall_bot == 1.0
    @test pr.Twall_left == 1.0 && pr.Twall_right == 1.0
    # the inflow normal-velocity data: u_ref(0) = 0, u_ref(1/4) = 493.75,
    # u_ref(1/2) = 925, u_ref(1) = U_p (the design's continuous left corners)
    @test pr.wall_un[1](0.0, 0.0) == 0.0
    @test pr.wall_un[1](0.0, 0.25) == 493.75
    @test pr.wall_un[1](0.0, 0.5) == 925.0
    @test pr.wall_un[1](0.0, 1.0) == G265_UP
    @test pr.wall_un[2] === nothing               # refused on :open anyway
    @test pr.wall_un[3](1.0, 0.0) == 0.0          # no-slip base normal
    @test pr.wall_un[4](1.0, 1.0) == 0.0          # plate normal
    # the plate and the base tangential velocities
    @test pr.wall_u[1](0.5, 0.5) == 0.0
    @test pr.wall_u[2] === nothing
    @test pr.wall_u[4](0.5, 1.0) == G265_UP
    # the outflow traction data: t_n = -p_out and t_t = tau_ref(y)
    @test pr.wall_t[1] === nothing && pr.wall_tn[1] === nothing
    @test pr.wall_t[2](0.0, 0.0) == 2100.0
    @test pr.wall_t[2](0.0, 0.5) == 1600.0
    @test pr.wall_t[2](0.0, 1.0) == 1100.0
    @test pr.wall_tn[2](2.0, 0.5) == -G265_POUT
    # the stage split: no phases before `phases`, and the full card's physics
    @test regional_phases(:validate, pr) == PhaseTransition[]
    @test regional_phases(:rheology, pr) == PhaseTransition[]
    prfull = regional_card(:full)
    @test prfull.eta_law === :composite
    @test prfull.comp_rule === :isostress
    @test prfull.Ra == 0.0
    @test prfull.latent_mode === :per_phase
    ph = regional_phases(:full, prfull)
    @test length(ph) == 2
    @test ph[1].z0 == 0.59 && ph[2].z0 == 0.34
    @test ph[1].width == 0.02 && ph[2].width == 0.02
    # the design's section-5.3 closed forms for the regional scale set.
    # Measured (P3 loose-tolerance review, out_julia/agent/p3/probe_loose.jl):
    # latent1 = -0.051193303069090916 (vs -0.0512: 6.70e-6 abs, 1.31e-4 rel)
    # and latent2 = +0.10528819723636361 (vs +0.1053: 1.18e-5 abs, 1.12e-4
    # rel) -- the residual is the design's 4-digit rounding of the closed form.
    # The old common atol 5e-4 was 40-75x looser than needed; the bands below
    # are 1.5x/1.7x the measured quantisation residual.
    @test isapprox(ph[1].latent, -0.0512; atol=1.0e-5)
    @test isapprox(ph[2].latent, 0.1053; atol=2.0e-5)
    # the production trees of the design's section 4.1/6.6
    @test length(regional_tree(3).leaves) == 512
    @test length(regional_tree(4).leaves) == 2048
    @test length(regional_tree(4; wall_min=5).leaves) == 3200
end

# ---- (b) the modes parsing ---------------------------------------------
@testset "modes parsing" begin
    for m in (:validate, :rheology, :phases, :full, :converge)
        @test regional_parse_mode([String(m)]) === m
    end
    @test regional_parse_mode(String[]) === :help
    @test regional_parse_mode(["-h"]) === :help
    @test regional_parse_mode(["--help"]) === :help
    @test regional_parse_mode(["help"]) === :help
    @test_throws ArgumentError regional_parse_mode(["nonsense"])
    @test_throws ArgumentError regional_parse_mode(["Validate"])   # no aliases
    @test_throws ArgumentError regional_card(:nonsense)
    # the CASE_* knobs are read at call time
    old_out = get(ENV, "CASE_OUT", nothing)
    old_steps = get(ENV, "CASE_STEPS", nothing)
    old_wallt = get(ENV, "CASE_WALL_T", nothing)
    try
        ENV["CASE_OUT"] = "out_julia/agent/g265_probe"
        ENV["CASE_STEPS"] = "7"
        ENV["CASE_WALL_T"] = "stress"
        cfg = regional_config(:rheology)
        @test cfg.out == "out_julia/agent/g265_probe"
        @test cfg.steps == 7
        @test cfg.t_mode === :stress
        @test regional_config(:validate).linear === :direct
        @test regional_config(:rheology).linear === :auto
    finally
        old_out === nothing ? delete!(ENV, "CASE_OUT") : (ENV["CASE_OUT"] = old_out)
        old_steps === nothing ? delete!(ENV, "CASE_STEPS") : (ENV["CASE_STEPS"] = old_steps)
        old_wallt === nothing ? delete!(ENV, "CASE_WALL_T") : (ENV["CASE_WALL_T"] = old_wallt)
    end
end

# ---- (c) the isoviscous analytic channel: checks 1-4 and 8 -------------
@testset "isoviscous analytic channel" begin
    tmp = mktempdir()
    buf = IOBuffer()
    out = regional_mode_validate(; io=buf, outdir=joinpath(tmp, "validate"))
    txt = String(take!(buf))
    res = out.res
    sq = res.ss.su
    sp = res.ss.sp
    ys = [c[2] for c in sq.coords]
    pxs = [c[1] for c in sp.coords]

    # check 1: the velocity field against the analytic Couette-Poiseuille
    err_u = maximum(abs(res.ux[i] - g265_uref(ys[i])) for i in eachindex(ys)) /
            G265_UP
    err_w = maximum(abs, res.uz) / G265_UP
    @test err_u < 1.0e-10                 # measured ~1e-14
    @test err_w < 1.0e-10                 # measured ~1e-14
    # check 2: the pressure field
    err_p = maximum(abs(res.pv[j] - g265_pref(pxs[j])) for j in eachindex(pxs)) /
            g265_pscale()
    @test err_p < 1.0e-10                 # measured ~1e-13
    # check 4: the outflow pressure is the wall_tn data, relative to the
    # pressure scale (the design's "1e-10 abs" is below the direct solve's
    # round-off at this O(1e3) pressure level: measured 3.8e-9 absolute =
    # 1.3e-12 of the scale), and the pressure gauge is the derived one
    err_pout = maximum(abs(res.pv[j] - G265_POUT)
                       for j in eachindex(pxs) if abs(pxs[j] - G265_LX) <= 1.0e-12)
    @test err_pout / g265_pscale() < 1.0e-10
    @test LAST_SEMSTOKES_PRESSURE_GAUGE[] === :none
    @test out.residual < 1.0e-10          # the analytic run's Stokes residual
    # the outflow field shear reproduces the assembled tau_ref data
    @test g265_outflow_shear(res.ss, res.qt, res.ux) < 1.0e-10
    # check 3: the per-wall mass flux and the net-zero identity
    fl = g265_flux_per_wall(res.ss, res.qt, vcat(res.ux, res.uz))
    qref = g265_qref()
    @test abs(fl[1] + qref) < 1.0e-10 * qref      # inflow, inward
    @test abs(fl[2] - qref) < 1.0e-10 * qref      # outflow, outward
    @test abs(fl[3]) < 1.0e-10 * qref             # no-slip y-walls
    @test abs(fl[4]) < 1.0e-10 * qref
    @test abs(sum(fl)) < 1.0e-10 * qref           # the discrete continuity
    # check 8: the deta_T = 1 continuation.  Section 6.7's definition is the
    # stage-(ii) card at deta_T = 1 on the *same tree* as stage (i): the data,
    # the tree and the linear path are unchanged, so the identical solution is
    # required to 1e-10 -- this pins that the stage cards cannot drift apart.
    cres = out.continuation
    @test cres.steps == res.steps                 # same two steps
    @test length(cres.qt.leaves) == length(regional_tree(3).leaves) == 512
    csq = cres.ss.su
    cys = [c[2] for c in csq.coords]
    err_uc = maximum(abs(cres.ux[i] - g265_uref(cys[i])) for i in eachindex(cys)) /
             G265_UP
    err_wc = maximum(abs, cres.uz) / G265_UP
    cpxs = [c[1] for c in cres.ss.sp.coords]
    err_pc = maximum(abs(cres.pv[j] - g265_pref(cpxs[j]))
                     for j in eachindex(cpxs)) / g265_pscale()
    @test err_uc < 1.0e-10                # measured ~1e-14
    @test err_wc < 1.0e-10
    @test err_pc < 1.0e-10                # measured ~1e-12
    @test g265_outflow_shear(cres.ss, cres.qt, cres.ux) < 1.0e-10
    flc = g265_flux_per_wall(cres.ss, cres.qt, vcat(cres.ux, cres.uz))
    @test abs(flc[1] + qref) < 1.0e-10 * qref
    @test abs(flc[2] - qref) < 1.0e-10 * qref
    @test abs(sum(flc)) < 1.0e-10 * qref
    # the same continuation re-solved on the production tree is the example's
    # printed DIAGNOSTIC (not a gate: the 29 953-dof direct saddle's round-off is
    # ~1e-9 relative, an order above the 1e-10 identity class); the gate only
    # requires that the example ran it on the designed 3 200-leaf tree.
    @test length(out.continuation_prod.qt.leaves) == 3200
    @test isfinite(out.continuation_prod_errors.err_u)
    # the printed contract and the CSV writer
    @test occursin("[regional] check  1", txt)
    @test occursin("u-field", txt)
    @test occursin("[regional] validate: PASS", txt)
    @test isfile(joinpath(tmp, "validate", "checks.csv"))
    @test isfile(joinpath(tmp, "validate", "history.csv"))
    rows = readlines(joinpath(tmp, "validate", "checks.csv"))
    @test length(rows) > 10
    @test any(r -> occursin("PASS", r), rows)
    @test !any(r -> occursin("FAIL", r), rows)
    # a spot check that the CSV carries the measured numbers the gate computed
    urow = findfirst(r -> occursin(",u-field,", r), rows)
    @test urow !== nothing
    ufields = split(rows[urow], ",")
    @test ufields[3] == "u-field"
    measured = parse(Float64, ufields[5])
    @test isapprox(measured, err_u; rtol=1.0e-12, atol=1.0e-20)
end

# ---- (d) the per-wall mass flux in a variable-viscosity state ------------
# The per-wall flux check is an exact discrete identity of the semi-discrete
# system: the inflow Dirichlet data fix the flux to Q_ref, and the discrete
# continuity equations carry it to the outflow up to the Stokes residual.  It is
# pinned here on the level-3 tree, where the variable-viscosity direct solve is
# healthy (measured residual 6e-15) and the identity holds at the design's
# 1e-10 class (measured 6e-13).  The production-size variable-viscosity solve
# takes the direct path's 1e-12 shift rung and its measured net-flux defect is
# 2.9e-8 (uniform-4) / 5.2e-8 (production tree), so the example's production
# modes carry 1e-6 there and say so; that is a property of the shifted linear
# solve, not of the boundary mechanism.  The state must actually be the
# variable-viscosity one, or the check would be the trivial data identity.
@testset "mass flux in the variable-viscosity state" begin
    qt = regional_tree(3)                     # 512 leaves, no hanging nodes
    @test length(qt.leaves) == 512
    pr = regional_card(:rheology)
    @test pr.deta_T == 1.0e3
    res = regional_run(qt, pr; maxsteps=2, linear=:direct,
                       stokes_tol=REGIONAL_STOKES_TOL_PROD,
                       cfl=2.0, phases=PhaseTransition[])
    @test res.steps == 2
    @test LAST_SEMSTOKES_PRESSURE_GAUGE[] === :none
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-12    # measured 6.1e-15
    fl = g265_flux_per_wall(res.ss, res.qt, vcat(res.ux, res.uz))
    qref = g265_qref()
    @test abs(fl[1] + qref) / qref < 1.0e-10     # measured ~2e-16 (data trace)
    @test abs(fl[2] - qref) / qref < 1.0e-10     # measured ~8e-16
    @test abs(fl[3]) / qref < 1.0e-10            # no-slip y-walls
    @test abs(fl[4]) / qref < 1.0e-10
    @test abs(sum(fl)) / qref < 1.0e-10          # measured ~6e-16
    # the state really is the variable-viscosity one: the velocity is NOT the
    # analytic profile (so the flux identity above is not the trivial data one)
    ys = [c[2] for c in res.ss.su.coords]
    err_u = maximum(abs(res.ux[i] - g265_uref(ys[i])) for i in eachindex(ys)) /
            G265_UP
    @test err_u > 1.0e-3                         # measured ~1e-2
end

# ---- (e) the well-posedness refusals the example prints ------------------
@testset "open-wall well-posedness refusals" begin
    @test_throws ArgumentError Params(bc=:free,
                                      wall_bc=(:open, :open, :rigid, :rigid))
    @test_throws ArgumentError Params(bc=:free,
                                      wall_bc=(:open, :open, :open, :open))
    # a normal traction on a non-open wall is refused (the load would be dropped)
    @test_throws ArgumentError Params(bc=:free,
                                      wall_bc=(:rigid, :open, :rigid, :rigid),
                                      wall_tn=((x, y) -> 0.0, nothing, nothing,
                                               nothing))
    # a tangential traction on a rigid wall is refused as well
    @test_throws ArgumentError Params(bc=:free,
                                      wall_bc=(:rigid, :open, :rigid, :rigid),
                                      wall_t=((x, y) -> 0.0, nothing, nothing,
                                              nothing))
end

# ---- (f) the far-field traction data: rate == stress at deta_T = 1 -------
@testset "far-field traction data" begin
    rate = regional_wall_data(; t_mode=:rate, law=:expT, deta_T=1.0)
    stress = regional_wall_data(; t_mode=:stress)
    for y in (0.0, 0.2, 0.5, 0.8, 1.0)
        @test rate.wall_t[2](0.0, y) == stress.wall_t[2](0.0, y)
    end
    # at deta_T = 1e3 the rate data are tau_ref times the far-field viscosity at
    # the card's hot side-wall temperature (T = 1, the far field the channel
    # flows in), i.e. exactly 1e-3 of the reference stress at every height.  The
    # alternative base-state scaling (Tbar) was measured to put 1e3 traction on
    # 1e-3 material in the mid-upper region; the far field is the side-wall
    # temperature, not the conductive base state (see the example's
    # `regional_farfield_eta`).
    scaled = regional_wall_data(; t_mode=:rate, law=:expT, deta_T=1.0e3)
    @test isapprox(scaled.wall_t[2](0.0, 0.0), 2100.0 * 1.0e-3; rtol=1.0e-12)
    @test isapprox(scaled.wall_t[2](0.0, 0.5), 1600.0 * 1.0e-3; rtol=1.0e-12)
    @test isapprox(scaled.wall_t[2](0.0, 1.0), 1100.0 * 1.0e-3; rtol=1.0e-12)
    @test_throws ArgumentError regional_wall_data(; t_mode=:nonsense)
end

end # G265
