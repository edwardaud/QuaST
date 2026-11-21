# G251: the variable-`dt` Maxwell update is consistent -- no fixed elastic step.
#
# The driver's `dt` is adaptive (the CFL + growth rule, `run.jl`'s `cfl>0`
# block), and `_advance_maxwell_stress!` feeds the actual step into the
# backward-Euler kernel.  ASPECT's `viscoelastic` model offers `Use fixed
# elastic time step`, `Fixed elastic time step` and `Use stress averaging` for
# runs whose numerical step is far from the relaxation timescale (seismic
# cycles, ...).  The question this gate answers with evidence is whether our
# variable-`dt` handling needs such a parameter to be trustworthy.
#
# The hypothesis is that it does not, because backward Euler is consistent on a
# variable step sequence.  One step of
#
#     dtau/dt = 2 G ehat - tau / tau_M,        tau_M = eta / G,
#
# has local truncation error `O(h^2)` for any step `h`, so over steps
# `h_1..h_n` summing to `T` the global error is `O(max h)`: refining the
# sequence while keeping the total time fixed must halve the error.  The
# composed driver update
#
#     tau^{n+1} = maxwell_stress_step(corotate(tau^n, w, h_n), ehat, h_n, eta, G)
#
# has the same property: the linearised rotation `1 + h ad_W` is itself `O(h)`
# consistent with the ODE's `(W tau - tau W)` term.  The analytic targets are
# `maxwell_stress_exact(T, ...)` for the spin-free kernel and
# `maxwell_stress_exact_corotational(T, ...)` when the spin is present.
#
# Two levels of evidence, because "the kernel converges" and "the driver uses
# that kernel on the actual step" are different claims.
#
#   (a) Kernel.  Three step-size profiles that all sum to a *fixed* total time:
#       uniform, a 10:1 alternating one whose consecutive steps differ by a
#       factor 10, and a fixed-seed random one drawn uniformly from a factor-10
#       band (`[0.2, 2.0]` relative weights, seed 251).  Each is integrated at
#       three nested resolutions.  Resolution level `l` repeats the base
#       relative profile `2^l` times at half the scale, so every step of level
#       `l+1` is exactly half a level-`l` step and the total time is unchanged:
#       this is a first-order refinement of the *same* variable-step scheme,
#       not a fresh random draw.  The assertions are:
#         * `1.5 < ratio < 2.5` for every profile/variant at both refinements
#           (first order, the G244 convention);
#         * the finest relative error is below `5e-3` of the analytic value;
#         * the three profiles agree with *each other* to `< 2e-3` at the
#           finest resolution, and their spread itself halves under refinement,
#           i.e. they converge to the same value and not merely each to
#           something;
#         * at the coarsest resolution the profiles genuinely differ
#           (`> 1e-3`), so the agreement is not an artifact of steps that are
#           all the same.
#       The co-rotational variant is not vacuous: it tracks the co-rotational
#       curve much better than the spin-free one, which a term-free update
#       could not do.
#
#   (b) Driver.  A short run whose `dt` is *forced* to change every step
#       (`dt0 -> dt_growth = 2` per step, capped by `dtmax`), so the reported
#       `dt` sequence is non-uniform by construction.  The `callback` captures
#       the step's `dt` and the step's solved velocity `(ux, uz)`; replaying
#       exactly that `(h_n, u_n)` sequence from zero through
#       `_advance_maxwell_stress!` reproduces the driver's returned
#       `txx/tyy/txy` **bit for bit**.  This is the right statement because it
#       isolates the only thing in question -- does the update see the actual
#       per-step `dt` and the step's velocity, composed once per step? -- from
#       the spatial and coupling discretisations, which a comparison against a
#       spatially varying analytic stress could not separate.  The intermediate
#       history is pinned too: a 6-step run's returned stress equals the 6-step
#       replay prefix bit for bit.  The negative controls are the same replay
#       with (i) a uniform step `T/n`, (ii) the first `dt` repeated and (iii)
#       every step shifted by one position (a `dt_prev` off-by-one); each
#       differs from the driver's stress by `O(10%)` of `tmax`, so the bitwise
#       match is not vacuous.
#
# Tolerances and their sources, stated before they are used:
# * order bounds `1.5..2.5` (the G244 convention): measured `1.988..1.997` for
#   every profile/variant pair; the `O(h^2)` correction is below 0.6%.
# * finest relative error `5e-3`: measured at most `1.95e-3` (co-rotational,
#   10:1 profile).
# * cross-profile agreement `2e-3`: measured at most `8.3e-4` at the finest
#   level.
# * coarsest cross-profile spread `> 1e-3`: measured `3.0e-3` (spin-free) and
#   `3.1e-3` (co-rotational) as the maximum pairwise difference (uniform
#   against 10:1; the random profile sits in between).
# * driver replay: exact equality (measured 0.0).  Negative-control differences
#   measured `3.36e-1` (uniform), `9.26e-1` (first `dt`) and `3.73e-1`
#   (shifted) of `tmax`, asserted against `1e-2`.
using Test, QuaST, Printf, Random

@testset "G251 variable dt" begin
    # ---- (a) kernel: three step-size profiles, two variants ----------------
    # A profile is a cycle of relative weights; level `l` repeats it `2^l` times
    # at scale `2^-l` so the summed time is fixed and every step is exactly
    # halved by one refinement.
    function g251_steps(kind::Symbol, level::Int; T::Float64, nbase::Int=64)
        w = if kind === :uniform
            ones(Float64, nbase)
        elseif kind === :factor10
            [isodd(i) ? 10.0 : 1.0 for i in 1:nbase]
        elseif kind === :random
            rng = MersenneTwister(251)
            0.2 .+ 1.8 .* rand(rng, nbase)   # uniform on [0.2, 2.0]
        else
            error("unknown step profile $kind")
        end
        nrep = 2^level
        scale = T / (nrep * sum(w))
        return [scale * w[mod1(k, nbase)] for k in 1:(nbase * nrep)]
    end
    function g251_free(hs, edot, eta, G)
        tau = 0.0
        for h in hs
            tau = maxwell_stress_step(tau, edot, h, eta, G)
        end
        return tau
    end
    function g251_corot(hs, e1, e2, e3, w, eta, G)
        t1 = t2 = t3 = 0.0
        for h in hs
            r1, r2, r3 = maxwell_stress_corotate(t1, t2, t3, w, h)
            t1 = maxwell_stress_step(r1, e1, h, eta, G)
            t2 = maxwell_stress_step(r2, e2, h, eta, G)
            t3 = maxwell_stress_step(r3, e3, h, eta, G)
        end
        return (t1, t2, t3)
    end

    kinds = (:uniform, :factor10, :random)
    levels = 0:2
    # spin-free card of G244: eta = 100, G = 1e3, tau_M = 0.1, T = tau_M
    eta_f, G_f = 100.0, 1.0e3
    tauM_f = eta_f / G_f
    edot_f = 1.0e-3
    T_f = tauM_f
    # co-rotational card of G246/G250: eta = 2, G = 50, tau_M = 0.04,
    # spin w = 0.3, T = tau_M (the analytic reference is the objective curve)
    eta_c, G_c = 2.0, 50.0
    tauM_c = eta_c / G_c
    e1, e2, e3 = 0.24, -0.18, 0.20
    w0 = 0.30
    T_c = tauM_c

    free_val = zeros(3, 3)                # [profile, level]
    free_err = zeros(3, 3)
    cor_val = zeros(3, 3, 3)              # [profile, level, component]
    cor_err = zeros(3, 3)
    for (ik, kind) in enumerate(kinds), (il, level) in enumerate(levels)
        # spin-free
        hs = g251_steps(kind, level; T=T_f)
        Tsum = sum(hs)
        @test abs(Tsum - T_f) <= 1.0e-14 * T_f
        tau = g251_free(hs, edot_f, eta_f, G_f)
        ref = maxwell_stress_exact(Tsum, edot_f, eta_f, G_f)
        free_val[ik, il] = tau
        free_err[ik, il] = abs(tau - ref) / abs(ref)
        # co-rotational
        hs = g251_steps(kind, level; T=T_c)
        Tsum = sum(hs)
        @test abs(Tsum - T_c) <= 1.0e-14 * T_c
        got = g251_corot(hs, e1, e2, e3, w0, eta_c, G_c)
        ref = maxwell_stress_exact_corotational(Tsum, e1, e2, e3, w0,
                                                eta_c, G_c)
        scale = maximum(abs.(ref))
        cor_val[ik, il, :] = collect(got)
        cor_err[ik, il] = maximum(abs.(collect(got) .- ref)) / scale
    end

    # first order in each profile and variant, at both refinements
    for ik in eachindex(kinds)
        @test 1.5 < free_err[ik, 1] / free_err[ik, 2] < 2.5
        @test 1.5 < free_err[ik, 2] / free_err[ik, 3] < 2.5
        @test free_err[ik, 3] < 5.0e-3
        @test 1.5 < cor_err[ik, 1] / cor_err[ik, 2] < 2.5
        @test 1.5 < cor_err[ik, 2] / cor_err[ik, 3] < 2.5
        @test cor_err[ik, 3] < 5.0e-3
    end

    # the three profiles converge to the *same* analytic value: the pairwise
    # spread at each level (maximum spread, so the random profile cannot hide a
    # disagreement between the two deterministic ones) shrinks at first order
    free_spread = zeros(3)
    cor_spread = zeros(3)
    for il in eachindex(levels)
        free_spread[il] = maximum(abs.(free_val[:, il]) .-
                                  minimum(free_val[:, il])) / abs(maxwell_stress_exact(
                                      T_f, edot_f, eta_f, G_f))
        cor_spread[il] = maximum(maximum(abs.(cor_val[ia, il, :] .-
                                             cor_val[ib, il, :]))
                                 for ia in 1:3 for ib in 1:3) /
                        maximum(abs.(maxwell_stress_exact_corotational(
                            T_c, e1, e2, e3, w0, eta_c, G_c)))
    end
    @test free_spread[1] > 1.0e-3       # measured 3.0e-3: profiles differ
    @test cor_spread[1] > 1.0e-3        # measured 3.1e-3
    @test 1.5 < free_spread[1] / free_spread[2] < 2.5
    @test 1.5 < free_spread[2] / free_spread[3] < 2.5
    @test free_spread[3] < 2.0e-3       # measured 7.6e-4
    @test 1.5 < cor_spread[1] / cor_spread[2] < 2.5
    @test 1.5 < cor_spread[2] / cor_spread[3] < 2.5
    @test cor_spread[3] < 2.0e-3        # measured 7.8e-4

    # the spin is really in the co-rotational run: the objective reference is
    # measurably off the spin-free curve, and the discrete run tracks the
    # objective one several times better than it tracks the spin-free one.
    cor_ref = maxwell_stress_exact_corotational(T_c, e1, e2, e3, w0, eta_c, G_c)
    cor_scale = maximum(abs.(cor_ref))
    spinfree_ref = (maxwell_stress_exact(T_c, e1, eta_c, G_c),
                    maxwell_stress_exact(T_c, e2, eta_c, G_c),
                    maxwell_stress_exact(T_c, e3, eta_c, G_c))
    spin_effect = maximum(abs.(cor_ref .- spinfree_ref)) / cor_scale
    @test spin_effect > 5.0e-3          # measured 7.0e-3
    for ik in eachindex(kinds)
        @test cor_err[ik, 3] < 0.5 * spin_effect
    end

    # report: the two refined resolutions and the observed order per profile
    @printf("G251 (a) kernel, T = tau_M (spin-free eta=%.0f G=%.0f; co-rotational eta=%.0f G=%.0f w=%.1f)\n",
            eta_f, G_f, eta_c, G_c, w0)
    for (ik, kind) in enumerate(kinds)
        @printf("  %-8s spin-free err: %.3e (64 steps) -> %.3e (128) -> %.3e (256), order %.3f / %.3f, tau=%.15f\n",
                kind, free_err[ik, 1], free_err[ik, 2], free_err[ik, 3],
                free_err[ik, 1] / free_err[ik, 2],
                free_err[ik, 2] / free_err[ik, 3], free_val[ik, 3])
        @printf("  %-8s co-rot    err: %.3e (64 steps) -> %.3e (128) -> %.3e (256), order %.3f / %.3f\n",
                kind, cor_err[ik, 1], cor_err[ik, 2], cor_err[ik, 3],
                cor_err[ik, 1] / cor_err[ik, 2],
                cor_err[ik, 2] / cor_err[ik, 3])
    end
    @printf("G251 (a): spin-free analytic = %.15f ; co-rotational analytic = (%.12f, %.12f, %.12f) ; profile spread -> %.1e / %.1e ; spin effect on reference = %.1e\n",
            maxwell_stress_exact(T_f, edot_f, eta_f, G_f), cor_ref...,
            free_spread[3], cor_spread[3], spin_effect)

    # ---- (b) driver: a forced-varying dt, replayed through the kernel ------
    # The card is G247's steady lid-driven cavity (Ra = 0, a temperature-
    # independent viscosity, the top wall tangential velocity 1): the flow and
    # the viscosity are nonzero while the temperature does not feed back, so
    # the stress history is driven by a genuine strain rate and spin.  The
    # `eta_multiplier` only gives the viscosity a spatial shape.
    function g251_tree(nref::Int)
        qt = Quadtree(2)
        for _ in 1:nref, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        return qt
    end
    eta_mult = (x, y) -> 1.0 + 0.5 * sinpi(x) * sinpi(y)
    G_drv = 10.0
    function g251_driver_run(; nsteps::Int)
        qt = g251_tree(2)
        ss = make_semstokes(qt, 2)
        N = ss.su.N
        pr = Params(; Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0,
                    eta_multiplier=eta_mult, shear_modulus=G_drv, bc=:free,
                    wall_bc=(:free, :free, :free, :rigid),
                    wall_u=(nothing, nothing, nothing, (x, y) -> 1.0))
        # (`deta_T = 1` makes :expT exactly eta = 1 for every T, so the material
        # viscosity is `eta_mult` alone and can be reconstructed exactly.)
        dts = Float64[]
        us = Vector{Vector{Float64}}()
        zs = Vector{Vector{Float64}}()
        cb = s -> begin
            push!(dts, s.dt)
            push!(us, copy(s.ux))
            push!(zs, copy(s.uz))
            nothing
        end
        r = run_semquad_simulation(qt, pr, zeros(Float64, N); p=2,
                                   maxsteps=nsteps, tstop=1.0e9, dt0=1.0e-3,
                                   dtmax=0.3, dtmin=1.0e-30, dt_growth=2.0,
                                   cfl=1.0e6, linear=:direct, ev=0.0,
                                   maxpic=1, callback=cb)
        return (r=r, dts=dts, us=us, zs=zs, pr=pr, sq=ss.su, N=N)
    end
    function g251_replay(dts, us, zs, eta_nodes, G, sq)
        tx = zeros(Float64, length(eta_nodes))
        ty = zeros(Float64, length(eta_nodes))
        tz = zeros(Float64, length(eta_nodes))
        for k in eachindex(dts)
            QuaST._advance_maxwell_stress!(tx, ty, tz, sq, us[k], zs[k],
                                           dts[k], eta_nodes, G)
        end
        return tx, ty, tz
    end
    function g251_contrast(ax, ay, az, bx, by, bz)
        m = max(maximum(abs.(ax .- bx)), maximum(abs.(ay .- by)),
                maximum(abs.(az .- bz)))
        return m / max(maximum(abs, bx), maximum(abs, by),
                       maximum(abs, bz))
    end

    run = g251_driver_run(; nsteps=10)
    @test length(run.r.qt.leaves) == 16   # not one leaf (HANDOVER section 8)
    @test run.r.txx !== nothing && run.r.tyy !== nothing &&
          run.r.txy !== nothing
    # the step really is forced to change every step: dt0, then exactly 2x per
    # step while below dtmax, and the last step is the dtmax clamp
    @test length(run.dts) == 10
    @test run.dts[1] == 1.0e-3
    @test run.dts[2:9] == 2.0 .* run.dts[1:8]
    @test run.dts[10] == 0.3
    @test all(diff(run.dts) .> 0.0)
    @test maximum(run.dts) / minimum(run.dts) > 100.0
    # the total time is the sum of the steps the update saw, and the history
    # column records the same sequence
    tseq = 0.0
    for h in run.dts
        tseq += h
    end
    @test run.r.t == tseq
    @test [row[2] for row in run.r.history] == run.dts

    # the material (unsoftened) nodal viscosity, exactly as `_nodal_eta` samples
    # `eta_base_probe` for this card: eta_law=:expT with deta_T=1 is 1 for every T
    sq = run.sq
    leaves = Set(sq.leaves)
    th = zeros(Float64, sq.N)
    eta_nodes = [QuaST.semquad_eta_at(sq, th, x, y, run.pr, leaves) *
                 eta_mult(x, y) for (x, y) in sq.coords]
    @test minimum(eta_nodes) < maximum(eta_nodes)   # spatially varying eta

    # the driver's returned stress is exactly the composition of the per-step
    # updates, in order, with the step's actual dt and solved velocity
    tx, ty, tz = g251_replay(run.dts, run.us, run.zs, eta_nodes, G_drv, sq)
    tmax = max(maximum(abs, run.r.txx), maximum(abs, run.r.tyy),
               maximum(abs, run.r.txy))
    @test tmax > 1.0                     # measured 9.1: not a 0 = 0 comparison
    @test tx == run.r.txx && ty == run.r.tyy && tz == run.r.txy

    # the intermediate stress history is pinned too: a 6-step run is the
    # bitwise prefix of the 10-step run, and the replay prefix reproduces it
    run6 = g251_driver_run(; nsteps=6)
    @test run6.dts == run.dts[1:6]
    @test all(run6.us[k] == run.us[k] for k in 1:6)
    @test all(run6.zs[k] == run.zs[k] for k in 1:6)
    @test maximum(abs, run6.r.txy) > 1.0
    tx6, ty6, tz6 = g251_replay(run.dts[1:6], run.us[1:6], run.zs[1:6],
                                eta_nodes, G_drv, sq)
    @test tx6 == run6.r.txx && ty6 == run6.r.tyy && tz6 == run6.r.txy

    # negative controls: the bitwise match is not vacuous, the step sequence is
    # what enters the answer
    n = length(run.dts)
    txu, tyu, tzu = g251_replay(fill(tseq / n, n), run.us, run.zs, eta_nodes,
                                G_drv, sq)
    d_uniform = g251_contrast(txu, tyu, tzu, run.r.txx, run.r.tyy, run.r.txy)
    txf, tyf, tzf = g251_replay(fill(run.dts[1], n), run.us, run.zs,
                                eta_nodes, G_drv, sq)
    d_first = g251_contrast(txf, tyf, tzf, run.r.txx, run.r.tyy, run.r.txy)
    dts_shift = vcat(run.dts[1], run.dts[1:(end - 1)])
    txs, tys, tzs = g251_replay(dts_shift, run.us, run.zs, eta_nodes, G_drv, sq)
    d_shift = g251_contrast(txs, tys, tzs, run.r.txx, run.r.tyy, run.r.txy)
    @test d_uniform > 1.0e-2             # measured 3.36e-1
    @test d_first > 1.0e-2               # measured 9.26e-1
    @test d_shift > 1.0e-2               # measured 3.73e-1
    @printf("G251 (b) driver dt = [%s] (ratio %.0f, dtmax-clamped last step)\n",
            join([@sprintf("%.2e", h) for h in run.dts], " "),
            maximum(run.dts) / minimum(run.dts))
    @printf("G251 (b): replay matches driver stress bitwise (tmax=%.3e, 10 steps and 6-step prefix); contrasts uniform dt=%.2e first dt=%.2e shifted dt=%.2e (tol 1e-2)\n",
            tmax, d_uniform, d_first, d_shift)
end
