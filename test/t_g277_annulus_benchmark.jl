# G277 — the published 2-D annulus benchmark harness (stage 0.5c2).
#
# What this gate pins (the *cheap* part of the benchmark; no multi-hour run):
#
#   (a) the benchmark card wiring: Ra/H/eta law (isoviscous), free-slip ring at
#       the published radii with the gap as the length unit, the paper initial
#       condition, the resolution -> (level, cells_y) mapping and the DOF
#       arithmetic against the space actually built;
#   (b) the steady-state criterion arithmetic: `semquad_steady_step`'s truth
#       table, including the `steady_vrms` extension, the strict threshold, the
#       max(1, |x|) normalisation and the window/counter semantics through the
#       driver (a run must stop after `window` settled steps, not at maxsteps);
#   (c) the diagnostics' definitions against the analytic conductive limits:
#       Nu = 1 at both walls on the exact log-conduction state, the reference
#       gradients equal R_wall|ln f|, Vrms = 0, and the area mean of the log
#       conduction state matching its closed-form value to round-off;
#   (d) the stage-0.5c2 fix of the mapped entropy-viscosity wall rows: on the
#       annulus the rows switched off are the **thermal circles**
#       (`nu_wall_sides`), not the logical angular rows — the wedge used to
#       zero them on its insulating radial cuts;
#   (e) a very coarse smoke of the benchmark setup: one short paper-IC run on a
#       192-leaf ring, checking the free-slip wall rows, both gauges, the
#       history/diagnostics identities and the pressure-pin defect diagnostic.
#
# The published comparison numbers (Davies et al. 2022 / ASPECT tables) are NOT
# in this gate: they belong in the record and the docs, because a gate that
# embeds a multi-hour run cannot run in the suite and a gate that embeds a
# number without the run would be a tautology.
using Test
using QuaST
const MS277 = QuaST

# The shipped harness, included in an isolated module so this gate pins the
# *same* card code `examples/annulus_benchmark.jl` runs (its `main` is guarded
# by `abspath(PROGRAM_FILE) == @__FILE__` and does not execute on include).
module G277Harness
using QuaST, Printf
include(joinpath(@__DIR__, "..", "examples", "annulus_benchmark.jl"))
end
const HB = G277Harness

"""Area mean of Tbar(r) = ln(r/R1)/ln(R0/R1) by the closed-form antiderivative."""
function g277_log_conduction_mean(g)
    R0, R1 = g.R0, g.R1
    lnf = log(R0 / R1)
    F(r) = (0.5 * r^2 * log(r / R1) - 0.25 * r^2) / lnf
    return 2.0 / (R1^2 - R0^2) * (F(R1) - F(R0))
end

@testset "G277 annulus benchmark harness" begin

    # ==================================================================
    # (a) the benchmark card wiring
    # ==================================================================
    @testset "(a) card and resolution wiring" begin
        pr = HB.annulus_benchmark_params()
        @test pr.Ra == 1.0e5
        @test pr.H == 0.0
        @test pr.eta_law === :expT
        @test pr.deta_T == 1.0
        @test pr.Twall_top == 0.0
        @test pr.Twall_bot == 1.0
        @test pr.bc === :free
        @test pr.wall_bc == (:free, :free, :free, :free)
        # :expT with deta_T = 1 is exactly constant viscosity eta = 1, i.e. the
        # benchmark's isoviscous eta_ref = 1 (a different deta_T silently makes
        # the case non-isoviscous; this is the check that the card is not)
        @test MS277.eta_at(0.0, 0.0, pr)[1] == 1.0
        @test MS277.eta_at(1.0, 1.0, pr)[1] == 1.0
        @test MS277.eta_at(0.37, 0.42, pr)[1] == 1.0

        g = HB.annulus_benchmark_geometry()
        @test g.R0 == 1.22
        @test g.R1 == 2.22
        # the gap is the length unit d: the published radii differ by 1 to
        # published precision, and the geometry contract (stage 0.5a) tolerates
        # the one-ulp gap of the Float64 literals 1.22 / 2.22
        @test abs((g.R1 - g.R0) - 1.0) <= 2.0 * eps(1.0)
        @test g.Φ == 2.0 * pi
        @test g.seam === :periodic
        @test MS277._geom_periodic(g)
        @test MS277.nu_wall_sides(g) == (:xminus, :xplus)  # the circles

        # resolution -> tree: nx = 8, ny = 96 is level 3 with 12 angular base
        # cells, a uniform square-cell ring (the published 128x1536 mesh is the
        # same rule at level 7 / cells_y = 12)
        qt = HB.annulus_benchmark_tree(8, 96)
        @test qt.seam === :periodic
        @test qt.nx0 == 1
        @test qt.ny0 == 12
        @test qt.maxlevel == 3
        @test length(qt.leaves) == 8 * 96
        @test all(lf -> lf[1] == 3, qt.leaves)
        @test MS277.leaf_hx(qt, 3) == 1.0 / 96.0
        @test MS277.leaf_hy(qt, 3) == 1.0 / 96.0

        # the DOF arithmetic of the card against the space actually built
        ss = make_semstokes(qt, 2; geom=g)
        d = HB.annulus_benchmark_dofs(8, 96)
        @test d.leaves == 8 * 96
        @test d.T == (2 * 8 + 1) * (2 * 96)      # Q2 ring nodes: seam shared
        @test d.p == (8 + 1) * 96                # Q1 pressure nodes
        @test d.u == 2 * d.T
        @test d.saddle == 2 * d.T + d.p
        @test ss.su.N == d.T
        @test ss.sp.N == d.p
        @test 2 * ss.su.N + ss.sp.N == d.saddle

        # the resolution parser and the published-spec divisibility rule
        @test HB.parse_resolution("128x1536") == (128, 1536)
        @test HB.parse_resolution("512x2048") == (512, 2048)
        @test HB.parse_resolution("32") == (32, 384)
        @test ispow2(128) && 1536 % 128 == 0
        @test ispow2(512) && 2048 % 512 == 0
        # refusals: non-power-of-two nx, and ny not a multiple of nx
        @test_throws ArgumentError HB.annulus_benchmark_tree(3, 36)
        @test_throws ArgumentError HB.annulus_benchmark_tree(8, 100)
    end

    # ==================================================================
    # (b) the steady-state criterion arithmetic
    # ==================================================================
    @testset "(b) steady-state criterion" begin
        f = MS277.semquad_steady_step
        tol = 1.0e-6
        # all three quantities settled (thresholds from the definitions:
        # |dNu| < tol*max(1, 9.5), |dVrms| < tol*max(1, 193))
        @test f(0.0, 0.0, 0.0, 9.5, 193.0, tol; include_vrms=true)
        @test f(0.9e-6, 9.4e-6, 1.9e-4, 9.5, 193.0, tol; include_vrms=true)
        # strict comparisons: equality at the threshold is not settled
        @test !f(tol, 0.0, 0.0, 9.5, 193.0, tol; include_vrms=true)
        @test f(0.999e-6, 0.0, 0.0, 9.5, 193.0, tol; include_vrms=true)
        @test !f(0.0, tol * 9.5, 0.0, 9.5, 193.0, tol; include_vrms=true)
        @test f(0.0, 0.999 * tol * 9.5, 0.0, 9.5, 193.0, tol;
                include_vrms=true)
        # the V_RMS condition is opt-in: the historical criterion ignores it
        @test !f(0.0, 0.0, 2.0e-4, 9.5, 193.0, tol; include_vrms=true)
        @test f(0.0, 0.0, 2.0e-4, 9.5, 193.0, tol)
        @test f(0.0, 0.0, 1.92e-4, 9.5, 193.0, tol; include_vrms=true)
        # max(1, |x|): for a small quantity the test stays absolute instead of
        # dividing by the small value (Nu = 0.5 -> tolerance 1e-6, not 5e-7)
        @test !f(0.0, 1.1e-6, 0.0, 0.5, 0.2, tol; include_vrms=true)
        @test f(0.0, 0.9e-6, 0.0, 0.5, 0.2, tol; include_vrms=true)
        # one unsettled quantity fails the step no matter how well the others
        # are settled (conjunction, not disjunction)
        @test !f(1.0e-3, 0.0, 0.0, 9.5, 193.0, tol; include_vrms=true)
        @test !f(0.0, 1.0e-3, 0.0, 9.5, 193.0, tol; include_vrms=true)
        @test !f(0.0, 0.0, 1.0e-1, 9.5, 193.0, tol; include_vrms=true)
    end

    # ==================================================================
    # (c) the diagnostics against the analytic conductive limits
    # ==================================================================
    @testset "(c) conductive limits of the diagnostics" begin
        g = HB.annulus_benchmark_geometry()
        pr = HB.annulus_benchmark_params()
        qt = HB.annulus_benchmark_tree(8, 96)    # 768 leaves
        ss = make_semstokes(qt, 2; geom=g)
        z = zeros(Float64, ss.su.N)
        # on the exact log-conduction state (theta = 0) the wall flux of theta
        # is identically zero in the diagnostic, so Nu = ref/ref = 1 at *both*
        # walls: this pins the log normalisation and both wall signs
        nut, nub = MS277.semquad_nu_walls(ss.su, qt, z, pr)
        @test nut == 1.0
        @test nub == 1.0
        @test MS277.semquad_vrms(ss.su, qt, z, z) == 0.0
        # the reference gradients are the log-conduction ones --dTbar/dr =
        # 1/(r|ln f|), i.e. the benchmark multipliers R_wall|ln f| invert to 1
        bot_ref, top_ref = MS277.semquad_base_wall_refs(pr, g)
        lf = abs(log(g.R0 / g.R1))
        @test isapprox(bot_ref, 1.0 / (g.R0 * lf); rtol=1.0e-14)
        @test isapprox(top_ref, 1.0 / (g.R1 * lf); rtol=1.0e-14)
        @test isapprox(g.R0 * lf * bot_ref, 1.0; rtol=1.0e-14)
        @test isapprox(g.R1 * lf * top_ref, 1.0; rtol=1.0e-14)
        @test isapprox(g.R0 * bot_ref, g.R1 * top_ref; rtol=1.0e-14)
        # <T> of the conduction state: the discrete area mean equals the
        # closed-form log-conduction mean to round-off on every level.  This is
        # not automatic: the mean weighs the interpolated Tbar by the mapped
        # measure r dr dphi, so a wrong metric (e.g. a missing r factor or a
        # wrong angular Jacobian) moves it by per-cent, not 1e-13.
        mt = g277_log_conduction_mean(g)
        @test 0.3 < mt < 0.5                    # 0.4025..., not the 0.5 of a
        #                                         linear-in-r profile
        err = Float64[]
        for (nx, ny) in ((4, 48), (8, 96), (16, 192))
            q = HB.annulus_benchmark_tree(nx, ny)
            s = make_semstokes(q, 2; geom=g)
            push!(err, abs(MS277.semquad_mean_T(s.su, q, zeros(s.su.N), pr) - mt))
        end
        @test err[1] < 1.0e-11
        @test err[2] < 1.0e-11
        @test err[3] < 1.0e-11
    end

    # ==================================================================
    # (d) the mapped entropy-viscosity wall rows are the thermal circles
    # ==================================================================
    @testset "(d) mapped EV wall rows" begin
        # --- the ring: the seam rows are NOT walls, the circles are ----------
        gr = HB.annulus_benchmark_geometry()
        qtr = Quadtree(2; cells_x=1, cells_y=2, seam=:periodic)
        for _ in 1:2, lf in collect(qtr.leaves)
            refine!(qtr, lf)
        end
        balance!(qtr)
        ssr = make_semstokes(qtr, 2; geom=gr)
        pr = HB.annulus_benchmark_params()
        rr(c) = hypot(c[1], c[2])
        th(c) = 0.25 + 0.15 * sinpi(3.0 * atan(c[2], c[1])) +
                0.1 * log(rr(c) / gr.R0)
        thr = [th(c) for c in ssr.su.coords]
        uxr = [0.3 + 0.2 * sin(4.0 * atan(c[2], c[1])) for c in ssr.su.coords]
        uzr = [-0.2 + 0.15 * cos(3.0 * atan(c[2], c[1])) for c in ssr.su.coords]
        fldr = MS277.semquad_ev_field(gr, ssr.su, qtr, pr, thr, uxr, uzr;
                                      dt=1.0e-2, ev=1.0e3, ev_cmax=0.052,
                                      ev_alpha=2, ev_timestep=10)
        nx = 4
        ny = 8
        n_circle = 0
        n_seam_interior = 0
        for (i, lf) in enumerate(ssr.su.leaves)
            _, ix, iy = lf
            on_circle = ix == 0 || ix == nx - 1
            @test MS277._amr_wall_leaf(gr, ssr.su, lf) == on_circle
            if on_circle
                n_circle += 1
                @test fldr[i] == 0.0
            else
                @test fldr[i] > 0.0
                if iy == 0 || iy == ny - 1
                    n_seam_interior += 1
                end
            end
        end
        @test n_circle == 2 * ny                 # first/last radial row
        @test n_seam_interior > 0                # interior-radial seam leaves
        # --- the wedge: the radial cuts are NOT walls either (the stage-0.5c2
        #     fix; the old logical-`y` statement zeroed exactly these) ---------
        gw = PolarAnnulus(gr.R0, gr.R1, 0.5 * pi; seam=:wall)
        qtw = Quadtree(2; cells_x=1, cells_y=2)
        for _ in 1:2, lf in collect(qtw.leaves)
            refine!(qtw, lf)
        end
        balance!(qtw)
        ssw = make_semstokes(qtw, 2; geom=gw)
        rg(c) = hypot(c[1], c[2])
        thw = [0.25 + 0.15 * sinpi(3.0 * atan(c[2], c[1])) +
               0.08 * log(rg(c) / gw.R0) for c in ssw.su.coords]
        uxw = [0.3 + 0.2 * sin(4.0 * atan(c[2], c[1])) for c in ssw.su.coords]
        uzw = [-0.2 + 0.15 * cos(3.0 * (rg(c) - gw.R0)) for c in ssw.su.coords]
        fldw = MS277.semquad_ev_field(gw, ssw.su, qtw, pr, thw, uxw, uzw;
                                      dt=1.0e-2, ev=1.0e3, ev_cmax=0.052,
                                      ev_alpha=2, ev_timestep=10)
        ang_cut_interior = 0
        for (i, lf) in enumerate(ssw.su.leaves)
            _, ix, iy = lf
            on_circle = ix == 0 || ix == nx - 1
            @test MS277._amr_wall_leaf(gw, ssw.su, lf) == on_circle
            if on_circle
                @test fldw[i] == 0.0
            else
                # the leaf is interior in the radial direction: EV must be
                # alive even when it sits on a logical-y (radial-cut) row
                @test fldw[i] > 0.0
                (iy == 0 || iy == ny - 1) && (ang_cut_interior += 1)
            end
        end
        @test ang_cut_interior > 0
        @test MS277.nu_wall_sides(gw) == (:xminus, :xplus)
    end

    # ==================================================================
    # (e) a very coarse smoke of the benchmark setup
    # ==================================================================
    @testset "(e) coarse benchmark smoke" begin
        g = HB.annulus_benchmark_geometry()
        pr = HB.annulus_benchmark_params()
        qt = HB.annulus_benchmark_tree(4, 48)   # 192 leaves
        ss = make_semstokes(qt, 2; geom=g)
        dt = 2.0e-5
        # steady_tol = 1.0 with window 2: after the first step (which has no
        # previous Nu) the criterion must fire on the second settled step, so
        # the run stops at step 3 and not at maxsteps = 10.  This pins the
        # window/counter semantics through the driver.
        res = run_semquad_simulation(qt, pr, zeros(Float64, ss.su.N);
                                     geom=:annulus, annulus_R0=g.R0,
                                     annulus_R1=g.R1, annulus_Phi=g.Φ,
                                     annulus_seam=:periodic, p=2,
                                     tstop=10 * dt, dt0=dt, dtmax=dt,
                                     dtmin=1.0e-16, cfl=0.0, maxsteps=10,
                                     linear=:direct, advection=:galerkin,
                                     maxpic=1, annulus_ic=:firedrake,
                                     steady_tol=1.0, steady_window=2,
                                     steady_vrms=true, tmin=0.0)
        @test res.steps == 3
        @test length(res.history) == 3
        @test all(row -> length(row) == 18, res.history)
        @test all(isfinite, vcat(res.history...))
        @test res.qt.seam === :periodic
        @test MS277.LAST_SEMSTOKES_ROTATION_GAUGE[] === :rotation
        @test MS277.LAST_SEMSTOKES_PRESSURE_GAUGE[] === :pin
        # the history row is the returned state's diagnostics
        row = res.history[end]
        nut, nub = MS277.semquad_nu_walls(res.ss.su, res.qt, res.theta, pr)
        @test row[3] == nut
        @test row[4] == nub
        @test row[5] == MS277.semquad_vrms(res.ss.su, res.qt, res.ux, res.uz)
        @test row[6] == MS277.semquad_mean_T(res.ss.su, res.qt, res.theta, pr)
        # the paper IC has spun the flow up (it is not the zero state)
        @test row[5] > 1.0e-3
        @test isfinite(nut) && isfinite(nub)
        # free-slip circles: the nodal normal velocity vanishes
        bmask = MS277._semquad_boundary_face_mask(res.qt, res.ss.su)
        frames = MS277._semquad_wall_frames(g, res.ss.su, bmask)
        wnorm = 0.0
        for (node, cs) in frames
            c = cs[1]
            wnorm = max(wnorm, abs(c.nx * res.ux[node] + c.ny * res.uz[node]))
        end
        @test wnorm <= 1.0e-10
        # the pressure-pin caveat's direct measure: the dropped continuity row
        # is the volume integral of div u; it must be finite and small against
        # the velocity scale (the benchmark record quotes its value on the
        # converged states; the gate only pins the diagnostic's scale)
        divv = HB.annulus_div_volume(g, res.ss.su, res.ux, res.uz)
        area = MS277.total_measure(g, res.ss.su)
        umax = maximum(hypot.(res.ux, res.uz))
        @test isfinite(divv)
        @test area > 0.0
        @test abs(divv) <= 1.0e-2 * umax * area
        # the energy diagnostic on the *initial* linear profile: the arc-length
        # averaged radial gradient of T = R1 - r + perturbation is -1, so the
        # pair reads R_wall|ln f| (a 1% band covers the O(h^2) wall-derivative
        # error; this is the diagnostic-definition check, not a benchmark run)
        th_ic = [MS277._annulus_ic_theta(:firedrake, g, pr, c[1], c[2])
                 for c in ss.su.coords]
        nut0, nub0 = MS277.semquad_nu_walls(ss.su, qt, th_ic, pr)
        lf = abs(log(g.R0 / g.R1))
        @test isapprox(nut0, g.R1 * lf; rtol=1.0e-2)
        @test isapprox(nub0, g.R0 * lf; rtol=1.0e-2)
    end
end
