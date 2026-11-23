# G262: the pressure gauge of the boundary-condition feature's stage 3, and the
# well-posedness of an open-wall configuration.
#
# Today's solver pins one nodal pressure dof to zero and drops its continuity
# equation (`solve_var_core.jl`, `pfree0`/`pf`).  That is exact **iff** the
# constant pressure is in the null space of the reduced gradient, i.e. iff the
# normal velocity is Dirichlet on the whole boundary: then every admissible test
# function has `v.n = 0` on the boundary, `int div v = int_{dOmega} v.n = 0`, and
# the dropped row is a linear combination of the others.  An `:open` wall
# destroys that: a boundary test function with a non-zero normal trace exists, so
# the constant is not in the null space and a pin would replace a continuity
# equation by `p_a = 0` -- a different problem (docs/dev/BC_DESIGN.md §3.5).
# Stage 3 makes the gauge derived: **pin iff no wall is `:open`**.
#
# This gate pins:
#   (a) the traction-free open lid (bottom and sides `:rigid`, top `:open`, no
#       load, no body force): the exact solution is `u = 0`, `p = 0`, and the
#       no-pin saddle factorises -- this is the discrete inf-sup verification the
#       design could not run (BC_DESIGN.md §7 question 1);
#   (b) the hydrostatic free-slip/open configuration (sides `:free`, bottom
#       `:rigid`, top `:open`, body force `f = (0,-1)`): exact `u = 0`,
#       `p = 1 - y`, with the pressure level fixed by the open boundary, not by a
#       pin;
#   (c) the decision itself: `LAST_SEMSTOKES_PRESSURE_GAUGE` is `:pin` exactly
#       for the configurations with no `:open` wall, and the returned `C_p` has
#       `np - 1` columns when pinned, `np` when not -- two independent
#       observations of the same decision (the Ref and the constraint matrix);
#   (d) the refused configurations and their messages: all four walls `:open`
#       (pure traction), and the configurations that leave a rigid translation
#       admissible (both x-walls open / both y-walls open), refused by `Params`
#       and by a direct solver call with the same text;
#   (e) the numeric rigid-mode rank test the solver uses: the 3x3 Gram matrix of
#       the rigid modes restricted to the Dirichlet dofs.  The valid open
#       configuration's eigenvalues are all O(1) and the helper accepts it; a
#       hand-restricted dof set (all `u_x` dofs, no `u_z` dof) is singular
#       (eigenvalue exactly 0) and the same helper refuses it -- so the test is
#       not a name for the combinatorial rule only;
#   (f) the pin is not just absent but unnecessary: the open-wall solves take no
#       direct-solver regularisation (`LAST_SEMSTOKES_SHIFT == 0`) and their true
#       reduced residual is small (the no-pin saddle is nonsingular).
#
# Measured numbers are in the comments.

using Test, QuaST, LinearAlgebra

function g262_tree()
    qt = Quadtree(2)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

g262_zero(x, y) = 0.0

@testset "G262 open walls: derived pressure gauge and well-posedness" begin
    qt = g262_tree()
    ss = make_semstokes(qt, 2)
    np = ss.sp.N
    zp = [c[2] for c in ss.sp.coords]
    z0 = g262_zero
    eta1 = (x, y) -> 1.0

    function g262_asm(; wall_bc, kwargs...)
        return solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                        bc=:free, wall_bc=wall_bc,
                                        linear=:direct, assemble_only=true,
                                        kwargs...)
    end

    # ---- (a) traction-free open lid: u = 0, p = 0, unpinned saddle ---------
    # bottom + both sides :rigid, top :open with no wall_tn (zero normal
    # traction) and no body force: sigma.n = 0 on the top wall is satisfied by
    # u = 0, p = 0.  Nothing else can be a solution, so this is exactly the test
    # that the unpinned saddle has full rank.
    r = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                 bc=:free,
                                 wall_bc=(:rigid, :rigid, :rigid, :open),
                                 linear=:direct)
    @test maximum(abs, r[1]) < 1.0e-12            # measured 0.0
    @test maximum(abs, r[2]) < 1.0e-12            # measured 0.0
    @test LAST_SEMSTOKES_PRESSURE_GAUGE[] === :none
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-12     # measured 0.0
    # no regularisation of the direct solve was needed: the saddle is
    # nonsingular, not merely "solvable after a shift"
    @test LAST_SEMSTOKES_SHIFT[] == 0.0
    @test LAST_SEMSTOKES_REGULARIZED[] == false
    a = g262_asm(wall_bc=(:rigid, :rigid, :rigid, :open))
    @test all(iszero, a.fv)                        # no load at all
    @test size(a.Cp, 2) == np                      # unpinned pressure space

    # ---- (b) hydrostatic free-slip/open configuration ---------------------
    # sides :free (normal velocity Dirichlet zero, tangential natural), bottom
    # :rigid, top :open with zero traction; body force f_z = -1.  With u = 0,
    # -div sigma = f gives grad p = (0,-1) and the top traction -p n = 0 gives
    # p(top) = 0: p = 1 - y.  The pressure level is set by the open boundary.
    rp = solve_semquad_stokes_var(ss, qt, eta1, z0, (x, y) -> -1.0, z0, z0;
                                  bc=:free,
                                  wall_bc=(:free, :free, :free, :open),
                                  linear=:direct)
    @test maximum(abs, rp[1]) < 1.0e-12            # measured 9.98e-16
    @test maximum(abs.(rp[2] .- (1.0 .- zp))) < 1.0e-10   # measured 7.16e-15
    @test LAST_SEMSTOKES_PRESSURE_GAUGE[] === :none
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-12      # measured 7.22e-15

    # ---- (c) the pin is on exactly when no wall is open -------------------
    # One row per configuration; `expected_pin` is the rule "no wall is :open".
    # The two observations must agree: the recorded gauge and the C_p column
    # count (np - 1 with a pin, np without).
    configs = [
        (name="all :free", bc=(:free, :free, :free, :free)),
        (name="all :rigid", bc=(:rigid, :rigid, :rigid, :rigid)),
        (name="mixed closed", bc=(:free, :free, :free, :rigid)),
        (name="top open", bc=(:rigid, :rigid, :rigid, :open)),
        (name="bottom open", bc=(:rigid, :rigid, :open, :rigid)),
        (name="xminus open", bc=(:open, :rigid, :rigid, :rigid)),
        (name="xplus open", bc=(:rigid, :open, :rigid, :rigid)),
        (name="top open, free sides", bc=(:free, :free, :free, :open)),
        (name="corner open", bc=(:open, :rigid, :rigid, :open)),
    ]
    for c in configs
        expected_pin = !any(w -> w === :open, c.bc)
        asm = g262_asm(wall_bc=c.bc)
        @test LAST_SEMSTOKES_PRESSURE_GAUGE[] === (expected_pin ? :pin : :none)
        @test size(asm.Cp, 2) == (expected_pin ? np - 1 : np)
        # the pin exists in the *legacy* sense: a pinned solve's C_p has one
        # row fewer than the pressure space, an unpinned one has every row
        @test size(asm.Cp, 1) == np
        # a resolve with the same configuration reports the same decision (the
        # Ref is per-solve, not sticky)
        solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                 bc=:free, wall_bc=c.bc, linear=:direct)
        @test LAST_SEMSTOKES_PRESSURE_GAUGE[] === (expected_pin ? :pin : :none)
    end
    # the default path (library defaults, `bc=:free`, no per-wall tuple) still
    # pins, which is the historical behaviour bit for bit
    solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0; linear=:direct)
    @test LAST_SEMSTOKES_PRESSURE_GAUGE[] === :pin

    # ---- (d) refused configurations and their exact messages --------------
    all_open = (:open, :open, :open, :open)
    @test_throws ArgumentError Params(wall_bc=all_open)
    err = try
        solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                 bc=:free, wall_bc=all_open, linear=:direct)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    msg = err isa ArgumentError ? sprint(showerror, err) : ""
    @test occursin("all four walls are :open", msg)
    @test occursin("pure traction", msg)
    @test occursin("rigid motion", msg)
    # an open pair with no Dirichlet normal component in that direction leaves
    # the corresponding rigid translation admissible; both layers refuse it and
    # name the motion.
    for (bc, word) in [((:open, :open, :free, :free), "x-translation"),
                       ((:free, :free, :open, :open), "y-translation")]
        @test_throws ArgumentError Params(wall_bc=bc)
        errg = try
            solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                     bc=:free, wall_bc=bc, linear=:direct)
            nothing
        catch e
            e
        end
        @test errg isa ArgumentError
        msgg = errg isa ArgumentError ? sprint(showerror, errg) : ""
        @test occursin(word, msgg)
        @test occursin("Gram", msgg)
        @test occursin("rigid-body mode", msgg)
    end
    # every non-refused open configuration of the table above solved; also a
    # single-open-corner one (xminus ∩ yplus) is well posed and unpinned
    rc = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                  bc=:free,
                                  wall_bc=(:open, :rigid, :rigid, :open),
                                  linear=:direct)
    @test LAST_SEMSTOKES_PRESSURE_GAUGE[] === :none
    @test all(isfinite, rc[1]) && all(isfinite, rc[2])
    @test maximum(abs, rc[1]) < 1.0e-12            # zero load, zero data

    # ---- (e) the numeric rigid-mode rank test ------------------------------
    # Valid configuration: take the final Dirichlet dof set of an open solve and
    # require the Gram matrix to have rank 3 with an O(1) eigen ratio.
    rg = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                  bc=:free,
                                  wall_bc=(:rigid, :rigid, :rigid, :open),
                                  linear=:direct, return_system=true)
    fixed = rg[3]
    G = QuaST._semquad_rigid_mode_gram(ss, fixed)
    lam = eigvals(Symmetric(G))
    @test length(lam) == 3
    @test lam[1] > 1.0e-3 * lam[3]                 # measured ratio 0.146
    @test QuaST._semquad_check_rigid_modes(ss, fixed,
                                           (:rigid, :rigid, :rigid, :open)) === nothing
    # Hand-restricted dof set: drop every u_z Dirichlet dof, so only `u_x` is
    # fixed and the y-translation is admissible.  The matrix test must see the
    # exact zero eigenvalue and the helper must refuse.
    fixed_bad = Dict{Int,Float64}(k => v for (k, v) in fixed if k <= ss.su.N)
    @test !isempty(fixed_bad)
    Gb = QuaST._semquad_rigid_mode_gram(ss, fixed_bad)
    lamb = eigvals(Symmetric(Gb))
    @test lamb[1] <= 1.0e-10 * lamb[3]             # measured exactly 0.0
    errb = try
        QuaST._semquad_check_rigid_modes(ss, fixed_bad,
                                         (:free, :free, :free, :free))
        nothing
    catch e
        e
    end
    @test errb isa ArgumentError
    msgb = errb isa ArgumentError ? sprint(showerror, errb) : ""
    @test occursin("rigid-body mode", msgb)
    @test occursin("eigenvalues", msgb)
    @test occursin("y-translation", msgb)          # the null vector is reported

    # ---- (f) the unpinned saddle is not merely shift-regularised ----------
    # Every open-wall solve above took `LAST_SEMSTOKES_SHIFT == 0`; additionally
    # the exact reduced residual of a driven open-wall solve is small.
    rdrv = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                    bc=:free,
                                    wall_bc=(:rigid, :rigid, :rigid, :open),
                                    wall_tn=(nothing, nothing, nothing,
                                             (x, y) -> 1.0 + 0.5 * x),
                                    linear=:direct)
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-12      # measured 1.86e-15
    @test LAST_SEMSTOKES_SHIFT[] == 0.0
    @test maximum(abs, rdrv[1]) > 1.0e-3           # the load drives a flow
    @test maximum(abs, rdrv[2]) > 1.0e-3
end
