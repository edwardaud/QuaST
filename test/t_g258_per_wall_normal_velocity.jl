# G258: per-wall prescribed normal velocity (boundary-condition feature, stage 1).
#
# Before this gate the normal component of every wall was Dirichlet with a value
# from the *global* `uDx`/`uDz` closures the driver hard-wires to zero: a card
# could not prescribe a normal velocity on one wall, and there was no `Params`
# or card knob for it (docs/dev/BC_DESIGN.md §1.3/§1.5).  `Params.wall_un`
# (four entries, order `(xminus, xplus, yminus, yplus)`) now refines that value
# per wall; `nothing` keeps the historical global closure, which is what makes
# the default path bit-identical.
#
# This gate pins:
#   (a) the pre-change default path, by hashes recorded from the checkout
#       *before* the feature (docs/dev/BC_PROGRESS.md stage 1 baseline);
#   (b) the analytic Couette solution driven through `wall_un` + `wall_u`, and
#       that it is bitwise the G238 global-closure answer (same fixed data);
#   (c) per-wall independence: a wall_un on one wall touches only that wall's
#       normal dofs, and is imposed exactly;
#   (d) the corner-owner table of docs/dev/BC_DESIGN.md §1.4/§3.4, per
#       component, for every combination of rigid/free walls;
#   (e) rejection of the invalid (`wall_u` on a resolved `:free` wall, silently
#       ignored before this stage) and of the **insufficient** combination
#       (normal data with non-zero net boundary flux), plus the balanced
#       through-flow that must be accepted;
#   (f) the restart guard: a checkpoint written with one `wall_un` is refused
#       when resumed with another (the marker format itself is pinned by G245).
#
# The numbers in (b) are measured; see the comments there for the exact values.
# The pin in (a) is not tautological: the five constants of each run were
# produced by the tree with the feature absent, and this file only compares.

using Test, QuaST, SHA

# The suite's small unit box: Quadtree(2) fully refined twice -> 8x8 square
# leaves, which carries corners and non-corner wall nodes for (b)-(e).
function g258_tree()
    qt = Quadtree(2)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

g258_hash(a) = bytes2hex(sha256(reinterpret(UInt8, vec(collect(Float64, a)))))
g258_zero(x, y) = 0.0

@testset "G258 per-wall normal velocity" begin
    # ---- (a) the default path is bit-identical to the pre-change reference ----
    # Two driver runs on a 16-leaf tree, both exactly as the stage-1 baseline
    # script ran them (same tree, theta, Params and keywords).  The hashes were
    # recorded there and are reproduced verbatim only if the whole wall path --
    # the fixed-dict write order, the u0 lift, the net-flux guard and the solve --
    # is unchanged.  Run A is the plain free-slip buoyancy card; run B is the
    # mixed per-wall card with a moving lid from `wall_u`, i.e. the path whose
    # threading `wall_un` could have disturbed.
    kw = (p=2, tstop=2.0e-3, dt0=1.0e-3, dtmax=1.0e-3, dtmin=1.0e-16,
          amr_every=0, ev=0.0, linear=:direct, maxpic=1, verbose=false,
          mass_form=:boussinesq)
    function g258_driver_tree()
        qt = Quadtree(4)
        for _ in 1:1, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        return qt
    end
    qtA = g258_driver_tree()
    ssA = make_semstokes(qtA, 2)
    thA = [0.1 * sinpi(x) * sinpi(z) for (x, z) in ssA.su.coords]
    prA = Params(Ra=1.0e3, H=0.0, eta_law=:expT, deta_T=1.0, bc=:free)
    rA = run_semquad_simulation(qtA, prA, thA; kw...)
    @test g258_hash(rA.theta) ==
          "51359c4befac180b3ea7f8fb82f42cbdc263a7aee1461faf0c032767f8b96fd0"
    @test g258_hash(rA.ux) ==
          "f92e85ee6439d1ba8ee88d8d59a9577b55dc29e9b967185fe0901ea56683bc3f"
    @test g258_hash(rA.uz) ==
          "c50e5e15f08b39e9882b2bd0cb143e78d9068b9262893a3388677b86141f3930"
    @test g258_hash(rA.pv) ==
          "77a6db263391a424c84f05e3d060495097e31f5fc794348ed327bf8626854c91"
    @test g258_hash(reduce(vcat, rA.history)) ==
          "4079e2e621d437460833aba96974b1247fbe4bd2fbbb14295e31e612381bc02d"

    qtB = g258_driver_tree()
    ssB = make_semstokes(qtB, 2)
    thB = zeros(Float64, ssB.su.N)
    prB = Params(Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0, bc=:free,
                 wall_bc=(:free, :free, :free, :rigid),
                 wall_u=(nothing, nothing, nothing, (x, y) -> 1.0))
    rB = run_semquad_simulation(qtB, prB, thB; kw...)
    @test g258_hash(rB.theta) ==
          "4cda5ddb4685b67e12d3993c0cad1621784dc54e18dd367471edee792eb62cb6"
    @test g258_hash(rB.ux) ==
          "ac74fb469bb6b100a0610cbbfb586719b81137333d82e773de9a8746ad9f3108"
    @test g258_hash(rB.uz) ==
          "bcaf3dca77535cabac2962b75485f6dbd7923c96e3771180f8142e884dc212e5"
    @test g258_hash(rB.pv) ==
          "500fec4bb77d3adc3c2a03606a3eeaf0972537817db8fc1a5250c16538c2e928"
    @test g258_hash(reduce(vcat, rB.history)) ==
          "c66ef70a507e25cdefa74e8a266a887649af7aec10bd61dc7a26dba187491ae4"
    # non-vacuous: the two reference runs are different, and the lid actually
    # drives the flow that run B pins (a zeroed ux would make the hash trivial)
    @test g258_hash(rB.ux) != g258_hash(rA.ux)
    @test maximum(abs, rB.ux) > 0.1

    # the defaults of the public surface
    @test Params().wall_un == (nothing, nothing, nothing, nothing)
    prdef = Params()
    @test prdef.wall_un === (nothing, nothing, nothing, nothing)
    prl = Params(wall_un=(nothing, nothing, nothing, (x, y) -> 2.0))
    @test prl.wall_un[4](0.3, 0.7) == 2.0
    @test_throws Exception Params(wall_un=(nothing, nothing, nothing))

    # ---- (b) Couette through wall_un on the side walls + wall_u on the lid ---
    # Constant viscosity, no buoyancy, no body force.  The four walls are
    # `:rigid`, the lid moves tangentially at U, and the side walls' *normal*
    # component (u_x) carries the Couette profile: exact u = (U y, 0), p = 0.
    # A linear velocity field is represented exactly by Q_2 with constant
    # viscosity and the boundary data are linear (integrated exactly), so the
    # only error left is the linear solve.  Measured: max|u_x - U y|/U =
    # 1.71e-14, max|u_z|/U = 5.62e-15, max|p| = 1.15e-12.
    U = 1.0
    qt = g258_tree()
    ss = make_semstokes(qt, 2)
    sq = ss.su
    N = sq.N
    xs = [c[1] for c in sq.coords]
    ys = [c[2] for c in sq.coords]
    z0 = g258_zero
    near(a, b) = abs(a - b) < 1.0e-12
    onx(i) = near(xs[i], 0.0) || near(xs[i], 1.0)
    ony(i) = near(ys[i], 0.0) || near(ys[i], 1.0)
    corner(i) = onx(i) && ony(i)

    r = solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0, z0, z0, z0, z0;
                                 bc=:rigid,
                                 wall_un=((x, y) -> U * y, (x, y) -> U * y,
                                          (x, y) -> 0.0, (x, y) -> 0.0),
                                 wall_u=(nothing, nothing, nothing,
                                         (x, y) -> U),
                                 linear=:direct)
    u = r[1]
    ux = u[1:N]
    uz = u[N + 1:end]
    @test maximum(abs.(ux .- U .* ys)) / U < 1.0e-10
    @test maximum(abs, uz) / U < 1.0e-10
    @test maximum(abs, r[2]) < 1.0e-10
    @test maximum(abs, ux) > 0.1 * U          # the flow is actually driven
    # G238's global-closure data (uDx = U y on every wall) writes exactly the
    # same values into `fixed` for this configuration -- same normal on the
    # x-walls, same tangential on the y-walls -- so the per-wall API must
    # reproduce that independently gated answer bit for bit.
    rleg = solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0, z0, z0,
                                    (x, y) -> U * y, z0;
                                    bc=:rigid, linear=:direct)
    @test r[3] == rleg[3]
    @test r[1] == rleg[1]
    @test r[2] == rleg[2]

    # ---- (c) per-wall independence ----------------------------------------
    # `y - 1/2` on one x-wall only: its nodal integral is exactly zero (a degree
    # 1 polynomial, integrated exactly), so the prescribed data keep zero net
    # flux and the closed-box consistency guard accepts it.  Setting it must
    # change only that wall's normal dofs; the other walls, the y-wall normals
    # and every tangential component stay exactly at their default values.
    fixed_def = solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0, z0, z0,
                                         z0, z0;
                                         bc=:free, linear=:direct)[3]
    f_xm = solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0, z0, z0, z0, z0;
                                    bc=:free,
                                    wall_un=((x, y) -> y - 0.5, nothing,
                                             nothing, nothing),
                                    linear=:direct)
    f_xp = solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0, z0, z0, z0, z0;
                                    bc=:free,
                                    wall_un=(nothing, (x, y) -> y - 0.5,
                                             nothing, nothing),
                                    linear=:direct)
    f_m = f_xm[3]
    f_p = f_xp[3]
    # the value is imposed exactly on the wall it was given
    for i in 1:N
        (onx(i) && near(xs[i], 0.0)) &&
            @test f_m[i] == ys[i] - 0.5
        (onx(i) && near(xs[i], 1.0)) && @test f_m[i] == 0.0
        (onx(i) && near(xs[i], 1.0)) && @test f_p[i] == ys[i] - 0.5
        (onx(i) && near(xs[i], 0.0)) && @test f_p[i] == 0.0
        ony(i) && @test f_m[N + i] == 0.0 && f_p[N + i] == 0.0
    end
    # only the named wall's dofs differ from the default run, and only between
    # the two runs does the *other* wall keep its value: no leak in either
    # direction (a global-closure implementation would move both walls)
    diffs_m = [k for k in keys(f_m) if !haskey(fixed_def, k) || f_m[k] != fixed_def[k]]
    diffs_p = [k for k in keys(f_p) if !haskey(fixed_def, k) || f_p[k] != fixed_def[k]]
    @test !isempty(diffs_m) && !isempty(diffs_p)
    @test all(k -> onx(k) && near(xs[k], 0.0), diffs_m)
    @test all(k -> onx(k) && near(xs[k], 1.0), diffs_p)
    # ... and the solution carries the prescribed value at those dofs exactly
    for i in 1:N
        (onx(i) && near(xs[i], 0.0)) && @test f_xm[1][i] == ys[i] - 0.5
        (onx(i) && near(xs[i], 1.0)) && @test f_xp[1][i] == ys[i] - 0.5
    end

    # ---- (d) the corner-owner table --------------------------------------
    # A corner node belongs to an x-wall and a y-wall, and the fixed loop writes
    # `(xminus, xplus, yminus, yplus)` in that order, so a later write owns the
    # dof: u_x at a corner is the x-wall's normal value unless the y-wall is
    # `:rigid`, whose tangential value is written later and owns it; u_z at a
    # corner is always the y-wall's normal value (the x-wall's tangential write,
    # if any, comes earlier).  The data are chosen to keep zero net flux: the
    # rigid-wall runs use one value for both x-walls and one for both y-walls.
    function g258_corner_fixed(wall_un, wall_bc, wall_u)
        return solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0, z0, z0,
                                        z0, z0; bc=:free,
                                        wall_un=wall_un, wall_bc=wall_bc,
                                        wall_u=wall_u,
                                        linear=:direct)[3]
    end
    g258_corners(fix) = [(i, xs[i], ys[i], fix[i], fix[N + i])
                         for i in 1:N if corner(i)]
    consts4(a, b, c, d) = ((x, y) -> a, (x, y) -> b,
                           (x, y) -> c, (x, y) -> d)
    # 1. all walls :free, four distinct normal values (net flux -4+1-2+5 = 0)
    f1 = g258_corner_fixed(consts4(4.0, 1.0, 2.0, 5.0),
                           (:same, :same, :same, :same),
                           (nothing, nothing, nothing, nothing))
    c1 = g258_corners(f1)
    @test length(c1) == 4
    owner1 = Dict((0.0, 0.0) => (4.0, 2.0), (0.0, 1.0) => (4.0, 5.0),
                  (1.0, 0.0) => (1.0, 2.0), (1.0, 1.0) => (1.0, 5.0))
    for (_, x, y, ux, uz) in c1
        @test ux == owner1[(x, y)][1]
        @test uz == owner1[(x, y)][2]
    end
    # 2. a rigid lid (yplus) with tangential value 7 owns u_x at the two top
    # corners; u_z there is still the yplus *normal* value.
    f2 = g258_corner_fixed(consts4(4.0, 4.0, 2.0, 2.0),
                           (:free, :free, :free, :rigid),
                           (nothing, nothing, nothing, (x, y) -> 7.0))
    c2 = g258_corners(f2)
    @test length(c2) == 4
    for (_, x, y, ux, uz) in c2
        @test ux == (y == 1.0 ? 7.0 : 4.0)
        @test uz == 2.0
    end
    # 3. same for a rigid floor (yminus): it owns u_x at the two bottom corners.
    f3 = g258_corner_fixed(consts4(4.0, 4.0, 2.0, 2.0),
                           (:free, :free, :rigid, :free),
                           (nothing, nothing, (x, y) -> 9.0, nothing))
    c3 = g258_corners(f3)
    @test length(c3) == 4
    for (_, x, y, ux, uz) in c3
        @test ux == (y == 0.0 ? 9.0 : 4.0)
        @test uz == 2.0
    end
    # 4. rigid x-walls write a tangential u_z of 6, but the y-wall's normal u_z
    # is written later and still owns every corner; the x-wall keeps its normal
    # u_x (the y-walls are :free and write no tangential u_x).
    f4 = g258_corner_fixed(consts4(4.0, 4.0, 2.0, 2.0),
                           (:rigid, :rigid, :free, :free),
                           ((x, y) -> 6.0, (x, y) -> 6.0,
                            nothing, nothing))
    c4 = g258_corners(f4)
    @test length(c4) == 4
    for (_, x, y, ux, uz) in c4
        @test ux == 4.0
        @test uz == 2.0                       # the y normal wins over the x tang
    end
    # the non-corner x-wall nodes do carry the x-wall's tangential value: the
    # overwrite above is a corner effect only
    for i in 1:N
        (onx(i) && !corner(i)) && @test f4[N + i] == 6.0
    end

    # ---- (e) invalid and insufficient configurations are rejected ----------
    # (e1) `wall_u` is a tangential value; on a resolved `:free` wall it used to
    # be silently ignored (pre-G258).  Both layers refuse it now.
    lid = (x, y) -> 1.0
    @test_throws ArgumentError Params(bc=:free, wall_u=(nothing, nothing,
                                                        nothing, lid))
    @test_throws ArgumentError Params(wall_u=(nothing, nothing, nothing, lid))
    # ... but the same tuple is fine when the wall resolves to :rigid
    @test Params(bc=:free, wall_bc=(:free, :free, :free, :rigid),
                 wall_u=(nothing, nothing, nothing, lid)) isa Params
    err = try
        solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0, z0, z0, z0, z0;
                                 bc=:free,
                                 wall_u=(nothing, nothing, nothing, lid),
                                 linear=:direct)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    msg = err isa ArgumentError ? sprint(showerror, err) : ""
    @test occursin("tangential", msg) && occursin("ignored", msg)

    # (e2) every wall keeps a Dirichlet normal component, so unbalanced normal
    # data are an inconsistent closed-box problem: refuse with the measured flux.
    one1 = (x, y) -> 1.0
    @test_throws ArgumentError solve_semquad_stokes_var(
        ss, qt, (x, y) -> 1.0, z0, z0, z0, z0; bc=:free,
        wall_un=(one1, nothing, nothing, nothing), linear=:direct)
    err2 = try
        solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0, z0, z0, z0, z0;
                                 bc=:free,
                                 wall_un=(one1, nothing, one1, nothing),
                                 linear=:direct)
        nothing
    catch e
        e
    end
    @test err2 isa ArgumentError
    msg2 = err2 isa ArgumentError ? sprint(showerror, err2) : ""
    @test occursin("net outward boundary flux", msg2)
    @test occursin("balance", msg2)

    # (e3) ... and a balanced prescription is accepted: a uniform through-flow
    # u = (U, 0), p = 0 with all walls :free (normal Dirichlet, tangential
    # natural; a uniform field has zero strain, so the natural condition holds).
    U2 = 1.0
    rp = solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0, z0, z0, z0, z0;
                                  bc=:free,
                                  wall_un=((x, y) -> U2, (x, y) -> U2,
                                           nothing, nothing),
                                  linear=:direct)
    @test maximum(abs.(rp[1][1:N] .- U2)) < 1.0e-10
    @test maximum(abs, rp[1][N + 1:end]) < 1.0e-10
    @test maximum(abs, rp[2]) < 1.0e-10

    # ---- (f) the restart guard records the wall values --------------------
    # `run_config` records `wall_bc` but not `wall_u` before this stage, and
    # `wall_un` is new; a checkpoint whose wall motion changes must be refused
    # instead of continuing with different boundary data (both fields are
    # function tuples, so they are recorded as one `none`/`custom` marker per
    # wall, the `eta_multiplier`/`rho_multiplier` convention).  This is the
    # end-to-end proof; G245 pins the marker format itself.
    qtck = g258_tree()
    ssck = make_semstokes(qtck, 2)
    thck = zeros(Float64, ssck.su.N)
    ckpath = joinpath(tempdir(), "g258_wall_un_restart.h5")
    prck = Params(Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0, bc=:free)
    ckkw = (p=2, tstop=2.0e-3, dt0=1.0e-3, dtmax=1.0e-3, dtmin=1.0e-16,
            amr_every=0, ev=0.0, linear=:direct, maxpic=1, verbose=false,
            mass_form=:boussinesq)
    run_semquad_simulation(qtck, prck, thck; ckkw...,
                           checkpoint_path=ckpath, checkpoint_every=1)
    prck2 = Params(Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0, bc=:free,
                   wall_un=((x, y) -> 0.5, (x, y) -> 0.5, nothing, nothing))
    errck = try
        run_semquad_simulation(qtck, prck2, thck; ckkw...,
                               restart_from=ckpath)
        nothing
    catch e
        e
    end
    @test errck isa ArgumentError
    msgck = errck isa ArgumentError ? sprint(showerror, errck) : ""
    @test occursin("restart configuration mismatch", msgck)
    @test occursin("wall_un", msgck)
    @test occursin("checkpoint=none,none,none,none", msgck)
    @test occursin("current=custom,custom,none,none", msgck)
    rm(ckpath; force=true)
end
