# G259: a prescribed *tangential traction* drives an exact Couette flow -- the
# sign gate of the boundary-condition feature's stage 2 (G258 was stage 1's
# per-wall normal velocity).
#
# Before stage 2 the Stokes assembly had no boundary-face integral at all: the
# natural (do-nothing) condition is `sigma.n = 0`, so any free boundary
# component was silently traction-free.  `Params.wall_t` now prescribes the
# tangential component of the **total** Cauchy traction `tbar = sigma.n` as a
# boundary-face RHS load `fv_i += int tbar.phi_i dS`: integration by parts puts
# the boundary term on the right-hand side with a **plus** sign, and the value
# is the global component in the
# positive coordinate direction (`t_x` on a horizontal wall, `t_z` on a vertical
# one), with no outward-normal table.
#
# Constant viscosity `eta0 = 3` (not 1: a missing or spurious `eta` factor in
# the load or in the expected profile must change the answer), top wall `:free`
# with `wall_t = tau0`, the other three walls `:rigid`, no body force.  The
# exact solution of `-div sigma = f` with that mixed condition is
# `u = (tau0*y/eta0, 0)`, `p = 0`: the tangential traction is pressure-free on
# an axis-aligned wall (`t_x = sigma_xy = eta0 du_x/dy`), so `u_x` is linear and
# the Q2 space contains it exactly; the Gauss data are integrated exactly and
# the only error left is the linear solve.  The bottom-wall variant prescribes
# the same positive `wall_t` on the *bottom* wall: because the global traction
# component is prescribed and the outward normal is `-e_y`, the same value now
# means `sigma_xy = -tau0` there, so the exact field is `u = (tau0*(1-y)/eta0, 0)`
# -- the opposite sign of the shear.  A normal-sign table or a flipped load sign
# fails one of the two variants.
#
# This gate pins:
#   (a) the top-traction Couette field against the analytic solution, the G238
#       *velocity-driven* Couette (different operator, same answer) and a
#       discrete mass/flux-conservation check of the resulting boundary flux;
#   (b) the bottom-wall variant with the opposite shear sign;
#   (c) the load itself: total load = `tau0*Lx`, support only on the loaded
#       `u_x` rows, and the exact Q2 nodal values `int tau0 phi ds` on the
#       loaded face (this is the `(h/2)*gl_w*phi` weak form, not a nodal
#       collocation); plus the x-wall swap (there `wall_t` loads the `u_z`
#       rows);
#   (d) the sign/eta discrimination: the wrong-signed profile and the
#       eta-less profile are both far outside the tolerance;
#   (e) the validations: a `wall_t` on a resolved `:rigid` wall is an error (the
#       tangential load there would be silently dropped), an unbalanced
#       traction is *not* mis-detected as a normal-flux error (the normal
#       component is untouched), and a *normal*-traction request is refused
#       explicitly (stage 3 owns it).
#
# The measured errors are comments next to the assertions.  The tolerances are
# solver-accuracy thresholds, not discretisation errors: the Q2 field is exact
# and the quadrature is exact, so 1e-10 is the G238 precedent and the measured
# values are two to four orders better.

using Test, QuaST

# The suite's small unit box: 16 square leaves (4x4), 8x8 Q2 elements, 81 nodes.
function g259_tree()
    qt = Quadtree(2)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

g259_zero(x, y) = 0.0

# Outward boundary flux `int_dOmega u.n dS` of a discrete velocity, integrated
# with the same 1-D Gauss rule the assembly uses (exact for the Q2 trace).  The
# exact Couette field has zero net flux, and the discrete continuity equation
# with a constant pressure test function enforces it, so this is the mass/flux
# conservation check of the loaded solve.
function g259_boundary_flux(ss, qt, u)
    sq = ss.su
    n = sq.p + 1
    nu = sq.N
    fd = make_fedata(ss.p, ss.q)
    flux = 0.0
    for lf in sq.leaves
        hx = QuaST.leaf_hx(sq, lf[1])
        hy = QuaST.leaf_hy(sq, lf[1])
        ids = sq.elem[lf]
        for dir in (:xminus, :xplus, :yminus, :yplus)
            QuaST.has_no_face_neighbor(qt, lf, dir) || continue
            if dir === :xminus || dir === :xplus
                iF = dir === :xminus ? 1 : n
                sgn = dir === :xplus ? 1.0 : -1.0
                for q in 1:fd.nq
                    val = 0.0
                    for j in 1:n
                        val += u[ids[(j - 1) * n + iF]] * fd.phi_v[j, q]
                    end
                    flux += sgn * (0.5 * hy * fd.gl_w[q]) * val
                end
            else
                jF = dir === :yminus ? 1 : n
                sgn = dir === :yplus ? 1.0 : -1.0
                for q in 1:fd.nq
                    val = 0.0
                    for i in 1:n
                        val += u[nu + ids[(jF - 1) * n + i]] * fd.phi_v[i, q]
                    end
                    flux += sgn * (0.5 * hx * fd.gl_w[q]) * val
                end
            end
        end
    end
    return flux
end

@testset "G259 traction-driven Couette (sign gate)" begin
    # eta0 != 1 on purpose: the exact slope is tau0/eta0, so a load that carries
    # an extra (or a missing) eta factor, or an implementation that prescribes
    # the *velocity* tau0, cannot reproduce it.
    eta0 = 3.0
    tau0 = 2.0
    U = tau0 / eta0
    z0 = g259_zero
    qt = g259_tree()
    ss = make_semstokes(qt, 2)
    sq = ss.su
    N = sq.N
    xs = [c[1] for c in sq.coords]
    ys = [c[2] for c in sq.coords]
    near(a, b) = abs(a - b) < 1.0e-12
    on_top(i) = near(ys[i], 1.0)
    on_bottom(i) = near(ys[i], 0.0)
    on_side(i) = near(xs[i], 0.0) || near(xs[i], 1.0)
    eta1 = (x, y) -> eta0
    tau = (x, y) -> tau0

    # ---- (a) top wall :free with wall_t = tau0; sides+bottom :rigid --------
    # Sides `:rigid` (not `:free`): the Couette field carries a non-zero shear on
    # the side walls, which a zero-tangential-traction `:free` wall would not
    # admit (this is exactly why G238 uses `bc = :rigid`); their normal value is
    # the exact `u_x` profile, their tangential value is zero.
    wall_un_top = ((x, y) -> U * y, (x, y) -> U * y, z0, z0)
    r = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                 bc=:free,
                                 wall_bc=(:rigid, :rigid, :rigid, :free),
                                 wall_un=wall_un_top,
                                 wall_t=(nothing, nothing, nothing, tau),
                                 linear=:direct)
    u = r[1]
    ux = u[1:N]
    uz = u[N + 1:end]
    p = r[2]
    err_top = maximum(abs.(ux .- U .* ys)) / U
    # measured 7.6e-15 (solver round-off; the Q2 field and the face load are
    # both integrated exactly, so 1e-10 is the G238 solver-accuracy precedent)
    @test err_top < 1.0e-10
    @test maximum(abs, uz) / U < 1.0e-10          # measured 7.7e-15
    @test maximum(abs, p) < 1.0e-10 * max(tau0, 1.0)  # measured 2.6e-12
    @test maximum(abs, ux) > 0.5 * U              # the flow is actually driven
    # no wrong-signed solution: the profile increases with y
    @test ux[argmax(ys)] > 0.5 * U && ux[argmin(ys)] < 0.5 * U

    # Cross-code-independent form: the velocity-driven Couette of G238
    # (constant eta = 1, `uDx = U*y` with `bc = :rigid`) is the same field.  The
    # two solves have different operators (eta0 vs 1), so this is a `1e-10`
    # comparison, not a bitwise one; measured 1.0e-14.
    rleg = solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0, z0, z0,
                                    (x, y) -> U * y, z0;
                                    bc=:rigid, linear=:direct)
    @test maximum(abs.(r[1] .- rleg[1])) / U < 1.0e-10
    @test maximum(abs.(r[2] .- rleg[2])) < 1.0e-10 * max(tau0, 1.0)

    # Mass/flux conservation: the exact field has `int u.n dS = 0` (the side
    # inflow balances the outflow, and the normal velocity vanishes on the top
    # and bottom walls), and the discrete continuity equations with the pinned
    # constant pressure enforce the same identity.  This closes the loop
    # "boundary load enters the RHS -> the solve still conserves mass".
    flux_top = g259_boundary_flux(ss, qt, u)
    @test abs(flux_top) < 1.0e-12 * max(U * 2.0 * (qt.Lx + 1.0), 1.0)
    # ... and the normal (non-tangential) boundary data really are the Couette
    # profile: the side-wall normal values are imposed exactly.
    for i in 1:N
        (on_side(i) && near(xs[i], 0.0)) && @test u[i] == U * ys[i]
        (on_side(i) && near(xs[i], 1.0)) && @test u[i] == U * ys[i]
    end

    # ---- (c) the load itself: total, support and exact Q2 nodal values -----
    # `assemble_only=true` returns the assembled RHS; with no body force it *is*
    # the traction load.  The 4-point Gauss rule is exact for `tau0*phi`, so the
    # nodal values are the analytic `int tau0 phi ds` (with `h = 1/4` the leaf
    # face: end node `tau0*h/6`, interior node `tau0*h*2/3`, shared node
    # `tau0*h/3`), and the total is `tau0*Lx`.
    asm = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                   bc=:free,
                                   wall_bc=(:rigid, :rigid, :rigid, :free),
                                   wall_un=wall_un_top,
                                   wall_t=(nothing, nothing, nothing, tau),
                                   linear=:direct, assemble_only=true)
    fv = asm.fv
    hface = QuaST.leaf_hx(sq, 2)                   # 4 leaves across a unit box
    @test isapprox(sum(fv[1:N]), tau0 * qt.Lx; rtol=1.0e-14)   # measured 2.0000000000000004
    @test sum(fv[N + 1:end]) == 0.0
    topnodes = [i for i in 1:N if on_top(i)]
    @test length(topnodes) == 9
    for i in 1:N
        on_top(i) || continue
        expected = if near(xs[i], 0.0) || near(xs[i], 1.0)
            tau0 * hface / 6.0
        elseif near(xs[i], 0.25) || near(xs[i], 0.75) || near(xs[i], 0.5)
            tau0 * hface / 3.0
        else
            tau0 * hface * (2.0 / 3.0)
        end
        @test isapprox(fv[i], expected; rtol=1.0e-14)
    end
    # non-vacuous: the load is nonzero and only on the loaded wall's rows
    @test count(!iszero, fv) == length(topnodes)
    @test all(i -> iszero(fv[i]), [i for i in 1:N if !on_top(i)])
    @test sum(fv[1:N]) > 0.5 * tau0

    # The x-wall swap: on a vertical wall `wall_t` is the global `t_z`, so the
    # load must appear on the `u_z` rows (`N+1:2N`), with total `tau0*height`.
    # (No solve: `assemble_only` shows the load vector; the configuration is
    # legal because xminus resolves to `:free`.)
    asmx = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                    bc=:free,
                                    wall_bc=(:free, :rigid, :rigid, :rigid),
                                    wall_t=(tau, nothing, nothing, nothing),
                                    linear=:direct, assemble_only=true)
    @test isapprox(sum(asmx.fv[N + 1:end]), tau0 * 1.0; rtol=1.0e-14)
    @test sum(asmx.fv[1:N]) == 0.0
    xmin_nodes = [i for i in 1:N if near(xs[i], 0.0)]
    @test length(xmin_nodes) == 9
    @test count(!iszero, asmx.fv) == length(xmin_nodes)   # all in the u_z rows

    # ---- (b) bottom-wall variant: the same positive wall_t, opposite shear --
    # With `tbar = sigma.n` and the outward normal `-e_y` on the bottom wall, a
    # positive global `t_x` there is `sigma_xy = -tau0`: the exact field is
    # `u = (tau0*(1-y)/eta0, 0)`.  This is the sign variant that a hidden
    # outward-normal sign table would flip.
    wall_un_bot = ((x, y) -> U * (1.0 - y), (x, y) -> U * (1.0 - y), z0, z0)
    r2 = solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                  bc=:free,
                                  wall_bc=(:rigid, :rigid, :free, :rigid),
                                  wall_un=wall_un_bot,
                                  wall_t=(nothing, nothing, tau, nothing),
                                  linear=:direct)
    ux2 = r2[1][1:N]
    uz2 = r2[1][N + 1:end]
    err_bot = maximum(abs.(ux2 .- U .* (1.0 .- ys))) / U
    @test err_bot < 1.0e-10                        # measured 5.4e-14
    @test maximum(abs, uz2) / U < 1.0e-10          # measured 1.4e-13
    @test maximum(abs, r2[2]) < 1.0e-10 * max(tau0, 1.0)  # measured 1.8e-11
    flux_bot = g259_boundary_flux(ss, qt, r2[1])
    @test abs(flux_bot) < 1.0e-12 * max(U * 2.0 * (qt.Lx + 1.0), 1.0)
    # the two variants are genuinely different fields (the same data on the
    # opposite wall flips the shear): a sign-blind implementation would make
    # them equal
    @test maximum(abs.(ux .- ux2)) > 0.1 * U

    # ---- (d) the sign and the eta factor are discriminated -----------------
    # A flipped load would give `-tau0*y/eta0`; a missing eta factor in the
    # expected relation would give `tau0*y` (eta0 = 3 makes the three profiles
    # differ by O(1) relative to U).  Asserting that the gate would reject them
    # keeps the exact comparison from being vacuous.
    @test maximum(abs.(ux .+ U .* ys)) / U > 1.0          # wrong sign: error 2U
    @test maximum(abs.(ux .- tau0 .* ys)) / U > 1.0       # missing eta: error 2U

    # ---- (e) validations ---------------------------------------------------
    # A tangential traction on a resolved `:rigid` wall targets a Dirichlet
    # component: the load would be assembled and then dropped by `Cu'`, a silent
    # no-op, so both layers refuse it.
    @test_throws ArgumentError Params(bc=:rigid,
                                      wall_t=(nothing, nothing, tau, nothing))
    @test_throws ArgumentError Params(wall_bc=(:free, :free, :rigid, :free),
                                      wall_t=(nothing, nothing, tau, nothing))
    # ... and the same tuple is accepted on a `:free` wall (the default bc)
    @test Params(wall_t=(nothing, nothing, tau, nothing)) isa Params
    err = try
        solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                 bc=:free,
                                 wall_bc=(:free, :free, :free, :rigid),
                                 wall_t=(nothing, nothing, nothing, tau),
                                 linear=:direct)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    msg = err isa ArgumentError ? sprint(showerror, err) : ""
    @test occursin("tangential", msg) && occursin("Dirichlet", msg) &&
          occursin("silently dropped", msg)

    # A *normal*-traction request is stage 3 (it needs the `:open` wall and the
    # derived pressure gauge): the keyword exists only to refuse it loudly.
    @test_throws ArgumentError Params(wall_tn=(tau, nothing, nothing, nothing))
    err2 = try
        solve_semquad_stokes_var(ss, qt, eta1, z0, z0, z0, z0;
                                 bc=:free,
                                 wall_tn=(nothing, nothing, nothing, tau),
                                 linear=:direct)
        nothing
    catch e
        e
    end
    @test err2 isa ArgumentError
    msg2 = err2 isa ArgumentError ? sprint(showerror, err2) : ""
    @test occursin("normal traction", msg2) && occursin(":open", msg2)
    @test occursin("pressure gauge becomes derived", msg2)

    # The defaults of the public surface: `nothing` (zero load), four entries.
    @test Params().wall_t == (nothing, nothing, nothing, nothing)
    @test Params().wall_t === (nothing, nothing, nothing, nothing)
    prl = Params(wall_t=(nothing, nothing, nothing, (x, y) -> 3.5))
    @test prl.wall_t[4](0.3, 0.7) == 3.5
    @test_throws Exception Params(wall_t=(nothing, nothing, nothing))
end
