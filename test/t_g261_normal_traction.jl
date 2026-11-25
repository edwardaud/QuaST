# G261: a prescribed *normal traction* on an `:open` wall -- the sign gate of the
# boundary-condition feature's stage 3 (G258 was stage 1's per-wall normal
# velocity, G259/G260 stage 2's tangential traction).
#
# Stage 3 adds `Params.wall_tn` and the `:open` wall.  An `:open` wall has **no**
# Dirichlet component: the natural condition of the weak form `sigma.n = 0`
# applies, and `wall_tn` prescribes the normal component of the **total** Cauchy
# traction `tbar = sigma.n`.  The boundary term of the weak form,
#
#     int 2 eta eps(u):eps(v) - int p div v = int f.v + int_{dOmega} (sigma.n).v dS,
#
# is a right-hand-side load (`fv_i += int tbar.phi_i dS`, plus sign) exactly as
# in stage 2; the new piece is that the load now also lands on the wall's
# *normal* component rows (`u_z` on a y-wall, `u_x` on an x-wall), and that the
# data are the global component in the positive coordinate direction, with the
# physical outward normal inside `tbar = sigma.n`.  Consequence, and the whole
# point of this gate: on the top wall (`n = +e_y`) `wall_tn = T` means
# `sigma_yy = T`, so with `u = 0` (no body force) the exact solution is
# `p = -T`; on the bottom wall (`n = -e_y`) the *same* data `T` means
# `sigma_yy = -T`, so the exact solution is `p = +T`; on `xminus` (`n = -e_x`)
# `wall_tn = T` is `sigma_xx = T` -> `p = +T`, and on `xplus` it is `p = -T`.
#
# **Why the pressure pin must be off.**  Today's solver pins one nodal pressure
# dof to zero and drops its continuity equation, which is exact only when the
# normal velocity is Dirichlet on the whole boundary (the constant pressure is
# then in the null space of the reduced gradient).  With an open wall the
# constant is not in that null space: a pin would replace a continuity equation
# by `p_a = 0` and the returned field would have to carry a zero at one node,
# i.e. it could not be the constant `-T` (the max-norm check below is O(T), not
# round-off, for the pinned system).  The gate asserts the derived gauge
# (`:none`) and the unpinned `C_p` (`np` columns), so the verified answer is the
# answer of the traction problem, not of a pinned variant.
#
# Pinned by this gate:
#   (a) the rigid box with the top `:open` and `wall_tn = T`: `u = 0`, `p = -T`,
#       with the gauge/C_p structural checks above;
#   (b) linearity/antisymmetry: `wall_tn = -T` gives `p = +T` (no sign is
#       hard-wired in the expected value);
#   (c) the body-force hydrostatic solution with an open top and zero normal
#       traction: `u = 0`, `p = 1 - y` (this fails if the pin is on, and it
#       fails if only the deviatoric stress is imposed, because with `u = 0` the
#       deviatoric normal traction is 0 and the pressure level would be free);
#   (d) the bottom-wall and the two side-wall sign variants above, plus the
#       claim that the two top/bottom answers are genuinely different;
#   (e) the load itself: total `int tbar dS` on the normal rows and zero on the
#       other component (top y-wall -> `u_z`; x-wall -> `u_x`), from
#       `assemble_only=true`;
#   (f) the validation: `wall_tn` on a resolved `:free`/`:rigid` wall is an
#       `ArgumentError` in `Params` and in the solver (the load would be silently
#       dropped -- the normal velocity is Dirichlet there), with the documented
#       message; and the public default is all `nothing`.
#
# The measured errors are in the comments.  Tolerances are solver-accuracy
# thresholds: with `u = 0` the Q_2 velocity is exact, the pressure is constant or
# linear (Q_1 exact), the constant tractions are integrated exactly by the
# `p+2`-point face rule, so only the linear solve is left (G238/G259 precedent:
# 1e-10; measured values are three orders better).

using Test, QuaST

# The suite's small unit box: Quadtree(2) refined twice -> 16 square leaves,
# 8x8 Q2 elements, 81 velocity nodes, 25 Q1 pressure nodes.
function g261_tree()
    qt = Quadtree(2)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

g261_zero(x, y) = 0.0

@testset "G261 normal traction on an open wall (sign gate)" begin
    qt = g261_tree()
    ss = make_semstokes(qt, 2)
    nu = ss.su.N
    np = ss.sp.N
    xs = [c[1] for c in ss.su.coords]
    ys = [c[2] for c in ss.su.coords]
    zp = [c[2] for c in ss.sp.coords]
    z0 = g261_zero
    eta1 = (x, y) -> 1.0
    near(a, b) = abs(a - b) < 1.0e-12

    function g261_solve(; wall_bc, wall_tn, fx=z0, fz=z0, kwargs...)
        return solve_semquad_stokes_var(ss, qt, eta1, fx, fz, z0, z0;
                                        bc=:free, wall_bc=wall_bc,
                                        wall_tn=wall_tn, linear=:direct,
                                        kwargs...)
    end

    # ---- (a) rigid box, top :open, wall_tn = T -> u = 0, p = -T ------------
    # n = +e_y on the top wall, tbar = sigma.n = T e_y, sigma_yy = T and with
    # u = 0 sigma_yy = -p, so p = -T everywhere (momentum with u = 0 forces a
    # constant pressure).  This is the total Cauchy traction: the pressure is
    # part of the data, so the answer is NOT p = 0.
    T = 2.5
    top_open = (:rigid, :rigid, :rigid, :open)
    r = g261_solve(wall_bc=top_open,
                   wall_tn=(nothing, nothing, nothing, (x, y) -> T))
    u = r[1]; p = r[2]
    @test maximum(abs, u) < 1.0e-12              # measured 3.59e-16
    @test maximum(abs.(p .+ T)) / T < 1.0e-10     # measured 2.18e-14 / T
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-12     # measured 5.07e-15
    # non-vacuous: the pressure is the data, not the pinned zero (a pinned gauge
    # would give |p| <= T at the pin and fail the max-norm test by O(T)); and it
    # is constant, which is the physical statement u = 0 at every node.
    @test maximum(abs, p) > 0.5 * T
    @test all(abs.(p .+ T) .< 1.0e-10 * T)
    # the pressure gauge is derived: no pin, and C_p has np columns (a pinned
    # closed-boundary solve has np - 1).  This is the structural proof that the
    # answer above is not the pinned variant.
    @test LAST_SEMSTOKES_PRESSURE_GAUGE[] === :none
    asm = g261_solve(wall_bc=top_open,
                     wall_tn=(nothing, nothing, nothing, (x, y) -> T),
                     assemble_only=true)
    @test size(asm.Cp, 2) == np
    @test all(iszero, asm.u0) && all(iszero, asm.p0)   # data, not Dirichlet values

    # ---- (b) linearity of the gauge in the data ---------------------------
    rneg = g261_solve(wall_bc=top_open,
                      wall_tn=(nothing, nothing, nothing, (x, y) -> -T))
    @test maximum(abs, rneg[1]) < 1.0e-12
    @test maximum(abs.(rneg[2] .- T)) / T < 1.0e-10
    @test maximum(abs.(r[2] .+ rneg[2])) / T < 1.0e-10
    @test maximum(abs.(r[2] .- rneg[2])) > T      # the two answers differ by 2T

    # ---- (c) body force + open top with t_n = 0 -> p = 1 - y --------------
    # -div sigma = f with u = 0 gives grad p = f = (0, -1): p = 1 - y, and at
    # the top p(1) = 0 satisfies the natural condition sigma.n = -p n = 0, i.e.
    # wall_tn = 0 (the default: no data).  A pin would force p = 0 at one node
    # and break the traction condition; a merely deviatoric traction would leave
    # the pressure level undetermined because the deviatoric part of u = 0 is 0.
    rp = g261_solve(wall_bc=top_open, wall_tn=(nothing, nothing, nothing, nothing),
                    fz=(x, y) -> -1.0)
    @test maximum(abs, rp[1]) < 1.0e-12           # measured 4.88e-17
    @test maximum(abs.(rp[2] .- (1.0 .- zp))) < 1.0e-10   # measured 2.44e-15
    @test abs(rp[2][argmax(zp)] ) < 1.0e-10       # p(top) = 0 (t_n = 0 there)
    @test abs(rp[2][argmin(zp)] - 1.0) < 1.0e-10  # p(bottom) = 1
    @test LAST_SEMSTOKES_PRESSURE_GAUGE[] === :none

    # ---- (d) the outward-normal sign variants -----------------------------
    # The same positive data on the *bottom* wall is sigma_yy = -T (outward
    # normal -e_y), so the exact pressure is +T -- a reversed sign vs the top.
    rb = g261_solve(wall_bc=(:rigid, :rigid, :open, :rigid),
                    wall_tn=(nothing, nothing, (x, y) -> T, nothing))
    @test maximum(abs, rb[1]) < 1.0e-12           # measured 3.99e-16
    @test maximum(abs.(rb[2] .- T)) / T < 1.0e-10  # measured 2.66e-15 / T
    @test maximum(abs.(r[2] .- rb[2])) > 1.5 * T  # genuinely opposite answers

    # ... and on the vertical walls: xminus (n = -e_x) -> p = +T, xplus -> p = -T
    rxm = g261_solve(wall_bc=(:open, :rigid, :rigid, :rigid),
                     wall_tn=((x, y) -> T, nothing, nothing, nothing))
    rxp = g261_solve(wall_bc=(:rigid, :open, :rigid, :rigid),
                     wall_tn=(nothing, (x, y) -> T, nothing, nothing))
    @test maximum(abs, rxm[1]) < 1.0e-12
    @test maximum(abs, rxp[1]) < 1.0e-12
    @test maximum(abs.(rxm[2] .- T)) / T < 1.0e-10   # measured 7.28e-15 / T
    @test maximum(abs.(rxp[2] .+ T)) / T < 1.0e-10   # measured 3.38e-15 / T
    @test maximum(abs.(rxm[2] .- rxp[2])) > 1.5 * T
    # the x-wall load goes to the u_x rows (its normal component), not to u_z
    asmx = g261_solve(wall_bc=(:open, :rigid, :rigid, :rigid),
                      wall_tn=((x, y) -> T, nothing, nothing, nothing),
                      assemble_only=true)
    @test isapprox(sum(asmx.fv[1:nu]), T * 1.0; rtol=1.0e-14)   # height = 1
    @test all(iszero, asmx.fv[nu + 1:end])

    # ---- (e) the load itself: total and support ---------------------------
    # Top y-wall: wall_tn is the global t_z, so exactly the u_z rows receive
    # int tbar phi dS; sum = int_0^Lx (1 + x/2) dx = 1.25 on this unit box.
    chi = (x, y) -> 1.0 + 0.5 * x
    asmt = g261_solve(wall_bc=top_open,
                      wall_tn=(nothing, nothing, nothing, chi),
                      assemble_only=true)
    @test isapprox(sum(asmt.fv[nu + 1:end]), 1.25; rtol=1.0e-14)
    @test all(iszero, asmt.fv[1:nu])
    @test count(!iszero, asmt.fv) > 0
    # the loaded face's Q2 nodal values are int tbar phi ds (the (h/2)*gl_w*phi
    # weight, not a nodal collocation): with h = 1/4 per leaf face, an end node
    # gets tbar*h/6, an interior node tbar*h*2/3 and a shared node tbar*h/3.
    hface = 1.0 / 4.0
    topnodes = [i for i in 1:nu if near(ys[i], 1.0)]
    @test length(topnodes) == 9
    for i in topnodes
        expected = if near(xs[i], 0.0) || near(xs[i], 1.0)
            chi(xs[i], 1.0) * hface / 6.0
        elseif near(xs[i], 0.25) || near(xs[i], 0.5) || near(xs[i], 0.75)
            chi(xs[i], 1.0) * hface / 3.0
        else
            chi(xs[i], 1.0) * hface * (2.0 / 3.0)
        end
        @test isapprox(asmt.fv[nu + i], expected; rtol=1.0e-14)
    end
    @test count(!iszero, asmt.fv) == length(topnodes)

    # ---- (f) a normal traction on a non-open wall is refused --------------
    # The normal velocity is Dirichlet on a :free/:rigid wall, so the assembled
    # load would be dropped by Cu' -- a silent no-op.  Both layers refuse it,
    # and the message must name the condition and the fix (:open, §3.5) so a
    # stage-2 user learns what changed.
    @test_throws ArgumentError Params(bc=:free,
                                      wall_tn=(nothing, nothing, nothing,
                                               (x, y) -> T))
    @test_throws ArgumentError Params(bc=:free, wall_bc=(:rigid, :rigid, :rigid,
                                                         :rigid),
                                      wall_tn=((x, y) -> T, nothing, nothing,
                                               nothing))
    err = try
        g261_solve(wall_bc=top_open,
                   wall_tn=(nothing, nothing, (x, y) -> T, nothing))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    msg = err isa ArgumentError ? sprint(showerror, err) : ""
    @test occursin("normal traction", msg)
    @test occursin(":open", msg)
    @test occursin("pressure gauge becomes derived", msg)
    # ... and the accepted case really is accepted by Params
    @test Params(bc=:free, wall_bc=top_open,
                 wall_tn=(nothing, nothing, nothing, (x, y) -> T)) isa Params

    # the public defaults
    @test Params().wall_tn == (nothing, nothing, nothing, nothing)
    @test Params().wall_tn === (nothing, nothing, nothing, nothing)
    prtn = Params(bc=:free, wall_bc=top_open,
                  wall_tn=(nothing, nothing, nothing, (x, y) -> 3.5))
    @test prtn.wall_tn[4](0.3, 0.7) == 3.5
    @test_throws Exception Params(wall_tn=(nothing, nothing, nothing))
end
