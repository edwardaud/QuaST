# G264: the combined regional-channel case -- a prescribed normal-velocity
# inflow, a no-slip channel, and an `:open` outflow carrying the exact total
# traction, all in one solve (boundary-condition feature, stage 4).
#
# The three per-wall mechanisms the earlier stages added are exercised together
# here; each stage's own gate used them separately (G258 normal velocity, G259
# tangential traction, G261/G262/G263 the open wall and its normal traction).
# The configuration is a plane channel `[0, L] x [0, H]` with square leaves:
#
#   yminus / yplus   :rigid  (no slip)
#   xminus           :rigid  with `wall_un` = the exact inflow profile `u_x(y)`
#   xplus            :open   with `wall_tn` = -p(x_end) and `wall_t` = eta u_x'(y)
#
# and the exact solution is plane Poiseuille.  Two driving modes are tested:
#
#   (a) body force `f_x = G`, zero pressure gradient: `p = p0` (constant), so the
#       outflow normal traction is `wall_tn = -p0` and
#           u_x = G y (H - y) / (2 eta),  u_y = 0,  p = p0;
#   (b) no body force, pressure gradient `dp/dx = -G`:
#           u_x = G y (H - y) / (2 eta),  u_y = 0,  p = p_out + G (L - x),
#       so `wall_tn = -p(L) = -p_out` and the same `u_x` profile and shear.
#
# In both modes `sigma·n` on the outflow (`n = +e_x`) is
# `(sigma_xx, sigma_xy) = (-p, eta du_x/dy)`, with the normal part `-p0` in (a)
# and `-p_out` in (b) and the tangential part `G (H/2 - y)` in both; the shear
# is eta-independent at constant eta.
#
# **What the task brief got wrong, and what this gate implements.**  The stage
# brief states the exact solution as "body force G ... p(x,z) = -G x + p0" and
# then the outflow traction `(-p, eta u')`.  Those two cannot both hold in this
# code's sign convention: the solve is `-div sigma = f` with `sigma = -p I +
# 2 eta eps`, so a constant body force `f_x = G` balances `-eta du/dy` and the
# exact pressure is *constant* (G261's `f_z = -1, p = 1 - y` is the same
# relation); a pressure gradient `dp/dx = -G` with no body force produces the
# same velocity profile.  Both are classic plane Poiseuille and both are tested
# here, so the brief's `p = -G x + p0` appears in (b) and its "constant body
# force" in (a).  The statement `wall_tn = -p(x_end)` is correct in both.
#
# Q2 contains the quadratic velocity and Q1 the linear/constant pressure
# exactly; the face quadrature is exact for the polynomial traction; thus the
# field errors are linear-solve round-off (the G238/G259 1e-10 precedent) and
# not discretisation errors.
#
# Pinned by this gate:
#   (a) body-force channel: field errors, constant pressure (zero gradient),
#       per-wall mass fluxes (inflow = outflow = G H^3/(12 eta), no-slip walls
#       zero, net zero), exact Dirichlet data, the derived (unpinned) gauge, and
#       the assembled load total = `G L H - p0 H` on the u_x rows and zero net
#       tangential load on the u_z rows;
#   (b) pressure-driven channel: the same plus the exact pressure gradient
#       `p(L) - p(0) = -G L` and `wall_tn = -p_out`;
#   (c) the inflow mechanism: the new per-wall `wall_un` API and the legacy
#       global `uDx` closure produce the identical `fixed` dict and a bitwise
#       identical solution; a `:free` (zero tangential traction) upstream wall
#       is a *different* problem -- the Poiseuille inflow carries the shear
#       `eta u'`, so the brief's "inflow normal velocity vs :rigid" comparison
#       only holds through the no-slip route, and the gate proves the :free
#       route is far outside the exact comparison;
#   (d) the free-slip outflow variant (`wall_t = nothing`): the exact solution
#       is *not* Poiseuille (the exact outflow shear `eta u'` is non-zero), so
#       the gate derives its required properties instead: the same Dirichlet
#       data and flux hold exactly; the solution is exactly linear in the
#       withheld-traction scale lambda (u(1/2) = (u(0)+u(1))/2 to round-off);
#       the difference `w = u_free - u_full` is exactly the linear response to
#       the withheld tangential load in the reduced saddle system
#       (`Kred dw + Bt dp = -Cu' L_t`, `Bs dw = 0`; its Dirichlet lift cancels),
#       it satisfies the energy identity `w' K w = -L_t' w`, and its strain norm
#       is under the derived energy scale `L ||eta u'||^2 / (2 eta^2)`, with the
#       channel trace estimate `||w_y||_L2(outflow) <= sqrt(2 L) ||eps(w)||`
#       verified on the field.  The measured deviation energy is 2.4% of the
#       scale, so the acceptance is a bounded, localised departure;
#   (e) direct vs multigrid: the design's G-path goal (1e-6 relative) on the
#       open channel at the 512-leaf tree;
#   (f) free-slip channel walls: the design's §5.6 plug flow (u = (U, 0), p = 0)
#       with a `:free` inflow wall, `:free` channel walls and an `:open` outflow
#       at zero normal traction -- the tangentially-free corner of the
#       composition matrix, and the one case where the inflow wall may be
#       `:free` (a uniform field has no shear to sustain);
#   (g) the example script `examples/channel_flow_2d.jl` is included and run
#       through `run_semquad_simulation`; its printed profile error and its
#       returned state are checked against the exact solution, and against the
#       gate's own recomputation and flux, so the demonstration is exercised,
#       not just documented.
#
# The measured numbers are comments next to the assertions.
using Test, QuaST, LinearAlgebra

# The demonstration script is *exercised*, not just documented: include it here
# (its `main` is guarded by `PROGRAM_FILE`, so nothing runs on include) and call
# `channel_demo` in section (g).
include(joinpath(@__DIR__, "..", "examples", "channel_flow_2d.jl"))

# ---------------------------------------------------------------------------
# The channel setup and its helpers.
# ---------------------------------------------------------------------------
const G264_G = 4.0        # body force / pressure-gradient magnitude
const G264_ETA = 2.0
const G264_H = 1.0
const G264_L = 2.0
const G264_P0 = 1.5       # body-force mode: p = p0 (and wall_tn = -p0)
const G264_POUT = 2.5     # pressure-driven mode: p(L) = p_out (wall_tn = -p_out)

g264_zero(x, y) = 0.0
g264_profile(y) = G264_G * y * (G264_H - y) / (2.0 * G264_ETA)
g264_shear(y) = G264_G * (G264_H / 2.0 - y)   # eta du/dy, eta-independent here
g264_umax() = G264_G * G264_H^2 / (8.0 * G264_ETA)   # 0.25
g264_flux() = G264_G * G264_H^3 / (12.0 * G264_ETA)  # 1/6

# `L x H` box of square leaves: a 4 x 2 base grid, `n` uniform refinements.
function g264_tree(n::Int)
    qt = Quadtree(n; Lx=G264_L, cells_x=4, cells_y=2)
    for _ in 1:n, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

# Outward boundary flux per wall, in the order (xminus, xplus, yminus, yplus),
# integrated with the assembly's own 1-D Gauss rule (exact for the Q2 trace).
function g264_flux_per_wall(ss, qt, u)
    su = ss.su
    n = su.p + 1
    nu = su.N
    fd = make_fedata(ss.p, ss.q)
    out = zeros(Float64, 4)
    for it in eachindex(su.leaves)
        lf = su.leaves[it]
        hx = QuaST.leaf_hx(su, lf[1]); hy = QuaST.leaf_hy(su, lf[1])
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

# L2 norm of u_y on the outflow face x = L, by the same 1-D Gauss rule.
function g264_outflow_unorm(ss, qt, u)
    su = ss.su
    n = su.p + 1
    nu = su.N
    fd = make_fedata(ss.p, ss.q)
    acc = 0.0
    for it in eachindex(su.leaves)
        lf = su.leaves[it]
        hy = QuaST.leaf_hy(su, lf[1])
        ids = su.elem[lf]
        QuaST.has_no_face_neighbor(qt, lf, :xplus) || continue
        for q in 1:fd.nq
            val = 0.0
            for j in 1:n
                val += u[nu + ids[(j - 1) * n + n]] * fd.phi_v[j, q]
            end
            acc += 0.5 * hy * fd.gl_w[q] * val^2
        end
    end
    return sqrt(acc)
end

# The combined channel solve.  `mode = :body` (f_x = G, p = p0) or
# `mode = :pressure` (f = 0, p = p_out + G (L - x)).  `inflow = :api` uses the
# per-wall `wall_un`; `inflow = :legacy` uses the positional `uDx` closure;
# `inflow = :free` resolves the upstream wall to a zero-tangential-traction
# `:free` wall (the trap case of (c)).  `lambda` scales the outflow tangential
# traction (1 = the exact traction, 0 = free slip).
const G264_WALL_BC = (:rigid, :open, :rigid, :rigid)

function g264_channel(ss, qt, mode::Symbol; inflow::Symbol=:api,
                      lambda::Float64=1.0, p0::Float64=G264_P0,
                      pout::Float64=G264_POUT, linear::Symbol=:direct,
                      wall_bc=G264_WALL_BC, kwargs...)
    fx = mode === :body ? (x, y) -> G264_G : g264_zero
    tn = mode === :body ? -p0 : -pout
    wall_tn = (nothing, (x, y) -> tn, nothing, nothing)
    wall_t = lambda == 0.0 ? (nothing, nothing, nothing, nothing) :
             (nothing, (x, y) -> lambda * g264_shear(y), nothing, nothing)
    # `inflow = :legacy` drops the per-wall normal value in favour of the
    # positional `uDx` closure; `:free` keeps the normal value but resolves the
    # upstream wall to a zero-tangential-traction wall.
    un = inflow == :legacy ?
         (nothing, nothing, nothing, nothing) :
         ((x, y) -> g264_profile(y), nothing, nothing, nothing)
    wbc = inflow == :free ? (:free, :open, :rigid, :rigid) : wall_bc
    # The legacy route: the upstream normal value comes from the positional
    # `uDx` closure, which the y-walls also use for their tangential value --
    # at y = 0, H it is `profile(0) = profile(H) = 0`, so the no-slip data are
    # unchanged.  `:free` keeps the normal value on the wall (free slip is a
    # normal-Dirichlet/tangential-natural wall).
    udx = inflow == :legacy ? (x, y) -> g264_profile(y) : g264_zero
    return solve_semquad_stokes_var(ss, qt, (x, y) -> G264_ETA, fx, g264_zero,
                                    udx, g264_zero;
                                    bc=:free, wall_bc=wbc, wall_un=un,
                                    wall_t=wall_t, wall_tn=wall_tn,
                                    linear=linear, kwargs...)
end

@testset "G264 channel flow: normal-velocity inflow, open outflow, exact traction" begin
    umax = g264_umax()
    Qex = g264_flux()
    qt = g264_tree(2)
    ss = make_semstokes(qt, 2)
    nu = ss.su.N
    np = ss.sp.N
    xs = [c[1] for c in ss.su.coords]
    ys = [c[2] for c in ss.su.coords]
    pxs = [c[1] for c in ss.sp.coords]
    near(a, b) = abs(a - b) < 1.0e-12

    # ---- (a) body-force channel: f_x = G, p = p0 ---------------------------
    # The body force balances the viscous term, so the exact pressure gradient
    # is zero and the outflow normal traction is -p0.  This is the brief's
    # "no-slip channel driven by a constant body force with an open outflow".
    rb = g264_channel(ss, qt, :body)
    ub = rb[1]
    uxb = ub[1:nu]
    uzb = ub[nu + 1:end]
    pb = rb[2]
    errb_u = maximum(abs.(uxb .- g264_profile.(ys))) / umax
    errb_w = maximum(abs, uzb) / umax
    @test errb_u < 1.0e-10                 # measured 9.9e-15
    @test errb_w < 1.0e-10                 # measured 2.9e-14
    @test maximum(abs.(pb .- G264_P0)) < 1.0e-10          # measured 2.3e-13
    @test maximum(pb) - minimum(pb) < 1.0e-10             # zero pressure gradient
    @test LAST_SEMSTOKES_PRESSURE_GAUGE[] === :none
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-12             # measured 3.8e-14
    @test maximum(abs, ub) > 0.9 * umax    # the flow is actually driven
    @test maximum(abs, ub) < 1.1 * umax
    # the prescribed normal velocity is imposed exactly at every inflow node
    for i in 1:nu
        near(xs[i], 0.0) && @test ub[i] == g264_profile(ys[i])
    end
    # no-slip on the channel walls is imposed exactly
    for i in 1:nu
        (near(ys[i], 0.0) || near(ys[i], G264_H)) &&
            @test uxb[i] == 0.0 && uzb[i] == 0.0
    end
    # mass flux through each boundary: the inflow and outflow at Qex, the
    # no-slip walls exactly zero, the net by continuity zero
    flb = g264_flux_per_wall(ss, qt, ub)
    @test abs(flb[1] + Qex) < 1.0e-12 * max(Qex, 1.0)     # measured -Qex
    @test abs(flb[2] - Qex) < 1.0e-12 * max(Qex, 1.0)     # measured +Qex
    @test abs(flb[3]) < 1.0e-16 && abs(flb[4]) < 1.0e-16
    @test abs(sum(flb)) < 1.0e-12 * max(Qex, 1.0)         # measured 1.7e-16
    # the assembled load: the volume drive G over L x H plus the outflow normal
    # traction -p0 over H, on the u_x rows; the tangential load has zero net
    # because int_0^H G (H/2 - y) dy = 0 (but is not zero node by node)
    asb = g264_channel(ss, qt, :body; assemble_only=true)
    @test isapprox(sum(asb.fv[1:nu]),
                   G264_G * G264_L * G264_H - G264_P0 * G264_H; rtol=1.0e-14)
    @test abs(sum(asb.fv[nu + 1:end])) < 1.0e-14
    @test sum(asb.fv[1:nu]) > 0.0
    @test size(asb.Cp, 2) == np          # unpinned pressure space
    # the traction is *data* (it lands in fv); the Dirichlet lift u0 carries
    # only the prescribed inflow velocity and the pressure has no Dirichlet
    # data at all (the gauge is derived, so p0 is empty/zero)
    @test all(iszero, asb.p0)
    @test any(!iszero, asb.u0)

    # ---- (b) pressure-driven channel: f = 0, dp/dx = -G -------------------
    # Same velocity profile, driven by the pressure gradient instead; the
    # pressure is linear in x (Q1-exact) and the outflow holds p(L) = p_out.
    rp = g264_channel(ss, qt, :pressure)
    up = rp[1]
    uxp = up[1:nu]
    uzp = up[nu + 1:end]
    pp = rp[2]
    errp_u = maximum(abs.(uxp .- g264_profile.(ys))) / umax
    errp_w = maximum(abs, uzp) / umax
    pex = G264_POUT .+ G264_G .* (G264_L .- pxs)
    @test errp_u < 1.0e-10                 # measured 8.9e-15
    @test errp_w < 1.0e-10                 # measured 2.1e-14
    @test maximum(abs.(pp .- pex)) < 1.0e-10              # measured 2.9e-13
    # the pressure gradient is exactly -G: the wall-to-wall difference is -G L
    @test isapprox(pp[argmax(pxs)] - pp[argmin(pxs)], -G264_G * G264_L;
                   rtol=1.0e-12, atol=1.0e-10)           # measured 9.2e-14
    @test LAST_SEMSTOKES_PRESSURE_GAUGE[] === :none
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-12             # measured 3.7e-14
    flp = g264_flux_per_wall(ss, qt, up)
    @test abs(flp[1] + Qex) < 1.0e-12 * max(Qex, 1.0)
    @test abs(flp[2] - Qex) < 1.0e-12 * max(Qex, 1.0)
    @test abs(sum(flp)) < 1.0e-12 * max(Qex, 1.0)         # measured 1.4e-16
    asp = g264_channel(ss, qt, :pressure; assemble_only=true)
    @test isapprox(sum(asp.fv[1:nu]), -G264_POUT * G264_H; rtol=1.0e-14)
    @test abs(sum(asp.fv[nu + 1:end])) < 1.0e-14
    # the two driving modes share the velocity and differ only in the pressure:
    @test maximum(abs.(uxb .- uxp)) / umax < 1.0e-10
    @test maximum(abs.(pb .- pp)) > 1.0

    # ---- (c) the inflow mechanism ----------------------------------------
    # (c1) The new per-wall API and the legacy global closure write the same
    # Dirichlet values, so the `fixed` dict and the solution must be bitwise
    # identical: the stage-1 equivalence, re-pinned in the combined case.
    rleg = g264_channel(ss, qt, :pressure; inflow=:legacy)
    @test rleg[3] == rp[3]
    @test rleg[1] == rp[1] && rleg[2] == rp[2]
    # (c2) A `:free` inflow (normal velocity Dirichlet, zero tangential
    # traction) is a *different* problem: the Poiseuille inflow carries the
    # shear eta du/dy = G(H/2 - y), which a zero-tangential-traction wall does
    # not admit.  The design's "the wall must be :rigid, not :free" argument;
    # this is why the brief's "normal velocity vs :rigid" comparison is only
    # exact through the no-slip route.
    rfree = g264_channel(ss, qt, :pressure; inflow=:free)
    @test LAST_SEMSTOKES_PRESSURE_GAUGE[] === :none
    errfree = maximum(abs.(rfree[1][1:nu] .- g264_profile.(ys))) / umax
    @test errfree > 1.0e-2                 # measured 5.6e-2: a real difference
    # ... while the *normal* velocity on that wall is still imposed exactly:
    # the difference is the tangential condition, not a missing inflow.
    for i in 1:nu
        near(xs[i], 0.0) && @test rfree[1][i] == g264_profile(ys[i])
    end
    # (c3) the two mechanisms are separate components: wall_un is the wall's
    # normal velocity and needs a Dirichlet normal component (accepted on the
    # :rigid inflow wall), while a tangential *velocity* on the :open outflow
    # wall has no Dirichlet component and is refused; a tangential value on the
    # :rigid inflow wall (the no-slip u_y = 0) is legal.
    z0c = (x, y) -> 0.0
    @test Params(bc=:free, wall_bc=G264_WALL_BC,
                 wall_un=((x, y) -> g264_profile(y), nothing, nothing, nothing),
                 wall_u=(z0c, nothing, nothing, nothing),
                 wall_t=(nothing, (x, y) -> g264_shear(y), nothing, nothing),
                 wall_tn=(nothing, (x, y) -> -G264_POUT, nothing, nothing)) isa Params
    @test_throws ArgumentError Params(bc=:free, wall_bc=G264_WALL_BC,
                                      wall_u=(nothing, z0c, nothing, nothing))

    # ---- (d) the free-slip outflow variant --------------------------------
    # The exact Poiseuille field is *not* admissible with `wall_t = nothing`:
    # its outflow shear is `eta u' = G(H/2 - y) != 0`.  What is derived exactly
    # is the relation between the free-slip solution u_F and the full-traction
    # solution u_T (the exact Poiseuille, (b)): they share the operator and the
    # Dirichlet data, so `w = u_F - u_T` is driven *only* by the withheld
    # tangential load L_t:
    #     2 eta ||eps(w)||^2 = -int_{Gamma_out} t_P . w dS,  t_P = eta u',
    # i.e. `w' K w = -L_t' w` and `Cu' (K w + L_t) = 0` in the discrete system.
    # The gate accepts the variant on those exact identities, on the exact
    # data/flux, and on the derived energy scale `L ||t_P||^2 / (2 eta^2)`.
    r0 = g264_channel(ss, qt, :pressure; lambda=0.0)      # free-slip outflow
    r1 = rp                                              # exact full traction
    r05 = g264_channel(ss, qt, :pressure; lambda=0.5)
    @test LAST_SEMSTOKES_PRESSURE_GAUGE[] === :none
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-12
    # the Dirichlet data and the flux are unchanged
    for i in 1:nu
        near(xs[i], 0.0) && @test r0[1][i] == g264_profile(ys[i])
    end
    fl0 = g264_flux_per_wall(ss, qt, r0[1])
    @test abs(fl0[1] + Qex) < 1.0e-12 * max(Qex, 1.0)
    @test abs(fl0[2] - Qex) < 1.0e-12 * max(Qex, 1.0)
    @test abs(sum(fl0)) < 1.0e-12 * max(Qex, 1.0)
    # it is NOT the full-traction solution: the departure is real and O(Umax)
    w = r0[1] .- r1[1]
    @test maximum(abs, w) > 0.1 * umax     # measured 0.18 Umax
    @test maximum(abs, r0[1][1:nu]) < 2.0 * umax
    # exact linearity in the withheld-traction scale lambda
    @test maximum(abs.(r05[1] .- (0.5 .* r0[1] .+ 0.5 .* r1[1]))) < 1.0e-12 * umax
    # The withheld load and the assembled reduced saddle: with the same
    # constraint set and Dirichlet data, the difference of the two solutions is
    # exactly the linear response to `Lt` in the reduced saddle system,
    #     Kred dw + Bt dp = -Cu' Lt,   Bs dw = 0,
    # with dw, dp the free-coordinate differences (`w = Cu dw`, `p0_c - p1_c =
    # Cp dp`; the Dirichlet lift cancels).  This is the discrete form of "the
    # free-slip solution is the exact Poiseuille field minus the response to the
    # withheld shear traction" and fails if the load, the constraints or the
    # operator are wrong.
    asm0 = g264_channel(ss, qt, :pressure; lambda=0.0, assemble_only=true)
    asm1 = g264_channel(ss, qt, :pressure; lambda=1.0, assemble_only=true)
    Lt = asm1.fv .- asm0.fv
    dw = asm0.Cu \ w
    dp = asm0.Cp \ (r0[2] .- r1[2])
    scal = norm(asm0.Cu' * Lt)
    @test maximum(abs, Lt) > 0.0           # measured 0.146
    @test norm(asm0.Kred * dw + asm0.Bt * dp + asm0.Cu' * Lt) <= 1.0e-9 * scal
    @test norm(asm0.Bs * dw) <= 1.0e-9 * max(scal, 1.0)   # measured 4.4e-16
    # the same statement without the pressure multiplier: tested with `w`
    # itself (admissible test functions are discretely divergence-free), the
    # withheld load does negative work equal to the strain energy of `w`
    Kw = asm0.K * w
    @test norm(asm0.Cp' * asm0.Bc * w) <= 1.0e-9 * max(scal, 1.0)  # measured 4.4e-16
    @test dot(w, Kw) > 0.0                 # measured 3.2135e-2
    @test dot(Lt, w) < 0.0                 # measured -3.2135e-2
    @test isapprox(dot(w, Kw), -dot(Lt, w); rtol=1.0e-10, atol=1.0e-14)
    # the derived energy scale: t_P = G(H/2 - y) on the outflow, so
    # ||t_P||^2 = int_0^H G^2 (H/2 - y)^2 dy = G^2 H^3/12 and
    # ||eps(w)||^2 <= L ||t_P||^2 / (2 eta^2) = 1/3 (measured 8.03e-3, 2.4%).
    strain2 = dot(w, Kw) / (2.0 * G264_ETA)
    @test strain2 > 0.0
    tP2 = G264_G^2 * G264_H^3 / 12.0
    bound = G264_L * tP2 / (2.0 * G264_ETA^2)
    @test strain2 <= bound
    # the channel trace estimate used to derive it: w_y = 0 on the inflow and
    # both no-slip walls, so |w_y(L,y)|^2 <= L int (d_x w_y)^2, and the gate
    # verifies ||w_y||_L2(outflow) <= sqrt(2 L) ||eps(w)|| on the field
    # (measured 3.2e-2 against 1.8e-1).
    wy = g264_outflow_unorm(ss, qt, w)
    @test wy > 0.0
    @test wy <= sqrt(2.0 * G264_L) * sqrt(strain2)

    # ---- (e) direct vs multigrid on the open channel ----------------------
    # The design's G-path goal (<= 1e-6 relative) for the channel on the
    # 512-leaf tree (4 851 saddle dofs).
    qt3 = g264_tree(3)
    ss3 = make_semstokes(qt3, 2)
    rd = g264_channel(ss3, qt3, :pressure; linear=:direct)
    rmg = g264_channel(ss3, qt3, :pressure; linear=:mg,
                       mg_operator=:matrixfree, mg_threads=1, mg_tol=1.0e-10)
    @test LAST_SEMSTOKES_RESIDUAL[] <= 1.0e-9             # measured 4.7e-11
    @test norm(rd[1] - rmg[1]) / max(norm(rd[1]), 1.0) <= 1.0e-6  # measured 2.8e-10
    @test norm(rd[2] - rmg[2]) / max(norm(rd[2]), 1.0) <= 1.0e-6  # measured 2.1e-10
    errd = maximum(abs.(rd[1][1:ss3.su.N] .- g264_profile.([c[2] for c in ss3.su.coords]))) /
           g264_umax()
    @test errd < 1.0e-8                    # measured 1.1e-14

    # ---- (f) free-slip channel walls: the plug-flow limit ------------------
    # The design's §5.6 case: an inflow normal velocity on a `:free` upstream
    # wall, a `:open` outflow with zero normal traction, and `:free` channel
    # walls.  A uniform field carries no strain and no shear, so the exact
    # solution is u = (U, 0), p = 0 -- the plug-flow limit of §5.4, in which the
    # inflow wall *may* be `:free` because the shear it would have to sustain is
    # zero.  This is the one configuration in which all four walls are free in
    # their tangential direction, and it pins that `:free` + `:open` +
    # `wall_un` compose.
    U0 = 1.0
    rplug = solve_semquad_stokes_var(ss, qt, (x, y) -> G264_ETA, g264_zero,
                                     g264_zero, g264_zero, g264_zero;
                                     bc=:free,
                                     wall_bc=(:free, :open, :free, :free),
                                     wall_un=((x, y) -> U0, nothing, nothing,
                                              nothing),
                                     wall_tn=(nothing, (x, y) -> 0.0, nothing,
                                              nothing),
                                     linear=:direct)
    @test maximum(abs.(rplug[1][1:nu] .- U0)) < 1.0e-12   # measured 5.5e-14
    @test maximum(abs, rplug[1][nu + 1:end]) < 1.0e-12    # measured 4.5e-14
    @test maximum(abs, rplug[2]) < 1.0e-10                # measured 2.5e-12
    @test LAST_SEMSTOKES_PRESSURE_GAUGE[] === :none
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-12             # measured 7.0e-14
    flplug = g264_flux_per_wall(ss, qt, rplug[1])
    @test abs(flplug[1] + U0 * G264_H) < 1.0e-12          # inflow: U*H
    @test abs(flplug[2] - U0 * G264_H) < 1.0e-12          # measured 6.7e-15
    @test abs(sum(flplug)) < 1.0e-12                      # measured 6.7e-15

    # ---- (g) the example script, exercised -------------------------------
    # `examples/channel_flow_2d.jl` builds this channel as a card and runs it
    # through the driver; the gate includes the file and calls the same function
    # the script's `main` calls, then checks the reported error and the state
    # against its own recomputation and flux.
    demo_out = IOBuffer()
    demo = channel_demo(io=demo_out)
    line = String(take!(demo_out))
    @test demo.res.steps == 1
    @test occursin("[channel]", line)
    @test occursin("err_u=", line)
    # the script's exact solution: G = 2, eta = 1, H = 1, L = 2, p_out = 2
    demo_u(y) = 2.0 * y * (1.0 - y) / 2.0          # = y (1 - y)
    demo_umax = 2.0 * 1.0^2 / 8.0                  # 0.25
    demo_coords = demo.res.ss.su.coords
    demo_err = maximum(abs(demo.res.ux[i] - demo_u(demo_coords[i][2]))
                       for i in eachindex(demo_coords)) / demo_umax
    @test demo_err < 1.0e-10
    @test demo.err_u == demo_err                   # the script reports this exact value
    @test demo.err_uz < 1.0e-10
    @test demo.err_p < 1.0e-10
    @test maximum(abs, demo.res.ux) > 0.9 * demo_umax
    demo_px = [c[1] for c in demo.res.ss.sp.coords]
    @test maximum(abs.(demo.res.pv .- (2.0 .+ 2.0 .* (2.0 .- demo_px)))) < 1.0e-10
    @test LAST_SEMSTOKES_PRESSURE_GAUGE[] === :none
    # ... and the flux of the demo's state, with the gate's own integrator
    demo_flux = g264_flux_per_wall(demo.res.ss, demo.res.qt,
                                   vcat(demo.res.ux, demo.res.uz))
    @test abs(demo_flux[1] + 1.0 / 6.0) < 1.0e-12
    @test abs(demo_flux[2] - 1.0 / 6.0) < 1.0e-12
    @test abs(sum(demo_flux)) < 1.0e-12
end

