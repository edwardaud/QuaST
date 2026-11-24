# G274: the radial physics of the 2-D polar annulus (stage 0.5b1).
#
# Stage 0.5a built the `PolarAnnulus` geometry; this gate pins the physics wiring
# that runs on it:
#
#   (a) the conductive base state: the radial log profile
#       `Tbar(r) = ln(R1/r)/ln(R1/R0)` (H = 0) and its H-inclusive analogue are
#       the energy step's fixed point, the energy-assembly source is zero, and
#       the wall Nusselt pair is `Nu_inner = Nu_outer = 1` — for several
#       resolutions and polynomial orders.  The Nusselt normalisation is checked
#       against its documented scale-invariant form
#       `Nu_wall = r_wall |ln f| <−dT/dr>_wall` (ANNULUS_REFS §6.2) with a
#       non-zero manufactured `theta`, and against the published multipliers
#       `R0 |ln f|` / `R1 |ln f|`;
#
#   (b) internal heating (H != 0): the documented H-inclusive profile
#       `T = (1 - H(R1^2-R0^2)/4) ln(R1/r)/ln(R1/R0) + (H/4)(R1^2-r^2)` and its
#       derivative; the state is still an exact fixed point (the `+H` source is
#       absorbed by `lap(Tbar) = -H`); the per-wall reference flux
#       `Q_ref(r) = -2*pi*r dTbar/dr` satisfies `Q_ref(R1)-Q_ref(R0) = H*A`, and
#       the discrete wall fluxes of the total-temperature field close the same
#       identity (a genuine discretisation check, not the algebra);
#
#   (c) hydrostatic balance: the conduction state is force-free (the
#       perturbation buoyancy is `Ra theta e_r`, so `theta = 0` gives `u = 0`
#       and a constant dynamic pressure), and a constant manufactured
#       perturbation `theta = c` gives `u = 0` with the exact analytic pressure
#       `p = Ra c r + const` (integrated hydrostatic profile plus gauge
#       offset).  The gauge convention of a closed all-rigid boundary is
#       checked (`LAST_SEMSTOKES_PRESSURE_GAUGE == :pin`), so the absolute
#       level is not compared — only differences/offsets;
#
#   (d) an end-to-end `geom=:annulus` driver run: finite, writes 18-column
#       history rows, reproduces the diagnostics of its own final state, keeps
#       `Ra*W = U` and `int div u = 0`, checkpoints/restarts bitwise, and the
#       driver keeps every documented restriction (gap = 1, rigid walls,
#       insulation on the wedge cuts, no box-only diagnostics, no
#       reference-state/density hook, no `:periodic` seam);
#
#   (e) the closure wiring: the viscosity law's depth argument is `r - R0`, the
#       phase-transition depth/`z0` and phase buoyancy use the same height, the
#       latent/adiabatic source and the `H_half_life` heating-scale plumbing run
#       on the annulus, and the weak-zone (physical-position) multiplier is
#       accepted while the Cartesian side-temperature control is refused.
#
# The box path is *not* re-pinned here: its acceptance is the external stage-0
# protocol (the nine state-dump hashes unchanged from
# `out_julia/agent/b_rs.txt`, and the full gate suite green).  G274 only adds
# annulus numbers.
#
# WHY THE ASSERTIONS ARE NOT TAUTOLOGICAL (what each block catches):
#   * the base-state block compares the production accessors with an
#     independently written `C1 ln r + C2 - H r^2/4` form and with the published
#     ANNULUS_REFS digits (`Tbar(1.5)`, `dTbar/dr(R0)`, `Q_cond`, the two
#     multipliers); a swapped radius, a linear-in-y profile, a missing `|ln f|`
#     or a sign error fails.  The fixed-point check is the *energy step*, not
#     the profile: a wrong `Tbar`/`Tbar'`/`Tbar''` or a non-absorbed `H` leaves a
#     non-zero source and moves `theta` (the box quadratic in `y` would fail
#     this on the annulus);
#   * the Nusselt block evaluates the production wall derivative on a
#     manufactured `theta = a (r - R1)` (exactly representable: affine in the
#     radial logical coordinate) and compares with the analytic
#     `1 - a r_wall |ln f|`; the *ratio* `(Nu_in - 1)/(Nu_out - 1) = R0/R1` is
#     independent of the derivative and pins the per-wall radii.  The box
#     single-gradient convention would give `1/(R |ln f|) ~ 0.75/1.37`, so an
#     unscaled or commonly-scaled formula fails;
#   * with H != 0 the flux identity is checked twice: analytically from the
#     gate's own closed forms, and *discretely* by integrating the one-sided
#     mapped wall derivative of the nodal total temperature and comparing
#     `Q_out - Q_in` with `H*A` (the latter converges with p; the gate uses a
#     resolved tree so the error is far below the bound);
#   * the hydrostatic block's `theta = c` case has the exact analytic pressure
#     `Ra c r`; the old vertical-only load would give a pressure spread
#     `Ra c R1` and a non-zero velocity, so both the `u = 0` and the
#     `Ra c (R1-R0)` spread are non-trivial;
#   * the driver block recomputes every checked history column from the returned
#     state, and the identities `Ra*W = U` / `int div u = 0` are properties of
#     the discrete solution, not of the driver's bookkeeping; the
#     checkpoint/restart comparison is bitwise against an uninterrupted run;
#   * the closure block compares the depth-dependent law and the phase anomaly
#     with independently evaluated `h = r - R0` expressions at points where a
#     `y`-based evaluation would differ, and checks the heating-scale linearity
#     of the assembled source.
#
# The gate uses `linear=:direct` everywhere: it is the bitwise-reproducible
# path, so the checkpoint comparison can be exact.

using Test, QuaST, LinearAlgebra

const MS274 = QuaST

# ---------------------------------------------------------------------------
# fixtures and independent oracles
# ---------------------------------------------------------------------------

const G274_R0 = 1.22
const G274_R1 = 2.22
const G274_PHI = 0.5 * pi

g274_geom() = PolarAnnulus(G274_R0, G274_R1, G274_PHI)

# `Quadtree(level; ...)` starts at the base cells; refine to the requested
# uniform level.  The annulus ignores the tree's Lx (its logical domain is the
# unit square), and `cells_x = 1`, `cells_y = 4` gives one radial and four
# angular base cells (the benchmark wedge is 4-fold symmetric).
function g274_tree(level::Int)
    qt = Quadtree(level; cells_x=1, cells_y=4)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

function g274_space(level::Int, p::Int)
    qt = g274_tree(level)
    g = g274_geom()
    return g, qt, make_semstokes(qt, p; geom=g)
end

g274_lnr() = log(G274_R1 / G274_R0)

# Independent closed form: C1 ln r + C2 - H r^2/4 with the wall conditions.
function g274_C1(H::Float64)
    return (1.0 - 0.25 * H * (G274_R1^2 - G274_R0^2)) / log(G274_R0 / G274_R1)
end
function g274_Tbar(H::Float64, r::Float64)
    C1 = g274_C1(H)
    C2 = 0.25 * H * G274_R1^2 - C1 * log(G274_R1)
    return -0.25 * H * r * r + C1 * log(r) + C2
end
g274_dTdr(H::Float64, r::Float64) = -0.5 * H * r + g274_C1(H) / r

# The per-unit-length radial conduction flux, in the documented ANNULUS_REFS
# form `Q_ref(r) = pi H r^2 - 2 pi C1` (its section 6.5), times the wedge's
# angular fraction `Phi/(2 pi)` (the wedge's walls are arcs of length `Phi R`,
# so an absolute flux must carry the wedge fraction; the Nusselt ratio does
# not).  `g274_Qref` is the full-ring value the published numbers use.
g274_Qref(H::Float64, r::Float64) = pi * H * r * r - 2.0 * pi * g274_C1(H)
g274_Qref_wedge(H::Float64, r::Float64) = (G274_PHI / (2.0 * pi)) * g274_Qref(H, r)
g274_area() = 0.5 * G274_PHI * (G274_R1^2 - G274_R0^2)   # wedge area
g274_area_full() = pi * (G274_R1^2 - G274_R0^2)

# Manufactured perturbation that is affine in the radial logical coordinate
# (`r = R0 + h` is affine in S, so `a (r - R1)` is exactly representable).
function g274_linear_theta(ss, a::Float64)
    return [a * (hypot(x, y) - G274_R1) for (x, y) in ss.su.coords]
end

# The driver IC: a 4-fold perturbation vanishing on both circles.
function g274_theta0(ss)
    th = zeros(Float64, ss.su.N)
    for (gid, (x, y)) in enumerate(ss.su.coords)
        h = hypot(x, y) - G274_R0
        th[gid] = 0.05 * sinpi(h) * sin(4.0 * atan(y, x))
    end
    return th
end

# Independent area-weighted mean of the base state over the wedge (the angular
# integral cancels, so it is a 1-D radial integral).
function g274_mean_Tbar(H::Float64; n::Int=40001)
    s = 0.0
    for i in 1:n
        r = G274_R0 + (G274_R1 - G274_R0) * (i - 0.5) / n
        s += g274_Tbar(H, r) * r
    end
    dr = (G274_R1 - G274_R0) / n
    return s * dr / (0.5 * (G274_R1^2 - G274_R0^2))
end

# Independent volume integral of div(u) on the mapped leaves (G270 pattern).
function g274_div_volume(g, sq, ux, uz)
    p = sq.p
    fd = make_fedata(p, p - 1)
    s = 0.0
    for lf in sq.leaves
        m = leaf_metric(g, sq, lf, fd)
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = MS274.quad_measure(m, fd, qx, qy)
            duxdx = 0.0; duzdy = 0.0
            for j in 1:p+1, i in 1:p+1
                k = (j - 1) * (p + 1) + i
                gx, gz = basis_grad(m, fd, qx, qy, i, j)
                duxdx += ux[ids[k]] * gx
                duzdy += uz[ids[k]] * gz
            end
            s += w * (duxdx + duzdy)
        end
    end
    return s
end

@testset "G274 annulus radial physics (stage 0.5b1)" begin
    g = g274_geom()
    @test g.R1 - g.R0 == 1.0000000000000002   # the published gap, Float64
    lnr = g274_lnr()
    @test abs(lnr - 0.598656337139023) <= 1e-14
    dt = 1.0e-3                              # the driver/picard step of (d)/(e)

    # ======================================================================
    # (a) the conduction base state and the wall Nusselt normalisation (H = 0)
    # ======================================================================
    @testset "(a) conduction base state and Nu normalisation" begin
        pr0 = Params(Ra=1000.0, H=0.0, eta_law=:expT, eta_b=0.0, bc=:rigid)
        # the profile and its derivative, against the gate's independent form
        for r in (G274_R0, 1.3, 1.5, 1.72, 2.0, G274_R1)
            @test abs(MS274._annulus_Tbar(pr0, g, r) - g274_Tbar(0.0, r)) <=
                  1e-14
            @test abs(MS274._annulus_dTbar_dr(pr0, g, r) - g274_dTdr(0.0, r)) <=
                  1e-14
        end
        # the published ANNULUS_REFS §5.4 numbers (the conventions bible)
        @test abs(g274_Tbar(0.0, 1.5) - 0.6548700205) <= 1e-10
        @test abs(g274_Tbar(0.0, 1.72) - 0.4262594234) <= 1e-10
        @test abs(-g274_dTdr(0.0, G274_R0) - 1.3691864268) <= 1e-10
        @test abs(-g274_dTdr(0.0, G274_R1) - 0.7524357841) <= 1e-10
        @test abs(2.0 * pi / lnr - 10.495479488628) <= 1e-12
        # wall values and the per-wall reference gradients / multipliers
        @test MS274._annulus_Tbar(pr0, g, G274_R0) == 1.0
        @test abs(MS274._annulus_Tbar(pr0, g, G274_R1)) <= 1e-15
        bot_ref, top_ref = MS274.semquad_base_wall_refs(pr0, g)
        @test abs(1.0 / bot_ref - G274_R0 * lnr) <= 1e-13   # 0.7303607313096079
        @test abs(1.0 / top_ref - G274_R1 * lnr) <= 1e-13   # 1.3290170684486309
        @test abs(1.0 / bot_ref - 0.7303607313096079) <= 1e-13
        @test abs(1.0 / top_ref - 1.3290170684486309) <= 1e-13

        # the energy step leaves the analytic conduction state unchanged; the
        # assembly source is zero (the base curvature is absorbed exactly)
        for (level, p, dt) in ((2, 2, 1.0e-4), (3, 2, 1.0e-4), (2, 4, 1.0e-4),
                               (3, 3, 1.0e-5))
            g2, qt, ss = g274_space(level, p)
            sq = ss.su
            th0 = zeros(Float64, sq.N)
            u0 = zeros(Float64, sq.N)
            _, _, _, F = assemble_semquad_energy(sq, qt, pr0, u0, u0;
                                                 advection=:skew, theta=th0)
            @test maximum(abs, F) <= 1e-14
            th = solve_semquad_energy(sq, qt, pr0, th0, th0, u0, u0, dt, dt,
                                      true)
            @test maximum(abs, th) <= 1e-12
            nut, nub = semquad_nu_walls(sq, qt, th0, pr0)
            @test nut == 1.0 && nub == 1.0
            # the area-weighted mean of the full temperature is the analytic
            # radial mean of the log profile (a wrong measure or a missed
            # base state moves it).  The bound is the mapped mean quadrature's
            # truncation for `r ln r` at `fd = make_fedata(p, p-1)` (measured
            # ~1.5e-11 at p = 2 and below 1e-13 at p >= 3).
            @test abs(semquad_mean_T(sq, qt, th0, pr0) - g274_mean_Tbar(0.0)) <=
                  1e-9
        end

        # per-wall Nusselt normalisation on a manufactured theta = a (r - R1)
        g2, qt, ss = g274_space(2, 3)
        sq = ss.su
        a = 0.2
        th = g274_linear_theta(ss, a)
        nut, nub = semquad_nu_walls(sq, qt, th, pr0)
        @test abs(nut - (1.0 - a * G274_R1 * lnr)) <= 1e-11
        @test abs(nub - (1.0 - a * G274_R0 * lnr)) <= 1e-11
        @test abs((nub - 1.0) / (nut - 1.0) - G274_R0 / G274_R1) <= 1e-9
        # non-vacuous: the pair is genuinely different at the two walls
        @test abs(nut - nub) > 1e-2
    end

    # ======================================================================
    # (b) internal heating: the H-inclusive profile and the flux identity
    # ======================================================================
    @testset "(b) internal heating H != 0" begin
        H = 2.0
        prH = Params(Ra=1000.0, H=H, eta_law=:expT, eta_b=0.0, bc=:rigid)
        # the documented H-inclusive profile and derivative
        for r in (G274_R0, 1.35, 1.5, 1.72, 2.0, G274_R1)
            @test abs(MS274._annulus_Tbar(prH, g, r) - g274_Tbar(H, r)) <= 2e-14
            @test abs(MS274._annulus_dTbar_dr(prH, g, r) - g274_dTdr(H, r)) <=
                  2e-14
        end
        @test MS274._annulus_Tbar(prH, g, G274_R0) <= 1.0 + 1e-15
        @test abs(MS274._annulus_Tbar(prH, g, G274_R0) - 1.0) <= 1e-15
        @test abs(MS274._annulus_Tbar(prH, g, G274_R1)) <= 1e-15
        # `lap(Tbar) = -H` exactly, i.e. the mapped base-curvature source (which
        # the energy assembly adds as `lap(Tbar) + H`) is zero
        for (x, y) in ((1.5, 0.2), (-0.4, 1.9), (0.8, -1.2))
            @test MS274._mapped_base_curvature(prH, g, x, y) == 0.0
        end
        # the analytic flux identity Q_ref(R1) - Q_ref(R0) = H * A
        @test abs(g274_Qref(H, G274_R1) - g274_Qref(H, G274_R0) -
                  H * g274_area_full()) <= 1e-12
        @test abs(g274_Qref_wedge(H, G274_R1) - g274_Qref_wedge(H, G274_R0) -
                  H * g274_area()) <= 1e-13
        # the H-inclusive state is also an exact fixed point of the energy step
        for (level, p, dt) in ((2, 3, 1.0e-4), (3, 3, 1.0e-5))
            g2, qt, ss = g274_space(level, p)
            sq = ss.su
            th0 = zeros(Float64, sq.N)
            u0 = zeros(Float64, sq.N)
            _, _, _, F = assemble_semquad_energy(sq, qt, prH, u0, u0;
                                                 advection=:skew,
                                                 theta=th0)
            @test maximum(abs, F) <= 1e-14
            th = solve_semquad_energy(sq, qt, prH, th0, th0, u0, u0, dt, dt,
                                      true)
            @test maximum(abs, th) <= 1e-12
            nut, nub = semquad_nu_walls(sq, qt, th0, prH)
            @test nut == 1.0 && nub == 1.0
            # the documented per-wall closed identity with the production pair
            @test abs(nut * g274_Qref(H, G274_R1) -
                      nub * g274_Qref(H, G274_R0) - H * g274_area_full()) <=
                  1e-12
        end
        # discrete flux balance: the one-sided mapped wall derivative of the
        # nodal total temperature closes `Q_out - Q_in = H A` to discretisation
        # order (the profile is analytic but `ln r` is not a polynomial)
        g2, qt, ss = g274_space(3, 4)
        sq = ss.su
        Tf = [MS274.semquad_base_T(prH, g, x, y) for (x, y) in sq.coords]
        st_out, _ = MS274._semquad_wall_flux(g, sq, Tf, :xplus)
        st_in, _ = MS274._semquad_wall_flux(g, sq, Tf, :xminus)
        Qout = -st_out
        Qin = st_in
        @test abs(Qout - g274_Qref_wedge(H, G274_R1)) /
              g274_Qref_wedge(H, G274_R1) <= 1e-4
        @test abs(Qin - g274_Qref_wedge(H, G274_R0)) /
              g274_Qref_wedge(H, G274_R0) <= 1e-4
        @test abs((Qout - Qin) - H * g274_area()) / (H * g274_area()) <= 1e-5
        # the per-wall normalisation is only meaningful while the conductive
        # radial flux keeps one sign at both walls; H = 4 flips the inner one
        # and the diagnostic must refuse rather than divide by a sign change
        prbig = Params(Ra=1.0, H=4.0, eta_law=:expT, eta_b=0.0, bc=:rigid)
        bot_big, top_big = MS274.semquad_base_wall_refs(prbig, g)
        @test bot_big < 0.0 && top_big > 0.0
        @test_throws ArgumentError semquad_nu_walls(sq, qt, zeros(Float64, sq.N),
                                                    prbig)
    end

    # ======================================================================
    # (c) hydrostatic balance
    # ======================================================================
    @testset "(c) hydrostatic balance and the pressure gauge" begin
        g2, qt, ss = g274_space(2, 3)
        sq = ss.su
        pr0 = Params(Ra=1000.0, H=0.0, eta_law=:expT, eta_b=0.0, bc=:rigid)
        zfun = (x, y) -> 0.0
        eta1 = (x, y) -> 1.0
        leaves = Set(qt.leaves)
        # (c1) the conduction state is force-free: the production buoyancy
        # closures of theta = 0 are identically zero, so the Stokes solution is
        # u = 0 and the dynamic pressure is the gauge constant
        th0 = zeros(Float64, sq.N)
        thf0 = (x, y) -> MS274._semquad_eval_at(sq, th0, x, y, leaves)
        fx0, fz0 = MS274._semquad_buoyancy_load(pr0, g, thf0,
                                                PhaseTransition[])
        for (x, y) in ((1.5, 0.2), (-0.4, 1.9), (0.8, -1.2), (2.0, 0.0))
            @test fx0(x, y) == 0.0 && fz0(x, y) == 0.0
        end
        u, pp, = solve_semquad_stokes_var(ss, qt, eta1, fx0, fz0, zfun, zfun;
                                          bc=:rigid, linear=:direct)
        @test maximum(abs, u) <= 1e-12
        @test maximum(pp) - minimum(pp) <= 1e-12
        @test MS274.LAST_SEMSTOKES_PRESSURE_GAUGE[] === :pin
        # (c2) a constant manufactured perturbation theta = c: f = Ra c e_r must
        # be balanced by the analytic hydrostatic pressure p(r) = Ra c r + const
        c = 0.1
        thc = fill(c, sq.N)
        thf = (x, y) -> MS274._semquad_eval_at(sq, thc, x, y, leaves)
        fx, fz = MS274._semquad_buoyancy_load(pr0, g, thf,
                                              PhaseTransition[])
        # the load is the radial vector Ra c e_r, both components
        for (x, y) in ((1.5, 0.2), (-0.4, 1.9), (0.8, -1.2))
            r = hypot(x, y)
            @test abs(fx(x, y) - pr0.Ra * c * x / r) <= 1e-9 * pr0.Ra * c
            @test abs(fz(x, y) - pr0.Ra * c * y / r) <= 1e-9 * pr0.Ra * c
        end
        # non-vacuous: the horizontal component is not zero (a vertical-only
        # load would have fx = 0)
        @test max(abs(fx(1.5, 1.0)), abs(fz(-1.0, 1.5))) > 1e-3 * pr0.Ra * c
        u2, pp2, = solve_semquad_stokes_var(ss, qt, eta1, fx, fz, zfun, zfun;
                                            bc=:rigid, linear=:direct)
        @test maximum(abs, u2) <= 1e-10
        # the dynamic pressure is the analytic integral of the buoyancy plus the
        # gauge constant: p_i - Ra c r_i is constant over the pressure nodes,
        # and the inner-to-outer pressure spread is Ra c (R1 - R0)
        off = [pp2[i] - pr0.Ra * c * hypot(ss.sp.coords[i][1],
                                           ss.sp.coords[i][2])
               for i in 1:ss.sp.N]
        @test maximum(off) - minimum(off) <= 1e-9 * pr0.Ra * c * G274_R1
        @test abs((maximum(pp2) - minimum(pp2)) -
                  pr0.Ra * c * (G274_R1 - G274_R0)) <= 1e-9 * pr0.Ra * c
    end

    # ======================================================================
    # (d) end-to-end annulus driver run, checkpoint identity, restrictions
    # ======================================================================
    @testset "(d) annulus driver end to end" begin
        pr = Params(Ra=100.0, H=0.0, eta_law=:expT, eta_b=0.0, bc=:rigid)
        function g274_run(; nsteps::Int=4, kwargs...)
            qt = g274_tree(2)
            ss = make_semstokes(qt, 2; geom=g274_geom())
            th0 = g274_theta0(ss)
            return run_semquad_simulation(qt, pr, th0; geom=:annulus,
                                          annulus_R0=G274_R0,
                                          annulus_R1=G274_R1,
                                          annulus_Phi=G274_PHI,
                                          p=2, tstop=nsteps * dt, dt0=dt,
                                          dtmax=dt, dtmin=1.0e-16,
                                          linear=:direct, maxsteps=nsteps,
                                          amr_every=0, advection=:skew,
                                          maxpic=1, verbose=false, kwargs...)
        end
        straight = g274_run()
        @test straight.steps == 4
        @test length(straight.history) == 4
        @test all(r -> length(r) == 18, straight.history)
        @test all(isfinite, vcat(straight.history...))
        @test all(isfinite, straight.theta) && all(isfinite, straight.ux) &&
              all(isfinite, straight.uz) && all(isfinite, straight.pv)
        @test geometry_identity(straight.ss.su.geom) ==
              geometry_identity(g274_geom())
        # the run advanced the state and wrote the annulus wedge area
        @test maximum(abs, straight.theta) <= 0.1
        @test abs(total_measure(g274_geom(), straight.ss.su) -
                  0.5 * G274_PHI * (G274_R1^2 - G274_R0^2)) <= 1e-13
        # history columns are exactly the diagnostics of the returned state
        sq = straight.ss.su
        qt = straight.qt
        th = straight.theta
        nut, nub = semquad_nu_walls(sq, qt, th, pr)
        row = straight.history[end]
        @test row[3] == nut && row[4] == nub
        @test row[5] == semquad_vrms(sq, qt, straight.ux, straight.uz)
        @test row[6] == semquad_mean_T(sq, qt, th, pr)
        @test row[9] == semquad_max_divergence(sq, qt, straight.ux,
                                               straight.uz)
        @test row[10] == minimum_leaf_height(straight.ss.su.geom, qt)
        # the small-Ra run is a genuine flow and stays near Nu = 1
        @test row[5] > 1e-3
        @test abs(row[3] - 1.0) <= 1e-2 && abs(row[4] - 1.0) <= 1e-2
        # the discrete identities the gate checks: Ra*W = U and int div u = 0
        @test abs(row[18]) <= 1e-6
        umax = maximum(hypot.(straight.ux, straight.uz))
        area = total_measure(g274_geom(), sq)
        @test abs(g274_div_volume(g274_geom(), sq, straight.ux, straight.uz)) <=
              1e-8 * max(umax * area, 1.0)

        # checkpoint / restart: the same annulus geometry resumes bitwise
        ck = tempname() * ".h5"
        first_half = g274_run(nsteps=2, checkpoint_path=ck,
                              checkpoint_every=1)
        @test isfile(ck)
        rest = g274_run(nsteps=4, restart_from=ck)
        @test rest.steps == 4
        # a restarted run starts a fresh history at the resumed step, so it must
        # reproduce the uninterrupted run's rows 3..4 exactly
        @test length(rest.history) == 2
        @test rest.history == straight.history[3:4]
        @test maximum(abs, rest.theta - straight.theta) == 0.0
        @test rest.qt.leaves == straight.qt.leaves
        rm(ck; force=true)
        # a checkpoint of another geometry is refused before any state is used
        ck2 = tempname() * ".h5"
        g274_run(nsteps=2, checkpoint_path=ck2, checkpoint_every=1)
        err = try
            run_semquad_simulation(g274_tree(2), pr, Float64[];
                                   p=2, tstop=dt, dt0=dt, maxsteps=1,
                                   linear=:direct, geom=CartesianBox(
                                       Quadtree(2)),
                                   restart_from=ck2, verbose=false)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("geometry", sprint(showerror, err))
        rm(ck2; force=true)

        # driver restrictions: the annulus card keeps every documented refusal.
        # The refusals fire before the mesh/physics setup, so an empty theta0
        # is enough (G126 precedent).
        qt0 = g274_tree(2)
        ann = (; geom=:annulus, annulus_R0=G274_R0, annulus_R1=G274_R1,
               annulus_Phi=G274_PHI)
        function g274_refusal(prcard; kwargs...)
            opts = merge(ann, (; kwargs...))
            try
                run_semquad_simulation(qt0, prcard, Float64[];
                                       opts..., maxsteps=0, tstop=0.0,
                                       linear=:direct, verbose=false)
                return nothing
            catch e
                return e
            end
        end
        # (i) the geometry knob itself
        e = try
            run_semquad_simulation(qt0, pr, Float64[]; geom=:deck,
                                   maxsteps=0, tstop=0.0, linear=:direct,
                                   verbose=false)
            nothing
        catch e2
            e2
        end
        @test e isa ArgumentError && occursin(":annulus", sprint(showerror, e))
        # (ii) the gap must be the length unit of the closures and of Ra
        e = g274_refusal(pr; annulus_R1=2.5)
        @test e isa ArgumentError && occursin("R1 - R0 = 1",
                                              sprint(showerror, e))
        # (iii) the wedge has no Cartesian side-temperature control
        e = g274_refusal(Params(Ra=100.0, H=0.0, eta_law=:expT, eta_b=0.0,
                                bc=:rigid, side_thermal=:fixed))
        @test e isa ArgumentError && occursin("side_thermal",
                                              sprint(showerror, e))
        # (iv) box-only diagnostic families
        e = g274_refusal(pr; nu_cbf=true)
        @test e isa ArgumentError && occursin("nu_cbf", sprint(showerror, e))
        e = g274_refusal(pr; extdiag=true)
        @test e isa ArgumentError && occursin("extdiag", sprint(showerror, e))
        # (v) rigid walls only in stage 0.5b1: the free-slip curved wall needs
        # the general normal-velocity rows of stage 0.5b2, and an :open wall
        # changes the pressure gauge.  The default Params is bc = :free, so a
        # plain annulus card is refused until the caller passes a rigid one.
        e = g274_refusal(Params(Ra=100.0, H=0.0, eta_law=:expT, eta_b=0.0))
        @test e isa ArgumentError && occursin("rigid", sprint(showerror, e))
        e = g274_refusal(Params(Ra=100.0, H=0.0, eta_law=:expT, eta_b=0.0,
                                bc=:rigid,
                                wall_bc=(:same, :same, :same, :open)))
        @test e isa ArgumentError && occursin("rigid", sprint(showerror, e))
        # (vi) the reference state / density hook stay refused on a mapped
        # geometry in this stage (the annulus base state is the Boussinesq
        # radial conduction state; a TALA/ALA annulus is a later stage)
        e = g274_refusal(Params(Ra=100.0, H=0.0, eta_law=:expT, eta_b=0.0,
                                bc=:rigid,
                                refstate=SemRefState(Tsurf=0.0)))
        @test e isa ArgumentError && occursin("reference state",
                                              sprint(showerror, e))
        e = g274_refusal(Params(Ra=100.0, H=0.0, eta_law=:expT, eta_b=0.0,
                                bc=:rigid,
                                rho_multiplier=(x, y) -> 1.0))
        @test e isa ArgumentError && occursin("rho_multiplier",
                                              sprint(showerror, e))
        # (vii) the mapped-geometry restrictions stage 0 already owns
        e = g274_refusal(pr; symmetry=:x)
        @test e isa ArgumentError && occursin("symmetry", sprint(showerror, e))
        e = g274_refusal(pr; pic_mode=:newton)
        @test e isa ArgumentError && occursin("pic_mode", sprint(showerror, e))
        # (viii) the closed ring is refused by the geometry constructor itself
        # (G273 pins the message); the wedge is the constructible topology here.
        @test_throws ArgumentError PolarAnnulus(G274_R0, G274_R1, 2.0 * pi;
                                                seam=:periodic)
    end

    # ======================================================================
    # (e) closure wiring: depth-dependent laws and the annulus decisions
    # ======================================================================
    @testset "(e) depth-dependent closures and the annulus decisions" begin
        g2, qt, ss = g274_space(2, 3)
        sq = ss.su
        leaves = Set(qt.leaves)
        th0 = zeros(Float64, sq.N)
        # (e1) the viscosity law's depth argument is the height h = r - R0:
        # `semquad_eta_at` must equal the analytic law evaluated at the radial
        # base state and h (the box's `y` would give a different answer off the
        # y axis)
        prl = Params(Ra=1.0, H=0.0, eta_law=:expTz, eta_b=0.5, eta_c=0.7,
                     bc=:rigid)
        for (x, y) in ((1.5, 0.2), (-0.4, 1.9), (0.8, -1.2))
            r = hypot(x, y)
            h = r - G274_R0
            T = g274_Tbar(0.0, r)
            want = eta_at(T, h, prl)[1]
            got = MS274.semquad_eta_at(sq, th0, x, y, prl, leaves)
            @test abs(got - want) <= 1e-13 * want
            # the same law evaluated at the Cartesian y is a different number
            # at these points (the wiring is not a no-op)
            @test abs(eta_at(T, y, prl)[1] - want) > 1e-3 * want
        end
        # (e2) phase-transition depth z0 and the phase buoyancy use the height:
        # the production mapped method equals the gate's h-based anomaly, and it
        # differs from the y-based evaluation at a point with y != h
        pt = PhaseTransition(0.5, 0.05, 0.02, 0.0, 1.0, 0.5)
        phases = [pt]
        prp = Params(Ra=100.0, H=0.0, eta_law=:expT, eta_b=0.0, bc=:rigid)
        for (x, y) in ((1.5, 0.2), (-0.4, 1.9), (0.8, -1.2))
            r = hypot(x, y)
            h = r - G274_R0
            Tad = g274_Tbar(0.0, r)
            for theta in (0.1, -0.2)
                Tg = Tad + theta
                want = -prp.Ra * pt.B *
                       (phase_gamma(pt, h, Tg) - phase_gamma(pt, h, Tad))
                got = MS274.semquad_phase_buoyancy(prp, phases, g2, x, y,
                                                   theta)
                @test abs(got - want) <= 1e-13 * abs(want)
                # y-based depth would give a different anomaly here
                want_y = -prp.Ra * pt.B *
                         (phase_gamma(pt, y, Tg) - phase_gamma(pt, y, Tad))
                @test abs(want_y - want) > 1e-8
            end
        end
        # (e3) latent heat and adiabatic heating assemble on the annulus: the
        # operator is finite and the sources are present (the mapped energy body
        # is shared with the validation geometries, but the annulus base state
        # and height are new)
        prlat = Params(Ra=100.0, H=0.0, eta_law=:expT, eta_b=0.0, bc=:rigid,
                       Di=1.0, latent_L=0.5)
        thp = g274_theta0(ss)
        ux = zeros(Float64, sq.N)
        uz = zeros(Float64, sq.N)
        for (gid, (x, y)) in enumerate(sq.coords)
            ux[gid] = 0.01 * (y - G274_R0) * (hypot(x, y) - G274_R1)
            uz[gid] = -0.01 * (x) * (hypot(x, y) - G274_R1)
        end
        M, K, C, F = assemble_semquad_energy(sq, qt, prlat, ux, uz;
                                             advection=:skew, theta=thp,
                                             phases=phases,
                                             heating_adiabatic=true,
                                             heating_dissipation=false)
        @test all(isfinite, vcat(M.colptr, M.nzval, K.nzval, C.nzval, F))
        @test maximum(abs, F) > 0.0
        # (e4) the H_half_life / rho_heating plumbing: the assembled source is
        # exactly linear in the driver's `heating_scale`, because the mapped
        # energy body multiplies the radiogenic term by it
        prh = Params(Ra=100.0, H=2.0, eta_law=:expT, eta_b=0.0, bc=:rigid,
                     rho_heating=true)
        zux = zeros(Float64, sq.N)
        zuz = zeros(Float64, sq.N)
        _, _, _, F1 = assemble_semquad_energy(sq, qt, prh, zux, zuz;
                                             advection=:skew, theta=th0,
                                             rho_heating=true,
                                             heating_scale=1.0)
        _, _, _, F2 = assemble_semquad_energy(sq, qt, prh, zux, zuz;
                                             advection=:skew, theta=th0,
                                             rho_heating=true,
                                             heating_scale=0.5)
        @test maximum(abs, F1) > 0.0
        @test maximum(abs.(F2 .- 0.5 .* F1)) <= 1e-14 * maximum(abs, F1)
        # (e5) the weak zone (a physical-position multiplier) is accepted and
        # changes the flow; it was a box-only path by assumption, not by
        # geometry.  `side_thermal=:fixed` is refused in (d).
        mulf = weak_zone_factor((1.6, 0.4), (1.6, 1.4); halfwidth=0.25,
                                factor=0.2)
        prw = Params(Ra=100.0, H=0.0, eta_law=:expT, eta_b=0.0, bc=:rigid,
                     eta_multiplier=mulf)
        qt2 = g274_tree(2)
        ss2 = make_semstokes(qt2, 2; geom=g274_geom())
        th2 = g274_theta0(ss2)
        rw = run_semquad_simulation(qt2, prw, th2; geom=:annulus,
                                    annulus_R0=G274_R0, annulus_R1=G274_R1,
                                    annulus_Phi=G274_PHI, p=2,
                                    tstop=dt, dt0=dt, maxsteps=1,
                                    linear=:direct, verbose=false)
        @test rw.steps == 1 && all(isfinite, rw.theta)
        # (e6) the latent heat runs end to end on the annulus (the history gains
        # the latent-free closure column)
        prl2 = Params(Ra=100.0, H=0.0, eta_law=:expT, eta_b=0.0, bc=:rigid,
                      latent_L=0.5)
        qt3 = g274_tree(2)
        ss3 = make_semstokes(qt3, 2; geom=g274_geom())
        reslat = run_semquad_simulation(qt3, prl2, g274_theta0(ss3);
                                        geom=:annulus, annulus_R0=G274_R0,
                                        annulus_R1=G274_R1,
                                        annulus_Phi=G274_PHI, p=2,
                                        tstop=dt, dt0=dt, maxsteps=1,
                                        linear=:direct, phases=phases,
                                        verbose=false)
        @test reslat.steps == 1
        @test length(reslat.history[end]) == 19
        @test all(isfinite, reslat.history[end])
    end
end
