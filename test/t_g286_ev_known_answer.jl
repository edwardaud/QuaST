# G286 (remediation batch P0 part 1): a known-answer gate for the
# entropy-viscosity operator (AUDIT_VV F4 / AUDIT_GATES M11 (i), mutation m03).
#
# The hole: the only independent EV gate (G85) asserted *properties* -- turning
# EV on changes K, K is finite, entropy_variation >= 0 -- and G149 is a
# field-consistency mirror.  The audit's mutation m03 (the residual x-gradient
# scale `2/h -> 1/h` in `src/physics/entropy_viscosity.jl`) survived G85+G149;
# only the gate-local pre-fix mirror G188(a) caught it.  So the EV *amplitude*
# had no known answer.
#
# The known answer used here: a **uniform-gradient steady state** on the box.
# Take theta = A (x - 1/2) (a Q_2-representable linear field, exactly steady:
# theta_prev = theta and dt > 0 give dtheta/dt = 0), u = (U, 0) constant.  Then
# every quantity of the EV operator is analytic:
#
#   * the residual is  R = |U A| * |Tbar(y) + theta - avgT|  because the
#     advective term U dtheta/dx is exact, dtheta/dt = 0, lap(theta) = 0 and the
#     base state is Tbar = 1 - y (H = 0), whose gradient contributes 0 (u_z = 0);
#   * Rmax is the max of that expression over the leaf's Gauss points, umax = U;
#   * he = sqrt(2) h and the EV viscosity is
#         nu = min(ev_cmax U he, ev he^2 Rmax / entropy_variation)
#     with the entropy variation an explicit weighted quadrature of
#     (Tmid - avgT)^2 over the same Gauss points (avgT = (minT+maxT)/2 from the
#     nodal total temperature);
#   * the wall rows (logical top/bottom) have nu = 0 exactly.
#
# The gate recomputes all of those from the *definition* (leaf geometry and the
# Gauss abscissae; the only library calls are the quadrature rule and the space,
# no EV arithmetic) and asserts:
#   (a) the EV globals match the analytic min/max/mean/u_infty;
#   (b) `semquad_ev_cell_stats` returns the analytic (Rmax, umax) per leaf;
#   (c) `semquad_ev_field` returns the analytic per-leaf nu;
#   (d) the production energy assembly returns the same field through `ev_out`
#       and its stiffness scales exactly by (1 + nu) for a supplied uniform EV
#       field: K(nu = 1) == 2 K(nu = 0) entry by entry (a dropped `1 +` in
#       `cK = w (1 + nu_ev)` is red, and it is not a property check).
#
# Mutation evidence (this campaign): with the audit's m03 mutation applied to
# `src/physics/entropy_viscosity.jl` (the residual gradient scale `2/h -> 1/h`),
# (b)/(c)/(d) are red at O(1) relative error (Rmax and nu halve); on the fixed
# tree all rows pass with the measured values printed.
using Test
using LinearAlgebra
using QuaST
const MS286 = QuaST

const G286_A = 1.0          # theta = A (x - 1/2)
const G286_U = 0.5          # u = (U, 0)
const G286_EV = 0.33
const G286_CMAX = 0.052
const G286_TIMESTEP = 5

g286_tmid(x, y) = (1.0 - y) + G286_A * (x - 0.5)   # Tbar(y) + theta(x)

"""Analytic (Rmax, umax) of one box leaf for the uniform-gradient state."""
function g286_Rmax(lf, p)
    fd = make_fedata(p, p)
    h = 1.0 / (1 << lf[1])
    x0 = lf[2] * h
    y0 = lf[3] * h
    Rmax = 0.0
    for qy in 1:fd.nq, qx in 1:fd.nq
        x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * h
        y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * h
        R = abs(G286_U * G286_A) * abs(g286_tmid(x, y) - 0.5)
        Rmax = max(Rmax, R)
    end
    return Rmax, G286_U
end

"""Analytic EV globals of the uniform-gradient state (no EV arithmetic)."""
function g286_globals(sq, p, leaves)
    # nodal total temperature: Tbar + theta
    minT = Inf; maxT = -Inf
    for (x, y) in sq.coords
        T = g286_tmid(x, y)
        minT = min(minT, T); maxT = max(maxT, T)
    end
    avgT = 0.5 * (minT + maxT)
    fd = make_fedata(p, p)
    eint = 0.0; area = 0.0; emin = Inf; emax = -Inf
    for lf in leaves
        h = 1.0 / (1 << lf[1])
        x0 = lf[2] * h; y0 = lf[3] * h
        for qy in 1:fd.nq, qx in 1:fd.nq
            x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * h
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * h
            e = (g286_tmid(x, y) - avgT)^2
            w = 0.25 * h * h * fd.gl_w[qx] * fd.gl_w[qy]
            eint += w * e; area += w
            emin = min(emin, e); emax = max(emax, e)
        end
    end
    avge = eint / area
    return (minT=minT, maxT=maxT, avgT=avgT,
            entropy_variation=max(emax - avge, avge - emin),
            field_variation=maxT - minT, u_infty=G286_U,
            avge=avge)
end

"""Analytic EV viscosity of one leaf for the uniform-gradient state."""
function g286_nu(lf, p, ana)
    h = 1.0 / (1 << lf[1])
    he = sqrt(leaf_hx_from_h(h) * leaf_hy_from_h(h)) * sqrt(2.0)
    Rmax, _ = g286_Rmax(lf, p)
    nu_max = G286_CMAX * G286_U * he
    nu_e = G286_EV * he * he * Rmax / ana.entropy_variation
    nu = min(nu_max, nu_e)
    (lf[3] == 0 || lf[3] == (1 << lf[1]) - 1) && (nu = 0.0)
    return nu
end
leaf_hx_from_h(h) = h
leaf_hy_from_h(h) = h

function g286_tree(L)
    qt = Quadtree(L + 2)
    for _ in 1:L, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    return qt
end

@testset "G286 EV known answer on a uniform-gradient state" begin
    L = 5
    p = 2
    qt = g286_tree(L)
    sq = make_semquad(qt, p)
    @test sq.N == (2 * (1 << L) + 1)^2   # Q_2: two intervals per leaf edge
    @test length(qt.leaves) == (1 << L)^2
    pr = Params(Ra=1.0, H=0.0, eta_law=:const, bc=:free, supg=0.0)
    th = [G286_A * (x - 0.5) for (x, y) in sq.coords]
    ux = fill(G286_U, sq.N)
    uz = zeros(Float64, sq.N)
    ana = g286_globals(sq, p, qt.leaves)

    # the fixture really is the advertised state: the linear field is nodal-exact
    for (i, (x, y)) in enumerate(sq.coords)
        @test abs(th[i] - G286_A * (x - 0.5)) <= 1.0e-15
    end

    # ---- (a) globals ---------------------------------------------------------
    evg = MS286.semquad_ev_globals(sq, qt, pr, th, copy(th), ux, uz, 0.0)
    @test evg.minT ≈ ana.minT atol=1.0e-14
    @test evg.maxT ≈ ana.maxT atol=1.0e-14
    @test evg.avgT ≈ ana.avgT atol=1.0e-14
    @test evg.u_infty ≈ ana.u_infty atol=1.0e-15
    @test evg.field_variation ≈ ana.field_variation atol=1.0e-14
    @test evg.entropy_variation ≈ ana.entropy_variation rtol=1.0e-12

    # ---- (b) per-cell stats: the EV residual amplitude -----------------------
    # every row: bottom (wall), interior, top (wall); the wall rows are not
    # special for the residual, only for the returned nu.
    test_leaves = [(L, 0, 0), (L, 1, 0), (L, 0, 1), (L, 3, 2), (L, (1 << L) - 2, 3)]
    ncell = 0
    for lf in test_leaves
        Rmax, umax = MS286.semquad_ev_cell_stats(sq, pr, th, copy(th), ux, uz,
                                                 1.0, lf, evg; alpha=2)
        Ra, ua = g286_Rmax(lf, p)
        # the production residual differs from the definition only by the
        # round-off of the Q_2 second-derivative cancellation (measured relative
        # 1.2e-10); the mutation is O(1) relative, so this is a tight pin.
        @test Rmax ≈ Ra rtol=1.0e-9
        @test umax ≈ ua atol=1.0e-15
        ncell += 1
    end
    @test ncell == 5
    # a non-trivial residual: the amplitude is O(1) in this state (a mutation
    # that zeroes or halves it cannot hide inside a tolerance)
    R0, _ = MS286.semquad_ev_cell_stats(sq, pr, th, copy(th), ux, uz, 1.0,
                                        (L, (1 << L) - 1, 1), evg; alpha=2)
    @test R0 > 0.4

    # ---- (c) the field -------------------------------------------------------
    field = MS286.semquad_ev_field(sq, qt, pr, th, ux, uz;
                                   dt=1.0, ev=G286_EV, ev_cmax=G286_CMAX,
                                   ev_alpha=2, ev_timestep=G286_TIMESTEP)
    ana_nu = [g286_nu(lf, p, ana) for lf in sq.leaves]
    @test length(field) == length(ana_nu)
    worst = maximum(abs(field[i] - ana_nu[i]) /
                    max(abs(ana_nu[i]), 1.0e-300) for i in eachindex(field))
    println("G286 EV known answer: max relative nu error = ", worst,
            "  (nu range ", minimum(ana_nu), " .. ", maximum(ana_nu), ")")
    @test worst <= 1.0e-9
    # the wall rows are exactly zero; at least one interior value is positive
    wall = [ana_nu[i] for (i, lf) in enumerate(sq.leaves)
            if lf[3] == 0 || lf[3] == (1 << lf[1]) - 1]
    @test all(==(0.0), wall)
    @test maximum(ana_nu) > 0.0
    # the EV field is genuinely amplitude-sensitive: the entropy-limited branch
    # (not the cmax cap) is active, so the `ev` factor is observable.
    @test maximum(ana_nu) < G286_CMAX * G286_U * sqrt(2.0) * (1.0 / (1 << L))

    # ---- (d) the production energy assembly ---------------------------------
    buf = zeros(Float64, length(sq.leaves))
    _, Kev, _, _ = assemble_semquad_energy(sq, qt, pr, ux, uz;
        theta=th, theta_prev=copy(th), dt=1.0, ev=G286_EV,
        ev_cmax=G286_CMAX, ev_alpha=2, ev_timestep=G286_TIMESTEP,
        evg=evg, ev_out=buf)
    @test maximum(abs(buf[i] - ana_nu[i]) for i in eachindex(buf)) <= 1.0e-12
    # a dropped/wrong `(1 + nu_ev)` coefficient: with a supplied *uniform* EV
    # field nu0 = 1 the assembled stiffness must be exactly twice the nu0 = 0
    # operator (cK = w (1 + nu_ev)); 2 is a power of two, so the doubling is
    # bitwise exact.
    nleaf = length(sq.leaves)
    _, K0, _, _ = assemble_semquad_energy(sq, qt, pr, ux, uz;
        theta=th, theta_prev=copy(th), dt=1.0,
        ev_field=zeros(Float64, nleaf))
    _, K1, _, _ = assemble_semquad_energy(sq, qt, pr, ux, uz;
        theta=th, theta_prev=copy(th), dt=1.0,
        ev_field=ones(Float64, nleaf))
    @test K1.colptr == K0.colptr
    @test K1.rowval == K0.rowval
    @test K1.nzval == 2.0 .* K0.nzval
    @test norm(K1 - K0) > 0.0
    # and the assembled operator really used the EV field: Kev (nu from the
    # analytic state) differs from K0
    @test norm(Kev - K0) > 0.0
    @test all(isfinite, Kev.nzval) && all(isfinite, buf)
end
