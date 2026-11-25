# G275: curved-wall boundary conditions, the rotation gauge and the annular
# initial condition (stage 0.5b2).
#
# Stage 0.5a built the polar-annulus geometry and stage 0.5b1 the radial physics
# and the annular driver, both with **rigid** walls only (the mapped solver
# fixed Cartesian components, so a :free wall would have been a single global
# component, not `u.n = 0`).  This gate pins the stage-0.5b2 boundary layer:
#
#   (a) the general curved-wall linear constraint rows:
#       (a1) on the annulus circles the free-slip row is *exactly* `u.n = 0` at
#            every wall node: every column of the affine constraint matrix `Cu`
#            gives an admissible nodal velocity whose mapped outward normal
#            component is zero (and whose tangential component is not), and the
#            four wedge corners are pinned by both walls;
#       (a2) the conduction state is force-free (`theta = c` gives u = 0 and the
#            analytic hydrostatic pressure `p = Ra c r`), and a non-hydrostatic
#            radial drive produces a genuine flow whose solved wall normal
#            velocity vanishes to round-off (no spurious normal flow at the
#            curved wall);
#       (a3) a purely tangential manufactured drive (the restricted circular
#            Couette field, all four walls Dirichlet) reproduces the prescribed
#            tangential wall values and has zero normal velocity on the circles
#            and the prescribed through-flow on the radial cuts -- the
#            `wall_u` / `wall_un` wall-frame data;
#       (a4) the *inhomogeneous* free-slip row: `wall_un` on a curved :free wall
#            is imposed as `u.n = g` exactly (checked on the affine data `u0` and
#            on the solved wall values), not silently dropped;
#       (a5) the corner overwrite rule at the four wedge corners for the
#            free/free and the rigid/free mixed pairs (the later wall wins; on
#            the box that rule is the historical write order).
#
#   (b) the analytic gates:
#       (b1) **circular Couette**.  `u_phi = A r + B/r` with the ASPECT
#            constants `A = 2`, `B = -3/ln 2` is an exact steady Stokes solution
#            with `p = const` between concentric rotating cylinders.  A *wedge*
#            with free-slip radial sides is NOT the Couette problem: the two
#            radial cuts see a through-flow `u_theta = A r + B/r`, so free slip
#            there is inconsistent with the exact field.  The genuinely
#            wedge-analytic configuration is the restriction of the ring
#            solution with **Dirichlet data on all four walls** (circles:
#            tangential `f(R)`, cuts: normal `+-f(r)`), and that is what this
#            gate solves: velocity L2 order 3 / H1 order 2, the inner/outer wall
#            torques `+-2 B Phi` (the published full-ring torque `4 pi B` at
#            `Phi = 2 pi`) and the torque-balance residual `T_in + T_out`
#            converge.  The full ring (where Couette is also exact with free-slip
#            circles and no cuts) is stage 0.5c; the assertion that the
#            *published* free-slip ring benchmark is reachable is deferred with
#            it.  A free-slip-cut control demonstrates that the wedges differ
#            O(1) from the Couette restriction, i.e. the all-Dirichlet choice is
#            load-bearing;
#       (b2) the `:open` (traction) walls on the curved circles: the exact
#            azimuthal flow driven by a prescribed tangential traction
#            `t.tau = -2 eta B/r^2` (the `wall_t` data in the mapped n/tau frame)
#            with the derived unpinned pressure gauge is reproduced, and a
#            prescribed uniform normal traction `t.n = -p0` gives `u = 0`,
#            `p = p0` (the pressure level fixed by the traction, gauge `:none`).
#
#   (c) manufactured Stokes convergence on the annulus: the ASPECT k = 2
#       manufactured solution rotated by pi/4 is tangent to the radial cuts
#       (`v_theta = 0` at `theta = 0, pi/2`) with `v_r = 0` on both circles, so
#       it is the exact solution with Dirichlet velocities on the circles (data
#       `f(R) sin 2 theta`, position dependent) and **free-slip radial cuts**.
#       The ASPECT k = 2 densities/pressures are used as the body force and the
#       exact pressure.  Expected orders for Q_2/Q_1: velocity L2 3, H1 2,
#       pressure L2 2; measured values are reported by the gate.
#
#   (d) the solid-body-rotation nullspace of a free-slip curved wall: the
#       design's gauge is the zero-net-angular-momentum projection (the L2
#       projection onto `(-y, x)`).  With free-slip circles and open radial cuts
#       the rotation is admissible; the solver engages the gauge, the returned
#       field has zero discrete angular momentum measured independently, is
#       M-orthogonal to the rotation mode (velocity mass matrix), and agrees
#       across two nested meshes at the discretisation order, while the
#       *unprojected* representatives differ by the removed rotation (so the
#       gauge is not a no-op).  A torque-imbalanced load, a translation-
#       nullspace mapped configuration and the all-`:open` pure-traction card
#       are refused.
#
#   (e) the annular initial condition: `annulus_ic = :firedrake` (Davies et al.
#       2022 Eq. (41)) and `:aspect` (the ASPECT spherical hexagonal
#       perturbation, angular mode 4) are the documented **total**
#       temperatures, and the driver exposes them as `theta = T_ic - Tbar(r)`
#       through the existing `theta_ic` re-sampling hook.  The profiles are
#       checked against independently written formulas (including the
#       ASPECT argument-order conversion), the wall values, the 4-fold angular
#       pattern and the base-state subtraction; an end-to-end free-slip annulus
#       run (direct and `:mg`/`:auto`) is finite, writes history, keeps its
#       checked identities and refuses to combine the knob with a user IC.  The
#       periodic ring remains the 0.5c prerequisite (refused by the constructor;
#       the published Davies benchmark is the ring).
#
# WHY THE ASSERTIONS ARE NOT TAUTOLOGICAL (what each block catches):
#   * (a1) reads the *affine constraint matrix* the solve imposes: a row written
#     on the wrong vector (tangent instead of normal), a component-fix instead
#     of the linear combination `n.u = 0`, or a dropped row changes the check by
#     O(1).  The tangent variant is measured at O(1) in the same block, so a
#     rotation of the frame cannot pass by accident.  (a2) catches a wrong
#     normal at the solution level (a spurious normal flow), (a3) checks the
#     prescribed wall-frame data against the analytic field, (a4) would pass
#     neither if the affine offset `g/n_p` were missing nor if the sign of `n`
#     were flipped (the solved wall normal velocity would be 0 or `-g`), (a5)
#     pins the corner ownership rule;
#   * (b1) the exact Couette field is not a polynomial and the wedge is curved:
#     reproducing it at the observed order and the *torque* (a boundary
#     functional of the numerical stress) is a genuine discretisation test; the
#     free-slip-cut control shows why the all-Dirichlet data are needed.  (b2)
#     the traction data are the *wall-frame* components `t.tau`, `t.n`: a
#     global-component convention (the box's) would flip the inner-circle sign
#     and the flow would not match; the uniform normal traction fixes the
#     pressure level, which only the derived gauge (`:none`) allows;
#   * (c) the exact solution is smooth, the body force is the exact manufactured
#     density/pressure pair, and the free-slip cuts carry a nonzero tangential
#     velocity (so the natural condition is exercised).  The measured orders are
#     the Q_2/Q_1 orders of the scheme; a wrong body-force sign, a dropped
#     free-slip row or a Cartesian cut condition breaks the orders and the
#     solution;
#   * (d) the rotation mode of the *mapped* annulus space is only approximated
#     by the nodal field `(-y, x)` (the interpolant of a linear function is not
#     a rigid motion on a curved element), so the discrete nullspace is
#     approximate: the gate therefore measures the gauge by (i) the independent
#     quadrature functional, (ii) the velocity mass matrix (the L2 projection
#     the design names) and (iii) the *cross-mesh* agreement of the projected
#     fields, with the unprojected fields as the control.  A wrong weight, a
#     wrong sign or a missing projection fails (iii) and (i);
#   * (e) the IC formulas are compared with independently written expressions
#     (the ASPECT `sin(4 phi_A + pi/2)` in its own argument order) and the
#     driver run re-computes every checked history column from the returned
#     state; the identities are properties of the discrete solution.
#
# The gate pins `linear=:direct` where a bitwise/exact comparison is made; the
# `:mg`/`:auto` runs are compared on their (documented) tolerance.

using Test, QuaST, LinearAlgebra

const MS275 = QuaST

# ---------------------------------------------------------------------------
# fixtures and independent oracles
# ---------------------------------------------------------------------------

const G275_R0 = 1.22
const G275_R1 = 2.22
const G275_PHI = 0.5 * pi

g275_geom(; R0=G275_R0, R1=G275_R1, Phi=G275_PHI) = PolarAnnulus(R0, R1, Phi)
g275_lnr() = log(G275_R1 / G275_R0)

function g275_tree(level; R0=G275_R0, R1=G275_R1)
    qt = Quadtree(level; cells_x=1, cells_y=4)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

function g275_space(level, p; R0=G275_R0, R1=G275_R1, Phi=G275_PHI)
    g = g275_geom(; R0=R0, R1=R1, Phi=Phi)
    qt = g275_tree(level; R0=R0, R1=R1)
    return g, qt, make_semstokes(qt, p; geom=g)
end

g275_zero(x, y) = 0.0
g275_one(x, y) = 1.0

# --- the wall-row system, rebuilt from the internals the solve uses ---------
# `wbc` is the *resolved* 4-tuple (no :same); the closures are the Dirichlet
# pair plus the per-wall data, exactly as `_solve_semquad_stokes_var_core`
# passes them to `_semquad_mapped_wall_data`.
function g275_wall_system(g, ss, qt, wbc;
                          wall_u=(nothing, nothing, nothing, nothing),
                          wall_un=(nothing, nothing, nothing, nothing))
    sq = ss.su
    nu = sq.N
    bmask = MS275._semquad_boundary_face_mask(qt, sq)
    cons_u = semquad_constraints(sq, qt)
    cons = Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
    for (c, (ms, ws)) in cons_u
        cons[c] = (ms, ws)
        cons[nu + c] = (nu .+ ms, ws)
    end
    wd = MS275._semquad_mapped_wall_data(g, sq, bmask, wbc, wall_u, wall_un,
                                         g275_zero, g275_zero, cons)
    cons2, offs = MS275._semquad_merge_wall_constraints(cons, wd)
    u0, Cu, _ = MS275._affine_constraint(2 * nu, cons2, wd.fixed, offs)
    conds = MS275._semquad_wall_frames(g, sq, bmask)
    return (bmask=bmask, conds=conds, wd=wd, cons=cons2, offs=offs, u0=u0,
            Cu=Cu)
end

# largest |n.u| over every column of the affine constraint matrix, per free
# (normal-Dirichlet) wall node -- the exact statement "the row is u.n = 0"
function g275_row_residual(ws, sq)
    nu = sq.N
    worst = 0.0
    worst_tau = 0.0
    nfree = 0
    for (node, cs) in ws.conds
        (haskey(ws.wd.fixed, node) && haskey(ws.wd.fixed, nu + node)) && continue
        c = cs[1]
        nfree += 1
        for j in 1:size(ws.Cu, 2)
            vx = ws.Cu[node, j]
            vy = ws.Cu[nu + node, j]
            worst = max(worst, abs(c.nx * vx + c.ny * vy))
            worst_tau = max(worst_tau, abs(c.tx * vx + c.ty * vy))
        end
    end
    return worst, worst_tau, nfree
end

# nodal outward normal velocity at every wall node of a solved field
function g275_wall_un(ws, sq, u)
    nu = sq.N
    worst = 0.0
    for (node, cs) in ws.conds
        c = cs[1]
        worst = max(worst, abs(c.nx * u[node] + c.ny * u[nu + node]))
    end
    return worst
end

function g275_add(a, b)
    return (a[1] + b[1], a[2] + b[2])
end

# --- manufactured rotated ASPECT k = 2 family -------------------------------
# Unrotated family (ANNULUS_REFS 7.4, R0 = 1, R1 = 2, C = -1):
#   f(r) = A r + B/r, g(r) = (A/2) r + (B/r) ln r + C/r,
#   h(r) = (2g - f)/r, m(r) = g'' - g'/r - (k^2-1) g/r^2 + f/r^2 + f'/r.
# Rotated by beta = pi/4 (k = 2) the fields are
#   v_r = -2 g(r) cos 2t, v_t = f(r) sin 2t, p' = -2 h(r) cos 2t,
# and the solver's body force is `f = grad p' - div(2 eps(u)) = +2 m cos 2t e_r`
# (the residual of the rotated family, identified analytically and checked by
# the convergence of the discrete solution).
const G275_A = 2.0
const G275_B = -3.0 / log(2.0)
g275_f(r) = G275_A * r + G275_B / r
g275_g(r) = 0.5 * G275_A * r + G275_B * log(r) / r - 1.0 / r
g275_gp(r) = 0.5 * G275_A + G275_B * (1.0 - log(r)) / r^2 + 1.0 / r^2
g275_gpp(r) = G275_B * (2.0 * log(r) - 3.0) / r^3 - 2.0 / r^3
g275_fp(r) = G275_A - G275_B / r^2
g275_m(r) = g275_gpp(r) - g275_gp(r) / r - 3.0 * g275_g(r) / r^2 +
            g275_f(r) / r^2 + g275_fp(r) / r
g275_h(r) = (2.0 * g275_g(r) - g275_f(r)) / r

function g275_manu_u(x, y)
    r = hypot(x, y)
    th = atan(y, x)
    vr = -2.0 * g275_g(r) * cos(2.0 * th)
    vt = g275_f(r) * sin(2.0 * th)
    c = cos(th); s = sin(th)
    return (vr * c - vt * s, vr * s + vt * c)
end
g275_manu_p(x, y) = -2.0 * g275_h(hypot(x, y)) * cos(2.0 * atan(y, x))
function g275_manu_fx(x, y)
    r = hypot(x, y)
    return 2.0 * g275_m(r) * cos(2.0 * atan(y, x)) * x / r
end
function g275_manu_fz(x, y)
    r = hypot(x, y)
    return 2.0 * g275_m(r) * cos(2.0 * atan(y, x)) * y / r
end
g275_manu_wall_u(x, y) = g275_f(hypot(x, y)) * sin(2.0 * atan(y, x))

# L2 / H1 velocity and mean-removed L2 pressure errors (Gauss quadrature, same
# rule as the assembly; the exact gradient is a central difference of the exact
# closure, which is far more accurate than the error it measures).
function g275_manu_errors(g, ss, u, pv)
    sq = ss.su
    nu = sq.N
    p = sq.p
    q = ss.q
    fd = make_fedata(p, p - 1)
    rule_p = gll_rule(q)
    eu2 = 0.0; eh1 = 0.0; ep2 = 0.0; epm = 0.0; area = 0.0
    for lf in sq.leaves
        m = leaf_metric(g, sq, lf, fd)
        ids = sq.elem[lf]
        pids = ss.sp.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = MS275.quad_measure(m, fd, qx, qy)
            x, y = quad_point(m, sq, lf, fd, qx, qy)
            uxq = 0.0; uyq = 0.0
            dux = [0.0, 0.0]; duy = [0.0, 0.0]
            for j in 1:p+1, i in 1:p+1
                k = (j - 1) * (p + 1) + i
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                uxq += u[ids[k]] * b
                uyq += u[nu + ids[k]] * b
                gx, gy = basis_grad(m, fd, qx, qy, i, j)
                dux[1] += u[ids[k]] * gx; dux[2] += u[ids[k]] * gy
                duy[1] += u[nu + ids[k]] * gx; duy[2] += u[nu + ids[k]] * gy
            end
            ue = g275_manu_u(x, y)
            hs = 1.0e-6
            up = g275_manu_u(x + hs, y); um = g275_manu_u(x - hs, y)
            vp = g275_manu_u(x, y + hs); vm = g275_manu_u(x, y - hs)
            duex = ((up[1] - um[1]) / (2hs), (vp[1] - vm[1]) / (2hs),
                    (up[2] - um[2]) / (2hs), (vp[2] - vm[2]) / (2hs))
            eu2 += w * ((uxq - ue[1])^2 + (uyq - ue[2])^2)
            eh1 += w * ((dux[1] - duex[1])^2 + (dux[2] - duex[2])^2 +
                        (duy[1] - duex[3])^2 + (duy[2] - duex[4])^2)
            pe = g275_manu_p(x, y)
            phq = 0.0
            for j in 1:q+1, i in 1:q+1
                k = (j - 1) * (q + 1) + i
                Li, _ = lagrange_eval(rule_p, fd.gl_x[qx])
                Lj, _ = lagrange_eval(rule_p, fd.gl_x[qy])
                phq += pv[pids[k]] * Li[i] * Lj[j]
            end
            ep2 += w * (phq - pe)^2
            epm += w * (phq - pe)
            area += w
        end
    end
    pe2 = max(ep2 / area - (epm / area)^2, 0.0)
    return sqrt(eu2 / area), sqrt(eh1 / area), sqrt(pe2)
end

# --- Couette machinery ------------------------------------------------------
# `u = (A r + B/r) e_phi`, exact on the hole-free ring; on the wedge the
# restriction with Dirichlet data on all four walls.
function g275_couette_u(x, y)
    r = hypot(x, y)
    f = G275_A * r + G275_B / r
    return (-f * y / r, f * x / r)
end
g275_couette_p(x, y) = 0.0
const G275_COUETTE_TORQUE_IN = 2.0 * G275_B * G275_PHI   # fluid torque, inner
const G275_COUETTE_TORQUE_OUT = -G275_COUETTE_TORQUE_IN

function g275_couette_errors(g, ss, u)
    sq = ss.su
    nu = sq.N
    p = sq.p
    fd = make_fedata(p, p - 1)
    eu2 = 0.0; eh1 = 0.0; area = 0.0
    for lf in sq.leaves
        m = leaf_metric(g, sq, lf, fd)
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = MS275.quad_measure(m, fd, qx, qy)
            x, y = quad_point(m, sq, lf, fd, qx, qy)
            uxq = 0.0; uyq = 0.0
            dux = [0.0, 0.0]; duy = [0.0, 0.0]
            for j in 1:p+1, i in 1:p+1
                k = (j - 1) * (p + 1) + i
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                uxq += u[ids[k]] * b
                uyq += u[nu + ids[k]] * b
                gx, gy = basis_grad(m, fd, qx, qy, i, j)
                dux[1] += u[ids[k]] * gx; dux[2] += u[ids[k]] * gy
                duy[1] += u[nu + ids[k]] * gx; duy[2] += u[nu + ids[k]] * gy
            end
            ue = g275_couette_u(x, y)
            hs = 1.0e-6
            up = g275_couette_u(x + hs, y); um = g275_couette_u(x - hs, y)
            vp = g275_couette_u(x, y + hs); vm = g275_couette_u(x, y - hs)
            duex = ((up[1] - um[1]) / (2hs), (vp[1] - vm[1]) / (2hs),
                    (up[2] - um[2]) / (2hs), (vp[2] - vm[2]) / (2hs))
            eu2 += w * ((uxq - ue[1])^2 + (uyq - ue[2])^2)
            eh1 += w * ((dux[1] - duex[1])^2 + (dux[2] - duex[2])^2 +
                        (duy[1] - duex[3])^2 + (duy[2] - duex[4])^2)
            area += w
        end
    end
    return sqrt(eu2 / area), sqrt(eh1 / area)
end

# torque on one circular wall from the assembled numerical stress
# `sigma = 2 eps(u) - p I`, `t = sigma.n`, `T = int (x t_y - y t_x) ds` with the
# mapped face Jacobian and the wall GLL rule.  (The pressure contributes nothing
# to the *tangential* traction but is kept in the Cartesian torque integrand.)
function g275_wall_torque(g, ss, u, pv, side)
    sq = ss.su
    nu = sq.N
    n = sq.p + 1
    q = ss.q
    rule_p = gll_rule(q)
    isx = side === :xminus || side === :xplus
    T = 0.0
    for lf in sq.leaves
        MS275._on_logical_wall(lf, side, sq) || continue
        ids = sq.elem[lf]
        pids = ss.sp.elem[lf]
        fm = MS275._mapped_face_metric(g, sq, lf, side, sq.xi)
        for qq in 1:n
            xi, eta = MS275._wall_ref_point(side, sq.xi[qq])
            inv = jacobian_inverse(g, sq, lf, xi, eta)
            if isx
                i = side === :xplus ? n : 1
                j = qq
            else
                i = qq
                j = side === :yplus ? n : 1
            end
            dxi = [0.0, 0.0, 0.0, 0.0]
            for a in 1:n
                dxi[1] += sq.D[i, a] * u[ids[(j - 1) * n + a]]
                dxi[3] += sq.D[i, a] * u[nu + ids[(j - 1) * n + a]]
                dxi[2] += sq.D[j, a] * u[ids[(a - 1) * n + i]]
                dxi[4] += sq.D[j, a] * u[nu + ids[(a - 1) * n + i]]
            end
            ux_x = dxi[1] * inv[1] + dxi[2] * inv[2]
            ux_y = dxi[1] * inv[3] + dxi[2] * inv[4]
            uy_x = dxi[3] * inv[1] + dxi[4] * inv[2]
            uy_y = dxi[3] * inv[3] + dxi[4] * inv[4]
            # pressure at the wall node from the pressure GLL basis
            pq = 0.0
            Li, _ = lagrange_eval(rule_p, xi)
            Lj, _ = lagrange_eval(rule_p, eta)
            for bb in 1:q+1, aa in 1:q+1
                pq += pv[pids[(bb - 1) * (q + 1) + aa]] * Li[aa] * Lj[bb]
            end
            sxx = 2.0 * ux_x - pq
            syy = 2.0 * uy_y - pq
            sxy = ux_y + uy_x
            nx_, ny_ = face_normal(fm, qq)
            tx = sxx * nx_ + sxy * ny_
            ty = sxy * nx_ + syy * ny_
            xq, yq = face_point(fm, qq)
            w = face_jacobian(fm, qq) * sq.w[qq]
            T += w * (xq * ty - yq * tx)
        end
    end
    return T
end

# independent volume integral of div(u) (G270/G274 pattern)
function g275_div_volume(g, sq, ux, uz)
    p = sq.p
    fd = make_fedata(p, p - 1)
    s = 0.0
    for lf in sq.leaves
        m = leaf_metric(g, sq, lf, fd)
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = MS275.quad_measure(m, fd, qx, qy)
            duxdx = 0.0; duzdy = 0.0
            for j in 1:p+1, i in 1:p+1
                k = (j - 1) * (p + 1) + i
                gx, gy = basis_grad(m, fd, qx, qy, i, j)
                duxdx += ux[ids[k]] * gx
                duzdy += uz[ids[k]] * gy
            end
            s += w * (duxdx + duzdy)
        end
    end
    return s
end

# independent annular IC total-temperature formulas (the oracles)
function g275_ic_firedrake(x, y)
    r = hypot(x, y)
    th = atan(y, x)
    return (G275_R1 - r) + 0.02 * cos(4.0 * th) * sinpi(r - G275_R0)
end
function g275_ic_aspect(x, y)
    r = hypot(x, y)
    d = G275_R1 - G275_R0
    s = (G275_R1 - r) / d
    # ASPECT: T = s + 0.2 s(1-s) sin(angular_mode*phi_A + (90 + 2*offset) deg)
    # with phi_A = atan2(x, y) (ASPECT's argument order; angle from +y) and
    # angular mode 4, offset 0.  Written in that form, not as cos(4 theta).
    phi_A = atan(x, y)
    return s + 0.2 * s * (1.0 - s) * sin(4.0 * phi_A + 0.5 * pi)
end

# ===========================================================================
@testset "G275 annulus curved walls, gauge and IC (stage 0.5b2)" begin
    # ======================================================================
    # (a) the general curved-wall constraint rows
    # ======================================================================
    @testset "(a1) the free-slip row is exactly u.n = 0" begin
        g, qt, ss = g275_space(2, 2)
        sq = ss.su
        nu = sq.N
        ws = g275_wall_system(g, ss, qt, (:free, :free, :free, :free))
        worst, worst_tau, nfree = g275_row_residual(ws, sq)
        @test nfree == 76                       # the level-2 uniform wall nodes
        @test worst <= 1.0e-15                  # the row IS the normal component
        # non-vacuous: the same test with the *tangent* is O(1), so a rotated
        # frame cannot pass this block
        @test worst_tau > 1.0
        # the four wedge corners are fixed by both wall rows (free/free 2x2)
        corners = [node for (node, cs) in ws.conds if length(cs) >= 2]
        @test length(corners) == 4
        @test all(n -> haskey(ws.wd.fixed, n) && haskey(ws.wd.fixed, nu + n),
                  corners)
        @test all(n -> ws.wd.fixed[n] == 0.0 && ws.wd.fixed[nu + n] == 0.0,
                  corners)
        # the tangential component really is free: the maximum |tau.u| over the
        # Cu columns at the free wall nodes is O(1) (measured in `worst_tau`),
        # so the admissible space is not the zero space and the normal row is
        # not accidentally the tangential one
        @test maximum(abs, ws.Cu) > 0.5
        println("G275 a1 row: nodes = ", nfree, "  max|n.u| on Cu = ", worst,
                "  max|tau.u| on Cu = ", worst_tau)
    end

    @testset "(a2) conduction state and a non-hydrostatic radial drive" begin
        g, qt, ss = g275_space(2, 2)
        sq = ss.su
        nu = sq.N
        pr = Params(Ra=1000.0, H=0.0, eta_law=:expT, eta_b=0.0, bc=:free)
        # constant perturbation theta = c: f = Ra c e_r is hydrostatic, and the
        # exact dynamic pressure is Ra c r -- the curved free-slip wall must not
        # generate a spurious normal flow.
        c = 0.1
        fx = (x, y) -> pr.Ra * c * x / hypot(x, y)
        fz = (x, y) -> pr.Ra * c * y / hypot(x, y)
        u, p, = solve_semquad_stokes_var(ss, qt, g275_one, fx, fz, g275_zero,
                                         g275_zero; linear=:direct, bc=:free,
                                         wall_bc=(:free, :free, :free, :free))
        @test maximum(abs, u) <= 1.0e-12
        @test MS275.LAST_SEMSTOKES_PRESSURE_GAUGE[] === :pin
        @test MS275.LAST_SEMSTOKES_ROTATION_GAUGE[] === :none
        # exact pressure slope: p - Ra c r is constant over the pressure nodes
        off = [p[i] - pr.Ra * c * hypot(ss.sp.coords[i][1],
                                        ss.sp.coords[i][2])
               for i in 1:ss.sp.N]
        @test maximum(off) - minimum(off) <= 1.0e-10
        @test abs((maximum(p) - minimum(p)) - pr.Ra * c * (G275_R1 - G275_R0)) <=
              1.0e-9
        # a non-hydrostatic radial drive: genuine flow, zero solved wall normal
        thf = (x, y) -> 0.3 * (hypot(x, y) - G275_R0) * cos(2.0 * atan(y, x))
        f2x = (x, y) -> 1.0e3 * thf(x, y) * x / hypot(x, y)
        f2z = (x, y) -> 1.0e3 * thf(x, y) * y / hypot(x, y)
        ws = g275_wall_system(g, ss, qt, (:free, :free, :free, :free))
        u2, p2, = solve_semquad_stokes_var(ss, qt, g275_one, f2x, f2z,
                                           g275_zero, g275_zero;
                                           linear=:direct, bc=:free,
                                           wall_bc=(:free, :free, :free, :free))
        @test maximum(abs, u2) > 1.0e-3          # a real flow
        # solved normal velocity at the curved wall (not just on Cu)
        @test g275_wall_un(ws, sq, u2) <= 1.0e-12
        @test maximum(abs, p2) > 0.0
        @test abs(g275_div_volume(g, sq, u2[1:nu], u2[nu+1:end])) <=
              1.0e-9 * max(maximum(abs, u2) * total_measure(g, sq), 1.0)
        println("G275 a2: |u|max = ", maximum(abs, u2),
                "  solved max|u.n| at the wall = ", g275_wall_un(ws, sq, u2))
    end

    @testset "(a3) purely tangential drive: wall-frame data" begin
        g = PolarAnnulus(1.0, 2.0, G275_PHI)
        qt = g275_tree(2; R0=1.0, R1=2.0)
        ss = make_semstokes(qt, 2; geom=g)
        sq = ss.su
        nu = sq.N
        wu = (x, y) -> g275_f(hypot(x, y))
        un_lo = (x, y) -> -g275_f(hypot(x, y))
        un_hi = (x, y) -> +g275_f(hypot(x, y))
        u, p, = solve_semquad_stokes_var(ss, qt, g275_one, g275_zero,
                                         g275_zero, g275_zero, g275_zero;
                                         linear=:direct, bc=:rigid,
                                         wall_bc=(:rigid, :rigid, :rigid, :rigid),
                                         wall_u=(wu, wu, nothing, nothing),
                                         wall_un=(nothing, nothing, un_lo, un_hi))
        bmask = MS275._semquad_boundary_face_mask(qt, sq)
        conds = MS275._semquad_wall_frames(g, sq, bmask)
        worst_n = 0.0
        worst_t = 0.0
        for (node, cs) in conds
            c = cs[1]
            ut = c.tx * u[node] + c.ty * u[nu + node]
            if c.wi == 1 || c.wi == 2          # circles: tangential Dirichlet
                worst_n = max(worst_n, abs(c.nx * u[node] + c.ny * u[nu + node]))
                worst_t = max(worst_t, abs(ut - g275_f(hypot(c.x, c.y))))
            else                               # cuts: normal through-flow data
                un = c.nx * u[node] + c.ny * u[nu + node]
                et = hypot(c.x, c.y)
                gval = c.wi == 3 ? -g275_f(et) : g275_f(et)
                worst_n = max(worst_n, abs(un - gval))
            end
        end
        @test worst_n <= 1.0e-13                # circles: u.n = 0; cuts: u.n = data
        @test worst_t <= 1.0e-13                # circles: u.tau = f(R)
        @test abs(maximum(abs, u) - abs(g275_f(1.0))) <= 1.0e-3
        println("G275 a3: max|u.n - data| = ", worst_n,
                "  max|u.tau - f(R)| = ", worst_t)
    end

    @testset "(a4) inhomogeneous free-slip row (wall_un on a :free circle)" begin
        g, qt, ss = g275_space(2, 2)
        sq = ss.su
        nu = sq.N
        # zero net flux over the wedge arc: int_0^{pi/2} cos(2 theta) dtheta = 0
        gfun = (x, y) -> cos(2.0 * atan(y, x))
        wbc = (:free, :free, :rigid, :rigid)
        wun = (gfun, nothing, nothing, nothing)
        ws = g275_wall_system(g, ss, qt, wbc; wall_un=wun)
        # (i) the affine *data* u0 has n.u = g exactly at every free wall node
        wn = 0.0
        checked = 0
        for (node, cs) in ws.conds
            c = cs[1]
            c.wi == 1 || continue
            (haskey(ws.wd.fixed, node) && haskey(ws.wd.fixed, nu + node)) &&
                continue
            checked += 1
            wn = max(wn, abs(c.nx * ws.u0[node] + c.ny * ws.u0[nu + node] -
                             gfun(c.x, c.y)))
        end
        @test checked > 10
        @test wn <= 1.0e-14
        # (ii) the solved field has n.u = g at the curved wall, and a nonzero
        # flow (the data are inhomogeneous)
        u, p, = solve_semquad_stokes_var(ss, qt, g275_one, g275_zero,
                                         g275_zero, g275_zero, g275_zero;
                                         linear=:direct, bc=:free,
                                         wall_bc=wbc, wall_un=wun)
        wn2 = 0.0
        for (node, cs) in ws.conds
            c = cs[1]
            c.wi == 1 || continue
            (haskey(ws.wd.fixed, node) && haskey(ws.wd.fixed, nu + node)) &&
                continue
            wn2 = max(wn2, abs(c.nx * u[node] + c.ny * u[nu + node] -
                               gfun(c.x, c.y)))
        end
        @test wn2 <= 1.0e-12
        @test maximum(abs, u) > 1.0e-6
        println("G275 a4: |u0.n - g| = ", wn, "  |u.n - g| = ", wn2,
                "  |u|max = ", maximum(abs, u))
    end

    @testset "(a5) corner overwrite rule" begin
        g, qt, ss = g275_space(2, 2)
        sq = ss.su
        nu = sq.N
        # free/free: two independent rows solve the 2x2 -> both components fixed
        wsf = g275_wall_system(g, ss, qt, (:free, :free, :free, :free))
        cf = [node for (node, cs) in wsf.conds if length(cs) >= 2]
        @test length(cf) == 4
        @test all(n -> wsf.wd.fixed[n] == 0.0 && wsf.wd.fixed[nu + n] == 0.0, cf)
        # rigid circles (Couette data) + free cuts: the free cut is written last
        # and owns the corner (box rule); the circle's tangential data there are
        # overwritten by the free row u.n_cut = 0.
        wu = (x, y) -> g275_f(hypot(x, y))
        wsm = g275_wall_system(g, ss, qt, (:rigid, :rigid, :free, :free);
                               wall_u=(wu, wu, nothing, nothing))
        cm = [node for (node, cs) in wsm.conds if length(cs) >= 2]
        @test length(cm) == 4
        for node in cm
            cs = wsm.conds[node]
            ccut = cs[end]                     # the later wall: a free cut
            @test ccut.wi == 3 || ccut.wi == 4
            ux = wsm.wd.fixed[node]
            uy = wsm.wd.fixed[nu + node]
            # the free row is satisfied ...
            @test abs(ccut.nx * ux + ccut.ny * uy) <= 1.0e-12
            # ... and the circle data are NOT kept (the corner is not the
            # circle's tangential value)
            @test hypot(ux, uy) <= 1.0e-12
        end
        # control: the circle data really are O(1) there (so the check above is
        # not vacuous): f at the two radii
        @test abs(g275_f(2.22)) > 1.0
    end

    # ======================================================================
    # (b) circular Couette and the open-wall traction data
    # ======================================================================
    @testset "(b1) circular Couette: the wedge-analytic gate" begin
        # The ring's only walls are the circles and Couette is tangential there,
        # so the ring is where free-slip Couette is exact; on a wedge the radial
        # cuts cut the flow and free slip there is inconsistent.  The exact
        # wedge problem is therefore the *restriction* with Dirichlet data on
        # all four walls (circle tangential + cut normal data).
        eus = Float64[]
        ehs = Float64[]
        tq = Float64[]
        for level in 1:3
            g = PolarAnnulus(1.0, 2.0, G275_PHI)
            qt = g275_tree(level; R0=1.0, R1=2.0)
            ss = make_semstokes(qt, 2; geom=g)
            wu = (x, y) -> g275_f(hypot(x, y))
            un_lo = (x, y) -> -g275_f(hypot(x, y))
            un_hi = (x, y) -> +g275_f(hypot(x, y))
            u, p, = solve_semquad_stokes_var(ss, qt, g275_one, g275_zero,
                                             g275_zero, g275_zero, g275_zero;
                                             linear=:direct, bc=:rigid,
                                             wall_bc=(:rigid, :rigid, :rigid,
                                                      :rigid),
                                             wall_u=(wu, wu, nothing, nothing),
                                             wall_un=(nothing, nothing, un_lo,
                                                      un_hi))
            eu, eh = g275_couette_errors(g, ss, u)
            Tin = g275_wall_torque(g, ss, u, p, :xminus)
            Tout = g275_wall_torque(g, ss, u, p, :xplus)
            push!(eus, eu); push!(ehs, eh)
            push!(tq, abs(Tin - G275_COUETTE_TORQUE_IN))
            if level == 3
                println("G275 b1 level 3: T_in = ", Tin, " (exact ",
                        G275_COUETTE_TORQUE_IN, ")  T_out = ", Tout,
                        "  T_in+T_out = ", Tin + Tout)
                @test eu <= 3.0e-4              # measured 9.6e-5
                @test eh <= 2.0e-2              # measured 5.0e-3
                @test abs(Tin - G275_COUETTE_TORQUE_IN) <= 0.1
                @test abs(Tout - G275_COUETTE_TORQUE_OUT) <= 0.1
                @test abs(Tin + Tout) <= 5.0e-2 # torque balance (momentum)
                @test abs(maximum(abs, u) - abs(g275_f(1.0))) <= 1.0e-9
            end
        end
        # observed orders: velocity L2 3 / H1 2 (design/ASPECT orders); the
        # boundary-functional torque is one order lower than H1 in the worst
        # case, measured >= 1.5
        ord_u = log2(eus[2] / eus[3])
        ord_h = log2(ehs[2] / ehs[3])
        ord_t = log2(tq[2] / tq[3])
        println("G275 b1 Couette: |u|L2 = ", eus,
                "  |u|H1 = ", ehs, "  Tin_err = ", tq)
        println("G275 b1 orders (level2->3): u_L2 = ", ord_u,
                "  u_H1 = ", ord_h, "  torque = ", ord_t)
        @test ord_u >= 2.6
        @test ord_h >= 1.7
        @test ord_t >= 1.4
        # free-slip-cut control: the same circle data with free-slip cuts is NOT
        # the Couette restriction (the cuts cut the through-flow)
        g = PolarAnnulus(1.0, 2.0, G275_PHI)
        qt = g275_tree(2; R0=1.0, R1=2.0)
        ss = make_semstokes(qt, 2; geom=g)
        wu = (x, y) -> g275_f(hypot(x, y))
        ufs, pfs, = solve_semquad_stokes_var(ss, qt, g275_one, g275_zero,
                                             g275_zero, g275_zero, g275_zero;
                                             linear=:direct, bc=:rigid,
                                             wall_bc=(:rigid, :rigid, :free,
                                                      :free),
                                             wall_u=(wu, wu, nothing, nothing))
        nu = ss.su.N
        ux = ufs[1:nu]
        uy = ufs[nu+1:end]
        exx = [g275_couette_u(c[1], c[2])[1] for c in ss.su.coords]
        exy = [g275_couette_u(c[1], c[2])[2] for c in ss.su.coords]
        diff = maximum(hypot.(ux - exx, uy - exy))
        @test diff > 0.1                        # O(1): not the Couette problem
    end

    @testset "(b2) open walls: traction data in the mapped n/tau frame" begin
        # exact azimuthal flow u = (B/r) e_phi, B = -tau0 R0^2/(2 eta), p = 0:
        # circles :open with wall_t = t.tau (outward normal!), cuts rigid with
        # the exact normal through-flow.  The pressure gauge is derived (:none)
        # and the tangential traction data fix the shear.
        tau0 = 0.7
        Bco = -tau0 * 1.0 / 2.0
        uth(r) = Bco / r
        g = PolarAnnulus(1.0, 2.0, G275_PHI)
        qt = g275_tree(2; R0=1.0, R1=2.0)
        ss = make_semstokes(qt, 2; geom=g)
        nu = ss.su.N
        t_in = (x, y) -> 2.0 * Bco / 1.0^2      # t.tau = -sigma_rtheta on inner
        t_out = (x, y) -> -2.0 * Bco / 2.0^2    # t.tau = +sigma_rtheta on outer
        tn0 = (x, y) -> 0.0
        un_lo = (x, y) -> -uth(hypot(x, y))
        un_hi = (x, y) -> +uth(hypot(x, y))
        u, p, = solve_semquad_stokes_var(ss, qt, g275_one, g275_zero,
                                         g275_zero, g275_zero, g275_zero;
                                         linear=:direct, bc=:rigid,
                                         wall_bc=(:open, :open, :rigid, :rigid),
                                         wall_t=(t_in, t_out, nothing, nothing),
                                         wall_tn=(tn0, tn0, nothing, nothing),
                                         wall_un=(nothing, nothing, un_lo, un_hi))
        ex = zeros(2 * nu)
        for (gid, (x, y)) in enumerate(ss.su.coords)
            r = hypot(x, y)
            ex[gid] = -uth(r) * y / r
            ex[nu + gid] = uth(r) * x / r
        end
        @test maximum(abs, u - ex) <= 1.0e-4    # measured 7.1e-6
        @test maximum(abs, p) <= 1.0e-3         # p = 0 analytically
        println("G275 b2: max|u-uexact| = ", maximum(abs, u - ex),
                "  max|p| = ", maximum(abs, p))
        @test MS275.LAST_SEMSTOKES_PRESSURE_GAUGE[] === :none
        # uniform normal traction: t.n = -p0 on the open circles, rigid cuts;
        # exact u = 0, p = p0 (the traction level *is* the dynamic pressure)
        p0 = 0.3
        tno = (x, y) -> -p0
        u2, p2, = solve_semquad_stokes_var(ss, qt, g275_one, g275_zero,
                                           g275_zero, g275_zero, g275_zero;
                                           linear=:direct, bc=:rigid,
                                           wall_bc=(:open, :open, :rigid,
                                                    :rigid),
                                           wall_tn=(tno, tno, nothing, nothing))
        @test maximum(abs, u2) <= 1.0e-12
        @test maximum(abs, p2 .- p0) <= 1.0e-10
        @test MS275.LAST_SEMSTOKES_PRESSURE_GAUGE[] === :none
    end

    # ======================================================================
    # (c) manufactured Stokes convergence on the annulus
    # ======================================================================
    @testset "(c) manufactured Stokes convergence (free-slip cuts)" begin
        eus = Float64[]; ehs = Float64[]; eps_ = Float64[]
        for level in 1:3
            g = PolarAnnulus(1.0, 2.0, G275_PHI)
            qt = g275_tree(level; R0=1.0, R1=2.0)
            ss = make_semstokes(qt, 2; geom=g)
            u, p, = solve_semquad_stokes_var(ss, qt, g275_one,
                                             g275_manu_fx, g275_manu_fz,
                                             g275_zero, g275_zero;
                                             linear=:direct, bc=:rigid,
                                             wall_bc=(:rigid, :rigid, :free,
                                                      :free),
                                             wall_u=(g275_manu_wall_u,
                                                     g275_manu_wall_u,
                                                     nothing, nothing))
            eu, eh, ep = g275_manu_errors(g, ss, u, p)
            push!(eus, eu); push!(ehs, eh); push!(eps_, ep)
            # the free-slip cuts carry a nonzero tangential velocity in the
            # exact field, so the natural condition is genuinely exercised
            if level == 1
                nu = ss.su.N
                bmask = MS275._semquad_boundary_face_mask(qt, ss.su)
                conds = MS275._semquad_wall_frames(g, ss.su, bmask)
                tmax = 0.0
                for (node, cs) in conds
                    c = cs[1]
                    c.wi >= 3 || continue
                    tmax = max(tmax, abs(c.tx * u[node] +
                                         c.ty * u[nu + node]))
                end
                @test tmax > 0.1
            end
        end
        # expected Q2/Q1 orders: u_L2 = p+1 = 3, u_H1 = p = 2, p_L2 = p = 2
        ord_u = log2(eus[2] / eus[3])
        ord_h = log2(ehs[2] / ehs[3])
        ord_p = log2(eps_[2] / eps_[3])
        println("G275 c manufactured: u_L2 = ", eus, "  u_H1 = ", ehs,
                "  p_L2 = ", eps_)
        println("G275 c orders (level2->3): u_L2 = ", ord_u, "  u_H1 = ", ord_h,
                "  p_L2 = ", ord_p)
        @test ord_u >= 2.6                     # measured 2.97
        @test ord_h >= 1.7                     # measured 1.97
        @test ord_p >= 1.7                     # measured 1.99
        @test eus[3] <= 5.0e-4                 # measured 2.1e-4
        @test ehs[3] <= 3.0e-2                 # measured 1.1e-2
        @test eps_[3] <= 4.0e-2                # measured 1.4e-2
        # divergence-free: the exact velocity is divergence free and the
        # constraint is imposed, so the discrete field must be too
        g = PolarAnnulus(1.0, 2.0, G275_PHI)
        qt = g275_tree(2; R0=1.0, R1=2.0)
        ss = make_semstokes(qt, 2; geom=g)
        nu = ss.su.N
        u, p, = solve_semquad_stokes_var(ss, qt, g275_one, g275_manu_fx,
                                         g275_manu_fz, g275_zero, g275_zero;
                                         linear=:direct, bc=:rigid,
                                         wall_bc=(:rigid, :rigid, :free, :free),
                                         wall_u=(g275_manu_wall_u,
                                                 g275_manu_wall_u, nothing,
                                                 nothing))
        @test abs(g275_div_volume(g, ss.su, u[1:nu], u[nu+1:end])) <= 1.0e-9
    end

    # ======================================================================
    # (d) the rotation gauge
    # ======================================================================
    @testset "(d) rotation gauge: zero net angular momentum" begin
        # free-slip circles + open cuts: a solid-body rotation is admissible;
        # the load is torque-free (radial buoyancy-like body force)
        thf = (x, y) -> 0.3 * (hypot(x, y) - G275_R0) * cos(2.0 * atan(y, x))
        fxf = (x, y) -> thf(x, y) * x / hypot(x, y)
        fzf = (x, y) -> thf(x, y) * y / hypot(x, y)
        wbc = (:free, :free, :open, :open)
        sols = Dict{Int,Any}()
        for level in (2, 3)
            g, qt, ss = g275_space(level, 2)
            sq = ss.su
            u, p, = solve_semquad_stokes_var(ss, qt, g275_one, fxf, fzf,
                                             g275_zero, g275_zero;
                                             linear=:direct, bc=:free,
                                             wall_bc=wbc)
            om = MS275.LAST_SEMSTOKES_ROTATION_SHIFT[]
            S, Q = MS275._semquad_rotation_moment(g, sq, u)
            @test MS275.LAST_SEMSTOKES_ROTATION_GAUGE[] === :rotation
            @test abs(om) * sqrt(Q) > 1.0e-3   # a nonzero rotation was removed
            @test abs(S) <= 1.0e-8             # measured 2.5e-9 / 1.6e-10
            # equivalence with the L2 projection: u is M-orthogonal to (-y,x)
            M = assemble_semquad_velocity_mass(sq)
            z = zeros(2 * sq.N)
            for (gid, (x, y)) in enumerate(sq.coords)
                z[gid] = -y
                z[sq.N + gid] = x
            end
            Mz = M * z
            Mu = M * u
            # The design's gauge is zero *angular momentum*; the L2 projection
            # onto the nodal mode differs by O(h^(p+1)) on the curved geometry.
            # Both statements are measured: the M-orthogonality residual is
            # small (the two conventions agree in the continuum limit) and the
            # discrete angular-momentum mode norm S(Z_h) differs from int r^2 by
            # that discretisation order.
            mres = abs(dot(z, Mu)) /
                   (sqrt(abs(dot(z, Mz))) * sqrt(abs(dot(u, Mu))))
            @test mres <= 1.0e-5
            SZ, Q = MS275._semquad_rotation_moment(g, sq, z)
            @test abs(SZ - Q) / Q <= 1.0e-6
            # the rotation actually removed is the nonzero pre-projection
            # angular momentum omega * S(Z_h)
            @test abs(om * SZ) >= 1.0e-3
            println("G275 d level ", level, ": omega = ", om, "  S_after = ", S,
                    "  S(Z_h) = ", SZ, "  int r^2 = ", Q,
                    "  |S(Z)-Q|/Q = ", abs(SZ - Q) / Q,
                    "  M-residual = ", mres)
            sols[level] = (g=g, sq=sq, u=copy(u), om=om, SZ=SZ)
        end
        # nested meshes: the projected (canonical) fields agree at common nodes,
        # i.e. the gauge is mesh-consistent
        s2 = sols[2]; s3 = sols[3]
        worst = 0.0
        ncommon = 0
        for (gid, (x, y)) in enumerate(s2.sq.coords)
            j = findfirst(c -> abs(c[1] - x) < 1.0e-12 &&
                               abs(c[2] - y) < 1.0e-12, s3.sq.coords)
            j === nothing && continue
            ncommon += 1
            N2 = s2.sq.N; N3 = s3.sq.N
            worst = max(worst, abs(s2.u[gid] - s3.u[j]),
                        abs(s2.u[N2 + gid] - s3.u[N3 + j]))
        end
        @test ncommon > 100
        @test worst <= 5.0e-4                 # measured 1.0e-4
        println("G275 d cross-mesh: common nodes = ", ncommon,
                "  projected max diff = ", worst)
        # the gauge is not a no-op: each level's *pre-projection* field carries
        # omega * S(Z_h) of angular momentum (asserted in the loop above), which
        # the returned field has removed
        @test abs(s2.om * s2.SZ) >= 1.0e-3
        @test abs(s3.om * s3.SZ) >= 1.0e-3
        # direct vs multigrid on the gauge configuration
        g, qt, ss = g275_space(2, 2)
        ud, pd, = solve_semquad_stokes_var(ss, qt, g275_one, fxf, fzf,
                                           g275_zero, g275_zero;
                                           linear=:direct, bc=:free,
                                           wall_bc=wbc)
        um, pm, = solve_semquad_stokes_var(ss, qt, g275_one, fxf, fzf,
                                           g275_zero, g275_zero;
                                           linear=:mg, bc=:free, wall_bc=wbc)
        @test MS275.LAST_SEMSTOKES_ROTATION_GAUGE[] === :rotation
        @test maximum(abs, ud - um) <= 1.0e-9
        @test MS275.LAST_SEMSTOKES_RESIDUAL[] <= 1.0e-8
        # refusals of configurations the gauge cannot fix:
        # (i) a torque-imbalanced load has no steady solution
        err = try
            solve_semquad_stokes_var(ss, qt, g275_one, g275_one, g275_zero,
                                     g275_zero, g275_zero;
                                     linear=:direct, bc=:free, wall_bc=wbc)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("net torque", sprint(showerror, err))
        # (ii) the pure-traction card (all four walls :open)
        @test_throws ArgumentError solve_semquad_stokes_var(
            ss, qt, g275_one, g275_zero, g275_zero, g275_zero, g275_zero;
            linear=:direct, bc=:rigid,
            wall_bc=(:open, :open, :open, :open))
        # (iii) a mapped configuration with a translation nullspace: open x-walls
        # + free y-walls on the box map leave the x-translation admissible
        gid = AffineBox((1.0, 0.0, 0.0, 1.0), (0.0, 0.0), 1.0)
        qtb = g275_tree(1)
        ssb = make_semstokes(qtb, 2; geom=gid)
        err2 = try
            solve_semquad_stokes_var(ssb, qtb, g275_one, g275_zero,
                                     g275_zero, g275_zero, g275_zero;
                                     linear=:direct, bc=:free,
                                     wall_bc=(:open, :open, :free, :free))
            nothing
        catch e
            e
        end
        @test err2 isa ArgumentError
        @test occursin("translation", sprint(showerror, err2))
    end

    # ======================================================================
    # (e) the annular initial condition and the driver
    # ======================================================================
    @testset "(e) annular IC and the free-slip driver" begin
        g = g275_geom()
        pr0 = Params(Ra=100.0, H=0.0, eta_law=:expT, eta_b=0.0, bc=:free)
        # the total temperatures against independently written formulas
        pts = ((1.5, 0.2), (2.0, 0.3), (1.4, 1.2), (1.22 + 0.4, 0.1))
        for (x, y) in pts
            @test abs(MS275._annulus_ic_total(:firedrake, g, x, y) -
                      g275_ic_firedrake(x, y)) <= 1.0e-15
            @test abs(MS275._annulus_ic_total(:aspect, g, x, y) -
                      g275_ic_aspect(x, y)) <= 1.0e-15
            # theta = T_total - Tbar(r): the base-state subtraction is explicit
            @test abs(MS275._annulus_ic_theta(:firedrake, g, pr0, x, y) -
                      (g275_ic_firedrake(x, y) -
                       MS275.semquad_base_T(pr0, g, x, y))) <= 1.0e-15
            @test abs(MS275._annulus_ic_theta(:aspect, g, pr0, x, y) -
                      (g275_ic_aspect(x, y) -
                       MS275.semquad_base_T(pr0, g, x, y))) <= 1.0e-15
        end
        # the wall values vanish (both ICs are the documented linear wall data)
        for kind in (:firedrake, :aspect)
            for phi in (0.0, 0.3, 0.7, 1.2)
                xin = G275_R0 * cos(phi)
                yin = G275_R0 * sin(phi)
                xout = G275_R1 * cos(phi)
                yout = G275_R1 * sin(phi)
                @test abs(MS275._annulus_ic_theta(kind, g, pr0, xin, yin)) <=
                      1.0e-14
                @test abs(MS275._annulus_ic_theta(kind, g, pr0, xout, yout)) <=
                      1.0e-13
            end
        end
        # the 4-fold angular pattern: equal at theta and theta + pi/2, opposite
        # at theta = pi/4 (cos(4 theta) = 1 at 0/pi/2 and -1 at pi/4), and the
        # perturbation amplitude is the documented one
        r = G275_R0 + 0.37
        h = r - G275_R0
        d = G275_R1 - G275_R0
        s = (G275_R1 - r) / d
        for kind in (:firedrake, :aspect)
            linpart = kind === :firedrake ? (G275_R1 - r) : s
            base = MS275.semquad_base_T(pr0, g, r, 0.0)
            t0 = MS275._annulus_ic_theta(kind, g, pr0, r, 0.0)
            tq = MS275._annulus_ic_theta(kind, g, pr0, 0.0, r)
            @test abs(t0 - tq) <= 1.0e-12
            pert0 = t0 - (linpart - base)
            if kind === :firedrake
                @test abs(pert0 - 0.02 * sinpi(h)) <= 1.0e-14
            else
                @test abs(pert0 - 0.2 * s * (1.0 - s)) <= 1.0e-14
            end
            th45 = MS275._annulus_ic_theta(kind, g, pr0, r * cos(0.25 * pi),
                                           r * sin(0.25 * pi))
            base45 = MS275.semquad_base_T(pr0, g, r * cos(0.25 * pi),
                                          r * sin(0.25 * pi))
            pert45 = th45 - (linpart - base45)
            @test abs(pert45 + pert0) <= 1.0e-13
            @test abs(pert0) > 1.0e-3
        end
        # driver: free-slip circles + free-slip cuts, the paper's IC
        dt = 1.0e-3
        pr = Params(Ra=100.0, H=0.0, eta_law=:expT, eta_b=0.0, bc=:free,
                    wall_bc=(:free, :free, :free, :free))
        function g275_run(; nsteps=3, kwargs...)
            qt = g275_tree(2)
            ss = make_semstokes(qt, 2; geom=g275_geom())
            return run_semquad_simulation(qt, pr, zeros(Float64, ss.su.N);
                                          geom=:annulus, annulus_R0=G275_R0,
                                          annulus_R1=G275_R1,
                                          annulus_Phi=G275_PHI, p=2,
                                          tstop=nsteps * dt, dt0=dt,
                                          dtmax=dt, dtmin=1.0e-16,
                                          linear=:direct, maxsteps=nsteps,
                                          amr_every=0, advection=:skew,
                                          maxpic=1, annulus_ic=:firedrake,
                                          verbose=false, kwargs...)
        end
        straight = g275_run()
        @test straight.steps == 3
        @test length(straight.history) == 3
        @test all(r -> length(r) == 18, straight.history)
        @test all(isfinite, vcat(straight.history...))
        @test all(isfinite, straight.theta) && all(isfinite, straight.ux) &&
              all(isfinite, straight.uz)
        @test geometry_identity(straight.ss.su.geom) ==
              geometry_identity(g275_geom())
        sq = straight.ss.su
        nu = sq.N
        # the IC was applied and stays a perturbation of the base state
        @test maximum(abs, straight.theta) <= 0.2
        @test minimum(straight.theta) >= -0.2
        # the wall rows are active in the driver: normal velocity at the
        # free-slip circles is zero (the IC is not a wall data check, the
        # velocity solve is)
        bmask = MS275._semquad_boundary_face_mask(straight.qt, sq)
        conds = MS275._semquad_wall_frames(g275_geom(), sq, bmask)
        wnorm = 0.0
        for (node, cs) in conds
            c = cs[1]
            wnorm = max(wnorm, abs(c.nx * straight.ux[node] +
                                   c.ny * straight.uz[node]))
        end
        @test wnorm <= 1.0e-10
        # history rows are the diagnostics of the returned state
        row = straight.history[end]
        nut, nub = semquad_nu_walls(sq, straight.qt, straight.theta, pr)
        @test row[3] == nut && row[4] == nub
        @test row[5] == semquad_vrms(sq, straight.qt, straight.ux, straight.uz)
        @test row[6] == semquad_mean_T(sq, straight.qt, straight.theta, pr)
        @test row[9] == semquad_max_divergence(sq, straight.qt, straight.ux,
                                               straight.uz)
        @test row[10] == minimum_leaf_height(straight.ss.su.geom, straight.qt)
        @test row[5] > 1.0e-3                     # a genuine flow
        # Nusselt pair: a real transient convection state around Nu = 1 (the
        # exact conduction state is Nu = 1; the initial perturbation drives it)
        @test isfinite(row[3]) && isfinite(row[4])
        @test abs(row[3] - 1.0) <= 0.5
        @test abs(row[4] - 1.0) <= 0.5
        # the identities the suite checks: int div u = 0 exactly (continuity)
        # and Ra W = U up to the free-slip boundary term (only the *normal*
        # velocity is Dirichlet on a free-slip wall, so the natural tangential
        # traction enters the identity at discretisation order; the all-rigid
        # G274 run had it exact, this one measures the boundary term)
        @test abs(row[18]) <= 1.0e-2
        umax = maximum(hypot.(straight.ux, straight.uz))
        area = total_measure(g275_geom(), sq)
        @test abs(g275_div_volume(g275_geom(), sq, straight.ux, straight.uz)) <=
              1.0e-8 * max(umax * area, 1.0)
        println("G275 e driver: Nu = (", row[3], ", ", row[4], ")  Vrms = ",
                row[5], "  meanT = ", row[6], "  maxdiv = ", row[9],
                "  bal = ", row[18], "  steps = ", straight.steps)
        # `:mg` and `:auto` paths: same documented tolerance, same gauge
        mg = g275_run(nsteps=2, linear=:mg)
        @test mg.steps == 2
        @test length(mg.history) == 2
        @test all(isfinite, vcat(mg.history...))
        au = g275_run(nsteps=2, linear=:auto)
        @test au.steps == 2
        @test all(isfinite, vcat(au.history...))
        # level-2 tree: auto resolves to direct (the documented crossover), so
        # the two runs are bitwise; the mg run agrees to its solve tolerance
        direct2 = g275_run(nsteps=2, linear=:direct)
        @test au.theta == direct2.theta
        @test abs(mg.history[end][5] - direct2.history[end][5]) <= 1.0e-6
        # IC knob refusals and the ring prerequisite
        err = try
            qt = g275_tree(2)
            ss = make_semstokes(qt, 2; geom=CartesianBox(qt))
            run_semquad_simulation(qt, pr, zeros(Float64, ss.su.N); p=2,
                                   tstop=dt, dt0=dt, maxsteps=1,
                                   linear=:direct, amr_every=0,
                                   annulus_ic=:firedrake, verbose=false)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("annulus", sprint(showerror, err))
        err = try
            g275_run(theta_ic=(x, y) -> 0.0)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("theta_ic", sprint(showerror, err))
        ringerr = try
            PolarAnnulus(G275_R0, G275_R1, 2.0 * pi; seam=:periodic)
            nothing
        catch e
            e
        end
        @test ringerr isa ArgumentError
        @test occursin("periodic", sprint(showerror, ringerr))
    end
end
