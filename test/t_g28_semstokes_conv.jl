# G28: manufactured Stokes convergence on mixed quadtrees (constant-viscosity
# reference path) -- remediation batch P0 part 1, AUDIT_VV F3.
#
# u = (sin(pi x) cos(pi y), -cos(pi x) sin(pi y)) (divergence free),
# p = sin(pi x) sin(pi y); f = -Delta u + grad p.  With Q_2/Q_1 the velocity
# error must be O(h^3) in L2 and O(h^2) in H1, and the (mean-subtracted)
# pressure error O(h^2).
#
# **Strengthened in this batch.**  The old gate measured a pointwise 21x21
# sample norm and asserted `log2(ratio) > 2.0` (velocity L2) and `> 1.0`
# (pressure) per level -- thresholds a first-order-pressure or 2.0-order
# velocity implementation passes (AUDIT_VV F3).  The gate now:
#   * measures the errors with a **quadrature norm** of the SEM field (the GLL
#     field integrated with the mapped element metric, exact for the Q_2/Q_1
#     interpolants) instead of a pointwise sample;
#   * uses **four levels** (h = 1/2, 1/4, 1/8, 1/16, the same mixed-tree
#     construction as before, extended by one refinement round) and asserts the
#     **observed order** of every consecutive pair against the G269-style
#     margins: velocity L2 >= 2.7 (expected 3), velocity H1 >= 1.8 (expected 2),
#     pressure L2 >= 1.7 (expected 2);
#   * prints the measured order table, so the evidence is in the log.
#
# Old thresholds: 2.0 / (none) / 1.0.  New: 2.7 / 1.8 / 1.7.  Measured orders
# on this fixture (this campaign): velocity L2 3.197, 3.061, 3.022; velocity H1
# 2.139, 2.026, 2.015; pressure L2 2.419, 2.220, 2.079.  Nothing was loosened;
# an implementation at order 2.2 velocity / 1.5 pressure is now red.
using Test
using LinearAlgebra
using QuaST

# Quadrature errors of the Stokes field: the interpolated velocity/pressure are
# integrated against the analytic manufactured solution with the element metric
# (a proper norm of the discrete field, not a point sample).
function g28_errors(g, ss, u, pv, uex, graduex, pex)
    sq = ss.su
    nu = sq.N; p = sq.p; q = ss.q
    fdq = make_fedata(p, p)
    fd = make_fedata(p, q)
    rule_p = gll_rule(q)
    eu2 = 0.0; eh1 = 0.0; ep2 = 0.0; epm = 0.0; area = 0.0
    for lf in sq.leaves
        m = leaf_metric(g, sq, lf, fdq)
        ids = sq.elem[lf]; pids = ss.sp.elem[lf]
        for qy in 1:fdq.nq, qx in 1:fdq.nq
            w = QuaST.quad_measure(m, fdq, qx, qy)
            x, y = QuaST.quad_point(m, sq, lf, fdq, qx, qy)
            uxq = 0.0; uzq = 0.0
            dux = (0.0, 0.0); duz = (0.0, 0.0)
            for j in 1:p+1, i in 1:p+1
                k = (j - 1) * (p + 1) + i
                b = fdq.phi_v[i, qx] * fdq.phi_v[j, qy]
                uxq += u[ids[k]] * b; uzq += u[nu + ids[k]] * b
                gx, gz = QuaST.basis_grad(m, fdq, qx, qy, i, j)
                dux = (dux[1] + u[ids[k]] * gx, dux[2] + u[ids[k]] * gz)
                duz = (duz[1] + u[nu + ids[k]] * gx,
                       duz[2] + u[nu + ids[k]] * gz)
            end
            ue = uex(x, y); ge = graduex(x, y)
            eu2 += w * ((uxq - ue[1])^2 + (uzq - ue[2])^2)
            eh1 += w * ((dux[1] - ge[1])^2 + (dux[2] - ge[2])^2 +
                        (duz[1] - ge[3])^2 + (duz[2] - ge[4])^2)
            Li, _ = lagrange_eval(rule_p, fdq.gl_x[qx])
            Lj, _ = lagrange_eval(rule_p, fdq.gl_x[qy])
            phq = 0.0
            for j in 1:q+1, i in 1:q+1
                k = (j - 1) * (q + 1) + i
                phq += pv[pids[k]] * Li[i] * Lj[j]
            end
            ep2 += w * (phq - pex(x, y))^2
            epm += w * (phq - pex(x, y))
            area += w
        end
    end
    return sqrt(eu2 / area), sqrt(eh1 / area),
           sqrt(max(ep2 / area - (epm / area)^2, 0.0))
end

g28_order(e1, e2) = log2(e1 / e2)

@testset "G28 quadtree Stokes convergence" begin
    ux_ex(x, y) = sinpi(x) * cospi(y)
    uz_ex(x, y) = -cospi(x) * sinpi(y)
    p_ex(x, y) = sinpi(x) * sinpi(y)
    fx(x, y) = 2 * pi^2 * ux_ex(x, y) + pi * cospi(x) * sinpi(y)
    fz(x, y) = 2 * pi^2 * uz_ex(x, y) + pi * sinpi(x) * cospi(y)
    uex(x, y) = (ux_ex(x, y), uz_ex(x, y))
    gradex(x, y) = (pi * cospi(x) * cospi(y), -pi * sinpi(x) * sinpi(y),
                    pi * sinpi(x) * sinpi(y), -pi * cospi(x) * cospi(y))

    qt = Quadtree(9)
    hs = Float64[]; eu2 = Float64[]; eu1 = Float64[]; ep2 = Float64[]
    for step in 1:4
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        base = minimum(l for (l, _, _) in qt.leaves)
        refine_where!(qt, (l, ix, iy) -> l < base + 1 &&
                      abs((ix + 0.5) / (1 << l) - 0.5) < 0.25 &&
                      abs((iy + 0.5) / (1 << l) - 0.5) < 0.25)
        @test max_face_level_jump(qt) <= 1
        ss = make_semstokes(qt, 2)
        u, pv, _, _, _ = solve_semquad_stokes(ss, qt, fx, fz, ux_ex, uz_ex)
        g = CartesianBox(qt)
        a, b, c = g28_errors(g, ss, u, pv, uex, gradex, p_ex)
        push!(hs, 1.0 / (1 << base)); push!(eu2, a); push!(eu1, b); push!(ep2, c)
        # the flow is genuine: the load produces an O(1) velocity (a dropped load
        # would give the zero solution, whose errors would "converge" trivially)
        @test maximum(abs, u) > 0.1
    end
    ou2 = [g28_order(eu2[k], eu2[k + 1]) for k in 1:3]
    ou1 = [g28_order(eu1[k], eu1[k + 1]) for k in 1:3]
    op2 = [g28_order(ep2[k], ep2[k + 1]) for k in 1:3]
    println("G28 quadtree Stokes convergence (quadrature norm, p = 2):")
    println("   h        u_L2        ord      u_H1        ord      p_L2        ord")
    for k in eachindex(hs)
        println("  ", rpad(hs[k], 8),
                rpad(round(eu2[k], sigdigits=5), 11),
                rpad(k > 1 ? round(ou2[k - 1], digits=3) : "  -", 9),
                rpad(round(eu1[k], sigdigits=5), 11),
                rpad(k > 1 ? round(ou1[k - 1], digits=3) : "  -", 9),
                rpad(round(ep2[k], sigdigits=5), 11),
                k > 1 ? round(op2[k - 1], digits=3) : "  -")
    end
    for (nm, e) in (("u L2", eu2), ("u H1", eu1), ("p L2", ep2))
        for k in 1:(length(e) - 1)
            @test e[k + 1] < e[k]
        end
        @test all(isfinite, e)
    end
    # observed order on every consecutive pair, not a per-level floor
    for o in ou2
        @test o >= 2.7                 # design: velocity L2 order 3
    end
    for o in ou1
        @test o >= 1.8                 # design: velocity H1 order 2
    end
    for o in op2
        @test o >= 1.7                 # design: pressure L2 order 2
    end
    # and the finest-pair measurement repeats the design order within the
    # G269-style +/-0.3 margin (a 2.4-order implementation is now red)
    @test ou2[end] >= 2.7
    @test ou1[end] >= 1.8
    @test op2[end] >= 1.7
end
