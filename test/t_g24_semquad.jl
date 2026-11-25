# Quadtree SEM step 1: hanging-node constraints for the continuous Q_p space,
# and the Poisson convergence order -- remediation batch P0 part 1,
# AUDIT_VV F3.
#
# First testset (unchanged): a linear field must be reproduced exactly on
# uniform and 2:1 mixed meshes, and every constrained dof must equal the
# coarse-face interpolation of its masters.
#
# Second testset (strengthened): exact(x,y) = sin(pi x) sin(pi y) solves
# -lap u = f on the mixed tree.  The old gate measured a pointwise 21x21 sample
# norm and asserted `log2(err[1]/err[2]) > 2.0` and `> 2.0` -- a 2.0-order
# implementation passes although Q_2 is expected to give L2 order 3
# (AUDIT_VV F3).  The gate now measures a **quadrature norm** of the SEM field
# (integrated with the element metric, exact for the Q_2 interpolant) on the
# same three mixed-tree levels and asserts the **observed order** of both
# consecutive pairs is >= 2.7 (G269-style margin: expected 3 minus 0.3), printing
# the table.
#
# Old thresholds: 2.0 and 2.0.  New: 2.7 (both pairs).  Measured orders on this
# fixture (this campaign): 3.075 and 3.040.  Nothing was loosened.
using Test
using LinearAlgebra
using QuaST

@testset "G24 quadtree SEM constraints" begin
    lin(x, y) = 1.0 + 0.3x - 0.2y
    zero_f(x, y) = 0.0

    # uniform 2x2: no constraints, exact linear
    qt = Quadtree(4)
    refine!(qt, (0, 0, 0))
    sq = make_semquad(qt, 2)
    u, cons = solve_semquad_poisson(sq, qt, zero_f, lin)
    @test isempty(cons)
    @test maximum(abs(u[i] - lin(sq.coords[i]...)) for i in 1:sq.N) < 1.0e-9

    # mixed 2:1 tree: constraints present and satisfied exactly
    qt2 = Quadtree(4)
    refine!(qt2, (0, 0, 0))
    refine!(qt2, (1, 0, 0))
    @test balance!(qt2)
    sq2 = make_semquad(qt2, 2)
    u2, cons2 = solve_semquad_poisson(sq2, qt2, zero_f, lin)
    @test !isempty(cons2)
    worst = 0.0
    for (c, (ms, ws)) in cons2
        worst = max(worst, abs(u2[c] - sum(w * u2[m] for (m, w) in zip(ms, ws))))
    end
    @test worst < 1.0e-9
    @test maximum(abs(u2[i] - lin(sq2.coords[i]...)) for i in 1:sq2.N) < 1.0e-8

    # refinement in the middle (not touching the domain boundary) keeps the
    # linear exactness and produces constraints
    qt3 = Quadtree(4)
    refine!(qt3, (0, 0, 0))
    for dy in 0:1, dx in 0:1
        refine!(qt3, (1, dx, dy))
    end
    refine!(qt3, (2, 1, 1))
    @test balance!(qt3)
    sq3 = make_semquad(qt3, 2)
    u3, cons3 = solve_semquad_poisson(sq3, qt3, zero_f, lin)
    @test !isempty(cons3)
    @test maximum(abs(u3[i] - lin(sq3.coords[i]...)) for i in 1:sq3.N) < 1.0e-8
end


@testset "G24 quadtree SEM Poisson convergence" begin
    exact(x, y) = sinpi(x) * sinpi(y)
    rhs(x, y) = 2 * pi^2 * exact(x, y)
    grad(x, y) = (pi * cospi(x) * sinpi(y), pi * sinpi(x) * cospi(y))
    # quadrature L2 / H1 error of the Q_2 field against the exact solution
    function g24_errors(g, sq, u)
        p = sq.p
        fd = make_fedata(p, p)
        e2 = 0.0; eh1 = 0.0; area = 0.0
        for lf in sq.leaves
            m = leaf_metric(g, sq, lf, fd)
            ids = sq.elem[lf]
            for qy in 1:fd.nq, qx in 1:fd.nq
                w = QuaST.quad_measure(m, fd, qx, qy)
                x, y = QuaST.quad_point(m, sq, lf, fd, qx, qy)
                uq = 0.0; gxq = 0.0; gzq = 0.0
                for j in 1:p+1, i in 1:p+1
                    k = (j - 1) * (p + 1) + i
                    uq += u[ids[k]] * fd.phi_v[i, qx] * fd.phi_v[j, qy]
                    gx, gz = QuaST.basis_grad(m, fd, qx, qy, i, j)
                    gxq += u[ids[k]] * gx; gzq += u[ids[k]] * gz
                end
                ge = grad(x, y)
                e2 += w * (uq - exact(x, y))^2
                eh1 += w * ((gxq - ge[1])^2 + (gzq - ge[2])^2)
                area += w
            end
        end
        return sqrt(e2 / area), sqrt(eh1 / area)
    end
    qt = Quadtree(8)
    refine!(qt, (0, 0, 0))
    hs = Float64[]; errs = Float64[]; errs_h1 = Float64[]
    for step in 1:3
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        base = minimum(l for (l, _, _) in qt.leaves)
        refine_where!(qt, (l, ix, iy) -> l < base + 2 &&
                      abs((ix + 0.5) / (1 << l) - 0.5) < 0.25 &&
                      abs((iy + 0.5) / (1 << l) - 0.5) < 0.25)
        @test max_face_level_jump(qt) <= 1
        sq = make_semquad(qt, 2)
        u, cons = solve_semquad_poisson(sq, qt, rhs, exact)
        l2, h1 = g24_errors(CartesianBox(qt), sq, u)
        push!(hs, 1.0 / (1 << base)); push!(errs, l2); push!(errs_h1, h1)
        @test maximum(abs, u) > 0.1      # a genuine solution, not the zero field
    end
    ords = [log2(errs[k] / errs[k + 1]) for k in 1:2]
    ords_h1 = [log2(errs_h1[k] / errs_h1[k + 1]) for k in 1:2]
    println("G24 mixed-tree Poisson convergence (quadrature norm, Q_2):")
    println("   h        L2          ord      H1          ord")
    for k in eachindex(hs)
        println("  ", rpad(hs[k], 8),
                rpad(round(errs[k], sigdigits=5), 11),
                rpad(k > 1 ? round(ords[k - 1], digits=3) : "  -", 9),
                rpad(round(errs_h1[k], sigdigits=5), 11),
                k > 1 ? round(ords_h1[k - 1], digits=3) : "  -")
    end
    @test errs[2] < errs[1] && errs[3] < errs[2]
    @test all(isfinite, errs)
    for o in ords
        @test o >= 2.7                 # design: Q_2 Poisson L2 order 3
    end
    # the H1 order is the lower one (2) and is pinned separately, so a wrong
    # derivative/metric cannot hide behind the L2 order
    for o in ords_h1
        @test o >= 1.8
    end
end
