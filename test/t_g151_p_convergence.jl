# G151: p-convergence and manufactured solutions for Poisson and the Stokes pair.
using Test, QuaST, LinearAlgebra

# G151: p-convergence and manufactured solutions.
#
# this solver assumes "SEM => high order", but until now that was never measured.
# These checks pin the two properties that matter:
#   * fixed mesh, p = 2..6: the true L2 error (independent Gauss quadrature, NOT
#     dot(e, M e), which measures against the nodal interpolant) falls by a
#     large factor per order -- exponential, not algebraic;
#   * a non-uniform 2:1 quadtree with hanging-node C'AC constraints has the same
#     error-vs-p behaviour as the uniform mesh at the same h (the fine cells and
#     constraints add no measurable error);
#   * the Q_p/Q_{p-1} Stokes pair reproduces an exact polynomial to round-off
#     and the discrete divergence vanishes for it.
#
# The numbers here are deliberately only "factor > 5 per order" / "within 20% of
# uniform"; the measured ratios are much larger (see docs/dev/SOLVER.md).

function _g151_l2(sq, u, uex)
    p = sq.p
    n = p + 1
    rule = gll_rule(p)
    nq = p + 6
    gq, gw = gauss_legendre(nq)
    GL = zeros(Float64, n, nq)
    for k in 1:nq
        L, _ = lagrange_eval(rule, gq[k])
        GL[:, k] .= L
    end
    e2 = 0.0
    for lf in sq.leaves
        hx = QuaST.leaf_hx(sq, lf[1]); hy = QuaST.leaf_hy(sq, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = sq.elem[lf]
        for qy in 1:nq, qx in 1:nq
            x = x0 + 0.5 * (gq[qx] + 1.0) * hx
            y = y0 + 0.5 * (gq[qy] + 1.0) * hy
            uh = 0.0
            for j in 1:n, i in 1:n
                uh += u[ids[(j - 1) * n + i]] * GL[i, qx] * GL[j, qy]
            end
            d = uh - uex(x, y)
            e2 += 0.25 * hx * hy * gw[qx] * gw[qy] * d * d
        end
    end
    return sqrt(e2)
end

function _g151_uniform(level::Int)
    qt = Quadtree(level)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

@testset "G151 p-convergence and manufactured solutions" begin
    # ---- Poisson, fixed 4x4 mesh, p = 2..6 ----
    qt4 = _g151_uniform(2)
    uex = (x, y) -> sin(pi * x) * sin(pi * y)
    f = (x, y) -> 2 * pi^2 * uex(x, y)
    errs4 = Float64[]
    for p in 2:6
        sq = make_semquad(qt4, p)
        u, _ = solve_semquad_poisson(sq, qt4, f, (x, y) -> 0.0)
        push!(errs4, _g151_l2(sq, u, uex))
    end
    @test all(errs4[k + 1] < errs4[k] / 5 for k in 1:4)
    @test errs4[end] < 1.0e-8
    # the last ratio is the cleanest measure of exponential convergence
    @test errs4[end - 1] / errs4[end] > 20

    # ---- Poisson, non-uniform 2:1 tree (28 leaves) vs uniform 4x4 ----
    qtn = Quadtree(5)
    for _ in 1:2, lf in collect(qtn.leaves)
        refine!(qtn, lf)
    end
    for iy in 0:1, ix in 0:1
        refine!(qtn, (2, ix, iy))
    end
    balance!(qtn)
    @test length(qtn.leaves) == 28
    @test max_face_level_jump(qtn) == 1
    # The 28-leaf tree is 12 level-2 leaves + 16 level-3 leaves: 3/4 of the
    # area is at h=1/4 and 1/4 at h=1/8, so the right reference is the uniform
    # 4x4 mesh (h=1/4), not 8x8.  The error should be slightly *smaller* than
    # uniform (the refined quarter helps a little), never larger.
    qtu = _g151_uniform(2)
    for p in 2:4
        sqn = make_semquad(qtn, p)
        sqf = make_semquad(qtu, p)
        un, _ = solve_semquad_poisson(sqn, qtn, f, (x, y) -> 0.0)
        uf, _ = solve_semquad_poisson(sqf, qtu, f, (x, y) -> 0.0)
        en = _g151_l2(sqn, un, uex)
        ef = _g151_l2(sqf, uf, uex)
        # constraints may not dominate the error: the ratio stays near the
        # coarse-area fraction sqrt(3/4) ~ 0.87, and never exceeds 1.
        @test 0.80 < en / ef < 1.00
    end
    @test length(semquad_constraints(make_semquad(qtn, 2), qtn)) > 0

    # ---- Stokes: exact polynomial on the 5x2 / Lx=2.5 forest ----
    qt5 = Quadtree(3; cells_x=5, cells_y=2)
    for _ in 1:3, lf in collect(qt5.leaves)
        refine!(qt5, lf)
    end
    balance!(qt5)
    uxex = (x, y) -> 2.0 * x * y
    uzex = (x, y) -> -y^2
    for p in 2:4
        ss = make_semstokes(qt5, p)
        us, _, _, _, _ = solve_semquad_stokes_var(ss, qt5, (x, y) -> 1.0,
                                                  (x, y) -> 0.0, (x, y) -> 2.0,
                                                  uxex, uzex; bc=:rigid)
        ns = ss.su.N
        e = 0.0
        for (g, c) in enumerate(ss.su.coords)
            e = max(e, abs(us[g] - uxex(c...)), abs(us[ns + g] - uzex(c...)))
        end
        @test e < 1.0e-10
        @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-10
    end
end
