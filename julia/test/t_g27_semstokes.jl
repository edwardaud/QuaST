# Quadtree Taylor-Hood Stokes: the polynomial field u=(2xy,-y^2), p=0 is
# divergence free and satisfies -div(2 eps(u)) = (0,2); Q_2/Q_1 must reproduce
# it exactly on uniform and 2:1 mixed quadtrees.
using Test
using LinearAlgebra
using MantleSEM

@testset "G27 quadtree Stokes polynomial" begin
    ux_ex(x, y) = 2.0 * x * y
    uz_ex(x, y) = -y^2
    fx0(x, y) = 0.0
    fz0(x, y) = 2.0
    for build in (() -> begin
                      qt = Quadtree(4); refine!(qt, (0, 0, 0)); qt
                  end,
                  () -> begin
                      qt = Quadtree(4); refine!(qt, (0, 0, 0)); refine!(qt, (1, 0, 0)); balance!(qt); qt
                  end)
        qt = build()
        ss = make_semstokes(qt, 2)
        u, pv, fixed, cons2, cons_p = solve_semquad_stokes(ss, qt, fx0, fz0, ux_ex, uz_ex)
        nu = ss.su.N
        n = ss.p + 1
        err = 0.0
        for lf in ss.su.leaves
            h = 2.0^(-lf[1]); x0 = lf[2] * h; y0 = lf[3] * h
            ids = ss.su.elem[lf]
            for j in 1:n, i in 1:n
                x = x0 + 0.5 * (ss.su.xi[i] + 1.0) * h
                y = y0 + 0.5 * (ss.su.xi[j] + 1.0) * h
                g = ids[(j - 1) * n + i]
                err = max(err, abs(u[g] - ux_ex(x, y)), abs(u[nu + g] - uz_ex(x, y)))
            end
        end
        @test err < 1.0e-8
        # pressure is constant zero up to the pin
        @test maximum(abs, pv) < 1.0e-8
    end
end
