# Quadtree SEM step 1: hanging-node constraints for the continuous Q_p space.
# A linear field must be reproduced exactly on uniform and 2:1 mixed meshes, and
# every constrained dof must equal the coarse-face interpolation of its masters.
using Test
using LinearAlgebra
using MantleSEM

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
    function l2e(sq, u)
        e = 0.0; M = 20
        for k in 0:M, l in 0:M
            x = 1.0e-6 + (1 - 2.0e-6) * k / M
            y = 1.0e-6 + (1 - 2.0e-6) * l / M
            e += (semquad_eval(sq, u, x, y) - exact(x, y))^2
        end
        sqrt(e / (M + 1)^2)
    end
    qt = Quadtree(8)
    refine!(qt, (0, 0, 0))
    errs = Float64[]
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
        push!(errs, l2e(sq, u))
    end
    @test log2(errs[1] / errs[2]) > 2.0
    @test log2(errs[2] / errs[3]) > 2.0
end
