# Quadtree DG Poisson: uniform tree must reproduce the dg2d solution, linear
# fields must be exact, and coarse-fine faces must be rejected explicitly.
using Test
using LinearAlgebra
using QuaST

function l2err_quad(dg, u, exact)
    e = 0.0
    M = 16
    for k in 0:M, l in 0:M
        x = 1.0e-6 + (1 - 2.0e-6) * k / M
        y = 1.0e-6 + (1 - 2.0e-6) * l / M
        e += (dgquad_eval(dg, u, x, y) - exact(x, y))^2
    end
    return sqrt(e / (M + 1)^2)
end

@testset "G19 quadtree DG Poisson" begin
    qt = Quadtree(4)
    refine!(qt, (0, 0, 0))
    for dy in 0:1, dx in 0:1
        refine!(qt, (1, dx, dy))
    end
    @test length(qt.leaves) == 16
    @test is_uniform(qt)

    exact(x, y) = sinpi(x) * sinpi(y)
    rhs(x, y) = 2 * pi^2 * exact(x, y)
    zero_bc(x, y) = 0.0

    dgq = make_dgquad(qt, 2)
    A, b = assemble_dgquad_poisson(dgq, qt, rhs, zero_bc; sigma_C=50.0)
    @test maximum(abs, A - A') < 1.0e-10
    u = A \ b

    dgd = make_dg2d(4, 4, 2)
    Ad, bd = assemble_dg2d_poisson(dgd, rhs, zero_bc; sigma_C=50.0)
    ud = Ad \ bd
    err = 0.0
    for k in 0:12, l in 0:12
        x = 1.0e-6 + (1 - 2.0e-6) * k / 12
        y = 1.0e-6 + (1 - 2.0e-6) * l / 12
        err = max(err, abs(dgquad_eval(dgq, u, x, y) - dg2d_eval(dgd, ud, x, y)))
    end
    @test err < 1.0e-8

    lin(x, y) = x + 2y
    A2, b2 = assemble_dgquad_poisson(dgq, qt, (x, y) -> 0.0, lin; sigma_C=50.0)
    u2 = A2 \ b2
    @test l2err_quad(dgq, u2, lin) < 1.0e-9

    # coarse-fine (hanging) faces: linear fields must still be exact
    qt2 = Quadtree(4)
    refine!(qt2, (0, 0, 0))
    refine!(qt2, (1, 0, 0))
    @test balance!(qt2)
    @test !is_uniform(qt2)
    dgq2 = make_dgquad(qt2, 2)
    A3, b3 = assemble_dgquad_poisson(dgq2, qt2, (x, y) -> 0.0, lin; sigma_C=50.0)
    u3 = A3 \ b3
    @test l2err_quad(dgq2, u3, lin) < 1.0e-8
end


@testset "G19 adaptive convergence and conservation" begin
    exact(x, y) = sinpi(x) * sinpi(y)
    rhs(x, y) = 2 * pi^2 * exact(x, y)
    zero_bc(x, y) = 0.0

    qt = Quadtree(9)
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
        @test total_area(qt) ≈ 1.0
        dg = make_dgquad(qt, 1)
        A, b = assemble_dgquad_poisson(dg, qt, rhs, zero_bc; sigma_C=50.0)
        u = A \ b
        push!(errs, l2err_quad(dg, u, exact))
    end
    @test log2(errs[1] / errs[2]) > 1.5
    @test log2(errs[2] / errs[3]) > 1.5

    # element-wise conservation on the finest adaptive mesh
    dg = make_dgquad(qt, 1)
    A, b = assemble_dgquad_poisson(dg, qt, rhs, zero_bc; sigma_C=50.0)
    u = A \ b
    r = A * u - b
    n = dg.p + 1
    worst = 0.0
    for lf in dg.leaves
        idx = [QuaST._gidq(dg, lf, i, j) for j in 1:n for i in 1:n]
        worst = max(worst, abs(sum(r[idx])))
    end
    @test worst < 1.0e-8 * max(1.0, norm(r, Inf))
end
