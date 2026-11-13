# 2D DG SIPG Poisson: convergence order, exact linear reproduction, symmetry
# and element-wise conservation.
using Test
using LinearAlgebra
using QuaST

function l2err(dg, u, exact)
    e = 0.0
    M = 20
    for k in 0:M, l in 0:M
        x = 1.0e-6 + (1 - 2.0e-6) * k / M
        y = 1.0e-6 + (1 - 2.0e-6) * l / M
        e += (dg2d_eval(dg, u, x, y) - exact(x, y))^2
    end
    return sqrt(e / (M + 1)^2)
end

@testset "G17 2D DG SIPG Poisson" begin
    exact(x, y) = sinpi(x) * sinpi(y)
    rhs(x, y) = 2 * pi^2 * exact(x, y)
    zero_bc(x, y) = 0.0

    lin(x, y) = x + 2y
    for p in (1, 2, 3)
        dg = make_dg2d(4, 4, p)
        A, b = assemble_dg2d_poisson(dg, (x, y) -> 0.0, lin; sigma_C=50.0)
        u = A \ b
        @test l2err(dg, u, lin) < 1.0e-9
    end

    dg8 = make_dg2d(8, 8, 1)
    A8, b8 = assemble_dg2d_poisson(dg8, rhs, zero_bc; sigma_C=50.0)
    e1a = l2err(dg8, A8 \ b8, exact)
    dg16 = make_dg2d(16, 16, 1)
    A16, b16 = assemble_dg2d_poisson(dg16, rhs, zero_bc; sigma_C=50.0)
    e1b = l2err(dg16, A16 \ b16, exact)
    @test log2(e1a / e1b) > 1.5

    dg4 = make_dg2d(4, 4, 2)
    A4, b4 = assemble_dg2d_poisson(dg4, rhs, zero_bc; sigma_C=50.0)
    e2a = l2err(dg4, A4 \ b4, exact)
    dg8b = make_dg2d(8, 8, 2)
    A8b, b8b = assemble_dg2d_poisson(dg8b, rhs, zero_bc; sigma_C=50.0)
    e2b = l2err(dg8b, A8b \ b8b, exact)
    @test log2(e2a / e2b) > 2.0

    dg = make_dg2d(4, 4, 2)
    A, b = assemble_dg2d_poisson(dg, rhs, zero_bc; sigma_C=50.0)
    @test maximum(abs, A - A') < 1.0e-10
    u = A \ b
    r = A * u - b
    n = dg.p + 1
    for ey in 1:dg.ney, ex in 1:dg.nex
        idx = (ey - 1) * dg.nex * n^2 + (ex - 1) * n^2 .+ (1:n^2)
        @test abs(sum(r[idx])) < 1.0e-8 * max(1.0, norm(r, Inf))
    end
end
