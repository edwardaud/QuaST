using Test, MantleSEM, LinearAlgebra

# G145: generic damped Newton with a finite-difference Jacobian.
#
# The driver is validated on (1) a linear system (one step), (2) a smooth scalar
# root, (3) a coupled 2-D system, and (4) an impossible problem, where it must
# report converged == false rather than loop forever.

@testset "G145 generic Newton" begin
    # 1. linear system: x - A^{-1} b = 0 is solved in one step
    A = [4.0 1.0; 1.0 3.0]
    b = [1.0, 2.0]
    F1 = x -> A * x .- b
    x1, ok1, it1, h1 = newton_solve(F1, [0.0, 0.0])
    @test ok1
    @test it1 <= 2
    @test norm(A * x1 .- b) < 1.0e-9

    # 2. scalar: x^2 - 2 = 0
    F2 = x -> [x[1]^2 - 2.0]
    x2, ok2, it2, h2 = newton_solve(F2, [1.0])
    @test ok2
    @test abs(x2[1] - sqrt(2.0)) < 1.0e-8
    @test it2 <= 6
    @test issorted(h2[end-2:end], rev=true)     # residual decreasing

    # 3. coupled 2-D: x + y^2 = 3, x^2 - y = 1  (root ~ (1.5, 1.0))
    F3 = x -> [x[1] + x[2]^2 - 3.0, x[1]^2 - x[2] - 1.0]
    x3, ok3, it3, _ = newton_solve(F3, [1.0, 1.0])
    @test ok3
    @test norm(F3(x3)) < 1.0e-8
    @test it3 <= 8

    # 4. no root: |F| cannot go below 1 -> must report failure, not hang
    F4 = x -> [1.0 + x[1]^2]
    x4, ok4, it4, _ = newton_solve(F4, [0.0]; maxit=6)
    @test !ok4
    @test it4 <= 6

    # JFNK on the same problems (matrix-free Jacobian-vector products)
    xj, okj, _ = jfnk_solve(F1, [0.0, 0.0])
    @test okj
    @test norm(A * xj .- b) < 1.0e-8
    xj2, okj2, _ = jfnk_solve(F2, [1.0])
    @test okj2
    @test abs(xj2[1] - sqrt(2.0)) < 1.0e-7
    xj3, okj3, _ = jfnk_solve(F3, [1.0, 1.0])
    @test okj3
    @test norm(F3(xj3)) < 1.0e-7

    # the residual history starts at the initial residual
    @test h2[1] ≈ sqrt(abs(1.0^2 - 2.0)) rtol = 1.0e-12
end
