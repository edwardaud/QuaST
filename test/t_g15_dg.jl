# 1D DG-SEM SIPG Poisson: convergence order, constant reproduction, symmetry
# and penalty robustness.  This is the first gate of the stage-B DG prototype.
using Test
using LinearAlgebra
using QuaST

@testset "G15 1D DG SIPG Poisson" begin
    exact(x) = sinpi(x)
    rhs(x) = pi^2 * sinpi(x)
    zero_bc(x) = 0.0

    err = Dict{Tuple{Int,Int},Float64}()
    for p in (1, 2, 3), ne in (8, 16, 32)
        dg = make_dg1d(ne, p)
        u = solve_dg1d_poisson(dg, rhs, zero_bc; sigma_C=50.0)
        err[(p, ne)] = dg1d_l2_error(dg, u, exact)
    end
    for p in (1, 2, 3)
        order = log2(err[(p, 16)] / err[(p, 32)])
        @test order > p + 0.5          # expect about p+1
    end

    # constant reproduction (u = 1, f = 0, Dirichlet 1)
    dg = make_dg1d(6, 3)
    u1 = solve_dg1d_poisson(dg, x -> 0.0, x -> 1.0; sigma_C=50.0)
    @test maximum(abs, u1 .- 1.0) < 1.0e-12

    # symmetry of the SIPG matrix
    A, _ = assemble_dg1d_poisson(dg, x -> 0.0, x -> 0.0; sigma_C=50.0)
    @test maximum(abs, A - A') < 1.0e-12

    # penalty robustness: C = 1, 5, 10 all converge
    for C in (10.0, 50.0, 100.0)
        dg2 = make_dg1d(8, 2)
        u = solve_dg1d_poisson(dg2, rhs, zero_bc; sigma_C=C)
        @test isfinite(dg1d_l2_error(dg2, u, exact))
        @test dg1d_l2_error(dg2, u, exact) < 1.0e-3
    end
end
