# 2D DG upwind advection: exact reproduction of streamwise-constant fields,
# diagonal advection convergence, and the inflow boundary data.
using Test
using LinearAlgebra
using QuaST

@testset "G16 2D DG upwind advection" begin
    # u = (1,0), T = y is constant along the streamlines
    for p in (1, 2, 3)
        dg = make_dg2d(4, 4, p)
        A, b = assemble_dg2d_advection(dg, 1.0, 0.0, (x, y) -> y)
        u = A \ b
        err = 0.0
        n = p + 1
        for ey in 1:dg.ney, ex in 1:dg.nex, j in 1:n, i in 1:n
            x = (ex - 1) * dg.hx + 0.5 * (dg.xi[i] + 1) * dg.hx
            y = (ey - 1) * dg.hy + 0.5 * (dg.xi[j] + 1) * dg.hy
            err = max(err, abs(u[(ey-1)*dg.nex*(n^2) + (ex-1)*n^2 + (j-1)*n + i] - y))
        end
        @test err < 1.0e-10
    end

    # u = (0,1), T = x
    dg = make_dg2d(4, 4, 2)
    A, b = assemble_dg2d_advection(dg, 0.0, 1.0, (x, y) -> x)
    u = A \ b
    @test maximum(abs, [dg2d_eval(dg, u, 0.13 + 0.1k, 0.27 + 0.05k) - (0.13 + 0.1k)
                        for k in 0:4]) < 1.0e-10

    # diagonal advection of sin(pi(x-y)): convergence under h-refinement
    ux = uy = 1.0 / sqrt(2.0)
    f(x, y) = sinpi(x - y)
    errs = Float64[]
    for ne in (4, 8)
        dg = make_dg2d(ne, ne, 2)
        A, b = assemble_dg2d_advection(dg, ux, uy, f)
        u = A \ b
        e = 0.0
        for k in 0:20, l in 0:20
            x = 0.02 + 0.96k / 20
            y = 0.02 + 0.96l / 20
            e = max(e, abs(dg2d_eval(dg, u, x, y) - f(x, y)))
        end
        push!(errs, e)
    end
    @test errs[2] < 0.3 * errs[1]     # at least first-order reduction
    @test errs[2] < 5.0e-2
end
