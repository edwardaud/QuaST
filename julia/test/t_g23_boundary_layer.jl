# Boundary-layer advection-diffusion: convergence order and AMR efficiency.
# T = exp(-y/lambda) sin(pi x), u = (0,-1) (inflow at the top), kappa = 1.
using Test
using LinearAlgebra
using MantleSEM

const LAMBDA = 0.1
_bl(x, y) = exp(-y / LAMBDA) * sinpi(x)
_src(x, y) = (1 / LAMBDA + pi^2 - 1 / LAMBDA^2) * _bl(x, y)

function bl_solve(qt, p)
    dg = make_dgquad(qt, p)
    Ad, bd = assemble_dgquad_poisson(dg, qt, _src, _bl; sigma_C=50.0)
    Aa, ba = assemble_dgquad_advection(dg, qt, 0.0, -1.0, _bl)
    u = (Ad + Aa) \ (bd + ba)
    e = 0.0
    M = 24
    for k in 0:M, l in 0:M
        x = 1.0e-7 + (1 - 2.0e-7) * k / M
        y = 1.0e-7 + (1 - 2.0e-7) * l / M
        e += (dgquad_eval(dg, u, x, y) - _bl(x, y))^2
    end
    return dg.N, sqrt(e / (M + 1)^2), dg, u
end

@testset "G23 boundary-layer advection-diffusion" begin
    p = 2
    errs = Float64[]
    dofs = Int[]
    for lev in 2:4
        qt = Quadtree(8)
        for _ in 1:lev
            for lf in collect(qt.leaves)
                refine!(qt, lf)
            end
        end
        d, e, _, _ = bl_solve(qt, p)
        push!(dofs, d); push!(errs, e)
    end
    @test log2(errs[1] / errs[2]) > 2.5
    @test log2(errs[2] / errs[3]) > 2.5

    # adaptive from level 2, three cycles
    qt = Quadtree(8)
    for _ in 1:2
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
    end
    d, e, dg, u = bl_solve(qt, p)
    for _ in 1:3
        eta = dgquad_jump_indicator(dg, qt, u)
        amr_cycle!(qt, eta; refine_frac=0.3)
        d, e, dg, u = bl_solve(qt, p)
    end
    @test d <= 0.5 * dofs[end]
    @test e < 1.5 * errs[end]
    @test total_area(qt) ≈ 1.0
    @test max_face_level_jump(qt) <= 1
    low = count(lf -> (lf[3] + 0.5) * 2.0^(-lf[1]) < 0.25, qt.leaves)
    @test low / length(qt.leaves) > 0.6
end
