# AMR efficiency: at comparable (or fewer) degrees of freedom the adaptive mesh
# must beat the uniform mesh on a localized manufactured solution.
using Test
using LinearAlgebra
using QuaST

const SIG = 0.12
_r2(x, y) = (x - 0.5)^2 + (y - 0.5)^2
_ex(x, y) = exp(-_r2(x, y) / SIG^2)
_f(x, y) = -_ex(x, y) * (4 * _r2(x, y) / SIG^4 - 4 / SIG^2)

function err_solve(qt, p)
    dg = make_dgquad(qt, p)
    A, b = assemble_dgquad_poisson(dg, qt, _f, _ex; sigma_C=50.0)
    u = A \ b
    e = 0.0
    M = 20
    for k in 0:M, l in 0:M
        x = 1.0e-6 + (1 - 2.0e-6) * k / M
        y = 1.0e-6 + (1 - 2.0e-6) * l / M
        e += (dgquad_eval(dg, u, x, y) - _ex(x, y))^2
    end
    return dg.N, sqrt(e / (M + 1)^2)
end

@testset "G21 AMR efficiency" begin
    p = 2
    # uniform level 3 (64 leaves) and level 4 (256 leaves)
    qu3 = Quadtree(8)
    for _ in 1:3
        for lf in collect(qu3.leaves)
            refine!(qu3, lf)
        end
    end
    d3, e3 = err_solve(qu3, p)
    qu4 = Quadtree(8)
    for _ in 1:4
        for lf in collect(qu4.leaves)
            refine!(qu4, lf)
        end
    end
    d4, e4 = err_solve(qu4, p)

    # adaptive: start level 2, two and three cycles
    function adaptive(cycles)
        qt = Quadtree(8)
        for _ in 1:2
            for lf in collect(qt.leaves)
                refine!(qt, lf)
            end
        end
        dg = make_dgquad(qt, p)
        A, b = assemble_dgquad_poisson(dg, qt, _f, _ex; sigma_C=50.0)
        u = A \ b
        for _ in 1:cycles
            eta = dgquad_jump_indicator(dg, qt, u)
            amr_cycle!(qt, eta; refine_frac=0.3)
            dg = make_dgquad(qt, p)
            A, b = assemble_dgquad_poisson(dg, qt, _f, _ex; sigma_C=50.0)
            u = A \ b
        end
        return err_solve(qt, p)
    end
    da2, ea2 = adaptive(2)
    da3, ea3 = adaptive(3)

    # 2 cycles: at about the coarse dof count the adaptive error is much smaller
    @test da2 <= 1.2 * d3
    @test ea2 < 0.5 * e3
    # 3 cycles: same error as the finest uniform mesh with far fewer dofs
    @test ea3 < 1.2 * e4
    @test da3 < 0.6 * d4
end
