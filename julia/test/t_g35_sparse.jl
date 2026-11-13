# G35: sparse quadtree operators at a scale where dense assembly would be
# impractical.  The Poisson matrix must be sparse with O(N) nonzeros and the
# sparse LU must reproduce the manufactured solution; the Stokes saddle must
# solve the exact polynomial on a uniform tree.
using Test
using LinearAlgebra
using SparseArrays
using MantleSEM

function uniform_tree(level::Int)
    qt = Quadtree(level + 2)
    for _ in 1:level
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
    end
    return qt
end

@testset "G35 sparse quadtree operators" begin
    uex(x, y) = sinpi(x) * sinpi(y)
    rhs(x, y) = 2 * pi^2 * uex(x, y)

    qt = uniform_tree(5)
    sq = make_semquad(qt, 2)
    A, _ = assemble_semquad_poisson(sq, qt, rhs, (x, y) -> 0.0)
    @test issparse(A)
    @test nnz(A) < 100 * sq.N
    u, _ = solve_semquad_poisson(sq, qt, rhs, (x, y) -> 0.0)
    err = maximum(abs(u[i] - uex(sq.coords[i]...)) for i in 1:sq.N)
    @test err < 1.0e-3

    qts = uniform_tree(4)
    ss = make_semstokes(qts, 2)
    us, ps, _, _, _ = solve_semquad_stokes(ss, qts,
                                           (x, y) -> 0.0, (x, y) -> 2.0,
                                           (x, y) -> 2x * y, (x, y) -> -y^2)
    nu = ss.su.N
    errs = maximum(max(abs(us[g] - 2 * ss.su.coords[g][1] * ss.su.coords[g][2]),
                       abs(us[nu + g] + ss.su.coords[g][2]^2)) for g in 1:nu)
    @test errs < 1.0e-8
    # p_exact = 0; the sparse saddle LU leaves ~3e-8 roundoff at Nvel=1089
    @test maximum(abs, ps) < 1.0e-6
end
