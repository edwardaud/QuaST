# G32: SEM quadtree gradient-jump indicator and one AMR cycle.
#
# Two manufactured problems:
#  - a boundary layer at x=0: the indicator must concentrate leaves at the wall;
#  - a localized Gaussian: AMR must reach the same error as a uniform level-4
#    mesh with fewer degrees of freedom (the acceptance criterion in miniature).
using Test
using LinearAlgebra
using QuaST

function uniform_tree(level::Int)
    qt = Quadtree(level + 3)
    for _ in 1:level
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
    end
    return qt
end

function l2err(sq, u, f)
    M = 20
    e = 0.0
    for k in 0:M, l in 0:M
        x = (k + 0.5) / (M + 1)
        y = (l + 0.5) / (M + 1)
        e += (semquad_eval(sq, u, x, y) - f(x, y))^2
    end
    return sqrt(e / (M + 1)^2)
end

@testset "G32 SEM quadtree AMR indicator" begin
    # constant field: no jumps, no boundary derivative
    qt0 = uniform_tree(2)
    sq0 = make_semquad(qt0, 2)
    eta0 = semquad_gradient_jump(sq0, qt0, ones(Float64, sq0.N))
    @test maximum(abs, values(eta0)) < 1.0e-12

    # boundary layer at x=0
    eps = 0.1
    uex(x, y) = exp(-x / eps) * sinpi(y)
    rhs(x, y) = (pi^2 - 1.0 / eps^2) * uex(x, y)
    qt = uniform_tree(3)
    sq = make_semquad(qt, 2)
    u, _ = solve_semquad_poisson(sq, qt, rhs, uex)
    e0 = l2err(sq, u, uex)
    eta = semquad_gradient_jump(sq, qt, u)
    @test all(isfinite, values(eta))

    frac_left_before = count(lf -> lf[2] * 2.0^(-lf[1]) < 0.25, qt.leaves) / length(qt.leaves)
    nr, _, _ = amr_cycle_balanced!(qt, eta; refine_frac=0.5, coarsen_frac=0.0)
    @test nr > 0
    @test max_face_level_jump(qt) <= 1
    sq2 = make_semquad(qt, 2)
    u2, _ = solve_semquad_poisson(sq2, qt, rhs, uex)
    e1 = l2err(sq2, u2, uex)
    @test e1 < e0
    frac_left_after = count(lf -> lf[2] * 2.0^(-lf[1]) < 0.25, qt.leaves) / length(qt.leaves)
    @test frac_left_after > frac_left_before

    qtu = uniform_tree(4)
    squ = make_semquad(qtu, 2)
    uu, _ = solve_semquad_poisson(squ, qtu, rhs, uex)
    eu = l2err(squ, uu, uex)
    @test sq2.N < squ.N
    @test e1 < e0                    # fewer dofs than uniform L4, less error than L3
    @test eu < e1                    # ... but the uniform L4 mesh is still better

    # localized Gaussian at the centre
    eps2 = 0.05
    ug(x, y) = exp(-((x - 0.5)^2 + (y - 0.5)^2) / eps2^2)
    fg(x, y) = (4.0 / eps2^2) *
               (1.0 - ((x - 0.5)^2 + (y - 0.5)^2) / eps2^2) * ug(x, y)
    qtg = uniform_tree(3)
    sqg = make_semquad(qtg, 2)
    ug0, _ = solve_semquad_poisson(sqg, qtg, fg, ug)
    eg0 = l2err(sqg, ug0, ug)
    etag = semquad_gradient_jump(sqg, qtg, ug0)
    nrg, _, _ = amr_cycle_balanced!(qtg, etag; refine_frac=0.5, coarsen_frac=0.0)
    @test nrg > 0
    sqg2 = make_semquad(qtg, 2)
    ug1, _ = solve_semquad_poisson(sqg2, qtg, fg, ug)
    eg1 = l2err(sqg2, ug1, ug)
    @test eg1 < eg0

    qtgu = uniform_tree(4)
    squg = make_semquad(qtgu, 2)
    ugu, _ = solve_semquad_poisson(squg, qtgu, fg, ug)
    egu = l2err(squg, ugu, ug)
    @test sqg2.N < squg.N
    @test eg1 < eg0                  # fewer dofs than uniform L4, less error than L3
    @test egu < eg1                  # ... but the uniform L4 mesh is still better

end
