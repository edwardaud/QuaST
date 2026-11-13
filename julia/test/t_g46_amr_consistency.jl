# G46: AMR field consistency.  After one refine cycle the new mesh contains all
# old GLL nodes; theta and both velocity components transferred onto the new
# mesh must reproduce the old nodal values there, and must satisfy the new
# hanging-node constraints.
using Test
using LinearAlgebra
using MantleSEM

function uniform_tree(level::Int)
    qt = Quadtree(level + 3)
    for _ in 1:level
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
    end
    return qt
end

@testset "G46 AMR field consistency" begin
    qt = uniform_tree(3)
    sq = make_semquad(qt, 2)
    theta = [sinpi(sq.coords[i][1]) * sinpi(sq.coords[i][2]) for i in 1:sq.N]
    ux = [1.0 + 0.1 * sq.coords[i][1] + 0.2 * sq.coords[i][2] for i in 1:sq.N]
    uz = [0.3 * sq.coords[i][1] - 0.2 * sq.coords[i][2] for i in 1:sq.N]

    eta = semquad_gradient_jump(sq, qt, theta)
    nr, _ = amr_cycle!(qt, eta; refine_frac=0.3, coarsen_frac=0.0)
    @test nr > 0
    @test max_face_level_jump(qt) <= 1

    sq2 = make_semquad(qt, 2)
    theta2 = transfer_semquad(sq, theta, sq2)
    ux2 = transfer_semquad(sq, ux, sq2)
    uz2 = transfer_semquad(sq, uz, sq2)

    tol = 1.0e-12
    idx = Dict{Tuple{Int,Int},Int}()
    for g in 1:sq2.N
        idx[(round(Int, sq2.coords[g][1] / tol), round(Int, sq2.coords[g][2] / tol))] = g
    end
    worst = 0.0
    for g in 1:sq.N
        g2 = idx[(round(Int, sq.coords[g][1] / tol), round(Int, sq.coords[g][2] / tol))]
        worst = max(worst, abs(theta2[g2] - theta[g]), abs(ux2[g2] - ux[g]),
                    abs(uz2[g2] - uz[g]))
    end
    @test worst < 1.0e-12

    cons = semquad_constraints(sq2, qt)
    @test !isempty(cons)
    worstc = 0.0
    for (c, (ms, ws)) in cons
        worstc = max(worstc, abs(theta2[c] - sum(w * theta2[m] for (m, w) in zip(ms, ws))),
                     abs(ux2[c] - sum(w * ux2[m] for (m, w) in zip(ms, ws))),
                     abs(uz2[c] - sum(w * uz2[m] for (m, w) in zip(ms, ws))))
    end
    @test worstc < 1.0e-10
end
