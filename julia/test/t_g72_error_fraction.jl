# G72: ASPECT-style error-fraction (Dörfler) marking, coarsening, hard budget.
#
# ASPECT's refine_and_coarsen_fixed_fraction refines the largest-error cells
# until their cumulative error reaches Refinement fraction of the total, and
# coarsens the smallest-error cells until Coarsening fraction of the total.
# This is not a fixed fraction of the cell count.
using Test
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

function eta_dict(qt, high::NTuple{3,Int}, highval::Float64, lowval::Float64)
    eta = Dict{NTuple{3,Int},Float64}()
    for lf in qt.leaves
        eta[lf] = lf == high ? highval : lowval
    end
    return eta
end

@testset "G72 error-fraction marking" begin
    qt = uniform_tree(2)
    n0 = length(qt.leaves)
    high = first(sort(collect(qt.leaves)))
    eta = eta_dict(qt, high, 10.0, 1.0)
    nr, nc = amr_cycle!(qt, eta; refine_frac=0.5, coarsen_frac=0.0,
                        marking=:error_fraction)
    @test nr > 0
    # the single dominant-error leaf must be refined, not just 50% of cells
    @test !(high in qt.leaves)
    @test all((high[1] + 1, 2 * high[2] + dx, 2 * high[3] + dy) in qt.leaves
              for dx in 0:1, dy in 0:1)
    @test max_face_level_jump(qt) <= 1

    # max_level prevents refining the dominant leaf
    qt2 = uniform_tree(2)
    eta2 = eta_dict(qt2, first(sort(collect(qt2.leaves))), 10.0, 1.0)
    n2 = length(qt2.leaves)
    amr_cycle!(qt2, eta2; refine_frac=1.0, coarsen_frac=0.0,
               marking=:error_fraction, max_level=2)
    @test length(qt2.leaves) == n2

    # coarsening removes the smallest-error leaves
    qt3 = uniform_tree(3)
    n3 = length(qt3.leaves)
    eta3 = Dict{NTuple{3,Int},Float64}(lf => 1.0 for lf in qt3.leaves)
    nr3, nc3 = amr_cycle!(qt3, eta3; refine_frac=0.0, coarsen_frac=0.5,
                          marking=:error_fraction)
    @test nc3 > 0
    @test length(qt3.leaves) < n3
    @test max_face_level_jump(qt3) <= 1

    # hard leaf budget after 2:1 balancing
    qt4 = uniform_tree(2)
    n4 = length(qt4.leaves)
    eta4 = Dict{NTuple{3,Int},Float64}(lf => 1.0 for lf in qt4.leaves)
    amr_cycle!(qt4, eta4; refine_frac=1.0, coarsen_frac=0.0,
               marking=:error_fraction, max_leaves=n4)
    @test length(qt4.leaves) <= n4
    @test max_face_level_jump(qt4) <= 1
end
