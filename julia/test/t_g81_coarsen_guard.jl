# G81: representation-error coarsening guard.
#
# Leaves with a large intra-leaf theta range are protected from coarsening;
# this keeps the plume/active region from being flattened by coarsening.
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

@testset "G81 coarsening guard" begin
    qt = uniform_tree(3)
    lf = first(sort(collect(qt.leaves)))
    eta = Dict{NTuple{3,Int},Float64}(l => 1.0 for l in qt.leaves)
    n0 = length(qt.leaves)
    amr_cycle!(qt, eta; refine_frac=0.0, coarsen_frac=1.0,
               marking=:global, protect=Set([lf]))
    @test lf in qt.leaves
    @test length(qt.leaves) <= n0

    qt2 = uniform_tree(3)
    n2 = length(qt2.leaves)
    eta2 = Dict{NTuple{3,Int},Float64}(l => 1.0 for l in qt2.leaves)
    amr_cycle!(qt2, eta2; refine_frac=0.0, coarsen_frac=1.0,
               marking=:global, protect=Set(collect(qt2.leaves)))
    @test length(qt2.leaves) == n2
end
