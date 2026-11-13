# G73: wall minimum-level enforcement (ASPECT boundary refinement analogue).
#
# Error-fraction coarsening can remove the thermal boundary layer.  After every
# AMR cycle the wall leaves are refined back to wall_min_level within
# wall_strip; this is the numerical counterpart of ASPECT's boundary
# refinement / minimum refinement criterion.
using Test
using QuaST

function uniform_tree(level::Int)
    qt = Quadtree(level + 2)
    for _ in 1:level
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
    end
    return qt
end

@testset "G73 wall minimum level" begin
    qt = uniform_tree(2)
    n0 = length(qt.leaves)
    nr = refine_wall_to_min!(qt, 4; strip=0.1)
    @test nr > 0
    @test length(qt.leaves) > n0
    bad = 0
    for lf in qt.leaves
        h = 2.0^(-lf[1])
        zc = (lf[3] + 0.5) * h
        if (lf[3] == 0 || lf[3] == 2^lf[1] - 1 ||
            zc <= 0.1 || zc >= 0.9) && lf[1] < 4
            bad += 1
        end
    end
    @test bad == 0
    @test max_face_level_jump(qt) <= 1

    qt2 = uniform_tree(2)
    n2 = length(qt2.leaves)
    @test refine_wall_to_min!(qt2, 0) == 0
    @test length(qt2.leaves) == n2

    @test_throws ArgumentError refine_wall_to_min!(uniform_tree(2), 3; strip=0.6)
end
