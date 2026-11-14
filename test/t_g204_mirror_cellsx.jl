# G204: the x-mirror with cells_x > 1 — an involution that respects the actual cell count.
using QuaST, Test

# G204: the x-mirror used by `symmetry=:x` must use the ACTUAL number of cells
# in x (cells_x * 2^l), not 2^l.  The old formula was only correct for
# cells_x == 1 and silently produced invalid / non-mirror indices for the 2.5x1
# (cells_x=5) forests used by 2b and by the realistic case.

@testset "G204 mirror leaves with cells_x > 1" begin
    for (cx, cy) in ((1, 1), (1, 2), (5, 2), (2, 1))
        Lx = cx / cy
        qt = Quadtree(3; cells_x=cx, cells_y=cy, Lx=Lx)
        for l in 0:3
            ns = QuaST.nside_x(qt, l)
            for ix in unique([0, 1, ns ÷ 2, ns - 1])
                (0 <= ix < ns) || continue
                ml = QuaST._mirror_leaf(qt, (l, ix, 0))
                @test ml[1] == l
                @test ml[2] == ns - 1 - ix
                @test 0 <= ml[2] < ns
                # mirror is an involution
                @test QuaST._mirror_leaf(qt, ml) == (l, ix, 0)
            end
        end
    end

    # a refined symmetric pair stays in the tree and is genuinely mirrored
    qt = Quadtree(3; cells_x=5, cells_y=2, Lx=2.5)
    lf = (0, 0, 0)
    QuaST._refine_pair!(qt, lf, 3, :x)
    kids = [(1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
    ml = QuaST._mirror_leaf(qt, (1, 0, 0))
    @test ml in qt.leaves
    @test ml == (1, QuaST.nside_x(qt, 1) - 1, 0)

    # unit box behaviour is unchanged: cells_x=1 gives 2^l - 1 - ix as before
    qt1 = Quadtree(2; cells_x=1, cells_y=1, Lx=1.0)
    for l in 0:2, ix in 0:(2^l - 1)
        @test QuaST._mirror_leaf(qt1, (l, ix, 0))[2] == 2^l - 1 - ix
    end
end
