# G43: the geometric wall tree must build at production wall levels without
# the O(N^2) balance-per-level hang.
using Test
using QuaST

@testset "G43 level-11 wall tree build" begin
    t0 = time()
    qt = wall_graded_tree(3, 11; strip=0.15)
    build = time() - t0
    st = semquad_level_stats(qt; strip=0.15)
    @test st.wall_max == 11
    @test max_face_level_jump(qt) <= 1
    @test st.n == length(qt.leaves)
    @test st.n < 30_000
    @test build < 120.0
end
