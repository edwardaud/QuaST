# G42: geometric wall-graded quadtree for the extreme-case initial mesh.
using Test
using QuaST

@testset "G42 geometric wall-graded tree" begin
    qt = wall_graded_tree(2, 8; strip=0.2)
    st = semquad_level_stats(qt; strip=0.2)
    @test st.wall_max == 8
    @test st.interior_max < 8
    @test st.wall_max > st.interior_max
    @test max_face_level_jump(qt) <= 1
    @test sum(values(st.histogram)) == st.n
    @test st.n == length(qt.leaves)
    @test st.n < 5000

    qt2 = wall_graded_tree(2, 9; strip=0.2)
    st2 = semquad_level_stats(qt2; strip=0.2)
    @test st2.wall_max == 9
    @test st2.n < 2 * st.n + 1000
end
