# G41: leaf-level distribution diagnostic for AMR boundary-layer tracking.
using Test
using QuaST

@testset "G41 AMR level statistics" begin
    qt = Quadtree(8)
    for _ in 1:3
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
    end
    for _ in 1:3
        for lf in collect(qt.leaves)
            l, ix, iy = lf
            h = 2.0^(-l)
            y0 = iy * h
            if l < 6 && (y0 + h <= 0.2 || y0 >= 0.8)
                refine!(qt, lf)
            end
        end
        balance!(qt)
    end
    st = semquad_level_stats(qt; strip=0.2)
    @test st.wall_max == 6
    @test st.interior_max >= 3
    @test st.wall_max > st.interior_max
    @test sum(values(st.histogram)) == st.n
    @test st.n == length(qt.leaves)

    qtu = Quadtree(6)
    for _ in 1:4
        for lf in collect(qtu.leaves)
            refine!(qtu, lf)
        end
    end
    stu = semquad_level_stats(qtu; strip=0.1)
    @test stu.wall_max == 4
    @test stu.interior_max == 4
end
