using Test, MantleSEM
@testset "G91 DWR indicator kernel" begin
    qt = Quadtree(2)
    sq = make_semquad(qt, 2)
    r = ones(sq.N); z = ones(sq.N)
    eta = dwr_leaf_indicator(sq, qt, r, z)
    @test length(eta) == 1
    @test eta[(0,0,0)] ≈ 9.0
    @test_throws ArgumentError dwr_leaf_indicator(sq, qt, ones(sq.N-1), z)
    w = goal_upper_half(sq; z0=0.5)
    @test length(w) == sq.N
    @test all(0 .<= w .<= 1)
    @test maximum(w) > minimum(w)
end
