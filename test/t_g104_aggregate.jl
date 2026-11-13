using Test, QuaST
@testset "G104 aggregate indicator" begin
    qt=Quadtree(2); refine!(qt,(0,0,0)); balance!(qt)
    for lf in collect(qt.leaves); refine!(qt,lf); end; balance!(qt); sq=make_semquad(qt,2)
    eta=Dict{NTuple{3,Int},Float64}()
    for lf in qt.leaves; eta[lf]=1.0; end
    a=aggregate_indicator_to_level(eta,0)
    @test length(a)==1
    @test a[(0,0,0)]≈Float64(length(qt.leaves))
    b=aggregate_indicator_to_level(eta,1)
    @test all(v≈4.0 for v in values(b))
end
