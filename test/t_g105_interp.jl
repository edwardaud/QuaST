using Test, QuaST
@testset "G105 SEM interpolation" begin
    qt=Quadtree(2); refine!(qt,(0,0,0)); balance!(qt); src=make_semquad(qt,3)
    dst=make_semquad(qt,2)
    zc=fill(2.5,src.N); z1=[x+2y for (x,y) in src.coords]
    @test maximum(abs, interpolate_semquad(src,zc,dst) .- 2.5) < 1e-12
    @test maximum(abs, interpolate_semquad(src,z1,dst) .- [x+2y for (x,y) in dst.coords]) < 1e-12
end
