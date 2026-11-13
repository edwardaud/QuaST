using Test, QuaST, LinearAlgebra
@testset "G97 assemble-only Stokes" begin
    qt=Quadtree(1); ss=make_semstokes(qt,2)
    S=solve_semquad_stokes_var(ss,qt,(x,y)->1.0,(x,y)->0.0,(x,y)->2.0,(x,y)->0.0,(x,y)->0.0;bc=:free,assemble_only=true)
    @test size(S.K,1)==2*ss.su.N
    @test size(S.B,1)==ss.sp.N
    @test length(S.fr)==size(S.Cu,2)
    @test length(S.gr)==size(S.Cp,2)
    @test size(S.Kred,1)==size(S.Cu,2)
end
