using Test, QuaST, LinearAlgebra
@testset "G96 velocity mass/goal RHS" begin
    qt=Quadtree(1); ss=make_semstokes(qt,2); sq=ss.su
    Mv=assemble_semquad_velocity_mass(sq)
    @test size(Mv,1)==2*sq.N
    @test Mv[1:sq.N,1:sq.N] == semquad_mass(sq)
    w=ones(sq.N); gu=velocity_goal_rhs(sq,w,w)
    @test length(gu)==2*sq.N
    @test isapprox(dot(gu,ones(2*sq.N)), 2.0; rtol=1e-12)
end
