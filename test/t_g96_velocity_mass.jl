using Test, QuaST, LinearAlgebra
@testset "G96 velocity mass" begin
    qt=Quadtree(1); ss=make_semstokes(qt,2); sq=ss.su
    Mv=assemble_semquad_velocity_mass(sq)
    @test size(Mv,1)==2*sq.N
    @test Mv[1:sq.N,1:sq.N] == semquad_mass(sq)
    @test Mv[sq.N+1:end,sq.N+1:end] == semquad_mass(sq)
end
