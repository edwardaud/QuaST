using Test, QuaST, SparseArrays, LinearAlgebra
@testset "G95 reduced Stokes residual" begin
    qt=Quadtree(1); ss=make_semstokes(qt,2)
    out=solve_semquad_stokes_var(ss,qt,(x,y)->1.0,(x,y)->0.0,(x,y)->2.0,(x,y)->0.0,(x,y)->0.0;bc=:free,linear=:direct,return_system=true)
    u,pv,fixed,cons2,cons_p,K,B,Cu,Cp,u0,p0,fv=out
    ru,rp=stokes_reduced_residual(K,B,Cu,Cp,u0,p0,fv,u,pv)
    @test norm(ru) < 1e-8
    @test norm(rp) < 1e-8
end
