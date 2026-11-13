using Test, MantleSEM, SparseArrays, LinearAlgebra, Random
@testset "G93 Stokes adjoint duality" begin
    qt = Quadtree(1); ss = make_semstokes(qt, 2)
    out = solve_semquad_stokes_var(ss, qt, (x,y)->1.0,
        (x,y)->0.0, (x,y)->2.0, (x,y)->0.0, (x,y)->0.0;
        bc=:free, linear=:direct, return_system=true)
    u,pv,fixed,cons2,cons_p,K,B,Cu,Cp,u0,p0,fv = out
    Kred=Cu'*K*Cu; Bs=-Cp'*B*Cu; Bt=sparse(transpose(Bs))
    nf=size(Cu,2); nfp=size(Cp,2)
    Random.seed!(1)
    fr=randn(nf); gr=randn(nfp); gu=randn(2*ss.su.N); gp=randn(ss.sp.N)
    sol=[Kred Bt; Bs spzeros(nfp,nfp)] \ vcat(fr,gr)
    uf=sol[1:nf]; pf=sol[nf+1:end]
    z,q=solve_stokes_adjoint_system(K,B,Cu,Cp,gu,gp)
    lhs=dot(Cu*uf,gu)+dot(Cp*pf,gp)
    rhs=dot(Cu'*z,fr)+dot(Cp'*q,gr)
    @test isapprox(lhs,rhs; rtol=1e-10, atol=1e-10)
end
