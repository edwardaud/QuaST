using Test, MantleSEM
@testset "G94 Stokes residual/DWR indicator" begin
    qt=Quadtree(1); ss=make_semstokes(qt,2)
    ru=zeros(2*ss.su.N); rp=zeros(ss.sp.N)
    z=ones(length(ru)); q=ones(length(rp))
    eta=stokes_dwr_indicator(ss.su,qt,ru,rp,z,q)
    @test length(eta)==1
    @test eta[(0,0,0)] == 0.0
    ru2=copy(ru); ru2[1]=1.0
    eta2=stokes_dwr_indicator(ss.su,qt,ru2,rp,z,q)
    @test eta2[(0,0,0)] > 0.0
end
