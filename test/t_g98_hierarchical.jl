using Test, QuaST, LinearAlgebra
@testset "G98 hierarchical Stokes residual" begin
    qtH=Quadtree(2); ssH=make_semstokes(qtH,2)
    uH,pH,_,_,_=solve_semquad_stokes_var(ssH,qtH,(x,y)->1.0,(x,y)->0.0,(x,y)->2.0,(x,y)->0.0,(x,y)->0.0;bc=:free,linear=:direct)
    qtL=deepcopy(qtH)
    for lf in collect(qtL.leaves); refine!(qtL,lf); end
    balance!(qtL); ssL=make_semstokes(qtL,2)
    ru,rp=hierarchical_stokes_residual(ssH,qtH,ssL,qtL,(x,y)->1.0,(x,y)->0.0,(x,y)->2.0,uH,pH)
    @test length(ru) > 0
    @test length(rp) > 0
    @test all(isfinite,ru) && all(isfinite,rp)
    @test norm(ru)+norm(rp) > 0
end
