using Test, MantleSEM
@testset "G106 dual weights" begin
    qtc=Quadtree(2); refine!(qtc,(0,0,0)); balance!(qtc); sqc=make_semquad(qtc,2)
    qtf=deepcopy(qtc); for lf in collect(qtf.leaves); refine!(qtf,lf); end; balance!(qtf); sqf=make_semquad(qtf,3)
    zc=fill(2.5,sqf.N); zl=[x+2y for (x,y) in sqf.coords]; zq=[x^3+0.5y for (x,y) in sqf.coords]
    @test maximum(abs, dual_weights_semquad(zc,sqf,sqc)) < 1e-12
    @test maximum(abs, dual_weights_semquad(zl,sqf,sqc)) < 1e-12
    @test maximum(abs, dual_weights_semquad(zq,sqf,sqc)) > 1e-4
end
