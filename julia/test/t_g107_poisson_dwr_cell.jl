using Test, MantleSEM
@testset "G107 Poisson DWR cell term" begin
    # u_h and z live on the fine space; sqc is the coarse space for I_H.
    qtc=Quadtree(2); refine!(qtc,(0,0,0)); balance!(qtc); sqc=make_semquad(qtc,2)
    qtf=deepcopy(qtc); for lf in collect(qtf.leaves); refine!(qtf,lf); end; balance!(qtf); sqf=make_semquad(qtf,3)
    f=(x,y)->2*pi^2*sin(pi*x)*sin(pi*y)
    u0=(x,y)->0.0
    uf,_=solve_semquad_poisson(sqf,qtf,f,u0)
    zf,_=solve_semquad_poisson(sqf,qtf,(x,y)->1.0,u0)
    eta=poisson_dwr_cell_term(sqf,qtf,uf,zf,f,sqc)
    @test length(eta)==length(qtf.leaves)
    @test all(isfinite,values(eta))
    @test maximum(abs,values(eta)) > 0
    # Quadratic u_h with f = -Delta u_h gives an exactly zero residual, so the
    # cell term must vanish at roundoff.  With the (2/h)^2 factor omitted this
    # is O(h^2) instead, so the check pins the scaling.
    upoly=[x*x+y*y for (x,y) in sqf.coords]
    zsin=[sin(pi*x)*cos(pi*y) for (x,y) in sqf.coords]
    eta0=poisson_dwr_cell_term(sqf,qtf,upoly,zsin,(x,y)->-4.0,sqc)
    @test maximum(abs,values(eta0)) < 1e-10
end
