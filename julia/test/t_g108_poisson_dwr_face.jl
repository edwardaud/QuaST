using Test, MantleSEM
@testset "G108 Poisson DWR face term" begin
    # u_h and the adjoint live on the fine space; sqc is the coarse space for I_H.
    qtc=Quadtree(2); refine!(qtc,(0,0,0)); balance!(qtc); sqc=make_semquad(qtc,2)
    qtf=deepcopy(qtc); for lf in collect(qtf.leaves); refine!(qtf,lf); end; balance!(qtf); sqf=make_semquad(qtf,3)
    u0=(x,y)->0.0
    uf,_=solve_semquad_poisson(sqf,qtf,(x,y)->2*pi^2*sin(pi*x)*sin(pi*y),u0)
    zf,_=solve_semquad_poisson(sqf,qtf,(x,y)->1.0,u0)
    eta=poisson_dwr_face_terms(sqf,qtf,uf,zf,sqc)
    @test length(eta)==length(qtf.leaves)
    @test all(isfinite,values(eta))
    @test maximum(abs,values(eta)) > 0
    # z - I_H z vanishes on the skeleton of the coarse space: this is why a
    # coarse-face assembly is identically zero.  Check it at coarse-face
    # midpoints, i.e. away from the coarse GLL nodes.
    dw=dual_weights_semquad(uf,sqf,sqc); leavesf=Set(sqf.leaves)
    worst=0.0
    for pf in MantleSEM.face_pairs(qtc)
        L=pf.leafL; h=2.0^(-L[1])
        if pf.dir===:xplus || pf.dir===:xminus
            xm = pf.dir===:xplus ? L[2]*h+h : L[2]*h
            ym = pf.offset+0.5*pf.len
        else
            xm = pf.offset+0.5*pf.len
            ym = pf.dir===:yplus ? L[3]*h+h : L[3]*h
        end
        worst=max(worst,abs(MantleSEM._semquad_eval_at(sqf,dw,xm,ym,leavesf)))
    end
    @test worst < 1e-12
end
