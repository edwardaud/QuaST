using Test, QuaST, LinearAlgebra
@testset "G99 hierarchical DWR indicator" begin
    qtH=Quadtree(2); ssH=make_semstokes(qtH,2)
    uH,pH,_,_,_=solve_semquad_stokes_var(ssH,qtH,(x,y)->1.0,(x,y)->0.0,(x,y)->2.0,(x,y)->0.0,(x,y)->0.0;bc=:free,linear=:direct)
    qtL=deepcopy(qtH); for lf in collect(qtL.leaves); refine!(qtL,lf); end; balance!(qtL)
    ssL=make_semstokes(qtL,2)
    ru,rp=hierarchical_stokes_residual(ssH,qtH,ssL,qtL,(x,y)->1.0,(x,y)->0.0,(x,y)->2.0,uH,pH)
    S=solve_semquad_stokes_var(ssL,qtL,(x,y)->1.0,(x,y)->0.0,(x,y)->2.0,(x,y)->0.0,(x,y)->0.0;bc=:free,assemble_only=true)
    w=goal_upper_half(ssL.su;z0=0.5)
    gu=velocity_goal_rhs(ssL.su,zeros(ssL.su.N),w)
    z,q=solve_stokes_adjoint_system(S.K,S.B,S.Cu,S.Cp,gu,zeros(ssL.sp.N))
    eta=stokes_dwr_indicator(ssL.su,qtL,S.Cu*ru,S.Cp*rp,z,q)
    @test length(eta)==length(qtL.leaves)
    @test all(v->v>=0,values(eta))
    @test sum(values(eta))>0
    @test all(isfinite,values(eta))
end
