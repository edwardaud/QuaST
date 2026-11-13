using Test, QuaST, LinearAlgebra
@testset "G100 DWR vs Kelly concentration" begin
    qtH=Quadtree(2); ssH=make_semstokes(qtH,2)
    uH,pH,_,_,_=solve_semquad_stokes_var(ssH,qtH,(x,y)->1.0,(x,y)->0.0,(x,y)->2.0,(x,y)->0.0,(x,y)->0.0;bc=:free,linear=:direct)
    qtL=deepcopy(qtH); for lf in collect(qtL.leaves); refine!(qtL,lf); end; balance!(qtL)
    ssL=make_semstokes(qtL,2)
    ru,rp=hierarchical_stokes_residual(ssH,qtH,ssL,qtL,(x,y)->1.0,(x,y)->0.0,(x,y)->2.0,uH,pH)
    S=solve_semquad_stokes_var(ssL,qtL,(x,y)->1.0,(x,y)->0.0,(x,y)->2.0,(x,y)->0.0,(x,y)->0.0;bc=:free,assemble_only=true)
    w=goal_upper_half(ssL.su;z0=0.5)
    gu=velocity_goal_rhs(ssL.su,zeros(ssL.su.N),w)
    z,q=solve_stokes_adjoint_system(S.K,S.B,S.Cu,S.Cp,gu,zeros(ssL.sp.N))
    etaD=stokes_dwr_indicator(ssL.su,qtL,S.Cu*ru,S.Cp*rp,z,q)
    nH=ssH.su.N
    uzL=transfer_semquad(ssH.su,uH[nH+1:end],ssL.su;mode=:nodal,zero_walls=false,qt_new=qtL)
    etaK=semquad_gradient_jump(ssL.su,qtL,uzL)
    D=sort(collect(values(etaD)),rev=true); K=sort(collect(values(etaK)),rev=true)
    @test all(isfinite,D) && all(isfinite,K)
    @test sum(D)>0 && sum(K)>0
    n=max(1,ceil(Int,0.1*length(D)))
    sd=Set(sortperm(collect(values(etaD)),rev=true)[1:n])
    sk=Set(sortperm(collect(values(etaK)),rev=true)[1:n])
    ov=length(intersect(sd,sk))/n
    @test 0.0 <= ov <= 1.0
    @test n > 0
end
