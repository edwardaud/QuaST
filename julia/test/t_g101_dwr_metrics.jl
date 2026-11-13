using Test, MantleSEM, LinearAlgebra
@testset "G101 DWR metrics" begin
    qtH=Quadtree(2); ssH=make_semstokes(qtH,2)
    uH,pH,_,_,_=solve_semquad_stokes_var(ssH,qtH,(x,y)->1.0,(x,y)->0.0,(x,y)->2.0,(x,y)->0.0,(x,y)->0.0;bc=:free,linear=:direct)
    qtL=deepcopy(qtH); for lf in collect(qtL.leaves); refine!(qtL,lf); end; balance!(qtL)
    ssL=make_semstokes(qtL,2)
    ru,rp=hierarchical_stokes_residual(ssH,qtH,ssL,qtL,(x,y)->1.0,(x,y)->0.0,(x,y)->2.0,uH,pH)
    S=solve_semquad_stokes_var(ssL,qtL,(x,y)->1.0,(x,y)->0.0,(x,y)->2.0,(x,y)->0.0,(x,y)->0.0;bc=:free,assemble_only=true)
    w=goal_upper_half(ssL.su;z0=0.5); gu=velocity_goal_rhs(ssL.su,zeros(ssL.su.N),w)
    z,q=solve_stokes_adjoint_system(S.K,S.B,S.Cu,S.Cp,gu,zeros(ssL.sp.N))
    etaD=stokes_dwr_indicator(ssL.su,qtL,S.Cu*ru,S.Cp*rp,z,q)
    nH=ssH.su.N
    uxL=transfer_semquad(ssH.su,uH[1:nH],ssL.su;mode=:nodal,zero_walls=false,qt_new=qtL)
    uzL=transfer_semquad(ssH.su,uH[nH+1:end],ssL.su;mode=:nodal,zero_walls=false,qt_new=qtL)
    pL=transfer_semquad(ssH.sp,pH,ssL.sp;mode=:nodal,zero_walls=false,qt_new=qtL)
    uL,pL2,_,_,_=solve_semquad_stokes_var(ssL,qtL,(x,y)->1.0,(x,y)->0.0,(x,y)->2.0,(x,y)->0.0,(x,y)->0.0;bc=:free,linear=:direct)
    Jc=dot(gu,vcat(uxL,uzL)); Jf=dot(gu,uL); est=-dot(ru,S.Cu'*z)-dot(rp,S.Cp'*q)
    etaK=semquad_gradient_jump(ssL.su,qtL,uzL)
    n=max(1,ceil(Int,0.1*length(etaD)))
    sd=sortperm(collect(values(etaD)),rev=true)[1:n]; sk=sortperm(collect(values(etaK)),rev=true)[1:n]
    lvs=sort(collect(qtL.leaves)); hot=Set(lf for lf in lvs if (lf[3]+0.5)*2.0^(-lf[1])>0.5)
    fd=count(i->lvs[i] in hot,sd)/n; fk=count(i->lvs[i] in hot,sk)/n
    println("effectivity=",est/(Jf-Jc)," hotD=",fd," hotK=",fk)
    @test isfinite(est); @test abs(Jf-Jc)>0; @test 0<=fd<=1 && 0<=fk<=1
end
