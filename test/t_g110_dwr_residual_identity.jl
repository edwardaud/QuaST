using Test, QuaST, LinearAlgebra
@testset "G110 DWR residual identity (conforming mesh)" begin
    # For fields vanishing on the boundary (so the Nitsche terms drop out) and
    # any phi in the same space:
    #   (f,phi) - a(u_h,phi) = sum_K int_K (f+Delta u_h) phi
    #                          - sum_F int_F [dn u_h] phi
    # This pins sign, (2/h)^2 scaling and quadrature against the assembled
    # operator itself.
    #
    # Open issue, not asserted: on a nonconforming 2:1 mesh the same identity
    # leaves a residual of about 7e-10 (relative ~3e-2).  It is not understood
    # yet, so no claim is made for nonconforming meshes.
    qH=Quadtree(3); refine!(qH,(0,0,0)); balance!(qH)        # 4 coarse leaves
    qt=deepcopy(qH); for lf in collect(qt.leaves); refine!(qt,lf); end; balance!(qt)
    sq=make_semquad(qt,3); sH=make_semquad(qH,3)
    u0=(x,y)->0.0
    f=(x,y)->2*pi^2*sin(pi*x)*sin(pi*y)
    g=(x,y)->exp(-((x-0.45)^2+(y-0.55)^2)/0.05)
    uh,_=solve_semquad_poisson(sq,qt,f,u0)
    zh,_=solve_semquad_poisson(sq,qt,g,u0)
    bnodes=[k for (k,(x,y)) in enumerate(sq.coords)
            if x<1e-12 || x>1-1e-12 || y<1e-12 || y>1-1e-12]
    @test !isempty(bnodes)
    uh[bnodes].=0.0; zh[bnodes].=0.0
    phi=dual_weights_semquad(zh,sq,sH)
    @test maximum(abs,phi[bnodes]) < 1e-14
    A,b=assemble_semquad_poisson(sq,qt,f,u0)
    lhs=dot(b,phi)-dot(uh,A*phi)
    cell=sum(values(poisson_dwr_cell_term(sq,qt,uh,zh,f,sH)))
    face=sum(values(poisson_dwr_face_terms(sq,qt,uh,zh,sH)))
    rhs=cell-face
    @test isfinite(lhs) && isfinite(rhs)
    @test abs(rhs)>1e-9
    @test abs(lhs-rhs) < 1e-12*max(1.0,abs(rhs))
end
