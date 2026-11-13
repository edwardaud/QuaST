using Test, MantleSEM
@testset "G109 Poisson DWR face term on a 2:1 mesh" begin
    # Mixed levels so that faces are split 2:1 across nonconforming interfaces.
    qt=Quadtree(3); refine!(qt,(0,0,0)); refine!(qt,(1,1,1)); balance!(qt); sq=make_semquad(qt,2)
    sH=make_semquad(Quadtree(1),2)          # genuinely coarser: one Q2 element
    @test any(pf->pf.leafL[1]!=pf.leafR[1], MantleSEM.face_pairs(qt))
    # Hanging nodes must satisfy the constraint, otherwise the field is
    # discontinuous and the face fluxes cannot cancel.
    cons=MantleSEM.semquad_constraints(sq,qt)
    function project!(v)
        for (c,(ms,ws)) in cons
            v[c]=sum(ws[k]*v[ms[k]] for k in eachindex(ms))
        end
        return v
    end
    # u_h in Q2 => the normal derivative is continuous, so its jump vanishes
    # pointwise.  The element fluxes are individually non-zero, but their sum
    # must cancel at roundoff; this checks that the coarse-side and fine-side
    # quadratures agree across the 2:1 interfaces.
    upoly=[1.0+2x-3y+x*y+x*x+2y*y for (x,y) in sq.coords]
    zpoly=project!([x^3 + y^3 - x for (x,y) in sq.coords])  # degree p+1: phi != 0
    eta_poly=poisson_dwr_face_terms(sq,qt,upoly,zpoly,sH)
    @test maximum(abs,values(eta_poly)) > 0        # not vacuous
    @test abs(sum(values(eta_poly))) < 1e-12
    # Same mesh and dual weight, but a u_h outside Q2: non-zero total jump.
    usin=project!([sin(pi*x)*sin(pi*y) for (x,y) in sq.coords])
    eta_sin=poisson_dwr_face_terms(sq,qt,usin,zpoly,sH)
    @test abs(sum(values(eta_sin))) > 1e-8
end
