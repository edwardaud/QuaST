using Test, MantleSEM, LinearAlgebra
@testset "G111 Poisson DWR effectivity with enriched adjoint" begin
    # The dual weight must come from an *enriched* adjoint:
    #     phi = z_enr - I_h z_enr,   I_h : primal space
    # Then sum(eta) = R(u_h)(phi) = J(u_h) - J(u), i.e. minus the goal error,
    # so the effectivity must be -1.  The un-enriched construction (adjoint on
    # the primal space, minus an interpolation onto a coarser space) gives
    # 1e-3 level effectivities and is not a DWR estimator.
    function goal_gauss(sq, qt, u, g; nq::Int=8)
        xq, wq = gauss_legendre(nq); s = 0.0; leaves = Set(sq.leaves)
        for lf in qt.leaves
            h = 2.0^(-lf[1]); x0 = lf[2]*h; y0 = lf[3]*h
            for j in 1:nq, i in 1:nq
                x = x0 + 0.5*(xq[i]+1)*h; y = y0 + 0.5*(xq[j]+1)*h
                s += 0.25*h*h*wq[i]*wq[j]*g(x,y)*
                     MantleSEM._semquad_eval_at(sq, u, x, y, leaves)
            end
        end
        return s
    end
    function uniform_tree(L::Int)
        qt = Quadtree(L)
        for _ in 1:L, lf in collect(qt.leaves); refine!(qt, lf); end
        balance!(qt)
        return qt
    end
    u0 = (x,y) -> 0.0
    uex = (x,y) -> sin(pi*x)*sin(pi*y)
    f = (x,y) -> 2*pi^2*uex(x,y)
    g = (x,y) -> exp(-((x-0.5)^2 + (y-0.6)^2)/0.02)
    xg, wg = gauss_legendre(200); Jex = 0.0
    for j in 1:200, i in 1:200
        x = 0.5*(xg[i]+1); y = 0.5*(xg[j]+1)
        Jex += 0.25*wg[i]*wg[j]*g(x,y)*uex(x,y)
    end
    for (p, L) in ((2,3), (2,4), (3,3))
        qt = uniform_tree(L); sq = make_semquad(qt, p)
        uh,_ = solve_semquad_poisson(sq, qt, f, u0)
        err = goal_gauss(sq, qt, uh, g) - Jex
        # enriched adjoint: order p+1 on the same mesh
        sqE = make_semquad(qt, p+1)
        zhE,_ = solve_semquad_poisson(sqE, qt, g, u0)
        eta = poisson_dwr_estimate(sq, qt, uh, zhE, sqE, qt, f)
        est = sum(values(eta))
        @test length(eta) == length(qt.leaves)
        @test all(isfinite, values(eta))
        @test abs(est) > 0
        @test abs(est/err + 1) < 2.0e-2
    end
end
