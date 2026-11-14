# G116: the strain-rate evaluator — exact on representable polynomials, zero for rigid rotation.
using Test, QuaST

@testset "G116 strain-rate evaluator" begin
    # Exact polynomial test: ux = x^2 y, uz = x y^2 has degree 3 <= p = 4, so a
    # p=4 SEM interpolation is exact and the gradient must match the analytic
    # values at any interior point.
    ux_f(x, y) = x^2 * y
    uz_f(x, y) = x * y^2
    edot_f(x, y) = begin
        exx = 2.0 * x * y
        ezz = 2.0 * x * y
        exz = 0.5 * (x^2 + y^2)
        sqrt(0.5 * (exx^2 + ezz^2 + 2.0 * exz^2))
    end
    function interp_into(sq, f)
        # nodal interpolation of an exactly representable function is exact
        return [f(x, y) for (x, y) in sq.coords]
    end

    # uniform mesh
    qt = Quadtree(4)
    for _ in 1:3, lf in collect(qt.leaves); refine!(qt, lf); end
    balance!(qt)
    sq = make_semquad(qt, 4)
    ux = interp_into(sq, ux_f)
    uz = interp_into(sq, uz_f)
    leaves = Set(qt.leaves)
    for (x, y) in ((0.123, 0.456), (0.777, 0.222), (0.5, 0.5), (0.911, 0.089))
        e = semquad_strain_rate_at(sq, ux, uz, x, y, leaves)
        @test e ≈ edot_f(x, y) atol=1.0e-12
    end

    # 2:1 mesh: refine one quadrant only, then balance
    qt2 = Quadtree(5)
    for _ in 1:2, lf in collect(qt2.leaves); refine!(qt2, lf); end
    refine!(qt2, (2, 1, 1)); refine!(qt2, (3, 3, 3))
    balance!(qt2)
    sq2 = make_semquad(qt2, 4)
    ux2 = interp_into(sq2, ux_f)
    uz2 = interp_into(sq2, uz_f)
    leaves2 = Set(qt2.leaves)
    for (x, y) in ((0.31, 0.29), (0.62, 0.61), (0.44, 0.87), (0.19, 0.13))
        e = semquad_strain_rate_at(sq2, ux2, uz2, x, y, leaves2)
        @test e ≈ edot_f(x, y) atol=1.0e-12
    end

    # rigid-body rotation has zero strain rate
    xr = [x for (x, y) in sq.coords]
    yr = [y for (x, y) in sq.coords]
    e_rot = semquad_strain_rate_at(sq, -yr, xr, 0.37, 0.61, leaves)
    @test abs(e_rot) < 1.0e-13

    # pure shear u = (y, 0): |edot| = 0.5 under the ASPECT convention
    e_ps = semquad_strain_rate_at(sq, yr, zeros(length(yr)), 0.42, 0.58, leaves)
    @test e_ps ≈ 0.5 atol=1.0e-13
end
