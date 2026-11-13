# G33: variable-viscosity quadtree Taylor-Hood Stokes.
#
# Exact polynomial u=(2xy,-y^2), p=0 with eta(x,y)=1+0.1x (or 1+0.1x+0.05y).
# The stress form -div(2 eta eps(u)) = f is evaluated analytically:
#   eta=1+0.1x:          f=(-0.4y, 2)
#   eta=1+0.1x+0.05y:    f=(-0.4y-0.1x, 2+0.1y)
# The polynomial is Q_2 and must be reproduced exactly on a 2:1 mixed tree.
using Test
using LinearAlgebra
using QuaST

@testset "G33 variable-viscosity quadtree Stokes" begin
    qt = Quadtree(6)
    refine!(qt, (0, 0, 0))
    refine!(qt, (1, 0, 0))
    @test balance!(qt)
    ss = make_semstokes(qt, 2)
    nu = ss.su.N
    uex(x, y) = 2x * y
    uez(x, y) = -y^2

    function nodal_error(u)
        e = 0.0
        for g in 1:nu
            x, y = ss.su.coords[g]
            e = max(e, abs(u[g] - uex(x, y)), abs(u[nu + g] - uez(x, y)))
        end
        return e
    end

    eta1(x, y) = 1.0 + 0.1x
    u1, p1, _, _, _ = solve_semquad_stokes_var(ss, qt, eta1,
                                               (x, y) -> -0.4y,
                                               (x, y) -> 2.0,
                                               uex, uez)
    @test nodal_error(u1) < 1.0e-8
    @test maximum(abs, p1) < 1.0e-8

    # constant-eta cross-check against the existing solver
    u0, p0, _, _, _ = solve_semquad_stokes(ss, qt, (x, y) -> 0.0,
                                           (x, y) -> 2.0, uex, uez)
    uv, pv, _, _, _ = solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0,
                                               (x, y) -> 0.0,
                                               (x, y) -> 2.0, uex, uez)
    @test maximum(abs.(uv - u0)) < 1.0e-9
    @test maximum(abs.(pv - p0)) < 1.0e-9

    eta2(x, y) = 1.0 + 0.1x + 0.05y
    u2, p2, _, _, _ = solve_semquad_stokes_var(ss, qt, eta2,
                                               (x, y) -> -0.4y - 0.1x,
                                               (x, y) -> 2.0 + 0.3y,
                                               uex, uez)
    @test nodal_error(u2) < 1.0e-8
    @test maximum(abs, p2) < 1.0e-8
end
