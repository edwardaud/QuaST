# G52: pointwise divergence diagnostic on the quadtree.
#
# The conservative energy form assumes div u = 0, while Q2-Q1 only enforces the
# divergence weakly.  Before using max|div u| to interpret the H=5/AMR energy
# error, the diagnostic itself must be exact on polynomials: the Q2 field
# u=(2xy,-y^2) is divergence free, and u=(x^2,y^2) has div u = 2x+2y.
using Test
using LinearAlgebra
using QuaST

@testset "G52 quadtree divergence diagnostic" begin
    qt = Quadtree(6)
    refine!(qt, (0, 0, 0))
    refine!(qt, (1, 0, 0))
    @test balance!(qt)
    sq = make_semquad(qt, 2)

    ux = [2.0 * c[1] * c[2] for c in sq.coords]
    uz = [-c[2]^2 for c in sq.coords]
    @test semquad_max_divergence(sq, qt, ux, uz) < 1.0e-12

    ux2 = [c[1]^2 for c in sq.coords]
    uz2 = [c[2]^2 for c in sq.coords]
    exact = maximum(2.0 * c[1] + 2.0 * c[2] for c in sq.coords)
    @test semquad_max_divergence(sq, qt, ux2, uz2) ≈ exact rtol=1.0e-12

    # Q2 streamfunction psi=x(1-x)y(1-y): divergence free and zero normal flow
    # on all four walls.  Then int u_z dTbar/dz must vanish for Tbar=1-y.
    psi_ux(c) = c[1] * (1.0 - c[1]) * (1.0 - 2.0 * c[2])
    psi_uz(c) = -(1.0 - 2.0 * c[1]) * c[2] * (1.0 - c[2])
    ux3 = [psi_ux(c) for c in sq.coords]
    uz3 = [psi_uz(c) for c in sq.coords]
    @test semquad_max_divergence(sq, qt, ux3, uz3) < 1.0e-12
    pr0 = Params(Ra=1.0e4, H=0.0, eta_law=:const, bc=:free)
    @test abs(semquad_base_advection_integral(sq, qt, pr0, ux3, uz3)) < 1.0e-12

    # same divergence-free field: the L2 growth moment must also vanish
    th = [c[1] + c[2] for c in sq.coords]
    @test abs(semquad_theta2_divu_integral(sq, qt, th, ux3, uz3)) < 1.0e-12
end
