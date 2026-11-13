using Test, MantleSEM, LinearAlgebra

# G138: prescribed heat flux at a wall.
#
# With u = 0, no internal heating and the Boussinesq base Tbar = 1 - y, the
# steady perturbation problem with theta = 0 at the bottom and an outward normal
# derivative q at the top is
#     lap(theta) = 0,  theta(0) = 0,  theta'(1) = q   =>   theta = q y .
# Q2/Q3 represent the linear profile exactly, so the discrete solution must be
# q*y and the Dirichlet wall must carry the same flux (conservation).

@testset "G138 heat-flux wall condition" begin
    for Lx in (1.0, 2.0)
        qt = Quadtree(4; Lx=Lx)
        for _ in 1:2, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        sq = make_semquad(qt, 2)
        pr = Params(Ra=1.0e4, H=0.0)
        q = 0.75
        z = zeros(sq.N)
        # huge dt makes the step the steady solve (K + C) theta = F
        th = solve_semquad_energy(sq, qt, pr, z, z, z, z, 1.0e10, 1.0e10, true;
                                  flux_top=q)
        exact = [q * y for (_, y) in sq.coords]
        @test maximum(abs.(th .- exact)) < 1.0e-6
        # the top wall was left free, so its nodes are not pinned
        @test maximum(abs, th[[g for g in 1:sq.N if sq.coords[g][2] >= 1.0 - 1e-14]]) > 0.5
        # conservation: the bottom (Dirichlet) wall carries the imposed flux
        qb = semquad_cbf_theta_flux(sq, qt, pr, th, z, z; wall=:bottom)
        @test qb ≈ q * total_area(qt) rtol = 1.0e-6
    end
    # both walls flux-conditioned is undetermined and must be rejected
    qt = Quadtree(3)
    sq = make_semquad(qt, 2)
    pr = Params(Ra=1.0e4, H=0.0)
    z = zeros(sq.N)
    @test_throws ArgumentError solve_semquad_energy(sq, qt, pr, z, z, z, z,
                                                    1.0, 1.0, true;
                                                    flux_top=1.0, flux_bot=1.0)
end
