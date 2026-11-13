using Test, MantleSEM, LinearAlgebra

# G137: consistent boundary flux (CBF) for the temperature equation.
#
# The CBF flux is the residual of the energy operator summed over the wall nodes,
#   sum_{i in wall} (K theta + C theta - F)_i = -int_wall dtheta/dn ds,
# which is the discrete reaction and therefore consistent with the volume
# budget.  The gate checks:
#   1. a conduction state gives Nu == 1 exactly (zero perturbation flux), at
#      Lx = 1 and Lx = 2 -- this pins the normalisation and the sign;
#   2. on a short convective run the CBF and the pointwise wall-gradient fluxes
#      agree to a few percent, which is the expected discretisation difference.

@testset "G137 consistent boundary flux" begin
    # 1. conduction: T = 1 - y, theta = 0, so both fluxes vanish and Nu == 1
    for Lx in (1.0, 2.0)
        qt = Quadtree(4; Lx=Lx)
        for _ in 1:2, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        sq = make_semquad(qt, 2)
        pr = Params(Ra=1.0e4, H=0.0)
        z = zeros(sq.N)
        ntc, nbc = semquad_nu_cbf(sq, qt, pr, z, z, z)
        @test ntc ≈ 1.0 atol = 1.0e-12
        @test nbc ≈ 1.0 atol = 1.0e-12
    end

    # 2. a short convective run, then compare the two flux formulas
    for Lx in (1.0, 2.0)
        qt = Quadtree(4; Lx=Lx)
        for _ in 1:2, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        ss = make_semstokes(qt, 2)
        pr = Params(Ra=1.0e5, H=0.0, eta_law=:expT, deta_T=1.0e-8)
        th0 = [0.05 * cos(pi * x / Lx) * sin(pi * y) for (x, y) in ss.su.coords]
        st = run_semquad_simulation(qt, pr, th0; p=2, tstop=5.0e-2, dt0=1.0e-4,
                                    maxsteps=600, verbose=false)
        ss2 = make_semstokes(st.qt, 2)
        ntg, nbg = semquad_nu_walls(ss2.su, st.qt, st.theta, pr)
        ntc, nbc = semquad_nu_cbf(ss2.su, st.qt, pr, st.theta, st.ux, st.uz)
        @test isfinite(ntc) && isfinite(nbc)
        @test abs(ntc - ntg) / abs(ntg) < 0.05
        @test abs(nbc - nbg) / abs(nbg) < 0.05
    end
end
