# G165: the Tosi linear-law tangent, including the eta0 scaling and clipping.
using Test, QuaST, LinearAlgebra, Printf

# G165: Tosi tangent for the *linear* law (benchmark Case 1/3) and the eta0
# scaling of the analytic derivatives.
#
# Two defects are pinned here:
#   1. `semquad_deta_tosi` used to return `(0, 0)` whenever
#      `eta_star == sigma_y == 0`.  The strain-rate derivative really is zero
#      there, but the *temperature* derivative is not: `eta_lin = exp(-ln(etaT)
#      T)` gives `deta/dT = -ln(etaT) eta_lin`.  With the zero it returned, the
#      Tosi Newton for Case 1 had no temperature coupling in its Jacobian and
#      degraded to a fixed-point iteration (~0.6 per step instead of quadratic).
#   2. The derivatives were in absolute units while `semquad_eta_tosi(...)[1]`
#      is nondimensional, so they were wrong by a factor `eta0` whenever
#      `eta0 != 1` (G156's law has eta0 = 1, so it could not see this).
#
# The former section (4) checked the same tangent through the steady-map JVP.
# That map was removed with the steady-state solver; the direct derivative
# checks above are what pin the defect it was written for.

@testset "G165 Tosi linear-law tangent" begin
    # ---- (1) linear law, unclipped, eta0 != 1 -----------------------------
    law = TosiLaw(etaT=1.0e5, etaZ=1.0, eta_star=0.0, sigma_y=0.0,
                  eta_init=0.1, eta0=4.0, eta_min=1.0e-9, eta_max=1.0e9,
                  z_is_depth=false)
    he = 1.0e-7
    for (T, y) in ((0.2, 0.5), (0.5, 0.3), (0.8, 0.7))
        fd_T = (semquad_eta_tosi(T + he, 0.5, y, law)[1] -
                semquad_eta_tosi(T - he, 0.5, y, law)[1]) / (2he)
        de, dT = semquad_deta_tosi(T, 0.5, y, law)
        @test de == 0.0
        @test abs(dT - fd_T) <= 1.0e-6 * max(abs(fd_T), 1.0e-12)
        @test dT < 0.0        # viscosity decreases with temperature
    end

    # ---- (2) linear law clipped at both bounds -> zero derivative ---------
    lawc = TosiLaw(etaT=1.0e5, etaZ=1.0, eta_star=0.0, sigma_y=0.0,
                   eta_init=0.1, eta0=1.0, eta_min=0.5, eta_max=2.0,
                   z_is_depth=false)
    # eta_lin(0.5) = 1e-2.5 ~ 3.2e-3 < 0.5 -> clipped low
    @test semquad_deta_tosi(0.5, 0.5, 0.5, lawc) == (0.0, 0.0)
    # eta_lin(0) = 1 in [0.5, 2]; eta_lin(-0.1)? use T=-1 -> e^11.5 clipped high
    @test semquad_deta_tosi(-1.0, 0.5, 0.5, lawc) == (0.0, 0.0)
    # inside the window (T = -0.05 -> eta_lin ~ 1.8)
    fd_T = (semquad_eta_tosi(-0.05 + he, 0.5, 0.5, lawc)[1] -
            semquad_eta_tosi(-0.05 - he, 0.5, 0.5, lawc)[1]) / (2he)
    @test abs(semquad_deta_tosi(-0.05, 0.5, 0.5, lawc)[2] - fd_T) <=
          1.0e-6 * abs(fd_T)

    # ---- (3) plastic law: eta0 scaling of both derivatives ----------------
    lawp = TosiLaw(etaT=1.0e5, etaZ=10.0, eta_star=1.0e-3, sigma_y=4.0,
                   eta_init=0.1, eta0=3.0, eta_min=1.0e-12, eta_max=1.0e12,
                   z_is_depth=true)
    for (T, e, y) in ((0.2, 0.1, 0.5), (0.5, 0.5, 0.3), (0.8, 1.0, 0.7))
        hep = 1.0e-6
        fd_e = (semquad_eta_tosi(T, e + hep, y, lawp)[1] -
                semquad_eta_tosi(T, e - hep, y, lawp)[1]) / (2hep)
        fd_T2 = (semquad_eta_tosi(T + hep, e, y, lawp)[1] -
                 semquad_eta_tosi(T - hep, e, y, lawp)[1]) / (2hep)
        de, dT = semquad_deta_tosi(T, e, y, lawp)
        @test abs(de - fd_e) <= 1.0e-5 * max(abs(fd_e), 1.0e-8)
        @test abs(dT - fd_T2) <= 1.0e-5 * max(abs(fd_T2), 1.0e-8)
    end

end
