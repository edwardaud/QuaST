# G119: the Tosi viscoplastic law — equations (6)-(8) against an independent implementation.
using Test, QuaST

@testset "G119 Tosi viscoplastic law" begin
    # Case 5a parameters from Tosi et al. (2015) Table 1 and the ASPECT prm files:
    # Ra = 1e2, Delta eta_T = 1e5, Delta eta_z = 10, Gamma = 1e-3, sigma_Y = 4.
    t = TosiLaw(etaT=1.0e5, etaZ=10.0, eta_star=1.0e-3, sigma_y=4.0,
                eta_init=1.0e-1, eta0=1.0e-1, eta_min=1.0e-6, eta_max=10.0,
                z_is_depth=true)

    # Independent re-implementation of the paper's equations (7), (8), (6):
    #   eta_lin = exp(-ln(etaT) T + ln(etaZ) z),  eta_pl = eta* + sigma_Y/sqrt(edot:edot)
    #   eta = 2 / (1/eta_lin + 1/eta_pl),  clamp,  eta_nd = eta/eta0
    # with sqrt(edot:edot) = sqrt(2) eps_II and z = 1 - y (depth).
    function ref_eta(T, eps, y; z_is_depth=true)
        z = z_is_depth ? 1.0 - y : y
        eta_lin = exp(-log(1.0e5) * T + log(10.0) * z)
        eta_pl = 1.0e-3 + 4.0 / (sqrt(2.0) * eps)
        return clamp(2.0 / (1.0 / eta_lin + 1.0 / eta_pl), 1.0e-6, 10.0)
    end

    # (1) zero strain rate -> eta_init, exactly
    e_nd, e_abs = semquad_eta_tosi(0.5, 0.0, 0.5, t)
    @test e_abs == 1.0e-1
    @test e_nd == 1.0                       # explicit law above uses eta0 = 1e-1

    # (2) full formula on a grid, including the factor 2 of equation (6)
    for T in (0.0, 0.25, 0.5, 0.75, 1.0), y in (0.0, 0.3, 0.6, 1.0),
        eps in (1.0e-4, 1.0e-2, 1.0, 1.0e2)
        @test semquad_eta_tosi(T, eps, y, t)[2] ≈ ref_eta(T, eps, y) rtol=1e-14
    end

    # (3) nondimensional value is exactly the absolute one over eta0
    for (T, eps, y) in ((0.2, 1.0e-3, 0.7), (0.9, 1.0, 0.1), (0.0, 0.0, 0.5))
        e_nd, e_abs = semquad_eta_tosi(T, eps, y, t)
        @test e_nd == e_abs / t.eta0
    end

    # (4) plastic limit: as eps -> infinity eta -> 2 eta_lin/(1 + eta_lin/eta*)
    T, y = 0.3, 0.6
    z = 1.0 - y
    eta_lin = exp(-log(1.0e5) * T + log(10.0) * z)
    limit = 2.0 / (1.0 / eta_lin + 1.0 / 1.0e-3)
    @test semquad_eta_tosi(T, 1.0e9, y, t)[2] ≈ clamp(limit, 1.0e-6, 10.0) rtol=1e-5

    # (5) the depth switch changes the etaZ factor
    zt = TosiLaw(etaZ=10.0, z_is_depth=true)
    zf = TosiLaw(etaZ=10.0, z_is_depth=false)
    @test semquad_eta_tosi(0.0, 1.0, 0.25, zt)[2] ≈ ref_eta(0.0, 1.0, 0.25; z_is_depth=true)
    @test semquad_eta_tosi(0.0, 1.0, 0.25, zf)[2] ≈ ref_eta(0.0, 1.0, 0.25; z_is_depth=false)
    @test semquad_eta_tosi(0.0, 1.0, 0.25, zt)[2] > semquad_eta_tosi(0.0, 1.0, 0.25, zf)[2]

    # (6) clamps in absolute units, both sides
    tight = TosiLaw(eta_min=1.0e-2, eta_max=1.0, eta0=1.0)
    @test semquad_eta_tosi(1.0, 1.0e-3, 1.0, tight)[2] == 1.0e-2   # very small eta
    @test semquad_eta_tosi(0.0, 1.0e-3, 0.0, tight)[2] == 1.0      # very large eta
    @test semquad_eta_tosi(1.0, 1.0e-3, 1.0, tight)[1] == 1.0e-2   # nd mirrors it

    # (7) the default normalization is eta0 = 1 (settled empirically; see tosi_law.jl)
    @test TosiLaw().eta0 == 1.0
    # (8) guards
    @test_throws ArgumentError TosiLaw(etaT=0.5)
    @test_throws ArgumentError TosiLaw(etaZ=0.0)
    @test_throws ArgumentError TosiLaw(eta0=0.0)
    @test_throws ArgumentError TosiLaw(eta_min=0.0)
    @test_throws ArgumentError TosiLaw(eta_min=1.0, eta_max=0.5)
    @test_throws ArgumentError TosiLaw(eta_star=-1.0, eta_max=1.0)
end
