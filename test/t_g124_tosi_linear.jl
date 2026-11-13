using Test, QuaST

@testset "G124 Tosi linear branch (Case 1/3 control)" begin
    # Case 1: eta = exp(-ln(etaT) T), no depth term (etaZ = 1), no plasticity
    # (sigma_y = 0, eta_star = 0).  The viscosity must then be independent of the
    # strain rate -- including at eps = 0, where the eta_init branch of the
    # viscoplastic law must NOT apply.
    law = TosiLaw(etaT=1.0e5, etaZ=1.0, eta_star=0.0, sigma_y=0.0,
                  eta_init=1.0e-1, eta0=1.0, eta_min=1.0e-6, eta_max=10.0,
                  z_is_depth=true)

    for T in (0.0, 0.25, 0.5, 0.75, 1.0), y in (0.0, 0.5, 1.0)
        want = clamp(exp(-log(1.0e5) * T), 1.0e-6, 10.0)
        for eps in (0.0, 1.0e-8, 1.0e-2, 1.0, 1.0e3)
            @test semquad_eta_tosi(T, eps, y, law)[2] ≈ want rtol=1e-14
        end
    end

    # etaZ = 1 removes the depth term entirely
    @test semquad_eta_tosi(0.4, 1.0, 0.0, law)[2] ≈
          semquad_eta_tosi(0.4, 1.0, 1.0, law)[2] rtol=1e-14

    # and the viscosity range is exactly [1e-5, 1] as the paper states for Case 1
    @test semquad_eta_tosi(1.0, 1.0, 0.5, law)[2] ≈ 1.0e-5 rtol=1e-12
    @test semquad_eta_tosi(0.0, 1.0, 0.5, law)[2] ≈ 1.0 rtol=1e-12

    # the viscoplastic law (sigma_y > 0) still uses eta_init at eps = 0
    vp = TosiLaw(etaT=1.0e5, etaZ=10.0, eta_star=1.0e-3, sigma_y=4.0,
                 eta_init=1.0e-1, eta0=1.0)
    @test semquad_eta_tosi(0.5, 0.0, 0.5, vp)[2] == 1.0e-1
    @test semquad_eta_tosi(0.5, 1.0e-6, 0.5, vp)[2] != 1.0e-1
end
