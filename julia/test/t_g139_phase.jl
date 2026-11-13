using Test, MantleSEM

# G139: phase-transition buoyancy.
#
# Gamma(y,T) = 0.5 (1 + tanh(((z0-y) - slope (T-Tref))/width)) and the upward
# body force is -Ra B (Gamma(y, Tad+theta) - Gamma(y, Tad)).
# Checks: empty list -> 0 (historical path), zero B -> 0, an analytic value with
# a nonzero Clapeyron slope, and the sign (heating with slope > 0 lowers Gamma,
# so the material gets lighter and the force is upward).

@testset "G139 phase buoyancy" begin
    pr = Params(Ra=1.0e4, H=0.0)
    pt = PhaseTransition(0.7, 0.02, 0.02, 0.5, 0.1, 1.0)   # slope = 0.02, B = 1
    phases = [pt]
    for y in (0.55, 0.65, 0.7, 0.75, 0.85)
        @test semquad_phase_buoyancy(pr, PhaseTransition[], y, 0.0) == 0.0
        # zero anomaly -> zero force
        @test semquad_phase_buoyancy(pr, phases, y, 0.0) == 0.0
        for th in (0.05, 0.1, -0.1)
            f = semquad_phase_buoyancy(pr, phases, y, th)
            dG = phase_gamma(pt, y, semquad_base_T(pr, y) + th) -
                 phase_gamma(pt, y, semquad_base_T(pr, y))
            @test f ≈ -pr.Ra * pt.B * dG rtol = 1.0e-14
            # heating (theta > 0) lowers Gamma for slope > 0 -> lighter -> up
            th > 0 && @test f >= 0.0
            th < 0 && @test f <= 0.0
        end
    end
    # B = 0 is a no-op
    pt0 = PhaseTransition(0.7, 0.02, 0.02, 0.5, 0.1, 0.0)
    @test semquad_phase_buoyancy(pr, [pt0], 0.7, 0.5) == 0.0
    # a sharp (width -> 0) transition is a step and stays finite
    ptsharp = PhaseTransition(0.7, 0.0, 0.0, 0.5, 0.1, 1.0)
    @test semquad_phase_buoyancy(pr, [ptsharp], 0.6, 0.0) == 0.0
end
