# G239: the card's composite prefactors still satisfy their own calibration.
#
# The calibration derives `comp_Ad` and `comp_As` from two
# conditions at the reference strain rate `edot_ref = 1.65e3` measured on the
# Ra = 1e5 constant-viscosity precursor:
#
#   (i)  eta_eff(T=0.5, z=0.5, edot_ref) = 1
#   (ii) eta_eff(T=0.1)/eta_eff(T=0.9)  = 1e4
#
# The card carries the resulting literals.  This gate recomputes both through the
# public closure, so an edit to the literals or a drift in `eta_composite_at` fails
# here instead of being discovered in a production run.  It does not re-run the
# precursor (that is a 120-step solve); it checks the two conditions the calibration
# was solved for, at the strain rate it recorded.

using Test, QuaST

@testset "G239 composite prefactors match their calibration" begin
    # the card's literals (examples/case_realistic_2d.jl)
    card_AD = 1.6240183119e5
    card_AS = 2.6796419238e-1
    edot_ref = 1.65e3

    pr = Params(Ra=1.0e5, H=10.0, eta_law=:composite, comp_rule=:isostress,
                comp_n=3.0, comp_Ed=12.0, comp_Es=17.0,
                comp_Ad=card_AD, comp_As=card_AS,
                comp_eps_min=1.0e-12, eta_T0=0.5,
                eta_min=1.0e-6, eta_max=1.0e8, bc=:free)

    eta_c, clip_c = eta_composite_at(0.5, edot_ref, 0.5, pr)
    @test !clip_c
    @test isapprox(eta_c, 1.0; rtol=1.0e-10)

    eta_lo, clip_lo = eta_composite_at(0.1, edot_ref, 0.5, pr)
    eta_hi, clip_hi = eta_composite_at(0.9, edot_ref, 0.5, pr)
    @test !clip_lo && !clip_hi
    @test isapprox(eta_lo / eta_hi, 1.0e4; rtol=1.0e-9)

    # the dislocation branch is what carries the strain-rate dependence: with
    # comp_As = 0 the closure must be strain-rate independent
    pr_diff = Params(Ra=1.0e5, H=10.0, eta_law=:composite, comp_rule=:isostress,
                     comp_n=3.0, comp_Ed=12.0, comp_Es=17.0,
                     comp_Ad=card_AD, comp_As=0.0,
                     comp_eps_min=1.0e-12, eta_T0=0.5,
                     eta_min=1.0e-6, eta_max=1.0e8, bc=:free)
    a, _ = eta_composite_at(0.5, 1.0e3, 0.5, pr_diff)
    b, _ = eta_composite_at(0.5, 1.0e6, 0.5, pr_diff)
    @test isapprox(a, b; rtol=1.0e-12)
end
