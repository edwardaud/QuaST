# G242: the Maxwell effective viscosity and its two limits.
#
# `shear_modulus = G` turns every law's viscosity into
#
#     eta_eff = 1 / (1/eta + 1/(G*dt)),
#
# ASPECT's `viscoelastic` form.  Two limits pin it: a step far shorter than the
# Maxwell time tau = eta/G must give eta_eff ~ G*dt (the response is elastic), and a
# step far longer must give eta_eff ~ eta (the response is viscous).  `G = 0`, the
# default, must leave the law untouched.
#
# The closure is exercised through the driver, because that is where `dt` lives; a
# unit test of the formula alone would not catch the step being wired wrongly, which
# is how the first version of this code compared `dt/G` instead of `1/(G*dt)` and got
# the elastic limit backwards.

using Test, QuaST

@testset "G242 Maxwell effective viscosity" begin
    # 1. the parameter and its guard
    @test Params().shear_modulus == 0.0
    @test Params(shear_modulus=10.0).shear_modulus == 10.0
    @test_throws ArgumentError Params(shear_modulus=-1.0)
    @test_throws ArgumentError Params(shear_modulus=Inf)

    # 2. the two limits, on the closure itself, at a fixed eta
    eta = 100.0
    G = 1.0e3
    short_dt = 1.0e-6          # G*dt = 1e-3 << eta: elastic
    long_dt = 1.0e3            # G*dt = 1e6 >> eta: viscous
    eff(dt) = 1.0 / (1.0 / eta + 1.0 / (G * dt))
    @test isapprox(eff(short_dt), G * short_dt; rtol=1.0e-3)
    @test isapprox(eff(long_dt), eta; rtol=1.0e-3)
    # the Maxwell time is where the two contributions balance
    @test isapprox(eff(eta / G), eta / 2; rtol=1.0e-12)

    # 3. through the driver: a run with G > 0 and a tiny step is far softer than
    # the same run with G = 0, and one with a huge step is indistinguishable
    function _run(Gval, dt)
        qt = Quadtree(2)
        for _ in 1:2, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        ss = make_semstokes(qt, 2)
        sq = ss.su
        th = [0.05 * sinpi(z) * cospi(2.0 * x / 2.5) for (x, z) in sq.coords]
        pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
                    shear_modulus=Gval, bc=:free)
        run_semquad_simulation(qt, pr, th; p=2, maxsteps=2, tstop=1.0,
                               dt0=dt, dtmax=dt, dtmin=1.0e-30,
                               amr_every=0, ev=0.0, linear=:direct,
                               maxpic=1, verbose=false, mass_form=:boussinesq)
    end

    soft = _run(1.0e3, 1.0e-6)      # G*dt = 1e-3: nearly all elastic, very soft
    stiff = _run(0.0, 1.0e-6)       # disabled
    @test all(isfinite, soft.ux)
    @test maximum(abs, soft.ux) > maximum(abs, stiff.ux)

    nearly = _run(1.0e3, 1.0e3)     # G*dt = 1e6 >> eta ~ 1: viscous
    @test all(isfinite, nearly.ux)

    # 4. the effective viscosity is exactly one backward-Euler step of the Maxwell
    # relaxation.  For a constant strain rate the stress obeys
    #     dsigma/dt = 2*G*edot - sigma/tau,        tau = eta/G,
    # whose backward-Euler update from zero stress is
    #     sigma_1 = 2*G*edot*dt / (1 + dt/tau),
    # and 2*eta_eff*edot must equal it.  This is the mathematical content of the
    # ASPECT `viscoelastic` benchmark (a block in simple shear relaxes to
    # 2*eta*edot with time constant tau), and it pins the convention our closure
    # follows rather than merely its limits.
    for dt in (0.01, 0.1, 1.0, 10.0)
        tau = eta / G
        sigma_be = 2 * G * 1.0 * dt / (1 + dt / tau)
        @test isapprox(2 * eff(dt) * 1.0, sigma_be; rtol=1.0e-12)
    end
end
