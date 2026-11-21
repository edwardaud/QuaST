# G244: the Maxwell stress update against its analytic curve.
#
# `maxwell_stress_step` is one backward-Euler step of
#
#     dtau/dt = 2 G edot - tau / tau_M,      tau_M = eta / G,
#
# and the continuous solution from rest is `2 eta edot (1 - exp(-t/tau_M))`.  The gate
# pins four things: the steady state, the elastic first step, first-order convergence
# of the iteration to the analytic curve as `dt` shrinks, and the `G = 0` escape.
#
# This is the transient that the ASPECT viscoelastic benchmark measures and the one
# 2a's effective-viscosity form cannot produce, because it carries no history.

using Test, QuaST

@testset "G244 Maxwell stress update" begin
    eta = 100.0
    G = 1.0e3
    tau_M = eta / G
    edot = 1.0e-3

    # the steady state of the iteration is the viscous stress
    tau = 0.0
    for _ in 1:20_000
        tau = maxwell_stress_step(tau, edot, tau_M / 100, eta, G)
    end
    @test isapprox(tau, 2 * eta * edot; rtol=1.0e-3)

    # the first step from rest with dt << tau_M is the elastic one
    dt_small = 1.0e-4 * tau_M
    @test isapprox(maxwell_stress_step(0.0, edot, dt_small, eta, G),
                   2 * G * edot * dt_small; rtol=1.0e-3)

    # the discrete path converges to the analytic curve at first order
    function err(dt)
        n = round(Int, 5 * tau_M / dt)
        tau = 0.0
        for _ in 1:n
            tau = maxwell_stress_step(tau, edot, dt, eta, G)
        end
        return abs(tau - maxwell_stress_exact(n * dt, edot, eta, G))
    end
    e1 = err(tau_M / 50)
    e2 = err(tau_M / 100)
    @test e2 < e1
    @test 1.5 < e1 / e2 < 2.5          # first order

    # G = 0 is off, not "elastic with zero modulus"
    @test maxwell_stress_step(7.0, edot, 1.0, eta, 0.0) == 7.0
    @test maxwell_stress_exact(1.0, edot, eta, 0.0) == 0.0

    # and the two agree in the limit dt -> 0 at a fixed time
    t = tau_M                      # 20 000 steps of tau_M/20 000
    tau_small = 0.0
    dt = tau_M / 20_000
    for _ in 1:20_000
        tau_small = maxwell_stress_step(tau_small, edot, dt, eta, G)
    end
    @test isapprox(tau_small, maxwell_stress_exact(t, edot, eta, G); rtol=1.0e-4)
end
