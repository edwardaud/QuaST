# Maxwell stress history, step 1: the update kernel.
#
# This file holds the physics of the stress update and nothing else.  Wiring the
# three components into the state vector, the AMR transfer, the checkpoint and the
# `div(tau)` force term is the next step (docs/dev/PHASE2.md); keeping the kernel
# separate means it can be gated against the analytic curve before any of that
# machinery exists, and a failure there cannot be confused with a transport bug.

"""
    maxwell_stress_step(tau_old, edot, dt, eta, G)

One backward-Euler step of the Maxwell relaxation

    dtau/dt = 2 G edot - tau / tau_M,        tau_M = eta / G,

from `tau_old` with the strain rate `edot` held over the step:

    tau_new = (tau_old + 2 G edot dt) / (1 + dt / tau_M).

`G <= 0` returns `tau_old` unchanged, matching the `shear_modulus = 0` convention of
`Params`: the viscoelastic terms are off, not elastic with zero modulus.

The steady state of the iteration is `2 eta edot`, the purely viscous stress, and the
first step from rest in the limit `dt << tau_M` is `2 G edot dt`, the elastic one. Both
are gated by G244.
"""
function maxwell_stress_step(tau_old::Float64, edot::Float64, dt::Float64,
                             eta::Float64, G::Float64)
    G <= 0.0 && return tau_old
    tau_M = eta / G
    return (tau_old + 2.0 * G * edot * dt) / (1.0 + dt / tau_M)
end

"""
    maxwell_stress_exact(t, edot, eta, G)

The continuous solution of the same equation from rest,

    tau(t) = 2 eta edot (1 - exp(-t / tau_M)),

used by the gate as the reference the discrete update converges to as `dt` shrinks.
"""
function maxwell_stress_exact(t::Float64, edot::Float64, eta::Float64, G::Float64)
    G <= 0.0 && return 0.0
    tau_M = eta / G
    return 2.0 * eta * edot * (1.0 - exp(-t / tau_M))
end
