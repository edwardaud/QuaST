# The Maxwell update kernels: one backward-Euler step of the objective equation
#
#     dtau/dt = 2 G ehat + (W tau - tau W) - tau / tau_M,     tau_M = eta / G,
#
# split into the scalar relaxation (`maxwell_stress_step`) and the co-rotational
# (Jaumann) increment (`maxwell_stress_corotate`) that the driver composes.  The
# update is *objective*: a Maxwell stress that does not rotate with the material
# is not frame-indifferent, and every mantle-convection run is a large-strain run.
#
# The capability names, states and fixtures live in:
#   - maxwell_stress_step.md (the scalar kernel)
#   - maxwell_stress_corotate.md (the co-rotational increment)
#
# The analytic reference for the composed update is `maxwell_stress_exact_corotational`
# (src/physics/maxwell_reference.jl); the driver-side composition is
# `_advance_maxwell_stress!` in src/driver/maxwell_state.jl.

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
    maxwell_stress_corotate(txx, tyy, txy, wxy, dt)

The **objective** (co-rotational, Jaumann) part of the Maxwell stress update:
the in-plane components of `tau + dt*(W tau - tau W)` for the deviatoric stress
`tau = [txx txy; txy tyy]` and the spin

    W = [0 wxy; -wxy 0],     wxy = (du_x/dy - du_z/dx) / 2,

with `(x, y)` the in-plane axes (`u_z` is the second in-plane velocity
component, the code's `uz`).  `wxy` is the off-diagonal component of the spin,
the antisymmetric part of the velocity gradient.

The in-plane component form follows row by row (`W_yx = -W_xy`):

    (W tau - tau W)_xx = 2 wxy txy
    (W tau - tau W)_yy = -2 wxy txy
    (W tau - tau W)_xy = wxy (tyy - txx)

Row `xx`: `(W tau)_xx = W_xy tau_yx = wxy txy` and
`(tau W)_xx = tau_xy W_yx = -wxy txy`, whose difference is twice the first.
Row `yy` is the negative of row `xx` because `W_yx = -W_xy`.  Row `xy`:
`(W tau)_xy = W_xy tau_yy = wxy tyy` and `(tau W)_xy = tau_xx W_xy = wxy txx`,
so their difference is `wxy (tyy - txx)`.  That last sign is easy to get
backwards -- the naive `wxy (txx - tyy)` rotates the material the wrong way;
`test/t_g250_corotation.jl` pins these three lines against an explicit matrix
product `W*T - T*W` before it runs a rotation.

The out-of-plane normal component is untouched, `(W tau - tau W)_zz = 0`, and
the in-plane trace is preserved (the two normal components are exact
negatives), so a deviatoric stress stays deviatoric: no projection is needed
on this term.  The spin is a pure rotation generator about the out-of-plane
axis, so the stored triple is closed under it.

This is the rotation *linearised* over the step, `1 + dt*ad_W`, not the exact
exponential of a finite rotation.  That is deliberate: it is the same
first-order linearisation as the backward-Euler relaxation it is composed with,
and it is the lagged numerator of the update in ASPECT's `viscoelastic` model,
`tau^{n+1} = (tau^n + 2 G dt ehat + dt (W tau^n - tau^n W)) / (1 + dt/tau_M)`.
The driver composes it as

    tau^{n+1} = maxwell_stress_step(maxwell_stress_corotate(tau^n, wxy, dt),
                                    ehat^n, dt, eta, G),

`dt = 0` or `wxy = 0` returns the input components exactly, which is why the
term is a no-op for a rotation-free flow and costs a handful of flops per node.
"""
function maxwell_stress_corotate(txx::Float64, tyy::Float64, txy::Float64,
                                 wxy::Float64, dt::Float64)
    return (txx + 2.0 * dt * wxy * txy,
            tyy - 2.0 * dt * wxy * txy,
            txy + dt * wxy * (tyy - txx))
end
