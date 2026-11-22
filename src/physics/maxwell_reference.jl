# The analytic references of the Maxwell update.
#
# `maxwell_stress_exact` is the continuous solution of the scalar relaxation from
# rest, `maxwell_stress_exact_corotational` the objective solution of the full
# two-component equation under a constant spin.  Both are used by the gates as
# the curves the discrete update converges to as `dt` shrinks, so they are
# independent of the driver and of the mesh.

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

"""
    maxwell_stress_exact_corotational(t, edot_xx, edot_zz, edot_xz, wxy, eta, G)

The continuous solution from rest of the **objective** Maxwell equation

    dtau/dt = 2 G ehat + (W tau - tau W) - tau / tau_M,     tau_M = eta / G,

with a constant deviatoric strain rate `ehat` (the three in-plane components
`edot_xx = ehat_xx`, `edot_zz = ehat_zz`, `edot_xz = ehat_xz`; the out-of-plane
component is `-tr/3`, implied) and a constant spin `wxy = W_xy`.  It is the
reference for `maxwell_stress_exact` when the spin is on, the curve the discrete
update converges to as `dt` shrinks.

Decompose the in-plane tensor into

    m = (tau_xx + tau_yy)/2,   a = (tau_xx - tau_yy)/2,   b = tau_xy.

The spin term acts on `(a, b)` as a rotation at rate `-2 wxy` (`m` is invariant,
and so is the implied `tau_zz = -2m`), so with `E_m = (ehat_xx + ehat_zz)/2`,
`E_a = (ehat_xx - ehat_zz)/2`, `E_b = ehat_xz` the three scalar equations are

    dm/dt = 2 G E_m - m/tau_M
    d/dt [a; b] = 2 G [E_a; E_b] + (A2 - 1/tau_M) [a; b],
        A2 = [0 2 wxy; -2 wxy 0],

whose solution from `tau = 0` is

    m(t) = 2 eta E_m (1 - exp(-t/tau_M))
    [a; b](t) = M^{-1} (exp(M t) - I) 2 G [E_a; E_b],
        M = A2 - 1/tau_M,   exp(M t) = exp(-t/tau_M) [cos 2wxy t  sin 2wxy t;
                                                       -sin 2wxy t  cos 2wxy t],

and `tau_xx = m + a`, `tau_yy = m - a`, `tau_xy = b`.  The steady state is
`tau = 2 eta ehat + tau_M (W tau - tau W)`, i.e. the co-rotational (Jaumann)
steady stress: the rotation term does *not* vanish in a steady flow, which is
why the equilibrated `2 eta ehat` fixed point of the spin-free update moves by
`O(tau_M |wxy|)` once the term is on.

`G <= 0` returns `(0, 0, 0)`, matching `maxwell_stress_exact`: with the
viscoelastic terms off there is no stress history.
"""
function maxwell_stress_exact_corotational(t::Float64, edot_xx::Float64,
                                           edot_zz::Float64, edot_xz::Float64,
                                           wxy::Float64, eta::Float64,
                                           G::Float64)
    G <= 0.0 && return (0.0, 0.0, 0.0)
    tau_M = eta / G
    decay = exp(-t / tau_M)
    # the in-plane trace of ehat drives m; the third (out-of-plane) component is
    # implied by trace zero and never entered the two normal components directly.
    E_m = 0.5 * (edot_xx + edot_zz)
    E_a = 0.5 * (edot_xx - edot_zz)
    E_b = edot_xz
    m = 2.0 * eta * E_m * (1.0 - decay)
    # exp(M t) for M = A2 - 1/tau_M
    th = 2.0 * wxy * t
    E1 = decay * [cos(th) sin(th); -sin(th) cos(th)]
    # M^{-1} = 1/(1/tau_M^2 + 4 wxy^2) [-1/tau_M  -2 wxy;  2 wxy  -1/tau_M]
    det = 1.0 / tau_M^2 + 4.0 * wxy^2
    Mi = (1.0 / det) * [-1.0 / tau_M -2.0 * wxy; 2.0 * wxy -1.0 / tau_M]
    v = Mi * (E1 - [1.0 0.0; 0.0 1.0]) * (2.0 * G * [E_a, E_b])
    return (m + v[1], m - v[1], v[2])
end
