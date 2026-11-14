"""Apply the viscosity bounds (hard or smooth floor, hard ceiling)."""
@inline function _clip_eta(eta::Float64, pr::Params)
    clipped = false
    if isfinite(pr.eta_min)
        if pr.eta_soft > 0.0
            # smooth floor: eta_min + w*log(1+exp((eta-eta_min)/w))
            t = (eta - pr.eta_min) / pr.eta_soft
            eta = pr.eta_min + pr.eta_soft * log1p(exp(min(t, 50.0)))
            clipped = t < 0.0
        elseif eta < pr.eta_min
            eta = pr.eta_min
            clipped = true
        end
    end
    if isfinite(pr.eta_max) && eta > pr.eta_max
        eta = pr.eta_max
        clipped = true
    end
    return eta, clipped
end

"""eta(T) and whether a bound clipped it."""
@inline function eta_of_T(T::Float64, pr::Params)
    pr.eta_law === :expTz &&
        throw(ArgumentError("eta_law=:expTz depends on z; use eta_at(T, z, pr)"))
    eta = if pr.eta_law === :expT
        exp(-log(pr.deta_T) * T)
    else
        pr.eta_const
    end
    return _clip_eta(eta, pr)
end

"""
    eta_at(T, z, pr)

Viscosity at temperature `T` and height `z` (z = 0 bottom, 1 top).

`:expTz` is the joint temperature/depth exponential law of ASPECT's
`nondimensional` material model (source/material_model/nondimensional.cc:73-75):

    eta = (compressible ? Di : 1)/Ra * exp(-b*(T - T_ad) + c*depth)

For the Boussinesq benchmark runs `T_ad = 0`, `depth = 1 - z`, and the
`1/Ra` prefactor is absorbed by our body-force scaling (Ra multiplies the
buoyancy, viscosity is O(1)); this is the same nondimensionalisation that
reproduces Blankenbach 2a with `:expT`.  `eta_b` and `eta_c` are ASPECT's
"Viscosity temperature prefactor" and "Viscosity depth prefactor" verbatim
(2b: b = ln(16384), c = ln(64), from benchmarks/blankenbach/base_case2b.prm).

Only the Taylor-Hood assembly calls `eta_at`; the experimental psi/RT velocity
spaces and the potential projection still call `eta_of_T`, which rejects
`:expTz` because it has no z.
"""
@inline function eta_at(T::Float64, z::Float64, pr::Params)
    pr.eta_law === :tosi &&
        throw(ArgumentError("eta_law=:tosi depends on the strain rate; use " *
                            "semquad_eta_tosi / the rheology path instead"))
    pr.eta_law === :composite &&
        throw(ArgumentError("eta_law=:composite depends on the strain rate; " *
                            "use eta_composite_at(T, edot, z, pr)"))
    if pr.eta_law === :expTz
        return _clip_eta(exp(-pr.eta_b * T + pr.eta_c * (1.0 - z)), pr)
    end
    if pr.eta_law === :arrhenius
        # Nondimensional Arrhenius law, normalised to eta(T=1)=1:
        #   eta = exp( E/(T + T0) - E/(1 + T0) )
        # T is the absolute temperature in [0,1] (T=1 bottom, T=0 top), T0 > 0
        # is the dimensionless surface offset.  As E -> 0 this tends to 1.
        return _clip_eta(exp(pr.eta_E * (1.0 / (T + pr.eta_T0) -
                                         1.0 / (1.0 + pr.eta_T0))), pr)
    end
    return eta_of_T(T, pr)
end

"""
    deta_dT_at(T, z, pr)

Derivative `d eta / d T` of the temperature-dependent viscosity laws, used by
the analytic tangent-linear Newton map (`semquad_newton.jl`).  The hard/smooth
viscosity bounds are not differentiated: at a bound the derivative is returned
as zero, which is the correct one-sided value for a hard clamp (and a deliberate
approximation for the smooth floor).  `:tosi` is rejected because its viscosity
also depends on the strain rate.
"""
@inline function deta_dT_at(T::Float64, z::Float64, pr::Params)
    pr.eta_law === :tosi &&
        throw(ArgumentError("deta_dT_at does not support eta_law=:tosi; its " *
                            "viscosity also depends on the strain rate"))
    pr.eta_law === :composite &&
        throw(ArgumentError("deta_dT_at does not support eta_law=:composite; " *
                            "use deta_composite_at(T, edot, z, pr)"))
    if pr.eta_law === :expTz
        raw = exp(-pr.eta_b * T + pr.eta_c * (1.0 - z))
        eta, clipped = _clip_eta(raw, pr)
        clipped && return 0.0
        return -pr.eta_b * eta
    elseif pr.eta_law === :arrhenius
        raw = exp(pr.eta_E * (1.0 / (T + pr.eta_T0) -
                              1.0 / (1.0 + pr.eta_T0)))
        eta, clipped = _clip_eta(raw, pr)
        clipped && return 0.0
        return eta * (-pr.eta_E / (T + pr.eta_T0)^2)
    elseif pr.eta_law === :expT
        raw = exp(-log(pr.deta_T) * T)
        eta, clipped = _clip_eta(raw, pr)
        clipped && return 0.0
        return -log(pr.deta_T) * eta
    end
    return 0.0
end

# ---------------------------------------------------------------------------
# :composite diffusion + dislocation creep (ASPECT `diffusion dislocation`)
# ---------------------------------------------------------------------------

"""
    composite_Tnd(T, pr)

Nondimensional absolute temperature used by the `:composite` Arrhenius
exponents: `T + eta_T0`, optionally floored at `comp_Tcut` (`0` = no floor).
`T` is the code's nondimensional temperature (1 at the bottom wall, 0 at the
top), so `eta_T0` is the surface-temperature offset, exactly as in
`:arrhenius`, and `comp_Tcut` is the regularisation that stops
`exp(E/(n T_nd))` from overflowing as `T_nd -> 0`.
"""
@inline function composite_Tnd(T::Float64, pr::Params)
    Tnd = T + pr.eta_T0
    return pr.comp_Tcut > 0.0 ? max(Tnd, pr.comp_Tcut) : Tnd
end

"""
    composite_branch_inv(T, z, pr) -> (inv_diff, inv_disl_unit, T_nd, depth)

The two **inverse** viscosities of the `:composite` law (ASPECT's
`diffusion dislocation`, source/material_model/diffusion_dislocation.cc) at a
*unit* strain rate:

    inv_diff        = A_d exp(-(E_d + V_d (1-z)) / T_nd)
    inv_disl(ed)    = inv_disl_unit * ed^((n-1)/n),
    inv_disl_unit   = A_s exp(-(E_s + V_s (1-z)) / (n T_nd))

`depth = 1-z` is returned because the analytic temperature derivative needs the
activation-volume coefficients `E + V*depth`.
"""
@inline function composite_branch_inv(T::Float64, z::Float64, pr::Params)
    Tnd = composite_Tnd(T, pr)
    depth = 1.0 - z
    inv_diff = pr.comp_Ad * exp(-(pr.comp_Ed + pr.comp_Vd * depth) / Tnd)
    inv_disl_unit = pr.comp_As *
                    exp(-(pr.comp_Es + pr.comp_Vs * depth) /
                        (pr.comp_n * Tnd))
    return inv_diff, inv_disl_unit, Tnd, depth
end

"""
    composite_inv_eta(T, edot, z, pr) -> (inv_diff, inv_disl, T_nd, edot_used)

The two **inverse** viscosities of the `:composite` law evaluated at the
**total** strain rate, i.e. the `comp_rule = :isostrain` combination:

    inv_diff = A_d exp(-(E_d + V_d (1-z)) / T_nd)
    inv_disl = A_s exp(-(E_s + V_s (1-z)) / (n T_nd)) * edot_used^((n-1)/n)
    eta_eff  = 1 / (inv_diff + inv_disl)

This is the *legacy* combination (both mechanisms see the whole strain rate,
the harmonic mean of the branch viscosities); the default `:isostress` rule
does **not** use this function -- see [`composite_isostress_state`](@ref).

Working with the inverse viscosities instead of `eta_diff`/`eta_disl` keeps the
`A -> 0` limits (one mechanism switched off) finite: `A_s = 0` gives
`inv_disl = 0` and hence the pure diffusion law, and vice versa.  `edot_used =
max(edot, comp_eps_min)`, so `edot -> 0` is the diffusion limit suggested by
ASPECT's "Minimum strain rate" parameter instead of `eta_disl = Inf`.
"""
@inline function composite_inv_eta(T::Float64, edot::Float64, z::Float64,
                                   pr::Params)
    inv_diff, inv_disl_unit, Tnd, _ = composite_branch_inv(T, z, pr)
    ed = max(edot, pr.comp_eps_min)
    q = (pr.comp_n - 1.0) / pr.comp_n
    inv_disl = inv_disl_unit * ed^q
    return inv_diff, inv_disl, Tnd, ed
end

"""
    CompositeIsoState

Diagnostics of one `comp_rule = :isostress` quadrature-point solve, returned by
[`composite_isostress_state`](@ref):

| field | meaning |
|---|---|
| `eta` | unclipped effective viscosity `tau/(2 ed)`, `ed = max(edot, comp_eps_min)` |
| `x` | `eta/eta_disl(ed)`, the solution of `k x + x^n = 1` |
| `ed_d` | strain rate carried by the diffusion branch, `k x ed` |
| `ed_s` | strain rate carried by the dislocation branch, `x^n ed` |
| `inv_diff`, `inv_disl` | the two inverse branch viscosities at `ed` |
| `ed` | the floored strain rate the solve used |
| `iters` | Newton/bisection iterations (0 in the closed-form limits) |
| `converged` | whether the residual tolerance was reached |

`ed_d + ed_s == ed` up to round-off, and `eta == ed_d/(inv_diff*ed)` when
neither viscosity bound is active.
"""
struct CompositeIsoState
    eta::Float64
    x::Float64
    ed_d::Float64
    ed_s::Float64
    inv_diff::Float64
    inv_disl::Float64
    ed::Float64
    iters::Int
    converged::Bool
end

"""
    COMP_ISO_MAXIT

Iteration cap of the `comp_rule = :isostress` scalar solve (Newton with a
bisection safeguard).  The measured worst case over the 495-point constitutive
card (`t_g200`/`t_g201`) is 6 iterations; the cap exists so a pathological
state cannot stall a time step, and `CompositeIsoState.converged` reports when
it is hit (the returned value is then the best bracket point, never a NaN).
"""
const COMP_ISO_MAXIT = 100

"""
    COMP_ISO_TOL

Absolute tolerance of the `comp_rule = :isostress` scalar solve, applied to the
dimensionless residual `g(x) = k x + x^n - 1`.  Near the root
`x dg/dx = 1 + (n-1) x^n >= 1`, so `|g| <= COMP_ISO_TOL` bounds the relative
error of `x` (and therefore of `eta`) by `COMP_ISO_TOL = 1e-15`, i.e. within a
few ulp of the exact root.  The loop also stops when the Newton update falls
below `4 eps(x)` (the residual cannot be evaluated more accurately there).
"""
const COMP_ISO_TOL = 1.0e-15

"""
    composite_isostress_state(T, edot, z, pr) -> CompositeIsoState

Solve the **isostress** partition of the two creep mechanisms at one quadrature
point -- ASPECT's `diffusion_dislocation` law and the `comp_rule = :isostress`
default:

    edot_d(tau) + edot_s(tau) = edot,
    edot_d = tau/(2 eta_diff),   edot_s = A_s exp(-(E_s+V_s(1-z))/(R T)) tau^n

Because `edot_s = (B tau/2)^n` with `B = inv_disl_unit`, the viscosity
`eta = tau/(2 ed)` is obtained from the equivalent scalar equation

    g(x) = k x + x^n - 1 = 0,      x = eta/eta_disl(ed) in (0, 1],
    k = eta_disl(ed)/eta_diff = inv_diff/inv_disl(ed),

which is monotone increasing with `g(0+) = -1` and `g(1) = k >= 0`, so a
bracketed root always exists in `[0, 1]` for any `n > 0`.  The solver is Newton
on `x` (started at the harmonic value `1/(1+k)`, which is the exact `n = 1`
solution) with a bisection safeguard on `[0, 1]`, the tolerance
[`COMP_ISO_TOL`](@ref) and the cap [`COMP_ISO_MAXIT`](@ref); then
`eta = x/inv_disl(ed)`, which is `tau/(2 ed)` by construction.

Limits handled in closed form (no iteration, and identical to the
`:isostrain` value bit for bit): `comp_As = 0` (pure diffusion), `comp_Ad = 0`
(pure dislocation) and `n = 1` (both branches linear, where the two
combination rules coincide).  The two combination rules also agree in the
one-mechanism limits of the *physics* (one branch overwhelming the other), but
not in between: the isostress viscosity is larger by up to 41% for `n = 3.5`
(docs/dev/PHYSICS.md section 14.4b).
"""
@inline function composite_isostress_state(T::Float64, edot::Float64,
                                           z::Float64, pr::Params)
    inv_diff, inv_disl_unit, _, _ = composite_branch_inv(T, z, pr)
    n = pr.comp_n
    ed = max(edot, pr.comp_eps_min)
    q = (n - 1.0) / n
    inv_disl = inv_disl_unit * ed^q
    if inv_disl == 0.0
        # pure diffusion (comp_As == 0, or its exponent underflowed)
        return CompositeIsoState(1.0 / inv_diff, 0.0, ed, 0.0, inv_diff, 0.0,
                                 ed, 0, true)
    elseif inv_diff == 0.0
        # pure dislocation (comp_Ad == 0, or its exponent underflowed)
        return CompositeIsoState(1.0 / inv_disl, 1.0, 0.0, ed, 0.0, inv_disl,
                                 ed, 0, true)
    elseif n == 1.0
        # Both branches are linear, so the isostress partition and the
        # harmonic (isostrain) mean are the same function of edot.
        eta = 1.0 / (inv_diff + inv_disl)
        return CompositeIsoState(eta, eta * inv_disl, eta * inv_diff * ed,
                                 eta * inv_disl * ed, inv_diff, inv_disl, ed,
                                 0, true)
    end
    k = inv_diff / inv_disl
    if !isfinite(k)
        # inv_diff overflowed relative to inv_disl: the diffusion branch carries
        # the whole rate (x <= k^(-1/n) is below the representable range).
        return CompositeIsoState(1.0 / inv_diff, 0.0, ed, 0.0, inv_diff,
                                 inv_disl, ed, 0, true)
    end
    # bracket: g(0+) = -1 < 0 and g(1) = k >= 0 for any n > 0.  For the physical
    # n >= 1 the tighter bound 1/(1+k) is also valid (g(1/(1+k)) <= 0), so it is
    # used as the lower end; for n < 1 only [0, 1] is guaranteed and the
    # bisection safeguard falls back to it.
    x = 1.0 / (1.0 + k)
    lo = n >= 1.0 ? x : 0.0
    hi = 1.0
    x = x > 0.0 ? x : 0.5          # k = Inf is handled above; guard round-off
    it = 0
    ok = false
    while it < COMP_ISO_MAXIT
        it += 1
        xn = x^n
        g = k * x + xn - 1.0
        if abs(g) <= COMP_ISO_TOL
            ok = true
            break
        end
        if g < 0.0
            lo = x
        else
            hi = x
        end
        # Newton step, safeguarded by the bracket (bisection when the step
        # leaves it, is not finite, or the derivative is not positive)
        dg = k + n * xn / x
        xnew = isfinite(dg) && dg > 0.0 ? x - g / dg : NaN
        if !isfinite(xnew) || xnew <= lo || xnew >= hi
            xnew = 0.5 * (lo + hi)
        end
        if abs(xnew - x) <= 4.0 * eps(x)
            x = xnew
            ok = true
            break
        end
        x = xnew
    end
    # ed_d and ed_s are the stable products of the partition (never a
    # difference ed - ed_d, which would cancel at the one-mechanism limits)
    xn = x^n
    return CompositeIsoState(x / inv_disl, x, (k * x) * ed, xn * ed,
                             inv_diff, inv_disl, ed, it, ok)
end

"""
    eta_composite_at(T, edot, z, pr) -> (eta, clipped)

Effective viscosity of the `:composite` law at temperature `T`, strain-rate
invariant `edot = sqrt(0.5 edot:edot)` and height `z` (0 bottom, 1 top).  The
combination is selected by `pr.comp_rule`:

    comp_rule = :isostress (default):  the scalar solve of
        `composite_isostress_state`, i.e. tau/(2 eta_diff) + edot_s(tau) = edot
    comp_rule = :isostrain (legacy):   eta = 1 / (1/eta_diff + 1/eta_disl)

followed by the same `eta_min/eta_max` clip (`_clip_eta`) as every other law
(the clip is applied **once**, to the effective viscosity, exactly like
ASPECT's Newton path).  `clipped` reports whether a bound was hit, so the
caller can count clamp events (`t_g192`, `t_g193`).  `:composite` is only
reachable through this function -- `eta_at(T, z, pr)` throws, because the law
needs the strain rate.

In the code's nondimensionalisation the viscosity unit is the reference
viscosity `eta0` of the Boussinesq scaling (`dimensionless` ASPECT), so
`comp_Ad`/`comp_As` are `A_i * eta0^(...)`-style prefactors in units of
`1/eta0`; docs/dev/PHYSICS.md gives the exact ASPECT mapping (including the
factor 2 of ASPECT's `eta = stress/edot/2` convention, which is absorbed into
the prefactors here).
"""
@inline function eta_composite_at(T::Float64, edot::Float64, z::Float64,
                                  pr::Params)
    pr.eta_law === :composite ||
        throw(ArgumentError("eta_composite_at requires eta_law=:composite"))
    if pr.comp_rule === :isostrain
        inv_diff, inv_disl, _, _ = composite_inv_eta(T, edot, z, pr)
        inv = inv_diff + inv_disl
        # inv == 0 can only happen for A_d = A_s = 0 (rejected by Params) or
        # when both exponents underflow; eta = Inf then hits a finite eta_max.
        eta = inv > 0.0 ? 1.0 / inv : Inf
        return _clip_eta(eta, pr)
    end
    return _clip_eta(composite_isostress_state(T, edot, z, pr).eta, pr)
end

"""
    deta_composite_at(T, edot, z, pr) -> (deta_dT, deta_deps)

Analytic derivatives of `:composite`, in the same nondimensional units as
`eta_composite_at`.  The two combination rules have different derivatives.

`comp_rule = :isostrain` (legacy, the exact derivative of the harmonic mean):

    deta_dT   = -eta^2 [ (E_d + V_d(1-z)) inv_diff
                         + (E_s + V_s(1-z))/n inv_disl ] / T_nd^2
    deta_deps = -eta^2 * q * inv_disl / edot_used,   q = (n-1)/n

`comp_rule = :isostress` (implicit differentiation of
`F(tau, T, edot) = tau/(2 eta_d) + edot_s(tau) - edot = 0`, with
`F_tau = (ed_d + n ed_s)/tau`, `F_T = (ed_d Q_d + ed_s Q_s)/T_nd^2`,
`Q_i = E_i + V_i(1-z)`):

    deta_dT   = -eta (ed_d Q_d + ed_s Q_s) / (T_nd^2 (ed_d + n ed_s))
    deta_deps = -(n-1) ed_s eta / (edot (ed_d + n ed_s))

with `ed_d = k x ed`, `ed_s = x^n ed` from `composite_isostress_state`; both
reduce to the isostrain values in the one-mechanism limits and coincide with
the isostrain formulas at `n = 1`.

(`deta_dT < 0`, `deta_deps < 0` for `n > 1`.)  Both are zero when the
`eta_min/eta_max` clamp is active -- the one-sided value of a hard clamp, the
same convention as `deta_dT_at`.  `deta_dT` is also zero while `comp_Tcut` is
active (the exponent is constant there) and `deta_deps` is zero for
`edot <= comp_eps_min` (the strain-rate floor freezes the dislocation branch).
`deta_deps` is the `Kgeo` coefficient of the tangent infrastructure
(`solve_semquad_stokes_var(...; geo_u=..., geo_coef=...)`); `deta_dT` is the
energy-coupling term of the analytic Newton.
"""
@inline function deta_composite_at(T::Float64, edot::Float64, z::Float64,
                                   pr::Params)
    pr.eta_law === :composite ||
        throw(ArgumentError("deta_composite_at requires eta_law=:composite"))
    if pr.comp_rule === :isostrain
        inv_diff, inv_disl, Tnd, ed = composite_inv_eta(T, edot, z, pr)
        inv = inv_diff + inv_disl
        inv > 0.0 || return 0.0, 0.0
        eta, clipped = _clip_eta(1.0 / inv, pr)
        clipped && return 0.0, 0.0
        depth = 1.0 - z
        deta_dT = 0.0
        if !(pr.comp_Tcut > 0.0 && T + pr.eta_T0 <= pr.comp_Tcut)
            deta_dT = -(eta * eta) *
                      ((pr.comp_Ed + pr.comp_Vd * depth) * inv_diff +
                       (pr.comp_Es + pr.comp_Vs * depth) /
                       pr.comp_n * inv_disl) / (Tnd * Tnd)
        end
        deta_deps = 0.0
        if edot > pr.comp_eps_min && inv_disl > 0.0
            q = (pr.comp_n - 1.0) / pr.comp_n
            deta_deps = -(eta * eta) * q * inv_disl / ed
        end
        return deta_dT, deta_deps
    end
    st = composite_isostress_state(T, edot, z, pr)
    eta, clipped = _clip_eta(st.eta, pr)
    clipped && return 0.0, 0.0
    _, _, Tnd, depth = composite_branch_inv(T, z, pr)
    den = st.ed_d + pr.comp_n * st.ed_s
    deta_dT = 0.0
    if !(pr.comp_Tcut > 0.0 && T + pr.eta_T0 <= pr.comp_Tcut)
        deta_dT = -eta * (st.ed_d * (pr.comp_Ed + pr.comp_Vd * depth) +
                          st.ed_s * (pr.comp_Es + pr.comp_Vs * depth)) /
                  (Tnd * Tnd * den)
    end
    deta_deps = 0.0
    if edot > pr.comp_eps_min
        deta_deps = -(pr.comp_n - 1.0) * st.ed_s * eta / (edot * den)
    end
    return deta_dT, deta_deps
end
