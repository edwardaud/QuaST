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

"""
    maxwell_force_stress(txx, tyy, txy, eta, dt, G)

The previous step's deviatoric stress as the **momentum equation** sees it
(2b step 2), i.e. the elastic force term `div(tau)` of ASPECT's viscoelastic
formulation.

One backward-Euler step of the Maxwell update splits the new stress into

    tau^{n+1} = 2 eta_eff eps^{n+1} + a tau^n,      a = eta/(eta + G dt),
    eta_eff   = 1/(1/eta + 1/(G dt)),

and the first term is already carried by the Stokes operator through the
effective viscosity.  The second is an explicit load: momentum gains
`div(a tau^n)`, whose weak form against the velocity test functions is
`-int a tau^n : eps(v)`.  This function returns the three nodal components of
`a tau^n` (`eta` is the *material*, nodal viscosity; `G dt` is the same product
the 2a closure uses), which is exactly the tensor field the Stokes assembly
turns into that load.

The two states the scheme needs follow from `a`:

    tau = 0      -> no elastic force,
    tau = 2 eta eps  ->  eta_eff + a eta = eta,

so an equilibrated viscoelastic flow is the purely viscous one (material `eta`),
not the `eta_eff` one; the two agree as `G dt >> eta` (the coefficient is
`tau_M/dt`).  G247 pins that limit.

`G <= 0` is rejected: with `shear_modulus = 0` the term must not be assembled at
all (the caller's convention), and the formula above would silently return
`tau` itself.
"""
function maxwell_force_stress(txx::Vector{Float64}, tyy::Vector{Float64},
                              txy::Vector{Float64}, eta::Vector{Float64},
                              dt::Float64, G::Float64)
    G > 0.0 || throw(ArgumentError("maxwell_force_stress requires G > 0; with " *
                                   "shear_modulus = 0 the elastic force term " *
                                   "must not be assembled"))
    dt > 0.0 || throw(ArgumentError("maxwell_force_stress requires dt > 0"))
    n = length(eta)
    (length(txx) == n && length(tyy) == n && length(txy) == n) ||
        throw(ArgumentError("the Maxwell stress fields and eta must have the " *
                            "same length; got ($(length(txx)), " *
                            "$(length(tyy)), $(length(txy))) against $n"))
    sx = Vector{Float64}(undef, n)
    sy = Vector{Float64}(undef, n)
    sz = Vector{Float64}(undef, n)
    Gdt = G * dt
    @inbounds for i in 1:n
        a = eta[i] / (eta[i] + Gdt)
        sx[i] = a * txx[i]
        sy[i] = a * tyy[i]
        sz[i] = a * txy[i]
    end
    return (sx, sy, sz)
end

# 2a effective-viscosity softening for the per-step Newton viscosity factory
# (`newton_make_eta` in driver/run.jl).  The Newton residual assembles
# `2 eta_eff eps(u)` with the softened viscosity, so every derivative closure it
# hands to the analytic tangent has to be softened with the chain rule
#
#     d eta_eff / dx = (eta_eff/eta)^2 d eta / dx.
#
# A `nothing` derivative entry (e.g. the strain-rate derivative of a
# temperature-only law) stays `nothing`: the tangent code uses `=== nothing` to
# decide whether a block exists at all.  G = 0 returns `mk` unchanged.
function _ve_soften_eta_factory(mk::Tuple, G::Float64, dt::Float64)
    G <= 0.0 && return mk
    eta_raw = mk[1]
    eta_eff = (x, y) -> begin
        e = eta_raw(x, y)
        1.0 / (1.0 / e + 1.0 / (G * dt))
    end
    r2 = (x, y) -> begin
        e = eta_raw(x, y)
        ee = eta_eff(x, y)
        (ee / e)^2
    end
    dT = mk[2] === nothing ? nothing : ((x, y) -> r2(x, y) * mk[2](x, y))
    de = length(mk) >= 3 && mk[3] !== nothing ?
         ((x, y, ed) -> r2(x, y) * mk[3](x, y, ed)) : nothing
    if length(mk) >= 5
        dp = mk[4] === nothing ? nothing :
             ((x, y, ed, pq) -> r2(x, y) * mk[4](x, y, ed, pq))
        dchi = mk[5] === nothing ? nothing :
               ((x, y, ed) -> r2(x, y) * mk[5](x, y, ed))
        return (eta_eff, dT, de, dp, dchi)
    end
    return (eta_eff, dT, de)
end
