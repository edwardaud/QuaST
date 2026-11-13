# Physical parameters, base state, viscosity law and diagnostics.

"""
    Params(; Ra, H, eta_law, deta_T, eta_b, eta_c, ...)

Viscosity laws and their parameter semantics (**the two families do not share a
meaning**; mixing them up silently changes the viscosity by orders of magnitude):

- `eta_law = :expT` -- `eta = exp(-log(deta_T) * T)`, i.e. `deta_T` is a
  **viscosity ratio**: `eta(T=0) = 1` and `eta(T=1) = 1/deta_T`.  `deta_T = 1`
  is constant viscosity, `deta_T = 1e4` makes the hot end 1e4 times weaker.
  (`deta_T = 1e-8` therefore means the *hot* end is 1e8 times **stiffer**, which
  is the opposite of the usual mantle law.)
- `eta_law = :expTz` -- `eta = exp(-eta_b * T + eta_c * (1 - z))`, where `eta_b`
  and `eta_c` are **coefficients** (ASPECT's "Viscosity temperature/depth
  prefactor"); `eta_b = log(1e4)` gives a factor 1e4 across the box.
- `eta_law = :arrhenius` -- uses `eta_E` and `eta_T0`; `eta_law = :const` (and
  any other value) returns `eta_const`.
- `eta_law = :composite` -- diffusion + dislocation creep (ASPECT's
  `diffusion dislocation` material model), with two combination rules selected
  by `comp_rule`:

      eta_diff = comp_Ad^-1 exp((comp_Ed + comp_Vd*(1-z)) / T_nd)
      eta_disl = comp_As^-1 exp((comp_Es + comp_Vs*(1-z)) / (comp_n*T_nd))
                 * eps_II^((1-comp_n)/comp_n)

      comp_rule = :isostress (default, ASPECT's law):
          both mechanisms carry the same stress, the *partial* strain rates
          add:  tau/(2 eta_diff) + A_s exp(-(E_s+V_s(1-z))/(R T)) tau^n = eps_II
          solved for tau at every quadrature point; eta_eff = tau/(2 eps_II).
      comp_rule = :isostrain (legacy, pre-2026-11-08 results):
          eta_eff = 1 / (1/eta_diff + 1/eta_disl), both mechanisms evaluated at
          the *total* strain rate (the harmonic mean).

  with `T_nd = max(T + eta_T0, comp_Tcut)` (`comp_Tcut = 0` = off), `z` the
  height (0 bottom, 1 top, so `1-z` is depth) and `eps_II` the second invariant
  of the strain rate with the code's `sqrt(0.5 edot:edot)` convention.  The two
  rules agree exactly in the one-mechanism limits and differ by up to 41%
  (n = 3.5) where both mechanisms contribute; ASPECT 2.5.0-pre solves
  `:isostress` (measured, docs/dev/PHYSICS.md section 14).  The `eta_min/eta_max`
  bounds of `_clip_eta` apply to `eta_eff` (clip once, after the combination),
  and `comp_eps_min` floors the strain rate of the dislocation branch (so that
  `eps_II -> 0` is the diffusion limit with a finite viscosity instead of
  `eta_disl = Inf` and an infinite derivative).  `comp_n` is ASPECT's "Stress
  exponent for dislocation creep" (n = 3 in the mantle); `comp_Ad`/`comp_As`
  are ASPECT's prefactors in units of `1/eta0` of the code's
  nondimensionalisation (see docs/dev/PHYSICS.md for the exact ASPECT mapping,
  including the factor 2 that ASPECT's `eta = stress/edot/2` convention puts
  into the prefactors).
"""
struct Params
    Ra::Float64
    H::Float64
    eta_law::Symbol       # :const, :expT, :expTz or :arrhenius
    deta_T::Float64
    eta_b::Float64        # :expTz temperature prefactor b (ASPECT "Viscosity temperature prefactor")
    eta_c::Float64        # :expTz depth prefactor c (ASPECT "Viscosity depth prefactor")
    eta_E::Float64        # :arrhenius nondimensional activation energy E/(R dT)
    eta_T0::Float64       # :arrhenius nondimensional surface temperature offset
    yield_c::Float64      # Drucker-Prager cohesion (nondimensional stress)
    yield_phi::Float64    # friction angle (rad); 0 = pressure independent
    yield_taumax::Float64 # cap on the yield stress (nondimensional)
    yield_eps_min::Float64# strain-rate floor in eta_yield = tau_y/(2 eps)
    yield_damper::Float64 # ASPECT plastic damper viscosity eta_d (0 = off)
    eta_const::Float64
    eta_min::Float64
    eta_max::Float64
    bc::Symbol            # :free or :rigid
    supg::Float64
    eta_soft::Float64     # smooth-floor width; 0 = hard clip
    div_penalty::Float64  # gamma * int (div u)(div v); 0 = off
    eta_custom::Union{Nothing,Function}   # eta(x,z), overrides the T law in Stokes
    rho_heating::Bool     # add -T*H to the energy source (rho=(1-T) heating)
    rho_cp::Bool          # weight M and C by rho=1-T (ASPECT rho*cp LHS)
    tosi::Union{Nothing,TosiLaw}  # :tosi viscoplastic law (absolute units + eta0)
    Di::Float64           # dissipation number alpha*g*D/cp; 0 disables both terms
    heating_adiabatic::Bool     # -Di*T*u_z source in the energy equation
    heating_dissipation::Bool   # (Di/Ra)*2*eta*edot:edot source
    refstate::Union{Nothing,SemRefState}  # adiabatic base state; nothing = conductive
    rho_ref_weight::Bool  # weight energy M/C by rho(y)*cp of the refstate
    compress_p::Bool      # add -rho Di gamma p to the momentum equation
    compress_gamma::Float64  # gamma factor of that term
    Twall_top::Float64    # physical wall temperature (top); theta_wall = Twall - Tad
    Twall_bot::Float64    # physical wall temperature (bottom)
    deviatoric::Bool      # use eps - (1/3) div u I in the stress and shear heating
    latent_L::Float64     # nondimensional latent heat L/(cp dT) of the phase transitions
    latent_mode::Symbol   # :global (latent_L for every transition) or :per_phase
    # :composite diffusion + dislocation creep (ASPECT `diffusion dislocation`)
    comp_Ad::Float64      # diffusion-creep prefactor A_d (units 1/eta0)
    comp_Ed::Float64      # diffusion activation energy E_d/(R dT), nondimensional
    comp_Vd::Float64      # diffusion activation volume V_d, nondimensional depth factor
    comp_As::Float64      # dislocation-creep prefactor A_s (units 1/eta0)
    comp_Es::Float64      # dislocation activation energy E_s/(R dT)
    comp_Vs::Float64      # dislocation activation volume (depth factor)
    comp_n::Float64       # dislocation stress exponent n (~3)
    comp_Tcut::Float64    # floor on T_nd in the Arrhenius exponent; 0 = off
    comp_eps_min::Float64 # strain-rate floor of the dislocation branch
    comp_rule::Symbol     # :isostress (ASPECT, default) or :isostrain (legacy)
    # Depth-dependent Drucker-Prager yield (docs/dev/PHYSICS.md §8): with depth
    # d = 1-y the cohesion and the friction coefficient are linear,
    #   c(d)  = yield_c + yield_c_grad*d
    #   mu(d) = sin(yield_phi) + yield_mu_grad*d
    #   tau_y = min(c(d)*cos(yield_phi) + mu(d)*p, yield_taumax)
    # Both gradients default to 0, which is the historical constant yield
    # surface bit for bit.
    yield_c_grad::Float64   # d(cohesion)/d(depth)
    yield_mu_grad::Float64  # d(mu)/d(depth), mu = pressure coefficient
    # Strain weakening (docs/dev/PHYSICS.md §9): a scalar field chi in [0,1] with
    #   f(chi) = 1 - (1-weak_f)*chi,   tau_y <- f(chi)*tau_y
    weak_f::Float64         # residual weakening factor f_weak in [0,1]; 1 = off
    weak_eps_ref::Float64   # reference (end) plastic strain; <= 0 = off
    weak_heal_time::Float64 # healing time tau_heal; Inf = no healing
    weak_mode::Symbol       # :rate or :linear (ASPECT accumulation)
end

function Params(; Ra::Real=1.0e4, H::Real=0.0, eta_law::Symbol=:const,
                deta_T::Real=1.0, eta_b::Real=0.0, eta_c::Real=0.0,
                eta_E::Real=0.0, eta_T0::Real=0.0,
                yield_c::Real=Inf, yield_phi::Real=0.0,
                yield_taumax::Real=Inf, yield_eps_min::Real=1.0e-30,
                yield_damper::Real=0.0,
                eta_const::Real=1.0,
                eta_min::Real=-Inf, eta_max::Real=Inf,
                bc::Symbol=:free, supg::Real=0.0, eta_soft::Real=0.0,
                div_penalty::Real=0.0,
                eta_custom=nothing, rho_heating::Bool=false,
                rho_cp::Bool=false, tosi=nothing,
                Di::Real=0.0, heating_adiabatic::Bool=false,
                heating_dissipation::Bool=false, refstate=nothing,
                rho_ref_weight::Bool=false,
                compress_p::Bool=false, compress_gamma::Real=1.0,
                Twall_top::Real=0.0, Twall_bot::Real=1.0,
                deviatoric::Bool=false, latent_L::Real=0.0,
                latent_mode::Symbol=:global,
                comp_Ad::Real=0.0, comp_Ed::Real=0.0, comp_Vd::Real=0.0,
                comp_As::Real=0.0, comp_Es::Real=0.0, comp_Vs::Real=0.0,
                comp_n::Real=3.0, comp_Tcut::Real=0.0,
                comp_eps_min::Real=1.0e-12, comp_rule::Symbol=:isostress,
                yield_c_grad::Real=0.0, yield_mu_grad::Real=0.0,
                weak_f::Real=1.0, weak_eps_ref::Real=0.0,
                weak_heal_time::Real=Inf, weak_mode::Symbol=:rate)
    eta_law in (:const, :expT, :expTz, :arrhenius, :tosi, :composite) ||
        throw(ArgumentError("eta_law must be :const, :expT, :expTz, " *
                            ":arrhenius, :tosi or :composite"))
    if eta_law === :tosi
        tosi isa TosiLaw ||
            throw(ArgumentError("eta_law=:tosi requires tosi=TosiLaw(...)"))
        (isfinite(yield_c) || isfinite(yield_taumax) || yield_phi > 0.0) &&
            throw(ArgumentError(":tosi already contains the plastic branch; " *
                                "do not also set yield_c / yield_phi / yield_taumax"))
    elseif tosi !== nothing
        throw(ArgumentError("tosi=... requires eta_law=:tosi"))
    end
    bc in (:free, :rigid) || throw(ArgumentError("bc must be :free or :rigid"))
    eta_soft >= 0 || throw(ArgumentError("eta_soft must be >= 0"))
    div_penalty >= 0 || throw(ArgumentError("div_penalty must be >= 0"))
    if eta_law === :expT
        deta_T > 0 || throw(ArgumentError("deta_T must be > 0"))
    elseif eta_law === :expTz
        eta_b > 0 || throw(ArgumentError("eta_b must be > 0 for :expTz"))
    elseif eta_law === :arrhenius
        eta_E >= 0 || throw(ArgumentError("eta_E must be >= 0 for :arrhenius " *
                                          "(eta_E = 0 is the constant limit)"))
        eta_T0 > 0 || throw(ArgumentError("eta_T0 must be > 0 for :arrhenius " *
                                          "(T + eta_T0 must stay positive)"))
    elseif eta_law === :composite
        # ASPECT's `diffusion dislocation` requires both prefactors (its grain
        # size term is folded into `comp_Ad` here), a finite stress exponent,
        # and the same surface-temperature offset as :arrhenius.
        comp_Ad >= 0 || throw(ArgumentError("comp_Ad must be >= 0 for :composite"))
        comp_As >= 0 || throw(ArgumentError("comp_As must be >= 0 for :composite"))
        comp_Ad + comp_As > 0 ||
            throw(ArgumentError(":composite needs at least one of comp_Ad, " *
                                "comp_As to be > 0"))
        comp_Ed >= 0 || throw(ArgumentError("comp_Ed must be >= 0 for :composite"))
        comp_Vd >= 0 || throw(ArgumentError("comp_Vd must be >= 0 for :composite"))
        comp_Es >= 0 || throw(ArgumentError("comp_Es must be >= 0 for :composite"))
        comp_Vs >= 0 || throw(ArgumentError("comp_Vs must be >= 0 for :composite"))
        comp_n > 0 || throw(ArgumentError("comp_n must be > 0 for :composite " *
                                          "(n = 1 makes the dislocation branch " *
                                          "strain-rate independent)"))
        comp_Tcut >= 0 || throw(ArgumentError("comp_Tcut must be >= 0 " *
                                              "(0 = off)"))
        comp_eps_min > 0 ||
            throw(ArgumentError("comp_eps_min must be > 0 for :composite"))
        comp_rule in (:isostress, :isostrain) ||
            throw(ArgumentError("comp_rule must be :isostress (ASPECT's " *
                                "isostress partition, the default) or " *
                                ":isostrain (the legacy harmonic mean at the " *
                                "total strain rate)"))
        eta_T0 > 0 || throw(ArgumentError("eta_T0 must be > 0 for :composite " *
                                          "(T + eta_T0 must stay positive)"))
        if isfinite(yield_c) || isfinite(yield_taumax) || yield_phi > 0.0
            throw(ArgumentError(":composite does not support the Drucker-Prager " *
                                "plastic branch; use eta_law=:tosi or apply the " *
                                "yield separately"))
        end
    end
    Di >= 0 || throw(ArgumentError("Di must be >= 0"))
    latent_L >= 0 ||
        throw(ArgumentError("latent_L must be >= 0 (it is L/(cp dT) of the " *
                            "phase transitions)"))
    # Latent heat per transition (docs/dev/PHYSICS.md §7.8): `:global` is the
    # historical single `latent_L` for every transition (bitwise unchanged),
    # `:per_phase` uses each `PhaseTransition.latent` and may be negative.
    latent_mode in (:global, :per_phase) ||
        throw(ArgumentError("latent_mode must be :global (one latent_L for " *
                            "every phase transition, the historical " *
                            "behaviour) or :per_phase (PhaseTransition.latent, " *
                            "see phase_latent_coefficient)"))
    yield_c >= 0 || throw(ArgumentError("yield_c must be >= 0"))
    yield_taumax > 0 || throw(ArgumentError("yield_taumax must be > 0"))
    yield_eps_min > 0 || throw(ArgumentError("yield_eps_min must be > 0"))
    yield_damper >= 0 ||
        throw(ArgumentError("yield_damper must be >= 0"))
    0.0 <= yield_phi < 0.5 * pi ||
        throw(ArgumentError("yield_phi must be in [0, pi/2)"))
    # Depth-dependent yield: the linear profiles must stay physical over the
    # whole depth range d in [0,1] (a negative cohesion or friction coefficient
    # would turn the yield surface inside out and is always a parameter error).
    c_top = Float64(yield_c)
    c_bot = Float64(yield_c) + Float64(yield_c_grad)
    (c_top >= 0.0 && c_bot >= 0.0) ||
        throw(ArgumentError("the cohesion profile c(d) = yield_c + " *
                            "yield_c_grad*d must stay >= 0 for d in [0,1] " *
                            "(c(0)=$c_top, c(1)=$c_bot)"))
    mu_top = sin(Float64(yield_phi))
    mu_bot = mu_top + Float64(yield_mu_grad)
    (mu_top >= 0.0 && mu_bot >= 0.0) ||
        throw(ArgumentError("the friction profile mu(d) = sin(yield_phi) + " *
                            "yield_mu_grad*d must stay >= 0 for d in [0,1] " *
                            "(mu(0)=$mu_top, mu(1)=$mu_bot)"))
    isfinite(yield_c_grad) ||
        throw(ArgumentError("yield_c_grad must be finite"))
    isfinite(yield_mu_grad) ||
        throw(ArgumentError("yield_mu_grad must be finite"))
    0.0 <= weak_f <= 1.0 ||
        throw(ArgumentError("weak_f must be in [0, 1] (1 = no weakening)"))
    weak_eps_ref >= 0 ||
        throw(ArgumentError("weak_eps_ref must be >= 0 (0 = weakening off)"))
    weak_heal_time > 0 ||
        throw(ArgumentError("weak_heal_time must be > 0 (Inf = no healing)"))
    weak_mode in (:rate, :linear) ||
        throw(ArgumentError("weak_mode must be :rate or :linear"))
    if weak_eps_ref > 0.0 && weak_f < 1.0 &&
       !(isfinite(yield_c) || isfinite(yield_taumax) || yield_phi > 0.0)
        throw(ArgumentError("strain weakening scales the Drucker-Prager yield " *
                            "stress, so it needs an active yield surface " *
                            "(yield_c / yield_phi / yield_taumax)"))
    end
    if isfinite(yield_c) || isfinite(yield_taumax) || yield_phi > 0.0
        # eta = min(eta_visc, tau_y/(2 eps)) can be driven to zero by a small
        # yield stress, so a finite floor is required for a meaningful plastic
        # branch.
        isfinite(eta_min) ||
            throw(ArgumentError("yielding requires a finite eta_min"))
    end
    Params(Float64(Ra), Float64(H), eta_law, Float64(deta_T),
           Float64(eta_b), Float64(eta_c), Float64(eta_E), Float64(eta_T0),
           Float64(yield_c), Float64(yield_phi), Float64(yield_taumax),
           Float64(yield_eps_min), Float64(yield_damper),
           Float64(eta_const),
           Float64(eta_min), Float64(eta_max), bc, Float64(supg),
           Float64(eta_soft), Float64(div_penalty), eta_custom,
           rho_heating, rho_cp, tosi, Float64(Di),
           heating_adiabatic, heating_dissipation, refstate, rho_ref_weight,
           compress_p, Float64(compress_gamma),
           Float64(Twall_top), Float64(Twall_bot), deviatoric,
           Float64(latent_L), latent_mode,
           Float64(comp_Ad), Float64(comp_Ed), Float64(comp_Vd),
           Float64(comp_As), Float64(comp_Es), Float64(comp_Vs),
           Float64(comp_n), Float64(comp_Tcut), Float64(comp_eps_min),
           comp_rule,
           Float64(yield_c_grad), Float64(yield_mu_grad),
           Float64(weak_f), Float64(weak_eps_ref), Float64(weak_heal_time),
           weak_mode)
end

@inline Tbar(z::Float64, H::Float64) = (1.0 - z) + 0.5 * H * z * (1.0 - z)
@inline dTbar(z::Float64, H::Float64) = -1.0 + 0.5 * H * (1.0 - 2.0 * z)

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

# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

"""Wall Nusselt numbers (x-average of -dT/dz at top and bottom)."""
function nu_walls(m::Mesh, rule::GLLRule, theta::Vector{Float64}, pr::Params)
    p = m.p
    _, dLtop = lagrange_eval(rule, 1.0)
    _, dLbot = lagrange_eval(rule, -1.0)
    ez_top = m.nze - 1
    dzfac_top = 2.0 / z_size(m, ez_top)
    dzfac_bot = 2.0 / z_size(m, 0)
    st = 0.0
    sb = 0.0
    wsum = 0.0
    @inbounds for ex in 0:m.nxe-1
        for i in 1:p+1
            ix = ex * p + i
            w = 0.5 * m.hx * rule.weights[i]
            wsum += w
            dtop = 0.0
            dbot = 0.0
            for j in 1:p+1
                gtop = (ez_top * p + j - 1) * m.nvx + ix
                gbot = (j - 1) * m.nvx + ix
                dtop += theta[gtop] * dLtop[j]
                dbot += theta[gbot] * dLbot[j]
            end
            st += w * dtop * dzfac_top
            sb += w * dbot * dzfac_bot
        end
    end
    Nu_top = 1.0 + 0.5 * pr.H - st / wsum
    Nu_bot = 1.0 - 0.5 * pr.H - sb / wsum
    return Nu_top, Nu_bot
end

"""Volume RMS velocity."""
function vrms(m::Mesh, fd::FEData, ux::Vector{Float64}, uz::Vector{Float64})
    p = m.p
    pv = fd.phi_v
    s = 0.0
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            for qy in 1:fd.nq
                for qx in 1:fd.nq
                    uxq = 0.0
                    uzq = 0.0
                    for j in 1:p+1
                        py = pv[j, qy]
                        for i in 1:p+1
                            k = (j - 1) * (p + 1) + i
                            b = pv[i, qx] * py
                            uxq += ux[g[k]] * b
                            uzq += uz[g[k]] * b
                        end
                    end
                    w = fd.gl_w[qx] * fd.gl_w[qy] * m.hx * hze / 4.0
                    s += w * (uxq * uxq + uzq * uzq)
                end
            end
        end
    end
    return sqrt(s / m.Lx)
end

"""Energy Nusselt number 1 + H/2 + <u_z theta>."""
function nu_energy(m::Mesh, fd::FEData, pr::Params, theta::Vector{Float64},
                   ux::Vector{Float64}, uz::Vector{Float64})
    p = m.p
    pv = fd.phi_v
    s = 0.0
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            for qy in 1:fd.nq
                for qx in 1:fd.nq
                    tq = 0.0
                    uzq = 0.0
                    for j in 1:p+1
                        py = pv[j, qy]
                        for i in 1:p+1
                            k = (j - 1) * (p + 1) + i
                            b = pv[i, qx] * py
                            tq += theta[g[k]] * b
                            uzq += uz[g[k]] * b
                        end
                    end
                    w = fd.gl_w[qx] * fd.gl_w[qy] * m.hx * hze / 4.0
                    s += w * uzq * tq
                end
            end
        end
    end
    return 1.0 + 0.5 * pr.H + s / m.Lx
end

"""Volume integral ``int theta div(u) dV`` (the divergence-induced energy-budget
term).  With theta=0 on all walls, the continuous budget is
`Nu_top - Nu_bot - H = int theta div(u) + d/dt int theta`; this diagnostic
isolates the first term."""
function theta_divu_integral(m::Mesh, fd::FEData, theta::Vector{Float64},
                             ux::Vector{Float64}, uz::Vector{Float64})
    p = m.p
    pv = fd.phi_v
    dpv = fd.dphi_v
    s = 0.0
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            for qy in 1:fd.nq
                for qx in 1:fd.nq
                    thq = 0.0
                    div = 0.0
                    for j in 1:p+1
                        for i in 1:p+1
                            k = (j - 1) * (p + 1) + i
                            b = pv[i, qx] * pv[j, qy]
                            thq += theta[g[k]] * b
                            bx = dpv[i, qx] * pv[j, qy] * (2.0 / m.hx)
                            bz = pv[i, qx] * dpv[j, qy] * (2.0 / hze)
                            div += ux[g[k]] * bx + uz[g[k]] * bz
                        end
                    end
                    w = fd.gl_w[qx] * fd.gl_w[qy] * m.hx * hze / 4.0
                    s += w * thq * div
                end
            end
        end
    end
    return s
end

"""Maximum |div u| at the quadrature points (a check on the discrete constraint)."""
function max_divergence(m::Mesh, fd::FEData, ux::Vector{Float64},
                        uz::Vector{Float64})    p = m.p
    pv = fd.phi_v
    dpv = fd.dphi_v
    d = 0.0
    @inbounds for ez in 0:m.nze-1
        hze = z_size(m, ez)
        for ex in 0:m.nxe-1
            g = element_vel_gids(m, ex, ez)
            for qy in 1:fd.nq
                for qx in 1:fd.nq
                    div = 0.0
                    for j in 1:p+1
                        for i in 1:p+1
                            k = (j - 1) * (p + 1) + i
                            bx = dpv[i, qx] * pv[j, qy] * (2.0 / m.hx)
                            bz = pv[i, qx] * dpv[j, qy] * (2.0 / hze)
                            div += ux[g[k]] * bx + uz[g[k]] * bz
                        end
                    end
                    d = max(d, abs(div))
                end
            end
        end
    end
    return d
end
