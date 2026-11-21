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
    # Half-life of the radiogenic heating, in the code's time unit.  `Inf` keeps
    # `H` constant, the historical path bit for bit.  The source decays as
    # `H * 2^(-t/half_life)`, the form ASPECT's `Radioactive decay` subsection
    # uses with per-isotope half-lives.  The conductive base state `Tbar` keeps
    # the initial `H`, so `T = Tbar + theta` stays exact and only the source
    # decays.
    H_half_life::Float64
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
    # Multiplicative factor on the viscosity *after* the law: `eta_new =
    # eta * eta_multiplier(x, z)`.  `nothing` leaves every law untouched, bit for
    # bit.  This is the weak-zone hook: a line or rectangle of reduced viscosity
    # becomes `eta_multiplier = weak_zone_factor(...)`, without reimplementing the
    # law, which is how ASPECT's `weak zone` plugin is used next to a material
    # model rather than instead of it.
    eta_multiplier::Union{Nothing,Function}
    # Multiplicative factor on the *density* wherever the buoyancy and the mass
    # terms use it: `rho_new = rho * rho_multiplier(x, z)` (same signature as
    # `eta_multiplier`).  `nothing` leaves every density path untouched, bit for
    # bit.  This is the density half of the sticky-air hook: a light layer above
    # `z_a` is `rho_multiplier = (x, z) -> z > z_a ? eps : 1.0` combined with a
    # low-viscosity `eta_multiplier`, without reimplementing a material model
    # (docs/dev/PHYSICS.md; ASPECT's sticky-air recipes).  It scales a density
    # that is *already* in a term -- it does not add a compositional density
    # anomaly, so with a Boussinesq/refstate buoyancy `Ra rho(y) alpha(y) theta`
    # it acts on the thermal anomaly only (a temperature-neutral layer carries
    # no buoyancy force); the phase-transition buoyancy
    # (`semquad_phase_buoyancy`), which is its own density anomaly anchored to
    # `rho0`, is not scaled.
    rho_multiplier::Union{Nothing,Function}
    # Maxwell viscoelasticity (ASPECT's `viscoelastic`, docs/dev/PHASE2.md 2a and
    # 2b): the shear modulus `G` scales the elastic response in both the
    # effective-viscosity closure eta_eff = 1/(1/eta + 1/(G*dt)) and the stress
    # history tau^{n+1} = (corotate(tau^n) + 2 G dt ehat)/(1 + dt/tau_M),
    # tau_M = eta/G.  Both use the driver's **actual** step `dt`, which is
    # adaptive (CFL + growth limits); that is consistent, because backward Euler
    # with the actual step converges to the analytic Maxwell curve at first order
    # for any variable step sequence summing to the same time (G251), so a fixed
    # elastic time step is not needed.  ASPECT's `Use fixed elastic time step`,
    # `Fixed elastic time step` and `Use stress averaging` evaluate the elastic
    # response on a chosen timescale instead of the numerical one; they are
    # modelling options for special cases such as seismic cycles, not a
    # consistency requirement, and are deliberately not implemented here (G251).
    # `G <= 0` disables every viscoelastic term and leaves every law bit-identical.
    shear_modulus::Float64
    # Per-wall *tangential* velocity, `(xminus, xplus, yminus, yplus)`; `nothing`
    # falls back to the global `uDx`/`uDz` closures, which is the historical
    # behaviour.  The normal component still comes from `uDx`/`uDz`, so a wall with a
    # prescribed tangential value must also be `:rigid` in `wall_bc`.  This is what
    # makes a lid-driven cavity expressible: the top wall's tangential value no
    # longer forces the x-walls' normal velocity (docs/dev/PHASE2.md 2a cross-check).
    wall_u::NTuple{4,Union{Nothing,Function}}
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
    # Side walls: `:insulating` is the historical natural boundary (zero normal
    # flux), and `:fixed` replaces it with the physical temperatures below.  The
    # two knobs are separate so a run cannot change its side-wall physics by
    # setting a temperature and forgetting the switch.
    side_thermal::Symbol  # :insulating (default) or :fixed
    Twall_left::Float64   # physical wall temperature (x = 0), used by :fixed
    Twall_right::Float64  # physical wall temperature (x = Lx), used by :fixed
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
    # Per-wall velocity condition, in the order (xminus, xplus, yminus, yplus).
    # `:same` means "follow `bc`", which is the default and therefore bit-identical
    # to the historical single-`bc` behaviour; `:free`/`:rigid` override one wall.
    # A mixed tuple is only accepted with `linear=:direct` until the multigrid
    # constraint operator carries the tuple (docs/dev/PHASE1.md 1.3 step 2).
    wall_bc::NTuple{4,Symbol}
end

function Params(; Ra::Real=1.0e4, H::Real=0.0, H_half_life::Real=Inf,
                eta_law::Symbol=:const,
                deta_T::Real=1.0, eta_b::Real=0.0, eta_c::Real=0.0,
                eta_E::Real=0.0, eta_T0::Real=0.0,
                yield_c::Real=Inf, yield_phi::Real=0.0,
                yield_taumax::Real=Inf, yield_eps_min::Real=1.0e-30,
                yield_damper::Real=0.0,
                eta_const::Real=1.0,
                eta_min::Real=-Inf, eta_max::Real=Inf,
                bc::Symbol=:free, supg::Real=0.0, eta_soft::Real=0.0,
                wall_bc::NTuple{4,Symbol}=(:same, :same, :same, :same),
                shear_modulus::Real=0.0,
                wall_u::NTuple{4,Union{Nothing,Function}}=(nothing, nothing, nothing, nothing),
                div_penalty::Real=0.0,
                eta_custom=nothing, eta_multiplier=nothing,
                rho_multiplier=nothing,
                rho_heating::Bool=false,
                rho_cp::Bool=false, tosi=nothing,
                Di::Real=0.0, heating_adiabatic::Bool=false,
                heating_dissipation::Bool=false, refstate=nothing,
                rho_ref_weight::Bool=false,
                compress_p::Bool=false, compress_gamma::Real=1.0,
                Twall_top::Real=0.0, Twall_bot::Real=1.0,
                side_thermal::Symbol=:insulating,
                Twall_left::Real=0.0, Twall_right::Real=0.0,
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
    (isfinite(shear_modulus) && shear_modulus >= 0.0) ||
        throw(ArgumentError("shear_modulus must be finite and >= 0 (0 disables " *
                            "the viscoelastic relaxation)"))
    length(wall_bc) == 4 ||
        throw(ArgumentError("wall_bc must have four entries (xminus, xplus, " *
                            "yminus, yplus)"))
    all(w -> w in (:same, :free, :rigid), wall_bc) ||
        throw(ArgumentError("wall_bc entries must be :same, :free or :rigid"))
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
    Params(Float64(Ra), Float64(H), Float64(H_half_life), eta_law,
           Float64(deta_T),
           Float64(eta_b), Float64(eta_c), Float64(eta_E), Float64(eta_T0),
           Float64(yield_c), Float64(yield_phi), Float64(yield_taumax),
           Float64(yield_eps_min), Float64(yield_damper),
           Float64(eta_const),
           Float64(eta_min), Float64(eta_max), bc, Float64(supg),
           Float64(eta_soft), Float64(div_penalty), eta_custom, eta_multiplier,
           rho_multiplier,
           Float64(shear_modulus), NTuple{4,Union{Nothing,Function}}(wall_u),
           rho_heating, rho_cp, tosi, Float64(Di),
           heating_adiabatic, heating_dissipation, refstate, rho_ref_weight,
           compress_p, Float64(compress_gamma),
           Float64(Twall_top), Float64(Twall_bot), side_thermal,
           Float64(Twall_left), Float64(Twall_right), deviatoric,
           Float64(latent_L), latent_mode,
           Float64(comp_Ad), Float64(comp_Ed), Float64(comp_Vd),
           Float64(comp_As), Float64(comp_Es), Float64(comp_Vs),
           Float64(comp_n), Float64(comp_Tcut), Float64(comp_eps_min),
           comp_rule,
           Float64(yield_c_grad), Float64(yield_mu_grad),
           Float64(weak_f), Float64(weak_eps_ref), Float64(weak_heal_time),
           weak_mode, NTuple{4,Symbol}(wall_bc))
end

@inline Tbar(z::Float64, H::Float64) = (1.0 - z) + 0.5 * H * z * (1.0 - z)
@inline dTbar(z::Float64, H::Float64) = -1.0 + 0.5 * H * (1.0 - 2.0 * z)

