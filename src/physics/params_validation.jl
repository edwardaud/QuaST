# `Params` keyword constructor: defaults and all card validation.  The wall-BC
# rules stay here for now (they duplicate stokes validators; merging them is a
# later behaviour-visible wave); throw order and messages are frozen.
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
                wall_un::NTuple{4,Union{Nothing,Function}}=(nothing, nothing, nothing, nothing),
                wall_t::NTuple{4,Union{Nothing,Function}}=(nothing, nothing, nothing, nothing),
                # BC stage 3: the *normal* traction.  It carries `-p`, so it is
                # only legal on an `:open` wall (free normal velocity) and it
                # forces the derived pressure gauge (no nodal pin).  `nothing`
                # everywhere is the historical card bit for bit.
                wall_tn::NTuple{4,Union{Nothing,Function}}=(nothing, nothing, nothing, nothing),
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
                weak_heal_time::Real=Inf, weak_mode::Symbol=:rate,
                eta_ref::Real=1.0)
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
    # The reference viscosity of the Rayleigh number must be a positive finite
    # number in the law's units: it divides every law viscosity in the flow
    # solve, so a zero or negative value has no physical meaning.
    (isfinite(eta_ref) && eta_ref > 0.0) ||
        throw(ArgumentError("eta_ref must be finite and > 0; it is the viscosity " *
                            "the Rayleigh number is defined with, in the units " *
                            "the material law returns (the historical default " *
                            "is 1.0)"))
    length(wall_bc) == 4 ||
        throw(ArgumentError("wall_bc must have four entries (xminus, xplus, " *
                            "yminus, yplus)"))
    all(w -> w in (:same, :free, :rigid, :open), wall_bc) ||
        throw(ArgumentError("wall_bc entries must be :same, :free, :rigid or " *
                            ":open"))
    # `wall_u` is a *tangential* value, so it exists only on a `:rigid` wall.
    # On a `:free` wall the tangential component is traction-free (natural), and
    # on an `:open` wall there is no Dirichlet component at all; in both cases
    # the value would be silently ignored -- the pre-G258 behaviour, corrected
    # here because a silently ignored boundary value is worse than an error.
    # The condition checked is the *resolved* one, so a `:same` entry follows
    # `bc`; the solver repeats the check for direct library calls (G258 (e)).
    _resolved_wall(i) = wall_bc[i] === :same ? bc : wall_bc[i]
    _wall_names = ("xminus", "xplus", "yminus", "yplus")
    for i in 1:4
        if wall_u[i] !== nothing && _resolved_wall(i) !== :rigid
            throw(ArgumentError("wall_u[$i] (wall $(_wall_names[i])) sets a " *
                                "tangential velocity, but that wall resolves to " *
                                ":$(_resolved_wall(i)): " *
                                (_resolved_wall(i) === :open ?
                                 "an :open wall has no Dirichlet component (the " *
                                 "natural traction condition applies to both " *
                                 "components)" :
                                 "a :free wall's tangential component is " *
                                 "traction-free") *
                                ", so the value would be silently ignored.  Set " *
                                "wall_bc[$i] = :rigid (or `bc = :rigid`) for a " *
                                "wall whose tangential motion you prescribe, or " *
                                "drop wall_u[$i]."))
        end
    end
    # BC stage 1/3: `wall_un` is a *normal velocity* (Dirichlet data), so it only
    # exists where the normal component is Dirichlet: not on an `:open` wall.
    for i in 1:4
        if wall_un[i] !== nothing && _resolved_wall(i) === :open
            throw(ArgumentError("wall_un[$i] (wall $(_wall_names[i])) sets a " *
                                "normal *velocity*, but that wall resolves to " *
                                ":open, which has no Dirichlet component: the " *
                                "natural traction condition applies and the " *
                                "normal velocity is an unknown, so the value " *
                                "would be silently ignored.  Set wall_bc[$i] = " *
                                ":free (or :rigid) for a wall whose normal " *
                                "velocity you prescribe, or use wall_tn[$i] for " *
                                "a normal *traction* on the open wall, or drop " *
                                "wall_un[$i]."))
        end
    end
    # BC stage 2/3: a tangential *traction* is a data term on the free side, so
    # it only exists where the tangential velocity is free: a `:free` or an
    # `:open` wall.  On a `:rigid` wall the tangential component is Dirichlet,
    # so the assembled load would be dropped by the constraint matrix: a silent
    # no-op.  The solver repeats this validation for direct library calls
    # (`_semquad_validate_wall_traction`).
    for i in 1:4
        if wall_t[i] !== nothing &&
           !(_resolved_wall(i) === :free || _resolved_wall(i) === :open)
            throw(ArgumentError("wall_t[$i] (wall $(_wall_names[i])) sets a " *
                                "tangential *traction*, but that wall resolves " *
                                "to :$(_resolved_wall(i)), where the tangential " *
                                "component is Dirichlet.  The boundary load would " *
                                "be assembled onto a dof with no equation and " *
                                "silently dropped.  A tangential traction needs a " *
                                "wall with a free tangential component (a :free " *
                                "or :open wall); set wall_bc[$i] = :free, or " *
                                "drop wall_t[$i]."))
        end
    end
    # BC stage 3: the normal component of the total Cauchy traction carries the
    # pressure (`t_n = -p + 2 eta eps_nn`), so it needs a free normal velocity --
    # the `:open` wall -- and with it the derived pressure gauge.  On a `:free`
    # or `:rigid` wall the normal velocity is Dirichlet and the load would be
    # dropped by the constraint matrix: refuse instead of ignoring.  The solver
    # repeats this validation (`_semquad_validate_wall_tnormal`).
    for i in 1:4
        if wall_tn[i] !== nothing && _resolved_wall(i) !== :open
            throw(ArgumentError("wall_tn[$i] (wall $(_wall_names[i])) sets a " *
                                "*normal traction*, but that wall resolves to " *
                                ":$(_resolved_wall(i)), whose normal component " *
                                "is Dirichlet.  The load would be assembled onto " *
                                "a dof with no equation and silently dropped.  A " *
                                "normal traction carries the pressure and needs a " *
                                "wall with a free normal velocity: set " *
                                "wall_bc[$i] = :open (an :open wall has no " *
                                "Dirichlet component, so its natural condition " *
                                "IS the traction level and the pressure gauge " *
                                "becomes derived), " *
                                "or drop wall_tn[$i]."))
        end
    end
    # BC stage 3: the well-posedness rules of a configuration with `:open` walls.
    # (1) all four walls open is the pure traction problem: the velocity is
    # determined only up to a rigid motion (and the data must satisfy net force
    # and torque compatibility), which this solver does not gauge -- refuse.
    # (2) the rigid-mode rank test.  On this axis-aligned box the exact test of
    # "no rigid mode is admissible" is combinatorial: a rigid translation in x
    # is blocked iff some x-wall keeps a Dirichlet normal component (then u_x is
    # fixed along a whole vertical line), likewise for y; the rotation is blocked
    # by any wall that keeps a Dirichlet normal component (its mode value varies
    # along that wall).  So the configuration needs at least one non-`:open`
    # x-wall AND one non-`:open` y-wall.  The solver repeats it on the actual
    # Dirichlet dof set with a 3x3 Gram matrix of the rigid modes (G262).
    _nopen = count(==(:open), map(_resolved_wall, 1:4))
    if _nopen > 0
        _nopen == 4 &&
            throw(ArgumentError("all four walls are :open.  That is the pure " *
                                "traction problem: with the natural condition " *
                                "sigma.n = 0 on the whole boundary the velocity " *
                                "is determined only up to a rigid motion (and the " *
                                "data must satisfy zero net force and torque " *
                                "compatibility), so the discrete saddle is " *
                                "singular and the pressure is not fixed by the " *
                                "boundary.  Leave at least one x-wall " *
                                "(xminus/xplus) and one y-wall (yminus/yplus) " *
                                "with a Dirichlet normal component, i.e. " *
                                "wall_bc = :free or :rigid on those walls."))
        _xok = any(i -> _resolved_wall(i) !== :open, (1, 2))
        _yok = any(i -> _resolved_wall(i) !== :open, (3, 4))
        if !(_xok && _yok)
            _missing = !_xok ? "x-translation" : "y-translation"
            _pair = !_xok ? "xminus/xplus" : "yminus/yplus"
            throw(ArgumentError("the wall configuration leaves the " *
                                "$_missing unconstrained: both $_pair walls are " *
                                ":open, so no wall imposes a Dirichlet normal " *
                                "component in that direction and the velocity " *
                                "of a solution is determined only up to that " *
                                "rigid translation (the 3x3 rigid-mode Gram " *
                                "test would be " *
                                "singular).  An :open wall is well posed only " *
                                "when the normal velocity is Dirichlet on at " *
                                "least one x-wall and one y-wall: set " *
                                "wall_bc = :free (or :rigid) on one wall of " *
                                "that pair."))
        end
    end
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
    # Latent heat per transition: `:global` is the
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
           NTuple{4,Union{Nothing,Function}}(wall_un),
           NTuple{4,Union{Nothing,Function}}(wall_t),
           NTuple{4,Union{Nothing,Function}}(wall_tn),
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
           weak_mode, NTuple{4,Symbol}(wall_bc), Float64(eta_ref))
end

