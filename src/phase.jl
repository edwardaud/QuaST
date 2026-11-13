"""
    PhaseTransition

Depth-based phase transition following ASPECT's `PhaseFunction`:

    Gamma(z,T) = 0.5 * (1 + tanh(deviation / width))
    deviation  = z - (z0 + slope * (T - Tref))

with `z` the height coordinate (0 bottom, 1 top). `slope = dz/dT` is the
Clapeyron slope expressed as a height shift per temperature; `width` is the
transition width in height. `drho` is the density jump (kg/m^3, the same number
ASPECT's `Phase transition density jumps` takes) and `B` the dimensionless
phase-buoyancy number (`drho / (rho0 alpha0 DeltaT)`).

`latent` is the optional **per-transition latent-heat coefficient** in the
nondimensional units of the energy equation (docs/dev/PHYSICS.md §7.8).  It is the
coefficient of `-DGamma/Dt` in

    rho cp DT/Dt = ... - latent * DGamma/Dt          (our operator, eq. 1005)

i.e. **minus** ASPECT's `T dS_A` (`latent heat` heating model, which assembles
`+T dS_A DGamma/Dt`):

    latent = -T_phase * dS_A / (cp dT),   dS_A = gamma * drho / rho_ref^2

see [`phase_latent_coefficient`](@ref) and [`per_phase_latent`](@ref).  A
negative `latent` (positive Clapeyron slope, exothermic transition) *heats* a
descending parcel; a positive one (negative slope, endothermic) cools it.

`latent` is used only when `Params.latent_mode = :per_phase`; the default mode
`:global` ignores it and applies `Params.latent_L` to every transition (the
historical behaviour, bit for bit).  `nothing` means "not set" — `:per_phase`
then throws rather than silently falling back.
"""
struct PhaseTransition
    z0::Float64
    width::Float64
    slope::Float64
    Tref::Float64
    drho::Float64
    B::Float64
    latent::Union{Nothing,Float64}
end

# Six-argument form: the historical constructor, `latent` unset.
PhaseTransition(z0::Real, width::Real, slope::Real, Tref::Real, drho::Real,
                B::Real) =
    PhaseTransition(Float64(z0), Float64(width), Float64(slope), Float64(Tref),
                    Float64(drho), Float64(B), nothing)

# Seven-argument form with `Real` arguments (the generated constructor needs
# `Float64`): converts, then dispatches to it.
PhaseTransition(z0::Real, width::Real, slope::Real, Tref::Real, drho::Real,
                B::Real, latent::Union{Nothing,Real}) =
    PhaseTransition(Float64(z0), Float64(width), Float64(slope), Float64(Tref),
                    Float64(drho), Float64(B),
                    latent === nothing ? nothing : Float64(latent))

"""
    phase_latent_coefficient(pt, T_abs; rho_ref=1.0, cp=1.0, dT=1.0,
                             g=1.0, d=1.0) -> Float64

The latent-heat coefficient of one transition in the nondimensional units of
the energy equation, from ASPECT's entropy change of a phase transition
(`source/material_model/latent_heat.cc`, `entropy_change`):

    dS_A = gamma * drho / rho_ref^2                    [J/(kg K)]

with `gamma` the Clapeyron slope in Pa/K and `drho` the density jump.  ASPECT's
`latent heat` heating model assembles `+rho T dS_A DGamma/Dt`, i.e. its
coefficient of `+DGamma/Dt` is `L_A = T dS_A/(cp dT)`.  Our operator
(`src/semquad.jl`, eq. 1005) writes the term as `-rho L DGamma/Dt`, so

    latent = -L_A = -T_abs * gamma * drho / (rho_ref^2 cp)
           = -T_abs * slope * drho * g * d / (rho_ref * cp * dT)

(the second form uses `gamma = slope * rho_ref * g * d / dT`, docs/dev/PHYSICS.md
§2.4: `slope = dz/dT` is the Clapeyron slope in code units; note that `dT`
cancels between `gamma` and the `1/(cp dT)` of the nondimensionalisation, so the
coefficient carries it only once, through `gamma`).  Equivalently
`latent = -theta_abs * slope * B * Di` with `Di = alpha g d/cp` and
`B = drho/(rho0 alpha dT)`.)

`T_abs` is the **absolute** nondimensional temperature at the phase depth,
`semquad_base_T(pr, pt.z0) + T_surface` where `T_surface` is the surface offset
of the temperature axis (`Params.eta_T0` for the exponential-rheology cases, 0
when `T = 0` is the surface).  It is evaluated once at the phase depth (the
reference temperature), not at the local quadrature-point temperature.

`rho_ref, cp, dT, g, d` are the ASPECT scale set of docs/dev/PHYSICS.md §14.2; all
default to 1, which *is* the code's nondimensional unit system (then `drho`
must be given in units of `rho0`).  `rho_ref` cancels when `drho` is in units
of `rho_ref` (`latent = -T_abs*slope*drho_nd/(cp*dT)`).

Sign: `slope > 0` (exothermic, the 410) gives `latent < 0`, so for a
downwelling (`u_z < 0`, `Gamma_z u_z > 0`) the source `-latent*Gamma_z*u_z` is
positive = heating.  `slope < 0` (endothermic, the 660) gives `latent > 0` =
cooling.  This is ASPECT's physics with the opposite sign of the stored number,
because the two operators differ by the sign of the `DGamma/Dt` term.

Measured against ASPECT with the realistic case's scale set
(`rho_ref = 3300`, `cp = 1000`, `dT = 1000`, `g = 9.81`, `d = 2.9e6`,
`T_surface = 0.5`): the 410 gives `latent = -5.937e-2` and the 660
`+1.306e-1`, i.e. the negatives of ASPECT's `+0.060` / `-0.131` (the 1 % is the
difference between the run's actual phase-depth temperature, 1195.5 K /
1578.2 K, and the rounded 1200 K / 1580 K of the ASPECT table).
"""
@inline function phase_latent_coefficient(pt::PhaseTransition, T_abs::Float64;
                                          rho_ref::Float64=1.0,
                                          cp::Float64=1.0, dT::Float64=1.0,
                                          g::Float64=1.0, d::Float64=1.0)
    gamma = pt.slope * rho_ref * g * d / dT            # Pa/K
    dS_A = gamma * pt.drho / (rho_ref * rho_ref)       # J/(kg K)
    return -T_abs * dS_A / cp
end

"""
    per_phase_latent(pr, phases; T_surface=0.0, rho_ref=1.0, cp=1.0, dT=1.0,
                     g=1.0, d=1.0) -> Vector{PhaseTransition}

A copy of `phases` in which every `latent` coefficient is filled in from
[`phase_latent_coefficient`](@ref), evaluated at each transition's own
reference temperature `semquad_base_T(pr, pt.z0) + T_surface`.  Use with
`Params.latent_mode = :per_phase`.  The scale-set keywords are the ones of
`phase_latent_coefficient` (`docs/dev/PHYSICS.md` §14.2).
"""
function per_phase_latent(pr::Params, phases::Vector{PhaseTransition};
                          T_surface::Float64=0.0,
                          rho_ref::Float64=1.0, cp::Float64=1.0,
                          dT::Float64=1.0, g::Float64=1.0, d::Float64=1.0)
    out = Vector{PhaseTransition}(undef, length(phases))
    for (i, pt) in enumerate(phases)
        T_abs = semquad_base_T(pr, pt.z0) + T_surface
        out[i] = PhaseTransition(pt.z0, pt.width, pt.slope, pt.Tref, pt.drho,
                                 pt.B,
                                 phase_latent_coefficient(pt, T_abs;
                                                          rho_ref=rho_ref,
                                                          cp=cp, dT=dT,
                                                          g=g, d=d))
    end
    return out
end

@inline function phase_deviation(pt::PhaseTransition, z::Float64, T::Float64)
    return (pt.z0 - z) - pt.slope * (T - pt.Tref)
end

"""Phase fraction `Gamma` in [0,1]."""
@inline function phase_gamma(pt::PhaseTransition, z::Float64, T::Float64)
    if pt.width <= 0.0
        return phase_deviation(pt, z, T) > 0.0 ? 1.0 : 0.0
    end
    return 0.5 * (1.0 + tanh(phase_deviation(pt, z, T) / pt.width))
end

"""`dGamma/dT` (analytic)."""
@inline function phase_dgamma_dT(pt::PhaseTransition, z::Float64, T::Float64)
    pt.width <= 0.0 && return 0.0
    s = tanh(phase_deviation(pt, z, T) / pt.width)
    return -0.5 * pt.slope / pt.width * (1.0 - s * s)
end

"""`dGamma/dz` (analytic)."""
@inline function phase_dgamma_dz(pt::PhaseTransition, z::Float64, T::Float64)
    pt.width <= 0.0 && return 0.0
    s = tanh(phase_deviation(pt, z, T) / pt.width)
    return -0.5 / pt.width * (1.0 - s * s)
end

"""
    phase_d2gamma_dT2(pt, z, T)

`d^2 Gamma / dT^2` (analytic), the second derivative the apparent-heat-capacity
Newton tangent needs (`Gamma_T` is `phase_dgamma_dT`):

    Gamma_TT = -slope^2 * tanh(dev/w) * (1 - tanh(dev/w)^2) / w^2

It is `slope * d(Gamma_z)/dT` and vanishes for `width <= 0` (the same switch as
every other phase derivative).
"""
@inline function phase_d2gamma_dT2(pt::PhaseTransition, z::Float64, T::Float64)
    pt.width <= 0.0 && return 0.0
    s = tanh(phase_deviation(pt, z, T) / pt.width)
    return -(pt.slope * pt.slope) * s * (1.0 - s * s) / (pt.width * pt.width)
end

"""
    phase_d2gamma_dzdT(pt, z, T)

`d^2 Gamma / (dz dT)` (analytic) = `d(Gamma_z)/dT`, needed by the latent source
tangent (`Gamma_z` is `phase_dgamma_dz`):

    Gamma_zT = -slope * tanh(dev/w) * (1 - tanh(dev/w)^2) / w^2

With `slope == 0` the phase function is independent of `T`, so the derivative is
exactly zero (`Gamma_z` is then a function of `z` only).
"""
@inline function phase_d2gamma_dzdT(pt::PhaseTransition, z::Float64, T::Float64)
    pt.width <= 0.0 && return 0.0
    pt.slope == 0.0 && return 0.0
    s = tanh(phase_deviation(pt, z, T) / pt.width)
    return -pt.slope * s * (1.0 - s * s) / (pt.width * pt.width)
end

