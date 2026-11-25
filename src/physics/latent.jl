# ---- per-transition latent-heat coefficients -------
# The energy equation carries the latent heat of every transition as
#
#     (1 + sum_i L_i Gamma_T,i) Dtheta/Dt = lap(theta) - u_z dTbar/dz
#                                           - sum_i L_i Gamma_z,i u_z
#
# `L_i` is `Params.latent_L` (one value for all transitions) in `:global` mode,
# the historical behaviour and still the default, or `PhaseTransition.latent`
# in `:per_phase` mode.  The sign convention is the operator's: `L_i` is the
# coefficient of `-DGamma_i/Dt`, so it is *minus* ASPECT's `T dS_A/(cp dT)`
# (`phase_latent_coefficient`); a negative value (positive Clapeyron slope)
# heats descending material.
"""
    semquad_latent_coeff(pr, pt) -> Float64

The latent coefficient actually applied to transition `pt`: `pr.latent_L` in
`latent_mode = :global` (bitwise the historical value) or `pt.latent` in
`:per_phase`.  Prefer [`semquad_latent_coefficients`](@ref) inside an assembly
loop (it hoists the mode test and the `nothing` check out of the quadrature
loop).
"""
@inline function semquad_latent_coeff(pr::Params, pt::PhaseTransition)
    pr.latent_mode === :per_phase && return _semquad_per_phase_coeff(pt)
    return pr.latent_L
end

@inline function _semquad_per_phase_coeff(pt::PhaseTransition)
    pt.latent === nothing &&
        throw(ArgumentError("latent_mode=:per_phase needs PhaseTransition.latent " *
                            "for every transition (z0=$(pt.z0)); set it with " *
                            "phase_latent_coefficient or per_phase_latent, or " *
                            "use latent_mode=:global with latent_L"))
    return pt.latent::Float64
end

"""
    semquad_latent_active(pr, phases) -> Bool

`true` when the energy equation must carry the latent-heat terms: a non-empty
`phases` list and a non-zero coefficient (`latent_L` in `:global` mode, at
least one per-phase coefficient in `:per_phase` mode).  Throws in `:per_phase`
mode when a transition has no coefficient, so that a misconfigured run fails
loudly instead of silently dropping its latent heat.
"""
@inline function semquad_latent_active(pr::Params,
                                       phases::Vector{PhaseTransition})
    isempty(phases) && return false
    if pr.latent_mode === :per_phase
        any = false
        for pt in phases
            c = _semquad_per_phase_coeff(pt)
            c != 0.0 && (any = true)
        end
        return any
    end
    return pr.latent_L != 0.0
end

"""
    semquad_latent_coefficients(pr, phases) -> Vector{Float64}

`semquad_latent_coeff` for the whole list, materialised once per assembly (the
quadrature loop then indexes a `Vector{Float64}` instead of testing a `Symbol`
per quadrature point).  In `:global` mode every entry is `pr.latent_L`, so the
arithmetic of the historical path is reproduced exactly.
"""
function semquad_latent_coefficients(pr::Params,
                                     phases::Vector{PhaseTransition})
    if pr.latent_mode === :per_phase
        out = Vector{Float64}(undef, length(phases))
        for (i, pt) in enumerate(phases)
            out[i] = _semquad_per_phase_coeff(pt)
        end
        return out
    end
    return fill(pr.latent_L, length(phases))
end
