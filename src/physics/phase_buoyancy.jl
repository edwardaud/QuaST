# Phase transitions for this solver: buoyancy of the phase anomaly (and, later, the
# latent-heat source).
#
# The phase function and its derivatives are the generic ones in `phase.jl`
# (ported from ASPECT's `PhaseFunction`):
#
#     Gamma(z, T) = 0.5 (1 + tanh(deviation / width)),
#     deviation   = (z0 - z) - slope (T - Tref).
#
# Only the *anomaly* with respect to the base state drives flow (a phase
# transition in the reference adiabat must not create a spurious force), so the
# buoyancy uses
#
#     dGamma = Gamma(y, T) - Gamma(y, Tad(y)),   T = Tad + theta,
#
# and the upward momentum gains  -Ra * B * dGamma  for every transition, i.e. a
# positive `B` (dense material above the transition) gives a downward force when
# the phase fraction exceeds its base value.  This mirrors the earlier structured-grid layer's
# `phase_force` (assembly.jl) without the ALA density factor, which this solver
# applies through `semquad_rho` when a reference state is set.
#
# Cross-check against ASPECT (`source/material_model/latent_heat.cc`): ASPECT
# uses the full anomaly `Delta rho * Gamma(y, T)` without subtracting the base
# state.  The difference is `-Ra * B * Gamma(y, Tad(y))`, a function of y alone,
# i.e. a pressure gradient: it changes the pressure but not the velocity (up to
# the O(h^p) error of representing that gradient in the discrete pressure space).
# The sign convention (positive B = denser with depth) is the same.

"""
    semquad_phase_buoyancy(pr, phases, y, theta) -> Float64

Upward body force (already multiplied by `Ra`) from the phase anomalies at
height `y` and perturbation temperature `theta`.  Zero for an empty `phases`.
"""
@inline function semquad_phase_buoyancy(pr::Params, phases::Vector{PhaseTransition},
                                        y::Float64, theta::Float64)
    isempty(phases) && return 0.0
    Tad = semquad_base_T(pr, y)
    T = Tad + theta
    s = 0.0
    for pt in phases
        dG = phase_gamma(pt, y, T) - phase_gamma(pt, y, Tad)
        s += pr.Ra * pt.B * dG
    end
    return -s
end
