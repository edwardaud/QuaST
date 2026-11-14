# The Tosi et al. (2015) viscoplastic law.
#
# Kept in its own file, included before `physics.jl`, because `Params` stores a
# `TosiLaw` and Julia needs the type before the struct definition.

"""
    TosiLaw

Tosi et al. (2015) viscoplastic law (their equations 6-8), in **absolute**
units, together with the reference viscosity used to make it nondimensional:

    eta_lin  = exp(-ln(etaT) * T + ln(etaZ) * zdepth)
    eta_pl   = eta_star + sigma_y / sqrt(edot:edot)
    eta      = 2 / (1/eta_lin + 1/eta_pl)
    eta      = clamp(eta, eta_min, eta_max),   edot = 0 -> eta_init
    eta_nd   = eta / eta0

`eps_II` passed to `semquad_eta_tosi` is the second invariant with the
`sqrt(0.5 edot:edot)` convention (as returned by `semquad_strain_rate_at`), so
`sqrt(edot:edot) = sqrt(2) * eps_II`.

`eta0` is the viscosity used to make the law nondimensional.  The benchmark
the paper's equation (7) carries a pressure (depth) term with `etaZ = 10`, which
requires the viscosity to grow downwards, i.e. `z = 1 - y`.  The ASPECT plugin
instead calls `viscolin(etaT, etaZ, T, 0)`, dropping the term; setting
`z_is_depth = false` reproduces the plugin's behaviour (its Case 5a column is in
the paper's Table 2 either way, so this must be an explicit choice).

`eta0` makes the law nondimensional (`eta_nd = eta_abs / eta0`).  The benchmark
law is *already* nondimensional (its values live in [1e-6, 10]), so the default
is `eta0 = 1`.  This was settled empirically: with `eta0 = 0.1` the effective
Rayleigh number drops by a factor 10 and the run never convects (Nu = 1, Vrms
decaying 0.03 -> 0.01), while with `eta0 = 1` it reproduces the earlier structured-grid layer's Case 5a
trajectory at t = 0.136 to within 5-8% (Nu 2.91/3.71 vs 2.70/3.46, Vrms 49.7 vs
52.3).  The ASPECT comment "eta0 is not the same as the Tosi material model
Reference viscosity" is easy to misread; trust the empirical test.
"""
struct TosiLaw
    etaT::Float64
    etaZ::Float64
    eta_star::Float64
    sigma_y::Float64
    eta_init::Float64
    eta0::Float64
    eta_min::Float64
    eta_max::Float64
    z_is_depth::Bool
end

function TosiLaw(; etaT::Real=1.0e5, etaZ::Real=10.0, eta_star::Real=1.0e-3,
                 sigma_y::Real=4.0, eta_init::Real=1.0e-1, eta0::Real=1.0,
                 eta_min::Real=1.0e-6, eta_max::Real=10.0,
                 z_is_depth::Bool=true)
    etaT > 1 || throw(ArgumentError("TosiLaw: etaT must be > 1"))
    etaZ > 0 || throw(ArgumentError("TosiLaw: etaZ must be > 0"))
    eta_star >= 0 || throw(ArgumentError("TosiLaw: eta_star must be >= 0"))
    sigma_y >= 0 || throw(ArgumentError("TosiLaw: sigma_y must be >= 0"))
    eta_init > 0 || throw(ArgumentError("TosiLaw: eta_init must be > 0"))
    eta0 > 0 || throw(ArgumentError("TosiLaw: eta0 must be > 0"))
    0 < eta_min <= eta_max ||
        throw(ArgumentError("TosiLaw: need 0 < eta_min <= eta_max"))
    TosiLaw(Float64(etaT), Float64(etaZ), Float64(eta_star), Float64(sigma_y),
            Float64(eta_init), Float64(eta0), Float64(eta_min), Float64(eta_max),
            z_is_depth)
end

"""
    semquad_eta_tosi(T, eps_II, y, t::TosiLaw) -> (eta_nd, eta_abs)

Tosi viscoplastic viscosity at absolute temperature `T`, strain-rate invariant
`eps_II = sqrt(0.5 edot:edot)` and height `y`.  Returns the nondimensional
viscosity (`eta_abs / eta0`) and the absolute one (after the `[eta_min, eta_max]`
clamp).  `eps_II == 0` returns `eta_init`.
"""
@inline function semquad_eta_tosi(T::Float64, eps_II::Float64, y::Float64,
                                  t::TosiLaw)
    z = t.z_is_depth ? (1.0 - y) : y
    gamma_T = log(t.etaT)
    gamma_z = log(t.etaZ)
    eta_lin = exp(-gamma_T * T + gamma_z * z)
    if t.eta_star == 0.0 && t.sigma_y == 0.0
        # Purely linear law (Case 1/3 of the benchmark): the viscosity does not
        # depend on the strain rate at all, so the eta_init branch must not
        # apply -- otherwise a Stokes solve starting from u = 0 would use
        # eta_init instead of eta_lin(T, z).
        eta = clamp(eta_lin, t.eta_min, t.eta_max)
        return eta / t.eta0, eta
    end
    eps_II <= 0.0 && return t.eta_init / t.eta0, t.eta_init
    # sqrt(edot:edot) = sqrt(2) * eps_II
    eta_pl = t.eta_star + t.sigma_y / (sqrt(2.0) * eps_II)
    eta = 2.0 / (1.0 / eta_lin + 1.0 / eta_pl)
    eta = clamp(eta, t.eta_min, t.eta_max)
    return eta / t.eta0, eta
end

"""
    semquad_deta_tosi(T, eps_II, y, t) -> (deta_deps, deta_dT)

Analytic derivatives of the Tosi viscosity with respect to the strain-rate
invariant `eps_II = sqrt(0.5 edot:edot)` and temperature, **in the same
nondimensional units as `semquad_eta_tosi(...)[1]`** (i.e. divided by `eta0`;
the two differ whenever `eta0 != 1`).

The purely linear law (benchmark Case 1/3: `eta_star = sigma_y = 0`) has no
strain-rate dependence but *does* depend on temperature, `deta/dT =
-gamma_T eta_lin`, so `deta_dT` is not zero there; returning zero made the Tosi
Newton a fixed-point iteration for Case 1 (linear rate ~0.6/step) instead of a
Newton.  Derivatives are zero in the `eta_init` branch and at the
`[eta_min, eta_max]` clamp (consistent with the hard clamp used by
`semquad_eta_tosi`).  `deta_dp` is identically zero for this law: `sigma_y` has
no pressure term.
"""
@inline function semquad_deta_tosi(T::Float64, eps_II::Float64, y::Float64,
                                   t::TosiLaw)
    z = t.z_is_depth ? (1.0 - y) : y
    gamma_T = log(t.etaT)
    gamma_z = log(t.etaZ)
    eta_lin = exp(-gamma_T * T + gamma_z * z)
    if t.eta_star == 0.0 && t.sigma_y == 0.0
        # Purely linear law: eta = clamp(eta_lin).  No strain-rate derivative,
        # but the temperature derivative is the whole point of Case 1/3.
        eta = clamp(eta_lin, t.eta_min, t.eta_max)
        (eta <= t.eta_min || eta >= t.eta_max) && return 0.0, 0.0
        return 0.0, (-gamma_T * eta_lin) / t.eta0
    end
    eps_II <= 0.0 && return 0.0, 0.0
    eta_pl = t.eta_star + t.sigma_y / (sqrt(2.0) * eps_II)
    eta = 2.0 / (1.0 / eta_lin + 1.0 / eta_pl)
    (eta <= t.eta_min || eta >= t.eta_max) && return 0.0, 0.0
    deta_deps = -(eta * eta) / (2.0 * eta_pl * eta_pl) *
                (t.sigma_y / (sqrt(2.0) * eps_II * eps_II))
    deta_dT = -gamma_T * (eta * eta) / (2.0 * eta_lin)
    return deta_deps / t.eta0, deta_dT / t.eta0
end
