"""
    CreepParams

Dislocation-creep parameters in SI units:
`A` prefactor (Pa^-n s^-1), `n` stress exponent, `E` activation energy (J/mol),
`V` activation volume (m^3/mol).
"""
struct CreepParams
    A::Float64
    n::Float64
    E::Float64
    V::Float64
end

const GAS_CONSTANT = 8.314462618

"""
    eta_dislocation(eps, p, T, cp; emin=1e-30)

ASPECT's dislocation-creep viscosity (rheology/dislocation_creep.cc):

    eta = 0.5 * A^(-1/n) * eps^((1-n)/n) * exp((E + p V)/(n R T))
"""
@inline function eta_dislocation(eps::Float64, p::Float64, T::Float64,
                                 cp::CreepParams; emin::Float64=1.0e-30)
    e = max(eps, emin)
    return 0.5 * cp.A^(-1.0/cp.n) *
           exp((cp.E + p * cp.V) / (cp.n * GAS_CONSTANT * T)) *
           e^((1.0 - cp.n) / cp.n)
end

"""
    YieldParams

Drucker-Prager parameters: cohesion `c` (Pa), friction angle `phi` (rad),
maximum yield stress `taumax` (Pa).
"""
struct YieldParams
    c::Float64
    phi::Float64
    taumax::Float64
end

"""
    yield_stress(p, yp)

2-D Drucker-Prager yield stress (ASPECT rheology/drucker_prager.cc):

    tau_y = min(c cos(phi) + p sin(phi), taumax)
"""
@inline function yield_stress(p::Float64, yp::YieldParams)
    return min(yp.c * cos(yp.phi) + p * sin(yp.phi), yp.taumax)
end

"""
    eta_plastic(eps, p, yp; emin=1e-30)

Plastic viscosity `tau_y / (2 eps)`.
"""
@inline function eta_plastic(eps::Float64, p::Float64, yp::YieldParams;
                             emin::Float64=1.0e-30)
    return yield_stress(p, yp) / (2.0 * max(eps, emin))
end

"""
    eta_visco_plastic(eps, p, T, cp, yp; etamin=1e18, etamax=1e28)

Effective viscosity: the dislocation viscosity, replaced by the plastic
viscosity once the viscous stress reaches the yield stress, then bounded by
`[etamin, etamax]`.
"""
@inline function eta_visco_plastic(eps::Float64, p::Float64, T::Float64,
                                   cp::CreepParams, yp::YieldParams;
                                   etamin::Float64=1.0e18,
                                   etamax::Float64=1.0e28)
    eta = eta_dislocation(eps, p, T, cp)
    if 2.0 * eta * eps >= yield_stress(p, yp)
        eta = eta_plastic(eps, p, yp)
    end
    return clamp(eta, etamin, etamax)
end

"""
    RheologyModel

Complete visco-plastic rheology: dislocation creep, Drucker-Prager yield,
reference/minimum strain rates and viscosity bounds.
"""
struct RheologyModel
    creep::CreepParams
    yield::YieldParams
    ref_eps::Float64
    min_eps::Float64
    etamin::Float64
    etamax::Float64
    Tsurf::Float64
    dT::Float64
    p0::Float64
    eps_scale::Float64
    eta0::Float64
end

"""
    eta_rheology_nondim(rh, eps_nd, p_nd, T_nd)

Map nondimensional `eps, p, T` to SI, evaluate the visco-plastic viscosity and
return it normalised by `rh.eta0`.
"""
@inline function eta_rheology_nondim(rh::RheologyModel, eps_nd::Float64,
                                     p_nd::Float64, T_nd::Float64)
    eps = eps_nd * rh.eps_scale
    p = p_nd * rh.p0
    T = rh.Tsurf + rh.dT * T_nd
    return eta_visco_plastic(eps, p, T, rh.creep, rh.yield;
                             etamin=rh.etamin, etamax=rh.etamax) / rh.eta0
end


"""
    deta_deps_dislocation(eps, p, T, cp)

Derivative of the dislocation viscosity with respect to `eps`:

    d eta / d eps = eta * (1-n)/(n eps)
"""
@inline function deta_deps_dislocation(eps::Float64, p::Float64, T::Float64,
                                       cp::CreepParams; emin::Float64=1.0e-30)
    e = max(eps, emin)
    return eta_dislocation(e, p, T, cp) * (1.0 - cp.n) / (cp.n * e)
end

"""
    eta_plastic_smooth(eps, p, yp; eps0)

Smooth plastic viscosity `tau_y / (2 sqrt(eps^2 + eps0^2))`.
"""
@inline function eta_plastic_smooth(eps::Float64, p::Float64, yp::YieldParams;
                                    eps0::Float64=1.0e-20)
    return yield_stress(p, yp) / (2.0 * sqrt(eps * eps + eps0 * eps0))
end

"""
    deta_deps_plastic_smooth(eps, p, yp; eps0)

Derivative of `eta_plastic_smooth` with respect to `eps`.
"""
@inline function deta_deps_plastic_smooth(eps::Float64, p::Float64, yp::YieldParams;
                                          eps0::Float64=1.0e-20)
    s = sqrt(eps * eps + eps0 * eps0)
    return -yield_stress(p, yp) * eps / (2.0 * s * s * s)
end

"Unclamped dislocation/plastic viscosity (raw)."
@inline function eta_visco_plastic_raw(eps::Float64, p::Float64, T::Float64,
                                       cp::CreepParams, yp::YieldParams)
    eta = eta_dislocation(eps, p, T, cp)
    if 2.0 * eta * eps >= yield_stress(p, yp)
        eta = eta_plastic(eps, p, yp)
    end
    return eta
end

"softplus_k(t,k) = log(1+exp(k t))/k, stable for large |t|."
@inline function softplus_k(t::Float64, k::Float64)
    k * t > 30.0 && return t
    return log1p(exp(k * t)) / k
end

"Logistic sigmoid with sharpness k."
@inline function sigmoid_k(t::Float64, k::Float64)
    k * t > 0.0 && return 1.0 / (1.0 + exp(-k * t))
    e = exp(k * t)
    return e / (1.0 + e)
end

"""
    smooth_cap_log(eta, emin, emax, k)

Smooth clamp of `eta` in log space:

    y = l + softplus_k(x-l,k) - softplus_k(x-u,k)

Returns `(exp(y), dy/dx)`.
"""
@inline function smooth_cap_log(eta::Float64, emin::Float64, emax::Float64,
                                k::Float64)
    x = log(eta); l = log(emin); u = log(emax)
    y = l + softplus_k(x - l, k) - softplus_k(x - u, k)
    f = sigmoid_k(x - l, k) - sigmoid_k(x - u, k)
    return exp(y), f
end

"""
    eta_visco_plastic_smooth(eps, p, T, cp, yp; eps0=1e-12, pnorm=10)

Smooth visco-plastic viscosity: p-norm soft minimum of dislocation and
smoothed Drucker-Prager plastic viscosity, with analytic derivative.

Returns `(eta, deta_deps)`.
"""
function eta_visco_plastic_smooth(eps::Float64, p::Float64, T::Float64,
                                  cp::CreepParams, yp::YieldParams;
                                  eps0::Float64=1e-20, pnorm::Float64=10.0)
    e = max(eps, 1e-30)
    ed = eta_dislocation(e, p, T, cp)
    ep = eta_plastic_smooth(e, p, yp; eps0=eps0)
    ded = deta_deps_dislocation(e, p, T, cp)
    dep = deta_deps_plastic_smooth(e, p, yp; eps0=eps0)
    wd = ed^(-pnorm)
    wp = ep^(-pnorm)
    eta = (wd + wp)^(-1.0 / pnorm)
    deta = eta^(pnorm + 1.0) * (wd / ed * ded + wp / ep * dep)
    return eta, deta
end


"Numerically stable smooth visco-plastic viscosity and derivative (log-sum-exp)."
function eta_vp_smooth_stable(eps::Float64, p::Float64, T::Float64,
                              cp::CreepParams, yp::YieldParams;
                              eps0::Float64=1e-20, pnorm::Float64=10.0)
    e = max(eps, 1e-30)
    ed = eta_dislocation(e, p, T, cp)
    ep = eta_plastic_smooth(e, p, yp; eps0=eps0)
    ded = deta_deps_dislocation(e, p, T, cp)
    dep = deta_deps_plastic_smooth(e, p, yp; eps0=eps0)
    la = -pnorm * log(ed)
    lb = -pnorm * log(ep)
    m = max(la, lb)
    lse = m + log(exp(la - m) + exp(lb - m))
    eta = exp(-lse / pnorm)
    delta = log(ep) - log(ed)
    if pnorm * delta > 700.0
        deta = dep
    elseif pnorm * delta < -700.0
        deta = ded
    else
        t = exp(-pnorm * delta)
        deta = eta * (ded / ed + t * dep / ep) / (1.0 + t)
    end
    return eta, deta
end

"""
    spd_factor(eta, deta, edot, exx, ezz, exz, dxd, dzd, dexz; safety=0.9)

Port of ASPECT `Utilities::compute_spd_factor`: returns alpha in (0,1] that
scales the geometric Jacobian term so the effective viscous modulus stays SPD.
`deta` is d eta / d edot; `(exx,ezz,exz)` is the full strain rate,
`(dxd,dzd,dexz)` its deviator.
"""
function spd_factor(eta::Float64, deta::Float64, edot::Float64,
                    exx::Float64, ezz::Float64, exz::Float64,
                    dxd::Float64, dzd::Float64, dexz::Float64;
                    safety::Float64=0.9)
    (edot <= 1e-30 || abs(deta) <= 1e-30) && return 1.0
    ee = exx^2 + ezz^2 + 2.0 * exz^2
    dd = dxd^2 + dzd^2 + 2.0 * dexz^2
    (ee <= 1e-30 || dd <= 1e-30) && return 1.0
    bnorm = abs(deta) / edot * sqrt(dd)
    bdot = deta / edot * (dxd * exx + dzd * ezz + 2.0 * dexz * exz)
    norm_ab = sqrt(ee) * bnorm
    norm_ab <= 1e-30 && return 1.0
    one_minus = 1.0 - bdot / norm_ab
    denom = one_minus^2 * norm_ab
    denom <= safety * 2.0 * eta && return 1.0
    return max(0.0, safety * (2.0 * eta) / denom)
end

"""
    TosiParams

Tosi et al. (2015) viscoplastic law (ASPECT benchmark `tosi_et_al_2015_gcubed`):

    eta_lin   = exp(-ln(etaT) * T + ln(etaZ) * z)
    eta_plast = eta_star + sigma_y / eps_II
    eta       = 2 / (1/eta_lin + 1/eta_plast)   (harmonic mean)
    eta       = clamp(eta, eta_min, eta_max)
    eps == 0  -> eta = eta_init

`use_z=false` reproduces the ASPECT plugin, which calls
`viscolin(eta_T, eta_Z, T, 0)` and therefore drops the etaZ term.
"""
struct TosiParams
    etaT::Float64
    etaZ::Float64
    eta_star::Float64
    sigma_y::Float64
    eta_min::Float64
    eta_max::Float64
    eta_init::Float64
    eta0::Float64
    use_z::Bool
end

"""
    eta_tosi(T, eps, z, tp) -> (eta, deta_deps)
"""
function eta_tosi(T::Float64, eps::Float64, z::Float64, tp::TosiParams)
    eps <= 1e-30 && return (tp.eta_init / tp.eta0, 0.0)
    eta_lin = exp(-log(tp.etaT) * T + (tp.use_z ? log(tp.etaZ) * z : 0.0))
    if tp.eta_star == 0.0 && tp.sigma_y == 0.0
        eta = eta_lin
        deta = 0.0
    else
        ep = sqrt(2.0) * eps
        eta_p = tp.eta_star + tp.sigma_y / ep
        eta = 2.0 / (1.0 / eta_lin + 1.0 / eta_p)
        deta_p = -tp.sigma_y / (sqrt(2.0) * eps * eps)
        deta = eta * eta / (2.0 * eta_p * eta_p) * deta_p
    end
    if eta < tp.eta_min
        return (tp.eta_min / tp.eta0, 0.0)
    elseif eta > tp.eta_max
        return (tp.eta_max / tp.eta0, 0.0)
    end
    return (eta / tp.eta0, deta / tp.eta0)
end


