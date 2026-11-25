# The viscosity closures the quadtree loop evaluates, and their analytic
# derivatives for the Newton paths.
#
# `semquad_eta_at` is the dispatch every caller uses: it adds the base-state
# temperature to `theta` and evaluates the card's law, with the plastic branch
# when the card enables yielding.  The `semquad_deta_*` functions are the
# derivatives of that closure with respect to the temperature, the pressure and
# the weakening field, and `semquad_dp_deta_T` / `spd_factor` are the SPD
# stabilisation the variable-viscosity assembly needs (a port of ASPECT's
# `Utilities::compute_spd_factor`).

"""
    semquad_eta_at(sq, theta, x, y, pr, leaves)

Viscosity at `(x,y)` used by the quadtree convection loop:

    T = Tbar(y, H) + theta(x,y),   eta = eta_at(T, y, pr).

This supports `:expT` (temperature only) and `:expTz` (temperature + depth) with
exactly the same formula and `eta_min/eta_max` bounds as the uniform assembler.
"""
function semquad_eta_at(sq::SemQuad, theta::Vector{Float64}, x::Float64,
                        y::Float64, pr::Params, leaves::Set{NTuple{3,Int}})
    # Stage 0.5b1: on a mapped geometry the temperature is `base + theta` with
    # the *geometry's* conductive base state (the annulus's radial log/H
    # profile) and the depth argument of the law is the physical height
    # `h = height(g,x,y)`.  The box keeps its historical viscosity row
    # *verbatim*: `Tbar(y,H)` and the depth `y`, even when a reference state is
    # set — the box viscosity sees the Boussinesq base while the
    # buoyancy/energy use the adiabat, and the box
    # path must stay bitwise (G197 pins it).  The branch on the concrete
    # geometry keeps the box arithmetic identical; the mapped branch is only
    # reached on a mapped space (where the driver refuses a reference state).
    g = sq.geom
    if g isa CartesianBox
        T = Tbar(y, pr.H) + _semquad_eval_at(sq, theta, x, y, leaves)
        return eta_at(T, y, pr)[1]
    end
    h = height(g, x, y)
    T = semquad_base_T(pr, g, x, y) + _semquad_eval_at(sq, theta, x, y, leaves)
    return eta_at(T, h, pr)[1]
end

"""
    semquad_eta_at(sq, theta, edot_field, pv, sp, x, y, pr, leaves)

Viscosity at `(x,y)` using a *nodal* strain-rate field (the Picard-averaged one
from `blend_field`).  Used only when `pic_eta_relax < 1`; the pointwise method
below is the default and is bit-identical to the pre-averaging behaviour.
"""
function semquad_eta_at(sq::SemQuad, theta::Vector{Float64},
                        edot_field::Vector{Float64},
                        pv::Vector{Float64}, sp::SemQuad,
                        x::Float64, y::Float64, pr::Params,
                        leaves::Set{NTuple{3,Int}})
    g = sq.geom
    local h, T
    if g isa CartesianBox
        h = y
        T = Tbar(y, pr.H) + _semquad_eval_at(sq, theta, x, y, leaves)
    else
        h = height(g, x, y)
        T = semquad_base_T(pr, g, x, y) +
            _semquad_eval_at(sq, theta, x, y, leaves)
    end
    edot = _semquad_eval_at(sq, edot_field, x, y, leaves)
    p = _semquad_eval_at(sp, pv, x, y, leaves)
    return semquad_visco_plastic_eta(T, h, edot, p, pr)
end

"""
    semquad_eta_at(sq, theta, ux, uz, pv, sp, x, y, pr, leaves)

Viscosity at `(x,y)` including the plastic branch.  `ux/uz/pv` are the velocity
and pressure of the previous Picard iterate (lagged), `sp` the pressure space.
When yielding is inactive this is identical to the 5-argument method.
"""
function semquad_eta_at(sq::SemQuad, theta::Vector{Float64},
                        ux::Vector{Float64}, uz::Vector{Float64},
                        pv::Vector{Float64}, sp::SemQuad,
                        x::Float64, y::Float64, pr::Params,
                        leaves::Set{NTuple{3,Int}})
    g = sq.geom
    local h, T
    if g isa CartesianBox
        h = y
        T = Tbar(y, pr.H) + _semquad_eval_at(sq, theta, x, y, leaves)
    else
        h = height(g, x, y)
        T = semquad_base_T(pr, g, x, y) +
            _semquad_eval_at(sq, theta, x, y, leaves)
    end
    yield_active(pr) || return eta_at(T, h, pr)[1]
    edot = semquad_strain_rate_at(sq, ux, uz, x, y, leaves)
    p = _semquad_eval_at(sp, pv, x, y, leaves)
    return semquad_visco_plastic_eta(T, h, edot, p, pr)
end

"""
    semquad_eta_at(sq, theta, ux, uz, pv, sp, chi, x, y, pr, leaves)
    semquad_eta_at(sq, theta, edot_field, pv, sp, chi, x, y, pr, leaves)

The two yielding methods above with the strain-weakening field `chi`.
`chi === nothing` is exactly the corresponding
weakening-off method (bit for bit), so the driver can pass `nothing` when
`weak_active(pr)` is false and every old call site keeps its behaviour.
"""
function semquad_eta_at(sq::SemQuad, theta::Vector{Float64},
                        ux::Vector{Float64}, uz::Vector{Float64},
                        pv::Vector{Float64}, sp::SemQuad,
                        chi::Union{Nothing,Vector{Float64}},
                        x::Float64, y::Float64, pr::Params,
                        leaves::Set{NTuple{3,Int}})
    chi === nothing && return semquad_eta_at(sq, theta, ux, uz, pv, sp, x, y,
                                             pr, leaves)
    g = sq.geom
    local h, T
    if g isa CartesianBox
        h = y
        T = Tbar(y, pr.H) + _semquad_eval_at(sq, theta, x, y, leaves)
    else
        h = height(g, x, y)
        T = semquad_base_T(pr, g, x, y) +
            _semquad_eval_at(sq, theta, x, y, leaves)
    end
    yield_active(pr) || return eta_at(T, h, pr)[1]
    edot = semquad_strain_rate_at(sq, ux, uz, x, y, leaves)
    p = _semquad_eval_at(sp, pv, x, y, leaves)
    c = _semquad_eval_at(sq, chi, x, y, leaves)
    return semquad_visco_plastic_eta(T, h, edot, p, c, pr)
end

function semquad_eta_at(sq::SemQuad, theta::Vector{Float64},
                        edot_field::Vector{Float64},
                        pv::Vector{Float64}, sp::SemQuad,
                        chi::Union{Nothing,Vector{Float64}},
                        x::Float64, y::Float64, pr::Params,
                        leaves::Set{NTuple{3,Int}})
    chi === nothing && return semquad_eta_at(sq, theta, edot_field, pv, sp, x,
                                             y, pr, leaves)
    g = sq.geom
    local h, T
    if g isa CartesianBox
        h = y
        T = Tbar(y, pr.H) + _semquad_eval_at(sq, theta, x, y, leaves)
    else
        h = height(g, x, y)
        T = semquad_base_T(pr, g, x, y) +
            _semquad_eval_at(sq, theta, x, y, leaves)
    end
    edot = _semquad_eval_at(sq, edot_field, x, y, leaves)
    p = _semquad_eval_at(sp, pv, x, y, leaves)
    c = _semquad_eval_at(sq, chi, x, y, leaves)
    return semquad_visco_plastic_eta(T, h, edot, p, c, pr)
end


"""
    semquad_deta_dp(T, y, edot, p, chi, pr)   /   (T, y, edot, p, pr)

Pressure derivative of the visco-plastic viscosity, `deta/dp`.

Drucker-Prager: `tau_y = min(f(chi)(c(d) cos(phi) + mu(d) p), tau_max)` and
`eta_pl = tau_y/(2 edot)`, with `eta = clip(min(eta_visc, eta_pl))`.  The
derivative is `f mu(d)/(2 edot)` only when the plastic branch is active, the
yield stress is below `tau_max`, and the viscosity is not clipped; otherwise it
is zero.  This is the term that couples the velocity rows to the pressure
column in the Newton Jacobian (a non-symmetric saddle block).
"""
function semquad_deta_dp(T::Float64, y::Float64, edot::Float64, p::Float64,
                         chi::Float64, pr::Params)
    mu = semquad_yield_friction(y, pr)
    mu != 0.0 || return 0.0
    f = weak_active(pr) ? semquad_weak_factor(chi, pr) : 1.0
    eta_v, _ = eta_at(T, y, pr)
    tau_raw = semquad_yield_cohesion(y, pr) * cos(pr.yield_phi) + p * mu
    tau = f * tau_raw
    tau < pr.yield_taumax || return 0.0
    ed = max(edot, pr.yield_eps_min)
    eta_pl = tau / (2.0 * ed)
    eta_pl < eta_v || return 0.0
    eta_b = _damper_blend(eta_v, eta_pl, pr)
    _, clipped = _clip_eta(min(eta_v, eta_b), pr)
    clipped && return 0.0
    # the damper divides by D = 1 + eta_d/eta_v (eta_v does not depend on p)
    return (f * mu / (2.0 * ed)) / (1.0 + pr.yield_damper / eta_v)
end

semquad_deta_dp(T::Float64, y::Float64, edot::Float64, p::Float64,
                pr::Params) = semquad_deta_dp(T, y, edot, p, 0.0, pr)

"""
    semquad_deta_chi(T, y, edot, p, chi, pr)

Derivative of the visco-plastic viscosity with respect to the weakening field,

    d eta / d chi = -(1 - weak_f) * tau_raw / (2 edot) / (1 + eta_d/eta_v)

when the plastic branch is active, the weakened yield stress is below
`tau_max`, the viscosity is not clipped, and `weak_active(pr)`; zero otherwise.
This is the derivative that couples the weakening equation to the momentum
equation in the per-step Newton. Note `tau_raw` may be
negative (a strongly negative dynamic pressure), so the sign is not hard-wired.
"""
function semquad_deta_chi(T::Float64, y::Float64, edot::Float64, p::Float64,
                          chi::Float64, pr::Params)
    weak_active(pr) || return 0.0
    tau_raw = semquad_yield_cohesion(y, pr) * cos(pr.yield_phi) +
              p * semquad_yield_friction(y, pr)
    f = semquad_weak_factor(chi, pr)
    tau = f * tau_raw
    tau < pr.yield_taumax || return 0.0
    eta_v, _ = eta_at(T, y, pr)
    ed = max(edot, pr.yield_eps_min)
    eta_pl = tau / (2.0 * ed)
    eta_pl < eta_v || return 0.0
    eta_b = _damper_blend(eta_v, eta_pl, pr)
    _, clipped = _clip_eta(min(eta_v, eta_b), pr)
    clipped && return 0.0
    return (-(1.0 - pr.weak_f) * tau_raw / (2.0 * ed)) /
           (1.0 + pr.yield_damper / eta_v)
end

"""
    semquad_deta_dp_eps(T, y, edot, p, pr)

Strain-rate derivative of the visco-plastic viscosity, `d eta / d edot`.

In the plastic branch `eta_pl = tau_y(p)/(2 edot)` with `tau_y` independent of
the strain rate, so `d eta_pl / d edot = -eta_pl / edot` (strain-rate
weakening, negative).  The derivative is zero when the viscous branch wins, at
a `edot <= yield_eps_min` (the floor makes the viscosity constant there), and
at the `[eta_min, eta_max]` clip.  Units are the same nondimensional viscosity
units as `semquad_visco_plastic_eta`.

Note for the Newton: in the fully yielded limit (`eta_v >> eta_pl`) this
derivative is `-eta/edot`, which makes the geometric tangent
`Kgeo ~ sum w (deta/dedot)/edot T T'` exactly cancel the viscous stiffness
(`2 eta` vs `2 eta`), i.e. the perfectly plastic tangent is singular.  That is
a property of the yield-stress law, not of the implementation; the DP Newton
therefore offers `dp_tangent=:yielded_zero` (freeze the plastic branch) as
well as the full tangent.
"""
function semquad_deta_dp_eps(T::Float64, y::Float64, edot::Float64, p::Float64,
                              chi::Float64, pr::Params)
    edot <= pr.yield_eps_min && return 0.0
    eta_v, _ = eta_at(T, y, pr)
    tau_y = semquad_yield_stress(p, y, chi, pr)
    eta_pl = tau_y / (2.0 * edot)
    eta_pl < eta_v || return 0.0
    eta_b = _damper_blend(eta_v, eta_pl, pr)
    _, clipped = _clip_eta(min(eta_v, eta_b), pr)
    clipped && return 0.0
    return (-eta_pl / edot) / (1.0 + pr.yield_damper / eta_v)
end

semquad_deta_dp_eps(T::Float64, y::Float64, edot::Float64, p::Float64,
                    pr::Params) = semquad_deta_dp_eps(T, y, edot, p, 0.0, pr)

"""
    semquad_dp_deta_T(T, y, edot, p, chi, pr)   /   (T, y, edot, p, pr)

Temperature derivative of the visco-plastic viscosity, `d eta / d T`: the
viscous branch's `deta_dT_at` when the viscous branch wins, zero in the
plastic branch (the yield stress does not depend on T), and zero at the clip.
"""
function semquad_dp_deta_T(T::Float64, y::Float64, edot::Float64, p::Float64,
                           chi::Float64, pr::Params)
    eta_v, _ = eta_at(T, y, pr)
    tau_y = semquad_yield_stress(p, y, chi, pr)
    eta_pl = tau_y / (2.0 * max(edot, pr.yield_eps_min))
    # plastic branch wins
    if eta_pl < eta_v
        eta_b = _damper_blend(eta_v, eta_pl, pr)
        _, clipped = _clip_eta(min(eta_v, eta_b), pr)
        clipped && return 0.0
        # eta_b = (eta_d + eta_pl)/(1 + eta_d/eta_v): the denominator carries
        # eta_v(T), so the damper gives the yielded branch a T dependence
        (pr.yield_damper > 0.0) || return 0.0
        D = 1.0 + pr.yield_damper / eta_v
        return (pr.yield_damper + eta_pl) * pr.yield_damper *
               deta_dT_at(T, y, pr) / (eta_v^2 * D^2)
    end
    _, clipped = _clip_eta(min(eta_v, eta_pl), pr)
    clipped && return 0.0
    return deta_dT_at(T, y, pr)
end

semquad_dp_deta_T(T::Float64, y::Float64, edot::Float64, p::Float64,
                  pr::Params) = semquad_dp_deta_T(T, y, edot, p, 0.0, pr)

# The SPD stabilisation factor of the visco-plastic Jacobian: ASPECT's
# `Utilities::compute_spd_factor`, used by the variable-viscosity assembly.

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
