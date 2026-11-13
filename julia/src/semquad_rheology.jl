# Visco-plastic rheology for the quadtree SEM path (route C).
#
# The viscosity law itself (`eta_at`) is shared with route A; this file adds the
# strain-rate dependent plastic branch on top of it:
#
#     tau_y   = min(c cos(phi) + p sin(phi), tau_max)      Drucker-Prager
#     eta_pl  = tau_y / (2 |edot|)
#     eta     = min(eta_visc, eta_pl)                      then eta_min/max clip
#
# Nondimensionalisation (route C, Boussinesq):
#   viscosity  [eta0],  length [H],  time [H^2/kappa],  pressure and stress
#   [eta0 kappa / H^2],  strain rate [kappa / H^2].
# `yield_c` and `yield_taumax` are therefore in units of `eta0 kappa / H^2`, and
# `p` is the *dynamic* pressure of the Stokes solve (top-mean zero).  Using the
# dynamic rather than the total pressure is the standard Boussinesq
# approximation; the hydrostatic part is a function of depth only and is
# deliberately not included here (route C has no reference state yet, see P3).
#
# Depth dependence (docs/dev/PHYSICS.md §8): with the depth d = 1-y the cohesion
# and the friction coefficient are linear profiles,
#
#     c(d)  = yield_c + yield_c_grad*d
#     mu(d) = sin(yield_phi) + yield_mu_grad*d
#     tau_y = min(c(d)*cos(yield_phi) + mu(d)*p, yield_taumax)
#
# and the weakening factor of §9 multiplies the whole uncapped stress:
#
#     f(chi) = 1 - (1-weak_f)*clamp(chi,0,1),    tau_y = min(f*tau_raw, taumax).
#
# With zero gradients and f = 1 every expression below is the historical one
# bit for bit (G150/G162/G163/G168 unchanged).



"""
    yield_active(pr)

True when any Drucker-Prager parameter is set (all defaults are inert).
"""
@inline function yield_active(pr::Params)
    return isfinite(pr.yield_c) || isfinite(pr.yield_taumax) || pr.yield_phi > 0.0
end

"""
    weak_active(pr)

True when the strain-weakening field of docs/dev/PHYSICS.md §9 is switched on: a
positive reference plastic strain, a residual weakening factor below one, and an
active Drucker-Prager yield surface (weakening only scales the yield stress).
"""
@inline function weak_active(pr::Params)
    return pr.weak_eps_ref > 0.0 && pr.weak_f < 1.0 && yield_active(pr)
end

"""
    semquad_weak_factor(chi, pr)

The single source of truth for the strain-weakening factor

    f(chi) = 1 - (1 - weak_f)*clamp(chi, 0, 1)   in [weak_f, 1].

`chi` is clamped so a slightly out-of-range transported field cannot make the
yield stress negative or above its unweakened value.  `weak_f = 1` gives
`f == 1` exactly, which is the weakening-off limit.
"""
@inline function semquad_weak_factor(chi::Float64, pr::Params)
    pr.weak_f >= 1.0 && return 1.0
    return 1.0 - (1.0 - pr.weak_f) * clamp(chi, 0.0, 1.0)
end

"""
    semquad_yield_cohesion(y, pr)

Depth-dependent cohesion profile `c(d) = yield_c + yield_c_grad*d` with
`d = 1-y` (depth measured down from the top wall, `y` is the height).
"""
@inline function semquad_yield_cohesion(y::Float64, pr::Params)
    return pr.yield_c + pr.yield_c_grad * (1.0 - y)
end

"""
    semquad_yield_friction(y, pr)

Depth-dependent friction coefficient (the pressure coefficient of the
Drucker-Prager cone) `mu(d) = sin(yield_phi) + yield_mu_grad*d`, `d = 1-y`.
"""
@inline function semquad_yield_friction(y::Float64, pr::Params)
    return sin(pr.yield_phi) + pr.yield_mu_grad * (1.0 - y)
end

"""
    semquad_yield_stress(p, y, chi, pr)
    semquad_yield_stress(p, y, pr)      # chi = 0, first weakening level
    semquad_yield_stress(p, pr)         # surface value (y = 1), chi = 0

Drucker-Prager yield stress

    tau_y = min( f(chi) * (c(d)*cos(phi) + mu(d)*p), tau_max ),   d = 1-y,

in the nondimensional stress units of route C.  The two lower-arity methods are
convenience forms used by the steady solvers and the older gates: with zero
depth gradients `semquad_yield_stress(p, y, pr)` does not depend on `y` and the
2-argument form is exactly the historical constant-coefficient surface.
"""
@inline function semquad_yield_stress(p::Float64, y::Float64, chi::Float64,
                                      pr::Params)
    tau = semquad_yield_cohesion(y, pr) * cos(pr.yield_phi) +
          p * semquad_yield_friction(y, pr)
    weak_active(pr) && (tau *= semquad_weak_factor(chi, pr))
    return min(tau, pr.yield_taumax)
end

@inline semquad_yield_stress(p::Float64, y::Float64, pr::Params) =
    semquad_yield_stress(p, y, 0.0, pr)

@inline semquad_yield_stress(p::Float64, pr::Params) =
    semquad_yield_stress(p, 1.0, 0.0, pr)

"""
    semquad_strain_rate_at(sq, ux, uz, x, y, leaves)

Second invariant of the strain-rate tensor,
`|edot| = sqrt(0.5 (exx^2 + ezz^2 + 2 exz^2))`, at `(x,y)`; this is the
convention used by ASPECT and by `eta_plastic` on route A, so a pure shear flow
`u = (y, 0)` gives `|edot| = 0.5`.
"""
function semquad_strain_rate_at(sq::SemQuad, ux::Vector{Float64},
                                uz::Vector{Float64}, x::Float64, y::Float64,
                                leaves::Set{NTuple{3,Int}})
    uxx, uxy = _semquad_grad_at(sq, ux, x, y, leaves)
    uzx, uzz = _semquad_grad_at(sq, uz, x, y, leaves)
    exz = 0.5 * (uxy + uzx)
    return sqrt(0.5 * (uxx^2 + uzz^2 + 2.0 * exz^2))
end

"""
    _damper_blend(eta_v, eta_pl, pr)

ASPECT's plastic damper (`use_plastic_damper`, `drucker_prager.cc`):

    eta_pl_damped = (eta_d + eta_pl) / (1 + eta_d / eta_v)

with `eta_d = pr.yield_damper`.  `eta_d = 0` returns `eta_pl` unchanged, so the
historical hard `min(eta_v, eta_pl)` law is bit-identical.  Since
`eta_pl < eta_v` implies `eta_pl_damped < eta_v`, the branch structure is
preserved; the damper only raises the yielded viscosity and, because the
denominator carries `eta_v(T)`, it gives the yielded branch a *temperature*
dependence (which is what regularises the Newton tangent).
"""
@inline function _damper_blend(eta_v::Float64, eta_pl::Float64, pr::Params)
    pr.yield_damper > 0.0 || return eta_pl
    return (pr.yield_damper + eta_pl) / (1.0 + pr.yield_damper / eta_v)
end

"""
    semquad_visco_plastic_eta(T, y, edot, p, chi, pr)
    semquad_visco_plastic_eta(T, y, edot, p, pr)     # chi = 0

Viscosity at absolute temperature `T` (height `y`, depth `d = 1-y`) with the
plastic branch: `min(eta_visc, tau_y/(2 edot))`, then the `eta_min/eta_max`
clip.  With `pr.yield_damper > 0` the plastic branch is first blended by
`_damper_blend`.  `chi` is the weakening field (docs/dev/PHYSICS.md §9); the
4-argument method is the first weakening level and is bit-identical to the
historical constant-coefficient law when the depth gradients are zero.
"""
function semquad_visco_plastic_eta(T::Float64, y::Float64, edot::Float64,
                                   p::Float64, chi::Float64, pr::Params)
    eta_v, _ = eta_at(T, y, pr)
    tau_y = semquad_yield_stress(p, y, chi, pr)
    eta_pl = tau_y / (2.0 * max(edot, pr.yield_eps_min))
    return _clip_eta(min(eta_v, _damper_blend(eta_v, eta_pl, pr)), pr)[1]
end

semquad_visco_plastic_eta(T::Float64, y::Float64, edot::Float64, p::Float64,
                          pr::Params) =
    semquad_visco_plastic_eta(T, y, edot, p, 0.0, pr)

"""
    semquad_plastic_strain_rate(T, y, edot, p, chi, pr)

The plastic strain rate that drives the weakening field: the second invariant of
the total strain rate `edot` when the plastic branch is active (`eta_pl <
eta_v`, i.e. the material is on the yield surface) and zero otherwise.  This is
the accumulation ASPECT's `visco_plastic` uses for its "plastic strain"
weakening mechanism (`rheology/strain_dependent.cc`,
`fill_reaction_outputs`: `delta_e_ii = edot_ii*dt` when `plastic_yielding`).
The alternative "excess" definition `edot - tau_y/(2 eta_v)` is not used: it
vanishes in the fully yielded limit, where ASPECT still accumulates.
"""
function semquad_plastic_strain_rate(T::Float64, y::Float64, edot::Float64,
                                     p::Float64, chi::Float64, pr::Params)
    yield_active(pr) || return 0.0
    eta_v, _ = eta_at(T, y, pr)
    tau_y = semquad_yield_stress(p, y, chi, pr)
    eta_pl = tau_y / (2.0 * max(edot, pr.yield_eps_min))
    return eta_pl < eta_v ? edot : 0.0
end

semquad_plastic_strain_rate(T::Float64, y::Float64, edot::Float64, p::Float64,
                            pr::Params) =
    semquad_plastic_strain_rate(T, y, edot, p, 0.0, pr)

"""
    semquad_plastic_strain_rate_field(sq, sp, qt, pr, theta, ux, uz, pv, chi)

Nodal plastic strain-rate field `eps_pl(x,y)` of the state `(theta, ux, uz, pv,
chi)` on the velocity/theta mesh (`sq`), from `semquad_plastic_strain_rate` at
every node; `sp` is the pressure space of `pv`.  Used by the diagnostics and by
the gates; the driver evaluates the rate at the quadrature points of the
weakening assembly instead.
"""
function semquad_plastic_strain_rate_field(sq::SemQuad, sp::SemQuad,
                                           qt::Quadtree, pr::Params,
                                           theta::Vector{Float64},
                                           ux::Vector{Float64},
                                           uz::Vector{Float64},
                                           pv::Vector{Float64},
                                           chi::Vector{Float64})
    leaves = Set(qt.leaves)
    out = zeros(Float64, sq.N)
    for (g, (x, y)) in enumerate(sq.coords)
        T = Tbar(y, pr.H) + _semquad_eval_at(sq, theta, x, y, leaves)
        ed = semquad_strain_rate_at(sq, ux, uz, x, y, leaves)
        p = _semquad_eval_at(sp, pv, x, y, leaves)
        c = _semquad_eval_at(sq, chi, x, y, leaves)
        out[g] = semquad_plastic_strain_rate(T, y, ed, p, c, pr)
    end
    return out
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
    T = Tbar(y, pr.H) + _semquad_eval_at(sq, theta, x, y, leaves)
    edot = _semquad_eval_at(sq, edot_field, x, y, leaves)
    p = _semquad_eval_at(sp, pv, x, y, leaves)
    return semquad_visco_plastic_eta(T, y, edot, p, pr)
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
    T = Tbar(y, pr.H) + _semquad_eval_at(sq, theta, x, y, leaves)
    yield_active(pr) || return eta_at(T, y, pr)[1]
    edot = semquad_strain_rate_at(sq, ux, uz, x, y, leaves)
    p = _semquad_eval_at(sp, pv, x, y, leaves)
    return semquad_visco_plastic_eta(T, y, edot, p, pr)
end

"""
    semquad_eta_at(sq, theta, ux, uz, pv, sp, chi, x, y, pr, leaves)
    semquad_eta_at(sq, theta, edot_field, pv, sp, chi, x, y, pr, leaves)

The two yielding methods above with the strain-weakening field `chi`
(docs/dev/PHYSICS.md §9).  `chi === nothing` is exactly the corresponding
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
    T = Tbar(y, pr.H) + _semquad_eval_at(sq, theta, x, y, leaves)
    yield_active(pr) || return eta_at(T, y, pr)[1]
    edot = semquad_strain_rate_at(sq, ux, uz, x, y, leaves)
    p = _semquad_eval_at(sp, pv, x, y, leaves)
    c = _semquad_eval_at(sq, chi, x, y, leaves)
    return semquad_visco_plastic_eta(T, y, edot, p, c, pr)
end

function semquad_eta_at(sq::SemQuad, theta::Vector{Float64},
                        edot_field::Vector{Float64},
                        pv::Vector{Float64}, sp::SemQuad,
                        chi::Union{Nothing,Vector{Float64}},
                        x::Float64, y::Float64, pr::Params,
                        leaves::Set{NTuple{3,Int}})
    chi === nothing && return semquad_eta_at(sq, theta, edot_field, pv, sp, x,
                                             y, pr, leaves)
    T = Tbar(y, pr.H) + _semquad_eval_at(sq, theta, x, y, leaves)
    edot = _semquad_eval_at(sq, edot_field, x, y, leaves)
    p = _semquad_eval_at(sp, pv, x, y, leaves)
    c = _semquad_eval_at(sq, chi, x, y, leaves)
    return semquad_visco_plastic_eta(T, y, edot, p, c, pr)
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
equation in the per-step Newton (docs/dev/PHYSICS.md §9.4).  Note `tau_raw` may be
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
