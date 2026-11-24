# The plastic (Drucker-Prager) branch of the visco-plastic rheology and the
# strain-rate helpers it needs.  The yielding control functions read the card,
# the plastic strain rate and the visco-plastic viscosity combine the yield
# stress with the viscous law, and `semquad_strain_rate_at` is the second
# invariant of the strain-rate tensor the branch is evaluated at.
#
#     tau_y   = min(c cos(phi) + p sin(phi), tau_max)      Drucker-Prager
#     eta_pl  = tau_y / (2 |edot|)
#     eta     = min(eta_visc, eta_pl)                      then eta_min/max clip
#
# `yield_c` and `yield_taumax` are in units of `eta0 kappa / H^2` and `p` is the
# *dynamic* pressure of the Stokes solve (top-mean zero).  The depth profiles of
# `docs/dev/PHYSICS.md` section 8 and the weakening factor of section 9 enter
# through `semquad_yield_stress`.  With zero gradients and `f = 1` every
# expression is the historical one bit for bit.

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

in the nondimensional stress units of this solver.  The two lower-arity methods are
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
convention used by ASPECT and by `eta_plastic` on the earlier structured-grid layer, so a pure shear flow
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
    g = sq.geom
    out = zeros(Float64, sq.N)
    for (gid, (x, y)) in enumerate(sq.coords)
        # Box: the historical viscosity row (`Tbar(y,H)`, depth `y`), even with
        # a reference state (docs/dev/PHYSICS.md §3.3, frozen box behaviour).
        # Mapped (stage 0.5b1): the geometry's base state and `height(g,x,y)`.
        local h, T
        if g isa CartesianBox
            h = y
            T = Tbar(y, pr.H) + _semquad_eval_at(sq, theta, x, y, leaves)
        else
            h = height(g, x, y)
            T = semquad_base_T(pr, g, x, y) +
                _semquad_eval_at(sq, theta, x, y, leaves)
        end
        ed = semquad_strain_rate_at(sq, ux, uz, x, y, leaves)
        p = _semquad_eval_at(sp, pv, x, y, leaves)
        c = _semquad_eval_at(sq, chi, x, y, leaves)
        out[gid] = semquad_plastic_strain_rate(T, h, ed, p, c, pr)
    end
    return out
end
