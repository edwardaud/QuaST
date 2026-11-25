# Stage 0.5b1: the geometry-dependent conductive base state (the 2-D polar annulus)
#
# This file is included after the geometry layer (`core/geometry.jl`), because
# its methods dispatch on the concrete geometry types (`CartesianBox`,
# `AffineBox`, `CurvedBox`, `PolarAnnulus`) and on `Params`; the scalar base
# state (`semquad_base_T(pr, z)`, `Tbar`, `_mapped_base_curvature`) is defined
# in `physics/refstate.jl` and this file only adds the geometry dispatch and the
# annulus radial log/H solution.
#
# `semquad_base_T(pr, z)` is the Cartesian base state: the internally heated
# conduction profile `Tbar(z,H)` (or the reference adiabat).  The 2-D polar
# annulus needs the *radial* conduction solution instead.  With `r = R0 + h`
# (the annulus height `h = r - R0`; the physics closures take the nondimensional
# height in `[0,1]`, i.e. the gap is the length unit, so `R1 - R0 = 1`), uniform
# volumetric heating `H`, `T(R0) = 1`, `T(R1) = 0` and `k = 1`:
#
#     d2T/dr2 + (1/r) dT/dr = -H          (the 2-D radial Laplacian)
#
#     Tbar(r)   = (1 - H (R1^2-R0^2)/4) ln(R1/r)/ln(R1/R0)
#                 + (H/4) (R1^2 - r^2)
#     Tbar'(r)  = -(1 - H (R1^2-R0^2)/4) / (r ln(R1/R0)) - H r/2
#     Tbar''(r) =  (1 - H (R1^2-R0^2)/4) / (r^2 ln(R1/R0)) - H/2
#
# (ANNULUS_REFS §5.4/§6.5).  `H = 0` is the published logarithmic profile
# `ln(R1/r)/ln(R1/R0)`, `dTbar/dr = -1/(r ln(R1/R0)) = 1/(r ln f)` with
# `f = R0/R1` (note `ln(R1/R0) = |ln f| > 0`), `d2Tbar/dr2 = 1/(r^2 ln(R1/R0))`.
# The profile satisfies `Tbar'' + Tbar'/r = -H` *identically*, so the `+H`
# volumetric source is absorbed in the base state exactly as the box's
# `Tbar(z,H)` absorbs it; the first line is written as the harmonic part
# (`1 - H(R1^2-R0^2)/4` is the harmonic part's temperature drop) plus the
# particular solution `(H/4)(R1^2-r^2)`, which keeps both wall values exact.
#
# The accessors below are the geometry dispatch the physics wiring uses:
#
#   * `semquad_base_T(pr, g, x, y)` and `semquad_base_gradient(pr, g, x, y)`:
#     the base temperature and the derivative along the physical height
#     `h = height(g,x,y)` (`dTbar/dh = dTbar/dr` on the annulus).  Every mapped
#     physics body replaces `semquad_base_T(pr, h)` / `semquad_base_gradient(pr,
#     h)` by these, and the box/validation geometries choose the *same*
#     quadratic-in-`h` state as before, so the box path evaluates the identical
#     `Tbar(y,H)` expression and the stage-0 mapped gates see identical values.
#   * `semquad_base_wall_refs(pr, g) -> (bottom_ref, top_ref)`: the wall base
#     gradients `-dTbar/dh` at `h = 0` / `h = 1` that the Nusselt normalisation
#     divides by (`semquad_nu_walls`); the annulus evaluates them at the exact
#     radii `R0`/`R1`, because the benchmark normalisation is per wall:
#     `Nu_wall = ⟨-∂T/∂r⟩_wall / (-dTbar/dr|_{r_wall})`, which for `H = 0`
#     equals the ANNULUS_REFS multiplier form
#     `Nu_wall = r_wall |ln f| ⟨-∂T/∂r⟩ / ΔT` (its §6.2/§6.3).
#
# Nothing in the historical base state changes: the box bodies keep
# `Tbar(y,H)` and `dTbar(y,H)` verbatim (the nine state-dump hashes), and the
# two-argument methods are untouched.
#
# `_mapped_base_curvature` for the annulus is the `lap(Tbar) + H` source of the
# mapped energy assembly.  The H-inclusive log profile is the exact steady
# radial conduction solution, so `lap(Tbar) = -H` and the source is *exactly
# zero* — no `H*(1-|grad h|^2)` or curvature term survives (unlike the generic
# mapped formula, which is only algebraically zero on the annulus and would
# leave round-off).  This is why the H-inclusive annulus base state is a fixed
# point of the energy step.

"""
    _annulus_ln_ratio(g) -> Float64

`ln(R1/R0) > 0` (the `|ln f|`), evaluated from
the geometry's two radii.
"""
@inline _annulus_ln_ratio(g::PolarAnnulus) = log(g.R1 / g.R0)

"""
    _annulus_Tbar(pr, g, r) -> Float64

Radial conduction base state at radius `r`; `H = 0` gives
`ln(R1/r)/ln(R1/R0)`, and the H-inclusive form is the solution of
`T'' + T'/r = -H` with `T(R0) = 1`, `T(R1) = 0` (see the section header).
"""
@inline function _annulus_Tbar(pr::Params, g::PolarAnnulus, r::Float64)
    H = pr.H
    lnr = _annulus_ln_ratio(g)
    drop = 1.0 - 0.25 * H * (g.R1 * g.R1 - g.R0 * g.R0)
    return drop * (log(g.R1 / r) / lnr) + 0.25 * H * (g.R1 * g.R1 - r * r)
end

"""
    _annulus_dTbar_dr(pr, g, r) -> Float64

`dTbar/dr` of [`_annulus_Tbar`](@ref); at `H = 0` this is `1/(r ln f) < 0`.
"""
@inline function _annulus_dTbar_dr(pr::Params, g::PolarAnnulus, r::Float64)
    H = pr.H
    lnr = _annulus_ln_ratio(g)
    drop = 1.0 - 0.25 * H * (g.R1 * g.R1 - g.R0 * g.R0)
    return -drop / (r * lnr) - 0.5 * H * r
end

# --- the geometry-dispatch accessors ---------------------------------------
# One method per concrete geometry: the box, the two validation maps and the
# annulus.  The box values are the historical `Tbar(y,H)`/`dTbar(y,H)`; the
# validation geometries keep the cut-3/4 `height` convention.

@inline function semquad_base_T(pr, ::CartesianBox, x::Float64, y::Float64)
    return semquad_base_T(pr, y)
end
@inline function semquad_base_T(pr, g::AffineBox, x::Float64, y::Float64)
    return semquad_base_T(pr, height(g, x, y))
end
@inline function semquad_base_T(pr, g::CurvedBox, x::Float64, y::Float64)
    return semquad_base_T(pr, height(g, x, y))
end
@inline function semquad_base_T(pr, g::PolarAnnulus, x::Float64, y::Float64)
    return _annulus_Tbar(pr, g, hypot(x, y))
end

@inline function semquad_base_gradient(pr, ::CartesianBox, x::Float64,
                                       y::Float64)
    return semquad_base_gradient(pr, y)
end
@inline function semquad_base_gradient(pr, g::AffineBox, x::Float64,
                                       y::Float64)
    return semquad_base_gradient(pr, height(g, x, y))
end
@inline function semquad_base_gradient(pr, g::CurvedBox, x::Float64,
                                       y::Float64)
    return semquad_base_gradient(pr, height(g, x, y))
end
@inline function semquad_base_gradient(pr, g::PolarAnnulus, x::Float64,
                                       y::Float64)
    return _annulus_dTbar_dr(pr, g, hypot(x, y))
end

"""
    semquad_base_wall_refs(pr, g) -> (bottom_ref, top_ref)

The wall base gradients `-dTbar/dh` at the physical bottom/inner (`h = 0`) and
top/outer (`h = 1`) walls, i.e. the conductive reference flux the wall Nusselt
pair is normalised by.  Generic mapped/box method: the quadratic-in-`h` base
state evaluated at `h = 0`/`1`.  The annulus evaluates the log/H profile at the
exact radii `R0`/`R1` (where `-dTbar/dr = 1/(R0 |ln f|)` and `1/(R1 |ln f|)` at
`H = 0`, recovering the ANNULUS_REFS multipliers `r_wall |ln f| / ΔT` once
inverted).  The caller (`semquad_nu_walls`) owns the sign/scale convention.
"""
@inline function semquad_base_wall_refs(pr, g::AbstractGeometry)
    return (-semquad_base_gradient(pr, 0.0), -semquad_base_gradient(pr, 1.0))
end
@inline function semquad_base_wall_refs(pr, g::PolarAnnulus)
    return (-_annulus_dTbar_dr(pr, g, g.R0),
            -_annulus_dTbar_dr(pr, g, g.R1))
end

# `lap(Tbar) + H` for the annulus: the exact steady radial conduction state has
# `lap(Tbar) = -H` identically (see the section header), so the absorbed source
# is exactly zero, not the round-off of the generic `Tbar''|grad h|^2 + Tbar'
# lap h` formula.  A reference state is not wired for a mapped geometry (the
# driver refuses it); refusing here too keeps a direct assembly call honest
# instead of silently dropping the adiabat's curvature.
@inline function _mapped_base_curvature(pr, g::PolarAnnulus, x, y)
    pr.refstate === nothing ||
        throw(ArgumentError("the annulus base state does not support " *
                            "Params.refstate: the radial log/conductive " *
                            "profile is the Boussinesq base state (the " *
                            "reference-state adiabat is box-only in stage " *
                            "0.5b1)"))
    return 0.0
end

# ---------------------------------------------------------------------------
# Stage 0.5b2: the documented annular initial conditions
# ---------------------------------------------------------------------------
# The benchmarks' initial conditions are **total temperatures** `T_ic(x,y)`
# (ANNULUS_REFS §4.6), linear in `r` plus a 4-fold angular perturbation that
# vanishes on both circles.  The solver's variable is the perturbation
# `theta = T - Tbar(r)`, so the driver converts `T_ic` through the annulus base
# state (identical zero at the walls because both ICs are `s = (R1-r)/d` at
# leading order):
#
#   :firedrake  Davies et al. (2022) Eq. (41), the published paper's IC:
#                 T = (R1 - r) + A cos(4 atan2(y,x)) sin(pi (r - R0)),  A = 0.02.
#               (r - R0) is the nondimensional height (gap = 1), so the linear
#               part is the published `r_max - r`.
#   :aspect     the ASPECT `spherical hexagonal perturbation`, angular mode 4,
#               rotation offset 0 (the Davies/ASPECT cases' IC):
#                 s = (R1 - r)/d,  phi_A = atan2(x, y)  [ASPECT's argument order]
#                 T = s + 0.2 s (1-s) sin(4 phi_A + pi/2)
#                   = s + 0.2 s (1-s) cos(4 atan2(y, x))
#               (sin(4 phi_A + pi/2) = cos(4 phi_A) and cos(4(pi/2 - theta)) =
#               cos(4 theta); the paper's and ASPECT's angular dependences agree,
#               the radial envelopes differ — both are documented and both are
#               exposed, the published *targets* are steady-state scalars.)
#
# Both fields are benchmarks' *linear-in-r* initial conditions, not the log
# conduction state used for the Nusselt normalisation (ANNULUS_REFS §4.6/§5.4
# spells that difference out); `theta_ic` is the difference, so it would be a
# nonzero smooth field if the base were used as the total.
"""
    _annulus_ic_total(kind, g, x, y) -> Float64

Documented benchmark initial *total* temperature at `(x,y)` (`kind` is
`:firedrake` or `:aspect`, see the section header).  Relative accuracy only: the
values are smooth functions of the position and both vanish at `r = R0`/`R1`.
"""
function _annulus_ic_total(kind::Symbol, g::PolarAnnulus, x::Float64,
                           y::Float64)
    r = hypot(x, y)
    th = atan(y, x)                      # the paper's angle atan2(y, x)
    if kind === :firedrake
        A = 0.02
        return (g.R1 - r) + A * cos(4.0 * th) * sinpi(r - g.R0)
    elseif kind === :aspect
        d = g.R1 - g.R0
        s = (g.R1 - r) / d
        return s + 0.2 * s * (1.0 - s) * cos(4.0 * th)
    end
    throw(ArgumentError("unknown annular initial condition :$kind; use " *
                        ":firedrake (the published paper's Eq. (41)) or " *
                        ":aspect (the ASPECT spherical hexagonal perturbation)."))
end

"""
    _annulus_ic_theta(kind, g, pr, x, y) -> Float64

The annular initial condition as the solver's perturbation:
`_annulus_ic_total(kind, g, x, y) - Tbar(r)` with the geometry's radial log/H
base state.  This is the function the driver samples for `annulus_ic=...` and
re-samples after every AMR cycle (`theta_ic`).
"""
function _annulus_ic_theta(kind::Symbol, g::PolarAnnulus, pr::Params,
                           x::Float64, y::Float64)
    return _annulus_ic_total(kind, g, x, y) -
           semquad_base_T(pr, g, x, y)
end

# The geometry dispatch costs inference budget exactly like the stage-0.5a
# geometry API: `SemQuad.geom` is a non-parametric `AbstractGeometry` field, so
# a call site that only knows the abstract type resolves one method per concrete
# subtype, and Julia's default budget (3) would give up at the fourth.  These
# are inference settings, not code paths (no body, dispatch result or
# allocation on a concrete call changes); a fifth production geometry raises
# them to 5.
Base.Experimental.@max_methods 4 function semquad_base_T end
Base.Experimental.@max_methods 4 function semquad_base_gradient end
