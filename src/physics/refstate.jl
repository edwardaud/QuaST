# Reference state: `SemRefState`, its constructors and the base-state
# accessors used by the energy assembly.  The EOS profile type and integrator
# are in eos_profile.jl, included before this file.
# --- SemRefState -----------------------------------------------------------
"""
    SemRefState

Adiabatic reference state for the quadtree SEM path:

- `Tsurf`  surface (top) temperature,
- `alpha`, `cp`, `g`, `rho0`  constant thermal expansivity, specific heat,
  gravity and density,
- `rho(z)`, `alpha(z)`, `cp(z)` profiles (constant here, kept as functions so a
  stratified state can be added without changing the interface),
- `Tad(z)`, `dTad_dz(z)` for the energy equation.

`eos = :analytic` (default) is the closed-form constant-property state above.
`eos = :adams_williamson` / `:murnaghan` instead integrate the equation of state
in depth (see [`SemRefProfile`](@ref)); `alpha_fun`
and `cp_fun` then give the depth dependence of the expansivity and the specific
heat, and `K0`, `K0prime` the Murnaghan bulk modulus at the surface and its
pressure derivative (in units of `rho0 g d`).
"""
struct SemRefState
    Tsurf::Float64
    alpha::Float64
    cp::Float64
    g::Float64
    rho0::Float64
    rho_fun::Union{Nothing,Function}   # optional rho(y); nothing = constant rho0
    prof::Union{Nothing,SemRefProfile} # optional EOS-integrated state
end

function SemRefState(; Tsurf::Real=0.0, alpha::Real=1.0, cp::Real=1.0,
                     g::Real=1.0, rho0::Real=1.0,
                     rho_fun::Union{Nothing,Function}=nothing,
                     eos::Symbol=:analytic, nz::Int=129,
                     alpha_fun::Union{Nothing,Function}=nothing,
                     cp_fun::Union{Nothing,Function}=nothing,
                     K0::Real=1.0, K0prime::Real=4.0)
    alpha >= 0 || throw(ArgumentError("alpha must be >= 0"))
    cp > 0 || throw(ArgumentError("cp must be > 0"))
    g > 0 || throw(ArgumentError("g must be > 0"))
    rho0 > 0 || throw(ArgumentError("rho0 must be > 0"))
    eos in (:analytic, :adams_williamson, :murnaghan) ||
        throw(ArgumentError("eos must be :analytic, :adams_williamson or " *
                            ":murnaghan"))
    if eos === :analytic
        (alpha_fun === nothing && cp_fun === nothing) ||
            throw(ArgumentError("alpha_fun/cp_fun need eos=:adams_williamson " *
                                "or eos=:murnaghan"))
        return SemRefState(Float64(Tsurf), Float64(alpha), Float64(cp),
                           Float64(g), Float64(rho0), rho_fun, nothing)
    end
    nz >= 5 || throw(ArgumentError("nz must be >= 5"))
    (rho_fun === nothing) ||
        throw(ArgumentError("eos=:$(eos) integrates its own density; rho_fun " *
                            "is only meaningful for eos=:analytic"))
    af = alpha_fun === nothing ? (y -> Float64(alpha)) : alpha_fun
    cf = cp_fun === nothing ? (y -> Float64(cp)) : cp_fun
    (af(0.5) > 0.0 && af(1.0) > 0.0) ||
        throw(ArgumentError("alpha_fun must stay positive on [0,1]"))
    (cf(0.5) > 0.0 && cf(1.0) > 0.0) ||
        throw(ArgumentError("cp_fun must stay positive on [0,1]"))
    if eos === :murnaghan
        isfinite(K0) && K0 > 0.0 ||
            throw(ArgumentError("eos=:murnaghan needs a finite K0 > 0"))
    end
    prof = _eos_profile(eos=eos, Tsurf=Float64(Tsurf), rho_surf=Float64(rho0),
                        g=Float64(g), alpha_fun=af, cp_fun=cf, nz=nz,
                        K0=Float64(K0), K0prime=Float64(K0prime))
    SemRefState(Float64(Tsurf), Float64(alpha), Float64(cp), Float64(g),
                Float64(rho0), nothing, prof)
end

# positional 6-argument form (the pre-EOS layout, used by nothing in the tree
# but kept so a stored/hand-built state cannot break)
SemRefState(Tsurf::Float64, alpha::Float64, cp::Float64, g::Float64,
            rho0::Float64, rho_fun::Union{Nothing,Function}) =
    SemRefState(Tsurf, alpha, cp, g, rho0, rho_fun, nothing)

"""Adiabatic temperature at height `z` (0 = bottom, 1 = top)."""
@inline semquad_Tad(rs::SemRefState, z::Float64) =
    (rs.prof === nothing ?
     rs.Tsurf * exp(rs.alpha * rs.g * (1.0 - z) / rs.cp) :
     _prof_T(rs.prof, z))::Float64

"""Adiabatic gradient dTad/dz at height `z` (negative: temperature drops upward)."""
@inline semquad_dTad_dz(rs::SemRefState, z::Float64) =
    (rs.prof === nothing ?
     -rs.alpha * rs.g / rs.cp * semquad_Tad(rs, z) :
     _prof_dT(rs.prof, z))::Float64

@inline semquad_rho(rs::SemRefState, z::Float64) =
    rs.prof !== nothing ? _prof_rho(rs.prof, z) :
    (rs.rho_fun === nothing ? rs.rho0 : rs.rho_fun(z))::Float64
@inline semquad_alpha(rs::SemRefState, z::Float64) =
    (rs.prof !== nothing ? rs.prof.alpha_fun(z)::Float64 : rs.alpha)::Float64
@inline semquad_cp(rs::SemRefState, z::Float64) =
    (rs.prof !== nothing ? rs.prof.cp_fun(z)::Float64 : rs.cp)::Float64

"""`drho/dz` of the reference state (EOS: from the integrated equation of state)."""
@inline semquad_drho_dz(rs::SemRefState, z::Float64) =
    rs.prof !== nothing ? _prof_drho(rs.prof, z) :
    (rs.rho_fun === nothing ? 0.0 : _fd_drho(rs.rho_fun, z))

# central difference of a user-supplied rho_fun (the pre-EOS behaviour of the
# anelastic mass form; kept bit for bit)
@inline function _fd_drho(f::Function, y::Float64)
    hd = 1.0e-6
    yp = min(y + hd, 1.0); ym = max(y - hd, 0.0)
    (f(yp) - f(ym)) / (yp - ym)
end

"""Base-state vertical derivative used by the energy assembly.

With a reference state this is `dTad/dz`; without one it keeps the conductive
profile `Tbar = (1-z) + H/2 z(1-z)` and its derivative, so the default path is
unchanged.
"""
@inline semquad_base_gradient(pr, z::Float64) =
    (pr.refstate === nothing ? dTbar(z, pr.H) :
     semquad_dTad_dz(pr.refstate, z))::Float64

"""
Laplacian of the base temperature.

The energy equation is solved for `theta = T - Tad`, but the diffusion term is
`lap(T) = lap(theta) + lap(Tad)`.  For the exponential adiabat

    Tad(z) = Tsurf exp(alpha g (1-z)/cp)   =>   d2Tad/dz2 = (alpha g/cp)^2 Tad,

so the theta equation carries the source `+lap(Tad)`.  For the Boussinesq base
`Tbar = (1-z) + H/2 z(1-z)` is linear and this is identically zero, which is why
the term only appears once a reference state is used (King 2010: it is 0.156 in
the volume-integrated energy budget, exactly the closure error it removes).

With an EOS state the curvature comes from the integrated state instead:
`lap(Tad) = (Lambda^2 - dLambda/dz) Tad`, the exact second derivative of the
adiabat whose first derivative is the `dTad/dz` used by the advection term.
"""
@inline semquad_base_laplacian(pr, z::Float64) =
    (pr.refstate === nothing ? 0.0 :
     (pr.refstate.prof !== nothing ? _prof_d2T(pr.refstate.prof, z) :
      (pr.refstate.alpha * pr.refstate.g / pr.refstate.cp)^2 *
         semquad_Tad(pr.refstate, z)))::Float64

"""Base-state temperature used where the full T is needed (adiabatic heating)."""
@inline semquad_base_T(pr, z::Float64) =
    (pr.refstate === nothing ? Tbar(z, pr.H) : semquad_Tad(pr.refstate, z))::Float64

"""
    _mapped_base_curvature(pr, g, x, y) -> Float64

The physical Laplacian of the base state in the mapped metric,
`lap(Tbar(h(x,y)))`, as it enters the theta equation's source — i.e. the term
the energy assembly adds and the entropy-viscosity residual subtracts.  With
`h = height(g,x,y)`, `|grad h|` and `lap h = height_laplacian(g,x,y)`:

    lap(Tbar(h)) + H = Tbar''(h)|grad h|^2 + Tbar'(h) lap(h) + H.

For the **Boussinesq base** (`refstate === nothing`, `Tbar(h) = 1-h+H/2 h(1-h)`)
`Tbar'' = -H`, so

    lap(Tbar(h)) + H = H*(1 - |grad h|^2) + Tbar'(h) lap(h).

On the box (and the annulus) `|grad h| = 1` and the `H*(1-|grad h|^2)` term is
zero, which is why the historical box body drops it (the `+H` source is absorbed
by the quadratic `Tbar`); on a mapped validation geometry with `|grad h| != 1`
it is *not* zero.  Cut 3's first mapped body used `semquad_base_laplacian(pr,h)`
alone, which is the reference-state second derivative (`0` for the Boussinesq
base) and therefore silently dropped `H*(1-|grad h|^2)`; this helper is the
cut-4 correction, shared by the energy assembly and the EV residual so the two
cannot drift (cut-4 defect B1). For a
**reference state** the base is the adiabat and the term is exactly
`lap(Tad(h)) = Tad''(h)|grad h|^2 + Tad'(h) lap(h)` (no absorbed `+H`).

This is a mapped-only helper: the box bodies are the historical statements and
do not call it.  It is evaluated at one quadrature point, in cold-per-point
physics code (not an operator kernel of its own).
"""
@inline function _mapped_base_curvature(pr, g, x, y)
    h = height(g, x, y)
    ghx, ghy = grad_height(g, x, y)
    lap_h = height_laplacian(g, x, y)
    gh2 = ghx * ghx + ghy * ghy
    if pr.refstate === nothing
        return pr.H * (1.0 - gh2) + semquad_base_gradient(pr, h) * lap_h
    end
    return semquad_base_laplacian(pr, h) * gh2 +
           semquad_base_gradient(pr, h) * lap_h
end
