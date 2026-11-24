# Reference state for this solver (P3.2 foundation).
#
# the earlier structured-grid layer carries a grid-based `RefState` (refstate.jl) with separate rho, alpha,
# cp and beta profiles.  this solver evaluates everything at quadrature points, so a
# closed-form state is cheaper and exactly differentiable; this module provides
# the constant-property adiabat and the interface the energy assembly uses.
#
#   dTad/dz = - alpha * g * Tad / cp        (z = height, gravity downward)
#
# which integrates to Tad(z) = Tsurf * exp(alpha g (1 - z) / cp).
#
# Section 11 of docs/dev/PHYSICS.md adds `eos = :adams_williamson` and
# `eos = :murnaghan`: the same interface, but with the reference state obtained
# by *integrating an equation of state in depth* instead of the closed form, so
# that rho(y), alpha(y), cp(y), Tad(y), dTad/dz(y), lap(Tad)(y) and drho/dz(y)
# all come from one EOS.

# --- EOS-integrated reference state (docs/dev/PHYSICS.md section 11) -----------
#
# State variables on a uniform grid in height y (0 bottom, 1 top), integrated
# from the surface down:
#
#   dT/dy   = -Lambda(y) T,            Lambda = alpha(y) g / cp(y)
#   drho/dy = -rho g / phi(y),         phi = K_S/rho  ("seismic parameter")
#   dp/dy   = -rho g                   (hydrostatic)
#
# `eos = :adams_williamson` uses the thermodynamic closure K_S = rho cp/alpha
# (phi = cp/alpha, so dln rho/dy = -Lambda: the density and the adiabat are
# proportional).  `eos = :murnaghan` uses the Murnaghan/BM3 fit
# K_S = K0 + K0'(p - p_surf), which makes the density ODE nonlinear and couples
# it to the hydrostatic pressure.  Both are written in the code's
# nondimensionalisation: rho in units of rho0, y in units of d, p and K_S in
# units of rho0 g d, alpha in units of alpha0, T in units of dT.  The Boussinesq
# buoyancy is Ra rho(y) alpha(y) theta, so rho and alpha are the only state
# variables the momentum equation sees.
#
# The table is interpolated with cubic Hermite whose node derivatives are the
# ODE right-hand sides; the *derivatives* the physics needs (dTad/dz,
# lap(Tad), drho/dz) are evaluated from the same ODEs at the query point, so
# they are exactly the EOS expressions and not finite differences of the table.
struct SemRefProfile
    n::Int
    z::Vector{Float64}       # ascending grid, 0 ... 1
    T::Vector{Float64}
    dT::Vector{Float64}      # node derivatives (ODE right-hand sides)
    rho::Vector{Float64}
    drho::Vector{Float64}
    p::Vector{Float64}
    dp::Vector{Float64}
    alpha_fun::Function
    cp_fun::Function
    eos::Symbol
    g::Float64
    K0::Float64
    K0prime::Float64
end

"Adiabatic lapse-rate coefficient `Lambda(y) = alpha(y) g / cp(y)`."
# The `::Float64` on the field calls is load-bearing (docs/dev/BUG_LEDGER.md S15):
# `alpha_fun`/`cp_fun` are `Function` *fields* of `SemRefProfile`, which are not
# specialised, so the call inferred as `Any`.  That made `_prof_dT` -> `dTad_dz`
# -> `semquad_base_gradient` -> the energy kernel's `src` all `Any`, and
# `F[ids[k]] += ws * phi[k]` boxed inside the quadrature loop even on the
# default path (`refstate === nothing`), where the branch is never taken:
# measured 0.743 -> 0.396 ms and 1.86 -> 1.14 MB per energy assembly, i.e. a
# 1.88x on the 20 energy assemblies of one per-step Newton (docs/dev/BUG_LEDGER.md
# §4.2).  The values are unchanged (a typeassert on a Float64).  `_prof_phi`
# below keeps its unasserted `cp_fun`/`alpha_fun` calls: it is only on the
# anelastic-mass path, which this audit did not measure, and it is recorded as
# a cold `Any` rather than changed blind.
@inline _prof_lambda(pf::SemRefProfile, y::Float64) =
    (pf.alpha_fun(y)::Float64) * pf.g / (pf.cp_fun(y)::Float64)

"Seismic parameter `phi = K_S/rho` at height `y`, for a density `rho_y` and pressure `p_y`."
@inline function _prof_phi(pf::SemRefProfile, y::Float64, rho_y::Float64,
                           p_y::Float64)
    if pf.eos === :murnaghan
        K = pf.K0 + pf.K0prime * p_y
        K > 0.0 || throw(ArgumentError("Murnaghan K_S = K0 + K0'*(p - p_surf) " *
                                       "is not positive at y=$y (K_S=$K)"))
        return K / rho_y
    end
    return pf.cp_fun(y) / pf.alpha_fun(y)
end

# Cubic Hermite interpolation of a tabulated value with exact node derivatives.
@inline function _prof_val(v::Vector{Float64}, d::Vector{Float64},
                           n::Int, y::Float64)
    last = n - 1
    t = clamp(y, 0.0, 1.0) * last
    i = min(Int(floor(t)) + 1, last)
    i < 1 && (i = 1)
    tt = t - (i - 1)
    h = 1.0 / last
    tt2 = tt * tt
    tt3 = tt2 * tt
    h00 = 2tt3 - 3tt2 + 1.0
    h10 = tt3 - 2tt2 + tt
    h01 = -2tt3 + 3tt2
    h11 = tt3 - tt2
    h00 * v[i] + h10 * h * d[i] + h01 * v[i + 1] + h11 * h * d[i + 1]
end

"""
    _eos_profile(; eos, Tsurf, rho_surf, g, alpha_fun, cp_fun, nz, K0, K0prime)

Integrate the EOS in depth from the surface (`y = 1`, `T = Tsurf`, `rho =
rho_surf`, `p = 0`) down to `y = 0` with RK4 (8 substeps per grid interval) and
return a [`SemRefProfile`](@ref).  `nz >= 5` grid points.
"""
function _eos_profile(; eos::Symbol, Tsurf::Float64, rho_surf::Float64,
                      g::Float64, alpha_fun::Function, cp_fun::Function,
                      nz::Int, K0::Float64, K0prime::Float64)
    n = nz
    h = 1.0 / (n - 1)
    z = [(i - 1) * h for i in 1:n]
    T = zeros(Float64, n)
    rho = zeros(Float64, n)
    p = zeros(Float64, n)
    T[n] = Tsurf; rho[n] = rho_surf; p[n] = 0.0
    function rhs(y::Float64, s::NTuple{3,Float64})
        Ts, rs, ps = s
        Lam = alpha_fun(y) * g / cp_fun(y)
        dr = if eos === :murnaghan
            K = K0 + K0prime * ps
            K > 0.0 || throw(ArgumentError("Murnaghan K_S = K0 + K0'*(p - " *
                                           "p_surf) is not positive at y=$y " *
                                           "(K_S=$K); raise K0 or K0prime"))
            -rs * rs * g / K
        else
            -Lam * rs
        end
        (-Lam * Ts, dr, -rs * g)
    end
    substeps = 8
    dz = -h / substeps
    for i in (n - 1):-1:1
        s1, s2, s3 = T[i + 1], rho[i + 1], p[i + 1]
        y = z[i + 1]
        for _ in 1:substeps
            k1 = rhs(y, (s1, s2, s3))
            k2 = rhs(y + dz / 2, (s1 + dz / 2 * k1[1], s2 + dz / 2 * k1[2],
                                  s3 + dz / 2 * k1[3]))
            k3 = rhs(y + dz / 2, (s1 + dz / 2 * k2[1], s2 + dz / 2 * k2[2],
                                  s3 + dz / 2 * k2[3]))
            k4 = rhs(y + dz, (s1 + dz * k3[1], s2 + dz * k3[2], s3 + dz * k3[3]))
            s1 += dz / 6 * (k1[1] + 2k2[1] + 2k3[1] + k4[1])
            s2 += dz / 6 * (k1[2] + 2k2[2] + 2k3[2] + k4[2])
            s3 += dz / 6 * (k1[3] + 2k2[3] + 2k3[3] + k4[3])
            y += dz
        end
        T[i], rho[i], p[i] = s1, s2, s3
    end
    dT = zeros(Float64, n); drho = zeros(Float64, n); dp = zeros(Float64, n)
    for i in 1:n
        Lam = alpha_fun(z[i]) * g / cp_fun(z[i])
        dT[i] = -Lam * T[i]
        if eos === :murnaghan
            K = K0 + K0prime * p[i]
            K > 0.0 || throw(ArgumentError("Murnaghan K_S = K0 + K0'*(p - " *
                                           "p_surf) is not positive at y=" *
                                           "$(z[i]) (K_S=$K); raise K0 or " *
                                           "K0prime"))
            drho[i] = -rho[i] * rho[i] * g / K
        else
            drho[i] = -Lam * rho[i]
        end
        dp[i] = -rho[i] * g
    end
    SemRefProfile(n, z, T, dT, rho, drho, p, dp, alpha_fun, cp_fun, eos, g,
                  K0, K0prime)
end

# --- EOS accessors -----------------------------------------------------------
# `::Float64` on the four accessors that go through `_prof_lambda` keeps the
# base-profile chain type stable (see the note at `_prof_lambda`).
@inline _prof_T(pf::SemRefProfile, y::Float64) = _prof_val(pf.T, pf.dT, pf.n, y)
@inline _prof_dT(pf::SemRefProfile, y::Float64) =
    (-_prof_lambda(pf, y) * _prof_T(pf, y))::Float64
# d2T/dy2 = d/dy(-Lambda T) = Lambda^2 T - Lambda' T
function _prof_d2T(pf::SemRefProfile, y::Float64)
    Lam = _prof_lambda(pf, y)
    eps = 1.0e-6
    yp = min(y + eps, 1.0); ym = max(y - eps, 0.0)
    dLam = (_prof_lambda(pf, yp) - _prof_lambda(pf, ym)) / (yp - ym)
    return ((Lam * Lam - dLam) * _prof_T(pf, y))::Float64
end
@inline _prof_rho(pf::SemRefProfile, y::Float64) =
    _prof_val(pf.rho, pf.drho, pf.n, y)
@inline _prof_p(pf::SemRefProfile, y::Float64) = _prof_val(pf.p, pf.dp, pf.n, y)
@inline _prof_drho(pf::SemRefProfile, y::Float64) =
    -_prof_rho(pf, y) * pf.g /
    _prof_phi(pf, y, _prof_rho(pf, y), _prof_p(pf, y))

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
in depth (see [`SemRefProfile`](@ref), docs/dev/PHYSICS.md section 11); `alpha_fun`
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
adiabat whose first derivative is the `dTad/dz` used by the advection term
(docs/dev/PHYSICS.md section 11).
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
cannot drift (see `docs/dev/GEOM_PROGRESS.md`, cut-4 defect B1).  For a
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
