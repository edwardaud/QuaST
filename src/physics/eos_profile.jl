# Reference-state EOS profile: the grid-integrated Adams-Williamson / Murnaghan
# state (`SemRefProfile`) and its Hermite/ODE accessors.  Included before
# refstate.jl, whose `SemRefState.prof` field refers to this type.
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
# Section 11 adds `eos = :adams_williamson` and
# `eos = :murnaghan`: the same interface, but with the reference state obtained
# by *integrating an equation of state in depth* instead of the closed form, so
# that rho(y), alpha(y), cp(y), Tad(y), dTad/dz(y), lap(Tad)(y) and drho/dz(y)
# all come from one EOS.

# --- EOS-integrated reference state -----------
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
# The `::Float64` on the field calls is load-bearing:
# `alpha_fun`/`cp_fun` are `Function` *fields* of `SemRefProfile`, which are not
# specialised, so the call inferred as `Any`.  That made `_prof_dT` -> `dTad_dz`
# -> `semquad_base_gradient` -> the energy kernel's `src` all `Any`, and
# `F[ids[k]] += ws * phi[k]` boxed inside the quadrature loop even on the
# default path (`refstate === nothing`), where the branch is never taken:
# measured 0.743 -> 0.396 ms and 1.86 -> 1.14 MB per energy assembly, i.e. a
# 1.88x on the 20 energy assemblies of one per-step Newton.  The values are
# unchanged (a typeassert on a Float64).  `_prof_phi`
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

