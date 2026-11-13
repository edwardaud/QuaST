"""
    RefState

One-dimensional adiabatic reference state on a grid `z` (0 = bottom, 1 = top):

- `rho`, `alpha`, `cp`, `k`, `beta`: reference profiles
  (density, thermal expansion, specific heat, conductivity, compressibility),
- `Tad`: adiabatic temperature,
- `pad`: hydrostatic pressure (surface value `Psurf` at z=1),
- `g`: gravity magnitude (positive downward),
- `rho0, alpha0, cp0, k0, beta0`: reference values used for normalisation.

The profiles satisfy, in the continuum limit,

    dTad/dz = - alpha g Tad / cp
    dpad/dz = - rho g

with z pointing upward.
"""
struct RefState
    z::Vector{Float64}
    rho::Vector{Float64}
    alpha::Vector{Float64}
    cp::Vector{Float64}
    k::Vector{Float64}
    beta::Vector{Float64}
    Tad::Vector{Float64}
    pad::Vector{Float64}
    g::Float64
    rho0::Float64
    alpha0::Float64
    cp0::Float64
    k0::Float64
    beta0::Float64
end

# Backwards-compatible constructor: profiles are already normalised.
RefState(z, rho, alpha, cp, Tad, pad, g) =
    RefState(z, rho, alpha, cp, rho .* cp, zeros(length(z)), Tad, pad, g,
             1.0, 1.0, 1.0, 1.0, 1.0)

"""
    build_refstate(; nz=257, rho_fun, alpha_fun, cp_fun, k_fun=nothing,
                     beta_fun=nothing, g=1.0, rho0=1, alpha0=1, cp0=1, k0=1,
                     beta0=1, Tsurf=0.0, Psurf=0.0)

Build a [`RefState`](@ref) by integrating the hydrostatic and adiabatic gradients
downward from the surface (z=1). `k_fun` defaults to `rho Cp` (constant
nondimensional diffusivity); `beta_fun` defaults to zero.
"""
function build_refstate(; nz::Int=257,
                        rho_fun::Function = z -> 1.0,
                        alpha_fun::Function = z -> 1.0,
                        cp_fun::Function = z -> 1.0,
                        k_fun::Union{Nothing,Function}=nothing,
                        beta_fun::Union{Nothing,Function}=nothing,
                        g::Float64=1.0,
                        rho0::Float64=1.0,
                        alpha0::Float64=1.0,
                        cp0::Float64=1.0,
                        k0::Float64=1.0,
                        beta0::Float64=1.0,
                        Tsurf::Float64=0.0,
                        Psurf::Float64=0.0)
    nz >= 3 || throw(ArgumentError("nz must be >= 3"))
    z = collect(range(0.0, 1.0; length=nz))
    rho = [rho_fun(zi) for zi in z]
    alpha = [alpha_fun(zi) for zi in z]
    cp = [cp_fun(zi) for zi in z]
    k = k_fun === nothing ? rho .* cp : [k_fun(zi) for zi in z]
    beta = beta_fun === nothing ? zeros(nz) : [beta_fun(zi) for zi in z]
    Tad = zeros(nz); pad = zeros(nz)
    Tad[end] = Tsurf; pad[end] = Psurf
    for i in (nz-1):-1:1
        dz = z[i+1] - z[i]
        zm = 0.5 * (z[i] + z[i+1])
        # integrate downward: T and p increase with depth
        Tad[i] = Tad[i+1] * exp(alpha_fun(zm) * g * dz / cp_fun(zm))
        pad[i] = pad[i+1] + rho_fun(zm) * g * dz
    end
    return RefState(z, rho, alpha, cp, k, beta, Tad, pad, g,
                    rho0, alpha0, cp0, k0, beta0)
end

"""
    refstate_at(rs, z)

Linearly interpolate `(rho, alpha, cp, k, beta, Tad, pad)` at height `z`
(clamped).
"""
@inline function refstate_at(rs::RefState, z::Float64)
    n = length(rs.z)
    if z <= rs.z[1]
        return rs.rho[1], rs.alpha[1], rs.cp[1], rs.k[1], rs.beta[1],
               rs.Tad[1], rs.pad[1]
    elseif z >= rs.z[n]
        return rs.rho[n], rs.alpha[n], rs.cp[n], rs.k[n], rs.beta[n],
               rs.Tad[n], rs.pad[n]
    end
    i = searchsortedlast(rs.z, z)
    t = (z - rs.z[i]) / (rs.z[i+1] - rs.z[i])
    return (rs.rho[i] + t*(rs.rho[i+1]-rs.rho[i]),
            rs.alpha[i] + t*(rs.alpha[i+1]-rs.alpha[i]),
            rs.cp[i] + t*(rs.cp[i+1]-rs.cp[i]),
            rs.k[i] + t*(rs.k[i+1]-rs.k[i]),
            rs.beta[i] + t*(rs.beta[i+1]-rs.beta[i]),
            rs.Tad[i] + t*(rs.Tad[i+1]-rs.Tad[i]),
            rs.pad[i] + t*(rs.pad[i+1]-rs.pad[i]))
end

"""
    refstate_drho(rs, z)

Interpolated derivative `drho/dz` at height `z` (central difference on the grid,
linear interpolation between nodes).
"""
@inline function refstate_drho(rs::RefState, z::Float64)
    n = length(rs.z)
    if n < 4
        return (rs.rho[end] - rs.rho[1]) / (rs.z[end] - rs.z[1])
    end
    i = clamp(searchsortedlast(rs.z, z), 2, n-2)
    # central differences at nodes i and i+1
    d1 = (rs.rho[i+1] - rs.rho[i-1]) / (rs.z[i+1] - rs.z[i-1])
    d2 = (rs.rho[i+2] - rs.rho[i]) / (rs.z[i+2] - rs.z[i])
    t = (z - rs.z[i]) / (rs.z[i+1] - rs.z[i])
    return d1 + t * (d2 - d1)
end

