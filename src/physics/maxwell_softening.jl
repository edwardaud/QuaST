# The 2a effective-viscosity softening for the per-step Newton viscosity factory.
#
# The Newton residual assembles `2 eta_eff eps(u)` with the softened viscosity, so
# every derivative closure it hands to the analytic tangent has to be softened
# with the chain rule `d eta_eff / dx = (eta_eff/eta)^2 d eta / dx`.  A `nothing`
# derivative entry stays `nothing` (the tangent code uses it to decide whether a
# block exists).  `G = 0` returns the factory unchanged.

# 2a effective-viscosity softening for the per-step Newton viscosity factory
# (`newton_make_eta` in driver/run.jl).  The Newton residual assembles
# `2 eta_eff eps(u)` with the softened viscosity, so every derivative closure it
# hands to the analytic tangent has to be softened with the chain rule
#
#     d eta_eff / dx = (eta_eff/eta)^2 d eta / dx.
#
# A `nothing` derivative entry (e.g. the strain-rate derivative of a
# temperature-only law) stays `nothing`: the tangent code uses `=== nothing` to
# decide whether a block exists at all.  G = 0 returns `mk` unchanged.
function _ve_soften_eta_factory(mk::Tuple, G::Float64, dt::Float64)
    G <= 0.0 && return mk
    eta_raw = mk[1]
    eta_eff = (x, y) -> begin
        e = eta_raw(x, y)
        1.0 / (1.0 / e + 1.0 / (G * dt))
    end
    r2 = (x, y) -> begin
        e = eta_raw(x, y)
        ee = eta_eff(x, y)
        (ee / e)^2
    end
    dT = mk[2] === nothing ? nothing : ((x, y) -> r2(x, y) * mk[2](x, y))
    de = length(mk) >= 3 && mk[3] !== nothing ?
         ((x, y, ed) -> r2(x, y) * mk[3](x, y, ed)) : nothing
    if length(mk) >= 5
        dp = mk[4] === nothing ? nothing :
             ((x, y, ed, pq) -> r2(x, y) * mk[4](x, y, ed, pq))
        dchi = mk[5] === nothing ? nothing :
               ((x, y, ed) -> r2(x, y) * mk[5](x, y, ed))
        return (eta_eff, dT, de, dp, dchi)
    end
    return (eta_eff, dT, de)
end
