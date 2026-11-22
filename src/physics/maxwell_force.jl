# The previous step's stress as the momentum equation sees it.
#
# The elastic force term is `div(a tau^n)` with `a = eta/(eta + G*dt)`, not
# `div(tau)`: the factor is what makes the equilibrated fixed point of a
# viscoelastic flow the *material*-viscosity flow.  `maxwell_force_stress`
# returns the three force components the variable-viscosity assembly loads.
#

"""
    maxwell_force_stress(txx, tyy, txy, eta, dt, G)

The previous step's deviatoric stress as the **momentum equation** sees it
(2b step 2), i.e. the elastic force term `div(tau)` of ASPECT's viscoelastic
formulation.

One backward-Euler step of the Maxwell update splits the new stress into

    tau^{n+1} = 2 eta_eff ehat^{n+1} + a tau^n,     a = eta/(eta + G dt),
    eta_eff   = 1/(1/eta + 1/(G dt)),

where `ehat` is the **deviatoric** strain rate (G249); the load assembled here
leaves out the co-rotational increment of the old stress, which the paragraph
further down states and justifies.  The first term is already carried by the
Stokes operator through the effective viscosity.  The second is an explicit
load: momentum gains `div(a tau^n)`, whose weak form against the velocity test
functions is `-int a tau^n : eps(v)`.  This function returns the three nodal
components of `a tau^n` (`eta` is the *material*, nodal viscosity; `G dt` is
the same product the 2a closure uses), which is exactly the tensor field the
Stokes assembly turns into that load.

The two states the scheme needs follow from `a`:

    tau = 0          -> no elastic force,
    tau = 2 eta ehat ->  eta_eff + a eta = eta,

so an equilibrated viscoelastic flow is the purely viscous one (material `eta`),
not the `eta_eff` one; the two agree as `G dt >> eta` (the coefficient is
`tau_M/dt`).  G247 pins that limit.

With the co-rotational term on, the exact split of the update is

    tau^{n+1} = 2 eta_eff ehat^{n+1} + a R(tau^n),
        R(tau) = tau + dt (W tau - tau W),     a = 1/(1 + dt/tau_M),

but the load assembled here is `a tau^n`, without the rotation increment.  That
is a deliberate scope choice: the split, its `a`, its fixed-point identity and
the gated force term stay exactly as 2b step 3 left them, the load is bit-identical
for a rotation-free flow, and the omitted piece `a dt (W tau - tau W)` is the
lagged part of the same first-order scheme.  The *stress state* -- what the
diagnostics report and what the next step's update reads -- does carry the
rotation in full (`_advance_maxwell_stress!`); only this one-step lagged load is
left out.  In the equilibrated limit the missing load is `O(tau_M |W|)` of the
momentum stress while its effect on the velocity fixed point is
`O((tau_M/dt) tau_M |W|)`, so for the fast-relaxation regime the gates run in
(`dt >> tau_M`) the material-viscosity fixed point of G247 is preserved, and for
`W = 0` nothing changes at all (G250).

`G <= 0` is rejected: with `shear_modulus = 0` the term must not be assembled at
all (the caller's convention), and the formula above would silently return
`tau` itself.
"""
function maxwell_force_stress(txx::Vector{Float64}, tyy::Vector{Float64},
                              txy::Vector{Float64}, eta::Vector{Float64},
                              dt::Float64, G::Float64)
    G > 0.0 || throw(ArgumentError("maxwell_force_stress requires G > 0; with " *
                                   "shear_modulus = 0 the elastic force term " *
                                   "must not be assembled"))
    dt > 0.0 || throw(ArgumentError("maxwell_force_stress requires dt > 0"))
    n = length(eta)
    (length(txx) == n && length(tyy) == n && length(txy) == n) ||
        throw(ArgumentError("the Maxwell stress fields and eta must have the " *
                            "same length; got ($(length(txx)), " *
                            "$(length(tyy)), $(length(txy))) against $n"))
    sx = Vector{Float64}(undef, n)
    sy = Vector{Float64}(undef, n)
    sz = Vector{Float64}(undef, n)
    Gdt = G * dt
    @inbounds for i in 1:n
        a = eta[i] / (eta[i] + Gdt)
        sx[i] = a * txx[i]
        sy[i] = a * tyy[i]
        sz[i] = a * txy[i]
    end
    return (sx, sy, sz)
end
