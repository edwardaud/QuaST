# Uniform steady-state entry point for the quadtree semi-quad solver.
#
# Three steady Newtons exist, one per rheology:
#
#   :map  `semquad_newton_steady`   eta(T) (or the Tosi law through the same map,
#                                   with `eta_law=:tosi`), optional EV
#   :tosi `semquad_tosi_newton`     Tosi eta(edot) inner Picard + Kgeo (G161)
#   :dp   `semquad_dp_newton`       Drucker-Prager inner (u,p) Newton + Kgeo/Kp (G168)
#
# `semquad_steady_newton` dispatches on the parameters so that a caller does not
# have to know which one applies.  This is the "steady entry" the time driver
# does not have: `run_semquad_simulation` always time-steps, and this function
# is what a driver-integration would call when a steady solve is requested.

"""
    semquad_steady_newton(sq, qt, ss, pr, theta0; kind=:auto, dt, ...)

Solve for the steady state of the quadtree semi-quad discretisation, choosing the
Newton variant from the rheology:

* `kind=:auto` -> `:tosi` when `pr.tosi !== nothing`, `:dp` when
  `yield_active(pr)` (finite cohesion / friction / yield cap), otherwise `:map`;
* `:map`  -> `semquad_newton_steady` (`eta(T)`; `ev`/`ev_tangent` forwarded),
* `:tosi` -> `semquad_tosi_newton` (`eta(edot)`, inner Picard),
* `:dp`   -> `semquad_dp_newton` (Drucker-Prager, inner `(u,p)` Newton).

Common keywords (`dt, tol, maxit, krylov, restart, verbose, advection,
picard_steps, linear`) are forwarded to all variants; `ev`, `ev_cmax`,
`ev_alpha`, `ev_timestep`, `ev_tangent` only to `:map`, `tosi_tol`/`tosi_maxit`
only to `:tosi`, and `dp_tol`/`dp_maxit`/`dp_kgeo`/`dp_kp` only to `:dp`.
Returns the variant's result named tuple, with `kind` added.
"""
function semquad_steady_newton(sq::SemQuad, qt::Quadtree, ss::SemStokes,
                               pr::Params, theta0::Vector{Float64};
                               kind::Symbol=:auto,
                               dt::Float64=1.0e-2, tol::Float64=1.0e-8,
                               maxit::Int=15, krylov::Int=40,
                               restart::Int=20, verbose::Bool=false,
                               advection::Symbol=:galerkin,
                               picard_steps::Int=0,
                               linear::Symbol=:direct,
                               ev::Float64=0.0, ev_cmax::Float64=0.052,
                               ev_alpha::Int=2, ev_timestep::Int=10,
                               ev_tangent::Bool=true,
                               tosi_tol::Float64=1.0e-3,
                               tosi_maxit::Int=20,
                               dp_tol::Float64=1.0e-3,
                               dp_maxit::Int=30,
                               dp_kgeo::Bool=true, dp_kp::Bool=true)
    k = kind
    if k === :auto
        k = pr.tosi !== nothing ? :tosi : (yield_active(pr) ? :dp : :map)
    end
    common = (dt=dt, tol=tol, maxit=maxit, krylov=krylov, restart=restart,
              verbose=verbose, advection=advection, picard_steps=picard_steps,
              linear=linear)
    r = if k === :tosi
        semquad_tosi_newton(sq, qt, ss, pr, theta0; common...,
                            tosi_tol=tosi_tol, tosi_maxit=tosi_maxit)
    elseif k === :dp
        semquad_dp_newton(sq, qt, ss, pr, theta0; common...,
                          dp_tol=dp_tol, dp_maxit=dp_maxit,
                          dp_kgeo=dp_kgeo, dp_kp=dp_kp)
    elseif k === :map
        semquad_newton_steady(sq, qt, ss, pr, theta0; common...,
                              ev=ev, ev_cmax=ev_cmax, ev_alpha=ev_alpha,
                              ev_timestep=ev_timestep, ev_tangent=ev_tangent)
    else
        throw(ArgumentError("semquad_steady_newton: kind must be :auto, :map, " *
                            ":tosi or :dp (got :$k)"))
    end
    return merge(r, (kind=k,))
end
