# BDF2 energy step on the quadtree SEM mesh (time-dependent AMR building block).
#
# theta is the perturbation T - Tbar; the top/bottom walls are homogeneous
# Dirichlet, the side walls are natural (insulating).  The hanging-node
# constraints and the fixed wall dofs are lifted with the same affine map used
# by the Stokes solver, so the reduced system stays symmetric.

"""
    solve_semquad_energy(sq, qt, pr, theta_n, theta_nm1, ux, uz, dt, dt_prev,
                         firststep; advection=:galerkin)

One implicit BDF2 (or backward-Euler) step of

    d(theta)/dt + u.grad(theta) = lap(theta) - u_z dTbar/dz

on the quadtree, with `M, K, C, F` from `assemble_semquad_energy`.  `Tbar` is
the steady conductive profile, so its `+H` is already absorbed and no explicit
source is added here.  Returns the full theta vector (wall values zero).

`return_parts=true` instead returns the assembled operator as a named tuple
(`M, K, C, F, A, rhs, u0, Cfree, Ar, ca, cb, cc, firststep, ev_field`), where
`A = ca*M + K + C` is the time-step operator and `rhs` its right-hand side
(`ca`/`cb`/`cc` are the BDF2 mass coefficients in units of `1/dt`); this is what
the per-step Newton (`semquad_step_newton`) uses for its residual and its
`d r_E / d theta` block.  The default path is unchanged.

`phases` switches on the latent heat of the phase transitions as an apparent
heat capacity (docs/RHEOLOGY.md §7): `M` and `C` carry the factor
`(1 + latent_L*Gamma_T)` and `F` the advective source `-latent_L*Gamma_z*u_z`.
`latent_theta` selects the temperature at which those two coefficients are
evaluated; the driver passes the current nonlinear iterate so the capacity is
implicit, while the BDF2 levels themselves stay at the start-of-step `theta_n`.
`nothing` (default) means `theta_n`, i.e. the same matrices the pre-existing
assembly-level latent code produces, and with `phases` empty nothing changes.
"""
function solve_semquad_energy(sq::SemQuad, qt::Quadtree, pr::Params,
                              theta_n::Vector{Float64}, theta_nm1::Vector{Float64},
                              ux::Vector{Float64}, uz::Vector{Float64},
                              dt::Float64, dt_prev::Float64, firststep::Bool;
                              advection::Symbol=:galerkin,
                              ev::Float64=0.0, ev_cmax::Float64=0.052,
                              ev_alpha::Int=2, ev_timestep::Int=0,
                              ev_out::Union{Nothing,Vector{Float64}}=nothing,
                              eta_fun::Union{Nothing,Function}=nothing,
                              flux_top::Union{Nothing,Float64}=nothing,
                              flux_bot::Union{Nothing,Float64}=nothing,
                              phases::Vector{PhaseTransition}=PhaseTransition[],
                              latent_theta::Union{Nothing,Vector{Float64}}=nothing,
                              return_parts::Bool=false)
    rho_ref = (pr.rho_ref_weight && pr.refstate !== nothing) ?
              (y -> semquad_rho(pr.refstate, y) * pr.refstate.cp) : nothing
    evg = ev > 0.0 ?
          semquad_ev_globals(sq, qt, pr, theta_n, theta_nm1, ux, uz,
                             dt / dt_prev) : nothing
    # `return_parts` hands the assembled operator back to the caller instead of
    # solving it (the per-step Newton needs `A_E`, `rhs` and the constraint map
    # for its residual and its `d r_E / d theta` block).  The EV field is
    # recomputed from `ev`/`evg` and recorded into `ev_out` (or into an internal
    # buffer when the caller does not supply one).
    ev_out_used = ev_out
    if return_parts && ev > 0.0 && ev_out_used === nothing
        ev_out_used = zeros(Float64, length(sq.leaves))
    end
    M, K, C, F = assemble_semquad_energy(sq, qt, pr, ux, uz;
                                         advection=advection,
                                         theta=theta_n,
                                         theta_lat=latent_theta,
                                         phases=phases,
                                         rho_heating=pr.rho_heating,
                                         rho_cp=pr.rho_cp,
                                         heating_adiabatic=pr.heating_adiabatic,
                                         heating_dissipation=pr.heating_dissipation,
                                         eta_fun=eta_fun, rho_ref=rho_ref,
                                         ev=ev, ev_cmax=ev_cmax,
                                         ev_alpha=ev_alpha,
                                         ev_timestep=ev_timestep,
                                         evg=evg, theta_prev=theta_nm1,
                                         dt=dt_prev, ev_out=ev_out_used)
    fixed = Dict{Int,Float64}()
    for g in 1:sq.N
        y = sq.coords[g][2]
        # theta_wall = Twall - Tad: zero for the Boussinesq base (Tad = Tbar,
        # Twall = 1 at the bottom and 0 at the top), non-zero once a reference
        # state shifts the base off the physical wall temperature.  A wall with a
        # prescribed flux is left free; the flux enters the right-hand side.
        y <= 1.0e-14 && flux_bot === nothing &&
            (fixed[g] = pr.Twall_bot - semquad_base_T(pr, y))
        y >= 1.0 - 1.0e-14 && flux_top === nothing &&
            (fixed[g] = pr.Twall_top - semquad_base_T(pr, y))
    end
    if flux_top !== nothing || flux_bot !== nothing
        flux_top !== nothing && flux_bot !== nothing &&
            throw(ArgumentError("at least one wall must keep a Dirichlet condition; " *
                                "two flux walls leave theta undetermined"))
        F = copy(F)
        flux_top === nothing || _add_semquad_wall_flux!(F, sq, qt, flux_top, :top)
        flux_bot === nothing || _add_semquad_wall_flux!(F, sq, qt, flux_bot, :bottom)
    end
    cons = semquad_constraints(sq, qt)
    u0, Cfree, _ = _affine_constraint(sq.N, cons, fixed)

    if firststep
        A = (1.0 / dt) * M + K + C
        rhs = (1.0 / dt) * (M * theta_n) + F
        ca = 1.0 / dt
        cb = 1.0 / dt
        cc = 0.0
    else
        w = dt / dt_prev
        a = (2.0 * w + 1.0) / (1.0 + w)
        b = 1.0 + w
        c = w * w / (1.0 + w)
        A = (a / dt) * M + K + C
        rhs = (b / dt) * (M * theta_n) - (c / dt) * (M * theta_nm1) + F
        ca = a / dt
        cb = b / dt
        cc = c / dt
    end
    Ar = Cfree' * A * Cfree
    if return_parts
        # `ca`/`cb`/`cc` are the mass coefficients of `A` and of `rhs` in units
        # of 1/dt: A = ca*M + K + C and rhs carries cb*M*theta (+ -cc*M*theta_prev).
        return (M=M, K=K, C=C, F=F, A=A, rhs=rhs, u0=u0, Cfree=Cfree, Ar=Ar,
                ca=ca, cb=cb, cc=cc, firststep=firststep,
                ev_field=ev_out_used)
    end
    br = Cfree' * (rhs - A * u0)
    ur = Ar \ br
    return u0 + Cfree * ur
end

"""
    _add_semquad_wall_flux!(F, sq, qt, q, wall)

Add the boundary term `int_wall q phi ds` to the energy right-hand side.  `q` is
the prescribed *outward normal* derivative `dtheta/dn` on that wall (for the
bottom wall, where the outward normal is -z, a positive `q` therefore means
`-dtheta/dz > 0`, i.e. heat entering from below).
"""
function _add_semquad_wall_flux!(F::Vector{Float64}, sq::SemQuad, qt::Quadtree,
                                 q::Float64, wall::Symbol)
    wall in (:top, :bottom) || throw(ArgumentError("wall must be :top or :bottom"))
    n = sq.p + 1
    for lf in sq.leaves
        (wall === :top ? lf[3] == sq.ny0 * (1 << lf[1]) - 1 : lf[3] == 0) || continue
        hx = leaf_hx(sq, lf[1])
        ids = sq.elem[lf]
        row = wall === :top ? n : 1
        for i in 1:n
            F[ids[(row - 1) * n + i]] += 0.5 * hx * sq.w[i] * q
        end
    end
    return F
end
