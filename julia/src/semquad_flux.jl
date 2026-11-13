# Consistent boundary flux (CBF) for the temperature equation on the quadtree.
#
# The wall Nusselt number is currently obtained by differentiating theta at the
# wall (`semquad_nu_walls`), which is a pointwise gradient and loses accuracy on
# coarse meshes (the King L4 difference of +1.5% / +4% in Nu_top / Nu_bot is
# exactly this).  Gresho & Sani's consistent boundary flux uses the *residual*
# of the discrete equation instead: for  K theta + C theta = F  with theta = 0
# on the wall, the residual at a wall node is the reaction, and because the wall
# shape functions sum to one,
#
#     sum_{i in wall} r_i = -int_wall dtheta/dn ds ,
#
# i.e. the flux is a by-product of the volume residual and is consistent with
# the discrete solution and the global budget (this is the same construction as
# route A's cbf.jl).
#
# Normalisation: the returned value is `-sum r` = `int_wall dtheta/dn ds`, the
# same quantity `semquad_nu_walls` differentiates pointwise, so the Nusselt
# number is `base_ref - flux / Lx` with the same base reference.

"""
    semquad_cbf_theta_flux(sq, qt, pr, theta, ux, uz; wall=:top, kwargs...)

Wall flux of the perturbation temperature, `int_wall dtheta/dn ds` (n is the
outward normal), from the residual of the energy operator assembled with the
same options as the run (`advection`, `rho_heating`, `heating_adiabatic`, ...).

`ev_field` is the elementwise entropy-viscosity field used by the energy step
that produced `theta` (one value per leaf, in `sq.leaves` order).  Passing it is
what makes the residual consistent with the EV-stabilised operator; without it
the CBF flux omits the EV diffusion and is biased low (2b: -4.5% in Nu_top).
The field cannot be recomputed from the post-step state because the EV weights
depend on `theta^n` and `theta^(n-1)`, and `theta^(n-1)` is no longer available.
"""
function semquad_cbf_theta_flux(sq::SemQuad, qt::Quadtree, pr::Params,
                                theta::Vector{Float64}, ux::Vector{Float64},
                                uz::Vector{Float64};
                                wall::Symbol=:top,
                                advection::Symbol=:galerkin,
                                ev_field::Union{Nothing,Vector{Float64}}=nothing,
                                kwargs...)
    wall in (:top, :bottom) ||
        throw(ArgumentError("wall must be :top or :bottom"))
    _, K, C, F = assemble_semquad_energy(sq, qt, pr, ux, uz; advection=advection,
                                         theta=theta, ev_field=ev_field, kwargs...)
    r = K * theta .+ C * theta .- F
    n = sq.p + 1
    seen = Set{Int}()
    s = 0.0
    for lf in sq.leaves
        (wall === :top ? lf[3] == sq.ny0 * (1 << lf[1]) - 1 : lf[3] == 0) || continue
        ids = sq.elem[lf]
        row = wall === :top ? n : 1
        for i in 1:n
            g = ids[(row - 1) * n + i]
            g in seen && continue
            push!(seen, g)
            s += r[g]
        end
    end
    # The residual at a Dirichlet node is the reaction, and the weak form gives
    #   (K theta + C theta - F)_i = int phi_i dtheta/dn ds
    # with n the *outward* normal.  Converting to the z-derivative integral (the
    # quantity `semquad_nu_walls` differentiates) needs a sign flip at the
    # bottom, where the outward normal is -z.
    return wall === :top ? s : -s
end

"""
    semquad_nu_cbf(sq, qt, pr, theta, ux, uz; kwargs...)

Wall Nusselt numbers from the consistent boundary flux, in the same normalisation
as `semquad_nu_walls` (which is `base_ref - pointwise gradient`).  Pass
`ev_field` (the elementwise EV field of the step that produced `theta`) to make
the flux consistent with an EV-stabilised run.
"""
function semquad_nu_cbf(sq::SemQuad, qt::Quadtree, pr::Params,
                        theta::Vector{Float64}, ux::Vector{Float64},
                        uz::Vector{Float64}; ev_field::Union{Nothing,Vector{Float64}}=nothing,
                        kwargs...)
    A = total_area(qt)
    st = semquad_cbf_theta_flux(sq, qt, pr, theta, ux, uz; wall=:top,
                                ev_field=ev_field, kwargs...)
    sb = semquad_cbf_theta_flux(sq, qt, pr, theta, ux, uz; wall=:bottom,
                                ev_field=ev_field, kwargs...)
    top_ref = -semquad_base_gradient(pr, 1.0)
    bot_ref = -semquad_base_gradient(pr, 0.0)
    # Noether: with theta = 0 on both walls the outward normal derivative at the
    # top is +d/dz and at the bottom -d/dz, so the signs are the same as the
    # pointwise version.
    return top_ref - st / A, bot_ref - sb / A
end
