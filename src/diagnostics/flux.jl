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
# the earlier structured-grid layer's cbf.jl).
#
# Normalisation: the returned value is `-sum r` = `int_wall dtheta/dn ds`, the
# same quantity `semquad_nu_walls` differentiates pointwise, so the Nusselt
# number is `base_ref - flux / Lx` with the same base reference.

"""
    semquad_cbf_theta_flux(sq, qt, pr, theta, ux, uz; wall=:top, kwargs...)

Wall flux of the perturbation temperature, `int_wall dtheta/dn ds` (n is the
outward normal), from the residual of the energy operator assembled with the
same options as the run (`advection`, `rho_heating`, `heating_adiabatic`, ...).

The wall is the log**ical** side the geometry names for it (`nu_wall_sides(g)`:
`:yplus`/`:yminus` on the box-in-height geometries, `:xplus`/`:xminus` — the
circles — on the annulus), selected with `_on_logical_wall`.  `wall=:top` is the
cold/outer wall of the geometry's Nusselt pair and `wall=:bottom` the hot/inner
one; the returned value is the wall-integrated derivative in the box's
"positive upward" convention (+d/dz on the top, +d/dz on the bottom), which the
Nusselt normalisers divide by the wall measure.  On the box the method is the
historical physical-row body, verbatim.

`ev_field` is the elementwise entropy-viscosity field used by the energy step
that produced `theta` (one value per leaf, in `sq.leaves` order).  Passing it is
what makes the residual consistent with the EV-stabilised operator; without it
the CBF flux omits the EV diffusion and is biased low (2b: -4.5% in Nu_top).
The field cannot be recomputed from the post-step state because the EV weights
depend on `theta^n` and `theta^(n-1)`, and `theta^(n-1)` is no longer available.

The stage-0.5 audit's defect D2 was that this function had no geometry dispatch
at all: it always selected the logical `y` rows, so on the annulus it summed the
reaction of free (periodic-seam) rows — returning `1.9e-16` where the true wall
flux is `+1.05` — while the mapped `semquad_nu_cbf` normalisation already used
the mapped wall measure.
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
    g = sq.geom
    g isa CartesianBox &&
        return _semquad_cbf_theta_flux_box!(sq, qt, pr, theta, ux, uz;
                                            wall=wall, advection=advection,
                                            ev_field=ev_field, kwargs...)
    return _semquad_cbf_theta_flux_mapped!(g, sq, qt, pr, theta, ux, uz;
                                          wall=wall, advection=advection,
                                          ev_field=ev_field, kwargs...)
end

# Historical box body, verbatim: the wall rows are the logical `y` rows, which
# on the box are the physical Dirichlet walls.  Untouched bit for bit.
function _semquad_cbf_theta_flux_box!(sq::SemQuad, qt::Quadtree, pr::Params,
                                      theta::Vector{Float64},
                                      ux::Vector{Float64},
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

# Mapped body: the same residual sum, on the geometry's own Nusselt wall side
# (`nu_wall_sides`), with the wall node order and the "positive upward" sign
# flip of the box body.  On the box-in-height validation geometries the side is
# `:yplus`/`:yminus`, so the sum is identical to the box body on the box map.
function _semquad_cbf_theta_flux_mapped!(g::AbstractGeometry, sq::SemQuad,
                                         qt::Quadtree, pr::Params,
                                         theta::Vector{Float64},
                                         ux::Vector{Float64},
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
    sideb, sidet = nu_wall_sides(g)
    side = wall === :top ? sidet : sideb
    seen = Set{Int}()
    s = 0.0
    for lf in sq.leaves
        _on_logical_wall(lf, side, sq) || continue
        ids = sq.elem[lf]
        for ii in 1:n
            gid = _logical_wall_node(ids, n, side, ii)
            gid in seen && continue
            push!(seen, gid)
            s += r[gid]
        end
    end
    return wall === :top ? s : -s
end

"""
    semquad_nu_cbf(sq, qt, pr, theta, ux, uz; kwargs...)

Wall Nusselt numbers from the consistent boundary flux, in the same normalisation
as `semquad_nu_walls` (which is `base_ref - pointwise gradient`).  Pass
`ev_field` (the elementwise EV field of the step that produced `theta`) to make
the flux consistent with an EV-stabilised run.

The box body is the historical one, bit for bit (`top_ref - st/total_area`).
The generic mapped body keeps the same box convention (`top_ref - st/wt` with
the mapped wall measure `wt`, and `semquad_base_wall_refs(pr, g)` for the
references); the annulus dispatches to its per-wall benchmark normalisation,
`(top_ref - st/wt)/top_ref` and `(bot_ref + sb/wb)/bot_ref`, exactly the
`semquad_nu_walls` annulus definition (docs/dev/ANNULUS_REFS.md §6.2/§6.5).
Before the D2/D3 fix the mapped body summed the logical `y` rows and normalised
by the *Cartesian* base gradient, so the annulus pair came back `(1,1)` on a
conduction state whose true pair is `(0.900, 0.901)`.
"""
function semquad_nu_cbf(sq::SemQuad, qt::Quadtree, pr::Params,
                        theta::Vector{Float64}, ux::Vector{Float64},
                        uz::Vector{Float64}; ev_field::Union{Nothing,Vector{Float64}}=nothing,
                        kwargs...)
    g = sq.geom
    g isa CartesianBox ||
        return _semquad_nu_cbf_mapped(g, sq, qt, pr, theta, ux, uz;
                                      ev_field=ev_field, kwargs...)
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

"""
    _semquad_nu_cbf_mapped(g, sq, qt, pr, theta, ux, uz; ev_field=nothing, kwargs...)

Mapped method of [`semquad_nu_cbf`](@ref).  The consistent-boundary-flux
residual sum is geometry-generic once its wall rows come from
`nu_wall_sides(g)` (D2); the normalisation is each wall's physical measure
(`boundary_measure`) and the references are the geometry's base wall gradients
`semquad_base_wall_refs(pr, g)` (D3), not the Cartesian `semquad_base_gradient`
at `h = 0/1`.  The generic mapped form keeps the box's absolute convention
(`ref - flux/measure`, the normalisation `semquad_nu_walls` uses for the
box-in-height geometries); a mapped geometry whose benchmark normalisation is
per wall (the annulus) overrides it with [`_semquad_nu_cbf_mapped`](@ref)'s
`PolarAnnulus` method.
"""
function _semquad_nu_cbf_mapped(g::AbstractGeometry, sq::SemQuad,
                                qt::Quadtree, pr::Params,
                                theta::Vector{Float64}, ux::Vector{Float64},
                                uz::Vector{Float64};
                                ev_field::Union{Nothing,Vector{Float64}}=nothing,
                                kwargs...)
    st = semquad_cbf_theta_flux(sq, qt, pr, theta, ux, uz; wall=:top,
                                ev_field=ev_field, kwargs...)
    sb = semquad_cbf_theta_flux(sq, qt, pr, theta, ux, uz; wall=:bottom,
                                ev_field=ev_field, kwargs...)
    sideb, sidet = nu_wall_sides(g)
    wt = boundary_measure(g, sq, sidet)
    wb = boundary_measure(g, sq, sideb)
    bot_ref, top_ref = semquad_base_wall_refs(pr, g)
    return top_ref - st / wt, bot_ref - sb / wb
end

# Annulus override: the benchmark Nusselt pair divides each wall's perturbation
# flux by that wall's own conductive reference (the same definition
# `_semquad_nu_walls_mapped(g::PolarAnnulus, ...)` implements), so the CBF pair
# is `(top_ref - st/wt)/top_ref` and `(bot_ref + sb/wb)/bot_ref` with
# `sb = -flux_bottom` (the upward-positive CBF convention).
function _semquad_nu_cbf_mapped(g::PolarAnnulus, sq::SemQuad, qt::Quadtree,
                                pr::Params, theta::Vector{Float64},
                                ux::Vector{Float64}, uz::Vector{Float64};
                                ev_field::Union{Nothing,Vector{Float64}}=nothing,
                                kwargs...)
    st = semquad_cbf_theta_flux(sq, qt, pr, theta, ux, uz; wall=:top,
                                ev_field=ev_field, kwargs...)
    sb = semquad_cbf_theta_flux(sq, qt, pr, theta, ux, uz; wall=:bottom,
                                ev_field=ev_field, kwargs...)
    sideb, sidet = nu_wall_sides(g)
    wt = boundary_measure(g, sq, sidet)
    wb = boundary_measure(g, sq, sideb)
    bot_ref, top_ref = semquad_base_wall_refs(pr, g)
    return (top_ref - st / wt) / top_ref, (bot_ref - sb / wb) / bot_ref
end
