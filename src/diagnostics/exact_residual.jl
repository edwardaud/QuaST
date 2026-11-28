# Exact constrained-residual diagnostic (box and mapped base
# curvature integral).
"""
    _semquad_base_curvature_integral(g, sq, pr) -> Float64

Volume integral of the mapped base-state Laplacian `lap(Tbar(h))` (plus the
absorbed `+H` for the Boussinesq base), `int _mapped_base_curvature dV`, on the
quadrature the mapped diagnostics use.  This is the term the mapped energy
assembly adds as a source and the mapped entropy-viscosity residual subtracts;
the mapped exact residual of [`semquad_exact_residual`](@ref) must subtract it
from its identity.  It is identically zero on the box (`|grad h| = 1`,
`lap y = 0`), which is why the box body does not evaluate it.  Mapped-only.
"""
function _semquad_base_curvature_integral(g::AbstractGeometry, sq::SemQuad,
                                          pr::Params)
    p = sq.p
    fd = make_fedata(p, p - 1)
    s = 0.0
    for lf in sq.leaves
        m = leaf_metric(g, sq, lf, fd)
        for qy in 1:fd.nq, qx in 1:fd.nq
            x, y = quad_point(m, sq, lf, fd, qx, qy)
            s += quad_measure(m, fd, qx, qy) * _mapped_base_curvature(pr, g, x, y)
        end
    end
    return s
end

"""
    semquad_exact_residual(sq, qt, pr, theta, theta_n, theta_nm1, ux, uz,
                           dt, dt_prev, firststep, ev_field; advection=:skew,
                           phases=PhaseTransition[], latent_theta=nothing,
                           eta_fun=nothing)

The **exact** residual of the quoted pointwise Nusselt pair against the scheme's
own discrete identity (eq. `(ID1)`):

    R = F_pt - (dH/dt + sum(C theta) + Ibase + latent_src)
    F_pt = flux_top - flux_bot,   dH/dt = 1'(ca M theta - cb M theta_n + cc M theta_nm1)

where `M, K, C` are rebuilt with the *same* operator the step used (the stored
elementwise EV field and the same apparent-capacity temperature) and
`ca/cb/cc` are the BDF2 coefficients of that step.  `R/A` is in Nusselt units.
It vanishes (to the linear solver's floor) for the time-step CBF and is `O(h)`
for the pointwise pair, so a non-zero `R/A` at a transient state is a property
of the pointwise family, **not** an error of the run; the historical `closure`
column is a different
quantity and does not close for any definition (NUSSELT_AUDIT §4).

`theta_n`/`theta_nm1` are the two previous BDF2 levels of the step that produced
`theta`; `ev_field` is the elementwise EV field of that step (`nothing` when
`ev == 0`), so the operator is the one that produced the state and no EV field
is recomputed.  `latent_theta` is the temperature at which the step evaluated
the apparent capacity (`nothing` means `theta_n`, the lagged/default choice);
pass it explicitly when `maxpic > 1`, where it is the last iterate.

Returns a named tuple with `R`, `R_over_A = R / total_area(qt)` and the terms of
the identity (`flux`, `dHdt`, `sumC`, `Ibase`, `latent_src`, `rhs`).  This is a
pure diagnostic: no solver path calls it unless the caller asks for it (the
driver's opt-in `extdiag=true`).
"""
function semquad_exact_residual(sq::SemQuad, qt::Quadtree, pr::Params,
                                theta::Vector{Float64},
                                theta_n::Vector{Float64},
                                theta_nm1::Vector{Float64},
                                ux::Vector{Float64}, uz::Vector{Float64},
                                dt::Float64, dt_prev::Float64,
                                firststep::Bool,
                                ev_field::Union{Nothing,Vector{Float64}};
                                advection::Symbol=:skew,
                                phases::Vector{PhaseTransition}=PhaseTransition[],
                                latent_theta::Union{Nothing,Vector{Float64}}=nothing,
                                eta_fun::Union{Nothing,Function}=nothing)
    rho_ref = (pr.rho_ref_weight && pr.refstate !== nothing) ?
              (y -> semquad_rho(pr.refstate, y) * pr.refstate.cp) : nothing
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
                                         ev=0.0, ev_field=ev_field)
    # The BDF2 mass coefficients of the step that produced `theta` (the first
    # step is backward Euler, cc == 0).
    if firststep
        ca = 1.0 / dt; cb = 1.0 / dt; cc = 0.0
    else
        w = dt / dt_prev
        ca = (2.0 * w + 1.0) / (1.0 + w) / dt
        cb = (1.0 + w) / dt
        cc = w * w / (1.0 + w) / dt
    end
    dHdt = sum(M * (ca .* theta .- cb .* theta_n .+ cc .* theta_nm1))
    sumC = sum(C * theta)
    bud = semquad_heat_budget(sq, qt, pr, theta; phases=phases, ux=ux, uz=uz,
                              eta_fun=eta_fun)
    Ibase = semquad_base_advection_integral(sq, qt, pr, ux, uz)
    latent_src = bud.latent_src === nothing ? 0.0 : bud.latent_src
    rhs = dHdt + sumC + Ibase + latent_src
    # A mapped geometry's assembled source carries the physical Laplacian of the
    # base state `lap(Tbar(h))` (`_mapped_base_curvature`), which is identically
    # zero on the box (|grad h| = 1, lap y = 0) but not on a curved map; the
    # identity (ID1) then reads
    # `Qc = dHdt + sumC + Ibase + latent_src - int lap(Tbar(h)) dV`.  The box
    # body keeps the historical `rhs` bit for bit.
    g = sq.geom
    if !(g isa CartesianBox)
        rhs -= _semquad_base_curvature_integral(g, sq, pr)
    end
    # The `R/A` normalisation is the Nusselt-unit "area": the box keeps
    # `total_area(qt)` bit for bit (height 1 makes the area equal the wall
    # length); a mapped geometry uses the top wall's physical measure
    # (`_nusselt_area`, GEOM_DESIGN §6.6).
    area = _nusselt_area(sq, qt)
    R = bud.flux - rhs
    return (R=R, R_over_A=R / area, flux=bud.flux, dHdt=dHdt, sumC=sumC,
            Ibase=Ibase, latent_src=latent_src, rhs=rhs, area=area)
end

