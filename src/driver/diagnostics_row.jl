# The per-step diagnostics row of the run history.
#
# `_step_diagnostics` computes every column of one `history.csv` row from the
# state the step just produced: the wall Nusselt pair, the RMS and maximum
# velocity, the mean temperature, the grid spacings, the divergence, the historical
# `closure` columns, the opt-in exact residual and the optional column families
# (latent-free closure, consistent boundary flux).  It also returns the scalar
# values the caller needs afterwards (`Nut`, `Nub`, `Vr`, `Ta`, `closure`), so the
# loop keeps no duplicate computation.  The function is pure with respect to the
# state: it reads the arrays and returns the row.
#
# The column layout is a public format: the first eighteen columns are fixed and
# every optional column is appended in a stated order, so an existing
# `history.csv` stays byte-identical when a card does not ask for the extras.

"""
    _nu_pair_defined(geom, pr) -> Bool

Whether the per-wall Nusselt pair `semquad_nu_walls` is defined on this
geometry.  The box always is (its reference gradients are `±(1 + H/2)` and
`H` is non-negative).  A mapped geometry needs the H-inclusive base conductive
flux `-dTbar/dh` to point outward at **both** walls, i.e. both entries of
`semquad_base_wall_refs(pr, geom)` must be positive; the annulus fails that when
the internal heating reverses the base flux at the inner circle, above

    H_crit = 4 / ((R1^2 - R0^2) - 2 R0^2 ln(R1/R0))   (~2.4127 for 1.22..2.22)

At `H = H_crit` the inner reference is exactly zero.  `semquad_nu_walls` refuses
such a state by design (`src/diagnostics/global.jl`); this predicate lets the
driver decide *before* the refusal whether the run's history can carry the pair
at all.  Diagnostic only: no arithmetic result of the solve reads it.
"""
@inline function _nu_pair_defined(::CartesianBox, ::Params)
    return true
end
@inline function _nu_pair_defined(geom::AbstractGeometry, pr::Params)
    bot_ref, top_ref = semquad_base_wall_refs(pr, geom)
    return bot_ref > 0.0 && top_ref > 0.0
end

"""
    _nu_pair_or_nan(sq, qt, theta, pr, nu_undefined) -> (Nut, Nub)

The two Nusselt columns of the history row.  `nu_undefined = :refuse` (the
default) always calls `semquad_nu_walls`, so an undefined mapped pair throws its
own `ArgumentError` verbatim.  `:nan` (opt-in, `run_semquad_simulation`) returns
`(NaN, NaN)` instead when `_nu_pair_defined` is false, keeping the rest of the
run (trajectory, every other column) alive; see `run.jl`'s `nu_undefined`
docstring.  On a defined card both settings are bit-identical.
"""
@inline function _nu_pair_or_nan(sq::SemQuad, qt::Quadtree,
                                 theta::Vector{Float64}, pr::Params,
                                 nu_undefined::Symbol)
    if nu_undefined === :nan && !_nu_pair_defined(sq.geom, pr)
        return NaN, NaN
    end
    return semquad_nu_walls(sq, qt, theta, pr)
end

function _step_diagnostics(sq::SemQuad, qt::Quadtree, pr::Params,
                           theta::Vector{Float64},
                           theta_prev::Vector{Float64},
                           theta_nm1_used::Vector{Float64},
                           ux::Vector{Float64}, uz::Vector{Float64},
                           dt::Float64, dt_prev::Float64, step::Int,
                           t::Float64, ev_field, phases, advection::Symbol,
                           eta_for_energy, W_last::Float64, U_last::Float64,
                           bal_last::Float64; extdiag::Bool, nu_cbf::Bool,
                           lat_theta_used, nu_undefined::Symbol=:refuse)
    Nut, Nub = _nu_pair_or_nan(sq, qt, theta, pr, nu_undefined)
    Vr = semquad_vrms(sq, qt, ux, uz)
    Ta = semquad_mean_T(sq, qt, theta, pr)
    g = sq.geom
    # CFL/hmin history (GEOM_DESIGN E15): the box keeps the historical
    # `minimum(leaf_hy)` bit for bit; a mapped geometry uses the physical extent
    # of each leaf along the height direction.
    hmin_hist = if g isa CartesianBox
        minimum(leaf_hy(sq, lf[1]) for lf in qt.leaves)
    else
        minimum_leaf_height(g, qt)
    end
    umax_hist = maximum(hypot.(ux, uz))
    div_max = semquad_max_divergence(sq, qt, ux, uz)
    Ibase = semquad_base_advection_integral(sq, qt, pr, ux, uz)
    # `closure` / `closure_nolatent` are the **historical** columns and their
    # layout must stay byte-identical: on the box they are
    # `Nut - Nub - (H - Ibase)` (plus `latent_dgdt/A` when latent heat is on),
    # i.e. the *steady* flux balance evaluated at a transient state.  They are
    # **not** a flux error and do not close for any Nu definition: the `(DEC)`
    # number equals `-dH/dt` plus three
    # bookkeeping terms (the skew-vs-PDE latent/advection difference, the
    # CBF's `O(h)` wall functional and the solver's constraint residual),
    # divided by `A`.  Read it as the enthalpy-rate residual of the
    # historical formula, never as "the budget misses X".  The exact
    # residual of the quoted pointwise Nu pair is the opt-in `resid_A`
    # column below (`extdiag=true`), which is
    # zero for the time-step CBF and `O(h)` for the pointwise family.
    #
    # **Mapped normalisation (stage-0.5 audit defect B5).**  The box formula is
    # an area-1 bookkeeping identity: for the box base state
    # `top_ref - bot_ref = H`, so substituting the box Nusselt definition
    # `Nut - Nub = H - flux/A_box` gives
    #
    #     closure_box = (Ibase + latent_dgdt - flux) / A_box,
    #
    # with `flux` the net outward wall integral and `A_box = total_area(qt)`
    # (the box area, which equals the wall length only because the box height
    # is 1).  On a mapped domain `H` and `Ibase` are volume-integrated rates
    # while `Nut`/`Nub` are per-wall ratios, so the raw `Nut - Nub - H` form
    # stops being a budget residual (the audit measured `-1.0043` on the
    # annulus at `H = 1`, dominated by `-H`).  The mapped column is the same
    # physical statement evaluated with the physical measure:
    #
    #     closure_mapped = (Ibase + latent_dgdt - flux) / A_phys,
    #
    # `A_phys = total_measure(g, sq)`; it reduces to the box formula on the box
    # map because there `A_phys = A_box = total_area(qt)` and the box Nusselt
    # identities hold, so the box branch below is kept bit for bit and the
    # mapped branch is the general statement.  The box's own doc record of the
    # column is unchanged — this is a mapped-only
    # branch.
    #
    # The latent bookkeeping of the historical formula: the assembled
    # energy equation carries the apparent capacity `1 + sum_i L_i
    # Gamma_T,i` and the source `-sum_i L_i Gamma_z,i u_z`, so the
    # volume-integrated *steady* balance has two extra terms:
    #   int L_i Gamma_z,i u_z dV        (latent source)
    #   int L_i Gamma_T,i u.grad theta  (capacity's advective correction)
    # `latent_int` (the latent enthalpy) is NOT one of them: its time
    # derivative is the transient, not a steady source.
    if g isa CartesianBox
        closure_nolatent = Nut - Nub - (pr.H - Ibase)
        closure = closure_nolatent
        if semquad_latent_active(pr, phases)
            lb = semquad_heat_budget(sq, qt, pr, theta; phases=phases,
                                     ux=ux, uz=uz, eta_fun=eta_for_energy)
            # `latent_dgdt` is a Float64 here: the velocity is always supplied.
            # The Nusselt-unit normalisation is the box's `total_area(qt)` (bit
            # for bit); a mapped geometry uses the physical measure below.
            closure += (lb.latent_dgdt::Float64) / _nusselt_area(sq, qt)
        end
    else
        # Mapped: the same physical steady-balance residual with the physical
        # measure.  The heat budget already supplies the mapped net wall flux
        # and the mapped base-advection / latent terms (its own quadrature), so
        # the column cannot drift from the budget it claims to report.
        lbm = semquad_heat_budget(sq, qt, pr, theta; phases=phases,
                                  ux=ux, uz=uz, eta_fun=eta_for_energy)
        Aphys = total_measure(g, sq)
        closure_nolatent = (lbm.base_adv - lbm.flux) / Aphys
        closure = closure_nolatent
        lgd = lbm.latent_dgdt
        lgd === nothing || (closure += (lgd::Float64) / Aphys)
    end
    # Opt-in exact residual of the quoted pointwise pair, per unit area:
    #   R/A = (F_pt - (dH/dt + sum(C theta) + Ibase + latent_src)) / A
    # with the operator rebuilt from this step's stored EV field and its
    # apparent-capacity temperature (`semquad_exact_residual`).  Default off: no
    # column, no extra
    # assembly, and every existing history row stays byte-identical.
    resid_A = NaN
    if extdiag
        ex = semquad_exact_residual(sq, qt, pr, theta, theta_prev,
                                    theta_nm1_used, ux, uz, dt, dt_prev,
                                    step == 1, ev_field;
                                    advection=advection, phases=phases,
                                    latent_theta=lat_theta_used,
                                    eta_fun=eta_for_energy)
        resid_A = ex.R_over_A
    end
    usrms_surf, umax_surf = semquad_surface_velocity(sq, qt, ux, uz; wall=:top)
    # W_last / U_last come from solve_flow!, where they are evaluated with the
    # same (theta, eta) pair as the Stokes solve; see the comment there.
    row = [t, dt, Nut, Nub, Vr, Ta, Float64(sq.N),
           Float64(length(qt.leaves)), div_max, hmin_hist, umax_hist,
           umax_hist * dt / hmin_hist, closure,
           usrms_surf, umax_surf, W_last, U_last, bal_last]
    if semquad_latent_active(pr, phases)
        # The latent-free definition, appended as column 19 only when the
        # latent term is active, so the default history stays 18 wide (the
        # same rule as the nu_cbf columns below) and every non-latent run
        # keeps its historical history byte for byte.
        push!(row, closure_nolatent)
    end
    if nu_cbf
        # consistent boundary flux (see semquad_flux.jl): appended only when
        # requested, so the default history stays 18 wide
        rho_ref_cbf = (pr.rho_ref_weight && pr.refstate !== nothing) ?
                      (y -> semquad_rho(pr.refstate, y) * pr.refstate.cp) : nothing
        ntc, nbc = semquad_nu_cbf(sq, qt, pr, theta, ux, uz;
                                  advection=advection,
                                  ev_field=ev_field,
                                  rho_heating=pr.rho_heating,
                                  rho_cp=pr.rho_cp,
                                  heating_adiabatic=pr.heating_adiabatic,
                                  heating_dissipation=pr.heating_dissipation,
                                  eta_fun=eta_for_energy,
                                  rho_ref=rho_ref_cbf,
                                  phases=phases)
        push!(row, ntc)
        push!(row, nbc)
    end
    if extdiag
        # exact residual of the pointwise pair (appended only when
        # requested; see the block above).
        push!(row, resid_A)
    end
    return (row=row, Nut=Nut, Nub=Nub, Vr=Vr, Ta=Ta, closure=closure)
end
