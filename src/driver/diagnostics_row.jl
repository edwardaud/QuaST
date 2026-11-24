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

function _step_diagnostics(sq::SemQuad, qt::Quadtree, pr::Params,
                           theta::Vector{Float64},
                           theta_prev::Vector{Float64},
                           theta_nm1_used::Vector{Float64},
                           ux::Vector{Float64}, uz::Vector{Float64},
                           dt::Float64, dt_prev::Float64, step::Int,
                           t::Float64, ev_field, phases, advection::Symbol,
                           eta_for_energy, W_last::Float64, U_last::Float64,
                           bal_last::Float64; extdiag::Bool, nu_cbf::Bool,
                           lat_theta_used)
    Nut, Nub = semquad_nu_walls(sq, qt, theta, pr)
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
    # layout must stay byte-identical: they are `Nut - Nub - (H - Ibase)`
    # (plus `latent_dgdt/A` when latent heat is on), i.e. the *steady*
    # flux balance evaluated at a transient state.  They are **not** a flux
    # error and do not close for any Nu definition: by `(DEC)` of
    # docs/dev/VALIDATION.md §4 the number equals `-dH/dt` plus three
    # bookkeeping terms (the skew-vs-PDE latent/advection difference, the
    # CBF's `O(h)` wall functional and the solver's constraint residual),
    # divided by `A`.  Read it as the enthalpy-rate residual of the
    # historical formula, never as "the budget misses X".  The exact
    # residual of the quoted pointwise Nu pair is the opt-in `resid_A`
    # column below (`extdiag=true`, docs/dev/VALIDATION.md §3.2), which is
    # zero for the time-step CBF and `O(h)` for the pointwise family.
    #
    # The latent bookkeeping of the historical formula: the assembled
    # energy equation carries the apparent capacity `1 + sum_i L_i
    # Gamma_T,i` and the source `-sum_i L_i Gamma_z,i u_z`, so the
    # volume-integrated *steady* balance has two extra terms
    # (docs/dev/PHYSICS.md §7.3):
    #   int L_i Gamma_z,i u_z dV        (latent source)
    #   int L_i Gamma_T,i u.grad theta  (capacity's advective correction)
    # `latent_int` (the latent enthalpy) is NOT one of them: its time
    # derivative is the transient, not a steady source.
    closure_nolatent = Nut - Nub - (pr.H - Ibase)
    closure = closure_nolatent
    if semquad_latent_active(pr, phases)
        lb = semquad_heat_budget(sq, qt, pr, theta; phases=phases,
                                 ux=ux, uz=uz, eta_fun=eta_for_energy)
        # `latent_dgdt` is a Float64 here: the velocity is always supplied.
        # The Nusselt-unit normalisation is the box's `total_area(qt)` (bit for
        # bit); a mapped geometry uses the top wall measure (see
        # `semquad_exact_residual`).
        closure += (lb.latent_dgdt::Float64) / _nusselt_area(sq, qt)
    end
    # Opt-in exact residual of the quoted pointwise pair, per unit area:
    #   R/A = (F_pt - (dH/dt + sum(C theta) + Ibase + latent_src)) / A
    # with the operator rebuilt from this step's stored EV field and its
    # apparent-capacity temperature (`semquad_exact_residual`,
    # docs/dev/VALIDATION.md §2.3/§3.2).  Default off: no column, no extra
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
        # requested; see the block above and docs/dev/VALIDATION.md §3.2)
        push!(row, resid_A)
    end
    return (row=row, Nut=Nut, Nub=Nub, Vr=Vr, Ta=Ta, closure=closure)
end
