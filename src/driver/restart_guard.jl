# The run configuration and the restart guard of the production entry point.
#
# `run_semquad_simulation` records every setting that changes the trajectory in
# a `run_config` dictionary: the checkpoint stores it and a restart compares the
# stored values with the live run's.  `_run_config` builds that dictionary;
# `_restart_guard` then loads the `restart_from` checkpoint, refuses a state
# written under another physics revision, warns for a file that predates
# revision tracking, compares the stored configuration, checks the temperature
# field and returns the resumed `(qt, p, theta0)`.  Both are moved verbatim out
# of the entry point's body, which now only calls them.
#
# The configuration comparison separates the *physics parameter values* (the
# `_RESTART_PHYSICS_KEYS` below: `Ra`, the viscosity law and its parameters, the
# wall temperatures, `bc`, the yield/latent/phase/reference-state settings, ...)
# from the numerics and trajectory settings (AMR, advection, EV, Picard, CFL,
# dt bounds, ...).  A numerics change is always refused; a physics change is
# refused by default (the restart was written by other equations) and can be
# continued only with the explicit `restart_physics=:allow` protocol, which
# warns once and tells the driver to reset the first resumed time step (G291).
# The `physics_revision` check and the geometry check stay hard refusals in
# every mode: they are not parameter values a user can intentionally change.

"""
    _wall_value_marker(t)

Stable comparable marker for one per-wall tuple of function-or-`nothing`
entries (`wall_u`, `wall_un`): one `"none"`/`"custom"` token per wall,
comma-joined in the `(xminus, xplus, yminus, yplus)` order.  A function has no
canonical value to compare, so -- exactly like `eta_multiplier` and
`rho_multiplier` -- the guard records only *whether* a custom value is present,
not which one: a restart that swaps one function for another is not detectable,
and the docs state that limit.
"""
_wall_value_marker(t::NTuple{4,Union{Nothing,Function}}) =
    join((e === nothing ? "none" : "custom" for e in t), ",")

"""
    _RESTART_PHYSICS_KEYS

The `run_config` keys that record **physics parameter values** rather than
numerics or trajectory settings.  The restart guard compares every recorded key
against the live configuration; a differing *numerics* key (AMR cadence and
fractions, advection, EV, Picard controls, CFL, dt bounds, `mass_form`,
`stokes_scaling`, `latent_scheme`) is always refused, and so is a differing
`geometry` or `physics_revision`.  A differing key in this tuple means the
stored state was produced by *other equations*: the default
`restart_physics=:refuse` refuses it with the differing values, and
`restart_physics=:allow` continues deliberately (one warning, first resumed
`dt` reset to `dt0`, everything else kept; G291).

Why a separate tuple: the values are recorded either way (a checkpoint carries
them so the comparison *can* be made), but only the physics keys may be
overridden by the explicit protocol.  Not every entry changes the trajectory in
every card -- e.g. `deta_T` is inert for `eta_law=:expTz`, `supg`,
`div_penalty` and `eta_custom` are currently unwired `Params` fields -- but the
guard is deliberately conservative: a field of `Params` that could reach the
equations is listed, so wiring one up later cannot silently change what a
restart continues.
"""
const _RESTART_PHYSICS_KEYS = (
    # thermal forcing, heating and the base state
    "Ra", "H", "H_half_life", "Di", "heating_adiabatic", "heating_dissipation",
    "rho_heating", "rho_cp", "rho_ref_weight", "compress_p", "compress_gamma",
    # viscosity law and its parameters
    "eta_law", "deta_T", "eta_b", "eta_c", "eta_E", "eta_T0", "eta_const",
    "eta_min", "eta_max", "eta_soft", "eta_ref", "eta_custom",
    "eta_multiplier", "rho_multiplier",
    # boundary conditions and wall data
    "bc", "wall_bc", "wall_u", "wall_un", "wall_t", "wall_tn",
    "Twall_top", "Twall_bot", "Twall_left", "Twall_right", "side_thermal",
    # rheology: Tosi law, composite law, yielding, weakening, viscoelasticity
    "tosi_etaT", "tosi_etaZ", "tosi_eta_star", "tosi_sigma_y", "tosi_eta_init",
    "tosi_eta0", "tosi_eta_min", "tosi_eta_max", "tosi_z_is_depth",
    "comp_Ad", "comp_Ed", "comp_Vd", "comp_As", "comp_Es", "comp_Vs", "comp_n",
    "comp_Tcut", "comp_eps_min", "comp_rule",
    "yield_c", "yield_phi", "yield_taumax", "yield_eps_min", "yield_damper",
    "yield_c_grad", "yield_mu_grad",
    "weak_f", "weak_eps_ref", "weak_heal_time", "weak_mode", "weak_chi0",
    "shear_modulus", "deviatoric",
    # phase transitions, latent heat and reference-state physics
    "phases", "latent_L", "latent_mode",
    "refstate", "ref_Tsurf", "ref_alpha", "ref_cp", "ref_g", "ref_rho0",
    "ref_nz", "ref_K0", "ref_K0prime",
    # numerical terms that are physical-model coefficients of Params
    "supg", "div_penalty",
)

"""Is `k` a physics parameter value recorded in `_RESTART_PHYSICS_KEYS`?"""
_restart_physics_key(k::AbstractString) = k in _RESTART_PHYSICS_KEYS

"""
    _run_config(pr; settings...)

Build the `run_config` dictionary of one run: the canonical value of every
setting that changes the trajectory, taken from `pr` and from the driver
keywords listed in the signature (each is recorded under the key of the same
name in the body).  The checkpoint stores it and [`_restart_guard`](@ref)
compares it, so a resumed run cannot silently continue with a different setup;
a key an older checkpoint does not carry is skipped by the comparison, which is
why adding a key here does not invalidate historical files.
"""
function _run_config(pr::Params;
                     amr_every::Int,
                     refine_frac::Float64,
                     coarsen_frac::Float64,
                     amr_passes::Int,
                     max_leaves::Int,
                     max_level::Int,
                     wall_min_level::Int,
                     wall_strip::Float64,
                     symmetry::Symbol,
                     transfer_mode::Symbol,
                     amr_hysteresis::Int,
                     amr_coarsen_gate::Bool,
                     amr_cap_aware::Bool,
                     amr_indicator::Symbol,
                     hotspot_weight::Float64,
                     hotspot_frac::Float64,
                     hotspot_min_level::Int,
                     hotspot_mode::Symbol,
                     hotspot_zmin::Float64,
                     zero_side_eta::Bool,
                     side_strip::Float64,
                     top_wall_level::Int,
                     top_wall_frac::Float64,
                     advection::Symbol,
                     ev::Float64,
                     ev_cmax::Float64,
                     ev_alpha::Int,
                     maxpic::Int,
                     pic_tol::Float64,
                     pic_relax::Float64,
                     pic_mode::Symbol,
                     pic_reject::Bool,
                     cfl::Float64,
                     dtmax::Float64,
                     dtmin::Float64,
                     mass_form::Symbol,
                     stokes_scaling::Symbol,
                     latent_scheme::Symbol,
                     phases::Vector{PhaseTransition},
                     weak_chi0::Float64,
                     geom::AbstractGeometry)
    run_config = Dict{String,Any}(
        "amr_every" => amr_every,
        "refine_frac" => refine_frac, "coarsen_frac" => coarsen_frac,
        "amr_passes" => amr_passes,
        "max_leaves" => max_leaves, "max_level" => max_level,
        "wall_min_level" => wall_min_level, "wall_strip" => wall_strip,
        "symmetry" => symmetry, "transfer_mode" => transfer_mode,
        # The balanced cycle is the only AMR policy; the key stays so a
        # checkpoint written before the classic policy was removed still
        # matches its recorded `balanced`. A file that
        # predates the key skips the comparison, like every key added later.
        "amr_policy" => "balanced", "amr_hysteresis" => amr_hysteresis,
        # The coarsening_allowed pre-check changes the mesh trajectory too,
        # so a restart must not switch it silently either; old checkpoints do not
        # carry the key and skip the comparison, like every key added later.
        "amr_coarsen_gate" => amr_coarsen_gate,
        # Opt-in cap-aware fallback changes the mesh trajectory when enabled;
        # old checkpoints lack the key and skip the comparison, new ones
        # cannot silently switch the mode on resume.
        "amr_cap_aware" => amr_cap_aware,
        "amr_indicator" => amr_indicator,
        "hotspot_weight" => hotspot_weight, "hotspot_frac" => hotspot_frac,
        "hotspot_min_level" => hotspot_min_level, "hotspot_mode" => hotspot_mode,
        "hotspot_zmin" => hotspot_zmin, "zero_side_eta" => zero_side_eta,
        "side_strip" => side_strip, "top_wall_level" => top_wall_level,
        "top_wall_frac" => top_wall_frac, "advection" => advection,
        "ev" => ev, "ev_cmax" => ev_cmax, "ev_alpha" => ev_alpha,
        "maxpic" => maxpic, "pic_tol" => pic_tol, "pic_relax" => pic_relax,
        "pic_mode" => pic_mode, "pic_reject" => pic_reject,
        "cfl" => cfl, "dtmax" => dtmax, "dtmin" => dtmin,
        "mass_form" => mass_form, "deviatoric" => pr.deviatoric,
        # Solver-side equilibration of the direct saddle (`:none` is the
        # historical path bit for bit).  It can change the answer of a failing
        # solve, so a restart must not switch it silently; old checkpoints do
        # not carry the key and are read as `none` by the guard below.
        "stokes_scaling" => stokes_scaling,
        # A decaying heating rate changes the trajectory as much as any of the
        # above; old checkpoints do not carry the key and skip the comparison.
        "H_half_life" => pr.H_half_life,
        # A weak zone is part of the rheology; a restart must not lose it.  The
        # key stores only whether one is set, because a function has no value to
        # compare; a user who changes it is told to start a fresh run.
        "eta_multiplier" => pr.eta_multiplier === nothing ? "none" : "custom",
        # The density hook changes the buoyancy (and, with a reference state,
        # the anelastic mass constraint), so it is part of the physics a restart
        # must not lose.  Same convention as `eta_multiplier`: a function has no
        # value to compare, only whether one is set.
        "rho_multiplier" => pr.rho_multiplier === nothing ? "none" : "custom",
        # The side walls are part of the physics of the run.
        "side_thermal" => pr.side_thermal,
        "Twall_left" => pr.Twall_left, "Twall_right" => pr.Twall_right)
    # Physics parameters that were missing from the guard before G290/G291.
    # The checkpoint recorded the numerics and *some* of the physics (viscosity
    # extras, yield, phases, ...), so a restart could silently change `Ra`, the
    # viscosity law, the wall temperatures or `bc` and continue the stored state
    # under other equations.  These are the `Params` values that were missing;
    # the guard classifies them with `_RESTART_PHYSICS_KEYS`, so a numerics
    # change stays refused in every mode while a physics change needs the
    # explicit `restart_physics=:allow` protocol.  Old checkpoints do not carry
    # the keys and skip the comparison (`haskey(run_config, k) || continue` in
    # `_restart_guard`), which is the historical silent behaviour.
    run_config["Ra"] = pr.Ra
    run_config["eta_law"] = pr.eta_law
    run_config["bc"] = pr.bc
    run_config["deta_T"] = pr.deta_T
    run_config["eta_b"] = pr.eta_b
    run_config["eta_c"] = pr.eta_c
    run_config["eta_E"] = pr.eta_E
    run_config["eta_T0"] = pr.eta_T0
    run_config["eta_const"] = pr.eta_const
    run_config["eta_min"] = pr.eta_min
    run_config["eta_max"] = pr.eta_max
    # Function-valued law override: only its presence is comparable, the
    # `eta_multiplier`/`rho_multiplier` marker convention.
    run_config["eta_custom"] = pr.eta_custom === nothing ? "none" : "custom"
    run_config["eta_soft"] = pr.eta_soft
    run_config["Twall_top"] = pr.Twall_top
    run_config["Twall_bot"] = pr.Twall_bot
    run_config["rho_ref_weight"] = pr.rho_ref_weight
    run_config["compress_p"] = pr.compress_p
    run_config["compress_gamma"] = pr.compress_gamma
    # Inert `Params` fields today (declared and stored, not read by the driver
    # yet): recorded so a future wiring cannot silently change a restart.
    run_config["supg"] = pr.supg
    run_config["div_penalty"] = pr.div_penalty
    # The shape and normalisation of the viscosity law change the trajectory just
    # as much as the AMR settings, so they are validated too (only for :tosi;
    # the other laws are fully described by their Params fields).
    if pr.eta_law === :tosi
        tl = pr.tosi
        run_config["tosi_etaT"] = tl.etaT
        run_config["tosi_etaZ"] = tl.etaZ
        run_config["tosi_eta_star"] = tl.eta_star
        run_config["tosi_sigma_y"] = tl.sigma_y
        run_config["tosi_eta_init"] = tl.eta_init
        run_config["tosi_eta0"] = tl.eta0
        run_config["tosi_eta_min"] = tl.eta_min
        run_config["tosi_eta_max"] = tl.eta_max
        run_config["tosi_z_is_depth"] = tl.z_is_depth
    elseif pr.eta_law === :composite
        # The composite law's parameters are not reconstructible from the
        # Params fields the other laws use, so store them explicitly: a restart
        # must not silently continue with a different rheology.
        run_config["comp_Ad"] = pr.comp_Ad
        run_config["comp_Ed"] = pr.comp_Ed
        run_config["comp_Vd"] = pr.comp_Vd
        run_config["comp_As"] = pr.comp_As
        run_config["comp_Es"] = pr.comp_Es
        run_config["comp_Vs"] = pr.comp_Vs
        run_config["comp_n"] = pr.comp_n
        run_config["comp_Tcut"] = pr.comp_Tcut
        run_config["comp_eps_min"] = pr.comp_eps_min
        # The combination rule changes the rheology (isostress vs isostrain), so
        # a restart must not switch it.  Old
        # checkpoints predate this key and skip the comparison, as with every
        # other key added after them.
        run_config["comp_rule"] = pr.comp_rule
        # `eta_T0`, `eta_min` and `eta_max` are recorded unconditionally above
        # (they belong to the law fields the other laws use too), so the
        # composite branch does not repeat them.
    end
    # Heating terms and the reference state also change the trajectory.
    run_config["Di"] = pr.Di
    run_config["heating_adiabatic"] = pr.heating_adiabatic
    run_config["heating_dissipation"] = pr.heating_dissipation
    run_config["rho_heating"] = pr.rho_heating
    run_config["rho_cp"] = pr.rho_cp
    run_config["H"] = pr.H
    # Drucker-Prager yield surface, including the depth-dependent profiles
    # (`yield_c_grad`, `yield_mu_grad`) and the
    # strain-weakening parameters (§9).  They change the trajectory exactly like
    # the viscosity law, so a restart must not silently continue with another
    # yield surface.  They are stored unconditionally; old checkpoints without
    # these keys skip the comparison (`haskey(run_config, k) || continue`).
    run_config["yield_c"] = pr.yield_c
    run_config["yield_phi"] = pr.yield_phi
    run_config["yield_taumax"] = pr.yield_taumax
    run_config["yield_eps_min"] = pr.yield_eps_min
    run_config["yield_damper"] = pr.yield_damper
    run_config["yield_c_grad"] = pr.yield_c_grad
    run_config["yield_mu_grad"] = pr.yield_mu_grad
    # Strain-weakening parameters and the initial value
    # of the field.  A checkpoint also stores the field itself (`weak`), so a
    # restart continues the same weakening state; these keys guard the physics.
    run_config["weak_f"] = pr.weak_f
    run_config["weak_eps_ref"] = pr.weak_eps_ref
    run_config["weak_heal_time"] = pr.weak_heal_time
    run_config["weak_mode"] = pr.weak_mode
    run_config["weak_chi0"] = weak_chi0
    # Latent heat (apparent heat capacity) and the phase list: both change the
    # trajectory, so a restart must not switch them silently.  The phase list is
    # stored as a canonical string; old checkpoints without these keys simply
    # skip them (`haskey(run_config, k) || continue` below).
    # Viscoelasticity and the per-wall tuple change the trajectory, so a restart
    # must not switch them silently.  The guard below tolerates keys an older
    # checkpoint does not carry (`haskey(run_config, k) || continue`), so adding
    # these does not invalidate existing files.
    # The physics revision is a *behaviour* version, not a parameter value:
    # 958af5e and 8c27a02 changed what the same `shear_modulus` means by
    # changing the Maxwell stress update, which no value comparison can see.
    # `PHYSICS_REVISION` is the single source of truth; its changelog is
    # `src/physics/revision.jl`.  The restart guard below refuses a file from
    # another revision and warns for a file that predates the key (it is
    # compared there with its own message, not by the generic loop).
    run_config["physics_revision"] = PHYSICS_REVISION
    run_config["shear_modulus"] = pr.shear_modulus
    # The Rayleigh-number reference viscosity divides every law viscosity in the
    # flow solve, so it changes the trajectory like any other physics parameter
    # (G257).  Old checkpoints do not carry the key and are resumed with the
    # default 1.0, which is the historical behaviour; a new checkpoint records it.
    run_config["eta_ref"] = pr.eta_ref
    run_config["wall_bc"] = join(string.(pr.wall_bc), ",")
    # The wall *motion* is part of the trajectory.  `wall_u` was a pre-existing
    # gap here (only `wall_bc` was recorded; BC_DESIGN.md §3.6 item 4) and
    # `wall_un` is new with G258.  Both are per-wall tuples of functions, which
    # cannot be serialised or compared, so each gets the
    # `eta_multiplier`/`rho_multiplier` marker: `none`/`custom` per wall.  The
    # guard tolerates keys an older checkpoint does not carry, so adding these
    # does not invalidate historical files.
    run_config["wall_u"] = _wall_value_marker(pr.wall_u)
    run_config["wall_un"] = _wall_value_marker(pr.wall_un)
    # BC stage 2/3: the per-wall *traction* changes the load vector, so it
    # changes the trajectory exactly like the wall motion.  Same marker (a
    # closure has no comparable value): `wall_t` is the tangential and
    # `wall_tn` the normal component.
    run_config["wall_t"] = _wall_value_marker(pr.wall_t)
    run_config["wall_tn"] = _wall_value_marker(pr.wall_tn)
    run_config["latent_L"] = pr.latent_L
    run_config["latent_mode"] = pr.latent_mode
    run_config["latent_scheme"] = latent_scheme
    run_config["phases"] = isempty(phases) ? "none" :
        join([string(pt.z0, ",", pt.width, ",", pt.slope, ",", pt.Tref, ",",
                    pt.drho, ",", pt.B) for pt in phases], ";")
    if pr.refstate === nothing
        run_config["refstate"] = "none"
    else
        rs = pr.refstate
        run_config["refstate"] = rs.prof === nothing ? "adiabat" :
                                 "eos:" * string(rs.prof.eos)
        run_config["ref_Tsurf"] = rs.Tsurf
        run_config["ref_alpha"] = rs.alpha
        run_config["ref_cp"] = rs.cp
        run_config["ref_g"] = rs.g
        run_config["ref_rho0"] = rs.rho0
        if rs.prof !== nothing
            run_config["ref_nz"] = rs.prof.n
            run_config["ref_K0"] = rs.prof.K0
            run_config["ref_K0prime"] = rs.prof.K0prime
        end
    end
    # The geometry/mapping layer is part of the trajectory: the same tree on
    # another geometry is another physical problem (the box path is the
    # identity map; a mapped geometry moves every leaf and every wall measure).
    # The record is the canonical `geometry_identity` string, so it carries the
    # kind *and* every shape parameter; a restart under a different geometry is
    # refused by the same key-by-key comparison below.  Old checkpoints have no
    # key and were necessarily written by the box (stage 0 has no other
    # production geometry); the guard reads them as the box and refuses a
    # mapped run's attempt to continue one.
    run_config["geometry"] = geometry_identity(geom)
    return run_config
end

"""
    _restart_guard(qt, p, theta0, run_config, restart_from;
                   restart_physics::Symbol = :refuse)

Apply the `restart_from` checkpoint to a run that is about to start.  With an
empty `restart_from` this is a no-op and `restart === nothing`.  Otherwise it
loads the file and validates it before any space is built: the stored physics
revision must equal `PHYSICS_REVISION` (a file without the key is resumed with
a warning that names the stress history it carries), the stored `run_config`
must match the live one key by key (`stokes_scaling` absent means `none`), and
the loaded `theta` must be finite.  Returns `(restart, qt, p, theta0,
physics_changed)` -- the loaded checkpoint, the mesh, polynomial degree and
perturbation to resume from, and whether the physics parameters differed, or
the arguments unchanged when there is no restart.

`restart_physics` is the protocol for a checkpoint written with *other physics
parameter values* (`_RESTART_PHYSICS_KEYS`): `:refuse` (the default) raises an
`ArgumentError` that lists every differing key with its checkpoint and current
value and names the `:allow` opt-in; `:allow` warns once with the same list and
reports `physics_changed = true`, so the driver can reset the first resumed
time step.  Numerics/trajectory settings, the geometry and the physics revision
are refused in either mode -- they are not values an intentional physics
continuation may override.
"""
function _restart_guard(qt::Quadtree, p::Int, theta0::Vector{Float64},
                        run_config::AbstractDict, restart_from::String;
                        restart_physics::Symbol = :refuse)
    restart_physics in (:refuse, :allow) ||
        throw(ArgumentError("restart_physics must be :refuse or :allow (got " *
                            ":$restart_physics): :refuse (the default) stops a " *
                            "restart whose physics differs from the checkpoint, " *
                            ":allow continues it deliberately with one warning " *
                            "and a reset first time step."))
    # Optional exact restart: replace the mesh, p and initial state with the
    # checkpoint contents before any space is built.  tstop/maxsteps remain
    # absolute, so they must be larger than the values stored in the file.
    physics_mism = String[]
    restart = isempty(restart_from) ? nothing : load_semquad_state(restart_from)
    if restart !== nothing
        # ------------------------------------------------------------------
        # Physics revision.  `run_config` compares *values*, and the two commits
        # that made the Maxwell stress update deviatoric (958af5e) and
        # co-rotational (8c27a02) changed the behaviour of unchanged parameter
        # values: a checkpoint from before them carries a stress history
        # produced by a different update.  `PHYSICS_REVISION`
        # (`src/physics/revision.jl`, which holds the dated changelog) is the
        # marker for such changes, so a stored revision that differs from it
        # refuses the restart.  This check is deliberately independent of
        # `shear_modulus`: the revision is a property of the *physics code*, not
        # of one switch, and a state written by another revision is not a state
        # this one is known to continue.
        stored_rev = get(restart.config, "physics_revision", nothing)
        if stored_rev === nothing
            # Every checkpoint written before this key existed lands here.  A
            # refusal would invalidate all of them, including the files whose
            # physics cannot be affected (`shear_modulus = 0` carries no stress
            # history at all, and neither does a file without the stress
            # datasets), and there is nothing to migrate against: the HDF5 state
            # has no marker for "already checked under this revision".  So warn
            # -- naming whether *this* file carries a stress history, which is
            # the state the revision change affects -- and resume.  The warning
            # is what makes the continuation non-silent; G252 pins its text,
            # that the resumed trajectory is bit-identical to the
            # uninterrupted one, and the refusal below.
            # "Carries a stress history" would be wrong for a malformed file
            # that has only one or two of the datasets: that is not a history a
            # run can continue, it is a partial state the stress init below
            # refuses (when this run is viscoelastic at all).  Say which case
            # the file is in, so the warning and the real error agree.  With
            # `shear_modulus = 0` nothing reads the datasets, so a partial file
            # is merely ignored -- the wording stays conditional.
            has_all = restart.txx !== nothing && restart.tyy !== nothing &&
                      restart.txy !== nothing
            has_any = restart.txx !== nothing || restart.tyy !== nothing ||
                      restart.txy !== nothing
            effect = has_all ?
                "It carries a Maxwell stress history (txx/tyy/txy) that was " *
                "advanced by the older update; resuming continues that history " *
                "under the current update." :
                has_any ?
                "It carries only part of a Maxwell stress state (some but not " *
                "all of txx/tyy/txy), which is a malformed checkpoint rather " *
                "than a stress history; a viscoelastic restart refuses it " *
                "instead of resuming an unknown state." :
                "It carries no Maxwell stress history, so there is no stored " *
                "stress state for the revision change to affect."
            @warn "checkpoint $(restart_from) predates physics revision " *
                  "tracking: it has no `physics_revision` key.  The current " *
                  "physics is revision $(PHYSICS_REVISION) (the Maxwell stress " *
                  "update became deviatoric in 958af5e and co-rotational in " *
                  "8c27a02).  $(effect)  Resuming under revision " *
                  "$(PHYSICS_REVISION)."
        else
            stored_num = tryparse(Float64, stored_rev)
            if stored_num === nothing || stored_num != Float64(PHYSICS_REVISION)
                shown = stored_num === nothing ? stored_rev :
                        (isinteger(stored_num) ? string(Int(stored_num)) :
                         string(stored_num))
                throw(ArgumentError("checkpoint $restart_from was written " *
                    "under physics revision $shown, but this code is physics " *
                    "revision $(PHYSICS_REVISION): the physics changed between " *
                    "revision $shown and revision $(PHYSICS_REVISION) (the " *
                    "Maxwell stress update was made deviatoric in 958af5e and " *
                    "co-rotational in 8c27a02), so the stored state -- in " *
                    "particular a Maxwell stress history -- is not compatible " *
                    "with the update that would continue it.  Refusing the " *
                    "restart: resume from a checkpoint written by physics " *
                    "revision $(PHYSICS_REVISION), or replay the run from the " *
                    "beginning."))
            end
        end
        # Physics mismatches are collected separately: they may be continued
        # deliberately with `restart_physics = :allow` (G291), while every other
        # key difference is a hard refusal.  The mode only exists for keys
        # *both* sides record; a key an older checkpoint does not carry is still
        # skipped (`haskey(run_config, k) || continue`), which is the historical
        # silent behaviour for every file written before the key existed.
        if !isempty(restart.config)
            mism = String[]
            for (k, v) in restart.config
                haskey(run_config, k) || continue
                # The revision has its own comparison and message above; the
                # generic value loop must not also reject a non-canonical
                # spelling of the same number (e.g. "2" against "2.0").
                k == "physics_revision" && continue
                s = config_value_string(run_config[k])
                s == v && continue
                if _restart_physics_key(k)
                    push!(physics_mism, "$k: checkpoint=$v, current=$s")
                else
                    push!(mism, "$k: checkpoint=$v, current=$s")
                end
            end
            # `stokes_scaling` postdates the old checkpoints.  A file without
            # the key was written by a solver that only had the historical
            # `:none` path, so "absent" means `none` -- not "skip the check":
            # the equilibration changes the answer of a failing solve, so a
            # resume must not switch it on silently.
            if !haskey(restart.config, "stokes_scaling")
                cur = config_value_string(get(run_config, "stokes_scaling",
                                              "none"))
                cur == "none" ||
                    push!(mism, "stokes_scaling: checkpoint=none (key absent), " *
                                "current=$cur")
            end
            # The geometry key postdates the old checkpoints, which were all
            # written by the box (stage 0's only production geometry).  Absent
            # therefore means `CartesianBox`; a mapped geometry must not resume
            # a box state silently, so it is a mismatch rather than a skip.
            # Numerics/trajectory settings and the geometry are refused in both
            # `restart_physics` modes: the protocol only covers parameter values
            # of the physics, not a different discrete setup or another physical
            # domain.
            isempty(mism) ||
                throw(ArgumentError("restart configuration mismatch in " *
                                    "$(restart_from):\n  " * join(mism, "\n  ")))
        end
        # A physics-parameter difference: decide with the explicit protocol.
        # `:refuse` (the default) asks the user to state that the continuation
        # is intentional; `:allow` warns once and proceeds.  The driver then
        # resets the first resumed time step to `dt0` (a checkpoint's dt was
        # CFL-limited for the checkpoint's flow; carrying it into stronger
        # physics is the inherited-dt blow-up G291 pins), keeping the BDF2
        # history otherwise.
        if !isempty(physics_mism)
            if restart_physics === :refuse
                throw(ArgumentError("restart configuration mismatch in " *
                    "$(restart_from):\n  " *
                    join(physics_mism, "\n  ") *
                    "\n\nThe differing keys are physics parameter values, not " *
                    "numerics or trajectory settings: the stored state (theta, " *
                    "theta_prev, velocity, pressure, Maxwell stress) was " *
                    "produced by the checkpoint's physics, so continuing it " *
                    "under the current physics changes the equations of the " *
                    "run.  Refusing the restart: pass restart_physics=:allow " *
                    "to continue anyway -- the driver then warns once with " *
                    "this list, resets the first resumed time step to dt0 (the " *
                    "CFL ramp restarts safely) and keeps the BDF2 history -- " *
                    "or replay the run from the beginning with the current " *
                    "physics."))
            else
                @warn "restart $(restart_from) continues a stored state with " *
                      "changed physics (restart_physics=:allow); the checkpoint " *
                      "was written with other parameter values:\n  " *
                      join(physics_mism, "\n  ") *
                      "\nThe state is continued under the current physics; the " *
                      "first resumed time step is reset to dt0 so the CFL ramp " *
                      "restarts from a safe step, and the BDF2 history " *
                      "(theta_prev) is otherwise kept."
            end
        end
        # The geometry record is new with stage 0 cut 1 and the check above only
        # sees files that carry *some* config.  A file without the key (including
        # one whose whole config string is empty, which the old writer could
        # produce from a raw `save_semquad_state` call) was written by the box,
        # so an absent record may be resumed as the box but never as a mapped
        # geometry.
        if !haskey(restart.config, "geometry")
            cur = config_value_string(get(run_config, "geometry", "CartesianBox"))
            startswith(cur, "CartesianBox(") ||
                throw(ArgumentError("restart configuration mismatch in " *
                    "$(restart_from):\n  geometry: checkpoint=CartesianBox " *
                    "(key absent), current=$cur"))
        end
        # The other direction (AUDIT_BUGS R5): the checkpoint records a geometry
        # but the live config does not.  The generic value loop above only
        # rejects keys *both* sides know (`haskey(run_config, k) || continue`),
        # and the branch just above only covers a checkpoint *without* the key,
        # so a checkpointed mapped state used to resume silently as a box when a
        # direct `_restart_guard` caller built a config without the key.  The
        # driver's `_run_config` always sets it, hence latent, but the guard's
        # contract is "a resumed run cannot silently continue with a different
        # setup": geometry is exactly that kind of setup, so compare (refuse)
        # rather than skip.
        if haskey(restart.config, "geometry") && !haskey(run_config, "geometry")
            throw(ArgumentError("restart configuration mismatch in " *
                "$(restart_from):\n  geometry: " *
                "checkpoint=$(restart.config["geometry"]), current=(not " *
                "recorded: the live run_config has no geometry key).  The " *
                "checkpoint was written on a recorded geometry but this run " *
                "does not state which geometry it continues, so the resume " *
                "cannot be shown to be the same physical problem.  Rebuild the " *
                "live config with its geometry (the driver's `_run_config` " *
                "sets `geometry` from the run geometry), or resume in a run " *
                "configured for $(restart.config["geometry"])."))
        end
        qt = restart.qt
        p = restart.p
        theta0 = restart.theta
        all(isfinite, theta0) ||
            throw(ArgumentError("checkpoint $(restart_from) has non-finite theta"))
    end
    return (restart = restart, qt = qt, p = p, theta0 = theta0,
            physics_changed = !isempty(physics_mism))
end
