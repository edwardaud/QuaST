# The AMR cycle of the time loop, extracted from `run.jl`.
#
# Stage 2 of the `RunState` refactor:
# the block that ran every `amr_every` steps in `run_semquad_simulation` now
# lives here as one function.  It is the same code in the same order -- the
# pass, the stage-1 `snapshot!` immediately after it, the transfers, the flow
# invalidation, the tree-cache drop and the two verbose reports -- so a card's
# trajectory and its log are unchanged.
#
# The block rebinds sixteen locals of the time loop, so the function returns
# every value the caller must rebind; the call site in `run.jl` is the call
# plus those rebindings.  `qt` is mutated in place and is not returned.  The
# profiling slot and the report lines moved with the block, so the call site
# stays a single call.
#
# Keyword names follow the driver's config names where they exist
# (`amr_passes`, `amr_indicator`), so the moved `semquad_amr_passes!` call is
# verbatim.

"""
    _amr_step!(ss, qt, theta, theta_prev; <amr keywords>, ux, uz, sq, diag,
               chi, chi_prev, chi_iter, txx, tyy, txy,
               weak, stress, linear, tprof, profile_steps, verbose)
        -> NamedTuple

One AMR cycle of `run_semquad_simulation`'s time loop, extracted from the loop
body.

Runs `semquad_amr_passes!`, snapshots `diag` immediately after it (the stage-1
ordering rule: the snapshot must precede the verbose print that reads
`diag.amr_stats`/`diag.amr_balanced`), transfers the carried fields to the new
mesh, invalidates the flow state and reports the cycle.

Returns the named tuple `(ss, theta, theta_prev, sq, chi, chi_prev, chi_iter,
txx, tyy, txy, ux, uz, pv, edot_field, ev_field, velocity_stale)`: every loop
local the caller must rebind.  The `chi*` fields and `txx`/`tyy`/`txy` are
returned unchanged when `weak`/`stress` is false; the caller rebinds them
unconditionally, which is then a no-op.  `tprof[5]` is the `amr` slot of the
caller's per-step profile, updated over the same region as before the
extraction.

`amr_cooldown` is the caller-owned `AmrCooldown` hysteresis state of the balanced
cycle.  It is forwarded to `semquad_amr_passes!`, which ages it in place, so the
driver must pass the same object to every cycle (and to its `initial_refine`
cycles); the function itself neither creates nor stores it.
"""
function _amr_step!(ss::SemStokes, qt::Quadtree, theta::Vector{Float64},
                    theta_prev::Vector{Float64};
                    # The AMR policy, forwarded verbatim to the pass.
                    refine_frac::Float64, coarsen_frac::Float64,
                    amr_passes::Int, max_level::Int, max_leaves::Int,
                    wall_min_level::Int, wall_strip::Float64,
                    symmetry::Symbol, transfer_mode::Symbol,
                    hotspot_weight::Float64, hotspot_frac::Float64,
                    hotspot_min_level::Int, hotspot_mode::Symbol,
                    hotspot_zmin::Float64, zero_side_eta::Bool,
                    side_strip::Float64, top_wall_level::Int,
                    top_wall_frac::Float64, amr_hysteresis::Int,
                    amr_coarsen_gate::Bool,
                    # The caller-owned hysteresis state of the balanced cycle;
                    # the pass ages it in place, so it must be the same object
                    # across every AMR call of the run.
                    amr_cooldown::AmrCooldown,
                    # The in-loop indicator's velocity and selection (see the
                    # comment at the call).
                    ux::Vector{Float64}, uz::Vector{Float64},
                    amr_indicator::Symbol,
                    # Driver context: the stage-1 snapshot target, the current
                    # spaces, the fields the block carries across the mesh
                    # change, the config flags, and the step profile/report.
                    diag::RunDiagnostics, sq::SemQuad,
                    chi, chi_prev, chi_iter, txx, tyy, txy,
                    weak::Bool, stress::Bool, linear::Symbol,
                    tprof::Union{Nothing,Vector{Float64}},
                    profile_steps::Bool, verbose::Bool)
    tp = profile_steps ? time_ns() : 0
    old_sq = sq
    ss, theta, theta_prev, nr = semquad_amr_passes!(
        ss, qt, theta, theta_prev;
        refine_frac=refine_frac, coarsen_frac=coarsen_frac,
        passes=amr_passes, max_level=max_level,
        max_leaves=max_leaves, wall_min_level=wall_min_level,
        wall_strip=wall_strip, symmetry=symmetry,
        transfer_mode=transfer_mode,
        hotspot_weight=hotspot_weight, hotspot_frac=hotspot_frac,
        hotspot_min_level=hotspot_min_level,
        hotspot_mode=hotspot_mode, hotspot_zmin=hotspot_zmin, zero_side_eta=zero_side_eta, side_strip=side_strip,
        top_wall_level=top_wall_level, top_wall_frac=top_wall_frac,
        amr_hysteresis=amr_hysteresis,
        amr_coarsen_gate=amr_coarsen_gate,
        amr_cooldown=amr_cooldown,
        # The in-loop indicator runs on the velocity the step just
        # solved, which is on the current mesh (`sq`).  `:theta`
        # ignores these; `:velocity`/`:both` need them or
        # `semquad_amr_passes!` would have to guess (the historical bug
        # was reading an undefined local and throwing on the first AMR
        # cycle -- G234 pins a real `amr_every=1` velocity cycle).
        ux=ux, uz=uz,
        indicator=amr_indicator)
    # The pass wrote `LAST_AMR_STATS` and `LAST_AMR_BALANCED`; snapshot
    # them before the verbose print below reads them.
    snapshot!(diag)
    sq = ss.su
    # The weakening field lives on the scalar space, so it follows the
    # same transfer as theta (no wall Dirichlet data: `zero_walls=false`).
    if weak
        chi = transfer_semquad(old_sq, chi, sq; mode=transfer_mode,
                               zero_walls=false, qt_new=qt)
        chi_prev = transfer_semquad(old_sq, chi_prev, sq;
                                    mode=transfer_mode,
                                    zero_walls=false, qt_new=qt)
        chi_iter = copy(chi)
    end
    # The Maxwell stress history is a set of scalar fields on the same
    # space, so it follows the same transfer as `chi` (2b step 1).
    if stress
        txx = transfer_semquad(old_sq, txx, sq; mode=transfer_mode,
                               zero_walls=false, qt_new=qt)
        tyy = transfer_semquad(old_sq, tyy, sq; mode=transfer_mode,
                               zero_walls=false, qt_new=qt)
        txy = transfer_semquad(old_sq, txy, sq; mode=transfer_mode,
                               zero_walls=false, qt_new=qt)
    end
    ux = zeros(Float64, sq.N)
    uz = zeros(Float64, sq.N)
    pv = zeros(Float64, ss.sp.N)
    edot_field = nothing      # the mesh changed: the old field is stale
    ev_field = nothing        # same for the elementwise EV field
    # old tree hierarchies (`:auto` may resolve to MG on the new tree).
    # `_semquad_mg_drop_tree_entries!` keeps the per-level cache (G221):
    # a coarse level whose leaf set did not change survives the cycle.
    (linear === :mg || linear === :auto) && _semquad_mg_drop_tree_entries!()
    velocity_stale = true
    tprof !== nothing && (tprof[5] += (time_ns() - tp) / 1.0e9)
    verbose && @printf("    AMR: N=%d leaves=%d maxjump=%d refined=%d coarsened=%d merged=%d\n",
                       sq.N, length(qt.leaves), max_face_level_jump(qt), nr,
                       diag.amr_stats.coarsened, diag.amr_stats.merged)
    if verbose
        # Balanced-cycle diagnostics:
        # the recirculation counter and the coarsening-gate ledger.
        d = diag.amr_balanced
        @printf("      balanced: gate_refused=%d extended=%d recirc=%d post=%+d tau_h=%.3e eligible=%d\n",
                d.gate_refused, d.gate_extended, d.recirc, d.post,
                d.tau_h, d.eligible)
    end

    return (ss=ss, theta=theta, theta_prev=theta_prev, sq=sq,
            chi=chi, chi_prev=chi_prev, chi_iter=chi_iter,
            txx=txx, tyy=tyy, txy=txy,
            ux=ux, uz=uz, pv=pv,
            edot_field=edot_field, ev_field=ev_field,
            velocity_stale=velocity_stale)
end
