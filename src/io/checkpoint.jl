# Checkpoint / restart for the quadtree SEM convection path.
#
# The full state of `run_semquad_simulation` is the quadtree, the two BDF2
# temperature levels, the velocity/pressure fields, the elementwise entropy
# viscosity (EV) used by the last energy assembly, and the scalars (t, dt,
# dt_prev, step, p).  Writing all of them lets a run resume exactly, instead of
# re-interpolating a visualisation grid (which perturbs the Kelly indicator).
#
# The EV field is elementwise and is *not* a function of the current state: the
# last energy step used theta^n, theta^(n-1) and u^n, and after the step only
# theta^(n+1)/theta^n survive.  The consistent-boundary-flux diagnostic needs
# the exact diffusion operator that produced theta^(n+1), so the EV values must
# be recorded rather than recomputed (see `semquad_flux.jl` and G149).
#
# The file is split by concern -- the state itself here, the cooldown datasets,
# the configuration codec and the tree rebuild in `checkpoint_*.jl` -- so a
# change to one of them cannot disturb the others.

"""
    save_semquad_state(path, qt, theta, theta_prev, ux, uz, pv; t, dt, dt_prev,
                       step, p, config=Dict(), meta=Dict(), ev=nothing,
                       weak=nothing, weak_prev=nothing, txx=nothing,
                       tyy=nothing, txy=nothing, cooldown=nothing)

Write an HDF5 checkpoint of the quadtree SEM state.

`config` is the set of run settings that affect the trajectory (AMR cadence and
marking, advection, EV, Picard controls, ...).  It is stored as a compact
`key=value` string and compared against the live settings by
`run_semquad_simulation(restart_from=...)`, so a restart cannot silently
continue with a different configuration.  `meta` is free-form provenance and is
not validated.  `ev` is the elementwise entropy-viscosity field used by the last
energy assembly, aligned with `sort(collect(qt.leaves))`; when omitted, no EV
dataset is written (an EV-free run, or an old checkpoint).  `weak`/`weak_prev`
are the strain-weakening field at the two BDF2 levels of the next step; both
are optional so an old checkpoint still loads.
`txx`/`tyy`/`txy` are the three nodal Maxwell stress-history components; all
three are optional and written only when the
stress history is active, so an old checkpoint (and every `shear_modulus = 0`
run) has none and loads with `nothing`.

`cooldown` is the `AmrCooldown` hysteresis state of the balanced AMR cycle
(`src/amr/cycle.jl`). When supplied it is
written as two integer datasets (`cooldown_no_refine`/`cooldown_no_coarsen`, one
column per key: level, ix, iy, remaining cycles) and the integer attribute
`cooldown_cycle`; they are **state**, not configuration, so they are deliberately
absent from the `config` string and the restart guard.  When omitted (an old
writer, or a caller that does not track the cycle's hysteresis) no cooldown data
is written; `load_semquad_state` then returns an empty cooldown.  The datasets
are optional on read, so old checkpoints load and restart exactly as before.

`velocity_stale` is the optional restart marker for the final checkpoint of a
run: when the last step ran an AMR cycle, the driver re-solves the flow after the
loop so the *returned* velocity belongs to the final mesh, but the uninterrupted
trajectory's next step would start with the zero velocity the AMR pass left.
Storing the re-solved field plus this marker keeps the file useful for
diagnostics while `run_semquad_simulation` restores the trajectory state (zeros)
on restart.  Absent/false is the historical behaviour.
"""
function save_semquad_state(path::AbstractString, qt::Quadtree,
                            theta::AbstractVector{<:Real},
                            theta_prev::AbstractVector{<:Real},
                            ux::AbstractVector{<:Real},
                            uz::AbstractVector{<:Real},
                            pv::AbstractVector{<:Real};
                            t::Real, dt::Real, dt_prev::Real, step::Int, p::Int,
                            config::AbstractDict=Dict{String,Any}(),
                            meta::AbstractDict=Dict{String,Any}(),
                            ev::Union{Nothing,AbstractVector{<:Real}}=nothing,
                            weak::Union{Nothing,AbstractVector{<:Real}}=nothing,
                            weak_prev::Union{Nothing,AbstractVector{<:Real}}=nothing,
                            txx::Union{Nothing,AbstractVector{<:Real}}=nothing,
                            tyy::Union{Nothing,AbstractVector{<:Real}}=nothing,
                            txy::Union{Nothing,AbstractVector{<:Real}}=nothing,
                            cooldown::Union{Nothing,AmrCooldown}=nothing,
                            velocity_stale::Bool=false)
    leaves = sort(collect(qt.leaves))
    nl = length(leaves)
    nl > 0 || throw(ArgumentError("cannot checkpoint an empty tree"))
    if ev !== nothing
        length(ev) == nl ||
            throw(ArgumentError("EV field has $(length(ev)) entries, tree has $nl leaves"))
    end
    if weak !== nothing
        length(weak) == length(theta) ||
            throw(ArgumentError("weakening field has $(length(weak)) entries, " *
                                "the state has $(length(theta))"))
    end
    if weak_prev !== nothing
        length(weak_prev) == length(theta) ||
            throw(ArgumentError("previous weakening field has " *
                                "$(length(weak_prev)) entries, the state has " *
                                "$(length(theta))"))
    end
    # The three Maxwell stress-history components (2b step 1) are optional and
    # live on the same scalar layout as `theta`; each is validated independently
    # (the driver passes all three or none).
    for (name, val) in (("txx", txx), ("tyy", tyy), ("txy", txy))
        val === nothing && continue
        length(val) == length(theta) ||
            throw(ArgumentError("$name Maxwell stress field has " *
                                "$(length(val)) entries, the state has " *
                                "$(length(theta))"))
    end
    L = Matrix{Int32}(undef, 3, nl)
    for (k, lf) in enumerate(leaves)
        L[1, k] = Int32(lf[1]); L[2, k] = Int32(lf[2]); L[3, k] = Int32(lf[3])
    end
    # A caller that passes a path under a directory that does not exist yet should
    # get the directory created, not an HDF5 error halfway through a run.
    md = dirname(abspath(path))
    isdir(md) || mkpath(md)
    h5open(path, "w") do f
        f["leaves"] = L
        f["theta"] = collect(Float64, theta)
        f["theta_prev"] = collect(Float64, theta_prev)
        f["ux"] = collect(Float64, ux)
        f["uz"] = collect(Float64, uz)
        f["pv"] = collect(Float64, pv)
        ev !== nothing && (f["ev"] = collect(Float64, ev))
        weak !== nothing && (f["weak"] = collect(Float64, weak))
        weak_prev !== nothing && (f["weak_prev"] = collect(Float64, weak_prev))
        # Maxwell stress history (2b step 1); absent when the state is off, so
        # an old checkpoint and a `shear_modulus = 0` run stay byte-identical.
        txx !== nothing && (f["txx"] = collect(Float64, txx))
        tyy !== nothing && (f["tyy"] = collect(Float64, tyy))
        txy !== nothing && (f["txy"] = collect(Float64, txy))
        if cooldown !== nothing
            # Optional `:balanced` hysteresis state (empty dict -> no dataset).
            isempty(cooldown.no_refine) ||
                (f["cooldown_no_refine"] = _cooldown_matrix(cooldown.no_refine))
            isempty(cooldown.no_coarsen) ||
                (f["cooldown_no_coarsen"] = _cooldown_matrix(cooldown.no_coarsen))
            HDF5.attrs(f)["cooldown_cycle"] = Int64(cooldown.cycle)
        end
        # Optional restart marker (final checkpoint after an AMR step): the
        # stored velocity is diagnostic and the trajectory's next step starts
        # from zero.  Absent on old files and on in-loop checkpoints.
        velocity_stale && (HDF5.attrs(f)["velocity_stale"] = Int64(1))
        HDF5.attrs(f)["t"] = Float64(t)
        HDF5.attrs(f)["dt"] = Float64(dt)
        HDF5.attrs(f)["dt_prev"] = Float64(dt_prev)
        HDF5.attrs(f)["step"] = Int64(step)
        HDF5.attrs(f)["p"] = Int64(p)
        # The tree's maxlevel is part of the state: refine! refuses to go below
        # it, so dropping it would silently stop future refinement.
        HDF5.attrs(f)["maxlevel"] = Int64(qt.maxlevel)
        # The base cell counts are part of the geometry: without them a 5x2 tree
        # is rebuilt as a single-root one and the leaves no longer partition the
        # box (see the guard in load_semquad_state).
        HDF5.attrs(f)["nx0"] = Int64(qt.nx0)
        HDF5.attrs(f)["ny0"] = Int64(qt.ny0)
        # A stretched single-root box (cells_x=cells_y=1, Lx != 1) has the same
        # nx0/ny0 as the unit square, so the width must be stored explicitly.
        HDF5.attrs(f)["Lx"] = Float64(qt.Lx)
        # Stage 0.5c1: the tree topology (`:none` or `:periodic`).  A ring
        # checkpoint must rebuild a periodic tree, or the resumed mesh would be
        # the cut ring (T = 0 and T = 1 separate nodes, seam faces as walls).
        # Absent in old files = the bounded topology (`:none`).
        HDF5.attrs(f)["seam"] = String(qt.seam)
        for (k, v) in meta
            HDF5.attrs(f)[string(k)] = v
        end
        if !isempty(config)
            HDF5.attrs(f)["config"] = join([string(k, "=", config_value_string(v))
                                            for (k, v) in sort(collect(config), by=first)],
                                           "\n")
        end
    end
    return path
end

"""
    load_semquad_state(path)

Read a checkpoint written by `save_semquad_state` and rebuild the quadtree from
its leaf list.  Returns a named tuple `(qt, theta, theta_prev, ux, uz, pv, t,
dt, dt_prev, step, p, config, ev, weak, weak_prev, txx, tyy, txy, cooldown,
velocity_stale)`.

`ev` is the elementwise entropy-viscosity field, or `nothing` for checkpoints
written before it was recorded.  Old checkpoints that predate `nx0`/`ny0` are
recovered by inferring the base cells from the leaves: this is what makes the
multi-root 2b checkpoints (5x2 base) loadable.  A stretched single-root box
whose `Lx` was not stored cannot be recovered and falls back to `Lx = 1`; new
checkpoints always store `Lx`.  `weak`/`weak_prev` are the two BDF2 levels of
the strain-weakening field, or `nothing` when weakening is off or the file
predates it.  `txx`/`tyy`/`txy` are the Maxwell stress-history components, or
`nothing` when the stress state is off or the
file predates it.  `cooldown` is the optional `AmrCooldown`
hysteresis state of the balanced AMR cycle; a checkpoint without the cooldown
datasets (every file written before the cooldown existed) returns an **empty**
cooldown (`cycle == 0`), which is exactly the
historical restart behaviour. `velocity_stale`
is the optional marker described in `save_semquad_state`: true means the stored
`ux/uz/pv` are the end-of-run diagnostic re-solve and a restart must zero them to
continue the trajectory (false/absent = use them verbatim).
"""
function load_semquad_state(path::AbstractString)
    isfile(path) || throw(ArgumentError("checkpoint not found: $path"))
    return h5open(path, "r") do f
        haskey(f, "leaves") ||
            throw(ArgumentError("checkpoint $path has no `leaves` dataset " *
                                "(not a quadtree state file)"))
        L = read(f["leaves"])
        nl = size(L, 2)
        nl > 0 || throw(ArgumentError("checkpoint $path has an empty leaf list"))
        leaves = [(Int(L[1, k]), Int(L[2, k]), Int(L[3, k])) for k in 1:nl]
        maxleaf = Int(maximum(L[1, :]))
        # Old checkpoints (before maxlevel was stored) fall back to the deepest
        # leaf, which was the source of a silent "cannot refine any deeper" bug.
        maxlev = haskey(HDF5.attrs(f), "maxlevel") ?
                 Int(HDF5.attrs(f)["maxlevel"]) : maxleaf
        maxlev >= maxleaf ||
            throw(ArgumentError("checkpoint maxlevel=$maxlev is below its " *
                                "deepest leaf $maxleaf"))
        has_nx0 = haskey(HDF5.attrs(f), "nx0")
        has_ny0 = haskey(HDF5.attrs(f), "ny0")
        Lx_attr = haskey(HDF5.attrs(f), "Lx") ? Float64(HDF5.attrs(f)["Lx"]) : nothing
        # Stage 0.5c1: the stored tree topology.  Old files predate the key and
        # were all bounded, so an absent attribute is `:none`.
        seam = haskey(HDF5.attrs(f), "seam") ?
               Symbol(HDF5.attrs(f)["seam"]) : :none
        seam in (:none, :periodic) ||
            throw(ArgumentError("checkpoint $path stores seam = :$seam; only " *
                                ":none and :periodic exist"))
        if has_nx0 && has_ny0
            nx0 = Int(HDF5.attrs(f)["nx0"])
            ny0 = Int(HDF5.attrs(f)["ny0"])
            Lx = Lx_attr === nothing ?
                 (nx0 == 1 && ny0 == 1 ? 1.0 : nx0 / ny0) : Lx_attr
            qt = _rebuild_tree(leaves, maxlev; nx0=nx0, ny0=ny0, Lx=Lx, seam=seam)
            qt === nothing &&
                throw(ArgumentError("checkpoint leaf list is not a partition " *
                                    "for the stored base $(nx0)x$(ny0) and " *
                                    "seam = :$seam"))
        else
            # Old format: infer the base cells from the wall leaves.  Try every
            # candidate; a valid pair must rebuild the exact leaf set.
            cx, cy = _checkpoint_leaf_candidates(leaves)
            found = Tuple{Quadtree,Int,Int}[]
            for nx0 in cx, ny0 in cy
                Lx = Lx_attr === nothing ? nx0 / ny0 : Lx_attr
                qt = _rebuild_tree(leaves, maxlev; nx0=nx0, ny0=ny0, Lx=Lx,
                                   seam=seam)
                qt === nothing || push!(found, (qt, nx0, ny0))
            end
            isempty(found) &&
                throw(ArgumentError("checkpoint leaf list is not a partition " *
                                    "for any base cells in $(cx)x$(cy); the " *
                                    "file is corrupt or the tree is not 2:1-balanced"))
            # A single valid candidate is the normal case.  If several bases
            # happen to produce the same leaves (only possible when no leaf
            # touches a wall at full depth), prefer the smallest; the geometry
            # is identical either way, and the leaf set is what is restored.
            qt, nx0, ny0 = found[1]
            if length(found) > 1
                sort!(found, by=x -> (x[1].nx0 * x[1].ny0, x[1].nx0))
                qt, nx0, ny0 = found[1]
            end
        end
        length(qt.leaves) == nl ||
            throw(ArgumentError("checkpoint leaf list is not a partition " *
                                "($(length(qt.leaves)) built vs $nl stored)"))
        ev = haskey(f, "ev") ? read(f["ev"]) : nothing
        ev === nothing || length(ev) == nl ||
            throw(ArgumentError("checkpoint EV field has $(length(ev)) entries, " *
                                "tree has $nl leaves"))
        weak = haskey(f, "weak") ? read(f["weak"]) : nothing
        weak === nothing || length(weak) == length(read(f["theta"])) ||
            throw(ArgumentError("checkpoint weakening field has " *
                                "$(length(weak)) entries, the state has " *
                                "$(length(read(f["theta"])))"))
        weak_prev = haskey(f, "weak_prev") ? read(f["weak_prev"]) : nothing
        weak_prev === nothing || length(weak_prev) == length(read(f["theta"])) ||
            throw(ArgumentError("checkpoint previous weakening field has " *
                                "$(length(weak_prev)) entries, the state has " *
                                "$(length(read(f["theta"])))"))
        # Maxwell stress history (2b step 1): absent from every checkpoint that
        # predates it (and from every `shear_modulus = 0` run), which loads as
        # `nothing`; the driver then starts the history from zero stress.
        txx = haskey(f, "txx") ? read(f["txx"]) : nothing
        txx === nothing || length(txx) == length(read(f["theta"])) ||
            throw(ArgumentError("checkpoint txx Maxwell stress field has " *
                                "$(length(txx)) entries, the state has " *
                                "$(length(read(f["theta"])))"))
        tyy = haskey(f, "tyy") ? read(f["tyy"]) : nothing
        tyy === nothing || length(tyy) == length(read(f["theta"])) ||
            throw(ArgumentError("checkpoint tyy Maxwell stress field has " *
                                "$(length(tyy)) entries, the state has " *
                                "$(length(read(f["theta"])))"))
        txy = haskey(f, "txy") ? read(f["txy"]) : nothing
        txy === nothing || length(txy) == length(read(f["theta"])) ||
            throw(ArgumentError("checkpoint txy Maxwell stress field has " *
                                "$(length(txy)) entries, the state has " *
                                "$(length(read(f["theta"])))"))
        cfg = haskey(HDF5.attrs(f), "config") ?
              parse_config_string(HDF5.attrs(f)["config"]) : Dict{String,String}()
        cool = _load_cooldown(f)
        vstale = haskey(HDF5.attrs(f), "velocity_stale") &&
                 Int(HDF5.attrs(f)["velocity_stale"]) != 0
        (qt = qt,
         theta = read(f["theta"]),
         theta_prev = read(f["theta_prev"]),
         ux = read(f["ux"]),
         uz = read(f["uz"]),
         pv = read(f["pv"]),
         t = Float64(HDF5.attrs(f)["t"]),
         dt = Float64(HDF5.attrs(f)["dt"]),
         dt_prev = Float64(HDF5.attrs(f)["dt_prev"]),
         step = Int(HDF5.attrs(f)["step"]),
         p = Int(HDF5.attrs(f)["p"]),
         config = cfg,
         ev = ev,
         weak = weak,
         weak_prev = weak_prev,
         txx = txx,
         tyy = tyy,
         txy = txy,
         cooldown = cool,
         velocity_stale = vstale)
    end
end
