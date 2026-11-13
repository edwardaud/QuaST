# Checkpoint / restart for the quadtree SEM convection path (route C).
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
# Kept in its own file so the legacy time loop can be restored from the
# c-route-baseline-2026-10-30 tag/archive without touching this code.

# --- optional `:balanced` AMR cooldown state --------------------------------
# The cooldown is a Dict of leaf keys -> remaining cycles.  It is serialised as
# an integer matrix with four rows (level, ix, iy, remaining) and one column per
# key, sorted by key, so the file is deterministic; an empty dict writes no
# dataset.  The number of AMR cycles the state has been aged through is one
# integer attribute.  All of this is optional and absent from old checkpoints.

"""Serialise one cooldown dict to a deterministic 4 x n Int64 matrix."""
function _cooldown_matrix(d::Dict{NTuple{3,Int},Int})
    ks = sort(collect(d))
    M = Matrix{Int64}(undef, 4, length(ks))
    for (j, (k, v)) in enumerate(ks)
        M[1, j] = k[1]
        M[2, j] = k[2]
        M[3, j] = k[3]
        M[4, j] = v
    end
    return M
end

"""Rebuild one cooldown dict from a 4 x n integer matrix (validated)."""
function _cooldown_dict(M::AbstractMatrix)
    size(M, 1) == 4 ||
        throw(ArgumentError("cooldown matrix must have 4 rows " *
                            "(level, ix, iy, remaining), got $(size(M, 1))"))
    d = Dict{NTuple{3,Int},Int}()
    for j in 1:size(M, 2)
        l = Int(M[1, j]); ix = Int(M[2, j]); iy = Int(M[3, j])
        n = Int(M[4, j])
        (l >= 0 && ix >= 0 && iy >= 0 && n >= 0) ||
            throw(ArgumentError("cooldown matrix column $j is not a " *
                                "non-negative leaf/count: ($l, $ix, $iy, $n)"))
        d[(l, ix, iy)] = n
    end
    return d
end

"""Read the optional cooldown datasets/attribute of an open checkpoint."""
function _load_cooldown(f::HDF5.File)
    has_nr = haskey(f, "cooldown_no_refine")
    has_nc = haskey(f, "cooldown_no_coarsen")
    has_cyc = haskey(HDF5.attrs(f), "cooldown_cycle")
    # Old checkpoints have neither a dataset nor the attribute.  A new but
    # (temporarily) empty cooldown keeps its cycle count in the attribute, so a
    # file whose dicts are empty still restores the counter.
    (has_nr || has_nc || has_cyc) || return AmrCooldown()
    empty_d = Dict{NTuple{3,Int},Int}()
    nr = has_nr ? _cooldown_dict(read(f["cooldown_no_refine"])) : empty_d
    nc = has_nc ? _cooldown_dict(read(f["cooldown_no_coarsen"])) :
                  Dict{NTuple{3,Int},Int}()
    cyc = has_cyc ? Int(HDF5.attrs(f)["cooldown_cycle"]) : 0
    cyc >= 0 ||
        throw(ArgumentError("checkpoint cooldown_cycle=$cyc is negative"))
    return AmrCooldown(nr, nc, cyc)
end

"""
    save_semquad_state(path, qt, theta, theta_prev, ux, uz, pv; t, dt, dt_prev,
                       step, p, config=Dict(), meta=Dict(), ev=nothing,
                       weak=nothing, weak_prev=nothing, cooldown=nothing)

Write an HDF5 checkpoint of the quadtree SEM state.

`config` is the set of run settings that affect the trajectory (AMR cadence and
marking, advection, EV, Picard controls, ...).  It is stored as a compact
`key=value` string and compared against the live settings by
`run_semquad_simulation(restart_from=...)`, so a restart cannot silently
continue with a different configuration.  `meta` is free-form provenance and is
not validated.  `ev` is the elementwise entropy-viscosity field used by the last
energy assembly, aligned with `sort(collect(qt.leaves))`; when omitted, no EV
dataset is written (an EV-free run, or an old checkpoint).  `weak`/`weak_prev`
are the strain-weakening field at the two BDF2 levels of the next step
(docs/RHEOLOGY.md §9); both are optional so an old checkpoint still loads.

`cooldown` is the `AmrCooldown` hysteresis state of `amr_policy=:balanced`
(`julia/src/amr_balanced.jl`, docs/AMR.md §3.5).  When supplied it is
written as two integer datasets (`cooldown_no_refine`/`cooldown_no_coarsen`, one
column per key: level, ix, iy, remaining cycles) and the integer attribute
`cooldown_cycle`; they are **state**, not configuration, so they are deliberately
absent from the `config` string and the restart guard.  When omitted (the classic
path, or an old writer) no cooldown data is written; `load_semquad_state` then
returns an empty cooldown.  The datasets are optional on read, so old
checkpoints load and restart exactly as before.

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
    L = Matrix{Int32}(undef, 3, nl)
    for (k, lf) in enumerate(leaves)
        L[1, k] = Int32(lf[1]); L[2, k] = Int32(lf[2]); L[3, k] = Int32(lf[3])
    end
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

"""Canonical string form of a run setting, used to compare checkpoints."""
config_value_string(v::Symbol) = string(v)
config_value_string(v::Bool) = v ? "true" : "false"
config_value_string(v::Real) = string(Float64(v))
config_value_string(v) = string(v)

"""Parse the `key=value` config string stored in a checkpoint."""
function parse_config_string(s::AbstractString)
    out = Dict{String,String}()
    for line in split(s, '\n')
        isempty(strip(line)) && continue
        i = findfirst('=', line)
        i === nothing && continue
        out[String(line[1:prevind(line, i)])] = String(line[nextind(line, i):end])
    end
    return out
end

"""
    _checkpoint_leaf_candidates(leaves)

Candidate base-cell counts implied by the leaves that touch the right/top wall.
A leaf `(l, ix, iy)` on the right wall satisfies `ix + 1 == nx0 * 2^l`, so every
level whose largest `ix` is a right-wall leaf contributes `(max(ix)+1) / 2^l`.
Candidates are validated by rebuilding the tree and comparing the leaf set, so
a level with an interior extreme only produces a harmless candidate.
"""
function _checkpoint_leaf_candidates(leaves::Vector{NTuple{3,Int}})
    maxx = Dict{Int,Int}()
    maxy = Dict{Int,Int}()
    for (l, ix, iy) in leaves
        maxx[l] = max(get(maxx, l, -1), ix)
        maxy[l] = max(get(maxy, l, -1), iy)
    end
    cx = Set{Int}([1])
    cy = Set{Int}([1])
    for (l, mx) in maxx
        q = 1 << l
        (mx + 1) % q == 0 && push!(cx, (mx + 1) ÷ q)
    end
    for (l, my) in maxy
        q = 1 << l
        (my + 1) % q == 0 && push!(cy, (my + 1) ÷ q)
    end
    return sort(collect(cx)), sort(collect(cy))
end

"""
    _rebuild_tree(leaves, maxlev; nx0, ny0, Lx)

Rebuild a tree with the given base cells and return it only when its leaf set is
exactly `leaves`; otherwise return `nothing`.
"""
function _rebuild_tree(leaves::Vector{NTuple{3,Int}}, maxlev::Int;
                       nx0::Int, ny0::Int, Lx::Float64)
    qt = Quadtree(maxlev; Lx=Lx, cells_x=nx0, cells_y=ny0)
    for (l, ix, iy) in leaves
        node = (0, ix >> l, iy >> l)
        for lev in 1:l
            node in qt.leaves && refine!(qt, node)
            node = (lev, ix >> (l - lev), iy >> (l - lev))
        end
    end
    length(qt.leaves) == length(leaves) || return nothing
    Set(qt.leaves) == Set(leaves) || return nothing
    return qt
end

"""
    load_semquad_state(path)

Read a checkpoint written by `save_semquad_state` and rebuild the quadtree from
its leaf list.  Returns a named tuple `(qt, theta, theta_prev, ux, uz, pv, t,
dt, dt_prev, step, p, config, ev, cooldown, velocity_stale)`.

`ev` is the elementwise entropy-viscosity field, or `nothing` for checkpoints
written before it was recorded.  Old checkpoints that predate `nx0`/`ny0` are
recovered by inferring the base cells from the leaves: this is what makes the
multi-root 2b checkpoints (5x2 base) loadable.  A stretched single-root box
whose `Lx` was not stored cannot be recovered and falls back to `Lx = 1`; new
checkpoints always store `Lx`.  `cooldown` is the optional `AmrCooldown`
hysteresis state of `amr_policy=:balanced`; a checkpoint without the cooldown
datasets returns an **empty** cooldown (`cycle == 0`), which is exactly the
historical restart behaviour (docs/BUG_LEDGER.md §10.5).  `velocity_stale`
is the optional marker described in `save_semquad_state`: true means the stored
`ux/uz/pv` are the end-of-run diagnostic re-solve and a restart must zero them to
continue the trajectory (false/absent = use them verbatim).
"""
function load_semquad_state(path::AbstractString)
    isfile(path) || throw(ArgumentError("checkpoint not found: $path"))
    return h5open(path, "r") do f
        haskey(f, "leaves") ||
            throw(ArgumentError("checkpoint $path has no `leaves` dataset " *
                                "(not a route-C state file)"))
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
        if has_nx0 && has_ny0
            nx0 = Int(HDF5.attrs(f)["nx0"])
            ny0 = Int(HDF5.attrs(f)["ny0"])
            Lx = Lx_attr === nothing ?
                 (nx0 == 1 && ny0 == 1 ? 1.0 : nx0 / ny0) : Lx_attr
            qt = _rebuild_tree(leaves, maxlev; nx0=nx0, ny0=ny0, Lx=Lx)
            qt === nothing &&
                throw(ArgumentError("checkpoint leaf list is not a partition " *
                                    "for the stored base $(nx0)x$(ny0)"))
        else
            # Old format: infer the base cells from the wall leaves.  Try every
            # candidate; a valid pair must rebuild the exact leaf set.
            cx, cy = _checkpoint_leaf_candidates(leaves)
            found = Tuple{Quadtree,Int,Int}[]
            for nx0 in cx, ny0 in cy
                Lx = Lx_attr === nothing ? nx0 / ny0 : Lx_attr
                qt = _rebuild_tree(leaves, maxlev; nx0=nx0, ny0=ny0, Lx=Lx)
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
         cooldown = cool,
         velocity_stale = vstale)
    end
end
