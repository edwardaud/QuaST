# The balanced AMR cycle: thresholds, plan execution, budget exchange and report.
# The balanced AMR policy.
#
# It is the only adaptive policy: `run_semquad_simulation` has used it since
# 2026-11-13, and the `:classic` fraction + hard-budget path
# (formerly `amr_cycle!` in `dgquad.jl`) was removed at `8c622df`, so
# `semquad_amr_passes!` and the driver's `initial_refine` both call
# `amr_cycle_balanced!` unconditionally.
#
# Measured motivation (production run E, `out_julia/case_realistic_E`, step
# 18800: 9994/10000 leaves, 3720 AMR cycles of which 723 changed anything, last
# structural change at step 17225, delta_top thinning to 0.073 = h/delta 0.213):
#
#   (a) the absolute `coarsen_theta_tol = 1e-2` protects 9892 of 9994 leaves
#       (99.0 %); with the `symmetry=:x` mirror closure the protect set covers
#       9980 leaves and only **14** are coarsenable, so no sibling quartet can
#       ever be merged;
#   (b) the classic coarsen loop charges *every* walked leaf to its `cum_c`
#       budget, merged or not (the removed `amr_cycle!`), so the budget is
#       spent by refused leaves: measured 94.8 % by protected leaves, 5.2 % by
#       incomplete quartets, 0.0 % by mergeable ones;
#   (c) at 9994 leaves the hard budget guard `length + 10 > max_leaves` breaks
#       the refinement loop **before the first candidate** (0 of the 82
#       Dörfler refinements the policy asks for), and the driver additionally
#       pins `refine_frac = 0` at the cap;
#   (d) the 10 %-of-height wall-strip protect set is 2040 leaves (20.4 %) and
#       carries 60.6 % of the raw indicator.
#
# The `:balanced` path keeps 2:1 balance, the x-mirror invariant (G208) and the
# leaf budget, and replaces the four mechanisms:
#
#   (a') coarsening is *relative*: the Dörfler-from-below threshold
#        `tau_c = coarsen_frac * total` (cumulative, from below) is compared
#        against a **complete quartet's** `max(eta_child)`, never against a
#        single child's eta and never against an absolute tolerance;
#   (b') refused / protected quartets are not charged: a merge is only counted
#        when it happened;
#   (c') budget exchange: refinement is never blocked by `max_leaves`; the
#        lowest-sum eligible quartet is merged to fund three slots (six under
#        `:x`), and only when *no* eligible quartet is left does the refinement
#        loop stop, with `budget_limited = true`;
#   (d') the wall protect set is only the first/last leaf rows; the strip's
#        minimum level is guaranteed by `refine_wall_to_min!`, and the merge
#        side refuses a quartet at `level <= l_min` inside the strip (the
#        quarter that the wall rule would re-split on the same pass);
#   (e') hysteresis: a threshold deadband `tau_h = min(tau_c, 0.5*tau_r)` plus
#        an explicit cycle cooldown (`AmrCooldown`) so a quartet that was just
#        merged is not re-refined and a leaf that was just created by a split is
#        not immediately merged again.  Both are **our** choices: ASPECT has no
# hysteresis and no cell budget.
#
# The coarsen *supply* is demand-driven: `deficit = n0 + 6*#candidates -
# max_leaves` leaves have to disappear to fit the Dörfler refinement set, capped
# by the relative cell budget `coarsen_frac * n0`.  An unconditional
# "coarsen 5 % of the error every cycle" is a one-way coarsening, not a trade:
# measured on the E tree, one such pass merges 1112 quartets (9994 -> 6682
# leaves) and removes only 1.48 % of the indicator.

"""
Diagnostics of the last `amr_cycle_balanced!` cycle (named tuple):
`(refined, coarsened, exchanged, n, want, eligible, dropped, tau_r, tau_c,
tau_h, deadband, budget_limited, budget_infeasible, merged, recirc, post)`.
`exchanged` counts the leaves freed by the budget exchange, `want` the leaves
the Dörfler refinement set would need, `eligible` the complete quartets that
passed every gate, `budget_limited` that refinement stopped because no eligible
quartet was left, `budget_infeasible` that the tree is still above `max_leaves`
after the cycle (the caller's hard budget pass then merges the lowest-error
quartets).

`merged` is the number of quartets this cycle merged, `recirc` how many of those
parents the post-merge `balance!`/`mirror_close!` split again (**the
recirculation counter**), `post` the leaves the
post-merge balancing added, and `parents` the parent nodes themselves (one entry
per merged quartet, both halves under `symmetry=:x`), so a probe can attribute
the recirculation.  `nmerge`/`npost` are the leaf counts after the coarsening
pass and after the refinement pass, i.e. before the post-merge balancing, so
`npost - nmerge == 3*refined - 3*(merges done by the exchange)` is checkable.
`recirc == 0` means every merge survived the balance exactly as merged; with
`coarsen_gate=true` the gate promises it.

Loop/guard counters:
`plan_sweeps`/`plan_rounds` the fixed-point sweeps and extension rounds the
coarsening plan used, `plan_capped` whether it stopped on `MAX_PLAN_SWEEPS`/
`MAX_PLAN_ROUNDS` instead of converging, `fund_calls`/`fund_hits` how often the
live-gated funding merge was invoked and how often it found a legal quartet,
`balance_ok`/`mirror_ok` the success flags of the post-merge passes (both used to
be ignored), and `mirror_rounds` the passes `mirror_close!` needed.

Cap-aware diagnostics (opt-in `cap_aware`): `cap_aware` echoes the option,
`cap_applied` says the fallback replaced the marking this cycle, `tau_r_all` is
the standard all-leaf threshold and `mark_target` the effective refine target
(`refine_frac * total` by default, the sum of the marked eta on a fallback
cycle); `refinable`/`refinable_mass` count the admissible (`level < max_level`,
not cooling) leaves and their indicator mass, `cap_prefix`/`cap_mass` the leaves
and mass of the standard marked prefix that sit at the cap (only computed on a
cycle that would otherwise stall).  On a fallback cycle `tau_r`/`tau_h` are the
effective fallback values and `want` counts the fallback leaves.  When the
option is off, or on but no fallback fires, `cap_applied` is false,
`cap_prefix`/`cap_mass`/`refinable`/`refinable_mass` are 0, and
`tau_r_all`/`mark_target` are the standard all-leaf values.
"""
const LAST_AMR_BALANCED = Ref((refined=0, coarsened=0, exchanged=0, n=0, want=0,
                              eligible=0, dropped=0, tau_r=0.0, tau_c=0.0,
                              tau_h=0.0, deadband=false, budget_limited=false,
                              budget_infeasible=false, merged=0, recirc=0,
                              post=0, parents=NTuple{3,Int}[], nmerge=0,
                              npost=0, gate_refused=0, gate_extended=0,
                              plan_sweeps=0, plan_rounds=0, plan_capped=false,
                              fund_calls=0, fund_hits=0, balance_ok=true,
                              mirror_ok=true, mirror_rounds=0,
                              cap_aware=false, cap_applied=false,
                              tau_r_all=0.0, mark_target=0.0, refinable=0,
                              refinable_mass=0.0, cap_prefix=0, cap_mass=0.0))

"""
    amr_cycle_balanced!(qt, eta; refine_frac=0.2, coarsen_frac=0.05,
                        hysteresis=0.5, max_level=typemax(Int),
                        max_leaves=typemax(Int), symmetry=:none,
                        protect=Set(), wall_min_level=0, wall_strip=0.1,
                        cooldown=nothing, hysteresis_cycles=1,
                        coarsen_cells=nothing, coarsen_gate=true,
                        cap_aware=false, geom)

One AMR cycle of the balanced policy, the only adaptive
policy.  Returns `(refined, coarsened, exchanged)` in leaves, and records the
cycle's threshold/limit diagnostics in [`LAST_AMR_BALANCED`](@ref).

The thresholds are cumulative error fractions, computed exactly by sorting
(deal.II bisects the same quantity):

    tau_r = eta value at the refine_frac * total point from above
    tau_c = eta value at the coarsen_frac * total point from below
    tau_h = min(tau_c, hysteresis * tau_r)          # coarsen gate (deadband)

A leaf is refined when `eta >= tau_r`, `level < max_level` and it is not in the
`no_refine` cooldown.  A quartet is merged only when all four children are
leaves, none is protected, every child has `level > l_min` (`l_min` is
`wall_min_level` inside `wall_strip`, 0 elsewhere), `max(eta_child) <= tau_h`,
and the mirror quartet (`symmetry=:x`) passes the same test.  `coarsen_cells`
caps the leaves dropped per cycle (default `round(coarsen_frac * n)`).

**Opt-in max-level fallback (`cap_aware=true`, default `false`).**  The
standard Dörfler marking above is unchanged.  If it produces no candidate
because every leaf in its marked prefix is at `level >= max_level` (a stopped
refinement level, not a cooldown artifact), the cycle substitutes the capped
prefix one-for-one: it takes the same number of leaves from the *admissible*
universe (`level < max_level`, not in `no_refine`), highest eta first.  The
effective threshold is the lowest marked eta, the effective target is the sum
of the marked eta, and `tau_r`/`tau_h`/the report/coarsen plan/budget exchange
then use those effective values (`want = per * marks`).  This keeps the
standard marked cardinality and bounds a sustained-cap batch to
`per * cap_prefix` leaves per cycle (`per = 6` under `symmetry=:x`, where the
mirror pair may make the actual split count smaller but never larger); the
mass-renormalised "Dörfler on the refinable submesh" variant was measured to
flood the leaf budget on the theta/0.2 ring (24.5k -> 30k leaves in ~35 steps)
and is deliberately not used.  This is a **progress fallback**, not a standard Dörfler property: the
literature and deal.II/ASPECT truncate capped marks and accept stagnation; the
fallback only fires on cycles that would otherwise make zero progress and
never rewrites the marking on a non-stalled cycle.  It requires
`coarsen_gate=true` (the legacy gate-off coarsening branch has no
mark-protection hook).

The wall strip is a **physical distance from the Dirichlet walls** (stage 0 cut
6): `geom` selects the criterion -- the historical Cartesian `y` rows on the
box, the physical `height` and `nu_wall_sides` of a mapped geometry through
[`_amr_in_wall_strip`](@ref).  The refinement and coarsening thresholds
themselves are geometry-independent: they consume the indicator dictionary,
whose mapped body is physical.

**`geom` is required** (stage-0.5 audit defect D5: the default was
`CartesianBox(qt)`, so a caller holding a mapped tree silently ran the box wall
criterion -- on an annulus ring the logical `y` direction is the periodic seam,
not the circles).  The tree stores no geometry, so every caller must state it:
box callers pass `CartesianBox(qt)` (the driver passes its own concrete
geometry), mapped callers the mapped geometry.  Omitting it raises an
`ArgumentError`.

`coarsen_gate` (default `true`) adds deal.II's `coarsening_allowed()` pre-check,
so a quartet is merged only when its parent's external faces keep the 2:1 rule
after
the whole pass, counting the rest of the merge plan (`-1` level) and the refine
plan (`+1`).  The post-merge `balance!` then has nothing to split in the merged
region (`LAST_AMR_BALANCED[].recirc == 0`).  Because the deadband's lowest-error
quartets are usually the geometrically illegal ones (measured 14 of 15),
the plan extends the supply past `tau_h` -- but never past `tau_r`, so a
refine-flagged cell is still never coarsened -- and reports how many merges were
taken that way (`extended`).  `coarsen_gate=false` is the historical behaviour.
"""
function amr_cycle_balanced!(qt::Quadtree, eta::Dict{NTuple{3,Int},Float64};
                             refine_frac::Float64=0.2,
                             coarsen_frac::Float64=0.05,
                             hysteresis::Float64=0.5,
                             max_level::Int=typemax(Int),
                             max_leaves::Int=typemax(Int),
                             symmetry::Symbol=:none,
                             protect::Set{NTuple{3,Int}}=Set{NTuple{3,Int}}(),
                             wall_min_level::Int=0,
                             wall_strip::Float64=0.1,
                             cooldown::Union{Nothing,AmrCooldown}=nothing,
                             hysteresis_cycles::Int=1,
                             coarsen_cells::Union{Nothing,Int}=nothing,
                             coarsen_gate::Bool=true,
                             cap_aware::Bool=false,
                             geom::Union{Nothing,AbstractGeometry}=nothing)
    geom === nothing &&
        throw(ArgumentError("amr_cycle_balanced! now requires the geometry: " *
                            "pass geom=CartesianBox(qt) for a box tree, or the " *
                            "mapped geometry the tree belongs to (the quadtree " *
                            "does not store it, and defaulting to the box " *
                            "silently applied the logical y wall criterion — the " *
                            "periodic seam on an annulus ring — instead of the " *
                            "physical walls; stage-0.5 audit defect D5)."))
    symmetry in (:none, :x) ||
        throw(ArgumentError("symmetry must be :none or :x"))
    max_level >= 1 || throw(ArgumentError("max_level must be >= 1"))
    max_leaves >= 1 || throw(ArgumentError("max_leaves must be >= 1"))
    0.0 <= refine_frac <= 1.0 ||
        throw(ArgumentError("refine_frac must be in [0, 1]"))
    0.0 <= coarsen_frac <= 1.0 ||
        throw(ArgumentError("coarsen_frac must be in [0, 1]"))
    0.0 <= hysteresis <= 1.0 ||
        throw(ArgumentError("hysteresis must be in [0, 1]"))
    hysteresis_cycles >= 0 ||
        throw(ArgumentError("hysteresis_cycles must be >= 0"))
    wall_min_level >= 0 ||
        throw(ArgumentError("wall_min_level must be >= 0"))
    0.0 < wall_strip < 0.5 ||
        throw(ArgumentError("wall_strip must be in (0, 0.5)"))
    _check_cap_aware(cap_aware, coarsen_gate)
    protect = symmetry === :x ? _mirror_protect(qt, protect) : protect
    cool = hysteresis_cycles > 0 ? cooldown : nothing
    cool === nothing || _age_cooldown!(cool, qt.leaves)
    n0 = length(qt.leaves)
    n0 > 0 || return 0, 0, 0
    total = 0.0
    for lf in qt.leaves
        total += max(get(eta, lf, 0.0), 0.0)
    end
    total > 0.0 || return 0, 0, 0
    per = symmetry === :x ? 6 : 3          # leaves added by one pair refinement

    eof(lf) = max(get(eta, lf, 0.0), 0.0)

    # --- minimum-refinement function `l_min(x)`: the wall strip floor ---------
    # The strip is a physical distance from the Dirichlet walls; `geom` (cut 6)
    # selects the box's historical `y` criterion or the mapped geometry's
    # physical height.  `_amr_in_wall_strip` keeps the historical box body.
    in_strip(lf) = _amr_in_wall_strip(geom, qt, lf, wall_strip)
    lmin(lf) = (wall_min_level > 0 && in_strip(lf)) ? wall_min_level : 0

    # --- (a') the two cumulative-error thresholds (exact, by sorting) --------
    asc = sort(collect(qt.leaves), by=lf -> (eof(lf), lf))
    desc = reverse(asc)     # `(eta, leaf)` descending == `(-eta, leaf)` ascending
    tau_r = 0.0
    if refine_frac > 0.0
        tgt_r = refine_frac * total
        cum = 0.0
        for lf in desc
            tau_r = eof(lf)
            cum += tau_r
            cum >= tgt_r && break
        end
    else
        tau_r = Inf
    end
    tau_c = 0.0
    if coarsen_frac > 0.0
        tgt_c = coarsen_frac * total
        cum = 0.0
        for lf in asc
            tau_c = eof(lf)
            cum += tau_c
            cum >= tgt_c && break
        end
    else
        tau_c = -Inf
    end
    tau_h = min(tau_c, hysteresis * tau_r)
    deadband = hysteresis * tau_r < tau_c
    # Global diagnostics are kept even when the opt-in fallback replaces the
    # effective marking thresholds below.
    tau_r_all = tau_r
    mark_target = refine_frac * total
    cap_applied = false
    refinable = 0
    refinable_mass = 0.0
    cap_prefix = 0
    cap_mass = 0.0

    # --- (c') the refinement demand (Dörfler from above) --------------------
    cand = NTuple{3,Int}[]
    want = 0
    for lf in desc
        lf[1] < max_level || continue
        eof(lf) >= tau_r || break        # descending: the rest is below tau_r
        (cool !== nothing && haskey(cool.no_refine, lf)) && continue
        push!(cand, lf)
        want += per
    end

    cap = _cap_aware_marking(cand, want, mark_target, tau_r, tau_h, deadband,
                             cap_applied, refinable, refinable_mass,
                             cap_prefix, cap_mass, desc, eof, total,
                             refine_frac, tau_c, hysteresis, max_level,
                             qt.maxlevel, cap_aware, cool, per)
    cand = cap.cand
    want = cap.want
    mark_target = cap.mark_target
    tau_r = cap.tau_r
    tau_h = cap.tau_h
    deadband = cap.deadband
    cap_applied = cap.cap_applied
    refinable = cap.refinable
    refinable_mass = cap.refinable_mass
    cap_prefix = cap.cap_prefix
    cap_mass = cap.cap_mass

    # --- the eligible quartets, lowest summed indicator first ---------------
    # `coarsen_frac` also caps the cell fraction: the whole pass may not drop
    # more than that many leaves (relative, like every other threshold here).
    capcells = coarsen_cells === nothing ? round(Int, coarsen_frac * n0) :
                                           coarsen_cells
    capcells = max(capcells, 0)
    deficit = max(0, n0 + want - max_leaves)
    need = min(deficit, capcells)
    # The refinement plan the loop below will actually split (the Dörfler prefix
    # up to `refine_frac * total`), needed by the coarsening gate: deal.II's
    # refine flags are an input to `coarsening_allowed`.
    will_refine = Set{NTuple{3,Int}}()
    if coarsen_gate
        cumw = 0.0
        for lf in cand
            cumw >= mark_target && break
            push!(will_refine, lf)
            if symmetry === :x
                ml = _mirror_leaf(qt, lf)
                ml != lf && push!(will_refine, ml)
            end
            cumw += eof(lf)
            if symmetry === :x
                ml = _mirror_leaf(qt, lf)
                ml != lf && (cumw += eof(ml))
            end
        end
    end
    seen = Set{NTuple{3,Int}}()
    qsum = Float64[]
    qkid = NTuple{3,Int}[]
    gate_refused = 0
    gate_extended = 0
    plan_sweeps = 0
    plan_rounds = 0
    plan_capped = false
    if coarsen_gate
        # deal.II `fix_coarsen_flags` semantics: plan the merge set before any
        # mutation, 2:1-legality against the plan's own grants and the refine
        # plan; extend past the deadband only when its legal quartets cannot pay.
        plan = _coarsen_plan(qt, eta, protect, lmin, cool, need, symmetry,
                             will_refine, tau_h, tau_r; asc=asc)
        qkid = plan.order
        gate_refused = plan.refused
        plan_sweeps = plan.sweeps
        plan_rounds = plan.rounds
        plan_capped = plan.capped
        gate_extended = plan.extended
    elseif tau_h > -Inf
        for lf in asc
            lf[1] > lmin(lf) || continue              # level > l_min
            parent = (lf[1] - 1, lf[2] >> 1, lf[3] >> 1)
            parent in seen && continue
            bx = lf[2] - lf[2] % 2; by = lf[3] - lf[3] % 2
            kids = ((lf[1], bx + dx, by + dy) for dy in 0:1 for dx in 0:1)
            kids = collect(kids)
            all(k -> k in qt.leaves, kids) || continue
            any(k -> k in protect, kids) && continue
            all(k -> k[1] > lmin(k), kids) || continue
            m = maximum(eof, kids)
            m <= tau_h || continue
            if symmetry === :x
                ml = _mirror_leaf(qt, lf)
                if ml != lf
                    mbx = ml[2] - ml[2] % 2; mby = ml[3] - ml[3] % 2
                    mk = [(ml[1], mbx + dx, mby + dy) for dy in 0:1, dx in 0:1]
                    all(k -> k in qt.leaves && !(k in protect) &&
                             k[1] > lmin(k), mk) || continue
                    maximum(eof, mk) <= tau_h || continue
                end
            end
            push!(seen, parent)
            push!(qsum, sum(eof, kids))
            push!(qkid, lf)
        end
        perm = sortperm(qsum)
        qsum = qsum[perm]
        qkid = qkid[perm]
    end

    nc = 0
    nex = 0
    dropped = 0
    cursor = 1
    mergedp = NTuple{3,Int}[]      # parents this cycle created by a merge
    note_merged(lf) = begin
        p = (lf[1] - 1, lf[2] >> 1, lf[3] >> 1)
        push!(mergedp, p)
        cool === nothing && return nothing
        cool.no_refine[p] = hysteresis_cycles
        if symmetry === :x
            # `_coarsen_pair!` merges the mirror quartet too: its parent has to
            # be cooled as well, or the mirror half of the pair is re-refined on
            # the next cycle (measured: 12 of 20 merged parents, G210).
            pm = _mirror_leaf(qt, p)
            if pm != p
                cool.no_refine[pm] = hysteresis_cycles
                push!(mergedp, pm)
            end
        end
        return nothing
    end
    # --- (1) the coarsening pass: pay for the refinement in advance ----------
    # With `coarsen_gate=true` the whole plan is executed: its grants may rely on
    # each other (deal.II's flag set is executed as a whole), so stopping in the
    # middle could leave a granted quartet facing a neighbour whose merge did not
    # happen.  The plan is sized to `need`, so the over-shoot is at most one batch.
    while cursor <= length(qkid) && (coarsen_gate || dropped < need)
        lf = qkid[cursor]
        cursor += 1
        lf in qt.leaves || continue
        m = _coarsen_pair!(qt, lf, symmetry, protect)
        m == 0 && continue
        nc += m
        dropped += 3 * m
        note_merged(lf)
    end

    # Live-gated funding merge for the gate path: the plan is exhausted but the
    # refinement still does not fit (`deficit > capcells`).  `_fund_live!` scans
    # the indicator order for a complete quartet whose merge `balance!` will not
    # undo; the already-applied merges are in the tree, so the `granted` set it
    # sees is empty (one reusable set for the whole cycle).
    empty_granted = Set{NTuple{3,Int}}()
    fund_calls = 0
    fund_hits = 0

    # --- (2) the refinement pass, never blocked by the budget ---------------
    # `(c')` A refinement that does not fit is funded by merging the next
    # lowest-sum eligible quartet (3 slots, 6 under `:x`); the loop stops only
    # when no eligible quartet is left (`budget_limited`).
    nmerge = length(qt.leaves)
    nr = 0
    cum = 0.0
    target = mark_target
    limited = false
    for lf in cand
        cum >= target && break
        lf in qt.leaves || continue
        lf[1] < max_level || continue
        (cool !== nothing && haskey(cool.no_refine, lf)) && continue
        while length(qt.leaves) + per > max_leaves
            progressed = false
            while cursor <= length(qkid)
                c = qkid[cursor]
                cursor += 1
                c in qt.leaves || continue
                m = _coarsen_pair!(qt, c, symmetry, protect)
                m == 0 && continue
                nc += m
                nex += 3 * m
                note_merged(c)
                progressed = true
                break
            end
            if !progressed && coarsen_gate
                fund_calls += 1
                m, ext, lff = _fund_live!(qt, asc, eta, protect, lmin, cool,
                                          will_refine, symmetry, tau_r, tau_h,
                                          empty_granted)
                if m > 0
                    nc += m
                    nex += 3 * m
                    ext && (gate_extended += m)
                    note_merged(lff)
                    progressed = true
                    fund_hits += 1
                end
            end
            if !progressed
                limited = true
                break
            end
        end
        limited && break
        n = _refine_pair!(qt, lf, max_level, symmetry)
        n == 0 && continue
        nr += n
        cum += eof(lf)
        if symmetry === :x
            ml = _mirror_leaf(qt, lf)
            ml != lf && (cum += eof(ml))
        end
        if cool !== nothing
            for dy in 0:1, dx in 0:1
                cool.no_coarsen[(lf[1] + 1, 2 * lf[2] + dx, 2 * lf[3] + dy)] =
                    hysteresis_cycles
            end
        end
    end

    n_post = length(qt.leaves)     # leaves before the post-merge balancing
    balance_ok = balance!(qt; symmetry=symmetry)
    mirror_rounds = 0
    mirror_ok = true
    if symmetry === :x
        _, mirror_ok = mirror_close!(qt; max_level=max_level)
        mirror_rounds = LAST_MIRROR_CLOSE[].rounds
        balance_ok = balance!(qt; symmetry=:x) && balance_ok
    end
    recirc = 0
    for p in mergedp
        p in qt.leaves || (recirc += 1)
    end
    LAST_AMR_BALANCED[] = (refined=nr, coarsened=nc, exchanged=nex,
                           n=length(qt.leaves), want=want,
                           eligible=length(qkid), dropped=dropped,
                           tau_r=tau_r, tau_c=tau_c, tau_h=tau_h,
                           deadband=deadband, budget_limited=limited,
                           budget_infeasible=length(qt.leaves) > max_leaves,
                           merged=length(mergedp), recirc=recirc,
                           post=length(qt.leaves) - n_post, parents=mergedp,
                           nmerge=nmerge, npost=n_post,
                           gate_refused=gate_refused,
                           gate_extended=gate_extended,
                           plan_sweeps=plan_sweeps, plan_rounds=plan_rounds,
                           plan_capped=plan_capped,
                           fund_calls=fund_calls, fund_hits=fund_hits,
                           balance_ok=balance_ok, mirror_ok=mirror_ok,
                           mirror_rounds=mirror_rounds,
                           cap_aware=cap_aware, cap_applied=cap_applied,
                           tau_r_all=tau_r_all, mark_target=mark_target,
                           refinable=refinable, refinable_mass=refinable_mass,
                           cap_prefix=cap_prefix, cap_mass=cap_mass)
    return nr, nc, nex
end
