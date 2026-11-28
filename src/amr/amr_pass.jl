# One adaptive pass of the coupled quadtree SEM convection loop.
#
# `semquad_amr_passes!` runs `passes` consecutive AMR cycles on the same time
# level: gradient-jump indicator, Dörfler marking, coarsening, the wall rule and
# the 2:1 balance, each cycle transferring `theta`, `theta_prev` and whatever the
# caller hands it.  It is the function the driver calls every `amr_every` steps;
# the policy it implements is described in the user guide.

"""
    semquad_amr_passes!(ss, qt, theta, theta_prev; passes=1, refine_frac=0.2,
                        coarsen_frac=0.0)

Run `passes` consecutive AMR cycles on the same time level.  Each pass recomputes
the gradient-jump indicator on the current mesh, refines/balances, rebuilds the
Taylor–Hood spaces, and transfers `theta`/`theta_prev`.  `passes=1` is the
original one-level-per-step behaviour; larger values are a numerical budget to
let interior plumes catch up with a heavily refined wall.  Every pass may trade
coarsening for refinement (the budget exchange), so `coarsen_frac` is not zeroed
after the first pass.  Returns `(ss, theta, theta_prev, nrefined)`.

The cycle is [`amr_cycle_balanced!`](@ref): relative
Dörfler thresholds from above and below, budget exchange instead of a refine
stop at the budget, a wall-row-only protect set and an `AmrCooldown` hysteresis.
It is the only adaptive policy: the `:classic` fraction + hard-budget path was
removed at `8c622df`. `refine_wall_to_min!` and the hard leaf
budget pass below are part of the cycle, not of a policy.

`indicator` selects the field the jump estimator is applied to: `:theta` (the
default and the production path), `:velocity`, or `:both` (the theta and
velocity indicators, each divided by its own total, added --
[`semquad_combined_jump`](@ref)).  The two non-theta modes require the velocity
that is currently on the mesh: pass it as `ux`/`uz` (scalar node layout, length
`ss.su.N`).  They are validated only when the selected indicator reads them, so
existing `:theta` callers are unaffected, and a multi-pass non-theta cycle
transfers the two components along with `theta` between passes.

`amr_cap_aware` (default `false`) forwards the opt-in max-level Dörfler
fallback of [`amr_cycle_balanced!`](@ref): when the standard marked prefix is
entirely at `max_level`, that cycle substitutes the capped prefix one-for-one
with the highest-eta refinable, non-cooled leaves (effective `tau_r` = lowest
marked eta, target = sum of the marked eta).  It requires
`amr_coarsen_gate=true` and leaves every historical default trajectory
bit-for-bit unchanged.

`amr_cooldown` is the `AmrCooldown` hysteresis state of the balanced cycle: pass
the same object to every call of a run so the no-refine/no-coarsen cooldowns and
the cycle counter survive across calls.  A direct caller that omits it gets a
fresh `AmrCooldown()` for that call only -- there is no shared hysteresis state
between separate calls without the keyword.  The production driver owns one state
for the whole run (`run_semquad_simulation` creates it, restores it on a resume,
returns it as `cooldown` and passes it here) exactly as it does for the
`initial_refine`/in-loop cycle calls.
"""
const LAST_AMR_STATS = Ref((refined=0, coarsened=0, merged=0, leaves=0))

function semquad_amr_passes!(ss::SemStokes, qt::Quadtree,
                             theta::Vector{Float64}, theta_prev::Vector{Float64};
                             refine_frac::Float64=0.2,
                             coarsen_frac::Float64=0.0,
                             passes::Int=1,
                             max_level::Int=typemax(Int),
                             max_leaves::Int=typemax(Int),
                             wall_min_level::Int=0,
                             wall_strip::Float64=0.1,
                             symmetry::Symbol=:none,
                             transfer_mode::Symbol=:nodal,
                             hotspot_weight::Float64=0.0, hotspot_frac::Float64=0.5,
                             hotspot_min_level::Int=0, hotspot_mode::Symbol=:theta, hotspot_zmin::Float64=0.0, zero_side_eta::Bool=false, side_strip::Float64=0.05,
                             top_wall_level::Int=0, top_wall_frac::Float64=0.5,
                             amr_cooldown::Union{Nothing,AmrCooldown}=nothing,
                             amr_hysteresis::Int=1,
                             amr_coarsen_gate::Bool=true,
                             amr_cap_aware::Bool=false,
                             indicator::Symbol=:theta,
                             # Velocity components on the scalar (Q2) node
                             # layout of `ss`.  Only `:velocity` and `:both`
                             # read them; `:theta` ignores them, so every
                             # existing caller keeps its bitwise result.
                             # They are required (and length-checked) exactly
                             # when the selected indicator needs them; the old
                             # code read `ux`/`uz` here without ever receiving
                             # them, which threw `UndefVarError` on the first
                             # AMR cycle of every `:velocity`/`:both` run (G234
                             # now pins that they are passed and used).
                             ux::Union{Nothing,Vector{Float64}}=nothing,
                             uz::Union{Nothing,Vector{Float64}}=nothing)
    passes >= 1 || throw(ArgumentError("passes must be >= 1"))
    amr_hysteresis >= 0 ||
        throw(ArgumentError("amr_hysteresis must be >= 0"))
    amr_cap_aware && !amr_coarsen_gate &&
        throw(ArgumentError("amr_cap_aware requires amr_coarsen_gate=true: " *
                            "the legacy gate-off coarsening branch has no " *
                            "mark-protection hook for the fallback marks"))
    indicator in (:theta, :velocity, :both) ||
        throw(ArgumentError("indicator must be :theta, :velocity or :both"))
    cool = amr_cooldown === nothing ? AmrCooldown() : amr_cooldown
    sq = ss.su
    g = sq.geom
    # Stage 0 cut 6: the indicator, the transfer and the balanced cycle are
    # mapped.  The remaining Cartesian-y AMR *policies* (top-wall forcing,
    # hotspot temperature binning, side-strip eta zeroing) are written against
    # the physical `y`/`x` and are not routed through the geometry yet; refuse
    # them on a mapped geometry instead of running the box bodies silently on a
    # mapped mesh (the driver refuses them before setup too).
    if !(g isa CartesianBox)
        (top_wall_level == 0 && hotspot_weight == 0.0 &&
         hotspot_min_level == 0 && !zero_side_eta) ||
            throw(ArgumentError("a mapped geometry ($(geometry_name(g))) does " *
                                "not support the Cartesian-y AMR policies " *
                                "(top_wall_level, hotspot_weight/" *
                                "hotspot_min_level, zero_side_eta) in stage 0: " *
                                "their wall rows and bins are physical-y " *
                                "criteria.  Pass top_wall_level=0, " *
                                "hotspot_weight=0.0, hotspot_min_level=0 and " *
                                "zero_side_eta=false."))
    end
    if indicator !== :theta
        ux === nothing &&
            throw(ArgumentError("indicator=$(indicator) needs the velocity " *
                                "x-component: pass ux (and uz) on the scalar " *
                                "node layout of ss"))
        uz === nothing &&
            throw(ArgumentError("indicator=$(indicator) needs the velocity " *
                                "z-component: pass uz (and ux) on the scalar " *
                                "node layout of ss"))
        length(ux) == sq.N ||
            throw(ArgumentError("ux node field has $(length(ux)) entries, " *
                                "mesh needs $(sq.N)"))
        length(uz) == sq.N ||
            throw(ArgumentError("uz node field has $(length(uz)) entries, " *
                                "mesh needs $(sq.N)"))
    end
    nr = 0
    nc = 0
    merged_total = 0
    for pass in 1:passes
        # `indicator` selects what the jump estimator is applied to.  `:theta` is
        # the historical field and the default; `:velocity` sees a grid-scale
        # velocity error that a smooth temperature cannot (PHASE1.md 1.1).
        # `:both` normalises each field by its own total before adding
        # (`semquad_combined_jump`), so the velocity's raw 1e13-1e14 scale
        # advantage at Ra=1e8 cannot turn the sum back into `:velocity`.
        eta = indicator === :velocity ?
              semquad_velocity_jump(sq, qt, ux, uz) :
              indicator === :both ?
              semquad_combined_jump(sq, qt, theta, ux, uz) :
              semquad_gradient_jump(sq, qt, theta)

        protect = Set{NTuple{3,Int}}()
        # (c') Only the first/last wall rows are protected: the strip's minimum
        # level is guaranteed by `refine_wall_to_min!` below, and the absolute
        # theta-range guard of the removed `:classic` policy has no replacement
        # -- the relative Dörfler cut inside `amr_cycle_balanced!` decides which
        # quartets may be coarsened.  On a mapped geometry these are the rows on
        # the geometry's physical Dirichlet walls (`nu_wall_sides`, the annulus's
        # circles), not the logical top/bottom rows.  The box branch is the
        # historical body, so the box path pays no new abstract-typed dispatch.
        if g isa CartesianBox
            for lf in qt.leaves
                if lf[3] == 0 || lf[3] == qt.ny0 * (1 << lf[1]) - 1
                    push!(protect, lf)
                end
            end
        else
            for lf in qt.leaves
                if _amr_wall_leaf(g, qt, lf)
                    push!(protect, lf)
                end
            end
        end
        if zero_side_eta
            for lf in qt.leaves
                hx = leaf_hx(qt, lf[1])
                xc = (lf[2] + 0.5) * hx
                if xc <= side_strip || xc >= sq.Lx - side_strip
                    eta[lf] = 0.0
                end
            end
        end
        if top_wall_level > 0
            n = sq.p + 1
            ds = Dict{NTuple{3,Int},Float64}()
            dmax_top = 0.0; dmax_bot = 0.0
            for lf in qt.leaves
                (lf[3] == 0 || lf[3] == qt.ny0 * (1 << lf[1]) - 1) || continue
                ids = sq.elem[lf]
                h = leaf_hy(qt, lf[1])
                jf = lf[3] == 0 ? 1 : n
                d = 0.0
                for j in 1:n
                    c = (2.0 / h) * sq.D[jf, j]
                    for i in 1:n
                        d = max(d, abs(c * theta[ids[(j - 1) * n + i]]))
                    end
                end
                ds[lf] = d
                if lf[3] == 0
                    dmax_bot = max(dmax_bot, d)
                else
                    dmax_top = max(dmax_top, d)
                end
            end
            if dmax_top + dmax_bot > 0.0
                target = 10.0 * maximum(values(eta))
                for (lf, d) in ds
                    lf[1] < top_wall_level || continue
                    dm = lf[3] == 0 ? dmax_bot : dmax_top
                    d > top_wall_frac * dm && (eta[lf] = max(eta[lf], target))
                end
            end
        end
        if hotspot_weight > 0.0 || hotspot_min_level > 0
            if hotspot_mode === :anomaly
                nb = 256
                sums = zeros(Float64, nb); cnts = zeros(Int, nb)
                for g in 1:sq.N
                    b = clamp(floor(Int, sq.coords[g][2] * nb) + 1, 1, nb)
                    sums[b] += theta[g]; cnts[b] += 1
                end
                means = [cnts[b] > 0 ? sums[b] / cnts[b] : 0.0
                         for b in 1:nb]
            end
            hotval = function (lf)
                ids = sq.elem[lf]
                m = -Inf
                for g in ids
                    v = theta[g]
                    if hotspot_mode === :anomaly
                        b = clamp(floor(Int, sq.coords[g][2] * nb) + 1, 1, nb)
                        v -= means[b]
                    end
                    m = max(m, v)
                end
                m
            end
            vals = Dict(lf => hotval(lf) for lf in qt.leaves)
            thmax = maximum(values(vals))
            if thmax > 0.0
                th0 = hotspot_frac * thmax
                if hotspot_weight > 0.0
                    for lf in qt.leaves
                        hot = (vals[lf] - th0) / (thmax - th0)
                        hot > 0.0 && (eta[lf] *= 1.0 + hotspot_weight * hot)
                    end
                end
                if hotspot_min_level > 0
                    target = 10.0 * maximum(values(eta))
                    for lf in qt.leaves
                        lf[1] < hotspot_min_level || continue
                        lf in protect && continue
                        zc = (lf[3] + 0.5) * leaf_hy(qt, lf[1])
                        zc >= hotspot_zmin || continue
                        vals[lf] > th0 &&
                            (eta[lf] = max(eta[lf], target))
                    end
                end
            end
        end
        # (a')+(b')+(d'): relative coarsening, budget exchange, hysteresis.
        # The exchange has to be able to trade on every pass, so the coarsen
        # fraction is not zeroed after pass 1.  The box branch is the historical
        # call body (no mapped `protect`/strip bodies), so the box path's
        # arithmetic is unchanged; stage-0.5 audit D5 makes `geom` a required
        # keyword and the box branch passes its already-concrete geometry, so no
        # default `CartesianBox` object is constructed.
        if g isa CartesianBox
            n, nc0, _nex = amr_cycle_balanced!(qt, eta;
                        refine_frac=refine_frac, coarsen_frac=coarsen_frac,
                        max_level=max_level, max_leaves=max_leaves,
                        symmetry=symmetry, protect=protect,
                        wall_min_level=wall_min_level, wall_strip=wall_strip,
                        cooldown=cool, hysteresis_cycles=amr_hysteresis,
                        coarsen_gate=amr_coarsen_gate,
                        cap_aware=amr_cap_aware, geom=g)
            refine_wall_to_min!(qt, wall_min_level; strip=wall_strip,
                                symmetry=symmetry, geom=g)
        else
            n, nc0, _nex = amr_cycle_balanced!(qt, eta;
                        refine_frac=refine_frac, coarsen_frac=coarsen_frac,
                        max_level=max_level, max_leaves=max_leaves,
                        symmetry=symmetry, protect=protect,
                        wall_min_level=wall_min_level, wall_strip=wall_strip,
                        cooldown=cool, hysteresis_cycles=amr_hysteresis,
                        coarsen_gate=amr_coarsen_gate,
                        cap_aware=amr_cap_aware, geom=g)
            refine_wall_to_min!(qt, wall_min_level; strip=wall_strip,
                                symmetry=symmetry, geom=g)
        end
        # The wall refinement runs after the cycle's budget exchange and can push
        # the tree over max_leaves.  Re-run the budget pass (no refine / no
        # coarsen fractions) so leaves <= max_leaves is a hard invariant.
        # Hard leaf budget.  The cycle itself does not coarsen a quartet that
        # the Dörfler cut did not select, and 2:1 balancing can still add
        # leaves, so merge the lowest-error complete quartets here (the removed
        # classic budget pass called the fraction cycle with zero fractions,
        # whose guard gave up as soon as the first leaf in indicator order could
        # not be merged).
        #
        # `symmetry=:x`: the repair pass closes a tree that the wall refinement
        # (or a legacy checkpoint) left asymmetric, and the paired merges of the
        # budget pass preserve closure, so the pass ends mirror-symmetric.
        if symmetry === :x
            mirror_close!(qt; max_level=max_level)
            balance!(qt; symmetry=:x)
        end
        eta_cap = copy(eta)
        for lf in qt.leaves
            get!(eta_cap, lf, 0.0)
        end
        merged, budget_ok = semquad_enforce_leaf_budget!(
            qt, eta_cap; max_leaves=max_leaves, protect=protect, symmetry=symmetry)
        merged_total += merged
        merged > 0 && @debug "AMR budget: merged $merged quartets -> $(length(qt.leaves)) leaves"
        budget_ok || @warn "leaf budget not reachable: $(length(qt.leaves)) leaves > max_leaves=$max_leaves (all remaining quartets are protected or unbalanced)"
        nr += n
        nc += nc0
        ss2 = make_semstokes(qt, ss.p; geom=ss.su.geom)
        theta = transfer_semquad(sq, theta, ss2.su; mode=transfer_mode,
                                 zero_walls=true, qt_new=qt)
        theta_prev = transfer_semquad(sq, theta_prev, ss2.su;
                                      mode=transfer_mode, zero_walls=true,
                                      qt_new=qt)
        if indicator !== :theta
            # A multi-pass velocity / combined cycle must see the velocity on
            # the tree the previous pass built.  Both components live on the
            # scalar node layout, so they follow the same transfer as theta;
            # `zero_walls=false` because a free-slip wall keeps its tangential
            # velocity (theta's wall value is the Dirichlet data, the
            # velocity's is not).
            ux = transfer_semquad(sq, ux, ss2.su; mode=transfer_mode,
                                  zero_walls=false, qt_new=qt)
            uz = transfer_semquad(sq, uz, ss2.su; mode=transfer_mode,
                                  zero_walls=false, qt_new=qt)
        end
        ss = ss2
        sq = ss2.su
    end
    LAST_AMR_STATS[] = (refined=nr, coarsened=nc, merged=merged_total,
                        leaves=length(qt.leaves))
    return ss, theta, theta_prev, nr
end
