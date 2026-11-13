# Coupled quadtree SEM convection loop: variable-viscosity Stokes + BDF2 energy
# + periodic AMR (gradient-jump indicator, refine/coarsen, solution transfer).
#
# This is the first end-to-end AMR convection driver.  It supports fixed or
# CFL-controlled dt and Picard iterations (`maxpic`); the dt update is done at
# the start of each step on the current mesh, because the AMR block rebuilds the
# mesh and zeroes the velocity at the end of the step.

"""
    semquad_eta_at(sq, theta, x, y, pr, leaves)

Viscosity at `(x,y)` used by the quadtree convection loop:

    T = Tbar(y, H) + theta(x,y),   eta = eta_at(T, y, pr).

This supports `:expT` (temperature only) and `:expTz` (temperature + depth) with
exactly the same formula and `eta_min/eta_max` bounds as the uniform assembler.
"""
function semquad_eta_at(sq::SemQuad, theta::Vector{Float64}, x::Float64,
                        y::Float64, pr::Params, leaves::Set{NTuple{3,Int}})
    T = Tbar(y, pr.H) + _semquad_eval_at(sq, theta, x, y, leaves)
    return eta_at(T, y, pr)[1]
end

"""
    refine_wall_to_min!(qt, min_level; strip=0.1, symmetry=:none)

Refine leaves whose centers lie within `strip` of the bottom/top walls until
they reach at least `min_level`, then balance.  This is the numerical
counterpart of ASPECT's boundary refinement criterion / minimum refinement:
error-fraction coarsening must not remove the thermal boundary layer.

`symmetry=:x` refines the mirror partner with every wall leaf and uses the
mirror-aware `balance!`, so the wall refinement cannot break the x-mirror
invariant (the wall criterion depends only on `iy`, so on a closed tree it is
symmetric by itself; the pairing matters for a tree that is already damaged).
"""
function refine_wall_to_min!(qt::Quadtree, min_level::Int; strip::Float64=0.1,
                             symmetry::Symbol=:none)
    min_level <= 0 && return 0
    0.0 < strip < 0.5 || throw(ArgumentError("strip must be in (0, 0.5)"))
    symmetry in (:none, :x) || throw(ArgumentError("symmetry must be :none or :x"))
    nref = 0
    for _ in 1:min_level
        changed = false
        for lf in collect(qt.leaves)
            h = leaf_hy(qt, lf[1])
            zc = (lf[3] + 0.5) * h
            if (lf[3] == 0 || lf[3] == qt.ny0 * (1 << lf[1]) - 1 ||
                zc <= strip || zc >= 1.0 - strip) && lf[1] < min_level
                if symmetry === :x
                    ml = _mirror_leaf(qt, lf)
                    (ml == lf || ml in qt.leaves) || continue
                    refine!(qt, lf) || continue
                    nref += 1
                    changed = true
                    ml != lf && refine!(qt, ml) && (nref += 1)
                elseif refine!(qt, lf)
                    nref += 1
                    changed = true
                end
            end
        end
        changed || break
        balance!(qt; symmetry=symmetry)
    end
    return nref
end

"""
    semquad_amr_passes!(ss, qt, theta, theta_prev; passes=1, refine_frac=0.2,
                        coarsen_frac=0.0, marking=:global)

Run `passes` consecutive AMR cycles on the same time level.  Each pass recomputes
the gradient-jump indicator on the current mesh, refines/balances, rebuilds the
Taylor–Hood spaces, and transfers `theta`/`theta_prev`.  `passes=1` is the
original one-level-per-step behaviour; larger values are a numerical budget to
let interior plumes catch up with a heavily refined wall.  Coarsening is applied
only on the first pass.  Returns `(ss, theta, theta_prev, nrefined)`.

`amr_policy=:classic` is the historical policy (and this unit-level helper's
default, so the AMR primitive itself stays classic): an absolute
`coarsen_theta_tol` protect set, the error-fraction coarsening of `amr_cycle!`
inside a hard `max_leaves` guard, and a `wall_strip`-wide wall protect set.
`amr_policy=:balanced` (docs/dev/AMR.md §3) replaces those three by the
relative Dörfler-from-below criterion, budget exchange and a wall-row-only
protect set, with an `AmrCooldown` hysteresis.  Since 2026-11-13 `:balanced` is
the default of `run_semquad_simulation` (docs/dev/AMR.md);
this helper keeps `:classic` so `t_g55`/`t_g208` still pin the primitive.
"""
const LAST_AMR_STATS = Ref((refined=0, coarsened=0, merged=0, leaves=0))

function semquad_amr_passes!(ss::SemStokes, qt::Quadtree,
                             theta::Vector{Float64}, theta_prev::Vector{Float64};
                             refine_frac::Float64=0.2,
                             coarsen_frac::Float64=0.0,
                             marking::Symbol=:global, passes::Int=1, amr_strategy::Symbol=:legacy,
                             refine_thresh::Float64=0.5, coarsen_thresh::Float64=0.1,
                             max_level::Int=typemax(Int),
                             max_leaves::Int=typemax(Int),
                             wall_min_level::Int=0,
                             wall_strip::Float64=0.1,
                             symmetry::Symbol=:none,
                             transfer_mode::Symbol=:nodal,
                             coarsen_theta_tol::Float64=Inf, hotspot_weight::Float64=0.0, hotspot_frac::Float64=0.5,
                             hotspot_min_level::Int=0, hotspot_mode::Symbol=:theta, hotspot_zmin::Float64=0.0, zero_side_eta::Bool=false, side_strip::Float64=0.05,
                             top_wall_level::Int=0, top_wall_frac::Float64=0.5,
                             amr_policy::Symbol=:classic,
                             amr_cooldown::Union{Nothing,AmrCooldown}=nothing,
                             amr_hysteresis::Int=1,
                             amr_coarsen_gate::Bool=true)
    passes >= 1 || throw(ArgumentError("passes must be >= 1"))
    amr_policy in (:classic, :balanced) ||
        throw(ArgumentError("amr_policy must be :classic or :balanced"))
    amr_hysteresis >= 0 ||
        throw(ArgumentError("amr_hysteresis must be >= 0"))
    balanced = amr_policy === :balanced
    cool = balanced ? (amr_cooldown === nothing ? SEMQUAD_AMR_COOLDOWN[] :
                       amr_cooldown) : nothing
    sq = ss.su
    nr = 0
    nc = 0
    merged_total = 0
    for pass in 1:passes
        eta = semquad_gradient_jump(sq, qt, theta)
        protect = Set{NTuple{3,Int}}()
        if balanced
            # (c') Only the first/last leaf rows are protected: the strip's
            # minimum level is guaranteed by `refine_wall_to_min!` below, and
            # the absolute `coarsen_theta_tol` guard is replaced by the
            # relative Dörfler cut inside `amr_cycle_balanced!`.
            for lf in qt.leaves
                if lf[3] == 0 || lf[3] == qt.ny0 * (1 << lf[1]) - 1
                    push!(protect, lf)
                end
            end
        end
        if !balanced && wall_min_level > 0
            # Wall cells get their resolution from refine_wall_to_min!, not
            # from the error-fraction budget.  Zero their indicator and protect
            # them from coarsening so boundary layers do not consume the whole
            # refinement budget / repeatedly displace interior plumes.
            for lf in qt.leaves
                h = leaf_hy(qt, lf[1])
                zc = (lf[3] + 0.5) * h
                if lf[3] == 0 || lf[3] == qt.ny0 * (1 << lf[1]) - 1 ||
                   zc <= wall_strip || zc >= 1.0 - wall_strip
                    eta[lf] = 0.0
                    push!(protect, lf)
                end
            end
        end
        if !balanced && isfinite(coarsen_theta_tol) && coarsen_frac > 0.0
            for lf in sq.leaves
                ids = sq.elem[lf]
                lo = minimum(theta[ids])
                hi = maximum(theta[ids])
                hi - lo > coarsen_theta_tol && push!(protect, lf)
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
        if balanced
            # (a')+(b')+(d'): relative coarsening, budget exchange, hysteresis.
            # The exchange has to be able to trade on every pass, so the
            # coarsen fraction is not zeroed after pass 1 here.
            n, nc0, _nex = amr_cycle_balanced!(qt, eta;
                        refine_frac=refine_frac, coarsen_frac=coarsen_frac,
                        max_level=max_level, max_leaves=max_leaves,
                        symmetry=symmetry, protect=protect,
                        wall_min_level=wall_min_level, wall_strip=wall_strip,
                        cooldown=cool, hysteresis_cycles=amr_hysteresis,
                        coarsen_gate=amr_coarsen_gate)
        elseif amr_strategy === :threshold
            n, nc0 = amr_cycle_hysteresis!(qt, eta;
                       refine_thresh=refine_thresh, coarsen_thresh=coarsen_thresh,
                       max_level=max_level, max_leaves=max_leaves,
                       symmetry=symmetry, protect=protect)
        else
            n, nc0 = amr_cycle!(qt, eta; refine_frac=refine_frac,
                          coarsen_frac=(pass == 1 ? coarsen_frac : 0.0),
                          marking=marking, max_level=max_level,
                          max_leaves=max_leaves, symmetry=symmetry,
                          protect=protect)
        end
        refine_wall_to_min!(qt, wall_min_level; strip=wall_strip,
                            symmetry=symmetry)
        # The wall refinement runs after amr_cycle!'s hard leaf budget and can
        # push the tree over max_leaves.  Re-run the budget pass (no refine /
        # no coarsen fractions) so leaves <= max_leaves is a hard invariant.
        # Hard leaf budget.  With coarsen_frac = 0 the cycle itself cannot
        # coarsen, and 2:1 balancing can still add leaves, so merge the
        # lowest-error complete quartets here (the old budget pass called
        # amr_cycle! with zero fractions, whose guard gave up as soon as the
        # first leaf in indicator order could not be merged).
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
        ss2 = make_semstokes(qt, ss.p)
        theta = transfer_semquad(sq, theta, ss2.su; mode=transfer_mode,
                                 zero_walls=true, qt_new=qt)
        theta_prev = transfer_semquad(sq, theta_prev, ss2.su;
                                      mode=transfer_mode, zero_walls=true,
                                      qt_new=qt)
        ss = ss2
        sq = ss2.su
    end
    LAST_AMR_STATS[] = (refined=nr, coarsened=nc, merged=merged_total,
                        leaves=length(qt.leaves))
    return ss, theta, theta_prev, nr
end

