# Coarsening-gate probe for `amr_policy=:balanced` (docs/dev/AMR.md §4.1).
#
#   julia --project=. examples/amr_coarsen_gate_probe.jl trace <ckpt.h5>
#
# Stage 1 of the `coarsening_allowed()` work: replay the production AMR pass on a
# checkpoint tree with the phases measured separately, so the leaves that the
# policy counters do not account for can be attributed:
#
#   cycle   = `amr_cycle_balanced!` (merges + refinements + post-merge balance)
#   wall    = `refine_wall_to_min!`
#   budget  = `semquad_enforce_leaf_budget!` (the hard `max_leaves` pass)
#
# For every cycle it reports
#
#   * `merged` quartet merges and `recirc`: how many of those parents the
#     post-merge `balance!` split again (the recirculation), split into merges
#     that were already 2:1-illegal at merge time (`recirc_merge`) and merges
#     that only became illegal because the refinement pass refined a neighbour
#     (`recirc_refine`),
#   * `foreign`: splits of leaves that were *not* merged this cycle (cascades),
#   * the leaves each phase re-created outside the policy counters
#     (`residual = leaves_after - leaves_before - 3*refined + 3*coarsened`),
#   * the 2:1 violation count / max jump before and after every phase,
#   * how much of the coarsening supply a 2:1 pre-check would keep, and the
#     face/level pairs of the quartets it would refuse.
#
# Nothing here writes into a run directory; pass a *copy* of a checkpoint.
#
#   CASE_SYMMETRY=none|x   mirror rule of the replay (default none: the
#                          production choice of `docs/dev/AMR.md` §3.6; `x`
#                          reproduces §3.6's arm)
#   NCYC=12                cycles to replay
#   CASE_MAXLEAVES=10000 CASE_COARSEN=0.05 CASE_REFINE=0.2
#   CASE_WALLMIN=5 CASE_WALLSTRIP=0.1 CASE_HYST=1
#   VERIFY=1               also run `semquad_amr_passes!` on a copy of the same
#                          state and check that the two agree (cycle 1), which
#                          validates the phase split
using QuaST, Printf

const MS = QuaST

envint(k, d) = parse(Int, get(ENV, k, string(d)))
envflt(k, d) = parse(Float64, get(ENV, k, string(d)))

const MAXLEAVES = envint("CASE_MAXLEAVES", 10000)
const COARSEN   = envflt("CASE_COARSEN", 0.05)
const REFINE    = envflt("CASE_REFINE", 0.2)
const WALLMIN   = envint("CASE_WALLMIN", 5)
const WALLSTRIP = envflt("CASE_WALLSTRIP", 0.1)
const HYST      = envint("CASE_HYST", 1)
const HYSPARAM  = envflt("CASE_HYSPARAM", 0.5)
const SYMMETRY  = Symbol(get(ENV, "CASE_SYMMETRY", "none"))
const NCYC      = envint("NCYC", 12)
const VERIFY    = envint("VERIFY", 1) == 1
# The coarsening pre-check (`coarsening_allowed`): 0 is the pre-gate behaviour,
# 1 the plan.  Default 0 here so `NCYC` replays of a pre-gate checkpoint
# reproduce the old numbers; `CASE_AMR_GATE=1` is the Stage-2/3 arm.
const GATE      = envint("CASE_AMR_GATE", 0) == 1

const DIRS = (:xminus, :xplus, :yminus, :yplus)

# --------------------------------------------------------------- tree utils ----
function copytree(qt::Quadtree)
    q = Quadtree(qt.maxlevel; Lx=qt.Lx, cells_x=qt.nx0, cells_y=qt.ny0)
    empty!(q.leaves)
    for lf in qt.leaves
        push!(q.leaves, lf)
    end
    return q
end

pq(lf::NTuple{3,Int}) = (lf[1] - 1, lf[2] >> 1, lf[3] >> 1)
kids_of(p::NTuple{3,Int}) = [(p[1] + 1, 2p[2] + dx, 2p[3] + dy) for dy in 0:1, dx in 0:1]

"Complete sibling quartet keys and their one representative leaf, one per quartet."
function completions(qt::Quadtree)
    out = NTuple{3,Int}[]
    for lf in sort(collect(qt.leaves))
        (lf[2] % 2 == 0 && lf[3] % 2 == 0) || continue
        all(k -> k in qt.leaves, kids_of(pq(lf))) && push!(out, lf)
    end
    return out
end

facechildren(dir::Symbol, kids::Matrix{NTuple{3,Int}}) =
    dir === :xminus ? vec(kids[:, 1]) :
    dir === :xplus  ? vec(kids[:, 2]) :
    dir === :yminus ? vec(kids[1, :]) : vec(kids[2, :])

"""
    gate_offenders(qt, kids)

External face neighbours of the would-be parent that are two levels finer than
it, i.e. the 2:1 violations a merge would create: `[(dir, child_level,
neighbour_leaf), ...]`.  Empty means `coarsening_allowed()` holds.  This is the
deal.II `fix_coarsen_flags` neighbour test of docs/dev/AMR.md §2.2.
"""
function gate_offenders(qt::Quadtree, kids::Matrix{NTuple{3,Int}})
    l = kids[1][1]
    inside = Set(vec(kids))
    off = Tuple{Symbol,Int,NTuple{3,Int}}[]
    for dir in DIRS, c in facechildren(dir, kids)
        c in qt.leaves || continue
        for nb in MS.neighbors(qt, c, dir)
            nb in inside && continue
            nb[1] > l && push!(off, (dir, l, nb))
        end
    end
    return off
end

"Leaf/face pairs that violate the 2:1 rule, and the worst jump."
function violations(qt::Quadtree)
    n = 0
    worst = 0
    for lf in qt.leaves, dir in DIRS
        for nb in MS.neighbors(qt, lf, dir)
            d = nb[1] - lf[1]
            d >= 2 && (n += 1)
            worst = max(worst, abs(d))
        end
    end
    return n, worst
end

row(qt, lf) = lf[3] == 0 || lf[3] == qt.ny0 * (1 << lf[1]) - 1
wall_rows(qt) = Set{NTuple{3,Int}}(lf for lf in qt.leaves if row(qt, lf))
eof(eta, lf) = max(get(eta, lf, 0.0), 0.0)
in_strip(qt, lf) = begin
    lf[3] == 0 && return true
    lf[3] == qt.ny0 * (1 << lf[1]) - 1 && return true
    zc = (lf[3] + 0.5) * MS.leaf_hy(qt, lf[1])
    return zc <= WALLSTRIP || zc >= 1.0 - WALLSTRIP
end
lmin(qt, lf) = (WALLMIN > 0 && in_strip(qt, lf)) ? WALLMIN : 0

"The two cumulative-error thresholds of `amr_cycle_balanced!`, recomputed here."
function thresholds(eta, leaves, refine_frac, coarsen_frac)
    total = sum(eof(eta, lf) for lf in leaves)
    asc = sort(collect(leaves), by=lf -> (eof(eta, lf), lf))
    desc = reverse(asc)
    tau_r = 0.0
    if refine_frac > 0.0
        tgt = refine_frac * total
        cum = 0.0
        for lf in desc
            tau_r = eof(eta, lf)
            cum += tau_r
            cum >= tgt && break
        end
    else
        tau_r = Inf
    end
    tau_c = 0.0
    if coarsen_frac > 0.0
        tgt = coarsen_frac * total
        cum = 0.0
        for lf in asc
            tau_c = eof(eta, lf)
            cum += tau_c
            cum >= tgt && break
        end
    else
        tau_c = -Inf
    end
    return (tau_r=tau_r, tau_c=tau_c, tau_h=min(tau_c, HYSPARAM * tau_r),
            total=total, asc=asc)
end

"The quartets `amr_cycle_balanced!` would consider merging, with the 2:1 verdict."
function eligible_quartets(qt, eta, protect, th)
    out = NamedTuple[]
    seen = Set{NTuple{3,Int}}()
    for lf in th.asc
        lf[1] > lmin(qt, lf) || continue
        p = pq(lf)
        p in seen && continue
        kids = kids_of(p)
        all(k -> k in qt.leaves, kids) || continue
        any(k -> k in protect, kids) && continue
        all(k -> k[1] > lmin(qt, k), kids) || continue
        maximum(k -> eof(eta, k), kids) <= th.tau_h || continue
        if SYMMETRY === :x
            ml = MS._mirror_leaf(qt, lf)
            if ml != lf
                mk = kids_of(pq(ml))
                all(k -> k in qt.leaves && !(k in protect) &&
                         k[1] > lmin(qt, k), mk) || continue
                maximum(k -> eof(eta, k), mk) <= th.tau_h || continue
            end
        end
        push!(seen, p)
        off = gate_offenders(qt, kids)
        push!(out, (leaf=lf, parent=p, level=lf[1], legal=isempty(off),
                    offenders=off, qsum=sum(k -> eof(eta, k), kids)))
    end
    return out
end

tally!(d::Dict{Tuple{Symbol,Int,Int},Int}, off) = begin
    for o in unique(off)
        key = (o[1], o[2], o[3][1])
        d[key] = get(d, key, 0) + 1
    end
    d
end

# --------------------------------------------------------------------- trace ----
function trace(ckpt::String)
    st = load_semquad_state(ckpt)
    qt = st.qt
    ss = make_semstokes(qt, st.p)
    sq = ss.su
    theta = copy(st.theta)
    thp = copy(st.theta_prev)
    cool = AmrCooldown()
    p = ss.p
    @printf("== trace %s ==\n", ckpt)
    @printf("step=%d leaves=%d N=%d p=%d maxlevel=%d Lx=%.3f cells=%dx%d maxjump=%d defect=%d\n",
            st.step, length(qt.leaves), sq.N, p, qt.maxlevel, qt.Lx, qt.nx0,
            qt.ny0, max_face_level_jump(qt), mirror_defect(qt))
    @printf("policy=balanced symmetry=%s gate=%d refine=%.3f coarsen=%.3f max_leaves=%d wall_min=%d wall_strip=%.3g hyst=%d\n",
            SYMMETRY, GATE, REFINE, COARSEN, MAXLEAVES, WALLMIN, WALLSTRIP, HYST)
    @printf("%-4s %6s %9s %5s %5s %5s %6s %5s %5s %5s %5s %5s %6s %5s %6s\n",
            "cyc", "n0", "tau_h", "elig", "legl", "illg", "merged", "illm",
            "rec", "post", "wall", "bmrg", "gref", "gext", "n1")
    t0 = time()
    S = Dict{Symbol,Int}(:n0 => 0, :pol => 0, :wall => 0, :close => 0,
                         :budget => 0, :ref => 0, :coa => 0, :bmrg => 0,
                         :rec => 0, :recM => 0, :recR => 0, :merged => 0,
                         :illm => 0, :foreign => 0, :splits => 0)
    for k in 1:NCYC
        eta = MS.semquad_gradient_jump(sq, qt, theta)
        protect = wall_rows(qt)
        th = thresholds(eta, qt.leaves, REFINE, COARSEN)
        elig = eligible_quartets(qt, eta, protect, th)
        nlegal = count(x -> x.legal, elig)
        n0 = length(qt.leaves)
        preclass = Dict(x.parent => x for x in elig)
        before = Set(qt.leaves)
        v0, mj0 = violations(qt)

        # ---- phase 1: the AMR cycle ---------------------------------------
        nr, nc, nex = amr_cycle_balanced!(qt, eta; refine_frac=REFINE,
                                          coarsen_frac=COARSEN,
                                          hysteresis=HYSPARAM,
                                          max_level=qt.maxlevel,
                                          max_leaves=MAXLEAVES,
                                          symmetry=SYMMETRY, protect=protect,
                                          wall_min_level=WALLMIN,
                                          wall_strip=WALLSTRIP,
                                          cooldown=cool,
                                          hysteresis_cycles=HYST,
                                          coarsen_gate=GATE)
        d = LAST_AMR_BALANCED[]
        n1 = length(qt.leaves)
        res_policy = n1 - n0 - 3 * nr + 3 * nc
        added = setdiff(Set(qt.leaves), before)
        mergedset = Set{NTuple{3,Int}}()
        survivors = NTuple{3,Int}[]
        for pp in added
            kids = kids_of(pp)
            all(k -> k in before, kids) || continue
            push!(survivors, pp)
            push!(mergedset, pp)
        end
        # every node that was merged and re-split is not in `added`; but its
        # parent is in `mergedp` only inside the library, so `recirc` comes from
        # the counter.  The re-split *children* are in `added` though: group them
        # by parent to find the cascades (splits of leaves that were not merged).
        childparents = Dict{NTuple{3,Int},Int}()
        survset = Set(survivors)
        for c in added
            c in survset && continue
            pcp = pq(c)
            childparents[pcp] = get(childparents, pcp, 0) + 1
        end
        foreign = 0
        for (pcp, cnt) in childparents
            cnt == 4 || continue
            # a re-split merged parent is not visible here (it is not `added`),
            # so a 4-child parent that was a leaf before is a foreign split
            pcp in before && (foreign += 1)
        end
        ilmerg = count(pp -> haskey(preclass, pp) && !preclass[pp].legal, survivors)
        v1, mj1 = violations(qt)

        # ---- phase 2: wall refinement -------------------------------------
        nref_wall = MS.refine_wall_to_min!(qt, WALLMIN; strip=WALLSTRIP,
                                           symmetry=SYMMETRY)
        n2 = length(qt.leaves)
        res_wall = n2 - n1 - 3 * nref_wall

        # ---- phase 3: (:x) closure, then the hard budget pass --------------
        nref_close = 0
        if SYMMETRY === :x
            nref_close, _ = MS.mirror_close!(qt; max_level=qt.maxlevel)
            balance!(qt; symmetry=:x)
        end
        n3 = length(qt.leaves)
        res_close = n3 - n2 - 3 * nref_close
        eta_cap = copy(eta)
        for lf in qt.leaves
            get!(eta_cap, lf, 0.0)
        end
        bmerg, bok = MS.semquad_enforce_leaf_budget!(
            qt, eta_cap; max_leaves=MAXLEAVES, protect=protect,
            symmetry=SYMMETRY)
        n4 = length(qt.leaves)
        res_budget = n4 - n3 + 3 * bmerg

        # ---- cross-check against the production entry point ---------------
        if VERIFY && k == 1
            qc = Quadtree(qt.maxlevel; Lx=qt.Lx, cells_x=qt.nx0, cells_y=qt.ny0)
            empty!(qc.leaves)
            for lf in before
                push!(qc.leaves, lf)
            end
            ssc = make_semstokes(qc, p)
            thc = copy(st.theta)
            thpc = copy(st.theta_prev)
            MS.semquad_amr_passes!(
                ssc, qc, thc, thpc; refine_frac=REFINE, coarsen_frac=COARSEN,
                marking=:error_fraction, max_level=qt.maxlevel,
                max_leaves=MAXLEAVES, wall_min_level=WALLMIN,
                wall_strip=WALLSTRIP, symmetry=SYMMETRY,
                coarsen_theta_tol=1.0e-2, amr_policy=:balanced,
                amr_hysteresis=HYST, amr_coarsen_gate=GATE)
            s = MS.LAST_AMR_STATS[]
            ok = length(qc.leaves) == n4 && s.refined == nr &&
                 s.coarsened == nc && s.merged == bmerg
            @printf("   verify cycle 1: pass leaves=%d (probe %d), refined=%d (probe %d), coarsened=%d (probe %d), merged=%d (probe %d) -> %s\n",
                    length(qc.leaves), n4, s.refined, nr, s.coarsened, nc,
                    s.merged, bmerg, ok ? "IDENTICAL" : "DIFFERS")
        end

        # ---- offender tallies ---------------------------------------------
        tal = Dict{Tuple{Symbol,Int,Int},Int}()
        for x in elig
            x.legal && continue
            tally!(tal, x.offenders)
        end
        # every merged parent the post-merge balance! split again, attributed:
        # `recM` = it was already 2:1-illegal at merge time (its offending
        # neighbour did not merge); `recR` = a neighbour was refined after the
        # merge, so the 2:1 violation was created by the refinement pass.
        rec = d.recirc
        recM = 0
        recR = 0
        rectal = Dict{Tuple{Symbol,Int,Int},Int}()
        for pp in d.parents
            pp in qt.leaves && continue
            pre = get(preclass, pp, nothing)
            if pre === nothing
                recM += 1
            elseif pre.legal
                recR += 1
            elseif all(o -> pq(o[3]) in mergedset, pre.offenders)
                recR += 1            # the offending neighbours merged too
            else
                recM += 1
                tally!(rectal, pre.offenders)
            end
        end
        @printf("%-4d %6d %9.2e %5d %5d %5d %6d %5d %5d %5d %5d %5d %6d %5d %6d\n",
                k, n0, th.tau_h, length(elig), nlegal, length(elig) - nlegal,
                d.merged, ilmerg, rec, d.post, nref_wall, bmerg,
                d.gate_refused, d.gate_extended, n4)
        @printf("     residual leaves: policy %+d (post %+d) | wall %+d | close %+d | budget %+d | total %+d | 2:1 before %d/%d after %d/%d\n",
                res_policy, d.post, res_wall, res_close, res_budget,
                res_policy + res_wall + res_close + res_budget, v0, mj0, v1, mj1)
        @printf("     check: n0=%d nmerge=%d npost=%d n1=%d | refined=%d coarsened=%d merged=%d post=%d recirc=%d gate_refused=%d extended=%d\n",
                n0, d.nmerge, d.npost, n1, nr, nc, d.merged, d.post, d.recirc,
                d.gate_refused, d.gate_extended)
        @printf("     gate would refuse %d of %d eligible (%.1f%%): %s\n",
                length(elig) - nlegal, length(elig),
                100.0 * (length(elig) - nlegal) / max(length(elig), 1),
                join(["$(x[1]) L$(x[2])->L$(x[3]) x$(tal[x])"
                      for x in sort(collect(keys(tal)), by=x -> (-tal[x], x))], " "))
        if !isempty(rectal)
            @printf("     recirculated merge-induced faces: %s\n",
                    join(["$(x[1]) L$(x[2])->L$(x[3]) x$(rectal[x])"
                          for x in sort(collect(keys(rectal)), by=x -> (-rectal[x], x))], " "))
        end
        flush(stdout)

        S[:n0] += n0; S[:pol] += res_policy; S[:wall] += res_wall
        S[:close] += res_close; S[:budget] += res_budget
        S[:ref] += nr; S[:coa] += nc; S[:bmrg] += bmerg
        S[:rec] += rec; S[:recM] += recM; S[:recR] += recR
        S[:merged] += d.merged; S[:illm] += ilmerg; S[:foreign] += foreign

        # ---- advance the state exactly like `semquad_amr_passes!` ----------
        ss2 = make_semstokes(qt, p)
        theta = transfer_semquad(sq, theta, ss2.su; mode=:nodal,
                                 zero_walls=true, qt_new=qt)
        thp = transfer_semquad(sq, thp, ss2.su; mode=:nodal,
                               zero_walls=true, qt_new=qt)
        ss = ss2
        sq = ss2.su
    end
    dt = time() - t0
    @printf("\nsummary: %d cycles in %.1f s (%.2f s/cycle)\n", NCYC, dt, dt / NCYC)
    @printf("  refined leaves %d, policy-coarsened quartets %d, budget-merged quartets %d\n",
            S[:ref], S[:coa], S[:bmrg])
    @printf("  recirculated merges %d of %d (%.1f%%): merge-induced %d, refinement-induced %d; foreign splits %d\n",
            S[:rec], S[:merged], 100.0 * S[:rec] / max(S[:merged], 1),
            S[:recM], S[:recR], S[:foreign])
    @printf("  leaves re-created outside the counters: policy %+d, wall %+d, close %+d, budget %+d (total %+d)\n",
            S[:pol], S[:wall], S[:close], S[:budget],
            S[:pol] + S[:wall] + S[:close] + S[:budget])
    @printf("  merged quartets a 2:1 pre-check would have refused (survivors): %d\n",
            S[:illm])
    return nothing
end

mode = length(ARGS) >= 1 ? ARGS[1] : "trace"
ckpt = length(ARGS) >= 2 ? ARGS[2] : "/tmp/s330/E_spinup.h5"

# -------------------------------------------------------------------- supply ----
"""
    supply(ckpt)

The coarsening supply of one tree in eta-ascending order: every complete sibling
quartet with its summed indicator, whether `amr_cycle_balanced!`'s criterion
would take it (`max(eta) <= tau_h`) and whether a 2:1 pre-check would allow the
merge.  Answers "how much does the gate cost the exchange, and how far down the
error order must the pass walk to find a legal quartet".
"""
function supply(ckpt::String)
    st = load_semquad_state(ckpt)
    qt = st.qt
    ss = make_semstokes(qt, st.p)
    sq = ss.su
    eta = MS.semquad_gradient_jump(sq, qt, st.theta)
    protect = wall_rows(qt)
    th = thresholds(eta, qt.leaves, REFINE, COARSEN)
    @printf("== supply %s ==\n", ckpt)
    @printf("step=%d leaves=%d tau_r=%.4e tau_c=%.4e tau_h=%.4e deadband=%s\n",
            st.step, length(qt.leaves), th.tau_r, th.tau_c, th.tau_h,
            HYSPARAM * th.tau_r < th.tau_c)
    rows = NamedTuple[]
    for lf in completions(qt)
        p = pq(lf)
        kids = kids_of(p)
        any(k -> k in protect, kids) && continue
        all(k -> k[1] > lmin(qt, k), kids) || continue
        off = gate_offenders(qt, kids)
        push!(rows, (parent=p, level=lf[1], qsum=sum(k -> eof(eta, k), kids),
                     maxe=maximum(k -> eof(eta, k), kids), legal=isempty(off),
                     eligible=maximum(k -> eof(eta, k), kids) <= th.tau_h,
                     offenders=off))
    end
    sort!(rows, by=r -> (r.qsum, r.parent))
    n = length(rows)
    nlegal = count(r -> r.legal, rows)
    nelig = count(r -> r.eligible, rows)
    neligl = count(r -> r.eligible && r.legal, rows)
    @printf("complete quartets %d: legal %d (%.1f%%), criterion-eligible %d, eligible AND legal %d\n",
            n, nlegal, 100.0 * nlegal / max(n, 1), nelig, neligl)
    @printf("%-5s %-6s %10s %10s %6s %6s  %s\n", "rank", "level", "qsum",
            "max eta", "elig", "legal", "offenders")
    for (i, r) in enumerate(rows)
        i > 30 && break
        @printf("%-5d L%-5d %10.3e %10.3e %6s %6s  %s\n", i, r.level, r.qsum,
                r.maxe, r.eligible ? "yes" : "-", r.legal ? "yes" : "NO",
                isempty(r.offenders) ? "" :
                join(["$(o[1]) L$(o[2])->L$(o[3][1])" for o in unique(r.offenders)], ","))
    end
    # how deep must the pass walk to pay for `want` refinements?
    for want in (15, 50, 150)
        k = 0
        legal = 0
        lasteta = 0.0
        for r in rows
            k += 1
            if r.legal
                legal += 1
                lasteta = r.qsum
            end
            legal >= want && break
        end
        @printf("  %3d legal quartets are found within the first %d of %d in eta order (taken sum %.3e)\n",
                want, k, n, sum(r.qsum for r in rows[1:min(k, n)]))
    end
    # the reference: what the hard budget pass would merge (it takes the lowest
    # summed eta among *all* complete quartets, retrying on refusal)
    q = copytree(qt)
    ecap = copy(eta)
    for lf in q.leaves
        get!(ecap, lf, 0.0)
    end
    n0 = length(q.leaves)
    merged, ok = MS.semquad_enforce_leaf_budget!(q, ecap;
                                                 max_leaves=n0 - 3 * count(r -> r.eligible, rows),
                                                 protect=protect, symmetry=:none)
    @printf("budget reference: merging down to %d leaves takes %d quartets (reached=%s), tree %d -> %d\n",
            n0 - 3 * nelig, merged, ok, n0, length(q.leaves))
    return nothing
end

if mode == "trace"
    trace(ckpt)
elseif mode == "supply"
    supply(ckpt)
else
    error("unknown mode `$mode`; use trace|supply")
end
