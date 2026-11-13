# AMR policy probe for the realistic card (docs/dev/AMR.md §3-§5).
#
#   julia --project=. examples/amr_policy_probe.jl quantify <ckpt.h5>
#   julia --project=. examples/amr_policy_probe.jl cycles   <ckpt.h5>
#   julia --project=. examples/amr_policy_probe.jl compare  <dirA> <dirB>
#
# `quantify` is the stage-1 measurement: it reproduces, on one checkpoint tree,
# every quantity the classic AMR policy compares, and the three mechanisms that
# freeze the production mesh (docs/dev/BUG_LEDGER.md §5.2-§5.5).  `cycles` replays
# `semquad_amr_passes!` on the same tree with `amr_policy=:classic|:balanced` and
# prints the per-cycle ledger; `CASE_SYMMETRY=none` (default `x`) replays it with
# the x-mirror pair-strict rule off (`docs/dev/AMR.md` §3.6).  `compare` reads two run
# directories (log + frames) and prints the stage-3 A/B table.
#
# Nothing here writes into the run directories; pass a *copy* of a checkpoint of
# a live run (`out_julia/case_realistic_E` is read-only for this work).
using QuaST, Printf

const MS = QuaST

# card knobs (same names as the case script's environment)
envint(k, d) = parse(Int, get(ENV, k, string(d)))
envflt(k, d) = parse(Float64, get(ENV, k, string(d)))
const MAXLEAVES = envint("CASE_MAXLEAVES", 10000)
const THETATOL = envflt("CASE_THETATOL", 1.0e-2)
const COARSEN = envflt("CASE_COARSEN", 0.05)
const REFINE = envflt("CASE_REFINE", 0.2)
const WALLMIN = envint("CASE_WALLMIN", 5)
const WALLSTRIP = envflt("CASE_WALLSTRIP", 0.1)
const POLICY = Symbol(get(ENV, "POLICY", "balanced"))
# x-mirror closure used by the `cycles` replay (default `x`, the value every
# earlier measurement used -- `docs/dev/AMR.md` §3.6).  `CASE_SYMMETRY=none`
# replays the same tree/indicator/policy with the pair-strict rule off, which is
# how `docs/dev/AMR.md` §3.6 isolates what the mirror constraint costs the supply.
const SYMMETRY = Symbol(get(ENV, "CASE_SYMMETRY", "x"))
const NCYC = envint("NCYC", 8)
const HYST = envint("CASE_HYST", 1)

mirrorof(qt, lf) = MS._mirror_leaf(qt, lf)
pct(a, b) = b == 0 ? 0.0 : 100.0 * a / b
q(xs, p) = begin
    isempty(xs) && return NaN
    ys = sort(xs); ys[clamp(ceil(Int, p * length(ys)), 1, length(ys))]
end
function copytree(qt)
    q = Quadtree(qt.maxlevel; Lx=qt.Lx, cells_x=qt.nx0, cells_y=qt.ny0)
    empty!(q.leaves)
    for lf in qt.leaves
        push!(q.leaves, lf)
    end
    return q
end
function census(qt)
    d = Dict{Int,Int}()
    for lf in qt.leaves
        d[lf[1]] = get(d, lf[1], 0) + 1
    end
    return join(["L$(l)=$(d[l])" for l in sort(collect(keys(d)))], " ")
end
function wall_row_levels(qt)
    top = [lf[1] for lf in qt.leaves if lf[3] == qt.ny0 * (1 << lf[1]) - 1]
    bot = [lf[1] for lf in qt.leaves if lf[3] == 0]
    pk(v) = isempty(v) ? "-" :
            join(["L$(l)=$(count(==(l), v))" for l in sort(unique(v))], " ")
    return pk(bot), pk(top)
end

"""Why a quartet merge through `lf` would be refused (`:ok` when it would not)."""
function refusal(qt, lf, protect)
    lf in qt.leaves || return :notleaf
    bx = lf[2] - lf[2] % 2; by = lf[3] - lf[3] % 2
    kids = [(lf[1], bx + dx, by + dy) for dy in 0:1, dx in 0:1]
    all(k -> k in qt.leaves, kids) || return :incomplete
    any(k -> k in protect, kids) && return :protect
    ml = mirrorof(qt, lf)
    if ml != lf
        mbx = ml[2] - ml[2] % 2; mby = ml[3] - ml[3] % 2
        for dy in 0:1, dx in 0:1
            k = (ml[1], mbx + dx, mby + dy)
            (k in qt.leaves && !(k in protect)) || return :mirror
        end
    end
    return :ok
end

# ---------------------------------------------------------------------------
# stage 1: quantify what the classic policy compares and why nothing moves
# ---------------------------------------------------------------------------
function quantify(ckpt::String)
    st = load_semquad_state(ckpt)
    qt = st.qt
    ss = make_semstokes(qt, st.p)
    sq = ss.su
    theta = st.theta
    n = length(qt.leaves)
    @printf("== %s ==\n", ckpt)
    @printf("step=%d t=%.6e dt=%.3e p=%d leaves=%d N=%d area=%.15g maxjump=%d defect=%d maxlevel=%d\n",
            st.step, st.t, st.dt, st.p, n, sq.N, total_area(qt),
            max_face_level_jump(qt), mirror_defect(qt), qt.maxlevel)
    @printf("levels: %s\n", census(qt))
    t0 = time()
    eta = semquad_gradient_jump(sq, qt, theta)
    tot = sum(max(v, 0.0) for v in values(eta))
    @printf("indicator: total=%.6e (%.2f s) min=%.3e max=%.3e mean=%.3e\n",
            tot, time() - t0, minimum(values(eta)), maximum(values(eta)), tot / n)

    # the card's indicator: the wall strip is zeroed when wall_min_level > 0
    wall = Set{NTuple{3,Int}}()
    row_only = Set{NTuple{3,Int}}()
    strip_only = Set{NTuple{3,Int}}()
    for lf in qt.leaves
        h = MS.leaf_hy(qt, lf[1])
        zc = (lf[3] + 0.5) * h
        row = (lf[3] == 0 || lf[3] == qt.ny0 * (1 << lf[1]) - 1)
        row && push!(row_only, lf)
        if row || zc <= WALLSTRIP || zc >= 1.0 - WALLSTRIP
            push!(wall, lf)
        end
        (!row && (zc <= WALLSTRIP || zc >= 1.0 - WALLSTRIP)) && push!(strip_only, lf)
    end
    eta_z = copy(eta)
    for lf in wall
        eta_z[lf] = 0.0
    end
    tot_z = sum(max(v, 0.0) for v in values(eta_z))
    @printf("card indicator (wall strip zeroed): total=%.6e (%.4f of raw)\n",
            tot_z, tot_z / tot)

    # 1. the absolute representation guard `coarsen_theta_tol`
    rng = Dict{NTuple{3,Int},Float64}()
    for lf in sq.leaves
        ids = sq.elem[lf]
        rng[lf] = maximum(theta[ids]) - minimum(theta[ids])
    end
    rv = collect(values(rng))
    @printf("per-leaf theta range (what coarsen_theta_tol=%.1e compares):\n", THETATOL)
    for p in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
        @printf("   p%-4.0f = %10.3e\n", 100p, q(rv, p == 0.0 ? 1e-9 : p))
    end
    ntol = count(v -> v > THETATOL, rv)
    @printf("   %d / %d leaves (%.1f%%) EXCEED the tolerance -> protected; %d below\n",
            ntol, n, pct(ntol, n), n - ntol)
    @printf("   leaves with range <= 1e-3: %d; == 0 exactly: %d\n",
            count(v -> v <= 1.0e-3, rv), count(v -> v == 0.0, rv))

    # 2. Doerfler above/below
    function doerfler(leaves, e, t_, frac)
        tgt = frac * t_
        cum = 0.0
        k = 0
        for lf in leaves
            cum >= tgt && break
            k += 1
            cum += max(e[lf], 0.0)
        end
        return k, cum
    end
    for (tag, e, t_) in (("raw", eta, tot), ("card", eta_z, tot_z))
        desc = sort(collect(qt.leaves), by=lf -> (-max(e[lf], 0.0), lf))
        asc = reverse(desc)
        for frac in (REFINE, COARSEN)
            k, cum = doerfler(desc, e, t_, frac)
            @printf("[%s] Doerfler above %.3f: %5d leaves (%.1f%%) carry %.4f of the error\n",
                    tag, frac, k, pct(k, n), cum / t_)
            k, cum = doerfler(asc, e, t_, frac)
            @printf("[%s] Doerfler below %.3f: %5d leaves (%.1f%%) carry %.4f of the error\n",
                    tag, frac, k, pct(k, n), cum / t_)
        end
        ncoa_cells = round(Int, COARSEN * n)
        cum_cells = sum(max(e[lf], 0.0) for lf in asc[1:ncoa_cells])
        @printf("[%s] cell-count reading: bottom %d leaves (%.1f%%) carry %.4f of the error\n",
                tag, ncoa_cells, pct(ncoa_cells, n), cum_cells / t_)
    end

    # 3. protect-set footprints
    thset = Set(lf for lf in qt.leaves if rng[lf] > THETATOL)
    raw = union(wall, thset)
    closed = MS._mirror_protect(qt, raw)
    @printf("protect sets: wall rows=%d, wall strip-only=%d, wall total=%d (%.1f%% of leaves, %.4f of the error)\n",
            length(row_only), length(strip_only), length(wall), pct(length(wall), n),
            sum(max(eta[lf], 0.0) for lf in wall) / tot)
    @printf("   theta>tol=%d (%.1f%% of leaves, %.4f of the error)\n",
            length(thset), pct(length(thset), n),
            sum(max(eta[lf], 0.0) for lf in thset) / tot)
    @printf("   raw union=%d (%.1f%%), mirror-closed=%d (%.1f%%), FREE=%d (%.2f%%), mirror closure adds %d\n",
            length(raw), pct(length(raw), n), length(closed), pct(length(closed), n),
            n - length(closed), pct(n - length(closed), n),
            count(lf -> !(lf in raw), closed))

    # 4. the cum_c consumption defect (dgquad.jl:433-445)
    function cumc_ledger(e, t_, protect)
        asc = sort(collect(qt.leaves), by=lf -> (max(e[lf], 0.0), lf))
        tgt = COARSEN * t_
        cum = 0.0
        walked = 0
        ch = Dict{Symbol,Float64}(:protect => 0.0, :incomplete => 0.0,
                                  :mirror => 0.0, :ok => 0.0, :notleaf => 0.0)
        nz = 0
        nok = 0
        for lf in asc
            cum >= tgt && break
            walked += 1
            v = max(e[lf], 0.0)
            r = refusal(qt, lf, protect)
            ch[r] += v
            nok += r === :ok ? 1 : 0
            v == 0.0 && (nz += 1)
            cum += v
        end
        return (tgt=tgt, walked=walked, cum=cum, ch=ch, zero=nz, nok=nok,
                tau_c=max(e[asc[min(walked, length(asc))]], 0.0))
    end
    for (tag, e, t_) in (("card", eta_z, tot_z), ("raw", eta, tot))
        L = cumc_ledger(e, t_, closed)
        @printf("[%s] coarsen-loop ledger: target %.3e reached in %d walks\n", tag, L.tgt, L.walked)
        @printf("   budget charged by PROTECTED leaves %.1f%%, INCOMPLETE %.1f%%, MIRROR %.1f%%, free+complete %.1f%%\n",
                pct(L.ch[:protect], L.tgt), pct(L.ch[:incomplete], L.tgt),
                pct(L.ch[:mirror], L.tgt), pct(L.ch[:ok], L.tgt))
        @printf("   zero-eta wall leaves walked: %d (%.1f%%); mergeable walks: %d\n",
                L.zero, pct(L.zero, L.walked), L.nok)
    end

    # 5. the hard budget guard + the driver's refine_frac pin
    desc_z = sort(collect(qt.leaves), by=lf -> (-max(eta_z[lf], 0.0), lf))
    tgt_r = REFINE * tot_z
    cum = 0.0
    iters = 0
    for lf in desc_z
        lf[1] < qt.maxlevel || continue
        cum >= tgt_r && break
        n + 10 > MAXLEAVES && break
        iters += 1
        cum += max(eta_z[lf], 0.0)
    end
    @printf("budget guard: leaves=%d, 'leaves+10 > %d' is %s at the first candidate -> %d iterations, cum %.3e of %.3e\n",
            n, MAXLEAVES, n + 10 > MAXLEAVES, iters, cum, tgt_r)
    @printf("   leaves+6 fits %d pair(s); unconstrained Doerfler wants %d refinements (+%d leaves -> %d)\n",
            max(0, (MAXLEAVES - n) ÷ 6), doerfler(desc_z, eta_z, tot_z, REFINE)[1],
            6 * doerfler(desc_z, eta_z, tot_z, REFINE)[1],
            n + 6 * doerfler(desc_z, eta_z, tot_z, REFINE)[1])

    # 6. counterfactual: relative-from-below with a wall-row-only protect
    for (tag, prot) in (("rows-only", row_only), ("none", Set{NTuple{3,Int}}()))
        q3 = copytree(qt)
        asc = sort(collect(q3.leaves), by=lf -> (max(eta_z[lf], 0.0), lf))
        cumc = 0.0
        merged = 0
        walked = 0
        tgt = COARSEN * tot_z
        while walked < length(asc) && walked <= 4000
            cumc >= tgt && break
            walked += 1
            lf = asc[walked]
            lf in q3.leaves || continue
            max(eta_z[lf], 0.0) <= 1.0e-2 || break   # (any cut; the point is the count)
            m = MS._coarsen_pair!(q3, lf, :x, prot)
            m > 0 || continue
            merged += m
            cumc += max(eta_z[lf], 0.0)
        end
        nraw = length(q3.leaves)
        balance!(q3; symmetry=:x)
        MS.mirror_close!(q3; max_level=qt.maxlevel)
        @printf("counterfactual %-9s: merged %d quartets, leaves %d -> %d (raw) -> %d (balanced, closed), net %+d, census %s\n",
                tag, merged, n, nraw, length(q3.leaves), length(q3.leaves) - n, census(q3))
    end
    return nothing
end

"the leaves the ledger walked, for the final count"
asc_walk(e, L) = begin
    # not used for counting; kept small so the script stays readable
    NTuple{3,Int}[]
end

# ---------------------------------------------------------------------------
# cycles: replay `semquad_amr_passes!` on a checkpoint tree
# ---------------------------------------------------------------------------
function cycles(ckpt::String)
    st = load_semquad_state(ckpt)
    qt = st.qt
    ss = make_semstokes(qt, st.p)
    theta = st.theta
    thp = st.theta_prev
    b, t = wall_row_levels(qt)
    @printf("start: step=%d leaves=%d defect=%d maxjump=%d | %s | %s / %s\n",
            st.step, length(qt.leaves), mirror_defect(qt), max_face_level_jump(qt),
            census(qt), b, t)
    @printf("replay: policy=%s symmetry=%s refine_frac=%.3f coarsen_frac=%.3f max_leaves=%d wall_min_level=%d wall_strip=%.3g\n",
            POLICY, SYMMETRY, REFINE, COARSEN, MAXLEAVES, WALLMIN, WALLSTRIP)
    t0 = time()
    for k in 1:NCYC
        n0 = length(qt.leaves)
        ss, theta, thp, _ = semquad_amr_passes!(
            ss, qt, theta, thp; refine_frac=REFINE, coarsen_frac=COARSEN,
            marking=:error_fraction, max_level=qt.maxlevel, max_leaves=MAXLEAVES,
            wall_min_level=WALLMIN, wall_strip=WALLSTRIP, symmetry=SYMMETRY,
            coarsen_theta_tol=THETATOL, amr_policy=POLICY, amr_hysteresis=HYST)
        s = MS.LAST_AMR_STATS[]
        b, t = wall_row_levels(qt)
        @printf("cycle %2d: %5d -> %5d (%+5d) | refined=%4d coarsened=%3d merged=%3d exchanged=%s | defect=%d maxjump=%d | %s | %s / %s\n",
                k, n0, length(qt.leaves), length(qt.leaves) - n0, s.refined,
                s.coarsened, s.merged, string(get(MS.LAST_AMR_BALANCED[], :exchanged, 0)),
                mirror_defect(qt), max_face_level_jump(qt), census(qt), b, t)
        flush(stdout)
    end
    @printf("%d cycles in %.1f s (%.2f s/cycle)\n", NCYC, time() - t0,
            (time() - t0) / max(NCYC, 1))
    return nothing
end

# ---------------------------------------------------------------------------
# compare: the stage-3 A/B table from two run directories
# ---------------------------------------------------------------------------
"""Parse one case-driver log: steps, AMR cycles, cost, config header."""
function parse_run(logpath::String)
    cfg = ""
    steps = Vector{NTuple{5,Float64}}()          # step, t, Nu_top, Nu_bot, leaves
    amr = 0
    changed = 0
    maxjump_bad = 0
    refsum = 0
    coasum = 0
    walls = Float64[]
    final = ""
    for line in eachline(logpath)
        if startswith(line, "[case] mode=")
            cfg = strip(line)
        elseif occursin("[case] DONE", line)
            m = match(r"wall=([0-9.]+) s \(([0-9.]+) s/step\)", line)
            m === nothing || push!(walls, parse(Float64, m.captures[2]))
        elseif occursin("AMR: N=", line)
            amr += 1
            m = match(r"maxjump=(\d+) refined=(\d+) coarsened=(\d+) merged=(\d+)", line)
            m === nothing && continue
            mj = parse(Int, m.captures[1])
            r = parse(Int, m.captures[2]) + parse(Int, m.captures[3]) +
                parse(Int, m.captures[4])
            mj > 1 && (maxjump_bad += 1)
            r > 0 && (changed += 1)
            refsum += parse(Int, m.captures[2])
            coasum += parse(Int, m.captures[3])
        elseif occursin(r"^\s+step \d+", line)
            m = match(r"step\s+(\d+) t=([0-9.e+-]+) dt=([0-9.e+-]+) Nu=([0-9.e+-]+)/([0-9.e+-]+).*leaves=(\d+)", line)
            m === nothing && continue
            push!(steps, (parse(Float64, m.captures[1]), parse(Float64, m.captures[2]),
                          parse(Float64, m.captures[4]), parse(Float64, m.captures[5]),
                          parse(Float64, m.captures[6])))
        elseif occursin("[case] final:", line)
            final = strip(line)
        end
    end
    return (cfg=cfg, steps=steps, amr=amr, changed=changed,
            maxjump_bad=maxjump_bad, refined=refsum, coarsened=coasum,
            wall=isempty(walls) ? NaN : walls[end], final=final)
end

"""x-averaged boundary-layer thickness of one 256x256 frame."""
function frame_bl(Tbin::String, meshcsv::String; ng::Int=256,
                  Twall_top=0.0, Twall_bot=1.0)
    T = reshape(reinterpret(Float64, read(Tbin)), ng, ng)   # T[i,j] = T(x_i,y_j)
    dy = 1.0 / (ng - 1)
    pb = [sum(view(T, :, j)) / ng for j in 1:ng]
    Tmid = pb[(ng + 1) ÷ 2]
    dtop = (3pb[ng] - 4pb[ng - 1] + pb[ng - 2]) / (2dy)
    dbot = (-3pb[1] + 4pb[2] - pb[3]) / (2dy)
    lb = Int[]; lt = Int[]
    for line in eachline(meshcsv)
        v = split(line, ",")
        l = parse(Int, v[1]); y0 = parse(Float64, v[3]); h = parse(Float64, v[4])
        y0 < 1e-12 && push!(lb, l)
        abs(y0 + h - 1.0) < 1e-12 && push!(lt, l)
    end
    hbot = isempty(lb) ? NaN : 1.0 / (2 * 2.0^maximum(lb))
    htop = isempty(lt) ? NaN : 1.0 / (2 * 2.0^maximum(lt))
    dtu = abs(Tmid - Twall_top) / abs(dtop)
    dbo = abs(Tmid - Twall_bot) / abs(dbot)
    return (Tmid=Tmid, Nu_top=-dtop, Nu_bot=dbot, delta_top=dtu, delta_bot=dbo,
            h_top=htop, h_bot=hbot, hd_top=htop / dtu, hd_bot=hbot / dbo,
            lvl_top=isempty(lt) ? 0 : maximum(lt),
            lvl_bot=isempty(lb) ? 0 : maximum(lb))
end

function frames_of(dir::String)
    fdir = joinpath(dir, "frames")
    isdir(fdir) || return String[]
    fs = sort(filter(f -> endswith(f, "_T.bin"), readdir(fdir; join=true)))
    return fs
end

function compare(dirA::String, dirB::String)
    A = parse_run(dirA * ".log")
    B = parse_run(dirB * ".log")
    for (tag, R, d) in (("classic ", A, dirA), ("balanced", B, dirB))
        st = R.steps
        isempty(st) && continue
        n0 = st[1][5]; n1 = st[end][5]
        @printf("%s %s\n", tag, R.cfg)
        @printf("  steps %d..%d, leaves %d -> %d (min %d max %d), N=%d\n",
                Int(st[1][1]), Int(st[end][1]), Int(n0), Int(n1),
                Int(minimum(x -> x[5], st)), Int(maximum(x -> x[5], st)),
                Int(st[end][5]))
        @printf("  AMR cycles %d, with any change %d (%.1f%%), refined total %d, coarsened total %d, maxjump>1 count %d\n",
                R.amr, R.changed, pct(R.changed, R.amr), R.refined, R.coarsened,
                R.maxjump_bad)
        @printf("  Nu_top %.6f -> %.6f (mean last 40 steps %.6f), Nu_bot %.6f -> %.6f (mean %.6f)\n",
                st[1][3], st[end][3],
                sum(x -> x[3], st[max(1, end - 39):end]) / length(st[max(1, end - 39):end]),
                st[1][4], st[end][4],
                sum(x -> x[4], st[max(1, end - 39):end]) / length(st[max(1, end - 39):end]))
        @printf("  cost: %.3f s/step (wall %.0f s)\n", R.wall, R.wall * length(st))
    end
    if !isempty(A.steps) && !isempty(B.steps)
        na = sum(x -> x[3], A.steps[max(1, end - 39):end]) / length(A.steps[max(1, end - 39):end])
        nb = sum(x -> x[3], B.steps[max(1, end - 39):end]) / length(B.steps[max(1, end - 39):end])
        @printf("Nu_top: classic %.6f vs balanced %.6f -> %+.3f %%\n", na, nb, pct(nb - na, abs(na)))
        ba = sum(x -> x[4], A.steps[max(1, end - 39):end]) / length(A.steps[max(1, end - 39):end])
        bb = sum(x -> x[4], B.steps[max(1, end - 39):end]) / length(B.steps[max(1, end - 39):end])
        @printf("Nu_bot: classic %.6f vs balanced %.6f -> %+.3f %%\n", ba, bb, pct(bb - ba, abs(ba)))
        @printf("cost: %.3f vs %.3f s/step -> %+.1f %%\n", A.wall, B.wall, pct(B.wall - A.wall, A.wall))
    end
    for (tag, d) in (("classic ", dirA), ("balanced", dirB))
        fs = frames_of(d)
        isempty(fs) && continue
        for f in unique([fs[1], fs[cld(length(fs), 2)], fs[end]])
            m = replace(f, "_T.bin" => "_mesh.csv")
            bl = frame_bl(f, m)
            nm = match(r"frame_(\d+)_T", basename(f))
            @printf("%s frame %s: Nu_top=%.4f Nu_bot=%.4f delta_top=%.5f delta_bot=%.5f h_top=%s h_bot=%s h/delta_top=%.3f h/delta_bot=%.3f (rows L%s/L%s)\n",
                    tag, nm.captures[1], bl.Nu_top, bl.Nu_bot, bl.delta_top,
                    bl.delta_bot, string(round(bl.h_top; digits=5)),
                    string(round(bl.h_bot; digits=5)), bl.hd_top, bl.hd_bot,
                    string(bl.lvl_bot), string(bl.lvl_top))
        end
    end
    return nothing
end


mode = length(ARGS) >= 1 ? ARGS[1] : "quantify"
if mode == "quantify"
    quantify(length(ARGS) >= 2 ? ARGS[2] : "/tmp/s330/E_spinup.h5")
elseif mode == "cycles"
    cycles(length(ARGS) >= 2 ? ARGS[2] : "/tmp/s330/E_spinup.h5")
elseif mode == "compare"
    length(ARGS) >= 3 || error("compare needs two run directories")
    compare(ARGS[2], ARGS[3])
else
    error("unknown mode `$mode`; use quantify|cycles|compare")
end
