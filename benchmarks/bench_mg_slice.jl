#!/usr/bin/env julia
# =============================================================================
# bench_mg_slice.jl -- stage-A instrument of the cache-blocked / slice-wise
# leaf-kernel question (docs/dev/MG.md).
#
#   STATE=F julia --project=. --threads=1 benchmarks/bench_mg_slice.jl
#   STATE=P1 julia --project=. --threads=1 benchmarks/bench_mg_slice.jl
#
# It builds the production MG hierarchy of the state and prints:
#
#   * CACHE: the L1d / L2 / L3 sizes this CPU reports (sysfs);
#   * WS / WSSUM / WSTAB: the exact byte size of one `SEMQUAD_KBLK`-lane
#     block's live data -- the kernel scratch, the eta lane-table slice, the id
#     slice, the basis tables, and the distinct 64 B lines the block's ids touch
#     in the source and destination vectors -- as a fraction of L1d;
#   * the raw leaf kernel per level (`raw_us`, per leaf, and the historical
#     scalar kernel `blk=0` as the in-run reference), so the per-leaf cost can be
#     compared across levels whose streamed vectors differ 4x in size;
#   * RAWCACHE: the finest level hot against the same call after streaming
#     32 MiB (24 MiB L3 + 8 MiB) through the caches, plus the same for the
#     gather form of the eta read and a flush control (a 1 MiB copy, which must
#     slow down when flushed -- if it does not, the flush is not working);
#   * a memory-bandwidth calibration and the kernel's position against it;
#   * with BENCH_ASM=1: the instruction mix of the compiled kernel body.
#
# The width sweep of §16.2 is four runs of this script with
# `SEMQUAD_KBLK` (src/semquadstokes.jl) set to 8 / 16 / 32 / 64 in an
# extracted tree (`git archive HEAD | tar -x -C /tmp/ms_blk16`, then
# `REPO=/tmp/ms_blk16`); the scalar column is the in-run normaliser for the
# session drift.
#
# The checkpoint is copied to a temp dir and opened read-only.  BLAS is pinned
# to one thread.  The script writes nothing into the repository.
# =============================================================================
using QuaST, Printf, LinearAlgebra, SparseArrays
import InteractiveUtils

const MS = QuaST
BLAS.set_num_threads(1)
const REPO = get(ENV, "REPO", normpath(joinpath(@__DIR__, "..")))
const STATE = get(ENV, "STATE", "F")
const CKPT = get(ENV, "CKPT", STATE == "P1" ?
                 joinpath(REPO, "out_julia", "s340", "ab_none", "resume.h5") :
                 joinpath(REPO, "out_julia", "_s7", "case", "pre_step6131.h5"))

function params_from_config(cfg::Dict{String,String})
    g(k, d) = haskey(cfg, k) ? parse(Float64, cfg[k]) : d
    s(k, d) = haskey(cfg, k) ? Symbol(cfg[k]) : d
    return Params(Ra=g("Ra", 1.0e5), H=g("H", 10.0),
                  eta_law=:composite, comp_rule=s("comp_rule", :isostress),
                  comp_Ad=g("comp_Ad", 1.6240183119e5),
                  comp_As=g("comp_As", 2.6796419238e-1),
                  comp_Ed=g("comp_Ed", 12.0), comp_Es=g("comp_Es", 17.0),
                  comp_Vd=g("comp_Vd", 0.0), comp_Vs=g("comp_Vs", 0.0),
                  comp_n=g("comp_n", 3.0), comp_Tcut=g("comp_Tcut", 0.0),
                  comp_eps_min=g("comp_eps_min", 1.0e-12),
                  eta_T0=g("eta_T0", 0.5), eta_min=g("eta_min", 1.0e-4),
                  eta_max=g("eta_max", 1.0e4),
                  bc=:free, latent_L=g("latent_L", 0.11),
                  latent_mode=s("latent_mode", :per_phase),
                  yield_c=g("yield_c", Inf), yield_phi=g("yield_phi", 0.0),
                  yield_taumax=g("yield_taumax", Inf),
                  yield_c_grad=g("yield_c_grad", 0.0),
                  yield_mu_grad=g("yield_mu_grad", 0.0),
                  weak_f=g("weak_f", 1.0), weak_eps_ref=g("weak_eps_ref", 0.0),
                  rho_heating=false, rho_cp=false, Di=g("Di", 0.0),
                  heating_adiabatic=false, heating_dissipation=false,
                  refstate=nothing, deviatoric=false)
end

function phases_from_config(cfg::Dict{String,String})
    haskey(cfg, "phases") || return PhaseTransition[]
    out = PhaseTransition[]
    for row in split(cfg["phases"], ';')
        a = parse.(Float64, split(row, ','))
        push!(out, PhaseTransition(a[1], a[2], a[3], a[4], a[5], a[6]))
    end
    return out
end

# best-of-`reps` mean-of-`inner` nanosecond timer
function best_ns(f, reps::Int=7; target_ns::Float64=2.0e6)
    f()
    t1 = @elapsed f()
    inner = max(1, ceil(Int, target_ns / max(t1 * 1e9, 1.0)))
    best = Inf
    for _ in 1:reps
        t0 = time_ns()
        for _ in 1:inner
            f()
        end
        t = (time_ns() - t0) / inner
        best = min(best, t)
    end
    return best
end

# 32 MiB streamed between timed calls, published through a mutable global so
# the loads cannot be deleted.  `zeros(...)` alone is not enough: the pages of an
# untouched anonymous mapping are the shared zero page, so reading 32 MiB of it
# touches 4 KiB of cache.
const FLUSH = fill!(zeros(3 * 1024 * 1024), 1.0)   # 24 MiB = L3
const FLUSH2 = fill!(zeros(1024 * 1024), 1.0)      # extra 8 MiB
const SINK = Ref(0.0)
function flushcache!()
    s = 0.0
    @inbounds for i in eachindex(FLUSH)
        s += FLUSH[i]
    end
    @inbounds for i in eachindex(FLUSH2)
        s += FLUSH2[i]
    end
    SINK[] = s
    return nothing
end
# cold: flush, then time exactly one call; best of `reps`
function cold_ns(f, reps::Int=9)
    f()
    best = Inf
    for _ in 1:reps
        flushcache!()
        t0 = time_ns()
        f()
        t = time_ns() - t0
        best = min(best, t)
    end
    return best
end

function cachesizes()
    out = Dict{Int,Int}()
    for c in 0:3
        d = "/sys/devices/system/cpu/cpu0/cache/index$c"
        isdir(d) || continue
        lvl = tryparse(Int, strip(read(joinpath(d, "level"), String)))
        lvl === nothing && continue
        siz = strip(read(joinpath(d, "size"), String))
        mult = endswith(siz, "K") ? 1024 : endswith(siz, "M") ? 1024^2 : 1
        val = tryparse(Int, siz[1:prevind(siz, end)]) * mult
        typ = strip(read(joinpath(d, "type"), String))
        typ == "Data" || typ == "Unified" || continue
        out[lvl] = max(get(out, lvl, 0), val)
    end
    return out
end

function main()
    nt = Threads.nthreads()
    @printf("STATE=%s NT=%d julia=%s cpu=%s\n", STATE, nt, VERSION, Sys.CPU_NAME)
    isfile(CKPT) || error("missing checkpoint $CKPT")
    tmpdir = mktempdir(; cleanup=false)
    copyto = joinpath(tmpdir, basename(CKPT))
    cp(CKPT, copyto; force=true)
    chmod(copyto, 0o444)
    st = load_semquad_state(copyto)
    qt = st.qt; p = st.p; theta = st.theta; ux = st.ux; uz = st.uz
    pr = params_from_config(st.config); ph = phases_from_config(st.config)
    ss = make_semstokes(qt, p)
    leaves = Set(qt.leaves)
    Tfun = (x, y) -> MS._semquad_eval_at(ss.su, theta, x, y, leaves)
    eta_fun = (x, y) -> begin
        T = Tbar(y, pr.H) + Tfun(x, y)
        ed = semquad_strain_rate_at(ss.su, ux, uz, x, y, leaves)
        eta_composite_at(T, ed, y, pr)[1]
    end

    mg = MS.make_semquad_mg(ss, qt, p, :free, false, eta_fun, nothing;
                            cycles=1, smoother=:jacobi,
                            coarse_operator=:rediscretize,
                            operator=:matrixfree, coarsen_factor=1.5,
                            smoothing=:auto, mg_threads=min(nt, 8))
    nlev = mg.geo.L + 1
    caches = cachesizes()
    @printf("CACHE L1d=%d L2=%d L3=%d bytes\n",
            get(caches, 1, 0), get(caches, 2, 0), get(caches, 3, 0))
    @printf("HIER levels=%d nfs=%s nleaf=%s mg_threads=%d blk=%d\n", nlev,
            join(mg.geo.nfs, ","),
            join([length(mg.geo.qts[k].leaves) for k in 1:nlev], ","),
            mg.threads, mg.mfops[1].blk)

    # ------------------------------------------------------------------
    # 1. exact working set of one lane block at the finest level
    # ------------------------------------------------------------------
    kf = nlev
    op = mg.mfops[kf]::MS.SemQuadKOp
    etav = mg.mfetas[kf]::AbstractVector{Float64}
    work = mg.mfwork[kf]::MS.SemQuadKWork
    scr = work.bs
    BLK = MS.SEMQUAD_KBLK
    n = op.n; nq = op.nq; n2 = n * n; nq2 = nq * nq; N_ = op.N
    nl = op.nleaf
    @printf("KERNEL p=%d n=%d nq=%d n2=%d nq2=%d N=%d nleaf=%d BLK=%d\n",
            op.p, n, nq, n2, nq2, N_, nl, BLK)
    rows = Tuple{String,String,Int}[]
    push!(rows, ("bxv  (gathered x nodes)", @sprintf("%dx%d", BLK, n2), sizeof(scr.bxv)))
    push!(rows, ("bzv  (gathered z nodes)", @sprintf("%dx%d", BLK, n2), sizeof(scr.bzv)))
    push!(rows, ("buxx (du_x/dx)", @sprintf("%dx%dx%d", BLK, nq, nq), sizeof(scr.buxx)))
    push!(rows, ("buxy (du_x/dy)", @sprintf("%dx%dx%d", BLK, nq, nq), sizeof(scr.buxy)))
    push!(rows, ("buzx (du_z/dx)", @sprintf("%dx%dx%d", BLK, nq, nq), sizeof(scr.buzx)))
    push!(rows, ("buzy (du_z/dy)", @sprintf("%dx%dx%d", BLK, nq, nq), sizeof(scr.buzy)))
    push!(rows, ("bt1  (1-D contract tmp)", @sprintf("%dx%dx%d", BLK, n, nq), sizeof(scr.bt1)))
    push!(rows, ("bF   (weighted grads)", @sprintf("%dx%dx%d", BLK, nq, nq), sizeof(scr.bF)))
    push!(rows, ("box  (leaf x out)", @sprintf("%dx%dx%d", BLK, n, n), sizeof(scr.box)))
    push!(rows, ("boz  (leaf z out)", @sprintf("%dx%dx%d", BLK, n, n), sizeof(scr.boz)))
    push!(rows, ("bshx/bshy/bswf/bone", @sprintf("4x%d", BLK), 4 * BLK * 8))
    push!(rows, ("bkofs (Int)", @sprintf("%d", BLK), BLK * 8))
    scratch_bytes = sizeof(scr.bxv) + sizeof(scr.bzv) + sizeof(scr.buxx) +
                    sizeof(scr.buxy) + sizeof(scr.buzx) + sizeof(scr.buzy) +
                    sizeof(scr.bt1) + sizeof(scr.bF) + sizeof(scr.box) +
                    sizeof(scr.boz) + 4 * BLK * 8 + BLK * 8
    eta_slice = nq2 * BLK * 8
    ids_slice = n2 * BLK * 8       # ids[:, lb:lb+BLK-1] is contiguous (column major)
    basis = sizeof(op.phi) + sizeof(op.dphi) + sizeof(op.phiT) +
            sizeof(op.dphiT) + sizeof(op.gl_w)
    # gather/scatter line footprint of one block: the distinct 64 B lines of
    # `v` (and of `out`) that the block's ids touch, per component
    seen = Set{Int}()
    gids = Int[]
    for b in 1:BLK
        li = b
        for idx in 1:n2
            g = op.ids[idx, li]
            push!(gids, g)
            push!(seen, (g - 1) >> 3)
        end
    end
    lines_per_comp = length(seen)
    nuniq = length(unique(gids))
    gather_lines = 2 * lines_per_comp * 64      # x and z halves of `v`
    scatter_lines = 2 * lines_per_comp * 64     # x and z halves of `out`
    for (name, shape, by) in rows
        @printf("WS %-26s %-12s %6d B\n", name, shape, by)
    end
    @printf("WS %-26s %-12s %6d B\n", "scratch total (one block)", "", scratch_bytes)
    @printf("WS %-26s %-12s %6d B\n", "eta lane-table slice", @sprintf("%dx%d", BLK, nq2), eta_slice)
    @printf("WS %-26s %-12s %6d B\n", "ids slice", @sprintf("%dx%d", BLK, n2), ids_slice)
    @printf("WS %-26s %-12s %6d B\n", "basis tables (phi,dphi,..)", "", basis)
    @printf("WS %-26s %-12s %6d B\n", "v gather lines (2 comps)", @sprintf("%d uniq ids", nuniq), gather_lines)
    @printf("WS %-26s %-12s %6d B\n", "out scatter lines (2 comps)", "", scatter_lines)
    live = scratch_bytes + eta_slice + ids_slice + basis
    @printf("WSSUM live_compute=%d B (%.1f%% of L1d)  +compulsory_stream=%d B  total=%d B\n",
            live, 100.0 * live / get(caches, 1, 1), gather_lines + scatter_lines,
            live + gather_lines + scatter_lines)
    @printf("WSTAB total_table=%d B (nblk=%d) per_worker_scratch=%d B nworkers_scratch=%d B\n",
            length(etav) * 8, cld(nl, BLK), scratch_bytes,
            mg.threads > 1 ? length(mg.par[kf].scr) * scratch_bytes : scratch_bytes)

    # ------------------------------------------------------------------
    # 2. the kernel per level: hot, and cold at the finest level
    # ------------------------------------------------------------------
    @printf("%-4s %6s %7s %10s %10s %10s %10s\n", "lev", "nleaf", "nf",
            "raw_us", "raw_perleaf_ns", "raw_blk0_us", "scalar_perleaf_ns")
    for k in 1:nlev
        opk = mg.mfops[k]::MS.SemQuadKOp
        etk = mg.mfetas[k]::AbstractVector{Float64}
        wk = mg.mfwork[k]::MS.SemQuadKWork
        nfk = mg.geo.nfs[k]
        x = randn(nfk); y = zeros(nfk)
        full = randn(2 * opk.N); op_out = zeros(2 * opk.N)
        MS._semquad_mg_apply!(y, x, mg, k)
        t_raw = best_ns(() -> MS._semquad_apply_K_raw!(op_out, opk, etk, full, wk))
        t_raw0 = best_ns(() -> MS._semquad_apply_K_raw!(op_out, opk, etk, full, wk;
                                                       blk=0))
        @printf("%-4d %6d %7d %10.3f %10.2f %10.3f %10.2f\n", k, opk.nleaf, nfk,
                t_raw / 1e3, t_raw / opk.nleaf, t_raw0 / 1e3, t_raw0 / opk.nleaf)
    end

    opf = mg.mfops[kf]::MS.SemQuadKOp
    etf = mg.mfetas[kf]::AbstractVector{Float64}
    wf = mg.mfwork[kf]::MS.SemQuadKWork
    fullf = randn(2 * opf.N); outf = zeros(2 * opf.N)
    t_hot = best_ns(() -> MS._semquad_apply_K_raw!(outf, opf, etf, fullf, wf), 9)
    t_cold = cold_ns(() -> MS._semquad_apply_K_raw!(outf, opf, etf, fullf, wf), 9)
    @printf("RAWCACHE lev=%d nleaf=%d hot_us=%.3f cold_us=%.3f ratio=%.3f hot_ns_leaf=%.2f cold_ns_leaf=%.2f\n",
            kf, opf.nleaf, t_hot / 1e3, t_cold / 1e3, t_cold / t_hot,
            t_hot / opf.nleaf, t_cold / opf.nleaf)
    # the gather form of the eta read (the pre-§15 kernel), same process
    MS.SEMQUAD_MG_ETA_LANES[] = false
    t_gat = best_ns(() -> MS._semquad_apply_K_raw!(outf, opf, etf, fullf, wf), 9)
    tg_cold = cold_ns(() -> MS._semquad_apply_K_raw!(outf, opf, etf, fullf, wf), 9)
    MS.SEMQUAD_MG_ETA_LANES[] = true
    @printf("RAWCACHE gather lev=%d hot_us=%.3f cold_us=%.3f ratio=%.3f\n",
            kf, t_gat / 1e3, tg_cold / 1e3, tg_cold / t_gat)

    # bandwidth calibration in the same process
    nn = 8_000_000
    aa = rand(nn); bb = rand(nn); cc = zeros(nn)
    t_copy = best_ns(() -> copyto!(cc, aa), 5; target_ns=1e7)
    tm = best_ns(() -> sum(cc), 5; target_ns=1e7)
    @printf("MEMBW copy_GBs=%.1f sum_GBs=%.1f\n", 2 * 8nn / t_copy, 8nn / tm)
    # control for the flush: a copy that fits L2 must slow down when flushed
    aa4 = rand(131_072); cc4 = zeros(131_072)   # 1 MiB each, L2-resident
    t4_hot = best_ns(() -> copyto!(cc4, aa4), 9; target_ns=1e6)
    t4_cold = cold_ns(() -> copyto!(cc4, aa4), 9)
    @printf("FLUSHCTL copy_L2_hot_us=%.2f cold_us=%.2f ratio=%.2f\n",
            t4_hot / 1e3, t4_cold / 1e3, t4_cold / t4_hot)
    t_fl = best_ns(flushcache!, 3; target_ns=2e7)
    @printf("FLUSHCTL bytes=%d us=%.1f GBs=%.1f\n",
            8 * (length(FLUSH) + length(FLUSH2)), t_fl / 1e3,
            8.0 * (length(FLUSH) + length(FLUSH2)) / t_fl)
    @printf("ROOFLINE raw_flops_per_leaf=756 raw_bytes_per_leaf=584 used_GBs=%.2f used_GFLOPs=%.2f\n",
            584.0 / (t_hot / opf.nleaf), 756.0 / (t_hot / opf.nleaf))

    if get(ENV, "BENCH_ASM", "0") != "0"
        io = IOBuffer()
        tt = (Vector{Float64}, Nothing, Nothing, MS.SemQuadKOp,
              MS.SemQuadEtaLane, Vector{Float64}, MS.SemQuadKScratch, Int, Int,
              Int, Val{2}, Val{false}, Val{true})
        InteractiveUtils.code_native(io, MS._semquad_kn_blk_body!, tt; syntax=:att,
                                     debuginfo=:none)
        s = String(take!(io))
        @printf("ASM BLK=%d lines=%d vgather=%d vaddpd=%d vmulpd=%d vsubpd=%d fmadd=%d scalar_arith=%d\n",
                BLK, count("\n", s), count("vgather", s), count("vaddpd", s),
                count("vmulpd", s), count("vsubpd", s),
                count("vfmadd", s) + count("vfnmadd", s),
                count("vaddsd", s) + count("vmulsd", s) + count("vsubsd", s))
    end
    rm(tmpdir; force=true, recursive=true)
end

main()
