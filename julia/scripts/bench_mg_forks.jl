#!/usr/bin/env julia
# =============================================================================
# bench_mg_forks.jl -- fork/join accounting and before/after of the S6
# consolidated V-cycle (docs/dev/MG.md).
#
#   BENCH_CASE=F BENCH_THR=8 julia --project=julia --threads=8 \
#       julia/scripts/bench_mg_forks.jl
#
# One process builds the production matrix-free hierarchy of the state
# (coarsen_factor=1.5, `smoothing=:auto`, `mg_threads=BENCH_THR`) and measures
# the two V-cycle paths **in the same process, alternating**:
#
#   per-call (`SEMQUAD_MG_REGION[] = false`, the historical path) -- one
#   fork/join per op, and consolidated (`true`, the S6 path) -- one fork/join
#   per level block.
#
# It prints
#
#   * forks per V-cycle (task-creation regions and worker tasks) for both paths;
#   * the per-level apply / Jacobi / subtract / transfer times and their fork
#     counts, and the decomposition of the threaded apply at the finest level
#     (`fill!` + parallel leaf phase + serial scatter + the two `Cu` products);
#   * the empty-fork cost (`_semquad_td_run!` with an empty body);
#   * the V-cycle for both paths, best and median of BENCH_REPS samples each,
#     alternating;
#   * with BENCH_SOLVE=1: the warm step, the cold solve and the saddle
#     iteration for both paths, alternating over BENCH_REPS rounds (min and
#     median), and one full solve per path compared bitwise.
#
# The checkpoint is copied to a temp dir and opened read-only; the script writes
# nothing into the repository.  BLAS is pinned to one thread.
#
# Environment:
#   BENCH_CASE  = F | P1      production checkpoint (default F)
#   BENCH_CKPT  = path        override the checkpoint path
#   BENCH_THR   = n           mg_threads (default 8); start Julia --threads=n
#   BENCH_REPS  = n           V-cycle samples and solve rounds per mode (7)
#   BENCH_SOLVE = 0 | 1       also measure step / cold solve / iteration (1)
#   BENCH_MODES = csv         measurement modes (default "percall,region";
#                             percall|region|region_tr|region_cu|region_both,
#                             the §13 A/B adds the last three)
# =============================================================================
using MantleSEM, Printf, LinearAlgebra, SparseArrays

const MS = MantleSEM
BLAS.set_num_threads(1)
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const CASE = get(ENV, "BENCH_CASE", "F")
const THR = parse(Int, get(ENV, "BENCH_THR", "8"))
const REPEATS = parse(Int, get(ENV, "BENCH_REPS", "7"))
const DOSOLVE = get(ENV, "BENCH_SOLVE", "1") != "0"
# Measurement modes: `(label, region, transfer, cu, lanes)`, where `region` is
# `SEMQUAD_MG_REGION`, `transfer` `SEMQUAD_MG_REGION_TRANSFER`, `cu`
# `SEMQUAD_MG_REGION_CU` and `lanes` `SEMQUAD_MG_ETA_LANES` (the eta lane
# pre-gather of docs/dev/MG.md).  The default is the S6 A/B of
# docs/dev/MG.md; `BENCH_MODES=region,region_tr,region_cu,region_both`
# adds the §13 phases, and a `_gather` suffix turns the lane table off for the
# §15 A/B (`region_both_gather` is the same configuration reading eta through
# the per-lane gather).
function _bench_mode(tok::AbstractString)
    lanes = true
    base = tok
    if endswith(base, "_gather")
        lanes = false
        base = base[1:(end - length("_gather"))]
    end
    base == "percall" && return (String(tok), false, false, false, lanes)
    base == "region" && return (String(tok), true, false, false, lanes)
    base == "region_tr" && return (String(tok), true, true, false, lanes)
    base == "region_cu" && return (String(tok), true, false, true, lanes)
    base == "region_both" && return (String(tok), true, true, true, lanes)
    error("unknown BENCH_MODES entry '$tok' " *
          "(percall|region|region_tr|region_cu|region_both, optional _gather)")
end
const MODES = [_bench_mode(strip(t))
               for t in split(get(ENV, "BENCH_MODES", "percall,region"), ',')]
function _setmode!(m)
    MS.SEMQUAD_MG_REGION[] = m[2]
    MS.SEMQUAD_MG_REGION_TRANSFER[] = m[3]
    MS.SEMQUAD_MG_REGION_CU[] = m[4]
    MS.SEMQUAD_MG_ETA_LANES[] = m[5]
    return nothing
end

# bit-pattern equality: `==` accepts a `-0.0` for a `+0.0`
function bitsame(a::AbstractVector{Float64}, b::AbstractVector{Float64})
    length(a) == length(b) || return false
    return reinterpret(UInt64, a) == reinterpret(UInt64, b)
end
const CKPT = get(ENV, "BENCH_CKPT",
                 CASE == "F" ?
                 joinpath(REPO, "out_julia", "_s7", "case", "pre_step6131.h5") :
                 joinpath(REPO, "out_julia", "s340", "ab_none", "resume.h5"))
const Z2 = (x, y) -> 0.0

"""The production card, rebuilt from the checkpoint's own `config` attribute."""
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

# best-of-`reps` mean-of-`inner` nanosecond timer (as bench_mg_apply.jl)
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

"""Fork count and worker-task count of one call of `f`."""
function forks_of(f)
    n0 = MS.SEMQUAD_MG_FORKS[]; k0 = MS.SEMQUAD_MG_FORK_TASKS[]
    f()
    return (MS.SEMQUAD_MG_FORKS[] - n0, MS.SEMQUAD_MG_FORK_TASKS[] - k0)
end

isfile(CKPT) || error("missing checkpoint $CKPT")
tmpdir = mktempdir()
copyto = joinpath(tmpdir, basename(CKPT))
cp(CKPT, copyto; force=true)
chmod(copyto, 0o444)
st = load_semquad_state(copyto)
qt = st.qt; p = st.p; theta = st.theta; ux = st.ux; uz = st.uz
pr = params_from_config(st.config)
ph = phases_from_config(st.config)
ss = make_semstokes(qt, p)
leaves = Set(qt.leaves)
Tfun = (x, y) -> MS._semquad_eval_at(ss.su, theta, x, y, leaves)
eta = (x, y) -> begin
    T = Tbar(y, pr.H) + Tfun(x, y)
    ed = semquad_strain_rate_at(ss.su, ux, uz, x, y, leaves)
    eta_composite_at(T, ed, y, pr)[1]
end
fz = (x, y) -> pr.Ra * Tfun(x, y)
if !isempty(ph)
    fz_thermal = fz
    fz = (x, y) -> fz_thermal(x, y) + semquad_phase_buoyancy(pr, ph, y, Tfun(x, y))
end
kw = (bc=:free, deviatoric=false, mass_form=:boussinesq, linear=:mg,
      mg_operator=:matrixfree, mg_vel_solver=:vcycle, mg_tol=1.0e-6,
      mg_threads=THR)

mg = MS.make_semquad_mg(ss, qt, p, :free, false, eta, nothing;
                        cycles=1, smoother=:jacobi,
                        coarse_operator=:rediscretize, operator=:matrixfree,
                        coarsen_factor=1.5, smoothing=:auto, mg_threads=THR)
L = mg.geo.L; nlev = L + 1
ser = MS.SEMQUAD_MG_SERIAL_DOFS[]
@printf("## case=%s thr=%d jl_threads=%d leaves=%d nlev=%d nfs=%s\n",
        CASE, THR, Threads.nthreads(), length(qt.leaves), nlev,
        join(mg.geo.nfs, ","))
@printf("## mg_threads=%d serial_dofs=%d blk=%d\n", mg.threads, ser,
        mg.mfops[1].blk)

# ---------------------------------------------------------------------------
# 1. forks per V-cycle and the V-cycle itself, one row per mode
# ---------------------------------------------------------------------------
b = randn(mg.geo.nfs[end])
fcount = Dict{String,Tuple{Int,Int}}()
vcs = Dict{String,Vector{Float64}}(m[1] => Float64[] for m in MODES)
_setmode!(MODES[1])
zref = MS.semquad_mg_apply(mg, b)          # the first mode (percall by default)
for m in MODES
    _setmode!(m)
    MS.semquad_mg_apply(mg, b)                              # warm
    MS.SEMQUAD_MG_FORKS[] = 0; MS.SEMQUAD_MG_FORK_TASKS[] = 0
    MS.SEMQUAD_MG_PHASES[] = 0
    z1 = MS.semquad_mg_apply(mg, b)
    fcount[m[1]] = (MS.SEMQUAD_MG_FORKS[], MS.SEMQUAD_MG_FORK_TASKS[])
    @printf("VCYCLE_FORKS mode=%s region=%d transfer=%d cu=%d lanes=%d forks=%d tasks=%d phases=%d\n",
            m[1], m[2], m[3], m[4], m[5], MS.SEMQUAD_MG_FORKS[],
            MS.SEMQUAD_MG_FORK_TASKS[], MS.SEMQUAD_MG_PHASES[])
    bitsame(z1, zref) || error("mode=$(m[1]) V-cycle is not bitwise the reference")
end
for _ in 1:5, m in MODES
    _setmode!(m)
    push!(vcs[m[1]], best_ns(() -> MS.semquad_mg_apply(mg, b), REPEATS))
end
med(v) = sort(v)[cld(length(v), 2)]
for m in MODES
    @printf("VCYCLE_US mode=%s min=%.1f med=%.1f\n", m[1],
            minimum(vcs[m[1]]) / 1e3, med(vcs[m[1]]) / 1e3)
end
for m in MODES[2:end]
    @printf("VCYCLE_RATIO %s/%s min=%.3f med=%.3f\n",
            MODES[1][1], m[1], minimum(vcs[MODES[1][1]]) / minimum(vcs[m[1]]),
            med(vcs[MODES[1][1]]) / med(vcs[m[1]]))
end

# empty fork/join region: the pure scheduler cost
for nt in unique([1, min(2, mg.threads), mg.threads])
    t = best_ns(() -> MS._semquad_td_run!(_ -> nothing, nt, 4 * nt), 21)
    @printf("NULL_FORK nt=%d us=%.3f\n", nt, t / 1e3)
end
if mg.threads > 1
    rr = MS.SemQuadParRegion(mg.threads)
    t = best_ns(() -> MS._semquad_region_run!(rr) do
                    nothing
                end, 21)
    @printf("NULL_REGION nt=%d us=%.3f\n", mg.threads, t / 1e3)
    # marginal cost of one phase: `_semquad_phase!` over 4*nt empty chunks,
    # measured as the slope between 21 and 1 phases inside one region
    nch = 4 * mg.threads
    t21 = best_ns(() -> MS._semquad_region_run!(rr) do
                      for _ in 1:21
                          MS._semquad_phase!(rr, (c, w) -> nothing, nch)
                      end
                  end, 21)
    t1 = best_ns(() -> MS._semquad_region_run!(rr) do
                     MS._semquad_phase!(rr, (c, w) -> nothing, nch)
                 end, 21)
    @printf("NULL_PHASE nt=%d chunks=%d us=%.3f\n", mg.threads, nch,
            (t21 - t1) / 20 / 1e3)
end
# per-V-cycle allocation of every mode (task objects + region state)
for m in MODES
    _setmode!(m)
    MS.semquad_mg_apply(mg, b)
    a = @allocated MS.semquad_mg_apply(mg, b)
    @printf("VCYCLE_ALLOC mode=%s bytes=%d\n", m[1], a)
end
_setmode!(MODES[end])

@printf("%-4s %6s %7s %6s %10s %10s %10s %8s %8s %8s %8s\n",
        "lev", "nleaf", "nf", "thr", "apply_us", "cu_us", "leaf_us",
        "scat_us", "cut_us", "sub_us", "jac_us")
for k in 1:nlev
    nf_ = mg.geo.nfs[k]
    op = mg.mfops[k]::MS.SemQuadKOp
    etav = mg.mfetas[k]::AbstractVector{Float64}
    work = mg.mfwork[k]::MS.SemQuadKWork
    Cu = mg.geo.Cus[k]
    x = randn(nf_); y = zeros(nf_)
    full = zeros(2 * op.N); op_out = zeros(2 * op.N)
    threaded = mg.threads > 1 && nf_ > ser
    MS._semquad_mg_apply!(y, x, mg, k)
    t_apply = best_ns(() -> MS._semquad_mg_apply!(y, x, mg, k))
    t_cu = best_ns(() -> mul!(full, Cu, x))
    t_cut = best_ns(() -> mul!(y, transpose(Cu), op_out))
    t_leaf = NaN; t_scat = NaN
    if threaded
        par = mg.par[k]::MS.SemQuadKPar
        t_raw = best_ns(() -> MS._semquad_apply_K_raw_threaded!(op_out, op, etav,
                                                                full, par))
        t_fill = best_ns(() -> fill!(op_out, 0.0))
        MS._semquad_apply_K_raw_threaded!(op_out, op, etav, full, par)
        t_scat = best_ns(() -> MS._semquad_scatter_elts!(op_out, op, par))
        t_leaf = t_raw - t_fill - t_scat
        @printf("%-4d %6d %7d %6s %10.1f %10.1f %10.1f %8.1f %8.1f",
                k, op.nleaf, nf_, "yes", t_apply / 1e3, t_cu / 1e3,
                t_leaf / 1e3, t_scat / 1e3, t_cut / 1e3)
    else
        @printf("%-4d %6d %7d %6s %10.1f %10.1f %10s %8s %8.1f",
                k, op.nleaf, nf_, "-", t_apply / 1e3, t_cu / 1e3, "-", "-",
                t_cut / 1e3)
    end
    n = nf_
    r = randn(n); xx = randn(n); yy = randn(n); D = mg.D[k]
    MS._semquad_td_sub!(r, xx, yy, mg.threads)
    fsub, _ = forks_of(() -> MS._semquad_td_sub!(r, xx, yy, mg.threads))
    t_sub = best_ns(() -> MS._semquad_td_sub!(r, xx, yy, mg.threads), 21)
    fjac, _ = forks_of(() -> MS._semquad_td_jacobi!(xx, r, D, 0.666, mg.threads))
    t_jac = best_ns(() -> MS._semquad_td_jacobi!(xx, r, D, 0.666, mg.threads), 21)
    @printf(" %8.2f(%d) %8.2f(%d)\n", t_sub / 1e3, fsub, t_jac / 1e3, fjac)
end
# ---------------------------------------------------------------------------
# per-level apply and leaf kernel, every mode alternating (docs/dev/MG.md
# §15): the eta-lane A/B at the finest level *and* at level 5, both layout
# settings interleaved in the same process, with the bitwise check per level
# ---------------------------------------------------------------------------
for k in 1:nlev
    nf_ = mg.geo.nfs[k]
    op = mg.mfops[k]::MS.SemQuadKOp
    etav = mg.mfetas[k]::AbstractVector{Float64}
    work = mg.mfwork[k]::MS.SemQuadKWork
    x = randn(nf_); y = zeros(nf_)
    full = zeros(2 * op.N); op_out = zeros(2 * op.N)
    mul!(full, mg.geo.Cus[k], x)
    t_ap = Dict{String,Vector{Float64}}(m[1] => Float64[] for m in MODES)
    t_raw = Dict{String,Vector{Float64}}(m[1] => Float64[] for m in MODES)
    t_leaf = Dict{String,Vector{Float64}}(m[1] => Float64[] for m in MODES)
    threaded = mg.threads > 1 && nf_ > ser
    par = threaded ? (mg.par[k]::MS.SemQuadKPar) : nothing
    t_fill = threaded ? best_ns(() -> fill!(op_out, 0.0)) : 0.0
    t_scat = threaded ? best_ns(() -> MS._semquad_scatter_elts!(op_out, op, par)) : 0.0
    ys = Dict{String,Vector{Float64}}()
    for _ in 1:5, m in MODES
        _setmode!(m)
        push!(t_ap[m[1]], best_ns(() -> MS._semquad_mg_apply!(y, x, mg, k), 7))
        push!(t_raw[m[1]],
              best_ns(() -> MS._semquad_apply_K_raw!(op_out, op, etav, full,
                                                     work), 7))
        threaded || continue
        push!(t_leaf[m[1]],
              best_ns(() -> MS._semquad_apply_K_raw_threaded!(op_out, op, etav,
                                                              full, par), 7) -
              t_fill - t_scat)
    end
    for m in MODES
        _setmode!(m)
        yy = zeros(nf_)
        MS._semquad_mg_apply!(yy, x, mg, k)
        ys[m[1]] = yy
    end
    eq = all(bitsame(ys[m[1]], ys[MODES[1][1]]) for m in MODES)
    for m in MODES
        @printf("APPLY_LEVEL lev=%d nleaf=%d nf=%d mode=%s apply_us=%.2f raw_us=%.2f apply_ratio=%.1f%% raw_ratio=%.1f%% equal=%s\n",
                k, op.nleaf, nf_, m[1], minimum(t_ap[m[1]]) / 1e3,
                minimum(t_raw[m[1]]) / 1e3,
                100 * (minimum(t_ap[m[1]]) / minimum(t_ap[MODES[1][1]]) - 1),
                100 * (minimum(t_raw[m[1]]) / minimum(t_raw[MODES[1][1]]) - 1),
                eq)
        threaded || continue
        @printf("APPLY_LEVEL_LEAF lev=%d nleaf=%d nf=%d mode=%s leaf_us=%.2f leaf_ratio=%.1f%%\n",
                k, op.nleaf, nf_, m[1], minimum(t_leaf[m[1]]) / 1e3,
                100 * (minimum(t_leaf[m[1]]) / minimum(t_leaf[MODES[1][1]]) - 1))
    end
end
_setmode!(MODES[end])
# one apply inside an open region per threaded level, with and without the
# split `Cu` products: the per-phase cost (§13)
if mg.threads > 1
    let
    oldtr = MS.SEMQUAD_MG_REGION_TRANSFER[]
    oldcu = MS.SEMQUAD_MG_REGION_CU[]
    for k in 1:nlev
        mg.geo.nfs[k] > ser || continue
        nf_ = mg.geo.nfs[k]
        par = mg.par[k]::MS.SemQuadKPar
        r = MS.SemQuadParRegion(par.nt)
        x = randn(nf_); y = zeros(nf_)
        MS.SEMQUAD_MG_REGION_CU[] = false
        MS._semquad_mg_apply!(y, x, mg, k)
        t_reg = best_ns(() -> MS._semquad_region_run!(r) do
                            MS._semquad_mg_apply!(y, x, mg, k, r)
                        end)
        MS.SEMQUAD_MG_REGION_CU[] = true
        y2 = zeros(nf_)
        t_cu = best_ns(() -> MS._semquad_region_run!(r) do
                           MS._semquad_mg_apply!(y2, x, mg, k, r)
                       end)
        @printf("APPLY_REGION lev=%d us=%.1f cu_us=%.1f equal=%s (per-call apply above)\n",
                k, t_reg / 1e3, t_cu / 1e3, y2 == y)
    end
    # finest-level phase accounting: the two serial products, the same products
    # as region phases, and the restriction with/without the transfer phase
    k = nlev
    nf_ = mg.geo.nfs[k]
    op = mg.mfops[k]::MS.SemQuadKOp
    etav = mg.mfetas[k]::AbstractVector{Float64}
    work = mg.mfwork[k]::MS.SemQuadKWork
    par = mg.par[k]::MS.SemQuadKPar
    r = MS.SemQuadParRegion(par.nt)
    x = randn(nf_); y = zeros(nf_)
    work.full .= 0.0; work.op_out .= 0.0
    t_cu1 = best_ns(() -> mul!(work.full, mg.geo.Cus[k], x))
    t_cu2 = best_ns(() -> mul!(y, transpose(mg.geo.Cus[k]), work.op_out))
    MS.SEMQUAD_MG_REGION_CU[] = true
    t_cu1r = best_ns(() -> MS._semquad_region_run!(r) do
                         MS._semquad_spmv_region!(work.full, mg.geo.Cuts[k], x,
                                                  par.nt, r)
                     end)
    t_cu2r = best_ns(() -> MS._semquad_region_run!(r) do
                         MS._semquad_spmv_at_region!(y, mg.geo.Cus[k],
                                                     work.op_out, par.nt, r)
                     end)
    rc = zeros(mg.geo.nfs[k - 1]); rv = randn(nf_)
    MS.SEMQUAD_MG_REGION_TRANSFER[] = false
    MS._semquad_mg_restrict!(rc, mg.geo.Ps[k - 1], rv, mg.threads)
    t_restr_ser = best_ns(() -> MS._semquad_mg_restrict!(rc,
                                                         mg.geo.Ps[k - 1], rv,
                                                         mg.threads))
    t_restr_off = best_ns(() -> MS._semquad_region_run!(r) do
                              MS._semquad_mg_restrict!(rc, mg.geo.Ps[k - 1], rv,
                                                       mg.threads, r)
                          end)
    MS.SEMQUAD_MG_REGION_TRANSFER[] = true
    t_restr_on = best_ns(() -> MS._semquad_region_run!(r) do
                             MS._semquad_mg_restrict!(rc, mg.geo.Ps[k - 1], rv,
                                                      mg.threads, r)
                         end)
    @printf("FINEST_PHASES lev=%d cu1_us=%.1f cu1_region_us=%.1f cu2_us=%.1f cu2_region_us=%.1f restrict_serial_us=%.1f restrict_region_off_us=%.1f restrict_region_on_us=%.1f\n",
            k, t_cu1 / 1e3, t_cu1r / 1e3, t_cu2 / 1e3, t_cu2r / 1e3,
            t_restr_ser / 1e3, t_restr_off / 1e3, t_restr_on / 1e3)
    # the §13 restriction: the finest threaded level restricts into the next
    # coarser level, whose dof count is below the threshold.  That is the call
    # the S6 path left serial inside the region.
    kt = 0
    for t in 2:nlev
        if mg.geo.nfs[t] > ser
            kt = t
            break
        end
    end
    if kt != 0
        par = mg.par[kt]::MS.SemQuadKPar
        r2 = MS.SemQuadParRegion(par.nt)
        rc2 = zeros(mg.geo.nfs[kt - 1]); rv2 = randn(mg.geo.nfs[kt])
        MS.SEMQUAD_MG_REGION_TRANSFER[] = false
        MS._semquad_mg_restrict!(rc2, mg.geo.Ps[kt - 1], rv2, mg.threads)
        t_s = best_ns(() -> MS._semquad_mg_restrict!(rc2, mg.geo.Ps[kt - 1],
                                                     rv2, mg.threads))
        t_off = best_ns(() -> MS._semquad_region_run!(r2) do
                            MS._semquad_mg_restrict!(rc2, mg.geo.Ps[kt - 1],
                                                     rv2, mg.threads, r2)
                        end)
        MS.SEMQUAD_MG_REGION_TRANSFER[] = true
        t_on = best_ns(() -> MS._semquad_region_run!(r2) do
                           MS._semquad_mg_restrict!(rc2, mg.geo.Ps[kt - 1],
                                                    rv2, mg.threads, r2)
                       end)
        @printf("RESTRICT_PHASE lev=%d nout=%d serial_us=%.1f region_off_us=%.1f region_on_us=%.1f\n",
                kt, mg.geo.nfs[kt - 1], t_s / 1e3, t_off / 1e3, t_on / 1e3)
    end
    MS.SEMQUAD_MG_REGION_TRANSFER[] = oldtr
    MS.SEMQUAD_MG_REGION_CU[] = oldcu
    end
end
# transfers at the finest level
k = nlev
r = randn(mg.geo.nfs[k]); rc = zeros(mg.geo.nfs[k - 1])
pyp = zeros(mg.geo.nfs[k]); ec = randn(mg.geo.nfs[k - 1])
if isempty(mg.Pts)
    @printf("TRANSFER_FORKS restrict=0 prolong=0 (lev=%d nf=%d/%d serial path)\n",
            k, mg.geo.nfs[k - 1], mg.geo.nfs[k])
else
    fr, _ = forks_of(() -> MS._semquad_mg_restrict!(rc, mg.geo.Ps[k - 1], r,
                                                    mg.threads))
    fp, _ = forks_of(() -> MS._semquad_mg_prolong!(pyp, mg.Pts[k - 1], ec,
                                                   mg.threads))
    @printf("TRANSFER_FORKS restrict=%d prolong=%d (lev=%d nf=%d/%d)\n", fr, fp, k,
            mg.geo.nfs[k - 1], mg.geo.nfs[k])
end

# ---------------------------------------------------------------------------
# in-place per-op breakdown of one V-cycle (docs/dev/MG.md)
#
# A replica of `_semquad_mg_vcycle!` / `_semquad_mg_vcycle_region!` that times
# every op in place, with the same region boundaries as the production path
# (one region around the pre-smoothing block + restriction, one around the
# prolongation + `x .+= py` + post-smoothing block, recursion in between).
# `bits` checks the replica against `semquad_mg_apply` for the path measured.
# ---------------------------------------------------------------------------
const BKEYS = (:fill, :apply, :sub, :jac, :restrict, :prolong, :axpy, :dense)
newacc() = Dict{Symbol,Float64}(k => 0.0 for k in BKEYS)

function vc_timed!(x::Vector{Float64}, b::Vector{Float64}, mg, lev::Int,
                   reg, acc)
    if lev == 1
        acc[:dense] += @elapsed x .= mg.cA \ b
        return nothing
    end
    acc[:fill] += @elapsed fill!(x, 0.0)
    if MS._semquad_mg_threaded_level(mg, lev)
        r = MS.SemQuadParRegion((mg.par[lev]::MS.SemQuadKPar).nt)
        MS._semquad_region_run!(r) do
            pre_timed!(x, b, mg, lev, r, acc)
            acc[:restrict] += @elapsed MS._semquad_mg_restrict!(
                mg.rcs[lev], mg.geo.Ps[lev - 1], mg.sr[lev], mg.threads, r)
        end
        vc_timed!(mg.ecs[lev], mg.rcs[lev], mg, lev - 1, reg, acc)
        MS._semquad_region_run!(r) do
            py = mg.pys[lev]
            acc[:prolong] += @elapsed MS._semquad_mg_prolong!(
                py, mg.Pts[lev - 1]::SparseMatrixCSC{Float64,Int}, mg.ecs[lev],
                mg.threads, r)
            acc[:axpy] += @elapsed x .+= py
            post_timed!(x, b, mg, lev, r, acc)
        end
        return nothing
    end
    pre_timed!(x, b, mg, lev, reg, acc)
    acc[:restrict] += @elapsed MS._semquad_mg_restrict!(
        mg.rcs[lev], mg.geo.Ps[lev - 1], mg.sr[lev], mg.threads, reg)
    vc_timed!(mg.ecs[lev], mg.rcs[lev], mg, lev - 1, reg, acc)
    if isempty(mg.Pts)
        acc[:prolong] += @elapsed mul!(mg.pys[lev], mg.geo.Ps[lev - 1],
                                       mg.ecs[lev])
    else
        acc[:prolong] += @elapsed MS._semquad_mg_prolong!(
            mg.pys[lev], mg.Pts[lev - 1]::SparseMatrixCSC{Float64,Int},
            mg.ecs[lev], mg.threads, reg)
    end
    acc[:axpy] += @elapsed x .+= mg.pys[lev]
    post_timed!(x, b, mg, lev, reg, acc)
    return nothing
end

function pre_timed!(x, b, mg, lev, reg, acc)
    for _ in 1:mg.pre[lev]
        acc[:apply] += @elapsed MS._semquad_mg_apply!(mg.sy[lev], x, mg, lev,
                                                      reg)
        acc[:sub] += @elapsed MS._semquad_td_sub!(mg.sr[lev], b, mg.sy[lev],
                                                  mg.threads, reg)
        acc[:jac] += @elapsed MS._semquad_td_jacobi!(x, mg.sr[lev], mg.D[lev],
                                                     mg.omega, mg.threads, reg)
    end
    # the residual that feeds the restriction (no Jacobi update after it)
    acc[:apply] += @elapsed MS._semquad_mg_apply!(mg.sy[lev], x, mg, lev, reg)
    acc[:sub] += @elapsed MS._semquad_td_sub!(mg.sr[lev], b, mg.sy[lev],
                                              mg.threads, reg)
    return nothing
end

function post_timed!(x, b, mg, lev, reg, acc)
    for _ in 1:mg.post[lev]
        acc[:apply] += @elapsed MS._semquad_mg_apply!(mg.sy[lev], x, mg, lev,
                                                      reg)
        acc[:sub] += @elapsed MS._semquad_td_sub!(mg.sr[lev], b, mg.sy[lev],
                                                  mg.threads, reg)
        acc[:jac] += @elapsed MS._semquad_td_jacobi!(x, mg.sr[lev], mg.D[lev],
                                                     mg.omega, mg.threads, reg)
    end
    return nothing
end

zrep = zeros(Float64, mg.geo.nfs[end])
for m in MODES
    _setmode!(m)
    vc_timed!(zrep, b, mg, nlev, nothing, newacc())              # warm
    best = Dict{Symbol,Float64}(k => Inf for k in BKEYS)
    tots = Float64[]
    for _ in 1:REPEATS
        acc = newacc()
        push!(tots, @elapsed vc_timed!(zrep, b, mg, nlev, nothing, acc))
        for k in BKEYS
            best[k] = min(best[k], acc[k])
        end
    end
    s = sum(best[k] for k in BKEYS)
    @printf("BREAKDOWN mode=%s total_us=%.0f parts_us=%.0f equal=%s",
            m[1], 1e6 * minimum(tots), 1e6 * s, bitsame(zrep, zref))
    for k in BKEYS
        @printf(" %s=%.0f", k, 1e6 * best[k])
    end
    @printf("\n")
end
_setmode!(MODES[end])

if DOSOLVE
    solve() = solve_semquad_stokes_var(ss, qt, eta, Z2, fz, Z2, Z2; kw...)
    solve()                                                 # JIT warm
    sys = solve_semquad_stokes_var(ss, qt, eta, Z2, fz, Z2, Z2; bc=:free,
                                   linear=:direct, assemble_only=true)
    op = semquad_kop(ss, qt; deviatoric=false)
    # the lane table of the finest level: this local closure is the §16 outer
    # operator, so the eta-lane A/B reaches the FGMRES residual/BFBT applies too
    ev = MS.semquad_eta_lanes(op, semquad_eta_field(op, eta))
    wk = SemQuadKWork(op.n, op.nq, 2 * op.N)
    Cu = SparseMatrixCSC{Float64,Int}(sparse(sys.Cu))
    mfapplyK = (out, v) -> MS._semquad_apply_Kred!(out, op, Cu, ev, v, wk)
    function coldsolve()
        empty!(MS.SEMQUAD_MG_CACHE); MS.SEMQUAD_PRESSURE_CACHE[] = nothing
        GC.gc()
        @elapsed solve()
    end
    # alternating cold solves and warm steps, REPEATS rounds per mode
    t_cold = Dict{String,Vector{Float64}}(m[1] => Float64[] for m in MODES)
    t_warm = Dict{String,Vector{Float64}}(m[1] => Float64[] for m in MODES)
    for _ in 1:REPEATS, m in MODES
        _setmode!(m)
        push!(t_cold[m[1]], coldsolve())
        GC.gc()
        push!(t_warm[m[1]], @elapsed solve())
    end
    nit = LAST_SEMSTOKES_SCHUR_ITERS[]
    # bitwise: one solve per mode, same caches as the timed ones
    us = Vector{Float64}[]; ps = Vector{Float64}[]
    nits = Int[]; ress = Float64[]
    for m in MODES
        _setmode!(m)
        u, pp, _, _, _ = solve_semquad_stokes_var(ss, qt, eta, Z2, fz, Z2, Z2;
                                                  kw...)
        push!(us, u); push!(ps, pp)
        push!(nits, LAST_SEMSTOKES_SCHUR_ITERS[])
        push!(ress, LAST_SEMSTOKES_RESIDUAL[])
    end
    bitwise_sol = all(bitsame(us[k], us[1]) && bitsame(ps[k], ps[1])
                      for k in 2:length(MODES))
    bitwise_scalar = all(nits[k] == nits[1] &&
                         reinterpret(UInt64, ress[k]) ==
                         reinterpret(UInt64, ress[1])
                         for k in 2:length(MODES))
    # iteration alone: rebuild the hierarchy + block, then time the saddle solve
    t_it = Dict{String,Vector{Float64}}(m[1] => Float64[] for m in MODES)
    nfork = Dict{String,Tuple{Int,Int}}()
    for m in MODES
        empty!(MS.SEMQUAD_MG_CACHE); MS.SEMQUAD_PRESSURE_CACHE[] = nothing
        mg2 = MS.make_semquad_mg(ss, qt, p, :free, false, eta, nothing;
                                 cycles=1, smoother=:jacobi,
                                 coarse_operator=:rediscretize,
                                 operator=:matrixfree, coarsen_factor=1.5,
                                 smoothing=:auto, mg_threads=THR, fine_Cu=sys.Cu)
        Fp, pscale = MS.semquad_mg_pressure_precond(ss, sys.Cu, sys.Cp, sys.Bs,
                                                    sys.Bt, nothing, eta;
                                                    mode=:auto, cache=false,
                                                    applyK=mfapplyK)
        it() = @elapsed MS.semquad_mg_saddle_solve(nothing, sys.Bs, sys.Bt, Fp,
                                                   pscale, vcat(sys.fr, sys.gr),
                                                   mg2; tol=1.0e-6, maxit=500,
                                                   restart=100,
                                                   vel_solver=:vcycle,
                                                   applyK=mfapplyK)
        _setmode!(m)
        it()
        for _ in 1:REPEATS
            GC.gc()
            push!(t_it[m[1]], it())
        end
        nfork[m[1]] = forks_of(it)
    end
    _setmode!(MODES[end])
    med(v) = sort(v)[cld(length(v), 2)]
    for (i, m) in enumerate(MODES)
        @printf("SOLVE case=%s thr=%d mode=%s nit=%d relres=%.17e cold_min=%.4f cold_med=%.4f step_min=%.4f step_med=%.4f iter_min=%.4f iter_med=%.4f per_iter_ms=%.2f itforks=%d bitwise=%s scalars=%s\n",
                CASE, THR, m[1], nits[i], ress[i],
                minimum(t_cold[m[1]]), med(t_cold[m[1]]),
                minimum(t_warm[m[1]]), med(t_warm[m[1]]),
                minimum(t_it[m[1]]), med(t_it[m[1]]),
                1e3 * med(t_it[m[1]]) / nit, nfork[m[1]][1] ÷ nit,
                bitwise_sol, bitwise_scalar)
    end
end

# ---------------------------------------------------------------------------
# instruction mix of the two eta indexings (docs/dev/MG.md): the same
# `_semquad_kn_blk_body!` compiled for the lane table (unit stride, `Val{true}`)
# and for the leaf-major gather (`Val{false}`), p=2, serial (STORE=false) and
# element-buffer (STORE=true) forms
# ---------------------------------------------------------------------------
if get(ENV, "BENCH_ASM", "0") != "0"
    using InteractiveUtils
    function _asm_counts(etatype, ::Val{STORE}, ::Val{LANE}) where {STORE,LANE}
        io = IOBuffer()
        tt = (Vector{Float64}, Nothing, Nothing, MS.SemQuadKOp, etatype,
              Vector{Float64}, MS.SemQuadKScratch, Int, Int, Int, Val{2},
              Val{STORE}, Val{LANE})
        code_native(io, MS._semquad_kn_blk_body!, tt; syntax=:att,
                    debuginfo=:none)
        s = String(take!(io))
        return (lines=count("\n", s), vgather=count("vgather", s),
                vaddpd=count("vaddpd", s), vmulpd=count("vmulpd", s),
                vmovupd=count("vmovupd", s), vmovsd=count("vmovsd", s))
    end
    for (store, name, et, lane) in ((false, "gather", Vector{Float64}, false),
                                    (true, "gather", Vector{Float64}, false),
                                    (false, "lane", MS.SemQuadEtaLane, true),
                                    (true, "lane", MS.SemQuadEtaLane, true))
        c = _asm_counts(et, Val(store), Val(lane))
        @printf("ASM eta=%s store=%s lines=%d vgather=%d vaddpd=%d vmulpd=%d vmovupd=%d vmovsd=%d\n",
                name, store, c.lines, c.vgather, c.vaddpd, c.vmulpd, c.vmovupd,
                c.vmovsd)
    end
end
rm(tmpdir; force=true, recursive=true)
