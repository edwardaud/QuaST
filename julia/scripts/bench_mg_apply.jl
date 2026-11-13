#!/usr/bin/env julia
# =============================================================================
# bench_mg_apply.jl -- stage-A instrument of the matrix-free velocity apply.
#
#   STATE=F  julia --project=julia --threads=8 julia/scripts/bench_mg_apply.jl
#   STATE=P1 julia --project=julia --threads=1 julia/scripts/bench_mg_apply.jl
#
# One process builds the production MG hierarchy for the state (coarsen 1.5,
# `:auto` smoothing, `mg_threads = min(Threads.nthreads(), 8)`), then prints,
# per level: the constrained apply `Cu' K (Cu v)` in us, its `Cu*v` / raw
# kernel / `Cu'*` split, the historical scalar kernel (`blk=0`), per-leaf and
# per-dof times, and allocations per apply.  It ends with the V-cycle time, the
# threaded/serial vector-op overhead on the small levels, and a memory-bandwidth
# calibration for the roofline of docs/MG.md SS11.1.
#
# The checkpoint is copied to a temp dir and opened read-only.  BLAS is pinned
# to one thread.  The script writes nothing into the repository.
# =============================================================================
using MantleSEM, Printf, LinearAlgebra, SparseArrays, Random

const MS = MantleSEM
BLAS.set_num_threads(1)
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const STATE = get(ENV, "STATE", "F")
const CKPT = STATE == "P1" ? joinpath(REPO, "out_julia", "s340", "ab_none", "resume.h5") :
                            joinpath(REPO, "out_julia", "_s7", "case", "pre_step6131.h5")
const Z2 = (x, y) -> 0.0

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
    return best, inner
end

function counted_allocs(f)
    f()
    return @allocated f()
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

    # production default hierarchy (coarsen_factor=1.5, :auto smoothing)
    mg = MS.make_semquad_mg(ss, qt, p, :free, false, eta_fun, nothing;
                            cycles=1, smoother=:jacobi,
                            coarse_operator=:rediscretize,
                            operator=:matrixfree, coarsen_factor=1.5,
                            smoothing=:auto,
                            mg_threads=min(nt, 8))
    L = mg.geo.L; nlev = L + 1
    thr_dbg = isdefined(MS, :SEMQUAD_MG_SERIAL_DOFS) ? MS.SEMQUAD_MG_SERIAL_DOFS[] : -1
    @printf("levels=%d nfs=%s nleaf=%s threads=%d blk=%d serial_dofs=%d\n", nlev,
            join(mg.geo.nfs, ","),
            join([length(mg.geo.qts[k].leaves) for k in 1:nlev], ","),
            mg.threads, mg.mfops[1].blk, thr_dbg)
    @printf("%-4s %6s %7s %10s %10s %10s %10s %10s %10s %12s\n",
            "lev", "nleaf", "nf", "apply_us", "cu_us", "raw_us", "cut_us",
            "raw_blk0", "perleaf_ns", "ns_per_dof")

    for k in 1:nlev
        nf = mg.geo.nfs[k]
        op = mg.mfops[k]::MS.SemQuadKOp
        etav = mg.mfetas[k]::AbstractVector{Float64}
        work = mg.mfwork[k]::MS.SemQuadKWork
        nleaf = op.nleaf
        x = randn(nf); y = zeros(nf)
        full = zeros(2 * op.N); op_out = zeros(2 * op.N)
        Cu = mg.geo.Cus[k]
        # make sure all buffers are touched
        MS._semquad_mg_apply!(y, x, mg, k)

        t_apply = best_ns(() -> MS._semquad_mg_apply!(y, x, mg, k))[1]
        t_cu = best_ns(() -> mul!(full, Cu, x))[1]
        t_cut = best_ns(() -> mul!(y, transpose(Cu), op_out))[1]
        t_raw = if mg.threads > 1
            par = mg.par[k]::MS.SemQuadKPar
            best_ns(() -> MS._semquad_apply_K_raw_threaded!(op_out, op, etav,
                                                            full, par))[1]
        else
            best_ns(() -> MS._semquad_apply_K_raw!(op_out, op, etav, full,
                                                   work))[1]
        end
        t_raw0 = best_ns(() -> MS._semquad_apply_K_raw!(op_out, op, etav,
                                                        full, work; blk=0))[1]
        # per-call allocation averaged over many calls (the macro above can
        # include the first-call box of the closure)
        alloc_loop = function (f, reps)
            f()
            a = @allocated for _ in 1:reps
                f()
            end
            return a ÷ reps
        end
        a_apply = alloc_loop(() -> MS._semquad_mg_apply!(y, x, mg, k), 50)
        a_raw = if mg.threads > 1
            par = mg.par[k]::MS.SemQuadKPar
            alloc_loop(() -> MS._semquad_apply_K_raw_threaded!(op_out, op,
                                                               etav, full, par), 50)
        else
            alloc_loop(() -> MS._semquad_apply_K_raw!(op_out, op, etav,
                                                      full, work), 50)
        end
        @printf("%-4d %6d %7d %10.3f %10.3f %10.3f %10.3f %10.3f %10.2f %12.3f  alloc=%d/%d\n",
                k, nleaf, nf, t_apply / 1e3, t_cu / 1e3, t_raw / 1e3,
                t_cut / 1e3, t_raw0 / 1e3, t_apply / nleaf, t_apply / nf,
                a_apply, a_raw)
        if k == nlev && mg.threads == 1
            for B in (0, 4, 8, 16)
                tb = best_ns(() -> MS._semquad_apply_K_raw!(op_out, op, etav,
                                                             full, work; blk=B))[1]
                @printf("  blk=%d raw_us=%.3f\n", B, tb / 1e3)
            end
        end
    end

    b = randn(mg.geo.nfs[end])
    z = zeros(mg.geo.nfs[end])
    t_vc = best_ns(() -> MS.semquad_mg_apply(mg, b))[1]
    @printf("vcycle_us=%.3f alloc_vcycle=%d\n", t_vc / 1e3,
            counted_allocs(() -> MS.semquad_mg_apply(mg, b)))

    # vector-op / task overhead on the small levels
    for k in (2, 3, 4, min(5, nlev))
        k <= nlev || continue
        nf = mg.geo.nfs[k]
        r = randn(nf); xx = randn(nf); yy = randn(nf); D = mg.D[k]
        t1 = best_ns(() -> MS._semquad_td_sub!(r, xx, yy, 1))[1]
        t8 = best_ns(() -> MS._semquad_td_sub!(r, xx, yy, min(8, nt)))[1]
        t1j = best_ns(() -> MS._semquad_td_jacobi!(xx, r, D, 0.666, 1))[1]
        t8j = best_ns(() -> MS._semquad_td_jacobi!(xx, r, D, 0.666, min(8, nt)))[1]
        @printf("vecops lev=%d nf=%d sub_us=%.3f/%.3f jac_us=%.3f/%.3f nt=%d\n",
                k, nf, t1 / 1e3, t8 / 1e3, t1j / 1e3, t8j / 1e3, min(8, nt))
    end

    # bandwidth / compute calibration in the same process
    n = 8_000_000
    aa = rand(n); bb = rand(n); cc = zeros(n)
    t_copy = best_ns(() -> copyto!(cc, aa), 5; target_ns=1e7)[1]
    triad! = () -> begin
        @inbounds for i in 1:n
            cc[i] = aa[i] + 2.0 * bb[i]
        end
    end
    t_triad = best_ns(triad!, 5; target_ns=1e7)[1]
    t_sum = best_ns(() -> sum(cc), 5; target_ns=1e7)[1]
    @printf("membw copy_GBs=%.1f triad_GBs=%.1f sum_GBs=%.1f\n",
            2 * 8n / t_copy, 3 * 8n / t_triad, 8n / t_sum)
end

main()
