#!/usr/bin/env julia
# =============================================================================
# bench_s1_solve.jl -- the cost of the two chol(S1) triangular solves in the
# w-BFBT pressure block, broken into parts, and the structure of `F1`.
#
#   julia --project=. --threads=1 benchmarks/bench_s1_solve.jl F
#   julia --project=. --threads=8 benchmarks/bench_s1_solve.jl F
#
# Measures, at the process's Julia thread count, on a production checkpoint:
#
#   (1) the factorisation SETUP, split into
#         D build (the BFBT weight) / S1 = Bs D^-1 Bt / symmetrise+shift /
#         cholesky(Symmetric(S1)),  each median of `REPS`;
#   (2) one forward/backward solve `F1 \ b` in three cache states -- hot, the
#       shipped allocating form, and flushed (96 MB) -- plus the allocation the
#       shipped form pays, and the `ldiv!` form (bitwise equal, verified);
#   (3) the multi-RHS solve `F1 \ B` for k = 4, 8, 16, 32 columns, fitted as
#       `t(k) = a + b k`: `b` is the arithmetic cost of one RHS and `a` the
#       fixed cost of touching L (indices + values + pointer chasing);
#   (4) the same matrix with other factorisation kernels -- forced supernodal,
#       relaxed amalgamation (`nrelax`) and a nested-dissection ordering -- each
#       a factor of the *same* matrix (same preconditioner up to round-off);
#   (5) the structure of F1: n, nnz(S1), nnz(L), the elimination-tree depth, the
#       forward/backward level counts and width histograms, the critical path of
#       each direction (the level-scheduling ceiling with unlimited threads), the
#       share of the work in wide levels, and the `SemQuadParRegion` barrier.
#
# The checkpoint is copied to a temporary directory and opened read-only;
# nothing is written into the repository.
#
# Environment:
#   S1C_REPS    build repeats (default 3)
#   S1C_BATCH   solves per timed batch (default 100)
#   S1C_THR     mg_threads (default = Julia threads)
# =============================================================================

using QuaST, Printf, LinearAlgebra, SparseArrays
const MS = QuaST

const REPO = normpath(joinpath(@__DIR__, ".."))
const CASE_DIRS = Dict(
    "F" => joinpath(REPO, "out_julia", "_s7", "case", "pre_step6131.h5"),
    "P1" => joinpath(REPO, "out_julia", "s340", "ab_none", "resume.h5"))
const CASE = isempty(ARGS) ? "F" : uppercase(ARGS[1])
const CKPT = get(CASE_DIRS, CASE, "")
const REPS = parse(Int, get(ENV, "S1C_REPS", "3"))
const BATCH = parse(Int, get(ENV, "S1C_BATCH", "100"))
const THR = parse(Int, get(ENV, "S1C_THR", string(Threads.nthreads())))
const MGTHR = MS._semquad_mg_thread_count(THR)
const Z2 = (x, y) -> 0.0

BLAS.set_num_threads(1)

med(v) = (s = sort(copy(v)); s[cld(length(s), 2)])
mn(v) = minimum(v)
tmin(f, nb, n) = (f(); mn([@elapsed(for _ in 1:n; f(); end) / n for _ in 1:nb]))

# ---------------------------------------------------------------------------
# CHOLMOD knobs (the factory functions are the only portable way to reach the
# `Common` fields: `cholesky` reads them at `analyze`/`factorize` time)
# ---------------------------------------------------------------------------
const CH = SparseArrays.CHOLMOD
chol_is_super(F) = Int(unsafe_load(pointer(F)).is_super)
"""`cholesky(Symmetric(A))` with CHOLMOD's `supernodal` Common field forced."""
function chol_super(A, mode::Int)
    return CH.@cholmod_param supernodal = mode begin
        cholesky(Symmetric(A))
    end
end

"""
`cholesky(Symmetric(A))` with CHOLMOD's ordering search forced to `nmethods`.

Julia's `CHOLMOD.symbolic` pins `nmethods` so that only natural + AMD are tried
(`SparseArrays/src/solvers/cholmod.jl`: "the default is to just use AMD"), so the
ordering search has to go through `analyze` directly.
"""
function chol_ord(A, nmethods::Int)
    S = CH.Sparse(A)
    F = CH.@cholmod_param nmethods = nmethods begin
        CH.analyze(S)
    end
    CH.cholesky!(F, S)
    return F
end

# ---------------------------------------------------------------------------
# symbolic structure of a CSC lower factor
# ---------------------------------------------------------------------------
"""
Row-dependency level scheduling of the forward solve `L y = b` (L lower, CSC):
row `i` reads `y[j]` for every stored `(i, j)`, `j < i`, so
`lev[i] = 1 + max(lev[j])` over those `j`.  Columns are scanned in increasing
order, a topological order of that DAG, so one pass suffices.
Returns `(lev, rowcnt)` with `rowcnt[i]` the strict-lower nonzeros of row `i`.
"""
function row_levels(L::SparseMatrixCSC, n::Int)
    rv = L.rowval; cp = L.colptr
    lev = zeros(Int, n); rowcnt = zeros(Int, n)
    @inbounds for j in 1:n
        lj = lev[j]
        for p in cp[j]:(cp[j + 1] - 1)
            i = rv[p]
            i == j && continue
            rowcnt[i] += 1
            lev[i] = ifelse(lev[i] > lj, lev[i], lj + 1)
        end
    end
    return lev .+ 1, rowcnt
end

"""
Backward-solve level scheduling of `L' x = y`: row `i` of `L'` is column `i` of
`L`, so `x[i]` reads `x[j]` for the stored `(j, i)` with `j > i`.  Sweeping `i`
down from `n` is a topological order of that DAG.
"""
function col_levels(L::SparseMatrixCSC, n::Int)
    rv = L.rowval; cp = L.colptr
    lev = zeros(Int, n); colcnt = zeros(Int, n)
    @inbounds for i in n:-1:1
        l = 0; c = 0
        for p in cp[i]:(cp[i + 1] - 1)
            j = rv[p]
            j == i && continue
            c += 1
            l = ifelse(lev[j] > l, lev[j], l)
        end
        lev[i] = l + 1
        colcnt[i] = c
    end
    return lev, colcnt
end

"""Elimination tree of `L` (column etree) and its depth."""
function col_etree(L::SparseMatrixCSC, n::Int)
    rv = L.rowval; cp = L.colptr
    par = zeros(Int, n)
    @inbounds for j in 1:n
        for p in cp[j]:(cp[j + 1] - 1)
            i = rv[p]
            if i > j
                par[j] = i
                break
            end
        end
    end
    depth = zeros(Int, n)
    for j in 1:n
        k = j; d = 0
        while par[k] != 0
            d += 1
            k = par[k]
        end
        depth[j] = d
    end
    return par, maximum(depth)
end

"""Level statistics of one direction: levels, widths, work, critical path."""
function level_stats(lev, cnt, n)
    nl = maximum(lev)
    width = zeros(Int, nl); work = zeros(Int, nl); wmax = zeros(Int, nl)
    for i in 1:n
        w = cnt[i] + 1
        width[lev[i]] += 1
        work[lev[i]] += w
        wmax[lev[i]] = max(wmax[lev[i]], w)
    end
    return (nl=nl, width=width, work=work, crit=sum(wmax), total=sum(work),
            wmax=wmax)
end

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

function main()
    isfile(CKPT) || error("checkpoint not found: $CKPT")
    tmpdir = mktempdir(; cleanup=false)
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
    eta_fun = (x, y) -> begin
        T = Tbar(y, pr.H) + Tfun(x, y)
        ed = semquad_strain_rate_at(ss.su, ux, uz, x, y, leaves)
        eta_composite_at(T, ed, y, pr)[1]
    end
    fz = (x, y) -> pr.Ra * Tfun(x, y)
    if !isempty(ph)
        fz_thermal = fz
        fz = (x, y) -> fz_thermal(x, y) +
                      semquad_phase_buoyancy(pr, ph, y, Tfun(x, y))
    end
    println("="^100)
    @printf("chol(S1) solve cost   case=%s julia_threads=%d mg_threads=%d BLAS=1\n",
            CASE, Threads.nthreads(), MGTHR)
    println("="^100)

    sys = solve_semquad_stokes_var(ss, qt, eta_fun, Z2, fz, Z2, Z2;
                                   bc=:free, deviatoric=false,
                                   mass_form=:boussinesq, linear=:mg,
                                   mg_operator=:matrixfree,
                                   assemble_only=:mg)
    nf = size(sys.Bs, 2); nfp = size(sys.Bs, 1); nftot = nf + nfp
    mfop = MS.semquad_kop(ss, qt; deviatoric=false)
    mfetav = MS.semquad_eta_lanes(mfop, MS.semquad_eta_field(mfop, eta_fun))
    mfwork = MS.SemQuadKWork(mfop.n, mfop.nq, 2 * ss.su.N)
    mfapplyK! = if MGTHR > 1
        par_outer = MS.SemQuadKPar(mfop, MGTHR)
        (out, v) -> MS._semquad_apply_Kred_threaded!(out, mfop, sys.Cu,
                                                     mfetav, v, mfwork,
                                                     par_outer)
    else
        (out, v) -> MS._semquad_apply_Kred!(out, mfop, sys.Cu, mfetav, v, mfwork)
    end
    mg = MS.make_semquad_mg(ss, qt, p, :free, false, eta_fun, nothing;
                            cycles=1, smoother=:jacobi,
                            coarse_operator=:rediscretize,
                            operator=:matrixfree, coarsen_layers=1,
                            coarsen_factor=1.5, coarse_direct_max=0,
                            smoothing=:auto, mg_threads=THR,
                            mg_diag=:constrained, fine_Cu=sys.Cu,
                            fine_mfop=mfop, fine_mfeta=mfetav)
    wfun = (x, y) -> sqrt(max(eta_fun(x, y), 1.0e-300))

    # ---- (1) the setup, charged in its four parts ------------------------
    d = MS._semquad_bfbt_diag(ss.su, sys.Cu, eta_fun, wfun; how=:exact)
    Dinvv = 1.0 ./ d
    Dinv = spdiagm(0 => Dinvv)
    t_D = tmin(() -> MS._semquad_bfbt_diag(ss.su, sys.Cu, eta_fun, wfun;
                                           how=:exact), REPS, 1)
    t_prod = tmin(() -> sys.Bs * Dinv * sys.Bt, REPS, 1)
    S1raw = sys.Bs * Dinv * sys.Bt
    dS1 = maximum(abs, diag(S1raw))
    t_sym = tmin(() -> 0.5 * (S1raw + SparseMatrixCSC(transpose(S1raw))), REPS, 1)
    S1 = 0.5 * (S1raw + SparseMatrixCSC(transpose(S1raw)))
    shift = 1.0e-14 * max(dS1, 1.0)
    S1s = S1 + shift * I
    t_chol = tmin(() -> cholesky(Symmetric(S1s)), REPS, 1)
    F1 = cholesky(Symmetric(S1s))
    issuccess(F1) || error("S1 factorisation failed")
    L = sparse(F1.L)
    n = size(S1, 1)
    nnzS1 = nnz(S1)
    nnzL = nnz(L)
    @printf("\nSETUP  n=%d nnz(S1)=%d (%.2f/row)\n", n, nnzS1, nnzS1 / n)
    @printf("  D build (:exact)        %8.2f ms\n", 1e3 * t_D)
    @printf("  S1 = Bs D^-1 Bt         %8.2f ms\n", 1e3 * t_prod)
    @printf("  symmetrise (S1+S1')/2   %8.2f ms\n", 1e3 * t_sym)
    @printf("  cholesky(Symmetric(S1)) %8.2f ms\n", 1e3 * t_chol)
    @printf("  TOTAL setup             %8.2f ms\n",
            1e3 * (t_D + t_prod + t_sym + t_chol))
    @printf("FACTOR nnz(L)=%d (%.2f/col)  fill vs S1 = %.2fx  diag(L) unit? %s  is_super=%d\n",
            nnzL, nnzL / n, nnzL / nnzS1, all(diag(L) .== 1.0), chol_is_super(F1))

    # ---- (5) structure ---------------------------------------------------
    flev, fcnt = row_levels(L, n)
    blev, bcnt = col_levels(L, n)
    fs = level_stats(flev, fcnt, n)
    bs = level_stats(blev, bcnt, n)
    par, edepth = col_etree(L, n)
    @printf("\nSTRUCTURE  forward solve (row dependencies)\n")
    @printf("  levels = %d ; width min %d median %d mean %.1f max %d\n",
            fs.nl, minimum(fs.width), med(fs.width), sum(fs.width) / fs.nl,
            maximum(fs.width))
    @printf("  levels with width < 8: %d (%.0f %%) ; work in levels of width >= 8: %.0f %%\n",
            count(<(8), fs.width), 100 * count(<(8), fs.width) / fs.nl,
            100 * sum(fs.work[fs.width .>= 8]) / fs.total)
    @printf("  work = %d nnz ; critical path = %d ; level-scheduling ceiling = %.2fx\n",
            fs.total, fs.crit, fs.total / fs.crit)
    @printf("STRUCTURE  backward solve (column dependencies)\n")
    @printf("  levels = %d ; width min %d median %d mean %.1f max %d\n",
            bs.nl, minimum(bs.width), med(bs.width), sum(bs.width) / bs.nl,
            maximum(bs.width))
    @printf("  levels with width < 8: %d (%.0f %%) ; work in levels of width >= 8: %.0f %%\n",
            count(<(8), bs.width), 100 * count(<(8), bs.width) / bs.nl,
            100 * sum(bs.work[bs.width .>= 8]) / bs.total)
    @printf("  work = %d nnz ; critical path = %d ; level-scheduling ceiling = %.2fx\n",
            bs.total, bs.crit, bs.total / bs.crit)
    @printf("  column etree depth = %d ; pair ceiling (fwd+bwd, unlimited threads) = %.2fx\n",
            edepth, (fs.total + bs.total) / (fs.crit + bs.crit))

    # barrier cost of one `SemQuadParRegion` phase on this thread count
    reg = MS.SemQuadParRegion(THR)
    ph_body = (c, w) -> nothing
    tph = 0.0
    if THR > 1
        MS._semquad_region_run!(reg) do
            for _ in 1:200
                MS._semquad_phase!(reg, ph_body, THR)
            end
        end
        tph = @elapsed MS._semquad_region_run!(reg) do
            for _ in 1:2000
                MS._semquad_phase!(reg, ph_body, THR)
            end
        end
        tph = 1e6 * tph / 2000
    end
    @printf("  SemQuadParRegion phase (barrier) = %.2f us ; %d forward levels => %.2f ms of pure barrier\n",
            tph, fs.nl, 1e-3 * tph * fs.nl)

    # ---- (2) one solve ---------------------------------------------------
    b1 = randn(n)
    rp2 = similar(b1)
    ldiv!(rp2, F1, b1)
    xa = F1 \ b1
    bitw = (ldiv!(rp2, F1, b1); rp2 == xa)
    al_alloc = @allocated (rp2 .= F1 \ b1)
    al_ldiv = @allocated ldiv!(rp2, F1, b1)
    t_alloc = 1e3 * tmin(() -> (rp2 .= F1 \ b1), 9, 20)
    t_ldiv = 1e3 * tmin(() -> ldiv!(rp2, F1, b1), 9, 20)
    @printf("\nSOLVE (one forward + one backward substitution, %d nnz)\n", 2 * nnzL)
    @printf("  hot, allocating  `rp2 .= F1 \\ rp`  %8.4f ms   (%d B allocated/call)\n",
            t_alloc, al_alloc)
    @printf("  hot, `ldiv!(rp2, F1, rp)`           %8.4f ms   (%d B allocated/call)\n",
            t_ldiv, al_ldiv)
    @printf("  bitwise equal to the allocating form: %s\n", bitw)
    @printf("  per nonzero: %.2f ns ; L is %.2f MB\n", 1e6 * t_ldiv / (2 * nnzL),
            12 * nnzL / 1e6)

    # ---- (3) the memory reference ---------------------------------------
    # A sparse matvec over L has the same nnz and the same access pattern as one
    # substitution pass: it measures the streaming rate a solve cannot beat for
    # one pass over the factor.
    x = randn(n); y = zeros(n)
    mul!(y, L, x)
    tmv = 1e3 * tmin(() -> mul!(y, L, x), 9, 20)
    @printf("\nMEMORY  one pass over L as a sparse matvec: %.4f ms  (%.1f GB/s)\n",
            tmv, (12 * nnzL + 8 * n) / (1e-3 * tmv) / 1e9)
    @printf("        one solve = two passes => %.4f ms of traffic ; hot solve %.4f ms (%.2fx of it)\n",
            2 * tmv, t_ldiv, t_ldiv / (2 * tmv))

    # ---- (4) other factorisations of the same matrix ---------------------
    variants = [("as shipped (Julia's CHOLMOD defaults)", () -> cholesky(Symmetric(S1s))),
                ("supernodal forced", () -> chol_super(S1s, 2)),
                ("full ordering search (nmethods=0)", () -> chol_ord(S1s, 0)),
                ("METIS only (nmethods=1)", () -> chol_ord(S1s, 1))]
    @printf("\nFACTOR VARIANTS of the same S1 (same preconditioner up to round-off)\n")
    @printf("  %-40s %8s %8s %8s %9s %9s\n", "variant", "nnz(L)", "chol ms",
            "hot ms", "levels", "ceiling")
    for (nm, mk) in variants
        local ok = true
        try
            mk()
        catch err
            ok = false
            @printf("  %-40s FAILED: %s\n", nm, sprint(showerror, err)[1:min(end, 60)])
        end
        ok || continue
        local tcv = 1e3 * tmin(mk, REPS, 1)
        local Fv = mk()
        local Lv = sparse(Fv.L)
        local xv = similar(b1)
        ldiv!(xv, Fv, b1)
        local dv = norm(xv - xa) / norm(xa)
        local thv = 1e3 * tmin(() -> ldiv!(xv, Fv, b1), 9, 20)
        local lvv, cvv = row_levels(Lv, n)
        local stv = level_stats(lvv, cvv, n)
        @printf("  %-40s %8d %8.2f %8.4f %9d %9.2f  (|dx|/|x| %.1e)\n",
                nm, nnz(Lv), tcv, thv, stv.nl, stv.total / stv.crit, dv)
    end

    println("\nSUMMARY case=$CASE julia_threads=$(Threads.nthreads()) mg_threads=$MGTHR " *
            "n=$n nnzS1=$nnzS1 nnzL=$nnzL flev=$(fs.nl) blev=$(bs.nl) etreedepth=$edepth")
    @printf("SUMMARY t_D=%.2f t_prod=%.2f t_sym=%.2f t_chol=%.2f t_setup=%.2f\n",
            1e3 * t_D, 1e3 * t_prod, 1e3 * t_sym, 1e3 * t_chol,
            1e3 * (t_D + t_prod + t_sym + t_chol))
    @printf("SUMMARY t_alloc=%.4f t_ldiv=%.4f alloc_bytes=%d ldiv_bytes=%d\n",
            t_alloc, t_ldiv, al_alloc, al_ldiv)
    @printf("SUMMARY fwd_ceiling=%.2f bwd_ceiling=%.2f pair_ceiling=%.2f crit_fwd=%d crit_bwd=%d tbarrier_us=%.2f\n",
            fs.total / fs.crit, bs.total / bs.crit,
            (fs.total + bs.total) / (fs.crit + bs.crit), fs.crit, bs.crit, tph)
    @printf("SUMMARY tmatvec=%.4f\n", tmv)
end

main()