#!/usr/bin/env julia
# =============================================================================
# bench_bfbt_leafdiag.jl -- the exact restricted BFBT diagonal, measured on the
# two production states (docs/dev/MG.md)
#
#   julia --project=. --threads=8 benchmarks/bench_bfbt_leafdiag.jl F
#   julia --project=. --threads=1 benchmarks/bench_bfbt_leafdiag.jl P1
#
# For each state it measures, at the process's Julia thread count:
#
#   * the construction of the BFBT weight `D` for the two leaf paths
#     (`:leaf` = the shipped lumped restriction `Cu' diag(M_u)`, `:exact` =
#     the leaf-local exact `diag(Cu' M_u Cu)`), median of `REPS`;
#   * the whole pressure block (`S1` + `chol`) with each weight, charged once;
#   * one FGMRES solve with each block (median of `REPS`) at the production
#     tolerance: iterations, true residual, wall clock;
#   * accuracy against the equilibrated direct solve (`|du|/|u|`, gauge-removed
#     `|dp|/|p|`), never against `relres`;
#   * the production warm solve (`solve_semquad_stokes_var`, cached pressure
#     block) with `mg_pressure_bfbt_d=:leaf` and with the shipped default
#     (`:auto` -> `:exact` on the matrix-free path), median of `REPS`.
#
# The checkpoint is copied to a temporary directory and opened read-only;
# nothing is written into the repository.
#
# Environment:
#   BFD_REPS    timed repeats (default 3)
#   BFD_TOL     outer tolerance (default 1e-6)
#   BFD_MAXIT   outer cap (default 500)
#   BFD_DIRECT  0 skips the direct anchor (no accuracy numbers)
#   BFD_THR     mg_threads (default = Julia threads)
# =============================================================================

using QuaST, Printf, LinearAlgebra, SparseArrays
const MS = QuaST

const REPO = normpath(joinpath(@__DIR__, ".."))
const CASE_DIRS = Dict(
    "F" => joinpath(REPO, "out_julia", "_s7", "case", "pre_step6131.h5"),
    "P1" => joinpath(REPO, "out_julia", "s340", "ab_none", "resume.h5"))
const CASE = isempty(ARGS) ? "F" : uppercase(ARGS[1])
const CKPT = get(CASE_DIRS, CASE, "")
const REPS = parse(Int, get(ENV, "BFD_REPS", "3"))
const TOL = parse(Float64, get(ENV, "BFD_TOL", "1.0e-6"))
const MAXIT = parse(Int, get(ENV, "BFD_MAXIT", "500"))
const DO_DIRECT = get(ENV, "BFD_DIRECT", "1") == "1"
const THR = parse(Int, get(ENV, "BFD_THR", string(Threads.nthreads())))
const MGTHR = MS._semquad_mg_thread_count(THR)
const Z2 = (x, y) -> 0.0

BLAS.set_num_threads(1)

med(v) = (s = sort(copy(v)); s[cld(length(s), 2)])

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
    @printf("exact BFBT diagonal   case=%s julia_threads=%d mg_threads=%d BLAS=1\n",
            CASE, Threads.nthreads(), MGTHR)
    @printf("checkpoint=%s  leaves=%d  saddle=%d (%d + %d)  tol=%.1e\n",
            copyto, length(qt.leaves), 2 * ss.su.N + ss.sp.N, 2 * ss.su.N,
            ss.sp.N, TOL)
    println("="^100)

    # ---- direct reference ------------------------------------------------
    local u_dir, p_dir
    if DO_DIRECT
        dkw = (; bc=:free, deviatoric=false, mass_form=:boussinesq,
                linear=:direct, stokes_scaling=:equilibrated)
        solve_direct() = solve_semquad_stokes_var(ss, qt, eta_fun, Z2, fz, Z2,
                                                  Z2; dkw...)
        solve_direct()
        t0 = time(); u_dir, p_dir = solve_direct(); t_dir = time() - t0
        @printf("direct reference (:equilibrated): %.3f s, relres %.3e\n",
                t_dir, LAST_SEMSTOKES_RESIDUAL[])
    else
        u_dir = Float64[]; p_dir = Float64[]
    end

    # ---- reduced saddle + hierarchy -------------------------------------
    sys = solve_semquad_stokes_var(ss, qt, eta_fun, Z2, fz, Z2, Z2;
                                   bc=:free, deviatoric=false,
                                   mass_form=:boussinesq, linear=:mg,
                                   mg_operator=:matrixfree,
                                   assemble_only=:mg)
    nf = size(sys.Bs, 2); nfp = size(sys.Bs, 1); nftot = nf + nfp
    rhs = vcat(sys.fr, sys.gr)
    expand(sol) = (sys.u0 + sys.Cu * sol[1:nf],
                   sys.p0 + sys.Cp * sol[(nf + 1):nftot])
    function accuracy(u, pv)
        DO_DIRECT || return (NaN, NaN)
        du = norm(u - u_dir) / norm(u_dir)
        pg = pv .- sum(pv) / length(pv)
        pdg = p_dir .- sum(p_dir) / length(p_dir)
        return du, norm(pg - pdg) / norm(pdg)
    end

    mfop = MS.semquad_kop(ss, qt; deviatoric=false)
    mfetav = MS.semquad_eta_lanes(mfop, MS.semquad_eta_field(mfop, eta_fun))
    mfwork = MS.SemQuadKWork(mfop.n, mfop.nq, 2 * ss.su.N)
    nt_outer = MS._semquad_mg_thread_count(THR)
    mfapplyK! = if nt_outer > 1
        par_outer = MS.SemQuadKPar(mfop, nt_outer)
        (out, v) -> MS._semquad_apply_Kred_threaded!(out, mfop, sys.Cu,
                                                     mfetav, v, mfwork,
                                                     par_outer)
    else
        (out, v) -> MS._semquad_apply_Kred!(out, mfop, sys.Cu, mfetav, v,
                                            mfwork)
    end
    mg = MS.make_semquad_mg(ss, qt, p, :free, false, eta_fun, nothing;
                            cycles=1, smoother=:jacobi,
                            coarse_operator=:rediscretize,
                            operator=:matrixfree, coarsen_layers=1,
                            coarsen_factor=1.5, coarse_direct_max=0,
                            smoothing=:auto, mg_threads=THR,
                            mg_diag=:constrained, fine_Cu=sys.Cu,
                            fine_mfop=mfop, fine_mfeta=mfetav)
    key = MS._saddle_key(qt, p, :free, nf, nfp)
    wfun = (x, y) -> sqrt(max(eta_fun(x, y), 1.0e-300))

    rows = NamedTuple[]
    for weight in (:leaf, :exact)
        tds = Float64[]
        for _ in 1:REPS
            push!(tds, @elapsed MS._semquad_bfbt_diag(ss.su, sys.Cu, eta_fun,
                                                      wfun; how=weight))
        end
        t_D = med(tds)
        t_Dmin = minimum(tds)
        d = MS._semquad_bfbt_diag(ss.su, sys.Cu, eta_fun, wfun; how=weight)
        buildwp() = MS.semquad_mg_pressure_precond(
            ss, sys.Cu, sys.Cp, sys.Bs, sys.Bt, nothing, eta_fun;
            mode=:auto, scale=0.0, weight=:inv_eta, key=key, cache=false,
            bfbt_d=weight, applyK=mfapplyK!)
        buildwp()                       # warm-up: first call pays the JIT
        tbs = Float64[]
        local Fp, pscale
        for _ in 1:REPS
            t0 = time()
            Fp, pscale = buildwp()
            push!(tbs, time() - t0)
        end
        t_build = med(tbs)
        Fp isa NamedTuple && Fp.kind === :bfbt ||
            error("expected the w-BFBT block, got $(Fp isa NamedTuple ? Fp.kind : Fp)")
        solve1() = MS.semquad_mg_saddle_solve(nothing, sys.Bs, sys.Bt, Fp,
                                              pscale, rhs, mg; tol=TOL,
                                              maxit=MAXIT, restart=100,
                                              vel_solver=:vcycle,
                                              applyK=mfapplyK!)
        sol, nit, res = solve1()[1:3]
        t0 = time(); solve1(); t_first = time() - t0
        ts = Float64[]
        for _ in 1:REPS
            t0 = time(); solve1(); push!(ts, time() - t0)
        end
        t_solve = med(ts)
        u, pv = expand(sol)
        du, dpg = accuracy(u, pv)
        @printf("\n[%s] D build %.4f s (min %.4f) | block build %.4f s | %d it | %.4f s (first %.4f s, %.2f ms/it)\n",
                weight, t_D, t_Dmin, t_build, nit, t_solve, t_first,
                1e3 * t_solve / nit)
        @printf("      relres %.4e  |du|/|u| %.4e  gauge-removed |dp|/|p| %.4e  contrast %.3e\n",
                res, du, dpg, maximum(d) / minimum(d))
        push!(rows, (weight=weight, t_D=t_D, t_Dmin=t_Dmin,
                     t_build=t_build, nit=nit,
                     t_solve=t_solve, t_first=t_first, res=res, du=du,
                     dpg=dpg, contrast=maximum(d) / minimum(d)))
    end

    # ---- production warm solve: same tree, cached pressure block ---------
    for weight in (:leaf, :exact)
        mgkw = (; bc=:free, deviatoric=false, mass_form=:boussinesq,
                linear=:mg, mg_operator=:matrixfree, mg_vel_solver=:vcycle,
                mg_threads=THR, mg_tol=TOL, mg_pressure_bfbt_d=weight)
        warm() = solve_semquad_stokes_var(ss, qt, eta_fun, Z2, fz, Z2, Z2;
                                          mgkw...)
        empty!(MS.SEMQUAD_MG_CACHE)
        MS.SEMQUAD_PRESSURE_CACHE[] = nothing
        warm()          # builds the hierarchy + the pressure block
        ts = Float64[]
        for _ in 1:REPS
            t0 = time(); warm(); push!(ts, time() - t0)
        end
        @printf("\nwarm solve [%s]: %.4f s (median of %d), %d it, relres %.4e\n",
                weight, med(ts), REPS, LAST_SEMSTOKES_SCHUR_ITERS[],
                LAST_SEMSTOKES_RESIDUAL[])
    end
    # ---- the shipped default (`:auto` -> `:exact` matrix-free) -----------
    let
        mgkw = (; bc=:free, deviatoric=false, mass_form=:boussinesq,
                linear=:mg, mg_operator=:matrixfree, mg_vel_solver=:vcycle,
                mg_threads=THR, mg_tol=TOL)
        warm() = solve_semquad_stokes_var(ss, qt, eta_fun, Z2, fz, Z2, Z2;
                                          mgkw...)
        empty!(MS.SEMQUAD_MG_CACHE)
        MS.SEMQUAD_PRESSURE_CACHE[] = nothing
        warm()
        stc = MS.SEMQUAD_PRESSURE_CACHE[]
        dshipped = 1.0 ./ stc.Fp.Dinvv
        ts = Float64[]
        for _ in 1:REPS
            t0 = time(); warm(); push!(ts, time() - t0)
        end
        @printf("\nwarm solve [shipped default]: %.4f s (median of %d), %d it, relres %.4e, D contrast %.3e\n",
                med(ts), REPS, LAST_SEMSTOKES_SCHUR_ITERS[],
                LAST_SEMSTOKES_RESIDUAL[], maximum(dshipped) / minimum(dshipped))
        println("SUMMARY default t_solve=$(round(med(ts); digits=4)) nit=$(LAST_SEMSTOKES_SCHUR_ITERS[]) contrast=$(maximum(dshipped) / minimum(dshipped))")
    end

    println("\nSUMMARY case=$CASE julia_threads=$(Threads.nthreads()) mg_threads=$MGTHR " *
            "nf=$nf nfp=$nfp")
    for r in rows
        @printf("SUMMARY weight=%s t_D=%.4f t_Dmin=%.4f t_build=%.4f nit=%d t_solve=%.4f t_first=%.4f relres=%.3e du=%.3e dpg=%.3e contrast=%.3e\n",
                r.weight, r.t_D, r.t_Dmin, r.t_build, r.nit, r.t_solve,
                r.t_first, r.res, r.du, r.dpg, r.contrast)
    end
end

main()
