#!/usr/bin/env julia
# =============================================================================
# bench_card_mg_tol.jl -- the realistic card's MG outer tolerance, measured on
# the three production states (docs/dev/CASE_REALISTIC.md §1.6)
#
#   julia --project=. --threads=8 benchmarks/bench_card_mg_tol.jl F [reps]
#   julia --project=. --threads=8 benchmarks/bench_card_mg_tol.jl P1
#   julia --project=. --threads=8 benchmarks/bench_card_mg_tol.jl C
#
# For each `mg_tol` in {1e-10, 1e-8, 1e-6, 1e-5} it runs the card's exact
# solver kwargs -- `linear=:auto`, `auto_fallback_tol = stokes_accept_tol =
# 1e-3`, so the selector resolves to matrix-free MG at these sizes -- on a
# read-only copy of the checkpoint, and records
#
#   * the FGMRES outer iteration count and the true reduced-saddle residual
#     (`LAST_SEMSTOKES_RESIDUAL`, the driver's acceptance metric),
#   * the warm solve (median of `reps`; the cold build is not timed),
#   * the auto decision / fallback count and whether the driver accepts,
#   * the difference of the returned (u, p) against the 1e-10 run:
#     `|du|/|u|` and the gauge-removed `|dp|/|p|`,
#   * the same difference against the equilibrated direct solve, for context
#     (not part of the card decision).
#
# The checkpoint is copied to a temporary directory and opened read-only;
# nothing is written into the repository.  `F` is
# out_julia/_s7/case/pre_step6131.h5 (8 188 leaves), `P1`
# out_julia/s340/ab_none/resume.h5 (10 000 leaves) and `C` the 8 192-leaf
# C-arm checkpoint out_julia/case_realistic_C/resume.h5 (8 260 leaves).
# =============================================================================

using QuaST, Printf, LinearAlgebra, SparseArrays

const MS = QuaST
const REPO = normpath(joinpath(@__DIR__, ".."))
const Z2 = (x, y) -> 0.0
const TOLS = (1.0e-10, 1.0e-8, 1.0e-6, 1.0e-5)
const STOKES_TOL = 1.0e-3

BLAS.set_num_threads(1)

@inline el(t0::UInt64) = 1.0e-9 * (time_ns() - t0)
med(v) = sort(v)[cld(length(v), 2)]
gauge(p) = p .- sum(p) / length(p)          # the closed box has a constant mode

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

function checkpoint_case(path::AbstractString)
    isfile(path) || error("checkpoint not found: $path")
    tmpdir = mktempdir(; cleanup=false)
    copyto = joinpath(tmpdir, basename(path))
    cp(path, copyto; force=true)
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
        fz = (x, y) -> fz_thermal(x, y) +
                      semquad_phase_buoyancy(pr, ph, y, Tfun(x, y))
    end
    return (; qt, ss, eta, fz, tag=basename(path), p, step=st.step, tmpdir,
            leaves=length(qt.leaves), nlev=length(unique(lf[1] for lf in qt.leaves)),
            n=ss.su.N, np=ss.sp.N)
end

clear_caches!() = (empty!(MS.SEMQUAD_MG_CACHE);
                   MS.SEMQUAD_PRESSURE_CACHE[] = nothing)

function main()
    length(ARGS) >= 1 || error("usage: bench_card_mg_tol.jl <F|P1|C> [reps]")
    case = ARGS[1]
    reps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
    path = case == "F" ? joinpath(REPO, "out_julia", "_s7", "case", "pre_step6131.h5") :
           case == "P1" ? joinpath(REPO, "out_julia", "s340", "ab_none", "resume.h5") :
           case == "C" ? joinpath(REPO, "out_julia", "case_realistic_C", "resume.h5") :
           error("unknown case $case")
    (; qt, ss, eta, fz, tag, p, step, tmpdir, leaves, nlev, n, np) =
        checkpoint_case(path)
    @printf("### case=%s checkpoint=%s step=%d leaves=%d nlev=%d p=%d Nu=%d Np=%d saddle=%d thr=%d reps=%d\n",
            case, tag, step, leaves, nlev, p, n, np, 2 * n + np,
            Threads.nthreads(), reps)

    base = (; bc=:free, deviatoric=false, mass_form=:boussinesq)
    # Exactly what case_realistic_2d.jl runs, plus the swept mg_tol.
    akw(tol) = merge(base, (; linear=:auto, auto_fallback_tol=STOKES_TOL,
                             stokes_accept_tol=STOKES_TOL, mg_tol=tol))
    dkw = merge(base, (; linear=:direct, stokes_scaling=:equilibrated))

    # Direct anchor (equilibrated): the accuracy context of the 1e-10 row.
    GC.gc(); t0 = time_ns()
    u_dir, p_dir = solve_semquad_stokes_var(ss, qt, eta, Z2, fz, Z2, Z2; dkw...)
    t_dir = el(t0)
    res_dir = MS.LAST_SEMSTOKES_RESIDUAL[]
    @printf("direct cold=%.3f s res=%.3e\n", t_dir, res_dir)

    rows = NamedTuple[]
    for tol in TOLS
        clear_caches!()
        MS.LAST_AUTO_FALLBACK[] = (count=0, last_reason="")
        GC.gc(); t0 = time_ns()
        cold_sol = solve_semquad_stokes_var(ss, qt, eta, Z2, fz, Z2, Z2;
                                            akw(tol)...)
        cold = el(t0)
        ch = MS.LAST_SEMSTOKES_AUTO_CHOICE[]
        warm = Float64[]
        local sol
        for _ in 1:reps
            GC.gc()
            t = @elapsed (sol = solve_semquad_stokes_var(
                ss, qt, eta, Z2, fz, Z2, Z2; akw(tol)...))
            push!(warm, t)
        end
        fb = MS.LAST_AUTO_FALLBACK[].count
        res = MS.LAST_SEMSTOKES_RESIDUAL[]
        nit = MS.LAST_SEMSTOKES_SCHUR_ITERS[]
        u = sol[1]; pv = sol[2]
        dud = norm(u - u_dir) / norm(u_dir)
        dpd = norm(gauge(pv) - gauge(p_dir)) / norm(gauge(p_dir))
        push!(rows, (tol=tol, cold=cold, warm=warm, nit=nit, res=res, fb=fb,
                     choice=ch.choice, accepted=(res <= STOKES_TOL),
                     u=u, pv=pv, dud=dud, dpd=dpd,
                     threads=ch.mg_threads, dofs=ch.saddle_dofs,
                     contrast=ch.contrast))
        @printf("tol=%.0e cold=%.3f warm_min=%.3f warm_med=%.3f it=%d res=%.3e fb=%d accepted=%s |du_dir|=%.3e |dp_dir|=%.3e\n",
                tol, cold, minimum(warm), med(warm), nit, res, fb,
                res <= STOKES_TOL, dud, dpd)
    end

    println("--- difference against mg_tol=1e-10")
    ref = rows[1]
    for r in rows
        du = norm(r.u - ref.u) / norm(ref.u)
        dpg = norm(gauge(r.pv) - gauge(ref.pv)) / norm(gauge(ref.pv))
        @printf("tol=%.0e |du|/|u|=%.3e gauge-removed |dp|/|p|=%.3e it=%d res=%.3e warm_med=%.3f accepted=%s\n",
                r.tol, du, dpg, r.nit, r.res, med(r.warm), r.accepted)
    end

    println("SUMMARY case=$case thr=$(Threads.nthreads()) leaves=$leaves saddle=$(2 * n + np) reps=$reps t_direct=$(round(t_dir; digits=3)) res_direct=$(round(res_dir; sigdigits=3))")
    for r in rows
        du = norm(r.u - ref.u) / norm(ref.u)
        dpg = norm(gauge(r.pv) - gauge(ref.pv)) / norm(gauge(ref.pv))
        @printf("SUMMARY tol=%.0e cold=%.3f warm_min=%.3f warm_med=%.3f it=%d res=%.3e fb=%d accepted=%s du=%.3e dpg=%.3e dud=%.3e dpd=%.3e choice=%s threads=%d dofs=%d contrast=%.3e\n",
                r.tol, r.cold, minimum(r.warm), med(r.warm), r.nit, r.res, r.fb,
                r.accepted, du, dpg, r.dud, r.dpd, r.choice, r.threads,
                r.dofs, r.contrast)
    end
    rm(tmpdir; force=true, recursive=true)
end

main()
