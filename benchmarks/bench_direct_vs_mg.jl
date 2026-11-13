#!/usr/bin/env julia
# =============================================================================
# bench_direct_vs_mg.jl -- same-session head-to-head of the direct saddle solve
# and the current matrix-free multigrid defaults on the two production states.
#
#   julia --project=. --threads=8 \
#       benchmarks/bench_direct_vs_mg.jl <F|P1> [reps]
#
# Both paths run in one process on one read-only copy of the checkpoint, so the
# ratio does not carry the 4-6 % session drift of separate runs.  `--threads=N`
# is the Julia process; the MG V-cycle gets `mg_threads=N` (the production
# default at this size, `min(8, nthreads)`), the direct path takes its production
# keywords (`assemble_threads` is left at its serial default; its LU is serial).
#
# Reported for each path: assembly, MG setup (hierarchy + w-BFBT), direct
# saddle build + factorisation/solve, MG FGMRES iteration, cold full solve, warm
# in-tree step, and peak RSS (VmHWM, reset through /proc/self/clear_refs just
# before the measured solve).  The direct reference is `linear=:direct` with
# `stokes_scaling=:equilibrated`, the configuration the MG accuracy rows use.
# Nothing is written into the repository: the checkpoint is copied to a temp dir
# and opened read-only.
# =============================================================================

using QuaST, Printf, LinearAlgebra, SparseArrays, Random

const MS = QuaST
const REPO = normpath(joinpath(@__DIR__, ".."))
const Z2 = (x, y) -> 0.0
BLAS.set_num_threads(1)

@inline el(t0::UInt64) = 1.0e-9 * (time_ns() - t0)

function rss_gb(field::AbstractString)
    for line in eachline("/proc/self/status")
        startswith(line, field) && return parse(Float64, split(line)[2]) / 1024^2
    end
    return NaN
end
peak_rss_gb() = rss_gb("VmHWM:")
rss_gb() = rss_gb("VmRSS:")

function reset_peak!()
    try
        open("/proc/self/clear_refs", "w") do io
            write(io, "5")          # Linux: reset VmHWM to current RSS
        end
        return true
    catch
        return false
    end
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

"""Load a production checkpoint into the production closure objects."""
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
    return (; qt, ss, eta, fz, tag=basename(path), p, step=st.step, tmpdir)
end

function tmin(f, n::Int)
    f()
    best = Inf
    for _ in 1:n
        GC.gc()
        best = min(best, @elapsed f())
    end
    return best
end

"""The direct branch's post-assembly work, exactly as
`solve_semquad_stokes_var(linear=:direct, stokes_scaling=:equilibrated)` runs it:
saddle build, unshifted LU, residual, then the equilibrated LU and residual."""
function direct_saddle_block(sys, nfp)
    t0 = time_ns()
    S = [sys.Kred sys.Bt; sys.Bs spzeros(nfp, nfp)]
    rhs = vcat(sys.fr, sys.gr)
    rden0 = max(norm(sys.fr), norm(sys.gr), 1.0)
    rel = x -> begin
        rv = norm(sys.Kred * x[1:end-nfp] +
                  sys.Bt * x[(end-nfp+1):end] - sys.fr) / rden0
        rd = norm(sys.Bs * x[1:end-nfp] - sys.gr) / rden0
        max(rv, rd)
    end
    t_un = @elapsed x_un = S \ rhs
    r_un = rel(x_un)
    t_eq = @elapsed ((sol_scaled, _) =
        MS._semquad_stokes_equil_solve(S, sys.Kred, sys.Bt, sys.Bs, rhs))
    r_sc = rel(sol_scaled)
    return el(t0), r_un, r_sc, t_un, t_eq
end

function main()
    length(ARGS) >= 1 || error("usage: bench_direct_vs_mg.jl <F|P1> [reps]")
    case = ARGS[1]
    reps = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
    thr = parse(Int, get(ENV, "BENCH_THR", string(Threads.nthreads())))
    path = case == "F" ?
        joinpath(REPO, "out_julia", "_s7", "case", "pre_step6131.h5") :
        case == "P1" ?
        joinpath(REPO, "out_julia", "s340", "ab_none", "resume.h5") :
        error("unknown case $case")
    (; qt, ss, eta, fz, tag, p, step, tmpdir) = checkpoint_case(path)
    nf = 2 * ss.su.N
    @printf("### case=%s checkpoint=%s step=%d leaves=%d levels=%d p=%d vdofs=%d thr=%d reps=%d\n",
            case, tag, step, length(qt.leaves),
            length(unique(lf[1] for lf in qt.leaves)), p, nf, thr, reps)

    dkw = (; bc=:free, deviatoric=false, mass_form=:boussinesq,
             linear=:direct, stokes_scaling=:equilibrated)
    mkw = (; bc=:free, deviatoric=false, mass_form=:boussinesq,
             linear=:mg, mg_operator=:matrixfree, mg_vel_solver=:vcycle,
             mg_threads=thr, mg_tol=1.0e-6)
    solve_direct() = solve_semquad_stokes_var(ss, qt, eta, Z2, fz, Z2, Z2; dkw...)
    solve_mg() = solve_semquad_stokes_var(ss, qt, eta, Z2, fz, Z2, Z2; mkw...)

    # JIT / allocator warm-up; never measured.
    solve_direct(); solve_mg()
    GC.gc()

    # Assembly: the direct path's full assembly (forms K, Kred) and the
    # matrix-free MG path's reduced assembly (no K, no Kred).
    local sysd
    t_asm_d = Inf
    for _ in 1:3
        GC.gc()
        t = @elapsed (sysd = solve_semquad_stokes_var(ss, qt, eta, Z2, fz, Z2, Z2;
                                                      bc=:free, deviatoric=false,
                                                      mass_form=:boussinesq,
                                                      linear=:direct,
                                                      assemble_only=true))
        t_asm_d = min(t_asm_d, t)
    end
    nfp = size(sysd.Bs, 1)
    t_asm_m = tmin(() -> solve_semquad_stokes_var(ss, qt, eta, Z2, fz, Z2, Z2;
                                                  bc=:free, deviatoric=false,
                                                  mass_form=:boussinesq,
                                                  linear=:mg,
                                                  mg_operator=:matrixfree,
                                                  assemble_only=:mg), 2)
    sysmg = solve_semquad_stokes_var(ss, qt, eta, Z2, fz, Z2, Z2;
                                     bc=:free, deviatoric=false,
                                     mass_form=:boussinesq, linear=:mg,
                                     mg_operator=:matrixfree,
                                     assemble_only=:mg)

    # MG setup: same calls the production solve makes, including the finest
    # operator/eta hand-off (`fine_mfop`/`fine_mfeta`).
    mfop = MS.semquad_kop(ss, qt; deviatoric=false)
    mfetav = MS.semquad_eta_field(mfop, eta)
    mfwork = MS.SemQuadKWork(mfop.n, mfop.nq, 2 * ss.su.N)
    nt_outer = MS._semquad_mg_thread_count(thr)
    mfapplyK = if nt_outer > 1
        par = MS.SemQuadKPar(mfop, nt_outer)
        (out, v) -> MS._semquad_apply_Kred_threaded!(out, mfop, sysmg.Cu, mfetav,
                                                     v, mfwork, par)
    else
        (out, v) -> MS._semquad_apply_Kred!(out, mfop, sysmg.Cu, mfetav, v,
                                            mfwork)
    end
    local mg
    t_hier = Inf
    for _ in 1:2
        empty!(MS.SEMQUAD_MG_CACHE)
        t = @elapsed (mg = MS.make_semquad_mg(ss, qt, p, :free, false, eta,
                                              nothing; cycles=1,
                                              smoother=:jacobi,
                                              coarse_operator=:rediscretize,
                                              operator=:matrixfree,
                                              coarsen_layers=1,
                                              coarsen_factor=1.5,
                                              coarse_direct_max=0,
                                              smoothing=:auto,
                                              mg_threads=thr,
                                              fine_Cu=sysmg.Cu,
                                              fine_mfop=mfop,
                                              fine_mfeta=mfetav))
        t_hier = min(t_hier, t)
    end
    local Fp, pscale
    t_bfbt = Inf
    for _ in 1:2
        t = @elapsed ((Fp, pscale) =
            MS.semquad_mg_pressure_precond(ss, sysmg.Cu, sysmg.Cp, sysmg.Bs,
                                           sysmg.Bt, nothing, eta; mode=:auto,
                                           scale=0.0, cache=false,
                                           applyK=mfapplyK))
        t_bfbt = min(t_bfbt, t)
    end
    rhs = vcat(sysmg.fr, sysmg.gr)
    local sol_mg, nit, relres
    t_iter = Inf
    for _ in 1:3
        GC.gc()
        t = @elapsed ((sol_mg, nit, relres, _) =
            MS.semquad_mg_saddle_solve(nothing, sysmg.Bs, sysmg.Bt, Fp, pscale,
                                       rhs, mg; tol=1.0e-6, maxit=500,
                                       restart=100, vel_solver=:vcycle,
                                       applyK=mfapplyK))
        t_iter = min(t_iter, t)
    end
    t_setup_m = t_hier + t_bfbt

    # Direct post-assembly work, timed on the same assembled system.
    local t_fac, r_un, r_sc, t_un, t_eq
    t_fac = Inf
    for _ in 1:2
        GC.gc()
        t, ru, rs, tu, te = direct_saddle_block(sysd, nfp)
        t_fac = min(t_fac, t)
        r_un, r_sc, t_un, t_eq = ru, rs, tu, te
    end

    # Cold solves, alternating the two paths.  Cold = clear the MG caches (the
    # direct path has no solver cache); the warm rows are taken afterwards as
    # consecutive in-tree solves.
    cold_d = Float64[]; cold_m = Float64[]
    warm_d = Float64[]; warm_m = Float64[]
    peak_d = NaN; peak_m = NaN; base_d = NaN; base_m = NaN
    du = dp = dpg = NaN
    d_phase = :none; d_shift = NaN; d_scaled = 0; d_res = NaN
    for rep in 1:reps
        GC.gc()
        reset_peak!()
        base_d = rss_gb()
        push!(cold_d, @elapsed (ud, pd, _, _, _) = solve_direct())
        peak_d = peak_rss_gb()
        d_phase = LAST_SEMSTOKES_LADDER_PHASE[]
        d_shift = LAST_SEMSTOKES_SHIFT[]
        d_scaled = LAST_SEMSTOKES_SCALED[]
        d_res = LAST_SEMSTOKES_RESIDUAL[]
        empty!(MS.SEMQUAD_MG_CACHE); MS.SEMQUAD_PRESSURE_CACHE[] = nothing
        GC.gc()
        reset_peak!()
        base_m = rss_gb()
        push!(cold_m, @elapsed (um, pm, _, _, _) = solve_mg())
        peak_m = peak_rss_gb()
        du = norm(um - ud) / norm(ud)
        dp = norm(pm - pd) / norm(pd)
        udg = ud .- sum(ud) / length(ud)
        pdg = pd .- sum(pd) / length(pd)
        dpg = norm((pm .- sum(pm) / length(pm)) - pdg) / norm(pdg)
    end
    # Warm in-tree steps: consecutive solves, no cache clear and no GC between
    # them, alternating the two paths as a production sequence would.
    for _ in 1:3
        push!(warm_d, @elapsed solve_direct())
        push!(warm_m, @elapsed solve_mg())
    end
    # Control: the direct path without solver-side equilibration.  It is the
    # older production configuration; the equilibrated path above pays one extra
    # (unshifted) factorisation before the scaled one.
    dnone = merge(dkw, (; stokes_scaling=:none))
    solve_direct_none() = solve_semquad_stokes_var(ss, qt, eta, Z2, fz, Z2, Z2;
                                                   dnone...)
    solve_direct_none()
    GC.gc(); t_none_cold = @elapsed solve_direct_none()
    t_none_warm = @elapsed solve_direct_none()
    none_phase = LAST_SEMSTOKES_LADDER_PHASE[]
    none_res = LAST_SEMSTOKES_RESIDUAL[]
    med(v) = sort(v)[cld(length(v), 2)]
    @printf("### direct phase=%s shift=%.1e scaled=%d relative_residual=%.3e\n",
            d_phase, d_shift, d_scaled, d_res)
    @printf("### asm direct=%.4f mg_reduced=%.4f  setup_mg hier=%.4f bfbt=%.4f total=%.4f\n",
            t_asm_d, t_asm_m, t_hier, t_bfbt, t_setup_m)
    @printf("### mg levels=%d nfs=%s threads=%d smoothing=%s omega=%.3f\n",
            mg.geo.L + 1, join(mg.geo.nfs, ","), mg.threads,
            MS.LAST_SEMSTOKES_MG_PROFILE[][1], mg.omega)
    @printf("### direct saddle block=%.4f s (unshifted LU %.4f + equilibrated LU %.4f) r_unshifted=%.2e r_equilibrated=%.2e\n",
            t_fac, t_un, t_eq, r_un, r_sc)
    @printf("### direct/noequil cold=%.4f warm=%.4f phase=%s res=%.3e\n",
            t_none_cold, t_none_warm, none_phase, none_res)
    @printf("### iter mg=%.4f s nit=%d per_it=%.2f ms true_relres=%.3e\n",
            t_iter, nit, 1e3 * t_iter / max(nit, 1), relres)
    @printf("### mg_assembly_inferred=%.4f s (cold - hier - bfbt - iter)\n",
            minimum(cold_m) - t_hier - t_bfbt - t_iter)
    @printf("### cold direct=%.4f/%.4f mg=%.4f/%.4f s (min/med)\n",
            minimum(cold_d), med(cold_d), minimum(cold_m), med(cold_m))
    @printf("### warm direct=%.4f/%.4f mg=%.4f/%.4f s (min/med)\n",
            minimum(warm_d), med(warm_d), minimum(warm_m), med(warm_m))
    @printf("### peak direct=%.2f GB (base %.2f)  mg=%.2f GB (base %.2f)\n",
            peak_d, base_d, peak_m, base_m)
    @printf("### acc du=%.3e dp=%.3e dpg=%.3e\n", du, dp, dpg)
    @printf("RESULT case=%s step=%d leaves=%d thr=%d asm_d=%.4f asm_m=%.4f hier=%.4f bfbt=%.4f setup_m=%.4f fac_d=%.4f iter_m=%.4f nit=%d per_it=%.2f cold_d=%.4f cold_m=%.4f warm_d=%.4f warm_m=%.4f cold_med_d=%.4f cold_med_m=%.4f warm_med_d=%.4f warm_med_m=%.4f peak_d=%.2f peak_m=%.2f base_d=%.2f base_m=%.2f du=%.3e dp=%.3e dpg=%.3e res_m=%.3e r_un=%.2e r_sc=%.2e\n",
            case, step, length(qt.leaves), thr, t_asm_d, t_asm_m, t_hier,
            t_bfbt, t_setup_m, t_fac, t_iter, nit, 1e3 * t_iter / max(nit, 1),
            minimum(cold_d), minimum(cold_m), minimum(warm_d), minimum(warm_m),
            med(cold_d), med(cold_m), med(warm_d), med(warm_m), peak_d,
            peak_m, base_d, base_m, du, dp, dpg, relres, r_un, r_sc)
    rm(tmpdir; force=true, recursive=true)
end

main()
