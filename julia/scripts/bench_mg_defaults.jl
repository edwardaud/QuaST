#!/usr/bin/env julia
# =============================================================================
# bench_mg_defaults.jl -- cost/accuracy of the MF-MG default variants.
#
#   julia --project=julia --threads=1 \
#       julia/scripts/bench_mg_defaults.jl <case> <variant> [reps]
#
# <case>    = F | P1            production checkpoints
#             l5 | l6            uniform p=4 trees (constant eta)
#             blob19             blob frame-19 p=4 AMR tree (real eta)
# <variant> = before            coarsen_factor=1.0, smoothing=:auto (old default)
#             after             production defaults (matrix-free: factor 2.0)
#             taper:a,b,...     factor 2.0, smoothing=:taper, pre=post=<vector>
#             unif:n            factor 2.0, smoothing=:uniform, n pre + n post
#
# Prints one `RESULT ...` line of key=value pairs: hierarchy level count, setup
# (hierarchy + w-BFBT), per-iteration, iterations, total cold solve, accuracy
# against the direct solve (velocity and pressure), and the resolved smoothing
# schedule.  Checkpoint files are copied to a temp dir and opened read-only;
# nothing is written into the repository.
# =============================================================================

using MantleSEM, Printf, LinearAlgebra, SparseArrays, Random

const MS = MantleSEM
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const Z2 = (x, y) -> 0.0
const ONE = (x, y) -> 1.0
BLAS.set_num_threads(1)

@inline el(t0::UInt64) = 1.0e-9 * (time_ns() - t0)

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
    return (; qt, ss, eta, fz, tag=basename(path), p)
end

function uniform_tree(level::Int)
    qt = Quadtree(level)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

const BLOBREF = normpath(joinpath(REPO, "out_julia", "fresh_blob_kelly_1024"))
const BLOBPR = Params(Ra=1.0e8, H=5.0, eta_law=:expTz,
                      eta_b=log(16384.0), eta_c=log(64.0), bc=:free)

function load_blob_tree(knot::Int)
    lvs = Set{NTuple{3,Int}}()
    for ln in eachline(joinpath(BLOBREF, "frame_$(knot)_mesh.csv"))
        isempty(strip(ln)) && continue
        a = split(ln, ',')
        l = parse(Int, a[1]); h = 2.0^(-l)
        push!(lvs, (l, round(Int, parse(Float64, a[2]) / h),
                    round(Int, parse(Float64, a[3]) / h)))
    end
    return Quadtree(6, 1.0, 1, 1, lvs)
end

function load_blob_T(knot::Int)
    T = Array{Float64}(undef, 256, 256)
    open(joinpath(BLOBREF, "frame_$(knot)_T.bin"), "r") do io
        read!(io, T)
    end
    return T
end

function blob_case()
    qt = load_blob_tree(19)
    ss = make_semstokes(qt, 4)
    T = load_blob_T(19)
    h = 1.0 / (size(T, 1) - 1); n = size(T, 1)
    Tf = (x, y) -> begin
        fx = clamp(x / h, 0.0, n - 1.0000001)
        fy = clamp(y / h, 0.0, n - 1.0000001)
        i = Int(floor(fx)) + 1; j = Int(floor(fy)) + 1
        tx = fx - (i - 1); ty = fy - (j - 1)
        (1 - tx) * (1 - ty) * T[j, i] + tx * (1 - ty) * T[j, i + 1] +
            (1 - tx) * ty * T[j + 1, i] + tx * ty * T[j + 1, i + 1]
    end
    eta = (x, y) -> eta_at(clamp(Tf(x, y), 0.0, 1.0), y, BLOBPR)[1]
    fz = (x, y) -> BLOBPR.Ra * (Tf(x, y) - Tbar(y, BLOBPR.H))
    return (; qt, ss, eta, fz, tag="blob-frame19", p=4)
end

function uniform_case(level::Int)
    qt = uniform_tree(level)
    ss = make_semstokes(qt, 4)
    eta = ONE
    fz = (x, y) -> sin(pi * x / qt.Lx) * sin(pi * y)
    return (; qt, ss, eta, fz, tag="uniform-L$level", p=4)
end

"""Parse `<variant>` into the keyword arguments of the production solve."""
function variant_kw(v::AbstractString)
    if v == "before"
        # The historical default: one tree layer per step and `:auto`, which
        # resolved to the taper on F/P1/blob and to uniform 2+2 on uniform
        # trees.  `before` pins the taper form; uniform cases use
        # `factor:1.0:unif:2`.
        return (; mg_coarsen_factor=1.0, mg_smoothing=:taper, label=v)
    elseif v == "after"
        return (; label=v)
    elseif startswith(v, "taper:")
        vec = parse.(Int, split(v[7:end], ','))
        return (; mg_coarsen_factor=2.0, mg_smoothing=:taper,
                 mg_pre=vec, mg_post=vec, label=v)
    elseif startswith(v, "unif:")
        n = parse(Int, v[6:end])
        return (; mg_coarsen_factor=2.0, mg_smoothing=:uniform,
                 mg_pre=n, mg_post=n, label=v)
    elseif startswith(v, "factor:")
        rest = split(v[8:end], ':')
        fac = parse(Float64, rest[1])
        if length(rest) == 1
            return (; mg_coarsen_factor=fac, mg_smoothing=:auto, label=v)
        elseif rest[2] == "unif"
            n = parse(Int, rest[3])
            return (; mg_coarsen_factor=fac, mg_smoothing=:uniform,
                     mg_pre=n, mg_post=n, label=v)
        elseif rest[2] == "taper"
            vec = parse.(Int, split(rest[3], ','))
            return (; mg_coarsen_factor=fac, mg_smoothing=:taper,
                     mg_pre=vec, mg_post=vec, label=v)
        end
        error("bad factor variant $v")
    end
    error("unknown variant $v")
end

const BASE_KW = (; bc=:free, deviatoric=false, mass_form=:boussinesq,
                 linear=:mg, mg_operator=:matrixfree,
                 mg_vel_solver=:vcycle,
                 mg_threads=parse(Int, get(ENV, "MG_BENCH_THR", "1")))

function run_case(case::AbstractString, variant::AbstractString, reps::Int)
    t_start = time()
    (; qt, ss, eta, fz, tag, p) = case in ("F", "P1") ?
        checkpoint_case(joinpath(REPO, "out_julia",
                                 case == "F" ? "_s7/case/pre_step6131.h5" :
                                               "s340/ab_none/resume.h5")) :
        case == "l5" ? uniform_case(5) :
        case == "l6" ? uniform_case(6) :
        case == "blob19" ? blob_case() :
        error("unknown case $case")
    vk = variant_kw(variant)
    # Production checkpoints are measured at the stated 1e-6 target; the
    # uniform/blob cases keep the historical 1e-10 protocol their documented
    # iteration counts were measured with.
    tol = parse(Float64, get(ENV, "MG_BENCH_TOL",
                             case in ("F", "P1") ? "1.0e-6" : "1.0e-10"))
    kw = merge(BASE_KW, (; mg_tol=tol,
               (k => v for (k, v) in pairs(vk) if k !== :label)...))
    c_factor = get(kw, :mg_coarsen_factor, 1.5)
    c_smooth = get(kw, :mg_smoothing, :auto)
    c_pre = get(kw, :mg_pre, 2)
    c_post = get(kw, :mg_post, 2)
    c_omega = get(kw, :mg_omega, nothing)
    c_smoother = get(kw, :mg_smoother, :jacobi)
    @printf("### case=%s state=%s leaves=%d levels=%d vdofs=%d variant=%s thr=%d\n",
            uppercase(case), tag, length(qt.leaves),
            length(unique(lf[1] for lf in qt.leaves)), 2 * ss.su.N,
            vk.label, BASE_KW.mg_threads)

    # Static reduced saddle blocks (the same ones the production solve builds).
    sys = solve_semquad_stokes_var(ss, qt, eta, Z2, fz, Z2, Z2;
                                   bc=:free, linear=:direct,
                                   assemble_only=true)
    nf = size(sys.Bs, 2); nfp = size(sys.Bs, 1)
    @printf("### nf=%d nfp=%d nftot=%d\n", nf, nfp, nf + nfp)

    # Direct reference: one untimed JIT call, then the measured one.  The
    # production direct path on the high-contrast AMR tree is not accurate
    # enough unshifted (measured 1.3e-3 on F), so the reference uses the
    # documented `:equilibrated` congruence; it is a diagnostic only.
    dscale = Symbol(get(ENV, "MG_BENCH_DIRECT_SCALING", "equilibrated"))
    solve_semquad_stokes_var(ss, qt, eta, Z2, fz, Z2, Z2; bc=:free,
                             linear=:direct, stokes_scaling=dscale)
    GC.gc()
    t_dir = @elapsed (ud, pd, _, _, _) =
        solve_semquad_stokes_var(ss, qt, eta, Z2, fz, Z2, Z2; bc=:free,
                                 linear=:direct, stokes_scaling=dscale)
    @printf("### direct=%.4f s scaling=%s res=%.2e\n", t_dir, dscale,
            LAST_SEMSTOKES_RESIDUAL[])

    # Warm-up solve of this variant (JIT / allocator), discarded.
    empty!(MS.SEMQUAD_MG_CACHE); MS.SEMQUAD_PRESSURE_CACHE[] = nothing
    solve_semquad_stokes_var(ss, qt, eta, Z2, fz, Z2, Z2; kw...)

    # Full production solve, cold cache each rep, best of `reps`.
    total = Inf; nit = 0; res = 0.0; u = nothing; pv = nothing
    for _ in 1:reps
        empty!(MS.SEMQUAD_MG_CACHE); MS.SEMQUAD_PRESSURE_CACHE[] = nothing
        GC.gc()
        t = @elapsed begin
            uu, pp, _, _, _ = solve_semquad_stokes_var(ss, qt, eta, Z2, fz, Z2,
                                                       Z2; kw...)
        end
        if t < total
            total = t; nit = LAST_SEMSTOKES_SCHUR_ITERS[]
            res = LAST_SEMSTOKES_RESIDUAL[]
            u = uu; pv = pp
        end
    end

    # Component accounting: cold hierarchy, cold w-BFBT, then the saddle solve.
    # Each is timed at its second call (same process, same arrays) so the JIT
    # and first-touch of the component itself are out of the number.
    local mg
    t_hier = Inf
    for _ in 1:2
        empty!(MS.SEMQUAD_MG_CACHE)
        t = @elapsed (mg = MS.make_semquad_mg(ss, qt, p, :free, false, eta,
                                              nothing; cycles=1,
                                              smoother=c_smoother,
                                              coarse_operator=:rediscretize,
                                              operator=:matrixfree,
                                              coarsen_layers=1,
                                              coarsen_factor=c_factor,
                                              coarse_direct_max=0,
                                              pre=c_pre, post=c_post,
                                              smoothing=c_smooth,
                                              omega=c_omega,
                                              mg_threads=kw.mg_threads,
                                              fine_Cu=sys.Cu))
        t_hier = min(t_hier, t)
    end
    mfop = mg.mfops[end]; mfetav = mg.mfetas[end]; mfwork = mg.mfwork[end]
    Cu = mg.geo.Cus[end]
    nt_outer = MS._semquad_mg_thread_count(kw.mg_threads)
    mfapplyK = if nt_outer > 1
        par = MS.SemQuadKPar(mfop, nt_outer)
        (out, v) -> MS._semquad_apply_Kred_threaded!(out, mfop, Cu, mfetav, v,
                                                     mfwork, par)
    else
        (out, v) -> MS._semquad_apply_Kred!(out, mfop, Cu, mfetav, v, mfwork)
    end
    zz = zeros(nf); mfapplyK(zz, ones(nf))
    local Fp, pscale
    t_bfbt = Inf
    for _ in 1:2
        t = @elapsed ((Fp, pscale) =
            MS.semquad_mg_pressure_precond(ss, sys.Cu, sys.Cp, sys.Bs, sys.Bt,
                                           nothing, eta; mode=:auto, scale=0.0,
                                           cache=false, applyK=mfapplyK))
        t_bfbt = min(t_bfbt, t)
    end
    rhs = vcat(sys.fr, sys.gr)
    local sol
    t_iter = Inf
    for _ in 1:2
        GC.gc()
        t = @elapsed (sol, it2, rel2, _) =
            MS.semquad_mg_saddle_solve(nothing, sys.Bs, sys.Bt, Fp, pscale, rhs,
                                       mg; tol=tol, maxit=500, restart=100,
                                       vel_solver=:vcycle, applyK=mfapplyK)
        t_iter = min(t_iter, t)
    end
    asm = total - t_hier - t_bfbt - t_iter
    per_it = 1000.0 * t_iter / max(nit, 1)
    du = norm(u - ud) / norm(ud)
    dp = norm(pv - pd) / norm(pd)
    pvg = pv .- sum(pv) / length(pv); pdg = pd .- sum(pd) / length(pd)
    dpg = norm(pvg - pdg) / norm(pdg)
    @printf("### hierarchy nlev=%d nfs=%s\n", mg.geo.L + 1,
            join(mg.geo.nfs, ","))
    @printf("### resolved smoothing=%s omega=%.4f pre=%s post=%s\n",
            MS.LAST_SEMSTOKES_MG_PROFILE[][1], mg.omega,
            join(mg.pre, ","), join(mg.post, ","))
    @printf("### setup hier=%.4f bfbt=%.4f asm_inferred=%.4f\n",
            t_hier, t_bfbt, asm)
    @printf("### iter=%.4f s per_it=%.2f ms nit=%d true_relres=%.3e\n",
            t_iter, per_it, nit, res)
    @printf("### acc du=%.3e dp=%.3e dpg=%.3e (direct res=%.2e)\n", du, dp,
            dpg, LAST_SEMSTOKES_RESIDUAL[])
    @printf("RESULT case=%s variant=%s thr=%d nlev=%d nfs=%s hier=%.4f bfbt=%.4f asm=%.4f total=%.4f iter=%.4f per_it=%.2f nit=%d res=%.3e du=%.3e dp=%.3e dpg=%.3e sched=%s omega=%.4f wall=%.1f\n",
            case, vk.label, BASE_KW.mg_threads, mg.geo.L + 1,
            join(mg.geo.nfs, ","), t_hier, t_bfbt, asm, total, t_iter, per_it,
            nit, res, du, dp, dpg, MS.LAST_SEMSTOKES_MG_PROFILE[][1], mg.omega,
            time() - t_start)
    return nothing
end

function main()
    length(ARGS) >= 2 || error("usage: bench_mg_defaults.jl <case> <variant> [reps]")
    reps = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 2
    run_case(ARGS[1], ARGS[2], reps)
end
main()
