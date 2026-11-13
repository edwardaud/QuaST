#!/usr/bin/env julia
# =============================================================================
# bench_auto_crossover.jl -- re-measure the `linear=:auto` crossover with the
# current code: same-session A/B of
#
#   direct : `linear=:direct, stokes_scaling=:equilibrated`  (the
#            tolerance-reaching reference the MG accuracy rows use)
#   mg     : `linear=:mg` with the shipped defaults (mg_operator=:matrixfree,
#            mg_vel_solver=:vcycle, mg_smoothing=:auto, w-BFBT pressure block,
#            mg_threads = min(8, Threads.nthreads()) as `:auto` sets it)
#
# on the seven states the stale selector table names (docs/dev/SOLVER.md
# §18.3/§19.2): uniform L4/L5/L6 (p=4, constant eta) and the blob frame-19
# trees (real eta, 16 384x bounds), plus the 25-frame blob trajectory and the
# two production checkpoints.
#
#   julia --project=julia --threads=8 \
#       julia/scripts/bench_auto_crossover.jl <state> [state...]
#
#   state = L2 | L3 | L4 | L5 | L6   unit box, uniform p=4, eta = 1,
#                                 fz = sin(pi x) sin(pi y)
#           blob19                frame-19 AMR tree, 1 024 leaves
#           blob19x4              every refinable frame-19 leaf refined once
#                                 (2 344 leaves)
#           blob19l7              frame-19 leaves in a maxlevel-7 tree, every
#                                 leaf refined once (4 096 leaves)
#           traj                  all 25 stored frames of the blob trajectory
#           traj<k>               frame k of that trajectory (traj14 = the
#                                 frame nearest the documented 639-leaf mean)
#           mid<L>                the same tree at p=4 with the smooth
#                                 stratification eta = exp(ln(50) y), contrast
#                                 exactly 50 (the 10 < contrast <= 100 band that
#                                 no other row samples)
#           L<k>p2                the same uniform tree at p = 2 (the degree of
#                                 the production states; 4x the leaves per saddle
#                                 dof, so the pessimistic side for MG)
#           F | P1                the two production checkpoints
#
# Every number is one process per state, both paths on the same tree and the
# same closure objects, `BLAS.set_num_threads(1)`; the tree/eta/forcing
# reconstruction of the blob states is identical to
# out_julia/agent/section18/bench_scale_case.jl (the instrument of the old
# table) and of julia/scripts/bench_mg_defaults.jl.  Cold clears the MG solver
# caches and calls `GC.gc()`; warm is three consecutive in-tree solves after
# the cold one, and the table lists the minima.  Trajectory frames use a
# lighter protocol (cold + 2 warm) because there are 25 of them.
#
# The checkpoint copies are read-only and nothing is written into the repo.
# =============================================================================

using MantleSEM, Printf, LinearAlgebra, SparseArrays, Random

const MS = MantleSEM
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const Z2 = (x, y) -> 0.0
const ONE = (x, y) -> 1.0
BLAS.set_num_threads(1)

const MGTOL = parse(Float64, get(ENV, "BENCH_MGTOL", "1.0e-6"))
const MGMAXIT = 500
const BLOBREF = normpath(joinpath(REPO, "out_julia", "fresh_blob_kelly_1024"))
const BLOBPR = Params(Ra=1.0e8, H=5.0, eta_law=:expTz,
                      eta_b=log(16384.0), eta_c=log(64.0), bc=:free)

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
    return (; qt, ss, eta, fz, tag=basename(path), p, step=st.step, tmpdir,
            saddle_note=(length(qt.leaves), length(unique(lf[1] for lf in qt.leaves))))
end

function uniform_tree(level::Int)
    qt = Quadtree(level)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

function blob_leaves(knot::Int; level::Int=6)
    lvs = Set{NTuple{3,Int}}()
    for ln in eachline(joinpath(BLOBREF, "frame_$(knot)_mesh.csv"))
        isempty(strip(ln)) && continue
        a = split(ln, ',')
        l = parse(Int, a[1]); h = 2.0^(-l)
        push!(lvs, (l, round(Int, parse(Float64, a[2]) / h),
                    round(Int, parse(Float64, a[3]) / h)))
    end
    return Quadtree(level, 1.0, 1, 1, lvs)
end

function blob_T(knot::Int)
    T = Array{Float64}(undef, 256, 256)
    open(joinpath(BLOBREF, "frame_$(knot)_T.bin"), "r") do io
        read!(io, T)
    end
    return T
end

function blob_Tfun(T::Matrix{Float64})
    h = 1.0 / (size(T, 1) - 1); n = size(T, 1)
    return (x, y) -> begin
        fx = clamp(x / h, 0.0, n - 1.0000001)
        fy = clamp(y / h, 0.0, n - 1.0000001)
        i = Int(floor(fx)) + 1; j = Int(floor(fy)) + 1
        tx = fx - (i - 1); ty = fy - (j - 1)
        (1 - tx) * (1 - ty) * T[j, i] + tx * (1 - ty) * T[j, i + 1] +
            (1 - tx) * ty * T[j + 1, i] + tx * ty * T[j + 1, i + 1]
    end
end

"""Build one named state: (; qt, ss, eta, fz, tag, tmpdir)."""
function build_state(name::AbstractString)
    lname = lowercase(name)
    if lname in ("l2", "l3", "l4", "l5", "l6")
        L = parse(Int, lname[2:end])
        qt = uniform_tree(L)
        ss = make_semstokes(qt, 4)
        fz = (x, y) -> sin(pi * x / qt.Lx) * sin(pi * y)
        return (; qt, ss, eta=ONE, fz, tag="uniform-L$L-p4-const-eta", tmpdir="")
    elseif startswith(lname, "l") && endswith(lname, "p2")
        # the same uniform family at p = 2, the degree of the production states:
        # MG costs per *leaf*, direct per *dof*, so a p=2 tree has 4x the leaves
        # of a p=4 tree at the same saddle size and is the pessimistic side
        L = parse(Int, lname[2:(end - 2)])
        qt = uniform_tree(L)
        ss = make_semstokes(qt, 2)
        fz = (x, y) -> sin(pi * x / qt.Lx) * sin(pi * y)
        return (; qt, ss, eta=ONE, fz, tag="uniform-L$L-p2-const-eta", tmpdir="")
    elseif startswith(lname, "mid")
        # synthetic mid-contrast band (10 < contrast <= 100): a uniform p=4 tree
        # with the smooth stratification eta = exp(ln(50) y), contrast exactly 50
        # (the `linear=:auto` band that neither the uniform nor the blob rows
        # sample)
        L = parse(Int, lstrip(lname[4:end], 'l'))
        qt = uniform_tree(L)
        ss = make_semstokes(qt, 4)
        c = log(50.0)
        eta = (x, y) -> exp(c * y)
        fz = (x, y) -> sin(pi * x / qt.Lx) * sin(pi * y)
        return (; qt, ss, eta, fz, tag="mid50-L$L-p4-stratified", tmpdir="")
    elseif lname in ("blob19", "blob19x4", "blob19l7")
        if lname == "blob19l7"
            # frame-19 leaves in a maxlevel-7 tree, every leaf refined once:
            # exactly 4 096 leaves (the old table's f19x16 row)
            qt = blob_leaves(19; level=7)
            for lf in collect(qt.leaves)
                refine!(qt, lf)
            end
            balance!(qt)
        else
            qt = blob_leaves(19; level=6)
            if lname == "blob19x4"
                # every *refinable* leaf once (maxlevel 6 keeps the L6 leaves):
                # 1 024 -> 2 344 leaves (the old table's f19x4 row)
                for lf in collect(qt.leaves)
                    refine!(qt, lf)
                end
                balance!(qt)
            end
        end
        ss = make_semstokes(qt, 4)
        Tf = blob_Tfun(blob_T(19))
        eta = (x, y) -> MantleSEM.eta_at(clamp(Tf(x, y), 0.0, 1.0), y, BLOBPR)[1]
        fz = (x, y) -> BLOBPR.Ra * (Tf(x, y) - Tbar(y, BLOBPR.H))
        return (; qt, ss, eta, fz, tag="blob-frame19-" * lname, tmpdir="")
    elseif startswith(lname, "traj")
        knot = parse(Int, lname[5:end])
        qt = blob_leaves(knot; level=6)
        ss = make_semstokes(qt, 4)
        Tf = blob_Tfun(blob_T(knot))
        eta = (x, y) -> MantleSEM.eta_at(clamp(Tf(x, y), 0.0, 1.0), y, BLOBPR)[1]
        fz = (x, y) -> BLOBPR.Ra * (Tf(x, y) - Tbar(y, BLOBPR.H))
        return (; qt, ss, eta, fz, tag="blob-traj-f$knot", tmpdir="")
    elseif lname == "f"
        return checkpoint_case(joinpath(REPO, "out_julia", "_s7", "case",
                                        "pre_step6131.h5"))
    elseif lname == "p1"
        return checkpoint_case(joinpath(REPO, "out_julia", "s340", "ab_none",
                                        "resume.h5"))
    end
    error("unknown state $name")
end

mlist(; cold::Bool) = cold ? (empty!(MS.SEMQUAD_MG_CACHE);
                              MS.SEMQUAD_PRESSURE_CACHE[] = nothing; GC.gc()) :
                            GC.gc()

"""One state: cold + warm A/B, MG setup decomposition, accuracy."""
function measure(name::AbstractString; reps::Int=3)
    (; qt, ss, eta, fz, tag, tmpdir) = build_state(name)
    thr = Threads.nthreads()
    mgthr = MS._semquad_mg_thread_count(thr)
    nf = 2 * ss.su.N
    sad = nf + ss.sp.N
    contrast = semquad_eta_contrast(ss.su, eta)

    dkw = (; bc=:free, deviatoric=false, mass_form=:boussinesq,
             linear=:direct, stokes_scaling=:equilibrated)
    mkw = (; bc=:free, deviatoric=false, mass_form=:boussinesq,
             linear=:mg, mg_operator=:matrixfree, mg_vel_solver=:vcycle,
             mg_threads=mgthr, mg_tol=MGTOL, mg_maxit=MGMAXIT)
    solve_direct() = solve_semquad_stokes_var(ss, qt, eta, Z2, fz, Z2, Z2; dkw...)
    solve_mg() = solve_semquad_stokes_var(ss, qt, eta, Z2, fz, Z2, Z2; mkw...)

    # JIT warm-up on a level-3 uniform p=4 tree (never measured), both paths.
    let qtw = uniform_tree(3), ssw = make_semstokes(qtw, 4)
        fzw = (x, y) -> sin(pi * x / qtw.Lx) * sin(pi * y)
        solve_semquad_stokes_var(ssw, qtw, ONE, Z2, fzw, Z2, Z2; dkw...)
        solve_semquad_stokes_var(ssw, qtw, ONE, Z2, fzw, Z2, Z2; mkw...)
    end
    GC.gc()

    # Cold solves, alternating the two paths.  "Cold" clears the MG solver
    # caches (the per-level hierarchy cache and the pressure-preconditioner
    # cache), so it pays hierarchy + pressure setup; "warm" is a consecutive
    # in-tree solve that reuses both, exactly as a production sequence does.
    cold_d = Float64[]; cold_m = Float64[]
    du = dp = dpg = NaN; nit = 0; res_m = NaN; res_d = NaN
    for _ in 1:reps
        mlist(; cold=true)
        push!(cold_d, @elapsed (ud, pd, _, _, _) = solve_direct())
        res_d = LAST_SEMSTOKES_RESIDUAL[]
        mlist(; cold=true)
        push!(cold_m, @elapsed (um, pm, _, _, _) = solve_mg())
        nit = LAST_SEMSTOKES_SCHUR_ITERS[]
        res_m = LAST_SEMSTOKES_RESIDUAL[]
        du = norm(um - ud) / max(norm(ud), 1.0)
        dp = norm(pm - pd) / max(norm(pd), 1.0)
        udg = ud .- sum(ud) / length(ud)
        pdg = pd .- sum(pd) / length(pd)
        dpg = norm((pm .- sum(pm) / length(pm)) - pdg) / norm(pdg)
    end
    # Warm: consecutive in-tree solves, no cache clear, no GC between them.
    warm_d = Float64[]; warm_m = Float64[]
    for _ in 1:3
        push!(warm_d, @elapsed solve_direct())
        push!(warm_m, @elapsed solve_mg())
    end
    med(v) = sort(v)[cld(length(v), 2)]
    wd = minimum(warm_d); wm = minimum(warm_m)
    cd = minimum(cold_d); cm = minimum(cold_m)
    t_setup = cm - wm
    @printf("### state=%s tag=%s leaves=%d levels=%d vdof=%d saddle=%d contrast=%.1f thr=%d mg_thr=%d tol=%.0e\n",
            name, tag, length(qt.leaves),
            length(unique(lf[1] for lf in qt.leaves)), nf, sad, contrast, thr,
            mgthr, MGTOL)
    @printf("### mg setup (cold-warm)=%.4f s  share_of_cold=%.1f%%  share_of_cold_med=%.1f%%\n",
            t_setup, 100 * t_setup / max(cm, 1.0e-12),
            100 * (med(cold_m) - med(warm_m)) / max(med(cold_m), 1.0e-12))
    @printf("### direct cold=%.4f/%.4f warm=%.4f/%.4f res=%.2e | mg cold=%.4f/%.4f warm=%.4f/%.4f res=%.2e nit=%d\n",
            cd, med(cold_d), wd, med(warm_d), res_d, cm, med(cold_m), wm,
            med(warm_m), res_m, nit)
    @printf("### acc du=%.3e dp=%.3e dpg=%.3e\n", du, dp, dpg)
    @printf("RESULT state=%s tag=%s leaves=%d levels=%d vdof=%d saddle=%d contrast=%.1f thr=%d mg_thr=%d cold_d=%.4f cold_m=%.4f warm_d=%.4f warm_m=%.4f ratio_cold=%.3f ratio_warm=%.3f nit=%d setup_m=%.4f res_d=%.2e res_m=%.2e du=%.3e dp=%.3e dpg=%.3e\n",
            name, tag, length(qt.leaves),
            length(unique(lf[1] for lf in qt.leaves)), nf, sad, contrast, thr,
            mgthr, cd, cm, wd, wm, cd / cm, wd / wm, nit, t_setup, res_d,
            res_m, du, dp, dpg)
    flush(stdout)
    isempty(tmpdir) || rm(tmpdir; force=true, recursive=true)
    return nothing
end

"""All 25 stored trajectory frames: per-frame warm A/B + the run mean."""
function measure_traj(; reps::Int=2)
    thr = Threads.nthreads()
    mgthr = MS._semquad_mg_thread_count(thr)
    ratios = Float64[]
    rows = Any[]
    tot_d = 0.0; tot_m = 0.0
    for knot in 1:25
        (; qt, ss, eta, fz) = build_state("traj$knot")
        dkw = (; bc=:free, deviatoric=false, mass_form=:boussinesq,
                 linear=:direct, stokes_scaling=:equilibrated)
        mkw = (; bc=:free, deviatoric=false, mass_form=:boussinesq,
                 linear=:mg, mg_operator=:matrixfree, mg_vel_solver=:vcycle,
                 mg_threads=mgthr, mg_tol=MGTOL, mg_maxit=MGMAXIT)
        sd() = solve_semquad_stokes_var(ss, qt, eta, Z2, fz, Z2, Z2; dkw...)
        sm() = solve_semquad_stokes_var(ss, qt, eta, Z2, fz, Z2, Z2; mkw...)
        sd(); sm(); GC.gc()                      # JIT/allocator warm
        wd = Float64[]; wm = Float64[]; nit = 0
        for _ in 1:reps
            push!(wd, @elapsed sd())
            push!(wm, @elapsed sm())
            nit = LAST_SEMSTOKES_SCHUR_ITERS[]
        end
        q = minimum(wm) / minimum(wd)          # literal MG/direct time ratio
        push!(ratios, 1.0 / q)                 # table convention: >1 = MG faster
        tot_d += minimum(wd); tot_m += minimum(wm)
        push!(rows, (knot=knot, leaves=length(qt.leaves), sad=2 * ss.su.N + ss.sp.N,
                     wd=min(wd...), wm=min(wm...), nit=nit))
        @printf("TRAJ frame=%2d leaves=%4d saddle=%6d warm_d=%.4f warm_m=%.4f ratio_warm=%.3f nit=%d\n",
                knot, rows[end].leaves, rows[end].sad, rows[end].wd,
                rows[end].wm, 1.0 / q, nit)
        flush(stdout)
    end
    m = sum(ratios) / length(ratios)
    g = exp(sum(log.(ratios)) / length(ratios))
    @printf("TRAJMEAN frames=%d thr=%d mg_thr=%d mean_ratio_warm=%.4f geo=%.4f min=%.4f max=%.4f timew_ratio_warm=%.4f\n",
            length(ratios), thr, mgthr, m, g, minimum(ratios), maximum(ratios),
            tot_d / tot_m)
    flush(stdout)
    return nothing
end

function main()
    isempty(ARGS) && error("usage: bench_auto_crossover.jl <state> [state...]")
    for a in ARGS
        if lowercase(a) == "traj"
            measure_traj()
        else
            measure(a)
        end
    end
end
main()
