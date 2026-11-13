#!/usr/bin/env julia
# =============================================================================
# bench_mg_level_cache.jl -- the per-level MG hierarchy cache (G221) on a real
# AMR trajectory: levels rebuilt vs reused per cycle, setup time with and
# without reuse, the end-to-end step, and the bitwise A/B.
#
#   BENCH_CASE=F julia --project=julia --threads=8 \
#       julia/scripts/bench_mg_level_cache.jl
#
# What it runs.  Two passes over the *same* 30-step trajectory from a production
# checkpoint, in one process, on a read-only copy of the file:
#
#   pass A  SEMQUAD_MG_LEVEL_CACHE[] = true   -- only the levels whose leaf set
#           changed are rebuilt after each AMR cycle;
#   pass B  SEMQUAD_MG_LEVEL_CACHE[] = false  -- every level is rebuilt on every
#           build (the historical behaviour, the forced from-scratch reference).
#
# Both passes start from `load_semquad_state` of the same copy with all caches
# emptied, run the production driver (`run_semquad_simulation`) with
# `amr_every` taken from the checkpoint's own config (5 on `F`/`P1`), and record
# per step: leaves, the hierarchy levels rebuilt/reused, the `make_semquad_mg`
# wall time (`SEMQUAD_MG_LEVEL_STATS.last_build_s`), the saddle solve's
# iteration count and true residual, the step wall, and a full copy of
# `theta`/`ux`/`uz` at every step that follows an AMR cycle.  Those copies are
# compared with `==`, so the evidence is bitwise, not a tolerance.
#
# Printed: one `CYCLE` line per AMR cycle (the build at the first step after the
# cycle, which is the only cold-tree build of that cycle), one `PASS` line per
# pass, one `BITWISE` line per compared step, and a `SUMMARY` block.
#
# Environment:
#   BENCH_CASE  = F | P1      production checkpoint (default F)
#   BENCH_CKPT  = path        override the checkpoint path
#   BENCH_THR   = n           mg_threads (default 8); start Julia with --threads=n
#   BENCH_STEPS = n           steps to run from the checkpoint (default 30)
#   BENCH_TOL   = x           MG saddle tolerance (default 1e-6)
#   BENCH_RA    = x           Rayleigh number of the card (default 1e5; the
#                             checkpoints do not store Ra, every other physics
#                             parameter is rebuilt from the checkpoint's config)
#
# The checkpoint is copied to a temp directory, `chmod 444`ed and opened
# read-only; nothing is written into the repository.
# =============================================================================

using MantleSEM, Printf, LinearAlgebra, SparseArrays
const MS = MantleSEM
BLAS.set_num_threads(1)

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const CASE = get(ENV, "BENCH_CASE", "F")
const THR = parse(Int, get(ENV, "BENCH_THR", "8"))
const NSTEPS = parse(Int, get(ENV, "BENCH_STEPS", "30"))
const TOL = parse(Float64, get(ENV, "BENCH_TOL", "1.0e-6"))
const RA = parse(Float64, get(ENV, "BENCH_RA", "1.0e5"))
const CKPT = get(ENV, "BENCH_CKPT",
                 CASE == "F" ?
                 joinpath(REPO, "out_julia", "_s7", "case", "pre_step6131.h5") :
                 joinpath(REPO, "out_julia", "s340", "ab_none", "resume.h5"))

# ---- the card, rebuilt from the checkpoint's own `config` --------------------
function params_from_config(cfg::Dict{String,String})
    g(k, d) = haskey(cfg, k) ? parse(Float64, cfg[k]) : d
    s(k, d) = haskey(cfg, k) ? Symbol(cfg[k]) : d
    b(k, d) = haskey(cfg, k) ? cfg[k] == "true" : d
    return Params(Ra=RA, H=g("H", 10.0),
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
                  weak_heal_time=g("weak_heal_time", Inf),
                  weak_mode=s("weak_mode", :rate),
                  rho_heating=b("rho_heating", false), rho_cp=b("rho_cp", false),
                  Di=g("Di", 0.0),
                  heating_adiabatic=b("heating_adiabatic", false),
                  heating_dissipation=b("heating_dissipation", false),
                  refstate=nothing, deviatoric=b("deviatoric", false))
end

function phases_from_config(cfg::Dict{String,String}, pr::Params)
    haskey(cfg, "phases") || return PhaseTransition[]
    cfg["phases"] == "none" && return PhaseTransition[]
    out = PhaseTransition[]
    for row in split(cfg["phases"], ';')
        a = parse.(Float64, split(row, ','))
        push!(out, PhaseTransition(a[1], a[2], a[3], a[4], a[5], a[6]))
    end
    # The checkpoint stores `latent_mode` and the six phase numbers, not the
    # per-phase coefficients: `case_realistic_2d.jl` derives them from the
    # Clapeyron slope and the density jump at its ASPECT scale set
    # (docs/RHEOLOGY.md §7.8/§14.2), so a resume has to use the same formula.
    if pr.latent_mode === :per_phase && !isempty(out) && out[1].latent === nothing
        out = per_phase_latent(pr, out; rho_ref=3300.0, cp=1000.0, dT=1000.0,
                               g=9.81, d=2.9e6, T_surface=0.5)
    end
    return out
end

# `run_semquad_simulation` validates its `run_config` against the checkpoint on
# restart, so every trajectory-affecting setting is taken from the file.
# `coarsen_theta_tol` is an exception: it is *not* in `run_config`, and the card
# that produced these checkpoints ran with the documented 1e-2 (docs/
# CASE_REALISTIC.md §"E is the classic-policy run"); without it the classic
# coarsening strip-mines the tree (8188 -> 4810 leaves in one cycle) and the
# trajectory is not the production one.
function driver_kwargs(cfg::Dict{String,String}; amr_every::Int,
                       coarsen_theta_tol::Float64=1.0e-2)
    g(k, d) = haskey(cfg, k) ? parse(Float64, cfg[k]) : d
    i(k, d) = Int(round(g(k, Float64(d))))
    s(k, d) = haskey(cfg, k) ? Symbol(cfg[k]) : d
    b(k, d) = haskey(cfg, k) ? cfg[k] == "true" : d
    return (amr_every=amr_every, amr_strategy=s("amr_strategy", :legacy),
            amr_passes=i("amr_passes", 1), marking=s("marking", :error_fraction),
            refine_frac=g("refine_frac", 0.2), coarsen_frac=g("coarsen_frac", 0.0),
            coarsen_theta_tol=coarsen_theta_tol,
            max_leaves=i("max_leaves", typemax(Int)),
            max_level=i("max_level", typemax(Int)),
            wall_min_level=i("wall_min_level", 0), wall_strip=g("wall_strip", 0.1),
            symmetry=s("symmetry", :none), transfer_mode=s("transfer_mode", :nodal),
            hotspot_weight=g("hotspot_weight", 0.0), hotspot_frac=g("hotspot_frac", 0.5),
            hotspot_min_level=i("hotspot_min_level", 0),
            hotspot_mode=s("hotspot_mode", :theta),
            hotspot_zmin=g("hotspot_zmin", 0.0),
            zero_side_eta=b("zero_side_eta", false), side_strip=g("side_strip", 0.05),
            top_wall_level=i("top_wall_level", 0),
            top_wall_frac=g("top_wall_frac", 0.5),
            advection=s("advection", :galerkin), ev=g("ev", 0.0),
            ev_cmax=g("ev_cmax", 0.052), ev_alpha=i("ev_alpha", 2),
            maxpic=i("maxpic", 1), pic_tol=g("pic_tol", 1.0e-6),
            pic_relax=g("pic_relax", 1.0), pic_mode=s("pic_mode", :picard),
            pic_reject=b("pic_reject", false),
            cfl=g("cfl", 0.0), dtmax=g("dtmax", Inf), dtmin=g("dtmin", 1.0e-14),
            mass_form=s("mass_form", :anelastic),
            latent_scheme=s("latent_scheme", :implicit))
end

# ---- one pass over the trajectory ------------------------------------------
function run_pass(copyto::String, reuse::Bool; label::String, steps::Int=NSTEPS)
    MS.SEMQUAD_MG_LEVEL_CACHE[] = reuse
    empty!(MS.SEMQUAD_MG_CACHE)
    MS.SEMQUAD_PRESSURE_CACHE[] = nothing
    MS.reset_semquad_mg_level_stats!()
    st = load_semquad_state(copyto)
    pr = params_from_config(st.config)
    ph = phases_from_config(st.config, pr)
    step0 = st.step
    amr = haskey(st.config, "amr_every") ?
          Int(round(parse(Float64, st.config["amr_every"]))) : 0
    kw = driver_kwargs(st.config; amr_every=amr)
    rows = NamedTuple[]
    snaps = Dict{Int,NamedTuple}()
    tprev = Ref(time())
    cb = function (s)
        t = time() - tprev[]
        tprev[] = time()
        sst = MS.semquad_mg_level_cache_stats()
        push!(rows, (step=s.step, t=s.t, leaves=length(s.qt.leaves),
                     levels=length(unique(lf[1] for lf in s.qt.leaves)),
                     fhash=MS._semquad_mg_leaf_hash(s.qt.leaves),
                     build_s=sst.last_build_s, ps=sst.prolongations,
                     rebuilt=copy(sst.last_rebuilt), reused=copy(sst.last_reused),
                     hits=sst.hits, misses=sst.misses, calls=sst.builds,
                     nit=LAST_SEMSTOKES_SCHUR_ITERS[],
                     relres=LAST_SEMSTOKES_RESIDUAL[], wall_s=t))
        if (s.step - step0 - 1) % amr == 0 || s.step == step0 + steps
            snaps[s.step] = (theta=copy(s.theta), ux=copy(s.ux), uz=copy(s.uz),
                             nit=LAST_SEMSTOKES_SCHUR_ITERS[],
                             relres=LAST_SEMSTOKES_RESIDUAL[])
        end
        return nothing
    end
    t0 = time()
    res = run_semquad_simulation(st.qt, pr, st.theta; restart_from=copyto,
                                 tstop=Inf, maxsteps=step0 + steps,
                                 dt0=st.dt, linear=:mg, mg_operator=:matrixfree,
                                 mg_vel_solver=:vcycle, mg_threads=THR,
                                 mg_tol=TOL, verbose=false, callback=cb,
                                 phases=ph, kw...)
    wall = time() - t0
    stats = MS.semquad_mg_level_cache_stats()
    @printf("PASS %s reuse=%d step_end=%d wall=%.2f s builds=%d hits=%d misses=%d hitrate=%.3f total_build_s=%.3f cached_levels=%d bytes=%.2f MB\n",
            label, reuse ? 1 : 0, res.steps, wall, stats.builds, stats.hits,
            stats.misses, stats.hits / max(stats.hits + stats.misses, 1),
            stats.total_build_s, stats.cached_levels, stats.bytes / 2.0^20)
    return (rows=rows, snaps=snaps, wall=wall, stats=stats, res=res,
            step0=step0, amr=amr)
end

# ---- the report ------------------------------------------------------------
function report(A, B)
    @printf("\n%-6s %6s %7s %8s %10s  %-12s %-12s %9s %9s %8s %8s %8s %5s\n",
            "cycle", "step", "leaves", "levels", "tree-hash", "rebuilt",
            "reused", "buildA_s", "buildB_s", "save_ms", "stepA_s", "stepB_s",
            "dP")
    ncyc = 0
    psprev = Ref(0)
    for (ra, rb) in zip(A.rows, B.rows)
        (ra.step == rb.step) || error("pass A/B step misalignment")
        isfirst = ra.step > A.step0 + 1 && (ra.step - A.step0 - 1) % A.amr == 0
        isfirst || continue
        ncyc += 1
        dP = ra.ps - psprev[]
        psprev[] = ra.ps
        @printf("%-6d %6d %7d %8d %10s  %-12s %-12s %9.4f %9.4f %8.1f %8.4f %8.4f %5d\n",
                ncyc, ra.step, ra.leaves, ra.levels,
                string(ra.fhash % 0x100000000; base=16), string(ra.rebuilt),
                string(ra.reused), ra.build_s, rb.build_s,
                1000 * (rb.build_s - ra.build_s), ra.wall_s, rb.wall_s, dP)
    end
    cycles = ncyc

    @printf("\n")
    nbit = 0
    okbit = true
    for step in sort(collect(keys(A.snaps)))
        sa = A.snaps[step]; sb = B.snaps[step]
        eq = sa.theta == sb.theta && sa.ux == sb.ux && sa.uz == sb.uz &&
             sa.nit == sb.nit && sa.relres == sb.relres
        okbit &= eq
        nbit += 1
        @printf("BITWISE step=%d equal=%s nitA=%d nitB=%d relresA=%.6e relresB=%.6e\n",
                step, eq, sa.nit, sb.nit, sa.relres, sb.relres)
    end
    @printf("BITWISE all=%s compared=%d\n", okbit, nbit)

    medA = sort([r.wall_s for r in A.rows])[cld(length(A.rows), 2)]
    medB = sort([r.wall_s for r in B.rows])[cld(length(B.rows), 2)]
    sumA = sum(r.build_s for r in A.rows)
    sumB = sum(r.build_s for r in B.rows)
    @printf("SUMMARY case=%s thr=%d steps=%d cycles=%d bitwise=%s build_sum_A=%.3f build_sum_B=%.3f build_total_A=%.3f build_total_B=%.3f build_total_save=%.3f wall_A=%.2f wall_B=%.2f step_median_A=%.4f step_median_B=%.4f hits_A=%d misses_A=%d hitrate_A=%.3f bytes_A=%.1fMB cached_A=%d\n",
            CASE, THR, NSTEPS, cycles, okbit, sumA, sumB, A.stats.total_build_s,
            B.stats.total_build_s, B.stats.total_build_s - A.stats.total_build_s,
            A.wall, B.wall, medA, medB, A.stats.hits, A.stats.misses,
            A.stats.hits / max(A.stats.hits + A.stats.misses, 1),
            A.stats.bytes / 2.0^20, A.stats.cached_levels)
    return okbit
end

# ---- main ------------------------------------------------------------------
# `BENCH_MAXLEAVES` rewrites the seed copy's leaf budget.  Both production
# checkpoints sit *at* their budget (`F`: 8188 of 8192 leaves, `P1`: 10 000 of
# 10 000), where the classic policy sets `refine_frac=0` and the budget
# enforcement pins the tree: the AMR cycles then change nothing at all.  A
# slightly larger budget puts the run back into the regime the card was designed
# for, where every cycle refines and coarsens a few per cent of the leaves --
# which is the case this cache exists for.  Only that one string in the seed
# copy's `config` changes; the state (tree/theta/ux/uz/pv/dt/t/step) is copied
# verbatim, and the restart guard still compares everything else.
const MAXLEAVES = parse(Int, get(ENV, "BENCH_MAXLEAVES", "0"))

function seed_copy(src::String, dst::String; max_leaves::Int=0)
    copyto = joinpath(dirname(dst), basename(src))
    cp(src, copyto; force=true)
    if max_leaves > 0
        st = load_semquad_state(copyto)
        cfg = copy(st.config)
        cfg["max_leaves"] = string(Float64(max_leaves))
        save_semquad_state(copyto, st.qt, st.theta, st.theta_prev, st.ux, st.uz,
                           st.pv; t=st.t, dt=st.dt, dt_prev=st.dt_prev,
                           step=st.step, p=st.p, config=cfg,
                           meta=Dict("seed_from" => src,
                                     "max_leaves" => string(max_leaves)),
                           cooldown=st.cooldown, ev=st.ev,
                           velocity_stale=st.velocity_stale)
        @printf("## seed: max_leaves %s -> %d\n",
                get(st.config, "max_leaves", "?"), max_leaves)
    end
    chmod(copyto, 0o444)
    return copyto
end

tmpdir = mktempdir()
copyto = seed_copy(CKPT, joinpath(tmpdir, basename(CKPT)); max_leaves=MAXLEAVES)
st0 = load_semquad_state(copyto)
amr0 = haskey(st0.config, "amr_every") ?
       Int(round(parse(Float64, st0.config["amr_every"]))) : 0
@printf("## case=%s thr=%d ckpt_step=%d leaves=%d p=%d amr_every=%d steps=%d tol=%.1e Ra=%.1e\n",
        CASE, THR, st0.step, length(st0.qt.leaves), st0.p, amr0, NSTEPS, TOL, RA)

# JIT: a short pass (2 steps) through each branch *with the same callback
# closure type* the measured passes use, so neither measured pass pays
# compilation and the two timings differ only by the cache.
const ORDER = get(ENV, "BENCH_ORDER", "AB")   # "AB" or "BA": which pass runs first
if ORDER == "BA"
    run_pass(copyto, false; label="JIT-B", steps=2)
    run_pass(copyto, true; label="JIT-A", steps=2)
    B = run_pass(copyto, false; label="B")
    A = run_pass(copyto, true; label="A")
else
    run_pass(copyto, true; label="JIT-A", steps=2)
    run_pass(copyto, false; label="JIT-B", steps=2)
    A = run_pass(copyto, true; label="A")
    B = run_pass(copyto, false; label="B")
end
report(A, B)

MS.SEMQUAD_MG_LEVEL_CACHE[] = true
rm(tmpdir; force=true, recursive=true)
