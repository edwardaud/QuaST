#!/usr/bin/env julia
# =============================================================================
# bench_mg_reduced_assembly.jl -- setup and solve cost of the matrix-free MG
# path, for the before/after A/B of docs/dev/MG.md
#
#   BENCH_CASE=F BENCH_THR=1 BENCH_NEW=1 julia --project=julia --threads=1 \
#       julia/scripts/bench_mg_reduced_assembly.jl
#
# `BENCH_NEW=0` runs the same script against the pre-change tree (it then skips
# the reduced-assembly measurements, which do not exist there), so one command
# measures both sides of the A/B.  Each row is one process: the checkpoint is
# solved `BENCH_REPS` times with empty caches (the row quotes the best), the
# hierarchy and w-BFBT builds are timed separately (best of 2), and the saddle
# solve is timed on its own so the "assembly" row can be the remainder
#
#     t_asm = t_full - t_hierarchy - t_bfbt - t_iteration
#
# like docs/dev/MG.md and docs/dev/MG.md  One `RESULT key=value`
# line is printed.  The checkpoint is copied to a temp directory, `chmod 444`ed
# and opened read-only; the script writes nothing into the repository.
#
# Environment:
#   BENCH_CASE = F | P1   production checkpoint (default F)
#   BENCH_CKPT = path     override the checkpoint path
#   BENCH_THR  = n        mg_threads (default 1); start Julia with --threads=n
#   BENCH_NEW  = 0 | 1    pre-change tree (0) or the reduced assembly (1)
#   BENCH_REPS = n        cold solves per row (default 3)
# =============================================================================

using MantleSEM, Printf, LinearAlgebra, SparseArrays
const MS = MantleSEM
BLAS.set_num_threads(1)
const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const CASE = get(ENV, "BENCH_CASE", "F")
const THR = parse(Int, get(ENV, "BENCH_THR", "1"))
const NEW = get(ENV, "BENCH_NEW", "0") == "1"
const REPS = parse(Int, get(ENV, "BENCH_REPS", "3"))
const CKPT = get(ENV, "BENCH_CKPT",
                 CASE == "F" ?
                 joinpath(REPO, "out_julia", "_s7", "case", "pre_step6131.h5") :
                 joinpath(REPO, "out_julia", "s340", "ab_none", "resume.h5"))
const Z2 = (x, y) -> 0.0
const ONE = (x, y) -> 1.0

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
@printf("## case=%s thr=%d leaves=%d nlev=%d vdofs=%d new=%d\n", CASE, THR,
        length(qt.leaves), length(unique(lf[1] for lf in qt.leaves)),
        2 * ss.su.N, NEW)

solve() = solve_semquad_stokes_var(ss, qt, eta, Z2, fz, Z2, Z2; kw...)
solve()                                                    # JIT warm
sys = solve_semquad_stokes_var(ss, qt, eta, Z2, fz, Z2, Z2; bc=:free,
                               linear=:direct, assemble_only=true)
Cu = SparseMatrixCSC{Float64,Int}(sparse(sys.Cu))
op = semquad_kop(ss, qt; deviatoric=false)
ev = semquad_eta_field(op, eta)
wk = SemQuadKWork(op.n, op.nq, 2 * op.N)
mfapplyK = (out, v) -> MS._semquad_apply_Kred!(out, op, Cu, ev, v, wk)

function coldsolve()
    empty!(MS.SEMQUAD_MG_CACHE); MS.SEMQUAD_PRESSURE_CACHE[] = nothing
    GC.gc()
    @elapsed solve()
end
t_full = minimum([coldsolve() for _ in 1:REPS])
nit = LAST_SEMSTOKES_SCHUR_ITERS[]
GC.gc()
t_warm = @elapsed solve()          # same tree, same eta: the in-block step

function buildmg(; geom=:minimal)
    empty!(MS.SEMQUAD_MG_CACHE)
    extra = NEW ? (; geom_assembly=geom) : (;)
    fine = NEW ? (; fine_mfop=op, fine_mfeta=ev) : (;)
    MS.make_semquad_mg(ss, qt, p, :free, false, eta, nothing; cycles=1,
                       smoother=:jacobi, coarse_operator=:rediscretize,
                       operator=:matrixfree, coarsen_factor=1.5,
                       smoothing=:auto, mg_threads=THR,
                       fine_Cu=sys.Cu, extra..., fine...)
end
t_hier = minimum([begin GC.gc(); @elapsed buildmg() end for _ in 1:2])
t_hier_full = NEW ? minimum([begin GC.gc(); @elapsed buildmg(geom=:full) end
                             for _ in 1:2]) : NaN

function buildfp()
    MS.semquad_mg_pressure_precond(ss, sys.Cu, sys.Cp, sys.Bs, sys.Bt, nothing,
                                   eta; mode=:auto, cache=false,
                                   applyK=mfapplyK)
end
t_fp = minimum([begin GC.gc(); @elapsed buildfp() end for _ in 1:2])

# the saddle solve alone: rebuild the hierarchy and the block, then time it
empty!(MS.SEMQUAD_MG_CACHE); MS.SEMQUAD_PRESSURE_CACHE[] = nothing
mg = MS.make_semquad_mg(ss, qt, p, :free, false, eta, nothing; cycles=1,
                        smoother=:jacobi, coarse_operator=:rediscretize,
                        operator=:matrixfree, coarsen_factor=1.5,
                        smoothing=:auto, mg_threads=THR, fine_Cu=sys.Cu)
Fp, pscale = MS.semquad_mg_pressure_precond(ss, sys.Cu, sys.Cp, sys.Bs,
                                            sys.Bt, nothing, eta; mode=:auto,
                                            cache=false, applyK=mfapplyK)
function it()
    @elapsed MS.semquad_mg_saddle_solve(nothing, sys.Bs, sys.Bt, Fp, pscale,
                                        vcat(sys.fr, sys.gr), mg; tol=1.0e-6,
                                        maxit=500, restart=100,
                                        vel_solver=:vcycle, applyK=mfapplyK)
end
it()
t_it = minimum([begin GC.gc(); it() end for _ in 1:2])
t_asm = t_full - t_hier - t_fp - t_it

# standalone assemblies: the full one (K formed, eta sampled in the leaf loop)
# and, on the new tree, the reduced one (no K, no Kred, no eta in the loop; its
# `fr` uses the matrix-free K u0, so it also builds the operator's eta field)
t_asm_full = minimum([begin GC.gc(); @elapsed solve_semquad_stokes_var(
    ss, qt, eta, Z2, fz, Z2, Z2; bc=:free, linear=:direct, assemble_only=true)
end for _ in 1:2])
t_asm_min = NaN
if NEW
    t_asm_min = minimum([begin GC.gc(); @elapsed solve_semquad_stokes_var(
        ss, qt, eta, Z2, fz, Z2, Z2; bc=:free, linear=:mg,
        mg_operator=:matrixfree, assemble_only=:mg)
    end for _ in 1:2])
end
nk = -1
if NEW
    n0 = MS.SEMSTOKES_K_ASSEMBLIES[]
    coldsolve()
    nk = MS.SEMSTOKES_K_ASSEMBLIES[] - n0
end

@printf("RESULT case=%s thr=%d new=%d reps=%d nit=%d t_full=%.4f t_warm=%.4f t_hier=%.4f t_hier_full=%.4f t_bfbt=%.4f t_iter=%.4f t_asm=%.4f t_asm_full=%.4f t_asm_min=%.4f kcount=%d\n", CASE, THR, NEW, REPS, nit, t_full, t_warm, t_hier, t_hier_full, t_fp, t_it, t_asm, t_asm_full, t_asm_min, nk)
rm(tmpdir; force=true, recursive=true)
