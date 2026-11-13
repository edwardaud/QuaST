#!/usr/bin/env julia
# =============================================================================
# profile_mg_current.jl -- current-composition profile of ONE warm matrix-free
# MG solve on a production state (docs/MG.md, "current composition").
#
#   MG_PROFILE_CASE=F  julia --project=julia --threads=8 \
#       julia/scripts/profile_mg_current.jl
#   MG_PROFILE_CASE=P1 julia --project=julia --threads=1 \
#       julia/scripts/profile_mg_current.jl
#
# What it measures, all with the *shipped* defaults (matrix-free MG, 1.5
# coarsening, :auto smoothing, w-BFBT S2 matrix-free, region-threaded V-cycle,
# Cu phases, eta lane table):
#
#   1. the production warm solve (`solve_semquad_stokes_var(linear=:mg)`,
#      caches warm: the in-tree production step) and the cold solve, split into
#      assembly / hierarchy build / w-BFBT build / FGMRES iteration / the final
#      residual check + u,p update;
#   2. one outer iteration: V-cycle, Bs zc, w-BFBT (the two chol(S1) triangular
#      solves and the matrix-free S2), the outer operator apply (K, Bt, Bs) and
#      the FGMRES own work (MGS, true residual, norms, Hessenberg, restart);
#   3. one V-cycle: the finest-level applies (the leaf apply), the coarse-level
#      applies, the transfers and the smoothing vector ops, by level;
#   4. the FGMRES true-residual history and the iteration-count headroom.
#
# The solver is not modified and nothing here is on a production path.  The
# components come from two mechanisms:
#
#   (a) timers around the production entry points (`solve_semquad_stokes_var`,
#       `assemble_only=:mg`, `make_semquad_mg`, `semquad_mg_pressure_precond`,
#       `semquad_mg_saddle_solve`, `semquad_mg_apply`);
#   (b) a *replica* of the two loops the production API does not expose
#       (`_semquad_mg_vcycle!` / `_semquad_mg_vcycle_region!` and
#       `semquad_mg_saddle_solve` + `_semquad_pressure_block!` + `fgmres_restarted`),
#       calling the same module internals in the same order with `time_ns()`
#       around each primitive.
#
# Both replicas are checked **bitwise** (Float64 bit patterns) against the
# production functions on the same inputs before their numbers are used, and the
# script aborts on a mismatch.  The script also refuses to run if the shipped
# switches have been moved off the region/eta-lane defaults it knows how to
# mirror, so it can never describe a configuration other than the one shipped.
#
# Environment (all optional):
#   MG_PROFILE_CASE    F (default) or P1; selects the checkpoint below
#   MG_PROFILE_CKPT    an explicit checkpoint path (overrides CASE)
#   MG_PROFILE_THR     mg_threads (default: the Julia process' thread count,
#                      which is the production `linear=:auto` choice at this
#                      size; start Julia with --threads=N)
#   MG_PROFILE_TOL     outer FGMRES tolerance (default 1e-6)
#   MG_PROFILE_REPEATS samples of the single-V-cycle timing (default 7)
#
# BLAS is pinned to one thread, as in every number in docs/.
# =============================================================================

using MantleSEM, Printf, LinearAlgebra, SparseArrays, Random

const MS = MantleSEM

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const CASE_DIRS = Dict(
    "F" => joinpath(REPO, "out_julia", "_s7", "case", "pre_step6131.h5"),
    "P1" => joinpath(REPO, "out_julia", "s340", "ab_none", "resume.h5"))
const CASE = uppercase(get(ENV, "MG_PROFILE_CASE", "F"))
const CKPT = get(ENV, "MG_PROFILE_CKPT", get(CASE_DIRS, CASE, ""))
const THR = parse(Int, get(ENV, "MG_PROFILE_THR", string(Threads.nthreads())))
const REPEATS = parse(Int, get(ENV, "MG_PROFILE_REPEATS", "7"))
const TOL = parse(Float64, get(ENV, "MG_PROFILE_TOL", "1.0e-6"))
const MGTHR = MS._semquad_mg_thread_count(THR)
const Z2 = (x, y) -> 0.0

BLAS.set_num_threads(1)

@inline el(t0::UInt64) = 1.0e-9 * (time_ns() - t0)

function timings(f::F, n::Int) where {F}
    f()                                        # compile / warm the closure
    t = Float64[]
    for _ in 1:n
        push!(t, @elapsed f())
    end
    return t
end
besttime(f, n) = minimum(timings(f, n))
medtime(f, n) = (t = timings(f, n); sort!(t)[cld(length(t), 2)])
med(v) = (s = sort(copy(v)); s[cld(length(s), 2)])
bitwise(a, b) = length(a) == length(b) &&
                reinterpret(UInt64, a) == reinterpret(UInt64, b)

# ---------------------------------------------------------------------------
# parameter card rebuilt from the checkpoint's `config` attribute
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# accumulators
# ---------------------------------------------------------------------------
mutable struct VCProf
    t_leaf::Vector{Float64}     # matrix-free leaf phase (kernel + scatter)
    t_cug::Vector{Float64}      # Cu * v (or the region phase)
    t_cus::Vector{Float64}      # Cu' * y (or the region phase)
    t_sub::Vector{Float64}      # b - y (smoothing residual and residual eval)
    t_jac::Vector{Float64}      # damped-Jacobi update
    t_restr::Vector{Float64}    # restriction  rc = P' r
    t_prol::Vector{Float64}     # prolongation py = P e
    t_axpy::Vector{Float64}     # x .+= py
    t_fill::Vector{Float64}     # fill!(x, 0)
    t_coarse::Float64           # coarsest-level Cholesky solve
    n_apply::Vector{Int}
    n_pre::Vector{Int}
    n_post::Vector{Int}
    n_res::Vector{Int}
    t_total::Float64
end
VCProf(n::Int) = VCProf(zeros(n), zeros(n), zeros(n), zeros(n), zeros(n),
                        zeros(n), zeros(n), zeros(n), zeros(n), 0.0,
                        zeros(Int, n), zeros(Int, n), zeros(Int, n),
                        zeros(Int, n), 0.0)
function addv!(a::VCProf, b::VCProf)
    for f in fieldnames(VCProf)
        fa = getfield(a, f); fb = getfield(b, f)
        if fa isa Vector{Float64}
            fa .+= fb
        elseif fa isa Vector{Int}
            fa .+= fb
        elseif fa isa Float64
            setfield!(a, f, fa + fb)
        end
    end
    return a
end
function scale!(a::VCProf, s::Float64)
    for f in fieldnames(VCProf)
        fa = getfield(a, f)
        fa isa Vector{Float64} && (fa .*= s)
    end
    a.t_coarse *= s
    a.t_total *= s
    return a
end

mutable struct SolveProf
    t_applyK::Float64
    t_copy::Float64
    t_Bt::Float64
    t_Bs::Float64
    t_vcycle::Float64
    t_Bszc::Float64
    t_pblock::Float64
    t_f1a::Float64
    t_s2::Float64
    t_f1b::Float64
    t_total::Float64
    n_applyA::Int
    n_vcycle::Int
    n_pre::Int
    n_f1::Int
    n_s2::Int
end
SolveProf() = SolveProf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0, 0, 0, 0, 0)

mutable struct FGProf
    t_mgs::Float64
    t_norm::Float64
    t_hess::Float64
    t_true::Float64
    t_copy::Float64
    t_restart::Float64
    t_total::Float64
    hist::Vector{Float64}
    hest::Vector{Float64}
    r0::Float64
    n_restart::Int
end
FGProf(m::Int) = FGProf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, zeros(m),
                        zeros(m), 0.0, 0)

# ---------------------------------------------------------------------------
# the matrix-free apply with its three parts
#
# Same functions, in the same order, as `_semquad_apply_Kred_region!` (threaded
# level inside a region), `_semquad_apply_Kred_threaded!` (per-call threaded
# form, unused by the region V-cycle but kept for completeness) and
# `_semquad_apply_Kred!` (serial level), with a clock around each part.
# ---------------------------------------------------------------------------
function apply_prof!(out::Vector{Float64}, v::AbstractVector{Float64},
                     mg::MS.SemQuadMG, lev::Int,
                     region::Union{Nothing,MS.SemQuadParRegion},
                     P::VCProf)
    op = mg.mfops[lev]::MS.SemQuadKOp
    Cu = mg.geo.Cus[lev]
    Cut = mg.geo.Cuts[lev]
    etav = mg.mfetas[lev]
    work = mg.mfwork[lev]::MS.SemQuadKWork
    if mg.threads > 1 && mg.geo.nfs[lev] > MS.SEMQUAD_MG_SERIAL_DOFS[]
        par = mg.par[lev]::MS.SemQuadKPar
        if region !== nothing
            t0 = time_ns(); MS._semquad_spmv_region!(work.full, Cut, v, par.nt,
                                                     region); P.t_cug[lev] += el(t0)
            t0 = time_ns()
            MS._semquad_apply_K_raw_region!(work.op_out, op, etav, work.full,
                                            par, region)
            P.t_leaf[lev] += el(t0)
            t0 = time_ns()
            MS._semquad_spmv_at_region!(out, Cu, work.op_out, par.nt, region)
            P.t_cus[lev] += el(t0)
        else
            t0 = time_ns(); mul!(work.full, Cu, v); P.t_cug[lev] += el(t0)
            t0 = time_ns()
            MS._semquad_apply_K_raw_threaded!(work.op_out, op, etav, work.full,
                                              par)
            P.t_leaf[lev] += el(t0)
            t0 = time_ns(); mul!(out, transpose(Cu), work.op_out)
            P.t_cus[lev] += el(t0)
        end
    else
        t0 = time_ns(); mul!(work.full, Cu, v); P.t_cug[lev] += el(t0)
        t0 = time_ns()
        MS._semquad_apply_K_raw!(work.op_out, op, etav, work.full, work)
        P.t_leaf[lev] += el(t0)
        t0 = time_ns(); mul!(out, transpose(Cu), work.op_out)
        P.t_cus[lev] += el(t0)
    end
    P.n_apply[lev] += 1
    return out
end

# ---------------------------------------------------------------------------
# one V-cycle, per level, timed
#
# `vcycle_prof!` mirrors `_semquad_mg_vcycle!`; a level above
# `SEMQUAD_MG_SERIAL_DOFS` dispatches to `vcycle_region_prof!`, which mirrors
# `_semquad_mg_vcycle_region!` (two fork/join regions per level).  The arithmetic
# is the production arithmetic: the same module functions on the same buffers.
# ---------------------------------------------------------------------------
function smooth_prof!(x::Vector{Float64}, b::Vector{Float64}, mg::MS.SemQuadMG,
                      lev::Int, ns::Int,
                      region::Union{Nothing,MS.SemQuadParRegion}, P::VCProf,
                      pre::Bool)
    D = mg.D[lev]
    r = mg.sr[lev]
    y = mg.sy[lev]
    for _ in 1:ns
        apply_prof!(y, x, mg, lev, region, P)
        t0 = time_ns()
        if mg.threads > 1
            MS._semquad_td_sub!(r, b, y, mg.threads, region)
        else
            @. r = b - y
        end
        P.t_sub[lev] += el(t0)
        t0 = time_ns()
        if mg.threads > 1
            MS._semquad_td_jacobi!(x, r, D, mg.omega, mg.threads, region)
        else
            @. x += mg.omega * r / D
        end
        P.t_jac[lev] += el(t0)
    end
    pre ? (P.n_pre[lev] += ns) : (P.n_post[lev] += ns)
    return nothing
end

function vcycle_prof!(x::Vector{Float64}, b::Vector{Float64}, mg::MS.SemQuadMG,
                      lev::Int, P::VCProf)
    if lev == 1
        t0 = time_ns(); x .= mg.cA \ b; P.t_coarse += el(t0)
        return nothing
    end
    if MS._semquad_mg_threaded_level(mg, lev)
        vcycle_region_prof!(x, b, mg, lev, P)
        return nothing
    end
    t0 = time_ns(); fill!(x, 0.0); P.t_fill[lev] += el(t0)
    smooth_prof!(x, b, mg, lev, mg.pre[lev], nothing, P, true)
    apply_prof!(mg.sy[lev], x, mg, lev, nothing, P)
    t0 = time_ns()
    if mg.threads > 1
        MS._semquad_td_sub!(mg.sc[lev], b, mg.sy[lev], mg.threads)
    else
        @. mg.sc[lev] = b - mg.sy[lev]
    end
    P.t_sub[lev] += el(t0)
    P.n_res[lev] += 1
    t0 = time_ns()
    MS._semquad_mg_restrict!(mg.rcs[lev], mg.geo.Ps[lev - 1], mg.sc[lev],
                             mg.threads)
    P.t_restr[lev] += el(t0)
    ec = mg.ecs[lev]
    vcycle_prof!(ec, mg.rcs[lev], mg, lev - 1, P)
    t0 = time_ns()
    if mg.threads > 1 && !isempty(mg.Pts)
        MS._semquad_mg_prolong!(mg.pys[lev],
                                mg.Pts[lev - 1]::SparseMatrixCSC{Float64,Int},
                                ec, mg.threads)
    else
        mul!(mg.pys[lev], mg.geo.Ps[lev - 1], ec)
    end
    P.t_prol[lev] += el(t0)
    t0 = time_ns(); x .+= mg.pys[lev]; P.t_axpy[lev] += el(t0)
    smooth_prof!(x, b, mg, lev, mg.post[lev], nothing, P, false)
    return nothing
end

function vcycle_region_prof!(x::Vector{Float64}, b::Vector{Float64},
                             mg::MS.SemQuadMG, lev::Int, P::VCProf)
    t0 = time_ns(); fill!(x, 0.0); P.t_fill[lev] += el(t0)
    rc = mg.rcs[lev]
    nw = mg.operator === :matrixfree ? (mg.par[lev]::MS.SemQuadKPar).nt :
         mg.threads
    region = MS.SemQuadParRegion(nw)
    MS._semquad_region_run!(region) do
        smooth_prof!(x, b, mg, lev, mg.pre[lev], region, P, true)
        apply_prof!(mg.sy[lev], x, mg, lev, region, P)
        t0 = time_ns()
        if mg.threads > 1
            MS._semquad_td_sub!(mg.sc[lev], b, mg.sy[lev], mg.threads, region)
        else
            @. mg.sc[lev] = b - mg.sy[lev]
        end
        P.t_sub[lev] += el(t0)
        P.n_res[lev] += 1
        t0 = time_ns()
        MS._semquad_mg_restrict!(rc, mg.geo.Ps[lev - 1], mg.sc[lev],
                                 mg.threads, region)
        P.t_restr[lev] += el(t0)
    end
    ec = mg.ecs[lev]
    vcycle_prof!(ec, rc, mg, lev - 1, P)
    py = mg.pys[lev]
    MS._semquad_region_run!(region) do
        t0 = time_ns()
        if isempty(mg.Pts)
            mul!(py, mg.geo.Ps[lev - 1], ec)
        else
            MS._semquad_mg_prolong!(py,
                                    mg.Pts[lev - 1]::SparseMatrixCSC{Float64,Int},
                                    ec, mg.threads, region)
        end
        P.t_prol[lev] += el(t0)
        t0 = time_ns(); x .+= py; P.t_axpy[lev] += el(t0)
        smooth_prof!(x, b, mg, lev, mg.post[lev], region, P, false)
    end
    return nothing
end

function mg_apply_prof!(z::Vector{Float64}, b::AbstractVector{Float64},
                        mg::MS.SemQuadMG, P::VCProf)
    bb = Vector{Float64}(b)
    t0 = time_ns()
    vcycle_prof!(z, bb, mg, mg.geo.L + 1, P)
    for _ in 2:mg.cycles
        rz = mg.sc[end]
        apply_prof!(rz, z, mg, mg.geo.L + 1, nothing, P)
        r = bb .- rz
        dz = zeros(Float64, length(bb))
        vcycle_prof!(dz, r, mg, mg.geo.L + 1, P)
        z .+= dz
    end
    P.t_total += el(t0)
    return z
end

# ---------------------------------------------------------------------------
# the w-BFBT pressure block, timed per factor
# ---------------------------------------------------------------------------
function pressure_block_prof!(wP::Vector{Float64}, rp::Vector{Float64},
                              rp2::Vector{Float64}, s1::Vector{Float64},
                              s2::Vector{Float64}, Fp, pscale::Float64,
                              nf::Int, S::SolveProf)
    t_all = time_ns()
    if Fp isa NamedTuple && Fp.kind === :pcd
        error("this profile measures the shipped w-BFBT block, not :pcd")
    elseif Fp isa NamedTuple
        t0 = time_ns(); rp2 .= Fp.F1 \ rp; S.t_f1a += el(t0); S.n_f1 += 1
        t0 = time_ns()
        if Fp.S2 === nothing
            dinv = Fp.Dinvv
            mul!(s1, Fp.Bt, rp2)
            @. s1 *= dinv
            if Fp.Kred === nothing
                Fp.applyK(s2, s1)
            else
                mul!(s2, Fp.Kred, s1)
            end
            @. s2 *= dinv
            mul!(rp, Fp.Bs, s2)
        else
            mul!(rp, Fp.S2, rp2)
        end
        S.t_s2 += el(t0); S.n_s2 += 1
        t0 = time_ns(); rp2 .= Fp.F1 \ rp; S.t_f1b += el(t0); S.n_f1 += 1
        @. wP[(nf + 1):end] = -pscale * rp2
    else
        error("this profile measures the shipped w-BFBT block, not :mass")
    end
    S.t_pblock += el(t_all)
    return wP
end

# ---------------------------------------------------------------------------
# FGMRES, same loop as `fgmres_restarted` with a clock per component and the
# true-residual history recorded per iteration index
# ---------------------------------------------------------------------------
function fgmres_prof(applyA, b::Vector{Float64}; precond=nothing,
                     tol::Float64=1.0e-10, maxit::Int=200, restart::Int=30,
                     x0::Union{Nothing,Vector{Float64}}=nothing, work=nothing,
                     F::FGProf=FGProf(maxit))
    n = length(b)
    x = x0 === nothing ? zeros(Float64, n) : copy(x0)
    nb = norm(b)
    nb == 0.0 && return x, 0.0, 0
    nit = 0
    true_rel = Inf
    dx = zeros(Float64, n)
    t_start = time_ns()
    while nit < maxit
        tr = time_ns()
        r = b - applyA(x)
        beta = norm(r)
        F.t_restart += el(tr); F.n_restart += 1
        nit == 0 && (F.r0 = beta / nb)
        true_rel = beta / nb
        true_rel <= tol && break
        m = min(restart, maxit - nit)
        V, Z, W, H = if work === nothing || size(work.V, 1) != n ||
                        size(work.V, 2) != m + 1
            (zeros(Float64, n, m + 1), zeros(Float64, n, m),
             zeros(Float64, n, m), zeros(Float64, m + 1, m))
        else
            (work.V, work.Z, work.W, work.H)
        end
        V[:, 1] = r ./ beta
        g = zeros(Float64, m + 1); g[1] = beta
        cs = zeros(Float64, m); sn = zeros(Float64, m)
        y = zeros(Float64, 0)
        k_used = 0
        for k in 1:m
            nit += 1
            z = precond === nothing ? V[:, k] : precond(V[:, k])
            tk = time_ns(); Z[:, k] = z; F.t_copy += el(tk)
            tk = time_ns(); w = applyA(z); F.t_copy += el(tk)
            tk = time_ns(); W[:, k] = w; F.t_copy += el(tk)
            tk = time_ns()
            for i in 1:k
                vi = view(V, :, i)
                H[i, k] = dot(w, vi)
                axpy!(-H[i, k], vi, w)
            end
            F.t_mgs += el(tk)
            tk = time_ns()
            H[k + 1, k] = norm(w)
            if H[k + 1, k] > 1e-14
                @views V[:, k + 1] .= w ./ H[k + 1, k]
            else
                @views fill!(V[:, k + 1], 0.0)
            end
            F.t_norm += el(tk)
            tk = time_ns()
            for i in 1:(k - 1)
                t = cs[i] * H[i, k] + sn[i] * H[i + 1, k]
                H[i + 1, k] = -sn[i] * H[i, k] + cs[i] * H[i + 1, k]
                H[i, k] = t
            end
            den = hypot(H[k, k], H[k + 1, k])
            if den == 0.0
                k_used = k
                break
            end
            cs[k] = H[k, k] / den; sn[k] = H[k + 1, k] / den
            H[k, k] = cs[k] * H[k, k] + sn[k] * H[k + 1, k]
            H[k + 1, k] = 0.0
            g[k + 1] = -sn[k] * g[k]
            g[k] = cs[k] * g[k]
            k_used = k
            F.hest[nit] = abs(g[k + 1]) / nb
            y = H[1:k, 1:k] \ g[1:k]
            F.t_hess += el(tk)
            tk = time_ns()
            mul!(dx, view(W, :, 1:k), y)
            @. dx = r - dx
            true_rel = norm(dx) / nb
            F.t_true += el(tk)
            F.hist[nit] = true_rel
            true_rel <= tol && break
        end
        k_used == 0 && break
        tr = time_ns()
        y = H[1:k_used, 1:k_used] \ g[1:k_used]
        mul!(dx, view(Z, :, 1:k_used), y)
        x .+= dx
        F.t_restart += el(tr)
    end
    F.t_total += el(t_start)
    return x, true_rel, nit
end

# ---------------------------------------------------------------------------
# the outer saddle solve, timed per part (same body as
# `semquad_mg_saddle_solve`)
# ---------------------------------------------------------------------------
function saddle_solve_prof(Kred, Bs, Bt, Fp, pscale::Float64,
                           b::Vector{Float64}, mg::MS.SemQuadMG;
                           tol::Float64=1.0e-10, maxit::Int=500,
                           restart::Int=100, vel_solver::Symbol=:vcycle,
                           applyK::Union{Nothing,Function}=nothing,
                           S::SolveProf=SolveProf(),
                           F::FGProf=FGProf(maxit))
    vel_solver === :vcycle ||
        error("this profile measures the shipped vel_solver=:vcycle")
    nf = size(Bs, 2); nfp = size(Bs, 1); nftot = nf + nfp
    applyK! = applyK === nothing ? (out, v) -> mul!(out, Kred, v) : applyK
    wA = zeros(Float64, nftot)
    wP = zeros(Float64, nftot)
    zc = zeros(Float64, nf)
    rp = zeros(Float64, nfp)
    rp2 = zeros(Float64, nfp)
    s1 = zeros(Float64, nf)
    s2 = zeros(Float64, nf)
    ku = zeros(Float64, nf)
    function applyA(v::Vector{Float64})
        ta = time_ns(); applyK!(ku, view(v, 1:nf)); S.t_applyK += el(ta)
        ta = time_ns(); wA[1:nf] .= ku; S.t_copy += el(ta)
        ta = time_ns(); wA[1:nf] .+= Bt * v[(nf + 1):nftot]; S.t_Bt += el(ta)
        ta = time_ns(); wA[(nf + 1):nftot] = Bs * v[1:nf]; S.t_Bs += el(ta)
        S.n_applyA += 1
        return wA
    end
    function precond(v::Vector{Float64})
        t_all = time_ns()
        t1 = time_ns(); zc .= MS.semquad_mg_apply(mg, v[1:nf])
        S.t_vcycle += el(t1); S.n_vcycle += 1
        t2 = time_ns()
        wP[1:nf] .= zc
        rp .= v[(nf + 1):nftot] .- Bs * zc
        S.t_Bszc += el(t2)
        pressure_block_prof!(wP, rp, rp2, s1, s2, Fp, pscale, nf, S)
        S.n_pre += 1
        return wP
    end
    ts = time_ns()
    sol, relres, nit = fgmres_prof(applyA, b; precond=precond, tol=tol,
                                   maxit=maxit, restart=restart,
                                   work=MS._semquad_fgmres_work(nftot,
                                                                min(restart, maxit)),
                                   F=F)
    S.t_total += el(ts)
    return sol, nit, relres
end

# The production driver's final block: true-residual check and x = x0 + C y.
function tail_block(ufree::Vector{Float64}, pfree::Vector{Float64},
                    fr::Vector{Float64}, gr::Vector{Float64}, Bs, Bt, Cu, Cp,
                    u0::Vector{Float64}, p0::Vector{Float64}, mfapplyK)
    nf = length(ufree)
    Ku = zeros(Float64, nf)
    t0 = time_ns()
    mfapplyK(Ku, ufree)
    Ku .+ Bt * pfree .- fr
    r2 = Bs * ufree .- gr
    rden = max(norm(fr), norm(gr), 1.0)
    norm(Ku) / rden; norm(r2) / rden
    u = u0 + Cu * ufree
    p = p0 + Cp * pfree
    return el(t0)
end

# ---------------------------------------------------------------------------
# the shipped configuration this instrument knows how to mirror
# ---------------------------------------------------------------------------
function assert_shipped(mg::MS.SemQuadMG, mgthr::Int)
    mg.operator === :matrixfree ||
        error("shipped defaults moved: operator=$(mg.operator), expected :matrixfree")
    mg.smoother === :jacobi ||
        error("shipped defaults moved: smoother=$(mg.smoother), expected :jacobi")
    mg.cycles == 1 || error("this profile measures cycles=1")
    mg.threads == mgthr ||
        error("hierarchy mg_threads=$(mg.threads), expected $mgthr")
    MS.SEMQUAD_MG_REGION[] ||
        error("SEMQUAD_MG_REGION is off; this profile measures the shipped region V-cycle")
    MS.SEMQUAD_MG_REGION_TRANSFER[] ||
        error("SEMQUAD_MG_REGION_TRANSFER is off; this profile measures the shipped path")
    MS.SEMQUAD_MG_REGION_CU[] ||
        error("SEMQUAD_MG_REGION_CU is off; this profile measures the shipped path")
    MS.SEMQUAD_MG_ETA_LANES[] ||
        error("SEMQUAD_MG_ETA_LANES is off; this profile measures the shipped lane table")
    MS.SEMQUAD_MG_LEVEL_CACHE[] ||
        error("SEMQUAD_MG_LEVEL_CACHE is off; this profile measures the shipped cache")
    return nothing
end

# ===========================================================================
function main()
    isfile(CKPT) || error("checkpoint not found: $CKPT (set MG_PROFILE_CASE or MG_PROFILE_CKPT)")
    nthreads = Threads.nthreads()
    println("="^104)
    @printf("MF MG current-composition profile   case=%s   Julia threads=%d   mg_threads=%d   BLAS threads=1\n",
            CASE, nthreads, MGTHR)
    @printf("checkpoint=%s\nouter tol=%.1e   V-cycle samples=%d   cycles=1\n",
            CKPT, TOL, REPEATS)
    @printf("shipped switches: region=%s transfer=%s cu=%s eta_lanes=%s serial_dofs=%d level_cache=%s\n",
            MS.SEMQUAD_MG_REGION[], MS.SEMQUAD_MG_REGION_TRANSFER[],
            MS.SEMQUAD_MG_REGION_CU[], MS.SEMQUAD_MG_ETA_LANES[],
            MS.SEMQUAD_MG_SERIAL_DOFS[], MS.SEMQUAD_MG_LEVEL_CACHE[])
    println("="^104)

    # ---------------- state ------------------------------------------------
    tmpdir = mktempdir(; cleanup=false)
    copyto = joinpath(tmpdir, basename(CKPT))
    cp(CKPT, copyto; force=true)
    chmod(copyto, 0o444)
    t_load0 = time_ns()
    st = load_semquad_state(copyto)
    t_load = el(t_load0)
    qt = st.qt; p = st.p; theta = st.theta; ux = st.ux; uz = st.uz
    pr = params_from_config(st.config)
    ph = phases_from_config(st.config)
    ss = make_semstokes(qt, p)
    length(theta) == ss.su.N ||
        error("checkpoint theta has $(length(theta)) entries, velocity space has $(ss.su.N)")
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
    @printf("\ncheckpoint copy %s (mode 444) loaded in %.2f s; step %d\n",
            copyto, t_load, st.step)
    @printf("state: %d leaves, p=%d, velocity dofs %d, pressure dofs %d\n",
            length(qt.leaves), p, ss.su.N, ss.sp.N)
    @printf("rheology: composite isostress, Ra=%g, H=%g, %d phase transitions, bc=free\n",
            pr.Ra, pr.H, length(ph))

    # ---------------- production solve functions ------------------------------------
    mgkw = (; bc=:free, deviatoric=false, mass_form=:boussinesq, linear=:mg,
             mg_operator=:matrixfree, mg_vel_solver=:vcycle, mg_threads=THR,
             mg_tol=TOL)
    solve_mg() = solve_semquad_stokes_var(ss, qt, eta_fun, Z2, fz, Z2, Z2;
                                          mgkw...)

    # JIT / first-touch warm-up; never measured.
    t_first = @elapsed solve_mg()
    @printf("\nfirst-ever MG solve in this process (JIT + first touch, discarded): %.3f s\n",
            t_first)

    # cold full solve (hierarchy + w-BFBT built inside the call), min of 2
    cold = Float64[]
    for _ in 1:2
        empty!(MS.SEMQUAD_MG_CACHE); MS.SEMQUAD_PRESSURE_CACHE[] = nothing
        GC.gc()
        push!(cold, @elapsed solve_mg())
    end
    t_cold = minimum(cold)
    nit_cold = LAST_SEMSTOKES_SCHUR_ITERS[]
    res_cold = LAST_SEMSTOKES_RESIDUAL[]

    # warm full solve: the in-tree production step, medians of five consecutive
    # solves (the machine's run-to-run spread is larger than the phase split)
    warm = Float64[]
    for _ in 1:5
        push!(warm, @elapsed solve_mg())
    end
    t_warm = med(warm)
    nit = LAST_SEMSTOKES_SCHUR_ITERS[]
    relres = LAST_SEMSTOKES_RESIDUAL[]

    # ---------------- assembly (reduced MF geometry) --------------------------------
    asmkw = (; bc=:free, deviatoric=false, mass_form=:boussinesq,
              linear=:mg, mg_operator=:matrixfree, mg_threads=THR,
              assemble_only=:mg)
    sys = solve_semquad_stokes_var(ss, qt, eta_fun, Z2, fz, Z2, Z2; asmkw...)
    nf = size(sys.Bs, 2); nfp = size(sys.Bs, 1); nftot = nf + nfp
    t_asm = medtime(() -> solve_semquad_stokes_var(ss, qt, eta_fun, Z2, fz, Z2,
                                                   Z2; asmkw...), 3)

    # ---------------- the finest-level operator and the apply closure ---------------
    mfop = MS.semquad_kop(ss, qt; deviatoric=false)
    mfetav = MS.semquad_eta_lanes(mfop, MS.semquad_eta_field(mfop, eta_fun))
    mfwork = MS.SemQuadKWork(mfop.n, mfop.nq, 2 * ss.su.N)
    nt_outer = MS._semquad_mg_thread_count(THR)
    mfapplyK = if nt_outer > 1
        par_outer = MS.SemQuadKPar(mfop, nt_outer)
        (out, v) -> MS._semquad_apply_Kred_threaded!(out, mfop, sys.Cu, mfetav,
                                                     v, mfwork, par_outer)
    else
        (out, v) -> MS._semquad_apply_Kred!(out, mfop, sys.Cu, mfetav, v,
                                            mfwork)
    end
    zz = zeros(nf); mfapplyK(zz, ones(nf))

    # ---------------- warm hierarchy (geometry cached, per-level operators) --------
    build_mg() = MS.make_semquad_mg(ss, qt, p, :free, false, eta_fun, nothing;
                                    cycles=1, smoother=:jacobi,
                                    coarse_operator=:rediscretize,
                                    operator=:matrixfree, coarsen_layers=1,
                                    coarsen_factor=1.5, coarse_direct_max=0,
                                    smoothing=:auto, mg_threads=THR,
                                    mg_diag=:constrained, fine_Cu=sys.Cu,
                                    fine_mfop=mfop, fine_mfeta=mfetav)
    st0 = MS.semquad_mg_level_cache_stats()
    t_mg_warm = medtime(build_mg, 3)
    st1 = MS.semquad_mg_level_cache_stats()
    mg = build_mg()
    nlev = mg.geo.L + 1
    assert_shipped(mg, MGTHR)
    @printf("\nMG hierarchy: %d levels, dofs %s, mg_threads=%d, smoothing=%s omega=%.4f, pre=%s post=%s\n",
            nlev, join(mg.geo.nfs, ", "), mg.threads,
            MS.LAST_SEMSTOKES_MG_PROFILE[][1], mg.omega, join(mg.pre, ","),
            join(mg.post, ","))

    # ---------------- warm w-BFBT (pressure cache hit) -----------------------------
    key = MS._saddle_key(qt, p, :free, nf, nfp)
    b0 = MS.SEMSTOKES_PRESSURE_BUILDS[]; r0 = MS.SEMSTOKES_PRESSURE_REUSES[]
    build_fp(cache::Bool) = MS.semquad_mg_pressure_precond(
        ss, sys.Cu, sys.Cp, sys.Bs, sys.Bt, nothing, eta_fun;
        mode=:auto, scale=0.0, weight=:inv_eta, key=key,
        eta_tol=MS.SEMQUAD_PRESSURE_DEFAULT_ETA_TOL, cache=cache,
        bfbt_d=:leaf, bfbt_s2=:free, bfbt_s1=:chol, applyK=mfapplyK)
    t_bfbt_warm = medtime(() -> build_fp(true), 3)
    Fp, pscale = build_fp(true)
    Fp isa NamedTuple && Fp.kind in (:bfbt, :bfbt_m1) ||
        error("the shipped pressure block is not w-BFBT (got $(Fp isa NamedTuple ? Fp.kind : :mass))")
    reused = MS.LAST_SEMSTOKES_PRESSURE_REUSED[]
    builds_delta = MS.SEMSTOKES_PRESSURE_BUILDS[] - b0
    reuses_delta = MS.SEMSTOKES_PRESSURE_REUSES[] - r0

    # ---------------- the FGMRES iteration alone -----------------------------------
    rhs = vcat(sys.fr, sys.gr)
    solveit() = MS.semquad_mg_saddle_solve(nothing, sys.Bs, sys.Bt, Fp, pscale,
                                           rhs, mg; tol=TOL, maxit=500,
                                           restart=100, vel_solver=:vcycle,
                                           applyK=mfapplyK)
    sol_ref, nit_ref, res_ref, _ = solveit()
    it_times = timings(solveit, 5)
    t_iter = med(it_times)
    (nit_ref == nit && res_ref == relres) ||
        @printf("NOTE: iteration alone gives nit=%d relres=%.4e, the warm solve gave nit=%d relres=%.4e\n",
                nit_ref, res_ref, nit, relres)

    # ---------------- one V-cycle replica ------------------------------------------
    Random.seed!(20261112)
    b = randn(nf)
    zref = MS.semquad_mg_apply(mg, b)
    mg_apply_prof!(zeros(nf), b, mg, VCProf(nlev))     # compile the replica
    V = VCProf(nlev)
    vts = Float64[]
    for _ in 1:REPEATS
        Vp = VCProf(nlev)
        z = zeros(nf)
        mg_apply_prof!(z, b, mg, Vp)
        bitwise(z, zref) ||
            error("V-cycle replica is not bitwise equal to semquad_mg_apply")
        push!(vts, Vp.t_total)
        addv!(V, Vp)
    end
    v_mean = sum(vts) / REPEATS
    v_min = minimum(vts)
    scale!(V, 1.0 / REPEATS)
    v_prod = besttime(() -> MS.semquad_mg_apply(mg, b), REPEATS)
    @printf("\none V-cycle: production semquad_mg_apply %.3f ms (best of %d) | replica mean %.3f ms, min %.3f ms | bitwise equal\n",
            1e3 * v_prod, REPEATS, 1e3 * v_mean, 1e3 * v_min)
    n_app_fine = V.n_apply[nlev]
    n_app_coarse = sum(V.n_apply[2:(nlev - 1)])
    @printf("it does %d finest-level + %d coarse-level MF applies, %d restrictions, %d prolongations\n",
            n_app_fine ÷ REPEATS, n_app_coarse ÷ REPEATS, nlev - 1, nlev - 1)

    # ---------------- the outer solve replica --------------------------------------
    saddle_solve_prof(nothing, sys.Bs, sys.Bt, Fp, pscale, rhs, mg;
                      tol=TOL, maxit=500, restart=100,
                      applyK=mfapplyK, S=SolveProf())   # compile the closures
    S = SolveProf()
    FG = FGProf(500)
    sol_p, nit_p, res_p = saddle_solve_prof(nothing, sys.Bs, sys.Bt, Fp, pscale,
                                            rhs, mg; tol=TOL, maxit=500,
                                            restart=100, applyK=mfapplyK,
                                            S=S, F=FG)
    (bitwise(sol_p, sol_ref) && nit_p == nit_ref && res_p == res_ref) ||
        error("saddle-solve replica is not bitwise equal to semquad_mg_saddle_solve")
    @printf("outer solve: production %.3f s (median of 5, %.3f-%.3f) | replica %.3f s | bitwise equal\n",
            t_iter, minimum(it_times), maximum(it_times), S.t_total)

    # ---------------- cold setup context -------------------------------------------
    t_hier_cold = Inf
    for _ in 1:2
        empty!(MS.SEMQUAD_MG_CACHE)
        GC.gc()
        t_hier_cold = min(t_hier_cold, @elapsed build_mg())
    end
    tc = Float64[]
    for _ in 1:2
        MS.SEMQUAD_PRESSURE_CACHE[] = nothing; GC.gc()
        push!(tc, @elapsed build_fp(true))
    end
    t_bfbt_cold = minimum(tc)

    # ---------------- tail cross-check ---------------------------------------------
    ufree = sol_p[1:nf]; pfree = sol_p[(nf + 1):nftot]
    u0z = zeros(Float64, 2 * ss.su.N)
    p0z = zeros(Float64, ss.sp.N)
    t_tail = besttime(() -> tail_block(ufree, pfree, sys.fr, sys.gr, sys.Bs,
                                       sys.Bt, sys.Cu, sys.Cp, u0z, p0z,
                                       mfapplyK), 3)

    # ---------------- print the tables ---------------------------------------------
    lvl_nf(k) = mg.geo.nfs[k]
    pl(x, tot) = 100.0 * x / tot

    # --- production rows ---
    @printf("\nproduction rows (same process, shipped defaults):\n")
    @printf("  %-58s %9.4f s\n", "cold full solve (caches cleared, min of 2)", t_cold)
    @printf("  %-58s %9.4f s  (%d it, relres %.3e)\n", "warm full solve (median of 5)", t_warm, nit, relres)
    @printf("  %-58s %9.4f s\n", "assembly (assemble_only=:mg, median of 3)", t_asm)
    @printf("  %-58s %9.4f s  (geometry builds +%d, cached levels %d -> cache hit=%s)\n",
            "hierarchy rebuild, warm geometry cache", t_mg_warm,
            st1.builds - st0.builds, st1.cached_levels,
            st1.builds == st0.builds)
    @printf("  %-58s %9.4f s  (cache reusable=%s, builds +%d, reuses +%d)\n",
            "w-BFBT pressure precond, warm cache", t_bfbt_warm, reused,
            builds_delta, reuses_delta)
    @printf("  %-58s %9.4f s  (%d it, relres %.3e)\n", "FGMRES iteration alone (median of 5)", t_iter, nit_ref, res_ref)
    @printf("  %-58s %9.4f s  (cross-check; remainder below)\n", "final residual check + u,p update, measured", t_tail)

    # --- warm table ---
    # Setup is measured directly on its three production calls; the iteration is
    # the remainder of the measured warm solve, so the table closes by
    # construction.  The replica below distributes that slot by its measured
    # shares (the direct `semquad_mg_saddle_solve` measurement is quoted as a
    # cross-check).
    t_setup = t_asm + t_mg_warm + t_bfbt_warm
    t_iter_slot = t_warm - t_setup
    tail = t_warm - (t_setup + t_iter)
    @printf("\nWARM SOLVE COMPOSITION (one in-tree production step; closes to the measured warm solve)\n")
    @printf("  %-46s %9.4f s %7.1f %%\n", "assembly (reduced MF geometry)", t_asm, pl(t_asm, t_warm))
    @printf("  %-46s %9.4f s %7.1f %%\n", "hierarchy build (warm: geometry cached)", t_mg_warm, pl(t_mg_warm, t_warm))
    @printf("  %-46s %9.4f s %7.1f %%\n", "w-BFBT build (warm: cache hit)", t_bfbt_warm, pl(t_bfbt_warm, t_warm))
    @printf("  %-46s %9.4f s %7.1f %%\n", "setup subtotal", t_setup, pl(t_setup, t_warm))
    @printf("  %-46s %9.4f s %7.1f %%\n", "FGMRES iterations (slot = warm - setup)", t_iter_slot, pl(t_iter_slot, t_warm))
    @printf("  %-46s %9.4f s %7.1f %%\n", "WARM SOLVE (measured, median of 5)", t_warm, 100.0)
    @printf("  %-46s %9.4f s\n", "sum of the three setup rows + iteration", t_setup + t_iter_slot)
    @printf("  cross-check: iteration alone (median of 5) %.4f s (%+.1f %% vs the slot); measured finalisation %.4f s\n",
            t_iter, 100.0 * (t_iter - t_iter_slot) / t_iter_slot, t_tail)

    # --- iteration table ---
    fsc = t_iter_slot / S.t_total
    t_app = S.t_applyK + S.t_copy + S.t_Bt + S.t_Bs
    t_pre = S.t_vcycle + S.t_Bszc + S.t_pblock
    t_fg = S.t_total - t_app - t_pre
    fg_sum = FG.t_mgs + FG.t_norm + FG.t_hess + FG.t_true + FG.t_copy + FG.t_restart
    fgsc = fg_sum > 0 ? t_fg / fg_sum : 0.0
    @printf("\nITERATION COMPOSITION (per outer iteration; replica scaled by %.4f to the warm-solve iteration slot)\n", fsc)
    @printf("  %-46s %9.3f ms %7.1f %% %9.3f ms/it\n", "V-cycle", 1e3 * S.t_vcycle * fsc,
            pl(S.t_vcycle, S.t_total), 1e3 * S.t_vcycle * fsc / nit)
    @printf("  %-46s %9.3f ms %7.1f %% %9.3f ms/it\n", "Bs * zc (continuity of the correction)", 1e3 * S.t_Bszc * fsc,
            pl(S.t_Bszc, S.t_total), 1e3 * S.t_Bszc * fsc / nit)
    @printf("  %-46s %9.3f ms %7.1f %% %9.3f ms/it\n", "w-BFBT block", 1e3 * S.t_pblock * fsc,
            pl(S.t_pblock, S.t_total), 1e3 * S.t_pblock * fsc / nit)
    @printf("  %-46s %9.3f ms %7.1f %% %9.3f ms/it\n", "  F1 \\ rp (first chol(S1) solve)", 1e3 * S.t_f1a * fsc,
            pl(S.t_f1a, S.t_total), 1e3 * S.t_f1a * fsc / nit)
    @printf("  %-46s %9.3f ms %7.1f %% %9.3f ms/it\n", "  S2 apply (matrix-free)", 1e3 * S.t_s2 * fsc,
            pl(S.t_s2, S.t_total), 1e3 * S.t_s2 * fsc / nit)
    @printf("  %-46s %9.3f ms %7.1f %% %9.3f ms/it\n", "  F1 \\ rp (second) + scaling", 1e3 * S.t_f1b * fsc,
            pl(S.t_f1b, S.t_total), 1e3 * S.t_f1b * fsc / nit)
    @printf("  %-46s %9.3f ms %7.1f %% %9.3f ms/it\n", "outer operator apply A v", 1e3 * t_app * fsc,
            pl(t_app, S.t_total), 1e3 * t_app * fsc / nit)
    @printf("  %-46s %9.3f ms %7.1f %% %9.3f ms/it\n", "  MF velocity apply K", 1e3 * S.t_applyK * fsc,
            pl(S.t_applyK, S.t_total), 1e3 * S.t_applyK * fsc / nit)
    @printf("  %-46s %9.3f ms %7.1f %% %9.3f ms/it\n", "  Bt * p", 1e3 * S.t_Bt * fsc,
            pl(S.t_Bt, S.t_total), 1e3 * S.t_Bt * fsc / nit)
    @printf("  %-46s %9.3f ms %7.1f %% %9.3f ms/it\n", "  Bs * u", 1e3 * S.t_Bs * fsc,
            pl(S.t_Bs, S.t_total), 1e3 * S.t_Bs * fsc / nit)
    @printf("  %-46s %9.3f ms %7.1f %% %9.3f ms/it\n", "  wA copy", 1e3 * S.t_copy * fsc,
            pl(S.t_copy, S.t_total), 1e3 * S.t_copy * fsc / nit)
    @printf("  %-46s %9.3f ms %7.1f %% %9.3f ms/it\n", "FGMRES own work", 1e3 * t_fg * fsc,
            pl(t_fg, S.t_total), 1e3 * t_fg * fsc / nit)
    @printf("  %-46s %9.3f ms %7.1f %% %9.3f ms/it\n", "  MGS orthogonalisation", 1e3 * FG.t_mgs * fgsc * fsc,
            pl(FG.t_mgs * fgsc, S.t_total), 1e3 * FG.t_mgs * fgsc * fsc / nit)
    @printf("  %-46s %9.3f ms %7.1f %% %9.3f ms/it\n", "  true residual (W[1:k] y, r-dx, norm)", 1e3 * FG.t_true * fgsc * fsc,
            pl(FG.t_true * fgsc, S.t_total), 1e3 * FG.t_true * fgsc * fsc / nit)
    @printf("  %-46s %9.3f ms %7.1f %% %9.3f ms/it\n", "  norm(w) + basis update", 1e3 * FG.t_norm * fgsc * fsc,
            pl(FG.t_norm * fgsc, S.t_total), 1e3 * FG.t_norm * fgsc * fsc / nit)
    @printf("  %-46s %9.3f ms %7.1f %% %9.3f ms/it\n", "  Hessenberg + Givens", 1e3 * FG.t_hess * fgsc * fsc,
            pl(FG.t_hess * fgsc, S.t_total), 1e3 * FG.t_hess * fgsc * fsc / nit)
    @printf("  %-46s %9.3f ms %7.1f %% %9.3f ms/it\n", "  Z/W copies", 1e3 * FG.t_copy * fgsc * fsc,
            pl(FG.t_copy * fgsc, S.t_total), 1e3 * FG.t_copy * fgsc * fsc / nit)
    @printf("  %-46s %9.3f ms %7.1f %% %9.3f ms/it\n", "  restart residual b-Ax + x += Z y", 1e3 * FG.t_restart * fgsc * fsc,
            pl(FG.t_restart * fgsc, S.t_total), 1e3 * FG.t_restart * fgsc * fsc / nit)
    @printf("  %-46s %9.3f ms %7.1f %% %9.3f ms/it\n", "ITERATION SLOT (warm solve - setup)", 1e3 * t_iter_slot, 100.0, 1e3 * t_iter_slot / nit)
    @printf("  counts per iteration: 1 MF apply (K), 1 V-cycle, 1 Bs zc, 2 F1 solves, 1 S2 apply\n")

    # --- V-cycle table ---
    t_fine_leaf = V.t_leaf[nlev]
    t_fine_cu = V.t_cug[nlev] + V.t_cus[nlev]
    t_fine = t_fine_leaf + t_fine_cu
    t_coarse_leaf = sum(V.t_leaf[2:(nlev - 1)])
    t_coarse_cu = sum(V.t_cug[2:(nlev - 1)]) + sum(V.t_cus[2:(nlev - 1)])
    t_coarse = t_coarse_leaf + t_coarse_cu
    t_vec = sum(V.t_sub) + sum(V.t_jac)
    t_trans = sum(V.t_restr) + sum(V.t_prol)
    t_rest = V.t_coarse + sum(V.t_axpy) + sum(V.t_fill)
    t_acct = t_fine + t_coarse + t_vec + t_trans + t_rest
    vscale = (S.t_vcycle * fsc / nit) / V.t_total
    @printf("\nV-CYCLE COMPOSITION (one cycle, per-cycle ms; replica scaled by %.4f to the measured V-cycle slot)\n", vscale)
    @printf("  %-46s %9.3f ms %7.1f %%  (%d applies)\n", "finest-level applies (the leaf apply)", 1e3 * t_fine * vscale,
            pl(t_fine, V.t_total), V.n_apply[nlev] ÷ REPEATS)
    @printf("  %-46s %9.3f ms %7.1f %%\n", "  of which leaf phase (kernel+scatter)", 1e3 * t_fine_leaf * vscale,
            pl(t_fine_leaf, V.t_total))
    @printf("  %-46s %9.3f ms %7.1f %%\n", "  of which Cu gather + Cu' scatter", 1e3 * t_fine_cu * vscale,
            pl(t_fine_cu, V.t_total))
    @printf("  %-46s %9.3f ms %7.1f %%  (%d applies)\n", "coarse-level applies", 1e3 * t_coarse * vscale,
            pl(t_coarse, V.t_total), n_app_coarse ÷ REPEATS)
    @printf("  %-46s %9.3f ms %7.1f %%\n", "  of which leaf phase (kernel+scatter)", 1e3 * t_coarse_leaf * vscale,
            pl(t_coarse_leaf, V.t_total))
    @printf("  %-46s %9.3f ms %7.1f %%\n", "  of which Cu gather + Cu' scatter", 1e3 * t_coarse_cu * vscale,
            pl(t_coarse_cu, V.t_total))
    @printf("  %-46s %9.3f ms %7.1f %%\n", "transfers (restriction + prolongation)", 1e3 * t_trans * vscale,
            pl(t_trans, V.t_total))
    @printf("  %-46s %9.3f ms %7.1f %%\n", "smoothing vector ops (b-Ax + Jacobi)", 1e3 * t_vec * vscale,
            pl(t_vec, V.t_total))
    @printf("  %-46s %9.3f ms %7.1f %%\n", "coarse Cholesky + x += py + fill!", 1e3 * t_rest * vscale,
            pl(t_rest, V.t_total))
    @printf("  %-46s %9.3f ms %7.1f %%\n", "region/fork overhead + unaccounted", 1e3 * (V.t_total - t_acct) * vscale,
            pl(V.t_total - t_acct, V.t_total))
    @printf("  %-46s %9.3f ms %7.1f %%\n", "V-CYCLE (replica total)", 1e3 * V.t_total * vscale, 100.0)

    # --- per-level V-cycle ---
    @printf("\n  per-level V-cycle (ms per cycle, replica mean of %d):\n", REPEATS)
    @printf("  %5s %8s %7s %7s %8s %8s %8s %7s %7s %8s %8s %6s\n",
            "level", "dofs", "applies", "leaf", "Cu", "b-Ax", "jacobi",
            "restr", "prol", "axpy", "total", "thr")
    for k in nlev:-1:1
        tot = V.t_leaf[k] + V.t_cug[k] + V.t_cus[k] + V.t_sub[k] +
              V.t_jac[k] + V.t_restr[k] + V.t_prol[k] + V.t_axpy[k] +
              V.t_fill[k] + (k == 1 ? V.t_coarse : 0.0)
        @printf("  %5d %8d %7d %7.3f %8.3f %8.3f %8.3f %7.3f %7.3f %8.3f %8.3f %6s\n",
                k, lvl_nf(k), V.n_apply[k] ÷ REPEATS, 1e3 * V.t_leaf[k],
                1e3 * (V.t_cug[k] + V.t_cus[k]), 1e3 * V.t_sub[k],
                1e3 * V.t_jac[k], 1e3 * V.t_restr[k], 1e3 * V.t_prol[k],
                1e3 * V.t_axpy[k], 1e3 * tot,
                k == 1 ? "-" : (MS._semquad_mg_threaded_level(mg, k) ? "yes" : "no"))
    end

    # --- residual history and headroom ---
    hist = FG.hist[1:nit_ref]
    @printf("\nRESIDUAL HISTORY (true relres after k iterations, replica = production bitwise)\n")
    @printf("  initial ||b||/||b|| = %.3e, first iteration %.3e, last %.3e\n",
            FG.r0, hist[1], hist[nit_ref])
    idx = unique(vcat([1, 2, 3, 5, 10, 20], collect(20:10:max(20, nit_ref - 1)), [nit_ref]))
    for k in idx
        k > nit_ref && continue
        @printf("  k=%3d  relres=%.4e\n", k, hist[k])
    end
    hes = FG.hest[1:nit_ref]
    @printf("  Hessenberg estimate |g[k+1]|/||b|| at k=1,10,40,last: %.3e, %.3e, %.3e, %.3e\n",
            hes[1], hes[min(10, nit_ref)], hes[min(40, nit_ref)], hes[nit_ref])
    cross(x) = (i = findfirst(v -> v < x, hist); i === nothing ? 0 : i)
    @printf("  iterations to cross 0.9 / 0.5 / 0.1 / 1e-3 / 1e-6: %d / %d / %d / %d / %d\n",
            cross(0.9), cross(0.5), cross(0.1), cross(1e-3), cross(1e-6))
    @printf("  contraction per iteration by bucket:\n")
    for (lo, hi) in ((1, 10), (11, 20), (21, 30), (31, 40), (41, 50),
                     (51, 60), (61, nit_ref))
        lo > nit_ref && continue
        hi = min(hi, nit_ref)
        @printf("    k=%3d-%3d  rate=%.4f\n", lo, hi,
                (hist[hi] / hist[lo])^(1.0 / (hi - lo + 1)))
    end
    k_last = max(1, nit_ref - 10)
    rho = (hist[nit_ref] / hist[k_last])^(1.0 / (nit_ref - k_last))
    k_rho = hist[1] > TOL ? max(1, ceil(Int, log(TOL / hist[1]) / log(rho))) :
            nit_ref
    @printf("  asymptotic rate over the last %d iterations: rho = %.4f (residual x %.3f per iteration)\n",
            nit_ref - k_last, rho, rho)
    @printf("  first-iteration drop: 1 -> %.3e (%.2fx); average rate to tol: %.4f\n",
            hist[1], 1.0 / hist[1], (TOL)^(1.0 / nit_ref))
    @printf("  at the last-10 rate from iteration 1 the solve would need %d iterations (cut of %d, %.0f %%)\n",
            k_rho, nit_ref - k_rho, 100.0 * (nit_ref - k_rho) / nit_ref)
    @printf("  hypothetical sustained rates from the observed first residual:\n")
    for rh in (0.90, 0.85, 0.80, 0.75, 0.70)
        kk = ceil(Int, log(TOL / hist[1]) / log(rh))
        @printf("    rho=%.2f -> %3d iterations\n", rh, kk)
    end

    # --- cold setup context ---
    @printf("\nCOLD SETUP CONTEXT (a cache miss: what the warm solve does NOT pay)\n")
    @printf("  %-46s %9.4f s\n", "hierarchy build (make_semquad_mg, cold, min of 2)", t_hier_cold)
    @printf("  %-46s %9.4f s\n", "w-BFBT build (cold, min of 2)", t_bfbt_cold)
    @printf("  %-46s %9.4f s\n", "assembly (unchanged by the cache state)", t_asm)
    @printf("  %-46s %9.4f s\n", "FGMRES iteration (same iteration count)", t_iter)
    @printf("  cold solve measured: %.4f s | sum hierarchy+BFBT+assembly+iteration slot = %.4f s\n",
            t_cold, t_hier_cold + t_bfbt_cold + t_asm + t_iter_slot)
    @printf("  cold solve: %d it, relres %.3e\n", nit_cold, res_cold)

    # --- machine-readable summary ---
    println("\nSUMMARY")
    @printf("case=%s thr=%d mg_threads=%d nf=%d nfp=%d nlev=%d nit=%d relres=%.6e\n",
            CASE, THR, mg.threads, nf, nfp, nlev, nit, relres)
    @printf("warm=%.4f warm_min=%.4f warm_max=%.4f cold=%.4f\n",
            t_warm, minimum(warm), maximum(warm), t_cold)
    @printf("setup=%.4f iter_slot=%.4f iter_direct=%.4f iter_min=%.4f iter_max=%.4f direct_minus_slot=%.4f\n",
            t_setup, t_iter_slot, t_iter, minimum(it_times), maximum(it_times),
            t_iter - t_iter_slot)
    @printf("asm=%.4f hier_warm=%.4f bfbt_warm=%.4f tail_measured=%.4f\n",
            t_asm, t_mg_warm, t_bfbt_warm, t_tail)
    @printf("hier_cold=%.4f bfbt_cold=%.4f\n", t_hier_cold, t_bfbt_cold)
    @printf("vcycle_ms=%.4f vcycle_prod_ms=%.4f fine_applies=%d coarse_applies=%d\n",
            1e3 * V.t_total, 1e3 * v_prod, V.n_apply[nlev] ÷ REPEATS,
            n_app_coarse ÷ REPEATS)
    @printf("fine_ms=%.4f fine_leaf_ms=%.4f fine_cu_ms=%.4f coarse_ms=%.4f coarse_leaf_ms=%.4f coarse_cu_ms=%.4f\n",
            1e3 * t_fine, 1e3 * t_fine_leaf, 1e3 * t_fine_cu, 1e3 * t_coarse,
            1e3 * t_coarse_leaf, 1e3 * t_coarse_cu)
    @printf("vec_ms=%.4f transfers_ms=%.4f rest_ms=%.4f region_ms=%.4f\n",
            1e3 * t_vec, 1e3 * t_trans, 1e3 * t_rest, 1e3 * (V.t_total - t_acct))
    @printf("iter_ms_per_it=%.4f iter_direct_ms_per_it=%.4f vcycle=%.4f Bszc=%.4f pblock=%.4f f1a=%.4f s2=%.4f f1b=%.4f apply=%.4f applyK=%.4f Bt=%.4f Bs=%.4f fgmres=%.4f\n",
            1e3 * t_iter_slot / nit, 1e3 * t_iter / nit, 1e3 * S.t_vcycle * fsc,
            1e3 * S.t_Bszc * fsc,
            1e3 * S.t_pblock * fsc, 1e3 * S.t_f1a * fsc, 1e3 * S.t_s2 * fsc,
            1e3 * S.t_f1b * fsc, 1e3 * t_app * fsc, 1e3 * S.t_applyK * fsc,
            1e3 * S.t_Bt * fsc, 1e3 * S.t_Bs * fsc, 1e3 * t_fg * fsc)
    @printf("mgs=%.4f true=%.4f norm=%.4f hess=%.4f copies=%.4f restart=%.4f\n",
            1e3 * FG.t_mgs * fgsc * fsc, 1e3 * FG.t_true * fgsc * fsc,
            1e3 * FG.t_norm * fgsc * fsc, 1e3 * FG.t_hess * fgsc * fsc,
            1e3 * FG.t_copy * fgsc * fsc, 1e3 * FG.t_restart * fgsc * fsc)
    @printf("rho_last=%.5f k_rho=%d first_drop=%.3e\n", rho, k_rho, hist[1])
    @printf("fsc=%.5f vscale=%.5f\n", fsc, vscale)

    rm(tmpdir; force=true, recursive=true)
    return nothing
end

main()
