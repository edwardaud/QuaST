#!/usr/bin/env julia
# =============================================================================
# profile_mg_vcycle.jl -- component-level profile of one matrix-free (MF) MG
# V-cycle and of one full `linear=:mg` saddle solve on the production tree.
#
#   julia --project=. benchmarks/profile_mg_vcycle.jl
#   julia --project=. --threads=8 benchmarks/profile_mg_vcycle.jl
#
# The state is a read-only *copy* of a production checkpoint (default
# out_julia/_s7/case/pre_step6131.h5, the `F` state of docs/dev/PERFORMANCE.md
# §3.3: 8 188 leaves, p=2, 69 388 saddle dofs, eta contrast 6.09e6) with the
# parameter card of examples/case_realistic_2d.jl rebuilt from the
# checkpoint's own `config` attribute (composite isostress rheology, Ra=1e5,
# H=10, free-slip, deviatoric=false, boussinesq mass form).
#
# The solver is not modified and nothing here is on a production path.  The
# components are measured in two ways:
#
#   (a) a timer around the production entry points themselves
#       (`assemble_only`, `make_semquad_mg`, `semquad_mg_pressure_precond`,
#       `semquad_mg_apply`, `semquad_mg_saddle_solve`), and
#   (b) a *replica* of the two loops the production code does not expose
#       (`_semquad_mg_vcycle!` and `semquad_mg_saddle_solve`), which calls the
#       same module internals in the same order with a `time_ns()` around each
#       primitive.
#
# Both replicas are checked **bitwise** against the production functions on the
# same inputs (`z_replica == semquad_mg_apply(mg, b)`, `sol_replica ==
# sol_production`, `wP_replica == _semquad_pressure_block!`), and the script
# aborts on any mismatch.  The split therefore describes the production
# arithmetic, not a re-implementation of it.
#
# Environment (all optional):
#   MG_PROFILE_CKPT     checkpoint path (copied to a temp dir, opened "r")
#   MG_PROFILE_TOL      outer FGMRES tolerance (default 1e-6)
#   MG_PROFILE_MG_THR   `mg_threads` of the V-cycle / outer apply (default 1)
#   MG_PROFILE_REPEATS  samples of the single-V-cycle timing (default 7)
#   MG_PROFILE_CGVAR    "1" = additionally time the vel_solver=:cg variant
#   MG_PROFILE_AB       "1" = additionally measure the coarsen_factor=2.0
#                       hierarchy A/B (changes the preconditioner; opt-in)
#
# The thread count comes from the Julia process (`--threads=N`); BLAS is pinned
# to one thread, as in every number in docs/.
# =============================================================================

using QuaST, Printf, LinearAlgebra, SparseArrays, Random

const MS = QuaST

const REPO = normpath(joinpath(@__DIR__, ".."))
const CKPT = get(ENV, "MG_PROFILE_CKPT",
                 joinpath(REPO, "out_julia", "_s7", "case", "pre_step6131.h5"))
const TOL = parse(Float64, get(ENV, "MG_PROFILE_TOL", "1.0e-6"))
const MGTHR = parse(Int, get(ENV, "MG_PROFILE_MG_THR", "1"))
const REPEATS = parse(Int, get(ENV, "MG_PROFILE_REPEATS", "7"))
const CGVARIANT = lowercase(strip(get(ENV, "MG_PROFILE_CGVAR", "0"))) in
                  ("1", "true", "yes", "on")
const AB = lowercase(strip(get(ENV, "MG_PROFILE_AB", "0"))) in
           ("1", "true", "yes", "on")
const Z2 = (x, y) -> 0.0

BLAS.set_num_threads(1)

@inline el(t0::UInt64) = 1.0e-9 * (time_ns() - t0)
tmin(f, n::Int) = (f(); minimum([@elapsed f() for _ in 1:n]))

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

# `phases = z0,width,slope,Tref,drho,B;...`; only these fields enter
# `semquad_phase_buoyancy`, so the six-argument PhaseTransition is enough
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
    t_pre::Vector{Float64}; t_pre_ap::Vector{Float64}; t_pre_vec::Vector{Float64}
    t_post::Vector{Float64}; t_post_ap::Vector{Float64}; t_post_vec::Vector{Float64}
    t_res::Vector{Float64}; t_res_ap::Vector{Float64}; t_res_vec::Vector{Float64}
    t_restr::Vector{Float64}; t_prol::Vector{Float64}; t_axpy::Vector{Float64}
    t_fill::Vector{Float64}
    t_coarse::Float64
    n_pre::Vector{Int}; n_post::Vector{Int}; n_coarse::Int
    t_total::Float64
end
VCProf(nlev::Int) = begin
    z() = zeros(Float64, nlev)
    VCProf(z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(), z(),
           0.0, zeros(Int, nlev), zeros(Int, nlev), 0, 0.0)
end
function addv!(a::VCProf, b::VCProf)
    for f in (:t_pre, :t_pre_ap, :t_pre_vec, :t_post, :t_post_ap, :t_post_vec,
              :t_res, :t_res_ap, :t_res_vec, :t_restr, :t_prol, :t_axpy,
              :t_fill)
        getfield(a, f) .+= getfield(b, f)
    end
    a.t_coarse += b.t_coarse
    a.n_pre .+= b.n_pre; a.n_post .+= b.n_post; a.n_coarse += b.n_coarse
    a.t_total += b.t_total
    return a
end
function scale!(a::VCProf, s::Float64)
    for f in (:t_pre, :t_pre_ap, :t_pre_vec, :t_post, :t_post_ap, :t_post_vec,
              :t_res, :t_res_ap, :t_res_vec, :t_restr, :t_prol, :t_axpy,
              :t_fill)
        getfield(a, f) .*= s
    end
    a.t_coarse *= s; a.t_total *= s
    return a
end

mutable struct SolveProf
    t_applyK::Float64; t_Bt::Float64; t_Bs::Float64; t_copy::Float64
    t_pre::Float64; t_vcycle::Float64; t_Bszc::Float64; t_pblock::Float64
    t_f1a::Float64; t_s2::Float64; t_f1b::Float64
    n_applyA::Int; n_pre::Int; n_vcycle::Int; n_f1::Int; n_s2::Int
    t_total::Float64
end
SolveProf() = SolveProf(zeros(Float64, 11)..., 0, 0, 0, 0, 0, 0.0)
function addv!(a::SolveProf, b::SolveProf)
    for f in (:t_applyK, :t_Bt, :t_Bs, :t_copy, :t_pre, :t_vcycle, :t_Bszc,
              :t_pblock, :t_f1a, :t_s2, :t_f1b)
        setfield!(a, f, getfield(a, f) + getfield(b, f))
    end
    for f in (:n_applyA, :n_pre, :n_vcycle, :n_f1, :n_s2)
        setfield!(a, f, getfield(a, f) + getfield(b, f))
    end
    a.t_total += b.t_total
    return a
end

# ---------------------------------------------------------------------------
# FGMRES sampled per iteration index k
#
# The orthogonalisation (modified Gram-Schmidt) is O(k) per iteration and the
# true residual is also O(k) (it recombines the k stored `A Z` columns), so the
# cost per iteration grows with k.  This replica of `fgmres_restarted` records
# every component per iteration index so the growth can be read directly; it is
# bitwise equal to the production routine (same operations, same order).
# ---------------------------------------------------------------------------
mutable struct FGProf
    t_mgs::Float64; t_norm::Float64; t_hess::Float64; t_true::Float64
    t_copy::Float64; t_restart::Float64
    mgs_k::Vector{Float64}; norm_k::Vector{Float64}; hess_k::Vector{Float64}
    true_k::Vector{Float64}; pre_k::Vector{Float64}; app_k::Vector{Float64}
    copy_k::Vector{Float64}
    t_total::Float64
end
FGProf(m::Int) = FGProf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, zeros(m), zeros(m),
                        zeros(m), zeros(m), zeros(m), zeros(m), zeros(m), 0.0)

"""`fgmres_restarted` with a timer per component and per iteration index."""
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
        F.t_restart += el(tr)
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
            tk = time_ns()
            z = precond === nothing ? V[:, k] : precond(V[:, k])
            F.pre_k[k] += el(tk)
            tk = time_ns(); Z[:, k] = z; F.copy_k[k] += el(tk); F.t_copy += el(tk)
            tk = time_ns(); w = applyA(z); F.app_k[k] += el(tk)
            tk = time_ns(); W[:, k] = w; F.copy_k[k] += el(tk); F.t_copy += el(tk)
            tk = time_ns()
            for i in 1:k
                vi = view(V, :, i)
                H[i, k] = dot(w, vi)
                axpy!(-H[i, k], vi, w)
            end
            F.mgs_k[k] += el(tk); F.t_mgs += el(tk)
            tk = time_ns()
            H[k+1, k] = norm(w)
            if H[k+1, k] > 1e-14
                @views V[:, k+1] .= w ./ H[k+1, k]
            else
                @views fill!(V[:, k+1], 0.0)
            end
            F.norm_k[k] += el(tk); F.t_norm += el(tk)
            tk = time_ns()
            for i in 1:k-1
                t = cs[i] * H[i, k] + sn[i] * H[i+1, k]
                H[i+1, k] = -sn[i] * H[i, k] + cs[i] * H[i+1, k]
                H[i, k] = t
            end
            den = hypot(H[k, k], H[k+1, k])
            if den == 0.0
                k_used = k
                break
            end
            cs[k] = H[k, k] / den; sn[k] = H[k+1, k] / den
            H[k, k] = cs[k] * H[k, k] + sn[k] * H[k+1, k]
            H[k+1, k] = 0.0
            g[k+1] = -sn[k] * g[k]
            g[k] = cs[k] * g[k]
            k_used = k
            y = H[1:k, 1:k] \ g[1:k]
            F.hess_k[k] += el(tk); F.t_hess += el(tk)
            tk = time_ns()
            mul!(dx, view(W, :, 1:k), y)
            @. dx = r - dx
            true_rel = norm(dx) / nb
            F.true_k[k] += el(tk); F.t_true += el(tk)
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
# one V-cycle, per level, timed
# ---------------------------------------------------------------------------
function smooth_prof!(x::Vector{Float64}, b::Vector{Float64}, mg, lev::Int,
                      ns::Int, V::VCProf, pre::Bool)
    D = mg.D[lev]
    r = mg.sr[lev]
    y = mg.sy[lev]
    nt = mg.threads
    for _ in 1:ns
        t0 = time_ns()
        MS._semquad_mg_apply!(y, x, mg, lev)
        dap = el(t0)
        t0 = time_ns()
        if nt > 1
            MS._semquad_td_sub!(r, b, y, nt)
            MS._semquad_td_jacobi!(x, r, D, mg.omega, nt)
        else
            @. r = b - y
            @. x += mg.omega * r / D
        end
        dvec = el(t0)
        if pre
            V.t_pre_ap[lev] += dap; V.t_pre_vec[lev] += dvec
            V.n_pre[lev] += 1
        else
            V.t_post_ap[lev] += dap; V.t_post_vec[lev] += dvec
            V.n_post[lev] += 1
        end
    end
    return nothing
end

function vcycle_prof!(x::Vector{Float64}, b::Vector{Float64}, mg, lev::Int,
                      V::VCProf)
    if lev == 1
        t0 = time_ns()
        x .= mg.cA \ b
        V.t_coarse += el(t0)
        V.n_coarse += 1
        return nothing
    end
    t0 = time_ns(); fill!(x, 0.0); V.t_fill[lev] += el(t0)
    t0 = time_ns(); smooth_prof!(x, b, mg, lev, mg.pre[lev], V, true)
    V.t_pre[lev] += el(t0)
    r = mg.sc[lev]
    y = mg.sy[lev]
    nt = mg.threads
    t0 = time_ns()
    MS._semquad_mg_apply!(y, x, mg, lev)
    dap = el(t0)
    t0 = time_ns()
    if nt > 1
        MS._semquad_td_sub!(r, b, y, nt)
    else
        @. r = b - y
    end
    dvec = el(t0)
    V.t_res_ap[lev] += dap; V.t_res_vec[lev] += dvec
    V.t_res[lev] += dap + dvec
    t0 = time_ns()
    MS._semquad_mg_restrict!(mg.rcs[lev], mg.geo.Ps[lev - 1], r, nt)
    V.t_restr[lev] += el(t0)
    vcycle_prof!(mg.ecs[lev], mg.rcs[lev], mg, lev - 1, V)
    t0 = time_ns()
    if nt > 1 && !isempty(mg.Pts)
        MS._semquad_mg_prolong!(mg.pys[lev],
                                mg.Pts[lev - 1]::SparseMatrixCSC{Float64,Int},
                                mg.ecs[lev], nt)
    else
        mul!(mg.pys[lev], mg.geo.Ps[lev - 1], mg.ecs[lev])
    end
    V.t_prol[lev] += el(t0)
    t0 = time_ns(); x .+= mg.pys[lev]; V.t_axpy[lev] += el(t0)
    t0 = time_ns(); smooth_prof!(x, b, mg, lev, mg.post[lev], V, false)
    V.t_post[lev] += el(t0)
    return nothing
end

"""`semquad_mg_apply(mg, b)` (mg.cycles V-cycles) with the replica timed."""
function mg_apply_prof!(z::Vector{Float64}, b::AbstractVector{Float64}, mg,
                        V::VCProf)
    bb = Vector{Float64}(b)
    t0 = time_ns()
    vcycle_prof!(z, bb, mg, mg.geo.L + 1, V)
    for _ in 2:mg.cycles
        r = mg.sc[end]
        MS._semquad_mg_apply!(r, z, mg, mg.geo.L + 1)
        rr = bb .- r
        dz = zeros(Float64, length(bb))
        vcycle_prof!(dz, rr, mg, mg.geo.L + 1, V)
        z .+= dz
    end
    V.t_total += el(t0)
    return z
end

# ---------------------------------------------------------------------------
# w-BFBT pressure block, timed per factor
# ---------------------------------------------------------------------------
function pressure_block_prof!(wP::Vector{Float64}, rp::Vector{Float64},
                              rp2::Vector{Float64}, s1::Vector{Float64},
                              s2::Vector{Float64}, Fp, pscale::Float64,
                              nf::Int, S::SolveProf)
    t_all = time_ns()
    if Fp isa NamedTuple && Fp.kind === :pcd
        t0 = time_ns(); rp2 .= Fp.FM \ rp; S.t_f1a += el(t0); S.n_f1 += 1
        t0 = time_ns(); mul!(rp, Fp.FF, rp2); S.t_s2 += el(t0); S.n_s2 += 1
        t0 = time_ns(); rp2 .= Fp.FA \ rp; S.t_f1b += el(t0); S.n_f1 += 1
        @. wP[(nf + 1):end] = -pscale * rp2
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
        t0 = time_ns(); rp2 .= Fp \ rp; S.t_f1a += el(t0); S.n_f1 += 1
        @. wP[(nf + 1):end] = -pscale * rp2
    end
    S.t_pblock += el(t_all)
    return wP
end

# ---------------------------------------------------------------------------
# the outer saddle solve, timed per part
# ---------------------------------------------------------------------------
function saddle_solve_prof(Kred, Bs, Bt, Fp, pscale::Float64,
                           b::Vector{Float64}, mg;
                           tol::Float64=1.0e-10, maxit::Int=500,
                           restart::Int=100, vel_solver::Symbol=:vcycle,
                           vel_tol::Float64=1.0e-2, vel_maxit::Int=10,
                           applyK::Union{Nothing,Function}=nothing,
                           S::SolveProf=SolveProf(), F::FGProf=FGProf(maxit))
    nf = size(Bs, 2); nfp = size(Bs, 1); nftot = nf + nfp
    applyK! = applyK === nothing ? (out, v) -> mul!(out, Kred, v) : applyK
    cgiters = Ref(0)
    wA = zeros(Float64, nftot)
    wP = zeros(Float64, nftot)
    zc = zeros(Float64, nf)
    rp = zeros(Float64, nfp)
    rp2 = zeros(Float64, nfp)
    s1 = zeros(Float64, nf)
    s2 = zeros(Float64, nf)
    ku = zeros(Float64, nf)
    cgwork = MS.SemQuadCGWork(nf)
    # NB: the closure-local timers must not reuse a name of the enclosing
    # scope -- Julia captures and shares it, which silently corrupts the
    # outer clock.
    function applyA(v::Vector{Float64})
        ta = time_ns()
        applyK!(ku, view(v, 1:nf))
        S.t_applyK += el(ta)
        ta = time_ns(); wA[1:nf] .= ku; S.t_copy += el(ta)
        ta = time_ns(); wA[1:nf] .+= Bt * v[(nf + 1):nftot]; S.t_Bt += el(ta)
        ta = time_ns(); wA[(nf + 1):nftot] = Bs * v[1:nf]; S.t_Bs += el(ta)
        S.n_applyA += 1
        return wA
    end
    function precond(v::Vector{Float64})
        t_all = time_ns()
        if vel_solver === :cg
            t1 = time_ns()
            _, nc, _ = MS.semquad_mg_cg(applyK!, v[1:nf], mg; tol=vel_tol,
                                        maxit=vel_maxit, work=cgwork)
            S.t_vcycle += el(t1)
            cgiters[] += nc
            copyto!(zc, cgwork.z)
        else
            t1 = time_ns()
            zc .= MS.semquad_mg_apply(mg, v[1:nf])
            S.t_vcycle += el(t1)
        end
        S.n_vcycle += 1
        t2 = time_ns()
        wP[1:nf] .= zc
        rp .= v[(nf + 1):nftot] .- Bs * zc
        S.t_Bszc += el(t2)
        pressure_block_prof!(wP, rp, rp2, s1, s2, Fp, pscale, nf, S)
        S.n_pre += 1
        S.t_pre += el(t_all)
        return wP
    end
    t_solve = time_ns()
    sol, relres, nit = fgmres_prof(applyA, b; precond=precond, tol=tol,
                                   maxit=maxit, restart=restart,
                                   work=MS._semquad_fgmres_work(
                                       nftot, min(restart, maxit)), F=F)
    S.t_total += el(t_solve)
    return sol, nit, relres, cgiters[]
end

# ---------------------------------------------------------------------------
# setup anatomy: the body of `_semquad_mg_geometry` plus the matrix-free
# per-level operator setup, timed piece by piece
# ---------------------------------------------------------------------------
function setup_anatomy(ss, qt, p::Int, bc::Symbol, deviatoric::Bool,
                       eta_fun::Function, nlev_real::Int)
    A = Dict{String,Any}()
    empty!(MS.SEMQUAD_MG_CACHE)
    t0 = time_ns()
    trees = MS._mg_hierarchy_trees(qt; layers=1, factor=1.0)
    A["coarsen"] = el(t0)
    A["nlev"] = length(trees)
    A["leaves"] = [length(q.leaves) for q in trees]
    nlev = length(trees); L = nlev - 1
    ss_t = Vector{Any}(undef, nlev); Cu_t = Vector{Any}(undef, nlev)
    nfs = zeros(Int, nlev); fr = Vector{Any}(undef, nlev)
    for k in ("t_mk", "t_asm", "t_cu", "t_fr")
        A[k] = zeros(nlev)
    end
    for k in 1:nlev
        q = trees[k]
        t0 = time_ns(); s = make_semstokes(q, p); A["t_mk"][k] = el(t0)
        t0 = time_ns()
        asm = solve_semquad_stokes_var(s, q, (x, y) -> 1.0, Z2, Z2, Z2, Z2;
                                       bc=bc, deviatoric=deviatoric,
                                       assemble_only=true)
        A["t_asm"][k] = el(t0)
        t0 = time_ns()
        Cu = SparseMatrixCSC{Float64,Int}(sparse(asm.Cu))
        A["t_cu"][k] = el(t0)
        ss_t[k] = s; Cu_t[k] = Cu; nfs[k] = size(Cu, 2)
        t0 = time_ns(); fr[k] = MS._mg_free_rows(Cu, 2 * s.su.N)
        A["t_fr"][k] = el(t0)
    end
    A["t_prol"] = zeros(L); A["t_pmul"] = zeros(L)
    for k in 1:L
        t0 = time_ns()
        Pfull = MS._mg_prolongation_full(ss_t[k + 1].su, ss_t[k].su, p)
        A["t_prol"][k] = el(t0)
        t0 = time_ns()
        P = Pfull[fr[k + 1], :] * Cu_t[k]
        P = SparseMatrixCSC{Float64,Int}(sparse(P))
        A["t_pmul"][k] = el(t0)
    end
    for k in ("t_kop", "t_eta", "t_diag")
        A[k] = zeros(nlev)
    end
    ops = Vector{Any}(undef, nlev); etas = Vector{Any}(undef, nlev)
    works = Vector{Any}(undef, nlev)
    for k in 1:nlev
        t0 = time_ns()
        opk = semquad_kop(ss_t[k], trees[k]; deviatoric=deviatoric)
        A["t_kop"][k] = el(t0)
        t0 = time_ns()
        etas[k] = semquad_eta_field(opk, eta_fun)
        A["t_eta"][k] = el(t0)
        ops[k] = opk
        works[k] = SemQuadKWork(opk.n, opk.nq, 2 * opk.N)
    end
    for k in 1:nlev
        t0 = time_ns()
        MS._semquad_diag_reduced(ops[k]::SemQuadKOp,
                                 Cu_t[k]::SparseMatrixCSC{Float64,Int},
                                 etas[k]::Vector{Float64}; mode=:constrained)
        A["t_diag"][k] = el(t0)
    end
    n1 = nfs[1]
    A1 = Matrix{Float64}(undef, n1, n1)
    t0 = time_ns()
    MS._semquad_dense_operator!(A1, ops[1]::SemQuadKOp,
                                Cu_t[1]::SparseMatrixCSC{Float64,Int},
                                etas[1]::Vector{Float64},
                                works[1]::SemQuadKWork)
    A["t_dense"] = el(t0)
    t0 = time_ns()
    cA = cholesky(Symmetric(A1); check=false)
    A["t_chol"] = el(t0)
    A["cA_ok"] = issuccess(cA)
    A["sum"] = sum(A["coarsen"]) + sum(A["t_mk"]) + sum(A["t_asm"]) +
               sum(A["t_cu"]) + sum(A["t_fr"]) + sum(A["t_prol"]) +
               sum(A["t_pmul"]) + sum(A["t_kop"]) + sum(A["t_eta"]) +
               sum(A["t_diag"]) + A["t_dense"] + A["t_chol"]
    length(trees) == nlev_real ||
        error("anatomy rebuilt $(length(trees)) levels, the real hierarchy has $nlev_real")
    A["t_all_mk"] = sum(A["t_mk"]); A["t_all_asm"] = sum(A["t_asm"])
    A["t_all_eta"] = sum(A["t_eta"]); A["t_all_diag"] = sum(A["t_diag"])
    return A
end

"""Elementwise minimum of two anatomy passes (a pause can only inflate a row)."""
function min_anatomy!(A::Dict{String,Any}, B::Dict{String,Any})
    for k in ("coarsen", "t_dense", "t_chol")
        A[k] = min(A[k], B[k])
    end
    for k in ("t_mk", "t_asm", "t_cu", "t_fr", "t_prol", "t_pmul", "t_kop",
              "t_eta", "t_diag")
        A[k] .= min.(A[k], B[k])
    end
    A["sum"] = sum(A["coarsen"]) + sum(A["t_mk"]) + sum(A["t_asm"]) +
               sum(A["t_cu"]) + sum(A["t_fr"]) + sum(A["t_prol"]) +
               sum(A["t_pmul"]) + sum(A["t_kop"]) + sum(A["t_eta"]) +
               sum(A["t_diag"]) + A["t_dense"] + A["t_chol"]
    A["t_all_mk"] = sum(A["t_mk"]); A["t_all_asm"] = sum(A["t_asm"])
    A["t_all_eta"] = sum(A["t_eta"]); A["t_all_diag"] = sum(A["t_diag"])
    return A
end

# ===========================================================================
function main()
    nthreads = Threads.nthreads()
    println("="^100)
    @printf("MF-MG component profile   Julia threads=%d   mg_threads request=%d   BLAS threads=1\n",
            nthreads, MGTHR)
    @printf("checkpoint=%s\nouter tolerance=%.1e   V-cycle samples=%d   cycles=1\n",
            CKPT, TOL, REPEATS)
    println("="^100)

    # ---------------- state ------------------------------------------------
    isfile(CKPT) || error("checkpoint not found: $CKPT")
    tmpdir = mktempdir(; cleanup=false)
    copyto = joinpath(tmpdir, basename(CKPT))
    cp(CKPT, copyto; force=true)
    chmod(copyto, 0o444)
    t0 = time_ns(); st = load_semquad_state(copyto); t_load = el(t0)
    @printf("\ncheckpoint copy: %s (%d bytes, mode 444, opened read-only)  loaded in %.2f s\n",
            copyto, filesize(copyto), t_load)
    qt = st.qt; p = st.p; theta = st.theta; ux = st.ux; uz = st.uz
    pr = params_from_config(st.config)
    ph = phases_from_config(st.config)
    ss = make_semstokes(qt, p)
    length(theta) == ss.su.N ||
        error("checkpoint theta has $(length(theta)) entries, velocity space has $(ss.su.N)")
    length(ux) == ss.su.N && length(uz) == ss.su.N ||
        error("checkpoint velocity arrays do not match the velocity space")
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
    @printf("state: %d leaves, p=%d, velocity dofs %d, pressure dofs %d, %d tree levels\n",
            length(qt.leaves), p, ss.su.N, ss.sp.N,
            length(unique(lf[1] for lf in qt.leaves)))
    @printf("rheology: composite isostress, Ra=%g, H=%g, %d phase transitions, bc=free\n",
            pr.Ra, pr.H, length(ph))
    emin = Inf; emax = 0.0
    for (x, y) in ss.su.coords
        e = eta_fun(x, y); emin = min(emin, e); emax = max(emax, e)
    end
    @printf("eta contrast on the velocity nodes: %.3e .. %.3e = %.3e x\n", emin,
            emax, emax / emin)

    # ---------------- the static saddle blocks ----------------------------
    t_asm_alone = tmin(() -> solve_semquad_stokes_var(ss, qt, eta_fun, Z2, fz,
                                                      Z2, Z2; bc=:free,
                                                      deviatoric=false,
                                                      mass_form=:boussinesq,
                                                      assemble_only=true), 2)
    sys = solve_semquad_stokes_var(ss, qt, eta_fun, Z2, fz, Z2, Z2; bc=:free,
                                   deviatoric=false, mass_form=:boussinesq,
                                   assemble_only=true)
    nf = size(sys.Bs, 2); nfp = size(sys.Bs, 1); nftot = nf + nfp
    @printf("\nassemble_only (direct-style, keeps K): %.3f s (best of 2)\n",
            t_asm_alone)
    @printf("reduced saddle: nf=%d nfp=%d nftot=%d  nnz(Kred)=%d (%.0f/row)  nnz(Bs)=%d\n",
            nf, nfp, nftot, nnz(sys.Kred), nnz(sys.Kred) / nf, nnz(sys.Bs))

    # ---------------- the full production MG solve ------------------------
    # This script reproduces the *historical* profile in docs/dev/MG.md:
    # coarsen_factor=1.0 and the explicit taper.  The production default shipped
    # after that profile is `mg_coarsen_factor=nothing` (-> 1.5) with the
    # uniform auto schedule; set `MG_PROFILE_NEW=1` to measure the new default.
    newdef = lowercase(strip(get(ENV, "MG_PROFILE_NEW", "0"))) in
             ("1", "true", "yes", "on")
    mg_kw = (; mg_operator=:matrixfree, mg_vel_solver=:vcycle,
             mg_pressure_cache=false, mg_tol=TOL, mg_threads=MGTHR,
             mg_smoothing=newdef ? :auto : :taper,
             mg_coarsen_factor=newdef ? 1.5 : 1.0,
             mg_pressure_mode=:auto)
    empty!(MS.SEMQUAD_MG_CACHE); MS.SEMQUAD_PRESSURE_CACHE[] = nothing
    t_first = @elapsed solve_semquad_stokes_var(ss, qt, eta_fun, Z2, fz, Z2,
                                                Z2; bc=:free, deviatoric=false,
                                                mass_form=:boussinesq,
                                                linear=:mg, mg_kw...)
    @printf("\nfirst-ever MG solve in this process (JIT + first touch, discarded): %.3f s\n",
            t_first)
    t_full = Inf; nit_full = 0; res_full = 0.0
    for _ in 1:3
        empty!(MS.SEMQUAD_MG_CACHE); MS.SEMQUAD_PRESSURE_CACHE[] = nothing
        t = @elapsed solve_semquad_stokes_var(ss, qt, eta_fun, Z2, fz, Z2, Z2;
                                              bc=:free, deviatoric=false,
                                              mass_form=:boussinesq,
                                              linear=:mg, mg_kw...)
        t < t_full && (t_full = t)
        nit_full = LAST_SEMSTOKES_SCHUR_ITERS[]
        res_full = LAST_SEMSTOKES_RESIDUAL[]
    end
    @printf("cold-cache full MG solve: %.3f s (best of 3), %d outer iterations, true relres %.2e\n",
            t_full, nit_full, res_full)
    # warm caches: same tree again
    empty!(MS.SEMQUAD_MG_CACHE); MS.SEMQUAD_PRESSURE_CACHE[] = nothing
    solve_semquad_stokes_var(ss, qt, eta_fun, Z2, fz, Z2, Z2; bc=:free,
                             deviatoric=false, mass_form=:boussinesq,
                             linear=:mg, mg_pressure_cache=true, mg_tol=TOL,
                             mg_operator=:matrixfree, mg_vel_solver=:vcycle,
                             mg_threads=MGTHR)
    t_warm = @elapsed solve_semquad_stokes_var(ss, qt, eta_fun, Z2, fz, Z2, Z2;
                                               bc=:free, deviatoric=false,
                                               mass_form=:boussinesq,
                                               linear=:mg, mg_pressure_cache=true,
                                               mg_tol=TOL,
                                               mg_operator=:matrixfree,
                                               mg_vel_solver=:vcycle,
                                               mg_threads=MGTHR)
    @printf("warm-cache full MG solve (hierarchy + BFBT reused): %.3f s\n",
            t_warm)

    # ---------------- hierarchy and pressure block ------------------------
    mgopt = (; cycles=1, smoother=:jacobi,
               coarse_operator=:rediscretize, operator=:matrixfree,
               coarsen_factor=newdef ? 1.5 : 1.0,
               smoothing=newdef ? :auto : :taper,
               fine_Cu=newdef ? sys.Cu : nothing,
               mg_threads=MGTHR)
    empty!(MS.SEMQUAD_MG_CACHE)
    t_mg_cold = Inf
    for _ in 1:2                                     # cold cache every time
        empty!(MS.SEMQUAD_MG_CACHE)
        t = @elapsed MS.make_semquad_mg(ss, qt, p, :free, false, eta_fun,
                                        nothing; mgopt...)
        t < t_mg_cold && (t_mg_cold = t)
    end
    mg = MS.make_semquad_mg(ss, qt, p, :free, false, eta_fun, nothing;
                            mgopt...)
    t_mg_warm = tmin(() -> MS.make_semquad_mg(ss, qt, p, :free, false, eta_fun,
                                              nothing; mgopt...), 5)
    L = mg.geo.L; nlev = L + 1
    mg.smoother === :jacobi || error("this profile assumes the Jacobi smoother")
    mg.operator === :matrixfree || error("this profile assumes operator=:matrixfree")
    @printf("\nhierarchy: %d levels, dofs per level (coarse first) %s\n", nlev,
            join(mg.geo.nfs, ", "))
    @printf("  make_semquad_mg, cold cache : %.3f s (best of 2)\n", t_mg_cold)
    @printf("  make_semquad_mg, warm cache : %.3f s (per-level MF operator setup, best of 5)\n",
            t_mg_warm)
    @printf("  geometry part (coarsen + coarse assembly + transfers): %.3f s\n",
            t_mg_cold - t_mg_warm)
    @printf("  resolved smoothing %s, omega=%.4f, pre=%s post=%s, mg_threads=%d\n",
            LAST_SEMSTOKES_MG_PROFILE[][1], mg.omega, join(mg.pre, ","),
            join(mg.post, ","), mg.threads)
    V_profile = LAST_SEMSTOKES_MG_PROFILE[][1]

    mfop = mg.mfops[end]; mfetav = mg.mfetas[end]; mfwork = mg.mfwork[end]
    Cu = mg.geo.Cus[end]
    nt_outer = MS._semquad_mg_thread_count(MGTHR)
    mfapplyK = if nt_outer > 1
        par_outer = MS.SemQuadKPar(mfop, nt_outer)
        (out, v) -> MS._semquad_apply_Kred_threaded!(out, mfop, Cu, mfetav, v,
                                                     mfwork, par_outer)
    else
        (out, v) -> MS._semquad_apply_Kred!(out, mfop, Cu, mfetav, v, mfwork)
    end
    zz = zeros(nf); mfapplyK(zz, ones(nf))
    t_bfbt = tmin(() -> MS.semquad_mg_pressure_precond(ss, sys.Cu, sys.Cp, sys.Bs,
                                                    sys.Bt, sys.Kred, eta_fun;
                                                    mode=:auto, scale=0.0,
                                                    cache=false,
                                                    applyK=mfapplyK), 2)
    Fp, pscale = MS.semquad_mg_pressure_precond(ss, sys.Cu, sys.Cp, sys.Bs, sys.Bt,
                                             sys.Kred, eta_fun; mode=:auto,
                                             scale=0.0, cache=false,
                                             applyK=mfapplyK)
    @printf("  w-BFBT build (auto -> %s, pscale=%.2f): %.3f s (best of 2)\n",
            Fp isa NamedTuple ? Fp.kind : :mass, pscale, t_bfbt)
    wfun = (x, y) -> sqrt(max(eta_fun(x, y), 1.0e-300))
    t_diagd = tmin(() -> MS._semquad_bfbt_diag(ss.su, sys.Cu, eta_fun, wfun;
                                               how=:leaf), 2)
    d = MS._semquad_bfbt_diag(ss.su, sys.Cu, eta_fun, wfun; how=:leaf)
    Dinv = spdiagm(0 => 1.0 ./ d)
    t_s1 = tmin(() -> sys.Bs * Dinv * sys.Bt, 2)
    S1 = sys.Bs * Dinv * sys.Bt
    S1 = 0.5 * (S1 + sparse(transpose(S1)))
    dS1 = maximum(abs, diag(S1))
    S1 = S1 + (1.0e-14 * max(dS1, 1.0)) * I
    t_ch = tmin(() -> cholesky(Symmetric(S1)), 2)
    t_smp = 1e3 * tmin(() -> MS._semquad_eta_sample(ss.su, eta_fun), 5)
    @printf("    D = diag(Cu' Mu(sqrt eta) Cu) : %.3f s\n", t_diagd)
    @printf("    S1 = Bs D^-1 Bt (+sym, +1e-14 shift): %.3f s\n", t_s1)
    @printf("    chol(S1)                      : %.3f s\n", t_ch)
    @printf("    S2                            : %s\n",
            Fp isa NamedTuple && Fp.S2 === nothing ? "matrix-free (no build)" :
            "formed")
    @printf("    eta sample (256 nodes, :auto + cache metric): %.2f ms\n", t_smp)

    # three anatomy passes: the first pays the JIT of the constant-eta closure
    # specialization, and the elementwise minimum of all three removes the
    # occasional GC pause from a single row
    A = setup_anatomy(ss, qt, p, :free, false, eta_fun, nlev)
    A = setup_anatomy(ss, qt, p, :free, false, eta_fun, nlev)
    A = min_anatomy!(A, setup_anatomy(ss, qt, p, :free, false, eta_fun, nlev))
    @printf("\nsetup anatomy (rebuilds the hierarchy from scratch; %d levels):\n",
            A["nlev"])
    @printf("  %-46s %8s\n", "component", "s")
    @printf("  %-46s %8.3f\n", "coarsening (_mg_hierarchy_trees)", A["coarsen"])
    @printf("  %-46s %8.3f\n", "make_semstokes, all levels", sum(A["t_mk"]))
    @printf("  %-46s %8.3f\n", "per-level assembly, all levels", sum(A["t_asm"]))
    @printf("  %-46s %8.3f\n", "sparse(Cu), all levels", sum(A["t_cu"]))
    @printf("  %-46s %8.3f\n", "_mg_free_rows, all levels", sum(A["t_fr"]))
    @printf("  %-46s %8.3f\n", "_mg_prolongation_full, all levels",
            sum(A["t_prol"]))
    @printf("  %-46s %8.3f\n", "Pfull[free,:] * C, all levels", sum(A["t_pmul"]))
    @printf("  %-46s %8.3f\n", "semquad_kop, all levels", sum(A["t_kop"]))
    @printf("  %-46s %8.3f\n", "semquad_eta_field, all levels", sum(A["t_eta"]))
    @printf("  %-46s %8.3f\n", "_semquad_diag_reduced (exact diag), all levels",
            sum(A["t_diag"]))
    @printf("  %-46s %8.3f\n", "dense coarse operator + chol", A["t_dense"] +
            A["t_chol"])
    @printf("  %-46s %8.3f\n", "sum of the parts", A["sum"])
    @printf("  %-46s %8.3f\n", "make_semquad_mg, cold cache (measured)",
            t_mg_cold)
    @printf("\n  per-level setup (s):\n")
    @printf("  %5s %7s %9s %9s %8s %9s %11s %8s %9s %8s\n", "level", "leaves",
            "make_sem", "assemble", "free_r", "Pfull", "P=Cf'PfullCc", "kop",
            "eta_field", "diag")
    for k in 1:nlev
        @printf("  %5d %7d %9.4f %9.4f %8.4f %9s %11s %8.4f %9.4f %8.4f\n", k,
                A["leaves"][k], A["t_mk"][k], A["t_asm"][k], A["t_fr"][k],
                k <= L ? @sprintf("%.4f", A["t_prol"][k]) : "-",
                k <= L ? @sprintf("%.4f", A["t_pmul"][k]) : "-",
                A["t_kop"][k], A["t_eta"][k], A["t_diag"][k])
    end

    # ---------------- one V-cycle ------------------------------------------
    Random.seed!(20261111)
    b = randn(nf)
    zref = MS.semquad_mg_apply(mg, b)
    V = VCProf(nlev)
    mg_apply_prof!(zeros(nf), b, mg, VCProf(nlev))   # compile the replica
    t_rep = Float64[]
    for _ in 1:REPEATS
        Vrep = VCProf(nlev)
        z = zeros(nf)
        mg_apply_prof!(z, b, mg, Vrep)
        z == zref || error("V-cycle replica is not bitwise equal to " *
                           "semquad_mg_apply (max |dz| = $(maximum(abs, z .- zref)))")
        push!(t_rep, Vrep.t_total)
        addv!(V, Vrep)
    end
    v_mean = sum(t_rep) / REPEATS
    v_min = minimum(t_rep)
    scale!(V, 1.0 / REPEATS)
    t_vc_real = tmin(() -> MS.semquad_mg_apply(mg, b), REPEATS)
    @printf("\none V-cycle: production semquad_mg_apply %.3f ms (best of %d) | replica clock mean %.3f ms, min %.3f ms | bitwise equal\n",
            1e3 * t_vc_real, REPEATS, 1e3 * v_mean, 1e3 * v_min)
    @printf("operator applies per V-cycle: %d smoothing + %d residual\n",
            round(Int, (sum(V.n_pre) + sum(V.n_post)) / REPEATS), nlev - 1)

    t_smooth = sum(V.t_pre) + sum(V.t_post)
    t_sm_ap = sum(V.t_pre_ap) + sum(V.t_post_ap)
    t_sm_vec = sum(V.t_pre_vec) + sum(V.t_post_vec)
    t_acc = t_smooth + sum(V.t_res) + sum(V.t_restr) + sum(V.t_prol) +
            V.t_coarse + sum(V.t_axpy) + sum(V.t_fill)
    @printf("\nV-cycle split (ms, mean of %d):\n", REPEATS)
    row(name, t) = @printf("  %-44s %8.3f  %6.1f %%\n", name, 1e3 * t,
                           100 * t / V.t_total)
    row("smoothing, per level (pre+post)", t_smooth)
    row("  of which operator apply (MF element loop)", t_sm_ap)
    row("  of which vector ops (b-Ax + jacobi update)", t_sm_vec)
    row("residual feeding the restriction (apply+sub)", sum(V.t_res))
    row("restriction  rc = P' r", sum(V.t_restr))
    row("prolongation  py = P e", sum(V.t_prol))
    row("coarse-level solve (Cholesky, level 1)", V.t_coarse)
    row("x .+= py", sum(V.t_axpy))
    row("fill!(x, 0)", sum(V.t_fill))
    row("total accounted", t_acc)
    row("V-cycle total (replica clock)", V.t_total)

    @printf("\n  per-level V-cycle (ms):\n")
    @printf("  %5s %7s %8s %8s %9s %10s %8s %8s %8s %8s\n", "level", "dofs",
            "pre", "post", "pre_apply", "post_apply", "resid", "restr", "prol",
            "total")
    for k in nlev:-1:1
        tot = V.t_pre[k] + V.t_post[k] + V.t_res[k] + V.t_restr[k] +
              V.t_prol[k] + V.t_axpy[k] + V.t_fill[k] +
              (k == 1 ? V.t_coarse : 0.0)
        @printf("  %5d %7d %8.3f %8.3f %9.3f %10.3f %8.3f %8.3f %8.3f %8.3f\n",
                k, mg.geo.nfs[k], 1e3 * V.t_pre[k], 1e3 * V.t_post[k],
                1e3 * V.t_pre_ap[k], 1e3 * V.t_post_ap[k], 1e3 * V.t_res[k],
                1e3 * V.t_restr[k], 1e3 * V.t_prol[k], 1e3 * tot)
    end

    # ---------------- the outer Krylov solve ------------------------------
    bsol = vcat(sys.fr, sys.gr)
    let rp = randn(nfp), rp2a = zeros(nfp), rp2b = zeros(nfp),
        s1 = zeros(nf), s2 = zeros(nf), wPa = zeros(nftot), wPb = zeros(nftot)
        MS._semquad_pressure_block!(wPa, copy(rp), rp2a, s1, s2, Fp, pscale, nf)
        pressure_block_prof!(wPb, copy(rp), rp2b, s1, s2, Fp, pscale, nf,
                             SolveProf())
        wPa[(nf + 1):end] == wPb[(nf + 1):end] ||
            error("pressure-block replica is not bitwise equal")
        @printf("\npressure-block replica: bitwise equal to _semquad_pressure_block!\n")
    end
    sol_ref, nit_ref, res_ref, cg_ref = MS.semquad_mg_saddle_solve(sys.Kred,
                                                                   sys.Bs,
                                                                   sys.Bt, Fp,
                                                                   pscale,
                                                                   bsol, mg;
                                                                   tol=TOL,
                                                                   maxit=500,
                                                                   restart=100,
                                                                   vel_solver=:vcycle,
                                                                   applyK=mfapplyK)
    t_iter = tmin(() -> MS.semquad_mg_saddle_solve(sys.Kred, sys.Bs, sys.Bt, Fp,
                                                   pscale, bsol, mg; tol=TOL,
                                                   maxit=500, restart=100,
                                                   vel_solver=:vcycle,
                                                   applyK=mfapplyK), 2)
    # warm-up: compiles the instrumented closures (their JIT would otherwise be
    # booked into the measured run)
    saddle_solve_prof(sys.Kred, sys.Bs, sys.Bt, Fp, pscale, bsol, mg; tol=TOL,
                      maxit=500, restart=100, vel_solver=:vcycle,
                      applyK=mfapplyK, S=SolveProf())
    S = SolveProf()
    FG = FGProf(500)
    sol_p, nit_p, res_p, cg_p = saddle_solve_prof(sys.Kred, sys.Bs, sys.Bt, Fp,
                                                  pscale, bsol, mg; tol=TOL,
                                                  maxit=500, restart=100,
                                                  vel_solver=:vcycle,
                                                  applyK=mfapplyK, S=S, F=FG)
    sol_p == sol_ref || error("saddle-solve replica is not bitwise equal to " *
                              "semquad_mg_saddle_solve (max |dz| = $(maximum(abs, sol_p .- sol_ref)))")
    @printf("outer solve: production semquad_mg_saddle_solve %.3f s (best of 2) | replica %.3f s | bitwise equal\n",
            t_iter, S.t_total)
    @printf("  %d outer iterations, true relres %.3e, tol %.1e, restart 100\n",
            nit_ref, res_ref, TOL)
    @printf("  per iteration: %.3f ms production / %.3f ms replica\n",
            1e3 * t_iter / nit_ref, 1e3 * S.t_total / nit_ref)

    t_app = S.t_applyK + S.t_Bt + S.t_Bs + S.t_copy
    t_pre_all = S.t_vcycle + S.t_Bszc + S.t_pblock
    t_fg = S.t_total - t_app - t_pre_all
    @printf("\nouter-iteration split (s over %d iterations, %% of the iteration, ms/iteration):\n",
            nit_ref)
    row2(name, t) = @printf("  %-50s %8.3f %6.1f %% %8.3f ms/it\n", name, t,
                            100 * t / S.t_total, 1e3 * t / nit_ref)
    row2("outer operator apply A v, total", t_app)
    row2("  MF velocity apply K (Cu' K Cu v)", S.t_applyK)
    row2("  Bt * p", S.t_Bt)
    row2("  Bs * u", S.t_Bs)
    row2("  wA copy", S.t_copy)
    row2("right preconditioner, total", t_pre_all)
    row2("  MG velocity solve (1 V-cycle)", S.t_vcycle)
    row2("  Bs * zc (continuity of the correction)", S.t_Bszc)
    row2("  w-BFBT apply (pressure block)", S.t_pblock)
    row2("    F1 \\ (first Cholesky solve)", S.t_f1a)
    row2("    S2 apply (MF K apply + Bs/Bt, no build)", S.t_s2)
    row2("    F1 \\ (second Cholesky solve) + scaling", S.t_f1b)
    row2("FGMRES bookkeeping (vector ops: axpy/dot/norm, true residual)", t_fg)
    row2("total accounted", t_app + t_pre_all + t_fg)
    @printf("  counts: applyA %d, precond %d, V-cycles %d, F1 solves %d, S2 applies %d\n",
            S.n_applyA, S.n_pre, S.n_vcycle, S.n_f1, S.n_s2)

    # ---- FGMRES internals, measured inside the outer loop -----------------
    f_pre = sum(FG.pre_k); f_app = sum(FG.app_k)
    f_copy = FG.t_copy
    t_fg_in = FG.t_mgs + FG.t_norm + FG.t_hess + FG.t_true + FG.t_restart
    @printf("\nFGMRES internals (s over %d iterations, %% of the iteration, ms/iteration):\n",
            nit_ref)
    row4(name, t) = @printf("  %-50s %8.3f %6.1f %% %8.3f ms/it\n", name, t,
                            100 * t / FG.t_total, 1e3 * t / nit_ref)
    row4("preconditioner calls (from the FGMRES side)", f_pre)
    row4("operator calls (from the FGMRES side)", f_app)
    row4("Z[:,k] = z and W[:,k] = w copies", f_copy)
    row4("MGS orthogonalisation (dot + axpy!, k terms)", FG.t_mgs)
    row4("norm(w) + V[:,k+1] update", FG.t_norm)
    row4("Hessenberg: Givens loop + y = H[1:k,1:k] \\ g[1:k]", FG.t_hess)
    row4("true residual: mul!(dx,W[1:k],y) + r-dx + norm", FG.t_true)
    row4("restart residual b-A x (1 per cycle) + x += Z y", FG.t_restart)
    row4("FGMRES own work (everything except precond/apply)", t_fg_in)
    row4("FGMRES total (its own clock)", FG.t_total)
    @printf("  true-residual recomputations: %d (one per iteration) + %d restart residual(s)\n",
            nit_ref, 1)
    # per-k growth
    println("\n  per-iteration cost by iteration index k (ms/iteration, bucket means):")
    @printf("  %-9s %7s %7s %9s %9s %9s %9s %9s\n", "k", "MGS", "norm",
            "Hessenberg", "true_res", "precond", "apply", "total")
    edges = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60),
             (61, nit_ref)]
    for (lo, hi) in edges
        lo > nit_ref && continue
        hi = min(hi, nit_ref)
        n = hi - lo + 1
        @printf("  %-9s %7.3f %7.3f %9.3f %9.3f %9.3f %9.3f %9.3f\n",
                string(lo, "-", hi),
                1e3 * sum(FG.mgs_k[lo:hi]) / n, 1e3 * sum(FG.norm_k[lo:hi]) / n,
                1e3 * sum(FG.hess_k[lo:hi]) / n,
                1e3 * sum(FG.true_k[lo:hi]) / n,
                1e3 * sum(FG.pre_k[lo:hi]) / n, 1e3 * sum(FG.app_k[lo:hi]) / n,
                1e3 * (sum(FG.mgs_k[lo:hi]) + sum(FG.norm_k[lo:hi]) +
                       sum(FG.hess_k[lo:hi]) + sum(FG.true_k[lo:hi]) +
                       sum(FG.pre_k[lo:hi]) + sum(FG.app_k[lo:hi]) +
                       sum(FG.copy_k[lo:hi])) / n)
    end
    @printf("  growth from k~5 to k~65 (MGS+true_resid): %.2f ms -> %.2f ms\n",
            1e3 * (FG.mgs_k[5] + FG.true_k[5]),
            1e3 * (FG.mgs_k[min(65, nit_ref)] + FG.true_k[min(65, nit_ref)]))

    # ---------------- accounting ------------------------------------------
    t_asm_mf = t_full - (t_mg_cold + t_bfbt + t_iter)
    @printf("\nfull-solve accounting (cold cache, warm JIT):\n")
    row3(name, t) = @printf("  %-50s %8.3f %6.1f %%\n", name, t, 100 * t / t_full)
    row3("assembly, inferred remainder", t_asm_mf)
    row3("hierarchy build (make_semquad_mg, cold)", t_mg_cold)
    row3("w-BFBT build", t_bfbt)
    row3("FGMRES iterations", t_iter)
    row3("sum", t_asm_mf + t_mg_cold + t_bfbt + t_iter)
    @printf("  %-50s %8.3f\n", "measured full solve (best of 3)", t_full)
    @printf("  cross-check: assemble_only standalone %.3f s (also forms the sparse K + Kred, which the MF path never builds)\n",
            t_asm_alone)
    @printf("  setup (assembly + hierarchy + BFBT): %.3f s = %.0f %% of the solve\n",
            t_asm_mf + t_mg_cold + t_bfbt,
            100 * (t_asm_mf + t_mg_cold + t_bfbt) / t_full)
    @printf("  iteration: %.3f s = %.0f %% of the solve\n", t_iter,
            100 * t_iter / t_full)

    if CGVARIANT
        _, nit_cg, res_cg, cgin = MS.semquad_mg_saddle_solve(sys.Kred, sys.Bs,
                                                             sys.Bt, Fp,
                                                             pscale, bsol, mg;
                                                             tol=TOL,
                                                             maxit=500,
                                                             restart=100,
                                                             vel_solver=:cg,
                                                             vel_tol=1.0e-2,
                                                             vel_maxit=10,
                                                             applyK=mfapplyK)
        @printf("\nvel_solver=:cg variant: %d outer, %d inner CG (%.2f per outer), true relres %.3e\n",
                nit_cg, cgin, cgin / max(nit_cg, 1), res_cg)
    end

    # ---------------- A/B: skip the <2x hierarchy step ---------------------
    # The top finding of the profile is that one hierarchy step coarsens the
    # dof count by 1.7 % and yet costs a third of the V-cycle.  `coarsen_factor`
    # is the shipped switch for exactly that (`make_semquad_mg`'s documented
    # "skip steps that would coarsen by less than 2x"); this measures what it
    # buys on this tree.  Opt-in, because it changes the preconditioner.
    if AB
        println("\n--- A/B: coarsen_factor=2.0 (skip hierarchy steps that coarsen < 2x) ---")
        empty!(MS.SEMQUAD_MG_CACHE)
        t0 = time_ns()
        mg2 = MS.make_semquad_mg(ss, qt, p, :free, false, eta_fun, nothing;
                                 cycles=1, smoother=:jacobi,
                                 coarse_operator=:rediscretize,
                                 operator=:matrixfree, mg_threads=MGTHR,
                                 coarsen_factor=2.0, smoothing=:uniform,
                                 fine_Cu=sys.Cu)
        t_mg2 = el(t0)
        @printf("  hierarchy: %d levels, dofs %s, build %.3f s (factor 1.0: %d levels, %.3f s)\n",
                mg2.geo.L + 1, join(mg2.geo.nfs, ", "), t_mg2, nlev, t_mg_cold)
        MS.semquad_mg_apply(mg2, b)                 # compile/warm
        t_vc2 = tmin(() -> MS.semquad_mg_apply(mg2, b), REPEATS)
        @printf("  one V-cycle: %.3f ms (factor 1.0: %.3f ms, production best of %d) = %.2fx\n",
                1e3 * t_vc2, 1e3 * t_vc_real, REPEATS, t_vc_real / t_vc2)
        MS.semquad_mg_saddle_solve(sys.Kred, sys.Bs, sys.Bt, Fp, pscale, bsol,
                                   mg2; tol=TOL, maxit=500, restart=100,
                                   vel_solver=:vcycle, applyK=mfapplyK)
        sol2, nit2, res2, cg2 = MS.semquad_mg_saddle_solve(sys.Kred, sys.Bs,
                                                           sys.Bt, Fp, pscale,
                                                           bsol, mg2; tol=TOL,
                                                           maxit=500,
                                                           restart=100,
                                                           vel_solver=:vcycle,
                                                           applyK=mfapplyK)
        t_it2 = tmin(() -> MS.semquad_mg_saddle_solve(sys.Kred, sys.Bs, sys.Bt,
                                                      Fp, pscale, bsol, mg2;
                                                      tol=TOL, maxit=500,
                                                      restart=100,
                                                      vel_solver=:vcycle,
                                                      applyK=mfapplyK), 2)
        @printf("  saddle solve: %d it, true relres %.3e, %.3f s (factor 1.0: %d it, %.3f s) = %.2fx\n",
                nit2, res2, t_it2, nit_ref, t_iter, t_iter / t_it2)
        @printf("  end to end with the same assembly and BFBT: %.3f s vs %.3f s = %.2fx\n",
                t_mg2 + t_bfbt + t_asm_mf + t_it2,
                t_mg_cold + t_bfbt + t_asm_mf + t_iter,
                (t_mg_cold + t_bfbt + t_asm_mf + t_iter) /
                (t_mg2 + t_bfbt + t_asm_mf + t_it2))
    end


    # ---------------- machine-readable summary ----------------------------
    println("\nSUMMARY")
    @printf("threads=%d mg_threads=%d nf=%d nfp=%d nftot=%d leaves=%d nlev=%d tol=%.1e\n",
            nthreads, mg.threads, nf, nfp, nftot, length(qt.leaves), nlev, TOL)
    @printf("t_full_cold=%.4f t_warm=%.4f t_first=%.4f nit=%d relres=%.4e\n",
            t_full, t_warm, t_first, nit_full, res_full)
    @printf("t_asm_alone=%.4f t_asm_inferred=%.4f t_hier=%.4f t_hier_geo=%.4f t_bfbt=%.4f t_iter=%.4f\n",
            t_asm_alone, t_asm_mf, t_mg_cold, t_mg_cold - t_mg_warm, t_bfbt,
            t_iter)
    @printf("vcycle_ms=%.4f vcycle_replica_mean=%.4f vcycle_replica_min=%.4f smooth_ms=%.4f smooth_apply_ms=%.4f smooth_vec_ms=%.4f resid_ms=%.4f restr_ms=%.4f prol_ms=%.4f coarse_ms=%.4f\n",
            1e3 * t_vc_real, 1e3 * v_mean, 1e3 * v_min, 1e3 * t_smooth,
            1e3 * t_sm_ap, 1e3 * t_sm_vec, 1e3 * sum(V.t_res),
            1e3 * sum(V.t_restr), 1e3 * sum(V.t_prol), 1e3 * V.t_coarse)
    @printf("iter_ms_per_it=%.4f it_total=%.4f apply=%.4f applyK=%.4f Bt=%.4f Bs=%.4f copy=%.4f vcycle=%.4f Bszc=%.4f pblock=%.4f f1a=%.4f s2=%.4f f1b=%.4f fgmres=%.4f\n",
            1e3 * t_iter / nit_ref, t_iter, t_app, S.t_applyK, S.t_Bt, S.t_Bs,
            S.t_copy, S.t_vcycle, S.t_Bszc, S.t_pblock, S.t_f1a, S.t_s2,
            S.t_f1b, t_fg)
    @printf("fgmres_own=%.4f mgs=%.4f norm=%.4f hess=%.4f true=%.4f restart=%.4f zcopies=%.4f\n",
            t_fg_in, FG.t_mgs, FG.t_norm, FG.t_hess, FG.t_true, FG.t_restart,
            f_copy)
    @printf("final_residual=%.6e\n", res_ref)
    @printf("setup_frac=%.4f iter_frac=%.4f\n",
            (t_asm_mf + t_mg_cold + t_bfbt) / t_full, t_iter / t_full)
    @printf("per_level_ms_fine_to_coarse=")
    for k in nlev:-1:1
        tot = V.t_pre[k] + V.t_post[k] + V.t_res[k] + V.t_restr[k] +
              V.t_prol[k] + V.t_axpy[k] + V.t_fill[k] +
              (k == 1 ? V.t_coarse : 0.0)
        @printf("%.4f%s", 1e3 * tot, k == 1 ? "" : ",")
    end
    println()
    rm(tmpdir; force=true, recursive=true)
    return nothing
end

main()
