# ============================================================================
# S3 / MF-4: CG on the SPD velocity block + flexible outer GMRES
# ============================================================================

"""
    SemQuadCGWork(n)

Preallocated scratch for `semquad_mg_cg` (the inner velocity CG of MF-4), so the
outer FGMRES does not allocate a handful of length-`n` vectors per preconditioner
application.
"""
struct SemQuadCGWork
    z::Vector{Float64}
    r::Vector{Float64}
    p::Vector{Float64}
    Ap::Vector{Float64}
    w::Vector{Float64}
end

SemQuadCGWork(n::Int) = SemQuadCGWork(zeros(n), zeros(n), zeros(n), zeros(n),
                                      zeros(n))

"""
    semquad_mg_cg(applyK!, b, mg; tol=1.0e-2, maxit=10, work=SemQuadCGWork(length(b)))

Preconditioned conjugate gradients for the SPD reduced velocity block `K z = b`
with the multigrid hierarchy as the preconditioner: one `semquad_mg_apply`
(i.e. `mg.cycles` V-cycles) per CG iteration.  `applyK!` is the matrix-vector
product of `K` (`mul!` on `Kred`, or the matrix-free `semquad_apply_Kred!`).

This is the inner solver of MF-4: the outer FGMRES then sees a *much* better
velocity inverse than a fixed number of V-cycles (two V-cycles are only a ~2%
approximate inverse at p=4 L4), at the cost of a data-dependent iteration
count -- exactly why the outer Krylov method must be flexible.

Returns `(z, iters, relres)` with `z === work.z`.  It bails out (returning the
current iterate) if the preconditioner loses positivity or a search direction
becomes non-positive, so a non-SPD level operator cannot make the outer
iteration diverge silently.
"""
function semquad_mg_cg(applyK!::Function, b::Vector{Float64}, mg::SemQuadMG;
                       tol::Float64=1.0e-2, maxit::Int=10,
                       work::SemQuadCGWork=SemQuadCGWork(length(b)))
    n = length(b)
    z = work.z; r = work.r; p = work.p; Ap = work.Ap; w = work.w
    fill!(z, 0.0)
    copyto!(r, b)
    bn = norm(r)
    bn == 0.0 && return z, 0, 0.0
    rho = 0.0
    rel = 1.0
    it = 0
    for k in 1:maxit
        it = k
        w .= semquad_mg_apply(mg, r)
        rho_new = dot(r, w)
        if !(rho_new > 0.0) || !isfinite(rho_new)
            break                      # preconditioner lost positivity
        end
        if k == 1
            p .= w
        else
            beta = rho_new / rho
            @. p = w + beta * p
        end
        applyK!(Ap, p)
        den = dot(p, Ap)
        if !(den > 0.0) || !isfinite(den)
            break                      # search direction not an ascent direction
        end
        alpha = rho_new / den
        @. z += alpha * p
        @. r -= alpha * Ap
        rho = rho_new
        rel = norm(r) / bn
        rel <= tol && break
    end
    return z, it, rel
end


"""
    semquad_mg_saddle_solve(Kred, Bs, Bt, Fp, pscale, b, mg; tol, maxit, restart,
                            vel_solver, vel_tol, vel_maxit)

Solve the reduced saddle

    [Kred  Bt] [u]   [fr]
    [Bs    0 ] [p] = [gr]

with flexible GMRES (`fgmres_restarted`) and the block-triangular right
preconditioner

    P = [Khat  Bt]     Khat   = K^{-1} approximated by the multigrid hierarchy,
        [0    -Shat]   Shat^{-1} = pscale * Fp^{-1}

where `Fp` is either the pressure-block Cholesky factor of the mass/stiffness/LSC
mode or the three-factor w-BFBT `(kind=:bfbt, F1, S2)` named tuple.

`vel_solver=:cg` (MF-4, default) replaces the fixed V-cycle count by an inner CG
on `Kred` preconditioned by the hierarchy (`semquad_mg_cg`), i.e. the
deal.II/ASPECT scheme: FGMRES outside, CG + multigrid inside.  The
preconditioner is then data dependent, which is why the outer loop must be
FGMRES.  `vel_solver=:vcycle` keeps the historical fixed `mg.cycles` V-cycles
(a *linear* preconditioner, so the two are interchangeable there).

The outer stopping criterion is the **true** reduced-saddle residual
(`fgmres_restarted`), never the preconditioned one.  Returns
`(sol, nit, relres, cgiters)`.
"""
function semquad_mg_saddle_solve(Kred::Union{Nothing,AbstractMatrix{Float64}},
                                 Bs::AbstractMatrix{Float64},
                                 Bt::AbstractMatrix{Float64},
                                 Fp, pscale::Float64, b::Vector{Float64},
                                 mg::SemQuadMG;
                                 tol::Float64=1.0e-10, maxit::Int=500,
                                 restart::Int=100, vel_solver::Symbol=:vcycle,
                                 vel_tol::Float64=1.0e-2, vel_maxit::Int=10,
                                 applyK::Union{Nothing,Function}=nothing)
    vel_solver in (:cg, :vcycle) ||
        throw(ArgumentError("MG velocity solver must be :cg or :vcycle"))
    nf = size(Bs, 2)
    nfp = size(Bs, 1)
    nftot = nf + nfp
    length(b) == nftot || throw(ArgumentError(
        "MG saddle right-hand side has $(length(b)) entries, expected $nftot"))
    nf == size(mg.geo.Cus[end], 2) || throw(ArgumentError(
        "Kred has $nf rows but the MG fine level has $(size(mg.geo.Cus[end], 2)) rows"))
    # S3 stage C: the velocity operator is either the sparse `Kred` or, on the
    # fully matrix-free path, the leaf-local `Cu' K (Cu v)` apply.
    Kred === nothing && applyK === nothing &&
        throw(ArgumentError("semquad_mg_saddle_solve needs Kred or applyK"))
    applyK! = applyK === nothing ? (out, v) -> mul!(out, Kred, v) : applyK
    cgiters = Ref(0)
    # Reused scratch: FGMRES copies the returned vectors into Z/W immediately,
    # so returning the same buffer every call is safe and keeps the outer
    # iteration allocation-free on the large levels.
    wA = zeros(Float64, nftot)
    wP = zeros(Float64, nftot)
    zc = zeros(Float64, nf)
    rp = zeros(Float64, nfp)
    rp2 = zeros(Float64, nfp)
    s1 = zeros(Float64, nf)
    s2 = zeros(Float64, nf)
    ku = zeros(Float64, nf)
    cgwork = SemQuadCGWork(nf)
    function applyA(v::Vector{Float64})
        applyK!(ku, view(v, 1:nf))
        wA[1:nf] .= ku
        wA[1:nf] .+= Bt * v[(nf + 1):nftot]
        wA[(nf + 1):nftot] = Bs * v[1:nf]
        return wA
    end
    function precond(v::Vector{Float64})
        if vel_solver === :cg
            _, nc, _ = semquad_mg_cg(applyK!, v[1:nf], mg; tol=vel_tol,
                                     maxit=vel_maxit, work=cgwork)
            cgiters[] += nc
            copyto!(zc, cgwork.z)
        else
            zc .= semquad_mg_apply(mg, v[1:nf])
        end
        wP[1:nf] .= zc
        rp .= v[(nf + 1):nftot] .- Bs * zc
        _semquad_pressure_block!(wP, rp, rp2, s1, s2, Fp, pscale, nf)
        return wP
    end
    sol, relres, nit = fgmres_restarted(applyA, b; precond=precond, tol=tol,
                                        maxit=maxit, restart=restart,
                                        work=_semquad_fgmres_work(nftot,
                                                                  min(restart,
                                                                      maxit)))
    return sol, nit, relres, cgiters[]
end

