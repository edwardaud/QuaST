# Restarted GMRES with optional right preconditioning.
# applyA(v) -> A*v ; precond(v) -> M^{-1} v (optional).

"""
    FGMRESWork

Reusable storage for `fgmres_restarted`: the `n x (m+1)` basis `V` and the
`n x m` blocks `Z` (preconditioned directions) and `W` (`A Z` columns) plus the
Hessenberg `H`.  A saddle solve allocates ~100 MB of these on the 1024-leaf
blob tree, and first-touching fresh pages costs ~0.1 s per solve; reusing one
workspace across solves removes that.  Every entry the algorithm reads is
written in the same step (the only exception, `V[:, k+1]` after an exact
breakdown, is explicitly zeroed), so a reused workspace produces bitwise the
same iterates as a freshly zeroed one.
"""
struct FGMRESWork
    V::Matrix{Float64}
    Z::Matrix{Float64}
    W::Matrix{Float64}
    H::Matrix{Float64}
end

FGMRESWork(n::Int, m::Int) = FGMRESWork(zeros(Float64, n, m + 1),
                                        zeros(Float64, n, m),
                                        zeros(Float64, n, m),
                                        zeros(Float64, m + 1, m))

"""
    fgmres_restarted(applyA, b; precond=nothing, tol=1e-10, maxit=200,
                     restart=30, x0=nothing, work=nothing)

Right-preconditioned **flexible** GMRES (Saad 1993): the preconditioner may be
a different operator at every application (an inner CG whose iteration count
depends on the right-hand side, a V-cycle with a data-dependent smoother, ...).
Standard GMRES cannot do this because it applies the *same* preconditioner to
the final correction.  FGMRES stores each preconditioned direction
`Z[:, k] = M_k^{-1} v_k` and updates `x += Z y`, which stays valid for a
varying `M_k` (this is what ASPECT/deal.II do for the Stokes saddle: FGMRES
outside, CG+AMG/multigrid inside).

The stopping criterion is the **true** residual `norm(b - A x) / norm(b)`,
recomputed from the current iterate at every inner step.  The Hessenberg
estimate `|g[k+1]|` is not used: with a variable preconditioner it can be far
from the true residual, and even with a fixed one it drifts (S2's lesson, see
`docs/dev/SOLVER.md` §5).  The true residual is assembled from the stored
`A Z[:, k]` columns, `r_k = r_0 - W[1:k] y`, so it costs `O(k n)` and needs no
extra operator application.  (`applyA` and `precond` may therefore return a
reused scratch vector: the values are copied into `Z`/`W` immediately.)

Returns `(x, true_relres, nit)` with `nit` the number of Arnoldi steps
(iterations).  `maxit` caps `nit`.
"""
function fgmres_restarted(applyA, b::Vector{Float64};
                          precond=nothing, tol::Float64=1e-10,
                          maxit::Int=200, restart::Int=30,
                          x0::Union{Nothing,Vector{Float64}}=nothing,
                          work::Union{Nothing,FGMRESWork}=nothing)
    n = length(b)
    x = x0 === nothing ? zeros(Float64, n) : copy(x0)
    nb = norm(b)
    nb == 0.0 && return x, 0.0, 0
    nit = 0
    true_rel = Inf
    dx = zeros(Float64, n)
    while nit < maxit
        r = b - applyA(x)
        beta = norm(r)
        true_rel = beta / nb
        true_rel <= tol && break
        m = min(restart, maxit - nit)
        V, Z, W, H = if work === nothing || size(work.V, 1) != n ||
                        size(work.V, 2) != m + 1
            (zeros(Float64, n, m + 1), zeros(Float64, n, m), zeros(Float64, n, m),
             zeros(Float64, m + 1, m))
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
            Z[:, k] = z
            w = applyA(z)
            W[:, k] = w                 # copy before the orthogonalisation
            for i in 1:k
                vi = view(V, :, i)
                H[i, k] = dot(w, vi)
                axpy!(-H[i, k], vi, w)
            end
            H[k+1, k] = norm(w)
            if H[k+1, k] > 1e-14
                @views V[:, k+1] .= w ./ H[k+1, k]
            else
                # Exact breakdown: `V[:, k+1]` is not a basis vector.  A fresh
                # `zeros` workspace left the column zero; a reused one must be
                # zeroed explicitly to keep the iterates identical.
                @views fill!(V[:, k+1], 0.0)
            end
            for i in 1:k-1
                t = cs[i] * H[i, k] + sn[i] * H[i+1, k]
                H[i+1, k] = -sn[i] * H[i, k] + cs[i] * H[i+1, k]
                H[i, k] = t
            end
            den = hypot(H[k, k], H[k+1, k])
            if den == 0.0
                k_used = k        # lucky breakdown: the correction is exact
                break
            end
            cs[k] = H[k, k] / den; sn[k] = H[k+1, k] / den
            H[k, k] = cs[k] * H[k, k] + sn[k] * H[k+1, k]
            H[k+1, k] = 0.0
            g[k+1] = -sn[k] * g[k]
            g[k] = cs[k] * g[k]
            k_used = k
            # ---- true residual of the current iterate x + Z[1:k] y ----
            y = H[1:k, 1:k] \ g[1:k]
            mul!(dx, view(W, :, 1:k), y)      # dx = A Z y
            @. dx = r - dx                    # dx = b - A (x + Z y)
            true_rel = norm(dx) / nb
            true_rel <= tol && break
        end
        k_used == 0 && break
        y = H[1:k_used, 1:k_used] \ g[1:k_used]
        mul!(dx, view(Z, :, 1:k_used), y)
        x .+= dx
    end
    return x, true_rel, nit
end

function gmres_restarted(applyA, b::Vector{Float64};
                         precond=nothing, tol::Float64=1e-10,
                         maxit::Int=200, restart::Int=30,
                         x0::Union{Nothing,Vector{Float64}}=nothing,
                         stagnation::Float64=0.0)
    n = length(b)
    x = x0 === nothing ? zeros(Float64, n) : copy(x0)
    nb = norm(b)
    nb == 0.0 && return x, 0.0, 0
    it_total = 0
    while it_total < maxit
        r = b - applyA(x)
        beta = norm(r)
        beta / nb < tol && break
        # G1 (docs/dev/BUG_LEDGER.md S16): honour `maxit` *inside* the cycle.  The
        # historical loop always ran `restart` Arnoldi steps and only tested the
        # budget at the top, so `maxit=80, restart=30` really used 90 vectors.
        m = min(restart, maxit - it_total)
        V = zeros(Float64, n, restart + 1)
        H = zeros(Float64, restart + 1, restart)
        V[:, 1] = r ./ beta
        g = zeros(Float64, restart + 1); g[1] = beta
        cs = zeros(Float64, restart); sn = zeros(Float64, restart)
        k_used = 0
        for k in 1:m
            it_total += 1
            w = precond === nothing ? V[:, k] : precond(V[:, k])
            w = applyA(w)
            for i in 1:k
                H[i, k] = dot(w, V[:, i])
                w .-= H[i, k] .* V[:, i]
            end
            H[k+1, k] = norm(w)
            if H[k+1, k] > 1e-14
                V[:, k+1] = w ./ H[k+1, k]
            end
            for i in 1:k-1
                t = cs[i] * H[i, k] + sn[i] * H[i+1, k]
                H[i+1, k] = -sn[i] * H[i, k] + cs[i] * H[i+1, k]
                H[i, k] = t
            end
            den = hypot(H[k, k], H[k+1, k])
            # G3 (docs/dev/BUG_LEDGER.md S16): a lucky breakdown (both entries
            # zero) made `cs = H[k,k]/den` a NaN that propagated into every
            # later iterate; `fgmres_restarted` already guards this, the
            # non-flexible version did not.  Measured trigger: a diagonal system
            # with a cluster plus 5 tiny outliers returned rel = NaN before this
            # line and 1e-1 after it.  A healthy solve never has den == 0, so
            # nothing else changes.
            if den == 0.0
                # `H[1:k-1,1:k-1]` is still an upper triangular matrix with a
                # nonzero diagonal (every earlier step had `den != 0`), so the
                # correction is taken from the directions built so far instead
                # of dividing by zero.
                k_used = k - 1
                break
            end
            cs[k] = H[k, k] / den; sn[k] = H[k+1, k] / den
            H[k, k] = cs[k] * H[k, k] + sn[k] * H[k+1, k]
            H[k+1, k] = 0.0
            g[k+1] = -sn[k] * g[k]
            g[k] = cs[k] * g[k]
            k_used = k
            abs(g[k+1]) / nb < tol && break
        end
        # No usable Krylov direction at all (complete break-down at the first
        # step): the iterate cannot be improved, so do not spin on the budget.
        # Mirrors the same line in `fgmres_restarted`.
        k_used == 0 && break
        y = H[1:k_used, 1:k_used] \ g[1:k_used]
        dx = V[:, 1:k_used] * y
        x .+= precond === nothing ? dx : precond(dx)
        # G2 (docs/dev/BUG_LEDGER.md S16): a full restart cycle that bought less
        # than a factor `stagnation` cannot reach `tol` in the budget that is
        # left, so stop instead of burning the rest of `maxit`.  The test uses
        # the Hessenberg estimate `|g[k+1]|/nb` (already computed), so the cost
        # is one division and one comparison per *cycle* (measured <0.01% of a
        # cycle).  `stagnation = 0` disables it, and that is the default, so
        # every other caller of this routine is bit for bit unchanged.
        if stagnation > 0.0 && k_used > 0 &&
           abs(g[k_used+1]) / nb > (beta / nb) / stagnation
            break
        end
    end
    return x, norm(b - applyA(x)) / nb, it_total
end
