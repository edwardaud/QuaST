# Sparse direct solves with a true-residual contract.
#
# B5/B6 of the Python reference: never accept an iterative internal estimate,
# always recompute A x - b, and fail loudly instead of continuing silently.

struct LinearResidualError <: Exception
    name::String
    res::Float64
    bar::Float64
end

function Base.showerror(io::IO, e::LinearResidualError)
    print(io, "LinearResidualError(", e.name, "): true residual ",
          e.res, " exceeds bar ", e.bar)
end

"""
Solve ``A x = b`` by sparse LU and return ``(x, res)`` with
``res = ||A x - b|| / max(||b||, 1)``.  Throws if the true residual exceeds
``res_bar``; a singular factorisation is rethrown with context.
"""
function solve_linear(A::SparseMatrixCSC{Float64,Int}, b::Vector{Float64};
                      res_bar::Float64=1.0e-8, name::String="linear solve",
                      refine::Int=2, equilibrate::Bool=false)
    if equilibrate
        # Ruiz iterative equilibration: scale rows/columns to unit inf-norm.
        n = size(A, 1)
        dr = ones(n); dc = ones(n)
        rows = rowvals(A); vals = nonzeros(A)
        for _ in 1:10
            rowmax = zeros(n); colmax = zeros(n)
            @inbounds for j in 1:n
                for p in nzrange(A, j)
                    i = rows[p]
                    v = abs(vals[p]) * dr[i] * dc[j]
                    rowmax[i] = max(rowmax[i], v)
                    colmax[j] = max(colmax[j], v)
                end
            end
            @inbounds for i in 1:n
                rowmax[i] > 0.0 && (dr[i] /= sqrt(rowmax[i]))
            end
            @inbounds for j in 1:n
                colmax[j] > 0.0 && (dc[j] /= sqrt(colmax[j]))
            end
        end
        Dr = spdiagm(0 => dr)
        Dc = spdiagm(0 => dc)
        As = Dr * A * Dc
        F = try
            lu(As)
        catch err
            throw(ErrorException("$name: equilibrated sparse LU failed: $(sprint(showerror, err))"))
        end
        x = Dc * (F \ (Dr * b))
        r = A * x - b
        res = norm(r) / max(norm(b), 1.0)
        for _ in 1:refine
            res <= res_bar && break
            x .+= Dc * (F \ (Dr * r))
            r = A * x - b
            res = norm(r) / max(norm(b), 1.0)
        end
        if !isfinite(res) || res > res_bar
            throw(LinearResidualError(name, res, res_bar))
        end
        return x, res
    end
    F = try
        lu(A)
    catch err
        throw(ErrorException("$name: sparse LU failed: $(sprint(showerror, err))"))
    end
    x = F \ b
    r = A * x - b
    res = norm(r) / max(norm(b), 1.0)
    # Iterative refinement with the same factorisation: this is the standard
    # remedy when a hard viscosity floor makes the matrix ill-conditioned.
    for _ in 1:refine
        res <= res_bar && break
        delta = F \ r
        x .+= delta
        r = A * x - b
        res = norm(r) / max(norm(b), 1.0)
    end
    if !isfinite(res) || res > res_bar
        throw(LinearResidualError(name, res, res_bar))
    end
    return x, res
end

"""
Preconditioned Uzawa (pressure-correction) solve of the reduced Stokes saddle
system.  The velocity block and the sparse approximate Schur complement
``S = B diag(A_v)^{-1} B^T`` are factorised separately; this avoids the
ill-conditioned full saddle LU at n>=64.
"""
function solve_stokes_uzawa(sys::StokesSystem, dm::DofMap;
                            tol::Float64=1.0e-8, maxiter::Int=300,
                            omega::Float64=1.0, verbose::Bool=false)
    nvel = dm.nux + dm.nuz
    Av = sys.A[1:nvel, 1:nvel]
    B = sys.A[nvel+1:end, 1:nvel]
    Bt = sparse(transpose(B))
    fv = sys.f[1:nvel]
    np = size(B, 1)
    Fv = lu(Av)
    Dinv = spdiagm(0 => 1.0 ./ diag(Av))
    S = B * Dinv * Bt + 1.0e-10 * I
    Fs = lu(S)
    u = Fv \ fv
    p = zeros(Float64, np)
    r = B * u
    res = norm(r) / max(norm(fv), 1.0)
    it = 0
    for k in 1:maxiter
        it = k
        p .+= omega * (Fs \ r)
        u = Fv \ (fv - Bt * p)
        r = B * u
        ru = Av * u + Bt * p - fv
        res = sqrt(norm(ru)^2 + norm(r)^2) / max(norm(fv), 1.0)
        verbose && k % 10 == 0 && @printf("  uzawa %3d  res=%.3e\n", k, res)
        res <= tol && break
    end
    if !isfinite(res) || res > tol
        throw(LinearResidualError("Stokes-uzawa", res, tol))
    end
    return u, p, res, it
end

"""
Right-preconditioned restarted GMRES for the reduced Stokes system, with the
block-diagonal preconditioner ``P = diag(A_v, B diag(A_v)^{-1} B^T)``.
Useful when the full saddle LU loses accuracy at n>=64.
"""
function solve_stokes_gmres(sys::StokesSystem, dm::DofMap;
                            tol::Float64=1.0e-8, maxiter::Int=200,
                            restart::Int=50, verbose::Bool=false)
    nvel = dm.nux + dm.nuz
    Av = sys.A[1:nvel, 1:nvel]
    B = sys.A[nvel+1:end, 1:nvel]
    Bt = sparse(transpose(B))
    fv = sys.f[1:nvel]
    np = size(B, 1)
    Fv = lu(Av)
    Dinv = spdiagm(0 => 1.0 ./ diag(Av))
    S = B * Dinv * Bt + 1.0e-10 * I
    Fs = lu(S)

    function precond(r::Vector{Float64})
        z = Vector{Float64}(undef, length(r))
        z[1:nvel] = Fv \ r[1:nvel]
        z[nvel+1:end] = Fs \ r[nvel+1:end]
        return z
    end

    b = sys.f
    x = zeros(Float64, length(b))
    r = copy(b)
    beta = norm(r)
    it = 0
    while it < maxiter && beta > tol * max(norm(b), 1.0)
        V = zeros(Float64, length(b), restart + 1)
        H = zeros(Float64, restart + 1, restart)
        cs = zeros(Float64, restart)
        sn = zeros(Float64, restart)
        g = zeros(Float64, restart + 1)
        V[:, 1] = r / beta
        g[1] = beta
        j = 0
        for k in 1:restart
            j = k
            z = precond(V[:, k])
            w = sys.A * z
            for i in 1:k
                H[i, k] = dot(w, V[:, i])
                w .-= H[i, k] .* V[:, i]
            end
            H[k+1, k] = norm(w)
            if H[k+1, k] > 1.0e-14
                V[:, k+1] = w ./ H[k+1, k]
            end
            for i in 1:k-1
                temp = cs[i] * H[i, k] + sn[i] * H[i+1, k]
                H[i+1, k] = -sn[i] * H[i, k] + cs[i] * H[i+1, k]
                H[i, k] = temp
            end
            rr = hypot(H[k, k], H[k+1, k])
            if rr == 0.0
                cs[k] = 1.0; sn[k] = 0.0
            else
                cs[k] = H[k, k] / rr
                sn[k] = H[k+1, k] / rr
            end
            H[k, k] = cs[k] * H[k, k] + sn[k] * H[k+1, k]
            H[k+1, k] = 0.0
            g[k+1] = -sn[k] * g[k]
            g[k] = cs[k] * g[k]
            it += 1
            if abs(g[k+1]) <= tol * max(norm(b), 1.0)
                break
            end
        end
        y = H[1:j, 1:j] \ g[1:j]
        x .+= precond(V[:, 1:j] * y)
        r = b - sys.A * x
        beta = norm(r)
        verbose && @printf("  gmres it=%3d  true_res=%.3e\n", it, beta / max(norm(b), 1.0))
        if !isfinite(beta)
            break
        end
    end
    res = beta / max(norm(b), 1.0)
    if !isfinite(res) || res > tol
        throw(LinearResidualError("Stokes-gmres", res, tol))
    end
    return x[1:nvel], x[nvel+1:end], res, it
end

"""
Exact-Schur Uzawa: build ``S = B A_v^{-1} B^T`` explicitly (dense, Npres^2) and
factor it.  For n=64 Npres~4k this is affordable once per Stokes solve and
removes the slow convergence of the diagonal-Schur iteration.
"""
function solve_stokes_schur(sys::StokesSystem, dm::DofMap;
                            tol::Float64=1.0e-8, maxiter::Int=50,
                            verbose::Bool=false)
    nvel = dm.nux + dm.nuz
    Av = sys.A[1:nvel, 1:nvel]
    B = sys.A[nvel+1:end, 1:nvel]
    Bt = sparse(transpose(B))
    fv = sys.f[1:nvel]
    np = size(B, 1)
    Fv = lu(Av)
    # exact Schur complement (dense)
    X = Fv \ Matrix(Bt)
    S = Matrix(B) * X
    S = S + 1.0e-10 * I
    Fs = lu(S)
    u = Fv \ fv
    p = zeros(Float64, np)
    r = B * u
    res = norm(r) / max(norm(fv), 1.0)
    it = 0
    for k in 1:maxiter
        it = k
        p .+= Fs \ r
        u = Fv \ (fv - Bt * p)
        r = B * u
        ru = Av * u + Bt * p - fv
        res = sqrt(norm(ru)^2 + norm(r)^2) / max(norm(fv), 1.0)
        verbose && @printf("  schur %3d  res=%.3e\n", k, res)
        res <= tol && break
    end
    if !isfinite(res) || res > tol
        throw(LinearResidualError("Stokes-schur", res, tol))
    end
    return u, p, res, it
end

# Matrix-free application of S = B A_v^{-1} B^T (+ tiny shift), SPD.
function _schur_apply(Fv, B, Bt, x::Vector{Float64})
    y = Fv \ (Bt * x)
    return B * y .+ 1.0e-10 .* x
end

function _cg_schur(Fv, B, Bt, b::Vector{Float64}; Fs=nothing,
                   tol::Float64=1.0e-10, maxiter::Int=100)
    x = zeros(Float64, length(b))
    r = copy(b)
    if Fs === nothing
        p = copy(r)
        rz = dot(r, r)
    else
        z = Fs \ r
        p = copy(z)
        rz = dot(r, z)
    end
    bnorm2 = max(dot(b, b), 1.0e-300)
    for _ in 1:maxiter
        Ap = _schur_apply(Fv, B, Bt, p)
        den = dot(p, Ap)
        den == 0.0 && break
        alpha = rz / den
        x .+= alpha .* p
        r .-= alpha .* Ap
        rsnew = dot(r, r)
        rsnew <= tol^2 * bnorm2 && break
        if Fs === nothing
            rznew = rsnew
            p .= r .+ (rznew / rz) .* p
        else
            z = Fs \ r
            rznew = dot(r, z)
            p .= z .+ (rznew / rz) .* p
        end
        rz = rznew
    end
    return x
end

"""
Matrix-free exact-Schur Uzawa: the Schur system is solved by CG, so each
Stokes solve costs a handful of ``A_v`` back-substitutions instead of Npres
dense right-hand sides.  This is the practical n>=64 solver.
"""
function solve_stokes_schur_cg(sys::StokesSystem, dm::DofMap;
                               tol::Float64=1.0e-8, maxiter::Int=20,
                               verbose::Bool=false)
    nvel = dm.nux + dm.nuz
    Av = sys.A[1:nvel, 1:nvel]
    B = sys.A[nvel+1:end, 1:nvel]
    Bt = sparse(transpose(B))
    fv = sys.f[1:nvel]
    np = size(B, 1)
    Fv = lu(Av)
    Dinv = spdiagm(0 => 1.0 ./ diag(Av))
    Sdiag = B * Dinv * Bt + 1.0e-10 * I
    Fs = lu(Sdiag)
    u = Fv \ fv
    p = zeros(Float64, np)
    r = B * u
    res = norm(r) / max(norm(fv), 1.0)
    it = 0
    for k in 1:maxiter
        it = k
        delta = _cg_schur(Fv, B, Bt, r; Fs=Fs, tol=1.0e-10, maxiter=100)
        p .+= delta
        u = Fv \ (fv - Bt * p)
        r = B * u
        ru = Av * u + Bt * p - fv
        res = sqrt(norm(ru)^2 + norm(r)^2) / max(norm(fv), 1.0)
        verbose && @printf("  schur-cg %3d  res=%.3e\n", k, res)
        res <= tol && break
    end
    if !isfinite(res) || res > tol
        throw(LinearResidualError("Stokes-schur-cg", res, tol))
    end
    return u, p, res, it
end
