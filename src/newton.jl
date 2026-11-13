# Generic damped Newton with a finite-difference Jacobian and a backtracking line
# search, plus a Jacobian-free Newton-Krylov variant (stage C).
#
# Route C's nonlinearity is the viscosity/buoyancy fixed point: given theta the
# Stokes solve and the energy step are linear, but eta(T) and the buoyancy make
# the coupled map nonlinear.  Both drivers below take a residual `F` and solve
# `F(x) = 0`; the second one avoids assembling the Jacobian, which matters when
# one evaluation of `F` costs a full Stokes + energy solve.

"""
    newton_solve(F, x0; tol=1e-8, maxit=25, fd_step=1e-7, verbose=false)

Solve `F(x) = 0` from `x0` by a damped Newton iteration with a forward-difference
Jacobian and Armijo backtracking.

Returns `(x, converged, iters, residual_history)`.  On failure the best iterate
so far is returned with `converged == false`.
"""
function newton_solve(F::Function, x0::AbstractVector{Float64};
                      tol::Float64=1.0e-8, maxit::Int=25,
                      fd_step::Float64=1.0e-7,
                      max_backtrack::Int=12, armijo::Float64=1.0e-4,
                      verbose::Bool=false)
    x = Float64.(x0)
    n = length(x)
    r = Float64.(F(x))
    nr = norm(r)
    hist = [nr]
    for it in 1:maxit
        nr <= tol && return (x, true, it - 1, hist)
        J = zeros(Float64, length(r), n)
        for j in 1:n
            xp = copy(x)
            h = fd_step * max(1.0, abs(x[j]))
            xp[j] += h
            J[:, j] .= (Float64.(F(xp)) .- r) ./ h
        end
        dx = try
            -(J \ r)
        catch
            return (x, false, it, hist)
        end
        t = 1.0
        accepted = false
        for _ in 1:max_backtrack
            xt = x .+ t .* dx
            rt = Float64.(F(xt))
            if norm(rt) < (1.0 - armijo * t) * nr
                x = xt
                r = rt
                nr = norm(rt)
                accepted = true
                break
            end
            t *= 0.5
        end
        accepted || return (x, false, it, hist)
        push!(hist, nr)
        verbose && @printf("  newton %2d: |F| = %.3e (t = %.3g)\n", it, nr, t)
    end
    return (x, nr <= tol, maxit, hist)
end

"""
    jfnk_solve(F, x0; tol=1e-8, maxit=15, krylov=20, fd_step=1e-7)

Jacobian-free Newton-Krylov: the same damped Newton as `newton_solve`, but the
linear system `J dx = -F` is solved by GMRES with matrix-free products
`J v ~ (F(x + eps v) - F(x)) / eps`.  Each Krylov iteration costs one extra
evaluation of `F`.

The inner solve stops at the last non-degenerate Krylov step (`jmax`), so small
problems (n = 1 or 2) terminate instead of hitting a singular Hessenberg system.

Returns `(x, converged, iters)`.
"""
function jfnk_solve(F::Function, x0::AbstractVector{Float64};
                    tol::Float64=1.0e-8, maxit::Int=15, krylov::Int=20,
                    fd_step::Float64=1.0e-7, verbose::Bool=false)
    x = Float64.(x0)
    r = Float64.(F(x))
    nr = norm(r)
    for it in 1:maxit
        nr <= tol && return (x, true, it - 1)
        beta = norm(r)
        beta == 0.0 && return (x, true, it - 1)
        m = min(krylov, length(x))
        Q = zeros(Float64, length(x), m + 1)
        H = zeros(Float64, m + 1, m)
        Q[:, 1] .= r ./ beta
        jmax = 0
        for j in 1:m
            eps = fd_step * max(1.0, norm(x)) / max(norm(Q[:, j]), 1.0e-30)
            Jv = (Float64.(F(x .+ eps .* Q[:, j])) .- r) ./ eps
            w = copy(Jv)
            for i in 1:j
                H[i, j] = dot(w, Q[:, i])
                w = w .- H[i, j] .* Q[:, i]
            end
            H[j + 1, j] = norm(w)
            jmax = j
            if H[j + 1, j] > 1.0e-14
                Q[:, j + 1] .= w ./ H[j + 1, j]
            else
                break
            end
        end
        jmax == 0 && return (x, false, it)
        rhs = zeros(Float64, jmax + 1)
        rhs[1] = beta
        y = H[1:(jmax + 1), 1:jmax] \ rhs
        dx = -(Q[:, 1:jmax] * y)
        t = 1.0
        accepted = false
        for _ in 1:12
            rt = Float64.(F(x .+ t .* dx))
            if norm(rt) < (1.0 - 1.0e-4 * t) * nr
                x = x .+ t .* dx
                r = rt
                nr = norm(rt)
                accepted = true
                break
            end
            t *= 0.5
        end
        accepted || return (x, false, it)
        verbose && @printf("  jfnk %2d: |F| = %.3e (jmax = %d)\n", it, nr, jmax)
    end
    return (x, nr <= tol, maxit)
end
