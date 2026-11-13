# Ruiz-scaled trust-region Newton for nonlinear systems F(x)=0.
# Generic: Ffun(x) -> Vector, Jfun(x) -> AbstractMatrix.

"""
    ruiz_scaling(J; iters=5)

Row/column infinity-norm equilibration. Returns `(dr, dc)` such that
`Diagonal(dr) * J * Diagonal(dc)` has rows/columns of order one.
"""
function ruiz_scaling(J; iters::Int=5)
    n = size(J, 1)
    dr = ones(Float64, n)
    dc = ones(Float64, n)
    for _ in 1:iters
        S = Diagonal(dr) * J * Diagonal(dc)
        rn = vec(sum(abs.(S), dims=2))
        cn = vec(sum(abs.(S), dims=1))
        dr ./= sqrt.(max.(rn, 1e-30))
        dc ./= sqrt.(max.(cn, 1e-30))
    end
    # remove the arbitrary global scaling drift
    dr ./= exp(sum(log.(dr)) / n)
    dc ./= exp(sum(log.(dc)) / n)
    return dr, dc
end

"""
    trust_region_newton(Ffun, Jfun, x0; maxit=50, tol=1e-8, delta=1.0)

Ruiz-scaled trust-region Newton. The step is computed in scaled variables,
`Js = Dr J Dc`, `Fs = Dr F`, and the trust radius limits `||Dc^{-1} dx||`.
"""
function trust_region_newton(Ffun, Jfun, x0::Vector{Float64};
                             maxit::Int=50, tol::Float64=1e-8,
                             delta::Float64=1.0, verbose::Bool=false)
    x = copy(x0)
    F = Ffun(x)
    for it in 1:maxit
        nF = norm(F)
        nF < tol && return x, nF, it - 1
        J = Jfun(x)
        dr, dc = ruiz_scaling(J)
        Fs = dr .* F
        Js = Diagonal(dr) * J * Diagonal(dc)
        ys = -(Js \ Fs)
        ny = norm(ys)
        if ny > delta
            ys .*= delta / ny
            ny = delta
        end
        dx = dc .* ys
        Fn = Ffun(x .+ dx)
        Fsn = dr .* Fn
        pred = norm(Fs) - norm(Fs .+ Js * ys)
        rho = pred > 0 ? (norm(Fs) - norm(Fsn)) / pred : -1.0
        if rho > 0.1 && isfinite(norm(Fsn))
            x = x .+ dx
            F = Fn
            delta = max(delta, 2.0 * ny)
        else
            delta *= 0.25
        end
        verbose && @info "tr" it norm(F) delta rho
        delta < 1e-14 && break
    end
    return x, norm(F), maxit
end
