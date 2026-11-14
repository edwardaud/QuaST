# Production multigrid diagonal and affine-constraint helpers (split from stokes/solve.jl).
"""
    _semquad_diag_full!(d, op, etav)

Diagonal of the **unconstrained** matrix-free velocity stiffness: a scatter-add
of the leaf-local element diagonals
`Kloc_xx[k,k] = sum_q w eta (a2 gx_k^2 + gz_k^2)` (and the mirror for `z`).  It
costs `O(nq^2 n^2)` per leaf once, i.e. the same as a single apply, and is what
lets the matrix-free multigrid build a Jacobi diagonal without any sparse `K`
(MF-3).  The deviatoric `-2/3 div(u) div(v)` term does not touch the diagonal.
"""
function _semquad_diag_full!(d::Vector{Float64}, op::SemQuadKOp,
                             etav::AbstractVector{Float64})
    n = op.n; nq = op.nq; N = op.N
    phi = op.phi; dphi = op.dphi; gl_w = op.gl_w; ids = op.ids
    a2 = op.deviatoric ? 4.0 / 3.0 : 2.0
    dlocx = zeros(Float64, n * n)
    dlocz = zeros(Float64, n * n)
    fill!(d, 0.0)
    nq2 = nq * nq
    @inbounds for li in 1:op.nleaf
        ihx = op.ihx[li]; ihy = op.ihy[li]; wf = op.wfac[li]
        fill!(dlocx, 0.0); fill!(dlocz, 0.0)
        k = (li - 1) * nq2
        for qy in 1:nq, qx in 1:nq
            k += 1
            cw = wf * gl_w[qx] * gl_w[qy] * etav[k]
            for j in 1:n, i in 1:n
                gx = dphi[i, qx] * ihx * phi[j, qy]
                gz = phi[i, qx] * dphi[j, qy] * ihy
                idx = (j - 1) * n + i
                dlocx[idx] += cw * (a2 * gx * gx + gz * gz)
                dlocz[idx] += cw * (a2 * gz * gz + gx * gx)
            end
        end
        for idx in 1:(n * n)
            g = ids[idx, li]
            d[g] += dlocx[idx]
            d[N + g] += dlocz[idx]
        end
    end
    return d
end

"""
    _semquad_diag_reduced(op, Cu, etav; mode::Symbol=:constrained)

Jacobi diagonal of the matrix-free *reduced* level operator `Cu' * K * Cu`.

`mode=:constrained` (the default, and the fix of docs/dev/MG.md
§5.2) is the **exact** reduced diagonal `diag(Cu' K Cu)`, computed without ever
forming `K` or the product: for every leaf, the constraint rows of its local
dofs give a dense `m x ncol` matrix `W` (`m = 2 n^2` local dofs, `ncol` the
reduced columns that touch the leaf, including the free-row identity entries),
and the leaf contributes `diag(W' K_e W)`.  Summing over leaves is exactly the
diagonal of the assembled product because `K = sum_e K_e` and each `K_e` has no
cross-leaf entries.  Cost is one dense `2n^2 x 2n^2` element matrix per leaf
(the assembler's local block), i.e. a few V-cycle applies' worth of leaf work
once per hierarchy build (measured 0.098 s for all eight levels of the 8188-leaf
production tree).

`mode=:lumped` is the historical `Cu' * diag(K)` approximation (the standard
lumped diagonal), kept for A/B only.  On a hanging-node tree it is *not*
`diag(Cu' K Cu)`: the Lagrange interpolation weights are negative, a hanging row
can cancel to <= 0, and the positivity clip below then makes that row a `1e12`
amplifier in the damped-Jacobi smoother (measured: one V-cycle maps
`||b|| = 1e3` to `||z|| = 1.6e170` on the production tree).
"""
function _semquad_diag_reduced(op::SemQuadKOp,
                               Cu::SparseMatrixCSC{Float64,Int},
                               etav::AbstractVector{Float64};
                               mode::Symbol=:constrained)
    mode in (:constrained, :lumped) ||
        throw(ArgumentError("MG Jacobi diagonal mode must be :constrained or " *
                            ":lumped (got :$mode)"))
    d = if mode === :lumped || !_semquad_diag_has_hanging(Cu)
        # `:lumped` is the historical path; on a conforming tree `Cu` only
        # selects free rows, so `Cu' * diag(K)` *is* `diag(Cu' K Cu)` and the
        # old arithmetic is kept bit for bit.
        _semquad_diag_reduced_lumped(op, Cu, etav)
    else
        _semquad_diag_reduced_constrained(op, Cu, etav)
    end
    # A constrained diagonal is positive up to round-off; the clip is a guard,
    # not a modelling choice (on the production tree it is never active).
    dmax = maximum(d)
    dmax > 0.0 || throw(ArgumentError("matrix-free Jacobi diagonal is not positive"))
    @inbounds for i in eachindex(d)
        d[i] = max(d[i], 1.0e-12 * dmax)
    end
    return d
end

"""
    _semquad_diag_has_hanging(Cu) -> Bool

`Cu` has a hanging-node (interpolation) row iff it is not a pure row selector:
a conforming `Cu` stores exactly one `1.0` per reduced column and nothing else,
so `nnz(Cu) == size(Cu, 2) && all(isone, nonzeros(Cu))` identifies it exactly
(`semquad_constraints` never emits a one-term hanging row: such a node would
coincide with a coarse node and is skipped).
"""
@inline function _semquad_diag_has_hanging(Cu::SparseMatrixCSC{Float64,Int})
    vals = nonzeros(Cu)
    return !(length(vals) == size(Cu, 2) && all(isone, vals))
end

"""
    _semquad_diag_reduced_lumped(op, Cu, etav)

Historical (defective on hanging rows) `Cu' * diag(K)` diagonal of the reduced
matrix-free operator.  Kept only as the `mode=:lumped` A/B reference of
`_semquad_diag_reduced`.
"""
function _semquad_diag_reduced_lumped(op::SemQuadKOp,
                                      Cu::SparseMatrixCSC{Float64,Int},
                                      etav::AbstractVector{Float64})
    dfull = zeros(Float64, 2 * op.N)
    _semquad_diag_full!(dfull, op, etav)
    return Vector{Float64}(transpose(Cu) * dfull)
end

"""
    _semquad_leaf_stiffness!(Kloc, op, etav, li, gx, gz)

The `2n^2 x 2n^2` element stiffness of leaf `li`, accumulated exactly like the
assembler's local block in `_semquad_assemble_block!` (same quadrature points,
weights and expression trees).  `gx`/`gz` are length-`n^2` scratch vectors.
"""
function _semquad_leaf_stiffness!(Kloc::Matrix{Float64}, op::SemQuadKOp,
                                  etav::AbstractVector{Float64}, li::Int,
                                  gx::Vector{Float64}, gz::Vector{Float64})
    n = op.n; nq = op.nq; nv = n * n
    phi = op.phi; dphi = op.dphi; gl_w = op.gl_w
    a2 = op.deviatoric ? 4.0 / 3.0 : 2.0
    ihx = op.ihx[li]; ihy = op.ihy[li]; wf = op.wfac[li]
    fill!(Kloc, 0.0)
    k = (li - 1) * nq * nq
    @inbounds for qy in 1:nq, qx in 1:nq
        k += 1
        w = wf * gl_w[qx] * gl_w[qy]
        etaq = etav[k]
        for j in 1:n, i in 1:n
            kk = (j - 1) * n + i
            gx[kk] = dphi[i, qx] * phi[j, qy] * ihx
            gz[kk] = phi[i, qx] * dphi[j, qy] * ihy
        end
        if op.deviatoric
            for l in 1:nv
                xl = gx[l]; zl = gz[l]
                @simd for kk in 1:nv
                    Kloc[kk, l] += etaq * (a2 * gx[kk] * xl + gz[kk] * zl) * w
                    Kloc[kk, nv + l] += etaq * (gz[kk] * xl -
                                                (2.0 / 3.0) * gx[kk] * zl) * w
                    Kloc[nv + kk, l] += etaq * (gx[kk] * zl -
                                                (2.0 / 3.0) * gz[kk] * xl) * w
                    Kloc[nv + kk, nv + l] += etaq * (a2 * gz[kk] * zl +
                                                     gx[kk] * xl) * w
                end
            end
        else
            for l in 1:nv
                xl = gx[l]; zl = gz[l]
                @simd for kk in 1:nv
                    Kloc[kk, l] += etaq * (a2 * gx[kk] * xl + gz[kk] * zl) * w
                    Kloc[kk, nv + l] += etaq * (gz[kk] * xl) * w
                    Kloc[nv + kk, l] += etaq * (gx[kk] * zl) * w
                    Kloc[nv + kk, nv + l] += etaq * (a2 * gz[kk] * zl +
                                                     gx[kk] * xl) * w
                end
            end
        end
    end
    return Kloc
end

"""
    _semquad_diag_reduced_constrained(op, Cu, etav)

Exact `diag(Cu' K Cu)` of the matrix-free reduced velocity operator; see
`_semquad_diag_reduced`.  The reduced columns of a leaf are read from
`transpose(Cu)` (column `g` = the constraint row of full dof `g`), so no
`K`-matrix and no `Cu' K Cu` product is ever formed.
"""
function _semquad_diag_reduced_constrained(op::SemQuadKOp,
                                           Cu::SparseMatrixCSC{Float64,Int},
                                           etav::AbstractVector{Float64})
    n = op.n; nq = op.nq; N = op.N
    nv = n * n; m = 2 * nv
    nf = size(Cu, 2)
    d = zeros(Float64, nf)
    Ct = SparseMatrixCSC{Float64,Int}(sparse(transpose(Cu)))
    # Per-leaf scratch.  Every local dof contributes at most `p+1` columns
    # (one for a free row, `p+1` for a hanging row), so `(p+2)*m` bounds `ncol`.
    bound = (op.p + 2) * m
    colidx = Vector{Int}(undef, bound)
    Wmat = zeros(Float64, m, bound)
    Kmat = zeros(Float64, m, bound)
    Kloc = zeros(Float64, m, m)
    gx = zeros(Float64, nv)
    gz = zeros(Float64, nv)
    stamp = zeros(Int32, nf)
    slot = zeros(Int32, nf)
    ids = op.ids
    @inbounds for li in 1:op.nleaf
        sid = Int32(li)
        # Reduced columns that this leaf can contribute to.
        ncol = 0
        for a in 1:m
            g = a <= nv ? ids[a, li] : N + ids[a - nv, li]
            for k in nzrange(Ct, g)
                w = Ct.nzval[k]
                w == 0.0 && continue
                i = Ct.rowval[k]
                if stamp[i] != sid
                    stamp[i] = sid
                    ncol += 1
                    colidx[ncol] = i
                    slot[i] = Int32(ncol)
                end
            end
        end
        ncol == 0 && continue
        fill!(Wmat, 0.0)
        for a in 1:m
            g = a <= nv ? ids[a, li] : N + ids[a - nv, li]
            for k in nzrange(Ct, g)
                w = Ct.nzval[k]
                w == 0.0 && continue
                Wmat[a, slot[Ct.rowval[k]]] = w
            end
        end
        _semquad_leaf_stiffness!(Kloc, op, etav, li, gx, gz)
        WC = view(Wmat, :, 1:ncol)
        KC = view(Kmat, :, 1:ncol)
        mul!(KC, Kloc, WC)
        for j in 1:ncol
            s = 0.0
            for a in 1:m
                s += Wmat[a, j] * Kmat[a, j]
            end
            d[colidx[j]] += s
        end
    end
    return d
end

"""
    _semquad_dense_operator(mfops, mfetas, mfwork, Cus, lev, nf)

Materialise the (tiny) coarsest level operator by applying the matrix-free
operator to the unit vectors.  Used by the fully matrix-free hierarchy (MF-3)
in place of the sparse `A[1]`: the coarsest tree has only the base leaves, so
`nf` is `O(100)` and this costs less than one fine-level V-cycle.
"""
function _semquad_dense_operator!(A::Matrix{Float64}, op::SemQuadKOp,
                                  Cu::SparseMatrixCSC{Float64,Int},
                                  etav::AbstractVector{Float64},
                                  work::SemQuadKWork)
    nf = size(A, 1)
    e = zeros(Float64, nf)
    y = zeros(Float64, nf)
    @inbounds for i in 1:nf
        fill!(e, 0.0)
        e[i] = 1.0
        _semquad_apply_Kred!(y, op, Cu, etav, e, work)
        for j in 1:nf
            A[j, i] = y[j]
        end
    end
    @inbounds for j in 1:nf, i in 1:nf
        A[i, j] = 0.5 * (A[i, j] + A[j, i])
    end
    return A
end
"""Affine constraint u = u0 + C u_free from hanging constraints and fixed dofs."""
function _affine_constraint(N::Int, cons::Dict{Int,Tuple{Vector{Int},Vector{Float64}}},
                            fixed::Dict{Int,Float64})
    u0 = zeros(Float64, N)
    for (i, v) in fixed
        u0[i] = v
    end
    # move fixed masters into u0, iterate to resolve chains
    work = Dict(c => (copy(ms), copy(ws)) for (c, (ms, ws)) in cons)
    active = trues(length(work))
    keysv = collect(keys(work))
    keep = Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
    for _ in 1:8
        keep = Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
        changed = false
        for (c, (ms, ws)) in work
            haskey(fixed, c) && continue
            nms = Int[]; nws = Float64[]
            for (m, w) in zip(ms, ws)
                if haskey(fixed, m)
                    u0[c] += w * fixed[m]
                    changed = true
                elseif haskey(work, m) && haskey(fixed, m) == false
                    push!(nms, m); push!(nws, w)
                else
                    push!(nms, m); push!(nws, w)
                end
            end
            keep[c] = (nms, nws)
        end
        work = keep
        changed || break
    end
    # free dofs: not fixed and not constrained
    free = [i for i in 1:N if !haskey(fixed, i) && !haskey(work, i)]
    fmap = Dict(v => k for (k, v) in enumerate(free))
    I = Int[]; J = Int[]; V = Float64[]
    for (k, i) in enumerate(free)
        push!(I, i); push!(J, k); push!(V, 1.0)
    end
    for (c, (ms, ws)) in work
        for (m, w) in zip(ms, ws)
            push!(I, c); push!(J, fmap[m]); push!(V, w)
        end
    end
    C = sparse(I, J, V, N, length(free))
    return u0, C, free
end

