# ===========================================================================
# The inner solve of the w-BFBT `S1` block (docs/dev/MG.md)
#
# The w-BFBT pressure block applies `Shat^{-1} = S1^{-1} S2 S1^{-1}` and does
# that twice per outer iteration: two solves with
# `F1 = cholesky(Symmetric(S1))`.  `S1` is the SPD reduced operator
# `S1 = Bs diag(1/D) Bt` (7 424 x 7 424 on state `F`), factored once per tree.
#
# This file holds the inner solve as a first-class object so that the exact
# CHOLMOD solve, a level-scheduled parallel solve over the same factor, and an
# approximate iterative solve can be swapped and measured:
#
#   SemQuadS1Chol   -- `ldiv!(x, F1, b)`: the exact solve.  Bitwise identical
#                      to the shipped `x .= F1 \ b` (same CHOLMOD call, without
#                      the 240 KB allocation `\` pays per call).
#   SemQuadS1Level  -- the same factor, applied by two topological
#                      level-scheduled triangular solves: one row's arithmetic
#                      is unchanged (it sums its nonzeros in storage order, as
#                      the serial reference does), so the result is bitwise
#                      identical to the serial level-scheduled solve and
#                      independent of the thread count.
#   SemQuadS1CG     -- CG to `tol` preconditioned by the diagonal or by an
#                      incomplete Cholesky (IC(0)) of `S1`.
# ===========================================================================

# ---------------------------------------------------------------------------
# :chol -- the exact factor solve
# ---------------------------------------------------------------------------
"""
    SemQuadS1Chol(F1)

The exact inner solve: `F1 = cholesky(Symmetric(S1))` from CHOLMOD, applied with
`ldiv!` (the same call as `F1 \\ b`, without allocating the result).
"""
struct SemQuadS1Chol
    F1::Any
end

_semquad_s1_solve!(x::Vector{Float64}, S::SemQuadS1Chol,
                   b::Vector{Float64}) = ldiv!(x, S.F1, b)

_semquad_s1_iters(::SemQuadS1Chol) = 0

# ---------------------------------------------------------------------------
# :level -- level-scheduled sparse triangular solves over the same factor
# ---------------------------------------------------------------------------
"""
    SemQuadS1Level

The same `F1` factor, applied by two level-scheduled sparse triangular solves.

`L` (from `sparse(F1.L)`, the permuted factor `P S1 P' = L L'`) is renumbered
twice -- once in forward-*row* dependency order and once in backward-*column*
dependency order -- so the rows of one level are a contiguous range in both
orderings.  The solve then sweeps the levels in order; rows inside a level are
independent by construction.

A level runs as one [`SemQuadParRegion`](@ref) phase only when its work pays for
the barrier (`minwork`); otherwise it runs on the calling task with no
synchronisation, which is what most levels need (on `F`: 90 % of the forward
levels and 55 % of the backward ones are narrower than 8 rows).
"""
mutable struct SemQuadS1Level
    n::Int
    fperm::Vector{Int}       # forward position  -> original dof index
    bperm::Vector{Int}       # backward position -> original dof index
    fbmap::Vector{Int}       # backward position -> forward position
    fptr::Vector{Int}        # CSR of the strict lower triangle, forward order
    fcol::Vector{Int}
    fval::Vector{Float64}
    bptr::Vector{Int}        # the same in backward order (columns of L as rows)
    brow::Vector{Int}
    bval::Vector{Float64}
    fdiag::Vector{Float64}
    bdiag::Vector{Float64}
    frows::Vector{NTuple{2,Int}}   # (first,last) row range of each forward level
    fwork::Vector{Int}             # and its nonzero count
    brows::Vector{NTuple{2,Int}}   # the same for the backward levels
    bwork::Vector{Int}
    minwork::Int
    chunkwork::Int
    region::Any              # SemQuadParRegion, or nothing at one thread
    y::Vector{Float64}       # scratch, length n (forward result)
    z::Vector{Float64}       # scratch, length n (backward result)
    iters::Base.RefValue{Int}
end

_semquad_s1_iters(S::SemQuadS1Level) = S.iters[]

"""Forward row-dependency levels of a CSC lower factor."""
function _s1_row_levels(L::SparseMatrixCSC, n::Int)
    rv = L.rowval; cp = L.colptr
    lev = zeros(Int, n); cnt = zeros(Int, n)
    @inbounds for j in 1:n
        lj = lev[j]
        for p in cp[j]:(cp[j + 1] - 1)
            i = rv[p]
            i == j && continue
            cnt[i] += 1
            lev[i] = ifelse(lev[i] > lj, lev[i], lj + 1)
        end
    end
    return lev .+ 1, cnt
end

"""Backward column-dependency levels of a CSC lower factor."""
function _s1_col_levels(L::SparseMatrixCSC, n::Int)
    rv = L.rowval; cp = L.colptr
    lev = zeros(Int, n); cnt = zeros(Int, n)
    @inbounds for i in n:-1:1
        l = 0; c = 0
        for p in cp[i]:(cp[i + 1] - 1)
            j = rv[p]
            j == i && continue
            c += 1
            l = ifelse(lev[j] > l, lev[j], l)
        end
        lev[i] = l + 1
        cnt[i] = c
    end
    return lev, cnt
end

"""Contiguous row ranges of each level in `order`, and each level's nnz work."""
function _s1_level_ranges(lev, order::Vector{Int}, cnt::Vector{Int})
    nl = maximum(lev)
    lo = fill(length(order) + 1, nl)
    hi = zeros(Int, nl)
    work = zeros(Int, nl)
    @inbounds for pos in 1:length(order)
        l = lev[order[pos]]
        lo[l] = min(lo[l], pos)
        hi[l] = max(hi[l], pos)
        work[l] += cnt[order[pos]]
    end
    rows = NTuple{2,Int}[]
    wk = Int[]
    for l in 1:nl
        hi[l] >= lo[l] || continue
        push!(rows, (lo[l], hi[l]))
        push!(wk, work[l])
    end
    return rows, wk
end

"""
    SemQuadS1Level(F1, S1s; nthreads=1, minwork=3000, chunkwork=600)

Build the level schedule of `F1` (the two renumberings and the two CSR copies).
`minwork` is the nonzero count a level must reach before it is worth a region
phase; `chunkwork` is the nnz per chunk handed to one worker.
"""
function SemQuadS1Level(F1, S1s::SparseMatrixCSC{Float64,Int};
                        nthreads::Int=1, minwork::Int=0, chunkwork::Int=0)
    n = size(S1s, 1)
    L = sparse(F1.L)                     # permuted lower factor, CSC
    rv = L.rowval; nz = nonzeros(L); cp = L.colptr
    p = Vector{Int}(F1.p)                # Julia's CHOLMOD `p` is already 1-based
    flev, fcnt = _s1_row_levels(L, n)
    blev, bcnt = _s1_col_levels(L, n)
    forder = sortperm(flev)              # stable: a level keeps ascending order
    border = sortperm(blev)
    finv = invperm(forder)
    binv = invperm(border)
    # --- forward: CSR of the strict lower triangle, forward order ----------
    fptr = Vector{Int}(undef, n + 1)
    fptr[1] = 1
    @inbounds for l in 1:n
        fptr[l + 1] = fptr[l] + fcnt[forder[l]]
    end
    fcol = Vector{Int}(undef, fptr[n + 1] - 1)
    fval = Vector{Float64}(undef, fptr[n + 1] - 1)
    fdiag = Vector{Float64}(undef, n)
    pos = copy(fptr)
    @inbounds for i in 1:n
        for pp in cp[i]:(cp[i + 1] - 1)
            j = rv[pp]
            if j == i
                fdiag[finv[i]] = nz[pp]
                continue
            end
            r = finv[j]                  # L[j, i] belongs to row j
            q = pos[r]
            fcol[q] = finv[i]
            fval[q] = nz[pp]
            pos[r] = q + 1
        end
    end
    # --- backward: rows of L' are columns of L, in backward order ----------
    bptr = Vector{Int}(undef, n + 1)
    bptr[1] = 1
    @inbounds for l in 1:n
        bptr[l + 1] = bptr[l] + bcnt[border[l]]
    end
    brow = Vector{Int}(undef, bptr[n + 1] - 1)
    bval = Vector{Float64}(undef, bptr[n + 1] - 1)
    bdiag = Vector{Float64}(undef, n)
    @inbounds for l in 1:n
        o = border[l]
        q = bptr[l]
        for pp in cp[o]:(cp[o + 1] - 1)
            j = rv[pp]
            if j == o
                bdiag[l] = nz[pp]
                continue
            end
            brow[q] = binv[j]
            bval[q] = nz[pp]
            q += 1
        end
    end
    frows, fwork = _s1_level_ranges(flev, forder, fcnt)
    brows, bwork = _s1_level_ranges(blev, border, bcnt)
    fbmap = Vector{Int}(undef, n)
    @inbounds for l in 1:n
        fbmap[l] = finv[border[l]]
    end
    mw = minwork > 0 ? minwork : 3000
    cw = chunkwork > 0 ? chunkwork : 600
    region = nthreads > 1 ? SemQuadParRegion(nthreads) : nothing
    return SemQuadS1Level(n, p[forder], p[border], fbmap, fptr, fcol, fval, bptr,
                          brow, bval, fdiag, bdiag, frows, fwork, brows, bwork,
                          mw, cw, region, zeros(Float64, n), zeros(Float64, n),
                          Ref(0))
end

@inline function _s1_fwd_rows!(y::Vector{Float64}, S::SemQuadS1Level, lo::Int,
                               hi::Int)
    fptr = S.fptr; fcol = S.fcol; fval = S.fval; dg = S.fdiag
    @inbounds for l in lo:hi
        s = y[l]
        for p in fptr[l]:(fptr[l + 1] - 1)
            s -= fval[p] * y[fcol[p]]
        end
        y[l] = s / dg[l]
    end
    return nothing
end

@inline function _s1_bwd_rows!(x::Vector{Float64}, S::SemQuadS1Level, lo::Int,
                               hi::Int)
    bptr = S.bptr; brow = S.brow; bval = S.bval; dg = S.bdiag
    @inbounds for l in lo:hi
        s = x[l]
        for p in bptr[l]:(bptr[l + 1] - 1)
            s -= bval[p] * x[brow[p]]
        end
        x[l] = s / dg[l]
    end
    return nothing
end

@inline function _s1_fwd_chunk!(y, S, lo, hi, c, nc)
    q = cld(hi - lo + 1, nc)
    a = lo + (c - 1) * q
    b = min(hi, a + q - 1)
    a <= b && _s1_fwd_rows!(y, S, a, b)
    return nothing
end

@inline function _s1_bwd_chunk!(x, S, lo, hi, c, nc)
    q = cld(hi - lo + 1, nc)
    a = lo + (c - 1) * q
    b = min(hi, a + q - 1)
    a <= b && _s1_bwd_rows!(x, S, a, b)
    return nothing
end

"""One forward level: serial on the calling task, or one region phase."""
@inline function _s1_fwd_level!(y, S, lo, hi, work)
    r = S.region
    if r === nothing || work < S.minwork
        _s1_fwd_rows!(y, S, lo, hi)
        return nothing
    end
    nc = clamp(div(work, S.chunkwork), 1, r.nt)
    if nc <= 1
        _s1_fwd_rows!(y, S, lo, hi)
    else
        _semquad_phase!(r, (c, w) -> _s1_fwd_chunk!(y, S, lo, hi, c, nc), nc)
    end
    return nothing
end

"""One backward level: serial on the calling task, or one region phase."""
@inline function _s1_bwd_level!(x, S, lo, hi, work)
    r = S.region
    if r === nothing || work < S.minwork
        _s1_bwd_rows!(x, S, lo, hi)
        return nothing
    end
    nc = clamp(div(work, S.chunkwork), 1, r.nt)
    if nc <= 1
        _s1_bwd_rows!(x, S, lo, hi)
    else
        _semquad_phase!(r, (c, w) -> _s1_bwd_chunk!(x, S, lo, hi, c, nc), nc)
    end
    return nothing
end

"""
    _semquad_s1_solve!(x, S::SemQuadS1Level, b)

`x = S1 \\ b` by level-scheduled forward/backward substitution over the stored
factor.  `x` must not alias `b`.

The whole solve runs as one [`SemQuadParRegion`](@ref) body (one fork/join per
solve, not per level); only levels whose work pays for the barrier are published
as phases, the rest run on the calling task with no synchronisation.
"""
function _semquad_s1_solve!(x::Vector{Float64}, S::SemQuadS1Level,
                            b::Vector{Float64})
    if S.region === nothing
        _s1_sweep!(x, S, b)
    else
        _semquad_region_run!(S.region) do
            _s1_sweep!(x, S, b)
        end
    end
    S.iters[] += 1
    return x
end

function _s1_sweep!(x::Vector{Float64}, S::SemQuadS1Level, b::Vector{Float64})
    n = S.n
    y = S.y; z = S.z
    fp = S.fperm; bp = S.bperm
    @inbounds for l in 1:n
        y[l] = b[fp[l]]
    end
    @inbounds for l in 1:length(S.frows)
        lo, hi = S.frows[l]
        _s1_fwd_level!(y, S, lo, hi, S.fwork[l])
    end
    fb = S.fbmap
    @inbounds for l in 1:n
        z[l] = y[fb[l]]
    end
    @inbounds for l in 1:length(S.brows)
        lo, hi = S.brows[l]
        _s1_bwd_level!(z, S, lo, hi, S.bwork[l])
    end
    @inbounds for l in 1:n
        x[bp[l]] = z[l]
    end
    return nothing
end

# ---------------------------------------------------------------------------
# :cg -- CG preconditioned by the diagonal or by IC(0)
# ---------------------------------------------------------------------------
"""
    SemQuadS1CG(A; precond=:diag, tol=1e-2, maxit=50)

CG on `A` (the same `S1 + shift I` the factor is built from) to a *relative*
residual `tol`, preconditioned by `Diagonal(A)` (`precond=:diag`) or by an IC(0)
factor of `A` (`precond=:ichol`).  `iters` counts the CG iterations from the
first call on (zero it before a timed run); the tolerance sweep reports its
growth, which is the trade the approximate inner solve makes against the exact
one.
"""
mutable struct SemQuadS1CG
    A::SparseMatrixCSC{Float64,Int}
    precond::Symbol
    diag::Vector{Float64}
    icptr::Vector{Int}
    iccol::Vector{Int}
    icval::Vector{Float64}
    icdiag::Vector{Float64}
    icptrT::Vector{Int}
    icrowT::Vector{Int}
    icvalT::Vector{Float64}
    tol::Float64
    maxit::Int
    iters::Base.RefValue{Int}
    r::Vector{Float64}
    z::Vector{Float64}
    p::Vector{Float64}
    q::Vector{Float64}
end

"""Deterministic dot product (no BLAS, so the result does not depend on threads)."""
_semquad_s1_iters(S::SemQuadS1CG) = S.iters[]

@inline function _s1_dot(x::Vector{Float64}, y::Vector{Float64})
    s = 0.0
    @inbounds @simd for i in eachindex(x)
        s += x[i] * y[i]
    end
    return s
end

@inline function _s1_norm(x::Vector{Float64})
    return sqrt(_s1_dot(x, x))
end

"""IC(0) of `A`: `(ptr, col, val, diag)` of the incomplete factor and its transpose."""
function _s1_ic0(A::SparseMatrixCSC{Float64,Int})
    n = size(A, 1)
    cp = A.colptr; rv = A.rowval; av = nonzeros(A)
    # CSR of the strict lower triangle (ascending columns): entry `(r, c)` of
    # `A` with `r > c` lives in column `c` of the CSC and belongs to row `r`
    ptr = Vector{Int}(undef, n + 1)
    ptr[1] = 1
    cnt = zeros(Int, n)
    @inbounds for c in 1:n
        for p in cp[c]:(cp[c + 1] - 1)
            r = rv[p]
            r > c && (cnt[r] += 1)
        end
    end
    @inbounds for i in 1:n
        ptr[i + 1] = ptr[i] + cnt[i]
    end
    col = Vector{Int}(undef, ptr[n + 1] - 1)
    aval = Vector{Float64}(undef, ptr[n + 1] - 1)
    fillpos = copy(ptr)
    adiag = Vector{Float64}(undef, n)
    @inbounds for c in 1:n
        for p in cp[c]:(cp[c + 1] - 1)
            r = rv[p]
            if r == c
                adiag[c] = av[p]
            elseif r > c
                q = fillpos[r]
                col[q] = c
                aval[q] = av[p]
                fillpos[r] = q + 1
            end
        end
    end
    val = copy(aval)
    diag = Vector{Float64}(undef, n)
    w = zeros(Float64, n)
    mark = zeros(Int, n)
    @inbounds for i in 1:n
        d = adiag[i]
        for p in ptr[i]:(ptr[i + 1] - 1)
            j = col[p]
            s = aval[p]
            for q in ptr[j]:(ptr[j + 1] - 1)
                k = col[q]
                if mark[k] == i
                    s -= w[k] * val[q]
                end
            end
            v = s / diag[j]
            val[p] = v
            w[j] = v
            mark[j] = i
            d -= v * v
        end
        d > 0.0 || throw(ArgumentError("IC(0) of S1 broke down at row $i"))
        diag[i] = sqrt(d)
    end
    # transpose of the strictly-lower pattern (for the backward substitution)
    tptr = Vector{Int}(undef, n + 1)
    tptr[1] = 1
    tcnt = zeros(Int, n)
    @inbounds for i in 1:n, p in ptr[i]:(ptr[i + 1] - 1)
        tcnt[col[p]] += 1
    end
    @inbounds for i in 1:n
        tptr[i + 1] = tptr[i] + tcnt[i]
    end
    trow = Vector{Int}(undef, tptr[n + 1] - 1)
    tval = Vector{Float64}(undef, tptr[n + 1] - 1)
    fillpos = copy(tptr)
    @inbounds for i in 1:n, p in ptr[i]:(ptr[i + 1] - 1)
        j = col[p]
        q = fillpos[j]
        trow[q] = i
        tval[q] = val[p]
        fillpos[j] = q + 1
    end
    return ptr, col, val, diag, tptr, trow, tval
end

"""
    SemQuadS1CG(A; precond=:diag, tol=1e-2, maxit=50)
"""
function SemQuadS1CG(A::SparseMatrixCSC{Float64,Int};
                     precond::Symbol=:diag, tol::Float64=1.0e-2, maxit::Int=50)
    n = size(A, 1)
    precond in (:diag, :ichol) ||
        throw(ArgumentError("S1 CG preconditioner must be :diag or :ichol"))
    diag = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        diag[i] = A[i, i]
    end
    if precond === :ichol
        ptr, col, val, dg, tptr, trow, tval = _s1_ic0(A)
    else
        ptr = Int[]; col = Int[]; val = Float64[]; dg = Float64[]
        tptr = Int[]; trow = Int[]; tval = Float64[]
    end
    return SemQuadS1CG(A, precond, diag, ptr, col, val, dg, tptr, trow, tval,
                       tol, maxit, Ref(0), zeros(Float64, n), zeros(Float64, n),
                       zeros(Float64, n), zeros(Float64, n))
end

@inline function _s1_precond!(z::Vector{Float64}, S::SemQuadS1CG,
                              r::Vector{Float64})    if S.precond === :diag
        dg = S.diag
        @inbounds @simd for i in eachindex(r)
            z[i] = r[i] / dg[i]
        end
    else
        ptr = S.icptr; col = S.iccol; val = S.icval; dg = S.icdiag
        n = length(r)
        @inbounds for i in 1:n
            s = r[i]
            for p in ptr[i]:(ptr[i + 1] - 1)
                s -= val[p] * z[col[p]]
            end
            z[i] = s / dg[i]
        end
        tptr = S.icptrT; trow = S.icrowT; tval = S.icvalT
        @inbounds for i in n:-1:1
            s = z[i]
            for p in tptr[i]:(tptr[i + 1] - 1)
                s -= tval[p] * z[trow[p]]
            end
            z[i] = s / dg[i]
        end
    end
    return nothing
end

function _semquad_s1_solve!(x::Vector{Float64}, S::SemQuadS1CG,
                            b::Vector{Float64})
    r = S.r; z = S.z; p = S.p; q = S.q
    fill!(x, 0.0)
    copyto!(r, b)
    nb = _s1_norm(b)
    nb == 0.0 && return x
    _s1_precond!(z, S, r)
    copyto!(p, z)
    rz = _s1_dot(r, z)
    tol = S.tol * nb
    for _ in 1:S.maxit
        mul!(q, S.A, p)
        pq = _s1_dot(p, q)
        pq == 0.0 && break
        al = rz / pq
        @inbounds @simd for i in eachindex(x)
            x[i] += al * p[i]
        end
        @inbounds @simd for i in eachindex(r)
            r[i] -= al * q[i]
        end
        S.iters[] += 1
        _s1_norm(r) <= tol && break
        _s1_precond!(z, S, r)
        rz2 = _s1_dot(r, z)
        be = rz2 / rz
        rz = rz2
        @inbounds @simd for i in eachindex(p)
            p[i] = z[i] + be * p[i]
        end
    end
    return x
end

# ---------------------------------------------------------------------------
# building a scheme by name
# ---------------------------------------------------------------------------
"""
    _semquad_s1_inner(kind, F1, S1s, nthreads; tol, maxit, minwork, chunkwork)

Resolve the `bfbt_s1_solve` name to the inner-solve object.  `:chol` and
`:mass` keep the exact CHOLMOD solve; `:level` uses the level schedule; the
`:cg_*` names use CG.
"""
function _semquad_s1_inner(kind::Symbol, F1, S1s::SparseMatrixCSC{Float64,Int},
                           nthreads::Int; tol::Float64=1.0e-2, maxit::Int=50,
                           minwork::Int=0, chunkwork::Int=0)
    kind === :chol && return SemQuadS1Chol(F1)
    kind === :level &&
        return SemQuadS1Level(F1, S1s; nthreads=nthreads, minwork=minwork,
                              chunkwork=chunkwork)
    kind === :cg_diag &&
        return SemQuadS1CG(S1s; precond=:diag, tol=tol, maxit=maxit)
    kind === :cg_ichol &&
        return SemQuadS1CG(S1s; precond=:ichol, tol=tol, maxit=maxit)
    throw(ArgumentError("unknown bfbt_s1_solve=$kind " *
                        "(expected :chol, :level, :cg_diag, :cg_ichol)"))
end
