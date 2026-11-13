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

"""Assemble and solve the constant-viscosity quadtree Stokes system."""
function solve_semquad_stokes(ss::SemStokes, qt::Quadtree,
                              fx::Function, fz::Function,
                              uDx::Function, uDz::Function;
                              linear::Symbol=:direct, schur_maxiter::Int=200,
                              schur_precond::Symbol=:mass)
    linear in (:direct, :schur_cg) ||
        throw(ArgumentError("linear must be :direct or :schur_cg"))
    schur_precond in (:none, :mass) ||
        throw(ArgumentError("schur_precond must be :none or :mass"))
    p = ss.p; q = ss.q
    su = ss.su; sp = ss.sp
    nu = su.N; np = sp.N
    Nu2 = 2 * nu
    IK = Int[]; JK = Int[]; VK = Float64[]
    IB = Int[]; JB = Int[]; VB = Float64[]
    fv = zeros(Float64, Nu2)
    Axx, Azz, Axz, Bx, Bz = _sem_stokes_elem(p, q)
    n = p + 1
    for lf in su.leaves
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = su.elem[lf]
        pids = sp.elem[lf]
        # The reference-space element matrices are already the physical ones for
        # square leaves: the area jacobian (hx*hy/4) cancels the derivative
        # factors (2/hx)(2/hy) for the second-order terms, and leaves the aspect
        # ratios hy/hx and hx/hy (both 1 for the unit square).
        axx = hy / hx
        azz = hx / hy
        # scatter the velocity block (local order: (j-1)n+i for each component)
        for jb in 1:n, ib in 1:n, ja in 1:n, ia in 1:n
            ki = uids[(jb - 1) * n + ib]; kj = uids[(ja - 1) * n + ia]
            push!(IK, ki); push!(JK, kj)
            push!(VK, axx * Axx[(jb - 1) * n + ib, (ja - 1) * n + ia])
            push!(IK, nu + ki); push!(JK, nu + kj)
            push!(VK, azz * Azz[(jb - 1) * n + ib, (ja - 1) * n + ia])
            push!(IK, ki); push!(JK, nu + kj)
            push!(VK, Axz[(jb - 1) * n + ib, (ja - 1) * n + ia])
            push!(IK, nu + ki); push!(JK, kj)
            push!(VK, Axz[(ja - 1) * n + ia, (jb - 1) * n + ib])
        end
        # divergence coupling: int q d(ux)/dx = (hy/2) int_ref q d/dxi,
        #                      int q d(uz)/dy = (hx/2) int_ref q d/deta
        for b in 1:(q + 1), a in 1:(q + 1), j in 1:n, i in 1:n
            pi_ = pids[(b - 1) * (q + 1) + a]
            uj = uids[(j - 1) * n + i]
            push!(IB, pi_); push!(JB, uj)
            push!(VB, 0.5 * hy * Bx[(b - 1) * (q + 1) + a, (j - 1) * n + i])
            push!(IB, pi_); push!(JB, nu + uj)
            push!(VB, 0.5 * hx * Bz[(b - 1) * (q + 1) + a, (j - 1) * n + i])
        end
        # loads
        for j in 1:n, i in 1:n
            x = x0 + 0.5 * (su.xi[i] + 1.0) * hx
            y = y0 + 0.5 * (su.xi[j] + 1.0) * hy
            w = 0.25 * hx * hy * su.w[i] * su.w[j]
            g = uids[(j - 1) * n + i]
            fv[g] += w * fx(x, y)
            fv[nu + g] += w * fz(x, y)
        end
    end
    K = sparse(IK, JK, VK, Nu2, Nu2)
    B = sparse(IB, JB, VB, np, Nu2)
    # velocity Dirichlet on the domain boundary
    fixed = Dict{Int,Float64}()
    for lf in su.leaves
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = su.elem[lf]
        for dir in (:xminus, :xplus, :yminus, :yplus)
            has_no_face_neighbor(qt, lf, dir) || continue
            if dir === :xminus || dir === :xplus
                iF = dir === :xminus ? 1 : n
                xf = dir === :xplus ? x0 + hx : x0
                for j in 1:n
                    yv = y0 + 0.5 * (su.xi[j] + 1.0) * hy
                    g = uids[(j - 1) * n + iF]
                    fixed[g] = uDx(xf, yv)
                    fixed[nu + g] = uDz(xf, yv)
                end
            else
                jF = dir === :yminus ? 1 : n
                yf = dir === :yplus ? y0 + hy : y0
                for i in 1:n
                    xv = x0 + 0.5 * (su.xi[i] + 1.0) * hx
                    g = uids[(jF - 1) * n + i]
                    fixed[g] = uDx(xv, yf)
                    fixed[nu + g] = uDz(xv, yf)
                end
            end
        end
    end
    cons_u = semquad_constraints(su, qt)
    # lift to the two velocity components
    cons2 = Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
    for (c, (ms, ws)) in cons_u
        cons2[c] = (ms, ws)
        cons2[nu + c] = (nu .+ ms, ws)
    end
    u0, Cu, _ = _affine_constraint(Nu2, cons2, fixed)
    # pressure: hanging constraints and one pin
    cons_p = semquad_constraints(sp, qt)
    pf = Dict{Int,Float64}()
    pfree0 = setdiff(1:np, collect(keys(cons_p)))
    isempty(pfree0) || (pf[first(pfree0)] = 0.0)
    p0, Cp, _ = _affine_constraint(np, cons_p, pf)
    # reduced saddle system: K u - B' p = f, -B u = 0 (the weak form has
    # -int p div v - int q div u, so the pressure blocks carry a minus sign)
    Fp = nothing
    if linear === :schur_cg && schur_precond === :mass
        Mfull = assemble_semquad_pressure_mass(sp)
        Mred = Cp' * Mfull * Cp
        Fp = cholesky(Symmetric(Mred))
        issuccess(Fp) ||
            throw(ArgumentError("pressure mass preconditioner is not SPD"))
    end
    CpB = Cp' * B * Cu
    Kred = Cu' * K * Cu
    fr = Cu' * (fv - K * u0 + B' * p0)
    gr = Cp' * (B * u0)
    nf = size(Cu, 2); nfp = size(Cp, 2)
    Bs = -CpB
    Bt = sparse(transpose(Bs))
    nit = 0
    LAST_SEMSTOKES_SHIFT[] = 0.0
    LAST_SEMSTOKES_REFINE_ITERS[] = 0
    LAST_SEMSTOKES_LADDER[] = nothing
    LAST_SEMSTOKES_LADDER_PHASE[] = nothing
    if linear === :schur_cg
        # Saddle [Kred (-CpB'); (-CpB) 0] with Bs = -CpB, Bt = Bs':
        #   Kred u + Bt p = fr,   Bs u = gr,
        # solved by a matrix-free Schur CG on S = Bs Kred^{-1} Bt.
        # A constrained/hanging velocity mode can leave a zero eigenvalue in
        # Kred after AMR; regularize with a 1e-12 relative diagonal shift so the
        # sparse LU is defined.  The shift is below the G44 exactness tolerance.
        dmax = maximum(abs, diag(Kred))
        Kred = Kred + (1.0e-12 * max(dmax, 1.0)) * I
        # Row/column equilibration: wall-refined AMR trees have h_min/h_max
        # ~1e-4 and Kred condition numbers ~1e12; the scaled LU is much better
        # conditioned and removes the artificial large velocity from the shift.
        d = sqrt.(max.(diag(Kred), 1.0e-14 * dmax))
        Dinv = spdiagm(0 => 1.0 ./ d)
        Kscaled = Dinv * Kred * Dinv
        Fv = lu(Kscaled)
        ufree = Dinv * (Fv \ (Dinv * fr))
        r = Bs * ufree - gr
        pfree = zeros(Float64, nfp)
        z = Fp === nothing ? copy(r) : Fp \ r
        pp = copy(z)
        rho = dot(r, z)
        bnorm = max(dot(r, r), 1.0e-300)
        for it in 1:schur_maxiter
            nit = it
            Ap = Bs * (Dinv * (Fv \ (Dinv * (Bt * pp))))
            den = dot(pp, Ap)
            den == 0.0 && break
            alpha = rho / den
            pfree .+= alpha .* pp
            r .-= alpha .* Ap
            rs = dot(r, r)
            rs <= 1.0e-18 * bnorm && break
            z = Fp === nothing ? copy(r) : Fp \ r
            rho_new = dot(r, z)
            pp .= z .+ (rho_new / rho) .* pp
            rho = rho_new
        end
        ufree = Dinv * (Fv \ (Dinv * (fr - Bt * pfree)))
    else
        S = [Kred Bt; Bs spzeros(nfp, nfp)]
        rhs = vcat(fr, gr)
        if linear === :lagged
            # S2: reuse the previous LU as a preconditioner and polish with
            # iterative refinement; refactor only on a cache miss / stall.
            key = _saddle_key(qt, ss.su.p, bc, nf, nfp)
            sol, rn = _lagged_solve!(S, rhs, key; tol=1.0e-8, maxit=25)
            LAST_SEMSTOKES_REGULARIZED[] = false
            if rn > 1.0e-6
                dmax = maximum(abs, diag(S)) + 1.0
                Ssh = S + (1.0e-12 * dmax) * I
                sol, rn = _lagged_solve!(Ssh, rhs, key; tol=1.0e-8, maxit=6,
                                         force_refactor=true)
                LAST_SEMSTOKES_REGULARIZED[] = true
            end
        else
            sol = S \ rhs
            # See `solve_semquad_stokes_var`: the unshifted LU can be unstable on
            # free-slip wall trees; retry with a 1e-12 relative diagonal shift only
            # when the first solve leaves a bad residual.
            rden0 = max(norm(fr), norm(gr), 1.0)
            bad = norm(Kred * sol[1:nf] + Bt * sol[(nf + 1):(nf + nfp)] - fr) >
                  1.0e-8 * rden0 ||
                  norm(Bs * sol[1:nf] - gr) > 1.0e-8 * rden0
            LAST_SEMSTOKES_REGULARIZED[] = bad
            if bad
                dmax = maximum(abs, diag(S)) + 1.0
                sol = (S + (1.0e-12 * dmax) * I) \ rhs
            end
        end
        ufree = sol[1:nf]
        pfree = sol[nf+1:end]
    end
    # True reduced-saddle residual of the system that was actually solved
    # (Schur uses the regularized Kred).  This is what distinguishes a
    # converged solve from a singular/ill-conditioned direct solve.
    r1 = Kred * ufree + Bt * pfree - fr
    r2 = Bs * ufree - gr
    rden = max(norm(fr), norm(gr), 1.0)
    LAST_SEMSTOKES_RESIDUAL[] = max(norm(r1), norm(r2)) / rden
    LAST_SEMSTOKES_SCHUR_ITERS[] = nit
    u = u0 + Cu * ufree
    pv = p0 + Cp * pfree
    return u, pv, fixed, cons2, cons_p
end

"""Evaluate a quadtree velocity/pressure field at (x,y)."""
function semstokes_eval(ss::SemStokes, u::Vector{Float64}, pv::Vector{Float64},
                        x::Float64, y::Float64)
    nu = ss.su.N
    n = ss.p + 1
    rule = gll_rule(ss.p)
    for lf in ss.su.leaves
        hx = leaf_hx(ss.su, lf[1]); hy = leaf_hy(ss.su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        if x >= x0 - 1.0e-14 && x <= x0 + hx + 1.0e-14 &&
           y >= y0 - 1.0e-14 && y <= y0 + hy + 1.0e-14
            xi = 2.0 * (x - x0) / hx - 1.0
            eta = 2.0 * (y - y0) / hy - 1.0
            Lx, _ = lagrange_eval(rule, xi)
            Ly, _ = lagrange_eval(rule, eta)
            ids = ss.su.elem[lf]
            ux = 0.0; uz = 0.0
            for j in 1:n, i in 1:n
                g = ids[(j - 1) * n + i]
                ux += u[g] * Lx[i] * Ly[j]
                uz += u[nu + g] * Lx[i] * Ly[j]
            end
            q = ss.q
            nq = q + 1
            rq = gll_rule(q)
            Lxq, _ = lagrange_eval(rq, xi)
            Lyq, _ = lagrange_eval(rq, eta)
            pids = ss.sp.elem[lf]
            pp = 0.0
            for b in 1:nq, a in 1:nq
                pp += pv[pids[(b - 1) * nq + a]] * Lxq[a] * Lyq[b]
            end
            return ux, uz, pp
        end
    end
    throw(ArgumentError("point ($x,$y) not in the tree"))
end

