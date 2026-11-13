"""
    SemQuadBFBTDiagWork(nloc, nfp, hcap)

Per-block scratch of the leaf-local BFBT-diagonal assembly
([`_semquad_bfbt_diag_exact`](@ref)): one partial diagonal and the small
per-leaf buffers.  Each block owns its `d`, so the block partials are reduced in
fixed block order -- the result does not depend on how the blocks are scheduled
across threads.  `hcap` bounds the per-leaf master-list storage (two components
times the local row count times the largest row degree of `Cu`).
"""
struct SemQuadBFBTDiagWork
    d::Vector{Float64}         # partial diagonal over this block's leaves
    dloc::Vector{Float64}      # scalar leaf mass diagonal, (p+1)^2 entries
    free_pos::Matrix{Int}      # (p+1)^2 x 2: local position of a free row
    free_col::Matrix{Int}      # its Cu column
    nfree::Vector{Int}         # per component
    master::Matrix{Bool}       # free local positions that master a hanger here
    hpos::Vector{Int}          # constrained local positions (2*(p+1)^2)
    hptr::Vector{Int}          # start of each constrained row's entries
    hcol::Vector{Int}          # the constraint's master columns
    hval::Vector{Float64}      # and weights
    nh::Vector{Int}            # constrained rows per component
    tpos::Vector{Int}          # local position -> slot in `tidx`
    tidx::Vector{Int}          # positions whose mass entries are needed
    bt::Vector{Float64}        # basis values at one quadrature point
    m::Matrix{Float64}         # local mass submatrix over `tidx`
end

function SemQuadBFBTDiagWork(nloc::Int, nfp::Int, hcap::Int)
    hcap = max(hcap, 1)
    return SemQuadBFBTDiagWork(
        zeros(Float64, nfp), zeros(Float64, nloc),
        zeros(Int, nloc, 2), zeros(Int, nloc, 2), zeros(Int, 2),
        falses(nloc, 2),
        zeros(Int, 2 * nloc), zeros(Int, 2 * nloc + 1),
        zeros(Int, hcap), zeros(Float64, hcap),
        zeros(Int, 2), zeros(Int, nloc), zeros(Int, nloc),
        zeros(Float64, nloc), zeros(Float64, nloc, nloc))
end

"""
    _semquad_bfbt_diag_block!(wk, su, wfun, fd, rc, rv, rn, leaves, lo, hi)

One block of the leaf-local exact BFBT diagonal: scan `leaves[lo:hi]`, fold the
hanging-node constraint into the leaf mass block, and scatter-add the result
into `wk.d`.
"""
function _semquad_bfbt_diag_block!(wk::SemQuadBFBTDiagWork, su::SemQuad,
                                   wfun::Function, fd::FEData,
                                   rc::Vector{Int}, rv::Vector{Int},
                                   rn::Vector{Float64},
                                   leaves::Vector{NTuple{3,Int}}, lo::Int,
                                   hi::Int)
    p = su.p
    n = p + 1
    nloc = n * n
    N = su.N
    nq = fd.nq
    glx = fd.gl_x
    glw = fd.gl_w
    phi = fd.phi_p
    dloc = wk.dloc
    d = wk.d
    fill!(d, 0.0)
    @inbounds for li in lo:hi
        lf = leaves[li]
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = su.elem[lf]
        fill!(dloc, 0.0)
        fill!(wk.tpos, 0)
        fill!(wk.master, false)
        tlen = 0
        hcur = 1
        # ---- classify both components' local rows -----------------------
        for comp in 1:2
            off = (comp - 1) * N
            base = (comp - 1) * nloc
            nf = 0
            nh = 0
            for a in 1:nloc
                r = ids[a] + off
                lp = rc[r]; hp = rc[r + 1] - 1
                if hp < lp
                    continue                      # fixed / not a free dof
                elseif hp == lp && rn[lp] == 1.0
                    nf += 1
                    wk.free_pos[nf, comp] = a
                    wk.free_col[nf, comp] = rv[lp]
                else
                    nh += 1
                    wk.hpos[base + nh] = a
                    wk.hptr[base + nh] = hcur
                    for k in lp:hp
                        wk.hcol[hcur] = rv[k]
                        wk.hval[hcur] = rn[k]
                        hcur += 1
                    end
                end
            end
            wk.nfree[comp] = nf
            wk.nh[comp] = nh
            wk.hptr[base + nh + 1] = hcur
            # free positions that master a hanging row here
            for ih in 1:nh
                a = wk.hpos[base + ih]
                for k in wk.hptr[base + ih]:(wk.hptr[base + ih + 1] - 1)
                    j = wk.hcol[k]
                    for f in 1:nf
                        if wk.free_col[f, comp] == j
                            wk.master[wk.free_pos[f, comp], comp] = true
                        end
                    end
                end
                if wk.tpos[a] == 0
                    tlen += 1
                    wk.tidx[tlen] = a
                    wk.tpos[a] = tlen
                end
            end
            for f in 1:nf
                a = wk.free_pos[f, comp]
                if wk.master[a, comp] && wk.tpos[a] == 0
                    tlen += 1
                    wk.tidx[tlen] = a
                    wk.tpos[a] = tlen
                end
            end
        end
        # ---- one quadrature pass: leaf mass diagonal + the T sub-block --
        if tlen > 0
            fill!(view(wk.m, 1:tlen, 1:tlen), 0.0)
        end
        for qy in 1:nq, qx in 1:nq
            wq = 0.25 * hx * hy * glw[qx] * glw[qy] *
                 wfun(x0 + 0.5 * (glx[qx] + 1.0) * hx,
                      y0 + 0.5 * (glx[qy] + 1.0) * hy)
            for j in 1:n, i in 1:n
                ph = phi[i, qx] * phi[j, qy]
                dloc[(j - 1) * n + i] += wq * ph * ph
            end
            if tlen > 0
                for t in 1:tlen
                    a = wk.tidx[t]
                    i = (a - 1) % n + 1
                    j = (a - 1) ÷ n + 1
                    wk.bt[t] = phi[i, qx] * phi[j, qy]
                end
                for t2 in 1:tlen
                    b2 = wq * wk.bt[t2]
                    for t1 in 1:t2
                        wk.m[t1, t2] += wk.bt[t1] * b2
                    end
                end
            end
        end
        if tlen > 0
            for t2 in 2:tlen, t1 in 1:(t2 - 1)
                wk.m[t2, t1] = wk.m[t1, t2]
            end
        end
        # ---- fold the constraint into D ---------------------------------
        for comp in 1:2
            base = (comp - 1) * nloc
            nf = wk.nfree[comp]
            nh = wk.nh[comp]
            for f in 1:nf
                a = wk.free_pos[f, comp]
                wk.master[a, comp] || (d[wk.free_col[f, comp]] += dloc[a])
            end
            nh == 0 && continue
            # the free row itself (the 1 in the column)
            for f in 1:nf
                a = wk.free_pos[f, comp]
                if wk.master[a, comp]
                    t = wk.tpos[a]
                    d[wk.free_col[f, comp]] += wk.m[t, t]
                end
            end
            for ih in 1:nh
                a = wk.hpos[base + ih]
                ta = wk.tpos[a]
                p0 = wk.hptr[base + ih]
                p1 = wk.hptr[base + ih + 1] - 1
                # w_hj^2 M_hh for every master of this hanging row
                for k in p0:p1
                    w = wk.hval[k]
                    d[wk.hcol[k]] += w * w * wk.m[ta, ta]
                end
                # 2 w_hj w_h'j M_hh' for h' below h sharing a master
                for ih2 in (ih + 1):nh
                    a2 = wk.hpos[base + ih2]
                    ta2 = wk.tpos[a2]
                    q0 = wk.hptr[base + ih2]
                    q1 = wk.hptr[base + ih2 + 1] - 1
                    mval = wk.m[ta, ta2]
                    for k in p0:p1
                        j = wk.hcol[k]
                        w = wk.hval[k]
                        for q in q0:q1
                            if wk.hcol[q] == j
                                d[j] += 2.0 * w * wk.hval[q] * mval
                            end
                        end
                    end
                end
                # 2 w_hj M_jh when the master's own row is in this leaf
                for k in p0:p1
                    j = wk.hcol[k]
                    for f in 1:nf
                        if wk.free_col[f, comp] == j
                            af = wk.free_pos[f, comp]
                            d[j] += 2.0 * wk.hval[k] * wk.m[wk.tpos[af], ta]
                            break
                        end
                    end
                end
            end
        end
    end
    return nothing
end

"""
    _semquad_bfbt_diag_exact(su, Cu, wfun; nblocks=16) -> Vector{Float64}

Leaf-local exact BFBT weight `diag(Cu' M_u(w) Cu)`, `w = sqrt(eta)`: for every
free dof, the leaf mass contributions are summed with the hanging-node
constraint folded in *locally*, instead of forming the sparse product
`Cu' M Cu`.

For a free column `j` of `Cu` the entries are `1` at the free row `j` and the
constraint weights `w_hj` at the hanging rows `h` that name `j` as a master, so

    D_j = M_jj + 2 sum_h w_hj M_jh + sum_{h,h'} w_hj w_h'j M_hh'.

`M` is assembled from leaf blocks, so every term above is a sum of leaf-local
entries: `M_jj` is the leaf mass diagonal, and the coupling terms only exist
between rows that share a leaf (a hanging row shares its leaf with its
constraint's local masters).  The leaf loop therefore needs the mass diagonal
(for the free rows) and a small dense sub-block over the leaf's hanging rows
plus the local free rows that master them.  The right-hand sides never read
the global free-dof numbering except through `Cu` itself, so the construction
also handles the case where the two velocity components have different free
sets (free-slip walls constrain different rows for `u` and `w`).

The leaves are split into a fixed `nblocks` contiguous blocks and the block
partials are reduced in block order, so the diagonal is bitwise identical at 1
and 8 Julia threads.  It equals the sparse route to round-off (different
summation order), not bitwise; the gate `G226` pins the difference.
"""
function _semquad_bfbt_diag_exact(su::SemQuad,
                                  Cu::SparseMatrixCSC{Float64,Int},
                                  wfun::Function; nblocks::Int=16)
    p = su.p
    n = p + 1
    nloc = n * n
    nfp = size(Cu, 2)
    fd = make_fedata(2, p)
    Ct = sparse(transpose(Cu))
    rc = Ct.colptr
    rv = Ct.rowval
    rn = nonzeros(Ct)
    nleaf = length(su.leaves)
    # `hcol`/`hval` hold the constrained rows' master lists for one leaf and
    # both components.  `maximum(diff(rc))` is the largest row degree in `Cu`
    # (3 on the production trees, but a multi-level chain can be denser), so
    # this bound cannot overflow.
    maxdeg = isempty(rn) ? 1 : maximum(diff(rc))
    nb = clamp(nblocks, 1, max(nleaf, 1))
    works = Vector{SemQuadBFBTDiagWork}(undef, nb)
    los = Vector{Int}(undef, nb)
    his = Vector{Int}(undef, nb)
    base, rem = divrem(nleaf, nb)
    off = 1
    for b in 1:nb
        cnt = base + (b <= rem ? 1 : 0)
        los[b] = off
        his[b] = off + cnt - 1
        off += cnt
        works[b] = SemQuadBFBTDiagWork(nloc, nfp, 2 * nloc * maxdeg)
    end
    if nb == 1
        _semquad_bfbt_diag_block!(works[1], su, wfun, fd, rc, rv, rn, su.leaves,
                                  los[1], his[1])
    else
        Threads.@threads :static for b in 1:nb
            _semquad_bfbt_diag_block!(works[b], su, wfun, fd, rc, rv, rn,
                                      su.leaves, los[b], his[b])
        end
    end
    d = works[1].d
    for b in 2:nb
        d .+= works[b].d
    end
    return d
end

"""
    _semquad_bfbt_diag(su, Cu, eta_fun, wfun; how=:leaf)

`D = diag(Cu' M_u(sqrt(eta)) Cu)` for the BFBT factors: the diagonal of the
velocity mass matrix, restricted to the free dofs of the reduced saddle.

* `how=:exact` ([`_semquad_bfbt_diag_exact`](@ref)) is the exact restricted
  diagonal computed leaf by leaf with the hanging-node constraint folded in
  locally.  It is the shipped default on the matrix-free path (see
  `mg_pressure_bfbt_d`) and equals the `:matrix` route to round-off.
* `how=:leaf` is the historical *lumped* restriction: the leaf mass diagonal
  restricted as `Cux' diag(M) + Cuz' diag(M)`.  A free column of `Cu` also
  appears as a master in its hanging-node rows, so `diag(C' M C)` folds cross
  terms in that the lumped form misses (1.54 % relative change on the
  production `F` tree, 0.086 % on `P1`) and the lumped form can turn a hanging
  row non-positive on `F` (16 dofs, clipped to a 1e12 contrast).  The build
  cost of the two leaf paths is the same: both are dominated by the
  leaf/quadrature-point `wfun` evaluations.
* `how=:matrix` is the historical sparse reference (full `N x N` assembly +
  `Cux' Mv Cux + Cuz' Mv Cuz`), kept for the gate that cross-checks the leaf
  paths.  On `F` it costs 0.87-0.99 s.
"""
function _semquad_bfbt_diag(su::SemQuad, Cu::SparseMatrixCSC{Float64,Int},
                            eta_fun::Function, wfun::Function;
                            how::Symbol=:leaf)
    nsu = su.N
    d = if how === :matrix
        Cux = Cu[1:nsu, :]
        Cuz = Cu[(nsu + 1):(2 * nsu), :]
        Mv = assemble_semquad_pressure_mass(su; weight=wfun)
        Mv_red = Cux' * Mv * Cux + Cuz' * Mv * Cuz
        Vector{Float64}(diag(Mv_red))
    elseif how === :exact
        _semquad_bfbt_diag_exact(su, Cu, wfun)
    elseif how === :leaf
        Cux = Cu[1:nsu, :]
        Cuz = Cu[(nsu + 1):(2 * nsu), :]
        dfull = zeros(Float64, nsu)
        _semquad_mass_diag!(dfull, su, wfun)
        Vector{Float64}(transpose(Cux) * dfull) +
            Vector{Float64}(transpose(Cuz) * dfull)
    else
        throw(ArgumentError("_semquad_bfbt_diag: how must be :leaf, :exact or " *
                            ":matrix, got $(how)"))
    end
    # Positive definiteness of `D` is what makes `S1 = B D^{-1} B'` SPD; the
    # lumped diagonal of a hanging row can come out non-positive (Lagrange
    # weights change sign), so clip it the way `_semquad_diag_reduced` does.
    dmax = maximum(d)
    dmax > 0.0 || throw(ArgumentError("BFBT velocity-mass diagonal is not positive"))
    @inbounds for i in eachindex(d)
        d[i] = max(d[i], 1.0e-12 * dmax)
    end
    return d
end
