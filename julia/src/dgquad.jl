# DG Poisson operator on the quadtree (stage B).
#
# This file implements the volume, boundary (Nitsche) and *same-level* face
# terms on the quadtree leaves, reusing the verified formulas of `dg2d.jl`.
# Coarse-fine (2:1) sub-faces are detected and rejected with an explicit error
# for now: they need the coarse-side interpolation to the fine quadrature and
# are added in the next step, so the current operator can never silently
# produce a wrong hanging-face result.

struct DGQuad
    p::Int
    leaves::Vector{NTuple{3,Int}}
    base::Dict{NTuple{3,Int},Int}
    N::Int
    xi::Vector{Float64}
    w::Vector{Float64}
    D::Matrix{Float64}
end

function make_dgquad(qt::Quadtree, p::Int)
    p >= 1 || throw(ArgumentError("p must be >= 1"))
    rule = gll_rule(p)
    n = p + 1
    D = zeros(Float64, n, n)
    for i in 1:n
        _, dL = lagrange_eval(rule, rule.nodes[i])
        for j in 1:n
            D[i, j] = dL[j]
        end
    end
    leaves = sort(collect(qt.leaves))
    base = Dict{NTuple{3,Int},Int}()
    off = 0
    for lf in leaves
        base[lf] = off
        off += n^2
    end
    DGQuad(p, leaves, base, off, copy(rule.nodes), copy(rule.weights), D)
end

@inline _hleaf(lf::NTuple{3,Int}) = 2.0^(-lf[1])
@inline _gidq(dg::DGQuad, lf::NTuple{3,Int}, i::Int, j::Int) =
    dg.base[lf] + (j - 1) * (dg.p + 1) + i

"""True if the tree has no coarse-fine (hanging) faces."""
function is_uniform(qt::Quadtree)
    return all(abs(pf.len - _hleaf(pf.leafL)) < 1e-15 for pf in face_pairs(qt))
end

"""SIPG terms of one interior sub-face (fine-side quadrature, coarse interpolation)."""
function _add_subface!(A, dg::DGQuad, L::NTuple{3,Int}, R::NTuple{3,Int},
                       dir::Symbol, kappa::Float64, sigma_C::Float64)
    p = dg.p
    n = p + 1
    hL = _hleaf(L); hR = _hleaf(R)
    xL = L[2] * hL; yL = L[3] * hL
    xR = R[2] * hR; yR = R[3] * hR
    wq = 0.5 * hR .* dg.w
    sig = sigma_C * (p + 1)^2 / hR
    vertical = dir === :xplus || dir === :xminus
    nsign = dir === :xplus || dir === :yplus ? 1.0 : -1.0
    BL = zeros(n, n)
    if vertical
        iFL = dir === :xplus ? n : 1
        iFR = dir === :xplus ? 1 : n
        pos = yR .+ 0.5 .* (dg.xi .+ 1) .* hR
        xiL = 2.0 .* (pos .- yL) ./ hL .- 1.0
        for q in 1:n
            Lb, _ = lagrange_eval(gll_rule(p), xiL[q])
            BL[q, :] .= Lb
        end
        cL = (2.0 / hL) .* dg.D[iFL, :]
        cR = (2.0 / hR) .* dg.D[iFR, :]
    else
        iFL = dir === :yplus ? n : 1
        iFR = dir === :yplus ? 1 : n
        pos = xR .+ 0.5 .* (dg.xi .+ 1) .* hR
        xiL = 2.0 .* (pos .- xL) ./ hL .- 1.0
        for q in 1:n
            Lb, _ = lagrange_eval(gll_rule(p), xiL[q])
            BL[q, :] .= Lb
        end
        cL = (2.0 / hL) .* dg.D[iFL, :]
        cR = (2.0 / hR) .* dg.D[iFR, :]
    end
    BR = Matrix{Float64}(I, n, n)
    trace(B, iF, i, j, q) = vertical ? (i == iF ? B[q, j] : 0.0) :
                                        (j == iF ? B[q, i] : 0.0)
    deriv(B, c, i, j, q) = vertical ? c[i] * B[q, j] : c[j] * B[q, i]
    gL(i, j) = _gidq(dg, L, i, j)
    gR(i, j) = _gidq(dg, R, i, j)
    for (gt, Bt, ct, iFt, st) in ((gL, BL, cL, iFL, 1.0), (gR, BR, cR, iFR, -1.0))
        for (gu, Bu, cu, iFu, su) in ((gL, BL, cL, iFL, 1.0), (gR, BR, cR, iFR, -1.0))
            for jt in 1:n, it in 1:n, ju in 1:n, iu in 1:n
                a = 0.0
                for q in 1:n
                    tt = trace(Bt, iFt, it, jt, q)
                    tu = trace(Bu, iFu, iu, ju, q)
                    dt = deriv(Bt, ct, it, jt, q)
                    du = deriv(Bu, cu, iu, ju, q)
                    a += wq[q] * (-0.5 * nsign * st * du * tt
                                  - 0.5 * nsign * su * dt * tu
                                  + sig * su * st * tu * tt)
                end
                a == 0.0 && continue
                A[gt(it, jt), gu(iu, ju)] += a
            end
        end
    end
    return A
end

"""
    assemble_dgquad_poisson(dg, qt, f, uD; kappa=1.0, sigma_C=50.0)

SIPG Poisson operator on the quadtree, Dirichlet on all outer sides.  Throws
`ArgumentError` if the tree has coarse-fine faces (not implemented yet).
"""
function assemble_dgquad_poisson(dg::DGQuad, qt::Quadtree, f::Function, uD::Function;
                                 kappa::Float64=1.0, sigma_C::Float64=50.0)
    p = dg.p
    n = p + 1
    N = dg.N
    A = zeros(Float64, N, N)
    b = zeros(Float64, N)
    W = Diagonal(dg.w)
    DW = dg.D' * W * dg.D
    Ke = kappa .* (kron(W, DW) .+ kron(DW, W))
    for lf in dg.leaves
        idx = dg.base[lf] .+ (1:n^2)
        A[idx, idx] .+= Ke
        h = _hleaf(lf)
        x0 = lf[2] * h; y0 = lf[3] * h
        for j in 1:n, i in 1:n
            x = x0 + 0.5 * (dg.xi[i] + 1) * h
            y = y0 + 0.5 * (dg.xi[j] + 1) * h
            b[_gidq(dg, lf, i, j)] += 0.25 * h * h * dg.w[i] * dg.w[j] * f(x, y)
        end
    end
    # interior sub-faces (same-level and 2:1 coarse-fine)
    for pf in face_pairs(qt)
        _add_subface!(A, dg, pf.leafL, pf.leafR, pf.dir, kappa, sigma_C)
    end
    # Dirichlet boundaries
    for lf in dg.leaves
        h = _hleaf(lf)
        sig = sigma_C * (p + 1)^2 / h
        x0 = lf[2] * h; y0 = lf[3] * h
        for dir in FACE_DIRS
            has_no_face_neighbor(qt, lf, dir) || continue
            if dir === :xminus
                c = (2.0 / h) .* dg.D[1, :]
                for j in 1:n
                    wf = 0.5 * h * dg.w[j]
                    iF = _gidq(dg, lf, 1, j)
                    Td = uD(x0, y0 + 0.5 * (dg.xi[j] + 1) * h)
                    for i in 1:n
                        gi = _gidq(dg, lf, i, j)
                        A[iF, gi] += wf * c[i]
                        A[gi, iF] += wf * c[i]
                        b[gi] += wf * c[i] * Td
                    end
                    A[iF, iF] += sig * wf
                    b[iF] += sig * wf * Td
                end
            elseif dir === :xplus
                c = (2.0 / h) .* dg.D[n, :]
                for j in 1:n
                    wf = 0.5 * h * dg.w[j]
                    iF = _gidq(dg, lf, n, j)
                    Td = uD(x0 + h, y0 + 0.5 * (dg.xi[j] + 1) * h)
                    for i in 1:n
                        gi = _gidq(dg, lf, i, j)
                        A[iF, gi] += -wf * c[i]
                        A[gi, iF] += -wf * c[i]
                        b[gi] += -wf * c[i] * Td
                    end
                    A[iF, iF] += sig * wf
                    b[iF] += sig * wf * Td
                end
            elseif dir === :yminus
                c = (2.0 / h) .* dg.D[1, :]
                for i in 1:n
                    wf = 0.5 * h * dg.w[i]
                    iF = _gidq(dg, lf, i, 1)
                    Td = uD(x0 + 0.5 * (dg.xi[i] + 1) * h, y0)
                    for j in 1:n
                        gj = _gidq(dg, lf, i, j)
                        A[iF, gj] += wf * c[j]
                        A[gj, iF] += wf * c[j]
                        b[gj] += wf * c[j] * Td
                    end
                    A[iF, iF] += sig * wf
                    b[iF] += sig * wf * Td
                end
            else
                c = (2.0 / h) .* dg.D[n, :]
                for i in 1:n
                    wf = 0.5 * h * dg.w[i]
                    iF = _gidq(dg, lf, i, n)
                    Td = uD(x0 + 0.5 * (dg.xi[i] + 1) * h, y0 + h)
                    for j in 1:n
                        gj = _gidq(dg, lf, i, j)
                        A[iF, gj] += -wf * c[j]
                        A[gj, iF] += -wf * c[j]
                        b[gj] += -wf * c[j] * Td
                    end
                    A[iF, iF] += sig * wf
                    b[iF] += sig * wf * Td
                end
            end
        end
    end
    return A, b
end

"""Evaluate a DGQuad field at (x,y) on the leaf containing the point."""
function dgquad_eval(dg::DGQuad, u::Vector{Float64}, x::Float64, y::Float64)
    n = dg.p + 1
    for lf in dg.leaves
        h = _hleaf(lf)
        x0 = lf[2] * h; y0 = lf[3] * h
        if x >= x0 - 1e-14 && x <= x0 + h + 1e-14 &&
           y >= y0 - 1e-14 && y <= y0 + h + 1e-14
            xi = 2.0 * (x - x0) / h - 1.0
            eta = 2.0 * (y - y0) / h - 1.0
            Lx, _ = lagrange_eval(gll_rule(dg.p), xi)
            Ly, _ = lagrange_eval(gll_rule(dg.p), eta)
            val = 0.0
            for j in 1:n, i in 1:n
                val += u[_gidq(dg, lf, i, j)] * Lx[i] * Ly[j]
            end
            return val
        end
    end
    throw(ArgumentError("point ($x,$y) not in the tree"))
end


# ---------------------------------------------------------------------------
# Gradient-jump error indicator and one AMR cycle (stage B).
# ---------------------------------------------------------------------------

"""Normal-derivative traces of one sub-face at the fine-side quadrature points."""
function _subface_derivs(dg::DGQuad, L::NTuple{3,Int}, R::NTuple{3,Int},
                         dir::Symbol, u::Vector{Float64})
    p = dg.p
    n = p + 1
    hL = _hleaf(L); hR = _hleaf(R)
    xL = L[2] * hL; yL = L[3] * hL
    xR = R[2] * hR; yR = R[3] * hR
    wq = 0.5 * hR .* dg.w
    vertical = dir === :xplus || dir === :xminus
    BL = zeros(n, n)
    if vertical
        iFL = dir === :xplus ? n : 1
        iFR = dir === :xplus ? 1 : n
        pos = yR .+ 0.5 .* (dg.xi .+ 1) .* hR
        xiL = 2.0 .* (pos .- yL) ./ hL .- 1.0
        for q in 1:n
            Lb, _ = lagrange_eval(gll_rule(p), xiL[q])
            BL[q, :] .= Lb
        end
        cL = (2.0 / hL) .* dg.D[iFL, :]
        cR = (2.0 / hR) .* dg.D[iFR, :]
        gidL = (i, a) -> _gidq(dg, L, i, a)
        gidR = (i, q) -> _gidq(dg, R, i, q)
    else
        iFL = dir === :yplus ? n : 1
        iFR = dir === :yplus ? 1 : n
        pos = xR .+ 0.5 .* (dg.xi .+ 1) .* hR
        xiL = 2.0 .* (pos .- xL) ./ hL .- 1.0
        for q in 1:n
            Lb, _ = lagrange_eval(gll_rule(p), xiL[q])
            BL[q, :] .= Lb
        end
        cL = (2.0 / hL) .* dg.D[iFL, :]
        cR = (2.0 / hR) .* dg.D[iFR, :]
        gidL = (i, a) -> _gidq(dg, L, a, i)
        gidR = (i, q) -> _gidq(dg, R, q, i)
    end
    dL = zeros(n); dR = zeros(n)
    for q in 1:n
        for a in 1:n
            da = 0.0
            for i in 1:n
                da += cL[i] * u[gidL(i, a)]
            end
            dL[q] += BL[q, a] * da
        end
        for i in 1:n
            dR[q] += cR[i] * u[gidR(i, q)]
        end
    end
    return wq, dL, dR
end

"""Gradient-jump indicator per leaf: (h/2) * sum_faces int [du.n]^2 ds."""
function dgquad_jump_indicator(dg::DGQuad, qt::Quadtree, u::Vector{Float64})
    eta = Dict{NTuple{3,Int},Float64}()
    for lf in dg.leaves
        eta[lf] = 0.0
    end
    for pf in face_pairs(qt)
        L, R, dir = pf.leafL, pf.leafR, pf.dir
        wq, dL, dR = _subface_derivs(dg, L, R, dir, u)
        J = 0.0
        for q in eachindex(wq)
            J += wq[q] * (dL[q] - dR[q])^2
        end
        hL = _hleaf(L); hR = _hleaf(R)
        eta[L] += 0.5 * hL * J
        eta[R] += 0.5 * hR * J
    end
    return eta
end

"""One AMR cycle: mark top/bottom fractions, refine/coarsen, balance.

`marking=:global` refines the globally largest indicators (the DG prototype
behaviour).  `marking=:level` refines the largest `refine_frac` fraction *within
each level*; this prevents a heavily refined wall (many large-gradient leaves)
from consuming the whole budget and leaves room for interior plumes/hotspots.
`marking=:error_fraction` uses ASPECT's `refine_and_coarsen_fixed_fraction`
semantics (Dörfler marking): `refine_frac`/`coarsen_frac` are fractions of the
*total error*, not of the cell count.  `max_leaves` is a hard post-balance
budget (default: none).
`max_level` prevents refinement above that quadtree level (default: no cap);
coarsening is unaffected. `symmetry=:x` marks the mirror leaf
`(l, 2^l-1-x, z)` together with every selected leaf, preserving an initially
mirror-symmetric mesh: every pass is pair-strict (refine both or neither, merge
both or neither), `balance!` splits mirror pairs and `mirror_close!` repairs a
tree that was damaged before the fix.  `protect` is a set of leaves that must not
be coarsened (representation-error guard).  Returns the number of refined and
coarsened leaves."""
function _refine_pair!(qt::Quadtree, lf::NTuple{3,Int}, max_level::Int,
                       symmetry::Symbol)
    lf[1] < max_level || return 0
    if symmetry === :x
        # pair-strict: never split one side of a mirror pair alone.  In a closed
        # tree the partner is always a leaf, so this only rejects legacy
        # asymmetric input (which `mirror_close!` then repairs).
        ml = _mirror_leaf(qt, lf)
        (ml == lf || (ml[1] < max_level && ml in qt.leaves)) || return 0
        refine!(qt, lf) || return 0
        n = 1
        if ml != lf && refine!(qt, ml)
            n += 1
        end
        return n
    end
    refine!(qt, lf) || return 0
    return 1
end

function _coarsen_pair!(qt::Quadtree, lf::NTuple{3,Int}, symmetry::Symbol,
                        protect::Set{NTuple{3,Int}})
    # coarsen! merges the whole sibling quad; protect if any sibling is
    # protected, otherwise the protected leaf would disappear indirectly.
    bx = lf[2] - lf[2] % 2
    by = lf[3] - lf[3] % 2
    for dy in 0:1, dx in 0:1
        (lf[1], bx + dx, by + dy) in protect && return 0
    end
    ml = lf
    if symmetry === :x
        # pair-strict: the mirror quartet has to be complete and unprotected
        # *before* either side is merged, otherwise one side would be merged
        # alone and the tree loses the invariant (5 non-mirrored leaves per
        # one-sided merge).
        ml = _mirror_leaf(qt, lf)
        if ml != lf
            mbx = ml[2] - ml[2] % 2
            mby = ml[3] - ml[3] % 2
            for dy in 0:1, dx in 0:1
                k = (ml[1], mbx + dx, mby + dy)
                (k in qt.leaves && !(k in protect)) || return 0
            end
        end
    end
    coarsen!(qt, lf) || return 0
    n = 1
    if symmetry === :x && ml != lf && coarsen!(qt, ml)
        n += 1
    end
    return n
end

function amr_cycle!(qt::Quadtree, eta::Dict{NTuple{3,Int},Float64};
                    refine_frac::Float64=0.3, coarsen_frac::Float64=0.0,
                    marking::Symbol=:global,
                    max_level::Int=typemax(Int),
                    max_leaves::Int=typemax(Int),
                    symmetry::Symbol=:none,
                    protect::Set{NTuple{3,Int}}=Set{NTuple{3,Int}}())
    marking in (:global, :level, :error_fraction) ||
        throw(ArgumentError("marking must be :global, :level or :error_fraction"))
    symmetry in (:none, :x) ||
        throw(ArgumentError("symmetry must be :none or :x"))
    max_level >= 1 || throw(ArgumentError("max_level must be >= 1"))
    max_leaves >= 1 || throw(ArgumentError("max_leaves must be >= 1"))
    # `symmetry=:x` works with a mirror-closed protect set, so a pair is not
    # refused just because only one side's `coarsen_theta_tol` guard fired.
    protect = symmetry === :x ? _mirror_protect(qt, protect) : protect
    leaves = sort(collect(qt.leaves), by=lf -> (-eta[lf], lf))
    ncoa = round(Int, coarsen_frac * length(leaves))
    nr = 0
    nc = 0
    if marking === :error_fraction
        # ASPECT's refine_and_coarsen_fixed_fraction semantics: refine the
        # largest-error cells until their cumulative error reaches
        # refine_frac of the total, and coarsen the smallest-error cells until
        # coarsen_frac of the total.  This is Dörfler marking, not a fixed
        # fraction of cells.
        total = sum(max(eta[lf], 0.0) for lf in leaves)
        if total > 0.0
            target = refine_frac * total
            cum = 0.0
            for lf in leaves
                lf[1] < max_level || continue
                cum >= target && break
                # hard leaf budget: a refined leaf adds 4-1 = 3 leaves (and 6
                # more with :x symmetry), so stop before crossing max_leaves
                length(qt.leaves) + 10 > max_leaves && break
                nr += _refine_pair!(qt, lf, max_level, symmetry)
                cum += max(eta[lf], 0.0)
                if symmetry === :x
                    ml = _mirror_leaf(qt, lf)
                    cum += max(get(eta, ml, 0.0), 0.0)
                end
            end
        end
        if coarsen_frac > 0.0 && total > 0.0
            target_c = coarsen_frac * total
            cum_c = 0.0
            for lf in reverse(leaves)
                cum_c >= target_c && break
                nc += _coarsen_pair!(qt, lf, symmetry, protect)
                cum_c += max(eta[lf], 0.0)
                if symmetry === :x
                    ml = _mirror_leaf(qt, lf)
                    cum_c += max(get(eta, ml, 0.0), 0.0)
                end
            end
        end
    elseif marking === :level
        for l in sort(unique(lf[1] for lf in leaves))
            l < max_level || continue
            lv = [lf for lf in leaves if lf[1] == l]
            nl = round(Int, refine_frac * length(lv))
            for k in 1:min(nl, length(lv))
                length(qt.leaves) + 10 > max_leaves && break
                nr += _refine_pair!(qt, lv[k], max_level, symmetry)
            end
        end
    else
        cand = [lf for lf in leaves if lf[1] < max_level]
        nref = round(Int, refine_frac * length(cand))
        for k in 1:min(nref, length(cand))
            length(qt.leaves) + 10 > max_leaves && break
            nr += _refine_pair!(qt, cand[k], max_level, symmetry)
        end
    end
    if marking !== :error_fraction && ncoa > 0
        for k in (length(leaves)-ncoa+1):length(leaves)
            nc += _coarsen_pair!(qt, leaves[k], symmetry, protect)
        end
    end
    balance!(qt; symmetry=symmetry)
    # `symmetry=:x`: repair a tree damaged before this pass (a legacy
    # checkpoint), then re-balance.  The repair only refines the coarser side of
    # a mismatch, so it can leave the tree temporarily 2:1-unbalanced; the
    # mirror-aware `balance!` fixes that and preserves closure (it splits pairs).
    # On a closed tree the repair is a no-op: one O(#leaves) hash lookup each.
    if symmetry === :x
        mirror_close!(qt; max_level=max_level)
        balance!(qt; symmetry=:x)
    end
    # Hard leaf budget: the pre-cycle check can overshoot through 2:1
    # balancing, so enforce the budget after balancing by coarsening the
    # lowest-error current leaves.
    #
    # The merge is undone whenever 2:1 balancing has to re-split the parent
    # (its neighbor is one level finer than the parent).  The lowest-error
    # quartet is then coarsened and re-split forever: on the Ra=1e8 blob step
    # that hit `max_leaves` (958 -> 1030 leaves) this loop ran its full
    # `guard = 10000` iterations, each one a full-tree `balance!` (2 passes,
    # ~8.2k `neighbors` calls, ~2.1e5 `_collect_face!` visits), for 31.5 s of
    # pure no-op work.  Detect the exact fixed point (the leaf set is unchanged
    # after coarsen+balance) and stop: every other quartet has a higher
    # indicator, so the old loop could never reach it either.  Over-budget
    # trees are then merged down by `semquad_enforce_leaf_budget!` in
    # `semquad_amr_passes!`, which has a `blocked` set for exactly this case.
    #
    # With `symmetry=:x` the merge is atomic over the mirror pair
    # (`_coarsen_pair!`): a pair whose mirror quartet is protected or incomplete
    # is refused, so the invariant survives at the cost of leaving the tree up to
    # one pair (6 leaves) above the budget.
    guard = 0
    while length(qt.leaves) > max_leaves && guard < 10000
        guard += 1
        before = sort(collect(qt.leaves))
        cur = sort(collect(qt.leaves), by=lf -> (get(eta, lf, 0.0), lf))
        progressed = false
        for lf in cur
            if symmetry === :x
                m = _coarsen_pair!(qt, lf, symmetry, protect)
                m > 0 || continue
                nc += m
                progressed = true
                break
            end
            lf in protect && continue
            coarsen!(qt, lf) || continue
            nc += 1
            progressed = true
            break
        end
        progressed || break
        balance!(qt; symmetry=symmetry)
        # A successful merge drops the leaf count by 3 (6 with :x symmetry);
        # an unchanged count means balance re-split the parent, i.e. the exact
        # fixed point of the old loop.
        if length(qt.leaves) >= length(before) &&
           sort(collect(qt.leaves)) == before
            break
        end
    end
    return nr, nc
end

"""
    run_dgquad_advection_subcycled(dg, qt, ux, uy, T0; tstop, cfl, inflow,
                                   subcycle=true)

Explicit DG advection with **level-local time steps**:

    dt_l = cfl * h_l / |u|,   m_l = round(dt_l / dt_f),

where `dt_f` is the finest-level step.  Each fine step evaluates the global DG
operator once; a leaf at level `l` is updated only every `m_l` fine steps with
its own `dt_l`.  `subcycle=false` uses `dt_f` for every leaf and is the
reference integrator.  This is a first-order prototype for the subcycling
needed by the wall-refined SEM convection runs.
"""
function run_dgquad_advection_subcycled(dg::DGQuad, qt::Quadtree,
                                        ux::Float64, uy::Float64,
                                        T0::Vector{Float64};
                                        tstop::Float64,
                                        cfl::Float64=0.3,
                                        inflow::Function=(x, y) -> 0.0,
                                        subcycle::Bool=true)
    speed = hypot(ux, uy)
    speed > 0.0 || throw(ArgumentError("advection speed must be nonzero"))
    A, b = assemble_dgquad_advection(dg, qt, ux, uy, inflow)
    h = [_hleaf(lf) for lf in dg.leaves]
    dtl = cfl .* h ./ speed
    dtf = minimum(dtl)
    m = subcycle ? [max(1, round(Int, dtl[k] / dtf)) for k in eachindex(dtl)] :
                   ones(Int, length(dtl))
    nsteps = max(1, ceil(Int, tstop / dtf))
    T = copy(T0)
    nv = dg.p + 1
    nv2 = nv * nv
    for s in 1:nsteps
        dT = b - A * T
        for (k, lf) in enumerate(dg.leaves)
            s % m[k] == 0 || continue
            off = dg.base[lf]
            step = subcycle ? dtl[k] : dtf
            for q in 1:nv2
                T[off + q] += step * dT[off + q]
            end
        end
    end
    return T
end


# ---------------------------------------------------------------------------
# Steady advection on the quadtree (upwind, coarse-fine interpolation).
# ---------------------------------------------------------------------------

"""Return `(A, b)` for `u . grad T = 0` with inflow values on `u.n < 0`."""
function assemble_dgquad_advection(dg::DGQuad, qt::Quadtree, ux::Float64,
                                   uy::Float64, inflow::Function)
    p = dg.p
    n = p + 1
    N = dg.N
    A = zeros(Float64, N, N)
    b = zeros(Float64, N)
    w = dg.w
    M1 = Diagonal(w)
    DW = dg.D' * M1
    for lf in dg.leaves
        h = _hleaf(lf)
        idx = dg.base[lf] .+ (1:n^2)
        Ke = -ux * 0.5 * h .* kron(M1, DW) .- uy * 0.5 * h .* kron(DW, M1)
        A[idx, idx] .+= Ke
    end
    # interior sub-faces
    for pf in face_pairs(qt)
        L, R, dir = pf.leafL, pf.leafR, pf.dir
        hL = _hleaf(L); hR = _hleaf(R)
        xL = L[2] * hL; yL = L[3] * hL
        xR = R[2] * hR; yR = R[3] * hR
        wq = 0.5 * hR .* w
        vertical = dir === :xplus || dir === :xminus
        nsign = dir === :xplus || dir === :yplus ? 1.0 : -1.0
        BL = zeros(n, n)
        if vertical
            iFL = dir === :xplus ? n : 1
            iFR = dir === :xplus ? 1 : n
            pos = yR .+ 0.5 .* (dg.xi .+ 1) .* hR
            xiL = 2.0 .* (pos .- yL) ./ hL .- 1.0
            for q in 1:n
                Lb, _ = lagrange_eval(gll_rule(p), xiL[q])
                BL[q, :] .= Lb
            end
            un = nsign * ux
            gL = (i, a) -> _gidq(dg, L, i, a)
            gR = (i, q) -> _gidq(dg, R, i, q)
        else
            iFL = dir === :yplus ? n : 1
            iFR = dir === :yplus ? 1 : n
            pos = xR .+ 0.5 .* (dg.xi .+ 1) .* hR
            xiL = 2.0 .* (pos .- xL) ./ hL .- 1.0
            for q in 1:n
                Lb, _ = lagrange_eval(gll_rule(p), xiL[q])
                BL[q, :] .= Lb
            end
            un = nsign * uy
            gL = (i, a) -> _gidq(dg, L, a, i)
            gR = (i, q) -> _gidq(dg, R, q, i)
        end
        if un >= 0.0
            for b in 1:n, a in 1:n
                c = 0.0
                for q in 1:n
                    c += wq[q] * BL[q, b] * BL[q, a]
                end
                A[gL(iFL, b), gL(iFL, a)] += un * c
            end
            for qp in 1:n, a in 1:n
                # the R element's outward normal is opposite to L's
                A[gR(iFR, qp), gL(iFL, a)] -= un * wq[qp] * BL[qp, a]
            end
        else
            for b in 1:n, q in 1:n
                A[gL(iFL, b), gR(iFR, q)] += un * wq[q] * BL[q, b]
            end
            for qp in 1:n
                A[gR(iFR, qp), gR(iFR, qp)] -= un * wq[qp]
            end
        end
    end
    # domain boundaries: the boundary term int (u.n) T_up v is present on
    # every boundary face (outflow uses the interior trace, inflow the data)
    for lf in dg.leaves
        h = _hleaf(lf)
        x0 = lf[2] * h; y0 = lf[3] * h
        for dir in FACE_DIRS
            has_no_face_neighbor(qt, lf, dir) || continue
            nsign = dir === :xplus || dir === :yplus ? 1.0 : -1.0
            if dir === :xminus || dir === :xplus
                un = nsign * ux
                iF = dir === :xplus ? n : 1
                for j in 1:n
                    wf = 0.5 * h * w[j]
                    xf = dir === :xplus ? x0 + h : x0
                    g = _gidq(dg, lf, iF, j)
                    if un < 0.0
                        Tin = inflow(xf, y0 + 0.5 * (dg.xi[j] + 1) * h)
                        b[g] -= un * wf * Tin
                    else
                        A[g, g] += un * wf
                    end
                end
            else
                un = nsign * uy
                iF = dir === :yplus ? n : 1
                for i in 1:n
                    wf = 0.5 * h * w[i]
                    yf = dir === :yplus ? y0 + h : y0
                    g = _gidq(dg, lf, i, iF)
                    if un < 0.0
                        Tin = inflow(x0 + 0.5 * (dg.xi[i] + 1) * h, yf)
                        b[g] -= un * wf * Tin
                    else
                        A[g, g] += un * wf
                    end
                end
            end
        end
    end
    return A, b
end

"""
    LAST_BUDGET_PASS

Diagnostics of the last [`semquad_enforce_leaf_budget!`](@ref) call:
`(merged, scans, guard, reached, capped, splits, tests, local_capped,
entry_21, unsplittable)`.  `scans` is the number of full leaf scans the pass
needed (one per merge, so `scans == merged` unless a merge was undone by the
rebalance), `guard` the loop-iteration count, `reached` whether the tree ended
at or below `max_leaves`, and `capped` whether the `guard > 100000` safety cap
fired instead of a candidate search ending.  The pass visits each sibling
quartet once per scan (`seenp` below), which is the same candidate order as the
historical full scan -- the first leaf of a quartet in set order -- at a quarter
of the lookups.

The last four fields are the local rebalance (`rebalance_local!`,
`docs/dev/AMR.md` §10): `splits` the leaves the 2:1 rebalance split
over the whole pass, `tests` the queue pops it needed, `local_capped` whether
the structural test cap fired, `entry_21` whether the tree was already 2:1 at
pass entry (see `rebalance_local!`: the equivalence with the global `balance!`
is proven under that precondition, so a `false` here means the first rebalance
seeded every violator), and `unsplittable` how many violating leaves were at
`maxlevel` (no refinement can fix those; they are counted, not re-queued).
"""
_LB
const LAST_BUDGET_PASS = Ref((merged=0, scans=0, guard=0, reached=true,
                              capped=false, splits=0, tests=0,
                              local_capped=false, entry_21=true,
                              unsplittable=0))

@inline function _record_budget_pass(merged::Int, scans::Int, guard::Int,
                                     reached::Bool, capped::Bool;
                                     splits::Int=0, tests::Int=0,
                                     local_capped::Bool=false,
                                     entry_21::Bool=true,
                                     unsplittable::Int=0)
    LAST_BUDGET_PASS[] = (merged=merged, scans=scans, guard=guard,
                          reached=reached, capped=capped, splits=splits,
                          tests=tests, local_capped=local_capped,
                          entry_21=entry_21, unsplittable=unsplittable)
    return merged, reached
end

"""
    _rebalance_after_merge!(qt, st, parent, mparent, symmetry, entry21)

The budget pass's per-merge 2:1 step: a **local** rebalance seeded at the merged
parent (`docs/dev/AMR.md` §3(a), `docs/dev/AMR.md`
§10).  `parent` and `mparent` are the merged parents (equal without `:x`), which
are exactly the leaves a merge can make non-2:1 (Lemma 1); `rebalance_local!`
adds their face neighbours and, when `entry21 == false` (the tree was not 2:1 at
pass entry), every other violator.  Returns `st`.
"""
@inline function _rebalance_after_merge!(qt::Quadtree, st::LocalBalance,
                                         parent::NTuple{3,Int},
                                         mparent::NTuple{3,Int},
                                         symmetry::Symbol, entry21::Bool)
    if symmetry === :x && mparent != parent
        rebalance_local!(qt, st; seeds=(parent, mparent), symmetry=:x,
                         all_violations=!entry21)
    else
        rebalance_local!(qt, st; seeds=(parent,), symmetry=symmetry,
                         all_violations=!entry21)
    end
    return st
end

"""
    semquad_enforce_leaf_budget!(qt, eta; max_leaves, protect, symmetry)

Enforce a hard leaf budget by merging the **lowest-error complete sibling
quartets** until the tree has at most `max_leaves` leaves (or no mergeable
quartet is left), re-balancing after every merge.  Returns `(merged, reached)`.

This is the `coarsen_frac == 0` enforcement path.  `amr_cycle!`'s old budget
guard walked the leaves in indicator order and gave up as soon as the first
leaf could not be merged, so a refine-only run could stay far above the budget
(measured: 3016 leaves against `max_leaves = 1600`).  Here the candidate is the
*parent* of a complete quartet, chosen by the summed indicator, so the merge
always makes progress; a quartet whose merge is undone by the 2:1 rebalance is
protected and never retried.

The rebalance after each merge is **local** (`rebalance_local!`,
`docs/dev/AMR.md`, `docs/dev/AMR.md` §10): the
merge is the only event that can create a 2:1 violation (Lemma 1), so the queue
starts at the new parent and its face neighbours instead of re-scanning the
10 000-leaf tree.  The result is the same leaf set the global `balance!` would
produce -- the pass re-establishes the 2:1 invariant after every merge, so the
next merge starts from a 2:1 tree (the theorem's precondition); a tree that
enters unbalanced (`entry_21 == false`, e.g. a legacy checkpoint) makes the
first rebalance seed every violator, which is the same closure again.

Every parent the rebalance splits goes into `blocked`, not just the merge's own:
a split re-forms the quartet one level down, and re-merging it would spin
(`docs/dev/AMR.md` §3(e), §5.2).

With `symmetry=:x` the merge is **atomic over the mirror pair**: the candidate is
only eligible when the mirror quartet is complete and unprotected too, and both
quartets are merged (or neither).  A pair whose mirror is protected is refused,
so the invariant survives; the documented cost is that the pass stops with
`reached == false` while the tree is up to one pair (6 leaves, less when the
refused pairs overlap) above `max_leaves` -- the old behaviour merged one side
and left 5 non-mirrored leaves per one-sided merge.

`eta` is updated in place: the merged parent gets the summed indicator of its
children, so it can be merged again at the next level.
"""
function semquad_enforce_leaf_budget!(qt::Quadtree, eta::Dict{NTuple{3,Int},Float64};
                                      max_leaves::Int=typemax(Int),
                                      protect::Set{NTuple{3,Int}}=Set{NTuple{3,Int}}(),
                                      symmetry::Symbol=:none)
    max_leaves >= 1 || throw(ArgumentError("max_leaves must be >= 1"))
    symmetry in (:none, :x) || throw(ArgumentError("symmetry must be :none or :x"))
    protect = symmetry === :x ? _mirror_protect(qt, protect) : protect
    merged = 0
    blocked = Set{NTuple{3,Int}}()
    # One entry per sibling quartet per scan: `_budget_quartet` reconstructs the
    # quartet from the parent, so all four children give the identical candidate
    # and only the first one in set order can ever win (`cand.eta < bestv` is
    # strict).  Skipping the other three is therefore the same candidate order
    # and the same values, at a quarter of the lookups.
    seenp = Set{NTuple{3,Int}}()
    guard = 0
    scans = 0
    # Local rebalance state (reused across merges) + the entry precondition.
    st = LOCAL_BALANCE
    entry21 = _tree_is_21(qt)
    splits_total = 0
    tests_total = 0
    while length(qt.leaves) > max_leaves
        guard += 1
        if guard > 100000
            return _record_budget_pass(merged, scans, guard, false, true;
                                       splits=splits_total, tests=tests_total,
                                       entry_21=entry21)
        end
        best = nothing
        bestv = Inf
        scans += 1
        empty!(seenp)
        for lf in qt.leaves
            p = (lf[1] - 1, lf[2] >> 1, lf[3] >> 1)
            p in seenp && continue
            push!(seenp, p)
            cand = _budget_quartet(qt, lf, eta, protect, blocked)
            cand === nothing && continue
            if symmetry === :x
                kid = cand.kids[1]
                ml = _mirror_leaf(qt, kid)
                if ml != kid
                    # merge both sides or neither
                    _budget_quartet(qt, ml, eta, protect, blocked) === nothing &&
                        continue
                end
            end
            if cand.eta < bestv
                bestv = cand.eta
                best = cand.kids
            end
        end
        best === nothing && return _record_budget_pass(merged, scans, guard,
                                                       false, false;
                                                       splits=splits_total,
                                                       tests=tests_total,
                                                       entry_21=entry21)
        n0 = length(qt.leaves)
        kid = best[1]
        parent = (kid[1] - 1, kid[2] >> 1, kid[3] >> 1)
        coarsen!(qt, kid) || return _record_budget_pass(merged, scans, guard,
                                                        false, false;
                                                        splits=splits_total,
                                                        tests=tests_total,
                                                        entry_21=entry21)
        eta[parent] = bestv
        merged += 1
        mparent = parent
        if symmetry === :x
            ml = _mirror_leaf(qt, kid)
            if ml != kid
                mkid = (ml[1], ml[2] - ml[2] % 2, ml[3] - ml[3] % 2)
                mparent = (ml[1] - 1, ml[2] >> 1, ml[3] >> 1)
                mcand = _budget_quartet(qt, mkid, eta, protect, blocked)
                if mcand !== nothing && coarsen!(qt, mkid)
                    eta[mparent] = mcand.eta
                    merged += 1
                end
            end
        end
        # Local 2:1 rebalance of the merge's neighbourhood (the seeds are the
        # new parent, its mirror half, and their face neighbours -- expanded by
        # `rebalance_local!`).  `entry21` is the theorem's precondition.
        _rebalance_after_merge!(qt, st, parent, mparent, symmetry, entry21)
        splits_total += st.nsplits
        tests_total += st.ntests
        # after a completed rebalance the tree is 2:1 again (the precondition of
        # the next merge's local pass); a capped one is not, so the next pass
        # re-seeds every violator.
        entry21 = !st.capped
        if length(qt.leaves) >= n0
            # The merge was undone by the rebalance: do not retry this quartet.
            # The four `(k[1], k[2] >> 1, k[3] >> 1)` entries are the historical
            # code's, verbatim: `k[1]` is the *child* level, so they are not the
            # merged parent (`parent` is pushed separately) -- but such a key is
            # a real quartet parent one level down, so they block the quartet
            # that a re-merge would form there.  They are load-bearing for the
            # bitwise trajectory (`docs/dev/AMR.md` §10.4); removing
            # them makes the pass merge more (see the 6/60 counterexample class
            # in the same section).  Every cascade parent the rebalance splits
            # has an incomplete quartet, so it needs no entry here (the linear
            # candidate scan cannot re-merge it).
            push!(blocked, parent)
            for k in best
                push!(blocked, (k[1], k[2] >> 1, k[3] >> 1))
            end
            push!(blocked, mparent)
        end
    end
    return _record_budget_pass(merged, scans, guard,
                               length(qt.leaves) <= max_leaves, false;
                               splits=splits_total, tests=tests_total,
                               local_capped=st.capped, entry_21=entry21,
                               unsplittable=st.unsplittable)
end

"""One budget candidate: the complete, unprotected, unblocked sibling quartet of
`lf`, with its summed indicator.  `nothing` when the quartet cannot be merged.

Allocation-free: the historical version built the 2x2 quartet as a
`Matrix{NTuple{3,Int}}` (a heap array per leaf per scan, 0.33 s/cycle on the
10k-leaf operating tree).  The four keys are returned in the column-major order
of that matrix (`kids[1,1], kids[2,1], kids[1,2], kids[2,2]`), so `kids[1]` is
the same child and the indicator sum accumulates in the same order."""
function _budget_quartet(qt::Quadtree, lf::NTuple{3,Int},
                         eta::Dict{NTuple{3,Int},Float64},
                         protect::Set{NTuple{3,Int}},
                         blocked::Set{NTuple{3,Int}})
    l, ix, iy = lf
    l > 0 || return nothing
    px, py = ix >> 1, iy >> 1
    parent = (l - 1, px, py)
    parent in blocked && return nothing
    bx = 2px
    by = 2py
    k1 = (l, bx, by)
    k2 = (l, bx, by + 1)
    k3 = (l, bx + 1, by)
    k4 = (l, bx + 1, by + 1)
    (k1 in qt.leaves && k2 in qt.leaves && k3 in qt.leaves && k4 in qt.leaves) ||
        return nothing
    (k1 in protect || k2 in protect || k3 in protect || k4 in protect) &&
        return nothing
    s = max(get(eta, k1, 0.0), 0.0)
    s += max(get(eta, k2, 0.0), 0.0)
    s += max(get(eta, k3, 0.0), 0.0)
    s += max(get(eta, k4, 0.0), 0.0)
    return (kids=(k1, k2, k3, k4), eta=s)
end
