# Quadtree SEM with hanging-node constraints (stage A-AMR, step 1: Poisson).
#
# The continuous Q_p space lives on the quadtree leaves; nodes shared by
# conforming faces get one global id (geometric hash), and a node that lies on
# the face of a coarser leaf is constrained by the coarse face's GLL
# interpolation.  The constraints are applied as a sparse matrix C (identity on
# free dofs, interpolation weights on constrained rows) and the system is solved
# as C' A C u = C' b, which is exact for linear constraints and keeps symmetry.
# Boundary conditions are Nitsche, as in the DG prototype (same formulas).

struct SemQuad
    p::Int
    Lx::Float64                 # box width; the height is 1
    nx0::Int                    # base cells in x (see Quadtree)
    ny0::Int                    # base cells in y
    leaves::Vector{NTuple{3,Int}}
    coords::Vector{NTuple{2,Float64}}
    elem::Dict{NTuple{3,Int},Vector{Int}}
    N::Int
    xi::Vector{Float64}
    w::Vector{Float64}
    D::Matrix{Float64}
end

# ---------------------------------------------------------------------------
# S4 stage 1: helpers shared by the threaded leaf assembly loops (Stokes in
# `stokes/assembly_var.jl` and energy here).  The threaded paths split the leaves into
# fixed contiguous blocks and concatenate the per-block COO buffers in block
# order, which keeps them deterministic run to run.
# ---------------------------------------------------------------------------

# Concatenate per-block buffers in block order (single allocation per array).
function _semquad_concat_blocks(parts::Vector{<:AbstractVector})
    T = eltype(parts[1])
    n = 0
    for p in parts
        n += length(p)
    end
    out = Vector{T}(undef, n)
    off = 1
    for p in parts
        m = length(p)
        copyto!(out, off, p, 1, m)
        off += m
    end
    return out
end

const _SEMQUAD_ASM_NO_THREADS_WARNED = Ref(false)
const _SEMQUAD_ASM_CLAMP_WARNED = Ref(false)

# Resolve the requested `assemble_threads` against the actual number of Julia
# threads.  1 stays 1 (the serial, bit-identical path); a request larger than
# the pool falls back to the pool size and a single-threaded process warns once
# and stays serial.
function _semquad_asm_thread_count(assemble_threads::Int)
    assemble_threads >= 1 ||
        throw(ArgumentError("assemble_threads must be >= 1"))
    assemble_threads == 1 && return 1
    nproc = Threads.nthreads()
    if nproc == 1
        if !_SEMQUAD_ASM_NO_THREADS_WARNED[]
            _SEMQUAD_ASM_NO_THREADS_WARNED[] = true
            @warn "assemble_threads=$(assemble_threads) requested, but Julia " *
                  "was started with 1 thread; falling back to serial assembly. " *
                  "Start Julia with --threads=N (or JULIA_NUM_THREADS=N) for " *
                  "leaf-level parallel assembly." maxlog=1
        end
        return 1
    end
    nt = min(assemble_threads, nproc)
    if nt < assemble_threads && !_SEMQUAD_ASM_CLAMP_WARNED[]
        _SEMQUAD_ASM_CLAMP_WARNED[] = true
        @warn "assemble_threads=$(assemble_threads) exceeds " *
              "Threads.nthreads()=$(nproc); using $nt blocks" maxlog=1
    end
    return nt
end

# Physical leaf extents; `nx0 == ny0 == 1, Lx == 1` is the historical unit square.
@inline leaf_hx(sq::SemQuad, l::Int) = sq.Lx / (sq.nx0 * (1 << l))
@inline leaf_hy(sq::SemQuad, l::Int) = 1.0 / (sq.ny0 * (1 << l))
@inline leaf_h(sq::SemQuad, l::Int) = leaf_hy(sq, l)

function make_semquad(qt::Quadtree, p::Int)
    p >= 1 || throw(ArgumentError("p must be >= 1"))
    rule = gll_rule_cached(p)
    n = p + 1
    D = zeros(Float64, n, n)
    for i in 1:n
        _, dL = lagrange_eval(rule, rule.nodes[i])
        for j in 1:n
            D[i, j] = dL[j]
        end
    end
    tol = 1.0e-12
    # One sorted leaf list for both the element build and the `leaves` field
    # (it used to sort the tree twice per mesh rebuild: `sort(collect(...))`
    # is 2.3 MB and ~0.3 ms per 10 000-leaf call, so the second copy was pure
    # waste; the sorted order is the same, hence bitwise identical).
    leaves = sort(collect(qt.leaves))
    coords = NTuple{2,Float64}[]
    idmap = Dict{NTuple{2,Int},Int}()
    elem = Dict{NTuple{3,Int},Vector{Int}}()
    for lf in leaves
        hx = leaf_hx(qt, lf[1]); hy = leaf_hy(qt, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = Vector{Int}(undef, n * n)
        for j in 1:n, i in 1:n
            x = x0 + 0.5 * (rule.nodes[i] + 1.0) * hx
            y = y0 + 0.5 * (rule.nodes[j] + 1.0) * hy
            key = (round(Int, x / tol), round(Int, y / tol))
            gid = get(idmap, key, 0)
            if gid == 0
                push!(coords, (x, y))
                gid = length(coords)
                idmap[key] = gid
            end
            ids[(j - 1) * n + i] = gid
        end
        elem[lf] = ids
    end
    SemQuad(p, qt.Lx, qt.nx0, qt.ny0, leaves, coords, elem, length(coords),
            copy(rule.nodes), copy(rule.weights), D)
end

"""Hanging-node constraints: fine face node = interpolation of the coarse face."""
function semquad_constraints(sq::SemQuad, qt::Quadtree)
    p = sq.p
    n = p + 1
    rule = gll_rule(p)
    cons = Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
    for pf in face_pairs(qt)
        L, R, dir = pf.leafL, pf.leafR, pf.dir
        hL = leaf_hy(sq, L[1]); hR = leaf_hy(sq, R[1])
        hxL = leaf_hx(sq, L[1]); hxR = leaf_hx(sq, R[1])
        hL > hR + 1.0e-15 || continue           # only coarse-fine, coarse = L
        if dir === :xplus || dir === :xminus
            iFL = dir === :xplus ? n : 1
            iFR = dir === :xplus ? 1 : n
            yL = L[3] * hL; yR = R[3] * hR
            for q in 1:n
                yq = yR + 0.5 * (sq.xi[q] + 1.0) * hR
                xiL = 2.0 * (yq - yL) / hL - 1.0
                minimum(abs.(sq.xi .- xiL)) < 1.0e-10 && continue
                vals, _ = lagrange_eval(rule, xiL)
                c = sq.elem[R][(q - 1) * n + iFR]
                ms = [sq.elem[L][(a - 1) * n + iFL] for a in 1:n]
                cons[c] = (ms, vals)
            end
        else
            jFL = dir === :yplus ? n : 1
            jFR = dir === :yplus ? 1 : n
            xL = L[2] * hxL; xR = R[2] * hxR
            for q in 1:n
                xq = xR + 0.5 * (sq.xi[q] + 1.0) * hxR
                xiL = 2.0 * (xq - xL) / hxL - 1.0
                minimum(abs.(sq.xi .- xiL)) < 1.0e-10 && continue
                vals, _ = lagrange_eval(rule, xiL)
                c = sq.elem[R][(jFR - 1) * n + q]
                ms = [sq.elem[L][(jFL - 1) * n + a] for a in 1:n]
                cons[c] = (ms, vals)
            end
        end
    end
    return resolve_constraints(cons)
end

"""Substitute constrained masters by their own constraints (multi-level faces)."""
function resolve_constraints(cons::Dict{Int,Tuple{Vector{Int},Vector{Float64}}})
    for _ in 1:8
        changed = false
        for (c, (ms, ws)) in collect(cons)
            newms = Int[]; newws = Float64[]
            for (m, w) in zip(ms, ws)
                if haskey(cons, m)
                    m2, w2 = cons[m]
                    for (mm, ww) in zip(m2, w2)
                        push!(newms, mm); push!(newws, w * ww)
                    end
                else
                    push!(newms, m); push!(newws, w)
                end
            end
            if newms != ms
                cons[c] = (newms, newws)
                changed = true
            end
        end
        changed || break
    end
    return cons
end

"""Assemble the SIPG/Nitsche Poisson operator on the quadtree SEM mesh."""
function assemble_semquad_poisson(sq::SemQuad, qt::Quadtree, f::Function, uD::Function;
                                  kappa::Float64=1.0, sigma_C::Float64=50.0)
    p = sq.p
    n = p + 1
    N = sq.N
    nv = n * n
    I = Int[]; J = Int[]; V = Float64[]
    b = zeros(Float64, N)
    W = Diagonal(sq.w)
    DW = sq.D' * W * sq.D
    # Reference-space stiffness terms; the physical metric enters as the aspect
    # ratios hy/hx and hx/hy (both 1 for the unit square).
    K1 = kron(W, DW)        # d/dxi  d/dxi
    K2 = kron(DW, W)        # d/deta d/deta
    for lf in sq.leaves
        ids = sq.elem[lf]
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        Ke = kappa .* ((hy / hx) .* K1 .+ (hx / hy) .* K2)
        for kk in 1:nv, ll in 1:nv
            v = Ke[kk, ll]
            v != 0.0 && (push!(I, ids[kk]); push!(J, ids[ll]); push!(V, v))
        end
        x0 = lf[2] * hx; y0 = lf[3] * hy
        for j in 1:n, i in 1:n
            x = x0 + 0.5 * (sq.xi[i] + 1.0) * hx
            y = y0 + 0.5 * (sq.xi[j] + 1.0) * hy
            b[ids[(j - 1) * n + i]] += 0.25 * hx * hy * sq.w[i] * sq.w[j] * f(x, y)
        end
    end
    # Nitsche boundaries (same formulas as the DG prototype).  The penalty uses
    # the face normal's inverse length: 1/hx on x-faces, 1/hy on y-faces.
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = sq.elem[lf]
        for dir in (:xminus, :xplus, :yminus, :yplus)
            has_no_face_neighbor(qt, lf, dir) || continue
            if dir === :xminus || dir === :xplus
                iF = dir === :xminus ? 1 : n
                sig = sigma_C * (p + 1)^2 / hx
                c = (2.0 / hx) .* sq.D[iF, :]
                s = dir === :xplus ? -1.0 : 1.0
                for j in 1:n
                    wf = 0.5 * hy * sq.w[j]
                    xf = dir === :xplus ? x0 + hx : x0
                    Td = uD(xf, y0 + 0.5 * (sq.xi[j] + 1.0) * hy)
                    gF = ids[(j - 1) * n + iF]
                    for i in 1:n
                        gi = ids[(j - 1) * n + i]
                        push!(I, gF); push!(J, gi); push!(V, s * wf * c[i])
                        push!(I, gi); push!(J, gF); push!(V, s * wf * c[i])
                        b[gi] += s * wf * c[i] * Td
                    end
                    push!(I, gF); push!(J, gF); push!(V, sig * wf)
                    b[gF] += sig * wf * Td
                end
            else
                jF = dir === :yminus ? 1 : n
                sig = sigma_C * (p + 1)^2 / hy
                c = (2.0 / hy) .* sq.D[jF, :]
                s = dir === :yplus ? -1.0 : 1.0
                for i in 1:n
                    wf = 0.5 * hx * sq.w[i]
                    yf = dir === :yplus ? y0 + hy : y0
                    Td = uD(x0 + 0.5 * (sq.xi[i] + 1.0) * hx, yf)
                    gF = ids[(jF - 1) * n + i]
                    for j in 1:n
                        gj = ids[(j - 1) * n + i]
                        push!(I, gF); push!(J, gj); push!(V, s * wf * c[j])
                        push!(I, gj); push!(J, gF); push!(V, s * wf * c[j])
                        b[gj] += s * wf * c[j] * Td
                    end
                    push!(I, gF); push!(J, gF); push!(V, sig * wf)
                    b[gF] += sig * wf * Td
                end
            end
        end
    end
    return sparse(I, J, V, N, N), b
end

"""Solve the constrained system (C' A C u = C' b) and return `(u, cons)`."""
function solve_semquad_poisson(sq::SemQuad, qt::Quadtree, f::Function, uD::Function;
                               kappa::Float64=1.0, sigma_C::Float64=50.0)
    A, b = assemble_semquad_poisson(sq, qt, f, uD; kappa=kappa, sigma_C=sigma_C)
    cons = semquad_constraints(sq, qt)
    free = [i for i in 1:sq.N if !haskey(cons, i)]
    fmap = Dict(v => k for (k, v) in enumerate(free))
    I = Int[]; J = Int[]; V = Float64[]
    for (k, i) in enumerate(free)
        push!(I, i); push!(J, k); push!(V, 1.0)
    end
    for (c, (ms, ws)) in cons, (m, w) in zip(ms, ws)
        push!(I, c); push!(J, fmap[m]); push!(V, w)
    end
    C = sparse(I, J, V, sq.N, length(free))
    ur = (C' * A * C) \ (C' * b)
    return C * ur, cons
end

"""Evaluate a quadtree SEM field at (x,y)."""
function semquad_eval(sq::SemQuad, u::Vector{Float64}, x::Float64, y::Float64)
    n = sq.p + 1
    rule = gll_rule(sq.p)
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        if x >= x0 - 1.0e-14 && x <= x0 + hx + 1.0e-14 &&
           y >= y0 - 1.0e-14 && y <= y0 + hy + 1.0e-14
            xi = 2.0 * (x - x0) / hx - 1.0
            eta = 2.0 * (y - y0) / hy - 1.0
            Lb, _ = lagrange_eval(rule, xi)
            Lc, _ = lagrange_eval(rule, eta)
            ids = sq.elem[lf]
            v = 0.0
            for j in 1:n, i in 1:n
                v += u[ids[(j - 1) * n + i]] * Lb[i] * Lc[j]
            end
            return v
        end
    end
    throw(ArgumentError("point ($x,$y) not in the tree"))
end

"""
    _semquad_leaf_at(sq, x, y, leaves)

Locate the leaf containing `(x,y)` by descending the quadtree from the root.
`leaves` is the set of `sq.leaves`, passed in to avoid rebuilding it per point.
"""
function _semquad_leaf_at(sq::SemQuad, x::Float64, y::Float64,
                          leaves::Set{NTuple{3,Int}})
    # Start from the *actual* level-0 base cell, not (0,0): a multi-root tree
    # (cells_x/cells_y > 1) has several base leaves, and checking only (0,0,0)
    # would return the first base cell for every point.  Refined trees hid this
    # because they have no level-0 leaves; once a coarse region survives
    # coarsening (e.g. an MG coarse level, or AMR with coarsening) every point
    # outside base cell 0 was evaluated with the wrong element.
    l = 0
    ix = clamp(floor(Int, x / leaf_hx(sq, 0)), 0, sq.nx0 - 1)
    iy = clamp(floor(Int, y / leaf_hy(sq, 0)), 0, sq.ny0 - 1)
    while true
        lf = (l, ix, iy)
        lf in leaves && return lf
        l += 1
        l <= 62 || throw(ArgumentError("point ($x,$y) not in the tree"))
        ix = clamp(floor(Int, x / leaf_hx(sq, l)), 0, sq.nx0 * (1 << l) - 1)
        iy = clamp(floor(Int, y / leaf_hy(sq, l)), 0, sq.ny0 * (1 << l) - 1)
    end
end

"""
    SemQuadEvalWork(n)

Two `n`-vector buffers for the out-parameter point evaluation
([`_semquad_eval_at!`](@ref), [`_semquad_eval_at_leaf!`](@ref)).  One per loop:
the transfer reuses it over every new node and the energy assembly over every
quadrature point, instead of allocating two `lagrange_values` results per call
(`--track-allocation=user` charged 109.9 MB to `lagrange_values` per 4 AMR
cycles on the 10 000-leaf state, essentially all of it the nodal transfer).
"""
struct SemQuadEvalWork
    Lb::Vector{Float64}
    Lc::Vector{Float64}
end

SemQuadEvalWork(n::Int) = SemQuadEvalWork(zeros(Float64, n), zeros(Float64, n))

"""
    _semquad_eval_at_leaf!(w, sq, u, x, y, lf)

Evaluate the field `u` at `(x,y)` knowing that the point lies in leaf `lf`
(the energy assembly's quadrature points and the transfer's per-leaf loop do).
Identical arithmetic to [`_semquad_eval_at`](@ref); only the point search is
skipped.  The caller is responsible for the leaf being the one the search
would return (true for every interior point).
"""
function _semquad_eval_at_leaf!(w::SemQuadEvalWork, sq::SemQuad,
                                u::Vector{Float64}, x::Float64, y::Float64,
                                lf::NTuple{3,Int})
    n = sq.p + 1
    hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
    x0 = lf[2] * hx; y0 = lf[3] * hy
    xi = 2.0 * (x - x0) / hx - 1.0
    eta = 2.0 * (y - y0) / hy - 1.0
    rule = gll_rule_cached(sq.p)
    lagrange_values!(w.Lb, rule, xi)
    lagrange_values!(w.Lc, rule, eta)
    ids = sq.elem[lf]
    v = 0.0
    for j in 1:n, i in 1:n
        v += u[ids[(j - 1) * n + i]] * w.Lb[i] * w.Lc[j]
    end
    return v
end

"""Evaluate a quadtree SEM field at (x,y) with an O(maxlevel) point search."""
function _semquad_eval_at(sq::SemQuad, u::Vector{Float64},
                          x::Float64, y::Float64,
                          leaves::Set{NTuple{3,Int}})
    return _semquad_eval_at!(SemQuadEvalWork(sq.p + 1), sq, u, x, y, leaves)
end

"""Out-parameter form of [`_semquad_eval_at`](@ref); same value bit for bit."""
function _semquad_eval_at!(w::SemQuadEvalWork, sq::SemQuad,
                           u::Vector{Float64}, x::Float64, y::Float64,
                           leaves::Set{NTuple{3,Int}})
    lf = _semquad_leaf_at(sq, x, y, leaves)
    return _semquad_eval_at_leaf!(w, sq, u, x, y, lf)
end

"""
    _semquad_grad_at(sq, u, x, y, leaves)

Gradient `(du/dx, du/dy)` of a quadtree SEM field at `(x,y)`, using the element
own GLL derivative matrices and the physical scaling `2/h`.
"""
function _semquad_grad_at(sq::SemQuad, u::Vector{Float64},
                          x::Float64, y::Float64,
                          leaves::Set{NTuple{3,Int}})
    lf = _semquad_leaf_at(sq, x, y, leaves)
    n = sq.p + 1
    hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
    x0 = lf[2] * hx; y0 = lf[3] * hy
    xi = 2.0 * (x - x0) / hx - 1.0
    eta = 2.0 * (y - y0) / hy - 1.0
    rule = gll_rule_cached(sq.p)
    Lb, dLb = lagrange_eval(rule, xi)
    Lc, dLc = lagrange_eval(rule, eta)
    ids = sq.elem[lf]
    dx = 0.0; dy = 0.0
    for j in 1:n, i in 1:n
        v = u[ids[(j - 1) * n + i]]
        dx += v * dLb[i] * Lc[j]
        dy += v * Lb[i] * dLc[j]
    end
    return dx * 2.0 / hx, dy * 2.0 / hy
end

"""
    semquad_mass(sq)

Global SEM mass matrix on `sq`, assembled with a `p+2` quadrature rule.
"""
function semquad_mass(sq::SemQuad)
    p = sq.p
    n = p + 1
    nv = n * n
    fd = make_fedata(p, p + 2)
    I = Int[]; J = Int[]; V = Float64[]
    # Every quadrature point contributes one nv x nv block (the duplicates are
    # summed by `sparse`), i.e. nq^2 * nv^2 entries per leaf.  Without the hint
    # the three arrays grew by repeated doubling: 1.37 GB per call on the
    # 10 000-leaf state, versus ~311 MB of exact capacity.
    sizehint!(I, fd.nq^2 * nv * nv * length(sq.leaves))
    sizehint!(J, fd.nq^2 * nv * nv * length(sq.leaves))
    sizehint!(V, fd.nq^2 * nv * nv * length(sq.leaves))
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                phi_k = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                for l in 1:n, m in 1:n
                    q = (l - 1) * n + m
                    phi_q = fd.phi_v[m, qx] * fd.phi_v[l, qy]
                    v = w * phi_k * phi_q
                    v != 0.0 && (push!(I, ids[k]); push!(J, ids[q]); push!(V, v))
                end
            end
        end
    end
    return sparse(I, J, V, sq.N, sq.N)
end

# Moved here from the removed DWR module: the DP steady-Newton PTC branch
# (`semquad_dp_steady.jl`) builds its reduced mass matrix from this.
function assemble_semquad_velocity_mass(sq::SemQuad)
    M = semquad_mass(sq)
    return [M spzeros(sq.N, sq.N); spzeros(sq.N, sq.N) M]
end

"""
    transfer_semquad(old, uold, new; mode=:nodal, zero_walls=false, qt_new=nothing)

Move a continuous quadtree SEM field from mesh `old` onto mesh `new`.

`mode=:nodal` evaluates the old field at the new GLL nodes (exact for
refinement, natural interpolation for coarsening).  `mode=:l2` solves the
global L2 projection `M_new u_new = b`, with `b_i = int phi_i u_old`; it is the
coarsening-consistent transfer and avoids the interface steps produced by
pointwise restriction.  With `zero_walls=true` the top/bottom nodes are held at
zero.  `qt_new` is optional; when supplied, hanging-node constraints are
applied after the projection.  Hanging constraints are otherwise the caller's
responsibility (`C'AC` in the assembly).
"""
function transfer_semquad(old::SemQuad, uold::Vector{Float64}, new::SemQuad;
                          mode::Symbol=:nodal, zero_walls::Bool=false,
                          qt_new::Union{Nothing,Quadtree}=nothing)
    mode in (:nodal, :l2) ||
        throw(ArgumentError("transfer mode must be :nodal or :l2"))
    length(uold) == old.N ||
        throw(ArgumentError("uold has $(length(uold)) entries, old mesh has $(old.N)"))
    old.Lx == new.Lx ||
        throw(ArgumentError("transfer_semquad needs equal box widths ($(old.Lx) vs $(new.Lx))"))
    oldleaves = Set(old.leaves)
    if mode === :nodal
        unew = zeros(Float64, new.N)
        n = new.p + 1
        w = SemQuadEvalWork(n)
        # Shared nodes (faces, corners, and the hanging nodes of a fine face)
        # are written by several new leaves.  Every duplicate evaluation sees
        # the same (x,y) and the same old leaf -- the coordinates are exact
        # dyadic sums and `_semquad_leaf_at` is deterministic -- so the node's
        # value is the same bit pattern; evaluate it once, at its first
        # occurrence.  This halves the evaluation count (90 000 -> 42 123 on
        # the 10 000-leaf state) and removes the two `lagrange_values`
        # allocations per evaluation.
        seen = falses(new.N)
        for lf in new.leaves
            hx = leaf_hx(new, lf[1]); hy = leaf_hy(new, lf[1])
            x0 = lf[2] * hx; y0 = lf[3] * hy
            ids = new.elem[lf]
            for j in 1:n, i in 1:n
                gid = ids[(j - 1) * n + i]
                seen[gid] && continue
                seen[gid] = true
                x = x0 + 0.5 * (new.xi[i] + 1.0) * hx
                y = y0 + 0.5 * (new.xi[j] + 1.0) * hy
                unew[gid] = _semquad_eval_at!(w, old, uold, x, y, oldleaves)
            end
        end
        return unew
    end

    p = new.p
    n = p + 1
    fd = make_fedata(p, p + 2)
    M = semquad_mass(new)
    b = zeros(Float64, new.N)
    for lf in new.leaves
        hx = leaf_hx(new, lf[1]); hy = leaf_hy(new, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = new.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            uq = _semquad_eval_at(old, uold, x, y, oldleaves)
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                phi = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                b[ids[k]] += w * phi * uq
            end
        end
    end
    if zero_walls
        free = [g for g in 1:new.N if 0.0 < new.coords[g][2] < 1.0]
        unew = zeros(Float64, new.N)
        isempty(free) || (unew[free] = M[free, free] \ b[free])
    else
        unew = M \ b
    end
    if qt_new !== nothing
        cons = semquad_constraints(new, qt_new)
        for (c, (ms, ws)) in cons
            unew[c] = sum(ws .* unew[ms])
        end
    end
    return unew
end

# ---------------------------------------------------------------------------
# Gradient-jump error indicator for the continuous quadtree SEM field.
# The value is continuous, but the normal derivative is not; the jump between
# the coarse and fine one-sided derivatives is the same indicator the DG
# prototype (removed with the classic policy) used, with the coarse side
# interpolated at the fine-side
# quadrature points.  Boundary faces add |du/dn|^2 so the thermal boundary
# layers are marked as well.
# ---------------------------------------------------------------------------

"""
    SemQuadJumpWork(n)

Per-call scratch of the gradient-jump indicator: one buffer per 1-D length `n`.
The face loop reuses it for every coarse/fine sub-face, so the indicator
allocates once per call instead of ~1.6 kB per face pair (the buffers were
`BL`, `wq`, `pos`, `xiL`, `cL`, `cR`, `dL`, `dR` and the two `lagrange_eval`
results; `--track-allocation=user` charged 67.4 MB to `_semquad_subface_derivs`
per 4 AMR cycles on the 10 000-leaf state).
"""
struct SemQuadJumpWork
    Lb::Vector{Float64}     # Lagrange values at the current fine point
    dLb::Vector{Float64}    # their derivatives (unused, `lagrange_eval!` writes them)
    dL::Vector{Float64}     # coarse-side one-sided derivative
    dR::Vector{Float64}     # fine-side one-sided derivative
    wq::Vector{Float64}     # fine-side face weights
    pos::Vector{Float64}    # physical face abscissae
    xiL::Vector{Float64}    # coarse-side reference abscissae
    cL::Vector{Float64}     # coarse-side normal derivative coefficients
    cR::Vector{Float64}     # fine-side normal derivative coefficients
    cB::Vector{Float64}     # boundary-face derivative coefficients
end

function SemQuadJumpWork(n::Int)
    SemQuadJumpWork(zeros(Float64, n), zeros(Float64, n), zeros(Float64, n),
                    zeros(Float64, n), zeros(Float64, n), zeros(Float64, n),
                    zeros(Float64, n), zeros(Float64, n), zeros(Float64, n),
                    zeros(Float64, n))
end

"""
    _semquad_subface_derivs!(w, sq, L, R, dir, u)

Workspace form of [`_semquad_subface_derivs`](@ref): fills `w.wq`, `w.dL` and
`w.dR` and returns them.  Same arithmetic and loop order as the allocating
form (the `BL` matrix is gone: row `q` is consumed as soon as it is evaluated,
exactly as the historical `BL[q, :] .= Lb` + `dL[q] += BL[q,a]*da` sequence).
"""
function _semquad_subface_derivs!(w::SemQuadJumpWork, sq::SemQuad,
                                  L::NTuple{3,Int}, R::NTuple{3,Int},
                                  dir::Symbol, u::Vector{Float64})
    p = sq.p
    n = p + 1
    hL = leaf_hy(sq, L[1]); hR = leaf_hy(sq, R[1])
    hxL = leaf_hx(sq, L[1]); hxR = leaf_hx(sq, R[1])
    xL = L[2] * hxL; yL = L[3] * hL
    xR = R[2] * hxR; yR = R[3] * hR
    rule = gll_rule_cached(p)
    idsL = sq.elem[L]
    idsR = sq.elem[R]
    vertical = dir === :xplus || dir === :xminus
    if vertical
        # x-face: normal derivative d/dx, parametrised by y
        hw = 0.5 * hR
        iFL = dir === :xplus ? n : 1
        iFR = dir === :xplus ? 1 : n
        for q in 1:n
            w.wq[q] = hw * sq.w[q]
            w.pos[q] = yR + 0.5 * (sq.xi[q] + 1.0) * hR
            w.xiL[q] = 2.0 * (w.pos[q] - yL) / hL - 1.0
        end
        hcL = 2.0 / hxL
        hcR = 2.0 / hxR
        for i in 1:n
            w.cL[i] = hcL * sq.D[iFL, i]
            w.cR[i] = hcR * sq.D[iFR, i]
        end
        for q in 1:n
            lagrange_eval!(w.Lb, w.dLb, rule, w.xiL[q])
            dq = 0.0
            for a in 1:n
                da = 0.0
                for i in 1:n
                    da += w.cL[i] * u[idsL[(a - 1) * n + i]]
                end
                dq += w.Lb[a] * da
            end
            w.dL[q] = dq
            dq = 0.0
            for i in 1:n
                dq += w.cR[i] * u[idsR[(q - 1) * n + i]]
            end
            w.dR[q] = dq
        end
    else
        # y-face: normal derivative d/dy, parametrised by x
        hw = 0.5 * hxR
        iFL = dir === :yplus ? n : 1
        iFR = dir === :yplus ? 1 : n
        for q in 1:n
            w.wq[q] = hw * sq.w[q]
            w.pos[q] = xR + 0.5 * (sq.xi[q] + 1.0) * hxR
            w.xiL[q] = 2.0 * (w.pos[q] - xL) / hxL - 1.0
        end
        hcL = 2.0 / hL
        hcR = 2.0 / hR
        for i in 1:n
            w.cL[i] = hcL * sq.D[iFL, i]
            w.cR[i] = hcR * sq.D[iFR, i]
        end
        for q in 1:n
            lagrange_eval!(w.Lb, w.dLb, rule, w.xiL[q])
            dq = 0.0
            for a in 1:n
                da = 0.0
                for i in 1:n
                    da += w.cL[i] * u[idsL[(i - 1) * n + a]]
                end
                dq += w.Lb[a] * da
            end
            w.dL[q] = dq
            dq = 0.0
            for i in 1:n
                dq += w.cR[i] * u[idsR[(q - 1) * n + i]]
            end
            w.dR[q] = dq
        end
    end
    return w.wq, w.dL, w.dR
end

"""One-sided normal-derivative traces of a coarse/fine sub-face at the fine-side
GLL points (SEM version of `dgquad._subface_derivs`)."""
function _semquad_subface_derivs(sq::SemQuad, L::NTuple{3,Int}, R::NTuple{3,Int},
                                 dir::Symbol, u::Vector{Float64})
    return _semquad_subface_derivs!(SemQuadJumpWork(sq.p + 1), sq, L, R, dir, u)
end

"""
    semquad_gradient_jump(sq, qt, u)

Per-leaf indicator `(h/2) * int_face [du.n]^2 ds` summed over interior faces,
plus `(h/2) * int_boundary (du.n)^2 ds` on the domain boundary.  The boundary
term is needed for thermal boundary layers; without it the estimator only sees
interior jumps.
"""
function semquad_gradient_jump(sq::SemQuad, qt::Quadtree, u::Vector{Float64})
    n = sq.p + 1
    eta = Dict{NTuple{3,Int},Float64}()
    # `Dict` growth is O(#leaves) rehashes; every consumer either looks a leaf
    # up or sorts the entries, so the table capacity has no arithmetic effect.
    sizehint!(eta, length(sq.leaves))
    for lf in sq.leaves
        eta[lf] = 0.0
    end
    w = SemQuadJumpWork(n)
    for pf in face_pairs(qt)
        L, R, dir = pf.leafL, pf.leafR, pf.dir
        wq, dL, dR = _semquad_subface_derivs!(w, sq, L, R, dir, u)
        J = 0.0
        for q in eachindex(wq)
            J += wq[q] * (dL[q] - dR[q])^2
        end
        # the (h/2) scale uses the face-normal element size
        normal_x = dir === :xplus || dir === :xminus
        hL = normal_x ? leaf_hx(sq, L[1]) : leaf_hy(sq, L[1])
        hR = normal_x ? leaf_hx(sq, R[1]) : leaf_hy(sq, R[1])
        eta[L] += 0.5 * hL * J
        eta[R] += 0.5 * hR * J
    end
    # Boundary faces: the one-sided normal derivative is the only trace.
    for lf in sq.leaves
        h = leaf_hy(sq, lf[1])
        hx = leaf_hx(sq, lf[1])
        ids = sq.elem[lf]
        for dir in (:xminus, :xplus, :yminus, :yplus)
            has_no_face_neighbor(qt, lf, dir) || continue
            # Left/right walls are insulating (natural Neumann, dtheta/dx=0);
            # a one-sided x-derivative there is not a physical jump and would
            # keep refining the whole side column forever.  Only top/bottom
            # Dirichlet walls contribute a boundary-jump term.
            (dir === :xminus || dir === :xplus) && continue
            if dir === :xminus || dir === :xplus
                iF = dir === :xminus ? 1 : n
                c = (2.0 / hx) .* sq.D[iF, :]
                for j in 1:n
                    d = 0.0
                    for i in 1:n
                        d += c[i] * u[ids[(j - 1) * n + i]]
                    end
                    wf = 0.5 * h * sq.w[j]
                    eta[lf] += 0.5 * h * wf * d^2
                end
            else
                jF = dir === :yminus ? 1 : n
                hc = 2.0 / h
                for i in 1:n
                    w.cB[i] = hc * sq.D[jF, i]
                end
                for i in 1:n
                    d = 0.0
                    for j in 1:n
                        d += w.cB[j] * u[ids[(j - 1) * n + i]]
                    end
                    wf = 0.5 * hx * sq.w[i]
                    eta[lf] += 0.5 * h * wf * d^2
                end
            end
        end
    end
    return eta
end

# ---------------------------------------------------------------------------
# Energy equation on the quadtree SEM mesh.
# The operator is for theta = T - Tbar, with Tbar = 1-z+(H/2)z(1-z).  Since
# Tbar is the steady conductive profile (lap(Tbar) + H = 0), the theta equation
# is d(theta)/dt + u.grad(theta) = lap(theta) - u_z dTbar/dz; the +H source is
# already absorbed by Tbar and must NOT be added again.  The same convention is
# used by the uniform-mesh assembler (assembly.jl).
# ---------------------------------------------------------------------------

