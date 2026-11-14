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

# ---------------------------------------------------------------------------
# Entropy viscosity on the quadtree (port of assembly.jl / ASPECT)
# ---------------------------------------------------------------------------

struct SEMEVGlobals
    minT::Float64
    maxT::Float64
    avgT::Float64
    entropy_variation::Float64
    field_variation::Float64
    u_infty::Float64
end

function semquad_ev_globals(sq::SemQuad, qt::Quadtree, pr::Params,
                            theta::Vector{Float64},
                            theta_prev::Union{Nothing,Vector{Float64}},
                            ux::Vector{Float64}, uz::Vector{Float64},
                            dt_ratio::Float64)
    minT = Inf; maxT = -Inf; uinf = 0.0
    for g in 1:sq.N
        y = sq.coords[g][2]
        tb = Tbar(y, pr.H)
        tn = tb + theta[g]
        t = if dt_ratio > 0.0 && theta_prev !== nothing
            (1.0 + dt_ratio) * tn - dt_ratio * (tb + theta_prev[g])
        else
            tn
        end
        minT = min(minT, t); maxT = max(maxT, t)
        uinf = max(uinf, hypot(ux[g], uz[g]))
    end
    avgT = 0.5 * (minT + maxT)
    fd = make_fedata(sq.p, sq.p)
    eint = 0.0; area = 0.0; emin = Inf; emax = -Inf
    for lf in sq.leaves
        h = leaf_hy(sq, lf[1])
        x0 = lf[2] * h; y0 = lf[3] * h
        ids = sq.elem[lf]
        n = sq.p + 1
        for qy in 1:fd.nq, qx in 1:fd.nq
            x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * h
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * h
            thn = 0.0; thp = 0.0
            for j in 1:n, i in 1:n
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                k = (j - 1) * n + i
                thn += theta[ids[k]] * b
                if theta_prev !== nothing
                    thp += theta_prev[ids[k]] * b
                end
            end
            Tmid = Tbar(y, pr.H) + 0.5 * (thn + thp)
            e = (Tmid - avgT)^2
            w = 0.25 * h * h * fd.gl_w[qx] * fd.gl_w[qy]
            eint += w * e; area += w
            emin = min(emin, e); emax = max(emax, e)
        end
    end
    avge = area > 0.0 ? eint / area : 0.0
    return SEMEVGlobals(minT, maxT, avgT,
                        max(emax - avge, avge - emin),
                        maxT - minT, uinf)
end

function semquad_ev_cell_stats(sq::SemQuad, pr::Params,
                               theta::Vector{Float64},
                               theta_prev::Union{Nothing,Vector{Float64}},
                               ux::Vector{Float64}, uz::Vector{Float64},
                               dt::Float64, lf::NTuple{3,Int},
                               evg::SEMEVGlobals; alpha::Int=2,
                               fd::Union{Nothing,FEData}=nothing,
                               d2tab::Union{Nothing,Vector{Vector{Float64}}}=nothing)
    h = leaf_hy(sq, lf[1])
    x0 = lf[2] * h; y0 = lf[3] * h
    ids = sq.elem[lf]
    n = sq.p + 1
    # `fd` and `d2tab` depend only on `(p, nq)`: the energy assembler builds
    # them once per assembly and hands them in (one `eigen` + 6 matrices per
    # leaf otherwise -- 57.8 MB and 44 ms per assembly on the 10 000-leaf
    # state).  The cold callers (`semquad_ev_field`, the tests) still get the
    # historical allocation: identical values, built the same way.
    fd_used = fd === nothing ? make_fedata(sq.p, sq.p) : fd
    Rmax = 0.0; umax = 0.0
    # Loop-invariant hoist (S3 stage 4): `gll_rule` and `lagrange_d2` were
    # called *inside* the qx/qy quadrature loop (four times per point: 144 per
    # leaf at p=4), and `gll_rule` recomputes the GLL Newton iteration and
    # allocates three vectors every time.  The rule and the 1-D second
    # derivative table depend only on `p` and the fixed quadrature abscissae, so
    # one rule and `nq` table entries are enough.  Values are identical, hence
    # `Rmax`/`umax` are bitwise unchanged (G188).
    d2tab_used = if d2tab === nothing
        ev_rule = gll_rule_cached(sq.p)
        t = Vector{Vector{Float64}}(undef, fd_used.nq)
        for k in 1:fd_used.nq
            t[k] = lagrange_d2(ev_rule, fd_used.gl_x[k])[3]
        end
        t
    else
        d2tab
    end
    for qy in 1:fd_used.nq, qx in 1:fd_used.nq
        y = y0 + 0.5 * (fd_used.gl_x[qy] + 1.0) * h
        d2x = d2tab_used[qx]
        d2y = d2tab_used[qy]
        uxq = 0.0; uzq = 0.0
        thn = 0.0; thp = 0.0
        gthx = 0.0; gthz = 0.0; lapth = 0.0
        gthxp = 0.0; gthzp = 0.0; lapp = 0.0
        for j in 1:n, i in 1:n
            k = (j - 1) * n + i
            b = fd_used.phi_v[i, qx] * fd_used.phi_v[j, qy]
            gx = fd_used.dphi_v[i, qx] * fd_used.phi_v[j, qy] * (2.0 / h)
            gz = fd_used.phi_v[i, qx] * fd_used.dphi_v[j, qy] * (2.0 / h)
            uxq += ux[ids[k]] * b
            uzq += uz[ids[k]] * b
            thn += theta[ids[k]] * b
            gthx += theta[ids[k]] * gx
            gthz += theta[ids[k]] * gz
            lapth += theta[ids[k]] *
                     (d2x[i] * fd_used.phi_v[j, qy] * (2.0 / h)^2 +
                      fd_used.phi_v[i, qx] * d2y[j] * (2.0 / h)^2)
            if theta_prev !== nothing
                thp += theta_prev[ids[k]] * b
                gthxp += theta_prev[ids[k]] * gx
                gthzp += theta_prev[ids[k]] * gz
                lapp += theta_prev[ids[k]] *
                        (d2x[i] * fd_used.phi_v[j, qy] * (2.0 / h)^2 +
                         fd_used.phi_v[i, qx] * d2y[j] * (2.0 / h)^2)
            end
        end
        Tmid = Tbar(y, pr.H) + 0.5 * (thn + thp)
        dTdt = dt > 0.0 ? (thn - thp) / dt : 0.0
        adv = 0.5 * uxq * (gthx + gthxp) +
              0.5 * uzq * (gthz + gthzp) + uzq * dTbar(y, pr.H)
        lap = 0.5 * (lapth + lapp)
        R = abs(dTdt + adv - lap)
        alpha == 2 && (R *= abs(Tmid - evg.avgT))
        Rmax = max(Rmax, R)
        umax = max(umax, hypot(uxq, uzq))
    end
    return Rmax, umax
end

function semquad_ev_nu(sq::SemQuad, pr::Params,
                       theta::Vector{Float64},
                       theta_prev::Union{Nothing,Vector{Float64}},
                       ux::Vector{Float64}, uz::Vector{Float64},
                       dt::Float64, lf::NTuple{3,Int}, evg::SEMEVGlobals;
                       ev::Float64, ev_cmax::Float64, ev_alpha::Int=2,
                       ev_timestep::Int=0,
                       fd::Union{Nothing,FEData}=nothing,
                       d2tab::Union{Nothing,Vector{Vector{Float64}}}=nothing)
    # element length for the entropy-viscosity scale: the geometric mean of the
    # two extents (exactly h on square leaves, since h*h is a power of two)
    h = sqrt(leaf_hx(sq, lf[1]) * leaf_hy(sq, lf[1]))
    he = sqrt(2.0) * h
    Rmax, umax = semquad_ev_cell_stats(sq, pr, theta, theta_prev, ux, uz,
                                        dt, lf, evg; alpha=ev_alpha,
                                        fd=fd, d2tab=d2tab)
    nu_max = ev_cmax * umax * he
    if abs(evg.u_infty) < 1.0e-50 || ev_timestep <= 1 ||
       abs(evg.entropy_variation) < 1.0e-50 ||
       abs(evg.field_variation) < 1.0e-50
        nu = nu_max
    else
        nu_e = ev_alpha == 2 ?
               ev * he * he * Rmax / evg.entropy_variation :
               ev * he * Rmax / evg.field_variation
        nu = min(nu_max, nu_e)
    end
    if lf[3] == 0 || lf[3] == sq.ny0 * (1 << lf[1]) - 1
        nu = 0.0
    end
    return nu
end

"""
    semquad_ev_field(sq, qt, pr, theta, ux, uz; dt, ev, ev_cmax, ev_alpha, ev_timestep)

Elementwise entropy-viscosity field (one value per leaf, in `sq.leaves` order)
exactly as `semquad_steady_parts` computes it: `theta_prev = theta` and
`dt_ratio = 0`, i.e. the steady form of the residual.  Exposed so the Newton
JVP can differentiate the field (the EV tangent, `ev_tangent=true`).

`semquad_steady_parts` fills the same values through
`assemble_semquad_energy(...; ev_out=...)`; this routine is the standalone
version used for finite differences.
"""
function semquad_ev_field(sq::SemQuad, qt::Quadtree, pr::Params,
                          theta::Vector{Float64},
                          ux::Vector{Float64}, uz::Vector{Float64};
                          dt::Float64=1.0e-2, ev::Float64=0.0,
                          ev_cmax::Float64=0.052, ev_alpha::Int=2,
                          ev_timestep::Int=10)
    evg = semquad_ev_globals(sq, qt, pr, theta, theta, ux, uz, 0.0)
    out = zeros(Float64, length(sq.leaves))
    fd_ev = make_fedata(sq.p, sq.p)
    d2tab = [lagrange_d2(gll_rule_cached(sq.p), fd_ev.gl_x[k])[3]
             for k in 1:fd_ev.nq]
    for (i, lf) in enumerate(sq.leaves)
        out[i] = semquad_ev_nu(sq, pr, theta, theta, ux, uz, dt, lf, evg;
                               ev=ev, ev_cmax=ev_cmax, ev_alpha=ev_alpha,
                               ev_timestep=ev_timestep, fd=fd_ev, d2tab=d2tab)
    end
    return out
end



# ---- per-transition latent-heat coefficients (docs/dev/PHYSICS.md §7.8) -------
# The energy equation carries the latent heat of every transition as
#
#     (1 + sum_i L_i Gamma_T,i) Dtheta/Dt = lap(theta) - u_z dTbar/dz
#                                           - sum_i L_i Gamma_z,i u_z
#
# `L_i` is `Params.latent_L` (one value for all transitions) in `:global` mode,
# the historical behaviour and still the default, or `PhaseTransition.latent`
# in `:per_phase` mode.  The sign convention is the operator's: `L_i` is the
# coefficient of `-DGamma_i/Dt`, so it is *minus* ASPECT's `T dS_A/(cp dT)`
# (`phase_latent_coefficient`); a negative value (positive Clapeyron slope)
# heats descending material.
"""
    semquad_latent_coeff(pr, pt) -> Float64

The latent coefficient actually applied to transition `pt`: `pr.latent_L` in
`latent_mode = :global` (bitwise the historical value) or `pt.latent` in
`:per_phase`.  Prefer [`semquad_latent_coefficients`](@ref) inside an assembly
loop (it hoists the mode test and the `nothing` check out of the quadrature
loop).
"""
@inline function semquad_latent_coeff(pr::Params, pt::PhaseTransition)
    pr.latent_mode === :per_phase && return _semquad_per_phase_coeff(pt)
    return pr.latent_L
end

@inline function _semquad_per_phase_coeff(pt::PhaseTransition)
    pt.latent === nothing &&
        throw(ArgumentError("latent_mode=:per_phase needs PhaseTransition.latent " *
                            "for every transition (z0=$(pt.z0)); set it with " *
                            "phase_latent_coefficient or per_phase_latent, or " *
                            "use latent_mode=:global with latent_L"))
    return pt.latent::Float64
end

"""
    semquad_latent_active(pr, phases) -> Bool

`true` when the energy equation must carry the latent-heat terms: a non-empty
`phases` list and a non-zero coefficient (`latent_L` in `:global` mode, at
least one per-phase coefficient in `:per_phase` mode).  Throws in `:per_phase`
mode when a transition has no coefficient, so that a misconfigured run fails
loudly instead of silently dropping its latent heat.
"""
@inline function semquad_latent_active(pr::Params,
                                       phases::Vector{PhaseTransition})
    isempty(phases) && return false
    if pr.latent_mode === :per_phase
        any = false
        for pt in phases
            c = _semquad_per_phase_coeff(pt)
            c != 0.0 && (any = true)
        end
        return any
    end
    return pr.latent_L != 0.0
end

"""
    semquad_latent_coefficients(pr, phases) -> Vector{Float64}

`semquad_latent_coeff` for the whole list, materialised once per assembly (the
quadrature loop then indexes a `Vector{Float64}` instead of testing a `Symbol`
per quadrature point).  In `:global` mode every entry is `pr.latent_L`, so the
arithmetic of the historical path is reproduced exactly.
"""
function semquad_latent_coefficients(pr::Params,
                                     phases::Vector{PhaseTransition})
    if pr.latent_mode === :per_phase
        out = Vector{Float64}(undef, length(phases))
        for (i, pt) in enumerate(phases)
            out[i] = _semquad_per_phase_coeff(pt)
        end
        return out
    end
    return fill(pr.latent_L, length(phases))
end

# Per-block buffers for the energy assembly (S4 stage 1).  `assemble_threads=1`
# is the historical serial path; >1 uses one of these per contiguous leaf block
# and merges the blocks in order (deterministic run to run; the load vector is a
# sum of block sums, everything else is assembled in the same leaf order as the
# serial path).
struct SemQuadEnergyScratch
    phi::Vector{Float64}
    gx::Vector{Float64}
    gz::Vector{Float64}
    Me::Matrix{Float64}
    Ke::Matrix{Float64}
    Ce::Matrix{Float64}
end

struct SemQuadEnergyBuf
    IM::Vector{Int}
    JM::Vector{Int}
    VM::Vector{Float64}
    IK::Vector{Int}
    JK::Vector{Int}
    VK::Vector{Float64}
    IC::Vector{Int}
    JC::Vector{Int}
    VC::Vector{Float64}
    F::Vector{Float64}
    scr::SemQuadEnergyScratch
end

struct SemQuadEnergyCfg
    pr::Params
    ux::Vector{Float64}
    uz::Vector{Float64}
    theta::Union{Nothing,Vector{Float64}}
    # Temperature at which the *latent-heat coefficients* are evaluated.  `nothing`
    # means "the same `theta` as everything else" (the historical behaviour, and
    # what a direct `assemble_semquad_energy(...; theta=...)` call gets).  The
    # time-stepping driver passes the current nonlinear iterate here so that the
    # apparent heat capacity is implicit in `theta` while the `rho_heating` /
    # `rho_cp` / adiabatic-heating densities keep using the start-of-step
    # temperature (bitwise unchanged).
    theta_lat::Union{Nothing,Vector{Float64}}
    rho_heating::Bool
    rho_cp::Bool
    ev::Float64
    ev_cmax::Float64
    ev_alpha::Int
    ev_timestep::Int
    evg::Union{Nothing,SEMEVGlobals}
    theta_prev::Union{Nothing,Vector{Float64}}
    dt::Float64
    ev_field::Union{Nothing,Vector{Float64}}
    ev_out::Union{Nothing,Vector{Float64}}
    heating_adiabatic::Bool
    heating_dissipation::Bool
    eta_fun::Union{Nothing,Function}
    rho_ref::Union{Nothing,Function}
    phases::Vector{PhaseTransition}
end

function _semquad_energy_buf_alloc(nleaf::Int, nv::Int, N::Int)
    IM = Int[]; JM = Int[]; VM = Float64[]
    sizehint!(IM, nv * nv * nleaf)
    sizehint!(JM, nv * nv * nleaf)
    sizehint!(VM, nv * nv * nleaf)
    IK = Int[]; JK = Int[]; VK = Float64[]
    sizehint!(IK, nv * nv * nleaf)
    sizehint!(JK, nv * nv * nleaf)
    sizehint!(VK, nv * nv * nleaf)
    IC = Int[]; JC = Int[]; VC = Float64[]
    sizehint!(IC, nv * nv * nleaf)
    sizehint!(JC, nv * nv * nleaf)
    sizehint!(VC, nv * nv * nleaf)
    scr = SemQuadEnergyScratch(zeros(Float64, nv), zeros(Float64, nv),
                               zeros(Float64, nv), zeros(Float64, nv, nv),
                               zeros(Float64, nv, nv), zeros(Float64, nv, nv))
    return SemQuadEnergyBuf(IM, JM, VM, IK, JK, VK, IC, JC, VC,
                            zeros(Float64, N), scr)
end

# Pool of energy-assembly buffers, keyed by the local block size `nv` (S4/S15
# workspace rule: the owner creates the scratch once, the loop reuses it).  At
# p=2 and 10 000 leaves one acquisition `sizehint!`s nine arrays to 810 000
# entries = 58.3 MB; without the pool that capacity was allocated and dropped on
# *every* `assemble_semquad_energy` call (~58 MB of garbage per energy step).
# `_SEMQUAD_ENERGY_BUF_USED` is the per-assembly cursor: `assemble_threads > 1`
# hands one buffer to each contiguous block, so the blocks never share scratch.
# Buffers are `empty!`ed and re-`sizehint!`ed on acquisition, so a smaller tree
# reuses the larger capacity and the COO order is the pushing order, as before
# (bitwise identical assembly).  At most four distinct `nv` keys are kept.
const _SEMQUAD_ENERGY_BUF_CACHE = Dict{Int,Vector{SemQuadEnergyBuf}}()
const _SEMQUAD_ENERGY_BUF_USED = Ref(0)

function _semquad_energy_buf(nleaf::Int, nv::Int, N::Int)
    length(_SEMQUAD_ENERGY_BUF_CACHE) > 4 && empty!(_SEMQUAD_ENERGY_BUF_CACHE)
    pool = get!(() -> SemQuadEnergyBuf[], _SEMQUAD_ENERGY_BUF_CACHE, nv)
    idx = _SEMQUAD_ENERGY_BUF_USED[] + 1
    if idx > length(pool)
        push!(pool, _semquad_energy_buf_alloc(nleaf, nv, N))
    end
    _SEMQUAD_ENERGY_BUF_USED[] = idx
    buf = pool[idx]
    cap = nv * nv * nleaf
    empty!(buf.IM); sizehint!(buf.IM, cap)
    empty!(buf.JM); sizehint!(buf.JM, cap)
    empty!(buf.VM); sizehint!(buf.VM, cap)
    empty!(buf.IK); sizehint!(buf.IK, cap)
    empty!(buf.JK); sizehint!(buf.JK, cap)
    empty!(buf.VK); sizehint!(buf.VK, cap)
    empty!(buf.IC); sizehint!(buf.IC, cap)
    empty!(buf.JC); sizehint!(buf.JC, cap)
    empty!(buf.VC); sizehint!(buf.VC, cap)
    resize!(buf.F, N)
    fill!(buf.F, 0.0)
    return buf
end

# Assemble energy leaves `lo:hi` into `buf` (buffers are per block, so disjoint
# ranges can run concurrently).
function _semquad_assemble_energy_block!(buf::SemQuadEnergyBuf,
                                        cfg::SemQuadEnergyCfg, sq::SemQuad,
                                        leafvec::Vector{NTuple{3,Int}},
                                        lo::Int, hi::Int, fd::FEData,
                                        leaves::Set{NTuple{3,Int}},
                                        fd_ev::Union{Nothing,FEData}=nothing,
                                        d2tab::Vector{Vector{Float64}}=Vector{Vector{Float64}}())
    hi < lo && return buf
    pr = cfg.pr
    ux = cfg.ux
    uz = cfg.uz
    theta = cfg.theta
    theta_lat = cfg.theta_lat
    rho_heating = cfg.rho_heating
    rho_cp = cfg.rho_cp
    ev = cfg.ev
    ev_cmax = cfg.ev_cmax
    ev_alpha = cfg.ev_alpha
    ev_timestep = cfg.ev_timestep
    evg = cfg.evg
    theta_prev = cfg.theta_prev
    dt = cfg.dt
    ev_field = cfg.ev_field
    ev_out = cfg.ev_out
    heating_adiabatic = cfg.heating_adiabatic
    heating_dissipation = cfg.heating_dissipation
    eta_fun = cfg.eta_fun
    rho_ref = cfg.rho_ref
    phases = cfg.phases
    # Hoisted out of the quadrature loop: `:global` gives one `latent_L` per
    # transition (bitwise the historical coefficient), `:per_phase` the stored
    # per-transition values (docs/dev/PHYSICS.md §7.8).
    lcoef = semquad_latent_coefficients(pr, phases)
    # `semquad_latent_active` is a pure function of `(pr, phases)`; it was
    # re-evaluated at every quadrature point (160 000 times per assembly on the
    # 10 000-leaf state).  Same value, hoisted.
    latent_active = semquad_latent_active(pr, phases)
    n = sq.p + 1
    nv = n * n
    IM = buf.IM; JM = buf.JM; VM = buf.VM
    IK = buf.IK; JK = buf.JK; VK = buf.VK
    IC = buf.IC; JC = buf.JC; VC = buf.VC
    F = buf.F
    scr = buf.scr
    phi = scr.phi
    gx = scr.gx
    gz = scr.gz
    Me = scr.Me
    Ke = scr.Ke
    Ce = scr.Ce
    evalw = SemQuadEvalWork(n)

    for ileaf in lo:hi
        lf = leafvec[ileaf]
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = sq.elem[lf]
        nu_ev = ev_field !== nothing ? ev_field[ileaf] :
                (ev > 0.0 ?
                 semquad_ev_nu(sq, pr, theta, theta_prev, ux, uz, dt, lf, evg;
                               ev=ev, ev_cmax=ev_cmax, ev_alpha=ev_alpha,
                               ev_timestep=ev_timestep, fd=fd_ev, d2tab=d2tab) : 0.0)
        ev_out !== nothing && (ev_out[ileaf] = nu_ev)
        for qy in 1:fd.nq, qx in 1:fd.nq
            x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            uxq = 0.0; uzq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                phi[k] = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                gx[k] = fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / hx)
                gz[k] = fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hy)
                uxq += ux[ids[k]] * phi[k]
                uzq += uz[ids[k]] * phi[k]
            end
            # `rho_ref`/`eta_fun` are `Union{Nothing,Function}` fields of the
            # config struct.  A direct call joins the result type with `Any`
            # (the field is not specialised), which used to make `rho`/`src`
            # dynamically typed *inside* the nv x nv element loop: 328M boxed
            # allocations, 5 GiB and ~2.5 s per energy assembly on the
            # 1024-leaf blob tree.  The `::Float64` assertions keep the value
            # identical and the loop monomorphic.
            rho = (rho_ref === nothing ? 1.0 : rho_ref(y))::Float64
            Tq = 0.0
            if rho_heating || rho_cp || (heating_adiabatic && pr.Di > 0.0) ||
               latent_active
                theta === nothing &&
                    throw(ArgumentError("theta is required when latent heat is on"))
                # The `::Vector{Float64}` assertion is load-bearing: `theta` is a
                # `Union{Nothing,Vector{Float64}}` config field, and without it
                # inference makes `Tq` (and `lat_cp`, and `rho` below) `Any`,
                # which re-boxes the nv x nv element loop.
                # The quadrature point lies strictly inside `lf` (Gauss points
                # are interior), so the leaf-local evaluation is the same leaf
                # `_semquad_leaf_at` would find -- the point search is skipped,
                # the arithmetic is identical.
                Tq = semquad_base_T(pr, y) +
                     _semquad_eval_at_leaf!(evalw, sq, theta::Vector{Float64},
                                            x, y, lf)
                (rho_heating || rho_cp) && (rho = max(1.0 - Tq, 1.0e-6))
            end
            # Latent heat as an apparent heat capacity: with
            #   rho cp DT/Dt = div(k grad T) - rho L (Gamma_T DT/Dt + Gamma_z u_z)
            # the material derivative is scaled by (1 + L Gamma_T) and the
            # advective part of the phase change becomes a source -L Gamma_z u_z.
            lat_cp = 0.0
            lat_src = 0.0
            if latent_active
                # `Tl` is a *second* temperature: the latent coefficients can be
                # evaluated at the nonlinear iterate while the density/heating
                # terms above stay at the start-of-step `theta`.  With
                # `theta_lat === nothing` (every direct assembly call, G140
                # included) `Tl` is exactly `Tq` and the assembled matrices are
                # bit-identical to the pre-existing latent implementation.
                Tl = Tq
                if theta_lat !== nothing && theta_lat !== theta
                    Tl = semquad_base_T(pr, y) +
                         _semquad_eval_at_leaf!(evalw, sq,
                                                theta_lat::Vector{Float64},
                                                x, y, lf)
                end
                for i in eachindex(phases)
                    Li = lcoef[i]
                    pt = phases[i]
                    lat_cp += Li * phase_dgamma_dT(pt, y, Tl)
                    lat_src -= Li * phase_dgamma_dz(pt, y, Tl) * uzq
                end
                # ASPECT's guard (`advection.cc`): density*c_P + latent_heat_LHS
                # must stay non-negative.  Here cp = rho = 1, so the apparent
                # capacity is 1 + lat_cp; below zero the energy equation turns
                # backward in time and the run blows up (measured: latent_L = 10
                # with the G194 transition gives 1 + L*Gamma_T = -1.5 and
                # max|theta| = 14 after 120 steps).  In `:per_phase` mode
                # `lat_cp` is the sum over transitions of `latent_i*Gamma_T,i`.
                if 1.0 + lat_cp < 0.0
                    throw(ArgumentError("latent heat: the apparent heat capacity " *
                                        "1 + sum_i latent_i*dGamma_i/dT = " *
                                        "$(1.0 + lat_cp) is negative at y=$y; " *
                                        "reduce the latent coefficients or widen " *
                                        "the phase transitions"))
                end
            end
            src = -rho * uzq * semquad_base_gradient(pr, y) + lat_src
            # theta = T - Tbar, so lap(T) = lap(theta) + lap(Tbar); the base
            # curvature is a genuine source (zero for the linear Boussinesq
            # profile, (alpha g/cp)^2 Tad for the exponential adiabat).
            src += semquad_base_laplacian(pr, y)
            if rho_heating
                src -= Tq * pr.H
            end
            if pr.Di > 0.0 && (heating_adiabatic || heating_dissipation)
                # Velocity gradients at this quadrature point (gx/gz are the
                # basis gradients, already scaled by 2/h).
                duxdx = 0.0; duxdz = 0.0; duzdx = 0.0; duzdz = 0.0
                for k in 1:nv
                    duxdx += ux[ids[k]] * gx[k]
                    duxdz += ux[ids[k]] * gz[k]
                    duzdx += uz[ids[k]] * gx[k]
                    duzdz += uz[ids[k]] * gz[k]
                end
                if heating_dissipation
                    exz = 0.5 * (duxdz + duzdx)
                    eed = duxdx^2 + duzdz^2 + 2.0 * exz^2      # edot:edot
                    if pr.deviatoric
                        # ASPECT embeds the 2-D flow in 3-D and uses the 1/3
                        # deviatoric projection (manual, "Two-dimensional
                        # models"), so eps_d:eps_d = eps:eps - (1/3) div(u)^2.
                        eed -= (1.0 / 3.0) * (duxdx + duzdz)^2
                    end
                    eta_q = (eta_fun === nothing ? 1.0 : eta_fun(x, y))::Float64
                    src += (pr.Di / pr.Ra) * 2.0 * eta_q * eed
                end
                if heating_adiabatic
                    # Tq already contains Tbar + theta (it is computed above when
                    # this term is on), so it must not be added again.
                    Tfull = theta === nothing ? semquad_base_T(pr, y) : Tq
                    # ASPECT's simplified adiabatic heating is alpha rho T (u.g):
                    # the density factor must stay (King has rho = exp(depth Di)).
                    rho_ad = (pr.refstate === nothing ? 1.0 :
                              semquad_rho(pr.refstate, y))::Float64
                    src -= pr.Di * rho_ad * Tfull * uzq
                end
            end
            # Local mass/stiffness/advection blocks.  The per-quadrature scalars
            # are hoisted once (their left-to-right product tree is preserved
            # exactly, e.g. `((1+lat_cp)*w)*rho*phi[k]` -> `cM*phi[k]`), and the
            # inner loop index is `k` so every `Me/Ke/Ce[k,l]` store is
            # unit-stride.  Each entry still accumulates over q in the same
            # order, so the assembled matrices are bitwise identical (G188).
            cM = (1.0 + lat_cp) * w * rho
            cK = w * (1.0 + nu_ev)
            ws = w * src
            for k in 1:nv
                F[ids[k]] += ws * phi[k]
            end
            for l in 1:nv
                pl = phi[l]
                xl = gx[l]
                zl = gz[l]
                cx = uxq * xl + uzq * zl
                @simd for k in 1:nv
                    pk = phi[k]
                    ck = cM * pk
                    Me[k, l] += ck * pl
                    Ke[k, l] += cK * (gx[k] * xl + gz[k] * zl)
                    Ce[k, l] += ck * cx
                end
            end
        end
        for l in 1:nv, k in 1:nv
            vm = Me[k, l]
            vm != 0.0 && (push!(IM, ids[k]); push!(JM, ids[l]); push!(VM, vm))
            vk = Ke[k, l]
            vk != 0.0 && (push!(IK, ids[k]); push!(JK, ids[l]); push!(VK, vk))
            vc = Ce[k, l]
            vc != 0.0 && (push!(IC, ids[k]); push!(JC, ids[l]); push!(VC, vc))
        end
        fill!(Me, 0.0)
        fill!(Ke, 0.0)
        fill!(Ce, 0.0)
    end
    return buf
end

# Split the energy leaves into `nt` contiguous blocks, assemble them (in
# parallel when Julia has more than one thread) and merge the block buffers in
# block order.  `nt == 1` is the historical serial path, returned without a
# copy or re-ordering.
function _semquad_assemble_energy_blocks(nt::Int, cfg::SemQuadEnergyCfg,
                                         sq::SemQuad, fd::FEData,
                                         leaves::Set{NTuple{3,Int}})
    n = sq.p + 1
    nv = n * n
    nleaf = length(sq.leaves)
    _SEMQUAD_ENERGY_BUF_USED[] = 0
    # EV quadrature data: `make_fedata(p, p)` ran once per *leaf* inside
    # `semquad_ev_cell_stats` (a GLL Newton iteration + a Gauss-Legendre
    # `eigen` + 6 matrices per leaf, 57.8 MB and 44 ms per assembly on the
    # 10 000-leaf state).  It depends only on `p`, so it is built once here and
    # threaded through; the values are identical, hence the EV field is bitwise
    # unchanged.
    fd_ev = (cfg.ev > 0.0 && cfg.ev_field === nothing) ?
            make_fedata(sq.p, sq.p) : nothing
    d2tab = fd_ev === nothing ? Vector{Vector{Float64}}() :
            [lagrange_d2(gll_rule_cached(sq.p), fd_ev.gl_x[k])[3]
             for k in 1:fd_ev.nq]
    if nt <= 1
        buf = _semquad_energy_buf(nleaf, nv, sq.N)
        _semquad_assemble_energy_block!(buf, cfg, sq, sq.leaves, 1, nleaf, fd,
                                        leaves, fd_ev, d2tab)
        # the COO arrays are consumed by `sparse` inside the caller; `F` is
        # returned to the caller (and `return_parts=true` keeps it across
        # further assemblies), so it must not alias the pooled buffer.
        return (IM=buf.IM, JM=buf.JM, VM=buf.VM,
                IK=buf.IK, JK=buf.JK, VK=buf.VK,
                IC=buf.IC, JC=buf.JC, VC=buf.VC, F=copy(buf.F))
    end
    bufs = Vector{SemQuadEnergyBuf}(undef, nt)
    los = Vector{Int}(undef, nt)
    his = Vector{Int}(undef, nt)
    base, rem = divrem(nleaf, nt)
    off = 1
    for t in 1:nt
        cnt = base + (t <= rem ? 1 : 0)
        los[t] = off
        his[t] = off + cnt - 1
        off += cnt
        bufs[t] = _semquad_energy_buf(cnt, nv, sq.N)
    end
    Threads.@threads :static for t in 1:nt
        _semquad_assemble_energy_block!(bufs[t], cfg, sq, sq.leaves, los[t],
                                        his[t], fd, leaves, fd_ev, d2tab)
    end
    F = zeros(Float64, sq.N)
    for t in 1:nt
        F .+= bufs[t].F
    end
    return (IM=_semquad_concat_blocks([b.IM for b in bufs]),
            JM=_semquad_concat_blocks([b.JM for b in bufs]),
            VM=_semquad_concat_blocks([b.VM for b in bufs]),
            IK=_semquad_concat_blocks([b.IK for b in bufs]),
            JK=_semquad_concat_blocks([b.JK for b in bufs]),
            VK=_semquad_concat_blocks([b.VK for b in bufs]),
            IC=_semquad_concat_blocks([b.IC for b in bufs]),
            JC=_semquad_concat_blocks([b.JC for b in bufs]),
            VC=_semquad_concat_blocks([b.VC for b in bufs]), F=F)
end


"""
    assemble_semquad_energy(sq, qt, pr, ux, uz; advection=:skew)

Assemble `M` (mass), `K` (diffusion), `C` (advection of theta) and `F` (base
state `-u_z dTbar/dz`) on the quadtree.  `ux,uz` live on the same Q_p
quadtree space as theta; `advection` is `:galerkin`, `:skew` (`0.5(C-C')`) or
`:conservative` (`-C'`).  `rho_heating=true` adds the ASPECT-style
`rho H` correction `-T H` (since `rho=1-T`, `rho H - H = -T H`) and requires
`theta`.  No boundary conditions or hanging constraints are
applied here; the caller forms `C' A C` with `ThetaMap`.

`ev_field` reuses an elementwise EV field from an earlier assembly (one value per
leaf, in `sq.leaves` order) instead of recomputing it from `ev`/`evg`; this is
what lets the CBF residual use the exact operator that produced the state.
`ev_out` records the field that was actually used, so the driver can checkpoint
it.  Both are length-checked against `sq.leaves`.

`assemble_threads=1` (default) assembles the leaves serially, bit for bit as
before.  `assemble_threads=N>1` (S4 stage 1) cuts the leaves into `N` fixed
contiguous blocks, assembles each block into its own COO buffers and merges them
in block order; the result is deterministic run to run and the COO order equals
the serial one, but `F` is a sum of block sums (so it can differ at round-off).
With 1 Julia thread the request warns once and falls back to serial.

`phases` turns on the latent heat of the phase transitions as an apparent heat
capacity (docs/dev/PHYSICS.md §7): `M` and `C` are weighted by
`(1 + sum_i latent_i * Gamma_T,i(y,T))` and the source gains
`-sum_i latent_i * Gamma_z,i * u_z` (`Gamma_i` is the same `phase_gamma` the
phase buoyancy uses).  The coefficient `latent_i` is `pr.latent_L` for every
transition in `latent_mode = :global` (the historical behaviour) and
`pt.latent` in `latent_mode = :per_phase` (docs/dev/PHYSICS.md §7.8, where the
sign convention — the coefficient is ASPECT's `T dS_A` with the opposite sign —
is derived).  `theta_lat` evaluates the capacity/source coefficients at a
different temperature than `theta` (the current iterate of a time step, so the
capacity is implicit); it defaults to `theta` and only changes anything when
phases are active.
"""
function assemble_semquad_energy(sq::SemQuad, qt::Quadtree, pr::Params,
                                 ux::Vector{Float64}, uz::Vector{Float64};
                                 advection::Symbol=:skew,
                                 theta::Union{Nothing,Vector{Float64}}=nothing,
                                 theta_lat::Union{Nothing,Vector{Float64}}=nothing,
                                 rho_heating::Bool=false,
                                 rho_cp::Bool=false,
                                 ev::Float64=0.0, ev_cmax::Float64=0.052,
                                 ev_alpha::Int=2, ev_timestep::Int=0,
                                 evg::Union{Nothing,SEMEVGlobals}=nothing,
                                 theta_prev::Union{Nothing,Vector{Float64}}=nothing,
                                 dt::Float64=0.0,
                                 ev_field::Union{Nothing,Vector{Float64}}=nothing,
                                 ev_out::Union{Nothing,Vector{Float64}}=nothing,
                                 heating_adiabatic::Bool=false,
                                 heating_dissipation::Bool=false,
                                 eta_fun::Union{Nothing,Function}=nothing,
                                 rho_ref::Union{Nothing,Function}=nothing,
                                 phases::Vector{PhaseTransition}=PhaseTransition[],
                                 assemble_threads::Int=1)
    advection in (:galerkin, :skew, :conservative) ||
        throw(ArgumentError("advection must be galerkin, skew or conservative"))
    assemble_threads >= 1 ||
        throw(ArgumentError("assemble_threads must be >= 1"))
    (rho_heating || rho_cp || (heating_adiabatic && pr.Di > 0.0)) &&
        theta === nothing &&
        throw(ArgumentError("theta is required when rho_heating/rho_cp/adiabatic heating is on"))
    # `ev_field` supplies the elementwise viscosity of a previous assembly (the
    # consistent-boundary-flux residual must use the operator that produced the
    # current theta).  When it is given, the EV globals are not needed and the
    # field is used verbatim; otherwise the field is recomputed from `ev`.
    if ev > 0.0 && ev_field === nothing && (evg === nothing || theta === nothing)
        throw(ArgumentError("evg and theta are required when ev>0"))
    end
    if ev_field !== nothing
        length(ev_field) == length(sq.leaves) ||
            throw(ArgumentError("ev_field has $(length(ev_field)) entries, " *
                                "mesh has $(length(sq.leaves)) leaves"))
    end
    if ev_out !== nothing
        length(ev_out) == length(sq.leaves) ||
            throw(ArgumentError("ev_out has $(length(ev_out)) entries, " *
                                "mesh has $(length(sq.leaves)) leaves"))
    end
    if theta_lat !== nothing
        length(theta_lat) == sq.N ||
            throw(ArgumentError("theta_lat has $(length(theta_lat)) entries, " *
                                "mesh has $(sq.N) dofs"))
    end
    N = sq.N
    p = sq.p
    fd = make_fedata(p, p - 1)
    leaves = Set(sq.leaves)
    cfg = SemQuadEnergyCfg(pr, ux, uz, theta, theta_lat, rho_heating, rho_cp,
                           ev, ev_cmax, ev_alpha, ev_timestep, evg, theta_prev,
                           dt, ev_field, ev_out, heating_adiabatic,
                           heating_dissipation, eta_fun, rho_ref, phases)
    asm = _semquad_assemble_energy_blocks(
        _semquad_asm_thread_count(assemble_threads), cfg, sq, fd, leaves)
    IM = asm.IM; JM = asm.JM; VM = asm.VM
    IK = asm.IK; JK = asm.JK; VK = asm.VK
    IC = asm.IC; JC = asm.JC; VC = asm.VC
    F = asm.F

    M = sparse(IM, JM, VM, N, N)
    K = sparse(IK, JK, VK, N, N)
    C = sparse(IC, JC, VC, N, N)
    if advection === :skew
        C = 0.5 * (C - C')
    elseif advection === :conservative
        C = -C'
    end
    return M, K, C, F
end

"""
    assemble_semquad_energy_latent_jvp(sq, qt, pr, phases, dtheta, theta, ux, uz;
                                       advection=:galerkin, rho_ref=nothing,
                                       rho_heating=false, rho_cp=false)

Directional derivative of the **latent-heat** part of
[`assemble_semquad_energy`](@ref) in the temperature direction `dtheta`, at the
temperature `theta` (the apparent heat capacity is
`1 + sum_i latent_i * Gamma_T,i` and the source is
`-sum_i latent_i * Gamma_z,i u_z`; `latent_i` is `pr.latent_L` in `:global`
mode or `PhaseTransition.latent` in `:per_phase` mode, docs/dev/PHYSICS.md §7.8):

    Md_ij = int  L_i Gamma_TT,i(y,T) dtheta phi_i phi_j
    Cd_ij = int  L_i Gamma_TT,i(y,T) dtheta phi_i (u.grad phi_j)
    Fd_i  = int -L_i Gamma_zT,i(y,T) dtheta u_z phi_i

so the per-step Newton's energy tangent is
`Cfree' (A_E dtheta + ca*Md*theta - cb*Md*theta_n + cc*Md*theta_prev + Cd*theta - Fd)`
(the `ca/cb/cc` are the BDF2 coefficients of `solve_semquad_energy`).  The phase
derivatives are the shared `phase_d2gamma_dT2` / `phase_d2gamma_dzdT`; an empty
`phases` list or all-zero coefficients returns zero operators.  `Cd` is
transformed by the same `advection` rule as the assembled `C`
(`:skew` / `:conservative`).

As in the rest of the per-step Newton, the temperature derivative of the
`rho_heating`/`rho_cp` density factor is not differentiated (it is a separate,
pre-existing approximation).
"""
function assemble_semquad_energy_latent_jvp(sq::SemQuad, qt::Quadtree,
                                            pr::Params,
                                            phases::Vector{PhaseTransition},
                                            dtheta::Vector{Float64},
                                            theta::Vector{Float64},
                                            ux::Vector{Float64},
                                            uz::Vector{Float64};
                                            advection::Symbol=:galerkin,
                                            rho_ref::Union{Nothing,Function}=nothing,
                                            rho_heating::Bool=false,
                                            rho_cp::Bool=false)
    N = sq.N
    n = sq.p + 1
    nv = n * n
    if !semquad_latent_active(pr, phases)
        return (spzeros(Float64, N, N), spzeros(Float64, N, N),
                zeros(Float64, N))
    end
    lcoef = semquad_latent_coefficients(pr, phases)
    per_phase = pr.latent_mode === :per_phase
    fd = make_fedata(sq.p, sq.p - 1)
    IM = Int[]; JM = Int[]; VM = Float64[]
    sizehint!(IM, nv * nv * length(sq.leaves))
    sizehint!(JM, nv * nv * length(sq.leaves))
    sizehint!(VM, nv * nv * length(sq.leaves))
    IC = Int[]; JC = Int[]; VC = Float64[]
    sizehint!(IC, nv * nv * length(sq.leaves))
    sizehint!(JC, nv * nv * length(sq.leaves))
    sizehint!(VC, nv * nv * length(sq.leaves))
    F = zeros(Float64, N)
    phi = zeros(Float64, nv)
    gx = zeros(Float64, nv)
    gz = zeros(Float64, nv)
    Me = zeros(Float64, nv, nv)
    Ce = zeros(Float64, nv, nv)
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            uxq = 0.0; uzq = 0.0; Tq = 0.0; dthq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                phi[k] = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                gx[k] = fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / hx)
                gz[k] = fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hy)
                uxq += ux[ids[k]] * phi[k]
                uzq += uz[ids[k]] * phi[k]
                Tq += theta[ids[k]] * phi[k]
                dthq += dtheta[ids[k]] * phi[k]
            end
            Tq += semquad_base_T(pr, y)
            rho = (rho_ref === nothing ? 1.0 : rho_ref(y))::Float64
            (rho_heating || rho_cp) && (rho = max(1.0 - Tq, 1.0e-6))
            gTT = 0.0; gzT = 0.0
            # `:global` keeps the historical arithmetic exactly (one `latent_L`
            # applied *after* the sum of the second derivatives); `:per_phase`
            # weights each transition inside the sum, which is a different
            # rounding even when the coefficients happen to be equal.
            if per_phase
                for i in eachindex(phases)
                    Li = lcoef[i]
                    gTT += Li * phase_d2gamma_dT2(phases[i], y, Tq)
                    gzT += Li * phase_d2gamma_dzdT(phases[i], y, Tq)
                end
                cdir = dthq
            else
                for pt in phases
                    gTT += phase_d2gamma_dT2(pt, y, Tq)
                    gzT += phase_d2gamma_dzdT(pt, y, Tq)
                end
                cdir = pr.latent_L * dthq
            end
            ws = w * (cdir * gTT)
            cM = ws * rho
            for k in 1:nv
                F[ids[k]] += (-(cdir * gzT) * uzq) * w * phi[k]
            end
            for l in 1:nv
                pl = phi[l]
                cx = uxq * gx[l] + uzq * gz[l]
                @simd for k in 1:nv
                    pk = phi[k]
                    ck = cM * pk
                    Me[k, l] += ck * pl
                    Ce[k, l] += ck * cx
                end
            end
        end
        for l in 1:nv, k in 1:nv
            vm = Me[k, l]
            vm != 0.0 && (push!(IM, ids[k]); push!(JM, ids[l]); push!(VM, vm))
            vc = Ce[k, l]
            vc != 0.0 && (push!(IC, ids[k]); push!(JC, ids[l]); push!(VC, vc))
        end
        fill!(Me, 0.0)
        fill!(Ce, 0.0)
    end
    Md = sparse(IM, JM, VM, N, N)
    Cd = sparse(IC, JC, VC, N, N)
    if advection === :skew
        Cd = 0.5 * (Cd - Cd')
    elseif advection === :conservative
        Cd = -Cd'
    end
    return (Md, Cd, F)
end
