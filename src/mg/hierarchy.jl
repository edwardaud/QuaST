# Scalable Stokes solve on a quadtree forest: geometric multigrid for the
# velocity block + a pressure Schur block, used as a right preconditioner for
# GMRES on the reduced saddle system.
#
# The hierarchy is built by *coarsening the tree*, not by assuming a uniform
# grid: repeatedly merge every 2x2 quartet of same-level siblings, balance, and
# repeat.  Each coarse level keeps the same base cells and box, so every coarse
# leaf is a union of fine leaves.  Per level:
#   * A_k = C_k' K_k C_k from the production assembler (`assemble_only`), with
#     `C_k` the hanging-node + Dirichlet affine constraint of that tree;
#   * P_k = (fine free rows of the nodal interpolation) * C_{k-1}; a coarse
#     field evaluated on the fine tree lies in the fine constrained space by
#     construction (coarse leaves are unions of fine leaves), so selecting the
#     fine free rows is exact;
#   * V-cycle with 2+2 damped Jacobi (omega=2/3) and a Cholesky coarse solve.
#
# On a uniform 5x2 forest, p=2, eta=1, this gives a level-independent V-cycle
# factor of 0.33 and 55 GMRES iterations for the saddle (93k -> 371k dofs,
# 7.4 -> 21.5 s).  The non-uniform path is validated on wall-refined 2b trees
# in G153; it uses the same `P = free rows * Pnodal * C_coarse` construction as
# deal.II's global-coarsening multigrid (see out_julia/agent/RESEARCH...).
#
# The saddle is solved by right-preconditioned GMRES with a block-triangular
# preconditioner: MG velocity solve + pressure Schur solve (1/eta-weighted mass
# for low contrast, LSC commutator above 1000x, selected automatically).

const SEMQUAD_MG_CACHE = Dict{Any,Any}()

const _SEMQUAD_MG_NO_THREADS_WARNED = Ref(false)
const _SEMQUAD_MG_CLAMP_WARNED = Ref(false)

# Resolve the requested `mg_threads` against the actual Julia thread pool: 1 is
# the serial (bitwise, historical) path, a request larger than the pool falls
# back to the pool size and a single-threaded process warns once and stays
# serial.  Same policy as `_semquad_asm_thread_count` (docs/dev/SOLVER.md §12).
function _semquad_mg_thread_count(mg_threads::Int)
    mg_threads >= 1 || throw(ArgumentError("mg_threads must be >= 1"))
    mg_threads == 1 && return 1
    nproc = Threads.nthreads()
    if nproc == 1
        if !_SEMQUAD_MG_NO_THREADS_WARNED[]
            _SEMQUAD_MG_NO_THREADS_WARNED[] = true
            @warn "mg_threads=$(mg_threads) requested, but Julia was started " *
                  "with 1 thread; falling back to the serial V-cycle. Start " *
                  "Julia with --threads=N (or JULIA_NUM_THREADS=N) for the " *
                  "threaded leaf-local V-cycle." maxlog=1
        end
        return 1
    end
    nt = min(mg_threads, nproc)
    if nt < mg_threads && !_SEMQUAD_MG_CLAMP_WARNED[]
        _SEMQUAD_MG_CLAMP_WARNED[] = true
        @warn "mg_threads=$(mg_threads) exceeds Threads.nthreads()=$(nproc); " *
              "using $nt leaf blocks" maxlog=1
    end
    return nt
end

# Resolved V-cycle smoothing schedule of the last hierarchy that was built:
# `(schedule::Symbol, omega::Float64)` where `schedule` is the *resolved* one
# (`:auto` already mapped to `:uniform`/`:taper`).  Diagnostic for the gate that
# checks the `mg_smoothing`/`mg_omega` driver plumbing (G184); the solve itself
# reads the schedule from `SemQuadMG.pre`/`post`/`omega`.
const LAST_SEMSTOKES_MG_PROFILE = Ref{Tuple{Symbol,Float64}}((:none, NaN))

struct SemQuadMGGeom
    p::Int
    L::Int
    Lx::Float64
    nx0::Int
    ny0::Int
    bc::Symbol
    deviatoric::Bool
    geom::AbstractGeometry                      # the geometry every level lives on
    qts::Vector{Quadtree}                       # coarsest first, fine last
    sss::Vector{SemStokes}
    Cus::Vector{SparseMatrixCSC{Float64,Int}}   # full -> free per level
    Cuts::Vector{SparseMatrixCSC{Float64,Int}}  # transpose of Cus (free -> full)
    Ps::Vector{SparseMatrixCSC{Float64,Int}}    # reduced prolongation k -> k+1
    nfs::Vector{Int}
    free_rows::Vector{Vector{Int}}              # full rows that are free
    coarse_direct::Bool                         # coarsest level solved by a direct factorisation
end

"""
    SemQuadMG

One multigrid hierarchy.  `operator` selects how every level applies its
velocity operator:

* `:sparse` -- the rediscretised/Galerkin sparse `A[k]` (historical path, the
  default, bit-identical to before S3);
* `:matrixfree` -- `semquad_apply_Kred!` on the level's leaf-local operator
  (`mfops[k]`, `mfetas[k]`, `mfwork[k]`).  `A` is then **empty**: no velocity
  operator matrix exists anywhere in the hierarchy (MF-3).

`sc[k]`/`sr[k]` are per-level scratch vectors (residual and smoother residual)
so a V-cycle does not allocate on the large levels.

`threads`/`par` are the S5-2 threaded-V-cycle state: `threads == 1` is the
historical serial path (bitwise), `threads = n > 1` gives every level a
`SemQuadKPar` with `n` fixed contiguous leaf blocks.  The leaf-local work of the
matrix-free apply, the damped-Jacobi sweeps and the residual evaluation is then
spread over the Julia thread pool, but the scatter-adds stay in the serial
leaf order, so a threaded V-cycle is **bitwise identical** to the serial one
(G190).
"""
struct SemQuadMG
    geo::SemQuadMGGeom
    A::Vector{SparseMatrixCSC{Float64,Int}}
    D::Vector{Vector{Float64}}
    cA::Any
    cycles::Int
    pre::Vector{Int}
    post::Vector{Int}
    omega::Float64
    smoother::Symbol
    LF::Vector{Any}
    UF::Vector{Any}
    operator::Symbol
    mfops::Vector{AnySemQuadKOp}
    mfetas::Vector{SemQuadEtaLane}
    mfwork::Vector{Any}
    sc::Vector{Vector{Float64}}
    sr::Vector{Vector{Float64}}
    sy::Vector{Vector{Float64}}
    threads::Int
    par::Vector{Any}                 # SemQuadKPar per level (empty when threads == 1)
    Pts::Vector{Any}                 # transposed Ps per level (empty when threads == 1)
    rcs::Vector{Vector{Float64}}     # restriction output per level
    ecs::Vector{Vector{Float64}}     # coarse error per level
    pys::Vector{Vector{Float64}}     # prolongation output per level
end

"""
    _mg_sweeps(v, nlev, what)

Normalise a pre/post smoothing specification to a `nlev`-vector: an `Int` means
the same number of sweeps on every level, a vector is read **coarse-first**
(`v[1]` is the coarsest level, `v[end]` the fine one, i.e. hierarchy order) and
must have one entry per level.
"""
function _mg_sweeps(v::Union{Integer,AbstractVector{<:Integer}}, nlev::Int,
                    what::AbstractString)
    if v isa Integer
        v >= 0 || throw(ArgumentError("MG $what sweeps must be >= 0"))
        return fill(Int(v), nlev)
    end
    length(v) == nlev ||
        throw(ArgumentError("MG $what sweeps has $(length(v)) entries, " *
                            "expected $nlev (one per level, coarse first)"))
    all(x -> x >= 0, v) ||
        throw(ArgumentError("MG $what sweeps must be >= 0"))
    return Int.(v)
end

"""
    _mg_smoothing_schedule(schedule, nlev, pre, post)

Per-level smoothing-sweep vectors of `make_semquad_mg`, **coarse first**
(`[1]` is the coarsest level, `[end]` the fine one).

* `:uniform` -- `pre` and `post` sweeps on every level (the historical V-cycle);
* `:taper` -- `1, 2, 2, 4, 4, 8, 8, ...` sweeps from the fine level down.  A
  smoothing sweep on the next coarser level costs ~1/1.7-4 of the finer one, so
  this spends the smoothing budget where it is cheapest.  See
  docs/dev/SOLVER.md §14 for the measurement.
"""
function _mg_smoothing_schedule(schedule::Symbol, nlev::Int, pre::Int,
                                post::Int)
    if schedule === :uniform
        return fill(pre, nlev), fill(post, nlev)
    elseif schedule === :taper
        # position from the fine level: k = 1 (fine) .. nlev (coarsest)
        v = Int[min(1 << (k ÷ 2), 8) for k in 1:nlev]
        return reverse(v), reverse(v)
    end
    throw(ArgumentError("MG smoothing schedule must be :uniform or :taper"))
end

"""
    _mg_smoothing_profile(smoothing, geo, omega) -> (schedule, omega, heavy)

Resolve `smoothing`/`omega` for the hierarchy `geo`.

`:auto` (the default) keeps the textbook uniform 2+2 schedule unless the
hierarchy is **top-heavy**, i.e. its finest coarsening step keeps more than half
of the fine-level dofs.  What a top-heavy hierarchy wants depends on the
polynomial degree: at p=2 (the production P1 tree) one extra uniform sweep is
the measured winner (uniform 3+3: 65 outer iterations against 84 for 2+2 and 90
for the taper), while at p>=4 an apply is 4-16x more expensive and the measured
front-loaded `:taper` schedule (8,8,... on the cheap levels) stays.  The
6-level F tree is not top-heavy at either p and keeps 2+2.  `raise` reports the
p=2/auto decision so `make_semquad_mg` can raise the default sweep count to 3
without overriding an explicit `pre`/`post`.

`omega=nothing` picks the damping that goes with the resolved schedule: 2/3 for
`:uniform`, 0.55 for `:taper`.
"""
function _mg_smoothing_profile(smoothing::Symbol, geo::SemQuadMGGeom,
                               omega::Union{Nothing,Float64})
    heavy = false
    raise = false
    sched = if smoothing === :auto
        nlev = length(geo.nfs)
        heavy = nlev >= 3 && geo.nfs[end - 1] > 0.5 * geo.nfs[end]
        if heavy && geo.p <= 2
            # p=2: the top-heavy production P1 hierarchy is cheapest with one
            # extra uniform sweep (65 iterations against 90 for the taper).
            raise = true
            :uniform
        elseif heavy
            # p>=4: one apply is 4-16x more expensive, so the measured
            # front-loaded taper (8,8,... on the cheap levels) stays.
            :taper
        else
            :uniform
        end
    else
        smoothing
    end
    om = omega === nothing ? (sched === :taper ? 0.55 : 2.0 / 3.0) : omega
    0.0 < om < 1.0 ||
        throw(ArgumentError("MG Jacobi damping omega must be in (0,1)"))
    return sched, om, raise
end

"""
    _semquad_uniform_level(Lx, nx0, ny0, l)

The uniform level-`l` tree with the same base cells and box width.  Kept for
tests and for building reference hierarchies; `Quadtree(l; ...)` only creates
the level-0 base leaves (`maxlevel` is a cap).
"""
function _semquad_uniform_level(Lx::Float64, nx0::Int, ny0::Int, l::Int)
    qt = Quadtree(l; Lx=Lx, cells_x=nx0, cells_y=ny0)
    for _ in 1:l, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

"""
    _mg_copy_tree(qt)

An independent copy of `qt` (the leaves set is copied, so merging does not touch
the input tree).
"""
function _mg_copy_tree(qt::Quadtree)
    qc = Quadtree(qt.maxlevel; Lx=qt.Lx, cells_x=qt.nx0, cells_y=qt.ny0)
    empty!(qc.leaves)          # the constructor seeds the level-0 base leaves
    for lf in qt.leaves
        push!(qc.leaves, lf)
    end
    _invalidate_faces!(qc)     # the leaves were replaced wholesale
    return qc
end

"""
    _mg_coarsen_layer!(qc)

One *layer* of coarsening, in place: merge complete 2x2 sibling quartets at the
deepest level that has any, then `balance!`, and stop.  Returns `false` (with
`qc` unchanged) when nothing merged, i.e. the hierarchy has bottomed out.
`balance!` may re-refine a merged leaf, but only into its four original
children, so every coarse leaf remains a union of input leaves -- that is what
makes the nodal prolongation exact.
"""
function _mg_coarsen_layer!(qc::Quadtree)
    levels = sort(unique(lf[1] for lf in qc.leaves); rev=true)
    merged = false
    for l in levels
        l == 0 && continue
        for lf in sort(collect(qc.leaves))
            lf in qc.leaves || continue
            lf[1] == l || continue
            coarsen!(qc, lf) && (merged = true)
        end
        merged && break
    end
    merged || return false
    balance!(qc)
    max_face_level_jump(qc) <= 1 ||
        error("MG coarsening produced a tree that is not 2:1-balanced")
    return true
end

"""
    _mg_coarsen_tree(qt; layers=1, factor=1.0)

Coarsen one hierarchy step: `layers` merge+balance layers, and then further
layers while the leaf count has not yet dropped by `factor` (a step never does
fewer than `layers` layers).  Merging exactly one layer per step is the
historical rule (`layers=1, factor=1.0`); on a graded AMR tree one layer near
the fine end coarsens by only ~1.7x, which leaves a barely-coarser level that
costs almost as much as the fine one -- `factor=2.0` merges exactly those extra
layers, while remaining a bitwise no-op on uniform and wall-graded trees (one
layer there already coarsens by 4x / 2.2x).

Returns a tree with no more leaves than `qt`; equality means the hierarchy has
bottomed out.
"""
function _mg_coarsen_tree(qt::Quadtree; layers::Int=1, factor::Float64=1.0)
    layers >= 1 || throw(ArgumentError("MG coarsening needs layers >= 1"))
    (factor >= 1.0 && isfinite(factor)) ||
        throw(ArgumentError("MG coarsening factor must be >= 1"))
    qc = _mg_copy_tree(qt)
    n0 = length(qc.leaves)
    done = 0
    while done < layers || (factor > 1.0 && length(qc.leaves) * factor > n0)
        _mg_coarsen_layer!(qc) || break
        done += 1
    end
    return qc
end

"""
    _mg_hierarchy_trees(qt; layers=1, factor=1.0)

Coarsest-first list of trees obtained by repeated `_mg_coarsen_tree`.  The last
element is `qt` itself.  A uniform tree reduces to the same base leaves in
`maxlevel` steps; a graded tree usually reaches the base in fewer.

With the historical `layers=1, factor=1.0` a uniform 2^k tree gets `k+1` levels
and a graded AMR tree one level per tree layer; `factor=2.0` additionally skips
hierarchy steps that would coarsen by less than 2x.
"""
function _mg_hierarchy_trees(qt::Quadtree; layers::Int=1, factor::Float64=1.0)
    trees = Quadtree[qt]
    q = qt
    for _ in 1:(qt.maxlevel + 2)
        qc = _mg_coarsen_tree(q; layers=layers, factor=factor)
        length(qc.leaves) < length(q.leaves) || break
        pushfirst!(trees, qc)   # trees is coarse-first
        q = qc
        length(q.leaves) == qt.nx0 * qt.ny0 && break
    end
    return trees
end

"""
    _mg_free_rows(Cu, N)

Rows of the affine constraint matrix `Cu` (full -> free) that are free: a free
row has exactly one stored entry, equal to 1.  Hanging-node rows are linear
combinations and `_affine_constraint` skips a constrained node that coincides
with a coarse node, so a single 1 identifies free rows unambiguously.
"""
function _mg_free_rows(Cu::SparseMatrixCSC{Float64,Int}, N::Int)
    count = zeros(Int, N)
    onlyone = trues(N)
    rows, _, vals = findnz(Cu)
    for k in eachindex(rows)
        i = rows[k]
        count[i] += 1
        abs(vals[k] - 1.0) > 1.0e-14 && (onlyone[i] = false)
    end
    return [i for i in 1:N if count[i] == 1 && onlyone[i]]
end

"""
    _mg_prolongation_full(fine::SemQuad, coarse::SemQuad, p)

Nodal interpolation `Pfull` (2*Nfine x 2*Ncoarse) from the coarse Q_p velocity
space to the fine one.  Every fine GLL node lies in exactly one coarse leaf
(the coarse tree is a coarsening of the fine one), so evaluating the coarse
tensor-product basis there is exact.  Both velocity components share the same
scalar matrix.

Cut 5 dispatches on the geometry: the box body is the historical expression
(physical fine coordinates, physical→reference inversion of the coarse leaf,
bitwise unchanged); a mapped geometry uses **logical** coordinates
(GEOM_DESIGN §2.1 rule 4 / §6.5): the fine node's global logical position comes
from `global_logical` (leaf index + GLL index, never a physical position), the
coarse leaf is located by the O(maxlevel) logical descent, and the coarse
reference abscissa from `leaf_ref`.  The weights are therefore the same 1-D GLL
interpolation weights on every geometry — the map is a global diffeomorphism, so
the coarse/fine logical nesting is unchanged — which is what keeps the coarse
spaces nested and the V-cycle correct on a curved mesh.
"""
function _mg_prolongation_full(fine::SemQuad, coarse::SemQuad, p::Int)
    return fine.geom isa CartesianBox ?
           _mg_prolongation_full_box(fine, coarse, p) :
           _mg_prolongation_full_mapped(fine.geom, fine, coarse, p)
end

# Historical box body, verbatim.
function _mg_prolongation_full_box(fine::SemQuad, coarse::SemQuad, p::Int)
    n = p + 1
    rule = gll_rule(p)
    leaves_c = Set(coarse.leaves)
    Nf = fine.N
    Nc = coarse.N
    I = Int[]; J = Int[]; V = Float64[]
    sizehint!(I, 4 * Nf * n * n)
    sizehint!(J, 4 * Nf * n * n)
    sizehint!(V, 4 * Nf * n * n)
    for g in 1:Nf
        x, y = fine.coords[g]
        lf = _semquad_leaf_at(coarse, x, y, leaves_c)
        hx = leaf_hx(coarse, lf[1]); hy = leaf_hy(coarse, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        xi = 2.0 * (x - x0) / hx - 1.0
        eta = 2.0 * (y - y0) / hy - 1.0
        Lx, _ = lagrange_eval(rule, xi)
        Ly, _ = lagrange_eval(rule, eta)
        cids = coarse.elem[lf]
        for j in 1:n, i in 1:n
            w = Lx[i] * Ly[j]
            w == 0.0 && continue
            c = cids[(j - 1) * n + i]
            push!(I, g); push!(J, c); push!(V, w)
            push!(I, Nf + g); push!(J, Nc + c); push!(V, w)
        end
    end
    return sparse(I, J, V, 2 * Nf, 2 * Nc)
end

# Mapped body: logical positions only (no physical coordinate is read, so a
# curved face is located by construction, GEOM_DESIGN §6.5).  A node shared by
# several fine leaves must contribute its row **once**: the box body iterates the
# global node list, so it has no duplicates; this body iterates leaf-local nodes,
# so it masks already-emitted rows (a duplicate row is summed by `sparse`, which
# would multiply the interpolation weight of every shared node — the first
# implementation did exactly that and the V-cycle diverged, G271 (d)).
function _mg_prolongation_full_mapped(g::AbstractGeometry, fine::SemQuad,
                                      coarse::SemQuad, p::Int)
    n = p + 1
    rule = gll_rule(p)
    leaves_c = Set(coarse.leaves)
    Nf = fine.N
    Nc = coarse.N
    I = Int[]; J = Int[]; V = Float64[]
    sizehint!(I, 4 * Nf * n * n)
    sizehint!(J, 4 * Nf * n * n)
    sizehint!(V, 4 * Nf * n * n)
    seen = falses(Nf)
    for it in eachindex(fine.leaves)
        lf = fine.leaves[it]
        uids = fine.elem[lf]
        for j in 1:n, i in 1:n
            gid = uids[(j - 1) * n + i]
            seen[gid] && continue
            seen[gid] = true
            S, T = global_logical(fine, lf, rule.nodes[i], rule.nodes[j])
            lc = locate_leaf_logical(coarse, S, T, leaves_c)
            xi, eta = leaf_ref(coarse, lc, S, T)
            Lx, _ = lagrange_eval(rule, xi)
            Ly, _ = lagrange_eval(rule, eta)
            cids = coarse.elem[lc]
            for jj in 1:n, ii in 1:n
                w = Lx[ii] * Ly[jj]
                w == 0.0 && continue
                c = cids[(jj - 1) * n + ii]
                push!(I, gid); push!(J, c); push!(V, w)
                push!(I, Nf + gid); push!(J, Nc + c); push!(V, w)
            end
        end
    end
    return sparse(I, J, V, 2 * Nf, 2 * Nc)
end

