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
# serial.  Same policy as `_semquad_asm_thread_count` (docs/S3_DESIGN.md §12).
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
    mfops::Vector{Any}
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
  docs/S3_DESIGN.md §14 for the measurement.
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
"""
function _mg_prolongation_full(fine::SemQuad, coarse::SemQuad, p::Int)
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

# ---------------------------------------------------------------------------
# G221 -- per-level hierarchy cache: reuse the levels whose leaf set did not
# change across an AMR cycle (docs/dev/MG.md)
# ---------------------------------------------------------------------------
# `make_semquad_mg` rebuilds the whole hierarchy whenever the tree changes, and
# one production AMR cycle changes ~1-2 % of the leaves.  The coarse levels are
# a *coarsening of the leaf set*, so most of them are identical from one cycle
# to the next; this cache keeps them.
#
# What the cache stores is a level's **topology-only** state: the coarse
# `Quadtree`, its `SemStokes` space, its full->free constraint `Cu`, its
# free-row list and the prolongation to the next finer level.  Every one of
# those is a deterministic function of the level's own leaf set (plus
# `p`/BC/`Lx`/base cells, all in the key), so a cache hit and a from-scratch
# rebuild produce the same numbers: a hit is bitwise (G221 (c)).
#
# What it deliberately does **not** store -- the level's operator data:
#   * `mfetas[k]`: the viscosity field sampled at that level's Gauss points,
#     in both layouts (`SemQuadEtaLane`, docs/dev/MG.md);
#   * `D[k] = diag(Cu' K(eta) Cu)`: the exact Jacobi diagonal;
#   * `cA`: the coarsest-level dense operator / factorisation.
# All three are functions of the eta field sampled at *build* time, and the
# production solve samples the current viscosity on every build (that is the
# shipped policy: the geometry is reused per tree, the operator is
# rediscretised -- docs/dev/MG.md).  Caching them would make a reused
# level depend on the eta of the cycle that built it, so a solve after an AMR
# cycle would stop being bitwise the solve of a forced from-scratch build
# (G221 (c)).  They are recomputed on every build, exactly as before.  The
# level's leaf-local operator data (`semquad_kop`) *is* reused bitwise, through
# its own tree-keyed cache (`SEMQUAD_KOP_CACHE`), which is keyed by the same
# leaf-set fingerprint.
#
# The entries live in `SEMQUAD_MG_CACHE` (the per-tree geometry cache) under
# `SemQuadMGLevelKey`s, so `empty!(SEMQUAD_MG_CACHE)` still means "forget every
# hierarchy"; the AMR cycle instead calls `_semquad_mg_drop_tree_entries!`,
# which drops only the per-tree entries and keeps the levels.

# `false` = the historical behaviour (rebuild every level on every build), used
# by the A/B and by the gates' forced-rebuild reference (G221 (c)).
const SEMQUAD_MG_LEVEL_CACHE = Ref(true)

"""
    SemQuadMGLevelKey(idx, nleaf, stamp, p, nx0, ny0, Lx, bc, deviatoric)

Cache key of one coarse MG level: the level index (coarse first) plus the
fingerprint of the level's **own** leaf set (`_semquad_mg_leaf_hash`) and the
settings that change the level objects for the same leaf set.  Two levels with
the same leaves and the same settings always get the same key, and the key does
not depend on anything sampled at build time (eta) -- see the block comment
above.
"""
struct SemQuadMGLevelKey
    idx::Int
    nleaf::Int
    stamp::UInt64
    p::Int
    nx0::Int
    ny0::Int
    Lx::Float64
    bc::Symbol
    deviatoric::Bool
end

Base.:(==)(a::SemQuadMGLevelKey, b::SemQuadMGLevelKey) =
    a.idx == b.idx && a.nleaf == b.nleaf && a.stamp == b.stamp &&
    a.p == b.p && a.nx0 == b.nx0 && a.ny0 == b.ny0 && a.Lx == b.Lx &&
    a.bc === b.bc && a.deviatoric == b.deviatoric
Base.hash(k::SemQuadMGLevelKey, h::UInt) =
    hash(k.stamp, hash(k.idx, hash(k.nleaf, hash(k.p, hash(k.bc,
                                                        hash(k.Lx, h))))))

function Base.show(io::IO, k::SemQuadMGLevelKey)
    print(io, "SemQuadMGLevelKey(level=", k.idx, ", leaves=", k.nleaf,
          ", stamp=", string(k.stamp; base=16), ")")
end

"""
    SemQuadMGLevel

One coarse level of a hierarchy: the cached, topology-only state.  `P` is the
prolongation from this level to level `idx + 1`; `pstamp` records which pair of
level fingerprints it was built from, so a level that survives while its finer
neighbour changes rebuilds only the transfer.
"""
mutable struct SemQuadMGLevel
    idx::Int
    nleaf::Int
    stamp::UInt64
    qt::Quadtree
    ss::SemStokes
    Cu::SparseMatrixCSC{Float64,Int}
    nf::Int
    free_rows::Vector{Int}
    P::Union{Nothing,SparseMatrixCSC{Float64,Int}}
    pstamp::Tuple{Int,UInt64,Int,UInt64}
end

"""
    SemQuadMGLevelStats

Counters of the per-level cache: builds (hierarchy builds), hits, misses,
prolongations built, the level indices rebuilt/reused in the *last* build, and
the wall time of the last/whole set of `make_semquad_mg` calls (two clock reads
per build; the benchmark in `julia/scripts/bench_mg_level_cache.jl` reads them).
"""
mutable struct SemQuadMGLevelStats
    builds::Int
    hits::Int
    misses::Int
    prolongations::Int
    last_rebuilt::Vector{Int}
    last_reused::Vector{Int}
    last_build_s::Float64
    total_build_s::Float64
end

const SEMQUAD_MG_LEVEL_STATS = SemQuadMGLevelStats(0, 0, 0, 0, Int[], Int[],
                                                   0.0, 0.0)

"""Reset the per-level cache counters (`SEMQUAD_MG_LEVEL_STATS`)."""
function reset_semquad_mg_level_stats!()
    s = SEMQUAD_MG_LEVEL_STATS
    s.builds = 0
    s.hits = 0
    s.misses = 0
    s.prolongations = 0
    empty!(s.last_rebuilt)
    empty!(s.last_reused)
    s.last_build_s = 0.0
    s.total_build_s = 0.0
    return s
end

"""
    _semquad_mg_leaf_hash(leaves) -> UInt64

Deterministic 64-bit fingerprint of a leaf set: the XOR of a splitmix64 finalised
mix of every `(level, ix, iy)` triple.  XOR is order independent, so this is the
hash of the *sorted* leaf set without sorting it, it never depends on Julia's
`hash` seed or on `objectid`, and equal leaf sets always give equal values --
inside one process and across processes (G221 (a)).  The leaf count is carried
separately in the cache key.
"""
function _semquad_mg_leaf_hash(leaves)::UInt64
    h = UInt64(0)
    @inbounds for lf in leaves
        z = (UInt64(lf[1]) << 42) ⊻ (UInt64(lf[2]) << 21) ⊻ UInt64(lf[3])
        z += 0x9e3779b97f4a7c15
        z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
        z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
        h ⊻= (z ⊻ (z >> 31))
    end
    return h
end

"""Number of per-level entries currently held by `SEMQUAD_MG_CACHE`."""
function _semquad_mg_level_count()
    n = 0
    for k in keys(SEMQUAD_MG_CACHE)
        k isa SemQuadMGLevelKey && (n += 1)
    end
    return n
end

"""Approximate bytes held by the per-level cache (`Base.summarysize` of the
cached level objects, including their trees, spaces, `Cu` and transfers)."""
function semquad_mg_level_cache_bytes()
    tot = 0
    for (k, v) in SEMQUAD_MG_CACHE
        k isa SemQuadMGLevelKey && (tot += Base.summarysize(v))
    end
    return tot
end

"""
    semquad_mg_level_cache_stats()

Snapshot of the per-level cache for the docs/probes: cumulative `builds`,
`hits`, `misses` and `prolongations`, the level indices rebuilt/reused in the
last build, the number of cached levels and their memory footprint.
"""
function semquad_mg_level_cache_stats()
    s = SEMQUAD_MG_LEVEL_STATS
    return (builds=s.builds, hits=s.hits, misses=s.misses,
            prolongations=s.prolongations,
            cached_levels=_semquad_mg_level_count(),
            bytes=semquad_mg_level_cache_bytes(),
            last_build_s=s.last_build_s, total_build_s=s.total_build_s,
            last_rebuilt=copy(s.last_rebuilt),
            last_reused=copy(s.last_reused))
end

"""Insert `lv` under `k`, evicting every older topology of the same level index
(one live entry per level: an AMR trajectory never needs more, and the cache
stays bounded by the hierarchy depth)."""
function _semquad_mg_level_cache_put!(k::SemQuadMGLevelKey, lv::SemQuadMGLevel)
    for kk in collect(keys(SEMQUAD_MG_CACHE))
        kk isa SemQuadMGLevelKey || continue
        kk.idx == k.idx && !(kk == k) && delete!(SEMQUAD_MG_CACHE, kk)
    end
    SEMQUAD_MG_CACHE[k] = lv
    return lv
end

"""Drop the per-*tree* hierarchy entries of `SEMQUAD_MG_CACHE` (the
`SemQuadMGGeom`s) and keep the per-level ones.  Called by the driver after an
AMR cycle, where the old code called `empty!`: the tree changed, so the
per-tree geometry is stale, but a level whose own leaf set survived is still
exactly the level a from-scratch build would produce."""
function _semquad_mg_drop_tree_entries!()
    for k in collect(keys(SEMQUAD_MG_CACHE))
        k isa SemQuadMGLevelKey && continue
        delete!(SEMQUAD_MG_CACHE, k)
    end
    return nothing
end

"""`Cu` of one coarse level, via the reduced (`:mg`) or the historical
K-forming (`:full`) assembly (see `make_semquad_mg`'s `geom_assembly`)."""
function _semquad_mg_level_Cu(q::Quadtree, s::SemStokes, bc::Symbol,
                              deviatoric::Bool, geom_assembly::Symbol)
    z2 = (x, y) -> 0.0
    one1 = (x, y) -> 1.0
    if geom_assembly === :minimal
        # The hierarchy reads `Cu` only.  `assemble_only=:mg` is the reduced
        # matrix-free assembly: it forms no velocity stiffness `K` (or `Kred`)
        # and does not evaluate the viscosity closure (which only enters `K`).
        # `Cu` is a function of the tree, the BCs and the affine Dirichlet data,
        # so this is bitwise the full assembly's `Cu` (G220).
        asm = solve_semquad_stokes_var(s, q, one1, z2, z2, z2, z2;
                                       bc=bc, deviatoric=deviatoric,
                                       linear=:mg, mg_operator=:matrixfree,
                                       assemble_only=:mg)
    else
        # `:full` is the historical K-forming assembly; it exists for the A/B
        # and the bitwise gate, not for production.
        asm = solve_semquad_stokes_var(s, q, one1, z2, z2, z2, z2;
                                       bc=bc, deviatoric=deviatoric,
                                       assemble_only=true)
    end
    return SparseMatrixCSC{Float64,Int}(sparse(asm.Cu))
end

function _semquad_mg_geometry(ss::SemStokes, qt::Quadtree, p::Int,
                              bc::Symbol, deviatoric::Bool;
                              coarsen_layers::Int=1,
                              coarsen_factor::Float64=1.0,
                              coarse_direct_max::Int=0,
                              fine_Cu::Union{Nothing,SparseMatrixCSC{Float64,Int}}=nothing,
                              fine_ss::Union{Nothing,SemStokes}=nothing,
                              geom_assembly::Symbol=:minimal)
    coarse_direct_max >= 0 ||
        throw(ArgumentError("MG coarse_direct_max must be >= 0"))
    geom_assembly in (:minimal, :full) ||
        throw(ArgumentError("MG geom_assembly must be :minimal or :full"))
    key = (p, qt.nx0, qt.ny0, qt.Lx, bc, deviatoric,
           length(qt.leaves), hash(qt.leaves),
           coarsen_layers, coarsen_factor, coarse_direct_max, geom_assembly)
    if haskey(SEMQUAD_MG_CACHE, key)
        cached = SEMQUAD_MG_CACHE[key]::SemQuadMGGeom
        Set(cached.qts[end].leaves) == Set(qt.leaves) && return cached
    end
    reuse = SEMQUAD_MG_LEVEL_CACHE[]
    stats = SEMQUAD_MG_LEVEL_STATS
    stats.builds += 1
    empty!(stats.last_rebuilt)
    empty!(stats.last_reused)
    trees = _mg_hierarchy_trees(qt; layers=coarsen_layers,
                                factor=coarsen_factor)
    nlev = length(trees)
    stamps = UInt64[_semquad_mg_leaf_hash(q.leaves) for q in trees]
    levels = Vector{SemQuadMGLevel}(undef, nlev)
    for idx in 1:nlev
        q = trees[idx]
        # The finest level is the caller's tree: its constraint matrix `Cu` is
        # already available from the caller's assembly and does not depend on
        # eta, so re-assembling it here with eta=1 is pure duplicate work
        # (0.17-0.41 s of the hierarchy build on the production tree).  It is
        # **never cached**: the caller supplies the tree (the driver mutates it
        # in place on the next AMR cycle) and its `Cu` on every build.  Every
        # *coarse* level is a private tree built by `_mg_coarsen_tree`, so it is
        # cacheable.  The fallback keeps standalone/reference callers bitwise
        # unchanged.
        cacheable = idx < nlev
        nleaf = length(q.leaves)
        lv = nothing
        if cacheable && reuse
            lkey = SemQuadMGLevelKey(idx, nleaf, stamps[idx], p, q.nx0, q.ny0,
                                     q.Lx, bc, deviatoric)
            got = get(SEMQUAD_MG_CACHE, lkey, nothing)
            if got isa SemQuadMGLevel
                lv = got
                stats.hits += 1
                push!(stats.last_reused, idx)
            end
        end
        if lv === nothing
            fine = idx == nlev && fine_Cu !== nothing
            s = fine && fine_ss !== nothing ? fine_ss : make_semstokes(q, p)
            Cu = if fine
                size(fine_Cu, 1) == 2 * s.su.N ||
                    throw(ArgumentError("fine_Cu has $(size(fine_Cu, 1)) rows, " *
                                        "expected $(2 * s.su.N); it does not belong " *
                                        "to the finest MG tree"))
                fine_Cu
            else
                _semquad_mg_level_Cu(q, s, bc, deviatoric, geom_assembly)
            end
            lv = SemQuadMGLevel(idx, nleaf, stamps[idx], q, s, Cu,
                                size(Cu, 2), _mg_free_rows(Cu, 2 * s.su.N),
                                nothing, (0, UInt64(0), 0, UInt64(0)))
            if cacheable
                stats.misses += 1
                push!(stats.last_rebuilt, idx)
                reuse && _semquad_mg_level_cache_put!(
                    SemQuadMGLevelKey(idx, nleaf, stamps[idx], p, q.nx0, q.ny0,
                                      q.Lx, bc, deviatoric), lv)
            end
        end
        levels[idx] = lv
    end
    # S3 stage 1 (b): stop the V-cycle at the level where a direct factorisation
    # is cheap enough, instead of coarsening further.  `nf` is ascending (coarse
    # first), so this keeps the *finer* part of the hierarchy and factorises its
    # new coarsest level; the finest level is never cut away (that would leave
    # no coarse-grid correction at all).
    coarse_direct = false
    kcut = 1
    if coarse_direct_max > 0 && nlev > 2
        for k in 1:(nlev - 1)
            levels[k].nf <= coarse_direct_max || break
            kcut = k
        end
        kcut > 1 && (coarse_direct = true)
    end
    kept = kcut:nlev
    L = length(kept) - 1
    Ps = SparseMatrixCSC{Float64,Int}[]
    for k in kcut:(nlev - 1)
        lv = levels[k]
        f = levels[k + 1]
        pstamp = (lv.nleaf, lv.stamp, f.nleaf, f.stamp)
        if reuse && lv.P !== nothing && lv.pstamp == pstamp
            push!(Ps, lv.P)
        else
            Pfull = _mg_prolongation_full(f.ss.su, lv.ss.su, p)
            P = Pfull[f.free_rows, :] * lv.Cu
            size(P, 1) == f.nf && size(P, 2) == lv.nf ||
                error("MG prolongation has the wrong size: $(size(P)) for " *
                      "$(f.nf) x $(lv.nf)")
            P = SparseMatrixCSC{Float64,Int}(sparse(P))
            push!(Ps, P)
            lv.P = P
            lv.pstamp = pstamp
            stats.prolongations += 1
        end
    end
    qts = Quadtree[levels[k].qt for k in kept]
    sss = SemStokes[levels[k].ss for k in kept]
    Cus = SparseMatrixCSC{Float64,Int}[levels[k].Cu for k in kept]
    # Transposed constraint matrices: `Cu * v` is a row gather over `Cu'`, the
    # split form of the first sparse product (`_semquad_spmv_region!`).  The
    # transpose is a pure re-indexing (no arithmetic) and costs <1 ms per level
    # (2.4 MB in total on the production F hierarchy), so it is part of the
    # geometry instead of the threaded-only state: every `SemQuadMG` -- including
    # the ones the gates build by hand -- shares it.
    Cuts = SparseMatrixCSC{Float64,Int}[SparseMatrixCSC{Float64,Int}(
                                          sparse(transpose(Cu))) for Cu in Cus]
    nfs = Int[levels[k].nf for k in kept]
    free_rows = Vector{Int}[levels[k].free_rows for k in kept]
    geo = SemQuadMGGeom(p, L, qt.Lx, qt.nx0, qt.ny0, bc, deviatoric,
                        qts, sss, Cus, Cuts, Ps, nfs, free_rows, coarse_direct)
    SEMQUAD_MG_CACHE[key] = geo
    return geo
end

"""
    make_semquad_mg(ss, qt, p, bc, deviatoric, eta_fun, Kred; cycles=1,
                    smoother=:jacobi, coarse_operator=:rediscretize,
                    operator=:sparse, coarsen_layers=1, coarsen_factor=1.0,
                    coarse_direct_max=0, pre=2, post=2, smoothing=:uniform,
                    omega=2/3)

Build the MG hierarchy for the Stokes velocity block on `qt`, uniform or not.
The fine level reuses the caller's `Kred`; the coarse levels are re-assembled
with the same `eta_fun` (rediscretisation, standard for smooth variable
viscosity; hanging constraints are reassembled on every coarse tree).

`coarsen_layers`/`coarsen_factor` control the hierarchy (see
`_mg_coarsen_tree`): `1/1.0` is the historical one-tree-layer-per-step rule,
`coarsen_factor=1.5` (the production matrix-free default, resolved by
`solve_semquad_stokes_var`) removes a near-duplicate finest step while keeping a
7-level P1 hierarchy, `2.0` merges more aggressively (F-only win; P1 regresses),
and `coarse_direct_max=n>0` stops the V-cycle at the finest level with at most
`n` dofs and solves that level with a sparse Cholesky instead of coarsening
further (`geo.coarse_direct`; the matrix-free path assembles that one level
sparsely, because a dense `O(nf^2)` materialisation would cost far more than it
saves).

`pre`/`post` are the Jacobi sweeps per level (an `Int`, or one entry per level
**coarse first**); `smoothing=:auto` (default) uses uniform 2+2, raised to 3+3
when the hierarchy is top-heavy (its finest step keeps >50 % of the fine dofs,
the measured winner on P1), and `omega` (damping; `nothing` = the schedule's
default) overrides the damping factor.  `smoothing=:taper` remains selectable
(`1, 2, 2, 4, 4, 8, ...` from the fine level down, omega=0.55); explicit
`pre`/`post` vectors are accepted with any schedule.

`fine_Cu` (optional) is the caller's full->free constraint matrix of the finest
tree.  `_semquad_mg_geometry` otherwise re-assembles the finest level with eta=1
only to read that matrix; passing it removes the duplicate assembly (the
0.17-0.41 s finest row of docs/dev/MG.md) and is bitwise inert (G219).

`fine_mfop`/`fine_mfeta` (optional, `operator=:matrixfree` only) are the
caller's finest-level leaf-local operator (`semquad_kop(ss, qt)`) and its
viscosity field (`semquad_eta_field`).  The solve builds both before it asks for
the hierarchy, so without them `make_semquad_mg` resamples the same 131 008
quadrature points a second time (0.17 s of the F build at one thread); passing
them is bitwise inert (G220) and only removes the duplicate sampling.

Every level's viscosity field is stored as a `SemQuadEtaLane`: the leaf-major
sample plus the block-major lane table the leaf-blocked kernel reads
(docs/dev/MG.md).  The table is built here, once per level, never per
apply; it is a re-layout, so it changes no value (G225).

`geom_assembly=:minimal` (default) builds the coarse levels' `Cu` with the
reduced matrix-free assembly (`assemble_only=:mg`): the hierarchy reads only
`Cu`, so no velocity stiffness `K`, no `Kred` and no viscosity evaluation are
needed there.  `:full` keeps the historical `K`-forming assembly (0.039 s of the
F build at one thread) as the A/B and bitwise-gate reference; the two produce the
same `Cu` and therefore the same hierarchy, V-cycle and solve (G220).

`operator=:matrixfree` (S3/MF-2/MF-3) replaces every level's sparse matvec by
the matrix-free leaf-local apply `semquad_apply_Kred!` of MF-1, and stores no
velocity operator matrix at all: `A` is empty, the Jacobi diagonal comes from
the leaf-local element diagonals, and the coarsest solve comes from a dense
matrix materialised by applying the matrix-free operator to the unit vectors.
The hierarchy then keeps only the sparse `Cu`/`P` transfer operators.  The
viscosity is sampled once per level at the assembler's Gauss points, so each
matrix-free level operator is the same operator as the rediscretised sparse one
(G175) and the smoother is unchanged on conforming trees.  `Kred` is accepted
and ignored in that mode (it is the caller's outer-Krylov operator).
`operator=:matrixfree` always coarsens by rediscretisation.

`mg_diag` selects the matrix-free Jacobi diagonal: `:constrained` (default) is
the exact `diag(Cu' K Cu)`, computed leaf-locally without forming `K`;
`:lumped` is the historical `Cu' diag(K)` (only for A/B -- on hanging rows the
negative interpolation weights cancel, so it is not the reduced diagonal and its
positivity clip can make a row a `1e12` smoother amplifier; see
docs/MG_REALISTIC_DIAGNOSIS.md §5.2).  The sparse operator path always uses
`diag(A[k])` and is unaffected.

`mg_threads=1` (default) is the serial, bitwise V-cycle.  `mg_threads=n>1`
(S5-2) threads the leaf-local work of every level -- the matrix-free apply, the
damped-Jacobi sweeps and the residual evaluation -- over `n` fixed contiguous
leaf blocks (`Threads.@threads :static`), with one `SemQuadKScratch` per block
and a per-leaf element buffer merged **serially in leaf order**, so the result
is bitwise identical to the serial V-cycle (G190; measured 2.5-3.1x V-cycle
speedup on 4096-leaf trees, docs/S3_DESIGN.md §19.2).  Only the matrix-free
operator path is threaded (the sparse `A[k]` path keeps its serial `mul!`); a
request larger than `Threads.nthreads()` is clamped with a warning, and a
single-threaded process warns once and stays serial.
"""
function make_semquad_mg(ss::SemStokes, qt::Quadtree, p::Int, bc::Symbol,
                         deviatoric::Bool, eta_fun::Function,
                         Kred::Union{Nothing,AbstractMatrix}; cycles::Int=1,
                         smoother::Symbol=:jacobi,
                         coarse_operator::Symbol=:rediscretize,
                         operator::Symbol=:sparse,
                         coarsen_layers::Int=1,
                         coarsen_factor::Float64=1.0,
                         coarse_direct_max::Int=0,
                         pre::Union{Integer,AbstractVector{<:Integer}}=2,
                         post::Union{Integer,AbstractVector{<:Integer}}=2,
                         smoothing::Symbol=:auto,
                         omega::Union{Nothing,Float64}=nothing,
                         mg_threads::Int=1,
                         mg_diag::Symbol=:constrained,
                         fine_Cu::Union{Nothing,SparseMatrixCSC{Float64,Int}}=nothing,
                         geom_assembly::Symbol=:minimal,
                         fine_mfop::Union{Nothing,SemQuadKOp}=nothing,
                         fine_mfeta::Union{Nothing,AbstractVector{Float64}}=nothing)
    smoother in (:jacobi, :sgs) ||
        throw(ArgumentError("MG smoother must be :jacobi or :sgs"))
    coarse_operator in (:rediscretize, :galerkin) ||
        throw(ArgumentError("MG coarse_operator must be :rediscretize or :galerkin"))
    operator in (:sparse, :matrixfree) ||
        throw(ArgumentError("MG operator must be :sparse or :matrixfree"))
    coarsen_layers >= 1 ||
        throw(ArgumentError("MG coarsen_layers must be >= 1"))
    (coarsen_factor >= 1.0 && isfinite(coarsen_factor)) ||
        throw(ArgumentError("MG coarsen_factor must be >= 1"))
    coarse_direct_max >= 0 ||
        throw(ArgumentError("MG coarse_direct_max must be >= 0"))
    smoothing in (:auto, :uniform, :taper) ||
        throw(ArgumentError("MG smoothing must be :auto, :uniform or :taper"))
    mg_threads >= 1 || throw(ArgumentError("mg_threads must be >= 1"))
    mg_diag in (:constrained, :lumped) ||
        throw(ArgumentError("mg_diag must be :constrained or :lumped"))
    if operator === :matrixfree
        smoother === :sgs &&
            throw(ArgumentError("the :sgs smoother needs operator=:sparse"))
        coarse_operator === :galerkin &&
            @warn "operator=:matrixfree coarsens by rediscretisation; " *
                  "ignoring coarse_operator=:galerkin"
        coarse_operator = :rediscretize
    end
    geom_assembly in (:minimal, :full) ||
        throw(ArgumentError("MG geom_assembly must be :minimal or :full"))
    operator === :sparse && (fine_mfop !== nothing || fine_mfeta !== nothing) &&
        throw(ArgumentError("fine_mfop/fine_mfeta need operator=:matrixfree"))
    _t_build0 = time_ns()
    geo = _semquad_mg_geometry(ss, qt, p, bc, deviatoric;
                               coarsen_layers=coarsen_layers,
                               coarsen_factor=coarsen_factor,
                               coarse_direct_max=coarse_direct_max,
                               fine_Cu=fine_Cu, fine_ss=ss,
                               geom_assembly=geom_assembly)
    L = geo.L
    mfops = Any[]
    mfetas = SemQuadEtaLane[]
    mfwork = Any[]
    A = Vector{SparseMatrixCSC{Float64,Int}}()
    D = Vector{Vector{Float64}}()
    cA = nothing
    if operator === :matrixfree
        # S3/MF-3: the hierarchy stores *no* velocity operator matrix.  Every
        # level applies MF-1's leaf-local operator; the Jacobi diagonal is the
        # *constrained* diagonal `diag(Cu' K Cu)` (`mg_diag=:constrained`, the
        # default; `:lumped` keeps the old `Cu' diag(K)` for A/B), and the
        # coarsest solve comes from a dense matrix materialised by applying the
        # matrix-free operator to the unit vectors
        # (`_semquad_dense_operator!`), which is O(nf^2) applies on the base
        # leaves only.
        for k in 1:(L + 1)
            fine_k = k == L + 1
            if fine_k && fine_mfop !== nothing
                # The caller (the solve) has already built the finest level's
                # leaf-local operator; `semquad_kop` is cached on the tree, so
                # this is the same object.  Rebuilding it here would be cheap
                # but recomputing its eta field is not: the caller's
                # `semquad_eta_field` sampled the identical 131 008 points and
                # the hierarchy used to do it a second time (0.17 s of the F
                # build at one thread, docs/dev/MG.md).
                opk = fine_mfop
                (opk.N == geo.sss[k].su.N && opk.p == geo.sss[k].su.p &&
                 opk.q == geo.sss[k].sp.p && opk.deviatoric == deviatoric &&
                 opk.nleaf == length(geo.qts[k].leaves)) ||
                    throw(ArgumentError("fine_mfop does not belong to the " *
                                        "finest MG tree (N=$(opk.N), p=$(opk.p), " *
                                        "nleaf=$(opk.nleaf)); it must be " *
                                        "semquad_kop(ss, qt)"))
            else
                opk = semquad_kop(geo.sss[k], geo.qts[k]; deviatoric=deviatoric)
            end
            push!(mfops, opk)
            if fine_k && fine_mfeta !== nothing
                length(fine_mfeta) == length(opk.xq) ||
                    throw(ArgumentError("fine_mfeta has $(length(fine_mfeta)) " *
                                        "values, expected one per leaf/quadrature " *
                                        "point ($(length(opk.xq)))"))
                push!(mfetas, semquad_eta_lanes(opk, fine_mfeta))
            else
                push!(mfetas, semquad_eta_lanes(opk,
                                                semquad_eta_field(opk, eta_fun)))
            end
            push!(mfwork, SemQuadKWork(opk.n, opk.nq, 2 * opk.N))
        end
        for k in 1:(L + 1)
            push!(D, _semquad_diag_reduced(mfops[k]::SemQuadKOp, geo.Cus[k],
                                           mfetas[k]::SemQuadEtaLane;
                                           mode=mg_diag))
        end
        if geo.coarse_direct
            # Stage-1 (b) cut: the coarsest level is a real coarse mesh, not the
            # base leaves, so materialising a dense operator would cost O(nf^2)
            # applies; assemble that level's sparse rediscretised operator and
            # factorise it instead.
            asm1 = solve_semquad_stokes_var(geo.sss[1], geo.qts[1], eta_fun,
                                            (x, y) -> 0.0, (x, y) -> 0.0,
                                            (x, y) -> 0.0, (x, y) -> 0.0;
                                            bc=bc, deviatoric=deviatoric,
                                            assemble_only=true)
            A1 = SparseMatrixCSC{Float64,Int}(sparse(asm1.Kred))
            cA = cholesky(Symmetric(A1); check=false)
            issuccess(cA) ||
                throw(ArgumentError("coarsest matrix-free velocity operator is not SPD"))
        else
            n1 = geo.nfs[1]
            A1 = Matrix{Float64}(undef, n1, n1)
            _semquad_dense_operator!(A1, mfops[1]::SemQuadKOp, geo.Cus[1],
                                     mfetas[1]::SemQuadEtaLane,
                                     mfwork[1]::SemQuadKWork)
            cA = cholesky(Symmetric(A1); check=false)
            issuccess(cA) ||
                throw(ArgumentError("coarsest matrix-free velocity operator is not SPD"))
        end
    else
        A = Vector{SparseMatrixCSC{Float64,Int}}(undef, L + 1)
        A[L + 1] = SparseMatrixCSC{Float64,Int}(sparse(Kred))
        if coarse_operator === :galerkin
            # P'AP from the fine operator; exact Galerkin coarsening.  For
            # constant eta it equals the rediscretised operator (to round-off);
            # for strong variable viscosity it can be better conditioned.
            for k in L:-1:1
                A[k] = SparseMatrixCSC{Float64,Int}(sparse(geo.Ps[k]' *
                                                           A[k + 1] * geo.Ps[k]))
            end
        else
            for k in 1:L
                asm = solve_semquad_stokes_var(geo.sss[k], geo.qts[k], eta_fun,
                                               (x, y) -> 0.0, (x, y) -> 0.0,
                                               (x, y) -> 0.0, (x, y) -> 0.0;
                                               bc=bc, deviatoric=deviatoric,
                                               assemble_only=true)
                A[k] = SparseMatrixCSC{Float64,Int}(sparse(asm.Kred))
            end
        end
        for k in 1:L
            size(A[k], 1) == geo.nfs[k] ||
                error("coarse MG operator has $(size(A[k],1)) rows, expected $(geo.nfs[k])")
        end
        size(A[L + 1], 1) == geo.nfs[L + 1] ||
            error("fine MG operator has $(size(A[L + 1],1)) rows, expected $(geo.nfs[L + 1])")
        D = [Vector{Float64}(diag(A[k])) for k in 1:(L + 1)]
        cA = cholesky(Symmetric(A[1]))
    end
    any(x -> x <= 0.0, D[L + 1]) &&
        throw(ArgumentError("velocity block has a non-positive diagonal; " *
                            "MG requires an SPD reduced operator"))
    LF = Any[]; UF = Any[]
    if smoother === :sgs
        for k in 1:(L + 1)
            push!(LF, LowerTriangular(A[k]))
            push!(UF, UpperTriangular(A[k]))
        end
    end
    sc = [zeros(Float64, geo.nfs[k]) for k in 1:(L + 1)]
    sr = [zeros(Float64, geo.nfs[k]) for k in 1:(L + 1)]
    sy = [zeros(Float64, geo.nfs[k]) for k in 1:(L + 1)]
    sched, om, auto_heavy = _mg_smoothing_profile(smoothing, geo, omega)
    LAST_SEMSTOKES_MG_PROFILE[] = (sched, om)
    pre_v, post_v = if sched === :uniform || pre isa AbstractVector ||
                       post isa AbstractVector
        # Explicit per-level vectors are accepted with any resolved schedule
        # (they are the `:taper` measurement knob: `smoothing=:taper,
        # pre=[...], post=[...]` keeps omega=0.55 and uses the given shape).
        # A top-heavy `:auto` hierarchy raises the *default* 2+2 to 3+3; an
        # explicit non-default sweep count is respected.
        p = (smoothing === :auto && auto_heavy && pre isa Integer && pre == 2) ?
            3 : pre
        q = (smoothing === :auto && auto_heavy && post isa Integer && post == 2) ?
            3 : post
        (_mg_sweeps(p, L + 1, "pre"), _mg_sweeps(q, L + 1, "post"))
    else
        _mg_smoothing_schedule(sched, L + 1, Int(pre), Int(post))
    end
    # S5-2: one SemQuadKPar per matrix-free level.  The sparse operator path
    # keeps its serial `mul!` (`par` is empty), so `mg_threads` only ever
    # changes the matrix-free hierarchy.
    nt = _semquad_mg_thread_count(mg_threads)
    par = Any[]
    Pts = Any[]
    if nt > 1 && operator === :matrixfree
        par = Any[SemQuadKPar(mfops[k]::SemQuadKOp, nt) for k in 1:(L + 1)]
        # Transposed transfer operators: the prolongation is a row gather over
        # `P'`'s CSC (`_semquad_mg_prolong!`), which is bitwise the `mul!`
        # scatter but parallelisable over rows.
        Pts = Any[SparseMatrixCSC{Float64,Int}(sparse(transpose(geo.Ps[k])))
                  for k in 1:L]
    elseif nt > 1
        nt = 1
    end
    # Per-level V-cycle scratch (restriction output, coarse error, prolongation
    # output).  Fully overwritten on every use, so this is bitwise the
    # allocating `P' * r` / `zeros` / `P * ec` form (and takes the per-V-cycle
    # allocations, ~2 MB at L6, out of the GC).
    rcs = Vector{Vector{Float64}}(undef, L + 1)
    ecs = Vector{Vector{Float64}}(undef, L + 1)
    pys = Vector{Vector{Float64}}(undef, L + 1)
    rcs[1] = zeros(Float64, 0); ecs[1] = zeros(Float64, 0)
    pys[1] = zeros(Float64, 0)
    for k in 2:(L + 1)
        rcs[k] = zeros(Float64, geo.nfs[k - 1])
        ecs[k] = zeros(Float64, geo.nfs[k - 1])
        pys[k] = zeros(Float64, geo.nfs[k])
    end
    _SEMT = SEMQUAD_MG_LEVEL_STATS
    _SEMT.last_build_s = (time_ns() - _t_build0) / 1.0e9
    _SEMT.total_build_s += _SEMT.last_build_s
    return SemQuadMG(geo, A, D, cA, cycles, pre_v, post_v, om, smoother,
                     LF, UF, operator, mfops, mfetas, mfwork, sc, sr, sy,
                     nt, par, Pts, rcs, ecs, pys)
end

# Threaded element-wise helpers (S5-2).  Every entry is computed independently,
# so splitting the range changes no floating-point operation: these are bitwise
# with respect to the threaded and serial forms.  The range is cut into
# `SEMQUAD_MG_CHUNKS_PER_THREAD * nt` chunks pulled by `nt` workers
# (`_semquad_td_run!`), so a slow (E-)core or an SMT sibling does not set the
# wall time of a memory-bound pass and the work never spreads over more than
# `nt` threads.
#
# S6 adds an optional `region`: when it is given (only from the consolidated
# V-cycle path, `_semquad_mg_vcycle_region!`), the same chunk body is published
# as one *phase* of an already fork/joined worker set instead of creating a new
# one per call.  The chunk body is a shared named function, so both drivers run
# the identical arithmetic on identical index ranges, and the phase order is the
# call order of the per-call path (G223 pins it bitwise).
@inline function _semquad_td_lo(n::Int, nc::Int, b::Int)
    return ((b - 1) * n) ÷ nc + 1
end
@inline function _semquad_td_hi(n::Int, nc::Int, b::Int)
    return (b * n) ÷ nc
end

"""
    SEMQUAD_MG_SERIAL_DOFS

Levels with at most this many reduced velocity dofs run their vector work
**serially** even in a threaded V-cycle.  A `_semquad_td_run!` call costs a
`@sync` + `nt` `@spawn`s + one atomic hand-out; on the small levels that is
larger than the work, so threading them is a loss (measured: the apply is 4.2x
slower at 318 dofs and 1.2x at 1 278, the Jacobi update 43x at 318; break-even
near 4 930; docs/dev/MG.md).  The serial
and threaded paths are bitwise identical (G190), so this threshold changes speed
only.  `0` disables it (every level threaded); `typemax(Int)` makes the whole
V-cycle serial.
"""
const SEMQUAD_MG_SERIAL_DOFS = Ref(20480)

"""
    SEMQUAD_MG_REGION

`true` (default): the threaded levels run their V-cycle ops inside the S6
consolidated regions (`_semquad_mg_vcycle_region!`, one fork/join per level
block instead of one per op); `false`: the historical per-call fork/join path
(`_semquad_mg_vcycle!`).  The two paths run the same functions on the same
chunks in the same order, so they are bitwise identical (G223); the switch
exists for the gate's A/B and for the measurement of docs/dev/MG.md
`SEMQUAD_MG_SERIAL_DOFS` still decides which levels are threaded at all.
"""
const SEMQUAD_MG_REGION = Ref(true)

"""
    SEMQUAD_MG_REGION_TRANSFER

`true` (default): a restriction/prolongation that runs **inside** an open
`SemQuadParRegion` publishes its output rows as a phase of that region, even
when its output vector is below `SEMQUAD_MG_SERIAL_DOFS`.  The threshold decides
which *levels* open a region; it is not a reason to serialise a transfer that a
finer threaded level already pays a barrier for.  The chunk bodies are the
serial ones (`_semquad_restrict_chunk!`, `_semquad_prolong_chunk!`), so the
result is bitwise the serial `mul!` (G190/G224).  `false` restores the S6
behaviour (a transfer whose output is below the threshold stays serial even
inside a region), for the A/B of docs/dev/MG.md
"""
const SEMQUAD_MG_REGION_TRANSFER = Ref(true)

"""
    SEMQUAD_MG_REGION_CU

`true` (default): the two sparse constraint products of a matrix-free apply,
`full = Cu * v` and `out = Cu' * y`, publish their output entries as phases of
the open region (`_semquad_spmv_region!`) instead of calling sparse `mul!` on the
calling task.  Both products are output-entry independent: `Cu' * y` computes
one entry per column of `Cu` with the `mul!` dot-product order, and `Cu * v` one
entry per row of `Cu` through the cached `Cu` transpose with the `mul!`
column-order accumulation.  The arithmetic is therefore identical to `mul!`
(G224 pins it) and the split is only a scheduling change.  `false` restores the
serial products for the A/B of docs/dev/MG.md
"""
const SEMQUAD_MG_REGION_CU = Ref(true)


"""Effective worker count for a length-`n` vector op (`nt <= 1` stays serial)."""
@inline function _semquad_mg_nt(n::Int, nt::Int)
    return (nt <= 1 || n <= SEMQUAD_MG_SERIAL_DOFS[]) ? 1 : nt
end

"""`r[lo:hi] .= b[lo:hi] .- y[lo:hi]` (one chunk of `_semquad_td_sub!`)."""
@inline function _semquad_td_sub_chunk!(r::Vector{Float64}, b::Vector{Float64},
                                        y::Vector{Float64}, lo::Int, hi::Int)
    @inbounds @simd for i in lo:hi
        r[i] = b[i] - y[i]
    end
    return nothing
end

"""`x[lo:hi] .+= omega .* r[lo:hi] ./ D[lo:hi]` (one chunk of the Jacobi update)."""
@inline function _semquad_td_jacobi_chunk!(x::Vector{Float64},
                                           r::Vector{Float64},
                                           D::Vector{Float64}, omega::Float64,
                                           lo::Int, hi::Int)
    @inbounds @simd for i in lo:hi
        x[i] += omega * r[i] / D[i]
    end
    return nothing
end

"""`r .= b .- y` over `nt` workers (bitwise with the broadcast)."""
function _semquad_td_sub!(r::Vector{Float64}, b::Vector{Float64},
                          y::Vector{Float64}, nt::Int,
                          region::Union{Nothing,SemQuadParRegion}=nothing)
    n = length(r)
    if nt <= 1 || n <= SEMQUAD_MG_SERIAL_DOFS[]
        _semquad_td_sub_chunk!(r, b, y, 1, n)
        return r
    end
    nc = _semquad_td_nchunks(n, nt)
    if region === nothing
        _semquad_td_run!(nt, nc) do blk
            _semquad_td_sub_chunk!(r, b, y, _semquad_td_lo(n, nc, blk),
                                   _semquad_td_hi(n, nc, blk))
        end
    else
        _semquad_phase!(region,
                        (c, w) -> _semquad_td_sub_chunk!(r, b, y,
                                                         _semquad_td_lo(n, nc, c),
                                                         _semquad_td_hi(n, nc, c)),
                        nc)
    end
    return r
end

"""`x .+= omega .* r ./ D` over `nt` workers (bitwise with `@.`)."""
function _semquad_td_jacobi!(x::Vector{Float64}, r::Vector{Float64},
                             D::Vector{Float64}, omega::Float64, nt::Int,
                             region::Union{Nothing,SemQuadParRegion}=nothing)
    n = length(x)
    if nt <= 1 || n <= SEMQUAD_MG_SERIAL_DOFS[]
        _semquad_td_jacobi_chunk!(x, r, D, omega, 1, n)
        return x
    end
    nc = _semquad_td_nchunks(n, nt)
    if region === nothing
        _semquad_td_run!(nt, nc) do blk
            _semquad_td_jacobi_chunk!(x, r, D, omega, _semquad_td_lo(n, nc, blk),
                                      _semquad_td_hi(n, nc, blk))
        end
    else
        _semquad_phase!(region,
                        (c, w) -> _semquad_td_jacobi_chunk!(x, r, D, omega,
                                                           _semquad_td_lo(n, nc, c),
                                                           _semquad_td_hi(n, nc, c)),
                        nc)
    end
    return x
end

"""
    _semquad_mg_apply!(y, x, mg, lev[, region])

`y .= A[lev] * x`, matrix-free or sparse depending on `mg.operator`.  With a
`region` the threaded branch runs the leaf phase inside that region's worker set
instead of forking per call (S6).
"""
@inline function _semquad_mg_apply!(y::Vector{Float64}, x::Vector{Float64},
                                    mg::SemQuadMG, lev::Int,
                                    region::Union{Nothing,SemQuadParRegion}=nothing)
    if mg.operator === :sparse
        mul!(y, mg.A[lev], x)
    elseif mg.threads > 1 && mg.geo.nfs[lev] > SEMQUAD_MG_SERIAL_DOFS[]
        if region === nothing
            _semquad_apply_Kred_threaded!(y, mg.mfops[lev]::SemQuadKOp,
                                          mg.geo.Cus[lev],
                                          mg.mfetas[lev]::SemQuadEtaLane, x,
                                          mg.mfwork[lev]::SemQuadKWork,
                                          mg.par[lev]::SemQuadKPar)
        else
            _semquad_apply_Kred_region!(y, mg.mfops[lev]::SemQuadKOp,
                                        mg.geo.Cus[lev],
                                        mg.geo.Cuts[lev],
                                        mg.mfetas[lev]::SemQuadEtaLane, x,
                                        mg.mfwork[lev]::SemQuadKWork,
                                        mg.par[lev]::SemQuadKPar, region)
        end
    else
        _semquad_apply_Kred!(y, mg.mfops[lev]::SemQuadKOp, mg.geo.Cus[lev],
                             mg.mfetas[lev]::SemQuadEtaLane, x,
                             mg.mfwork[lev]::SemQuadKWork)
    end
    return y
end

"""`r .= b - A[lev] * x` (reuses the level scratch, no allocation)."""
@inline function _semquad_mg_residual!(r::Vector{Float64},
                                       b::Vector{Float64}, x::Vector{Float64},
                                       mg::SemQuadMG, lev::Int,
                                       region::Union{Nothing,SemQuadParRegion}=nothing)
    if mg.operator === :sparse
        r .= b
        mul!(r, mg.A[lev], x, -1.0, 1.0)
    else
        y = mg.sy[lev]
        _semquad_mg_apply!(y, x, mg, lev, region)
        if mg.threads > 1
            _semquad_td_sub!(r, b, y, mg.threads, region)
        else
            @. r = b - y
        end
    end
    return r
end

function _semquad_mg_smooth!(x::Vector{Float64}, b::Vector{Float64},
                             mg::SemQuadMG, lev::Int, ns::Int,
                             region::Union{Nothing,SemQuadParRegion}=nothing)
    D = mg.D[lev]
    r = mg.sr[lev]
    if mg.smoother === :sgs
        LF = mg.LF[lev]; UF = mg.UF[lev]
        for _ in 1:ns
            _semquad_mg_residual!(r, b, x, mg, lev, region)
            x .+= LF \ r
            _semquad_mg_residual!(r, b, x, mg, lev, region)
            x .+= UF \ r
        end
        return nothing
    end
    for _ in 1:ns
        _semquad_mg_residual!(r, b, x, mg, lev, region)
        if mg.threads > 1
            _semquad_td_jacobi!(x, r, D, mg.omega, mg.threads, region)
        else
            @. x += mg.omega * r / D
        end
    end
    return nothing
end

# Transfer operators (S5-2).  `mul!` on the sparse `P` uses the SparseArrays
# kernels `_spmatmul!` (`C[rv] = muladd(nzv, αxj, C[rv])`) and
# `_At_or_Ac_mul_B!` (`tmp = muladd(nzv, x[rv], tmp)` per column), so a manual
# kernel that uses the *same* `muladd` order is bitwise identical to `mul!`
# (checked on every level in G190).  That is what lets the threaded path split
# the restriction by output column and the prolongation by output row (through
# the cached transposed CSC) without changing a single bit.

"""One chunk of `_semquad_mg_restrict!`: output columns `lo:hi`."""
@inline function _semquad_restrict_chunk!(rc::Vector{Float64},
                                          P::SparseMatrixCSC{Float64,Int},
                                          r::Vector{Float64}, lo::Int, hi::Int)
    @inbounds for col in lo:hi
        tmp = 0.0
        for k in nzrange(P, col)
            tmp = muladd(P.nzval[k], r[P.rowval[k]], tmp)
        end
        rc[col] = muladd(tmp, 1.0, rc[col])
    end
    return nothing
end

"""`rc .= P' * r` (bitwise the `mul!` kernel, threaded over output columns)."""
function _semquad_mg_restrict!(rc::Vector{Float64},
                               P::SparseMatrixCSC{Float64,Int},
                               r::Vector{Float64}, nt::Int,
                               region::Union{Nothing,SemQuadParRegion}=nothing)
    fill!(rc, 0.0)
    nc = length(rc)
    # Inside an open region the transfer is a phase whose only cost is a barrier,
    # so the *output* size no longer decides whether it is worth publishing: a
    # level above the threshold restricts into a level below it on every cycle
    # (docs/dev/MG.md).  `SEMQUAD_MG_REGION_TRANSFER[] = false` restores
    # the per-size serial decision for the A/B; the chunk body is the same, so
    # both settings are bitwise the serial `mul!` (G224).
    nt = (region !== nothing && SEMQUAD_MG_REGION_TRANSFER[]) ? nt :
         _semquad_mg_nt(nc, nt)
    if nt <= 1
        _semquad_restrict_chunk!(rc, P, r, 1, nc)
        return rc
    end
    nch = _semquad_td_nchunks(nc, nt)
    if region === nothing
        _semquad_td_run!(nt, nch) do c
            _semquad_restrict_chunk!(rc, P, r, _semquad_td_lo(nc, nch, c),
                                     _semquad_td_hi(nc, nch, c))
        end
    else
        _semquad_phase!(region,
                        (c, w) -> _semquad_restrict_chunk!(rc, P, r,
                                                           _semquad_td_lo(nc, nch, c),
                                                           _semquad_td_hi(nc, nch, c)),
                        nch)
    end
    return rc
end

"""One chunk of `_semquad_mg_prolong!`: output rows `lo:hi`."""
@inline function _semquad_prolong_chunk!(y::Vector{Float64},
                                         Pt::SparseMatrixCSC{Float64,Int},
                                         e::Vector{Float64}, lo::Int, hi::Int)
    @inbounds for i in lo:hi
        tmp = 0.0
        for k in nzrange(Pt, i)
            tmp = muladd(Pt.nzval[k], e[Pt.rowval[k]], tmp)
        end
        y[i] = muladd(tmp, 1.0, y[i])
    end
    return nothing
end

"""`y .= P * e` (bitwise the `mul!` kernel, threaded over output rows via `Pt`)."""
function _semquad_mg_prolong!(y::Vector{Float64},
                              Pt::SparseMatrixCSC{Float64,Int},
                              e::Vector{Float64}, nt::Int,
                              region::Union{Nothing,SemQuadParRegion}=nothing)
    fill!(y, 0.0)
    nf = length(y)
    nt = (region !== nothing && SEMQUAD_MG_REGION_TRANSFER[]) ? nt :
         _semquad_mg_nt(nf, nt)
    if nt <= 1
        _semquad_prolong_chunk!(y, Pt, e, 1, nf)
        return y
    end
    nch = _semquad_td_nchunks(nf, nt)
    if region === nothing
        _semquad_td_run!(nt, nch) do c
            _semquad_prolong_chunk!(y, Pt, e, _semquad_td_lo(nf, nch, c),
                                    _semquad_td_hi(nf, nch, c))
        end
    else
        _semquad_phase!(region,
                        (c, w) -> _semquad_prolong_chunk!(y, Pt, e,
                                                          _semquad_td_lo(nf, nch, c),
                                                          _semquad_td_hi(nf, nch, c)),
                        nch)
    end
    return y
end

"""True when level `lev` runs its ops threaded (worker count and dof threshold)."""
@inline function _semquad_mg_threaded_level(mg::SemQuadMG, lev::Int)
    return SEMQUAD_MG_REGION[] && mg.threads > 1 &&
           mg.geo.nfs[lev] > SEMQUAD_MG_SERIAL_DOFS[]
end

function _semquad_mg_vcycle!(x::Vector{Float64}, b::Vector{Float64},
                             mg::SemQuadMG, lev::Int)
    if lev == 1
        x .= mg.cA \ b
        return nothing
    end
    if _semquad_mg_threaded_level(mg, lev)
        _semquad_mg_vcycle_region!(x, b, mg, lev)
        return nothing
    end
    fill!(x, 0.0)
    _semquad_mg_smooth!(x, b, mg, lev, mg.pre[lev])
    # Scratch buffers are per level and fully overwritten (`_semquad_mg_vcycle!`
    # either assigns `x .= cA \ b` or `fill!(x, 0.0)` before writing), so
    # reusing them is bitwise the `P' * r` / `zeros` / `P * ec` version.
    rc = mg.rcs[lev]
    _semquad_mg_restrict!(rc, mg.geo.Ps[lev - 1],
                          _semquad_mg_residual!(mg.sc[lev], b, x, mg, lev),
                          mg.threads)
    ec = mg.ecs[lev]
    _semquad_mg_vcycle!(ec, rc, mg, lev - 1)
    py = mg.pys[lev]
    if mg.threads > 1 && !isempty(mg.Pts)
        _semquad_mg_prolong!(py, mg.Pts[lev - 1]::SparseMatrixCSC{Float64,Int},
                             ec, mg.threads)
    else
        mul!(py, mg.geo.Ps[lev - 1], ec)
    end
    x .+= py
    _semquad_mg_smooth!(x, b, mg, lev, mg.post[lev])
    return nothing
end

"""
    _semquad_mg_vcycle_region!(x, b, mg, lev)

The S6 consolidated V-cycle body of one *threaded* level: the pre-smoothing
sweeps, the residual and the restriction run inside **one** fork/join
(`_semquad_region_run!`), and the prolongation, the `x .+= py` and the
post-smoothing sweeps inside a second one.  The level's ops are the same
functions, in the same order, with the same chunk bodies as the per-call path
(`_semquad_mg_vcycle!`); only the task creation moved out of the per-op path, so
the result is bitwise identical (G223) and the level's fork count drops from one
per op to two.
"""
function _semquad_mg_vcycle_region!(x::Vector{Float64}, b::Vector{Float64},
                                    mg::SemQuadMG, lev::Int)
    fill!(x, 0.0)
    rc = mg.rcs[lev]
    # The leaf apply hands worker `w` its own scratch slot (`par.scr[w]`), and
    # `SemQuadKPar` clamps its worker count to the level's leaf count, so the
    # region uses that same count: a level with fewer leaves than `mg.threads`
    # runs its phases on fewer workers (same chunks, same arithmetic).  Without
    # the clamp a small level would index past `par.scr` (G223 builds such a
    # level on purpose).
    nw = mg.operator === :matrixfree ? (mg.par[lev]::SemQuadKPar).nt : mg.threads
    region = SemQuadParRegion(nw)
    _semquad_region_run!(region) do
        _semquad_mg_smooth!(x, b, mg, lev, mg.pre[lev], region)
        _semquad_mg_restrict!(rc, mg.geo.Ps[lev - 1],
                              _semquad_mg_residual!(mg.sc[lev], b, x, mg, lev,
                                                    region),
                              mg.threads, region)
    end
    ec = mg.ecs[lev]
    _semquad_mg_vcycle!(ec, rc, mg, lev - 1)
    py = mg.pys[lev]
    _semquad_region_run!(region) do
        if isempty(mg.Pts)
            mul!(py, mg.geo.Ps[lev - 1], ec)
        else
            _semquad_mg_prolong!(py,
                                 mg.Pts[lev - 1]::SparseMatrixCSC{Float64,Int},
                                 ec, mg.threads, region)
        end
        x .+= py
        _semquad_mg_smooth!(x, b, mg, lev, mg.post[lev], region)
    end
    return nothing
end


"""
    semquad_mg_apply(mg, b)

Apply the (fixed, linear) MG approximate inverse: `cycles` V-cycles.
"""
function semquad_mg_apply(mg::SemQuadMG, b::AbstractVector{<:Real})
    bb = Vector{Float64}(b)
    z = zeros(Float64, length(bb))
    _semquad_mg_vcycle!(z, bb, mg, mg.geo.L + 1)
    for _ in 2:mg.cycles
        rz = mg.sc[end]
        _semquad_mg_apply!(rz, z, mg, mg.geo.L + 1)
        r = bb .- rz
        dz = zeros(Float64, length(bb))
        _semquad_mg_vcycle!(dz, r, mg, mg.geo.L + 1)
        z .+= dz
    end
    return z
end


"""
    SemQuadPressureCache

Cached pressure-block approximate inverse of the `linear=:mg` saddle solve
(S3 stage A).  The three-factor w-BFBT costs ~1.1 s to build on the 1024-leaf
blob tree, but it depends only on the tree, the pressure space and the viscosity
field: the tree changes every few AMR cycles and `eta` moves a fraction of a
percent per step.  The outer solver is *flexible* GMRES, so a slightly stale
`Shat^{-1}` is a perfectly valid preconditioner -- exactly the S2 stale-factor
idea (`StaleSaddle`) applied to the pressure block instead of the whole saddle.

`eta` is the sampled viscosity vector the factors were built from; a rebuild is
triggered when `max|eta_new/eta_old - 1|` exceeds the caller's threshold.
"""
mutable struct SemQuadPressureCache
    key::UInt64              # tree + geometry fingerprint (`_saddle_key`)
    mode::Symbol             # resolved mode (:mass, :bfbt, :lsc, :stiffness)
    pscale::Float64
    weight::Symbol
    dweight::Symbol          # BFBT weight construction (:leaf, :exact, :matrix)
    s1solve::Symbol          # BFBT inner solve (:chol, :level, :cg_diag, :cg_ichol)
    s1tol::Float64           # its relative tolerance (CG kinds)
    eta::Vector{Float64}     # eta sample the factors were built from
    Fp::Any                  # (kind=:bfbt, F1, S1solve, S2) or a Cholesky factorisation
    builds::Int
end

const SEMQUAD_PRESSURE_CACHE = Ref{Union{Nothing,SemQuadPressureCache}}(nothing)
# Default cache threshold of the pressure block (S3-press A3, retuned in S5-1).
# The blob's viscosity moves 5.7-8.9% per *step* (25-37% between stored frames),
# so the old 1e-2 threshold almost never hit.  The measured threshold curve
# (docs/S3_DESIGN.md §13.3) shows the outer iteration count is *flat* (39.7 at
# 1e-12, 39.9 even at Inf), so the threshold only trades build time against
# staleness; the AMR tree changes every 5 steps and forces a rebuild anyway, so
# 1.0 buys one build per tree block (21 builds instead of 41 in the 25-frame
# blob run, §19.1).  `Inf` is a *supported but not default* value: it never
# rebuilds on eta drift (the tree fingerprint still wins) and is only safe for
# smoothly evolving viscosity -- no phase change and no yield onset inside a
# tree block (docs/S3_DESIGN.md §19.1).
const SEMQUAD_PRESSURE_DEFAULT_ETA_TOL = 1.0
# Diagnostics for gates/probes: cumulative builds and reuses, and what the last
# call did.
const SEMSTOKES_PRESSURE_BUILDS = Ref(0)
const SEMSTOKES_PRESSURE_REUSES = Ref(0)
const LAST_SEMSTOKES_PRESSURE_REUSED = Ref(false)
const LAST_SEMSTOKES_PRESSURE_JUMP = Ref(0.0)
# Resolved mode of the last pressure block (diagnostic for gates/probes): one of
# :mass, :stiffness, :lsc, :bfbt, :bfbt_m1, :pcd.
const LAST_SEMSTOKES_PRESSURE_MODE = Ref(:none)

"""
    _semquad_pressure_precond_mode(pre, weight) -> (mode, weight)

Resolve the public `mg_pressure_precond` keyword (`:bfbt`, `:mass`,
`:scaled_mass`, `:pcd`, `:lsc`) into the internal `(mode, weight)` pair of the
factor builder:

| `mg_pressure_precond` | internal mode | weight |
|---|---|---|
| `:bfbt` | `:bfbt` | (unused) |
| `:mass` | `:mass` | `:none` (unweighted `M_p`) |
| `:scaled_mass` | `:mass` | `mg_pressure_weight` (`:inv_eta` by default) |
| `:pcd` | `:pcd` | (unused) |
| `:lsc` | `:lsc` | (unused) |

`:mass` is the *unweighted* pressure mass matrix on purpose: it is the variant
the spectral-equivalence theory names, and keeping it distinct from
`:scaled_mass` makes the two measurable side by side (docs/S3_DESIGN.md §13).
"""
function _semquad_pressure_precond_mode(pre::Symbol, weight::Symbol)
    pre === :bfbt && return (:bfbt, weight)
    pre === :lsc && return (:lsc, weight)
    pre === :pcd && return (:pcd, weight)
    pre === :mass && return (:mass, :none)
    pre === :scaled_mass && return (:mass, weight)
    throw(ArgumentError("mg_pressure_precond must be :auto, :bfbt, :mass, " *
                        ":scaled_mass, :pcd or :lsc"))
end

"""
    _semquad_eta_sample(su, eta_fun; nsamples=256)

Viscosity sampled on a fixed, evenly strided set of velocity nodes (`<= 256`
samples, the same set the `:auto` pressure-mode decision uses).  Used as the
change metric of the pressure-cache: the sample is deterministic for a given
tree, so `max|eta_new/eta_old - 1|` measures how much the viscosity field moved
between two Stokes solves.
"""
function _semquad_eta_sample(su::SemQuad, eta_fun::Function; nsamples::Int=256)
    N = su.N
    stride = max(1, N ÷ nsamples)
    vals = Vector{Float64}(undef, length(1:stride:N))
    i = 0
    @inbounds for g in 1:stride:N
        e = eta_fun(su.coords[g]...)
        e > 0.0 || throw(ArgumentError("eta_fun returned $e <= 0"))
        vals[i += 1] = e
    end
    return vals
end

"""Number of velocity nodes `_semquad_eta_sample` visits on a space of `N` nodes."""
_semquad_eta_sample_count(N::Integer; nsamples::Int=256) =
    length(1:max(1, Int(N) ÷ nsamples):Int(N))

"""
    semquad_eta_contrast(su, eta_fun; npoints=SEMQUAD_AUTO_CONTRAST_GRID) -> Float64

Viscosity contrast `max(eta)/min(eta)` of the **actual** eta field the solve
uses, sampled on a fixed Cartesian grid of `(npoints+1)^2` points spanning
`[0, Lx] x [0, 1]` (default 32 intervals -> 33x33 = 1 089 points).  This is the
`contrast` input of `semquad_choose_linear` / `linear=:auto`.

Why a global grid instead of the <=256 strided velocity nodes used by the
pressure-cache metric (`_semquad_eta_sample`): measured on the production blob
frame-19 eta field, the node sample under-estimates the full quadrature-field
contrast by up to 5x (985x vs 4 926x on the 1024-leaf tree, 995x vs 5 299x on
the 4096-leaf tree), while 33x33 grid points give 5 087x (-3% / -9% vs the two
trees) with the same value on every tree (docs/S3_DESIGN.md §20).  Sampling a
strided *subset of leaves* is worse still (808-892x on the refined trees) for
the same reason: the viscosity extrema are spatially localised.  The grid cost
is fixed and independent of the tree: measured ~20 us on the production blob
closure (1 089 evaluations, ~18 ns each; a constant eta closure is folded to
nothing), vs 0.7-2.6 ms for a full O(N) quadrature-field pass on the
1024-4096-leaf trees.

The grid sample can still miss a sub-grid-scale extremum, but the selector's
contrast cut-offs are 10x/100x and the measured fields sit at 1x and ~5 000x;
grid resolution is reported in docs/S3_DESIGN.md §20.
"""
function semquad_eta_contrast(su::SemQuad, eta_fun::Function;
                              npoints::Int=SEMQUAD_AUTO_CONTRAST_GRID)
    n = max(1, npoints)
    Lx = su.Lx
    lo = Inf; hi = 0.0
    @inbounds for j in 0:n, i in 0:n
        e = eta_fun(Lx * (i / n), j / n)
        e > 0.0 || throw(ArgumentError("eta_fun returned $e <= 0 at " *
                                       "($(Lx * i / n), $(j / n))"))
        lo = min(lo, e); hi = max(hi, e)
    end
    return hi / lo
end

"""
    semquad_mg_pressure_precond(ss, Cu, Cp, Bs, Bt, Kred, eta_fun; mode=:auto,
                                scale=0.0, weight=:inv_eta, key=nothing,
                                eta_tol=SEMQUAD_PRESSURE_DEFAULT_ETA_TOL,
                                cache=true)

Pressure-block approximate inverse for the `linear=:mg` saddle solve (and the
MG-accelerated Newton).  Returns `(Fp, pscale)`:

* `mode=:auto` picks `:mass` for viscosity contrast <= 100 and the three-factor
  w-BFBT for stronger contrast (sampled on <= 256 velocity nodes);
* `:mass` / `:stiffness` build `Cp' M Cp` with the `weight`-scaled pressure
  mass/stiffness; `:lsc` uses `Bs diag(Kred)^{-1} Bt`; `:pcd` builds the
  pressure convection-diffusion factors `A_p^{-1} F_p M_p^{-1}` (see
  `_semquad_build_pressure_precond`);
* `:bfbt` builds `S1 = Bs D^{-1} Bt`, `S2 = Bs D^{-1} Kred D^{-1} Bt` with
  `D = diag(Cu' M_u(sqrt(eta)) Cu)` (the *restricted* diagonal, not the lumped
  row sums; `bfbt_d` selects the construction -- see
  [`_semquad_bfbt_diag`](@ref)) and `F1 = chol(S1)`.  `bfbt_d=:auto` resolves
  to the exact leaf-local `:exact` weight; `:leaf` (lumped) and `:matrix`
  (sparse reference) stay selectable for the A/B and the gate.

`Fp` is either a `NamedTuple` `(kind=:bfbt, F1, S1solve, S2=...)` or
`(kind=:mass, F=...)`; `pscale` is the block scaling used by the caller.

**The `S1` inner solve.**  `bfbt_s1_solve` selects how the two `F1` solves of the
w-BFBT apply are carried out (see `julia/src/semquad_s1solve.jl` and
docs/dev/MG.md):

| value | the solve | measured |
|---|---|---|
| `:chol` (default) | `ldiv!(x, F1, b)`, the exact CHOLMOD solve | 0.42-0.46 ms per solve on `F` |
| `:level` | the same factor, level-scheduled parallel substitutions | 0.43x of `:chol`: no level can pay for a barrier on this factor |
| `:cg_diag` / `:cg_ichol` | CG to `bfbt_s1_tol` with the diagonal / an IC(0) | 51-75 CG iterations per solve, 17-22 ms: 40x `:chol` |

The default is the exact solve; the other kinds exist because the measurement is
reproducible, and docs/dev/MG.md records why neither of them ships.
`bfbt_s1_tol` / `bfbt_s1_maxit` configure the CG kinds, and
`bfbt_s1_minwork` / `bfbt_s1_chunk` the level schedule.  The pressure cache is
keyed on the resolved kind and tolerance, so switching them rebuilds.

**Caching (S3 stage A).**  When `key` is given (the tree/geometry fingerprint
from `_saddle_key`) and `cache=true`, the factors are reused as long as the key,
the resolved mode, the scaling and the weight match and the sampled viscosity
has moved by at most `eta_tol` (relative, max norm).  Otherwise they are
rebuilt.  `key=nothing` disables the cache, which keeps the Newton path (one
factorisation per nonlinear solve) bit-identical to before.

`eta_tol` semantics (S5-1 retune, docs/S3_DESIGN.md §19.1):

| `eta_tol` | rebuilds when |
|---|---|
| `1e-12` | the sampled viscosity changes at all (the pre-cache behaviour) |
| `1.0` (default) | the tree fingerprint changes, or any sampled `eta` moves by >100% (a phase change / yield onset inside one tree block) |
| `Inf` | **only** when the tree fingerprint changes; eta drift is always accepted |

The outer FGMRES is flexible, so a stale `Shat^{-1}` is a valid preconditioner;
§13.3 measured the outer iteration count to be flat (39.7 at `1e-12`, 39.9 at
`Inf`) because BFBT's mass is set by the geometry and `eta` only enters through
`D = diag(M_u(sqrt(eta)))`.  `Inf` is therefore *available* but **not the
default**: it is only for smoothly evolving viscosity (no phase change, no yield
onset) -- otherwise the factors can go stale by orders of magnitude and the
outer iteration count must be re-measured.
"""
function semquad_mg_pressure_precond(ss::SemStokes, Cu, Cp, Bs, Bt, Kred,
                                     eta_fun::Function;
                                     mode::Symbol=:auto, scale::Float64=0.0,
                                     weight::Symbol=:inv_eta,
                                     key::Union{Nothing,UInt64}=nothing,
                                     eta_tol::Float64=SEMQUAD_PRESSURE_DEFAULT_ETA_TOL,
                                     cache::Bool=true,
                                     bfbt_d::Symbol=:leaf,
                                     bfbt_s2::Symbol=:free,
                                     bfbt_s1::Symbol=:chol,
                                     bfbt_s1_solve::Symbol=:chol,
                                     bfbt_s1_tol::Float64=1.0e-2,
                                     bfbt_s1_maxit::Int=50,
                                     bfbt_s1_minwork::Int=0,
                                     bfbt_s1_chunk::Int=0,
                                     nthreads::Int=1,
                                     applyK::Union{Nothing,Function}=nothing)
    bfbt_d === :auto && (bfbt_d = :exact)
    bfbt_d in (:leaf, :exact, :matrix) ||
        throw(ArgumentError("bfbt_d must be :auto, :leaf, :exact or :matrix"))
    bfbt_s1_solve in (:chol, :level, :cg_diag, :cg_ichol) ||
        throw(ArgumentError("bfbt_s1_solve must be :chol, :level, :cg_diag or " *
                            ":cg_ichol"))
    LAST_SEMSTOKES_PRESSURE_REUSED[] = false
    LAST_SEMSTOKES_PRESSURE_JUMP[] = 0.0
    use_cache = cache && key !== nothing
    # One sampling pass serves both the `:auto` decision and the cache metric.
    esamp = (use_cache || mode === :auto) ?
            _semquad_eta_sample(ss.su, eta_fun) : Float64[]
    pmode = mode
    if pmode === :auto
        pmode = (maximum(esamp) / minimum(esamp) > 100.0) ? :bfbt : :mass
    end
    pscale = scale == 0.0 ? (pmode === :lsc ? 0.1 : pmode === :bfbt ? 0.4 : 1.0) :
             scale
    st = SEMQUAD_PRESSURE_CACHE[]
    if use_cache && st !== nothing && st.key == key && st.mode === pmode &&
       st.weight === weight && st.dweight === bfbt_d && st.pscale == pscale &&
       st.s1solve === bfbt_s1_solve && st.s1tol == bfbt_s1_tol &&
       length(st.eta) == length(esamp)
        jump = 0.0
        @inbounds for i in eachindex(esamp)
            jump = max(jump, abs(esamp[i] / st.eta[i] - 1.0))
        end
        LAST_SEMSTOKES_PRESSURE_JUMP[] = jump
        if jump <= eta_tol
            SEMSTOKES_PRESSURE_REUSES[] += 1
            LAST_SEMSTOKES_PRESSURE_REUSED[] = true
            Fp = st.Fp
            # A cached matrix-free S2 holds the *previous* `Kred`; rebind it to
            # the current physics so a stale cache does not also hold a stale
            # operator alive.
            if Fp isa NamedTuple && Fp.S2 === nothing && applyK !== nothing
                Fp = merge(Fp, (applyK=applyK, Kred=nothing))
            end
            LAST_SEMSTOKES_PRESSURE_MODE[] =
                Fp isa NamedTuple ? Fp.kind : st.mode
            return Fp, st.pscale
        end
    end
    Fp, pscale = _semquad_build_pressure_precond(ss, Cu, Cp, Bs, Bt, Kred,
                                                 eta_fun, pmode, pscale, weight;
                                                 bfbt_d=bfbt_d, bfbt_s2=bfbt_s2,
                                                 bfbt_s1=bfbt_s1,
                                                 bfbt_s1_solve=bfbt_s1_solve,
                                                 bfbt_s1_tol=bfbt_s1_tol,
                                                 bfbt_s1_maxit=bfbt_s1_maxit,
                                                 bfbt_s1_minwork=bfbt_s1_minwork,
                                                 bfbt_s1_chunk=bfbt_s1_chunk,
                                                 nthreads=nthreads,
                                                 applyK=applyK)
    if use_cache
        n = st === nothing ? 1 : st.builds + 1
        SEMQUAD_PRESSURE_CACHE[] = SemQuadPressureCache(key, pmode, pscale,
                                                        weight, bfbt_d,
                                                        bfbt_s1_solve, bfbt_s1_tol,
                                                        esamp, Fp, n)
    end
    SEMSTOKES_PRESSURE_BUILDS[] += 1
    LAST_SEMSTOKES_PRESSURE_MODE[] = Fp isa NamedTuple ? Fp.kind : pmode
    return Fp, pscale
end

"""
    _semquad_build_pressure_precond(ss, Cu, Cp, Bs, Bt, Kred, eta_fun, pmode,
                                    pscale, weight)

Build the pressure-block approximate inverse selected by the resolved `pmode`:
the uncached body of `semquad_mg_pressure_precond`.
"""
function _semquad_build_pressure_precond(ss::SemStokes, Cu, Cp, Bs, Bt, Kred,
                                         eta_fun::Function, pmode::Symbol,
                                         pscale::Float64, weight::Symbol;
                                         bfbt_d::Symbol=:leaf,
                                         bfbt_s2::Symbol=:free,
                                         bfbt_s1::Symbol=:chol,
                                         bfbt_s1_solve::Symbol=:chol,
                                         bfbt_s1_tol::Float64=1.0e-2,
                                         bfbt_s1_maxit::Int=50,
                                         bfbt_s1_minwork::Int=0,
                                         bfbt_s1_chunk::Int=0,
                                         nthreads::Int=1,
                                         applyK::Union{Nothing,Function}=nothing)
    su = ss.su
    sp = ss.sp
    # The Schur complement S = B K^{-1} B' scales like 1/eta, so the pressure
    # block of the block-triangular preconditioner uses the *weighted* mass
    # matrix int q_i q_j / eta, not the unweighted mass (which is only
    # spectrally equivalent for constant viscosity).  `:lsc` uses the
    # least-squares commutator B diag(K)^{-1} B' and `:stiffness` the weighted
    # pressure Laplacian; both are diagnostic options.
    pweight = (x, y) -> begin
        e = eta_fun(x, y)
        e > 0.0 || throw(ArgumentError("eta_fun returned $e <= 0"))
        weight === :eta ? e : weight === :none ? 1.0 : 1.0 / e
    end
    if pmode === :pcd
        # Pressure convection-diffusion (PCD, Elman-Silvester-Wathen ch. 9):
        #   Shat^{-1} = A_p^{-1} F_p M_p^{-1},   F_p = A_p + C_p,
        # with A_p the pressure-space version of the momentum operator's
        # Laplacian (here -div(eta grad q), Neumann on the whole boundary, made
        # definite by the caller's pinned pressure dof `Cp`), M_p the
        # *unweighted* pressure mass matrix, and C_p the pressure convection
        # operator built from the advecting velocity.
        #
        # The Stokes momentum equation of this code has **no advection term**
        # (infinite-Prandtl convection), so C_p = 0 and F_p = A_p identically;
        # the three factors then collapse to M_p^{-1} in exact arithmetic and
        # the iteration count equals that of `mg_pressure_precond=:mass`.  The
        # mode is kept (and wired) because the measurement is the point: on the
        # 1024-leaf blob tree it costs 2.7x more per apply than `:mass` for the
        # same outer iteration count (docs/S3_DESIGN.md §13).
        Ap = assemble_semquad_pressure_stiffness(sp;
                                                 weight=(x, y) -> eta_fun(x, y))
        Mp = assemble_semquad_pressure_mass(sp)
        Ared = Cp' * Ap * Cp
        Ared = 0.5 * (Ared + sparse(transpose(Ared)))
        Mred = Cp' * Mp * Cp
        Mred = 0.5 * (Mred + sparse(transpose(Mred)))
        FA = cholesky(Symmetric(Ared))
        issuccess(FA) ||
            throw(ArgumentError("PCD pressure Laplacian factorisation failed"))
        FM = cholesky(Symmetric(Mred))
        issuccess(FM) || throw(ArgumentError("PCD pressure mass is not SPD"))
        # `FF` is the *matrix* F_p = A_p + C_p (C_p = 0): the middle factor is a
        # matvec, only the two outer ones are triangular solves.
        return (kind=:pcd, FA=FA, FF=Ared, FM=FM), pscale
    end
    if pmode === :bfbt
        # Three-factor w-BFBT (Rudi-Stadler-Ghattas 2017):
        #   D = diag(M_u(sqrt(eta))),
        #   Shat^{-1} = S1^{-1} S2 S1^{-1},
        #   S1 = B D^{-1} B',  S2 = B D^{-1} K D^{-1} B'.
        wfun = (x, y) -> sqrt(max(eta_fun(x, y), 1.0e-300))
        d = _semquad_bfbt_diag(su, Cu, eta_fun, wfun; how=bfbt_d)
        Dinvv = 1.0 ./ d
        Dinv = spdiagm(0 => Dinvv)
        S1 = Bs * Dinv * Bt
        S1 = 0.5 * (S1 + sparse(transpose(S1)))
        dS1 = maximum(abs, diag(S1))
        S1 = S1 + (1.0e-14 * max(dS1, 1.0)) * I
        # `S2 = Bs D^{-1} Kred D^{-1} Bt` (327 nnz/row, 0.42 s to form on the
        # 1024-leaf tree) or its matrix-free application
        # `Bs (D^{-1} (Kred (D^{-1} (Bt r))))`, which costs the same per apply
        # (3.0 vs 3.1 ms) and no build time.  Forming it needs a sparse `Kred`;
        # on the fully matrix-free path (stage C) only the apply exists.
        S2 = (bfbt_s2 === :free || Kred === nothing) ? nothing :
             begin
                 S2m = Bs * Dinv * Kred * Dinv * Bt
                 0.5 * (S2m + sparse(transpose(S2m)))
             end
        # When S2 is applied matrix-free, keep whichever velocity operator the
        # caller can provide: the leaf-local apply of the fully matrix-free path
        # (stage C), or else the sparse `Kred`.
        s2Kred = (S2 === nothing && applyK === nothing) ? Kred : nothing
        s2apply = S2 === nothing ? applyK : nothing
        if bfbt_s1 === :mass
            # (b2) spectral equivalence: S1 = B D^{-1} B' is spectrally
            # equivalent to the 1/eta-weighted pressure mass matrix, so its
            # inverse can be replaced by the (much cheaper) mass inverse.
            Mfull = assemble_semquad_pressure_mass(sp; weight=pweight)
            Mred = Cp' * Mfull * Cp
            Mred = 0.5 * (Mred + sparse(transpose(Mred)))
            F1 = cholesky(Symmetric(Mred))
            issuccess(F1) || throw(ArgumentError("BFBT S1 (mass) is not SPD"))
            return (kind=:bfbt_m1, F1=F1, S2=S2, Dinvv=Dinvv, Bs=Bs, Bt=Bt,
                    Kred=s2Kred, applyK=s2apply), pscale
        end
        F1 = cholesky(Symmetric(S1))
        issuccess(F1) || throw(ArgumentError("BFBT S1 is not SPD"))
        # The inner solve is a first-class object (`semquad_s1solve.jl`): the
        # exact CHOLMOD solve by default, a level-scheduled solve over the same
        # factor or a loose-tolerance CG behind `bfbt_s1_solve`.
        S1i = _semquad_s1_inner(bfbt_s1_solve, F1, S1, nthreads; tol=bfbt_s1_tol,
                                maxit=bfbt_s1_maxit, minwork=bfbt_s1_minwork,
                                chunkwork=bfbt_s1_chunk)
        return (kind=:bfbt, F1=F1, S1solve=S1i, S2=S2, Dinvv=Dinvv, Bs=Bs, Bt=Bt,
                Kred=s2Kred, applyK=s2apply), pscale
    end
    Mfull = pmode === :mass ?
            assemble_semquad_pressure_mass(sp; weight=pweight) :
            pmode === :stiffness ?
            assemble_semquad_pressure_stiffness(sp; weight=pweight) :
            nothing
    Mred = if Mfull === nothing
        Kred === nothing &&
            throw(ArgumentError("mg_pressure_precond=:lsc (and " *
                                "mg_pressure_mode=:lsc) needs `diag(Kred)`, i.e. " *
                                "the assembled velocity operator; use " *
                                "mg_operator=:sparse with :lsc"))
        dK = max.(diag(Kred), 1.0e-14 * maximum(abs, diag(Kred)))
        DinvK = spdiagm(0 => 1.0 ./ dK)
        Bs * DinvK * Bt
    else
        Cp' * Mfull * Cp
    end
    Mred = 0.5 * (Mred + sparse(transpose(Mred)))
    F = cholesky(Symmetric(Mred))
    issuccess(F) || throw(ArgumentError("pressure Schur preconditioner is not SPD"))
    return F, pscale
end

"""
    _semquad_mass_diag!(d, su, w)

Diagonal of the velocity mass matrix `int w(x,y) u_i u_j` on the velocity space
`su`, computed leaf by leaf and scatter-added (`d` has length `su.N`).  The
assembled matrix has `(p+1)^4` entries per quadrature point; the diagonal has
`(p+1)^2`, so this is ~25x cheaper at p=4 and needs no sparse matrix at all.
"""
function _semquad_mass_diag!(d::Vector{Float64}, su::SemQuad, w::Function)
    p = su.p
    n = p + 1
    # `assemble_semquad_pressure_mass(su)` builds `make_fedata(2, su.p)` and uses
    # `fd.phi_p`, i.e. the degree-`p` nodal basis on the degree-`p` GLL nodes --
    # which is exactly the velocity basis of `su`.  Use the same table so the
    # leaf-wise diagonal is the assembled matrix's diagonal, not a different
    # quadrature of it.
    fd = make_fedata(2, p)
    dloc = zeros(Float64, n * n)
    fill!(d, 0.0)
    @inbounds for lf in su.leaves
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = su.elem[lf]
        fill!(dloc, 0.0)
        for qy in 1:fd.nq, qx in 1:fd.nq
            wq = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy] *
                 w(x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx,
                   y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy)
            for j in 1:n, i in 1:n
                ph = fd.phi_p[i, qx] * fd.phi_p[j, qy]
                dloc[(j - 1) * n + i] += wq * ph * ph
            end
        end
        for idx in 1:(n * n)
            d[uids[idx]] += dloc[idx]
        end
    end
    return d
end

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
# ============================================================================
# S3 / MF-4: CG on the SPD velocity block + flexible outer GMRES
# ============================================================================

"""
    SemQuadCGWork(n)

Preallocated scratch for `semquad_mg_cg` (the inner velocity CG of MF-4), so the
outer FGMRES does not allocate a handful of length-`n` vectors per preconditioner
application.
"""
struct SemQuadCGWork
    z::Vector{Float64}
    r::Vector{Float64}
    p::Vector{Float64}
    Ap::Vector{Float64}
    w::Vector{Float64}
end

SemQuadCGWork(n::Int) = SemQuadCGWork(zeros(n), zeros(n), zeros(n), zeros(n),
                                      zeros(n))

"""
    semquad_mg_cg(applyK!, b, mg; tol=1.0e-2, maxit=10, work=SemQuadCGWork(length(b)))

Preconditioned conjugate gradients for the SPD reduced velocity block `K z = b`
with the multigrid hierarchy as the preconditioner: one `semquad_mg_apply`
(i.e. `mg.cycles` V-cycles) per CG iteration.  `applyK!` is the matrix-vector
product of `K` (`mul!` on `Kred`, or the matrix-free `semquad_apply_Kred!`).

This is the inner solver of MF-4: the outer FGMRES then sees a *much* better
velocity inverse than a fixed number of V-cycles (two V-cycles are only a ~2%
approximate inverse at p=4 L4), at the cost of a data-dependent iteration
count -- exactly why the outer Krylov method must be flexible.

Returns `(z, iters, relres)` with `z === work.z`.  It bails out (returning the
current iterate) if the preconditioner loses positivity or a search direction
becomes non-positive, so a non-SPD level operator cannot make the outer
iteration diverge silently.
"""
function semquad_mg_cg(applyK!::Function, b::Vector{Float64}, mg::SemQuadMG;
                       tol::Float64=1.0e-2, maxit::Int=10,
                       work::SemQuadCGWork=SemQuadCGWork(length(b)))
    n = length(b)
    z = work.z; r = work.r; p = work.p; Ap = work.Ap; w = work.w
    fill!(z, 0.0)
    copyto!(r, b)
    bn = norm(r)
    bn == 0.0 && return z, 0, 0.0
    rho = 0.0
    rel = 1.0
    it = 0
    for k in 1:maxit
        it = k
        w .= semquad_mg_apply(mg, r)
        rho_new = dot(r, w)
        if !(rho_new > 0.0) || !isfinite(rho_new)
            break                      # preconditioner lost positivity
        end
        if k == 1
            p .= w
        else
            beta = rho_new / rho
            @. p = w + beta * p
        end
        applyK!(Ap, p)
        den = dot(p, Ap)
        if !(den > 0.0) || !isfinite(den)
            break                      # search direction not an ascent direction
        end
        alpha = rho_new / den
        @. z += alpha * p
        @. r -= alpha * Ap
        rho = rho_new
        rel = norm(r) / bn
        rel <= tol && break
    end
    return z, it, rel
end

const SEMQUAD_FGMRES_WORK = Ref{Any}(nothing)

"""
    _semquad_fgmres_work(n, m)

The reusable FGMRES workspace of the saddle solve, (re)allocated when the system
size `n` or the block width `m` changes.  A fresh `n x 100` workspace is ~100 MB
on the 1024-leaf blob tree and costs ~0.1 s of page faults per solve, so it is
kept across solves (see `FGMRESWork`).
"""
function _semquad_fgmres_work(n::Int, m::Int)
    w = SEMQUAD_FGMRES_WORK[]
    if w === nothing || !(w isa FGMRESWork) || size(w.V, 1) != n ||
       size(w.V, 2) != m + 1
        w = FGMRESWork(n, m)
        SEMQUAD_FGMRES_WORK[] = w
    end
    return w
end

"""
    semquad_mg_saddle_solve(Kred, Bs, Bt, Fp, pscale, b, mg; tol, maxit, restart,
                            vel_solver, vel_tol, vel_maxit)

Solve the reduced saddle

    [Kred  Bt] [u]   [fr]
    [Bs    0 ] [p] = [gr]

with flexible GMRES (`fgmres_restarted`) and the block-triangular right
preconditioner

    P = [Khat  Bt]     Khat   = K^{-1} approximated by the multigrid hierarchy,
        [0    -Shat]   Shat^{-1} = pscale * Fp^{-1}

where `Fp` is either the pressure-block Cholesky factor of the mass/stiffness/LSC
mode or the three-factor w-BFBT `(kind=:bfbt, F1, S2)` named tuple.

`vel_solver=:cg` (MF-4, default) replaces the fixed V-cycle count by an inner CG
on `Kred` preconditioned by the hierarchy (`semquad_mg_cg`), i.e. the
deal.II/ASPECT scheme: FGMRES outside, CG + multigrid inside.  The
preconditioner is then data dependent, which is why the outer loop must be
FGMRES.  `vel_solver=:vcycle` keeps the historical fixed `mg.cycles` V-cycles
(a *linear* preconditioner, so the two are interchangeable there).

The outer stopping criterion is the **true** reduced-saddle residual
(`fgmres_restarted`), never the preconditioned one.  Returns
`(sol, nit, relres, cgiters)`.
"""
function semquad_mg_saddle_solve(Kred::Union{Nothing,AbstractMatrix{Float64}},
                                 Bs::AbstractMatrix{Float64},
                                 Bt::AbstractMatrix{Float64},
                                 Fp, pscale::Float64, b::Vector{Float64},
                                 mg::SemQuadMG;
                                 tol::Float64=1.0e-10, maxit::Int=500,
                                 restart::Int=100, vel_solver::Symbol=:vcycle,
                                 vel_tol::Float64=1.0e-2, vel_maxit::Int=10,
                                 applyK::Union{Nothing,Function}=nothing)
    vel_solver in (:cg, :vcycle) ||
        throw(ArgumentError("MG velocity solver must be :cg or :vcycle"))
    nf = size(Bs, 2)
    nfp = size(Bs, 1)
    nftot = nf + nfp
    length(b) == nftot || throw(ArgumentError(
        "MG saddle right-hand side has $(length(b)) entries, expected $nftot"))
    nf == size(mg.geo.Cus[end], 2) || throw(ArgumentError(
        "Kred has $nf rows but the MG fine level has $(size(mg.geo.Cus[end], 2)) rows"))
    # S3 stage C: the velocity operator is either the sparse `Kred` or, on the
    # fully matrix-free path, the leaf-local `Cu' K (Cu v)` apply.
    Kred === nothing && applyK === nothing &&
        throw(ArgumentError("semquad_mg_saddle_solve needs Kred or applyK"))
    applyK! = applyK === nothing ? (out, v) -> mul!(out, Kred, v) : applyK
    cgiters = Ref(0)
    # Reused scratch: FGMRES copies the returned vectors into Z/W immediately,
    # so returning the same buffer every call is safe and keeps the outer
    # iteration allocation-free on the large levels.
    wA = zeros(Float64, nftot)
    wP = zeros(Float64, nftot)
    zc = zeros(Float64, nf)
    rp = zeros(Float64, nfp)
    rp2 = zeros(Float64, nfp)
    s1 = zeros(Float64, nf)
    s2 = zeros(Float64, nf)
    ku = zeros(Float64, nf)
    cgwork = SemQuadCGWork(nf)
    function applyA(v::Vector{Float64})
        applyK!(ku, view(v, 1:nf))
        wA[1:nf] .= ku
        wA[1:nf] .+= Bt * v[(nf + 1):nftot]
        wA[(nf + 1):nftot] = Bs * v[1:nf]
        return wA
    end
    function precond(v::Vector{Float64})
        if vel_solver === :cg
            _, nc, _ = semquad_mg_cg(applyK!, v[1:nf], mg; tol=vel_tol,
                                     maxit=vel_maxit, work=cgwork)
            cgiters[] += nc
            copyto!(zc, cgwork.z)
        else
            zc .= semquad_mg_apply(mg, v[1:nf])
        end
        wP[1:nf] .= zc
        rp .= v[(nf + 1):nftot] .- Bs * zc
        _semquad_pressure_block!(wP, rp, rp2, s1, s2, Fp, pscale, nf)
        return wP
    end
    sol, relres, nit = fgmres_restarted(applyA, b; precond=precond, tol=tol,
                                        maxit=maxit, restart=restart,
                                        work=_semquad_fgmres_work(nftot,
                                                                  min(restart,
                                                                      maxit)))
    return sol, nit, relres, cgiters[]
end

"""
    _semquad_pressure_block!(wP, rp, rp2, s1, s2, Fp, pscale, nf)

Apply the pressure block of the block-triangular preconditioner to `rp` and
write `-pscale * Shat^{-1} rp` into `wP[(nf+1):end]`.

* `Fp isa NamedTuple` (`kind=:bfbt` or `:bfbt_m1`) is the three-factor form
  `Shat^{-1} = S1^{-1} S2 S1^{-1}`: one `F1` solve, then `S2` either as the
  precomputed sparse matrix or matrix-free as
  `Bs D^{-1} Kred D^{-1} Bt` (`Fp.S2 === nothing`), then a second `F1` solve;
* `kind=:pcd` is the pressure convection-diffusion form
  `Shat^{-1} = A_p^{-1} F_p M_p^{-1}` (three factors, `F_p = A_p` here because
  Stokes has no advection term);
* anything else is a plain Cholesky factorisation (`:mass`, `:stiffness`,
  `:lsc`).

All scratch is caller-provided, so the outer FGMRES stays allocation-free on the
large levels.
"""
@inline function _semquad_pressure_block!(wP::Vector{Float64},
                                          rp::Vector{Float64},
                                          rp2::Vector{Float64},
                                          s1::Vector{Float64},
                                          s2::Vector{Float64}, Fp, pscale::Float64,
                                          nf::Int)
    if Fp isa NamedTuple && Fp.kind === :pcd
        # PCD: Shat^{-1} = A_p^{-1} F_p M_p^{-1} (F_p = A_p for Stokes, see
        # `_semquad_build_pressure_precond`).  `rp` is consumed before it is
        # overwritten, so no length-`nfp` scratch is needed beyond `rp2`.
        rp2 .= Fp.FM \ rp
        mul!(rp, Fp.FF, rp2)
        rp2 .= Fp.FA \ rp
        @. wP[(nf + 1):end] = -pscale * rp2
    elseif Fp isa NamedTuple
        _semquad_s1_apply!(rp2, Fp, rp)
        if Fp.S2 === nothing
            # S2 matrix-free: Bs (D^{-1} (Kred (D^{-1} (Bt r))))
            mul!(s1, Fp.Bt, rp2)
            dinv = Fp.Dinvv
            @. s1 *= dinv
            if Fp.Kred === nothing
                Fp.applyK(s2, s1)
            else
                mul!(s2, Fp.Kred, s1)
            end
            @. s2 *= dinv
            mul!(rp, Fp.Bs, s2)
        else
            mul!(rp, Fp.S2, rp2)
        end
        _semquad_s1_apply!(rp2, Fp, rp)
        @. wP[(nf + 1):end] = -pscale * rp2
    else
        rp2 .= Fp \ rp
        @. wP[(nf + 1):end] = -pscale * rp2
    end
    return wP
end

"""
    _semquad_s1_apply!(x, Fp, b)

Apply the `S1^{-1}` factor of the w-BFBT block: the exact CHOLMOD solve when the
block carries no inner-solve object (the `:bfbt_m1` mass variant and any cached
block built before the object existed), otherwise the object's scheme.
"""
@inline function _semquad_s1_apply!(x::Vector{Float64}, Fp, b::Vector{Float64})
    S1i = haskey(Fp, :S1solve) ? Fp.S1solve : nothing
    if S1i === nothing
        ldiv!(x, Fp.F1, b)
    else
        _semquad_s1_solve!(x, S1i, b)
    end
    return x
end
