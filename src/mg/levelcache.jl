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
    SemQuadMGLevelKey(idx, nleaf, stamp, p, nx0, ny0, Lx, bc, deviatoric, wall_bc,
                      geom)

Cache key of one coarse MG level: the level index (coarse first) plus the
fingerprint of the level's **own** leaf set (`_semquad_mg_leaf_hash`) and the
settings that change the level objects for the same leaf set.  Two levels with
the same leaves and the same settings always get the same key, and the key does
not depend on anything sampled at build time (eta) -- see the block comment
above.

Cut 5 adds the geometry stamp: the level stores `SemStokes` spaces that carry
the geometry, and the mapped assembly of a level is a function of it, so a box
and an annulus with the same coarse leaf set must never share a cached level
(GEOM_DESIGN §2.5).  The stamp is `geometry_stamp(g)`, the same identity every
cut-5 cache key uses.
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
    wall_bc::NTuple{4,Symbol}
    geom::UInt64
end

Base.:(==)(a::SemQuadMGLevelKey, b::SemQuadMGLevelKey) =
    a.idx == b.idx && a.nleaf == b.nleaf && a.stamp == b.stamp &&
    a.p == b.p && a.nx0 == b.nx0 && a.ny0 == b.ny0 && a.Lx == b.Lx &&
    a.bc === b.bc && a.deviatoric == b.deviatoric && a.wall_bc == b.wall_bc &&
    a.geom == b.geom
Base.hash(k::SemQuadMGLevelKey, h::UInt) =
    hash(k.geom, hash(k.stamp, hash(k.idx, hash(k.nleaf, hash(k.p, hash(k.bc,
              hash(k.wall_bc, hash(k.Lx, h))))))))

function Base.show(io::IO, k::SemQuadMGLevelKey)
    print(io, "SemQuadMGLevelKey(level=", k.idx, ", leaves=", k.nleaf,
          ", stamp=", string(k.stamp; base=16),
          ", geom=", string(k.geom; base=16), ")")
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
per build; the benchmark in `benchmarks/bench_mg_level_cache.jl` reads them).
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
separately in the cache key.  Cut 5: the function is `_leafset_hash` in the
geometry layer now (one definition, shared with the metric cache); this name is
kept because the MG cache keys and G221 use it.
"""
function _semquad_mg_leaf_hash(leaves)::UInt64
    return _leafset_hash(leaves)
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
                              deviatoric::Bool, geom_assembly::Symbol;
                              wall_bc::NTuple{4,Symbol}=(:same, :same, :same, :same))
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
                                       wall_bc=wall_bc,
                                       linear=:mg, mg_operator=:matrixfree,
                                       assemble_only=:mg)
    else
        # `:full` is the historical K-forming assembly; it exists for the A/B
        # and the bitwise gate, not for production.
        asm = solve_semquad_stokes_var(s, q, one1, z2, z2, z2, z2;
                                       bc=bc, deviatoric=deviatoric,
                                       wall_bc=wall_bc,
                                       assemble_only=true)
    end
    return SparseMatrixCSC{Float64,Int}(sparse(asm.Cu))
end

