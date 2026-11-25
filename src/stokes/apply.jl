# ===========================================================================
# S6: one fork/join per level per sweep (`SemQuadParRegion`)
#
# `_semquad_td_run!` above forks and joins **per call**: `@sync` + `nt`
# `@spawn`s + one atomic hand-out cost 6-9 us regardless of the work.  On the
# levels above `SEMQUAD_MG_SERIAL_DOFS` a 6-level V-cycle still makes ~15 such
# calls (applies, Jacobi updates, subtracts, transfers) and pays that overhead
# for every one of them.
#
# A `SemQuadParRegion` keeps `nt-1` worker tasks alive for the *whole sequence*
# of parallel phases of one level block (one fork/join per region instead of one
# per call).  The calling task is worker 1: it publishes a phase, runs its own
# share of the chunks and then waits for the other workers to finish.  Between
# phases it runs the serial parts of the level (the sparse `Cu` products, the
# scatter) while the workers wait for the next phase.  The protocol is
#
#     coordinator (worker 1)              worker w = 2..nt
#     ----------------------              -----------------
#     publish(job, nc): job, nc, chunk=0, pub += 1
#     run chunks (dynamic hand-out)       wait pub > last (yield)
#     wait arrive == nt-1                 last = pub
#     arrive = 0                          run chunks (dynamic hand-out)
#                                         arrive += 1
#
# so a phase costs one atomic publish + one barrier instead of a task-creation
# region, and the chunk->worker assignment stays dynamic.  Which worker runs
# which chunk never changes an arithmetic result: every chunk of a phase writes
# only its own entries (or, for the apply, its own leaf columns, which are
# scattered serially in leaf order afterwards), and the *phase order* is exactly
# the call order of the per-call path.  The consolidated path is therefore
# bitwise identical to the per-call one (G223).
#
# `SEMQUAD_MG_PHASES` counts the phases executed (a diagnostic next to
# `SEMQUAD_MG_FORKS`).
const SEMQUAD_MG_PHASES = Ref(0)

"""
    SemQuadParRegion

Persistent worker set of the S6 consolidation; see the comment block above.
"""
mutable struct SemQuadParRegion
    nt::Int                      # workers, including the calling task (worker 1)
    pub::Threads.Atomic{Int}     # phases published so far
    arrive::Threads.Atomic{Int}  # workers that finished the current phase
    stop::Threads.Atomic{Int}    # region over: the workers return
    job::Base.RefValue{Any}      # current phase body, `f(c, w)`
    nc::Threads.Atomic{Int}      # chunks of the current phase
    chunk::Threads.Atomic{Int}   # dynamic hand-out of the current phase
    err::Base.RefValue{Any}      # first exception a worker caught (re-thrown)
end

function SemQuadParRegion(nt::Int)
    return SemQuadParRegion(nt, Threads.Atomic{Int}(0), Threads.Atomic{Int}(0),
                            Threads.Atomic{Int}(0), Ref{Any}(nothing),
                            Threads.Atomic{Int}(0), Threads.Atomic{Int}(0),
                            Ref{Any}(nothing))
end

"""Chunk loop of one phase on worker `w` (a function barrier for `f`)."""
@inline function _semquad_phase_chunks!(f::F, r::SemQuadParRegion, nc::Int,
                                        w::Int) where {F}
    while true
        c = Threads.atomic_add!(r.chunk, 1) + 1
        c > nc && break
        f(c, w)
    end
    return nothing
end

"""Worker loop of a region: run every published phase, then return on `stop`."""
function _semquad_region_worker!(r::SemQuadParRegion, w::Int)
    last = 0
    while true
        while (p = r.pub[]) == last
            r.stop[] != 0 && return nothing
            yield()                      # a co-scheduled worker must run
        end
        last = p
        # `pub` is incremented after `job`/`nc`/`chunk` are written (seq_cst),
        # so the phase data below is the one published with `last`.
        try
            _semquad_phase_chunks!(r.job[], r, r.nc[], w)
        catch err
            # Never leave the coordinator waiting for an arrival that will not
            # happen: record the first failure and still arrive.  The
            # coordinator re-throws it after the phase (G223 exercises this by
            # construction: a wrong worker/scratch pairing throws instead of
            # hanging).
            r.err[] === nothing && (r.err[] = err)
        end
        Threads.atomic_add!(r.arrive, 1)
    end
end

"""Run `body()` as worker 1 of `r`, with the region's workers alive."""
function _semquad_region_run!(body::F, r::SemQuadParRegion) where {F}
    if r.nt <= 1
        body()
        r.err[] === nothing || throw(r.err[])
        return nothing
    end
    r.pub[] = 0                       # workers start with no phase seen
    r.arrive[] = 0
    r.stop[] = 0
    r.err[] = nothing
    SEMQUAD_MG_FORKS[] += 1
    SEMQUAD_MG_FORK_TASKS[] += r.nt - 1
    Threads.@sync begin
        for w in 2:r.nt
            Threads.@spawn _semquad_region_worker!(r, w)
        end
        try
            body()
        finally
            r.stop[] = 1
        end
    end
    r.job[] = nothing                 # do not keep the last phase's vectors alive
    r.err[] === nothing || throw(r.err[])
    return nothing
end

"""Publish `f(c, w)` over `nc` chunks as the next phase of `r` and run it."""
function _semquad_phase!(r::SemQuadParRegion, f::F, nc::Int) where {F}
    if nc <= 1
        f(1, 1)
        return nothing
    end
    SEMQUAD_MG_PHASES[] += 1
    r.job[] = f
    r.nc[] = nc
    r.chunk[] = 0
    Threads.atomic_add!(r.pub, 1)
    _semquad_phase_chunks!(f, r, nc, 1)
    while r.arrive[] != r.nt - 1
        yield()
    end
    r.arrive[] = 0
    r.err[] === nothing || throw(r.err[])
    return nothing
end

"""Number of chunks a vector op of length `n` uses on `nt` workers."""
@inline function _semquad_td_nchunks(n::Int, nt::Int)
    return clamp(SEMQUAD_MG_CHUNKS_PER_THREAD * nt, 1, n)
end

"""
    SemQuadKPar

Threaded scratch of the leaf-blocked matrix-free apply (`mg_threads > 1`, S5-2).

The leaf list is cut into `nchunks` fixed contiguous blocks; block `c` computes
the *element* vectors of its leaves into the columns `bounds[c]:bounds[c+1]-1`
of `elx`/`elz`, using the scratch of the worker that happens to pull it
(`scr[w]`, one `SemQuadKScratch` per worker).  After the parallel pass the
element vectors are scattered into the node vector **serially, in leaf order**,
which is exactly the order the serial kernel accumulates in, so the threaded
apply is **bitwise identical** to the serial one (G190).  A merge of per-block
*node* buffers in block order -- the `assemble_threads` discipline -- would only
be deterministic, not bitwise: the summation order of a node shared by leaves in
different blocks changes with the block count.

`nchunks = SEMQUAD_MG_CHUNKS_PER_THREAD * nt` (capped at `nleaf`) and every
chunk *starts* on a `SEMQUAD_KBLK` block boundary so the lane table of
[`SemQuadEtaLane`](@ref) covers it exactly; the chunks are pulled dynamically by
the `nt` workers: on a hybrid CPU (P+E cores) or with SMT siblings the per-core
speed differs by up to 2-3x, and one chunk per thread
would let the slowest core set the wall time.  Which worker computes a leaf
never changes its element vectors, so the merge stays bitwise serial regardless
of the assignment.

`elx`/`elz` hold one column per leaf, so their size is `(p+1)^2 x nleaf` per
component (`4096` leaves at p=4: `25 x 4096` doubles each).
"""
struct SemQuadKPar
    nt::Int                      # workers (<= Threads.nthreads())
    nchunks::Int
    bounds::Vector{Int}          # length nchunks+1: chunk c is bounds[c]:bounds[c+1]-1
    scr::Vector{SemQuadKScratch} # one per worker
    elx::Matrix{Float64}         # (n*n, nleaf) leaf-local x outputs
    elz::Matrix{Float64}         # (n*n, nleaf) leaf-local z outputs
end

const SEMQUAD_MG_CHUNKS_PER_THREAD = 4

function SemQuadKPar(op::AbstractSemQuadKOp, nt::Int)
    nl = op.nleaf
    nt = clamp(nt, 1, max(nl, 1))
    BLK = SEMQUAD_KBLK
    # Chunk *starts* sit on the `SEMQUAD_KBLK` block grid, so a chunk's leaf
    # window is a whole number of lane-table blocks.
    # The partition stays contiguous, covers 1:nl and is still proportional to
    # the leaf count within one block; it decides only *which worker* computes a
    # leaf, and the element vectors are scattered in leaf order, so the values
    # do not depend on it (G190/G225).  `nchunks` itself is unchanged: the
    # region's phase protocol uses it, and a phase with one chunk runs inline
    # (below a 32-leaf level the chunks after the first are simply empty).
    nchunks = clamp(SEMQUAD_MG_CHUNKS_PER_THREAD * nt, 1, nl)
    bounds = Vector{Int}(undef, nchunks + 1)
    for c in 0:nchunks
        cut = min(cld((c * nl) ÷ nchunks, BLK) * BLK, nl)
        bounds[c + 1] = cut + 1
    end
    scr = [SemQuadKScratch(op.n, op.nq;
                           mapped = op isa SemQuadKOpMapped) for _ in 1:nt]
    return SemQuadKPar(nt, nchunks, bounds, scr, zeros(op.n * op.n, nl),
                       zeros(op.n * op.n, nl))
end

# `Val(n)` dispatch of the element-buffer kernel (same list as the serial path).
# Two concrete kernel families: the historical box body and the cut-5 mapped
# body; both write the same element-vector buffers in the same leaf order.
function _semquad_kn_blk_store_any!(op::AbstractSemQuadKOp,
                                    etav::AbstractVector{Float64},
                                    v::AbstractVector{Float64},
                                    scr::SemQuadKScratch, lo::Int, hi::Int,
                                    elx::Matrix{Float64}, elz::Matrix{Float64})
    n = op.n
    if op isa SemQuadKOpMapped
        n == 2 && return _semquad_kn_blk_mapped_store!(op, etav, v, scr, lo, hi, elx, elz, Val(2))
        n == 3 && return _semquad_kn_blk_mapped_store!(op, etav, v, scr, lo, hi, elx, elz, Val(3))
        n == 4 && return _semquad_kn_blk_mapped_store!(op, etav, v, scr, lo, hi, elx, elz, Val(4))
        n == 5 && return _semquad_kn_blk_mapped_store!(op, etav, v, scr, lo, hi, elx, elz, Val(5))
        n == 6 && return _semquad_kn_blk_mapped_store!(op, etav, v, scr, lo, hi, elx, elz, Val(6))
        n == 7 && return _semquad_kn_blk_mapped_store!(op, etav, v, scr, lo, hi, elx, elz, Val(7))
        n == 8 && return _semquad_kn_blk_mapped_store!(op, etav, v, scr, lo, hi, elx, elz, Val(8))
        return _semquad_kn_blk_mapped_store!(op, etav, v, scr, lo, hi, elx, elz, Val(n))
    end
    n == 2 && return _semquad_kn_blk_store!(op, etav, v, scr, lo, hi, elx, elz, Val(2))
    n == 3 && return _semquad_kn_blk_store!(op, etav, v, scr, lo, hi, elx, elz, Val(3))
    n == 4 && return _semquad_kn_blk_store!(op, etav, v, scr, lo, hi, elx, elz, Val(4))
    n == 5 && return _semquad_kn_blk_store!(op, etav, v, scr, lo, hi, elx, elz, Val(5))
    n == 6 && return _semquad_kn_blk_store!(op, etav, v, scr, lo, hi, elx, elz, Val(6))
    n == 7 && return _semquad_kn_blk_store!(op, etav, v, scr, lo, hi, elx, elz, Val(7))
    n == 8 && return _semquad_kn_blk_store!(op, etav, v, scr, lo, hi, elx, elz, Val(8))
    return _semquad_kn_blk_store!(op, etav, v, scr, lo, hi, elx, elz, Val(n))
end

"""
    _semquad_scatter_elts!(out, op, par, lo=1, hi=op.nleaf)

Serial scatter-add of the element vectors of the leaves `lo:hi` into the node
vector: `out[ids[idx, li]] += elx[idx, li]` in increasing `li`, then increasing
`idx`.  This is the exact accumulation order of the serial kernel.
"""
function _semquad_scatter_elts!(out::Vector{Float64}, op::AbstractSemQuadKOp,
                                par::SemQuadKPar, lo::Int=1, hi::Int=op.nleaf)
    n2 = op.n * op.n
    N_ = op.N
    ids = op.ids
    elx = par.elx; elz = par.elz
    @inbounds for li in lo:hi
        for idx in 1:n2
            g = ids[idx, li]
            out[g] += elx[idx, li]
            out[N_ + g] += elz[idx, li]
        end
    end
    return out
end

"""
    _semquad_apply_K_raw_threaded!(out, op, etav, v, par)

`out .= K * v` with the leaf-local work spread over `par.nchunks` fixed
contiguous leaf blocks (dynamic scheduling) and the scatter-add done serially in
leaf order.  Bitwise identical to `_semquad_apply_K_raw!`; see `SemQuadKPar`.
"""
function _semquad_apply_K_raw_threaded!(out::Vector{Float64}, op::AbstractSemQuadKOp,
                                        etav::AbstractVector{Float64},
                                        v::AbstractVector{Float64},
                                        par::SemQuadKPar)
    fill!(out, 0.0)
    nchunks = par.nchunks
    bounds = par.bounds
    scr = par.scr
    elx = par.elx; elz = par.elz
    if par.nt <= 1
        for c in 1:nchunks
            _semquad_kn_blk_store_any!(op, etav, v, scr[1], bounds[c],
                                       bounds[c + 1] - 1, elx, elz)
        end
    else
        SEMQUAD_MG_FORKS[] += 1
        SEMQUAD_MG_FORK_TASKS[] += par.nt
        counter = Threads.Atomic{Int}(0)
        Threads.@sync for w in 1:par.nt
            Threads.@spawn begin
                sw = scr[w]
                while true
                    c = Threads.atomic_add!(counter, 1) + 1
                    c > nchunks && break
                    _semquad_kn_blk_store_any!(op, etav, v, sw, bounds[c],
                                               bounds[c + 1] - 1, elx, elz)
                end
            end
        end
    end
    _semquad_scatter_elts!(out, op, par)
    return out
end

"""Leaf chunk `c` of the threaded apply with the worker's own scratch."""
@inline function _semquad_apply_leaf_chunk!(op::AbstractSemQuadKOp,
                                            etav::AbstractVector{Float64},
                                            v::AbstractVector{Float64},
                                            scr::SemQuadKScratch,
                                            par::SemQuadKPar, c::Int)
    b = par.bounds
    _semquad_kn_blk_store_any!(op, etav, v, scr, b[c], b[c + 1] - 1,
                               par.elx, par.elz)
    return nothing
end

"""
    _semquad_apply_K_raw_region!(out, op, etav, v, par, region)

`_semquad_apply_K_raw_threaded!` with its leaf phase published into an already
open `SemQuadParRegion` (S6).  The `fill!` and the serial leaf-order scatter stay
on the calling task; only the leaf compute becomes a phase.  Same chunks, same
kernel, same scatter order as the per-call form, so the result is bitwise equal
(G223); only the task creation moves out of the per-op path.
"""
function _semquad_apply_K_raw_region!(out::Vector{Float64}, op::AbstractSemQuadKOp,
                                      etav::AbstractVector{Float64},
                                      v::AbstractVector{Float64},
                                      par::SemQuadKPar,
                                      region::SemQuadParRegion)
    # One scratch slot per region worker: the region must not have more workers
    # than the apply's scratch set (`_semquad_mg_vcycle_region!` builds it as
    # `SemQuadParRegion(par.nt)`; a level with fewer leaves than `mg.threads`
    # clamps `par.nt`).
    region.nt <= par.nt ||
        throw(ArgumentError("region has $(region.nt) workers but the apply " *
                            "has $(par.nt) scratch slots"))
    fill!(out, 0.0)
    _semquad_phase!(region,
                    (c, w) -> _semquad_apply_leaf_chunk!(op, etav, v, par.scr[w],
                                                         par, c), par.nchunks)
    _semquad_scatter_elts!(out, op, par)
    return out
end

"""
    _semquad_spmv_row_chunk!(y, At, x, lo, hi)

`y[lo:hi]` of `y = A * x`, `A` given by its transposed CSC: row `i` is a dot
product over column `i` of `At`, accumulated in increasing column order of `A`
with the same `muladd`s as sparse `mul!` and starting from `0.0`.  Sparse `mul!`
starts every output entry from zero (`_rmul_or_fill!`), so assigning the
accumulated value is bitwise: the `1.0` `muladd` reproduces `mul!`'s
`tmp + C[i]` (the accumulator cannot be `-0.0` when it starts at `+0.0`).
"""
@inline function _semquad_spmv_row_chunk!(y::Vector{Float64},
                                          At::SparseMatrixCSC{Float64,Int},
                                          x::AbstractVector{Float64},
                                          lo::Int, hi::Int)
    @inbounds for i in lo:hi
        tmp = 0.0
        for k in nzrange(At, i)
            tmp = muladd(At.nzval[k], x[At.rowval[k]], tmp)
        end
        y[i] = muladd(tmp, 1.0, 0.0)
    end
    return nothing
end

"""
    _semquad_spmv_at_chunk!(y, A, x, lo, hi)

`y[lo:hi]` of `y = A' * x`: output entry `j` is the dot product over column `j`
of `A` in `nzrange` order, the same arithmetic as the `_At_or_Ac_mul_B!` kernel
`mul!` uses.  No fill is needed because every output entry is assigned.
"""
@inline function _semquad_spmv_at_chunk!(y::Vector{Float64},
                                         A::SparseMatrixCSC{Float64,Int},
                                         x::AbstractVector{Float64},
                                         lo::Int, hi::Int)
    @inbounds for j in lo:hi
        tmp = 0.0
        for k in nzrange(A, j)
            tmp = muladd(A.nzval[k], x[A.rowval[k]], tmp)
        end
        y[j] = muladd(tmp, 1.0, 0.0)
    end
    return nothing
end

"""
    _semquad_spmv_region!(y, At, x, nt, region)

`y .= A * x` split over the workers of an already open region, with `A` given by
its transposed CSC `At` (so each output row is one column dot product of `At`).
Rows are independent and `_semquad_spmv_row_chunk!` is the `mul!` arithmetic, so
the result is bitwise the sparse `mul!` (G224).  Every output entry is assigned,
so the phase needs no `fill!` first.
"""
function _semquad_spmv_region!(y::Vector{Float64},
                               At::SparseMatrixCSC{Float64,Int},
                               x::AbstractVector{Float64}, nt::Int,
                               region::SemQuadParRegion)
    n = length(y)
    nc = _semquad_td_nchunks(n, nt)
    _semquad_phase!(region,
                    (c, w) -> _semquad_spmv_row_chunk!(y, At, x,
                                                       _semquad_td_lo(n, nc, c),
                                                       _semquad_td_hi(n, nc, c)),
                    nc)
    return y
end

"""
    _semquad_spmv_at_region!(y, A, x, nt, region)

`y .= A' * x` split over the workers of an already open region: output entry `j`
is the dot product over column `j` of `A`, the `_At_or_Ac_mul_B!` arithmetic of
sparse `mul!` (G224).  Every output entry is assigned, so the phase needs no
`fill!` first.
"""
function _semquad_spmv_at_region!(y::Vector{Float64},
                                  A::SparseMatrixCSC{Float64,Int},
                                  x::AbstractVector{Float64}, nt::Int,
                                  region::SemQuadParRegion)
    n = length(y)
    nc = _semquad_td_nchunks(n, nt)
    _semquad_phase!(region,
                    (c, w) -> _semquad_spmv_at_chunk!(y, A, x,
                                                      _semquad_td_lo(n, nc, c),
                                                      _semquad_td_hi(n, nc, c)),
                    nc)
    return y
end

"""
    _semquad_apply_Kred_region!(out, op, Cu, Cut, etav, v, work, par, region)

Constrained apply `out .= Cu' * K * (Cu * v)` inside an open region (the S6
analogue of `_semquad_apply_Kred_threaded!`).  With `SEMQUAD_MG_REGION_CU[]`
(the default) the two sparse `Cu` products are region phases too -- `Cu * v` as
a row gather over `Cut`, `Cu' * y` over the columns of `Cu` -- and the leaf apply
is the existing leaf phase.  The products are output-entry independent and keep
the `mul!` arithmetic, so the result is bitwise both the serial and the
serial-`Cu`/threaded-leaf form (G224).  `SEMQUAD_MG_REGION_CU[] = false` calls
`mul!` on the calling task for the A/B.
"""
function _semquad_apply_Kred_region!(out::Vector{Float64}, op::AbstractSemQuadKOp,
                                     Cu::SparseMatrixCSC{Float64,Int},
                                     Cut::SparseMatrixCSC{Float64,Int},
                                     etav::AbstractVector{Float64},
                                     v::AbstractVector{Float64},
                                     work::SemQuadKWork, par::SemQuadKPar,
                                     region::SemQuadParRegion)
    if SEMQUAD_MG_REGION_CU[]
        _semquad_spmv_region!(work.full, Cut, v, par.nt, region)
    else
        mul!(work.full, Cu, v)
    end
    _semquad_apply_K_raw_region!(work.op_out, op, etav, work.full, par, region)
    if SEMQUAD_MG_REGION_CU[]
        _semquad_spmv_at_region!(out, Cu, work.op_out, par.nt, region)
    else
        mul!(out, transpose(Cu), work.op_out)
    end
    return out
end

"""
    _semquad_apply_Kred_threaded!(out, op, Cu, etav, v, work, par)

Constrained threaded apply `out .= Cu' * K * (Cu * v)` (the `mg_threads > 1`
analogue of `_semquad_apply_Kred!`).  The two sparse `Cu` products stay serial
(they are cheap next to the leaf loop and sparse `mul!` is already
deterministic); only the leaf-local part is threaded, and it is bitwise
identical to the serial one.
"""
function _semquad_apply_Kred_threaded!(out::Vector{Float64}, op::AbstractSemQuadKOp,
                                       Cu::SparseMatrixCSC{Float64,Int},
                                       etav::AbstractVector{Float64},
                                       v::AbstractVector{Float64},
                                       work::SemQuadKWork, par::SemQuadKPar)
    mul!(work.full, Cu, v)
    _semquad_apply_K_raw_threaded!(work.op_out, op, etav, work.full, par)
    mul!(out, transpose(Cu), work.op_out)
    return out
end

"""
    _semquad_apply_K_raw!(out, op, etav, v, work; blk=op.blk)

Raw matrix-free apply of the unconstrained velocity stiffness: `out = K * v`,
with `v` the full `2*N` velocity vector (x block first, then z block).  `etav`
is the viscosity at every leaf/quadrature point of `op`.

`blk > 0` selects the leaf-blocked kernel (stage D): `SEMQUAD_KBLK` leaves per
pass with the lane index innermost.  The width is a compile-time constant, so
`blk` only enables the blocked path (any positive value is the same width); it
is not a per-call batch size.  `blk=0` selects the historical scalar kernel,
which is what the correctness gate compares against.
"""
function _semquad_apply_K_raw!(out::Vector{Float64}, op::SemQuadKOp,
                              etav::AbstractVector{Float64},
                              v::AbstractVector{Float64}, work::SemQuadKWork;
                              blk::Int=op.blk)
    nb = (blk > 0 && op.nleaf >= 2) ? blk : 0
    n = op.n
    nb > 0 && return _semquad_apply_K_blk!(out, op, etav, v, work, nb)
    n == 2 && return _semquad_kn!(out, op, etav, v, work, Val(2))
    n == 3 && return _semquad_kn!(out, op, etav, v, work, Val(3))
    n == 4 && return _semquad_kn!(out, op, etav, v, work, Val(4))
    n == 5 && return _semquad_kn!(out, op, etav, v, work, Val(5))
    n == 6 && return _semquad_kn!(out, op, etav, v, work, Val(6))
    n == 7 && return _semquad_kn!(out, op, etav, v, work, Val(7))
    n == 8 && return _semquad_kn!(out, op, etav, v, work, Val(8))
    return _semquad_kn!(out, op, etav, v, work, Val(n))   # rare: dynamic dispatch
end

function _semquad_kn!(out::Vector{Float64}, op::SemQuadKOp,
                      etav::AbstractVector{Float64},
                      v::AbstractVector{Float64}, work::SemQuadKWork,
                      ::Val{N}) where {N}
    nq = N + 1
    N_ = op.N
    phi = op.phi; dphi = op.dphi; phiT = op.phiT; dphiT = op.dphiT
    gl_w = op.gl_w
    ids = op.ids
    a2 = op.deviatoric ? 4.0 / 3.0 : 2.0
    c2 = op.deviatoric ? 2.0 / 3.0 : 0.0
    xv = work.xv; zv = work.zv
    uxx = work.uxx; uxy = work.uxy; uzx = work.uzx; uzy = work.uzy
    t1 = work.t1; t2 = work.t2
    F = work.F; ox = work.ox; oz = work.oz
    fill!(out, 0.0)
    nq2 = nq * nq
    @inbounds for li in 1:op.nleaf
        ihx = op.ihx[li]; ihy = op.ihy[li]; wf = op.wfac[li]
        for j in 1:N, i in 1:N
            g = ids[(j - 1) * N + i, li]
            xv[i, j] = v[g]
            zv[i, j] = v[N_ + g]
        end
        # velocity gradients at the Gauss points (2/h carried by sb/sc)
        _semquad_contract!(uxx, xv, dphi, phi, ihx, 1.0, t1, Val(N))
        _semquad_contract!(uxy, xv, phi, dphi, 1.0, ihy, t1, Val(N))
        _semquad_contract!(uzx, zv, dphi, phi, ihx, 1.0, t1, Val(N))
        _semquad_contract!(uzy, zv, phi, dphi, 1.0, ihy, t1, Val(N))
        fill!(ox, 0.0); fill!(oz, 0.0)
        # The assembler's local blocks are
        #   Kxx = a2 gx gx' + gz gz',      Kzz = a2 gz gz' + gx gx'
        #   Kxz = gz gx' - c2 gx gz',      Kzx = gx gz' - c2 gz gx'
        # (c2 = 2/3 for the deviatoric form, 0 otherwise), and each output
        # splits into two separable pieces:
        #   out_x = Dx*(W.ihx.*(a2 ux_x - c2 uz_y))*Ey' + Ex*(W.ihy.*(ux_y + uz_x))*Dy'
        #   out_z = Ex*(W.ihy.*(a2 uz_y - c2 ux_x))*Dy' + Dx*(W.ihx.*(uz_x + ux_y))*Ex'
        k0 = (li - 1) * nq2
        k = k0
        for qy in 1:nq, qx in 1:nq
            k += 1
            cw1 = wf * gl_w[qx] * gl_w[qy] * etav[k] * ihx
            F[qx, qy] = cw1 * (a2 * uxx[qx, qy] - c2 * uzy[qx, qy])
        end
        _semquad_accum!(ox, dphi, F, phiT, t1, Val(N))
        k = k0
        for qy in 1:nq, qx in 1:nq
            k += 1
            cw2 = wf * gl_w[qx] * gl_w[qy] * etav[k] * ihy
            F[qx, qy] = cw2 * (uxy[qx, qy] + uzx[qx, qy])
        end
        _semquad_accum!(ox, phi, F, dphiT, t2, Val(N))
        k = k0
        for qy in 1:nq, qx in 1:nq
            k += 1
            cw2 = wf * gl_w[qx] * gl_w[qy] * etav[k] * ihy
            F[qx, qy] = cw2 * (a2 * uzy[qx, qy] - c2 * uxx[qx, qy])
        end
        _semquad_accum!(oz, phi, F, dphiT, t2, Val(N))
        k = k0
        for qy in 1:nq, qx in 1:nq
            k += 1
            cw1 = wf * gl_w[qx] * gl_w[qy] * etav[k] * ihx
            F[qx, qy] = cw1 * (uzx[qx, qy] + uxy[qx, qy])
        end
        _semquad_accum!(oz, dphi, F, phiT, t1, Val(N))
        for j in 1:N, i in 1:N
            g = ids[(j - 1) * N + i, li]
            out[g] += ox[i, j]
            out[N_ + g] += oz[i, j]
        end
    end
    return out
end

"""
    _semquad_kn_mapped!(out, op, etav, v, work, ::Val{N})

Scalar matrix-free velocity stiffness for the **mapped** operator (cut 5).  The
correctness twin of the blocked mapped kernel
[`_semquad_kn_blk_mapped!`](@ref): it does the four reference-derivative
contractions, the per-quadrature-point chain rule with the cached `J^{-T}`, the
design's four weighted reference-space pieces and the four accumulations, leaf by
leaf.  The box operator is never dispatched here (GEOM_DESIGN §4.3 F5).
"""
function _semquad_kn_mapped!(out::Vector{Float64}, op::SemQuadKOpMapped,
                             etav::AbstractVector{Float64},
                             v::AbstractVector{Float64}, work::SemQuadKWork,
                             ::Val{N}) where {N}
    nq = N + 1
    N_ = op.N
    nq2 = nq * nq
    phi = op.phi; dphi = op.dphi; phiT = op.phiT; dphiT = op.dphiT
    gl_w = op.gl_w
    ids = op.ids
    a2 = op.deviatoric ? 4.0 / 3.0 : 2.0
    c2 = op.deviatoric ? 2.0 / 3.0 : 0.0
    xv = work.xv; zv = work.zv
    uxi_x = work.uxx; ueta_x = work.uxy
    uxi_z = work.uzx; ueta_z = work.uzy
    t1 = work.t1
    Fx1 = work.F
    Fx2 = work.F2::Matrix{Float64}
    Fz1 = work.F3::Matrix{Float64}
    Fz2 = work.F4::Matrix{Float64}
    ox = work.ox; oz = work.oz
    detJ = op.detJ
    j11v = op.j11; j12v = op.j12; j21v = op.j21; j22v = op.j22
    fill!(out, 0.0)
    @inbounds for li in 1:op.nleaf
        for j in 1:N, i in 1:N
            g = ids[(j - 1) * N + i, li]
            xv[i, j] = v[g]
            zv[i, j] = v[N_ + g]
        end
        _semquad_contract!(uxi_x, xv, dphi, phi, 1.0, 1.0, t1, Val(N))
        _semquad_contract!(ueta_x, xv, phi, dphi, 1.0, 1.0, t1, Val(N))
        _semquad_contract!(uxi_z, zv, dphi, phi, 1.0, 1.0, t1, Val(N))
        _semquad_contract!(ueta_z, zv, phi, dphi, 1.0, 1.0, t1, Val(N))
        k0 = (li - 1) * nq2
        k = k0
        for qy in 1:nq, qx in 1:nq
            k += 1
            W = detJ[k] * gl_w[qx] * gl_w[qy] * etav[k]
            j11 = j11v[k]; j12 = j12v[k]
            j21 = j21v[k]; j22 = j22v[k]
            ax = uxi_x[qx, qy]; bx = ueta_x[qx, qy]
            az = uxi_z[qx, qy]; bz = ueta_z[qx, qy]
            uxx = j11 * ax + j12 * bx
            uxy = j21 * ax + j22 * bx
            uzx = j11 * az + j12 * bz
            uzy = j21 * az + j22 * bz
            A = W * (a2 * uxx - c2 * uzy)
            Bq = W * (uxy + uzx)
            Az = W * (a2 * uzy - c2 * uxx)
            Fx1[qx, qy] = A * j11 + Bq * j21
            Fx2[qx, qy] = A * j12 + Bq * j22
            Fz1[qx, qy] = Bq * j11 + Az * j21
            Fz2[qx, qy] = Bq * j12 + Az * j22
        end
        fill!(ox, 0.0); fill!(oz, 0.0)
        _semquad_accum!(ox, dphi, Fx1, phiT, t1, Val(N))
        _semquad_accum!(ox, phi, Fx2, dphiT, t1, Val(N))
        _semquad_accum!(oz, dphi, Fz1, phiT, t1, Val(N))
        _semquad_accum!(oz, phi, Fz2, dphiT, t1, Val(N))
        for j in 1:N, i in 1:N
            g = ids[(j - 1) * N + i, li]
            out[g] += ox[i, j]
            out[N_ + g] += oz[i, j]
        end
    end
    return out
end

"""
    _semquad_apply_K_raw!(out, op::SemQuadKOpMapped, etav, v, work)

Mapped counterpart of the raw apply: `out = K * v` with the cut-5 mapped kernel
family.  The blocked variant is the default (`blk > 0`), exactly as on the box.
"""
function _semquad_apply_K_raw!(out::Vector{Float64}, op::SemQuadKOpMapped,
                              etav::AbstractVector{Float64},
                              v::AbstractVector{Float64}, work::SemQuadKWork;
                              blk::Int=op.blk)
    nb = (blk > 0 && op.nleaf >= 2) ? blk : 0
    n = op.n
    nb > 0 && return _semquad_apply_K_blk_mapped!(out, op, etav, v, work, nb)
    n == 2 && return _semquad_kn_mapped!(out, op, etav, v, work, Val(2))
    n == 3 && return _semquad_kn_mapped!(out, op, etav, v, work, Val(3))
    n == 4 && return _semquad_kn_mapped!(out, op, etav, v, work, Val(4))
    n == 5 && return _semquad_kn_mapped!(out, op, etav, v, work, Val(5))
    n == 6 && return _semquad_kn_mapped!(out, op, etav, v, work, Val(6))
    n == 7 && return _semquad_kn_mapped!(out, op, etav, v, work, Val(7))
    n == 8 && return _semquad_kn_mapped!(out, op, etav, v, work, Val(8))
    return _semquad_kn_mapped!(out, op, etav, v, work, Val(n))
end

# `Val(n)` dispatch of the serial blocked mapped kernel.
function _semquad_apply_K_blk_mapped!(out::Vector{Float64},
                                     op::SemQuadKOpMapped,
                                     etav::AbstractVector{Float64},
                                     v::AbstractVector{Float64},
                                     work::SemQuadKWork, nb::Int)
    n = op.n
    n == 2 && return _semquad_kn_blk_mapped!(out, op, etav, v, work, nb, Val(2))
    n == 3 && return _semquad_kn_blk_mapped!(out, op, etav, v, work, nb, Val(3))
    n == 4 && return _semquad_kn_blk_mapped!(out, op, etav, v, work, nb, Val(4))
    n == 5 && return _semquad_kn_blk_mapped!(out, op, etav, v, work, nb, Val(5))
    n == 6 && return _semquad_kn_blk_mapped!(out, op, etav, v, work, nb, Val(6))
    n == 7 && return _semquad_kn_blk_mapped!(out, op, etav, v, work, nb, Val(7))
    n == 8 && return _semquad_kn_blk_mapped!(out, op, etav, v, work, nb, Val(8))
    return _semquad_kn_blk_mapped!(out, op, etav, v, work, nb, Val(n))
end

"""
    semquad_apply_K!(out, ss, qt, eta, v; deviatoric=false, work=nothing)

Matrix-free `out .= K(eta) * v` for the **unconstrained** quadtree velocity
stiffness on `qt` (the operator `solve_semquad_stokes_var` assembles), with `v`
the full `2*N` velocity vector (x block first, then z block).

`eta` is either a function of `(x, y)` -- evaluated at every leaf/quadrature
point on every call -- or a precomputed `Vector` from
`semquad_eta_field(op, eta_fun)` in the ordering of `semquad_kop(ss, qt)`, which
is what the MG smoother uses so the viscosity is sampled once per operator.
`work` reuses the scratch buffers; passing `nothing` allocates them.

The quadrature rule and the viscosity sampling points are the assembler's, so
`out` equals the sparse `K * v` to round-off for any `eta` (G175).
"""
function semquad_apply_K!(out::AbstractVector{Float64}, ss::SemStokes,
                          qt::Quadtree, eta, v::AbstractVector{Float64};
                          deviatoric::Bool=false, work=nothing)
    op = semquad_kop(ss, qt; deviatoric=deviatoric)
    length(v) == 2 * op.N || throw(ArgumentError(
        "v has $(length(v)) entries, expected 2*N = $(2 * op.N)"))
    length(out) == 2 * op.N || throw(ArgumentError(
        "out has $(length(out)) entries, expected 2*N = $(2 * op.N)"))
    w = work === nothing ? SemQuadKWork(op) : work
    etav = _semquad_resolve_eta(op, eta)
    return _semquad_apply_K_raw!(out, op, etav, v, w)
end

"""
    semquad_apply_Kred!(out, ss, qt, Cu, eta, v; deviatoric=false, work=nothing)

Constrained apply `out .= Cu' * K * (Cu * v)` on the reduced (free-dof) velocity
space, i.e. exactly `Kred * v` without ever forming `Kred = Cu' * K * Cu`.  `Cu`
is the velocity affine-constraint matrix returned by
`solve_semquad_stokes_var(...; assemble_only=true)` / the MG geometry; it has at
most `p+1` nonzeros per constrained row, so the two sparse products are
negligible next to the leaf loop.
"""
function semquad_apply_Kred!(out::AbstractVector{Float64}, ss::SemStokes,
                             qt::Quadtree, Cu::SparseMatrixCSC{Float64,Int},
                             eta, v::AbstractVector{Float64};
                             deviatoric::Bool=false, work=nothing)
    op = semquad_kop(ss, qt; deviatoric=deviatoric)
    N2 = 2 * op.N
    w = work === nothing ? SemQuadKWork(op) : work
    nf = size(Cu, 2)
    length(v) == nf || throw(ArgumentError(
        "v has $(length(v)) entries, expected $(nf) free dofs"))
    length(out) == nf || throw(ArgumentError(
        "out has $(length(out)) entries, expected $(nf) free dofs"))
    etav = _semquad_resolve_eta(op, eta)
    return _semquad_apply_Kred!(out, op, Cu, etav, v, w)
end

"""
    _semquad_apply_Kred!(out, op, Cu, etav, v, work)

Allocation-free core of `semquad_apply_Kred!`, taking the resolved operator and
viscosity field.  This is what the matrix-free multigrid levels call.
"""
function _semquad_apply_Kred!(out::Vector{Float64}, op::AbstractSemQuadKOp,
                              Cu::SparseMatrixCSC{Float64,Int},
                              etav::AbstractVector{Float64},
                              v::AbstractVector{Float64}, work::SemQuadKWork)
    mul!(work.full, Cu, v)                     # full-space constrained prolongation
    _semquad_apply_K_raw!(work.op_out, op, etav, work.full, work)
    mul!(out, transpose(Cu), work.op_out)
    return out
end

