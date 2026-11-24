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
            _semquad_apply_Kred_threaded!(y, mg.mfops[lev]::AnySemQuadKOp,
                                          mg.geo.Cus[lev],
                                          mg.mfetas[lev]::SemQuadEtaLane, x,
                                          mg.mfwork[lev]::SemQuadKWork,
                                          mg.par[lev]::SemQuadKPar)
        else
            _semquad_apply_Kred_region!(y, mg.mfops[lev]::AnySemQuadKOp,
                                        mg.geo.Cus[lev],
                                        mg.geo.Cuts[lev],
                                        mg.mfetas[lev]::SemQuadEtaLane, x,
                                        mg.mfwork[lev]::SemQuadKWork,
                                        mg.par[lev]::SemQuadKPar, region)
        end
    else
        _semquad_apply_Kred!(y, mg.mfops[lev]::AnySemQuadKOp, mg.geo.Cus[lev],
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


