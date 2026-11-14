# Threading utilities shared by the leaf-blocked assembly loops (S4 stage 1).
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

