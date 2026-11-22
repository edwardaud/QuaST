# The optional `:balanced` AMR cooldown state of a checkpoint.
#
# The balanced cycle carries a hysteresis cooldown between steps (docs/dev/AMR.md
# section 4); a checkpoint has to store it or a restart would forget which leaves
# are cooling down.  The datasets are serialised as one deterministic 4 x n Int64
# matrix and validated on read, so a truncated or reordered file fails loudly
# instead of resuming an unknown hysteresis state.

# --- optional `:balanced` AMR cooldown state --------------------------------
# The cooldown is a Dict of leaf keys -> remaining cycles.  It is serialised as
# an integer matrix with four rows (level, ix, iy, remaining) and one column per
# key, sorted by key, so the file is deterministic; an empty dict writes no
# dataset.  The number of AMR cycles the state has been aged through is one
# integer attribute.  All of this is optional and absent from old checkpoints.

"""Serialise one cooldown dict to a deterministic 4 x n Int64 matrix."""
function _cooldown_matrix(d::Dict{NTuple{3,Int},Int})
    ks = sort(collect(d))
    M = Matrix{Int64}(undef, 4, length(ks))
    for (j, (k, v)) in enumerate(ks)
        M[1, j] = k[1]
        M[2, j] = k[2]
        M[3, j] = k[3]
        M[4, j] = v
    end
    return M
end

"""Rebuild one cooldown dict from a 4 x n integer matrix (validated)."""
function _cooldown_dict(M::AbstractMatrix)
    size(M, 1) == 4 ||
        throw(ArgumentError("cooldown matrix must have 4 rows " *
                            "(level, ix, iy, remaining), got $(size(M, 1))"))
    d = Dict{NTuple{3,Int},Int}()
    for j in 1:size(M, 2)
        l = Int(M[1, j]); ix = Int(M[2, j]); iy = Int(M[3, j])
        n = Int(M[4, j])
        (l >= 0 && ix >= 0 && iy >= 0 && n >= 0) ||
            throw(ArgumentError("cooldown matrix column $j is not a " *
                                "non-negative leaf/count: ($l, $ix, $iy, $n)"))
        d[(l, ix, iy)] = n
    end
    return d
end

"""Read the optional cooldown datasets/attribute of an open checkpoint."""
function _load_cooldown(f::HDF5.File)
    has_nr = haskey(f, "cooldown_no_refine")
    has_nc = haskey(f, "cooldown_no_coarsen")
    has_cyc = haskey(HDF5.attrs(f), "cooldown_cycle")
    # Old checkpoints have neither a dataset nor the attribute.  A new but
    # (temporarily) empty cooldown keeps its cycle count in the attribute, so a
    # file whose dicts are empty still restores the counter.
    (has_nr || has_nc || has_cyc) || return AmrCooldown()
    empty_d = Dict{NTuple{3,Int},Int}()
    nr = has_nr ? _cooldown_dict(read(f["cooldown_no_refine"])) : empty_d
    nc = has_nc ? _cooldown_dict(read(f["cooldown_no_coarsen"])) :
                  Dict{NTuple{3,Int},Int}()
    cyc = has_cyc ? Int(HDF5.attrs(f)["cooldown_cycle"]) : 0
    cyc >= 0 ||
        throw(ArgumentError("checkpoint cooldown_cycle=$cyc is negative"))
    return AmrCooldown(nr, nc, cyc)
end
