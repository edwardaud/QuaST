# The Picard iteration controls.
#
# The lagged viscosity and the strain-rate-dependent laws feed `eps -> eta -> u`
# back into each iteration, so the loop needs three things: a field blend for the
# averaged quantities, a divergence test for the adaptive relaxation, and the
# relaxation schedule itself.  All three are pure functions of their arguments.

"""
    blend_field(prev, new, relax)

Convex blend `(1-relax) prev + relax new`; `relax == 1` returns `new` exactly
(so the default path stays bit-identical), `relax == 0` keeps `prev`.
"""
function blend_field(prev::Vector{Float64}, new::Vector{Float64}, relax::Float64)
    0.0 <= relax <= 1.0 || throw(ArgumentError("relax must be in [0,1]"))
    relax == 1.0 && return new
    relax == 0.0 && return prev
    length(prev) == length(new) ||
        throw(ArgumentError("blend_field: length mismatch"))
    return (1.0 - relax) .* prev .+ relax .* new
end

"""
    picard_diverging(changes; window=3, factor=10.0)

True when the Picard change has grown by more than `factor` for `window`
consecutive iterations, i.e. the fixed-point iteration is diverging rather than
stalling.  `changes` is the sequence of relative updates, oldest first.
"""
function picard_diverging(changes::AbstractVector{Float64};
                          window::Int=3, factor::Float64=10.0)
    window >= 1 || throw(ArgumentError("window must be >= 1"))
    factor > 1.0 || throw(ArgumentError("factor must be > 1"))
    length(changes) < window + 1 && return false
    last_n = changes[(end - window + 1):end]
    prev_n = changes[(end - window):(end - 1)]
    return all(last_n .> factor .* max.(prev_n, 0.0))
end

"""
    picard_adapt_relax(relax, prev_change, change; rmin=0.05, shrink=0.5,
                       grow=1.2, contract=0.7)

Adaptive under-relaxation for the Picard fixed point.  `relax` is the weight of
the newest iterate.  When the relative change did not contract (`change >=
prev_change`) the step halved (`shrink`), floored at `rmin`; when it contracted
by at least a factor `contract` the step grows by `grow`, capped at 1.  This is
the standard damped-fixed-point safeguard: a viscosity that depends strongly on
the strain rate makes the undamped map oscillate, and shrinking the step keeps
it in the contraction region.

Returns the new relaxation weight.
"""
function picard_adapt_relax(relax::Float64, prev_change::Float64,
                            change::Float64; rmin::Float64=0.05,
                            shrink::Float64=0.5, grow::Float64=1.2,
                            contract::Float64=0.7)
    0.0 < shrink < 1.0 || throw(ArgumentError("shrink must be in (0,1)"))
    grow > 1.0 || throw(ArgumentError("grow must be > 1"))
    0.0 < contract < 1.0 || throw(ArgumentError("contract must be in (0,1)"))
    0.0 < rmin <= 1.0 || throw(ArgumentError("rmin must be in (0,1]"))
    if change >= prev_change
        return max(rmin, shrink * relax)
    elseif change < contract * prev_change
        return min(1.0, grow * relax)
    end
    return relax
end
