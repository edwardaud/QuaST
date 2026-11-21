# Picard stabilization helpers for the visco-plastic rheology of this solver.
#
# The plastic branch depends on the strain-rate invariant, which in turn depends
# on the velocity being solved for.  Evaluating it from the previous iterate is
# the cheapest lag; ASPECT additionally averages the strain rate over the
# nonlinear iterations ("viscosity averaging") to damp the oscillation that a
# strongly strain-rate dependent viscosity produces.  `pic_eta_relax` here is
# that averaging weight (1 = use the newest field, 0 = freeze it).

"""
    semquad_strain_rate_field(sq, ux, uz)

Nodal second invariant of the strain-rate tensor, `sqrt(0.5(exx^2+ezz^2+2exz^2))`,
computed element-locally with each element's own derivative matrix.
"""
function semquad_strain_rate_field(sq::SemQuad, ux::Vector{Float64},
                                   uz::Vector{Float64})
    n = sq.p + 1
    out = zeros(Float64, sq.N)
    for lf in sq.leaves
        h = leaf_hy(sq, lf[1])
        f = 2.0 / h
        ids = sq.elem[lf]
        for j in 1:n, i in 1:n
            dux_dx = 0.0; dux_dy = 0.0; duz_dx = 0.0; duz_dy = 0.0
            for a in 1:n
                dux_dx += sq.D[i, a] * ux[ids[(j - 1) * n + a]]
                dux_dy += sq.D[j, a] * ux[ids[(a - 1) * n + i]]
                duz_dx += sq.D[i, a] * uz[ids[(j - 1) * n + a]]
                duz_dy += sq.D[j, a] * uz[ids[(a - 1) * n + i]]
            end
            exx = dux_dx * f
            ezz = duz_dy * f
            exz = 0.5 * (dux_dy * f + duz_dx * f)
            out[ids[(j - 1) * n + i]] = sqrt(0.5 * (exx^2 + ezz^2 + 2.0 * exz^2))
        end
    end
    return out
end

"""
    semquad_strain_rate_components(sq, ux, uz)

Nodal *components* of the strain-rate tensor of `(ux, uz)` on the scalar layout
of `sq`, returned as the named tuple `(exx, ezz, exz)`:

    exx = dux/dx,   ezz = duz/dy,   exz = (dux/dy + duz/dx)/2.

Computed element-locally with each element's own derivative matrix, the same
loop as [`semquad_strain_rate_field`](@ref), which only returns the second
invariant `sqrt(0.5(exx^2+ezz^2+2exz^2))`.  The two would be redundant if the
invariant were enough: the Maxwell stress history of docs/dev/PHASE2.md 2b
carries the tensor components, so the shear one in particular needs `exz`.

`semquad_strain_rate_field` scales *every* derivative with `leaf_hy` (written for
the square mesh); the components here scale the x derivative with `leaf_hx` and
the y derivative with `leaf_hy`, so they are also correct on a stretched box.
On the square mesh `leaf_hx == leaf_hy` and the invariant of these components is
the field above.
"""
function semquad_strain_rate_components(sq::SemQuad, ux::Vector{Float64},
                                        uz::Vector{Float64})
    n = sq.p + 1
    exx = zeros(Float64, sq.N)
    ezz = zeros(Float64, sq.N)
    exz = zeros(Float64, sq.N)
    for lf in sq.leaves
        fx = 2.0 / leaf_hx(sq, lf[1])
        fy = 2.0 / leaf_hy(sq, lf[1])
        ids = sq.elem[lf]
        for j in 1:n, i in 1:n
            dux_dx = 0.0; dux_dy = 0.0; duz_dx = 0.0; duz_dy = 0.0
            for a in 1:n
                dux_dx += sq.D[i, a] * ux[ids[(j - 1) * n + a]]
                dux_dy += sq.D[j, a] * ux[ids[(a - 1) * n + i]]
                duz_dx += sq.D[i, a] * uz[ids[(j - 1) * n + a]]
                duz_dy += sq.D[j, a] * uz[ids[(a - 1) * n + i]]
            end
            g = ids[(j - 1) * n + i]
            exx[g] = dux_dx * fx
            ezz[g] = duz_dy * fy
            exz[g] = 0.5 * (dux_dy * fy + duz_dx * fx)
        end
    end
    return (exx=exx, ezz=ezz, exz=exz)
end

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
