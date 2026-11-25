# ---------------------------------------------------------------------------
# Named input guards (remediation batch P0 part 1: AUDIT_VV F5/F6, AUDIT_BUGS R5)
#
# Every function here is a *cold* throw path wrapped around an allocation-free
# scan: the success branch is one comparison per element (or per quadrature
# point), so the box hot path stays bitwise identical and allocation-identical
# (the A/B against the parent commit measures 0 B extra in `geom3`/`geom4`).
#
# The contract these helpers implement: a non-finite or sign-invalid *input
# field* is refused at the earliest assembly/driver entry, with the offending
# field named and the value (and, where one is known, the coordinate)
# reported.  Before this batch the same inputs silently produced a NaN state,
# or a `SingularException` from `\` several seconds into a solve with no clue
# which input was responsible (AUDIT_VV F5/F6, `robustness.log` lines 14-33).
# ---------------------------------------------------------------------------

"""Index of the first non-finite entry of `v`, or `0` when all entries are finite.

Allocation-free: plain indexed loop over a `Float64` vector (`all(isfinite, v)`
is also allocation-free for dense vectors, but this variant also returns the
position for the error message).
"""
@inline function _first_nonfinite(v::AbstractVector{Float64})
    @inbounds for i in eachindex(v)
        isfinite(v[i]) || return Int(i)
    end
    return 0
end

"""
    _require_finite!(field, name, where)

Throw an `ArgumentError` naming `name` if `field` contains a NaN or an Inf.
`where` says which entry (driver argument, assembly, checkpoint, ...) supplied
the field, so the message points at the caller's input rather than at the
solver's internals.  Returns `nothing` on the success path.
"""
function _require_finite!(field::AbstractVector{Float64}, name::AbstractString,
                          where::AbstractString)
    i = _first_nonfinite(field)
    i == 0 && return nothing
    throw(ArgumentError("$name contains a non-finite value " *
                        "(got $(field[i]) at index $i): $where must be " *
                        "finite.  A NaN or Inf here propagates into the " *
                        "assembled operator and the linear solve and would " *
                        "otherwise surface as a NaN state or a bare " *
                        "SingularException.  Find the closure, initial " *
                        "condition or checkpoint that produced it."))
end

"""
    _require_finite_value!(x, name, where)

Scalar variant of [`_require_finite!`](@ref) for one assembled entry (a load at
a quadrature point, a wall datum, ...).
"""
@inline function _require_finite_value!(x::Float64, name::AbstractString,
                                        where::AbstractString)
    isfinite(x) && return nothing
    throw(ArgumentError("$name is non-finite (got $x) at $where.  The " *
                        "assembled system must not be built from a NaN or " *
                        "Inf input; find the closure that returned it."))
end

"""
    _require_eta!(eta_value, x, y)

Refuse a non-positive or non-finite viscosity sampled from the raw solve API's
`eta_fun` (AUDIT_VV F6).  The production material laws clamp to a positive
`eta_min`, so no production path can hit this; a hand-written closure passed
straight to `solve_semquad_stokes_var` can, and before this guard it was
accepted silently (`eta = -1` returned a finite field at residual 1.6e-15).
"""
@inline function _require_eta!(eta_value::Float64, x::Float64, y::Float64)
    (isfinite(eta_value) && eta_value > 0.0) && return nothing
    local why
    if !isfinite(eta_value)
        why = "is non-finite"
    else
        why = "is non-positive"
    end
    throw(ArgumentError("the viscosity field eta(x, y) $why (got " *
                        "$(eta_value)) at (x=$(x), y=$(y)).  The raw " *
                        "solve API requires a finite, strictly positive " *
                        "viscosity: a non-positive viscosity makes the " *
                        "velocity stiffness indefinite or singular and " *
                        "silently returns a wrong or NaN field.  Production " *
                        "laws clamp to a positive eta_min; a hand-written " *
                        "closure must do the same."))
end
"""
    _require_load!(x, name, xpos, ypos)

Cold-path guard for one assembled load sample: `x` must be finite.  The sampled
coordinates are passed as numbers (not as a pre-built string!) so the *success*
path allocates nothing -- an interpolated message argument would allocate at
every quadrature point and break the box allocation contract.
"""
@inline function _require_load!(x::Float64, name::AbstractString,
                                xpos::Float64, ypos::Float64)
    isfinite(x) && return nothing
    throw(ArgumentError("$name is non-finite (got $x) at (x=$(xpos), " *
                        "y=$(ypos)).  The assembled system must not be built " *
                        "from a NaN or Inf input; find the closure that " *
                        "returned it."))
end
"""
    _require_step_theta!(theta, step, t)

Per-step driver guard: the temperature field must be finite at the entry of the
run's time loop.  `step`/`t` are passed as numbers (not interpolated into a
message string) so the check costs one scan and allocates nothing -- the box
allocation A/B measures 0 B extra.
"""
function _require_step_theta!(theta::AbstractVector{Float64}, step::Int,
                              t::Float64)
    i = _first_nonfinite(theta)
    i == 0 && return nothing
    throw(ArgumentError("the temperature field theta contains a non-finite " *
        "value (got $(theta[i]) at index $i) at the per-step entry of the " *
        "driver (step=$(step), t=$(t)): the previous step left the state " *
        "non-finite.  This is a failed integration, not a bad input: inspect " *
        "the last step's diagnostics (CFL/dt, viscosity range, Stokes " *
        "residual) instead of continuing with NaNs."))
end