# The weak-zone hook: one multiplicative factor on top of whatever rheology law
# is in use, so no law has to know about it (ASPECT reaches the same effect with
# a `weak zone` material model beside its rheology plugin).  `weak_active`,
# `semquad_weak_factor` and the `Params.eta_multiplier` hook are the activity
# predicates and the weakening factor.



"""
    weak_zone_factor(a, b; halfwidth, factor=0.1, taper=0.0)

Multiplier on the viscosity for a straight weak zone between the points `a` and
`b`, the cheap stand-in for a plate boundary: material within `halfwidth` of the
segment keeps `factor` times its law viscosity, everything else is untouched.

    pr = Params(...; eta_multiplier=weak_zone_factor((0.5, 1.0), (0.5, 0.6),
                                                    halfwidth=0.03, factor=0.05))

`taper > 0` smooths the edge with a cosine over that fraction of `halfwidth`, so
the factor stays continuous in `x` and `z`.  The multiplier never exceeds one: it
can weaken the material, not stiffen it.

ASPECT reaches the same effect with a `weak zone` material model beside its
rheology plugin; here it is a factor on top of the law, so no law has to know
about it (G232).
"""
function weak_zone_factor(a::Tuple{<:Real,<:Real}, b::Tuple{<:Real,<:Real};
                          halfwidth::Real, factor::Real=0.1, taper::Real=0.0)
    halfwidth > 0.0 || throw(ArgumentError("halfwidth must be > 0"))
    0.0 < factor <= 1.0 || throw(ArgumentError("factor must be in (0, 1]"))
    0.0 <= taper < 1.0 || throw(ArgumentError("taper must be in [0, 1)"))
    ax, az = Float64(a[1]), Float64(a[2])
    bx, bz = Float64(b[1]), Float64(b[2])
    len2 = (bx - ax)^2 + (bz - az)^2
    len2 > 0.0 || throw(ArgumentError("the two ends of the weak zone coincide"))
    hw = Float64(halfwidth)
    f = Float64(factor)
    tp = Float64(taper)
    return function (x::Real, z::Real)
        s = clamp(((Float64(x) - ax) * (bx - ax) +
                   (Float64(z) - az) * (bz - az)) / len2, 0.0, 1.0)
        dx = Float64(x) - (ax + s * (bx - ax))
        dz = Float64(z) - (az + s * (bz - az))
        d = sqrt(dx * dx + dz * dz)
        tp <= 0.0 && return d <= hw ? f : 1.0
        inner = (1.0 - tp) * hw
        d <= inner && return f
        d >= hw && return 1.0
        w = 0.5 * (1.0 - cos(pi * (d - inner) / (hw - inner)))
        return f + (1.0 - f) * w
    end
end
