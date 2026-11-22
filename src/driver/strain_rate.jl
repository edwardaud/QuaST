# Strain-rate and spin fields on the quadtree SEM velocity.
#
# `semquad_strain_rate_field` returns the nodal strain-rate invariant, the
# `_components` variant returns the three tensor components the plastic branch and
# the Maxwell update need, and `semquad_spin_component` returns `W_xy`, the
# vorticity component the co-rotational term is built from.  The x derivatives use
# the leaf's own `leaf_hx` rather than the uniform `leaf_hy`, so a stretched leaf
# does not corrupt the field.

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
    semquad_spin_component(sq, ux, uz)

Nodal off-diagonal component of the **spin** (the antisymmetric part of the
velocity gradient) on the scalar layout of `sq`:

    wxy = W_xy = (du_x/dy - du_z/dx) / 2,     W = (grad v - grad v^T)/2.

It is the companion of [`semquad_strain_rate_components`](@ref) -- the symmetric
part is the strain rate -- and the co-rotational Maxwell stress update needs
both from the same velocity: `_advance_maxwell_stress!` calls these two
accessors on the same `(ux, uz)`, so the driving strain rate and the spin of a
step can never come from different velocity fields.  Same element-local loop and
stretched-box scaling (`leaf_hx` for the x derivative, `leaf_hy` for the y one)
as the strain-rate components.

The sign convention is the one the update is gated with: for a counter-clockwise
rigid rotation `u = (-Omega*y, Omega*x)` this returns `-Omega`, and the
co-rotational term `dt (W tau - tau W)` then rotates the stress with the
material (test/t_g250_corotation.jl).
"""
function semquad_spin_component(sq::SemQuad, ux::Vector{Float64},
                                uz::Vector{Float64})
    n = sq.p + 1
    wxy = zeros(Float64, sq.N)
    for lf in sq.leaves
        fx = 2.0 / leaf_hx(sq, lf[1])
        fy = 2.0 / leaf_hy(sq, lf[1])
        ids = sq.elem[lf]
        for j in 1:n, i in 1:n
            dux_dy = 0.0; duz_dx = 0.0
            for a in 1:n
                dux_dy += sq.D[j, a] * ux[ids[(a - 1) * n + i]]
                duz_dx += sq.D[i, a] * uz[ids[(j - 1) * n + a]]
            end
            wxy[ids[(j - 1) * n + i]] = 0.5 * (dux_dy * fy - duz_dx * fx)
        end
    end
    return wxy
end
