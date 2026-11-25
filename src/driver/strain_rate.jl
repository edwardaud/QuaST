# Strain-rate and spin fields on the quadtree SEM velocity.
#
# `semquad_strain_rate_field` returns the nodal strain-rate invariant, the
# `_components` variant returns the three tensor components the plastic branch and
# the Maxwell update need, and `semquad_spin_component` returns `W_xy`, the
# vorticity component the co-rotational term is built from.  The x derivatives use
# the leaf's own `leaf_hx` rather than the uniform `leaf_hy`, so a stretched leaf
# does not corrupt the field.
#
# Stage 0 cut 6: each reconstructor dispatches on the stored geometry.  The box
# method is the historical body, character for character (the nine-hash dump and
# the box layer gates pin it); a mapped geometry computes the physical gradient
# of the interpolant by the chain rule with `J^{-T}` at every GLL node
# (`grad u = J^{-T} * (du/dxi, du/deta)`), i.e. the same metric rule the mapped
# operators use.  These are the fields every strain-rate-dependent law (Tosi,
# composite), the plastic branch, the strain-weakening field and the Maxwell
# stress advance consume.

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
computed element-locally with each element's own derivative matrix.  The box
body is the historical one (every derivative scaled by the single `leaf_hy`);
the mapped body uses the physical gradient `J^{-T}(du/dxi, du/deta)` per node.
"""
function semquad_strain_rate_field(sq::SemQuad, ux::Vector{Float64},
                                   uz::Vector{Float64})
    return _semquad_strain_rate_field(sq.geom, sq, ux, uz)
end

# Historical box body, verbatim: do not rewrite it.
function _semquad_strain_rate_field(::CartesianBox, sq::SemQuad,
                                    ux::Vector{Float64}, uz::Vector{Float64})
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

# Mapped body: physical gradients of the two velocity components at every node.
function _semquad_strain_rate_field(g::AbstractGeometry, sq::SemQuad,
                                    ux::Vector{Float64}, uz::Vector{Float64})
    n = sq.p + 1
    out = zeros(Float64, sq.N)
    for lf in sq.leaves
        ids = sq.elem[lf]
        for j in 1:n, i in 1:n
            dux_dxi = 0.0; dux_deta = 0.0; duz_dxi = 0.0; duz_deta = 0.0
            for a in 1:n
                dux_dxi += sq.D[i, a] * ux[ids[(j - 1) * n + a]]
                dux_deta += sq.D[j, a] * ux[ids[(a - 1) * n + i]]
                duz_dxi += sq.D[i, a] * uz[ids[(j - 1) * n + a]]
                duz_deta += sq.D[j, a] * uz[ids[(a - 1) * n + i]]
            end
            inv = jacobian_inverse(g, sq, lf, sq.xi[i], sq.xi[j])
            dux_dx = inv[1] * dux_dxi + inv[2] * dux_deta
            dux_dy = inv[3] * dux_dxi + inv[4] * dux_deta
            duz_dx = inv[1] * duz_dxi + inv[2] * duz_deta
            duz_dy = inv[3] * duz_dxi + inv[4] * duz_deta
            exx = dux_dx
            ezz = duz_dy
            exz = 0.5 * (dux_dy + duz_dx)
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
invariant were enough: the Maxwell stress history
carries the tensor components, so the shear one in particular needs `exz`.

The historical box body scales *every* derivative with `leaf_hy` (written for
the square mesh).  The box method here is the stretched-box one (x derivative
with `leaf_hx`, y derivative with `leaf_hy`, so it is also correct on a
stretched box); on the square mesh both give the same invariant as above.
`semquad_strain_rate_field`'s box body keeps its historical single-`h` scaling
(GEOM_DESIGN §1.6 item 5; the design recommended a separate fix and this cut
does not change box values).  The mapped body uses the physical gradient.
"""
function semquad_strain_rate_components(sq::SemQuad, ux::Vector{Float64},
                                        uz::Vector{Float64})
    return _semquad_strain_rate_components(sq.geom, sq, ux, uz)
end

# Historical box body, verbatim: do not rewrite it.
function _semquad_strain_rate_components(::CartesianBox, sq::SemQuad,
                                         ux::Vector{Float64},
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

# Mapped body: the physical gradient at every node.
function _semquad_strain_rate_components(g::AbstractGeometry, sq::SemQuad,
                                         ux::Vector{Float64},
                                         uz::Vector{Float64})
    n = sq.p + 1
    exx = zeros(Float64, sq.N)
    ezz = zeros(Float64, sq.N)
    exz = zeros(Float64, sq.N)
    for lf in sq.leaves
        ids = sq.elem[lf]
        for j in 1:n, i in 1:n
            dux_dxi = 0.0; dux_deta = 0.0; duz_dxi = 0.0; duz_deta = 0.0
            for a in 1:n
                dux_dxi += sq.D[i, a] * ux[ids[(j - 1) * n + a]]
                dux_deta += sq.D[j, a] * ux[ids[(a - 1) * n + i]]
                duz_dxi += sq.D[i, a] * uz[ids[(j - 1) * n + a]]
                duz_deta += sq.D[j, a] * uz[ids[(a - 1) * n + i]]
            end
            inv = jacobian_inverse(g, sq, lf, sq.xi[i], sq.xi[j])
            dux_dx = inv[1] * dux_dxi + inv[2] * dux_deta
            dux_dy = inv[3] * dux_dxi + inv[4] * dux_deta
            duz_dx = inv[1] * duz_dxi + inv[2] * duz_deta
            duz_dy = inv[3] * duz_dxi + inv[4] * duz_deta
            gp = ids[(j - 1) * n + i]
            exx[gp] = dux_dx
            ezz[gp] = duz_dy
            exz[gp] = 0.5 * (dux_dy + duz_dx)
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
as the strain-rate components; a mapped geometry uses the physical gradient.

The sign convention is the one the update is gated with: for a counter-clockwise
rigid rotation `u = (-Omega*y, Omega*x)` this returns `-Omega`, and the
co-rotational term `dt (W tau - tau W)` then rotates the stress with the
material (test/t_g250_corotation.jl).
"""
function semquad_spin_component(sq::SemQuad, ux::Vector{Float64},
                                uz::Vector{Float64})
    return _semquad_spin_component(sq.geom, sq, ux, uz)
end

# Historical box body, verbatim: do not rewrite it.
function _semquad_spin_component(::CartesianBox, sq::SemQuad,
                                 ux::Vector{Float64}, uz::Vector{Float64})
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

# Mapped body: physical gradients at every node.
function _semquad_spin_component(g::AbstractGeometry, sq::SemQuad,
                                 ux::Vector{Float64}, uz::Vector{Float64})
    n = sq.p + 1
    wxy = zeros(Float64, sq.N)
    for lf in sq.leaves
        ids = sq.elem[lf]
        for j in 1:n, i in 1:n
            dux_dxi = 0.0; dux_deta = 0.0; duz_dxi = 0.0; duz_deta = 0.0
            for a in 1:n
                dux_dxi += sq.D[i, a] * ux[ids[(j - 1) * n + a]]
                dux_deta += sq.D[j, a] * ux[ids[(a - 1) * n + i]]
                duz_dxi += sq.D[i, a] * uz[ids[(j - 1) * n + a]]
                duz_deta += sq.D[j, a] * uz[ids[(a - 1) * n + i]]
            end
            inv = jacobian_inverse(g, sq, lf, sq.xi[i], sq.xi[j])
            dux_dy = inv[3] * dux_dxi + inv[4] * dux_deta
            duz_dx = inv[1] * duz_dxi + inv[2] * duz_deta
            wxy[ids[(j - 1) * n + i]] = 0.5 * (dux_dy - duz_dx)
        end
    end
    return wxy
end
