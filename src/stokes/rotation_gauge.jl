# Rotation-gauge post-processing: angular-momentum projection onto the rigid
# rotation mode and the torque-balance load check.  Moved verbatim from
# wall_constraints.jl; the wall-row builder stays in that file and the solve
# core calls these functions after the boundary rows are applied.
"""
    _semquad_rotation_moment(g, sq, u) -> (S, Q)

Discrete angular momentum `S = int (x u_y - y u_x) dA` of the velocity field
`u = [u_x; u_y]` and the rotation-mode norm `Q = int (x^2 + y^2) dA`, both with
the mapped measure and the velocity space's Gauss quadrature (exact for the
polynomial integrands: `u` is degree `p` per leaf, the rotation mode linear).
"""
function _semquad_rotation_moment(g::AbstractGeometry, sq::SemQuad,
                                  u::Vector{Float64})
    p = sq.p
    n = p + 1
    nu = sq.N
    fd = make_fedata(p, p - 1)
    S = 0.0
    Q = 0.0
    for lf in sq.leaves
        m = leaf_metric(g, sq, lf, fd)
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = quad_measure(m, fd, qx, qy)
            x, y = quad_point(m, sq, lf, fd, qx, qy)
            uxq = 0.0
            uyq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                uxq += u[ids[k]] * b
                uyq += u[nu + ids[k]] * b
            end
            S += w * (x * uyq - y * uxq)
            Q += w * (x * x + y * y)
        end
    end
    return S, Q
end

"""
    _semquad_project_rotation!(u, g, sq) -> omega

Remove the solid-body rotation from the velocity field in place
(`u_x += omega y`, `u_y -= omega x`), leaving the zero-net-angular-momentum
representative `int r x u dA = 0` in the **discrete** functional: `omega` is
`S(u)/S(Z_h)` where `Z_h` is the represented rotation mode (the nodal `(-y,x)`
interpolated by the velocity space), not `S(u)/int r^2 dA`.  On the mapped
annulus the nodal interpolation of a linear field is not exactly linear, so
`S(Z_h) = int (x Z_y - y Z_x) dA` differs from `int r^2 dA` by `O(h^(p+1))`;
normalising by `S(Z_h)` is what makes the *reported* discrete angular momentum
zero to round-off (the design's condition), and in the continuum limit both
normalisations are the L2 projection onto `(-y,x)` (the gate measures the
`O(h^(p+1))` difference).  Returns `omega` so the caller records the removed
rotation.
"""
function _semquad_project_rotation!(u::Vector{Float64}, g::AbstractGeometry,
                                    sq::SemQuad)
    p = sq.p
    n = p + 1
    nu = sq.N
    fd = make_fedata(p, p - 1)
    S = 0.0
    SZ = 0.0
    for lf in sq.leaves
        m = leaf_metric(g, sq, lf, fd)
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = quad_measure(m, fd, qx, qy)
            x, y = quad_point(m, sq, lf, fd, qx, qy)
            uxq = 0.0
            uyq = 0.0
            zxq = 0.0
            zyq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                uxq += u[ids[k]] * b
                uyq += u[nu + ids[k]] * b
                cx, cy = sq.coords[ids[k]]
                zxq += (-cy) * b
                zyq += cx * b
            end
            S += w * (x * uyq - y * uxq)
            SZ += w * (x * zyq - y * zxq)
        end
    end
    omega = S / SZ
    for (gid, pp) in enumerate(sq.coords)
        x = pp[1]
        y = pp[2]
        u[gid] += omega * y
        u[nu + gid] -= omega * x
    end
    return omega
end

"""
    _semquad_check_rotation_load(su, fv, gauge)

Solvability check of a rotation-gauge configuration: the assembled load must
have zero net torque about the origin, `int Z . f dA = 0` for the nodal rotation
mode `Z = (-y, x)`, because the rotation mode is in the kernel of the operator
and no wall in a free-slip/open configuration can exert a torque.  An
inconsistent load has no steady Stokes solution; the pinned solve would return a
field that does not balance it, so the card is refused instead with the measured
torque.
"""
function _semquad_check_rotation_load(su::SemQuad, fv::Vector{Float64},
                                      gauge::Symbol)
    gauge === :rotation || return nothing
    nu = su.N
    ztf = 0.0
    nz = 0.0
    nf = 0.0
    for (gid, p) in enumerate(su.coords)
        zx = -p[2]
        zy = p[1]
        fx = fv[gid]
        fy = fv[nu + gid]
        ztf += fx * zx + fy * zy
        nz += zx * zx + zy * zy
        nf += fx * fx + fy * fy
    end
    tol = 1.0e-6 * sqrt(nz) * sqrt(nf)
    abs(ztf) <= tol && return nothing
    throw(ArgumentError("the boundary configuration admits a solid-body " *
        "rotation (a free-slip curved wall with an `:open` wall; the solver " *
        "removes it with the zero-net-angular-momentum gauge), but the " *
        "assembled load has a net torque about the origin of $ztf (threshold " *
        "$tol = 1e-6 * |Z| * |f|; the mapped rotation mode is only " *
        "approximated by the nodal field, so a smooth torque-free load has an " *
        "O(h^(p+1)) consistency defect).  A rigid rotation has zero strain rate and " *
        "zero divergence, so a torque imbalance cannot be balanced by a " *
        "steady Stokes solution and the projected velocity would not satisfy " *
        "the momentum equation.  Either make the load torque-free (the " *
        "annulus buoyancy `Ra theta e_r` is), or leave a rigid wall that can " *
        "exert a torque.  Test: _semquad_check_rotation_load, G275."))
end
