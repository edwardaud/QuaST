# Flow and volume diagnostics: base advection, theta^2 div u, max
# divergence, surface velocity, Vrms and mean temperature (box + mapped).
"""
    semquad_base_advection_integral(sq, qt, pr, ux, uz)

Volume integral `int u_z dTbar/dz`.  For a divergence-free velocity with zero
normal flow on all walls this is zero (integrate by parts:
`int u.grad(Tbar) = -int Tbar div u = 0`).  A nonzero value measures how much
the discrete velocity violates that identity; it is the source term of the mean
theta/energy budget and the leading candidate for the H=5/AMR closure error.
"""
function semquad_base_advection_integral(sq::SemQuad, qt::Quadtree, pr::Params,
                                         ux::Vector{Float64}, uz::Vector{Float64})
    g = sq.geom
    g isa CartesianBox ||
        return _semquad_base_advection_integral_mapped(g, sq, qt, pr, ux, uz)
    p = sq.p
    n = p + 1
    fd = make_fedata(p, p - 1)
    s = 0.0
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        y0 = lf[3] * hy
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            uzq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                uzq += uz[ids[k]] * fd.phi_v[i, qx] * fd.phi_v[j, qy]
            end
            s += w * uzq * dTbar(y, pr.H)
        end
    end
    return s
end

"""
    semquad_theta2_divu_integral(sq, qt, theta, ux, uz)

Volume integral `int theta^2 div u`.  The conservative/skew advection conserves
the L2 norm only when `div u = 0`; this moment is the leading divergence-induced
growth term of the theta amplitude.  It is a diagnostic, not a gate on a
physical run.
"""
function semquad_theta2_divu_integral(sq::SemQuad, qt::Quadtree,
                                      theta::Vector{Float64},
                                      ux::Vector{Float64}, uz::Vector{Float64})
    g = sq.geom
    g isa CartesianBox ||
        return _semquad_theta2_divu_integral_mapped(g, sq, qt, theta, ux, uz)
    p = sq.p
    n = p + 1
    fd = make_fedata(p, p - 1)
    s = 0.0
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            thq = 0.0; dux = 0.0; duz = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                thq += theta[ids[k]] * fd.phi_v[i, qx] * fd.phi_v[j, qy]
                dux += ux[ids[k]] * fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / hx)
                duz += uz[ids[k]] * fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hy)
            end
            s += w * thq * thq * (dux + duz)
        end
    end
    return s
end

"""
    semquad_max_divergence(sq, qt, ux, uz)

Maximum pointwise `|div u|` at the GLL nodes of the quadtree.  Q2-Q1 enforces
the divergence only weakly (`int q div u = 0` for the pressure test space), so
this is not expected to be zero; it is the diagnostic needed to judge whether
the conservative advection form `-C'` (which assumes `div u = 0`) is consistent
with the discrete velocity.
"""
function semquad_max_divergence(sq::SemQuad, qt::Quadtree,
                                ux::Vector{Float64}, uz::Vector{Float64})
    g = sq.geom
    g isa CartesianBox ||
        return _semquad_max_divergence_mapped(g, sq, qt, ux, uz)
    p = sq.p
    n = p + 1
    worst = 0.0
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        ids = sq.elem[lf]
        for j in 1:n, i in 1:n
            dux = 0.0; duz = 0.0
            for a in 1:n
                dux += sq.D[i, a] * ux[ids[(j - 1) * n + a]]
                duz += sq.D[j, a] * uz[ids[(a - 1) * n + i]]
            end
            dux *= 2.0 / hx
            duz *= 2.0 / hy
            worst = max(worst, abs(dux + duz))
        end
    end
    return worst
end

"""
    semquad_surface_velocity(sq, qt, ux, uz; wall=:top)

RMS and maximum of the velocity magnitude along a horizontal wall, integrated
with the wall-adjacent GLL rule (the same one `semquad_nu_walls` uses):

    u_surf^RMS = sqrt( int |u|^2 dx / int dx ),   u_surf^max = max |u|.

These are two of the diagnostics reported in Tosi et al. (2015), Table 2.
"""
function semquad_surface_velocity(sq::SemQuad, qt::Quadtree,
                                  ux::Vector{Float64}, uz::Vector{Float64};
                                  wall::Symbol=:top)
    wall in (:top, :bottom) ||
        throw(ArgumentError("wall must be :top or :bottom"))
    g = sq.geom
    g isa CartesianBox ||
        return _semquad_surface_velocity_mapped(g, sq, qt, ux, uz; wall=wall)
    p = sq.p
    n = p + 1
    jF = wall === :top ? n : 1
    s2 = 0.0; wsum = 0.0; vmax = 0.0
    for lf in sq.leaves
        h = leaf_hy(sq, lf[1])
        hx = leaf_hx(sq, lf[1])
        y0 = lf[3] * h
        on_wall = wall === :top ? (y0 + h >= 1.0 - 1.0e-14) : (y0 <= 1.0e-14)
        on_wall || continue
        ids = sq.elem[lf]
        for i in 1:n
            w = 0.5 * hx * sq.w[i]
            g = ids[(jF - 1) * n + i]
            v = hypot(ux[g], uz[g])
            s2 += w * v * v
            wsum += w
            vmax = max(vmax, v)
        end
    end
    return sqrt(s2 / wsum), vmax
end

"""Volume RMS velocity on the quadtree (Gauss quadrature, same as `vrms`)."""
function semquad_vrms(sq::SemQuad, qt::Quadtree, ux::Vector{Float64},
                      uz::Vector{Float64})
    g = sq.geom
    g isa CartesianBox || return _semquad_vrms_mapped(g, sq, qt, ux, uz)
    p = sq.p
    n = p + 1
    fd = make_fedata(p, p - 1)
    s = 0.0
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            uxq = 0.0; uzq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                uxq += ux[ids[k]] * b
                uzq += uz[ids[k]] * b
            end
            s += w * (uxq * uxq + uzq * uzq)
        end
    end
    # volume mean (the domain area is exactly 1 on the unit square, so this is
    # bit identical there)
    return sqrt(s / total_area(qt))
end

"""Volume mean of the full temperature T = Tbar + theta (Gauss quadrature)."""
function semquad_mean_T(sq::SemQuad, qt::Quadtree, theta::Vector{Float64},
                        pr::Params)
    g = sq.geom
    g isa CartesianBox || return _semquad_mean_T_mapped(g, sq, qt, theta, pr)
    p = sq.p
    n = p + 1
    fd = make_fedata(p, p - 1)
    s = 0.0
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        y0 = lf[3] * hy
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            thq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                thq += theta[ids[k]] * fd.phi_v[i, qx] * fd.phi_v[j, qy]
            end
            s += w * (semquad_base_T(pr, y) + thq)
        end
    end
    return s / total_area(qt)
end

function _semquad_base_advection_integral_mapped(g::AbstractGeometry,
                                                 sq::SemQuad, qt::Quadtree,
                                                 pr::Params,
                                                 ux::Vector{Float64},
                                                 uz::Vector{Float64})
    p = sq.p
    n = p + 1
    fd = make_fedata(p, p - 1)
    s = 0.0
    for lf in sq.leaves
        m = leaf_metric(g, sq, lf, fd)
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = quad_measure(m, fd, qx, qy)
            x, y = quad_point(m, sq, lf, fd, qx, qy)
            uxq = 0.0
            uzq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                uxq += ux[ids[k]] * b
                uzq += uz[ids[k]] * b
            end
            ugh = upward_velocity(g, x, y, uxq, uzq)
            s += w * ugh * semquad_base_gradient(pr, g, x, y)
        end
    end
    return s
end

function _semquad_theta2_divu_integral_mapped(g::AbstractGeometry,
                                              sq::SemQuad, qt::Quadtree,
                                              theta::Vector{Float64},
                                              ux::Vector{Float64},
                                              uz::Vector{Float64})
    p = sq.p
    n = p + 1
    fd = make_fedata(p, p - 1)
    s = 0.0
    for lf in sq.leaves
        m = leaf_metric(g, sq, lf, fd)
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = quad_measure(m, fd, qx, qy)
            thq = 0.0
            dux = 0.0
            duz = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                thq += theta[ids[k]] * fd.phi_v[i, qx] * fd.phi_v[j, qy]
                gx, gz = basis_grad(m, fd, qx, qy, i, j)
                dux += ux[ids[k]] * gx
                duz += uz[ids[k]] * gz
            end
            s += w * thq * thq * (dux + duz)
        end
    end
    return s
end

function _semquad_max_divergence_mapped(g::AbstractGeometry, sq::SemQuad,
                                        qt::Quadtree, ux::Vector{Float64},
                                        uz::Vector{Float64})
    n = sq.p + 1
    worst = 0.0
    for lf in sq.leaves
        ids = sq.elem[lf]
        for j in 1:n, i in 1:n
            inv = jacobian_inverse(g, sq, lf, sq.xi[i], sq.xi[j])
            dux_dxi = 0.0
            dux_deta = 0.0
            duz_dxi = 0.0
            duz_deta = 0.0
            for a in 1:n
                dux_dxi += sq.D[i, a] * ux[ids[(j - 1) * n + a]]
                dux_deta += sq.D[j, a] * ux[ids[(a - 1) * n + i]]
                duz_dxi += sq.D[i, a] * uz[ids[(j - 1) * n + a]]
                duz_deta += sq.D[j, a] * uz[ids[(a - 1) * n + i]]
            end
            dux_dx = dux_dxi * inv[1] + dux_deta * inv[2]
            duz_dy = duz_dxi * inv[3] + duz_deta * inv[4]
            worst = max(worst, abs(dux_dx + duz_dy))
        end
    end
    return worst
end

function _semquad_surface_velocity_mapped(g::AbstractGeometry, sq::SemQuad,
                                          qt::Quadtree, ux::Vector{Float64},
                                          uz::Vector{Float64};
                                          wall::Symbol=:top)
    sideb, sidet = nu_wall_sides(g)
    side = wall === :top ? sidet : sideb
    n = sq.p + 1
    s2 = 0.0
    wsum = 0.0
    vmax = 0.0
    for lf in sq.leaves
        _on_logical_wall(lf, side, sq) || continue
        ids = sq.elem[lf]
        fm = _mapped_face_metric(g, sq, lf, side, sq.xi)
        @inbounds for q in 1:n
            w = face_jacobian(fm, q) * sq.w[q]
            gid = _logical_wall_node(ids, n, side, q)
            v = hypot(ux[gid], uz[gid])
            s2 += w * v * v
            wsum += w
            vmax = max(vmax, v)
        end
    end
    return sqrt(s2 / wsum), vmax
end

function _semquad_vrms_mapped(g::AbstractGeometry, sq::SemQuad, qt::Quadtree,
                              ux::Vector{Float64}, uz::Vector{Float64})
    p = sq.p
    n = p + 1
    fd = make_fedata(p, p - 1)
    s = 0.0
    for lf in sq.leaves
        m = leaf_metric(g, sq, lf, fd)
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = quad_measure(m, fd, qx, qy)
            uxq = 0.0
            uzq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                uxq += ux[ids[k]] * b
                uzq += uz[ids[k]] * b
            end
            s += w * (uxq * uxq + uzq * uzq)
        end
    end
    return sqrt(s / total_measure(g, sq, fd))
end

function _semquad_mean_T_mapped(g::AbstractGeometry, sq::SemQuad,
                                qt::Quadtree, theta::Vector{Float64},
                                pr::Params)
    p = sq.p
    n = p + 1
    fd = make_fedata(p, p - 1)
    s = 0.0
    for lf in sq.leaves
        m = leaf_metric(g, sq, lf, fd)
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = quad_measure(m, fd, qx, qy)
            x, y = quad_point(m, sq, lf, fd, qx, qy)
            thq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                thq += theta[ids[k]] * fd.phi_v[i, qx] * fd.phi_v[j, qy]
            end
            s += w * (semquad_base_T(pr, g, x, y) + thq)
        end
    end
    return s / total_measure(g, sq, fd)
end

