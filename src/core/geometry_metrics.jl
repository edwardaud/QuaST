# Geometry metrics: leaf/face metrics, quadrature-loop accessors, map curvature
# and the geometry metric cache.  Included after the geometry types and map
# files; the cache entry field types are defined in this file.
# ---------------------------------------------------------------------------
# 5. Leaf metric and the quadrature-loop accessors
# ---------------------------------------------------------------------------

"""
    BoxLeafMetric

The box metric: the seven scalars every box kernel reads, built once per leaf
and constant across the leaf.  `ihx = 2.0/hx`, `ihy = 2.0/hy` (the literal
reciprocals the code forms today) and `wfac = 0.25*hx*hy` (the volume factor
`0.25*hx*hy` in its historical association).  See GEOM_DESIGN E4/E5.
"""
struct BoxLeafMetric
    x0::Float64
    y0::Float64
    hx::Float64
    hy::Float64
    ihx::Float64
    ihy::Float64
    wfac::Float64
end

@inline function leaf_metric(::CartesianBox, lay, lf::NTuple{3,Int})
    hx = leaf_hx(lay, lf[1])
    hy = leaf_hy(lay, lf[1])
    x0 = lf[2] * hx
    y0 = lf[3] * hy
    return BoxLeafMetric(x0, y0, hx, hy, 2.0 / hx, 2.0 / hy, 0.25 * hx * hy)
end

# The quadrature-rule form of the box metric is the historical seven scalars
# (the 3-arg method), **not** a mapped metric built from the box map: the cold
# measure helpers (e.g. `leaf_measure`) can be called on a box geometry and must
# keep the box's historical values.  No production box kernel reaches this
# (the box bodies use the 3-arg form directly).
@inline leaf_metric(g::CartesianBox, lay, lf::NTuple{3,Int}, ::FEData) =
    leaf_metric(g, lay, lf)

"""
    MappedLeafMetric

The mapped metric: flat leaf-major arrays over the `nq x nq` quadrature points,
built once per `(geometry, leaf, p, nq)` and only read in the kernel loops.
`jinv[1..4]` are the four `J^{-T}` components of [`jacobian_inverse`](@ref) in
the same order; `xq`/`yq` are the mapped quadrature points and `detJ` the
volume factor.  A mapped kernel indexes them flat (`(qy-1)*nq + qx`).
"""
struct MappedLeafMetric
    xq::Vector{Float64}
    yq::Vector{Float64}
    detJ::Vector{Float64}
    jinv::NTuple{4,Vector{Float64}}
    nq::Int
end

@inline _leaf_metric_index(m::MappedLeafMetric, qx::Int, qy::Int) =
    (qy - 1) * m.nq + qx

"""
    leaf_metric(g, lay, lf, fd) -> MappedLeafMetric

The mapped leaf metric of a non-box geometry at the `fd.nq x fd.nq` quadrature
points (the map, Jacobian and inverse are evaluated once per point; the kernels
only index the arrays).  The box path deliberately does not use this: its
methods are the seven `BoxLeafMetric` scalars.

Cut 5: the metric is **cached** per `(geometry, leaf-set, layout, p, nq)` (see
[`_geom_cache`](@ref) and the cache section below), so a warm mapped assembly
only reads precomputed arrays; the body below is the raw builder that fills a
cache entry.
"""
@inline function leaf_metric(g::AbstractGeometry, lay, lf::NTuple{3,Int},
                             fd::FEData)
    t = _geom_cache(g, lay, fd)
    lm = t.lm
    lm === nothing && (lm = _ensure_leaf_metrics!(t))
    return (lm::Vector{MappedLeafMetric})[t.index[lf]]
end

# Raw builder of one mapped leaf metric; only the cache fill calls it, so a
# mapped assembly does not rebuild it per call (GEOM_DESIGN §2.5).
function _build_mapped_leaf_metric(g::AbstractGeometry, lay,
                                   lf::NTuple{3,Int}, fd::FEData)
    nq = fd.nq
    n2 = nq * nq
    xq = Vector{Float64}(undef, n2)
    yq = Vector{Float64}(undef, n2)
    detJ = Vector{Float64}(undef, n2)
    j11 = Vector{Float64}(undef, n2)
    j12 = Vector{Float64}(undef, n2)
    j21 = Vector{Float64}(undef, n2)
    j22 = Vector{Float64}(undef, n2)
    @inbounds for qy in 1:nq, qx in 1:nq
        k = (qy - 1) * nq + qx
        xi = fd.gl_x[qx]
        eta = fd.gl_x[qy]
        S, T = global_logical(lay, lf, xi, eta)
        x, y = logical_map(g, S, T)
        J, dj = element_jacobian(g, lay, lf, xi, eta)
        inv = _jacobian_inverse_from_jac(J, dj)
        xq[k] = x
        yq[k] = y
        detJ[k] = dj
        j11[k] = inv[1]
        j12[k] = inv[2]
        j21[k] = inv[3]
        j22[k] = inv[4]
    end
    return MappedLeafMetric(xq, yq, detJ, (j11, j12, j21, j22), nq)
end

"""
    quad_weight(m, fd, qx, qy) -> Float64

Volume weight of quadrature point `(qx,qy)`: `det J` for a mapped metric, the
stored `wfac` for a box metric (which the caller multiplies by the rule
weights), i.e. the historical `0.25*hx*hy*gl_w[qx]*gl_w[qy]` (GEOM_DESIGN E4).
"""
@inline quad_weight(m::BoxLeafMetric, fd::FEData, qx::Int, qy::Int) =
    m.wfac * fd.gl_w[qx] * fd.gl_w[qy]
@inline quad_weight(m::MappedLeafMetric, fd::FEData, qx::Int, qy::Int) =
    m.detJ[_leaf_metric_index(m, qx, qy)]

"""
    quad_measure(m, fd, qx, qy) -> Float64

The **fully weighted** quadrature volume measure `|det J| * gl_w[qx] * gl_w[qy]`
— the single quantity a volume quadrature loop sums.  The box method is the
historical expression of GEOM_DESIGN E4, `wfac * gl_w[qx] * gl_w[qy]` with
`wfac = 0.25*hx*hy` (the same operand and association as
`0.25*hx*hy*gl_w[qx]*gl_w[qy]`).

*Cut-2 interface note.*  `quad_weight` above is **not** interchangeable between
the two metric types: its box method keeps the historical E4 form, which already
contains the rule weights, because G266 pins it bit for bit against
`0.25*hx*hy*gl_w`; its mapped method returns the bare `detJ`, as the cut-1
docstring states.  A generic kernel that must run on both metrics therefore
uses this accessor, which includes the rule weights on both sides.  The two
accessors are pinned separately: `quad_weight` by G266 (per-metric historical
form), `quad_measure` by the cut-2 gates (assembled operators).
"""
@inline quad_measure(m::BoxLeafMetric, fd::FEData, qx::Int, qy::Int) =
    m.wfac * fd.gl_w[qx] * fd.gl_w[qy]
@inline quad_measure(m::MappedLeafMetric, fd::FEData, qx::Int, qy::Int) =
    m.detJ[_leaf_metric_index(m, qx, qy)] * fd.gl_w[qx] * fd.gl_w[qy]

"""
    quad_point(m, lay, lf, fd, qx, qy) -> (x, y)

Physical quadrature point.  The box method is the historical point map
(GEOM_DESIGN E3); the mapped method returns the precomputed array entry.
"""
@inline function quad_point(m::BoxLeafMetric, lay, lf::NTuple{3,Int},
                            fd::FEData, qx::Int, qy::Int)
    return (m.x0 + 0.5 * (fd.gl_x[qx] + 1.0) * m.hx,
            m.y0 + 0.5 * (fd.gl_x[qy] + 1.0) * m.hy)
end
@inline function quad_point(m::MappedLeafMetric, lay, lf::NTuple{3,Int},
                            fd::FEData, qx::Int, qy::Int)
    k = _leaf_metric_index(m, qx, qy)
    return (m.xq[k], m.yq[k])
end

"""
    basis_grad(m, fd, qx, qy, i, j) -> (gx, gz)

Physical gradient of the tensor-product basis function `(i,j)` at the
quadrature point `(qx,qy)`.  The box method is `dphi*phi*ihx`,
`phi*dphi*ihy` in that exact association (GEOM_DESIGN E5); the mapped method is
the chain rule, `Jinv11*dphi*phi + Jinv12*phi*dphi` per component.
"""
@inline function basis_grad(m::BoxLeafMetric, fd::FEData, qx::Int, qy::Int,
                            i::Int, j::Int)
    return (fd.dphi_v[i, qx] * fd.phi_v[j, qy] * m.ihx,
            fd.phi_v[i, qx] * fd.dphi_v[j, qy] * m.ihy)
end
@inline function basis_grad(m::MappedLeafMetric, fd::FEData, qx::Int, qy::Int,
                            i::Int, j::Int)
    k = _leaf_metric_index(m, qx, qy)
    dxi = fd.dphi_v[i, qx] * fd.phi_v[j, qy]
    deta = fd.phi_v[i, qx] * fd.dphi_v[j, qy]
    gx = dxi * m.jinv[1][k] + deta * m.jinv[2][k]
    gz = dxi * m.jinv[3][k] + deta * m.jinv[4][k]
    return (gx, gz)
end

"""
    basis_grad_p(m, fd, qx, qy, i, j) -> (gx, gz)

Physical gradient of the **pressure** tensor-product basis function `(i,j)`
(the `fd.phi_p`/`fd.dphi_p` family, degree `fd.q`) at the quadrature point.
Needed because the pressure mass/stiffness of `stokes/assembly.jl` live on the
lower-degree pressure space; `basis_grad` is the velocity (degree `fd.p`)
family.  The box method is `dphi_p*phi_p*ihx` in the historical association; the
mapped method is the same chain rule as `basis_grad`.
"""
@inline function basis_grad_p(m::BoxLeafMetric, fd::FEData, qx::Int, qy::Int,
                              i::Int, j::Int)
    return (fd.dphi_p[i, qx] * fd.phi_p[j, qy] * m.ihx,
            fd.phi_p[i, qx] * fd.dphi_p[j, qy] * m.ihy)
end
@inline function basis_grad_p(m::MappedLeafMetric, fd::FEData, qx::Int,
                              qy::Int, i::Int, j::Int)
    k = _leaf_metric_index(m, qx, qy)
    dxi = fd.dphi_p[i, qx] * fd.phi_p[j, qy]
    deta = fd.phi_p[i, qx] * fd.dphi_p[j, qy]
    gx = dxi * m.jinv[1][k] + deta * m.jinv[2][k]
    gz = dxi * m.jinv[3][k] + deta * m.jinv[4][k]
    return (gx, gz)
end

# ---------------------------------------------------------------------------
# 5b. Map curvature: the second derivatives of the map (cut 3)
# ---------------------------------------------------------------------------
#
# The energy path is the only part of the solver that differentiates twice: the
# entropy-viscosity residual needs the physical Laplacian of the temperature
# perturbation, and the base-state source needs `lap(Tbar(h))` (`GEOM_DESIGN`
# §2.7/§4.3 F4/§8 risk 10).  The physical Laplacian of a function represented on
# the reference element is, with `Jinv = J^{-T}` (the `jacobian_inverse` order
# `(xi_x, eta_x, xi_y, eta_y)`) and `H_p` the Hessian of the physical coordinate
# `p` in the leaf reference coordinates,
#
#     lap f = sum_{j,k} M[j,k] f_{xi_j xi_k} + sum_j C[j] f_{xi_j}
#     M     = Jinv^T Jinv                       (the metric tensor)
#     C[j]  = -sum_p Jinv[p,j] * tr(M * H_p)    (the map-curvature term)
#
# The `C` term is the one a naive implementation drops: it vanishes for an
# affine map (`H_p = 0`) and for the axis-aligned box, so the box cannot see it,
# but on any map whose Jacobian varies it is the same order as the `M` term
# (the measured failure is in `test/t_g269_curved_energy.jl`, part (d)).  The
# cross metric `M[xi,eta]` is nonzero whenever `x_eta != 0` or `y_xi != 0`
# (a sheared or curved element), so the mixed reference second derivative
# `f_{xi eta}` cannot be dropped either.
#
# THE BOX METHOD IS THE HISTORICAL EXPRESSION, NOT THE TRUE BOX CURVATURE.
# The historical entropy-viscosity code uses `(2/h)^2` for **both** directions
# with the single `h = leaf_hy` (it assumes square leaves, `GEOM_DESIGN` §1.6
# item 1) and no cross/curvature term (F4).  The box path keeps that body
# verbatim, so its `map_curvature` method returns exactly those historical
# coefficients (`m11 = m22 = (2.0/h)^2`, `h = leaf_hy`, `m12 = c1 = c2 = 0`);
# a stretched box (`Lx != 1`) is therefore *not* reproduced by the mapped
# formula, deliberately, because the nine state-dump hashes are the stronger
# contract and the historical EV is what produced every recorded run.  The
# mapped method below is the true curvature.  The two coincide on square leaves
# (the measured cross-check is in G269 (c)).

@inline _lap_index(m, qx::Int, qy::Int) = (qy - 1) * m.nq + qx

"""
    element_hessian(g, lay, lf, xi, eta) -> (x11, x12, x22, y11, y12, y22)

Second derivatives of the map with respect to the **leaf reference**
coordinates `(xi, eta)`, in the order

    x_{xi xi}, x_{xi eta}, x_{eta eta}, y_{xi xi}, y_{xi eta}, y_{eta eta}.

Affine geometries (`CartesianBox`, `AffineBox`) have a constant Jacobian, so
this is identically zero; `CurvedBox` carries the analytic values.  A curved
geometry added later (the annulus) must implement this method — it is part of
the geometry interface from cut 3 on, because the energy path cannot build a
correct physical Laplacian without it.
"""
@inline element_hessian(::CartesianBox, lay, lf::NTuple{3,Int},
                        xi::Float64, eta::Float64) =
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
@inline element_hessian(::AffineBox, lay, lf::NTuple{3,Int},
                        xi::Float64, eta::Float64) =
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
@inline function element_hessian(g::CurvedBox, lay, lf::NTuple{3,Int},
                                 xi::Float64, eta::Float64)
    # x = Lx*S (affine in S, so x'' = 0); y = T*d(S) with
    #   y_xi  = T*d'(S)*dS,  y_xi xi = T*d''(S)*dS^2,  y_xi eta = d'(S)*dS*dT.
    S, T = global_logical(lay, lf, xi, eta)
    l = lf[1]
    dS = (1.0 / leaf_nside_x(lay, l)) * 0.5
    dT = (1.0 / leaf_nside_y(lay, l)) * 0.5
    dp = _curved_dp(g, S)
    dpp = _curved_dpp(g, S)
    return (0.0, 0.0, 0.0,
            T * dpp * dS * dS, dp * dS * dT, 0.0)
end

"""
    BoxMapCurvature

The box map-curvature coefficients: the historical entropy-viscosity values
`m11 = m22 = (2.0/h)^2` with the single `h = leaf_hy`, and zero elsewhere
(`GEOM_DESIGN` §1.6 item 1/F4).  Not the true box curvature; see the section
header.
"""
struct BoxMapCurvature
    m11::Float64   # coefficient of f_{xi xi}
    m12::Float64   # coefficient of f_{xi eta} (metric tensor entry; lap uses 2*m12)
    m22::Float64   # coefficient of f_{eta eta}
    c1::Float64    # coefficient of f_xi   (map-curvature term)
    c2::Float64    # coefficient of f_eta  (map-curvature term)
end

"""
    MappedMapCurvature

The mapped map-curvature coefficients, flat leaf-major arrays over the
`nq x nq` quadrature points (built once per leaf by `map_curvature`):
`m11/m12/m22` are the entries of the metric tensor `Jinv^T Jinv` and
`c1/c2` the map-curvature coefficients `C[xi]`, `C[eta]` above.
"""
struct MappedMapCurvature
    m11::Vector{Float64}
    m12::Vector{Float64}
    m22::Vector{Float64}
    c1::Vector{Float64}
    c2::Vector{Float64}
    nq::Int
end

"""
    map_curvature(::CartesianBox, lay, lf, fd) -> BoxMapCurvature
    map_curvature(g::AbstractGeometry, lay, lf, fd) -> MappedMapCurvature

Second-derivative coefficients of the physical Laplacian at the quadrature
points of leaf `lf`.  The `CartesianBox` method returns the **historical**
entropy-viscosity coefficients (`(2.0/h)^2` in both directions with
`h = leaf_hy`, no cross or curvature term), so a box kernel routed through it
would reproduce the historical `semquad_ev_cell_stats` arithmetic bit for bit;
the production box body is not routed through it (F4), it stays verbatim.  The
mapped method is the chain rule of the section header, evaluated from
`jacobian_inverse` and `element_hessian`:
`M11 = xi_x^2 + xi_y^2`, `M12 = xi_x eta_x + xi_y eta_y`, `M22 = eta_x^2 + eta_y^2`,
`C[xi] = -(xi_x tr(M H_x) + xi_y tr(M H_y))`, `C[eta] = -(eta_x tr(M H_x) + eta_y tr(M H_y))`
with `tr(M H_p) = M11 Hp11 + 2 M12 Hp12 + M22 Hp22`.
"""
@inline function map_curvature(::CartesianBox, lay, lf::NTuple{3,Int},
                               fd::FEData)
    h = leaf_hy(lay, lf[1])          # the historical single scale, not hx
    ihe2 = (2.0 / h)^2
    return BoxMapCurvature(ihe2, 0.0, ihe2, 0.0, 0.0)
end

# Cut 5: cached exactly like the leaf metric (same key, lazy per-entry table),
# because the mapped energy/EV path asks for one curvature per leaf per call.
function map_curvature(g::AbstractGeometry, lay, lf::NTuple{3,Int}, fd::FEData)
    t = _geom_cache(g, lay, fd)
    if t.mc === nothing
        _ensure_map_curvatures!(t)
    end
    mc = t.mc::Vector{MappedMapCurvature}
    return mc[t.index[lf]]
end

# Raw builder of one mapped map-curvature record.
function _build_mapped_map_curvature(g::AbstractGeometry, lay,
                                     lf::NTuple{3,Int}, fd::FEData)
    nq = fd.nq
    n2 = nq * nq
    m11 = Vector{Float64}(undef, n2)
    m12 = Vector{Float64}(undef, n2)
    m22 = Vector{Float64}(undef, n2)
    c1 = Vector{Float64}(undef, n2)
    c2 = Vector{Float64}(undef, n2)
    @inbounds for qy in 1:nq, qx in 1:nq
        k = (qy - 1) * nq + qx
        xi = fd.gl_x[qx]
        eta = fd.gl_x[qy]
        inv = jacobian_inverse(g, lay, lf, xi, eta)
        xi_x, eta_x, xi_y, eta_y = inv
        m11k = xi_x * xi_x + xi_y * xi_y
        m12k = xi_x * eta_x + xi_y * eta_y
        m22k = eta_x * eta_x + eta_y * eta_y
        H = element_hessian(g, lay, lf, xi, eta)
        tHx = m11k * H[1] + 2.0 * m12k * H[2] + m22k * H[3]
        tHy = m11k * H[4] + 2.0 * m12k * H[5] + m22k * H[6]
        m11[k] = m11k
        m12[k] = m12k
        m22[k] = m22k
        c1[k] = -(xi_x * tHx + xi_y * tHy)
        c2[k] = -(eta_x * tHx + eta_y * tHy)
    end
    return MappedMapCurvature(m11, m12, m22, c1, c2, nq)
end

"""
    lap_phi(mc, fd, d2tab, qx, qy, i, j) -> Float64

Physical Laplacian of the tensor-product nodal basis function `(i,j)` at the
quadrature point `(qx,qy)`.  `d2tab[q][a]` is the second derivative of the 1-D
GLL basis function `a` at the 1-D quadrature point `q` (the table
`semquad_ev_cell_stats` builds from `lagrange_d2`).  The box method is the
historical entropy-viscosity term
`d2x[i] * phi[j] * (2.0/h)^2 + phi[i] * d2y[j] * (2.0/h)^2` in its exact
association; the mapped method applies the chain rule of the section header:

    m11 * f_{xi xi} + 2*m12 * f_{xi eta} + m22 * f_{eta eta}
        + c1 * f_xi + c2 * f_eta.
"""
@inline function lap_phi(mc::BoxMapCurvature, fd::FEData,
                         d2tab::Vector{Vector{Float64}}, qx::Int, qy::Int,
                         i::Int, j::Int)
    return d2tab[qx][i] * fd.phi_v[j, qy] * mc.m11 +
           fd.phi_v[i, qx] * d2tab[qy][j] * mc.m22
end
@inline function lap_phi(mc::MappedMapCurvature, fd::FEData,
                         d2tab::Vector{Vector{Float64}}, qx::Int, qy::Int,
                         i::Int, j::Int)
    k = _lap_index(mc, qx, qy)
    dxi = fd.dphi_v[i, qx] * fd.phi_v[j, qy]
    deta = fd.phi_v[i, qx] * fd.dphi_v[j, qy]
    d2xi = d2tab[qx][i] * fd.phi_v[j, qy]
    d2eta = fd.phi_v[i, qx] * d2tab[qy][j]
    return mc.m11[k] * d2xi +
           2.0 * mc.m12[k] * (fd.dphi_v[i, qx] * fd.dphi_v[j, qy]) +
           mc.m22[k] * d2eta + mc.c1[k] * dxi + mc.c2[k] * deta
end

# ---------------------------------------------------------------------------
# 6. Face metric and the face-loop accessors
# ---------------------------------------------------------------------------

"""
    BoxFaceMetric

The box face metric: `half` is the historical face factor `0.5*hy` on an
x-face / `0.5*hx` on a y-face (GEOM_DESIGN E8), and `(nx,ny)`/`(tx,ty)` are the
outward unit normal and the positive tangent (the direction of increasing
logical face parameter, GEOM_DESIGN §2.7).
"""
struct BoxFaceMetric
    half::Float64
    nx::Float64
    ny::Float64
    tx::Float64
    ty::Float64
end

"""
    face_metric(g, lay, lf, side, fd) -> BoxFaceMetric | MappedFaceMetric

Face metric of `side in (:xminus, :xplus, :yminus, :yplus)` for one leaf.  The
box method returns the historical factors; a mapped geometry precomputes, per
face quadrature point, the face Jacobian `ds/dq`, the outward unit normal and
the positive tangent.
"""
@inline function face_metric(::CartesianBox, lay, lf::NTuple{3,Int},
                             side::Symbol, fd::FEData)
    hx = leaf_hx(lay, lf[1])
    hy = leaf_hy(lay, lf[1])
    if side === :xminus
        return BoxFaceMetric(0.5 * hy, -1.0, 0.0, 0.0, 1.0)
    elseif side === :xplus
        return BoxFaceMetric(0.5 * hy, 1.0, 0.0, 0.0, 1.0)
    elseif side === :yminus
        return BoxFaceMetric(0.5 * hx, 0.0, -1.0, 1.0, 0.0)
    elseif side === :yplus
        return BoxFaceMetric(0.5 * hx, 0.0, 1.0, 1.0, 0.0)
    end
    throw(ArgumentError("unknown face side $side"))
end

"""
    MappedFaceMetric

The mapped face metric: arrays over the `nq` face quadrature points, one entry
per point: the face Jacobian `ds/dq`, the outward unit normal `(nx,ny)`, the
positive tangent `(tx,ty)` and the physical point `(xq,yq)`.
"""
struct MappedFaceMetric
    xq::Vector{Float64}
    yq::Vector{Float64}
    jac::Vector{Float64}
    nx::Vector{Float64}
    ny::Vector{Float64}
    tx::Vector{Float64}
    ty::Vector{Float64}
    nq::Int
end

function face_metric(g::AbstractGeometry, lay, lf::NTuple{3,Int},
                     side::Symbol, fd::FEData)
    return _mapped_face_metric(g, lay, lf, side, fd.gl_x)
end

"""
    _mapped_face_metric(g, lay, lf, side, qs) -> MappedFaceMetric

The mapped face metric evaluated at the **arbitrary** face parameters `qs`
(face parameter in `[-1,1]`; the physical point is the map image).  This is the
core of [`face_metric(g, lay, lf, side, fd)`](@ref), which feeds the Gauss
quadrature points of `fd`; the diagnostics call it with the **GLL nodes**
(`sq.xi`) because the box wall bodies integrate with the GLL rule (E8), and a
mapped wall integral must use the same rule to be comparable.

Cut 5: cached per `(geometry, leaf-set, layout, p, side, face-parameter rule)`,
the same cache entry as the leaf metric (the face table is a lazy slot, so a
face-metric rule that no volume kernel uses, like the GLL nodes, does not build
a volume table).
"""
function _mapped_face_metric(g::AbstractGeometry, lay, lf::NTuple{3,Int},
                             side::Symbol, qs::Vector{Float64})
    t = _geom_cache_nq(g, lay, length(qs))
    slot = get(t.fm, (side, length(qs)), nothing)
    if slot === nothing || slot[1] != qs
        _ensure_face_metrics!(t, side, qs)
        slot = t.fm[(side, length(qs))]
    end
    fms = slot[2]
    return fms[t.index[lf]]
end

# Raw builder of one mapped face metric at the arbitrary face parameters `qs`.
function _build_mapped_face_metric(g::AbstractGeometry, lay,
                                   lf::NTuple{3,Int}, side::Symbol,
                                   qs::Vector{Float64})
    nq = length(qs)
    xq = Vector{Float64}(undef, nq)
    yq = Vector{Float64}(undef, nq)
    jac = Vector{Float64}(undef, nq)
    nx = Vector{Float64}(undef, nq)
    ny = Vector{Float64}(undef, nq)
    tx = Vector{Float64}(undef, nq)
    ty = Vector{Float64}(undef, nq)
    is_x = side === :xminus || side === :xplus
    is_x || side === :yminus || side === :yplus ||
        throw(ArgumentError("unknown face side $side"))
    sgn = (side === :xplus || side === :yplus) ? 1.0 : -1.0
    @inbounds for q in 1:nq
        qr = qs[q]
        xi, eta = is_x ? (sgn, qr) : (qr, sgn)
        J, _ = element_jacobian(g, lay, lf, xi, eta)
        xq[q], yq[q] = element_map(g, lay, lf, xi, eta)
        if is_x
            # face parameter eta; tangent (x_eta, y_eta), outward +-(y_eta,-x_eta)
            a = J[2]
            b = J[4]
            j = hypot(a, b)
            jac[q] = j
            tx[q] = a / j
            ty[q] = b / j
            nx[q] = sgn * b / j
            ny[q] = -sgn * a / j
        else
            # face parameter xi; tangent (x_xi, y_xi), outward +-(-y_xi,x_xi)
            a = J[1]
            b = J[3]
            j = hypot(a, b)
            jac[q] = j
            tx[q] = a / j
            ty[q] = b / j
            nx[q] = -sgn * b / j
            ny[q] = sgn * a / j
        end
    end
    return MappedFaceMetric(xq, yq, jac, nx, ny, tx, ty, nq)
end

@inline face_jacobian(fm::BoxFaceMetric) = fm.half
@inline face_jacobian(fm::MappedFaceMetric, q::Int) = fm.jac[q]

"""
    face_weight(fm, fd, q) -> Float64

Face quadrature weight `ds/dq * gl_w[q]`.  The box method `half*gl_w[q]` is the
historical `0.5*h*gl_w[q]` (GEOM_DESIGN E8) with `half = 0.5*h` stored.
"""
@inline face_weight(fm::BoxFaceMetric, fd::FEData, q::Int) =
    fm.half * fd.gl_w[q]
@inline face_weight(fm::MappedFaceMetric, fd::FEData, q::Int) =
    fm.jac[q] * fd.gl_w[q]

@inline face_normal(fm::BoxFaceMetric, ::Int=1) = (fm.nx, fm.ny)
@inline face_normal(fm::MappedFaceMetric, q::Int) = (fm.nx[q], fm.ny[q])
@inline face_tangent(fm::BoxFaceMetric, ::Int=1) = (fm.tx, fm.ty)
@inline face_tangent(fm::MappedFaceMetric, q::Int) = (fm.tx[q], fm.ty[q])

@inline face_point(fm::MappedFaceMetric, q::Int) = (fm.xq[q], fm.yq[q])

"""
    face_point(lm, side, fd, q) -> (x, y)
    face_point(g, lay, lf, side, fd, q) -> (x, y)

Physical face quadrature point.  The box method is the historical wall-node
expression (GEOM_DESIGN E10 for the fixed coordinate, E3 for the moving one);
the mapped method returns the precomputed array entry.
"""
@inline function face_point(lm::BoxLeafMetric, side::Symbol, fd::FEData,
                            q::Int)
    if side === :xminus
        return (lm.x0, lm.y0 + 0.5 * (fd.gl_x[q] + 1.0) * lm.hy)
    elseif side === :xplus
        return (lm.x0 + lm.hx, lm.y0 + 0.5 * (fd.gl_x[q] + 1.0) * lm.hy)
    elseif side === :yminus
        return (lm.x0 + 0.5 * (fd.gl_x[q] + 1.0) * lm.hx, lm.y0)
    elseif side === :yplus
        return (lm.x0 + 0.5 * (fd.gl_x[q] + 1.0) * lm.hx, lm.y0 + lm.hy)
    end
    throw(ArgumentError("unknown face side $side"))
end
@inline function face_point(g::CartesianBox, lay, lf::NTuple{3,Int},
                            side::Symbol, fd::FEData, q::Int)
    return face_point(leaf_metric(g, lay, lf), side, fd, q)
end
@inline function face_point(g::AbstractGeometry, lay, lf::NTuple{3,Int},
                            side::Symbol, fd::FEData, q::Int)
    return face_point(face_metric(g, lay, lf, side, fd), q)
end

# ---------------------------------------------------------------------------
# 6b. The metric / curvature / face cache (stage 0 cut 5, GEOM_DESIGN §2.5)
# ---------------------------------------------------------------------------
#
# The mapped path evaluates the map and the Jacobian at every leaf/quadrature
# point.  Doing that inside every assembly, energy step, diagnostic and matrix-
# free apply is what made every mapped call allocate a fresh `MappedLeafMetric`
# per leaf (measured +1.9 MB per assembly at 1024 leaves, cut-2/cut-3 records).
# This section caches those tables so a **warm** mapped assembly or apply only
# reads precomputed arrays and allocates nothing of its own.
#
# The key is exactly the design's `(geometry identity, tree identity/hash, p,
# nq)`:
#
#   * `geometry_stamp(g)` — a 64-bit deterministic stamp of the canonical
#     identity (`geometry_identity` hashed through the raw Float64 bits of every
#     shape parameter plus a type tag).  An equal geometry gives an equal stamp;
#     a changed parameter gives a different stamp, so a box and an annulus can
#     never share an entry (the collision class GEOM_DESIGN §8 risk 5 warns
#     about);
#   * `_leafset_hash(leaves)` — the order-independent splitmix64 leaf-set
#     fingerprint the MG caches already use (the same function, one definition);
#   * `nleaf`, `nx0`, `ny0`, `Lx` — the layout, because the leaf hash alone does
#     not say how a `(level, ix, iy)` maps to a logical position;
#   * `p`, `nq` — the polynomial degree and the quadrature-rule size.  `p` is
#     redundant for the *leaf metric* itself (only `nq` enters the rule) but it
#     is in the key as the design specifies and it keeps the entries of two
#     spaces that share a rule size separate rather than aliased.
#
# **Invalidation rule.**  An entry is a pure function of its key and is never
# invalidated: a geometry change, a leaf-set change, a layout change or a
# different rule is a *new key*, so a stale hit is impossible.  There is no
# manual invalidation on the production path (an AMR cycle changes the tree and
# therefore the key).  The only supported clearing operation is
# [`empty_semquad_geometry_cache!`](@ref), for memory management in long
# processes and for gates that want a cold build; the caches grow with the
# number of distinct (geometry, tree) pairs a process sees, which is bounded by
# the tree trajectory.  A future time-dependent (ALE) map must add its state
# version to the stamp — that is the only way an entry could go stale.
#
# The three tables are lazy slots on one entry, so a caller pays only for what
# it uses (a face-only wall loop does not build a volume table, and the entropy
# viscosity's curvature does not build face tables).
#
# The box path never touches any of this: it has its own seven-scalar
# `BoxLeafMetric` methods, which stay the historical expressions.
#
# `SemQuad` is included after this file, so the degree is read generically from
# the layout object (`SemQuad.p`/`Quadtree` has none) at a cache *miss* only.

"""
    geometry_stamp(g::AbstractGeometry) -> UInt64

Deterministic 64-bit stamp of the geometry's canonical identity: a type tag and
the raw bit pattern of every shape parameter.  Equal geometries (same kind, same
`Float64` parameters) give equal stamps; any changed parameter changes the
stamp.  This is the geometry identity every cut-5 cache key carries (the metric
cache, `SEMQUAD_KOP_CACHE`, `SEMQUAD_MG_CACHE`, `SemQuadMGLevelKey` and
`_saddle_key`).  It is a hash, not the identity string, because it must not
allocate on a cache lookup; a 64-bit collision is the same accepted risk as
every other Julia hash key in this code base.
"""
function geometry_stamp(g::CartesianBox)
    return hash((UInt8(1), reinterpret(UInt64, g.Lx)))
end
function geometry_stamp(g::AffineBox)
    A = g.A
    return hash((UInt8(2), reinterpret(UInt64, A[1]), reinterpret(UInt64, A[2]),
                 reinterpret(UInt64, A[3]), reinterpret(UInt64, A[4]),
                 reinterpret(UInt64, g.b[1]), reinterpret(UInt64, g.b[2]),
                 reinterpret(UInt64, g.Lx)))
end
function geometry_stamp(g::CurvedBox)
    return hash((UInt8(3), reinterpret(UInt64, g.a),
                 reinterpret(UInt64, g.Lx)))
end

"""
    _leafset_hash(leaves) -> UInt64

Deterministic 64-bit fingerprint of a leaf set: the XOR of a splitmix64 mix of
every `(level, ix, iy)` triple.  XOR is order independent, so this is the hash of
the *sorted* leaf set without sorting it; equal leaf sets always give equal
values inside one process.  One definition, shared by the geometry cache and the
MG caches (`_semquad_mg_leaf_hash` forwards here).
"""
function _leafset_hash(leaves)::UInt64
    h = UInt64(0)
    @inbounds for lf in leaves
        z = (UInt64(lf[1]) << 42) ⊻ (UInt64(lf[2]) << 21) ⊻ UInt64(lf[3])
        z += 0x9e3779b97f4a7c15
        z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
        z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
        h ⊻= (z ⊻ (z >> 31))
    end
    return h
end

"""
    GeometryCacheKey

Key of one geometry-cache entry: the design's `(geometry identity, tree
identity/hash, p, nq)` plus the layout dimensions.  All fields are bits values,
so a `Dict` lookup does not allocate.
"""
struct GeometryCacheKey
    geom::UInt64
    tree::UInt64
    nleaf::Int
    nx0::Int
    ny0::Int
    Lx::Float64
    p::Int
    nq::Int
end

Base.:(==)(a::GeometryCacheKey, b::GeometryCacheKey) =
    a.geom == b.geom && a.tree == b.tree && a.nleaf == b.nleaf &&
    a.nx0 == b.nx0 && a.ny0 == b.ny0 && a.Lx == b.Lx && a.p == b.p &&
    a.nq == b.nq
Base.hash(k::GeometryCacheKey, h::UInt) =
    hash(k.nq, hash(k.p, hash(k.tree, hash(k.geom, hash(k.Lx, hash(k.nleaf,
         hash(k.ny0, hash(k.nx0, h))))))))

"""
    GeometryCacheEntry

One cache entry: the canonical leaf order and the lazy tables.  `leaves` is the
**sorted** leaf list (the order `make_semquad` and `semquad_kop` use), `index`
maps a leaf key to its ordinal in that order, and `lm`/`mc`/`fm` are the lazy
per-leaf metric / map-curvature / face tables.  `lay` is kept for the fast path
(a same-object hit skips the key computation) and is not part of the identity.
"""
mutable struct GeometryCacheEntry
    key::GeometryCacheKey
    geom::AbstractGeometry
    lay::Any
    leaves::Vector{NTuple{3,Int}}
    index::Dict{NTuple{3,Int},Int}
    nq::Int
    lm::Union{Nothing,Vector{MappedLeafMetric}}
    mc::Union{Nothing,Vector{MappedMapCurvature}}
    fm::Dict{Tuple{Symbol,Int},Any}   # (side, face-parameter count) => (qs, table)
end

"""
    SEMQUAD_GEOM_CACHE

The metric / curvature / face cache (`Dict{GeometryCacheKey,GeometryCacheEntry}`,
see the section header for the key and the invalidation rule).
`SEMQUAD_GEOM_CACHE_LAST` is the one-entry fast path consumed by the hot
per-leaf accessors; it is a pure memo, never a second source of truth.
"""
const SEMQUAD_GEOM_CACHE = Dict{GeometryCacheKey,GeometryCacheEntry}()
const SEMQUAD_GEOM_CACHE_LAST = Ref{Any}(nothing)

"""
    empty_semquad_geometry_cache!()

Forget every cached metric/curvature/face table.  This is a memory-management
and gate-support hook only: the caches are keyed by the full geometry+tree
identity, so correctness never depends on calling it (invalidation rule in the
section header).  It returns the number of entries dropped.
"""
function empty_semquad_geometry_cache!()
    n = length(SEMQUAD_GEOM_CACHE)
    empty!(SEMQUAD_GEOM_CACHE)
    SEMQUAD_GEOM_CACHE_LAST[] = nothing
    return n
end

"""
    semquad_geometry_cache_stats()

Diagnostics for the gates/docs: the number of live entries, how many of them have
the leaf-metric / curvature / face tables built, and the total cached leaf
records.  Read-only; never allocates on the production path.
"""
function semquad_geometry_cache_stats()
    n = length(SEMQUAD_GEOM_CACHE)
    nlm = 0; nmc = 0; nfm = 0; nleaf = 0
    for (_, e) in SEMQUAD_GEOM_CACHE
        e.lm === nothing || (nlm += 1)
        e.mc === nothing || (nmc += 1)
        nfm += length(e.fm)
        nleaf += length(e.leaves)
    end
    return (entries=n, metric_tables=nlm, curvature_tables=nmc,
            face_tables=nfm, leaves=nleaf)
end

# Degree of the space the layout belongs to: a `SemQuad` has `p`, a bare
# `Quadtree` layout does not (only its `Lx`/base cells are used).  Cold path
# only (cache miss), so the `hasproperty` query never touches a hot loop.
@inline _layout_p(lay) = hasproperty(lay, :p) ? lay.p : 0

@inline function _geom_cache(g::AbstractGeometry, lay, fd::FEData)
    return _geom_cache_nq(g, lay, fd.nq)
end

@inline function _geom_cache_nq(g::AbstractGeometry, lay, nq::Int)
    t = SEMQUAD_GEOM_CACHE_LAST[]
    if t isa GeometryCacheEntry && t.geom === g && t.lay === lay && t.nq == nq
        return t::GeometryCacheEntry
    end
    return _geom_cache_slow(g, lay, nq)
end

function _geom_cache_slow(g::AbstractGeometry, lay, nq::Int)
    key = GeometryCacheKey(geometry_stamp(g), _leafset_hash(lay.leaves),
                           length(lay.leaves), lay.nx0, lay.ny0,
                           Float64(lay.Lx), _layout_p(lay), nq)
    t = get(SEMQUAD_GEOM_CACHE, key, nothing)
    if t === nothing
        leaves = sort(collect(lay.leaves))
        index = Dict{NTuple{3,Int},Int}()
        sizehint!(index, length(leaves))
        @inbounds for (li, lf) in enumerate(leaves)
            index[lf] = li
        end
        t = GeometryCacheEntry(key, g, lay, leaves, index, nq, nothing,
                               nothing, Dict{Tuple{Symbol,Int},Any}())
        SEMQUAD_GEOM_CACHE[key] = t
    end
    SEMQUAD_GEOM_CACHE_LAST[] = t
    return t::GeometryCacheEntry
end

# A rule of `nq` points; the metric build only reads `nq` and `gl_x` (the same
# Gauss-Legendre nodes every `make_fedata` builds for that `nq`).
_geom_rule(nq::Int) = make_fedata(2, 2; nq=nq)

function _ensure_leaf_metrics!(t::GeometryCacheEntry)
    t.lm === nothing || return t.lm
    fd = _geom_rule(t.nq)
    lm = Vector{MappedLeafMetric}(undef, length(t.leaves))
    @inbounds for li in eachindex(t.leaves)
        lm[li] = _build_mapped_leaf_metric(t.geom, t.lay, t.leaves[li], fd)
    end
    t.lm = lm
    return lm
end

function _ensure_map_curvatures!(t::GeometryCacheEntry)
    t.mc === nothing || return t.mc
    fd = _geom_rule(t.nq)
    mc = Vector{MappedMapCurvature}(undef, length(t.leaves))
    @inbounds for li in eachindex(t.leaves)
        mc[li] = _build_mapped_map_curvature(t.geom, t.lay, t.leaves[li], fd)
    end
    t.mc = mc
    return mc
end

function _ensure_face_metrics!(t::GeometryCacheEntry, side::Symbol,
                               qs::Vector{Float64})
    key = (side, length(qs))
    fms = Vector{MappedFaceMetric}(undef, length(t.leaves))
    @inbounds for li in eachindex(t.leaves)
        fms[li] = _build_mapped_face_metric(t.geom, t.lay, t.leaves[li], side, qs)
    end
    t.fm[key] = (copy(qs), fms)
    return fms
end

# ---------------------------------------------------------------------------
