# G273: the 2-D polar annulus geometry `PolarAnnulus` (stage 0.5a).
#
# This gate is geometry only (no physics wiring: that is stage 0.5b).  It pins
#
#   (a) the map `(S,T) -> (r,phi) = (R0 + (R1-R0) S, Phi T) -> (r cos phi, r sin phi)`
#       as a *geometric* object (radius/angle/collinearity), the analytic inverse
#       (including the atan branch of the wedge angles > pi), and the node map
#       images through `make_semquad`;
#   (b) the leaf Jacobian, `det J`, `J^{-T}` and the map Hessian against closed
#       forms written here from the independent matrix-product derivation and
#       against a fourth-order finite difference of the public map;
#   (c) the physical measures of the plane annulus: `total_measure`, a direct
#       `sum(detJ*w)`, the per-leaf measures, the leaf sizes, the four wall
#       lengths `Phi R0`, `Phi R1`, `R1-R0`, `R1-R0` and the perimeter
#       `Phi (R0+R1) + 2(R1-R0)` of the wedge;
#   (d) the boundary registry and the face metric at **arbitrary** face
#       parameters (the cut-1/4 surface): outward normals (-e_r inner, +e_r
#       outer, -e_phi / +e_phi on the radial sides), positive tangents
#       (+e_phi on the circles, +e_r on the radial sides), unit length,
#       orthogonality and the analytic face Jacobians;
#   (e) `height = r - R0`, `|grad h| = 1`, `lap h = 1/r`, `gravity = -e_r` with
#       unit constant magnitude, and the `u . grad h` velocity accessor;
#   (f) the cut-3 second-derivative surface (`element_hessian` / `map_curvature`
#       / `lap_phi`): the physical Laplacians of the coordinate functions
#       `S, T, S^2, T^2, S*T` equal their analytic values, and a smooth nodal
#       field's discrete Laplacian equals a physical finite-difference Laplacian
#       -- with the dropped-curvature variant shown to fail by O(1);
#   (g) the canonical identity (every parameter, including the seam, is in it),
#       the cut-5 cache stamp, the constructor validation and the seam decision.
#
# WHY THE ASSERTIONS ARE NOT TAUTOLOGICAL (what each block catches):
#   * the map checks never read `logical_map` back into themselves: they test
#     `hypot`, `atan`, collinearity and node counts, so a swapped r/phi, a
#     forgotten `R1-R0` factor (`r = R0 + S` instead of `R0 + (R1-R0)S`), a
#     wrong angle scale (`phi = T` instead of `Phi*T`) or a wrong orientation
#     all fail -- the `thick` fixture (`R1-R0 = 2`) is there because a
#     `R1-R0 = 1` geometry cannot see the forgotten gap factor;
#   * the analytic inverse is fed points built from `(r, phi)` polar data, not
#     from the map, and the `threequarter`/`fullturn` fixtures put points with
#     `phi > pi` in the gate: an inverse without the `atan` branch (`+2pi`) gives
#     a negative `T`;
#   * `det J` is compared with `Phi r d dS dT` (an independent closed form) and
#     with a finite difference of `element_map`; `J^T J^{-T} = I` is a second,
#     independent pin of the `(xi_x, eta_x, xi_y, eta_y)` convention -- if
#     `jacobian_inverse` returned `J^{-1}` instead of `J^{-T}`, or permuted the
#     components, the product is not the identity;
#   * the measures are integrated by the production cold bodies and compared
#     with the closed forms `Phi/2 (R1^2-R0^2)`, `Phi R0`, `Phi R1`, `R1-R0`
#     (a wrong `det J`, a missing leaf, a wrong wall rule or a `2*pi*r` weight
#     would move them); the direct `sum(detJ*w)` is a second, quadrature-only
#     route to the same area;
#   * the face normals/tangents are the analytic vectors of the polar frame and
#     the face Jacobians the analytic `R dphi/dq` / `dr/dq`; the face metric is
#     evaluated by the cached generic body at arbitrary parameters, so a wrong
#     normal sign, a swapped tangent, a missing `Phi` or a cache-index defect
#     fails;
#   * `height`, `grad_height`, `height_laplacian` and `gravity` are compared with
#     the analytic radial values and the Laplacian additionally with a finite
#     difference of `height`;
#   * the second-derivative block is the annulus analogue of G269 (d): the exact
#     Laplacians of five *known* functions of the logical coordinates are
#     computed from `map_curvature` + `lap_phi`, so a dropped map-curvature
#     term, a wrong metric tensor or a wrong Hessian fails; the FD Laplacian of a
#     smooth nodal field goes through `semquad_eval` (mapped inverse + leaf
#     location), so the inverse map and the curvature surface are checked
#     together;
#   * the identity assertions pin the exact checkpoint string (a missing
#     parameter -- including the seam -- or a changed float format fails), and
#     the constructor assertions pin every refusal with `@test_throws`; the
#     `:periodic` refusal additionally pins that the message is *actionable*.
#
# The box path is not exercised here (G266..G272 pin it); this gate only reads
# the generic mapped bodies the annulus inherits, which is exactly stage 0.5a's
# acceptance criterion.

using Test, QuaST, LinearAlgebra

const MS273 = QuaST

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

# The fixtures cover: the published radius ratio as an exactly-unit gap (the
# benchmark is R0 = 1.22, R1 = 2.22); the published radii in Float64, where
# `R1 - R0 = 1 + 2^-52` (so a `R1 - R0 == 1` assumption is caught); a
# three-quarter wedge (the atan branch); a full-turn *cut ring* (`seam = :wall`
# at Phi = 2pi: the seam sides coincide physically); and a thick annulus with
# `R1 - R0 = 2`, which catches a hard-coded unit gap.
g273_geoms() = [
    "quarter" => PolarAnnulus(0.55, 1.55, 0.5pi),
    "published" => PolarAnnulus(1.22, 2.22, pi),
    "threequarter" => PolarAnnulus(1.0, 2.0, 1.5pi),
    "fullturn" => PolarAnnulus(1.0, 2.0, 2.0pi),
    "thick" => PolarAnnulus(1.0, 3.0, 0.5pi),
]

function g273_refine(qt::Quadtree, n)
    for _ in 1:n, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

# a uniform 4x4 tree, a mixed-level (hanging-node) tree, and a 2x2-base cell
# layout; every tree has the logical `Lx = 1` the annulus is defined on
function g273_uniform_tree()
    return "uniform4x4" => g273_refine(Quadtree(2), 2)
end
function g273_mixed_tree()
    qm = g273_refine(Quadtree(3), 3)
    for lf in sort(collect(qm.leaves))[1:2]
        refine!(qm, lf)
    end
    balance!(qm)
    return "mixed" => qm
end
g273_cells_tree() = "cells2x2l2" =>
    g273_refine(Quadtree(2; cells_x=2, cells_y=2), 2)

# the two trees the per-leaf loops use (the mixed tree keeps the per-leaf
# logical steps dS/dT honest on hanging leaves)
g273_loop_trees() = [g273_uniform_tree(), g273_mixed_tree()]

# a single representative leaf: enough for per-leaf formulas, keeps the gate
# seconds long (the measures block audits every leaf of a tree)
g273_one_leaf(lay) = sort(collect(lay.leaves))[1]

# global logical position of a leaf-reference point, written here from the
# definition (independent of the production `global_logical`)
function g273_ST(lay, lf, xi, eta)
    l = lf[1]
    S = (lf[2] + 0.5 * (xi + 1.0)) / (lay.nx0 * (1 << l))
    T = (lf[3] + 0.5 * (eta + 1.0)) / (lay.ny0 * (1 << l))
    return (S, T)
end

g273_dS(lay, lf) = 0.5 / (lay.nx0 * (1 << lf[1]))      # dS/dxi
g273_dT(lay, lf) = 0.5 / (lay.ny0 * (1 << lf[1]))      # dT/deta

# wrapped angular difference in [0, pi]
g273_angdiff(a, b) = abs(mod(a - b + pi, 2.0 * pi) - pi)

# 4th-order central finite difference of a univariate function
g273_fd4(f, x, h) =
    (-f(x + 2.0h) + 8.0 * f(x + h) - 8.0 * f(x - h) + f(x - 2.0h)) / (12.0h)

# finite difference of the element Jacobian `(x_xi, x_eta, y_xi, y_eta)` and its
# determinant from the public map (h = 1e-3 gives ~1e-13 truncation here)
function g273_fd_jacobian(g, lay, lf, xi, eta; h=1.0e-3)
    fx(a, b) = element_map(g, lay, lf, a, b)[1]
    fy(a, b) = element_map(g, lay, lf, a, b)[2]
    a = g273_fd4(t -> fx(t, eta), xi, h)
    b = g273_fd4(t -> fx(xi, t), eta, h)
    c = g273_fd4(t -> fy(t, eta), xi, h)
    e = g273_fd4(t -> fy(xi, t), eta, h)
    return ((a, b, c, e), a * e - b * c)
end

# the mapped physical Laplacian from the coefficients, with the deliberate
# "dropped map-curvature term" variant (the wrong implementation G273 rules out)
function g273_lap_from(mc, qx::Int, qy::Int, d; drop_c::Bool=false)
    k = (qy - 1) * mc.nq + qx
    c1 = drop_c ? 0.0 : mc.c1[k]
    c2 = drop_c ? 0.0 : mc.c2[k]
    return mc.m11[k] * d[3] + 2.0 * mc.m12[k] * d[4] + mc.m22[k] * d[5] +
           c1 * d[1] + c2 * d[2]
end

# reference-space derivatives (f_xi, f_eta, f_xi xi, f_xi eta, f_eta eta) and the
# exact physical Laplacian of the five coordinate functions used in block (f).
function g273_coord_fields(g, lay, lf, xi, eta)
    d = g.R1 - g.R0
    dS = g273_dS(lay, lf)
    dT = g273_dT(lay, lf)
    S, T = g273_ST(lay, lf, xi, eta)
    r = g.R0 + d * S
    fS = (dS, 0.0, 0.0, 0.0, 0.0)
    fT = (0.0, dT, 0.0, 0.0, 0.0)
    fS2 = (2.0 * S * dS, 0.0, 2.0 * dS * dS, 0.0, 0.0)
    fT2 = (0.0, 2.0 * T * dT, 0.0, 0.0, 2.0 * dT * dT)
    fST = (dS * T, S * dT, 0.0, dS * dT, 0.0)
    lapS = 1.0 / (d * r)
    lapT = 0.0
    lapS2 = 2.0 / (d * d) + 2.0 * S * lapS
    lapT2 = 2.0 / (g.Φ * g.Φ * r * r)
    lapST = T * lapS
    return ((fS, lapS, "S"), (fT, lapT, "T"), (fS2, lapS2, "S^2"),
            (fT2, lapT2, "T^2"), (fST, lapST, "S*T"))
end

# nodal value of the field nm at leaf-reference point (i, j)
function g273_field_value(nm, lay, lf, i, j, nodes)
    S, T = g273_ST(lay, lf, nodes[i], nodes[j])
    nm == "S" && return S
    nm == "T" && return T
    nm == "S^2" && return S * S
    nm == "T^2" && return T * T
    return S * T
end

@testset "G273 polar annulus geometry (stage 0.5a)" begin
    p = 2

    # ======================================================================
    # (a) the map and the analytic inverse
    # ======================================================================
    @testset "map and analytic inverse" begin
        # the axes and the far seam: per geometry, no tree needed
        for (gn, g) in g273_geoms()
            @test logical_map(g, 0.0, 0.0) == (g.R0, 0.0)
            x1, y1 = logical_map(g, 1.0, 0.0)
            @test x1 == g.R1 && y1 == 0.0
            xf, yf = logical_map(g, 0.0, 1.0)
            @test abs(hypot(xf, yf) - g.R0) <= 1e-15 * g.R0
            if g.Φ == 0.5pi
                @test abs(xf) <= 1e-15 * g.R0 && yf > 0.0
            elseif g.Φ == pi
                @test abs(yf) <= 1e-15 * g.R0 && xf < 0.0
            elseif g.Φ == 1.5pi
                @test abs(xf) <= 1e-15 * g.R0 && yf < 0.0
            elseif g.Φ == 2.0pi
                @test abs(yf) <= 1e-15 * g.R0 && xf > 0.0
            end
        end
        loop_trees = [g273_loop_trees()...; g273_cells_tree()]
        for (gn, g) in g273_geoms(), (tn, qt) in loop_trees
            d = g.R1 - g.R0
            lf = g273_one_leaf(qt)
            # radius/angle/collinearity: purely geometric statements about the
            # images (a wrong r/phi scale or a swapped pair fails them)
            for (xi, eta) in ((-1.0, -1.0), (0.6, -0.4))
                x, y = element_map(g, qt, lf, xi, eta)
                S, T = g273_ST(qt, lf, xi, eta)
                r = g.R0 + d * S
                @test abs(hypot(x, y) - r) <= 1e-14 * g.R1
                @test g273_angdiff(atan(y, x), g.Φ * T) <= 1e-14
                x2, y2 = element_map(g, qt, lf, 0.5 * xi, eta)
                S2, _ = g273_ST(qt, lf, 0.5 * xi, eta)
                @test abs(x * y2 - y * x2) <= 1e-14 * g.R1 * (g.R0 + d * S2)
            end
            # the analytic inverse: round trip on (S,T)
            for S in (0.0, 1.0), T in (0.0, 0.5, 1.0)
                S2, T2 = element_inverse(g, qt, logical_map(g, S, T)...)
                @test abs(S2 - S) <= 1e-13
                @test abs(T2 - T) <= 1e-13
            end
            # points built from (r, phi) polar data, NOT through the map; the
            # phi > pi entries are the atan-branch catcher
            for (S, phi) in ((0.13, 0.1), (0.5, 0.5 * g.Φ), (0.4, 0.8 * pi))
                phi <= g.Φ || continue
                r = g.R0 + d * S
                S2, T2 = element_inverse(g, qt, r * cos(phi), r * sin(phi))
                @test abs(S2 - S) <= 1e-13
                @test abs(T2 - phi / g.Φ) <= 1e-13
            end
            # leaf-reference inverse and leaf location
            leaves = Set(qt.leaves)
            x, y = element_map(g, qt, lf, 0.17, -0.31)
            xi2, eta2 = element_inverse(g, qt, lf, x, y)
            @test abs(xi2 - 0.17) <= 1e-12
            @test abs(eta2 + 0.31) <= 1e-12
            @test locate_leaf(g, qt, x, y, leaves) == lf
            # space-level map image: every SEM node lies in the annulus sector,
            # and a conforming side tiled by k leaf faces has p*k+1 nodes
            sq = make_semquad(qt, p; geom=g)
            @test all(c -> g.R0 * (1.0 - 1e-14) <= hypot(c...) <=
                           g.R1 * (1.0 + 1e-14), sq.coords)
            kin = count(l -> l[2] == 0, sq.leaves)
            n0 = count(c -> abs(hypot(c...) - g.R0) <= 1e-14 * g.R1, sq.coords)
            @test n0 == p * kin + 1
            kout = count(l -> l[2] == sq.nx0 * (1 << l[1]) - 1, sq.leaves)
            n1 = count(c -> abs(hypot(c...) - g.R1) <= 1e-14 * g.R1, sq.coords)
            @test n1 == p * kout + 1
        end
        println("G273 (a) map/inverse: ok")
    end

    # ======================================================================
    # (b) Jacobian, detJ, J^{-T}, map Hessian
    # ======================================================================
    @testset "Jacobian, detJ and J^{-T}" begin
        for (gn, g) in g273_geoms(), (tn, qt) in g273_loop_trees()
            d = g.R1 - g.R0
            lf = g273_one_leaf(qt)
            for (xi, eta) in ((-0.6, -0.8), (0.7, 0.55))
                S, T = g273_ST(qt, lf, xi, eta)
                r = g.R0 + d * S
                phi = g.Φ * T
                dS = g273_dS(qt, lf)
                dT = g273_dT(qt, lf)
                J, dj = element_jacobian(g, qt, lf, xi, eta)
                # independent derivation: J = J_ST * diag(d, Phi) * diag(dS, dT)
                cphi = cos(phi)
                sphi = sin(phi)
                Ja = ((cphi * d * dS, -r * sphi * g.Φ * dT),
                      (sphi * d * dS, r * cphi * g.Φ * dT))
                sc = max(abs(Ja[1][1]), abs(Ja[1][2]), abs(Ja[2][1]),
                         abs(Ja[2][2]))
                @test maximum(abs(J[k] - Ja[(k - 1) ÷ 2 + 1][(k - 1) % 2 + 1])
                              for k in 1:4) <= 1e-13 * sc
                # detJ: closed form det J = Phi r d dS dT and positivity
                dj_exact = g.Φ * r * d * dS * dT
                @test abs(dj - dj_exact) <= 1e-13 * dj_exact
                @test dj > 0.0
                # finite difference of the public map
                Jfd, djfd = g273_fd_jacobian(g, qt, lf, xi, eta)
                scfd = maximum(abs, Jfd)
                @test maximum(abs(J[k] - Jfd[k]) for k in 1:4) <= 1e-9 * scfd
                @test abs(dj - djfd) <= 1e-9 * max(abs(djfd), 1.0e-3)
                # J^{-T} in the (xi_x, eta_x, xi_y, eta_y) convention and the
                # chain-rule identity J^T J^{-T} = I (J^{-1} would fail)
                inv = jacobian_inverse(g, qt, lf, xi, eta)
                inva = (cphi / (d * dS), -sphi / (g.Φ * r * dT),
                        sphi / (d * dS), cphi / (g.Φ * r * dT))
                @test maximum(abs(inv[k] - inva[k]) for k in 1:4) <=
                      1e-13 * maximum(abs, inva)
                G = [J[1] J[3]; J[2] J[4]]
                Hm = [inv[1] inv[2]; inv[3] inv[4]]
                @test maximum(abs.(G * Hm - [1.0 0.0; 0.0 1.0])) <= 1e-14
                # the map Hessian: closed form, exact zeros and finite difference
                H = MS273.element_hessian(g, qt, lf, xi, eta)
                Ha = (0.0, -g.Φ * d * dS * dT * sphi,
                      -g.Φ * g.Φ * r * dT * dT * cphi,
                      0.0, g.Φ * d * dS * dT * cphi,
                      -g.Φ * g.Φ * r * dT * dT * sphi)
                @test maximum(abs(H[k] - Ha[k]) for k in 1:6) <=
                      1e-13 * maximum(abs, Ha)
                @test H[1] == 0.0 && H[4] == 0.0
                Hfd = ((g273_fd4(t -> element_jacobian(g, qt, lf, t, eta)[1][1],
                                 xi, 1.0e-3),
                        g273_fd4(t -> element_jacobian(g, qt, lf, xi, t)[1][1],
                                 eta, 1.0e-3),
                        g273_fd4(t -> element_jacobian(g, qt, lf, xi, t)[1][2],
                                 eta, 1.0e-3),
                        g273_fd4(t -> element_jacobian(g, qt, lf, t, eta)[1][3],
                                 xi, 1.0e-3),
                        g273_fd4(t -> element_jacobian(g, qt, lf, t, eta)[1][4],
                                 xi, 1.0e-3),
                        g273_fd4(t -> element_jacobian(g, qt, lf, xi, t)[1][4],
                                 eta, 1.0e-3)))
                @test maximum(abs(H[k] - Hfd[k]) for k in 1:6) <=
                      1e-7 * maximum(abs, Ha)
            end
        end
        println("G273 (b) Jacobian/Hessian: ok")
    end

    # ======================================================================
    # (c) physical measures of the plane annulus
    # ======================================================================
    @testset "plane-annulus measures" begin
        for (gn, g) in g273_geoms(), (tn, qt) in g273_loop_trees()
            d = g.R1 - g.R0
            sq = make_semquad(qt, p; geom=g)
            A = g.Φ / 2.0 * (g.R1^2 - g.R0^2)
            tm = total_measure(g, sq)
            @test abs(tm - A) <= 1e-13 * A
            # direct quadrature-only route: sum(detJ * w) over every leaf
            fd = make_fedata(2, 2; nq=4)
            sdirect = 0.0
            for lf in sq.leaves, qy in 1:fd.nq, qx in 1:fd.nq
                _, dj = element_jacobian(g, sq, lf, fd.gl_x[qx], fd.gl_x[qy])
                sdirect += dj * fd.gl_w[qx] * fd.gl_w[qy]
            end
            @test abs(sdirect - A) <= 1e-13 * A
            @test abs(sdirect - tm) <= 1e-13 * A
            @test abs(sum(leaf_measure(g, sq, lf) for lf in sq.leaves) - A) <=
                  1e-12 * A
            # the plane measure carries no extra weight (not an axisymmetric
            # shell: measure_weight stays 1.0, ANNULUS_REFS section 3)
            lf = g273_one_leaf(sq)
            x, y = element_map(g, sq, lf, 0.2, -0.3)
            @test measure_weight(g, x, y) == 1.0
            # leaf sizes: radial d/nside_x, angular Phi*r/nside_y at the centre
            dS = g273_dS(sq, lf)
            dT = g273_dT(sq, lf)
            S, _ = g273_ST(sq, lf, 0.0, 0.0)
            minx = d * 2.0 * dS
            miny = g.Φ * (g.R0 + d * S) * 2.0 * dT
            @test abs(leaf_min_size(g, sq, lf) - min(minx, miny)) <=
                  1e-12 * max(minx, miny)
            @test abs(leaf_vertical_size(g, sq, lf) - minx) <= 1e-12 * minx
            @test maximum(abs(leaf_normal_size(g, sq, lf, side) - minx)
                          for side in (:xminus, :xplus)) <= 1e-12 * minx
            @test maximum(abs(leaf_normal_size(g, sq, lf, side) - miny)
                          for side in (:yminus, :yplus)) <= 1e-12 * miny
            # the four wall lengths of the wedge
            @test abs(boundary_measure(g, sq, :xminus) - g.Φ * g.R0) <=
                  1e-13 * max(g.Φ * g.R0, 1.0)
            @test abs(boundary_measure(g, sq, :xplus) - g.Φ * g.R1) <=
                  1e-13 * max(g.Φ * g.R1, 1.0)
            @test abs(boundary_measure(g, sq, :yminus) - d) <= 1e-13 * max(d, 1.0)
            @test abs(boundary_measure(g, sq, :yplus) - d) <= 1e-13 * max(d, 1.0)
            # perimeter: the mapped body summed over the boundary faces, and
            # the four wall measures must add up to it
            bmask = MS273._semquad_boundary_face_mask(qt, sq)
            per = perimeter(g, sq, bmask)
            per_exact = g.Φ * (g.R0 + g.R1) + 2.0 * d
            @test abs(per - per_exact) <= 1e-12 * per_exact
            @test abs(per - (boundary_measure(g, sq, :xminus) +
                             boundary_measure(g, sq, :xplus) +
                             boundary_measure(g, sq, :yminus) +
                             boundary_measure(g, sq, :yplus))) <=
                  1e-12 * per_exact
        end
        println("G273 (c) measures: ok")
    end

    # ======================================================================
    # (d) boundary registry and face metric at arbitrary face parameters
    # ======================================================================
    @testset "boundary registry and face metric" begin
        qs = [-0.6, 0.0, 0.85]                # arbitrary face parameters
        for (gn, g) in g273_geoms()
            bnd = boundaries(g)
            @test length(bnd) == 4
            @test [b.side for b in bnd] == [:xminus, :xplus, :yminus, :yplus]
            @test all(b -> b.kind === :wall, bnd)  # stage 0.5a: a four-walled wedge
            @test [b.tag for b in bnd] == [:inner, :outer, :seam_lo, :seam_hi]
        end
        for (gn, g) in g273_geoms()
            d = g.R1 - g.R0
            sq = make_semquad(g273_uniform_tree()[2], p; geom=g)
            for side in (:xminus, :xplus, :yminus, :yplus)
                onwall = [lf for lf in sq.leaves
                          if MS273._on_logical_wall(lf, side, sq)]
                @test !isempty(onwall)
                lf = onwall[1]
                fm = MS273._mapped_face_metric(g, sq, lf, side, qs)
                @test fm.nq == length(qs)
                dS = g273_dS(sq, lf)
                dT = g273_dT(sq, lf)
                for q in eachindex(qs)
                    fp = MS273.face_point(fm, q)
                    n = MS273.face_normal(fm, q)
                    tau = MS273.face_tangent(fm, q)
                    r = hypot(fp...)
                    if side === :xminus || side === :xplus
                        R = side === :xminus ? g.R0 : g.R1
                        sgn = side === :xplus ? 1.0 : -1.0
                        # the point is the map image of the face, and the
                        # analytic outward normal is +-e_r / tangent +e_phi
                        Tface = (lf[3] + 0.5 * (qs[q] + 1.0)) /
                                (sq.ny0 * (1 << lf[1]))
                        @test abs(r - R) <= 1e-14 * g.R1 &&
                              g273_angdiff(atan(fp[2], fp[1]), g.Φ * Tface) <=
                              1e-13
                        er = (fp[1] / r, fp[2] / r)
                        ephi = (-fp[2] / r, fp[1] / r)
                        @test maximum(abs.((n[1] - sgn * er[1],
                                            n[2] - sgn * er[2]))) <= 1e-13
                        @test maximum(abs.((tau[1] - ephi[1],
                                            tau[2] - ephi[2]))) <= 1e-13
                        @test abs(dot(n, fp) - sgn * R) <= 1e-13 * R
                        @test abs(MS273.face_jacobian(fm, q) - R * g.Φ * dT) <=
                              1e-13 * max(R * g.Φ * dT, 1.0)
                    else
                        # a radial side: the point is on the ray phi = 0 / Phi
                        # with outward normal -/+e_phi and tangent e_r
                        phiside = side === :yminus ? 0.0 : g.Φ
                        Sface = (lf[2] + 0.5 * (qs[q] + 1.0)) /
                                (sq.nx0 * (1 << lf[1]))
                        @test abs(r - (g.R0 + d * Sface)) <= 1e-13 * g.R1 &&
                              g273_angdiff(atan(fp[2], fp[1]), phiside) <= 1e-13
                        nexp = side === :yminus ? (0.0, -1.0) :
                               (-sin(g.Φ), cos(g.Φ))
                        texp = (cos(phiside), sin(phiside))
                        @test maximum(abs.((n[1] - nexp[1], n[2] - nexp[2]))) <=
                              1e-13
                        @test maximum(abs.((tau[1] - texp[1],
                                            tau[2] - texp[2]))) <= 1e-13
                        @test abs(dot(n, fp)) <= 1e-13 * g.R1
                        @test abs(MS273.face_jacobian(fm, q) - d * dS) <=
                              1e-13 * max(d * dS, 1.0)
                    end
                    @test abs(hypot(n...) - 1.0) <= 1e-14 &&
                          abs(dot(n, tau)) <= 1e-14
                end
            end
            # the rule-point surface the kernels consume (`face_metric(fd)`):
            # face weight = face Jacobian * rule weight against the analytic
            # Jacobian (a wrong wall rule or a missing Phi fails)
            fd = make_fedata(p, p)
            for side in (:xminus, :xplus, :yminus, :yplus)
                lf = [l for l in sq.leaves
                      if MS273._on_logical_wall(l, side, sq)][1]
                fm = face_metric(g, sq, lf, side, fd)
                jac = if side === :xminus
                    g.R0 * g.Φ * g273_dT(sq, lf)
                elseif side === :xplus
                    g.R1 * g.Φ * g273_dT(sq, lf)
                else
                    d * g273_dS(sq, lf)
                end
                @test maximum(abs(face_weight(fm, fd, q) - jac * fd.gl_w[q])
                              for q in 1:fd.nq) <= 1e-13 * max(jac, 1.0)
            end
        end
        println("G273 (d) boundaries/face metric: ok")
    end

    # ======================================================================
    # (e) radial height, its gradient/Laplacian, gravity
    # ======================================================================
    @testset "radial height, Laplacian and gravity" begin
        for (gn, g) in g273_geoms(), (tn, qt) in g273_loop_trees()
            d = g.R1 - g.R0
            lf = g273_one_leaf(qt)
            for (xi, eta) in ((-0.7, 0.3), (0.0, 0.0))
                x, y = element_map(g, qt, lf, xi, eta)
                S, _ = g273_ST(qt, lf, xi, eta)
                r = g.R0 + d * S
                h = height(g, x, y)
                @test abs(h - d * S) <= 1e-14 * max(d, 1.0)
                @test abs(depth(g, x, y) - (1.0 - h)) <= 1e-15
                gh = grad_height(g, x, y)
                @test max(abs(gh[1] - x / r), abs(gh[2] - y / r)) <= 1e-15
                @test abs(hypot(gh...) - 1.0) <= 1e-15   # |grad h| = 1
                @test abs(height_laplacian(g, x, y) - 1.0 / r) <= 1e-14 / r
                gr = gravity(g, x, y)
                @test max(abs(gr[1] + x / r), abs(gr[2] + y / r)) <= 1e-15
                @test gravity_magnitude(g, x, y) == 1.0
                # u . grad h is the radial velocity (the physics wiring hook)
                @test abs(upward_velocity(g, x, y, 1.3, -0.7) -
                          (1.3 * x / r - 0.7 * y / r)) <= 1e-15
            end
            # wall values: h = 0 on the inner circle, h = d on the outer
            for T in (0.1, 0.9)
                @test abs(height(g, logical_map(g, 0.0, T)...)) <= 1e-15 * g.R1
                @test abs(height(g, logical_map(g, 1.0, T)...) - d) <=
                      1e-15 * g.R1
            end
            # finite-difference check of lap h = 1/r at an interior point
            x, y = logical_map(g, 0.4, 0.35)
            r = g.R0 + d * 0.4
            δ = 1.0e-4 * g.R1
            hc = height(g, x, y)
            lapfd = (height(g, x + δ, y) + height(g, x - δ, y) +
                     height(g, x, y + δ) + height(g, x, y - δ) -
                     4.0 * hc) / (δ * δ)
            @test abs(lapfd - 1.0 / r) <= 1e-5 / r
            # on a circular wall gravity is the *inward* radial direction (the
            # outward normal on the inner wall, its negative on the outer) and
            # is orthogonal to the positive tangent
            sq = make_semquad(qt, p; geom=g)
            for side in (:xminus, :xplus)
                lf2 = [l for l in sq.leaves
                       if MS273._on_logical_wall(l, side, sq)][1]
                fm = MS273._mapped_face_metric(g, sq, lf2, side, [0.0])
                fp = MS273.face_point(fm, 1)
                n = MS273.face_normal(fm, 1)
                tau = MS273.face_tangent(fm, 1)
                gr = gravity(g, fp...)
                sgn = side === :xminus ? 1.0 : -1.0
                @test max(abs(gr[1] - sgn * n[1]), abs(gr[2] - sgn * n[2])) <=
                      1e-14
                @test abs(dot(gr, tau)) <= 1e-14 && dot(gr, fp) < 0.0
            end
        end
        println("G273 (e) height/gravity: ok")
    end

    # ======================================================================
    # (f) the second-derivative surface: coordinate-Laplacian identities
    # ======================================================================
    @testset "map-curvature Laplacian identities" begin
        fd = make_fedata(p, p)
        d2tab = [MS273.lagrange_d2(MS273.gll_rule_cached(p), fd.gl_x[k])[3]
                 for k in 1:fd.nq]
        for (gn, g) in (("quarter", g273_geoms()[1][2]),
                        ("thick", g273_geoms()[5][2]))
            qt = g273_uniform_tree()[2]
            sq = make_semquad(qt, p; geom=g)
            err = Dict{String,Float64}()
            errC = Dict{String,Float64}()
            for lf in sq.leaves
                mc = MS273.map_curvature(g, sq, lf, fd)
                for qy in 1:fd.nq, qx in 1:fd.nq
                    for (derr, exact, nm) in
                        g273_coord_fields(g, sq, lf, fd.gl_x[qx], fd.gl_x[qy])
                        v = 0.0
                        vc = 0.0
                        for j in 1:(p + 1), i in 1:(p + 1)
                            fij = g273_field_value(nm, sq, lf, i, j, sq.xi)
                            lp = MS273.lap_phi(mc, fd, d2tab, qx, qy, i, j)
                            v += fij * lp
                            vc += fij * g273_lap_from(mc, qx, qy, derr;
                                                      drop_c=true)
                        end
                        err[nm] = max(get(err, nm, 0.0), abs(v - exact))
                        errC[nm] = max(get(errC, nm, 0.0), abs(vc - exact))
                    end
                end
            end
            for nm in ("S", "T", "S^2", "T^2", "S*T")
                @test err[nm] < 1.0e-11      # measured at round-off
            end
            # dropping the map-curvature term (`C`) is an O(1) failure on the
            # annulus: this is the term the box cannot see (cut-3 F4)
            @test maximum(values(errC)) > 0.1
            println("G273 (f) ", rpad(gn, 8), " lap identities err=",
                    maximum(values(err)), "  drop-C err=",
                    maximum(values(errC)))
        end
    end

    # ======================================================================
    # (f2) the same surface end to end: FD Laplacian of a smooth nodal field
    # ======================================================================
    @testset "FD Laplacian of a smooth nodal field" begin
        g = g273_geoms()[1][2]                     # quarter wedge
        qt = g273_uniform_tree()[2]
        sq = make_semquad(qt, p; geom=g)
        fd = make_fedata(p, p)
        d2tab = [MS273.lagrange_d2(MS273.gll_rule_cached(p), fd.gl_x[k])[3]
                 for k in 1:fd.nq]
        smooth(x, y) = sin(2.0 * (x - g.R0)) * cos(3.0 * (y - g.R0)) +
                       0.2 * x * y
        thn = [smooth(c...) for c in sq.coords]
        fthev(x, y) = semquad_eval(sq, thn, x, y)
        δ = 1.0e-2 * (g.R1 - g.R0) / (qt.nx0 * (1 << qt.maxlevel))
        function fd4lap(x, y)
            fxx = (-fthev(x + 2δ, y) + 16.0 * fthev(x + δ, y) - 30.0 * fthev(x, y) +
                   16.0 * fthev(x - δ, y) - fthev(x - 2δ, y)) / (12.0 * δ * δ)
            fyy = (-fthev(x, y + 2δ) + 16.0 * fthev(x, y + δ) - 30.0 * fthev(x, y) +
                   16.0 * fthev(x, y - δ) - fthev(x, y - 2δ)) / (12.0 * δ * δ)
            return fxx + fyy
        end
        maxfull = 0.0
        maxdrop = 0.0
        scale = 0.0
        ntest = 0
        for lf in sq.leaves
            mc = MS273.map_curvature(g, sq, lf, fd)
            ids = sq.elem[lf]
            for qy in 1:fd.nq, qx in 1:fd.nq
                x, y = MS273.quad_point(leaf_metric(g, sq, lf, fd), sq, lf,
                                        fd, qx, qy)
                r = hypot(x, y)
                # keep the FD stencil strictly inside the wedge (the seam rays
                # and the two circles are the boundaries of the field's domain)
                ang = atan(y, x)
                (r > g.R0 + 3.0 * δ && r < g.R1 - 3.0 * δ &&
                 3.0 * δ / r < ang < g.Φ - 3.0 * δ / r) || continue
                ld = 0.0
                ldc = 0.0
                for j in 1:(p + 1), i in 1:(p + 1)
                    k = (j - 1) * (p + 1) + i
                    lp = MS273.lap_phi(mc, fd, d2tab, qx, qy, i, j)
                    ld += thn[ids[k]] * lp
                    dxi = fd.dphi_v[i, qx] * fd.phi_v[j, qy]
                    deta = fd.phi_v[i, qx] * fd.dphi_v[j, qy]
                    d2xi = d2tab[qx][i] * fd.phi_v[j, qy]
                    d2eta = fd.phi_v[i, qx] * d2tab[qy][j]
                    ldc += thn[ids[k]] * g273_lap_from(
                        mc, qx, qy, (dxi, deta, d2xi,
                                     fd.dphi_v[i, qx] * fd.dphi_v[j, qy],
                                     d2eta); drop_c=true)
                end
                lfd = fd4lap(x, y)
                maxfull = max(maxfull, abs(ld - lfd))
                maxdrop = max(maxdrop, abs(ldc - lfd))
                scale = max(scale, abs(lfd))
                ntest += 1
            end
        end
        @test ntest >= 20
        @test scale > 0.0
        @test maxfull / scale < 1.0e-4       # measured ~3e-10
        @test maxdrop / scale > 0.1          # a dropped curvature term fails
        println("G273 (f2) FD Laplacian: points=", ntest,
                " full rel=", maxfull / scale, "  drop-C rel=", maxdrop / scale)
    end

    # ======================================================================
    # (g) canonical identity, cache stamp, constructor validation
    # ======================================================================
    @testset "canonical identity and constructor validation" begin
        gq = PolarAnnulus(0.55, 1.55, 0.5pi)
        @test geometry_name(gq) === :PolarAnnulus
        # exact checkpoint string: R0, R1, Phi and the seam are all in it (a
        # dropped parameter fails the equality; the seam cannot be dropped)
        @test geometry_identity(gq) ==
              "PolarAnnulus(R0=0.55,R1=1.55,Phi=1.5707963267948966,seam=:wall)"
        @test geometry_identity(PolarAnnulus(1.22, 2.22, pi)) ==
              "PolarAnnulus(R0=1.22,R1=2.22,Phi=3.141592653589793,seam=:wall)"
        @test geometry_identity(PolarAnnulus(1.0, 2.0, 2.0pi)) ==
              "PolarAnnulus(R0=1.0,R1=2.0,Phi=6.283185307179586,seam=:wall)"
        # equal parameters give an equal identity and stamp
        @test geometry_identity(PolarAnnulus(0.55, 1.55, 0.5pi; seam=:wall)) ==
              geometry_identity(gq)
        @test geometry_stamp(PolarAnnulus(0.55, 1.55, 0.5pi)) ==
              geometry_stamp(gq)
        # each parameter moves the identity and the cache stamp
        base_id = geometry_identity(gq)
        base_st = geometry_stamp(gq)
        variants = (PolarAnnulus(0.56, 1.55, 0.5pi),
                    PolarAnnulus(0.55, 1.56, 0.5pi),
                    PolarAnnulus(0.55, 1.55, 0.25pi),
                    PolarAnnulus(0.55, 1.55, pi))
        for v in variants
            @test geometry_identity(v) != base_id
            @test geometry_stamp(v) != base_st
        end
        @test allunique([base_id; [geometry_identity(v) for v in variants]])
        @test allunique([base_st; [geometry_stamp(v) for v in variants]])
        # the seam is part of the identity: a ring record cannot alias a wedge
        # record (the :periodic object itself is refused, see the next block)
        @test geometry_identity(gq) !=
              "PolarAnnulus(R0=0.55,R1=1.55,Phi=1.5707963267948966,seam=:periodic)"
        # the annulus never collides with the box in the cut-5 cache key
        @test geometry_stamp(gq) != geometry_stamp(CartesianBox(1.0))
        # constructor validation
        @test_throws ArgumentError PolarAnnulus(0.0, 1.0, 0.5pi)
        @test_throws ArgumentError PolarAnnulus(-1.0, 1.0, 0.5pi)
        @test_throws ArgumentError PolarAnnulus(1.0, 1.0, 0.5pi)
        @test_throws ArgumentError PolarAnnulus(2.0, 1.0, 0.5pi)
        @test_throws ArgumentError PolarAnnulus(1.0, 2.0, 0.0)
        @test_throws ArgumentError PolarAnnulus(1.0, 2.0, -0.5)
        @test_throws ArgumentError PolarAnnulus(1.0, 2.0, 2.0pi + 1.0e-9)
        @test_throws ArgumentError PolarAnnulus(NaN, 2.0, pi)
        @test_throws ArgumentError PolarAnnulus(1.0, Inf, pi)
        @test_throws ArgumentError PolarAnnulus(1.0, 2.0, NaN)
        @test_throws ArgumentError PolarAnnulus(1.0, 2.0, Inf)
        @test_throws ArgumentError PolarAnnulus(1.0, 2.0, pi; seam=:ring)
        # the boundaries of the allowed ranges are accepted; Real arguments convert
        @test geometry_name(PolarAnnulus(1.0, 2.0, 2.0pi)) === :PolarAnnulus
        @test geometry_name(PolarAnnulus(1.0, 2.0, 1.0e-6)) === :PolarAnnulus
        @test geometry_name(PolarAnnulus(1, 2, 1; seam=:wall)) === :PolarAnnulus
        @test geometry_name(PolarAnnulus(1, 2, 1, :wall)) === :PolarAnnulus
        # a non-unit gap is accepted by the map (all formulas carry R1-R0)
        gt = PolarAnnulus(1.0, 3.0, 0.5pi)
        @test logical_map(gt, 1.0, 0.0) == (3.0, 0.0)
        # ... while the published 1.22/2.22 pair really is a non-exactly-unit
        # gap in Float64 (the constructor must not demand R1-R0 == 1)
        gp = PolarAnnulus(1.22, 2.22, pi)
        @test gp.R1 - gp.R0 != 1.0
        @test abs(gp.R1 - gp.R0 - 1.0) <= 1e-15
    end

    # ======================================================================
    # (h) the seam decision
    # ======================================================================
    @testset "seam mode decision" begin
        gq = PolarAnnulus(1.0, 2.0, 0.5pi)
        # :wall is the wedge: four walls, seam tags present, the radial sides are
        # real walls of length R1-R0 each
        bnd = boundaries(gq)
        @test all(b -> b.kind === :wall, bnd)
        @test bnd[3].tag === :seam_lo && bnd[4].tag === :seam_hi
        qt = g273_uniform_tree()[2]
        sq = make_semquad(qt, p; geom=gq)
        @test abs(boundary_measure(gq, sq, :yminus) - (gq.R1 - gq.R0)) <= 1e-13
        # :periodic is refused with an actionable message (what is missing and
        # what to do instead), because the seam is a quadtree topology feature
        perr = try
            PolarAnnulus(1.0, 2.0, 2.0pi; seam=:periodic)
            nothing
        catch e
            e
        end
        @test perr isa ArgumentError
        pmsg = sprint(showerror, perr)
        @test occursin(":periodic", pmsg)
        @test occursin("topology", pmsg)
        @test occursin("seam = :wall", pmsg)
        @test occursin("ring", pmsg)
        # a full-turn *cut ring* is a wedge (Phi = 2pi, seam = :wall): the map is
        # 2pi-periodic to round-off, but the geometry does NOT treat the seam as
        # glued -- the registry has four walls and the perimeter counts the two
        # radial sides on top of the circles
        gr = PolarAnnulus(1.0, 2.0, 2.0pi)
        @test gr.seam === :wall
        x0, y0 = logical_map(gr, 0.4, 0.0)
        x1, y1 = logical_map(gr, 0.4, 1.0)
        @test hypot(x1 - x0, y1 - y0) <= 1e-15 * (gr.R0 + gr.R1)
        sqr = make_semquad(qt, p; geom=gr)
        per = perimeter(gr, sqr, MS273._semquad_boundary_face_mask(qt, sqr))
        ring = gr.Φ * (gr.R0 + gr.R1)
        @test abs(per - (ring + 2.0 * (gr.R1 - gr.R0))) <= 1e-12 * per
        @test per - ring > 0.5 * (gr.R1 - gr.R0)   # the seam is counted, not glued
        @test all(b -> b.kind === :wall, boundaries(gr))
        # the circles are still the Nusselt walls whatever the seam decision
        @test nu_wall_sides(gr) == (:xminus, :xplus)
        @test nu_wall_sides(gq) == (:xminus, :xplus)
    end
end
