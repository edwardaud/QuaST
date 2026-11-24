# G266: the geometry/mapping layer (stage 0, cut 1).
#
# Two jobs, both in-process (no recorded SHA literal, so a Julia/LAPACK upgrade
# cannot make them fail spuriously):
#
#   (a) the box path is the *historical arithmetic*: the geometry-built
#       coordinates, node ids, leaf/face metrics, transfer, AMR indicator and
#       checkpoint geometry record are compared, bit for bit, against verbatim
#       historical implementations carried in this file as the oracle (the G217
#       pattern);
#   (b) the map identities on the two validation geometries: point -> inverse
#       -> point, det J against an independent analytic formula and against a
#       4th-order finite difference of the map, outward unit normals against the
#       analytic normal, face Jacobians/areas against analytic lengths/areas,
#       and the partition-of-unity / patch test on the mapped space.
#
# Why the assertions are not tautological:
#   * the coordinates/ids are rebuilt by the *old* constructor body here (E1-E3
#     + the round(x/tol) dedup) and compared to `make_semquad`'s geometry-built
#     output -- the new code is a different code path, not a re-evaluation of
#     the same expression;
#   * the leaf/face metric assertions pin the exact operand and association of
#     each E-table expression (`0.25*hx*hy`, `2.0/hx`, `dphi*phi*ihx`,
#     `0.5*h*gl_w`): a reassociation is a different Float64 and fails the
#     `isequal` test even when it is mathematically equal;
#   * the transfer/indicator reconstructions consume the geometry API
#     (`element_map`, `element_inverse`, `locate_leaf`, `leaf_metric`,
#     `element_jacobian`), while `transfer_semquad`/`semquad_gradient_jump`
#     still run the historical bodies, so the two sides are independent;
#   * the map-identity checks use an analytic formula written here (or a finite
#     difference of the public map), never `element_jacobian` itself, so a
#     wrong Jacobian cannot make them pass;
#   * the patch test asks the mapped interpolation to reproduce a known
#     polynomial of the *global logical* coordinates (exact for every
#     geometry) and, for the affine map, of the physical coordinates; a wrong
#     inverse map, wrong leaf location or wrong node coordinate makes it fail;
#   * the checkpoint assertions check the driver wrote the canonical identity
#     into the config and that `_restart_guard` refuses a changed geometry --
#     the guard is the production code path, not a re-implementation.

using Test, QuaST, LinearAlgebra

const MS266 = QuaST

# ---------------------------------------------------------------------------
# Historical oracle #1: the pre-cut constructor body (GEOM_DESIGN E1-E3 plus the
# `round(x/tol)` node hash).  Returns the coordinates and the element ids.
# ---------------------------------------------------------------------------
function g266_hist_semquad(qt::Quadtree, p::Int)
    rule = MS266.gll_rule_cached(p)
    n = p + 1
    tol = 1.0e-12
    leaves = sort(collect(qt.leaves))
    coords = NTuple{2,Float64}[]
    idmap = Dict{NTuple{2,Int},Int}()
    elem = Dict{NTuple{3,Int},Vector{Int}}()
    for lf in leaves
        hx = MS266.leaf_hx(qt, lf[1]); hy = MS266.leaf_hy(qt, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = Vector{Int}(undef, n * n)
        for j in 1:n, i in 1:n
            x = x0 + 0.5 * (rule.nodes[i] + 1.0) * hx
            y = y0 + 0.5 * (rule.nodes[j] + 1.0) * hy
            key = (round(Int, x / tol), round(Int, y / tol))
            gid = get(idmap, key, 0)
            if gid == 0
                push!(coords, (x, y))
                gid = length(coords)
                idmap[key] = gid
            end
            ids[(j - 1) * n + i] = gid
        end
        elem[lf] = ids
    end
    return (coords=coords, elem=elem, N=length(coords), leaves=leaves)
end

# Historical oracle #2: nodal transfer (verbatim G217's oracle).
function g266_hist_transfer(old::SemQuad, uold::Vector{Float64}, new::SemQuad)
    oldleaves = Set(old.leaves)
    unew = zeros(Float64, new.N)
    n = new.p + 1
    for lf in new.leaves
        hx = MS266.leaf_hx(new, lf[1]); hy = MS266.leaf_hy(new, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = new.elem[lf]
        for j in 1:n, i in 1:n
            x = x0 + 0.5 * (new.xi[i] + 1.0) * hx
            y = y0 + 0.5 * (new.xi[j] + 1.0) * hy
            unew[ids[(j - 1) * n + i]] =
                MS266._semquad_eval_at(old, uold, x, y, oldleaves)
        end
    end
    return unew
end

# Historical oracle #3: the per-pair normal-derivative kernel and the full AMR
# indicator, verbatim from G217's oracle (the pre-refactor code).
function g266_hist_subface_derivs(sq::SemQuad, L, R, dir, u)
    p = sq.p
    n = p + 1
    hL = MS266.leaf_hy(sq, L[1]); hR = MS266.leaf_hy(sq, R[1])
    hxL = MS266.leaf_hx(sq, L[1]); hxR = MS266.leaf_hx(sq, R[1])
    xL = L[2] * hxL; yL = L[3] * hL
    xR = R[2] * hxR; yR = R[3] * hR
    rule = gll_rule(p)
    BL = zeros(n, n)
    vertical = dir === :xplus || dir === :xminus
    if vertical
        wq = 0.5 * hR .* sq.w
        iFL = dir === :xplus ? n : 1
        iFR = dir === :xplus ? 1 : n
        pos = yR .+ 0.5 .* (sq.xi .+ 1) .* hR
        xiL = 2.0 .* (pos .- yL) ./ hL .- 1.0
        for q in 1:n
            Lb, _ = lagrange_eval(rule, xiL[q])
            BL[q, :] .= Lb
        end
        cL = (2.0 / hxL) .* sq.D[iFL, :]
        cR = (2.0 / hxR) .* sq.D[iFR, :]
        gidL = (i, a) -> sq.elem[L][(a - 1) * n + i]
        gidR = (i, q) -> sq.elem[R][(q - 1) * n + i]
    else
        wq = 0.5 * hxR .* sq.w
        iFL = dir === :yplus ? n : 1
        iFR = dir === :yplus ? 1 : n
        pos = xR .+ 0.5 .* (sq.xi .+ 1) .* hxR
        xiL = 2.0 .* (pos .- xL) ./ hxL .- 1.0
        for q in 1:n
            Lb, _ = lagrange_eval(rule, xiL[q])
            BL[q, :] .= Lb
        end
        cL = (2.0 / hL) .* sq.D[iFL, :]
        cR = (2.0 / hR) .* sq.D[iFR, :]
        gidL = (i, a) -> sq.elem[L][(i - 1) * n + a]
        gidR = (i, q) -> sq.elem[R][(q - 1) * n + i]
    end
    dL = zeros(Float64, n)
    dR = zeros(Float64, n)
    for q in 1:n
        for a in 1:n
            da = 0.0
            for i in 1:n
                da += cL[i] * u[gidL(i, a)]
            end
            dL[q] += BL[q, a] * da
        end
        for i in 1:n
            dR[q] += cR[i] * u[gidR(i, q)]
        end
    end
    return wq, dL, dR
end

function g266_hist_indicator(sq::SemQuad, qt::Quadtree, u::Vector{Float64})
    n = sq.p + 1
    eta = Dict{NTuple{3,Int},Float64}()
    for lf in sq.leaves
        eta[lf] = 0.0
    end
    for pf in face_pairs(qt)
        L, R, dir = pf.leafL, pf.leafR, pf.dir
        wq, dL, dR = g266_hist_subface_derivs(sq, L, R, dir, u)
        J = 0.0
        for q in eachindex(wq)
            J += wq[q] * (dL[q] - dR[q])^2
        end
        normal_x = dir === :xplus || dir === :xminus
        hL = normal_x ? MS266.leaf_hx(sq, L[1]) : MS266.leaf_hy(sq, L[1])
        hR = normal_x ? MS266.leaf_hx(sq, R[1]) : MS266.leaf_hy(sq, R[1])
        eta[L] += 0.5 * hL * J
        eta[R] += 0.5 * hR * J
    end
    for lf in sq.leaves
        h = MS266.leaf_hy(sq, lf[1])
        hx = MS266.leaf_hx(sq, lf[1])
        ids = sq.elem[lf]
        for dir in (:xminus, :xplus, :yminus, :yplus)
            MS266.has_no_face_neighbor(qt, lf, dir) || continue
            (dir === :xminus || dir === :xplus) && continue
            jF = dir === :yminus ? 1 : n
            hc = 2.0 / h
            cB = [hc * sq.D[jF, i] for i in 1:n]
            for i in 1:n
                d = 0.0
                for j in 1:n
                    d += cB[j] * u[ids[(j - 1) * n + i]]
                end
                wf = 0.5 * hx * sq.w[i]
                eta[lf] += 0.5 * h * wf * d^2
            end
        end
    end
    return eta
end

# ---------------------------------------------------------------------------
# Geometry reconstruction #1: nodal transfer through the geometry API only.
# ---------------------------------------------------------------------------
function g266_geo_transfer(old::SemQuad, uold::Vector{Float64}, new::SemQuad, g)
    oldleaves = Set(old.leaves)
    unew = zeros(Float64, new.N)
    n = new.p + 1
    rule = MS266.gll_rule_cached(new.p)
    w = MS266.SemQuadEvalWork(n)
    seen = falses(new.N)
    for lf in new.leaves
        ids = new.elem[lf]
        for j in 1:n, i in 1:n
            gid = ids[(j - 1) * n + i]
            seen[gid] && continue
            seen[gid] = true
            x, y = element_map(g, new, lf, new.xi[i], new.xi[j])
            lfo = locate_leaf(g, old, x, y, oldleaves)
            xi, eta = element_inverse(g, old, lfo, x, y)
            lagrange_values!(w.Lb, rule, xi)
            lagrange_values!(w.Lc, rule, eta)
            oids = old.elem[lfo]
            v = 0.0
            for jj in 1:n, ii in 1:n
                v += uold[oids[(jj - 1) * n + ii]] * w.Lb[ii] * w.Lc[jj]
            end
            unew[gid] = v
        end
    end
    return unew
end

# ---------------------------------------------------------------------------
# Geometry reconstruction #2: the AMR indicator through the geometry API.  The
# face point and the coarse reference abscissa come from element_map /
# element_inverse; the face weight from the face Jacobian |d(x,y)/d(face
# parameter)| of element_jacobian; the derivative scales from leaf_metric.
# ---------------------------------------------------------------------------
function g266_geo_indicator(sq::SemQuad, qt::Quadtree, u::Vector{Float64})
    g = sq.geom
    n = sq.p + 1
    rule = gll_rule(sq.p)
    eta = Dict{NTuple{3,Int},Float64}()
    for lf in sq.leaves
        eta[lf] = 0.0
    end
    for pf in face_pairs(qt, g)
        L, R, dir = pf.leafL, pf.leafR, pf.dir
        lmL = leaf_metric(g, sq, L)
        lmR = leaf_metric(g, sq, R)
        if dir === :xplus || dir === :xminus
            iFL = dir === :xplus ? n : 1
            iFR = dir === :xplus ? 1 : n
            rf = dir === :xplus ? -1.0 : 1.0     # R's face that touches L
            cL = [lmL.ihx * sq.D[iFL, i] for i in 1:n]
            cR = [lmR.ihx * sq.D[iFR, i] for i in 1:n]
            J = 0.0
            for q in 1:n
                x, y = element_map(g, sq, R, rf, sq.xi[q])
                _, xiL = element_inverse(g, sq, L, x, y)
                Lb, _ = lagrange_eval(rule, xiL)
                Jq, _ = element_jacobian(g, sq, R, rf, sq.xi[q])
                wq = hypot(Jq[2], Jq[4]) * sq.w[q]
                dL = 0.0
                for a in 1:n
                    da = 0.0
                    for i in 1:n
                        da += cL[i] * u[sq.elem[L][(a - 1) * n + i]]
                    end
                    dL += Lb[a] * da
                end
                dR = 0.0
                for i in 1:n
                    dR += cR[i] * u[sq.elem[R][(q - 1) * n + i]]
                end
                J += wq * (dL - dR)^2
            end
            eta[L] += 0.5 * lmL.hx * J
            eta[R] += 0.5 * lmR.hx * J
        else
            jFL = dir === :yplus ? n : 1
            jFR = dir === :yplus ? 1 : n
            rf = dir === :yplus ? -1.0 : 1.0
            cL = [lmL.ihy * sq.D[jFL, i] for i in 1:n]
            cR = [lmR.ihy * sq.D[jFR, i] for i in 1:n]
            J = 0.0
            for q in 1:n
                x, y = element_map(g, sq, R, sq.xi[q], rf)
                xiL, _ = element_inverse(g, sq, L, x, y)
                Lb, _ = lagrange_eval(rule, xiL)
                Jq, _ = element_jacobian(g, sq, R, sq.xi[q], rf)
                wq = hypot(Jq[1], Jq[3]) * sq.w[q]
                dL = 0.0
                for a in 1:n
                    da = 0.0
                    for i in 1:n
                        da += cL[i] * u[sq.elem[L][(i - 1) * n + a]]
                    end
                    dL += Lb[a] * da
                end
                dR = 0.0
                for i in 1:n
                    dR += cR[i] * u[sq.elem[R][(q - 1) * n + i]]
                end
                J += wq * (dL - dR)^2
            end
            eta[L] += 0.5 * lmL.hy * J
            eta[R] += 0.5 * lmR.hy * J
        end
    end
    # boundary terms, y-walls only (the x-walls are insulating in the indicator)
    for lf in sq.leaves
        lm = leaf_metric(g, sq, lf)
        ids = sq.elem[lf]
        for dir in (:xminus, :xplus, :yminus, :yplus)
            MS266.has_no_face_neighbor(qt, lf, dir) || continue
            (dir === :xminus || dir === :xplus) && continue
            jF = dir === :yminus ? 1 : n
            cB = [lm.ihy * sq.D[jF, i] for i in 1:n]
            for i in 1:n
                d = 0.0
                for j in 1:n
                    d += cB[j] * u[ids[(j - 1) * n + i]]
                end
                wf = 0.5 * lm.hx * sq.w[i]
                eta[lf] += 0.5 * lm.hy * wf * d^2
            end
        end
    end
    return eta
end

# 4th-order central difference of a scalar function.
function g266_fd4(f, x; h=1.0e-4)
    return (-f(x + 2h) + 8f(x + h) - 8f(x - h) + f(x - 2h)) / (12h)
end

# Total length of one boundary side of a mapped geometry, integrated with a
# high-order Gauss rule through the geometry face metric.
function g266_boundary_length(g, qt, side, p=2; nq=24)
    sq = make_semquad(qt, p; geom=g)
    fd = make_fedata(2, 1; nq=nq)
    tot = 0.0
    for lf in sq.leaves
        MS266.has_no_face_neighbor(qt, lf, side) || continue
        fm = face_metric(g, sq, lf, side, fd)
        for q in 1:fd.nq
            tot += face_jacobian(fm, q) * fd.gl_w[q]
        end
    end
    return tot
end

# Total physical area of a mapped geometry, integrated from detJ at the Gauss
# points of every leaf.
function g266_total_area(g, qt, p=2; nq=12)
    sq = make_semquad(qt, p; geom=g)
    fd = make_fedata(2, 2; nq=nq)
    tot = 0.0
    for lf in sq.leaves
        for qy in 1:fd.nq, qx in 1:fd.nq
            _, dj = element_jacobian(g, sq, lf, fd.gl_x[qx], fd.gl_x[qy])
            tot += dj * fd.gl_w[qx] * fd.gl_w[qy]
        end
    end
    return tot
end

# The 64-point arc length of the CurvedBox top wall, y = 1 + a*sin(pi*x/Lx),
# written here from the analytic curve (not from the geometry object).
function g266_curved_top_length(Lx, a)
    x, w = gauss_legendre(64)
    s = 0.0
    for k in eachindex(x)
        S = 0.5 * (x[k] + 1.0)
        s += 0.5 * w[k] * sqrt(1.0 + (a * pi / Lx * cos(pi * S))^2)
    end
    return Lx * s
end

# A mixed/hanging tree, the production shapes, and the wall-graded tree.
function g266_trees()
    out = Pair{String,Quadtree}[]
    q = Quadtree(3)
    for _ in 1:2, lf in collect(q.leaves)
        refine!(q, lf)
    end
    balance!(q)
    push!(out, "unit" => q)
    q5 = Quadtree(2; cells_x=5, cells_y=2)
    for _ in 1:2, lf in collect(q5.leaves)
        refine!(q5, lf)
    end
    balance!(q5)
    push!(out, "cells52" => q5)
    qs = Quadtree(2; Lx=2.5)
    for _ in 1:2, lf in collect(qs.leaves)
        refine!(qs, lf)
    end
    balance!(qs)
    push!(out, "stretched" => qs)
    qm = Quadtree(2)
    for lf in collect(qm.leaves)
        refine!(qm, lf)
    end
    for lf in sort(collect(qm.leaves))[1:3]
        refine!(qm, lf)
    end
    balance!(qm)
    push!(out, "mixed" => qm)
    push!(out, "wall" => wall_graded_tree(3, 6))
    return out
end

@testset "G266 geometry identity and map identities" begin

    # ======================================================================
    # (a.1) box coordinates and node ids: the geometry-built space equals the
    # historical constructor body, bit for bit.
    # ======================================================================
    @testset "box coordinates and node identity" begin
        for (tag, qt) in g266_trees(), p in (2, 3)
            sq = make_semquad(qt, p)
            @test sq.geom isa CartesianBox
            @test sq.geom === CartesianBox(qt)
            @test geometry_identity(sq.geom) ==
                  string("CartesianBox(Lx=", qt.Lx, ")")
            hist = g266_hist_semquad(qt, p)
            @test sq.N == hist.N
            @test sq.leaves == hist.leaves
            @test isequal(sq.coords, hist.coords)
            @test sq.elem == hist.elem
            @test length(unique(sq.coords)) == sq.N
        end
        ss = make_semstokes(g266_trees()[4].second, 2)
        @test ss.su.geom === ss.sp.geom
        @test ss.su.geom isa CartesianBox
    end

    # ======================================================================
    # (a.2) box leaf and face metric: every field is the historical operand in
    # the historical association (GEOM_DESIGN E4/E5/E8/E10).
    # ======================================================================
    @testset "box leaf and face metric expressions" begin
        for (tag, qt) in g266_trees()[1:4], p in (2, 3)
            fd = make_fedata(p, p + 2)   # the historical assembly quadrature
            sq = make_semquad(qt, p)
            g = CartesianBox(qt)
            nbad = 0
            for lf in sq.leaves[1:min(12, length(sq.leaves))]
                hx = MS266.leaf_hx(qt, lf[1]); hy = MS266.leaf_hy(qt, lf[1])
                x0 = lf[2] * hx; y0 = lf[3] * hy
                lm = leaf_metric(g, sq, lf)
                nbad += isequal(lm.x0, x0) &&
                        isequal(lm.y0, y0) &&
                        isequal(lm.hx, hx) && isequal(lm.hy, hy) &&
                        isequal(lm.ihx, 2.0 / hx) &&
                        isequal(lm.ihy, 2.0 / hy) &&
                        isequal(lm.wfac, 0.25 * hx * hy) ? 0 : 1
                # E4 and E3: the historical volume weight and point map
                for qy in 1:fd.nq, qx in 1:fd.nq
                    w_hist = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
                    qp = quad_point(lm, sq, lf, fd, qx, qy)
                    nbad += isequal(quad_weight(lm, fd, qx, qy), w_hist) &&
                            isequal(qp[1], x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx) &&
                            isequal(qp[2], y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy) ? 0 : 1
                end
                # E5: dphi*phi*(2.0/h), in that association
                for j in 1:p + 1, i in 1:p + 1, qy in 1:fd.nq, qx in 1:fd.nq
                    bg = basis_grad(lm, fd, qx, qy, i, j)
                    nbad += isequal(bg[1], fd.dphi_v[i, qx] * fd.phi_v[j, qy] *
                                           (2.0 / hx)) &&
                            isequal(bg[2], fd.phi_v[i, qx] * fd.dphi_v[j, qy] *
                                           (2.0 / hy)) ? 0 : 1
                end
                # E8/E10: face weight, face point, unit normal and tangent
                for side in (:xminus, :xplus, :yminus, :yplus)
                    fm = face_metric(g, sq, lf, side, fd)
                    if side === :xminus || side === :xplus
                        xf = side === :xminus ? x0 : x0 + hx
                        nx_an = side === :xminus ? -1.0 : 1.0
                        nbad += isequal(face_jacobian(fm), 0.5 * hy) &&
                                face_normal(fm) == (nx_an, 0.0) &&
                                face_tangent(fm) == (0.0, 1.0) ? 0 : 1
                        for q in 1:fd.nq
                            fp = face_point(g, sq, lf, side, fd, q)
                            nbad += isequal(face_weight(fm, fd, q),
                                            0.5 * hy * fd.gl_w[q]) &&
                                    isequal(fp[1], xf) &&
                                    isequal(fp[2], y0 + 0.5 * (fd.gl_x[q] + 1.0) * hy) ? 0 : 1
                        end
                    else
                        yf = side === :yminus ? y0 : y0 + hy
                        ny_an = side === :yminus ? -1.0 : 1.0
                        nbad += isequal(face_jacobian(fm), 0.5 * hx) &&
                                face_normal(fm) == (0.0, ny_an) &&
                                face_tangent(fm) == (1.0, 0.0) ? 0 : 1
                        for q in 1:fd.nq
                            fp = face_point(g, sq, lf, side, fd, q)
                            nbad += isequal(face_weight(fm, fd, q),
                                            0.5 * hx * fd.gl_w[q]) &&
                                    isequal(fp[1], x0 + 0.5 * (fd.gl_x[q] + 1.0) * hx) &&
                                    isequal(fp[2], yf) ? 0 : 1
                        end
                    end
                end
            end
            @test nbad == 0
            # the map, inverse and inverse Jacobian are the E11/E5 scalars
            nbad2 = 0
            for lf in sq.leaves[1:min(8, length(sq.leaves))]
                lm = leaf_metric(g, sq, lf)
                for (xi, eta) in ((-0.3, 0.7), (0.1, -0.2), (1.0, -1.0))
                    x, y = element_map(g, sq, lf, xi, eta)
                    xr, er = element_inverse(g, sq, lf, x, y)
                    nbad2 += isequal(x, lm.x0 + 0.5 * (xi + 1.0) * lm.hx) &&
                             isequal(y, lm.y0 + 0.5 * (eta + 1.0) * lm.hy) &&
                             isequal(xr, 2.0 * (x - lm.x0) / lm.hx - 1.0) &&
                             isequal(er, 2.0 * (y - lm.y0) / lm.hy - 1.0) &&
                             jacobian_inverse(g, sq, lf, xi, eta) ==
                                 (2.0 / lm.hx, 0.0, 0.0, 2.0 / lm.hy) ? 0 : 1
                end
            end
            @test nbad2 == 0
        end
        @test perimeter(CartesianBox(2.5)) == 2.0 * (2.5 + 1.0)
    end

    # ======================================================================
    # (a.3) topology: the logical face pairs reproduce the historical list, and
    # the logical hanging constraints reproduce the historical weights.
    # ======================================================================
    @testset "logical face pairs and constraints" begin
        for (tag, qt) in g266_trees()
            hist = face_pairs(qt)
            logi = face_pairs_logical(qt)
            @test length(hist) == length(logi)
            @test all(((h.leafL, h.leafR, h.dir) == (l.leafL, l.leafR, l.dir)
                       for (h, l) in zip(hist, logi)))
            @test face_pairs(qt, CartesianBox(qt)) == hist
            if qt.nx0 == 1 && qt.ny0 == 1
                @test hist == logi
            else
                # multi-root: the historical offset/len are physical, the
                # logical ones are (S,T); the box map relates them exactly
                nbad = 0
                for (h, l) in zip(hist, logi)
                    xface = h.dir === :xplus || h.dir === :xminus
                    scale = xface ? 1.0 : qt.Lx
                    nbad += (isapprox(h.offset, scale * l.offset;
                                      rtol=1.0e-15, atol=1.0e-15) &&
                             isapprox(h.len, scale * l.len;
                                      rtol=1.0e-15, atol=1.0e-15)) ? 0 : 1
                end
                @test nbad == 0
            end
            @test face_pairs(qt, CurvedBox()) == logi
        end
        # hanging constraints: the historical physical formula and the logical
        # formula are the same Float64s on the box (different code paths)
        for p in (2, 3)
            for (tag, qt) in g266_trees()[4:5]
                sqb = make_semquad(qt, p)
                cb = semquad_constraints(sqb, qt; geom=CartesianBox(qt))
                cl = semquad_constraints(sqb, qt; geom=CurvedBox())
                @test length(cb) > 0
                @test Set(keys(cb)) == Set(keys(cl))
                @test all(isequal(cb[c][1], cl[c][1]) &&
                          isequal(cb[c][2], cl[c][2]) for c in keys(cb))
            end
        end
    end

    # ======================================================================
    # (a.4) transfer and AMR indicator through the geometry are bitwise the
    # existing implementations.
    # ======================================================================
    @testset "transfer and indicator through the geometry" begin
        qt_old = g266_trees()[4].second
        sq_old = make_semquad(qt_old, 2)
        u = [0.3 * sinpi(x) * sinpi(y) + 0.05 * sinpi(3x) * sinpi(2y)
             for (x, y) in sq_old.coords]
        qt_new = deepcopy(qt_old)
        for lf in sort(collect(qt_new.leaves))[1:2]
            refine!(qt_new, lf)
        end
        balance!(qt_new)
        sq_new = make_semquad(qt_new, 2)
        g = CartesianBox(qt_old)
        actual = transfer_semquad(sq_old, u, sq_new; mode=:nodal)
        @test isequal(actual, g266_hist_transfer(sq_old, u, sq_new))
        @test isequal(actual, g266_geo_transfer(sq_old, u, sq_new, g))
        for (tag, qt) in g266_trees()
            sq = make_semquad(qt, 2)
            f = [0.3 * sinpi(x) * sinpi(y) + 0.05 * sinpi(3x) * sinpi(2y)
                 for (x, y) in sq.coords]
            eta_actual = semquad_gradient_jump(sq, qt, f)
            eta_hist = g266_hist_indicator(sq, qt, f)
            eta_geo = g266_geo_indicator(sq, qt, f)
            @test Set(keys(eta_actual)) == Set(sq.leaves)
            @test all(isequal(eta_actual[lf], eta_hist[lf]) for lf in sq.leaves)
            nmis = count(lf -> !isequal(eta_actual[lf], eta_geo[lf]),
                         sq.leaves)
            if nmis > 0
                println("  G266 indicator reconstruction mismatch on tree ",
                        tag, ": ", nmis, "/", length(sq.leaves), " leaves")
                for lf in sq.leaves
                    if !isequal(eta_actual[lf], eta_geo[lf])
                        println("    leaf ", lf, " actual=", eta_actual[lf],
                                " geo=", eta_geo[lf], " hist=", eta_hist[lf])
                        break
                    end
                end
            end
            @test nmis == 0
        end
    end

    # ======================================================================
    # (b.1) validation-geometry constructors refuse a non-diffeomorphism
    # ======================================================================
    @testset "validation geometry guards" begin
        @test_throws ArgumentError AffineBox((1.0, 0.0, 0.0, -1.0), (0.0, 0.0), 1.0)
        @test_throws ArgumentError AffineBox((1.0, 2.0, 2.0, 1.0), (0.0, 0.0), 1.0)
        @test_throws ArgumentError CurvedBox(1.0, 1.0)
        @test_throws ArgumentError CurvedBox(-1.5, 1.0)
        @test_throws ArgumentError CartesianBox(-1.0)
    end

    # ======================================================================
    # (b.2) map identities on AffineBox and CurvedBox
    # ======================================================================
    g_aff = AffineBox((1.3, 0.4, -0.3, 0.9), (0.1, -0.05), 1.0)
    g_cur = CurvedBox(0.2, 1.0)
    detA = g_aff.A[1] * g_aff.A[4] - g_aff.A[2] * g_aff.A[3]
    qt_b = g266_trees()[4].second
    fd6 = make_fedata(2, 1; nq=6)

    @testset "point/inverse roundtrip and mesh conformation" begin
        for g in (g_aff, g_cur)
            sq = make_semquad(qt_b, 2; geom=g)
            @test sq.geom === g
            # conformation: the logical node identity is exact, so the number
            # of distinct physical positions is the dof count and distinct
            # nodes are far apart (no rounded-coordinate collision or crack)
            @test length(unique(sq.coords)) == sq.N
            worst = Inf
            for a in 1:sq.N, b in (a + 1):sq.N
                d = max(abs(sq.coords[a][1] - sq.coords[b][1]),
                        abs(sq.coords[a][2] - sq.coords[b][2]))
                d < worst && (worst = d)
            end
            @test worst > 1.0e-12
            leaves = sort(collect(sq.leaves))
            worst_rt = 0.0
            nfound = 0
            for t in 1:40
                lf = leaves[1 + (t * 7) % length(leaves)]
                xi = -0.95 + 1.9 * ((t * 37) % 97) / 96
                eta = -0.95 + 1.9 * ((t * 53) % 89) / 88
                x, y = element_map(g, sq, lf, xi, eta)
                xr, er = element_inverse(g, sq, lf, x, y)
                worst_rt = max(worst_rt, abs(xr - xi), abs(er - eta))
                nfound += locate_leaf(g, sq, x, y, Set(sq.leaves)) == lf ? 1 : 0
            end
            @test worst_rt < 1.0e-13
            @test nfound == 40
        end
    end

    @testset "element Jacobian and its inverse" begin
        for g in (g_aff, g_cur)
            sq = make_semquad(qt_b, 3; geom=g)
            errJ = 0.0
            errdet = 0.0
            errinv = 0.0
            for lf in sort(collect(sq.leaves))[1:6]
                for (xi, eta) in ((0.3, -0.2), (-0.6, 0.4), (0.0, 0.0))
                    J, dj = element_jacobian(g, sq, lf, xi, eta)
                    # independent 4th-order finite difference of the map
                    ex = (q) -> element_map(g, sq, lf, q, eta)
                    ey = (q) -> element_map(g, sq, lf, xi, q)
                    d1x = g266_fd4(q -> ex(q)[1], xi)
                    d1y = g266_fd4(q -> ex(q)[2], xi)
                    d2x = g266_fd4(q -> ey(q)[1], eta)
                    d2y = g266_fd4(q -> ey(q)[2], eta)
                    errJ = max(errJ, abs(J[1] - d1x), abs(J[3] - d1y),
                               abs(J[2] - d2x), abs(J[4] - d2y))
                    errdet = max(errdet,
                                 abs(dj - (d1x * d2y - d2x * d1y)))
                    # analytic detJ, written here from the map definition
                    l = lf[1]
                    hs = 1.0 / (qt_b.nx0 * (1 << l))
                    ht = 1.0 / (qt_b.ny0 * (1 << l))
                    if g isa AffineBox
                        dj_an = detA * g.Lx * hs * ht / 4.0
                    else
                        S, _ = global_logical(sq, lf, xi, eta)
                        dj_an = g.Lx * hs * ht / 4.0 * (1.0 + g.a * sinpi(S))
                    end
                    errdet = max(errdet, abs(dj - dj_an) / abs(dj_an))
                    # J^{-T} is the chain rule the kernels use: J^T * Jinv = I
                    inv = jacobian_inverse(g, sq, lf, xi, eta)
                    errinv = max(errinv,
                                 abs(J[1] * inv[1] + J[3] * inv[3] - 1.0),
                                 abs(J[1] * inv[2] + J[3] * inv[4]),
                                 abs(J[2] * inv[1] + J[4] * inv[3]),
                                 abs(J[2] * inv[2] + J[4] * inv[4] - 1.0))
                end
            end
            @test errJ < 1.0e-9
            @test errdet < 1.0e-9
            @test errinv < 1.0e-13
        end
        # the affine Jacobian is constant on the leaf; the curved one is not
        sq = make_semquad(qt_b, 2; geom=g_aff)
        lf = first(sort(collect(sq.leaves)))
        J0, dj0 = element_jacobian(g_aff, sq, lf, -0.7, 0.2)
        @test all(element_jacobian(g_aff, sq, lf, xi, eta) == (J0, dj0)
                  for (xi, eta) in ((0.3, 0.4), (0.9, -0.8), (0.0, 0.0)))
        sqc = make_semquad(qt_b, 2; geom=g_cur)
        lf = first(sort(collect(sqc.leaves)))
        djs = [element_jacobian(g_cur, sqc, lf, xi, -0.2)[2]
               for xi in (-0.9, -0.3, 0.0, 0.5, 0.9)]
        @test maximum(djs) > minimum(djs) * 1.05
    end

    @testset "face normals, tangents and face Jacobians" begin
        # affine: the outward normal of the image of a side is A^{-T} n_ref
        AinvT = (1.0 / detA) .* [g_aff.A[4] -g_aff.A[3]; -g_aff.A[2] g_aff.A[1]]
        sq = make_semquad(qt_b, 2; geom=g_aff)
        worst = 0.0
        for side in (:xminus, :xplus, :yminus, :yplus), lf in sq.leaves
            MS266.has_no_face_neighbor(qt_b, lf, side) || continue
            fm = face_metric(g_aff, sq, lf, side, fd6)
            nref = side === :xminus ? (-1.0, 0.0) :
                   side === :xplus ? (1.0, 0.0) :
                   side === :yminus ? (0.0, -1.0) : (0.0, 1.0)
            n_an = AinvT * [nref[1], nref[2]]
            n_an = n_an ./ hypot(n_an[1], n_an[2])
            for q in 1:fd6.nq
                nx, ny = face_normal(fm, q)
                tx, ty = face_tangent(fm, q)
                worst = max(worst, abs(hypot(nx, ny) - 1.0),
                            abs(hypot(tx, ty) - 1.0), abs(nx * tx + ny * ty),
                            abs(nx - n_an[1]), abs(ny - n_an[2]))
                px, py = face_point(g_aff, sq, lf, side, fd6, q)
                cx, cy = logical_map(g_aff, 0.5 * g_aff.Lx, 0.5)
                @test nx * (px - cx) + ny * (py - cy) > 0.0
            end
        end
        @test worst < 1.0e-13
        # curved: three straight sides have the exact Cartesian normals, the
        # top side the analytic normal of y = 1 + a*sin(pi*x/Lx)
        sq = make_semquad(qt_b, 2; geom=g_cur)
        worst = 0.0
        for side in (:xminus, :xplus, :yminus, :yplus), lf in sq.leaves
            MS266.has_no_face_neighbor(qt_b, lf, side) || continue
            fm = face_metric(g_cur, sq, lf, side, fd6)
            for q in 1:fd6.nq
                nx, ny = face_normal(fm, q)
                worst = max(worst, abs(hypot(nx, ny) - 1.0))
                if side === :xminus
                    worst = max(worst, abs(nx + 1.0), abs(ny))
                elseif side === :xplus
                    worst = max(worst, abs(nx - 1.0), abs(ny))
                elseif side === :yminus
                    worst = max(worst, abs(nx), abs(ny + 1.0))
                else
                    px, _ = face_point(g_cur, sq, lf, side, fd6, q)
                    S = px / g_cur.Lx
                    fp = g_cur.a * pi / g_cur.Lx * cos(pi * S)
                    den = hypot(-fp, 1.0)
                    worst = max(worst, abs(nx + fp / den), abs(ny - 1.0 / den))
                end
            end
        end
        @test worst < 1.0e-13
        # face integrals: the face Jacobian integrates to the analytic length
        exp_x = hypot(g_aff.A[2], g_aff.A[4])            # eta-edge image
        exp_y = hypot(g_aff.A[1], g_aff.A[3]) * g_aff.Lx # xi-edge image
        for side in (:xminus, :xplus)
            @test abs(g266_boundary_length(g_aff, qt_b, side) - exp_x) <=
                  1.0e-13 * exp_x
        end
        for side in (:yminus, :yplus)
            @test abs(g266_boundary_length(g_aff, qt_b, side) - exp_y) <=
                  1.0e-13 * exp_y
        end
        for side in (:xminus, :xplus)
            @test abs(g266_boundary_length(g_cur, qt_b, side) - 1.0) < 1.0e-13
        end
        @test abs(g266_boundary_length(g_cur, qt_b, :yminus) - g_cur.Lx) < 1.0e-13
        exp_top = g266_curved_top_length(g_cur.Lx, g_cur.a)
        @test abs(g266_boundary_length(g_cur, qt_b, :yplus) - exp_top) <=
              1.0e-13 * exp_top
        # element areas: sum detJ*w over every leaf = analytic area
        @test abs(g266_total_area(g_aff, qt_b) - detA * g_aff.Lx) <=
              1.0e-13 * detA * g_aff.Lx
        area_c = g_cur.Lx * (1.0 + 2.0 * g_cur.a / pi)
        @test abs(g266_total_area(g_cur, qt_b) - area_c) <= 1.0e-13 * area_c
    end

    @testset "partition of unity and patch test" begin
        for g in (CartesianBox(1.0), g_aff, g_cur)
            sq = make_semquad(qt_b, 2; geom=g)
            leaves = Set(sq.leaves)
            # (i) partition of unity: the interpolant of the constant field is
            # exactly 1 at 25 points strictly inside the domain
            ones_field = ones(Float64, sq.N)
            errpou = 0.0
            for t in 1:25
                S = 0.02 + 0.96 * ((t * 41) % 101) / 100
                T = 0.02 + 0.96 * ((t * 59) % 103) / 102
                x, y = logical_map(g, S, T)
                errpou = max(errpou, abs(semquad_eval(sq, ones_field, x, y) - 1.0))
            end
            @test errpou < 1.0e-14
            # (ii) patch test for a global-logical-linear field: exact on every
            # mapped space, because S and T are affine in the reference
            # coordinates on each leaf
            fnode = Vector{Float64}(undef, sq.N)
            for (gi, (x, y)) in enumerate(sq.coords)
                lf = locate_leaf(g, sq, x, y, leaves)
                xi, eta = element_inverse(g, sq, lf, x, y)
                S, T = global_logical(sq, lf, xi, eta)
                fnode[gi] = 1.0 + 2.0 * S - 3.0 * T
            end
            errlog = 0.0
            for t in 1:25
                S = 0.03 + 0.94 * ((t * 43) % 107) / 106
                T = 0.03 + 0.94 * ((t * 61) % 109) / 108
                x, y = logical_map(g, S, T)
                errlog = max(errlog, abs(semquad_eval(sq, fnode, x, y) -
                                         (1.0 + 2.0 * S - 3.0 * T)))
            end
            @test errlog < 1.0e-13
            # (iii) physical-linear field: exact only on an affine map (the
            # curved mapped Q2 does not contain it, and that is the point)
            plin = [1.0 + 2.0 * c[1] - 0.5 * c[2] for c in sq.coords]
            errpl = 0.0
            for t in 1:25
                S = 0.03 + 0.94 * ((t * 47) % 101) / 100
                T = 0.03 + 0.94 * ((t * 67) % 103) / 102
                x, y = logical_map(g, S, T)
                errpl = max(errpl, abs(semquad_eval(sq, plin, x, y) -
                                       (1.0 + 2.0 * x - 0.5 * y)))
            end
            if g isa CurvedBox
                @test errpl > 1.0e-5     # the mapped space cannot contain it
            else
                @test errpl < 1.0e-13    # affine maps contain P1 exactly
            end
            # hanging constraints on the mapped space: a constrained fine node
            # equals the interpolation of the same global function
            cons = semquad_constraints(sq, qt_b)
            if !isempty(cons)
                @test all(abs(sum(ws .* fnode[ms]) - fnode[c]) < 1.0e-13
                          for (c, (ms, ws)) in cons)
            end
        end
    end

    @testset "leaf metric arrays (hot-loop accessors)" begin
        fd = make_fedata(2, 2)
        for g in (g_aff, g_cur)
            sq = make_semquad(qt_b, 2; geom=g)
            errp = 0.0
            errw = 0.0
            errg = 0.0
            for lf in sort(collect(sq.leaves))[1:4]
                m = leaf_metric(g, sq, lf, fd)
                @test m.nq == fd.nq
                for qy in 1:fd.nq, qx in 1:fd.nq
                    xi = fd.gl_x[qx]; eta = fd.gl_x[qy]
                    x, y = element_map(g, sq, lf, xi, eta)
                    px, py = quad_point(m, sq, lf, fd, qx, qy)
                    errp = max(errp, abs(px - x), abs(py - y))
                    J, dj = element_jacobian(g, sq, lf, xi, eta)
                    errw = max(errw, abs(quad_weight(m, fd, qx, qy) - dj))
                    inv = (J[4] / dj, -J[3] / dj, -J[2] / dj, J[1] / dj)
                    for j in 1:3, i in 1:3
                        gx, gz = basis_grad(m, fd, qx, qy, i, j)
                        dxi = fd.dphi_v[i, qx] * fd.phi_v[j, qy]
                        deta = fd.phi_v[i, qx] * fd.dphi_v[j, qy]
                        errg = max(errg, abs(gx - (inv[1] * dxi + inv[2] * deta)),
                                   abs(gz - (inv[3] * dxi + inv[4] * deta)))
                    end
                end
                # affine: the metric is constant on the leaf; curved: it is not
                if g isa AffineBox
                    @test all(m.detJ[k] == m.detJ[1] for k in eachindex(m.detJ))
                else
                    @test maximum(m.detJ) > minimum(m.detJ) * 1.02
                end
            end
            @test errp < 1.0e-14
            @test errw < 1.0e-14
            @test errg < 1.0e-14
        end
    end

    @testset "height, depth, gravity and boundary registry" begin
        # box accessors are exactly today's y / 1-y / (0,1) / (0,-1)
        for (x, y) in ((0.1, 0.2), (2.0, 0.7))
            @test height(CartesianBox(2.5), x, y) == y
            @test depth(CartesianBox(2.5), x, y) == 1.0 - y
            @test grad_height(CartesianBox(2.5), x, y) == (0.0, 1.0)
            @test height_laplacian(CartesianBox(2.5), x, y) == 0.0
            @test gravity(CartesianBox(2.5), x, y) == (0.0, -1.0)
            @test gravity_magnitude(CartesianBox(2.5), x, y) == 1.0
        end
        # mapped depth gradients are the gradient of the analytic height
        for g in (g_aff, g_cur)
            errh = 0.0
            for (x, y) in ((0.3, 0.4), (0.7, 0.2))
                gh = grad_height(g, x, y)
                errh = max(errh,
                           abs(g266_fd4(q -> height(g, q, y), x) - gh[1]),
                           abs(g266_fd4(q -> height(g, x, q), y) - gh[2]),
                           abs(depth(g, x, y) + height(g, x, y) - 1.0))
                gv = gravity(g, x, y)
                @test abs(hypot(gv[1], gv[2]) - 1.0) < 1.0e-14
                @test gv[1] * gh[1] + gv[2] * gh[2] < 0.0   # downward
                @test gravity_magnitude(g, x, y) == 1.0
                @test measure_weight(g, x, y) == 1.0
            end
            @test errh < 1.0e-8
        end
        # the registry: every side is a wall for the three stage-0 geometries
        for g in (CartesianBox(1.0), g_aff, g_cur)
            b = boundaries(g)
            @test length(b) == 4
            @test [e.side for e in b] == [:xminus, :xplus, :yminus, :yplus]
            @test all(e.kind === :wall for e in b)
            @test [e.tag for e in b] == [:left, :right, :bottom, :top]
        end
    end

    # ======================================================================
    # (c) checkpoint geometry identity: the driver records it, a changed
    # geometry is refused, and an old (key-less) file loads as the box.
    # ======================================================================
    @testset "checkpoint geometry identity" begin
        # canonical formatting distinguishes every shape parameter
        baseA = (1.3, 0.4, -0.3, 0.9)
        baseb = (0.1, -0.05)
        base_id = geometry_identity(AffineBox(baseA, baseb, 1.0))
        ids = String[geometry_identity(CartesianBox(1.0)),
                     geometry_identity(CartesianBox(2.5)),
                     geometry_identity(AffineBox()),
                     base_id,
                     geometry_identity(AffineBox(baseA, baseb, 2.0)),
                     geometry_identity(CurvedBox(0.2, 1.0)),
                     geometry_identity(CurvedBox(0.3, 1.0))]
        @test length(unique(ids)) == length(ids)
        @test geometry_identity(CartesianBox(1.0)) == "CartesianBox(Lx=1.0)"
        @test geometry_identity(CurvedBox(0.2, 1.0)) == "CurvedBox(a=0.2,Lx=1.0)"
        # every affine parameter enters the record (dropping one would make two
        # different physical domains share a checkpoint key)
        for k in 1:4
            A2 = (k == 1 ? baseA[1] + 0.01 : baseA[1],
                  k == 2 ? baseA[2] + 0.01 : baseA[2],
                  k == 3 ? baseA[3] + 0.01 : baseA[3],
                  k == 4 ? baseA[4] + 0.01 : baseA[4])
            @test geometry_identity(AffineBox(A2, baseb, 1.0)) != base_id
        end
        for k in 1:2
            b2 = (k == 1 ? baseb[1] + 0.01 : baseb[1],
                  k == 2 ? baseb[2] + 0.01 : baseb[2])
            @test geometry_identity(AffineBox(baseA, b2, 1.0)) != base_id
        end
        @test geometry_identity(AffineBox(baseA, baseb, 2.0)) != base_id

        # the driver writes the canonical identity into the checkpoint config
        qt = Quadtree(1)
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        ss = make_semstokes(qt, 2)
        pr = Params(Ra=1.0e3, H=0.0, eta_law=:expT, deta_T=1.0,
                    Twall_top=0.0, Twall_bot=1.0, bc=:free)
        th0 = [0.01 * sinpi(x) * sinpi(y) for (x, y) in ss.su.coords]
        path = joinpath(tempdir(), "g266_geometry.h5")
        path_old = joinpath(tempdir(), "g266_geometry_old.h5")
        st = run_semquad_simulation(qt, pr, th0; p=2, tstop=1.0e-3,
                                    dt0=1.0e-4, cfl=0.5, maxsteps=2,
                                    verbose=false, checkpoint_path=path,
                                    checkpoint_every=0)
        @test st.steps == 2
        ld = load_semquad_state(path)
        idbox = geometry_identity(CartesianBox(st.qt))
        @test haskey(ld.config, "geometry")
        @test ld.config["geometry"] == idbox
        # the tree rebuild reproduces the same leaf list and coordinates
        @test Set(ld.qt.leaves) == Set(st.qt.leaves)
        @test ld.qt.Lx == st.qt.Lx
        @test isequal(make_semquad(ld.qt, 2).coords,
                      make_semquad(st.qt, 2).coords)
        gmap = CurvedBox(0.2, st.qt.Lx)
        cfg_bad = Dict{String,Any}("geometry" => geometry_identity(gmap))
        cfg_ok = Dict{String,Any}("geometry" => idbox)
        zeros_theta = zeros(Float64, length(st.theta))
        # a file written under another geometry is refused before any solve
        @test_throws ArgumentError QuaST._restart_guard(st.qt, 2, zeros_theta,
                                                        cfg_bad, path)
        guard = QuaST._restart_guard(st.qt, 2, zeros_theta, cfg_ok, path)
        @test guard.restart !== nothing
        @test Set(guard.qt.leaves) == Set(st.qt.leaves)
        # an old checkpoint (no geometry key) loads as the box, but cannot be
        # resumed by a mapped run: "absent" means CartesianBox
        save_semquad_state(path_old, st.qt, st.theta, st.theta, st.ux, st.uz,
                           st.pv; t=st.t, dt=1.0e-4, dt_prev=1.0e-4,
                           step=st.steps, p=2,
                           config=Dict{String,Any}(
                               "legacy_marker" => "1",
                               # a file from before cut 1 does carry the
                               # physics-revision record; only the geometry key
                               # is missing (that is the compatibility case)
                               "physics_revision" => string(PHYSICS_REVISION)))
        ld_old = load_semquad_state(path_old)
        @test !haskey(ld_old.config, "geometry")
        @test QuaST._restart_guard(st.qt, 2, zeros_theta, cfg_ok,
                                   path_old).restart !== nothing
        @test_throws ArgumentError QuaST._restart_guard(st.qt, 2, zeros_theta,
                                                        cfg_bad, path_old)
        rm(path; force=true)
        rm(path_old; force=true)
    end
end
