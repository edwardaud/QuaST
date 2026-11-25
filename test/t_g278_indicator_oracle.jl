# G278: the independent oracle for the box AMR gradient-jump indicator.
#
# The box indicator `semquad_gradient_jump` computes, per leaf K,
#
#   eta(K) = sum_{faces F of K} (h_n(K)/2) * int_F [ n . grad u ]^2 ds
#          + (h_n(K)/2) * int_{Dirichlet y-walls} (du/dn)^2 ds,
#
# with [.] the jump of the one-sided physical normal derivative of the Q_p
# interpolant, the fine side's GLL points as the quadrature on an interior pair,
# and h_n the physical extent of K in the face-normal direction.
#
# This gate reconstructs that definition **from scratch**: its own face
# enumeration from the physical leaf intervals (a 2:1 containment assertion, not
# `face_pairs`), its own tensor-product one-sided derivatives from the leaf's
# nodal values (not `_semquad_subface_derivs!`), its own GLL sub-face quadrature
# and physical measures, and its own `h_n*w/2` accumulation.  The production box
# body is then compared leaf by leaf against this reconstruction on
#
#   * fixed-seed random nodal fields and two smooth analytic fields, on three
#     trees (uniform unit L2, a hanging unit tree, the ASPECT-2b stretched
#     `cells_x = 5, cells_y = 2, Lx = 2.5` hanging tree) and p = 2 and 3,
#     with rtol 1e-13 / atol 1e-14,
#   * the exact linear / quadratic / kink fields, where the analytic estimator
#     is the exact value of the discrete estimator (the Q_p interpolant
#     reproduces those fields on every leaf): asserted **per leaf**, because the
#     kink case swaps a true jump for a spurious one a lattice line away and
#     the two errors cancel in the total (the old body's uniform-kink total
#     equals the exact total bit for bit while 8 of 16 leaves are wrong),
#   * the x-mirror symmetry of the estimator on a mirror-symmetric tree+field,
#   * non-negativity, and the identity-map dispatch against the mapped body.
#
# **The defect this catches.**  Until 2026-11-25 the box body's y-face fine-side
# one-sided derivative was
#     dR(q) = (2/hR) * sum_i D[jFR,i] * u[idsR[(q-1)*n + i]]
# i.e. it contracted the derivative row over the **x** node index of row q,
# instead of the **y** node index of column q:
#     dR(q) = (2/hR) * sum_i D[jFR,i] * u[idsR[(i-1)*n + q]].
# For a general field the wrong contraction is not even d/dx (measured distance
# to d/dx up to 3.86); it injects a spurious interior jump into every interior
# y-face, on uniform trees as well as hanging ones, at p = 2 and 3.  The
# "scope witnesses" testset deliberately re-implements the wrong contraction and
# measures its O(1) error against the reconstruction; on the parent commit the
# whole gate is red in the exact-field, random-field and symmetry groups.
#
# Non-vacuity (measured on this machine, 2026-11-25).  This file unchanged on
# the parent commit `eeb6726` reports **23 passed / 65 failed / 88 total**, with
# every failure in the defect groups: 18/18 random-and-smooth leaf checks, 34/40
# exact-field assertions (box-vs-exact total relative error 3.33e-1 (linear,
# uniform) to 112.7 (quadratic, stretched) and 0.79 on the kink), 6/18 scope
# witnesses (the y-face fine-side error is 2.9-3.3), 1/5 symmetry (the old body's
# mirror asymmetry is 0.66 of the maximum on this fixture; 8.28e-1 on the audit
# fixture), and 6/6 identity-map dispatch checks.  With the one-line fix the same
# file reports **88 passed / 0 failed / 88 total in 5.0 s** and every measured
# error drops to round-off (max per-leaf box-vs-oracle 3.5e-3 of the allowed
# tolerance, symmetry 6.0e-17, mapped-vs-box 1.8e-15 absolute).
using Test, LinearAlgebra, Random, Printf
using QuaST

const MS278 = QuaST

# ---------------------------------------------------------------------------
# trees (the audit's fixtures: uniform unit, hanging unit, ASPECT-2b stretched)
# ---------------------------------------------------------------------------
function g278_uniform(maxlevel::Int, level::Int; cx::Int=1, cy::Int=1,
                      Lx::Union{Nothing,Float64}=nothing)
    qt = Quadtree(maxlevel; Lx=Lx, cells_x=cx, cells_y=cy)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

function g278_hanging(maxlevel::Int, level::Int, nsel::Int; cx::Int=1,
                      cy::Int=1, Lx::Union{Nothing,Float64}=nothing)
    qt = g278_uniform(maxlevel, level; cx=cx, cy=cy, Lx=Lx)
    for lf in sort(collect(qt.leaves))[1:nsel]
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

# the three audit trees; `cells_x = 5, cells_y = 2, Lx = 2.5` is the ASPECT 2b
# geometry (square leaves, five base cells across).
g278_trees() = [
    ("uniform unit L2", g278_uniform(5, 2)),
    ("hanging unit L2+3", g278_hanging(6, 2, 3)),
    ("hanging 2b 2.5x1", g278_hanging(6, 2, 3; cx=5, cy=2)),
]

# ---------------------------------------------------------------------------
# exact box leaf geometry from the mesh parameters (not the indicator accessors
# `leaf_hx`/`leaf_hy`)
# ---------------------------------------------------------------------------
function g278_leaf_geom(qt::Quadtree, lf::NTuple{3,Int})
    l, ix, iy = lf
    hx = qt.Lx / (qt.nx0 * (1 << l))
    hy = 1.0 / (qt.ny0 * (1 << l))
    return (x0=ix * hx, y0=iy * hy, hx=hx, hy=hy)
end

g278_xn(q, p) = MS278.gll_rule_cached(p).nodes[q]
g278_wn(q, p) = MS278.gll_rule_cached(p).weights[q]

# one-sided physical normal derivative of the Q_p interpolant on leaf `lf` at a
# physical point on one of its faces.  Node layout u[ids[(j-1)*n + i]] with
# i = x node (fast), j = y node (slow); the normal coordinate is snapped to the
# face the point lies on.  Scaling 2/h is applied explicitly.
function g278_one_sided(qt::Quadtree, u::Vector{Float64}, sq::SemQuad,
                        lf::NTuple{3,Int}, axis::Symbol, x::Float64,
                        y::Float64)
    g = g278_leaf_geom(qt, lf)
    n = sq.p + 1
    ids = sq.elem[lf]
    rule = MS278.gll_rule_cached(sq.p)
    if axis === :x
        xi = abs(x - g.x0) <= abs(x - (g.x0 + g.hx)) ? -1.0 : 1.0
        eta = 2.0 * (y - g.y0) / g.hy - 1.0
        Lx, dLx = MS278.lagrange_eval(rule, xi)
        Ly, = MS278.lagrange_eval(rule, eta)
        s = 0.0
        for j in 1:n, i in 1:n
            s += dLx[i] * Ly[j] * u[ids[(j - 1) * n + i]]
        end
        return (2.0 / g.hx) * s
    else
        eta = abs(y - g.y0) <= abs(y - (g.y0 + g.hy)) ? -1.0 : 1.0
        xi = 2.0 * (x - g.x0) / g.hx - 1.0
        Lx, = MS278.lagrange_eval(rule, xi)
        Ly, dLy = MS278.lagrange_eval(rule, eta)
        s = 0.0
        for j in 1:n, i in 1:n
            s += Lx[i] * dLy[j] * u[ids[(j - 1) * n + i]]
        end
        return (2.0 / g.hy) * s
    end
end

# ---------------------------------------------------------------------------
# face enumeration from the definition: every leaf side, the neighbouring
# leaves' sub-faces; the fine side's GLL rule is the quadrature; both one-sided
# derivatives are evaluated at the same physical point.  A 2:1 containment
# assertion replaces `face_pairs`' bookkeeping.
# ---------------------------------------------------------------------------
struct G278Face
    kind::Symbol                  # :interior | :boundary
    L::NTuple{3,Int}              # coarse leaf (interior) / the leaf (boundary)
    R::NTuple{3,Int}              # fine leaf (interior)
    axis::Symbol
    xq::Vector{Float64}
    yq::Vector{Float64}
    wq::Vector{Float64}
    hL::Float64                   # normal extent of the L side
    hR::Float64                   # normal extent of the R side
end

function g278_enum_faces(qt::Quadtree, p::Int)
    n = p + 1
    leaves = sort(collect(qt.leaves))
    out = G278Face[]
    for lf in leaves
        g = g278_leaf_geom(qt, lf)
        for dir in (:xminus, :xplus, :yminus, :yplus)
            nbs = NTuple{3,Int}[]
            for nb in leaves
                nb === lf && continue
                gn = g278_leaf_geom(qt, nb)
                if dir === :xminus
                    ok = abs((gn.x0 + gn.hx) - g.x0) < 1e-12 &&
                         min(g.y0 + g.hy, gn.y0 + gn.hy) -
                         max(g.y0, gn.y0) > 1e-12
                elseif dir === :xplus
                    ok = abs(gn.x0 - (g.x0 + g.hx)) < 1e-12 &&
                         min(g.y0 + g.hy, gn.y0 + gn.hy) -
                         max(g.y0, gn.y0) > 1e-12
                elseif dir === :yminus
                    ok = abs((gn.y0 + gn.hy) - g.y0) < 1e-12 &&
                         min(g.x0 + g.hx, gn.x0 + gn.hx) -
                         max(g.x0, gn.x0) > 1e-12
                else
                    ok = abs(gn.y0 - (g.y0 + g.hy)) < 1e-12 &&
                         min(g.x0 + g.hx, gn.x0 + gn.hx) -
                         max(g.x0, gn.x0) > 1e-12
                end
                ok && push!(nbs, nb)
            end
            if isempty(nbs)
                # the left/right walls are insulating: no boundary term
                dir in (:xminus, :xplus) && continue
                xq = Vector{Float64}(undef, n)
                yq = Vector{Float64}(undef, n)
                wq = Vector{Float64}(undef, n)
                yface = dir === :yminus ? g.y0 : g.y0 + g.hy
                for q in 1:n
                    xq[q] = g.x0 + 0.5 * (g278_xn(q, p) + 1.0) * g.hx
                    yq[q] = yface
                    wq[q] = 0.5 * g.hx * g278_wn(q, p)
                end
                push!(out, G278Face(:boundary, lf, lf, :y, xq, yq, wq,
                                    g.hy, g.hy))
            else
                for nb in nbs
                    # one record per pair: from the coarse side, or from the
                    # lexicographically smaller leaf at equal level
                    if lf[1] > nb[1]
                        continue
                    elseif lf[1] == nb[1] && !(lf < nb)
                        continue
                    end
                    gf = g278_leaf_geom(qt, nb)
                    axis = (dir === :xminus || dir === :xplus) ? :x : :y
                    if axis === :x
                        @assert gf.y0 >= g.y0 - 1e-12 &&
                                gf.y0 + gf.hy <= g.y0 + g.hy + 1e-12
                        xface = (dir === :xplus) ? g.x0 + g.hx : g.x0
                        xq = Vector{Float64}(undef, n)
                        yq = Vector{Float64}(undef, n)
                        wq = Vector{Float64}(undef, n)
                        for q in 1:n
                            yq[q] = gf.y0 + 0.5 * (g278_xn(q, p) + 1.0) * gf.hy
                            xq[q] = xface
                            wq[q] = 0.5 * gf.hy * g278_wn(q, p)
                        end
                        push!(out, G278Face(:interior, lf, nb, :x, xq, yq, wq,
                                            g.hx, gf.hx))
                    else
                        @assert gf.x0 >= g.x0 - 1e-12 &&
                                gf.x0 + gf.hx <= g.x0 + g.hx + 1e-12
                        yface = (dir === :yplus) ? g.y0 + g.hy : g.y0
                        xq = Vector{Float64}(undef, n)
                        yq = Vector{Float64}(undef, n)
                        wq = Vector{Float64}(undef, n)
                        for q in 1:n
                            xq[q] = gf.x0 + 0.5 * (g278_xn(q, p) + 1.0) * gf.hx
                            yq[q] = yface
                            wq[q] = 0.5 * gf.hx * g278_wn(q, p)
                        end
                        push!(out, G278Face(:interior, lf, nb, :y, xq, yq, wq,
                                            g.hy, gf.hy))
                    end
                end
            end
        end
    end
    return out
end

# eta(K) = sum_F (h_n(K)/2) * sum_q w_q * (jump_q)^2 from the definition
function g278_eta(qt::Quadtree, faces::Vector{G278Face}, dnormal::Function)
    eta = Dict{NTuple{3,Int},Float64}(lf => 0.0 for lf in qt.leaves)
    for f in faces
        if f.kind === :boundary
            J = 0.0
            for q in eachindex(f.wq)
                d = dnormal(f.L, f.axis, f.xq[q], f.yq[q])
                J += f.wq[q] * d * d
            end
            eta[f.L] += 0.5 * f.hL * J
        else
            J = 0.0
            for q in eachindex(f.wq)
                dL = dnormal(f.L, f.axis, f.xq[q], f.yq[q])
                dR = dnormal(f.R, f.axis, f.xq[q], f.yq[q])
                J += f.wq[q] * (dL - dR)^2
            end
            eta[f.L] += 0.5 * f.hL * J
            eta[f.R] += 0.5 * f.hR * J
        end
    end
    return eta
end

# exact analytic estimator for the polynomial / kink fields
function g278_analytic(kind::Symbol, qt::Quadtree; a=0.2, b=0.3, A=0.7,
                       B=0.3, slope=1.0, yk=0.5)
    if kind === :linear
        f = (x, y) -> a * x + b * y
        dn = (lf, ax, x, y) -> ax === :x ? a : b
    elseif kind === :quadratic
        f = (x, y) -> A * x^2 + B * y^2
        dn = (lf, ax, x, y) -> 2.0 * (ax === :x ? A * x : B * y)
    elseif kind === :kink
        f = (x, y) -> y <= yk ? 0.0 : slope * (y - yk)
        dn = (lf, ax, x, y) -> begin
            g = g278_leaf_geom(qt, lf)
            ax === :x ? 0.0 : (g.y0 >= yk - 1e-12 ? slope : 0.0)
        end
    else
        error("unknown analytic kind $kind")
    end
    return f, dn
end

g278_totals(d::Dict) = sum(values(d); init=0.0)

# leaf-by-leaf agreement: |a-b| <= atol + rtol*|b| for every leaf
function g278_leafwise_max(a::Dict, b::Dict, leaves; rtol::Float64,
                           atol::Float64)
    return maximum(abs(a[lf] - b[lf]) / (atol + rtol * abs(b[lf]))
                   for lf in leaves)
end

@testset "G278 box indicator independent oracle" begin
    # ==================================================================
    # (1) fixed-seed random and smooth fields: leaf-by-leaf against the
    #     from-scratch reconstruction, rtol 1e-13 / atol 1e-14.
    # ==================================================================
    @testset "random and smooth fields vs from-scratch oracle" begin
        rng = MersenneTwister(0x51e11)
        smooth1(x, y) = sin(2.0 * x + 0.3) * cos(3.0 * y - 0.1) + 0.2 * x * y
        smooth2(x, y) = exp(-x) * sin(3.0 * y + 0.5) + 0.1 * x^2 * y
        checked = 0
        for (tag, qt) in g278_trees(), p in (2, 3)
            sq = make_semquad(qt, p)
            faces = g278_enum_faces(qt, p)
            ufields = [
                ("random", randn(rng, sq.N)),
                ("smooth1", [smooth1(c[1], c[2]) for c in sq.coords]),
                ("smooth2", [smooth2(c[1], c[2]) for c in sq.coords]),
            ]
            for (ftag, u) in ufields
                box = semquad_gradient_jump(sq, qt, u)
                orc = g278_eta(qt, faces,
                               (lf, ax, x, y) ->
                                   g278_one_sided(qt, u, sq, lf, ax, x, y))
                worst = g278_leafwise_max(box, orc, qt.leaves;
                                          rtol=1.0e-13, atol=1.0e-14)
                @printf("G278 oracle %-19s p=%d %-8s worst/tol = %.3e, total rel = %.3e\n",
                        tag, p, ftag, worst,
                        abs(g278_totals(box) - g278_totals(orc)) /
                        max(abs(g278_totals(orc)), 1.0e-300))
                @test worst <= 1.0
                checked += 1
            end
        end
        @test checked == 18
    end

    # ==================================================================
    # (2) exact linear / quadratic / kink fields, compared per leaf.
    #     The kink total can equal the exact total while individual
    #     leaves are wrong (the defect moves the jump one lattice line
    #     away and the total cancels), so this must be leaf by leaf.
    # ==================================================================
    @testset "exact fields per leaf" begin
        checked = 0
        for (tag, qt) in g278_trees()
            # the kink fixture needs y = 0.5 to be a leaf boundary
            @test all(lf -> !(g278_leaf_geom(qt, lf).y0 < 0.5 - 1.0e-12 &&
                              g278_leaf_geom(qt, lf).y0 +
                              g278_leaf_geom(qt, lf).hy > 0.5 + 1.0e-12),
                      qt.leaves)
            for p in (2, 3)
                sq = make_semquad(qt, p)
                faces = g278_enum_faces(qt, p)
                for kind in (:linear, :quadratic, :kink)
                    f, dn = g278_analytic(kind, qt)
                    u = [f(c[1], c[2]) for c in sq.coords]
                    box = semquad_gradient_jump(sq, qt, u)
                    ex = g278_eta(qt, faces, dn)
                    scale = maximum(abs, values(ex))
                    worst = maximum(abs(box[lf] - ex[lf]) /
                                    (1.0e-12 * scale +
                                     1.0e-13 * abs(ex[lf]))
                                    for lf in qt.leaves)
                    rel = abs(g278_totals(box) - g278_totals(ex)) /
                          max(abs(g278_totals(ex)), 1.0e-300)
                    @printf("G278 exact %-8s %-19s p=%d total rel = %.3e, worst/tol = %.3e\n",
                            kind, tag, p, rel, worst)
                    @test worst <= 1.0
                    @test rel <= 1.0e-13
                    checked += 1
                end
            end
        end
        @test checked == 18
    end

    # ==================================================================
    # (3) scope witnesses: the production y-face fine side is the
    #     one-sided d/dy of the definition (≤1e-13); the x-face fine side
    #     is unchanged (≤1e-13); and the deliberately reintroduced
    #     wrong-index contraction (the pre-fix line) is an O(1) error.
    # ==================================================================
    @testset "scope witnesses" begin
        for (tag, qt) in g278_trees(), p in (2, 3)
            sq = make_semquad(qt, p)
            n = p + 1
            f(x, y) = sin(2.0 * x + 0.3) * cos(3.0 * y - 0.1) + 0.2 * x * y
            u = [f(c[1], c[2]) for c in sq.coords]
            w = MS278.SemQuadJumpWork(n)
            errx = 0.0
            erry = 0.0
            err_bad = 0.0
            bad_max = 0.0
            for pf in face_pairs(qt)
                L, R, dir = pf.leafL, pf.leafR, pf.dir
                wq, dL, dR = MS278._semquad_subface_derivs!(w, sq, L, R, dir,
                                                            u)
                gR = g278_leaf_geom(qt, R)
                if dir === :xminus || dir === :xplus
                    # the fine side's touching face is the opposite side: for
                    # dir = :xplus the fine leaf R lies to the right of L, so R
                    # touches L on its :xminus face (production `iFR` = 1)
                    xface = dir === :xplus ? gR.x0 : gR.x0 + gR.hx
                    for q in 1:n
                        yq = gR.y0 + 0.5 * (sq.xi[q] + 1.0) * gR.hy
                        errx = max(errx, abs(dR[q] -
                            g278_one_sided(qt, u, sq, R, :x, xface, yq)))
                    end
                else
                    yface = dir === :yplus ? gR.y0 : gR.y0 + gR.hy
                    jFR = dir === :yplus ? 1 : n
                    for q in 1:n
                        xq = gR.x0 + 0.5 * (sq.xi[q] + 1.0) * gR.hx
                        dref = g278_one_sided(qt, u, sq, R, :y, xq, yface)
                        erry = max(erry, abs(dR[q] - dref))
                        # the pre-fix contraction, reintroduced on purpose:
                        # contract over the x index i of row q
                        dq = 0.0
                        for i in 1:n
                            dq += (2.0 / gR.hy) * sq.D[jFR, i] *
                                  u[sq.elem[R][(q - 1) * n + i]]
                        end
                        err_bad = max(err_bad, abs(dq - dref))
                        bad_max = max(bad_max, abs(dq))
                    end
                end
            end
            @printf("G278 scope %-19s p=%d y-face fine err = %.3e, x-face fine err = %.3e, wrong-index err = %.3e (|buggy|~%.2f)\n",
                    tag, p, erry, errx, err_bad, bad_max)
            @test erry <= 1.0e-13      # the corrected line is the definition
            @test errx <= 1.0e-13      # the x-face branch was never wrong
            @test err_bad > 1.0e-2     # O(1): the defect is measurable
        end
    end

    # ==================================================================
    # (4) x-mirror symmetry on a mirror-symmetric tree and field, and
    #     non-negativity.
    # ==================================================================
    @testset "mirror symmetry and non-negativity" begin
        qt = g278_uniform(4, 2)
        @test length(qt.leaves) == 16
        sq = make_semquad(qt, 2)
        u = [sinpi(x) * sinpi(y) + 0.1 * sinpi(3.0 * x) * sinpi(y)
             for (x, y) in sq.coords]
        box = semquad_gradient_jump(sq, qt, u)
        mirror(lf) = (lf[1], (1 << lf[1]) - 1 - lf[2], lf[3])
        scale = maximum(abs, values(box))
        asym = maximum(abs(box[lf] - box[mirror(lf)]) for lf in qt.leaves)
        @printf("G278 symmetry: max|eta(L)-eta(mirror L)|/max = %.3e (old body 8.28e-1)\n",
                asym / scale)
        @test asym <= 1.0e-13 * scale
        @test all(v -> v >= 0.0, values(box))
        orc = g278_eta(qt, g278_enum_faces(qt, 2),
                       (lf, ax, x, y) ->
                           g278_one_sided(qt, u, sq, lf, ax, x, y))
        asym_o = maximum(abs(orc[lf] - orc[mirror(lf)]) for lf in qt.leaves)
        @test asym_o <= 1.0e-13 * maximum(abs, values(orc))
        @test all(v -> v >= 0.0, values(orc))
    end

    # ==================================================================
    # (5) dispatch identity: the mapped body on the identity
    #     AffineBox is the box body up to round-off (the box body is the
    #     box specialisation of the same estimator).
    # ==================================================================
    @testset "identity-map dispatch" begin
        rng = MersenneTwister(0x1de17)
        for (tag, qt) in g278_trees(), p in (2, 3)
            sq = make_semquad(qt, p)
            sqm = make_semquad(qt, p;
                               geom=AffineBox((1.0, 0.0, 0.0, 1.0),
                                              (0.0, 0.0), qt.Lx))
            f(x, y) = sin(2.0 * x + 0.3) * cos(3.0 * y - 0.1) + 0.2 * x * y
            u = [f(c[1], c[2]) for c in sq.coords] .+ 0.1 .* randn(rng, sq.N)
            box = semquad_gradient_jump(sq, qt, u)
            mp = semquad_gradient_jump(sqm, qt, u)
            worst = g278_leafwise_max(mp, box, qt.leaves; rtol=1.0e-13,
                                      atol=1.0e-14)
            @printf("G278 dispatch %-19s p=%d mapped-vs-box maxabs = %.3e, worst/tol = %.3e\n",
                    tag, p,
                    maximum(abs(mp[lf] - box[lf]) for lf in qt.leaves), worst)
            @test worst <= 1.0
        end
    end
end
