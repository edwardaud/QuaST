# G272: mapped transfer, AMR indicator and AMR cycle (stage 0, cut 6).
#
# The gate has five parts, mirroring the cut:
#
#   (a) transfer: the nodal transfer of a logical polynomial of degree <= p is
#       exact on `AffineBox` and `CurvedBox` (the new nodes are the map images
#       of the logical nodes, so evaluating the old interpolant at the node's
#       logical position reproduces the polynomial; the map only supplies the
#       coordinate).  The L2 transfer of the same polynomial is exact too (the
#       mapped mass and load are exact for it).  A deliberately wrong
#       "box-formula" evaluation point on `CurvedBox` is caught with its
#       measured error.  On the box map the mapped transfer is bit-identical to
#       the historical box body.
#
#   (b) indicator: the mapped body is the **physical** estimator
#           eta(leaf) = sum_faces (h_face/2) * int_face [d u / d n_phys]^2 ds
#                      + (h/2) * int_wall (du/dn_phys)^2 ds
#       with the true physical one-sided normal derivative, the physical face
#       measure from the face metric, and `h` the physical normal extent of the
#       leaf (`leaf_normal_size`); boundary terms are added on the geometry's
#       physical Dirichlet walls (`nu_wall_sides`).  It is compared, leaf for
#       leaf, with an independent oracle that rebuilds the map, Jacobian, face
#       normals and normal sizes analytically in this file (never through the
#       geometry API): agreement is the non-tautological check.  The estimator
#       is invariant under a rigid rotation of the whole configuration.  On the
#       box map it reduces to the historical box arithmetic up to the identified
#       historical index defect, measured below: the pre-cut box body computes
#       the fine-side one-sided derivative on a horizontal face in the *x* index
#       (`src/amr/indicator.jl`, `u[idsR[(q-1)*n + i]]` summed over `i`), i.e.
#       d/dx instead of d/dy, so a linear field, whose normal derivative is
#       continuous, gets a spurious interior indicator.  The box body was
#       corrected on 2026-11-25 (the one-line index fix) and now computes the
#       corrected estimator; this gate keeps the pre-fix kernel verbatim as
#       `g272_hist_subface_derivs`, so (b.3)'s "the historical body is
#       measurably different" assertions are genuine defect detectors instead
#       of self-comparisons (audit D10).  The nine-hash process dump is
#       invariant under the correction (its AMR cycle is inert); the
#       independent pin of the arithmetic is test/t_g278_indicator_oracle.jl.
#
#   (c) physical-vs-logical decision: on a `CurvedBox` whose physical cell size
#       in the layer direction varies by the annulus's `(R0+1)/R0 = 2.82`
#       factor, the physical and logical estimators are compared.  The logical
#       estimator is the corrected box body applied to the same logical dofs
#       (the "logical indicator" of GEOM_DESIGN §6.8: reference derivatives,
#       logical cell size and logical face measure).  The measured per-leaf
#       ratio eta_phys/eta_log on the wall-strip leaves equals the analytic
#       prediction derived from the map (formula below) to < 1e-6 relative, it
#       grows monotonically with the level towards the map's cell-size ratio,
#       and therefore the logical estimator under-weights the physically
#       compressed region by that ratio.  On the box map the two estimators
#       coincide (the map is a similarity), which is what "the box path is
#       already physical" means.
#
#   (d) end-to-end: a mapped `run_semquad_simulation` with `initial_refine > 0`
#       and `amr_every > 0` completes several steps, the leaf count evolves, the
#       history rows are written and finite, the state is finite, a checkpoint
#       written mid-run resumes bit-identically to the uninterrupted run (the
#       geometry record is part of the checkpoint config), the strain-rate
#       dependent law paths run mapped, and the still-refused mapped paths name
#       their missing piece.
#
#   (e) box path: the transfer function is the pre-cut body bit for bit, the
#       indicator is the corrected body (checked against the gate-local
#       corrected copy and against the pre-fix body, which must differ), the
#       new wall-strip/protect helpers reproduce the historical box predicates,
#       the box-default AMR cycle and wall refinement equal the explicit
#       `geom=CartesianBox` calls, and the mapped transfer/indicator on the box
#       map reproduce the box bodies' *corrected* arithmetic (the box map is a
#       similarity).  The nine-hash state dump A/B against a worktree
#       of `2898002` and the affected-gate batch are the process evidence in
#       `docs/dev/GEOM_PROGRESS.md`; the suite half of the box invariant is the
#       existing box AMR gates plus this testset.
#
# Every assertion names the wrong implementation it catches.

using Test
using LinearAlgebra
using Printf
using QuaST

const MS272 = QuaST

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

function g272_tree(level::Int; base::Int=level)
    qt = Quadtree(level + 2)
    for _ in 1:base, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

# uniform tree ready for a coarsening transfer: a level-(l+1) tree and its
# level-l parent tree (the same layout)
function g272_uniform_pair(l::Int)
    qtf = Quadtree(l + 2)
    for _ in 1:(l + 1), lf in collect(qtf.leaves)
        refine!(qtf, lf)
    end
    balance!(qtf)
    qtc = Quadtree(l + 2)
    for _ in 1:l, lf in collect(qtc.leaves)
        refine!(qtc, lf)
    end
    balance!(qtc)
    return (qtq_fine=qtf, qtc=qtc)
end

function g272_field(sq, p::Int, f)
    u = zeros(Float64, sq.N)
    n = p + 1
    for lf in sq.leaves, j in 1:n, i in 1:n
        S, T = MS272.global_logical(sq, lf, sq.xi[i], sq.xi[j])
        u[sq.elem[lf][(j - 1) * n + i]] = f(S, T)
    end
    return u
end

function g272_poly(S, T)
    # total degree 2: exactly reproduced by Q2 on every leaf (logical and, on an
    # affine map, physical polynomial too)
    return 1.0 + 2.0 * S - 3.0 * T + 0.5 * S * S + 0.7 * S * T
end

function g272_maxrel(a, b)
    return maximum(abs(a[i] - b[i]) / max(abs(a[i]), abs(b[i]), 1.0e-300)
                   for i in eachindex(a))
end

# ---------------------------------------------------------------------------
# Historical oracles (the pre-cut bodies, verbatim, the G266 pattern)
# ---------------------------------------------------------------------------

# nodal transfer, pre-cut body (physical target coordinates + physical search)
function g272_hist_nodal_transfer(old::SemQuad, uold::Vector{Float64},
                                  new::SemQuad)
    oldleaves = Set(old.leaves)
    unew = zeros(Float64, new.N)
    n = new.p + 1
    for lf in new.leaves
        hx = MS272.leaf_hx(new, lf[1]); hy = MS272.leaf_hy(new, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = new.elem[lf]
        for j in 1:n, i in 1:n
            x = x0 + 0.5 * (new.xi[i] + 1.0) * hx
            y = y0 + 0.5 * (new.xi[j] + 1.0) * hy
            unew[ids[(j - 1) * n + i]] =
                MS272._semquad_eval_at(old, uold, x, y, oldleaves)
        end
    end
    return unew
end

# the pre-2026-11-25 gradient-jump indicator, verbatim: the gate-local per-pair
# kernel below carries the defect (the fine-side y-face derivative contracts
# over the x index of row q).  It is what the box path computed until the fix,
# and it is used here as the *defect witness* of (b.3); the production body must
# NOT be called for it (that was the tautology audit D10).
function g272_hist_subface_derivs(sq::SemQuad, L, R, dir, u)
    p = sq.p
    n = p + 1
    hL = MS272.leaf_hy(sq, L[1]); hR = MS272.leaf_hy(sq, R[1])
    hxL = MS272.leaf_hx(sq, L[1]); hxR = MS272.leaf_hx(sq, R[1])
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
        gidR = (i, q) -> sq.elem[R][(q - 1) * n + i]     # WRONG (pre-fix)
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

function g272_hist_indicator(sq::SemQuad, qt::Quadtree, u::Vector{Float64})
    n = sq.p + 1
    eta = Dict{NTuple{3,Int},Float64}()
    for lf in sq.leaves
        eta[lf] = 0.0
    end
    for pf in face_pairs(qt)
        L, R, dir = pf.leafL, pf.leafR, pf.dir
        wq, dL, dR = g272_hist_subface_derivs(sq, L, R, dir, u)
        J = 0.0
        for q in eachindex(wq)
            J += wq[q] * (dL[q] - dR[q])^2
        end
        normal_x = dir === :xplus || dir === :xminus
        hL = normal_x ? MS272.leaf_hx(sq, L[1]) : MS272.leaf_hy(sq, L[1])
        hR = normal_x ? MS272.leaf_hx(sq, R[1]) : MS272.leaf_hy(sq, R[1])
        eta[L] += 0.5 * hL * J
        eta[R] += 0.5 * hR * J
    end
    for lf in sq.leaves
        h = MS272.leaf_hy(sq, lf[1])
        hx = MS272.leaf_hx(sq, lf[1])
        ids = sq.elem[lf]
        for dir in (:yminus, :yplus)
            MS272.has_no_face_neighbor(qt, lf, dir) || continue
            jF = dir === :yminus ? 1 : n
            for i in 1:n
                d = 0.0
                for j in 1:n
                    d += (2.0 / h) * sq.D[jF, j] * u[ids[(j - 1) * n + i]]
                end
                wf = 0.5 * hx * sq.w[i]
                eta[lf] += 0.5 * h * wf * d^2
            end
        end
    end
    return eta
end

# the same body with the fine-side y-face derivative taken in the correct
# (second) index -- used as the "logical indicator" of part (c) and to show that
# the mapped body is the corrected arithmetic on the box map
function g272_corrected_indicator(sq::SemQuad, qt::Quadtree,
                                  u::Vector{Float64})
    n = sq.p + 1
    eta = Dict{NTuple{3,Int},Float64}()
    for lf in sq.leaves
        eta[lf] = 0.0
    end
    rule = MS272.gll_rule_cached(sq.p)
    for pf in face_pairs(qt)
        L, R, dir = pf.leafL, pf.leafR, pf.dir
        hL = MS272.leaf_hy(sq, L[1]); hR = MS272.leaf_hy(sq, R[1])
        hxL = MS272.leaf_hx(sq, L[1]); hxR = MS272.leaf_hx(sq, R[1])
        xL = L[2] * hxL; yL = L[3] * hL
        xR = R[2] * hxR; yR = R[3] * hR
        idsL = sq.elem[L]; idsR = sq.elem[R]
        dL = zeros(n); dR = zeros(n); wq = zeros(n)
        vertical = dir === :xplus || dir === :xminus
        if vertical
            hw = 0.5 * hR
            iFL = dir === :xplus ? n : 1
            iFR = dir === :xplus ? 1 : n
            cL = (2.0 / hxL) .* sq.D[iFL, :]
            cR = (2.0 / hxR) .* sq.D[iFR, :]
            for q in 1:n
                wq[q] = hw * sq.w[q]
                pos = yR + 0.5 * (sq.xi[q] + 1.0) * hR
                xiL = 2.0 * (pos - yL) / hL - 1.0
                Lb, _ = MS272.lagrange_eval(rule, xiL)
                dq = 0.0
                for a in 1:n
                    da = 0.0
                    for i in 1:n
                        da += cL[i] * u[idsL[(a - 1) * n + i]]
                    end
                    dq += Lb[a] * da
                end
                dL[q] = dq
                dq = 0.0
                for i in 1:n
                    dq += cR[i] * u[idsR[(q - 1) * n + i]]
                end
                dR[q] = dq
            end
        else
            hw = 0.5 * hxR
            jFL = dir === :yplus ? n : 1
            jFR = dir === :yplus ? 1 : n
            cL = (2.0 / hL) .* sq.D[jFL, :]
            cR = (2.0 / hR) .* sq.D[jFR, :]
            for q in 1:n
                wq[q] = hw * sq.w[q]
                pos = xR + 0.5 * (sq.xi[q] + 1.0) * hxR
                xiL = 2.0 * (pos - xL) / hxL - 1.0
                Lb, _ = MS272.lagrange_eval(rule, xiL)
                dq = 0.0
                for a in 1:n
                    da = 0.0
                    for i in 1:n
                        da += cL[i] * u[idsL[(i - 1) * n + a]]
                    end
                    dq += Lb[a] * da
                end
                dL[q] = dq
                dq = 0.0
                for i in 1:n
                    dq += cR[i] * u[idsR[(i - 1) * n + q]]
                end
                dR[q] = dq
            end
        end
        J = 0.0
        for q in 1:n
            J += wq[q] * (dL[q] - dR[q])^2
        end
        normal_x = vertical
        hLs = normal_x ? MS272.leaf_hx(sq, L[1]) : MS272.leaf_hy(sq, L[1])
        hRs = normal_x ? MS272.leaf_hx(sq, R[1]) : MS272.leaf_hy(sq, R[1])
        eta[L] += 0.5 * hLs * J
        eta[R] += 0.5 * hRs * J
    end
    for lf in sq.leaves
        h = MS272.leaf_hy(sq, lf[1])
        hx = MS272.leaf_hx(sq, lf[1])
        ids = sq.elem[lf]
        for dir in (:yminus, :yplus)
            MS272.has_no_face_neighbor(qt, lf, dir) || continue
            jF = dir === :yminus ? 1 : n
            for i in 1:n
                d = 0.0
                for j in 1:n
                    d += (2.0 / h) * sq.D[jF, j] * u[ids[(j - 1) * n + i]]
                end
                wf = 0.5 * hx * sq.w[i]
                eta[lf] += 0.5 * h * wf * d^2
            end
        end
    end
    return eta
end

# historical wall-strip predicates of the balanced cycle / wall refinement
# (`zc = (lf[3]+0.5)*leaf_hy`, first/last row) and of the protect set
function g272_hist_in_strip(qt::Quadtree, lf::NTuple{3,Int}, strip::Float64)
    lf[3] == 0 && return true
    lf[3] == qt.ny0 * (1 << lf[1]) - 1 && return true
    zc = (lf[3] + 0.5) * MS272.leaf_hy(qt, lf[1])
    return zc <= strip || zc >= 1.0 - strip
end
function g272_hist_wall_leaf(qt::Quadtree, lf::NTuple{3,Int})
    return lf[3] == 0 || lf[3] == qt.ny0 * (1 << lf[1]) - 1
end

# ---------------------------------------------------------------------------
# Independent analytic oracle for the mapped indicator (parts b and c)
#
# The map, its Jacobian and everything derived from it are written here in
# closed form for the two validation geometries; nothing below calls
# `element_jacobian`, `leaf_metric`, `face_metric`, `leaf_normal_size` or
# `global_logical`.  A wrong cached metric, chain rule, normal, face measure or
# leaf size in the production body therefore cannot make the comparison pass.
# ---------------------------------------------------------------------------

g272_map(g::AffineBox, S, T) =
    (g.A[1] * g.Lx * S + g.A[2] * T + g.b[1],
     g.A[3] * g.Lx * S + g.A[4] * T + g.b[2])
g272_map(g::CurvedBox, S, T) =
    (g.Lx * S, T * (1.0 + g.a * sinpi(S)))

g272_jst(g::AffineBox, S, T) = (g.A[1] * g.Lx, g.A[2], g.A[3] * g.Lx, g.A[4])
function g272_jst(g::CurvedBox, S, T)
    d = 1.0 + g.a * sinpi(S)
    dp = g.a * pi * cospi(S)
    return (g.Lx, 0.0, T * dp, d)
end

# (x, y, J_leaf, det, J^{-T}) at a leaf reference point
function g272_lm(g, lay, lf::NTuple{3,Int}, xi::Float64, eta::Float64)
    l = lf[1]
    nsx = lay.nx0 * (1 << l)
    nsy = lay.ny0 * (1 << l)
    S = (lf[2] + 0.5 * (xi + 1.0)) / nsx
    T = (lf[3] + 0.5 * (eta + 1.0)) / nsy
    x, y = g272_map(g, S, T)
    Js = g272_jst(g, S, T)
    J1 = Js[1] / (2.0 * nsx); J2 = Js[2] / (2.0 * nsy)
    J3 = Js[3] / (2.0 * nsx); J4 = Js[4] / (2.0 * nsy)
    det = J1 * J4 - J2 * J3
    inv = (J4 / det, -J3 / det, -J2 / det, J1 / det)
    return (x=x, y=y, J=(J1, J2, J3, J4), det=det, inv=inv)
end

# physical gradient of the SEM field at a leaf reference point, from the
# analytic inverse Jacobian
function g272_grad(g, sq::SemQuad, u::Vector{Float64}, lf::NTuple{3,Int},
                   xi::Float64, eta::Float64)
    n = sq.p + 1
    rule = MS272.gll_rule_cached(sq.p)
    Lb, dLb = MS272.lagrange_eval(rule, xi)
    Lc, dLc = MS272.lagrange_eval(rule, eta)
    ids = sq.elem[lf]
    dxi = 0.0; deta = 0.0
    for j in 1:n, i in 1:n
        v = u[ids[(j - 1) * n + i]]
        dxi += v * dLb[i] * Lc[j]
        deta += v * Lb[i] * dLc[j]
    end
    inv = g272_lm(g, sq, lf, xi, eta).inv
    return (inv[1] * dxi + inv[2] * deta, inv[3] * dxi + inv[4] * deta)
end

# face tangent of the leaf at a reference point of `side`: returns (jac, nx, ny)
function g272_face(g, sq::SemQuad, lf::NTuple{3,Int}, side::Symbol,
                   t::Float64)
    is_x = side === :xminus || side === :xplus
    sgn = (side === :xplus || side === :yplus) ? 1.0 : -1.0
    xi, eta = is_x ? (sgn, t) : (t, sgn)
    lm = g272_lm(g, sq, lf, xi, eta)
    if is_x
        a = lm.J[2]; b = lm.J[4]
    else
        a = lm.J[1]; b = lm.J[3]
    end
    jac = hypot(a, b)
    if is_x
        nx = sgn * b / jac; ny = -sgn * a / jac
    else
        nx = -sgn * b / jac; ny = sgn * a / jac
    end
    return (jac=jac, nx=nx, ny=ny, x=lm.x, y=lm.y)
end

# physical normal extent of a leaf in the direction of `side`: 2/|grad s| with
# s the reference coordinate normal to the face, at the face midpoint
function g272_normal_size(g, sq::SemQuad, lf::NTuple{3,Int}, side::Symbol)
    is_x = side === :xminus || side === :xplus
    sgn = (side === :xplus || side === :yplus) ? 1.0 : -1.0
    xi, eta = is_x ? (sgn, 0.0) : (0.0, sgn)
    inv = g272_lm(g, sq, lf, xi, eta).inv
    nrm = is_x ? hypot(inv[1], inv[3]) : hypot(inv[2], inv[4])
    return 2.0 / nrm
end

g272_flip(side::Symbol) = side === :xplus ? :xminus :
                          side === :xminus ? :xplus :
                          side === :yplus ? :yminus : :yplus

function g272_oracle(g, sq::SemQuad, qt::Quadtree, u::Vector{Float64})
    n = sq.p + 1
    rule = MS272.gll_rule_cached(sq.p)
    eta = Dict{NTuple{3,Int},Float64}()
    for lf in sq.leaves
        eta[lf] = 0.0
    end
    # interior sub-faces: the coarser leaf is leafL, the fine side leafR; take
    # the fine side's physical normal and dot both one-sided analytic gradients
    # with it (orientation independent, squared jump)
    for pf in face_pairs(qt, g)
        L, R, dir = pf.leafL, pf.leafR, pf.dir
        sideR = g272_flip(dir)
        is_x = dir === :xplus || dir === :xminus
        J = 0.0
        for q in 1:n
            t = sq.xi[q]
            fmet = g272_face(g, sq, R, sideR, t)
            xiR, etaR = is_x ? ((sideR === :xplus ? 1.0 : -1.0), t) :
                              (t, (sideR === :yplus ? 1.0 : -1.0))
            gR = g272_grad(g, sq, u, R, xiR, etaR)
            dR = fmet.nx * gR[1] + fmet.ny * gR[2]
            # global logical position of the fine face point (closed form)
            nsidex_R = sq.nx0 * (1 << R[1])
            nsidey_R = sq.ny0 * (1 << R[1])
            S = (R[2] + 0.5 * (xiR + 1.0)) / nsidex_R
            T = (R[3] + 0.5 * (etaR + 1.0)) / nsidey_R
            # coarse leaf reference abscissa (index arithmetic only)
            nside_xL = sq.nx0 * (1 << L[1])
            nside_yL = sq.ny0 * (1 << L[1])
            xiL = 2.0 * (S * nside_xL - L[2]) - 1.0
            etaL = 2.0 * (T * nside_yL - L[3]) - 1.0
            gL = g272_grad(g, sq, u, L, xiL, etaL)
            dL = fmet.nx * gL[1] + fmet.ny * gL[2]
            J += fmet.jac * sq.w[q] * (dL - dR)^2
        end
        hL = g272_normal_size(g, sq, L, dir)
        hR = g272_normal_size(g, sq, R, sideR)
        eta[L] += 0.5 * hL * J
        eta[R] += 0.5 * hR * J
    end
    # boundary faces on the physical Dirichlet walls
    for lf in sq.leaves
        for dir in nu_wall_sides(g)
            MS272.has_no_face_neighbor(qt, lf, dir) || continue
            h = g272_normal_size(g, sq, lf, dir)
            acc = 0.0
            for q in 1:n
                fmet = g272_face(g, sq, lf, dir, sq.xi[q])
                is_x = dir === :xminus || dir === :xplus
                sgn = (dir === :xplus || dir === :yplus) ? 1.0 : -1.0
                xiq, etaq = is_x ? (sgn, sq.xi[q]) : (sq.xi[q], sgn)
                gr = g272_grad(g, sq, u, lf, xiq, etaq)
                d = fmet.nx * gr[1] + fmet.ny * gr[2]
                acc += fmet.jac * sq.w[q] * d^2
            end
            eta[lf] += 0.5 * h * acc
        end
    end
    return eta
end

# per-leaf prediction of eta_phys/eta_log for a field that depends on the
# physical height only (u = u(T); then both estimators have the same one-sided
# derivative structure and the ratio is a pure map factor):
#
#   y-face node at (S,T):  eta_phys/eta_log = h_p * jac_p * |grad T|^2 /
#                                            (h_T * jac_l)
#   h_p = h_T/|grad T|(face midpoint), jac_p/jac_l = sqrt(1 + (T d'/Lx)^2),
#   |grad T|^2 = (1 + (T d'/(d Lx))^2)/d^2,   d(S) = 1 + a sin(pi S).
#
# A uniform tree (all leaves the same level) makes the coarse/fine face weights
# identical; the per-leaf value is the face-contribution-weighted average.
function g272_ratio_prediction(g::CurvedBox, sq::SemQuad, qt::Quadtree,
                               u::Vector{Float64})
    n = sq.p + 1
    rule = MS272.gll_rule_cached(sq.p)
    num = Dict{NTuple{3,Int},Float64}()
    den = Dict{NTuple{3,Int},Float64}()
    for lf in sq.leaves
        num[lf] = 0.0; den[lf] = 0.0
    end
    function terms(S, T)
        d = 1.0 + g.a * sinpi(S)
        dp = g.a * pi * cospi(S)
        gT2 = (1.0 + (T * dp / (d * g.Lx))^2) / d^2
        jr = sqrt(1.0 + (T * dp / g.Lx)^2)
        return gT2, jr, d
    end
    function contributions(ll, jf, S_of_q, Tfix, jump)
        hT = 1.0 / (sq.ny0 * (1 << ll[1]))
        hS = sq.Lx / (sq.nx0 * (1 << ll[1]))
        Ss = [S_of_q(q) for q in 1:n]
        Smid = 0.5 * (minimum(Ss) + maximum(Ss))
        _, _, dmid = terms(Smid, Tfix)
        dpmid = g.a * pi * cospi(Smid)
        gTmid = sqrt(1.0 + (Tfix * dpmid / (dmid * g.Lx))^2) / dmid
        hp = hT / gTmid
        nacc = 0.0; dacc = 0.0
        for q in 1:n
            gT2, jr, _ = terms(Ss[q], Tfix)
            nacc += hp * (hS / 2.0) * jr * sq.w[q] * gT2
            dacc += hT * (hS / 2.0) * sq.w[q]
        end
        r = nacc / dacc
        wf = 0.5 * hT * (sq.Lx * hS / 2.0) * (sum(sq.w) * jump^2)
        num[ll] += wf * r
        den[ll] += wf
    end
    function logd(ll, jf, q)
        h = 1.0 / (sq.ny0 * (1 << ll[1]))
        v = 0.0
        for j in 1:n
            v += (2.0 / h) * sq.D[jf, j] * u[sq.elem[ll][(j - 1) * n + q]]
        end
        return v
    end
    for pf in face_pairs(qt)
        L, R, dir = pf.leafL, pf.leafR, pf.dir
        (dir === :xplus || dir === :xminus) && continue
        sideR = dir === :yplus ? :yminus : :yplus
        jfR = sideR === :yminus ? 1 : n
        jfL = dir === :yminus ? 1 : n
        Tfix = (R[3] + 0.5 * (sideR === :yminus ? 0.0 : 2.0)) /
               (sq.ny0 * (1 << R[1]))
        dR = logd(R, jfR, 1)
        hxL = sq.Lx / (sq.nx0 * (1 << L[1]))
        hxR = sq.Lx / (sq.nx0 * (1 << R[1]))
        xL0 = L[2] * (sq.Lx / (sq.nx0 * (1 << L[1])))
        xR0 = R[2] * (sq.Lx / (sq.nx0 * (1 << R[1])))
        xiL0 = 2.0 * (xR0 - xL0) / hxL - 1.0
        Lb, _ = MS272.lagrange_eval(rule, xiL0)
        dL = 0.0
        for a in 1:n
            da = 0.0
            for j in 1:n
                da += (2.0 / (1.0 / (sq.ny0 * (1 << L[1])))) * sq.D[jfL, j] *
                      u[sq.elem[L][(j - 1) * n + a]]
            end
            dL += Lb[a] * da
        end
        jump = dL - dR
        S_R(q) = (xR0 + 0.5 * (sq.xi[q] + 1.0) * hxR) / sq.Lx
        S_L(q) = (xL0 + 0.5 * (sq.xi[q] + 1.0) * hxL) / sq.Lx
        contributions(R, jfR, S_R, Tfix, jump)
        contributions(L, jfL, S_L, Tfix, jump)
    end
    for lf in sq.leaves
        for dir in (:yminus, :yplus)
            MS272.has_no_face_neighbor(qt, lf, dir) || continue
            jf = dir === :yminus ? 1 : n
            hx = sq.Lx / (sq.nx0 * (1 << lf[1]))
            x0 = lf[2] * hx
            Tfix = (lf[3] + 0.5 * (dir === :yminus ? 0.0 : 2.0)) /
                   (sq.ny0 * (1 << lf[1]))
            d = logd(lf, jf, 1)
            S_q(q) = (x0 + 0.5 * (sq.xi[q] + 1.0) * hx) / sq.Lx
            contributions(lf, jf, S_q, Tfix, d)
        end
    end
    out = Dict{NTuple{3,Int},Float64}()
    for lf in sq.leaves
        out[lf] = num[lf] / den[lf]
    end
    return out
end

# ===========================================================================
@testset "G272 mapped transfer, indicator and AMR" begin

# ===========================================================================
@testset "G272 (a) mapped transfer" begin
    # ---- (a.1) nodal transfer of a logical P2 is exact on both geoms -------
    # Catches a wrong old-field location (physical instead of logical), a wrong
    # new-node position (map not applied), or a wrong leaf/abscissa conversion.
    # A logical polynomial of degree <= p is reproduced by the old Q_p
    # interpolant on every leaf; the map image supplies only the node position,
    # so evaluating at the node's logical position is exact.
    gaff = AffineBox((1.30, 0.40, -0.30, 0.90), (0.30, -0.20), 1.0)
    gcur = CurvedBox(0.20, 1.0)
    qtp = g272_uniform_pair(2)
    for (tag, g) in (("AffineBox", gaff), ("CurvedBox", gcur))
        s_old = make_semquad(qtp.qtq_fine, 2; geom=g)
        s_new = make_semquad(qtp.qtc, 2; geom=g)
        uold = g272_field(s_old, 2, g272_poly)
        # coarsening: fine -> coarse; the coarse nodes are a subset of the fine
        # nodes for p = 2, so the comparison is against the exact polynomial
        un = transfer_semquad(s_old, uold, s_new; mode=:nodal)
        exact = g272_field(s_new, 2, g272_poly)
        err = maximum(abs, un - exact)
        @printf("G272 a.1 %-10s coarsening nodal P2: max|err| = %.3e\n",
                tag, err)
        @test err < 1.0e-13
        # refinement: coarse parent -> refined children (all old nodes are new
        # nodes, every other new node is a genuine interpolation point)
        un2 = transfer_semquad(s_new, exact, s_old; mode=:nodal)
        @test maximum(abs, un2 - uold) < 1.0e-13
        # L2 arm: exact for the same polynomial (mapped mass and load)
        ul = transfer_semquad(s_old, uold, s_new; mode=:l2, zero_walls=false)
        errl = maximum(abs, ul - exact)
        @printf("G272 a.1 %-10s coarsening L2   P2: max|err| = %.3e\n",
                tag, errl)
        @test errl < 1.0e-12
        # a constant field is preserved exactly (catches a measure/normalisation
        # error in the mapped L2 mass or load)
        uconst = fill(0.375, s_old.N)
        uc = transfer_semquad(s_old, uconst, s_new; mode=:l2)
        @test maximum(abs, uc .- 0.375) < 1.0e-14
    end

    # ---- (a.2) box map: the mapped nodal transfer is bit-identical to the
    #      historical box body (the probe measured exact 0.0; allow the last
    #      bit so the assertion is not tied to one libm)
    qtp2 = g272_uniform_pair(3)
    sbox_old = make_semquad(qtp2.qtq_fine, 2; geom=CartesianBox(1.0))
    sbox_new = make_semquad(qtp2.qtc, 2; geom=CartesianBox(1.0))
    ubox = g272_field(sbox_old, 2, (S, T) -> 0.3 * sinpi(S) * sinpi(T) +
                                         0.05 * sinpi(3S) * sinpi(2T))
    tbox = transfer_semquad(sbox_old, ubox, sbox_new; mode=:nodal)
    @test tbox == g272_hist_nodal_transfer(sbox_old, ubox, sbox_new)
    gmap = AffineBox((1.0, 0.0, 0.0, 1.0), (0.0, 0.0), 1.0)
    sm_old = make_semquad(qtp2.qtq_fine, 2; geom=gmap)
    sm_new = make_semquad(qtp2.qtc, 2; geom=gmap)
    tmap = transfer_semquad(sm_old, ubox, sm_new; mode=:nodal)
    @printf("G272 a.2 box map: mapped-vs-box nodal transfer rel = %.3e\n",
            g272_maxrel(tmap, tbox))
    @test g272_maxrel(tmap, tbox) < 1.0e-14

    # ---- (a.3) the wrong element_map evaluation point is caught ------------
    # On CurvedBox the box formula `x0 + (xi+1)hx/2` is NOT the map image of the
    # node; a transfer that used it evaluates the old field at the wrong
    # physical point.  The measured failure is the fraction of the field scale.
    sf = make_semquad(qtp2.qtq_fine, 2; geom=gcur)
    sc = make_semquad(qtp2.qtc, 2; geom=gcur)
    uf = g272_field(sf, 2, g272_poly)
    tex = g272_field(sc, 2, g272_poly)
    ucorrect = transfer_semquad(sf, uf, sc; mode=:nodal)
    ecorrect = maximum(abs, ucorrect - tex)
    uwrong = zeros(Float64, sc.N)
    oldleaves = Set(sf.leaves)
    for lf in sc.leaves
        hx = MS272.leaf_hx(sc, lf[1]); hy = MS272.leaf_hy(sc, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = sc.elem[lf]
        for j in 1:3, i in 1:3
            x = x0 + 0.5 * (sc.xi[i] + 1.0) * hx     # box formula, wrong map
            y = y0 + 0.5 * (sc.xi[j] + 1.0) * hy
            uwrong[ids[(j - 1) * 3 + i]] =
                MS272._semquad_eval_at(sf, uf, x, y, oldleaves)
        end
    end
    ewrong = maximum(abs, uwrong - tex)
    @printf("G272 a.3 CurvedBox: correct max|err| = %.3e, box-formula evaluation point max|err| = %.3e (field max %.3e)\n",
            ecorrect, ewrong, maximum(abs, tex))
    @test ecorrect < 1.0e-13
    @test ewrong > 1.0e-2

    # ---- (a.4) L2 transfer beats nodal transfer for a smooth coarsened field
    #      (the G80 statement, mapped; the gate's own mass matrix is used, so a
    #      wrong mapped measure shows up as a larger L2 error)
    H = (S, T) -> 0.2 * exp(-((S - 0.5) / 0.2)^2 -
                            ((T - 0.35) / 0.15)^2) * sinpi(T)
    qf = wall_graded_tree(2, 7; strip=0.15)
    qc = wall_graded_tree(2, 5; strip=0.15)
    for (tag, g) in (("AffineBox", gaff), ("CurvedBox", gcur))
        sff = make_semquad(qf, 4; geom=g)
        scc = make_semquad(qc, 4; geom=g)
        uff = g272_field(sff, 4, H)
        tn = transfer_semquad(sff, uff, scc; mode=:nodal, zero_walls=true)
        tl = transfer_semquad(sff, uff, scc; mode=:l2, zero_walls=true)
        te = g272_field(scc, 4, H)
        M = semquad_mass(scc)
        en = sqrt(abs(dot(tn - te, M * (tn - te))))
        el = sqrt(abs(dot(tl - te, M * (tl - te))))
        @printf("G272 a.4 %-10s smooth coarsening: nodal=%.3e L2=%.3e\n",
                tag, en, el)
        @test el <= en
        @test all(isfinite, tl)
    end

    # ---- (a.5) transferring between two different geometries is refused -----
    # Catches a silent box-into-mapped (or rotated-into-curved) mixing that
    # would move every node and measure.
    @test_throws ArgumentError transfer_semquad(sbox_old, ubox, sc)
end

# ===========================================================================
@testset "G272 (b) mapped indicator" begin
    gaff = AffineBox((1.30, 0.40, -0.30, 0.90), (0.30, -0.20), 1.0)
    gcur = CurvedBox(0.20, 1.0)
    qt = g272_tree(2)
    fphys = (x, y) -> 0.3 * sinpi(x) * cospi(2y) + 0.05 * sin(3x + 0.2)
    for (tag, g) in (("AffineBox", gaff), ("CurvedBox", gcur))
        sq = make_semquad(qt, 2; geom=g)
        u = [fphys(x, y) for (x, y) in sq.coords]   # physical field, not logical
        eta_impl = semquad_gradient_jump(sq, qt, u)
        eta_orac = g272_oracle(g, sq, qt, u)
        rel = maximum(abs(eta_impl[lf] - eta_orac[lf]) /
                      max(abs(eta_orac[lf]), 1.0e-300) for lf in qt.leaves)
        @printf("G272 b.1 %-10s impl-vs-analytic-oracle rel = %.3e\n",
                tag, rel)
        @test rel < 1.0e-12
        @test all(eta_impl[lf] >= 0.0 for lf in qt.leaves)
    end

    # ---- (b.2) rigid rotation invariance of the estimator ------------------
    # The physical quantity must not move when the geometry and the field are
    # rigidly rotated; a logical-gradient or reference-measure implementation
    # changes under the rotation.
    th = 0.7
    c = cos(th); s = sin(th)
    g0 = AffineBox((1.0, 0.0, 0.0, 1.0), (0.0, 0.0), 1.0)
    gr = AffineBox((c, -s, s, c), (0.0, 0.0), 1.0)
    s0 = make_semquad(qt, 2; geom=g0)
    sr = make_semquad(qt, 2; geom=gr)
    q = (x, y) -> x^2 + x * y + 0.3 * y^2       # physical P2: exactly contained
    u0 = [q(x, y) for (x, y) in s0.coords]
    ur = [q(c * x + s * y, -s * x + c * y) for (x, y) in sr.coords]
    e0 = semquad_gradient_jump(s0, qt, u0)
    er = semquad_gradient_jump(sr, qt, ur)
    scale = maximum(abs, values(e0))
    absdiff = maximum(abs(er[lf] - e0[lf]) for lf in qt.leaves)
    @printf("G272 b.2 rotation invariance: max|diff|/max = %.3e\n",
            absdiff / scale)
    @test absdiff <= 1.0e-13 * scale
    @test scale > 0.0                    # the invariance test is not vacuous

    # ---- (b.3) box map: the mapped body is the *corrected* box arithmetic --
    # On the box map the mapped body must reproduce the historical box
    # algorithm bit for bit up to the identified fine-side y-face index defect
    # (d/dx used instead of d/dy).  Against the corrected algorithm it agrees
    # to round-off.  Measured historical difference: the linear field below has
    # a continuous normal derivative, so the true interior indicator is zero;
    # the historical box body reports a spurious interior value and the mapped
    # body does not.
    gmap = AffineBox((1.0, 0.0, 0.0, 1.0), (0.0, 0.0), 1.0)
    qtc = g272_tree(2)
    sqm = make_semquad(qtc, 2; geom=gmap)
    u = [fphys(x, y) for (x, y) in sqm.coords]
    eta_impl = semquad_gradient_jump(sqm, qtc, u)
    eta_corr = g272_corrected_indicator(sqm, qtc, u)
    eta_hist = g272_hist_indicator(sqm, qtc, u)
    rel_corr = maximum(abs(eta_impl[lf] - eta_corr[lf]) /
                       max(abs(eta_corr[lf]), 1.0e-300) for lf in qtc.leaves)
    rel_hist = maximum(abs(eta_impl[lf] - eta_hist[lf]) /
                       max(abs(eta_hist[lf]), 1.0e-300) for lf in qtc.leaves)
    @printf("G272 b.3 box map: impl-vs-corrected rel = %.3e, impl-vs-historical rel = %.3e\n",
            rel_corr, rel_hist)
    @test rel_corr < 1.0e-13
    @test rel_hist > 1.0e-2          # the historical defect is measurable
    # a linear field: continuous normal derivative -> zero interior indicator
    ulin = [0.2 * x + 0.3 * y for (x, y) in sqm.coords]
    eta_lin_map = semquad_gradient_jump(sqm, qtc, ulin)
    eta_lin_hist = g272_hist_indicator(sqm, qtc, ulin)
    nleaf = length(qtc.leaves)
    # boundary term for u = a x + b y: only the y-walls have a normal
    # derivative, (b); per wall leaf (h/2)*int ds * b^2 over the wall face
    # (the gate computes it from the box expressions, not the indicator)
    exp_boundary = Dict{NTuple{3,Int},Float64}()
    for lf in qtc.leaves
        exp_boundary[lf] = 0.0
        h = MS272.leaf_hy(qtc, lf[1]); hx = MS272.leaf_hx(qtc, lf[1])
        for dir in (:yminus, :yplus)
            MS272.has_no_face_neighbor(qtc, lf, dir) || continue
            exp_boundary[lf] += 0.5 * h * (0.5 * hx * sum(sqm.w)) * 0.3^2
        end
    end
    ex_map = maximum(abs(eta_lin_map[lf] - exp_boundary[lf]) for lf in qtc.leaves)
    ex_hist = maximum(abs(eta_lin_hist[lf] - exp_boundary[lf])
                      for lf in qtc.leaves)
    @printf("G272 b.3 linear field: mapped-vs-exact boundary = %.3e, historical box body = %.3e (spurious interior term)\n",
            ex_map, ex_hist)
    @test ex_map < 1.0e-14
    @test ex_hist > 1.0e-2 * maximum(values(exp_boundary))
end

# ===========================================================================
@testset "G272 (c) physical vs logical indicator" begin
    # The annulus at R0 = 0.55 has a physical cell-size ratio
    # (R0+1)/R0 = 2.82 across the layer; CurvedBox(a) with a = 1/2.82 - 1 has
    # the same ratio in its layer-direction metric.  The field u = sin(pi T)
    # is a boundary-layer-shaped field: the estimator mass sits on the
    # y-normal faces, where the physical/logical ratio is the pure map factor
    #   F(S,T) = |grad T| * sqrt(1 + (T d'(S)/Lx)^2)
    #           = sqrt(1 + (T d'/(d Lx))^2)/d * sqrt(1 + (T d'/Lx)^2)
    # (h_p/h_T = 1/|grad T|, jac_p/jac_l = sqrt(1+(T d'/Lx)^2), and the
    # normal-derivative factor |grad T|).  At the T = 0 wall F = 1/d(S), so F
    # spans d_max/d_min = 1/(1+a) = (R0+1)/R0 across the layer.
    R0 = 0.55
    kratio = (R0 + 1.0) / R0
    a = 1.0 / kratio - 1.0
    g = CurvedBox(a, 1.0)
    qbox = CartesianBox(1.0)
    @printf("G272 c: R0 = %.2f, (R0+1)/R0 = %.6f, CurvedBox a = %.6f\n",
            R0, kratio, a)
    prev_ratio = 0.0
    meas_span = Float64[]
    pred_span = Float64[]
    for level in (4, 5, 6)
        qt = g272_tree(level; base=level)
        sq = make_semquad(qt, 2; geom=g)
        sqb = make_semquad(qt, 2; geom=qbox)
        # identical logical dofs are the precondition of the comparison
        @test sq.N == sqb.N && sq.elem == sqb.elem
        u = zeros(Float64, sq.N)
        for lf in sq.leaves, j in 1:3, i in 1:3
            S, T = MS272.global_logical(sq, lf, sq.xi[i], sq.xi[j])
            u[sq.elem[lf][(j - 1) * 3 + i]] = sinpi(T)
        end
        eta_p = semquad_gradient_jump(sq, qt, u)
        eta_l = g272_corrected_indicator(sqb, qt, u)  # the logical estimator
        pred = g272_ratio_prediction(g, sq, qt, u)
        # wall-strip leaves: the annulus inner-boundary-layer analogue
        ncol = 1 << level
        wall = [(level, ix, 0) for ix in 0:(ncol - 1)]
        worst = maximum(abs(eta_p[lf] / eta_l[lf] - pred[lf]) / pred[lf]
                        for lf in wall)
        # the two extreme columns: centre (physically compressed) and edge
        lf_c = (level, ncol ÷ 2, 0)
        lf_t = (level, 0, 0)
        rms = eta_p[lf_c] / eta_l[lf_c]
        rmt = eta_p[lf_t] / eta_l[lf_t]
        push!(meas_span, rms / rmt)
        push!(pred_span, pred[lf_c] / pred[lf_t])
        @printf("G272 c level %d: wall-leaf measured/predicted max rel = %.3e; compressed=%.6f tall=%.6f span=%.6f (predicted %.6f)\n",
                level, worst, rms, rmt, rms / rmt, pred[lf_c] / pred[lf_t])
        # the per-leaf ratio on the wall leaves is the analytic map factor
        @test worst < 1.0e-6
        # both estimators are positive and the ratio is > 1 where compressed
        @test rms > rmt > 1.0e-6
        # span grows monotonically with resolution towards (R0+1)/R0
        @test level == 4 || rms / rmt > prev_ratio
        prev_ratio = rms / rmt
        @test abs(rms / rmt - pred[lf_c] / pred[lf_t]) < 1.0e-6 * pred[lf_c] / pred[lf_t]
    end
    @printf("G272 c: measured compressed/tall span levels 4-6 = %s\n",
            join(round.(meas_span, digits=4), ", "))
    @printf("G272 c: predicted span = %s; (R0+1)/R0 = %.6f\n",
            join(round.(pred_span, digits=4), ", "), kratio)
    @test meas_span[end] > 0.95 * kratio        # converged to the design ratio
    @test meas_span[end] < kratio
    @test all(diff(meas_span) .> 0)

    # on the box map the two estimators coincide (a similarity map): the
    # physical-vs-logical difference is a property of a non-similar map only
    qt = g272_tree(2)
    sq = make_semquad(qt, 2; geom=AffineBox((1.0, 0.0, 0.0, 1.0),
                                            (0.0, 0.0), 1.0))
    u = [0.3 * sinpi(x) * sinpi(y) for (x, y) in sq.coords]
    ep = semquad_gradient_jump(sq, qt, u)
    el = g272_corrected_indicator(make_semquad(qt, 2; geom=CartesianBox(1.0)),
                                  qt, u)
    rel = maximum(abs(ep[lf] - el[lf]) / max(abs(el[lf]), 1.0e-300)
                  for lf in qt.leaves)
    @printf("G272 c box map: physical-vs-logical rel = %.3e\n", rel)
    @test rel < 1.0e-13
end

# ===========================================================================
@testset "G272 (d) end-to-end mapped AMR driver" begin
    gcur = CurvedBox(0.20, 1.0)
    gaff = AffineBox((1.30, 0.40, -0.30, 0.90), (0.30, -0.20), 1.0)
    dt = 1.0e-4
    pr = Params(Ra=1.0e3, H=0.0, bc=:rigid, eta_law=:expT, deta_T=1.0)
    function g272_driver_run(g; nsteps=4, ck="", ckevery=0, restart_from="",
                             pr_run=nothing, p=2, kw...)
        prr = pr_run === nothing ? pr : pr_run
        qt = g272_tree(2)
        ss = make_semstokes(qt, p; geom=g)
        th0 = zeros(Float64, ss.su.N)
        for lf in qt.leaves, j in 1:p+1, i in 1:p+1
            S, T = MS272.global_logical(ss.su, lf, ss.su.xi[i], ss.su.xi[j])
            th0[ss.su.elem[lf][(j - 1) * (p + 1) + i]] =
                0.05 * sinpi(S) * sinpi(T)
        end
        return run_semquad_simulation(qt, prr, th0; geom=g, p=p,
                                      tstop=nsteps * dt, dt0=dt, dtmax=dt,
                                      dtmin=1.0e-16, linear=:direct,
                                      maxsteps=nsteps, amr_every=2,
                                      initial_refine=2, symmetry=:none,
                                      advection=:skew, maxpic=1,
                                      checkpoint_path=ck,
                                      checkpoint_every=ckevery,
                                      restart_from=restart_from, verbose=false,
                                      kw...)
    end
    for (tag, g) in (("CurvedBox", gcur), ("AffineBox", gaff))
        res = g272_driver_run(g)
        leaves = length(res.qt.leaves)
        @printf("G272 d %-10s steps=%d leaves=%d hist=%d finite=%s theta_max=%.3e\n",
                tag, res.steps, leaves, length(res.history),
                all(isfinite, res.theta), maximum(abs, res.theta))
        @test res.steps == 4
        @test length(res.history) == 4
        @test length(res.history[end]) == 18
        @test all(isfinite, vcat(res.history...))
        @test all(isfinite, res.theta) && all(isfinite, res.ux) &&
              all(isfinite, res.uz)
        @test leaves != 16      # the AMR cycles evolved the mesh
        @test res.ss.su.geom isa typeof(g)
        @test geometry_identity(res.ss.su.geom) == geometry_identity(g)
        # the mapped indicator drives the cycle: the theta-jump indicator of
        # the returned state is finite and positive somewhere
        @test any(>(0.0), values(semquad_gradient_jump(res.ss.su, res.qt,
                                                       res.theta)))
    end

    # ---- (d.2) mapped checkpoint / resume is the uninterrupted trajectory --
    let dir = mktempdir()
        ck = joinpath(dir, "g272_mapped_ck.h5")
        a = g272_driver_run(gcur; nsteps=2, ck=ck, ckevery=2)
        b = g272_driver_run(gcur; nsteps=4, restart_from=ck)
        c = g272_driver_run(gcur; nsteps=4)
        @printf("G272 d.2 restart: a=%d b=%d c=%d theta_equal=%s hist_tail=%s leaves_equal=%s\n",
                a.steps, b.steps, c.steps, b.theta == c.theta,
                b.history == c.history[3:end],
                Set(b.qt.leaves) == Set(c.qt.leaves))
        @test a.steps == 2 && b.steps == 4 && c.steps == 4
        @test b.theta == c.theta
        @test b.history == c.history[3:end]
        @test Set(b.qt.leaves) == Set(c.qt.leaves)
        ld = load_semquad_state(ck)
        @test get(ld.config, "geometry", "") == geometry_identity(gcur)
        # the resumed run refuses a different geometry (the record is used)
        @test_throws ArgumentError g272_driver_run(gaff; nsteps=6,
                                                   restart_from=ck)
    end

    # ---- (d.3) strain-rate-dependent law paths run mapped ----------------
    # These consume the mapped strain-rate reconstructions (field / components
    # / spin); a box-body reconstructor would silently use 2/leaf_hy.
    pr_tosi = Params(Ra=1.0e3, H=0.0, bc=:rigid, eta_law=:tosi,
                     tosi=TosiLaw(z_is_depth=false, eta_star=1.0e-3,
                                  sigma_y=4.0, eta_min=1.0e-6, eta_max=10.0))
    res = g272_driver_run(gaff; nsteps=2, pr_run=pr_tosi)
    @printf("G272 d.3 tosi mapped: steps=%d finite=%s\n", res.steps,
            all(isfinite, res.theta))
    @test res.steps == 2
    @test all(isfinite, res.theta)
    pr_yield = Params(Ra=1.0e3, H=0.0, bc=:rigid, eta_law=:expT, deta_T=1.0,
                      yield_c=1.0, yield_phi=0.3, yield_taumax=1.0e6,
                      yield_eps_min=1.0e-8, eta_min=1.0e-4, eta_max=10.0)
    res = g272_driver_run(gaff; nsteps=2, pr_run=pr_yield)
    @printf("G272 d.3 yielding mapped: steps=%d finite=%s\n", res.steps,
            all(isfinite, res.theta))
    @test res.steps == 2
    @test all(isfinite, res.theta)
    # the velocity indicator also runs mapped (its jump body is the mapped one)
    res = g272_driver_run(gcur; nsteps=2, amr_indicator=:velocity)
    @printf("G272 d.3 velocity indicator mapped: steps=%d leaves=%d\n",
            res.steps, length(res.qt.leaves))
    @test res.steps == 2
    @test all(isfinite, res.theta)

    # ---- (d.4) the still-refused mapped paths name their missing piece ----
    qt = g272_tree(2)
    ss = make_semstokes(qt, 2; geom=gcur)
    th0 = zeros(Float64, ss.su.N)
    dt1 = 1.0e-4
    function g272_refusal(g; kw...)
        return run_semquad_simulation(qt, pr, th0; geom=g, p=2, tstop=dt1,
                                      dt0=dt1, dtmax=dt1, maxsteps=1,
                                      amr_every=0, initial_refine=0,
                                      maxpic=1, verbose=false, kw...)
    end
    for (tag, kw, needle) in (
            ("symmetry=:x", (symmetry=:x,), "GEOM_DESIGN"),
            ("hotspot", (hotspot_weight=1.0,), "Cartesian-y AMR policies"),
            ("newton", (pic_mode=:newton, linear=:direct), "mapped Newton"),
            # stage-0.5 fix campaign (audit F6): the stated reason is gate
            # coverage now, not a claimed box reference-matrix assumption
            ("schur_cg", (linear=:schur_cg,), "no mapped gate"))
        err = nothing
        try
            g272_refusal(gcur; kw...)
        catch e
            err = e
        end
        msg = err === nothing ? "" : sprint(showerror, err)
        @printf("G272 d.4 refusal %-12s: %s\n", tag,
                err === nothing ? "NO ERROR" :
                first(replace(msg, "\n" => " "), 110))
        @test err isa ArgumentError
        @test occursin(needle, msg)
    end
    # the Maxwell stress advance is mapped, but the elastic force load into the
    # Stokes rhs is still box-only; the driver refuses it with that reason
    pr_max = Params(Ra=1.0e3, H=0.0, bc=:rigid, eta_law=:expT, deta_T=1.0,
                    shear_modulus=1.0)
    err = try
        g272_driver_run(gcur; nsteps=2, pr_run=pr_max)
        nothing
    catch e
        e
    end
    msg = err === nothing ? "" : sprint(showerror, err)
    @printf("G272 d.4 refusal maxwell     : %s\n",
            err === nothing ? "NO ERROR" : first(replace(msg, "\n" => " "), 110))
    @test err isa ArgumentError
    @test occursin("Maxwell stress", msg)
    # the driver must accept a mapped `initial_refine`+`amr_every` card: the
    # run above does; this asserts the two old refusals are really gone
    err = try
        g272_refusal(gcur; initial_refine=0, amr_every=0)
        nothing
    catch e
        e
    end
    @test err === nothing
end

# ===========================================================================
@testset "G272 (e) box path and dispatch identity" begin
    # ---- (e.1) transfer is the pre-cut body; the indicator is the corrected
    #      body, pinned against the gate-local corrected copy, and the carried
    #      pre-fix body must be measurably different (non-tautology, D10) ----
    for (lvl, base) in ((2, 2), (3, 2), (3, 1))
        qt = g272_tree(lvl; base=base)
        sq = make_semquad(qt, 2)
        u = [0.3 * sinpi(x) * sinpi(y) + 0.05 * sinpi(3x) * sinpi(2y)
             for (x, y) in sq.coords]
        eta = semquad_gradient_jump(sq, qt, u)
        # Before 2026-11-25 this compared production with
        # `g272_hist_indicator`, whose interior kernel was the production
        # function itself: a self-comparison that never pinned the suspect line
        # (audit D10).  It now compares against the gate-local corrected body.
        eta_corr = g272_corrected_indicator(sq, qt, u)
        @test all(isequal(eta[lf], eta_corr[lf]) for lf in qt.leaves)
        eta_hist = g272_hist_indicator(sq, qt, u)
        @test any(!isequal(eta[lf], eta_hist[lf]) for lf in qt.leaves)
        qt2 = deepcopy(qt)
        for lf in sort(collect(qt2.leaves))[1:min(3, length(qt2.leaves))]
            refine!(qt2, lf)
        end
        balance!(qt2)
        sq2 = make_semquad(qt2, 2)
        @test transfer_semquad(sq, u, sq2; mode=:nodal) ==
              g272_hist_nodal_transfer(sq, u, sq2)
    end

    # ---- (e.2) the wall-strip and protect predicates are the historical ones
    for (lvl, base) in ((2, 2), (4, 2), (4, 1))
        qt = g272_tree(lvl; base=base)
        for strip in (0.05, 0.1, 0.15)
            for lf in qt.leaves
                @test MS272._amr_in_wall_strip(CartesianBox(1.0), qt, lf, strip) ==
                      g272_hist_in_strip(qt, lf, strip)
            end
        end
        for lf in qt.leaves
            @test MS272._amr_wall_leaf(CartesianBox(1.0), qt, lf) ==
                  g272_hist_wall_leaf(qt, lf)
        end
    end

    # ---- (e.3) the AMR cycle and the wall refinement REQUIRE the geometry
    #      (stage-0.5 audit D5): the old default `CartesianBox(qt)` silently
    #      refined the logical y strips (the periodic seam on a ring); omitting
    #      `geom` must now throw before touching the tree, and the explicit-box
    #      call is the historical box body.
    qt = g272_tree(2)
    sq = make_semquad(qt, 2)
    u = [0.3 * sinpi(x) * sinpi(y) for (x, y) in sq.coords]
    eta = semquad_gradient_jump(sq, qt, u)
    qa = deepcopy(qt); qb = deepcopy(qt)
    @test_throws ArgumentError amr_cycle_balanced!(qa, eta; refine_frac=0.2,
                                                   coarsen_frac=0.05,
                                                   wall_min_level=1,
                                                   wall_strip=0.15,
                                                   hysteresis_cycles=0)
    @test Set(qa.leaves) == Set(qt.leaves)   # refused before any mutation
    rb = amr_cycle_balanced!(qb, eta; refine_frac=0.2, coarsen_frac=0.05,
                             wall_min_level=1, wall_strip=0.15,
                             hysteresis_cycles=0, geom=CartesianBox(1.0))
    # the box body really refines here, i.e. the refusal above is not hiding a
    # no-op: the cycle added leaves to the explicit-geometry arm
    @test rb[1] > 0
    @test length(qb.leaves) > length(qt.leaves)
    qc = deepcopy(qt); qd = deepcopy(qt)
    @test_throws ArgumentError refine_wall_to_min!(qc, 3; strip=0.15)
    @test Set(qc.leaves) == Set(qt.leaves)
    rd = refine_wall_to_min!(qd, 3; strip=0.15, geom=CartesianBox(1.0))
    @test rd > 0
    @test length(qd.leaves) > length(qt.leaves)

    # ---- (e.4) wall_graded_tree defaults are the historical box tree ------
    for (base, wl, strip) in ((2, 5, 0.15), (3, 6, 0.2))
        t0 = wall_graded_tree(base, wl; strip=strip)
        t1 = wall_graded_tree(base, wl; strip=strip, geom=CartesianBox(1.0))
        t2 = wall_graded_tree(base, wl; strip=strip,
                              geom=AffineBox((1.0, 0.0, 0.0, 1.0),
                                             (0.0, 0.0), 1.0))
        @test sort(collect(t0.leaves)) == sort(collect(t1.leaves))
        # every stage-0 geometry has height = logical T, so the height-based
        # strip gives the same tree; the annulus (height = r - R0) is where the
        # two bodies will differ
        @test sort(collect(t0.leaves)) == sort(collect(t2.leaves))
    end
end

end
