# Gradient-jump error indicator of the continuous quadtree SEM field.
# ---------------------------------------------------------------------------
# Gradient-jump error indicator for the continuous quadtree SEM field.
# The value is continuous, but the normal derivative is not; the jump between
# the coarse and fine one-sided derivatives is the same indicator the DG
# prototype (removed with the classic policy) used, with the coarse side
# interpolated at the fine-side
# quadrature points.  Boundary faces add |du/dn|^2 so the thermal boundary
# layers are marked as well.
# ---------------------------------------------------------------------------

"""
    SemQuadJumpWork(n)

Per-call scratch of the gradient-jump indicator: one buffer per 1-D length `n`.
The face loop reuses it for every coarse/fine sub-face, so the indicator
allocates once per call instead of ~1.6 kB per face pair (the buffers were
`BL`, `wq`, `pos`, `xiL`, `cL`, `cR`, `dL`, `dR` and the two `lagrange_eval`
results; `--track-allocation=user` charged 67.4 MB to `_semquad_subface_derivs`
per 4 AMR cycles on the 10 000-leaf state).
"""
struct SemQuadJumpWork
    Lb::Vector{Float64}     # Lagrange values at the current fine point
    dLb::Vector{Float64}    # their derivatives (unused, `lagrange_eval!` writes them)
    dL::Vector{Float64}     # coarse-side one-sided derivative
    dR::Vector{Float64}     # fine-side one-sided derivative
    wq::Vector{Float64}     # fine-side face weights
    pos::Vector{Float64}    # physical face abscissae
    xiL::Vector{Float64}    # coarse-side reference abscissae
    cL::Vector{Float64}     # coarse-side normal derivative coefficients
    cR::Vector{Float64}     # fine-side normal derivative coefficients
    cB::Vector{Float64}     # boundary-face derivative coefficients
end

function SemQuadJumpWork(n::Int)
    SemQuadJumpWork(zeros(Float64, n), zeros(Float64, n), zeros(Float64, n),
                    zeros(Float64, n), zeros(Float64, n), zeros(Float64, n),
                    zeros(Float64, n), zeros(Float64, n), zeros(Float64, n),
                    zeros(Float64, n))
end

"""
    _semquad_subface_derivs!(w, sq, L, R, dir, u)

Workspace form of [`_semquad_subface_derivs`](@ref): fills `w.wq`, `w.dL` and
`w.dR` and returns them.  Same arithmetic and loop order as the allocating
form (the `BL` matrix is gone: row `q` is consumed as soon as it is evaluated,
exactly as the historical `BL[q, :] .= Lb` + `dL[q] += BL[q,a]*da` sequence).
"""
function _semquad_subface_derivs!(w::SemQuadJumpWork, sq::SemQuad,
                                  L::NTuple{3,Int}, R::NTuple{3,Int},
                                  dir::Symbol, u::Vector{Float64})
    p = sq.p
    n = p + 1
    hL = leaf_hy(sq, L[1]); hR = leaf_hy(sq, R[1])
    hxL = leaf_hx(sq, L[1]); hxR = leaf_hx(sq, R[1])
    xL = L[2] * hxL; yL = L[3] * hL
    xR = R[2] * hxR; yR = R[3] * hR
    rule = gll_rule_cached(p)
    idsL = sq.elem[L]
    idsR = sq.elem[R]
    vertical = dir === :xplus || dir === :xminus
    if vertical
        # x-face: normal derivative d/dx, parametrised by y
        hw = 0.5 * hR
        iFL = dir === :xplus ? n : 1
        iFR = dir === :xplus ? 1 : n
        for q in 1:n
            w.wq[q] = hw * sq.w[q]
            w.pos[q] = yR + 0.5 * (sq.xi[q] + 1.0) * hR
            w.xiL[q] = 2.0 * (w.pos[q] - yL) / hL - 1.0
        end
        hcL = 2.0 / hxL
        hcR = 2.0 / hxR
        for i in 1:n
            w.cL[i] = hcL * sq.D[iFL, i]
            w.cR[i] = hcR * sq.D[iFR, i]
        end
        for q in 1:n
            lagrange_eval!(w.Lb, w.dLb, rule, w.xiL[q])
            dq = 0.0
            for a in 1:n
                da = 0.0
                for i in 1:n
                    da += w.cL[i] * u[idsL[(a - 1) * n + i]]
                end
                dq += w.Lb[a] * da
            end
            w.dL[q] = dq
            dq = 0.0
            for i in 1:n
                dq += w.cR[i] * u[idsR[(q - 1) * n + i]]
            end
            w.dR[q] = dq
        end
    else
        # y-face: normal derivative d/dy, parametrised by x
        hw = 0.5 * hxR
        iFL = dir === :yplus ? n : 1
        iFR = dir === :yplus ? 1 : n
        for q in 1:n
            w.wq[q] = hw * sq.w[q]
            w.pos[q] = xR + 0.5 * (sq.xi[q] + 1.0) * hxR
            w.xiL[q] = 2.0 * (w.pos[q] - xL) / hxL - 1.0
        end
        hcL = 2.0 / hL
        hcR = 2.0 / hR
        for i in 1:n
            w.cL[i] = hcL * sq.D[iFL, i]
            w.cR[i] = hcR * sq.D[iFR, i]
        end
        for q in 1:n
            lagrange_eval!(w.Lb, w.dLb, rule, w.xiL[q])
            dq = 0.0
            for a in 1:n
                da = 0.0
                for i in 1:n
                    da += w.cL[i] * u[idsL[(i - 1) * n + a]]
                end
                dq += w.Lb[a] * da
            end
            w.dL[q] = dq
            dq = 0.0
            for i in 1:n
                dq += w.cR[i] * u[idsR[(q - 1) * n + i]]
            end
            w.dR[q] = dq
        end
    end
    return w.wq, w.dL, w.dR
end

"""One-sided normal-derivative traces of a coarse/fine sub-face at the fine-side
GLL points (SEM version of `dgquad._subface_derivs`)."""
function _semquad_subface_derivs(sq::SemQuad, L::NTuple{3,Int}, R::NTuple{3,Int},
                                 dir::Symbol, u::Vector{Float64})
    return _semquad_subface_derivs!(SemQuadJumpWork(sq.p + 1), sq, L, R, dir, u)
end

"""
    semquad_velocity_jump(sq, qt, ux, uz)

Per-leaf indicator built from the gradient jump of the *velocity*, the sum of the
two components' jumps.  The default `semquad_gradient_jump` on `theta` sees the
thermal boundary layers but not a velocity field that oscillates on the grid
scale while the temperature stays smooth, which is what happens at high Rayleigh
number (docs/dev/PHASE1.md).  This indicator is the same estimator applied to a
field that carries that error.

`sx` and `sz` are the velocity components on the scalar node layout of `sq`
(the Q2 space of the same tree), so no interpolation is needed.
"""
function semquad_velocity_jump(sq::SemQuad, qt::Quadtree,
                               ux::Vector{Float64}, uz::Vector{Float64})
    length(ux) == sq.N || throw(ArgumentError("ux has $(length(ux)) entries, " *
                                              "mesh has $(sq.N) dofs"))
    length(uz) == sq.N || throw(ArgumentError("uz has $(length(uz)) entries, " *
                                              "mesh has $(sq.N) dofs"))
    jx = semquad_gradient_jump(sq, qt, ux)
    jz = semquad_gradient_jump(sq, qt, uz)
    for (lf, v) in jx
        jx[lf] = v + get(jz, lf, 0.0)
    end
    return jx
end
"""
    semquad_combined_jump(sq, qt, theta, ux, uz)

The two jump estimators **added as they are**, `jt + jv`, with no rescaling.

`theta` alone is blind to a grid-scale velocity error, which is the high-Rayleigh
number case; the velocity alone vanishes at t = 0, where the flow is at rest, so
the cycle refines barely at all and the mesh stays coarse.  The plain sum has the
right limit at both ends: at rest it is exactly the theta indicator, and once the
flow develops the velocity term contributes wherever the velocity field is rough.

No normalization is involved.  An earlier version rescaled the velocity part to
theta's total, which makes a negligible velocity field contribute half of the
*shape* and reproduces the `:velocity` collapse; the historical
`coarsen_theta_tol` guard that motivated that rescaling no longer exists in the
balanced cycle (it was a parameter of the removed classic policy).
"""
function semquad_combined_jump(sq::SemQuad, qt::Quadtree, theta::Vector{Float64},
                               ux::Vector{Float64}, uz::Vector{Float64})
    jt = semquad_gradient_jump(sq, qt, theta)
    jv = semquad_velocity_jump(sq, qt, ux, uz)
    for (lf, v) in jv
        jt[lf] = get(jt, lf, 0.0) + v
    end
    return jt
end

"""
    semquad_gradient_jump(sq, qt, u)

Per-leaf indicator `(h/2) * int_face [du.n]^2 ds` summed over interior faces,
plus `(h/2) * int_boundary (du.n)^2 ds` on the domain boundary.  The boundary
term is needed for thermal boundary layers; without it the estimator only sees
interior jumps.
"""
function semquad_gradient_jump(sq::SemQuad, qt::Quadtree, u::Vector{Float64})
    n = sq.p + 1
    eta = Dict{NTuple{3,Int},Float64}()
    # `Dict` growth is O(#leaves) rehashes; every consumer either looks a leaf
    # up or sorts the entries, so the table capacity has no arithmetic effect.
    sizehint!(eta, length(sq.leaves))
    for lf in sq.leaves
        eta[lf] = 0.0
    end
    w = SemQuadJumpWork(n)
    for pf in face_pairs(qt)
        L, R, dir = pf.leafL, pf.leafR, pf.dir
        wq, dL, dR = _semquad_subface_derivs!(w, sq, L, R, dir, u)
        J = 0.0
        for q in eachindex(wq)
            J += wq[q] * (dL[q] - dR[q])^2
        end
        # the (h/2) scale uses the face-normal element size
        normal_x = dir === :xplus || dir === :xminus
        hL = normal_x ? leaf_hx(sq, L[1]) : leaf_hy(sq, L[1])
        hR = normal_x ? leaf_hx(sq, R[1]) : leaf_hy(sq, R[1])
        eta[L] += 0.5 * hL * J
        eta[R] += 0.5 * hR * J
    end
    # Boundary faces: the one-sided normal derivative is the only trace.
    for lf in sq.leaves
        h = leaf_hy(sq, lf[1])
        hx = leaf_hx(sq, lf[1])
        ids = sq.elem[lf]
        for dir in (:xminus, :xplus, :yminus, :yplus)
            has_no_face_neighbor(qt, lf, dir) || continue
            # Left/right walls are insulating (natural Neumann, dtheta/dx=0);
            # a one-sided x-derivative there is not a physical jump and would
            # keep refining the whole side column forever.  Only top/bottom
            # Dirichlet walls contribute a boundary-jump term.
            (dir === :xminus || dir === :xplus) && continue
            if dir === :xminus || dir === :xplus
                iF = dir === :xminus ? 1 : n
                c = (2.0 / hx) .* sq.D[iF, :]
                for j in 1:n
                    d = 0.0
                    for i in 1:n
                        d += c[i] * u[ids[(j - 1) * n + i]]
                    end
                    wf = 0.5 * h * sq.w[j]
                    eta[lf] += 0.5 * h * wf * d^2
                end
            else
                jF = dir === :yminus ? 1 : n
                hc = 2.0 / h
                for i in 1:n
                    w.cB[i] = hc * sq.D[jF, i]
                end
                for i in 1:n
                    d = 0.0
                    for j in 1:n
                        d += w.cB[j] * u[ids[(j - 1) * n + i]]
                    end
                    wf = 0.5 * hx * sq.w[i]
                    eta[lf] += 0.5 * h * wf * d^2
                end
            end
        end
    end
    return eta
end

