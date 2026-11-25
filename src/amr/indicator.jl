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
                dq += w.cR[i] * u[idsR[(i - 1) * n + q]]
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

The theta and velocity jump indicators combined so that neither field can
dominate the other by its raw magnitude.

**Rule.**  Each field is divided by the statistic the Dörfler marking actually
consumes -- its total mass over the leaves -- and the two dimensionless shapes
are added:

```text
eta_both = total(jt) * (jt / total(jt) + jv / total(jv))
         = jt + (total(jt) / total(jv)) * jv
```

The second form is what the code evaluates.  It is the same ranking as the pure
`jt/total(jt) + jv/total(jv)` sum because `total(jt) > 0` is a common positive
factor, which changes no threshold and no selected leaf under the mass-based
cut (`refine_frac` is a fraction of the total; docs/dev/AMR.md §2.2).  Keeping
the factor makes the return value literally the theta indicator in the at-rest
limit instead of a rescaled copy of it.  Either field's absolute units then
cancel: multiplying `theta` by any positive constant scales the whole
combination by a positive constant (the cut is invariant), and multiplying the
velocity by one changes nothing at all because `jv` and `total(jv)` are both
quadratic in `u`.  That is the operational meaning of "neither dominates".

**Limits, both exact.**

- `total(jv) == 0` (the flow is at rest, or its velocity jump is identically
  zero): the velocity term carries no information, and the result is exactly
  `semquad_gradient_jump(sq, qt, theta)`.  This is what lets `:both` run at
  `t = 0` without collapsing the mesh the way a velocity-only indicator does.
- `total(jt) == 0` (a uniform `theta`, no temperature jump): there is no theta
  scale to weight against, so the result is the velocity indicator itself (the
  marking is scale-invariant, so this is the only informative term).

The historical plain sum `jt + jv` is **not** this: at Ra = 1e8 on the developed
2.5 x 1 card the velocity total is 3.19e14 against theta's 6.11, so the sum is
the velocity dictionary to printer precision and `:both` is an alias of
`:velocity` rather than a second indicator.  G234 pins the normalised rule:
exact at-rest equality with `:theta`, equal mass from the two fields, invariance
to rescaling either field, and a selected set that differs from both fields' when
both have signal.
"""
function semquad_combined_jump(sq::SemQuad, qt::Quadtree, theta::Vector{Float64},
                               ux::Vector{Float64}, uz::Vector{Float64})
    jt = semquad_gradient_jump(sq, qt, theta)
    jv = semquad_velocity_jump(sq, qt, ux, uz)
    total_t = 0.0
    for v in values(jt)
        total_t += max(v, 0.0)
    end
    total_v = 0.0
    for v in values(jv)
        total_v += max(v, 0.0)
    end
    # At rest the velocity indicator is identically zero: return the theta
    # dictionary itself (exact equality, no division by zero).
    total_v > 0.0 || return jt
    # No temperature signal: the velocity is the only informative field.  The
    # mass cut is scale-invariant, so returning `jv` changes no selection.
    total_t > 0.0 || return jv
    w = total_t / total_v
    for (lf, v) in jv
        jt[lf] = get(jt, lf, 0.0) + w * v
    end
    return jt
end

"""
    semquad_gradient_jump(sq, qt, u)

Per-leaf indicator `(h/2) * int_face [du.n]^2 ds` summed over interior faces,
plus `(h/2) * int_boundary (du.n)^2 ds` on the domain boundary.  The boundary
term is needed for thermal boundary layers; without it the estimator only sees
interior jumps.

The value is continuous, but the normal derivative is not; the jump between the
coarse and fine one-sided derivatives is the same indicator the DG prototype
used, with the coarse side interpolated at the fine-side quadrature points.

Stage 0 cut 6: a mapped geometry uses the **physical** variant (GEOM_DESIGN
§6.8); the box body below is the box specialisation of the same estimator:

  * the normal derivative is the true physical one-sided `n . grad u`, with
    `n` the outward unit normal of the fine side's face (from the cached
    `MappedFaceMetric`) and `grad u` the chain rule at the same physical point;
  * the face measure is the physical face Jacobian `ds/dq` (not the reference
    length) and the coarse side is evaluated at the fine node's **logical**
    position on the coarse face (`global_logical` + `leaf_ref`), never at a
    rounded physical position (GEOM_DESIGN §2.1 rule 4, §6.8);
  * the `(h/2)` scale is `leaf_normal_size`, the physical normal extent of the
    element ([`leaf_normal_size`](@ref)), which varies with the map — the whole
    point of the physical choice: a logical-gradient estimator would be wrong
    by the local physical cell-size ratio (`test/t_g272_mapped_amr.jl` (c));
  * the boundary term is added on the geometry's physical Dirichlet walls
    (`nu_wall_sides`, `:yminus`/`:yplus` for every stage-0 geometry; the
    annulus will register its circles there).

On the box map this reduces to the box estimator up to floating-point
association (the chain rule with the cached `J^{-1}` instead of the precomputed
`2/h` coefficients, and `2/|grad xi|` instead of `leaf_hx`): measured max
absolute difference 1.7e-18 and max relative difference 1.0e-16 on general
smooth fields (G272 (b) prints it).  The box path itself never takes this body.

**History (2026-11-25).**  Until this date the box body's fine-side y-face
one-sided derivative contracted the wrong nodal index -- `u[idsR[(q-1)*n+i]]`
(the x index of row q) instead of `u[idsR[(i-1)*n+q]]` (the y index of column
q).  The defect is recorded in `docs/dev/AMR.md` §2.1, `docs/dev/BUG_AUDIT_INDICATOR.md`
and `docs/dev/BUG_LEDGER.md` §19; the box body below now contains the corrected
index and reproduces the corrected estimator.  The nine-hash process dump
(`out_julia/agent/state_dump_run.jl`) does **not** pin this arithmetic -- its
AMR cycle is inert, so its hashes are invariant under the correction; the
independent oracle of `test/t_g278_indicator_oracle.jl` is the pin.
"""
function semquad_gradient_jump(sq::SemQuad, qt::Quadtree, u::Vector{Float64})
    return _semquad_gradient_jump(sq.geom, sq, qt, u)
end

# Box body: the box specialisation of the estimator above (precomputed 2/h
# scalings instead of the mapped chain rule), with the y-face fine-side index
# corrected on 2026-11-25.  G217/G266 carry gate-local copies of this arithmetic
# and G278 is the independent oracle.  The nine-hash process dump does not pin
# the indicator (its AMR cycle is inert; docs/dev/AMR.md §2.1).
function _semquad_gradient_jump(::CartesianBox, sq::SemQuad, qt::Quadtree,
                               u::Vector{Float64})
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

# ---------------------------------------------------------------------------
# Mapped body (cut 6): physical normal derivatives and physical face measures.
# ---------------------------------------------------------------------------

# One-sided physical normal derivative of `u` on leaf `lf` at the face point
# `(xi,eta)` with face normal `(nx,ny)`.
@inline function _mapped_normal_derivative(g::AbstractGeometry, sq::SemQuad,
                                          u::Vector{Float64},
                                          lf::NTuple{3,Int}, xi::Float64,
                                          eta::Float64, nx::Float64,
                                          ny::Float64)
    gx, gy = _semquad_grad_ref(g, sq, u, lf, xi, eta)
    return nx * gx + ny * gy
end

function _semquad_gradient_jump(g::AbstractGeometry, sq::SemQuad,
                                qt::Quadtree, u::Vector{Float64})
    n = sq.p + 1
    eta = Dict{NTuple{3,Int},Float64}()
    sizehint!(eta, length(sq.leaves))
    for lf in sq.leaves
        eta[lf] = 0.0
    end
    # Interior sub-faces: the coarse leaf is `leafL`, the fine side `leafR`; the
    # physical normal is taken from the fine side's cached face metric and both
    # one-sided gradients are dotted with the same vector, so the jump is
    # orientation-independent.
    for pf in face_pairs(qt, g)
        L, R, dir = pf.leafL, pf.leafR, pf.dir
        sideR = _flip_dir(dir)
        fm = _mapped_face_metric(g, sq, R, sideR, sq.xi)
        J = 0.0
        @inbounds for q in 1:n
            tq = sq.xi[q]
            xiR, etaR = _wall_ref_point(sideR, tq)
            dR = _mapped_normal_derivative(g, sq, u, R, xiR, etaR,
                                           fm.nx[q], fm.ny[q])
            # Coarse side: evaluate at the same logical face point, located on
            # the coarse leaf by logical arithmetic only.  Stage 0.5c1: on a
            # ring a cross-seam pair has the fine face one logical period away
            # from the coarse leaf, so the abscissa is wrapped into the coarse
            # leaf (the pair's two faces are the same physical curve).  A
            # bounded tree never takes the branch: its abscissa is already in
            # [-1, 1].
            S, T = global_logical(sq, R, xiR, etaR)
            xiL, etaL = leaf_ref(sq, L, S, T)
            if qt.seam === :periodic && (dir === :yplus || dir === :yminus)
                nL = leaf_nside_y(sq, L[1])
                etaL < -1.0 && (etaL += 2.0 * nL)
                etaL > 1.0 && (etaL -= 2.0 * nL)
            end
            dL = _mapped_normal_derivative(g, sq, u, L, xiL, etaL,
                                           fm.nx[q], fm.ny[q])
            J += fm.jac[q] * sq.w[q] * (dL - dR)^2
        end
        hL = leaf_normal_size(g, sq, L, dir)
        hR = leaf_normal_size(g, sq, R, sideR)
        eta[L] += 0.5 * hL * J
        eta[R] += 0.5 * hR * J
    end
    # Boundary faces on the physical Dirichlet walls only (the logical sides the
    # geometry's Nusselt pair names; the insulating sides add no physical jump,
    # exactly the box body's "skip the left/right walls").
    for lf in sq.leaves
        for dir in nu_wall_sides(g)
            has_no_face_neighbor(qt, lf, dir) || continue
            fm = _mapped_face_metric(g, sq, lf, dir, sq.xi)
            h = leaf_normal_size(g, sq, lf, dir)
            acc = 0.0
            @inbounds for q in 1:n
                xiq, etaq = _wall_ref_point(dir, sq.xi[q])
                d = _mapped_normal_derivative(g, sq, u, lf, xiq, etaq,
                                              fm.nx[q], fm.ny[q])
                acc += fm.jac[q] * sq.w[q] * d^2
            end
            eta[lf] += 0.5 * h * acc
        end
    end
    return eta
end
