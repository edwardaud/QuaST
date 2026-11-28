# Quadtree SEM diagnostics: wall Nusselt numbers, Vrms, mean temperature and the
# energy Nusselt number.  These are the acceptance quantities for the adaptive
# extreme-case runs, so they must match the uniform-mesh definitions in
# physics.jl on exactly representable fields.

"""
    semquad_nu_walls(sq, qt, theta, pr)

Wall Nusselt numbers from the one-sided GLL derivative at the top/bottom
leaves, integrated over x:

    Nu_top = 1 + H/2 - <dtheta/dy>_top,
    Nu_bot = 1 - H/2 - <dtheta/dy>_bot.
"""
function semquad_nu_walls(sq::SemQuad, qt::Quadtree, theta::Vector{Float64},
                          pr::Params)
    g = sq.geom
    g isa CartesianBox || return _semquad_nu_walls_mapped(g, sq, qt, theta, pr)
    p = sq.p
    n = p + 1
    st = 0.0; sb = 0.0; wt = 0.0; wb = 0.0
    for lf in sq.leaves
        h = leaf_hy(sq, lf[1])
        hx = leaf_hx(sq, lf[1])
        y0 = lf[3] * h
        ids = sq.elem[lf]
        if y0 + h >= 1.0 - 1.0e-14
            for i in 1:n
                d = 0.0
                for j in 1:n
                    d += sq.D[n, j] * theta[ids[(j - 1) * n + i]]
                end
                d *= 2.0 / h
                w = 0.5 * hx * sq.w[i]
                st += w * d
                wt += w
            end
        end
        if y0 <= 1.0e-14
            for i in 1:n
                d = 0.0
                for j in 1:n
                    d += sq.D[1, j] * theta[ids[(j - 1) * n + i]]
                end
                d *= 2.0 / h
                w = 0.5 * hx * sq.w[i]
                sb += w * d
                wb += w
            end
        end
    end
    # Nu = -(dT/dy)_wall / ((T_bot - T_top)/1) with T = base + theta, so the
    # conductive reference flux is the base-state gradient at the wall (for the
    # Boussinesq base this is 1 +- H/2, so the old expressions are recovered).
    top_ref = -semquad_base_gradient(pr, 1.0)
    bot_ref = -semquad_base_gradient(pr, 0.0)
    return top_ref - st / wt, bot_ref - sb / wb
end

"""
    semquad_nu_energy(sq, qt, pr, theta, ux, uz) -> Float64

Energy Nusselt number `1 + H/2 + <u·grad h θ>` — the volume-mean convective
heat-flux functional in Nusselt units, with `u_z` on the box and
`upward_velocity(g,x,y,ux,uz) = u.grad h` on a mapped geometry, normalised by
`total_area(qt)` (box, bit for bit) resp. `total_measure(g, sq)` (mapped).

It is **not** the wall Nusselt pair of the benchmark: it is an auxiliary
volume-mean quantity, it is not a history column in any run (box or mapped),
and its annulus value is not the published benchmark Nu.  The mapped body
existed since cut 4 but was ungated until the stage-0.5 fix campaign's G279
(the audit's B2 row found the body correct and only the `GEOM_PROGRESS`
sentence "stays box-normalised" stale); G279 pins the box/identity-map bitwise
equality, the annulus conduction value `1 + H/2`, and a manufactured annulus
state against the analytic `1 + H/2 + <θ>`.
"""
function semquad_nu_energy(sq::SemQuad, qt::Quadtree, pr::Params,
                           theta::Vector{Float64}, ux::Vector{Float64},
                           uz::Vector{Float64})
    g = sq.geom
    g isa CartesianBox ||
        return _semquad_nu_energy_mapped(g, sq, qt, pr, theta, ux, uz)
    p = sq.p
    n = p + 1
    fd = make_fedata(p, p - 1)
    s = 0.0
    for lf in sq.leaves
        hx = leaf_hx(sq, lf[1]); hy = leaf_hy(sq, lf[1])
        ids = sq.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            uzq = 0.0; thq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                uzq += uz[ids[k]] * b
                thq += theta[ids[k]] * b
            end
            s += w * uzq * thq
        end
    end
    return 1.0 + 0.5 * pr.H + s / total_area(qt)
end

# ===========================================================================
# Mapped bodies (stage 0 cut 4)
# ===========================================================================
#
# Every public diagnostics function above dispatches once on `sq.geom`: the
# `CartesianBox` body above is the historical statement, character for
# character (that is what keeps the nine state-dump hashes), and the mapped body
# below is new code that reads the geometry layer's metric, face metric and
# physical height (GEOM_DESIGN §3 cut 4, §5.5/G270).
#
# The routing rules, applied uniformly:
#   * the quadrature point, the volume measure and the physical basis gradients
#     come from the leaf metric (`quad_point`/`quad_measure`/`basis_grad`);
#   * every base-state closure (`Tbar`, `rho`, phase functions) is evaluated at
#     the physical **height** `h = height(g,x,y)`, and every directional term
#     uses `u . grad h = upward_velocity(g,x,y,ux,uz)` instead of `u_z`;
#   * every wall integral uses the **mapped outward unit normal** and the mapped
#     face Jacobian; the box's one-sided `2/h` derivative and sign table are the
#     box body's alone (the mapped wall flux is `grad(theta) . n` at the wall GLL
#     nodes, integrated with the wall GLL rule so that the box-map cross-check is
#     a round-off comparison; the signs are `+n` on the top side and `-n` on the
#     bottom, i.e. the same "positive upward" flux convention the box uses);
#   * every volume mean is normalised by the physical total measure
#     (`total_measure`), never by `total_area(qt)`.
#
# `nu_wall_sides(g)` names the two logical sides the Nusselt pair uses.  For
# every stage-0 geometry it is `(:yminus, :yplus)`; the annulus's inner/outer
# circles are `xminus`/`xplus`, which is why the diagnostics ask the geometry.

"""
    _semquad_wall_flux(g, sq, theta, side) -> (flux, measure)

Mapped wall integral on logical `side` of the perturbation temperature: the
outward-normal derivative `grad(theta) . n` at the wall GLL nodes, integrated
with the wall GLL rule (`face_jacobian * sq.w[q]`), and the wall measure with
the same rule.  The wall-node order and the GLL rule are the box wall bodies'
(E8), so on the box map the two agree to round-off.  The derivative is the
one-sided GLL derivative in the wall-normal reference direction composed with
the physical `J^{-T}` of the mapped element (`chain rule`), i.e. the mapped twin
of the box's `2/h` one-sided derivative.
"""
function _semquad_wall_flux(g::AbstractGeometry, sq::SemQuad,
                            theta::Vector{Float64}, side::Symbol)
    n = sq.p + 1
    isx = side === :xminus || side === :xplus
    st = 0.0
    wt = 0.0
    for lf in sq.leaves
        _on_logical_wall(lf, side, sq) || continue
        ids = sq.elem[lf]
        fm = _mapped_face_metric(g, sq, lf, side, sq.xi)
        @inbounds for q in 1:n
            xi, eta = _wall_ref_point(side, sq.xi[q])
            inv = jacobian_inverse(g, sq, lf, xi, eta)
            if isx
                i = side === :xplus ? n : 1
                j = q
            else
                i = q
                j = side === :yplus ? n : 1
            end
            dxi = 0.0
            deta = 0.0
            for a in 1:n
                dxi += sq.D[i, a] * theta[ids[(j - 1) * n + a]]
                deta += sq.D[j, a] * theta[ids[(a - 1) * n + i]]
            end
            gx = dxi * inv[1] + deta * inv[2]
            gy = dxi * inv[3] + deta * inv[4]
            nx, ny = face_normal(fm, q)
            w = face_jacobian(fm, q) * sq.w[q]
            st += w * (gx * nx + gy * ny)
            wt += w
        end
    end
    return st, wt
end

function _semquad_nu_walls_mapped(g::AbstractGeometry, sq::SemQuad,
                                  qt::Quadtree, theta::Vector{Float64},
                                  pr::Params)
    sideb, sidet = nu_wall_sides(g)
    # outward normal derivative, as the box body's `st`/`sb` are the upward
    # derivative on both walls; the box sign table (E18) becomes the `+`/`-`
    # normal convention here: Nu_bot = ref - (-flux_out_bot)/measure.
    st, wt = _semquad_wall_flux(g, sq, theta, sidet)
    sb, wb = _semquad_wall_flux(g, sq, theta, sideb)
    bot_ref, top_ref = semquad_base_wall_refs(pr, g)
    return top_ref - st / wt, bot_ref + sb / wb
end

# ---------------------------------------------------------------------------
# Stage 0.5b1: the annulus Nusselt pair
# ---------------------------------------------------------------------------
# The annulus normalisation is the benchmark one (ANNULUS_REFS §6.2/§6.5): the
# *per-wall* radial conduction flux.  With the arc-length weighted wall average
# `⟨·⟩`, the outer (top) wall has
#
#     ⟨-∂T/∂r⟩_top = -dTbar/dr(R1) - ⟨∂θ/∂r⟩_top,   st/wt = ⟨∂θ/∂r⟩_top
#     Nu_top = ⟨-∂T/∂r⟩_top / (-dTbar/dr(R1)) = (top_ref - st/wt) / top_ref
#
# and the inner wall (outward normal `-e_r`, so `sb/wb = -⟨∂θ/∂r⟩_inner`)
#
#     Nu_bot = (bot_ref + sb/wb) / bot_ref,   bot_ref = -dTbar/dr(R0).
#
# At `H = 0` the reference gradient is `1/(R |ln f|)`, so this is exactly the
# documented multiplier form `Nu_wall = R_wall |ln f| ⟨-∂T/∂r⟩ / ΔT` with
# `ΔT = 1` (ANNULUS_REFS §6.2; the published gadopt scales 1.3290170684486309 /
# 0.7303607313096079 are `R1 |ln f|` and `R0 |ln f|`), and a conductive state
# (`θ = 0`) gives `Nu = 1` exactly at both walls.  With `H ≠ 0` the per-wall
# reference is the H-inclusive base flux `Q_ref(r) = -2πr dTbar/dr`, i.e. this
# numerator is exactly `Q_wall/Q_ref(r_wall)`; the documented identity then
# reads `Nu_top Q_ref(R1) - Nu_bot Q_ref(R0) = H A` (ANNULUS_REFS §6.5), which
# the annulus gate checks.  The generic mapped body above keeps the box's
# single-gradient convention (it is the mapped twin of the box body and is used
# only by the box-in-height validation geometries), so the annulus dispatches to
# this method.
function _semquad_nu_walls_mapped(g::PolarAnnulus, sq::SemQuad,
                                  qt::Quadtree, theta::Vector{Float64},
                                  pr::Params)
    sideb, sidet = nu_wall_sides(g)
    st, wt = _semquad_wall_flux(g, sq, theta, sidet)
    sb, wb = _semquad_wall_flux(g, sq, theta, sideb)
    bot_ref, top_ref = semquad_base_wall_refs(pr, g)
    (top_ref > 0.0 && bot_ref > 0.0) ||
        throw(ArgumentError("the annulus wall Nusselt normalisation needs a " *
                            "positive conductive reference flux at both walls " *
                            "(-dTbar/dr = ($bot_ref, $top_ref) at r = " *
                            "($(g.R0), $(g.R1))); with H = $(pr.H) the " *
                            "internal heating has changed the sign of the " *
                            "radial base flux at a wall.  Reduce H so that " *
                            "-dTbar/dr > 0 at both radii (the per-wall " *
                            "Q_ref(r_wall) normalisation of ANNULUS_REFS " *
                            "§6.5)."))
    return (top_ref - st / wt) / top_ref, (bot_ref + sb / wb) / bot_ref
end

function _semquad_nu_energy_mapped(g::AbstractGeometry, sq::SemQuad,
                                   qt::Quadtree, pr::Params,
                                   theta::Vector{Float64},
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
            x, y = quad_point(m, sq, lf, fd, qx, qy)
            uxq = 0.0
            uzq = 0.0
            thq = 0.0
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                b = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                uxq += ux[ids[k]] * b
                uzq += uz[ids[k]] * b
                thq += theta[ids[k]] * b
            end
            # energy Nusselt: the convective heat flux is `theta (u . grad h)`.
            s += w * upward_velocity(g, x, y, uxq, uzq) * thq
        end
    end
    return 1.0 + 0.5 * pr.H + s / total_measure(g, sq, fd)
end

