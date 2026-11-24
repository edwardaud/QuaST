# Hanging-node constraints of the quadtree SEM space.
#
# The constrained node is a fine face node; its value is the coarse face's GLL
# interpolation.  The box method is the historical physical-coordinate body
# (unchanged); a mapped geometry computes the same thing from **logical**
# coordinates (GEOM_DESIGN §1.5(b)/§2.1 rule 4): the pair topology comes from
# `face_pairs(qt, geom)` (index arithmetic) and the coarse reference abscissa
# from the fine node's global logical position, never from a physical distance.
"""
    semquad_constraints(sq, qt; geom=sq.geom)

Hanging-node constraints of the space `sq` on tree `qt`.  The box body is the
historical one; a mapped geometry uses the logical method below.
"""
function semquad_constraints(sq::SemQuad, qt::Quadtree;
                             geom::AbstractGeometry=sq.geom)
    return _semquad_constraints(geom, sq, qt)
end

# Historical box body, verbatim (GEOM_DESIGN E2/E3/E8/E11): do not rewrite it.
function _semquad_constraints(::CartesianBox, sq::SemQuad, qt::Quadtree)
    p = sq.p
    n = p + 1
    rule = gll_rule(p)
    cons = Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
    for pf in face_pairs(qt)
        L, R, dir = pf.leafL, pf.leafR, pf.dir
        hL = leaf_hy(sq, L[1]); hR = leaf_hy(sq, R[1])
        hxL = leaf_hx(sq, L[1]); hxR = leaf_hx(sq, R[1])
        hL > hR + 1.0e-15 || continue           # only coarse-fine, coarse = L
        if dir === :xplus || dir === :xminus
            iFL = dir === :xplus ? n : 1
            iFR = dir === :xplus ? 1 : n
            yL = L[3] * hL; yR = R[3] * hR
            for q in 1:n
                yq = yR + 0.5 * (sq.xi[q] + 1.0) * hR
                xiL = 2.0 * (yq - yL) / hL - 1.0
                minimum(abs.(sq.xi .- xiL)) < 1.0e-10 && continue
                vals, _ = lagrange_eval(rule, xiL)
                c = sq.elem[R][(q - 1) * n + iFR]
                ms = [sq.elem[L][(a - 1) * n + iFL] for a in 1:n]
                cons[c] = (ms, vals)
            end
        else
            jFL = dir === :yplus ? n : 1
            jFR = dir === :yplus ? 1 : n
            xL = L[2] * hxL; xR = R[2] * hxR
            for q in 1:n
                xq = xR + 0.5 * (sq.xi[q] + 1.0) * hxR
                xiL = 2.0 * (xq - xL) / hxL - 1.0
                minimum(abs.(sq.xi .- xiL)) < 1.0e-10 && continue
                vals, _ = lagrange_eval(rule, xiL)
                c = sq.elem[R][(jFR - 1) * n + q]
                ms = [sq.elem[L][(jFL - 1) * n + a] for a in 1:n]
                cons[c] = (ms, vals)
            end
        end
    end
    return resolve_constraints(cons)
end

# Mapped body: the coarser leaf is `pf.leafL`; the fine node's global logical
# coordinate along the face is computed from the fine leaf alone (canonical
# formula), then converted to the coarse leaf's reference abscissa:
#   xi_L = 2*(T_fine*nside_L - L[3]) - 1.
# The coarse GLL nodes that coincide with a fine node are skipped (they are the
# shared nodes, not constraints), exactly as in the box body; multi-level
# masters are resolved afterwards by `resolve_constraints`.
function _semquad_constraints(g::AbstractGeometry, sq::SemQuad, qt::Quadtree)
    p = sq.p
    n = p + 1
    rule = gll_rule(p)
    cons = Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
    for pf in face_pairs(qt, g)
        L, R, dir = pf.leafL, pf.leafR, pf.dir
        lL = L[1]; lR = R[1]
        lR > lL || continue                     # only coarse-fine, coarse = L
        if dir === :xplus || dir === :xminus
            iFL = dir === :xplus ? n : 1
            iFR = dir === :xplus ? 1 : n
            nsideL = qt.ny0 * (1 << lL)
            nsideR = qt.ny0 * (1 << lR)
            for q in 1:n
                Tf = (R[3] + 0.5 * (sq.xi[q] + 1.0)) / nsideR
                xiL = 2.0 * (Tf * nsideL - L[3]) - 1.0
                minimum(abs.(sq.xi .- xiL)) < 1.0e-10 && continue
                vals, _ = lagrange_eval(rule, xiL)
                c = sq.elem[R][(q - 1) * n + iFR]
                ms = [sq.elem[L][(a - 1) * n + iFL] for a in 1:n]
                cons[c] = (ms, vals)
            end
        else
            jFL = dir === :yplus ? n : 1
            jFR = dir === :yplus ? 1 : n
            nsideL = qt.nx0 * (1 << lL)
            nsideR = qt.nx0 * (1 << lR)
            for q in 1:n
                Sf = (R[2] + 0.5 * (sq.xi[q] + 1.0)) / nsideR
                xiL = 2.0 * (Sf * nsideL - L[2]) - 1.0
                minimum(abs.(sq.xi .- xiL)) < 1.0e-10 && continue
                vals, _ = lagrange_eval(rule, xiL)
                c = sq.elem[R][(jFR - 1) * n + q]
                ms = [sq.elem[L][(jFL - 1) * n + a] for a in 1:n]
                cons[c] = (ms, vals)
            end
        end
    end
    return resolve_constraints(cons)
end

"""Substitute constrained masters by their own constraints (multi-level faces)."""
function resolve_constraints(cons::Dict{Int,Tuple{Vector{Int},Vector{Float64}}})
    for _ in 1:8
        changed = false
        for (c, (ms, ws)) in collect(cons)
            newms = Int[]; newws = Float64[]
            for (m, w) in zip(ms, ws)
                if haskey(cons, m)
                    m2, w2 = cons[m]
                    for (mm, ww) in zip(m2, w2)
                        push!(newms, mm); push!(newws, w * ww)
                    end
                else
                    push!(newms, m); push!(newws, w)
                end
            end
            if newms != ms
                cons[c] = (newms, newws)
                changed = true
            end
        end
        changed || break
    end
    return cons
end
