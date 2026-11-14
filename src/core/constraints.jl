# Hanging-node constraints of the quadtree SEM space.
"""Hanging-node constraints: fine face node = interpolation of the coarse face."""
function semquad_constraints(sq::SemQuad, qt::Quadtree)
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

