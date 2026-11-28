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

# Affine-constraint builder shared by the Stokes solves, energy step and
# weakening tangent.  It lives with the other constraint primitives so the
# stokes diagonal/preconditioner file does not own a core affine map.
"""
    _affine_constraint(N, cons, fixed, offset=nothing) -> (u0, C, free)

Affine constraint `u = u0 + C u_free` from the (homogeneous) linear relations
`cons` and the Dirichlet values `fixed`.

`offset` (stage 0.5b2, default `nothing`) is an optional per-dof *constant* term
added to `u0` for constrained dofs whose relation is **affine** rather than
homogeneous: a curved free-slip wall's normal-velocity data
`n . u = g` with `g != 0` eliminates one Cartesian component with the constant
`g/n_p` (see `stokes/wall_constraints.jl`).  The box path and every historical
call site pass no offsets, so `u0` is initialised to zeros exactly as before and
the arithmetic is bit-identical; offsets are folded *before* the fixed-master
resolution below, so a constraint whose master is fixed accumulates its constant
and its folded data in the same `u0` entry.
"""
# Two positional methods rather than a default argument: the 3-argument box call
# (the hot path, used twice per solve) then dispatches statically to the
# 4-argument body without the default-argument forwarder's extra call frame,
# which the stage-0.5b2 allocation A/B measured at +2 small allocations per
# solve.
@inline _affine_constraint(N::Int,
                           cons::Dict{Int,Tuple{Vector{Int},Vector{Float64}}},
                           fixed::Dict{Int,Float64}) =
    _affine_constraint(N, cons, fixed, nothing)

function _affine_constraint(N::Int, cons::Dict{Int,Tuple{Vector{Int},Vector{Float64}}},
                            fixed::Dict{Int,Float64},
                            offset::Union{Nothing,Dict{Int,Float64}})
    u0 = zeros(Float64, N)
    if offset !== nothing
        for (i, v) in offset
            u0[i] = v
        end
    end
    for (i, v) in fixed
        u0[i] = v
    end
    # move fixed masters into u0, iterate to resolve chains
    work = Dict(c => (copy(ms), copy(ws)) for (c, (ms, ws)) in cons)
    active = trues(length(work))
    keysv = collect(keys(work))
    keep = Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
    for _ in 1:8
        keep = Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
        changed = false
        for (c, (ms, ws)) in work
            haskey(fixed, c) && continue
            nms = Int[]; nws = Float64[]
            for (m, w) in zip(ms, ws)
                if haskey(fixed, m)
                    u0[c] += w * fixed[m]
                    changed = true
                elseif haskey(work, m) && haskey(fixed, m) == false
                    push!(nms, m); push!(nws, w)
                else
                    push!(nms, m); push!(nws, w)
                end
            end
            keep[c] = (nms, nws)
        end
        work = keep
        changed || break
    end
    # free dofs: not fixed and not constrained
    free = [i for i in 1:N if !haskey(fixed, i) && !haskey(work, i)]
    fmap = Dict(v => k for (k, v) in enumerate(free))
    I = Int[]; J = Int[]; V = Float64[]
    for (k, i) in enumerate(free)
        push!(I, i); push!(J, k); push!(V, 1.0)
    end
    for (c, (ms, ws)) in work
        for (m, w) in zip(ms, ws)
            push!(I, c); push!(J, fmap[m]); push!(V, w)
        end
    end
    C = sparse(I, J, V, N, length(free))
    return u0, C, free
end
