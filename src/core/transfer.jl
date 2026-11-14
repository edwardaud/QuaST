# Solution transfer between quadtree SEM meshes (nodal restriction and L2 projection).
"""
    transfer_semquad(old, uold, new; mode=:nodal, zero_walls=false, qt_new=nothing)

Move a continuous quadtree SEM field from mesh `old` onto mesh `new`.

`mode=:nodal` evaluates the old field at the new GLL nodes (exact for
refinement, natural interpolation for coarsening).  `mode=:l2` solves the
global L2 projection `M_new u_new = b`, with `b_i = int phi_i u_old`; it is the
coarsening-consistent transfer and avoids the interface steps produced by
pointwise restriction.  With `zero_walls=true` the top/bottom nodes are held at
zero.  `qt_new` is optional; when supplied, hanging-node constraints are
applied after the projection.  Hanging constraints are otherwise the caller's
responsibility (`C'AC` in the assembly).
"""
function transfer_semquad(old::SemQuad, uold::Vector{Float64}, new::SemQuad;
                          mode::Symbol=:nodal, zero_walls::Bool=false,
                          qt_new::Union{Nothing,Quadtree}=nothing)
    mode in (:nodal, :l2) ||
        throw(ArgumentError("transfer mode must be :nodal or :l2"))
    length(uold) == old.N ||
        throw(ArgumentError("uold has $(length(uold)) entries, old mesh has $(old.N)"))
    old.Lx == new.Lx ||
        throw(ArgumentError("transfer_semquad needs equal box widths ($(old.Lx) vs $(new.Lx))"))
    oldleaves = Set(old.leaves)
    if mode === :nodal
        unew = zeros(Float64, new.N)
        n = new.p + 1
        w = SemQuadEvalWork(n)
        # Shared nodes (faces, corners, and the hanging nodes of a fine face)
        # are written by several new leaves.  Every duplicate evaluation sees
        # the same (x,y) and the same old leaf -- the coordinates are exact
        # dyadic sums and `_semquad_leaf_at` is deterministic -- so the node's
        # value is the same bit pattern; evaluate it once, at its first
        # occurrence.  This halves the evaluation count (90 000 -> 42 123 on
        # the 10 000-leaf state) and removes the two `lagrange_values`
        # allocations per evaluation.
        seen = falses(new.N)
        for lf in new.leaves
            hx = leaf_hx(new, lf[1]); hy = leaf_hy(new, lf[1])
            x0 = lf[2] * hx; y0 = lf[3] * hy
            ids = new.elem[lf]
            for j in 1:n, i in 1:n
                gid = ids[(j - 1) * n + i]
                seen[gid] && continue
                seen[gid] = true
                x = x0 + 0.5 * (new.xi[i] + 1.0) * hx
                y = y0 + 0.5 * (new.xi[j] + 1.0) * hy
                unew[gid] = _semquad_eval_at!(w, old, uold, x, y, oldleaves)
            end
        end
        return unew
    end

    p = new.p
    n = p + 1
    fd = make_fedata(p, p + 2)
    M = semquad_mass(new)
    b = zeros(Float64, new.N)
    for lf in new.leaves
        hx = leaf_hx(new, lf[1]); hy = leaf_hy(new, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        ids = new.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            uq = _semquad_eval_at(old, uold, x, y, oldleaves)
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                phi = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                b[ids[k]] += w * phi * uq
            end
        end
    end
    if zero_walls
        free = [g for g in 1:new.N if 0.0 < new.coords[g][2] < 1.0]
        unew = zeros(Float64, new.N)
        isempty(free) || (unew[free] = M[free, free] \ b[free])
    else
        unew = M \ b
    end
    if qt_new !== nothing
        cons = semquad_constraints(new, qt_new)
        for (c, (ms, ws)) in cons
            unew[c] = sum(ws .* unew[ms])
        end
    end
    return unew
end

