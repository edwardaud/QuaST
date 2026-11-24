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

Mapped geometries (stage 0 cut 6).  Both meshes must carry the **same**
geometry; the box body above is the historical arithmetic, character for
character, and runs unchanged whenever both spaces are `CartesianBox`.  Under a
map the node identity is logical, so the transfer is formulated on the logical
identity (GEOM_DESIGN §2.1 rule 4 and §6.5):

  * the **new node's coordinate** is `element_map(g, new, lf, xi_i, xi_j)` (the
    map image of its logical position), but the *old field is evaluated at the
    node's logical position*, not at a recovered physical point: the old leaf is
    `locate_leaf_logical(old, S, T)` and the old reference abscissa is
    `leaf_ref(old, lf_old, S, T)`.  This is the right formulation because the
    map is a single global diffeomorphism, so the old polynomial interpolant
    must be evaluated at the same logical point both meshes share; it avoids a
    map inversion (which a general or stated ALE map may not provide in closed
    form) and is exact for every node that the two logical subdivisions share.
    The nodal transfer is then **exact for any field that is a polynomial of
    degree <= p per logical direction** on the old mesh (the map image only
    locates the node, it does not enter the interpolation), which is the
    design's G272 (a) statement;
  * the L2 transfer's mass and load use the mapped measure (`semquad_mass` and
    `quad_measure`) and the same logical evaluation of the old field.
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
    if old.geom isa CartesianBox && new.geom isa CartesianBox
        return _transfer_semquad_box(old, uold, new; mode=mode,
                                     zero_walls=zero_walls, qt_new=qt_new)
    end
    # A mapped transfer is a transfer on one physical domain: two different maps
    # (including box vs mapped) would move every node and every measure.
    geometry_identity(old.geom) == geometry_identity(new.geom) ||
        throw(ArgumentError("transfer_semquad needs the same geometry on both " *
                            "meshes (old=$(geometry_identity(old.geom)), " *
                            "new=$(geometry_identity(new.geom)))"))
    return _transfer_semquad_mapped(new.geom, old, uold, new; mode=mode,
                                    zero_walls=zero_walls, qt_new=qt_new)
end

# Historical box body, verbatim: do not rewrite it (GEOM_DESIGN §4,
# `test/t_g266_geometry_identity.jl` pins it bit for bit).
function _transfer_semquad_box(old::SemQuad, uold::Vector{Float64},
                               new::SemQuad; mode::Symbol=:nodal,
                               zero_walls::Bool=false,
                               qt_new::Union{Nothing,Quadtree}=nothing)
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

# ---------------------------------------------------------------------------
# Mapped transfer (cut 6): the same two algorithms on the logical identity.
# ---------------------------------------------------------------------------

"""
    _transfer_semquad_mapped(g, old, uold, new; mode, zero_walls, qt_new)

Mapped transfer: the new node's logical position gives the old-field leaf and
reference abscissa (`locate_leaf_logical` + `leaf_ref`), the old field is
evaluated there with the reference-space Lagrange form, and the L2 arm uses the
mapped mass/measure.  The map is used for *coordinates* only (`element_map`
images): a node's value is the old interpolant at the node's logical position,
which is what both meshes share (GEOM_DESIGN §2.1 rule 4, §6.5).
"""
function _transfer_semquad_mapped(g::AbstractGeometry, old::SemQuad,
                                  uold::Vector{Float64}, new::SemQuad;
                                  mode::Symbol=:nodal,
                                  zero_walls::Bool=false,
                                  qt_new::Union{Nothing,Quadtree}=nothing)
    oldleaves = Set(old.leaves)
    if mode === :nodal
        unew = zeros(Float64, new.N)
        n = new.p + 1
        w = SemQuadEvalWork(n)
        # Shared nodes are written by several new leaves; the logical position
        # is canonical (`global_logical`), so the first occurrence is the value.
        seen = falses(new.N)
        for lf in new.leaves
            ids = new.elem[lf]
            for j in 1:n, i in 1:n
                gid = ids[(j - 1) * n + i]
                seen[gid] && continue
                seen[gid] = true
                S, T = global_logical(new, lf, new.xi[i], new.xi[j])
                lfo = locate_leaf_logical(old, S, T, oldleaves)
                xio, etao = leaf_ref(old, lfo, S, T)
                unew[gid] = _semquad_eval_ref!(w, old, uold, xio, etao, lfo)
            end
        end
        return unew
    end

    p = new.p
    n = p + 1
    fd = make_fedata(p, p + 2)
    M = semquad_mass(new)
    b = zeros(Float64, new.N)
    we = SemQuadEvalWork(old.p + 1)
    for lf in new.leaves
        lm = leaf_metric(g, new, lf, fd)
        ids = new.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            qx_l = fd.gl_x[qx]; qy_l = fd.gl_x[qy]
            w = quad_measure(lm, fd, qx, qy)
            S, T = global_logical(new, lf, qx_l, qy_l)
            lfo = locate_leaf_logical(old, S, T, oldleaves)
            xio, etao = leaf_ref(old, lfo, S, T)
            uq = _semquad_eval_ref!(we, old, uold, xio, etao, lfo)
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                phi = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                b[ids[k]] += w * phi * uq
            end
        end
    end
    # The walls are the level sets of the physical height: a node is free when
    # its height is strictly inside the layer (box: `y`, exactly the historical
    # comparison through the `height` accessor).
    if zero_walls
        free = [gid for gid in 1:new.N
                if 0.0 < height(g, new.coords[gid][1], new.coords[gid][2]) < 1.0]
        unew = zeros(Float64, new.N)
        isempty(free) || (unew[free] = M[free, free] \ b[free])
    else
        unew = M \ b
    end
    if qt_new !== nothing
        cons = semquad_constraints(new, qt_new; geom=g)
        for (c, (ms, ws)) in cons
            unew[c] = sum(ws .* unew[ms])
        end
    end
    return unew
end
