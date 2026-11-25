# G287 (remediation batch P0 part 1): curved-wall `:rigid` rows at hanging wall
# nodes (AUDIT_GATES M11 (ii), mutation m16b).
#
# The hole: `wall_constraints.jl`'s hanging-wall pass (the `:rigid` branch that
# the audit mutated at line 328) was never exercised by a gate, so m16b
# survived G275 and G276.
#
# **Reachability finding (measured, recorded here as part of the fix).**  On an
# axis-aligned quadtree the *balanced* wall nodes of a logical side are a
# partition of that side into leaf faces, so a fine node can never lie strictly
# inside a coarser leaf's wall face: the only way for a node to be both a wall
# node and a hanging node is to be the interior endpoint of a face *parallel* to
# the wall, and there the coarse/fine face endpoints coincide (they are the
# shared corner nodes, which `semquad_constraints` skips by design).  The gate
# *asserts* this: for clustered ring and wedge trees the intersection
# `keys(conds) ∩ keys(cons_hang)` is empty (printed 0).  So the branch is dead
# test surface on the production path, exactly as the audit suspected, but it is
# not dead *code* -- a direct caller of `_semquad_mapped_wall_data` can declare a
# wall node hanging (the internal API takes `cons_hang` as an argument).
#
# The gate therefore pins the branch with a white-box state that a *physical*
# fixture would need: a clustered annulus tree with a `:rigid` inner circle, one
# real wall node declared hanging with two synthetic masters, and the exact
# translation field u = (0.3, -0.2) prescribed on the walls.  The assertions are
# the wall-frame Dirichlet values at that node and the affine representation of
# the admissible velocity (`u0`, `Cu`) -- a state, not a mirror.  With the
# audit's m16b flip (`wbc === :rigid` -> `wbc !== :rigid` at the hanging pass),
# the node takes the free-slip branch: no `fixed` value, a `cons` row instead,
# and `u0`/`Cu` change; every assertion below is then red.  On the fixed tree
# they pass.
#
# Why the assertions are not tautological: the prescribed translation is
# reconstructed from the *mapped wall frame* (n, tau at the node), so a wrong
# normal/tangent or swapped un/ut is red; the affine row is checked as `u0` fixed
# and `Cu` row zero, so a branch that leaves the hanging interpolation in place
# is red; and the reachability count is an assertion about the tree itself, so a
# future tree change that makes the branch reachable is visible.
using Test
using LinearAlgebra
using QuaST
const MS287 = QuaST

g287_zero(x, y) = 0.0
g287_udx(x, y) = 0.3
g287_udz(x, y) = -0.2

function g287_tree(level; seam=:periodic, cells_y=4)
    qt = Quadtree(level + 2; cells_x=1, cells_y=cells_y,
                  seam=seam === :periodic ? :periodic : :none)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    # cluster toward the inner circle and the seam
    MS287.refine_where!(qt, (l, ix, iy) -> l < level && ix < 3 && iy < 6)
    balance!(qt)
    return qt
end

function g287_wall_intersection(g, qt, p)
    ss = make_semstokes(qt, p; geom=g)
    sq = ss.su
    bmask = MS287._semquad_boundary_face_mask(qt, sq)
    conds = MS287._semquad_wall_frames(g, sq, bmask)
    cons = semquad_constraints(sq, qt)
    return count(node -> haskey(cons, node), collect(keys(conds))), sq, bmask, conds
end

@testset "G287 curved rigid wall rows at a hanging wall node" begin
    g = PolarAnnulus(1.0, 2.0, 2pi; seam=:periodic)

    # ---- (a) reachability: balanced ring trees have no hanging wall nodes ----
    nreach = 0
    for trial in 1:3
        qt = g287_tree(trial)
        n, sq, bmask, conds = g287_wall_intersection(g, qt, 2)
        @test n == 0
        @test !isempty(conds)                  # the tree does have wall nodes
        nreach += n
    end
    println("G287 reachability: hanging-wall nodes over balanced ring trees = ",
            nreach, " (structural: wall faces partition the side)")
    @test nreach == 0
    # the ring's seam *does* produce hanging nodes, so the count above is not
    # zero because constraints are absent: force a one-sided seam refinement
    # (G276 (b4)'s recipe) and assert the constraint set is non-empty.
    qtc = Quadtree(5; cells_x=1, cells_y=4, seam=:periodic)
    for _ in 1:3, lf in collect(qtc.leaves)
        refine!(qtc, lf)
    end
    for lf in collect(qtc.leaves)
        lf[3] == 0 && lf[2] <= 4 && refine!(qtc, lf)
    end
    balance!(qtc)
    ssc = make_semstokes(qtc, 2; geom=g)
    @test !isempty(semquad_constraints(ssc.su, qtc))

    # ---- (b) white-box: one wall node on the rigid inner circle is hanging ---
    qt = g287_tree(3)
    ss = make_semstokes(qt, 2; geom=g)
    sq = ss.su
    nu = sq.N
    bmask = MS287._semquad_boundary_face_mask(qt, sq)
    conds = MS287._semquad_wall_frames(g, sq, bmask)
    cons_hang = semquad_constraints(sq, qt)
    # inner circle = wall 1 (`:xminus`, the `:inner` boundary of PolarAnnulus)
    node = 0
    for (nd, cs) in conds
        if length(cs) == 1 && cs[1].wi == 1 && !haskey(cons_hang, nd)
            node = nd
            break
        end
    end
    @test node != 0
    @test haskey(conds, node) && length(conds[node]) == 1
    c = conds[node][1]
    @test c.wi == 1 && abs(hypot(c.x, c.y) - 1.0) < 1.0e-12   # on the inner circle
    # synthetic masters: two other free wall nodes (the rigid branch does not
    # read them; the free branch of the mutation does, so they must be valid)
    others = [nd for nd in keys(conds)
              if nd != node && nd != nu + node && !haskey(cons_hang, nd)]
    @test length(others) >= 2
    cons_hang2 = copy(cons_hang)
    cons_hang2[node] = ([others[1], others[2]], [0.5, 0.5])
    cons_hang2[nu + node] = ([nu + others[1], nu + others[2]], [0.5, 0.5])

    wbc = (:rigid, :rigid, :rigid, :rigid)
    wd = MS287._semquad_mapped_wall_data(g, sq, bmask, wbc,
                                         (nothing, nothing, nothing, nothing),
                                         (nothing, nothing, nothing, nothing),
                                         g287_udx, g287_udz, cons_hang2)
    @test wd.umax > 0.0
    # the exact translation reconstructed in the wall frame = uD at the node
    unv = 0.3 * c.nx - 0.2 * c.ny
    utv = 0.3 * c.tx - 0.2 * c.ty
    @test wd.un[1][node] ≈ unv atol=1.0e-15
    @test wd.fixed[node] ≈ 0.3 atol=1.0e-14
    @test wd.fixed[nu + node] ≈ -0.2 atol=1.0e-14
    # a rigid row replaces the hanging row: no interpolation row survives
    @test !haskey(wd.cons, node) && !haskey(wd.cons, nu + node)
    @test !haskey(wd.offsets, node) && !haskey(wd.offsets, nu + node)
    # the affine state of the admissible velocity: the node is Dirichlet
    cons2, offs = MS287._semquad_merge_wall_constraints(cons_hang2, wd)
    u0, Cu, _ = MS287._affine_constraint(2 * nu, cons2, wd.fixed, offs)
    @test u0[node] ≈ 0.3 atol=1.0e-13
    @test u0[nu + node] ≈ -0.2 atol=1.0e-13
    @test maximum(abs, Cu[node, :]) == 0.0
    @test maximum(abs, Cu[nu + node, :]) == 0.0
    # contrast: with the wall declared `:free` (the mutation's branch for a
    # flipped test on a `:free` wall) the same node is a constraint, not fixed
    wdf = MS287._semquad_mapped_wall_data(g, sq, bmask,
                                          (:free, :free, :free, :free),
                                          (nothing, nothing, nothing, nothing),
                                          (nothing, nothing, nothing, nothing),
                                          g287_udx, g287_udz, cons_hang2)
    @test !haskey(wdf.fixed, node) || haskey(wdf.cons, node)
    @test haskey(wdf.cons, node)           # the free branch keeps a row here
    # the free branch's affine state is not the prescribed translation: this is
    # the state a flipped branch test would produce instead
    cons2f, offsf = MS287._semquad_merge_wall_constraints(cons_hang2, wdf)
    u0f, Cuf, _ = MS287._affine_constraint(2 * nu, cons2f, wdf.fixed, offsf)
    @test abs(u0f[node] - 0.3) > 1.0e-6
end
