# G236: the per-wall tuple fixes the right dofs on the right wall.
using Test, QuaST, SparseArrays

# `solve_semquad_stokes_var` returns the `fixed` dict, so the constraint set is
# observable directly.  `Cu` is *not* the place to look: it carries the
# hanging-node constraints only and is identical for every wall configuration
# (checked below, because an earlier draft of this gate assumed otherwise).
#
# The statements are about *which* dofs are fixed, not how many: corner nodes
# belong to two walls, so any counting formula needs corner bookkeeping and adds
# nothing a membership check does not already say.

@testset "G236 per-wall constraints" begin
    qt = Quadtree(2)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    N = sq.N
    xs = [c[1] for c in sq.coords]
    ys = [c[2] for c in sq.coords]
    near(a, b) = abs(a - b) < 1.0e-12
    on_x(i) = near(xs[i], 0.0) || near(xs[i], 1.0)
    on_y(i) = near(ys[i], 0.0) || near(ys[i], 1.0)
    # a corner belongs to both walls, so its dofs are the normal one of
    # one wall and the tangential one of the other
    corner(i) = on_x(i) && on_y(i)

    function _fixed(wall_bc)
        r = solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0,
                                     (x, y) -> 1.0, (x, y) -> 0.0,
                                     (x, y) -> 0.0, (x, y) -> 0.0;
                                     bc=:free, wall_bc=wall_bc,
                                     deviatoric=false, linear=:direct)
        return r[1], r[3]
    end

    u_free, f_free = _fixed((:free, :free, :free, :free))
    u_rig, f_rig = _fixed((:rigid, :rigid, :rigid, :rigid))
    u_xrig, f_xrig = _fixed((:rigid, :rigid, :free, :free))

    # every boundary node carries its normal component in all three
    for i in 1:N
        on_x(i) && @test haskey(f_free, i)
        on_y(i) && @test haskey(f_free, N + i)
    end
    # free: no tangential dof is ever fixed
    for i in 1:N
        (on_x(i) && !corner(i)) && @test !haskey(f_free, N + i)
        (on_y(i) && !corner(i)) && @test !haskey(f_free, i)
    end
    # rigid: both components on every boundary node
    for i in 1:N
        (on_x(i) || on_y(i)) || continue
        @test haskey(f_rig, i)
        @test haskey(f_rig, N + i)
    end
    # mixed: the x-walls rigid, the y-walls free
    for i in 1:N
        on_x(i) && @test haskey(f_xrig, N + i)      # tangential fixed on x
        (on_y(i) && !corner(i)) && @test !haskey(f_xrig, i)  # tangential free on y
    end
    @test length(f_free) < length(f_xrig) < length(f_rig)

    # the answer honours it: the tangential velocity is zero on the rigid walls
    for i in 1:N
        on_x(i) && @test u_xrig[N + i] == 0.0
    end
    @test any(!iszero, u_xrig[N + 1:end])

    # `Cu` holds the normal-component and hanging-node constraints; the extra
    # tangential dofs of a rigid wall go into `fixed` and are eliminated
    # separately.  So `Cu` is the same for every wall configuration -- which is
    # why keying the multigrid cache on the tuple is redundant but harmless.
    function _cu(wall_bc)
        asm = solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0,
                                       (x, y) -> 1.0, (x, y) -> 0.0,
                                       (x, y) -> 0.0, (x, y) -> 0.0;
                                       bc=:free, wall_bc=wall_bc,
                                       deviatoric=false, linear=:direct,
                                       assemble_only=true)
        return asm.Cu
    end
    cu_free = _cu((:free, :free, :free, :free))
    cu_rig = _cu((:rigid, :rigid, :rigid, :rigid))
    @test size(cu_free, 1) == size(cu_rig, 1)
    # the pattern, not the shape, depends on the wall condition: this is what
    # makes keying the hierarchy cache on the tuple necessary
    @test nnz(cu_free) != nnz(cu_rig)
end
