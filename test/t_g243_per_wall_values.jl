# G243: per-wall tangential values make a lid-driven cavity expressible.
#
# The global `uDx`/`uDz` closures apply to every wall, so prescribing the top wall's
# tangential velocity also prescribed the x-walls' *normal* velocity, which is a
# different problem (and the reason the ASPECT lid-driven viscoelastic benchmark
# could not be reproduced).  `Params.wall_u` gives one tangential value per wall.
#
# The test is a lid-driven cavity: top wall `:rigid` with tangential `U`, the other
# three `:free` with `uDx = uDz = 0`.  Requirements: `ux = U` on the top wall,
# `ux = 0` on the x-walls (their normal), and the flow differs from what the global
# closure produces.

using Test, QuaST

@testset "G243 per-wall tangential values" begin
    U = 1.0
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
    z0 = (x, y) -> 0.0
    lid = (x, y) -> U

    # the lid-driven cavity: only the top wall is rigid, and its tangential value
    # comes from wall_u
    r = solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0, z0, z0, z0, z0;
                                 bc=:free,
                                 wall_bc=(:free, :free, :free, :rigid),
                                 wall_u=(nothing, nothing, nothing, lid),
                                 deviatoric=false, linear=:direct)
    u = r[1]
    ux = u[1:N]

    # the top wall moves at U, the x-walls' normal velocity is zero
    for i in 1:N
        near(ys[i], 1.0) && @test isapprox(ux[i], U; atol=1.0e-10)
        # a corner belongs to the lid as well, where ux = U is correct
        on_side = (near(xs[i], 0.0) || near(xs[i], 1.0)) &&
                  !(near(ys[i], 0.0) || near(ys[i], 1.0))
        on_side && @test abs(ux[i]) < 1.0e-12
    end
    @test maximum(abs, ux) > 0.1 * U          # the lid actually drives something

    # the same card without wall_u: the global closure puts U on the x-walls too, so
    # the answer must differ
    r2 = solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0, z0, z0,
                                  (x, y) -> U, z0;
                                  bc=:free,
                                  wall_bc=(:free, :free, :free, :rigid),
                                  deviatoric=false, linear=:direct)
    @test r2[1] != u

    # the default is all-nothing, i.e. the historical behaviour
    @test Params().wall_u == (nothing, nothing, nothing, nothing)
end
