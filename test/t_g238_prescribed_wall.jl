# G238: a prescribed, position-dependent wall velocity gives the analytic shear.
#
# `bc=:rigid` fixes both components, with the normal one from `uDx`/`uDz`.  With
# `uDx(x,y) = U*y` on a lid and floor at y = 1 and y = 0 the boundary data is
# exactly the Couette field: ux = U*y on the y-walls (tangential) and on the x-walls
# (normal), uz = 0 everywhere.  The exact solution of the Stokes problem with
# constant viscosity, no buoyancy and no body force is therefore
# `u = (U*y, 0)`, `p = 0`, and the discrete answer must reproduce it to solver
# accuracy.  This is the analytic wall-velocity test that "add a Poiseuille gate"
# was asking for, and it needs no new API.

using Test, QuaST

@testset "G238 prescribed wall velocity reproduces Couette" begin
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

    # no buoyancy, no body force: the flow is driven only by the lid
    uDx = (x, y) -> U * y
    uDz = (x, y) -> 0.0
    z0 = (x, y) -> 0.0
    r = solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0, z0, z0, uDx, uDz;
                                 bc=:rigid, deviatoric=false, linear=:direct)
    u = r[1]
    ux = u[1:N]
    uz = u[N + 1:end]

    err = maximum(abs.(ux .- U .* ys)) / U
    @test err < 1.0e-10
    @test maximum(abs, uz) < 1.0e-10

    # the same data through the per-wall tuple: all walls rigid is `:same`
    pr = Params(Ra=0.0, H=0.0, eta_law=:expT, deta_T=1.0, bc=:rigid)
    @test pr.wall_bc == (:same, :same, :same, :same)
end
