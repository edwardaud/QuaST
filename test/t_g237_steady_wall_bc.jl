# G237: the DP/Newton steady path honours Params.wall_bc.
using Test, QuaST

# `semquad_dp_stokes_newton` builds its solver through `_semquad_newton_saddle`,
# which now takes the per-wall tuple, and the driver passes `pr.wall_bc` at both of
# its call sites.  The assertions below are `@test_broken` because the steady answer
# still does **not** change when the tuple changes: something inside the Newton
# saddle rebuilds the constraint set from the single `bc` symbol, so passing the
# tuple is necessary but not yet sufficient.  Recorded as an open correctness gap in
# docs/dev/PHASE1.md (item B1); when the saddle starts honouring the tuple these
# turn into ordinary tests.

@testset "G237 steady path honours wall_bc" begin
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

    # a conductive base state plus a small perturbation: a body force in x, no
    # buoyancy, so the flow is driven only by the boundary data
    th = [0.05 * sinpi(z) * cospi(2.0 * x / 2.5) for (x, z) in sq.coords]

    function _solve(wall_bc)
        pr = Params(Ra=1.0e3, H=0.0, eta_law=:expT, deta_T=1.0, bc=:rigid,
                    wall_bc=wall_bc)
        res = semquad_dp_stokes_newton(sq, qt, ss, pr, th; maxit=8,
                                       picard_warmup=3, linear=:direct,
                                       verbose=false)
        return res
    end

    a = _solve((:same, :same, :same, :same))          # all rigid, as bc says
    b = _solve((:rigid, :rigid, :free, :free))        # y-walls free
    @test all(isfinite, a.u)
    @test all(isfinite, b.u)
    @test_broken a.u != b.u

    # with no drifting forcing the difference must show in the tangential
    # component on the y-walls, which is what :free releases
    du = maximum(abs, a.u .- b.u)
    @test_broken du > 0.0
end
