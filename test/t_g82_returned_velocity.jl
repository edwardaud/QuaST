# G82: the returned velocity must match the returned mesh/theta.
#
# A final AMR cycle resets ux/uz; the driver must re-solve before returning,
# otherwise cfl/vorticity diagnostics see a zero velocity field.
using Test
using QuaST

@testset "G82 returned velocity after AMR" begin
    qt = wall_graded_tree(2, 6; strip=0.10)
    sq = make_semquad(qt, 3)
    th = [0.05 * exp(-((x - 0.5) / 0.2)^2 - ((y - 0.4) / 0.2)^2) * sinpi(y)
          for (x, y) in sq.coords]
    pr = Params(Ra=1.0e7, H=5.0, eta_law=:expT, deta_T=1.0e3,
                bc=:free, eta_min=1.0e-6, eta_max=1.0e6)
    # Policy-agnostic gate: the balanced cycle is the only AMR policy.
    res = run_semquad_simulation(qt, pr, th;
                                 p=3, tstop=1.0, dt0=1.0e-9, dtmax=1.0e-9,
                                 cfl=0.0, maxsteps=1, maxpic=1,
                                 initial_refine=0, amr_every=1,
                                 refine_frac=0.2,
                                 coarsen_frac=0.0, max_level=6,
                                 wall_min_level=6, wall_strip=0.10,
                                 symmetry=:x, linear=:direct,
                                 stokes_tol=1.0e-6, advection=:skew,
                                 verbose=false)
    sp = hypot.(res.ux, res.uz)
    @test maximum(sp) > 0.0
    @test length(res.ux) == res.ss.su.N
end
