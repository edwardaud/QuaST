# G113: the driver honours pr.bc — :rigid walls differ from :free and lower Vrms.
using Test, QuaST

@testset "G113 driver honours pr.bc (free vs rigid)" begin
    # Before the fix, solve_flow! hard-coded bc=:free, so pr.bc had no effect
    # and these two runs would be bit-identical.  :rigid adds the tangential
    # no-slip condition on all four walls, which must reduce Vrms.
    function run_bc(bc::Symbol)
        qt = Quadtree(5)
        for _ in 1:3, lf in collect(qt.leaves); refine!(qt, lf); end
        balance!(qt)
        ss = make_semstokes(qt, 2)
        theta0 = [0.12 * exp(-((x - 0.5) / 0.12)^2 - ((z - 0.35) / 0.10)^2) * sinpi(z)
                  for (x, z) in ss.su.coords]
        pr = Params(Ra=1.0e8, H=5.0, eta_law=:expTz,
                    eta_b=log(16384.0), eta_c=log(64.0), bc=bc)
        dt = 8.0e-11
        res = run_semquad_simulation(qt, pr, theta0;
            p=2, tstop=5 * dt, dt0=dt, dtmax=dt, dtmin=1.0e-16,
            amr_every=0, ev=0.11, ev_cmax=0.052, advection=:skew,
            linear=:direct, maxpic=1, pic_tol=1.0e-6, verbose=false)
        return semquad_vrms(res.ss.su, res.qt, res.ux, res.uz)
    end
    vf = run_bc(:free)
    vr = run_bc(:rigid)
    @test isfinite(vf) && isfinite(vr)
    @test vf > 0.0
    @test vr < vf                                    # rigid walls slow the flow
    @test abs(vr / vf - 1.0) > 1.0e-3                # not just round-off
end
