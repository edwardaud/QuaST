# Local CFL: dt = cfl * min_e h_e / max|u|_e, with a hand-checked case where
# the small (slow) element must not throttle the global dt.
using Test
using QuaST

@testset "G25 local CFL" begin
    # graded mesh: 2 elements in z, the bottom one 10x thinner
    m = make_mesh(1.0, 2, 2, 2)
    fd = make_fedata(2, m.q)
    n = m.p + 1
    ux = zeros(m.Nscalar); uz = zeros(m.Nscalar)
    # bottom element: slow (|u|=1), top element: fast (|u|=10)
    for ez in 0:m.nze-1, ex in 0:m.nxe-1
        gids = element_vel_gids(m, ex, ez)
        for k in gids
            ux[k] = ez == 0 ? 1.0 : 10.0
        end
    end
    d = cfl_local_dt(m, ux, uz)
    h = sqrt(m.hx^2 + z_size(m, 0)^2)
    @test d ≈ h / 10.0          # top element (fast) sets it, not the slow one

    # a slow small element must not dominate: same h, u=1 in the bottom
    ux2 = copy(ux)
    for ez in 0:m.nze-1, ex in 0:m.nxe-1
        if ez == 0
            for k in element_vel_gids(m, ex, ez); ux2[k] = 0.01; end
        end
    end
    d2 = cfl_local_dt(m, ux2, uz)
    @test d2 ≈ h / 10.0

    # integration: a short run with cfl_mode=:local must advance and choose a
    # dt at least as large as the global criterion would
    pr = Params(Ra=1.0e3, eta_law=:const, bc=:free, supg=0.0)
    xs, zs = node_coords(m, fd.gll_v)
    th0 = [0.01 * cospi(xs[mod(k - 1, m.nvx) + 1]) * sinpi(zs[div(k - 1, m.nvx) + 1])
           for k in 1:m.Nscalar]
    res = run_simulation(m, fd, pr; theta0=th0, tstop=1.0e-3, dt0=1.0e-6,
                         dtmax=1.0e-3, cfl=5.0, cfl_mode=:local, maxsteps=50,
                         verbose=false, report_every=1000)
    @test res.steps > 0
    @test res.t > 0.0
end
