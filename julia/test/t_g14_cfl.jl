# CFL length basis: GLL node spacing (default) vs element diameter.
#
# The element-diameter convention is ASPECT's `CFL number`; the node-spacing
# default keeps the nodal Courant number at `cfl`.  The high-Ra runs use the
# element basis to make long time integrations feasible, so the two lengths must
# be well defined and the option must be accepted by run_simulation.
using Test
using MantleSEM

@testset "G14 CFL length basis" begin
    m = make_mesh(1.0, 8, 8, 4; zgrading=:tanh, zbeta=3.0)
    fd = make_fedata(4, m.q)
    xs, zs = node_coords(m, fd.gll_v)
    hn = min(minimum(diff(xs)), minimum(diff(zs)))
    he = minimum(sqrt(m.hx^2 + z_size(m, ez)^2) for ez in 0:m.nze-1)

    @test cfl_length(m, fd, :node) ≈ hn rtol=1e-14
    @test cfl_length(m, fd, :element) ≈ he rtol=1e-14
    @test he > hn               # p4 nodes are ~0.17 of the element width
    @test_throws ArgumentError cfl_length(m, fd, :bogus)

    # the option is plumbed through run_simulation
    pr = Params(Ra=1e4, eta_const=1.0)
    th = [0.01 * sinpi(xs[mod(k-1,m.nvx)+1]) * sinpi(zs[div(k-1,m.nvx)+1])
          for k in 1:m.Nscalar]
    r = run_simulation(m, fd, pr; theta0=th, tstop=1.0e-9, dt0=1.0e-10,
                       cfl=2.0, cfl_basis=:element, maxsteps=1, maxpic=1,
                       verbose=false)
    @test r.steps == 1
    @test r.t > 0.0
end
