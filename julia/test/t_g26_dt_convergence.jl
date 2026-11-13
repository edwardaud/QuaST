# dt convergence of the implicit BDF2 energy step: with a tiny Ra the
# perturbation decays as exp(-2 pi^2 t) and the dt error must be second order.
using Test
using LinearAlgebra
using MantleSEM

@testset "G26 dt convergence" begin
    m = make_mesh(1.0, 4, 4, 2)
    fd = make_fedata(2, m.q)
    xs, zs = node_coords(m, fd.gll_v)
    th0 = [sinpi(xs[mod(k - 1, m.nvx) + 1]) * sinpi(zs[div(k - 1, m.nvx) + 1])
           for k in 1:m.Nscalar]
    pr = Params(Ra=1.0, eta_law=:const, bc=:free, supg=0.0)
    tstop = 0.05
    sols = Vector{Vector{Float64}}()
    for dt in (2.0e-3, 1.0e-3, 5.0e-4)
        res = run_simulation(m, fd, pr; theta0=th0, tstop=tstop, dt0=dt, dtmax=dt,
                             cfl=1.0e9, maxsteps=400, verbose=false)
        push!(sols, copy(res.theta))
    end
    # isolate the dt error: the difference between two dt solutions is O(dt^2),
    # while the (identical) spatial discretisation cancels
    d12 = norm(sols[1] - sols[2]) / norm(sols[2])
    d23 = norm(sols[2] - sols[3]) / norm(sols[3])
    @test log2(d12 / d23) > 1.5
end
