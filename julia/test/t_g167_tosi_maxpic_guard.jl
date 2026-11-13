using Test, MantleSEM, LinearAlgebra

# G167: the `maxpic > 1` guard for the Tosi (strain-rate dependent) rheology.
#
# With strain-rate weakening, iterating the Picard loop inside one step grows
# `u` until the CFL bound computed at iteration 1 is violated and the energy
# solve diverges (measured, even with pic_u_relax = 0.3).  The lagged scheme
# (`maxpic = 1`) reproduces the reference trajectory to 5-8%, so the driver
# rejects `maxpic > 1` for `eta_law=:tosi` instead of diverging silently.
#
# The guard had no dedicated gate: this one pins the error, the message, and
# the fact that `maxpic = 1` still runs (and that non-Tosi laws accept
# `maxpic > 1`, which G142 exercises in depth).

@testset "G167 Tosi maxpic guard" begin
    tosi = TosiLaw(etaT=1.0e5, etaZ=1.0, eta_star=0.0, sigma_y=0.0,
                   eta_init=0.1, eta0=1.0, eta_min=0.1, eta_max=1.0,
                   z_is_depth=false)
    qt = Quadtree(3; cells_x=2, cells_y=2)
    for _ in 1:1, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    th0 = [0.01 * cospi(x) * sinpi(y) for (x, y) in sq.coords]
    for (i, (x, y)) in enumerate(sq.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th0[i] = 0.0)
    end

    pr = Params(Ra=100.0, H=0.0, eta_law=:tosi, tosi=tosi, bc=:free)
    @test_throws ArgumentError run_semquad_simulation(
        qt, pr, th0; p=2, tstop=1.0e-4, dt0=1.0e-5, maxsteps=10, maxpic=2,
        verbose=false)
    # the message must name the lagged-viscosity requirement
    err = try
        run_semquad_simulation(qt, pr, th0; p=2, tstop=1.0e-4, dt0=1.0e-5,
                               maxsteps=10, maxpic=3, verbose=false)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("maxpic=1", sprint(showerror, err))

    # maxpic = 1 runs
    res = run_semquad_simulation(qt, pr, th0; p=2, tstop=1.0e-4, dt0=1.0e-5,
                                 maxsteps=3, maxpic=1, verbose=false)
    @test res.steps == 3
    @test all(isfinite, res.theta)
    @test all(isfinite, res.ux)
    @test all(isfinite, res.uz)

    # a temperature-dependent (non-Tosi) law accepts maxpic > 1
    pr2 = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
                 Twall_top=0.0, Twall_bot=1.0, bc=:free)
    res2 = run_semquad_simulation(qt, pr2, th0; p=2, tstop=1.0e-4, dt0=1.0e-5,
                                  maxsteps=2, maxpic=3, verbose=false)
    @test res2.steps == 2
    @test all(isfinite, res2.theta)
end
