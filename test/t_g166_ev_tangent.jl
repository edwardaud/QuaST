using Test, QuaST, LinearAlgebra, Printf

# G166: entropy-viscosity (EV) tangent.
#
# `semquad_steady_step` recomputes the elementwise EV field from
# `(theta, u)` on every evaluation, so freezing it in the JVP (the historical
# `ev_tangent=false` behaviour) drops a real term: on the state below the
# frozen-EV JVP is **16%** away from a central difference of the map, which is
# why the EV Newton only converged linearly (~0.6 per step).
#
# `semquad_ev_field` is the standalone field (bit-identical to the `ev_out` of
# `assemble_semquad_energy`); `semquad_steady_jv(...; ev_tangent=true)` takes a
# central difference of that field along `(v, du)` and adds
# `-D_ev(d nu) theta` to the energy linearisation.  The gate checks:
#   1. the standalone field equals the assembled one,
#   2. the tangent JVP matches the central difference of the map to <1e-5
#      (the frozen one does not),
#   3. the EV Newton with the tangent needs fewer iterations and lands on the
#      same fixed point.

@testset "G166 EV tangent" begin
    const_B = 9.70406052783923433184
    const_C = 4.1588830833596718565
    qt = Quadtree(5; cells_x=5, cells_y=2)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    N = sq.N
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz, eta_b=const_B, eta_c=const_C,
                Twall_top=0.0, Twall_bot=1.0, bc=:free)
    th = [0.5 * sinpi(x / 2.5) * sinpi(y) for (x, y) in sq.coords]
    for (i, (x, y)) in enumerate(sq.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th[i] = 0.0)
    end

    step = semquad_steady_step(sq, qt, ss, pr, th; dt=1.0e-2, ev=0.11)
    @test step.parts.ev_field !== nothing

    # (1) standalone field == assembled field
    evf = semquad_ev_field(sq, qt, pr, th, step.u[1:N], step.u[(N + 1):(2N)];
                           dt=1.0e-2, ev=0.11, ev_cmax=0.052, ev_alpha=2,
                           ev_timestep=step.ev_timestep)
    @test evf == step.parts.ev_field

    # (2) JVP vs central difference of the map
    v = [sinpi(2x / 2.5) * sinpi(y) for (x, y) in sq.coords]
    for (i, (x, y)) in enumerate(sq.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (v[i] = 0.0)
    end
    v ./= norm(v)
    Jfrozen = semquad_steady_jv(sq, qt, ss, pr, th, v, step; dt=1.0e-2,
                                ev_tangent=false) .- v
    Jtang = semquad_steady_jv(sq, qt, ss, pr, th, v, step; dt=1.0e-2,
                              ev_tangent=true) .- v
    h = 1.0e-5
    sp = semquad_steady_step(sq, qt, ss, pr, th .+ h .* v; dt=1.0e-2, ev=0.11)
    sm = semquad_steady_step(sq, qt, ss, pr, th .- h .* v; dt=1.0e-2, ev=0.11)
    fd = (sp.theta .- sm.theta) ./ (2h) .- v
    @test norm(Jtang - fd) / norm(fd) < 1.0e-5
    @test norm(Jfrozen - fd) / norm(fd) > 1.0e-2   # the dropped term is visible
    @printf("G166: JVP-vs-FD frozen %.3e, tangent %.3e\n",
            norm(Jfrozen - fd) / norm(fd), norm(Jtang - fd) / norm(fd))

    # (3) Newton with the tangent: fewer iterations, same fixed point.
    # Start from the developed ASPECT 2b initial condition (a cold synthetic
    # state sends both variants to different branches, which says nothing about
    # the Jacobian).
    g = read_temperature_ascii(joinpath(@__DIR__, "..", "examples", "data",
                                       "initial_temperature_case2b.txt"))
    th_ic = [interp_temperature(g, x, y) - (1.0 - y) for (x, y) in sq.coords]
    for (i, (x, y)) in enumerate(sq.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th_ic[i] = 0.0)
    end
    r0 = semquad_newton_steady(sq, qt, ss, pr, th_ic; dt=1.0e-2, tol=1.0e-8,
                               maxit=25, krylov=40, restart=20, ev=0.11,
                               ev_tangent=false)
    r1 = semquad_newton_steady(sq, qt, ss, pr, th_ic; dt=1.0e-2, tol=1.0e-8,
                               maxit=25, krylov=40, restart=20, ev=0.11,
                               ev_tangent=true)
    @test r0.converged
    @test r1.converged
    @test r1.iters < r0.iters
    @test norm(r1.theta - r0.theta) / norm(r0.theta) < 1.0e-6
    @printf("G166: Newton frozen %d iters, tangent %d iters\n", r0.iters,
            r1.iters)
end
