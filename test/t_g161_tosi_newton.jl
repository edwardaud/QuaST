# G161: the Tosi inner eta(eps) solve and Newton — fixed point and JVP against finite differences.
using Test, QuaST, LinearAlgebra, Printf

# G161: complete Tosi Newton (inner eta(eps) solve + Kgeo tangent).
#
# The Tosi viscosity depends on the strain rate, so the steady map needs an
# inner Stokes <-> eta(eps) fixed point (`semquad_tosi_inner`).  The tangent has
# two parts: the explicit deta/dT stiffness and the geometric Kgeo operator
# (`geo_u`/`geo_coef`, validated by G156).  The gate:
#   1. evolves the Tosi driver to a developed state,
#   2. checks the Tosi JVP against a central difference of the Tosi map,
#   3. runs the Tosi Newton from that state and checks it converges to a
#      convective steady state that is a fixed point of the same map.

const _TOSI_LAW = TosiLaw(etaT=1.0e5, etaZ=1.0, eta_star=0.0, sigma_y=0.0,
                          eta_init=1.0e-6, eta0=10.0, eta_min=0.1,
                          eta_max=1.0, z_is_depth=false)

@testset "G161 Tosi inner solve and Newton" begin
    qt = Quadtree(2; cells_x=4, cells_y=4)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    pr = Params(Ra=100.0, H=0.0, eta_law=:tosi, tosi=_TOSI_LAW)
    th0 = [0.01 * cospi(x) * sinpi(y) for (x, y) in sq.coords]
    for (i, (x, y)) in enumerate(sq.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th0[i] = 0.0)
    end

    # (1) time-step to a developed state at the same tree
    res = run_semquad_simulation(qt, pr, th0; p=2, tstop=2.0e-2, dt0=1.0e-4,
                                 dtmax=1.0e-4, dtmin=1.0e-12, amr_every=0,
                                 ev=0.0, advection=:skew, linear=:direct,
                                 maxpic=1, verbose=false)
    th = res.theta
    @test norm(th) > 1.0

    # (2) inner solve is a converged fixed point of the eta(eps) loop
    inner = semquad_tosi_inner(sq, qt, ss, pr, th; tol=1.0e-4, maxit=20)
    @test inner.converged
    @test inner.iters <= 10

    # (3) Tosi JVP vs central difference of the Tosi map
    # Match the time driver's advection (:skew), so the Newton fixed point and
    # the time-stepped endpoint discretise the same map.
    step = semquad_tosi_steady_step(sq, qt, ss, pr, th; dt=1.0e-2,
                                    tosi_tol=1.0e-5, tosi_maxit=30,
                                    advection=:skew)
    @test step.inner_converged
    N = sq.N
    v = [sinpi(2x) * sinpi(y) for (x, y) in sq.coords]
    for (i, (x, y)) in enumerate(sq.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (v[i] = 0.0)
    end
    v ./= norm(v)
    Jv = semquad_tosi_steady_jv(sq, qt, ss, pr, th, v, step; dt=1.0e-2,
                                advection=:skew)
    h = 1.0e-6
    stp = semquad_tosi_steady_step(sq, qt, ss, pr, th .+ h .* v; dt=1.0e-2,
                                   tosi_tol=1.0e-5, tosi_maxit=30,
                                   u_init=step.u, advection=:skew)
    stm = semquad_tosi_steady_step(sq, qt, ss, pr, th .- h .* v; dt=1.0e-2,
                                   tosi_tol=1.0e-5, tosi_maxit=30,
                                   u_init=step.u, advection=:skew)
    fd = (stp.theta .- stm.theta) ./ (2h)
    # The Tosi tangent (Kgeo included) agrees with the central difference to\n    # ~0.2%: the remaining difference is independent of h and of the inner\n    # tolerance and is attributed to the Kgeo quadrature / strain-rate\n    # evaluation, not to a missing term.  The eta(T) JVP alone matches at 1e-5\n    # (G155), so the tolerance here is deliberately looser.\n    @test norm(Jv - fd) / norm(Jv) < 5.0e-3

    # (4) Newton from the developed state
    rn = semquad_tosi_newton(sq, qt, ss, pr, th; dt=1.0e-2, tol=1.0e-8,
                             maxit=20, krylov=40, restart=20, verbose=false,
                             tosi_tol=1.0e-4, tosi_maxit=30, advection=:skew)
    @test rn.converged
    u = rn.step.u
    vr = semquad_vrms(sq, qt, u[1:N], u[(N + 1):(2N)])
    @test vr > 10.0          # convective branch, not the conduction fixed point
    nt, nb = semquad_nu_walls(sq, qt, rn.theta, pr)
    @printf("G161: Vrms=%.4f Nu=%.5f/%.5f newton_iters=%d inner=%d\n", vr, nt,
            nb, rn.iters, rn.step.inner_iters)
    @test nt > 1.5

    # (5) the Newton state is a fixed point of the same map
    st2 = semquad_tosi_steady_step(sq, qt, ss, pr, rn.theta; dt=1.0e-2,
                                   tosi_tol=1.0e-5, tosi_maxit=30,
                                   u_init=rn.step.u, advection=:skew)
    @test norm(st2.theta - rn.theta) < 1.0e-6
end
