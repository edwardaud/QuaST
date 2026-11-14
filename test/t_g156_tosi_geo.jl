# G156: the Tosi geometric Jacobian — d eta/d eps_II and d eta/dT against central differences.
using Test, QuaST, LinearAlgebra

# G156: strain-rate geometric Jacobian (ASPECT A') for the Tosi rheology.
#
# The geometric operator `Kgeo = sum_q w (deta/dedot)/edot T(q) T(q)'` must
# reproduce a central difference of the *production* nonlinear viscous residual
#   R(u) = K(eta(eps(u))) u
# as `(R(u+eps v)-R(u-eps v))/(2eps) = K v + Kgeo v`.
# The Tosi law has no pressure dependence, so d eta/dp == 0; the derivative set
# here is complete for :tosi.  (The Drucker-Prager plastic branch would need a
# pressure column, which is deliberately out of scope.)

@testset "G156 Tosi geometric Jacobian" begin
    qt = Quadtree(2; cells_x=2, cells_y=2)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    N = sq.N
    tosi = TosiLaw(etaT=1.0e5, etaZ=10.0, eta_star=1.0e-3, sigma_y=4.0,
                   eta_init=0.1, eta0=1.0, eta_min=1.0e-12,
                   eta_max=1.0e12, z_is_depth=true)
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:tosi, tosi=tosi, bc=:free)

    # d eta / d eps_II and d eta / dT against central differences
    he = 1.0e-6
    for (T, e, y) in ((0.2, 0.1, 0.5), (0.5, 0.5, 0.3), (0.8, 1.0, 0.7))
        fd_e = (semquad_eta_tosi(T, e + he, y, tosi)[1] -
                semquad_eta_tosi(T, e - he, y, tosi)[1]) / (2he)
        fd_T = (semquad_eta_tosi(T + he, e, y, tosi)[1] -
                semquad_eta_tosi(T - he, e, y, tosi)[1]) / (2he)
        de, dT = semquad_deta_tosi(T, e, y, tosi)
        @test de ≈ fd_e rtol = 1.0e-5
        @test dT ≈ fd_T rtol = 1.0e-5
    end

    ux = [sin(pi * x) * cos(pi * y) for (x, y) in sq.coords]
    uz = [-cos(pi * x) * sin(pi * y) for (x, y) in sq.coords]
    for (i, (x, y)) in enumerate(sq.coords)
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (ux[i] = 0.0; uz[i] = 0.0)
    end
    u = vcat(ux, uz)
    leaves = Set(qt.leaves)
    z2 = (x, y) -> 0.0
    eta_fun = (x, y) -> begin
        ed = semquad_strain_rate_at(sq, ux, uz, x, y, leaves)
        semquad_eta_tosi(Tbar(y, pr.H), ed, y, tosi)[1]
    end
    deta_fun = (x, y, ed) -> semquad_deta_tosi(Tbar(y, pr.H), ed, y, tosi)[1]

    sys0 = solve_semquad_stokes_var(ss, qt, eta_fun, z2, z2, z2, z2;
                                    bc=:free, linear=:direct,
                                    assemble_only=true)
    @test sys0.Kgeo === nothing

    sys = solve_semquad_stokes_var(ss, qt, eta_fun, z2, z2, z2, z2;
                                   bc=:free, linear=:direct,
                                   assemble_only=true, geo_u=u,
                                   geo_coef=deta_fun)
    @test sys.Kgeo !== nothing
    @test norm(sys.Kgeo - sys.Kgeo') / norm(sys.Kgeo) < 1.0e-12
    @test norm(sys.K - sys0.K) / norm(sys0.K) < 1.0e-14
    @test norm(sys.Kred - sys.Cu' * (sys.K + sys.Kgeo) * sys.Cu) /
          norm(sys.Kred) < 1.0e-12
    @test 0.02 < norm(sys.Kgeo) / norm(sys.K) < 1.0

    v = vcat([sin(pi * x) * sin(2pi * y) for (x, y) in sq.coords],
             [sin(2pi * x) * sin(pi * y) for (x, y) in sq.coords])
    for (i, (x, y)) in enumerate(sq.coords)
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (v[i] = 0.0; v[N + i] = 0.0)
    end
    v ./= norm(v)
    eps = 1.0e-6
    up = u .+ eps .* v
    um = u .- eps .* v
    Kp = solve_semquad_stokes_var(ss, qt,
                                  (x, y) -> begin
                                      ed = semquad_strain_rate_at(
                                          sq, up[1:N], up[(N + 1):(2N)], x, y,
                                          leaves)
                                      semquad_eta_tosi(Tbar(y, pr.H), ed, y,
                                                       tosi)[1]
                                  end,
                                  z2, z2, z2, z2; bc=:free, linear=:direct,
                                  assemble_only=true).K
    Km = solve_semquad_stokes_var(ss, qt,
                                  (x, y) -> begin
                                      ed = semquad_strain_rate_at(
                                          sq, um[1:N], um[(N + 1):(2N)], x, y,
                                          leaves)
                                      semquad_eta_tosi(Tbar(y, pr.H), ed, y,
                                                       tosi)[1]
                                  end,
                                  z2, z2, z2, z2; bc=:free, linear=:direct,
                                  assemble_only=true).K
    fd = (Kp * up - Km * um) ./ (2eps)
    lin = sys.K * v + sys.Kgeo * v
    @test norm(fd - lin) / norm(lin) < 1.0e-6
end