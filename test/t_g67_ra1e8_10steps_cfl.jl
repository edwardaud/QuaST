# G67: Ra=1e8 CFL-safe 10-step window.
#
# Starting from the 5-step strong state (t=5e-8), run 10 steps with
# dt=4e-11 (wall level9 CFL~0.25).  The physical advance is only 4e-10, but
# this is the CFL-safe 10-step consistency gate; the fixed dt=1e-9 run is the
# invalid CFL~25 case.
using Test
using LinearAlgebra
using Printf
using QuaST

function h67_eval_static(m, fd, theta, x, y)
    p = m.p
    hx = m.hx
    hz = z_size(m, 0)
    ex = min(floor(Int, x / hx), m.nxe - 1)
    ez = min(floor(Int, y / hz), m.nze - 1)
    x0 = ex * hx
    z0 = z_left(m, ez)
    xi = 2.0 * (x - x0) / hx - 1.0
    eta = 2.0 * (y - z0) / hz - 1.0
    vx, _ = QuaST.lagrange_eval(fd.gll_v, xi)
    vy, _ = QuaST.lagrange_eval(fd.gll_v, eta)
    g = element_vel_gids(m, ex, ez)
    s = 0.0
    for j in 1:p+1, i in 1:p+1
        s += theta[g[(j - 1) * (p + 1) + i]] * vx[i] * vy[j]
    end
    return s
end

function h67_interior_refine!(qt, theta, frac, maxlevel)
    sq = make_semquad(qt, 4)
    eta = semquad_gradient_jump(sq, qt, theta)
    cand = Tuple{Int,NTuple{3,Int}}[]
    for (i, lf) in enumerate(qt.leaves)
        h = 2.0^(-lf[1])
        zc = (lf[3] + 0.5) * h
        if 0.10 < zc < 0.90 && lf[1] < maxlevel
            push!(cand, (i, lf))
        end
    end
    sort!(cand; by=t -> -eta[t[2]])
    n = min(length(cand), round(Int, frac * length(cand)))
    for k in 1:n
        refine!(qt, cand[k][2])
    end
    balance!(qt)
    return qt
end

@testset "G67 Ra=1e8 CFL-safe 10 steps" begin
    m = make_mesh(1.0, 16, 16, 4; zgrading=:uniform)
    fd = make_fedata(4, m.q)
    xs, zs = node_coords(m, fd.gll_v)
    theta0 = zeros(Float64, m.nvx * m.nvz)
    for iz in eachindex(zs), ix in eachindex(xs)
        g = (iz - 1) * m.nvx + ix
        x = xs[ix]; z = zs[iz]
        if z > 0.65
            theta0[g] = 0.03 * exp(-((z - 0.9) / 0.05)^2) * ((1.0 - z) / 0.1) *
                        (1.0 + 0.5 * cospi(x))
        end
    end
    pr = Params(Ra=1.0e8, H=5.0, eta_law=:expT, deta_T=1.0e3, bc=:free,
                eta_min=1.0e-6, eta_max=1.0e6)
    res = run_simulation(m, fd, pr; theta0=theta0, tstop=5.0e-8,
                         dt0=1.0e-8, dtmax=1.0e-8, dtmin=1.0e-15, cfl=20.0,
                         maxsteps=5, maxpic=1, steady_tol=0.0,
                         verbose=false, linear=:direct, advection=:skew,
                         velocity_space=:taylor_hood)
    th0 = copy(res.theta)
    res10 = run_simulation(m, fd, pr; theta0=th0, tstop=6.0e-8, dt0=4.0e-11,
                           dtmax=4.0e-11, cfl=20.0, maxsteps=10, maxpic=1,
                           steady_tol=0.0, verbose=false, linear=:direct,
                           advection=:skew, velocity_space=:taylor_hood)
    th10 = copy(res10.theta)

    function run_amr(level::Int)
        qt = wall_graded_tree(2, level; strip=0.10)
        sq = make_semquad(qt, 4)
        tha = [h67_eval_static(m, fd, th0, sq.coords[i][1], sq.coords[i][2])
               for i in 1:sq.N]
        for _ in 1:2
            h67_interior_refine!(qt, tha, 0.30, 6)
            sq = make_semquad(qt, 4)
            tha = [h67_eval_static(m, fd, th0, sq.coords[i][1], sq.coords[i][2])
                   for i in 1:sq.N]
        end
        resa = run_semquad_simulation(qt, pr, tha; p=4, tstop=1.0,
                                      dt0=4.0e-11, dtmax=4.0e-11, cfl=20.0,
                                      maxsteps=10, maxpic=1, amr_every=0,
                                      linear=:schur_cg, schur_maxiter=1000, stokes_tol=1.0e-6,
                                      advection=:skew, verbose=false)
        st = semquad_level_stats(qt; strip=0.10)
        return resa, st, qt
    end

    r8, st8, qt8 = run_amr(8)
    r9, st9, qt9 = run_amr(9)
    e89 = norm(transfer_semquad(r8.ss.su, r8.theta, r9.ss.su) - r9.theta) /
          norm(r9.theta)
    th10_on9 = [h67_eval_static(m, fd, th10, r9.ss.su.coords[i][1],
                                r9.ss.su.coords[i][2]) for i in 1:r9.ss.su.N]
    e19 = norm(th10_on9 - r9.theta) / norm(r9.theta)
    @printf("G67 N8=%d N9=%d e89=%.6e e19=%.6e thmax=%.3e Vrms=%.3e\n",
            r8.ss.su.N, r9.ss.su.N, e89, e19, maximum(abs, r9.theta),
            r9.history[end][5])

    @test st8.wall_max == 8
    @test st9.wall_max == 9
    @test st8.interior_max >= 5
    @test st9.interior_max >= 5
    @test e89 < 1.0e-5
    @test e19 < 1.0e-4
    @test maximum(abs, r9.theta) < 0.1
    @test isfinite(r9.history[end][5])
    @test max_face_level_jump(qt8) <= 1
    @test max_face_level_jump(qt9) <= 1
end
