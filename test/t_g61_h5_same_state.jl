# G61: developed-state same-state one-step AMR vs static at H=5.
#
# The static n=16 (level-4) run is advanced to t=0.05; both codes then take one
# BDF1 step from the same theta with dt=1e-4.  The AMR tree is a nested
# refinement of level 4 (central leaves at level 5), so the static theta is
# representable exactly on the AMR nodes.  This tests the developed flow at
# t=0.05, unlike G60's short smooth transient.
using Test
using LinearAlgebra
using Printf
using QuaST

function uniform_tree(level::Int)
    qt = Quadtree(level + 2)
    for _ in 1:level
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
    end
    return qt
end

function eval_static(m, fd, theta, x, y)
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

@testset "G61 H=5 developed same-state one step" begin
    m = make_mesh(1.0, 16, 16, 4; zgrading=:uniform)
    fd = make_fedata(4, m.q)
    xs, zs = node_coords(m, fd.gll_v)
    theta0 = zeros(Float64, m.nvx * m.nvz)
    for iz in eachindex(zs), ix in eachindex(xs)
        theta0[(iz - 1) * m.nvx + ix] = 0.01 * sinpi(xs[ix]) * sinpi(zs[iz])
    end
    pr = Params(Ra=1.0e4, H=5.0, eta_law=:expT, deta_T=1.0, bc=:free)

    res = run_simulation(m, fd, pr; theta0=theta0, tstop=0.05,
                         dt0=1.0e-4, dtmax=1.0e-3, cfl=1.0, maxsteps=5000,
                         maxpic=2, pic_tol=1.0e-6, pic_relax=0.7,
                         report_every=1, verbose=false, linear=:direct,
                         advection=:skew, velocity_space=:taylor_hood)
    th0 = copy(res.theta)
    res1 = run_simulation(m, fd, pr; theta0=th0, tstop=1.0, dt0=1.0e-4,
                          dtmax=1.0e-4, cfl=0.0, maxsteps=1, maxpic=1,
                          verbose=false, linear=:direct, advection=:skew,
                          velocity_space=:taylor_hood)
    th1 = copy(res1.theta)

    qt = uniform_tree(4)
    for lf in collect(qt.leaves)
        h = 2.0^(-lf[1])
        xc = (lf[2] + 0.5) * h
        yc = (lf[3] + 0.5) * h
        if 0.3 < xc < 0.7 && 0.3 < yc < 0.7
            refine!(qt, lf)
        end
    end
    balance!(qt)
    ss = make_semstokes(qt, 4)
    sq = ss.su
    th0a = [eval_static(m, fd, th0, sq.coords[i][1], sq.coords[i][2]) for i in 1:sq.N]
    res2 = run_semquad_simulation(qt, pr, th0a; p=4, tstop=1.0, dt0=1.0e-4,
                                  dtmax=1.0e-4, cfl=0.0, maxsteps=1, maxpic=1,
                                  amr_every=0, linear=:direct, advection=:skew,
                                  verbose=false)
    th1s = [eval_static(m, fd, th1, sq.coords[i][1], sq.coords[i][2]) for i in 1:sq.N]
    err = norm(res2.theta - th1s) / norm(th1s)
    @printf("G61 err=%.6e N_amr=%d N_static=%d Nu_amr=%.6f/%.6f Nu_static=%.6f/%.6f\n",
            err, sq.N, m.Nscalar, res2.history[end][3], res2.history[end][4],
            res1.Nu_top, res1.Nu_bot)

    @test err < 1.0e-4
    @test abs(res2.history[end][3] - res1.Nu_top) < 1.0e-6
    @test abs(res2.history[end][4] - res1.Nu_bot) < 1.0e-6
    @test sq.N > m.Nscalar
    @test max_face_level_jump(qt) <= 1
end
