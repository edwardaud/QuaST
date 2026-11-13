# G75: same-DOF theta L2 at 1e-8 (static n32 vs AMR L7).
#
# Both have N~16.6k; the static n32 and AMR L7 fields are transferred to a
# common level9 reference tree, and compared with the AMR L8 field.  This is
# the same-DOF efficiency gate for the clean error-fraction/wall-min setup.
using Test
using LinearAlgebra
using Printf
using MantleSEM

const H75_TOP = (x, y) -> begin
    if y > 0.65
        0.03 * exp(-((y - 0.9) / 0.05)^2) * ((1.0 - y) / 0.1) *
        (1.0 + 0.5 * cospi(x))
    else
        0.0
    end
end

function h75_eval_static(m, fd, theta, x, y)
    p = m.p
    hx = m.hx
    hz = z_size(m, 0)
    ex = min(floor(Int, x / hx), m.nxe - 1)
    ez = min(floor(Int, y / hz), m.nze - 1)
    x0 = ex * hx
    z0 = z_left(m, ez)
    xi = 2.0 * (x - x0) / hx - 1.0
    eta = 2.0 * (y - z0) / hz - 1.0
    vx, _ = MantleSEM.lagrange_eval(fd.gll_v, xi)
    vy, _ = MantleSEM.lagrange_eval(fd.gll_v, eta)
    g = element_vel_gids(m, ex, ez)
    s = 0.0
    for j in 1:p+1, i in 1:p+1
        s += theta[g[(j - 1) * (p + 1) + i]] * vx[i] * vy[j]
    end
    return s
end

function h75_clean(level::Int, pr)
    qt = wall_graded_tree(2, level; strip=0.005)
    sq = make_semquad(qt, 4)
    th0 = [H75_TOP(x, y) for (x, y) in sq.coords]
    # Policy-agnostic gate calibrated on classic; pin the policy explicitly.
    return run_semquad_simulation(qt, pr, th0;
                                  p=4, tstop=1.0e-8, dt0=1.0e-9,
                                  dtmax=1.0e-9, cfl=0.25, maxsteps=10,
                                  maxpic=1, initial_refine=1, theta_ic=H75_TOP,
                                  amr_every=1, marking=:error_fraction,
                                  amr_policy=:classic,
                                  refine_frac=0.2, coarsen_frac=0.1,
                                  max_leaves=30000, max_level=level,
                                  wall_min_level=level, wall_strip=0.005,
                                  linear=:schur_cg, schur_maxiter=2000,
                                  stokes_tol=1.0e-6, advection=:skew,
                                  verbose=false)
end

@testset "G75 same-DOF theta L2" begin
    pr = Params(Ra=1.0e8, H=5.0, eta_law=:expT, deta_T=1.0e3, bc=:free,
                eta_min=1.0e-6, eta_max=1.0e6)
    m = make_mesh(1.0, 32, 32, 4; zgrading=:uniform)
    fd = make_fedata(4, m.q)
    xs, zs = node_coords(m, fd.gll_v)
    th0 = zeros(Float64, m.nvx * m.nvz)
    for iz in eachindex(zs), ix in eachindex(xs)
        th0[(iz - 1) * m.nvx + ix] = H75_TOP(xs[ix], zs[iz])
    end
    rs = run_simulation(m, fd, pr; theta0=th0, tstop=1.0e-8,
                        dt0=1.0e-9, dtmax=1.0e-9, cfl=20.0, maxsteps=10,
                        maxpic=1, steady_tol=0.0, linear=:direct,
                        advection=:skew, velocity_space=:taylor_hood,
                        verbose=false)
    r7 = h75_clean(7, pr)
    r8 = h75_clean(8, pr)

    qref = wall_graded_tree(2, 9; strip=0.10)
    sqref = make_semquad(qref, 4)
    t7 = transfer_semquad(r7.ss.su, r7.theta, sqref)
    t8 = transfer_semquad(r8.ss.su, r8.theta, sqref)
    ts = [h75_eval_static(m, fd, rs.theta, x, y) for (x, y) in sqref.coords]
    e7 = norm(t7 - t8) / norm(t8)
    es = norm(ts - t8) / norm(t8)
    @printf("G75 Nstatic=%d N7=%d e_static=%.6e e_amr=%.6e ratio=%.1f\n",
            m.Nscalar, r7.ss.su.N, es, e7, es / e7)

    @test abs(r7.ss.su.N - m.Nscalar) / m.Nscalar < 0.1
    @test e7 < 1.0e-5
    @test e7 < es / 100.0
    @test r7.steps == 10
    @test r8.steps == 10
end
