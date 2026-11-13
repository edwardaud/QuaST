# G64: high-Ra H=5 topstrong feature tracking on a convective state.
#
# Static n=16 (Ra=1e7, H=5, deta_T=1e3, topstrong IC) is advanced to
# t=5e-7; the AMR wall7/wall8 trees then get an interior-gradient pre-refine
# (2 passes, 30%, max level 6) and one BDF1 step from the same theta.
# Gate: both wall and interior plume reach level 5+, and level7 vs level8
# self-convergence stays below 1e-3.
using Test
using LinearAlgebra
using Printf
using MantleSEM

function h64_eval_static(m, fd, theta, x, y)
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

function h64_interior_refine!(qt, theta, frac, maxlevel)
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

@testset "G64 high-Ra H=5 plume tracking" begin
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
    pr = Params(Ra=1.0e7, H=5.0, eta_law=:expT, deta_T=1.0e3, bc=:free)
    res = run_simulation(m, fd, pr; theta0=theta0, tstop=5.0e-7,
                         dt0=1.0e-9, dtmax=1.0e-7, dtmin=1.0e-15, cfl=20.0,
                         maxsteps=5000, maxpic=1, steady_tol=0.0,
                         verbose=false, linear=:direct, advection=:skew,
                         velocity_space=:taylor_hood)
    th0 = copy(res.theta)

    function run_amr_plume(level::Int)
        qt = wall_graded_tree(2, level; strip=0.10)
        sq = make_semquad(qt, 4)
        tha = [h64_eval_static(m, fd, th0, sq.coords[i][1], sq.coords[i][2])
               for i in 1:sq.N]
        for _ in 1:2
            h64_interior_refine!(qt, tha, 0.30, 6)
            sq = make_semquad(qt, 4)
            tha = [h64_eval_static(m, fd, th0, sq.coords[i][1], sq.coords[i][2])
                   for i in 1:sq.N]
        end
        resa = run_semquad_simulation(qt, pr, tha; p=4, tstop=1.0,
                                      dt0=1.0e-8, dtmax=1.0e-8, cfl=20.0,
                                      maxsteps=1, maxpic=1, amr_every=0,
                                      linear=:schur_cg, schur_maxiter=1000, stokes_tol=1.0e-6,
                                      advection=:skew, verbose=false)
        st = semquad_level_stats(qt; strip=0.10)
        return resa, st, qt
    end

    r7, st7, qt7 = run_amr_plume(7)
    r8, st8, qt8 = run_amr_plume(8)
    e78 = norm(transfer_semquad(r7.ss.su, r7.theta, r8.ss.su) - r8.theta) /
          norm(r8.theta)
    @printf("G64 N7=%d N8=%d e78=%.6e wall7=%d int7=%d wall8=%d int8=%d\n",
            r7.ss.su.N, r8.ss.su.N, e78, st7.wall_max, st7.interior_max,
            st8.wall_max, st8.interior_max)

    @test st7.wall_max == 7
    @test st8.wall_max == 8
    @test st7.interior_max >= 5
    @test st8.interior_max >= 5
    @test e78 < 1.0e-3
    @test r8.ss.su.N > r7.ss.su.N
    @test max_face_level_jump(qt7) <= 1
    @test max_face_level_jump(qt8) <= 1
end
