# G74: clean dynamic AMR with ASPECT-style error-fraction marking.
#
# Analytic topstrong, initial_refine=1, amr_every=1, marking=:error_fraction
# (refine/coarsen fractions of total error), coarsen_frac=0.1, hard
# max_leaves=30000, max_level=7/8, wall_min_level=7/8, wall_strip=0.01.
# Two steps to t=2e-10; compare on a fixed level9 reference tree.
using Test
using LinearAlgebra
using Printf
using MantleSEM

const H74_TOPSTRONG = (x, y) -> begin
    if y > 0.65
        0.03 * exp(-((y - 0.9) / 0.05)^2) * ((1.0 - y) / 0.1) *
        (1.0 + 0.5 * cospi(x))
    else
        0.0
    end
end

function h74_run(level::Int)
    pr = Params(Ra=1.0e8, H=5.0, eta_law=:expT, deta_T=1.0e3, bc=:free,
                eta_min=1.0e-6, eta_max=1.0e6)
    qt = wall_graded_tree(2, level; strip=0.10)
    sq = make_semquad(qt, 4)
    theta0 = [H74_TOPSTRONG(x, y) for (x, y) in sq.coords]
    # Policy-agnostic gate calibrated on classic; pin the policy explicitly.
    return run_semquad_simulation(qt, pr, theta0;
                                  p=4, tstop=2.0e-10, dt0=1.0e-10,
                                  dtmax=1.0e-10, cfl=0.25, maxsteps=2,
                                  maxpic=1, initial_refine=1,
                                  theta_ic=H74_TOPSTRONG,
                                  amr_every=1, marking=:error_fraction,
                                  amr_policy=:classic,
                                  refine_frac=0.2, coarsen_frac=0.1,
                                  max_leaves=30000, max_level=level,
                                  wall_min_level=level, wall_strip=0.01,
                                  linear=:schur_cg, schur_maxiter=2000,
                                  stokes_tol=1.0e-6, advection=:skew,
                                  verbose=false)
end

@testset "G74 clean error-fraction level7/8" begin
    r7 = h74_run(7)
    r8 = h74_run(8)
    qref = wall_graded_tree(2, 9; strip=0.10)
    sqref = make_semquad(qref, 4)
    t7 = transfer_semquad(r7.ss.su, r7.theta, sqref)
    t8 = transfer_semquad(r8.ss.su, r8.theta, sqref)
    e78 = norm(t7 - t8) / norm(t8)
    st7 = semquad_level_stats(r7.qt; strip=0.10)
    st8 = semquad_level_stats(r8.qt; strip=0.10)
    @printf("G74 N7=%d N8=%d e78=%.6e wall=%d/%d int=%d/%d\n",
            r7.ss.su.N, r8.ss.su.N, e78, st7.wall_max, st8.wall_max,
            st7.interior_max, st8.interior_max)

    @test r7.steps == 2
    @test r8.steps == 2
    @test st7.wall_max == 7
    @test st8.wall_max == 8
    @test st7.interior_max >= 5
    @test st8.interior_max >= 5
    @test st7.interior_max <= 6
    @test st8.interior_max <= 6
    @test r7.ss.su.N < r8.ss.su.N
    @test e78 < 1.0e-5
    @test maximum(abs, r7.theta) < 0.1
    @test maximum(abs, r8.theta) < 0.1
    @test max_face_level_jump(r7.qt) <= 1
    @test max_face_level_jump(r8.qt) <= 1
end
