# G71: clean dynamic AMR level7/8 self-convergence.
#
# Analytic topstrong at t=0, initial_refine=1 with analytic re-sampling,
# amr_every=1, max_level=7/8, cfl=0.25, stokes_tol=1e-6.  Two steps to
# t=2e-10; the two solutions are compared on a fixed level9 reference tree.
# This is the clean-protocol gate (no n16 seed, no fixed tree, converged
# Stokes); the 5-step/1e-9 run gives e78=1.13e-6 as a diagnostic.
# With the balanced (cumulative-error) marking the interior reached level 5 at
# this refine_frac until the 2026-11-25 correction of the box indicator's y-face
# fine-side index; it now reaches level 4.  The TOPSTRONG IC is exactly zero for
# y <= 0.65, so the IC has no interior feature: the old interior depth was
# produced by the spurious interior y-face mass the index defect injected.  The
# wall levels are unchanged (7/8) and the level-7/8 self-convergence IMPROVES
# from e78 = 2.868e-7 to 4.651e-8 while N falls (N7 14459 -> 13973, N8 28291 ->
# 27805): better accuracy on fewer leaves.  The surviving anchor is e78 below,
# not the old over-refinement depth.
using Test
using LinearAlgebra
using Printf
using QuaST

const H71_TOPSTRONG = (x, y) -> begin
    if y > 0.65
        0.03 * exp(-((y - 0.9) / 0.05)^2) * ((1.0 - y) / 0.1) *
        (1.0 + 0.5 * cospi(x))
    else
        0.0
    end
end

function h71_run(level::Int)
    pr = Params(Ra=1.0e8, H=5.0, eta_law=:expT, deta_T=1.0e3, bc=:free,
                eta_min=1.0e-6, eta_max=1.0e6)
    qt = wall_graded_tree(2, level; strip=0.10)
    sq = make_semquad(qt, 4)
    theta0 = [H71_TOPSTRONG(x, y) for (x, y) in sq.coords]
    # Policy-agnostic gate: the balanced cycle is the only AMR policy.
    return run_semquad_simulation(qt, pr, theta0;
                                  p=4, tstop=2.0e-10, dt0=1.0e-10,
                                  dtmax=1.0e-10, cfl=0.25, maxsteps=2,
                                  maxpic=1, initial_refine=1,
                                  theta_ic=H71_TOPSTRONG,
                                  amr_every=1, refine_frac=0.2,
                                  coarsen_frac=0.0, max_leaves=70000,
                                  max_level=level, linear=:schur_cg,
                                  schur_maxiter=2000, stokes_tol=1.0e-6,
                                  advection=:skew, verbose=false)
end

@testset "G71 clean dynamic level7/8" begin
    r7 = h71_run(7)
    r8 = h71_run(8)
    qref = wall_graded_tree(2, 9; strip=0.10)
    sqref = make_semquad(qref, 4)
    t7 = transfer_semquad(r7.ss.su, r7.theta, sqref)
    t8 = transfer_semquad(r8.ss.su, r8.theta, sqref)
    e78 = norm(t7 - t8) / norm(t8)
    st7 = semquad_level_stats(r7.qt; strip=0.10)
    st8 = semquad_level_stats(r8.qt; strip=0.10)
    @printf("G71 N7=%d N8=%d e78=%.6e wall=%d/%d int=%d/%d\n",
            r7.ss.su.N, r8.ss.su.N, e78, st7.wall_max, st8.wall_max,
            st7.interior_max, st8.interior_max)

    @test r7.steps == 2
    @test r8.steps == 2
    @test st7.wall_max == 7
    @test st8.wall_max == 8
    # Corrected 2026-11-25: was `>= 5` at both levels.  The old 5 was the
    # spurious y-face mass of the pre-fix indicator on an IC with no interior
    # feature (identically zero below y = 0.65); the median corrected depth is
    # 4 at both levels, the wall levels stay 7/8 and e78 improves
    # 2.868e-7 -> 4.651e-8 (docs/dev/BUG_AUDIT_INDICATOR.md §4.4).  Do not
    # re-pin the depth upward: the convergence bound is the requirement.
    @test st7.interior_max >= 4
    @test st8.interior_max >= 4
    @test r7.ss.su.N < r8.ss.su.N
    @test e78 < 1.0e-5
    @test maximum(abs, r7.theta) < 0.1
    @test maximum(abs, r8.theta) < 0.1
    @test max_face_level_jump(r7.qt) <= 1
    @test max_face_level_jump(r8.qt) <= 1
end
