# G77: same-state one-step L8 vs L9 at 5e-8 (isolates space from path/CFL).
#
# The L8/L9 long runs differ in initial tree, AMR path and wall CFL, so their
# 3.36% e89 is not a clean spatial error.  Here the L9 state is transferred to
# an L8 tree that is the same tree with wall9 coarsened to wall8; both take one
# CFL-safe step (dt=2e-10) from the same theta.
using Test
using LinearAlgebra
using Printf
using QuaST

const H77_TOP = (x, y) -> begin
    if y > 0.65
        0.03 * exp(-((y - 0.9) / 0.05)^2) * ((1.0 - y) / 0.1) *
        (1.0 + 0.5 * cospi(x))
    else
        0.0
    end
end

@testset "G77 same-state one-step" begin
    pr = Params(Ra=1.0e8, H=5.0, eta_law=:expT, deta_T=1.0e3, bc=:free,
                eta_min=1.0e-6, eta_max=1.0e6)
    qt9 = wall_graded_tree(2, 9; strip=0.005)
    sq9 = make_semquad(qt9, 4)
    th0 = [H77_TOP(x, y) for (x, y) in sq9.coords]
    # Policy-agnostic gate calibrated on classic; pin the policy explicitly.
    r9 = run_semquad_simulation(qt9, pr, th0;
                                p=4, tstop=5.0e-8, dt0=5.0e-9,
                                dtmax=5.0e-9, cfl=0.25, maxsteps=10,
                                maxpic=1, initial_refine=1, theta_ic=H77_TOP,
                                amr_every=1, marking=:error_fraction,
                                amr_policy=:classic,
                                refine_frac=0.2, coarsen_frac=0.1,
                                max_leaves=70000, max_level=9,
                                wall_min_level=9, wall_strip=0.005,
                                linear=:schur_cg, schur_maxiter=2000,
                                stokes_tol=1.0e-6, advection=:skew,
                                verbose=false)

    qt8 = deepcopy(r9.qt)
    for lf in collect(qt8.leaves)
        lf[1] == 9 || continue
        h = 2.0^(-lf[1])
        zc = (lf[3] + 0.5) * h
        (lf[3] == 0 || lf[3] == 2^lf[1] - 1 || zc <= 0.01 || zc >= 0.99) || continue
        coarsen!(qt8, lf)
    end
    balance!(qt8)
    sq8 = make_semquad(qt8, 4)
    th8 = transfer_semquad(r9.ss.su, r9.theta, sq8)

    function one_step(qt, th, level::Int)
        run_semquad_simulation(qt, pr, th;
                               p=4, tstop=1.0, dt0=2.0e-10,
                               dtmax=2.0e-10, cfl=0.0, maxsteps=1,
                               maxpic=1, amr_every=0, initial_refine=0,
                               max_level=level, wall_min_level=0,
                               linear=:schur_cg, schur_maxiter=2000,
                               stokes_tol=1.0e-6, advection=:skew,
                               verbose=false)
    end

    s9 = one_step(deepcopy(r9.qt), copy(r9.theta), 9)
    s8 = one_step(qt8, copy(th8), 8)
    t8 = transfer_semquad(s8.ss.su, s8.theta, s9.ss.su)
    e89 = norm(t8 - s9.theta) / norm(s9.theta)
    @printf("G77 N9=%d N8=%d e89=%.6e Vrms=%.3e/%.3e\n",
            r9.ss.su.N, sq8.N, e89, s9.history[end][5], s8.history[end][5])

    @test e89 < 1.0e-5
    @test abs(s9.history[end][5] - s8.history[end][5]) /
          s9.history[end][5] < 1.0e-3
    @test maximum(abs, s9.theta) < 0.1
end
