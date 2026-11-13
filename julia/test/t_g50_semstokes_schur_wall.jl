# G50: Schur-CG on a wall-refined quadtree with steep viscosity.
#
# The direct solver is reliable at this size (~4k scalar dofs) and provides the
# reference.  The Schur-CG right-hand side must be Bs*u - gr, not gr - Bs*u: the
# wrong sign leaves the pressure solve inconsistent and produces velocities that
# differ by O(1) (the Ra=1e8 smoke's Vrms blow-up).  The pressure-mass
# preconditioner (spectrally equivalent to the Schur complement) reaches
# res < 1e-7 in ~130 iterations and matches direct to < 1e-6 in velocity.
using Test
using LinearAlgebra
using MantleSEM

@testset "G50 Schur-CG vs direct on wall-refined steep eta" begin
    qt = wall_graded_tree(2, 6; strip=0.15)
    ss = make_semstokes(qt, 2)
    sq = ss.su
    theta = [0.01 * cospi(sq.coords[i][1]) * sinpi(sq.coords[i][2]) for i in 1:sq.N]
    pr = Params(Ra=1.0e8, H=5.0, eta_law=:expT, deta_T=1.0e3, bc=:free,
                eta_min=1.0e-6, eta_max=1.0e6)
    leaves = Set(sq.leaves)
    ev = (x, y) -> MantleSEM._semquad_eval_at(sq, theta, x, y, leaves)
    eta_fun = (x, y) -> clamp(exp(-log(pr.deta_T) * (Tbar(y, pr.H) + ev(x, y))),
                              pr.eta_min, pr.eta_max)
    fz = (x, y) -> pr.Ra * ev(x, y)
    zero2 = (x, y) -> 0.0

    ud, pd, _, _, _ = solve_semquad_stokes_var(ss, qt, eta_fun, zero2, fz,
                                               zero2, zero2;
                                               bc=:free, linear=:direct)
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-8

    # The pressure mass matrix must integrate to the domain area; it is the
    # spectrally-equivalent Schur preconditioner.
    Mfull = assemble_semquad_pressure_mass(ss.sp)
    @test sum(Mfull) ≈ 1.0 rtol=1.0e-12

    us, ps, _, _, _ = solve_semquad_stokes_var(ss, qt, eta_fun, zero2, fz,
                                               zero2, zero2;
                                               bc=:free, linear=:schur_cg,
                                               schur_maxiter=300,
                                               schur_precond=:mass)
    @test LAST_SEMSTOKES_SCHUR_ITERS[] < 300
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-5
    @test norm(us - ud) / norm(ud) < 1.0e-5
    @test norm(ps - pd) / norm(pd) < 1.0e-3
end
