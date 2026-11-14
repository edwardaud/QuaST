# G152: scalable Stokes (linear=:mg) on uniform and AMR trees across viscosity contrasts.
using Test, QuaST, LinearAlgebra

# G152: scalable Stokes path (`linear=:mg`) on a uniform quadtree.
#
# The MG hierarchy coarsens a uniform forest by levels and uses the production
# assembler at every level (rediscretised eta), a damped-Jacobi V-cycle and a
# pressure Schur block.  GMRES on the reduced saddle is right-preconditioned by
# a block-triangular operator (MG velocity solve + pressure Schur solve).
#
# Checks:
#   * constant and 100x viscosity: solution agrees with the direct saddle solve
#     to the direct solve's own accuracy, residual <= 1e-9;
#   * a 2b-like 16384x contrast converges (auto-selects the LSC pressure block);
#   * free-slip BC and the exact stretched-box polynomial;
#   * a non-uniform tree is rejected explicitly (hierarchical coarsening with
#     hanging constraints is not implemented yet).

function _g152_tree(level::Int; Lx::Union{Nothing,Float64}=nothing)
    qt = Lx === nothing ? Quadtree(level; cells_x=5, cells_y=2) :
                          Quadtree(level; Lx=Lx)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

@testset "G152 scalable Stokes (MG)" begin
    # ---- constant viscosity, rigid walls ----
    qt = _g152_tree(3)
    ss = make_semstokes(qt, 2)
    Lx = qt.Lx
    fz = (x, y) -> sin(pi * x / Lx) * sin(pi * y)
    z2 = (x, y) -> 0.0
    u0, p0, _, _, _ = solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0, z2, fz,
                                               z2, z2; bc=:rigid, linear=:direct)
    u1, p1, _, _, _ = solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0, z2, fz,
                                               z2, z2; bc=:rigid, linear=:mg)
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-9
    @test norm(u1 - u0) / norm(u0) < 1.0e-8
    @test LAST_SEMSTOKES_SCHUR_ITERS[] < 100

    # ---- 100x smooth viscosity ----
    eta = (x, y) -> exp(-log(100.0) * (1 - y))
    u0, _, _, _, _ = solve_semquad_stokes_var(ss, qt, eta, z2, fz, z2, z2;
                                              bc=:rigid, linear=:direct)
    u1, _, _, _, _ = solve_semquad_stokes_var(ss, qt, eta, z2, fz, z2, z2;
                                              bc=:rigid, linear=:mg)
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-9
    @test norm(u1 - u0) / norm(u0) < 1.0e-6
    @test LAST_SEMSTOKES_SCHUR_ITERS[] < 150

    # ---- 2b-like 16384x contrast: auto LSC block ----
    eta2 = (x, y) -> exp(-log(16384.0) * (1 - y))
    u0, _, _, _, _ = solve_semquad_stokes_var(ss, qt, eta2, z2, fz, z2, z2;
                                              bc=:free, linear=:direct)
    u1, _, _, _, _ = solve_semquad_stokes_var(ss, qt, eta2, z2, fz, z2, z2;
                                              bc=:free, linear=:mg,
                                              schur_maxiter=400)
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-8
    @test norm(u1 - u0) / norm(u0) < 1.0e-5
    @test LAST_SEMSTOKES_SCHUR_ITERS[] < 300

    # ---- free slip on a moderate contrast ----
    etaf = (x, y) -> exp(-2 * (1 - y))
    u0, _, _, _, _ = solve_semquad_stokes_var(ss, qt, etaf, z2, fz, z2, z2;
                                              bc=:free, linear=:direct)
    u1, _, _, _, _ = solve_semquad_stokes_var(ss, qt, etaf, z2, fz, z2, z2;
                                              bc=:free, linear=:mg)
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-9
    @test norm(u1 - u0) / norm(u0) < 1.0e-8
    @test LAST_SEMSTOKES_SCHUR_ITERS[] < 100

    # ---- exact polynomial on a stretched uniform box ----
    qts = _g152_tree(3; Lx=2.5)
    sss = make_semstokes(qts, 2)
    uxex = (x, y) -> 2.0 * x * y
    uzex = (x, y) -> -y^2
    us, _, _, _, _ = solve_semquad_stokes_var(sss, qts, (x, y) -> 1.0, z2,
                                              (x, y) -> 2.0, uxex, uzex;
                                              bc=:rigid, linear=:mg,
                                              mg_cycles=3, mg_tol=1.0e-11)
    ns = sss.su.N
    e = 0.0
    for (g, c) in enumerate(sss.su.coords)
        e = max(e, abs(us[g] - uxex(c...)), abs(us[ns + g] - uzex(c...)))
    end
    # The stretched box is much worse conditioned than the square-leaf
    # production meshes (2.5:1 leaves): with the same solver the polynomial is
    # reproduced to ~2e-8, while every square-leaf case above is <1e-8.
    @test e < 1.0e-7

    # ---- non-uniform tree: hierarchical coarsening path ----
    qtn = Quadtree(3)
    for _ in 1:2, lf in collect(qtn.leaves)
        refine!(qtn, lf)
    end
    refine!(qtn, (2, 0, 0))
    balance!(qtn)
    @test max_face_level_jump(qtn) <= 1
    ssn = make_semstokes(qtn, 2)
    u0n, _, _, _, _ = solve_semquad_stokes_var(ssn, qtn, (x, y) -> 1.0, z2,
                                               fz, z2, z2; bc=:rigid,
                                               linear=:direct)
    u1n, _, _, _, _ = solve_semquad_stokes_var(ssn, qtn, (x, y) -> 1.0, z2,
                                               fz, z2, z2; bc=:rigid,
                                               linear=:mg)
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-9
    @test norm(u1n - u0n) / norm(u0n) < 1.0e-8
end
