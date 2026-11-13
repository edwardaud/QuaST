# G44: matrix-free Schur-CG Stokes on the quadtree.
#
# The same exact variable-viscosity polynomial as G33, solved with
# linear=:schur_cg, must match the direct saddle solve; this is the solver mode
# needed for the large wall-graded AMR meshes.
using Test
using LinearAlgebra
using QuaST

@testset "G44 quadtree Stokes Schur-CG" begin
    qt = Quadtree(6)
    refine!(qt, (0, 0, 0))
    refine!(qt, (1, 0, 0))
    @test balance!(qt)
    ss = make_semstokes(qt, 2)
    nu = ss.su.N
    uex(x, y) = 2x * y
    uez(x, y) = -y^2
    eta1(x, y) = 1.0 + 0.1x
    fx(x, y) = -0.4y
    fz(x, y) = 2.0

    u_d, p_d, _, _, _ = solve_semquad_stokes_var(ss, qt, eta1, fx, fz,
                                                 uex, uez; linear=:direct)
    u_s, p_s, _, _, _ = solve_semquad_stokes_var(ss, qt, eta1, fx, fz,
                                                 uex, uez; linear=:schur_cg)
    @test maximum(abs.(u_s - u_d)) < 1.0e-8
    @test maximum(abs.(p_s - p_d)) < 1.0e-8

    err = 0.0
    for g in 1:nu
        x, y = ss.su.coords[g]
        err = max(err, abs(u_s[g] - uex(x, y)), abs(u_s[nu + g] - uez(x, y)))
    end
    @test err < 1.0e-8
    @test maximum(abs, p_s) < 1.0e-8
end
