# G31: solution transfer between quadtree SEM meshes.
#
# The time-dependent AMR cycle must move theta and velocity across refine and
# coarsen events.  On a 2:1 mixed tree the transfer is the old piecewise-Q_p
# field evaluated at the new GLL nodes; it must be exact for polynomials of
# degree <= p and must satisfy the new hanging-node constraints.
using Test
using LinearAlgebra
using QuaST

@testset "G31 SEM quadtree solution transfer" begin
    lin(x, y) = 1.0 + 0.3x - 0.2y
    quad(x, y) = x^2 - y^2 + 0.5x * y

    qtc = Quadtree(6)
    refine!(qtc, (0, 0, 0))
    sqc = make_semquad(qtc, 2)

    qtm = Quadtree(6)
    refine!(qtm, (0, 0, 0))
    refine!(qtm, (1, 0, 0))
    @test balance!(qtm)
    @test max_face_level_jump(qtm) == 1
    sqm = make_semquad(qtm, 2)
    cons = semquad_constraints(sqm, qtm)
    @test !isempty(cons)

    for f in (lin, quad)
        uc = [f(sqc.coords[i]...) for i in 1:sqc.N]
        um = [f(sqm.coords[i]...) for i in 1:sqm.N]
        ucm = transfer_semquad(sqc, uc, sqm)
        @test maximum(abs(ucm[i] - um[i]) for i in 1:sqm.N) < 1.0e-11
        umc = transfer_semquad(sqm, um, sqc)
        @test maximum(abs(umc[i] - uc[i]) for i in 1:sqc.N) < 1.0e-11
    end

    ucm = transfer_semquad(sqc, [lin(sqc.coords[i]...) for i in 1:sqc.N], sqm)
    worst = 0.0
    for (c, (ms, ws)) in cons
        worst = max(worst, abs(ucm[c] - sum(w * ucm[m] for (m, w) in zip(ms, ws))))
    end
    @test worst < 1.0e-11
end
