# G80: L2 projection transfer vs nodal restriction.
#
# Coarsening with pointwise nodal values can leave interface steps.  The L2
# projection solves M u = b with b_i = int phi_i u_old and should have a
# smaller L2 error on a smooth analytic field, while keeping walls at zero.
using Test
using LinearAlgebra
using Printf
using MantleSEM

const H80 = (x, y) -> 0.2 * exp(-((x - 0.5) / 0.2)^2 -
                                ((y - 0.35) / 0.15)^2) * sinpi(y)

@testset "G80 L2 transfer" begin
    qf = wall_graded_tree(2, 7; strip=0.15)
    qc = wall_graded_tree(2, 5; strip=0.15)
    sf = make_semquad(qf, 4)
    sc = make_semquad(qc, 4)
    thf = [H80(x, y) for (x, y) in sf.coords]
    tn = transfer_semquad(sf, thf, sc; mode=:nodal, zero_walls=true)
    tl = transfer_semquad(sf, thf, sc; mode=:l2, zero_walls=true)
    tex = [H80(x, y) for (x, y) in sc.coords]
    M = semquad_mass(sc)
    en = sqrt(abs(dot(tn - tex, M * (tn - tex))))
    el = sqrt(abs(dot(tl - tex, M * (tl - tex))))
    @printf("G80 nodal=%.6e L2=%.6e\n", en, el)
    @test el <= en * 1.05
    @test maximum(abs, tl[findall(p -> p[2] <= 0.0 || p[2] >= 1.0, sc.coords)]) < 1e-12
    @test maximum(abs, tl) < 0.25
end
