# G83: rho=(1-T) heating correction in the SEM energy source.
#
# ASPECT's constant heating is rho*H; our base state already absorbs H, so the
# correction is (rho-1)H = -T H.  With T>1 this is a negative (cooling) source.
using Test
using LinearAlgebra
using QuaST

@testset "G83 rho H heating" begin
    qt = wall_graded_tree(2, 5; strip=0.15)
    sq = make_semquad(qt, 3)
    th = fill(0.5, sq.N)
    for g in 1:sq.N
        y = sq.coords[g][2]
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (th[g] = 0.0)
    end
    pr0 = Params(Ra=1.0e6, H=5.0, eta_law=:expT, deta_T=1.0e3,
                 bc=:free, rho_heating=false)
    pr1 = Params(Ra=1.0e6, H=5.0, eta_law=:expT, deta_T=1.0e3,
                 bc=:free, rho_heating=true)
    ux = zeros(sq.N); uz = zeros(sq.N)
    _, _, _, F0 = assemble_semquad_energy(sq, qt, pr0, ux, uz;
                                          theta=th, rho_heating=false)
    _, _, _, F1 = assemble_semquad_energy(sq, qt, pr1, ux, uz;
                                          theta=th, rho_heating=true)
    @test norm(F1 - F0) > 0.0
    @test dot(F1 - F0, th) < 0.0   # T>1: rho H is a net sink
    @test_throws ArgumentError assemble_semquad_energy(sq, qt, pr1, ux, uz;
                                                       rho_heating=true)
end
