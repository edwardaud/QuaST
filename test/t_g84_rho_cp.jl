# G84: rho(T)cp weighting of the SEM mass/advection matrices.
using Test
using LinearAlgebra
using QuaST

@testset "G84 rho cp energy" begin
    qt = wall_graded_tree(2, 5; strip=0.15)
    sq = make_semquad(qt, 3)
    th = fill(0.4, sq.N)
    for g in 1:sq.N
        y = sq.coords[g][2]
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (th[g] = 0.0)
    end
    pr0 = Params(Ra=1.0e6, H=5.0, eta_law=:expT, deta_T=1.0e3,
                 bc=:free, rho_heating=false, rho_cp=false)
    pr1 = Params(Ra=1.0e6, H=5.0, eta_law=:expT, deta_T=1.0e3,
                 bc=:free, rho_heating=true, rho_cp=true)
    ux = zeros(sq.N); uz = zeros(sq.N)
    M0, _, _, F0 = assemble_semquad_energy(sq, qt, pr0, ux, uz)
    M1, _, _, F1 = assemble_semquad_energy(sq, qt, pr1, ux, uz;
                                           theta=th, rho_heating=true,
                                           rho_cp=true)
    @test norm(M1 - M0) > 0.0
    @test norm(F1 - F0) > 0.0
    @test all(isfinite, M1.nzval)
    @test all(isfinite, F1)
    @test_throws ArgumentError assemble_semquad_energy(sq, qt, pr1, ux, uz;
                                                       rho_cp=true)
end
