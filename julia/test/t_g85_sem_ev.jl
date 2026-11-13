# G85: SEM entropy-viscosity diffusion term.
using Test
using LinearAlgebra
using MantleSEM

@testset "G85 SEM entropy viscosity" begin
    qt = wall_graded_tree(2, 5; strip=0.15)
    sq = make_semquad(qt, 3)
    th = [0.05 * cospi(x) * sinpi(y) for (x, y) in sq.coords]
    for g in 1:sq.N
        y = sq.coords[g][2]
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (th[g] = 0.0)
    end
    ux = fill(1.0e-3, sq.N); uz = fill(0.0, sq.N)
    pr = Params(Ra=1.0e5, H=5.0, eta_law=:expT, deta_T=1.0e3, bc=:free)
    evg = MantleSEM.semquad_ev_globals(sq, qt, pr, th, copy(th), ux, uz, 0.0)
    _, K0, _, _ = assemble_semquad_energy(sq, qt, pr, ux, uz;
                                          theta=th, dt=1.0e-9)
    _, K1, _, _ = assemble_semquad_energy(sq, qt, pr, ux, uz;
                                          theta=th, theta_prev=copy(th),
                                          dt=1.0e-9, ev=0.11, ev_cmax=0.052,
                                          evg=evg, ev_timestep=5)
    @test norm(K1 - K0) > 0.0
    @test all(isfinite, K1.nzval)
    @test evg.entropy_variation >= 0.0
    @test_throws ArgumentError assemble_semquad_energy(sq, qt, pr, ux, uz;
                                                       theta=th, ev=0.11)
end
