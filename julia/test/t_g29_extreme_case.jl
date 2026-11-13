# G29: extreme-case definition (Ra=1e8, Delta eta=1e3, H=5).
#
# This gate fixes the source-backed numbers of EXTREME_CASE.md before any AMR or
# high-Ra run uses them: the conductive base state, the H=5 Nusselt convention,
# the viscosity endpoints, and the integrated heating source.
using Test
using MantleSEM

@testset "G29 extreme H=5 case definition" begin
    H = 5.0
    Ra = 1.0e8
    deta = 1.0e3

    # conductive base state Tbar = 1 - z + (H/2) z(1-z)
    @test Tbar(0.0, H) ≈ 1.0
    @test Tbar(0.5, H) ≈ 1.125
    @test Tbar(1.0, H) ≈ 0.0
    @test dTbar(0.0, H) ≈ 1.5
    @test dTbar(1.0, H) ≈ -3.5

    # viscosity endpoints: eta(T=0)=1, eta(T=1)=1/Delta eta
    pr = Params(Ra=Ra, H=H, eta_law=:expT, deta_T=deta, bc=:free)
    @test eta_at(0.0, 0.5, pr)[1] ≈ 1.0
    @test eta_at(1.0, 0.5, pr)[1] ≈ 1.0e-3
    @test eta_at(0.5, 0.5, pr)[1] ≈ 1.0 / sqrt(deta)

    # conductive Nusselt values with the H-aware wall convention
    m = make_mesh(1.0, 4, 4, 2)
    fd = make_fedata(m.p, m.q)
    th = zeros(Float64, m.Nscalar)
    ux = zeros(Float64, m.Nscalar)
    uz = zeros(Float64, m.Nscalar)
    Nu_top, Nu_bot = nu_walls(m, fd.gll_v, th, pr)
    @test Nu_top ≈ 1.0 + 0.5 * H
    @test Nu_bot ≈ 1.0 - 0.5 * H
    @test Nu_top - Nu_bot ≈ H
    @test nu_energy(m, fd, pr, th, ux, uz) ≈ 1.0 + 0.5 * H

    # H=0 must reduce to the usual Nu=1 conductive state
    pr0 = Params(Ra=Ra, H=0.0, eta_law=:expT, deta_T=deta, bc=:free)
    Nu_top0, Nu_bot0 = nu_walls(m, fd.gll_v, th, pr0)
    @test Nu_top0 ≈ 1.0
    @test Nu_bot0 ≈ 1.0

    # The theta equation has no explicit +H: Tbar already satisfies
    # lap(Tbar) + H = 0, so the volume source integrates to zero for u=0.
    dmt = ThetaMap(collect(1:m.Nscalar), m.Nscalar)
    em = assemble_energy(m, fd, pr, ux, uz, dmt;
                         ev=0.0, theta_now=th, theta_prev=th, dt=1.0)
    @test sum(em.F) ≈ 0.0 atol=1.0e-12
end
