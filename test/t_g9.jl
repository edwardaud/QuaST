using Test
using QuaST

@testset "G9: internal heating, conductive state, rigid walls" begin
    m = make_mesh(1.0, 8, 8, 4)
    fd = make_fedata(m.p, m.q)
    @test isapprox(Tbar(0.0, 5.0), 1.0, atol=1e-15)
    @test isapprox(Tbar(1.0, 5.0), 0.0, atol=1e-15)
    @test isapprox(Tbar(0.5, 5.0), 0.5 + 0.5 * 5.0 * 0.25, atol=1e-15)
    @test isapprox(dTbar(0.0, 5.0), -1.0 + 2.5, atol=1e-15)

    # conductive state, no motion, H=5: Nu_top = 1+H/2, Nu_bot = 1-H/2
    pr = Params(Ra=0.0, H=5.0, eta_law=:const, bc=:rigid)
    dm = make_dofmap(m, :rigid)
    theta = zeros(Float64, m.Nscalar)
    sol = solve_stokes(m, fd, pr, theta, dm)
    @test maximum(abs, sol.ux) < 1e-12
    @test maximum(abs, sol.uz) < 1e-12
    Nu_top, Nu_bot = nu_walls(m, fd.gll_v, theta, pr)
    @test isapprox(Nu_top, 3.5, atol=1e-12)
    @test isapprox(Nu_bot, -1.5, atol=1e-12)
    @test isapprox(nu_energy(m, fd, pr, theta, sol.ux, sol.uz), 3.5, atol=1e-12)

    # H=0 free-slip conductive state: Nu=1, u=0
    pr0 = Params(Ra=1e4, H=0.0, eta_law=:const, bc=:free)
    dm0 = make_dofmap(m, :free)
    sol0 = solve_stokes(m, fd, pr0, theta, dm0)
    @test maximum(abs, sol0.ux) < 1e-10
    @test maximum(abs, sol0.uz) < 1e-10
    @test isapprox(nu_walls(m, fd.gll_v, theta, pr0)[1], 1.0, atol=1e-12)
end
