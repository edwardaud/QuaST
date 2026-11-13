# G58: consistent boundary flux (CBF) on exact temperature fields.
#
# ASPECT computes the Dirichlet-boundary heat flux with the Gresho CBF method
# (heat_flux_map.cc, Gresho et al. 1987), not with the strong FE gradient.
# cbf_wall_flux returns the integrated outward flux (bottom, top).  For the
# conductive profile theta=0, u=0 it must be bottom=1.5, top=3.5 (H=5), and for
# the linear perturbation theta=0.1 z it must be bottom=1.6, top=3.4.
using Test
using QuaST

@testset "G58 CBF wall flux" begin
    m = make_mesh(1.0, 8, 8, 4; zgrading=:uniform)
    fd = make_fedata(4, m.q)
    xs, zs = node_coords(m, fd.gll_v)
    N = m.nvx * m.nvz
    pr = Params(Ra=1.0e4, H=5.0, eta_law=:expT, deta_T=1.0, bc=:free)
    ux = zeros(N); uz = zeros(N)

    th0 = zeros(N)
    bot, top = cbf_wall_flux(m, fd, pr, th0, th0, ux, uz, 1.0e-3;
                             theta_prev2=th0, dt_prev=1.0e-3)
    @test bot ≈ 1.5 atol=1e-10
    @test top ≈ 3.5 atol=1e-10
    @test bot + top ≈ pr.H atol=1e-10

    th = zeros(N)
    for iz in eachindex(zs), ix in eachindex(xs)
        th[(iz - 1) * m.nvx + ix] = 0.1 * zs[iz]
    end
    bot, top = cbf_wall_flux(m, fd, pr, th, th, ux, uz, 1.0e-3;
                             theta_prev2=th, dt_prev=1.0e-3)
    @test bot ≈ 1.6 atol=1e-10
    @test top ≈ 3.4 atol=1e-10
    @test bot + top ≈ pr.H atol=1e-10
end
