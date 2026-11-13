using Test
using QuaST

@testset "mesh and reduced DOF maps" begin
    m = make_mesh(1.0, 4, 3, 4)
    @test m.nvx == 17
    @test m.nvz == 13
    @test m.Nscalar == 221
    @test m.Nvel == 442
    @test m.q == 3
    @test m.npx == 13
    @test m.npz == 10
    @test m.Npres == 130

    # shared edge between adjacent elements: every node on the interface
    g0 = element_vel_gids(m, 0, 0)
    g1 = element_vel_gids(m, 1, 0)
    for j in 1:m.p+1
        k0 = (j - 1) * (m.p + 1) + (m.p + 1)
        k1 = (j - 1) * (m.p + 1) + 1
        @test g0[k0] == g1[k1]
    end
    @test vel_gid(m, 0, 0, 5, 1) == vel_gid(m, 1, 0, 1, 1)

    dm = make_dofmap(m, :free)
    @test dm.nux == 195
    @test dm.nuz == 187
    @test dm.npres == 129
    @test dm.ntotal == 511
    @test dm.red_p[dm.pin] == 0

    dmr = make_dofmap(m, :rigid)
    @test dmr.nux < dm.nux
    @test dmr.nuz < dm.nuz
    @test dmr.ntotal == dmr.nux + dmr.nuz + dmr.npres

    dmt = make_theta_map(m)
    @test dmt.n == (m.nvz - 2) * m.nvx
    @test dmt.red[1] == 0
    @test dmt.red[(m.nvz-1)*m.nvx+1] == 0

    xs, zs = node_coords(m, gll_rule(4))
    @test isapprox(xs[1], 0.0, atol=1e-15)
    @test isapprox(xs[end], 1.0, atol=1e-15)
    @test isapprox(zs[1], 0.0, atol=1e-15)
    @test isapprox(zs[end], 1.0, atol=1e-15)
    @test issorted(xs)
    @test issorted(zs)
end
