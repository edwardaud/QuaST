# G57: the divergence-budget diagnostic int(theta div u).
#
# The H=5 closure error should satisfy
#   Nu_top - Nu_bot - H = int theta div(u) + d/dt int theta.
# This gate checks the first term on fields with known divergence:
#   u=(x,0), theta=1        -> int theta div u = 1
#   u=(x,0), theta=y        -> 1/2
#   u=(2xy,-y^2), any theta -> 0 (exactly divergence-free)
using Test
using MantleSEM

@testset "G57 theta divu integral" begin
    m = make_mesh(1.0, 6, 6, 2; zgrading=:uniform)
    fd = make_fedata(2, m.q)
    xs, zs = node_coords(m, fd.gll_v)
    nu = m.nvx * m.nvz
    ux = zeros(nu); uz = zeros(nu)
    for iz in eachindex(zs), ix in eachindex(xs)
        g = (iz - 1) * m.nvx + ix
        ux[g] = xs[ix]
        uz[g] = 0.0
    end
    th1 = ones(nu)
    thy = zeros(nu)
    for iz in eachindex(zs), ix in eachindex(xs)
        thy[(iz - 1) * m.nvx + ix] = zs[iz]
    end
    @test theta_divu_integral(m, fd, th1, ux, uz) ≈ 1.0 atol=1e-12
    @test theta_divu_integral(m, fd, thy, ux, uz) ≈ 0.5 atol=1e-12

    # u=(2xy,-y^2) is pointwise divergence free.
    for iz in eachindex(zs), ix in eachindex(xs)
        g = (iz - 1) * m.nvx + ix
        ux[g] = 2.0 * xs[ix] * zs[iz]
        uz[g] = -zs[iz]^2
    end
    th = [1.0 + xs[mod1(i, m.nvx)] + zs[cld(i, m.nvx)] for i in 1:nu]
    @test abs(theta_divu_integral(m, fd, th, ux, uz)) < 1e-12
end
