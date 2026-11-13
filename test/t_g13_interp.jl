# Mesh-to-mesh transfer of a nodal field (used for n=16 -> n=32 -> n=64
# restarts of the King run).  The transfer evaluates the source Q_p expansion
# at the destination nodes, so its error is the source interpolation error.
using Test
using QuaST

@testset "G13 theta transfer between meshes" begin
    f(x, z) = sinpi(x) * sinpi(z)
    ms = make_mesh(1.0, 4, 4, 4; zgrading=:uniform)
    fds = make_fedata(4, ms.q)
    xs, zs = node_coords(ms, fds.gll_v)
    ths = [f(xs[mod(k-1,ms.nvx)+1], zs[div(k-1,ms.nvx)+1]) for k in 1:ms.Nscalar]

    # self-transfer is the identity
    same = interpolate_theta(ms, fds, ths, ms, fds.gll_v)
    @test maximum(abs, same .- ths) < 1.0e-13

    # coarse -> fine: the error is the Q4 interpolation error of the source
    for md_n in (16, 32)
        md = make_mesh(1.0, md_n, md_n, 4; zgrading=:uniform)
        fdd = make_fedata(4, md.q)
        thd = interpolate_theta(ms, fds, ths, md, fdd.gll_v)
        xd, zd = node_coords(md, fdd.gll_v)
        err = 0.0
        for iz in eachindex(zd), ix in eachindex(xd)
            err = max(err, abs(thd[(iz-1)*md.nvx+ix] - f(xd[ix], zd[iz])))
        end
        @test err < 1.0e-5
    end

    # a constant field stays constant
    cst = fill(0.37, ms.Nscalar)
    md = make_mesh(1.0, 12, 12, 4; zgrading=:uniform)
    fdd = make_fedata(4, md.q)
    out = interpolate_theta(ms, fds, cst, md, fdd.gll_v)
    @test maximum(abs, out .- 0.37) < 1.0e-13
end
