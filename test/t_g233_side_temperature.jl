# G233: prescribed temperature on the side walls.
using Test, QuaST

# G233 (Phase 0, item 0.4): `Params.side_thermal = :fixed` replaces the historical
# insulating (natural) side wall with the physical temperatures `Twall_left` and
# `Twall_right`, using the same `theta_wall = Twall - Tbar(y)` convention as the
# top and bottom walls.  Three things must hold:
#
#   (a) every node on a side wall takes exactly the prescribed value;
#   (b) `:insulating` is the default and leaves those dofs free, bit for bit;
#   (c) asking for both a fixed side temperature and a side flux is refused.

@testset "G233 side-wall temperature" begin
    function _g233_tree(level::Int)
        q = Quadtree(level)
        for _ in 1:level, lf in collect(q.leaves)
            refine!(q, lf)
        end
        return q
    end

    qt = _g233_tree(2)
    sq = make_semquad(qt, 2)
    th = zeros(Float64, sq.N)
    ux = zeros(Float64, sq.N)
    uz = zeros(Float64, sq.N)

    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
                side_thermal=:fixed, Twall_left=1.0, Twall_right=0.25)
    theta = solve_semquad_energy(sq, qt, pr, th, th, ux, uz, 1.0e-3, 1.0e-3, true)
    @test length(theta) == sq.N

    # (a) the wall values are the physical temperature minus the base state
    nwall = 0
    for g in 1:sq.N
        x, y = sq.coords[g]
        want = nothing
        x <= 1.0e-14 && (want = 1.0 - (1.0 - y))
        x >= sq.Lx - 1.0e-14 && (want = 0.25 - (1.0 - y))
        want === nothing && continue
        nwall += 1
        @test isapprox(theta[g], want; atol=1.0e-12)
    end
    @test nwall > 0                       # the tree really has side nodes

    # (b) the default is insulating, and the two runs differ
    pr2 = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0)
    @test pr2.side_thermal === :insulating
    theta2 = solve_semquad_energy(sq, qt, pr2, th, th, ux, uz, 1.0e-3, 1.0e-3, true)
    @test theta2 != theta

    # (c) fixed temperature and a side flux cannot both be asked for
    @test_throws ArgumentError solve_semquad_energy(
        sq, qt, pr, th, th, ux, uz, 1.0e-3, 1.0e-3, true; flux_side=0.5)
end
