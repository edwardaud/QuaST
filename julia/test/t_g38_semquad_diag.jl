# G38: quadtree diagnostics against exactly representable fields.
using Test
using MantleSEM

function uniform_tree(level::Int)
    qt = Quadtree(level + 2)
    for _ in 1:level
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
    end
    return qt
end

@testset "G38 quadtree diagnostics" begin
    qt = uniform_tree(3)
    sq = make_semquad(qt, 2)
    pr = Params(H=5.0)

    th = [0.3 * sq.coords[i][2] for i in 1:sq.N]
    Nut, Nub = semquad_nu_walls(sq, qt, th, pr)
    @test Nut ≈ 1.0 + 0.5 * 5.0 - 0.3
    @test Nub ≈ 1.0 - 0.5 * 5.0 - 0.3
    @test Nut - Nub ≈ 5.0

    pr0 = Params(H=0.0)
    Nut0, Nub0 = semquad_nu_walls(sq, qt, th, pr0)
    @test Nut0 ≈ 0.7
    @test Nub0 ≈ 0.7

    ux = fill(2.0, sq.N)
    uz = zeros(Float64, sq.N)
    @test semquad_vrms(sq, qt, ux, uz) ≈ 2.0

    th0 = zeros(Float64, sq.N)
    @test semquad_mean_T(sq, qt, th0, pr) ≈ 0.5 + 5.0 / 12.0
    uz2 = fill(2.0, sq.N)
    @test semquad_nu_energy(sq, qt, pr, th0, ux, uz2) ≈ 1.0 + 2.5
    @test semquad_nu_energy(sq, qt, pr, th, ux, uz2) ≈ 1.0 + 2.5 + 0.3
end
