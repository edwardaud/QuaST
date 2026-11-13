# G36: BDF2 energy step on the quadtree.
#
# Pure diffusion of theta = exp(-2 pi^2 t) cos(pi x) sin(pi y): zero normal
# derivative on the insulating sides, zero value on the top/bottom walls.  The
# dt-difference ratio isolates the time error (the spatial discretisation
# cancels), so this is the quadtree analogue of G26.
using Test
using LinearAlgebra
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

@testset "G36 quadtree BDF2 energy step" begin
    L = 3
    qt = uniform_tree(L)
    sq = make_semquad(qt, 2)
    pr = Params(Ra=1.0, H=0.0, eta_law=:const, bc=:free, supg=0.0)
    th0 = [cospi(sq.coords[i][1]) * sinpi(sq.coords[i][2]) for i in 1:sq.N]
    exactf(t, x, y) = exp(-2 * pi^2 * t) * cospi(x) * sinpi(y)

    function run(dt::Float64, nsteps::Int)
        th = copy(th0)
        thm1 = copy(th0)
        t = 0.0
        for s in 1:nsteps
            thn = solve_semquad_energy(sq, qt, pr, th, thm1,
                                       zeros(Float64, sq.N), zeros(Float64, sq.N),
                                       dt, dt, s == 1; advection=:galerkin)
            thm1 = th
            th = thn
            t += dt
        end
        return th, t
    end

    tha, _ = run(2.0e-3, 10)
    thb, _ = run(1.0e-3, 20)
    thc, tc = run(5.0e-4, 40)
    d12 = norm(tha - thb) / norm(thb)
    d23 = norm(thb - thc) / norm(thc)
    @test log2(d12 / d23) > 1.5

    ex = [exactf(tc, sq.coords[i][1], sq.coords[i][2]) for i in 1:sq.N]
    @test norm(thc - ex) / norm(ex) < 0.05
end
