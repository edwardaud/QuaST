# Manufactured Stokes convergence on mixed quadtrees.
# u = (sin(pi x) cos(pi y), -cos(pi x) sin(pi y)) (divergence free),
# p = sin(pi x) sin(pi y); f = -Delta u + grad p.  With Q_2/Q_1 the velocity
# error must be ~h^3 and the (mean-subtracted) pressure error ~h^2.
using Test
using LinearAlgebra
using MantleSEM

@testset "G28 quadtree Stokes convergence" begin
    ux_ex(x, y) = sinpi(x) * cospi(y)
    uz_ex(x, y) = -cospi(x) * sinpi(y)
    p_ex(x, y) = sinpi(x) * sinpi(y)
    fx(x, y) = 2 * pi^2 * ux_ex(x, y) + pi * cospi(x) * sinpi(y)
    fz(x, y) = 2 * pi^2 * uz_ex(x, y) + pi * sinpi(x) * cospi(y)

    function errors(ss, u, pv)
        M = 20
        eu = 0.0; ep = 0.0
        pm = 0.0
        for k in 0:M, l in 0:M
            x = 1.0e-6 + (1 - 2.0e-6) * k / M
            y = 1.0e-6 + (1 - 2.0e-6) * l / M
            ux, uz, pp = semstokes_eval(ss, u, pv, x, y)
            eu += (ux - ux_ex(x, y))^2 + (uz - uz_ex(x, y))^2
            pm += pp - p_ex(x, y)
        end
        pm /= (M + 1)^2
        for k in 0:M, l in 0:M
            x = 1.0e-6 + (1 - 2.0e-6) * k / M
            y = 1.0e-6 + (1 - 2.0e-6) * l / M
            _, _, pp = semstokes_eval(ss, u, pv, x, y)
            ep += (pp - pm - p_ex(x, y))^2
        end
        return sqrt(eu / (M + 1)^2), sqrt(ep / (M + 1)^2)
    end

    qt = Quadtree(9)
    refuse = Float64[]; pref = Float64[]
    for step in 1:3
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        base = minimum(l for (l, _, _) in qt.leaves)
        refine_where!(qt, (l, ix, iy) -> l < base + 1 &&
                      abs((ix + 0.5) / (1 << l) - 0.5) < 0.25 &&
                      abs((iy + 0.5) / (1 << l) - 0.5) < 0.25)
        @test max_face_level_jump(qt) <= 1
        ss = make_semstokes(qt, 2)
        u, pv, _, _, _ = solve_semquad_stokes(ss, qt, fx, fz, ux_ex, uz_ex)
        eu, ep = errors(ss, u, pv)
        push!(refuse, eu); push!(pref, ep)
    end
    @test log2(refuse[1] / refuse[2]) > 2.0
    @test log2(refuse[2] / refuse[3]) > 2.0
    @test log2(pref[1] / pref[2]) > 1.0
    @test log2(pref[2] / pref[3]) > 1.0
end
