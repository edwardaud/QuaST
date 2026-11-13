# G37: free-slip boundary conditions in the quadtree variable-viscosity Stokes.
#
# Manufactured divergence-free flow
#   u = ( sin(pi x) cos(pi y), -cos(pi x) sin(pi y) ),  p = sin(pi x) sin(pi y)
# is free-slip: ux=0 on x=0,1 and uz=0 on y=0,1, tangential components free.
# A sign error in the pressure coupling would make the pressure error stop
# converging; this gate checks the velocity and pressure orders on mixed trees.
using Test
using LinearAlgebra
using QuaST

@testset "G37 free-slip quadtree Stokes" begin
    ux_ex(x, y) = sinpi(x) * cospi(y)
    uz_ex(x, y) = -cospi(x) * sinpi(y)
    p_ex(x, y) = sinpi(x) * sinpi(y)
    fx(x, y) = 2 * pi^2 * ux_ex(x, y) + pi * cospi(x) * sinpi(y)
    fz(x, y) = 2 * pi^2 * uz_ex(x, y) + pi * sinpi(x) * cospi(y)

    function errors(ss, u, pv)
        M = 20
        acc_u = 0.0; acc_p = 0.0; pm = 0.0
        for k in 0:M, l in 0:M
            x = (k + 0.5) / (M + 1); y = (l + 0.5) / (M + 1)
            ux, uz, pp = semstokes_eval(ss, u, pv, x, y)
            acc_u += (ux - ux_ex(x, y))^2 + (uz - uz_ex(x, y))^2
            pm += pp - p_ex(x, y)
        end
        pm /= (M + 1)^2
        for k in 0:M, l in 0:M
            x = (k + 0.5) / (M + 1); y = (l + 0.5) / (M + 1)
            _, _, pp = semstokes_eval(ss, u, pv, x, y)
            acc_p += (pp - pm - p_ex(x, y))^2
        end
        return sqrt(acc_u / (M + 1)^2), sqrt(acc_p / (M + 1)^2)
    end

    qt = Quadtree(9)
    eu = Float64[]; ep = Float64[]
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
        u, pv, _, _, _ = solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0,
                                                  fx, fz,
                                                  (x, y) -> 0.0, (x, y) -> 0.0;
                                                  bc=:free)
        e1, e2 = errors(ss, u, pv)
        push!(eu, e1); push!(ep, e2)
    end
    @test log2(eu[1] / eu[2]) > 2.0
    @test log2(eu[2] / eu[3]) > 2.0
    @test log2(ep[1] / ep[2]) > 1.0
    @test log2(ep[2] / ep[3]) > 1.0
end
