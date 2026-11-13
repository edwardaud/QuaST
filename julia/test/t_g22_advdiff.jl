# Steady advection-diffusion on the quadtree: the linear field must be exact on
# uniform and on 2:1 mixed meshes (advection + SIPG diffusion + inflow BCs).
using Test
using LinearAlgebra
using MantleSEM

function nodal_err(dg, u, exact)
    n = dg.p + 1
    e = 0.0
    for lf in dg.leaves
        h = 2.0^(-lf[1])
        x0 = lf[2] * h; y0 = lf[3] * h
        for j in 1:n, i in 1:n
            x = x0 + 0.5 * (dg.xi[i] + 1) * h
            y = y0 + 0.5 * (dg.xi[j] + 1) * h
            e = max(e, abs(u[MantleSEM._gidq(dg, lf, i, j)] - exact(x, y)))
        end
    end
    return e
end

@testset "G22 quadtree advection-diffusion" begin
    ux = 1.0; uy = 0.5
    lin(x, y) = 1.0 + 0.3x - 0.2y
    fadv(x, y) = ux * 0.3 + uy * (-0.2)

    for (label, build) in (("uniform", () -> begin
            qt = Quadtree(4); refine!(qt, (0, 0, 0))
            for dy in 0:1, dx in 0:1
                refine!(qt, (1, dx, dy))
            end
            qt
        end), ("mixed", () -> begin
            qt = Quadtree(4); refine!(qt, (0, 0, 0)); refine!(qt, (1, 0, 0)); balance!(qt)
            qt
        end))
        qt = build()
        dg = make_dgquad(qt, 2)
        Ad, bd = assemble_dgquad_poisson(dg, qt, fadv, lin; sigma_C=50.0)
        Aa, ba = assemble_dgquad_advection(dg, qt, ux, uy, lin)
        u = (Ad + Aa) \ (bd + ba)
        @test nodal_err(dg, u, lin) < 1.0e-9
    end
end
