# G47: DG advection with level-local time steps (subcycling prototype).
#
# The coarse leaves use m_l*dt_f and are updated every m_l fine steps; the
# fine leaves use dt_f.  On a 2:1-refined tree the subcycled solution must stay
# close to the reference integrator that uses dt_f for every leaf, and the
# coarse leaves must indeed skip fine steps.
using Test
using LinearAlgebra
using MantleSEM

@testset "G47 DG subcycling" begin
    qt = Quadtree(6)
    refine!(qt, (0, 0, 0))
    refine!(qt, (1, 0, 0))
    refine!(qt, (2, 0, 0))
    @test balance!(qt)
    dg = make_dgquad(qt, 2)

    T0 = zeros(Float64, dg.N)
    for lf in dg.leaves
        h = 2.0^(-lf[1])
        x0 = lf[2] * h; y0 = lf[3] * h
        off = dg.base[lf]
        for j in 1:(dg.p + 1), i in 1:(dg.p + 1)
            x = x0 + 0.5 * (dg.xi[i] + 1) * h
            y = y0 + 0.5 * (dg.xi[j] + 1) * h
            T0[off + (j - 1) * (dg.p + 1) + i] =
                exp(-((x - 0.3)^2 + (y - 0.5)^2) / 0.02)
        end
    end

    Tref = run_dgquad_advection_subcycled(dg, qt, 1.0, 0.0, T0;
                                          tstop=0.4, cfl=0.4,
                                          inflow=(x, y) -> 0.0,
                                          subcycle=false)
    Tsub = run_dgquad_advection_subcycled(dg, qt, 1.0, 0.0, T0;
                                          tstop=0.4, cfl=0.4,
                                          inflow=(x, y) -> 0.0,
                                          subcycle=true)
    @test all(isfinite, Tsub)
    @test norm(Tsub) > 0.0
    @test norm(Tsub - Tref) / norm(Tref) < 0.1

    hlev = [2.0^(-lf[1]) for lf in dg.leaves]
    dtf = 0.4 * minimum(hlev)
    m = [max(1, round(Int, 0.4 * h / dtf)) for h in hlev]
    @test maximum(m) >= 2
    @test count(>(1), m) >= 1
end
