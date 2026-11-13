# G49: an AMR rebuild must not freeze the CFL time step.
#
# The AMR block at the end of a step rebuilds the mesh and zeroes ux/uz.  An
# end-of-step CFL update would then see umax=0 and dt could only shrink, never
# grow -- the Ra=1e8 smoke sat at dtmin for exactly this reason.  With the dt
# update at the start of the step, an AMR run whose tree does not change
# (refine_frac=0) must reproduce the non-AMR dt sequence exactly.
using Test
using LinearAlgebra
using QuaST

function uniform_tree(level::Int)
    qt = Quadtree(level + 2)
    for _ in 1:level
        for lf in collect(qt.leaves)
            refine!(qt, lf)
        end
    end
    return qt
end

@testset "G49 AMR CFL growth" begin
    qt_ad = uniform_tree(2)
    qt_st = uniform_tree(2)
    sq = make_semquad(qt_ad, 2)
    theta0 = [0.01 * cospi(sq.coords[i][1]) * sinpi(sq.coords[i][2]) for i in 1:sq.N]
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0e2, bc=:free)

    kw = (tstop=1.0e-3, dt0=1.0e-6, cfl=20.0, maxpic=1,
          maxsteps=10, linear=:direct, verbose=false)
    # The dt invariant is calibrated against the classic cycle; pin it.
    ad = run_semquad_simulation(qt_ad, pr, theta0;
                                amr_every=1, refine_frac=0.0, coarsen_frac=0.0,
                                amr_policy=:classic, kw...)
    st = run_semquad_simulation(qt_st, pr, theta0;
                                amr_every=0, kw...)

    dt_ad = [h[2] for h in ad.history]
    dt_st = [h[2] for h in st.history]
    @test length(dt_ad) == 10
    @test all(diff(dt_ad) .> 0.0)          # the CFL growth branch is reachable
    @test dt_ad[end] > 1.2 * dt_ad[1]
    @test dt_ad ≈ dt_st rtol=1e-12         # AMR bookkeeping does not change dt
    @test ad.t ≈ st.t rtol=1e-12
end
