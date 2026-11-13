using Test, MantleSEM, LinearAlgebra

# G142: maxpic > 1 lands near the lagged answer.
#
# Historical context: with the pre-fix time loop (`theta_iter` used as the BDF2
# `theta^n`) the undamped Picard iteration drifted to a different equation and
# this gate asserted that `maxpic = 6` with `pic_relax = 1` diverges.  After the
# `theta^n` fix (docs/RHEOLOGY.md §3) the iteration is a genuine fixed point of
# the BDF2 step, so the undamped variant converges as well and all three rows
# (lagged reference, adaptive damping, undamped) agree to a few percent.

function _run_pic(; maxpic, deta_T, relax=1.0, adapt=false)
    qt = Quadtree(4)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=deta_T)
    th0 = [0.05 * cos(pi * x) * sin(pi * y) for (x, y) in ss.su.coords]
    st = run_semquad_simulation(qt, pr, th0; p=2, tstop=5.0e-3, dt0=5.0e-5,
                                maxsteps=40, maxpic=maxpic, pic_tol=1.0e-9,
                                pic_relax=relax, pic_adapt=adapt,
                                pic_relax_min=0.05, verbose=false)
    ss2 = make_semstokes(st.qt, 2)
    return semquad_vrms(ss2.su, st.qt, st.ux, st.uz)
end

@testset "G142 maxpic > 1 with adaptive damping" begin
    for deta_T in (1.0e-8, 1.0, 3.0)
        ref = _run_pic(maxpic=1, deta_T=deta_T)
        @test isfinite(ref) && ref > 0.0
        # adaptive: bounded and in the same ballpark as the lagged answer
        vad = _run_pic(maxpic=6, deta_T=deta_T, adapt=true)
        @test isfinite(vad)
        @test vad < 3.0 * ref
        @test vad > 0.1 * ref
        # undamped (theta^n fix, docs/RHEOLOGY.md §3): converges to the same
        # state, and no longer throws or diverges
        threw = false
        local vun = NaN
        try
            vun = _run_pic(maxpic=6, deta_T=deta_T, relax=1.0)
        catch err
            threw = true
        end
        @test !threw
        @test isfinite(vun)
        @test 0.1 * ref < vun < 3.0 * ref
        # with the fixed iteration both variants reach the same fixed point
        @test abs(vad - vun) <= 0.05 * ref
    end
end
