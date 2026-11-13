using Test, QuaST

# G143: saturation / error-bar protocol.
#
# The protocol reads the diagnostic history rows (t, dt, Nu_top, Nu_bot, Vrms,
# mean_T, ...) and reports whether the last `window` steps are flat, with half
# the spread as the error bar.  Synthetic histories pin the decision, and a short
# real run checks that the function accepts the actual row format.

function _row(t, nut, nub, vr, ta)
    return [t, 1.0e-4, nut, nub, vr, ta, 100.0, 16.0, 1.0e-9, 0.06, 1.0,
            1.0e-4, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0]
end

@testset "G143 saturation protocol" begin
    # perfectly calm history -> saturated with zero drift
    hist = [_row(0.01 * i, 2.0, 2.0, 1.0, 0.5) for i in 1:10]
    s = semquad_saturation(hist; window=5)
    @test s.saturated
    @test s.n == 10
    @test maximum(abs, collect(s.drift)) < 1.0e-15
    @test maximum(abs, collect(s.err)) < 1.0e-15
    @test s.t_to ≈ 0.10

    # a linear ramp is not saturated and the drift is order 1e-2
    hist = [_row(0.01 * i, 2.0 + 1.0e-3 * i, 2.0 + 1.0e-3 * i, 1.0, 0.5)
            for i in 1:10]
    s = semquad_saturation(hist; window=5, rtol=1.0e-3)
    @test !s.saturated
    @test s.drift.Nu_top > 1.0e-3
    @test s.err.Nu_top ≈ 2.0e-3    # half the spread of five steps of 1e-3

    # a bump inside the window is detected
    hist = [_row(0.01 * i, 2.0, 2.0, 1.0, 0.5) for i in 1:10]
    hist[8][5] = 1.5
    s = semquad_saturation(hist; window=5)
    @test !s.saturated
    @test s.drift.Vrms ≈ 0.5 / 1.1 rtol = 1.0e-12   # spread 0.5 over mean 1.1
    @test s.err.Vrms ≈ 0.25

    # too little history -> not saturated, NaNs
    s = semquad_saturation(hist[1:2]; window=5)
    @test !s.saturated
    @test isnan(s.drift.Vrms)

    # a calm tail after a transient is saturated (only the window matters)
    hist = vcat([_row(0.01 * i, 4.0, 4.0, 9.0, 0.6) for i in 1:10],
                [_row(0.1 + 0.01 * i, 2.0, 2.0, 1.0, 0.5) for i in 1:6])
    s = semquad_saturation(hist; window=5)
    @test s.saturated

    # the function accepts the real history format from the driver
    qt = Quadtree(4)
    for _ in 1:1, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0e-8)
    th0 = [0.01 * cos(pi * x) * sin(pi * y) for (x, y) in ss.su.coords]
    st = run_semquad_simulation(qt, pr, th0; p=2, tstop=2.0e-4, dt0=5.0e-5,
                                maxsteps=8, verbose=false)
    @test length(st.history[1]) == 18
    s2 = semquad_saturation(st.history; window=3, rtol=1.0e-2)
    @test s2.n == length(st.history)
    @test haskey(s2.drift, :Nu_top) && haskey(s2.err, :mean_T)
    @test isfinite(s2.drift.Nu_top)

    # optional CBF columns: 19/20 only when requested, and they must agree with a
    # direct evaluation of the same state
    stc = run_semquad_simulation(qt, pr, th0; p=2, tstop=2.0e-4, dt0=5.0e-5,
                                 maxsteps=2, verbose=false, nu_cbf=true)
    @test length(stc.history[1]) == 20
    ssc = make_semstokes(stc.qt, 2)
    ntc, nbc = semquad_nu_cbf(ssc.su, stc.qt, pr, stc.theta, stc.ux, stc.uz)
    @test stc.history[end][19] ≈ ntc rtol = 1.0e-12
    @test stc.history[end][20] ≈ nbc rtol = 1.0e-12
    # the first four monitored columns are unchanged by the extra diagnostics
    @test semquad_saturation(stc.history; window=2).n == length(stc.history)

    # argument checks
    @test_throws ArgumentError semquad_saturation(hist; window=1)
    @test_throws ArgumentError semquad_saturation(hist; rtol=0.0)
end
