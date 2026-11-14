# G230: the default linear solver is :auto — direct below the thresholds, MG above.
using Test, QuaST, LinearAlgebra

# G230: the library default of `run_semquad_simulation` is `linear=:auto`
# since 2026-11-13 (docs/dev/SOLVER.md §4; the flip follows the staged pattern of
# docs/dev/AMR.md).
#
# G191 tests the selector itself with an *explicit* `linear=:auto` and pins the
# decision table; this gate is the one that exercises the **default**, so a
# future revert to `:direct` fails here without touching G191:
#   (a) below the selector threshold (uniform L2, 747 dofs, contrast ~1) a
#       driver call with no `linear` records an `:auto` decision that resolves
#       to `:direct`, and that answer is bitwise the explicit `:direct` one;
#   (b) above it (uniform L4, 10 851 dofs, contrast 1) the default resolves to
#       `:mg`, with no fallback, and two default runs are bitwise equal;
#   (c) `linear=:direct` and `linear=:mg` remain selectable: an explicit call
#       still records `:none`, so the default is not hard-wired;
#   (d) `pic_mode=:newton` without `linear=:direct` fails with a message that
#       names the fix, and the same call runs with `linear=:direct` explicit.
#
# A `:none` record at (a) or (b) means the driver stopped forwarding `:auto` by
# default -- that is the assertion that pins the library default itself.

function _g230_uniform(level::Int)
    qt = Quadtree(level)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

function _g230_run(qt, p; maxsteps::Int=2, kw...)
    ss = make_semstokes(qt, p)
    th = [0.01 * cospi(x) * sinpi(y) for (x, y) in ss.su.coords]
    for (i, (x, y)) in enumerate(ss.su.coords)
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (th[i] = 0.0)
    end
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
                Twall_top=0.0, Twall_bot=1.0, bc=:free)
    # No `linear` kwarg in the base call: the default is what (a)/(b) assert.
    return run_semquad_simulation(qt, pr, th; p=p, tstop=2.0e-4, dt0=1.0e-4,
                                  maxsteps=maxsteps, verbose=false, kw...)
end

@testset "G230 default linear is :auto" begin
    # ---- (a) below the threshold: default == explicit :direct, bitwise ------
    qt2 = _g230_uniform(2)                        # 16 leaves
    ss2 = make_semstokes(qt2, 4)
    @test 2 * ss2.su.N + ss2.sp.N == 747          # < SEMQUAD_AUTO_DOFS_CONTRAST_100
    r_auto = _g230_run(qt2, 4)
    ch = LAST_SEMSTOKES_AUTO_CHOICE[]
    @test ch.choice === :direct                   # not :none: the default is `:auto`
    @test ch.saddle_dofs == 747
    @test 1.0 <= ch.contrast <= 1.1               # near-constant eta
    @test ch.mg_threads == 0
    @test occursin("saddle_dofs=747", ch.reason)
    r_dir = _g230_run(qt2, 4; linear=:direct)
    @test LAST_SEMSTOKES_AUTO_CHOICE[].choice === :none
    @test r_auto.theta == r_dir.theta
    @test r_auto.ux == r_dir.ux && r_auto.uz == r_dir.uz
    @test r_auto.pv == r_dir.pv && r_auto.history == r_dir.history

    # ---- (b) above the threshold: default resolves to :mg -------------------
    qt4 = _g230_uniform(4)                        # 256 leaves
    ss4 = make_semstokes(qt4, 4)
    @test 2 * ss4.su.N + ss4.sp.N == 10_851       # >= 2 000, contrast 1
    LAST_AUTO_FALLBACK[] = (count=0, last_reason="")
    m1 = _g230_run(qt4, 4; maxsteps=1)
    ch4 = LAST_SEMSTOKES_AUTO_CHOICE[]
    @test ch4.choice === :mg
    @test ch4.saddle_dofs == 10_851
    @test ch4.mg_threads == min(8, Threads.nthreads())
    @test occursin("saddle_dofs=10851", ch4.reason)
    @test LAST_AUTO_FALLBACK[].count == 0
    m2 = _g230_run(qt4, 4; maxsteps=1)
    @test m1.theta == m2.theta
    @test m1.ux == m2.ux && m1.uz == m2.uz && m1.history == m2.history

    # ---- (c) explicit paths stay selectable and record :none ---------------
    r_mg = _g230_run(qt2, 4; linear=:mg)
    @test LAST_SEMSTOKES_AUTO_CHOICE[].choice === :none
    @test all(isfinite, r_mg.theta) && all(isfinite, r_mg.ux)
    r_d2 = _g230_run(qt2, 4; linear=:direct)
    @test LAST_SEMSTOKES_AUTO_CHOICE[].choice === :none
    @test r_d2.theta == r_dir.theta

    # ---- (d) Newton requires the direct path, named in the message ----------
    err = try
        _g230_run(qt2, 2; maxsteps=1, pic_mode=:newton)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("pic_mode=:newton requires linear=:direct",
                   sprint(showerror, err))
    rn = _g230_run(qt2, 2; maxsteps=1, linear=:direct, pic_mode=:newton,
                   newton_maxit=3)
    @test rn.steps == 1
    @test all(isfinite, rn.theta)
    @test LAST_SEMSTOKES_AUTO_CHOICE[].choice === :none
end
