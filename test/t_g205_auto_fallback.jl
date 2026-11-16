# G205: the linear=:auto MG-to-direct fallback below and above the thresholds.
using Test, QuaST, LinearAlgebra

# G205: `linear=:auto` safety net (S7).
#
# The 8192-leaf realistic case found the hole: at 7498 leaves (N=31251,
# saddle_dofs ~72k) `semquad_choose_linear` switches from `:direct` to the
# matrix-free MG (then the >=60k rule, now the
# >=20k rule), and the MG
# solve on that non-uniform AMR tree returns MG-FGMRES relres=4.33 / a true
# saddle residual of 7.28, so `run_semquad_simulation` aborted at step 75/76
# ('Stokes residual 7.2817 exceeds stokes_tol=0.001').  The safety net:
#
# 1. `:auto` below the selector thresholds still picks `:direct` and is
#    **bitwise** the explicit direct answer, with the fallback counter at 0;
# 2. when the auto-selected MG solve returns a true residual above
#    `auto_fallback_tol`, the solve is redone with `:direct` -- the returned
#    field is bitwise the explicit direct answer, `LAST_AUTO_FALLBACK[].count`
#    increments by exactly one, the stored residual is the direct one, and
#    `LAST_SEMSTOKES_AUTO_CHOICE` still reports the selector's `:mg` decision;
# 3. N failing auto solves -> count == N (one increment per *solve*, not per
#    iteration or per Picard step);
# 4. explicit `linear=:mg` is untouched: a forced-bad MG solve returns its bad
#    answer (residual above the tolerance, so the driver still fails loudly) and
#    does not touch the counter;
# 5. argument-validation errors are not masked by the fallback.
#
# The forced MG failure is `mg_maxit=1` (one FGMRES iteration against a tight
# `mg_tol`), which is cheap, deterministic and hits the same code path as the
# production failure: the outer solve stops with a true residual far above the
# threshold.

const _G205_Z2 = (x, y) -> 0.0
const _G205_ONE = (x, y) -> 1.0

function _g205_uniform(level::Int)
    qt = Quadtree(level)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

function _g205_refined(level::Int; cx::Int=1, cy::Int=1)
    qt = Quadtree(level; cells_x=cx, cells_y=cy)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

_g205_reset!() = (LAST_AUTO_FALLBACK[] = (count=0, last_reason=""); nothing)

# one `:auto` solve that the selector sends to MG but that is forced to stop
# after a single FGMRES iteration -> true residual >> auto_fallback_tol
function _g205_failing_auto(ss, qt, fz; ftol::Float64=1.0e-9)
    return solve_semquad_stokes_var(ss, qt, _G205_ONE, _G205_Z2, fz, _G205_Z2,
                                    _G205_Z2; bc=:free, linear=:auto,
                                    mg_maxit=1, mg_tol=1.0e-14,
                                    auto_fallback_tol=ftol)
end

@testset "G205 auto MG->direct fallback" begin
    z2 = _G205_Z2

    # Uniform L5, p=2: saddle_dofs >= 2_000 with contrast 1, i.e. the selector's
    # low-contrast rule 3 picks MG -- the cheapest tree that reaches the MG
    # branch (the p=4 G191 tree is 4x more expensive per solve).
    qt = _g205_uniform(5)
    ss = make_semstokes(qt, 2)
    dofs = 2 * ss.su.N + ss.sp.N
    @test dofs >= QuaST.SEMQUAD_AUTO_DOFS_CONTRAST_100
    @test semquad_choose_linear(dofs, 1.0) === :mg
    fz = (x, y) -> sin(pi * x) * sin(pi * y)

    # reference answer: the explicit direct solve of the same problem
    _g205_reset!()
    res_d = solve_semquad_stokes_var(ss, qt, _G205_ONE, z2, fz, z2, z2; bc=:free)
    u_d, p_d = res_d[1], res_d[2]
    rd = LAST_SEMSTOKES_RESIDUAL[]
    @test LAST_SEMSTOKES_AUTO_CHOICE[].choice === :none
    @test rd < 1.0e-9
    @test LAST_AUTO_FALLBACK[].count == 0

    # ---- 1. auto -> direct below the thresholds: bitwise, no fallback -----
    # 2b L2 tree with a ~8100x contrast: below both thresholds -> direct.
    qt_s = _g205_refined(2; cx=5, cy=2)
    ss_s = make_semstokes(qt_s, 2)
    eta_hc = (x, y) -> exp(9.0 * (x / qt_s.Lx))
    fz_s = (x, y) -> sin(pi * x / qt_s.Lx) * sin(pi * y)
    _g205_reset!()
    res_a = solve_semquad_stokes_var(ss_s, qt_s, eta_hc, z2, fz_s, z2, z2;
                                     bc=:free, linear=:auto)
    ch_s = LAST_SEMSTOKES_AUTO_CHOICE[]
    @test ch_s.choice === :direct
    @test ch_s.saddle_dofs == 2 * ss_s.su.N + ss_s.sp.N
    @test LAST_AUTO_FALLBACK[].count == 0
    res_ad = solve_semquad_stokes_var(ss_s, qt_s, eta_hc, z2, fz_s, z2, z2;
                                      bc=:free)
    @test res_a[1] == res_ad[1] && res_a[2] == res_ad[2]      # bitwise
    @test LAST_AUTO_FALLBACK[].count == 0

    # ---- 2. auto MG with a true residual above the tolerance --------------
    _g205_reset!()
    res_f = _g205_failing_auto(ss, qt, fz)
    ch_f = LAST_SEMSTOKES_AUTO_CHOICE[]
    fb = LAST_AUTO_FALLBACK[]
    @test ch_f.choice === :mg                    # the decision that was overridden
    @test ch_f.saddle_dofs == dofs
    @test fb.count == 1
    @test occursin("true residual", fb.last_reason)
    @test occursin("saddle_dofs=$dofs", fb.last_reason)
    @test res_f[1] == u_d && res_f[2] == p_d     # bitwise the direct answer
    @test LAST_SEMSTOKES_RESIDUAL[] == rd        # the stored residual is direct's
    @test LAST_SEMSTOKES_RESIDUAL[] <= 1.0e-9
    @info "G205 fallback" dofs = dofs residual = LAST_SEMSTOKES_RESIDUAL[] reason = fb.last_reason

    # ---- 3. exactly one increment per failing solve -----------------------
    _g205_reset!()
    for i in 1:3
        _g205_failing_auto(ss, qt, fz)
        @test LAST_AUTO_FALLBACK[].count == i
    end
    # a normal auto MG solve (converged) does not increment the counter
    solve_semquad_stokes_var(ss, qt, _G205_ONE, z2, fz, z2, z2; bc=:free,
                             linear=:auto)
    @test LAST_AUTO_FALLBACK[].count == 3

    # ---- 4. explicit :mg is untouched: bad answer returned, no fallback ---
    _g205_reset!()
    res_mg = solve_semquad_stokes_var(ss, qt, _G205_ONE, z2, fz, z2, z2;
                                      bc=:free, linear=:mg,
                                      mg_operator=:matrixfree, mg_threads=1,
                                      mg_maxit=1, mg_tol=1.0e-14)
    @test LAST_AUTO_FALLBACK[].count == 0
    @test LAST_SEMSTOKES_AUTO_CHOICE[].choice === :none
    @test LAST_SEMSTOKES_RESIDUAL[] > 1.0e-5     # the bad MG answer is NOT fixed
    @test !(res_mg[1] == u_d && res_mg[2] == p_d)

    # ---- 5. validation errors are not masked ------------------------------
    _g205_reset!()
    @test_throws ArgumentError solve_semquad_stokes_var(ss, qt, _G205_ONE, z2,
                                                        fz, z2, z2; bc=:free,
                                                        linear=:auto,
                                                        mg_threads=0)
    @test LAST_AUTO_FALLBACK[].count == 0
end
