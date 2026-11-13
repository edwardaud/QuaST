using Test, MantleSEM, LinearAlgebra

# G184 / S3 stage 2 (stage 0): the driver forwards `mg_smoothing`/`mg_omega`.
#
# `run_semquad_simulation` used to forward only `mg_operator`/`mg_vel_solver`, so
# there was no way to roll the tapered V-cycle schedule back from a production
# driver (docs/dev/SOLVER.md §14.9).  This gate pins the new keyword pass-through
# and its two invariants:
#
#   (a) the values actually reach the hierarchy builder -- `make_semquad_mg`
#       records the resolved `(schedule, omega)` in `LAST_SEMSTOKES_MG_PROFILE`,
#       and `:auto` is resolved there, so a driver-level `:taper`/`0.5` must show
#       up verbatim instead of the `:auto` default;
#   (b) the defaults change nothing: on a uniform tree `:auto` resolves to the
#       historical `:uniform` + 2/3, and a run that passes those values
#       explicitly is bitwise identical to a run that passes nothing.  The
#       `linear=:direct` path must ignore the keywords bitwise.

const _G184_Z2 = (x, y) -> 0.0

function _g184_uniform(level::Int; cx::Int=1, cy::Int=1)
    qt = Quadtree(level; cells_x=cx, cells_y=cy)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

function _g184_run(qt; linear::Symbol=:mg, kwargs...)
    p = 2
    ss = make_semstokes(qt, p)
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
                Twall_top=0.0, Twall_bot=1.0, bc=:free)
    theta0 = [0.01 * cos(pi * x) * sin(pi * y) for (x, y) in ss.su.coords]
    for (i, (x, y)) in enumerate(ss.su.coords)
        (y <= 1.0e-14 || y >= 1.0 - 1.0e-14) && (theta0[i] = 0.0)
    end
    return run_semquad_simulation(qt, pr, theta0; p=p, tstop=1.0e-6,
                                  dt0=1.0e-7, maxsteps=2, linear=linear,
                                  verbose=false, kwargs...)
end

@testset "G184 mg_smoothing/mg_omega plumbing" begin
    qt = _g184_uniform(2)                       # 16 leaves, 3 hierarchy levels
    @test length(qt.leaves) == 16

    # (b) default: the historical `:uniform` schedule at omega = 2/3, and an
    # explicit `:uniform`/2/3 is bitwise the same run.
    r_def = _g184_run(qt)
    @test LAST_SEMSTOKES_MG_PROFILE[] == (:uniform, 2.0 / 3.0)
    r_uni = _g184_run(qt; mg_smoothing=:uniform, mg_omega=2.0 / 3.0)
    @test LAST_SEMSTOKES_MG_PROFILE[] == (:uniform, 2.0 / 3.0)
    @test r_uni.theta == r_def.theta
    @test r_uni.history == r_def.history

    # (a) both keywords reach the solve; `omega=nothing` takes the schedule's
    # own default (0.55 for :taper, 2/3 for :uniform).
    _g184_run(qt; mg_smoothing=:taper, mg_omega=0.5)
    @test LAST_SEMSTOKES_MG_PROFILE[] == (:taper, 0.5)
    _g184_run(qt; mg_smoothing=:taper)
    @test LAST_SEMSTOKES_MG_PROFILE[] == (:taper, 0.55)
    _g184_run(qt; mg_smoothing=:uniform)
    @test LAST_SEMSTOKES_MG_PROFILE[] == (:uniform, 2.0 / 3.0)
    _g184_run(qt; mg_smoothing=:uniform, mg_omega=0.4)
    @test LAST_SEMSTOKES_MG_PROFILE[] == (:uniform, 0.4)

    # A different schedule/damping is a different preconditioner, so the
    # iterates (and hence theta) must differ; this catches a "recorded but not
    # forwarded" implementation.
    r_tap = _g184_run(qt; mg_smoothing=:taper, mg_omega=0.5)
    @test maximum(abs.(r_tap.theta .- r_def.theta)) > 0.0

    # (b) direct is untouched, byte for byte, whether or not the new keywords
    # are present (G150 pins the values themselves).
    r_dir = _g184_run(qt; linear=:direct)
    r_dir2 = _g184_run(qt; linear=:direct, mg_smoothing=:taper,
                       mg_omega=0.42)
    @test r_dir.theta == r_dir2.theta
    @test r_dir.history == r_dir2.history

    # Invalid values fail at the driver entry, before any solve.
    @test_throws ArgumentError _g184_run(qt; mg_smoothing=:bogus)
    @test_throws ArgumentError _g184_run(qt; mg_omega=1.5)
end
