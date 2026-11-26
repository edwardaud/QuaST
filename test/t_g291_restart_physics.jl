# G291 — the physics-aware continuation protocol of the restart guard.
#
# A checkpoint stores the `Params` values that change the trajectory, so a
# restart can no longer silently change the physics: before G290/G291 a
# checkpoint that recorded `Ra = 1e4` resumed into a run with `Ra = 1e5`
# without a word, and the inherited CFL-limited `dt` of the weaker flow blew the
# continuation up (measured max|theta| ~ 30 after 12 steps on the coarse ring
# fixture; the gate's boundedness arm below is the green counterpart).
#
# What this gate pins:
#   (a) without `restart_physics=:allow` a changed `Ra` (and `Twall_bot`) is
#       refused with an `ArgumentError` naming every differing key and its
#       checkpoint/current value and telling the user to pass
#       `restart_physics=:allow`; a numerics change (`cfl`) is refused even
#       with `:allow` (the protocol covers physics parameter values only); an
#       unknown mode symbol is refused up front;
#   (b) with `:allow` the continuation runs: one warning lists the same keys,
#       the returned θ at the stored step equals the checkpoint θ, the first
#       resumed step's dt is exactly `dt0`, and a case-2.1 -> case-2.2
#       continuation over 20 CFL-ramp steps stays bounded (max|theta| below a
#       measured bound);
#   (c) a same-physics restart is unchanged: passing `:allow` explicitly (or
#       the default) still resumes bitwise;
#   (d) a checkpoint written *before* the physics keys existed (the config
#       without them, the historical format) still resumes: the absent keys are
#       skipped, no warning is emitted and the historical silent behaviour is
#       kept (the first step inherits the checkpoint's dt).
#
# The measured numbers in the comments come from `out_julia/agent/g29_probe2.jl`
# (2026-11-26, Julia 1.12.7, WSL); the fixtures are the 8x96 periodic ring of
# G290, so this gate is the cheap regression of the continuation protocol, not a
# benchmark run.
using Test
using QuaST
using Logging
using Printf
using HDF5

const MS291 = QuaST

@testset "G291 restart physics protocol" begin
    R0, R1 = 1.22, 2.22
    g = PolarAnnulus(R0, R1, 2pi; seam=:periodic)
    mktempdir() do dir
        qt = Quadtree(3; cells_x=1, cells_y=12, seam=:periodic)
        for _ in 1:3, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        ss = make_semstokes(qt, 2; geom=g)
        n0 = ss.su.N
        pr21 = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
                      Twall_top=0.0, Twall_bot=1.0, bc=:free,
                      wall_bc=(:free, :free, :free, :free))
        pr22 = Params(Ra=1.0e5, H=0.0, eta_law=:expT, deta_T=1.0,
                      Twall_top=0.0, Twall_bot=1.0, bc=:free,
                      wall_bc=(:free, :free, :free, :free))
        pr22_wall = Params(Ra=1.0e5, H=0.0, eta_law=:expT, deta_T=1.0,
                           Twall_top=0.0, Twall_bot=0.5, bc=:free,
                           wall_bc=(:free, :free, :free, :free))
        function run_card(pr; maxsteps, ic=:none, restart="", ckpt="",
                          dt0=1.0e-3, cfl=10.0, physics=:refuse, cb=nothing)
            return run_semquad_simulation(qt, pr, zeros(Float64, n0);
                geom=:annulus, annulus_R0=R0, annulus_R1=R1, annulus_Phi=2pi,
                annulus_seam=:periodic, p=2, tstop=Inf, dt0=dt0, dtmax=1.0e-2,
                dtmin=1.0e-16, cfl=cfl, maxsteps=maxsteps, linear=:direct,
                amr_every=0, advection=:galerkin, maxpic=1, annulus_ic=ic,
                checkpoint_path=ckpt, checkpoint_every=0, restart_from=restart,
                restart_physics=physics, verbose=false, callback=cb)
        end

        # fixture A: the cfl=10 card, 6-step case-2.1 checkpoint
        ckA = joinpath(dir, "g291_ckA.h5")
        run_card(pr21; maxsteps=6, ic=:aspect, ckpt=ckA)
        stA = load_semquad_state(ckA)
        @test stA.step == 6
        @test haskey(stA.config, "Ra") && haskey(stA.config, "eta_law")
        @test haskey(stA.config, "deta_T") && haskey(stA.config, "bc")
        @test haskey(stA.config, "Twall_top") && haskey(stA.config, "Twall_bot")
        @test stA.config["Ra"] == string(1.0e4)
        # fixture B: the same card with cfl=0 (fixed dt), for the exact-dt arm
        ckB = joinpath(dir, "g291_ckB.h5")
        run_card(pr21; maxsteps=6, ic=:aspect, ckpt=ckB, cfl=0.0)
        stB = load_semquad_state(ckB)
        @test stB.step == 6

        # ================================================================
        # (a) the refusal protocol
        # ================================================================
        err = try
            run_card(pr22; maxsteps=9, restart=ckA)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = err === nothing ? "" : sprint(showerror, err)
        @test occursin("restart configuration mismatch", msg)
        @test occursin("Ra: checkpoint=10000.0, current=100000.0", msg)
        @test occursin("restart_physics=:allow", msg)
        @test occursin("physics parameter values", msg)
        @printf("G291 (a) refusal message:\n%s\n", msg)
        # ... and a second physics key is listed too (both named, both values)
        err2 = try
            run_card(pr22_wall; maxsteps=9, restart=ckA)
            nothing
        catch e
            e
        end
        @test err2 isa ArgumentError
        msg2 = err2 === nothing ? "" : sprint(showerror, err2)
        @test occursin("Ra: checkpoint=10000.0, current=100000.0", msg2)
        @test occursin("Twall_bot: checkpoint=1.0, current=0.5", msg2)
        # a numerics setting is refused in every mode (the protocol is physics)
        err3 = try
            run_card(pr21; maxsteps=9, restart=ckA, cfl=5.0, physics=:allow)
            nothing
        catch e
            e
        end
        @test err3 isa ArgumentError
        @test occursin("cfl: checkpoint=10.0, current=5.0",
                       err3 === nothing ? "" : sprint(showerror, err3))
        # an unknown mode symbol fails before any work
        @test_throws ArgumentError run_card(pr21; maxsteps=1, physics=:maybe)

        # ================================================================
        # (b) the :allow continuation
        # ================================================================
        logs, r0 = Test.collect_test_logs(min_level=Logging.Warn) do
            run_card(pr22; maxsteps=stA.step, restart=ckA, physics=:allow)
        end
        warns = [l for l in logs
                 if l.level == Logging.Warn &&
                    occursin("changed physics", l.message)]
        @test length(warns) == 1
        wmsg = isempty(warns) ? "" : warns[1].message
        @test occursin("restart_physics=:allow", wmsg)
        @test occursin("Ra: checkpoint=10000.0, current=100000.0", wmsg)
        @test occursin("reset", lowercase(wmsg))
        # the no-step arm returns the checkpoint's state under the new physics
        @test r0.steps == stA.step
        @test r0.theta == stA.theta
        @test r0.t == stA.t

        # the first resumed step's dt is exactly dt0 (cfl=0 keeps it fixed)
        first_dt = Ref(NaN)
        s1 = run_card(pr22; maxsteps=stB.step + 1, restart=ckB, physics=:allow,
                      cfl=0.0, dt0=1.0e-4, cb=st -> (first_dt[] = st.dt))
        @test first_dt[] == 1.0e-4
        @test s1.steps == stB.step + 1
        @test maximum(abs, s1.theta) < 1.0

        # the case-2.1 -> case-2.2 CFL-ramp continuation stays bounded.
        # Measured on the 8x96 fixture (out_julia/agent/g29_probe2.jl): over the
        # 20 steps after the restart max|theta| rises from 0.239 (checkpoint) to
        # a peak 0.9565 and ends at 0.7965, while max|u| overshoots to ~5.6e2
        # and decays (finite, bounded).  The unfixed inherited-dt path blew up
        # to max|theta| ~ 30 after 12 steps on the coarse 12-leaf fixture (see
        # the module header); the bound below is the measured peak with a 2x
        # margin, so a real blow-up fails it.
        s20 = run_card(pr22; maxsteps=stA.step + 20, restart=ckA,
                       physics=:allow, cfl=10.0, dt0=1.0e-4)
        @test s20.steps == stA.step + 20
        @test s20.t > stA.t
        @test all(isfinite, s20.theta) && all(isfinite, s20.ux)
        @test maximum(abs, s20.theta) < 2.0
        # the CFL ramp really restarted from dt0 and is growing (5%/step from
        # dt0 = 1e-4: 1.05^20 = 2.65 -> dt = 2.65e-4 after the 20th step)
        @test s20.history[1][2] > 1.0e-4
        @test s20.history[end][2] > 1.5e-4
        @test s20.history[end][2] > s20.history[1][2]

        # ================================================================
        # (c) same-physics restarts are unchanged (bitwise), allow or not
        # ================================================================
        straight = run_card(pr21; maxsteps=10, ic=:aspect, cfl=0.0)
        resumed = run_card(pr21; maxsteps=10, restart=ckB, physics=:allow,
                           cfl=0.0)
        @test resumed.theta == straight.theta
        @test resumed.ux == straight.ux
        @test resumed.uz == straight.uz
        @test resumed.pv == straight.pv
        @test resumed.t == straight.t
        @test resumed.history == straight.history[7:10]

        # ================================================================
        # (d) a checkpoint without the physics keys keeps the historical path
        # ================================================================
        ck_old = joinpath(dir, "g291_legacy.h5")
        cfg = copy(stB.config)
        for k in QuaST._RESTART_PHYSICS_KEYS
            delete!(cfg, k)
        end
        @test !haskey(cfg, "Ra")
        save_semquad_state(ck_old, stB.qt, stB.theta, stB.theta_prev, stB.ux,
                           stB.uz, stB.pv; t=stB.t, dt=stB.dt,
                           dt_prev=stB.dt_prev, step=stB.step, p=stB.p,
                           config=cfg, ev=stB.ev, weak=stB.weak,
                           weak_prev=stB.weak_prev, txx=stB.txx, tyy=stB.tyy,
                           txy=stB.txy, cooldown=stB.cooldown,
                           velocity_stale=stB.velocity_stale)
        legacy_dt = Ref(NaN)
        logs_old, r_old = Test.collect_test_logs(min_level=Logging.Warn) do
            run_card(pr22; maxsteps=stB.step + 2, restart=ck_old, cfl=0.0,
                     dt0=1.0e-4, cb=st -> (legacy_dt[] = st.dt))
        end
        @test r_old.steps == stB.step + 2
        # no physics warning: the file predates the keys and is resumed as before
        @test isempty([l for l in logs_old
                       if l.level == Logging.Warn &&
                          occursin("changed physics", l.message)])
        # ... and no dt reset: the first resumed step inherits the stored dt
        @test legacy_dt[] == stB.dt
    end
end
