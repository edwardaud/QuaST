# G290 — the annulus same-card checkpoint restart is bitwise.
#
# What this gate pins (fast ring gate; the full 16x192 benchmark is a record,
# not a gate):
#
#   (a) a checkpoint written at step 5 of the 8x96 periodic ring and resumed to
#       step 10 reproduces the uninterrupted run bit for bit: theta, ux, uz, pv,
#       t, steps and the 5 continued history rows.  The resumed mesh is the
#       checkpoint's periodic ring (the file carries the seam topology and the
#       geometry record), so dropping either would change `resumed.qt.seam` or
#       be refused below;
#   (b) the resumed path under `linear=:mg` agrees with the `:direct` run to the
#       MG saddle tolerance (sanity arm: the bitwise restart is not tied to one
#       linear solver);
#   (c) the stored state is load-bearing: a copy of the same checkpoint with
#       `theta_prev` zeroed, and one with `dt` doubled, resume to a *different*
#       state than the unmodified checkpoint.  This is the gate's "the restart
#       must not silently drop a field" sensitivity: the bitwise comparison is
#       not satisfied by a resume path that ignores the BDF2 history or dt;
#   (d) a checkpoint whose geometry record is another geometry is refused by
#       the restart guard (both geometries named), and a checkpoint whose
#       stored tree seam is `:none` is refused rather than run as a half-glued
#       ring (`check_seam_compat`).
#
# The annulus benchmark record (Davies et al. case 2.1 at 16x192) is
# `benchmarks/results/annulus_linear-paths_8x96-16x192.md` and
# `docs/dev/BUG_FIX_LOG.md`; this gate is the cheap regression that the restart
# the continuations rely on keeps being exact.
using Test
using QuaST
using HDF5

const MS290 = QuaST

@testset "G290 annulus same-card checkpoint restart is bitwise" begin
    R0, R1 = 1.22, 2.22
    g = PolarAnnulus(R0, R1, 2pi; seam=:periodic)
    mktempdir() do dir
        # 8x96 periodic ring (768 leaves); the benchmark's 16x192 rule at level 3
        qt = Quadtree(3; cells_x=1, cells_y=12, seam=:periodic)
        for _ in 1:3, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        @test qt.seam === :periodic
        @test length(qt.leaves) == 8 * 96
        ss = make_semstokes(qt, 2; geom=g)
        n0 = ss.su.N
        pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
                    Twall_top=0.0, Twall_bot=1.0, bc=:free,
                    wall_bc=(:free, :free, :free, :free))
        function run_card(pr; maxsteps, ic=:none, restart="", ckpt="",
                          linear=:direct, cb=nothing)
            return run_semquad_simulation(qt, pr, zeros(Float64, n0);
                geom=:annulus, annulus_R0=R0, annulus_R1=R1, annulus_Phi=2pi,
                annulus_seam=:periodic, p=2, tstop=Inf, dt0=1.0e-3,
                dtmax=1.0e-2, dtmin=1.0e-16, cfl=10.0, maxsteps=maxsteps,
                linear=linear, amr_every=0, advection=:galerkin, maxpic=1,
                annulus_ic=ic, checkpoint_path=ckpt, checkpoint_every=0,
                restart_from=restart, verbose=false, callback=cb)
        end

        # ---- (a) same-card restart is bitwise --------------------------------
        ck = joinpath(dir, "g290_ck.h5")
        straight = run_card(pr; maxsteps=10, ic=:aspect)
        first_leg = run_card(pr; maxsteps=5, ic=:aspect, ckpt=ck)
        @test isfile(ck)
        st = load_semquad_state(ck)
        @test st.step == 5
        @test st.qt.seam === :periodic
        @test st.config["geometry"] == geometry_identity(g)
        resumed = run_card(pr; maxsteps=10, restart=ck)
        @test resumed.steps == 10
        @test resumed.theta == straight.theta
        @test resumed.ux == straight.ux
        @test resumed.uz == straight.uz
        @test resumed.pv == straight.pv
        @test resumed.t == straight.t
        @test length(resumed.history) == 5
        @test resumed.history == straight.history[6:10]
        @test Set(resumed.qt.leaves) == Set(straight.qt.leaves)
        @test resumed.qt.seam === :periodic
        @test maximum(abs, first_leg.theta) > 0.0   # the fixture is not the zero state

        # ---- (b) the resumed path under MG agrees with the direct run -------
        # Measured (out_julia/agent/g29_probe2.jl, 5 resumed steps):
        # max|dtheta| = 4.7e-10, max|dux| = 6.9e-7, max|dpv| = 1.3e-4.
        mg = run_card(pr; maxsteps=10, restart=ck, linear=:mg)
        @test mg.steps == 10
        @test maximum(abs, mg.theta - straight.theta) < 1.0e-8
        @test maximum(abs, mg.ux - straight.ux) < 1.0e-5
        @test maximum(abs, mg.uz - straight.uz) < 1.0e-5
        @test maximum(abs, mg.pv - straight.pv) < 1.0e-3
        @test mg.t == resumed.t

        # ---- (c) the stored fields are load-bearing -------------------------
        baseline = run_card(pr; maxsteps=10, restart=ck)
        @test baseline.theta == straight.theta   # non-vacuity of the copies below
        # BDF2 history dropped
        noprev = joinpath(dir, "g290_noprev.h5")
        save_semquad_state(noprev, st.qt, st.theta, copy(st.theta), st.ux, st.uz,
                           st.pv; t=st.t, dt=st.dt, dt_prev=st.dt_prev,
                           step=st.step, p=st.p, config=st.config, ev=st.ev,
                           weak=st.weak, weak_prev=st.weak_prev, txx=st.txx,
                           tyy=st.tyy, txy=st.txy, cooldown=st.cooldown,
                           velocity_stale=st.velocity_stale)
        r_noprev = run_card(pr; maxsteps=10, restart=noprev)
        @test r_noprev.theta != baseline.theta
        @test maximum(abs, r_noprev.theta - baseline.theta) > 1.0e-10
        # dt / dt_prev doubled: the BDF2 ratio of the first resumed step changes
        dt2 = joinpath(dir, "g290_dt2.h5")
        save_semquad_state(dt2, st.qt, st.theta, st.theta_prev, st.ux, st.uz,
                           st.pv; t=st.t, dt=2.0 * st.dt, dt_prev=st.dt_prev,
                           step=st.step, p=st.p, config=st.config, ev=st.ev,
                           weak=st.weak, weak_prev=st.weak_prev, txx=st.txx,
                           tyy=st.tyy, txy=st.txy, cooldown=st.cooldown,
                           velocity_stale=st.velocity_stale)
        r_dt2 = run_card(pr; maxsteps=10, restart=dt2)
        @test maximum(abs, r_dt2.theta - baseline.theta) > 1.0e-10

        # ---- (d) dropping the geometry or the seam is refused --------------
        wedge = PolarAnnulus(R0, R1, 0.5pi; seam=:wall)
        ck_wedge = joinpath(dir, "g290_wedge.h5")
        save_semquad_state(ck_wedge, st.qt, st.theta, st.theta_prev, st.ux,
                           st.uz, st.pv; t=st.t, dt=st.dt,
                           dt_prev=st.dt_prev, step=st.step, p=st.p,
                           config=merge(copy(st.config),
                               Dict{String,Any}("geometry" =>
                                   geometry_identity(wedge))))
        err = try
            run_card(pr; maxsteps=10, restart=ck_wedge)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = err === nothing ? "" : sprint(showerror, err)
        @test occursin("restart configuration mismatch", msg)
        @test occursin("PolarAnnulus", msg) && occursin("seam=:", msg)
        @test occursin("seam=:wall", msg) && occursin("seam=:periodic", msg)
        # the tree seam dropped from the file: the same config, a bounded tree
        ck_seam = joinpath(dir, "g290_seam.h5")
        cp(ck, ck_seam; force=true)
        h5open(ck_seam, "r+") do f
            HDF5.attrs(f)["seam"] = "none"
        end
        @test load_semquad_state(ck_seam).qt.seam === :none
        err2 = try
            run_card(pr; maxsteps=10, restart=ck_seam)
            nothing
        catch e
            e
        end
        @test err2 isa ArgumentError
        msg2 = err2 === nothing ? "" : sprint(showerror, err2)
        @test occursin("seam", lowercase(msg2))
        @test occursin("periodic", lowercase(msg2))
    end
end
