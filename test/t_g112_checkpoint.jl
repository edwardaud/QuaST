# G112: SEM checkpoint / exact restart — a restarted run is bit-identical to an uninterrupted one.
using Test, QuaST

@testset "G112 SEM checkpoint / exact restart" begin
    # The checkpoint stores the complete state: quadtree (leaves AND maxlevel),
    # theta, theta_prev, t, dt, dt_prev, step, p.  maxlevel matters because
    # refine! refuses to go below qt.maxlevel, so dropping it silently blocks
    # future refinement.
    #
    # With that stored, a restarted run must be *bit-identical* to an
    # uninterrupted one, because the assembly loops iterate the sorted
    # `sq.leaves` vector, not the tree's Set.
    function run_seg(; nsteps::Int, amr::Int, ckpt::String="", every::Int=0,
                     restart_from::String="", p::Int=2, maxlevel::Int=5,
                     cfl::Float64=0.0, dt0::Float64=1.0e-5, dtmax::Float64=1.0e-5)
        qt = restart_from == "" ? Quadtree(maxlevel) : Quadtree(2)
        if restart_from == ""
            for _ in 1:3, lf in collect(qt.leaves); refine!(qt, lf); end
            balance!(qt)
            ss = make_semstokes(qt, p)
            theta0 = [0.05 * sinpi(x) * sinpi(z) for (x, z) in ss.su.coords]
        else
            theta0 = Float64[]                      # replaced by the checkpoint
        end
        pr = Params(Ra=1.0e6, H=0.0, eta_law=:expTz,
                    eta_b=log(16.0), eta_c=log(4.0), bc=:free)
        dt = dt0
        # with adaptive dt the number of steps is the only stop criterion
        tstop = cfl > 0.0 ? 1.0e9 : nsteps * dt
        # Restart gate: the balanced cycle is the only AMR policy, so nothing to pin.
        run_semquad_simulation(qt, pr, theta0;
            p=p, tstop=tstop, dt0=dt, dtmax=dtmax, dtmin=1.0e-16, cfl=cfl,
            maxsteps=nsteps,
            amr_every=amr, symmetry=:x,
            ev=0.0, advection=:skew, linear=:direct, maxpic=1,
            pic_tol=1.0e-6, verbose=false,
            restart_from=restart_from, checkpoint_path=ckpt,
            checkpoint_every=every)
    end

    # (a) checkpoint on an AMR step, restart, compare with an uninterrupted run
    straight = run_seg(nsteps=10, amr=2)
    @test straight.steps == 10

    tmp = tempname() * ".h5"
    first_half = run_seg(nsteps=5, amr=2, ckpt=tmp, every=1)
    @test isfile(tmp)
    st = load_semquad_state(tmp)
    @test st.step == 5
    @test st.p == 2
    @test st.qt.maxlevel == first_half.qt.maxlevel
    @test st.qt.leaves == first_half.qt.leaves
    @test maximum(abs, st.theta - first_half.theta) == 0.0
    @test abs(st.t - 5 * 1.0e-5) < 1e-18

    restarted = run_seg(nsteps=10, amr=2, restart_from=tmp, ckpt=tmp, every=5)
    @test restarted.steps == 10
    @test restarted.qt.leaves == straight.qt.leaves
    @test maximum(abs, restarted.theta - straight.theta) == 0.0
    @test abs(restarted.t - straight.t) < 1e-18
    @test length(restarted.history) == 5   # only the 5 continued steps
    rm(tmp; force=true)

    # (b) checkpoint between AMR cycles, then a long horizon: must stay exact
    # even though the restart has to refine deeper than its own leaves
    straight2 = run_seg(nsteps=14, amr=2)
    tmp2 = tempname() * ".h5"
    run_seg(nsteps=7, amr=2, ckpt=tmp2, every=1)
    st2 = load_semquad_state(tmp2)
    @test st2.qt.maxlevel == 5                 # deeper leaves still reachable
    mid2 = run_seg(nsteps=14, amr=2, restart_from=tmp2)
    @test mid2.steps == 14
    @test abs(mid2.t - straight2.t) < 1e-18
    @test mid2.qt.leaves == straight2.qt.leaves
    @test maximum(abs, mid2.theta - straight2.theta) == 0.0
    rm(tmp2; force=true)

    # (d) adaptive dt: dt and dt_prev must be restored exactly (BDF2 ratio)
    a4 = run_seg(nsteps=12, amr=2, cfl=0.6, dt0=2.0e-5, dtmax=8.0e-5)
    tmp4 = tempname() * ".h5"
    b4 = run_seg(nsteps=6, amr=2, ckpt=tmp4, every=1, cfl=0.6, dt0=2.0e-5, dtmax=8.0e-5)
    st4 = load_semquad_state(tmp4)
    @test st4.dt == b4.history[end][2]        # stored dt is the last step dt
    c4 = run_seg(nsteps=12, amr=2, restart_from=tmp4, cfl=0.6, dt0=2.0e-5, dtmax=8.0e-5)
    @test c4.steps == 12
    @test c4.qt.leaves == a4.qt.leaves
    @test maximum(abs, c4.theta - a4.theta) == 0.0
    rm(tmp4; force=true)

    # (c) p comes from the checkpoint, not from the caller
    tmp3 = tempname() * ".h5"
    run_seg(nsteps=4, amr=0, ckpt=tmp3, every=1, p=3)
    st3 = load_semquad_state(tmp3)
    @test st3.p == 3
    lo = run_seg(nsteps=8, amr=0, restart_from=tmp3, p=2)   # caller says p=2
    @test lo.ss.su.p == 3
    @test lo.ss.su.N == length(st3.theta)
    rm(tmp3; force=true)

    # (e) regression for the two bugs this gate caught:
    #   - maxlevel was not stored, so refine! refused to go deeper than the
    #     checkpoint's deepest leaf;
    #   - the AMR on the final step was skipped, so a checkpoint written at an
    #     AMR step stored a pre-AMR mesh.
    # p=3, amr_every=3 and a restart exactly on an AMR step exercise both.
    function run_seg3(; nsteps::Int, restart_from::String="", ckpt::String="", every::Int=0)
        qt = restart_from == "" ? Quadtree(6) : Quadtree(2)
        if restart_from == ""
            for _ in 1:3, lf in collect(qt.leaves); refine!(qt, lf); end
            balance!(qt)
            ss = make_semstokes(qt, 3)
            theta0 = [0.1 * sinpi(x) * sinpi(z) for (x, z) in ss.su.coords]
        else
            theta0 = Float64[]
        end
        pr = Params(Ra=1.0e7, H=0.0, eta_law=:expTz,
                    eta_b=log(1024.0), eta_c=log(16.0), bc=:free)
        dt = 2.0e-5
        run_semquad_simulation(qt, pr, theta0;
            p=3, tstop=nsteps * dt, dt0=dt, dtmax=dt, dtmin=1.0e-16,
            amr_every=3, symmetry=:x, refine_frac=0.25,
            max_leaves=4000, max_level=6, ev=0.0, advection=:skew,
            linear=:direct, maxpic=1, pic_tol=1.0e-6, verbose=false,
            restart_from=restart_from, checkpoint_path=ckpt, checkpoint_every=every)
    end
    a5 = run_seg3(nsteps=24)
    tmp5 = tempname() * ".h5"
    run_seg3(nsteps=12, ckpt=tmp5, every=1)     # 12 % 3 == 0 -> AMR step
    st5 = load_semquad_state(tmp5)
    @test st5.qt.maxlevel == 6
    c5 = run_seg3(nsteps=24, restart_from=tmp5)
    @test c5.qt.leaves == a5.qt.leaves
    @test c5.ss.su.N == a5.ss.su.N
    @test maximum(abs, c5.theta - a5.theta) == 0.0
    rm(tmp5; force=true)

    # (f) the run configuration is stored and validated on restart
    tmp6 = tempname() * ".h5"
    run_seg(nsteps=4, amr=2, ckpt=tmp6, every=1)
    st6 = load_semquad_state(tmp6)
    @test !isempty(st6.config)
    @test st6.config["amr_every"] == "2.0"
    @test_throws ArgumentError run_seg(nsteps=6, amr=3, restart_from=tmp6)
    @test run_seg(nsteps=6, amr=2, restart_from=tmp6).steps == 6
    # a checkpoint without a stored config (old file) is still accepted
    old = tempname() * ".h5"
    save_semquad_state(old, st6.qt, st6.theta, st6.theta_prev, st6.ux, st6.uz, st6.pv;
                       t=st6.t, dt=st6.dt, dt_prev=st6.dt_prev, step=st6.step, p=st6.p)
    @test isempty(load_semquad_state(old).config)
    @test run_seg(nsteps=6, amr=2, restart_from=old).steps == 6
    rm(tmp6; force=true); rm(old; force=true)
end