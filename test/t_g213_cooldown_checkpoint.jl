# G213: the AMR cooldown checkpoint and a bitwise resume.
using Test, QuaST, LinearAlgebra, HDF5, Printf

# G213: persist the `:balanced` AMR cooldown in the checkpoint, and prove that a
# resume is bitwise identical (docs/dev/BUG_LEDGER.md §10.5).
#
# The four parts:
#   1. save/load round trip of a non-trivial `AmrCooldown` (mirror halves of
#      merged pairs, the four children of split leaves, and the cycle counter);
#   2. a small driver case: 10 steps uninterrupted vs 6 steps with an in-loop
#      checkpoint at step 5 + resume to 10.  theta/ux/uz/pv/history must be
#      bitwise identical, and the first AMR cycle after the resume must be the
#      one the uninterrupted run executed;
#   3. the legacy path: a checkpoint without cooldown data must load an empty
#      cooldown and resume exactly as the pre-change code did.  To reproduce an
#      old in-loop file, the fixture also rewrites `dt_prev` to the previous
#      step's `dt` (the old writer stored it before `dt_prev = dt`, which is the
#      separate ordering fix this gate pins).
#
# **2026-11-25 re-fixture.**  The box element's gradient-jump indicator had a
# wrong y-face index until this date; the corrected indicator changed this card's
# trajectory (the old 20-step card went quiet by cycle 20, so its end cooldown
# aged out and stripping the cooldown from the step-10 checkpoint moved nothing).
# The card is now the localized wall-plume IC below with `max_leaves = 520`
# against a 640-leaf start: every cycle must trade, the cooldown is non-empty at
# the end, and stripping it changes the resumed state.  The old fixture's pins
# (row (46,46,46,0), leaves 640, N 2853, hash a579666395604f77) are superseded;
# see docs/dev/BUG_FIX_LOG.md and BUG_AUDIT_INDICATOR.md §4.4.

# ---- helpers ---------------------------------------------------------------
fnv(a) = begin
    h = 0xcbf29ce484222325
    for b in reinterpret(UInt8, a)
        h ⊻= UInt64(b)
        h *= 0x100000001b3
    end
    string(h, base=16, pad=16)
end
fnvall(xs...) = fnv(reinterpret(UInt8, reduce(vcat, [collect(Float64, x) for x in xs])))

function g213_case()
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz,
                eta_b=9.70406052783923433184,
                eta_c=4.1588830833596718565,
                Twall_top=0.0, Twall_bot=1.0, bc=:free)
    qt = Quadtree(7; cells_x=5, cells_y=2, Lx=2.5)
    for _ in 1:3, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2)
    # Localized wall plume (2026-11-25 re-fixture): a theta feature whose
    # gradient jump stays concentrated, so the balanced cycle keeps refining and
    # merging for the whole 10-step window instead of going quiet after cycle 10.
    th = [0.15 * sinpi(y) * exp(-((x - 0.4)^2) / 0.05) +
          0.05 * sinpi(y) * cospi(2.0 * x / qt.Lx) for (x, y) in ss.su.coords]
    for (g, (_, y)) in enumerate(ss.su.coords)
        (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th[g] = 0.0)
    end
    return pr, qt, th
end

function g213_drv(; maxsteps, checkpoint="", restart="", cb=nothing)
    pr, qt, th = g213_case()
    return run_semquad_simulation(qt, pr, th; p=2, tstop=Inf,
                                  dt0=1.0e-4, maxsteps=maxsteps, amr_every=1,
                                  coarsen_frac=0.05, refine_frac=0.2,
                                  max_level=7,
                                  max_leaves=520, linear=:direct,
                                  symmetry=:x,
                                  verbose=false, cfl=0.5,
                                  checkpoint_path=checkpoint,
                                  checkpoint_every=5,
                                  callback=cb,
                                  restart_from=restart)
end

"Rows of a `/balanced` cycle as a comparable tuple."
cycle_row(d) = (d.refined, d.coarsened, d.merged, d.recirc)

@testset "G213 cooldown checkpoint and bitwise resume" begin
    # ---- 1. save/load round trip of a non-trivial cooldown -----------------
    @testset "cooldown round trip" begin
        pr, qt, th = g213_case()
        ss = make_semstokes(qt, 2)
        eta = semquad_gradient_jump(ss.su, qt, th)
        rows = Set(lf for lf in qt.leaves
                   if lf[3] == 0 || lf[3] == qt.ny0 * (1 << lf[1]) - 1)
        cool = AmrCooldown()
        nr, nc, nex = amr_cycle_balanced!(qt, eta; geom=CartesianBox(qt), refine_frac=0.2,
                                          coarsen_frac=0.05, max_level=7,
                                          max_leaves=520, symmetry=:x,
                                          protect=rows, cooldown=cool,
                                          coarsen_gate=true)
        d = LAST_AMR_BALANCED[]
        @printf("  round trip: refined=%d coarsened=%d merged=%d cycle=%d nref=%d ncoa=%d\n",
                nr, nc, length(d.parents), cool.cycle,
                length(cool.no_refine), length(cool.no_coarsen))
        @test cool.cycle == 1
        @test !isempty(cool.no_refine)       # merged quartets
        @test !isempty(cool.no_coarsen)      # children of split leaves
        @test !isempty(d.parents)
        # every parent this cycle merged is cooled ...
        @test all(p -> haskey(cool.no_refine, p), d.parents)
        # ... including the mirror half of every merged pair (`_coarsen_pair!`)
        p = first(d.parents)
        pm = QuaST._mirror_leaf(qt, p)
        @test pm == p || (pm in d.parents && haskey(cool.no_refine, pm))
        # a whole split quartet is cooled as well
        kids = Dict{NTuple{3,Int},Int}()
        for k in keys(cool.no_coarsen)
            par = (k[1] - 1, k[2] >> 1, k[3] >> 1)
            kids[par] = get(kids, par, 0) + 1
        end
        @test any(==(4), values(kids))
        # ... and the state round-trips verbatim, including the cycle counter
        n = ss.su.N
        z = zeros(Float64, n)
        zp = zeros(Float64, ss.sp.N)
        path = joinpath(mktempdir(), "g213_roundtrip.h5")
        save_semquad_state(path, qt, th, th, z, z, zp;
                           t=1.25e-4, dt=2.0e-5, dt_prev=1.9e-5, step=7, p=2,
                           cooldown=cool, velocity_stale=true)
        ld = load_semquad_state(path)
        @test ld.cooldown.no_refine == cool.no_refine
        @test ld.cooldown.no_coarsen == cool.no_coarsen
        @test ld.cooldown.cycle == 1
        @test ld.velocity_stale
        @test Set(ld.qt.leaves) == Set(qt.leaves)
        h5open(path, "r") do f
            @test haskey(f, "cooldown_no_refine")
            @test haskey(f, "cooldown_no_coarsen")
            @test Int(HDF5.attrs(f)["cooldown_cycle"]) == 1
            @test size(read(f["cooldown_no_refine"]), 1) == 4
            @test size(read(f["cooldown_no_refine"]), 2) ==
                  length(cool.no_refine)
        end
        # a file written without the kwarg has no cooldown data and loads empty
        path2 = joinpath(mktempdir(), "g213_legacy_roundtrip.h5")
        save_semquad_state(path2, qt, th, th, z, z, zp;
                           t=1.25e-4, dt=2.0e-5, dt_prev=1.9e-5, step=7, p=2)
        h5open(path2, "r") do f
            @test !haskey(f, "cooldown_no_refine")
            @test !haskey(f, "cooldown_no_coarsen")
            @test !haskey(HDF5.attrs(f), "cooldown_cycle")
        end
        ld2 = load_semquad_state(path2)
        @test isempty(ld2.cooldown.no_refine)
        @test isempty(ld2.cooldown.no_coarsen)
        @test ld2.cooldown.cycle == 0
        @test !ld2.velocity_stale
        # an explicitly empty cooldown writes the counter only (still loads empty)
        path3 = joinpath(mktempdir(), "g213_empty_roundtrip.h5")
        save_semquad_state(path3, qt, th, th, z, z, zp;
                           t=1.25e-4, dt=2.0e-5, dt_prev=1.9e-5, step=7, p=2,
                           cooldown=AmrCooldown())
        h5open(path3, "r") do f
            @test !haskey(f, "cooldown_no_refine")
            @test Int(HDF5.attrs(f)["cooldown_cycle"]) == 0
        end
        @test load_semquad_state(path3).cooldown.cycle == 0
        # an empty cooldown that has already been aged keeps its counter
        empty5 = AmrCooldown(Dict{NTuple{3,Int},Int}(),
                             Dict{NTuple{3,Int},Int}(), 5)
        path4 = joinpath(mktempdir(), "g213_empty5_roundtrip.h5")
        save_semquad_state(path4, qt, th, th, z, z, zp;
                           t=1.25e-4, dt=2.0e-5, dt_prev=1.9e-5, step=7, p=2,
                           cooldown=empty5)
        ld4 = load_semquad_state(path4)
        @test isempty(ld4.cooldown.no_refine)
        @test isempty(ld4.cooldown.no_coarsen)
        @test ld4.cooldown.cycle == 5
    end

    # ---- 2. bitwise resume from an in-loop checkpoint ----------------------
    @testset "bitwise resume" begin
        a_cycle = Ref{Any}(nothing)
        cbA = st -> begin
            if st.step == 6
                a_cycle[] = (row=cycle_row(LAST_AMR_BALANCED[]),
                             leaves=length(st.qt.leaves), N=st.sq.N)
            end
        end
        a = g213_drv(maxsteps=10; cb=cbA)
        @test a_cycle[] !== nothing
        @printf("  uninterrupted: theta=%s uxuz=%s pv=%s hist=%s\n",
                fnv(reinterpret(UInt8, a.theta)),
                fnv(reinterpret(UInt8, vcat(a.ux, a.uz))),
                fnv(reinterpret(UInt8, a.pv)),
                fnv(reinterpret(UInt8, reduce(vcat, a.history))))

        path = joinpath(mktempdir(), "g213_resume.h5")
        ck5 = joinpath(dirname(path), "g213_resume_step5.h5")
        cbB = st -> begin
            if st.step == 6
                cp(path, ck5; force=true)
            end
        end
        b1 = g213_drv(maxsteps=6; checkpoint=path, cb=cbB)
        st5 = load_semquad_state(ck5)
        @printf("  checkpoint: step=%d cycle=%d nref=%d ncoa=%d stale=%s\n",
                st5.step, st5.cooldown.cycle,
                length(st5.cooldown.no_refine),
                length(st5.cooldown.no_coarsen), st5.velocity_stale)
        @test st5.step == 5
        @test st5.cooldown.cycle == 5
        @test !isempty(st5.cooldown.no_refine)
        @test !isempty(st5.cooldown.no_coarsen)
        @test !st5.velocity_stale          # in-loop checkpoint stores zeros

        b_cycle = Ref{Any}(nothing)
        cbB2 = st -> begin
            if st.step == 6
                b_cycle[] = (row=cycle_row(LAST_AMR_BALANCED[]),
                             leaves=length(st.qt.leaves), N=st.sq.N)
            end
        end
        b2 = g213_drv(maxsteps=10; restart=ck5, cb=cbB2)
        @test b_cycle[] !== nothing
        @printf("  resume:       theta=%s uxuz=%s pv=%s hist=%s\n",
                fnv(reinterpret(UInt8, b2.theta)),
                fnv(reinterpret(UInt8, vcat(b2.ux, b2.uz))),
                fnv(reinterpret(UInt8, b2.pv)),
                fnv(reinterpret(UInt8, reduce(vcat, b2.history))))
        @test b2.steps == 10
        @test b2.t == a.t
        @test b2.theta == a.theta
        @test b2.ux == a.ux
        @test b2.uz == a.uz
        @test b2.pv == a.pv
        @test Set(b2.qt.leaves) == Set(a.qt.leaves)
        @test b2.history == a.history[6:10]
        # the first AMR cycle after the resume is the uninterrupted one, and the
        # final hysteresis state is the one the uninterrupted run ended with.
        # The end-state check is stronger than the old mid-run `cycle` global
        # observation it replaces: the returned `cooldown` is compared key by key
        # (both dicts, not just one integer), so every cooling leaf and its
        # remaining count has to agree.
        @test b_cycle[].row == a_cycle[].row
        @test b_cycle[].leaves == a_cycle[].leaves
        @test b_cycle[].N == a_cycle[].N
        @test !isempty(a.cooldown.no_refine)
        @test !isempty(a.cooldown.no_coarsen)
        @test a.cooldown.cycle == 10
        @test b2.cooldown.cycle == a.cooldown.cycle
        @test b2.cooldown.no_refine == a.cooldown.no_refine
        @test b2.cooldown.no_coarsen == a.cooldown.no_coarsen
        @printf("  end cooldown: cycle=%d nref=%d ncoa=%d\n",
                a.cooldown.cycle, length(a.cooldown.no_refine),
                length(a.cooldown.no_coarsen))

        # Sensitivity: the same checkpoint with the cooldown stripped diverges.
        nocool = joinpath(dirname(path), "g213_resume_nocool.h5")
        save_semquad_state(nocool, st5.qt, st5.theta, st5.theta_prev,
                           st5.ux, st5.uz, st5.pv;
                           t=st5.t, dt=st5.dt, dt_prev=st5.dt_prev,
                           step=st5.step, p=st5.p, config=st5.config)
        b3 = g213_drv(maxsteps=10; restart=nocool)
        @printf("  no cooldown:  theta=%s hist=%s\n",
                fnv(reinterpret(UInt8, b3.theta)),
                fnv(reinterpret(UInt8, reduce(vcat, b3.history))))
        # the cooldown is load-bearing on the repaired card: without it the
        # resumed cycles re-refine leaves that were just merged (measured
        # cellwise max|dtheta| 4.4e-5 after the 2026-11-25 re-fixture; the old
        # card's stripped arm was bit-identical, which is why it was re-fixtured
        # rather than re-pinned).
        @test b3.theta != a.theta
        @test load_semquad_state(nocool).cooldown.cycle == 0
    end

    # ---- 3. old-checkpoint compatibility (empty cooldown path) -------------
    @testset "legacy checkpoint" begin
        # Build a step-10 in-loop checkpoint exactly as the pre-change writer
        # would have: no cooldown datasets, no velocity marker, and `dt_prev`
        # from the previous step (the old write position).  The state itself is
        # the same bitwise trajectory.
        path = joinpath(mktempdir(), "g213_legacy_src.h5")
        ck5 = joinpath(dirname(path), "g213_legacy_step5.h5")
        cbB = st -> (st.step == 6 && cp(path, ck5; force=true))
        b1 = g213_drv(maxsteps=6; checkpoint=path, cb=cbB)
        st5 = load_semquad_state(ck5)
        @test st5.step == 5
        legacy = joinpath(dirname(path), "g213_legacy.h5")
        save_semquad_state(legacy, st5.qt, st5.theta, st5.theta_prev,
                           st5.ux, st5.uz, st5.pv;
                           t=st5.t, dt=st5.dt,
                           dt_prev=b1.history[4][2],   # the old write position
                           step=st5.step, p=st5.p, config=st5.config)
        ld = load_semquad_state(legacy)
        @test isempty(ld.cooldown.no_refine)
        @test isempty(ld.cooldown.no_coarsen)
        @test ld.cooldown.cycle == 0
        @test !ld.velocity_stale
        h5open(legacy, "r") do f
            @test !haskey(f, "cooldown_no_refine")
            @test !haskey(f, "cooldown_no_coarsen")
        end

        got = Ref{Any}(nothing)
        cbL = st -> begin
            if st.step == 6
                got[] = (row=cycle_row(LAST_AMR_BALANCED[]),
                         leaves=length(st.qt.leaves), N=st.sq.N)
            end
        end
        b = g213_drv(maxsteps=10; restart=legacy, cb=cbL)
        @printf("  legacy resume: row=%s leaves=%d N=%d theta=%s\n",
                got[].row, got[].leaves, got[].N,
                fnv(reinterpret(UInt8, b.theta)))
        # The pre-change semantics on the REPAIRED fixture: an old in-loop
        # checkpoint (no cooldown datasets, `dt_prev` from the previous step)
        # loads an empty cooldown and resumes on the legacy path.  These are the
        # fixed-tree values measured on 2026-11-25; the superseded fixture's
        # pins were row (46,46,46,0), 640 leaves, N=2853, hash
        # a579666395604f77 (buggy indicator, old card).
        @test got[].row == (14, 0, 0, 0)
        @test got[].leaves == 502
        @test got[].N == 2325
        @test fnv(reinterpret(UInt8, b.theta)) == "6a380e31ee4c07f6"
    end
end
