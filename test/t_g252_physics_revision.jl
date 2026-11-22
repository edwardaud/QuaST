# G252: the physics revision -- a *behaviour* version of the physics, not a
# parameter value.
#
# `run_config` records the values of a run's parameters, so the restart guard
# can refuse a restart that changes one of them.  It cannot see a change of what
# those values *mean*: `958af5e` projected the Maxwell stress update onto the
# deviatoric strain rate and `8c27a02` added the co-rotational term, and neither
# touched a single parameter value.  A checkpoint written before them with
# `shear_modulus > 0` carries a stress history produced by the older update;
# resuming it silently would continue that history under different physics.
#
# `PHYSICS_REVISION` (`src/physics/revision.jl`, whose comment is the dated
# changelog) is the single source of truth.  This gate pins:
#
#   (a) one definition of the constant in the tree, both behaviour-changing
#       commits in its changelog, and the key stored `run_config` under;
#   (a2) the *source hash* that makes the revision enforceable: the manifest in
#       `src/physics/revision.jl` covers exactly the behaviour-defining sources
#       (the whole of `src/physics/viscoelastic.jl` and the
#       `_advance_maxwell_stress!` function region of `src/driver/run.jl`), the
#       recomputed SHA-256 of each matches the recorded short hash, and the
#       changelog has an entry for every revision up to `PHYSICS_REVISION`.
#       Non-vacuity: the same check run on a temp copy of the tree reports a
#       one-line append to `viscoelastic.jl`, does **not** report an append
#       outside the covered function in `run.jl`, and does report an edit
#       *inside* it -- which is why that file is covered by function region
#       rather than whole file -- and its failure message names the file and
#       says what to do (bump the revision and the hash, add a changelog entry,
#       or refresh the hash alone if the physics did not change);
#   (b) a fresh checkpoint carries `physics_revision = PHYSICS_REVISION`, and a
#       restart within the same revision is bit-identical to the uninterrupted
#       run (theta/ux/uz/pv/history *and* the Maxwell stress history);
#   (c) a checkpoint whose stored revision is edited to another value is
#       refused, and the message names both revisions and says the physics
#       changed between them.  Non-vacuity control: the *same edited file* with
#       the revision restored resumes successfully and bit-identically, so the
#       refusal comes from the revision and not from a corrupted file.  A
#       non-numeric revision is refused too, with the raw value in the message;
#   (d) a checkpoint with the key deleted (the format of every file written
#       before the marker existed) is *resumed with a warning*: a value cannot
#       be refused when it is absent, the affected state (a Maxwell stress
#       history) is named in the warning, and the resumed trajectory is still
#       bit-identical to the uninterrupted one -- for a file that carries a
#       stress history and for one that does not (G = 0), whose warning says it
#       is unaffected.
using Test, QuaST, Logging, Printf, SHA

@testset "G252 physics revision" begin
    # ---- (a) the single source of truth --------------------------------
    @test PHYSICS_REVISION isa Integer
    @test PHYSICS_REVISION >= 2       # revision 1 is the pre-958af5e state
    root = normpath(joinpath(@__DIR__, ".."))
    rev_src = read(joinpath(root, "src", "physics", "revision.jl"), String)
    @test occursin("const PHYSICS_REVISION = $(PHYSICS_REVISION)", rev_src)
    @test occursin("958af5e", rev_src) && occursin("8c27a02", rev_src)
    @test occursin("2026-11-21", rev_src)
    # one definition only: a second one could drift from this gate's number
    defs = String[]
    for (dir, _, fs) in walkdir(joinpath(root, "src"))
        for f in fs
            endswith(f, ".jl") || continue
            p = joinpath(dir, f)
            occursin("const PHYSICS_REVISION", read(p, String)) &&
                push!(defs, relpath(p, root))
        end
    end
    @test length(defs) == 1
    @test defs[1] == joinpath("src", "physics", "revision.jl")
    # and the driver records it in `run_config` next to the other guard keys
    run_src = read(joinpath(root, "src", "driver", "run.jl"), String)
    @test occursin("run_config[\"physics_revision\"] = PHYSICS_REVISION", run_src)

    # ---- (a2) the source hash makes the revision enforceable ---------------
    # `PHYSICS_REVISION` only works if the next physics change remembers to bump
    # it.  `PHYSICS_SOURCE_HASHES` (revision.jl) records a short SHA-256 of the
    # behaviour-defining sources, and this gate recomputes them on every suite
    # run: a change to the Maxwell update cannot land without the revision
    # decision being made visible in revision.jl.
    #
    # The hash rule, identical to the one documented in revision.jl: SHA-256 of
    # the bytes with line endings normalised (`\r\n`, `\r` -> `\n`), truncated to
    # 16 hex digits.  A spec is either a whole file or `path#function_name!`, the
    # latter covering the lines from `function <name>(` through its matching
    # top-level `end`.  Both forms are supported; the shipped manifest hashes
    # whole files, because each covered file now has exactly one subject (the
    # Maxwell kernels in `viscoelastic.jl`, the driver-side stress state in
    # `maxwell_state.jl`).
    g252_normalise(p) = replace(read(p, String), "\r\n" => "\n", "\r" => "\n")
    function g252_function_region(path, marker)
        lines = split(g252_normalise(path), '\n')
        i = findfirst(l -> startswith(l, marker), lines)
        i === nothing && return nothing
        j = findnext(l -> l == "end", lines, i + 1)
        j === nothing && return nothing
        return join(lines[i:j], "\n") * "\n"
    end
    g252_short(s) = bytes2hex(sha256(codeunits(s)))[1:16]
    # `spec` is `path` for a whole file or `path#function_name!` for a region.
    function g252_source_text(tree, spec)
        parts = split(spec, '#'; limit=2)
        path = joinpath(tree, parts[1])
        isfile(path) || return nothing
        length(parts) == 1 && return g252_normalise(path)
        return g252_function_region(path, "function " * parts[2] * "(")
    end
    # The manifest is the truth to check against; the expected list is pinned
    # here so coverage cannot be shortened by deleting an entry from it.
    const_covered = String[first(h) for h in QuaST.PHYSICS_SOURCE_HASHES]
    @test const_covered == ["src/physics/viscoelastic.jl",
                            "src/driver/maxwell_state.jl"]
    @test length(unique(const_covered)) == length(const_covered)
    @test all(h -> last(h) isa String && occursin(r"^[0-9a-f]{16}$", last(h)),
              QuaST.PHYSICS_SOURCE_HASHES)
    function g252_changes(tree)
        changed = Tuple{String,String,String}[]
        for (spec, stored) in QuaST.PHYSICS_SOURCE_HASHES
            txt = g252_source_text(tree, spec)
            current = txt === nothing ? "<source or function missing>" :
                      g252_short(txt)
            current == stored || push!(changed, (spec, stored, current))
        end
        return changed
    end
    function g252_hash_message(changed)
        io = IOBuffer()
        print(io, "the physics revision source hash is stale: a ",
                  "behaviour-defining source of the Maxwell update changed.\n")
        for (spec, stored, current) in changed
            print(io, "  ", spec, "\n      recorded ", stored,
                      "\n      current  ", current, "\n")
        end
        print(io, """
        Decide whether a stored Maxwell stress history means something different now:
          * if it does: bump PHYSICS_REVISION in src/physics/revision.jl, replace
            PHYSICS_SOURCE_HASHES with the current values below, and add a dated
            changelog entry for the new revision in the same file -- all in one
            commit;
          * if it does not (comment/docstring/formatting edit): replace the hash
            and leave the revision alone.
        Current values (paste into PHYSICS_SOURCE_HASHES):
        """)
        for (spec, _, current) in changed
            print(io, "    \"", spec, "\" => \"", current, "\",\n")
        end
        return String(take!(io))
    end
    # the real check: no covered source may differ from its recorded hash
    changed = g252_changes(root)
    if !isempty(changed)
        error(g252_hash_message(changed))
    end
    @test isempty(changed)
    # the changelog carries an entry for the current revision (so bumping the
    # constant without documenting it fails here)
    revs = sort([parse(Int, m.captures[1])
                 for m in eachmatch(r"(?m)^# Revision (\d+)", rev_src)])
    @test revs == collect(1:Int(PHYSICS_REVISION))
    log_first = match(Regex("(?m)^# Revision $(Int(PHYSICS_REVISION)) "), rev_src)
    @test log_first !== nothing

    # non-vacuity: the detector must report a changed covered source and stay
    # quiet for an identical tree and for a change *outside* the covered region.
    mktempdir() do tmp
        mkpath(joinpath(tmp, "src", "physics"))
        mkpath(joinpath(tmp, "src", "driver"))
        cp(joinpath(root, "src", "physics", "viscoelastic.jl"),
           joinpath(tmp, "src", "physics", "viscoelastic.jl"))
        tmp_run = joinpath(tmp, "src", "driver", "run.jl")
        cp(joinpath(root, "src", "driver", "run.jl"), tmp_run)
        tmp_state = joinpath(tmp, "src", "driver", "maxwell_state.jl")
        cp(joinpath(root, "src", "driver", "maxwell_state.jl"), tmp_state)
        # an unchanged copy is not reported (the detector is not always-fail)
        @test isempty(g252_changes(tmp))
        # a change to a driver file that is *not* covered is not reported
        open(tmp_run, "a") do io
            write(io, "\n# G252 non-vacuity sentinel: unrelated driver edit\n")
        end
        @test isempty(g252_changes(tmp))
        # ... but any change to the covered state file is, and is named
        state_text = read(tmp_state, String)
        edited = replace(state_text, "tr / 3.0" => "tr / 2.0"; count=1)
        @test edited != state_text
        write(tmp_state, edited)
        changed2 = g252_changes(tmp)
        @test length(changed2) == 1
        @test changed2[1][1] == "src/driver/maxwell_state.jl"
        msg2 = g252_hash_message(changed2)
        @test occursin("src/driver/maxwell_state.jl", msg2)
        @test occursin("bump PHYSICS_REVISION", msg2)
        @test occursin("changelog entry", msg2)
        @test occursin(changed2[1][2], msg2) && occursin(changed2[1][3], msg2)
        # the whole covered file is behaviour, structural text included: a bare
        # comment appended to it is reported
        write(tmp_state, state_text)
        @test isempty(g252_changes(tmp))
        open(joinpath(tmp, "src", "physics", "viscoelastic.jl"), "a") do io
            write(io, "\n# G252 non-vacuity sentinel: covered file edit\n")
        end
        changed3 = g252_changes(tmp)
        @test length(changed3) == 1
        @test changed3[1][1] == "src/physics/viscoelastic.jl"
        @printf("G252 (a2): covered %d sources, hash check clean; non-vacuity reports %s\n",
                length(const_covered), changed3[1][1])
    end

    # A tree that is not one leaf (HANDOVER section 8); 16 leaves.
    function g252_tree(nref::Int=2)
        qt = Quadtree(2)
        for _ in 1:nref, lf in collect(qt.leaves)
            refine!(qt, lf)
        end
        balance!(qt)
        return qt
    end
    function g252_ic(ss)
        th = [0.05 * sinpi(y) * cospi(2.0 * x) for (x, y) in ss.su.coords]
        for (i, (x, y)) in enumerate(ss.su.coords)
            (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th[i] = 0.0)
        end
        return th
    end
    # `deta_T = 1` makes :expT exactly eta = 1, so the material viscosity the
    # stress update samples is the known constant 1 (the card of G246/G251).
    g252_pr(; G) = Params(; Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0,
                          shear_modulus=G)
    function g252_run(pr; nsteps::Int, kw...)
        qt = g252_tree()
        ss = make_semstokes(qt, 2)
        th0 = g252_ic(ss)
        run_semquad_simulation(qt, pr, th0; p=2, maxsteps=nsteps, tstop=1.0,
                               dt0=1.0e-4, dtmax=1.0e-4, dtmin=1.0e-30,
                               cfl=0.0, linear=:direct, ev=0.0, maxpic=1, kw...)
    end
    # Write a copy of a checkpoint with its stored config edited.  The state is
    # passed through verbatim, so the copy differs from the original only in the
    # config keys the caller changes: this is what lets the edited file be a
    # valid checkpoint for the non-vacuity control.
    function g252_revise(src::String, dst::String, edit::Function)
        s = load_semquad_state(src)
        cfg = copy(s.config)
        edit(cfg)
        save_semquad_state(dst, s.qt, s.theta, s.theta_prev, s.ux, s.uz, s.pv;
                           t=s.t, dt=s.dt, dt_prev=s.dt_prev, step=s.step,
                           p=s.p, config=cfg, ev=s.ev, weak=s.weak,
                           weak_prev=s.weak_prev, txx=s.txx, tyy=s.tyy,
                           txy=s.txy, cooldown=s.cooldown,
                           velocity_stale=s.velocity_stale)
        return dst
    end
    # Bit-identical comparison of everything a resume must reproduce.  History
    # is compared separately because a restarted run returns only its own rows.
    function g252_fields_equal(a, b)
        return a.theta == b.theta && a.ux == b.ux && a.uz == b.uz &&
               a.pv == b.pv && a.t == b.t && a.steps == b.steps &&
               a.txx == b.txx && a.tyy == b.tyy && a.txy == b.txy
    end

    mktempdir() do dir
        # ---- (b) a fresh checkpoint, same-revision restart ----------------
        prG = g252_pr(G=1.0e3)
        ck = joinpath(dir, "g252_rev2.h5")
        ra = g252_run(prG; nsteps=3, checkpoint_path=ck, checkpoint_every=3)
        @test ra.steps == 3
        ld = load_semquad_state(ck)
        @test haskey(ld.config, "physics_revision")
        @test ld.config["physics_revision"] == string(Float64(PHYSICS_REVISION))
        # the file carries the stress history this revision produced, so the
        # key-less warning below has something to talk about
        @test ld.txx !== nothing && ld.tyy !== nothing && ld.txy !== nothing
        @test maximum(abs, ld.txx) > 0.0

        # `rc` is the uninterrupted trajectory; `rb` resumes from step 3 at the
        # same physics revision.  Everything must be bitwise equal.
        rc = g252_run(prG; nsteps=6)
        rb = g252_run(prG; nsteps=6, restart_from=ck)
        @test g252_fields_equal(rb, rc)
        @test length(rb.history) == 3 && rb.history == rc.history[4:6]
        @test Set(rb.qt.leaves) == Set(rc.qt.leaves)

        # ---- (c) an edited revision is refused ---------------------------
        ck_v1 = joinpath(dir, "g252_rev1.h5")
        g252_revise(ck, ck_v1, cfg -> (cfg["physics_revision"] = "1.0"))
        @test load_semquad_state(ck_v1).config["physics_revision"] == "1.0"
        @test load_semquad_state(ck_v1).theta == ld.theta   # state untouched
        err = try
            g252_run(prG; nsteps=6, restart_from=ck_v1)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = err === nothing ? "" : sprint(showerror, err)
        @test occursin("physics revision 1", msg)
        @test occursin("physics revision $(PHYSICS_REVISION)", msg)
        @test occursin("the physics changed between", msg)
        @test occursin("Refusing the restart", msg)
        @printf("G252 mismatch message:\n%s\n", msg)

        # the non-vacuity control: restore the revision on the *same* file and
        # it resumes exactly like the original, so the refusal above was the
        # revision and not a malformed file
        ck_v1_fixed = joinpath(dir, "g252_rev1_fixed.h5")
        g252_revise(ck_v1, ck_v1_fixed,
                    cfg -> (cfg["physics_revision"] =
                                string(Float64(PHYSICS_REVISION))))
        rfix = g252_run(prG; nsteps=6, restart_from=ck_v1_fixed)
        @test g252_fields_equal(rfix, rc)
        @test rfix.history == rc.history[4:6]

        # a revision that is not a number is refused too, naming the raw value
        ck_bad = joinpath(dir, "g252_rev_unknowable.h5")
        g252_revise(ck, ck_bad, cfg -> (cfg["physics_revision"] = "two"))
        err2 = try
            g252_run(prG; nsteps=6, restart_from=ck_bad)
            nothing
        catch e
            e
        end
        @test err2 isa ArgumentError
        msg2 = err2 === nothing ? "" : sprint(showerror, err2)
        @test occursin("two", msg2) && occursin("revision $(PHYSICS_REVISION)",
                                                msg2)

        # ---- (d) a key-less checkpoint: warn, then resume identically ----
        ck_old = joinpath(dir, "g252_nokey.h5")
        g252_revise(ck, ck_old, cfg -> delete!(cfg, "physics_revision"))
        @test !haskey(load_semquad_state(ck_old).config, "physics_revision")
        @test load_semquad_state(ck_old).txx !== nothing   # stress history kept
        logs, rw = Test.collect_test_logs(min_level=Logging.Warn) do
            g252_run(prG; nsteps=6, restart_from=ck_old)
        end
        warns = [l for l in logs if l.level == Logging.Warn &&
                                    occursin("predates physics revision", l.message)]
        @test length(warns) == 1
        warn_txt = isempty(warns) ? "" : warns[1].message
        @test occursin("carries a Maxwell stress history", warn_txt)
        @test occursin("no `physics_revision` key", warn_txt)
        @printf("G252 key-less (G > 0) warning:\n%s\n", warn_txt)
        # the warning is the only effect: the resumed trajectory is bitwise the
        # uninterrupted one
        @test g252_fields_equal(rw, rc)
        @test rw.history == rc.history[4:6]

        # ---- (e) G = 0: no stress history, the warning says so ----------
        pr0 = g252_pr(G=0.0)
        ck0 = joinpath(dir, "g252_g0.h5")
        g252_run(pr0; nsteps=1, checkpoint_path=ck0, checkpoint_every=1)
        ld0 = load_semquad_state(ck0)
        @test haskey(ld0.config, "physics_revision")
        @test ld0.txx === nothing && ld0.tyy === nothing && ld0.txy === nothing
        ck0_old = joinpath(dir, "g252_g0_nokey.h5")
        g252_revise(ck0, ck0_old, cfg -> delete!(cfg, "physics_revision"))
        logs0, r0 = Test.collect_test_logs(min_level=Logging.Warn) do
            g252_run(pr0; nsteps=1, restart_from=ck0_old)
        end
        warns0 = [l for l in logs0 if l.level == Logging.Warn &&
                                      occursin("predates physics revision", l.message)]
        @test length(warns0) == 1
        warn0_txt = isempty(warns0) ? "" : warns0[1].message
        @test occursin("carries no Maxwell stress history", warn0_txt)
        @printf("G252 key-less (G = 0) warning:\n%s\n", warn0_txt)
        # a G = 0 resume with `maxsteps` equal to the stored step runs no step
        # but must return the loaded state (the warning path does not alter it)
        @test r0.steps == 1 && r0.theta == ld0.theta
    end
end
