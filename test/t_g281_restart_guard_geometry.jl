# G281 (remediation batch P0 part 1): the restart guard compares a checkpoint
# geometry that the live `run_config` does not carry.
#
# AUDIT_BUGS R5: `_restart_guard` compared only keys present on *both* sides
# (`for (k, v) in restart.config; haskey(run_config, k) || continue`), and its
# separate absent-geometry branch only covered the opposite direction (a
# checkpoint that lacks the key).  A checkpoint that carries a geometry record
# therefore resumed silently when a direct `_restart_guard` caller passed a
# config without the key.  The hole is latent through the driver -- whose
# `_run_config` always sets `geometry` -- but the guard's contract is "a resumed
# run cannot silently continue with a different setup", and geometry is exactly
# that kind of setup.
#
# Why every assertion is not tautological:
#   (a) reproduces R5 exactly: a real checkpoint file written with the annulus
#       geometry record, then a direct guard call whose live config has no
#       geometry key.  Before the fix the call returned `restart = <loaded>`
#       (the audit's `battery2.jl` measured that); now it must throw an
#       `ArgumentError` naming `geometry` and the checkpoint identity.  The
#       gate asserts the message text, so a guard that throws for the wrong
#       reason (e.g. a `p` mismatch) is red;
#   (b) the same guard call with the matching geometry record resumes, so the
#       gate is not satisfied by "throw on every checkpoint";
#   (c)/(d) the two previously-shipped refusal directions are re-asserted
#       (checkpoint box + live annulus refused; checkpoint without the record
#       resumed as a box); they pin that the new branch did not widen into the
#       old ones, and (c) also checks that the *mismatch* message names both
#       sides;
#   (e) the reachable driver path is unchanged: a box run that checkpoints after
#       step 1 and resumes to step 3 is bitwise equal to the uninterrupted run
#       (the same construction as G112/G148).  This is the regression row the
#       fix must not disturb.
#
# Measured red-before evidence: on the parent tree the same direct call in (a)
# returned `restart !== nothing` (AUDIT_BUGS R5 / `battery2.log`); the gate's
# `@test_throws` row is red there.  On the fixed tree all rows pass.
using Test
using QuaST
const MS281 = QuaST

z281(x, y) = 0.0

@testset "G281 restart guard: checkpoint geometry vs live run_config" begin
    tmp = mktempdir()
    ck = joinpath(tmp, "g281_ck.h5")
    # --- the annulus fixture: a balanced ring tree with a real (non-box) state.
    g = PolarAnnulus(1.0, 2.0, 2pi; seam=:periodic)
    qt = Quadtree(2; cells_x=1, cells_y=4, seam=:periodic)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2; geom=g)
    th = zeros(Float64, ss.su.N)          # finite, wall-consistent, deterministic
    gid = MS281.geometry_identity(g)
    MS281.save_semquad_state(ck, qt, th, copy(th), zeros(Float64, ss.su.N),
                             zeros(Float64, ss.su.N), zeros(Float64, ss.sp.N);
                             t=0.0, dt=1.0e-3, dt_prev=1.0e-3, step=0, p=2,
                             config=Dict{String,Any}("geometry" => gid))
    st = load_semquad_state(ck)
    @test st.config["geometry"] == gid    # the fixture really carries the record
    @test !haskey(st.config, "physics_revision")   # and no revision key (warns)

    # (a) R5: live config without the geometry key must be refused.
    err = try
        MS281._restart_guard(qt, 2, th, Dict{String,Any}("p" => "2"), ck)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("geometry", msg)
    @test occursin("PolarAnnulus", msg)
    @test occursin("no geometry key", msg)
    @test occursin("restart configuration mismatch", msg)

    # (b) the same checkpoint resumes when the live config carries the record.
    guard = MS281._restart_guard(qt, 2, th,
                                 Dict{String,Any}("p" => "2",
                                                  "geometry" => gid), ck)
    @test guard.restart !== nothing
    @test guard.qt.leaves == qt.leaves
    @test guard.theta0 == th
    @test guard.p == 2

    # (c) the shipped direction: a box-live config must not resume an annulus
    # checkpoint (both sides named).
    err2 = try
        MS281._restart_guard(qt, 2, th,
                             Dict{String,Any}("p" => "2",
                                 "geometry" => MS281.geometry_identity(CartesianBox(qt))),
                             ck)
        nothing
    catch e
        e
    end
    @test err2 isa ArgumentError
    msg2 = sprint(showerror, err2)
    @test occursin("geometry", msg2)
    @test occursin("CartesianBox", msg2) && occursin("PolarAnnulus", msg2)

    # (d) a checkpoint with *no* geometry record is a historical box file: it
    # resumes under a box config and is refused under an annulus config.
    ck_old = joinpath(tmp, "g281_old.h5")
    qtb = Quadtree(2)
    for _ in 1:2, lf in collect(qtb.leaves)
        refine!(qtb, lf)
    end
    balance!(qtb)
    ssb = make_semstokes(qtb, 2)
    thb = zeros(Float64, ssb.su.N)
    MS281.save_semquad_state(ck_old, qtb, thb, copy(thb), zeros(Float64, ssb.su.N),
                             zeros(Float64, ssb.su.N), zeros(Float64, ssb.sp.N);
                             t=0.0, dt=1.0e-3, dt_prev=1.0e-3, step=0, p=2)
    @test !haskey(load_semquad_state(ck_old).config, "geometry")
    ok = MS281._restart_guard(qtb, 2, thb,
                              Dict{String,Any}("p" => "2",
                                  "geometry" => MS281.geometry_identity(CartesianBox(qtb))),
                              ck_old)
    @test ok.restart !== nothing
    err3 = try
        MS281._restart_guard(qtb, 2, thb,
                             Dict{String,Any}("p" => "2", "geometry" => gid), ck_old)
        nothing
    catch e
        e
    end
    @test err3 isa ArgumentError
    @test occursin("key absent", sprint(showerror, err3))

    # (e) the reachable driver path is unchanged: checkpoint mid-run and resume
    # to the same step gives the uninterrupted trajectory bit for bit.
    function run_card(nsteps; restart="", ckpt="")
        q = Quadtree(3)
        for _ in 1:2, lf in collect(q.leaves)
            refine!(q, lf)
        end
        balance!(q)
        s = make_semstokes(q, 2)
        th0 = [0.05 * sinpi(x) * sinpi(y) for (x, y) in s.su.coords]
        pr = Params(Ra=1.0e3, H=0.0, eta_law=:expT, deta_T=1.0, eta_b=0.0)
        dt = 1.0e-5
        return run_semquad_simulation(q, pr, th0;
            p=2, tstop=nsteps * dt, dt0=dt, dtmax=dt, dtmin=1.0e-16,
            maxsteps=nsteps, amr_every=0, linear=:direct, verbose=false,
            restart_from=restart, checkpoint_path=ckpt, checkpoint_every=1)
    end
    ck_mid = joinpath(tmp, "g281_mid.h5")
    full = run_card(3)
    part = run_card(1; ckpt=ck_mid)
    @test isfile(ck_mid)
    resumed = run_card(3; restart=ck_mid)   # the restored step counter is absolute
    @test resumed.theta == full.theta
    @test resumed.ux == full.ux
    @test resumed.uz == full.uz
    @test resumed.pv == full.pv
end
