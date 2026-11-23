# G255: the Maxwell viscosity regime is reported at run time, and the card can
# reach it.
#
# Turning viscoelasticity on does not add a memory term to the same problem: the
# 2a closure `eta_eff = 1/(1/eta + 1/(G dt))` is the harmonic mean of `eta` and
# `G dt`, so it caps the viscosity from above and can remove decades of material
# contrast from what the Stokes solve sees (measured on the developed production
# state: material 3.3e6, effective ~5e2 at `G dt = 1.485`, docs/dev/PHASE2.md).
# The card could not set `shear_modulus` at all and the driver said nothing, so
# a reader could believe a viscoelastic card is "the same problem plus memory".
# This gate pins:
#
#   (a) the card knob `CASE_SHEAR_MODULUS` (G241's convention: the card is an
#       example script, so the assertion is on its source), its forwarding into
#       `Params`, its header line, and the warning comment that points at the
#       regime change;
#   (b) `maxwell_regime` on a hand-built field: both ranges and contrasts, the
#       `G*dt/eta` range, the two limits (`G*dt >> eta` keeps the material
#       contrast, `G*dt << eta` collapses it to 1), read-only-ness, and that
#       `G = 0`, `dt = 0` and an empty field are rejected;
#   (c) through the driver: exactly two `Info` records when
#       `shear_modulus > 0` -- one at the first stress step, one at the end of
#       the run -- and every printed material/effective number equals a
#       *direct* recomputation from the state the run actually used (the
#       step-1 field is resampled from `theta0`, the step-2 field from the
#       callback state of step 1).  The tolerance is the printed precision: the
#       gate requires the identical `%.6e` text, i.e. relative agreement to
#       5e-7 (the values are in fact bitwise equal, because both sides go
#       through `semquad_eta_at` and `maxwell_regime`);
#   (d) the report is a pure diagnostic: `regime_report = false` silences both
#       records and leaves every returned field, the history and the stress
#       state bit-identical, so it changes no number;
#   (e) `shear_modulus = 0` prints nothing, has no stress state, and cannot have
#       reached the new computation (`maxwell_regime` throws for `G = 0`, so a
#       completed G = 0 run is itself the proof that the report path did not
#       run); the run is bit-identical to the same run through the off keyword,
#       and it is the same run twice, so the report state carries nothing
#       across runs.  The absolute G = 0 trajectory stays pinned by G150 and the
#       pre-existing G = 0 gates.  The co-rotation report stays a *separate*
#       record (G254): it is still emitted, and neither message carries the
#       other's wording.
#
# The helper functions are at file scope rather than inside the testset: a
# function defined inside a `@testset` is a closure, and an assignment to a name
# that already exists in the testset's local scope would modify that outer
# variable (the shape of G254's accumulator must not depend on where it is
# defined).
using Test, QuaST, Logging, Printf

function g255_tree(nref::Int, base::Int=4)
    qt = Quadtree(base)
    for _ in 1:nref, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

# The material-viscosity field the driver samples for a non-yielding,
# non-composite law: the same public closure (`semquad_eta_at`) at the same
# nodal coordinates.  This is the "same run's viscosity field" the gate
# recomputes from, not an analytic formula.
function g255_eta_field(sq, theta, pr, qt)
    leaves = Set(qt.leaves)
    return [QuaST.semquad_eta_at(sq, theta, x, y, pr, leaves)
            for (x, y) in sq.coords]
end

# Two steps of a temperature-dependent-viscosity card with zero buoyancy and no
# AMR: the velocity is exactly zero, so the only thing the run exercises is the
# Maxwell bookkeeping and its reporting.  `cfl = 0` fixes the step, so the `dt`
# the report quotes is the one passed in.
function g255_run(dt::Float64; nsteps::Int, G::Float64,
                  regime_report::Bool=true, corotation_guard::Bool=true)
    qt = g255_tree(2, 3)          # 16 leaves, not one leaf (HANDOVER section 8)
    ss = make_semstokes(qt, 2)
    pr = Params(; Ra=0.0, H=0.0, eta_law=:expTz, eta_b=log(1.0e4), eta_c=0.0,
                shear_modulus=G, bc=:free)
    # `theta` is the anomaly T - Tbar, so it must vanish on the top and bottom
    # walls; the x variation makes the viscosity field genuinely 2-D.
    theta0 = [0.1 * sinpi(y) * (1.0 + 0.5 * cospi(2.0 * x / 3.0))
              for (x, y) in ss.su.coords]
    caps = NamedTuple[]
    cb = s -> begin
        push!(caps, (step=s.step, dt=s.dt, theta=copy(s.theta), sq=s.sq,
                     qt=s.qt))
        nothing
    end
    logs, res = Test.collect_test_logs(min_level=Logging.Info) do
        run_semquad_simulation(qt, pr, theta0; p=2, maxsteps=nsteps,
                               tstop=1.0e9, dt0=dt, dtmax=dt, dtmin=1.0e-30,
                               cfl=0.0, linear=:direct, ev=0.0, maxpic=1,
                               callback=cb, regime_report=regime_report,
                               corotation_guard=corotation_guard)
    end
    return (logs=logs, r=res, caps=caps, qt=qt, ss=ss, pr=pr, theta0=theta0)
end

g255_regime_msgs(run) =
    [l.message for l in run.logs
     if l.level == Logging.Info && occursin("Maxwell regime", l.message)]
g255_corot_msgs(run) =
    [l.message for l in run.logs
     if l.level == Logging.Info &&
        occursin("Maxwell co-rotation validity", l.message)]

# Every number the report prints, in message order, so the gate cannot silently
# skip one.
const G255_REPORT_FIELDS = (:material_min, :material_max, :material_contrast,
                            :effective_min, :effective_max,
                            :effective_contrast, :gdt_over_eta_min,
                            :gdt_over_eta_max)

# The printed-precision check: each field must appear as its own `%.6e` text.
function g255_assert_numbers(txt::String, r::NamedTuple)
    for f in G255_REPORT_FIELDS
        @test occursin(@sprintf("%.6e", getfield(r, f)), txt)
    end
end

@testset "G255 the Maxwell regime report" begin
    # ---- (a) the card knob ------------------------------------------------
    card = normpath(joinpath(@__DIR__, "..", "examples",
                             "case_realistic_2d.jl"))
    @test isfile(card)
    src = read(card, String)
    # read from the environment, with an empty default (= off)
    @test occursin("CASE_SHEAR_MODULUS", src)
    @test occursin("get(ENV, \"CASE_SHEAR_MODULUS\", \"\")", src)
    @test occursin("isempty(raw)", src)
    @test occursin("isfinite(G) && G >= 0.0", src)
    # forwarded into Params, which is where the driver reads it
    @test occursin("shear_modulus=SHEAR_MODULUS", src)
    # printed in the header line, with its argument (G258 appended the per-wall
    # `wall_un=%s` argument after it, so the printf argument list no longer ends
    # with `, SHEAR_MODULUS)`)
    @test occursin("shear_modulus=%.3g", src)
    @test occursin(", SHEAR_MODULUS,", src)
    @test occursin("wall_un=%s", src)
    # the warning comment near the knob points at the regime change, with the
    # measured numbers of docs/dev/PHASE2.md and the report it points at
    @test occursin("WARNING", src)
    @test occursin("1.485", src)
    @test occursin("5e2", src)
    @test occursin("3.3e6", src)
    @test occursin("free-surface topography", src)
    @test occursin("regime report", src)
    # the library side the card relies on: 0 is off, a positive value is read
    @test Params().shear_modulus == 0.0
    @test Params(shear_modulus=2.5).shear_modulus == 2.5
    @test_throws ArgumentError Params(shear_modulus=-1.0)

    # ---- (b) the regime arithmetic, by hand -------------------------------
    eta = [1.0e-4, 1.0e-2, 1.0, 1.0e2]
    Gdt = 1.0e-2                        # G = 10, dt = 1e-3
    eta_before = copy(eta)
    mr = QuaST.maxwell_regime(eta, 1.0e-3, 10.0)
    @test eta == eta_before             # read-only
    @test mr.material_min == 1.0e-4
    @test mr.material_max == 1.0e2
    @test mr.material_contrast == 1.0e6
    @test mr.effective_min == 1.0 / (1.0 / 1.0e-4 + 1.0 / Gdt)
    @test mr.effective_max == 1.0 / (1.0 / 1.0e2 + 1.0 / Gdt)
    @test mr.effective_contrast == mr.effective_max / mr.effective_min
    @test mr.effective_contrast < mr.material_contrast
    @test mr.gdt_over_eta_min == Gdt / 1.0e2
    @test mr.gdt_over_eta_max == Gdt / 1.0e-4
    @test mr.gdt_over_eta_max == 100.0
    # the two limits: `G*dt >> eta` leaves the material contrast in place,
    # `G*dt << eta` flattens the field onto `G*dt`, i.e. contrast ~ 1.  The
    # asymptotic corrections are `eta_max/(G dt)` and `G dt/eta_min` respectively,
    # which is why the tolerances below are 1e-3 and 1e-4 rather than machine
    # epsilon.
    big_ = QuaST.maxwell_regime(eta, 1.0e3, 1.0e3)      # G*dt = 1e6
    @test isapprox(big_.effective_contrast, big_.material_contrast;
                   rtol=1.0e-3)
    @test big_.effective_contrast > 1.0e5
    small_ = QuaST.maxwell_regime(eta, 1.0e-9, 1.0)     # G*dt = 1e-9
    @test isapprox(small_.effective_contrast, 1.0; rtol=1.0e-4)
    @test small_.effective_contrast < 1.001
    @test isapprox(small_.effective_min, 1.0e-9; rtol=1.0e-4)
    # the disabled convention is rejected, not silently measured
    @test_throws ArgumentError QuaST.maxwell_regime(eta, 1.0e-3, 0.0)
    @test_throws ArgumentError QuaST.maxwell_regime(eta, 0.0, 10.0)
    @test_throws ArgumentError QuaST.maxwell_regime(Float64[], 1.0e-3, 10.0)

    # ---- (c) the driver report, and its numbers ---------------------------
    G = 10.0
    dt = 1.0e-3
    run_on = g255_run(dt; nsteps=2, G=G)
    @test run_on.r.steps == 2
    @test length(run_on.caps) == 2
    msgs = g255_regime_msgs(run_on)
    @test length(msgs) == 2                  # first stress step + end of run
    @test occursin("first stress step, step 1", msgs[1])
    @test occursin("end of run, step 2", msgs[2])
    @test occursin("G=1.000000e+01", msgs[1])
    # exactly one record each, not one per stress step
    @test count(m -> occursin("first stress step", m), msgs) == 1
    @test count(m -> occursin("end of run", m), msgs) == 1
    # the step-1 material field is what the driver sampled: resample it here
    # from `theta0` through the same public closure, and require every printed
    # number to equal the recomputation to the printed precision.
    sq0 = run_on.ss.su
    eta0 = g255_eta_field(sq0, run_on.theta0, run_on.pr, run_on.qt)
    exp1 = QuaST.maxwell_regime(eta0, run_on.caps[1].dt, G)
    @test exp1.material_contrast > 1.0                    # not a 0 == 0 case
    @test exp1.effective_contrast > 1.0
    @test exp1.effective_contrast < exp1.material_contrast
    g255_assert_numbers(msgs[1], exp1)
    # the step-2 material field is the callback state of step 1 (no AMR), which
    # is exactly the start-of-step-2 temperature the driver sampled
    eta1 = g255_eta_field(run_on.caps[1].sq, run_on.caps[1].theta,
                          run_on.pr, run_on.caps[1].qt)
    exp2 = QuaST.maxwell_regime(eta1, run_on.caps[2].dt, G)
    g255_assert_numbers(msgs[2], exp2)
    # what the record *means*, in one line: the measured collapse and the
    # softening, not just two ranges
    collapse = exp1.material_contrast / exp1.effective_contrast
    @test collapse > 1.0
    for m in msgs
        @test occursin(@sprintf("%.3g", collapse), m)
        @test occursin("decades removed", m)
        @test occursin("softer, different problem", m)
        @test occursin("not the same problem plus memory", m)
    end
    # the G254 report is still there, as its own record
    corots = g255_corot_msgs(run_on)
    @test length(corots) == 1
    @test !occursin("Maxwell regime", corots[1])
    @test !occursin("co-rotation validity", msgs[1])
    @printf("G255 (c) first-step report:\n%s\n", msgs[1])
    @printf("G255 (c) end-of-run report:\n%s\n", msgs[2])

    # ---- (d) the report is a pure diagnostic ------------------------------
    run_off = g255_run(dt; nsteps=2, G=G, regime_report=false)
    @test isempty(g255_regime_msgs(run_off))
    @test run_off.r.theta == run_on.r.theta
    @test run_off.r.ux == run_on.r.ux && run_off.r.uz == run_on.r.uz
    @test run_off.r.pv == run_on.r.pv
    @test run_off.r.t == run_on.r.t && run_off.r.steps == run_on.r.steps
    @test run_off.r.history == run_on.r.history
    @test run_off.r.txx == run_on.r.txx && run_off.r.tyy == run_on.r.tyy &&
          run_off.r.txy == run_on.r.txy
    @test length(g255_corot_msgs(run_off)) == 1     # the other report is inert

    # ---- (e) shear_modulus = 0: silent, inert, and the new path unreached --
    run_zero = g255_run(dt; nsteps=2, G=0.0)
    @test isempty(g255_regime_msgs(run_zero))
    @test run_zero.r.txx === nothing && run_zero.r.tyy === nothing &&
          run_zero.r.txy === nothing
    # the report path cannot have run: the measurement itself refuses G = 0,
    # so a completed G = 0 run proves nothing was sampled for a report
    @test_throws ArgumentError QuaST.maxwell_regime(eta0, dt, 0.0)
    # and the keyword is inert at G = 0: same numbers on and off
    run_zero_off = g255_run(dt; nsteps=2, G=0.0, regime_report=false)
    @test isempty(g255_regime_msgs(run_zero_off))
    @test run_zero.r.theta == run_zero_off.r.theta
    @test run_zero.r.ux == run_zero_off.r.ux
    @test run_zero.r.uz == run_zero_off.r.uz
    @test run_zero.r.history == run_zero_off.r.history
    # the report state is local: the same G = 0 run twice is bit-identical
    # (the absolute trajectory is pinned by G150; this pins the wiring)
    run_zero2 = g255_run(dt; nsteps=2, G=0.0)
    @test run_zero2.r.history == run_zero.r.history
    @test run_zero2.r.theta == run_zero.r.theta

    @printf("G255: material contrast %.4e, effective %.4e at G*dt = %.3e; collapse %.3gx (%.2f decades); G = 0 silent and inert\n", exp1.material_contrast, exp1.effective_contrast, G * dt, collapse, log10(collapse))
end
