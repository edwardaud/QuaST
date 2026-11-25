# G285 (remediation batch P0 part 1): the driver refuses an invalid time
# window / step card at entry (AUDIT_VV F8).
#
# Before this batch `dt0 < 0` was accepted and the driver *integrated
# backwards* (t = -0.002 after two steps, finite output, no error;
# `robustness.log` line 33); `dtmax < dt0` silently clamped the first step and
# `tstop <= 0` silently ran nothing.  The fix validates `dt0 > 0`,
# `dtmax >= dt0`, `tstop > 0` (except the documented build-only sentinel
# `tstop = 0` with `maxsteps = 0`), and `dtmin > 0`, `maxsteps >= 0`, all before
# any setup work.
#
# Why every assertion is not tautological:
#   * each refusal row is a distinct invalid card and asserts the exception type
#     plus the parameter name in the message; the pre-fix behaviour is recorded
#     in the header (and was a measured NO-TRAP), so these rows are red on the
#     parent;
#   * the `tstop = 0, maxsteps = 0` row is the *opposite* assertion: that
#     documented build-only card must stay accepted (G274 (v) depends on it), so
#     a blunt `tstop > 0` guard is red here;
#   * the valid-card row runs a 2-step box integration and asserts the returned
#     state is finite and the step count/time are the documented ones; the
# "bitwise unchanged" part of the contract is the campaign's nine-hash dump and
# the box allocation A/B, not a re-pin here -- a raw-byte hash would be a
# machine-dependent regression pin
#     and the task forbids re-pinning, so the gate asserts determinism between
#     two identical runs instead.
using Test
using QuaST
const MS285 = QuaST

function g285_case(qt)
    ss = make_semstokes(qt, 2)
    th0 = [0.1 * sinpi(x) * sinpi(y) for (x, y) in ss.su.coords]
    pr = Params(Ra=1.0e3, H=0.0, eta_law=:expT, deta_T=1.0, eta_b=0.0)
    return ss, th0, pr
end

function g285_err(f)
    try
        f()
        return nothing
    catch e
        return e
    end
end

@testset "G285 the driver refuses an invalid time/step card" begin
    qt = Quadtree(3)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss, th0, pr = g285_case(qt)
    base = (p=2, tstop=1.0e-6, dt0=1.0e-7, dtmax=1.0e-7, dtmin=1.0e-16,
            maxsteps=1, linear=:direct, verbose=false)

    # ---- the audit probe: a negative first step must not integrate backwards
    e = g285_err(() -> run_semquad_simulation(qt, pr, th0;
                    base..., dt0=-1.0e-3, dtmax=-1.0e-3))
    @test e isa ArgumentError
    @test occursin("dt0", sprint(showerror, e))
    e = g285_err(() -> run_semquad_simulation(qt, pr, th0; base..., dt0=0.0))
    @test e isa ArgumentError
    @test occursin("dt0", sprint(showerror, e))
    # ---- dtmax < dt0 silently clamped the first step before the fix
    e = g285_err(() -> run_semquad_simulation(qt, pr, th0; base...,
                                              dt0=1.0e-3, dtmax=1.0e-4))
    @test e isa ArgumentError
    @test occursin("dtmax", sprint(showerror, e))
    # ---- negative / zero stop time
    e = g285_err(() -> run_semquad_simulation(qt, pr, th0; base..., tstop=-1.0))
    @test e isa ArgumentError
    @test occursin("tstop", sprint(showerror, e))
    e = g285_err(() -> run_semquad_simulation(qt, pr, th0; base..., tstop=0.0,
                                              maxsteps=2))
    @test e isa ArgumentError
    @test occursin("tstop", sprint(showerror, e))
    # ---- the adjacent invalid controls found while writing the guard
    e = g285_err(() -> run_semquad_simulation(qt, pr, th0; base..., dtmin=0.0))
    @test e isa ArgumentError
    @test occursin("dtmin", sprint(showerror, e))
    e = g285_err(() -> run_semquad_simulation(qt, pr, th0; base..., maxsteps=-1))
    @test e isa ArgumentError
    @test occursin("maxsteps", sprint(showerror, e))
    # `tstop = Inf` is a *supported* sentinel (stop on maxsteps; the adaptive-dt
    # cards use it), so the invalid value next to it is `NaN`.
    e = g285_err(() -> run_semquad_simulation(qt, pr, th0; base..., tstop=NaN))
    @test e isa ArgumentError
    @test occursin("tstop", sprint(showerror, e))

    # ---- the documented build-only sentinel stays accepted (G274 (v) uses it)
    r0 = run_semquad_simulation(qt, pr, th0; p=2, tstop=0.0, dt0=1.0e-3,
                                maxsteps=0, linear=:direct, verbose=false)
    @test r0.steps == 0

    # ---- a valid card is unchanged: it integrates, deterministically --------
    dt = 2.0e-6
    kw = (p=2, tstop=2 * dt, dt0=dt, dtmax=dt, dtmin=1.0e-16, maxsteps=2,
          amr_every=0, linear=:direct, verbose=false)
    ra = run_semquad_simulation(qt, pr, copy(th0); kw...)
    rb = run_semquad_simulation(qt, pr, copy(th0); kw...)
    @test ra.steps == 2
    @test ra.t == 2 * dt
    @test all(isfinite, ra.theta)
    @test ra.theta == rb.theta             # deterministic, same process
    @test ra.ux == rb.ux && ra.pv == rb.pv
end
