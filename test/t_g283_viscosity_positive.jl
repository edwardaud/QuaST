# G283 (remediation batch P0 part 1): the raw solve API refuses a negative or
# non-positive viscosity field (AUDIT_VV F6).
#
# Before this batch `solve_semquad_stokes_var(ss, qt, (x,y) -> -1.0, ...)`
# returned a *finite* field at residual 1.6e-15 (a sign-flipped operator is
# still solvable), `eta = 0` returned a finite field, and `eta = 1e-300` was
# accepted (`robustness.log` lines 22-24).  A negative viscosity is not a
# physical card and the raw API had no contract; the fix refuses every
# non-positive or non-finite sampled value at the assembly point
# (`_require_eta!` in `src/core/guards.jl`), with the offending value and
# coordinate in the message.  Production laws clamp to a positive `eta_min`, so
# no production path changes (asserted below by a driver smoke row).
#
# Why every assertion is not tautological:
#   * the refusals use the exact audit probes (`eta = -1`, `0`) plus the
#     sign-only boundary `-1e-300`, and assert both the exception type and that
#     the message contains the offending value and says "non-positive".  A
#     solver that silently returns a field is red; a guard that throws for a
#     different reason is red;
#   * `eta = +1e-300` stays accepted -- this is deliberate scope: the audit
#     asked for negative/non-positive refusal, and the tiny-positive case is
#     documented here as *accepted* rather than silently changing behaviour;
#   * "a valid field unchanged" is a real known-answer: Stokes is homogeneous of
#     degree zero under a common rescaling of `eta` and the load, so the
#     manufactured solve with `(eta = 2, 2 f)` must equal the one with
#     `(eta = 1, f)` to round-off; and both must reproduce the manufactured
#     field.  A mutation that clamps or rescales a valid eta fails this row;
#   * the production path row runs the driver with `eta_law = :expT` and asserts
#     the returned state is finite and the run advances: the guard must not
#     touch clamped production viscosities.
using Test
using QuaST
const MS283 = QuaST

z2(x, y) = 0.0
one2(x, y) = 1.0

@testset "G283 the raw solve API refuses a non-positive viscosity" begin
    qt = Quadtree(3)
    for _ in 1:3, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    ss = make_semstokes(qt, 2)

    # ---- the audit's probes: -1 and 0 ---------------------------------------
    for bad in (-1.0, 0.0, -1.0e-300)
        e = try
            solve_semquad_stokes_var(ss, qt, (x, y) -> bad, z2, one2, z2, z2;
                                     bc=:rigid, linear=:direct)
            nothing
        catch err
            err
        end
        @test e isa ArgumentError
        msg = sprint(showerror, e)
        @test occursin("non-positive", msg)
        @test occursin(string(bad), msg)          # the offending value is named
        @test occursin("viscosity field eta(x, y)", msg)
    end
    # +1e-300 is positive: still accepted (documented scope, audit F6 wording)
    u_tiny, = solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0e-300, z2, one2,
                                       z2, z2; bc=:rigid, linear=:direct)
    @test all(isfinite, u_tiny)

    # ---- a valid field is unchanged: eta/load homogeneous rescaling ---------
    # Free-slip manufactured Stokes problem (G37's field), so the exact solution
    # is non-trivial: u = (sin pi x cos pi y, -cos pi x sin pi y).
    ux_ex(x, y) = sinpi(x) * cospi(y)
    uz_ex(x, y) = -cospi(x) * sinpi(y)
    p_ex(x, y) = sinpi(x) * sinpi(y)
    fx(x, y) = 2 * pi^2 * ux_ex(x, y) + pi * cospi(x) * sinpi(y)
    fz(x, y) = 2 * pi^2 * uz_ex(x, y) + pi * sinpi(x) * cospi(y)
    u1, p1, = solve_semquad_stokes_var(ss, qt, one2, fx, fz, ux_ex, uz_ex;
                                       bc=:free, linear=:direct)
    u2, p2, = solve_semquad_stokes_var(ss, qt, (x, y) -> 2.0,
                                       (x, y) -> 2 * fx(x, y),
                                       (x, y) -> 2 * fz(x, y), ux_ex, uz_ex;
                                       bc=:free, linear=:direct)
    @test maximum(abs.(u1 - u2)) <= 1.0e-11 * max(maximum(abs, u1), 1.0)
    # the pressure scales oppositely to eta under this rescaling: with
    # (eta -> 2 eta, f -> 2 f) the momentum equation is satisfied by the same u
    # and p -> 2 p (both solves pin the same gauge dof), so p2 == 2 p1.
    @test maximum(abs.(p2 - 2 .* p1)) <= 1.0e-9 * max(maximum(abs, p1), 1.0)
    # and the valid field reproduces the manufactured solution at this resolution
    @test maximum(abs(u1[i] - ux_ex(ss.su.coords[i]...)) for i in 1:ss.su.N) < 2.0e-3

    # ---- production laws stay untouched ------------------------------------
    # A driver run with a clamped material law must still advance: the guard
    # sees only positive eta from the law.
    q = Quadtree(3)
    for _ in 1:2, lf in collect(q.leaves)
        refine!(q, lf)
    end
    balance!(q)
    s = make_semstokes(q, 2)
    th0 = [0.05 * sinpi(x) * sinpi(y) for (x, y) in s.su.coords]
    pr = Params(Ra=1.0e3, H=0.0, eta_law=:expT, deta_T=1.0, eta_b=0.0,
                eta_min=1.0e-8, eta_max=1.0e8)
    dt = 1.0e-5
    res = run_semquad_simulation(q, pr, th0; p=2, tstop=2 * dt, dt0=dt,
                                 dtmax=dt, maxsteps=2, amr_every=0,
                                 linear=:direct, verbose=false)
    @test res.steps == 2
    @test all(isfinite, res.theta)
end
