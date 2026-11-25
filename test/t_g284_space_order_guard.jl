# G284 (remediation batch P0 part 1): `p >= 2` at the space constructors
# (AUDIT_VV F7).
#
# Before this batch `make_semquad(qt, 1)` happily built a `Q_1` SemQuad, while
# `make_semstokes(qt, 1)` reached the pressure-space builder at `p - 1 = 0` and
# threw the self-contradictory "p must be >= 1" -- the requirement is `p >= 2`
# (Taylor-Hood `Q_p` velocity with `Q_{p-1}` pressure; `Q_1`/`Q_0` is not
# stable).  The fix moves the `p >= 2` rule into `make_semquad` and into
# `make_semstokes` (both with an actionable message) and keeps the historical
# `Q_{p-1}` pressure space buildable through the internal
# `_make_semquad_space` that `make_semstokes` calls at `p - 1 = 1`.
#
# Why every assertion is not tautological:
#   * the two refusals are the audit's exact probes; the gate asserts the
#     message contains "p >= 2" *and* does not contain the contradictory
#     "must be >= 1".  A fix that only changed the message at the old site
#     would still fail the `make_semquad(qt, 1)` row; a fix that forgot the
#     pressure-space path would fail the `make_semstokes(qt, 2)` row;
#   * "all legitimate p unchanged" is measured, not assumed: for p = 2, 3, 4
#     the public space equals the historical body (`_make_semquad_space`) in
#     every field (N, coords bitwise, leaf/element maps), the velocity degree
#     and the pressure degree are the Taylor-Hood values, and the production
#     p = 2 driver still advances.  A refactor that changed any coordinate
#     fails the bitwise row.
using Test
using QuaST
const MS284 = QuaST

@testset "G284 p >= 2 at the space constructors" begin
    qt = Quadtree(3)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)

    # ---- the audit probes ---------------------------------------------------
    e1 = try
        make_semquad(qt, 1)
        nothing
    catch e
        e
    end
    @test e1 isa ArgumentError
    m1 = sprint(showerror, e1)
    @test occursin("p >= 2", m1)
    @test !occursin("must be >= 1", m1)       # the contradictory message is gone
    e2 = try
        make_semstokes(qt, 1)
        nothing
    catch e
        e
    end
    @test e2 isa ArgumentError
    m2 = sprint(showerror, e2)
    @test occursin("p >= 2", m2)
    @test occursin("Q_1/Q_0", m2)             # names the actual instability
    @test !occursin("must be >= 1", m2)
    # p = 0 / negative are the same class (the public guard, not the GLL one)
    @test_throws ArgumentError make_semquad(qt, 0)
    @test_throws ArgumentError make_semstokes(qt, 0)

    # ---- legitimate degrees are unchanged (bitwise against the historical
    # internal builder, which is the pre-fix body verbatim) -------------------
    for p in 2:4
        pub = make_semquad(qt, p)
        ref = MS284._make_semquad_space(qt, p)
        @test pub.p == p && pub.N == ref.N
        @test pub.coords == ref.coords               # bitwise
        @test pub.leaves == ref.leaves
        @test pub.elem == ref.elem
        @test pub.xi == ref.xi                       # the GLL nodes, bitwise
        s = make_semstokes(qt, p)
        @test s.p == p
        @test s.q == p - 1                           # Taylor-Hood pressure
        @test s.sp.p == p - 1
        @test s.su.N == ref.N
        if p == 2
            # the production pair: Q_2 velocity / Q_1 pressure, 81 pressure
            # dofs on this tree (the historical number; the pressure space is
            # exactly the internal Q_1 builder).
            @test s.sp.N == MS284._make_semquad_space(qt, 1).N
        end
    end

    # ---- the production p = 2 driver still advances -------------------------
    ss = make_semstokes(qt, 2)
    th0 = [0.05 * sinpi(x) * sinpi(y) for (x, y) in ss.su.coords]
    pr = Params(Ra=1.0e3, H=0.0, eta_law=:expT, deta_T=1.0, eta_b=0.0)
    dt = 1.0e-5
    res = run_semquad_simulation(qt, pr, th0; p=2, tstop=2 * dt, dt0=dt,
                                 dtmax=dt, maxsteps=2, amr_every=0,
                                 linear=:direct, verbose=false)
    @test res.steps == 2
    @test all(isfinite, res.theta)
end
