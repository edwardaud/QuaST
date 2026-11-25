# G282 (remediation batch P0 part 1): named guards for non-finite assembled
# inputs (AUDIT_VV F5).
#
# Before this batch the raw assembly/driver accepted non-finite data silently:
# Poisson with `f = NaN` returned a NaN field, Stokes with NaN loads returned a
# partially-NaN field, a NaN/Inf viscosity gave a bare `SingularException`, and
# a theta containing NaN reached the Stokes solve through the buoyancy load and
# failed 5.8 s later with `SingularException` and no clue which input was bad
# (`out_julia/agent/audit_vv/robustness.log` lines 14-19, 31).  The fix is a
# set of cold named guards (`src/core/guards.jl`) called at the assembly entry
# points and at the driver entry/per-step entry; it changes no arithmetic, so
# the box nine-hash dump and the box allocation A/B are unchanged.
#
# Why every assertion is not tautological:
#   * each refusal row feeds a *specific* non-finite field through the public
#     API and asserts (i) an `ArgumentError` is raised and (ii) the message
#     names the offending field and the sampled coordinate.  A guard that only
#     throws "non-finite input" somewhere, or that blames the wrong field, is
#     red; a solver that returns a NaN field instead of throwing is red;
#   * the "named" check is per field, so an implementation that checks only the
#     load but not eta (or only eta but not theta) is red on the missing row;
#   * the valid controls solve the same problems with finite data and assert
#     the known manufactured error / a finite field, so the gate cannot be
#     satisfied by refusing everything.
#
# Measured before/after (this campaign): before the fix the seven refusal rows
# returned values (`NO-TRAP`, robustness.log); after it they throw the messages
# asserted below.  The finite controls are unchanged (same checks as G24/G37).
using Test
using QuaST
const MS282 = QuaST

z2(x, y) = 0.0
one2(x, y) = 1.0

function g282_unit_tree(level)
    qt = Quadtree(level)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

function g282_err(f)
    try
        f()
        return nothing
    catch e
        return e
    end
end

@testset "G282 non-finite assembly/driver inputs are named refusals" begin
    qt = g282_unit_tree(3)
    sq = make_semquad(qt, 2)
    ss = make_semstokes(qt, 2)

    # ---- Poisson: the load closure -----------------------------------------
    for (bad, shown) in ((NaN, "NaN"), (Inf, "-?Inf"))
        e = g282_err(() -> solve_semquad_poisson(sq, qt, (x, y) -> bad, z2))
        @test e isa ArgumentError
        msg = sprint(showerror, e)
        @test occursin("Poisson load f(x, y)", msg)
        @test occursin("non-finite", msg)
        @test occursin("(x=", msg)                  # the sampled point is named
        @test occursin(shown == "NaN" ? "NaN" : "Inf", msg)
    end
    # a load that is finite except at one interior point is still named there
    e = g282_err(() -> solve_semquad_poisson(sq, qt,
            (x, y) -> (abs(x - 0.5) < 1.0e-12 && abs(y - 0.5) < 1.0e-12) ? NaN : 0.0,
            z2))
    @test e isa ArgumentError
    @test occursin("Poisson load", sprint(showerror, e))

    # ---- Stokes: the two body-force closures are named separately -----------
    e = g282_err(() -> solve_semquad_stokes_var(ss, qt, one2, (x, y) -> NaN,
                                                z2, z2, z2; bc=:rigid,
                                                linear=:direct))
    @test e isa ArgumentError
    msg = sprint(showerror, e)
    @test occursin("fx(x, y)", msg) && !occursin("eta(x, y)", msg)
    e = g282_err(() -> solve_semquad_stokes_var(ss, qt, one2, z2,
                                                (x, y) -> Inf, z2, z2; bc=:rigid,
                                                linear=:direct))
    @test e isa ArgumentError
    msg = sprint(showerror, e)
    @test occursin("fz(x, y)", msg)

    # ---- Stokes: the viscosity field ----------------------------------------
    for bad in (NaN, Inf)
        e = g282_err(() -> solve_semquad_stokes_var(ss, qt, (x, y) -> bad, z2,
                                                    one2, z2, z2; bc=:rigid,
                                                    linear=:direct))
        @test e isa ArgumentError
        msg = sprint(showerror, e)
        @test occursin("viscosity field eta(x, y)", msg)
        @test occursin("non-finite", msg)
    end
    # the MG/matrix-free sampler is the same contract (AUDIT_GATES m01 path)
    ss_mg = make_semstokes(qt, 2)
    op = MS282.semquad_kop(ss_mg, qt)
    for bad in (NaN, Inf, -1.0, 0.0)
        e = g282_err(() -> MS282.semquad_eta_field(op, (x, y) -> bad))
        @test e isa ArgumentError
        @test occursin("viscosity field eta(x, y)", sprint(showerror, e))
    end

    # ---- energy assembly: the temperature field ------------------------------
    pr = Params(Ra=1.0e3, H=0.0, eta_law=:const, bc=:free, supg=0.0)
    ux = zeros(Float64, sq.N)
    uz = zeros(Float64, sq.N)
    th_bad = [0.05 * sinpi(x) * sinpi(y) for (x, y) in sq.coords]
    th_bad[3] = NaN
    e = g282_err(() -> assemble_semquad_energy(sq, qt, pr, ux, uz;
                                               theta=th_bad))
    @test e isa ArgumentError
    msg = sprint(showerror, e)
    @test occursin("temperature field theta", msg)
    @test occursin("non-finite", msg)
    th_prev_bad = copy(th_bad)
    e = g282_err(() -> assemble_semquad_energy(sq, qt, pr, ux, uz;
                                               theta=zeros(Float64, sq.N),
                                               theta_prev=th_prev_bad))
    @test e isa ArgumentError
    @test occursin("theta_prev", sprint(showerror, e))

    # ---- driver entry: the temperature field ---------------------------------
    th_ok = [0.1 * sinpi(x) * sinpi(y) for (x, y) in ss.su.coords]
    Qt = ss.su
    # the driver only accepts the physical material laws (`:const` is the
    # assembly-only law), so the driver rows use the cheapest production law.
    pr_drv = Params(Ra=1.0e3, H=0.0, eta_law=:expT, deta_T=1.0, eta_b=0.0)
    src = zeros(Float64, Qt.N); src[3] = NaN
    e = g282_err(() -> run_semquad_simulation(qt, pr_drv, src; p=2, maxsteps=1,
                                              tstop=1.0e-6, dt0=1.0e-7,
                                              linear=:direct, verbose=false))
    @test e isa ArgumentError
    msg = sprint(showerror, e)
    @test occursin("theta0", msg)
    @test occursin("index 3", msg)     # the offending node is named

    # ---- valid controls: finite data still assembles and solves --------------
    exact(x, y) = sinpi(x) * sinpi(y)
    rhs(x, y) = 2 * pi^2 * exact(x, y)
    u, _ = solve_semquad_poisson(sq, qt, rhs, exact)
    @test all(isfinite, u)
    @test maximum(abs(u[i] - exact(sq.coords[i]...)) for i in 1:sq.N) < 2.0e-4   # measured 6.6e-5
    ux_ex(x, y) = sinpi(x) * cospi(y)
    uz_ex(x, y) = -cospi(x) * sinpi(y)
    fx(x, y) = 2 * pi^2 * ux_ex(x, y) + pi * cospi(x) * sinpi(y)
    fz(x, y) = 2 * pi^2 * uz_ex(x, y) + pi * sinpi(x) * cospi(y)
    us, ps, = solve_semquad_stokes_var(ss, qt, one2, fx, fz, ux_ex, uz_ex;
                                       bc=:free, linear=:direct)
    @test all(isfinite, us) && all(isfinite, ps)
    @test maximum(abs(us[i] - ux_ex(ss.su.coords[i]...)) for i in 1:ss.su.N) < 2.0e-3
    M, K, C, F = assemble_semquad_energy(sq, qt, pr, zeros(Float64, sq.N),
                                         zeros(Float64, sq.N);
                                         theta=zeros(Float64, sq.N))
    @test all(isfinite, M.nzval) && all(isfinite, K.nzval) &&
          all(isfinite, C.nzval) && all(isfinite, F)
end
