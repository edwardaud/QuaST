# G294 — the undefined Nusselt pair on a strongly heated mapped card, and the
# opt-in `nu_undefined=:nan` driver keyword.
#
# What this gate pins:
#
#   (a) the *default* driver behaviour is unchanged: the H = 10 ring card still
#       throws the per-wall Nusselt refusal on its first history row, verbatim;
#   (b) with `nu_undefined=:nan` the same card integrates (2-3 steps), the two
#       Nu history columns are NaN, the steady-state criterion never fires (so
#       the run reaches maxsteps), and every state array stays finite;
#   (c) the box path is unaffected: a box card run with the default, with an
#       explicit `:refuse` and with `:nan` produces bit-identical history rows
#       and state (a defined pair calls the usual diagnostic in all three);
#   (d) the warning is emitted exactly once per run, and not at all on a card
#       whose pair is defined;
#   (e) `_nu_pair_defined` pins the analytic limit H_crit (~2.4127 for the
#       1.22/2.22 ring) and the box always-defined branch.
#
# The runs are deliberately tiny (level <= 2 ring trees, 0.1 s per Stokes
# solve); the physics is the cheap isoviscous `:expT` card, because the refusal
# and the NaN plumbing are properties of the geometry/H pair, not of the
# rheology.  The realistic composite card that needs `:nan` in production is
# `examples/annulus_realistic_demo.jl` (its pilot is recorded, not gated).
using Test
using QuaST
using Logging: with_logger
const MS294 = QuaST

const G294_R0 = 1.22
const G294_R1 = 2.22

"""A tiny closed ring tree: level 2 = 4 radial x 8 angular = 32 leaves."""
function g294_ring_tree()
    qt = Quadtree(2; cells_x=1, cells_y=2, seam=:periodic)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

"""The ring card: isoviscous, free slip, the gate's chosen H."""
g294_ring_params(H) = MS294.Params(Ra=1.0e5, H=H, eta_law=:expT, deta_T=1.0,
                                  bc=:free)

"""`theta0 = 0.05 sin(pi h) cos(2 phi)`, vanishing on both circles."""
function g294_theta0(ss, g)
    th = Vector{Float64}(undef, ss.su.N)
    for (i, c) in enumerate(ss.su.coords)
        r = hypot(c[1], c[2])
        h = (r - g.R0) / (g.R1 - g.R0)
        phi = atan(c[2], c[1])
        th[i] = 0.05 * sinpi(h) * cos(2.0 * phi)
        (h <= 1.0e-12 || h >= 1.0 - 1.0e-12) && (th[i] = 0.0)
    end
    return th
end

const G294_RING_KW = (p=2, geom=:annulus, annulus_R0=G294_R0,
                      annulus_R1=G294_R1, annulus_Phi=2.0 * pi,
                      annulus_seam=:periodic, linear=:direct,
                      advection=:galerkin, cfl=0.5, dt0=1.0e-7,
                      dtmax=1.0e-2, dtmin=1.0e-16, maxpic=1)

# A minimal logger that records (level, message) pairs, so the gate can count
# the driver warning without depending on Test's internal logger type.
mutable struct G294Logger <: Base.CoreLogging.AbstractLogger
    messages::Vector{Tuple{Base.CoreLogging.LogLevel,String}}
end
G294Logger() = G294Logger(Tuple{Base.CoreLogging.LogLevel,String}[])
Base.CoreLogging.min_enabled_level(::G294Logger) = Base.CoreLogging.Warn
Base.CoreLogging.shouldlog(::G294Logger, ::Base.CoreLogging.LogLevel,
                           ::Module, ::Symbol, ::Any) = true
Base.CoreLogging.catch_exceptions(::G294Logger) = false
function Base.CoreLogging.handle_message(l::G294Logger,
                                         level::Base.CoreLogging.LogLevel,
                                         message, ::Module, ::Symbol, ::Any,
                                         ::Any, ::Any; kwargs...)
    push!(l.messages, (level, string(message)))
    return nothing
end
g294_nu_warnings(logger) =
    count(m -> occursin("nu_undefined", m[2]), logger.messages)

@testset "G294 undefined mapped Nusselt pair / nu_undefined" begin

    g = MS294.PolarAnnulus(G294_R0, G294_R1, 2.0 * pi; seam=:periodic)
    qt = g294_ring_tree()
    ss = make_semstokes(qt, 2; geom=g)
    theta0 = g294_theta0(ss, g)
    @test length(qt.leaves) == 32
    @test all(isfinite, theta0)

    # ==================================================================
    # (e) the analytic definition of "defined"
    # ==================================================================
    @testset "(e) _nu_pair_defined and the H_crit limit" begin
        # box: always defined, even at H = 10
        qtb = Quadtree(1; Lx=2.5, cells_x=5, cells_y=2)
        gb = MS294.CartesianBox(qtb)
        prb = g294_ring_params(10.0)
        @test MS294._nu_pair_defined(gb, prb)
        # ring: defined up to H_crit = 4/((R1^2-R0^2) - 2 R0^2 ln(R1/R0))
        lnr = log(G294_R1 / G294_R0)
        Hcrit = 4.0 / ((G294_R1^2 - G294_R0^2) - 2.0 * G294_R0^2 * lnr)
        @test isapprox(Hcrit, 2.412661916668996; rtol=1.0e-13)
        @test MS294._nu_pair_defined(g, g294_ring_params(0.0))
        @test MS294._nu_pair_defined(g, g294_ring_params(2.0))
        @test MS294._nu_pair_defined(g, g294_ring_params(Hcrit - 1.0e-6))
        @test !MS294._nu_pair_defined(g, g294_ring_params(Hcrit + 1.0e-6))
        @test !MS294._nu_pair_defined(g, g294_ring_params(10.0))
        # and the refusal it guards, verbatim: H = 10 ring throws, H = 2 works
        for H in (10.0,)
            pr = g294_ring_params(H)
            @test_throws ArgumentError MS294.semquad_nu_walls(
                ss.su, qt, zeros(ss.su.N), pr)
        end
        nut, nub = MS294.semquad_nu_walls(ss.su, qt, zeros(ss.su.N),
                                          g294_ring_params(2.0))
        @test nut == 1.0 && nub == 1.0
    end

    # ==================================================================
    # (a) the default still refuses the H = 10 ring
    # ==================================================================
    @testset "(a) default refusal is unchanged" begin
        pr10 = g294_ring_params(10.0)
        err = try
            run_semquad_simulation(qt, pr10, copy(theta0);
                                   G294_RING_KW..., maxsteps=1, tstop=Inf)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("the annulus wall Nusselt normalisation needs a " *
                       "positive conductive reference flux at both walls", msg)
        @test occursin("H = 10.0", msg)
        # an invalid `nu_undefined` value fails before any solve
        @test_throws ArgumentError run_semquad_simulation(
            qt, pr10, copy(theta0); G294_RING_KW...,
            nu_undefined=:whatever, maxsteps=1, tstop=Inf)
    end

    # ==================================================================
    # (b) + (d) `:nan` integrates, the columns are NaN, the criterion cannot
    # fire, and the warning appears exactly once
    # ==================================================================
    @testset "(b,d) :nan run and the one warning" begin
        pr10 = g294_ring_params(10.0)
        logger = G294Logger()
        res = with_logger(logger) do
            run_semquad_simulation(qt, pr10, copy(theta0);
                                   G294_RING_KW..., maxsteps=3, tstop=Inf,
                                   nu_undefined=:nan,
                                   steady_tol=1.0, steady_window=2, tmin=0.0,
                                   steady_vrms=true)
        end
        # the criterion cannot fire on a NaN Nu: the run reaches maxsteps
        @test res.steps == 3
        @test length(res.history) == 3
        # the two Nu columns are NaN, every other diagnostic is finite
        for row in res.history
            @test length(row) == 18
            @test isnan(row[3]) && isnan(row[4])
            @test isfinite(row[5])   # Vrms
            @test isfinite(row[6])   # meanT
        end
        @test all(isfinite, res.theta)
        @test all(isfinite, res.ux)
        @test all(isfinite, res.uz)
        @test all(isfinite, res.pv)
        # the warning: exactly one, naming the card and H
        @test g294_nu_warnings(logger) == 1
        warnmsg = logger.messages[findfirst(m -> occursin("nu_undefined", m[2]),
                                            logger.messages)][2]
        @test occursin("H=10.0", warnmsg)
        @test occursin("-dTbar/dh", warnmsg)
        @test occursin("NaN", warnmsg)
        # and a defined card run with :nan emits no warning at all
        logger2 = G294Logger()
        _ = with_logger(logger2) do
            run_semquad_simulation(qt, g294_ring_params(2.0), copy(theta0);
                                   G294_RING_KW..., maxsteps=1, tstop=Inf,
                                   nu_undefined=:nan)
        end
        @test g294_nu_warnings(logger2) == 0
    end

    # ==================================================================
    # (c) the box path is bit-identical with the keyword on
    # ==================================================================
    @testset "(c) box invariance" begin
        qtb = Quadtree(2; Lx=2.5, cells_x=5, cells_y=2)
        ssb = make_semstokes(qtb, 2)
        thb = [0.05 * sinpi(c[2]) * cospi(2.0 * c[1] / 2.5)
               for c in ssb.su.coords]
        prb = g294_ring_params(10.0)
        kwb = (p=2, linear=:direct, advection=:galerkin, cfl=0.5,
               dt0=1.0e-7, dtmax=1.0e-2, dtmin=1.0e-16, maxpic=1,
               maxsteps=2, tstop=Inf)
        r_default = run_semquad_simulation(qtb, prb, copy(thb); kwb...)
        r_refuse = run_semquad_simulation(qtb, prb, copy(thb);
                                          kwb..., nu_undefined=:refuse)
        r_nan = run_semquad_simulation(qtb, prb, copy(thb);
                                       kwb..., nu_undefined=:nan)
        @test r_default.history == r_refuse.history == r_nan.history
        @test r_default.theta == r_refuse.theta == r_nan.theta
        @test r_default.ux == r_nan.ux
        for row in r_nan.history
            @test isfinite(row[3]) && isfinite(row[4])
        end
    end
end
