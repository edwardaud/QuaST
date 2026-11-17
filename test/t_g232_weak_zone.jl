# G232: the weak-zone multiplier.
using Test, QuaST, LinearAlgebra

# G232 (Phase 0, item 0.3): `Params.eta_multiplier` scales whatever the
# viscosity law returned, and `weak_zone_factor` builds the factor of a straight
# weak zone.  Four things must hold:
#
#   (a) the factor is a pure multiplier: inside the zone `factor`, outside 1, and
#       the distance is measured to the *segment*, not to its ends;
#   (b) `taper > 0` makes it continuous, and it stays within [factor, 1];
#   (c) a uniform factor of 1 reproduces the unscaled run bit for bit, so the
#       default path is untouched;
#   (d) a factor < 1 changes the flow, i.e. the hook is actually wired into the
#       closure the Stokes solve uses.

@testset "G232 weak-zone multiplier" begin
    # (a) geometry: the distance is to the segment
    f = weak_zone_factor((1.0, 0.0), (1.0, 1.0); halfwidth=0.1, factor=0.25)
    @test f(1.0, 0.5) == 0.25             # on the segment
    @test f(1.05, 0.5) == 0.25            # inside
    @test f(1.2, 0.5) == 1.0              # outside
    @test f(1.0, -0.05) == 0.25           # near the end, on the axis
    @test f(1.0, -0.2) == 1.0             # past the end, beyond halfwidth
    @test f(0.0, 0.5) == 1.0              # far in x

    # a diagonal segment: the projection, not the bounding box
    g = weak_zone_factor((0.0, 0.0), (1.0, 1.0); halfwidth=0.05, factor=0.5)
    @test g(0.5, 0.5) == 0.5
    @test g(0.5 + 0.03, 0.5 - 0.03) == 0.5
    @test g(0.5 + 0.5, 0.5 - 0.5) == 1.0

    # (b) the taper is continuous and bounded
    h = weak_zone_factor((0.0, 0.5), (1.0, 0.5); halfwidth=0.1, factor=0.2,
                         taper=0.5)
    @test h(0.5, 0.5) == 0.2
    @test h(0.5, 0.6) == 1.0              # outside
    vals = [h(0.5, 0.5 + 0.001 * k) for k in 0:100]
    @test all(v -> 0.2 <= v <= 1.0, vals)
    @test maximum(abs, diff(vals)) < 0.05     # no jump
    @test vals[1] == 0.2 && vals[end] == 1.0

    # argument checks
    @test_throws ArgumentError weak_zone_factor((0.0, 0.0), (1.0, 1.0);
                                                halfwidth=0.0)
    @test_throws ArgumentError weak_zone_factor((0.0, 0.0), (1.0, 1.0);
                                                halfwidth=0.1, factor=0.0)
    @test_throws ArgumentError weak_zone_factor((0.0, 0.0), (1.0, 1.0);
                                                halfwidth=0.1, factor=1.5)
    @test_throws ArgumentError weak_zone_factor((0.5, 0.5), (0.5, 0.5);
                                                halfwidth=0.1)

    # (c) and (d) through the driver
    function _g232_uniform(level::Int)
        q = Quadtree(level)
        for _ in 1:level, lf in collect(q.leaves)
            refine!(q, lf)
        end
        return q
    end

    qt = _g232_uniform(2)
    sq = make_semquad(qt, 2)
    th = [0.1 * sinpi(x) * sinpi(z) for (x, z) in sq.coords]
    # the law and the wall condition come from `Params`, not from keywords
    kw = (p=2, tstop=2.0e-3, dt0=1.0e-3, dtmax=1.0e-3, dtmin=1.0e-16,
          amr_every=0, ev=0.0, linear=:direct, maxpic=1, verbose=false,
          mass_form=:boussinesq)

    plain = run_semquad_simulation(qt, Params(Ra=1.0e4, H=0.0, eta_law=:expT,
                                              deta_T=1.0), copy(th); kw...)
    unit = run_semquad_simulation(qt, Params(Ra=1.0e4, H=0.0, eta_law=:expT,
                                             deta_T=1.0,
                                             eta_multiplier=(x, z) -> 1.0),
                                  copy(th); kw...)
    @test plain.theta == unit.theta        # (c) bit for bit
    @test plain.ux == unit.ux

    weak = run_semquad_simulation(qt, Params(Ra=1.0e4, H=0.0, eta_law=:expT,
                                             deta_T=1.0,
                                             eta_multiplier=(x, z) -> 0.1),
                                  copy(th); kw...)
    @test all(isfinite, weak.theta)
    @test maximum(abs, weak.ux - plain.ux) > 0.0      # (d) the hook acts
end
