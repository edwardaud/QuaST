# G231: radiogenic heating that decays with a half-life.
using Test, QuaST, LinearAlgebra

# G231 (Phase 0, item 0.2): `Params.H_half_life` decays the radiogenic source as
# `H * 2^(-t / half_life)`, the form ASPECT's `Radioactive decay` subsection uses
# with per-isotope half-lives.  The decay enters as one multiplicative scale on the
# assembly's source, so three things must hold:
#
#   (a) the assembled source is exactly linear in that scale.  That is the
#       property the feature rests on: no second term appears, only a factor;
#   (b) `H_half_life = Inf` reproduces the constant-`H` assembly bit for bit, so
#       every existing pin keeps its numbers;
#   (c) a driver path that would not apply the scale refuses, and says why.

@testset "G231 radiogenic heating with a half-life" begin
    function _g231_uniform(level::Int)
        q = Quadtree(level)
        for _ in 1:level, lf in collect(q.leaves)
            refine!(q, lf)
        end
        return q
    end

    qt = _g231_uniform(2)
    sq = make_semquad(qt, 2)
    th = zeros(Float64, sq.N)
    ux = zeros(Float64, sq.N)
    uz = zeros(Float64, sq.N)

    pr_const = Params(Ra=1.0e4, H=10.0, rho_heating=true, eta_law=:const)
    pr_inf = Params(Ra=1.0e4, H=10.0, rho_heating=true, eta_law=:const,
                    H_half_life=Inf)

    source(pr, scale) = assemble_semquad_energy(
        sq, qt, pr, ux, uz; rho_heating=true, theta=th,
        heating_scale=scale)[4]

    # (b) the default is the historical path, bit for bit
    F1 = source(pr_const, 1.0)
    F_inf = source(pr_inf, 1.0)
    @test F1 == F_inf
    @test maximum(abs, F1) > 0.0          # the source is really there

    # (a) exact linearity in the scale, at the values a driver produces
    for s in (0.5, 0.25, exp(-log(2.0) * 0.7 / 0.35), 0.0)
        @test source(pr_const, s) == s * F1
    end
    @test source(pr_const, exp(-log(2.0) * 0.0 / 1.0)) == F1   # t = 0

    # (c) the driver refuses a half-life on a path that holds `H` constant, and
    # the message names the parameter.  The driver only accepts the physical
    # laws, so these runs use `:expT` with `deta_T = 1` (constant viscosity).
    pr_decay = Params(Ra=1.0e4, H=10.0, rho_heating=true, eta_law=:expT,
                      deta_T=1.0, H_half_life=0.5)
    err = try
        run_semquad_simulation(
            qt, pr_decay, copy(th);
            p=2, tstop=0.0, dt0=1.0e-3, dtmax=1.0e-3, dtmin=1.0e-16,
            amr_every=0, ev=0.0, linear=:direct, maxpic=1, verbose=false,
            pic_mode=:newton)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("H_half_life", sprint(showerror, err))

    # a plain Picard run with a half-life runs and finishes its two steps
    res = run_semquad_simulation(
        qt, pr_decay, copy(th);
        p=2, tstop=2.0e-3, dt0=1.0e-3, dtmax=1.0e-3, dtmin=1.0e-16,
        amr_every=0, ev=0.0, linear=:direct, maxpic=1, verbose=false)
    @test all(isfinite, res.theta)
    @test res.steps == 2
end
