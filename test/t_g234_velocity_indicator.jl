# G234: the AMR indicator can be applied to the velocity field, or to both fields.
using Test, QuaST, LinearAlgebra

# G234 (Phase 1, item 1.1): `amr_indicator` selects which field the gradient-jump
# estimator sees.
#
#   (a) `:theta` is the default and bit-identical to not passing the keyword, so
#       every existing pin keeps its numbers; the velocity estimator is the exact
#       sum of its two components and refuses a wrong-length field; the primitive
#       that the driver forwards the velocity to validates it and really refines.
#   (b) `:velocity` really drives AMR: with `amr_every=1` the tree changes and
#       the resulting tree differs from the theta arm's, and with `amr_every=0`
#       the switch is inert (bit-identical to the default run, tree untouched).
#       `amr_every` is **not** in the shared keyword splat: a splat placed after
#       an explicit keyword overrides it silently in Julia, which is how this
#       gate used to run its "AMR" arm with `amr_every=0` and stay green over a
#       path that threw `UndefVarError: ux` on its first AMR cycle.
#   (c) `:both` normalises each field by its own total before adding
#       (`semquad_combined_jump`): each term carries a total, so neither field
#       dominates by raw magnitude.  It is exactly `:theta` when the flow is at
#       rest, differs from both fields on a state where both have signal, and is
#       invariant to rescaling either field.  The plain sum `jt + jv` was
#       dominated by the velocity term by 13-14 orders of magnitude on a
#       developed state (3.19e14 against 6.11 total) and made `:both` an alias
#       of `:velocity`.
#   (d) anything else is refused before the run starts.

@testset "G234 velocity-gradient AMR indicator" begin
    function _g234_uniform(level::Int)
        # Two levels of headroom: `Quadtree(level)` would cap `maxlevel` at
        # `level`, and then no AMR arm below could refine at all (the gate would
        # again be green over a frozen tree).
        q = Quadtree(level + 2)
        for _ in 1:level, lf in collect(q.leaves)
            refine!(q, lf)
        end
        balance!(q)
        return q
    end

    qt = _g234_uniform(3)
    n0 = length(qt.leaves)
    sq = make_semquad(qt, 2)
    th = [sinpi(x) * sinpi(z) for (x, z) in sq.coords]
    # Re-fixtured 2026-11-25 (audit D12): the old global grid-scale velocity
    # field made the corrected combined 20% head coincide with the theta head,
    # so the "the combined indicator is not theta" property below lost its
    # discriminating power.  Two localized wave packets put the velocity
    # indicator's mass away from the theta indicator's interior peak; the
    # normalised mixture then selects a genuinely mixed head, different from
    # both fields'.
    ux = [sin(6pi * x) * cospi(2z) *
          exp(-40.0 * ((x - 0.25)^2 + (z - 0.30)^2)) for (x, z) in sq.coords]
    uz = [cospi(2x) * sin(6pi * z) *
          exp(-40.0 * ((x - 0.75)^2 + (z - 0.70)^2)) for (x, z) in sq.coords]

    # (a) the estimators themselves
    jv = semquad_velocity_jump(sq, qt, ux, uz)
    jx = semquad_gradient_jump(sq, qt, ux)
    jz = semquad_gradient_jump(sq, qt, uz)
    @test length(jv) == length(sq.leaves)
    @test sum(values(jv)) > 0.0
    @test all(abs.(collect(values(jv)) .- (collect(values(jx)) .+ collect(values(jz)))) .< 1.0e-12)
    @test_throws ArgumentError semquad_velocity_jump(sq, qt, ux[1:end-1], uz)

    # (a1) `semquad_amr_passes!` requires the velocity exactly when the selected
    # indicator reads it; the historical bug read an undefined local instead.
    @test_throws ArgumentError semquad_amr_passes!(
        make_semstokes(qt, 2), deepcopy(qt), copy(th), copy(th);
        indicator=:velocity)
    @test_throws ArgumentError semquad_amr_passes!(
        make_semstokes(qt, 2), deepcopy(qt), copy(th), copy(th);
        indicator=:both, ux=ux, uz=uz[1:end-1])

    # (a2) one real velocity cycle of the primitive refines and transfers.  A
    # second pass runs on the *changed* tree, so it needs the velocity that the
    # first pass carried across the transfer (a length mismatch would throw).
    qv = deepcopy(qt)
    ssq = make_semstokes(qv, 2)
    thv, thpv = copy(th), copy(th)
    ssq, thv, thpv, nrv = semquad_amr_passes!(
        ssq, qv, thv, thpv; indicator=:velocity, ux=ux, uz=uz,
        refine_frac=0.8, coarsen_frac=0.0, max_level=5,
        wall_min_level=0, wall_strip=0.1)
    @test nrv > 0
    @test qv.leaves != qt.leaves
    @test length(thv) == ssq.su.N
    q1 = deepcopy(qt)
    s1 = make_semstokes(q1, 2)
    t1, p1 = copy(th), copy(th)
    s1, t1, p1, _ = semquad_amr_passes!(
        s1, q1, t1, p1; indicator=:velocity, ux=ux, uz=uz,
        refine_frac=0.8, coarsen_frac=0.0, max_level=5,
        wall_min_level=0, wall_strip=0.1)
    q2 = deepcopy(qt)
    s2 = make_semstokes(q2, 2)
    t2, p2 = copy(th), copy(th)
    s2, t2, p2, n2 = semquad_amr_passes!(
        s2, q2, t2, p2; indicator=:velocity, ux=ux, uz=uz,
        refine_frac=0.8, coarsen_frac=0.0, max_level=5,
        wall_min_level=0, wall_strip=0.1, passes=2)
    @test n2 > nrv                                # pass 2 really refined again
    @test length(q2.leaves) > length(q1.leaves)   # therefore on a new tree ...
    @test length(t2) == s2.su.N                   # ... with the transferred field
    @test length(p2) == s2.su.N

    # (b) `:theta` is unchanged, and the AMR switch is wired
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expT, deta_T=1.0)
    th0 = [0.05 * sinpi(x) * sinpi(z) for (x, z) in sq.coords]
    kw = (p=2, tstop=6.0e-3, dt0=2.0e-3, dtmax=2.0e-3, dtmin=1.0e-16,
          ev=0.0, linear=:direct, maxpic=1, verbose=false,
          mass_form=:boussinesq)
    a = run_semquad_simulation(deepcopy(qt), pr, copy(th0); kw...)
    b = run_semquad_simulation(deepcopy(qt), pr, copy(th0);
                               amr_indicator=:theta, kw...)
    @test a.theta == b.theta
    @test a.ux == b.ux
    # with no AMR cycle the indicator is never read: the switch must be inert
    d = run_semquad_simulation(deepcopy(qt), pr, copy(th0);
                               amr_indicator=:velocity, kw...)
    @test d.theta == a.theta
    @test d.ux == a.ux
    @test length(d.qt.leaves) == n0

    # The AMR arms pass `amr_every` explicitly, before the splat (the splat does
    # not contain it, so it cannot win).  These three calls used to throw
    # `UndefVarError: ux` at their first AMR cycle.
    th_amr = run_semquad_simulation(deepcopy(qt), pr, copy(th0);
                                    amr_indicator=:theta, amr_every=1,
                                    refine_frac=0.3, coarsen_frac=0.0,
                                    max_level=4, max_leaves=4000,
                                    wall_min_level=0, wall_strip=0.1, kw...)
    ve_amr = run_semquad_simulation(deepcopy(qt), pr, copy(th0);
                                    amr_indicator=:velocity, amr_every=1,
                                    refine_frac=0.3, coarsen_frac=0.0,
                                    max_level=4, max_leaves=4000,
                                    wall_min_level=0, wall_strip=0.1, kw...)
    bo_amr = run_semquad_simulation(deepcopy(qt), pr, copy(th0);
                                    amr_indicator=:both, amr_every=1,
                                    refine_frac=0.3, coarsen_frac=0.0,
                                    max_level=4, max_leaves=4000,
                                    wall_min_level=0, wall_strip=0.1, kw...)
    @test all(isfinite, th_amr.theta)
    @test all(isfinite, ve_amr.theta)
    @test all(isfinite, bo_amr.theta)
    @test length(th_amr.qt.leaves) > n0            # theta really adapted
    @test length(ve_amr.qt.leaves) > n0            # velocity really adapted
    @test ve_amr.qt.leaves != th_amr.qt.leaves     # ... to a different tree
    @test length(bo_amr.qt.leaves) > n0
    @test bo_amr.qt.leaves != th_amr.qt.leaves
    @test bo_amr.qt.leaves != ve_amr.qt.leaves
    for r in (th_amr, ve_amr, bo_amr)
        @test max_face_level_jump(r.qt) <= 1
        @test maximum(lf[1] for lf in r.qt.leaves) <= 4
        @test length(r.theta) == r.ss.su.N
    end

    # (c) the combined indicator: each field divided by its own total mass, then
    # added.  The rule, literally: jt + (total(jt)/total(jv)) * jv.
    jt = semquad_gradient_jump(sq, qt, th)
    jc = semquad_combined_jump(sq, qt, th, ux, uz)
    total_t = sum(values(jt))
    total_v = sum(values(jv))
    @test total_t > 0.0
    @test total_v > 0.0
    @test all(abs(jc[lf] - (jt[lf] + (total_t / total_v) * jv[lf])) <=
              1.0e-12 * jc[lf] for lf in keys(jt))
    # equal mass from the two fields: total(combined) - total(theta) == total(theta)
    @test isapprox(sum(values(jc)) - total_t, total_t; rtol=1.0e-10)
    # at rest the combined indicator *is* the theta one, exactly: the cycle then
    # cannot collapse the mesh, which is the :velocity failure mode
    zero_u = zeros(sq.N)
    jc0 = semquad_combined_jump(sq, qt, th, zero_u, zero_u)
    @test jc0 == jt

    # The velocity indicator's raw units can dominate by many orders of
    # magnitude (measured 13-14 on the developed Ra=1e8 card).  Rescaling each
    # velocity component by 1e7 -- (1e7)^2 = 1e14 in the quadratic jump
    # indicator -- must leave the normalised combination unchanged, and its
    # selected Dörfler-prefix head must be neither field's.  The old plain sum
    # would have been the velocity indicator to the last printed digit here.
    uxs = 1.0e7 .* ux
    uzs = 1.0e7 .* uz
    jvs = semquad_velocity_jump(sq, qt, uxs, uzs)
    jcs = semquad_combined_jump(sq, qt, th, uxs, uzs)
    @test all(abs(jcs[lf] - jc[lf]) <= 1.0e-12 * jc[lf] for lf in keys(jc))
    k = max(1, round(Int, 0.2 * length(sq.leaves)))
    topk(d) = Set(sort(collect(keys(d)), by=lf -> -d[lf])[1:k])
    # Tie-aware: the scale-invariance claim is a claim about the selected
    # *values*, not about a tie-break at the prefix boundary.  The old exact set
    # equality failed after the 2026-11-25 indicator correction because one
    # boundary leaf `(3,0,0)` swapped with `(3,0,7)` whose eta agreed to
    # round-off -- a tie, not a behaviour change (audit D12).  Comparing the
    # sorted prefix values allows a swap among equal values and still catches a
    # real ranking change (a leaf's value moving across the cut).
    prefix_values(d) = sort([d[lf] for lf in keys(d)]; rev=true)[1:k]
    pv_s, pv_c = prefix_values(jcs), prefix_values(jc)
    @test all(abs(pv_s[i] - pv_c[i]) <=
              1.0e-9 * abs(pv_c[i]) + 1.0e-300 for i in 1:k)
    @test topk(jc) != topk(jt)         # not theta (strengthened fixture)
    @test topk(jc) != topk(jvs)        # not velocity (strengthened fixture)
    plain = copy(jt)
    for (lf, v) in jvs
        plain[lf] += v
    end
    @test topk(plain) != topk(jc)      # the state does discriminate the rules
    @test topk(plain) == topk(jvs)     # (the old plain sum *was* the velocity one)

    # (d) the guard
    for bad in (:bogus, :Theta)
        err = try
            run_semquad_simulation(deepcopy(qt), pr, copy(th0);
                                   amr_indicator=bad, kw...)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("amr_indicator", sprint(showerror, err))
    end
end
