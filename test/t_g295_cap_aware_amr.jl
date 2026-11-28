# G295 the opt-in cap-aware Dörfler fallback: marking, report and default bitwise.
#
# Fixture T1 (no solver): Quadtree(3) refined to 16 L2 leaves, then one L2 leaf
# refined to four L3 children (19 leaves).  eta = 10 on the four capped L3
# leaves, 1 on the 15 refinable L2 leaves, so the standard 20 % Dörfler prefix
# is entirely L3 and the standard candidate set is empty:
#
#   total = 55, tau_r = 10, cap prefix = 2 leaves (mass 20), candidates 0.
#
# The count-preserving fallback replaces the 2 capped prefix leaves by the top
# 2 refinable leaves (eta 1 each): want = 6, mark_target = 2, and the cycle
# adds 6 leaves (19 -> 25), not a mass-renormalised flood.
#
# The second arm checks that cap_aware=true is inert when max_level cannot bind
# (max_level > qt.maxlevel), when refine_frac=0, and when no refinable leaf
# exists.  The default/omitted keyword must keep the historical (0,0,0) cycle.
using Test
using QuaST

function t295_fixture()
    qt = Quadtree(3)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    refine!(qt, (2, 0, 0))                       # four capped L3 leaves
    eta = Dict{NTuple{3,Int},Float64}()
    for lf in qt.leaves
        eta[lf] = lf[1] == 3 ? 10.0 : 1.0
    end
    return qt, eta
end

@testset "G295 cap-aware Dörfler fallback" begin
    # -- (a) the historical standard cycle must still freeze -----------------
    qt, eta = t295_fixture()
    box = CartesianBox(qt)
    old = copy(qt.leaves)
    r = amr_cycle_balanced!(qt, eta; refine_frac=0.2, coarsen_frac=0.05,
                            max_level=3, max_leaves=typemax(Int),
                            cooldown=nothing, coarsen_gate=true,
                            cap_aware=false, geom=box)
    d = deepcopy(LAST_AMR_BALANCED[])
    @test r == (0, 0, 0)
    @test qt.leaves == old
    @test d.refined == 0 && d.coarsened == 0 && d.exchanged == 0
    @test d.want == 0 && d.eligible == 0
    @test d.tau_r == 10.0 && d.tau_r_all == 10.0
    @test d.cap_aware == false && d.cap_applied == false
    @test d.cap_prefix == 0 && d.cap_mass == 0.0

    # -- (b) fallback marks the exact refinable Dörfler prefix ---------------
    qt, eta = t295_fixture()
    box = CartesianBox(qt)
    old = copy(qt.leaves)
    r = amr_cycle_balanced!(qt, eta; refine_frac=0.2, coarsen_frac=0.05,
                            max_level=3, max_leaves=typemax(Int),
                            cooldown=nothing, coarsen_gate=true,
                            cap_aware=true, geom=box)
    d = deepcopy(LAST_AMR_BALANCED[])
    @test r == (2, 0, 0)
    @test length(qt.leaves) == 25
    @test d.cap_aware == true && d.cap_applied == true
    @test d.cap_prefix == 2
    @test d.cap_mass == 20.0
    @test d.refinable == 15
    @test d.refinable_mass == 15.0
    @test d.tau_r_all == 10.0
    @test d.tau_r == 1.0                 # lowest marked eta
    @test d.mark_target == 2.0           # sum of the two marked eta
    @test d.tau_h == min(d.tau_c, 0.5 * d.tau_r)
    @test d.deadband == (0.5 * d.tau_r < d.tau_c)
    @test d.want == 6
    @test d.refined == 2 && d.coarsened == 0 && d.exchanged == 0
    @test d.nmerge == 19 && d.npost == 25
    @test d.npost - d.nmerge == 3 * d.refined - d.exchanged
    @test d.recirc == 0
    @test max_face_level_jump(qt) <= 1
    added = setdiff(qt.leaves, old)
    @test length(added) == 8           # two splits, four L3 children each
    # every removed leaf is one of the 15 L2 leaves, never a capped L3 leaf
    removed = setdiff(old, qt.leaves)
    @test length(removed) == 2
    @test all(lf -> lf[1] == 2, removed)
    # the two removed L2 leaves are exactly the top two refinable leaves in
    # the deterministic (eta, leaf) order of the fallback prefix
    expected_pfx = sort(filter(lf -> lf[1] == 2, collect(old));
                        by=lf -> (eta[lf], lf), rev=true)[1:2]
    @test Set(removed) == Set(expected_pfx)

    # -- (b2) fallback + budget exchange: the marked leaves are protected ---
    # One capped L3 leaf (eta 10) and a refinable L2 universe with two high
    # eta=0.9 leaves and 13 low eta=0.1 leaves.  The standard prefix is the
    # capped leaf; the fallback marks the highest-eta L2 leaf.  With
    # max_leaves=20 the cycle must coarsen the legal low-eta quartets to fund the
    # mark, so this arm exercises the effective tau_r/tau_h in the coarsen
    # plan and the mark-protection property.
    qt, eta = t295_fixture()
    for lf in keys(eta)
        eta[lf] = lf[1] == 3 ? 10.0 : 0.1
    end
    eta[(2, 3, 3)] = 0.9
    eta[(2, 3, 2)] = 0.9
    marked = (2, 3, 3)
    marked_parent = (1, 1, 1)
    r = amr_cycle_balanced!(qt, eta; refine_frac=0.2, coarsen_frac=0.05,
                            max_level=3, max_leaves=20, cooldown=nothing,
                            coarsen_gate=true, cap_aware=true,
                            geom=CartesianBox(qt))
    d = deepcopy(LAST_AMR_BALANCED[])
    @test r == (1, 2, 0)
    @test d.cap_applied == true
    @test d.cap_prefix == 1
    @test d.want == 3
    @test d.refined == 1 && d.coarsened == 2 && d.exchanged == 0
    @test d.tau_r == 0.9
    @test d.tau_h == min(d.tau_c, 0.5 * d.tau_r)
    @test d.deadband == (0.5 * d.tau_r < d.tau_c)
    @test marked_parent ∉ d.parents     # the marked quartet was not merged
    @test d.recirc == 0
    @test d.npost - d.nmerge == 3 * d.refined - d.exchanged
    @test max_face_level_jump(qt) <= 1

    # -- (b3) cooldown-only emptiness must not fire the cap fallback --------
    qt, eta = t295_fixture()
    for lf in keys(eta)
        eta[lf] = lf[1] == 2 ? 10.0 : 0.1   # refinable mass dominates
    end
    cool = AmrCooldown()
    for lf in qt.leaves
        lf[1] == 2 && (cool.no_refine[lf] = 1)
    end
    old = copy(qt.leaves)
    r = amr_cycle_balanced!(qt, eta; refine_frac=0.2, coarsen_frac=0.05,
                            max_level=3, max_leaves=typemax(Int),
                            cooldown=cool, hysteresis_cycles=1,
                            coarsen_gate=true, cap_aware=true,
                            geom=CartesianBox(qt))
    d = deepcopy(LAST_AMR_BALANCED[])
    @test r == (0, 0, 0)
    @test qt.leaves == old
    @test d.cap_applied == false

    # -- (c) default/omitted keyword and non-binding cap are inert ----------
    for kwargs in ((;), (; cap_aware=false))
        qt, eta = t295_fixture()
        old = copy(qt.leaves)
        r = amr_cycle_balanced!(qt, eta; refine_frac=0.2, coarsen_frac=0.05,
                                max_level=3, max_leaves=typemax(Int),
                                cooldown=nothing, coarsen_gate=true,
                                geom=CartesianBox(qt), kwargs...)
        @test r == (0, 0, 0)
        @test qt.leaves == old
    end
    # max_level above the tree's own maximum: the cap cannot bind, so the
    # fallback is skipped and the cycle is the historical one
    qt, eta = t295_fixture()
    old = copy(qt.leaves)
    r = amr_cycle_balanced!(qt, eta; refine_frac=0.2, coarsen_frac=0.05,
                            max_level=4, max_leaves=typemax(Int),
                            cooldown=nothing, coarsen_gate=true,
                            cap_aware=true, geom=CartesianBox(qt))
    @test r == (0, 0, 0)
    @test qt.leaves == old
    # refine_frac=0 is an explicit no-refine cycle; the fallback must not fire
    qt, eta = t295_fixture()
    old = copy(qt.leaves)
    r = amr_cycle_balanced!(qt, eta; refine_frac=0.0, coarsen_frac=0.05,
                            max_level=3, max_leaves=typemax(Int),
                            cooldown=nothing, coarsen_gate=true,
                            cap_aware=true, geom=CartesianBox(qt))
    @test r == (0, 0, 0)
    @test qt.leaves == old
    # every leaf already at the cap: no admissible universe, no refinement
    qt2 = Quadtree(2)
    for _ in 1:2, lf in collect(qt2.leaves)
        refine!(qt2, lf)
    end
    eta2 = Dict(lf => 1.0 for lf in qt2.leaves)
    old2 = copy(qt2.leaves)
    r = amr_cycle_balanced!(qt2, eta2; refine_frac=0.2, coarsen_frac=0.05,
                            max_level=2, max_leaves=typemax(Int),
                            cooldown=nothing, coarsen_gate=true,
                            cap_aware=true, geom=CartesianBox(qt2))
    @test r == (0, 0, 0)
    @test qt2.leaves == old2
    @test LAST_AMR_BALANCED[].cap_applied == false

    # -- (d) gate-off is refused before any mutation -------------------------
    qt, eta = t295_fixture()
    old = copy(qt.leaves)
    @test_throws ArgumentError amr_cycle_balanced!(
        qt, eta; refine_frac=0.2, coarsen_frac=0.05, max_level=3,
        max_leaves=typemax(Int), cooldown=nothing, coarsen_gate=false,
        cap_aware=true, geom=CartesianBox(qt))
    @test qt.leaves == old

    # -- (e) a non-stalled fallback-capable cycle is bitwise standard --------
    qt, eta = t295_fixture()
    # make the refinable universe dominant so the standard candidate set is
    # non-empty: eta 10 on the 15 L2 leaves, 1 on the capped L3 leaves
    for lf in keys(eta)
        eta[lf] = lf[1] == 2 ? 10.0 : 1.0
    end
    old = copy(qt.leaves)
    qta = deepcopy(qt); qtb = deepcopy(qt)
    ra = amr_cycle_balanced!(qta, eta; refine_frac=0.2, coarsen_frac=0.05,
                             max_level=3, max_leaves=typemax(Int),
                             cooldown=nothing, coarsen_gate=true,
                             cap_aware=false, geom=CartesianBox(qta))
    da = deepcopy(LAST_AMR_BALANCED[])
    rb = amr_cycle_balanced!(qtb, eta; refine_frac=0.2, coarsen_frac=0.05,
                             max_level=3, max_leaves=typemax(Int),
                             cooldown=nothing, coarsen_gate=true,
                             cap_aware=true, geom=CartesianBox(qtb))
    db = deepcopy(LAST_AMR_BALANCED[])
    @test ra == rb
    @test qta.leaves == qtb.leaves
    @test da.cap_applied == false && db.cap_applied == false
    @test da.tau_r == db.tau_r && da.mark_target == db.mark_target
    @test da.want == db.want && da.refined == db.refined
    @test da.npost == db.npost && da.recirc == db.recirc
end
