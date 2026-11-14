using Test, QuaST, Random, Printf

# G211: the coarsening pre-check `coarsening_allowed` (docs/dev/AMR.md §4,
# docs/dev/AMR.md).
#
# `docs/dev/AMR.md` §4.1 measured the churn: at the operating point (10000 leaves)
# every quartet the coarsening criterion selects is 2:1-illegal and every merge
# is undone by the post-merge `balance!` -- 100 % of the merges, in the `:none`
# arm and in the `:x` arm alike -- so the AMR pass did 4x the work for no
# structural change.  deal.II
# refuses such a merge up front (`coarsening_allowed`, tria.cc:10935-10996) with
# the level each neighbour will have *after* the pass (its own coarsening -1, a
# pending refinement +1).  This gate pins:
#
#   1. the predicate: a merge next to a two-level-finer leaf is refused, one whose
#      faces stay within a level is granted, and a neighbour that is itself being
#      merged (granted) or refined (`will_refine`) is accounted for;
#   2. the plan `_coarsen_plan`: it refuses the criterion's illegal quartet, pays
#      with the lowest-error legal quartet -- extending past the deadband when it
#      must -- and never takes a quartet holding a refine-flagged child;
#   3. a `:balanced` cycle with the gate merges only legal quartets, reports
#      `recirc == 0` and leaves a tree on which `balance!` changes nothing;
#   4. the pre-gate behaviour (`coarsen_gate=false`) is still reachable and it
#      *does* recirculate -- the two arms differ only in the gate;
#   5. invariants over random cycles: face 2:1, `mirror_defect == 0` under `:x`,
#      leaf budget, area, and the recirculation stays a small fraction;
#   6. the flag is wired through the driver and is part of `run_config`.

# ---- helpers ---------------------------------------------------------------
g211_kids(lf) = QuaST._quartet_kids(lf)
g211_pq(lf) = QuaST._pq(lf)
g211_row(qt, lf) = lf[3] == 0 || lf[3] == qt.ny0 * (1 << lf[1]) - 1
g211_rows(qt) = Set(lf for lf in qt.leaves if g211_row(qt, lf))
g211_allowed(qt, lf) = QuaST._coarsening_allowed(
    qt, g211_kids(lf), Set{NTuple{3,Int}}(), Set{NTuple{3,Int}}())

"Uniform level-`base` tree on the unit square, then `refine!` each extra leaf."
function g211_tree(maxlev::Int, base::Int, extra::NTuple{3,Int}...)
    qt = Quadtree(maxlev)
    for _ in 1:base, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    for lf in extra
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

@testset "G211 coarsening_allowed" begin
    # ---- 1. the predicate ------------------------------------------------
    @testset "predicate: illegal next to a finer leaf, legal otherwise" begin
        # docs/dev/AMR.md §2's minimal repro: 16 leaves at
        # L2, refine (2,1,3) -> 19.  The quartet of (2,2,3) has (2,1,2) and the
        # three L3 children of (2,1,3) on its xminus face.
        qt = g211_tree(5, 2, (2, 1, 3))
        @test length(qt.leaves) == 19
        illegal = (2, 2, 3)
        @test all(k -> k in qt.leaves, g211_kids(illegal))
        @test !g211_allowed(qt, illegal)
        # ... and the same quartet becomes allowed once its finer neighbours are
        # themselves part of this pass's merge plan (deal.II's user_flag_set):
        # all three L3 children belong to the quartet of (2,1,3), so granting
        # that parent leaves the face ending at level 2 against the parent's 1.
        granted = Set([g211_pq((3, 2, 6))])
        @test (2, 1, 3) in granted
        @test QuaST._coarsening_allowed(qt, g211_kids(illegal),
                                            Set{NTuple{3,Int}}(), granted)
        @test !QuaST._coarsening_allowed(qt, g211_kids(illegal),
                                             Set{NTuple{3,Int}}(),
                                             Set{NTuple{3,Int}}())
        # a quartet away from the refined leaf is legal ...
        legal = (2, 2, 0)                       # the quartet of parent (1,1,0)
        @test all(k -> k in qt.leaves, g211_kids(legal))
        @test g211_allowed(qt, legal)
        # ... until a neighbour of it is scheduled for refinement (+1 level)
        @test !QuaST._coarsening_allowed(qt, g211_kids(legal),
                                             Set([(2, 2, 2)]),
                                             Set{NTuple{3,Int}}())
        # the pre-gate `coarsen!` path and its undo: -3 then +3, net zero
        q2 = g211_tree(5, 2, (2, 1, 3))
        @test coarsen!(q2, illegal)
        @test length(q2.leaves) == 16
        @test max_face_level_jump(q2) == 2
        balance!(q2)
        @test length(q2.leaves) == 19
        @test max_face_level_jump(q2) <= 1
    end

    # ---- 2. the plan: refuse illegal, pay with the lowest legal -----------
    @testset "plan: illegal refused, legal granted, extension counted" begin
        qt = g211_tree(5, 2, (2, 1, 3))
        # eta: the illegal quartet at 0, one legal quartet at 0.8, the rest 1.0.
        # tau_h = 0.5, tau_r = 1.0: the 0.8 quartet is legal but above the
        # deadband, i.e. exactly the "extended" supply.
        low = Set(g211_kids((2, 2, 3)))
        ext = Set(g211_kids((2, 2, 0)))
        eta = Dict{NTuple{3,Int},Float64}(lf => (lf in low ? 0.0 :
                                                 lf in ext ? 0.8 : 1.0)
                                         for lf in qt.leaves)
        plan = QuaST._coarsen_plan(qt, eta, Set{NTuple{3,Int}}(),
                                      _ -> 0, nothing, 3, :none,
                                      Set{NTuple{3,Int}}(), 0.5, 1.0)
        @printf("  plan: order=%s refused=%d extended=%d\n",
                string(plan.order), plan.refused, plan.extended)
        @test plan.order == [(2, 2, 0)]         # the illegal 0-eta quartet lost
        @test plan.refused == 1
        @test plan.extended == 1
        # the `m < tau_r` rule: with tau_r = 0.8 the extended quartet is gone too
        plan2 = QuaST._coarsen_plan(qt, eta, Set{NTuple{3,Int}}(),
                                       _ -> 0, nothing, 3, :none,
                                       Set{NTuple{3,Int}}(), 0.5, 0.8)
        @test isempty(plan2.order)
        # ... and nothing is planned when there is nothing to fund
        plan3 = QuaST._coarsen_plan(qt, eta, Set{NTuple{3,Int}}(),
                                       _ -> 0, nothing, 0, :none,
                                       Set{NTuple{3,Int}}(), 0.5, 1.0)
        @test isempty(plan3.order)
    end

    # ---- 3. one cycle: gate on vs gate off --------------------------------
    @testset "cycle: only legal merges, balance! is a no-op, recirc == 0" begin
        # The finer neighbour quartet (the children of (2,1,3)) is *protected*,
        # so the illegal quartet next to it cannot be legalised by merging the
        # neighbour ("or the neighbour itself will be coarsened in this pass"):
        # the gate has to refuse it, while the historical path merges it and the
        # post-merge `balance!` immediately undoes that (the `docs/dev/AMR.md` §4.1 churn).
        function one_cycle(gate::Bool)
            qt = g211_tree(5, 2, (2, 1, 3))
            low = Set(g211_kids((2, 2, 3)))
            protect = Set([(3, 2, 6), (3, 3, 6), (3, 2, 7), (3, 3, 7)])
            @test all(k -> k in qt.leaves, protect)
            eta = Dict{NTuple{3,Int},Float64}(lf => (lf in low ? 0.0 : 1.0)
                                             for lf in qt.leaves)
            nr, nc, nex = amr_cycle_balanced!(qt, eta; refine_frac=0.0,
                                             coarsen_frac=0.5, max_level=5,
                                             max_leaves=16, symmetry=:none,
                                             protect=protect,
                                             coarsen_gate=gate)
            d = LAST_AMR_BALANCED[]
            return (qt=qt, d=d, nc=nc, nr=nr,
                    illegal_alive=all(k -> k in low, low))
        end
        off = one_cycle(false)
        on = one_cycle(true)
        @printf("  cycle: gate off -> coarsened=%d recirc=%d leaves=%d | gate on -> coarsened=%d recirc=%d refused=%d leaves=%d\n",
                off.nc, off.d.recirc, length(off.qt.leaves), on.nc, on.d.recirc,
                on.d.gate_refused, length(on.qt.leaves))
        # before: the criterion's illegal quartet is merged and immediately undone
        @test off.nc == 1
        @test off.d.recirc == 1
        @test length(off.qt.leaves) == 19        # net zero: pure churn
        @test off.illegal_alive                  # the re-split restored it
        # after: the illegal quartet is refused and survives; the plan pays with
        # legal quartets only, and the post-merge `balance!` has nothing to split
        @test on.nc >= 1
        @test on.d.gate_refused == 1
        @test on.d.recirc == 0
        @test on.illegal_alive
        @test all(p -> p in on.qt.leaves, on.d.parents)
        snapshot = Set(on.qt.leaves)
        balance!(on.qt)
        @test Set(on.qt.leaves) == snapshot      # the gate made the tree 2:1
        @test max_face_level_jump(on.qt) <= 1
        @test isapprox(total_area(on.qt), on.qt.Lx; rtol=0.0, atol=1.0e-14)
        @test on.d.merged == on.nc
    end

    # ---- 4. mirror and random-cycle invariants ---------------------------
    @testset "invariants: mirror, 2:1, budget, area" begin
        # `:x` tree: uniform L2 plus a mirror pair of refinements
        function mirror_tree()
            q = Quadtree(6; Lx=1.0, cells_x=1, cells_y=1)
            for _ in 1:2, lf in collect(q.leaves)
                refine!(q, lf)
            end
            balance!(q)
            refine!(q, (2, 0, 3))
            refine!(q, (2, 3, 3))
            balance!(q; symmetry=:x)
            return q
        end
        q = mirror_tree()
        @test mirror_defect(q) == 0
        n0 = length(q.leaves)
        eta = Dict{NTuple{3,Int},Float64}(lf => (lf[2] < 2 ? 0.0 : 1.0)
                                          for lf in q.leaves)
        nr, nc, _ = amr_cycle_balanced!(q, eta; refine_frac=0.0,
                                        coarsen_frac=0.5, max_level=6,
                                        max_leaves=n0 - 6, symmetry=:x,
                                        protect=Set{NTuple{3,Int}}(),
                                        cooldown=AmrCooldown(),
                                        coarsen_gate=true)
        d = LAST_AMR_BALANCED[]
        @printf("  mirror: refined=%d coarsened=%d extended=%d refused=%d recirc=%d leaves %d -> %d defect=%d jump=%d\n",
                nr, nc, d.gate_extended, d.gate_refused, d.recirc, n0,
                length(q.leaves), mirror_defect(q), max_face_level_jump(q))
        @test nc == 4                            # two mirror pairs
        @test d.recirc == 0
        @test all(p -> p in q.leaves, d.parents)
        @test mirror_defect(q) == 0
        @test max_face_level_jump(q) <= 1
        snapshot = Set(q.leaves)
        balance!(q; symmetry=:x)
        @test Set(q.leaves) == snapshot
        @test isapprox(total_area(q), q.Lx; rtol=0.0, atol=1.0e-14)

        # random cycles on both bases, gate on: invariants hold, the
        # recirculation is a small fraction of the merges (a refinement outside
        # the plan can still undo one; docs/dev/AMR.md §4.7)
        for (cx, cy) in ((1, 1), (5, 2))
            rng = MersenneTwister(20261110)
            qq = Quadtree(7; cells_x=cx, cells_y=cy, Lx=cx / cy)
            for _ in 1:3, lf in collect(qq.leaves)
                refine!(qq, lf)
            end
            balance!(qq)
            cool = AmrCooldown()
            merged_tot = 0
            recirc_tot = 0
            for k in 1:12
                eta = Dict{NTuple{3,Int},Float64}()
                for lf in qq.leaves
                    eta[lf] = rand(rng)^2 *
                              exp(-2.0 * (lf[3] + 0.5) / (qq.ny0 * (1 << lf[1])))
                end
                amr_cycle_balanced!(qq, eta; refine_frac=0.25,
                                    coarsen_frac=0.08, max_level=7,
                                    max_leaves=900, symmetry=:x,
                                    protect=g211_rows(qq), cooldown=cool,
                                    hysteresis_cycles=1, coarsen_gate=true)
                d = LAST_AMR_BALANCED[]
                merged_tot += d.merged
                recirc_tot += d.recirc
                @test all(p -> p in qq.leaves || d.recirc > 0, d.parents)
                @test mirror_defect(qq) == 0
                @test max_face_level_jump(qq) <= 1
                @test isapprox(total_area(qq), qq.Lx; rtol=0.0, atol=1.0e-14)
                semquad_enforce_leaf_budget!(
                    qq, Dict{NTuple{3,Int},Float64}(lf => get(eta, lf, 0.0)
                                                    for lf in qq.leaves);
                    max_leaves=900, protect=g211_rows(qq), symmetry=:x)
                @test length(qq.leaves) <= 900 + 60   # the gate can leave the hard
                                              # pass without a mergeable quartet
                                              # (`reached=false`; measured +34/+40
                                              # here against +18 without the gate)
                @test max_face_level_jump(qq) <= 1
            end
            @printf("  random %dx%d: 12 gate cycles -> %d leaves, merged %d, recirc %d, defect=%d jump=%d\n",
                    cx, cy, length(qq.leaves), merged_tot, recirc_tot,
                    mirror_defect(qq), max_face_level_jump(qq))
            @test recirc_tot <= merged_tot ÷ 4   # most merges survive the balance
        end
    end

    # ---- 5. driver wiring --------------------------------------------------
    @testset "driver: the gate is wired into run_config" begin
        pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz,
                    eta_b=9.70406052783923433184,
                    eta_c=4.1588830833596718565,
                    Twall_top=0.0, Twall_bot=1.0, bc=:free)
        function drv(; gate::Bool=true, checkpoint::String="")
            qt = Quadtree(5; cells_x=5, cells_y=2, Lx=2.5)
            for _ in 1:2, lf in collect(qt.leaves)
                refine!(qt, lf)
            end
            balance!(qt)
            ss = make_semstokes(qt, 2)
            th = [0.1 * sinpi(y) * cospi(x / qt.Lx) for (x, y) in ss.su.coords]
            for (g, (_, y)) in enumerate(ss.su.coords)
                (y <= 1.0e-12 || y >= 1.0 - 1.0e-12) && (th[g] = 0.0)
            end
            return run_semquad_simulation(qt, pr, th; p=2, tstop=8.0e-4,
                                          dt0=1.0e-4, maxsteps=5, amr_every=1,
                                          coarsen_frac=0.0, refine_frac=0.5,
                                          max_level=5,
                                          max_leaves=120, linear=:direct,
                                          amr_coarsen_gate=gate, verbose=false,
                                          checkpoint_path=checkpoint,
                                          checkpoint_every=5)
        end
        rb = drv(; gate=true)
        rb0 = drv(; gate=false)
        # both arms reach the driver path and are finite
        @test rb.steps == 5 && rb0.steps == 5
        @test all(isfinite, rb.theta) && all(isfinite, rb0.theta)
        @test max_face_level_jump(rb.qt) <= 1
        @test max_face_level_jump(rb0.qt) <= 1
        @test length(rb.qt.leaves) <= 120 + 18
        # the flag is part of run_config (a restart cannot switch it silently)
        path = joinpath(mktempdir(), "g211.h5")
        drv(; gate=true, checkpoint=path)
        st = load_semquad_state(path)
        @test st.config["amr_coarsen_gate"] == "true"
        path0 = joinpath(mktempdir(), "g211_off.h5")
        drv(; gate=false, checkpoint=path0)
        @test load_semquad_state(path0).config["amr_coarsen_gate"] == "false"
        @test haskey(st.config, "amr_policy")
    end
end
