# G210: the balanced AMR policy — relative coarsening, budget exchange and cooldown.
using Test, QuaST, Random, Printf

# G210: the balanced AMR policy, the only adaptive one
# since the classic policy was removed at `8c622df`.
#
# The production run E is pinned at 9994/10000 leaves with 3720 AMR cycles of
# which only 723 changed anything: the classic policy (absolute
# `coarsen_theta_tol`, refused leaves charged to the coarsening budget, hard
# budget guard, strip-wide protect set, driver `refine_frac=0` pin) cannot trade.
# The measurement campaign covered the three mechanisms on the live tree; this
# gate pins the replacement:
#
#   1. coarsening is *relative*: a quartet is merged only when
#      `max(eta_child) <= tau_h` with `tau_h = min(tau_c, hysteresis*tau_r)`,
#      both thresholds cumulative error fractions of the *current* tree;
#   2. the budget is paid for by coarsening ("exchange"), never a reason to stop
#      refining;
#   3. hysteresis: a quartet that was just merged is not re-refined on the next
#      cycle (and without the cooldown it is);
#   4. the wall protect set is the first/last leaf rows only;
#   5. invariants: 2:1 balance, `mirror_defect == 0` under `symmetry=:x`, the
#      leaf budget, area preservation, and never a silent `refined=0 coarsened=0`
#      cycle while a refinement is wanted;
#   6. the cycle is wired into the driver and the recorded `run_config` keeps
#      `amr_policy = "balanced"`.

# ---- helpers ---------------------------------------------------------------
# G210 pins the balanced policy exactly as measured, i.e.
# with the coarsening pre-check OFF (it did not exist then); G211 pins the gate
# itself.
# assertion in it directly comparable with those sections.
g210_cycle!(qt, eta; kw...) =
    QuaST.amr_cycle_balanced!(qt, eta; geom=CartesianBox(qt), coarsen_gate=false, kw...)

function g210_seed_tree(cx::Int, cy::Int, base::Int, maxlev::Int)
    qt = Quadtree(maxlev; cells_x=cx, cells_y=cy, Lx=cx / cy)
    for _ in 1:base, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

g210_zc(qt, lf) = (lf[3] + 0.5) * QuaST.leaf_hy(qt, lf[1])
g210_row(qt, lf) = lf[3] == 0 || lf[3] == qt.ny0 * (1 << lf[1]) - 1
g210_rows(qt) = Set(lf for lf in qt.leaves if g210_row(qt, lf))
g210_kids(p) = [(p[1] + 1, 2p[2] + dx, 2p[3] + dy) for dy in 0:1, dx in 0:1]

"eta = 1 everywhere, 0 on the given set (the low-error tail)."
function g210_eta(qt, low::Set{NTuple{3,Int}})
    eta = Dict{NTuple{3,Int},Float64}()
    for lf in qt.leaves
        eta[lf] = lf in low ? 0.0 : 1.0
    end
    return eta
end

@testset "G210 balanced AMR policy" begin
    # ---- 1. relative coarsening: only quartets below `tau_h` merge ----------
    @testset "relative coarsening criterion" begin
        qt = g210_seed_tree(5, 2, 3, 7)          # 640 leaves, uniform L3
        @test length(qt.leaves) == 640
        before = Set(qt.leaves)
        low = Set(lf for lf in qt.leaves if g210_zc(qt, lf) < 0.25)
        eta = g210_eta(qt, low)
        cool = AmrCooldown()
        nr, nc, nex = g210_cycle!(qt, eta; refine_frac=0.2,
                                          coarsen_frac=0.05, max_level=7,
                                          max_leaves=640, symmetry=:x,
                                          protect=g210_rows(qt), cooldown=cool)
        d = LAST_AMR_BALANCED[]
        @printf("  rel: tau_r=%.3f tau_c=%.3f tau_h=%.3f deadband=%s eligible=%d refined=%d coarsened=%d exchanged=%d leaves=%d\n",
                d.tau_r, d.tau_c, d.tau_h, d.deadband, d.eligible, nr, nc, nex,
                length(qt.leaves))
        newleaves = setdiff(Set(qt.leaves), before)
        mergeparents = [p for p in newleaves if all(k -> k in before, g210_kids(p))]
        @test nc > 0
        @test nr > 0
        @test d.tau_r == 1.0                     # the indicator has two values
        @test d.tau_c == 1.0
        @test d.tau_h == 0.5                     # min(tau_c, 0.5*tau_r)
        @test d.deadband
        # every merged quartet was entirely below the coarsen gate
        @test !isempty(mergeparents)
        @test all(p -> maximum(k -> eta[k], g210_kids(p)) <= d.tau_h,
                  mergeparents)
        @test !isempty(cool.no_refine)           # the cooldown recorded the merges
        @test mirror_defect(qt) == 0
        @test max_face_level_jump(qt) <= 1
    end

    # ---- 2. budget exchange: refinement continues at the cap --------------
    @testset "budget exchange" begin
        # the same state refines, funded by merging the low tail
        qb = g210_seed_tree(5, 2, 3, 7)
        low = Set(lf for lf in qb.leaves if g210_zc(qb, lf) < 0.25)
        eta = g210_eta(qb, low)
        nr_b, nc_b, nex_b = g210_cycle!(qb, eta; refine_frac=0.2,
                                                coarsen_frac=0.05, max_level=7,
                                                max_leaves=640, symmetry=:x,
                                                protect=g210_rows(qb))
        d = LAST_AMR_BALANCED[]
        @printf("  exchange: balanced refined=%d coarsened=%d exchanged=%d leaves=%d budget_limited=%s\n",
                nr_b, nc_b, nex_b, length(qb.leaves), d.budget_limited)
        @test nr_b > 0
        @test nex_b > 0                     # the refinement was paid for
        @test length(qb.leaves) <= 640 + 18 # budget (G208's documented slack)
        @test mirror_defect(qb) == 0
        @test max_face_level_jump(qb) <= 1
        @test isapprox(total_area(qb), qb.Lx; rtol=0.0, atol=1.0e-14)
        # a wanted refinement is never silently skipped: `budget_limited` says so
        @test nr_b > 0 || d.budget_limited
    end

    # ---- 3. hysteresis: no refine-after-merge churn -----------------------
    @testset "hysteresis" begin
        function churn(hyst::Int)
            qt = g210_seed_tree(5, 2, 3, 7)
            before = Set(qt.leaves)
            low = Set(lf for lf in qt.leaves if g210_zc(qt, lf) < 0.3)
            cool = AmrCooldown()
            g210_cycle!(qt, g210_eta(qt, low); refine_frac=0.2,
                        coarsen_frac=0.05, max_level=7, max_leaves=640,
                        symmetry=:x, protect=g210_rows(qt),
                        cooldown=cool, hysteresis_cycles=hyst)
            # the quartets the first cycle merged (geometric, independent of the
            # cooldown bookkeeping so both arms see the same parents)
            parents = sort([p for p in setdiff(Set(qt.leaves), before)
                            if all(k -> k in before, g210_kids(p))])
            isempty(parents) && return (0, 0, 0)
            if hyst > 0
                @test all(p -> haskey(cool.no_refine, p), parents)
            end
            pset = Set(parents)
            # the just-merged parents become the *highest* error leaves
            eta2 = Dict{NTuple{3,Int},Float64}(lf => (lf in pset ? 100.0 : 0.001)
                                               for lf in qt.leaves)
            g210_cycle!(qt, eta2; refine_frac=0.5, coarsen_frac=0.0,
                        max_level=7, max_leaves=800, symmetry=:x,
                        protect=g210_rows(qt), cooldown=cool,
                        hysteresis_cycles=hyst)
            refined_back = count(p -> !(p in qt.leaves), parents)
            return (length(parents), refined_back, LAST_AMR_BALANCED[].refined)
        end
        n1, churn1, nr1 = churn(1)
        n0, churn0, nr0 = churn(0)
        @printf("  hysteresis: cooldown=1 parents=%d re-refined=%d (cycle refined=%d) | cooldown=0 parents=%d re-refined=%d (cycle refined=%d)\n",
                n1, churn1, nr1, n0, churn0, nr0)
        @test n1 == n0 > 0
        @test churn1 == 0            # cooled leaves are not re-refined
        @test churn0 > 0             # without the cooldown they are
    end

    # ---- 4. the wall protect set is the rows only -------------------------
    @testset "wall rows only" begin
        # a thick strip (0.25) on a L4 tree so a whole quartet lies inside the
        # strip but off the wall rows: (iy 2,3) and (4,5) of 32 rows.
        qt = g210_seed_tree(5, 2, 4, 7)
        @test length(qt.leaves) == 2560
        before = Set(qt.leaves)
        prot = g210_rows(qt)
        low = Set(lf for lf in qt.leaves
                  if g210_zc(qt, lf) < 0.25 && !g210_row(qt, lf))
        @test !isempty(low)
        nr, nc, nex = g210_cycle!(qt, g210_eta(qt, low);
                                          refine_frac=0.2, coarsen_frac=0.05,
                                          max_level=7, max_leaves=2560,
                                          symmetry=:x, protect=prot,
                                          wall_min_level=0)
        newleaves = setdiff(Set(qt.leaves), before)
        strip_merge = count(p -> all(k -> k in low, g210_kids(p)), newleaves)
        # the rows may be *refined* (they are refine candidates), never merged
        # below their starting level
        rowmin = minimum(lf[1] for lf in qt.leaves if g210_row(qt, lf))
        @printf("  wall rows only: protect=%d, merged %d quartets, %d strip-only merges, row min level=%d\n",
                length(prot), nc, strip_merge, rowmin)
        @test nc > 0
        @test strip_merge > 0              # a strip, off-row quartet did merge
        @test rowmin >= 3                  # the rows were not coarsened below L3
        @test mirror_defect(qt) == 0
    end

    # ---- 5. invariants over a random sequence -----------------------------
    @testset "invariants over random cycles" begin
        for (cx, cy) in ((1, 1), (5, 2))
            rng = MersenneTwister(20261110)
            qt = g210_seed_tree(cx, cy, 3, 7)
            cool = AmrCooldown()
            for k in 1:12
                eta = Dict{NTuple{3,Int},Float64}()
                for lf in qt.leaves
                    eta[lf] = rand(rng)^2 *
                              exp(-2.0 * (lf[3] + 0.5) / (qt.ny0 * (1 << lf[1])))
                end
                g210_cycle!(qt, eta; refine_frac=0.25,
                            coarsen_frac=0.08, max_level=7,
                            max_leaves=900, symmetry=:x,
                            protect=g210_rows(qt), cooldown=cool,
                            hysteresis_cycles=1)
                d = LAST_AMR_BALANCED[]
                # the cycle reports an over-budget end; the driver's hard pass
                # (same one `:classic` uses) then merges it back to the target
                @test d.budget_infeasible == (length(qt.leaves) > 900)
                merged, ok = semquad_enforce_leaf_budget!(
                    qt, Dict{NTuple{3,Int},Float64}(lf => get(eta, lf, 0.0)
                                                    for lf in qt.leaves);
                    max_leaves=900, protect=g210_rows(qt), symmetry=:x)
                @test ok
                @test mirror_defect(qt) == 0
                @test max_face_level_jump(qt) <= 1
                @test length(qt.leaves) <= 900 + 18
                @test isapprox(total_area(qt), qt.Lx; rtol=0.0, atol=1.0e-14)
                @test d.tau_h <= d.tau_c + 1.0e-15
                @test d.tau_h <= 0.5 * d.tau_r + 1.0e-15
                @test d.refined > 0 || d.coarsened > 0 || d.budget_limited ||
                      d.eligible == 0
            end
            @printf("  random %dx%d: 12 cycles -> %d leaves, defect=%d, maxjump=%d\n",
                    cx, cy, length(qt.leaves), mirror_defect(qt),
                    max_face_level_jump(qt))
        end
    end

    # ---- 6. the driver path: the cycle is wired and the policy is recorded --
    @testset "driver wiring" begin
        pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz,
                    eta_b=9.70406052783923433184,
                    eta_c=4.1588830833596718565,
                    Twall_top=0.0, Twall_bot=1.0, bc=:free)
        function drv(; checkpoint::String="")
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
            res = run_semquad_simulation(qt, pr, th; p=2, tstop=8.0e-4,
                                         dt0=1.0e-4, maxsteps=5, amr_every=1,
                                         coarsen_frac=0.0, refine_frac=0.5,
                                         max_level=5,
                                         max_leaves=120, linear=:direct,
                                         verbose=false,
                                         checkpoint_path=checkpoint,
                                         checkpoint_every=5)
            return res
        end
        rb = drv()
        @test rb.steps == 5
        @test all(isfinite, rb.theta)
        @test max_face_level_jump(rb.qt) <= 1
        @test length(rb.qt.leaves) <= 120 + 18
        # the policy is part of run_config (kept as the constant "balanced" so a
        # checkpoint written before the classic policy was removed still matches)
        path = joinpath(mktempdir(), "g210.h5")
        drv(; checkpoint=path)
        st = load_semquad_state(path)
        @test st.config["amr_policy"] == "balanced"
        @test st.config["amr_hysteresis"] == "1.0"
    end

    # ---- 7. uniform eta: the deadband suppresses coarsening ----------------
    @testset "deadband with a flat indicator" begin
        qt = g210_seed_tree(1, 1, 3, 7)          # 64 leaves, uniform L3
        eta = Dict{NTuple{3,Int},Float64}(lf => 1.0 for lf in qt.leaves)
        nr, nc, nex = g210_cycle!(qt, eta; refine_frac=0.25,
                                          coarsen_frac=0.1, hysteresis=0.5,
                                          max_level=7, max_leaves=1000,
                                          symmetry=:none, protect=g210_rows(qt))
        d = LAST_AMR_BALANCED[]
        @printf("  flat: tau_r=%.3f tau_c=%.3f tau_h=%.3f deadband=%s refined=%d coarsened=%d\n",
                d.tau_r, d.tau_c, d.tau_h, d.deadband, nr, nc)
        @test d.tau_r == 1.0 && d.tau_c == 1.0
        @test d.deadband          # 0.5*tau_r < tau_c
        @test d.tau_h == 0.5
        @test nc == 0             # nothing is below half the (uniform) error
        @test nr > 0
    end
end
