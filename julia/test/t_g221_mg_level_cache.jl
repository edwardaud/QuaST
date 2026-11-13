using Test, MantleSEM, LinearAlgebra, SparseArrays, Random

# G221: the per-level MG hierarchy cache -- reusing a coarse level across AMR
# cycles (docs/MG.md).
#
# `make_semquad_mg` used to rebuild every level whenever the tree changed, and
# one AMR cycle changes only ~1-2 % of the leaves, so most coarse levels are
# unchanged.  The cache stores a level's **topology-only** state (tree, space,
# `Cu`, free rows, prolongation) keyed by `(level index, leaf-set fingerprint,
# p/BC/...)`; the operator data that depends on the viscosity sampled at build
# time (`mfetas`, `D`, the coarse factor) is deliberately *not* cached, so a
# cache hit is bitwise the from-scratch build.
#
#   (a) the key is a stable fingerprint of the level's own leaf set: two builds
#       of the same tree (different `Quadtree` objects, different leaf order)
#       reuse every coarse level and return the same objects, and the hash does
#       not depend on set order;
#   (b) a changed leaf set invalidates exactly the affected levels: after
#       refining one leaf, the levels rebuilt are precisely the levels whose
#       leaf set changed, the others come back as the *same objects*, the reuse
#       result is bitwise the forced from-scratch build, and an eta change
#       invalidates nothing (the shipped policy);
#   (c) a 4-cycle AMR trajectory with reuse on and off produces bitwise
#       identical `theta`/`ux`/`uz` at every step, with hits > 0 on the reuse
#       run and 0 on the forced-rebuild run;
#   (d) the switch and the single global: `SEMQUAD_MG_LEVEL_CACHE[] = false`
#       rebuilds every level, and `empty!(SEMQUAD_MG_CACHE)` drops the levels.
#
# The measured setup/A-B numbers are in docs/MG.md; time is not gated
# here.

const _G221_Z2 = (x, y) -> 0.0
const _G221_ONE = (x, y) -> 1.0
const _G221_TWO = (x, y) -> 2.0

"""
A "staircase" tree: one corner cell refined four times, so the leaves sit at
every level 0..4.  Coarsening one layer per step (the historical
`coarsen_factor=1.0`) then gives a *deep* hierarchy out of 20-odd leaves, which
is what (a)/(b) need and what keeps the gate fast.
"""
function _g221_stair()
    qt = Quadtree(5; cells_x=5, cells_y=2, Lx=2.5)
    lf = (0, 0, 0)
    for _ in 1:4
        lf in qt.leaves || break
        refine!(qt, lf) || break
        balance!(qt)
        lf = (lf[1] + 1, 2 * lf[2], 2 * lf[3])       # lower-left child
    end
    balance!(qt)
    return qt
end

"~250-leaf graded 5x2 tree: hanging nodes, used for the AMR trajectory (c)."
function _g221_tree()
    qt = Quadtree(3; cells_x=5, cells_y=2, Lx=2.5)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    for _ in 1:2
        for lf in collect(qt.leaves)
            lf[1] >= 5 && continue
            hx = 2.5 / (5 * (1 << lf[1]))
            hy = 1.0 / (2 * (1 << lf[1]))
            x0 = lf[2] * hx; y0 = lf[3] * hy
            (x0 < 0.55 && y0 < 0.45) && refine!(qt, lf)
        end
        balance!(qt)
    end
    return qt
end

"Reduced-assembly `Cu` of `qt` (what the solve hands the hierarchy)."
function _g221_Cu(ss, qt)
    sys = solve_semquad_stokes_var(ss, qt, _G221_ONE, _G221_Z2, _G221_Z2,
                                   _G221_Z2, _G221_Z2; bc=:free, linear=:mg,
                                   mg_operator=:matrixfree, assemble_only=:mg)
    return SparseMatrixCSC{Float64,Int}(sparse(sys.Cu))
end

_g221_build(ss, qt, Cu, eta; factor::Float64=1.0) =
    MantleSEM.make_semquad_mg(ss, qt, 2, :free, false, eta, nothing; cycles=1,
                              smoother=:jacobi, coarse_operator=:rediscretize,
                              operator=:matrixfree, coarsen_factor=factor,
                              smoothing=:auto, mg_threads=1, fine_Cu=Cu)

"V-cycle on a deterministic right-hand side (the bitwise probe)."
_g221_vcycle(mg) = MantleSEM.semquad_mg_apply(mg, [0.1 + 0.01 * i
                                                   for i in 1:mg.geo.nfs[end]])

@testset "G221 per-level MG hierarchy cache" begin
    Random.seed!(20261112)
    MS = MantleSEM
    qt = _g221_stair()
    ss = make_semstokes(qt, 2)
    Cu = _g221_Cu(ss, qt)
    nlev = length(MS._mg_hierarchy_trees(qt; layers=1, factor=1.0))
    @test nlev >= 4                      # a deep hierarchy to reuse levels from
    @test length(qt.leaves) < 64
    @test MantleSEM.SEMQUAD_MG_LEVEL_CACHE[]     # the shipped default

    # ---- (a) the key is stable across identical trees --------------------
    empty!(MS.SEMQUAD_MG_CACHE)
    MS.reset_semquad_mg_level_stats!()
    mg1 = _g221_build(ss, qt, Cu, _G221_ONE)
    st = MS.semquad_mg_level_cache_stats()
    @test st.builds == 1
    @test st.hits == 0
    @test st.misses == nlev - 1          # every coarse level built; fine level not cached
    @test st.cached_levels == nlev - 1
    @test st.bytes > 0
    @test st.last_build_s > 0.0
    @test mg1.geo.L + 1 == nlev

    # the fingerprint is a function of the leaf *set*, not of the iteration
    # order or of the `Quadtree` object
    leaves = sort(collect(qt.leaves))
    h0 = MS._semquad_mg_leaf_hash(qt.leaves)
    @test h0 == MS._semquad_mg_leaf_hash(reverse(leaves))
    @test h0 == MS._semquad_mg_leaf_hash(shuffle(leaves))
    @test h0 == MS._semquad_mg_leaf_hash(MS._mg_copy_tree(qt).leaves)
    @test h0 != MS._semquad_mg_leaf_hash(leaves[2:end])   # a changed set moves it

    # a second build of the *same* leaves (a different tree object) reuses every
    # coarse level and returns the very same objects
    qt2 = MS._mg_copy_tree(qt)
    ss2 = make_semstokes(qt2, 2)
    Cu2 = _g221_Cu(ss2, qt2)
    MS._semquad_mg_drop_tree_entries!()
    mg2 = _g221_build(ss2, qt2, Cu2, _G221_ONE)
    st = MS.semquad_mg_level_cache_stats()
    @test st.hits == nlev - 1 && st.misses == nlev - 1   # cumulative
    @test isempty(st.last_rebuilt)
    @test st.last_reused == collect(1:(nlev - 1))
    @test all(mg2.geo.qts[k] === mg1.geo.qts[k] for k in 1:(nlev - 1))
    @test all(mg2.geo.Cus[k] === mg1.geo.Cus[k] for k in 1:(nlev - 1))
    @test all(mg2.geo.Ps[k] === mg1.geo.Ps[k] for k in 1:mg1.geo.L)
    @test mg2.geo.Cus[end] == mg1.geo.Cus[end]           # the caller's own Cu
    @test mg2.geo.nfs == mg1.geo.nfs
    @test mg2.geo.free_rows == mg1.geo.free_rows
    @test _g221_vcycle(mg2) == _g221_vcycle(mg1)
    # ... and the operator data is rebuilt, not reused (it depends on eta)
    mg3 = _g221_build(ss2, qt2, Cu2, _G221_TWO)
    st = MS.semquad_mg_level_cache_stats()
    @test isempty(st.last_rebuilt)                      # eta does not invalidate
    @test mg3.mfetas != mg2.mfetas && mg3.D != mg2.D
    @test all(mg3.mfetas[k] != mg2.mfetas[k] for k in 1:(nlev - 1))

    # ---- (b) a changed leaf set invalidates exactly the affected levels --
    trees_old = MS._mg_hierarchy_trees(qt; layers=1, factor=1.0)
    # a leaf that survives into the finest *coarse* level (an unmerged leaf of
    # the staircase): refining it changes that level but not the ones below
    cand = sort([lf for lf in trees_old[end - 1].leaves if lf in qt.leaves])
    @test !isempty(cand)
    qt3 = MS._mg_copy_tree(qt)
    refine!(qt3, cand[end]) || error("G221: could not refine the chosen leaf")
    balance!(qt3)
    ss3 = make_semstokes(qt3, 2)
    Cu3 = _g221_Cu(ss3, qt3)
    trees_new = MS._mg_hierarchy_trees(qt3; layers=1, factor=1.0)
    @test length(trees_old) == length(trees_new)
    changed = [k for k in 1:(length(trees_new) - 1)
               if MS._semquad_mg_leaf_hash(trees_old[k].leaves) !=
                  MS._semquad_mg_leaf_hash(trees_new[k].leaves)]
    @test !isempty(changed)              # the refined coarse level changed
    @test length(changed) < nlev - 1     # ... but not every level
    MS._semquad_mg_drop_tree_entries!()
    mg4 = _g221_build(ss3, qt3, Cu3, _G221_ONE)
    st = MS.semquad_mg_level_cache_stats()
    @test sort(st.last_rebuilt) == changed
    @test sort(st.last_reused) == setdiff(collect(1:(nlev - 1)), changed)
    kept = setdiff(collect(1:(nlev - 1)), changed)
    @test all(mg4.geo.qts[k] === mg1.geo.qts[k] for k in kept)
    # the reuse result is bitwise the from-scratch build (the forced reference)
    MS.SEMQUAD_MG_LEVEL_CACHE[] = false
    empty!(MS.SEMQUAD_MG_CACHE)
    m4b = _g221_build(ss3, qt3, Cu3, _G221_ONE)
    MS.SEMQUAD_MG_LEVEL_CACHE[] = true
    @test m4b.geo.nfs == mg4.geo.nfs
    @test all(m4b.geo.Cus[k] == mg4.geo.Cus[k] for k in 1:nlev)
    @test all(m4b.geo.Ps[k] == mg4.geo.Ps[k] for k in 1:mg4.geo.L)
    @test m4b.geo.free_rows == mg4.geo.free_rows
    @test m4b.D == mg4.D
    @test m4b.mfetas == mg4.mfetas
    @test _g221_vcycle(m4b) == _g221_vcycle(mg4)

    # ---- (c) reuse is bitwise identical to a forced rebuild --------------
    pr = Params(Ra=1.0e4, H=0.0, eta_law=:expTz, eta_b=log(100.0),
                eta_c=log(10.0), bc=:free)
    function traj(reuse::Bool)
        MS.SEMQUAD_MG_LEVEL_CACHE[] = reuse
        empty!(MS.SEMQUAD_MG_CACHE)
        MS.SEMQUAD_PRESSURE_CACHE[] = nothing
        MS.reset_semquad_mg_level_stats!()
        q = _g221_tree()
        s = make_semstokes(q, 2)
        th0 = [0.01 * cos(pi * x / 2.5) * sin(pi * y) for (x, y) in s.su.coords]
        states = NamedTuple[]
        cb = function (st)
            push!(states, (step=st.step, theta=copy(st.theta), ux=copy(st.ux),
                           uz=copy(st.uz), leaves=length(st.qt.leaves)))
            return nothing
        end
        # Policy-agnostic cache gate calibrated on classic; pin the policy.
        res = run_semquad_simulation(q, pr, th0; p=2, tstop=Inf, maxsteps=8,
                                     dt0=1.0e-4, cfl=5.0, maxpic=1,
                                     amr_every=2, refine_frac=0.2,
                                     amr_policy=:classic,
                                     coarsen_frac=0.05, marking=:error_fraction,
                                     max_leaves=400, max_level=6,
                                     linear=:mg, mg_operator=:matrixfree,
                                     mg_vel_solver=:vcycle, mg_tol=1.0e-8,
                                     verbose=false, callback=cb)
        return (states=states, stats=MS.semquad_mg_level_cache_stats(), res=res)
    end
    a = traj(true)
    b = traj(false)
    @test length(a.states) == 8 && length(b.states) == 8
    @test a.res.steps == b.res.steps
    @test all(x.theta == y.theta && x.ux == y.ux && x.uz == y.uz
              for (x, y) in zip(a.states, b.states))
    @test a.stats.hits > 0                     # the cache really hit
    @test b.stats.hits == 0                    # the reference rebuilt everything
    @test b.stats.misses > a.stats.misses      # and it built more levels
    @test length(unique(s.leaves for s in a.states)) > 1   # the mesh moved
    @test all(s.leaves == t.leaves for (s, t) in zip(a.states, b.states))
    @info "G221 trajectory" hits = a.stats.hits misses = a.stats.misses forced_misses = b.stats.misses build_s = a.stats.total_build_s forced_build_s = b.stats.total_build_s bytes = a.stats.bytes

    # ---- (d) the switch and the single global ----------------------------
    MS.SEMQUAD_MG_LEVEL_CACHE[] = false
    empty!(MS.SEMQUAD_MG_CACHE)
    MS.reset_semquad_mg_level_stats!()
    mg5 = _g221_build(ss, qt, Cu, _G221_ONE)
    st = MS.semquad_mg_level_cache_stats()
    @test st.hits == 0 && st.misses == nlev - 1
    @test st.cached_levels == 0                  # nothing stored while switched off
    @test _g221_vcycle(mg5) == _g221_vcycle(mg1)
    MS.SEMQUAD_MG_LEVEL_CACHE[] = true
    MS._semquad_mg_drop_tree_entries!()
    mg6 = _g221_build(ss, qt, Cu, _G221_ONE)     # still cold: the level cache was empty
    st = MS.semquad_mg_level_cache_stats()
    @test st.cached_levels == nlev - 1
    @test st.hits == 0 && st.misses == 2 * (nlev - 1)
    @test _g221_vcycle(mg6) == _g221_vcycle(mg1)
    empty!(MS.SEMQUAD_MG_CACHE)                  # one global: this drops the levels
    @test MS.semquad_mg_level_cache_stats().cached_levels == 0
    MS.reset_semquad_mg_level_stats!()
    _g221_build(ss, qt, Cu, _G221_ONE)
    @test MS.semquad_mg_level_cache_stats().misses == nlev - 1   # cold again
end
