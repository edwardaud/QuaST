# G189: the pressure-cache threshold semantics.
using Test, QuaST, LinearAlgebra, Random

# G189 / S5-1: the *threshold semantics* of the pressure-block cache
# (docs/dev/SOLVER.md §19.1).
#
# §13.3 measured the outer iteration count of the `linear=:mg` saddle solve to
# be flat in the cache tolerance (39.7 at 1e-12 ... 39.9 at Inf) while the
# build count falls from one build per solve to one per tree, so the S5-1
# retune moved the default `mg_pressure_eta_tol` from 0.5 to 1.0 and documented
# `Inf` as an explicit opt-in.  This gate pins the *semantics* of that knob:
#
#   1. the documented default is 1.0;
#   2. on a synthetic 5%/step viscosity drift the build count is monotone
#      non-increasing in the tolerance and `Inf` builds exactly once;
#   3. at the default 1.0 a *sudden* jump above 100% (phase change / yield
#      onset inside one tree block) still rebuilds;
#   4. at `Inf` the eta drift never rebuilds -- unbounded staleness, which is
#      exactly why it is not the default -- but the TREE fingerprint always
#      wins: a changed tree rebuilds even at `Inf`;
#   5. the tolerance is a *cost* knob, not a correctness knob: through the
#      public API a stale-`Inf` solve reproduces the fresh-build answer to
#      <=1e-8 relative and the iteration count stays within +10%.

const _G189_Z2 = (x, y) -> 0.0

function _g189_tree(level::Int)
    # `maxlevel = level + 1` so that one leaf can be refined once more, which
    # makes a genuinely different tree of the same base resolution (the cache
    # key covers the leaf set).
    qt = Quadtree(level + 1)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

@testset "G189 pressure-cache threshold semantics" begin
    Random.seed!(20261107)

    # 64 leaves at p=2 keeps the whole gate a few seconds; the eta law gives
    # enough contrast for `:auto`/BFBT to be the expensive, measured path.
    qt = _g189_tree(3)
    p = 2
    base = (x, y) -> exp(-log(1.0e4) * (1.0 - y))
    ss = make_semstokes(qt, p)
    z2 = _G189_Z2
    sys = solve_semquad_stokes_var(ss, qt, base, z2, z2, z2, z2; bc=:free,
                                   linear=:direct, assemble_only=true)
    key = QuaST._saddle_key(qt, p, :free, size(sys.Cu, 2), size(sys.Cp, 2))
    precond(eta; tol, kw...) = QuaST.semquad_mg_pressure_precond(
        ss, sys.Cu, sys.Cp, sys.Bs, sys.Bt, sys.Kred, eta;
        mode=:bfbt, key=key, eta_tol=tol, kw...)

    # ---- 1. the documented default -----------------------------------------
    @test SEMQUAD_PRESSURE_DEFAULT_ETA_TOL == 1.0

    # ---- 2. build count vs tolerance on a synthetic drift ------------------
    # The blob's viscosity moves 5.7-8.9% per *step*; 5% per query reproduces
    # that scale.  The jump is measured against the cached (last built) field,
    # so a tolerance of 0.1 accumulates two steps and so on.
    nq = 20
    eta_k(k) = (x, y) -> 1.05^k * exp(-log(1.0e4) * (1.0 - y))
    function builds_for(tol)
        SEMQUAD_PRESSURE_CACHE[] = nothing
        b0 = SEMSTOKES_PRESSURE_BUILDS[]
        for k in 0:nq
            precond(eta_k(k); tol=tol)
        end
        return SEMSTOKES_PRESSURE_BUILDS[] - b0
    end
    tols = (1.0e-12, 0.1, 0.5, 1.0, Inf)
    nb = [builds_for(t) for t in tols]
    @info "G189 builds vs tol" tols = tols builds = Tuple(nb)
    @test nb[1] == nq + 1                    # 1e-12: one build per query
    for i in 1:(length(tols) - 1)
        @test nb[i + 1] <= nb[i]             # monotone non-increasing
    end
    @test nb[end] == 1                       # Inf: one build, never rebuilt
    # 0.5 and 1.0 sit strictly between "every query" and "never"
    @test 1 < nb[4] < nb[3] < nb[1]
    # the default 1.0 halves the build count of the previous default 0.5
    @test nb[4] < nb[3]

    # ---- 3. a sudden jump rebuilds at the default 1.0 ----------------------
    SEMQUAD_PRESSURE_CACHE[] = nothing
    b0 = SEMSTOKES_PRESSURE_BUILDS[]
    precond(base; tol=SEMQUAD_PRESSURE_DEFAULT_ETA_TOL)      # build
    @test SEMSTOKES_PRESSURE_BUILDS[] == b0 + 1
    sudden = (x, y) -> 2.5 * exp(-log(1.0e4) * (1.0 - y))    # +150%, one step
    precond(sudden; tol=SEMQUAD_PRESSURE_DEFAULT_ETA_TOL)
    @test !LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test LAST_SEMSTOKES_PRESSURE_JUMP[] > 1.0
    @test SEMSTOKES_PRESSURE_BUILDS[] == b0 + 2
    # exactly 100% is *at* the boundary: the test is `jump <= eta_tol`, so a
    # doubling of eta inside one tree block is still accepted at 1.0
    SEMQUAD_PRESSURE_CACHE[] = nothing
    precond(base; tol=SEMQUAD_PRESSURE_DEFAULT_ETA_TOL)      # build
    twice = (x, y) -> 2.0 * base(x, y)
    precond(twice; tol=SEMQUAD_PRESSURE_DEFAULT_ETA_TOL)
    @test LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test LAST_SEMSTOKES_PRESSURE_JUMP[] == 1.0

    # ---- 4. `Inf`: eta never forces a rebuild, the tree always does --------
    SEMQUAD_PRESSURE_CACHE[] = nothing
    b0 = SEMSTOKES_PRESSURE_BUILDS[]
    precond(base; tol=Inf)
    @test SEMSTOKES_PRESSURE_BUILDS[] == b0 + 1
    huge = (x, y) -> 1.0e3 * exp(-log(1.0e4) * (1.0 - y))    # +1e5 %
    precond(huge; tol=Inf)
    @test LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test LAST_SEMSTOKES_PRESSURE_JUMP[] > 100.0
    @test SEMSTOKES_PRESSURE_BUILDS[] == b0 + 1
    # the tree fingerprint is *always* consulted first: a new tree rebuilds
    # even at Inf (this is the guarantee that Inf is not "cache forever").
    qt2 = Quadtree(4)
    for _ in 1:3, lf in collect(qt2.leaves)
        refine!(qt2, lf)
    end
    balance!(qt2)
    refine!(qt2, sort(collect(qt2.leaves))[1])
    balance!(qt2)
    @test Set(qt2.leaves) != Set(qt.leaves)
    ss2 = make_semstokes(qt2, p)
    sys2 = solve_semquad_stokes_var(ss2, qt2, huge, z2, z2, z2, z2;
                                    bc=:free, linear=:direct, assemble_only=true)
    key2 = QuaST._saddle_key(qt2, p, :free, size(sys2.Cu, 2),
                                 size(sys2.Cp, 2))
    @test key2 != key
    QuaST.semquad_mg_pressure_precond(ss2, sys2.Cu, sys2.Cp, sys2.Bs,
                                          sys2.Bt, sys2.Kred, huge;
                                          mode=:bfbt, key=key2, eta_tol=Inf)
    @test !LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test SEMSTOKES_PRESSURE_BUILDS[] == b0 + 2
    @test LAST_SEMSTOKES_PRESSURE_JUMP[] == 0.0              # eta did not move

    # ---- 5. the threshold is a cost knob, not a correctness knob -----------
    # Same eta field (`drift`), two thresholds: `Inf` reuses the factors built
    # for `base` (stale by 30%), 1e-12 rebuilds them.  The *answer* must be the
    # same; only the build count and (slightly) the iteration count may move.
    fz = (x, y) -> sin(pi * x / qt.Lx) * sin(pi * y)
    solve1(eta; kw...) = solve_semquad_stokes_var(ss, qt, eta, z2, fz, z2, z2;
                                                  bc=:free, linear=:mg,
                                                  mg_pressure_mode=:bfbt, kw...)
    drift = (x, y) -> 1.3 * base(x, y)
    SEMQUAD_PRESSURE_CACHE[] = nothing
    solve1(base)                                    # build the factors at base
    b0 = SEMSTOKES_PRESSURE_BUILDS[]
    uS, pS, = solve1(drift; mg_pressure_eta_tol=Inf)          # stale by 30%
    @test LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test 0.29 < LAST_SEMSTOKES_PRESSURE_JUMP[] < 0.31
    itS = LAST_SEMSTOKES_SCHUR_ITERS[]
    resS = LAST_SEMSTOKES_RESIDUAL[]
    uF, pF, = solve1(drift; mg_pressure_eta_tol=1.0e-12)      # fresh factors
    itF = LAST_SEMSTOKES_SCHUR_ITERS[]
    resF = LAST_SEMSTOKES_RESIDUAL[]
    @test SEMSTOKES_PRESSURE_BUILDS[] == b0 + 1               # only the second
    @test LAST_SEMSTOKES_RESIDUAL[] <= 1.0e-8
    du = norm(uS - uF) / max(norm(uF), 1.0)
    dp = norm(pS - pF) / max(norm(pF), 1.0)
    @info "G189 stale-at-Inf answer" du = du dp = dp it_stale = itS it_fresh = itF res_stale = resS res_fresh = resF
    @test du <= 1.0e-8
    @test dp <= 1.0e-8
    @test itS <= 1.1 * itF
end
