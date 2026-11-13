using Test, MantleSEM, LinearAlgebra, Random

# G179 / S3 stage A: correctness of the cached pressure-block approximate inverse.
#
# The three-factor w-BFBT of the `linear=:mg` saddle solve costs ~1.1 s on the
# 1024-leaf blob tree and is rebuilt on every Stokes solve, although it depends
# only on the tree, the pressure space and the viscosity field.  Stage A caches
# it, keyed by the tree/geometry fingerprint (`_saddle_key`) and invalidated when
# the sampled viscosity has moved by more than `mg_pressure_eta_tol`.
#
# The gate checks the three behaviours that make the cache safe:
#   1. the SAME tree and the SAME eta reuses the factors -- and the reused
#      factors are bitwise the ones a fresh build produces (no aliasing, no
#      accidental use of a different tree's factors);
#   2. an eta jump above the threshold REBUILDS (a stale factor beyond the
#      tolerance is not silently accepted), and a jump below it reuses;
#   3. a TREE change rebinds the cache (the fingerprint covers the leaf set), and
#      so does a mode switch;
#   4. through the public API (`linear=:mg`), a second identical solve reuses the
#      factors and produces the same solution.

const _G179_Z2 = (x, y) -> 0.0

function _g179_tree(level::Int; cx::Int=1, cy::Int=1, extra::Int=0)
    # `maxlevel = level + 1` so that `extra` leaves can be refined once more,
    # which makes a genuinely different tree of the same base resolution.
    qt = Quadtree(level + 1; cells_x=cx, cells_y=cy)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    for lf in sort(collect(qt.leaves))[1:extra]
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

function _g179_system(qt, p::Int, eta)
    ss = make_semstokes(qt, p)
    sys = solve_semquad_stokes_var(ss, qt, eta, _G179_Z2, _G179_Z2, _G179_Z2,
                                   _G179_Z2; bc=:free, linear=:direct,
                                   assemble_only=true)
    return ss, sys
end

function _g179_precond(ss, sys, qt, p::Int, eta; mode::Symbol=:bfbt, kw...)
    key = MantleSEM._saddle_key(qt, p, :free, size(sys.Cu, 2), size(sys.Cp, 2))
    return MantleSEM.semquad_mg_pressure_precond(ss, sys.Cu, sys.Cp, sys.Bs,
                                                 sys.Bt, sys.Kred, eta;
                                                 mode=mode, key=key, kw...)
end

# Action of the pressure approximate inverse on one vector; identical factors
# must give identical action.  Goes through the production code path
# (`_semquad_pressure_block!`) so it covers both the sparse and the matrix-free
# `S2` (stage B) and the plain Cholesky modes.
function _g179_action(Fp, r)
    nf = Fp isa NamedTuple ? max(size(Fp.Bt, 2), size(Fp.Bs, 2)) : 0
    wP = zeros(Float64, nf + length(r))
    rp = copy(r)
    rp2 = similar(r)
    s1 = zeros(Float64, nf)
    s2 = zeros(Float64, nf)
    MantleSEM._semquad_pressure_block!(wP, rp, rp2, s1, s2, Fp, 1.0, nf)
    return wP[(nf + 1):end]
end

@testset "G179 cached pressure preconditioner" begin
    Random.seed!(20261106)

    # 16 leaves at p=2: 2x 16384 contrast so `:auto` would pick BFBT anyway, but
    # the gate forces `mode=:bfbt` to exercise the expensive path.
    qt = _g179_tree(2)
    p = 2
    # exp(-b (1-y)): 16384x from top to bottom, the 2b law's shape.
    eta = (x, y) -> exp(-log(16384.0) * (1.0 - y))
    ss, sys = _g179_system(qt, p, eta)
    nfp = size(sys.Cp, 2)
    r = randn(nfp)

    # ---- 1. same tree + same eta: reuse, and the action is identical --------
    SEMQUAD_PRESSURE_CACHE[] = nothing
    b0 = SEMSTOKES_PRESSURE_BUILDS[]
    r0 = SEMSTOKES_PRESSURE_REUSES[]
    F1, ps1 = _g179_precond(ss, sys, qt, p, eta)
    @test !LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test SEMSTOKES_PRESSURE_BUILDS[] == b0 + 1
    a1 = _g179_action(F1, r)
    F2, ps2 = _g179_precond(ss, sys, qt, p, eta)
    @test LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test SEMSTOKES_PRESSURE_BUILDS[] == b0 + 1        # no rebuild
    @test SEMSTOKES_PRESSURE_REUSES[] == r0 + 1
    @test ps2 == ps1
    a2 = _g179_action(F2, r)
    # The published guarantee is "same tree + eta reuses"; the reused factors
    # are the same object, so the action is bitwise identical.
    @test a2 == a1
    @info "G179 reuse" nfp = nfp builds = SEMSTOKES_PRESSURE_BUILDS[] -
          b0 reuses = SEMSTOKES_PRESSURE_REUSES[] - r0

    # ---- 2. eta jump above / below the threshold ---------------------------
    # The jump is measured against the *cached* viscosity, so the sequence below
    # is stated in terms of the change from the previous accepted build.
    eta5 = (x, y) -> 1.05 * exp(-log(16384.0) * (1.0 - y))    # +5% everywhere
    _g179_precond(ss, sys, qt, p, eta5; eta_tol=1.0e-2)       # explicit 1e-2
    jump5 = LAST_SEMSTOKES_PRESSURE_JUMP[]
    @test !LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test jump5 > 1.0e-2
    @test SEMSTOKES_PRESSURE_BUILDS[] == b0 + 2
    # ... and with a tolerance above the jump it reuses the new factors
    _g179_precond(ss, sys, qt, p, eta5; eta_tol=0.1)
    @test LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test SEMSTOKES_PRESSURE_BUILDS[] == b0 + 2
    # a 4.3% jump back to the 1.005 field is above a 1e-2 tolerance
    eta1 = (x, y) -> 1.005 * exp(-log(16384.0) * (1.0 - y))
    _g179_precond(ss, sys, qt, p, eta1; eta_tol=1.0e-2)
    @test !LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test LAST_SEMSTOKES_PRESSURE_JUMP[] > 1.0e-2
    @test SEMSTOKES_PRESSURE_BUILDS[] == b0 + 3
    # ... and a 0.3% nudge of the same field is below it: reuse
    eta1b = (x, y) -> 1.003 * 1.005 * exp(-log(16384.0) * (1.0 - y))
    _g179_precond(ss, sys, qt, p, eta1b; eta_tol=1.0e-2)
    @test LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test 0.0 < LAST_SEMSTOKES_PRESSURE_JUMP[] <= 1.0e-2
    @test SEMSTOKES_PRESSURE_BUILDS[] == b0 + 3
    # the same tolerance rejects a 4.7% jump from the cached state
    _g179_precond(ss, sys, qt, p, eta5; eta_tol=1.0e-2)
    @test !LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test SEMSTOKES_PRESSURE_BUILDS[] == b0 + 4
    @info "G179 eta threshold" jump5 = jump5

    # ---- 3. tree change and mode switch rebind the cache -------------------
    qt2 = _g179_tree(2; extra=1)                 # one leaf refined: other leaves
    @test Set(qt2.leaves) != Set(qt.leaves)
    eta1f = (x, y) -> 1.005 * exp(-log(16384.0) * (1.0 - y))
    ss2, sys2 = _g179_system(qt2, p, eta1f)
    _g179_precond(ss2, sys2, qt2, p, eta1f)      # same eta, but a new key
    @test !LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test SEMSTOKES_PRESSURE_BUILDS[] == b0 + 5
    _g179_precond(ss2, sys2, qt2, p, eta1f)
    @test LAST_SEMSTOKES_PRESSURE_REUSED[]
    Fm, _ = _g179_precond(ss2, sys2, qt2, p, eta1f; mode=:mass)
    @test !LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test !(Fm isa NamedTuple)
    @test SEMSTOKES_PRESSURE_BUILDS[] == b0 + 6
    # A mode switch never reuses the other mode's factors.
    _g179_precond(ss2, sys2, qt2, p, eta1f; mode=:mass)
    @test LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test SEMSTOKES_PRESSURE_BUILDS[] == b0 + 6
    # `cache=false` never touches or reuses the cache.
    nb = SEMSTOKES_PRESSURE_BUILDS[]
    _g179_precond(ss2, sys2, qt2, p, eta1f; mode=:mass, cache=false)
    @test !LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test SEMSTOKES_PRESSURE_BUILDS[] == nb + 1

    # ---- 4. through the public `linear=:mg` API ---------------------------
    # The matrix-free S2 of stage B is the same operator as the formed one: a
    # fresh build with `bfbt_s2=:matrix` must give the same action.
    SEMQUAD_PRESSURE_CACHE[] = nothing
    F3, _ = _g179_precond(ss, sys, qt, p, eta; bfbt_s2=:matrix)
    @test !LAST_SEMSTOKES_PRESSURE_REUSED[]
    a3 = _g179_action(F3, r)
    @test norm(a3 - a1) / norm(a1) <= 1.0e-12
    @info "G179 S2 matrix-free vs formed" rel = norm(a3 - a1) / norm(a1)

    SEMQUAD_PRESSURE_CACHE[] = nothing
    fz = (x, y) -> sin(pi * x / qt.Lx) * sin(pi * y)
    solve_mg(eta; kw...) = solve_semquad_stokes_var(ss, qt, eta, _G179_Z2, fz,
                                                   _G179_Z2, _G179_Z2;
                                                   bc=:free, linear=:mg,
                                                   mg_pressure_mode=:bfbt, kw...)
    u1, p1, = solve_mg(eta)
    @test !LAST_SEMSTOKES_PRESSURE_REUSED[]
    u2, p2, = solve_mg(eta)
    @test LAST_SEMSTOKES_PRESSURE_REUSED[]
    # Same preconditioner, same right-hand side: the two solves agree far below
    # the Stokes tolerance (a cached factor must not perturb the answer).
    @test norm(u2 - u1) / max(norm(u1), 1.0) <= 1.0e-10
    @test norm(p2 - p1) / max(norm(p1), 1.0) <= 1.0e-10
    # ... and a 5% eta jump rebuilds at a 1e-2 tolerance while still converging
    # to the same answer.
    u3, _, = solve_mg(eta5; mg_pressure_eta_tol=1.0e-2)
    @test !LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test LAST_SEMSTOKES_RESIDUAL[] <= 1.0e-8
    @info "G179 API" du = norm(u2 - u1) / max(norm(u1), 1.0) builds =
          SEMSTOKES_PRESSURE_BUILDS[] reuses = SEMSTOKES_PRESSURE_REUSES[]

    # ---- 5. the S5-1 default 1.0 and the `Inf` opt-in ---------------------
    # (docs/dev/SOLVER.md §19.1) The default moved 0.5 -> 1.0 because §13.3's
    # curve is flat in the iteration count while the build count still halves;
    # `Inf` is supported as an explicit opt-in and never rebuilds on eta drift,
    # but the tree fingerprint always wins.
    @test SEMQUAD_PRESSURE_DEFAULT_ETA_TOL == 1.0
    SEMQUAD_PRESSURE_CACHE[] = nothing
    nb = SEMSTOKES_PRESSURE_BUILDS[]
    eta30 = (x, y) -> 1.3 * exp(-log(16384.0) * (1.0 - y))     # +30% everywhere
    _g179_precond(ss, sys, qt, p, eta)                  # build at the default
    @test SEMSTOKES_PRESSURE_BUILDS[] == nb + 1
    _g179_precond(ss, sys, qt, p, eta30)                # 0.30 <= 1.0: reuse
    @test LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test 0.29 < LAST_SEMSTOKES_PRESSURE_JUMP[] < 0.31
    @test SEMSTOKES_PRESSURE_BUILDS[] == nb + 1
    # A synthetic *sudden* jump above the threshold (a phase change / yield
    # onset inside one tree block: eta grows 2.5x) still rebuilds at 1.0.
    eta250 = (x, y) -> 2.5 * exp(-log(16384.0) * (1.0 - y))
    _g179_precond(ss, sys, qt, p, eta250)
    @test !LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test LAST_SEMSTOKES_PRESSURE_JUMP[] > 1.0
    @test SEMSTOKES_PRESSURE_BUILDS[] == nb + 2
    # `Inf` accepts that same sequence without rebuilding ...
    _g179_precond(ss, sys, qt, p, eta; eta_tol=Inf)
    @test LAST_SEMSTOKES_PRESSURE_REUSED[]
    eta1000 = (x, y) -> 1000.0 * exp(-log(16384.0) * (1.0 - y))
    _g179_precond(ss, sys, qt, p, eta1000; eta_tol=Inf)   # unbounded staleness
    @test LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test LAST_SEMSTOKES_PRESSURE_JUMP[] > 100.0
    @test SEMSTOKES_PRESSURE_BUILDS[] == nb + 2
    # ... but a TREE change still rebuilds at `Inf`: the fingerprint wins.
    ss2, sys2 = _g179_system(qt2, p, eta1000)
    _g179_precond(ss2, sys2, qt2, p, eta1000; eta_tol=Inf)
    @test !LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test SEMSTOKES_PRESSURE_BUILDS[] == nb + 3
    @test LAST_SEMSTOKES_PRESSURE_JUMP[] == 0.0        # eta did not move
    # the `Inf` path through the public API still converges to the same answer
    u4, p4, = solve_mg(eta30; mg_pressure_eta_tol=Inf)
    @test LAST_SEMSTOKES_RESIDUAL[] <= 1.0e-8
    u5, p5, = solve_mg(eta; mg_pressure_eta_tol=Inf)
    @test norm(u5 - u1) / max(norm(u1), 1.0) <= 1.0e-10
    @test norm(p5 - p1) / max(norm(p1), 1.0) <= 1.0e-10
    @info "G179 S5-1 thresholds" default_tol = SEMQUAD_PRESSURE_DEFAULT_ETA_TOL builds = SEMSTOKES_PRESSURE_BUILDS[] - nb
end
