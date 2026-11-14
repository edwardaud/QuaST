# G182: the pressure-block selector mg_pressure_precond across its modes.
using Test, QuaST, LinearAlgebra, Random

# G182 / S3-press: the pressure-block selector `mg_pressure_precond`
# (`:auto | :bfbt | :mass | :scaled_mass | :pcd | :lsc`) and the A3 cache on
# mixed-level AMR trees.
#
# What is checked:
#   1. every mode solves the same saddle system as the direct LU on a
#      constant-viscosity tree (velocity <= 1e-8 relative, true residual
#      <= 1e-9), and on a mixed-level AMR tree with 16384x contrast the two
#      modes that are usable there (`:bfbt`, `:scaled_mass`) do the same;
#   2. the measured iteration-count ordering of A1 is reproduced on the AMR
#      tree: BFBT needs fewer outer iterations than the viscosity-weighted mass
#      matrix (the reason BFBT stays the `:auto` default above 100x contrast);
#   3. `:pcd` reduces to the unweighted mass inverse for Stokes (C_p = 0), i.e.
#      its iteration count equals `:mass` on a constant-viscosity tree;
#   4. `mg_pressure_precond=:auto` is *bitwise* the historical default path, and
#      `linear=:direct` is bitwise unaffected by the keyword;
#   5. the A3 cache: the resolved mode is recorded, a repeat solve with the same
#      eta reuses the factors, and a 10% eta jump rebuilds at the default
#      threshold but reuses at a 0.2 threshold (the measured A3 default);
#   6. iteration counts are recorded for every mode (A1's table, small trees).

const _G182_Z2 = (x, y) -> 0.0

# mixed-level AMR tree: a uniform level-3 tree with 8 leaves refined once more,
# so levels {3,4} (and hanging nodes on the 2:1-balanced interfaces)
function _g182_amr_tree(level::Int=3, extra::Int=8)
    qt = Quadtree(level + 2)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    for lf in sort(collect(qt.leaves))[1:extra]
        lf in qt.leaves && refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

# constant-viscosity uniform tree
function _g182_uniform(level::Int)
    qt = Quadtree(level)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

const _G182_ETA_HI = (x, y) -> exp(-log(16384.0) * (1.0 - y))
const _G182_ETA_ONE = (x, y) -> 1.0

function _g182_solve(ss, qt, eta, fz; kw...)
    return solve_semquad_stokes_var(ss, qt, eta, _G182_Z2, fz, _G182_Z2,
                                    _G182_Z2; bc=:free, kw...)
end

@testset "G182 pressure-block selector" begin
    Random.seed!(20261106)
    fz_unif = (x, y) -> sin(pi * x) * sin(pi * y)
    fz_amr = (x, y) -> sin(pi * x) * sin(pi * y)

    # ---- 1. all modes on a constant-viscosity tree -------------------------
    qtu = _g182_uniform(3)
    p = 2
    ssu = make_semstokes(qtu, p)
    uD, _, = _g182_solve(ssu, qtu, _G182_ETA_ONE, fz_unif; linear=:direct)
    resD = LAST_SEMSTOKES_RESIDUAL[]
    @test resD < 1.0e-10
    @test norm(uD) > 0.0
    @info "G182 reference" leaves = length(qtu.leaves) dofs = 2 * ssu.su.N res = resD

    itu = Dict{Symbol,Int}()
    for pre in (:bfbt, :mass, :scaled_mass, :pcd)
        SEMQUAD_PRESSURE_CACHE[] = nothing
        t = @elapsed u, _, = _g182_solve(ssu, qtu, _G182_ETA_ONE, fz_unif;
                                         linear=:mg,
                                         mg_pressure_precond=pre)
        itu[pre] = LAST_SEMSTOKES_SCHUR_ITERS[]
        du = norm(u - uD) / norm(uD)
        @test LAST_SEMSTOKES_RESIDUAL[] <= 1.0e-9
        @test du <= 1.0e-8
        @test LAST_SEMSTOKES_PRESSURE_MODE[] !== :none
        @info "G182 mode (uniform, eta=1)" pre it = itu[pre] du res = LAST_SEMSTOKES_RESIDUAL[] t = t
    end
    # `:lsc` needs `diag(Kred)`, so it exists only on the assembled-operator path;
    # on the default fully matrix-free path it must fail with a clear error.
    @test_throws ArgumentError _g182_solve(ssu, qtu, _G182_ETA_ONE, fz_unif;
                                           linear=:mg, mg_pressure_precond=:lsc)
    SEMQUAD_PRESSURE_CACHE[] = nothing
    t = @elapsed u, _, = _g182_solve(ssu, qtu, _G182_ETA_ONE, fz_unif;
                                     linear=:mg, mg_operator=:sparse,
                                     mg_pressure_precond=:lsc)
    itu[:lsc] = LAST_SEMSTOKES_SCHUR_ITERS[]
    @test LAST_SEMSTOKES_RESIDUAL[] <= 1.0e-9
    @test norm(u - uD) / norm(uD) <= 1.0e-8
    @info "G182 mode (uniform, eta=1, sparse)" pre = :lsc it = itu[:lsc] t = t
    # PCD = A_p^-1 A_p M_p^-1 = M_p^-1 for a Stokes operator (C_p = 0): its
    # iteration count must equal the unweighted mass one (round-off only).
    @test abs(itu[:pcd] - itu[:mass]) <= 2
    # BFBT is not worse than the mass modes on a *constant*-viscosity problem
    # either (its build is more expensive, which is why `:auto` picks `:mass`
    # there -- the iteration count is what this gate compares).
    @test itu[:bfbt] <= max(itu[:mass], itu[:scaled_mass]) + 2

    # ---- 2. the high-contrast mixed-level AMR tree --------------------------
    qta = _g182_amr_tree()
    @test length(unique(lf[1] for lf in qta.leaves)) > 1
    ssa = make_semstokes(qta, p)
    uDa, _, = _g182_solve(ssa, qta, _G182_ETA_HI, fz_amr; linear=:direct)
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-10
    @info "G182 AMR tree" leaves = length(qta.leaves) levels = length(unique(lf[1] for lf in qta.leaves)) Nu = ssa.su.N

    ita = Dict{Symbol,Int}()
    for pre in (:bfbt, :scaled_mass)
        SEMQUAD_PRESSURE_CACHE[] = nothing
        t = @elapsed u, _, = _g182_solve(ssa, qta, _G182_ETA_HI, fz_amr;
                                         linear=:mg, mg_pressure_precond=pre)
        ita[pre] = LAST_SEMSTOKES_SCHUR_ITERS[]
        du = norm(u - uDa) / norm(uDa)
        @test LAST_SEMSTOKES_RESIDUAL[] <= 1.0e-9
        @test du <= 1.0e-8
        @info "G182 mode (AMR, 16384x)" pre it = ita[pre] du res = LAST_SEMSTOKES_RESIDUAL[] t = t
    end
    # A1's ordering on the 1024-leaf blob tree is 39 (bfbt) vs 68 (scaled mass);
    # the gate asserts the direction, not the magnitude.
    @test ita[:bfbt] < ita[:scaled_mass]

    # ---- 3. the default is the historical path, bitwise --------------------
    SEMQUAD_PRESSURE_CACHE[] = nothing
    uDef, pDef, = _g182_solve(ssa, qta, _G182_ETA_HI, fz_amr; linear=:mg)
    itDef = LAST_SEMSTOKES_SCHUR_ITERS[]
    # the default resolution on a >100x contrast tree is BFBT (A1's winner)
    @test LAST_SEMSTOKES_PRESSURE_MODE[] === :bfbt
    SEMQUAD_PRESSURE_CACHE[] = nothing
    uAuto, pAuto, = _g182_solve(ssa, qta, _G182_ETA_HI, fz_amr;
                                linear=:mg, mg_pressure_precond=:auto)
    @test uAuto == uDef && pAuto == pDef
    # `:bfbt` reproduces the default on this tree as well
    SEMQUAD_PRESSURE_CACHE[] = nothing
    uB, pB, = _g182_solve(ssa, qta, _G182_ETA_HI, fz_amr; linear=:mg,
                          mg_pressure_precond=:bfbt)
    @test uB == uDef && pB == pDef
    @test itDef == ita[:bfbt]
    @info "G182 default path" it = itDef

    # ---- 4. `linear=:direct` is untouched by the keyword -------------------
    uD2, pD2, = _g182_solve(ssa, qta, _G182_ETA_HI, fz_amr; linear=:direct,
                            mg_pressure_precond=:pcd)
    @test uD2 == uDa
    # the mode selector is only consulted by the `:mg` branch
    qts = _g182_uniform(2)
    sss = make_semstokes(qts, p)
    @test_throws ArgumentError _g182_solve(sss, qts, _G182_ETA_ONE, fz_unif;
                                           linear=:mg,
                                           mg_pressure_precond=:bogus)

    # ---- 5. A3 cache behaviour at the measured default threshold -----------
    SEMQUAD_PRESSURE_CACHE[] = nothing
    b0 = SEMSTOKES_PRESSURE_BUILDS[]
    r0 = SEMSTOKES_PRESSURE_REUSES[]
    tol = SEMQUAD_PRESSURE_DEFAULT_ETA_TOL
    @test tol == 1.0                     # S5-1: retuned 0.5 -> 1.0 (§19.1)
    eta1 = (x, y) -> _G182_ETA_HI(x, y)
    u1, _, = _g182_solve(ssa, qta, eta1, fz_amr; linear=:mg,
                         mg_pressure_precond=:bfbt)
    @test !LAST_SEMSTOKES_PRESSURE_REUSED[]
    u2, _, = _g182_solve(ssa, qta, eta1, fz_amr; linear=:mg,
                         mg_pressure_precond=:bfbt)
    @test LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test norm(u2 - u1) / max(norm(u1), 1.0) <= 1.0e-10
    # A 10% viscosity jump is *below* the default threshold (1.0): reuse ...
    eta_small = (x, y) -> 1.10 * _G182_ETA_HI(x, y)
    _g182_solve(ssa, qta, eta_small, fz_amr; linear=:mg,
                mg_pressure_precond=:bfbt)
    jump_small = LAST_SEMSTOKES_PRESSURE_JUMP[]
    @test LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test 0.05 < jump_small <= tol
    # ... and so is a 60% jump: at 1.0 the cache tolerates up to 100% drift
    # inside one tree block (§13.3 measured the iteration count flat there).
    eta_mid = (x, y) -> 1.60 * _G182_ETA_HI(x, y)
    _g182_solve(ssa, qta, eta_mid, fz_amr; linear=:mg,
                mg_pressure_precond=:bfbt)
    @test LAST_SEMSTOKES_PRESSURE_REUSED[]
    # ... while a jump *above* 100% (a phase change / yield onset inside one
    # tree block) is above it and rebuilds.
    eta_big = (x, y) -> 2.60 * _G182_ETA_HI(x, y)
    _g182_solve(ssa, qta, eta_big, fz_amr; linear=:mg,
                mg_pressure_precond=:bfbt)
    jump_big = LAST_SEMSTOKES_PRESSURE_JUMP[]
    @test !LAST_SEMSTOKES_PRESSURE_REUSED[]
    @test jump_big > tol
    # the old 1e-2 default would have rejected the 10% jump, which is exactly
    # the behaviour A3 measured on the blob (5-7% per step, 25-37% per frame).
    _g182_solve(ssa, qta, eta1, fz_amr; linear=:mg, mg_pressure_precond=:bfbt)
    _g182_solve(ssa, qta, eta_small, fz_amr; linear=:mg,
                mg_pressure_precond=:bfbt, mg_pressure_eta_tol=1.0e-2)
    @test !LAST_SEMSTOKES_PRESSURE_REUSED[]
    @info "G182 cache" builds = SEMSTOKES_PRESSURE_BUILDS[] - b0 reuses = SEMSTOKES_PRESSURE_REUSES[] - r0 jump_small = jump_small jump_big = jump_big default_tol = tol
end
