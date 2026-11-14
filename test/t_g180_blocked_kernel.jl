# G180: the blocked leaf kernel is bitwise the scalar reference apply.
﻿using Test, QuaST, LinearAlgebra, Random

# G180 / S3 stage D: the leaf-blocked SIMD kernel is the same operator as the
# scalar reference apply.
#
# Stage D processes `SEMQUAD_KBLK` leaves per kernel call with the block index
# innermost, so every `(p+1)^2` / `nq^2` inner loop becomes a unit-stride vector
# operation across independent leaves.  The per-leaf arithmetic is unchanged
# (same contraction order, same accumulation order), so the result must be
# *bitwise* identical to the scalar kernel -- anything else would silently
# change the preconditioner the gates have been validating.  This gate checks
# that on every geometry the matrix-free path runs on:
#
#   1. uniform trees at p=2 and p=4 (the cheap and the production degree),
#   2. a non-uniform AMR tree with hanging nodes (the blob/2b situation),
#   3. a wall-refined tree (mixed levels 3/4/5, free-slip boundaries),
#
# with constant, steep (16384x) and function-valued viscosity, and it also
# compares both kernels against the assembled sparse `K * v` (G175's 1e-12
# equivalence, re-checked for the blocked kernel).

const _G180_Z2 = (x, y) -> 0.0

function _g180_uniform(level::Int; cx::Int=1, cy::Int=1)
    qt = Quadtree(level; cells_x=cx, cells_y=cy)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

function _g180_amr(level::Int=4)
    # Refine the lower-left quarter one extra level: a genuinely graded tree
    # with hanging nodes on every interface.
    qt = _g180_uniform(level)
    for lf in sort(collect(qt.leaves))
        lf[2] < 2^(level - 1) && lf[3] < 2^(level - 1) && refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

function _g180_wall()
    qt = Quadtree(5)
    for _ in 1:3, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    refine_wall_to_min!(qt, 5; strip=0.1)
    return qt
end

# Assembled sparse K*v from the production assembler (`assemble_only`).
function _g180_sparse_K(ss, qt, eta)
    sys = solve_semquad_stokes_var(ss, qt, eta, _G180_Z2, _G180_Z2, _G180_Z2,
                                   _G180_Z2; bc=:free, linear=:direct,
                                   assemble_only=true)
    return sys.K
end

function _g180_kernel(ss, qt, eta, v; blk::Int)
    op = QuaST.semquad_kop(ss, qt)
    etav = QuaST.semquad_eta_field(op, eta)
    out = zeros(Float64, 2 * op.N)
    work = QuaST.SemQuadKWork(op.n, op.nq, 2 * op.N)
    QuaST._semquad_apply_K_raw!(out, op, etav, v, work; blk=blk)
    return out
end

@testset "G180 blocked leaf kernel == scalar reference apply" begin
    Random.seed!(20261106)
    BLK = QuaST.SEMQUAD_KBLK
    @test BLK >= 4                      # at least one SIMD register of doubles

    cases = [
        ("uniform p=2 L3", _g180_uniform(3), 2),
        ("uniform p=4 L3", _g180_uniform(3), 4),
        ("uniform 5x2 p=2 L3", _g180_uniform(3; cx=5, cy=2), 2),
        ("AMR p=4 (graded)", _g180_amr(4), 4),
        ("AMR p=2 (graded)", _g180_amr(3), 2),
        ("wall-refined p=2", _g180_wall(), 2),
    ]
    etas = [
        ("const", (x, y) -> 1.0),
        ("16384x", (x, y) -> exp(-log(16384.0) * (1.0 - y))),
        ("bilinear", (x, y) -> 1.0 + 3.0 * (1.0 - x) + 5.0 * y * y),
    ]
    worst_scalar = 0.0
    worst_blocked = 0.0
    for (name, qt, p) in cases
        ss = make_semstokes(qt, p)
        nleaves = length(qt.leaves)
        for (ename, eta) in etas
            K = _g180_sparse_K(ss, qt, eta)
            v = randn(2 * ss.su.N)
            us = _g180_kernel(ss, qt, eta, v; blk=0)
            ub = _g180_kernel(ss, qt, eta, v; blk=BLK)
            usparse = K * v
            ds = norm(ub - us) / norm(us)
            dsp = norm(us - usparse) / norm(usparse)
            dbp = norm(ub - usparse) / norm(usparse)
            # blocked vs scalar: same arithmetic, same order -> bitwise equal
            @test ub == us
            @test ds == 0.0
            @test dsp <= 1.0e-12
            @test dbp <= 1.0e-12
            worst_scalar = max(worst_scalar, dsp)
            worst_blocked = max(worst_blocked, dbp)
            @info "G180" case = name eta = ename leaves = nleaves p = p biased = ds sparse_vs_scalar = dsp sparse_vs_blocked = dbp
        end
    end
    @test worst_scalar <= 1.0e-12
    @test worst_blocked <= 1.0e-12
    @info "G180 worst over all cases" scalar_vs_sparse = worst_scalar blocked_vs_sparse = worst_blocked

    # A block that does not divide the leaf count must still be exact: drop the
    # block size to 5 on a 64-leaf tree (last block has 4 leaves) and to 3.
    qt = _g180_uniform(3)
    ss = make_semstokes(qt, 4)
    eta = etas[2][2]
    v = randn(2 * ss.su.N)
    uref = _g180_kernel(ss, qt, eta, v; blk=0)
    for blk in (2, 3, 5, 7, 9, 64, 1000)
        @test _g180_kernel(ss, qt, eta, v; blk=blk) == uref
    end

    # `deviatoric=true` uses the second c2 branch of the four separable pieces.
    ssv = make_semstokes(qt, 4)
    opt = QuaST.semquad_kop(ssv, qt; deviatoric=true)
    etav = QuaST.semquad_eta_field(opt, eta)
    w1 = QuaST.SemQuadKWork(opt.n, opt.nq, 2 * opt.N)
    w2 = QuaST.SemQuadKWork(opt.n, opt.nq, 2 * opt.N)
    o1 = zeros(Float64, 2 * opt.N); o2 = zeros(Float64, 2 * opt.N)
    QuaST._semquad_apply_K_raw!(o1, opt, etav, v, w1; blk=0)
    QuaST._semquad_apply_K_raw!(o2, opt, etav, v, w2; blk=BLK)
    @test o2 == o1
end
