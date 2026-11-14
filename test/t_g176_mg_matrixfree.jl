# G176: the matrix-free velocity apply inside the multigrid smoother.
using Test, QuaST, LinearAlgebra, SparseArrays, Random

# G176 / S3-MF-2: the matrix-free velocity apply inside the multigrid smoother.
#
# `make_semquad_mg(...; operator=:matrixfree)` replaces every level's
# `mul!(y, A[k], x)` by MF-1's `semquad_apply_Kred!`.  Because the matrix-free
# level operator *is* the rediscretised sparse one to round-off (G175), the
# Jacobi diagonal and therefore the whole V-cycle must be unchanged: the two
# hierarchies have to agree as preconditioners to Krylov tolerance.
#
# The default is `operator=:sparse` at the `make_semquad_mg` level;
# `solve_semquad_stokes_var` exposes the new path as `mg_operator=:matrixfree`
# (which became the default for `linear=:mg` in S3/MF-4, see G177).
#
# Acceptance (S3 design, MF-2): one V-cycle at p=4 level 4 < 40 ms.

const _G176_Z2 = (x, y) -> 0.0
const _G176_ONE = (x, y) -> 1.0

function _g176_uniform(level::Int; cx::Int=1, cy::Int=1)
    qt = Quadtree(level; cells_x=cx, cells_y=cy)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

function _g176_build(ss, qt, p, Kred, operator::Symbol)
    return QuaST.make_semquad_mg(ss, qt, p, :free, false, _G176_ONE, Kred;
                                     cycles=2, operator=operator)
end

function _g176_time(f, nrep::Int)
    f()
    t = @elapsed for _ in 1:nrep
        f()
    end
    return 1000.0 * t / nrep
end

@testset "G176 matrix-free MG smoother" begin
    Random.seed!(20251107)

    # ---- the two hierarchies are the same preconditioner -------------------
    for (level, cx, cy) in ((4, 1, 1), (4, 5, 2), (3, 1, 1))
        qt = _g176_uniform(level; cx=cx, cy=cy)
        ss = make_semstokes(qt, 4)
        asm = solve_semquad_stokes_var(ss, qt, _G176_ONE, _G176_Z2, _G176_Z2,
                                       _G176_Z2, _G176_Z2;
                                       bc=:free, linear=:direct,
                                       assemble_only=true)
        Kred = asm.Kred
        mgS = _g176_build(ss, qt, 4, Kred, :sparse)
        mgF = _g176_build(ss, qt, 4, Kred, :matrixfree)
        @test mgF.operator === :matrixfree
        @test mgF.D[1] == mgS.D[1] || isapprox(mgF.D[1], mgS.D[1], rtol=1e-12)
        b = randn(size(Kred, 1))
        zS = QuaST.semquad_mg_apply(mgS, b)
        zF = QuaST.semquad_mg_apply(mgF, b)
        d = norm(zS - zF) / norm(zS)
        @test d < 1.0e-10
        @info "G176 preconditioner" (level, cx, cy) leaves = length(qt.leaves) diff = d

        # cost, warm (JIT already done above)
        aS = _g176_time(() -> QuaST.semquad_mg_apply(mgS, b), 30)
        aF = _g176_time(() -> QuaST.semquad_mg_apply(mgF, b), 30)
        zz = zeros(length(b))
        vS = _g176_time(() -> QuaST._semquad_mg_vcycle!(zz, b, mgS,
                                                            mgS.geo.L + 1), 30)
        vF = _g176_time(() -> QuaST._semquad_mg_vcycle!(zz, b, mgF,
                                                            mgF.geo.L + 1), 30)
        @info "G176 cost" (level, cx, cy) apply2V_sparse_ms = aS apply2V_mf_ms = aF vcycle_sparse_ms = vS vcycle_mf_ms = vF
        # acceptance: one V-cycle at p=4 level 4 (256 leaves, 8450 velocity
        # dofs) must be well under 40 ms (MF-2) and under 15 ms (MF-3), see the
        # design's MF-2/MF-3 acceptance rows.
        level == 4 && cx == 1 && cy == 1 && @test vF < 40.0
        level == 4 && cx == 1 && cy == 1 && @test vF < 15.0
        level == 4 && cx == 1 && cy == 1 && @test aF < 40.0
        # the matrix-free path must never be catastrophically slower
        @test aF < 3.0 * aS

        # ---- MF-3: no sparse velocity operator anywhere in the hierarchy ----
        @test isempty(mgF.A)
        @test isempty(mgF.LF) && isempty(mgF.UF)
        @test mgF.cA isa LinearAlgebra.Cholesky
        @test length(mgF.mfops) == mgF.geo.L + 1
        @test mgF.mfops[1] isa SemQuadKOp
        # every matrix-free level operator is the sparse rediscretised one
        for k in 1:(mgF.geo.L + 1)
            x = randn(mgF.geo.nfs[k])
            yF = zeros(mgF.geo.nfs[k])
            yS = zeros(mgF.geo.nfs[k])
            QuaST._semquad_mg_apply!(yF, x, mgF, k)
            mul!(yS, mgS.A[k], x)
            @test norm(yF - yS) / norm(yS) < 1.0e-12
        end
        # ... and the Jacobi diagonal agrees on a conforming tree
        for k in 1:(mgF.geo.L + 1)
            @test mgF.D[k] ≈ mgS.D[k] rtol = 1.0e-12
        end
        @test all(>(0.0), mgF.D[end])
    end

    # ---- stronger viscosity contrast, same operator -----------------------
    qt = _g176_uniform(4)
    ss = make_semstokes(qt, 4)
    eta = (x, y) -> exp(-log(16384.0) * (1 - y))
    asm = solve_semquad_stokes_var(ss, qt, eta, _G176_Z2, _G176_Z2, _G176_Z2,
                                   _G176_Z2; bc=:free, linear=:direct,
                                   assemble_only=true)
    mgS = QuaST.make_semquad_mg(ss, qt, 4, :free, false, eta, asm.Kred;
                                    cycles=2, operator=:sparse)
    mgF = QuaST.make_semquad_mg(ss, qt, 4, :free, false, eta, asm.Kred;
                                    cycles=2, operator=:matrixfree)
    b = randn(size(asm.Kred, 1))
    @test norm(QuaST.semquad_mg_apply(mgS, b) -
               QuaST.semquad_mg_apply(mgF, b)) /
          norm(QuaST.semquad_mg_apply(mgS, b)) < 1.0e-10
    # the viscosity is sampled at the same points as the assembler, so even the
    # 16384x operator agrees level-by-level
    yS = zeros(length(b)); yF = zeros(length(b))
    QuaST._semquad_mg_apply!(yS, b, mgS, mgS.geo.L + 1)
    QuaST._semquad_mg_apply!(yF, b, mgF, mgF.geo.L + 1)
    @test norm(yS - yF) / norm(yS) < 1.0e-12

    # ---- end-to-end: the solved velocity is the same ----------------------
    Lx = qt.Lx
    fz = (x, y) -> sin(pi * x / Lx) * sin(pi * y)
    uD, _, _, _, _ = solve_semquad_stokes_var(ss, qt, eta, _G176_Z2, fz,
                                              _G176_Z2, _G176_Z2;
                                              bc=:free, linear=:direct)
    uS, _, _, _, _ = solve_semquad_stokes_var(ss, qt, eta, _G176_Z2, fz,
                                              _G176_Z2, _G176_Z2;
                                              bc=:free, linear=:mg,
                                              mg_operator=:sparse)
    resS = LAST_SEMSTOKES_RESIDUAL[]; itS = LAST_SEMSTOKES_SCHUR_ITERS[]
    uF, _, _, _, _ = solve_semquad_stokes_var(ss, qt, eta, _G176_Z2, fz,
                                              _G176_Z2, _G176_Z2;
                                              bc=:free, linear=:mg,
                                              mg_operator=:matrixfree)
    resF = LAST_SEMSTOKES_RESIDUAL[]; itF = LAST_SEMSTOKES_SCHUR_ITERS[]
    @test norm(uD) > 1.0e-8
    @test resF < 1.0e-9
    @test resS < 1.0e-9
    # The two hierarchies differ only by round-off, so the flexible GMRES (with
    # its true-residual stopping criterion) may stop one iteration apart.
    @test abs(itF - itS) <= 2
    @test norm(uF - uD) / norm(uD) < 1.0e-8
    @test norm(uF - uS) / norm(uS) < 1.0e-8
    @info "G176 end-to-end" direct_vs_mf = norm(uF - uD) / norm(uD) iters = itF res = resF

    # ---- the default is now the matrix-free path (MF-4) -------------------
    # `mg_operator` defaults to :matrixfree, so a solve that does not pass it is
    # the matrix-free one; `:sparse` stays available as an explicit option.
    u0, _, _, _, _ = solve_semquad_stokes_var(ss, qt, eta, _G176_Z2, fz,
                                              _G176_Z2, _G176_Z2;
                                              bc=:free, linear=:mg)
    @test u0 == uF
    @test_throws ArgumentError solve_semquad_stokes_var(ss, qt, eta, _G176_Z2,
                                                        fz, _G176_Z2, _G176_Z2;
                                                        bc=:free, linear=:mg,
                                                        mg_operator=:bogus)

    # ---- non-uniform tree: rediscretised coarse levels, still MF -----------
    qtw = Quadtree(4; cells_x=5, cells_y=2)
    for _ in 1:2, lf in collect(qtw.leaves)
        refine!(qtw, lf)
    end
    balance!(qtw)
    refine_wall_to_min!(qtw, 4; strip=0.1)
    balance!(qtw)
    @test length(unique(lf[1] for lf in qtw.leaves)) > 1
    ssw = make_semstokes(qtw, 2)
    etaw = (x, y) -> exp(-log(16384.0) * (1 - y))
    Lxw = qtw.Lx
    fzw = (x, y) -> sin(pi * x / Lxw) * sin(pi * y)
    uwD, _, _, _, _ = solve_semquad_stokes_var(ssw, qtw, etaw, _G176_Z2, fzw,
                                               _G176_Z2, _G176_Z2;
                                               bc=:free, linear=:direct)
    uwS, _, _, _, _ = solve_semquad_stokes_var(ssw, qtw, etaw, _G176_Z2, fzw,
                                               _G176_Z2, _G176_Z2;
                                               bc=:free, linear=:mg,
                                               mg_operator=:sparse)
    res_wS = LAST_SEMSTOKES_RESIDUAL[]
    uwF, _, _, _, _ = solve_semquad_stokes_var(ssw, qtw, etaw, _G176_Z2, fzw,
                                               _G176_Z2, _G176_Z2;
                                               bc=:free, linear=:mg,
                                               mg_operator=:matrixfree)
    res_wF = LAST_SEMSTOKES_RESIDUAL[]
    @test norm(uwD) > 1.0e-8
    @test res_wF < 1.0e-9
    @test res_wS < 1.0e-9
    @test norm(uwS - uwD) / norm(uwD) < 1.0e-8
    @test norm(uwF - uwD) / norm(uwD) < 1.0e-8
    # On the hanging-node tree the lumped matrix-free Jacobi diagonal differs
    # from the assembled one only on constrained rows; it must stay a sensible
    # damped-Jacobi scaling (same magnitude, strictly positive).
    asmw = solve_semquad_stokes_var(ssw, qtw, etaw, _G176_Z2, fzw, _G176_Z2,
                                    _G176_Z2; bc=:free, linear=:direct,
                                    assemble_only=true)
    mgwS = QuaST.make_semquad_mg(ssw, qtw, 2, :free, false, etaw, asmw.Kred;
                                     cycles=2, operator=:sparse)
    mgwF = QuaST.make_semquad_mg(ssw, qtw, 2, :free, false, etaw, asmw.Kred;
                                     cycles=2, operator=:matrixfree)
    @test isempty(mgwF.A)
    for k in 1:(mgwF.geo.L + 1)
        dS = mgwS.D[k]; dF = mgwF.D[k]
        @test all(>(0.0), dF)
        @test 0.5 < minimum(dF ./ dS) && maximum(dF ./ dS) < 2.0
        x = randn(mgwF.geo.nfs[k])
        yF = zeros(mgwF.geo.nfs[k]); yS = zeros(mgwF.geo.nfs[k])
        QuaST._semquad_mg_apply!(yF, x, mgwF, k)
        mul!(yS, mgwS.A[k], x)
        @test norm(yF - yS) / norm(yS) < 1.0e-12
    end
    @info "G176 nonuniform" leaves = length(qtw.leaves) sparse_vs_direct =
        norm(uwS - uwD) / norm(uwD) mf_vs_direct = norm(uwF - uwD) / norm(uwD)
end
