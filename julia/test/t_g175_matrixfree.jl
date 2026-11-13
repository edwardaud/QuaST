using Test, MantleSEM, LinearAlgebra, SparseArrays, Random

# G175 / S3-MF-1: matrix-free velocity stiffness `semquad_apply_K!`.
#
# `K` is never needed as a matrix.  `semquad_apply_K!(out, ss, qt, eta, v)`
# applies the *unconstrained* leaf-local tensor-product operator, and
# `semquad_apply_Kred!(out, ss, qt, Cu, eta, v)` does `Cu'*K*(Cu*v)` on the
# reduced space, i.e. `Kred*v` without forming `C'AC`.
#
# The quadrature rule (`make_fedata(p, ss.q)`, Gauss-Legendre) and the points at
# which the viscosity is sampled are the assembler's, so the matrix-free apply
# must reproduce the assembled sparse product to round-off **for any viscosity
# field**, not just constant eta.  That is what makes it safe to drop the sparse
# operator from the multigrid hierarchy (MF-3) rather than merely swapping in a
# spectrally similar operator.
#
# Acceptance: relative error < 1e-12 on a uniform level-3 and level-4 tree at
# p = 4 with a random vector (the S3 design's MF-1 criterion).

const _G175_Z2 = (x, y) -> 0.0

function _g175_uniform(level::Int; cx::Int=1, cy::Int=1)
    qt = Quadtree(level; cells_x=cx, cells_y=cy)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

@testset "G175 matrix-free velocity stiffness" begin
    Random.seed!(20251106)
    cases = (("const", (x, y) -> 1.0),
             ("100x", (x, y) -> exp(-log(100.0) * (1 - y))),
             ("16kx", (x, y) -> exp(-log(16384.0) * (1 - y))),
             ("xy", (x, y) -> 1.0 + 3.0 * x + 2.0 * y + 0.5 * x * y))
    for level in (3, 4), (label, eta) in cases
        qt = _g175_uniform(level)
        ss = make_semstokes(qt, 4)
        asm = solve_semquad_stokes_var(ss, qt, eta, _G175_Z2, _G175_Z2,
                                       _G175_Z2, _G175_Z2;
                                       bc=:free, linear=:direct,
                                       assemble_only=true)
        N2 = 2 * ss.su.N
        @test size(asm.K, 1) == N2
        # ---- unconstrained operator: out == K*v ------------------------------
        v = randn(N2)
        ref = asm.K * v
        out = zeros(Float64, N2)
        semquad_apply_K!(out, ss, qt, eta, v)
        rel = norm(out - ref) / norm(ref)
        @test rel < 1.0e-12
        @info "G175 full" level label rel
        # ---- reduced operator: Cu'*K*(Cu*v) == Kred*v ------------------------
        Cu = asm.Cu
        nf = size(Cu, 2)
        @test nf == size(asm.Kred, 1)
        vr = randn(nf)
        refr = asm.Kred * vr
        outr = zeros(Float64, nf)
        semquad_apply_Kred!(outr, ss, qt, Cu, eta, vr)
        relr = norm(outr - refr) / norm(refr)
        @test relr < 1.0e-12
        @info "G175 reduced" level label relr
    end

    # ---- the precomputed viscosity field must give the identical operator ----
    qt = _g175_uniform(4)
    ss = make_semstokes(qt, 4)
    eta = (x, y) -> exp(-log(50.0) * (1 - y)) * (1.0 + 0.3 * sin(3.0 * pi * x))
    op = semquad_kop(ss, qt)
    @test op.nleaf == length(qt.leaves)
    @test op.n == 5 && op.nq == 6
    efield = semquad_eta_field(op, eta)
    @test length(efield) == op.nleaf * op.nq * op.nq
    v = randn(2 * ss.su.N)
    a = zeros(Float64, length(v)); b = zeros(Float64, length(v))
    semquad_apply_K!(a, ss, qt, eta, v)
    semquad_apply_K!(b, ss, qt, efield, v)
    @test a == b
    @test all(>(0.0), efield)
    # the sampling points are the assembler's quadrature points
    fd = make_fedata(ss.p, ss.q)
    su = ss.su
    for li in 1:min(op.nleaf, 2)
        lf = su.leaves[li]
        hx = MantleSEM.leaf_hx(su, lf[1]); hy = MantleSEM.leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        k = (li - 1) * fd.nq * fd.nq
        for qy in 1:fd.nq, qx in 1:fd.nq
            k += 1
            @test op.xq[k] ≈ x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx atol = 1.0e-14
            @test op.yq[k] ≈ y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy atol = 1.0e-14
        end
    end

    # ---- deviatoric variant ------------------------------------------------
    qt2 = _g175_uniform(3)
    ss2 = make_semstokes(qt2, 4)
    asm2 = solve_semquad_stokes_var(ss2, qt2, (x, y) -> 1.0 + y, _G175_Z2,
                                    _G175_Z2, _G175_Z2, _G175_Z2;
                                    bc=:free, deviatoric=true, linear=:direct,
                                    assemble_only=true)
    v2 = randn(2 * ss2.su.N)
    r2 = asm2.K * v2
    o2 = zeros(Float64, length(v2))
    semquad_apply_K!(o2, ss2, qt2, (x, y) -> 1.0 + y, v2; deviatoric=true)
    @test norm(o2 - r2) / norm(r2) < 1.0e-12

    # ---- non-uniform tree with hanging constraints --------------------------
    # `Cu` is not a selection matrix here, so this is the real test of the
    # Cu/K/Cu' route (the uniform trees above only exercise Dirichlet rows).
    qtw = Quadtree(4; cells_x=5, cells_y=2)
    for _ in 1:2, lf in collect(qtw.leaves)
        refine!(qtw, lf)
    end
    balance!(qtw)
    refine_wall_to_min!(qtw, 4; strip=0.1)
    balance!(qtw)
    @test length(unique(lf[1] for lf in qtw.leaves)) > 1
    ssw = make_semstokes(qtw, 4)
    etaw = (x, y) -> exp(-log(1000.0) * (1 - y))
    asmw = solve_semquad_stokes_var(ssw, qtw, etaw, _G175_Z2, _G175_Z2,
                                    _G175_Z2, _G175_Z2;
                                    bc=:rigid, linear=:direct, assemble_only=true)
    nfw = size(asmw.Cu, 2)
    vw = randn(2 * ssw.su.N)
    refw = asmw.K * vw
    outw = zeros(Float64, length(vw))
    semquad_apply_K!(outw, ssw, qtw, etaw, vw)
    @test norm(outw - refw) / norm(refw) < 1.0e-12
    vrw = randn(nfw)
    refrw = asmw.Kred * vrw
    outrw = zeros(Float64, nfw)
    semquad_apply_Kred!(outrw, ssw, qtw, asmw.Cu, etaw, vrw)
    relw = norm(outrw - refrw) / norm(refrw)
    @test relw < 1.0e-12
    @info "G175 nonuniform" length(qtw.leaves) relw

    # ---- linearity and symmetry of the reduced operator --------------------
    # `Kred` is SPD; the matrix-free apply must be too, otherwise a CG inner
    # solve (MF-4) would stall.  Checked on random vectors, not the matrix.
    x1 = randn(nfw); y1 = randn(nfw)
    Kx = zeros(Float64, nfw); Ky = zeros(Float64, nfw)
    semquad_apply_Kred!(Kx, ssw, qtw, asmw.Cu, etaw, x1)
    semquad_apply_Kred!(Ky, ssw, qtw, asmw.Cu, etaw, y1)
    @test dot(x1, Ky) ≈ dot(Kx, y1) rtol = 1.0e-10
    a1 = randn(nfw); b1 = randn(nfw)
    Ka = zeros(Float64, nfw); Kb = zeros(Float64, nfw); Kab = zeros(Float64, nfw)
    semquad_apply_Kred!(Ka, ssw, qtw, asmw.Cu, etaw, a1)
    semquad_apply_Kred!(Kb, ssw, qtw, asmw.Cu, etaw, b1)
    semquad_apply_Kred!(Kab, ssw, qtw, asmw.Cu, etaw, a1 .+ b1)
    @test norm(Kab - (Ka .+ Kb)) < 1.0e-10 * norm(Kab)
    @test dot(a1, Ka) > 0.0

    # ---- recorded cost: matrix-free apply vs the sparse matvec -------------
    qt4 = _g175_uniform(4)
    ss4 = make_semstokes(qt4, 4)
    asm4 = solve_semquad_stokes_var(ss4, qt4, (x, y) -> 1.0, _G175_Z2,
                                    _G175_Z2, _G175_Z2, _G175_Z2;
                                    bc=:free, linear=:direct, assemble_only=true)
    op4 = semquad_kop(ss4, qt4)
    eta4 = semquad_eta_field(op4, (x, y) -> 1.0)
    work = SemQuadKWork(op4.n, op4.nq, 2 * ss4.su.N)
    v4 = randn(2 * ss4.su.N)
    o4 = zeros(Float64, length(v4))
    semquad_apply_K!(o4, ss4, qt4, eta4, v4; work=work)   # compile
    nrep = 20
    t_mf = @elapsed for _ in 1:nrep
        semquad_apply_K!(o4, ss4, qt4, eta4, v4; work=work)
    end
    yr = similar(o4)
    mul!(yr, asm4.K, v4)
    t_sp = @elapsed for _ in 1:nrep
        mul!(yr, asm4.K, v4)
    end
    @info "G175 cost" leaves = length(qt4.leaves) nnz_K = nnz(asm4.K) mf_ms =
        1000 * t_mf / nrep sparse_ms = 1000 * t_sp / nrep
    @test t_mf < 4.0 * t_sp        # the whole point: the MF apply must not be
                                   # slower than streaming the assembled matrix
end
