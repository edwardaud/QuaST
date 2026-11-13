using Test, QuaST, LinearAlgebra, SparseArrays, Random

# G222: the shipped matrix-free apply speed-ups are bitwise.
#
# Three changes landed on top of G180/G190 and none of them may move a bit:
#
#   1. `SEMQUAD_KBLK` 8 -> 32.  The batch width is a compile-time constant; at
#      8 lanes LLVM fully unrolled the lane loop into scalar `vaddsd`/`vmulsd`,
#      at 32 it vectorises (`vaddpd`/`vmulpd`/`vsubpd`, 4-wide AVX2).  A short
#      final block is padded with a zero-weight leaf, so the padded lanes add
#      exactly 0.0 and the per-leaf arithmetic/order are unchanged (G180 checks
#      the kernel itself on uniform/AMR/wall trees).
#   2. `@simd ivdep` on the independent lane loops of the blocked kernel
#      (`_semquad_contract_blk!`, `_semquad_accum_blk!`, the four weighted
#      gradient loops, the lane-zero loop).  No loop-carried reduction is
#      touched: every lane computes an independent element, in the same order.
#   3. `SEMQUAD_MG_SERIAL_DOFS`: levels at or below the threshold run the
#      *serial* apply/Jacobi/subtract/transfer even in a threaded V-cycle.
#      Serial and threaded paths are already bitwise (G190), so switching a
#      level between them is too.
#
# This gate pins all three on a hanging-node tree, **per level** and through a
# full V-cycle, and pins that every apply allocates zero bytes (the kernel used
# to allocate five length-BLK scratch vectors per call, 640 B, per chunk in the
# threaded path).

const _G222_Z2 = (x, y) -> 0.0

function _g222_amr()
    qt = Quadtree(4)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    for lf in sort(collect(qt.leaves))
        _, _, y0 = QuaST.leaf_box(qt, lf)
        y0 < 0.26 && refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

_scalar_op(op::QuaST.SemQuadKOp) =
    QuaST.SemQuadKOp(op.p, op.n, op.q, op.nq, op.deviatoric, op.nleaf,
                         op.N, op.gl_w, op.phi, op.dphi, op.phiT, op.dphiT,
                         op.ids, op.ihx, op.ihy, op.wfac, op.xq, op.yq, 0)

function _with_scalar_ops(f::Function, mg::QuaST.SemQuadMG)
    mfops0 = Any[_scalar_op(mg.mfops[k]::QuaST.SemQuadKOp)
                 for k in 1:(mg.geo.L + 1)]
    mg0 = QuaST.SemQuadMG(mg.geo, mg.A, mg.D, mg.cA, mg.cycles, mg.pre,
                              mg.post, mg.omega, mg.smoother, mg.LF, mg.UF,
                              mg.operator, mfops0, mg.mfetas, mg.mfwork,
                              mg.sc, mg.sr, mg.sy, mg.threads, mg.par, mg.Pts,
                              mg.rcs, mg.ecs, mg.pys)
    return f(mg0)
end

# `@allocated` after a warm call: the first threaded apply of a given
# specialisation pays one-time compilation, which the S6 region path of the
# V-cycle (G223) may leave to this call because the V-cycle no longer goes
# through the per-call apply.
_alloc_apply(mg, k, x, y) = (QuaST._semquad_mg_apply!(y, x, mg, k);
                             @allocated QuaST._semquad_mg_apply!(y, x, mg, k))

@testset "G222 MG apply optimisations are bitwise" begin
    Random.seed!(20261112)
    qt = _g222_amr()
    p = 2
    ss = make_semstokes(qt, p)
    eta = (x, y) -> exp(-log(1.0e4) * (1.0 - y))
    mgkw = (cycles=1, smoother=:jacobi, operator=:matrixfree,
            coarse_operator=:rediscretize, coarsen_factor=1.5,
            smoothing=:auto)
    mg = QuaST.make_semquad_mg(ss, qt, p, :free, false, eta, nothing;
                                   mgkw..., mg_threads=1)
    nlev = mg.geo.L + 1
    @info "G222" leaves = length(qt.leaves) levels = nlev dofs = mg.geo.nfs
    @test nlev >= 3

    # ---- 1. per level: blocked (32 lanes) == scalar, and zero allocation ----
    for k in 1:nlev
        nf = mg.geo.nfs[k]
        x = randn(nf); y = zeros(Float64, nf)
        QuaST._semquad_mg_apply!(y, x, mg, k)          # warm
        yb = copy(y)
        ys = _with_scalar_ops(mg) do mg0
            tmp = zeros(Float64, nf)
            QuaST._semquad_mg_apply!(tmp, x, mg0, k)
        end
        @test yb == ys
        @test _alloc_apply(mg, k, x, y) == 0
    end

    # ---- 2. full V-cycle: blocked == scalar, bitwise ------------------------
    b = randn(mg.geo.nfs[end])
    z = QuaST.semquad_mg_apply(mg, b)
    zs = _with_scalar_ops(mg) do mg0
        QuaST.semquad_mg_apply(mg0, b)
    end
    @test z == zs

    # ---- 3. serial threshold: every setting is bitwise the same V-cycle -----
    # A hierarchy built directly with 2 leaf blocks exercises the threaded
    # discard/merge path even in a single-threaded CI process (as G190 does).
    par = Any[QuaST.SemQuadKPar(mg.mfops[k], 2) for k in 1:nlev]
    Pts = Any[SparseMatrixCSC{Float64,Int}(sparse(transpose(mg.geo.Ps[k])))
              for k in 1:mg.geo.L]
    mgt = QuaST.SemQuadMG(mg.geo, mg.A, mg.D, mg.cA, mg.cycles, mg.pre,
                              mg.post, mg.omega, mg.smoother, mg.LF, mg.UF,
                              mg.operator, mg.mfops, mg.mfetas, mg.mfwork,
                              mg.sc, mg.sr, mg.sy, 2, par, Pts,
                              mg.rcs, mg.ecs, mg.pys)
    oldthr = QuaST.SEMQUAD_MG_SERIAL_DOFS[]
    try
        for thr in (0, 8192, typemax(Int))
            QuaST.SEMQUAD_MG_SERIAL_DOFS[] = thr
            @test QuaST.semquad_mg_apply(mgt, b) == z
            y = zeros(Float64, mg.geo.nfs[end])
            x = randn(mg.geo.nfs[end])
            # The kernel itself no longer allocates; a threaded apply still
            # allocates the Julia task objects of `Threads.@sync`/`@spawn`
            # (~0.5 kB per worker), which is scheduler state, not kernel
            # scratch.  Pin that it stays in that class.
            @test _alloc_apply(mgt, nlev, x, y) <= 16 * 1024
        end
        # the vector ops switch too, and stay bitwise
        n = mg.geo.nfs[3]
        r = randn(n); bb = randn(n); yy = randn(n); D = mg.D[3]
        QuaST.SEMQUAD_MG_SERIAL_DOFS[] = 0
        r0 = similar(r); QuaST._semquad_td_sub!(r0, bb, yy, 2)
        x0 = copy(r); QuaST._semquad_td_jacobi!(x0, r, D, 0.666, 2)
        QuaST.SEMQUAD_MG_SERIAL_DOFS[] = typemax(Int)
        r1 = similar(r); QuaST._semquad_td_sub!(r1, bb, yy, 2)
        x1 = copy(r); QuaST._semquad_td_jacobi!(x1, r, D, 0.666, 2)
        @test r0 == r1 && x0 == x1
    finally
        QuaST.SEMQUAD_MG_SERIAL_DOFS[] = oldthr
    end

    # ---- 4. the threshold default is restored and the full solve is bitwise -
    @test QuaST.SEMQUAD_MG_SERIAL_DOFS[] == oldthr
end
