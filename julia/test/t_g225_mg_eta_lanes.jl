using Test, MantleSEM, LinearAlgebra, SparseArrays, Random

# G225: the eta lane pre-gather is a pure re-layout and is bitwise.
#
# The matrix-free apply used to read its viscosity one lane at a time,
# `etav[kofs[b] + kq]`, which LLVM compiles to a `vgather` per quadrature point
# (48 per 32-lane block at p = 2: 4 weighted-gradient loops x 4x4 Gauss points,
# docs/dev/MG.md).  §15 pre-gathers the field into a **lane-major
# table** built once per operator construction (`SemQuadEtaLane`, per level in
# the MG hierarchy), so the kernel reads a contiguous lane vector instead:
#
#     lane[(bi-1)*nq2*BLK + (kq-1)*BLK + b]  =  etav[((bi-1)*BLK + b - 1)*nq2 + kq]
#
# with the last block padded with its final leaf -- exactly the leaf the
# kernel's padded lanes already read (weight 0.0).  `SEMQUAD_MG_ETA_LANES`
# selects the table (default) or the historical gather, and the threaded chunk
# bounds are block-aligned (`SemQuadKPar`) so a chunk window is a whole number of
# lane blocks.
#
# Nothing here may move a bit.  The gate pins:
#
#   (a) the layout: every lane-table entry is bitwise the leaf-major value it
#       was built from, the padding repeats the last leaf, the table is the
#       documented size, and the builder is idempotent on an existing table;
#   (b) per level: apply with the table == apply with the gather == the
#       unchanged `_semquad_apply_Kred!` given a plain `Vector` field, on a tree
#       whose leaf count is **not** a multiple of `SEMQUAD_KBLK` (so the last
#       block is padded);
#   (c) per level at 2 workers: the same three-way equality through the threaded
#       element-buffer kernel, which is the path the aligned chunk bounds exist
#       for; every chunk boundary is on the block grid and the partition still
#       covers `1:nleaf`;
#   (d) per V-cycle: table == gather == the 1-worker reference, at a mixed
#       serial threshold and at `thr = 0` (every level threaded);
#   (e) end to end: a `linear=:mg` solve gives the same `u`/`p` bit patterns,
#       the same `nit` and the same true residual under both settings.
#
# In a 1-thread process the 2-worker `SemQuadKPar`/region path is co-scheduled
# on the calling thread (the G190/G222/G223/G224 construction); with
# `--threads=8` the same gate also runs the 8-worker form.

const _G225_Z2 = (x, y) -> 0.0

"""A hanging-node tree refined until the finest leaf count is **not** a multiple
of `SEMQUAD_KBLK`, so the last lane block is padded (G180's non-dividing case,
now on the eta table)."""
function _g225_tree()
    qt = Quadtree(4)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    for lf in sort(collect(qt.leaves))
        _, _, y0 = MantleSEM.leaf_box(qt, lf)
        y0 < 0.26 && refine!(qt, lf)
    end
    balance!(qt)
    lf = sort(collect(qt.leaves))[1]
    refine!(qt, lf)
    balance!(qt)
    return qt
end

"""`mg` rebuilt with `nw` workers (the G190/G222/G223/G224 construction); the
geometry, including the cached `Cu` transposes, is shared."""
function _g225_workers(mg::MantleSEM.SemQuadMG, nw::Int)
    nlev = mg.geo.L + 1
    par = Any[MantleSEM.SemQuadKPar(mg.mfops[k], nw) for k in 1:nlev]
    Pts = Any[SparseMatrixCSC{Float64,Int}(sparse(transpose(mg.geo.Ps[k])))
              for k in 1:mg.geo.L]
    return MantleSEM.SemQuadMG(mg.geo, mg.A, mg.D, mg.cA, mg.cycles, mg.pre,
                               mg.post, mg.omega, mg.smoother, mg.LF, mg.UF,
                               mg.operator, mg.mfops, mg.mfetas, mg.mfwork,
                               mg.sc, mg.sr, mg.sy, nw, par, Pts,
                               mg.rcs, mg.ecs, mg.pys)
end

"""Bit-pattern equality of two `Float64` vectors: `==` would accept a `-0.0`
for a `+0.0`, a real change in the bytes."""
function _g225_bitsame(a::AbstractVector{Float64}, b::AbstractVector{Float64})
    length(a) == length(b) || return false
    return reinterpret(UInt64, collect(a)) == reinterpret(UInt64, collect(b))
end

"""Number of lane-table entries that are not bitwise the leaf-major value they
must hold (padding included), re-derived here independently of the builder."""
function _g225_lane_errors(e::MantleSEM.SemQuadEtaLane,
                           op::MantleSEM.SemQuadKOp)
    BLK = MantleSEM.SEMQUAD_KBLK
    nl = op.nleaf
    nq2 = op.nq * op.nq
    bad = 0
    @inbounds for bi in 1:cld(nl, BLK)
        lo = (bi - 1) * BLK + 1
        hi = min(nl, bi * BLK)
        for kq in 1:nq2, b in 1:BLK
            li = lo + min(b - 1, hi - lo)      # the padding repeats the last leaf
            a = e.lane[(bi - 1) * nq2 * BLK + (kq - 1) * BLK + b]
            c = e.scalar[(li - 1) * nq2 + kq]
            bad += reinterpret(UInt64, a) == reinterpret(UInt64, c) ? 0 : 1
        end
    end
    return bad
end

"""Number of padded lanes of the last block whose value is not the last leaf's
own eta (1 per mismatch)."""
function _g225_pad_errors(e::MantleSEM.SemQuadEtaLane, op::MantleSEM.SemQuadKOp)
    BLK = MantleSEM.SEMQUAD_KBLK
    nl = op.nleaf
    nq2 = op.nq * op.nq
    nb = nl - ((cld(nl, BLK) - 1) * BLK)       # leaves in the last block
    nb == BLK && return 0                      # nothing to pad
    bi = cld(nl, BLK)
    lo = (bi - 1) * BLK + 1
    bad = 0
    @inbounds for kq in 1:nq2, b in (nb + 1):BLK
        a = e.lane[(bi - 1) * nq2 * BLK + (kq - 1) * BLK + b]
        c = e.scalar[((lo + nb - 1) - 1) * nq2 + kq]     # the last real leaf
        bad += reinterpret(UInt64, a) == reinterpret(UInt64, c) ? 0 : 1
    end
    return bad
end

@testset "G225 the eta lane table is bitwise" begin
    Random.seed!(20261112)
    qt = _g225_tree()
    p = 2
    ss = make_semstokes(qt, p)
    eta = (x, y) -> exp(-log(1.0e4) * (1.0 - y))     # steep, non-constant
    mg = MantleSEM.make_semquad_mg(ss, qt, p, :free, false, eta, nothing;
                                   cycles=1, smoother=:jacobi,
                                   operator=:matrixfree,
                                   coarse_operator=:rediscretize,
                                   coarsen_factor=1.5, smoothing=:auto,
                                   mg_threads=1)
    nlev = mg.geo.L + 1
    nfs = mg.geo.nfs
    BLK = MantleSEM.SEMQUAD_KBLK
    @info "G225" leaves = length(qt.leaves) levels = nlev dofs = nfs threads = Threads.nthreads()
    @test nlev >= 3
    @test any(mg.mfops[k].nleaf % BLK != 0 for k in 1:nlev)   # padding is covered

    # ---- (a) the layout ------------------------------------------------
    for k in 1:nlev
        op = mg.mfops[k]::MantleSEM.SemQuadKOp
        e = mg.mfetas[k]
        @test e isa MantleSEM.SemQuadEtaLane
        nq2 = op.nq * op.nq
        nblk = cld(op.nleaf, BLK)
        @test length(e) == op.nleaf * nq2                 # the AbstractVector size
        @test size(e) == (op.nleaf * nq2,)
        @test length(e.scalar) == op.nleaf * nq2
        @test length(e.lane) == nblk * nq2 * BLK
        @test e.nleaf == op.nleaf && e.nq2 == nq2
        # every table entry is bitwise its leaf-major value, padding included
        @test _g225_lane_errors(e, op) == 0
        @test _g225_pad_errors(e, op) == 0
        # the leaf-major view is the sampled field, bit for bit
        @test _g225_bitsame(e, MantleSEM.semquad_eta_field(op, eta))
        @test _g225_bitsame(e, e.scalar) && e === MantleSEM.semquad_eta_lanes(op, e)
        # rebuilding from the flat field gives the same table
        e2 = MantleSEM.semquad_eta_lanes(op, MantleSEM.semquad_eta_field(op, eta))
        @test _g225_bitsame(e.lane, e2.lane)
        # a mismatched operator is refused rather than silently mis-indexed
        if k < nlev && mg.mfops[k + 1].nleaf != e.nleaf
            @test_throws ArgumentError MantleSEM.semquad_eta_lanes(mg.mfops[k + 1], e)
        end
    end

    # ---- (b) per level: table == gather == the plain-Vector path --------
    oldlanes = MantleSEM.SEMQUAD_MG_ETA_LANES[]
    try
        for k in 1:nlev
            op = mg.mfops[k]::MantleSEM.SemQuadKOp
            work = mg.mfwork[k]::MantleSEM.SemQuadKWork
            x = randn(nfs[k])
            MantleSEM.SEMQUAD_MG_ETA_LANES[] = true
            yl = zeros(Float64, nfs[k])
            MantleSEM._semquad_mg_apply!(yl, x, mg, k)
            MantleSEM.SEMQUAD_MG_ETA_LANES[] = false
            yg = zeros(Float64, nfs[k])
            MantleSEM._semquad_mg_apply!(yg, x, mg, k)
            # the unchanged code path: a plain leaf-major Vector, gather kernel
            yp = zeros(Float64, nfs[k])
            MantleSEM._semquad_apply_Kred!(yp, op, mg.geo.Cus[k],
                                           MantleSEM.semquad_eta_field(op, eta),
                                           x, work)
            @test _g225_bitsame(yl, yg)
            @test _g225_bitsame(yg, yp)
            # and the raw kernel alone, on the same full-space vector
            full = zeros(2 * op.N); mul!(full, mg.geo.Cus[k], x)
            o1 = zeros(2 * op.N); o2 = zeros(2 * op.N)
            MantleSEM.SEMQUAD_MG_ETA_LANES[] = true
            MantleSEM._semquad_apply_K_raw!(o1, op, mg.mfetas[k], full, work)
            MantleSEM.SEMQUAD_MG_ETA_LANES[] = false
            MantleSEM._semquad_apply_K_raw!(o2, op, mg.mfetas[k], full, work)
            @test _g225_bitsame(o1, o2)
        end
        MantleSEM.SEMQUAD_MG_ETA_LANES[] = oldlanes
    finally
        MantleSEM.SEMQUAD_MG_ETA_LANES[] = oldlanes
    end

    # ---- (c) chunk bounds on the block grid, and the 2-worker apply -----
    for k in 1:nlev
        op = mg.mfops[k]::MantleSEM.SemQuadKOp
        for nw in (2, 4,max(2, min(8, Threads.nthreads())))
            par = MantleSEM.SemQuadKPar(op, nw)
            @test par.bounds[1] == 1
            @test par.bounds[end] == op.nleaf + 1
            @test issorted(par.bounds)
            @test all((par.bounds[c] - 1) % BLK == 0
                      for c in 1:(par.nchunks + 1) if par.bounds[c] <= op.nleaf)
        end
    end
    mgw = _g225_workers(mg, 2)
    oldthr = MantleSEM.SEMQUAD_MG_SERIAL_DOFS[]
    try
        MantleSEM.SEMQUAD_MG_SERIAL_DOFS[] = 0        # thread every level
        for k in 1:nlev
            x = randn(nfs[k])
            MantleSEM.SEMQUAD_MG_ETA_LANES[] = true
            yr = zeros(Float64, nfs[k])
            MantleSEM._semquad_mg_apply!(yr, x, mgw, k)
            MantleSEM.SEMQUAD_MG_ETA_LANES[] = false
            ys = zeros(Float64, nfs[k])
            MantleSEM._semquad_mg_apply!(ys, x, mgw, k)
            @test _g225_bitsame(yr, ys)
            # ... and both are bitwise the 1-worker serial apply
            y1 = zeros(Float64, nfs[k])
            MantleSEM._semquad_mg_apply!(y1, x, mg, k)
            @test _g225_bitsame(yr, y1)
        end

        # ---- (d) the V-cycle -------------------------------------------
        b = randn(nfs[end])
        MantleSEM.SEMQUAD_MG_ETA_LANES[] = true
        zref = MantleSEM.semquad_mg_apply(mg, b)      # 1 worker, table on
        for mgx in (mg, mgw)
            for thr in (nfs[end - 1], 0)
                MantleSEM.SEMQUAD_MG_SERIAL_DOFS[] = thr
                for lanes in (true, false)
                    MantleSEM.SEMQUAD_MG_ETA_LANES[] = lanes
                    @test _g225_bitsame(MantleSEM.semquad_mg_apply(mgx, b), zref)
                end
            end
        end
        MantleSEM.SEMQUAD_MG_SERIAL_DOFS[] = oldthr
        MantleSEM.SEMQUAD_MG_ETA_LANES[] = true
    finally
        MantleSEM.SEMQUAD_MG_SERIAL_DOFS[] = oldthr
        MantleSEM.SEMQUAD_MG_ETA_LANES[] = oldlanes
    end

    # ---- (e) end to end: u, p, nit and the true residual ---------------
    kw = (bc=:free, deviatoric=false, mass_form=:boussinesq, linear=:mg,
          mg_operator=:matrixfree, mg_vel_solver=:vcycle, mg_tol=1.0e-6,
          mg_threads=1)
    fz = (x, y) -> 1.0e5 * (y - 0.5)
    try
        MantleSEM.SEMQUAD_MG_ETA_LANES[] = true
        u1, p1, _, _, _ = solve_semquad_stokes_var(ss, qt, eta, _G225_Z2, fz,
                                                   _G225_Z2, _G225_Z2; kw...)
        n1 = LAST_SEMSTOKES_SCHUR_ITERS[]
        r1 = LAST_SEMSTOKES_RESIDUAL[]
        MantleSEM.SEMQUAD_MG_ETA_LANES[] = false
        u2, p2, _, _, _ = solve_semquad_stokes_var(ss, qt, eta, _G225_Z2, fz,
                                                   _G225_Z2, _G225_Z2; kw...)
        n2 = LAST_SEMSTOKES_SCHUR_ITERS[]
        r2 = LAST_SEMSTOKES_RESIDUAL[]
        @info "G225 solve" nit = n1 relres = r1
        @test _g225_bitsame(u1, u2)
        @test _g225_bitsame(p1, p2)
        @test n1 == n2
        @test reinterpret(UInt64, r1) == reinterpret(UInt64, r2)
    finally
        MantleSEM.SEMQUAD_MG_ETA_LANES[] = oldlanes
    end
end
