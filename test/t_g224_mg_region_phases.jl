using Test, QuaST, LinearAlgebra, SparseArrays, Random

# G224: the S7 region phases are bitwise the serial path and add no fork/joins.
#
# Two costs were left serial inside the S6 region of a threaded level
# (docs/dev/MG.md):
#
#   1. the restriction into the next coarser level, which stays serial when its
#      *output* is below `SEMQUAD_MG_SERIAL_DOFS` (`SEMQUAD_MG_REGION_TRANSFER`
#      publishes it as a region phase instead);
#   2. the two sparse `Cu` products of the level's matrix-free apply
#      (`SEMQUAD_MG_REGION_CU` splits `Cu * v` row-wise through the cached `Cu`
#      transpose and `Cu' * y` column-wise, both as region phases).
#
# The gate pins:
#
#   (a) per kernel: `_semquad_spmv_region!` == `mul!(_, Cu, _)` and
#       `_semquad_spmv_at_region!` == `mul!(_, Cu', _)` on every level, for the
#       process' workers and for 2 workers (a co-scheduled worker in a 1-thread
#       process);
#   (b) per apply: the region apply is bitwise the serial `_semquad_apply_Kred!`
#       with the split `Cu` on, with it off, and on every level;
#   (c) per transfer: a restriction/prolongation inside a region is bitwise the
#       serial one at a **mixed** threshold (`SEMQUAD_MG_SERIAL_DOFS =
#       nfs[end-1]`, so only the finest level opens a region and its
#       restriction output is below the threshold -- exactly the §13 case) and
#       at `thr = 0`;
#   (d) per V-cycle: region path == per-call path == the 1-worker serial
#       hierarchy for every (transfer, cu) setting, at the mixed threshold and
#       at `thr = 0`; the fork count stays exactly 2 per threaded level (the
#       phases are barriers, not task sets) and the phases account for the
#       switched-on costs;
#   (e) `mg_threads = 1` stays inert: no fork, no phase, same V-cycle.
#
# In a 1-thread process the 2-worker region is co-scheduled on the calling
# thread (the G190/G222/G223 construction); with `--threads=8` the same gate
# also runs 8 workers.

const _G224_Z2 = (x, y) -> 0.0

function _g224_tree()
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

"""`mg` rebuilt with `nw` workers (the G190/G222/G223 construction); the
geometry, including the cached `Cu` transposes, is shared."""
function _g224_workers(mg::QuaST.SemQuadMG, nw::Int)
    nlev = mg.geo.L + 1
    par = Any[QuaST.SemQuadKPar(mg.mfops[k], nw) for k in 1:nlev]
    Pts = Any[SparseMatrixCSC{Float64,Int}(sparse(transpose(mg.geo.Ps[k])))
              for k in 1:mg.geo.L]
    return QuaST.SemQuadMG(mg.geo, mg.A, mg.D, mg.cA, mg.cycles, mg.pre,
                               mg.post, mg.omega, mg.smoother, mg.LF, mg.UF,
                               mg.operator, mg.mfops, mg.mfetas, mg.mfwork,
                               mg.sc, mg.sr, mg.sy, nw, par, Pts,
                               mg.rcs, mg.ecs, mg.pys)
end

"""Forks, worker tasks and phases of one call of `f`."""
function _g224_forks(f)
    f0 = QuaST.SEMQUAD_MG_FORKS[]
    t0 = QuaST.SEMQUAD_MG_FORK_TASKS[]
    p0 = QuaST.SEMQUAD_MG_PHASES[]
    f()
    return (QuaST.SEMQUAD_MG_FORKS[] - f0,
            QuaST.SEMQUAD_MG_FORK_TASKS[] - t0,
            QuaST.SEMQUAD_MG_PHASES[] - p0)
end

"""Bit-pattern equality of two `Float64` vectors: `==` would accept a `-0.0`
for a `+0.0`, a real change in the bytes."""
function _g224_bitsame(a::Vector{Float64}, b::Vector{Float64})
    length(a) == length(b) || return false
    return reinterpret(UInt64, a) == reinterpret(UInt64, b)
end

"""Expected region phases of one V-cycle with one threaded level block: the
`pre`/`post` Jacobi sweeps, the residual applies, the transfers and the split
`Cu` phases of `pre + post + 1` applies on the threaded level.  The
prolongation is always a phase (its output is the threaded level itself); the
restriction is a phase when the transfer switch is on *or* when its output is
above the threshold (the pre-S6 rule still applies above the threshold)."""
function _g224_phases(mg::QuaST.SemQuadMG, lev::Int, transfer::Bool,
                      cu::Bool, thr::Int)
    napp = mg.pre[lev] + mg.post[lev] + 1
    # pre sweeps (apply+sub+jac), the restriction residual (apply+sub), the
    # prolongation, post sweeps (apply+sub+jac)
    base = 3 * (mg.pre[lev] + mg.post[lev]) + 3
    restr = (transfer || mg.geo.nfs[lev - 1] > thr) ? 1 : 0
    return base + restr + (cu ? 2 * napp : 0)
end

@testset "G224 region transfer and split Cu are bitwise" begin
    Random.seed!(20261113)
    qt = _g224_tree()
    p = 2
    ss = make_semstokes(qt, p)
    eta = (x, y) -> exp(-log(1.0e4) * (1.0 - y))
    mg = QuaST.make_semquad_mg(ss, qt, p, :free, false, eta, nothing;
                                   cycles=1, smoother=:jacobi,
                                   operator=:matrixfree,
                                   coarse_operator=:rediscretize,
                                   coarsen_factor=1.5, smoothing=:auto,
                                   mg_threads=1)
    nlev = mg.geo.L + 1
    nfs = mg.geo.nfs
    @info "G224" leaves = length(qt.leaves) levels = nlev dofs = nfs threads = Threads.nthreads()
    @test nlev >= 4
    @test nfs[end - 1] < nfs[end]

    b = randn(nfs[end])
    zref = QuaST.semquad_mg_apply(mg, b)          # serial, 1 worker
    oldthr = QuaST.SEMQUAD_MG_SERIAL_DOFS[]
    oldreg = QuaST.SEMQUAD_MG_REGION[]
    oldtr = QuaST.SEMQUAD_MG_REGION_TRANSFER[]
    oldcu = QuaST.SEMQUAD_MG_REGION_CU[]
    try
        for nw in unique([2, max(2, min(8, Threads.nthreads()))])
            mgw = _g224_workers(mg, nw)
            @test mgw.threads == nw
            @test length(mgw.geo.Cuts) == nlev
            # the cached transpose is exactly the re-indexing of Cu
            for k in 1:nlev
                @test size(mgw.geo.Cuts[k]) == reverse(size(mgw.geo.Cus[k]))
                @test nnz(mgw.geo.Cuts[k]) == nnz(mgw.geo.Cus[k])
            end

            # ---- (a) the two sparse products, as kernels and as phases -----
            for k in 1:nlev
                op = mgw.mfops[k]::QuaST.SemQuadKOp
                Cu = mgw.geo.Cus[k]
                Cut = mgw.geo.Cuts[k]
                par = mgw.par[k]::QuaST.SemQuadKPar
                nwlev = min(nw, par.nt)
                region = QuaST.SemQuadParRegion(nwlev)
                x = randn(nfs[k])
                full0 = zeros(2 * op.N); full1 = zeros(2 * op.N)
                mul!(full0, Cu, x)
                QuaST._semquad_region_run!(region) do
                    QuaST._semquad_spmv_region!(full1, Cut, x, par.nt,
                                                    region)
                end
                @test _g224_bitsame(full1, full0)
                y0 = zeros(nfs[k]); y1 = zeros(nfs[k])
                src = randn(2 * op.N)
                mul!(y0, transpose(Cu), src)
                QuaST._semquad_region_run!(region) do
                    QuaST._semquad_spmv_at_region!(y1, Cu, src, par.nt,
                                                       region)
                end
                @test _g224_bitsame(y1, y0)
                # and the whole apply, both settings, against the serial one
                yref = zeros(nfs[k]); y = zeros(nfs[k])
                QuaST._semquad_apply_Kred!(yref, op, Cu,
                                               mgw.mfetas[k]::AbstractVector{Float64}, x,
                                               mgw.mfwork[k]::QuaST.SemQuadKWork)
                for cu in (false, true)
                    QuaST.SEMQUAD_MG_REGION_CU[] = cu
                    fill!(y, NaN)
                    QuaST._semquad_region_run!(region) do
                        QuaST._semquad_mg_apply!(y, x, mgw, k, region)
                    end
                    @test _g224_bitsame(y, yref)
                end
                QuaST.SEMQUAD_MG_REGION_CU[] = true
            end

            # ---- (b) transfers inside a region, every level ---------------
            for k in 2:nlev
                par = mgw.par[k]::QuaST.SemQuadKPar
                region = QuaST.SemQuadParRegion(min(nw, par.nt))
                rc0 = zeros(nfs[k - 1]); rc1 = zeros(nfs[k - 1])
                r = randn(nfs[k])
                QuaST._semquad_mg_restrict!(rc0, mgw.geo.Ps[k - 1], r, 1)
                for tr in (false, true)
                    QuaST.SEMQUAD_MG_REGION_TRANSFER[] = tr
                    QuaST._semquad_region_run!(region) do
                        QuaST._semquad_mg_restrict!(rc1, mgw.geo.Ps[k - 1],
                                                        r, nw, region)
                    end
                    @test _g224_bitsame(rc1, rc0)
                end
                QuaST.SEMQUAD_MG_REGION_TRANSFER[] = true
                Pt = mgw.Pts[k - 1]::SparseMatrixCSC{Float64,Int}
                y0 = zeros(nfs[k]); y1 = zeros(nfs[k])
                e = randn(nfs[k - 1])
                QuaST._semquad_mg_prolong!(y0, Pt, e, 1)
                for tr in (false, true)
                    QuaST.SEMQUAD_MG_REGION_TRANSFER[] = tr
                    QuaST._semquad_region_run!(region) do
                        QuaST._semquad_mg_prolong!(y1, Pt, e, nw, region)
                    end
                    @test _g224_bitsame(y1, y0)
                end
                QuaST.SEMQUAD_MG_REGION_TRANSFER[] = true
            end

            # ---- (c)+(d) V-cycles: mixed threshold, then every level ------
            mixed = nfs[end - 1]
            nthreaded = count(t -> t >= 2 && nfs[t] > mixed, 1:nlev)
            @test nthreaded == 1
            for thr in (mixed, 0)
                QuaST.SEMQUAD_MG_SERIAL_DOFS[] = thr
                nthreaded = count(t -> t >= 2 && nfs[t] > thr, 1:nlev)
                expected_forks = 2 * nthreaded
                for (tr, cu) in ((false, false), (true, false), (false, true),
                                 (true, true))
                    QuaST.SEMQUAD_MG_REGION[] = false
                    z0 = QuaST.semquad_mg_apply(mgw, b)
                    f0, _, ph0 = _g224_forks(() -> QuaST.semquad_mg_apply(mgw, b))
                    QuaST.SEMQUAD_MG_REGION[] = true
                    QuaST.SEMQUAD_MG_REGION_TRANSFER[] = tr
                    QuaST.SEMQUAD_MG_REGION_CU[] = cu
                    z1 = QuaST.semquad_mg_apply(mgw, b)
                    f1, _, ph1 = _g224_forks(() -> QuaST.semquad_mg_apply(mgw, b))
                    @info "G224 V-cycle" nw = nw thr = thr transfer = tr cu = cu percall_forks = f0 region_forks = f1 phases = ph1
                    @test _g224_bitsame(z0, z1)
                    @test _g224_bitsame(z1, zref)
                    @test ph0 == 0
                    @test f1 == expected_forks
                    if thr == mixed
                        # exactly the finest level is threaded: its restriction
                        # output is the next level, below the threshold
                        @test f0 > f1
                        want = _g224_phases(mg, nlev, tr, cu, thr)
                        @test ph1 == want
                    else
                        want = sum(_g224_phases(mg, t, tr, cu, thr)
                                   for t in 2:nlev if nfs[t] > thr)
                        @test ph1 == want
                        @test f0 > 2 * expected_forks
                    end
                end
            end
            QuaST.SEMQUAD_MG_SERIAL_DOFS[] = nfs[end - 1]
            QuaST.SEMQUAD_MG_REGION_TRANSFER[] = true
            QuaST.SEMQUAD_MG_REGION_CU[] = true
        end

        # ---- (e) one thread: the region is never opened -------------------
        QuaST.SEMQUAD_MG_SERIAL_DOFS[] = 0     # would thread every level
        QuaST.SEMQUAD_MG_REGION[] = true
        QuaST.SEMQUAD_MG_REGION_TRANSFER[] = true
        QuaST.SEMQUAD_MG_REGION_CU[] = true
        @test mg.threads == 1
        z = QuaST.semquad_mg_apply(mg, b)
        f, _, ph = _g224_forks(() -> QuaST.semquad_mg_apply(mg, b))
        @test _g224_bitsame(z, zref)
        @test f == 0
        @test ph == 0
    finally
        QuaST.SEMQUAD_MG_SERIAL_DOFS[] = oldthr
        QuaST.SEMQUAD_MG_REGION[] = oldreg
        QuaST.SEMQUAD_MG_REGION_TRANSFER[] = oldtr
        QuaST.SEMQUAD_MG_REGION_CU[] = oldcu
    end
end
