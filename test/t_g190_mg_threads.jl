# G190: the threaded V-cycle is bitwise the serial one.
using Test, QuaST, LinearAlgebra, Random

# G190 / S5-2: the threaded (leaf-blocked) matrix-free V-cycle is *bitwise* the
# serial one.
#
# `mg_threads=N>1` cuts the leaf list of every matrix-free level into N fixed
# contiguous blocks and threads the leaf-local work: the `semquad_apply_K!`
# kernel (the dominant cost), the damped-Jacobi sweeps and the residual
# evaluation.  Each block owns its own `SemQuadKScratch`; the kernel writes each
# leaf's *element* vector into a per-leaf column (`SemQuadKPar.elx/elz`) and the
# scatter-add into the node vector is then done **serially, in leaf order**,
# which is exactly the accumulation order of the serial kernel.  A merge of
# per-block *node* buffers in block order -- the `assemble_threads` discipline
# of §12 -- would only be deterministic here: the float association of a node
# shared by leaves in different blocks changes with the block count.
#
# The gate checks
#   1. the element-buffer kernel reproduces the serial blocked kernel bitwise,
#      on a uniform AND on a hanging-node (AMR) tree, for nt = 2, 4, 8, and it
#      is bitwise repeatable;
#   2. `_semquad_apply_Kred_threaded!` is bitwise `_semquad_apply_Kred!`;
#   3. `semquad_mg_apply` at mg_threads = 2/4/8 is bitwise the serial V-cycle
#      (parallel *and* through a directly built hierarchy with nt blocks);
#   4. the whole `solve_semquad_stokes_var(...; linear=:mg)` is bitwise equal
#      across thread counts, with the same outer iteration count and residual
#      (this covers the threaded outer operator: FGMRES residual + BFBT `S2`);
#   5. `mg_threads=1` (the default) is the serial path bitwise, and the sparse
#      operator path ignores `mg_threads` (bitwise too);
#   6. the plumbing: `mg_threads=0` throws, a request above `Threads.nthreads()`
#      clamps with a warning, a single-threaded process warns once and stays
#      serial.
# The block machinery is exercised directly (like G181 does for the assembly),
# so the gate is meaningful in a single-threaded CI process as well.

const _G190_Z2 = (x, y) -> 0.0
const _G190_ONE = (x, y) -> 1.0

function _g190_uniform(level::Int)
    qt = Quadtree(level)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

# A graded tree with hanging nodes: uniform L2 then refine the wall strip twice.
function _g190_amr()
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

@testset "G190 threaded V-cycle is bitwise" begin
    Random.seed!(20261107)

    for (case, qt) in (("uniform L4", _g190_uniform(4)), ("AMR", _g190_amr()))
        ss = make_semstokes(qt, 4)
        eta = (x, y) -> exp(-log(1000.0) * (1.0 - y))
        op = semquad_kop(ss, qt; deviatoric=false)
        etav = semquad_eta_field(op, eta)
        work = SemQuadKWork(op.n, op.nq, 2 * op.N)
        v = randn(2 * op.N)
        ref = zeros(Float64, 2 * op.N)
        semquad_apply_K!(ref, ss, qt, etav, v; work=work)
        out = similar(ref)
        # ---- 1. element-buffer kernel == serial blocked kernel, bitwise -----
        for nt in (2, 4, 8)
            par = QuaST.SemQuadKPar(op, nt)
            @test par.nt == nt
            @test par.bounds[1] == 1 && par.bounds[end] == op.nleaf + 1
            QuaST._semquad_apply_K_raw_threaded!(out, op, etav, v, par)
            @test out == ref
            out2 = similar(ref)
            QuaST._semquad_apply_K_raw_threaded!(out2, op, etav, v, par)
            @test out2 == out                    # repeatable run to run
        end
        # more blocks than leaves is clamped, not an error
        parbig = QuaST.SemQuadKPar(op, op.nleaf + 7)
        @test parbig.nt == op.nleaf
        # ---- 2. constrained (Kred) apply ------------------------------------
        sys = solve_semquad_stokes_var(ss, qt, eta, _G190_Z2, _G190_Z2,
                                       _G190_Z2, _G190_Z2; bc=:free,
                                       linear=:direct, assemble_only=true)
        Cu = sys.Cu
        nf = size(Cu, 2)
        vr = randn(nf)
        kr1 = zeros(Float64, nf)
        QuaST._semquad_apply_Kred!(kr1, op, Cu, etav, vr, work)
        kr2 = zeros(Float64, nf)
        for nt in (2, 4)
            par = QuaST.SemQuadKPar(op, nt)
            QuaST._semquad_apply_Kred_threaded!(kr2, op, Cu, etav, vr, work,
                                                    par)
            @test kr2 == kr1
        end
        @info "G190 apply bitwise" case = case leaves = length(qt.leaves) nf = nf
        # ---- 3. V-cycle ------------------------------------------------------
        mgkw = (cycles=1, smoother=:jacobi, operator=:matrixfree,
                coarse_operator=:rediscretize, smoothing=:auto)
        mg1 = QuaST.make_semquad_mg(ss, qt, 4, :free, false, eta, nothing;
                                        mgkw..., mg_threads=1)
        @test mg1.threads == 1
        @test isempty(mg1.par)
        b = randn(mg1.geo.nfs[end])
        z1 = QuaST.semquad_mg_apply(mg1, b)
        # default `mg_threads` is the serial path bitwise
        mgd = QuaST.make_semquad_mg(ss, qt, 4, :free, false, eta, nothing;
                                        mgkw...)
        @test QuaST.semquad_mg_apply(mgd, b) == z1
        for nt in (2, 4, 8)
            # a hierarchy built *directly* with nt blocks (works in a
            # single-threaded process as well: `@threads` then runs them in
            # sequence and the merge is still the same code)
            par = Any[QuaST.SemQuadKPar(mg1.mfops[k], nt)
                      for k in 1:(mg1.geo.L + 1)]
            mgt = QuaST.SemQuadMG(mg1.geo, mg1.A, mg1.D, mg1.cA,
                                      mg1.cycles, mg1.pre, mg1.post, mg1.omega,
                                      mg1.smoother, mg1.LF, mg1.UF,
                                      mg1.operator, mg1.mfops, mg1.mfetas,
                                      mg1.mfwork, mg1.sc, mg1.sr, mg1.sy, nt,
                                      par, mg1.Pts, mg1.rcs, mg1.ecs, mg1.pys)
            @test QuaST.semquad_mg_apply(mgt, b) == z1
        end
        # the sparse operator path is not threaded; mg_threads must not change it
        spkw = (cycles=1, smoother=:jacobi, coarse_operator=:rediscretize,
                smoothing=:auto)
        mgs1 = QuaST.make_semquad_mg(ss, qt, 4, :free, false, eta,
                                         sys.Kred; operator=:sparse, spkw...,
                                         mg_threads=1)
        mgs4 = QuaST.make_semquad_mg(ss, qt, 4, :free, false, eta,
                                         sys.Kred; operator=:sparse, spkw...,
                                         mg_threads=4)
        @test mgs4.threads == 1
        @test isempty(mgs4.par) && isempty(mgs4.Pts)
        @test QuaST.semquad_mg_apply(mgs4, b) ==
              QuaST.semquad_mg_apply(mgs1, b)
        # ---- 4. the whole saddle solve --------------------------------------
        fz = (x, y) -> sin(pi * x / qt.Lx) * sin(pi * y)
        solve(nt) = solve_semquad_stokes_var(ss, qt, eta, _G190_Z2, fz,
                                             _G190_Z2, _G190_Z2;
                                             bc=:free, linear=:mg,
                                             mg_threads=nt)
        u1, p1, = solve(1)
        it1 = LAST_SEMSTOKES_SCHUR_ITERS[]
        res1 = LAST_SEMSTOKES_RESIDUAL[]
        @test LAST_SEMSTOKES_RESIDUAL[] <= 1.0e-8
        for nt in (2, 4, 8)
            ut, pt, = solve(nt)
            # bitwise: the V-cycle, the outer operator (`applyK`) and the BFBT
            # `S2` are all the same threaded apply
            @test ut == u1
            @test pt == p1
            @test LAST_SEMSTOKES_SCHUR_ITERS[] == it1
            @test LAST_SEMSTOKES_RESIDUAL[] == res1
        end
        @info "G190 solve bitwise" case = case outer = it1 res = res1
    end

    # ---- 5. option validation and the thread-count policy -------------------
    qts = _g190_uniform(2)
    sss = make_semstokes(qts, 4)
    fzs = (x, y) -> sin(pi * x / qts.Lx) * sin(pi * y)
    @test_throws ArgumentError QuaST.make_semquad_mg(
        sss, qts, 4, :free, false, _G190_ONE, nothing; mg_threads=0)
    @test_throws ArgumentError solve_semquad_stokes_var(
        sss, qts, _G190_ONE, _G190_Z2, fzs, _G190_Z2, _G190_Z2;
        bc=:free, linear=:mg, mg_threads=0)
    nproc = Threads.nthreads()
    mgkw = (operator=:matrixfree, cycles=1)
    # the warnings are emitted once per process (`maxlog=1`); earlier solves in
    # this gate already requested threads, so re-arm the one-shot flags
    if nproc == 1
        local mgw
        QuaST._SEMQUAD_MG_NO_THREADS_WARNED[] = false
        @test_logs (:warn, r"falling back to the serial V-cycle") mgw =
            QuaST.make_semquad_mg(sss, qts, 4, :free, false, _G190_ONE,
                                      nothing; mgkw..., mg_threads=8)
        @test mgw.threads == 1
        @test isempty(mgw.par)
    else
        local mgw
        QuaST._SEMQUAD_MG_CLAMP_WARNED[] = false
        @test_logs (:warn, r"using .* leaf blocks") mgw =
            QuaST.make_semquad_mg(sss, qts, 4, :free, false, _G190_ONE,
                                      nothing; mgkw..., mg_threads=nproc + 1)
        @test mgw.threads == nproc
        @test length(mgw.par) == mgw.geo.L + 1
        # the coarsest level holds a single leaf, so it runs one worker; the
        # fine level gets all nproc workers
        @test mgw.par[1].nt == 1
        @test mgw.par[end].nt == min(nproc, length(qts.leaves))
        @test length(mgw.Pts) == mgw.geo.L
    end
    # a threaded V-cycle on a small tree still matches serial (the merge path
    # with more blocks than leaves on some levels cannot mis-order anything)
    mg4 = QuaST.make_semquad_mg(sss, qts, 4, :free, false, _G190_ONE,
                                    nothing; mgkw...,
                                    mg_threads=min(4, nproc))
    b = randn(mg4.geo.nfs[end])
    mg1 = QuaST.make_semquad_mg(sss, qts, 4, :free, false, _G190_ONE,
                                    nothing; mgkw...)
    @test QuaST.semquad_mg_apply(mg4, b) ==
          QuaST.semquad_mg_apply(mg1, b)
end
