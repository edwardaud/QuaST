using Test, MantleSEM, LinearAlgebra, SparseArrays, Random

# G223: the S6 consolidated V-cycle is bitwise the per-call path, and it really
# removes fork/joins.
#
# `_semquad_mg_vcycle_region!` runs a threaded level's pre-smoothing sweeps, its
# residual and its restriction inside **one** fork/join (`SemQuadParRegion`), and
# its prolongation, `x .+= py` and post-smoothing sweeps inside a second one,
# instead of creating a task set per op (`_semquad_td_run!` and the threaded leaf
# apply).  The ops, their chunk bodies and their order are unchanged, so the
# result must be bit-identical:
#
#   (a) per op: an apply / subtract / Jacobi / restriction / prolongation
#       driven inside a region equals the per-call form, on every level;
#   (b) per V-cycle: `SEMQUAD_MG_REGION[] = true` equals `false` (and both equal
#       the 1-worker serial hierarchy) for every serial threshold, on a
#       multi-level hanging-node tree, with 2 workers and with the process'
#       worker count (8 on the production machine);
#   (c) the fork count actually drops: with `thr = 0` the region path's fork
#       count is exactly two per threaded level, below the per-call count, and
#       it publishes phases (`SEMQUAD_MG_PHASES`);
#   (d) the threaded saddle solve (FGMRES + w-BFBT pressure block + V-cycle) is
#       bitwise between the two paths: same `sol`, same `nit`.
#
# In a 1-thread process the workers are co-scheduled on the calling thread (the
# same construction G190/G222 use with `SemQuadKPar(op, 2)`), so (a)-(d) run in
# CI too; with `--threads=8` the same gate runs the 8-worker hierarchy.

const _G223_Z2 = (x, y) -> 0.0

function _g223_tree()
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
    for lf in sort(collect(qt.leaves))
        x0, _, y0 = MantleSEM.leaf_box(qt, lf)
        (x0 < 0.2 && y0 > 0.5) && refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

"""`mg` rebuilt with `nw` workers (the G190/G222 construction)."""
function _g223_workers(mg::MantleSEM.SemQuadMG, nw::Int)
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

"""Forks, worker tasks and phases of one call of `f`."""
function _g223_forks(f)
    f0 = MantleSEM.SEMQUAD_MG_FORKS[]
    t0 = MantleSEM.SEMQUAD_MG_FORK_TASKS[]
    p0 = MantleSEM.SEMQUAD_MG_PHASES[]
    f()
    return (MantleSEM.SEMQUAD_MG_FORKS[] - f0,
            MantleSEM.SEMQUAD_MG_FORK_TASKS[] - t0,
            MantleSEM.SEMQUAD_MG_PHASES[] - p0)
end

@testset "G223 S6 consolidated V-cycle is bitwise" begin
    Random.seed!(20261112)
    qt = _g223_tree()
    p = 2
    ss = make_semstokes(qt, p)
    eta = (x, y) -> exp(-log(1.0e4) * (1.0 - y))
    mg = MantleSEM.make_semquad_mg(ss, qt, p, :free, false, eta, nothing;
                                   cycles=1, smoother=:jacobi,
                                   operator=:matrixfree,
                                   coarse_operator=:rediscretize,
                                   coarsen_factor=1.5, smoothing=:auto,
                                   mg_threads=1)
    nlev = mg.geo.L + 1
    @info "G223" leaves = length(qt.leaves) levels = nlev dofs = (mg.geo.nfs) threads = Threads.nthreads()
    @test nlev >= 4

    b = randn(mg.geo.nfs[end])
    zref = MantleSEM.semquad_mg_apply(mg, b)          # serial, 1 worker
    oldthr = MantleSEM.SEMQUAD_MG_SERIAL_DOFS[]
    oldreg = MantleSEM.SEMQUAD_MG_REGION[]
    try
        for nw in unique([2, max(2, min(8, Threads.nthreads()))])
            mgw = _g223_workers(mg, nw)
            @test mgw.threads == nw

            # ---- (a) per op inside a region == per call ---------------------
            MantleSEM.SEMQUAD_MG_SERIAL_DOFS[] = 0
            for k in 2:nlev
                # one scratch slot per region worker: size the region to the
                # level's apply (a level with few leaves clamps `par.nt`)
                nwlev = min(nw, (mgw.par[k]::MantleSEM.SemQuadKPar).nt)
                region = MantleSEM.SemQuadParRegion(nwlev)
                nf = mg.geo.nfs[k]
                x = randn(nf)
                y0 = zeros(nf); y1 = zeros(nf)
                MantleSEM._semquad_mg_apply!(y0, x, mgw, k)
                MantleSEM._semquad_region_run!(region) do
                    MantleSEM._semquad_mg_apply!(y1, x, mgw, k, region)
                end
                @test y1 == y0
                yy = randn(nf); src = randn(nf); D = mgw.D[k]
                r0 = zeros(nf); r1 = zeros(nf)
                MantleSEM._semquad_td_sub!(r0, src, yy, nw)
                MantleSEM._semquad_region_run!(region) do
                    MantleSEM._semquad_td_sub!(r1, src, yy, nw, region)
                end
                @test r1 == r0
                j0 = copy(x); j1 = copy(x)
                MantleSEM._semquad_td_jacobi!(j0, r0, D, 0.666, nw)
                MantleSEM._semquad_region_run!(region) do
                    MantleSEM._semquad_td_jacobi!(j1, r0, D, 0.666, nw, region)
                end
                @test j1 == j0
            end
            # transfers: the restriction of level k lands on level k-1
            for k in 2:nlev
                region = MantleSEM.SemQuadParRegion(nw)   # transfers use no scratch
                nclo = mg.geo.nfs[k - 1]
                r = randn(mg.geo.nfs[k])
                c0 = zeros(nclo); c1 = zeros(nclo)
                MantleSEM._semquad_mg_restrict!(c0, mg.geo.Ps[k - 1], r, nw)
                MantleSEM._semquad_region_run!(region) do
                    MantleSEM._semquad_mg_restrict!(c1, mg.geo.Ps[k - 1], r, nw,
                                                    region)
                end
                @test c1 == c0
                e = randn(nclo)
                y0 = zeros(mg.geo.nfs[k]); y1 = zeros(mg.geo.nfs[k])
                Pt = SparseMatrixCSC{Float64,Int}(sparse(transpose(
                                                       mg.geo.Ps[k - 1])))
                MantleSEM._semquad_mg_prolong!(y0, Pt, e, nw)
                MantleSEM._semquad_region_run!(region) do
                    MantleSEM._semquad_mg_prolong!(y1, Pt, e, nw, region)
                end
                @test y1 == y0
            end

            # ---- (b)+(c) per V-cycle, every threshold ----------------------
            for thr in (0, 20480, typemax(Int))
                MantleSEM.SEMQUAD_MG_SERIAL_DOFS[] = thr
                MantleSEM.SEMQUAD_MG_REGION[] = false
                z0 = MantleSEM.semquad_mg_apply(mgw, b)
                f0, _, ph0 = _g223_forks(() -> MantleSEM.semquad_mg_apply(mgw, b))
                MantleSEM.SEMQUAD_MG_REGION[] = true
                z1 = MantleSEM.semquad_mg_apply(mgw, b)
                f1, _, ph1 = _g223_forks(() -> MantleSEM.semquad_mg_apply(mgw, b))
                @info "G223 V-cycle" nw = nw thr = thr percall_forks = f0 region_forks = f1 phases = ph1
                @test z0 == z1
                @test z1 == zref
                @test ph0 == 0
                if thr == 0
                    @test f0 > 2 * (nlev - 1)   # the per-call path forks per op
                    @test f1 == 2 * (nlev - 1)  # two regions per threaded level
                    @test ph1 > 0
                else
                    @test f1 == 0 && ph1 == 0   # nothing is threaded at all
                end
            end
            MantleSEM.SEMQUAD_MG_SERIAL_DOFS[] = typemax(Int)
            MantleSEM.SEMQUAD_MG_REGION[] = true
            @test MantleSEM.semquad_mg_apply(mgw, b) == zref
            _, _, phs = _g223_forks(() -> MantleSEM.semquad_mg_apply(mgw, b))
            @test phs == 0
        end
    finally
        MantleSEM.SEMQUAD_MG_SERIAL_DOFS[] = oldthr
        MantleSEM.SEMQUAD_MG_REGION[] = oldreg
    end

    # ---- (d) the threaded saddle solve is bitwise -------------------------
    fzb = (x, y) -> 1.0                       # nonzero buoyancy: nit > 0
    sys = solve_semquad_stokes_var(ss, qt, eta, _G223_Z2, fzb, _G223_Z2,
                                   _G223_Z2; bc=:free, linear=:direct,
                                   assemble_only=true)
    Cu = SparseMatrixCSC{Float64,Int}(sparse(sys.Cu))
    op = semquad_kop(ss, qt; deviatoric=false)
    ev = semquad_eta_field(op, eta)
    wk = SemQuadKWork(op.n, op.nq, 2 * op.N)
    applyK = (out, v) -> MantleSEM._semquad_apply_Kred!(out, op, Cu, ev, v, wk)
    mgw = _g223_workers(mg, max(2, min(8, Threads.nthreads())))
    Fp, pscale = MantleSEM.semquad_mg_pressure_precond(ss, sys.Cu, sys.Cp,
                                                       sys.Bs, sys.Bt, nothing,
                                                       eta; mode=:auto,
                                                       cache=false,
                                                       applyK=applyK)
    rhs = vcat(sys.fr, sys.gr)
    sols = Vector{Float64}[]
    nits = Int[]
    try
        for region in (false, true)
            MantleSEM.SEMQUAD_MG_REGION[] = region
            sol, nit, relres, _ = MantleSEM.semquad_mg_saddle_solve(
                nothing, sys.Bs, sys.Bt, Fp, pscale, rhs, mgw; tol=1.0e-10,
                maxit=200, restart=100, vel_solver=:vcycle, applyK=applyK)
            @test nit > 0
            @test isfinite(relres)
            push!(sols, sol)
            push!(nits, nit)
        end
    finally
        MantleSEM.SEMQUAD_MG_REGION[] = oldreg
    end
    @test sols[1] == sols[2]
    @test nits[1] == nits[2]
end
