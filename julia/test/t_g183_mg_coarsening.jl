using Test, MantleSEM, LinearAlgebra, SparseArrays, Random

# G183 / S3 stage 1: h-coarsening control and the tapered V-cycle schedule.
#
# Stage 1 changed three things in the multigrid hierarchy, and this gate pins all
# of them:
#
#   (a) `_mg_coarsen_tree(qt; layers, factor)` can do more than one tree layer
#       per hierarchy step, and `factor` stops a step once the leaf count has
#       dropped by that factor.  Every coarse leaf must still be a *union of
#       input leaves* (that is what makes the nodal prolongation exact) and every
#       level must stay 2:1-balanced.
#   (b) `coarse_direct_max=n` cuts the hierarchy at the finest level with at most
#       `n` dofs and solves that level by a direct (sparse Cholesky)
#       factorisation instead of coarsening further.
#   (c) the per-level smoothing schedule (`smoothing=:auto|:uniform|:taper`,
#       `omega`): `:auto` keeps the historical uniform 2+2 schedule unless the
#       hierarchy is top-heavy (finest coarse level keeps more than half of the
#       fine dofs), in which case it uses `1, 2, 2, 4, 4, 8, ...` sweeps from the
#       fine level down with omega=0.55.
#
# The load-bearing assertions are the *no-regression* ones: on a uniform tree the
# new default must reproduce the historical hierarchy and preconditioner
# bitwise, and on a mixed-level tree the solution must stay identical to the
# direct solve to 1e-8.  Measured performance and the negative results (more
# aggressive coarsening costs outer iterations) are in docs/dev/SOLVER.md §14.

const _G183_Z2 = (x, y) -> 0.0
const _G183_ONE = (x, y) -> 1.0

function _g183_uniform(level::Int; cx::Int=1, cy::Int=1)
    qt = Quadtree(level; cells_x=cx, cells_y=cy)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

# A graded AMR-like tree with mixed leaf levels {2,3,4,5} (the production blob
# tree has levels 3-6): a 2x2 seed grid refined towards a disk, box-by-box, so
# the leaves transition gradually instead of collapsing to a uniform mesh.
function _g183_blob_tree(level::Int=5; radius::Float64=0.30, cx::Float64=0.35,
                         cy::Float64=0.35, seed::Int=2)
    qt = Quadtree(level)
    for _ in 1:seed, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    for _ in 1:(level - seed)
        for lf in collect(qt.leaves)
            lf[1] >= level && continue
            h = 1.0 / (1 << lf[1])
            x0 = lf[2] * h; y0 = lf[3] * h
            far = max((x0 - cx)^2 + (y0 - cy)^2,
                      (x0 + h - cx)^2 + (y0 - cy)^2,
                      (x0 - cx)^2 + (y0 + h - cy)^2,
                      (x0 + h - cx)^2 + (y0 + h - cy)^2)
            far < radius^2 && refine!(qt, lf)
        end
        balance!(qt)
    end
    return qt
end

# Mixed-level tree with levels {2,3} (the shape G182 uses): its finest coarsening
# step keeps > 50% of the dofs, so it is top-heavy by construction.
function _g183_mixed_tree()
    qt = Quadtree(4)
    for _ in 1:2, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    for (l, ix, iy) in collect(qt.leaves)
        (l == 2 && ix % 2 == 0 && iy % 2 == 0) && refine!(qt, (l, ix, iy))
    end
    balance!(qt)
    return qt
end

# Every leaf of every coarse tree must be the union of leaves of the input tree
# (equivalently: the input leaves must tile each coarse leaf exactly).
function _g183_union_of_leaves(coarse::Quadtree, fine::Quadtree)
    fs = Set(fine.leaves)
    return all(lf -> _g183_covered(lf, fs, fine.maxlevel), coarse.leaves)
end

function _g183_covered(lf::NTuple{3,Int}, fs::Set{NTuple{3,Int}}, maxlevel::Int)
    lf in fs && return true
    lf[1] >= maxlevel && return false
    for dy in 0:1, dx in 0:1
        _g183_covered((lf[1] + 1, 2lf[2] + dx, 2lf[3] + dy), fs, maxlevel) ||
            return false
    end
    return true
end

function _g183_nfs(ss, qt, p, bc; kw...)
    return MantleSEM._semquad_mg_geometry(ss, qt, p, bc, false; kw...)
end

@testset "G183 MG coarsening + tapered V-cycle" begin
    Random.seed!(20251107)

    # ---- (a0) the historical rule is unchanged ---------------------------
    # One tree layer per step on a graded tree: leaf counts strictly decrease
    # from the fine end, the last (finest) tree is the input tree, every level is
    # 2:1 and every coarse leaf is a union of input leaves.
    qb = _g183_blob_tree(5)
    @test length(unique(lf[1] for lf in qb.leaves)) > 1
    @test max_face_level_jump(qb) <= 1
    hist = MantleSEM._mg_hierarchy_trees(qb)      # coarse first
    counts = [length(q.leaves) for q in hist]
    @test counts[end] == length(qb.leaves)
    @test issorted(counts) && all(diff(counts) .> 0)
    @test hist[end].leaves == qb.leaves
    for q in hist
        @test max_face_level_jump(q) <= 1
        @test _g183_union_of_leaves(q, qb)
    end
    @info "G183 historical hierarchy" leaf_counts = counts

    # ---- (a1) layers / factor coarsen further, same invariants -----------
    for (layers, factor) in ((2, 1.0), (1, 2.0), (1, 3.0), (3, 2.0))
        tr = MantleSEM._mg_hierarchy_trees(qb; layers=layers, factor=factor)
        c = [length(q.leaves) for q in tr]
        @test issorted(c) && all(diff(c) .> 0)
        @test c[end] == length(qb.leaves)
        @test length(tr) <= length(hist)          # never more levels
        for q in tr
            @test max_face_level_jump(q) <= 1
            @test _g183_union_of_leaves(q, qb)
        end
        @info "G183 coarsening variant" layers = layers factor = factor counts = c
    end
    # one step of "layers=2" equals two steps of "layers=1"
    @test length(MantleSEM._mg_coarsen_tree(qb; layers=2).leaves) ==
          length(hist[end - 2].leaves)
    # ... and `factor=2` never coarsens a step by less than 2x
    c2 = [length(q.leaves) for q in
          MantleSEM._mg_hierarchy_trees(qb; factor=2.0)]
    @test all(c2[i + 1] >= 2 * c2[i] for i in 1:(length(c2) - 1))

    # ---- (a2) no-op on a uniform tree -----------------------------------
    qu = _g183_uniform(3)
    h1 = MantleSEM._mg_hierarchy_trees(qu; layers=1, factor=1.0)
    h2 = MantleSEM._mg_hierarchy_trees(qu; layers=1, factor=2.0)
    h3 = MantleSEM._mg_hierarchy_trees(qu; layers=2, factor=1.0)
    @test length(h1) == 4                       # 1 -> 4 -> 16 -> 64 leaves
    @test [q.leaves for q in h1] == [q.leaves for q in h2]   # bitwise trees
    # layers=2 is *not* a no-op: on a uniform tree one layer is exactly 4x, so
    # two layers per step coarsens by at least 4^2 and skips a level.
    @test length(h3) < length(h1)
    c3 = [length(q.leaves) for q in h3]
    @test c3[end] == length(qu.leaves)
    @test all(c3[i + 1] >= 4 * c3[i] for i in 1:(length(c3) - 1))
    @info "G183 uniform hierarchy" leaf_counts = [length(q.leaves) for q in h1]

    # ---- (a3) the preconditioner is bitwise-identical where the
    #           hierarchies coincide (uniform tree, factor 1 vs 2) ----------
    ss = make_semstokes(qu, 4)
    asm = solve_semquad_stokes_var(ss, qu, _G183_ONE, _G183_Z2, _G183_Z2,
                                   _G183_Z2, _G183_Z2; bc=:free,
                                   linear=:direct, assemble_only=true)
    mgA = MantleSEM.make_semquad_mg(ss, qu, 4, :free, false, _G183_ONE,
                                    asm.Kred; cycles=1, operator=:matrixfree,
                                    coarsen_factor=1.0)
    mgB = MantleSEM.make_semquad_mg(ss, qu, 4, :free, false, _G183_ONE,
                                    asm.Kred; cycles=1, operator=:matrixfree,
                                    coarsen_factor=2.0)
    @test mgA.geo.L == mgB.geo.L
    @test mgA.geo.nfs == mgB.geo.nfs
    @test mgA.pre == mgB.pre && mgA.post == mgB.post
    @test mgA.omega == mgB.omega
    b = [sin(0.5 * i) + cos(0.13 * i * i) for i in 1:mgA.geo.nfs[end]]
    zA = MantleSEM.semquad_mg_apply(mgA, b)
    zB = MantleSEM.semquad_mg_apply(mgB, b)
    @test zA == zB                              # bitwise, not just close
    @test mgA.pre == fill(2, mgA.geo.L + 1)     # the historical 2+2 schedule
    @test mgA.omega == 2.0 / 3.0

    # ---- (a4) level count and coarse dofs --------------------------------
    geo = _g183_nfs(ss, qu, 4, :free)
    @test geo.L >= 2
    @test issorted(geo.nfs)
    @test geo.nfs[end] == size(geo.Cus[end], 2)
    @test geo.nfs[end] == length(geo.free_rows[end])
    @test geo.nfs[1] > 0
    for k in 1:(geo.L + 1)
        @test size(geo.Cus[k], 2) == geo.nfs[k]
    end
    for k in 1:geo.L
        @test size(geo.Ps[k]) == (geo.nfs[k + 1], geo.nfs[k])
    end
    @test geo.coarse_direct == false             # default: no cut

    # ---- (b) coarse-direct threshold ------------------------------------
    # Cut the uniform tree at the finest level with <= 400 dofs: the hierarchy
    # keeps its finer part, the new coarsest level is a real mesh, and it is
    # factorised directly (not the base-leaf dense matrix).
    gcut = _g183_nfs(ss, qu, 4, :free; coarse_direct_max=400)
    @test gcut.coarse_direct
    @test gcut.L < geo.L
    @test gcut.nfs[1] <= 400
    @test length(gcut.qts) == gcut.L + 1
    @test length(gcut.Ps) == gcut.L
    mgC = MantleSEM.make_semquad_mg(ss, qu, 4, :free, false, _G183_ONE,
                                    asm.Kred; cycles=1, operator=:matrixfree,
                                    coarse_direct_max=400)
    @test mgC.geo.coarse_direct
    @test isempty(mgC.A)                        # still matrix-free on the fine levels
    @test mgC.cA isa SparseArrays.CHOLMOD.Factor
    zC = MantleSEM.semquad_mg_apply(mgC, b)
    @test all(isfinite, zC)
    @test norm(zC) > 0.0
    # the cut hierarchy still solves the saddle to the direct answer
    fzb = (x, y) -> sin(pi * x) * sin(pi * y)
    uD, _, _, _, _ = solve_semquad_stokes_var(ss, qu, _G183_ONE, _G183_Z2, fzb,
                                              _G183_Z2, _G183_Z2; bc=:free,
                                              linear=:direct)
    uC, _, _, _, _ = solve_semquad_stokes_var(ss, qu, _G183_ONE, _G183_Z2, fzb,
                                              _G183_Z2, _G183_Z2; bc=:free,
                                              linear=:mg,
                                              mg_coarse_direct_max=400)
    @test LAST_SEMSTOKES_RESIDUAL[] < 1.0e-9
    @test norm(uC - uD) / norm(uD) < 1.0e-8
    # the sparse hierarchy supports the same cut
    mgS = MantleSEM.make_semquad_mg(ss, qu, 4, :free, false, _G183_ONE,
                                    asm.Kred; cycles=1, operator=:sparse,
                                    coarse_direct_max=400)
    @test mgS.geo.coarse_direct
    @test mgS.geo.nfs == mgC.geo.nfs
    @test all(isfinite, MantleSEM.semquad_mg_apply(mgS, b))
    @info "G183 coarse-direct cut" nfs = gcut.nfs L = gcut.L

    # ---- (c) smoothing schedule: resolution + no-regression --------------
    # uniform tree -> historical uniform schedule, and the auto default is
    # bitwise the same preconditioner as the explicit historical setting
    mgU = MantleSEM.make_semquad_mg(ss, qu, 4, :free, false, _G183_ONE,
                                    asm.Kred; cycles=1, operator=:matrixfree,
                                    smoothing=:auto)
    mgH = MantleSEM.make_semquad_mg(ss, qu, 4, :free, false, _G183_ONE,
                                    asm.Kred; cycles=1, operator=:matrixfree,
                                    smoothing=:uniform, omega=2.0 / 3.0)
    @test mgU.pre == mgH.pre && mgU.post == mgH.post
    @test mgU.omega == mgH.omega == 2.0 / 3.0
    @test MantleSEM.semquad_mg_apply(mgU, b) ==
          MantleSEM.semquad_mg_apply(mgH, b)

    # top-heavy mixed-level tree: the p=4 auto default keeps the measured
    # front-loaded taper with omega = 0.55 (the p=2 top-heavy rule, uniform 3+3,
    # is pinned by G219 on the production P1-shaped tree)
    qm = _g183_mixed_tree()
    ssm = make_semstokes(qm, 4)
    gm = _g183_nfs(ssm, qm, 4, :free)
    @test gm.nfs[end - 1] > 0.5 * gm.nfs[end]        # top-heavy by construction
    mgm = MantleSEM.make_semquad_mg(ssm, qm, 4, :free, false, _G183_ONE,
                                    nothing; cycles=1,
                                    operator=:matrixfree)
    nlev = mgm.geo.L + 1
    @test mgm.pre == reverse([min(1 << (k ÷ 2), 8) for k in 1:nlev])
    @test mgm.omega == 0.55
    @test mgm.pre[end] == 1                          # one sweep on the fine level
    @test mgm.pre[1] >= 4                            # many on the cheapest levels
    # explicit :uniform on the same tree restores the historical 2+2 schedule
    mgm2 = MantleSEM.make_semquad_mg(ssm, qm, 4, :free, false, _G183_ONE,
                                     nothing; cycles=1,
                                     operator=:matrixfree, smoothing=:uniform)
    @test mgm2.pre == fill(2, nlev) && mgm2.omega == 2.0 / 3.0
    # an explicit omega overrides the schedule default
    mgm3 = MantleSEM.make_semquad_mg(ssm, qm, 4, :free, false, _G183_ONE,
                                     nothing; cycles=1,
                                     operator=:matrixfree, smoothing=:taper,
                                     omega=0.6)
    @test mgm3.omega == 0.6 && mgm3.pre == mgm.pre

    # ---- (c2) solution unchanged <= 1e-8 on a mixed-level tree -----------
    # 16384x contrast, levels {2,3}: the new default must reproduce the direct
    # solution, and the two schedules must agree with each other.
    qmx = _g183_mixed_tree()
    ssmx = make_semstokes(qmx, 4)
    etam = (x, y) -> exp(-log(16384.0) * (1 - y))
    fzm = (x, y) -> sin(pi * x / qmx.Lx) * sin(pi * y)
    z2 = _G183_Z2
    uD, _, _, _, _ = solve_semquad_stokes_var(ssmx, qmx, etam, z2, fzm, z2, z2;
                                              bc=:free, linear=:direct)
    resD = LAST_SEMSTOKES_RESIDUAL[]
    uA, _, _, _, _ = solve_semquad_stokes_var(ssmx, qmx, etam, z2, fzm, z2, z2;
                                              bc=:free, linear=:mg)
    resA = LAST_SEMSTOKES_RESIDUAL[]; itA = LAST_SEMSTOKES_SCHUR_ITERS[]
    uH, _, _, _, _ = solve_semquad_stokes_var(ssmx, qmx, etam, z2, fzm, z2, z2;
                                              bc=:free, linear=:mg,
                                              mg_smoothing=:uniform,
                                              mg_omega=2.0 / 3.0)
    resH = LAST_SEMSTOKES_RESIDUAL[]; itH = LAST_SEMSTOKES_SCHUR_ITERS[]
    @test norm(uD) > 1.0e-8
    @test resD < 1.0e-11
    @test resA < 1.0e-9
    @test resH < 1.0e-9
    @test norm(uA - uD) / norm(uD) < 1.0e-8
    @test norm(uH - uD) / norm(uD) < 1.0e-8
    @test norm(uA - uH) / norm(uA) < 1.0e-8
    @test abs(itA - itH) <= 10        # the schedule is a preconditioner change
    @info "G183 mixed-level solution" du_auto = norm(uA - uD) / norm(uD) du_uniform = norm(uH - uD) / norm(uH) iters = (itA, itH) res = (resA, resH)

    # ---- (c3) the direct path is untouched -------------------------------
    uD2, _, _, _, _ = solve_semquad_stokes_var(ssmx, qmx, etam, z2, fzm, z2, z2;
                                               bc=:free, linear=:direct,
                                               mg_coarsen_factor=2.0,
                                               mg_coarsen_layers=2,
                                               mg_smoothing=:taper,
                                               mg_omega=0.5,
                                               mg_coarse_direct_max=1000)
    @test uD2 == uD
    @test LAST_SEMSTOKES_RESIDUAL[] == resD

    # ---- argument validation ---------------------------------------------
    @test_throws ArgumentError MantleSEM._mg_coarsen_tree(qu; layers=0)
    @test_throws ArgumentError MantleSEM._mg_coarsen_tree(qu; factor=0.5)
    @test_throws ArgumentError MantleSEM._mg_hierarchy_trees(qu; factor=Inf)
    @test_throws ArgumentError MantleSEM.make_semquad_mg(ss, qu, 4, :free,
                                                         false, _G183_ONE,
                                                         asm.Kred;
                                                         coarsen_layers=0)
    @test_throws ArgumentError MantleSEM.make_semquad_mg(ss, qu, 4, :free,
                                                         false, _G183_ONE,
                                                         asm.Kred;
                                                         coarsen_factor=0.5)
    @test_throws ArgumentError MantleSEM.make_semquad_mg(ss, qu, 4, :free,
                                                         false, _G183_ONE,
                                                         asm.Kred;
                                                         coarse_direct_max=-1)
    @test_throws ArgumentError MantleSEM.make_semquad_mg(ss, qu, 4, :free,
                                                         false, _G183_ONE,
                                                         asm.Kred;
                                                         smoothing=:bogus)
    @test_throws ArgumentError MantleSEM.make_semquad_mg(ss, qu, 4, :free,
                                                         false, _G183_ONE,
                                                         asm.Kred; omega=1.5)
    @test_throws ArgumentError MantleSEM.make_semquad_mg(ss, qu, 4, :free,
                                                         false, _G183_ONE,
                                                         asm.Kred;
                                                         smoothing=:taper,
                                                         pre=[1, 2])
    @test_throws ArgumentError solve_semquad_stokes_var(ss, qu, _G183_ONE,
                                                        _G183_Z2, fzb,
                                                        _G183_Z2, _G183_Z2;
                                                        bc=:free, linear=:mg,
                                                        mg_coarsen_factor=0.9)
    @test_throws ArgumentError solve_semquad_stokes_var(ss, qu, _G183_ONE,
                                                        _G183_Z2, fzb,
                                                        _G183_Z2, _G183_Z2;
                                                        bc=:free, linear=:mg,
                                                        mg_smoothing=:bogus)
    @test_throws ArgumentError solve_semquad_stokes_var(ss, qu, _G183_ONE,
                                                        _G183_Z2, fzb,
                                                        _G183_Z2, _G183_Z2;
                                                        bc=:free, linear=:mg,
                                                        mg_omega=2.0)
end
