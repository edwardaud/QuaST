# G271: mapped multigrid and matrix-free kernels (stage 0 cut 5).
using Test, QuaST, LinearAlgebra, SparseArrays, Random
using QuaST: Quadtree, SemStokes, SemQuad, CartesianBox, AffineBox, CurvedBox,
             make_semstokes, make_semquad_mg, semquad_kop, semquad_apply_K!,
             semquad_apply_Kred!, semquad_eta_field, semquad_eta_lanes,
             SEMQUAD_KOP_CACHE, SEMQUAD_MG_CACHE, SemQuadKWork, SemQuadKOp,
             SemQuadKOpMapped, SemQuadMGGeom, SemQuadMGLevelKey,
             _semquad_kop_key, _saddle_key, geometry_stamp, geometry_identity,
             _semquad_mg_apply!, _semquad_apply_K_raw!, semquad_mg_apply,
             _semquad_mg_level_count, _semquad_mg_leaf_hash

# ---------------------------------------------------------------------------
# G271 -- the mapped matrix-free operator and the mapped multigrid hierarchy.
#
# Cut 4 left the matrix-free operator (`stokes/kop.jl`, `stokes/apply.jl`,
# `stokes/diag.jl`) and the whole `src/mg/` layer on the historical box kernels,
# and the driver/solve refused a mapped geometry for `linear=:mg`.  Cut 5 routes
# them through the geometry and adds the metric/curvature/face cache.  This gate
# is the proof:
#
#   (a) mapped sparse `K` vs the mapped matrix-free apply: relative difference
#       <= 1e-12 (the box analogue is G175, exact to round-off);
#   (b) mapped `linear=:mg` vs mapped `linear=:direct`: relative difference
#       <= 1e-6 (the BC feature's G-path precedent).  Both solve the same
#       discrete saddle; the direct path has a true reduced residual ~1e-12 and
#       the MG path stops at its own `mg_tol = 1e-10` true residual, so 1e-6 is
#       4 orders above the tolerance the two solutions are computed to -- it is
#       the solution agreement, not a residual relaxation;
#   (c) two geometries on one tree produce two distinct operators, two distinct
#       `SEMQUAD_KOP_CACHE` entries, two distinct MG hierarchies and two
#       distinct `_saddle_key`s, and the second warm mapped apply allocates
#       exactly 0 B (the GEOM_DESIGN §2.5 steady state);
#   (d) the pre-cut-5 key (no geometry) *does* collide on exactly that pair, and
#       the failure it would cause is measured: the wrong (box) velocity
#       operator solves the mapped system *silently* (its own residual is round-
#       off) to a wrong velocity, and the wrong (box) MG hierarchy is a much
#       worse preconditioner for the mapped operator;
#   (e) the mapped hierarchy's coarse levels are **logical** coarsenings: their
#       leaf sets equal the box hierarchy's level by level, each mapped coarse
#       operator equals the mapped sparse rediscretisation on that coarse tree,
#       the Jacobi diagonal is the exact `diag(Cu' K Cu)`, and the prolongation
#       weights equal the box hierarchy's.  On the box map (`AffineBox(I,0,Lx)`)
#       the whole mapped hierarchy (operators, diagonal, transfer) reproduces
#       the box hierarchy; on `AffineBox`/`CurvedBox` the coarse operators are
#       the *mapped* ones, so a box coarse operator is caught.
#   (f) the mapped driver runs the linear paths this cut supports (`:direct`,
#       `:mg`, `:auto`) end to end, and `:schur_cg`/`:lagged` are still refused
#       with an actionable message.
#
# Non-tautology.  Every comparison is between two *independent* implementations
# or between a production path and a gate-side reconstruction:
#   * (a) compares the matrix-free kernel family against the assembled sparse
#     operator: the sparse path was proved against the box and the affine map by
#     G175/G267/G268, and the matrix-free path is a different code body (it
#     never forms a matrix), so a transposed `J^{-T}`, a missing chain-rule term
#     or a wrong measure shows up as an O(1) difference;
#   * (b) compares two different solvers: a sparse LU on the mapped assembled
#     saddle vs the matrix-free mapped kernels + V-cycle preconditioner +
#     FGMRES.  A shared bug would have to be present in both the assembled and
#     matrix-free mapped bodies, which are different code;
#   * (c)/(d) compare *different* things that must differ: the box and mapped
#     metrics are different functions on the same leaf set, and the gate also
#     reconstructs the pre-cut key itself to show the collision is real rather
#     than asserted;
#   * (e) compares the hierarchy levels against the sparse mapped assembly on
#     the same coarse tree (different code paths: rediscretised assembly vs the
#     leaf-local matrix-free operator) and against the box hierarchy whose
#     logical coarsening is geometry-independent;
#   * (f) checks the driver's own recorded step count and refusal message.
# ---------------------------------------------------------------------------

const _G271_Z2 = (x, y) -> 0.0
const _G271_GAFF = AffineBox((1.25, 0.35, -0.20, 0.95), (0.10, -0.05), 1.0)
const _G271_GCUR = CurvedBox(0.20, 1.0)
const _G271_GEOMETRIES = (("curved", _G271_GCUR), ("affine", _G271_GAFF))

function _g271_uniform(level::Int; cx::Int=1, cy::Int=1)
    qt = Quadtree(level; cells_x=cx, cells_y=cy)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

# A non-uniform tree with genuine hanging constraints: refine the lower-left
# quadrant one level beyond the uniform refinement, then balance.  The tree is
# built with `maxlevel = level + 1` (a `Quadtree(level)` is a hard cap at
# `level`, so refining it again would silently be a no-op).
function _g271_mixed(level::Int)
    qt = Quadtree(level + 1)
    for _ in 1:level, lf in collect(qt.leaves)
        refine!(qt, lf)
    end
    balance!(qt)
    for lf in sort(collect(qt.leaves))
        l = lf[1]
        n = 1 << l
        (lf[2] < n ÷ 2 && lf[3] < n ÷ 2) && refine!(qt, lf)
    end
    balance!(qt)
    return qt
end

_g271_sparse(ss, qt, eta; deviatoric::Bool=false,
             fx::Function=_G271_Z2, fz::Function=_G271_Z2) =
    solve_semquad_stokes_var(ss, qt, eta, fx, fz, _G271_Z2,
                             _G271_Z2; bc=:rigid, deviatoric=deviatoric,
                             linear=:direct, assemble_only=true)

_g271_hierarchy(ss, qt, p::Int, eta, Kred; operator::Symbol=:matrixfree,
                mg_threads::Int=1) =
    make_semquad_mg(ss, qt, p, :rigid, false, eta, Kred; cycles=1,
                    operator=operator, mg_threads=mg_threads)

# `@allocated` helpers in function scope (a global-scope measurement would count
# the boxing of the globals, not the kernel).
function _g271_alloc_raw(op, etav, v, work, out)
    _semquad_apply_K_raw!(out, op, etav, v, work)
    return @allocated _semquad_apply_K_raw!(out, op, etav, v, work)
end
function _g271_alloc_mg_apply(mg, lev, x, y)
    _semquad_mg_apply!(y, x, mg, lev)
    return @allocated _semquad_mg_apply!(y, x, mg, lev)
end

@testset "G271 mapped multigrid and matrix-free operator" begin
    Random.seed!(20261124)

    # ======================================================================
    # (a) mapped sparse K == mapped matrix-free apply
    # ======================================================================
    println("== G271 (a) mapped sparse K vs mapped matrix-free apply ==")
    for (gtag, g) in _G271_GEOMETRIES
        for (ttag, qt) in (("uniform", _g271_uniform(3)),
                           ("mixed", _g271_mixed(3)))
            for (etag, eta) in (("var", (x, y) -> 1.0 + 0.2 * x - 0.1 * y),
                                ("contrast", (x, y) -> exp(-log(100.0) * (1.0 - y))))
                p = (gtag == "affine" && ttag == "uniform" && etag == "var") ? 3 : 2
                ss = make_semstokes(qt, p; geom=g)
                asm = _g271_sparse(ss, qt, eta)
                N2 = 2 * ss.su.N
                v = randn(N2)
                ref = asm.K * v
                out = zeros(Float64, N2)
                semquad_apply_K!(out, ss, qt, eta, v)
                rel = norm(out - ref) / norm(ref)
                @test rel < 1.0e-12
                vr = randn(size(asm.Cu, 2))
                refr = asm.Kred * vr
                outr = zeros(Float64, size(asm.Cu, 2))
                semquad_apply_Kred!(outr, ss, qt, asm.Cu, eta, vr)
                relr = norm(outr - refr) / norm(refr)
                @test relr < 1.0e-12
                println("  ", rpad(gtag * "/" * ttag, 16), " ", rpad(etag, 9),
                        " p=$p leaves=", length(qt.leaves),
                        " full=", rel, " reduced=", relr)
            end
        end
    end
    # deviatoric variant (the a2/c2 coefficients differ from the non-deviatoric
    # body; a swapped a2/c2 passes the constant-viscosity case but not this one
    # only if the stress form is wrong, so pin it explicitly)
    let qt = _g271_uniform(3), p = 2
        g = _G271_GCUR
        ss = make_semstokes(qt, p; geom=g)
        eta = (x, y) -> 1.0 + 0.3 * y
        asm = _g271_sparse(ss, qt, eta; deviatoric=true)
        v = randn(2 * ss.su.N)
        ref = asm.K * v
        out = zeros(Float64, length(v))
        semquad_apply_K!(out, ss, qt, eta, v; deviatoric=true)
        rel = norm(out - ref) / norm(ref)
        @test rel < 1.0e-12
        println("  deviatoric full=", rel)
    end

    # ======================================================================
    # (b) mapped linear=:mg vs mapped linear=:direct
    # ======================================================================
    println("== G271 (b) mapped :mg vs :direct ==")
    for (gtag, g) in _G271_GEOMETRIES
        for (ttag, qt) in (("uniform", _g271_uniform(3)),
                           ("mixed", _g271_mixed(3)))
            p = 2
            ss = make_semstokes(qt, p; geom=g)
            eta = (x, y) -> exp(-log(50.0) * (1.0 - y))
            fx = (x, y) -> sinpi(x) * sinpi(y)
            fz = (x, y) -> 0.5 * cospi(x) * sinpi(y)
            ud, pd = solve_semquad_stokes_var(ss, qt, eta, fx, fz, _G271_Z2,
                                              _G271_Z2; bc=:rigid,
                                              linear=:direct)
            rd = QuaST.LAST_SEMSTOKES_RESIDUAL[]
            um, pm = solve_semquad_stokes_var(ss, qt, eta, fx, fz, _G271_Z2,
                                              _G271_Z2; bc=:rigid, linear=:mg,
                                              mg_operator=:matrixfree,
                                              mg_tol=1.0e-10)
            rm = QuaST.LAST_SEMSTOKES_RESIDUAL[]
            du = norm(um - ud) / norm(ud)
            dp = norm(pm - pd) / norm(pd)
            @test du < 1.0e-6
            @test dp < 1.0e-6
            # both solves satisfy their own reduced saddle much tighter than the
            # 1e-6 agreement bound (direct ~round-off, MG at `mg_tol`=1e-10)
            @test rd < 1.0e-8
            @test rm < 1.0e-8
            println("  ", rpad(gtag * "/" * ttag, 16),
                    " |du|=", du, " |dp|=", dp,
                    " resid_direct=", rd, " resid_mg=", rm)
        end
    end

    # ======================================================================
    # (c) two geometries on one tree: distinct operators, caches, zero-allocation
    # ======================================================================
    println("== G271 (c) two geometries on one tree ==")
    let qt = _g271_uniform(4), p = 2, eta = (x, y) -> exp(-log(50.0) * (1.0 - y))
        gA = _G271_GCUR
        gB = CurvedBox(0.35, 1.0)
        ssA = make_semstokes(qt, p; geom=gA)
        ssB = make_semstokes(qt, p; geom=gB)
        opA = semquad_kop(ssA, qt)
        opB = semquad_kop(ssB, qt)
        # two *different* operators of the mapped type (a geometry-forgotten
        # cache would hand back the same object)
        @test opA isa SemQuadKOpMapped
        @test opB isa SemQuadKOpMapped
        @test opA !== opB
        # the leaf/dof structure is the tree's, not the geometry's
        @test opA.ids == opB.ids
        # ... but the metric is the geometry's
        @test opA.detJ != opB.detJ
        @test any(opA.j11 .!= opB.j11)
        keyA = _semquad_kop_key(ssA, qt, false)
        keyB = _semquad_kop_key(ssB, qt, false)
        @test keyA != keyB
        # the *only* difference between the two keys is the geometry stamp: the
        # tree/degree fields are identical, so the pre-cut key (which stopped at
        # field 8) would have collided
        @test keyA[1:8] == keyB[1:8]
        @test keyA[9] != keyB[9]
        nkop = count(k -> k isa Tuple && length(k) == 9 && k[9] isa UInt64,
                     keys(SEMQUAD_KOP_CACHE))
        @test nkop >= 2
        v = randn(2 * ssA.su.N)
        oa = zeros(Float64, length(v))
        ob = zeros(Float64, length(v))
        semquad_apply_K!(oa, ssA, qt, eta, v)
        semquad_apply_K!(ob, ssB, qt, eta, v)
        dop = norm(oa - ob) / norm(oa)
        @test dop > 1.0e-3      # measured O(1e-1): a shared operator is caught
        # MG hierarchies: two distinct geometry records and two distinct cache
        # entries under one tree
        asmA = _g271_sparse(ssA, qt, eta)
        asmB = _g271_sparse(ssB, qt, eta)
        mgA = _g271_hierarchy(ssA, qt, p, eta, asmA.Kred)
        mgB = _g271_hierarchy(ssB, qt, p, eta, asmB.Kred)
        @test mgA.geo !== mgB.geo
        @test mgA.geo.geom === gA
        @test mgB.geo.geom === gB
        @test geometry_identity(mgA.geo.geom) != geometry_identity(mgB.geo.geom)
        ngeo = count(x -> x isa SemQuadMGGeom, values(SEMQUAD_MG_CACHE))
        @test ngeo >= 2
        # a `SemQuadMGLevelKey` carries the geometry stamp too
        stampA = geometry_stamp(gA)
        stampB = geometry_stamp(gB)
        @test stampA != stampB
        # stage 0.5c1 adds the tree topology (seam) as the last key field
        lkA = SemQuadMGLevelKey(1, length(qt.leaves),
                                _semquad_mg_leaf_hash(qt.leaves), p, qt.nx0,
                                qt.ny0, qt.Lx, :rigid, false,
                                (:same, :same, :same, :same), stampA, qt.seam)
        lkB = SemQuadMGLevelKey(1, length(qt.leaves),
                                _semquad_mg_leaf_hash(qt.leaves), p, qt.nx0,
                                qt.ny0, qt.Lx, :rigid, false,
                                (:same, :same, :same, :same), stampB, qt.seam)
        @test lkA != lkB
        @test hash(lkA, zero(UInt)) != hash(lkB, zero(UInt))
        # `_saddle_key`: the pressure-cache/stale-LU key gains the geometry
        nf = size(asmA.Kred, 1)
        kA = _saddle_key(qt, gA, p, :rigid, nf, 0)
        kB = _saddle_key(qt, gB, p, :rigid, nf, 0)
        @test kA != kB
        @test kA == _saddle_key(qt, gA, p, :rigid, nf, 0)   # stable, not random
        # the two hierarchies are different preconditioners for the same b
        b = randn(nf)
        zA = semquad_mg_apply(mgA, b)
        zB = semquad_mg_apply(mgB, b)
        dz = norm(zA - zB) / norm(zA)
        @test dz > 1.0e-3      # measured O(1e-1)
        # ---- steady-state allocation: second warm mapped apply = 0 B ---------
        op = opA
        w = SemQuadKWork(op)
        etav = semquad_eta_lanes(op, semquad_eta_field(op, eta))
        outw = zeros(Float64, 2 * op.N)
        _semquad_apply_K_raw!(outw, op, etav, v, w)
        a1 = _g271_alloc_raw(op, etav, v, w, outw)
        a2 = _g271_alloc_raw(op, etav, v, w, outw)
        @test a1 == 0
        @test a2 == 0
        # the public entry too (cache hit + kernel: no per-apply allocation)
        semquad_apply_K!(outw, ssA, qt, etav, v; work=w)
        a3 = @allocated semquad_apply_K!(outw, ssA, qt, etav, v; work=w)
        a4 = @allocated semquad_apply_K!(outw, ssA, qt, etav, v; work=w)
        @test a3 == 0
        @test a4 == 0
        # the MG-level apply of the mapped hierarchy (what the V-cycle pays)
        bmg = randn(mgA.geo.nfs[end])
        ymg = zeros(Float64, length(bmg))
        m1 = _g271_alloc_mg_apply(mgA, mgA.geo.L + 1, bmg, ymg)
        m2 = _g271_alloc_mg_apply(mgA, mgA.geo.L + 1, bmg, ymg)
        @test m1 == 0
        @test m2 == 0
        println("  op diff=", dop, "  hierarchy diff=", dz,
                "  alloc raw=", (a1, a2), " public=", (a3, a4),
                " mg-apply=", (m1, m2))
        # ---- the metric cache itself: warm leaf_metric allocates nothing and
        # the metric table is keyed by the geometry -------------------------
        fd = QuaST.make_fedata(p, p)
        lm = QuaST.leaf_metric(gA, ssA.su, ssA.su.leaves[1], fd)
        a5 = @allocated QuaST.leaf_metric(gA, ssA.su, ssA.su.leaves[1], fd)
        @test a5 == 0
        lmB = QuaST.leaf_metric(gB, ssB.su, ssB.su.leaves[1], fd)
        @test lm !== lmB
        @test lm.detJ != lmB.detJ
        @test QuaST.leaf_metric(gA, ssA.su, ssA.su.leaves[1], fd) === lm
        # mapped assembly steady state: the second and third warm assemblies
        # allocate exactly the same bytes -- the per-call metric rebuild of
        # cuts 2-4 is gone (the COO buffers are the assembly's own structure)
        th = zeros(Float64, ssA.su.N)
        ux = zeros(Float64, ssA.su.N)
        uz = zeros(Float64, ssA.su.N)
        pr = Params(Ra=1.0e3, H=0.0, bc=:rigid)
        QuaST.assemble_semquad_energy(ssA.su, qt, pr, ux, uz)
        e2 = @allocated QuaST.assemble_semquad_energy(ssA.su, qt, pr, ux, uz)
        e3 = @allocated QuaST.assemble_semquad_energy(ssA.su, qt, pr, ux, uz)
        @test e2 == e3
        stats = QuaST.semquad_geometry_cache_stats()
        @test stats.metric_tables >= 2
        println("  assembly warm alloc=", (e2, e3), "  cache=", stats)
    end

    # ======================================================================
    # (d) a geometry-forgotten key is caught, and its failure is measured
    # ======================================================================
    println("== G271 (d) wrong cache key: collision + measured failure ==")
    let qt = _g271_uniform(3), p = 2,
        eta = (x, y) -> exp(-log(30.0) * (1.0 - y))
        # A shear affine map keeps the logical wall/dof structure of the box
        # (the Dirichlet dof set is logical) but changes the metric, so the two
        # spaces have identical dof numbering and different operators: exactly
        # the pair a geometry-forgotten key would confuse.
        gbox = CartesianBox(1.0)
        gshear = AffineBox((1.0, 0.6, 0.0, 1.0), (0.0, 0.0), 1.0)
        ssB = make_semstokes(qt, p; geom=gbox)
        ssM = make_semstokes(qt, p; geom=gshear)
        fx = (x, y) -> sinpi(x) * sinpi(y)
        fz = (x, y) -> 0.5 * cospi(x) * sinpi(y)
        # the same *data* on both geometries (nonzero, so the right-hand side is
        # a nontrivial vector and the wrong operator really changes the answer)
        asmB = _g271_sparse(ssB, qt, eta; fx=fx, fz=fz)
        asmM = _g271_sparse(ssM, qt, eta; fx=fx, fz=fz)
        @test size(asmB.Cu) == size(asmM.Cu)
        @test asmB.Cu == asmM.Cu          # same logical dofs and constraints
        # -- the pre-cut-5 key (no geometry): the collision is real --------------
        oldkey(ss) = hash((ss.p, ss.q, qt.nx0, qt.ny0, qt.Lx, false,
                           length(qt.leaves), hash(qt.leaves)))
        @test oldkey(ssB) == oldkey(ssM)
        # -- the cut-5 key distinguishes them -----------------------------------
        @test _semquad_kop_key(ssB, qt, false) != _semquad_kop_key(ssM, qt, false)
        @test geometry_stamp(gbox) != geometry_stamp(gshear)
        nf = size(asmM.Cu, 2)
        @test _saddle_key(qt, gbox, p, :rigid, nf, 0) !=
              _saddle_key(qt, gshear, p, :rigid, nf, 0)
        # -- the operators really differ (so a wrong hit is an O(1) error) -----
        dK = norm(asmB.K - asmM.K) / norm(asmM.K)
        @test dK > 1.0e-3
        # forcing: a nontrivial right-hand side (the same data on both geometries,
        # so the *system* differs only through the geometry)
        fx = (x, y) -> sinpi(x) * sinpi(y)
        fz = (x, y) -> 0.5 * cospi(x) * sinpi(y)
        ud, pd = solve_semquad_stokes_var(ssM, qt, eta, fx, fz, _G271_Z2,
                                          _G271_Z2; bc=:rigid, linear=:direct)
        # -- simulate the wrong-key hit: the mapped system solved with the box
        #    velocity operator (this is exactly the operator a box `Kop` cache
        #    entry would hand to the mapped solve) ------------------------------
        Kred_wrong = asmB.Cu' * asmB.K * asmB.Cu
        S_wrong = [Kred_wrong asmM.Bt; asmM.Bs spzeros(size(asmM.Bs, 1),
                                                     size(asmM.Cp, 2))]
        rhs = vcat(asmM.fr, asmM.gr)
        solw = S_wrong \ rhs
        uw = asmM.u0 + asmM.Cu * solw[1:nf]
        rw = norm(S_wrong * solw - rhs) / max(norm(rhs), 1.0)
        duw = norm(uw - ud) / norm(ud)
        # the wrong system is solved to round-off (the failure is *silent*:
        # the residual looks converged) but the velocity is wrong
        @test rw < 1.0e-10
        @test duw > 1.0e-3
        # -- the wrong MG hierarchy: a box hierarchy is a much worse
        #    preconditioner for the mapped operator (the mapped solve would
        #    converge slowly / differently, not to the wrong answer, because the
        #    outer FGMRES uses the true mapped operator) ------------------------
        asmBf = _g271_sparse(ssB, qt, eta)
        mgbox = _g271_hierarchy(ssB, qt, p, eta, asmBf.Kred)
        mgmap = _g271_hierarchy(ssM, qt, p, eta, asmM.Kred)
        b = randn(nf)
        zbox = semquad_mg_apply(mgbox, b)
        zmap = semquad_mg_apply(mgmap, b)
        rbox = norm(b - asmM.Kred * zbox) / norm(b)
        rmap = norm(b - asmM.Kred * zmap) / norm(b)
        @test rbox > rmap
        # measured ~3x: the wrong-hierarchy preconditioner is a *real*, measured
        # degradation (a wrong-key hit on `SEMQUAD_MG_CACHE` would pay it on
        # every V-cycle of a mapped solve)
        @test rbox > 2.0 * rmap
        println("  dK=", dK, "  wrong-op residual=", rw,
                "  wrong-op |du|=", duw,
                "  hierarchy residual box=", rbox, " mapped=", rmap)
    end

    # ======================================================================
    # (e) the coarse levels are logical coarsenings; the operators are mapped
    # ======================================================================
    println("== G271 (e) mapped hierarchy coarse operators ==")
    let p = 2, eta = (x, y) -> exp(-log(20.0) * (1.0 - y))
        qt = _g271_mixed(3)
        ssbox = make_semstokes(qt, p; geom=CartesianBox(1.0))
        asmbox = _g271_sparse(ssbox, qt, eta)
        mgbox = _g271_hierarchy(ssbox, qt, p, eta, asmbox.Kred)
        for (gtag, g) in _G271_GEOMETRIES
            ss = make_semstokes(qt, p; geom=g)
            asm = _g271_sparse(ss, qt, eta)
            mg = _g271_hierarchy(ss, qt, p, eta, asm.Kred)
            @test mg.geo.geom === g
            # the coarse trees are the *box* hierarchy's trees: the coarsening
            # is logical (leaf index arithmetic) and must not see the geometry
            nlev = mg.geo.L + 1
            @test nlev == mgbox.geo.L + 1
            @test mg.geo.nfs == mgbox.geo.nfs
            nleafdiff = 0
            for k in 1:nlev
                @test Set(mg.geo.qts[k].leaves) == Set(mgbox.geo.qts[k].leaves)
                nleafdiff += length(mg.geo.qts[k].leaves)
            end
            # every mapped coarse level operator is the mapped sparse
            # rediscretisation on that coarse tree (not the box one)
            maxop = 0.0; maxdiag = 0.0; maxP = 0.0
            for k in 1:nlev
                q = mg.geo.qts[k]
                s = mg.geo.sss[k]
                asmk = _g271_sparse(s, q, eta)
                x = randn(mg.geo.nfs[k])
                yM = zeros(Float64, mg.geo.nfs[k])
                yS = zeros(Float64, mg.geo.nfs[k])
                _semquad_mg_apply!(yM, x, mg, k)
                mul!(yS, asmk.Kred, x)
                maxop = max(maxop, norm(yM - yS) / norm(yS))
                dk = diag(asmk.Kred)
                maxdiag = max(maxdiag, norm(mg.D[k] - dk) / norm(dk))
                # ... and the box rediscretisation on the same tree is a
                # *different* operator on a curved/affine geometry
                if k < nlev
                    rel = norm(mg.geo.Ps[k] - mgbox.geo.Ps[k]) /
                          norm(mgbox.geo.Ps[k])
                    maxP = max(maxP, rel)
                end
            end
            @test maxop < 1.0e-12
            @test maxdiag < 1.0e-12
            # the prolongation is the *logical* GLL interpolation, identical to
            # the box hierarchy's (a physical-coordinate prolongation would
            # differ on a curved map)
            @test maxP < 1.0e-13
            # the box hierarchy is not the mapped one: on this tree the box
            # coarse operator and the mapped coarse operator differ measurably
            k1 = 1
            asmk_box = _g271_sparse(make_semstokes(mgbox.geo.qts[k1], p;
                                                   geom=CartesianBox(1.0)),
                                    mgbox.geo.qts[k1], eta)
            asmk_map = _g271_sparse(mg.geo.sss[k1], mg.geo.qts[k1], eta)
            @test size(asmk_box.Kred) == size(asmk_map.Kred)
            dcoarse = norm(asmk_box.Kred - asmk_map.Kred) /
                      norm(asmk_map.Kred)
            @test dcoarse > 1.0e-6
            println("  ", rpad(gtag, 8), " leaves/level=", nleafdiff,
                    " op=", maxop, " diag=", maxdiag, " P=", maxP,
                    " coarse box-vs-mapped=", dcoarse)
        end
        # ---- on the box map the whole mapped hierarchy is the box one --------
        gmap = AffineBox((1.0, 0.0, 0.0, 1.0), (0.0, 0.0), qt.Lx)
        ssmap = make_semstokes(qt, p; geom=gmap)
        asmmap = _g271_sparse(ssmap, qt, eta)
        mgmap = _g271_hierarchy(ssmap, qt, p, eta, asmmap.Kred)
        # the historical box hierarchy (sparse operator: the reference)
        mgboxS = _g271_hierarchy(ssbox, qt, p, eta, asmbox.Kred;
                                 operator=:sparse)
        @test mgmap.geo.L + 1 == mgboxS.geo.L + 1
        maxop = 0.0; maxD = 0.0; maxP = 0.0
        for k in 1:(mgmap.geo.L + 1)
            @test Set(mgmap.geo.qts[k].leaves) == Set(mgboxS.geo.qts[k].leaves)
            x = randn(mgmap.geo.nfs[k])
            yM = zeros(Float64, mgmap.geo.nfs[k])
            yB = zeros(Float64, mgboxS.geo.nfs[k])
            _semquad_mg_apply!(yM, x, mgmap, k)
            mul!(yB, mgboxS.A[k], x)
            maxop = max(maxop, norm(yM - yB) / norm(yB))
            maxD = max(maxD, norm(mgmap.D[k] - mgboxS.D[k]) / norm(mgboxS.D[k]))
            if k < mgmap.geo.L + 1
                maxP = max(maxP, norm(mgmap.geo.Ps[k] - mgboxS.geo.Ps[k]) /
                                norm(mgboxS.geo.Ps[k]))
            end
        end
        @test maxop < 1.0e-13
        @test maxD < 1.0e-13
        @test maxP < 1.0e-13
        println("  box map: op=", maxop, " D=", maxD, " P=", maxP)
        # ---- threaded mapped V-cycle is bitwise the serial one ---------------
        let ssC = make_semstokes(qt, p; geom=_G271_GCUR),
            asmC = _g271_sparse(make_semstokes(qt, p; geom=_G271_GCUR), qt, eta)
            mg1 = _g271_hierarchy(ssC, qt, p, eta, asmC.Kred; mg_threads=1)
            old = QuaST.SEMQUAD_MG_SERIAL_DOFS[]
            try
                QuaST.SEMQUAD_MG_SERIAL_DOFS[] = 0
                mg8 = _g271_hierarchy(ssC, qt, p, eta, asmC.Kred;
                                      mg_threads=8)
                b = randn(mg1.geo.nfs[end])
                z1 = semquad_mg_apply(mg1, b)
                z8 = semquad_mg_apply(mg8, b)
                @test z1 == z8
                println("  threads=", Threads.nthreads(),
                        " serial==threaded bits=", z1 == z8,
                        " rel=", norm(z1 - z8) / norm(z1))
            finally
                QuaST.SEMQUAD_MG_SERIAL_DOFS[] = old
            end
        end
    end

    # ======================================================================
    # (f) the mapped driver's linear paths
    # ======================================================================
    println("== G271 (f) mapped driver linear paths ==")
    let g = AffineBox((1.10, 0.15, 0.0, 1.0), (0.0, 0.0), 1.0)
        pr = Params(Ra=1.0e3, H=0.0, bc=:rigid, eta_law=:expT, deta_T=1.0)
        qt = _g271_uniform(2)       # 16 leaves: a short but real time loop
        p = 2
        ss = make_semstokes(qt, p; geom=g)
        th0 = zeros(Float64, ss.su.N)
        dt = 1.0e-4
        for lin in (:direct, :mg, :auto)
            res = run_semquad_simulation(qt, pr, th0; geom=g, p=p,
                                         tstop=2 * dt, dt0=dt, dtmax=dt,
                                         dtmin=1.0e-16, linear=lin,
                                         mg_operator=:matrixfree,
                                         maxsteps=2, amr_every=0,
                                         initial_refine=0, symmetry=:none,
                                         maxpic=1, verbose=false)
            @test res.steps == 2
            @test all(isfinite, res.theta)
            println("  driver linear=", lin, " steps=", res.steps,
                    " theta_max=", maximum(abs, res.theta))
        end
        # the mapped auto choice is recorded and is one of the supported paths
        for lin in (:direct, :mg)
            solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0, _G271_Z2,
                                     _G271_Z2, _G271_Z2, _G271_Z2;
                                     bc=:rigid, linear=lin,
                                     mg_operator=:matrixfree)
        end
        @test solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0, _G271_Z2,
                                       _G271_Z2, _G271_Z2, _G271_Z2;
                                       bc=:rigid, linear=:auto) isa Tuple
        @test QuaST.LAST_SEMSTOKES_AUTO_CHOICE[].choice in (:direct, :mg)
        # `:schur_cg` / `:lagged` stay refused for a mapped space, with the
        # actionable message (not a silent box-body run)
        for lin in (:schur_cg, :lagged)
            err = try
                solve_semquad_stokes_var(ss, qt, (x, y) -> 1.0, _G271_Z2,
                                         _G271_Z2, _G271_Z2, _G271_Z2;
                                         bc=:rigid, linear=lin)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            msg = sprint(showerror, err)
            @test occursin("mapped geometry", msg)
            @test occursin("box", msg)
        end
        # mapped `linear=:auto` must be able to resolve to the mapped MG: above
        # the selector's crossover (2*Nu+Np >= 2000, contrast 1) it picks :mg,
        # builds the mapped hierarchy through the auto path and solves.  This is
        # the design's cut-5 target ("linear=:auto/:mg work mapped").
        let qt2 = _g271_uniform(4), g2 = _G271_GAFF
            ss2 = make_semstokes(qt2, p; geom=g2)
            eta2 = (x, y) -> 1.0
            fx2 = (x, y) -> sinpi(x) * sinpi(y)
            fz2 = (x, y) -> 0.3 * cospi(x) * sinpi(y)
            dofs = 2 * ss2.su.N + ss2.sp.N
            @test dofs >= 2000          # the selector's contrast-100 crossover
            ua, pa = solve_semquad_stokes_var(ss2, qt2, eta2, fx2, fz2,
                                              _G271_Z2, _G271_Z2; bc=:rigid,
                                              linear=:auto)
            ch = QuaST.LAST_SEMSTOKES_AUTO_CHOICE[]
            @test ch.choice === :mg
            @test ch.saddle_dofs == dofs
            @test all(isfinite, ua)
            @test all(isfinite, pa)
            ud2, _ = solve_semquad_stokes_var(ss2, qt2, eta2, fx2, fz2,
                                              _G271_Z2, _G271_Z2; bc=:rigid,
                                              linear=:direct)
            da = norm(ua - ud2) / norm(ud2)
            @test da < 1.0e-6
            println("  mapped auto choice=", ch.choice, " saddle_dofs=",
                    ch.saddle_dofs, " contrast=", ch.contrast,
                    " |du|(auto vs direct)=", da)
        end
        # the mapped driver refuses them before the setup work
        for lin in (:schur_cg, :lagged)
            @test_throws ArgumentError run_semquad_simulation(
                qt, pr, th0; geom=g, p=p, tstop=dt, dt0=dt, dtmax=dt,
                linear=lin, maxsteps=1, amr_every=0, initial_refine=0,
                symmetry=:none, maxpic=1, verbose=false)
        end
    end
end
