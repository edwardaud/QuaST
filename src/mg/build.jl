# Build the geometric multigrid hierarchy: coarsen the tree, assemble each level and factor the coarsest.
function _semquad_mg_geometry(ss::SemStokes, qt::Quadtree, p::Int,
                              bc::Symbol, deviatoric::Bool;
                              wall_bc::NTuple{4,Symbol}=(:same, :same, :same, :same),
                              coarsen_layers::Int=1,
                              coarsen_factor::Float64=1.0,
                              coarse_direct_max::Int=0,
                              fine_Cu::Union{Nothing,SparseMatrixCSC{Float64,Int}}=nothing,
                              fine_free::Union{Nothing,Vector{Int}}=nothing,
                              fine_ss::Union{Nothing,SemStokes}=nothing,
                              geom_assembly::Symbol=:minimal)
    coarse_direct_max >= 0 ||
        throw(ArgumentError("MG coarse_direct_max must be >= 0"))
    geom_assembly in (:minimal, :full) ||
        throw(ArgumentError("MG geom_assembly must be :minimal or :full"))
    # Cut 5: the geometry identity enters the hierarchy cache key.  A box and a
    # mapped geometry on the same tree must never share a hierarchy: the levels
    # store geometry-carrying spaces and their operators are built from the
    # mapped assembly (GEOM_DESIGN §2.5, risk 5).  Stage 0.5c1 adds the tree's
    # `seam`: a ring hierarchy and a bounded hierarchy on the same leaf set are
    # different topologies (cross-seam constraints/prolongation).
    gstamp = geometry_stamp(ss.su.geom)
    key = (p, qt.nx0, qt.ny0, qt.Lx, bc, deviatoric, wall_bc,
           length(qt.leaves), hash(qt.leaves), gstamp, qt.seam,
           coarsen_layers, coarsen_factor, coarse_direct_max, geom_assembly)
    if haskey(SEMQUAD_MG_CACHE, key)
        cached = SEMQUAD_MG_CACHE[key]::SemQuadMGGeom
        Set(cached.qts[end].leaves) == Set(qt.leaves) && return cached
    end
    reuse = SEMQUAD_MG_LEVEL_CACHE[]
    stats = SEMQUAD_MG_LEVEL_STATS
    stats.builds += 1
    empty!(stats.last_rebuilt)
    empty!(stats.last_reused)
    trees = _mg_hierarchy_trees(qt; layers=coarsen_layers,
                                factor=coarsen_factor)
    nlev = length(trees)
    stamps = UInt64[_semquad_mg_leaf_hash(q.leaves) for q in trees]
    levels = Vector{SemQuadMGLevel}(undef, nlev)
    for idx in 1:nlev
        q = trees[idx]
        # The finest level is the caller's tree: its constraint matrix `Cu` is
        # already available from the caller's assembly and does not depend on
        # eta, so re-assembling it here with eta=1 is pure duplicate work
        # (0.17-0.41 s of the hierarchy build on the production tree).  It is
        # **never cached**: the caller supplies the tree (the driver mutates it
        # in place on the next AMR cycle) and its `Cu` on every build.  Every
        # *coarse* level is a private tree built by `_mg_coarsen_tree`, so it is
        # cacheable.  The fallback keeps standalone/reference callers bitwise
        # unchanged.
        cacheable = idx < nlev
        nleaf = length(q.leaves)
        lv = nothing
        if cacheable && reuse
            lkey = SemQuadMGLevelKey(idx, nleaf, stamps[idx], p, q.nx0, q.ny0,
                                     q.Lx, bc, deviatoric, wall_bc, gstamp,
                                     q.seam)
            got = get(SEMQUAD_MG_CACHE, lkey, nothing)
            if got isa SemQuadMGLevel
                lv = got
                stats.hits += 1
                push!(stats.last_reused, idx)
            end
        end
        if lv === nothing
            fine = idx == nlev && fine_Cu !== nothing
            s = fine && fine_ss !== nothing ? fine_ss :
                make_semstokes(q, p; geom=ss.su.geom)
            # The free-row list is taken from the assembly that produced Cu when
            # the caller supplies it (stage 0.5c1: a ring free-slip row can have
            # the same sparse signature as a free row, so recovering the list
            # from Cu alone is not reliable -- see _semquad_mg_level_Cu).  The
            # _mg_free_rows fallback keeps standalone callers working.
            Cu, free = if fine
                size(fine_Cu, 1) == 2 * s.su.N ||
                    throw(ArgumentError("fine_Cu has $(size(fine_Cu, 1)) rows, " *
                                        "expected $(2 * s.su.N); it does not belong " *
                                        "to the finest MG tree"))
                (fine_Cu, fine_free === nothing ?
                          _mg_free_rows(fine_Cu, 2 * s.su.N) : fine_free)
            else
                _semquad_mg_level_Cu(q, s, bc, deviatoric, geom_assembly;
                                     wall_bc=wall_bc)
            end
            lv = SemQuadMGLevel(idx, nleaf, stamps[idx], q, s, Cu,
                                size(Cu, 2), free,
                                nothing, (0, UInt64(0), 0, UInt64(0)))
            if cacheable
                stats.misses += 1
                push!(stats.last_rebuilt, idx)
                reuse && _semquad_mg_level_cache_put!(
                    SemQuadMGLevelKey(idx, nleaf, stamps[idx], p, q.nx0, q.ny0,
                                      q.Lx, bc, deviatoric, wall_bc, gstamp,
                                      q.seam), lv)
            end
        end
        levels[idx] = lv
    end
    # S3 stage 1 (b): stop the V-cycle at the level where a direct factorisation
    # is cheap enough, instead of coarsening further.  `nf` is ascending (coarse
    # first), so this keeps the *finer* part of the hierarchy and factorises its
    # new coarsest level; the finest level is never cut away (that would leave
    # no coarse-grid correction at all).
    coarse_direct = false
    kcut = 1
    if coarse_direct_max > 0 && nlev > 2
        for k in 1:(nlev - 1)
            levels[k].nf <= coarse_direct_max || break
            kcut = k
        end
        kcut > 1 && (coarse_direct = true)
    end
    kept = kcut:nlev
    L = length(kept) - 1
    Ps = SparseMatrixCSC{Float64,Int}[]
    for k in kcut:(nlev - 1)
        lv = levels[k]
        f = levels[k + 1]
        pstamp = (lv.nleaf, lv.stamp, f.nleaf, f.stamp)
        if reuse && lv.P !== nothing && lv.pstamp == pstamp
            push!(Ps, lv.P)
        else
            Pfull = _mg_prolongation_full(f.ss.su, lv.ss.su, p)
            P = Pfull[f.free_rows, :] * lv.Cu
            size(P, 1) == f.nf && size(P, 2) == lv.nf ||
                error("MG prolongation has the wrong size: $(size(P)) for " *
                      "$(f.nf) x $(lv.nf)")
            P = SparseMatrixCSC{Float64,Int}(sparse(P))
            push!(Ps, P)
            lv.P = P
            lv.pstamp = pstamp
            stats.prolongations += 1
        end
    end
    qts = Quadtree[levels[k].qt for k in kept]
    sss = SemStokes[levels[k].ss for k in kept]
    Cus = SparseMatrixCSC{Float64,Int}[levels[k].Cu for k in kept]
    # Transposed constraint matrices: `Cu * v` is a row gather over `Cu'`, the
    # split form of the first sparse product (`_semquad_spmv_region!`).  The
    # transpose is a pure re-indexing (no arithmetic) and costs <1 ms per level
    # (2.4 MB in total on the production F hierarchy), so it is part of the
    # geometry instead of the threaded-only state: every `SemQuadMG` -- including
    # the ones the gates build by hand -- shares it.
    Cuts = SparseMatrixCSC{Float64,Int}[SparseMatrixCSC{Float64,Int}(
                                          sparse(transpose(Cu))) for Cu in Cus]
    nfs = Int[levels[k].nf for k in kept]
    free_rows = Vector{Int}[levels[k].free_rows for k in kept]
    geo = SemQuadMGGeom(p, L, qt.Lx, qt.nx0, qt.ny0, bc, deviatoric,
                        ss.su.geom, qts, sss, Cus, Cuts, Ps, nfs, free_rows,
                        coarse_direct)
    SEMQUAD_MG_CACHE[key] = geo
    return geo
end

"""
    make_semquad_mg(ss, qt, p, bc, deviatoric, eta_fun, Kred; cycles=1,
                    smoother=:jacobi, coarse_operator=:rediscretize,
                    operator=:sparse, coarsen_layers=1, coarsen_factor=1.0,
                    coarse_direct_max=0, pre=2, post=2, smoothing=:uniform,
                    omega=2/3)

Build the MG hierarchy for the Stokes velocity block on `qt`, uniform or not.
The fine level reuses the caller's `Kred`; the coarse levels are re-assembled
with the same `eta_fun` (rediscretisation, standard for smooth variable
viscosity; hanging constraints are reassembled on every coarse tree).

`coarsen_layers`/`coarsen_factor` control the hierarchy (see
`_mg_coarsen_tree`): `1/1.0` is the historical one-tree-layer-per-step rule,
`coarsen_factor=1.5` (the production matrix-free default, resolved by
`solve_semquad_stokes_var`) removes a near-duplicate finest step while keeping a
7-level P1 hierarchy, `2.0` merges more aggressively (F-only win; P1 regresses),
and `coarse_direct_max=n>0` stops the V-cycle at the finest level with at most
`n` dofs and solves that level with a sparse Cholesky instead of coarsening
further (`geo.coarse_direct`; the matrix-free path assembles that one level
sparsely, because a dense `O(nf^2)` materialisation would cost far more than it
saves).

`pre`/`post` are the Jacobi sweeps per level (an `Int`, or one entry per level
**coarse first**); `smoothing=:auto` (default) uses uniform 2+2, raised to 3+3
when the hierarchy is top-heavy (its finest step keeps >50 % of the fine dofs,
the measured winner on P1), and `omega` (damping; `nothing` = the schedule's
default) overrides the damping factor.  `smoothing=:taper` remains selectable
(`1, 2, 2, 4, 4, 8, ...` from the fine level down, omega=0.55); explicit
`pre`/`post` vectors are accepted with any schedule.

`fine_Cu` (optional) is the caller's full->free constraint matrix of the finest
tree.  `_semquad_mg_geometry` otherwise re-assembles the finest level with eta=1
only to read that matrix; passing it removes the duplicate assembly (the
0.17-0.41 s finest row of docs/dev/MG.md) and is bitwise inert (G219).

`fine_mfop`/`fine_mfeta` (optional, `operator=:matrixfree` only) are the
caller's finest-level leaf-local operator (`semquad_kop(ss, qt)`) and its
viscosity field (`semquad_eta_field`).  The solve builds both before it asks for
the hierarchy, so without them `make_semquad_mg` resamples the same 131 008
quadrature points a second time (0.17 s of the F build at one thread); passing
them is bitwise inert (G220) and only removes the duplicate sampling.

Every level's viscosity field is stored as a `SemQuadEtaLane`: the leaf-major
sample plus the block-major lane table the leaf-blocked kernel reads
(docs/dev/MG.md).  The table is built here, once per level, never per
apply; it is a re-layout, so it changes no value (G225).

`geom_assembly=:minimal` (default) builds the coarse levels' `Cu` with the
reduced matrix-free assembly (`assemble_only=:mg`): the hierarchy reads only
`Cu`, so no velocity stiffness `K`, no `Kred` and no viscosity evaluation are
needed there.  `:full` keeps the historical `K`-forming assembly (0.039 s of the
F build at one thread) as the A/B and bitwise-gate reference; the two produce the
same `Cu` and therefore the same hierarchy, V-cycle and solve (G220).

`operator=:matrixfree` (S3/MF-2/MF-3) replaces every level's sparse matvec by
the matrix-free leaf-local apply `semquad_apply_Kred!` of MF-1, and stores no
velocity operator matrix at all: `A` is empty, the Jacobi diagonal comes from
the leaf-local element diagonals, and the coarsest solve comes from a dense
matrix materialised by applying the matrix-free operator to the unit vectors.
The hierarchy then keeps only the sparse `Cu`/`P` transfer operators.  The
viscosity is sampled once per level at the assembler's Gauss points, so each
matrix-free level operator is the same operator as the rediscretised sparse one
(G175) and the smoother is unchanged on conforming trees.  `Kred` is accepted
and ignored in that mode (it is the caller's outer-Krylov operator).
`operator=:matrixfree` always coarsens by rediscretisation.

`mg_diag` selects the matrix-free Jacobi diagonal: `:constrained` (default) is
the exact `diag(Cu' K Cu)`, computed leaf-locally without forming `K`;
`:lumped` is the historical `Cu' diag(K)` (only for A/B -- on hanging rows the
negative interpolation weights cancel, so it is not the reduced diagonal and its
positivity clip can make a row a `1e12` smoother amplifier; see
docs/dev/MG.md §5.2).  The sparse operator path always uses
`diag(A[k])` and is unaffected.

`mg_threads=1` (default) is the serial, bitwise V-cycle.  `mg_threads=n>1`
(S5-2) threads the leaf-local work of every level -- the matrix-free apply, the
damped-Jacobi sweeps and the residual evaluation -- over `n` fixed contiguous
leaf blocks (`Threads.@threads :static`), with one `SemQuadKScratch` per block
and a per-leaf element buffer merged **serially in leaf order**, so the result
is bitwise identical to the serial V-cycle (G190; measured 2.5-3.1x V-cycle
speedup on 4096-leaf trees, docs/dev/SOLVER.md §19.2).  Only the matrix-free
operator path is threaded (the sparse `A[k]` path keeps its serial `mul!`); a
request larger than `Threads.nthreads()` is clamped with a warning, and a
single-threaded process warns once and stays serial.
"""
function make_semquad_mg(ss::SemStokes, qt::Quadtree, p::Int, bc::Symbol,
                         deviatoric::Bool, eta_fun::Function,
                         Kred::Union{Nothing,AbstractMatrix}; cycles::Int=1,
                         smoother::Symbol=:jacobi,
                         coarse_operator::Symbol=:rediscretize,
                         operator::Symbol=:sparse,
                         coarsen_layers::Int=1,
                         coarsen_factor::Float64=1.0,
                         coarse_direct_max::Int=0,
                         pre::Union{Integer,AbstractVector{<:Integer}}=2,
                         post::Union{Integer,AbstractVector{<:Integer}}=2,
                         smoothing::Symbol=:auto,
                         omega::Union{Nothing,Float64}=nothing,
                         mg_threads::Int=1,
                         mg_diag::Symbol=:constrained,
                         fine_Cu::Union{Nothing,SparseMatrixCSC{Float64,Int}}=nothing,
                         fine_free::Union{Nothing,Vector{Int}}=nothing,
                         geom_assembly::Symbol=:minimal,
                         fine_mfop::Union{Nothing,AnySemQuadKOp}=nothing,
                         fine_mfeta::Union{Nothing,AbstractVector{Float64}}=nothing,
                         wall_bc::NTuple{4,Symbol}=(:same, :same, :same, :same))
    smoother in (:jacobi, :sgs) ||
        throw(ArgumentError("MG smoother must be :jacobi or :sgs"))
    coarse_operator in (:rediscretize, :galerkin) ||
        throw(ArgumentError("MG coarse_operator must be :rediscretize or :galerkin"))
    operator in (:sparse, :matrixfree) ||
        throw(ArgumentError("MG operator must be :sparse or :matrixfree"))
    coarsen_layers >= 1 ||
        throw(ArgumentError("MG coarsen_layers must be >= 1"))
    (coarsen_factor >= 1.0 && isfinite(coarsen_factor)) ||
        throw(ArgumentError("MG coarsen_factor must be >= 1"))
    coarse_direct_max >= 0 ||
        throw(ArgumentError("MG coarse_direct_max must be >= 0"))
    smoothing in (:auto, :uniform, :taper) ||
        throw(ArgumentError("MG smoothing must be :auto, :uniform or :taper"))
    mg_threads >= 1 || throw(ArgumentError("mg_threads must be >= 1"))
    mg_diag in (:constrained, :lumped) ||
        throw(ArgumentError("mg_diag must be :constrained or :lumped"))
    if operator === :matrixfree
        smoother === :sgs &&
            throw(ArgumentError("the :sgs smoother needs operator=:sparse"))
        coarse_operator === :galerkin &&
            @warn "operator=:matrixfree coarsens by rediscretisation; " *
                  "ignoring coarse_operator=:galerkin"
        coarse_operator = :rediscretize
    end
    geom_assembly in (:minimal, :full) ||
        throw(ArgumentError("MG geom_assembly must be :minimal or :full"))
    operator === :sparse && (fine_mfop !== nothing || fine_mfeta !== nothing) &&
        throw(ArgumentError("fine_mfop/fine_mfeta need operator=:matrixfree"))
    _t_build0 = time_ns()
    geo = _semquad_mg_geometry(ss, qt, p, bc, deviatoric; wall_bc=wall_bc,
                               coarsen_layers=coarsen_layers,
                               coarsen_factor=coarsen_factor,
                               coarse_direct_max=coarse_direct_max,
                               fine_Cu=fine_Cu, fine_free=fine_free,
                               fine_ss=ss, geom_assembly=geom_assembly)
    L = geo.L
    mfops = AnySemQuadKOp[]
    mfetas = SemQuadEtaLane[]
    mfwork = Any[]
    A = Vector{SparseMatrixCSC{Float64,Int}}()
    D = Vector{Vector{Float64}}()
    cA = nothing
    if operator === :matrixfree
        # S3/MF-3: the hierarchy stores *no* velocity operator matrix.  Every
        # level applies MF-1's leaf-local operator; the Jacobi diagonal is the
        # *constrained* diagonal `diag(Cu' K Cu)` (`mg_diag=:constrained`, the
        # default; `:lumped` keeps the old `Cu' diag(K)` for A/B), and the
        # coarsest solve comes from a dense matrix materialised by applying the
        # matrix-free operator to the unit vectors
        # (`_semquad_dense_operator!`), which is O(nf^2) applies on the base
        # leaves only.
        for k in 1:(L + 1)
            fine_k = k == L + 1
            if fine_k && fine_mfop !== nothing
                # The caller (the solve) has already built the finest level's
                # leaf-local operator; `semquad_kop` is cached on the tree, so
                # this is the same object.  Rebuilding it here would be cheap
                # but recomputing its eta field is not: the caller's
                # `semquad_eta_field` sampled the identical 131 008 points and
                # the hierarchy used to do it a second time (0.17 s of the F
                # build at one thread, docs/dev/MG.md).
                opk = fine_mfop
                (opk.N == geo.sss[k].su.N && opk.p == geo.sss[k].su.p &&
                 opk.q == geo.sss[k].sp.p && opk.deviatoric == deviatoric &&
                 opk.nleaf == length(geo.qts[k].leaves)) ||
                    throw(ArgumentError("fine_mfop does not belong to the " *
                                        "finest MG tree (N=$(opk.N), p=$(opk.p), " *
                                        "nleaf=$(opk.nleaf)); it must be " *
                                        "semquad_kop(ss, qt)"))
            else
                opk = semquad_kop(geo.sss[k], geo.qts[k]; deviatoric=deviatoric)
            end
            push!(mfops, opk)
            if fine_k && fine_mfeta !== nothing
                length(fine_mfeta) == length(opk.xq) ||
                    throw(ArgumentError("fine_mfeta has $(length(fine_mfeta)) " *
                                        "values, expected one per leaf/quadrature " *
                                        "point ($(length(opk.xq)))"))
                push!(mfetas, semquad_eta_lanes(opk, fine_mfeta))
            else
                push!(mfetas, semquad_eta_lanes(opk,
                                                semquad_eta_field(opk, eta_fun)))
            end
            push!(mfwork, SemQuadKWork(opk))
        end
        for k in 1:(L + 1)
            push!(D, _semquad_diag_reduced(mfops[k]::AnySemQuadKOp, geo.Cus[k],
                                           mfetas[k]::SemQuadEtaLane;
                                           mode=mg_diag))
        end
        if geo.coarse_direct
            # Stage-1 (b) cut: the coarsest level is a real coarse mesh, not the
            # base leaves, so materialising a dense operator would cost O(nf^2)
            # applies; assemble that level's sparse rediscretised operator and
            # factorise it instead.
            asm1 = solve_semquad_stokes_var(geo.sss[1], geo.qts[1], eta_fun,
                                            (x, y) -> 0.0, (x, y) -> 0.0,
                                            (x, y) -> 0.0, (x, y) -> 0.0;
                                            bc=bc, deviatoric=deviatoric,
                                            assemble_only=true)
            A1 = SparseMatrixCSC{Float64,Int}(sparse(asm1.Kred))
            cA = cholesky(Symmetric(A1); check=false)
            issuccess(cA) ||
                throw(ArgumentError("coarsest matrix-free velocity operator is not SPD"))
        else
            n1 = geo.nfs[1]
            A1 = Matrix{Float64}(undef, n1, n1)
            _semquad_dense_operator!(A1, mfops[1]::AnySemQuadKOp, geo.Cus[1],
                                     mfetas[1]::SemQuadEtaLane,
                                     mfwork[1]::SemQuadKWork)
            cA = cholesky(Symmetric(A1); check=false)
            issuccess(cA) ||
                throw(ArgumentError("coarsest matrix-free velocity operator is not SPD"))
        end
    else
        A = Vector{SparseMatrixCSC{Float64,Int}}(undef, L + 1)
        A[L + 1] = SparseMatrixCSC{Float64,Int}(sparse(Kred))
        if coarse_operator === :galerkin
            # P'AP from the fine operator; exact Galerkin coarsening.  For
            # constant eta it equals the rediscretised operator (to round-off);
            # for strong variable viscosity it can be better conditioned.
            for k in L:-1:1
                A[k] = SparseMatrixCSC{Float64,Int}(sparse(geo.Ps[k]' *
                                                           A[k + 1] * geo.Ps[k]))
            end
        else
            for k in 1:L
                asm = solve_semquad_stokes_var(geo.sss[k], geo.qts[k], eta_fun,
                                               (x, y) -> 0.0, (x, y) -> 0.0,
                                               (x, y) -> 0.0, (x, y) -> 0.0;
                                               bc=bc, deviatoric=deviatoric,
                                               assemble_only=true)
                A[k] = SparseMatrixCSC{Float64,Int}(sparse(asm.Kred))
            end
        end
        for k in 1:L
            size(A[k], 1) == geo.nfs[k] ||
                error("coarse MG operator has $(size(A[k],1)) rows, expected $(geo.nfs[k])")
        end
        size(A[L + 1], 1) == geo.nfs[L + 1] ||
            error("fine MG operator has $(size(A[L + 1],1)) rows, expected $(geo.nfs[L + 1])")
        D = [Vector{Float64}(diag(A[k])) for k in 1:(L + 1)]
        cA = cholesky(Symmetric(A[1]))
    end
    any(x -> x <= 0.0, D[L + 1]) &&
        throw(ArgumentError("velocity block has a non-positive diagonal; " *
                            "MG requires an SPD reduced operator"))
    LF = Any[]; UF = Any[]
    if smoother === :sgs
        for k in 1:(L + 1)
            push!(LF, LowerTriangular(A[k]))
            push!(UF, UpperTriangular(A[k]))
        end
    end
    sc = [zeros(Float64, geo.nfs[k]) for k in 1:(L + 1)]
    sr = [zeros(Float64, geo.nfs[k]) for k in 1:(L + 1)]
    sy = [zeros(Float64, geo.nfs[k]) for k in 1:(L + 1)]
    sched, om, auto_heavy = _mg_smoothing_profile(smoothing, geo, omega)
    LAST_SEMSTOKES_MG_PROFILE[] = (sched, om)
    pre_v, post_v = if sched === :uniform || pre isa AbstractVector ||
                       post isa AbstractVector
        # Explicit per-level vectors are accepted with any resolved schedule
        # (they are the `:taper` measurement knob: `smoothing=:taper,
        # pre=[...], post=[...]` keeps omega=0.55 and uses the given shape).
        # A top-heavy `:auto` hierarchy raises the *default* 2+2 to 3+3; an
        # explicit non-default sweep count is respected.
        p = (smoothing === :auto && auto_heavy && pre isa Integer && pre == 2) ?
            3 : pre
        q = (smoothing === :auto && auto_heavy && post isa Integer && post == 2) ?
            3 : post
        (_mg_sweeps(p, L + 1, "pre"), _mg_sweeps(q, L + 1, "post"))
    else
        _mg_smoothing_schedule(sched, L + 1, Int(pre), Int(post))
    end
    # S5-2: one SemQuadKPar per matrix-free level.  The sparse operator path
    # keeps its serial `mul!` (`par` is empty), so `mg_threads` only ever
    # changes the matrix-free hierarchy.
    nt = _semquad_mg_thread_count(mg_threads)
    par = Any[]
    Pts = Any[]
    if nt > 1 && operator === :matrixfree
        par = Any[SemQuadKPar(mfops[k]::AnySemQuadKOp, nt) for k in 1:(L + 1)]
        # Transposed transfer operators: the prolongation is a row gather over
        # `P'`'s CSC (`_semquad_mg_prolong!`), which is bitwise the `mul!`
        # scatter but parallelisable over rows.
        Pts = Any[SparseMatrixCSC{Float64,Int}(sparse(transpose(geo.Ps[k])))
                  for k in 1:L]
    elseif nt > 1
        nt = 1
    end
    # Per-level V-cycle scratch (restriction output, coarse error, prolongation
    # output).  Fully overwritten on every use, so this is bitwise the
    # allocating `P' * r` / `zeros` / `P * ec` form (and takes the per-V-cycle
    # allocations, ~2 MB at L6, out of the GC).
    rcs = Vector{Vector{Float64}}(undef, L + 1)
    ecs = Vector{Vector{Float64}}(undef, L + 1)
    pys = Vector{Vector{Float64}}(undef, L + 1)
    rcs[1] = zeros(Float64, 0); ecs[1] = zeros(Float64, 0)
    pys[1] = zeros(Float64, 0)
    for k in 2:(L + 1)
        rcs[k] = zeros(Float64, geo.nfs[k - 1])
        ecs[k] = zeros(Float64, geo.nfs[k - 1])
        pys[k] = zeros(Float64, geo.nfs[k])
    end
    _SEMT = SEMQUAD_MG_LEVEL_STATS
    _SEMT.last_build_s = (time_ns() - _t_build0) / 1.0e9
    _SEMT.total_build_s += _SEMT.last_build_s
    return SemQuadMG(geo, A, D, cA, cycles, pre_v, post_v, om, smoother,
                     LF, UF, operator, mfops, mfetas, mfwork, sc, sr, sy,
                     nt, par, Pts, rcs, ecs, pys)
end

