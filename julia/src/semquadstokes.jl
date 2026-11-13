# Quadtree Taylor-Hood Stokes (stage A-AMR, step 2).
#
# Velocity Q_p and pressure Q_{p-1} on the quadtree leaves, with the same
# hanging-node constraints as the scalar Poisson (semquad.jl).  The element
# matrices are the constant-viscosity (eta = 1) stress form
#   2 eps(u):eps(v) - p div(v) - q div(u) = f.v,
# assembled on the reference square (the h factors cancel in the stiffness and
# enter the divergence as h/2), with velocity Dirichlet values imposed through
# an affine constraint u = u0 + C u_free (u0 carries the boundary values and the
# fixed part of any hanging constraint).  One pressure dof is pinned to fix the
# constant null space.
#
# Gate: the polynomial field u = (2xy, -y^2), p = 0 is divergence free and
# satisfies -div(2 eps(u)) = (0, 2), so with f = (0, 2) and Q_2/Q_1 the discrete
# solution must reproduce it exactly on any (mixed) quadtree.

struct SemStokes
    p::Int
    q::Int
    su::SemQuad
    sp::SemQuad
end

# Diagnostics for the last quadtree Stokes solve: relative reduced-saddle
# residual and the number of Schur-CG iterations (0 for a direct solve).
const LAST_SEMSTOKES_RESIDUAL = Ref(0.0)
const LAST_SEMSTOKES_SCHUR_ITERS = Ref(0)
# Number of inner CG iterations spent in the velocity block of the `linear=:mg`
# solve (MF-4); 0 for every other path.
const LAST_SEMSTOKES_VEL_CG_ITERS = Ref(0)
# True when the direct path had to retry with the 1e-12 relative diagonal
# regularization (wall trees with free-slip; see the direct branch below).
const LAST_SEMSTOKES_REGULARIZED = Ref(false)

# --- the direct path's shift ladder (docs/dev/CASE_REALISTIC.md §13) -------------
#
# The two blocks of the reduced saddle are measured separately: `vel` is
# `||Kred u + Bt p - fr|| / rden` (momentum rows) and `div` is
# `||Bs u - gr|| / rden` (continuity rows).  `LAST_SEMSTOKES_RESIDUAL` is their
# max, so this only says *which* block limits a marginal solve.
const LAST_SEMSTOKES_RESID_SPLIT = Ref((vel=0.0, div=0.0))
# Relative diagonal shift of the factorisation behind the returned solution
# (0.0 = the unshifted LU, 1e-12/1e-10/1e-8 = a ladder rung; when the returned
# iterate comes from the refinement, the shift of the factorisation it was
# preconditioned by).  Reset by every solve, so it describes the last solve only.
const LAST_SEMSTOKES_SHIFT = Ref(0.0)
# Iterative-refinement iterations (GMRES with the last factorisation as
# preconditioner) spent after the ladder missed; 0 when none were needed.
const LAST_SEMSTOKES_REFINE_ITERS = Ref(0)
# Every rung the last direct solve tried, in order, as `(shift, residual)`.
# `nothing` unless the ladder fired (so a healthy solve carries no allocation).
const LAST_SEMSTOKES_LADDER = Ref{Any}(nothing)
# Which phase of the ladder produced the last direct answer (docs/
# CASE_REALISTIC.md §17.4, G212): `:healthy` (the unshifted LU passed),
# `:tiny` (the priority tiny-shift phase of rule 2, for a *lost* unshifted
# solve), `:ladder` (the 1e-12/1e-10/1e-8 rungs, including the marginal but
# accepted unshifted answer), `:refine` (the iterative refinement improved the
# best), `:fallback` (the last-resort tiny shifts of rule 7 won) or `:fail`
# (every rung, the refinement and -- when it ran -- the last-resort phase missed
# `accept_tol`; the caller's error is due).  `nothing` unless the last solve took
# the direct path; reset with `LAST_SEMSTOKES_LADDER`.
const LAST_SEMSTOKES_LADDER_PHASE = Ref{Any}(nothing)
# Cumulative number of direct solves whose answer came from the equilibrated
# path (`stokes_scaling = :equilibrated` engaged on a solve that failed the
# unscaled health check and its scaled residual was accepted).  Unlike the
# per-solve records above this one is a counter, not reset by a solve: it
# counts the engages of a whole run (`LAST_SEMSTOKES_SCALED[] = 0` restarts it).
const LAST_SEMSTOKES_SCALED = Ref(0)
# Cumulative number of times the **assembled velocity stiffness** `K` (the
# `2N x 2N` sparse operator of the leaf assembly) has been formed since the
# process started.  It is a counter, not a per-solve record: a caller reads it
# before and after a solve to prove that a path did not assemble `K` at all.
# The matrix-free `linear=:mg, mg_operator=:matrixfree` path leaves it
# unchanged (S3 stage C: `K` and `Kred` are `nothing`); the direct path and the
# sparse MG geometry increment it.  `assemble_only=:mg` (the reduced assembly)
# increments it zero times.
const SEMSTOKES_K_ASSEMBLIES = Ref(0)

# --- S6: production path selector (`linear=:auto`) ---------------------------
#
# `linear=:auto` chooses between the sparse direct LU and the matrix-free MG
# from two properties of the *current* saddle: its size and the viscosity
# contrast of the eta field actually passed to the solve.  The constants below
# are calibrated against a same-session A/B of the two paths on the states the
# previous table named, re-measured at `c7e0f55` after the optimisation
# campaign of docs/dev/MG.md§16 (that campaign's endpoint is a 2.3-2.6x
# faster warm MG solve than the one the previous table -- docs/S3_DESIGN.md
# §18.3/§19.2 -- was calibrated on, while the direct path is untouched).  The
# A/B and every row: `julia/scripts/bench_auto_crossover.jl`, one process per
# state, both paths on the same tree and closures, `BLAS=1`, direct
# `linear=:direct, stokes_scaling=:equilibrated` (the tolerance-reaching
# reference), MG `linear=:mg` with the shipped defaults and
# `mg_tol = 1e-6` (the direct reference's accuracy), warm = best of three
# in-tree solves after a cold build.  Ratio = t_direct / t_mg, so >1 means MG
# faster (the convention of docs/dev/SOLVER.md §4.2/§4.4, which carries the same
# table plus the cold column):
#
# | state                                 | saddle dofs | contrast | 1thr | 8thr |
# |---------------------------------------|-------------|----------|------|------|
# | uniform L4  (256 leaves, p=4)          |      10 851 |        1 | 3.86 | 3.94 |
# | uniform L5  (1 024 leaves, p=4)        |      42 691 |        1 | 7.36 | 7.21 |
# | uniform L6  (4 096 leaves, p=4)        |     169 347 |        1 | 6.97 | 7.59 |
# | uniform L4 at p=2 (256 leaves)         |       2 467 |        1 | 1.91 | 1.81 |
# | uniform L4, stratified eta (p=4)       |      10 851 |       50 | 4.04 | 3.48 |
# | blob f19 (1 024 leaves)                |      45 475 |    5 087 | 1.78 | 1.55 |
# | blob f19x4 (2 344 leaves)              |      99 559 |    5 087 | 2.41 | 2.44 |
# | blob f19x16 (4 096 leaves)             |     174 915 |    5 087 | 2.62 | 2.77 |
# | blob 25-frame trajectory (frame 1)     |       9 351 |    3 024 | 1.48 | 1.11 |
# | blob 25-frame trajectory (frame 14)    |      28 135 |    5 087 | 2.01 | 1.52 |
# | blob trajectory, 25-frame mean         | 9 351-45 475|    5 087 | 1.55 | 1.51 |
# | production `F` (8 188 leaves, p=2)     |      78 720 |  5.7e6   | 2.27 | 2.66 |
# | production `P1` (10 000 leaves, p=2)   |      95 308 |  3.3e6   | 1.60 | 2.06 |
#
# so the selector is: any contrast above 20k dofs, and contrast <= 100 above
# 2k dofs.  The old ladder (60k / 20k / 8k, with a contrast 10 rung) moved down
# 3-4x because MG got faster and the direct path did not, and it lost its
# middle contrast rung: the `10 < contrast <= 100` band used to earn MG only at
# 20k dofs with no measurement behind it, and the two stratified-50x rows above
# now show it costs the V-cycle nothing measurable against contrast 1 (1.74x
# cold at 2 803 dofs, 2.20x at 10 851, against 1.74x / 2.36x at contrast 1).
# The only contrast boundary left is 100, which is where the pressure block
# switches from the mass inverse to w-BFBT -- a property of the *MG
# configuration*, not a measured crossover.  Where each constant sits (cold
# ratios are the conservative column; `docs/dev/SOLVER.md` §4.2 lists them):
#
#   * 20 000 -- every measured high-contrast state at or above it wins
#     (19 413 dofs 1.24x at 8 threads, 28 135 dofs 1.32x cold / 2.01x warm,
#     45 475 dofs 1.14x cold / 1.78x warm, 99 559 1.59x cold); below it the
#     high-contrast rows are a tie (9 351 dofs, 1.09x cold; 17 135 and 19 413
#     dofs are 0.81-0.82x at `mg_tol=1e-10`).  The old 60k sat 1.32x above its
#     parity point (45 475); 20k sits 2.1x above the last measured
#     high-contrast tie and keeps the 17-19k frames that lose at the library
#     tolerance on the direct side.
#   * 2 000 -- 747 dofs is the last measured tie/loss (1.17x cold at one
#     thread, 0.91x cold at eight); 2 467 dofs (p=2) is 1.22-1.48x cold /
#     1.91x warm, 2 803 dofs (p=4) is 1.74x cold at both thread counts, and the
#     contrast-50 tree of the same size is 1.51-1.74x cold.  The old 8k was
#     never measured below 10 851; the new constant is bracketed by two.
#
# `saddle_dofs` uses the convention of that table: `2*Nu + Np`, i.e. the total
# velocity + pressure unknowns before Dirichlet/hanging elimination (the
# reduced nf+nfp is a few % smaller).
const SEMQUAD_AUTO_DOFS_MG_ANY = 20_000
const SEMQUAD_AUTO_DOFS_CONTRAST_100 = 2_000
const SEMQUAD_AUTO_CONTRAST_MID = 100.0
# Contrast sampler resolution: (GRID+1)^2 fixed Cartesian points spanning
# [0,Lx]x[0,1].  Measured on the production blob eta field, 32 intervals
# recovers the full quadrature contrast to -3%/-9% on the 1024/4096-leaf trees
# at ~23 us/1 089 evaluations, while the <=256 strided velocity-node sample
# misses by up to 5x (docs/S3_DESIGN.md §20).  Fixed cost, independent of N.
const SEMQUAD_AUTO_CONTRAST_GRID = 32
const SEMQUAD_AUTO_MG_MAX_THREADS = 8

# Diagnostics of the last `linear=:auto` resolution: the chosen concrete path
# (`:direct`/`:mg`; `:none` when `linear` was not `:auto`), a human-readable
# reason with the two measured properties, and the sampler cost.  Log it from a
# production driver (`MantleSEM.LAST_SEMSTOKES_AUTO_CHOICE[]`).
const _SEMSTOKES_AUTO_CHOICE_NONE =
    (choice=:none, reason="", saddle_dofs=0, contrast=1.0, samples=0,
     sample_ms=0.0, mg_threads=0)
const LAST_SEMSTOKES_AUTO_CHOICE = Ref(_SEMSTOKES_AUTO_CHOICE_NONE)

# Safety net of `linear=:auto`: when the selector picks `:mg` but that solve is
# not acceptable -- its true reduced-saddle residual is above the caller's
# `auto_fallback_tol` (the driver passes its `stokes_tol`), or the MG path
# throws -- the solve is redone with the sparse direct path instead of aborting
# the run.  `count` is the number of such solves and `last_reason` describes the
# last one; explicit `:mg` calls never touch either (they still fail loudly).
# Reset from a driver/gate with `LAST_AUTO_FALLBACK[] = (count=0,
# last_reason="")`.
const LAST_AUTO_FALLBACK = Ref((count=0, last_reason=""))

# Pre-built reasons for the explicit-path case: recording `:none` must not
# allocate on the hot per-solve path (this block runs for every explicit
# `:direct`/`:mg` call and is the *only* thing the selector adds to it).
const _SEMSTOKES_AUTO_OFF_REASONS = Dict{Symbol,String}(
    :direct => "linear=:direct (auto not requested)",
    :schur_cg => "linear=:schur_cg (auto not requested)",
    :mg => "linear=:mg (auto not requested)",
    :lagged => "linear=:lagged (auto not requested)")

"""Format a viscosity contrast for the `:auto` reason string."""
_fmt_eta_contrast(c::Real) = @sprintf("%.4g", c)

"""
    semquad_choose_linear(saddle_dofs, contrast) -> :direct | :mg

Production path selector behind `linear=:auto` (S6).  `saddle_dofs` is the
total number of velocity + pressure unknowns (`2*Nu + Np`, the convention of
the measured table in the comment above) and `contrast >= 1` is
`max(eta)/min(eta)` on the eta field the solve uses (sampled by
`semquad_eta_contrast`).  The rule, in order:

1. `saddle_dofs >= 20_000` -> `:mg`.  Measured: MG 1.24x faster at 19 413 dofs
   (high-contrast blob trajectory), 1.55-1.78x on the frame-19 tree (45 475
   dofs), 2.4-2.8x at 99 559-174 915, 6.97-7.59x on uniform trees; it also uses
   ~1/2-1/3 of the direct solve memory.
2. `contrast <= 100 && saddle_dofs >= 2_000` -> `:mg`.  Measured: 1.51-1.74x at
   2 803 dofs (uniform and stratified contrast-50 trees), 1.91-2.75x at
   2 467-10 851, 3.5-4.0x at 10 851 with contrast 50; the 747-dof tree is the
   last measured tie (0.91x cold at 8 threads).
3. otherwise -> `:direct`.  The measured high-contrast ties (9 351 dofs, 1.09x
   cold; 17 135-19 413 dofs at the library tolerance) and the sub-2k trees fall
   here; `:direct` is also the bitwise path, so ties go to it.

The returned symbol is what the solver actually runs; explicit `:direct` and
`:mg` calls never consult this function and keep their historical behaviour.
"""
function semquad_choose_linear(saddle_dofs::Integer, contrast::Real)
    choice, _ = _semquad_choose_linear_rule(saddle_dofs, contrast)
    return choice
end

"""
    _semquad_choose_linear_rule(saddle_dofs, contrast) -> (choice, reason)

The rule of [`semquad_choose_linear`](@ref) plus the human-readable reason used
for `LAST_SEMSTOKES_AUTO_CHOICE`.  Kept in one place so the logged reason can
never drift from the decision.
"""
function _semquad_choose_linear_rule(saddle_dofs::Integer, contrast::Real)
    dofs = Int(saddle_dofs)
    c = isnan(contrast) ? Inf : max(1.0, Float64(contrast))
    if dofs >= SEMQUAD_AUTO_DOFS_MG_ANY
        return (:mg, "saddle_dofs=$dofs >= $SEMQUAD_AUTO_DOFS_MG_ANY: " *
                     "MG wins 1.2-7.6x from this size (and uses less solver " *
                     "memory)")
    elseif c <= SEMQUAD_AUTO_CONTRAST_MID && dofs >= SEMQUAD_AUTO_DOFS_CONTRAST_100
        return (:mg, "saddle_dofs=$dofs >= $SEMQUAD_AUTO_DOFS_CONTRAST_100 and " *
                     "eta contrast $(_fmt_eta_contrast(c)) <= " *
                     "$SEMQUAD_AUTO_CONTRAST_MID (uniform-tree crossover is " *
                     "below 2.5k dofs)")
    end
    return (:direct, "saddle_dofs=$dofs and eta contrast " *
                     "$(_fmt_eta_contrast(c)) are below the measured MG " *
                     "crossover (high contrast needs 20k dofs; low contrast " *
                     "needs 2k)")
end

const _SEMQUAD_AUTO_SERIAL_WARNED = Ref(false)

"""
    semquad_auto_mg_threads() -> Int

Thread count used by `linear=:auto` when it picks the matrix-free MG:
`min(8, Threads.nthreads())`.  Measured (docs/S3_DESIGN.md §19.2), the threaded
V-cycle is 1.5x faster at 1024 leaves and 2.2x at 4096 leaves; 2 threads are
not worth it on this machine, 4/8 are.  A single-threaded process falls back to
1 and warns once.  This never applies to an explicit `linear=:mg` call, whose
`mg_threads` default stays 1 (bitwise-serial).
"""
function semquad_auto_mg_threads()
    nt = min(SEMQUAD_AUTO_MG_MAX_THREADS, Threads.nthreads())
    if nt == 1 && !_SEMQUAD_AUTO_SERIAL_WARNED[]
        _SEMQUAD_AUTO_SERIAL_WARNED[] = true
        @warn "linear=:auto selected the matrix-free MG, but Julia was " *
              "started with 1 thread: using the serial V-cycle " *
              "(mg_threads=1). Start Julia with --threads=$SEMQUAD_AUTO_MG_MAX_THREADS " *
              "for the measured 1.5-2.2x V-cycle speedup." maxlog=1
    end
    return nt
end

"""
    _semquad_resolve_auto(ss, eta_fun; schur_precond=:mass) -> (choice, mg_threads)

Resolve `linear=:auto` **before** the sparse assembly, so an MG decision never
pays for (or stores) the sparse velocity operator: `saddle_dofs = 2*Nu + Np`
(node counts, no full O(N) sample), `contrast` from the fixed 33x33-grid eta
sample (`semquad_eta_contrast`, ~4-23 us measured), then
[`semquad_choose_linear`](@ref).  On an MG decision it also forces the
calibrated MG configuration (`mg_operator=:matrixfree`, `mg_threads =
semquad_auto_mg_threads()`); `schur_precond=:mass` is required by the MG path,
so an auto request with another preconditioner falls back to `:direct` instead
of throwing.  Records `LAST_SEMSTOKES_AUTO_CHOICE`.
"""
function _semquad_resolve_auto(ss::SemStokes, eta_fun::Function;
                               schur_precond::Symbol=:mass)
    dofs = 2 * ss.su.N + ss.sp.N
    t0 = time_ns()
    contrast = semquad_eta_contrast(ss.su, eta_fun)
    sample_ms = (time_ns() - t0) / 1.0e6
    choice, reason = _semquad_choose_linear_rule(dofs, contrast)
    mg_threads = 0
    if choice === :mg && schur_precond !== :mass
        choice = :direct
        reason *= "; MG needs schur_precond=:mass, falling back to direct"
    elseif choice === :mg
        mg_threads = semquad_auto_mg_threads()
        reason *= "; using mg_operator=:matrixfree, mg_threads=$mg_threads"
    end
    LAST_SEMSTOKES_AUTO_CHOICE[] =
        (choice=choice, reason=reason, saddle_dofs=dofs, contrast=contrast,
         samples=(SEMQUAD_AUTO_CONTRAST_GRID + 1)^2, sample_ms=sample_ms,
         mg_threads=mg_threads)
    return choice, mg_threads
end

# --- stale-factor cache for the direct saddle (S2: lagged factorization) -----
#
# The reduced saddle A = [Kred Bt; Bs 0] depends on the viscosity field and on
# the tree, not on dt.  With AMR every few steps the tree changes rarely, and
# between two AMR cycles the viscosity changes only through the temperature
# update.  So the LU of the previous solve is a good preconditioner for the
# current one: we reuse it and apply iterative refinement (Wilkinson), and only
# refactor when the refinement stalls or the tree changes.  This is the same
# idea as a frozen/stale preconditioner in PETSc/Trilinos (`-ksp_reuse_preconditioner`,
# ASPECT rebuilding its AMG only every few steps); iterative refinement with an
# LU factor is the cheapest form of it.
mutable struct StaleSaddle
    key::UInt64
    F::Any
    nfac::Int
end

const STALE_SADDLE = Ref{Union{Nothing,StaleSaddle}}(nothing)
const LAST_SEMSTOKES_LAGGED_ITERS = Ref(0)
const LAST_SEMSTOKES_LAGGED_REFACTOR = Ref(false)
const SEMSTOKES_LAGGED_REFACTORS = Ref(0)

"""Order-independent fingerprint of the tree plus the solver geometry."""
function _saddle_key(qt::Quadtree, p::Int, bc::Symbol, nf::Int, nfp::Int)
    h = hash((p, bc, nf, nfp))
    lvs = collect(qt.leaves)
    sort!(lvs)
    for lf in lvs
        h = hash(lf, h)
    end
    return h
end

"""
    _lagged_solve!(S, rhs, key; tol=1e-8, maxit=6, force_refactor=false)

Solve `S x = rhs` reusing the cached factorization of the previous saddle when
the tree fingerprint still matches, and polishing with iterative refinement.
Returns `(x, relative_residual)`.  A refactor happens when the cache misses,
when refinement did not reach `tol`, or when `force_refactor` is set.
"""
function _lagged_solve!(S::SparseMatrixCSC, rhs::Vector{Float64}, key::UInt64;
                        tol::Float64=1.0e-8, maxit::Int=6, force_refactor::Bool=false)
    # Symmetric diagonal equilibration: the blob saddle at Ra=1e8 has a condition
    # number ~1e12, and a stale factorisation only behaves as a preconditioner
    # after scaling (same trick as the :schur_cg path uses on Kred).
    dg = abs.(diag(S))
    dg .= max.(dg, 1.0e-14 * max(maximum(dg), 1.0))
    d = sqrt.(dg)
    Ds = spdiagm(0 => 1.0 ./ d)
    Ss = Ds * S * Ds
    brhs = Ds * rhs

    st = STALE_SADDLE[]
    refactored = false
    if force_refactor || st === nothing || st.key != key
        st = StaleSaddle(key, lu(Ss), st === nothing ? 1 : st.nfac + 1)
        STALE_SADDLE[] = st
        SEMSTOKES_LAGGED_REFACTORS[] += 1
        refactored = true
    end
    F = st.F
    applyA = v -> Ss * v
    precond = r -> F \ r
    y0 = F \ brhs
    y, _, nit = gmres_restarted(applyA, brhs; precond=precond, tol=tol,
                                maxit=maxit, restart=maxit, x0=y0)
    x = Ds * y
    r = rhs - S * x
    rn = norm(r) / max(norm(rhs), 1.0)
    # GMRES minimise the preconditioned residual; accept on the true one, at the
    # Stokes tolerance (1e-6), otherwise refactor.
    if !(rn <= 1.0e-6)
        st = StaleSaddle(key, lu(Ss), st.nfac + 1)
        STALE_SADDLE[] = st
        SEMSTOKES_LAGGED_REFACTORS[] += 1
        refactored = true
        x = Ds * (st.F \ brhs)
        r = rhs - S * x
        rn = norm(r) / max(norm(rhs), 1.0)
        nit = -1
    end
    LAST_SEMSTOKES_LAGGED_ITERS[] = nit
    LAST_SEMSTOKES_LAGGED_REFACTOR[] = refactored
    return x, rn
end

# ---------------------------------------------------------------------------
# The direct path's diagonal-shift ladder (docs/dev/CASE_REALISTIC.md §13)
# ---------------------------------------------------------------------------

# Relative diagonal shifts tried, in order, by `_shift_ladder_solve`.  The first
# rung is the historical single retry of the direct branch (a wall-refined tree
# can make the *unshifted* sparse LU unstable, G164); the larger rungs trade a
# larger perturbation of the operator for a better-conditioned factorisation and
# are only reached when the previous rung still misses the caller's acceptance
# tolerance.
const SEMSTOKES_SHIFT_LADDER = (1.0e-12, 1.0e-10, 1.0e-8)

# Relative diagonal shifts tried *before* the ladder when the unshifted LU has
# lost the solution outright (docs/dev/CASE_REALISTIC.md §13.10).  The wall-refined
# realistic tree carries a cluster of near-null pressure modes on its level-5
# wall strips (a discrete inf-sup deficiency of that strip: measured smallest
# |eigenvalue| 4.6e-10 against sigma_max 1.2e5, the three smallest singular
# values of the 1e-12-shift operator being 4.6e-10, 5.4e-10 and 9.3e-10).  A
# shift has two jobs there -- it makes the (structurally zero) pressure block
# pivottable and it biases the answer by `shift*|x|` -- and the historical
# 1e-12 rung sits ~170x above the cluster, so its bias (1e-4 .. 1e-5 relative on
# the measured states) is what limits the residual.  A shift at the cluster's
# own scale keeps the factorization stable (measured growth 1e-4 against 1e15
# unshifted) and leaves a bias two orders smaller.  The residual is measured, so
# an unlucky shift is simply rejected; the ladder then continues as before.
const SEMSTOKES_TINY_SHIFTS = (1.0e-14, 1.0e-13)
# An unshifted relative residual above this means the factorization did not
# solve the system at all (the answer is worse than `x = 0`).  Only then does the
# *priority* tiny-shift phase (rule 2) run, so a solve that already produced a
# usable answer -- however marginal -- is bitwise unchanged on the historical
# path.  The threshold is deliberately not lowered: the last-resort phase (rule
# 7) covers the states just *below* it, where the historical rungs can still sit
# inside the wall-strip cluster (G step 7641 measured 0.9654, 3.5 % below 1.0;
# docs/dev/CASE_REALISTIC.md §17.3-§17.4).
const SEMSTOKES_LOST_TOL = 1.0
# Allowed `stokes_scaling` values of the direct path
# (docs/PRESSURE_EQUILIBRATION.md): `:none` is the historical unscaled saddle,
# bit for bit; `:equilibrated` is the symmetric congruence below.  No other
# value changes an existing path.
const SEMSTOKES_SCALINGS = (:none, :equilibrated)

"""
    _semquad_stokes_equil_diag(Kred, Bt, Bs) -> Vector{Float64}

Symmetric equilibration diagonal `d = [d_u; d_p]` for the reduced saddle
`S = [Kred Bt; Bs 0]` (docs/PRESSURE_EQUILIBRATION.md):

* `d_u = 1/sqrt(diag Kred)` removes the viscosity scale from the momentum block
  (the lid's `eta = 1e4` against the interior's `1e-3` is the whole reason the
  wall checkerboard is weak: `beta ~ 0.696/sqrt(eta_lid)`,
  docs/PRESSURE_NULLSPACE_DIAGNOSIS.md §7);
* `d_p = 2 max|Kred| / max(|Bt|, |Bs|)` is a single scalar that balances the
  constraint block against the momentum block: it makes `d_p |B|` comparable to
  `|K|`.  This is the max-norm equilibration of the two blocks, i.e. the
  matrix-derived form of the `eta/h` pressure scale, with a factor 2 of margin
  (the measured plateau starts at ~0.3 of `max|K|/max|B|` and extends over four
  decades, docs/PRESSURE_EQUILIBRATION.md §2.1).

The congruence `S -> D S D`, `rhs -> D rhs` has the exact same solution
(`x = D y`); it changes only the arithmetic of the factorisation.  In the
scaled operator the wall-strip pressure family is lifted far above the
round-off floor: the measured smallest eigenvalue of the scaled operator is a
*momentum* mode at `4.2e-11` relative (AMR 7641), not the `1.7e-14` pressure
mode of the unscaled system.
"""
function _semquad_stokes_equil_diag(Kred::SparseMatrixCSC, Bt::SparseMatrixCSC,
                                   Bs::SparseMatrixCSC)
    dk = diag(Kred)
    du = 1.0 ./ sqrt.(max.(dk, 1.0e-14 * maximum(dk)))
    bmax = max(maximum(abs, Bt), maximum(abs, Bs), 1.0e-300)
    dp = fill(2.0 * maximum(abs, Kred) / bmax, size(Bs, 1))
    return vcat(du, dp)
end

"""
    _semquad_stokes_equil_solve(S, Kred, Bt, Bs, rhs) -> (x, d)

One unshifted LU of the equilibrated saddle: `D = diag(d)`,
`x = D * ((D*S*D) \\ (D*rhs))`.  `x` solves `S x = rhs` to the accuracy of the
scaled factorisation (no shift of any kind); `d` is the equilibration diagonal
of [`_semquad_stokes_equil_diag`](@ref).  Exposed for the G214 gate, which
checks the diagonal and the congruence on a real saddle.
"""
function _semquad_stokes_equil_solve(S::SparseMatrixCSC, Kred::SparseMatrixCSC,
                                    Bt::SparseMatrixCSC, Bs::SparseMatrixCSC,
                                    rhs::Vector{Float64})
    d = _semquad_stokes_equil_diag(Kred, Bt, Bs)
    D = spdiagm(0 => d)
    return D * ((D * S * D) \ (D * rhs)), d
end

"""
    _shift_ladder_solve(S, rhs; accept_tol=Inf, bad_tol=1e-8, resid=nothing,
                        shifts=SEMSTOKES_SHIFT_LADDER,
                        tiny_shifts=SEMSTOKES_TINY_SHIFTS,
                        lost_tol=SEMSTOKES_LOST_TOL,
                        refine_maxit=4, refine_tol=1e-10)

Sparse direct solve of the reduced saddle `S x = rhs` with the diagonal-shift
ladder.  Returns `(x, r, shift, refine_iters, rungs)`:

* `x`, `r` — the accepted solution and its relative residual,
* `shift` — the relative diagonal shift of the factorisation behind it: `0.0`
  for the unshifted LU, otherwise one of `tiny_shifts`/`shifts`.  When the
  returned iterate comes from the refinement, it is the shift of the
  factorisation the refinement was preconditioned by,
* `refine_iters` — GMRES iterative-refinement iterations spent after the ladder
  (0 when the ladder's own rung was accepted),
* `rungs` — every rung tried, in order, as `(shift, residual)`; the refinement,
  when it ran, appends `(:refine, residual)`.  `nothing` for a healthy solve.

The phase that produced the returned answer is recorded in
`LAST_SEMSTOKES_LADDER_PHASE` (`:healthy`, `:tiny`, `:ladder`, `:refine`,
`:fallback` or `:fail`; the direct branch also records `:equilibrated` for a
`stokes_scaling = :equilibrated` answer -- see that keyword).

The rules, in the order they apply:

1. **Healthy solve.**  If the unshifted factorisation already satisfies
   `r <= bad_tol` and `r <= accept_tol`, it is returned untouched — one
   factorisation, no shift, `rungs = nothing`.  This is the historical
   behaviour of the direct branch and is what keeps healthy systems bitwise
   identical.
2. **Lost solution** (docs/dev/CASE_REALISTIC.md §13.10).  If `r > lost_tol` the
   unshifted LU did not solve the system at all (its answer is worse than
   `x = 0`): the near-null cluster has swallowed the factorisation, and the
   diagonal shift then has to be chosen at the *cluster's* scale rather than
   far above it, because the shift biases the answer by `shift*|x|`.  The
   `tiny_shifts` are tried in order and the first one that both improves on `r`
   and meets `accept_tol` is accepted; a rung that does not improve is rejected,
   so this phase can never make an answer worse, and if none improves the
   ladder continues exactly as before (`rungs` records every attempt).
3. **First rung.**  Otherwise the shifts are tried in order, *refactoring* each
   time (`lu(S + shift*dmax*I)`, `dmax = max|diag S| + 1` as in the historical
   retry).  The first rung whose residual is `<= accept_tol` is accepted and the
   ladder stops.  With the default `accept_tol = Inf` that is the first rung,
   i.e. exactly the old "retry once with 1e-12" behaviour, including its answer
   for a system whose shifted residual is *worse* than the unshifted one.
4. **Escalation.**  With a finite `accept_tol` a rung that still misses it
   escalates to the next shift.  The best solve seen -- *including the
   unshifted one* -- is remembered, and escalation stops as soon as that best
   meets the gate, so a larger shift (which perturbs the operator more) can
   never throw away an answer that already passed.
5. **Iterative refinement.**  If nothing meets `accept_tol`, a few GMRES
   iterations against the *unshifted* `S`, preconditioned by the best (else the
   last) rung's factorisation, polish the iterate (the `:lagged` machinery: the
   shift makes the factorisation stable, the refinement removes its
   perturbation).  Only an improvement is kept.
6. **Return the best.**  If even that misses `accept_tol` the best iterate is
   returned with its residual; the caller decides (the driver keeps the fatal
   error as the last resort).
7. **Last-resort tiny shifts** (docs/dev/CASE_REALISTIC.md §17.4, G212).  A finite
   unshifted residual at or below `lost_tol` skips rule 2, and the historical
   rungs can then sit inside the wall-strip cluster: G step 7641 measured
   `r = 0.9654` (3.5 % below the 1.0 trigger), `1e-12 -> 1.56e-2`, and four
   refinements only reached 4.47e-3.  If the ladder and its refinement (and
   rule 2, when it ran) all missed `accept_tol`, the unused `tiny_shifts` are
   tried here; the best answer seen is kept -- including the refinement's, so a
   tiny rung only wins by improving on it -- and the solve still fails if they
   all miss.  This costs at most `length(tiny_shifts)` factorisations and only
   runs on a solve that was about to abort: a healthy solve returns at rule 1,
   an accepted rung at rule 3/4, and `accept_tol = Inf` callers never enter the
   phase, so the historical numerics are bitwise unchanged.  `tiny_shifts = ()`
   disables both tiny phases; `lost_tol` only gates rule 2.

`resid(x)` measures the relative residual; the default is
`norm(S*x - rhs)/max(norm(rhs), 1)`.  The driver passes the block-split saddle
residual (momentum rows / continuity rows), which is the number its Stokes gate
checks.  A non-finite `resid` (NaN/Inf from a singular factorisation) is treated
as a failure and recorded as `Inf`, so the ladder fires and a finite rung can
still win; the old path accepted a NaN as "not bad" because `NaN > tol` is
false.
"""
function _shift_ladder_solve(S::SparseMatrixCSC, rhs::Vector{Float64};
                             accept_tol::Float64=Inf,
                             bad_tol::Float64=1.0e-8,
                             resid::Union{Nothing,Function}=nothing,
                             shifts=SEMSTOKES_SHIFT_LADDER,
                             tiny_shifts=SEMSTOKES_TINY_SHIFTS,
                             lost_tol::Float64=SEMSTOKES_LOST_TOL,
                             refine_maxit::Int=4,
                             refine_tol::Float64=1.0e-10)
    rel = resid === nothing ?
          (x -> norm(S * x - rhs) / max(norm(rhs), 1.0)) : resid
    x = S \ rhs
    r = rel(x)
    # A non-finite residual (NaN/Inf from a singular factorisation) is a
    # failure, not "healthy": record it as `Inf` so the ladder fires, a finite
    # rung can win, and the returned residual is then finite.  Before the ladder
    # a NaN was silently accepted as "not bad" (`NaN > tol` is false).
    r = isfinite(r) ? r : Inf
    if !(r > bad_tol) && !(r > accept_tol)
        LAST_SEMSTOKES_LADDER_PHASE[] = :healthy
        return x, r, 0.0, 0, nothing
    end
    dmax = maximum(abs, diag(S)) + 1.0
    rungs = Any[(0.0, r)]
    xbest, rbest, shbest = x, r, 0.0
    Fbest = nothing
    Fshift = nothing
    rshift = Inf
    shshift = 0.0
    # Set when rule 2 has tried the tiny shifts, so rule 7 does not repeat them.
    tiny_used = false
    # --- rule 2: the unshifted LU lost the solution --------------------------
    # A *non-finite* residual is not this case: that is a broken factorisation
    # (a NaN/Inf pivot), not a biased one, and the historical shifted rungs are
    # what recovers it (§13.9/G207).  The tiny shifts are for the finite-but-lost
    # regime the wall-refined realistic tree produces.
    if !isempty(tiny_shifts) && isfinite(r) && r > lost_tol
        tiny_used = true
        for sh in tiny_shifts
            F = lu(S + (sh * dmax) * I)
            xs = F \ rhs
            rs = rel(xs)
            push!(rungs, (Float64(sh), rs))
            if rs < rshift
                Fshift, rshift, shshift = F, rs, Float64(sh)
            end
            if isfinite(rs) && rs <= accept_tol && rs < r
                # A genuine improvement on a lost solve: accept it.  `rs < r`
                # is what keeps this phase monotone -- it can never hand back
                # something worse than the unshifted answer it replaced.
                xbest, rbest, shbest, Fbest = xs, rs, Float64(sh), F
                break
            end
            if isfinite(rs) && rs < rbest
                xbest, rbest, shbest, Fbest = xs, rs, Float64(sh), F
            end
        end
        if rbest < r && rbest <= accept_tol
            LAST_SEMSTOKES_LADDER_PHASE[] = :tiny
            return xbest, rbest, shbest, 0, rungs
        end
    end
    for sh in shifts
        F = lu(S + (sh * dmax) * I)
        xs = F \ rhs
        rs = rel(xs)
        push!(rungs, (Float64(sh), rs))
        if rs < rshift
            Fshift, rshift, shshift = F, rs, Float64(sh)
        end
        if rs <= accept_tol
            # This rung meets the gate: accept it.  With the default
            # `accept_tol = Inf` this is the historical "retry once with 1e-12
            # and take the answer", including for a system whose shifted
            # residual is worse than the unshifted one.
            xbest, rbest, shbest, Fbest = xs, rs, Float64(sh), F
            break
        end
        if rs < rbest
            xbest, rbest, shbest, Fbest = xs, rs, Float64(sh), F
        end
        # Stop escalating when what we already have passes: a larger shift
        # perturbs the operator more, so a rung that is *worse* than the
        # unshifted solve must not cost us the unshifted answer.  Measured on
        # the step-6130 realistic state: the unshifted residual is 4.0e-4 (it
        # passes the 1e-3 card gate) while the 1e-12 rung is 1.3e-3 (it does
        # not), so the ladder returns the unshifted solve instead of refining.
        rbest <= accept_tol && break
    end
    nref = 0
    phase = :ladder
    if rbest > accept_tol
        # The preconditioner of the refinement is the factorisation of the best
        # *shifted* rung: a shifted factorisation is the well-conditioned one
        # (that is the whole point of the rung), and the refinement then
        # iterates against the unshifted operator to remove the shift's bias.
        F = Fbest === nothing ? Fshift : Fbest
        shF = Fbest === nothing ? shshift : shbest
        if F !== nothing
            xtol = min(refine_tol, 0.01 * accept_tol)
            xr, _, nref = gmres_restarted(v -> S * v, rhs;
                                          precond=r -> F \ r,
                                          tol=xtol, maxit=refine_maxit,
                                          restart=refine_maxit, x0=xbest)
            rr = rel(xr)
            push!(rungs, (:refine, rr))
            if rr < rbest
                # The returned iterate came from the refinement, so report the
                # shift of the factorisation it was preconditioned by (its
                # solve is no longer what we return, but its conditioning is
                # what made the polish possible).
                xbest, rbest, shbest = xr, rr, shF
                phase = :refine
            end
        end
    end
    # --- rule 7: the last-resort tiny shifts ---------------------------------
    # A finite unshifted residual at or below `lost_tol` skips rule 2, and the
    # historical rungs can then sit inside the wall-strip cluster (G step 7641:
    # r = 0.9654, 1e-12 -> 1.56e-2, refinement -> 4.47e-3).  If the ladder has
    # already failed, the tiny shifts are the only untried rungs.  The best
    # answer seen so far is kept, so this phase can never make the answer worse;
    # only an improvement is returned.  It runs only on a solve that is about to
    # abort, and never when rule 2 already tried these shifts.
    if rbest > accept_tol && !isempty(tiny_shifts) && isfinite(r) && !tiny_used
        for sh in tiny_shifts
            F = lu(S + (sh * dmax) * I)
            xs = F \ rhs
            rs = rel(xs)
            rs = isfinite(rs) ? rs : Inf
            push!(rungs, (Float64(sh), rs))
            if rs < rbest
                xbest, rbest, shbest = xs, rs, Float64(sh)
                phase = :fallback
            end
            rbest <= accept_tol && break
        end
    end
    phase = rbest <= accept_tol ? phase : :fail
    LAST_SEMSTOKES_LADDER_PHASE[] = phase
    return xbest, rbest, shbest, nref, rungs
end

function make_semstokes(qt::Quadtree, p::Int)
    return SemStokes(p, p - 1, make_semquad(qt, p), make_semquad(qt, p - 1))
end

"""Reference element matrices for Q_p velocity / Q_q pressure (constant eta)."""
function _sem_stokes_elem(p::Int, q::Int)
    ru = gll_rule(p)
    n = p + 1
    D = zeros(Float64, n, n)
    for i in 1:n
        _, dL = lagrange_eval(ru, ru.nodes[i]); D[i, :] .= dL
    end
    W = Diagonal(ru.weights)
    W2 = kron(W, W)
    Dx = kron(Matrix{Float64}(I, n, n), D)
    Dz = kron(D, Matrix{Float64}(I, n, n))
    Axx = 2.0 .* (Dx' * W2 * Dx) .+ Dz' * W2 * Dz
    Azz = 2.0 .* (Dz' * W2 * Dz) .+ Dx' * W2 * Dx
    Axz = Dz' * W2 * Dx
    # pressure basis evaluated at the velocity GLL nodes
    rq = gll_rule(q)
    nq = q + 1
    E = zeros(Float64, n, nq)
    for i in 1:n
        val, _ = lagrange_eval(rq, ru.nodes[i])
        E[i, :] .= val
    end
    Bx = kron(E' * W, E' * W * D)
    Bz = kron(E' * W * D, E' * W)
    return Axx, Azz, Axz, Bx, Bz
end

"""
    assemble_semquad_pressure_mass(sp; weight=nothing, nq=0)

Sparse pressure mass matrix `M_ij = int weight(x,y) q_i q_j` on the quadtree
pressure space.  This is the standard spectrally-equivalent preconditioner for
the Taylor–Hood Schur complement (Elman, Silvester & Wathen 2014, §4.2): use
`Cp' * M * Cp` as the preconditioner for `S = Bs K^{-1} Bt`.  The full matrix
integrates to the domain area with `weight=nothing`.

`nq = 0` keeps the historical `p+2` Gauss points; the strain-weakening assembly
(docs/RHEOLOGY.md §9) passes the energy quadrature explicitly so that its
weighted mass matrix uses the same rule as the energy mass matrix.
"""
function assemble_semquad_pressure_mass(sp::SemQuad;
                                        weight::Union{Nothing,Function}=nothing,
                                        nq::Int=0)
    p = sp.p
    n = p + 1
    fd = make_fedata(2, p; nq=nq)   # Gauss points; fd.phi_p is the Q_p basis
    I = Int[]; J = Int[]; V = Float64[]
    for lf in sp.leaves
        hx = leaf_hx(sp, lf[1]); hy = leaf_hy(sp, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        pids = sp.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            weight === nothing || (w *= weight(x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx,
                                               y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy))
            for b in 1:n, a in 1:n
                pa = fd.phi_p[a, qx] * fd.phi_p[b, qy]
                ga = pids[(b - 1) * n + a]
                for b2 in 1:n, a2 in 1:n
                    pb = fd.phi_p[a2, qx] * fd.phi_p[b2, qy]
                    gb = pids[(b2 - 1) * n + a2]
                    v = w * pa * pb
                    v != 0.0 && (push!(I, ga); push!(J, gb); push!(V, v))
                end
            end
        end
    end
    return sparse(I, J, V, sp.N, sp.N)
end

"""
    assemble_semquad_pressure_stiffness(sp; weight=nothing)

Sparse pressure stiffness `int weight(x,y) grad q_i . grad q_j` on the quadtree
pressure space.  With one pressure dof pinned (the caller's `Cp`) it is SPD and
can serve as a Schur-complement preconditioner for strongly variable viscosity,
where a (weighted) pressure mass matrix is a poorer approximation.
"""
function assemble_semquad_pressure_stiffness(sp::SemQuad;
                                              weight::Union{Nothing,Function}=nothing)
    p = sp.p
    n = p + 1
    fd = make_fedata(2, p)
    I = Int[]; J = Int[]; V = Float64[]
    for lf in sp.leaves
        hx = leaf_hx(sp, lf[1]); hy = leaf_hy(sp, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        pids = sp.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            weight === nothing || (w *= weight(x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx,
                                               y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy))
            for b in 1:n, a in 1:n
                gxa = fd.dphi_p[a, qx] * fd.phi_p[b, qy] * (2.0 / hx)
                gza = fd.phi_p[a, qx] * fd.dphi_p[b, qy] * (2.0 / hy)
                ga = pids[(b - 1) * n + a]
                for b2 in 1:n, a2 in 1:n
                    gxb = fd.dphi_p[a2, qx] * fd.phi_p[b2, qy] * (2.0 / hx)
                    gzb = fd.phi_p[a2, qx] * fd.dphi_p[b2, qy] * (2.0 / hy)
                    gb = pids[(b2 - 1) * n + a2]
                    v = w * (gxa * gxb + gza * gzb)
                    v != 0.0 && (push!(I, ga); push!(J, gb); push!(V, v))
                end
            end
        end
    end
    return sparse(I, J, V, sp.N, sp.N)
end

# ============================================================================
# S3 / MF-1: matrix-free velocity stiffness `K * v`
# ============================================================================
#
# The assembled `K` of `solve_semquad_stokes_var` is a sum of leaf-local
# operators, and each leaf-local operator is a *tensor product* of 1-D GLL
# operators in the two grid directions.  Nothing about the apply needs the
# matrix: gather the `(p+1)^2` nodal values of each leaf, contract them to the
# four velocity-gradient components at the Gauss points, multiply by the local
# viscosity, and contract back.  That is `O(nq^2 n^2)` work per leaf
# (`n = p+1`, `nq = p+2`) instead of streaming a dense `2n^2 x 2n^2` block.
#
# The quadrature is the *same* Gauss-Legendre rule the assembler uses
# (`make_fedata(p, ss.q)`), and the viscosity is sampled at the same points, so
# `semquad_apply_K!` reproduces `K * v` to round-off for an arbitrary viscosity
# field -- not only for constant eta.  That exactness is what lets the MG
# hierarchy replace the sparse operator outright (MF-3) without changing the
# preconditioner's spectrum.
#
# Hanging constraints are *not* baked in: the operator is the unconstrained
# one, and the caller restricts/prolongs with the existing `Cu`/`Cp`
# (`semquad_apply_Kred!` does `Cu' * (K * (Cu * v))`), so `C'AC` is never
# formed.
#
# Performance: the innermost loop is always one of the Gauss-point directions
# (`nq` long), `phiT`/`dphiT` hold the transposed 1-D tables that the
# accumulation stage needs, and the leaf operator is specialised on the *static*
# polynomial degree (`Val(n)`), which lets LLVM unroll the length-`nq` loops
# (measured: 1.7x over the runtime-sized version, 0.31 ms vs 0.54 ms per apply
# at p=4 on 256 leaves).  The leaf-blocked kernel over `SEMQUAD_KBLK` lanes is
# `_semquad_kn_blk_core!` (stage D); its width is set so the lane axis survives
# to the vectoriser -- docs/dev/MG.md  The remaining cost is the
# gather/scatter, which stays scalar because a scatter has loop-carried
# dependences through shared nodes.

"""
    SemQuadKOp

Leaf-local geometry and 1-D basis data for the matrix-free velocity stiffness.
`ids[:, lf]` are the velocity-node ids of leaf `lf` in the assembler's
`(j-1)*n + i` order, `ihx`/`ihy` the `2/h` reference-to-physical scalings, and
`xq`/`yq` the quadrature-point coordinates (used only when the viscosity is
supplied as a function).
"""
struct SemQuadKOp
    p::Int
    n::Int
    q::Int
    nq::Int
    deviatoric::Bool
    nleaf::Int
    N::Int
    gl_w::Vector{Float64}
    phi::Matrix{Float64}                 # n x nq, basis values at Gauss points
    dphi::Matrix{Float64}                # n x nq, basis derivatives
    phiT::Matrix{Float64}                # nq x n, transposed (accumulation)
    dphiT::Matrix{Float64}
    ids::Matrix{Int}                     # n*n x nleaf
    ihx::Vector{Float64}                 # 2/hx per leaf
    ihy::Vector{Float64}                 # 2/hy per leaf
    wfac::Vector{Float64}                # hx*hy/4 per leaf
    xq::Vector{Float64}                  # nleaf*nq*nq quadrature coordinates
    yq::Vector{Float64}
    blk::Int                             # leaves per blocked kernel (0 = scalar)
end

"""
    SemQuadKScratch

The `SEMQUAD_KBLK`-lane scratch of the leaf-blocked kernel: the gathered nodal
values of one block, its velocity gradients at the Gauss points, the
tensor-product intermediate, the weighted gradients and the leaf-local output.
`SemQuadKWork` owns one instance for the serial path; the threaded path
(`SemQuadKPar`, S5-2) owns one per leaf block, so two threads never share a
lane buffer.
"""
struct SemQuadKScratch
    bxv::Matrix{Float64}                 # (BLK, n*n) gathered nodal x values
    bzv::Matrix{Float64}                 # (BLK, n*n) gathered nodal z values
    buxx::Array{Float64,3}               # (BLK, nq, nq) du_x/dx at Gauss points
    buxy::Array{Float64,3}               # (BLK, nq, nq) du_x/dy
    buzx::Array{Float64,3}               # (BLK, nq, nq) du_z/dx
    buzy::Array{Float64,3}               # (BLK, nq, nq) du_z/dy
    bt1::Array{Float64,3}                # (BLK, n, nq) tensor-product work
    bF::Array{Float64,3}                 # (BLK, nq, nq) weighted gradients
    box::Array{Float64,3}                # (BLK, n, n) leaf-local x output
    boz::Array{Float64,3}                # (BLK, n, n) leaf-local z output
    bshx::Vector{Float64}                # (BLK) 2/hx per block lane
    bshy::Vector{Float64}                # (BLK) 2/hy per block lane
    bswf::Vector{Float64}                # (BLK) hx hy / 4 per block lane
    bone::Vector{Float64}                # (BLK) 1.0, the unit contraction scale
    bkofs::Vector{Int}                   # (BLK) per-lane offset into etav
end

function SemQuadKScratch(n::Int, nq::Int)
    B = SEMQUAD_KBLK
    SemQuadKScratch(zeros(B, n * n), zeros(B, n * n),
                    zeros(B, nq, nq), zeros(B, nq, nq), zeros(B, nq, nq),
                    zeros(B, nq, nq), zeros(B, n, nq), zeros(B, nq, nq),
                    zeros(B, n, n), zeros(B, n, n),
                    zeros(Float64, B), zeros(Float64, B), zeros(Float64, B),
                    ones(Float64, B), zeros(Int, B))
end

"""
    SemQuadKWork

Preallocated scratch for `semquad_apply_K!`; one per operator, reused by every
smoother sweep so the apply does not allocate.

`bs` is the leaf-blocked scratch of the SIMD kernel (stage D) and is used by the
serial path only; the threaded path holds one `SemQuadKScratch` per block in
`SemQuadKPar` instead.
"""
struct SemQuadKWork
    xv::Matrix{Float64}
    zv::Matrix{Float64}
    uxx::Matrix{Float64}
    uxy::Matrix{Float64}
    uzx::Matrix{Float64}
    uzy::Matrix{Float64}
    t1::Matrix{Float64}
    t2::Matrix{Float64}
    F::Matrix{Float64}
    ox::Matrix{Float64}
    oz::Matrix{Float64}
    full::Vector{Float64}
    op_out::Vector{Float64}
    bs::SemQuadKScratch
end

function SemQuadKWork(n::Int, nq::Int, N2::Int)
    SemQuadKWork(zeros(n, n), zeros(n, n), zeros(nq, nq), zeros(nq, nq),
                 zeros(nq, nq), zeros(nq, nq), zeros(n, nq), zeros(n, nq),
                 zeros(nq, nq), zeros(n, n), zeros(n, n), zeros(N2), zeros(N2),
                 SemQuadKScratch(n, nq))
end

function SemQuadKOp(p::Int, n::Int, q::Int, nq::Int, deviatoric::Bool, nl::Int, N::Int,
                    gl_w::Vector{Float64}, phi::Matrix{Float64},
                    dphi::Matrix{Float64}, phiT::Matrix{Float64},
                    dphiT::Matrix{Float64}, ids::Matrix{Int},
                    ihx::Vector{Float64}, ihy::Vector{Float64},
                    wfac::Vector{Float64}, xq::Vector{Float64}, yq::Vector{Float64})
    SemQuadKOp(p, n, q, nq, deviatoric, nl, N, gl_w, phi, dphi, phiT, dphiT,
               ids, ihx, ihy, wfac, xq, yq, SEMQUAD_KBLK)
end

const SEMQUAD_KOP_CACHE = Dict{Any,SemQuadKOp}()

"""Build (or fetch from cache) the leaf-local data of the matrix-free `K`."""
function semquad_kop(ss::SemStokes, qt::Quadtree; deviatoric::Bool=false)
    su = ss.su
    key = (ss.p, ss.q, qt.nx0, qt.ny0, qt.Lx, deviatoric,
           length(qt.leaves), hash(qt.leaves))
    if haskey(SEMQUAD_KOP_CACHE, key)
        return SEMQUAD_KOP_CACHE[key]::SemQuadKOp
    end
    p = ss.p
    q = ss.q
    fd = make_fedata(p, q)
    n = p + 1
    nq = fd.nq
    leaves = sort(collect(qt.leaves))
    nl = length(leaves)
    ids = Matrix{Int}(undef, n * n, nl)
    ihx = Vector{Float64}(undef, nl)
    ihy = Vector{Float64}(undef, nl)
    wfac = Vector{Float64}(undef, nl)
    xq = Vector{Float64}(undef, nl * nq * nq)
    yq = Vector{Float64}(undef, nl * nq * nq)
    for (li, lf) in enumerate(leaves)
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = su.elem[lf]
        for idx in 1:(n * n)
            ids[idx, li] = uids[idx]
        end
        ihx[li] = 2.0 / hx; ihy[li] = 2.0 / hy; wfac[li] = 0.25 * hx * hy
        k = (li - 1) * nq * nq
        for qy in 1:nq, qx in 1:nq
            k += 1
            xq[k] = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx
            yq[k] = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
        end
    end
    op = SemQuadKOp(p, n, q, nq, deviatoric, nl, su.N, fd.gl_w,
                    copy(fd.phi_v), copy(fd.dphi_v),
                    Matrix(transpose(fd.phi_v)), Matrix(transpose(fd.dphi_v)),
                    ids, ihx, ihy, wfac, xq, yq, SEMQUAD_KBLK)
    SEMQUAD_KOP_CACHE[key] = op
    return op
end

"""Viscosity sampled once at every leaf/quadrature point of `op`."""
function semquad_eta_field(op::SemQuadKOp, eta::Function)
    vals = Vector{Float64}(undef, length(op.xq))
    @inbounds for i in eachindex(vals)
        vals[i] = eta(op.xq[i], op.yq[i])
    end
    return vals
end

@inline function _semquad_resolve_eta(op::SemQuadKOp, eta)
    eta isa Function && return semquad_eta_field(op, eta)
    length(eta) == length(op.xq) ||
        throw(ArgumentError("eta has $(length(eta)) values, expected one per " *
                            "leaf/quadrature point ($(length(op.xq)))"))
    return eta
end

"""
    SemQuadEtaLane

The viscosity field of one operator, sampled once per leaf/quadrature point and
held in **two layouts**:

* `scalar` -- the leaf-major field `nleaf*nq^2`, i.e. the layout
  [`semquad_eta_field`](@ref) returns and the scalar kernel, the Jacobi diagonal
  and the leaf stiffness read (`etav[(li-1)*nq^2 + kq]`);
* `lane` -- the **lane-major table** the leaf-blocked kernel reads:

      lane[(bi-1)*nq2*BLK + (kq-1)*BLK + b] = scalar[((bi-1)*BLK + b - 1)*nq2 + kq]

  one contiguous `SEMQUAD_KBLK`-lane vector per block and quadrature point.  The
  last block is padded with its **final leaf**, which is exactly the leaf the
  kernel's padded lanes already read (their weight is `0.0`), so the padded
  lanes contribute the same `0.0` as before.

The values are the same `Float64`s in the same order; only the addressing moves
-- a unit-stride lane read instead of the per-lane gather `etav[kofs[b] + kq]`
-- so every apply stays **bitwise** (G225).  This is the
`Table<2,VectorizedArray>` pre-gather of deal.II's step-37
(docs/dev/MG_LITERATURE.md item 1) at `SEMQUAD_KBLK = 32`.

Subtyping `AbstractVector{Float64}` (indexing leaf-major, i.e. through `scalar`)
lets every existing eta consumer keep working unchanged.
"""
struct SemQuadEtaLane <: AbstractVector{Float64}
    scalar::Vector{Float64}
    lane::Vector{Float64}
    nleaf::Int
    nq2::Int
end

Base.size(e::SemQuadEtaLane) = (length(e.scalar),)
Base.length(e::SemQuadEtaLane) = length(e.scalar)
Base.IndexStyle(::Type{SemQuadEtaLane}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(e::SemQuadEtaLane, i::Int) = e.scalar[i]
Base.parent(e::SemQuadEtaLane) = e.scalar

"""
    semquad_eta_lanes(op, etav) -> SemQuadEtaLane

Build the lane-major table of `etav` (a leaf-major field of `op`, or an existing
`SemQuadEtaLane`, which is returned unchanged).  Called **once per operator
construction** -- per level in the MG hierarchy -- never per apply: it is a pure
re-layout, so it changes no value and no arithmetic (G225).

Cost: one pass over `nleaf*nq^2` doubles and `nblk*nq^2*BLK` of storage, i.e.
the size of the field rounded up to a whole number of `SEMQUAD_KBLK` blocks
(`F` level 6: 131 008 -> 131 072 doubles, `P1` level 7: 160 000 -> 160 256).
"""
function semquad_eta_lanes(op::SemQuadKOp, etav::AbstractVector{Float64})
    nq2 = op.nq * op.nq
    nl = op.nleaf
    length(etav) == nl * nq2 ||
        throw(ArgumentError("eta has $(length(etav)) values, expected one per " *
                            "leaf/quadrature point ($(nl * nq2))"))
    BLK = SEMQUAD_KBLK
    nblk = cld(nl, BLK)
    lane = Vector{Float64}(undef, nblk * nq2 * BLK)
    @inbounds for bi in 1:nblk
        lo = (bi - 1) * BLK + 1
        hi = min(nl, bi * BLK)
        tb = (bi - 1) * nq2 * BLK
        for kq in 1:nq2
            kb = tb + (kq - 1) * BLK
            for b in 1:BLK
                # the last block repeats its final leaf: the kernel's padded
                # lane reads the same leaf and is weighted by 0.0
                li = lo + min(b - 1, hi - lo)
                lane[kb + b] = etav[(li - 1) * nq2 + kq]
            end
        end
    end
    scalar = etav isa Vector{Float64} ? etav : collect(etav)
    return SemQuadEtaLane(scalar, lane, nl, nq2)
end

function semquad_eta_lanes(op::SemQuadKOp, e::SemQuadEtaLane)
    (e.nleaf == op.nleaf && e.nq2 == op.nq * op.nq) ||
        throw(ArgumentError("SemQuadEtaLane was built for $(e.nleaf) leaves x " *
                            "$(e.nq2) quadrature points, but this operator has " *
                            "$(op.nleaf) x $(op.nq * op.nq)"))
    return e
end

# dest[qx,qy] = sb*sc * sum_{i,j} A[i,j] * B[i,qx] * C[j,qy]
# (one tensor-product gradient contraction; qy is the innermost index)
@inline function _semquad_contract!(dest::Matrix{Float64}, A::Matrix{Float64},
                                    B::Matrix{Float64}, C::Matrix{Float64},
                                    sb::Float64, sc::Float64,
                                    tmp::Matrix{Float64}, ::Val{N}) where {N}
    nq = N + 1
    @inbounds begin
        for qy in 1:nq, i in 1:N
            tmp[i, qy] = 0.0
        end
        for j in 1:N, i in 1:N
            a = A[i, j]
            for qy in 1:nq
                tmp[i, qy] += a * C[j, qy]
            end
        end
        for qy in 1:nq, i in 1:N
            tmp[i, qy] *= sc
        end
        for qy in 1:nq, qx in 1:nq
            dest[qx, qy] = 0.0
        end
        for qx in 1:nq, i in 1:N
            b = B[i, qx] * sb
            for qy in 1:nq
                dest[qx, qy] += tmp[i, qy] * b
            end
        end
    end
    return nothing
end

# dest[i,j] += sum_{qx,qy} B[i,qx] * F[qx,qy] * CT[qy,j]
# (one tensor-product accumulation; j is the innermost index)
@inline function _semquad_accum!(dest::Matrix{Float64}, B::Matrix{Float64},
                                 F::Matrix{Float64}, CT::Matrix{Float64},
                                 tmp::Matrix{Float64}, ::Val{N}) where {N}
    nq = N + 1
    @inbounds begin
        for qy in 1:nq, i in 1:N
            tmp[i, qy] = 0.0
        end
        for qx in 1:nq, i in 1:N
            b = B[i, qx]
            for qy in 1:nq
                tmp[i, qy] += b * F[qx, qy]
            end
        end
        for qy in 1:nq, i in 1:N
            t = tmp[i, qy]
            for j in 1:N
                dest[i, j] += t * CT[qy, j]
            end
        end
    end
    return nothing
end

"""
    SEMQUAD_MG_ETA_LANES

`true` (the default) has the leaf-blocked kernel read the viscosity through the
block-major lane table of [`SemQuadEtaLane`](@ref) -- one unit-stride
`SEMQUAD_KBLK`-lane read per quadrature point instead of a per-lane gather.
`false` restores the gather form `etav[kofs[b] + kq]` for the A/B; the two read
the same values, so the switch changes the instruction mix only, never a bit
(docs/dev/MG.md, G225).  The switch is consulted only when the eta field
carries a lane table, i.e. never for a plain `Vector{Float64}` field.
"""
const SEMQUAD_MG_ETA_LANES = Ref(true)

"""
    SEMQUAD_KBLK

Leaves per blocked kernel pass (stage D).  The block index is the innermost
loop index, so the `(p+1)^2` and `nq^2` inner loops become unit-stride vector
operations across `SEMQUAD_KBLK` independent leaves; the leaf-specific scalars
(`2/hx`, `2/hy`, `hx hy/4`, `eta`) are gathered into length-`BLK` vectors first.

**32, not 8** (docs/dev/MG.md): at `BLK = 8` LLVM fully unrolls the
8-lane loop into scalar `vaddsd`/`vmulsd` and the kernel is 2.0-2.4x slower
than at 16-32 lanes, where the loop survives to the vectoriser and becomes
`vaddpd`/`vmulpd`/`vsubpd` (4-wide AVX2).  32 is the measured best (1 thread,
F: raw apply 2 150 us at 8, 835 at 16, 789 at 32, 891 at 64).  A short final
block is padded with a zero-weight copy of its last leaf, so the padded lanes
contribute exactly `0.0` and the result stays bitwise (G180).
"""
const SEMQUAD_KBLK = 32

# dest[b,qx,qy] = sb[b]*sc[b] * sum_{i,j} A[b,(j-1)N+i] * B[i,qx] * C[j,qy]
@inline function _semquad_contract_blk!(dest::Array{Float64,3},
                                        A::Matrix{Float64},
                                        B::Matrix{Float64}, C::Matrix{Float64},
                                        sb::Vector{Float64}, sc::Vector{Float64},
                                        tmp::Array{Float64,3},
                                        ::Val{N}) where {N}
    nq = N + 1
    @inbounds begin
        for qy in 1:nq, i in 1:N
            @simd ivdep for b in 1:SEMQUAD_KBLK
                tmp[b, i, qy] = 0.0
            end
            for j in 1:N
                c = C[j, qy]
                @simd ivdep for b in 1:SEMQUAD_KBLK
                    tmp[b, i, qy] += A[b, (j - 1) * N + i] * c
                end
            end
            @simd ivdep for b in 1:SEMQUAD_KBLK
                tmp[b, i, qy] *= sc[b]
            end
        end
        for qy in 1:nq, qx in 1:nq
            @simd ivdep for b in 1:SEMQUAD_KBLK
                dest[b, qx, qy] = 0.0
            end
            for i in 1:N
                bi = B[i, qx]
                @simd ivdep for b in 1:SEMQUAD_KBLK
                    dest[b, qx, qy] += tmp[b, i, qy] * (bi * sb[b])
                end
            end
        end
    end
    return nothing
end

# dest[b,(j-1)N+i] += sum_{qx,qy} B[i,qx] * F[b,qx,qy] * CT[qy,j]
@inline function _semquad_accum_blk!(dest::Array{Float64,3}, B::Matrix{Float64},
                                     F::Array{Float64,3}, CT::Matrix{Float64},
                                     tmp::Array{Float64,3},
                                     ::Val{N}) where {N}
    nq = N + 1
    @inbounds begin
        for qy in 1:nq, i in 1:N
            @simd ivdep for b in 1:SEMQUAD_KBLK
                tmp[b, i, qy] = 0.0
            end
            for qx in 1:nq
                bi = B[i, qx]
                @simd ivdep for b in 1:SEMQUAD_KBLK
                    tmp[b, i, qy] += bi * F[b, qx, qy]
                end
            end
        end
        for qy in 1:nq, i in 1:N
            for j in 1:N
                ctj = CT[qy, j]
                @simd ivdep for b in 1:SEMQUAD_KBLK
                    dest[b, (j - 1) * N + i] += tmp[b, i, qy] * ctj
                end
            end
        end
    end
    return nothing
end

# The SEMQUAD_KBLK leaves of one block; `nb < BLK` at the end of the leaf list.
# A short final block is padded with a zero-weight copy of its last leaf.
function _semquad_apply_K_blk!(out::Vector{Float64}, op::SemQuadKOp,
                               etav::AbstractVector{Float64},
                               v::AbstractVector{Float64}, work::SemQuadKWork,
                               nb::Int)
    nb = clamp(nb, 1, SEMQUAD_KBLK)
    n = op.n
    n == 2 && return _semquad_kn_blk!(out, op, etav, v, work, nb, Val(2))
    n == 3 && return _semquad_kn_blk!(out, op, etav, v, work, nb, Val(3))
    n == 4 && return _semquad_kn_blk!(out, op, etav, v, work, nb, Val(4))
    n == 5 && return _semquad_kn_blk!(out, op, etav, v, work, nb, Val(5))
    n == 6 && return _semquad_kn_blk!(out, op, etav, v, work, nb, Val(6))
    n == 7 && return _semquad_kn_blk!(out, op, etav, v, work, nb, Val(7))
    n == 8 && return _semquad_kn_blk!(out, op, etav, v, work, nb, Val(8))
    return _semquad_kn_blk!(out, op, etav, v, work, nb, Val(n))
end

"""
    _semquad_kn_blk!(out, op, etav, v, work, nb, ::Val{N})

Leaf-blocked matrix-free velocity stiffness: the same operator as
`_semquad_kn!`, `SEMQUAD_KBLK` leaves per pass with the block index innermost.
Every inner loop has the *constant* trip count `SEMQUAD_KBLK` (the final block
is padded with zero-weight leaves), which is what lets LLVM vectorise across the
block and unroll the `(p+1)^2` / `nq^2` loops.

The per-leaf arithmetic (contraction order, accumulation order) is unchanged, so
the result is **bitwise** identical to the scalar kernel -- G180 checks it.
"""
function _semquad_kn_blk!(out::Vector{Float64}, op::SemQuadKOp,
                          etav::AbstractVector{Float64},
                          v::AbstractVector{Float64}, work::SemQuadKWork,
                          nbmax::Int, ::Val{N}) where {N}
    return _semquad_kn_blk_core!(out, nothing, nothing, op, etav, v, work.bs,
                                 1, op.nleaf, nbmax, Val(N), Val(false))
end

"""
    _semquad_kn_blk_store!(op, etav, v, scr, lo, hi, elx, elz, ::Val{N})

The element-buffer form of `_semquad_kn_blk!`: process the leaves `lo:hi` of
the block `scr` and write each leaf's element vector into the columns
`lo:hi` of `elx`/`elz` (each `(n*n) x nleaf`, leaf-major) **instead of**
scattering into the node vector.  The arithmetic is the same code
(`_semquad_kn_blk_core!` with `Val(true)`), so the element vectors are bitwise
the ones the serial kernel would have added.  The caller scatters them serially
in leaf order (`_semquad_scatter_elts!`); see `SemQuadKPar`.
"""
function _semquad_kn_blk_store!(op::SemQuadKOp, etav::AbstractVector{Float64},
                                v::AbstractVector{Float64},
                                scr::SemQuadKScratch, lo::Int, hi::Int,
                                elx::Matrix{Float64}, elz::Matrix{Float64},
                                ::Val{N}) where {N}
    return _semquad_kn_blk_core!(nothing, elx, elz, op, etav, v, scr, lo, hi,
                                 SEMQUAD_KBLK, Val(N), Val(true))
end

# The shared body of the serial blocked kernel and of the threaded
# element-buffer kernel.  `STORE=false` scatters the leaf-local vectors straight
# into `out` (the serial path, unchanged and bitwise); `STORE=true` writes them
# into the `li`-th column of `elx`/`elz`, and `out`/`elx`/`elz` that are not used
# are `nothing` (the branch is on a `Val`, so it disappears at compile time).
function _semquad_kn_blk_core!(out::Union{Vector{Float64},Nothing},
                               elx::Union{Matrix{Float64},Nothing},
                               elz::Union{Matrix{Float64},Nothing},
                               op::SemQuadKOp, etav::AbstractVector{Float64},
                               v::AbstractVector{Float64},
                               scr::SemQuadKScratch, lo::Int, hi::Int,
                               nbmax::Int, ::Val{N},
                               ::Val{STORE}) where {N,STORE}
    # The eta read is the only difference between the two kernel forms and it is
    # a pure addressing choice (docs/dev/MG.md): with a lane table the
    # lane axis is unit stride, so LLVM emits a vector load instead of the
    # per-lane `vgather` over `etav[kofs[b] + kq]`.  Both read the same
    # `Float64`s, so the result is bitwise (G225).  `SEMQUAD_MG_ETA_LANES =
    # false` picks the gather form for the A/B; a plain `Vector` field has no
    # table and always takes it.
    if etav isa SemQuadEtaLane && SEMQUAD_MG_ETA_LANES[]
        return _semquad_kn_blk_body!(out, elx, elz, op, etav.lane, v, scr,
                                     lo, hi, nbmax, Val(N), Val(STORE),
                                     Val(true))
    end
    return _semquad_kn_blk_body!(out, elx, elz, op, etav, v, scr, lo, hi,
                                 nbmax, Val(N), Val(STORE), Val(false))
end

# `LANE=true`: eta of lane `b` of block `bi` comes from the block-major lane
# table `etab[(bi-1)*nq2*BLK + (kq-1)*BLK + b]` (unit stride in `b`);
# `LANE=false`: from the leaf-major field `etab[kofs[b] + kq]` (per-lane gather).
function _semquad_kn_blk_body!(out::Union{Vector{Float64},Nothing},
                               elx::Union{Matrix{Float64},Nothing},
                               elz::Union{Matrix{Float64},Nothing},
                               op::SemQuadKOp, etab::AbstractVector{Float64},
                               v::AbstractVector{Float64},
                               scr::SemQuadKScratch, lo::Int, hi::Int,
                               nbmax::Int, ::Val{N}, ::Val{STORE},
                               ::Val{LANE}) where {N,STORE,LANE}
    BLK = SEMQUAD_KBLK
    nq = N + 1
    N_ = op.N
    n2 = N * N
    nq2 = nq * nq
    phi = op.phi; dphi = op.dphi; phiT = op.phiT; dphiT = op.dphiT
    gl_w = op.gl_w
    ids = op.ids
    a2 = op.deviatoric ? 4.0 / 3.0 : 2.0
    c2 = op.deviatoric ? 2.0 / 3.0 : 0.0
    xv = scr.bxv; zv = scr.bzv
    uxx = scr.buxx; uxy = scr.buxy; uzx = scr.buzx; uzy = scr.buzy
    t1 = scr.bt1
    F = scr.bF; ox = scr.box; oz = scr.boz
    ihx = op.ihx; ihy = op.ihy; wfac = op.wfac
    # per-leaf scalars and per-leaf offsets into `etav`, gathered once.  These
    # live in the scratch (not `zeros(BLK)` here) so the apply is allocation-free.
    shx = scr.bshx; shy = scr.bshy
    swf = scr.bswf
    one = scr.bone
    kofs = scr.bkofs
    STORE || fill!(out, 0.0)
    @inbounds for lb in lo:BLK:hi
        nb = min(BLK, hi - lb + 1)
        lplast = lb + nb - 1
        # lane-table base of this block.  `lb` is on the `SEMQUAD_KBLK` block
        # grid: the serial call starts at 1 and the threaded chunk bounds are
        # block-aligned (`SemQuadKPar`), so `(lb-1) ÷ BLK` is the block index.
        tbase = LANE ? (((lb - 1) ÷ BLK) * (nq2 * BLK)) : 0
        for b in 1:BLK
            # Pad a short final block with a zero-weight copy of its last leaf:
            # the gather stays in bounds and every contribution is exactly 0.0,
            # so the padded lanes do not change the result.
            li = b <= nb ? (lb + b - 1) : lplast
            shx[b] = ihx[li]; shy[b] = ihy[li]
            swf[b] = b <= nb ? wfac[li] : 0.0
            LANE || (kofs[b] = (li - 1) * nq2)
            for idx in 1:n2
                g = ids[idx, li]
                xv[b, idx] = v[g]
                zv[b, idx] = v[N_ + g]
            end
        end
        _semquad_contract_blk!(uxx, xv, dphi, phi, shx, one, t1, Val(N))
        _semquad_contract_blk!(uxy, xv, phi, dphi, one, shy, t1, Val(N))
        _semquad_contract_blk!(uzx, zv, dphi, phi, shx, one, t1, Val(N))
        _semquad_contract_blk!(uzy, zv, phi, dphi, one, shy, t1, Val(N))
        for idx in 1:n2
            # idx outer / b inner: the lane stores are unit-stride
            @simd ivdep for b in 1:BLK
                ox[b, idx] = 0.0
                oz[b, idx] = 0.0
            end
        end
        # same four separable pieces as the scalar kernel
        for qy in 1:nq, qx in 1:nq
            gw = gl_w[qx] * gl_w[qy]
            kq = (qy - 1) * nq + qx
            kb = LANE ? (tbase + (kq - 1) * BLK) : 0
            @simd ivdep for b in 1:BLK
                etaq = LANE ? etab[kb + b] : etab[kofs[b] + kq]
                cw1 = swf[b] * gw * etaq * shx[b]
                F[b, qx, qy] = cw1 * (a2 * uxx[b, qx, qy] - c2 * uzy[b, qx, qy])
            end
        end
        _semquad_accum_blk!(ox, dphi, F, phiT, t1, Val(N))
        for qy in 1:nq, qx in 1:nq
            gw = gl_w[qx] * gl_w[qy]
            kq = (qy - 1) * nq + qx
            kb = LANE ? (tbase + (kq - 1) * BLK) : 0
            @simd ivdep for b in 1:BLK
                etaq = LANE ? etab[kb + b] : etab[kofs[b] + kq]
                cw2 = swf[b] * gw * etaq * shy[b]
                F[b, qx, qy] = cw2 * (uxy[b, qx, qy] + uzx[b, qx, qy])
            end
        end
        _semquad_accum_blk!(ox, phi, F, dphiT, t1, Val(N))
        for qy in 1:nq, qx in 1:nq
            gw = gl_w[qx] * gl_w[qy]
            kq = (qy - 1) * nq + qx
            kb = LANE ? (tbase + (kq - 1) * BLK) : 0
            @simd ivdep for b in 1:BLK
                etaq = LANE ? etab[kb + b] : etab[kofs[b] + kq]
                cw2 = swf[b] * gw * etaq * shy[b]
                F[b, qx, qy] = cw2 * (a2 * uzy[b, qx, qy] - c2 * uxx[b, qx, qy])
            end
        end
        _semquad_accum_blk!(oz, phi, F, dphiT, t1, Val(N))
        for qy in 1:nq, qx in 1:nq
            gw = gl_w[qx] * gl_w[qy]
            kq = (qy - 1) * nq + qx
            kb = LANE ? (tbase + (kq - 1) * BLK) : 0
            @simd ivdep for b in 1:BLK
                etaq = LANE ? etab[kb + b] : etab[kofs[b] + kq]
                cw1 = swf[b] * gw * etaq * shx[b]
                F[b, qx, qy] = cw1 * (uzx[b, qx, qy] + uxy[b, qx, qy])
            end
        end
        _semquad_accum_blk!(oz, dphi, F, phiT, t1, Val(N))
        if STORE
            for b in 1:BLK
                b <= nb || continue          # padded lanes write nothing
                li = lb + b - 1
                for idx in 1:n2
                    elx[idx, li] = ox[b, idx]
                    elz[idx, li] = oz[b, idx]
                end
            end
        else
            for b in 1:BLK
                b <= nb || continue          # padded lanes write nothing
                li = lb + b - 1
                for idx in 1:n2
                    g = ids[idx, li]
                    out[g] += ox[b, idx]
                    out[N_ + g] += oz[b, idx]
                end
            end
        end
    end
    if STORE
        return nothing
    end
    return out
end

"""
    SEMQUAD_MG_FORKS, SEMQUAD_MG_FORK_TASKS

Diagnostics for the threaded matrix-free path: `SEMQUAD_MG_FORKS` counts
fork/join regions (each one is a `@sync` + `@spawn` set, i.e. one task-creation
event) and `SEMQUAD_MG_FORK_TASKS` the worker tasks those regions created.  Both
stay 0 in a serial run.  A `Ref{Int}` increment is a few ns and does not
allocate, so the counters do not move the timings they are used to explain
(G223 pins the fork count of the consolidated path against the per-call one).
"""
const SEMQUAD_MG_FORKS = Ref(0)
const SEMQUAD_MG_FORK_TASKS = Ref(0)

"""
    _semquad_td_run!(f, nt, nchunks)

Run `f(c)` for `c in 1:nchunks` on **at most `nt`** worker tasks that pull the
chunk index from a shared counter, so the work is dynamically balanced over the
`nt` workers but never spread over the rest of the Julia pool.  `nt <= 1` runs
the chunks in order on the calling task (same values, no scheduling).
"""
function _semquad_td_run!(f::F, nt::Int, nchunks::Int) where {F}
    if nt <= 1 || nchunks <= 1
        for c in 1:nchunks
            f(c)
        end
        return nothing
    end
    SEMQUAD_MG_FORKS[] += 1
    SEMQUAD_MG_FORK_TASKS[] += nt
    counter = Threads.Atomic{Int}(0)
    Threads.@sync for _ in 1:nt
        Threads.@spawn begin
            while true
                c = Threads.atomic_add!(counter, 1) + 1
                c > nchunks && break
                f(c)
            end
        end
    end
    return nothing
end

# ===========================================================================
# S6: one fork/join per level per sweep (`SemQuadParRegion`)
#
# `_semquad_td_run!` above forks and joins **per call**: `@sync` + `nt`
# `@spawn`s + one atomic hand-out cost 6-9 us regardless of the work.  On the
# levels above `SEMQUAD_MG_SERIAL_DOFS` a 6-level V-cycle still makes ~15 such
# calls (applies, Jacobi updates, subtracts, transfers) and pays that overhead
# for every one of them (docs/dev/MG.md, §12).
#
# A `SemQuadParRegion` keeps `nt-1` worker tasks alive for the *whole sequence*
# of parallel phases of one level block (one fork/join per region instead of one
# per call).  The calling task is worker 1: it publishes a phase, runs its own
# share of the chunks and then waits for the other workers to finish.  Between
# phases it runs the serial parts of the level (the sparse `Cu` products, the
# scatter) while the workers wait for the next phase.  The protocol is
#
#     coordinator (worker 1)              worker w = 2..nt
#     ----------------------              -----------------
#     publish(job, nc): job, nc, chunk=0, pub += 1
#     run chunks (dynamic hand-out)       wait pub > last (yield)
#     wait arrive == nt-1                 last = pub
#     arrive = 0                          run chunks (dynamic hand-out)
#                                         arrive += 1
#
# so a phase costs one atomic publish + one barrier instead of a task-creation
# region, and the chunk->worker assignment stays dynamic.  Which worker runs
# which chunk never changes an arithmetic result: every chunk of a phase writes
# only its own entries (or, for the apply, its own leaf columns, which are
# scattered serially in leaf order afterwards), and the *phase order* is exactly
# the call order of the per-call path.  The consolidated path is therefore
# bitwise identical to the per-call one (G223).
#
# `SEMQUAD_MG_PHASES` counts the phases executed (a diagnostic next to
# `SEMQUAD_MG_FORKS`).
const SEMQUAD_MG_PHASES = Ref(0)

"""
    SemQuadParRegion

Persistent worker set of the S6 consolidation; see the comment block above.
"""
mutable struct SemQuadParRegion
    nt::Int                      # workers, including the calling task (worker 1)
    pub::Threads.Atomic{Int}     # phases published so far
    arrive::Threads.Atomic{Int}  # workers that finished the current phase
    stop::Threads.Atomic{Int}    # region over: the workers return
    job::Base.RefValue{Any}      # current phase body, `f(c, w)`
    nc::Threads.Atomic{Int}      # chunks of the current phase
    chunk::Threads.Atomic{Int}   # dynamic hand-out of the current phase
    err::Base.RefValue{Any}      # first exception a worker caught (re-thrown)
end

function SemQuadParRegion(nt::Int)
    return SemQuadParRegion(nt, Threads.Atomic{Int}(0), Threads.Atomic{Int}(0),
                            Threads.Atomic{Int}(0), Ref{Any}(nothing),
                            Threads.Atomic{Int}(0), Threads.Atomic{Int}(0),
                            Ref{Any}(nothing))
end

"""Chunk loop of one phase on worker `w` (a function barrier for `f`)."""
@inline function _semquad_phase_chunks!(f::F, r::SemQuadParRegion, nc::Int,
                                        w::Int) where {F}
    while true
        c = Threads.atomic_add!(r.chunk, 1) + 1
        c > nc && break
        f(c, w)
    end
    return nothing
end

"""Worker loop of a region: run every published phase, then return on `stop`."""
function _semquad_region_worker!(r::SemQuadParRegion, w::Int)
    last = 0
    while true
        while (p = r.pub[]) == last
            r.stop[] != 0 && return nothing
            yield()                      # a co-scheduled worker must run
        end
        last = p
        # `pub` is incremented after `job`/`nc`/`chunk` are written (seq_cst),
        # so the phase data below is the one published with `last`.
        try
            _semquad_phase_chunks!(r.job[], r, r.nc[], w)
        catch err
            # Never leave the coordinator waiting for an arrival that will not
            # happen: record the first failure and still arrive.  The
            # coordinator re-throws it after the phase (G223 exercises this by
            # construction: a wrong worker/scratch pairing throws instead of
            # hanging).
            r.err[] === nothing && (r.err[] = err)
        end
        Threads.atomic_add!(r.arrive, 1)
    end
end

"""Run `body()` as worker 1 of `r`, with the region's workers alive."""
function _semquad_region_run!(body::F, r::SemQuadParRegion) where {F}
    if r.nt <= 1
        body()
        r.err[] === nothing || throw(r.err[])
        return nothing
    end
    r.pub[] = 0                       # workers start with no phase seen
    r.arrive[] = 0
    r.stop[] = 0
    r.err[] = nothing
    SEMQUAD_MG_FORKS[] += 1
    SEMQUAD_MG_FORK_TASKS[] += r.nt - 1
    Threads.@sync begin
        for w in 2:r.nt
            Threads.@spawn _semquad_region_worker!(r, w)
        end
        try
            body()
        finally
            r.stop[] = 1
        end
    end
    r.job[] = nothing                 # do not keep the last phase's vectors alive
    r.err[] === nothing || throw(r.err[])
    return nothing
end

"""Publish `f(c, w)` over `nc` chunks as the next phase of `r` and run it."""
function _semquad_phase!(r::SemQuadParRegion, f::F, nc::Int) where {F}
    if nc <= 1
        f(1, 1)
        return nothing
    end
    SEMQUAD_MG_PHASES[] += 1
    r.job[] = f
    r.nc[] = nc
    r.chunk[] = 0
    Threads.atomic_add!(r.pub, 1)
    _semquad_phase_chunks!(f, r, nc, 1)
    while r.arrive[] != r.nt - 1
        yield()
    end
    r.arrive[] = 0
    r.err[] === nothing || throw(r.err[])
    return nothing
end

"""Number of chunks a vector op of length `n` uses on `nt` workers."""
@inline function _semquad_td_nchunks(n::Int, nt::Int)
    return clamp(SEMQUAD_MG_CHUNKS_PER_THREAD * nt, 1, n)
end

"""
    SemQuadKPar

Threaded scratch of the leaf-blocked matrix-free apply (`mg_threads > 1`, S5-2).

The leaf list is cut into `nchunks` fixed contiguous blocks; block `c` computes
the *element* vectors of its leaves into the columns `bounds[c]:bounds[c+1]-1`
of `elx`/`elz`, using the scratch of the worker that happens to pull it
(`scr[w]`, one `SemQuadKScratch` per worker).  After the parallel pass the
element vectors are scattered into the node vector **serially, in leaf order**,
which is exactly the order the serial kernel accumulates in, so the threaded
apply is **bitwise identical** to the serial one (G190).  A merge of per-block
*node* buffers in block order -- the `assemble_threads` discipline -- would only
be deterministic, not bitwise: the summation order of a node shared by leaves in
different blocks changes with the block count.

`nchunks = SEMQUAD_MG_CHUNKS_PER_THREAD * nt` (capped at `nleaf`) and every
chunk *starts* on a `SEMQUAD_KBLK` block boundary so the lane table of
[`SemQuadEtaLane`](@ref) covers it exactly; the chunks are pulled dynamically by
the `nt` workers: on a hybrid CPU (P+E cores) or with SMT siblings the per-core
speed differs by up to 2-3x, and one chunk per thread
would let the slowest core set the wall time.  Which worker computes a leaf
never changes its element vectors, so the merge stays bitwise serial regardless
of the assignment.

`elx`/`elz` hold one column per leaf, so their size is `(p+1)^2 x nleaf` per
component (`4096` leaves at p=4: `25 x 4096` doubles each).
"""
struct SemQuadKPar
    nt::Int                      # workers (<= Threads.nthreads())
    nchunks::Int
    bounds::Vector{Int}          # length nchunks+1: chunk c is bounds[c]:bounds[c+1]-1
    scr::Vector{SemQuadKScratch} # one per worker
    elx::Matrix{Float64}         # (n*n, nleaf) leaf-local x outputs
    elz::Matrix{Float64}         # (n*n, nleaf) leaf-local z outputs
end

const SEMQUAD_MG_CHUNKS_PER_THREAD = 4

function SemQuadKPar(op::SemQuadKOp, nt::Int)
    nl = op.nleaf
    nt = clamp(nt, 1, max(nl, 1))
    BLK = SEMQUAD_KBLK
    # Chunk *starts* sit on the `SEMQUAD_KBLK` block grid, so a chunk's leaf
    # window is a whole number of lane-table blocks (docs/dev/MG.md).
    # The partition stays contiguous, covers 1:nl and is still proportional to
    # the leaf count within one block; it decides only *which worker* computes a
    # leaf, and the element vectors are scattered in leaf order, so the values
    # do not depend on it (G190/G225).  `nchunks` itself is unchanged: the
    # region's phase protocol uses it, and a phase with one chunk runs inline
    # (below a 32-leaf level the chunks after the first are simply empty).
    nchunks = clamp(SEMQUAD_MG_CHUNKS_PER_THREAD * nt, 1, nl)
    bounds = Vector{Int}(undef, nchunks + 1)
    for c in 0:nchunks
        cut = min(cld((c * nl) ÷ nchunks, BLK) * BLK, nl)
        bounds[c + 1] = cut + 1
    end
    scr = [SemQuadKScratch(op.n, op.nq) for _ in 1:nt]
    return SemQuadKPar(nt, nchunks, bounds, scr, zeros(op.n * op.n, nl),
                       zeros(op.n * op.n, nl))
end

# `Val(n)` dispatch of the element-buffer kernel (same list as the serial path).
function _semquad_kn_blk_store_any!(op::SemQuadKOp,
                                    etav::AbstractVector{Float64},
                                    v::AbstractVector{Float64},
                                    scr::SemQuadKScratch, lo::Int, hi::Int,
                                    elx::Matrix{Float64}, elz::Matrix{Float64})
    n = op.n
    n == 2 && return _semquad_kn_blk_store!(op, etav, v, scr, lo, hi, elx, elz, Val(2))
    n == 3 && return _semquad_kn_blk_store!(op, etav, v, scr, lo, hi, elx, elz, Val(3))
    n == 4 && return _semquad_kn_blk_store!(op, etav, v, scr, lo, hi, elx, elz, Val(4))
    n == 5 && return _semquad_kn_blk_store!(op, etav, v, scr, lo, hi, elx, elz, Val(5))
    n == 6 && return _semquad_kn_blk_store!(op, etav, v, scr, lo, hi, elx, elz, Val(6))
    n == 7 && return _semquad_kn_blk_store!(op, etav, v, scr, lo, hi, elx, elz, Val(7))
    n == 8 && return _semquad_kn_blk_store!(op, etav, v, scr, lo, hi, elx, elz, Val(8))
    return _semquad_kn_blk_store!(op, etav, v, scr, lo, hi, elx, elz, Val(n))
end

"""
    _semquad_scatter_elts!(out, op, par, lo=1, hi=op.nleaf)

Serial scatter-add of the element vectors of the leaves `lo:hi` into the node
vector: `out[ids[idx, li]] += elx[idx, li]` in increasing `li`, then increasing
`idx`.  This is the exact accumulation order of the serial kernel.
"""
function _semquad_scatter_elts!(out::Vector{Float64}, op::SemQuadKOp,
                                par::SemQuadKPar, lo::Int=1, hi::Int=op.nleaf)
    n2 = op.n * op.n
    N_ = op.N
    ids = op.ids
    elx = par.elx; elz = par.elz
    @inbounds for li in lo:hi
        for idx in 1:n2
            g = ids[idx, li]
            out[g] += elx[idx, li]
            out[N_ + g] += elz[idx, li]
        end
    end
    return out
end

"""
    _semquad_apply_K_raw_threaded!(out, op, etav, v, par)

`out .= K * v` with the leaf-local work spread over `par.nchunks` fixed
contiguous leaf blocks (dynamic scheduling) and the scatter-add done serially in
leaf order.  Bitwise identical to `_semquad_apply_K_raw!`; see `SemQuadKPar`.
"""
function _semquad_apply_K_raw_threaded!(out::Vector{Float64}, op::SemQuadKOp,
                                        etav::AbstractVector{Float64},
                                        v::AbstractVector{Float64},
                                        par::SemQuadKPar)
    fill!(out, 0.0)
    nchunks = par.nchunks
    bounds = par.bounds
    scr = par.scr
    elx = par.elx; elz = par.elz
    if par.nt <= 1
        for c in 1:nchunks
            _semquad_kn_blk_store_any!(op, etav, v, scr[1], bounds[c],
                                       bounds[c + 1] - 1, elx, elz)
        end
    else
        SEMQUAD_MG_FORKS[] += 1
        SEMQUAD_MG_FORK_TASKS[] += par.nt
        counter = Threads.Atomic{Int}(0)
        Threads.@sync for w in 1:par.nt
            Threads.@spawn begin
                sw = scr[w]
                while true
                    c = Threads.atomic_add!(counter, 1) + 1
                    c > nchunks && break
                    _semquad_kn_blk_store_any!(op, etav, v, sw, bounds[c],
                                               bounds[c + 1] - 1, elx, elz)
                end
            end
        end
    end
    _semquad_scatter_elts!(out, op, par)
    return out
end

"""Leaf chunk `c` of the threaded apply with the worker's own scratch."""
@inline function _semquad_apply_leaf_chunk!(op::SemQuadKOp,
                                            etav::AbstractVector{Float64},
                                            v::AbstractVector{Float64},
                                            scr::SemQuadKScratch,
                                            par::SemQuadKPar, c::Int)
    b = par.bounds
    _semquad_kn_blk_store_any!(op, etav, v, scr, b[c], b[c + 1] - 1,
                               par.elx, par.elz)
    return nothing
end

"""
    _semquad_apply_K_raw_region!(out, op, etav, v, par, region)

`_semquad_apply_K_raw_threaded!` with its leaf phase published into an already
open `SemQuadParRegion` (S6).  The `fill!` and the serial leaf-order scatter stay
on the calling task; only the leaf compute becomes a phase.  Same chunks, same
kernel, same scatter order as the per-call form, so the result is bitwise equal
(G223); only the task creation moves out of the per-op path.
"""
function _semquad_apply_K_raw_region!(out::Vector{Float64}, op::SemQuadKOp,
                                      etav::AbstractVector{Float64},
                                      v::AbstractVector{Float64},
                                      par::SemQuadKPar,
                                      region::SemQuadParRegion)
    # One scratch slot per region worker: the region must not have more workers
    # than the apply's scratch set (`_semquad_mg_vcycle_region!` builds it as
    # `SemQuadParRegion(par.nt)`; a level with fewer leaves than `mg.threads`
    # clamps `par.nt`).
    region.nt <= par.nt ||
        throw(ArgumentError("region has $(region.nt) workers but the apply " *
                            "has $(par.nt) scratch slots"))
    fill!(out, 0.0)
    _semquad_phase!(region,
                    (c, w) -> _semquad_apply_leaf_chunk!(op, etav, v, par.scr[w],
                                                         par, c), par.nchunks)
    _semquad_scatter_elts!(out, op, par)
    return out
end

"""
    _semquad_spmv_row_chunk!(y, At, x, lo, hi)

`y[lo:hi]` of `y = A * x`, `A` given by its transposed CSC: row `i` is a dot
product over column `i` of `At`, accumulated in increasing column order of `A`
with the same `muladd`s as sparse `mul!` and starting from `0.0`.  Sparse `mul!`
starts every output entry from zero (`_rmul_or_fill!`), so assigning the
accumulated value is bitwise: the `1.0` `muladd` reproduces `mul!`'s
`tmp + C[i]` (the accumulator cannot be `-0.0` when it starts at `+0.0`).
"""
@inline function _semquad_spmv_row_chunk!(y::Vector{Float64},
                                          At::SparseMatrixCSC{Float64,Int},
                                          x::AbstractVector{Float64},
                                          lo::Int, hi::Int)
    @inbounds for i in lo:hi
        tmp = 0.0
        for k in nzrange(At, i)
            tmp = muladd(At.nzval[k], x[At.rowval[k]], tmp)
        end
        y[i] = muladd(tmp, 1.0, 0.0)
    end
    return nothing
end

"""
    _semquad_spmv_at_chunk!(y, A, x, lo, hi)

`y[lo:hi]` of `y = A' * x`: output entry `j` is the dot product over column `j`
of `A` in `nzrange` order, the same arithmetic as the `_At_or_Ac_mul_B!` kernel
`mul!` uses.  No fill is needed because every output entry is assigned.
"""
@inline function _semquad_spmv_at_chunk!(y::Vector{Float64},
                                         A::SparseMatrixCSC{Float64,Int},
                                         x::AbstractVector{Float64},
                                         lo::Int, hi::Int)
    @inbounds for j in lo:hi
        tmp = 0.0
        for k in nzrange(A, j)
            tmp = muladd(A.nzval[k], x[A.rowval[k]], tmp)
        end
        y[j] = muladd(tmp, 1.0, 0.0)
    end
    return nothing
end

"""
    _semquad_spmv_region!(y, At, x, nt, region)

`y .= A * x` split over the workers of an already open region, with `A` given by
its transposed CSC `At` (so each output row is one column dot product of `At`).
Rows are independent and `_semquad_spmv_row_chunk!` is the `mul!` arithmetic, so
the result is bitwise the sparse `mul!` (G224).  Every output entry is assigned,
so the phase needs no `fill!` first.
"""
function _semquad_spmv_region!(y::Vector{Float64},
                               At::SparseMatrixCSC{Float64,Int},
                               x::AbstractVector{Float64}, nt::Int,
                               region::SemQuadParRegion)
    n = length(y)
    nc = _semquad_td_nchunks(n, nt)
    _semquad_phase!(region,
                    (c, w) -> _semquad_spmv_row_chunk!(y, At, x,
                                                       _semquad_td_lo(n, nc, c),
                                                       _semquad_td_hi(n, nc, c)),
                    nc)
    return y
end

"""
    _semquad_spmv_at_region!(y, A, x, nt, region)

`y .= A' * x` split over the workers of an already open region: output entry `j`
is the dot product over column `j` of `A`, the `_At_or_Ac_mul_B!` arithmetic of
sparse `mul!` (G224).  Every output entry is assigned, so the phase needs no
`fill!` first.
"""
function _semquad_spmv_at_region!(y::Vector{Float64},
                                  A::SparseMatrixCSC{Float64,Int},
                                  x::AbstractVector{Float64}, nt::Int,
                                  region::SemQuadParRegion)
    n = length(y)
    nc = _semquad_td_nchunks(n, nt)
    _semquad_phase!(region,
                    (c, w) -> _semquad_spmv_at_chunk!(y, A, x,
                                                      _semquad_td_lo(n, nc, c),
                                                      _semquad_td_hi(n, nc, c)),
                    nc)
    return y
end

"""
    _semquad_apply_Kred_region!(out, op, Cu, Cut, etav, v, work, par, region)

Constrained apply `out .= Cu' * K * (Cu * v)` inside an open region (the S6
analogue of `_semquad_apply_Kred_threaded!`).  With `SEMQUAD_MG_REGION_CU[]`
(the default) the two sparse `Cu` products are region phases too -- `Cu * v` as
a row gather over `Cut`, `Cu' * y` over the columns of `Cu` -- and the leaf apply
is the existing leaf phase.  The products are output-entry independent and keep
the `mul!` arithmetic, so the result is bitwise both the serial and the
serial-`Cu`/threaded-leaf form (G224).  `SEMQUAD_MG_REGION_CU[] = false` calls
`mul!` on the calling task for the A/B (docs/dev/MG.md).
"""
function _semquad_apply_Kred_region!(out::Vector{Float64}, op::SemQuadKOp,
                                     Cu::SparseMatrixCSC{Float64,Int},
                                     Cut::SparseMatrixCSC{Float64,Int},
                                     etav::AbstractVector{Float64},
                                     v::AbstractVector{Float64},
                                     work::SemQuadKWork, par::SemQuadKPar,
                                     region::SemQuadParRegion)
    if SEMQUAD_MG_REGION_CU[]
        _semquad_spmv_region!(work.full, Cut, v, par.nt, region)
    else
        mul!(work.full, Cu, v)
    end
    _semquad_apply_K_raw_region!(work.op_out, op, etav, work.full, par, region)
    if SEMQUAD_MG_REGION_CU[]
        _semquad_spmv_at_region!(out, Cu, work.op_out, par.nt, region)
    else
        mul!(out, transpose(Cu), work.op_out)
    end
    return out
end

"""
    _semquad_apply_Kred_threaded!(out, op, Cu, etav, v, work, par)

Constrained threaded apply `out .= Cu' * K * (Cu * v)` (the `mg_threads > 1`
analogue of `_semquad_apply_Kred!`).  The two sparse `Cu` products stay serial
(they are cheap next to the leaf loop and sparse `mul!` is already
deterministic); only the leaf-local part is threaded, and it is bitwise
identical to the serial one.
"""
function _semquad_apply_Kred_threaded!(out::Vector{Float64}, op::SemQuadKOp,
                                       Cu::SparseMatrixCSC{Float64,Int},
                                       etav::AbstractVector{Float64},
                                       v::AbstractVector{Float64},
                                       work::SemQuadKWork, par::SemQuadKPar)
    mul!(work.full, Cu, v)
    _semquad_apply_K_raw_threaded!(work.op_out, op, etav, work.full, par)
    mul!(out, transpose(Cu), work.op_out)
    return out
end

"""
    _semquad_apply_K_raw!(out, op, etav, v, work; blk=op.blk)

Raw matrix-free apply of the unconstrained velocity stiffness: `out = K * v`,
with `v` the full `2*N` velocity vector (x block first, then z block).  `etav`
is the viscosity at every leaf/quadrature point of `op`.

`blk > 0` selects the leaf-blocked kernel (stage D): `SEMQUAD_KBLK` leaves per
pass with the lane index innermost.  The width is a compile-time constant, so
`blk` only enables the blocked path (any positive value is the same width); it
is not a per-call batch size.  `blk=0` selects the historical scalar kernel,
which is what the correctness gate compares against.
"""
function _semquad_apply_K_raw!(out::Vector{Float64}, op::SemQuadKOp,
                              etav::AbstractVector{Float64},
                              v::AbstractVector{Float64}, work::SemQuadKWork;
                              blk::Int=op.blk)
    nb = (blk > 0 && op.nleaf >= 2) ? blk : 0
    n = op.n
    nb > 0 && return _semquad_apply_K_blk!(out, op, etav, v, work, nb)
    n == 2 && return _semquad_kn!(out, op, etav, v, work, Val(2))
    n == 3 && return _semquad_kn!(out, op, etav, v, work, Val(3))
    n == 4 && return _semquad_kn!(out, op, etav, v, work, Val(4))
    n == 5 && return _semquad_kn!(out, op, etav, v, work, Val(5))
    n == 6 && return _semquad_kn!(out, op, etav, v, work, Val(6))
    n == 7 && return _semquad_kn!(out, op, etav, v, work, Val(7))
    n == 8 && return _semquad_kn!(out, op, etav, v, work, Val(8))
    return _semquad_kn!(out, op, etav, v, work, Val(n))   # rare: dynamic dispatch
end

function _semquad_kn!(out::Vector{Float64}, op::SemQuadKOp,
                      etav::AbstractVector{Float64},
                      v::AbstractVector{Float64}, work::SemQuadKWork,
                      ::Val{N}) where {N}
    nq = N + 1
    N_ = op.N
    phi = op.phi; dphi = op.dphi; phiT = op.phiT; dphiT = op.dphiT
    gl_w = op.gl_w
    ids = op.ids
    a2 = op.deviatoric ? 4.0 / 3.0 : 2.0
    c2 = op.deviatoric ? 2.0 / 3.0 : 0.0
    xv = work.xv; zv = work.zv
    uxx = work.uxx; uxy = work.uxy; uzx = work.uzx; uzy = work.uzy
    t1 = work.t1; t2 = work.t2
    F = work.F; ox = work.ox; oz = work.oz
    fill!(out, 0.0)
    nq2 = nq * nq
    @inbounds for li in 1:op.nleaf
        ihx = op.ihx[li]; ihy = op.ihy[li]; wf = op.wfac[li]
        for j in 1:N, i in 1:N
            g = ids[(j - 1) * N + i, li]
            xv[i, j] = v[g]
            zv[i, j] = v[N_ + g]
        end
        # velocity gradients at the Gauss points (2/h carried by sb/sc)
        _semquad_contract!(uxx, xv, dphi, phi, ihx, 1.0, t1, Val(N))
        _semquad_contract!(uxy, xv, phi, dphi, 1.0, ihy, t1, Val(N))
        _semquad_contract!(uzx, zv, dphi, phi, ihx, 1.0, t1, Val(N))
        _semquad_contract!(uzy, zv, phi, dphi, 1.0, ihy, t1, Val(N))
        fill!(ox, 0.0); fill!(oz, 0.0)
        # The assembler's local blocks are
        #   Kxx = a2 gx gx' + gz gz',      Kzz = a2 gz gz' + gx gx'
        #   Kxz = gz gx' - c2 gx gz',      Kzx = gx gz' - c2 gz gx'
        # (c2 = 2/3 for the deviatoric form, 0 otherwise), and each output
        # splits into two separable pieces:
        #   out_x = Dx*(W.ihx.*(a2 ux_x - c2 uz_y))*Ey' + Ex*(W.ihy.*(ux_y + uz_x))*Dy'
        #   out_z = Ex*(W.ihy.*(a2 uz_y - c2 ux_x))*Dy' + Dx*(W.ihx.*(uz_x + ux_y))*Ex'
        k0 = (li - 1) * nq2
        k = k0
        for qy in 1:nq, qx in 1:nq
            k += 1
            cw1 = wf * gl_w[qx] * gl_w[qy] * etav[k] * ihx
            F[qx, qy] = cw1 * (a2 * uxx[qx, qy] - c2 * uzy[qx, qy])
        end
        _semquad_accum!(ox, dphi, F, phiT, t1, Val(N))
        k = k0
        for qy in 1:nq, qx in 1:nq
            k += 1
            cw2 = wf * gl_w[qx] * gl_w[qy] * etav[k] * ihy
            F[qx, qy] = cw2 * (uxy[qx, qy] + uzx[qx, qy])
        end
        _semquad_accum!(ox, phi, F, dphiT, t2, Val(N))
        k = k0
        for qy in 1:nq, qx in 1:nq
            k += 1
            cw2 = wf * gl_w[qx] * gl_w[qy] * etav[k] * ihy
            F[qx, qy] = cw2 * (a2 * uzy[qx, qy] - c2 * uxx[qx, qy])
        end
        _semquad_accum!(oz, phi, F, dphiT, t2, Val(N))
        k = k0
        for qy in 1:nq, qx in 1:nq
            k += 1
            cw1 = wf * gl_w[qx] * gl_w[qy] * etav[k] * ihx
            F[qx, qy] = cw1 * (uzx[qx, qy] + uxy[qx, qy])
        end
        _semquad_accum!(oz, dphi, F, phiT, t1, Val(N))
        for j in 1:N, i in 1:N
            g = ids[(j - 1) * N + i, li]
            out[g] += ox[i, j]
            out[N_ + g] += oz[i, j]
        end
    end
    return out
end

"""
    semquad_apply_K!(out, ss, qt, eta, v; deviatoric=false, work=nothing)

Matrix-free `out .= K(eta) * v` for the **unconstrained** quadtree velocity
stiffness on `qt` (the operator `solve_semquad_stokes_var` assembles), with `v`
the full `2*N` velocity vector (x block first, then z block).

`eta` is either a function of `(x, y)` -- evaluated at every leaf/quadrature
point on every call -- or a precomputed `Vector` from
`semquad_eta_field(op, eta_fun)` in the ordering of `semquad_kop(ss, qt)`, which
is what the MG smoother uses so the viscosity is sampled once per operator.
`work` reuses the scratch buffers; passing `nothing` allocates them.

The quadrature rule and the viscosity sampling points are the assembler's, so
`out` equals the sparse `K * v` to round-off for any `eta` (G175).
"""
function semquad_apply_K!(out::AbstractVector{Float64}, ss::SemStokes,
                          qt::Quadtree, eta, v::AbstractVector{Float64};
                          deviatoric::Bool=false, work=nothing)
    op = semquad_kop(ss, qt; deviatoric=deviatoric)
    length(v) == 2 * op.N || throw(ArgumentError(
        "v has $(length(v)) entries, expected 2*N = $(2 * op.N)"))
    length(out) == 2 * op.N || throw(ArgumentError(
        "out has $(length(out)) entries, expected 2*N = $(2 * op.N)"))
    w = work === nothing ? SemQuadKWork(op.n, op.nq, 2 * op.N) : work
    etav = _semquad_resolve_eta(op, eta)
    return _semquad_apply_K_raw!(out, op, etav, v, w)
end

"""
    semquad_apply_Kred!(out, ss, qt, Cu, eta, v; deviatoric=false, work=nothing)

Constrained apply `out .= Cu' * K * (Cu * v)` on the reduced (free-dof) velocity
space, i.e. exactly `Kred * v` without ever forming `Kred = Cu' * K * Cu`.  `Cu`
is the velocity affine-constraint matrix returned by
`solve_semquad_stokes_var(...; assemble_only=true)` / the MG geometry; it has at
most `p+1` nonzeros per constrained row, so the two sparse products are
negligible next to the leaf loop.
"""
function semquad_apply_Kred!(out::AbstractVector{Float64}, ss::SemStokes,
                             qt::Quadtree, Cu::SparseMatrixCSC{Float64,Int},
                             eta, v::AbstractVector{Float64};
                             deviatoric::Bool=false, work=nothing)
    op = semquad_kop(ss, qt; deviatoric=deviatoric)
    N2 = 2 * op.N
    w = work === nothing ? SemQuadKWork(op.n, op.nq, N2) : work
    nf = size(Cu, 2)
    length(v) == nf || throw(ArgumentError(
        "v has $(length(v)) entries, expected $(nf) free dofs"))
    length(out) == nf || throw(ArgumentError(
        "out has $(length(out)) entries, expected $(nf) free dofs"))
    etav = _semquad_resolve_eta(op, eta)
    return _semquad_apply_Kred!(out, op, Cu, etav, v, w)
end

"""
    _semquad_apply_Kred!(out, op, Cu, etav, v, work)

Allocation-free core of `semquad_apply_Kred!`, taking the resolved operator and
viscosity field.  This is what the matrix-free multigrid levels call.
"""
function _semquad_apply_Kred!(out::Vector{Float64}, op::SemQuadKOp,
                              Cu::SparseMatrixCSC{Float64,Int},
                              etav::AbstractVector{Float64},
                              v::AbstractVector{Float64}, work::SemQuadKWork)
    mul!(work.full, Cu, v)                     # full-space constrained prolongation
    _semquad_apply_K_raw!(work.op_out, op, etav, work.full, work)
    mul!(out, transpose(Cu), work.op_out)
    return out
end

"""
    _semquad_diag_full!(d, op, etav)

Diagonal of the **unconstrained** matrix-free velocity stiffness: a scatter-add
of the leaf-local element diagonals
`Kloc_xx[k,k] = sum_q w eta (a2 gx_k^2 + gz_k^2)` (and the mirror for `z`).  It
costs `O(nq^2 n^2)` per leaf once, i.e. the same as a single apply, and is what
lets the matrix-free multigrid build a Jacobi diagonal without any sparse `K`
(MF-3).  The deviatoric `-2/3 div(u) div(v)` term does not touch the diagonal.
"""
function _semquad_diag_full!(d::Vector{Float64}, op::SemQuadKOp,
                             etav::AbstractVector{Float64})
    n = op.n; nq = op.nq; N = op.N
    phi = op.phi; dphi = op.dphi; gl_w = op.gl_w; ids = op.ids
    a2 = op.deviatoric ? 4.0 / 3.0 : 2.0
    dlocx = zeros(Float64, n * n)
    dlocz = zeros(Float64, n * n)
    fill!(d, 0.0)
    nq2 = nq * nq
    @inbounds for li in 1:op.nleaf
        ihx = op.ihx[li]; ihy = op.ihy[li]; wf = op.wfac[li]
        fill!(dlocx, 0.0); fill!(dlocz, 0.0)
        k = (li - 1) * nq2
        for qy in 1:nq, qx in 1:nq
            k += 1
            cw = wf * gl_w[qx] * gl_w[qy] * etav[k]
            for j in 1:n, i in 1:n
                gx = dphi[i, qx] * ihx * phi[j, qy]
                gz = phi[i, qx] * dphi[j, qy] * ihy
                idx = (j - 1) * n + i
                dlocx[idx] += cw * (a2 * gx * gx + gz * gz)
                dlocz[idx] += cw * (a2 * gz * gz + gx * gx)
            end
        end
        for idx in 1:(n * n)
            g = ids[idx, li]
            d[g] += dlocx[idx]
            d[N + g] += dlocz[idx]
        end
    end
    return d
end

"""
    _semquad_diag_reduced(op, Cu, etav; mode::Symbol=:constrained)

Jacobi diagonal of the matrix-free *reduced* level operator `Cu' * K * Cu`.

`mode=:constrained` (the default, and the fix of docs/MG_REALISTIC_DIAGNOSIS.md
§5.2) is the **exact** reduced diagonal `diag(Cu' K Cu)`, computed without ever
forming `K` or the product: for every leaf, the constraint rows of its local
dofs give a dense `m x ncol` matrix `W` (`m = 2 n^2` local dofs, `ncol` the
reduced columns that touch the leaf, including the free-row identity entries),
and the leaf contributes `diag(W' K_e W)`.  Summing over leaves is exactly the
diagonal of the assembled product because `K = sum_e K_e` and each `K_e` has no
cross-leaf entries.  Cost is one dense `2n^2 x 2n^2` element matrix per leaf
(the assembler's local block), i.e. a few V-cycle applies' worth of leaf work
once per hierarchy build (measured 0.098 s for all eight levels of the 8188-leaf
production tree).

`mode=:lumped` is the historical `Cu' * diag(K)` approximation (the standard
lumped diagonal), kept for A/B only.  On a hanging-node tree it is *not*
`diag(Cu' K Cu)`: the Lagrange interpolation weights are negative, a hanging row
can cancel to <= 0, and the positivity clip below then makes that row a `1e12`
amplifier in the damped-Jacobi smoother (measured: one V-cycle maps
`||b|| = 1e3` to `||z|| = 1.6e170` on the production tree).
"""
function _semquad_diag_reduced(op::SemQuadKOp,
                               Cu::SparseMatrixCSC{Float64,Int},
                               etav::AbstractVector{Float64};
                               mode::Symbol=:constrained)
    mode in (:constrained, :lumped) ||
        throw(ArgumentError("MG Jacobi diagonal mode must be :constrained or " *
                            ":lumped (got :$mode)"))
    d = if mode === :lumped || !_semquad_diag_has_hanging(Cu)
        # `:lumped` is the historical path; on a conforming tree `Cu` only
        # selects free rows, so `Cu' * diag(K)` *is* `diag(Cu' K Cu)` and the
        # old arithmetic is kept bit for bit.
        _semquad_diag_reduced_lumped(op, Cu, etav)
    else
        _semquad_diag_reduced_constrained(op, Cu, etav)
    end
    # A constrained diagonal is positive up to round-off; the clip is a guard,
    # not a modelling choice (on the production tree it is never active).
    dmax = maximum(d)
    dmax > 0.0 || throw(ArgumentError("matrix-free Jacobi diagonal is not positive"))
    @inbounds for i in eachindex(d)
        d[i] = max(d[i], 1.0e-12 * dmax)
    end
    return d
end

"""
    _semquad_diag_has_hanging(Cu) -> Bool

`Cu` has a hanging-node (interpolation) row iff it is not a pure row selector:
a conforming `Cu` stores exactly one `1.0` per reduced column and nothing else,
so `nnz(Cu) == size(Cu, 2) && all(isone, nonzeros(Cu))` identifies it exactly
(`semquad_constraints` never emits a one-term hanging row: such a node would
coincide with a coarse node and is skipped).
"""
@inline function _semquad_diag_has_hanging(Cu::SparseMatrixCSC{Float64,Int})
    vals = nonzeros(Cu)
    return !(length(vals) == size(Cu, 2) && all(isone, vals))
end

"""
    _semquad_diag_reduced_lumped(op, Cu, etav)

Historical (defective on hanging rows) `Cu' * diag(K)` diagonal of the reduced
matrix-free operator.  Kept only as the `mode=:lumped` A/B reference of
`_semquad_diag_reduced`.
"""
function _semquad_diag_reduced_lumped(op::SemQuadKOp,
                                      Cu::SparseMatrixCSC{Float64,Int},
                                      etav::AbstractVector{Float64})
    dfull = zeros(Float64, 2 * op.N)
    _semquad_diag_full!(dfull, op, etav)
    return Vector{Float64}(transpose(Cu) * dfull)
end

"""
    _semquad_leaf_stiffness!(Kloc, op, etav, li, gx, gz)

The `2n^2 x 2n^2` element stiffness of leaf `li`, accumulated exactly like the
assembler's local block in `_semquad_assemble_block!` (same quadrature points,
weights and expression trees).  `gx`/`gz` are length-`n^2` scratch vectors.
"""
function _semquad_leaf_stiffness!(Kloc::Matrix{Float64}, op::SemQuadKOp,
                                  etav::AbstractVector{Float64}, li::Int,
                                  gx::Vector{Float64}, gz::Vector{Float64})
    n = op.n; nq = op.nq; nv = n * n
    phi = op.phi; dphi = op.dphi; gl_w = op.gl_w
    a2 = op.deviatoric ? 4.0 / 3.0 : 2.0
    ihx = op.ihx[li]; ihy = op.ihy[li]; wf = op.wfac[li]
    fill!(Kloc, 0.0)
    k = (li - 1) * nq * nq
    @inbounds for qy in 1:nq, qx in 1:nq
        k += 1
        w = wf * gl_w[qx] * gl_w[qy]
        etaq = etav[k]
        for j in 1:n, i in 1:n
            kk = (j - 1) * n + i
            gx[kk] = dphi[i, qx] * phi[j, qy] * ihx
            gz[kk] = phi[i, qx] * dphi[j, qy] * ihy
        end
        if op.deviatoric
            for l in 1:nv
                xl = gx[l]; zl = gz[l]
                @simd for kk in 1:nv
                    Kloc[kk, l] += etaq * (a2 * gx[kk] * xl + gz[kk] * zl) * w
                    Kloc[kk, nv + l] += etaq * (gz[kk] * xl -
                                                (2.0 / 3.0) * gx[kk] * zl) * w
                    Kloc[nv + kk, l] += etaq * (gx[kk] * zl -
                                                (2.0 / 3.0) * gz[kk] * xl) * w
                    Kloc[nv + kk, nv + l] += etaq * (a2 * gz[kk] * zl +
                                                     gx[kk] * xl) * w
                end
            end
        else
            for l in 1:nv
                xl = gx[l]; zl = gz[l]
                @simd for kk in 1:nv
                    Kloc[kk, l] += etaq * (a2 * gx[kk] * xl + gz[kk] * zl) * w
                    Kloc[kk, nv + l] += etaq * (gz[kk] * xl) * w
                    Kloc[nv + kk, l] += etaq * (gx[kk] * zl) * w
                    Kloc[nv + kk, nv + l] += etaq * (a2 * gz[kk] * zl +
                                                     gx[kk] * xl) * w
                end
            end
        end
    end
    return Kloc
end

"""
    _semquad_diag_reduced_constrained(op, Cu, etav)

Exact `diag(Cu' K Cu)` of the matrix-free reduced velocity operator; see
`_semquad_diag_reduced`.  The reduced columns of a leaf are read from
`transpose(Cu)` (column `g` = the constraint row of full dof `g`), so no
`K`-matrix and no `Cu' K Cu` product is ever formed.
"""
function _semquad_diag_reduced_constrained(op::SemQuadKOp,
                                           Cu::SparseMatrixCSC{Float64,Int},
                                           etav::AbstractVector{Float64})
    n = op.n; nq = op.nq; N = op.N
    nv = n * n; m = 2 * nv
    nf = size(Cu, 2)
    d = zeros(Float64, nf)
    Ct = SparseMatrixCSC{Float64,Int}(sparse(transpose(Cu)))
    # Per-leaf scratch.  Every local dof contributes at most `p+1` columns
    # (one for a free row, `p+1` for a hanging row), so `(p+2)*m` bounds `ncol`.
    bound = (op.p + 2) * m
    colidx = Vector{Int}(undef, bound)
    Wmat = zeros(Float64, m, bound)
    Kmat = zeros(Float64, m, bound)
    Kloc = zeros(Float64, m, m)
    gx = zeros(Float64, nv)
    gz = zeros(Float64, nv)
    stamp = zeros(Int32, nf)
    slot = zeros(Int32, nf)
    ids = op.ids
    @inbounds for li in 1:op.nleaf
        sid = Int32(li)
        # Reduced columns that this leaf can contribute to.
        ncol = 0
        for a in 1:m
            g = a <= nv ? ids[a, li] : N + ids[a - nv, li]
            for k in nzrange(Ct, g)
                w = Ct.nzval[k]
                w == 0.0 && continue
                i = Ct.rowval[k]
                if stamp[i] != sid
                    stamp[i] = sid
                    ncol += 1
                    colidx[ncol] = i
                    slot[i] = Int32(ncol)
                end
            end
        end
        ncol == 0 && continue
        fill!(Wmat, 0.0)
        for a in 1:m
            g = a <= nv ? ids[a, li] : N + ids[a - nv, li]
            for k in nzrange(Ct, g)
                w = Ct.nzval[k]
                w == 0.0 && continue
                Wmat[a, slot[Ct.rowval[k]]] = w
            end
        end
        _semquad_leaf_stiffness!(Kloc, op, etav, li, gx, gz)
        WC = view(Wmat, :, 1:ncol)
        KC = view(Kmat, :, 1:ncol)
        mul!(KC, Kloc, WC)
        for j in 1:ncol
            s = 0.0
            for a in 1:m
                s += Wmat[a, j] * Kmat[a, j]
            end
            d[colidx[j]] += s
        end
    end
    return d
end

"""
    _semquad_dense_operator(mfops, mfetas, mfwork, Cus, lev, nf)

Materialise the (tiny) coarsest level operator by applying the matrix-free
operator to the unit vectors.  Used by the fully matrix-free hierarchy (MF-3)
in place of the sparse `A[1]`: the coarsest tree has only the base leaves, so
`nf` is `O(100)` and this costs less than one fine-level V-cycle.
"""
function _semquad_dense_operator!(A::Matrix{Float64}, op::SemQuadKOp,
                                  Cu::SparseMatrixCSC{Float64,Int},
                                  etav::AbstractVector{Float64},
                                  work::SemQuadKWork)
    nf = size(A, 1)
    e = zeros(Float64, nf)
    y = zeros(Float64, nf)
    @inbounds for i in 1:nf
        fill!(e, 0.0)
        e[i] = 1.0
        _semquad_apply_Kred!(y, op, Cu, etav, e, work)
        for j in 1:nf
            A[j, i] = y[j]
        end
    end
    @inbounds for j in 1:nf, i in 1:nf
        A[i, j] = 0.5 * (A[i, j] + A[j, i])
    end
    return A
end
"""Affine constraint u = u0 + C u_free from hanging constraints and fixed dofs."""
function _affine_constraint(N::Int, cons::Dict{Int,Tuple{Vector{Int},Vector{Float64}}},
                            fixed::Dict{Int,Float64})
    u0 = zeros(Float64, N)
    for (i, v) in fixed
        u0[i] = v
    end
    # move fixed masters into u0, iterate to resolve chains
    work = Dict(c => (copy(ms), copy(ws)) for (c, (ms, ws)) in cons)
    active = trues(length(work))
    keysv = collect(keys(work))
    keep = Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
    for _ in 1:8
        keep = Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
        changed = false
        for (c, (ms, ws)) in work
            haskey(fixed, c) && continue
            nms = Int[]; nws = Float64[]
            for (m, w) in zip(ms, ws)
                if haskey(fixed, m)
                    u0[c] += w * fixed[m]
                    changed = true
                elseif haskey(work, m) && haskey(fixed, m) == false
                    push!(nms, m); push!(nws, w)
                else
                    push!(nms, m); push!(nws, w)
                end
            end
            keep[c] = (nms, nws)
        end
        work = keep
        changed || break
    end
    # free dofs: not fixed and not constrained
    free = [i for i in 1:N if !haskey(fixed, i) && !haskey(work, i)]
    fmap = Dict(v => k for (k, v) in enumerate(free))
    I = Int[]; J = Int[]; V = Float64[]
    for (k, i) in enumerate(free)
        push!(I, i); push!(J, k); push!(V, 1.0)
    end
    for (c, (ms, ws)) in work
        for (m, w) in zip(ms, ws)
            push!(I, c); push!(J, fmap[m]); push!(V, w)
        end
    end
    C = sparse(I, J, V, N, length(free))
    return u0, C, free
end

"""Assemble and solve the constant-viscosity quadtree Stokes system."""
function solve_semquad_stokes(ss::SemStokes, qt::Quadtree,
                              fx::Function, fz::Function,
                              uDx::Function, uDz::Function;
                              linear::Symbol=:direct, schur_maxiter::Int=200,
                              schur_precond::Symbol=:mass)
    linear in (:direct, :schur_cg) ||
        throw(ArgumentError("linear must be :direct or :schur_cg"))
    schur_precond in (:none, :mass) ||
        throw(ArgumentError("schur_precond must be :none or :mass"))
    p = ss.p; q = ss.q
    su = ss.su; sp = ss.sp
    nu = su.N; np = sp.N
    Nu2 = 2 * nu
    IK = Int[]; JK = Int[]; VK = Float64[]
    IB = Int[]; JB = Int[]; VB = Float64[]
    fv = zeros(Float64, Nu2)
    Axx, Azz, Axz, Bx, Bz = _sem_stokes_elem(p, q)
    n = p + 1
    for lf in su.leaves
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = su.elem[lf]
        pids = sp.elem[lf]
        # The reference-space element matrices are already the physical ones for
        # square leaves: the area jacobian (hx*hy/4) cancels the derivative
        # factors (2/hx)(2/hy) for the second-order terms, and leaves the aspect
        # ratios hy/hx and hx/hy (both 1 for the unit square).
        axx = hy / hx
        azz = hx / hy
        # scatter the velocity block (local order: (j-1)n+i for each component)
        for jb in 1:n, ib in 1:n, ja in 1:n, ia in 1:n
            ki = uids[(jb - 1) * n + ib]; kj = uids[(ja - 1) * n + ia]
            push!(IK, ki); push!(JK, kj)
            push!(VK, axx * Axx[(jb - 1) * n + ib, (ja - 1) * n + ia])
            push!(IK, nu + ki); push!(JK, nu + kj)
            push!(VK, azz * Azz[(jb - 1) * n + ib, (ja - 1) * n + ia])
            push!(IK, ki); push!(JK, nu + kj)
            push!(VK, Axz[(jb - 1) * n + ib, (ja - 1) * n + ia])
            push!(IK, nu + ki); push!(JK, kj)
            push!(VK, Axz[(ja - 1) * n + ia, (jb - 1) * n + ib])
        end
        # divergence coupling: int q d(ux)/dx = (hy/2) int_ref q d/dxi,
        #                      int q d(uz)/dy = (hx/2) int_ref q d/deta
        for b in 1:(q + 1), a in 1:(q + 1), j in 1:n, i in 1:n
            pi_ = pids[(b - 1) * (q + 1) + a]
            uj = uids[(j - 1) * n + i]
            push!(IB, pi_); push!(JB, uj)
            push!(VB, 0.5 * hy * Bx[(b - 1) * (q + 1) + a, (j - 1) * n + i])
            push!(IB, pi_); push!(JB, nu + uj)
            push!(VB, 0.5 * hx * Bz[(b - 1) * (q + 1) + a, (j - 1) * n + i])
        end
        # loads
        for j in 1:n, i in 1:n
            x = x0 + 0.5 * (su.xi[i] + 1.0) * hx
            y = y0 + 0.5 * (su.xi[j] + 1.0) * hy
            w = 0.25 * hx * hy * su.w[i] * su.w[j]
            g = uids[(j - 1) * n + i]
            fv[g] += w * fx(x, y)
            fv[nu + g] += w * fz(x, y)
        end
    end
    K = sparse(IK, JK, VK, Nu2, Nu2)
    B = sparse(IB, JB, VB, np, Nu2)
    # velocity Dirichlet on the domain boundary
    fixed = Dict{Int,Float64}()
    for lf in su.leaves
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = su.elem[lf]
        for dir in (:xminus, :xplus, :yminus, :yplus)
            has_no_face_neighbor(qt, lf, dir) || continue
            if dir === :xminus || dir === :xplus
                iF = dir === :xminus ? 1 : n
                xf = dir === :xplus ? x0 + hx : x0
                for j in 1:n
                    yv = y0 + 0.5 * (su.xi[j] + 1.0) * hy
                    g = uids[(j - 1) * n + iF]
                    fixed[g] = uDx(xf, yv)
                    fixed[nu + g] = uDz(xf, yv)
                end
            else
                jF = dir === :yminus ? 1 : n
                yf = dir === :yplus ? y0 + hy : y0
                for i in 1:n
                    xv = x0 + 0.5 * (su.xi[i] + 1.0) * hx
                    g = uids[(jF - 1) * n + i]
                    fixed[g] = uDx(xv, yf)
                    fixed[nu + g] = uDz(xv, yf)
                end
            end
        end
    end
    cons_u = semquad_constraints(su, qt)
    # lift to the two velocity components
    cons2 = Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
    for (c, (ms, ws)) in cons_u
        cons2[c] = (ms, ws)
        cons2[nu + c] = (nu .+ ms, ws)
    end
    u0, Cu, _ = _affine_constraint(Nu2, cons2, fixed)
    # pressure: hanging constraints and one pin
    cons_p = semquad_constraints(sp, qt)
    pf = Dict{Int,Float64}()
    pfree0 = setdiff(1:np, collect(keys(cons_p)))
    isempty(pfree0) || (pf[first(pfree0)] = 0.0)
    p0, Cp, _ = _affine_constraint(np, cons_p, pf)
    # reduced saddle system: K u - B' p = f, -B u = 0 (the weak form has
    # -int p div v - int q div u, so the pressure blocks carry a minus sign)
    Fp = nothing
    if linear === :schur_cg && schur_precond === :mass
        Mfull = assemble_semquad_pressure_mass(sp)
        Mred = Cp' * Mfull * Cp
        Fp = cholesky(Symmetric(Mred))
        issuccess(Fp) ||
            throw(ArgumentError("pressure mass preconditioner is not SPD"))
    end
    CpB = Cp' * B * Cu
    Kred = Cu' * K * Cu
    fr = Cu' * (fv - K * u0 + B' * p0)
    gr = Cp' * (B * u0)
    nf = size(Cu, 2); nfp = size(Cp, 2)
    Bs = -CpB
    Bt = sparse(transpose(Bs))
    nit = 0
    LAST_SEMSTOKES_SHIFT[] = 0.0
    LAST_SEMSTOKES_REFINE_ITERS[] = 0
    LAST_SEMSTOKES_LADDER[] = nothing
    LAST_SEMSTOKES_LADDER_PHASE[] = nothing
    if linear === :schur_cg
        # Saddle [Kred (-CpB'); (-CpB) 0] with Bs = -CpB, Bt = Bs':
        #   Kred u + Bt p = fr,   Bs u = gr,
        # solved by a matrix-free Schur CG on S = Bs Kred^{-1} Bt.
        # A constrained/hanging velocity mode can leave a zero eigenvalue in
        # Kred after AMR; regularize with a 1e-12 relative diagonal shift so the
        # sparse LU is defined.  The shift is below the G44 exactness tolerance.
        dmax = maximum(abs, diag(Kred))
        Kred = Kred + (1.0e-12 * max(dmax, 1.0)) * I
        # Row/column equilibration: wall-refined AMR trees have h_min/h_max
        # ~1e-4 and Kred condition numbers ~1e12; the scaled LU is much better
        # conditioned and removes the artificial large velocity from the shift.
        d = sqrt.(max.(diag(Kred), 1.0e-14 * dmax))
        Dinv = spdiagm(0 => 1.0 ./ d)
        Kscaled = Dinv * Kred * Dinv
        Fv = lu(Kscaled)
        ufree = Dinv * (Fv \ (Dinv * fr))
        r = Bs * ufree - gr
        pfree = zeros(Float64, nfp)
        z = Fp === nothing ? copy(r) : Fp \ r
        pp = copy(z)
        rho = dot(r, z)
        bnorm = max(dot(r, r), 1.0e-300)
        for it in 1:schur_maxiter
            nit = it
            Ap = Bs * (Dinv * (Fv \ (Dinv * (Bt * pp))))
            den = dot(pp, Ap)
            den == 0.0 && break
            alpha = rho / den
            pfree .+= alpha .* pp
            r .-= alpha .* Ap
            rs = dot(r, r)
            rs <= 1.0e-18 * bnorm && break
            z = Fp === nothing ? copy(r) : Fp \ r
            rho_new = dot(r, z)
            pp .= z .+ (rho_new / rho) .* pp
            rho = rho_new
        end
        ufree = Dinv * (Fv \ (Dinv * (fr - Bt * pfree)))
    else
        S = [Kred Bt; Bs spzeros(nfp, nfp)]
        rhs = vcat(fr, gr)
        if linear === :lagged
            # S2: reuse the previous LU as a preconditioner and polish with
            # iterative refinement; refactor only on a cache miss / stall.
            key = _saddle_key(qt, ss.su.p, bc, nf, nfp)
            sol, rn = _lagged_solve!(S, rhs, key; tol=1.0e-8, maxit=25)
            LAST_SEMSTOKES_REGULARIZED[] = false
            if rn > 1.0e-6
                dmax = maximum(abs, diag(S)) + 1.0
                Ssh = S + (1.0e-12 * dmax) * I
                sol, rn = _lagged_solve!(Ssh, rhs, key; tol=1.0e-8, maxit=6,
                                         force_refactor=true)
                LAST_SEMSTOKES_REGULARIZED[] = true
            end
        else
            sol = S \ rhs
            # See `solve_semquad_stokes_var`: the unshifted LU can be unstable on
            # free-slip wall trees; retry with a 1e-12 relative diagonal shift only
            # when the first solve leaves a bad residual.
            rden0 = max(norm(fr), norm(gr), 1.0)
            bad = norm(Kred * sol[1:nf] + Bt * sol[(nf + 1):(nf + nfp)] - fr) >
                  1.0e-8 * rden0 ||
                  norm(Bs * sol[1:nf] - gr) > 1.0e-8 * rden0
            LAST_SEMSTOKES_REGULARIZED[] = bad
            if bad
                dmax = maximum(abs, diag(S)) + 1.0
                sol = (S + (1.0e-12 * dmax) * I) \ rhs
            end
        end
        ufree = sol[1:nf]
        pfree = sol[nf+1:end]
    end
    # True reduced-saddle residual of the system that was actually solved
    # (Schur uses the regularized Kred).  This is what distinguishes a
    # converged solve from a singular/ill-conditioned direct solve.
    r1 = Kred * ufree + Bt * pfree - fr
    r2 = Bs * ufree - gr
    rden = max(norm(fr), norm(gr), 1.0)
    LAST_SEMSTOKES_RESIDUAL[] = max(norm(r1), norm(r2)) / rden
    LAST_SEMSTOKES_SCHUR_ITERS[] = nit
    u = u0 + Cu * ufree
    pv = p0 + Cp * pfree
    return u, pv, fixed, cons2, cons_p
end

"""Evaluate a quadtree velocity/pressure field at (x,y)."""
function semstokes_eval(ss::SemStokes, u::Vector{Float64}, pv::Vector{Float64},
                        x::Float64, y::Float64)
    nu = ss.su.N
    n = ss.p + 1
    rule = gll_rule(ss.p)
    for lf in ss.su.leaves
        hx = leaf_hx(ss.su, lf[1]); hy = leaf_hy(ss.su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        if x >= x0 - 1.0e-14 && x <= x0 + hx + 1.0e-14 &&
           y >= y0 - 1.0e-14 && y <= y0 + hy + 1.0e-14
            xi = 2.0 * (x - x0) / hx - 1.0
            eta = 2.0 * (y - y0) / hy - 1.0
            Lx, _ = lagrange_eval(rule, xi)
            Ly, _ = lagrange_eval(rule, eta)
            ids = ss.su.elem[lf]
            ux = 0.0; uz = 0.0
            for j in 1:n, i in 1:n
                g = ids[(j - 1) * n + i]
                ux += u[g] * Lx[i] * Ly[j]
                uz += u[nu + g] * Lx[i] * Ly[j]
            end
            q = ss.q
            nq = q + 1
            rq = gll_rule(q)
            Lxq, _ = lagrange_eval(rq, xi)
            Lyq, _ = lagrange_eval(rq, eta)
            pids = ss.sp.elem[lf]
            pp = 0.0
            for b in 1:nq, a in 1:nq
                pp += pv[pids[(b - 1) * nq + a]] * Lxq[a] * Lyq[b]
            end
            return ux, uz, pp
        end
    end
    throw(ArgumentError("point ($x,$y) not in the tree"))
end

# ---------------------------------------------------------------------------
# S4 stage 1: leaf-level parallel assembly
#
# The Stokes assembler walks `su.leaves` in tree order, accumulates one COO
# triplet list (K, optionally Kgeo/Kp, plus B) and one load vector.  That loop
# is embarrassingly parallel over leaves.  `assemble_threads > 1` cuts the leaf
# list into that many *fixed contiguous* blocks, assembles each block into its
# own COO buffers (one per block), and then concatenates the block buffers in
# block-index order.  The result is deterministic run to run: the leaf->block
# assignment and the merge order do not depend on the scheduler.
#
# It is deterministic run to run: the leaf->block assignment and the merge order
# do not depend on the scheduler.  Because the blocks are contiguous and merged
# in block order, the concatenated COO sequence is exactly the serial leaf
# sequence, so K/Kgeo/Kp/B come out bitwise equal to the serial path (measured;
# G181 only requires <=1e-15 relative).  The one place the summation order does
# change is the load vector: each block accumulates its own `fv` and the block
# sums are then added in order, so `fv` (and hence `fr` and the solution) can
# differ at round-off.  `assemble_threads=1` (the default) keeps the whole path
# bit for bit, so the G150 snapshot still pins direct-solver round-off.
# ---------------------------------------------------------------------------

struct SemQuadAsmScratch
    phi::Vector{Float64}
    gx::Vector{Float64}
    gz::Vector{Float64}
    Kloc::Matrix{Float64}
    Kgloc::Matrix{Float64}
    Bloc::Matrix{Float64}
    floc::Vector{Float64}
    Kploc::Matrix{Float64}
end

struct SemQuadAsmBuf
    IK::Vector{Int}
    JK::Vector{Int}
    VK::Vector{Float64}
    IB::Vector{Int}
    JB::Vector{Int}
    VB::Vector{Float64}
    IG::Vector{Int}
    JG::Vector{Int}
    VG::Vector{Float64}
    IP::Vector{Int}
    JP::Vector{Int}
    VP::Vector{Float64}
    fv::Vector{Float64}
    scr::SemQuadAsmScratch
end

# Everything the leaf kernel needs that is not a leaf or a buffer.
#
# `mf_path` is the matrix-free branch: the leaf loop then skips the local
# stiffness block **and** the viscosity closure (both only feed `K`/`Kgeo`/
# `Kp`), so the assembler produces `B`, `fv` and the constraint-free geometry
# data without paying for the operator the caller never stores or forms.
#
# `geo_coef`/`dp_coef` are **type parameters**, not `Union{Nothing,Function}`
# fields.  A function *field* is not specialised (docs/S3_DESIGN.md §17: the
# `semquad_rho` bug), so `geo_coef(x,y,edot)` inferred as `Any` and the Kgeo
# accumulation `Kgloc[k,l] += gc * Tux_k * Dux_l * w` was a boxed multiplication
# inside the innermost `(k,l)` loop: measured 13.0 of the 15.0 ms of the Kgeo
# branch and 336 MB of the 831 MB allocated by 3 Newton steps (docs/
# NEWTON_AUDIT.md).  With the parameter the call is concrete, `gc::Float64` and
# the loop is monomorphic; the arithmetic (and therefore the result) is
# unchanged, G202 pins it bitwise against the pre-fix worktree.
struct SemQuadAsmCfg{G,D}
    a2::Float64
    deviatoric::Bool
    mf_path::Bool
    geo_u::Union{Nothing,Vector{Float64}}
    geo_coef::G
    geo_spd_safety::Float64
    dp_u::Union{Nothing,Vector{Float64}}
    dp_pfield::Union{Nothing,Vector{Float64}}
    dp_coef::D
end

function _semquad_asm_buf(nleaf::Int, nv::Int, npq::Int, Nu2::Int;
                          has_K::Bool, has_geo::Bool, has_dp::Bool)
    IK = Int[]; JK = Int[]; VK = Float64[]
    if has_K
        # Exact counts (4*nv^2 velocity entries per leaf); this is purely an
        # allocation hint, the push! order is unchanged.
        sizehint!(IK, 4 * nv * nv * nleaf)
        sizehint!(JK, 4 * nv * nv * nleaf)
        sizehint!(VK, 4 * nv * nv * nleaf)
    end
    IB = Int[]; JB = Int[]; VB = Float64[]
    sizehint!(IB, 2 * npq * nv * nleaf)
    sizehint!(JB, 2 * npq * nv * nleaf)
    sizehint!(VB, 2 * npq * nv * nleaf)
    IG = Int[]; JG = Int[]; VG = Float64[]
    if has_geo
        sizehint!(IG, 4 * nv * nv * nleaf)
        sizehint!(JG, 4 * nv * nv * nleaf)
        sizehint!(VG, 4 * nv * nv * nleaf)
    end
    IP = Int[]; JP = Int[]; VP = Float64[]
    if has_dp
        sizehint!(IP, 2 * npq * nv * nleaf)
        sizehint!(JP, 2 * npq * nv * nleaf)
        sizehint!(VP, 2 * npq * nv * nleaf)
    end
    n2v = 2 * nv
    scr = SemQuadAsmScratch(zeros(Float64, nv), zeros(Float64, nv),
                            zeros(Float64, nv),
                            zeros(Float64, n2v, n2v),
                            zeros(Float64, n2v, n2v),
                            zeros(Float64, npq, n2v),
                            zeros(Float64, n2v),
                            zeros(Float64, n2v, npq))
    return SemQuadAsmBuf(IK, JK, VK, IB, JB, VB, IG, JG, VG, IP, JP, VP,
                         zeros(Float64, Nu2), scr)
end

# Assemble leaves `lo:hi` into `buf` (the buffers are per block, so this is
# safe to run concurrently on disjoint leaf ranges).
function _semquad_assemble_block!(buf::SemQuadAsmBuf, cfg::SemQuadAsmCfg,
                                  su::SemQuad, sp::SemQuad,
                                  leaves::Vector{NTuple{3,Int}}, lo::Int, hi::Int,
                                  eta_fun::Function, fx::Function, fz::Function,
                                  fd::FEData, pleaves::Set{NTuple{3,Int}})
    hi < lo && return buf
    n = su.p + 1
    q = sp.p
    nv = n * n
    npq = (q + 1)^2
    nu = su.N
    a2 = cfg.a2
    deviatoric = cfg.deviatoric
    mf_path = cfg.mf_path
    geo_u = cfg.geo_u
    geo_coef = cfg.geo_coef
    geo_spd_safety = cfg.geo_spd_safety
    dp_u = cfg.dp_u
    dp_pfield = cfg.dp_pfield
    dp_coef = cfg.dp_coef
    IK = buf.IK; JK = buf.JK; VK = buf.VK
    IB = buf.IB; JB = buf.JB; VB = buf.VB
    IG = buf.IG; JG = buf.JG; VG = buf.VG
    IP = buf.IP; JP = buf.JP; VP = buf.VP
    fv = buf.fv
    scr = buf.scr
    phi = scr.phi; gx = scr.gx; gz = scr.gz
    Kloc = scr.Kloc; Kgloc = scr.Kgloc; Bloc = scr.Bloc; floc = scr.floc
    Kploc = scr.Kploc
    for it in lo:hi
        lf = leaves[it]
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = su.elem[lf]
        pids = sp.elem[lf]
        fill!(Kloc, 0.0); fill!(Kgloc, 0.0); fill!(Bloc, 0.0); fill!(floc, 0.0)
        if dp_u !== nothing
            fill!(Kploc, 0.0)
        end
        for qy in 1:fd.nq, qx in 1:fd.nq
            x = x0 + 0.5 * (fd.gl_x[qx] + 1.0) * hx
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            # The viscosity enters the leaf assembly only through `K` and the
            # two Jacobian blocks, all of which are skipped on the matrix-free
            # path (`mf_path`): the operator's own eta field
            # (`semquad_eta_field`) samples the same points.  Evaluating the
            # closure here anyway cost 0.15 s of the F setup at one thread
            # (131 008 calls, measured; docs/dev/MG.md).  `etaq` stays
            # defined so the `K`-forming branches below compile unchanged.
            etaq = 0.0
            if !mf_path
                etaq = eta_fun(x, y)
            end
            for j in 1:n, i in 1:n
                k = (j - 1) * n + i
                phi[k] = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                gx[k] = fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / hx)
                gz[k] = fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hy)
            end
            # Performance (ported from Route D): fx/fz are spatially varying but
            # constant over the element basis loop; evaluating them once per
            # quadrature point instead of 2*nv times cuts the closure calls from
            # 1.4M to 158k on the 8800-leaf tree (measured ~0.5 s -> ~0.05 s).
            # Identical values.
            fxv = fx(x, y)
            fzv = fz(x, y)
            for k in 1:nv
                floc[k] += w * fxv * phi[k]
                floc[nv + k] += w * fzv * phi[k]
            end
            # Local stiffness block.  Four independent accumulations over the
            # (k, l) pairs.  The *inner* loop index is `k`, so every
            # `Kloc[...]` store and every `gx[k]`/`gz[k]` load is unit-stride
            # (the transposed order used to store with stride 2*nv and could
            # not vectorise).  The quadrature loop stays outermost, so each
            # entry still accumulates over q in the same order: the result is
            # bitwise identical (G188), and the per-q scalars (`etaq*w`,
            # `a2`-scaled gradients) are re-associated only where the
            # left-to-right expression tree is unchanged.
            if !mf_path
                if deviatoric
                    for l in 1:nv
                        xl = gx[l]; zl = gz[l]
                        @simd for k in 1:nv
                            Kloc[k, l] += etaq * (a2 * gx[k] * xl + gz[k] * zl) * w
                            # full 2 eta eps:eps(v) minus (2/3) eta div(u)div(v);
                            # the off-diagonal correction is a DIFFERENT product
                            # (gx_k gz_l), not a rescaling of gz_k gx_l.
                            Kloc[k, nv + l] += etaq * (gz[k] * xl -
                                                       (2.0 / 3.0) * gx[k] * zl) * w
                            Kloc[nv + k, l] += etaq * (gx[k] * zl -
                                                       (2.0 / 3.0) * gz[k] * xl) * w
                            Kloc[nv + k, nv + l] += etaq * (a2 * gz[k] * zl + gx[k] * xl) * w
                        end
                    end
                else
                    for l in 1:nv
                        xl = gx[l]; zl = gz[l]
                        @simd for k in 1:nv
                            Kloc[k, l] += etaq * (a2 * gx[k] * xl + gz[k] * zl) * w
                            Kloc[k, nv + l] += etaq * (gz[k] * xl) * w
                            Kloc[nv + k, l] += etaq * (gx[k] * zl) * w
                            Kloc[nv + k, nv + l] += etaq * (a2 * gz[k] * zl + gx[k] * xl) * w
                        end
                    end
                end
            end
            if geo_u !== nothing
                # Geometric (strain-rate) part of the Newton Jacobian:
                #   Kgeo = sum_q w (deta/dedot)/edot  T(q) T(q)'
                # with T the symmetric strain-rate operator acting on the
                # velocity basis.  It is kept separate from Kloc so the
                # residual `fv - K*u0` still uses the true (non-geometric) K.
                duxdx = 0.0; duxdy = 0.0; duzdx = 0.0; duzdy = 0.0
                for k in 1:nv
                    duxdx += geo_u[uids[k]] * gx[k]
                    duxdy += geo_u[uids[k]] * gz[k]
                    duzdx += geo_u[nu + uids[k]] * gx[k]
                    duzdy += geo_u[nu + uids[k]] * gz[k]
                end
                exd = duxdx; ezd = duzdy
                exz = 0.5 * (duxdy + duzdx)
                edot = sqrt(max(0.5 * (exd * exd + ezd * ezd +
                                       2.0 * exz * exz), 0.0))
                deta_q = edot > 0.0 ? geo_coef(x, y, edot) : 0.0
                # ASPECT `Utilities::compute_spd_factor` (stabilization): with a
                # strongly weakening law (Drucker-Prager plastic branch has
                # deta/dedot = -eta/edot) the geometric term can cancel the
                # viscous modulus and leave the velocity block singular.
                # `geo_spd_safety > 0` scales Kgeo by alpha in (0,1] so that
                # 2 eta + alpha * (deviator) stays SPD; 0 keeps G156's
                # unstabilized operator bit for bit.
                if geo_spd_safety > 0.0 && deta_q != 0.0
                    d_a = spd_factor(etaq, deta_q, edot, exd, ezd, exz,
                                     exd, ezd, exz; safety=geo_spd_safety)
                else
                    d_a = 1.0
                end
                gc = edot > 0.0 ? d_a * deta_q / edot : 0.0
                if gc != 0.0
                    for k in 1:nv
                        Tux_k = exd * gx[k] + exz * gz[k]
                        Tuz_k = ezd * gz[k] + exz * gx[k]
                        for l in 1:nv
                            Dux_l = exd * gx[l] + exz * gz[l]
                            Duz_l = ezd * gz[l] + exz * gx[l]
                            Kgloc[k, l] += gc * Tux_k * Dux_l * w
                            Kgloc[k, nv + l] += gc * Tux_k * Duz_l * w
                            Kgloc[nv + k, l] += gc * Tuz_k * Dux_l * w
                            Kgloc[nv + k, nv + l] += gc * Tuz_k * Duz_l * w
                        end
                    end
                end
            end
            if dp_u !== nothing
                # Pressure-derivative block: Kp_{i,a} = int 2 (deta/dp)
                # eps(u):eps(v_i) phi_a.  The strain rate is the one that
                # enters the viscosity, i.e. of `dp_u` at this quadrature point.
                duxdx = 0.0; duxdy = 0.0; duzdx = 0.0; duzdy = 0.0
                for k in 1:nv
                    duxdx += dp_u[uids[k]] * gx[k]
                    duxdy += dp_u[uids[k]] * gz[k]
                    duzdx += dp_u[nu + uids[k]] * gx[k]
                    duzdy += dp_u[nu + uids[k]] * gz[k]
                end
                dp_exd = duxdx; dp_ezd = duzdy
                dp_exz = 0.5 * (duxdy + duzdx)
                dp_edot = sqrt(max(0.5 * (dp_exd * dp_exd + dp_ezd * dp_ezd +
                                          2.0 * dp_exz * dp_exz), 0.0))
                dp_p = _semquad_eval_at(sp, dp_pfield, x, y, pleaves)
                dp_c = dp_coef(x, y, dp_edot, dp_p)
                if dp_c != 0.0
                    for b in 1:(q + 1), a in 1:(q + 1)
                        lp = (b - 1) * (q + 1) + a
                        psp = fd.phi_p[a, qx] * fd.phi_p[b, qy]
                        for k in 1:nv
                            Kploc[k, lp] += 2.0 * dp_c * psp *
                                            (dp_exd * gx[k] + dp_exz * gz[k]) * w
                            Kploc[nv + k, lp] += 2.0 * dp_c * psp *
                                                 (dp_ezd * gz[k] + dp_exz * gx[k]) * w
                        end
                    end
                end
            end
            for b in 1:(q + 1), a in 1:(q + 1)
                l = (b - 1) * (q + 1) + a
                ps = fd.phi_p[a, qx] * fd.phi_p[b, qy]
                for k in 1:nv
                    Bloc[l, k] += ps * gx[k] * w
                    Bloc[l, nv + k] += ps * gz[k] * w
                end
            end
        end
        for k in 1:nv
            g = uids[k]
            fv[g] += floc[k]
            fv[nu + g] += floc[nv + k]
            mf_path && continue
            for l in 1:nv
                gl = uids[l]
                push!(IK, g); push!(JK, gl); push!(VK, Kloc[k, l])
                push!(IK, nu + g); push!(JK, nu + gl); push!(VK, Kloc[nv + k, nv + l])
                push!(IK, g); push!(JK, nu + gl); push!(VK, Kloc[k, nv + l])
                push!(IK, nu + g); push!(JK, gl); push!(VK, Kloc[nv + k, l])
                if geo_u !== nothing
                    push!(IG, g); push!(JG, gl); push!(VG, Kgloc[k, l])
                    push!(IG, nu + g); push!(JG, nu + gl); push!(VG, Kgloc[nv + k, nv + l])
                    push!(IG, g); push!(JG, nu + gl); push!(VG, Kgloc[k, nv + l])
                    push!(IG, nu + g); push!(JG, gl); push!(VG, Kgloc[nv + k, l])
                end
            end
        end
        for l in 1:npq
            pg = pids[l]
            for k in 1:nv
                push!(IB, pg); push!(JB, uids[k]); push!(VB, Bloc[l, k])
                push!(IB, pg); push!(JB, nu + uids[k]); push!(VB, Bloc[l, nv + k])
            end
        end
        if dp_u !== nothing
            for l in 1:npq
                pg = pids[l]
                for k in 1:nv
                    push!(IP, uids[k]); push!(JP, pg); push!(VP, Kploc[k, l])
                    push!(IP, nu + uids[k]); push!(JP, pg)
                    push!(VP, Kploc[nv + k, l])
                end
            end
        end
    end
    return buf
end

# Split the leaves into `nt` contiguous blocks, assemble them (in parallel when
# Julia has more than one thread) and merge the block buffers in block order.
# `nt == 1` is the historical serial path: one buffer over all leaves, returned
# as-is (no copy, no re-ordering).
function _semquad_assemble_blocks(nt::Int, cfg::SemQuadAsmCfg,
                                  su::SemQuad, sp::SemQuad,
                                  eta_fun::Function, fx::Function, fz::Function,
                                  fd::FEData, pleaves::Set{NTuple{3,Int}})
    nleaf = length(su.leaves)
    n = su.p + 1
    nv = n * n
    npq = (sp.p + 1)^2
    Nu2 = 2 * su.N
    has_K = !cfg.mf_path
    has_geo = cfg.geo_u !== nothing
    has_dp = cfg.dp_u !== nothing
    if nt <= 1
        buf = _semquad_asm_buf(nleaf, nv, npq, Nu2; has_K=has_K,
                               has_geo=has_geo, has_dp=has_dp)
        _semquad_assemble_block!(buf, cfg, su, sp, su.leaves, 1, nleaf,
                                 eta_fun, fx, fz, fd, pleaves)
        return (IK=buf.IK, JK=buf.JK, VK=buf.VK,
                IB=buf.IB, JB=buf.JB, VB=buf.VB,
                IG=buf.IG, JG=buf.JG, VG=buf.VG,
                IP=buf.IP, JP=buf.JP, VP=buf.VP,
                fv=buf.fv)
    end
    bufs = Vector{SemQuadAsmBuf}(undef, nt)
    los = Vector{Int}(undef, nt)
    his = Vector{Int}(undef, nt)
    base, rem = divrem(nleaf, nt)
    off = 1
    for t in 1:nt
        cnt = base + (t <= rem ? 1 : 0)
        los[t] = off
        his[t] = off + cnt - 1
        off += cnt
        bufs[t] = _semquad_asm_buf(cnt, nv, npq, Nu2; has_K=has_K,
                                   has_geo=has_geo, has_dp=has_dp)
    end
    Threads.@threads :static for t in 1:nt
        _semquad_assemble_block!(bufs[t], cfg, su, sp, su.leaves, los[t],
                                 his[t], eta_fun, fx, fz, fd, pleaves)
    end
    fv = zeros(Float64, Nu2)
    for t in 1:nt
        fv .+= bufs[t].fv
    end
    IK = has_K ? _semquad_concat_blocks([b.IK for b in bufs]) : Int[]
    JK = has_K ? _semquad_concat_blocks([b.JK for b in bufs]) : Int[]
    VK = has_K ? _semquad_concat_blocks([b.VK for b in bufs]) : Float64[]
    IB = _semquad_concat_blocks([b.IB for b in bufs])
    JB = _semquad_concat_blocks([b.JB for b in bufs])
    VB = _semquad_concat_blocks([b.VB for b in bufs])
    IG = has_geo ? _semquad_concat_blocks([b.IG for b in bufs]) : Int[]
    JG = has_geo ? _semquad_concat_blocks([b.JG for b in bufs]) : Int[]
    VG = has_geo ? _semquad_concat_blocks([b.VG for b in bufs]) : Float64[]
    IP = has_dp ? _semquad_concat_blocks([b.IP for b in bufs]) : Int[]
    JP = has_dp ? _semquad_concat_blocks([b.JP for b in bufs]) : Int[]
    VP = has_dp ? _semquad_concat_blocks([b.VP for b in bufs]) : Float64[]
    return (IK=IK, JK=JK, VK=VK, IB=IB, JB=JB, VB=VB,
            IG=IG, JG=JG, VG=VG, IP=IP, JP=JP, VP=VP, fv=fv)
end

"""
    solve_semquad_stokes_var(ss, qt, eta_fun, fx, fz, uDx, uDz)

Variable-viscosity quadtree Taylor-Hood Stokes solve.  The weak form is

    int 2 eta(x,y) eps(u):eps(v) - p div(v) - q div(u) = int f.v,

with `eta_fun(x,y)` evaluated at the Gauss points, affine Dirichlet data
`u = uD` on the domain boundary, the same hanging-node constraints as the
scalar space (lifted to both velocity components), and one pinned pressure dof.
The element matrices are assembled with `make_fedata(p, p-1)` Gauss quadrature,
matching the uniform-mesh `assemble_stokes` convention.  With
`linear=:schur_cg`, `schur_precond=:mass` uses the pressure mass matrix
(spectrally equivalent to the Schur complement) and `schur_maxiter` caps the
PCG iterations; `LAST_SEMSTOKES_RESIDUAL` records the true reduced-saddle
residual and `LAST_SEMSTOKES_SCHUR_ITERS` the iteration count.

`linear=:auto` (S6, production) resolves to `:direct` or `:mg` **before the
assembly** from the current saddle, so nobody has to remember the rule:
`saddle_dofs = 2*Nu + Np` (node counts) and the viscosity contrast sampled on
the actual `eta_fun` (`semquad_eta_contrast`, fixed 33x33 grid), fed to
[`semquad_choose_linear`](@ref).  An MG decision also forces
`mg_operator=:matrixfree` and `mg_threads=min(8, Threads.nthreads())`
(`semquad_auto_mg_threads`); `schur_precond` must be `:mass` (the default) or
the selector falls back to `:direct`.  The decision and its reason are in
`LAST_SEMSTOKES_AUTO_CHOICE`.  The rule is re-calibrated at `c7e0f55` after the
MG optimisation campaign (docs/dev/MG.md§16): any contrast takes MG from
20k dofs, contrast <= 100 from 2k, direct below that -- the old contrast-10
rung is gone because the measurements cannot tell the two low bands apart
(`julia/scripts/bench_auto_crossover.jl` is the A/B; docs/dev/SOLVER.md §4.2/§4.4
carry the table and the old, pre-campaign numbers, which are superseded).
An `:auto` solve that resolves to `:direct` produces the **bitwise** direct
result; an `:auto` solve that resolves to `:mg` produces the same answer as an
explicit `linear=:mg` call (<=1e-10; the MG solve is not bitwise vs direct).
`solve_semquad_stokes_var` itself keeps `:direct` as its default (the
`run_semquad_simulation` default became `:auto` on 2026-11-13); pass `:auto`
explicitly here to use the selector.

`linear=:mg` is the scalable path for a **uniform** quadtree: geometric
multigrid for the velocity block (level coarsening, rediscretised eta) and a
block-triangular right preconditioner for GMRES.  `mg_cycles` V-cycles per
preconditioner application (default 1 since S3/MF-4: at p=4 L4 one V-cycle per
outer iteration needs 26 outer iterations but only 26 V-cycles in total, while
two need 21 but 42 -- measured 0.14 s vs 0.23 s for the whole saddle solve),
`mg_tol` the outer tolerance (1e-10).

`geo_u`/`geo_coef` add the strain-rate Jacobian `Kgeo` to the left-hand side
(Newton JVP; G156), and `dp_u`/`dp_pfield`/`dp_coef` add the pressure-derivative
block `(Kp)_{i,a} = int 2 (deta/dp) eps(u):eps(v_i) phi_a` returned as `Kp`
(`2N x np`) and `Kp_red = Cu' Kp Cp` (G163).  The Drucker-Prager saddle is then
non-symmetric: the momentum block is `Bt + Kp_red` while the constraint rows
have no pressure dependence.  Both options are only implemented for
`deviatoric=false`, and both default to "off", which keeps the assembly bit
identical to the historical one.

The direct branch checks its reduced residual and retries with a `1e-12 *
max|diag(S)|` diagonal shift when the unshifted LU is unstable (wall-refined
free-slip trees; `LAST_SEMSTOKES_REGULARIZED` records which path ran -- the flag
is only touched by the `:direct` branch).  `stokes_accept_tol` (default `Inf`)
turns that single retry into the shift ladder of
[`_shift_ladder_solve`](@ref): with a finite tolerance -- the production driver
passes its `stokes_tol` -- a rung that still misses it escalates to `1e-10` and
then `1e-8` relative shift, and a residual that survives the last rung is
polished with a few iterative-refinement steps against the unshifted operator.
`Inf` accepts the first rung unconditionally, which is the historical behaviour
bit for bit.  `stokes_lost_tol` (default `1.0`) is the second trigger of the
ladder: an unshifted relative residual above it means the factorisation did not
solve the system at all, and the diagonal shift then has to be chosen at the
scale of the near-null cluster rather than far above it
([`SEMSTOKES_TINY_SHIFTS`](@ref); docs/dev/CASE_REALISTIC.md §13.10).  `Inf`
disables that phase, and it never fires on a solve whose unshifted residual is
below `1.0`, so every answer that was already usable is bitwise unchanged.
The rung that produced the returned solution is in
`LAST_SEMSTOKES_SHIFT`, the refinement count in `LAST_SEMSTOKES_REFINE_ITERS`,
the per-rung residual history in `LAST_SEMSTOKES_LADDER`, and the two saddle
blocks separately in `LAST_SEMSTOKES_RESID_SPLIT` (docs/dev/CASE_REALISTIC.md §13).
`stokes_scaling` (default `:none`) selects the symmetric equilibration of the
reduced saddle used by the direct branch (`:equilibrated`;
docs/PRESSURE_EQUILIBRATION.md).  The scaling is an exact congruence
`S -> D*S*D`, `rhs -> D*rhs` with
`D = diag(1/sqrt(diag Kred), 2 max|Kred| / max(|Bt|,|Bs|))` and `x = D*y`: it
removes the viscosity and element-size scales from the operator, so the
wall-strip pressure family is lifted far above the round-off floor (the
measured smallest eigenvalue of the scaled operator is a momentum mode at
`4.2e-11` relative, instead of the unscaled `1.7e-14` pressure mode) and the
*unshifted* LU resolves it (measured 6.0e-10 on the AMR tree and 3.6e-8 on the
uniform-L5 control).  It is engaged only when
the historical unscaled unshifted solve misses its `1e-8`/`accept_tol` health
check, so a healthy solve is bitwise identical even with the option on; the
scaled answer is accepted when its true residual is within the 1e-6 accuracy
goal and is reported with `LAST_SEMSTOKES_LADDER_PHASE[] = :equilibrated` and
zero shift.  Otherwise the historical unscaled shift ladder runs unchanged as
the fallback.  `:none` is the historical path bit for bit, and the scaling is
never applied to the `:mg`/`:lagged`/`:schur_cg` paths.  The number of solves
answered by the scaled path is counted in `LAST_SEMSTOKES_SCALED[]` (cumulative;
not reset by a solve).  `run_semquad_simulation` exposes the keyword as
`stokes_scaling`, forwards it to every solve and stores it in the checkpoint's
`run_config` (docs/PRESSURE_EQUILIBRATION.md).
`mg_pressure_mode` selects the Schur approximation: `:auto` (default) uses the
1/eta-weighted pressure mass for contrast <= 100 and the three-factor w-BFBT
above it; `:mass`, `:stiffness`, `:lsc` and `:bfbt` force one.
`mg_pressure_precond` is the public selector of the pressure-block
approximate inverse: `:auto` (default) keeps the `mg_pressure_mode` resolution
above, `:bfbt` is the three-factor w-BFBT, `:mass` the *unweighted* pressure
mass matrix, `:scaled_mass` the `mg_pressure_weight`-scaled one (`:inv_eta` by
default), `:pcd` pressure convection-diffusion (`A_p^{-1} F_p M_p^{-1}`; with no
advection term in the momentum equation `C_p = 0`, so it reduces to the mass
inverse at three times the cost) and `:lsc` the least-squares commutator.  On
the 1024-leaf blob tree (16384x contrast) the measured outer iteration counts
are 39 (`:bfbt`) / 68 (`:scaled_mass`) / 239 (`:lsc`) / not converged
(`:mass`, `:pcd`), which is why `:auto` keeps BFBT there (docs/S3_DESIGN.md §13).
`mg_pressure_scale=0` means the mode's default (1 for mass, 0.1 for LSC,
0.4 for BFBT; measured scale curve in docs/S3_DESIGN.md §13).  `mg_restart`/`mg_maxit` control GMRES (default 100/500).  The outer loop is
**flexible** GMRES (FGMRES, `fgmres_restarted`) and stops on the **true**
reduced-saddle residual, never on the preconditioned one.
`mg_pressure_cache=true` (S3 stage A) caches the pressure-block factors across
Stokes solves, keyed by the tree fingerprint and invalidated when the sampled
viscosity moves by more than `mg_pressure_eta_tol` (default **1.0** since
S5-1; S3 used 1e-2 and then 0.5).  The BFBT build costs ~0.2-1.1 s on the
1024-leaf blob tree, and on the measured threshold curve (docs/S3_DESIGN.md
§13.3) the outer iteration count is flat from 1e-12 to `Inf` (39.7 -> 39.9),
so the threshold only trades build time against staleness: at **1.0** every
tree block is built exactly once (25 builds instead of 41 in the 25-frame blob
run), at the cost of accepting up to 100% eta drift inside a tree block.
`mg_pressure_eta_tol=Inf` is supported and means "never rebuild on eta drift"
(the tree fingerprint still forces a rebuild on every AMR change); it is only
safe for smoothly evolving viscosity -- **no phase change and no yield onset
inside a tree block** -- because nothing then bounds the staleness (§19.1).
`mg_operator=:matrixfree` (the default since S3/MF-4) builds every hierarchy
level from the leaf-local matrix-free apply of MF-1 and stores no velocity
operator matrix anywhere (44-79x faster hierarchy build, 1.26-1.48x faster
V-cycle at p=4); `:sparse` keeps the historical rediscretised/Galerkin `A[k]`.
`mg_vel_solver=:vcycle` (the default since S3/MF-6) solves the SPD velocity
block with the historical fixed `mg_cycles` V-cycles -- a *linear*
preconditioner, so the flexible outer GMRES reduces to ordinary GMRES there.
`:cg` is the MF-4 scheme that solves the velocity block inside the outer
iteration with CG on `Kred` preconditioned by the hierarchy (the
deal.II/ASPECT FGMRES + CG+MG scheme), controlled by `mg_vel_tol`/`mg_vel_maxit`
(1e-2/10).  Measured on the 1024-leaf blob tree: `:vcycle` needs 40 outer
iterations and 40 V-cycles, `:cg` needs 25 outer and 84 inner-CG (=V-cycle)
iterations, i.e. `:vcycle` is 33% faster per Stokes solve at 13 ms/V-cycle,
which is why it is the default; `:cg` remains available and is exercised by
G177.
`mg_smoother=:auto` uses damped Jacobi (faster per apply; the saddle iteration
count is dominated by the pressure block, and SGS did not reduce it on 2b's
16384x contrast).  `:sgs` is available for experiments; it gives a much better
velocity-block V-cycle but costs 2-3x per apply.  A Chebyshev smoother was
implemented and measured during the tuning pass; it diverges on the two
production trees (the reduced operator has near-null modes outside the
polynomial's window), so it is not shipped (docs/dev/MG.md).
`mg_coarse=:auto` rediscretises on uniform trees and uses the exact Galerkin
operator `P' A P` on non-uniform ones (same iteration count to a few percent,
but no coarse re-assembly per Picard iteration).
`mg_coarsen_layers`/`mg_coarsen_factor` control the h-coarsening of the
hierarchy (`_mg_coarsen_tree`): `1` is the one-tree-layer-per-step rule and
`mg_coarsen_factor` keeps merging until a step coarsens by at least that factor.
`mg_coarsen_factor=nothing` (the default) resolves to **1.5** on the
matrix-free path and `1.0` on the sparse path.  Measured on both production
states: 2.0 removes F's near-duplicate level (8 -> 6 levels, 70 -> 81
iterations, 5.42 -> 3.36 s at one thread) but on P1 it over-coarsens (8 -> 6
levels, 61 -> 177 iterations, 5.45 -> 8.03 s); 1.5 gets the F win (8 -> 6) and
keeps P1's 40 776-dof level (8 -> 7, 65 iterations with the top-heavy 3+3
schedule, 5.22 s).  Uniform and blob trees are a bitwise no-op at 1.5.
`mg_coarse_direct_max=n>0` stops the hierarchy at the finest level with at most
`n` dofs and factorises that level directly instead of coarsening further (also
measured: it costs more than it saves on the blob tree, default 0 = off).
`mg_smoothing=:auto` (the default) uses `mg_pre`/`mg_post` sweeps on every level
(`:uniform`, the historical 2+2) unless the hierarchy is top-heavy (its finest
coarsening step keeps more than half of the fine dofs): at p=2 the default then
rises to uniform 3+3, while at p>=4 it keeps the measured front-loaded
`:taper`.  `mg_omega` (`nothing` = the schedule's own default: 2/3 for
`:uniform`, 0.55 for `:taper`) overrides the damping.
Explicit `mg_pre`/`mg_post` vectors (coarse first) select any per-level
schedule; `mg_smoothing=:taper` keeps the historical `1, 2, 2, 4, 4, 8, ...`
shape with omega=0.55.  Measured: the 6-level F hierarchy is best at uniform
2+2 (81 iterations), the 7-level P1 hierarchy at uniform 3+3 (65); the old
taper gives 99 / 90 (docs/dev/MG.md).
A non-uniform tree is handled by hierarchical coarsening.

`mg_threads=1` (default) keeps the whole `linear=:mg` path serial and bitwise as
before.  `mg_threads=N>1` (S5-2) threads the **leaf-local** work of the
matrix-free V-cycle (the `semquad_apply_K!` kernel, the damped-Jacobi sweeps and
the residual evaluation) and of the outer matrix-free operator, using `N` fixed
contiguous leaf blocks and one scratch buffer per block; the scatter-add across
leaves that share boundary nodes stays serial and runs in leaf order, so the
threaded V-cycle and the whole solve are **bitwise identical** to the serial
ones (G190) -- the thread count changes the wall clock, not a single bit.  Only
`linear=:mg, mg_operator=:matrixfree` is threaded (the sparse path keeps its
serial `mul!`); a request larger than `Threads.nthreads()` is clamped with a
warning and a single-threaded process warns once and stays serial.  Measured on
uniform 4096-leaf (132k velocity dof) trees: 2.6x V-cycle and 1.9x whole-solve
speedup at 8 threads; at 1024 leaves the V-cycle gain is smaller (1.6x) and the
whole solve does not improve (docs/S3_DESIGN.md §19.2).

`assemble_threads=1` (default) assembles the Stokes leaves serially, exactly as
before (bitwise).  `assemble_threads=N>1` cuts the leaf list into `N` fixed
contiguous blocks, gives each block its own COO buffers, assembles the blocks
with `Threads.@threads` over the Julia thread pool and merges the block buffers
in block-index order.  The result is deterministic run to run.  Since block
order equals leaf order, the concatenated COO lists are the serial ones and
K/Kgeo/Kp/B are bitwise identical; only the load vector is assembled as a sum
of block sums, so `fv`/`fr`/the solution can differ at round-off (G181 checks
both the matrices and `fv` against serial to <=1e-15 relative).  If Julia has
only one thread, requesting >1 warns once and falls back to serial.

`linear=:auto` is additionally **safe**: if the selector picks `:mg` but the
solve is not acceptable -- the true reduced-saddle residual is above
`auto_fallback_tol` (keyword, default `1e-5`; `run_semquad_simulation` passes
its `stokes_tol`), or the MG path throws -- the same solve is redone with the
sparse direct path instead of aborting the run.  The event is recorded in
`LAST_AUTO_FALLBACK` (`count`, `last_reason`), warned once per process, and
`LAST_SEMSTOKES_AUTO_CHOICE` still reports the auto decision (`:mg`).  Explicit
`linear=:mg` is untouched and still fails loudly, and an `:auto` solve that
chose `:direct` is bitwise unchanged.  Measured trigger: the 8192-leaf
realistic case (7498 leaves, `saddle_dofs` ~72k), where the matrix-free MG
returned relres 4.33 before the exact constrained and BFBT diagonals; the
direct fallback recovered the solve.  After those fixes the same class of tree
converges (the card comment in `julia/scripts/case_realistic_2d.jl` and
docs/dev/SOLVER.md §4.4 carry the measurement), so the fallback remains a safety
net, not the expected path.

`assemble_only` selects what the call is allowed to build:

* `false` (default) -- solve;
* `true` -- assemble the full system and return it (`K`, `Kred`, `Bs`, `Bt`,
  `fr`, `gr`, ...), as every DWR/Newton/`return_system` caller expects;
* `:mg` -- the **reduced matrix-free MG assembly**: return the same
  `Cu`/`Cp`/`B`/`Bc`/`Bm`/`fv`/`u0`/`p0`/`Bs`/`Bt` but with `K === nothing` and
  `Kred === nothing`, and without evaluating the viscosity closure inside the
  leaf assembly (it only feeds `K`, which is not formed).  `fr` uses the
  matrix-free `K u0` apply, i.e. the same value the production solve uses.
  This is what `_semquad_mg_geometry` needs to build the hierarchy: it reads
  `Cu` only.  `SEMSTOKES_K_ASSEMBLIES` counts every formation of the sparse
  `K`; an `assemble_only=:mg` call does not move it.  The mode requires
  `linear=:mg, mg_operator=:matrixfree` and is bitwise for every object it
  returns (docs/dev/MG.md, G220).
"""
function solve_semquad_stokes_var(ss::SemStokes, qt::Quadtree,
                                  eta_fun::Function,
                                  fx::Function, fz::Function,
                                  uDx::Function, uDz::Function; kwargs...)
    # `linear=:auto` safety net (docstring above).  `assemble_only` never solves,
    # so it has no residual to judge; explicit paths are forwarded untouched.
    is_auto = get(kwargs, :linear, :direct) === :auto
    has_ftol = haskey(kwargs, :auto_fallback_tol)
    ftol = has_ftol ? Float64(kwargs[:auto_fallback_tol]) : 1.0e-5
    core = has_ftol ?
           (; (k => v for (k, v) in pairs(kwargs)
               if k !== :auto_fallback_tol)...) :
           kwargs
    if !is_auto
        return _solve_semquad_stokes_var_core(ss, qt, eta_fun, fx, fz, uDx, uDz;
                                              core...)
    end
    # Reset the selector record so a throw *before* the resolution (argument
    # validation) cannot be mistaken for an MG failure of this solve.
    LAST_SEMSTOKES_AUTO_CHOICE[] = _SEMSTOKES_AUTO_CHOICE_NONE
    res = try
        _solve_semquad_stokes_var_core(ss, qt, eta_fun, fx, fz, uDx, uDz; core...)
    catch err
        err isa InterruptException && rethrow()
        ch = LAST_SEMSTOKES_AUTO_CHOICE[]
        ch.choice === :mg || rethrow()
        return _solve_semquad_stokes_var_fallback(
            ss, qt, eta_fun, fx, fz, uDx, uDz, core, ch,
            "the MG solve threw $(typeof(err)): $(sprint(showerror, err))")
    end
    get(core, :assemble_only, false) !== false && return res
    ch = LAST_SEMSTOKES_AUTO_CHOICE[]
    if ch.choice === :mg && !(LAST_SEMSTOKES_RESIDUAL[] <= ftol)  # also NaN
        return _solve_semquad_stokes_var_fallback(
            ss, qt, eta_fun, fx, fz, uDx, uDz, core, ch,
            @sprintf("true residual %.3e > auto_fallback_tol %.1e (saddle_dofs=%d, contrast=%.4g)", LAST_SEMSTOKES_RESIDUAL[],
                     ftol, ch.saddle_dofs, ch.contrast))
    end
    return res
end

# Redo an unacceptable `linear=:auto` MG solve with the explicit `:direct` path
# (same arguments, same state, so the answer is the direct answer) and count it.
function _solve_semquad_stokes_var_fallback(ss::SemStokes, qt::Quadtree,
                                            eta_fun::Function, fx::Function,
                                            fz::Function, uDx::Function,
                                            uDz::Function, core, ch, why::String)
    prev = LAST_AUTO_FALLBACK[]
    LAST_AUTO_FALLBACK[] = (count=prev.count + 1, last_reason=why)
    @warn "linear=:auto resolved to :mg but that solve was not acceptable; " *
          "redoing it with linear=:direct ($why).  Explicit linear=:mg is not " *
          "affected; LAST_AUTO_FALLBACK[] counts these solves." maxlog=1
    direct = (; (k => v for (k, v) in pairs(core) if k !== :linear)...,
                 linear=:direct)
    res = _solve_semquad_stokes_var_core(ss, qt, eta_fun, fx, fz, uDx, uDz;
                                         direct...)
    # Report the selector's decision; the counter records which path answered.
    LAST_SEMSTOKES_AUTO_CHOICE[] = ch
    return res
end

function _solve_semquad_stokes_var_core(ss::SemStokes, qt::Quadtree,
                                  eta_fun::Function,
                                  fx::Function, fz::Function,
                                  uDx::Function, uDz::Function;
                                  mass_rho::Union{Nothing,Function}=nothing,
                                  mass_form::Symbol=:anelastic,
                                  mass_drho::Union{Nothing,Function}=nothing,
                                  compress_block::Union{Nothing,AbstractMatrix}=nothing,
                                  deviatoric::Bool=false,
                                  geo_u::Union{Nothing,Vector{Float64}}=nothing,
                                  geo_coef::Union{Nothing,Function}=nothing,
                                  geo_spd_safety::Float64=0.0,
                                  dp_u::Union{Nothing,Vector{Float64}}=nothing,
                                  dp_pfield::Union{Nothing,Vector{Float64}}=nothing,
                                  dp_coef::Union{Nothing,Function}=nothing,
                                  bc::Symbol=:rigid, linear::Symbol=:direct,
                                  stokes_accept_tol::Float64=Inf,
                                  stokes_lost_tol::Float64=SEMSTOKES_LOST_TOL,
                                  stokes_scaling::Symbol=:none,
                                  schur_maxiter::Int=200, schur_precond::Symbol=:mass,
                                  mg_cycles::Int=1, mg_tol::Float64=1.0e-10,
                                  mg_pressure_scale::Float64=0.0,
                                  mg_pressure_weight::Symbol=:inv_eta,
                                  mg_pressure_mode::Symbol=:auto,
                                  mg_pressure_precond::Symbol=:auto,
                                  mg_pressure_cache::Bool=true,
                                  mg_pressure_eta_tol::Float64=SEMQUAD_PRESSURE_DEFAULT_ETA_TOL,
                                  mg_pressure_bfbt_d::Symbol=:auto,
                                  mg_pressure_bfbt_s2::Symbol=:free,
                                  mg_pressure_bfbt_s1::Symbol=:chol,
                                  mg_pressure_bfbt_s1_solve::Symbol=:chol,
                                  mg_pressure_bfbt_s1_tol::Float64=1.0e-2,
                                  mg_pressure_bfbt_s1_maxit::Int=50,
                                  mg_pressure_bfbt_s1_minwork::Int=0,
                                  mg_pressure_bfbt_s1_chunk::Int=0,
                                  mg_smoother::Symbol=:auto,
                                  mg_coarse::Symbol=:auto,
                                  mg_coarsen_layers::Int=1,
                                  mg_coarsen_factor::Union{Nothing,Float64}=nothing,
                                  mg_coarse_direct_max::Int=0,
                                  mg_smoothing::Symbol=:auto,
                                  mg_pre::Union{Integer,AbstractVector{<:Integer}}=2,
                                  mg_post::Union{Integer,AbstractVector{<:Integer}}=2,
                                  mg_omega::Union{Nothing,Float64}=nothing,
                                  mg_operator::Symbol=:matrixfree,
                                  mg_vel_solver::Symbol=:vcycle,
                                  mg_vel_tol::Float64=1.0e-1,
                                  mg_vel_maxit::Int=10,
                                  mg_restart::Int=100, mg_maxit::Int=500,
                                  mg_threads::Int=1,
                                  mg_diag::Symbol=:constrained,
                                  assemble_threads::Int=1,
                                  return_system::Bool=false,
                                  assemble_only::Union{Bool,Symbol}=false)
    bc in (:free, :rigid) || throw(ArgumentError("bc must be :free or :rigid"))
    stokes_scaling in SEMSTOKES_SCALINGS ||
        throw(ArgumentError("stokes_scaling must be :none or :equilibrated"))
    (geo_u === nothing) == (geo_coef === nothing) ||
        throw(ArgumentError("geo_u and geo_coef must be given together"))
    if geo_u !== nothing
        length(geo_u) == 2 * ss.su.N ||
            throw(ArgumentError("geo_u has $(length(geo_u)) entries, expected $(2 * ss.su.N)"))
        deviatoric &&
            throw(ArgumentError("the geometric strain-rate Jacobian is only " *
                                "implemented for deviatoric=false"))
    end
    # Pressure-derivative (Drucker-Prager) Jacobian block `Kp`: the momentum
    # residual is `K(eta(T, edot(u), p)) u - B'p - f`, so its pressure
    # derivative is
    #     (Kp)_{i,a} = int 2 (deta/dp) eps(u):eps(v_i) phi_a
    # in the velocity rows and nothing in the constraint rows.  The resulting
    # saddle system is non-symmetric (velocity rows <-> pressure columns), which
    # the direct LU handles.  `dp_u` is the velocity whose strain rate enters
    # `deta/dp`, `dp_pfield` the nodal pressure field, `dp_coef(x,y,edot,p)` the
    # derivative itself (e.g. `semquad_deta_dp` wrapped with the temperature).
    if dp_u !== nothing || dp_pfield !== nothing || dp_coef !== nothing
        (dp_u !== nothing && dp_pfield !== nothing && dp_coef !== nothing) ||
            throw(ArgumentError("dp_u, dp_pfield and dp_coef must be given " *
                                "together"))
        length(dp_u) == 2 * ss.su.N ||
            throw(ArgumentError("dp_u has $(length(dp_u)) entries, expected $(2 * ss.su.N)"))
        length(dp_pfield) == ss.sp.N ||
            throw(ArgumentError("dp_pfield has $(length(dp_pfield)) entries, " *
                                "expected $(ss.sp.N)"))
        deviatoric &&
            throw(ArgumentError("the pressure-derivative Jacobian block is " *
                                "only implemented for deviatoric=false"))
    end
    linear in (:direct, :schur_cg, :mg, :lagged, :auto) ||
        throw(ArgumentError("linear must be :direct, :schur_cg, :mg, :lagged " *
                            "or :auto"))
    (assemble_only === false || assemble_only === true || assemble_only === :mg) ||
        throw(ArgumentError("assemble_only must be false (solve), true (assemble " *
                            "everything) or :mg (the reduced matrix-free MG " *
                            "geometry: no velocity stiffness K, no Kred)"))
    mg_cycles >= 1 || throw(ArgumentError("mg_cycles must be >= 1"))
    mg_tol > 0.0 || throw(ArgumentError("mg_tol must be > 0"))
    mg_pressure_scale >= 0.0 ||
        throw(ArgumentError("mg_pressure_scale must be >= 0 (0 = auto)"))
    mg_pressure_weight in (:inv_eta, :eta, :none) ||
        throw(ArgumentError("mg_pressure_weight must be :inv_eta, :eta or :none"))
    mg_pressure_mode in (:auto, :mass, :stiffness, :lsc, :bfbt) ||
        throw(ArgumentError("mg_pressure_mode must be :auto, :mass, :stiffness, :lsc or :bfbt"))
    mg_pressure_precond in (:auto, :bfbt, :mass, :scaled_mass, :pcd, :lsc) ||
        throw(ArgumentError("mg_pressure_precond must be :auto, :bfbt, :mass, " *
                            ":scaled_mass, :pcd or :lsc"))
    mg_pressure_eta_tol > 0.0 ||
        throw(ArgumentError("mg_pressure_eta_tol must be > 0 (Inf = never " *
                            "rebuild on eta drift; a tree change still rebuilds)"))
    mg_pressure_bfbt_d in (:auto, :leaf, :exact, :matrix) ||
        throw(ArgumentError("mg_pressure_bfbt_d must be :auto, :leaf, :exact " *
                            "or :matrix"))
    mg_pressure_bfbt_s2 in (:free, :matrix) ||
        throw(ArgumentError("mg_pressure_bfbt_s2 must be :free or :matrix"))
    mg_pressure_bfbt_s1 in (:chol, :mass) ||
        throw(ArgumentError("mg_pressure_bfbt_s1 must be :chol or :mass"))
    mg_pressure_bfbt_s1_solve in (:chol, :level, :cg_diag, :cg_ichol) ||
        throw(ArgumentError("mg_pressure_bfbt_s1_solve must be :chol, :level, " *
                            ":cg_diag or :cg_ichol"))
    mg_pressure_bfbt_s1_tol > 0.0 ||
        throw(ArgumentError("mg_pressure_bfbt_s1_tol must be > 0"))
    mg_pressure_bfbt_s1_maxit >= 1 ||
        throw(ArgumentError("mg_pressure_bfbt_s1_maxit must be >= 1"))
    mg_smoother in (:auto, :jacobi, :sgs) ||
        throw(ArgumentError("mg_smoother must be :auto, :jacobi or :sgs"))
    mg_coarse in (:auto, :rediscretize, :galerkin) ||
        throw(ArgumentError("mg_coarse must be :auto, :rediscretize or :galerkin"))
    mg_coarsen_layers >= 1 ||
        throw(ArgumentError("mg_coarsen_layers must be >= 1"))
    (mg_coarsen_factor === nothing ||
     (mg_coarsen_factor >= 1.0 && isfinite(mg_coarsen_factor))) ||
        throw(ArgumentError("mg_coarsen_factor must be >= 1 (or `nothing` = " *
                            "the matrix-free production default 1.5)"))
    mg_coarse_direct_max >= 0 ||
        throw(ArgumentError("mg_coarse_direct_max must be >= 0"))
    mg_smoothing in (:auto, :uniform, :taper) ||
        throw(ArgumentError("mg_smoothing must be :auto, :uniform or :taper"))
    _mg_sw_ok(v) = v isa Integer ? v >= 0 : all(x -> x >= 0, v)
    _mg_sw_ok(mg_pre) || throw(ArgumentError("mg_pre must be >= 0"))
    _mg_sw_ok(mg_post) || throw(ArgumentError("mg_post must be >= 0"))
    (mg_omega === nothing || 0.0 < mg_omega < 1.0) ||
        throw(ArgumentError("mg_omega must be in (0,1) or `nothing`"))
    mg_operator in (:sparse, :matrixfree) ||
        throw(ArgumentError("mg_operator must be :sparse or :matrixfree"))
    # Production default of the h-coarsening step.  `coarsen_factor=2.0` was
    # measured on the F tree (1.5x end to end, docs/dev/MG.md) but on
    # P1 it collapses the hierarchy from 8 to 6 levels and costs 177 outer
    # iterations instead of 61 (8.03 s against 5.45 s, measured at 1e-6).  The
    # shipped default is therefore 1.5: it still removes F's 1.7 %-coarsening
    # near-duplicate level (F: 8 -> 6 levels) but keeps P1's 40 776-dof level
    # (P1: 8 -> 7 levels), and wins on both states (F 5.42 -> 3.36 s,
    # P1 5.45 -> 5.12 s at one thread).  `mg_coarsen_factor=1.0` restores the
    # historical rule and `2.0` remains selectable; uniform/blob trees are a
    # bitwise no-op at 1.5 (their steps already coarsen by >=2x).
    coarsen_factor = mg_coarsen_factor === nothing ?
                     (mg_operator === :matrixfree ? 1.5 : 1.0) :
                     mg_coarsen_factor
    mg_vel_solver in (:cg, :vcycle) ||
        throw(ArgumentError("mg_vel_solver must be :cg or :vcycle"))
    mg_vel_tol > 0.0 || throw(ArgumentError("mg_vel_tol must be > 0"))
    mg_vel_maxit >= 1 || throw(ArgumentError("mg_vel_maxit must be >= 1"))
    mg_restart >= 5 || throw(ArgumentError("mg_restart must be >= 5"))
    mg_maxit >= 1 || throw(ArgumentError("mg_maxit must be >= 1"))
    mg_threads >= 1 || throw(ArgumentError("mg_threads must be >= 1"))
    mg_diag in (:constrained, :lumped) ||
        throw(ArgumentError("mg_diag must be :constrained or :lumped"))
    mass_form in (:anelastic, :rho_div, :boussinesq) ||
        throw(ArgumentError("mass_form must be :anelastic, :rho_div or :boussinesq"))
    schur_precond in (:none, :mass) ||
        throw(ArgumentError("schur_precond must be :none or :mass"))
    assemble_threads >= 1 ||
        throw(ArgumentError("assemble_threads must be >= 1"))
    # S6: resolve `linear=:auto` **before** the assembly, so an MG decision
    # never pays for or stores the sparse velocity operator.  The decision uses
    # only O(1) node counts and the fixed 33x33-grid eta sample (measured cost
    # in docs/S3_DESIGN.md §20); an explicit `:direct`/`:mg` call skips this
    # block entirely and runs exactly as before (bitwise for `:direct`).
    if linear === :auto
        auto_choice, auto_threads = _semquad_resolve_auto(ss, eta_fun;
                                                          schur_precond=schur_precond)
        linear = auto_choice
        if linear === :mg
            # The selector is calibrated on the matrix-free operator, and
            # `mg_threads=min(8, Threads.nthreads())` (S5-2 measured the gain).
            mg_operator = :matrixfree
            mg_threads = auto_threads
        end
    else
        # An `assemble_only` call never solves, so it must not overwrite the
        # `:auto` decision of the last real solve: the MG hierarchy builder and
        # the DWR/Newton probes assemble with `linear=:direct, assemble_only=true`
        # *inside* an auto MG solve, and recording `:none` there would hide the
        # selector's decision (it broke the auto diagnostics until S7).
        assemble_only !== false ||
            (LAST_SEMSTOKES_AUTO_CHOICE[] =
                 (choice=:none,
                  reason=get(_SEMSTOKES_AUTO_OFF_REASONS, linear,
                             "linear not :auto (auto not requested)"),
                  saddle_dofs=0, contrast=1.0, samples=0, sample_ms=0.0,
                  mg_threads=0))
    end
    # BFBT weight: the exact restricted diagonal
    # `diag(Cu' M_u(sqrt(eta)) Cu)` is the shipped default on the matrix-free
    # path (docs/dev/MG.md).  It is a zero-per-apply-cost change
    # that removes 10 outer iterations on `F` (81 -> 71) and leaves `P1` at 65;
    # its leaf-local construction is cheaper than the historical lumped `:leaf`
    # diagonal at 8 threads and the same at 1 (both are dominated by the
    # leaf/quadrature-point eta evaluations).  `:leaf` is kept as the A/B and
    # `:matrix` as the sparse reference for G226; the sparse-operator MG path
    # keeps `:leaf` because it is not the measured production path.  Resolved
    # after the `linear=:auto` block, which may switch `mg_operator`.
    bfbt_d = mg_pressure_bfbt_d === :auto ?
             (mg_operator === :matrixfree ? :exact : :leaf) :
             mg_pressure_bfbt_d
    # `assemble_only=:mg` is the *reduced* assembly of the matrix-free MG
    # geometry: the caller (the hierarchy builder) only reads `Cu`, so the
    # velocity stiffness `K`, the COO triplets behind it and `Kred` are not
    # formed at all.  It is defined only for the matrix-free MG path (the sparse
    # operator path and the direct path need `K`/`Kred`); it is opt-in and
    # returns the same `Cu`/`Cp`/`B`/`fv` bit for bit.
    kfree = assemble_only === :mg
    kfree && return_system &&
        throw(ArgumentError("assemble_only=:mg does not form K/Kred, so it is " *
                            "incompatible with return_system=true"))
    kfree && !(linear === :mg && mg_operator === :matrixfree) &&
        throw(ArgumentError("assemble_only=:mg is the reduced matrix-free MG " *
                            "assembly; it requires linear=:mg with " *
                            "mg_operator=:matrixfree (got linear=$linear, " *
                            "mg_operator=$mg_operator)"))
    LAST_SEMSTOKES_VEL_CG_ITERS[] = 0
    p = ss.p; q = ss.q
    su = ss.su; sp = ss.sp
    nu = su.N; np = sp.N
    Nu2 = 2 * nu
    fd = make_fedata(p, q)
    n = p + 1
    nv = n * n
    npq = (q + 1)^2
    # S3 stage C: `linear=:mg, mg_operator=:matrixfree` never needs the sparse
    # velocity operator.  The right-hand side and the outer FGMRES operator are
    # built from the leaf-local apply (`semquad_apply_K!` / `_apply_Kred!`), so
    # the `2n^2 x 2n^2` dense leaf blocks and their `4 nv^2` triplets are never
    # formed.  Everything that still needs `K` (return_system, the Newton
    # geometric/pressure-derivative blocks) keeps it, and so does a plain
    # `assemble_only=true`; only the reduced `assemble_only=:mg` geometry call
    # joins the solve path in the `K`-free branch.  On that branch the leaf
    # assembly also skips the viscosity closure (its `etaq` only enters `K` and
    # the two Jacobian blocks), which is 0.15 s of the F setup at one thread --
    # see `SemQuadAsmCfg.mf_path`.
    mf_path = linear === :mg && mg_operator === :matrixfree &&
              (assemble_only === false || kfree) &&
              !return_system && geo_u === nothing && dp_u === nothing
    pleaves = Set(sp.leaves)
    # Deviatoric stress 2 eta (eps - (1/3) div(u) I): the volumetric part of
    # 2 eta eps:eps(v) loses 2/3 eta div(u) div(v).  ASPECT's ALA always uses
    # this form (manual eq. 20); without it a compressible (div u = Di u_z)
    # flow is too stiff.  `deviatoric=false` keeps the historical operator.
    a2 = deviatoric ? 4.0 / 3.0 : 2.0
    # S4 stage 1: leaf assembly.  `assemble_threads=1` (default) is the
    # historical serial path, bit for bit; >1 cuts the leaves into fixed
    # contiguous blocks, assembles them in parallel and merges the block COO
    # buffers in block order.  The threaded result is deterministic run to run
    # but is *not* bitwise equal to the serial one (different summation order),
    # which is why the default stays 1.
    asm_cfg = SemQuadAsmCfg(a2, deviatoric, mf_path, geo_u, geo_coef,
                            geo_spd_safety, dp_u, dp_pfield, dp_coef)
    asm = _semquad_assemble_blocks(
        _semquad_asm_thread_count(assemble_threads), asm_cfg, su, sp,
        eta_fun, fx, fz, fd, pleaves)
    IK = asm.IK; JK = asm.JK; VK = asm.VK
    IB = asm.IB; JB = asm.JB; VB = asm.VB
    IG = asm.IG; JG = asm.JG; VG = asm.VG
    IP = asm.IP; JP = asm.JP; VP = asm.VP
    fv = asm.fv

    if mf_path
        K = nothing
    else
        SEMSTOKES_K_ASSEMBLIES[] += 1
        K = sparse(IK, JK, VK, Nu2, Nu2)
    end
    Kgeo = geo_u === nothing ? nothing : sparse(IG, JG, VG, Nu2, Nu2)
    Kp = dp_u === nothing ? nothing : sparse(IP, JP, VP, Nu2, np)
    Bref = sparse(IB, JB, VB, np, Nu2)
    # Mass constraint with a stratified reference density.  ASPECT's ALA mass
    # equation is div(rho u) = 0, whose Galerkin form is
    #     int q (rho div u + u_z drho/dy) dV = 0,
    # **not** `int q rho div u = 0`: the latter only enforces div u = 0 and, with
    # upwellings, breaks the exact cancellation between the advection of the
    # reference state and the adiabatic heating term (King 2010 then cools and
    # its energy budget does not close).  The momentum pressure term stays the
    # unweighted -int p div v, so Bm = Bref while Bc carries the density.
    # `mass_form=:rho_div` keeps the historical single rho-weighted matrix for
    # both blocks so the old behaviour stays regression-testable.
    # `mass_form=:boussinesq` is the **TALA** continuity equation `div u = 0`:
    # the reference density enters the buoyancy only, not the mass conservation
    # (ASPECT's `truncated anelastic liquid approximation`).  It is the
    # apples-to-apples TALA counterpart of the ALA form for the same reference
    # state, which is what docs/RHEOLOGY.md §10.3 measures; it is off unless
    # requested, so the default path is unchanged.
    Bm = Bref
    Bc = Bref
    if mass_rho !== nothing && mass_form !== :boussinesq
        if mass_form === :anelastic
            # derivative of the density profile at the quadrature points
            drho = mass_drho === nothing ? (y -> begin
                hd = 1.0e-6
                yp = min(y + hd, 1.0); ym = max(y - hd, 0.0)
                (mass_rho(yp) - mass_rho(ym)) / (yp - ym)
            end) : mass_drho
            Bc = semquad_stokes_B(ss, qt; rho_fun=mass_rho, drho_fun=drho)
        else
            Bc = semquad_stokes_B(ss, qt; rho_fun=mass_rho)
            Bm = Bc
        end
        linear === :schur_cg && mass_form === :anelastic &&
            throw(ArgumentError("mass_form=:anelastic makes the saddle system " *
                                "nonsymmetric (Bm != Bc'); use linear=:direct"))
    end

    # affine Dirichlet data on the domain boundary
    fixed = Dict{Int,Float64}()
    for lf in su.leaves
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = su.elem[lf]
        for dir in (:xminus, :xplus, :yminus, :yplus)
            has_no_face_neighbor(qt, lf, dir) || continue
            if dir === :xminus || dir === :xplus
                iF = dir === :xminus ? 1 : n
                xf = dir === :xplus ? x0 + hx : x0
                for j in 1:n
                    yv = y0 + 0.5 * (su.xi[j] + 1.0) * hy
                    g = uids[(j - 1) * n + iF]
                    fixed[g] = uDx(xf, yv)
                    bc === :rigid && (fixed[nu + g] = uDz(xf, yv))
                end
            else
                jF = dir === :yminus ? 1 : n
                yf = dir === :yplus ? y0 + hy : y0
                for i in 1:n
                    xv = x0 + 0.5 * (su.xi[i] + 1.0) * hx
                    g = uids[(jF - 1) * n + i]
                    fixed[nu + g] = uDz(xv, yf)
                    bc === :rigid && (fixed[g] = uDx(xv, yf))
                end
            end
        end
    end
    cons_u = semquad_constraints(su, qt)
    cons2 = Dict{Int,Tuple{Vector{Int},Vector{Float64}}}()
    for (c, (ms, ws)) in cons_u
        cons2[c] = (ms, ws)
        cons2[nu + c] = (nu .+ ms, ws)
    end
    u0, Cu, _ = _affine_constraint(Nu2, cons2, fixed)
    cons_p = semquad_constraints(sp, qt)
    pf = Dict{Int,Float64}()
    pfree0 = setdiff(1:np, collect(keys(cons_p)))
    isempty(pfree0) || (pf[first(pfree0)] = 0.0)
    p0, Cp, _ = _affine_constraint(np, cons_p, pf)

    CpB = Cp' * Bc * Cu
    # S3 stage C: on the fully matrix-free MG path there is no sparse `K` and no
    # `Kred = Cu' K Cu`.  `fr = Cu' (fv - K u0 + Bm' p0)` needs `K u0` on the
    # *full* velocity vector, which the leaf-local apply gives directly, and the
    # outer operator becomes `Cu' K (Cu v)` (`_semquad_apply_Kred!`).
    mfop = nothing
    mfapplyK = nothing
    mfpar = nothing
    if mf_path
        mfop = semquad_kop(ss, qt; deviatoric=deviatoric)
        # sampled once, then laid out for the lane kernel (docs/dev/MG.md
        # §15): the leaf-major field plus its block-major lane table
        mfetav = semquad_eta_lanes(mfop, semquad_eta_field(mfop, eta_fun))
        mfwork = SemQuadKWork(mfop.n, mfop.nq, Nu2)
        Ku0 = zeros(Float64, Nu2)
        _semquad_apply_K_raw!(Ku0, mfop, mfetav, u0, mfwork)
        fr = Cu' * (fv - Ku0 + Bm' * p0)
        # S5-2: the outer operator (FGMRES residual and the matrix-free BFBT
        # `S2`) uses the same threaded leaf blocks as the V-cycle.
        nt_outer = _semquad_mg_thread_count(mg_threads)
        if nt_outer > 1
            par_outer = SemQuadKPar(mfop, nt_outer)
            mfpar = par_outer
            mfapplyK = (out, v) -> _semquad_apply_Kred_threaded!(out, mfop, Cu,
                                                                 mfetav, v,
                                                                 mfwork,
                                                                 par_outer)
        else
            mfapplyK = (out, v) -> _semquad_apply_Kred!(out, mfop, Cu, mfetav,
                                                        v, mfwork)
        end
    else
        fr = Cu' * (fv - K * u0 + Bm' * p0)
    end
    Kred = mf_path ? nothing :
           Kgeo === nothing ? Cu' * K * Cu : Cu' * (K + Kgeo) * Cu
    Kp_red = Kp === nothing ? nothing : Cu' * Kp * Cp
    gr = Cp' * (Bc * u0)
    nf = size(Cu, 2); nfp = size(Cp, 2)
    Bs = -CpB
    Bt = Bm === Bc ? sparse(transpose(Bs)) : -(Cu' * Bm' * Cp)
    # Compressibility: in route C's (dynamic) pressure gauge the ALA density
    # perturbation -rho_ad Di gamma (p - p_ad) contributes +rho_ad Di gamma p
    # to the *upward* momentum.  The sign is not free: with ASPECT's own
    # temperature field this sign reproduces ASPECT's velocity to six digits
    # (Vrms 24.7657 vs 24.7657, max|u| 46.858 vs 46.858), while the opposite
    # sign gives Vrms 26.94 (the historical 8% King error).  See
    # ROUTE_C_STATUS.md for the operator-agreement test.
    if compress_block !== nothing
        Bmm = sparse(transpose(Bm)) - compress_block
        Bt = -(Cu' * Bmm * Cp)
    end
    # Pressure-block preconditioner for the saddle solve.  It is built after
    # Kred/Bs/Bt (and after the compressibility update of Bt) because the LSC
    # variant needs diag(Kred) and the final Bt.
    Fp = nothing
    if linear === :schur_cg && schur_precond === :mass
        Mfull = assemble_semquad_pressure_mass(sp)
        Mred = Cp' * Mfull * Cp
        Fp = cholesky(Symmetric(Mred))
        issuccess(Fp) ||
            throw(ArgumentError("pressure mass preconditioner is not SPD"))
    elseif linear === :mg && !kfree
        # Pressure-block approximate inverse (mass / LSC / PCD / three-factor
        # w-BFBT); shared with the MG-accelerated Newton solver, see
        # semquad_mg.jl.  `mg_pressure_precond` is the public selector
        # (:bfbt | :mass | :scaled_mass | :pcd | :lsc); its default `:auto`
        # keeps the historical `mg_pressure_mode` resolution, which picks the
        # three-factor w-BFBT above 100x viscosity contrast -- the measured
        # winner on the blob tree (docs/S3_DESIGN.md §13).  Not built for the
        # reduced `assemble_only=:mg` geometry call, which never uses it.
        pmode_arg = mg_pressure_mode
        pweight_arg = mg_pressure_weight
        if mg_pressure_precond !== :auto
            pmode_arg, pweight_arg =
                _semquad_pressure_precond_mode(mg_pressure_precond,
                                               mg_pressure_weight)
        end
        # S3 stage A: cached by (tree fingerprint, nf, nfp) and reused while the
        # sampled viscosity moves by at most `mg_pressure_eta_tol` (the outer
        # FGMRES is flexible, so a stale Shat^{-1} is a valid preconditioner).
        Fp, pscale = semquad_mg_pressure_precond(ss, Cu, Cp, Bs, Bt, Kred,
                                                 eta_fun;
                                                 mode=pmode_arg,
                                                 scale=mg_pressure_scale,
                                                 weight=pweight_arg,
                                                 key=_saddle_key(qt, p, bc, nf, nfp),
                                                 eta_tol=mg_pressure_eta_tol,
                                                 cache=mg_pressure_cache,
                                                 bfbt_d=bfbt_d,
                                                 bfbt_s2=mg_pressure_bfbt_s2,
                                                 bfbt_s1=mg_pressure_bfbt_s1,
                                                 bfbt_s1_solve=mg_pressure_bfbt_s1_solve,
                                                 bfbt_s1_tol=mg_pressure_bfbt_s1_tol,
                                                 bfbt_s1_maxit=mg_pressure_bfbt_s1_maxit,
                                                 bfbt_s1_minwork=mg_pressure_bfbt_s1_minwork,
                                                 bfbt_s1_chunk=mg_pressure_bfbt_s1_chunk,
                                                 nthreads=_semquad_mg_thread_count(mg_threads),
                                                 applyK=mfapplyK)
    end
    if assemble_only !== false
        return (K=K, B=Bref, Bc=Bc, Bm=Bm, Cu=Cu, Cp=Cp, u0=u0, p0=p0, fv=fv,
                fr=fr, gr=gr, Kred=Kred, Bs=Bs, Bt=Bt, Kgeo=Kgeo, Kp=Kp,
                Kp_red=Kp_red)
    end
    nit = 0
    # Every solve resets the direct-path shift record, so it describes this
    # solve only (a `:mg`/`:schur_cg`/`:lagged` solve reports "no shift").
    LAST_SEMSTOKES_SHIFT[] = 0.0
    LAST_SEMSTOKES_REFINE_ITERS[] = 0
    LAST_SEMSTOKES_LADDER[] = nothing
    LAST_SEMSTOKES_LADDER_PHASE[] = nothing
    if linear === :schur_cg
        # Saddle [Kred (-CpB'); (-CpB) 0] with Bs = -CpB, Bt = Bs':
        #   Kred u + Bt p = fr,   Bs u = gr,
        # solved by a matrix-free Schur CG on S = Bs Kred^{-1} Bt.
        # A constrained/hanging velocity mode can leave a zero eigenvalue in
        # Kred after AMR; regularize with a 1e-12 relative diagonal shift so the
        # sparse LU is defined.  The shift is below the G44 exactness tolerance.
        dmax = maximum(abs, diag(Kred))
        Kred = Kred + (1.0e-12 * max(dmax, 1.0)) * I
        # Row/column equilibration: wall-refined AMR trees have h_min/h_max
        # ~1e-4 and Kred condition numbers ~1e12; the scaled LU is much better
        # conditioned and removes the artificial large velocity from the shift.
        d = sqrt.(max.(diag(Kred), 1.0e-14 * dmax))
        Dinv = spdiagm(0 => 1.0 ./ d)
        Kscaled = Dinv * Kred * Dinv
        Fv = lu(Kscaled)
        ufree = Dinv * (Fv \ (Dinv * fr))
        r = Bs * ufree - gr
        pfree = zeros(Float64, nfp)
        z = Fp === nothing ? copy(r) : Fp \ r
        pp = copy(z)
        rho = dot(r, z)
        bnorm = max(dot(r, r), 1.0e-300)
        for it in 1:schur_maxiter
            nit = it
            Ap = Bs * (Dinv * (Fv \ (Dinv * (Bt * pp))))
            den = dot(pp, Ap)
            den == 0.0 && break
            alpha = rho / den
            pfree .+= alpha .* pp
            r .-= alpha .* Ap
            rs = dot(r, r)
            rs <= 1.0e-18 * bnorm && break
            z = Fp === nothing ? copy(r) : Fp \ r
            rho_new = dot(r, z)
            pp .= z .+ (rho_new / rho) .* pp
            rho = rho_new
        end
        ufree = Dinv * (Fv \ (Dinv * (fr - Bt * pfree)))
    elseif linear === :mg
        # Scalable path: geometric MG for the velocity block plus a pressure
        # Schur block, used as a block-triangular right preconditioner for
        # flexible GMRES on the reduced saddle (S3/MF-4).  The velocity block
        # is solved by the fixed `mg_cycles` V-cycles
        # (`mg_vel_solver=:vcycle`, the default since MF-6, 33% faster than the
        # inner-CG variant on the 1024-leaf blob tree); `mg_vel_solver=:cg`
        # keeps the MF-4 scheme, CG on the SPD `Kred` preconditioned by the
        # hierarchy (the deal.II/ASPECT FGMRES+CGMG form).  The criterion is
        # the *true* reduced-saddle residual (see `fgmres_restarted`), never
        # the preconditioned one.
        uniform = length(unique(lf[1] for lf in qt.leaves)) == 1
        smoother = mg_smoother === :auto ? :jacobi : mg_smoother
        coarse = mg_coarse === :auto ? (uniform ? :rediscretize : :galerkin) :
                 mg_coarse
        # S3/MF-2: the matrix-free hierarchy always coarsens by
        # rediscretisation (see make_semquad_mg).
        mg_operator === :matrixfree && (coarse = :rediscretize)
        mg = make_semquad_mg(ss, qt, p, bc, deviatoric, eta_fun, Kred;
                             cycles=mg_cycles, smoother=smoother,
                             coarse_operator=coarse, operator=mg_operator,
                             coarsen_layers=mg_coarsen_layers,
                             coarsen_factor=coarsen_factor,
                             coarse_direct_max=mg_coarse_direct_max,
                             smoothing=mg_smoothing, pre=mg_pre,
                             post=mg_post, omega=mg_omega,
                             mg_threads=mg_threads, mg_diag=mg_diag,
                             fine_Cu=Cu,
                             # On the matrix-free path the solve already built
                             # the finest level's leaf-local operator and its
                             # eta field; reuse both instead of resampling the
                             # viscosity a third time (0.17 s at one thread).
                             fine_mfop=mf_path ? mfop : nothing,
                             fine_mfeta=mf_path ? mfetav : nothing)
        Fp === nothing &&
            throw(ArgumentError("linear=:mg requires the pressure mass " *
                                "preconditioner (schur_precond=:mass)"))
        nftot = nf + nfp
        sol, nit, relres, cgiters =
            semquad_mg_saddle_solve(Kred, Bs, Bt, Fp, pscale, vcat(fr, gr), mg;
                                    tol=mg_tol, maxit=mg_maxit,
                                    restart=mg_restart,
                                    vel_solver=mg_vel_solver,
                                    vel_tol=mg_vel_tol,
                                    vel_maxit=mg_vel_maxit,
                                    applyK=mfapplyK)
        LAST_SEMSTOKES_VEL_CG_ITERS[] = cgiters
        ufree = sol[1:nf]
        pfree = sol[(nf + 1):nftot]
        relres > 1.0e-5 &&
            @warn "MG-FGMRES true relres=$relres exceeds 1e-5 at level " *
                  "$(mg.geo.L)"
    elseif linear === :lagged
        S = [Kred Bt; Bs spzeros(nfp, nfp)]
        rhs = vcat(fr, gr)
        # S2: reuse the previous LU as a preconditioner and polish with iterative
        # refinement; refactor only on a cache miss / stall.  The saddle depends
        # on the viscosity field and the tree, not on dt, so between two AMR
        # cycles the stale factor is a good preconditioner.
        key = _saddle_key(qt, ss.su.p, bc, nf, nfp)
        sol, rn = _lagged_solve!(S, rhs, key; tol=1.0e-8, maxit=25)
        LAST_SEMSTOKES_REGULARIZED[] = false
        if rn > 1.0e-8
            dmax = maximum(abs, diag(S)) + 1.0
            Ssh = S + (1.0e-12 * dmax) * I
            sol, rn = _lagged_solve!(Ssh, rhs, key; tol=1.0e-8, maxit=6,
                                     force_refactor=true)
            LAST_SEMSTOKES_REGULARIZED[] = true
        end
        ufree = sol[1:nf]
        pfree = sol[nf+1:end]
    else
        S = [Kred Bt; Bs spzeros(nfp, nfp)]
        rhs = vcat(fr, gr)
        # Wall-refined trees with free-slip walls make the *unshifted* sparse LU
        # unstable: on the 2b base=3/4 wall trees the factorisation has a growth
        # factor ~1e15 (max |U_ii| ~ 1e13-1e16 against a 2.9e-5 smallest pivot)
        # and the backward residual is ~1e3, while the same system solved with a
        # 1e-12 relative diagonal shift -- exactly what `_semquad_newton_saddle`
        # and the :schur_cg path already use -- is exact (residual 3e-11) and
        # gives the same velocity to 1e-6.  The near-null mode is a pressure mode
        # of the level-6 wall strip (|Bs'q|/|q| ~ 5e-3, velocity part 2e-4).
        #
        # `_shift_ladder_solve` keeps that single retry as its first rung and
        # only escalates (1e-10, 1e-8) while the residual misses
        # `stokes_accept_tol`, the driver's Stokes gate; the last resort is a few
        # iterative-refinement steps with the same factorisation.  The default
        # `accept_tol = Inf` stops after the first rung, so the well-conditioned
        # systems (uniform meshes, L2-L4 2b, the blob) stay bit-identical.
        # docs/dev/CASE_REALISTIC.md §13: on the 8192-leaf realistic tree the 1e-12
        # rung alone leaves 1.3e-3 (1.3x the card's 1e-3) and the driver aborted
        # the long run at step 6130.  §13.10: when the unshifted solve is *lost*
        # (`stokes_lost_tol`, default 1.0) the ladder tries the tiny shifts
        # first, because the shift's bias is `shift*|x|` and the near-null
        # cluster of the level-5 wall strip sits ~170x below the 1e-12 rung
        # (measured: 4.7e7 unshifted -> 5.2e-7 at 1e-14 on the step-6600 state,
        # while the 1e-12 rung leaves 1.1e-4).
        rden0 = max(norm(fr), norm(gr), 1.0)
        rel = function (x)
            rv = norm(Kred * x[1:nf] + Bt * x[(nf + 1):(nf + nfp)] - fr) / rden0
            rd = norm(Bs * x[1:nf] - gr) / rden0
            return max(rv, rd)
        end
        if stokes_scaling === :equilibrated
            # docs/PRESSURE_EQUILIBRATION.md: solver-side symmetric equilibration
            # of the same saddle.  First the historical unscaled unshifted solve,
            # exactly the ladder's rule 1: a solve that is healthy there is
            # returned untouched, so healthy paths stay bitwise identical even
            # with the option on, and the scaling is only paid for on a solve
            # that would otherwise enter the shift ladder.
            x_un = S \ rhs
            r_un = rel(x_un)
            r_un = isfinite(r_un) ? r_un : Inf
            if !(r_un > 1.0e-8) && !(r_un > stokes_accept_tol)
                sol, sh, nref, rungs = x_un, 0.0, 0, nothing
                LAST_SEMSTOKES_LADDER_PHASE[] = :healthy
            else
                # The congruence S -> D*S*D, rhs -> D*rhs leaves the solution
                # unchanged (`x = D*y`); it changes only the arithmetic of the
                # factorisation.  In the scaled operator the wall-strip pressure
                # family is lifted far above the round-off floor (the measured
                # smallest eigenvalue is a momentum mode at 4.2e-11 relative,
                # not the unscaled 1.7e-14 pressure mode), so the *unshifted* LU
                # resolves it: 6.0e-10 (AMR 7641) and 3.6e-8 (uniform L5,
                # round-off limited by the null component) against 3.7e-7 /
                # 1.2e-6 for the unscaled tiny-shift stopgap.  The 1e-6 gate is
                # the stated accuracy goal; if the scaled solve misses it, the
                # historical unscaled ladder is the unchanged fallback.
                sol_scaled, _ = _semquad_stokes_equil_solve(S, Kred, Bt, Bs, rhs)
                r_sc = rel(sol_scaled)
                r_sc = isfinite(r_sc) ? r_sc : Inf
                if r_sc <= 1.0e-6 && r_sc <= stokes_accept_tol
                    sol, sh, nref = sol_scaled, 0.0, 0
                    rungs = Any[(0.0, r_un), (:equilibrated, r_sc)]
                    LAST_SEMSTOKES_LADDER_PHASE[] = :equilibrated
                    LAST_SEMSTOKES_SCALED[] += 1
                else
                    sol, _, sh, nref, rungs =
                        _shift_ladder_solve(S, rhs; accept_tol=stokes_accept_tol,
                                            lost_tol=stokes_lost_tol,
                                            bad_tol=1.0e-8, resid=rel)
                end
            end
        else
            sol, _, sh, nref, rungs =
                _shift_ladder_solve(S, rhs; accept_tol=stokes_accept_tol,
                                    lost_tol=stokes_lost_tol,
                                    bad_tol=1.0e-8, resid=rel)
        end
        LAST_SEMSTOKES_REGULARIZED[] = sh > 0.0
        LAST_SEMSTOKES_SHIFT[] = sh
        LAST_SEMSTOKES_REFINE_ITERS[] = nref
        LAST_SEMSTOKES_LADDER[] = rungs
        ufree = sol[1:nf]
        pfree = sol[nf+1:end]
    end
    # True reduced-saddle residual of the system that was actually solved
    # (Schur uses the regularized Kred).  This is what distinguishes a
    # converged solve from a singular/ill-conditioned direct solve.
    r1 = if mf_path
        Ku = zeros(Float64, nf)
        mfapplyK(Ku, ufree)
        Ku .+ Bt * pfree .- fr
    else
        Kred * ufree + Bt * pfree - fr
    end
    r2 = Bs * ufree - gr
    rden = max(norm(fr), norm(gr), 1.0)
    LAST_SEMSTOKES_RESIDUAL[] = max(norm(r1), norm(r2)) / rden
    LAST_SEMSTOKES_RESID_SPLIT[] = (vel=norm(r1) / rden, div=norm(r2) / rden)
    LAST_SEMSTOKES_SCHUR_ITERS[] = nit
    u = u0 + Cu * ufree
    pv = p0 + Cp * pfree
    if return_system
        return u, pv, fixed, cons2, cons_p, K, Bref, Cu, Cp, u0, p0, fv
    end
    return u, pv, fixed, cons2, cons_p
end

"""
    semquad_stokes_B(ss, qt; rho_fun=nothing)

Quadrature-assembled divergence matrix on the quadtree,

    B[pi, uj] = int q_pi * rho(y) * dphi_j/dx  dV,     B[pi, vzj] likewise,

with the velocity/pressure bases of `ss`.  `rho_fun=nothing` gives the constant
density form used by the reference-element assembly, which lets the two be
compared (G129).  Passing `y -> rho(y)` gives the density-weighted divergence
`int q rho div(u)`, which alone enforces `div u = 0`, **not** `div(rho u) = 0`.
The anelastic constraint additionally needs the `int q u_z drho/dy` term, which
`drho_fun` adds (G130):

    int q (rho div u + u_z drho/dy) dV = 0    <=>    div(rho u) = 0.
"""
function semquad_stokes_B(ss, qt::Quadtree; rho_fun::Union{Nothing,Function}=nothing,
                          drho_fun::Union{Nothing,Function}=nothing)
    su = ss.su
    sp = ss.sp
    p = su.p
    n = p + 1
    q = sp.p
    nq1 = q + 1
    fd = make_fedata(p, q)
    nu = su.N
    np = sp.N
    I = Int[]; J = Int[]; V = Float64[]
    sizehint!(I, np * 4 * n * n)
    sizehint!(J, np * 4 * n * n)
    sizehint!(V, np * 4 * n * n)
    for lf in su.leaves
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        x0 = lf[2] * hx; y0 = lf[3] * hy
        uids = su.elem[lf]
        pids = sp.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            rho = rho_fun === nothing ? 1.0 : rho_fun(y)
            for b in 1:nq1, a in 1:nq1
                pq = fd.phi_p[a, qx] * fd.phi_p[b, qy]
                pq == 0.0 && continue
                piq = pids[(b - 1) * nq1 + a]
                for j in 1:n, i in 1:n
                    uj = uids[(j - 1) * n + i]
                    dpx = fd.dphi_v[i, qx] * fd.phi_v[j, qy] * (2.0 / hx)
                    dpy = fd.phi_v[i, qx] * fd.dphi_v[j, qy] * (2.0 / hy)
                    ph = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                    c = w * rho * pq
                    push!(I, piq); push!(J, uj); push!(V, c * dpx)
                    push!(I, piq); push!(J, nu + uj); push!(V, c * dpy)
                    if drho_fun !== nothing
                        # div(rho u) = rho div u + u_z drho/dy: the second term
                        # turns the rho-weighted incompressibility into the
                        # anelastic continuity equation.
                        push!(I, piq); push!(J, nu + uj)
                        push!(V, w * pq * drho_fun(y) * ph)
                    end
                end
            end
        end
    end
    return sparse(I, J, V, np, 2 * nu)
end

"""
    semquad_stokes_Pcm(ss, qt; rho_fun=nothing, gamma=1.0, Di=1.0)

Pressure-velocity block of the TALA/ALA compressibility term.  In ASPECT the
pressure dependence enters the *density* (code.cc:93-94),

    rho = rho_ad(y) * (1 - alpha (T - Tad) + Di gamma (p - p_ad)),

so its contribution to the momentum equation is a buoyancy force along gravity,

    f_z = -rho_ad(y) Di gamma (p - p_ad),

and nothing in the horizontal direction.  This block is therefore

    Pcm[i, pj] = int phi_i(y) * rho(y) * Di * gamma * phi_pj dV

in the **vertical** rows only; with rho = 1 it is the pressure mass matrix (up to
the Di*gamma factor).  `components=:both` reproduces the historical behaviour
that applied the same column to both velocity components: that is a spurious
horizontal force proportional to pressure (it over-drove King 2010 by ~25x in
Vrms) and is kept only so the old wiring can be regression-tested.

The `p_ad` part of `(p - p_ad)` is a function of y alone, so it is a gradient
force that can be absorbed into the pressure; dropping it does not change u.

Sign convention: the block itself is assembled positive and the saddle assembly
enters it into the upward momentum with a **plus** sign
(`Bmm = B' - Pcm` in `solve_semquad_stokes_var`).  That is the sign that
reproduces ASPECT's operator: prescribing ASPECT's own p field with the force
`-rho_ad Di gamma p` gives ASPECT's velocity to six digits (Vrms 24.7681 vs
24.7657, max|u| 46.861 vs 46.858), while the opposite sign gives 27.88.
See ROUTE_C_STATUS.md for the operator-agreement test.
"""
function semquad_stokes_Pcm(ss, qt::Quadtree; rho_fun::Union{Nothing,Function}=nothing,
                            gamma::Float64=1.0, Di::Float64=1.0,
                            components::Symbol=:z)
    components in (:z, :both) ||
        throw(ArgumentError("components must be :z or :both"))
    su = ss.su
    sp = ss.sp
    p = su.p
    n = p + 1
    q = sp.p
    nq1 = q + 1
    fd = make_fedata(p, q)
    nu = su.N
    np = sp.N
    I = Int[]; J = Int[]; V = Float64[]
    for lf in su.leaves
        hx = leaf_hx(su, lf[1]); hy = leaf_hy(su, lf[1])
        y0 = lf[3] * hy
        uids = su.elem[lf]
        pids = sp.elem[lf]
        for qy in 1:fd.nq, qx in 1:fd.nq
            w = 0.25 * hx * hy * fd.gl_w[qx] * fd.gl_w[qy]
            y = y0 + 0.5 * (fd.gl_x[qy] + 1.0) * hy
            c = w * Di * gamma * (rho_fun === nothing ? 1.0 : rho_fun(y))
            for b in 1:nq1, a in 1:nq1
                pq = fd.phi_p[a, qx] * fd.phi_p[b, qy]
                pq == 0.0 && continue
                piq = pids[(b - 1) * nq1 + a]
                for j in 1:n, i in 1:n
                    uj = uids[(j - 1) * n + i]
                    ph = fd.phi_v[i, qx] * fd.phi_v[j, qy]
                    ph == 0.0 && continue
                    if components === :z
                        push!(I, nu + uj); push!(J, piq); push!(V, c * ph * pq)
                    else
                        push!(I, uj); push!(J, piq); push!(V, c * ph * pq)
                        push!(I, nu + uj); push!(J, piq); push!(V, c * ph * pq)
                    end
                end
            end
        end
    end
    return sparse(I, J, V, 2 * nu, np)
end
